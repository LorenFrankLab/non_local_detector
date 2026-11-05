"""The fitted state bins of a detector and the continuous transition between them.

A detector's discrete states each own a block of state bins: one bin per
place bin of its environment, or a single bin without a position (No-Spike,
and Local without a position kernel). ``state_layout`` assigns the bins,
``initial_conditions`` builds the initial distribution over them, and
``continuous_state_transitions`` the transition between them, as a dense
matrix or a structured operator. A fit records the layout and transition,
with the parameters they were built from, in a ``StateSpace``
(``detector.state_space_``), which decoding reads.

Saved detectors name ``StateSpace`` by this module's path.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from logging import getLogger
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from non_local_detector.continuous_state_transitions import (
    TransitionContext,
    _check_block_environments,
    _check_continuous_transition,
    _make_transition_block,
    _multi_bin_local_transition,
)
from non_local_detector.environment import Environment, find_environment_by_name
from non_local_detector.initial_conditions import _check_state_initial_conditions
from non_local_detector.models._input_validation import _validate_transition_arguments
from non_local_detector.observation_models import ObservationModel
from non_local_detector.transition_operators import (
    DEFAULT_MAX_DENSE_BYTES,
    _one_bin_block,
)
from non_local_detector.types import DiscreteTransitions, TransitionRepresentation

if TYPE_CHECKING:
    from non_local_detector.transition_operators import (
        BlockTransitionOperator,
        LazyDenseTransition,
    )

logger = getLogger(__name__)


@dataclass(frozen=True, eq=False)
class StateSpace:
    """Fit-time layout of a detector's state bins and its continuous transition.

    ``fit`` builds it from the detector's parameters and fitted environments,
    and decoding reads it instead of the parameters, so changing them with
    ``set_params`` after a fit does not change what the fit decodes. The
    parameters it records are from ``recorded_layout_parameters``.

    ``discrete_transition_type``, ``discrete_initial_conditions`` and
    ``environments`` are recorded only so a later change to those parameters
    can be detected (``changed_layout_parameters``). Decoding does not read
    them.

    Attributes
    ----------
    state_names : tuple of str, length n_states
        Names of the discrete states.
    observation_models : tuple of ObservationModel, length n_states
        Observation model of each discrete state.
    continuous_transition_types : tuple of tuples, shape (n_states, n_states)
        Continuous transition of each pair of discrete states, as configured.
    local_position_std : float or None
        Width of the Local states' position kernel; None for single-bin Local
        states.
    discrete_transition_type : DiscreteTransitions
        The discrete transition, as configured.
    discrete_initial_conditions : np.ndarray, shape (n_states,)
        Initial probability of each discrete state, as configured.
    environments : tuple of dict, length n_environments
        Each environment's configuration (``Environment.get_params``), which
        holds no fitted grid; the fitted environments are the detector's
        ``environments_``.
    state_ind : np.ndarray of int, shape (n_state_bins,)
        Discrete state of each state bin. Read-only, as the other arrays.
    bin_sizes : np.ndarray of int, shape (n_states,)
        Number of bins of each discrete state.
    is_track_interior : np.ndarray of bool, shape (n_state_bins,)
        Whether each state bin is on the track interior.
    continuous_transitions : np.ndarray or LazyDenseTransition, shape (n_state_bins, n_state_bins)
        Probability of moving between bins given the discrete transition: a
        dense matrix, or a budgeted lazy view of ``transition_operator``.
    transition_operator : BlockTransitionOperator or None
        The structured transition operator; None for a dense transition.
    transition_representation : {"dense", "structured", "auto"}
        How the transition was built; "dense" once it is a dense matrix.
    max_dense_transition_bytes : int
        Byte limit for dense reads of the structured transition.
    """

    state_names: tuple[str, ...]
    observation_models: tuple[ObservationModel, ...]
    continuous_transition_types: tuple[tuple[Any, ...], ...]
    local_position_std: float | None
    discrete_transition_type: DiscreteTransitions
    discrete_initial_conditions: np.ndarray
    environments: tuple[dict[str, Any], ...]
    state_ind: np.ndarray
    bin_sizes: np.ndarray
    is_track_interior: np.ndarray
    continuous_transitions: "np.ndarray | LazyDenseTransition"
    transition_operator: "BlockTransitionOperator | None"
    transition_representation: TransitionRepresentation
    max_dense_transition_bytes: int

    def __post_init__(self) -> None:
        """Make the arrays read-only: decoding reads them."""
        self._set_arrays_read_only()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a copy or a pickle, whose arrays come back writable."""
        self.__dict__.update(state)
        self._set_arrays_read_only()

    def _set_arrays_read_only(self) -> None:
        for array in (
            self.state_ind,
            self.bin_sizes,
            self.is_track_interior,
            self.continuous_transitions,
        ):
            if isinstance(array, np.ndarray):
                array.setflags(write=False)

    @property
    def n_states(self) -> int:
        """Number of discrete states."""
        return len(self.bin_sizes)

    @property
    def n_state_bins(self) -> int:
        """Number of state bins over every discrete state."""
        return len(self.state_ind)

    @property
    def interior_state_ind(self) -> np.ndarray:
        """Discrete state of each interior state bin, shape (n_interior_bins,)."""
        return cast("np.ndarray", self.state_ind[self.is_track_interior])

    def has_place_bins(self, state: int) -> bool:
        """Whether a state's bins are its environment's place bins.

        Parameters
        ----------
        state : int
            Index of the discrete state.

        Returns
        -------
        bool
            True if the state has one bin per place bin of its environment,
            False if it has a single bin without a position (see
            ``has_place_bins``).
        """
        return has_place_bins(self.observation_models[state], self.local_position_std)

    def likelihood_is_local(self, state: int) -> bool:
        """Whether a state's likelihood is evaluated at the animal's position.

        A Local state without a position kernel has one bin, whose likelihood
        is evaluated at the interpolated position; a Local state with a
        kernel is evaluated at every place bin and weighted by the kernel.

        Parameters
        ----------
        state : int
            Index of the discrete state.

        Returns
        -------
        bool
            True for a Local state without a position kernel.
        """
        return bool(
            self.observation_models[state].is_local and self.local_position_std is None
        )

    def densified(self) -> "StateSpace":
        """Return a copy with the transition read into a dense matrix.

        Returns
        -------
        StateSpace
            A copy with ``continuous_transitions`` a dense matrix (read
            within ``max_dense_transition_bytes``), no operator, and
            ``transition_representation`` "dense".
        """
        return replace(
            self,
            continuous_transitions=np.asarray(self.continuous_transitions),
            transition_operator=None,
            transition_representation="dense",
        )


def layout_parameters(detector: Any) -> dict[str, Any]:
    """Return the detector's current values of the parameters a fit records.

    The values are the detector's own objects, not copies.

    Parameters
    ----------
    detector : _DetectorBase
        Detector whose current parameters to read.

    Returns
    -------
    parameters : dict
        ``state_names``, ``observation_models`` and
        ``continuous_transition_types`` (a tuple of rows) as tuples,
        ``local_position_std``, ``discrete_transition_type``,
        ``discrete_initial_conditions``, and ``environments`` as a tuple of
        ``Environment.get_params`` dicts, never the fitted grids.
    """
    return {
        "state_names": tuple(detector.state_names),
        "observation_models": tuple(detector.observation_models),
        "continuous_transition_types": tuple(
            tuple(row) for row in detector.continuous_transition_types
        ),
        "local_position_std": detector.local_position_std,
        "discrete_transition_type": detector.discrete_transition_type,
        "discrete_initial_conditions": detector.discrete_initial_conditions,
        "environments": tuple(env.get_params() for env in detector.environments),
    }


def recorded_layout_parameters(detector: Any) -> dict[str, Any]:
    """Return deep copies of ``layout_parameters`` for a ``StateSpace`` to record.

    Changing a parameter's objects in place afterwards leaves the recorded
    ones as they were.

    Parameters
    ----------
    detector : _DetectorBase
        Detector whose current parameters to record.

    Returns
    -------
    parameters : dict
        Copies of ``layout_parameters(detector)``.
    """
    return copy.deepcopy(layout_parameters(detector))


def fitted_states(
    detector: Any,
) -> tuple[Sequence[str], Sequence[ObservationModel]]:
    """Return the state names and observation models that label a detector's results.

    The fit's, from ``state_space_``; the detector's current parameters before
    a fit, for results built without one.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted or unfitted detector.

    Returns
    -------
    state_names : sequence of str, length n_states
    observation_models : sequence of ObservationModel, length n_states
    """
    states = vars(detector).get("state_space_", detector)
    return states.state_names, states.observation_models


def has_place_bins(
    observation: ObservationModel, local_position_std: float | None
) -> bool:
    """Whether a state's bins are its environment's place bins.

    No-Spike, and Local without a position kernel (``local_position_std`` is
    None), have a single state bin with no position instead.

    Parameters
    ----------
    observation : ObservationModel
        The state's observation model.
    local_position_std : float or None
        The detector's ``local_position_std``, which decides whether a Local
        state has a position kernel over place bins.

    Returns
    -------
    bool
        True if the state has one bin per place bin of its environment,
        False if it has a single bin without a position.
    """
    return not (
        observation.is_no_spike or (observation.is_local and local_position_std is None)
    )


def state_layout(
    observation_models: Sequence[ObservationModel],
    environments: Sequence[Environment],
    local_position_std: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each discrete state its bins.

    A state with place bins (see ``has_place_bins``) gets one bin per place
    bin of its environment; any other state gets one bin. The bins are
    numbered state by state, so each state's bins are one contiguous block.

    Parameters
    ----------
    observation_models : sequence of ObservationModel, length n_states
        Observation model of each discrete state.
    environments : sequence of Environment
        The fitted environments, looked up by the observation models'
        ``environment_name``.
    local_position_std : float or None
        Whether Local states have a position kernel (not None) over place
        bins.

    Returns
    -------
    state_ind : np.ndarray of int, shape (n_state_bins,)
        Discrete state of each state bin.
    bin_sizes : np.ndarray of int, shape (n_states,)
        Number of bins of each discrete state.
    is_track_interior : np.ndarray of bool, shape (n_state_bins,)
        Whether each state bin is on the track interior; True for a state
        without place bins.
    """
    bin_sizes = []
    state_ind = []
    is_track_interior = []
    for ind, obs in enumerate(observation_models):
        if not has_place_bins(obs, local_position_std):
            bin_sizes.append(1)
            state_ind.append(np.full(1, ind, dtype=int))
            is_track_interior.append(np.ones(1, dtype=bool))
        else:
            environment = find_environment_by_name(environments, obs.environment_name)
            bin_sizes.append(environment.place_bin_centers_.shape[0])
            state_ind.append(np.full(bin_sizes[-1], ind, dtype=int))
            is_track_interior.append(environment.is_track_interior_flat_)

    return (
        np.concatenate(state_ind),
        np.array(bin_sizes),
        np.concatenate(is_track_interior),
    )


def initial_conditions(
    *,
    observation_models: Sequence[ObservationModel],
    continuous_initial_conditions_types: Sequence[Any],
    environments: Sequence[Environment],
    local_position_std: float | None,
    state_names: Sequence[str],
    discrete_initial_conditions: np.ndarray,
    state_ind: np.ndarray,
    is_track_interior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the initial probability of each state bin.

    Parameters
    ----------
    observation_models : sequence of ObservationModel, length n_states
        Observation model of each discrete state.
    continuous_initial_conditions_types : sequence, length n_states
        Initial-condition type of each discrete state.
    environments : sequence of Environment
        The fitted environments.
    local_position_std : float or None
        Whether Local states have a position kernel; a multi-bin Local state
        starts spread over its place bins.
    state_names : sequence of str, length n_states
        Names of the discrete states, for error messages.
    discrete_initial_conditions : np.ndarray, shape (n_states,)
        Initial probability of each discrete state.
    state_ind : np.ndarray of int, shape (n_state_bins,)
        Discrete state of each state bin, from ``state_layout``.
    is_track_interior : np.ndarray of bool, shape (n_state_bins,)
        Whether each state bin is on the track interior, from
        ``state_layout``.

    Returns
    -------
    continuous_initial_conditions : np.ndarray, shape (n_state_bins,)
        Initial probability distribution over the bins within each state.
    initial_conditions : np.ndarray, shape (n_state_bins,)
        Overall initial probability distribution across all state bins.
    """
    logger.info("Fitting initial conditions...")

    ic_parts = []
    for state, (obs, cont_ic) in enumerate(
        zip(
            observation_models,
            continuous_initial_conditions_types,
            strict=False,
        )
    ):
        # Multi-bin local uses full spatial initial conditions
        effective_obs = obs
        if obs.is_local and local_position_std is not None:
            effective_obs = replace(obs, is_local=False)
        ic_parts.append(cont_ic.make_initial_conditions(effective_obs, environments))
        _check_state_initial_conditions(
            ic_parts[-1],
            state_name=state_names[state],
            initial_conditions=cont_ic,
            is_interior=is_track_interior[state_ind == state],
            discrete_weight=discrete_initial_conditions[state],
        )
    continuous_initial_conditions = np.concatenate(ic_parts)
    return (
        continuous_initial_conditions,
        continuous_initial_conditions * discrete_initial_conditions[state_ind],
    )


def continuous_state_transitions(
    continuous_transition_types: Sequence[Sequence[Any]],
    environments: Sequence[Environment],
    observation_models: Sequence[ObservationModel],
    local_position_std: float | None,
    state_ind: np.ndarray,
    bin_sizes: np.ndarray,
    *,
    context: TransitionContext,
    transition_representation: TransitionRepresentation = "dense",
    max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> "tuple[np.ndarray | LazyDenseTransition, BlockTransitionOperator | None]":
    """Construct the transition between the state bins of the continuous states.

    Parameters
    ----------
    continuous_transition_types : sequence of sequences, shape (n_states, n_states)
        Transition model of each pair of discrete states.
    environments : sequence of Environment
        The fitted environments.
    observation_models : sequence of ObservationModel, length n_states
        Observation model of each discrete state.
    local_position_std : float or None
        Whether Local states have a position kernel over place bins.
    state_ind : np.ndarray of int, shape (n_state_bins,)
        Discrete state of each state bin, from ``state_layout``.
    bin_sizes : np.ndarray of int, shape (n_states,)
        Number of bins of each discrete state, from ``state_layout``.
    context : TransitionContext
        Fit data the transitions may read.
    transition_representation : {"dense", "structured", "auto"}, optional
        By default "dense", which builds the full matrix. "structured"
        requires exact supported state-block operators; "auto" permits
        budgeted dense fallback blocks.
    max_dense_transition_bytes : int, optional
        Byte limit for retained fallback blocks, temporary dense graph
        distances, and explicit reads of the lazy view, by default
        256 MiB. Does not limit "dense".

    Returns
    -------
    continuous_state_transitions : np.ndarray or LazyDenseTransition, shape (n_state_bins, n_state_bins)
        Probability of transitioning between bins, assuming a transition
        between the corresponding discrete states occurs. A budgeted lazy
        view of the operator unless transition_representation="dense".
    transition_operator : BlockTransitionOperator or None
        The structured operator; None for "dense".
    """
    logger.info("Fitting continuous state transition...")

    _validate_transition_arguments(
        transition_representation, max_dense_transition_bytes
    )
    if transition_representation != "dense":
        from non_local_detector.transition_operators import (
            LazyDenseTransition,
            build_transition_operator,
        )

        operator = build_transition_operator(
            continuous_transition_types,
            environments,
            bin_sizes,  # type: ignore[arg-type]  # typed Sequence[int] in transition_operators; an ndarray works
            observation_models=observation_models,
            local_position_std=local_position_std,
            allow_dense_fallback=transition_representation == "auto",
            max_dense_bytes=int(max_dense_transition_bytes),
            context=context,
        )
        return (
            LazyDenseTransition(operator, max_bytes=int(max_dense_transition_bytes)),
            operator,
        )

    n_total_bins = len(state_ind)
    transitions = np.zeros((n_total_bins, n_total_bins))

    for from_state, row in enumerate(continuous_transition_types):
        for to_state, transition in enumerate(row):
            _check_continuous_transition(transition)
            inds = np.ix_(state_ind == from_state, state_ind == to_state)

            transition = _multi_bin_local_transition(
                transition,
                observation_models[from_state],
                observation_models[to_state],
                local_position_std,
            )

            n_row_bins = int(np.max(inds[0].shape))
            n_col_bins = int(np.max(inds[1].shape))

            # Blocks between a one-bin and a spatial state follow the
            # structured operator's rule, so both representations agree.
            _check_block_environments(
                transition,
                environments,
                observation_models[from_state].environment_name,
                observation_models[to_state].environment_name,
                n_row_bins,
                n_col_bins,
            )
            one_bin_block = _one_bin_block(
                transition, environments, n_row_bins, n_col_bins
            )
            if one_bin_block is not None:
                transitions[inds] = one_bin_block.entries(
                    np.arange(n_row_bins)[:, np.newaxis],
                    np.arange(n_col_bins)[np.newaxis],
                )
            else:
                transitions[inds] = _make_transition_block(
                    transition,
                    environments,
                    context,
                    (n_row_bins, n_col_bins),
                )
    return transitions, None
