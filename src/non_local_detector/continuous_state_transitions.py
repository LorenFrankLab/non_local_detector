"""Classes for constructing different types of movement models.

Every continuous transition satisfies the :class:`ContinuousTransition`
protocol: ``make_state_transition(environments, context=None)`` returns the
block of transition probabilities between the place bins of two states.
``context`` is a :class:`TransitionContext` holding the fit data that
data-driven transitions such as :class:`EmpiricalMovement` read.
"""

import inspect
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal

from non_local_detector import _validation as val
from non_local_detector.environment import Environment, find_environment_by_name
from non_local_detector.exceptions import ConfigurationError, ValidationError
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.observation_models import ObservationModel

if TYPE_CHECKING:
    from non_local_detector.transition_operators import Block


@dataclass(frozen=True, eq=False)
class TransitionContext:
    """Fit data a continuous transition may read.

    With ``position``, an unset ``is_training`` selects every time point and
    unset ``encoding_group_labels`` puts every time point in encoding group 0,
    as in a detector fit, so an ``EmpiricalMovement`` for another group gets
    no samples. ``is_training`` is squeezed to one dimension.

    Attributes
    ----------
    position : np.ndarray, shape (n_time, n_position_dims), optional
        Position of the animal.
    is_training : np.ndarray, shape (n_time,), optional
        Which time points to train on; all of them by default.
    encoding_group_labels : np.ndarray, shape (n_time,), optional
        Encoding group of each time point; group 0 by default.
    environment_labels : np.ndarray, shape (n_time,), optional
        Environment name of each time point. None means every time point is
        in every environment.
    """

    position: np.ndarray | None = None
    is_training: np.ndarray | None = None
    encoding_group_labels: np.ndarray | None = None
    environment_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        is_training = self.is_training
        if self.position is not None:
            n_time = self.position.shape[0]
            if is_training is None:
                is_training = np.ones((n_time,), dtype=bool)
            if self.encoding_group_labels is None:
                object.__setattr__(
                    self,
                    "encoding_group_labels",
                    np.zeros((n_time,), dtype=np.int32),
                )
        if is_training is not None:
            object.__setattr__(self, "is_training", np.asarray(is_training).squeeze())


@runtime_checkable
class ContinuousTransition(Protocol):
    """A model of movement between the place bins of two discrete states.

    The detectors call ``make_state_transition`` only for blocks between two
    spatial states or two one-bin states, and check that the block it
    returns has the two states' shape, is finite and non-negative, and has
    rows that sum to at most 1 (1 when every bin the walk can reach is
    modeled, 0 for bins nothing can leave). A block between a
    one-bin state (No-Spike, or Local without ``local_position_std``) and a
    spatial state is set without calling it: from the one-bin state it is
    uniform over the interior bins of the transition's destination
    environment (``environment2_name`` when set, else ``environment_name``,
    so ``Discrete`` cannot set it); into the one-bin state every bin moves
    there with probability 1. The environments a transition declares must
    be its states': the destination always, and the source
    (``environment_name``) for a block between two spatial states.

    A transition has a structured operator only if it implements
    ``transition_operators.SupportsStructuredBlock``, as the package's
    ``Uniform``, ``Identity``, ``Discrete`` and Euclidean ``RandomWalk`` (on a
    complete Cartesian grid) do. With ``transition_representation=
    "structured"`` any other transition raises ``UnsupportedTransitionError``;
    with ``"auto"`` its dense block is built from ``make_state_transition``,
    but a transition without ``SupportsStructuredBlock`` is refused when an
    environment's graph distances are deferred (as for an open-field grid fit
    with a non-dense representation). Use ``"dense"`` for such a transition.
    """

    @property
    def environment_name(self) -> str | None:
        """Name of the environment of the source bins; None for a transition
        without place bins, such as ``Discrete``."""
        ...

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,
    ) -> np.ndarray:
        """Transition probabilities from each source bin to each destination bin.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments, looked up by name.
        context : TransitionContext, optional
            Fit data, for transitions estimated from it.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_source_bins, n_destination_bins)
            Rows sum to at most 1: to 1, or to 0 for bins nothing can
            leave, or in between when probability leaves the modeled bins.
        """
        ...


def _normalize_row_probability(x: np.ndarray) -> np.ndarray:
    """Ensure the state transition matrix rows sum to 1.

    Parameters
    ----------
    x : np.ndarray, shape (n_rows, n_cols)

    Returns
    -------
    normalized_x : np.ndarray, shape (n_rows, n_cols)

    """
    # Handle cases where the sum is zero to avoid division by zero -> NaN
    row_sums = x.sum(axis=1, keepdims=True)
    # Use np.errstate to temporarily ignore invalid division warnings
    with np.errstate(invalid="ignore"):
        normalized_x = np.where(row_sums > 0, x / row_sums, 0.0)
    # Ensure any remaining NaNs (though unlikely with the above) are zero
    normalized_x[np.isnan(normalized_x)] = 0.0
    return normalized_x


def estimate_movement_var(position: np.ndarray) -> np.ndarray:
    """Estimate the movement variance based on position differences.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dim)
        Position of the animal.

    Returns
    -------
    movement_var : np.ndarray, shape (n_position_dim,)
        Variance of the movement per time bin.
    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    return np.cov(np.diff(position, axis=0), rowvar=False)


def _random_walk_on_track_graph(
    place_bin_centers: np.ndarray,
    movement_mean: float,
    movement_var: float,
    place_bin_center_ind_to_node: np.ndarray,
    distance_between_nodes: dict[int, dict[int, float]] | LazyGraphDistances,
) -> np.ndarray:
    """Estimates the random walk probabilities based on the spatial
    topology given by the track graph.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins,)
        Center positions of the bins (assumed 1D for track graph).
    movement_mean : float
        Mean displacement for the random walk.
    movement_var : float
        Variance of the displacement for the random walk.
    place_bin_center_ind_to_node : np.ndarray, shape (n_position_bins,)
        Mapping from the place bin center index to the corresponding node ID
        in the track graph. Bins not on the track might map to a sentinel value like -1.
    distance_between_nodes : dict[dict]
        Nested dictionary where `distance_between_nodes[node1][node2]` gives the
        shortest path distance between `node1` and `node2` on the track graph.

    Returns
    -------
    random_walk : np.ndarray, shape (n_position_bins, n_position_bins)
        Transition probabilities between place bins based on graph distance.
    """
    state_transition = np.zeros((place_bin_centers.size, place_bin_centers.size))
    gaussian = multivariate_normal(mean=movement_mean, cov=movement_var)

    for bin_ind1, node1 in enumerate(place_bin_center_ind_to_node):
        for bin_ind2, node2 in enumerate(place_bin_center_ind_to_node):
            try:
                state_transition[bin_ind1, bin_ind2] = gaussian.pdf(
                    distance_between_nodes[node1][node2]
                )
            except KeyError:
                # bins not on track interior will be -1 and not in distance
                # between nodes
                continue

    return state_transition


def _euclidean_random_walk(
    environment: Environment, movement_mean: float, movement_var: float
) -> np.ndarray:
    """Estimates the random walk probabilities based on the Euclidean distance

    Parameters
    ----------
    environment : Environment
    movement_mean : float
    movement_var : float

    Returns
    -------
    transition : np.ndarray, shape (n_position_bins, n_position_bins)

    """
    if environment.place_bin_centers_ is not None:
        place_bin_centers = environment.place_bin_centers_
    else:
        msg = "Environment must have defined place bin centers"
        raise ValueError(msg)

    return np.stack(
        [
            multivariate_normal(mean=center + movement_mean, cov=movement_var).pdf(
                place_bin_centers
            )
            for center in place_bin_centers
        ],
        axis=0,
    )


class _DropsLegacyEnvironment:
    """Unpickle without the environment that earlier releases stored.

    Before 0.7.0, ``make_state_transition`` stored the environment it was
    built on as ``environment`` (``environment1`` and ``environment2`` for
    ``Uniform``). Pickles made then carry those attributes, which would keep
    the old grid alive and break ``type(t)(**vars(t))``.
    """

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the fields, dropping legacy environment attributes."""
        for name in ("environment", "environment1", "environment2"):
            state.pop(name, None)
        self.__dict__.update(state)


class _EnvironmentBlock:
    """``SupportsStructuredBlock`` defaults of a transition within one
    environment.

    Its dense block spans the bins of ``environment_name`` in both
    directions, and it reads no graph distances.
    """

    environment_name: str

    def dense_block_shape(self, environments: Sequence[Environment]) -> tuple[int, int]:
        """``(n_bins, n_bins)`` of ``environment_name``, without building
        the block."""
        environment = find_environment_by_name(environments, self.environment_name)
        n_bins = len(cast(np.ndarray, environment.place_bin_centers_))
        return (n_bins, n_bins)

    def graph_distance_environment(
        self,
        environments: Sequence[Environment],  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> Environment | None:
        """None: ``make_state_transition`` reads no graph distances."""
        return None


def _no_structured_block(transition: object) -> "Block":
    """Refuse a structured block for a transition that has none."""
    from non_local_detector.transition_operators import (
        _DENSE_BLOCK_HINT,
        UnsupportedTransitionError,
    )

    msg = f"{type(transition).__name__} has no structured operator; {_DENSE_BLOCK_HINT}"
    raise UnsupportedTransitionError(msg)


_DIRECTIONS = ("inward", "outward")


def _validate_direction(direction: str | None) -> None:
    """Check a ``RandomWalk.direction``.

    Raises
    ------
    ValidationError
        If it is not None, "inward" or "outward" (case-insensitive).
    """
    if direction is not None and direction.lower() not in _DIRECTIONS:
        msg = f"direction must be 'inward' or 'outward', got {direction!r}"
        raise ValidationError(
            msg, hint="Use direction=None for an unconstrained random walk."
        )


@dataclass
class RandomWalk(_DropsLegacyEnvironment, _EnvironmentBlock):
    """A transition where the movement stays locally close in space.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit, defaults to "".
    movement_var : float, optional
        Variance of the displacement in one time step. Defaults to 6.0.
    movement_mean : float, optional
        Mean displacement in one time step. Defaults to 0.0.
    use_manifold_distance : bool, optional
        Whether to use graph distance (if available) or Euclidean distance.
        Defaults to False (Euclidean).
    direction : ("inward", "outward"), optional
        If using manifold distance on a 2D track, constrain movement direction
        relative to the track center (case-insensitive). Defaults to None (no
        constraint).
    use_diffusion : bool, optional
        Not stored. Releases up to 0.5.12 stored it, so their parameter sets
        (for example Spyglass rows) rebuild with it; only False, the default,
        is accepted, as the diffusion random walk it selected was removed.

    Raises
    ------
    ValidationError
        If ``direction`` is not None, "inward" or "outward", or
        ``use_diffusion`` is True.
    """

    environment_name: str = ""
    movement_var: float = 6.0
    movement_mean: float = 0.0
    use_manifold_distance: bool = False
    direction: str | None = None
    use_diffusion: InitVar[bool] = False

    def __post_init__(self, use_diffusion: bool) -> None:
        if use_diffusion:
            msg = "RandomWalk(use_diffusion=True) is no longer supported"
            raise ValidationError(
                msg,
                hint="The diffusion random walk of releases up to 0.5.12 was "
                "removed in 0.5.13. use_manifold_distance=True is the closest: "
                "a Gaussian in distance along the environment, not diffusion.",
            )
        _validate_direction(self.direction)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the fields, including those of releases up to 0.5.12.

        Those stored ``use_diffusion`` and neither ``use_manifold_distance``
        nor ``direction``; ``use_diffusion=False`` was this Euclidean walk.
        """
        if state.get("use_diffusion") is False:
            del state["use_diffusion"]
            state.setdefault("use_manifold_distance", False)
            state.setdefault("direction", None)
        super().__setstate__(state)

    def structured_block(self, environments: Sequence[Environment]) -> "Block":
        """Separable Gaussian block of a Euclidean random walk on a complete
        Cartesian grid.

        Raises
        ------
        UnsupportedTransitionError
            For a track graph, manifold distance, a direction, a
            non-separable covariance or a grid whose float32 product would
            underflow.
        """
        from non_local_detector.transition_operators import _gaussian_block

        return _gaussian_block(
            self, find_environment_by_name(environments, self.environment_name)
        )

    def graph_distance_environment(
        self, environments: Sequence[Environment]
    ) -> Environment | None:
        """The environment whose graph distances a manifold-distance random
        walk reads; None for a Euclidean one."""
        if not self.use_manifold_distance:
            return None
        return find_environment_by_name(environments, self.environment_name)

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        """Create a transition matrix for a given environment.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.
        context : TransitionContext, optional
            Ignored; this transition does not depend on the fit data.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized transition probability matrix.
        """
        environment = find_environment_by_name(environments, self.environment_name)
        # ``direction`` is only consulted on the N-D-grid manifold-distance path
        # (no track_graph + use_manifold_distance=True). On every other path it
        # would be silently ignored, so surface the misconfiguration loudly.
        if self.direction is not None and not (
            environment.track_graph is None and self.use_manifold_distance
        ):
            msg = (
                "direction-aware RandomWalk is only applied on an N-D grid "
                "environment with use_manifold_distance=True; got "
                f"track_graph={'set' if environment.track_graph is not None else 'None'}"
                f", use_manifold_distance={self.use_manifold_distance}. The "
                "direction= argument would otherwise be silently ignored."
            )
            raise ConfigurationError(
                (msg),
                hint=(
                    "Set use_manifold_distance=True on an environment without a "
                    "track_graph (a fitted N-D grid), or remove the direction= "
                    "argument."
                ),
            )
        if environment.track_graph is None:
            transition_matrix = self._handle_no_track_graph(environment)
        else:
            # Linearized position
            transition_matrix = self._handle_with_track_graph(environment)
        is_track_interior = environment.is_track_interior_flat_
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0
        return _normalize_row_probability(transition_matrix)

    def _handle_no_track_graph(self, environment: Environment) -> np.ndarray:
        """Calculate transition for environments without a defined track graph."""
        if not self.use_manifold_distance:
            transition_matrix = _euclidean_random_walk(
                environment, self.movement_mean, self.movement_var
            )
        else:
            distance_data = environment.distance_between_nodes_
            if isinstance(distance_data, LazyGraphDistances):
                # Deferred graph distances are densified within their own byte
                # budget; falling back to Euclidean would change the model.
                distance_data = distance_data.to_dense()
            if isinstance(distance_data, np.ndarray):
                transition_matrix = (
                    multivariate_normal(mean=self.movement_mean, cov=self.movement_var)
                    .pdf(distance_data.flat)
                    .reshape(distance_data.shape)
                )
            else:
                # Fallback for dict or None case
                transition_matrix = _euclidean_random_walk(
                    environment, self.movement_mean, self.movement_var
                )

            if self.direction is not None:
                if environment.track_graphDD is None or not isinstance(
                    distance_data, np.ndarray
                ):
                    msg = (
                        "direction-aware RandomWalk requires an N-D track graph "
                        "and an array-valued distance_between_nodes_; got "
                        f"track_graphDD={environment.track_graphDD!r}, "
                        "distance_between_nodes_="
                        f"{type(environment.distance_between_nodes_).__name__}."
                    )
                    raise ConfigurationError(
                        (msg),
                        hint=(
                            "Either remove the direction= argument, or fit the "
                            "environment as an N-D grid (do not pass a track_graph) "
                            "so it has a track_graphDD and an array-valued "
                            "distance_between_nodes_."
                        ),
                    )
                # ``direction`` may have been reassigned since construction.
                _validate_direction(self.direction)
                direction_func = {
                    "inward": np.greater_equal,
                    "outward": np.less_equal,
                }[self.direction.lower()]

                centrality = nx.closeness_centrality(
                    environment.track_graphDD, distance="distance"
                )
                center_node_id = list(centrality.keys())[
                    np.argmax(list(centrality.values()))
                ]
                transition_matrix *= direction_func(
                    distance_data[:, [center_node_id]],
                    distance_data[[center_node_id]],
                )

        return transition_matrix

    def _handle_with_track_graph(self, environment: Environment) -> np.ndarray:
        """Calculate transition for environments with a defined track graph (typically 1D)."""
        if environment.place_bin_centers_ is not None:
            n_position_dims = environment.place_bin_centers_.shape[1]
        else:
            msg = "Environment must have defined place bin centers"
            raise ValueError(msg)
        if n_position_dims != 1:
            msg = "Random walk with track graph is only implemented for 1D environments"
            raise NotImplementedError(msg)

        if environment.place_bin_centers_nodes_df_ is not None:
            place_bin_center_ind_to_node = np.asarray(
                environment.place_bin_centers_nodes_df_.node_id
            )
        else:
            msg = "Environment must have defined place_bin_centers_nodes_df_ for track graph"
            raise ValueError(msg)

        if environment.distance_between_nodes_ is None:
            msg = (
                "Environment must have defined distance_between_nodes_ for track graph"
            )
            raise ValueError(msg)

        return _random_walk_on_track_graph(
            environment.place_bin_centers_,
            self.movement_mean,
            self.movement_var,
            place_bin_center_ind_to_node,
            cast(
                dict[int, dict[int, float]] | LazyGraphDistances,
                environment.distance_between_nodes_,
            ),
        )


@dataclass
class Uniform(_DropsLegacyEnvironment, _EnvironmentBlock):
    """Transition where all valid destination bins are equally likely.

    Attributes
    ----------
    environment_name : str, optional
        Name of the source environment. Defaults to "".
    environment2_name : str, optional
        Name of the destination environment if different from the source.
        If None, assumes transition within the same environment. Defaults to None.
    """

    environment_name: str = ""
    environment2_name: str | None = None

    def structured_block(self, environments: Sequence[Environment]) -> "Block":
        """Rank-one block from the interior of ``environment_name`` to the
        interior of ``environment2_name`` (or the same environment)."""
        from non_local_detector.transition_operators import UniformBlock

        source = find_environment_by_name(
            environments, self.environment_name
        ).is_track_interior_flat_
        destination = (
            source.copy()
            if self.environment2_name is None
            else find_environment_by_name(
                environments, self.environment2_name
            ).is_track_interior_flat_
        )
        return UniformBlock(
            source,
            destination,
            np.array(1.0 / destination.sum() if destination.any() else 0.0),
        )

    def dense_block_shape(self, environments: Sequence[Environment]) -> tuple[int, int]:
        """``(n_bins1, n_bins2)`` of the two environments."""
        n_bins1, _ = super().dense_block_shape(environments)
        if self.environment2_name is None:
            return (n_bins1, n_bins1)
        environment2 = find_environment_by_name(environments, self.environment2_name)
        return (n_bins1, len(cast(np.ndarray, environment2.place_bin_centers_)))

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        """Create a uniform transition matrix between environments.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.
        context : TransitionContext, optional
            Ignored; this transition does not depend on the fit data.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_bins1, n_bins2)
            Row-normalized uniform transition probability matrix.
        """
        environment1 = find_environment_by_name(environments, self.environment_name)
        if environment1.place_bin_centers_ is not None:
            n_bins1 = environment1.place_bin_centers_.shape[0]
        else:
            msg = "Environment must have defined place bin centers"
            raise ValueError(msg)

        is_track_interior1 = environment1.is_track_interior_flat_

        if self.environment2_name is None:
            n_bins2 = n_bins1
            is_track_interior2 = is_track_interior1.copy()
        else:
            environment2 = find_environment_by_name(
                environments, self.environment2_name
            )
            if environment2.place_bin_centers_ is not None:
                n_bins2 = environment2.place_bin_centers_.shape[0]
            else:
                msg = "Environment must have defined place bin centers"
                raise ValueError(msg)

            is_track_interior2 = environment2.is_track_interior_flat_

        transition_matrix = np.ones((n_bins1, n_bins2))

        transition_matrix[~is_track_interior1] = 0.0
        transition_matrix[:, ~is_track_interior2] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Identity(_DropsLegacyEnvironment, _EnvironmentBlock):
    """A transition where the movement must stay within the same place bin.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    """

    environment_name: str = ""

    def structured_block(self, environments: Sequence[Environment]) -> "Block":
        """Identity block on the interior of ``environment_name``."""
        from non_local_detector.transition_operators import IdentityBlock

        return IdentityBlock(
            find_environment_by_name(
                environments, self.environment_name
            ).is_track_interior_flat_
        )

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        """Create an identity transition matrix for a given environment.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.
        context : TransitionContext, optional
            Ignored; this transition does not depend on the fit data.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Identity matrix where invalid bins have zero probability.
        """
        environment = find_environment_by_name(environments, self.environment_name)
        if environment.place_bin_centers_ is not None:
            n_bins = environment.place_bin_centers_.shape[0]
        else:
            msg = "Environment must have defined place bin centers"
            raise ValueError(msg)

        transition_matrix = np.identity(n_bins)

        is_track_interior = environment.is_track_interior_flat_
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class EmpiricalMovement(_DropsLegacyEnvironment, _EnvironmentBlock):
    """A transition matrix estimated from the animal's actual movement patterns.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    encoding_group : str or int, optional
        Name of encoding group to filter data by. Defaults to 0.
    speedup : int, optional
        Factor by which to speed up the estimated transitions. Corresponds to
        taking matrix powers of the one-step transition matrix. Defaults to 1.
    is_time_reversed : bool, optional
        If True, estimate transitions based on p(z_{t-1} | z_t) instead of p(z_{t+1} | z_t).
        Defaults to False.
    """

    environment_name: str = ""
    encoding_group: str | int = 0
    speedup: int = 1
    is_time_reversed: bool = False

    def structured_block(
        self,
        environments: Sequence[Environment],  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> "Block":
        """None exists; raises ``UnsupportedTransitionError``."""
        return _no_structured_block(self)

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,
    ) -> np.ndarray:
        """Create a transition matrix based on observed animal movement.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.
        context : TransitionContext
            Required; its ``position`` is binned. On a track graph,
            multi-dimensional position is linearized and one-column position
            is taken as linear (see ``Environment.decoding_position``). Only
            time points that are training points, in ``encoding_group`` and
            in this environment (all if ``environment_labels`` is None) are
            used; see ``TransitionContext`` for the defaults.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized transition probability matrix, potentially powered by `speedup`.

        Raises
        ------
        ValueError
            If ``context`` or its ``position`` is None.
        """
        if context is None or context.position is None:
            msg = (
                "EmpiricalMovement needs the animal's position: pass "
                "context=TransitionContext(position=...)"
            )
            raise ValueError(msg)
        environment = find_environment_by_name(environments, self.environment_name)

        position = context.position
        n_time = position.shape[0]
        # A context with position has set is_training and the group labels.
        is_training = cast(np.ndarray, context.is_training)
        is_encoding = (
            cast(np.ndarray, context.encoding_group_labels) == self.encoding_group
        )

        if context.environment_labels is None:
            is_environment = np.ones((n_time,), dtype=bool)
        else:
            is_environment = context.environment_labels == self.environment_name

        position = position if position.ndim > 1 else position[:, np.newaxis]
        position = environment.decoding_position(
            position[is_training & is_encoding & is_environment]
        )

        if self.is_time_reversed:
            samples = np.concatenate((position[1:], position[:-1]), axis=1)
        else:
            samples = np.concatenate((position[:-1], position[1:]), axis=1)
        # Fitted edges define the source and destination ranges, including
        # environments whose ranges were inferred from their position data.
        state_transition, _ = np.histogramdd(
            samples,
            bins=cast(tuple[np.ndarray, ...], environment.edges_) * 2,
        )
        original_shape = state_transition.shape
        n_position_dims = position.shape[1]
        shape_2d = np.prod(original_shape[:n_position_dims])
        state_transition = _normalize_row_probability(
            state_transition.reshape((shape_2d, shape_2d))
        )

        return np.linalg.matrix_power(state_transition, self.speedup)


@dataclass
class RandomWalkDirection1(_DropsLegacyEnvironment, _EnvironmentBlock):
    """A Gaussian random walk constrained to move in one direction (increasing index).

    Only valid for 1D environments.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    movement_var : float, optional
        Variance of the displacement. Defaults to 6.0.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def structured_block(
        self,
        environments: Sequence[Environment],  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> "Block":
        """None exists; raises ``UnsupportedTransitionError``."""
        return _no_structured_block(self)

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        """Create a unidirectional transition matrix.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.
        context : TransitionContext, optional
            Ignored; this transition does not depend on the fit data.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized upper triangular transition matrix.
        """
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.triu(random))


@dataclass
class RandomWalkDirection2(_DropsLegacyEnvironment, _EnvironmentBlock):
    """A Gaussian random walk constrained to move in one direction (decreasing index).

    Only valid for 1D environments.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit. Defaults to "".
    movement_var : float, optional
        Variance of the displacement. Defaults to 6.0.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def structured_block(
        self,
        environments: Sequence[Environment],  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> "Block":
        """None exists; raises ``UnsupportedTransitionError``."""
        return _no_structured_block(self)

    def make_state_transition(
        self,
        environments: Sequence[Environment],
        context: TransitionContext | None = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        """Create a unidirectional transition matrix.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.
        context : TransitionContext, optional
            Ignored; this transition does not depend on the fit data.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)
            Row-normalized lower triangular transition matrix.
        """
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.tril(random))


class Discrete:
    """Creates a continuous transition matrix for a discrete state space.

    Essentially, there is no continuous state space, so the transition matrix
    is just an identity matrix.
    """

    @property
    def environment_name(self) -> None:
        """None: a discrete state has no place bins."""
        return None

    def structured_block(
        self,
        environments: Sequence[Environment] = (),  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> "Block":
        """The (1, 1) identity block."""
        from non_local_detector.transition_operators import IdentityBlock

        return IdentityBlock(np.ones(1, dtype=bool))

    def dense_block_shape(
        self,
        environments: Sequence[Environment] = (),  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> tuple[int, int]:
        """(1, 1): a discrete state has one bin."""
        return (1, 1)

    def graph_distance_environment(
        self,
        environments: Sequence[Environment] = (),  # noqa: ARG002 -- SupportsStructuredBlock signature
    ) -> Environment | None:
        """None: no graph distances are read."""
        return None

    def make_state_transition(
        self,
        environments: Sequence[Environment] = (),  # noqa: ARG002 -- ContinuousTransition signature
        context: TransitionContext | None = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        """Create a continuous transition matrix for a discrete state space.

        Since there is no continuous spatial component, the transition matrix
        is effectively a 1x1 identity matrix representing staying within the
        single discrete "location".

        Parameters
        ----------
        environments : sequence of Environment, optional
            Ignored; a discrete state has no place bins.
        context : TransitionContext, optional
            Ignored; this transition does not depend on the fit data.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (1, 1)
            The 1x1 identity transition matrix.
        """
        return np.ones((1, 1))


def _multi_bin_local_transition(
    transition: ContinuousTransition,
    from_observation: ObservationModel,
    to_observation: ObservationModel,
    local_position_std: float | None,
) -> ContinuousTransition:
    """The transition to build for a block with a multi-bin Local state.

    A multi-bin Local state (``local_position_std`` set) has place bins, so a
    ``Discrete`` block into or out of it becomes ``Uniform`` over the Local
    state's environment; a (1, 1) ``Discrete`` block would otherwise be
    broadcast over every bin pair. Other transitions are returned unchanged.

    Parameters
    ----------
    transition : ContinuousTransition
    from_observation, to_observation : ObservationModel
        Observation models of the block's source and destination states.
    local_position_std : float or None
        The detector's Local width; None keeps a single-bin Local state.

    Returns
    -------
    transition : ContinuousTransition

    Raises
    ------
    ValueError
        If the ``Discrete`` block joins states in different environments.
    """
    if not isinstance(transition, Discrete) or local_position_std is None:
        return transition
    if not (from_observation.is_local or to_observation.is_local):
        return transition
    # Cross-environment transitions with multi-bin local are not supported.
    if from_observation.environment_name != to_observation.environment_name:
        msg = (
            f"Cross-environment Discrete() transition between "
            f"'{from_observation.environment_name}' and "
            f"'{to_observation.environment_name}' is not supported with "
            f"multi-bin local (local_position_std is set). Use explicit "
            f"Uniform(environment_name=..., environment2_name=...) instead."
        )
        raise ValueError(msg)
    local_observation = (
        from_observation if from_observation.is_local else to_observation
    )
    return Uniform(environment_name=local_observation.environment_name)


def _check_continuous_transition(transition: object) -> None:
    """Check that a transition has the ``ContinuousTransition`` members.

    Raises
    ------
    ValidationError
        If it lacks ``make_state_transition`` or ``environment_name``.
    """
    if not isinstance(transition, ContinuousTransition):
        missing = [
            name
            for name in ("make_state_transition", "environment_name")
            if not hasattr(transition, name)
        ]
        msg = f"{type(transition).__name__} is not a ContinuousTransition"
        raise ValidationError(
            msg,
            expected=(
                "make_state_transition(environments, context=None) and an "
                "environment_name attribute"
            ),
            got=f"an object without {' or '.join(missing)}",
            hint=(
                "Give the transition an environment_name: the environment of "
                "its source bins, or None if it has no place bins."
            ),
        )


@runtime_checkable
class _NamesDestinationEnvironment(Protocol):
    """A transition into another environment, such as ``Uniform``."""

    @property
    def environment2_name(self) -> str | None:
        """Name of the destination environment; None for the source's."""
        ...


def _declared_destination(transition: ContinuousTransition) -> str | None:
    """The environment a transition moves into: its ``environment2_name``
    when set, else its ``environment_name``."""
    if (
        isinstance(transition, _NamesDestinationEnvironment)
        and transition.environment2_name is not None
    ):
        return transition.environment2_name
    return transition.environment_name


def _check_block_environments(
    transition: ContinuousTransition,
    environments: Sequence[Environment],
    source_environment_name: str | None,
    destination_environment_name: str | None,
    source_size: int,
    target_size: int,
) -> None:
    """Check the environments a transition declares against its block's states.

    The dense and the structured constructors call this for every block, so
    a mismatch raises the same error from both. A block into a one-bin state
    (probability 1 into its single bin) and a transition without place bins
    (``environment_name=None``, as ``Discrete``) are not checked here.

    Parameters
    ----------
    transition : ContinuousTransition
    environments : sequence of Environment
    source_environment_name, destination_environment_name : str or None
        Environments of the block's source and destination states (their
        observation models' ``environment_name``); None when unknown, which
        skips that comparison.
    source_size, target_size : int
        Bins of the source and destination states.

    Raises
    ------
    ValidationError
        If a declared environment does not exist, the declared destination
        (``environment2_name`` when set, else ``environment_name``) is not
        the destination state's environment, or, between two spatial states,
        the declared source (``environment_name``) is not the source state's.
    """
    if target_size == 1 or transition.environment_name is None:
        return
    declared_source = transition.environment_name
    declared_destination = cast(str, _declared_destination(transition))
    find_environment_by_name(environments, declared_source)
    find_environment_by_name(environments, declared_destination)
    checks = [("destination", declared_destination, destination_environment_name)]
    if source_size > 1:
        checks.insert(0, ("source", declared_source, source_environment_name))
    for role, declared, state_environment in checks:
        if state_environment is not None and declared != state_environment:
            msg = (
                f"{type(transition).__name__} declares {role} environment "
                f"{declared!r}, but the {role} state is in environment "
                f"{state_environment!r}"
            )
            raise ValidationError(
                msg,
                expected=f"{role} environment {state_environment!r}",
                got=f"{role} environment {declared!r}",
                hint=(
                    "Name the states' environments in the transition: "
                    "environment_name is the source, and environment2_name "
                    "(Uniform) the destination when it differs, e.g. "
                    f"Uniform({declared_source!r}, {state_environment!r})."
                    if role == "destination"
                    else "Name the source state's environment in the "
                    "transition's environment_name."
                ),
            )


def _scalar_to_spatial_destination(
    transition: ContinuousTransition, environments: Sequence[Environment]
) -> np.ndarray:
    """Destination bins of a block from a state without place bins.

    A block from a single-bin state (such as Local or No-Spike) to a spatial
    state is uniform over the interior bins of the transition's declared
    destination environment (``environment2_name`` when set, else
    ``environment_name``), whatever the transition;
    ``_check_block_environments`` checks that this is the destination
    state's environment.

    Parameters
    ----------
    transition : ContinuousTransition
    environments : sequence of Environment

    Returns
    -------
    destination : np.ndarray, shape (n_destination_bins,)
        Boolean interior mask of the destination environment.

    Raises
    ------
    ValidationError
        If the transition's ``environment_name`` is None, or its declared
        destination names no environment.
    """
    destination = _declared_destination(transition)
    if destination is None:
        # No fallback may broadcast a scalar block across the destination bins.
        msg = (
            f"{type(transition).__name__} has environment_name=None, so it "
            "cannot set a block from a one-bin state to a spatial state"
        )
        raise ValidationError(
            msg,
            hint=(
                "Use a transition naming the destination environment, e.g. "
                "Uniform(environment_name=...)."
            ),
        )
    return find_environment_by_name(environments, destination).is_track_interior_flat_


def _check_block_shape(
    transition: object, shape: tuple[int, ...], expected: tuple[int, int]
) -> None:
    """Check the shape of a transition's block against its two states.

    Used for the dense block, a planned structured block and a planned dense
    shape alike, so every representation reports a mismatch the same way.

    Raises
    ------
    ValidationError
        If ``shape`` is not ``expected``.
    """
    if tuple(shape) == expected:
        return
    msg = (
        f"{type(transition).__name__} gives a block with the wrong shape for "
        "its two states"
    )
    raise ValidationError(
        msg,
        expected=f"shape {expected}, (source state bins, destination state bins)",
        got=f"shape {tuple(shape)}",
        hint=(
            "Return one row per bin of the source state's environment and "
            "one column per bin of the destination state's environment; a "
            "Discrete transition joins only states without place bins."
        ),
    )


def _make_transition_block(
    transition: ContinuousTransition,
    environments: Sequence[Environment],
    context: TransitionContext | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Call ``transition.make_state_transition(environments, context)`` and
    check the block it returns.

    Parameters
    ----------
    transition : ContinuousTransition
    environments : sequence of Environment
    context : TransitionContext or None
    shape : tuple of int
        ``(n_source_bins, n_destination_bins)`` of the two states.

    Returns
    -------
    state_transition_matrix : np.ndarray, shape (n_source_bins, n_destination_bins)

    Raises
    ------
    ValidationError
        If ``make_state_transition`` cannot take ``(environments, context)``,
        as a custom transition written before ``context`` existed, or returns
        a block that is not ``shape``, has negative values, or has a row that
        sums to more than 1 (beyond ``PROBABILITY_SUM_TOLERANCE``).
    DataError
        If the block has NaN or infinite values.
    """
    name = type(transition).__name__
    try:
        inspect.signature(transition.make_state_transition).bind(environments, context)
    except TypeError:
        msg = (
            f"{name}.make_state_transition cannot be "
            "called as make_state_transition(environments, context)"
        )
        raise ValidationError(
            msg,
            hint=(
                "Continuous transitions receive the fit data as a second "
                "argument; add `context=None` to the method's parameters, "
                "e.g. `def make_state_transition(self, environments, "
                "context=None)`."
            ),
        ) from None
    block = np.asarray(transition.make_state_transition(environments, context))
    _check_block_shape(transition, block.shape, shape)
    label = f"{name}.make_state_transition(...)"
    val.ensure_all_finite(block, label)
    val.ensure_all_non_negative(block, label)
    val.ensure_substochastic_matrix(block, label)
    return block
