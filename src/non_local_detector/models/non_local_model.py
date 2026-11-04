"""Detectors of non-local activity such as replay.

Four-state models, for sorted spikes and clusterless data, with Local,
No-Spike, Non-Local Continuous and Non-Local Fragmented states.
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import xarray as xr

from non_local_detector.exceptions import ValidationError
from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
from non_local_detector.models._input_validation import _validate_penalty_params
from non_local_detector.models.base import (
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    FrozenTransitionRows,
    Observations,
    StateNames,
    Stickiness,
)

__all__ = [
    "NonLocalClusterlessDetector",
    "NonLocalSortedSpikesDetector",
]


def _get_conditional_non_local_posterior(
    results: xr.Dataset,
    state_names: list[str],
    observation_models: Sequence[ObservationModel],
) -> xr.DataArray:
    """Posterior over position given that activity is non-local.

    Sums the posterior of every non-local state -- each state whose
    observation model is neither Local (``is_local``) nor No-Spike
    (``is_no_spike``) -- and normalizes across position.

    Parameters
    ----------
    results : xr.Dataset
        Results from the detector's ``predict`` method, containing
        ``acausal_posterior`` with a ``state`` coordinate.
    state_names : list of str, length n_states
        The detector's state names, as labelled in ``results``.
    observation_models : list of ObservationModel, length n_states
        The detector's observation models, in the same order.

    Returns
    -------
    conditional_posterior : xr.DataArray
        Posterior over position given a non-local state. Shape is
        ``(n_time, n_position_bins)`` for 1-D environments and
        ``(n_time, n_x_bins, n_y_bins)`` for 2-D. Rows where the non-local
        states have no posterior mass are NaN.

    Raises
    ------
    ValidationError
        If the detector has no non-local states, or ``results`` lacks them
        (it came from a different detector).
    """
    non_local_states = list(
        dict.fromkeys(
            name
            for name, obs in zip(state_names, observation_models, strict=True)
            if not obs.is_local and not obs.is_no_spike
        )
    )
    if not non_local_states:
        msg = "The detector has no non-local states"
        raise ValidationError(
            msg,
            expected="at least one observation model with is_local=False "
            "and is_no_spike=False",
            got=f"states {list(state_names)}",
        )
    result_states = set(results.acausal_posterior.coords["state"].values)
    missing = [name for name in non_local_states if name not in result_states]
    if missing:
        msg = f"results lack the detector's non-local states {missing}"
        raise ValidationError(
            msg,
            expected=f"results with states {list(state_names)}",
            got=f"states {sorted(result_states)}",
            hint="Pass results from this detector's predict method.",
        )

    acausal_posterior = sum(
        results.acausal_posterior.sel(state=name) for name in non_local_states
    )

    # Marginalize over position. After .sel(state=...), the position bins stay
    # stacked under the single "state_bins" dim (2D+) or appear as the unstacked
    # "position" dim (1D), so summing every non-"time" dim handles both. NB: this
    # is the stacked layout, not the named-position-dim convention used
    # elsewhere -- do not substitute _position_dims.get_position_dims here; it
    # would return [] for the stacked "state_bins" case and break normalization.
    position_dims = [d for d in acausal_posterior.dims if d != "time"]
    return cast(
        "xr.DataArray", acausal_posterior / acausal_posterior.sum(position_dims)
    )


class _NonLocalMixin:
    """Behaviour shared by the sorted-spikes and clusterless non-local detectors."""

    state_names: list[str]
    observation_models: tuple[ObservationModel, ...]
    non_local_position_penalty: float
    non_local_penalty_std: float

    def _non_local_position_penalty(self) -> float:
        """Return the non-local position penalty, validated where it is used.

        Validating at use, not only at construction, also checks values set
        later with ``set_params``.

        Returns
        -------
        penalty : float

        Raises
        ------
        ValidationError
            If the penalty is negative or NaN, or its standard deviation is
            not positive.
        """
        _validate_penalty_params(
            self.non_local_position_penalty, self.non_local_penalty_std
        )
        return self.non_local_position_penalty

    def get_conditional_non_local_posterior(self, results: xr.Dataset) -> xr.DataArray:
        """Posterior over position given that activity is non-local.

        The non-local states are those whose observation model is neither
        Local nor No-Spike, wherever they are listed. See
        :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(
            results, self.state_names, self.observation_models
        )


class NonLocalSortedSpikesDetector(_NonLocalMixin, SortedSpikesDetector):
    """Detector of non-local activity, such as replay, from sorted spike data.

    A four-state model:

    - Local: activity represents the animal's current position.
    - No-Spike: a near-silent firing model without position.
    - Non-Local Continuous: the represented position follows a random walk.
    - Non-Local Fragmented: the represented position jumps uniformly over the
      track.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (4,), optional
        Initial probabilities of the four states, by default
        ``[1.0, 0.0, 0.0, 0.0]``.
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial position distribution of each state, by default uniform for
        all four.
    discrete_transition_type : DiscreteTransitions, optional
        Discrete transition model, by default ``DiscreteStationaryCustom`` with
        self-transition probabilities 0.99 (Local), 0.98 (No-Spike) and 0.96
        (each Non-Local state): expected dwell times of 200, 100 and 50 ms in
        2 ms bins. Local and the Non-Local states enter No-Spike with
        probability 0.005.
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.0.
    discrete_transition_stickiness : Stickiness, shape (4,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[100000.0, 100000.0, 300.0, 300.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``Discrete()`` into Local and No-Spike, ``RandomWalk()`` within
        Non-Local Continuous, and ``Uniform()`` otherwise.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``ObservationModel(is_local=True)``,
        ``ObservationModel(is_no_spike=True)`` and ``ObservationModel()`` for
        each Non-Local state. The states' roles are read from these flags.
    environments : Environments, optional
        Spatial environment, by default ``Environment(environment_name="")``.
    sorted_spikes_algorithm : str, optional
        Sorted-spikes likelihood algorithm, a key of
        ``_SORTED_SPIKES_ALGORITHMS``, by default "sorted_spikes_kde".
    sorted_spikes_algorithm_params : dict or None, optional
        Parameters for the sorted-spikes algorithm's fit. If ``None`` (the
        default), the algorithm's registered ``default_params`` are
        copied into a fresh dict at fit time; a user dict replaces those
        defaults. A key the fit does not accept raises ``ValidationError``.
        User-supplied dicts are stored as-is for sklearn ``clone()``
        compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer the track interior from the positions visited during
        fitting, by default True. Ignored for environments with a track graph.
    state_names : StateNames, optional
        Names of the states, by default ``["Local", "No-Spike",
        "Non-Local Continuous", "Non-Local Fragmented"]``.
    sampling_frequency : float, optional
        Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
        time range; ``predict`` and ``estimate_parameters`` use the
        ``time_edges`` they are given. Transition probabilities are per decode
        bin. By default 500.0.
    no_spike_rate : float, optional
        Firing rate in Hz of each neuron in a No-Spike state, by default
        1e-10.
    non_local_position_penalty : float, optional
        Magnitude, in nats, of the Gaussian penalty subtracted from the
        Non-Local states' log-likelihood near the animal's position. Suppresses
        non-local detection near the current location. By default 0.0 (no
        penalty). Must be >= 0.
    non_local_penalty_std : float, optional
        Standard deviation of the penalty Gaussian, in the units of
        ``position`` (typically cm), by default 1.0. Distances are shortest
        paths on the track graph. Must be > 0.
    discrete_transition_prior_weight : float or np.ndarray, optional
        Dimensionless weight for data-adaptive Dirichlet prior scaling during
        EM re-estimation of discrete transitions. When > 0, the prior
        pseudo-counts are scaled by expected transition counts; a scalar
        applies to every row, an array of shape ``(n_states,)`` per row. By
        default 0.0 (the fixed-count prior from concentration and
        stickiness).
    frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
        Discrete transition rows held at their initial values during EM
        re-estimation. By default ``"no_spike"``: the rows of the states whose
        observation model has ``is_no_spike=True``, wherever they are listed, so
        a No-Spike self-transition cannot collapse to fit stray gaps between
        spikes. Models without a No-Spike state freeze nothing. Row indices or
        a boolean mask of shape ``(n_states,)`` select rows by position;
        ``None`` freezes nothing.
    local_position_std : float or None, optional
        Spatial representation of the Local state:

        - ``None`` (default): Local occupies a single bin; its likelihood is
          evaluated at the animal's interpolated position.
        - ``0.0``: Local spans all position bins, with all mass at the bin
          containing the animal; the likelihood is evaluated at bin centers.
        - ``> 0``: Local spans all position bins with a Gaussian kernel of this
          standard deviation (units of ``position``, typically cm) on the
          shortest-path track-graph distance, modelling spatial uncertainty.

    Examples
    --------
    >>> from non_local_detector import (
    ...     NonLocalSortedSpikesDetector,
    ...     time_edges_from_centers,
    ... )
    >>> detector = NonLocalSortedSpikesDetector()
    >>> detector.fit(position_time, position, spike_times)
    >>> results = detector.predict(
    ...     spike_times,
    ...     time_edges=time_edges_from_centers(position_time),
    ...     position=position,
    ...     position_time=position_time,
    ... )
    >>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
    """

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 0.1,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict[str, Any] | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        non_local_position_penalty: float = 0.0,
        non_local_penalty_std: float = 1.0,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ):
        params = _initialize_params(
            _ModelDefaults.non_local_defaults(),
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=continuous_initial_conditions_types,
            discrete_transition_type=discrete_transition_type,
            discrete_transition_stickiness=discrete_transition_stickiness,
            observation_models=observation_models,
            environments=environments,
            state_names=state_names,
            continuous_transition_types=continuous_transition_types,
        )

        super().__init__(
            discrete_initial_conditions=params["discrete_initial_conditions"],
            continuous_initial_conditions_types=params[
                "continuous_initial_conditions_types"
            ],
            discrete_transition_type=params["discrete_transition_type"],
            discrete_transition_concentration=discrete_transition_concentration,
            discrete_transition_stickiness=params["discrete_transition_stickiness"],
            discrete_transition_regularization=discrete_transition_regularization,
            continuous_transition_types=params["continuous_transition_types"],
            observation_models=params["observation_models"],
            environments=params["environments"],
            sorted_spikes_algorithm=sorted_spikes_algorithm,
            sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
            infer_track_interior=infer_track_interior,
            state_names=params["state_names"],
            sampling_frequency=sampling_frequency,
            no_spike_rate=no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_std)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_std = non_local_penalty_std


class NonLocalClusterlessDetector(_NonLocalMixin, ClusterlessDetector):
    """Detector of non-local activity, such as replay, from clusterless spike data.

    A four-state model:

    - Local: activity represents the animal's current position.
    - No-Spike: a near-silent firing model without position.
    - Non-Local Continuous: the represented position follows a random walk.
    - Non-Local Fragmented: the represented position jumps uniformly over the
      track.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (4,), optional
        Initial probabilities of the four states, by default
        ``[1.0, 0.0, 0.0, 0.0]``.
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial position distribution of each state, by default uniform for
        all four.
    discrete_transition_type : DiscreteTransitions, optional
        Discrete transition model, by default ``DiscreteStationaryCustom`` with
        self-transition probabilities 0.99 (Local), 0.98 (No-Spike) and 0.96
        (each Non-Local state): expected dwell times of 200, 100 and 50 ms in
        2 ms bins. Local and the Non-Local states enter No-Spike with
        probability 0.005.
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.0.
    discrete_transition_stickiness : Stickiness, shape (4,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[100000.0, 100000.0, 300.0, 300.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``Discrete()`` into Local and No-Spike, ``RandomWalk()`` within
        Non-Local Continuous, and ``Uniform()`` otherwise.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``ObservationModel(is_local=True)``,
        ``ObservationModel(is_no_spike=True)`` and ``ObservationModel()`` for
        each Non-Local state. The states' roles are read from these flags.
    environments : Environments, optional
        Spatial environment, by default ``Environment(environment_name="")``.
    clusterless_algorithm : str, optional
        Clusterless likelihood algorithm, a key of ``_CLUSTERLESS_ALGORITHMS``,
        by default "clusterless_kde".
    clusterless_algorithm_params : dict or None, optional
        Parameters for the clusterless algorithm's fit. If ``None`` (the
        default), the algorithm's registered ``default_params`` are
        copied into a fresh dict at fit time; a user dict replaces those
        defaults. A key the fit does not accept raises ``ValidationError``.
        User-supplied dicts are stored as-is for sklearn ``clone()``
        compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer the track interior from the positions visited during
        fitting, by default True. Ignored for environments with a track graph.
    state_names : StateNames, optional
        Names of the states, by default ``["Local", "No-Spike",
        "Non-Local Continuous", "Non-Local Fragmented"]``.
    sampling_frequency : float, optional
        Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
        time range; ``predict`` and ``estimate_parameters`` use the
        ``time_edges`` they are given. Transition probabilities are per decode
        bin. By default 500.0.
    no_spike_rate : float, optional
        Firing rate in Hz of each electrode in a No-Spike state, by default
        1e-10. The marks of its spikes are scored with the training-fitted
        reference density ``mark_reference_``.
    non_local_position_penalty : float, optional
        Magnitude, in nats, of the Gaussian penalty subtracted from the
        Non-Local states' log-likelihood near the animal's position. Suppresses
        non-local detection near the current location. By default 0.0 (no
        penalty). Must be >= 0.
    non_local_penalty_std : float, optional
        Standard deviation of the penalty Gaussian, in the units of
        ``position`` (typically cm), by default 1.0. Distances are shortest
        paths on the track graph. Must be > 0.
    discrete_transition_prior_weight : float or np.ndarray, optional
        Dimensionless weight for data-adaptive Dirichlet prior scaling during
        EM re-estimation of discrete transitions. When > 0, the prior
        pseudo-counts are scaled by expected transition counts; a scalar
        applies to every row, an array of shape ``(n_states,)`` per row. By
        default 0.0 (the fixed-count prior from concentration and
        stickiness).
    frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
        Discrete transition rows held at their initial values during EM
        re-estimation. By default ``"no_spike"``: the rows of the states whose
        observation model has ``is_no_spike=True``, wherever they are listed, so
        a No-Spike self-transition cannot collapse to fit stray gaps between
        spikes. Models without a No-Spike state freeze nothing. Row indices or
        a boolean mask of shape ``(n_states,)`` select rows by position;
        ``None`` freezes nothing.
    local_position_std : float or None, optional
        Spatial representation of the Local state:

        - ``None`` (default): Local occupies a single bin; its likelihood is
          evaluated at the animal's interpolated position.
        - ``0.0``: Local spans all position bins, with all mass at the bin
          containing the animal; the likelihood is evaluated at bin centers.
        - ``> 0``: Local spans all position bins with a Gaussian kernel of this
          standard deviation (units of ``position``, typically cm) on the
          shortest-path track-graph distance, modelling spatial uncertainty.

    Examples
    --------
    >>> from non_local_detector import (
    ...     NonLocalClusterlessDetector,
    ...     time_edges_from_centers,
    ... )
    >>> detector = NonLocalClusterlessDetector()
    >>> detector.fit(position_time, position, spike_times, spike_waveform_features)
    >>> results = detector.predict(
    ...     spike_times, spike_waveform_features,
    ...     time_edges=time_edges_from_centers(position_time),
    ...     position=position,
    ...     position_time=position_time,
    ... )
    >>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
    """

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 0.1,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict[str, Any] | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        non_local_position_penalty: float = 0.0,
        non_local_penalty_std: float = 1.0,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ):
        params = _initialize_params(
            _ModelDefaults.non_local_defaults(),
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=continuous_initial_conditions_types,
            discrete_transition_type=discrete_transition_type,
            discrete_transition_stickiness=discrete_transition_stickiness,
            observation_models=observation_models,
            environments=environments,
            state_names=state_names,
            continuous_transition_types=continuous_transition_types,
        )

        super().__init__(
            discrete_initial_conditions=params["discrete_initial_conditions"],
            continuous_initial_conditions_types=params[
                "continuous_initial_conditions_types"
            ],
            discrete_transition_type=params["discrete_transition_type"],
            discrete_transition_concentration=discrete_transition_concentration,
            discrete_transition_stickiness=params["discrete_transition_stickiness"],
            discrete_transition_regularization=discrete_transition_regularization,
            continuous_transition_types=params["continuous_transition_types"],
            observation_models=params["observation_models"],
            environments=params["environments"],
            clusterless_algorithm=clusterless_algorithm,
            clusterless_algorithm_params=clusterless_algorithm_params,
            infer_track_interior=infer_track_interior,
            state_names=params["state_names"],
            sampling_frequency=sampling_frequency,
            no_spike_rate=no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_std)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_std = non_local_penalty_std
