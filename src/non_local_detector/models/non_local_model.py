import numpy as np
import xarray as xr

from non_local_detector.exceptions import ValidationError
from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
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


def _get_conditional_non_local_posterior(
    results: xr.Dataset,
    state_names: list[str],
    observation_models: list[ObservationModel],
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
        raise ValidationError(
            "The detector has no non-local states",
            expected="at least one observation model with is_local=False "
            "and is_no_spike=False",
            got=f"states {list(state_names)}",
        )
    result_states = set(results.acausal_posterior.coords["state"].values)
    missing = [name for name in non_local_states if name not in result_states]
    if missing:
        raise ValidationError(
            f"results lack the detector's non-local states {missing}",
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
    return acausal_posterior / acausal_posterior.sum(position_dims)


def _validate_penalty_params(
    non_local_position_penalty: float, non_local_penalty_std: float
) -> None:
    """Validate non-local position penalty parameters.

    Parameters
    ----------
    non_local_position_penalty : float
        Magnitude of the Gaussian penalty. Must be non-negative.
    non_local_penalty_std : float
        Standard deviation of the penalty Gaussian in position units.
        Must be strictly positive.

    Raises
    ------
    ValidationError
        If ``non_local_position_penalty < 0`` or
        ``non_local_penalty_std <= 0``.
    """
    if non_local_position_penalty < 0:
        raise ValidationError(
            "non_local_position_penalty must be >= 0",
            expected="non_local_position_penalty >= 0",
            got=f"{non_local_position_penalty}",
            hint="A negative penalty would reward non-local states near the animal's "
            "position instead of suppressing them. Use 0.0 to disable the penalty.",
        )
    if non_local_penalty_std <= 0:
        raise ValidationError(
            "non_local_penalty_std must be > 0",
            expected="non_local_penalty_std > 0",
            got=f"{non_local_penalty_std}",
            hint="Standard deviation controls the width of the Gaussian penalty "
            "and is used as a divisor. A value <= 0 would cause divide-by-zero "
            "or invalid behavior.",
        )


class NonLocalSortedSpikesDetector(SortedSpikesDetector):
    """Detector for non-local neural activity using sorted spike data.

    This class implements a Hidden Markov Model to detect neural replay events
    and distinguish between local activity, no-spike periods, continuous non-local
    activity, and fragmented non-local activity.

    The model uses four states:
    - Local: Neural activity corresponding to the animal's current location
    - No-Spike: Periods with no significant neural activity
    - Non-Local Continuous: Replay events with continuous position decoding
    - Non-Local Fragmented: Replay events with fragmented/discrete position jumps

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (4,), optional
        Initial probabilities for the four states, by default [1.0, 0.0, 0.0, 0.0].
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Types of continuous initial conditions for each state, by default uniform.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transition matrix, by default DiscreteStationaryCustom.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.0.
    discrete_transition_stickiness : Stickiness, optional
        Stickiness parameters for each state, by default [1e5, 1e5, 300.0, 300.0].
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Types of continuous state transitions for each state, by default None.
    observation_models : Observations, optional
        Observation models for each state, by default includes local and no-spike models.
    environments : Environments, optional
        Environment specification, by default empty Environment.
    sorted_spikes_algorithm : str, optional
        Algorithm for sorted spikes likelihood, by default "sorted_spikes_kde".
    sorted_spikes_algorithm_params : dict or None, optional
        Parameters for the sorted spikes algorithm. If ``None`` (the
        default), the module-level
        ``_DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS`` values are copied
        into a fresh dict at fit time. User-supplied dicts are stored
        as-is for sklearn ``clone()`` compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer track interior, by default True.
    state_names : StateNames, optional
        Names of the four states, by default ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"].
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate for no-spike observations, by default 1e-10.
    non_local_position_penalty : float, optional
        Magnitude of the Gaussian penalty subtracted from the non-local
        states' log-likelihood near the animal's position. Suppresses
        non-local replay detection near the current location. By default
        ``0.0`` (no penalty).
    non_local_penalty_std : float, optional
        Standard deviation of the Gaussian penalty, in the same units as
        ``position`` (typically centimeters). Controls the spatial
        extent of the suppression zone around the animal; distances use
        shortest-path along the track graph. Must be strictly positive.
        By default ``1.0``.
    discrete_transition_prior_weight : float or np.ndarray, optional
        Dimensionless weight for data-adaptive Dirichlet prior scaling
        during EM re-estimation of discrete transitions. By default
        ``0.0`` (fixed-count prior).
    frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
        Discrete transition rows to freeze during EM re-estimation. By
        default ``"no_spike"``, which freezes the No-Spike state's row
        (found from ``observation_models``, wherever it is listed) so its
        self-transition cannot collapse to fit stray spike gaps. Row
        indices or a boolean mask select rows by position; ``None``
        disables row freezing.
    local_position_std : float or None, optional
        Controls the Local state's spatial representation:

        - ``None`` (default): legacy behavior. Local state occupies a
          single bin; the likelihood is evaluated at the animal's exact
          interpolated position (continuous).
        - ``0.0``: delta-kernel multi-bin local. Local state spans all
          position bins, with the kernel concentrated at the single bin
          containing the animal. The per-bin likelihood is evaluated at
          bin centers (discrete); only the animal's bin contributes
          finite mass, the rest are suppressed.
        - ``> 0``: multi-bin local with a Gaussian kernel of standard
          deviation ``local_position_std`` (same units as ``position``,
          typically centimeters) on the shortest-path track-graph
          distance. Models spatial uncertainty in the local
          representation.

        All three modes produce a valid posterior.

    Attributes
    ----------
    non_local_position_penalty : float
        Stored copy of the ``non_local_position_penalty`` constructor
        argument.
    non_local_penalty_std : float
        Stored copy of the ``non_local_penalty_std`` constructor argument.
    local_position_std : float or None
        Stored copy of the ``local_position_std`` constructor argument.

    Examples
    --------
    >>> detector = NonLocalSortedSpikesDetector()
    >>> detector.fit(position_time, position, spike_times)
    >>> results = detector.predict(
    ...     spike_times,
    ...     time_edges=time_edges,
    ...     position=position,
    ...     position_time=position_time,
    ... )
    >>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
    """

    def __init__(
        self,
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
        sorted_spikes_algorithm_params: dict | None = None,
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
            params["discrete_initial_conditions"],
            params["continuous_initial_conditions_types"],
            params["discrete_transition_type"],
            discrete_transition_concentration,
            params["discrete_transition_stickiness"],
            discrete_transition_regularization,
            params["continuous_transition_types"],
            params["observation_models"],
            params["environments"],
            sorted_spikes_algorithm,
            sorted_spikes_algorithm_params,
            infer_track_interior,
            params["state_names"],
            sampling_frequency,
            no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_std)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_std = non_local_penalty_std

    def get_conditional_non_local_posterior(self, results: xr.Dataset) -> xr.DataArray:
        """Posterior over position given that activity is non-local.

        The non-local states are those whose observation model is neither
        Local nor No-Spike, wherever they are listed. See
        :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(
            results, self.state_names, self.observation_models
        )


class NonLocalClusterlessDetector(ClusterlessDetector):
    """Detector for non-local neural activity using clusterless spike data.

    This class implements a Hidden Markov Model to detect neural replay events
    from clusterless (continuous) spike features, distinguishing between local
    activity, no-spike periods, continuous non-local activity, and fragmented
    non-local activity.

    The model uses four states:
    - Local: Neural activity corresponding to the animal's current location
    - No-Spike: Periods with no significant neural activity
    - Non-Local Continuous: Replay events with continuous position decoding
    - Non-Local Fragmented: Replay events with fragmented/discrete position jumps

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (4,), optional
        Initial probabilities for the four states, by default [1.0, 0.0, 0.0, 0.0].
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Types of continuous initial conditions for each state, by default uniform.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transition matrix, by default DiscreteStationaryCustom.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.0.
    discrete_transition_stickiness : Stickiness, optional
        Stickiness parameters for each state, by default [1e5, 1e5, 300.0, 300.0].
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Types of continuous state transitions for each state, by default None.
    observation_models : Observations, optional
        Observation models for each state, by default includes local and no-spike models.
    environments : Environments, optional
        Environment specification, by default empty Environment.
    clusterless_algorithm : str, optional
        Algorithm for clusterless likelihood computation, by default "clusterless_kde".
    clusterless_algorithm_params : dict or None, optional
        Parameters for the clusterless algorithm. If ``None`` (the
        default), the module-level
        ``_DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS`` values are copied
        into a fresh dict at fit time. User-supplied dicts are stored
        as-is for sklearn ``clone()`` compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer track interior, by default True.
    state_names : StateNames, optional
        Names of the four states, by default ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"].
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate for no-spike observations, by default 1e-10.
    non_local_position_penalty : float, optional
        Magnitude of the Gaussian penalty subtracted from the non-local
        states' log-likelihood near the animal's position. Suppresses
        non-local replay detection near the current location. By default
        ``0.0`` (no penalty).
    non_local_penalty_std : float, optional
        Standard deviation of the Gaussian penalty, in the same units as
        ``position`` (typically centimeters). Controls the spatial
        extent of the suppression zone around the animal; distances use
        shortest-path along the track graph. Must be strictly positive.
        By default ``1.0``.
    discrete_transition_prior_weight : float or np.ndarray, optional
        Dimensionless weight for data-adaptive Dirichlet prior scaling
        during EM re-estimation of discrete transitions. By default
        ``0.0`` (fixed-count prior).
    frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
        Discrete transition rows to freeze during EM re-estimation. By
        default ``"no_spike"``, which freezes the No-Spike state's row
        (found from ``observation_models``, wherever it is listed) so its
        self-transition cannot collapse to fit stray spike gaps. Row
        indices or a boolean mask select rows by position; ``None``
        disables row freezing.
    local_position_std : float or None, optional
        Controls the Local state's spatial representation:

        - ``None`` (default): legacy behavior. Local state occupies a
          single bin; the likelihood is evaluated at the animal's exact
          interpolated position (continuous).
        - ``0.0``: delta-kernel multi-bin local. Local state spans all
          position bins, with the kernel concentrated at the single bin
          containing the animal. The per-bin likelihood is evaluated at
          bin centers (discrete); only the animal's bin contributes
          finite mass, the rest are suppressed.
        - ``> 0``: multi-bin local with a Gaussian kernel of standard
          deviation ``local_position_std`` (same units as ``position``,
          typically centimeters) on the shortest-path track-graph
          distance. Models spatial uncertainty in the local
          representation.

        All three modes produce a valid posterior.

    Attributes
    ----------
    non_local_position_penalty : float
        Stored copy of the ``non_local_position_penalty`` constructor
        argument.
    non_local_penalty_std : float
        Stored copy of the ``non_local_penalty_std`` constructor argument.
    local_position_std : float or None
        Stored copy of the ``local_position_std`` constructor argument.

    Examples
    --------
    >>> detector = NonLocalClusterlessDetector()
    >>> detector.fit(position_time, position, spike_times, spike_waveform_features)
    >>> results = detector.predict(
    ...     spike_times,
    ...     spike_waveform_features,
    ...     time_edges=time_edges,
    ...     position=position,
    ...     position_time=position_time,
    ... )
    >>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
    """

    def __init__(
        self,
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
        clusterless_algorithm_params: dict | None = None,
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
            params["discrete_initial_conditions"],
            params["continuous_initial_conditions_types"],
            params["discrete_transition_type"],
            discrete_transition_concentration,
            params["discrete_transition_stickiness"],
            discrete_transition_regularization,
            params["continuous_transition_types"],
            params["observation_models"],
            params["environments"],
            clusterless_algorithm,
            clusterless_algorithm_params,
            infer_track_interior,
            params["state_names"],
            sampling_frequency,
            no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_std)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_std = non_local_penalty_std

    def get_conditional_non_local_posterior(self, results: xr.Dataset) -> xr.DataArray:
        """Posterior over position given that activity is non-local.

        The non-local states are those whose observation model is neither
        Local nor No-Spike, wherever they are listed. See
        :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(
            results, self.state_names, self.observation_models
        )
