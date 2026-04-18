import numpy as np
import xarray as xr

from non_local_detector.exceptions import ValidationError
from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
from non_local_detector.models.base import (
    _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
    _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    Observations,
    StateNames,
    Stickiness,
)


def _get_conditional_non_local_posterior(results: xr.Dataset) -> xr.DataArray:
    """Extract the conditional posterior probability for non-local states.

    Combines the posterior probabilities from both non-local continuous and
    non-local fragmented states (indices 2 and 3 in the model's state list),
    then normalizes across position.

    Parameters
    ----------
    results : xr.Dataset
        Results from a non-local detector's predict method containing
        acausal_posterior with a ``state`` coordinate.

    Returns
    -------
    conditional_posterior : xr.DataArray
        Normalized posterior probability for non-local activity across
        positions.  Shape is ``(n_time, n_position_bins)`` for 1-D
        environments and ``(n_time, n_x_bins, n_y_bins)`` for 2-D.

    Raises
    ------
    ValidationError
        If the results dataset contains fewer than 4 unique states, which
        indicates it did not come from a non-local detector.
    """
    state_names = results.acausal_posterior.coords["state"].values
    unique_states = list(dict.fromkeys(state_names))
    if len(unique_states) < 4:
        raise ValidationError(
            f"Expected at least 4 unique states in results, got {len(unique_states)}: "
            f"{unique_states}",
            expected="Results from a NonLocal detector with 4 states "
            "(e.g. Local, No-Spike, Non-Local Continuous, Non-Local Fragmented)",
            got=f"{len(unique_states)} states: {unique_states}",
            hint="This method should only be called on results from "
            "NonLocalSortedSpikesDetector or NonLocalClusterlessDetector",
        )
    non_local_continuous = unique_states[2]
    non_local_fragmented = unique_states[3]

    acausal_posterior = results.acausal_posterior.sel(
        state=non_local_continuous
    ) + results.acausal_posterior.sel(state=non_local_fragmented)

    # Sum over all position dimensions (works for 1D "position" and
    # 2D "x_position"/"y_position" alike)
    position_dims = [d for d in acausal_posterior.dims if d != "time"]
    return acausal_posterior / acausal_posterior.sum(position_dims)


def _validate_penalty_params(
    non_local_position_penalty: float, non_local_penalty_sigma: float
) -> None:
    """Validate non-local position penalty parameters."""
    if non_local_position_penalty < 0:
        raise ValidationError(
            "non_local_position_penalty must be >= 0",
            expected="non_local_position_penalty >= 0",
            got=f"{non_local_position_penalty}",
            hint="A negative penalty would reward non-local states near the animal's "
            "position instead of suppressing them. Use 0.0 to disable the penalty.",
        )
    if non_local_penalty_sigma <= 0:
        raise ValidationError(
            "non_local_penalty_sigma must be > 0",
            expected="non_local_penalty_sigma > 0",
            got=f"{non_local_penalty_sigma}",
            hint="Sigma controls the width of the Gaussian penalty and is used as a "
            "divisor. A value <= 0 would cause divide-by-zero or invalid behavior.",
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
        Stickiness parameters for each state, by default [1e6, 1e6, 300.0, 300.0].
    discrete_transition_regularization : float, optional
        Regularization parameter for discrete transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions, optional
        Types of continuous state transitions for each state, by default None.
    observation_models : Observations, optional
        Observation models for each state, by default includes local and no-spike models.
    environments : Environments, optional
        Environment specification, by default empty Environment.
    sorted_spikes_algorithm : str, optional
        Algorithm for sorted spikes likelihood, by default "sorted_spikes_kde".
    sorted_spikes_algorithm_params : dict, optional
        Parameters for the sorted spikes algorithm, by default default parameters.
    infer_track_interior : bool, optional
        Whether to infer track interior, by default True.
    state_names : StateNames, optional
        Names of the four states, by default ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"].
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate for no-spike observations, by default 1e-10.
    frozen_discrete_transition_rows : array-like or None, optional
        Discrete transition rows to freeze during EM re-estimation. By
        default ``(1,)``, which freezes the No-Spike row so its
        self-transition cannot collapse to fit stray spike gaps. Pass
        ``None`` to disable row freezing.

    Examples
    --------
    >>> detector = NonLocalSortedSpikesDetector()
    >>> detector.fit(position, spike_times, environments)
    >>> results = detector.predict(spike_times, time)
    >>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        non_local_position_penalty: float = 0.0,
        non_local_penalty_sigma: float = 1.0,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = (1,),
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
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_sigma)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_sigma = non_local_penalty_sigma

    @staticmethod
    def get_conditional_non_local_posterior(results: xr.Dataset) -> xr.DataArray:
        """Extract the conditional posterior probability for non-local states.

        See :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(results)


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
        Stickiness parameters for each state, by default [1e6, 1e6, 300.0, 300.0].
    discrete_transition_regularization : float, optional
        Regularization parameter for discrete transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions, optional
        Types of continuous state transitions for each state, by default None.
    observation_models : Observations, optional
        Observation models for each state, by default includes local and no-spike models.
    environments : Environments, optional
        Environment specification, by default empty Environment.
    clusterless_algorithm : str, optional
        Algorithm for clusterless likelihood computation, by default "clusterless_kde".
    clusterless_algorithm_params : dict, optional
        Parameters for the clusterless algorithm, by default default parameters.
    infer_track_interior : bool, optional
        Whether to infer track interior, by default True.
    state_names : StateNames, optional
        Names of the four states, by default ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"].
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate for no-spike observations, by default 1e-10.
    frozen_discrete_transition_rows : array-like or None, optional
        Discrete transition rows to freeze during EM re-estimation. By
        default ``(1,)``, which freezes the No-Spike row so its
        self-transition cannot collapse to fit stray spike gaps. Pass
        ``None`` to disable row freezing.

    Examples
    --------
    >>> detector = NonLocalClusterlessDetector()
    >>> detector.fit(position, spike_waveform_features, environments)
    >>> results = detector.predict(spike_waveform_features, time)
    >>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        non_local_position_penalty: float = 0.0,
        non_local_penalty_sigma: float = 1.0,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = (1,),
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
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_sigma)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_sigma = non_local_penalty_sigma

    @staticmethod
    def get_conditional_non_local_posterior(results: xr.Dataset) -> xr.DataArray:
        """Extract the conditional posterior probability for non-local states.

        See :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(results)
