import numpy as np

from non_local_detector.continuous_state_transitions import RandomWalk
from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models.base import (
    _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
    _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    Observations,
    StateNames,
    Stickiness,
)

environment = Environment(environment_name="")

observation_models = [
    ObservationModel(),
]

discrete_initial_conditions = np.ones((1,))
continuous_initial_conditions = [
    UniformInitialConditions(),
]

discrete_transition_stickiness = np.array([0.0])

discrete_transition_type = DiscreteStationaryDiagonal(diagonal_values=np.array([1.0]))

state_names = [
    "Continuous",
]


class SortedSpikesDecoder(SortedSpikesDetector):
    """Decoder for sorted spike data using a single continuous state.

    This class provides a simplified interface for decoding position from sorted
    spike data using Hidden Markov Models. It uses a single continuous state with
    uniform initial conditions and configurable transition models.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (1,), optional
        Initial probabilities for discrete states, by default np.ones((1,)).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Types of continuous initial conditions, by default [UniformInitialConditions()].
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transition, by default DiscreteStationaryDiagonal.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.0.
    discrete_transition_stickiness : Stickiness, optional
        Stickiness parameter for discrete transitions, by default np.array([0.0]).
    discrete_transition_regularization : float, optional
        Regularization parameter for discrete transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions, optional
        Types of continuous state transitions, by default None (uses RandomWalk).
    observation_models : Observations, optional
        Observation models for likelihood computation, by default [ObservationModel()].
    environments : Environments, optional
        Environment specification, by default Environment(environment_name="").
    sorted_spikes_algorithm : str, optional
        Algorithm for sorted spikes likelihood, by default "sorted_spikes_kde".
    sorted_spikes_algorithm_params : dict, optional
        Parameters for the sorted spikes algorithm, by default _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS.
    infer_track_interior : bool, optional
        Whether to infer track interior, by default True.
    state_names : StateNames, optional
        Names of the states, by default ["Continuous"].
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 500.
    no_spike_rate : float, optional
        Rate for no-spike observations, by default 1e-10.

    Examples
    --------
    >>> decoder = SortedSpikesDecoder()
    >>> decoder.fit(position, spike_times, environments)
    >>> decoded_position = decoder.predict(spike_times, time)
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = discrete_initial_conditions,
        continuous_initial_conditions_types: ContinuousInitialConditions = continuous_initial_conditions,
        discrete_transition_type: DiscreteTransitions = discrete_transition_type,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness = discrete_transition_stickiness,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations = observation_models,
        environments: Environments = environment,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = state_names,
        sampling_frequency: float = 500,
        no_spike_rate: float = 1e-10,
    ):
        if continuous_transition_types is None:
            continuous_transition_types = [
                [RandomWalk()],
            ]
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            sorted_spikes_algorithm,
            sorted_spikes_algorithm_params,
            infer_track_interior,
            state_names,
            sampling_frequency,
            no_spike_rate,
        )


class ClusterlessDecoder(ClusterlessDetector):
    """Decoder for clusterless spike data using a single continuous state.

    This class provides a simplified interface for decoding position from clusterless
    (continuous) spike data using Hidden Markov Models. It uses a single continuous
    state with uniform initial conditions and configurable transition models.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (1,), optional
        Initial probabilities for discrete states, by default np.ones((1,)).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Types of continuous initial conditions, by default [UniformInitialConditions()].
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transition, by default DiscreteStationaryDiagonal.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, optional
        Stickiness parameter for discrete transitions, by default np.array([0.0]).
    discrete_transition_regularization : float, optional
        Regularization parameter for discrete transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions, optional
        Types of continuous state transitions, by default None (uses RandomWalk).
    observation_models : Observations, optional
        Observation models for likelihood computation, by default [ObservationModel()].
    environments : Environments, optional
        Environment specification, by default Environment(environment_name="").
    clusterless_algorithm : str, optional
        Algorithm for clusterless likelihood computation, by default "clusterless_kde".
    clusterless_algorithm_params : dict, optional
        Parameters for the clusterless algorithm, by default _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS.
    infer_track_interior : bool, optional
        Whether to infer track interior, by default True.
    state_names : StateNames, optional
        Names of the states, by default ["Continuous"].
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate for no-spike observations, by default 1e-10.

    Examples
    --------
    >>> decoder = ClusterlessDecoder()
    >>> decoder.fit(position, spike_waveform_features, environments)
    >>> decoded_position = decoder.predict(spike_waveform_features, time)
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = discrete_initial_conditions,
        continuous_initial_conditions_types: ContinuousInitialConditions = continuous_initial_conditions,
        discrete_transition_type: DiscreteTransitions = discrete_transition_type,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = discrete_transition_stickiness,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations = observation_models,
        environments: Environments = environment,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = state_names,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ):
        if continuous_transition_types is None:
            continuous_transition_types = [
                [RandomWalk()],
            ]
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            clusterless_algorithm,
            clusterless_algorithm_params,
            infer_track_interior,
            state_names,
            sampling_frequency,
            no_spike_rate,
        )
