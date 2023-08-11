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

continuous_transition_types = [
    [RandomWalk()],
]

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
    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = discrete_initial_conditions,
        continuous_initial_conditions_types: ContinuousInitialConditions = continuous_initial_conditions,
        discrete_transition_type: DiscreteTransitions = discrete_transition_type,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness = discrete_transition_stickiness,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions = continuous_transition_types,
        observation_models: Observations = observation_models,
        environments: Environments = environment,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = state_names,
        sampling_frequency: float = 500,
        no_spike_rate: float = 1e-10,
    ):
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
    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = discrete_initial_conditions,
        continuous_initial_conditions_types: ContinuousInitialConditions = continuous_initial_conditions,
        discrete_transition_type: DiscreteTransitions = discrete_transition_type,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = discrete_transition_stickiness,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions = continuous_transition_types,
        observation_models: Observations = observation_models,
        environments: Environments = environment,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = state_names,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ):
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
