import numpy as np

from non_local_detector.continuous_state_transitions import (
    Discrete,
    RandomWalk,
    Uniform,
)
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
    Stickiness,
)

environment = Environment(environment_name="")

continuous_transition_types = [
    [Discrete(), Discrete(), Uniform(), Uniform()],
    [Discrete(), Discrete(), Uniform(), Uniform()],
    [Discrete(), Discrete(), RandomWalk(), Uniform()],
    [Discrete(), Discrete(), Uniform(), Uniform()],
]

observation_models = [
    ObservationModel(is_local=True),
    ObservationModel(is_no_spike=True),
    ObservationModel(),
    ObservationModel(),
]

discrete_initial_conditions = np.array([1.0, 0.0, 0.0, 0.0])
continuous_initial_conditions = [
    UniformInitialConditions(),
    UniformInitialConditions(),
    UniformInitialConditions(),
    UniformInitialConditions(),
]

discrete_transition_stickiness = np.array([30.0, 100_000.0, 30.0, 200.0])

discrete_transition_type = DiscreteStationaryDiagonal(
    diagonal_values=np.array([0.90, 0.90, 0.90, 0.98])
)

state_names = [
    "Local",
    "No-Spike",
    "Non-Local Continuous",
    "Non-Local Fragmented",
]


class NonLocalSortedSpikesDetector(SortedSpikesDetector):
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
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: list[str] | None = state_names,
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


class NonLocalClusterlessDetector(ClusterlessDetector):
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
        state_names: list[str] | None = state_names,
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
