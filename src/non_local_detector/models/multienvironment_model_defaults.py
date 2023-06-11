import numpy as np

from non_local_detector.continuous_state_transitions import (
    RandomWalk,
    Uniform,
)
from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel

_DEFAULT_ENVIRONMENT = Environment(environment_name="")

_DEFAULT_CONTINUOUS_TRANSITIONS = [
    [
        RandomWalk(environment_name="env1"),
        Uniform(environment_name="env1", environment_name2="env2"),
    ],
    [
        Uniform(environment_name="env2", environment_name2="env1"),
        RandomWalk(environment_name="env2"),
    ],
]

_DEFAULT_OBSERVATION_MODELS = [
    ObservationModel(environment_name="env1"),
    ObservationModel(environment_name="env2"),
]

_DEFAULT_DISCRETE_INITIAL_CONDITIONS = np.array([1.0, 0.0])
_DEFAULT_CONTINUOUS_INITIAL_CONDITIONS = [
    UniformInitialConditions(),
    UniformInitialConditions(),
]

_DEFAULT_DISCRETE_TRANSITION_STICKINESS = np.array([30.0, 30.0])

_DEFAULT_DISCRETE_TRANSITION_TYPE = DiscreteStationaryDiagonal(
    diagonal_values=np.array([0.999, 0.98])
)

_DEFAULT_STATE_NAMES = [
    "Environment1",
    "Environment2",
]

_DEFAULT_SORTED_SPIKES_MODEL_KWARGS = {
    "position_std": 6.0,
    "use_diffusion": False,
    "block_size": None,
}

_DEFAULT_CLUSTERLESS_MODEL_KWARGS = {
    "mark_std": 24.0,
    "position_std": 6.0,
}
