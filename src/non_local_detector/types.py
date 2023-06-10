import numpy as np

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel

env_types = Environment | list[Environment] | None
ct_types = list[
    list[
        Discrete
        | EmpiricalMovement
        | RandomWalk
        | RandomWalkDirection1
        | RandomWalkDirection2
        | Uniform
    ]
]
obs_types = list[ObservationModel] | None
cont_ic_types = list[UniformInitialConditions]
stickiness_types = float | np.ndarray
discrete_types = list[DiscreteStationaryDiagonal]
