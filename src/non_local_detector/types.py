from typing import Union

import numpy as np

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel

Environments = Union[Environment, list[Environment], None]
ContinuousTransitions = list[
    list[
        Union[
            Discrete,
            EmpiricalMovement,
            RandomWalk,
            RandomWalkDirection1,
            RandomWalkDirection2,
            Uniform,
        ]
    ]
]
Observations = Union[list[ObservationModel], None]
ContinuousInitialConditions = list[UniformInitialConditions]
Stickiness = Union[float, np.ndarray]
DiscreteTransitions = Union[
    DiscreteStationaryDiagonal,
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
]
StateNames = Union[list[str], None]
