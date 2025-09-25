from __future__ import annotations

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

Environments = Environment | list[Environment] | None
ContinuousTransitions = list[
    list[
        Discrete
        | EmpiricalMovement
        | RandomWalk
        | RandomWalkDirection1
        | RandomWalkDirection2
        | Uniform
    ]
]
Observations = list[ObservationModel] | ObservationModel | None
ContinuousInitialConditions = list[UniformInitialConditions]
Stickiness = float | np.ndarray
DiscreteTransitions = (
    DiscreteStationaryDiagonal
    | DiscreteNonStationaryCustom
    | DiscreteNonStationaryDiagonal
    | DiscreteStationaryCustom
)
StateNames = list[str] | None
