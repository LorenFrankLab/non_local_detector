"""Type definitions for the non-local neural detector package.

This module defines type aliases used throughout the package for improved
type hints and code clarity. These types represent common data structures
for environments, state transitions, observations, and model parameters
used in Hidden Markov Model-based neural decoding.
"""

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
"""Type alias for environment specifications.

Can be a single Environment instance, list of environments for multi-environment
models, or None for cases where no specific environment is needed.
"""

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
"""Type alias for continuous state transition specifications.

Nested list structure representing transition models between continuous
states. The outer list represents different discrete states, while the
inner list contains transition models for each state (e.g., RandomWalk,
Uniform, EmpiricalMovement, etc.).
"""

Observations = list[ObservationModel] | ObservationModel | None
"""Type alias for observation model specifications.

Can be a single ObservationModel, list of models for different conditions,
or None when observation models are not explicitly specified.
"""

ContinuousInitialConditions = list[UniformInitialConditions]
"""Type alias for continuous initial condition specifications.

List of initial condition models, typically UniformInitialConditions,
one for each discrete state in the HMM.
"""

Stickiness = float | np.ndarray
"""Type alias for discrete state transition stickiness parameters.

Can be a single float value applied to all states, or an array with
per-state stickiness values controlling the tendency to remain in
the current state.
"""

DiscreteTransitions = (
    DiscreteStationaryDiagonal
    | DiscreteNonStationaryCustom
    | DiscreteNonStationaryDiagonal
    | DiscreteStationaryCustom
)
"""Type alias for discrete state transition model types.

Union of all available discrete transition matrix types, including
stationary and non-stationary variants with diagonal or custom
transition patterns.
"""

StateNames = list[str] | None
"""Type alias for discrete state name specifications.

List of human-readable names for each discrete state in the HMM,
or None to use default state numbering.
"""
