"""Internal utilities for model default parameters.

This module provides DRY parameter initialization without changing public APIs.
Each default factory creates fresh instances to avoid mutable default issues.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from non_local_detector.continuous_state_transitions import (
    Discrete,
    RandomWalk,
    Uniform,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel


def _resolve_param(value: Any, default_factory: Callable[[], Any]) -> Any:
    """Resolve a parameter: use provided value or create default.

    Parameters
    ----------
    value : Any
        The provided value (may be None)
    default_factory : Callable
        Factory function to create the default

    Returns
    -------
    Any
        The resolved parameter value
    """
    return default_factory() if value is None else value


class _ModelDefaults:
    """Factory for model default parameters.

    Each method returns a dictionary of factory functions that create fresh instances.
    Raymond Hettinger style: explicit factory methods over magic.
    """

    @staticmethod
    def decoder_defaults():
        """Defaults for single-state decoder models (sorted and clusterless)."""
        return {
            "discrete_initial_conditions": lambda: np.ones((1,)),
            "continuous_initial_conditions_types": lambda: [UniformInitialConditions()],
            "discrete_transition_type": lambda: DiscreteStationaryDiagonal(
                diagonal_values=np.array([1.0])
            ),
            "discrete_transition_stickiness": lambda: np.array([0.0]),
            "observation_models": lambda: [ObservationModel()],
            "environments": lambda: Environment(environment_name=""),
            "state_names": lambda: ["Continuous"],
            "continuous_transition_types": lambda: [[RandomWalk()]],
        }

    @staticmethod
    def cont_frag_defaults():
        """Defaults for continuous vs fragmented classifier (sorted and clusterless)."""
        return {
            "discrete_initial_conditions": lambda: np.ones((2,)) / 2,
            "continuous_initial_conditions_types": lambda: [
                UniformInitialConditions(),
                UniformInitialConditions(),
            ],
            "discrete_transition_type": lambda: DiscreteStationaryDiagonal(
                diagonal_values=np.array([0.98, 0.98])
            ),
            "discrete_transition_stickiness": lambda: np.array([0.0, 0.0]),
            "observation_models": lambda: [ObservationModel(), ObservationModel()],
            "environments": lambda: Environment(environment_name=""),
            "state_names": lambda: ["Continuous", "Fragmented"],
            "continuous_transition_types": lambda: [
                [RandomWalk(), Uniform()],
                [Uniform(), Uniform()],
            ],
        }

    @staticmethod
    def multi_environment_defaults():
        """Defaults for multi-environment classifier (sorted and clusterless)."""
        return {
            "discrete_initial_conditions": lambda: np.array([1.0, 0.0]),
            "continuous_initial_conditions_types": lambda: [
                UniformInitialConditions(),
                UniformInitialConditions(),
            ],
            "discrete_transition_type": lambda: DiscreteStationaryDiagonal(
                diagonal_values=np.array([0.999, 0.98])
            ),
            "discrete_transition_stickiness": lambda: np.array([200.0, 30.0]),
            "observation_models": lambda: [
                ObservationModel(environment_name="env1"),
                ObservationModel(environment_name="env2"),
            ],
            "environments": lambda: [
                Environment(environment_name="env1"),
                Environment(environment_name="env2"),
            ],
            "state_names": lambda: ["env1", "env2"],
            "continuous_transition_types": lambda: [
                [
                    RandomWalk(environment_name="env1"),
                    Uniform(environment_name="env1", environment2_name="env2"),
                ],
                [
                    Uniform(environment_name="env2", environment2_name="env1"),
                    RandomWalk(environment_name="env2"),
                ],
            ],
        }

    @staticmethod
    def non_local_defaults():
        """Defaults for non-local detector (4-state, sorted and clusterless)."""
        no_spike_trans_prob = 1e-5
        local_prob = 0.999
        cont_non_local_prob = 0.98
        non_local_frag_prob = 0.98
        no_spike_prob = 0.98

        return {
            "discrete_initial_conditions": lambda: np.array([1.0, 0.0, 0.0, 0.0]),
            "continuous_initial_conditions_types": lambda: [
                UniformInitialConditions() for _ in range(4)
            ],
            "discrete_transition_type": lambda: DiscreteStationaryCustom(
                values=np.array(
                    [
                        [
                            local_prob,
                            no_spike_trans_prob,
                            (1 - local_prob - no_spike_trans_prob) / 2,
                            (1 - local_prob - no_spike_trans_prob) / 2,
                        ],
                        [
                            (1 - no_spike_prob) / 3,
                            no_spike_prob,
                            (1 - no_spike_prob) / 3,
                            (1 - no_spike_prob) / 3,
                        ],
                        [
                            (1 - cont_non_local_prob - no_spike_trans_prob) / 2,
                            no_spike_trans_prob,
                            cont_non_local_prob,
                            (1 - cont_non_local_prob - no_spike_trans_prob) / 2,
                        ],
                        [
                            (1 - non_local_frag_prob - no_spike_trans_prob) / 2,
                            no_spike_trans_prob,
                            (1 - non_local_frag_prob - no_spike_trans_prob) / 2,
                            non_local_frag_prob,
                        ],
                    ]
                )
            ),
            "discrete_transition_stickiness": lambda: np.array(
                [1e6, 1e6, 300.0, 300.0]
            ),
            "observation_models": lambda: [
                ObservationModel(is_local=True),
                ObservationModel(is_no_spike=True),
                ObservationModel(),
                ObservationModel(),
            ],
            "environments": lambda: Environment(environment_name=""),
            "state_names": lambda: [
                "Local",
                "No-Spike",
                "Non-Local Continuous",
                "Non-Local Fragmented",
            ],
            "continuous_transition_types": lambda: [
                [Discrete(), Discrete(), Uniform(), Uniform()],
                [Discrete(), Discrete(), Uniform(), Uniform()],
                [Discrete(), Discrete(), RandomWalk(), Uniform()],
                [Discrete(), Discrete(), Uniform(), Uniform()],
            ],
        }

    @staticmethod
    def nospike_cont_frag_defaults():
        """Defaults for no-spike/continuous/fragmented classifier (sorted and clusterless)."""
        no_spike_trans_prob = 1e-5
        no_spike_prob = 0.99
        cont_prob = 0.9
        frag_prob = 0.98

        return {
            "discrete_initial_conditions": lambda: np.ones((3,)) / 3,
            "continuous_initial_conditions_types": lambda: [
                UniformInitialConditions() for _ in range(3)
            ],
            "discrete_transition_type": lambda: DiscreteStationaryCustom(
                values=np.array(
                    [
                        [
                            no_spike_prob,
                            (1 - no_spike_prob) / 2,
                            (1 - no_spike_prob) / 2,
                        ],
                        [
                            no_spike_trans_prob,
                            cont_prob,
                            (1 - cont_prob - no_spike_trans_prob),
                        ],
                        [
                            no_spike_trans_prob,
                            (1 - frag_prob - no_spike_trans_prob),
                            frag_prob,
                        ],
                    ]
                )
            ),
            "discrete_transition_stickiness": lambda: np.array(
                [100_000.0, 30.0, 200.0]
            ),
            "observation_models": lambda: [
                ObservationModel(is_no_spike=True),
                ObservationModel(),
                ObservationModel(),
            ],
            "environments": lambda: Environment(environment_name=""),
            "state_names": lambda: ["No-Spike", "Continuous", "Fragmented"],
            "continuous_transition_types": lambda: [
                [Discrete(), Uniform(), Uniform()],
                [Discrete(), RandomWalk(), Uniform()],
                [Discrete(), Uniform(), Uniform()],
            ],
        }


def _initialize_params(defaults_dict: dict[str, Callable], **provided_params) -> dict:
    """Initialize parameters using defaults and provided values.

    Parameters
    ----------
    defaults_dict : dict
        Dictionary mapping parameter names to factory functions
    **provided_params : dict
        User-provided parameter values (may contain None)

    Returns
    -------
    dict
        Resolved parameters ready for use
    """
    resolved = {}
    for param_name, factory in defaults_dict.items():
        provided_value = provided_params.get(param_name)
        resolved[param_name] = _resolve_param(provided_value, factory)
    return resolved
