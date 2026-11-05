"""Type definitions for the non-local neural detector package.

This module defines type aliases used throughout the package for improved
type hints and code clarity. These types represent common data structures
for environments, state transitions, observations, and model parameters
used in Hidden Markov Model-based neural decoding.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

import numpy as np

from non_local_detector.continuous_state_transitions import ContinuousTransition
from non_local_detector.discrete_state_transitions import DiscreteTransition
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel

__all__ = [
    "ClusterlessAlgorithm",
    "ContinuousInitialConditions",
    "ContinuousTransitions",
    "DiscreteTransitions",
    "Environments",
    "FrozenTransitionRows",
    "InferenceMode",
    "Observations",
    "OutputMode",
    "OutputName",
    "ReturnOutputs",
    "SortedSpikesAlgorithm",
    "StateNames",
    "Stickiness",
    "TransitionRepresentation",
]


Environments = Environment | list[Environment] | None
"""Type alias for environment specifications.

Can be a single Environment instance, list of environments for multi-environment
models, or None for cases where no specific environment is needed.
"""

ContinuousTransitions = list[list[ContinuousTransition]]
"""Type alias for continuous state transition specifications.

Nested list structure representing transition models between continuous
states. The outer list represents the source discrete states and the inner
list the destination states. Each entry is any object satisfying the
``ContinuousTransition`` protocol (e.g., RandomWalk, Uniform,
EmpiricalMovement or a user-defined transition).
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

DiscreteTransitions = DiscreteTransition
"""Type alias for discrete state transition specifications.

Any object satisfying the ``DiscreteTransition`` protocol: the package's
stationary and covariate-dependent diagonal and custom transitions, or a
user-defined class with the same members.
"""

StateNames = list[str] | None
"""Type alias for discrete state name specifications.

List of human-readable names for each discrete state in the HMM,
or None to use default state numbering.
"""

FrozenTransitionRows = (
    Literal["no_spike"] | np.ndarray | list[int] | tuple[int, ...] | None
)
"""Type alias for the discrete transition rows held fixed during EM.

``"no_spike"`` selects the rows of the states whose observation model has
``is_no_spike=True``; integer indices or a boolean mask select rows by
position; ``None`` or an empty sequence freezes nothing.
"""

# ---------------------------------------------------------------------------
# String options of the detector methods
# ---------------------------------------------------------------------------
# These name the accepted strings for type checkers and readers; the methods
# still check the values at run time, since callers pass plain strings.

InferenceMode = Literal["dense", "checkpointed"]
"""Prediction driver: in-memory ``"dense"`` or on-disk ``"checkpointed"``."""

OutputMode = Literal["compact", "spatial"]
"""Checkpointed output: state probabilities only (``"compact"``) or also the
spatial posteriors (``"spatial"``). Dense prediction is always ``"spatial"``."""

TransitionRepresentation = Literal["dense", "structured", "auto"]
"""Continuous transition storage: the full matrix (``"dense"``), exact
state-block operators (``"structured"``), or operators with budgeted dense
fallback blocks (``"auto"``)."""

OutputName = Literal[
    "filter", "predictive", "predictive_posterior", "log_likelihood", "all"
]
"""One optional output of ``predict`` or ``estimate_parameters``.

``"filter"`` adds the causal posterior and state probabilities,
``"predictive"`` the one-step-ahead state probabilities,
``"predictive_posterior"`` the one-step-ahead posterior over state bins,
``"log_likelihood"`` the per-bin log likelihood, and ``"all"`` every one.
"""

ReturnOutputs = OutputName | Collection[OutputName] | None
"""A ``return_outputs`` value: one output name, any non-string collection of
them (list, tuple, set, frozenset), or None for the smoother only. A name
means the same alone or in a collection."""

SortedSpikesAlgorithm = Literal[
    "sorted_spikes_diffusion",
    "sorted_spikes_glm",
    "sorted_spikes_kde",
    "sorted_spikes_mrf",
]
"""Registered likelihood algorithm of a sorted-spikes detector
(``sorted_spikes_algorithm``)."""

ClusterlessAlgorithm = Literal[
    "clusterless_diffusion",
    "clusterless_gmm",
    "clusterless_kde",
    "clusterless_mark_glm",
]
"""Registered likelihood algorithm of a clusterless detector
(``clusterless_algorithm``)."""
