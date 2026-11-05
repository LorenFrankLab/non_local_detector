"""Decode position and detect non-local activity, such as replay, from spikes.

The models (detectors, classifiers and decoders) are in
:mod:`non_local_detector.models` and re-exported here, with the
:class:`Environment` and :class:`ObservationModel`, the transition and
initial-condition types, the time-bin helpers and the package's exceptions.
"""

import contextlib

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    Identity,
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
from non_local_detector.exceptions import (
    CompatibilityWarning,
    ConfigurationError,
    DataError,
    DataQualityWarning,
    NonLocalDetectorError,
    NonLocalDetectorWarning,
    NotFittedError,
    NumericalWarning,
    ParameterWarning,
    ValidationError,
)
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
    MultiEnvironmentClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    NoSpikeContFragClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.time_edges import (
    calculate_time_edges,
    time_edges_from_centers,
)

with contextlib.suppress(ImportError):
    from ._version import __version__

__all__ = [
    "ClusterlessDecoder",
    "CompatibilityWarning",
    "ConfigurationError",
    "ContFragClusterlessClassifier",
    "ContFragSortedSpikesClassifier",
    "DataError",
    "DataQualityWarning",
    "Discrete",
    "DiscreteNonStationaryCustom",
    "DiscreteNonStationaryDiagonal",
    "DiscreteStationaryCustom",
    "DiscreteStationaryDiagonal",
    "EmpiricalMovement",
    "Environment",
    "Identity",
    "MultiEnvironmentClusterlessClassifier",
    "MultiEnvironmentSortedSpikesClassifier",
    "NoSpikeContFragClusterlessClassifier",
    "NoSpikeContFragSortedSpikesClassifier",
    "NonLocalClusterlessDetector",
    "NonLocalDetectorError",
    "NonLocalDetectorWarning",
    "NonLocalSortedSpikesDetector",
    "NotFittedError",
    "NumericalWarning",
    "ObservationModel",
    "ParameterWarning",
    "RandomWalk",
    "RandomWalkDirection1",
    "RandomWalkDirection2",
    "SortedSpikesDecoder",
    "Uniform",
    "UniformInitialConditions",
    "ValidationError",
    "calculate_time_edges",
    "time_edges_from_centers",
]
