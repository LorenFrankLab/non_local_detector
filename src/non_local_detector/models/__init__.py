"""Ready-made detector, classifier and decoder models.

Each model fixes the discrete states and transitions of a ``SortedSpikesDetector``
or ``ClusterlessDetector`` (see :mod:`non_local_detector.models.base`): decoders
with a single continuous state, continuous-vs-fragmented classifiers,
multi-environment classifiers, and non-local detectors with Local, No-Spike and
non-local states.
"""

from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.models.decoder import ClusterlessDecoder, SortedSpikesDecoder
from non_local_detector.models.multienvironment_model import (
    MultiEnvironmentClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
)
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.models.nospike_cont_frag_model import (
    NoSpikeContFragClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
)

__all__ = [
    "ClusterlessDecoder",
    "ContFragClusterlessClassifier",
    "ContFragSortedSpikesClassifier",
    "MultiEnvironmentClusterlessClassifier",
    "MultiEnvironmentSortedSpikesClassifier",
    "NoSpikeContFragClusterlessClassifier",
    "NoSpikeContFragSortedSpikesClassifier",
    "NonLocalClusterlessDetector",
    "NonLocalSortedSpikesDetector",
    "SortedSpikesDecoder",
]
