from non_local_detector.detector import (
    ClusterlessDetector,
    SortedSpikesDetector,
)  # noqa

try:
    from ._version import __version__
except ImportError:
    pass
