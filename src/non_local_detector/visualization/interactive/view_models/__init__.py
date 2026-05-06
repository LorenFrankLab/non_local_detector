"""View-model dataclasses for the interactive viewer.

Pure data + transforms — no GUI dependencies. The Qt rendering layer
in ``panels/qt/`` consumes these.
"""

from non_local_detector.visualization.interactive.view_models.base import (
    RunBundle,
)
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)
from non_local_detector.visualization.interactive.view_models.series import (
    MetricSpec,
)

__all__ = ["EventOverlay", "MetricSpec", "RunBundle"]
