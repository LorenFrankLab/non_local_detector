"""View-model dataclasses for the interactive viewer.

Pure data + transforms — no GUI dependencies. The Qt rendering layer
in ``panels/qt/`` consumes these.
"""

from non_local_detector.visualization.interactive.view_models.base import (
    BinPayload,
    CellSlice,
    PositionGrid,
    RunBundle,
    ViewState,
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)
from non_local_detector.visualization.interactive.view_models.posterior import (
    PosteriorHeatmapModel,
)
from non_local_detector.visualization.interactive.view_models.series import (
    MetricSpec,
)

__all__ = [
    "BinPayload",
    "CellSlice",
    "EventOverlay",
    "MetricSpec",
    "PositionGrid",
    "PosteriorHeatmapModel",
    "RunBundle",
    "ViewState",
    "WindowPayload",
]
