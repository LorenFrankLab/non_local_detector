"""View-model dataclasses for the interactive viewer.

Pure data + transforms — no GUI dependencies. The Qt rendering layer
in ``panels/qt/`` consumes these.
"""

from non_local_detector.visualization.interactive.view_models.base import (
    BinPayload,
    CellSlice,
    PositionGrid,
    RunBundle,
    SpikeEvent,
    SpikeEventIndex,
    ViewState,
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)
from non_local_detector.visualization.interactive.view_models.likelihood import (
    LikelihoodHeatmapModel,
)
from non_local_detector.visualization.interactive.view_models.posterior import (
    PosteriorHeatmapModel,
)
from non_local_detector.visualization.interactive.view_models.projected_1d import (
    Projected1DModel,
)
from non_local_detector.visualization.interactive.view_models.projected_2d import (
    Projected2DFrame,
    Projected2DGeometry,
    Projected2DModel,
)
from non_local_detector.visualization.interactive.view_models.raster import (
    RasterModel,
    RasterPayload,
)
from non_local_detector.visualization.interactive.view_models.series import (
    IntervalSeriesModel,
    LineSeriesModel,
    MetricSpec,
    MultiLineSeriesModel,
    ScatterSeriesModel,
)
from non_local_detector.visualization.interactive.view_models.state_prob import (
    StateProbabilityModel,
)

__all__ = [
    "BinPayload",
    "CellSlice",
    "EventOverlay",
    "IntervalSeriesModel",
    "LikelihoodHeatmapModel",
    "LineSeriesModel",
    "MetricSpec",
    "MultiLineSeriesModel",
    "PositionGrid",
    "PosteriorHeatmapModel",
    "Projected1DModel",
    "Projected2DFrame",
    "Projected2DGeometry",
    "Projected2DModel",
    "RasterModel",
    "RasterPayload",
    "RunBundle",
    "ScatterSeriesModel",
    "SpikeEvent",
    "SpikeEventIndex",
    "StateProbabilityModel",
    "ViewState",
    "WindowPayload",
]
