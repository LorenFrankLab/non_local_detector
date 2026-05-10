"""PySide6 + pyqtgraph panel implementations.

Imports here pull GUI dependencies. Anything outside this package
must remain GUI-toolkit-free (CI gate enforces).
"""

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    EventOverlayMixin,
)
from non_local_detector.visualization.interactive.panels.qt.likelihood import (
    QtLikelihoodHeatmapPanel,
)
from non_local_detector.visualization.interactive.panels.qt.posterior import (
    QtPosteriorHeatmapPanel,
)
from non_local_detector.visualization.interactive.panels.qt.projected_2d import (
    QtProjected2DPanel,
)
from non_local_detector.visualization.interactive.panels.qt.raster import (
    QtRasterPanel,
)
from non_local_detector.visualization.interactive.panels.qt.series import (
    IntervalSeriesPanel,
    LineSeriesPanel,
    MultiLineSeriesPanel,
    ScatterSeriesPanel,
)
from non_local_detector.visualization.interactive.panels.qt.state_prob import (
    QtStateProbabilityPanel,
)

__all__ = [
    "ClickRecenterMixin",
    "EventOverlayMixin",
    "IntervalSeriesPanel",
    "LineSeriesPanel",
    "MultiLineSeriesPanel",
    "QtLikelihoodHeatmapPanel",
    "QtPosteriorHeatmapPanel",
    "QtProjected2DPanel",
    "QtRasterPanel",
    "QtStateProbabilityPanel",
    "ScatterSeriesPanel",
]
