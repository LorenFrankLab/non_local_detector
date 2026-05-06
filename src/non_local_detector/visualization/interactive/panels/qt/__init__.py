"""PySide6 + pyqtgraph panel implementations.

Imports here pull GUI dependencies. Anything outside this package
must remain GUI-toolkit-free (CI gate enforces).
"""

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    EventOverlayMixin,
)
from non_local_detector.visualization.interactive.panels.qt.posterior import (
    QtPosteriorHeatmapPanel,
)

__all__ = ["EventOverlayMixin", "QtPosteriorHeatmapPanel"]
