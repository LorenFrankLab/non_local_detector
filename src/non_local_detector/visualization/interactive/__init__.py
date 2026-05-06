"""Interactive decoder viewer (PySide6 + pyqtgraph).

The top-level imports below are GUI-toolkit-free; the ``[viewer]``
extra (``PySide6``, ``pyqtgraph``) is required only for the Qt
frontend in ``viewer/qt.py`` and ``panels/qt/``.
"""

from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.view_models import (
    EventOverlay,
    MetricSpec,
    RunBundle,
)

__all__ = [
    "EventOverlay",
    "InMemoryDecoderDataSource",
    "MetricSpec",
    "RunBundle",
]
