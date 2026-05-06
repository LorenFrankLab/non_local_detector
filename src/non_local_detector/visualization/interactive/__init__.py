"""Interactive decoder viewer (PySide6 + pyqtgraph).

Public API surfaced incrementally per the v1 implementation plan
(``docs/plans/2026-05-06-interactive-decoder-viewer.md``):

- Phase 1b: ``RunBundle``, ``EventOverlay``, ``MetricSpec``,
  ``InMemoryDecoderDataSource``.
- Phase 1c: ``PosteriorHeatmapModel`` + panel ABCs.
- Phase 2+: Qt panels, ``DecoderViewer``, ``launch``.

Imports below are kept lightweight so ``import
non_local_detector.visualization.interactive`` does not pull GUI
dependencies. The ``[viewer]`` extra (``PySide6``, ``pyqtgraph``) is
required only for the Qt frontend, which lives in ``viewer/qt.py`` and
``panels/qt/``.
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
