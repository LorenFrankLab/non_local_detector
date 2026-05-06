"""Viewer harness: state + orchestration (Qt-free) + Qt window.

- ``core`` — ``ViewerCore`` (no Qt imports). Owns ``ViewState``,
  active-run state, pinned-event state, overlay state, the navigator,
  and the stale-result rejection rule.
- ``backend`` — ``BackendAdapter`` Protocol the core uses for I/O the
  frontend has to provide.
- ``qt`` — Qt-specific implementation (``[viewer]`` extra required).
  Owns ``QApplication`` creation, ``QtBackendAdapter``, ``QtViewer``.
"""

from non_local_detector.visualization.interactive.viewer.backend import (
    BackendAdapter,
)
from non_local_detector.visualization.interactive.viewer.core import ViewerCore

__all__ = ["BackendAdapter", "ViewerCore"]
