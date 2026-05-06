"""``BackendAdapter`` Protocol — the seam between Qt-free core and frontend.

v1's Qt frontend implements this via ``QThreadPool`` + a ``_LoadSignals``
bridge object (statespacecheck pattern). v2's Panel/holoviews backend
will implement it via ``panel.io.unlocked()`` + a thread executor —
different plumbing, same interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        ViewState,
        WindowPayload,
    )


@runtime_checkable
class BackendAdapter(Protocol):
    """I/O seam the core uses to schedule work + marshal callbacks."""

    def schedule_window_load(
        self,
        state: ViewState,
        on_done: Callable[[WindowPayload], None],
    ) -> None:
        """Run a background load; call ``on_done`` on the UI thread when ready."""
        ...

    def post_to_ui_thread(self, fn: Callable[[], None]) -> None:
        """Marshal a callback onto the frontend's event loop."""
        ...
