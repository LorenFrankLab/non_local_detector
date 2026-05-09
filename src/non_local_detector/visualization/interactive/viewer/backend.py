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

    def build_payload(self, state: ViewState) -> WindowPayload:
        """Build a ``WindowPayload`` synchronously for ``state``.

        Synchronous-entry contract: callers (notably tests + the
        slice panel's bin-buffer hot path) hand a ``ViewState`` in,
        get back a fully-resolved ``WindowPayload`` without going
        through the executor / debounce path. Any frontend that
        implements ``BackendAdapter`` must support this — it's the
        only direct way to materialise a payload.
        """
        ...

    def set_required_outputs(self, outputs: set[str]) -> None:
        """Declare which optional ``WindowPayload`` fields the panels
        actually use, so the backend can skip loading the others.

        Required, NOT a default no-op: a default no-op would silently
        swallow output gating for any future backend that forgot to
        override, exactly the drift the Protocol cleanup is meant to
        prevent.
        """
        ...
