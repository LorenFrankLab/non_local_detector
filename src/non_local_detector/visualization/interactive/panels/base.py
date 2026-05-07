"""Panel ABCs for the interactive viewer.

Two protocols, one per axis kind:

- ``TimeAxisPanel``: window-based panel rendered in the left column.
- ``BinSyncedPanel``: point-based panel rendered to the right of the
  time-axis stack at the cursor's single time bin.

Concrete Qt implementations subclass the appropriate Protocol plus
``EventOverlayMixin`` for default overlay rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )


@runtime_checkable
class TimeAxisPanel(Protocol):
    """Window-based panel rendered in the left column."""

    def update_window(self, payload: WindowPayload) -> None:
        """Render the time-axis window described by ``payload``."""
        ...

    def x_link_target(self) -> Any:
        """Return the Qt/Panel object that other panels should link x-axes to."""
        ...

    def click_handler(self, callback: Callable[[float], None]) -> None:
        """Register a callback invoked when the user clicks empty space.

        The callback receives the clicked x-coordinate (absolute time
        in seconds). Used by the viewer to recenter on click.
        """
        ...

    def set_event_overlays(self, overlays: list[EventOverlay]) -> None:
        """Replace this panel's event-overlay set.

        Called by ``ViewerCore`` whenever ``bundle.event_overlays`` or
        per-overlay visibility changes. Idempotent: a new list fully
        replaces previously rendered markers.
        """
        ...


@runtime_checkable
class BinSyncedPanel(Protocol):
    """Point-based panel rendered to the right of the time-axis stack.

    The viewer drives bin-synced plugins on a buffered low-latency
    path: each new ``WindowPayload`` is handed to every panel via
    ``set_window_buffer``, then per-tick cursor moves dispatch
    ``update_for_index(t_idx)`` synchronously so the per-tick render
    can index into the cached arrays without re-fetching from the
    data source. Plugins may optionally implement ``rebind_after_swap``
    to drop run-local caches when ``ViewerCore.set_active_run`` swaps
    the active run; the viewer calls it via ``getattr`` so it stays
    optional from a typing standpoint.
    """

    def set_window_buffer(self, payload: WindowPayload) -> None:
        """Cache the latest window payload for per-tick row reads."""
        ...

    def update_for_index(self, t_idx: int) -> None:
        """Render the cursor's bin from the cached buffer.

        Out-of-buffer ``t_idx`` should be a no-op — the next window
        load will refresh the buffer and the viewer will re-issue
        the cursor update.
        """
        ...
