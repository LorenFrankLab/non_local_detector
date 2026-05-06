"""Panel ABCs for the interactive viewer.

Two protocols, one per axis kind:

- ``TimeAxisPanel``: window-based panel rendered in the left column
  (posterior heatmap, raster, state probabilities, generic series).
- ``BinSyncedPanel``: point-based panel rendered to the right of the
  time-axis stack at the cursor's single time bin (SlicePanel; future
  ``VideoOverlayPanel``).

Concrete Qt implementations subclass the appropriate Protocol plus the
``EventOverlayMixin`` (Phase 3) for default overlay rendering.
Project-specific panels in ``continuum-swr-replay`` follow the same
pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        BinPayload,
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
    """Point-based panel rendered to the right of the time-axis stack."""

    def update_for_index(self, t_idx: int, payload: BinPayload) -> None:
        """Render the single-bin readout described by ``payload``."""
        ...
