"""Shared Qt mixins for the panel layer.

``EventOverlayMixin`` provides the default ``set_event_overlays``
implementation against pyqtgraph primitives. Concrete panels mix it
in alongside ``pg.PlotWidget`` (or whatever they wrap) so they get
overlay support for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )


class EventOverlayMixin:
    """Default ``set_event_overlays`` for any panel wrapping a ``pg.PlotItem``.

    The implementation:

    - For ``EventOverlay.points(times=...)``: one ``pg.InfiniteLine``
      per ``times[i]``.
    - For ``EventOverlay.intervals(t_start=..., t_end=...)``: one
      ``pg.LinearRegionItem`` per ``(t_start[i], t_end[i])``.

    Idempotent: a new list fully replaces previously rendered markers.
    """

    _overlay_items: list[pg.GraphicsObject]  # populated lazily

    def set_event_overlays(self, overlays: list[EventOverlay]) -> None:
        plot_item = self._overlay_plot_item()
        # Remove the previously rendered overlay items.
        existing = getattr(self, "_overlay_items", None) or []
        for item in existing:
            plot_item.removeItem(item)
        new_items: list[pg.GraphicsObject] = []
        for overlay in overlays:
            new_items.extend(_render_overlay(plot_item, overlay))
        self._overlay_items = new_items

    def _overlay_plot_item(self) -> pg.PlotItem:
        """Return the ``pg.PlotItem`` overlay markers should attach to.

        Default: assumes the host class exposes ``getPlotItem()`` (e.g.
        ``pg.PlotWidget``). Subclasses with a non-standard layout
        override this.
        """
        get_plot = getattr(self, "getPlotItem", None)
        if get_plot is None:
            raise AttributeError(
                "EventOverlayMixin requires the host class to expose a "
                "`getPlotItem()` method (or override `_overlay_plot_item`)."
            )
        return get_plot()


def _render_overlay(
    plot_item: pg.PlotItem, overlay: EventOverlay
) -> list[pg.GraphicsObject]:
    """Build pyqtgraph items for one overlay; attach to ``plot_item``."""
    color = QColor(overlay.color)
    items: list[pg.GraphicsObject] = []
    if overlay.kind == "points":
        pen = pg.mkPen(color=color, width=1)
        for t in overlay.times or []:
            line = pg.InfiniteLine(pos=float(t), angle=90, pen=pen, movable=False)
            plot_item.addItem(line)
            items.append(line)
    elif overlay.kind == "intervals":
        brush_color = QColor(color)
        brush_color.setAlphaF(overlay.alpha)
        for start, end in zip(overlay.t_start or [], overlay.t_end or [], strict=True):
            region = pg.LinearRegionItem(
                values=(float(start), float(end)),
                movable=False,
                brush=brush_color,
            )
            plot_item.addItem(region)
            items.append(region)
    else:
        raise ValueError(
            f"Unknown overlay kind {overlay.kind!r}; expected 'points' or 'intervals'."
        )
    return items
