"""Shared Qt mixins for the panel layer.

- ``EventOverlayMixin`` — default ``set_event_overlays`` against
  pyqtgraph primitives (``InfiniteLine`` / ``LinearRegionItem``).
- ``ClickRecenterMixin`` — wires ``scene().sigMouseClicked`` to a
  user-supplied ``Callable[[float], None]`` invoked with the clicked
  x-coordinate (absolute time in seconds).
- ``CursorMarkersMixin`` — dashed center-line at ``t_center`` + a
  translucent ``LinearRegionItem`` covering the active bin's
  ``[t_lo, t_hi]``, both updated together via ``set_cursor_markers``.

Concrete panels mix these in alongside ``pg.PlotWidget`` (or whatever
they wrap) so they get the shared behavior for free.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtGui import QColor


def bone_lookup_table() -> np.ndarray:
    """matplotlib `bone_r` analogue as a uint8 LUT — used by heatmap panels."""
    n = 256
    t = np.linspace(0, 1, n)
    # white → blue-grey → black
    r = (1.0 - t) * 0.875 + (1.0 - 0.875) * (1.0 - t)
    g = (1.0 - t) * 0.875 + (1.0 - 0.875) * (1.0 - t)
    b = 1.0 - t
    return np.stack([r * 255, g * 255, b * 255, np.full(n, 255.0)], axis=-1).astype(
        np.uint8
    )


if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )


class ClickRecenterMixin:
    """Wire ``scene().sigMouseClicked`` to a ``click_handler`` callback.

    Subclasses must call ``self._install_click_recenter()`` after
    ``pg.PlotWidget.__init__`` has run so ``self.scene()`` is
    available. The callback receives the clicked x-coordinate.
    """

    _click_callback: Callable[[float], None] | None

    def _install_click_recenter(self) -> None:
        self._click_callback = None
        self.scene().sigMouseClicked.connect(self._handle_click)
        # Disable pyqtgraph's right-click ViewBox menu. It builds a
        # ``ViewBoxMenu`` with sub-templates (``axisCtrlTemplate_generic``)
        # whose Qt ``QAction`` parents leak across QtViewer instances and
        # accumulate enough state to crash ``setupUi`` after ~14 viewers in
        # a single process. Removing the menu eliminates that leak path
        # without affecting the panel's primary interactions (click → scrub,
        # spike-click → pin) which are wired explicitly.
        self.getPlotItem().setMenuEnabled(False)

    def click_handler(self, callback: Callable[[float], None]) -> None:
        self._click_callback = callback

    def _handle_click(self, mouse_event) -> None:
        if self._click_callback is None:
            return
        # Skip recenter if a child item already handled the click
        # (e.g. ScatterPlotItem.sigClicked on a spike — that's a
        # cell-pin action, not a scrub-target).
        if mouse_event.isAccepted():
            return
        scene_pos = mouse_event.scenePos()
        view_pos = self.getPlotItem().vb.mapSceneToView(scene_pos)
        self._click_callback(float(view_pos.x()))

    def x_link_target(self):
        return self.getPlotItem()


# Z-values for cursor markers — both negative so any panel-specific
# overlay (event lines, non-local shading, etc.) renders on top.
# The band is one step further back so the dashed center line stays
# legible against the band's translucent yellow.
_CURSOR_BAND_Z = -6
_CURSOR_LINE_Z = -5


class CursorMarkersMixin:
    """Adds a dashed center line + translucent active-bin band.

    Subclasses must call ``self._install_cursor_markers()`` after
    ``pg.PlotWidget.__init__`` so ``self.addItem`` is available. The
    viewer's cursor-dispatch path then drives both items via
    ``set_cursor_markers(t_center, t_lo, t_hi)``.

    The band gives the user an unambiguous "this is the bin you're
    looking at" cue when zoomed in far enough that bin width is
    visible; the dashed line is a precise pointer at ``t_center``
    that stays useful at any zoom level.
    """

    _center_line: pg.InfiniteLine
    _active_bin_band: pg.LinearRegionItem

    def _install_cursor_markers(self) -> None:
        self._center_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen(
                (100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine
            ),
            movable=False,
        )
        self._center_line.setZValue(_CURSOR_LINE_Z)
        self.addItem(self._center_line)
        self._active_bin_band = pg.LinearRegionItem(
            values=(0.0, 0.0),
            orientation="vertical",
            brush=pg.mkBrush(255, 255, 0, 40),
            pen=pg.mkPen(None),
            movable=False,
        )
        self._active_bin_band.setZValue(_CURSOR_BAND_Z)
        self.addItem(self._active_bin_band)

    def set_cursor_markers(
        self, t_center: float, t_lo: float, t_hi: float
    ) -> None:
        """Move both markers in lockstep. Called by ``QtViewer``."""
        self._center_line.setPos(float(t_center))
        self._active_bin_band.setRegion([float(t_lo), float(t_hi)])


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
    """Build pyqtgraph items for one overlay; attach to ``plot_item``.

    ``overlay.times`` / ``t_start`` / ``t_end`` are NumPy arrays, so
    we explicitly check ``is None`` — truthy comparison
    (``arr or []``) would raise ``ValueError: ambiguous truth value``
    on multi-element arrays.
    """
    color = QColor(overlay.color)
    items: list[pg.GraphicsObject] = []
    if overlay.kind == "points":
        pen = pg.mkPen(color=color, width=1)
        times = overlay.times if overlay.times is not None else ()
        for t in times:
            line = pg.InfiniteLine(pos=float(t), angle=90, pen=pen, movable=False)
            plot_item.addItem(line)
            items.append(line)
    elif overlay.kind == "intervals":
        brush_color = QColor(color)
        brush_color.setAlphaF(overlay.alpha)
        t_start = overlay.t_start if overlay.t_start is not None else ()
        t_end = overlay.t_end if overlay.t_end is not None else ()
        for start, end in zip(t_start, t_end, strict=True):
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
