"""Generic series panels: line, multi-line, scatter, interval.

Each panel consumes the matching view-model in
``view_models/series.py`` and provides one Qt rendering. The four
panels together cover every panel kind in the static
``plot_detector`` figure (see the plan's
"Coverage of `plot_detector`" table).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtGui import QColor

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
    RelativeTimeAxisMixin,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.series import (
        IntervalSeriesModel,
        LineSeriesModel,
        MultiLineSeriesModel,
        ScatterSeriesModel,
    )


# ---------------------------------------------------------------------------
# LineSeriesPanel
# ---------------------------------------------------------------------------


class LineSeriesPanel(
    pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin, RelativeTimeAxisMixin
):
    """Single line, optionally with fill-below shading + threshold lines."""

    def __init__(
        self,
        model: LineSeriesModel,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self.setLabel("left", model.name)
        self.setLabel("bottom", "Time [s]")
        if model.y_range is not None:
            self.setYRange(*model.y_range)
        pen = pg.mkPen(color=QColor(model.color), width=3)
        self._line = self.plot([], [], pen=pen, name=model.name)
        self._fill_curve: pg.FillBetweenItem | None = None
        if model.fill_below:
            zero = self.plot([], [], pen=None)
            fill_color = QColor(model.color)
            fill_color.setAlphaF(0.3)
            self._fill_curve = pg.FillBetweenItem(
                self._line, zero, brush=pg.mkBrush(fill_color)
            )
            self.addItem(self._fill_curve)
            self._zero_curve = zero
        for threshold in model.thresholds:
            line = pg.InfiniteLine(
                pos=float(threshold),
                angle=0,
                pen=pg.mkPen("k", style=pg.QtCore.Qt.DashLine, width=2),
            )
            self.addItem(line)
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            self._line.setData([], [])
            return
        t_start, t_stop = float(payload.time[0]), float(payload.time[-1])
        t, y = self._model.window(t_start, t_stop)
        # Render at relative coords against the panel's fixed x-range.
        t_rel = t - payload.t_center
        self._line.setData(t_rel, y)
        if self._fill_curve is not None:
            self._zero_curve.setData(t_rel, [0.0] * len(t_rel))


# ---------------------------------------------------------------------------
# MultiLineSeriesPanel
# ---------------------------------------------------------------------------


_DEFAULT_MULTI_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)


def _legend_html(items: list[tuple[str, str]]) -> str:
    """Inline legend rendered into the plot's ``setTitle`` slot."""
    return " &nbsp;&nbsp;&nbsp; ".join(
        f"<span style='color:{color};font-size:11pt'>━</span> {label}"
        for color, label in items
    )


class MultiLineSeriesPanel(
    pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin, RelativeTimeAxisMixin
):
    """Several lines on one panel sharing a common time axis + y-range.

    The legend is rendered into the plot's title bar (HTML-coloured
    line markers + line labels) instead of pyqtgraph's floating
    ``LegendItem`` so it never occludes the curves.
    """

    def __init__(
        self,
        model: MultiLineSeriesModel,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self.setLabel("left", model.name)
        self.setLabel("bottom", "Time [s]")
        if model.y_range is not None:
            self.setYRange(*model.y_range)
        self._lines: dict[str, pg.PlotDataItem] = {}
        legend_items: list[tuple[str, str]] = []
        for i, label in enumerate(model.ys):
            color = (
                model.colors[label]
                if model.colors and label in model.colors
                else _DEFAULT_MULTI_COLORS[i % len(_DEFAULT_MULTI_COLORS)]
            )
            self._lines[label] = self.plot(
                [], [], pen=pg.mkPen(color=color, width=3), name=label
            )
            legend_items.append((color, label))
        self.setTitle(_legend_html(legend_items))
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            for line in self._lines.values():
                line.setData([], [])
            return
        t_start, t_stop = float(payload.time[0]), float(payload.time[-1])
        t, ys = self._model.window(t_start, t_stop)
        # Render at relative coords against the panel's fixed x-range.
        t_rel = t - payload.t_center
        for label, line in self._lines.items():
            line.setData(t_rel, ys[label])


# ---------------------------------------------------------------------------
# ScatterSeriesPanel
# ---------------------------------------------------------------------------


class ScatterSeriesPanel(
    pg.PlotWidget,
    EventOverlayMixin,
    ClickRecenterMixin,
    CursorMarkersMixin,
    RelativeTimeAxisMixin,
):
    """Scatter of ``(t, y)`` points; click-on-point recenters."""

    def __init__(
        self,
        model: ScatterSeriesModel,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self.setLabel("left", model.name)
        self.setLabel("bottom", "Time [s]")
        if model.y_range is not None:
            self.setYRange(*model.y_range)
        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(QColor(model.color), width=2),
            brush=pg.mkBrush(model.color),
            size=8,
        )
        self.addItem(self._scatter)
        self._install_click_recenter()
        self._install_cursor_markers()
        self._overlay_items: list[pg.GraphicsObject] = []
        if model.click_recenters:
            self._scatter.sigClicked.connect(self._on_point_clicked)

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            self._scatter.setData([], [])
            return
        t_start, t_stop = float(payload.time[0]), float(payload.time[-1])
        t, y = self._model.window(t_start, t_stop)
        # Render at relative coords against the panel's fixed x-range.
        self._scatter.setData(t - payload.t_center, y)

    def _on_point_clicked(self, _scatter, points) -> None:
        if not points:
            return
        # Recenter on the first clicked point's x coordinate. The
        # scatter is now drawn in relative coords, so the click x is
        # already relative to ``t_center`` — the click handler in
        # ``QtViewer`` adds back the absolute offset.
        point = points[0]
        x = float(point.pos().x())
        if self._click_callback is not None:
            self._click_callback(x)


# ---------------------------------------------------------------------------
# IntervalSeriesPanel
# ---------------------------------------------------------------------------


class IntervalSeriesPanel(
    pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin, RelativeTimeAxisMixin
):
    """Shaded vertical bands per (t_start, t_end) pair."""

    def __init__(
        self,
        model: IntervalSeriesModel,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self.setLabel("left", model.name)
        self.setLabel("bottom", "Time [s]")
        # Hide the y-axis ticks — there's no meaningful y here.
        self.getPlotItem().hideAxis("left")
        self._regions: list[pg.LinearRegionItem] = []
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        # Clear previous regions.
        for region in self._regions:
            self.removeItem(region)
        self._regions = []
        if payload.time.size == 0:
            return
        t_start, t_stop = float(payload.time[0]), float(payload.time[-1])
        starts, ends = self._model.window(t_start, t_stop)
        brush_color = QColor(self._model.color)
        brush_color.setAlphaF(self._model.alpha)
        # Render at relative coords against the panel's fixed x-range.
        offset = float(payload.t_center)
        for s, e in zip(starts, ends, strict=True):
            region = pg.LinearRegionItem(
                values=(float(s) - offset, float(e) - offset),
                movable=False,
                brush=brush_color,
            )
            self.addItem(region)
            self._regions.append(region)
