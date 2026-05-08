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
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QColor

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
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


class LineSeriesPanel(pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin):
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
        self._line.setData(t, y)
        if self._fill_curve is not None:
            self._zero_curve.setData(t, [0.0] * len(t))


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


_LEGEND_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-size: 11pt; }"
)


def _legend_html(items: list[tuple[str, str]]) -> str:
    """Build the HTML for a sidebar legend from ``[(color, label), ...]``."""
    rows = [
        f"<span style='color:{color};font-size:14pt'>━</span> {label}"
        for color, label in items
    ]
    return "<br>".join(rows)


class _MultiLineSeriesPlot(pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin):
    """Internal plot widget — one line per ``model.ys`` entry.

    Lifted out of ``MultiLineSeriesPanel`` so the wrapper can lay the
    plot next to a sidebar legend.
    """

    def __init__(self, model: MultiLineSeriesModel, parent=None) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self.setLabel("left", model.name)
        self.setLabel("bottom", "Time [s]")
        if model.y_range is not None:
            self.setYRange(*model.y_range)
        self._lines: dict[str, pg.PlotDataItem] = {}
        for i, label in enumerate(model.ys):
            color = (
                model.colors[label]
                if model.colors and label in model.colors
                else _DEFAULT_MULTI_COLORS[i % len(_DEFAULT_MULTI_COLORS)]
            )
            self._lines[label] = self.plot(
                [], [], pen=pg.mkPen(color=color, width=3), name=label
            )
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            for line in self._lines.values():
                line.setData([], [])
            return
        t_start, t_stop = float(payload.time[0]), float(payload.time[-1])
        t, ys = self._model.window(t_start, t_stop)
        for label, line in self._lines.items():
            line.setData(t, ys[label])


class MultiLineSeriesPanel(QtWidgets.QWidget):
    """Multi-line plot with the legend laid out *outside* the plot.

    A pyqtgraph ``LegendItem`` floats inside the viewbox and can
    occlude the lines on small panels. This wrapper places the plot
    in one column and an HTML legend ``QLabel`` in a sidebar so the
    legend never overlaps the data. Mixin / pyqtgraph methods that
    callers expect on the panel forward to the inner plot widget via
    ``__getattr__``.
    """

    def __init__(
        self,
        model: MultiLineSeriesModel,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._plot = _MultiLineSeriesPlot(model)
        self._legend = QtWidgets.QLabel()
        self._legend.setStyleSheet(_LEGEND_STYLE)
        self._legend.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self._legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._refresh_legend()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._plot, stretch=1)
        layout.addWidget(self._legend, stretch=0)

    def __getattr__(self, name: str):
        plot = self.__dict__.get("_plot")
        if plot is None:
            raise AttributeError(name)
        return getattr(plot, name)

    def _refresh_legend(self) -> None:
        items: list[tuple[str, str]] = []
        for i, label in enumerate(self._plot._model.ys):
            color_value = (
                self._plot._model.colors.get(label)
                if self._plot._model.colors
                else None
            )
            if color_value is None:
                color_value = _DEFAULT_MULTI_COLORS[i % len(_DEFAULT_MULTI_COLORS)]
            items.append((color_value, label))
        self._legend.setText(_legend_html(items))


# ---------------------------------------------------------------------------
# ScatterSeriesPanel
# ---------------------------------------------------------------------------


class ScatterSeriesPanel(
    pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin, CursorMarkersMixin
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
        self._scatter.setData(t, y)

    def _on_point_clicked(self, _scatter, points) -> None:
        if not points:
            return
        # Recenter on the first clicked point's x coordinate.
        point = points[0]
        x = float(point.pos().x())
        if self._click_callback is not None:
            self._click_callback(x)


# ---------------------------------------------------------------------------
# IntervalSeriesPanel
# ---------------------------------------------------------------------------


class IntervalSeriesPanel(pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin):
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
        for s, e in zip(starts, ends, strict=True):
            region = pg.LinearRegionItem(
                values=(float(s), float(e)),
                movable=False,
                brush=brush_color,
            )
            self.addItem(region)
            self._regions.append(region)
