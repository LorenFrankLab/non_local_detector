"""``QtRasterPanel`` — per-cell spike-time raster, sorted by place-field peak."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtGui import QColor

from non_local_detector.analysis.posterior import _non_local_state_ids
from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.raster import (
        RasterModel,
    )


class QtRasterPanel(
    pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin, CursorMarkersMixin
):
    """Per-cell spike-time raster.

    Cells are sorted by place-field peak (handled by ``RasterModel``).
    Each cell occupies one y-row; spike times are drawn as short
    vertical line segments.

    When the active detector exposes ``Non-Local`` states, the panel
    also shades time-bins where the smoothed non-local probability
    exceeds ``non_local_threshold`` — a translucent band drawn
    behind the spikes makes those windows immediately visible
    against the raster.

    Click on a spike → ``event_clicked.emit(event_id)`` plus the
    compatibility ``spike_clicked.emit(cell_id, t)`` signal. The y
    coordinate of the clicked spot is mapped through
    ``RasterModel.sort_indices`` to the original cell id. Clicks on
    empty raster area still drive the inherited ``ClickRecenterMixin``
    recenter handler — the mixin skips when ``ScatterPlotItem.sigClicked``
    accepted the underlying mouse event.
    """

    cell_clicked = QtCore.Signal(int)
    spike_clicked = QtCore.Signal(int, float)
    event_clicked = QtCore.Signal(int)

    def __init__(
        self,
        model: RasterModel,
        spike_color: str = "k",
        non_local_color: str = "#d62728",
        non_local_alpha: float = 0.18,
        non_local_threshold: float = 0.5,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self._spike_color = spike_color
        self._non_local_threshold = float(non_local_threshold)
        self._non_local_brush = self._build_non_local_brush(
            non_local_color, non_local_alpha
        )
        self.setLabel("left", model.cell_label)
        self.setLabel("bottom", "Time [s]")
        self.setMouseEnabled(x=False, y=False)
        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(color=spike_color, width=2),
            brush=pg.mkBrush(spike_color),
            size=4,
            symbol="s",
        )
        self.addItem(self._scatter)
        self._pin_dot = pg.ScatterPlotItem(
            pen=pg.mkPen((255, 215, 0), width=3),
            brush=pg.mkBrush(255, 215, 0),
            size=10,
            symbol="o",
        )
        self._pin_dot.setZValue(21)
        self._pin_dot.setVisible(False)
        self.addItem(self._pin_dot)
        self._non_local_regions: list[pg.LinearRegionItem] = []
        self._install_click_recenter()
        self._install_cursor_markers()
        self._scatter.sigClicked.connect(self._handle_spike_click)
        self._overlay_items: list[pg.GraphicsObject] = []
        self._pin_y_range()

    @staticmethod
    def _build_non_local_brush(color: str, alpha: float):
        c = QColor(color)
        c.setAlphaF(alpha)
        return pg.mkBrush(c)

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            self._scatter.setData([], [])
            self._clear_non_local_regions()
            self._pin_y_range()
            return
        t_start = (
            float(payload.time_start)
            if payload.time_start is not None
            else float(payload.time[0])
        )
        t_stop = (
            float(payload.time_stop)
            if payload.time_stop is not None
            else float(payload.time[-1])
        )
        self._render_spikes(t_start, t_stop)
        self._render_non_local_regions(payload)

    def update_for_window(self, t_start: float, t_stop: float) -> None:
        """Direct entry for tests / callers without a payload."""
        self._render_spikes(t_start, t_stop)

    def rebind_after_swap(self) -> None:
        """Refresh axis label after the model rebinds.

        ``RasterModel.set_active_run`` may have changed the
        ``cell_label`` (e.g., switching to/from a multi-encoding-group
        detector); update the y-axis text to match.
        """
        self.setLabel("left", self._model.cell_label)
        self._pin_y_range()

    def _handle_spike_click(self, _scatter, points) -> None:
        """Resolve clicked spot's y-row to a cell_id and emit spike identity.

        Y-rows in the scatter are sorted positions; ``RasterModel.sort_indices``
        maps row index → original cell id. ``ScatterPlotItem.sigClicked``
        emits ``points`` as either a list of ``SpotItem`` or a numpy
        object array depending on pyqtgraph version — ``len()`` works
        for both.
        """
        if len(points) == 0:
            return
        spot = points[0]
        y_row = int(round(float(spot.pos().y())))
        sort_indices = self._model.sort_indices
        if not 0 <= y_row < sort_indices.size:
            return
        cell_id = int(sort_indices[y_row])
        self.cell_clicked.emit(cell_id)
        event_id = spot.data()
        if event_id is not None and int(event_id) >= 0:
            self.event_clicked.emit(int(event_id))
        self.spike_clicked.emit(cell_id, float(spot.pos().x()))

    def set_spike_pin_marker(self, t: float | None, cell_id: int | None) -> None:
        """Show/hide the pinned spike line plus row-local dot."""
        self.set_pin_marker(t)
        if t is None or cell_id is None:
            self._pin_dot.setVisible(False)
            return
        rows = np.flatnonzero(self._model.sort_indices == int(cell_id))
        if rows.size == 0:
            self._pin_dot.setVisible(False)
            return
        self._pin_dot.setData([float(t)], [float(rows[0])])
        self._pin_dot.setVisible(True)

    def _render_spikes(self, t_start: float, t_stop: float) -> None:
        raster = self._model.update_window(t_start, t_stop)
        xs: list[float] = []
        ys: list[float] = []
        event_ids: list[int] = []
        for y_row, cell_spikes in enumerate(raster.spike_times_per_cell):
            xs.extend(cell_spikes.tolist())
            ys.extend([float(y_row)] * cell_spikes.size)
            event_ids.extend(raster.event_ids_per_cell[y_row].astype(int).tolist())
        self._scatter.setData(xs, ys, data=event_ids)
        self._pin_y_range()

    def _pin_y_range(self) -> None:
        """Keep the full sorted cell axis visible without per-window autorange."""
        n_cells = int(self._model.sort_indices.size)
        if n_cells <= 0:
            y_min, y_max = -0.5, 0.5
        else:
            y_min, y_max = -0.5, float(n_cells) - 0.5
        vb = self.getViewBox()
        vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        vb.setYRange(y_min, y_max, padding=0)
        vb.setLimits(yMin=y_min, yMax=y_max)

    def _clear_non_local_regions(self) -> None:
        for region in self._non_local_regions:
            self.removeItem(region)
        self._non_local_regions = []

    def _render_non_local_regions(self, payload: WindowPayload) -> None:
        self._clear_non_local_regions()
        probs = payload.state_probabilities
        if probs is None or probs.size == 0:
            return
        nl_state_ids = _non_local_state_ids(self._model.detector)
        if nl_state_ids.size == 0:
            return  # No "Non-Local" states for this detector.
        nl_mass = probs[:, nl_state_ids].sum(axis=1)
        active = nl_mass > self._non_local_threshold
        if not active.any():
            return
        # Group consecutive active bins into contiguous spans.
        time = payload.time
        for span_start, span_stop in _contiguous_spans(active):
            t_lo = float(time[span_start])
            t_hi = float(time[span_stop - 1])
            region = pg.LinearRegionItem(
                values=(t_lo, t_hi),
                movable=False,
                brush=self._non_local_brush,
            )
            self.addItem(region)
            self._non_local_regions.append(region)


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, stop), ...]`` for runs of ``True`` in ``mask``.

    ``stop`` is exclusive (Python slice convention). Empty input or
    all-False mask returns ``[]``.
    """
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))
