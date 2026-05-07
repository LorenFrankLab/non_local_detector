"""``QtRasterPanel`` — per-cell spike-time raster, sorted by place-field peak."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    EventOverlayMixin,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.raster import (
        RasterModel,
    )


class QtRasterPanel(pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin):
    """Per-cell spike-time raster.

    Cells are sorted by place-field peak (handled by ``RasterModel``).
    Each cell occupies one y-row; spike times are drawn as short
    vertical line segments.
    """

    def __init__(
        self,
        model: RasterModel,
        spike_color: str = "k",
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self._spike_color = spike_color
        self.setLabel("left", model.cell_label)
        self.setLabel("bottom", "Time [s]")
        # Single ScatterPlotItem with tiny tick markers — much cheaper
        # than one PlotDataItem per cell.
        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(color=spike_color, width=1),
            brush=pg.mkBrush(spike_color),
            size=2,
            symbol="s",
        )
        self.addItem(self._scatter)
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            self._scatter.setData([], [])
            return
        t_start = float(payload.time[0])
        t_stop = float(payload.time[-1])
        raster = self._model.update_window(t_start, t_stop)
        xs: list[float] = []
        ys: list[float] = []
        for y_row, cell_spikes in enumerate(raster.spike_times_per_cell):
            xs.extend(cell_spikes.tolist())
            ys.extend([float(y_row)] * cell_spikes.size)
        self._scatter.setData(xs, ys)

    def update_for_window(self, t_start: float, t_stop: float) -> None:
        """Direct entry for tests / callers without a payload."""
        raster = self._model.update_window(t_start, t_stop)
        xs: list[float] = []
        ys: list[float] = []
        for y_row, cell_spikes in enumerate(raster.spike_times_per_cell):
            xs.extend(cell_spikes.tolist())
            ys.extend([float(y_row)] * cell_spikes.size)
        self._scatter.setData(xs, ys)

    def rebind_after_swap(self) -> None:
        """Refresh axis label after the model rebinds.

        ``RasterModel.set_active_run`` may have changed the
        ``cell_label`` (e.g., switching to/from a multi-encoding-group
        detector); update the y-axis text to match.
        """
        self.setLabel("left", self._model.cell_label)
