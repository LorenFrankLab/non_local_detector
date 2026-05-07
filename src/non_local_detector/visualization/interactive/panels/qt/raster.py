"""``QtRasterPanel`` — per-cell spike-time raster, sorted by place-field peak."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor

from non_local_detector.analysis.posterior import _non_local_state_ids
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

    When the active detector exposes ``Non-Local`` states, the panel
    also shades time-bins where the smoothed non-local probability
    exceeds ``non_local_threshold`` — a translucent band drawn
    behind the spikes makes those windows immediately visible
    against the raster.
    """

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
        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(color=spike_color, width=1),
            brush=pg.mkBrush(spike_color),
            size=2,
            symbol="s",
        )
        self.addItem(self._scatter)
        self._non_local_regions: list[pg.LinearRegionItem] = []
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    @staticmethod
    def _build_non_local_brush(color: str, alpha: float):
        c = QColor(color)
        c.setAlphaF(alpha)
        return pg.mkBrush(c)

    def update_window(self, payload: WindowPayload) -> None:
        if payload.time.size == 0:
            self._scatter.setData([], [])
            self._clear_non_local_regions()
            return
        t_start = float(payload.time[0])
        t_stop = float(payload.time[-1])
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

    def _render_spikes(self, t_start: float, t_stop: float) -> None:
        raster = self._model.update_window(t_start, t_stop)
        xs: list[float] = []
        ys: list[float] = []
        for y_row, cell_spikes in enumerate(raster.spike_times_per_cell):
            xs.extend(cell_spikes.tolist())
            ys.extend([float(y_row)] * cell_spikes.size)
        self._scatter.setData(xs, ys)

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
