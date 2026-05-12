"""Optional 2D projection panel for graph-linearized 1D decoders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.projected_2d import (
        Projected2DFrame,
        Projected2DGeometry,
        Projected2DModel,
    )


# Match the posterior + likelihood heatmap colormap so users can
# cross-reference posterior magnitude across panels. Alpha grows with
# posterior so low-mass bins fade against the white background.
_VIRIDIS_LUT = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
_BRUSH_LUT: list[pg.QtGui.QBrush] = [
    pg.mkBrush(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(45 + 210 * (i / 255.0)))
    for i, rgb in enumerate(_VIRIDIS_LUT)
]


# Provider returns ``(posterior_row, animal_xy)`` for a decoder time
# index. ``posterior_row`` is a flat ``(n_state_bins,)`` slice of
# ``acausal_posterior``; ``animal_xy`` is a precomputed ``(2,)`` raw
# track XY (or ``None``). Either may be ``None`` to signal "no data".
RowProvider = Callable[[int], tuple[np.ndarray | None, np.ndarray | None]]


class QtProjected2DPanel(pg.PlotWidget):
    """Bin-synced 2D view of a collapsed 1D posterior on the track graph."""

    def __init__(self, model: Projected2DModel, parent=None) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self._buffered_payload: WindowPayload | None = None
        self._row_provider: RowProvider | None = None
        self._last_t_idx: int | None = None
        self.setTitle(_initial_title(model))
        self.setLabel("left", "y")
        self.setLabel("bottom", "x")
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.15)

        self._graph_items: list[pg.PlotDataItem] = []
        self._posterior_scatter = pg.ScatterPlotItem()
        self._animal_marker = pg.ScatterPlotItem()
        self._animal_marker.setZValue(10)
        self.addItem(self._posterior_scatter)
        self.addItem(self._animal_marker)
        self._render_static_geometry()
        # Aspect-locked plots don't always autorange to the bounding
        # box of static items; force one frame here so the graph is
        # visible before the first cursor tick populates the scatter.
        self.autoRange()

    def set_window_buffer(self, payload: WindowPayload) -> None:
        self._buffered_payload = payload

    def set_row_provider(self, provider: RowProvider | None) -> None:
        """Register a single-row fallback for cursor ticks past the buffer.

        When the cursor moves past the buffered window (typical during
        fast playback — the async window-load lags), ``update_for_index``
        calls the provider to fetch the cursor's posterior row and
        animal XY synchronously. Without it the panel would freeze on
        the last buffered frame.
        """
        self._row_provider = provider

    def update_for_index(self, t_idx: int) -> None:
        self._last_t_idx = int(t_idx)
        frame = self._frame_for_index(t_idx)
        if frame is not None:
            self._render_frame(frame)

    def rebind_after_swap(self) -> None:
        self._buffered_payload = None
        self._last_t_idx = None
        self.setTitle(_initial_title(self._model))
        self._render_static_geometry()
        self._posterior_scatter.setData(x=[], y=[])
        self._animal_marker.setData(x=[], y=[])
        self.autoRange()

    def _frame_for_index(self, t_idx: int) -> Projected2DFrame | None:
        """Resolve a frame for ``t_idx`` from the buffer or the provider."""
        payload = self._buffered_payload
        if payload is not None and payload.posterior is not None:
            sl = payload.indices
            local_idx = t_idx - sl.start
            if 0 <= local_idx < payload.posterior.shape[0]:
                return self._model.update_for_index(payload, t_idx)
        # Synchronous fallback so the projection stays cursor-locked
        # during fast playback even before the async load lands.
        if self._row_provider is None:
            return None
        row, animal_xy = self._row_provider(t_idx)
        if row is None:
            return None
        return self._model.frame_at_row(row, animal_xy)

    def _render_static_geometry(self) -> None:
        for item in self._graph_items:
            self.removeItem(item)
        self._graph_items = []
        geometry: Projected2DGeometry = self._model.geometry()
        for segment in geometry.graph_segments:
            item = pg.PlotDataItem(
                segment[:, 0],
                segment[:, 1],
                pen=pg.mkPen((150, 150, 150), width=2),
            )
            item.setZValue(0)
            self.addItem(item)
            self._graph_items.append(item)

    def _render_frame(self, frame: Projected2DFrame) -> None:
        geometry: Projected2DGeometry = self._model.geometry()
        if not frame.available or geometry.bin_xy is None or frame.posterior is None:
            self.setTitle(frame.message or "Projected 2D unavailable")
            self._posterior_scatter.setData(x=[], y=[])
            self._animal_marker.setData(x=[], y=[])
            return
        self.setTitle("Projected 2D decode")
        bin_xy = geometry.bin_xy
        finite_xy = np.all(np.isfinite(bin_xy), axis=1)
        finite_p = np.isfinite(frame.posterior)
        mask = finite_xy & finite_p
        xy = bin_xy[mask]
        posterior = np.asarray(frame.posterior[mask], dtype=float)
        if xy.size == 0:
            self._posterior_scatter.setData(x=[], y=[])
        else:
            brushes = _posterior_brushes(posterior)
            self._posterior_scatter.setData(
                x=xy[:, 0],
                y=xy[:, 1],
                size=9,
                brush=brushes,
                pen=pg.mkPen((60, 60, 60, 120), width=0.5),
            )
        if frame.animal_xy is None:
            self._animal_marker.setData(x=[], y=[])
        else:
            xy_animal = np.asarray(frame.animal_xy, dtype=float)
            self._animal_marker.setData(
                x=[xy_animal[0]],
                y=[xy_animal[1]],
                size=14,
                symbol="o",
                brush=pg.mkBrush(255, 0, 255, 230),
                pen=pg.mkPen((255, 255, 255, 230), width=2),
            )


def _initial_title(model: Projected2DModel) -> str:
    return (
        "Projected 2D decode"
        if model.is_available
        else (model.message or "Projected 2D unavailable")
    )


def _posterior_brushes(posterior: np.ndarray) -> list[pg.QtGui.QBrush]:
    posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
    max_p = float(np.max(posterior)) if posterior.size else 0.0
    if max_p > 0.0:
        idx = np.clip((posterior / max_p * 255.0).astype(int), 0, 255)
    else:
        idx = np.zeros(posterior.shape, dtype=int)
    return [_BRUSH_LUT[int(i)] for i in idx]
