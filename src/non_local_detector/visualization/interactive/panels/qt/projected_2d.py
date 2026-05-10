"""Optional 2D projection panel for graph-linearized 1D decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.projected_2d import (
        Projected2DModel,
        Projected2DPayload,
    )


# Match the posterior + likelihood heatmap colormap so users can
# cross-reference posterior magnitude across panels. Alpha grows with
# posterior so low-mass bins fade against the white background.
_VIRIDIS_LUT = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
_BRUSH_LUT: list[pg.QtGui.QBrush] = [
    pg.mkBrush(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(45 + 210 * (i / 255.0)))
    for i, rgb in enumerate(_VIRIDIS_LUT)
]


class QtProjected2DPanel(pg.PlotWidget):
    """Bin-synced 2D view of a collapsed 1D posterior on the track graph."""

    def __init__(self, model: Projected2DModel, parent=None) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self._buffered_payload: WindowPayload | None = None
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

    def update_for_index(self, t_idx: int) -> None:
        self._last_t_idx = int(t_idx)
        if self._buffered_payload is None:
            return
        projected = self._model.update_for_index(self._buffered_payload, int(t_idx))
        self._render(projected)

    def rebind_after_swap(self) -> None:
        self._buffered_payload = None
        self._last_t_idx = None
        self.setTitle(_initial_title(self._model))
        self._render_static_geometry()
        self._posterior_scatter.setData(x=[], y=[])
        self._animal_marker.setData(x=[], y=[])
        self.autoRange()

    def _render_static_geometry(self) -> None:
        for item in self._graph_items:
            self.removeItem(item)
        self._graph_items = []
        for segment in self._model.geometry_payload().graph_segments:
            item = pg.PlotDataItem(
                segment[:, 0],
                segment[:, 1],
                pen=pg.mkPen((150, 150, 150), width=2),
            )
            item.setZValue(0)
            self.addItem(item)
            self._graph_items.append(item)

    def _render(self, projected: Projected2DPayload) -> None:
        if not projected.available:
            self.setTitle(projected.message or "Projected 2D unavailable")
            self._posterior_scatter.setData(x=[], y=[])
            self._animal_marker.setData(x=[], y=[])
            return
        assert projected.bin_xy is not None
        assert projected.posterior is not None
        self.setTitle("Projected 2D decode")
        finite_xy = np.all(np.isfinite(projected.bin_xy), axis=1)
        finite_p = np.isfinite(projected.posterior)
        mask = finite_xy & finite_p
        xy = projected.bin_xy[mask]
        posterior = np.asarray(projected.posterior[mask], dtype=float)
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
        if projected.animal_xy is None:
            self._animal_marker.setData(x=[], y=[])
        else:
            xy_animal = np.asarray(projected.animal_xy, dtype=float)
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
