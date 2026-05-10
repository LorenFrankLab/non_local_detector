"""``Qt2DImagePanel`` — bin-synced 2D ``(x, y)`` image at the cursor.

Used as the at-cursor view for 2D detectors. Renders a single-time-bin
slice of a posterior or likelihood window as a 2D image with x on the
horizontal axis, y on the vertical, and the animal's position overlaid
as a magenta dot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from non_local_detector.visualization.interactive.panels.qt._image_2d import (
    flat_to_rgba_image,
    image_2d_layout_from_grid,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        PositionGrid,
        WindowPayload,
    )


class _BinCollapse(Protocol):
    """Interface the panel needs from a view model.

    Both ``PosteriorHeatmapModel.collapse_at`` and
    ``LikelihoodHeatmapModel.collapse_at`` satisfy this — they share
    a ``(n_window, n_state_bins) + indices -> (len(indices), n_pos)``
    shape contract even though their keyword names differ. The
    panel always passes the window positionally so the parameter
    name on the underlying method is irrelevant.
    """

    def collapse_at(self, window: np.ndarray, indices: list[int], /) -> np.ndarray: ...


class Qt2DImagePanel(pg.PlotWidget):
    """Bin-synced 2D image of a collapsed posterior / likelihood row.

    Parameters
    ----------
    model
        View model exposing ``collapse_at(window, indices) ->
        (len(indices), n_pos)``. ``PosteriorHeatmapModel`` qualifies.
    grid
        ``PositionGrid`` with ``ndim == 2`` carrying the grid shape +
        2D interior mask.
    payload_field
        Which ``WindowPayload`` field to index for the source window.
        ``"posterior"`` (default) or ``"likelihood"`` — the panel
        pulls ``getattr(payload, payload_field)``.
    title
        Static title shown above the image.
    vmax
        Upper bound of the colormap range; values clipped to
        ``[0, vmax]`` before LUT indexing. Defaults to ``0.25`` to
        match the existing posterior heatmap.
    """

    def __init__(
        self,
        model: _BinCollapse,
        grid: PositionGrid,
        *,
        payload_field: str = "posterior",
        title: str = "Posterior at cursor",
        vmax: float = 0.25,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        if grid.ndim != 2 or grid.shape is None:
            raise ValueError(
                "Qt2DImagePanel requires a 2D PositionGrid; got "
                f"ndim={grid.ndim}, shape={grid.shape}."
            )
        self._model = model
        self._grid = grid
        self._payload_field = payload_field
        self._vmax = float(vmax)
        self._buffered_payload: WindowPayload | None = None
        self._last_t_idx: int | None = None

        self.setTitle(title)
        self.setLabel("left", "y")
        self.setLabel("bottom", "x")
        self.setAspectLocked(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)

        self._lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._image_item)
        self._animal_marker = pg.ScatterPlotItem()
        self._animal_marker.setZValue(10)
        self.addItem(self._animal_marker)

        self._apply_grid_geometry()

    def set_window_buffer(self, payload: WindowPayload) -> None:
        self._buffered_payload = payload

    def update_for_index(self, t_idx: int) -> None:
        self._last_t_idx = int(t_idx)
        payload = self._buffered_payload
        if payload is None:
            return
        window = getattr(payload, self._payload_field, None)
        if window is None:
            self._clear_image()
            return
        local_idx = int(t_idx - payload.indices.start)
        if local_idx < 0 or local_idx >= window.shape[0]:
            return
        flat = self._model.collapse_at(window, [local_idx])[0]
        rgba = flat_to_rgba_image(
            np.asarray(flat), self._grid.shape, self._lut, self._vmax
        )
        self._image_item.setImage(rgba, autoLevels=False)
        self._update_animal_marker(payload, local_idx)

    def rebind_after_swap(self, grid: PositionGrid | None = None) -> None:
        self._buffered_payload = None
        self._last_t_idx = None
        if grid is not None:
            if grid.ndim != 2 or grid.shape is None:
                raise ValueError(
                    "Qt2DImagePanel.rebind_after_swap requires a 2D "
                    f"PositionGrid; got ndim={grid.ndim}."
                )
            self._grid = grid
            self._apply_grid_geometry()
        self._clear_image()

    def _apply_grid_geometry(self) -> None:
        layout = image_2d_layout_from_grid(self._grid)
        self._image_item.setRect(
            QtCore.QRectF(layout.x_min, layout.y_min, layout.width, layout.height)
        )
        vb = self.getViewBox()
        vb.setRange(
            xRange=(layout.x_min, layout.x_max),
            yRange=(layout.y_min, layout.y_max),
            padding=0,
        )
        vb.setLimits(
            xMin=layout.x_min,
            xMax=layout.x_max,
            yMin=layout.y_min,
            yMax=layout.y_max,
        )

    def _clear_image(self) -> None:
        self._image_item.clear()
        self._animal_marker.setData(x=[], y=[])

    def _update_animal_marker(self, payload: WindowPayload, local_idx: int) -> None:
        position = payload.position
        if (
            position is None
            or position.ndim != 2
            or position.shape[1] != 2
            or local_idx >= position.shape[0]
        ):
            self._animal_marker.setData(x=[], y=[])
            return
        xy = np.asarray(position[local_idx], dtype=float)
        if not np.all(np.isfinite(xy)):
            self._animal_marker.setData(x=[], y=[])
            return
        self._animal_marker.setData(
            x=[float(xy[0])],
            y=[float(xy[1])],
            size=14,
            symbol="o",
            brush=pg.mkBrush(255, 0, 255, 230),
            pen=pg.mkPen((255, 255, 255, 230), width=2),
        )
