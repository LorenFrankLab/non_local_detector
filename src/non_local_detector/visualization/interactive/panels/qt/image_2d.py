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


# Minimum panel height so the viewbox doesn't collapse when this
# panel shares a right column with the cell-grid stack.
_TOP_IMAGE_MIN_HEIGHT = 240


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
        vmax: float | None = None,
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
        # ``vmax=None`` (default) → per-frame peak-normalize so the
        # cursor row's brightest bin always lands at the LUT top
        # regardless of its absolute magnitude. The at-cursor view
        # spans a single time bin where peak posterior mass can be
        # well below 0.25 (the 1D heatmap default), and a fixed vmax
        # buries the actual peak in the LUT's dark-purple lower
        # quartile. Pass a float to pin a fixed scale.
        self._vmax: float | None = float(vmax) if vmax is not None else None
        self._buffered_payload: WindowPayload | None = None
        self._last_t_idx: int | None = None

        self.setTitle(title)
        self.setLabel("left", "y")
        self.setLabel("bottom", "x")
        # Don't aspect-lock. Combined with ``setLimits`` clamping the
        # axes to the data bounds, aspect-lock would crop the image
        # whenever the panel is non-square (the common case in the
        # right column). Letting the image stretch to fill the panel
        # keeps axis labels honest (they always read the bin bounds)
        # and shows the full posterior / likelihood — the small
        # aspect distortion is acceptable for a non-square widget.
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        # Reserve enough vertical space that the panel doesn't get
        # starved by the cell-grid sibling in the right column.
        self.setMinimumHeight(_TOP_IMAGE_MIN_HEIGHT)

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
        flat = np.asarray(self._model.collapse_at(window, [local_idx])[0])
        if self._vmax is None:
            peak = float(np.nanmax(flat)) if flat.size else 0.0
            frame_vmax = peak if peak > 0.0 else 1.0
        else:
            frame_vmax = self._vmax
        rgba = flat_to_rgba_image(flat, self._grid.shape, self._lut, frame_vmax)
        self._image_item.setImage(rgba, autoLevels=False)
        # ``setImage`` resets the ImageItem's transform back to
        # pixel coords (rect = image shape), so the rect set during
        # construction is gone by the first cursor tick. Re-apply
        # it here so the image is always positioned in the data's
        # ``(x_min, y_min, width, height)`` rectangle.
        self._image_item.setRect(
            QtCore.QRectF(
                self._layout.x_min,
                self._layout.y_min,
                self._layout.width,
                self._layout.height,
            )
        )
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
        # Cache for ``update_for_index`` — ``setImage`` resets the
        # ImageItem transform on every call so we re-apply the rect
        # per render.
        self._layout = layout
        self._image_item.setRect(
            QtCore.QRectF(layout.x_min, layout.y_min, layout.width, layout.height)
        )
        vb = self.getViewBox()
        # Clamp axes to the data bounds + lock the data aspect ratio.
        # The panel's minimum height (``_TOP_IMAGE_MIN_HEIGHT``) gives
        # aspect-lock room to render the image at proper proportions
        # within the widget — non-square widgets get white margins on
        # the longer pixel-per-unit axis rather than extending the
        # visible data range past the bin bounds (which would
        # mislabel the axis ticks).
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
