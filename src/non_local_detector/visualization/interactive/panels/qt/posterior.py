"""``QtPosteriorHeatmapPanel`` — pyqtgraph rendering of the collapsed posterior.

Consumes ``PosteriorHeatmapModel`` (per-row schema-aware collapse) and
renders the ``(n_visible, n_pos)`` array as a ``pg.ImageItem``. Mixes
in ``EventOverlayMixin`` + ``ClickRecenterMixin``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
    PositionTraceMixin,
    position_grid_layout,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )


class QtPosteriorHeatmapPanel(
    pg.PlotWidget,
    EventOverlayMixin,
    ClickRecenterMixin,
    CursorMarkersMixin,
    PositionTraceMixin,
):
    """Time × position heatmap of the collapsed posterior.

    The model handles strategy selection (``CONDITIONAL_NON_LOCAL`` /
    ``CONDITIONAL_ON_SPATIAL`` / ``MARGINAL``); this panel just maps
    the resulting array to an ``ImageItem``.
    """

    def __init__(
        self,
        model: PosteriorHeatmapModel,
        position_centers: np.ndarray,
        vmax: float = 0.25,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getAxis("left").enableAutoSIPrefix(False)
        self._model = model
        self._set_position_grid(position_centers)
        self._vmax = float(vmax)
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._image_item.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        )
        self._image_item.setLevels((0.0, self._vmax))
        self.addItem(self._image_item)
        self.setLabel("left", "Position [cm]")
        self.setLabel("bottom", "Time [s]")
        self._install_click_recenter()
        self._install_cursor_markers()
        self._install_position_trace()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.posterior is None:
            self._image_item.clear()
            self._clear_position_trace()
            return
        collapsed = self._model.update_window(payload.posterior)
        self._set_image(collapsed, payload.time)
        self._set_position_trace(payload.time, payload.position)

    def update_for_array(self, time: np.ndarray, posterior: np.ndarray) -> None:
        """Direct entry point for tests / callers that already have an array."""
        collapsed = self._model.update_window(posterior)
        self._set_image(collapsed, np.asarray(time))

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind the y-axis position grid (called on M-key swap)."""
        self._set_position_grid(centers)

    def _set_position_grid(self, centers: np.ndarray) -> None:
        """Cache the layout used by ``setRect`` + the position trace.

        Pads by half a uniform step so pixel CENTERS sit at bin
        centers — the trace and the heatmap bins land on the same
        pixel rows even on non-uniform grids.
        """
        (
            self._position_centers,
            self._y0,
            self._y1,
            self._dy_half,
            self._uniform_step,
            self._arange_n_pos,
        ) = position_grid_layout(centers)
        y_min = self._y0 - self._dy_half
        y_max = self._y1 + self._dy_half
        vb = self.getViewBox()
        vb.disableAutoRange()
        vb.setYRange(y_min, y_max, padding=0)
        vb.setLimits(yMin=y_min, yMax=y_max)

    def _set_image(self, collapsed: np.ndarray, time: np.ndarray) -> None:
        # ImageItem rows are y, columns are x (axisOrder="row-major").
        # Our collapsed array is (n_visible, n_pos) — time on x, position
        # on y — so transpose before set.
        self._image_item.setImage(
            collapsed.T,
            autoLevels=False,
            levels=(0.0, self._vmax),
            autoDownsample=False,
        )
        # Map x to time, y to position so axes display real units.
        # Pad the y bounds by half a bin so each pixel CENTER sits at
        # the bin center, matching ``statespacecheck-paper-viewer``'s
        # convention. Without the pad the position trace would be
        # half a bin off the heatmap rows.
        if time.size and self._position_centers.size:
            x_min = float(time[0])
            x_extent = float(time[-1] - time[0]) if time.size > 1 else 1.0
            y_min = self._y0 - self._dy_half
            y_extent = (self._y1 - self._y0) + 2 * self._dy_half
            self._image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_extent, y_extent))
