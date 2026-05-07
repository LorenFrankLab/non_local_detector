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
    EventOverlayMixin,
    bone_lookup_table,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )


class QtPosteriorHeatmapPanel(pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin):
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
        self._model = model
        self._position_centers = np.asarray(position_centers).squeeze()
        self._vmax = float(vmax)
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._image_item.setLookupTable(bone_lookup_table())
        self._image_item.setLevels((0.0, self._vmax))
        self.addItem(self._image_item)
        self.setLabel("left", "Position [cm]")
        self.setLabel("bottom", "Time [s]")
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.posterior is None:
            self._image_item.clear()
            return
        collapsed = self._model.update_window(payload.posterior)
        self._set_image(collapsed, payload.time)

    def update_for_array(self, time: np.ndarray, posterior: np.ndarray) -> None:
        """Direct entry point for tests / callers that already have an array."""
        collapsed = self._model.update_window(posterior)
        self._set_image(collapsed, np.asarray(time))

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind the y-axis position grid (called on M-key swap)."""
        self._position_centers = np.asarray(centers).squeeze()

    def _set_image(self, collapsed: np.ndarray, time: np.ndarray) -> None:
        # ImageItem rows are y, columns are x (axisOrder="row-major").
        # Our collapsed array is (n_visible, n_pos) — time on x, position
        # on y — so transpose before set.
        self._image_item.setImage(collapsed.T, autoLevels=False)
        # Map x to time, y to position so axes display real units.
        if time.size and self._position_centers.size:
            x_min = float(time[0])
            x_extent = float(time[-1] - time[0]) if time.size > 1 else 1.0
            y_min = float(self._position_centers.min())
            y_extent = float(
                self._position_centers.max() - self._position_centers.min()
            )
            self._image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_extent, y_extent))
