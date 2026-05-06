"""``QtPosteriorHeatmapPanel`` — pyqtgraph rendering of the collapsed posterior.

Consumes ``PosteriorHeatmapModel`` (per-row schema-aware collapse) and
renders the ``(n_visible, n_pos)`` array as a ``pg.ImageItem``. Mixes
in ``EventOverlayMixin`` for free overlay support.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    EventOverlayMixin,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )


class QtPosteriorHeatmapPanel(pg.PlotWidget, EventOverlayMixin):
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
        self._image_item.setLookupTable(_bone_lookup_table())
        self._image_item.setLevels((0.0, self._vmax))
        self.addItem(self._image_item)
        self.setLabel("left", "Position [cm]")
        self.setLabel("bottom", "Time [s]")
        self._click_callback: Callable[[float], None] | None = None
        self.scene().sigMouseClicked.connect(self._handle_click)
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

    # ------------------------------------------------------------------
    # TimeAxisPanel protocol
    # ------------------------------------------------------------------

    def x_link_target(self):
        return self.getPlotItem()

    def click_handler(self, callback: Callable[[float], None]) -> None:
        self._click_callback = callback

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_click(self, mouse_event) -> None:
        if self._click_callback is None:
            return
        scene_pos = mouse_event.scenePos()
        view_pos = self.getPlotItem().vb.mapSceneToView(scene_pos)
        self._click_callback(float(view_pos.x()))


def _bone_lookup_table() -> np.ndarray:
    """matplotlib `bone_r` analogue as a uint8 LUT."""
    n = 256
    t = np.linspace(0, 1, n)
    # matplotlib bone_r: white → blue-grey → black
    r = (1.0 - t) * 0.875 + (1.0 - 0.875) * (1.0 - t)
    g = (1.0 - t) * 0.875 + (1.0 - 0.875) * (1.0 - t)
    b = 1.0 - t
    lut = np.stack(
        [r * 255, g * 255, b * 255, np.full(n, 255.0)],
        axis=-1,
    ).astype(np.uint8)
    return lut
