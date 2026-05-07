"""``QtLikelihoodHeatmapPanel`` — pyqtgraph rendering of the collapsed log-likelihood."""

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
    from non_local_detector.visualization.interactive.view_models.likelihood import (
        LikelihoodHeatmapModel,
    )


class QtLikelihoodHeatmapPanel(pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin):
    """Time × position heatmap of the population log-likelihood.

    Title-bar text mirrors the SlicePanel: "Likelihood across all
    spatial states; heatmap below shows non-local states only" so
    users see why the likelihood and posterior heatmaps may diverge
    for NL fits with ``local_position_std=1.0``.
    """

    def __init__(
        self,
        model: LikelihoodHeatmapModel,
        position_centers: np.ndarray,
        vmax: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self._position_centers = np.asarray(position_centers).squeeze()
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._image_item.setLookupTable(bone_lookup_table())
        self._image_item.setLevels((0.0, float(vmax)))
        self.addItem(self._image_item)
        self.setLabel("left", "Position [cm]")
        self.setLabel(
            "bottom",
            "Likelihood (peak-normalised, all spatial states)",
        )
        self._install_click_recenter()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.likelihood is None:
            self._image_item.clear()
            return
        collapsed = self._model.update_window(payload.likelihood)
        self._set_image(collapsed, payload.time)

    def update_for_array(self, time: np.ndarray, log_lik: np.ndarray) -> None:
        collapsed = self._model.update_window(log_lik)
        self._set_image(collapsed, np.asarray(time))

    def _set_image(self, collapsed: np.ndarray, time: np.ndarray) -> None:
        self._image_item.setImage(collapsed.T, autoLevels=False)
        if time.size and self._position_centers.size:
            x_min = float(time[0])
            x_extent = float(time[-1] - time[0]) if time.size > 1 else 1.0
            y_min = float(self._position_centers.min())
            y_extent = float(
                self._position_centers.max() - self._position_centers.min()
            )
            self._image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_extent, y_extent))
