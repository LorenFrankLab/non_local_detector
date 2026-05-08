"""``QtPosteriorHeatmapPanel`` — pyqtgraph rendering of the collapsed posterior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    HeatmapPanelBase,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )


class QtPosteriorHeatmapPanel(HeatmapPanelBase):
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
        super().__init__(
            position_centers=position_centers, vmax=vmax, parent=parent
        )
        self._model = model

    def update_window(self, payload: WindowPayload) -> None:
        if payload.posterior is None:
            self._image_item.clear()
            self._clear_position_trace()
            return
        collapsed = self._model.update_window(payload.posterior)
        self._set_image(
            collapsed,
            payload.time,
            time_start=payload.time_start,
            time_stop=payload.time_stop,
        )
        self._set_position_trace(payload.time, payload.position)

    def update_for_array(self, time: np.ndarray, posterior: np.ndarray) -> None:
        """Direct entry point for tests / callers that already have an array."""
        collapsed = self._model.update_window(posterior)
        self._set_image(collapsed, np.asarray(time))
