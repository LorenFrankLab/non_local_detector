"""``QtPosteriorHeatmapPanel`` — pyqtgraph rendering of the collapsed posterior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.analysis.posterior import PosteriorReduction
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


# Title strings for each reduction strategy. The heatmap is *not*
# always P(position) — for NL detectors it's the conditional non-local
# posterior; for NoSpike+ContFrag it's the conditional-on-spatial
# posterior. The title disambiguates so a user reading the heatmap
# doesn't mistake a conditional plot for a marginal one.
_REDUCTION_TITLES: dict[PosteriorReduction, str] = {
    PosteriorReduction.CONDITIONAL_NON_LOCAL: "Non-local posterior",
    PosteriorReduction.CONDITIONAL_ON_SPATIAL: "Spatial posterior",
    PosteriorReduction.MARGINAL: "Posterior",
}


class QtPosteriorHeatmapPanel(HeatmapPanelBase):
    """Time × position heatmap of the collapsed posterior.

    The model handles strategy selection (``CONDITIONAL_NON_LOCAL`` /
    ``CONDITIONAL_ON_SPATIAL`` / ``MARGINAL``); this panel just maps
    the resulting array to an ``ImageItem`` and labels the title with
    the active reduction so the heatmap's meaning is unambiguous.
    """

    def __init__(
        self,
        model: PosteriorHeatmapModel,
        position_centers: np.ndarray,
        vmax: float = 0.25,
        parent=None,
    ) -> None:
        super().__init__(position_centers=position_centers, vmax=vmax, parent=parent)
        self._model = model
        self._refresh_title()

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind grid + refresh title (the new run may use a new reduction)."""
        super().set_position_centers(centers)
        self._refresh_title()

    def _refresh_title(self) -> None:
        self.setTitle(_REDUCTION_TITLES.get(self._model.reduction, "Posterior"))

    def update_window(self, payload: WindowPayload) -> None:
        if payload.posterior is None:
            self._image_item.clear()
            self._clear_position_trace()
            return
        collapsed = self._model.update_window(payload.posterior)
        # Render at relative coords against the fixed
        # ``[-t_width/2, +t_width/2]`` x-range.
        rel_time = payload.time - payload.t_center
        rel_start = (
            payload.time_start - payload.t_center
            if payload.time_start is not None
            else None
        )
        rel_stop = (
            payload.time_stop - payload.t_center
            if payload.time_stop is not None
            else None
        )
        self._set_image(
            collapsed,
            rel_time,
            time_start=rel_start,
            time_stop=rel_stop,
        )
        self._set_position_trace(rel_time, payload.position)

    def update_for_array(self, time: np.ndarray, posterior: np.ndarray) -> None:
        """Direct entry point for tests / callers that already have an array."""
        collapsed = self._model.update_window(posterior)
        self._set_image(collapsed, np.asarray(time))
