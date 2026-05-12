"""Optional projected-1D heatmap for 2D decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    HeatmapPanelBase,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.projected_1d import (
        Projected1DModel,
    )


class QtProjected1DHeatmapPanel(HeatmapPanelBase):
    """Time × linear-position view of a 2D decode projected onto a graph."""

    def __init__(self, model: Projected1DModel, parent=None) -> None:
        super().__init__(
            position_centers=model.linear_centers,
            vmax=0.25,
            bottom_label="Time [s] (relative)",
            parent=parent,
        )
        self._model = model
        self._refresh_title()

    def rebind_after_swap(self) -> None:
        """Refresh geometry after the model has rebound to a new run."""
        self.set_position_centers(self._model.linear_centers)
        self._refresh_title()
        self._image_item.clear()
        self._clear_position_trace()

    def update_window(self, payload: WindowPayload) -> None:
        projected, position = self._model.update_window(payload)
        if projected is None:
            self._image_item.clear()
            self._clear_position_trace()
            self._refresh_title()
            return
        rel_time = payload.time - payload.t_center
        rel_start = (
            payload.time_start - payload.t_center
            if payload.time_start is not None
            else None
        )
        rel_stop = (
            payload.time_stop - payload.t_center if payload.time_stop is not None else None
        )
        self._set_image(
            projected,
            rel_time,
            time_start=rel_start,
            time_stop=rel_stop,
        )
        self._set_position_trace(rel_time, position)
        self.setTitle("Projected 1D posterior")

    def _refresh_title(self) -> None:
        self.setTitle(
            "Projected 1D posterior"
            if self._model.is_available
            else (self._model.message or "Projected 1D unavailable")
        )
