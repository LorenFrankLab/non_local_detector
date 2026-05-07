"""``QtLikelihoodHeatmapPanel`` — pyqtgraph rendering of the collapsed log-likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
    PositionTraceMixin,
    bone_lookup_table,
    position_grid_layout,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.likelihood import (
        LikelihoodHeatmapModel,
    )


class QtLikelihoodHeatmapPanel(
    pg.PlotWidget,
    EventOverlayMixin,
    ClickRecenterMixin,
    CursorMarkersMixin,
    PositionTraceMixin,
):
    """Time × position heatmap of the population log-likelihood.

    Title-bar text mirrors the SlicePanel: "Likelihood across all
    spatial states; heatmap below shows non-local states only" so
    users see why the likelihood and posterior heatmaps may diverge
    for NL fits with ``local_position_std=1.0``.

    When the active run was produced without ``log_likelihood``
    (default ``predict()`` call), the heatmap blanks and the panel's
    title displays a ``MISSING_DATA_MESSAGE`` explaining the
    re-``predict`` step needed to enable it. Empty/cleared display
    on its own is ambiguous (could be "no spikes in window" or
    "log_likelihood not requested"); the title disambiguates per the
    documented ``RunBundle`` contract.
    """

    MISSING_DATA_MESSAGE = (
        "Likelihood unavailable — re-run "
        "predict(return_outputs=['log_likelihood']) to enable this panel."
    )

    def __init__(
        self,
        model: LikelihoodHeatmapModel,
        position_centers: np.ndarray,
        vmax: float = 1.0,
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
        self._image_item.setLookupTable(bone_lookup_table())
        self._image_item.setLevels((0.0, self._vmax))
        self.addItem(self._image_item)
        self.setLabel("left", "Position [cm]")
        self.setLabel(
            "bottom",
            "Likelihood (peak-normalised, all spatial states)",
        )
        # Mirrors whatever was last passed to setTitle so tests +
        # post-swap rebind logic can read back the displayed message.
        self._title_message: str | None = None
        self._install_click_recenter()
        self._install_cursor_markers()
        self._install_position_trace()
        self._overlay_items: list[pg.GraphicsObject] = []

    def update_window(self, payload: WindowPayload) -> None:
        if payload.likelihood is None:
            self._image_item.clear()
            self._clear_position_trace()
            self._set_title_message(self.MISSING_DATA_MESSAGE)
            return
        collapsed = self._model.update_window(payload.likelihood)
        self._set_image(collapsed, payload.time)
        self._set_position_trace(payload.time, payload.position)
        self._set_title_message(None)

    def update_for_array(self, time: np.ndarray, log_lik: np.ndarray) -> None:
        collapsed = self._model.update_window(log_lik)
        self._set_image(collapsed, np.asarray(time))
        self._set_title_message(None)

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind the y-axis position grid (called on M-key swap)."""
        self._set_position_grid(centers)

    def _set_position_grid(self, centers: np.ndarray) -> None:
        """Cache layout for ``setRect`` + the position trace.

        See ``QtPosteriorHeatmapPanel._set_position_grid`` for the
        half-bin-pad convention.
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

    def _set_title_message(self, message: str | None) -> None:
        """Show ``message`` in the title bar; pass ``None`` to clear.

        ``pg.PlotWidget.setTitle(None)`` only hides the label; it does
        not touch the cached text. If we relied on that alone, any
        future code path that re-shows the label without resetting
        text would surface the stale warning. Clear the text first
        (via empty-string ``setTitle``) and then hide so both the
        ``titleLabel.text`` and the visible bar end up empty.
        """
        if self._title_message == message:
            return
        self._title_message = message
        if message is None:
            self.setTitle("")
            self.setTitle(None)
        else:
            self.setTitle(message)

    def _set_image(self, collapsed: np.ndarray, time: np.ndarray) -> None:
        self._image_item.setImage(
            collapsed.T,
            autoLevels=False,
            levels=(0.0, self._vmax),
            autoDownsample=False,
        )
        if time.size and self._position_centers.size:
            x_min = float(time[0])
            x_extent = float(time[-1] - time[0]) if time.size > 1 else 1.0
            y_min = self._y0 - self._dy_half
            y_extent = (self._y1 - self._y0) + 2 * self._dy_half
            self._image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_extent, y_extent))
