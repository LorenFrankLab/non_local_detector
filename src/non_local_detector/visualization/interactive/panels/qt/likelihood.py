"""``QtLikelihoodHeatmapPanel`` — pyqtgraph rendering of the collapsed log-likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtWidgets

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    HeatmapPanelBase,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.likelihood import (
        LikelihoodHeatmapModel,
    )


class QtLikelihoodHeatmapPanel(HeatmapPanelBase):
    """Time × position heatmap of the population log-likelihood.

    When the active run was produced without ``log_likelihood``
    (default ``predict()`` call), the heatmap blanks and a wrapping
    QLabel overlay surfaces ``MISSING_DATA_MESSAGE`` explaining the
    re-``predict`` step needed to enable it. The overlay (rather than
    a pyqtgraph title) is used so the message word-wraps and stays
    legible at any panel width — the title bar truncates instructions.
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
        super().__init__(
            position_centers=position_centers,
            vmax=vmax,
            bottom_label="Likelihood (peak-normalised, all spatial states)",
            parent=parent,
        )
        self._model = model
        self._title_message: str | None = None
        self._missing_label = QtWidgets.QLabel("", parent=self)
        self._missing_label.setAlignment(QtCore.Qt.AlignCenter)
        self._missing_label.setWordWrap(True)
        self._missing_label.setStyleSheet(
            "QLabel {"
            " color: rgb(60, 60, 60);"
            " background: rgba(255, 255, 255, 220);"
            " border: 1px solid rgb(180, 180, 180);"
            " padding: 10px;"
            " font-size: 11pt;"
            "}"
        )
        self._missing_label.setVisible(False)

    def update_window(self, payload: WindowPayload) -> None:
        if payload.likelihood is None:
            self._image_item.clear()
            self._clear_position_trace()
            self._set_title_message(self.MISSING_DATA_MESSAGE)
            return
        collapsed = self._model.update_window(payload.likelihood)
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
        self._set_title_message(None)

    def update_for_array(self, time: np.ndarray, log_lik: np.ndarray) -> None:
        collapsed = self._model.update_window(log_lik)
        self._set_image(collapsed, np.asarray(time))
        self._set_title_message(None)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt naming convention
        super().resizeEvent(event)
        # ``super().__init__`` issues resize events before ``__init__``
        # creates ``self._missing_label``; ``getattr(...)`` skips those.
        label = getattr(self, "_missing_label", None)
        if label is not None and label.isVisible():
            self._reposition_missing_label()

    def _reposition_missing_label(self) -> None:
        w, h = self.width(), self.height()
        target_w = min(max(w - 80, 200), 480)
        target_h = self._missing_label.heightForWidth(target_w)
        if target_h <= 0:
            target_h = 60
        x = max(0, (w - target_w) // 2)
        y = max(0, (h - target_h) // 2)
        self._missing_label.setGeometry(x, y, target_w, target_h)

    def _set_title_message(self, message: str | None) -> None:
        """Show or clear the missing-data overlay."""
        if self._title_message == message:
            return
        self._title_message = message
        if message is None:
            self._missing_label.setText("")
            self._missing_label.setVisible(False)
        else:
            self._missing_label.setText(message)
            self._reposition_missing_label()
            self._missing_label.setVisible(True)
            self._missing_label.raise_()
