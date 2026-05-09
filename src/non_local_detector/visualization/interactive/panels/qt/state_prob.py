"""``QtStateProbabilityPanel`` — multi-line per-state probability plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
    RelativeTimeAxisMixin,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.state_prob import (
        StateProbabilityModel,
    )


# Default per-line colors. Cycled when the schema has more states than
# entries in this list.
_DEFAULT_STATE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)


def _legend_html(items: list[tuple[str, str]]) -> str:
    """Inline legend rendered into the plot's ``setTitle`` slot.

    Single line so it fits in the title bar (which has a fixed height
    regardless of content). Putting the legend in the title keeps the
    panel's data area in step with other ``setTitle``-using panels
    (e.g. ``QtPosteriorHeatmapPanel``) — both reserve the same
    title-bar height inside the plot widget.
    """
    return " &nbsp;&nbsp;&nbsp; ".join(
        f"<span style='color:{color};font-size:11pt'>━</span> {label}"
        for color, label in items
    )


class QtStateProbabilityPanel(
    pg.PlotWidget,
    EventOverlayMixin,
    ClickRecenterMixin,
    CursorMarkersMixin,
    RelativeTimeAxisMixin,
):
    """One ``pg.PlotDataItem`` line per discrete state.

    The legend is rendered into the plot's title bar (HTML-coloured
    line markers + state names) instead of pyqtgraph's floating
    ``LegendItem`` so it never occludes the curves.
    """

    def __init__(
        self,
        model: StateProbabilityModel,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        self._model = model
        self.setLabel("left", "State probability")
        self.setLabel("bottom", "Time [s]")
        self.setYRange(0.0, 1.05)
        self._lines: list[pg.PlotDataItem] = []
        self._build_lines()
        self._install_click_recenter()
        self._install_cursor_markers()
        self._overlay_items: list[pg.GraphicsObject] = []

    def _build_lines(self) -> None:
        for line in self._lines:
            self.removeItem(line)
        self._lines = []
        legend_items: list[tuple[str, str]] = []
        for i, state_name in enumerate(self._model.state_names):
            color = _DEFAULT_STATE_COLORS[i % len(_DEFAULT_STATE_COLORS)]
            line = self.plot(
                [], [], pen=pg.mkPen(color=color, width=3), name=state_name
            )
            self._lines.append(line)
            legend_items.append((color, state_name))
        self.setTitle(_legend_html(legend_items))

    def update_window(self, payload: WindowPayload) -> None:
        if payload.state_probabilities is None:
            for line in self._lines:
                line.setData([], [])
            return
        data = self._model.update_window(payload.state_probabilities)
        # Render at relative coords against the panel's fixed x-range.
        self._set_data(payload.time - payload.t_center, data)

    def update_for_array(
        self, time: np.ndarray, state_probabilities: np.ndarray
    ) -> None:
        data = self._model.update_window(state_probabilities)
        self._set_data(np.asarray(time), data)

    def _set_data(self, time: np.ndarray, data: np.ndarray) -> None:
        # Schema may have changed since construction (M-key swap); rebuild lines
        # if the line count differs.
        if len(self._lines) != data.shape[1]:
            self._build_lines()
        for i, line in enumerate(self._lines):
            line.setData(time, data[:, i])

    def rebind_after_swap(self) -> None:
        """Rebuild lines + legend after the model's ``set_active_run`` runs."""
        self._build_lines()
