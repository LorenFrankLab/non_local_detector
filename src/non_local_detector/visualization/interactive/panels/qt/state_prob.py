"""``QtStateProbabilityPanel`` — multi-line per-state probability plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    CursorMarkersMixin,
    EventOverlayMixin,
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

_LEGEND_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-size: 11pt; }"
)


def _legend_html(items: list[tuple[str, str]]) -> str:
    """Build the HTML for a sidebar legend from ``[(color, label), ...]``."""
    rows = [
        f"<span style='color:{color};font-size:14pt'>━</span> {label}"
        for color, label in items
    ]
    return "<br>".join(rows)


class _StateProbabilityPlot(
    pg.PlotWidget, EventOverlayMixin, ClickRecenterMixin, CursorMarkersMixin
):
    """Internal plot widget — one ``pg.PlotDataItem`` line per discrete state.

    Owns the lines + mixin behaviours; the parent wrapper
    ``QtStateProbabilityPanel`` lays it out next to a sidebar legend.
    """

    def __init__(self, model: StateProbabilityModel, parent=None) -> None:
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
        for i, state_name in enumerate(self._model.state_names):
            color = _DEFAULT_STATE_COLORS[i % len(_DEFAULT_STATE_COLORS)]
            line = self.plot(
                [], [], pen=pg.mkPen(color=color, width=3), name=state_name
            )
            self._lines.append(line)

    def update_window(self, payload: WindowPayload) -> None:
        if payload.state_probabilities is None:
            for line in self._lines:
                line.setData([], [])
            return
        data = self._model.update_window(payload.state_probabilities)
        self._set_data(payload.time, data)

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
        """Rebuild lines after the model's ``set_active_run`` runs.

        Keeps the panel consistent with the new schema before the
        next payload arrives.
        """
        self._build_lines()


class QtStateProbabilityPanel(QtWidgets.QWidget):
    """State-probability plot with the legend laid out *outside* the plot.

    A pyqtgraph ``LegendItem`` floats inside the viewbox; on small
    panels it can occlude the data. This wrapper places the plot in
    one column and an HTML legend ``QLabel`` in a sidebar so the
    legend never overlaps the curves. Mixin / pyqtgraph methods that
    callers expect on the panel (``addItem``, ``set_event_overlays``,
    ``viewport``, ``scene``, ``getPlotItem``, ``set_cursor_markers``,
    ``click_handler``, ``x_link_target``) forward to the inner plot
    widget via ``__getattr__``.
    """

    def __init__(
        self,
        model: StateProbabilityModel,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._plot = _StateProbabilityPlot(model)
        self._legend = QtWidgets.QLabel()
        self._legend.setStyleSheet(_LEGEND_STYLE)
        self._legend.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self._legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._refresh_legend()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._plot, stretch=1)
        layout.addWidget(self._legend, stretch=0)

    def __getattr__(self, name: str):
        # Forward unknown attributes to the inner plot widget. ``_plot``
        # is set in ``__init__`` before any forwarding happens; the
        # ``__dict__`` check avoids infinite recursion if Qt's metaclass
        # asks for an attribute mid-construction.
        plot = self.__dict__.get("_plot")
        if plot is None:
            raise AttributeError(name)
        return getattr(plot, name)

    def rebind_after_swap(self) -> None:
        """Rebuild lines + legend after the model schema changes."""
        self._plot.rebind_after_swap()
        self._refresh_legend()

    def _refresh_legend(self) -> None:
        items: list[tuple[str, str]] = [
            (
                _DEFAULT_STATE_COLORS[i % len(_DEFAULT_STATE_COLORS)],
                state_name,
            )
            for i, state_name in enumerate(self._plot._model.state_names)
        ]
        self._legend.setText(_legend_html(items))
