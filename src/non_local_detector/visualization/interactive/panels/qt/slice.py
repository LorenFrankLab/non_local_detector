"""``QtSlicePanel`` — right-column per-bin slice readout.

Maintains a panel-side window cache (``set_window_buffer(payload)``)
mirroring the upstream statespacecheck pattern: per-tick
``update_for_index(t_idx)`` indexes into the cached arrays instead of
re-fetching from the data source. The viewer drives both window
loads (via the backend) and per-tick cursor updates (via the slider);
the buffer makes the cursor updates O(1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.slice import (
        SliceModel,
    )


# Pre-allocated per-cell row pool size. Cells beyond this threshold
# fall under the "(+K more)" truncation indicator. Mirrors upstream
# (panels.py:95).
MAX_PER_CELL_PLOTS = 6

_TOP_CURVE_PEN = pg.mkPen(color="#1f77b4", width=2)
_PREDICTIVE_PEN = pg.mkPen(
    color="#ff7f0e", width=1, style=QtCore.Qt.PenStyle.DashLine
)
_PER_CELL_PEN = pg.mkPen(color="#444444", width=1)


class _PerCellRow:
    """One pre-allocated per-cell row widget (label + tiny line plot)."""

    def __init__(self, position_centers: np.ndarray) -> None:
        self.container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = QtWidgets.QLabel("")
        self.label.setMinimumWidth(80)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMaximumHeight(40)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        _empty = np.empty(0, dtype=float)
        self.curve = self.plot.plot(_empty, _empty, pen=_PER_CELL_PEN)
        layout.addWidget(self.label)
        layout.addWidget(self.plot, stretch=1)
        self._position_centers = np.asarray(position_centers).squeeze()
        self.container.setVisible(False)

    def show_cell(self, label: str, place_field_norm: np.ndarray) -> None:
        self.label.setText(label)
        self.curve.setData(self._position_centers, place_field_norm)
        self.container.setVisible(True)

    def hide(self) -> None:
        self.container.setVisible(False)


class QtSlicePanel(QtWidgets.QWidget):
    """Right-column slice panel: top curve + predictive overlay + per-cell rows.

    The viewer calls ``set_window_buffer(payload)`` whenever a new
    ``WindowPayload`` commits, then ``update_for_index(t_idx)`` on
    every cursor tick. Out-of-buffer ``t_idx`` is a no-op (the next
    window load will refresh the buffer).
    """

    def __init__(
        self,
        model: SliceModel,
        position_centers: np.ndarray,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._position_centers = np.asarray(position_centers).squeeze()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._title_label = QtWidgets.QLabel("")
        self._title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._title_label)

        self._top_plot = pg.PlotWidget(background="w")
        self._top_plot.setLabel("left", "Probability / Likelihood")
        self._top_plot.setLabel("bottom", "Position [cm]")
        self._top_plot.setMouseEnabled(x=False, y=False)
        # Pre-init with explicit empty arrays so ``getData()`` always
        # returns ndarrays (pyqtgraph returns ``(None, None)`` when
        # data was set with empty Python lists or never set at all).
        _empty = np.empty(0, dtype=float)
        self._top_curve_item = self._top_plot.plot(_empty, _empty, pen=_TOP_CURVE_PEN)
        self._predictive_curve_item = self._top_plot.plot(
            _empty, _empty, pen=_PREDICTIVE_PEN
        )
        layout.addWidget(self._top_plot, stretch=2)

        # Pre-allocate the per-cell row pool up to MAX_PER_CELL_PLOTS.
        # Rows are hidden until ``update_for_index`` activates them; this
        # keeps per-tick rendering allocation-free.
        self._per_cell_rows: list[_PerCellRow] = [
            _PerCellRow(self._position_centers) for _ in range(MAX_PER_CELL_PLOTS)
        ]
        for row in self._per_cell_rows:
            layout.addWidget(row.container)

        self._truncation_label = QtWidgets.QLabel("")
        self._truncation_label.setStyleSheet("color: #888;")
        self._truncation_label.setVisible(False)
        layout.addWidget(self._truncation_label)

        layout.addStretch(1)

        self._buffered_payload: WindowPayload | None = None

    @property
    def model(self) -> SliceModel:
        return self._model

    def set_window_buffer(self, payload: WindowPayload) -> None:
        """Cache the latest window payload for per-tick row reads."""
        self._buffered_payload = payload

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind the position grid (called on M-key swap)."""
        self._position_centers = np.asarray(centers).squeeze()
        for row in self._per_cell_rows:
            row._position_centers = self._position_centers  # noqa: SLF001

    def rebind_after_swap(self) -> None:
        """Drop the stale buffer after the model schema changes."""
        self._buffered_payload = None
        self._top_curve_item.setData(np.empty(0, dtype=float), np.empty(0, dtype=float))
        self._predictive_curve_item.setData(np.empty(0, dtype=float), np.empty(0, dtype=float))
        for row in self._per_cell_rows:
            row.hide()
        self._truncation_label.setVisible(False)
        self._title_label.setText("")

    def update_for_index(self, t_idx: int) -> None:
        """Read row ``t_idx`` from the buffered window + render the slice.

        Out-of-buffer ``t_idx`` is a no-op: the viewer's next window
        load will refresh the buffer and re-issue the cursor update.
        """
        payload = self._buffered_payload
        if payload is None:
            return
        sl = payload.indices
        if t_idx < sl.start or t_idx >= sl.stop:
            return
        local_idx = t_idx - sl.start
        if payload.posterior is None:
            return
        posterior_row = payload.posterior[local_idx]
        log_lik_row = (
            payload.likelihood[local_idx] if payload.likelihood is not None else None
        )
        predictive_row = (
            payload.predictive[local_idx] if payload.predictive is not None else None
        )
        bin_payload = self._model.update_for_index(
            t_idx, posterior_row, log_lik_row=log_lik_row, predictive_row=predictive_row
        )
        self._render(bin_payload)

    def _render(self, bin_payload) -> None:
        self._title_label.setText(
            f"t={bin_payload.t:.3f} s — {bin_payload.top_curve_label}"
        )
        if bin_payload.top_curve is not None:
            self._top_curve_item.setData(
                self._position_centers, bin_payload.top_curve
            )
        else:
            self._top_curve_item.setData(np.empty(0, dtype=float), np.empty(0, dtype=float))
        if bin_payload.predictive_curve is not None:
            self._predictive_curve_item.setData(
                self._position_centers, bin_payload.predictive_curve
            )
        else:
            self._predictive_curve_item.setData(np.empty(0, dtype=float), np.empty(0, dtype=float))
        self._render_per_cell_rows(bin_payload.cells)

    def _render_per_cell_rows(self, cells) -> None:
        n_total = len(cells)
        n_shown = min(n_total, MAX_PER_CELL_PLOTS)
        for i in range(n_shown):
            cell = cells[i]
            row = self._per_cell_rows[i]
            row.show_cell(
                f"#{cell.cell_id}  (×{cell.spike_count})",
                cell.place_field_norm,
            )
        for i in range(n_shown, MAX_PER_CELL_PLOTS):
            self._per_cell_rows[i].hide()
        if n_total > MAX_PER_CELL_PLOTS:
            self._truncation_label.setText(
                f"(+{n_total - MAX_PER_CELL_PLOTS} more)"
            )
            self._truncation_label.setVisible(True)
        else:
            self._truncation_label.setVisible(False)
