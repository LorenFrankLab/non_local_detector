"""``QtSlicePanel`` — right-column per-bin slice readout.

Maintains a panel-side window cache (``set_window_buffer(payload)``)
mirroring the upstream statespacecheck pattern: per-tick
``update_for_index(t_idx)`` indexes into the cached arrays instead of
re-fetching from the data source. The viewer drives both window
loads (via the backend) and per-tick cursor updates (via the slider);
the buffer makes the cursor updates O(1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

OverlayMode = Literal["predictive", "smoothed", "off"]
_OVERLAY_MODE_CHOICES: tuple[tuple[OverlayMode, str], ...] = (
    ("predictive", "Predictive (causal)"),
    ("smoothed", "Smoothed (acausal)"),
    ("off", "Off"),
)

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
        overlay_mode: OverlayMode = "predictive",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._position_centers = np.asarray(position_centers).squeeze()
        self._overlay_mode: OverlayMode = overlay_mode

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Title row: bold prose on the left, overlay-source dropdown on the
        # right. The dropdown lets the user choose between the predictive
        # (causal) overlay — the prior the decoder used at this bin — and
        # the smoothed (acausal) overlay derived from the same
        # ``acausal_posterior`` the heatmap shows. See
        # docs/plans/2026-05-06-interactive-decoder-viewer.md (Milestone 6).
        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self._title_label = QtWidgets.QLabel("")
        self._title_label.setStyleSheet("font-weight: bold;")
        title_row.addWidget(self._title_label, stretch=1)
        title_row.addWidget(QtWidgets.QLabel("Overlay:"))
        self._overlay_combo = QtWidgets.QComboBox()
        for mode_key, mode_label in _OVERLAY_MODE_CHOICES:
            self._overlay_combo.addItem(mode_label, userData=mode_key)
        self._set_combo_to_mode(overlay_mode)
        self._overlay_combo.currentIndexChanged.connect(self._on_overlay_combo_changed)
        title_row.addWidget(self._overlay_combo)
        layout.addLayout(title_row)

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
        # Last successfully-rendered ``t_idx`` so pin/unpin can
        # re-render in place without waiting for a slider tick.
        self._last_t_idx: int | None = None
        # Pinned cells survive bin-window scrubbing — the panel keeps
        # showing them with ``spike_count=0`` (or the real count when
        # they're also active). Cell IDs are run-local, so swap clears.
        self._pinned_cell_ids: set[int] = set()

    @property
    def model(self) -> SliceModel:
        return self._model

    @property
    def overlay_mode(self) -> OverlayMode:
        return self._overlay_mode

    def set_overlay_mode(self, mode: OverlayMode) -> None:
        """Switch the overlay source between predictive / smoothed / off.

        ``"predictive"`` draws the predictive (causal) posterior collapsed
        via the active reduction — the prior the decoder used at the
        cursor bin. Hidden when ``predictive_posterior`` is missing from
        the buffered window.

        ``"smoothed"`` draws the acausal posterior collapsed the same
        way. Always available because ``acausal_posterior`` is always in
        a default ``predict()`` output.

        ``"off"`` hides the overlay entirely.

        Updates the dropdown to match and re-renders at the last bin so
        the change is visible without a slider nudge.
        """
        if mode not in {"predictive", "smoothed", "off"}:
            raise ValueError(
                f"overlay_mode must be 'predictive', 'smoothed', or 'off'; got {mode!r}"
            )
        if mode == self._overlay_mode:
            return
        self._overlay_mode = mode
        self._set_combo_to_mode(mode)
        self._maybe_rerender()

    def _set_combo_to_mode(self, mode: OverlayMode) -> None:
        """Sync the combo box to ``mode`` without firing the signal."""
        for i, (mode_key, _label) in enumerate(_OVERLAY_MODE_CHOICES):
            if mode_key == mode:
                with QtCore.QSignalBlocker(self._overlay_combo):
                    self._overlay_combo.setCurrentIndex(i)
                return

    def _on_overlay_combo_changed(self, idx: int) -> None:
        if not 0 <= idx < len(_OVERLAY_MODE_CHOICES):
            return
        new_mode: OverlayMode = _OVERLAY_MODE_CHOICES[idx][0]
        if new_mode == self._overlay_mode:
            return
        self._overlay_mode = new_mode
        self._maybe_rerender()

    @property
    def pinned_cell_ids(self) -> frozenset[int]:
        """Snapshot of currently pinned cell IDs (read-only view)."""
        return frozenset(self._pinned_cell_ids)

    def pin_cell(self, cell_id: int) -> None:
        """Pin ``cell_id`` so it stays visible across bin scrubbing."""
        self._model.cell_slice(cell_id)  # validates bounds; raises on bad id
        self._pinned_cell_ids.add(cell_id)
        self._maybe_rerender()

    def unpin_cell(self, cell_id: int) -> None:
        """Remove ``cell_id`` from the pin set (no-op if not pinned)."""
        self._pinned_cell_ids.discard(cell_id)
        self._maybe_rerender()

    def toggle_pin(self, cell_id: int) -> None:
        """Flip pin state for ``cell_id``."""
        self._model.cell_slice(cell_id)  # validate before flipping
        if cell_id in self._pinned_cell_ids:
            self._pinned_cell_ids.discard(cell_id)
        else:
            self._pinned_cell_ids.add(cell_id)
        self._maybe_rerender()

    def clear_pins(self) -> None:
        """Drop all pins."""
        if not self._pinned_cell_ids:
            return
        self._pinned_cell_ids.clear()
        self._maybe_rerender()

    def set_window_buffer(self, payload: WindowPayload) -> None:
        """Cache the latest window payload for per-tick row reads."""
        self._buffered_payload = payload

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind the position grid (called on M-key swap)."""
        self._position_centers = np.asarray(centers).squeeze()
        for row in self._per_cell_rows:
            row._position_centers = self._position_centers  # noqa: SLF001

    def rebind_after_swap(self) -> None:
        """Drop the stale buffer after the model schema changes.

        Pins are cleared because cell IDs are run-local — the new
        run's neuron 0 is a different physical cell from the old
        run's neuron 0, and silently re-applying pins across that
        boundary would surface the wrong place fields.
        """
        self._buffered_payload = None
        self._last_t_idx = None
        self._pinned_cell_ids.clear()
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
        # Overlay row depends on the user-selected mode: predictive picks
        # the causal-prior posterior, smoothed picks the same acausal row
        # the heatmap collapses, off hides the overlay. Each falls back
        # to ``None`` (overlay hidden) when the chosen array isn't in the
        # window — predictive_posterior is opt-in and therefore commonly
        # absent.
        if self._overlay_mode == "predictive":
            overlay_row = (
                payload.predictive[local_idx]
                if payload.predictive is not None
                else None
            )
        elif self._overlay_mode == "smoothed":
            overlay_row = posterior_row
        else:  # "off"
            overlay_row = None
        bin_payload = self._model.update_for_index(
            t_idx, posterior_row, log_lik_row=log_lik_row, predictive_row=overlay_row
        )
        self._last_t_idx = t_idx
        self._render(bin_payload)

    def _maybe_rerender(self) -> None:
        """Re-render at the last bin if the buffer is still around.

        Pin/unpin/clear call this so the user sees the change without
        having to nudge the slider. No-op when no payload has been
        loaded yet — the pin set is still recorded; the next
        ``update_for_index`` will surface it.
        """
        if self._last_t_idx is None or self._buffered_payload is None:
            return
        self.update_for_index(self._last_t_idx)

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
        merged = self._merge_pinned_and_active(cells)
        n_total = len(merged)
        n_shown = min(n_total, MAX_PER_CELL_PLOTS)
        pinned = self._pinned_cell_ids
        for i in range(n_shown):
            cell = merged[i]
            row = self._per_cell_rows[i]
            star = " ★" if cell.cell_id in pinned else ""
            row.show_cell(
                f"#{cell.cell_id}{star}  (×{cell.spike_count})",
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

    def _merge_pinned_and_active(self, active_cells) -> list:
        """Pinned cells first (sorted by cell_id), then active not-already-pinned.

        When a cell is both pinned and active, the active CellSlice
        wins so the spike-count is the real (>0) count from the bin
        rather than a placeholder 0.
        """
        active_by_id = {c.cell_id: c for c in active_cells}
        pinned_ids = self._pinned_cell_ids
        merged: list = []
        for cell_id in sorted(pinned_ids):
            if cell_id in active_by_id:
                merged.append(active_by_id[cell_id])
            else:
                merged.append(self._model.cell_slice(cell_id))
        for cell in active_cells:
            if cell.cell_id not in pinned_ids:
                merged.append(cell)
        return merged
