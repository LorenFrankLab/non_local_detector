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

OverlayMode = Literal["predictive", "filtered", "smoothed"]
_OVERLAY_MODE_CHOICES: tuple[tuple[OverlayMode, str], ...] = (
    ("predictive", "Predictive (causal)"),
    ("filtered", "Filtered"),
    ("smoothed", "Smoothed (acausal)"),
)

_TOP_CURVE_PEN = pg.mkPen(color="#1f77b4", width=2)
_PREDICTIVE_PEN = pg.mkPen(color="#ff7f0e", width=1, style=QtCore.Qt.PenStyle.DashLine)
_PER_CELL_PEN = pg.mkPen(color="#444444", width=1)
_SLICE_Y_MIN = -0.02
_SLICE_Y_MAX = 1.05
_PER_CELL_PALETTE = (
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (227, 119, 194),
    (23, 190, 207),
    (140, 86, 75),
)
_SLICE_LEGEND_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-size: 11pt; }"
)
_SLICE_READOUT_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 11pt; }"
)
_SLICE_CELL_HEADER_STYLE = (
    "QLabel { background-color: #f4f4f4; color: #202020; "
    "padding: 2px 6px; border: 1px solid #d8d8d8; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 10pt; }"
)
_SLICE_CELL_HEADER_PINNED_STYLE = (
    "QLabel { background-color: #fff2a8; color: #4d3700; "
    "padding: 2px 6px; border: 2px solid #d4b85a; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 10pt; "
    "font-weight: bold; }"
)


def _pin_slice_axes(plot: pg.PlotWidget, position_centers: np.ndarray) -> None:
    """Keep slice subplots from auto-ranging/reflowing on every tick."""
    centers = np.asarray(position_centers, dtype=float).squeeze()
    if centers.size == 0:
        return
    x_min = float(np.nanmin(centers))
    x_max = float(np.nanmax(centers))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    vb = plot.getViewBox()
    vb.disableAutoRange()
    vb.setXRange(x_min, x_max, padding=0)
    vb.setYRange(_SLICE_Y_MIN, _SLICE_Y_MAX, padding=0)
    vb.setLimits(
        xMin=x_min,
        xMax=x_max,
        yMin=_SLICE_Y_MIN,
        yMax=_SLICE_Y_MAX,
    )


class _PerCellRow:
    """One pre-allocated per-cell row widget (label + tiny line plot)."""

    def __init__(self, position_centers: np.ndarray) -> None:
        self.container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = QtWidgets.QLabel("")
        self.label.setMinimumWidth(80)
        self.label.setStyleSheet(_SLICE_CELL_HEADER_STYLE)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMaximumHeight(40)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        _pin_slice_axes(self.plot, position_centers)
        _empty = np.empty(0, dtype=float)
        self.curve = self.plot.plot(_empty, _empty, pen=_PER_CELL_PEN)
        layout.addWidget(self.label)
        layout.addWidget(self.plot, stretch=1)
        self._position_centers = np.asarray(position_centers).squeeze()
        self._last_label = ""
        size_policy = self.container.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.container.setSizePolicy(size_policy)
        self.container.setVisible(False)

    def show_cell(self, label: str, place_field_norm: np.ndarray) -> None:
        if label != self._last_label:
            self.label.setText(label)
            self.curve.setData(self._position_centers, place_field_norm)
            self._last_label = label
        if not self.container.isVisible():
            self.container.setVisible(True)

    def set_position_centers(self, centers: np.ndarray) -> None:
        self._position_centers = np.asarray(centers).squeeze()
        _pin_slice_axes(self.plot, self._position_centers)
        self._last_label = ""

    def hide(self) -> None:
        if self.container.isVisible():
            self.container.setVisible(False)
        self._last_label = ""


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
        overlay_mode: OverlayMode = "smoothed",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._position_centers = np.asarray(position_centers).squeeze()
        self._overlay_mode: OverlayMode = overlay_mode
        self._per_cell_visible = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Title row: bold prose on the left, overlay-source dropdown on the
        # right. The dropdown lets the user choose predictive (causal),
        # filtered, or smoothed (acausal) collapsed overlays. See
        # docs/plans/2026-05-06-interactive-decoder-viewer.md (Milestone 6).
        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self._title_label = QtWidgets.QLabel("")
        self._title_label.setStyleSheet("font-weight: bold;")
        self._title_label.setMinimumWidth(0)
        self._title_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._title_label.setText("Slice")
        title_row.addWidget(self._title_label, stretch=1)
        title_row.addWidget(QtWidgets.QLabel("Overlay:"))
        self._overlay_combo = QtWidgets.QComboBox()
        for mode_key, mode_label in _OVERLAY_MODE_CHOICES:
            self._overlay_combo.addItem(mode_label, userData=mode_key)
        self._set_combo_to_mode(overlay_mode)
        self._overlay_combo.currentIndexChanged.connect(self._on_overlay_combo_changed)
        title_row.addWidget(self._overlay_combo)
        layout.addLayout(title_row)

        self._legend_label = QtWidgets.QLabel(self._build_legend_html())
        self._legend_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._legend_label.setWordWrap(True)
        self._legend_label.setStyleSheet(_SLICE_LEGEND_STYLE)
        layout.addWidget(self._legend_label)

        self._top_plot = pg.PlotWidget(background="w")
        self._top_plot.setLabel("left", "Probability / Likelihood")
        self._top_plot.setLabel("bottom", "Position [cm]")
        self._top_plot.setMouseEnabled(x=False, y=False)
        _pin_slice_axes(self._top_plot, self._position_centers)
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

        self._readout_label = QtWidgets.QLabel("")
        self._readout_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._readout_label.setStyleSheet(_SLICE_READOUT_STYLE)
        layout.addWidget(self._readout_label)

        layout.addStretch(1)

        self._buffered_payload: WindowPayload | None = None
        # Last successfully-rendered ``t_idx`` so pin/unpin can
        # re-render in place without waiting for a slider tick.
        self._last_t_idx: int | None = None
        # Pinned cells survive bin-window scrubbing — the panel keeps
        # showing them with ``spike_count=0`` (or the real count when
        # they're also active). Cell IDs are run-local, so swap clears.
        self._pinned_cell_ids: set[int] = set()

    @staticmethod
    def _build_legend_html() -> str:
        return (
            "<span style='color:rgb(255,127,14);font-size:14pt'>━</span> "
            "Likelihood &nbsp;&nbsp; "
            "<span style='color:rgb(31,119,180);font-size:14pt'>━</span> "
            "Overlay &nbsp;&nbsp; "
            "<span style='color:rgb(44,160,44);font-size:14pt'>━</span> "
            "Cell place fields"
        )

    @property
    def model(self) -> SliceModel:
        return self._model

    @property
    def overlay_mode(self) -> OverlayMode:
        return self._overlay_mode

    def set_overlay_mode(self, mode: OverlayMode) -> None:
        """Switch the overlay source between predictive / filtered / smoothed.

        ``"predictive"`` draws the predictive (causal) posterior collapsed
        via the active reduction — the prior the decoder used at the
        cursor bin. Hidden when ``predictive_posterior`` is missing from
        the buffered window.

        ``"filtered"`` draws the one-bin filtered posterior,
        proportional to predictive posterior × likelihood, collapsed
        the same way. Hidden when either source is missing.

        ``"smoothed"`` draws the acausal posterior collapsed the same
        way. Always available because ``acausal_posterior`` is always in
        a default ``predict()`` output.

        Updates the dropdown to match and re-renders at the last bin so
        the change is visible without a slider nudge.
        """
        if mode not in {"predictive", "filtered", "smoothed"}:
            raise ValueError(
                "overlay_mode must be 'predictive', 'filtered', or "
                f"'smoothed'; got {mode!r}"
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

    def set_per_cell_visible(self, visible: bool) -> None:
        """Show/hide per-cell rows without disabling the population slice."""
        self._per_cell_visible = bool(visible)
        self._maybe_rerender()

    def set_position_centers(self, centers: np.ndarray) -> None:
        """Re-bind the position grid (called on M-key swap)."""
        self._position_centers = np.asarray(centers).squeeze()
        _pin_slice_axes(self._top_plot, self._position_centers)
        for row in self._per_cell_rows:
            row.set_position_centers(self._position_centers)

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
        self._predictive_curve_item.setData(
            np.empty(0, dtype=float), np.empty(0, dtype=float)
        )
        for row in self._per_cell_rows:
            row.hide()
        self._truncation_label.setVisible(False)
        self._title_label.setText("Slice")
        self._readout_label.setText("")

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
        # Overlay row depends on the user-selected mode. Predictive is
        # the causal prior, filtered is predictive × likelihood, and
        # smoothed is the acausal row the heatmap collapses. Predictive
        # posterior is opt-in and therefore commonly absent.
        predictive_row = (
            payload.predictive[local_idx] if payload.predictive is not None else None
        )
        if self._overlay_mode == "predictive":
            overlay_row = predictive_row
        elif self._overlay_mode == "filtered":
            overlay_row = _filtered_row(predictive_row, log_lik_row)
        elif self._overlay_mode == "smoothed":
            overlay_row = posterior_row
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
        self._readout_label.setText(
            f"t={bin_payload.t:.3f} s    cells={len(bin_payload.cells)}    "
            f"{bin_payload.top_curve_label}"
        )
        if bin_payload.top_curve is not None:
            self._top_curve_item.setData(self._position_centers, bin_payload.top_curve)
        else:
            self._top_curve_item.setData(
                np.empty(0, dtype=float), np.empty(0, dtype=float)
            )
        if bin_payload.predictive_curve is not None:
            self._predictive_curve_item.setData(
                self._position_centers, bin_payload.predictive_curve
            )
        else:
            self._predictive_curve_item.setData(
                np.empty(0, dtype=float), np.empty(0, dtype=float)
            )
        self._render_per_cell_rows(bin_payload.cells)

    def _render_per_cell_rows(self, cells) -> None:
        merged = self._merge_pinned_and_active(cells)
        if not self._per_cell_visible:
            for row in self._per_cell_rows:
                row.hide()
            self._truncation_label.setVisible(False)
            return
        n_total = len(merged)
        n_shown = min(n_total, MAX_PER_CELL_PLOTS)
        pinned = self._pinned_cell_ids
        for i in range(n_shown):
            cell = merged[i]
            row = self._per_cell_rows[i]
            star = " ★" if cell.cell_id in pinned else ""
            rgb = _PER_CELL_PALETTE[cell.cell_id % len(_PER_CELL_PALETTE)]
            row.curve.setPen(pg.mkPen(color=rgb, width=2))
            row.label.setStyleSheet(
                _SLICE_CELL_HEADER_PINNED_STYLE
                if cell.cell_id in pinned
                else _SLICE_CELL_HEADER_STYLE
            )
            row.show_cell(
                f"#{cell.cell_id}{star}  (×{cell.spike_count})",
                cell.place_field_norm,
            )
        for i in range(n_shown, MAX_PER_CELL_PLOTS):
            self._per_cell_rows[i].hide()
        if n_total > MAX_PER_CELL_PLOTS:
            self._truncation_label.setText(f"(+{n_total - MAX_PER_CELL_PLOTS} more)")
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


def _filtered_row(
    predictive_row: np.ndarray | None, log_lik_row: np.ndarray | None
) -> np.ndarray | None:
    """Return normalized ``predictive * likelihood`` in state-bin space."""
    if predictive_row is None or log_lik_row is None:
        return None
    predictive = np.asarray(predictive_row, dtype=float)
    log_lik = np.asarray(log_lik_row, dtype=float)
    if predictive.shape != log_lik.shape:
        return None
    finite = np.isfinite(log_lik)
    if not finite.any():
        return None
    likelihood = np.zeros_like(log_lik, dtype=float)
    likelihood[finite] = np.exp(log_lik[finite] - np.max(log_lik[finite]))
    filtered = np.nan_to_num(predictive, nan=0.0, posinf=0.0, neginf=0.0) * likelihood
    total = float(filtered.sum())
    if total <= 0.0:
        return None
    return filtered / total
