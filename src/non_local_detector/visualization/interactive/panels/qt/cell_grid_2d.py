"""``Qt2DCellGridPanel`` — per-cell 2D place-field rows at the cursor bin.

2D analogue of the per-cell-row stack inside ``QtSlicePanel``. For each
cell active in the cursor's time bin, render a small ``(n_y, n_x)``
viridis thumbnail of that cell's normalized place field, with a
header label naming the cell and its observed spike count.

Active cells come from ``SliceModel.update_for_index(...)``'s
``BinPayload.cells`` — same source the 1D slice panel consumes. The
panel just reshapes ``CellSlice.place_field_norm`` from ``(n_pos,)``
flat to ``(n_x, n_y)`` and routes through the shared NaN-aware
``flat_to_rgba_image`` helper so the per-cell thumbnails inherit the
same colormap + transparent-off-track rendering as the posterior /
likelihood at-cursor images above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from non_local_detector.visualization.interactive.panels.qt._image_2d import (
    flat_to_rgba_image,
    image_2d_layout_from_grid,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        PositionGrid,
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.slice import (
        SliceModel,
    )


# Match the 1D slice panel's per-cell row cap and styling so the
# 2D grid uses the same visual vocabulary (truncation label, header
# format with ``#id (×count)``).
MAX_PER_CELL_PLOTS = 6
_PER_CELL_IMAGE_MIN_HEIGHT = 100
_PER_CELL_HEADER_STYLE = (
    "QLabel { background-color: #f4f4f4; color: #202020; "
    "padding: 2px 6px; border: 1px solid #d8d8d8; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 10pt; }"
)
_TRUNCATION_LABEL_STYLE = (
    "QLabel { color: #707070; padding: 4px 6px; font-style: italic; }"
)
_TITLE_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-size: 11pt; font-weight: bold; }"
)


class _PerCellImageRow:
    """One pre-allocated per-cell row widget (header + 2D thumbnail)."""

    def __init__(self, lut: np.ndarray) -> None:
        self.container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.label = QtWidgets.QLabel("")
        self.label.setStyleSheet(_PER_CELL_HEADER_STYLE)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMinimumHeight(_PER_CELL_IMAGE_MIN_HEIGHT)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setAspectLocked(True)
        self._lut = lut
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self._image_item)
        layout.addWidget(self.label)
        layout.addWidget(self.plot)
        # Keep slot in layout when hidden so the column doesn't reflow
        # on every cursor tick.
        size_policy = self.container.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.container.setSizePolicy(size_policy)

    def apply_layout(self, grid: PositionGrid) -> None:
        """Rebind plot bounds for a new active run's grid geometry."""
        layout = image_2d_layout_from_grid(grid)
        self._image_item.setRect(
            QtCore.QRectF(layout.x_min, layout.y_min, layout.width, layout.height)
        )
        vb = self.plot.getViewBox()
        # ``setRange`` only — no ``setLimits``. Aspect-lock needs room
        # to extend the shorter visible axis when the row is non-square.
        vb.setRange(
            xRange=(layout.x_min, layout.x_max),
            yRange=(layout.y_min, layout.y_max),
            padding=0,
        )

    def show_cell(
        self, label: str, place_field_flat: np.ndarray, shape: tuple[int, int]
    ) -> None:
        self.label.setText(label)
        rgba = flat_to_rgba_image(
            np.asarray(place_field_flat), shape, self._lut, vmax=1.0
        )
        self._image_item.setImage(rgba, autoLevels=False)
        self.container.setVisible(True)

    def hide(self) -> None:
        self.container.setVisible(False)
        self._image_item.clear()

    def clear(self) -> None:
        self._image_item.clear()


class Qt2DCellGridPanel(QtWidgets.QWidget):
    """Bin-synced stack of per-cell 2D place-field thumbnails.

    Buffers the latest ``WindowPayload`` and on each cursor tick asks
    ``SliceModel`` for the active-cell list, then renders one
    thumbnail per cell (up to ``MAX_PER_CELL_PLOTS``).
    """

    def __init__(self, model: SliceModel, grid: PositionGrid, parent=None) -> None:
        super().__init__(parent=parent)
        if grid.ndim != 2 or grid.shape is None:
            raise ValueError(
                "Qt2DCellGridPanel requires a 2D PositionGrid; got "
                f"ndim={grid.ndim}, shape={grid.shape}."
            )
        self._model = model
        self._grid = grid
        self._buffered_payload: WindowPayload | None = None
        self._last_t_idx: int | None = None
        self._lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._title_label = QtWidgets.QLabel("Active cells at cursor")
        self._title_label.setStyleSheet(_TITLE_STYLE)
        layout.addWidget(self._title_label)

        self._rows: list[_PerCellImageRow] = []
        for _ in range(MAX_PER_CELL_PLOTS):
            row = _PerCellImageRow(self._lut)
            row.apply_layout(grid)
            row.hide()
            layout.addWidget(row.container)
            self._rows.append(row)

        self._truncation_label = QtWidgets.QLabel("")
        self._truncation_label.setStyleSheet(_TRUNCATION_LABEL_STYLE)
        self._truncation_label.setVisible(False)
        layout.addWidget(self._truncation_label)

    def set_window_buffer(self, payload: WindowPayload) -> None:
        self._buffered_payload = payload

    def update_for_index(self, t_idx: int) -> None:
        self._last_t_idx = int(t_idx)
        payload = self._buffered_payload
        if payload is None or payload.posterior is None:
            self._clear_rows()
            return
        sl = payload.indices
        local_idx = t_idx - sl.start
        if local_idx < 0 or local_idx >= payload.posterior.shape[0]:
            return
        log_lik_row = (
            payload.likelihood[local_idx] if payload.likelihood is not None else None
        )
        bin_payload = self._model.update_for_index(
            t_idx,
            payload.posterior[local_idx],
            log_lik_row=log_lik_row,
        )
        self._render_cells(bin_payload.cells)

    def rebind_after_swap(self, grid: PositionGrid | None = None) -> None:
        self._buffered_payload = None
        self._last_t_idx = None
        if grid is not None:
            if grid.ndim != 2 or grid.shape is None:
                raise ValueError(
                    "Qt2DCellGridPanel.rebind_after_swap requires a 2D "
                    f"PositionGrid; got ndim={grid.ndim}."
                )
            self._grid = grid
            for row in self._rows:
                row.apply_layout(grid)
        self._clear_rows()

    def _render_cells(self, cells: tuple) -> None:
        n_total = len(cells)
        n_shown = min(n_total, MAX_PER_CELL_PLOTS)
        for i in range(n_shown):
            cell = cells[i]
            self._rows[i].show_cell(
                f"#{cell.cell_id}  (×{cell.spike_count})",
                cell.place_field_norm,
                self._grid.shape,
            )
        for i in range(n_shown, MAX_PER_CELL_PLOTS):
            self._rows[i].hide()
        if n_total > MAX_PER_CELL_PLOTS:
            self._truncation_label.setText(f"(+{n_total - MAX_PER_CELL_PLOTS} more)")
            self._truncation_label.setVisible(True)
        else:
            self._truncation_label.setVisible(False)

    def _clear_rows(self) -> None:
        for row in self._rows:
            row.hide()
        self._truncation_label.setVisible(False)
