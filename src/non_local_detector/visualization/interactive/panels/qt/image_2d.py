"""``Qt2DImagePanel`` — bin-synced 2D ``(x, y)`` image at the cursor.

Used as the at-cursor view for 2D detectors. Renders a single-time-bin
slice of a posterior or likelihood window as a 2D image with x on the
horizontal axis, y on the vertical, and the animal's position overlaid
as a magenta dot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from non_local_detector.visualization.interactive.panels.qt._image_2d import (
    flat_to_rgba_image,
    image_2d_layout_from_grid,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        PositionGrid,
        WindowPayload,
    )


# Minimum panel height so the viewbox doesn't collapse when this
# panel shares a right column with the cell-grid stack. Sized so
# two image panels + a 4-row cell-grid fit in the default window
# height without forcing the QMainWindow to grow.
_TOP_IMAGE_MIN_HEIGHT = 200


# Provider returns ``(state_bin_row, animal_xy)`` for a single
# decoder time index. ``state_bin_row`` is a flat
# ``(n_state_bins,)`` array (what the model's ``collapse_at``
# consumes via a 1-row window); ``animal_xy`` is the ``(2,)`` raw
# animal position at that time bin or ``None``. Either may be
# ``None`` to mean "no fallback data available".
RowProvider = Callable[[int], tuple[np.ndarray | None, np.ndarray | None]]
OverlayMode = str


class _BinCollapse(Protocol):
    """Interface the panel needs from a view model.

    Both ``PosteriorHeatmapModel.collapse_at`` and
    ``LikelihoodHeatmapModel.collapse_at`` satisfy this — they share
    a ``(n_window, n_state_bins) + indices -> (len(indices), n_pos)``
    shape contract even though their keyword names differ. The
    panel always passes the window positionally so the parameter
    name on the underlying method is irrelevant.
    """

    def collapse_at(self, window: np.ndarray, indices: list[int], /) -> np.ndarray: ...


class Qt2DImagePanel(pg.PlotWidget):
    """Bin-synced 2D image of a collapsed posterior / likelihood row.

    Parameters
    ----------
    model
        View model exposing ``collapse_at(window, indices) ->
        (len(indices), n_pos)``. ``PosteriorHeatmapModel`` qualifies.
    grid
        ``PositionGrid`` with ``ndim == 2`` carrying the grid shape +
        2D interior mask.
    payload_field
        Which ``WindowPayload`` field to index for the source window.
        ``"posterior"`` (default) or ``"likelihood"`` — the panel
        pulls ``getattr(payload, payload_field)``.
    title
        Static title shown above the image.
    vmax
        Upper bound of the colormap range; values clipped to
        ``[0, vmax]`` before LUT indexing. Defaults to ``0.25`` to
        match the existing posterior heatmap.
    """

    def __init__(
        self,
        model: _BinCollapse,
        grid: PositionGrid,
        *,
        payload_field: str = "posterior",
        title: str = "Smoothed posterior at current time",
        vmax: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, background="w")
        if grid.ndim != 2 or grid.shape is None:
            raise ValueError(
                "Qt2DImagePanel requires a 2D PositionGrid; got "
                f"ndim={grid.ndim}, shape={grid.shape}."
            )
        self._model = model
        self._grid = grid
        self._payload_field = payload_field
        # ``vmax=None`` (default) → per-frame peak-normalize so the
        # cursor row's brightest bin always lands at the LUT top
        # regardless of its absolute magnitude. The at-cursor view
        # spans a single time bin where peak posterior mass can be
        # well below 0.25 (the 1D heatmap default), and a fixed vmax
        # buries the actual peak in the LUT's dark-purple lower
        # quartile. Pass a float to pin a fixed scale.
        self._vmax: float | None = float(vmax) if vmax is not None else None
        # Single-row fallback for cursor ticks past the buffered
        # window — the async window-load can't keep up with fast
        # playback, so without this the image freezes on the last
        # in-buffer frame and visibly lags the cursor.
        self._row_provider: RowProvider | None = None
        self._buffered_payload: WindowPayload | None = None
        self._last_t_idx: int | None = None
        self._overlay_mode: OverlayMode = "smoothed"

        self.setTitle(title)
        self.setLabel("left", "y")
        self.setLabel("bottom", "x")
        # Don't aspect-lock. Combined with ``setLimits`` clamping the
        # axes to the data bounds, aspect-lock would crop the image
        # whenever the panel is non-square (the common case in the
        # right column). Letting the image stretch to fill the panel
        # keeps axis labels honest (they always read the bin bounds)
        # and shows the full posterior / likelihood — the small
        # aspect distortion is acceptable for a non-square widget.
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        # Reserve enough vertical space that the panel doesn't get
        # starved by the cell-grid sibling in the right column.
        self.setMinimumHeight(_TOP_IMAGE_MIN_HEIGHT)

        self._lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._image_item)
        self._animal_marker = pg.ScatterPlotItem()
        self._animal_marker.setZValue(10)
        self.addItem(self._animal_marker)

        self._apply_grid_geometry()

    def set_window_buffer(self, payload: WindowPayload) -> None:
        self._buffered_payload = payload

    def set_row_provider(self, provider: RowProvider | None) -> None:
        """Register a single-row fallback for cursor ticks past the buffer.

        Without it, ``update_for_index`` early-returns when the
        cursor moves past the buffered window (which is common
        during fast playback — the async window-load can't keep
        up), leaving the image frozen on the last buffered frame
        and visibly lagging the slider.
        """
        self._row_provider = provider

    @property
    def overlay_mode(self) -> OverlayMode:
        return self._overlay_mode

    def set_overlay_mode(self, mode: OverlayMode) -> None:
        """Select the posterior-like source for 2D cursor images."""
        if mode not in {"predictive", "filtered", "smoothed"}:
            raise ValueError(
                "overlay mode must be one of "
                "['filtered', 'predictive', 'smoothed']; "
                f"got {mode!r}"
            )
        if mode == self._overlay_mode:
            return
        self._overlay_mode = mode
        self._refresh_title()
        if self._last_t_idx is not None:
            self.update_for_index(self._last_t_idx)

    def update_for_index(self, t_idx: int) -> None:
        self._last_t_idx = int(t_idx)
        payload = self._buffered_payload
        row: np.ndarray | None = None
        animal_xy: np.ndarray | None = None
        in_buffer = False
        if payload is not None:
            buf_local_idx = int(t_idx - payload.indices.start)
            if 0 <= buf_local_idx < payload.indices.stop - payload.indices.start:
                row = self._row_from_payload(payload, buf_local_idx)
                if row is not None:
                    in_buffer = True
                    animal_xy = self._animal_xy_from_payload(payload, buf_local_idx)
        if not in_buffer:
            # Async window-load hasn't reached this cursor bin yet.
            # Fetch the single row synchronously so the panel stays
            # in sync with the slider during playback.
            if self._row_provider is None:
                return
            row, animal_xy = self._row_provider(t_idx)
            if row is None:
                return
        window = np.asarray(row)[np.newaxis, :]
        flat = np.asarray(
            self._model.collapse_at(window, [0])[0], dtype=np.float64
        )
        # Mask off-track bins to NaN so they render transparent.
        # ``PosteriorHeatmapModel.collapse_at`` already preserves the
        # NaN at non-interior bins from ``_create_masked_posterior``,
        # but ``LikelihoodHeatmapModel`` maps ``NaN → -inf → 0`` and
        # emits a finite zero — without this mask those bins would
        # render as solid LUT-bottom pixels, hiding the actual track
        # geometry.
        is_interior = self._grid.is_interior
        if is_interior is not None and is_interior.size == flat.size:
            flat = np.where(is_interior, flat, np.nan)
        if self._vmax is None:
            peak = float(np.nanmax(flat)) if flat.size else 0.0
            frame_vmax = peak if peak > 0.0 else 1.0
        else:
            frame_vmax = self._vmax
        rgba = flat_to_rgba_image(flat, self._grid.shape, self._lut, frame_vmax)
        self._image_item.setImage(rgba, autoLevels=False)
        # ``setImage`` resets the ImageItem's transform back to
        # pixel coords (rect = image shape), so the rect set during
        # construction is gone by the first cursor tick. Re-apply
        # it here so the image is always positioned in the data's
        # ``(x_min, y_min, width, height)`` rectangle.
        self._image_item.setRect(
            QtCore.QRectF(
                self._layout.x_min,
                self._layout.y_min,
                self._layout.width,
                self._layout.height,
            )
        )
        self._render_animal_marker(animal_xy)

    def _row_from_payload(
        self, payload: WindowPayload, local_idx: int
    ) -> np.ndarray | None:
        if self._payload_field != "posterior":
            window = getattr(payload, self._payload_field, None)
            return None if window is None else window[local_idx]
        if self._overlay_mode == "smoothed":
            return None if payload.posterior is None else payload.posterior[local_idx]
        if self._overlay_mode == "predictive":
            return None if payload.predictive is None else payload.predictive[local_idx]
        if self._overlay_mode == "filtered":
            if payload.predictive is None or payload.likelihood is None:
                return None
            return _filtered_row(
                payload.predictive[local_idx],
                _linear_likelihood_row(payload.likelihood[local_idx]),
            )
        raise AssertionError(f"unhandled overlay_mode {self._overlay_mode!r}")

    def _refresh_title(self) -> None:
        if self._payload_field != "posterior":
            return
        labels = {
            "smoothed": "Smoothed posterior at current time",
            "predictive": "Predictive posterior at current time",
            "filtered": "Filtered posterior at current time",
        }
        self.setTitle(labels[self._overlay_mode])

    def rebind_after_swap(self, grid: PositionGrid | None = None) -> None:
        self._buffered_payload = None
        self._last_t_idx = None
        if grid is not None:
            if grid.ndim != 2 or grid.shape is None:
                raise ValueError(
                    "Qt2DImagePanel.rebind_after_swap requires a 2D "
                    f"PositionGrid; got ndim={grid.ndim}."
                )
            self._grid = grid
            self._apply_grid_geometry()
        self._clear_image()

    def _apply_grid_geometry(self) -> None:
        layout = image_2d_layout_from_grid(self._grid)
        # Cache for ``update_for_index`` — ``setImage`` resets the
        # ImageItem transform on every call so we re-apply the rect
        # per render.
        self._layout = layout
        self._image_item.setRect(
            QtCore.QRectF(layout.x_min, layout.y_min, layout.width, layout.height)
        )
        vb = self.getViewBox()
        # Clamp axes to the data bounds + lock the data aspect ratio.
        # The panel's minimum height (``_TOP_IMAGE_MIN_HEIGHT``) gives
        # aspect-lock room to render the image at proper proportions
        # within the widget — non-square widgets get white margins on
        # the longer pixel-per-unit axis rather than extending the
        # visible data range past the bin bounds (which would
        # mislabel the axis ticks).
        vb.setRange(
            xRange=(layout.x_min, layout.x_max),
            yRange=(layout.y_min, layout.y_max),
            padding=0,
        )
        vb.setLimits(
            xMin=layout.x_min,
            xMax=layout.x_max,
            yMin=layout.y_min,
            yMax=layout.y_max,
        )

    def _clear_image(self) -> None:
        self._image_item.clear()
        self._animal_marker.setData(x=[], y=[])

    def _animal_xy_from_payload(
        self, payload: WindowPayload, local_idx: int
    ) -> np.ndarray | None:
        """Pull animal XY out of a buffered payload at ``local_idx``."""
        position = payload.position
        if (
            position is None
            or position.ndim != 2
            or position.shape[1] != 2
            or local_idx >= position.shape[0]
        ):
            return None
        return np.asarray(position[local_idx], dtype=float)

    def _render_animal_marker(self, xy: np.ndarray | None) -> None:
        if xy is None or xy.size != 2 or not np.all(np.isfinite(xy)):
            self._animal_marker.setData(x=[], y=[])
            return
        # Hollow magenta ring — a filled dot covers ~5 cm of bins
        # underneath at typical column widths, hiding multimodal
        # posterior structure right where it's most interesting.
        self._animal_marker.setData(
            x=[float(xy[0])],
            y=[float(xy[1])],
            size=16,
            symbol="o",
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen((255, 0, 255, 255), width=2.5),
        )


def _linear_likelihood_row(log_lik_row: np.ndarray | None) -> np.ndarray | None:
    if log_lik_row is None:
        return None
    log = np.asarray(log_lik_row, dtype=np.float64)
    if log.size == 0:
        return log
    finite = np.isfinite(log)
    if not finite.any():
        return np.zeros_like(log)
    out = np.zeros_like(log)
    out[finite] = np.exp(log[finite] - log[finite].max())
    return out


def _filtered_row(
    predictive_row: np.ndarray | None, likelihood_row: np.ndarray | None
) -> np.ndarray | None:
    if predictive_row is None or likelihood_row is None:
        return None
    predictive = np.asarray(predictive_row, dtype=float)
    likelihood = np.asarray(likelihood_row, dtype=float)
    if predictive.shape != likelihood.shape:
        return None
    filtered = np.nan_to_num(
        predictive, nan=0.0, posinf=0.0, neginf=0.0
    ) * np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(filtered.sum())
    if total <= 0.0:
        return None
    return filtered / total
