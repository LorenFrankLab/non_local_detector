"""Shared 2D image rendering helpers for the bin-synced 2D panels.

Two pieces:

- ``Image2DLayout`` + ``image_2d_layout_from_grid`` — rect bounds for
  a 2D heatmap pinned to a ``PositionGrid`` (half-bin padding on
  each side so bin centers sit at pixel centers).
- ``flat_to_rgba_image`` — reshape ``(n_pos,)`` flat posterior/
  likelihood into an ``(n_y, n_x, 4)`` RGBA image with NaN
  rendering as transparent. Output shape matches pyqtgraph
  ``ImageItem(axisOrder="row-major")`` with x on the horizontal
  axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        PositionGrid,
    )


@dataclass(frozen=True)
class Image2DLayout:
    """Rect bounds for a 2D heatmap aligned to bin centers."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    shape: tuple[int, int]  # (n_x, n_y)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


def image_2d_layout_from_grid(grid: PositionGrid) -> Image2DLayout:
    """Build a rect that pads the bin-center extents by half a bin.

    Pulls unique x and y centers out of ``grid.centers`` (which is
    ``(n_pos, 2)`` C-order ravel of ``meshgrid(*edges, indexing="ij")``).
    With the canonical ordering, every ``n_y``-th flat entry is a new
    x value and the first ``n_y`` entries are the y values for x[0].
    """
    if grid.ndim != 2 or grid.shape is None:
        raise ValueError(
            "image_2d_layout_from_grid requires a 2D PositionGrid; got "
            f"ndim={grid.ndim}, shape={grid.shape}."
        )
    n_x, n_y = grid.shape
    centers = np.asarray(grid.centers)
    if centers.shape != (n_x * n_y, 2):
        raise ValueError(
            "PositionGrid.centers shape inconsistent with grid.shape: "
            f"expected ({n_x * n_y}, 2), got {centers.shape}."
        )
    x_centers = centers[::n_y, 0]
    y_centers = centers[:n_y, 1]
    dx_half = float(x_centers[-1] - x_centers[0]) / (2 * (n_x - 1)) if n_x > 1 else 0.0
    dy_half = float(y_centers[-1] - y_centers[0]) / (2 * (n_y - 1)) if n_y > 1 else 0.0
    return Image2DLayout(
        x_min=float(x_centers[0]) - dx_half,
        x_max=float(x_centers[-1]) + dx_half,
        y_min=float(y_centers[0]) - dy_half,
        y_max=float(y_centers[-1]) + dy_half,
        shape=(n_x, n_y),
    )


def flat_to_rgba_image(
    flat: np.ndarray,
    shape: tuple[int, int],
    lut: np.ndarray,
    vmax: float,
) -> np.ndarray:
    """Reshape ``(n_pos,)`` → ``(n_y, n_x, 4)`` RGBA with NaN→transparent.

    The ``.reshape(shape).T`` step takes flat → ``(n_x, n_y)`` →
    ``(n_y, n_x)``. ImageItem with ``axisOrder="row-major"`` reads
    rows as the vertical axis and columns as the horizontal axis,
    so the final layout has x on the horizontal axis as expected.

    Parameters
    ----------
    flat : np.ndarray, shape (n_pos,)
        Collapsed posterior / likelihood row. NaN entries (typically
        non-interior bins) render as fully transparent.
    shape : tuple[int, int]
        Grid shape ``(n_x, n_y)`` from ``PositionGrid.shape``.
    lut : np.ndarray, shape (256, 3 or 4)
        Lookup table. RGB columns are used; alpha is set per pixel
        based on finiteness.
    vmax : float
        Upper end of the colormap range. Values are clipped to
        ``[0, vmax]`` before LUT indexing.
    """
    arr = flat.reshape(shape).T.astype(np.float32, copy=False)  # (n_y, n_x)
    finite = np.isfinite(arr)
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    if vmax > 0:
        # Use np.divide with ``out`` + ``where`` to avoid RuntimeWarning
        # on NaN entries. Non-finite entries get LUT index 0; their
        # alpha stays 0 anyway so the choice doesn't render.
        scaled = np.zeros_like(arr)
        np.divide(arr, vmax, out=scaled, where=finite)
        idx = np.clip(scaled * 255.0, 0, 255).astype(np.int32)
    else:
        idx = np.zeros(arr.shape, dtype=np.int32)
    rgba[..., 0] = np.where(finite, lut[idx, 0], 0)
    rgba[..., 1] = np.where(finite, lut[idx, 1], 0)
    rgba[..., 2] = np.where(finite, lut[idx, 2], 0)
    rgba[..., 3] = np.where(finite, 255, 0)
    return rgba
