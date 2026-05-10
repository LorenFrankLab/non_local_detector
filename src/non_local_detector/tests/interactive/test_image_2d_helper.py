"""Tests for the ``_image_2d`` rendering helper.

Locks in the reshape ordering: flat-index ``k = i * n_y + j`` must
land at ``(i, j)`` in the gridded array, which then transposes to
``(j, i)`` in the row-major RGBA output so x stays on the horizontal
axis when ``ImageItem(axisOrder="row-major")`` consumes it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from non_local_detector.visualization.interactive.panels.qt._image_2d import (
    flat_to_rgba_image,
    image_2d_layout_from_grid,
)
from non_local_detector.visualization.interactive.view_models.base import (
    PositionGrid,
)


def _make_2d_grid(n_x: int, n_y: int) -> PositionGrid:
    """Build a PositionGrid that mirrors Environment.fit_place_grid."""
    x_centers = np.linspace(0.0, float(n_x - 1), n_x)
    y_centers = np.linspace(0.0, float(n_y - 1), n_y)
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel()], axis=1)
    is_interior_2d = np.ones((n_x, n_y), dtype=bool)
    env = SimpleNamespace(
        place_bin_centers_=centers,
        is_track_interior_=is_interior_2d,
        centers_shape_=(n_x, n_y),
    )
    return PositionGrid.from_environment(env)


@pytest.mark.unit
def test_layout_pads_centers_by_half_bin() -> None:
    """Layout rect should pad bin-center extents by half a bin per side."""
    grid = _make_2d_grid(n_x=4, n_y=5)  # x centers 0..3, y centers 0..4
    layout = image_2d_layout_from_grid(grid)

    # Bin size is 1.0 along both axes → half-bin is 0.5.
    assert layout.x_min == pytest.approx(-0.5)
    assert layout.x_max == pytest.approx(3.5)
    assert layout.y_min == pytest.approx(-0.5)
    assert layout.y_max == pytest.approx(4.5)
    assert layout.shape == (4, 5)


@pytest.mark.unit
def test_flat_to_rgba_preserves_x_horizontal_y_vertical() -> None:
    """A spike at (x=2, y=1) must land at RGBA row 1, column 2.

    Convention:
    - flat index of bin (i=2, j=1) is ``k = i*n_y + j = 2*3 + 1 = 7``
    - After ``reshape(n_x, n_y)``, position is ``arr[2, 1]``
    - After ``.T``, position is ``arr_T[1, 2]`` — row 1 (y=1), col 2 (x=2)
    - Row-major ImageItem treats row = y, column = x → correct.
    """
    n_x, n_y = 4, 3
    flat = np.zeros(n_x * n_y, dtype=np.float32)
    flat[2 * n_y + 1] = 1.0  # spike at bin (i=2, j=1) → physical (x=2, y=1)
    lut = np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[:, None], (1, 3))

    rgba = flat_to_rgba_image(flat, shape=(n_x, n_y), lut=lut, vmax=1.0)

    assert rgba.shape == (n_y, n_x, 4)
    # The single non-zero bin lives at (row=1, col=2) in the output.
    # All channels should be at the LUT top (255) at (1, 2).
    assert rgba[1, 2, 0] == 255
    assert rgba[1, 2, 3] == 255  # alpha
    # And nowhere else (only (1, 2) was set).
    other_alpha = rgba[..., 3].copy()
    other_alpha[1, 2] = 0
    # NaN-free flat means every other bin is finite-zero → alpha=255 too.
    # The point of this check is that the spike isn't somewhere wrong.
    nonzero_rgb = np.any(rgba[..., :3] > 0, axis=-1)
    assert nonzero_rgb.sum() == 1
    assert nonzero_rgb[1, 2]


@pytest.mark.unit
def test_flat_to_rgba_renders_nan_as_transparent() -> None:
    """Non-finite flat entries must produce alpha=0 in the RGBA output."""
    n_x, n_y = 3, 2
    flat = np.array([0.5, np.nan, 0.5, 0.5, np.nan, 0.5], dtype=np.float32)
    lut = np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[:, None], (1, 3))

    rgba = flat_to_rgba_image(flat, shape=(n_x, n_y), lut=lut, vmax=1.0)

    # NaN positions in flat are (i, j) = (0, 1) and (2, 0):
    #   k=0*n_y+1=1 → arr[0,1] → arr_T[1,0] → rgba[1, 0]
    #   k=2*n_y+0=4 → arr[2,0] → arr_T[0,2] → rgba[0, 2]
    assert rgba[1, 0, 3] == 0, "NaN at (i=0, j=1) must be transparent"
    assert rgba[0, 2, 3] == 0, "NaN at (i=2, j=0) must be transparent"
    # All finite entries should be opaque.
    finite_mask = np.array([[True, True, False], [False, True, True]])
    np.testing.assert_array_equal((rgba[..., 3] == 255), finite_mask)
