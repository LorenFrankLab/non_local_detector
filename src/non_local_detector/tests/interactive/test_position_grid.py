"""Tests for ``PositionGrid.from_environment`` across 1D and 2D."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from non_local_detector.visualization.interactive.view_models.base import (
    PositionGrid,
)


@pytest.mark.unit
def test_position_grid_1d_carries_flat_centers_and_mask() -> None:
    """1D environment produces ``(n_pos,)`` centers + flat interior mask."""
    centers_1d = np.linspace(0.0, 10.0, 5).reshape(-1, 1)
    is_interior = np.array([True, True, False, True, True])
    env = SimpleNamespace(
        place_bin_centers_=centers_1d,
        is_track_interior_=is_interior,
        centers_shape_=(5,),
    )

    grid = PositionGrid.from_environment(env)

    assert grid.ndim == 1
    assert grid.centers.shape == (5,)
    np.testing.assert_array_equal(grid.centers, centers_1d.squeeze(-1))
    assert grid.is_interior is not None
    np.testing.assert_array_equal(grid.is_interior, is_interior)
    # 2D-only fields stay None on the 1D path.
    assert grid.shape is None
    assert grid.is_interior_2d is None


@pytest.mark.unit
def test_position_grid_2d_carries_shape_and_2d_mask() -> None:
    """2D environment exposes the grid shape + reshaped mask.

    Pins the meshgrid convention (``indexing="ij"``, C-order ravel):
    flat index ``k`` maps to ``(i, j)`` with ``i = k // n_y``,
    ``j = k % n_y``. The 2D mask must be reshape-of-flat-with-shape
    in that order.
    """
    n_x, n_y = 3, 4
    # Mirror Environment.fit_place_grid: meshgrid(indexing="ij") then
    # stack ravel.
    x_centers = np.linspace(0.0, 6.0, n_x)
    y_centers = np.linspace(0.0, 8.0, n_y)
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    flat_centers = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_x * n_y, 2)
    # Mask off one corner so the 2D reshape is non-trivially asymmetric.
    is_interior_2d = np.ones((n_x, n_y), dtype=bool)
    is_interior_2d[0, 0] = False
    is_interior_2d[-1, -1] = False
    is_interior_flat = is_interior_2d.ravel()

    env = SimpleNamespace(
        place_bin_centers_=flat_centers,
        is_track_interior_=is_interior_2d,
        centers_shape_=(n_x, n_y),
    )

    grid = PositionGrid.from_environment(env)

    assert grid.ndim == 2
    assert grid.centers.shape == (n_x * n_y, 2)
    np.testing.assert_array_equal(grid.centers, flat_centers)
    assert grid.shape == (n_x, n_y)
    assert grid.is_interior is not None
    np.testing.assert_array_equal(grid.is_interior, is_interior_flat)
    assert grid.is_interior_2d is not None
    assert grid.is_interior_2d.shape == (n_x, n_y)
    np.testing.assert_array_equal(grid.is_interior_2d, is_interior_2d)
    # The 2D mask must agree with the flat mask under the canonical
    # reshape — this is the invariant the renderer relies on.
    np.testing.assert_array_equal(
        grid.is_interior_2d, grid.is_interior.reshape(grid.shape)
    )


@pytest.mark.unit
def test_position_grid_2d_without_track_interior_keeps_2d_fields_none() -> None:
    """``is_interior=None`` propagates through to the 2D mask."""
    env = SimpleNamespace(
        place_bin_centers_=np.zeros((6, 2)),
        is_track_interior_=None,
        centers_shape_=(2, 3),
    )

    grid = PositionGrid.from_environment(env)

    assert grid.ndim == 2
    assert grid.shape == (2, 3)
    assert grid.is_interior is None
    assert grid.is_interior_2d is None


@pytest.mark.unit
def test_position_grid_rejects_higher_dimensional_environments() -> None:
    env = SimpleNamespace(
        place_bin_centers_=np.zeros((10, 3)),
        is_track_interior_=None,
        centers_shape_=(2, 2, 2),  # ignored — the ndim check fires first
    )
    with pytest.raises(ValueError, match="1D or 2D"):
        PositionGrid.from_environment(env)
