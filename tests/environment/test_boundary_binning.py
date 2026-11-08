"""Positions at the edges of the grid land in the bin that contains them.

Reward wells sit at walls and track ends, so many samples fall exactly on the
outermost bin edge (clipped tracking, or projection onto a track end). These
tests pin down where such samples are binned and counted.
"""

import numpy as np
import pytest

from non_local_detector.environment import Environment

pytestmark = pytest.mark.unit

@pytest.mark.parametrize("n_dims", [1, 2])
def test_samples_on_bin_edges_bin_to_the_right_except_at_the_maximum(n_dims):
    """A sample on an interior edge goes to the bin it starts; the maximum
    goes to the last bin, as ``np.histogram`` counts them."""
    edges = np.arange(0.0, 101.0, 10.0)
    if n_dims == 1:
        position = edges[:, np.newaxis]
    else:
        position = np.column_stack([edges, np.linspace(0.0, 20.0, edges.size)])
    env = Environment(place_bin_size=10.0, infer_track_interior=False).fit_place_grid(
        position
    )

    bins = env.get_bin_ind(position)

    expected_x_bin = np.minimum(np.searchsorted(edges, edges, side="right") - 1, 9)
    x_centers = env.place_bin_centers_[bins, 0]
    np.testing.assert_array_equal(x_centers, edges[expected_x_bin] + 5.0)


def test_max_wall_samples_count_toward_the_interior_like_min_wall_samples():
    """Clipped tracking at both walls makes both outermost columns interior.

    Samples exactly at the maximum wall are binned into the last column by
    ``get_bin_ind``; the occupancy that infers the interior must count them
    there too, as it does samples at the minimum wall.
    """
    rng = np.random.default_rng(0)
    bulk = np.column_stack(
        [rng.uniform(10.0, 90.0, 4000), rng.uniform(0.0, 50.0, 4000)]
    )
    wall_y = rng.uniform(0.0, 50.0, 200)
    at_min = np.column_stack([np.zeros(200), wall_y])
    at_max = np.column_stack([np.full(200, 100.0), wall_y])
    env = Environment(place_bin_size=10.0).fit_place_grid(
        np.concatenate([bulk, at_min, at_max])
    )

    interior = env.is_track_interior_
    real = interior[1:-1, 1:-1]
    assert real[0].all(), "min-wall column should be interior"
    assert real[-1].all(), "max-wall column should be interior"
    np.testing.assert_array_equal(real, real[::-1])


def test_dilated_interior_excludes_padding_bins_on_both_sides():
    """Dilation does not extend the interior into the padding bins of get_grid."""
    rng = np.random.default_rng(1)
    position = rng.uniform([0.0, 0.0], [100.0, 50.0], size=(5000, 2))
    env = Environment(place_bin_size=10.0, dilate=True).fit_place_grid(position)

    interior = env.is_track_interior_
    for axis in range(2):
        assert not interior.take(0, axis=axis).any()
        assert not interior.take(-1, axis=axis).any()
    assert interior[1:-1, 1:-1].all()
