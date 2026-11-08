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
