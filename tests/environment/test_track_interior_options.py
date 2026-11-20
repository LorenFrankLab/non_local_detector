"""The interior inferred from occupancy, by value, and the N-D grid's extent.

Each occupancy is built by hand on a 5 x 5 grid of 1 cm bins (``position_range``
0-5 cm in both dimensions; `get_grid` pads one bin on every side, so the fitted
mask is 7 x 7 and its outer ring is padding, never interior). A sample at
``(x, y)`` counts in real bin ``(floor(x), floor(y))``.

Every multi-dimensional interior is first closed with the cross-shaped
structuring element, so the expected masks below are those after closing.
"""

import numpy as np
import pytest

from non_local_detector.environment import Environment

pytestmark = pytest.mark.unit

POSITION_RANGE = ((0.0, 5.0), (0.0, 5.0))


def _real_interior(occupied_bins, **options):
    """Fit the 5 x 5 grid with one sample per entry of ``occupied_bins``.

    Returns the interior of the real (unpadded) bins, shape (5, 5), after
    checking that no padding bin is interior.
    """
    position = np.asarray(occupied_bins, dtype=float) + 0.5
    environment = Environment(
        place_bin_size=1.0, position_range=POSITION_RANGE, **options
    ).fit_place_grid(position)
    interior = environment.is_track_interior_
    assert interior.shape == (7, 7)
    padding = np.ones((7, 7), dtype=bool)
    padding[1:-1, 1:-1] = False
    assert not interior[padding].any()
    return interior[1:-1, 1:-1]


RING = [(i, j) for i in range(5) for j in range(5) if i in (0, 4) or j in (0, 4)]


@pytest.mark.parametrize("fill_holes", [False, True])
def test_fill_holes_fills_the_hole_in_a_ring(fill_holes):
    """A one-bin ring around a 3 x 3 hole. Closing fills the hole's corners
    (each touches two ring bins); the plus-shaped rest needs fill_holes."""
    expected = (
        np.ones((5, 5), dtype=bool)
        if fill_holes
        else np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    )

    np.testing.assert_array_equal(_real_interior(RING, fill_holes=fill_holes), expected)


@pytest.mark.parametrize("dilate", [False, True])
def test_dilate_grows_the_interior_by_one_bin(dilate):
    """One occupied bin grows into its four edge neighbours."""
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    if dilate:
        expected[[1, 3, 2, 2], [2, 2, 1, 3]] = True

    np.testing.assert_array_equal(_real_interior([(2, 2)], dilate=dilate), expected)


def test_bin_count_threshold_needs_more_samples_than_the_threshold():
    """With threshold 2, a bin with 3 samples is interior; one with exactly 2
    is not."""
    occupied = [(1, 1)] * 3 + [(3, 3)] * 2
    expected = np.zeros((5, 5), dtype=bool)
    expected[1, 1] = True

    np.testing.assert_array_equal(
        _real_interior(occupied, bin_count_threshold=2), expected
    )


def test_2d_grid_covers_an_extent_that_is_not_a_multiple_of_the_bin_size():
    """x spans 1-11 cm (10 cm) and y 0-6 cm (6 cm) in 4 cm bins: ceil(10 / 4)
    = 3 bins of 10/3 cm in x and ceil(6 / 4) = 2 bins of 3 cm in y, each
    padded by one bin at both ends."""
    position = np.array([[1.0, 0.0], [11.0, 6.0], [6.0, 3.0]])

    environment = Environment(place_bin_size=4.0).fit_place_grid(
        position, infer_track_interior=False
    )

    assert environment.centers_shape_ == (5, 4)
    assert environment.place_bin_centers_.shape == (20, 2)
    x_edges, y_edges = environment.edges_
    assert x_edges.shape == (6,)
    assert x_edges[0] == pytest.approx(1.0 - 10.0 / 3.0, rel=1e-14)
    assert x_edges[-1] == pytest.approx(11.0 + 10.0 / 3.0, rel=1e-14)
    np.testing.assert_array_equal(y_edges, [-3.0, 0.0, 3.0, 6.0, 9.0])
