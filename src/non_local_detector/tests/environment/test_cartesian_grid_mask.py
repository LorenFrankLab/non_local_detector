"""Caller-supplied Cartesian masks define the fitted environment geometry."""

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.graph_distances import LazyGraphDistances

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("infer_track_interior", [True, False])
@pytest.mark.parametrize("compute_all_pairs_distances", [True, False])
def test_supplied_mask_controls_grid_without_aliasing(
    infer_track_interior, compute_all_pairs_distances
):
    """Explicit masks override occupancy inference and remain caller-owned."""
    position = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :3] = True
    expected = mask.copy()
    env = Environment(place_bin_size=1.0, is_track_interior=mask)

    env.fit_place_grid(
        position,
        infer_track_interior=infer_track_interior,
        compute_all_pairs_distances=compute_all_pairs_distances,
    )

    np.testing.assert_array_equal(env.is_track_interior_, expected)
    assert env.centers_shape_ == mask.shape
    assert env.track_graphDD is not None
    np.testing.assert_array_equal(
        [env.track_graphDD.nodes[node]["is_track_interior"] for node in range(16)],
        expected.ravel(),
    )
    distances = env.distance_between_nodes_
    assert isinstance(distances, (np.ndarray, LazyGraphDistances))
    assert distances[0, 2] == 2.0
    assert np.isinf(distances[0, 4])
    mask[:] = False
    np.testing.assert_array_equal(env.is_track_interior_, expected)


def test_refit_copies_the_current_supplied_mask():
    """Fitting again replaces the previous geometry with the configured mask."""
    position = np.array([[0.0, 0.0], [2.0, 2.0]])
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1:3] = True
    env = Environment(place_bin_size=1.0, is_track_interior=mask)
    env.fit_place_grid(position)
    mask[:] = False
    mask[2, 1:3] = True

    env.fit_place_grid(position)

    assert env.is_track_interior_ is not None
    assert env.track_graphDD is not None
    np.testing.assert_array_equal(env.is_track_interior_, mask)
    assert not np.shares_memory(env.is_track_interior_, mask)
    assert set(env.track_graphDD.edges) == {(9, 10)}


@pytest.mark.parametrize("shape", [(3, 4), (16,), (4, 4, 1)])
def test_supplied_mask_shape_must_match_cartesian_grid(shape):
    position = np.array([[0.0, 0.0], [2.0, 2.0]])
    env = Environment(place_bin_size=1.0, is_track_interior=np.ones(shape, bool))

    with pytest.raises(ValidationError, match="is_track_interior.*shape"):
        env.fit_place_grid(position)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_supplied_mask_requires_boolean_values(dtype):
    position = np.array([[0.0, 0.0], [2.0, 2.0]])
    with pytest.raises(ValidationError, match="is_track_interior.*bool"):
        Environment(
            place_bin_size=1.0, is_track_interior=np.ones((4, 4), dtype=dtype)
        ).fit_place_grid(position)
