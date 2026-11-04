"""Environment.is_track_interior_flat_ is the interior mask over flattened bins."""

import pickle

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import NotFittedError

pytestmark = pytest.mark.unit


@pytest.fixture
def open_field():
    rng = np.random.default_rng(0)
    return Environment(place_bin_size=5.0).fit_place_grid(rng.uniform(0, 40, (400, 2)))


def test_flattens_fitted_mask(open_field):
    mask = open_field.is_track_interior_flat_

    assert mask.shape == (len(open_field.place_bin_centers_),)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, open_field.is_track_interior_.ravel())
    assert not mask.all()  # the fixture's grid has exterior bins


def test_unfitted_environment_raises():
    with pytest.raises(NotFittedError, match="fit_place_grid"):
        Environment().is_track_interior_flat_  # noqa: B018


def test_unpickled_environment_has_mask(open_field):
    restored = pickle.loads(pickle.dumps(open_field))

    np.testing.assert_array_equal(
        restored.is_track_interior_flat_, open_field.is_track_interior_flat_
    )
