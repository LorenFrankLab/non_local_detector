"""Predict rejects an encoding model whose environment grid has since changed.

Every fitted encoding model is grid-bound: occupancy, place fields and interior
masks are stored per bin of the grid the environment had at fit time. The
model keeps a reference to the (mutable) environment, so refitting that
environment's grid (``Environment.fit_place_grid``) on other positions pairs the
stale per-bin arrays with the new bins. With the same number of interior bins
no shape check notices, and the likelihood is silently misattributed.
"""

import copy
import pickle

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.tests.likelihoods.conftest import ALGORITHMS

STALE_GRID = "different environment grid"


@pytest.fixture(scope="module")
def encoding_data(line_backends):
    return line_backends[0]


@pytest.fixture(scope="module")
def fitted(line_backends):
    return line_backends[1]


def _predict(predict, model, is_clusterless, encoding_data, *, is_local=False):
    rng = np.random.default_rng(0)
    spike_times = [np.array([2.1, 2.5, 2.9]), np.array([2.3, 2.7])]
    args = [encoding_data["position_time"], encoding_data["position"], spike_times]
    if is_clusterless:
        args.append([rng.standard_normal((len(t), 2)) + 5.0 for t in spike_times])
    return np.asarray(
        predict(*args, time_edges=np.linspace(2.0, 3.0, 11), **model, is_local=is_local)
    )


def _refit_grid(model, place_bin_size, position_range, position):
    """Copy ``model``'s environment and refit its grid in place.

    The fitted model keeps its fit-time arrays and now references the refit
    environment, as when the fit's own environment is refit. Only the
    environment is copied: patsy's GLM ``DesignInfo`` cannot be deep-copied.
    """
    environment = copy.deepcopy(model["environment"])
    model = {**model, "environment": environment}
    n_interior = int(environment.is_track_interior_.sum())
    environment.place_bin_size = place_bin_size
    environment.position_range = position_range
    environment.fit_place_grid(position=position, infer_track_interior=False)
    # Same interior-bin count: no shape check can catch the change.
    assert int(environment.is_track_interior_.sum()) == n_interior
    return model


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("name", ALGORITHMS)
def test_grid_refit_on_shifted_positions_is_rejected(
    fitted, encoding_data, name, is_local
):
    """Ten 10-cm bins over (37, 137) instead of (0, 100) invalidate the model."""
    predict, model, is_clusterless = fitted[name]
    stale = _refit_grid(model, 10.0, ((37.0, 137.0),), encoding_data["position"] + 37.0)
    with pytest.raises(ValidationError, match=STALE_GRID):
        _predict(predict, stale, is_clusterless, encoding_data, is_local=is_local)


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
def test_grid_refit_with_other_bin_geometry_is_rejected(fitted, encoding_data, name):
    """Ten 20-cm bins over (0, 200): same interior pattern, different bins."""
    predict, model, is_clusterless = fitted[name]
    stale = _refit_grid(model, 20.0, ((0.0, 200.0),), encoding_data["position"] * 2.0)
    with pytest.raises(ValidationError, match=STALE_GRID):
        _predict(predict, stale, is_clusterless, encoding_data)


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
def test_grid_refit_on_the_same_positions_keeps_the_model(fitted, encoding_data, name):
    """Refitting the same grid (as fit and EM do) predicts identically."""
    predict, model, is_clusterless = fitted[name]
    expected = _predict(predict, model, is_clusterless, encoding_data)
    same = _refit_grid(model, 10.0, ((0.0, 100.0),), encoding_data["position"])
    np.testing.assert_array_equal(
        _predict(predict, same, is_clusterless, encoding_data), expected
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    # Patsy DesignInfo serialization remains a separately tracked GLM defect.
    [
        name
        for name in ALGORITHMS
        if name not in ("sorted_spikes_glm", "clusterless_mark_glm")
    ],
)
def test_grid_check_survives_pickling(fitted, encoding_data, name):
    """A pickled model predicts identically and still detects a changed grid."""
    predict, model, is_clusterless = fitted[name]
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        _predict(predict, restored, is_clusterless, encoding_data),
        _predict(predict, model, is_clusterless, encoding_data),
    )
    stale = _refit_grid(
        restored, 10.0, ((37.0, 137.0),), encoding_data["position"] + 37.0
    )
    with pytest.raises(ValidationError, match=STALE_GRID):
        _predict(predict, stale, is_clusterless, encoding_data)


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
def test_model_without_a_grid_record_still_predicts(fitted, encoding_data, name):
    """Without the record, direct predict skips the check, as for ``rate_units``.

    ``test_direct_predictors_validate_supplied_units_and_default_to_hz`` pins
    the same leniency for a missing ``rate_units``.
    """
    predict, model, is_clusterless = fitted[name]
    legacy = {key: value for key, value in model.items() if key != "environment_grid"}
    np.testing.assert_array_equal(
        _predict(predict, legacy, is_clusterless, encoding_data),
        _predict(predict, model, is_clusterless, encoding_data),
    )
