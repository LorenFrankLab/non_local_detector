"""Uniform HMM grids remain required through cached and covariate paths."""

import copy

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    Environment,
    SortedSpikesDecoder,
    Uniform,
)
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.tests.models.test_failed_calls_preserve_model import (
    _assert_unchanged,
    _snapshot,
)


@pytest.fixture(scope="module", params=["sorted", "clusterless"])
def fitted_recording(request):
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    args = ([np.array([0.021, 0.067, 0.121])],)
    options = {
        "environments": Environment(place_bin_size=2, position_range=((0, 10),)),
        "continuous_transition_types": [[Uniform()]],
        "infer_track_interior": False,
    }
    if request.param == "clusterless":
        detector = ClusterlessDecoder(**options)
        args = (*args, [np.array([[0.1], [-0.1], [0.2]])])
    else:
        detector = SortedSpikesDecoder(**options)
    return detector.fit(time, position, *args), time, position, args


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", [1, 2])
@pytest.mark.parametrize("grid", ["nonuniform", "wrong_learned_width"])
def test_cached_prediction_validates_grid_before_hmm_work(
    fitted_recording, monkeypatch, n_chunks, grid
):
    """A cached array cannot make an invalid HMM clock safe."""
    from non_local_detector.models import base

    fitted, _, _, _ = fitted_recording
    detector = copy.deepcopy(fitted)
    detector.transition_time_bin_width_ = 0.002
    edges = (
        np.array([0.0, 0.002, 0.012]) if grid == "nonuniform" else np.arange(3) * 0.004
    )
    cached = np.zeros((2, detector.is_track_interior_state_bins_.sum()))
    snapshot = _snapshot(detector)

    def unexpected_hmm(*args, **kwargs):
        pytest.fail("Invalid edges must fail before the HMM runs")

    monkeypatch.setattr(base, "chunked_filter_smoother", unexpected_hmm)
    with pytest.raises((DataError, ValidationError), match="uniform|bin width"):
        detector._predict(edges, log_likelihoods=cached, n_chunks=n_chunks)
    _assert_unchanged(detector, snapshot)


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["predict", "estimate", "viterbi"])
def test_irregular_covariate_grid_rejected_before_transition_or_fit_work(
    fitted_recording, monkeypatch, entry
):
    """The guard applies before nonstationary transition and likelihood paths."""
    from non_local_detector.models import base

    fitted, time, position, args = fitted_recording
    detector = copy.deepcopy(fitted)
    detector.discrete_state_transitions_ = np.ones((2, 1, 1))
    detector.discrete_transition_coefficients_ = np.zeros((1, 1, 1))
    detector.discrete_transition_design_matrix_ = np.ones((2, 1))
    edges = np.array([0.0, 0.002, 0.012])

    def unexpected_work(*args, **kwargs):
        pytest.fail("Irregular grids must fail before covariate/fit/likelihood work")

    monkeypatch.setattr(base, "predict_discrete_state_transitions", unexpected_work)
    monkeypatch.setattr(detector, "fit", unexpected_work)
    monkeypatch.setattr(detector, "compute_log_likelihood", unexpected_work)
    snapshot = _snapshot(detector)
    with pytest.raises(DataError, match="uniform"):
        if entry == "predict":
            detector.predict(
                *args,
                time_edges=edges,
                discrete_transition_covariate_data={"speed": np.ones(2)},
            )
        elif entry == "estimate":
            detector.estimate_parameters(
                time,
                position,
                *args,
                time_edges=edges,
                discrete_transition_covariate_data={"speed": np.ones(2)},
            )
        else:
            detector.most_likely_sequence(*args, time_edges=edges)
    _assert_unchanged(detector, snapshot)


@pytest.mark.integration
def test_base_viterbi_with_custom_likelihood_cannot_bypass_uniformity(
    fitted_recording, monkeypatch
):
    from non_local_detector.core import row_slice_aware
    from non_local_detector.models.base import _DetectorBase

    fitted, _, _, _ = fitted_recording
    detector = copy.deepcopy(fitted)

    @row_slice_aware
    def custom_likelihood(*args, **kwargs):
        pytest.fail("Grid validation must precede a custom likelihood")

    monkeypatch.setattr(detector, "compute_log_likelihood", custom_likelihood)
    with pytest.raises(DataError, match="uniform"):
        _DetectorBase.most_likely_sequence(detector, np.array([0.0, 0.002, 0.012]))


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["predict", "viterbi", "likelihood", "cached"])
@pytest.mark.parametrize(
    "clock,tolerance",
    [
        (np.nan, 0.0),
        (np.inf, 0.0),
        (-0.002, 0.0),
        (0.002, np.inf),
        (0.002, -1.0),
        (0.002, 1.0),
    ],
)
def test_invalid_transition_clock_requires_refit(
    fitted_recording, clock, tolerance, entry
):
    """Unknown provenance cannot disable the learned-width comparison."""
    fitted, time, position, args = fitted_recording
    detector = copy.deepcopy(fitted)
    detector.transition_time_bin_width_ = clock
    detector._transition_time_bin_width_tolerance = tolerance
    edges = np.arange(3) * 0.004
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match="refit"):
        if entry == "predict":
            detector.predict(*args, time_edges=edges)
        elif entry == "viterbi":
            detector.most_likely_sequence(*args, time_edges=edges)
        elif entry == "likelihood":
            detector.compute_log_likelihood(time, position, *args, time_edges=edges)
        else:
            detector._predict(
                edges,
                log_likelihoods=np.zeros(
                    (2, detector.is_track_interior_state_bins_.sum())
                ),
            )
    _assert_unchanged(detector, snapshot)
