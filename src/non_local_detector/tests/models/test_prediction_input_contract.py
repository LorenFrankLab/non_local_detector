"""Missing observations and cached likelihoods retain input validation."""

import copy

import numpy as np
import pytest
import xarray as xr

from non_local_detector import DiscreteNonStationaryCustom, Environment, Uniform
from non_local_detector.exceptions import ValidationError
from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.tests.models.test_failed_calls_preserve_model import (
    _assert_unchanged,
    _snapshot,
)
from non_local_detector.tests.models.test_rate_units_and_persistence import (
    ALGORITHMS,
    _decoder,
)


def _fit_population(algorithm):
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    args = ([np.array([0.021, 0.067, 0.121, 0.167])],)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    return _decoder(algorithm).fit(time, position, *args), time, position, args


@pytest.fixture(scope="module", params=ALGORITHMS)
def fitted_population(request):
    return _fit_population(request.param)


@pytest.fixture(
    scope="module",
    params=[name for name in ALGORITHMS if name.startswith("clusterless")],
)
def fitted_marked_population(request):
    return _fit_population(request.param)


@pytest.mark.integration
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize("path", ["likelihood", "predict", "viterbi"])
def test_missing_tracking_cannot_hide_a_dropped_population(
    fitted_population, all_missing, path
):
    fitted, time, position, args = fitted_population
    # Prediction validation must not mutate fitted state. A shallow copy also
    # preserves Patsy's intentionally non-pickleable GLM DesignInfo.
    detector = copy.copy(fitted)
    position = np.full_like(position, np.nan) if all_missing else position
    dropped_args = tuple([] for _ in args)
    edges = np.arange(21) * 0.002
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match="population lengths"):
        if path == "likelihood":
            detector.compute_log_likelihood(
                time, position, *dropped_args, time_edges=edges
            )
        elif path == "predict":
            detector.predict(
                *dropped_args, time_edges=edges, position_time=time, position=position
            )
        else:
            detector.most_likely_sequence(
                *dropped_args, time_edges=edges, position_time=time, position=position
            )
    _assert_unchanged(detector, snapshot)


@pytest.mark.integration
def test_correct_empty_trains_remain_valid_for_missing_tracking(fitted_population):
    detector, time, position, args = fitted_population
    empty_args = ([np.array([])],)
    if len(args) == 2:
        empty_args = (*empty_args, [np.empty((0, 1))])
    position = np.full_like(position, np.nan)
    edges = np.arange(21) * 0.002
    likelihood = detector.compute_log_likelihood(
        time, position, *empty_args, time_edges=edges
    )
    np.testing.assert_array_equal(likelihood, 0.0)
    results = detector.predict(
        *empty_args, time_edges=edges, position_time=time, position=position
    )
    assert results.is_missing.all()
    np.testing.assert_allclose(results.acausal_posterior.sum("state_bins"), 1.0)


@pytest.mark.integration
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize("path", ["likelihood", "predict", "viterbi"])
def test_missing_tracking_cannot_hide_changed_waveform_features(
    fitted_marked_population, all_missing, path
):
    fitted, time, position, args = fitted_marked_population
    detector = copy.copy(fitted)
    position = np.full_like(position, np.nan) if all_missing else position
    changed_args = (args[0], [np.repeat(args[1][0], 2, axis=1)])
    edges = np.arange(21) * 0.002
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match="Waveform-feature dimensionality"):
        if path == "likelihood":
            detector.compute_log_likelihood(
                time, position, *changed_args, time_edges=edges
            )
        elif path == "predict":
            detector.predict(
                *changed_args, time_edges=edges, position_time=time, position=position
            )
        else:
            detector.most_likely_sequence(
                *changed_args, time_edges=edges, position_time=time, position=position
            )
    _assert_unchanged(detector, snapshot)


@pytest.fixture(scope="module", params=["sorted", "clusterless"])
def fitted_covariates(request):
    time = np.arange(20) * 0.002
    position = (50 + np.sin(time * 40))[:, None]
    args = ([np.array([0.011, 0.021])],)
    options = {
        "environments": Environment(place_bin_size=5, position_range=((40, 60),)),
        "continuous_transition_types": [[Uniform(), Uniform()], [Uniform(), Uniform()]],
        "discrete_transition_type": DiscreteNonStationaryCustom(
            values=np.array([[0.9, 0.1], [0.2, 0.8]]), formula="1 + speed"
        ),
        "infer_track_interior": False,
    }
    if request.param == "clusterless":
        detector = ContFragClusterlessClassifier(**options)
        args = (*args, [np.array([[0.1], [-0.1]])])
    else:
        detector = ContFragSortedSpikesClassifier(**options)
    detector.fit(
        time,
        position,
        *args,
        discrete_transition_covariate_data={"speed": np.linspace(0, 1, len(time))},
    )
    return detector, args


@pytest.mark.integration
@pytest.mark.parametrize("n_bins", [10, 30])
@pytest.mark.parametrize("path", ["predict", "cached"])
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_fitted_covariate_rows_are_checked_before_hmm(
    fitted_covariates, n_bins, path, n_chunks, monkeypatch
):
    from non_local_detector.models import base

    fitted, args = fitted_covariates
    detector = copy.copy(fitted)
    edges = np.arange(n_bins + 1) * 0.002

    def unexpected_work(*args, **kwargs):
        pytest.fail(
            "Misaligned stored transitions must fail before likelihood/HMM work"
        )

    monkeypatch.setattr(
        base, "chunked_filter_smoother_covariate_dependent", unexpected_work
    )
    monkeypatch.setattr(detector, "compute_log_likelihood", unexpected_work)
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match="covariate transitions"):
        if path == "predict":
            detector.predict(*args, time_edges=edges, n_chunks=n_chunks)
        else:
            detector._predict(
                edges,
                log_likelihoods=np.zeros(
                    (n_bins, detector.is_track_interior_state_bins_.sum())
                ),
                n_chunks=n_chunks,
            )
    _assert_unchanged(detector, snapshot)


@pytest.mark.integration
def test_new_aligned_covariates_replace_the_stored_grid(fitted_covariates):
    detector, args = fitted_covariates
    n_bins = 30
    result = detector.predict(
        *args,
        time_edges=np.arange(n_bins + 1) * 0.002,
        discrete_transition_covariate_data={"speed": np.linspace(0, 1, n_bins)},
        n_chunks=2,
    )
    assert result.sizes["time"] == n_bins
    np.testing.assert_allclose(
        result.acausal_posterior.sum("state_bins"), 1.0, rtol=1e-6, atol=1e-6
    )


@pytest.mark.integration
@pytest.mark.parametrize("mutated_input", ["edges", "missing"])
def test_saved_result_provenance_does_not_alias_caller_inputs(
    fitted_population, mutated_input
):
    detector, _, _, args = fitted_population
    edges = np.arange(21) * 0.002
    missing = np.zeros(20, dtype=bool)
    missing[0] = True
    result = detector.predict(*args, time_edges=edges, is_missing=missing)
    expected = result.copy(deep=True)
    if mutated_input == "edges":
        edges += 1.0
    else:
        missing[:] = ~missing
    xr.testing.assert_identical(result, expected)
