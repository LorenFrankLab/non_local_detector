"""A failed fit or estimation call leaves a fitted detector as it was.

Checks on ``estimate_parameters`` arguments run before the initial ``fit``
rebuilds environments, transitions, and the encoding model. An encoding refit
that fails keeps the previous encoding model and its stored log likelihood,
and a ``fit`` that fails after rebuilding the environments restores the
detector and its environments. Successful calls still store and replace their
outputs.
"""

import copy

import numpy as np
import pytest

from non_local_detector import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.tests.models._helpers import _assert_unchanged, _snapshot

FAMILIES = ["sorted", "clusterless"]
N_DECODE = 300


@pytest.fixture(scope="module", params=FAMILIES)
def family_data(request, sorted_sim, clusterless_sim):
    """``(detector class, fit arguments, predict arguments)`` for a family."""
    if request.param == "sorted":
        time, position, spike_times = sorted_sim
        fit_args = {
            "position_time": time,
            "position": position,
            "spike_times": spike_times,
        }
        predict_args = {"spike_times": spike_times}
        return NonLocalSortedSpikesDetector, fit_args, predict_args
    sim = clusterless_sim
    fit_args = {
        "position_time": sim.position_time,
        "position": sim.position,
        "spike_times": sim.spike_times,
        "spike_waveform_features": sim.spike_waveform_features,
    }
    predict_args = {
        "spike_times": sim.spike_times,
        "spike_waveform_features": sim.spike_waveform_features,
    }
    return NonLocalClusterlessDetector, fit_args, predict_args


@pytest.fixture(scope="module")
def fitted(family_data):
    """``(detector, posterior)``: a detector fit by EM that stores its log
    likelihood, and its decoded posterior.

    Shared across the module; tests that call the detector work on a deep
    copy. Decoding first also sets predict's transient attributes, so a copy's
    snapshot already includes them.
    """
    detector_cls, fit_args, _ = family_data
    detector = detector_cls()
    detector.estimate_parameters(
        **fit_args,
        time_edges=time_edges_from_centers(fit_args["position_time"]),
        max_iter=1,
        estimate_encoding_model=False,
        store_log_likelihood=True,
    )
    assert hasattr(detector, "log_likelihood_")
    return detector, _posterior(detector, family_data)


def _posterior(detector, family_data):
    _, fit_args, predict_args = family_data
    time = fit_args["position_time"][:N_DECODE]
    return detector.predict(
        **predict_args,
        time_edges=time_edges_from_centers(time),
        position=fit_args["position"][:N_DECODE],
        position_time=time,
    ).acausal_posterior.values


def _refit_changes_state(fit_args):
    """Arguments whose successful fit would change the grid and the encoding."""
    n = fit_args["position_time"].shape[0]
    return {**fit_args, "position": fit_args["position"] * 1.5}, np.arange(n) < n // 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bad_arguments", "error"),
    [
        ({"min_encoding_local_mass": -1.0}, ValueError),
        ({"min_encoding_local_ess": -1.0}, ValueError),
        ({"return_outputs": "not_an_output"}, ValueError),
        (
            {"save_log_likelihood_to_results": True, "return_outputs": "filter"},
            ValueError,
        ),
        ({"n_chunks": "more_than_time"}, ValueError),
        ({"time_edges": "decreasing"}, DataError),
        ({"time_edges": "repeated"}, DataError),
        ({"time_edges": "nonuniform"}, DataError),
        ({"encoding_update_damping": 0.5}, ValidationError),
    ],
    ids=[
        "negative-mass",
        "negative-ess",
        "unknown-output",
        "deprecated-flag-conflict",
        "too-many-chunks",
        "decreasing-edges",
        "repeated-edges",
        "nonuniform-edges",
        "damping",
    ],
)
def test_invalid_estimation_arguments_fail_before_fitting(
    family_data, fitted, bad_arguments, error
):
    reference, posterior_before = fitted
    detector = copy.deepcopy(reference)
    snapshot = _snapshot(detector)
    _, fit_args, _ = family_data
    refit_args, is_training = _refit_changes_state(fit_args)
    time_edges = time_edges_from_centers(fit_args["position_time"])
    bad_edges = time_edges.copy()
    if bad_arguments.get("time_edges") == "decreasing":
        bad_edges = time_edges[::-1]
    elif bad_arguments.get("time_edges") == "repeated":
        bad_edges[5] = bad_edges[4]
    elif bad_arguments.get("time_edges") == "nonuniform":
        bad_edges[5] += 0.25 * (time_edges[6] - time_edges[5])
    if "time_edges" in bad_arguments:
        bad_arguments = {**bad_arguments, "time_edges": bad_edges}
    if bad_arguments.get("n_chunks") == "more_than_time":
        n_bins = time_edges.shape[0] - 1
        bad_arguments = {**bad_arguments, "n_chunks": n_bins + 1}

    with pytest.raises(error):
        detector.estimate_parameters(
            **refit_args,
            is_training=is_training,
            **{"time_edges": time_edges, "max_iter": 1, **bad_arguments},
        )

    _assert_unchanged(detector, snapshot)
    np.testing.assert_array_equal(_posterior(detector, family_data), posterior_before)


@pytest.mark.unit
def test_failed_encoding_refit_keeps_model_and_stored_likelihood(family_data, fitted):
    reference, posterior_before = fitted
    detector = copy.deepcopy(reference)
    snapshot = _snapshot(detector)
    _, fit_args, _ = family_data
    bad_weights = np.ones(fit_args["position_time"].shape[0])
    bad_weights[10] = -1.0

    with pytest.raises(ValidationError):
        detector.fit_encoding_model(**fit_args, weights=bad_weights)

    _assert_unchanged(detector, snapshot)
    np.testing.assert_array_equal(_posterior(detector, family_data), posterior_before)


@pytest.mark.unit
def test_fit_failing_after_rebuilding_state_restores_detector(
    family_data, fitted, monkeypatch
):
    """The encoding fit fails after ``fit`` has rebuilt the environments for new
    position data; the detector, its environments, and its posterior are
    restored."""
    reference, posterior_before = fitted
    detector = copy.deepcopy(reference)
    snapshot = _snapshot(detector)
    detector_cls, fit_args, _ = family_data
    registry, name = (
        (_SORTED_SPIKES_ALGORITHMS, detector.sorted_spikes_algorithm)
        if detector_cls is NonLocalSortedSpikesDetector
        else (_CLUSTERLESS_ALGORITHMS, detector.clusterless_algorithm)
    )

    def failing_fit(**kwargs):
        raise RuntimeError("encoding backend failed")

    monkeypatch.setitem(registry, name, (failing_fit, registry[name][1]))
    refit_args, is_training = _refit_changes_state(fit_args)

    with pytest.raises(RuntimeError, match="encoding backend failed"):
        detector.fit(**refit_args, is_training=is_training)

    _assert_unchanged(detector, snapshot)
    np.testing.assert_array_equal(_posterior(detector, family_data), posterior_before)


@pytest.mark.unit
def test_successful_calls_store_and_replace_outputs(family_data, fitted):
    """After a failed call, successful calls still produce their outputs: an
    encoding refit replaces the model and drops the stale stored likelihood,
    and estimation stores a new one."""
    detector = copy.deepcopy(fitted[0])
    _, fit_args, _ = family_data
    bad_weights = np.full(fit_args["position_time"].shape[0], -1.0)
    with pytest.raises(ValidationError):
        detector.fit_encoding_model(**fit_args, weights=bad_weights)
    model_before = detector.encoding_model_

    n = fit_args["position_time"].shape[0]
    detector.fit_encoding_model(**fit_args, is_training=np.arange(n) < n // 2)
    assert detector.encoding_model_ is not model_before
    assert detector.encoding_model_.keys() == model_before.keys()
    assert not hasattr(detector, "log_likelihood_")

    detector.estimate_parameters(
        **fit_args,
        time_edges=time_edges_from_centers(fit_args["position_time"]),
        max_iter=1,
        store_log_likelihood=True,
    )
    assert np.all(np.isfinite(detector.log_likelihood_))
    posterior = _posterior(detector, family_data)
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, rtol=1e-5)
