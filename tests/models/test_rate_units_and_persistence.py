"""Physical rate units and model persistence through public decoder APIs.

- Every likelihood backend fits rates in Hz, so an empty bin's log likelihood
  is minus the rate times the bin width.
- Every backend keeps its units and decode results through ``save_model`` /
  ``load_model``.
- A saved model with missing or unknown encoding units, a legacy or unknown
  time contract, or invalid transition-clock provenance is rejected, before
  any likelihood work, with a hint to refit; ``estimate_parameters`` refits
  such a model from the original recording.
"""

import copy
import functools
import pickle

import numpy as np
import pytest
import xarray as xr

from non_local_detector.exceptions import ValidationError
from tests.models._helpers import ALGORITHMS, _assert_unchanged, _decoder, _snapshot


@pytest.mark.integration
# Five events at one position leave KMeans with duplicate points, which the GMM
# backend initializes from; the calibrated rate does not depend on the clusters.
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("position_frequency", [30, 500])
def test_known_rate_is_calibrated_through_public_fit_and_predict(
    algorithm, position_frequency
):
    """Five events over one second predict 0.01 events in an empty 2 ms bin.

    Evaluate the independently known homogeneous rate at its observed position,
    without assuming spatial fits coincide at unobserved locations. Three
    spatial bins avoid the unrelated singleton environment-index limitation;
    zero-centered data avoid ill-conditioned Gaussian mean/covariance arithmetic.
    """
    time = np.arange(position_frequency) / position_frequency
    position = np.zeros((len(time), 1))
    spikes = [np.array([0.1, 0.3, 0.5, 0.7, 0.9])]
    args = (spikes,)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.zeros((5, 1))])
    for width in (0.002, 0.004):
        # The detector's sampling frequency must match the decode bin width.
        detector = _decoder(
            algorithm, calibration=True, sampling_frequency=1 / width
        ).fit(time, position, *args)
        result = detector.predict(
            *args,
            time_edges=0.2 + np.arange(3) * width,
            return_outputs="log_likelihood",
        )
        # No event occurs here, so the Poisson/point-process log likelihood is
        # minus the expected count. This reference does not read fitted fields.
        np.testing.assert_allclose(
            result.log_likelihood.sel(position=0.0).to_numpy(),
            -5.0 * width,
            rtol=1e-5,
            atol=0,
        )
        assert np.all(np.isfinite(result.acausal_posterior))
    (encoding,) = detector.encoding_model_.values()
    assert encoding["rate_units"] == "Hz"
    assert encoding["encoding_exposure_seconds"] == pytest.approx(1.0)


@pytest.mark.integration
# The default position_std is below the 5-unit bin spacing; the test checks
# persistence, not the smoothing width.
@pytest.mark.filterwarnings(
    r"ignore:position_std \(.*\) is below the characteristic bin spacing:UserWarning"
)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_serializable_backend_round_trip_preserves_units_and_predictions(
    algorithm, tmp_path
):
    """Every backend keeps exact decode results and units."""
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121, 0.167])]
    args = (spikes,)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    detector = _decoder(algorithm).fit(time, position, *args)
    kwargs = {
        "time_edges": np.arange(81) * 0.002,
        "position_time": time,
        "position": position,
        "return_outputs": "log_likelihood",
    }
    expected = detector.predict(*args, **kwargs)
    path = tmp_path / "detector.pkl"
    detector.save_model(path)
    restored = detector.load_model(path)
    actual = restored.predict(*args, **kwargs)
    xr.testing.assert_identical(actual, expected)
    assert restored.time_contract_ == detector.time_contract_
    for model in restored.encoding_model_.values():
        assert model["rate_units"] == "Hz"
        assert model["encoding_exposure_seconds"] == pytest.approx(0.2)


@pytest.fixture(scope="module", params=["sorted_spikes_kde", "clusterless_kde"])
def contract_recording(request):
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121, 0.167])]
    args = (spikes,)
    if request.param.startswith("clusterless"):
        args = (*args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    return _decoder(request.param).fit(time, position, *args), time, position, args


def _set_transition_clock(width, tolerance):
    def corrupt(detector):
        detector.transition_time_bin_width_ = width
        detector._transition_time_bin_width_tolerance = tolerance

    return corrupt


def _delete_entry_units(detector):
    (encoding,) = detector.encoding_model_.values()
    del encoding["rate_units"]


def _set_entry_units(detector):
    (encoding,) = detector.encoding_model_.values()
    encoding["rate_units"] = "per_position_sample"


def _set_contract_version(detector):
    detector.time_contract_["version"] = 999


def _set_contract_rate_units(detector):
    detector.time_contract_ = {**detector.time_contract_, "rate_units": "per_bin"}


def _delete_time_contract(detector):
    del detector.time_contract_


def _delete_transition_clock(detector):
    del detector.transition_time_bin_width_


_UNKNOWN_UNITS = r"Encoding entry .* is missing or has unknown rate units"
_UNKNOWN_CONTRACT = "Saved model has a legacy or unknown time contract"
_INVALID_CLOCK = "Saved model has invalid transition clock provenance"

# Each corruption of a fitted model, and the cause its refit error names.
_CORRUPTIONS = {
    "missing-entry-units": (_delete_entry_units, _UNKNOWN_UNITS),
    "unknown-entry-units": (_set_entry_units, _UNKNOWN_UNITS),
    "unknown-contract-version": (_set_contract_version, _UNKNOWN_CONTRACT),
    "unknown-contract-rate-units": (_set_contract_rate_units, _UNKNOWN_CONTRACT),
    "missing-time-contract": (_delete_time_contract, _UNKNOWN_CONTRACT),
    "missing-transition-clock": (_delete_transition_clock, _UNKNOWN_CONTRACT),
    # Unknown provenance cannot disable the learned-width comparison.
    "nan-clock": (_set_transition_clock(np.nan, 0.0), _INVALID_CLOCK),
    "infinite-clock": (_set_transition_clock(np.inf, 0.0), _INVALID_CLOCK),
    "negative-clock": (_set_transition_clock(-0.002, 0.0), _INVALID_CLOCK),
    "infinite-tolerance": (_set_transition_clock(0.002, np.inf), _INVALID_CLOCK),
    "negative-tolerance": (_set_transition_clock(0.002, -1.0), _INVALID_CLOCK),
    "tolerance-above-bound": (_set_transition_clock(0.002, 1.0), _INVALID_CLOCK),
}


@pytest.mark.integration
@pytest.mark.parametrize("corruption", list(_CORRUPTIONS))
@pytest.mark.parametrize("entry_point", ["predict", "viterbi", "likelihood", "cached"])
def test_stale_or_unknown_contract_is_rejected_before_likelihood(
    contract_recording, corruption, entry_point, monkeypatch
):
    """A saved model whose units or transition clock cannot be trusted is
    refused, naming the cause, before any likelihood work and without changing
    the model. It is pickled after the corruption, as a stored model would be,
    so unpickling must not repair it."""
    fitted, time, position, args = contract_recording
    corrupt, message = _CORRUPTIONS[corruption]
    corrupted = copy.deepcopy(fitted)
    corrupt(corrupted)
    detector = pickle.loads(pickle.dumps(corrupted))
    edges = np.arange(6) * 0.002

    if entry_point != "likelihood":

        def unexpected_likelihood(*args, **kwargs):
            pytest.fail("An untrusted model must fail before evaluating likelihoods")

        monkeypatch.setattr(detector, "compute_log_likelihood", unexpected_likelihood)

    if entry_point == "predict":
        call = functools.partial(detector.predict, *args, time_edges=edges)
    elif entry_point == "viterbi":
        call = functools.partial(detector.most_likely_sequence, *args, time_edges=edges)
    elif entry_point == "likelihood":
        call = functools.partial(
            detector.compute_log_likelihood, time, position, *args, time_edges=edges
        )
    else:
        call = functools.partial(
            detector._predict,
            edges,
            log_likelihoods=np.zeros((5, detector.is_track_interior_state_bins_.sum())),
        )
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match=f"{message}; refit before decoding"):
        call()
    _assert_unchanged(detector, snapshot)


@pytest.mark.integration
# max_iter=1 is a speed cap, so EM stops before convergence.
@pytest.mark.filterwarnings("ignore:EM did not converge after max_iter:UserWarning")
def test_estimation_refits_legacy_contract_before_decoding(contract_recording):
    """The supported recovery path fits physical units from original recording."""
    fitted, time, position, args = contract_recording
    detector = copy.deepcopy(fitted)
    del detector.time_contract_
    del detector.transition_time_bin_width_
    for encoding in detector.encoding_model_.values():
        del encoding["rate_units"]
    edges = np.arange(81) * 0.002
    result = detector.estimate_parameters(
        time,
        position,
        *args,
        time_edges=edges,
        max_iter=1,
        estimate_discrete_transition=False,
        estimate_encoding_model=False,
    )
    assert np.all(np.isfinite(result.acausal_posterior))
    assert detector.time_contract_ == fitted.time_contract_
    assert detector.transition_time_bin_width_ is None
    for encoding in detector.encoding_model_.values():
        assert encoding["rate_units"] == "Hz"
        assert encoding["encoding_exposure_seconds"] == pytest.approx(0.2)
