"""Physical rate units and model persistence through public decoder APIs.

- Every likelihood backend fits rates in Hz, so an empty bin's log likelihood
  is minus the rate times the bin width.
- Every backend keeps its units and decode results through ``save_model`` /
  ``load_model``.
- A saved model with a legacy or unknown time contract or invalid
  transition-clock provenance is rejected, before any likelihood work, with a
  hint to refit; ``estimate_parameters`` refits such a model from the original
  recording.
"""

import copy
import pickle

import numpy as np
import pytest
import xarray as xr

from non_local_detector.exceptions import NotFittedError, ValidationError
from non_local_detector.models._input_validation import _decode_time_edges
from non_local_detector.models.base import _DetectorBase
from tests.conftest import em_stops_at_max_iter
from tests.models._helpers import (
    ALGORITHMS,
    ENTRY_POINTS,
    _assert_unchanged,
    _decoder,
    _snapshot,
    call_entry_point,
    forbid_likelihood_and_hmm_work,
)


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


def _set_contract_version(detector):
    detector.time_contract_ = {**detector.time_contract_, "version": 999}


def _delete_transition_clock(detector):
    del detector.transition_time_bin_width_


def _set_nan_transition_clock(detector):
    detector.transition_time_bin_width_ = np.nan


_UNKNOWN_CONTRACT = "Saved model has a legacy or unknown time contract"
_INVALID_CLOCK = "Saved model has invalid transition clock provenance"

# One corruption of a fitted model per refusal, and the cause its refit error
# names. A missing time contract, and missing or unknown encoding units, are
# refused at every entry point in test_model_unit_compatibility.
_CORRUPTIONS = {
    "unknown-contract-version": (_set_contract_version, _UNKNOWN_CONTRACT),
    "missing-transition-clock": (_delete_transition_clock, _UNKNOWN_CONTRACT),
    # Unknown provenance cannot disable the learned-width comparison.
    "nan-transition-clock": (_set_nan_transition_clock, _INVALID_CLOCK),
}


@pytest.mark.integration
@pytest.mark.parametrize("corruption", list(_CORRUPTIONS))
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_stale_or_unknown_contract_is_rejected_before_likelihood(
    contract_recording, corruption, entry_point, monkeypatch
):
    """A saved model whose time contract or transition clock cannot be trusted
    is refused, naming the cause, before any likelihood work and without
    changing the model. It is pickled after the corruption, as a stored model
    would be, so unpickling must not repair it."""
    fitted, time, position, args = contract_recording
    corrupt, message = _CORRUPTIONS[corruption]
    corrupted = copy.deepcopy(fitted)
    corrupt(corrupted)
    detector = pickle.loads(pickle.dumps(corrupted))
    forbid_likelihood_and_hmm_work(monkeypatch, detector)
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match=f"{message}; refit before decoding"):
        call_entry_point(
            detector, entry_point, time, position, args, np.arange(6) * 0.002
        )
    _assert_unchanged(detector, snapshot)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("width", "tolerance"),
    [
        (np.nan, 0.0),
        (np.inf, 0.0),
        # A negative width would also exceed the tolerance bound; zero does not.
        (0.0, 0.0),
        # An infinite tolerance would also exceed the bound; NaN does not.
        (0.002, np.nan),
        (0.002, -1.0),
        (0.002, 1.0),  # above the timestamp-precision bound for a 2 ms width
    ],
    ids=[
        "nan-width",
        "infinite-width",
        "zero-width",
        "nan-tolerance",
        "negative-tolerance",
        "tolerance-above-bound",
    ],
)
def test_invalid_transition_clock_provenance_is_refused(width, tolerance):
    """The learned bin width must be finite and positive, with a finite
    nonnegative tolerance within the precision bound, for edges of that width."""
    with pytest.raises(ValidationError, match=f"{_INVALID_CLOCK}; refit"):
        _decode_time_edges(np.arange(6) * 0.002, width, tolerance)


@pytest.mark.unit
def test_base_time_contract_check_fails_closed(contract_recording):
    """A subclass that does not override the hook still gets the contract check."""
    fitted, *_ = contract_recording
    with pytest.raises(NotFittedError, match="not fitted"):
        _DetectorBase._validate_time_contract(_decoder("sorted_spikes_kde"))
    detector = copy.deepcopy(fitted)
    _DetectorBase._validate_time_contract(detector)
    detector.time_contract_ = {**detector.time_contract_, "rate_units": "per_bin"}
    with pytest.raises(ValidationError, match=f"{_UNKNOWN_CONTRACT}; refit"):
        _DetectorBase._validate_time_contract(detector)


@pytest.mark.integration
# max_iter=1 is a speed cap, so EM stops before convergence.
@em_stops_at_max_iter
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
