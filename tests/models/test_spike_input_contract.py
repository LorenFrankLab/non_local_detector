"""Spike data is checked where it enters, the same way at fit and at decode.

``fit``, ``fit_encoding_model`` and ``estimate_parameters`` check the spike
arrays' structure as ``predict`` does, every entry point rejects non-finite
spike times (which were dropped silently), and a recording whose spikes all
fall outside the fit or decode time range warns of a unit or clock mismatch.
"""

import warnings

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder, Environment, SortedSpikesDecoder
from non_local_detector.exceptions import DataError, DataQualityWarning, ValidationError
from non_local_detector.models._input_validation import _validate_spike_data
from tests.models._helpers import _assert_unchanged, _snapshot

N_TIME = 100
TIME = np.arange(N_TIME) * 0.002
POSITION = (50 + 8 * np.sin(TIME * 40))[:, None]
EDGES = np.arange(41) * 0.002
EPOCH = 1.6e9  # a Unix time in seconds (2020-09-13)
SPIKE_TIMES = [np.array([0.021, 0.067, 0.121, 0.167]), np.array([0.03, 0.09])]
FEATURES = [
    np.array([[0.1], [-0.1], [0.2], [-0.2]]),
    np.array([[0.3], [-0.3]]),
]
FIT_METHODS = ["fit", "fit_encoding_model", "estimate_parameters"]
DECODE_METHODS = ["predict", "most_likely_sequence", "compute_log_likelihood"]


def _environment():
    return Environment(place_bin_size=5, position_range=((40, 60),))


def _sorted_decoder():
    return SortedSpikesDecoder(
        environments=_environment(),
        infer_track_interior=False,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


def _clusterless_decoder():
    return ClusterlessDecoder(
        environments=_environment(),
        infer_track_interior=False,
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={"disable_progress_bar": True},
    )


def _family(family):
    """The family's decoder factory and its valid spike data."""
    if family == "sorted":
        return _sorted_decoder, (SPIKE_TIMES,)
    return _clusterless_decoder, (SPIKE_TIMES, FEATURES)


@pytest.fixture(scope="module")
def fitted():
    """A fitted decoder of each family."""
    return {
        family: factory().fit(TIME, POSITION, *spike_data)
        for family, (factory, spike_data) in (
            (name, _family(name)) for name in ("sorted", "clusterless")
        )
    }


def _call(detector, method, spike_data, **extra):
    """Call ``method`` with the recording's position and ``spike_data``."""
    if method in ("fit", "fit_encoding_model"):
        return getattr(detector, method)(TIME, POSITION, *spike_data, **extra)
    if method == "estimate_parameters":
        return detector.estimate_parameters(
            TIME, POSITION, *spike_data, time_edges=EDGES, max_iter=1, **extra
        )
    if method == "compute_log_likelihood":
        return detector.compute_log_likelihood(
            TIME, POSITION, *spike_data, time_edges=EDGES, **extra
        )
    return getattr(detector, method)(
        *spike_data, time_edges=EDGES, position=POSITION, position_time=TIME, **extra
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", FIT_METHODS)
@pytest.mark.parametrize(
    "spike_times",
    [SPIKE_TIMES[0], [SPIKE_TIMES[0][:, np.newaxis]]],
    ids=["bare_array", "column_per_neuron"],
)
def test_sorted_fit_checks_spike_structure_like_predict(fitted, method, spike_times):
    """A bare array of one neuron's times raised ``TypeError: len() of unsized
    object`` at fit, and a 2-D array per neuron a NumPy concatenation error;
    predict rejected both with this ValidationError."""
    detector = fitted["sorted"].copy()
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match="spike_times for neuron 0 must be 1-D"):
        _call(detector, method, (spike_times,))
    _assert_unchanged(detector, snapshot)
    with pytest.raises(ValidationError, match="spike_times for neuron 0 must be 1-D"):
        detector.predict(spike_times, time_edges=EDGES)


@pytest.mark.unit
@pytest.mark.parametrize("method", FIT_METHODS)
def test_sorted_spike_structure_is_checked_before_fitting(monkeypatch, method):
    detector = _sorted_decoder()

    def unexpected_fit(*args, **kwargs):
        pytest.fail("spike_times must be checked before the internal fit")

    monkeypatch.setattr(type(detector), "_fit", unexpected_fit)
    with pytest.raises(ValidationError, match="neuron 0"):
        _call(detector, method, (SPIKE_TIMES[0],))


@pytest.mark.unit
@pytest.mark.parametrize("method", FIT_METHODS + DECODE_METHODS)
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
def test_non_finite_spike_times_raise(fitted, method, family, bad_value):
    """NaN spike times were dropped silently (the posterior did not change);
    they now raise, naming the unit and the count."""
    unit = "neuron" if family == "sorted" else "electrode"
    spike_times = [SPIKE_TIMES[0], np.append(SPIKE_TIMES[1], [bad_value, bad_value])]
    spike_data = (spike_times,) if family == "sorted" else (spike_times, FEATURES)
    if family == "clusterless":
        spike_data = (spike_times, [FEATURES[0], np.vstack([FEATURES[1], [[0], [0]]])])
    detector = fitted[family].copy()
    snapshot = _snapshot(detector)
    with pytest.raises(DataError, match=rf"{unit} 1: 2 of 4"):
        _call(detector, method, spike_data)
    _assert_unchanged(detector, snapshot)


@pytest.mark.unit
# max_iter=1 is a speed cap, so EM stops before convergence.
@pytest.mark.filterwarnings("ignore:EM did not converge after max_iter:UserWarning")
@pytest.mark.parametrize("method", ["fit", "fit_encoding_model", "estimate_parameters"])
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_spikes_all_outside_position_time_warn_at_fit(fitted, method, family):
    """Spike times on the Unix-epoch clock against position_time from 0 lie
    far outside the encoding period; the encoding model is fit to no spikes."""
    _, spike_data = _family(family)
    on_epoch = ([times + EPOCH for times in spike_data[0]], *spike_data[1:])
    detector = fitted[family].copy()
    with pytest.warns(DataQualityWarning, match="None of the 6 spike times"):
        _call(detector, method, on_epoch)


@pytest.mark.unit
def test_spikes_in_ms_warn_at_fit():
    """Spike times in ms of a recording that starts at 100 s lie about 1e5 s
    after its position_time span."""
    with pytest.warns(DataQualityWarning, match="position_time span"):
        _sorted_decoder().fit(
            TIME + 100.0, POSITION, [(times + 100.0) * 1_000 for times in SPIKE_TIMES]
        )


@pytest.mark.unit
@pytest.mark.parametrize("method", DECODE_METHODS)
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_spikes_all_outside_time_edges_warn_at_decode(fitted, method, family):
    _, spike_data = _family(family)
    on_epoch = ([times + EPOCH for times in spike_data[0]], *spike_data[1:])
    with pytest.warns(DataQualityWarning, match=r"decode time_edges \[0, 0.08\] s"):
        _call(fitted[family], method, on_epoch)


def _warns_out_of_range(spike_times, time_range):
    """Whether ``_validate_spike_data`` warns that the spikes are out of range."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_spike_data(
            spike_times,
            clusterless=False,
            time_range=time_range,
            range_name="decode time_edges",
        )
    return any("None of the" in str(warning.message) for warning in caught)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spike_times", "time_range", "warns"),
    [
        # The last replay event of a session, after the last spike.
        (SPIKE_TIMES, (0.18, 0.19), False),
        # A window before the first spike.
        (SPIKE_TIMES, (0.0, 0.01), False),
        # Spike times in ms of a recording from 100 to 110 s.
        ([np.array([100.021, 105.0, 109.9]) * 1_000], (100.0, 110.0), True),
        # Spikes from 0 against Unix-epoch edges, and the reverse.
        (SPIKE_TIMES, (EPOCH, EPOCH + 0.08), True),
        ([times + EPOCH for times in SPIKE_TIMES], (0.0, 0.08), True),
    ],
    ids=["after_last_spike", "before_first_spike", "ms", "epoch_edges", "epoch_spikes"],
)
def test_out_of_range_warning_needs_a_gap_larger_than_the_spike_span(
    spike_times, time_range, warns
):
    """A unit or clock mismatch puts the spikes farther from the range than
    they span; a quiet window next to the spikes is data."""
    assert _warns_out_of_range(spike_times, time_range) is warns


@pytest.mark.unit
@pytest.mark.parametrize("method", ["fit", "predict"])
def test_one_silent_neuron_does_not_warn(fitted, method):
    """Only a recording with no spike in range warns; a neuron whose spikes
    all lie outside is not a unit mismatch."""
    spike_times = [SPIKE_TIMES[0], SPIKE_TIMES[1] + 10.0]
    detector = fitted["sorted"].copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "None of the", DataQualityWarning)
        _call(detector, method, (spike_times,))


@pytest.mark.unit
def test_quiet_window_between_spikes_does_not_warn(fitted):
    """A decode window with spikes before and after it but none in it is a
    quiet period, not a clock mismatch."""
    spike_times = [np.array([0.001, 0.15]), np.array([0.19])]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "None of the", DataQualityWarning)
        fitted["sorted"].predict(spike_times, time_edges=EDGES[10:20])


@pytest.mark.unit
def test_no_spikes_does_not_warn(fitted):
    detector = fitted["sorted"].copy()
    empty = [np.array([]), np.array([])]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "None of the", DataQualityWarning)
        detector.fit(TIME, POSITION, empty)
        detector.predict(empty, time_edges=EDGES)


def _column_container(family):
    """One neuron's times as an (n_spikes, 1) array: n_spikes one-spike units."""
    column = SPIKE_TIMES[0][:, np.newaxis]
    if family == "sorted":
        return (column,)
    return column, [row[np.newaxis] for row in FEATURES[0]]


@pytest.mark.unit
@pytest.mark.parametrize("method", FIT_METHODS + DECODE_METHODS)
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_numeric_2d_spike_times_container_raises(fitted, method, family):
    """An (n_spikes, 1) array iterated as n_spikes units of one spike each,
    which every per-unit check accepted."""
    detector = fitted[family].copy()
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match="spike_times must be a list"):
        _call(detector, method, _column_container(family))
    _assert_unchanged(detector, snapshot)


@pytest.mark.unit
@pytest.mark.parametrize(
    "container",
    [tuple, lambda times: np.array(times, dtype=object)],
    ids=["tuple", "object_array"],
)
def test_sequence_and_object_array_containers_are_accepted(fitted, container):
    detector = fitted["sorted"]
    expected = detector.predict(SPIKE_TIMES, time_edges=EDGES).acausal_posterior
    result = detector.predict(container(SPIKE_TIMES), time_edges=EDGES)
    np.testing.assert_array_equal(result.acausal_posterior, expected)


EQUAL_LENGTH_SPIKES = [
    np.array([0.021, 0.067, 0.121, 0.167]),
    np.array([0.03, 0.09, 0.1, 0.15]),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "container",
    [
        lambda times: np.array(times, dtype=object),  # 2-D object array
        lambda times: [unit.astype(object) for unit in times],
    ],
    ids=["2d_object_array", "object_unit_arrays"],
)
@pytest.mark.parametrize("method", DECODE_METHODS)
def test_object_dtype_spike_times_match_float_arrays(fitted, container, method):
    """Object arrays of numbers decode as their float arrays, as before the
    spike checks: a 2-D object array of equal-length units, and object-dtype
    arrays per unit. (Fitting on them fails in np.interp, as it did before.)"""
    detector = fitted["sorted"]
    expected = _call(detector, method, (EQUAL_LENGTH_SPIKES,))
    result = _call(detector, method, (container(EQUAL_LENGTH_SPIKES),))
    if method == "predict":
        expected, result = expected.acausal_posterior, result.acausal_posterior
    elif method == "most_likely_sequence":
        expected, result = expected.to_numpy(), result.to_numpy()
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@pytest.mark.unit
def test_single_unit_object_array_matches_float_array(fitted):
    detector = fitted["sorted"]
    one_unit = [EQUAL_LENGTH_SPIKES[0], np.array([])]
    expected = detector.predict(one_unit, time_edges=EDGES).acausal_posterior
    as_object = [np.array([EQUAL_LENGTH_SPIKES[0]], dtype=object)[0], np.array([])]
    result = detector.predict(as_object, time_edges=EDGES).acausal_posterior
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_non_numeric_object_spike_times_raise(fitted):
    with pytest.raises(ValidationError, match="must be numeric"):
        fitted["sorted"].predict(
            [np.array(["a", "b"], dtype=object), SPIKE_TIMES[1]], time_edges=EDGES
        )
