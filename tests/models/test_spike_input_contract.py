"""Spike data is checked where it enters, the same way at fit and at decode.

``fit``, ``fit_encoding_model`` and ``estimate_parameters`` check the spike
arrays' structure as ``predict`` does.
"""

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder, Environment, SortedSpikesDecoder
from non_local_detector.exceptions import ValidationError
from tests.models._helpers import _assert_unchanged, _snapshot

N_TIME = 100
TIME = np.arange(N_TIME) * 0.002
POSITION = (50 + 8 * np.sin(TIME * 40))[:, None]
EDGES = np.arange(41) * 0.002
SPIKE_TIMES = [np.array([0.021, 0.067, 0.121, 0.167]), np.array([0.03, 0.09])]
FEATURES = [
    np.array([[0.1], [-0.1], [0.2], [-0.2]]),
    np.array([[0.3], [-0.3]]),
]
FIT_METHODS = ["fit", "fit_encoding_model", "estimate_parameters"]


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
