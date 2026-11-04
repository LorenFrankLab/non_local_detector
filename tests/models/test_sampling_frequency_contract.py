"""Decode time edges must be ``1 / sampling_frequency`` wide.

The detector's ``sampling_frequency`` sets the bin width of
``calculate_time_edges``; every method that takes ``time_edges`` rejects
edges of another width, within the tolerance that decides edges are uniform
(the larger of 4 ulp of the largest edge and a thousandth of the width).
"""

import numpy as np
import pytest

from non_local_detector import Environment, SortedSpikesDecoder
from non_local_detector.exceptions import ValidationError

N_TIME = 100
SPIKE_OFFSETS = np.array([0.021, 0.067, 0.121, 0.167])


def _decoder(sampling_frequency=500.0):
    return SortedSpikesDecoder(
        environments=Environment(place_bin_size=5, position_range=((40, 60),)),
        infer_track_interior=False,
        sampling_frequency=sampling_frequency,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


def _recording(start=0.0):
    time = start + np.arange(N_TIME) * 0.002
    position = (50 + 8 * np.sin((time - start) * 40))[:, None]
    return time, position, [start + SPIKE_OFFSETS]


def _decode(detector, method, edges, start=0.0):
    time, position, spike_times = _recording(start)
    if method == "estimate_parameters":
        return detector.estimate_parameters(
            time, position, spike_times, time_edges=edges, max_iter=1
        )
    if method == "compute_log_likelihood":
        return detector.compute_log_likelihood(
            time, position, spike_times, time_edges=edges
        )
    return getattr(detector, method)(
        spike_times, time_edges=edges, position=position, position_time=time
    )


METHODS = [
    "predict",
    "most_likely_sequence",
    "compute_log_likelihood",
    "estimate_parameters",
]


@pytest.fixture(scope="module")
def fitted():
    """Detectors fitted at origins 0, 1e5 and 1e6 s."""
    return {start: _decoder().fit(*_recording(start)) for start in (0.0, 1e5, 1e6)}


@pytest.mark.unit
# max_iter=1 is a speed cap, so EM stops before convergence.
@pytest.mark.filterwarnings("ignore:EM did not converge after max_iter:UserWarning")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("start", [0.0, 1e5, 1e6])
def test_calculate_time_edges_match_at_large_origins(fitted, method, start):
    detector = fitted[start].copy()
    edges = detector.calculate_time_edges((start, start + 0.08))
    _decode(detector, method, edges, start)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:EM did not converge after max_iter:UserWarning")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("start", [0.0, 1e5, 1e6])
def test_hand_built_edges_at_the_rate_match(fitted, method, start):
    detector = fitted[start].copy()
    edges = start + np.arange(41) / 500.0
    _decode(detector, method, edges, start)


@pytest.mark.unit
@pytest.mark.parametrize("method", METHODS)
def test_edges_at_another_rate_raise(fitted, method):
    """Edges at 250 Hz on a 500 Hz detector are twice the transitions' bin."""
    detector = fitted[0.0].copy()
    edges = np.arange(21) / 250.0
    with pytest.raises(ValidationError, match="sampling_frequency") as error:
        _decode(detector, method, edges)
    message = str(error.value)
    assert "0.002 s" in message
    assert "0.004 s" in message
    assert "sampling_frequency=250" in message


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:EM did not converge after max_iter:UserWarning")
@pytest.mark.parametrize("method", METHODS)
def test_relative_difference_within_the_uniformity_tolerance_passes(fitted, method):
    """A 1e-4 relative difference is below the thousandth-of-a-bin floor."""
    detector = fitted[0.0].copy()
    edges = np.arange(41) * 0.002 * (1 + 1e-4)
    _decode(detector, method, edges)


@pytest.mark.unit
def test_rate_mismatch_beyond_the_tolerance_raises(fitted):
    """Just above a thousandth of the width is a different rate."""
    detector = fitted[0.0].copy()
    with pytest.raises(ValidationError, match="sampling_frequency"):
        _decode(detector, "predict", np.arange(41) * 0.002 * (1 + 2e-3))


@pytest.mark.unit
def test_detector_at_the_edges_rate_decodes_them():
    time, position, spike_times = _recording()
    edges = np.arange(21) / 250.0
    detector = _decoder(sampling_frequency=250.0).fit(time, position, spike_times)
    detector.predict(spike_times, time_edges=edges)
