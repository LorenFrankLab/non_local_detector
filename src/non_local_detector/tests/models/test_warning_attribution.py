"""Warnings raised deep inside the library point at the caller's code."""

import warnings

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def recording():
    _, position, spike_times, time, *_ = make_simulated_data(seed=0, n_neurons=4)
    n = 2000
    time, position = time[:n], position[:n]
    return {
        "position_time": time,
        "position": position,
        "spike_times": [st[st <= time[-1]] for st in spike_times],
    }


def _sources(match, call):
    """Files that warnings whose message contains ``match`` were attributed to."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    sources = {w.filename for w in caught if match in str(w.message)}
    assert sources, f"no warning matching {match!r}"
    return sources


def _past_tracking(time):
    """Decode edges that run five bins past the tracked period."""
    edges = time_edges_from_centers(time)
    return np.concatenate([edges, edges[-1] + np.diff(edges[-2:]) * np.arange(1, 6)])


@pytest.mark.parametrize("entry", ["fit", "estimate_parameters"])
def test_one_dimensional_position_warning(recording, entry):
    detector = NonLocalSortedSpikesDetector()
    if entry == "fit":
        call = lambda: detector.fit(**recording)  # noqa: E731
    else:
        edges = time_edges_from_centers(recording["position_time"])
        call = lambda: detector.estimate_parameters(  # noqa: E731
            **recording, time_edges=edges, max_iter=1
        )
    assert _sources("appears to be 1D", call) == {__file__}


def test_em_not_converged_warning(recording):
    edges = time_edges_from_centers(recording["position_time"])
    detector = NonLocalSortedSpikesDetector()
    call = lambda: detector.estimate_parameters(  # noqa: E731
        **recording, time_edges=edges, max_iter=1
    )
    assert _sources("EM did not converge", call) == {__file__}


def test_deprecated_output_flag_warning(recording):
    detector = NonLocalSortedSpikesDetector().fit(**recording)
    edges = time_edges_from_centers(recording["position_time"])
    call = lambda: detector.predict(  # noqa: E731
        spike_times=recording["spike_times"],
        time_edges=edges,
        position=recording["position"],
        position_time=recording["position_time"],
        save_log_likelihood_to_results=True,
    )
    assert _sources("deprecated", call) == {__file__}


@pytest.mark.parametrize(
    "entry", ["predict", "compute_log_likelihood", "most_likely_sequence"]
)
def test_outside_tracking_warning(recording, entry):
    detector = NonLocalSortedSpikesDetector().fit(**recording)
    edges = _past_tracking(recording["position_time"])
    if entry == "compute_log_likelihood":
        call = lambda: detector.compute_log_likelihood(  # noqa: E731
            recording["position_time"],
            recording["position"],
            recording["spike_times"],
            time_edges=edges,
        )
    else:
        call = lambda: getattr(detector, entry)(  # noqa: E731
            spike_times=recording["spike_times"],
            time_edges=edges,
            position=recording["position"],
            position_time=recording["position_time"],
        )
    assert _sources("outside the tracked period", call) == {__file__}
