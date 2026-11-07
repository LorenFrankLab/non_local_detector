"""Integration tests for model classes using simulated data.

Tests the full API of NonLocalSortedSpikesDetector, ContFragSortedSpikesClassifier,
and SortedSpikesDecoder with synthetic spike data.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.models import (
    ContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def simulated_data():
    """Generate simulated data for testing.

    Uses same number of neurons as notebook.
    """
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=30)

    return {
        "speed": speed,
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "event_times": event_times,
        "sampling_frequency": sampling_frequency,
        "is_event": is_event,
        "place_fields": place_fields,
    }


@pytest.fixture(scope="module")
def nonlocal_detector(simulated_data):
    """NonLocalSortedSpikesDetector fitted on non-event periods; read-only."""
    return NonLocalSortedSpikesDetector(
        sampling_frequency=simulated_data["sampling_frequency"],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": (2**12),
        },
    ).fit(
        simulated_data["time"],
        simulated_data["position"],
        simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )


def median_map_error_while_running(results, simulated_data):
    """Median |MAP position - true position| over the non-event (running) bins."""
    posterior = results.acausal_posterior.unstack("state_bins").sum("state")
    map_position = posterior.idxmax("position").values
    true_position = np.asarray(simulated_data["position"]).squeeze()
    is_running = ~simulated_data["is_event"]
    return np.median(np.abs(map_position - true_position)[is_running])


def test_nonlocal_detector_is_local_while_running_and_non_local_in_events(
    simulated_data, nonlocal_detector
):
    """Running bins decode as Local; the simulated replay events as non-local.

    Measured: Local probability 0.80 while running and non-local probability
    0.59 in events. With each neuron's spikes circularly shifted (place tuning
    destroyed) they fall to 0.34 and 0.33.
    """
    time = simulated_data["time"]
    is_event = simulated_data["is_event"]

    results = nonlocal_detector.predict(
        spike_times=simulated_data["spike_times"],
        time_edges=time_edges_from_centers(time),
        position=simulated_data["position"],
        position_time=time,
    )

    state_probs = results.acausal_state_probabilities
    local = state_probs.sel(states="Local").values
    non_local = (
        state_probs.sel(states=["Non-Local Continuous", "Non-Local Fragmented"])
        .sum("states")
        .values
    )
    assert local[~is_event].mean() > 0.6, local[~is_event].mean()
    assert non_local[is_event].mean() > 0.45, non_local[is_event].mean()


def test_contfrag_classifier_decodes_position_while_running(simulated_data):
    """Measured median MAP error while running: 3.0 cm (29.8 cm with each
    neuron's spikes circularly shifted)."""
    time = simulated_data["time"]
    classifier = ContFragSortedSpikesClassifier(
        sampling_frequency=simulated_data["sampling_frequency"],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": (2**12),
        },
    ).fit(
        position_time=time,
        position=simulated_data["position"],
        spike_times=simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    results = classifier.predict(
        spike_times=simulated_data["spike_times"],
        time_edges=time_edges_from_centers(time),
    )

    error = median_map_error_while_running(results, simulated_data)
    assert error < 6.0, error


def test_sorted_spikes_decoder_decodes_position_while_running(simulated_data):
    """Measured median MAP error while running: 3.0 cm (28.7 cm with each
    neuron's spikes circularly shifted)."""
    time = simulated_data["time"]
    decoder = SortedSpikesDecoder(
        sampling_frequency=simulated_data["sampling_frequency"],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": (2**12),
        },
    ).fit(
        position_time=time,
        position=simulated_data["position"],
        spike_times=simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    results = decoder.predict(
        spike_times=simulated_data["spike_times"],
        time_edges=time_edges_from_centers(time),
        position=simulated_data["position"],
        position_time=time,
    )

    error = median_map_error_while_running(results, simulated_data)
    assert error < 6.0, error


def test_silent_period_shifts_mass_to_no_spike_state(simulated_data, nonlocal_detector):
    """Bins with no spikes from any neuron are attributed to the No-Spike state.

    Every spike in a 1 s window during running is removed, so those bins are
    empty for all 30 neurons. Over a 6 s decode window, the No-Spike
    probability must rise inside the silent second (about 0.59 on average)
    well above its level elsewhere (about 0.11), and the posterior stays valid.
    """
    window = slice(15_000, 24_000)  # 10-16 s
    time = simulated_data["time"][window]
    silent_start, silent_stop = 12.0, 13.0
    silenced = [
        spikes[(spikes < silent_start) | (spikes >= silent_stop)]
        for spikes in simulated_data["spike_times"]
    ]
    # Premise: the silent window really removed spikes.
    assert sum(len(s) for s in simulated_data["spike_times"]) > sum(
        len(s) for s in silenced
    )

    results = nonlocal_detector.predict(
        spike_times=silenced,
        time_edges=time_edges_from_centers(time),
        position=simulated_data["position"][window],
        position_time=time,
    )

    state_probs = results.acausal_state_probabilities
    assert np.all(np.isfinite(results.acausal_posterior.values))
    np.testing.assert_allclose(state_probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)

    no_spike = state_probs.sel(states="No-Spike").values
    is_silent = (time >= silent_start) & (time < silent_stop)
    silent_mean = no_spike[is_silent].mean()
    other_mean = no_spike[~is_silent].mean()
    assert silent_mean > 0.4, f"No-Spike mean in silent window = {silent_mean:.3f}"
    assert silent_mean > 3 * other_mean, (
        f"No-Spike mean: silent {silent_mean:.3f} vs elsewhere {other_mean:.3f}"
    )
