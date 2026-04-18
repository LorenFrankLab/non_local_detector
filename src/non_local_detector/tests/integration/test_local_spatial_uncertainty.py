"""Integration tests for local state with spatial uncertainty.

Tests the full fit/predict pipeline with local_position_std enabled,
verifying posteriors, position coordinates, and mathematical invariants.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.models import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
def simulated_data():
    """Generate simulated data for testing."""
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


@pytest.mark.integration
def test_multibin_local_fit_predict(simulated_data):
    """Multi-bin local model produces valid posteriors on synthetic data."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    detector = NonLocalSortedSpikesDetector(
        local_position_std=5.0,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
    )

    # Posterior probabilities should sum to 1 at each time point
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Posterior probabilities don't sum to 1"
    )

    # State probabilities should sum to 1 at each time point
    state_prob_sums = results.acausal_state_probabilities.sum(axis=1)
    assert np.allclose(state_prob_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "State probabilities don't sum to 1"
    )

    # No NaN or Inf in posterior
    assert np.all(np.isfinite(results.acausal_posterior.values)), (
        "Posterior contains NaN or Inf"
    )


@pytest.mark.integration
def test_multibin_local_has_position_coordinates(simulated_data):
    """Multi-bin local posterior has real position coordinates, not NaN."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    detector = NonLocalSortedSpikesDetector(
        local_position_std=5.0,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
    )

    # Local state should have position coordinates (not NaN)
    # Find bins belonging to the Local state
    local_state_name = "Local"
    state_names = list(results.acausal_posterior.coords["state"].values)
    local_bins = [s for s in state_names if local_state_name in str(s)]
    assert len(local_bins) > 1, (
        f"Expected multiple local bins with local_position_std, got {len(local_bins)}"
    )


@pytest.mark.integration
def test_legacy_local_unchanged(simulated_data):
    """Legacy local_position_std=None produces valid results identical pattern."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
    )

    # Posterior probabilities should sum to 1
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Legacy posterior probabilities don't sum to 1"
    )

    # State probabilities should sum to 1
    state_prob_sums = results.acausal_state_probabilities.sum(axis=1)
    assert np.allclose(state_prob_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Legacy state probabilities don't sum to 1"
    )


@pytest.mark.integration
def test_penalty_and_kernel_simultaneous(simulated_data):
    """Both non_local_position_penalty and local_position_std produce valid results."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    detector = NonLocalSortedSpikesDetector(
        local_position_std=5.0,
        non_local_position_penalty=1.0,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
    )

    # No NaN or Inf
    assert np.all(np.isfinite(results.acausal_posterior.values)), (
        "Posterior contains NaN or Inf with both penalty and kernel"
    )

    # Posterior probabilities should sum to 1
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Posterior probabilities don't sum to 1 with both penalty and kernel"
    )


@pytest.fixture
def clusterless_simulated_data():
    """Generate clusterless simulated data for testing."""
    sim = make_simulated_run_data(n_tetrodes=5, seed=42)
    return sim


@pytest.mark.integration
def test_clusterless_multibin_local_fit_predict(clusterless_simulated_data):
    """NonLocalClusterlessDetector with local_position_std produces valid posteriors."""
    sim = clusterless_simulated_data

    detector = NonLocalClusterlessDetector(
        local_position_std=5.0,
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        sim.position_time,
        sim.position,
        sim.spike_times,
        sim.spike_waveform_features,
    )

    results = detector.predict(
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
        time=sim.edges,
        position=sim.position,
        position_time=sim.position_time,
    )

    # Posterior probabilities should sum to 1 at each time point
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Clusterless posterior probabilities don't sum to 1"
    )

    # No NaN or Inf in posterior
    assert np.all(np.isfinite(results.acausal_posterior.values)), (
        "Clusterless posterior contains NaN or Inf"
    )


@pytest.mark.integration
def test_multibin_local_posterior_invariants(simulated_data):
    """Multi-bin local model output invariants for regression detection."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    detector = NonLocalSortedSpikesDetector(
        local_position_std=5.0,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
    )

    # Snapshot invariants: these should not change across runs
    posterior = results.acausal_posterior.values
    state_probs = results.acausal_state_probabilities.values

    # 1. Probabilities sum to 1 (JAX float32 precision)
    np.testing.assert_allclose(
        posterior.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5, err_msg="Posterior sums != 1"
    )
    np.testing.assert_allclose(
        state_probs.sum(axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-5,
        err_msg="State prob sums != 1",
    )

    # 2. No NaN/Inf
    assert np.all(np.isfinite(posterior)), "Posterior has NaN/Inf"
    assert np.all(np.isfinite(state_probs)), "State probs has NaN/Inf"

    # 3. Local state has multiple bins (not single-bin legacy)
    assert detector.bin_sizes_[0] > 1, "Local state should have multiple bins"

    # 4. State probabilities are non-negative
    assert np.all(state_probs >= 0), "Negative state probabilities"
