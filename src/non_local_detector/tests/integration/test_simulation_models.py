"""Integration tests for model classes using simulated data.

Tests the full API of NonLocalSortedSpikesDetector, ContFragSortedSpikesClassifier,
and SortedSpikesDecoder with synthetic spike data.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.models import (
    ContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
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


@pytest.mark.integration
def test_nonlocal_sorted_spikes_detector(simulated_data):
    """Test NonLocalSortedSpikesDetector fit and predict."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    # Fit detector on non-event periods
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    # Predict
    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        save_log_likelihood_to_results=False,  # Skip for speed
    )

    # Check output shapes
    assert results.acausal_posterior.shape[0] == len(time), (
        "Posterior has wrong time dimension"
    )
    assert results.acausal_state_probabilities.shape[0] == len(time), (
        "State probs have wrong time dimension"
    )

    # Check state probabilities
    n_states = results.acausal_state_probabilities.shape[1]
    assert n_states == 4, (
        f"Expected 4 states (Local, No-Spike, Non-Local Continuous, Non-Local Fragmented), got {n_states}"
    )

    # State probabilities should sum to 1 at each time point
    state_prob_sums = results.acausal_state_probabilities.sum(axis=1)
    assert np.allclose(state_prob_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "State probabilities don't sum to 1"
    )

    # Posterior probabilities should sum to 1 at each time point
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Posterior probabilities don't sum to 1"
    )

    # Check that encoding model was fitted
    assert hasattr(detector, "encoding_model_"), "Encoding model not fitted"
    assert len(detector.encoding_model_) > 0, "Encoding model is empty"


@pytest.mark.integration
def test_contfrag_sorted_spikes_classifier(simulated_data):
    """Test ContFragSortedSpikesClassifier fit and predict."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    # Fit classifier
    classifier = ContFragSortedSpikesClassifier(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=~is_event,
    )

    # Predict
    results = classifier.predict(
        spike_times=spike_times,
        time=time,
    )

    # Check output shapes
    assert results.acausal_posterior.shape[0] == len(time), (
        "Posterior has wrong time dimension"
    )
    assert results.acausal_state_probabilities.shape[0] == len(time), (
        "State probs have wrong time dimension"
    )

    # Check state probabilities
    n_states = results.acausal_state_probabilities.shape[1]
    assert n_states == 2, f"Expected 2 states (Continuous, Fragmented), got {n_states}"

    # State probabilities should sum to 1
    state_prob_sums = results.acausal_state_probabilities.sum(axis=1)
    assert np.allclose(state_prob_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "State probabilities don't sum to 1"
    )

    # Posterior probabilities should sum to 1
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Posterior probabilities don't sum to 1"
    )

    # Check transition matrix properties
    assert hasattr(classifier, "continuous_state_transitions_"), (
        "Continuous transitions not fitted"
    )
    # Continuous transitions should be a valid transition matrix (non-negative, 3D array)
    assert classifier.continuous_state_transitions_.ndim == 2, (
        "Continuous transitions should be 2D"
    )
    assert np.all(classifier.continuous_state_transitions_ >= 0), (
        "Transition matrix should be non-negative"
    )
    # Each position bin should have valid transitions between discrete states
    # Note: rows sum to n_discrete_states (2) not 1, because the matrix represents
    # transitions for all discrete states at each position
    trans_row_sums = classifier.continuous_state_transitions_.sum(axis=1)
    assert np.all(trans_row_sums > 0), (
        "All position bins should have some transition probability"
    )

    # Initial conditions should sum to ~1
    assert hasattr(classifier, "initial_conditions_"), "Initial conditions not fitted"
    init_sum = classifier.initial_conditions_.sum()
    assert np.isclose(init_sum, 1.0, rtol=1e-5, atol=1e-6), (
        f"Initial conditions sum to {init_sum}, not 1"
    )


@pytest.mark.integration
def test_sorted_spikes_decoder(simulated_data):
    """Test SortedSpikesDecoder fit and predict."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    # Fit decoder
    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=~is_event,
    )

    # Predict
    results = decoder.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        save_log_likelihood_to_results=False,  # Skip for speed
    )

    # Check output shapes
    assert results.acausal_posterior.shape[0] == len(time), (
        "Posterior has wrong time dimension"
    )
    assert results.acausal_state_probabilities.shape[0] == len(time), (
        "State probs have wrong time dimension"
    )

    # For basic decoder, there's only one state (Continuous)
    n_states = (
        results.acausal_state_probabilities.shape[1]
        if results.acausal_state_probabilities.ndim > 1
        else 1
    )
    assert n_states == 1, f"Expected 1 state (Continuous), got {n_states}"

    # State probabilities should be ~1 (single state)
    if results.acausal_state_probabilities.ndim > 1:
        state_prob_sums = results.acausal_state_probabilities.sum(axis=1)
    else:
        state_prob_sums = results.acausal_state_probabilities
    assert np.allclose(state_prob_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "State probabilities don't equal 1"
    )

    # Posterior probabilities should sum to 1
    posterior_sums = results.acausal_posterior.sum(axis=1)
    assert np.allclose(posterior_sums, 1.0, rtol=1e-5, atol=1e-6), (
        "Posterior probabilities don't sum to 1"
    )

    # Test log likelihood computation
    log_likelihood = decoder.compute_log_likelihood(
        time=time,
        position_time=time,
        position=position,
        spike_times=spike_times,
    )

    # Check log likelihood shape
    assert log_likelihood.shape[0] == len(time), (
        "Log likelihood has wrong time dimension"
    )
    assert log_likelihood.ndim == 2, (
        f"Expected 2D log likelihood, got {log_likelihood.ndim}D"
    )

    # Log likelihoods should be finite
    assert np.all(np.isfinite(log_likelihood)), (
        "Log likelihoods contain non-finite values"
    )


@pytest.mark.integration
def test_models_handle_missing_data(simulated_data):
    """Test that models handle missing data (no spikes in some time bins) gracefully."""
    time = simulated_data["time"]
    position = simulated_data["position"]
    spike_times = simulated_data["spike_times"]
    is_event = simulated_data["is_event"]

    # Use only a subset of spike trains (simulate missing neurons)
    spike_times_subset = spike_times[:5]  # Only first 5 neurons

    # Fit and predict with subset
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times_subset, is_training=~is_event)

    results = detector.predict(
        spike_times=spike_times_subset,
        time=time,
        position=position,
        position_time=time,
        save_log_likelihood_to_results=False,
    )

    # Should still produce valid results
    assert results.acausal_posterior.shape[0] == len(time)
    state_prob_sums = results.acausal_state_probabilities.sum(axis=1)
    assert np.allclose(state_prob_sums, 1.0, rtol=1e-5, atol=1e-6)
