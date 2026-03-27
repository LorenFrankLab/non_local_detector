"""Tests for posterior probability mathematical properties.

These tests verify that posterior distributions satisfy mathematical properties
that must hold regardless of implementation details (normalization, causality,
entropy bounds). These tests catch numerical issues and implementation bugs.
"""

from typing import Any

import numpy as np
import pytest

from non_local_detector.models.decoder import ClusterlessDecoder
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

# Test configuration constants
POSITION_STD = 6.0
WAVEFORM_STD = 24.0
BLOCK_SIZE = 100
NORMALIZATION_RTOL = 1e-5
NORMALIZATION_ATOL = 1e-8
SIMULATION_SEED = 200


@pytest.fixture
def decoder_with_results() -> dict[str, Any]:
    """Fit decoder and get posterior results for property testing.

    Returns dict with keys:
        - sim: ClusterlessSimOutput with full simulation data
        - decoder: Fitted ClusterlessDecoder
        - results: xr.Dataset with posterior results
        - test_edges: Time bin edges used for decoding
    """
    # Generate simulated data (4 tetrodes, 16 neurons, 3 runs)
    sim = make_simulated_run_data(
        n_tetrodes=4,
        place_field_means=np.arange(0, 160, 10),
        sampling_frequency=500,
        n_runs=3,
        seed=SIMULATION_SEED,
    )

    # 70/30 train/test split
    n_encode = int(0.7 * len(sim.position_time))

    # Training data
    encode_position_time = sim.position_time[:n_encode]
    encode_position = sim.position[:n_encode]
    encode_spike_times = [st[st <= encode_position_time[-1]] for st in sim.spike_times]
    encode_spike_waveform_features = [
        swf[st <= encode_position_time[-1]]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ]

    # Fit decoder on training data
    decoder = ClusterlessDecoder(
        environments=sim.environment,
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": POSITION_STD,
            "waveform_std": WAVEFORM_STD,
            "block_size": BLOCK_SIZE,
        },
    )

    decoder.fit(
        position_time=encode_position_time,
        position=encode_position,
        spike_times=encode_spike_times,
        spike_waveform_features=encode_spike_waveform_features,
    )

    # Test data
    test_position_time = sim.position_time[n_encode:]
    test_spike_times = [st[st > encode_position_time[-1]] for st in sim.spike_times]
    test_spike_waveform_features = [
        swf[st > encode_position_time[-1]]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ]

    # Create test edges (50 time bins for decoding)
    test_edges = np.linspace(test_position_time[0], test_position_time[-1], 51)

    # Predict on test data
    results = decoder.predict(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
    )

    return {
        "sim": sim,
        "decoder": decoder,
        "results": results,
        "test_edges": test_edges,
    }


@pytest.mark.slow
def test_posterior_normalized(decoder_with_results: dict[str, Any]) -> None:
    """Test that posterior probabilities sum to 1 at each time point.

    Posterior distributions must be properly normalized (sum to 1 over all
    states and positions) at each time point. Violations indicate numerical
    issues in the forward-backward algorithm.
    """
    results = decoder_with_results["results"]
    acausal_posterior = results.acausal_posterior.values

    # Sum over all state bins (includes position and discrete states)
    posterior_sum = np.sum(acausal_posterior, axis=1)

    # Should be very close to 1.0 at each time point
    np.testing.assert_allclose(
        posterior_sum,
        1.0,
        rtol=NORMALIZATION_RTOL,
        atol=NORMALIZATION_ATOL,
        err_msg=f"Posterior not normalized: min={posterior_sum.min():.10f}, "
        f"max={posterior_sum.max():.10f}, mean={posterior_sum.mean():.10f}",
    )


@pytest.mark.slow
def test_acausal_smoother_than_causal(decoder_with_results: dict[str, Any]) -> None:
    """Test that acausal posterior has lower entropy than causal.

    The acausal (smoothed) posterior uses both past and future observations,
    so it should be more concentrated (lower entropy) than the causal
    (filtered) posterior which only uses past observations.
    """
    results = decoder_with_results["results"]

    # Get both posteriors
    acausal_posterior = results.acausal_posterior.values
    # Causal posterior is the forward pass probability
    # We need to normalize causal_state_probabilities if available
    if "causal_state_probabilities" in results:
        causal_probs = results.causal_state_probabilities.values
    else:
        pytest.skip("Causal probabilities not available in results")
        return

    # Compute entropy for each time point (adding small epsilon to avoid log(0))
    eps = 1e-10
    acausal_entropy = -np.sum(
        acausal_posterior * np.log(acausal_posterior + eps), axis=1
    )
    causal_entropy = -np.sum(causal_probs * np.log(causal_probs + eps), axis=1)

    # Mean entropy should be lower for acausal (more information used)
    mean_acausal_entropy = np.mean(acausal_entropy)
    mean_causal_entropy = np.mean(causal_entropy)

    assert (
        mean_acausal_entropy <= mean_causal_entropy
    ), f"Acausal entropy ({mean_acausal_entropy:.3f}) should be <= causal entropy ({mean_causal_entropy:.3f})"


@pytest.mark.slow
def test_posterior_entropy_reasonable(decoder_with_results: dict[str, Any]) -> None:
    """Test that posterior entropy is in reasonable range.

    Entropy must be between 0 (certain) and log(n_bins) (uniform distribution).
    Values outside this range indicate numerical errors.
    """
    results = decoder_with_results["results"]
    acausal_posterior = results.acausal_posterior.values

    # Compute entropy at each time point
    eps = 1e-10
    entropy = -np.sum(acausal_posterior * np.log(acausal_posterior + eps), axis=1)

    # Maximum possible entropy is log(n_bins)
    n_bins = acausal_posterior.shape[1]
    max_entropy = np.log(n_bins)

    # All entropy values should be in [0, log(n_bins)]
    assert np.all(entropy >= 0), f"Found negative entropy: min={entropy.min():.3f}"
    assert np.all(
        entropy <= max_entropy
    ), f"Entropy {entropy.max():.3f} exceeds maximum {max_entropy:.3f}"

    # Check that we're not always at maximum entropy (would indicate uniform posterior)
    mean_entropy = np.mean(entropy)
    assert (
        mean_entropy < 0.95 * max_entropy
    ), f"Mean entropy {mean_entropy:.3f} too close to maximum {max_entropy:.3f} (uniform)"


@pytest.mark.slow
def test_posterior_no_nans_or_infs(decoder_with_results: dict[str, Any]) -> None:
    """Test that posterior contains no NaN or infinite values.

    NaN or infinite values indicate numerical instability in the computation.
    """
    results = decoder_with_results["results"]
    acausal_posterior = results.acausal_posterior.values

    assert np.all(
        np.isfinite(acausal_posterior)
    ), "Posterior contains NaN or Inf values"
    assert not np.any(np.isnan(acausal_posterior)), "Posterior contains NaN values"
    assert not np.any(np.isinf(acausal_posterior)), "Posterior contains Inf values"


@pytest.mark.slow
def test_posterior_all_nonnegative(decoder_with_results: dict[str, Any]) -> None:
    """Test that all posterior probabilities are non-negative.

    Probabilities must be >= 0 by definition. Negative values indicate
    numerical errors in the computation.
    """
    results = decoder_with_results["results"]
    acausal_posterior = results.acausal_posterior.values

    assert np.all(
        acausal_posterior >= 0
    ), f"Found negative probabilities: min={acausal_posterior.min():.10f}"


@pytest.mark.slow
def test_posterior_not_all_zero(decoder_with_results: dict[str, Any]) -> None:
    """Test that posterior is not all zeros.

    An all-zero posterior indicates complete failure of the decoder.
    """
    results = decoder_with_results["results"]
    acausal_posterior = results.acausal_posterior.values

    posterior_sum = np.sum(acausal_posterior)
    assert posterior_sum > 0, "Posterior is all zeros"


@pytest.mark.slow
def test_state_probabilities_normalized(decoder_with_results: dict[str, Any]) -> None:
    """Test that state probabilities sum to 1 at each time point.

    When marginalizing over positions, the discrete state probabilities
    should sum to 1.
    """
    results = decoder_with_results["results"]

    if "acausal_state_probabilities" in results:
        state_probs = results.acausal_state_probabilities.values

        # For decoders with only one state, values will be 1D array of all 1s
        if state_probs.ndim == 1:
            # Single state case - all probabilities should be 1.0
            np.testing.assert_allclose(
                state_probs,
                1.0,
                rtol=NORMALIZATION_RTOL,
                atol=NORMALIZATION_ATOL,
                err_msg=f"Single state probabilities not 1.0: min={state_probs.min():.10f}, "
                f"max={state_probs.max():.10f}",
            )
        else:
            # Multiple states - sum over states at each time point
            state_sum = np.sum(state_probs, axis=1)

            np.testing.assert_allclose(
                state_sum,
                1.0,
                rtol=NORMALIZATION_RTOL,
                atol=NORMALIZATION_ATOL,
                err_msg=f"State probabilities not normalized: min={state_sum.min():.10f}, "
                f"max={state_sum.max():.10f}",
            )
    else:
        pytest.skip("State probabilities not available in results")
