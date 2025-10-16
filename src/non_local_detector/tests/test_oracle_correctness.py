"""Oracle correctness tests for clusterless decoding.

These tests verify that decoders can accurately recover true position from
simulated data where we know the ground truth. They test fundamental correctness
of the encoding/decoding pipeline.
"""

import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


def get_decoded_position_bins(
    log_likelihood: np.ndarray, place_bin_centers: np.ndarray
) -> np.ndarray:
    """Get the most likely position bin for each time point.

    Parameters
    ----------
    log_likelihood : np.ndarray, shape (n_time, n_position_bins)
        Log-likelihood values for each time and position bin.
    place_bin_centers : np.ndarray, shape (n_position_bins, n_pos_dims)
        Centers of position bins.

    Returns
    -------
    decoded_positions : np.ndarray, shape (n_time, n_pos_dims)
        Most likely position at each time point.
    """
    # Get the bin index with highest likelihood at each time
    max_likelihood_bins = np.argmax(log_likelihood, axis=1)
    # Return the position at those bins
    return place_bin_centers[max_likelihood_bins]


def compute_top_k_accuracy(
    true_positions: np.ndarray,
    log_likelihood: np.ndarray,
    place_bin_centers: np.ndarray,
    k: int = 1,
    position_tolerance: float = 10.0,
) -> float:
    """Compute top-k accuracy: fraction of times true position is in top k bins.

    Parameters
    ----------
    true_positions : np.ndarray, shape (n_time, n_pos_dims)
        True positions at each time point.
    log_likelihood : np.ndarray, shape (n_time, n_position_bins)
        Log-likelihood values for each time and position bin.
    place_bin_centers : np.ndarray, shape (n_position_bins, n_pos_dims)
        Centers of position bins.
    k : int, optional
        Number of top candidates to consider. Default is 1.
    position_tolerance : float, optional
        Distance threshold for considering a position "correct". Default is 10.0.

    Returns
    -------
    accuracy : float
        Fraction of time points where true position is within top k bins.
    """
    n_time = min(log_likelihood.shape[0], len(true_positions))
    correct_count = 0

    # Get top k bin indices for each time point
    top_k_bins = np.argsort(-log_likelihood, axis=1)[:n_time, :k]  # Shape: (n_time, k)

    for t in range(n_time):
        true_pos = true_positions[t]
        # Get positions of top k bins
        top_k_positions = place_bin_centers[top_k_bins[t]]  # Shape: (k, n_pos_dims)

        # Check if any of the top k positions are within tolerance
        distances = np.linalg.norm(top_k_positions - true_pos, axis=1)
        if np.any(distances <= position_tolerance):
            correct_count += 1

    return correct_count / n_time


@pytest.mark.slow
def test_nonlocal_kde_top1_accuracy_high() -> None:
    """Test that KDE can decode position with >80% top-1 accuracy on simulated data.

    This is an oracle test: we simulate data where we know the true position,
    fit an encoding model, and verify we can accurately decode that position.
    """
    # Generate simulated data with good coverage
    n_tetrodes = 4
    place_field_means = np.arange(0, 160, 10)  # 16 neurons, good spatial coverage
    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=3,  # More runs for better coverage
        seed=42,
    )

    # Use first 70% for encoding, last 30% for decoding
    n_encode = int(0.7 * len(sim.position_time))

    # Fit encoding model on training data
    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=sim.position_time[:n_encode],
        position=sim.position[:n_encode],
        spike_times=[st[st < sim.position_time[n_encode]] for st in sim.spike_times],
        spike_waveform_features=[
            swf[sim.spike_times[i] < sim.position_time[n_encode]]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        environment=sim.environment,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Decode on test data
    test_time = sim.position_time[n_encode:]
    test_edges = np.linspace(test_time[0], test_time[-1], 50)  # 50 time bins

    log_likelihood = predict_clusterless_kde_log_likelihood(
        time=test_edges,
        position_time=sim.position_time,
        position=sim.position,
        spike_times=[
            st[(st >= test_time[0]) & (st <= test_time[-1])] for st in sim.spike_times
        ],
        spike_waveform_features=[
            swf[
                (sim.spike_times[i] >= test_time[0])
                & (sim.spike_times[i] <= test_time[-1])
            ]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        occupancy=encoding_model["occupancy"],
        occupancy_model=encoding_model["occupancy_model"],
        gpi_models=encoding_model["gpi_models"],
        encoding_spike_waveform_features=encoding_model[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=encoding_model["encoding_positions"],
        encoding_spike_weights=encoding_model["encoding_spike_weights"],
        environment=sim.environment,
        mean_rates=encoding_model["mean_rates"],
        summed_ground_process_intensity=encoding_model[
            "summed_ground_process_intensity"
        ],
        position_std=encoding_model["position_std"],
        waveform_std=encoding_model["waveform_std"],
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    # Get true positions at test time bin centers
    test_bin_centers = (test_edges[:-1] + test_edges[1:]) / 2
    true_positions = np.array(
        [
            sim.position[np.argmin(np.abs(sim.position_time - t))]
            for t in test_bin_centers
        ]
    )

    # Compute top-1 accuracy
    place_bin_centers = sim.environment.place_bin_centers_
    accuracy = compute_top_k_accuracy(
        true_positions, log_likelihood, place_bin_centers, k=1, position_tolerance=10.0
    )

    # Oracle test: we should be able to decode with high accuracy
    assert accuracy >= 0.80, (
        f"Top-1 accuracy {accuracy:.2%} is below threshold 80%. "
        f"Decoding should accurately recover true position from simulated data."
    )


@pytest.mark.slow
def test_nonlocal_kde_top3_accuracy_very_high() -> None:
    """Test that true position is in top-3 decoded positions >90% of the time."""
    # Use same setup as top-1 test
    n_tetrodes = 4
    place_field_means = np.arange(0, 160, 10)
    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=3,
        seed=42,
    )

    n_encode = int(0.7 * len(sim.position_time))

    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=sim.position_time[:n_encode],
        position=sim.position[:n_encode],
        spike_times=[st[st < sim.position_time[n_encode]] for st in sim.spike_times],
        spike_waveform_features=[
            swf[sim.spike_times[i] < sim.position_time[n_encode]]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        environment=sim.environment,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    test_time = sim.position_time[n_encode:]
    test_edges = np.linspace(test_time[0], test_time[-1], 50)

    log_likelihood = predict_clusterless_kde_log_likelihood(
        time=test_edges,
        position_time=sim.position_time,
        position=sim.position,
        spike_times=[
            st[(st >= test_time[0]) & (st <= test_time[-1])] for st in sim.spike_times
        ],
        spike_waveform_features=[
            swf[
                (sim.spike_times[i] >= test_time[0])
                & (sim.spike_times[i] <= test_time[-1])
            ]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        occupancy=encoding_model["occupancy"],
        occupancy_model=encoding_model["occupancy_model"],
        gpi_models=encoding_model["gpi_models"],
        encoding_spike_waveform_features=encoding_model[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=encoding_model["encoding_positions"],
        encoding_spike_weights=encoding_model["encoding_spike_weights"],
        environment=sim.environment,
        mean_rates=encoding_model["mean_rates"],
        summed_ground_process_intensity=encoding_model[
            "summed_ground_process_intensity"
        ],
        position_std=encoding_model["position_std"],
        waveform_std=encoding_model["waveform_std"],
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    test_bin_centers = (test_edges[:-1] + test_edges[1:]) / 2
    true_positions = np.array(
        [
            sim.position[np.argmin(np.abs(sim.position_time - t))]
            for t in test_bin_centers
        ]
    )

    place_bin_centers = sim.environment.place_bin_centers_
    accuracy = compute_top_k_accuracy(
        true_positions, log_likelihood, place_bin_centers, k=3, position_tolerance=10.0
    )

    assert accuracy >= 0.90, (
        f"Top-3 accuracy {accuracy:.2%} is below threshold 90%. "
        f"True position should be in top-3 decoded bins most of the time."
    )


@pytest.mark.slow
def test_delta_t_scaling_normalized() -> None:
    """Test that bin width doesn't drastically change decoding results.

    When we use different bin widths, the overall decoding accuracy should
    remain comparable. This tests that the Δt scaling in the likelihood is
    handled correctly.
    """
    # Simulate data
    n_tetrodes = 3
    place_field_means = np.arange(0, 120, 10)  # 12 neurons
    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=2,
        seed=100,
    )

    n_encode = int(0.7 * len(sim.position_time))

    # Fit encoding model once
    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=sim.position_time[:n_encode],
        position=sim.position[:n_encode],
        spike_times=[st[st < sim.position_time[n_encode]] for st in sim.spike_times],
        spike_waveform_features=[
            swf[sim.spike_times[i] < sim.position_time[n_encode]]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        environment=sim.environment,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    test_time = sim.position_time[n_encode:]

    # Decode with standard bin width (50 bins)
    test_edges_standard = np.linspace(test_time[0], test_time[-1], 51)  # 50 bins
    ll_standard = predict_clusterless_kde_log_likelihood(
        time=test_edges_standard,
        position_time=sim.position_time,
        position=sim.position,
        spike_times=[
            st[(st >= test_time[0]) & (st <= test_time[-1])] for st in sim.spike_times
        ],
        spike_waveform_features=[
            swf[
                (sim.spike_times[i] >= test_time[0])
                & (sim.spike_times[i] <= test_time[-1])
            ]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        **{
            k: encoding_model[k]
            for k in encoding_model
            if k not in ["environment", "block_size", "disable_progress_bar"]
        },
        environment=sim.environment,
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    # Decode with wider bins (25 bins)
    test_edges_wide = np.linspace(test_time[0], test_time[-1], 26)  # 25 bins
    ll_wide = predict_clusterless_kde_log_likelihood(
        time=test_edges_wide,
        position_time=sim.position_time,
        position=sim.position,
        spike_times=[
            st[(st >= test_time[0]) & (st <= test_time[-1])] for st in sim.spike_times
        ],
        spike_waveform_features=[
            swf[
                (sim.spike_times[i] >= test_time[0])
                & (sim.spike_times[i] <= test_time[-1])
            ]
            for i, swf in enumerate(sim.spike_waveform_features)
        ],
        **{
            k: encoding_model[k]
            for k in encoding_model
            if k not in ["environment", "block_size", "disable_progress_bar"]
        },
        environment=sim.environment,
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    # Get true positions for each bin width
    test_bin_centers_standard = (test_edges_standard[:-1] + test_edges_standard[1:]) / 2
    true_positions_standard = np.array(
        [
            sim.position[np.argmin(np.abs(sim.position_time - t))]
            for t in test_bin_centers_standard
        ]
    )

    test_bin_centers_wide = (test_edges_wide[:-1] + test_edges_wide[1:]) / 2
    true_positions_wide = np.array(
        [
            sim.position[np.argmin(np.abs(sim.position_time - t))]
            for t in test_bin_centers_wide
        ]
    )

    # Compute accuracy for both bin widths
    place_bin_centers = sim.environment.place_bin_centers_
    accuracy_standard = compute_top_k_accuracy(
        true_positions_standard,
        ll_standard,
        place_bin_centers,
        k=1,
        position_tolerance=10.0,
    )
    accuracy_wide = compute_top_k_accuracy(
        true_positions_wide, ll_wide, place_bin_centers, k=1, position_tolerance=10.0
    )

    # Both should have reasonable accuracy
    assert accuracy_standard >= 0.60, (
        f"Standard bin width accuracy {accuracy_standard:.2%} is too low"
    )
    assert accuracy_wide >= 0.50, (
        f"Wide bin width accuracy {accuracy_wide:.2%} is too low"
    )

    # Accuracies should be comparable (within 30 percentage points)
    accuracy_diff = abs(accuracy_standard - accuracy_wide)
    assert accuracy_diff <= 0.30, (
        f"Accuracy difference {accuracy_diff:.2%} is too large. "
        f"Bin width changes should not drastically affect decoding quality. "
        f"Standard: {accuracy_standard:.2%}, Wide: {accuracy_wide:.2%}"
    )
