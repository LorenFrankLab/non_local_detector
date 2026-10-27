"""Oracle correctness tests for clusterless decoding.

These tests verify that decoders can accurately recover true position from
simulated data where we know the ground truth. They test fundamental correctness
of the encoding/decoding pipeline.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    mark_reference_scale,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

# Test configuration constants
POSITION_STD = 6.0
WAVEFORM_STD = 24.0
POSITION_TOLERANCE = 10.0
TOP1_ACCURACY_THRESHOLD = 0.80
TOP3_ACCURACY_THRESHOLD = 0.90

pytestmark = pytest.mark.integration


def compute_top_k_accuracy(
    true_positions: npt.NDArray[np.floating],
    log_likelihood: npt.NDArray[np.floating],
    place_bin_centers: npt.NDArray[np.floating],
    k: int = 1,
    position_tolerance: float = POSITION_TOLERANCE,
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


@pytest.fixture(scope="module")
def kde_decoding_setup() -> dict[str, Any]:
    """Generate simulated data and encoding model for KDE decoding tests.

    Module-scoped: the tests only read the results.

    Returns
    -------
    dict with keys:
        - sim: ClusterlessSimOutput with full simulation data
        - encoding_model: Fitted KDE encoding model
        - test_edges: Time bin edges for decoding
        - log_likelihood: Log-likelihood array (n_time_bins, n_position_bins)
        - true_positions: True positions at test bin centers
        - place_bin_centers: Position bin centers from environment
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
    encoding_features = [
        swf[sim.spike_times[i] < sim.position_time[n_encode]]
        for i, swf in enumerate(sim.spike_waveform_features)
    ]
    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=sim.position_time[:n_encode],
        position=sim.position[:n_encode],
        spike_times=[st[st < sim.position_time[n_encode]] for st in sim.spike_times],
        spike_waveform_features=encoding_features,
        environment=sim.environment,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        block_size=100,
        disable_progress_bar=True,
    )
    mark_reference = fit_mark_references(
        encoding_features, mark_reference_scale(waveform_std=WAVEFORM_STD)
    )

    # Decode on test data
    test_time = sim.position_time[n_encode:]
    test_edges = np.linspace(test_time[0], test_time[-1], 50)  # 49 time bins

    log_likelihood = predict_clusterless_kde_log_likelihood(
        time_edges=test_edges,
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
        environment=sim.environment,
        mean_rates=encoding_model["mean_rates"],
        summed_ground_process_intensity=encoding_model[
            "summed_ground_process_intensity"
        ],
        position_std=encoding_model["position_std"],
        waveform_std=encoding_model["waveform_std"],
        mark_reference=mark_reference,
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

    return {
        "sim": sim,
        "encoding_model": encoding_model,
        "test_edges": test_edges,
        "log_likelihood": log_likelihood,
        "true_positions": true_positions,
        "place_bin_centers": sim.environment.place_bin_centers_,
    }


def test_nonlocal_kde_top1_accuracy_high(kde_decoding_setup: dict[str, Any]) -> None:
    """Test that KDE can decode position with >80% top-1 accuracy on simulated data.

    This is an oracle test: we simulate data where we know the true position,
    fit an encoding model, and verify we can accurately decode that position.
    """
    # Compute top-1 accuracy using shared fixture
    accuracy = compute_top_k_accuracy(
        kde_decoding_setup["true_positions"],
        kde_decoding_setup["log_likelihood"],
        kde_decoding_setup["place_bin_centers"],
        k=1,
    )

    # Oracle test: we should be able to decode with high accuracy
    assert accuracy >= TOP1_ACCURACY_THRESHOLD, (
        f"Top-1 accuracy {accuracy:.2%} is below threshold {TOP1_ACCURACY_THRESHOLD:.0%}. "
        f"Decoding should accurately recover true position from simulated data."
    )


def test_nonlocal_kde_top3_accuracy_very_high(
    kde_decoding_setup: dict[str, Any],
) -> None:
    """Test that true position is in top-3 decoded positions >90% of the time."""
    # Compute top-3 accuracy using shared fixture
    accuracy = compute_top_k_accuracy(
        kde_decoding_setup["true_positions"],
        kde_decoding_setup["log_likelihood"],
        kde_decoding_setup["place_bin_centers"],
        k=3,
    )

    assert accuracy >= TOP3_ACCURACY_THRESHOLD, (
        f"Top-3 accuracy {accuracy:.2%} is below threshold {TOP3_ACCURACY_THRESHOLD:.0%}. "
        f"True position should be in top-3 decoded bins most of the time."
    )
