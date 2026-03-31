"""Tests for clusterless likelihood model agreement.

These tests verify that different likelihood models (KDE, GMM) produce
qualitatively similar results on the same simulated data. This helps catch
regressions where one model drastically diverges from expected behavior.
"""

import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

# Constants
POSITION_STD = 6.0
WAVEFORM_STD = 24.0
BLOCK_SIZE = 100
LIKELIHOOD_CORRELATION_THRESHOLD = 0.50
SIMULATION_SEED = 100


@pytest.fixture
def clusterless_likelihood_comparison() -> dict:
    """Generate simulated data and fit both KDE and GMM models.

    Returns dict with keys:
        - sim: ClusterlessSimOutput with full simulation data
        - kde_encoding: Fitted KDE encoding model
        - gmm_encoding: Fitted GMM encoding model
        - test_edges: Time bin edges for decoding
        - kde_log_likelihood: KDE log-likelihood (n_time_bins, n_position_bins)
        - gmm_log_likelihood: GMM log-likelihood (n_time_bins, n_position_bins)
        - place_bin_centers: Position bin centers from environment
    """
    # Generate simulated data (3 tetrodes, 12 neurons, 2 runs)
    sim = make_simulated_run_data(
        n_tetrodes=3,
        place_field_means=np.arange(0, 120, 10),
        sampling_frequency=500,
        n_runs=2,
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
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=False)
    ]

    # Fit KDE encoding model
    kde_encoding = fit_clusterless_kde_encoding_model(
        position_time=encode_position_time,
        position=encode_position,
        spike_times=encode_spike_times,
        spike_waveform_features=encode_spike_waveform_features,
        environment=sim.environment,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        block_size=BLOCK_SIZE,
        disable_progress_bar=True,
    )

    # Fit GMM encoding model
    gmm_encoding = fit_clusterless_gmm_encoding_model(
        position_time=encode_position_time,
        position=encode_position,
        spike_times=encode_spike_times,
        spike_waveform_features=encode_spike_waveform_features,
        environment=sim.environment,
        disable_progress_bar=True,
    )

    # Test data
    # Use position time as the boundary for consistency
    split_time = encode_position_time[-1]
    test_position_time = sim.position_time[n_encode:]
    test_spike_times = [st[st >= split_time] for st in sim.spike_times]
    test_spike_waveform_features = [
        swf[st >= split_time]
        for st, swf in zip(sim.spike_times, sim.spike_waveform_features, strict=False)
    ]

    # Create test edges (decoding bins) - use >= for consistency
    test_edges = sim.edges[sim.edges >= split_time]

    # Predict KDE log-likelihood
    kde_log_likelihood = predict_clusterless_kde_log_likelihood(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
        occupancy=kde_encoding["occupancy"],
        occupancy_model=kde_encoding["occupancy_model"],
        gpi_models=kde_encoding["gpi_models"],
        encoding_spike_waveform_features=kde_encoding[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=kde_encoding["encoding_positions"],
        environment=sim.environment,
        mean_rates=kde_encoding["mean_rates"],
        summed_ground_process_intensity=kde_encoding["summed_ground_process_intensity"],
        position_std=kde_encoding["position_std"],
        waveform_std=kde_encoding["waveform_std"],
        is_local=False,
        block_size=BLOCK_SIZE,
        disable_progress_bar=True,
    )

    # Predict GMM log-likelihood
    gmm_log_likelihood = predict_clusterless_gmm_log_likelihood(
        time=test_edges,
        position_time=test_position_time,
        position=sim.position[n_encode:],
        spike_times=test_spike_times,
        spike_waveform_features=test_spike_waveform_features,
        **gmm_encoding,
        is_local=False,
    )

    place_bin_centers = sim.environment.place_bin_centers_

    return {
        "sim": sim,
        "kde_encoding": kde_encoding,
        "gmm_encoding": gmm_encoding,
        "test_edges": test_edges,
        "kde_log_likelihood": kde_log_likelihood,
        "gmm_log_likelihood": gmm_log_likelihood,
        "place_bin_centers": place_bin_centers,
    }


@pytest.mark.slow
def test_kde_gmm_likelihood_correlation(
    clusterless_likelihood_comparison: dict,
) -> None:
    """Test that KDE and GMM log-likelihoods are positively correlated.

    While the absolute values may differ, the relative rankings of position bins
    should be similar between models, indicated by positive correlation.
    """
    kde_ll = clusterless_likelihood_comparison["kde_log_likelihood"]
    gmm_ll = clusterless_likelihood_comparison["gmm_log_likelihood"]

    # Flatten to compare all (time, position) pairs
    kde_flat = kde_ll.flatten()
    gmm_flat = gmm_ll.flatten()

    # Compute Pearson correlation
    correlation = np.corrcoef(kde_flat, gmm_flat)[0, 1]

    assert correlation >= LIKELIHOOD_CORRELATION_THRESHOLD, (
        f"KDE/GMM likelihood correlation {correlation:.3f} below threshold {LIKELIHOOD_CORRELATION_THRESHOLD:.3f}"
    )


@pytest.mark.slow
def test_kde_gmm_rank_correlation(
    clusterless_likelihood_comparison: dict,
) -> None:
    """Test that KDE and GMM produce similar position rankings per time bin.

    For each time bin, the rank ordering of positions should be similar between
    models, measured by Spearman correlation (correlation of rankings).
    """
    kde_ll = clusterless_likelihood_comparison["kde_log_likelihood"]
    gmm_ll = clusterless_likelihood_comparison["gmm_log_likelihood"]

    # Compute rank correlations per time bin
    n_time_bins = kde_ll.shape[0]
    rank_correlations = []

    for t in range(n_time_bins):
        # Get ranks for this time bin (argsort of argsort gives ranks)
        kde_ranks = np.argsort(np.argsort(kde_ll[t, :]))
        gmm_ranks = np.argsort(np.argsort(gmm_ll[t, :]))

        # Compute Spearman correlation (Pearson on ranks)
        if np.std(kde_ranks) > 0 and np.std(gmm_ranks) > 0:
            corr = np.corrcoef(kde_ranks, gmm_ranks)[0, 1]
            rank_correlations.append(corr)

    # Average rank correlation across time bins
    mean_rank_correlation = np.mean(rank_correlations)

    assert mean_rank_correlation >= LIKELIHOOD_CORRELATION_THRESHOLD, (
        f"Mean rank correlation {mean_rank_correlation:.3f} below threshold {LIKELIHOOD_CORRELATION_THRESHOLD:.3f}"
    )
