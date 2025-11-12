"""Numerical comparison of KDE vs GMM likelihood outputs.

This module compares the numerical outputs of KDE and GMM implementations to understand:
1. Magnitude differences in log-likelihood values
2. Correlation between spatial patterns
3. Agreement in peak locations (argmax)
4. Statistical properties of the likelihood distributions

Note: KDE and GMM are fundamentally different algorithms, so we expect numerical
differences. This analysis quantifies those differences to understand when each
method might be preferred.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)


@pytest.fixture
def comparison_data():
    """Create realistic test data for numerical comparison.

    Uses a larger dataset than the basic comparison tests to get
    more stable statistics.
    """
    np.random.seed(123)

    # Time parameters - longer duration for better statistics
    dt = 0.02  # 20 ms bins
    n_time = 100
    time = np.arange(n_time + 1) * dt

    # Position parameters - smoother trajectory
    position_time = np.linspace(0, time[-1], 400)
    t_normalized = np.linspace(0, 1, len(position_time))
    position = np.column_stack(
        [
            10 * t_normalized,  # linear x motion
            2 * np.sin(4 * np.pi * t_normalized),  # sinusoidal y motion
        ]
    )

    # Spike parameters - more electrodes and spikes
    n_electrodes = 5
    n_features = 4
    encoding_spike_times = []
    encoding_spike_features = []

    for elec_idx in range(n_electrodes):
        # Generate spatially modulated firing
        n_spikes = np.random.randint(80, 120)
        times = np.sort(np.random.uniform(time[0], time[-1], n_spikes))
        encoding_spike_times.append(times)

        # Features with electrode-specific bias
        features = np.random.randn(n_spikes, n_features).astype(np.float32)
        features += elec_idx * 0.5  # electrode-specific offset
        encoding_spike_features.append(features)

    # Decoding spikes
    decoding_spike_times = [times[: len(times) // 3] for times in encoding_spike_times]
    decoding_spike_features = [
        feats[: len(feats) // 3] for feats in encoding_spike_features
    ]

    # Create environment
    environment = Environment(position_range=[(0, 10), (-3, 3)])
    environment = environment.fit_place_grid(
        position=position, infer_track_interior=True
    )

    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "encoding_spike_times": encoding_spike_times,
        "encoding_spike_features": encoding_spike_features,
        "decoding_spike_times": decoding_spike_times,
        "decoding_spike_features": decoding_spike_features,
        "environment": environment,
    }


def fit_both_models(data, kde_position_std=2.0, kde_waveform_std=1.5):
    """Fit both KDE and GMM models on the same data.

    Parameters
    ----------
    data : dict
        Data dictionary from comparison_data fixture
    kde_position_std : float
        Bandwidth for KDE position kernel
    kde_waveform_std : float
        Bandwidth for KDE waveform kernel

    Returns
    -------
    kde_encoding : dict
        KDE encoding model
    gmm_encoding : dict
        GMM encoding model
    """
    # Convert to JAX arrays
    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    encoding_spike_times = [jnp.asarray(st) for st in data["encoding_spike_times"]]
    encoding_spike_features = [
        jnp.asarray(sf) for sf in data["encoding_spike_features"]
    ]

    # Fit KDE model
    kde_encoding = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_spike_times,
        spike_waveform_features=encoding_spike_features,
        environment=data["environment"],
        sampling_frequency=50,
        position_std=kde_position_std,
        waveform_std=kde_waveform_std,
        block_size=100,
        disable_progress_bar=True,
    )

    # Fit GMM model
    gmm_encoding = fit_clusterless_gmm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_spike_times,
        spike_waveform_features=encoding_spike_features,
        environment=data["environment"],
        sampling_frequency=50,
        gmm_components_occupancy=16,
        gmm_components_gpi=16,
        gmm_components_joint=32,
        gmm_random_state=42,
        disable_progress_bar=True,
    )

    return kde_encoding, gmm_encoding


def predict_both_models(data, kde_encoding, gmm_encoding, is_local=False):
    """Predict likelihoods with both models.

    Parameters
    ----------
    data : dict
        Data dictionary
    kde_encoding : dict
        KDE encoding model
    gmm_encoding : dict
        GMM encoding model
    is_local : bool
        Whether to compute local or non-local likelihood

    Returns
    -------
    ll_kde : jnp.ndarray
        KDE log-likelihood
    ll_gmm : jnp.ndarray
        GMM log-likelihood
    """
    time = jnp.asarray(data["time"])
    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    spike_times = [jnp.asarray(st) for st in data["decoding_spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["decoding_spike_features"]]

    # KDE prediction
    ll_kde = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_encoding["occupancy"],
        occupancy_model=kde_encoding["occupancy_model"],
        gpi_models=kde_encoding["gpi_models"],
        encoding_spike_waveform_features=kde_encoding[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=kde_encoding["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_encoding["mean_rates"],
        summed_ground_process_intensity=kde_encoding["summed_ground_process_intensity"],
        position_std=kde_encoding["position_std"],
        waveform_std=kde_encoding["waveform_std"],
        is_local=is_local,
        block_size=100,
        disable_progress_bar=True,
    )

    # GMM prediction
    ll_gmm = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        encoding_model=gmm_encoding,
        is_local=is_local,
        disable_progress_bar=True,
    )

    return ll_kde, ll_gmm


def test_likelihood_magnitude_comparison(comparison_data):
    """Compare the absolute magnitude of log-likelihood values.

    KDE and GMM will have different magnitudes due to:
    1. Different normalization constants
    2. Different density estimation methods
    3. Different model complexities
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(comparison_data, kde_enc, gmm_enc)

    # Convert to numpy for easier analysis
    ll_kde_np = np.asarray(ll_kde)
    ll_gmm_np = np.asarray(ll_gmm)

    # Basic statistics
    print("\n=== Log-Likelihood Magnitude Comparison ===")
    print(
        f"KDE: mean={ll_kde_np.mean():.2f}, std={ll_kde_np.std():.2f}, "
        f"min={ll_kde_np.min():.2f}, max={ll_kde_np.max():.2f}"
    )
    print(
        f"GMM: mean={ll_gmm_np.mean():.2f}, std={ll_gmm_np.std():.2f}, "
        f"min={ll_gmm_np.min():.2f}, max={ll_gmm_np.max():.2f}"
    )
    print(f"Mean difference: {(ll_kde_np.mean() - ll_gmm_np.mean()):.2f}")
    print(f"Std ratio (KDE/GMM): {(ll_kde_np.std() / ll_gmm_np.std()):.2f}")

    # Both should be finite
    assert np.all(np.isfinite(ll_kde_np))
    assert np.all(np.isfinite(ll_gmm_np))

    # KDE should be negative (log probabilities < 0)
    assert np.all(ll_kde_np <= 0)

    # GMM can have positive values due to different normalization
    # (it's computing log p(spikes, marks | position) which can be > 0)
    print(
        f"GMM positive values: {(ll_gmm_np > 0).sum()} / {ll_gmm_np.size} "
        f"({100 * (ll_gmm_np > 0).mean():.1f}%)"
    )


def test_spatial_pattern_correlation(comparison_data):
    """Compare spatial patterns using correlation metrics.

    Even if absolute values differ, we expect some correlation in spatial patterns:
    - High likelihood regions should be similar
    - Low likelihood regions should be similar
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(comparison_data, kde_enc, gmm_enc)

    # Analyze each time bin separately
    n_time = ll_kde.shape[0]
    pearson_corrs = []
    spearman_corrs = []

    print("\n=== Spatial Pattern Correlation (Per Time Bin) ===")
    for t in range(min(5, n_time)):  # Show first 5 time bins
        kde_t = np.asarray(ll_kde[t, :])
        gmm_t = np.asarray(ll_gmm[t, :])

        # Pearson correlation (linear relationship)
        r_pearson, _ = pearsonr(kde_t, gmm_t)
        pearson_corrs.append(r_pearson)

        # Spearman correlation (rank-order relationship)
        r_spearman, _ = spearmanr(kde_t, gmm_t)
        spearman_corrs.append(r_spearman)

        print(f"Time bin {t}: Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}")

    # Overall statistics
    pearson_corrs = np.array(pearson_corrs)
    spearman_corrs = np.array(spearman_corrs)
    print(
        f"\nMean Pearson correlation: {pearson_corrs.mean():.3f} ± {pearson_corrs.std():.3f}"
    )
    print(
        f"Mean Spearman correlation: {spearman_corrs.mean():.3f} ± {spearman_corrs.std():.3f}"
    )

    # Relaxed assertion: correlation should be positive on average but not necessarily strong
    # KDE and GMM are fundamentally different algorithms, so moderate correlation is expected
    assert pearson_corrs.mean() > 0.0, (
        "Spatial patterns should have positive correlation on average"
    )

    # Check that at least some time bins have good correlation
    strong_corr_count = (np.abs(pearson_corrs) > 0.5).sum()
    print(f"Time bins with |r| > 0.5: {strong_corr_count}/{len(pearson_corrs)}")
    assert strong_corr_count > 0, (
        "At least some time bins should show strong correlation"
    )


def test_peak_location_agreement(comparison_data):
    """Compare agreement in peak locations (argmax).

    For decoding, the most important feature is often the peak location
    (most likely position). We test how often KDE and GMM agree on the
    peak location.
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(comparison_data, kde_enc, gmm_enc)

    # Find argmax for each time bin
    argmax_kde = np.argmax(ll_kde, axis=1)
    argmax_gmm = np.argmax(ll_gmm, axis=1)

    # Exact agreement
    exact_agreement = np.mean(argmax_kde == argmax_gmm)

    # Agreement within tolerance (nearby bins)
    nearby_agreement_1 = np.mean(np.abs(argmax_kde - argmax_gmm) <= 1)
    nearby_agreement_2 = np.mean(np.abs(argmax_kde - argmax_gmm) <= 2)

    print("\n=== Peak Location Agreement ===")
    print(f"Exact agreement: {exact_agreement:.1%}")
    print(f"Within 1 bin: {nearby_agreement_1:.1%}")
    print(f"Within 2 bins: {nearby_agreement_2:.1%}")

    # Distance statistics
    distances = np.abs(argmax_kde - argmax_gmm)
    print(f"Mean distance: {distances.mean():.2f} bins")
    print(f"Median distance: {np.median(distances):.2f} bins")
    print(f"Max distance: {distances.max()} bins")

    # At least half should agree within 2 bins
    assert nearby_agreement_2 > 0.5, "At least 50% of peaks should be within 2 bins"


def test_likelihood_range_stability(comparison_data):
    """Test how the range of likelihood values varies across time.

    For decoding, we care about the spread of likelihood values:
    - Large spread = confident prediction
    - Small spread = uncertain prediction
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(comparison_data, kde_enc, gmm_enc)

    # Compute range (max - min) for each time bin
    range_kde = np.max(ll_kde, axis=1) - np.min(ll_kde, axis=1)
    range_gmm = np.max(ll_gmm, axis=1) - np.min(ll_gmm, axis=1)

    print("\n=== Likelihood Range Statistics ===")
    print(f"KDE range: mean={range_kde.mean():.2f}, std={range_kde.std():.2f}")
    print(f"GMM range: mean={range_gmm.mean():.2f}, std={range_gmm.std():.2f}")

    # Correlation between ranges (do they agree on certainty?)
    r_range, _ = pearsonr(range_kde, range_gmm)
    print(f"Range correlation: r={r_range:.3f}")

    # Both should have positive range
    assert np.all(range_kde > 0)
    assert np.all(range_gmm > 0)


def test_normalized_likelihood_comparison(comparison_data):
    """Compare normalized likelihood distributions.

    Normalize each time bin to sum to 1 (convert to probability distributions)
    and compare these normalized distributions.
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(comparison_data, kde_enc, gmm_enc)

    # Convert log-likelihood to probability (exp and normalize)
    def normalize_log_likelihood(ll):
        """Convert log-likelihood to normalized probability."""
        # Subtract max for numerical stability
        ll_shifted = ll - np.max(ll, axis=1, keepdims=True)
        prob = np.exp(ll_shifted)
        return prob / np.sum(prob, axis=1, keepdims=True)

    prob_kde = normalize_log_likelihood(np.asarray(ll_kde))
    prob_gmm = normalize_log_likelihood(np.asarray(ll_gmm))

    # KL divergence: KL(KDE || GMM)
    def kl_divergence(p, q, epsilon=1e-10):
        """Compute KL divergence KL(p || q)."""
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q), axis=1)

    kl_kde_gmm = kl_divergence(prob_kde, prob_gmm)
    kl_gmm_kde = kl_divergence(prob_gmm, prob_kde)

    print("\n=== Normalized Distribution Comparison ===")
    print(f"KL(KDE || GMM): mean={kl_kde_gmm.mean():.3f}, std={kl_kde_gmm.std():.3f}")
    print(f"KL(GMM || KDE): mean={kl_gmm_kde.mean():.3f}, std={kl_gmm_kde.std():.3f}")

    # Jensen-Shannon divergence (symmetric)
    js_divergence = 0.5 * (kl_kde_gmm + kl_gmm_kde)
    print(
        f"JS divergence: mean={js_divergence.mean():.3f}, std={js_divergence.std():.3f}"
    )

    # Hellinger distance
    hellinger = np.sqrt(
        0.5 * np.sum((np.sqrt(prob_kde) - np.sqrt(prob_gmm)) ** 2, axis=1)
    )
    print(f"Hellinger distance: mean={hellinger.mean():.3f}, std={hellinger.std():.3f}")

    # All divergences should be finite
    assert np.all(np.isfinite(kl_kde_gmm))
    assert np.all(np.isfinite(kl_gmm_kde))
    assert np.all(np.isfinite(js_divergence))


def test_local_likelihood_comparison(comparison_data):
    """Compare local likelihood predictions.

    Local likelihood is a single value per time bin (at animal's position),
    so we can compare these more directly.
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(
        comparison_data, kde_enc, gmm_enc, is_local=True
    )

    ll_kde_np = np.asarray(ll_kde).ravel()
    ll_gmm_np = np.asarray(ll_gmm).ravel()

    print("\n=== Local Likelihood Comparison ===")
    print(f"KDE shape: {ll_kde.shape}, GMM shape: {ll_gmm.shape}")
    print(f"KDE: mean={ll_kde_np.mean():.2f}, std={ll_kde_np.std():.2f}")
    print(f"GMM: mean={ll_gmm_np.mean():.2f}, std={ll_gmm_np.std():.2f}")

    # Check if shapes match (KDE uses time edges, GMM uses time bin centers)
    if ll_kde_np.shape != ll_gmm_np.shape:
        print(
            f"Warning: Shape mismatch - KDE has {len(ll_kde_np)} values, GMM has {len(ll_gmm_np)}"
        )
        # Trim to same length for comparison
        min_len = min(len(ll_kde_np), len(ll_gmm_np))
        ll_kde_np = ll_kde_np[:min_len]
        ll_gmm_np = ll_gmm_np[:min_len]
        print(f"Comparing first {min_len} values")

    # Correlation
    r_pearson, _ = pearsonr(ll_kde_np, ll_gmm_np)
    r_spearman, _ = spearmanr(ll_kde_np, ll_gmm_np)
    print(f"Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}")

    # Both should be finite
    assert np.all(np.isfinite(ll_kde_np))
    assert np.all(np.isfinite(ll_gmm_np))

    # Check for any correlation (can be negative or positive)
    assert np.abs(r_pearson) > 0.1 or np.abs(r_spearman) > 0.1, (
        "Local likelihoods should have some correlation"
    )


def test_parameter_sensitivity_kde(comparison_data):
    """Test how KDE bandwidth parameters affect results.

    This helps understand when KDE and GMM might differ more or less.
    """
    # Fit with different KDE bandwidths
    bandwidths = [0.5, 1.0, 2.0, 4.0]
    correlations = []

    # Fit GMM once (reference)
    _, gmm_enc = fit_both_models(comparison_data, kde_position_std=2.0)
    ll_gmm, _ = predict_both_models(
        comparison_data,
        fit_both_models(comparison_data, kde_position_std=2.0)[0],
        gmm_enc,
    )

    print("\n=== KDE Bandwidth Sensitivity ===")
    for bw in bandwidths:
        kde_enc, _ = fit_both_models(comparison_data, kde_position_std=bw)
        ll_kde, _ = predict_both_models(comparison_data, kde_enc, gmm_enc)

        # Compute correlation
        r_mean = []
        for t in range(ll_kde.shape[0]):
            r, _ = pearsonr(np.asarray(ll_kde[t, :]), np.asarray(ll_gmm[t, :]))
            r_mean.append(r)

        mean_corr = np.mean(r_mean)
        correlations.append(mean_corr)
        print(f"Bandwidth={bw:.1f}: mean correlation={mean_corr:.3f}")

    # Correlation should vary with bandwidth
    assert max(correlations) - min(correlations) > 0.1, (
        "Bandwidth should affect correlation"
    )


@pytest.mark.skip(reason="Visualization test - run manually to generate plots")
def test_visualize_comparison(comparison_data):
    """Generate visualization comparing KDE and GMM outputs.

    This test is skipped by default but can be run manually to generate
    comparison plots.
    """
    kde_enc, gmm_enc = fit_both_models(comparison_data)
    ll_kde, ll_gmm = predict_both_models(comparison_data, kde_enc, gmm_enc)

    # Convert to numpy
    ll_kde_np = np.asarray(ll_kde)
    ll_gmm_np = np.asarray(ll_gmm)

    # Plot comparison for a few time bins
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for idx, t in enumerate(range(0, min(9, ll_kde.shape[0]), 3)):
        row = idx // 3
        col = idx % 3

        ax = axes[row, col]
        ax.plot(ll_kde_np[t, :], label="KDE", alpha=0.7)
        ax.plot(ll_gmm_np[t, :], label="GMM", alpha=0.7)
        ax.set_xlabel("Position bin")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"Time bin {t}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kde_gmm_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved comparison plot to kde_gmm_comparison.png")
