"""Investigate behavior with extreme feature values.

This script explores how both clusterless_kde.py and clusterless_kde_log.py
handle extreme feature distances that can cause numerical underflow.
"""

import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_original,
)
from non_local_detector.likelihoods.clusterless_kde import kde_distance
from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity as estimate_log,
)
from non_local_detector.likelihoods.clusterless_kde_log import log_kde_distance


def analyze_extreme_features():
    """Analyze behavior with extreme feature values."""
    print("=" * 80)
    print("EXTREME FEATURE VALUE INVESTIGATION")
    print("=" * 80)
    print()

    # Test configuration from failing test
    n_enc_spikes = 30
    n_dec_spikes = 10
    n_pos_bins = 20
    n_features = 4

    # Extreme feature values (large distances)
    np.random.seed(42)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 50 + 100)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 50 + 200)
    waveform_stds = jnp.array([10.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 2.0

    # Position distance
    enc_positions = jnp.array(np.random.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])

    print("Configuration:")
    print(f"  Decoding spikes: {n_dec_spikes}")
    print(f"  Encoding spikes: {n_enc_spikes}")
    print(f"  Position bins: {n_pos_bins}")
    print(f"  Feature dimensions: {n_features}")
    print()

    # Analyze feature distances
    print("-" * 80)
    print("FEATURE DISTANCE ANALYSIS")
    print("-" * 80)
    print()

    # Compute pairwise distances in feature space
    feature_distances = []
    for i in range(n_dec_spikes):
        for j in range(n_enc_spikes):
            dist = np.sqrt(np.sum(((dec_features[i] - enc_features[j]) / waveform_stds) ** 2))
            feature_distances.append(dist)

    feature_distances = np.array(feature_distances)

    print(f"Feature space distances (standardized by std):")
    print(f"  Min: {feature_distances.min():.2f} std")
    print(f"  Max: {feature_distances.max():.2f} std")
    print(f"  Mean: {feature_distances.mean():.2f} std")
    print(f"  Median: {np.median(feature_distances):.2f} std")
    print()

    # What does this mean for Gaussian PDF values?
    print("Expected Gaussian PDF values:")
    print(f"  At min distance ({feature_distances.min():.2f} std): {np.exp(-0.5 * feature_distances.min()**2):.2e}")
    print(f"  At mean distance ({feature_distances.mean():.2f} std): {np.exp(-0.5 * feature_distances.mean()**2):.2e}")
    print(f"  At max distance ({feature_distances.max():.2f} std): {np.exp(-0.5 * feature_distances.max()**2):.2e}")
    print()

    # Check for underflow
    underflow_threshold = 1e-300  # Approximate float64 underflow
    n_underflow = np.sum(np.exp(-0.5 * feature_distances**2) < underflow_threshold)
    print(f"Pairs expected to underflow: {n_underflow}/{len(feature_distances)} ({100*n_underflow/len(feature_distances):.1f}%)")
    print()

    # Compute position distances
    position_distance = kde_distance(interior_bins, enc_positions, position_std)
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    print("-" * 80)
    print("ORIGINAL IMPLEMENTATION (clusterless_kde.py)")
    print("-" * 80)
    print()

    ll_original = estimate_original(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
    )

    print("Results:")
    print(f"  Shape: {ll_original.shape}")
    print(f"  Dtype: {ll_original.dtype}")
    print()

    # Analyze values
    finite_mask = np.isfinite(ll_original)
    n_finite = np.sum(finite_mask)
    n_inf = np.sum(np.isinf(ll_original))
    n_nan = np.sum(np.isnan(ll_original))

    print(f"Value distribution:")
    print(f"  Finite values: {n_finite}/{ll_original.size} ({100*n_finite/ll_original.size:.1f}%)")
    print(f"  -Inf values: {n_inf}/{ll_original.size} ({100*n_inf/ll_original.size:.1f}%)")
    print(f"  NaN values: {n_nan}/{ll_original.size} ({100*n_nan/ll_original.size:.1f}%)")
    print()

    if n_finite > 0:
        print(f"Finite value statistics:")
        print(f"  Min: {ll_original[finite_mask].min():.4f}")
        print(f"  Max: {ll_original[finite_mask].max():.4f}")
        print(f"  Mean: {ll_original[finite_mask].mean():.4f}")
        print()

    # Show per-spike breakdown
    print("Per-spike breakdown:")
    for i in range(min(n_dec_spikes, 5)):
        n_finite_spike = np.sum(np.isfinite(ll_original[i]))
        print(f"  Spike {i}: {n_finite_spike}/{n_pos_bins} finite values")
    if n_dec_spikes > 5:
        print(f"  ... ({n_dec_spikes - 5} more spikes)")
    print()

    # Log-space with use_gemm=False (same as original)
    print("-" * 80)
    print("LOG IMPLEMENTATION with use_gemm=False (same computation)")
    print("-" * 80)
    print()

    ll_log_no_gemm = estimate_log(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
        use_gemm=False,
    )

    print("Results:")
    print(f"  Shape: {ll_log_no_gemm.shape}")
    print(f"  Dtype: {ll_log_no_gemm.dtype}")
    print()

    # Analyze values
    finite_mask_log = np.isfinite(ll_log_no_gemm)
    n_finite_log = np.sum(finite_mask_log)
    n_inf_log = np.sum(np.isinf(ll_log_no_gemm))
    n_nan_log = np.sum(np.isnan(ll_log_no_gemm))

    print(f"Value distribution:")
    print(f"  Finite values: {n_finite_log}/{ll_log_no_gemm.size} ({100*n_finite_log/ll_log_no_gemm.size:.1f}%)")
    print(f"  -Inf values: {n_inf_log}/{ll_log_no_gemm.size} ({100*n_inf_log/ll_log_no_gemm.size:.1f}%)")
    print(f"  NaN values: {n_nan_log}/{ll_log_no_gemm.size} ({100*n_nan_log/ll_log_no_gemm.size:.1f}%)")
    print()

    # Compare with original
    print("Comparison with original:")
    both_finite = finite_mask & finite_mask_log
    if np.any(both_finite):
        max_diff = np.max(np.abs(ll_original[both_finite] - ll_log_no_gemm[both_finite]))
        print(f"  Max difference (finite values): {max_diff:.2e}")

    agreement = np.sum(finite_mask == finite_mask_log)
    print(f"  Finite/Inf pattern agreement: {agreement}/{ll_original.size} ({100*agreement/ll_original.size:.1f}%)")
    print()

    # Log-space with use_gemm=True (GEMM optimization)
    print("-" * 80)
    print("LOG IMPLEMENTATION with use_gemm=True (GEMM optimization)")
    print("-" * 80)
    print()

    ll_log_gemm = estimate_log(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
        use_gemm=True,
    )

    print("Results:")
    print(f"  Shape: {ll_log_gemm.shape}")
    print(f"  Dtype: {ll_log_gemm.dtype}")
    print()

    # Analyze values
    finite_mask_gemm = np.isfinite(ll_log_gemm)
    n_finite_gemm = np.sum(finite_mask_gemm)
    n_inf_gemm = np.sum(np.isinf(ll_log_gemm))
    n_nan_gemm = np.sum(np.isnan(ll_log_gemm))

    print(f"Value distribution:")
    print(f"  Finite values: {n_finite_gemm}/{ll_log_gemm.size} ({100*n_finite_gemm/ll_log_gemm.size:.1f}%)")
    print(f"  -Inf values: {n_inf_gemm}/{ll_log_gemm.size} ({100*n_inf_gemm/ll_log_gemm.size:.1f}%)")
    print(f"  NaN values: {n_nan_gemm}/{ll_log_gemm.size} ({100*n_nan_gemm/ll_log_gemm.size:.1f}%)")
    print()

    if n_finite_gemm > 0:
        print(f"Finite value statistics:")
        print(f"  Min: {ll_log_gemm[finite_mask_gemm].min():.4f}")
        print(f"  Max: {ll_log_gemm[finite_mask_gemm].max():.4f}")
        print(f"  Mean: {ll_log_gemm[finite_mask_gemm].mean():.4f}")
        print()

    # Show per-spike breakdown
    print("Per-spike breakdown:")
    for i in range(min(n_dec_spikes, 5)):
        n_finite_spike = np.sum(np.isfinite(ll_log_gemm[i]))
        print(f"  Spike {i}: {n_finite_spike}/{n_pos_bins} finite values")
    if n_dec_spikes > 5:
        print(f"  ... ({n_dec_spikes - 5} more spikes)")
    print()

    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()

    print("| Implementation | Finite Values | -Inf Values | NaN Values |")
    print("|----------------|---------------|-------------|------------|")
    print(f"| Original       | {n_finite:3d}/{ll_original.size} ({100*n_finite/ll_original.size:5.1f}%) | {n_inf:3d}/{ll_original.size} ({100*n_inf/ll_original.size:5.1f}%) | {n_nan:3d}/{ll_original.size} ({100*n_nan/ll_original.size:4.1f}%) |")
    print(f"| Log (no GEMM)  | {n_finite_log:3d}/{ll_log_no_gemm.size} ({100*n_finite_log/ll_log_no_gemm.size:5.1f}%) | {n_inf_log:3d}/{ll_log_no_gemm.size} ({100*n_inf_log/ll_log_no_gemm.size:5.1f}%) | {n_nan_log:3d}/{ll_log_no_gemm.size} ({100*n_nan_log/ll_log_no_gemm.size:4.1f}%) |")
    print(f"| Log (GEMM)     | {n_finite_gemm:3d}/{ll_log_gemm.size} ({100*n_finite_gemm/ll_log_gemm.size:5.1f}%) | {n_inf_gemm:3d}/{ll_log_gemm.size} ({100*n_inf_gemm/ll_log_gemm.size:5.1f}%) | {n_nan_gemm:3d}/{ll_log_gemm.size} ({100*n_nan_gemm/ll_log_gemm.size:4.1f}%) |")
    print()

    # Conclusions
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()

    if n_finite_gemm > n_finite:
        improvement = n_finite_gemm - n_finite
        print(f"✅ GEMM optimization improves numerical stability:")
        print(f"   {improvement} more finite values ({100*improvement/ll_original.size:.1f}% of total)")
        print()
        print("   The GEMM approach computes in log-space throughout,")
        print("   avoiding the exp(log(...)) round-trip that can cause underflow.")
        print()
    elif n_finite_gemm == n_finite:
        print("⚖️  GEMM optimization has same numerical behavior:")
        print(f"   Both produce {n_finite} finite values")
        print()
        print("   For these extreme features, both approaches underflow similarly.")
        print()
    else:
        print("⚠️  GEMM optimization has worse numerical stability:")
        print(f"   {n_finite - n_finite_gemm} fewer finite values")
        print()

    if n_finite_log == n_finite:
        print("✓  Log version with use_gemm=False matches original exactly")
        print("   (as expected - same computation)")
    else:
        print("⚠️  Log version with use_gemm=False differs from original")
        print("   (unexpected - should be identical)")

    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if n_finite_gemm > n_finite * 1.1:  # At least 10% improvement
        print("**Use GEMM optimization for extreme features**")
        print()
        print(f"The GEMM approach (use_gemm=True) significantly improves")
        print(f"numerical stability, preserving {100*n_finite_gemm/ll_log_gemm.size:.1f}% finite values")
        print(f"vs {100*n_finite/ll_original.size:.1f}% for the original.")
        print()
        print("This is especially important for:")
        print("- High-dimensional waveform features")
        print("- Large feature distances")
        print("- Critical applications requiring numerical precision")
    else:
        print("**GEMM optimization does not significantly help with extreme features**")
        print()
        print(f"Both approaches produce similar amounts of underflow")
        print(f"({100*n_finite/ll_original.size:.1f}% vs {100*n_finite_gemm/ll_log_gemm.size:.1f}% finite).")
        print()
        print("With such extreme feature distances, underflow is unavoidable.")
        print("Consider:")
        print("- Using smaller waveform_std values")
        print("- Normalizing/scaling features differently")
        print("- Using dimensionality reduction on waveform features")

    print()


if __name__ == "__main__":
    analyze_extreme_features()
