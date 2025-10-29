"""Profile core clusterless KDE functions across different waveform feature dimensions.

This simplified script directly tests the core computation functions to isolate
the impact of feature dimensionality on GEMM vs linear performance.
"""

import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_ref,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity as estimate_log,
)


def time_function(func, *args, n_warmup=2, n_runs=5, **kwargs):
    """Time a function with warmup runs for JIT compilation."""
    # Warmup
    for _ in range(n_warmup):
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def profile_dimension(n_features, n_enc=200, n_dec=100, n_pos=500):
    """Profile performance for a specific feature dimension."""
    print(f"\n{'='*70}")
    print(f"Testing {n_features}D waveform features")
    print(f"  n_encoding={n_enc}, n_decoding={n_dec}, n_positions={n_pos}")
    print(f"{'='*70}")

    rng = np.random.default_rng(0)

    # Create test data
    dec_features = jnp.array(rng.normal(0, 1, (n_dec, n_features)))
    enc_features = jnp.array(rng.normal(0, 1, (n_enc, n_features)))
    waveform_stds = jnp.ones(n_features)
    occupancy = jnp.ones(n_pos) * 0.1
    mean_rate = 5.0
    position_distance = jnp.array(rng.exponential(1.0, (n_enc, n_pos)))

    # Reference implementation (log-space, simple)
    print("\nReference (clusterless_kde.py - log-space, simple):")
    mean_ref, std_ref = time_function(
        estimate_ref,
        decoding_spike_waveform_features=dec_features,
        encoding_spike_waveform_features=enc_features,
        waveform_stds=waveform_stds,
        occupancy=occupancy,
        mean_rate=mean_rate,
        position_distance=position_distance,
    )
    print(f"  Time: {mean_ref*1000:.2f} ± {std_ref*1000:.2f} ms")

    # Log-space with GEMM (vmap)
    print("\nOptimized with GEMM + vmap (use_gemm=True):")
    mean_gemm, std_gemm = time_function(
        estimate_log,
        decoding_spike_waveform_features=dec_features,
        encoding_spike_waveform_features=enc_features,
        waveform_stds=waveform_stds,
        occupancy=occupancy,
        mean_rate=mean_rate,
        position_distance=position_distance,
        use_gemm=True,
    )
    print(f"  Time: {mean_gemm*1000:.2f} ± {std_gemm*1000:.2f} ms")

    # Log-space without GEMM (fallback to reference algorithm in log file)
    print("\nOptimized without GEMM (use_gemm=False):")
    mean_no_gemm, std_no_gemm = time_function(
        estimate_log,
        decoding_spike_waveform_features=dec_features,
        encoding_spike_waveform_features=enc_features,
        waveform_stds=waveform_stds,
        occupancy=occupancy,
        mean_rate=mean_rate,
        position_distance=position_distance,
        use_gemm=False,
    )
    print(f"  Time: {mean_no_gemm*1000:.2f} ± {std_no_gemm*1000:.2f} ms")

    # Speedup analysis
    speedup_gemm = mean_ref / mean_gemm
    speedup_no_gemm = mean_ref / mean_no_gemm
    gemm_vs_no_gemm = mean_no_gemm / mean_gemm

    print("\nSpeedup vs reference:")
    print(f"  GEMM (vmap):     {speedup_gemm:.2f}x")
    print(f"  No GEMM:         {speedup_no_gemm:.2f}x")
    print(f"\nGEMM vs No-GEMM: {gemm_vs_no_gemm:.2f}x")

    return {
        "n_features": n_features,
        "ref_time": mean_ref,
        "ref_std": std_ref,
        "gemm_time": mean_gemm,
        "gemm_std": std_gemm,
        "no_gemm_time": mean_no_gemm,
        "no_gemm_std": std_no_gemm,
        "speedup_gemm": speedup_gemm,
        "speedup_no_gemm": speedup_no_gemm,
        "gemm_vs_no_gemm": gemm_vs_no_gemm,
    }


def main():
    """Run profiling across multiple feature dimensions."""
    print("="*70)
    print("FEATURE DIMENSION PROFILING (Core Functions Only)")
    print("="*70)
    print("\nTesting hypothesis: GEMM optimization performs better")
    print("with higher-dimensional waveform features.")
    print("\nTest configuration:")
    print("  - Encoding spikes: 200")
    print("  - Decoding spikes: 100")
    print("  - Position bins: 500")
    print("  - Feature dimensions: 2D, 4D, 6D, 8D, 10D")

    dimensions = [2, 4, 6, 8, 10]
    results = []

    for n_features in dimensions:
        result = profile_dimension(n_features)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Performance vs Feature Dimension")
    print("="*70)
    print("\nExecution time (ms):")
    print(f"{'Dim':<6} {'Reference':<15} {'GEMM':<15} {'No-GEMM':<15}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['n_features']:<6} "
            f"{r['ref_time']*1000:>6.2f} ± {r['ref_std']*1000:>4.2f}   "
            f"{r['gemm_time']*1000:>6.2f} ± {r['gemm_std']*1000:>4.2f}   "
            f"{r['no_gemm_time']*1000:>6.2f} ± {r['no_gemm_std']*1000:>4.2f}"
        )

    print("\nSpeedup vs Reference:")
    print(f"{'Dim':<6} {'GEMM (vmap)':<15} {'No-GEMM':<15} {'GEMM vs No-GEMM':<15}")
    print("-" * 70)
    for r in results:
        gemm_marker = "✓" if r['speedup_gemm'] > 1.0 else "✗"
        no_gemm_marker = "✓" if r['speedup_no_gemm'] > 1.0 else "✗"
        gemm_win = "✓" if r['gemm_vs_no_gemm'] > 1.0 else "✗"
        print(
            f"{r['n_features']:<6} "
            f"{r['speedup_gemm']:>6.2f}x {gemm_marker:<8} "
            f"{r['speedup_no_gemm']:>6.2f}x {no_gemm_marker:<8} "
            f"{r['gemm_vs_no_gemm']:>6.2f}x {gemm_win:<8}"
        )

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Find when GEMM wins vs linear
    gemm_wins_linear = [r for r in results if r['speedup_gemm'] > 1.0]
    if gemm_wins_linear:
        min_dim = min(r['n_features'] for r in gemm_wins_linear)
        print(f"\n✓ GEMM (vmap) beats linear reference at {min_dim}D+ features")
    else:
        print("\n✗ GEMM (vmap) never beats linear reference")

    # Find when GEMM wins vs no-GEMM
    gemm_wins_no_gemm = [r for r in results if r['gemm_vs_no_gemm'] > 1.0]
    if gemm_wins_no_gemm:
        min_dim = min(r['n_features'] for r in gemm_wins_no_gemm)
        print(f"✓ GEMM optimization is worthwhile at {min_dim}D+ features")
    else:
        print("✗ GEMM optimization overhead not justified at any dimension")

    # Trend analysis
    print("\nTrends:")
    gemm_speedups = [r['speedup_gemm'] for r in results]
    if gemm_speedups[-1] > gemm_speedups[0]:
        print(f"  GEMM speedup improves with dimension: {gemm_speedups[0]:.2f}x → {gemm_speedups[-1]:.2f}x")
    else:
        print(f"  GEMM speedup decreases with dimension: {gemm_speedups[0]:.2f}x → {gemm_speedups[-1]:.2f}x")

    # Best configuration per dimension
    print("\nRecommended configuration per dimension:")
    for r in results:
        times = {
            "GEMM (vmap)": r['gemm_time'],
            "No-GEMM": r['no_gemm_time'],
            "Reference": r['ref_time'],
        }
        best = min(times, key=times.get)
        speedup = r['ref_time'] / min(times.values())
        print(f"  {r['n_features']}D: {best} ({speedup:.2f}x faster)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
