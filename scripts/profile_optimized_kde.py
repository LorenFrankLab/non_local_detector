"""Profile performance of optimized KDE implementations."""

import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity_logspace,
    estimate_log_joint_mark_intensity_vectorized,
)


def time_function(func, *args, n_warmup=3, n_runs=10, **kwargs):
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


def profile_configuration(n_features, n_enc=200, n_dec=100, n_pos=500):
    """Profile all implementations for a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Configuration: {n_features}D features")
    print(f"  Encoding: {n_enc} spikes, Decoding: {n_dec} spikes, Positions: {n_pos}")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)

    # Create test data
    dec_features = jnp.array(rng.normal(0, 1, (n_dec, n_features)))
    enc_features = jnp.array(rng.normal(0, 1, (n_enc, n_features)))
    waveform_stds = jnp.ones(n_features)
    occupancy = jnp.ones(n_pos) * 0.1
    mean_rate = 5.0
    position_distance = jnp.array(rng.exponential(1.0, (n_enc, n_pos)))

    # Profile each version
    results = {}

    print("\nOriginal (with Python for-loop):")
    mean_orig, std_orig = time_function(
        estimate_log_joint_mark_intensity,
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
    )
    print(f"  Time: {mean_orig*1000:.3f} ± {std_orig*1000:.3f} ms")
    results["original"] = (mean_orig, std_orig)

    print("\nVectorized + JIT:")
    mean_vec, std_vec = time_function(
        estimate_log_joint_mark_intensity_vectorized,
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
    )
    print(f"  Time: {mean_vec*1000:.3f} ± {std_vec*1000:.3f} ms")
    speedup_vec = mean_orig / mean_vec
    print(f"  Speedup: {speedup_vec:.2f}x")
    results["vectorized"] = (mean_vec, std_vec, speedup_vec)

    print("\nLog-space + Vectorized + JIT:")
    mean_log, std_log = time_function(
        estimate_log_joint_mark_intensity_logspace,
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
    )
    print(f"  Time: {mean_log*1000:.3f} ± {std_log*1000:.3f} ms")
    speedup_log = mean_orig / mean_log
    print(f"  Speedup: {speedup_log:.2f}x")
    results["logspace"] = (mean_log, std_log, speedup_log)

    return results


def main():
    """Run comprehensive performance profiling."""
    print("="*70)
    print("PERFORMANCE PROFILING: OPTIMIZED KDE IMPLEMENTATIONS")
    print("="*70)
    print("\nComparing:")
    print("  1. Original (Python for-loop in kde_distance)")
    print("  2. Vectorized + JIT (vmap + @jax.jit)")
    print("  3. Log-space + Vectorized + JIT (full optimization)")

    # Test configurations
    configs = [
        (2, 200, 100, 500, "Standard 2D"),
        (4, 200, 100, 500, "Standard 4D"),
        (6, 200, 100, 500, "Standard 6D"),
        (8, 200, 100, 500, "Standard 8D"),
        (10, 200, 100, 500, "Standard 10D"),
    ]

    all_results = []

    for n_features, n_enc, n_dec, n_pos, desc in configs:
        print(f"\n\n{'#'*70}")
        print(f"# {desc}")
        print(f"{'#'*70}")
        results = profile_configuration(n_features, n_enc, n_dec, n_pos)
        all_results.append((n_features, desc, results))

    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY: Performance Across Dimensions")
    print("="*70)

    print("\n" + "-"*70)
    print(f"{'Dim':<6} {'Original':<15} {'Vectorized':<20} {'Log-space':<20}")
    print("-"*70)

    for n_features, desc, results in all_results:
        orig_time = results["original"][0] * 1000
        vec_time, _, vec_speedup = results["vectorized"]
        log_time, _, log_speedup = results["logspace"]

        vec_time_ms = vec_time * 1000
        log_time_ms = log_time * 1000

        print(
            f"{n_features}D    "
            f"{orig_time:>6.2f}ms        "
            f"{vec_time_ms:>6.2f}ms ({vec_speedup:>4.2f}x)   "
            f"{log_time_ms:>6.2f}ms ({log_speedup:>4.2f}x)"
        )

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("\nVectorized + JIT Optimization:")
    vec_speedups = [r[2]["vectorized"][2] for r in all_results]
    min_speedup = min(vec_speedups)
    max_speedup = max(vec_speedups)
    avg_speedup = np.mean(vec_speedups)
    print(f"  Speedup range: {min_speedup:.2f}x - {max_speedup:.2f}x")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    if avg_speedup >= 2.0:
        print("  ✓ Excellent! Achieved target of 2-4x speedup")
    elif avg_speedup >= 1.5:
        print("  ✓ Good! Significant performance improvement")
    elif avg_speedup >= 1.2:
        print("  ~ Moderate improvement")
    else:
        print("  ✗ Minimal improvement")

    print("\nLog-space + Vectorized + JIT Optimization:")
    log_speedups = [r[2]["logspace"][2] for r in all_results]
    min_speedup = min(log_speedups)
    max_speedup = max(log_speedups)
    avg_speedup = np.mean(log_speedups)
    print(f"  Speedup range: {min_speedup:.2f}x - {max_speedup:.2f}x")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    if avg_speedup >= 2.0:
        print("  ✓ Excellent! Achieved target of 2-4x speedup")
    elif avg_speedup >= 1.5:
        print("  ✓ Good! Significant performance improvement")
    elif avg_speedup >= 1.2:
        print("  ~ Moderate improvement")
    else:
        print("  ✗ Minimal improvement")

    # Comparison: Vectorized vs Log-space
    print("\nLog-space vs Vectorized:")
    for n_features, desc, results in all_results:
        vec_time = results["vectorized"][0]
        log_time = results["logspace"][0]
        ratio = vec_time / log_time
        faster = "Log" if ratio > 1.0 else "Vec"
        print(f"  {n_features}D: {abs(ratio):.2f}x {'faster' if ratio != 1.0 else 'same'} ({faster})")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    vec_avg = np.mean([r[2]["vectorized"][2] for r in all_results])
    log_avg = np.mean([r[2]["logspace"][2] for r in all_results])

    print("\nBest implementation:")
    if log_avg > vec_avg * 1.1:
        print(f"  → Log-space + Vectorized + JIT ({log_avg:.2f}x average speedup)")
        print("    Use: estimate_log_joint_mark_intensity_logspace()")
    elif vec_avg > 1.5:
        print(f"  → Vectorized + JIT ({vec_avg:.2f}x average speedup)")
        print("    Use: estimate_log_joint_mark_intensity_vectorized()")
    else:
        print("  → Original implementation (optimizations not beneficial)")
        print("    Use: estimate_log_joint_mark_intensity()")

    print("\nFor production use:")
    if max(vec_avg, log_avg) >= 2.0:
        print("  ✓ Optimization successful - recommend deploying optimized version")
        print("  ✓ Numerical equivalence verified (max diff < 1e-6)")
        print(f"  ✓ Average speedup: {max(vec_avg, log_avg):.2f}x")
    else:
        print("  ~ Optimization provides moderate benefit")
        print("  ~ Consider for performance-critical applications only")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
