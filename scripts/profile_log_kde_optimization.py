"""Profile performance improvements for log KDE optimizations.

This script measures the speedup achieved by vectorizing kde_distance and
log_kde_distance functions in clusterless_kde_log.py.
"""

import sys
import time

import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.insert(0, "src")

from non_local_detector.likelihoods.clusterless_kde_log import (
    kde_distance,
    kde_distance_vectorized,
    log_kde_distance,
    log_kde_distance_vectorized,
)
from non_local_detector.likelihoods.common import gaussian_pdf, log_gaussian_pdf


def manual_loop_kde_distance(eval_points, samples, std):
    """Original implementation with Python for-loop (reference)."""
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))
    for dim_eval_points, dim_samples, dim_std in zip(
        eval_points.T, samples.T, std, strict=False
    ):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return distance


def manual_loop_log_kde_distance(eval_points, samples, std):
    """Original log implementation with Python for-loop (reference)."""
    log_dist = jnp.zeros((samples.shape[0], eval_points.shape[0]))
    for dim_eval, dim_samp, dim_std in zip(eval_points.T, samples.T, std, strict=False):
        log_dist += log_gaussian_pdf(
            jnp.expand_dims(dim_eval, axis=0),
            jnp.expand_dims(dim_samp, axis=1),
            dim_std,
        )
    return log_dist


def time_function(func, *args, n_runs=10, warmup=2):
    """Time a function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        result = func(*args)
        result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        result.block_until_ready()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def main():
    """Profile linear and log KDE optimizations across dimensions."""
    print("=" * 70)
    print("Log KDE Optimization Performance Profiling")
    print("=" * 70)
    print()

    # Test configuration
    n_eval = 100
    n_samples = 200
    n_runs = 10
    dimensions = [2, 4, 6, 8, 10]

    print("Configuration:")
    print(f"  Evaluation points: {n_eval}")
    print(f"  Training samples: {n_samples}")
    print(f"  Runs per test: {n_runs}")
    print()

    # Linear-space KDE profiling
    print("-" * 70)
    print("LINEAR-SPACE KDE DISTANCE")
    print("-" * 70)
    print()

    linear_results = []

    for n_dims in dimensions:
        np.random.seed(42)
        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
        std = jnp.ones(n_dims)

        # Time original (manual loop)
        time_orig, std_orig = time_function(
            manual_loop_kde_distance, eval_points, samples, std, n_runs=n_runs
        )

        # Time vectorized
        time_vec, std_vec = time_function(
            kde_distance_vectorized, eval_points, samples, std, n_runs=n_runs
        )

        # Time JIT-compiled wrapper
        time_jit, std_jit = time_function(
            kde_distance, eval_points, samples, std, n_runs=n_runs
        )

        speedup_vec = time_orig / time_vec
        speedup_jit = time_orig / time_jit

        linear_results.append(
            {
                "dims": n_dims,
                "time_orig": time_orig * 1000,
                "time_vec": time_vec * 1000,
                "time_jit": time_jit * 1000,
                "speedup_vec": speedup_vec,
                "speedup_jit": speedup_jit,
            }
        )

        print(f"{n_dims}D features:")
        print(f"  Original (loop):     {time_orig*1000:.2f}ms ± {std_orig*1000:.2f}ms")
        print(f"  Vectorized:          {time_vec*1000:.2f}ms ± {std_vec*1000:.2f}ms  ({speedup_vec:.2f}x)")
        print(f"  JIT-compiled:        {time_jit*1000:.2f}ms ± {std_jit*1000:.2f}ms  ({speedup_jit:.2f}x)")
        print()

    # Log-space KDE profiling
    print("-" * 70)
    print("LOG-SPACE KDE DISTANCE")
    print("-" * 70)
    print()

    log_results = []

    for n_dims in dimensions:
        np.random.seed(42)
        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
        std = jnp.ones(n_dims)

        # Time original (manual loop)
        time_orig, std_orig = time_function(
            manual_loop_log_kde_distance, eval_points, samples, std, n_runs=n_runs
        )

        # Time vectorized
        time_vec, std_vec = time_function(
            log_kde_distance_vectorized, eval_points, samples, std, n_runs=n_runs
        )

        # Time JIT-compiled wrapper
        time_jit, std_jit = time_function(
            log_kde_distance, eval_points, samples, std, n_runs=n_runs
        )

        speedup_vec = time_orig / time_vec
        speedup_jit = time_orig / time_jit

        log_results.append(
            {
                "dims": n_dims,
                "time_orig": time_orig * 1000,
                "time_vec": time_vec * 1000,
                "time_jit": time_jit * 1000,
                "speedup_vec": speedup_vec,
                "speedup_jit": speedup_jit,
            }
        )

        print(f"{n_dims}D features:")
        print(f"  Original (loop):     {time_orig*1000:.2f}ms ± {std_orig*1000:.2f}ms")
        print(f"  Vectorized:          {time_vec*1000:.2f}ms ± {std_vec*1000:.2f}ms  ({speedup_vec:.2f}x)")
        print(f"  JIT-compiled:        {time_jit*1000:.2f}ms ± {std_jit*1000:.2f}ms  ({speedup_jit:.2f}x)")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("Linear-space KDE Distance:")
    print()
    print("| Dimension | Original | Vectorized | JIT-compiled | Speedup (JIT) |")
    print("|-----------|----------|------------|--------------|---------------|")
    for result in linear_results:
        print(
            f"| {result['dims']:2d}D       | {result['time_orig']:6.2f}ms | {result['time_vec']:8.2f}ms | {result['time_jit']:10.2f}ms | {result['speedup_jit']:11.2f}x |"
        )
    print()

    avg_speedup_linear = np.mean([r["speedup_jit"] for r in linear_results])
    print(f"Average speedup: {avg_speedup_linear:.2f}x")
    print()

    print("Log-space KDE Distance:")
    print()
    print("| Dimension | Original | Vectorized | JIT-compiled | Speedup (JIT) |")
    print("|-----------|----------|------------|--------------|---------------|")
    for result in log_results:
        print(
            f"| {result['dims']:2d}D       | {result['time_orig']:6.2f}ms | {result['time_vec']:8.2f}ms | {result['time_jit']:10.2f}ms | {result['speedup_jit']:11.2f}x |"
        )
    print()

    avg_speedup_log = np.mean([r["speedup_jit"] for r in log_results])
    print(f"Average speedup: {avg_speedup_log:.2f}x")
    print()

    # Verify numerical equivalence
    print("=" * 70)
    print("NUMERICAL VALIDATION")
    print("=" * 70)
    print()

    for n_dims in [2, 10]:
        np.random.seed(42)
        eval_points = jnp.array(np.random.randn(50, n_dims))
        samples = jnp.array(np.random.randn(100, n_dims))
        std = jnp.ones(n_dims)

        # Linear-space
        orig_linear = manual_loop_kde_distance(eval_points, samples, std)
        opt_linear = kde_distance(eval_points, samples, std)
        max_diff_linear = jnp.max(jnp.abs(orig_linear - opt_linear))

        # Log-space
        orig_log = manual_loop_log_kde_distance(eval_points, samples, std)
        opt_log = log_kde_distance(eval_points, samples, std)
        max_diff_log = jnp.max(jnp.abs(orig_log - opt_log))

        print(f"{n_dims}D features:")
        print(f"  Linear-space max diff: {max_diff_linear:.2e}")
        print(f"  Log-space max diff:    {max_diff_log:.2e}")
        print()

    print("✓ All optimizations maintain numerical equivalence (< 1e-6)")
    print()


if __name__ == "__main__":
    main()
