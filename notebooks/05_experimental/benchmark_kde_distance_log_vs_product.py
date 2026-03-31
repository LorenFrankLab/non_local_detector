"""Benchmark: kde_distance (product-space) vs log_kde_distance (log-space)

Compares speed and numerical accuracy of the two KDE distance implementations
across varying numbers of dimensions, samples, and evaluation points.

The product-space version multiplies Gaussian PDFs and can underflow to zero
for high-dimensional data. The log-space version sums log-PDFs and avoids this.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.clusterless_kde_log import (
    kde_distance,
    log_kde_distance,
)

# Ensure JIT compilation is triggered before timing
jax.config.update("jax_platform_name", "cpu")

rng = np.random.default_rng(42)


def benchmark_one(n_samples, n_eval, n_dims, n_repeats=5):
    """Benchmark both implementations and compare results."""
    samples = jnp.array(rng.standard_normal((n_samples, n_dims)), dtype=jnp.float32)
    eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)), dtype=jnp.float32)
    std = jnp.ones(n_dims, dtype=jnp.float32) * 1.0

    # Warmup (JIT compile)
    _ = kde_distance(eval_points, samples, std).block_until_ready()
    _ = log_kde_distance(eval_points, samples, std).block_until_ready()

    # Time product-space
    times_product = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result_product = kde_distance(eval_points, samples, std).block_until_ready()
        times_product.append(time.perf_counter() - t0)

    # Time log-space
    times_log = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result_log = log_kde_distance(eval_points, samples, std).block_until_ready()
        times_log.append(time.perf_counter() - t0)

    # Numerical comparison
    result_product_np = np.asarray(result_product)
    result_log_np = np.asarray(result_log)
    result_product_from_log = np.exp(np.asarray(result_log))

    n_zeros_product = np.sum(result_product_np == 0.0)
    n_zeros_from_log = np.sum(result_product_from_log == 0.0)
    n_total = result_product_np.size

    # Where product didn't underflow, check agreement
    nonzero_mask = result_product_np > 0.0
    if nonzero_mask.any():
        log_from_product = np.log(result_product_np[nonzero_mask])
        log_direct = result_log_np[nonzero_mask]
        max_log_diff = np.max(np.abs(log_from_product - log_direct))
    else:
        max_log_diff = float("inf")

    return {
        "n_samples": n_samples,
        "n_eval": n_eval,
        "n_dims": n_dims,
        "time_product_ms": np.median(times_product) * 1000,
        "time_log_ms": np.median(times_log) * 1000,
        "speedup": np.median(times_log) / np.median(times_product),
        "n_zeros_product": n_zeros_product,
        "n_zeros_from_log": n_zeros_from_log,
        "n_total": n_total,
        "pct_underflow_product": 100.0 * n_zeros_product / n_total,
        "max_log_diff_where_both_nonzero": max_log_diff,
    }


# ---- Run benchmarks ----
print("=" * 90)
print(f"{'n_samp':>7} {'n_eval':>7} {'n_dims':>6} | "
      f"{'product(ms)':>11} {'log(ms)':>9} {'slowdown':>8} | "
      f"{'underflow%':>10} {'max_log_diff':>12}")
print("=" * 90)

configs = [
    # Vary dimensions (the key parameter for underflow)
    (500, 200, 2),
    (500, 200, 4),
    (500, 200, 8),
    (500, 200, 12),
    (500, 200, 16),
    (500, 200, 24),
    (500, 200, 32),
    # Vary sample count
    (100, 200, 4),
    (1000, 200, 4),
    (5000, 200, 4),
    # Vary eval count
    (500, 50, 4),
    (500, 500, 4),
    (500, 2000, 4),
]

results = []
for n_samples, n_eval, n_dims in configs:
    r = benchmark_one(n_samples, n_eval, n_dims)
    results.append(r)
    print(
        f"{r['n_samples']:>7} {r['n_eval']:>7} {r['n_dims']:>6} | "
        f"{r['time_product_ms']:>11.2f} {r['time_log_ms']:>9.2f} {r['speedup']:>7.2f}x | "
        f"{r['pct_underflow_product']:>9.1f}% {r['max_log_diff_where_both_nonzero']:>12.2e}"
    )

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"\nDimensions where product-space underflow becomes significant (>1%):")
for r in results:
    if r["pct_underflow_product"] > 1.0:
        print(f"  n_dims={r['n_dims']}: {r['pct_underflow_product']:.1f}% underflow")

print(f"\nSpeed comparison (log-space slowdown factor):")
for r in results:
    print(f"  ({r['n_samples']}, {r['n_eval']}, {r['n_dims']}): {r['speedup']:.2f}x")
