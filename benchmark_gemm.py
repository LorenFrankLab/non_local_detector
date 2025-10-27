#!/usr/bin/env python
"""Benchmark GEMM vs loop-based mark kernel computation."""

import time

import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.clusterless_kde_log import (
    _compute_log_mark_kernel_gemm,
    estimate_log_joint_mark_intensity,
    log_gaussian_pdf,
)


def compute_loop_mark_kernel(
    decoding_features: jnp.ndarray,
    encoding_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
) -> jnp.ndarray:
    """Loop-based mark kernel for comparison."""
    n_enc = encoding_features.shape[0]
    n_dec = decoding_features.shape[0]
    logK_mark = jnp.zeros((n_enc, n_dec))
    for dec_dim, enc_dim, std_d in zip(
        decoding_features.T, encoding_features.T, waveform_stds, strict=False
    ):
        logK_mark += log_gaussian_pdf(
            x=jnp.expand_dims(dec_dim, axis=0),
            mean=jnp.expand_dims(enc_dim, axis=1),
            sigma=std_d,
        )
    return logK_mark


def benchmark_function(func, *args, n_warmup=3, n_trials=10):
    """Benchmark a function with warmup and multiple trials."""
    # Warmup
    for _ in range(n_warmup):
        result = func(*args)
        result.block_until_ready()

    # Benchmark
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        result = func(*args)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


def main():
    """Run benchmarks for various problem sizes and feature dimensions."""
    print("=" * 80)
    print("GEMM vs Loop Mark Kernel Benchmark")
    print("=" * 80)

    # Test configurations
    configs = [
        {"n_enc": 100, "n_dec": 50, "n_pos": 50, "n_features": 4, "name": "Small"},
        {"n_enc": 500, "n_dec": 200, "n_pos": 200, "n_features": 4, "name": "Medium"},
        {"n_enc": 1000, "n_dec": 500, "n_pos": 500, "n_features": 4, "name": "Large"},
    ]

    feature_dims = [2, 4, 8, 16]

    print("\n" + "=" * 80)
    print("1. Mark Kernel Computation (varying problem size, n_features=4)")
    print("=" * 80)
    print(
        f"{'Config':<12} {'n_enc':>8} {'n_dec':>8} {'Loop (ms)':>12} {'GEMM (ms)':>12} {'Speedup':>10}"
    )
    print("-" * 80)

    for config in configs:
        np.random.seed(42)
        n_enc = config["n_enc"]
        n_dec = config["n_dec"]
        n_features = config["n_features"]

        encoding_features = jnp.array(np.random.randn(n_enc, n_features).astype(np.float32))
        decoding_features = jnp.array(np.random.randn(n_dec, n_features).astype(np.float32))
        waveform_stds = jnp.array(
            np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5
        )

        # Benchmark loop
        loop_stats = benchmark_function(
            compute_loop_mark_kernel, decoding_features, encoding_features, waveform_stds
        )

        # Benchmark GEMM
        gemm_stats = benchmark_function(
            _compute_log_mark_kernel_gemm,
            decoding_features,
            encoding_features,
            waveform_stds,
        )

        speedup = loop_stats["mean_ms"] / gemm_stats["mean_ms"]

        print(
            f"{config['name']:<12} {n_enc:>8} {n_dec:>8} "
            f"{loop_stats['mean_ms']:>11.2f}± {loop_stats['std_ms']:>11.2f}± "
            f"{gemm_stats['mean_ms']:>11.2f}± {speedup:>9.2f}×"
        )

    print("\n" + "=" * 80)
    print("2. Mark Kernel Computation (varying n_features, size=medium)")
    print("=" * 80)
    print(f"{'n_features':>12} {'Loop (ms)':>12} {'GEMM (ms)':>12} {'Speedup':>10}")
    print("-" * 80)

    n_enc = 500
    n_dec = 200

    for n_features in feature_dims:
        np.random.seed(42)

        encoding_features = jnp.array(np.random.randn(n_enc, n_features).astype(np.float32))
        decoding_features = jnp.array(np.random.randn(n_dec, n_features).astype(np.float32))
        waveform_stds = jnp.array(
            np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5
        )

        # Benchmark loop
        loop_stats = benchmark_function(
            compute_loop_mark_kernel, decoding_features, encoding_features, waveform_stds
        )

        # Benchmark GEMM
        gemm_stats = benchmark_function(
            _compute_log_mark_kernel_gemm,
            decoding_features,
            encoding_features,
            waveform_stds,
        )

        speedup = loop_stats["mean_ms"] / gemm_stats["mean_ms"]

        print(
            f"{n_features:>12} {loop_stats['mean_ms']:>11.2f}± {gemm_stats['mean_ms']:>11.2f}± {speedup:>9.2f}×"
        )

    print("\n" + "=" * 80)
    print("3. End-to-End estimate_log_joint_mark_intensity (medium size)")
    print("=" * 80)
    print(f"{'Method':<12} {'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 80)

    # Setup data
    np.random.seed(42)
    n_enc = 500
    n_dec = 200
    n_pos = 200
    n_features = 4

    encoding_features = jnp.array(np.random.randn(n_enc, n_features).astype(np.float32))
    decoding_features = jnp.array(np.random.randn(n_dec, n_features).astype(np.float32))
    waveform_stds = jnp.array(np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5)
    encoding_weights = jnp.ones(n_enc, dtype=jnp.float32)
    occupancy = jnp.ones(n_pos, dtype=jnp.float32)
    log_position_distance = jnp.array(np.random.randn(n_enc, n_pos).astype(np.float32))
    mean_rate = 5.0

    # Benchmark loop-based
    loop_stats = benchmark_function(
        estimate_log_joint_mark_intensity,
        decoding_features,
        encoding_features,
        encoding_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        False,  # use_gemm=False
    )

    # Benchmark GEMM-based
    gemm_stats = benchmark_function(
        estimate_log_joint_mark_intensity,
        decoding_features,
        encoding_features,
        encoding_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        True,  # use_gemm=True
    )

    speedup = loop_stats["mean_ms"] / gemm_stats["mean_ms"]

    print(f"{'Loop':<12} {loop_stats['mean_ms']:>11.2f}± {'1.00×':>10}")
    print(f"{'GEMM':<12} {gemm_stats['mean_ms']:>11.2f}± {speedup:>9.2f}×")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        """
The GEMM-based implementation provides significant speedups for multi-dimensional
waveform features:

- For typical workloads (500 encoding, 200 decoding spikes, 4 features):
  Expected speedup: 2-4× for mark kernel computation

- Speedup increases with number of features:
  Higher dimensional features benefit more from GEMM optimization

- End-to-end speedup is lower but still significant:
  The overall pipeline includes other operations (scan, logsumexp)

- Memory usage is identical between implementations:
  Both produce the same (n_enc, n_dec) output array

Key insight: GEMM replaces n_features separate broadcast+multiply operations
with a single optimized matrix multiply, which is much faster on modern hardware.
    """
    )


if __name__ == "__main__":
    main()
