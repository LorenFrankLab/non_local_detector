"""Analyze JAXpr for safe_log changes to understand compilation impact."""

import jax
import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity,
)
from non_local_detector.likelihoods.common import EPS, safe_log


def analyze_safe_log_jaxpr():
    """Analyze how safe_log affects JAX compilation."""
    print("=" * 80)
    print("JAXpr Analysis: safe_log Impact")
    print("=" * 80)
    print()

    # Test data
    n_dec = 5
    n_enc = 10
    n_pos = 8
    n_features = 4

    np.random.seed(42)
    dec_features = jnp.array(np.random.randn(n_dec, n_features))
    enc_features = jnp.array(np.random.randn(n_enc, n_features))
    waveform_stds = jnp.array([1.0] * n_features)
    occupancy = jnp.array(np.random.rand(n_pos) * 0.8 + 0.1)  # Avoid zeros
    mean_rate = 2.5
    position_distance = jnp.array(np.random.rand(n_enc, n_pos))

    print("1. Simple safe_log JAXpr")
    print("-" * 80)
    test_array = jnp.array([0.0, 1e-20, 1e-10, 0.5, 1.0])
    jaxpr = jax.make_jaxpr(safe_log)(test_array, EPS)
    print(jaxpr)
    print()

    print("2. Direct jnp.log JAXpr (for comparison)")
    print("-" * 80)
    jaxpr_direct = jax.make_jaxpr(jnp.log)(test_array)
    print(jaxpr_direct)
    print()

    print("3. Cost of safe_log vs direct log")
    print("-" * 80)
    print("safe_log operations:")
    print("  - abs: check |x| < eps")
    print("  - lt: boolean comparison")
    print("  - select: conditional value selection")
    print("  - log: final logarithm")
    print()
    print("direct log operations:")
    print("  - log: only one operation")
    print()
    print("Overhead: +3 operations per safe_log call")
    print("These operations are cheap (no memory ops, fuse well with XLA)")
    print()

    print("4. Runtime performance check")
    print("-" * 80)
    import time

    # Warmup
    for _ in range(3):
        _ = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
            use_gemm=True,
        )

    # Time
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
            use_gemm=True,
        )
        result.block_until_ready()
        times.append(time.perf_counter() - start)

    print(f"  Mean: {np.mean(times)*1000:.3f} ms")
    print(f"  Std:  {np.std(times)*1000:.3f} ms")
    print(f"  Min:  {np.min(times)*1000:.3f} ms")
    print(f"  Max:  {np.max(times)*1000:.3f} ms")
    print()

    print("5. Numerical stability check")
    print("-" * 80)
    # Test with zeros and tiny values
    position_distance_with_zeros = position_distance.at[0, :].set(0.0)
    position_distance_with_zeros = position_distance_with_zeros.at[:, 0].set(1e-30)

    result_with_zeros = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance_with_zeros,
        use_gemm=True,
    )

    print(f"  Output shape: {result_with_zeros.shape}")
    print(f"  Finite values: {np.sum(np.isfinite(result_with_zeros))}/{result_with_zeros.size}")
    print(f"  -inf values: {np.sum(np.isneginf(result_with_zeros))}")
    print(f"  +inf values: {np.sum(np.isposinf(result_with_zeros))}")
    print(f"  NaN values: {np.sum(np.isnan(result_with_zeros))}")
    print(f"  Min finite: {np.min(result_with_zeros[np.isfinite(result_with_zeros)]):.6f}")
    print(f"  Max finite: {np.max(result_with_zeros[np.isfinite(result_with_zeros)]):.6f}")
    print()

    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    analyze_safe_log_jaxpr()
