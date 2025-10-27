#!/usr/bin/env python
"""Detailed JAX profiling for clusterless KDE implementations.

This script uses JAX's built-in profiling tools to analyze:
- Compilation time vs execution time
- HLO operations
- Memory allocation patterns
- XLA optimization effectiveness

Usage:
    python scripts/profile_jax_detailed.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde_log import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    kde_distance,
)


def create_test_data(n_encoding=100, n_decoding=50, n_positions=200):
    """Create small test dataset for profiling."""
    rng = np.random.default_rng(42)

    env = Environment(environment_name="profiling")
    position_range = (0.0, 100.0)
    env.place_bin_size = (position_range[1] - position_range[0]) / n_positions
    env.fit_place_grid(np.linspace(*position_range, n_positions)[:, None])

    t_pos = np.linspace(0, 10.0, n_encoding)
    pos = np.linspace(*position_range, n_encoding)[:, None]

    enc_spike_times = [np.sort(rng.uniform(0, 10.0, 20))]
    enc_spike_features = [rng.normal(0, 1, (20, 2))]

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=enc_spike_times,
        spike_waveform_features=enc_spike_features,
        environment=env,
        sampling_frequency=500,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    dec_features = rng.normal(0, 1, (n_decoding, 2))
    enc_features = enc_spike_features[0]
    waveform_std = jnp.asarray(encoding["waveform_std"])
    occupancy = encoding["occupancy"]
    mean_rate = encoding["mean_rates"][0]

    is_track_interior = env.is_track_interior_.ravel()
    interior_place_bin_centers = env.place_bin_centers_[is_track_interior]

    position_distance = kde_distance(
        interior_place_bin_centers,
        encoding["encoding_positions"][0],
        std=encoding["position_std"],
    )

    return {
        "decoding_features": jnp.asarray(dec_features),
        "encoding_features": jnp.asarray(enc_features),
        "waveform_std": waveform_std,
        "occupancy": occupancy,
        "mean_rate": mean_rate,
        "position_distance": position_distance,
    }


def profile_with_compilation_time():
    """Profile showing compilation vs execution time."""
    print("\n" + "="*60)
    print("COMPILATION VS EXECUTION TIME")
    print("="*60)

    data = create_test_data(n_encoding=100, n_decoding=50, n_positions=200)

    # Time first call (includes compilation)
    import time
    start = time.perf_counter()
    result1 = estimate_log_joint_mark_intensity(
        data["decoding_features"],
        data["encoding_features"],
        data["waveform_std"],
        data["occupancy"],
        data["mean_rate"],
        data["position_distance"],
        use_gemm=True,
    )
    result1.block_until_ready()
    first_call = time.perf_counter() - start

    # Time second call (no compilation)
    start = time.perf_counter()
    result2 = estimate_log_joint_mark_intensity(
        data["decoding_features"],
        data["encoding_features"],
        data["waveform_std"],
        data["occupancy"],
        data["mean_rate"],
        data["position_distance"],
        use_gemm=True,
    )
    result2.block_until_ready()
    second_call = time.perf_counter() - start

    print(f"\nFirst call (with compilation):  {first_call:.4f} s")
    print(f"Second call (cached):           {second_call:.4f} s")
    print(f"Compilation overhead:           {first_call - second_call:.4f} s")
    print(f"Speedup after compilation:      {first_call / second_call:.1f}x")


def profile_memory_usage():
    """Profile memory allocation patterns."""
    print("\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)

    # Test different dataset sizes
    sizes = [
        ("Small", 50, 25, 100),
        ("Medium", 200, 100, 500),
        ("Large", 500, 250, 1000),
    ]

    print(f"\n{'Size':<10} {'Enc':>6} {'Dec':>6} {'Pos':>6} {'Memory (MB)':>12}")
    print("-" * 50)

    for name, n_enc, n_dec, n_pos in sizes:
        data = create_test_data(n_encoding=n_enc, n_decoding=n_dec, n_positions=n_pos)

        # Trigger computation
        result = estimate_log_joint_mark_intensity(
            data["decoding_features"],
            data["encoding_features"],
            data["waveform_std"],
            data["occupancy"],
            data["mean_rate"],
            data["position_distance"],
            use_gemm=True,
        )
        result.block_until_ready()

        # Estimate memory from array sizes
        total_bytes = 0
        for key, val in data.items():
            if isinstance(val, (jnp.ndarray, np.ndarray)):
                total_bytes += val.nbytes
        total_bytes += result.nbytes

        print(f"{name:<10} {n_enc:>6} {n_dec:>6} {n_pos:>6} {total_bytes / 1024**2:>12.2f}")


def profile_optimization_strategies():
    """Compare different optimization strategies."""
    print("\n" + "="*60)
    print("OPTIMIZATION STRATEGY COMPARISON")
    print("="*60)

    data = create_test_data(n_encoding=200, n_decoding=100, n_positions=500)

    import time

    def time_variant(name, **kwargs):
        # Warmup
        for _ in range(2):
            result = estimate_log_joint_mark_intensity(
                data["decoding_features"],
                data["encoding_features"],
                data["waveform_std"],
                data["occupancy"],
                data["mean_rate"],
                data["position_distance"],
                **kwargs,
            )
            result.block_until_ready()

        # Time
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = estimate_log_joint_mark_intensity(
                data["decoding_features"],
                data["encoding_features"],
                data["waveform_std"],
                data["occupancy"],
                data["mean_rate"],
                data["position_distance"],
                **kwargs,
            )
            result.block_until_ready()
            times.append(time.perf_counter() - start)

        mean_time = np.mean(times)
        print(f"{name:<30} {mean_time:.6f} s")
        return mean_time

    print(f"\n{'Strategy':<30} {'Time':>12}")
    print("-" * 45)

    baseline = time_variant("Linear-space (use_gemm=False)", use_gemm=False)
    gemm = time_variant("GEMM optimization (default)", use_gemm=True)
    gemm_tiled = time_variant("GEMM + tiling", use_gemm=True, pos_tile_size=100)

    print(f"\nSpeedup vs baseline:")
    print(f"  GEMM:        {baseline / gemm:.2f}x")
    print(f"  GEMM+tiling: {baseline / gemm_tiled:.2f}x")


def analyze_compilation_cache():
    """Show how JAX caches compiled functions."""
    print("\n" + "="*60)
    print("COMPILATION CACHING ANALYSIS")
    print("="*60)

    data = create_test_data(n_encoding=100, n_decoding=50, n_positions=200)

    import time

    def time_call(label):
        start = time.perf_counter()
        result = estimate_log_joint_mark_intensity(
            data["decoding_features"],
            data["encoding_features"],
            data["waveform_std"],
            data["occupancy"],
            data["mean_rate"],
            data["position_distance"],
            use_gemm=True,
        )
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.6f} s")
        return elapsed

    print("\nSequential calls (same arguments):")
    t1 = time_call("  Call 1 (compiles)")
    t2 = time_call("  Call 2 (cached)")
    t3 = time_call("  Call 3 (cached)")

    print(f"\nCaching benefit: {t1 / t2:.1f}x faster after compilation")

    # Different input shapes trigger recompilation
    print("\nCalls with different input shapes:")
    data_small = create_test_data(n_encoding=50, n_decoding=25, n_positions=100)
    t4 = time_call("  Small input (recompiles)")
    t5 = time_call("  Small input again (cached)")

    print(f"\nNote: Changing input shapes triggers recompilation")
    print(f"      Keep consistent shapes for best performance!")


def main():
    """Run all profiling analyses."""
    print("\n" + "="*60)
    print("JAX PROFILING FOR CLUSTERLESS KDE")
    print("="*60)
    print(f"\nJAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    try:
        profile_with_compilation_time()
        profile_memory_usage()
        profile_optimization_strategies()
        analyze_compilation_cache()

        print("\n" + "="*60)
        print("PROFILING TIPS")
        print("="*60)
        print("""
1. JIT Compilation:
   - First call includes compilation time (~100-1000ms)
   - Subsequent calls are fast (~1-10ms)
   - Keep input shapes consistent to avoid recompilation

2. Memory Optimization:
   - Use block_size to control peak memory
   - Use pos_tile_size for very large position grids (>2000 bins)
   - GEMM optimization reduces intermediate arrays

3. Performance:
   - use_gemm=True: Best for multi-dimensional features (2D+ waveforms)
   - use_gemm=False: Simpler, matches reference exactly
   - Position tiling: Reduces memory at slight speed cost

4. GPU vs CPU:
   - GPU: Best for large datasets (>500 positions, >16 electrodes)
   - CPU: Fine for small-medium datasets
   - Set via: jax.config.update('jax_platform_name', 'gpu')

5. Advanced Profiling:
   - Use JAX profiler: jax.profiler.start_trace() / stop_trace()
   - View with TensorBoard
   - Analyze HLO with: jax.jit(func).lower(...).as_text()
        """)

    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
