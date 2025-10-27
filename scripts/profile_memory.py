#!/usr/bin/env python
"""Memory profiling for clusterless KDE implementations.

This script uses memory_profiler to track memory usage over time.

Installation:
    pip install memory_profiler matplotlib

Usage:
    # Basic usage
    python scripts/profile_memory.py

    # With line-by-line profiling
    python -m memory_profiler scripts/profile_memory.py

    # Generate memory usage plot
    mprof run scripts/profile_memory.py
    mprof plot
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model as fit_reference,
)
from non_local_detector.likelihoods.clusterless_kde import (
    predict_clusterless_kde_log_likelihood as predict_reference,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    fit_clusterless_kde_encoding_model as fit_log,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    predict_clusterless_kde_log_likelihood as predict_log,
)

try:
    from memory_profiler import profile as memory_profile
except ImportError:
    print("Warning: memory_profiler not installed")
    print("Install with: pip install memory_profiler")
    print("Running without memory profiling...")

    def memory_profile(func):
        return func


def create_test_data():
    """Create moderate-size test dataset."""
    rng = np.random.default_rng(42)

    env = Environment(environment_name="profiling")
    position_range = (0.0, 100.0)
    n_position_bins = 500
    env.place_bin_size = (position_range[1] - position_range[0]) / n_position_bins
    env.fit_place_grid(np.linspace(*position_range, n_position_bins)[:, None])

    # Encoding
    n_time_encoding = 2000
    n_electrodes = 16
    t_pos = np.linspace(0, 100.0, n_time_encoding)
    pos = np.linspace(*position_range, n_time_encoding)[:, None]

    enc_spike_times = []
    enc_spike_features = []
    for _ in range(n_electrodes):
        n_spikes = rng.poisson(200)
        spike_times = np.sort(rng.uniform(0, 100.0, n_spikes))
        spike_features = rng.normal(0, 1, (n_spikes, 2))
        enc_spike_times.append(spike_times)
        enc_spike_features.append(spike_features)

    # Decoding
    n_time_decoding = 500
    t_edges = np.linspace(0, 20.0, n_time_decoding)
    t_pos_dec = np.linspace(0, 20.0, n_time_decoding)
    pos_dec = np.linspace(*position_range, n_time_decoding)[:, None]

    dec_spike_times = []
    dec_spike_features = []
    for _ in range(n_electrodes):
        n_spikes = rng.poisson(100)
        spike_times = np.sort(rng.uniform(0, 20.0, n_spikes))
        spike_features = rng.normal(0, 1, (n_spikes, 2))
        dec_spike_times.append(spike_times)
        dec_spike_features.append(spike_features)

    return {
        "environment": env,
        "encoding": {
            "position_time": t_pos,
            "position": pos,
            "spike_times": enc_spike_times,
            "spike_waveform_features": enc_spike_features,
        },
        "decoding": {
            "time": t_edges,
            "position_time": t_pos_dec,
            "position": pos_dec,
            "spike_times": dec_spike_times,
            "spike_waveform_features": dec_spike_features,
        },
    }


@memory_profile
def profile_reference_implementation():
    """Profile reference implementation memory usage."""
    print("\nProfiling REFERENCE implementation...")
    data = create_test_data()

    # Encoding
    encoding = fit_reference(
        position_time=data["encoding"]["position_time"],
        position=data["encoding"]["position"],
        spike_times=data["encoding"]["spike_times"],
        spike_waveform_features=data["encoding"]["spike_waveform_features"],
        environment=data["environment"],
        sampling_frequency=500,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Decoding
    import jax.numpy as jnp
    log_likelihood = predict_reference(
        time=data["decoding"]["time"],
        position_time=data["decoding"]["position_time"],
        position=data["decoding"]["position"],
        spike_times=data["decoding"]["spike_times"],
        spike_waveform_features=data["decoding"]["spike_waveform_features"],
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        environment=data["environment"],
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=jnp.asarray(encoding["waveform_std"]),
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    return log_likelihood


@memory_profile
def profile_log_implementation():
    """Profile log-space implementation memory usage."""
    print("\nProfiling LOG-SPACE implementation...")
    data = create_test_data()

    # Encoding
    encoding = fit_log(
        position_time=data["encoding"]["position_time"],
        position=data["encoding"]["position"],
        spike_times=data["encoding"]["spike_times"],
        spike_waveform_features=data["encoding"]["spike_waveform_features"],
        environment=data["environment"],
        sampling_frequency=500,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Decoding
    import jax.numpy as jnp
    log_likelihood = predict_log(
        time=data["decoding"]["time"],
        position_time=data["decoding"]["position_time"],
        position=data["decoding"]["position"],
        spike_times=data["decoding"]["spike_times"],
        spike_waveform_features=data["decoding"]["spike_waveform_features"],
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        environment=data["environment"],
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=jnp.asarray(encoding["waveform_std"]),
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    return log_likelihood


def estimate_memory_requirements(n_electrodes, n_encoding_spikes, n_decoding_spikes, n_positions):
    """Estimate memory requirements for a given dataset size.

    Parameters
    ----------
    n_electrodes : int
        Number of electrodes
    n_encoding_spikes : int
        Total encoding spikes across all electrodes
    n_decoding_spikes : int
        Total decoding spikes across all electrodes
    n_positions : int
        Number of position bins

    Returns
    -------
    dict
        Estimated memory usage in MB for different components
    """
    bytes_per_float = 4  # float32

    # Encoding model
    occupancy = n_positions * bytes_per_float
    position_distance = n_encoding_spikes * n_positions * bytes_per_float  # per electrode
    encoding_features = n_encoding_spikes * 2 * bytes_per_float  # 2D features

    # Decoding (worst case: all in memory at once)
    mark_kernel = n_encoding_spikes * n_decoding_spikes * bytes_per_float  # per electrode
    joint_intensity = n_decoding_spikes * n_positions * bytes_per_float  # per electrode

    # Total per electrode (dominant terms)
    per_electrode = position_distance + mark_kernel + joint_intensity

    # Total
    total_bytes = occupancy + (per_electrode * n_electrodes) + encoding_features

    return {
        "occupancy_mb": occupancy / 1024**2,
        "per_electrode_mb": per_electrode / 1024**2,
        "total_mb": total_bytes / 1024**2,
        "breakdown": {
            "position_distance_mb": position_distance / 1024**2,
            "mark_kernel_mb": mark_kernel / 1024**2,
            "joint_intensity_mb": joint_intensity / 1024**2,
        },
    }


def main():
    """Run memory profiling."""
    print("="*60)
    print("MEMORY PROFILING FOR CLUSTERLESS KDE")
    print("="*60)

    # Estimate memory requirements
    print("\nMemory Requirements Estimation:")
    print("-" * 60)

    configs = [
        ("Small", 8, 800, 400, 200),
        ("Medium", 16, 3200, 1600, 500),
        ("Large", 32, 16000, 6400, 1000),
        ("Realistic", 64, 64000, 25600, 2000),
    ]

    print(f"{'Config':<12} {'Elec':>5} {'EncSpk':>7} {'DecSpk':>7} {'Pos':>5} {'Total (MB)':>12}")
    print("-" * 60)

    for name, n_elec, n_enc, n_dec, n_pos in configs:
        est = estimate_memory_requirements(n_elec, n_enc, n_dec, n_pos)
        print(f"{name:<12} {n_elec:>5} {n_enc:>7} {n_dec:>7} {n_pos:>5} {est['total_mb']:>12.1f}")

    print("\nDetailed breakdown for 'Medium' config:")
    est = estimate_memory_requirements(16, 3200, 1600, 500)
    print(f"  Occupancy:         {est['occupancy_mb']:.2f} MB")
    print(f"  Per electrode:     {est['per_electrode_mb']:.2f} MB")
    print(f"    - Position dist: {est['breakdown']['position_distance_mb']:.2f} MB")
    print(f"    - Mark kernel:   {est['breakdown']['mark_kernel_mb']:.2f} MB")
    print(f"    - Joint intens:  {est['breakdown']['joint_intensity_mb']:.2f} MB")
    print(f"  Total (16 elec):   {est['total_mb']:.2f} MB")

    # Run profiling
    print("\n" + "="*60)
    print("Running memory profiling...")
    print("="*60)

    try:
        ll_ref = profile_reference_implementation()
        ll_log = profile_log_implementation()

        print("\nVerifying numerical agreement...")
        diff = np.abs(np.asarray(ll_ref) - np.asarray(ll_log))
        print(f"Max difference: {np.max(diff):.2e}")
        print(f"Results match: {np.allclose(ll_ref, ll_log, rtol=1e-4, atol=1e-5)}")

    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("MEMORY OPTIMIZATION TIPS")
    print("="*60)
    print("""
1. Block Processing:
   - Use block_size parameter to limit peak memory
   - Trade-off: smaller blocks = less memory, slightly slower

2. Position Tiling (log-space only):
   - Use pos_tile_size for very large position grids (>2000 bins)
   - Processes positions in chunks, reduces n_enc * n_pos arrays

3. Data Types:
   - JAX uses float32 by default (4 bytes per float)
   - Can use float64 for higher precision (8 bytes)

4. Memory Profiling Tools:
   - memory_profiler: pip install memory_profiler
   - Run with: python -m memory_profiler script.py
   - Plot with: mprof run script.py && mprof plot

5. JAX Memory:
   - JAX pre-allocates GPU memory
   - Set XLA_PYTHON_CLIENT_PREALLOCATE=false to disable
   - Monitor with: jax.profiler.device_memory_profile()

6. For Large Datasets:
   - Process in batches (encode once, decode in chunks)
   - Use sparse representations where appropriate
   - Consider downsampling position grid if high-resolution not needed
    """)

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
