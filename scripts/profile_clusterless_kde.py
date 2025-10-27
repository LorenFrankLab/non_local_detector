#!/usr/bin/env python
"""Profile clusterless KDE implementations for performance comparison.

This script profiles both the reference (linear-space) and optimized (log-space)
implementations to compare:
- Execution time
- Memory usage
- JAX compilation time
- GPU vs CPU performance

Usage:
    python scripts/profile_clusterless_kde.py --help
    python scripts/profile_clusterless_kde.py --size medium
    python scripts/profile_clusterless_kde.py --size large --device gpu
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
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


@dataclass
class DatasetSize:
    """Configuration for synthetic dataset size."""

    name: str
    n_time_encoding: int  # Number of position samples for encoding
    n_time_decoding: int  # Number of time bins for decoding
    n_electrodes: int  # Number of electrodes
    n_encoding_spikes_per_electrode: int  # Average spikes per electrode during encoding
    n_decoding_spikes_per_electrode: int  # Average spikes per electrode during decoding
    n_position_bins: int  # Number of place bins (grid size)


DATASET_SIZES = {
    "tiny": DatasetSize(
        name="tiny",
        n_time_encoding=100,
        n_time_decoding=50,
        n_electrodes=4,
        n_encoding_spikes_per_electrode=20,
        n_decoding_spikes_per_electrode=10,
        n_position_bins=100,
    ),
    "small": DatasetSize(
        name="small",
        n_time_encoding=500,
        n_time_decoding=100,
        n_electrodes=8,
        n_encoding_spikes_per_electrode=50,
        n_decoding_spikes_per_electrode=30,
        n_position_bins=200,
    ),
    "medium": DatasetSize(
        name="medium",
        n_time_encoding=2000,
        n_time_decoding=500,
        n_electrodes=16,
        n_encoding_spikes_per_electrode=200,
        n_decoding_spikes_per_electrode=100,
        n_position_bins=500,
    ),
    "large": DatasetSize(
        name="large",
        n_time_encoding=5000,
        n_time_decoding=1000,
        n_electrodes=32,
        n_encoding_spikes_per_electrode=500,
        n_decoding_spikes_per_electrode=300,
        n_position_bins=1000,
    ),
    "realistic": DatasetSize(
        name="realistic",
        n_time_encoding=10000,  # ~10 minutes at 20 Hz
        n_time_decoding=2000,  # ~2 minutes at 20 Hz
        n_electrodes=64,  # Typical tetrode array
        n_encoding_spikes_per_electrode=1000,
        n_decoding_spikes_per_electrode=400,
        n_position_bins=2000,  # High-resolution spatial grid
    ),
}


def create_synthetic_data(size: DatasetSize, seed: int = 42):
    """Create synthetic data for profiling.

    Parameters
    ----------
    size : DatasetSize
        Configuration for dataset size
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with all data needed for profiling
    """
    rng = np.random.default_rng(seed)

    # Create environment
    env = Environment(environment_name="profiling")
    position_range = (0.0, 100.0)
    env.place_bin_size = (position_range[1] - position_range[0]) / size.n_position_bins
    env.fit_place_grid(
        np.linspace(*position_range, size.n_position_bins)[:, None]
    )

    # Encoding data
    encoding_duration = 100.0  # seconds
    t_pos_encoding = np.linspace(0, encoding_duration, size.n_time_encoding)
    pos_encoding = np.linspace(*position_range, size.n_time_encoding)[:, None]

    enc_spike_times = []
    enc_spike_features = []
    for _ in range(size.n_electrodes):
        n_spikes = int(
            rng.poisson(size.n_encoding_spikes_per_electrode)
        )
        spike_times = np.sort(rng.uniform(0, encoding_duration, n_spikes))
        # 2D waveform features
        spike_features = rng.normal(0, 1, (n_spikes, 2))
        enc_spike_times.append(spike_times)
        enc_spike_features.append(spike_features)

    # Decoding data
    decoding_duration = 20.0  # seconds
    t_edges_decoding = np.linspace(0, decoding_duration, size.n_time_decoding)
    t_pos_decoding = np.linspace(0, decoding_duration, size.n_time_decoding)
    pos_decoding = np.linspace(*position_range, size.n_time_decoding)[:, None]

    dec_spike_times = []
    dec_spike_features = []
    for _ in range(size.n_electrodes):
        n_spikes = int(
            rng.poisson(size.n_decoding_spikes_per_electrode)
        )
        spike_times = np.sort(rng.uniform(0, decoding_duration, n_spikes))
        spike_features = rng.normal(0, 1, (n_spikes, 2))
        dec_spike_times.append(spike_times)
        dec_spike_features.append(spike_features)

    return {
        "environment": env,
        "encoding": {
            "position_time": t_pos_encoding,
            "position": pos_encoding,
            "spike_times": enc_spike_times,
            "spike_waveform_features": enc_spike_features,
        },
        "decoding": {
            "time": t_edges_decoding,
            "position_time": t_pos_decoding,
            "position": pos_decoding,
            "spike_times": dec_spike_times,
            "spike_waveform_features": dec_spike_features,
        },
        "params": {
            "sampling_frequency": 500,
            "position_std": np.sqrt(12.5),
            "waveform_std": 24.0,
            "block_size": 100,
        },
    }


def get_memory_usage_mb():
    """Get current JAX device memory usage in MB."""
    try:
        import jax.profiler as profiler
        memory_stats = profiler.device_memory_profile()
        # Sum across all devices
        total_bytes = sum(stats.bytes_in_use for stats in memory_stats.values())
        return total_bytes / (1024**2)
    except Exception:
        # Fallback if profiler not available
        return None


def time_function(func, *args, n_warmup=2, n_runs=5, **kwargs):
    """Time a function with warmup and multiple runs.

    Parameters
    ----------
    func : callable
        Function to time
    *args, **kwargs
        Arguments to pass to function
    n_warmup : int
        Number of warmup runs (for JIT compilation)
    n_runs : int
        Number of timed runs

    Returns
    -------
    dict
        Timing statistics
    """
    # Warmup (JIT compilation)
    for _ in range(n_warmup):
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()  # Wait for JAX computation

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "times": times,
    }


def profile_encoding(data, implementation: Literal["reference", "log"]):
    """Profile encoding model fitting.

    Parameters
    ----------
    data : dict
        Dataset from create_synthetic_data
    implementation : str
        Which implementation to use

    Returns
    -------
    tuple
        (encoding_model, timing_stats)
    """
    fit_func = fit_reference if implementation == "reference" else fit_log

    def fit_wrapper():
        return fit_func(
            position_time=data["encoding"]["position_time"],
            position=data["encoding"]["position"],
            spike_times=data["encoding"]["spike_times"],
            spike_waveform_features=data["encoding"]["spike_waveform_features"],
            environment=data["environment"],
            sampling_frequency=data["params"]["sampling_frequency"],
            position_std=data["params"]["position_std"],
            waveform_std=data["params"]["waveform_std"],
            block_size=data["params"]["block_size"],
            disable_progress_bar=True,
        )

    print(f"\n  Fitting encoding model ({implementation})...")
    timing = time_function(fit_wrapper, n_warmup=1, n_runs=3)
    encoding = fit_wrapper()

    return encoding, timing


def profile_decoding(data, encoding, implementation: Literal["reference", "log", "log_gemm"]):
    """Profile decoding (likelihood prediction).

    Parameters
    ----------
    data : dict
        Dataset from create_synthetic_data
    encoding : dict
        Fitted encoding model
    implementation : str
        Which implementation to use

    Returns
    -------
    tuple
        (log_likelihood, timing_stats)
    """
    if implementation == "reference":
        predict_func = predict_reference
    else:
        predict_func = predict_log

    def predict_wrapper():
        return predict_func(
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
            block_size=data["params"]["block_size"],
            disable_progress_bar=True,
        )

    print(f"\n  Predicting log likelihood ({implementation})...")
    timing = time_function(predict_wrapper, n_warmup=2, n_runs=5)
    log_likelihood = predict_wrapper()

    return log_likelihood, timing


def print_comparison(reference_stats, log_stats, name=""):
    """Print comparison of timing statistics."""
    speedup = reference_stats["mean"] / log_stats["mean"]
    print(f"\n{name} Comparison:")
    print(f"  Reference: {reference_stats['mean']:.4f} ± {reference_stats['std']:.4f} s")
    print(f"  Log-space: {log_stats['mean']:.4f} ± {log_stats['std']:.4f} s")
    print(f"  Speedup:   {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Profile clusterless KDE implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with tiny dataset
  python scripts/profile_clusterless_kde.py --size tiny

  # Medium dataset with CPU
  python scripts/profile_clusterless_kde.py --size medium

  # Large realistic dataset
  python scripts/profile_clusterless_kde.py --size realistic

  # Use GPU if available
  python scripts/profile_clusterless_kde.py --size medium --device gpu
        """,
    )
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=list(DATASET_SIZES.keys()),
        help="Dataset size to use for profiling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use (cpu or gpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Configure JAX
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    print(f"\nJAX devices: {jax.devices()}")

    # Create dataset
    size = DATASET_SIZES[args.size]
    print(f"\n{'='*60}")
    print(f"Profiling Configuration: {size.name.upper()}")
    print(f"{'='*60}")
    print(f"  Encoding: {size.n_time_encoding} time points, {size.n_electrodes} electrodes")
    print(f"  Encoding spikes: ~{size.n_encoding_spikes_per_electrode * size.n_electrodes} total")
    print(f"  Decoding: {size.n_time_decoding} time bins")
    print(f"  Decoding spikes: ~{size.n_decoding_spikes_per_electrode * size.n_electrodes} total")
    print(f"  Position bins: {size.n_position_bins}")

    print("\nCreating synthetic dataset...")
    data = create_synthetic_data(size, seed=args.seed)

    # Profile Reference Implementation
    print(f"\n{'='*60}")
    print("REFERENCE IMPLEMENTATION (linear-space)")
    print(f"{'='*60}")
    enc_ref, enc_ref_timing = profile_encoding(data, "reference")
    ll_ref, ll_ref_timing = profile_decoding(data, enc_ref, "reference")

    # Profile Log-Space Implementation
    print(f"\n{'='*60}")
    print("LOG-SPACE IMPLEMENTATION (optimized)")
    print(f"{'='*60}")
    enc_log, enc_log_timing = profile_encoding(data, "log")
    ll_log, ll_log_timing = profile_decoding(data, enc_log, "log")

    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print_comparison(enc_ref_timing, enc_log_timing, "Encoding")
    print_comparison(ll_ref_timing, ll_log_timing, "Decoding")

    total_ref = enc_ref_timing["mean"] + ll_ref_timing["mean"]
    total_log = enc_log_timing["mean"] + ll_log_timing["mean"]
    print("\nTotal Time:")
    print(f"  Reference: {total_ref:.4f} s")
    print(f"  Log-space: {total_log:.4f} s")
    print(f"  Speedup:   {total_ref / total_log:.2f}x")

    # Verify numerical agreement
    print("\nNumerical Agreement:")
    diff = np.abs(np.asarray(ll_ref) - np.asarray(ll_log))
    print(f"  Max difference: {np.max(diff):.2e}")
    print(f"  Mean difference: {np.mean(diff):.2e}")
    print(f"  Within tolerance: {np.allclose(ll_ref, ll_log, rtol=1e-4, atol=1e-5)}")

    # Memory usage (if available)
    mem_usage = get_memory_usage_mb()
    if mem_usage is not None:
        print(f"\nMemory Usage: {mem_usage:.1f} MB")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
