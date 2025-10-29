"""Profile clusterless KDE performance across different waveform feature dimensions.

This script tests the hypothesis that GEMM optimization provides better performance
for higher-dimensional waveform features (4D, 8D, 10D) compared to 2D.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    fit_clusterless_kde_encoding_model as fit_log,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    predict_clusterless_kde_log_likelihood as predict_log,
)


def create_test_data(n_features, n_encoding=200, n_decoding=100, n_positions=500):
    """Create synthetic test data with specified number of features."""
    rng = np.random.default_rng(0)

    # Encoding data
    enc_spike_times = [
        np.sort(rng.uniform(0, 10, 50)) for _ in range(4)
    ]
    enc_spike_features = [
        rng.normal(0, 1, (len(times), n_features)) for times in enc_spike_times
    ]

    # Decoding data
    dec_spike_times = [
        np.sort(rng.uniform(0, 5, 25)) for _ in range(4)
    ]
    dec_spike_features = [
        rng.normal(0, 1, (len(times), n_features)) for times in dec_spike_times
    ]

    # Position data
    position_time = np.linspace(0, 10, n_encoding)
    position = np.linspace(0, 100, n_encoding)[:, None]

    # Decoding time
    time = np.linspace(0, 5, n_decoding)

    # Create environment - simple 1D linear track
    env = Environment(
        environment_name="test_track",
        place_bin_size=1.0,
        position_range=[(0.0, 100.0)],
    )
    # Fit environment to position data
    env.fit(position)

    return {
        "environment": env,
        "encoding": {
            "position_time": position_time,
            "position": position,
            "spike_times": enc_spike_times,
            "spike_waveform_features": enc_spike_features,
        },
        "decoding": {
            "time": time,
            "spike_times": dec_spike_times,
            "spike_waveform_features": dec_spike_features,
        },
    }


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


def profile_dimension(n_features):
    """Profile performance for a specific feature dimension."""
    print(f"\n{'='*70}")
    print(f"Testing {n_features}D waveform features")
    print(f"{'='*70}")

    data = create_test_data(n_features)

    # Common parameters
    position_std = np.sqrt(12.5)
    waveform_std = 24.0

    # Reference implementation
    print("\nReference (linear-space):")
    enc_ref = fit_clusterless_kde_encoding_model(
        position=data["encoding"]["position"],
        position_time=data["encoding"]["position_time"],
        spike_times=data["encoding"]["spike_times"],
        spike_waveform_features=data["encoding"]["spike_waveform_features"],
        environment=data["environment"],
        position_std=position_std,
        waveform_std=waveform_std,
        disable_progress_bar=True,
    )

    mean_ref, std_ref = time_function(
        predict_clusterless_kde_log_likelihood,
        time=data["decoding"]["time"],
        position=data["encoding"]["position"],
        position_time=data["encoding"]["position_time"],
        spike_times=data["decoding"]["spike_times"],
        spike_waveform_features=data["decoding"]["spike_waveform_features"],
        occupancy=enc_ref["occupancy"],
        mean_rate=enc_ref["mean_rate"],
        encoding_spike_times=enc_ref["encoding_spike_times"],
        encoding_spike_waveform_features=enc_ref["encoding_spike_waveform_features"],
        position_std=position_std,
        waveform_std=waveform_std,
    )
    print(f"  Prediction time: {mean_ref*1000:.2f} ± {std_ref*1000:.2f} ms")

    # Log-space with GEMM (vmap)
    print("\nLog-space with GEMM (vmap):")
    enc_log = fit_log(
        position=data["encoding"]["position"],
        position_time=data["encoding"]["position_time"],
        spike_times=data["encoding"]["spike_times"],
        spike_waveform_features=data["encoding"]["spike_waveform_features"],
        environment=data["environment"],
        position_std=position_std,
        waveform_std=waveform_std,
        disable_progress_bar=True,
    )

    mean_gemm, std_gemm = time_function(
        predict_log,
        time=data["decoding"]["time"],
        position=data["encoding"]["position"],
        position_time=data["encoding"]["position_time"],
        spike_times=data["decoding"]["spike_times"],
        spike_waveform_features=data["decoding"]["spike_waveform_features"],
        occupancy=enc_log["occupancy"],
        mean_rate=enc_log["mean_rate"],
        encoding_spike_times=enc_log["encoding_spike_times"],
        encoding_spike_waveform_features=enc_log["encoding_spike_waveform_features"],
        position_std=position_std,
        waveform_std=waveform_std,
        use_gemm=True,
    )
    print(f"  Prediction time: {mean_gemm*1000:.2f} ± {std_gemm*1000:.2f} ms")

    # Log-space without GEMM (linear fallback)
    print("\nLog-space without GEMM (linear fallback in log-space):")
    mean_no_gemm, std_no_gemm = time_function(
        predict_log,
        time=data["decoding"]["time"],
        position=data["encoding"]["position"],
        position_time=data["encoding"]["position_time"],
        spike_times=data["decoding"]["spike_times"],
        spike_waveform_features=data["decoding"]["spike_waveform_features"],
        occupancy=enc_log["occupancy"],
        mean_rate=enc_log["mean_rate"],
        encoding_spike_times=enc_log["encoding_spike_times"],
        encoding_spike_waveform_features=enc_log["encoding_spike_waveform_features"],
        position_std=position_std,
        waveform_std=waveform_std,
        use_gemm=False,
    )
    print(f"  Prediction time: {mean_no_gemm*1000:.2f} ± {std_no_gemm*1000:.2f} ms")

    # Speedup analysis
    speedup_gemm = mean_ref / mean_gemm
    speedup_no_gemm = mean_ref / mean_no_gemm

    print("\nSpeedup vs reference:")
    print(f"  GEMM (vmap):     {speedup_gemm:.2f}x")
    print(f"  No GEMM (log):   {speedup_no_gemm:.2f}x")

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
    }


def main():
    """Run profiling across multiple feature dimensions."""
    print("="*70)
    print("FEATURE DIMENSION PROFILING")
    print("="*70)
    print("\nTesting hypothesis: GEMM optimization performs better")
    print("with higher-dimensional waveform features.")
    print("\nTest configuration:")
    print("  - Encoding: 200 time points, 4 electrodes, ~200 spikes")
    print("  - Decoding: 100 time bins, 4 electrodes, ~100 spikes")
    print("  - Position bins: 200")
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
    print("\nPrediction time (ms):")
    print(f"{'Dim':<6} {'Reference':<15} {'GEMM (vmap)':<15} {'No GEMM':<15}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['n_features']:<6} "
            f"{r['ref_time']*1000:>6.2f} ± {r['ref_std']*1000:>4.2f}   "
            f"{r['gemm_time']*1000:>6.2f} ± {r['gemm_std']*1000:>4.2f}   "
            f"{r['no_gemm_time']*1000:>6.2f} ± {r['no_gemm_std']*1000:>4.2f}"
        )

    print("\nSpeedup vs Reference:")
    print(f"{'Dim':<6} {'GEMM (vmap)':<15} {'No GEMM':<15}")
    print("-" * 70)
    for r in results:
        gemm_marker = "✓" if r['speedup_gemm'] > 1.0 else "✗"
        no_gemm_marker = "✓" if r['speedup_no_gemm'] > 1.0 else "✗"
        print(
            f"{r['n_features']:<6} "
            f"{r['speedup_gemm']:>6.2f}x {gemm_marker:<8} "
            f"{r['speedup_no_gemm']:>6.2f}x {no_gemm_marker:<8}"
        )

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find crossover point for GEMM
    gemm_wins = [r for r in results if r['speedup_gemm'] > 1.0]
    if gemm_wins:
        min_dim = min(r['n_features'] for r in gemm_wins)
        print(f"\n✓ GEMM (vmap) is faster than reference for {min_dim}D+ features")
    else:
        print("\n✗ GEMM (vmap) is not faster than reference for any tested dimension")

    # Find best configuration per dimension
    print("\nBest configuration per dimension:")
    for r in results:
        if r['gemm_time'] < r['ref_time'] and r['gemm_time'] < r['no_gemm_time']:
            best = f"GEMM (vmap) - {r['speedup_gemm']:.2f}x faster"
        elif r['no_gemm_time'] < r['ref_time'] and r['no_gemm_time'] < r['gemm_time']:
            best = f"No GEMM (log) - {r['speedup_no_gemm']:.2f}x faster"
        else:
            best = "Reference (linear)"
        print(f"  {r['n_features']}D: {best}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
