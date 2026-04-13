"""Benchmark KDE clusterless decoding with raw vs localized spike features.

Generates simulated data from a polymer probe (4 shanks x 32 channels) and
compares decoding accuracy, timing, and memory for different feature
extraction strategies:

- **Raw**: All 32 channels as mark features (baseline)
- **Mode A**: Position only — (x_est, z_est, peak_amp) = 3 features
- **Mode B**: Local amplitudes only — peak + neighbors = 3 features
- **Mode C**: Combined — position + local amps = 5 features

Also runs a dimensionality sweep on a single-shank probe to demonstrate
the curse of dimensionality.

Usage:
    uv run python scripts/benchmark_localization.py
"""

import time as time_module
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.simulate.dense_probe_simulation import (
    make_dense_probe_run_data,
    make_probe_run_data,
)
from non_local_detector.simulate.probe_geometry import polymer_probe_config
from non_local_detector.simulate.spike_localization import localize_spikes


def decode_and_evaluate(
    sim,
    waveform_std,
    label,
    sampling_frequency=500,
    position_std=None,
):
    """Fit encoding model, predict, and evaluate decoding accuracy.

    Returns dict with accuracy metrics and timing.
    """
    if position_std is None:
        position_std = np.sqrt(12.5)

    # Split data: first 60% for encoding, last 40% for decoding
    n_time = len(sim.position_time)
    split = int(n_time * 0.6)
    t_split = sim.position_time[split]

    # Encoding data: spikes in first 60%
    enc_spike_times = []
    enc_features = []
    for times, feats in zip(sim.spike_times, sim.spike_waveform_features, strict=True):
        mask = times <= t_split
        enc_spike_times.append(times[mask])
        enc_features.append(feats[mask])

    # Decoding data: spikes in last 40%
    dec_spike_times = []
    dec_features = []
    for times, feats in zip(sim.spike_times, sim.spike_waveform_features, strict=True):
        mask = times > t_split
        dec_spike_times.append(times[mask])
        dec_features.append(feats[mask])

    enc_position_time = sim.position_time[:split]
    enc_position = sim.position[:split]

    # Decoding time bins
    dec_edges = sim.edges[sim.edges > t_split]
    if len(dec_edges) < 2:
        return {"label": label, "error": "Not enough decoding time bins"}

    n_features = enc_features[0].shape[1] if enc_features[0].shape[0] > 0 else 0
    total_enc_spikes = sum(t.size for t in enc_spike_times)
    total_dec_spikes = sum(t.size for t in dec_spike_times)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Features per spike: {n_features}")
    print(f"  Encoding spikes: {total_enc_spikes}")
    print(f"  Decoding spikes: {total_dec_spikes}")
    print(f"{'=' * 60}")

    # Fit
    t0 = time_module.perf_counter()
    encoding = fit_clusterless_kde_encoding_model(
        position_time=jnp.asarray(enc_position_time),
        position=jnp.asarray(enc_position),
        spike_times=[jnp.asarray(t) for t in enc_spike_times],
        spike_waveform_features=[jnp.asarray(f) for f in enc_features],
        environment=sim.environment,
        sampling_frequency=sampling_frequency,
        position_std=position_std,
        waveform_std=waveform_std,
        block_size=100,
        disable_progress_bar=True,
    )
    fit_time = time_module.perf_counter() - t0

    # Predict
    t0 = time_module.perf_counter()
    log_likelihood = predict_clusterless_kde_log_likelihood(
        time=jnp.asarray(dec_edges),
        position_time=jnp.asarray(enc_position_time),
        position=jnp.asarray(enc_position),
        spike_times=[jnp.asarray(t) for t in dec_spike_times],
        spike_waveform_features=[jnp.asarray(f) for f in dec_features],
        is_local=False,
        **encoding,
    )
    predict_time = time_module.perf_counter() - t0

    # Evaluate: decode most likely position at each time bin
    log_likelihood = np.asarray(log_likelihood)
    is_track_interior = sim.environment.is_track_interior_.ravel()
    interior_bins = sim.environment.place_bin_centers_[is_track_interior]

    # Decoded position = argmax over position bins (1D track)
    decoded_pos = interior_bins[np.argmax(log_likelihood, axis=1)].ravel()

    # True position at decoding time bin centers
    dec_bin_centers = (dec_edges[:-1] + dec_edges[1:]) / 2.0
    true_pos = np.interp(dec_bin_centers, sim.position_time, sim.position[:, 0])

    # Align lengths (log_likelihood has n_time rows from dec_edges)
    min_len = min(len(decoded_pos), len(true_pos))
    decoded_pos = decoded_pos[:min_len]
    true_pos = true_pos[:min_len]

    # Only evaluate bins where we have valid likelihood
    valid = np.all(np.isfinite(log_likelihood[:min_len]), axis=1)
    if valid.sum() == 0:
        return {
            "label": label,
            "n_features": n_features,
            "error": "No valid decoding bins",
        }

    decoded_pos = decoded_pos[valid]
    true_pos = true_pos[valid]

    # Metrics
    mae = np.mean(np.abs(decoded_pos - true_pos))
    corr = np.corrcoef(decoded_pos, true_pos)[0, 1] if min_len > 2 else np.nan
    median_error = np.median(np.abs(decoded_pos - true_pos))

    result = {
        "label": label,
        "n_features": n_features,
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "total_time_s": fit_time + predict_time,
        "mae": mae,
        "median_error": median_error,
        "correlation": corr,
        "n_valid_bins": int(valid.sum()),
        "total_enc_spikes": total_enc_spikes,
        "total_dec_spikes": total_dec_spikes,
    }

    print(f"  Fit time:      {fit_time:.2f}s")
    print(f"  Predict time:  {predict_time:.2f}s")
    print(f"  MAE:           {mae:.2f}")
    print(f"  Median error:  {median_error:.2f}")
    print(f"  Correlation:   {corr:.3f}")

    return result


def run_polymer_probe_benchmark():
    """Compare raw vs localized features on polymer probe simulation."""
    print("\n" + "#" * 60)
    print("  POLYMER PROBE BENCHMARK")
    print("  4 shanks x 32 channels, 35 um spacing")
    print("#" * 60)

    cfg = polymer_probe_config()
    place_field_means = np.arange(0, 175, 8)  # ~22 neurons
    common_kwargs = {
        "sampling_frequency": 500,
        "n_runs": 8,
        "place_field_means": place_field_means,
        "seed": 42,
    }

    # Generate raw data (no transform)
    print("\nGenerating raw simulation data...")
    sim_raw = make_probe_run_data(cfg, **common_kwargs)

    # Generate localized data for each mode
    transforms = {
        "Mode A (position only)": partial(
            localize_spikes,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=False,
        ),
        "Mode B (local amps)": partial(
            localize_spikes,
            n_neighbors=1,
            include_position=False,
            include_local_amplitudes=True,
        ),
        "Mode C (combined)": partial(
            localize_spikes,
            n_neighbors=1,
            include_position=True,
            include_local_amplitudes=True,
        ),
    }

    results = []

    # Raw baseline
    result = decode_and_evaluate(
        sim_raw,
        waveform_std=24.0,
        label="Raw (32 channels)",
        sampling_frequency=common_kwargs["sampling_frequency"],
    )
    results.append(result)

    # Localized modes
    for mode_name, transform in transforms.items():
        print(f"\nGenerating {mode_name} data...")
        sim = make_probe_run_data(cfg, feature_transform=transform, **common_kwargs)

        # Adjust waveform_std based on feature types
        n_feat = sim.spike_waveform_features[0].shape[1]
        # Use a broader bandwidth for position features, narrower for amplitudes
        if "position only" in mode_name:
            waveform_std = np.array([20.0, 40.0, 15.0])
        elif "local amps" in mode_name:
            waveform_std = np.array([15.0] * n_feat)
        else:  # combined
            waveform_std = np.array([20.0, 40.0] + [15.0] * (n_feat - 2))

        result = decode_and_evaluate(
            sim,
            waveform_std=waveform_std,
            label=mode_name,
            sampling_frequency=common_kwargs["sampling_frequency"],
        )
        results.append(result)

    return results


def run_dimensionality_sweep():
    """Demonstrate curse of dimensionality on single-shank probe."""
    print("\n" + "#" * 60)
    print("  DIMENSIONALITY SWEEP")
    print("  Single shank, varying number of active channels")
    print("#" * 60)

    dimensions = [4, 8, 16, 32]
    place_field_means = np.arange(0, 175, 8)

    results = []
    for n_dim in dimensions:
        print(f"\nGenerating data with {n_dim} channels...")
        sim = make_dense_probe_run_data(
            n_channels=64,
            n_active_channels=n_dim,
            vertical_spacing=35.0,
            n_columns=1,
            decay_constant=20.0,
            sampling_frequency=500,
            n_runs=8,
            place_field_means=place_field_means,
            seed=42,
        )

        result = decode_and_evaluate(
            sim,
            waveform_std=24.0,
            label=f"{n_dim} channels",
            sampling_frequency=500,
        )
        results.append(result)

    return results


def plot_results(polymer_results, sweep_results):
    """Plot benchmark results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -- Polymer probe comparison --
    valid = [r for r in polymer_results if "error" not in r]
    if valid:
        labels = [r["label"] for r in valid]
        x = range(len(labels))

        ax = axes[0]
        ax.bar(x, [r["correlation"] for r in valid])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Position correlation")
        ax.set_title("Decoding accuracy\n(polymer probe)")
        ax.set_ylim(0, 1)

        ax = axes[1]
        ax.bar(x, [r["total_time_s"] for r in valid])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Time (s)")
        ax.set_title("Compute time\n(polymer probe)")

    # -- Dimensionality sweep --
    valid_sweep = [r for r in sweep_results if "error" not in r]
    if valid_sweep:
        ax = axes[2]
        dims = [r["n_features"] for r in valid_sweep]
        corrs = [r["correlation"] for r in valid_sweep]
        times = [r["total_time_s"] for r in valid_sweep]

        ax.plot(dims, corrs, "o-", color="tab:blue", label="Correlation")
        ax.set_xlabel("Number of mark features")
        ax.set_ylabel("Position correlation", color="tab:blue")
        ax.set_title("Curse of dimensionality\n(single shank)")
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        ax2.plot(dims, times, "s--", color="tab:red", label="Time")
        ax2.set_ylabel("Compute time (s)", color="tab:red")

    plt.tight_layout()
    plt.savefig("benchmark_localization.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to benchmark_localization.png")
    plt.show()


def print_summary(polymer_results, sweep_results):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_results = polymer_results + sweep_results
    valid = [r for r in all_results if "error" not in r]

    if not valid:
        print("  No valid results to summarize.")
        return

    print(f"  {'Label':<25} {'Features':>8} {'Corr':>8} {'MAE':>8} {'Time(s)':>8}")
    print("-" * 70)
    for r in valid:
        print(
            f"  {r['label']:<25} {r['n_features']:>8} "
            f"{r['correlation']:>8.3f} {r['mae']:>8.1f} "
            f"{r['total_time_s']:>8.2f}"
        )


if __name__ == "__main__":
    polymer_results = run_polymer_probe_benchmark()
    sweep_results = run_dimensionality_sweep()
    print_summary(polymer_results, sweep_results)

    try:
        plot_results(polymer_results, sweep_results)
    except Exception as e:
        print(f"\nPlotting failed (no display?): {e}")
        print("Results printed above.")
