"""Time a fixed likelihood chunk on 60-second and one-hour recordings.

Usage: uv run python scripts/benchmark_chunk_likelihood_runtime.py /tmp/timings
Use --compare-ordering --spikes-per-second 100 for a paired comparison of
prepared ordering against checking each spike train on every chunk call.

For a baseline comparison, extract the baseline's src/ into a separate directory
and prepend the extracted src directory to PYTHONPATH. The script supports predictors both
before and after row_slice was introduced. It saves timings and likelihood
arrays for numerical comparison. No spike falls in the interval that the old
chunk-local clipping discarded, so both versions do the same scientific work.

CPU/GPU synchronization is included, fitting/JIT compilation excluded. Ordinary
measurements have three warmups and fifteen timed calls; paired measurements
have three warmups and thirty timed calls per path. No-Spike's full-timeline
median and per-unit spike ordering, when supported, are prepared once as in
detector._predict; their separate preparation costs are reported. These are
likelihood-only timings on a 50-bin track, not full HMM timings or a
production-sized spatial grid.
"""

import argparse
import inspect
import json
import statistics
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    common,
)
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood

BACKENDS = (
    "sorted_spikes_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
    "no_spike",
)
N_UNITS = 8


def measure(callback):
    for _ in range(3):
        jax.block_until_ready(callback())
    elapsed = []
    for _ in range(15):
        start = time.perf_counter()
        result = jax.block_until_ready(callback())
        elapsed.append((time.perf_counter() - start) * 1000)
    return np.asarray(result), {
        "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
    }


def measure_ordering_pair(prepared):
    """Alternate identical calls with/without ordering reuse in one process."""
    unprepared = partial(
        prepared.func,
        *prepared.args,
        **{
            key: value
            for key, value in prepared.keywords.items()
            if key != "_spike_time_order"
        },
    )
    callbacks = (unprepared, prepared)
    for _ in range(3):
        for callback in callbacks:
            jax.block_until_ready(callback())
    elapsed = [[], []]
    results = [None, None]
    for iteration in range(30):
        for index in (0, 1) if iteration % 2 == 0 else (1, 0):
            start = time.perf_counter()
            results[index] = jax.block_until_ready(callbacks[index]())
            elapsed[index].append((time.perf_counter() - start) * 1000)
    np.testing.assert_array_equal(np.asarray(results[0]), np.asarray(results[1]))
    return np.asarray(results[1]), {
        "median_ms": statistics.median(elapsed[1]),
        "min_ms": min(elapsed[1]),
        "max_ms": max(elapsed[1]),
        "unprepared_median_ms": statistics.median(elapsed[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spikes-per-second", type=int, default=20)
    parser.add_argument("--compare-ordering", action="store_true")
    options = parser.parse_args()
    if options.spikes_per_second <= 0:
        parser.error("--spikes-per-second must be positive")
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"source": common.__file__, "devices": str(jax.devices())}))
    rng = np.random.default_rng(411)
    position_time = np.linspace(0, 20, 2001)
    position = (
        50
        + 40 * np.sin(2 * np.pi * position_time / 20)
        + rng.normal(0, 0.15, len(position_time))
    )[:, None]
    env = Environment(
        environment_name="line", place_bin_size=2.0, position_range=((0.0, 100.0),)
    ).fit_place_grid(position=position, infer_track_interior=False)
    encoding_times = [np.sort(rng.uniform(0, 20, 300)) for _ in range(N_UNITS)]
    encoding_features = [rng.normal(20, 5, (300, 4)) for _ in range(N_UNITS)]
    recordings = {}
    for duration in (60, 3600):
        timeline = np.arange(round(duration / 0.002), dtype=np.float64) * 0.002
        interval = 1.0 / options.spikes_per_second
        unit_times = np.arange(interval / 2, duration, interval)
        # Keep the comparison valid even against the pre-row-slice baseline,
        # which drops spikes between the last chunk timestamp and the next row.
        unit_times = unit_times[
            (unit_times <= timeline[499]) | (unit_times >= timeline[500])
        ]
        spikes = [unit_times.copy() for _ in range(N_UNITS)]
        features = [
            np.random.default_rng(900 + unit).normal(20, 5, (len(unit_times), 4))
            for unit in range(N_UNITS)
        ]
        pt = np.linspace(0, duration, 50 * duration + 1)
        pos = (50 + 40 * np.sin(2 * np.pi * pt / 20))[:, None]
        recordings[duration] = (timeline, spikes, features, pt, pos)

    rows = []
    for name in BACKENDS:
        clusterless = name.startswith("clusterless")
        if name == "no_spike":
            predict = predict_no_spike_log_likelihood
            encoding = {}
        else:
            registry = (
                _CLUSTERLESS_ALGORITHMS if clusterless else _SORTED_SPIKES_ALGORITHMS
            )
            fit, predict = registry[name]
            fit_kwargs = {
                "position_time": position_time,
                "position": position,
                "spike_times": encoding_times,
                "environment": env,
                "disable_progress_bar": True,
            }
            if clusterless:
                fit_kwargs["spike_waveform_features"] = encoding_features
            if name == "clusterless_gmm":
                fit_kwargs.update(
                    gmm_components_occupancy=4,
                    gmm_components_gpi=4,
                    gmm_components_joint=8,
                    gmm_reg_covar=1e-3,
                )
            encoding = fit(**fit_kwargs)
        parameters = inspect.signature(predict).parameters
        for duration, (timeline, spikes, features, pt, pos) in recordings.items():
            row_slice = slice(0, 500)
            row_aware = "row_slice" in parameters
            tt = timeline if row_aware else timeline[row_slice]
            extra = {"row_slice": row_slice} if row_aware else {}
            preparation_ms = 0.0
            spike_order_preparation_ms = 0.0
            if name == "no_spike" and "_time_bin_size" in parameters:
                start = time.perf_counter()
                extra["_time_bin_size"] = np.median(np.diff(timeline))
                preparation_ms = (time.perf_counter() - start) * 1000
            if "_spike_time_order" in parameters:
                start = time.perf_counter()
                spike_order = common._SpikeTimeOrder()
                for unit_times in spikes:
                    spike_order.get(unit_times)
                extra["_spike_time_order"] = spike_order
                spike_order_preparation_ms = (time.perf_counter() - start) * 1000
            for local in (False,) if name == "no_spike" else (False, True):
                if name == "no_spike":
                    callback = partial(predict, tt, spikes, **extra)
                else:
                    args = (tt, pt, pos, spikes)
                    if clusterless:
                        args = (*args, features)
                    callback = partial(
                        predict, *args, **encoding, is_local=local, **extra
                    )
                if options.compare_ordering and "_spike_time_order" in extra:
                    result, timing = measure_ordering_pair(callback)
                else:
                    result, timing = measure(callback)
                row = dict(
                    backend=name,
                    duration_s=duration,
                    spikes_per_second=options.spikes_per_second,
                    local=local,
                    shape=result.shape,
                    preparation_ms=preparation_ms,
                    spike_order_preparation_ms=spike_order_preparation_ms,
                    **timing,
                )
                print(json.dumps(row), flush=True)
                rows.append(row)
                np.save(output / f"{name}_{duration}_{local}.npy", result)
    (output / "timings.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
