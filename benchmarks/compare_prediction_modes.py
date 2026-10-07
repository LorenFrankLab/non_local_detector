"""Compare detector prediction speed, memory and capacity across versions and modes.

Runs one fit and prediction of a simulated 2-D recording in this process and
writes a JSON report: fit time, compile-plus-first and warm prediction times,
peak host RSS, peak device memory and the acausal state probabilities (saved
as .npy for cross-version comparison). Run each configuration in its own
process so memory peaks are not shared. To benchmark another checkout, put its
``src`` first on ``PYTHONPATH``; modes it lacks are refused rather than faked.

Modes
-----
dense
    ``predict`` with defaults: full spatial posterior and state probabilities.
chunked
    ``predict(n_chunks=...)``: likelihoods computed in chunks of about
    ``--chunk-rows`` rows; outputs as in ``dense``.
compact
    ``predict(inference_mode="checkpointed", output_mode="compact")`` with
    structured transitions: state probabilities only, bounded working memory.

Example::

    uv run python benchmarks/compare_prediction_modes.py --family sorted \\
        --mode compact --duration 60 --output /tmp/compare/sorted-compact-60
"""

import argparse
import inspect
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil

SAMPLE_RATE = 500  # decode bins per second


def workload(args):
    """Encoding and decoding inputs shared with benchmark_native_pipeline.py."""
    encoding_rng = np.random.default_rng(7341)
    decoding_rng = np.random.default_rng(7342)
    tracking_time = np.arange(int(args.encoding_duration * 30) + 1) / 30
    position = np.column_stack(
        (
            args.arena * (0.5 + 0.4 * np.sin(tracking_time)),
            args.arena * (0.5 + 0.4 * np.cos(tracking_time * 0.71)),
        )
    )
    training_spikes = [
        np.sort(
            encoding_rng.uniform(
                0, args.encoding_duration, int(args.encoding_duration * args.spike_rate)
            )
        )
        for _ in range(args.population)
    ]
    decode_spikes = [
        np.sort(
            decoding_rng.uniform(0, args.duration, int(args.duration * args.spike_rate))
        )
        for _ in range(args.population)
    ]
    decode_time = np.arange(int(args.duration * 30) + 1) / 30
    decode_position = np.column_stack(
        (
            args.arena * (0.5 + 0.4 * np.sin(decode_time)),
            args.arena * (0.5 + 0.4 * np.cos(decode_time * 0.71)),
        )
    )
    fit = {
        "position_time": tracking_time,
        "position": position,
        "spike_times": training_spikes,
        "encoding_time_range": [0, args.encoding_duration],
    }
    predict = {
        "spike_times": decode_spikes,
        "position_time": decode_time,
        "position": decode_position,
        "time_edges": np.arange(int(round(args.duration * SAMPLE_RATE)) + 1)
        / SAMPLE_RATE,
    }
    if args.family == "clusterless":
        fit["spike_waveform_features"] = [
            encoding_rng.normal(size=(len(s), args.mark_dimensions))
            for s in training_spikes
        ]
        predict["spike_waveform_features"] = [
            decoding_rng.normal(size=(len(s), args.mark_dimensions))
            for s in decode_spikes
        ]
    return fit, predict


def device_peak_bytes(jax):
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument(
        "--mode", choices=["dense", "chunked", "compact"], default="dense"
    )
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--encoding-duration", type=float, default=10.0)
    parser.add_argument("--arena", type=float, default=180.0, help="cm")
    parser.add_argument("--bin-size", type=float, default=2.0, help="cm")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--spike-rate", type=float, default=None, help="Hz")
    parser.add_argument("--mark-dimensions", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=15_000)
    parser.add_argument("--checkpoint-chunk-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=2, help="warm predictions")
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.population is None:
        args.population = 64 if args.family == "sorted" else 8
    if args.spike_rate is None:
        args.spike_rate = 5.0 if args.family == "sorted" else 20.0

    import jax

    import non_local_detector
    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    args.output.mkdir(parents=True, exist_ok=True)
    fit_kwargs, predict_kwargs = workload(args)
    detector = (
        NonLocalSortedSpikesDetector
        if args.family == "sorted"
        else NonLocalClusterlessDetector
    )
    model = detector(
        environments=Environment(
            place_bin_size=args.bin_size,
            position_range=((0, args.arena), (0, args.arena)),
        ),
        infer_track_interior=False,
    )
    predict_parameters = inspect.signature(model.predict).parameters
    n_rows = len(predict_kwargs["time_edges"]) - 1
    if args.mode == "chunked":
        predict_kwargs["n_chunks"] = max(1, int(np.ceil(n_rows / args.chunk_rows)))
    elif args.mode == "compact":
        if "inference_mode" not in predict_parameters:
            parser.error("this checkout has no checkpointed inference (compact mode)")
        fit_kwargs["transition_representation"] = "structured"
        predict_kwargs.update(
            inference_mode="checkpointed",
            output_mode="compact",
            chunk_size=args.checkpoint_chunk_size,
            checkpoint_dir=args.output / "checkpoints",
        )

    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.02):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    report = {
        "family": args.family,
        "mode": args.mode,
        "duration_s": args.duration,
        "rows": n_rows,
        "arena_cm": args.arena,
        "bin_size_cm": args.bin_size,
        "population": args.population,
        "spike_rate_hz": args.spike_rate,
        "encoding_duration_s": args.encoding_duration,
        "backend": backend,
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
        "x64": jax.config.x64_enabled,
        "package_path": str(Path(non_local_detector.__file__).parent),
        "host": platform.node(),
        "status": "started",
    }
    try:
        start = time.perf_counter()
        model.fit(**fit_kwargs)
        report["fit_seconds"] = time.perf_counter() - start
        report["state_bins"] = int(model.state_ind_.shape[0])
        start = time.perf_counter()
        result = model.predict(**predict_kwargs)
        np.asarray(result["acausal_state_probabilities"])
        report["compile_and_first_predict_seconds"] = time.perf_counter() - start
        warm = []
        for number in range(args.repeat):
            profile = args.profile_dir is not None and number == args.repeat - 1
            if profile:
                jax.profiler.start_trace(str(args.profile_dir))
            start = time.perf_counter()
            result = model.predict(**predict_kwargs)
            states = np.asarray(result["acausal_state_probabilities"])
            warm.append(time.perf_counter() - start)
            if profile:
                jax.profiler.stop_trace()
        report["warm_predict_seconds"] = warm
        report["seconds_per_recording_second"] = (
            float(np.median(warm)) / args.duration if warm else None
        )
        report["output_variables"] = sorted(result.data_vars)
        report["output_bytes"] = int(
            sum(np.asarray(result[name]).nbytes for name in result.data_vars)
        )
        np.save(args.output / "acausal_state_probabilities.npy", states)
        report["status"] = "ok"
    except Exception as error:  # record the failure mode (e.g. out of memory)
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:400]}"
    finally:
        stop.set()
        sampler.join()
        report["peak_host_rss_bytes"] = peak_rss[0]
        report["peak_device_bytes"] = (
            device_peak_bytes(jax) if backend != "cpu" else None
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "package_path"}))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
