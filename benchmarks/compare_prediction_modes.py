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
reference64
    Benchmark-only float64 HMM inference with the same fitted model and
    float32 likelihood values as an x64-disabled run. Requires ``JAX_ENABLE_X64=1``;
    records source and promoted likelihood dtypes. Native compact stays float32.

Example::

    uv run python benchmarks/compare_prediction_modes.py --family sorted \\
        --mode compact --duration 60 --output /tmp/compare/sorted-compact-60
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# tqdm reads this when non_local_detector is imported inside main().
os.environ.setdefault("TQDM_DISABLE", "1")

SAMPLE_RATE = 500  # decode bins per second


@contextmanager
def float32_computation() -> Iterator[None]:
    """Use the normal float32 path across supported JAX versions, then restore."""
    import jax

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def output_dtypes(result: Any) -> dict[str, str]:
    """Record array precision independently of JAX's global x64 setting."""
    return {name: str(result[name].dtype) for name in sorted(result.data_vars)}


def _prepare_reference_likelihood(model: Any, predict: dict) -> Callable:
    """Bind the same global-edge preparation used by native prediction."""
    from non_local_detector.models.base import _prepare_likelihood_callback
    from non_local_detector.time_edges import _DecodeTimeGrid

    edges = predict["time_edges"]
    args = (predict["position_time"], predict["position"], predict["spike_times"])
    if "spike_waveform_features" in predict:
        args += (predict["spike_waveform_features"],)
    prepared = _prepare_likelihood_callback(
        model.compute_log_likelihood,
        edges,
        has_no_spike=any(obs.is_no_spike for obs in model.observation_models),
        log_likelihood_args=args,
        _time_grid=_DecodeTimeGrid.from_validated_edges(
            edges, uniform_width=(edges[-1] - edges[0]) / (len(edges) - 1)
        ),
    )
    centers = (edges[:-1] + edges[1:]) / 2

    def likelihood(
        full_edges: np.ndarray, *, row_slice: slice, is_missing: np.ndarray
    ) -> Any:
        return prepared(centers, *args, row_slice=row_slice, is_missing=is_missing)

    return likelihood


def reference64_prediction(
    model: Any,
    predict: dict,
    *,
    chunk_size: int,
    checkpoint_dir: Path,
) -> tuple[Any, dict[str, list[str]]]:
    """Run a bounded float64 reference without changing native prediction's API.

    Keep fitted model values fixed and evaluate likelihoods with x64 disabled,
    then promote inputs before inference. This isolates HMM precision from
    fitting and likelihood precision; it is not an end-to-end float64 oracle.
    """
    import jax

    from non_local_detector.checkpointed_inference import checkpointed_forward_backward
    from non_local_detector.models.base import _missing_bins

    if not jax.config.x64_enabled:
        raise ValueError("reference64 requires JAX_ENABLE_X64=1")
    interior = model.is_track_interior_state_bins_
    operator = model._continuous_transition_operator_.restricted(interior)
    operator = operator.bind_discrete(model.discrete_state_transitions_).fused()
    initial = np.asarray(model.initial_conditions_[interior], dtype=np.float64)

    def promote(leaf: Any) -> Any:
        array = np.asarray(leaf)
        return array.astype(np.float64) if array.dtype.kind == "f" else leaf

    operator = jax.tree_util.tree_map(promote, operator)
    transition_dtypes = sorted(
        {
            str(np.asarray(leaf).dtype)
            for leaf in jax.tree_util.tree_leaves(operator)
            if np.asarray(leaf).dtype.kind == "f"
        }
    )
    if transition_dtypes != ["float64"]:
        raise ValueError("reference64 transition inputs must actually be float64")
    prepared = _prepare_reference_likelihood(model, predict)
    observed_dtypes: set[str] = set()
    inference_dtypes: set[str] = set()

    def likelihood(
        full_edges: np.ndarray, *, row_slice: slice, is_missing: np.ndarray
    ) -> Any:
        with float32_computation():
            values = prepared(full_edges, row_slice=row_slice, is_missing=is_missing)
        observed_dtypes.add(str(values.dtype))
        promoted = values.astype(np.float64)
        inference_dtypes.add(str(promoted.dtype))
        if promoted.dtype != np.float64:
            raise ValueError("reference64 likelihood input must actually be float64")
        return promoted

    result = checkpointed_forward_backward(
        predict["time_edges"],
        initial,
        likelihood,
        transition_operator=operator,
        state_ind=model.state_ind_[interior],
        n_states=len(model.state_names),
        is_missing=_missing_bins(
            predict["time_edges"],
            predict.get("is_missing"),
            predict["position_time"],
            predict["position"],
        ),
        chunk_size=chunk_size,
        checkpoint_dir=checkpoint_dir,
        dtype=np.float64,
    ).dataset
    if set(output_dtypes(result).values()) != {"float64"}:
        raise ValueError("reference64 posterior must actually be float64")
    return result, {
        "likelihood_compute_dtypes": sorted(observed_dtypes),
        "likelihood_inference_dtypes": sorted(inference_dtypes),
        "transition_inference_dtypes": transition_dtypes,
    }


def workload(args):
    """Encoding and decoding inputs shared with benchmark_native_pipeline.py.

    ``rate_spread`` (default 0) gives units log-spaced rates spanning
    ``2**rate_spread`` around ``spike_rate``, so they differ in spike counts.
    """
    spread = getattr(args, "rate_spread", 0.0)
    unit_rates = (
        args.spike_rate * 2.0 ** (spread * np.linspace(-0.5, 0.5, args.population))
        if spread
        else np.full(args.population, args.spike_rate)
    )
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
                0, args.encoding_duration, int(args.encoding_duration * rate)
            )
        )
        for rate in unit_rates
    ]
    decode_spikes = [
        np.sort(decoding_rng.uniform(0, args.duration, int(args.duration * rate)))
        for rate in unit_rates
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


def set_population_defaults(args):
    """Fill unset population and spike rate: 64 units at 5 Hz or 8 electrodes at 20 Hz."""
    if args.population is None:
        args.population = 64 if args.family == "sorted" else 8
    if args.spike_rate is None:
        args.spike_rate = 5.0 if args.family == "sorted" else 20.0


def input_fingerprint(fit: dict, predict: dict) -> str:
    """Hash actual seeded inputs before adding mode-specific keyword arguments."""

    def describe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        return value

    return hashlib.sha256(
        json.dumps(describe([fit, predict]), sort_keys=True).encode()
    ).hexdigest()


def provenance(script):
    """Hashes, precision and device that identify what a report measured.

    Call after configuring x64 so the recorded setting is the one measured.
    """
    import jax

    import non_local_detector

    root = Path(non_local_detector.__file__).resolve().parent
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if "tests" not in path.relative_to(root).parts
    }
    return {
        "package_source_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
        "x64": jax.config.x64_enabled,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "device_kind": jax.local_devices()[0].device_kind,
    }


def device_peak_bytes(jax):
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument(
        "--mode",
        choices=["dense", "chunked", "compact", "reference64"],
        default="dense",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--encoding-duration", type=float, default=10.0)
    parser.add_argument("--arena", type=float, default=180.0, help="cm")
    parser.add_argument("--bin-size", type=float, default=2.0, help="cm")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--spike-rate", type=float, default=None, help="Hz")
    parser.add_argument(
        "--algorithm",
        default=None,
        help="likelihood algorithm (default: the detector default)",
    )
    parser.add_argument(
        "--rate-spread",
        type=float,
        default=0.0,
        help="log2 range of per-unit rates around --spike-rate (0: equal rates)",
    )
    parser.add_argument("--mark-dimensions", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=15_000)
    parser.add_argument(
        "--checkpoint-chunk-size",
        type=int,
        default=None,
        help="rows per checkpoint chunk (default: the detector default)",
    )
    parser.add_argument("--repeat", type=int, default=2, help="warm predictions")
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_population_defaults(args)

    import jax

    import non_local_detector
    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    backend = jax.default_backend()
    if args.mode == "reference64" and not jax.config.x64_enabled:
        parser.error("reference64 requires JAX_ENABLE_X64=1")
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    args.output.mkdir(parents=True, exist_ok=True)
    fit_kwargs, predict_kwargs = workload(args)
    input_sha256 = input_fingerprint(fit_kwargs, predict_kwargs)
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
        **(
            {}
            if args.algorithm is None
            else {
                (
                    "sorted_spikes_algorithm"
                    if args.family == "sorted"
                    else "clusterless_algorithm"
                ): args.algorithm
            }
        ),
    )
    predict_parameters = inspect.signature(model.predict).parameters
    n_rows = len(predict_kwargs["time_edges"]) - 1
    if args.mode == "chunked":
        predict_kwargs["n_chunks"] = max(1, int(np.ceil(n_rows / args.chunk_rows)))
    elif args.mode in {"compact", "reference64"}:
        if "inference_mode" not in predict_parameters:
            parser.error("this checkout has no checkpointed inference (compact mode)")
        fit_kwargs["transition_representation"] = "structured"
        if args.mode == "compact":
            predict_kwargs.update(
                inference_mode="checkpointed",
                output_mode="compact",
                checkpoint_dir=args.output / "checkpoints",
            )
            if args.checkpoint_chunk_size is not None:
                predict_kwargs["chunk_size"] = args.checkpoint_chunk_size

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
        "rate_spread": args.rate_spread,
        "algorithm": args.algorithm,
        "encoding_duration_s": args.encoding_duration,
        "input_sha256": input_sha256,
        "fit_x64": False if args.mode == "reference64" else jax.config.x64_enabled,
        "chunk_rows": args.chunk_rows if args.mode == "chunked" else None,
        "checkpoint_chunk_size": (
            args.checkpoint_chunk_size
            if args.mode in {"compact", "reference64"}
            else None
        ),
        "backend": backend,
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
        **provenance(__file__),
        "package_path": str(Path(non_local_detector.__file__).parent),
        "host": platform.node(),
        "status": "started",
    }
    try:
        start = time.perf_counter()
        if args.mode == "reference64":
            with float32_computation():
                model.fit(**fit_kwargs)
        else:
            model.fit(**fit_kwargs)
        report["fit_seconds"] = time.perf_counter() - start
        report["state_bins"] = int(model.state_ind_.shape[0])
        if (
            args.mode in {"compact", "reference64"}
            and args.checkpoint_chunk_size is None
        ):
            try:
                from non_local_detector.checkpointed_inference import (
                    default_chunk_size,
                )
            except ImportError:  # older checkouts default to 256 rows
                report["checkpoint_chunk_size"] = 256
            else:
                report["checkpoint_chunk_size"] = default_chunk_size(
                    int(np.count_nonzero(model.is_track_interior_state_bins_)),
                    np.float64 if args.mode == "reference64" else np.float32,
                )

        def predict_once() -> Any:
            if args.mode == "reference64":
                dataset, dtypes = reference64_prediction(
                    model,
                    predict_kwargs,
                    chunk_size=report["checkpoint_chunk_size"],
                    checkpoint_dir=args.output / "checkpoints",
                )
                report.update(dtypes)
                return dataset
            return model.predict(**predict_kwargs)

        start = time.perf_counter()
        result = predict_once()
        states = np.asarray(result["acausal_state_probabilities"])
        report["compile_and_first_predict_seconds"] = time.perf_counter() - start
        warm = []
        for number in range(args.repeat):
            profile = args.profile_dir is not None and number == args.repeat - 1
            if profile:
                jax.profiler.start_trace(str(args.profile_dir))
            start = time.perf_counter()
            result = predict_once()
            states = np.asarray(result["acausal_state_probabilities"])
            warm.append(time.perf_counter() - start)
            if profile:
                jax.profiler.stop_trace()
        report["warm_predict_seconds"] = warm
        report["seconds_per_recording_second"] = (
            float(np.median(warm)) / args.duration if warm else None
        )
        report["output_variables"] = sorted(result.data_vars)
        report["output_dtypes"] = output_dtypes(result)
        report["reference_scope"] = (
            "float64 HMM inference; fixed fit with JAX x64 disabled, float32 likelihood values"
            if args.mode == "reference64"
            else None
        )
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
