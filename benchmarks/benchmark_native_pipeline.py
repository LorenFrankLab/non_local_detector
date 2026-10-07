"""Synchronized native detector benchmark with explicit resource budgets.

Run small dense references before checkpointed/structured scale-up. CUDA runs
must request --platform gpu, which refuses silent CPU fallback. Spatial output
capacity is preflighted; the default production output is compact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--arena", type=float, default=10)
    parser.add_argument("--bin-size", type=float, default=2)
    parser.add_argument("--duration", type=float, default=1)
    parser.add_argument("--encoding-duration", type=float, default=10)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--mode", choices=["dense", "compact", "spatial"], default="dense"
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--host-budget-gib", type=float, default=8)
    parser.add_argument("--device-budget-gib", type=float)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--spike-rate", type=float, default=5)
    parser.add_argument("--mark-dimensions", type=int, default=2)
    parser.add_argument("--encoding-block-size", type=int)
    parser.add_argument("--position-block-size", type=int)
    parser.add_argument("--warm-duration", type=float)
    parser.add_argument("--selected-intervals", type=json.loads)
    parser.add_argument("--cleanup-spatial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-reference", action="store_true")
    args = parser.parse_args()
    if (
        not all(
            np.isfinite(value)
            for value in (
                args.arena,
                args.bin_size,
                args.duration,
                args.encoding_duration,
                args.spike_rate,
                args.host_budget_gib,
            )
        )
        or args.arena <= 0
        or args.bin_size <= 0
        or args.repeat < 1
        or args.duration <= 0
        or args.encoding_duration <= 0
        or args.population < 1
        or args.spike_rate < 0
        or args.mark_dimensions < 1
        or args.chunk_size < 1
        or args.host_budget_gib <= 0
        or (
            args.warm_duration is not None
            and (
                not np.isfinite(args.warm_duration)
                or not 0 < args.warm_duration <= args.duration
            )
        )
    ):
        parser.error("invalid duration, population, chunk size, or resource budget")
    if (args.encoding_block_size is None) != (args.position_block_size is None):
        parser.error("provide both encoding and position block sizes")
    if args.encoding_block_size is not None and (
        args.family != "clusterless"
        or args.encoding_block_size < 1
        or args.position_block_size < 1
    ):
        parser.error("positive encoding/position tile sizes apply to clusterless KDE")
    # Initialize the requested backend after parsing. Budgeted benchmarking
    # should not implicitly reserve 75% of a large CUDA card before measuring.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    import non_local_detector
    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    try:
        devices = jax.devices(args.platform)
    except RuntimeError as exc:
        parser.error(f"requested {args.platform} backend unavailable: {exc}")
    if not devices:
        parser.error(f"requested {args.platform} device unavailable")
    if args.platform == "gpu" and "NVIDIA" not in devices[0].device_kind.upper():
        parser.error("CUDA qualification requires an NVIDIA device")
    if args.device_budget_gib is not None and (
        args.platform != "gpu"
        or not np.isfinite(args.device_budget_gib)
        or args.device_budget_gib <= 0
    ):
        parser.error("a positive device budget applies only to CUDA runs")
    args.output.mkdir(parents=True, exist_ok=True)
    padded_side = int(np.ceil(args.arena / args.bin_size)) + 2
    upper_hidden = 2 * padded_side**2 + 2
    n_rows = int(round(args.duration * 500))
    budget = int(args.host_budget_gib * 1024**3)
    # Dense setup and inference retain several host/device buffers. Do not
    # allocate a large dense target merely to measure its failure.
    dense_estimate = 6 * upper_hidden**2 * 8 + 6 * n_rows * upper_hidden * 4
    if args.mode == "dense" and dense_estimate > budget:
        parser.error(
            f"dense allocation estimate {dense_estimate} exceeds budget {budget}"
        )
    source = Path(non_local_detector.__file__).resolve().parent
    benchmark_hash_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hashes_before = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
        if "tests" not in path.parts
    }
    if args.mode != "dense":
        # Screen major workspaces before allocation; actual RSS/allocator peaks
        # remain the qualification. Sorted KDE has no E x P decode kernel.
        encoding_count = max(1, int(args.encoding_duration * args.spike_rate))
        tracking_count = int(args.encoding_duration * 30) + 1
        if args.encoding_block_size is not None:
            encoding_workspace = (
                8
                * min(args.encoding_block_size, max(encoding_count, tracking_count))
                * min(args.position_block_size, upper_hidden)
                * 4
            )
        elif args.family == "clusterless":
            encoding_workspace = (
                4 * args.population * encoding_count * upper_hidden * 4
                + 4 * tracking_count * min(10_000, upper_hidden) * 4
            )
        else:
            encoding_workspace = (
                4 * max(encoding_count, tracking_count) * min(10_000, upper_hidden) * 4
            )
        working_estimate = (
            12 * args.chunk_size * upper_hidden * 4
            + encoding_workspace
            + args.population * encoding_count * (args.mark_dimensions + 4) * 8
            + n_rows * 256
            + 1024**3
        )
        if working_estimate > budget:
            parser.error(
                f"structured working estimate {working_estimate} exceeds host budget {budget}"
            )
    spatial_rows = n_rows
    if args.selected_intervals is not None:
        if args.mode != "spatial":
            parser.error("selected intervals require spatial output")
        spatial_rows = min(
            n_rows,
            int(sum(stop - start for start, stop in args.selected_intervals) * 500) + 2,
        )
    warm_rows = (
        n_rows if args.warm_duration is None else int(round(args.warm_duration * 500))
    )
    retained_rows = (
        max(spatial_rows, warm_rows)
        if args.cleanup_spatial
        else spatial_rows * args.repeat + warm_rows
    )
    spatial_estimate = retained_rows * upper_hidden * 4
    if (
        args.mode == "spatial"
        and spatial_estimate > shutil.disk_usage(args.output).free * 0.8
    ):
        parser.error("full spatial output exceeds available disk reserve")
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
                0,
                args.encoding_duration,
                int(args.encoding_duration * args.spike_rate),
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
    parameters = {
        "environments": Environment(
            place_bin_size=args.bin_size,
            position_range=((0, args.arena), (0, args.arena)),
        ),
        "infer_track_interior": False,
    }
    fit = {
        "position_time": tracking_time,
        "position": position,
        "spike_times": training_spikes,
        "encoding_time_range": [0, args.encoding_duration],
    }
    decode_tracking_time = np.arange(int(args.duration * 30) + 1) / 30
    decode_position = np.column_stack(
        (
            args.arena * (0.5 + 0.4 * np.sin(decode_tracking_time)),
            args.arena * (0.5 + 0.4 * np.cos(decode_tracking_time * 0.71)),
        )
    )
    predict = {
        "spike_times": decode_spikes,
        "position_time": decode_tracking_time,
        "position": decode_position,
        "time_edges": np.arange(n_rows + 1) / 500,
    }
    if args.family == "sorted":
        model = NonLocalSortedSpikesDetector(**parameters)
    else:
        if args.encoding_block_size is not None:
            parameters["clusterless_algorithm_params"] = {
                "position_std": 6.0,
                "waveform_std": 24.0,
                "block_size": 10_000,
                "encoding_block_size": args.encoding_block_size,
                "position_block_size": args.position_block_size,
            }
        model = NonLocalClusterlessDetector(**parameters)
        fit["spike_waveform_features"] = [
            encoding_rng.normal(size=(len(s), args.mark_dimensions))
            for s in training_spikes
        ]
        predict["spike_waveform_features"] = [
            decoding_rng.normal(size=(len(s), args.mark_dimensions))
            for s in decode_spikes
        ]
    if args.mode != "dense":
        fit["transition_representation"] = "structured"
        predict.update(
            inference_mode="checkpointed",
            output_mode=args.mode,
            chunk_size=args.chunk_size,
            checkpoint_dir=args.output / "checkpoints",
        )
    if args.selected_intervals is not None:
        predict["selected_intervals"] = args.selected_intervals
    input_hashes = {}
    for group, arguments in [("encoding", fit), ("decoding", predict)]:
        for name, value in arguments.items():
            items = (
                value if name in {"spike_times", "spike_waveform_features"} else [value]
            )
            for index, item in enumerate(items):
                if isinstance(item, np.ndarray):
                    digest = hashlib.sha256()
                    digest.update(str(item.shape).encode())
                    digest.update(str(item.dtype).encode())
                    digest.update(np.ascontiguousarray(item).tobytes())
                    input_hashes[f"{group}/{name}/{index}"] = digest.hexdigest()
    peak = [psutil.Process().memory_info().rss]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.02):
            peak[0] = max(peak[0], psutil.Process().memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    measured = []
    disk_bytes = []
    output_dtypes = {}
    checkpoint_bytes = 0
    evidence_precision = {}
    log_evidence = None

    def invoke(index, duration=None):
        call = dict(predict)
        if duration is not None:
            call["time_edges"] = np.arange(int(round(duration * 500)) + 1) / 500
            call.pop("selected_intervals", None)
        if args.mode == "spatial":
            call["result_path"] = args.output / f"spatial-{index}"
        result = model.predict(**call)
        # Force all returned device computation; disk spatial buffers were
        # synchronized before writing and are deliberately not read in full.
        for name in ["acausal_state_probabilities"]:
            np.asarray(result[name]).sum()
        if args.mode == "spatial" and result.sizes["time"]:
            # Exercise the lazy reader with bounded first/last row selections.
            # Whole-recording conversion would defeat the output-mode budget.
            sample = result.acausal_posterior.isel(
                time=sorted({0, result.sizes["time"] - 1})
            ).values
            np.testing.assert_allclose(
                np.nansum(sample, axis=-1), 1, rtol=1e-6, atol=1e-6
            )
        return result

    def release(index):
        if args.mode == "spatial":
            path = args.output / f"spatial-{index}"
            disk_bytes.append(
                sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
            )
            if args.cleanup_spatial:
                shutil.rmtree(path)

    try:
        with jax.default_device(devices[0]):
            start = time.perf_counter()
            model.fit(**fit)
            jax.block_until_ready(model.encoding_model_)
            fit_seconds = time.perf_counter() - start
            print(
                json.dumps(
                    {"fit_seconds": fit_seconds, "hidden_bins": model.n_state_bins_}
                ),
                flush=True,
            )
            start = time.perf_counter()
            warm = invoke("warm", args.warm_duration)
            first_seconds = time.perf_counter() - start
            del warm
            release("warm")
            gc.collect()
            for i in range(args.repeat):
                start = time.perf_counter()
                result = invoke(i)
                measured.append(time.perf_counter() - start)
                output_dtypes = {
                    name: str(variable.dtype)
                    for name, variable in result.data_vars.items()
                }
                checkpoint_bytes = result.attrs.get("checkpoint_bytes", 0)
                log_evidence = float(result.attrs["marginal_log_likelihoods"])
                evidence_precision = {
                    name: result.attrs.get(name, "legacy")
                    for name in ["evidence_dtype", "evidence_accumulation"]
                }
                assert all(dtype == "float32" for dtype in output_dtypes.values())
                if i == 0 and args.save_reference:
                    np.savez(
                        args.output / "reference.npz",
                        _marginal_log_likelihoods=np.asarray(
                            result.attrs["marginal_log_likelihoods"]
                        ),
                        **{name: np.asarray(result[name]) for name in result.data_vars},
                    )
                np.testing.assert_allclose(
                    result.acausal_state_probabilities.sum("states"),
                    1,
                    rtol=1e-6,
                    atol=1e-6,
                )
                del result
                release(i)
                print(
                    json.dumps({"repetition": i, "seconds": measured[-1]}), flush=True
                )
                gc.collect()
    finally:
        stopped.set()
        sampler.join()
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    high_water_bytes = int(
        high_water if sys.platform == "darwin" else high_water * 1024
    )
    peak[0] = max(peak[0], high_water_bytes)
    hashes_after = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
        if "tests" not in path.parts
    }
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "hardware": {
            "platform": platform.platform(),
            "devices": [str(d) for d in devices],
            "device_kinds": [d.device_kind for d in devices],
            "runtime": str(getattr(devices[0].client, "platform_version", "unknown")),
            "physical_ram": psutil.virtual_memory().total,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["jax", "jaxlib", "numpy", "scipy"]
        },
        "dimensions": {
            "rows": n_rows,
            "padded_hidden": model.n_state_bins_,
            "interior_hidden": int(model.is_track_interior_state_bins_.sum()),
            "grid_shapes": [
                environment.centers_shape_ for environment in model.environments
            ],
            "states": model.state_names,
            "encoding_spikes": [len(times) for times in training_spikes],
            "decoding_spikes": [len(times) for times in decode_spikes],
        },
        "fit_seconds": fit_seconds,
        "compile_plus_first_seconds": first_seconds,
        "warm_seconds": measured,
        "median_warm_seconds": float(np.median(measured)),
        "peak_process_rss_bytes": peak[0],
        "host_budget_bytes": budget,
        "device_budget_bytes": None
        if args.device_budget_gib is None
        else int(args.device_budget_gib * 1024**3),
        "device_memory_stats": devices[0].memory_stats(),
        "source_hashes": hashes_before,
        "benchmark_sha256": benchmark_hash_before,
        "benchmark_stable_during_run": benchmark_hash_before
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_stable_during_run": hashes_before == hashes_after,
        "input_hashes": input_hashes,
        "output_dtypes": output_dtypes,
        "evidence_precision": evidence_precision,
        "marginal_log_likelihoods": log_evidence,
        "result_store_bytes": disk_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "precision": {
            "jax_enable_x64": jax.config.jax_enable_x64,
            "jax_default_matmul_precision": str(
                jax.config.jax_default_matmul_precision
            ),
            "structured_contractions": "HIGHEST",
        },
        "allocator_environment": {
            name: os.environ.get(name)
            for name in [
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
            ]
        },
        "measurement": "20ms process RSS sampling plus OS process high-water RSS; includes interpreter/compiler/inputs; page cache excluded; CPU has no device allocator counter",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "dimensions",
                    "median_warm_seconds",
                    "peak_process_rss_bytes",
                ]
            }
        )
    )
    if peak[0] > budget:
        raise RuntimeError(f"measured peak {peak[0]} exceeds declared budget {budget}")
    if hashes_before != hashes_after or not report["benchmark_stable_during_run"]:
        raise RuntimeError(
            "imported implementation changed during the benchmark; rerun with frozen sources"
        )
    stats = devices[0].memory_stats() or {}
    if (
        args.device_budget_gib is not None
        and max(
            stats.get(name, 0)
            for name in ["peak_bytes_in_use", "peak_bytes_reserved", "peak_pool_bytes"]
        )
        > args.device_budget_gib * 1024**3
    ):
        raise RuntimeError(
            "measured device allocation/reservation exceeds declared budget"
        )


if __name__ == "__main__":
    main()
