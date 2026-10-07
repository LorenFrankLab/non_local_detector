"""Bounded Phase 7a benchmark; no production/full-hour claim.

Run duration ladders at fixed --chunk-size in isolated processes. This records
compile/first-run separately, >=5 synchronized warm end-to-end repetitions,
process high-water RSS and peak sampled RSS, checkpoint/output bytes, dimensions,
dtype and actual device. CPU evidence cannot qualify GPU configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import jax
import numpy as np
import psutil

from non_local_detector.checkpointed_inference import checkpointed_forward_backward
from non_local_detector.core import filter, smoother


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-budget-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "benchmark_checkpointed_inference.json",
    )
    parser.add_argument("--worker", choices=["dense", "checkpointed"])
    return parser.parse_args()


def worker(args):
    dtype = np.dtype(args.dtype)
    # Conservative bounded reference allocation preflight. Full-hour references
    # must never be launched through this script without an independent budget.
    estimated = (
        8 * args.rows * args.bins * dtype.itemsize + 3 * args.bins**2 * dtype.itemsize
    )
    if estimated > args.memory_budget_bytes:
        raise SystemExit(
            f"Estimated tractable-reference memory {estimated} exceeds budget {args.memory_budget_bytes}"
        )
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise SystemExit("float64 requires JAX_ENABLE_X64=1")
    rng = np.random.default_rng(701)
    matrix = rng.uniform(0.05, 1, (args.bins, args.bins)).astype(dtype)
    matrix /= matrix.sum(1, keepdims=True)
    initial = np.full(args.bins, 1 / args.bins, dtype=dtype)
    state_ind = np.arange(args.bins) % 4
    edges = np.arange(args.rows + 1, dtype=float) * 0.002
    columns = np.arange(args.bins, dtype=dtype)[None, :]

    def ll(time_edges, *, row_slice, is_missing):
        rows = np.arange(row_slice.start, row_slice.stop, dtype=dtype)[:, None]
        return -np.square(
            np.sin(
                rows * np.asarray(0.013, dtype=dtype)
                + columns * np.asarray(0.07, dtype=dtype)
            )
        ).astype(dtype)

    diagnostics = {}

    def run():
        if args.worker == "dense":
            values = ll(edges, row_slice=slice(0, args.rows), is_missing=None)
            (evidence, _), (causal, _) = filter(initial, matrix, values)
            acausal = smoother(matrix, causal)
            acausal.block_until_ready()
            result = np.column_stack(
                [
                    np.asarray(acausal)[:, state_ind == state].sum(1)
                    for state in range(4)
                ]
            )
            return result, float(evidence)
        result = checkpointed_forward_backward(
            edges,
            initial,
            ll,
            transition_matrix=matrix,
            state_ind=state_ind,
            chunk_size=args.chunk_size,
            dtype=dtype,
        )
        diagnostics.update(result.diagnostics)
        return (
            result.dataset.acausal_state_probabilities.values,
            result.marginal_log_likelihood,
        )

    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        started = time.perf_counter()
        outputs, evidence = run()
        first = time.perf_counter() - started
        assert outputs.dtype == dtype
        warm = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            current, current_evidence = run()
            warm.append(time.perf_counter() - started)
            np.testing.assert_array_equal(current, outputs)
            np.testing.assert_equal(current_evidence, evidence)
            del current
    finally:
        stop.set()
        thread.join()
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    high_water_bytes = (
        high_water if platform.system() == "Darwin" else high_water * 1024
    )
    return {
        "variant": args.worker,
        "rows": args.rows,
        "bins": args.bins,
        "states": 4,
        "chunk_size": args.chunk_size,
        "dtype": dtype.name,
        "requested_outputs": ["acausal_state_probabilities"],
        "first_run_including_compilation_seconds": first,
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "warm_min_seconds": min(warm),
        "warm_max_seconds": max(warm),
        "peak_sampled_rss_bytes": peak[0],
        "process_high_water_rss_bytes": high_water_bytes,
        "memory_sampling_interval_seconds": 0.005,
        "memory_budget_bytes": args.memory_budget_bytes,
        "device": str(jax.devices()[0]),
        "device_memory_stats": jax.devices()[0].memory_stats(),
        "output_bytes": outputs.nbytes,
        "checkpoint_bytes": diagnostics.get("checkpoint_bytes", 0),
        "evidence": evidence,
        "posterior_sum_checksum": float(outputs.sum()),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "python": sys.version,
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "diagnostics": diagnostics,
    }


def main():
    args = parse()
    if min(args.rows, args.bins, args.chunk_size) <= 0 or args.repeat < 5:
        raise SystemExit(
            "Positive dimensions and at least five repetitions are required"
        )
    if args.worker:
        print(
            json.dumps(worker(args), default=lambda value: np.asarray(value).tolist())
        )
        return
    variants = ["dense", "checkpointed"]
    random.Random(701).shuffle(variants)
    results = []
    for variant in variants:
        command = [
            sys.executable,
            __file__,
            "--worker",
            variant,
            "--rows",
            str(args.rows),
            "--bins",
            str(args.bins),
            "--chunk-size",
            str(args.chunk_size),
            "--repeat",
            str(args.repeat),
            "--dtype",
            args.dtype,
            "--memory-budget-bytes",
            str(args.memory_budget_bytes),
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=True, env=os.environ.copy()
        )
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    report = {
        "baseline_revision": "fe1b2e9",
        "seed": 701,
        "protocol": "Independent processes, randomized variant order, five or more synchronized warm end-to-end repetitions; memory includes compilation; not a paired interleaved timing claim",
        "workload": "Deterministic analytical likelihood, stationary dense transition; not a neural-backend or full-hour qualification",
        "filesystem_cache": "OS page-cache bytes are not included in process RSS and require separate production measurement",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
