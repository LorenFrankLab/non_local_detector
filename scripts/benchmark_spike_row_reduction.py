"""Time deterministic spike-row reductions against the serial loop and scatter.

Follows the JAX benchmarking protocol: compile-plus-first-call time is
recorded separately from steady-state execution; steady-state timings are
synchronized with ``block_until_ready`` after warm-up and reported as
min/median/max. XLA's temporary buffer size for each compiled reduction is
recorded, outputs are checked against ``jax.ops.segment_sum``, and a
recompilation check reports how many executables varying spike counts create.

GPU runs should select an idle device with CUDA_VISIBLE_DEVICES and set
XLA_PYTHON_CLIENT_PREALLOCATE=false.

Example: python scripts/benchmark_spike_row_reduction.py --require-backend gpu
"""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import non_local_detector.likelihoods.common as common

SPIKES = (16, 64, 256, 2_000, 20_000, 200_000)
COLUMNS = (64, 500, 16_930)
MAX_VALUES_BYTES = 2 * 1024**3


def candidates():
    """Reductions available in this checkout, keyed by a stable name."""
    found = {"segment_sum": None}
    for name in ("_serial_row_sum", "_ordered_spike_row_sum"):
        if hasattr(common, name):
            found["serial_loop"] = getattr(common, name)
            break
    if hasattr(common, "deterministic_segment_sum"):
        found["segmented_scan"] = common.deterministic_segment_sum
    return found


def compiled_reduction(name, function, n_rows, is_sorted):
    if name == "segment_sum":
        return jax.jit(
            lambda v, i: jax.ops.segment_sum(
                v, i, num_segments=n_rows, indices_are_sorted=is_sorted
            )
        )
    if name == "segmented_scan":
        return jax.jit(lambda v, i: function(v, i, n_rows, is_sorted))
    return jax.jit(lambda v, i: function(v, i, n_rows))


def measure(function, arguments, repeats):
    """Compile+first time, steady-state seconds, temp bytes and the output."""
    start = time.perf_counter()
    output = jax.block_until_ready(function(*arguments))
    compile_and_first = time.perf_counter() - start
    jax.block_until_ready(function(*arguments))  # second warm-up call
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        timings.append(time.perf_counter() - start)
    memory = function.lower(*arguments).compile().memory_analysis()
    temp_bytes = getattr(memory, "temp_size_in_bytes", None) if memory else None
    return compile_and_first, timings, temp_bytes, np.asarray(output)


def recompilation_check(found, n_rows, n_columns, rng):
    """Executables created and compile time across spike counts 1..64."""
    report = {}
    for name, function in found.items():
        if name == "segment_sum":
            continue
        cache_size = getattr(function, "_cache_size", None)
        before = cache_size() if cache_size else None
        start = time.perf_counter()
        for n_spikes in range(1, 65):
            values = jnp.asarray(
                rng.normal(size=(n_spikes, n_columns)).astype(np.float32)
            )
            ids = jnp.asarray(np.sort(rng.integers(0, n_rows, n_spikes)), jnp.int32)
            if name == "segmented_scan":
                jax.block_until_ready(function(values, ids, n_rows, True))
            else:
                jax.block_until_ready(function(values, ids, n_rows))
        report[name] = {
            "spike_counts": 64,
            "seconds": time.perf_counter() - start,
            "new_executables": (cache_size() - before) if cache_size else None,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--require-backend", choices=("cpu", "gpu"), default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "benchmark_spike_row_reduction.json",
    )
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("at least five synchronized warm repetitions are required")
    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    rng = np.random.default_rng(20261007)
    found = candidates()
    records = []
    for n_spikes in SPIKES:
        for n_columns in COLUMNS:
            if n_spikes * n_columns * 4 > MAX_VALUES_BYTES:
                continue
            values = jnp.asarray(
                rng.normal(size=(n_spikes, n_columns)).astype(np.float32)
            )
            for is_sorted in (True, False):
                ids = rng.integers(0, args.rows, n_spikes)
                # About 1% invalid rows, which every reduction must drop.
                invalid = rng.random(n_spikes) < 0.01
                ids[invalid] = rng.choice([-1, args.rows], invalid.sum())
                if is_sorted:
                    ids = np.sort(ids)
                ids = jnp.asarray(ids.astype(np.int32))
                record = {
                    "n_spikes": n_spikes,
                    "n_columns": n_columns,
                    "sorted": is_sorted,
                }
                reference = None
                for name, function in found.items():
                    compiled = compiled_reduction(name, function, args.rows, is_sorted)
                    first, timings, temp_bytes, output = measure(
                        compiled, (values, ids), args.repeats
                    )
                    if reference is None:
                        reference = output.astype(np.float64)
                    record[name] = {
                        "compile_and_first_seconds": first,
                        "min_seconds": float(np.min(timings)),
                        "median_seconds": float(np.median(timings)),
                        "max_seconds": float(np.max(timings)),
                        "temp_bytes": temp_bytes,
                        "max_abs_difference_vs_segment_sum": float(
                            np.max(np.abs(output - reference))
                        ),
                    }
                records.append(record)
                print(json.dumps(record), flush=True)
    recompilation = recompilation_check(found, args.rows, 500, rng)
    print(json.dumps({"recompilation": recompilation}), flush=True)
    source = Path(common.__file__).read_bytes()
    report = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "x64": jax.config.x64_enabled,
        "rows": args.rows,
        "repeats": args.repeats,
        "common_sha256": hashlib.sha256(source).hexdigest(),
        "measurement": (
            "compile+first separately; steady-state min/median/max of "
            "synchronized warmed calls; XLA temp bytes from memory_analysis"
        ),
        "records": records,
        "recompilation": recompilation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
