"""Time deterministic spike-row reductions against the serial loop and scatter.

Synchronized, warmed medians on the default JAX device. GPU runs should select
an idle device with CUDA_VISIBLE_DEVICES and set
XLA_PYTHON_CLIENT_PREALLOCATE=false.

Example: python scripts/benchmark_spike_row_reduction.py --output REPORT.json
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


def measure(function, arguments, repeats):
    for _ in range(2):
        jax.block_until_ready(function(*arguments))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "benchmark_spike_row_reduction.json",
    )
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("at least five synchronized warm repetitions are required")
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
                if is_sorted:
                    ids = np.sort(ids)
                ids = jnp.asarray(ids.astype(np.int32))
                record = {
                    "n_spikes": n_spikes,
                    "n_columns": n_columns,
                    "sorted": is_sorted,
                }
                for name, function in found.items():
                    if name == "segment_sum":
                        compiled = jax.jit(
                            lambda v, i, s=is_sorted: jax.ops.segment_sum(
                                v, i, num_segments=args.rows, indices_are_sorted=s
                            )
                        )
                    elif name == "segmented_scan":
                        compiled = jax.jit(
                            lambda v, i, f=function, s=is_sorted: f(v, i, args.rows, s)
                        )
                    else:
                        compiled = jax.jit(lambda v, i, f=function: f(v, i, args.rows))
                    record[f"{name}_seconds"] = measure(
                        compiled, (values, ids), args.repeats
                    )
                records.append(record)
                print(json.dumps(record), flush=True)
    source = Path(common.__file__).read_bytes()
    report = {
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "x64": jax.config.x64_enabled,
        "rows": args.rows,
        "common_sha256": hashlib.sha256(source).hexdigest(),
        "measurement": "median of synchronized warmed calls",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
