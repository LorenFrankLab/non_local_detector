"""Compare ordered sparse emissions with a trusted pre-fix source snapshot.

Example: extract common.py from commit 7381346 into a temporary file, then run
this script with --baseline-common PATH --platform cpu --output REPORT.json.
GPU runs should select an idle device with CUDA_VISIBLE_DEVICES and disable
preallocation. Results include warmed timings and changing-count cache behavior.
"""

import argparse
import ast
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import non_local_detector.likelihoods.common as common


def baseline_helpers(path: Path) -> tuple[Callable, Callable]:
    """Load only the emission function from a trusted repository snapshot."""
    source = path.read_text()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_poisson_nonlocal_log_likelihood"
    )
    code = ast.get_source_segment(source, node)
    namespace = dict(common.__dict__)
    exec(code, namespace)
    matrix = jax.jit(namespace["_poisson_nonlocal_log_likelihood"])
    ordered_code = code.replace(
        "matrix_is_safe, matrix_accumulation, xlogy_accumulation, None",
        "jnp.asarray(False), matrix_accumulation, xlogy_accumulation, None",
    )
    if ordered_code == code:
        raise ValueError("Baseline must contain the pre-fix matrix accumulation path")
    exec(ordered_code, namespace)
    return matrix, jax.jit(namespace["_poisson_nonlocal_log_likelihood"])


def measure(
    function: Callable, arguments: tuple, repeats: int
) -> tuple[float, np.ndarray]:
    """Warm both compilation and execution before synchronized measurements."""
    for _ in range(2):
        result = function(*arguments)
        result.block_until_ready()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*arguments)
        result.block_until_ready()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)), np.asarray(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-common", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--bins", type=int, nargs="+", default=[257, 8100, 32400])
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--neurons", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--sequential-chunks", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.rows, args.neurons, args.repeat, args.sequential_chunks, *args.bins)
        < 1
    ):
        parser.error("dimensions and repetition counts must be positive")
    jax.config.update("jax_enable_x64", False)
    device = jax.devices(args.platform)[0]
    matrix, fallback = baseline_helpers(args.baseline_common)
    runtime = Path(common.__file__).parent.parent
    sources = {
        str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime.rglob("*.py"))
        if "tests" not in path.relative_to(runtime).parts
    }
    report = {
        "jax": jax.__version__,
        "device": str(device),
        "device_kind": device.device_kind,
        "runtime_aggregate_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": sources,
        "baseline_common_sha256": hashlib.sha256(
            args.baseline_common.read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement": "median of synchronized warmed calls; includes host packing",
        "cases": [],
    }
    rng = np.random.default_rng(8171)
    with jax.default_device(device):
        for bins in args.bins:
            for burst in [False, True]:
                counts = rng.poisson(0.01, (args.rows, args.neurons)).astype(np.int32)
                if burst:
                    counts[[min(100, args.rows - 1), args.rows - 1]] = 1
                rates = jnp.asarray(
                    rng.uniform(0.1, 20, (args.neurons, bins)), dtype=jnp.float32
                )
                durations = jnp.full(args.rows, 0.002, dtype=jnp.float32)
                summed = rates.sum(0)
                row = {
                    "rows": args.rows,
                    "neurons": args.neurons,
                    "bins": bins,
                    "burst": burst,
                }
                outputs = []
                for label, function, metadata in [
                    ("matrix", matrix, jnp.asarray(counts)),
                    ("fallback", fallback, jnp.asarray(counts)),
                    ("packed", common._poisson_nonlocal_log_likelihood, counts),
                ]:
                    elapsed, result = measure(
                        function, (metadata, rates, durations, summed), args.repeat
                    )
                    row[f"{label}_median_seconds"] = elapsed
                    outputs.append(result)
                row["packed_over_matrix"] = (
                    row["packed_median_seconds"] / row["matrix_median_seconds"]
                )
                row["fallback_over_packed"] = (
                    row["fallback_median_seconds"] / row["packed_median_seconds"]
                )
                for label, output in zip(
                    ["matrix", "packed"], outputs[::2], strict=True
                ):
                    row[f"{label}_vs_ordered_max_difference"] = float(
                        np.max(np.abs(output - outputs[1]))
                    )
                report["cases"].append(row)
                print(json.dumps(row), flush=True)

        rates = jnp.asarray(
            rng.uniform(0.1, 20, (args.neurons, 257)), dtype=jnp.float32
        )
        durations = jnp.full(args.rows, 0.002, dtype=jnp.float32)
        summed = rates.sum(0)
        common._poisson_packed_log_likelihood.clear_cache()
        sequential = []
        for number in range(args.sequential_chunks):
            counts = rng.poisson(0.01, (args.rows, args.neurons)).astype(np.int32)
            start = time.perf_counter()
            common._poisson_nonlocal_log_likelihood(
                counts, rates, durations, summed
            ).block_until_ready()
            sequential.append(
                {
                    "number": number,
                    "seconds": time.perf_counter() - start,
                    "compiled_signatures": common._poisson_packed_log_likelihood._cache_size(),
                }
            )
        report["sequential_chunks"] = sequential
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
