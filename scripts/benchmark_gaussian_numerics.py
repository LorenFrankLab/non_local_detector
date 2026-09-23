"""Measure Gaussian score/EM runtime, compiled buffers, and preservation outputs.

Run against a frozen checkout via PYTHONPATH to compare implementations, e.g.:
    uv run python scripts/benchmark_gaussian_numerics.py /tmp/gaussian-current

Memory numbers describe XLA's compiled CPU buffers, not accelerator peak RSS.
Compilation is excluded from the synchronized steady-state timings.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods import gmm
from non_local_detector.likelihoods.gmm import _estimate_log_gaussian_prob

N_WARMUP_RUNS = 3
N_TIMED_RUNS = 15


# The same 3-warmup / 15-timed / median harness as
# ``scripts/benchmark_chunk_likelihood_runtime.py::measure``; consolidate the
# three copies under ``scripts/`` into a shared module if a fourth appears.
def measure(
    executable: jax.stages.Compiled, arguments: tuple[Any, ...]
) -> tuple[Any, dict[str, Any]]:
    """Time one compiled executable at steady state and size its buffers.

    Compilation is excluded: the executable is already compiled and is run
    ``N_WARMUP_RUNS`` times before the ``N_TIMED_RUNS`` timed, synchronized
    executions that the statistics come from.

    Parameters
    ----------
    executable : jax.stages.Compiled
        Compiled callable, as returned by ``jax.jit(...).lower(...).compile()``.
    arguments : tuple
        Positional arguments matching the signature the executable was lowered
        for.

    Returns
    -------
    result : Any
        Output of the final timed execution, with the executable's output pytree
        structure (for ``_em_fit_while_loop``: ``(params, lower_bound,
        n_iterations, converged)``).
    stats : dict
        ``median_ms`` (float) and ``p10_p90_ms`` (list of two floats) of the
        wall-clock times of the timed executions, in milliseconds, plus the
        compiled module's ``argument_bytes``, ``output_bytes`` and
        ``temporary_bytes`` from ``memory_analysis()``. These are XLA's compiled
        CPU buffer sizes, not accelerator peak RSS.
    """
    for _ in range(N_WARMUP_RUNS):
        result = jax.block_until_ready(executable(*arguments))
    timings = []
    for _ in range(N_TIMED_RUNS):
        start = time.perf_counter()
        result = jax.block_until_ready(executable(*arguments))
        timings.append((time.perf_counter() - start) * 1000)
    stats = executable.memory_analysis()
    return result, {
        "median_ms": statistics.median(timings),
        "p10_p90_ms": np.percentile(timings, [10, 90]).tolist(),
        "argument_bytes": stats.argument_size_in_bytes,
        "output_bytes": stats.output_size_in_bytes,
        "temporary_bytes": stats.temp_size_in_bytes,
    }


def check_em_iterations(
    executable: jax.stages.Compiled,
    arguments: tuple[Any, ...],
    covariance_type: str,
    expected_iterations: int,
) -> None:
    """Run the EM executable once, untimed, and require the full iteration count.

    The EM benchmark passes ``tol=0.0`` to force exactly ``expected_iterations``
    steps so every covariance type does the same amount of work. That is not a
    guarantee: ``_em_fit_while_loop`` continues only while ``delta > tol``, so a
    ``delta`` of exactly ``0.0`` (a degenerate fit whose lower bound stops
    moving) or of ``NaN`` (a non-finite lower bound) both fail the test and stop
    the loop early. Checking on one untimed execution here reports that in
    seconds, instead of after ``measure`` has already spent its warm-up and timed
    runs on a short-circuited loop.

    Parameters
    ----------
    executable : jax.stages.Compiled
        Compiled ``_em_fit_while_loop``.
    arguments : tuple
        Positional arguments for ``executable``.
    covariance_type : str
        Covariance type being benchmarked; used in the error message.
    expected_iterations : int
        Iteration count the benchmark requires.

    Raises
    ------
    RuntimeError
        If the loop ran a different number of iterations, or if its final lower
        bound is not finite.
    """
    _, final_lb, final_i, _ = jax.block_until_ready(executable(*arguments))
    iterations = int(final_i)
    lower_bound = float(final_lb)
    if iterations != expected_iterations or not np.isfinite(lower_bound):
        raise RuntimeError(
            f"EM ({covariance_type}) ran {iterations} of the requested "
            f"{expected_iterations} iterations with a final lower bound of "
            f"{lower_bound!r}. With tol=0.0 the loop stops as soon as the "
            "lower-bound delta is exactly 0.0 or NaN, so the input data is "
            "degenerate (identical or collinear samples) or produced a "
            "non-finite lower bound; the timings would compare unequal work."
        )


def main() -> None:
    """Benchmark Gaussian scoring and EM fitting and write the report.

    Writes ``scores.npz`` (the first 256 rows of each scored covariance kind, for
    cross-checking implementations) and ``measurements.json`` (the timing and
    compiled-buffer report) into the ``output`` directory, and prints the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=8)
    parser.add_argument("--em-iterations", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(14)
    X = jnp.asarray(rng.normal(size=(args.samples, args.dimensions)), dtype=jnp.float32)
    means = jnp.asarray(
        rng.normal(size=(args.components, args.dimensions)), dtype=jnp.float32
    )
    report = {
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "implementation": gmm.__file__,
        "shape": [args.samples, args.components, args.dimensions],
        "results": {},
    }
    scores = {}
    for kind in ("diag", "spherical"):
        shape = (
            (args.components, args.dimensions) if kind == "diag" else (args.components,)
        )
        precision = jnp.asarray(rng.uniform(0.5, 1.5, shape), dtype=jnp.float32)
        executable = _estimate_log_gaussian_prob.lower(
            X, means, precision, kind
        ).compile()
        result, stats = measure(executable, (X, means, precision))
        report["results"][kind] = stats
        scores[kind] = np.asarray(result)[:256]

    # Identical starting parameters and a fixed iteration count isolate fitting
    # cost from initialization, convergence, and restart-selection differences.
    weights = jnp.full(args.components, 1 / args.components, dtype=X.dtype)
    for kind in ("full", "tied", "diag", "spherical"):
        covariance = {
            "full": jnp.broadcast_to(
                jnp.eye(args.dimensions, dtype=X.dtype),
                (args.components, args.dimensions, args.dimensions),
            ),
            "tied": jnp.eye(args.dimensions, dtype=X.dtype),
            "diag": jnp.ones_like(means),
            "spherical": jnp.ones_like(weights),
        }[kind]
        params = (weights, means, covariance)
        arguments = (X, params, 0.0, args.em_iterations, 1e-6)
        executable = gmm._em_fit_while_loop.lower(*arguments, kind).compile()
        # Before paying for the warm-up and timed runs, not after them.
        check_em_iterations(executable, arguments, kind, args.em_iterations)
        result, stats = measure(executable, arguments)
        stats["iterations"] = int(result[2])
        report["results"][f"em_{kind}"] = stats
    np.savez(args.output / "scores.npz", **scores)
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
