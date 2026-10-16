"""Measure first-call and warm KDE kernel runtime without a wall-clock test gate.

The default workload matches the former 100 x 200 x 4 test probes. Each kernel
is checked against a NumPy Gaussian reference before reporting its timings.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from compare_prediction_modes import provenance

from non_local_detector.likelihoods.clusterless_kde import (
    kde_distance,
    log_kde_distance,
)


def measure(predict, repeats):
    """Synchronize the first call and every warm call; return output and timings."""
    start = time.perf_counter()
    first = jax.block_until_ready(predict())
    first_ms = (time.perf_counter() - start) * 1000
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(predict())
        timings.append((time.perf_counter() - start) * 1000)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(first))
    return result, {
        "compile_and_first_ms": first_ms,
        "warm_ms": {
            "min": min(timings),
            "median": float(np.median(timings)),
            "max": max(timings),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-points", type=int, default=100)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.eval_points, args.samples, args.dimensions) < 1:
        parser.error("workload dimensions must be positive")
    if args.repeats < 5:
        parser.error("--repeats must be at least 5")

    rng = np.random.default_rng(404)
    eval_points = rng.standard_normal((args.eval_points, args.dimensions)).astype(
        np.float32
    )
    samples = rng.standard_normal((args.samples, args.dimensions)).astype(np.float32)
    squared_distance = (
        (samples.astype(np.float64)[:, None, :] - eval_points[None, :, :]) ** 2
    ).sum(axis=-1)
    expected_log = -0.5 * (squared_distance + args.dimensions * np.log(2 * np.pi))
    inputs = (jnp.asarray(eval_points), jnp.asarray(samples), jnp.ones(args.dimensions))
    report = {
        **provenance(__file__),
        "jax": jax.__version__,
        "workload": vars(args),
        "kernels": {},
    }
    for name, kernel, expected in (
        ("linear", kde_distance, np.exp(expected_log)),
        ("log", log_kde_distance, expected_log),
    ):
        result, timing = measure(lambda _kernel=kernel: _kernel(*inputs), args.repeats)
        result = np.asarray(result)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-7)
        report["kernels"][name] = {**timing, "dtype": str(result.dtype)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
