"""Compare compiled HMM kernels with a Git baseline and an optional saved core.

Run from the repository with ``uv run python scripts/benchmark_hmm_conditioning.py``.
Use ``--baseline-core /path/to/core_before.py`` to compare uncommitted revisions.
Timings use precomputed likelihoods and exclude compilation and host diagnostics.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_core(source: str, name: str) -> dict:
    namespace = {"__name__": name}
    exec(compile(source, name, "exec"), namespace)
    return namespace


def measure(variants, implementation, args, repetitions, rng):
    compiled, compile_ms, samples = {}, {}, {}
    for name, namespace in variants.items():
        start = time.perf_counter()
        compiled[name] = jax.jit(namespace[implementation]).lower(*args).compile()
        compile_ms[name] = 1000 * (time.perf_counter() - start)
        samples[name] = []

    for _ in range(4):
        for function in compiled.values():
            jax.block_until_ready(function(*args))

    # Alternate implementations in random order to reduce thermal/load drift.
    # Every result is synchronized, and all inputs already reside on device.
    for _ in range(repetitions):
        for name in rng.permutation(list(variants)):
            start = time.perf_counter()
            jax.block_until_ready(compiled[name](*args))
            samples[name].append(1000 * (time.perf_counter() - start))

    result = {"compile_ms": compile_ms, "milliseconds": {}}
    for name, values in samples.items():
        result["milliseconds"][name] = {
            "median": float(np.median(values)),
            "quartiles": np.percentile(values, [25, 75]).tolist(),
            "samples": values,
        }
    result["current_to_reference_ratio"] = {
        name: {
            "median": float(np.median(np.asarray(samples["current"]) / values)),
            "quartiles": np.percentile(
                np.asarray(samples["current"]) / values, [25, 75]
            ).tolist(),
        }
        for name, values in samples.items()
        if name != "current"
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="main")
    parser.add_argument("--baseline-core", type=Path)
    parser.add_argument("--time-steps", type=int, default=1000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 200, 1000])
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if min(options.time_steps, options.repetitions, *options.sizes) < 1:
        parser.error("time steps, repetitions, and sizes must be positive")
    root = Path(__file__).resolve().parents[1]
    relative_core = "src/non_local_detector/core.py"
    sources = {
        "git_baseline": subprocess.check_output(
            ["git", "show", f"{options.baseline_ref}:{relative_core}"],
            cwd=root,
            text=True,
        ),
        "current": (root / relative_core).read_text(),
    }
    if options.baseline_core:
        sources["saved_baseline"] = options.baseline_core.read_text()
    variants = {name: load_core(source, name) for name, source in sources.items()}
    report = {
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "dtype": "float32",
        "time_steps": options.time_steps,
        "repetitions": options.repetitions,
        "baseline_ref": options.baseline_ref,
        "source_sha256": {
            name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in sources.items()
        },
        "load_average_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "cases": {},
    }
    data_rng, order_rng = np.random.default_rng(640), np.random.default_rng(184)

    def record(name, implementation, args):
        result = measure(variants, implementation, args, options.repetitions, order_rng)
        report["cases"][name] = result
        medians = {
            label: round(values["median"], 3)
            for label, values in result["milliseconds"].items()
        }
        print(f"{name}: median ms {medians}", flush=True)

    for n_states in options.sizes:
        prior = jnp.asarray(data_rng.dirichlet(np.ones(n_states)), dtype=jnp.float32)
        transition = jnp.asarray(
            data_rng.dirichlet(np.ones(n_states), size=n_states), dtype=jnp.float32
        )
        ll = jnp.asarray(
            data_rng.normal(-3.0, 2.0, (options.time_steps, n_states)),
            dtype=jnp.float32,
        )
        record(f"stationary_{n_states}", "_filter_impl", (prior, transition, ll))
        if n_states == 200:
            filtered = variants["current"]["_filter_jit"](prior, transition, ll)[1][0]
            record("smoother_200", "_smoother_impl", (transition, filtered))
            state_ind = np.repeat(np.arange(2), n_states // 2)
            continuous = np.asarray(transition).copy()
            for destination in range(2):
                block = continuous[:, state_ind == destination]
                continuous[:, state_ind == destination] = block / block.sum(
                    axis=1, keepdims=True
                )
            discrete = data_rng.dirichlet(np.ones(2), size=(options.time_steps, 2))
            record(
                "covariate_200",
                "_filter_covariate_dependent_impl",
                (
                    prior,
                    jnp.asarray(discrete, dtype=jnp.float32),
                    jnp.asarray(continuous, dtype=jnp.float32),
                    jnp.asarray(state_ind),
                    ll,
                ),
            )
    report["load_average_end"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    report["current_source_unchanged"] = (root / relative_core).read_text() == sources[
        "current"
    ]
    if options.output:
        options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
