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


def load_average() -> tuple[float, float, float] | None:
    return os.getloadavg() if hasattr(os, "getloadavg") else None


def random_model(rng, n_states: int, n_time: int) -> tuple[jnp.ndarray, ...]:
    """Prior, row-stochastic transition, and log-likelihoods as float32 arrays."""
    prior = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    ll = rng.normal(-3.0, 2.0, (n_time, n_states))
    return tuple(jnp.asarray(x, dtype=jnp.float32) for x in (prior, transition, ll))


def split_into_two_discrete_states(
    transition: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Assign half the state bins to each of two discrete states.

    Returns ``(state_ind, continuous)`` where ``continuous`` is ``transition``
    renormalized so each row sums to one within every destination block.
    """
    n_states = transition.shape[0]
    state_ind = np.repeat(np.arange(2), n_states // 2)
    continuous = np.asarray(transition).copy()
    for destination in range(2):
        block = continuous[:, state_ind == destination]
        continuous[:, state_ind == destination] = block / block.sum(
            axis=1, keepdims=True
        )
    return jnp.asarray(state_ind), jnp.asarray(continuous, dtype=jnp.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="main")
    parser.add_argument("--baseline-core", type=Path)
    parser.add_argument("--time-steps", type=int, default=1000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 200, 1000])
    parser.add_argument(
        "--extra-size",
        type=int,
        default=200,
        help="state count (must be even) for the smoother and covariate-dependent "
        "cases, which run once at this size in addition to the stationary sweep",
    )
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if min(options.time_steps, options.repetitions, *options.sizes) < 1:
        parser.error("time steps, repetitions, and sizes must be positive")
    if options.extra_size < 2 or options.extra_size % 2:
        parser.error("--extra-size must be an even number of states >= 2")

    root = Path(__file__).resolve().parents[1]
    relative_core = "src/non_local_detector/core.py"
    core_path = root / relative_core
    sources = {
        "git_baseline": subprocess.check_output(
            ["git", "show", f"{options.baseline_ref}:{relative_core}"],
            cwd=root,
            text=True,
        ),
        "current": core_path.read_text(),
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
        "load_average_start": load_average(),
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
        args = random_model(data_rng, n_states, options.time_steps)
        record(f"stationary_{n_states}", "_filter_impl", args)

    n_states = options.extra_size
    prior, transition, ll = random_model(data_rng, n_states, options.time_steps)
    filtered = variants["current"]["_filter_jit"](prior, transition, ll)[1][0]
    record(f"smoother_{n_states}", "_smoother_impl", (transition, filtered))

    state_ind, continuous = split_into_two_discrete_states(transition)
    discrete = jnp.asarray(
        data_rng.dirichlet(np.ones(2), size=(options.time_steps, 2)),
        dtype=jnp.float32,
    )
    record(
        f"covariate_{n_states}",
        "_filter_covariate_dependent_impl",
        (prior, discrete, continuous, state_ind, ll),
    )

    report["load_average_end"] = load_average()
    report["current_source_unchanged"] = core_path.read_text() == sources["current"]
    if not report["current_source_unchanged"]:
        print(
            f"WARNING: {relative_core} changed on disk during the run; the "
            "'current' timings may not correspond to the file as it is now",
            flush=True,
        )
    if options.output:
        options.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(
            "(pass --output to keep the full report: compile times, samples, hashes, load)"
        )


if __name__ == "__main__":
    main()
