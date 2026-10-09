"""Bounded transition-product benchmark; large runs do not claim dense parity.

Record setup/compile separately and at least five synchronized interleaved warm
measurements. Exact dense references run only when their allocation estimate
fits --dense-budget-bytes. CPU evidence cannot qualify CUDA/A100 precision.
"""

import argparse
import json
import pickle
import platform
import random
import resource
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from non_local_detector.environment import Environment
from non_local_detector.models._defaults import _ModelDefaults
from non_local_detector.models.base import _DetectorBase
from non_local_detector.transition_operators import (
    LazyDenseTransition,
    build_transition_operator,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent", type=float, default=20.0)
    parser.add_argument("--bin-size", type=float, default=2.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dense-budget-bytes", type=int, default=128 * 1024**2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "benchmark_transition_operators.json",
    )
    args = parser.parse_args()
    if args.repeat < 5:
        parser.error("at least five synchronized warm repetitions are required")
    dtype = np.dtype(args.dtype)
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        parser.error("float64 requires JAX_ENABLE_X64=1")
    rss = psutil.Process()
    start = time.perf_counter()
    env = Environment(place_bin_size=args.bin_size).fit_place_grid(
        np.array([[0.0, 0.0], [args.extent, args.extent]]),
        infer_track_interior=False,
        compute_all_pairs_distances=False,
    )
    environment_seconds = time.perf_counter() - start
    bins = len(env.place_bin_centers_)
    sizes = (1, 1, bins, bins)
    state_ind = np.repeat(np.arange(4), sizes)
    defaults = _ModelDefaults.non_local_defaults()
    observations = defaults["observation_models"]()
    descriptors = defaults["continuous_transition_types"]()
    weights = defaults["discrete_transition_type"]().make_state_transition()[0]
    start = time.perf_counter()
    operator = build_transition_operator(
        descriptors, [env], sizes, observations
    ).bind_discrete(weights)
    operator_seconds = time.perf_counter() - start
    vector = jnp.asarray(
        np.random.default_rng(51).uniform(0.01, 1, sum(sizes)), dtype=dtype
    )
    vector /= vector.sum()
    structured = jax.jit(lambda op, values: (op.forward(values), op.backward(values)))
    start = time.perf_counter()
    compiled = structured.lower(operator, vector).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(operator, vector))
    functions = {"structured": lambda: compiled(operator, vector)}
    dense_status = "not run: bounded dense-reference estimate exceeds budget"
    dense_bytes = sum(sizes) ** 2 * 8
    errors = None
    # Combined matrix, temporary conditional matrix and runtime cast coexist.
    if 3 * dense_bytes <= args.dense_budget_bytes:
        namespace = SimpleNamespace(
            continuous_transition_types=descriptors,
            environments=[env],
            observation_models=observations,
            local_position_std=None,
            state_ind_=state_ind,
            _get_environment_by_name=lambda name: env,
        )
        _DetectorBase.initialize_continuous_state_transition(namespace, descriptors)
        dense = (
            namespace.continuous_state_transitions_
            * weights[np.ix_(state_ind, state_ind)]
        )
        matrix = jnp.asarray(dense, dtype=dtype)
        dense_function = jax.jit(
            lambda values: (
                jnp.matmul(values, matrix, precision=jax.lax.Precision.HIGHEST),
                jnp.matmul(matrix, values, precision=jax.lax.Precision.HIGHEST),
            )
        )
        jax.block_until_ready(dense_function(vector))
        functions["dense"] = lambda: dense_function(vector)
        errors = [
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(
                functions["structured"](), functions["dense"](), strict=True
            )
        ]
        dense_status = "same-input highest-precision dense products measured"
    timings = {name: [] for name in functions}
    order = list(functions) * args.repeat
    random.Random(701).shuffle(order)
    for name in order:
        start = time.perf_counter()
        jax.block_until_ready(functions[name]())
        timings[name].append(time.perf_counter() - start)
    analysis = compiled.memory_analysis()
    result = {
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": args.dtype,
        "x64_enabled": jax.config.x64_enabled,
        "matmul_precision": "HIGHEST",
        "extent": args.extent,
        "bin_size": args.bin_size,
        "state_sizes": sizes,
        "environment_seconds": environment_seconds,
        "operator_seconds": operator_seconds,
        "compile_seconds": compile_seconds,
        "graph_storage_bytes": env.distance_between_nodes_.storage_nbytes,
        "operator_storage_bytes": operator.storage_nbytes,
        "operator_pickle_bytes": len(pickle.dumps(LazyDenseTransition(operator))),
        "dense_matrix_bytes": dense_bytes,
        "dense_reference_status": dense_status,
        "product_max_abs_errors": errors,
        "warm_seconds": timings,
        "warm_median_seconds": {
            name: statistics.median(values) for name, values in timings.items()
        },
        "compiler_argument_bytes": analysis.argument_size_in_bytes,
        "compiler_output_bytes": analysis.output_size_in_bytes,
        "compiler_temporary_bytes": analysis.temp_size_in_bytes,
        "rss_bytes": rss.memory_info().rss,
        "high_water_rss_platform_units": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "qualification": "isolated transition products only; no full-hour or unmeasured CUDA qualification",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
