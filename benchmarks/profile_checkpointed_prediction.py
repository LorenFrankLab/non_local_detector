"""Attribute checkpointed prediction time to its stages to find bottlenecks.

Runs the ``compare_prediction_modes.py`` workload in checkpointed compact mode,
then times one warm prediction with each stage instrumented:

- likelihood algorithms, split into local, non-local and no-spike calls
  (both the forward pass and the replay evaluate every chunk);
- the jitted forward-filter and backward-smoother chunk kernels;
- checkpoint writes and reads;
- everything else (Python glue, transfers, conversions).

Instrumented stages synchronize with ``block_until_ready``, which removes
asynchronous overlap, so stage times sum to slightly more than an
uninstrumented prediction; the uninstrumented warm time is reported too.
``--cprofile`` additionally records the host functions with the most own time
during an uninstrumented warm prediction, to break down "other".

Example::

    uv run python benchmarks/profile_checkpointed_prediction.py --family sorted \\
        --duration 30 --output /tmp/profile-sorted.json
"""

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np

# tqdm reads this when non_local_detector is imported inside main().
os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_prediction_modes import (  # noqa: E402
    provenance,
    set_population_defaults,
    workload,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--encoding-duration", type=float, default=10.0)
    parser.add_argument("--arena", type=float, default=180.0)
    parser.add_argument("--bin-size", type=float, default=2.0)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--spike-rate", type=float, default=None)
    parser.add_argument("--rate-spread", type=float, default=0.0)
    parser.add_argument("--mark-dimensions", type=int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="rows per checkpoint chunk (default: the detector default)",
    )
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_population_defaults(args)

    import jax

    import non_local_detector.checkpointed_inference as checkpointed
    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )
    from non_local_detector.likelihoods import (
        _CLUSTERLESS_ALGORITHMS,
        _SORTED_SPIKES_ALGORITHMS,
        no_spike,
    )

    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    fit_kwargs, predict_kwargs = workload(args)
    fit_kwargs["transition_representation"] = "structured"
    checkpoint_dir = args.output.with_suffix("") / "checkpoints"
    predict_kwargs.update(
        inference_mode="checkpointed",
        output_mode="compact",
        checkpoint_dir=checkpoint_dir,
    )
    if args.chunk_size is not None:
        predict_kwargs["chunk_size"] = args.chunk_size
    detector = (
        NonLocalSortedSpikesDetector
        if args.family == "sorted"
        else NonLocalClusterlessDetector
    )
    model = detector(
        environments=Environment(
            place_bin_size=args.bin_size,
            position_range=((0, args.arena), (0, args.arena)),
        ),
        infer_track_interior=False,
    )
    model.fit(**fit_kwargs)
    if args.chunk_size is None:
        args.chunk_size = checkpointed.default_chunk_size(
            int(np.count_nonzero(model.is_track_interior_state_bins_))
        )
    model.predict(**predict_kwargs)  # compile and warm
    start = time.perf_counter()
    model.predict(**predict_kwargs)
    uninstrumented = time.perf_counter() - start
    host_functions = None
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        model.predict(**predict_kwargs)
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
        host_functions = [
            {
                "function": f"{Path(file).name}:{line}({name})",
                "own_seconds": own,
                "cumulative_seconds": cumulative,
                "calls": calls_,
            }
            for (file, line, name), (_, calls_, own, cumulative, _) in sorted(
                stats.stats.items(), key=lambda item: -item[1][2]
            )[:25]
        ]

    seconds = defaultdict(float)
    calls = defaultdict(int)

    def timed(name, function, synchronize=True):
        @wraps(function)
        def wrapper(*inner_args, **inner_kwargs):
            begin = time.perf_counter()
            result = function(*inner_args, **inner_kwargs)
            if synchronize:
                result = jax.block_until_ready(result)
            seconds[name] += time.perf_counter() - begin
            calls[name] += 1
            return result

        return wrapper

    registry = (
        _SORTED_SPIKES_ALGORITHMS
        if args.family == "sorted"
        else _CLUSTERLESS_ALGORITHMS
    )
    originals = dict(registry)
    for name, (fit_function, predict_function) in originals.items():

        def split(predict_function=predict_function):
            local = timed("likelihood_local", predict_function)
            nonlocal_ = timed("likelihood_nonlocal", predict_function)

            def dispatch(*inner_args, **inner_kwargs):
                if inner_kwargs.get("is_local", False):
                    return local(*inner_args, **inner_kwargs)
                return nonlocal_(*inner_args, **inner_kwargs)

            return dispatch

        registry[name] = (fit_function, split())
    no_spike_original = no_spike.predict_no_spike_log_likelihood
    patched = {
        "no_spike": (no_spike, "predict_no_spike_log_likelihood", no_spike_original),
        "forward": (checkpointed, "_forward_chunk", checkpointed._forward_chunk),
        "backward": (checkpointed, "_backward_chunk", checkpointed._backward_chunk),
    }
    import non_local_detector.models.base as base

    no_spike_in_base = getattr(base, "predict_no_spike_log_likelihood", None)
    no_spike.predict_no_spike_log_likelihood = timed(
        "likelihood_no_spike", no_spike_original
    )
    if no_spike_in_base is not None:
        base.predict_no_spike_log_likelihood = no_spike.predict_no_spike_log_likelihood
    checkpointed._forward_chunk = timed("forward_filter_chunks", patched["forward"][2])
    checkpointed._backward_chunk = timed(
        "backward_smoother_chunks", patched["backward"][2]
    )
    savez, load = np.savez, np.load
    np.savez = timed("checkpoint_write", savez, synchronize=False)
    np.load = timed("checkpoint_read", load, synchronize=False)
    try:
        start = time.perf_counter()
        result = model.predict(**predict_kwargs)
        np.asarray(result["acausal_state_probabilities"])
        instrumented = time.perf_counter() - start
    finally:
        registry.update(originals)
        no_spike.predict_no_spike_log_likelihood = no_spike_original
        if no_spike_in_base is not None:
            base.predict_no_spike_log_likelihood = no_spike_in_base
        checkpointed._forward_chunk = patched["forward"][2]
        checkpointed._backward_chunk = patched["backward"][2]
        np.savez, np.load = savez, load

    stages = dict(seconds)
    stages["other"] = instrumented - sum(seconds.values())
    report = {
        "family": args.family,
        "duration_s": args.duration,
        "rows": len(predict_kwargs["time_edges"]) - 1,
        "bin_size_cm": args.bin_size,
        "arena_cm": args.arena,
        "population": args.population,
        "chunk_size": args.chunk_size,
        "backend": backend,
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
        **provenance(__file__),
        "host": platform.node(),
        "uninstrumented_warm_seconds": uninstrumented,
        "instrumented_seconds": instrumented,
        "stage_seconds": stages,
        "stage_fraction": {k: v / instrumented for k, v in stages.items()},
        "stage_calls": dict(calls),
        "host_functions_by_own_time": host_functions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    for name, value in sorted(stages.items(), key=lambda item: -item[1]):
        print(f"{name:26s} {value:9.3f} s  {100 * value / instrumented:5.1f}%")
    print(f"uninstrumented warm prediction: {uninstrumented:.3f} s")
    for entry in host_functions or []:
        print(
            f"  {entry['own_seconds']:8.3f} s own {entry['calls']:8d} calls  "
            f"{entry['function']}"
        )


if __name__ == "__main__":
    main()
