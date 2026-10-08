"""Time the per-step forward/backward kernels of checkpointed prediction.

Builds the restricted, fused structured transition operator that checkpointed
prediction uses for the ``compare_prediction_modes.py`` non-local detector
workload and times the jitted forward-filter and
backward-smoother chunk kernels on random inputs. Reports warm seconds per
step and the fusions and custom calls (e.g. cuBLAS) launched per step by the
compiled scan body. ``--xprof-dir`` additionally records an xprof trace of one
warm forward and one backward chunk for kernel-level timing.

Example::

    uv run python benchmarks/benchmark_checkpoint_scan_steps.py \\
        --chunk-size 4096 --output /tmp/scan-steps.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_prediction_modes import workload  # noqa: E402


def count_kernels(compiled_text):
    """Count fusions and custom calls (e.g. cuBLAS) in the compiled scan body.

    These are the kernels launched once per scan step.
    """
    computations = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"^(%[\w.\-]+) \(.*?\{\n(.*?)^\}", compiled_text, re.M | re.S
        )
    }
    bodies = re.findall(r"while\(.*?body=(%[\w.\-]+)", compiled_text)
    text = "\n".join(computations.get(body, "") for body in bodies)
    return {
        "scan_body_fusions": len(re.findall(r" fusion\(", text)),
        "scan_body_custom_calls": len(re.findall(r" custom-call\(", text)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument("--arena", type=float, default=180.0)
    parser.add_argument("--bin-size", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--xprof-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )
    from non_local_detector.checkpointed_inference import (
        _backward_chunk,
        _forward_chunk,
    )

    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    workload_args = argparse.Namespace(
        family=args.family,
        duration=1.0,
        encoding_duration=10.0,
        arena=args.arena,
        bin_size=args.bin_size,
        population=64 if args.family == "sorted" else 8,
        spike_rate=5.0 if args.family == "sorted" else 20.0,
        mark_dimensions=4,
    )
    fit_kwargs, _ = workload(workload_args)
    fit_kwargs["transition_representation"] = "structured"
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
    # The same restricted, bound operator that checkpointed prediction uses.
    operator = jax.device_put(
        model._continuous_transition_operator_.restricted(
            model.is_track_interior_state_bins_
        )
        .bind_discrete(np.asarray(model.discrete_state_transitions_))
        .fused()
    )
    n_bins, rows = operator.n_bins, args.chunk_size
    rng = np.random.default_rng(0)
    likelihoods = jnp.asarray(rng.normal(size=(rows, n_bins)), dtype=jnp.float32)
    initial = jnp.full(n_bins, 1 / n_bins, dtype=jnp.float32)
    evidence = jnp.zeros((), jnp.float32)

    def forward():
        return _forward_chunk(
            initial,
            evidence,
            likelihoods,
            None,
            operator,
            keep_rows=True,
            keep_predictive=False,
        )

    filtered = jax.block_until_ready(forward())[1][0]
    row_ids = jnp.arange(rows)

    def backward():
        return _backward_chunk(filtered, None, operator, filtered[-1], row_ids, rows)

    report = {
        "family": args.family,
        "state_bins": int(n_bins),
        "state_sizes": [int(size) for size in operator.state_sizes],
        "chunk_size": rows,
        "backend": backend,
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
    }
    for name, function, lowered in (
        (
            "forward",
            forward,
            _forward_chunk.lower(
                initial,
                evidence,
                likelihoods,
                None,
                operator,
                keep_rows=True,
                keep_predictive=False,
            ),
        ),
        (
            "backward",
            backward,
            _backward_chunk.lower(
                filtered, None, operator, filtered[-1], row_ids, rows
            ),
        ),
    ):
        jax.block_until_ready(function())
        seconds = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            jax.block_until_ready(function())
            seconds.append(time.perf_counter() - start)
        kernels = count_kernels(lowered.compile().as_text())
        per_step = float(np.median(seconds)) / rows
        report[name] = {
            "warm_seconds": seconds,
            "median_seconds_per_step": per_step,
            **kernels,
        }
        print(
            f"{name:8s} {1e6 * per_step:8.1f} us/step  fusions "
            f"{kernels['scan_body_fusions']:4d}  custom calls "
            f"{kernels['scan_body_custom_calls']:3d}"
        )
    if args.xprof_dir is not None:
        with jax.profiler.trace(str(args.xprof_dir)):
            jax.block_until_ready(forward())
            jax.block_until_ready(backward())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
