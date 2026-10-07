"""Check that repeated clusterless likelihood evaluations are bitwise identical.

Fits each clusterless algorithm on two small synthetic electrodes, evaluates
the detector log likelihood (local and non-local states) for rows 2-12
repeatedly, and reports the distinct SHA-256 digests per algorithm. A
collision-heavy case packs hundreds of decoding spikes into each bin. The
first digest of each case is recorded so separate runs can be compared, and
the exit status is nonzero unless every case has exactly one digest. Checkpointed replay requires exactly one.
GPU runs should select an idle device with CUDA_VISIBLE_DEVICES and set
XLA_PYTHON_CLIENT_PREALLOCATE=false.

Example: python scripts/check_likelihood_determinism.py --output REPORT.json
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # quiet per-electrode progress bars

import jax  # noqa: E402
import numpy as np  # noqa: E402

ALGORITHMS = (
    "clusterless_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
)


def build(algorithm, n_encoding=72, n_decoding=51, n_electrodes=2, crowded=False):
    """Fit one algorithm; return a function evaluating rows 2-12."""
    from non_local_detector import Environment, NonLocalClusterlessDetector

    rng = np.random.default_rng(7412)
    position_time = np.linspace(0, 2, 201)
    position = (4 + 3 * np.sin(position_time * 3))[:, None]
    spikes = [np.sort(rng.uniform(0, 2, n_encoding)) for _ in range(n_electrodes)]
    marks = [rng.normal(size=(n_encoding, 4)) for _ in range(n_electrodes)]
    model = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=0.5, position_range=((0, 8),)),
        infer_track_interior=False,
        clusterless_algorithm=algorithm,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(position_time, position, spikes, marks)
    # Crowded spikes fall in two 0.1 s bins; otherwise they span all rows.
    low, high = (0.4, 0.6) if crowded else (0, 1.6)
    decode_spikes = [
        np.sort(rng.uniform(low, high, n_decoding)) for _ in range(n_electrodes)
    ]
    decode_marks = [rng.normal(size=(n_decoding, 4)) for _ in range(n_electrodes)]
    # Same bin, unequal marks; this also makes the first train unsorted.
    decode_spikes[0][:3] = [0.41, 0.42, 0.43]
    edges = np.linspace(0, 1.6, 17)

    def evaluate():
        return np.asarray(
            model.compute_log_likelihood(
                position_time,
                position,
                decode_spikes,
                decode_marks,
                row_slice=slice(2, 12),
                time_edges=edges,
            )
        )

    return evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--require-backend", choices=("cpu", "gpu"), default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "check_likelihood_determinism.json",
    )
    args = parser.parse_args()
    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    cases = []
    for algorithm, (label, n_decoding, crowded) in [
        (algorithm, case)
        for algorithm in ALGORITHMS
        for case in (("default", 51, False), ("collision_heavy", 2000, True))
    ]:
        evaluate = build(algorithm, n_decoding=n_decoding, crowded=crowded)
        start = time.perf_counter()
        values = [evaluate() for _ in range(args.repetitions)]
        digests = [hashlib.sha256(value.tobytes()).hexdigest() for value in values]
        with np.errstate(invalid="ignore"):
            difference = max(
                float(np.nanmax(np.abs(value - values[0]))) for value in values
            )
        case = {
            "algorithm": algorithm,
            "case": label,
            "decoding_spikes_per_electrode": n_decoding,
            "shape": list(values[0].shape),
            "dtype": str(values[0].dtype),
            "distinct_sha256": len(set(digests)),
            "first_sha256": digests[0],
            "max_abs_difference": difference,
            "seconds": time.perf_counter() - start,
        }
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "x64": jax.config.x64_enabled,
        "repetitions": args.repetitions,
        "all_bitwise_identical": all(case["distinct_sha256"] == 1 for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(
        f"all bitwise identical: {report['all_bitwise_identical']}; wrote {args.output}"
    )
    return 0 if report["all_bitwise_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
