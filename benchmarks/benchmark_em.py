"""Time EM parameter estimation on the comparison workload.

Runs ``estimate_parameters`` twice in one process on a simulated session (the
``compare_prediction_modes.py`` encoding workload, decoded over its own
duration): the first call includes compilation, the second is warm. Reports
the EM iterations, marginal log-likelihoods, peak host RSS and peak device
memory. EM uses the dense filter/smoother, so memory grows with the session
length times the number of state bins.

Example::

    uv run python benchmarks/benchmark_em.py --family sorted --duration 60 \\
        --bin-size 4 --max-iter 3 --output /tmp/em-sorted
"""

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_prediction_modes import SAMPLE_RATE, workload  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--arena", type=float, default=180.0, help="cm")
    parser.add_argument("--bin-size", type=float, default=4.0, help="cm")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--spike-rate", type=float, default=None, help="Hz")
    parser.add_argument("--rate-spread", type=float, default=0.0)
    parser.add_argument("--mark-dimensions", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.population is None:
        args.population = 64 if args.family == "sorted" else 8
    if args.spike_rate is None:
        args.spike_rate = 5.0 if args.family == "sorted" else 20.0

    import jax

    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.encoding_duration = args.duration
    fit_kwargs, _ = workload(args)
    em_kwargs = {
        key: fit_kwargs[key]
        for key in ("position_time", "position", "spike_times")
        if key in fit_kwargs
    }
    if args.family == "clusterless":
        em_kwargs["spike_waveform_features"] = fit_kwargs["spike_waveform_features"]
    em_kwargs["time_edges"] = np.arange(int(round(args.duration * SAMPLE_RATE)) + 1) / (
        SAMPLE_RATE
    )
    detector = (
        NonLocalSortedSpikesDetector
        if args.family == "sorted"
        else NonLocalClusterlessDetector
    )
    algorithm = (
        {}
        if args.algorithm is None
        else {
            (
                "sorted_spikes_algorithm"
                if args.family == "sorted"
                else "clusterless_algorithm"
            ): args.algorithm
        }
    )

    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.02):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    report = {
        "family": args.family,
        "algorithm": args.algorithm,
        "duration_s": args.duration,
        "rows": len(em_kwargs["time_edges"]) - 1,
        "bin_size_cm": args.bin_size,
        "population": args.population,
        "spike_rate_hz": args.spike_rate,
        "rate_spread": args.rate_spread,
        "max_iter": args.max_iter,
        "n_chunks": args.n_chunks,
        "backend": backend,
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
        "host": platform.node(),
        "status": "started",
    }
    try:
        for label in ("first", "warm"):
            model = detector(
                environments=Environment(
                    place_bin_size=args.bin_size,
                    position_range=((0, args.arena), (0, args.arena)),
                ),
                infer_track_interior=False,
                **algorithm,
            )
            start = time.perf_counter()
            results = model.estimate_parameters(
                **em_kwargs, max_iter=args.max_iter, n_chunks=args.n_chunks
            )
            states = np.asarray(results["acausal_state_probabilities"])
            report[f"{label}_seconds"] = time.perf_counter() - start
        report["state_bins"] = int(model.state_ind_.shape[0])
        report["n_iter"] = int(getattr(model, "n_iter_", -1))
        report["marginal_log_likelihoods"] = [
            float(value)
            for value in np.atleast_1d(
                results.attrs.get("marginal_log_likelihoods", [])
            )
        ]
        np.save(args.output / "acausal_state_probabilities.npy", states)
        report["status"] = "ok"
    except Exception as error:  # record the failure mode (e.g. out of memory)
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:400]}"
    finally:
        stop.set()
        sampler.join()
        report["peak_host_rss_bytes"] = peak_rss[0]
        stats = (
            (jax.local_devices()[0].memory_stats() or {}) if backend != "cpu" else {}
        )
        report["peak_device_bytes"] = stats.get("peak_bytes_in_use")
        (args.output / "report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
