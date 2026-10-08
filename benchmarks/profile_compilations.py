"""Attribute XLA compilations during a first checkpointed prediction.

Fits a detector on the ``compare_prediction_modes.py`` workload, then times a
first checkpointed compact prediction while logging every XLA compilation by
function, followed by three warm predictions. Reports the compile count and
time per function and saves the smoothed state probabilities for comparison
between source trees.

Example::

    uv run python benchmarks/profile_compilations.py --family clusterless \\
        --algorithm clusterless_gmm --encoding-duration 120 --output /tmp/gmm
"""

import argparse
import collections
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_prediction_modes import workload  # noqa: E402

COMPILE_MESSAGE = re.compile(r"Finished XLA compilation of (\S+) in ([0-9.e-]+) sec")


class CompileLog(logging.Handler):
    """Collect XLA compile seconds and counts by function name."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.seconds = collections.defaultdict(float)
        self.counts = collections.Counter()

    def emit(self, record):
        match = COMPILE_MESSAGE.search(record.getMessage())
        if match:
            self.seconds[match.group(1)] += float(match.group(2))
            self.counts[match.group(1)] += 1

    def clear(self):
        self.seconds.clear()
        self.counts.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sorted", "clusterless"], default="sorted")
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds")
    parser.add_argument("--encoding-duration", type=float, default=10.0)
    parser.add_argument("--arena", type=float, default=180.0, help="cm")
    parser.add_argument("--bin-size", type=float, default=2.0, help="cm")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--spike-rate", type=float, default=None, help="Hz")
    parser.add_argument("--rate-spread", type=float, default=0.0)
    parser.add_argument("--mark-dimensions", type=int, default=4)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.population is None:
        args.population = 64 if args.family == "sorted" else 8
    if args.spike_rate is None:
        args.spike_rate = 5.0 if args.family == "sorted" else 20.0
    args.output.mkdir(parents=True, exist_ok=True)

    import jax

    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    compile_log = CompileLog()
    for name in (
        "jax._src.dispatch",
        "jax._src.interpreters.pxla",
        "jax._src.compiler",
    ):
        logger = logging.getLogger(name)
        logger.addHandler(compile_log)
        logger.setLevel(logging.DEBUG)
    jax.config.update("jax_log_compiles", True)

    fit_kwargs, predict_kwargs = workload(args)
    fit_kwargs["transition_representation"] = "structured"
    predict_kwargs.update(
        inference_mode="checkpointed",
        output_mode="compact",
        checkpoint_dir=args.output / "checkpoints",
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
    model = detector(
        environments=Environment(
            place_bin_size=args.bin_size,
            position_range=((0, args.arena), (0, args.arena)),
        ),
        infer_track_interior=False,
        **algorithm,
    )
    start = time.perf_counter()
    model.fit(**fit_kwargs)
    fit_seconds = time.perf_counter() - start

    compile_log.clear()
    start = time.perf_counter()
    states = np.asarray(model.predict(**predict_kwargs).acausal_state_probabilities)
    first_seconds = time.perf_counter() - start
    first_counts = dict(compile_log.counts)
    first_seconds_by_function = dict(compile_log.seconds)
    warm = []
    for _ in range(args.warm_repeats):
        start = time.perf_counter()
        np.asarray(model.predict(**predict_kwargs).acausal_state_probabilities)
        warm.append(time.perf_counter() - start)
    np.save(args.output / "acausal_state_probabilities.npy", states)

    top = sorted(first_seconds_by_function.items(), key=lambda item: -item[1])[:10]
    report = {
        "family": args.family,
        "algorithm": args.algorithm,
        "duration_s": args.duration,
        "encoding_duration_s": args.encoding_duration,
        "bin_size_cm": args.bin_size,
        "population": args.population,
        "spike_rate_hz": args.spike_rate,
        "rate_spread": args.rate_spread,
        "backend": jax.default_backend(),
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "fit_seconds": fit_seconds,
        "first_predict_seconds": first_seconds,
        "first_compilations": sum(first_counts.values()),
        "first_compile_seconds": sum(first_seconds_by_function.values()),
        "top_compiles": [[name, first_counts[name], value] for name, value in top],
        "warm_seconds": warm,
        "warm_compilations": sum(compile_log.counts.values())
        - sum(first_counts.values()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
