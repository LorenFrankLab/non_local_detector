"""Compare a fitted detector's pickle with the saved-model format.

Loads a detector pickle (as ``load_model`` loads legacy files), saves it with
``save_model``, and reports for both formats the file size and the median
save and load times, the time ``read_model_summary`` takes to read the place
fields and grids, and, given the inputs of a decode, the largest difference
between the posteriors the two loaded detectors decode.

Usage::

    python benchmarks/compare_model_formats.py detector.pkl \\
        --predict-inputs predict_kwargs.pkl --repeats 3

``predict_kwargs.pkl`` is a pickled dict of the keyword arguments of the
detector's ``predict`` (``spike_times``, ``time_edges``, ``position_time``,
``position``, and ``spike_waveform_features`` for clusterless models), for
example ``pickle.dump(dict(spike_times=..., time_edges=...), file)``. Without
it the decode comparison is skipped. Only pass pickles you trust.
"""

import argparse
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _median_seconds(function, repeats):
    """Median wall time of ``function()`` over ``repeats`` calls, and its result."""
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def main(argv=None):
    """Run the comparison and print a table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pickle", type=Path, help="A fitted detector pickle.")
    parser.add_argument(
        "--predict-inputs",
        type=Path,
        help="A pickled dict of predict keyword arguments (optional).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats (default 3)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the converted model (default: a temporary file).",
    )
    args = parser.parse_args(argv)

    from non_local_detector.model_format import read_model_summary
    from non_local_detector.models.base import _DetectorBase

    with tempfile.TemporaryDirectory() as directory:
        new_path = args.output or Path(directory) / "model.nld"
        pickle_copy = Path(directory) / "model.pkl"

        pickle_load, detector = _median_seconds(
            lambda: _DetectorBase.load_model(args.pickle), args.repeats
        )

        def write_pickle():
            with pickle_copy.open("wb") as file:
                pickle.dump(detector, file, protocol=pickle.HIGHEST_PROTOCOL)

        pickle_save, _ = _median_seconds(write_pickle, args.repeats)
        new_save, _ = _median_seconds(
            lambda: detector.save_model(new_path), args.repeats
        )
        new_load, loaded = _median_seconds(
            lambda: _DetectorBase.load_model(new_path), args.repeats
        )
        summary_time, _ = _median_seconds(
            lambda: read_model_summary(new_path), args.repeats
        )
        rows = [
            (
                "size (MB)",
                args.pickle.stat().st_size / 1e6,
                new_path.stat().st_size / 1e6,
            ),
            ("save (s)", pickle_save, new_save),
            ("load (s)", pickle_load, new_load),
            ("inspect (s)", pickle_load, summary_time),
        ]
        print(f"{type(detector).__name__}: {args.pickle}")
        print(f"{'':14}{'pickle':>12}{'saved model':>14}")
        for name, old, new in rows:
            print(f"{name:14}{old:12.4g}{new:14.4g}")
        print(
            "(inspect: a pickle must be loaded whole; read_model_summary for the new format)"
        )

        if args.predict_inputs is None:
            print("decode comparison skipped (no --predict-inputs)")
            return 0
        predict_kwargs = pickle.loads(args.predict_inputs.read_bytes())
        expected = detector.predict(**predict_kwargs)
        actual = loaded.predict(**predict_kwargs)
        for name in ("acausal_posterior", "acausal_state_probabilities"):
            difference = np.nanmax(
                np.abs(np.asarray(actual[name]) - np.asarray(expected[name]))
            )
            print(f"max |difference| of {name}: {difference:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
