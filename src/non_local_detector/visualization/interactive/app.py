"""CLI entry point for the interactive viewer.

Argparse + lazy dispatch — no Qt imports at module top level. The
``--backend qt`` branch lazy-imports ``viewer.qt.launch_qt`` only
after argparse decides which backend to use, so the package shell
stays GUI-toolkit-free.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# The four files a viewer bundle directory must contain. Bundles
# emitted by the ``devtools/bundle_from_statespacecheck.py`` CLI use
# this layout, and ``--run-from-dir`` expands a single dir argument
# into the four explicit paths ``--run`` consumes.
_BUNDLE_FILENAMES = {
    "results": "results.nc",
    "model": "model.pkl",
    "spikes": "spikes.npz",
    "position": "position.parquet",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m non_local_detector.visualization.interactive",
        description=(
            "Interactive viewer for non_local_detector decoder output. "
            "Loads one or more named runs and opens a Qt window with a "
            "posterior-heatmap panel + center-time slider. v1 ships the "
            "Qt backend only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP_EPILOG,
    )
    parser.add_argument(
        "--backend",
        choices=["qt"],
        default="qt",
        help="Frontend backend (default: qt; v2 will add panel).",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME:RESULTS.NC:MODEL.PKL:SPIKES.NPZ:POSITION.PARQUET",
        help=("Add a named run from four files. May be repeated for multi-run mode."),
    )
    parser.add_argument(
        "--run-from-dir",
        action="append",
        default=[],
        metavar="NAME:DIR/",
        dest="run_from_dir",
        help=(
            "Add a named run from a bundle directory containing "
            "results.nc + model.pkl + spikes.npz + position.parquet. "
            "May be repeated. Accepts the directory layout emitted by "
            "the bundle-from-statespacecheck-cache devtool."
        ),
    )
    parser.add_argument(
        "--t-width",
        type=float,
        default=1.0,
        help="Initial window width in seconds (default: 1.0).",
    )
    return parser


def _parse_run_arg(arg: str) -> dict[str, str]:
    """Split a ``name:results:model:spikes:position`` arg into a dict."""
    parts = arg.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--run expects exactly 5 colon-separated fields "
            f"(name:results.nc:model.pkl:spikes.npz:position.parquet); "
            f"got {len(parts)}: {arg!r}"
        )
    name, results, model, spikes, position = parts
    return {
        "name": name,
        "results": results,
        "model": model,
        "spikes": spikes,
        "position": position,
    }


def _parse_run_from_dir_arg(arg: str) -> dict[str, str]:
    """Split a ``name:dir/`` arg into the canonical five-field run spec.

    Splits on the *first* colon so directory paths may contain colons
    (rare on POSIX but legal). Validates the directory exists and
    contains the four bundle files; raises with the missing names so
    the user knows which file to add.
    """
    name, sep, dir_str = arg.partition(":")
    if not sep or not name or not dir_str:
        raise argparse.ArgumentTypeError(
            f"--run-from-dir expects 'name:dir/'; got {arg!r}"
        )
    bundle_dir = Path(dir_str)
    if not bundle_dir.is_dir():
        raise argparse.ArgumentTypeError(
            f"--run-from-dir bundle directory does not exist: {bundle_dir}"
        )
    paths = {key: bundle_dir / fname for key, fname in _BUNDLE_FILENAMES.items()}
    missing = [str(p.name) for p in paths.values() if not p.exists()]
    if missing:
        raise argparse.ArgumentTypeError(
            f"--run-from-dir {bundle_dir} is missing required bundle files: "
            f"{missing!r}. Expected: {sorted(_BUNDLE_FILENAMES.values())!r}"
        )
    return {"name": name, **{key: str(p) for key, p in paths.items()}}


def _load_run(spec: dict[str, str]):
    """Load one ``RunBundle`` from the four CLI files. Lazy-imports.

    Schema:

    - ``results.nc``: NetCDF written via
      ``_DetectorBase.save_results``. Required variables:
      ``acausal_posterior``, ``acausal_state_probabilities``.
      Optional: ``log_likelihood``, ``predictive_posterior``.
    - ``model.pkl``: pickled fitted detector written via
      ``_DetectorBase.save_model`` (``SortedSpikesDecoder`` /
      ``ContFragSortedSpikesClassifier`` /
      ``NoSpikeContFragSortedSpikesClassifier`` /
      ``NonLocalSortedSpikesDetector``).
    - ``spikes.npz``: ``np.savez`` with one object-dtype array
      ``spike_times`` of shape ``(n_neurons,)``, each entry an
      ``np.ndarray`` of float64 absolute spike timestamps.
    - ``position.parquet``: pandas DataFrame written via
      ``df.to_parquet(...)``. Index = absolute time in seconds.
      Required column: ``position`` (1D) or ``x_position`` +
      ``y_position`` (2D, v3+). Optional: ``speed``.
    """
    import numpy as np
    import pandas as pd

    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.view_models.base import (
        RunBundle,
    )

    # Use the canonical save/load pair so the state_bins MultiIndex
    # gets re-attached on load (xr.open_dataset alone drops it).
    results = _DetectorBase.load_results(spec["results"])
    detector = _DetectorBase.load_model(spec["model"])
    spike_times_npz = np.load(spec["spikes"], allow_pickle=True)
    spike_times = list(spike_times_npz["spike_times"])
    position_df = pd.read_parquet(spec["position"])
    if "position" in position_df.columns:
        position = position_df["position"].to_numpy()
    elif {"x_position", "y_position"}.issubset(position_df.columns):
        position = position_df[["x_position", "y_position"]].to_numpy()
    else:
        raise ValueError(
            f"position.parquet at {spec['position']!r} must contain "
            "either a 'position' column (1D) or 'x_position' + "
            "'y_position' columns (2D)."
        )
    position_time = position_df.index.to_numpy()
    speed = position_df["speed"].to_numpy() if "speed" in position_df.columns else None
    return spec["name"], RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=position_time,
        position=position,
        speed=speed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns the process exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.run and not args.run_from_dir:
        parser.error("at least one --run or --run-from-dir argument is required")

    bundles_dict = {}
    for raw_arg, parse_fn in (
        *((arg, _parse_run_arg) for arg in args.run),
        *((arg, _parse_run_from_dir_arg) for arg in args.run_from_dir),
    ):
        try:
            spec = parse_fn(raw_arg)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        name, bundle = _load_run(spec)
        if name in bundles_dict:
            parser.error(f"duplicate run name {name!r}")
        bundles_dict[name] = bundle

    if args.backend == "qt":
        from non_local_detector.visualization.interactive.viewer.qt import (
            launch_qt,
        )

        return launch_qt(bundles_dict, t_width=args.t_width)
    parser.error(f"unsupported backend {args.backend!r}")
    return 1


_HELP_EPILOG = """\
Examples:

  Single run from explicit files:
    python -m non_local_detector.visualization.interactive \\
      --run default:results.nc:model.pkl:spikes.npz:position.parquet

  Single run from a bundle directory (e.g. devtool output):
    python -m non_local_detector.visualization.interactive \\
      --run-from-dir continuous:bundles/continuous/

  Multi-run model swap:
    python -m non_local_detector.visualization.interactive \\
      --run-from-dir continuous:bundles/continuous/ \\
      --run-from-dir contfrag:bundles/contfrag/

For optional `predict()` outputs (log_likelihood, predictive_posterior),
re-run `predict(return_outputs=['log_likelihood', 'predictive_posterior'])`
and rebuild the bundle. Panels that need missing arrays disable
themselves with a clear title-bar message.
"""
