"""CLI entry point for the interactive viewer.

Argparse + lazy dispatch — no Qt imports at module top level. The
``--backend qt`` branch lazy-imports ``viewer.qt.launch_qt`` only
after argparse decides which backend to use, so the package shell
stays GUI-toolkit-free.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


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


def _load_run(spec: dict[str, str]):
    """Load one ``RunBundle`` from the four CLI files. Lazy-imports.

    Schema:

    - ``results.nc``: NetCDF written from ``predict()`` output. Required
      variables: ``acausal_posterior``, ``acausal_state_probabilities``.
      Optional: ``log_likelihood``, ``predictive_posterior``.
    - ``model.pkl``: joblib-pickled fitted detector
      (``SortedSpikesDecoder`` /
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
    import joblib  # type: ignore[import-untyped]
    import numpy as np
    import pandas as pd
    import xarray as xr

    from non_local_detector.visualization.interactive.view_models.base import (
        RunBundle,
    )

    results = xr.open_dataset(spec["results"])
    detector = joblib.load(spec["model"])
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

    if not args.run:
        parser.error("at least one --run argument is required")

    bundles_dict = {}
    for raw_arg in args.run:
        spec = _parse_run_arg(raw_arg)
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

  Single run:
    python -m non_local_detector.visualization.interactive \\
      --run default:results.nc:model.pkl:spikes.npz:position.parquet

  Multi-run model swap:
    python -m non_local_detector.visualization.interactive \\
      --run nl:nl_results.nc:nl_model.pkl:spikes.npz:position.parquet \\
      --run cf:cf_results.nc:cf_model.pkl:spikes.npz:position.parquet

For optional `predict()` outputs (log_likelihood, predictive_posterior),
re-run `predict(return_outputs=['log_likelihood', 'predictive_posterior'])`
and rebuild the bundle. Panels that need missing arrays disable
themselves with a clear title-bar message.
"""
