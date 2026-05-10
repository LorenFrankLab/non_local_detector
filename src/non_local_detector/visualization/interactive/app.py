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
        help=(
            "POSIX shorthand: add a named run from four colon-separated "
            "files. May be repeated for multi-run mode. Does NOT support "
            "paths containing ':' (Windows drive letters); use "
            "--run-files or --run-from-dir on Windows."
        ),
    )
    parser.add_argument(
        "--run-files",
        action="append",
        default=[],
        nargs=5,
        metavar=("NAME", "RESULTS.NC", "MODEL.PKL", "SPIKES.NPZ", "POSITION.PARQUET"),
        dest="run_files",
        help=(
            "Cross-platform: add a named run from five separate "
            "arguments. Each value is its own shell argument so paths "
            "containing ':' (e.g. Windows drive letters) work. "
            "May be repeated for multi-run mode."
        ),
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
    parser.add_argument(
        "--show-projected-2d",
        action="store_true",
        help=(
            "Show optional third column projecting the 1D decode onto the "
            "track graph in 2D. Experimental; graph-track decoders only."
        ),
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


def _parse_run_files_arg(values: list[str]) -> dict[str, str]:
    """Translate ``--run-files NAME RESULTS MODEL SPIKES POSITION`` to a spec.

    The argparse ``nargs=5`` already enforces exactly five values; this
    function validates that none is empty (an empty ``name`` would
    masquerade as a different run later).
    """
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            f"--run-files expects exactly 5 values "
            f"(name, results.nc, model.pkl, spikes.npz, position.parquet); "
            f"got {len(values)}: {values!r}"
        )
    name, results, model, spikes, position = values
    if not name:
        raise argparse.ArgumentTypeError("--run-files NAME must be non-empty")
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
    (rare on POSIX but legal). The bundle's canonical input is
    ``results.nc`` plus the three sidecars (``model.pkl``,
    ``spikes.npz``, ``position.parquet``); raises with the missing
    file names so the user knows which to add. ``results.zarr/`` is
    optional and consulted by ``_load_run`` for chunked acceleration —
    it doesn't satisfy the canonical-input requirement on its own.
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
    spec = {
        "name": name,
        "bundle_dir": str(bundle_dir),
        **{key: str(p) for key, p in paths.items()},
    }
    zarr_path = bundle_dir / "results.zarr"
    if zarr_path.is_dir():
        spec["zarr_cache"] = str(zarr_path)
    return spec


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
      ``y_position`` (2D, v3+). If ``position`` and
      ``x_position``/``y_position`` are all present, ``position`` is
      treated as the 1D decoder coordinate and the XY columns are
      exposed to the optional projected-2D panel. Optional:
      ``speed``.
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
    if spec.get("zarr_cache"):
        # Optional acceleration: the bundle ships a derived
        # ``results.zarr/`` cache next to the canonical ``results.nc``.
        # Validate shape / time grid before using; fail loud on a
        # stale cache instead of silently rendering misaligned data.
        # When the zarr backend isn't installed (``[viewer]`` without
        # ``[viewer-cache]``), fall back to the canonical NetCDF with
        # a one-line warning rather than aborting — the cache is meant
        # to be a transparent acceleration, not a hard requirement.
        import warnings

        from non_local_detector.visualization.interactive.data_source_zarr import (
            load_zarr_cache_or_fall_back,
        )

        try:
            results = load_zarr_cache_or_fall_back(
                zarr_path=Path(spec["zarr_cache"]),
                canonical_path=Path(spec["results"]),
                canonical_results=results,
            )
        except ImportError as exc:
            warnings.warn(
                f"results.zarr/ found at {spec['zarr_cache']!s} but the "
                f"zarr backend is unavailable ({exc}); falling back to "
                "results.nc. Install the cache backend with `pip install "
                "'non_local_detector[viewer-cache]'` to use the cache.",
                stacklevel=2,
            )
    detector = _DetectorBase.load_model(spec["model"])
    spike_times_npz = np.load(spec["spikes"], allow_pickle=True)
    spike_times = list(spike_times_npz["spike_times"])
    position_df = pd.read_parquet(spec["position"])
    sibling_2d = Path(spec["position"]).with_name("position_2d.parquet")
    if "position" in position_df.columns:
        position = position_df["position"].to_numpy()
        if sibling_2d.exists():
            # Separate parquet sidecar — used when position_2d sampled
            # on its own time grid (camera clock vs. linearization output).
            position_2d_df = pd.read_parquet(sibling_2d)
            position_2d = position_2d_df[["x_position", "y_position"]].to_numpy()
            position_2d_time = position_2d_df.index.to_numpy()
        elif {"x_position", "y_position"}.issubset(position_df.columns):
            # Co-muxed columns — same time grid as the linearized position.
            position_2d = position_df[["x_position", "y_position"]].to_numpy()
            position_2d_time = position_df.index.to_numpy()
        else:
            position_2d = None
            position_2d_time = None
    elif {"x_position", "y_position"}.issubset(position_df.columns):
        position = position_df[["x_position", "y_position"]].to_numpy()
        position_2d = None
        position_2d_time = None
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
        position_2d=position_2d,
        position_2d_time=position_2d_time,
        speed=speed,
    )


def _zarr_direct_eligible(spec: dict[str, str]) -> bool:
    """A spec qualifies for the zarr-direct path iff it carries a
    valid ``results.zarr/`` (set by ``_parse_run_from_dir_arg`` only).

    ``--run`` and ``--run-files`` never carry one; they ship four
    explicit paths and never imply a directory layout.
    """
    return "zarr_cache" in spec and "bundle_dir" in spec


def _resolve_data_source(
    specs: list[dict[str, str]],
):
    """Build the right ``DecoderDataSource`` for ``specs``.

    Three branches:

    1. Every spec is zarr-direct-eligible AND the zarr backend
       imports cleanly → ``ZarrDirectDecoderDataSource.for_directories``.
       This is the perf-win path (Phase 5.2).
    2. Mixed (some zarr-eligible, some not) → emit ``UserWarning``,
       fall back to eagerly loading every spec via ``_load_run`` and
       wrapping in ``InMemoryDecoderDataSource``. Mixed-mode
       one-data-source-per-run is harder than this phase warrants.
    3. None zarr-eligible OR zarr import fails → eager
       ``InMemoryDecoderDataSource``. (The ``_load_run`` path's
       existing ``ImportError`` fallback already emits its own
       ``UserWarning`` when the cache is present but the backend is
       missing.)
    """
    import warnings

    from non_local_detector.visualization.interactive.data_source import (
        InMemoryDecoderDataSource,
    )

    eligible = [s for s in specs if _zarr_direct_eligible(s)]
    all_eligible = len(eligible) == len(specs)

    if all_eligible and eligible:
        try:
            from non_local_detector.visualization.interactive.data_source_zarr_direct import (
                ZarrDirectDecoderDataSource,
            )

            dirs = {spec["name"]: Path(spec["bundle_dir"]) for spec in specs}
            return ZarrDirectDecoderDataSource.for_directories(dirs)
        except ImportError as exc:
            warnings.warn(
                f"All --run-from-dir bundles have results.zarr/ caches but "
                f"the zarr backend is unavailable ({exc}); falling back to "
                "eager InMemoryDecoderDataSource. Install the cache backend "
                "with `pip install 'non_local_detector[viewer-cache]'` to "
                "use the direct-zarr path.",
                stacklevel=2,
            )

    if eligible and not all_eligible:
        warnings.warn(
            "Mixed CLI: some runs carry a results.zarr/ cache but others "
            "don't. The viewer is degrading the cached runs to in-memory "
            "loads so a single InMemoryDecoderDataSource can hold every "
            "bundle. Use the build-viewer-cache devtool on the remaining "
            "runs (`python -m non_local_detector.visualization.interactive"
            ".devtools build-viewer-cache --run-dir <dir>`) to enable "
            "the direct-zarr path for the whole session.",
            stacklevel=2,
        )

    bundles: dict[str, object] = {}
    for spec in specs:
        name, bundle = _load_run(spec)
        bundles[name] = bundle
    return InMemoryDecoderDataSource(bundles)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns the process exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.run and not args.run_from_dir and not args.run_files:
        parser.error(
            "at least one --run / --run-files / --run-from-dir argument is required"
        )

    specs: list[dict[str, str]] = []
    seen_names: set[str] = set()

    def _add_spec(spec: dict[str, str]) -> None:
        name = spec["name"]
        if name in seen_names:
            parser.error(f"duplicate run name {name!r}")
        seen_names.add(name)
        specs.append(spec)

    for arg in args.run:
        try:
            _add_spec(_parse_run_arg(arg))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    for values in args.run_files:
        try:
            _add_spec(_parse_run_files_arg(values))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    for arg in args.run_from_dir:
        try:
            _add_spec(_parse_run_from_dir_arg(arg))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    data_source = _resolve_data_source(specs)

    if args.backend == "qt":
        from non_local_detector.visualization.interactive.viewer.qt import (
            launch_qt_with_source,
        )

        return launch_qt_with_source(
            data_source,
            t_width=args.t_width,
            show_projected_2d=args.show_projected_2d,
        )
    parser.error(f"unsupported backend {args.backend!r}")
    return 1


_HELP_EPILOG = """\
For sessions over ~1 hour, prefer `--run-from-dir` plus
`build-viewer-cache` for fluid scroll/playback — the cache lets the
viewer read each window directly from a chunked zarr array via
ZarrDirectDecoderDataSource (Phase 5 perf path), which is markedly
faster than the eager NetCDF path.

  Build the cache once:
    python -m non_local_detector.visualization.interactive.devtools \\
      build-viewer-cache --run-dir bundles/continuous/

  Then launch via the cached path (preferred):
    python -m non_local_detector.visualization.interactive \\
      --run-from-dir continuous:bundles/continuous/

Examples:

  Single run from a bundle directory:
    python -m non_local_detector.visualization.interactive \\
      --run-from-dir continuous:bundles/continuous/

  Single run from explicit files (cross-platform; safe for Windows
  drive-letter paths):
    python -m non_local_detector.visualization.interactive \\
      --run-files default results.nc model.pkl spikes.npz position.parquet

  Single run from explicit files (POSIX shorthand — does NOT support
  paths containing ':'):
    python -m non_local_detector.visualization.interactive \\
      --run default:results.nc:model.pkl:spikes.npz:position.parquet

  Multi-run model swap:
    python -m non_local_detector.visualization.interactive \\
      --run-from-dir continuous:bundles/continuous/ \\
      --run-from-dir contfrag:bundles/contfrag/

The direct-zarr path requires every spec on the command line to be
a `--run-from-dir` with a valid `results.zarr/` cache. Mixing
zarr-cached runs with non-cached runs (e.g. one `--run-from-dir`
plus one `--run`) is supported but emits a `UserWarning` and
degrades every run to the eager InMemoryDecoderDataSource —
build the cache for the remaining runs to use the direct-zarr path
across the whole session.

`--run-from-dir` requires a NetCDF results bundle (`results.nc` plus
`model.pkl`, `spikes.npz`, `position.parquet`). When a `results.zarr/`
acceleration cache sits next to `results.nc` and the zarr backend is
installed (the `[viewer-cache]` extra), the viewer reads large per-time
arrays lazily through the cache; otherwise it falls back to `results.nc`
with a warning. Build the cache with:

  python -m non_local_detector.visualization.interactive.devtools \\
    build-viewer-cache --run-dir bundles/continuous/

The cache is validated against `results.nc` on load (time grid + every
time-dim variable); a stale cache is rejected with a rebuild hint.

For optional `predict()` outputs (log_likelihood, predictive_posterior),
re-run `predict(return_outputs=['log_likelihood', 'predictive_posterior'])`
and rebuild the bundle. Panels that need missing arrays disable
themselves with a clear title-bar message.
"""
