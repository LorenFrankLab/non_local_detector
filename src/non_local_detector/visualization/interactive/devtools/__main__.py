"""CLI dispatch for ``python -m non_local_detector.visualization.interactive.devtools``.

Subcommands:

- ``bundle-from-statespacecheck-cache``: build a viewer bundle directory
  from the upstream ``statespacecheck-paper-viewer`` cache + intermediates
  layout. See ``bundle_from_statespacecheck_cache`` for parameter details.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from non_local_detector.visualization.interactive.devtools.bundle_from_statespacecheck import (  # noqa: E501
    MODEL_NAMES,
    bundle_from_statespacecheck_cache,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m non_local_detector.visualization.interactive.devtools",
        description="Devtools for building viewer bundles from external sources.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sub = subparsers.add_parser(
        "bundle-from-statespacecheck-cache",
        help=(
            "Bundle a statespacecheck cache + intermediates layout into a "
            "CLI-compatible viewer bundle directory."
        ),
    )
    sub.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing upstream's figure-4 sidecars.",
    )
    sub.add_argument(
        "--intermediates-dir",
        type=Path,
        required=True,
        help="Directory containing the source NetCDF + pickled fitted detector.",
    )
    sub.add_argument(
        "--model",
        choices=list(MODEL_NAMES),
        required=True,
        help="Upstream model key (continuous or contfrag).",
    )
    sub.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output bundle directory (created if missing).",
    )
    sub.add_argument(
        "--results-nc",
        type=Path,
        default=None,
        help="Override for the auto-resolved source NetCDF path.",
    )
    sub.add_argument(
        "--model-pkl",
        type=Path,
        default=None,
        help="Override for the auto-resolved source detector-pickle path.",
    )
    sub.add_argument(
        "--results-from-zarr",
        action="store_true",
        help=(
            "Substitute the per-model Zarr in --cache-dir for the source "
            "NetCDF. Required vars are revalidated; refuses to run if absent."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "bundle-from-statespacecheck-cache":
        out = bundle_from_statespacecheck_cache(
            cache_dir=args.cache_dir,
            intermediates_dir=args.intermediates_dir,
            model=args.model,
            out=args.out,
            results_nc=args.results_nc,
            model_pkl=args.model_pkl,
            results_from_zarr=args.results_from_zarr,
        )
        print(f"Wrote bundle directory: {out}")
        return 0
    parser.error(f"unknown subcommand {args.subcommand!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
