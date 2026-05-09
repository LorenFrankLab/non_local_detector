"""Devtools for building viewer bundles from external sources.

Each subcommand emits a CLI-compatible bundle directory containing
the four files the viewer's ``--run-from-dir`` flag consumes
(``results.nc``, ``model.pkl``, ``spikes.npz``,
``position.parquet``). No new bundle file format is introduced.

Currently shipped:

- ``bundle_from_statespacecheck_cache`` —
  ``python -m non_local_detector.visualization.interactive.devtools
  bundle-from-statespacecheck-cache``: bundles the upstream
  ``statespacecheck-paper-viewer`` cache + intermediates layout into
  a viewer bundle directory.
- ``bundle_from_detector`` —
  ``python -m non_local_detector.visualization.interactive.devtools
  bundle-from-detector``: packages a fitted detector + its
  ``predict()`` outputs + session sidecars (spike_times, position)
  into a ``--run-from-dir``-compatible bundle directory. Useful for
  archiving / sharing a notebook session via the CLI.
- ``build_viewer_cache`` —
  ``python -m non_local_detector.visualization.interactive.devtools
  build-viewer-cache``: writes a chunked ``results.zarr/`` next to an
  existing ``results.nc`` so ``--run-from-dir`` can lazy-load large
  per-time arrays.
"""

from __future__ import annotations

from non_local_detector.visualization.interactive.devtools.build_viewer_cache import (
    build_viewer_cache,
)
from non_local_detector.visualization.interactive.devtools.bundle_from_detector import (
    bundle_from_detector,
)
from non_local_detector.visualization.interactive.devtools.bundle_from_statespacecheck import (  # noqa: E501
    bundle_from_statespacecheck_cache,
)

__all__ = [
    "build_viewer_cache",
    "bundle_from_detector",
    "bundle_from_statespacecheck_cache",
]
