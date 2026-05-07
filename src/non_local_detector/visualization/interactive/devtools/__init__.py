"""Devtools for building viewer bundles from external sources.

Each subcommand emits a CLI-compatible bundle directory containing
the four files the viewer's ``--run`` flag consumes (``results.nc``,
``model.pkl``, ``spikes.npz``, ``position.parquet``). No new bundle
file format is introduced.

Currently shipped:

- ``bundle_from_statespacecheck_cache`` —
  ``python -m non_local_detector.visualization.interactive.devtools
  bundle-from-statespacecheck-cache``: bundles the upstream
  ``statespacecheck-paper-viewer`` cache + intermediates layout into
  a viewer bundle directory.
"""

from __future__ import annotations

from non_local_detector.visualization.interactive.devtools.bundle_from_statespacecheck import (  # noqa: E501
    bundle_from_statespacecheck_cache,
)

__all__ = ["bundle_from_statespacecheck_cache"]
