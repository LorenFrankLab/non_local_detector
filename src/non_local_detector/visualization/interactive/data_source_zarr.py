"""Optional zarr acceleration cache for the interactive viewer.

The viewer's canonical input is a ``--run-from-dir`` bundle holding
``results.nc`` + ``model.pkl`` + ``spikes.npz`` + ``position.parquet``.
A user can also drop a chunked ``results.zarr/`` next to ``results.nc``
(via ``python -m non_local_detector.visualization.interactive.devtools
build-viewer-cache --run-dir ...``); when the cache validates against
the canonical NetCDF, the viewer reads large per-time arrays lazily
through xarray's zarr backend instead of materialising them in RAM.

This module exposes only the helpers needed for that path —
``load_zarr_cache_or_fall_back`` is the validator the CLI calls, and
``_load_zarr_results`` / ``_restore_state_bins_multiindex`` handle the
``state_bins`` MultiIndex restoration that ``xr.open_zarr`` drops.

Bundle layout::

    bundle_dir/
    ├── results.nc           # canonical (always required)
    ├── results.zarr/        # optional acceleration cache
    ├── model.pkl
    ├── spikes.npz
    └── position.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _restore_state_bins_multiindex(results: xr.Dataset) -> xr.Dataset:
    """Reattach the ``state_bins`` MultiIndex after a lazy open.

    ``xr.open_zarr`` (like ``xr.open_dataset``) drops MultiIndex
    metadata; the per-cell collapse helpers expect ``state_bins`` to
    be a MultiIndex of ``(state_ind, position, ...)`` so the panels
    can pick rows by state. Mirrors ``_DetectorBase.load_results``'s
    NetCDF round-trip.
    """
    if "state_bins" not in results.coords:
        return results
    coord_names = [
        name
        for name, coord in results["state_bins"].coords.items()
        if coord.dims == ("state_bins",)
    ]
    if not coord_names:
        return results
    return results.set_index(state_bins=coord_names)


def _load_zarr_results(zarr_path: Path) -> xr.Dataset:
    """Open ``results.zarr`` lazily + restore ``state_bins`` MultiIndex.

    ``consolidated=True`` is preferred when the writer produced a
    consolidated metadata block, but we fall through to the
    non-consolidated path so a hand-written cache (or one written by
    an old ``to_zarr`` call) still loads.
    """
    try:
        results = xr.open_zarr(str(zarr_path), consolidated=True)
    except (KeyError, ValueError):
        # ``consolidated=True`` raises on stores without a
        # ``.zmetadata`` consolidation marker. Fall back; the cost is
        # one extra pass to scan group metadata.
        results = xr.open_zarr(str(zarr_path), consolidated=False)
    return _restore_state_bins_multiindex(results)


def load_zarr_cache_or_fall_back(
    zarr_path: Path,
    canonical_results: xr.Dataset,
) -> xr.Dataset:
    """Validate ``results.zarr`` against the canonical NetCDF; return it.

    The viewer treats ``results.zarr/`` as an *optional* acceleration
    cache sitting next to the canonical ``results.nc`` (canonical
    because ``_DetectorBase.save_results`` writes NetCDF). A user can
    build the cache via the ``build-viewer-cache`` devtool and then
    point ``--run-from-dir`` at the same directory; if the cache
    validates the viewer uses lazy chunked reads, otherwise we fail
    loud rather than silently render misaligned data.

    Validation is intentionally cheap (no full-array reads): the time
    grid must match and *every* time-dim data variable in the canonical
    results must be present in the cache with matching shape. The
    second check is what stops a stale cache from silently dropping
    optional outputs (``log_likelihood``, ``predictive_posterior``,
    ``causal_posterior``, ...) — without it the viewer's
    ``available_outputs`` would come from the cache and panels would
    quietly disable themselves even though the canonical ``.nc`` still
    has the data.

    Raises:
        ValueError: cache does not match canonical results. Message
            includes a rebuild command so the caller can recover.
        ImportError: zarr backend is unavailable. Callers that want to
            transparently fall back to the canonical NetCDF should
            catch this; ``app._load_run`` does so.
    """
    cached = _load_zarr_results(zarr_path)
    rebuild_hint = (
        "Rebuild with `python -m non_local_detector.visualization"
        ".interactive.devtools build-viewer-cache --run-dir "
        f"{zarr_path.parent!s}` or delete the cache."
    )

    if "time" not in cached.coords or "time" not in canonical_results.coords:
        raise ValueError(
            f"results.zarr cache at {zarr_path!s} is missing the 'time' "
            f"coordinate. {rebuild_hint}"
        )
    cached_time = np.asarray(cached["time"].values)
    canon_time = np.asarray(canonical_results["time"].values)
    if cached_time.shape != canon_time.shape or not np.array_equal(
        cached_time, canon_time
    ):
        raise ValueError(
            f"results.zarr cache at {zarr_path!s} is stale: time grid "
            f"differs from canonical results.nc (cached n_time="
            f"{cached_time.size}, canonical n_time={canon_time.size}). "
            f"{rebuild_hint}"
        )

    # Validate every canonical variable the viewer might consume — i.e.
    # anything with a ``time`` dim. Iterating data_vars (rather than
    # hard-coding names) means a future canonical output is covered
    # automatically without touching this validator.
    for var_name, canon_var in canonical_results.data_vars.items():
        if "time" not in canon_var.dims:
            continue
        if var_name not in cached.data_vars:
            raise ValueError(
                f"results.zarr cache at {zarr_path!s} is stale: missing "
                f"time-dim variable '{var_name}' present in canonical "
                f"results.nc. {rebuild_hint}"
            )
        if cached[var_name].shape != canon_var.shape:
            raise ValueError(
                f"results.zarr cache at {zarr_path!s} is stale: '{var_name}' "
                f"shape {cached[var_name].shape} != canonical "
                f"{canon_var.shape}. {rebuild_hint}"
            )
    return cached
