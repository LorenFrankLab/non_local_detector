"""Build a chunked ``results.zarr`` cache next to ``results.nc``.

The viewer treats ``results.zarr/`` as an optional acceleration cache:
when present and validated, ``--run-from-dir`` reads large per-time
arrays lazily through xarray's zarr backend instead of materialising
the whole NetCDF in RAM. The cache lives next to the canonical
``results.nc`` inside an existing bundle directory; this module does
the read + chunk + write step.

This is a viewer concern, not a model concern — ``_DetectorBase
.save_results`` continues to write NetCDF unchanged.
"""

from __future__ import annotations

from pathlib import Path

from non_local_detector.models.base import _DetectorBase

# Time-axis chunk for the four large per-time arrays. 1024 keeps each
# zarr chunk in the low-MB range for typical state-bin counts and lines
# up well with the viewer's window-load granularity.
DEFAULT_TIME_CHUNK = 1024


def build_viewer_cache(
    run_dir: Path,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    overwrite: bool = False,
) -> Path:
    """Read ``run_dir/results.nc`` and write ``run_dir/results.zarr/``.

    Parameters
    ----------
    run_dir
        Bundle directory holding ``results.nc`` (and the usual viewer
        sidecars; they are not touched here).
    time_chunk
        Chunk size along the ``time`` axis. The four big variables
        (``acausal_posterior``, ``log_likelihood``,
        ``predictive_posterior``, ``acausal_state_probabilities``) are
        rechunked to ``(time_chunk, ...)`` so window slices materialise
        only the visible chunks.
    overwrite
        If ``True``, replace any existing ``results.zarr/``. Default
        ``False`` raises ``FileExistsError`` to prevent accidental
        clobbering of a cache the viewer is currently using.

    Returns
    -------
    Path
        The directory of the written ``results.zarr/``.
    """
    run_dir = Path(run_dir)
    nc_path = run_dir / "results.nc"
    zarr_path = run_dir / "results.zarr"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"build-viewer-cache: no results.nc in {run_dir!s}; the cache "
            "command operates on an existing viewer bundle directory."
        )
    if zarr_path.exists() and not overwrite:
        raise FileExistsError(
            f"build-viewer-cache: {zarr_path!s} already exists. Pass "
            "--overwrite to replace it."
        )

    results = _DetectorBase.load_results(str(nc_path))
    # ``state_bins`` is a MultiIndex; ``to_zarr`` doesn't round-trip it
    # directly, so reset before writing — same trick ``save_results``
    # uses for NetCDF. The lazy-load helper restores the MultiIndex on
    # open so panels still get ``(state_ind, position, ...)`` rows.
    flat = results.reset_index("state_bins")
    chunks: dict[str, int] = {}
    if "time" in flat.dims:
        chunks["time"] = time_chunk
    if chunks:
        flat = flat.chunk(chunks)
    # Stamp the source ``results.nc`` mtime + size onto the cache
    # attrs. The validator compares these against the live file's stat
    # to detect a re-saved canonical that happens to share shape with
    # the cache (otherwise stale arrays would silently render). Stored
    # as strings because mtime_ns exceeds JSON's safe integer range
    # (2**53) and zarr attrs round-trip through .zattrs JSON.
    src_stat = nc_path.stat()
    flat = flat.assign_attrs(
        viewer_cache_source_mtime_ns=str(src_stat.st_mtime_ns),
        viewer_cache_source_size_bytes=str(src_stat.st_size),
    )
    flat.to_zarr(str(zarr_path), mode="w", consolidated=True)
    return zarr_path
