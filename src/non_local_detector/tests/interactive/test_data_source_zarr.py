"""Tests for the optional zarr-cache helpers.

The viewer's canonical input is a NetCDF bundle; ``results.zarr/`` is
an optional acceleration cache validated against the canonical
``results.nc`` by :func:`load_zarr_cache_or_fall_back`. These tests
pin the validator's contract (positive round-trip + the three
rejection paths) and the ``build-viewer-cache`` devtool that writes
the cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")  # Optional dep — skip when unavailable.

from non_local_detector.models.base import _DetectorBase
from non_local_detector.tests._simulated_detectors import FittedDetector
from non_local_detector.visualization.interactive.data_source_zarr import (
    _load_zarr_results,
    load_zarr_cache_or_fall_back,
)
from non_local_detector.visualization.interactive.devtools.build_viewer_cache import (
    build_viewer_cache,
)


def _write_canonical_nc(run_dir: Path, fitted: FittedDetector) -> Path:
    """Save the canonical ``results.nc`` next to which a cache may live."""
    run_dir.mkdir(parents=True, exist_ok=True)
    nc_path = run_dir / "results.nc"
    _DetectorBase.save_results(fitted.results, str(nc_path))
    return nc_path


@pytest.fixture
def cached_run_dir(tmp_path: Path, nl_fitted: FittedDetector) -> Path:
    """A bundle dir holding both ``results.nc`` and a fresh ``results.zarr``."""
    run_dir = tmp_path / "bundle"
    _write_canonical_nc(run_dir, nl_fitted)
    build_viewer_cache(run_dir)
    return run_dir


@pytest.mark.unit
def test_build_viewer_cache_writes_results_zarr(
    tmp_path: Path,
    nl_fitted: FittedDetector,
) -> None:
    """``build_viewer_cache`` writes a chunked ``results.zarr/``.

    Pins the devtool's contract: it operates in-place on a directory
    that already has ``results.nc`` and produces ``results.zarr/`` next
    to it without touching the canonical NetCDF.
    """
    run_dir = tmp_path / "bundle"
    nc_path = _write_canonical_nc(run_dir, nl_fitted)
    nc_mtime_before = nc_path.stat().st_mtime_ns

    out = build_viewer_cache(run_dir)

    assert out == run_dir / "results.zarr"
    assert out.is_dir()
    # The canonical NetCDF must not have been rewritten.
    assert nc_path.stat().st_mtime_ns == nc_mtime_before
    # The cache must round-trip the time grid.
    cached = _load_zarr_results(out)
    np.testing.assert_array_equal(
        np.asarray(cached["time"].values),
        np.asarray(nl_fitted.results["time"].values),
    )


@pytest.mark.unit
def test_build_viewer_cache_requires_results_nc(tmp_path: Path) -> None:
    """No ``results.nc`` → clear ``FileNotFoundError``."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="results.nc"):
        build_viewer_cache(tmp_path / "empty")


@pytest.mark.unit
def test_build_viewer_cache_refuses_to_overwrite_by_default(
    cached_run_dir: Path,
) -> None:
    """Existing cache must not be silently clobbered."""
    with pytest.raises(FileExistsError, match="results.zarr"):
        build_viewer_cache(cached_run_dir)


@pytest.mark.unit
def test_build_viewer_cache_overwrite_replaces_existing(
    cached_run_dir: Path,
    nl_fitted: FittedDetector,
) -> None:
    """``--overwrite`` rewrites the cache in place."""
    out = build_viewer_cache(cached_run_dir, overwrite=True)
    cached = _load_zarr_results(out)
    np.testing.assert_array_equal(
        np.asarray(cached["time"].values),
        np.asarray(nl_fitted.results["time"].values),
    )


@pytest.mark.unit
def test_load_zarr_cache_round_trips_against_canonical(
    cached_run_dir: Path,
    nl_fitted: FittedDetector,
) -> None:
    """A fresh cache validates and returns the lazy zarr Dataset."""
    canonical = _DetectorBase.load_results(str(cached_run_dir / "results.nc"))
    cached = load_zarr_cache_or_fall_back(
        zarr_path=cached_run_dir / "results.zarr",
        canonical_results=canonical,
    )
    np.testing.assert_array_equal(
        np.asarray(cached["time"].values),
        np.asarray(nl_fitted.results["time"].values),
    )
    # Lazy load preserves the per-time array shape so windowed
    # ``isel(time=sl).values`` materialises the same chunks the panels
    # expect. (MultiIndex restoration on the ``state_bins`` dim is a
    # downstream concern handled by the panel-level collapse helpers.)
    assert (
        cached["acausal_posterior"].shape
        == nl_fitted.results["acausal_posterior"].shape
    )


@pytest.mark.unit
def test_load_zarr_cache_rejects_stale_time_grid(
    cached_run_dir: Path,
) -> None:
    """A canonical with a different time grid must trigger rebuild error."""
    canonical = _DetectorBase.load_results(str(cached_run_dir / "results.nc"))
    # Simulate a canonical that drifted to a longer time axis (e.g. a
    # rerun extended the session). The cache no longer aligns.
    drifted = canonical.isel(time=slice(0, canonical.sizes["time"] - 1))
    with pytest.raises(ValueError, match="stale|time grid"):
        load_zarr_cache_or_fall_back(
            zarr_path=cached_run_dir / "results.zarr",
            canonical_results=drifted,
        )


@pytest.mark.unit
def test_load_zarr_cache_rejects_shape_mismatch(
    cached_run_dir: Path,
) -> None:
    """A posterior shape mismatch on the same time grid still rebuilds."""
    canonical = _DetectorBase.load_results(str(cached_run_dir / "results.nc"))
    # Trim a state-bin row so the time grid still matches but the
    # acausal_posterior shape differs from the cached store.
    trimmed = canonical.isel(state_bins=slice(0, canonical.sizes["state_bins"] - 1))
    with pytest.raises(ValueError, match="acausal_posterior"):
        load_zarr_cache_or_fall_back(
            zarr_path=cached_run_dir / "results.zarr",
            canonical_results=trimmed,
        )


@pytest.mark.unit
def test_load_zarr_cache_rejects_missing_optional_output(
    cached_run_dir: Path,
) -> None:
    """A cache lacking an optional output the canonical has must rebuild.

    Without this check the viewer's ``available_outputs`` would come
    from the cache and panels for ``log_likelihood`` /
    ``predictive_posterior`` would silently disable themselves even
    though the canonical ``results.nc`` still has the data. We
    simulate that by writing a *partial* cache (acausal_posterior +
    state probs only) and then loading it against the full canonical.
    """
    canonical = _DetectorBase.load_results(str(cached_run_dir / "results.nc"))
    # Pick an optional output the canonical actually has, so the test
    # is meaningful regardless of which fixture surfaced it.
    optional = next(
        (
            name
            for name in (
                "log_likelihood",
                "predictive_posterior",
                "causal_posterior",
            )
            if name in canonical.data_vars and "time" in canonical[name].dims
        ),
        None,
    )
    if optional is None:
        pytest.skip("Fixture has no optional time-dim output to drop")

    partial_cache_dir = cached_run_dir.parent / "partial"
    partial_cache_dir.mkdir()
    partial_zarr = partial_cache_dir / "results.zarr"
    canonical.drop_vars(optional).reset_index("state_bins").to_zarr(
        str(partial_zarr), mode="w", consolidated=True
    )

    with pytest.raises(ValueError, match=optional):
        load_zarr_cache_or_fall_back(
            zarr_path=partial_zarr,
            canonical_results=canonical,
        )
