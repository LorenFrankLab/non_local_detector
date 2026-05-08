"""End-to-end CLI smoke test for ``python -m ...interactive``.

Round-trips a Track 0 simulated bundle to disk, runs ``app.main(argv)``
against the four CLI files (``results.nc`` + ``model.pkl`` +
``spikes.npz`` + ``position.parquet``), and verifies the Qt window
constructs without crashing. Headless via the offscreen Qt platform.

Marked ``@pytest.mark.gui`` because it constructs an actual
``QtViewer``; ``[viewer]`` extra is required.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Force offscreen Qt platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from non_local_detector.models.base import _DetectorBase
from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
)
from non_local_detector.visualization.interactive.app import main as app_main

pytestmark = pytest.mark.gui


def _save_bundle_files(
    out_dir: Path,
    fitted: FittedDetector,
    session: SimulatedSession,
) -> dict[str, Path]:
    """Serialize a fitted detector + session to the four CLI files."""
    paths = {
        "results": out_dir / "results.nc",
        "model": out_dir / "model.pkl",
        "spikes": out_dir / "spikes.npz",
        "position": out_dir / "position.parquet",
    }

    # Use the canonical save_results / save_model so the round-trip
    # restores the state_bins MultiIndex on load.
    _DetectorBase.save_results(fitted.results, str(paths["results"]))
    fitted.detector.save_model(str(paths["model"]))

    spike_times_obj = np.empty(len(session.spike_times), dtype=object)
    for i, st in enumerate(session.spike_times):
        spike_times_obj[i] = np.asarray(st, dtype=np.float64)
    np.savez(paths["spikes"], spike_times=spike_times_obj)

    position_df = pd.DataFrame(
        {"position": session.position, "speed": session.speed},
        index=pd.Index(session.time, name="time"),
    )
    position_df.to_parquet(paths["position"])

    return paths


def _patch_launch_no_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``launch_qt_with_source`` to skip ``app.exec()``.

    ``app.main`` calls ``launch_qt_with_source`` directly, so patching
    the older ``launch_qt`` wrapper is not enough — the test would
    enter the Qt event loop and hang.
    """
    import non_local_detector.visualization.interactive.viewer.qt as qt_mod

    original = qt_mod.launch_qt_with_source

    def _no_block(data_source, **kwargs):
        kwargs["block"] = False
        return original(data_source, **kwargs)

    monkeypatch.setattr(qt_mod, "launch_qt_with_source", _no_block)


@pytest.mark.unit
def test_python_m_interactive_launches_against_simulated_bundle(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI: serialize a bundle, run main(argv), verify success.

    Doesn't assert pixel content — just that argparse + loaders +
    QtViewer construction round-trip without raising.
    """
    paths = _save_bundle_files(tmp_path, nl_fitted, sim_session)
    run_arg = (
        f"default:{paths['results']}:{paths['model']}:"
        f"{paths['spikes']}:{paths['position']}"
    )
    _patch_launch_no_block(monkeypatch)
    exit_code = app_main(["--run", run_arg, "--t-width", "0.5"])
    assert exit_code == 0


@pytest.mark.unit
def test_cli_rejects_malformed_run_arg(
    tmp_path: Path,
) -> None:
    """``--run`` requires exactly 5 colon-separated fields."""
    with pytest.raises(SystemExit) as exc_info:
        app_main(["--run", "default:results.nc:model.pkl"])  # only 3 fields
    # argparse exits non-zero on parse errors.
    assert exc_info.value.code != 0


@pytest.mark.unit
def test_cli_requires_at_least_one_run() -> None:
    with pytest.raises(SystemExit) as exc_info:
        app_main([])
    assert exc_info.value.code != 0


@pytest.mark.unit
def test_cli_run_from_dir_loads_bundle_directory(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--run-from-dir name:dir/`` expands to the four canonical paths."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _save_bundle_files(bundle_dir, nl_fitted, sim_session)
    _patch_launch_no_block(monkeypatch)
    exit_code = app_main(
        ["--run-from-dir", f"default:{bundle_dir}", "--t-width", "0.5"]
    )
    assert exit_code == 0


@pytest.mark.unit
def test_cli_run_from_dir_falls_back_when_zarr_backend_missing(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bundle with ``results.zarr/`` present must NOT abort when the
    zarr backend isn't installed — the viewer should warn and fall back
    to ``results.nc``.

    Pins the [viewer]-only contract: the cache is optional, so a user
    who installed only ``[viewer]`` (no ``zarr``) should still be able
    to open a bundle that happens to ship a cache directory.
    """
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _save_bundle_files(bundle_dir, nl_fitted, sim_session)
    # ``is_dir()`` is the only check ``_parse_run_from_dir_arg`` does —
    # an empty stub is enough to flip ``spec["zarr_cache"]`` on without
    # depending on zarr being installed at test time.
    (bundle_dir / "results.zarr").mkdir()

    # Force the zarr-cache loader to mimic a missing backend so the
    # fallback path is exercised regardless of whether the dev env
    # actually has zarr installed.
    import non_local_detector.visualization.interactive.data_source_zarr as dsz

    def _raise_missing(*_args, **_kwargs):
        raise ImportError("No module named 'zarr'")

    monkeypatch.setattr(dsz, "load_zarr_cache_or_fall_back", _raise_missing)

    _patch_launch_no_block(monkeypatch)
    with pytest.warns(UserWarning, match="zarr backend is unavailable"):
        exit_code = app_main(
            ["--run-from-dir", f"default:{bundle_dir}", "--t-width", "0.5"]
        )
    assert exit_code == 0


@pytest.mark.unit
def test_cli_run_from_dir_uses_zarr_cache_when_present(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``results.zarr/`` sits next to ``results.nc`` the viewer
    consumes it via ``load_zarr_cache_or_fall_back``.

    The bundle directory still contains the canonical ``results.nc`` +
    sidecars; ``results.zarr/`` is the optional acceleration cache
    written by the ``build-viewer-cache`` devtool. We sanity-check that
    ``--run-from-dir`` accepts the augmented bundle without error and
    that the cache loader was called (not just shadowed).
    """
    pytest.importorskip("zarr")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _save_bundle_files(bundle_dir, nl_fitted, sim_session)

    # Write the optional zarr cache from the canonical results.
    nl_fitted.results.reset_index("state_bins").to_zarr(
        str(bundle_dir / "results.zarr"), mode="w", consolidated=True
    )

    seen: dict[str, object] = {}
    import non_local_detector.visualization.interactive.data_source_zarr as dsz

    original_loader = dsz.load_zarr_cache_or_fall_back

    def _wrap_loader(zarr_path, canonical_results):
        seen["zarr_path"] = zarr_path
        return original_loader(zarr_path, canonical_results)

    monkeypatch.setattr(dsz, "load_zarr_cache_or_fall_back", _wrap_loader)
    # ``app._load_run`` does ``from ...data_source_zarr import
    # load_zarr_cache_or_fall_back`` *inside* the function — patching
    # the module attribute lets that import resolve to the wrapper.

    _patch_launch_no_block(monkeypatch)
    exit_code = app_main(
        ["--run-from-dir", f"default:{bundle_dir}", "--t-width", "0.5"]
    )
    assert exit_code == 0
    assert "zarr_path" in seen, "cache loader was not invoked"
    assert Path(seen["zarr_path"]).name == "results.zarr"


@pytest.mark.unit
def test_cli_run_from_dir_rejects_missing_directory(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no_such_dir"
    with pytest.raises(SystemExit) as exc_info:
        app_main(["--run-from-dir", f"default:{nonexistent}"])
    assert exc_info.value.code != 0


@pytest.mark.unit
def test_cli_run_from_dir_rejects_incomplete_bundle(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """Missing one of the four expected files must surface a clear error."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _save_bundle_files(bundle_dir, nl_fitted, sim_session)
    (bundle_dir / "spikes.npz").unlink()

    with pytest.raises(SystemExit) as exc_info:
        app_main(["--run-from-dir", f"default:{bundle_dir}"])
    assert exc_info.value.code != 0


@pytest.mark.unit
def test_cli_run_from_dir_rejects_malformed_arg() -> None:
    """Missing colon → clear argparse error."""
    with pytest.raises(SystemExit) as exc_info:
        app_main(["--run-from-dir", "no_colon_here"])
    assert exc_info.value.code != 0
