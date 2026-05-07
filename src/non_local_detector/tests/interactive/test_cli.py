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


@pytest.mark.unit
def test_python_m_interactive_launches_against_simulated_bundle(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """End-to-end CLI: serialize a bundle, run main(argv), verify success.

    Doesn't assert pixel content — just that argparse + loaders +
    QtViewer construction round-trip without raising. Patches
    ``launch_qt`` to ``block=False`` so the test doesn't enter the
    Qt event loop.
    """
    paths = _save_bundle_files(tmp_path, nl_fitted, sim_session)
    run_arg = (
        f"default:{paths['results']}:{paths['model']}:"
        f"{paths['spikes']}:{paths['position']}"
    )

    # Force the Qt window not to enter exec() — block=False — so the
    # test doesn't hang. We monkey-patch the launch function the CLI
    # calls.
    import non_local_detector.visualization.interactive.viewer.qt as qt_mod

    original_launch = qt_mod.launch_qt

    def _launch_no_block(bundles, **kwargs):
        kwargs.setdefault("block", False)
        kwargs["block"] = False
        return original_launch(bundles, **kwargs)

    qt_mod.launch_qt = _launch_no_block
    try:
        exit_code = app_main(["--run", run_arg, "--t-width", "0.5"])
    finally:
        qt_mod.launch_qt = original_launch

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
) -> None:
    """``--run-from-dir name:dir/`` expands to the four canonical paths."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _save_bundle_files(bundle_dir, nl_fitted, sim_session)

    import non_local_detector.visualization.interactive.viewer.qt as qt_mod

    original_launch = qt_mod.launch_qt

    def _launch_no_block(bundles, **kwargs):
        kwargs["block"] = False
        return original_launch(bundles, **kwargs)

    qt_mod.launch_qt = _launch_no_block
    try:
        exit_code = app_main(
            ["--run-from-dir", f"default:{bundle_dir}", "--t-width", "0.5"]
        )
    finally:
        qt_mod.launch_qt = original_launch

    assert exit_code == 0


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
