"""Tests for the ``bundle-from-detector`` devtool subcommand (Phase 6.4).

Round-trip: take a fitted detector + simulated session, run
``bundle_from_detector`` to write a bundle directory, then load it
via the canonical CLI ``--run-from-dir`` path and confirm the
viewer can construct a data source from the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
)

pytestmark = pytest.mark.gui


@pytest.mark.unit
def test_bundle_from_detector_round_trips_via_run_from_dir(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """In-process: ``bundle_from_detector`` then ``--run-from-dir`` load.

    Pins the contract: the bundle directory the devtool emits is
    structurally identical to the layout ``--run-from-dir`` consumes.
    """
    from non_local_detector.visualization.interactive.app import (
        _load_run,
        _parse_run_from_dir_arg,
    )
    from non_local_detector.visualization.interactive.devtools.bundle_from_detector import (
        bundle_from_detector,
    )

    out = bundle_from_detector(
        detector=nl_fitted.detector,
        results=nl_fitted.results,
        spike_times=sim_session.spike_times,
        position=sim_session.position,
        position_time=sim_session.time,
        speed=sim_session.speed,
        out=tmp_path / "bundle_out",
    )
    assert (out / "results.nc").exists()
    assert (out / "model.pkl").exists()
    assert (out / "spikes.npz").exists()
    assert (out / "position.parquet").exists()

    spec = _parse_run_from_dir_arg(f"default:{out}")
    name, bundle = _load_run(spec)
    assert name == "default"
    # Sidecar parity: position + spike_times round-tripped equal.
    np.testing.assert_array_equal(
        np.asarray(bundle.position), np.asarray(sim_session.position)
    )
    assert len(bundle.spike_times) == len(sim_session.spike_times)


@pytest.mark.unit
def test_bundle_from_detector_overwrites_when_flag_set(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``overwrite=True`` is required to replace existing files."""
    from non_local_detector.visualization.interactive.devtools.bundle_from_detector import (
        bundle_from_detector,
    )

    out_dir = tmp_path / "bundle_out"
    bundle_from_detector(
        detector=nl_fitted.detector,
        results=nl_fitted.results,
        spike_times=sim_session.spike_times,
        position=sim_session.position,
        position_time=sim_session.time,
        out=out_dir,
    )
    # Without --overwrite, second call raises FileExistsError.
    with pytest.raises(FileExistsError):
        bundle_from_detector(
            detector=nl_fitted.detector,
            results=nl_fitted.results,
            spike_times=sim_session.spike_times,
            position=sim_session.position,
            position_time=sim_session.time,
            out=out_dir,
        )
    # With overwrite=True it succeeds.
    bundle_from_detector(
        detector=nl_fitted.detector,
        results=nl_fitted.results,
        spike_times=sim_session.spike_times,
        position=sim_session.position,
        position_time=sim_session.time,
        out=out_dir,
        overwrite=True,
    )


@pytest.mark.unit
def test_bundle_from_detector_validates_before_writing(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """A bad-shape ``position`` raises BEFORE any sidecar is written,
    so a failed call doesn't leave a half-built directory on disk.

    Regression for the post-Phase-6 review finding: previously
    ``results.nc`` / ``model.pkl`` / ``spikes.npz`` were written
    eagerly and the position validator ran last, leaving a partial
    bundle if it raised."""
    from non_local_detector.visualization.interactive.devtools.bundle_from_detector import (
        bundle_from_detector,
    )

    out_dir = tmp_path / "bundle_partial"
    bad_position = np.zeros((10, 5))  # neither 1D nor (n, 2)
    with pytest.raises(ValueError, match="position must be shape"):
        bundle_from_detector(
            detector=nl_fitted.detector,
            results=nl_fitted.results,
            spike_times=sim_session.spike_times,
            position=bad_position,
            position_time=np.arange(10),
            out=out_dir,
        )
    # Directory may have been created (mkdir runs early to land
    # inside it); but no sidecar files should exist.
    if out_dir.exists():
        assert list(out_dir.iterdir()) == [], (
            f"bundle_from_detector left partial files behind in {out_dir}: "
            f"{[p.name for p in out_dir.iterdir()]!r}"
        )


@pytest.mark.unit
def test_bundle_from_detector_validates_speed_length_before_writing(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``speed`` mismatched against ``position_time`` raises before
    writing anything."""
    from non_local_detector.visualization.interactive.devtools.bundle_from_detector import (
        bundle_from_detector,
    )

    out_dir = tmp_path / "bundle_speed_mismatch"
    n = sim_session.position.size
    with pytest.raises(ValueError, match="speed length"):
        bundle_from_detector(
            detector=nl_fitted.detector,
            results=nl_fitted.results,
            spike_times=sim_session.spike_times,
            position=sim_session.position,
            position_time=sim_session.time,
            speed=np.zeros(n - 1),  # wrong length
            out=out_dir,
        )
    if out_dir.exists():
        assert list(out_dir.iterdir()) == []


@pytest.mark.unit
def test_devtools_package_exports_bundle_from_detector() -> None:
    """``bundle_from_detector`` is reachable from the package surface
    so notebook callers can import it without reaching into the
    submodule."""
    from non_local_detector.visualization.interactive import devtools

    assert hasattr(devtools, "bundle_from_detector")
    assert "bundle_from_detector" in devtools.__all__


@pytest.mark.unit
def test_bundle_from_detector_cli_round_trip(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """End-to-end: write the four sidecars, invoke ``main`` with the
    bundle-from-detector subcommand, then confirm the output dir loads
    as a viewer bundle."""

    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.app import (
        _load_run,
        _parse_run_from_dir_arg,
    )
    from non_local_detector.visualization.interactive.devtools.__main__ import main

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _DetectorBase.save_results(nl_fitted.results, str(src_dir / "results.nc"))
    nl_fitted.detector.save_model(str(src_dir / "model.pkl"))
    spikes_obj = np.empty(len(sim_session.spike_times), dtype=object)
    for i, st in enumerate(sim_session.spike_times):
        spikes_obj[i] = np.asarray(st, dtype=float)
    np.savez(str(src_dir / "spikes.npz"), spike_times=spikes_obj)
    pos_df = pd.DataFrame(
        {"position": sim_session.position},
        index=pd.Index(sim_session.time, name="time"),
    )
    pos_df.to_parquet(str(src_dir / "position.parquet"))

    out_dir = tmp_path / "bundle_out"
    code = main(
        [
            "bundle-from-detector",
            "--detector",
            str(src_dir / "model.pkl"),
            "--results",
            str(src_dir / "results.nc"),
            "--spikes",
            str(src_dir / "spikes.npz"),
            "--position",
            str(src_dir / "position.parquet"),
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0

    spec = _parse_run_from_dir_arg(f"default:{out_dir}")
    name, bundle = _load_run(spec)
    assert name == "default"
    assert bundle.position.size > 0
