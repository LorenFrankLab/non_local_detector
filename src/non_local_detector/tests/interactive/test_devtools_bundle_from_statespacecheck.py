"""Tests for the ``bundle-from-statespacecheck-cache`` devtool.

Synthesizes a statespacecheck-cache layout from the existing Track 0
fixtures (``cf_fitted`` + ``sim_session``) so the devtool can be
exercised end-to-end without the upstream cache being available.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from non_local_detector.analysis.place_fields import (
    extract_state_aligned_place_fields,
)
from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
)
from non_local_detector.visualization.interactive.devtools.bundle_from_statespacecheck import (  # noqa: E501
    bundle_from_statespacecheck_cache,
)


def _interior_place_fields(detector) -> np.ndarray:
    """Return upstream-format interior-only place fields for the cache sidecar."""
    full = extract_state_aligned_place_fields(detector)
    interior = np.asarray(detector.is_track_interior_state_bins_, dtype=bool)
    return full[:, interior].astype(np.float32)


def _attach_upstream_scalar_coords(results):
    """Attach the 0-D scalar coords found on real upstream NetCDFs.

    Real ``cont_results.nc`` / ``cont_frag_results.nc`` carry
    ``environments`` and ``encoding_groups`` as **0-D** coords (verified
    against ``cont_results.nc`` directly — not multi-element). They
    appear in ``list(ds["state_bins"].coords)`` regardless of dim, and
    the devtool's ``_DetectorBase.load_results`` call previously fed
    them to ``set_index`` which raised ``PandasMultiIndex only accepts
    1-dimensional variables``. Our fixture detectors don't emit these
    coords, so without this attach the devtool tests miss the real
    upstream schema entirely.
    """
    return results.assign_coords(
        environments="env0",
        encoding_groups="grp0",
    )


def _populate_intermediates(
    intermediates_dir: Path,
    fitted: FittedDetector,
    model_filename: str,
    *,
    add_upstream_scalar_coords: bool = False,
) -> None:
    """Mirror upstream's writer exactly: NetCDF + ``joblib.dump`` for the pkl.

    Critical that this stays as ``joblib.dump`` (not ``detector.save_model``).
    Joblib's numpy codec produces files that plain ``pickle.load``
    cannot read, and the devtool reads the source via ``joblib.load``;
    a pickle-flavoured fixture would silently let a pickle-only loader
    pass these tests while failing on the real upstream pkl.
    """
    from non_local_detector.models.base import _DetectorBase

    intermediates_dir.mkdir(parents=True, exist_ok=True)
    results = (
        _attach_upstream_scalar_coords(fitted.results)
        if add_upstream_scalar_coords
        else fitted.results
    )
    _DetectorBase.save_results(
        results, str(intermediates_dir / f"{model_filename}_results.nc")
    )
    joblib.dump(fitted.detector, str(intermediates_dir / f"{model_filename}_model.pkl"))


def _populate_cache(
    cache_dir: Path,
    session: SimulatedSession,
    fitted: FittedDetector,
    model: str,
    *,
    place_fields_override: np.ndarray | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / "figure04_meta.npz",
        time=session.time,
        linear_position=session.position,
        n_cells=len(session.spike_times),
    )
    spike_arr = np.empty(len(session.spike_times), dtype=object)
    for i, st in enumerate(session.spike_times):
        spike_arr[i] = np.asarray(st, dtype=np.float64)
    np.save(cache_dir / "figure04_spike_times.npy", spike_arr, allow_pickle=True)
    if place_fields_override is None:
        place_fields = _interior_place_fields(fitted.detector)
    else:
        place_fields = place_fields_override
    np.savez(
        cache_dir / f"figure04_{model}_place_fields.npz",
        place_fields=place_fields,
    )


@pytest.fixture
def staged_layout(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> tuple[Path, Path, Path]:
    """Build a fresh upstream-shaped layout under ``tmp_path``.

    Includes the scalar ``state_bins`` coords (``environments``,
    ``encoding_groups``) found on real upstream NetCDFs so the
    happy-path test covers the schema that ``cont_results.nc``
    actually has, not just the minimal fixture form.
    """
    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    out_dir = tmp_path / "bundle"
    _populate_intermediates(
        intermediates_dir, cf_fitted, "cont_frag", add_upstream_scalar_coords=True
    )
    _populate_cache(cache_dir, sim_session, cf_fitted, "contfrag")
    return cache_dir, intermediates_dir, out_dir


@pytest.mark.unit
def test_bundle_writes_loadable_run(
    staged_layout: tuple[Path, Path, Path],
) -> None:
    """End-to-end: devtool output must load via the existing ``--run`` loader."""
    from non_local_detector.visualization.interactive.app import _load_run

    cache_dir, intermediates_dir, out_dir = staged_layout
    bundle_from_statespacecheck_cache(
        cache_dir=cache_dir,
        intermediates_dir=intermediates_dir,
        model="contfrag",
        out=out_dir,
    )
    for filename in ("results.nc", "model.pkl", "spikes.npz", "position.parquet"):
        assert (out_dir / filename).exists(), f"missing {filename}"

    name, bundle = _load_run(
        {
            "name": "contfrag",
            "results": str(out_dir / "results.nc"),
            "model": str(out_dir / "model.pkl"),
            "spikes": str(out_dir / "spikes.npz"),
            "position": str(out_dir / "position.parquet"),
        }
    )
    assert name == "contfrag"
    assert "acausal_posterior" in bundle.results.data_vars
    assert "acausal_state_probabilities" in bundle.results.data_vars


@pytest.mark.unit
def test_bundle_creates_out_dir_if_missing(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    out_dir = tmp_path / "deeply" / "nested" / "out"
    _populate_intermediates(intermediates_dir, cf_fitted, "cont_frag")
    _populate_cache(cache_dir, sim_session, cf_fitted, "contfrag")
    assert not out_dir.exists()
    bundle_from_statespacecheck_cache(
        cache_dir=cache_dir,
        intermediates_dir=intermediates_dir,
        model="contfrag",
        out=out_dir,
    )
    assert out_dir.is_dir()


@pytest.mark.unit
def test_unknown_model_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        bundle_from_statespacecheck_cache(
            cache_dir=tmp_path / "cache",
            intermediates_dir=tmp_path / "intermediates",
            model="not_a_real_model",  # type: ignore[arg-type]
            out=tmp_path / "out",
        )


@pytest.mark.unit
def test_missing_required_results_var_raises(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """Source NetCDF without ``acausal_state_probabilities`` must be rejected."""
    from non_local_detector.models.base import _DetectorBase

    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    intermediates_dir.mkdir(parents=True)
    stripped = cf_fitted.results.drop_vars("acausal_state_probabilities")
    _DetectorBase.save_results(
        stripped, str(intermediates_dir / "cont_frag_results.nc")
    )
    joblib.dump(cf_fitted.detector, str(intermediates_dir / "cont_frag_model.pkl"))
    _populate_cache(cache_dir, sim_session, cf_fitted, "contfrag")

    with pytest.raises(ValueError, match="acausal_state_probabilities"):
        bundle_from_statespacecheck_cache(
            cache_dir=cache_dir,
            intermediates_dir=intermediates_dir,
            model="contfrag",
            out=tmp_path / "out",
        )


@pytest.mark.unit
def test_place_fields_shape_mismatch_raises(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    _populate_intermediates(intermediates_dir, cf_fitted, "cont_frag")
    bogus = np.zeros((3, 3), dtype=np.float32)
    _populate_cache(
        cache_dir, sim_session, cf_fitted, "contfrag", place_fields_override=bogus
    )

    with pytest.raises(ValueError, match="Place-field shape mismatch"):
        bundle_from_statespacecheck_cache(
            cache_dir=cache_dir,
            intermediates_dir=intermediates_dir,
            model="contfrag",
            out=tmp_path / "out",
        )


@pytest.mark.unit
def test_place_fields_value_mismatch_raises(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    _populate_intermediates(intermediates_dir, cf_fitted, "cont_frag")
    expected = _interior_place_fields(cf_fitted.detector)
    drift = expected.copy()
    drift[0, 0] += 5.0  # well outside the rtol/atol tolerances
    _populate_cache(
        cache_dir,
        sim_session,
        cf_fitted,
        "contfrag",
        place_fields_override=drift,
    )

    with pytest.raises(ValueError, match="Place-field values disagree"):
        bundle_from_statespacecheck_cache(
            cache_dir=cache_dir,
            intermediates_dir=intermediates_dir,
            model="contfrag",
            out=tmp_path / "out",
        )


@pytest.mark.unit
def test_results_nc_override_used(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``--results-nc`` override must take precedence over the auto-resolved path."""
    from non_local_detector.models.base import _DetectorBase

    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    intermediates_dir.mkdir(parents=True)
    # Auto-resolved path would be ``intermediates/cont_frag_results.nc`` —
    # don't put anything there. Put a valid file at a custom location and
    # pass it via the override.
    custom_results = tmp_path / "custom" / "my_results.nc"
    custom_results.parent.mkdir(parents=True)
    _DetectorBase.save_results(cf_fitted.results, str(custom_results))
    joblib.dump(cf_fitted.detector, str(intermediates_dir / "cont_frag_model.pkl"))
    _populate_cache(cache_dir, sim_session, cf_fitted, "contfrag")

    out_dir = tmp_path / "out"
    bundle_from_statespacecheck_cache(
        cache_dir=cache_dir,
        intermediates_dir=intermediates_dir,
        model="contfrag",
        out=out_dir,
        results_nc=custom_results,
    )
    assert (out_dir / "results.nc").exists()


@pytest.mark.unit
def test_results_from_zarr_validates_required_vars(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """Zarr fallback must refuse if required vars are absent."""
    pytest.importorskip("zarr")  # Optional dep — skip when unavailable.

    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    intermediates_dir.mkdir(parents=True)
    joblib.dump(cf_fitted.detector, str(intermediates_dir / "cont_frag_model.pkl"))
    _populate_cache(cache_dir, sim_session, cf_fitted, "contfrag")
    # Write a Zarr that lacks ``acausal_posterior`` to mimic upstream's
    # "optional" stance on it.
    zarr_path = cache_dir / "figure04_contfrag.zarr"
    cf_fitted.results.drop_vars(
        ["acausal_posterior", "acausal_state_probabilities"]
    ).reset_index("state_bins").to_zarr(str(zarr_path), consolidated=True, mode="w")

    with pytest.raises(ValueError, match="acausal_posterior"):
        bundle_from_statespacecheck_cache(
            cache_dir=cache_dir,
            intermediates_dir=intermediates_dir,
            model="contfrag",
            out=tmp_path / "out",
            results_from_zarr=True,
        )


@pytest.mark.unit
def test_zarr_and_nc_overrides_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        bundle_from_statespacecheck_cache(
            cache_dir=tmp_path / "cache",
            intermediates_dir=tmp_path / "intermediates",
            model="contfrag",
            out=tmp_path / "out",
            results_nc=tmp_path / "anywhere.nc",
            results_from_zarr=True,
        )


@pytest.mark.unit
def test_position_parquet_round_trips_time_index(
    staged_layout: tuple[Path, Path, Path],
    sim_session: SimulatedSession,
) -> None:
    """Output parquet must preserve the ``time`` index used by ``app._load_run``."""
    cache_dir, intermediates_dir, out_dir = staged_layout
    bundle_from_statespacecheck_cache(
        cache_dir=cache_dir,
        intermediates_dir=intermediates_dir,
        model="contfrag",
        out=out_dir,
    )
    df = pd.read_parquet(out_dir / "position.parquet")
    np.testing.assert_array_equal(df.index.to_numpy(), sim_session.time)
    np.testing.assert_array_equal(df["position"].to_numpy(), sim_session.position)


@pytest.mark.unit
def test_main_cli_dispatch(
    tmp_path: Path,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``__main__.main`` argv-style invocation writes the bundle dir."""
    from non_local_detector.visualization.interactive.devtools.__main__ import main

    cache_dir = tmp_path / "cache"
    intermediates_dir = tmp_path / "intermediates"
    out_dir = tmp_path / "out"
    _populate_intermediates(intermediates_dir, cf_fitted, "cont_frag")
    _populate_cache(cache_dir, sim_session, cf_fitted, "contfrag")

    code = main(
        [
            "bundle-from-statespacecheck-cache",
            "--cache-dir",
            str(cache_dir),
            "--intermediates-dir",
            str(intermediates_dir),
            "--model",
            "contfrag",
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0
    assert (out_dir / "results.nc").exists()
    assert (out_dir / "model.pkl").exists()
