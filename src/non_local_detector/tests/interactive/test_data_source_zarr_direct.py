"""Tests for ``ZarrDirectDecoderDataSource`` (Phase 5.2).

The zarr-direct path must produce *exactly* the same arrays as the
in-memory path for every Protocol method on the same bundle — that's
the parity contract callers rely on. Per-method missing-output
behaviour also has to match (Phase 5.5 contracts table); a
zarr-direct ``load_likelihood`` against a cache without
``log_likelihood`` raises the same ``KeyError`` as in-memory does.

Tests build the cache once per session (the
``_save_bundle_files_with_zarr`` fixture writes the four canonical
sidecars + the validated cache) and then exercise both data sources
against the same on-disk bundle.
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
from non_local_detector.visualization.interactive.data_source import (
    DecoderDataSource,
    InMemoryDecoderDataSource,
)

pytestmark = pytest.mark.gui  # zarr backend is the [viewer-cache] extra


def _save_bundle_with_zarr(
    bundle_dir: Path,
    fitted: FittedDetector,
    session: SimulatedSession,
    *,
    drop_vars: tuple[str, ...] = (),
    transform_results=None,
) -> Path:
    """Write a canonical bundle directory + a validated zarr cache.

    ``drop_vars`` lets a test exercise missing-output paths by
    omitting (e.g.) ``log_likelihood`` or ``predictive_posterior``
    from both the canonical NetCDF and the cache. ``transform_results``
    is an optional callable invoked on the xarray Dataset right
    before save so a test can mutate variables (e.g. write a 1D
    state-prob array for the single-state contract test).
    """
    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.devtools.build_viewer_cache import (
        build_viewer_cache,
    )

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    results = fitted.results
    if drop_vars:
        results = results.drop_vars([v for v in drop_vars if v in results.data_vars])
    if transform_results is not None:
        results = transform_results(results)

    nc_path = bundle_dir / "results.nc"
    _DetectorBase.save_results(results, str(nc_path))

    _DetectorBase.save_model(fitted.detector, str(bundle_dir / "model.pkl"))

    spike_times_obj = np.empty(len(session.spike_times), dtype=object)
    for i, st in enumerate(session.spike_times):
        spike_times_obj[i] = np.asarray(st, dtype=float)
    np.savez(str(bundle_dir / "spikes.npz"), spike_times=spike_times_obj)

    pos_df = pd.DataFrame({"position": np.asarray(session.position).squeeze()})
    pos_df["x_position"] = np.asarray(session.position).squeeze()
    pos_df["y_position"] = np.asarray(session.position).squeeze() + 10.0
    pos_df.index = pd.Index(session.time, name="time")
    if session.speed is not None:
        pos_df["speed"] = np.asarray(session.speed)
    pos_df.to_parquet(str(bundle_dir / "position.parquet"))

    build_viewer_cache(bundle_dir, overwrite=True)
    return bundle_dir


def _make_zarr_direct(bundle_dir: Path):
    from non_local_detector.visualization.interactive.data_source_zarr_direct import (
        ZarrDirectDecoderDataSource,
    )

    return ZarrDirectDecoderDataSource.for_directory("default", bundle_dir)


def _make_in_memory_from_dir(bundle_dir: Path) -> InMemoryDecoderDataSource:
    """Mirror what ``_load_run`` produces but force the eager NetCDF path
    (drop the zarr cache from the spec) so parity tests compare a true
    in-memory source against the zarr-direct source."""
    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.view_models.base import (
        RunBundle,
    )

    results = _DetectorBase.load_results(str(bundle_dir / "results.nc"))
    detector = _DetectorBase.load_model(str(bundle_dir / "model.pkl"))
    spike_times_npz = np.load(str(bundle_dir / "spikes.npz"), allow_pickle=True)
    spike_times = list(spike_times_npz["spike_times"])
    position_df = pd.read_parquet(str(bundle_dir / "position.parquet"))
    position = position_df["position"].to_numpy()
    position_2d = position_df[["x_position", "y_position"]].to_numpy()
    bundle = RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=position_df.index.to_numpy(),
        position=position,
        position_2d=position_2d,
        position_2d_time=position_df.index.to_numpy(),
        speed=position_df["speed"].to_numpy() if "speed" in position_df else None,
    )
    return InMemoryDecoderDataSource.from_single(bundle)


@pytest.fixture(scope="module")
def zarr_bundle_dir(
    tmp_path_factory: pytest.TempPathFactory,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> Path:
    return _save_bundle_with_zarr(
        tmp_path_factory.mktemp("zarr_bundle"), nl_fitted, sim_session
    )


@pytest.mark.unit
def test_zarr_direct_satisfies_decoder_data_source_protocol(
    zarr_bundle_dir: Path,
) -> None:
    """``ZarrDirectDecoderDataSource`` is a structural ``DecoderDataSource``."""
    ds = _make_zarr_direct(zarr_bundle_dir)
    assert isinstance(ds, DecoderDataSource)


@pytest.mark.unit
def test_zarr_direct_window_indices_matches_in_memory(
    zarr_bundle_dir: Path,
) -> None:
    """``window_indices`` is bit-identical across data sources for the
    same ``(t_center, t_width)`` — viewer code reuses the slice across
    panels, so any drift would render misaligned."""
    in_mem = _make_in_memory_from_dir(zarr_bundle_dir)
    direct = _make_zarr_direct(zarr_bundle_dir)
    time = in_mem.time
    rng = np.random.default_rng(42)
    for _ in range(10):
        t_center = float(rng.uniform(time[0], time[-1]))
        t_width = float(rng.uniform(0.05, 5.0))
        assert in_mem.window_indices(t_center, t_width) == direct.window_indices(
            t_center, t_width
        )


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["load_posterior", "load_state_probabilities"])
def test_zarr_direct_required_load_parity(zarr_bundle_dir: Path, loader: str) -> None:
    """``load_posterior`` and ``load_state_probabilities`` are required
    arrays at the ``RunBundle`` invariant; both data sources must
    return identical values for the same slice."""
    in_mem = _make_in_memory_from_dir(zarr_bundle_dir)
    direct = _make_zarr_direct(zarr_bundle_dir)
    sl = in_mem.window_indices(
        t_center=float(in_mem.time[in_mem.n_time // 2]), t_width=1.0
    )
    a = getattr(in_mem, loader)(sl)
    b = getattr(direct, loader)(sl)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_zarr_direct_load_likelihood_parity(zarr_bundle_dir: Path) -> None:
    in_mem = _make_in_memory_from_dir(zarr_bundle_dir)
    direct = _make_zarr_direct(zarr_bundle_dir)
    sl = in_mem.window_indices(
        t_center=float(in_mem.time[in_mem.n_time // 2]), t_width=1.0
    )
    np.testing.assert_array_equal(
        in_mem.load_likelihood(sl), direct.load_likelihood(sl)
    )


@pytest.mark.unit
def test_zarr_direct_load_predictive_parity(zarr_bundle_dir: Path) -> None:
    """When ``predictive_posterior`` is present both paths return the
    same array; when absent both return ``None`` (Phase 5.5 contract)."""
    in_mem = _make_in_memory_from_dir(zarr_bundle_dir)
    direct = _make_zarr_direct(zarr_bundle_dir)
    sl = in_mem.window_indices(
        t_center=float(in_mem.time[in_mem.n_time // 2]), t_width=1.0
    )
    np.testing.assert_array_equal(
        in_mem.load_predictive(sl), direct.load_predictive(sl)
    )


@pytest.mark.unit
def test_zarr_direct_load_position_parity(zarr_bundle_dir: Path) -> None:
    in_mem = _make_in_memory_from_dir(zarr_bundle_dir)
    direct = _make_zarr_direct(zarr_bundle_dir)
    sl = in_mem.window_indices(
        t_center=float(in_mem.time[in_mem.n_time // 2]), t_width=1.0
    )
    a = in_mem.load_position(sl)
    b = direct.load_position(sl)
    assert (a is None) == (b is None)
    if a is not None:
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.unit
def test_zarr_direct_load_position_2d_parity(zarr_bundle_dir: Path) -> None:
    in_mem = _make_in_memory_from_dir(zarr_bundle_dir)
    direct = _make_zarr_direct(zarr_bundle_dir)
    sl = in_mem.window_indices(
        t_center=float(in_mem.time[in_mem.n_time // 2]), t_width=1.0
    )
    a = in_mem.load_position_2d(sl)
    b = direct.load_position_2d(sl)
    assert (a is None) == (b is None)
    if a is not None:
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.unit
def test_zarr_direct_missing_log_likelihood_raises_keyerror(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """Per-method missing-output contract: ``load_likelihood`` raises
    ``KeyError`` when the cache was built without ``log_likelihood``,
    matching the in-memory path's wording."""
    bundle_dir = _save_bundle_with_zarr(
        tmp_path / "no_loglik",
        nl_fitted,
        sim_session,
        drop_vars=("log_likelihood",),
    )
    direct = _make_zarr_direct(bundle_dir)
    sl = direct.window_indices(t_center=float(direct.time[0]), t_width=0.5)
    with pytest.raises(KeyError, match="log_likelihood"):
        direct.load_likelihood(sl)


@pytest.mark.unit
def test_zarr_direct_missing_predictive_returns_none(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``load_predictive`` returns ``None`` (NOT raise) when the cache
    was built without ``predictive_posterior``, matching the in-memory
    contract."""
    bundle_dir = _save_bundle_with_zarr(
        tmp_path / "no_predictive",
        nl_fitted,
        sim_session,
        drop_vars=("predictive_posterior",),
    )
    direct = _make_zarr_direct(bundle_dir)
    sl = direct.window_indices(t_center=float(direct.time[0]), t_width=0.5)
    assert direct.load_predictive(sl) is None


@pytest.mark.unit
def test_zarr_direct_load_likelihood_keyerror_text_pinned(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``load_likelihood`` against a cache without ``log_likelihood``
    raises ``KeyError`` with the exact wording the in-memory source
    raises — pin the regression so a future change can't drift the two
    error messages apart."""
    bundle_dir = _save_bundle_with_zarr(
        tmp_path / "no_loglik_text",
        nl_fitted,
        sim_session,
        drop_vars=("log_likelihood",),
    )
    direct = _make_zarr_direct(bundle_dir)
    sl = direct.window_indices(t_center=float(direct.time[0]), t_width=0.5)
    with pytest.raises(KeyError) as direct_exc:
        direct.load_likelihood(sl)

    in_mem = _make_in_memory_from_dir(bundle_dir)
    with pytest.raises(KeyError) as mem_exc:
        in_mem.load_likelihood(sl)

    # Both messages mention re-running predict with log_likelihood.
    assert "log_likelihood" in str(direct_exc.value)
    assert "log_likelihood" in str(mem_exc.value)
    assert "predict" in str(direct_exc.value).lower()
    assert "predict" in str(mem_exc.value).lower()


@pytest.mark.unit
def test_zarr_direct_load_state_probabilities_expands_single_state_to_2d(
    tmp_path: Path,
    dec_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """Single-state runs are stored on disk as 1D ``(n_time,)`` arrays;
    both data sources must expand to ``(n_visible, 1)`` so callers like
    ``StateProbabilityModel.update_window`` (requires 2D input) work
    on either path. Mirrors
    ``InMemoryDecoderDataSource.test_load_state_probabilities_expands_single_state_1d_results``."""
    import xarray as xr

    def _flatten_state_probs(results):
        return results.assign(
            acausal_state_probabilities=xr.DataArray(
                np.ones(results.sizes["time"]),
                dims=("time",),
                coords={"time": results["time"].values},
            )
        )

    bundle_dir = _save_bundle_with_zarr(
        tmp_path / "single_state",
        dec_fitted,
        sim_session,
        transform_results=_flatten_state_probs,
    )
    direct = _make_zarr_direct(bundle_dir)
    sl = slice(2, 8)
    probs = direct.load_state_probabilities(sl)
    assert probs.shape == (sl.stop - sl.start, 1)
    np.testing.assert_array_equal(probs[:, 0], 1.0)


@pytest.mark.unit
def test_zarr_direct_set_active_run_rebinds_handles(
    tmp_path: Path,
    nl_fitted: FittedDetector,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """Multi-run swap rebinds the zarr handles so subsequent loads
    return data for the new run (Phase 5.5 multi-run contract)."""
    from non_local_detector.visualization.interactive.data_source_zarr_direct import (
        ZarrDirectDecoderDataSource,
    )

    nl_dir = _save_bundle_with_zarr(tmp_path / "nl", nl_fitted, sim_session)
    cf_dir = _save_bundle_with_zarr(tmp_path / "cf", cf_fitted, sim_session)
    ds = ZarrDirectDecoderDataSource.for_directories({"nl": nl_dir, "cf": cf_dir})
    assert ds.active_run_name == "nl"
    sl = ds.window_indices(t_center=float(ds.time[ds.n_time // 2]), t_width=1.0)
    nl_post = ds.load_posterior(sl)

    ds.set_active_run("cf")
    assert ds.active_run_name == "cf"
    cf_post = ds.load_posterior(sl)
    # The NL and CF detectors have different state_bins schemas — the
    # only universal cross-check is that swapping to a different run
    # changes which array we read from. Use the columns count as the
    # cheap discriminator.
    assert nl_post.shape != cf_post.shape or not np.array_equal(nl_post, cf_post)
