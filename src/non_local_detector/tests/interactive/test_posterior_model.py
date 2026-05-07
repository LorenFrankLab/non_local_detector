"""Tests for ``PosteriorHeatmapModel``."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.analysis.posterior import (
    PosteriorReduction,
    conditional_non_local_posterior,
)
from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
    first_finite_row_index,
)
from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.view_models.base import (
    RunBundle,
)
from non_local_detector.visualization.interactive.view_models.posterior import (
    PosteriorHeatmapModel,
)


def _model(detector) -> PosteriorHeatmapModel:
    return PosteriorHeatmapModel(detector)


def _full_window_posterior(bundle: RunBundle) -> np.ndarray:
    """Whole-session ``acausal_posterior`` slice (every time bin visible)."""
    return np.asarray(bundle.results["acausal_posterior"].values)


def _midwindow_slice(bundle: RunBundle, n_visible: int = 50) -> np.ndarray:
    """Pick a small contiguous slice with non-zero mass for assertion targets."""
    post = _full_window_posterior(bundle)
    start = first_finite_row_index(post)
    assert post.shape[0] - start >= n_visible
    return post[start : start + n_visible]


@pytest.mark.unit
class TestPosteriorHeatmapModelStrategySelection:
    """Construction picks the right reduction strategy per detector schema."""

    def test_nl_picks_conditional_non_local(self, nl_fitted: FittedDetector) -> None:
        assert (
            _model(nl_fitted.detector).reduction
            is PosteriorReduction.CONDITIONAL_NON_LOCAL
        )

    def test_cf_picks_marginal(self, cf_fitted: FittedDetector) -> None:
        assert _model(cf_fitted.detector).reduction is PosteriorReduction.MARGINAL

    def test_nsf_picks_conditional_on_spatial(self, nsf_fitted: FittedDetector) -> None:
        assert (
            _model(nsf_fitted.detector).reduction
            is PosteriorReduction.CONDITIONAL_ON_SPATIAL
        )

    def test_dec_picks_marginal(self, dec_fitted: FittedDetector) -> None:
        assert _model(dec_fitted.detector).reduction is PosteriorReduction.MARGINAL

    def test_user_override_supersedes_auto_detect(
        self, nl_fitted: FittedDetector
    ) -> None:
        model = PosteriorHeatmapModel(
            nl_fitted.detector, reduction=PosteriorReduction.MARGINAL
        )
        assert model.reduction is PosteriorReduction.MARGINAL


@pytest.mark.unit
class TestPosteriorHeatmapModelOutput:
    """``update_window`` returns a ``(n_visible, n_pos)`` array per detector."""

    def test_nl_matches_dataset_helper(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        """NL heatmap rows bit-identical (within float64) to the
        dataset-level ``conditional_non_local_posterior``.

        Both call sites route through ``_conditional_row``; the
        dataset helper's output is the canonical reference for the
        static-plot panel. The NaN re-padding the dataset helper does
        on non-interior bins falls outside the heatmap path, so we
        compare on the interior subset.
        """
        bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
            speed=sim_session.speed,
        )
        window = _midwindow_slice(bundle, n_visible=20)
        out = _model(bundle.detector).update_window(window)

        env = bundle.detector.environments[0]
        n_pos = int(env.place_bin_centers_.shape[0])
        assert out.shape == (window.shape[0], n_pos)

        # Run the dataset helper on the same slice and compare on
        # interior columns (the dataset helper writes NaN on
        # non-interior; the per-row heatmap leaves whatever the
        # reduction produced).
        sliced_results = nl_fitted.results.isel(
            time=slice(*_window_slice_bounds(bundle, n_visible=20))
        )
        expected = conditional_non_local_posterior(
            sliced_results, bundle.detector
        ).values
        interior = np.asarray(env.is_track_interior_)
        np.testing.assert_allclose(
            out[:, interior],
            expected[:, interior],
            atol=1e-12,
            equal_nan=True,
        )

    def test_cf_marginal_rows_sum_to_one(
        self, cf_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        bundle = RunBundle(
            results=cf_fitted.results,
            detector=cf_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        window = _midwindow_slice(bundle, n_visible=20)
        out = _model(bundle.detector).update_window(window)
        # Float32 precision in posterior storage; allow 1e-6.
        np.testing.assert_allclose(np.nansum(out, axis=-1), 1.0, atol=1e-6)

    def test_dec_marginal_rows_sum_to_one(
        self, dec_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        bundle = RunBundle(
            results=dec_fitted.results,
            detector=dec_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        window = _midwindow_slice(bundle, n_visible=20)
        out = _model(bundle.detector).update_window(window)
        np.testing.assert_allclose(np.nansum(out, axis=-1), 1.0, atol=1e-6)

    def test_nsf_conditional_on_spatial_active_rows_sum_to_one(
        self, nsf_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        bundle = RunBundle(
            results=nsf_fitted.results,
            detector=nsf_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        # Pick rows with positive spatial mass (>0.01 to skip
        # zero-mass / fill rows).
        no_spike = nsf_fitted.results["acausal_state_probabilities"].sel(
            states="No-Spike"
        )
        spatial_mass_per_t = 1.0 - no_spike.values
        active = np.flatnonzero(spatial_mass_per_t > 0.01)
        assert active.size >= 5
        idx_start = int(active[0])
        window = _full_window_posterior(bundle)[idx_start : idx_start + 20]

        out = _model(bundle.detector).update_window(window)
        # On rows with positive spatial mass, the reduction
        # renormalizes; sum should be 1.0 within float32 precision.
        rows_active = spatial_mass_per_t[idx_start : idx_start + 20] > 0.01
        np.testing.assert_allclose(np.nansum(out[rows_active], axis=-1), 1.0, atol=1e-6)

    def test_window_shape_matches_n_pos(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        env = bundle.detector.environments[0]
        n_pos = int(env.place_bin_centers_.shape[0])
        window = _midwindow_slice(bundle, n_visible=12)
        out = _model(bundle.detector).update_window(window)
        assert out.shape == (12, n_pos)

    def test_empty_window_preserves_n_pos_axis(self, nl_fitted: FittedDetector) -> None:
        """``update_window`` on an empty window must still return shape
        ``(0, n_pos)`` — downstream renderers infer the position grid
        width from this axis. Regression test for the
        ``np.empty((0, 0))`` bug.
        """
        env = nl_fitted.detector.environments[0]
        n_pos = int(env.place_bin_centers_.shape[0])
        n_state_bins = int(nl_fitted.detector.n_state_bins_)
        empty_window = np.empty((0, n_state_bins), dtype=np.float64)
        out = _model(nl_fitted.detector).update_window(empty_window)
        assert out.shape == (0, n_pos)

    def test_collapse_at_empty_indices_preserves_n_pos_axis(
        self, nl_fitted: FittedDetector
    ) -> None:
        """``collapse_at`` with empty ``indices`` returns shape ``(0, n_pos)``."""
        env = nl_fitted.detector.environments[0]
        n_pos = int(env.place_bin_centers_.shape[0])
        n_state_bins = int(nl_fitted.detector.n_state_bins_)
        window = np.zeros((5, n_state_bins), dtype=np.float64)
        out = _model(nl_fitted.detector).collapse_at(window, indices=[])
        assert out.shape == (0, n_pos)


@pytest.mark.unit
class TestPosteriorHeatmapModelSchemaSwap:
    """``set_active_run`` rebinds strategy + produces different rows per detector."""

    def test_swap_rebinds_reduction_strategy(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        expected = {
            "nl": PosteriorReduction.CONDITIONAL_NON_LOCAL,
            "cf": PosteriorReduction.MARGINAL,
            "nsf": PosteriorReduction.CONDITIONAL_ON_SPATIAL,
            "dec": PosteriorReduction.MARGINAL,
        }
        for name in ("nl", "cf", "nsf", "dec"):
            ds.set_active_run(name)
            model = PosteriorHeatmapModel(ds.active_run.detector)
            assert model.reduction is expected[name], (
                f"Run {name!r} expected {expected[name]!r}, got {model.reduction!r}"
            )

    def test_set_active_run_changes_reduction_in_place(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        ds.set_active_run("nl")
        model = PosteriorHeatmapModel(ds.active_run.detector)
        assert model.reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL
        ds.set_active_run("cf")
        model.set_active_run(ds.active_run.detector)
        assert model.reduction is PosteriorReduction.MARGINAL

    def test_swap_produces_distinct_outputs_at_same_index(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        """Same `t_idx`, different detector → different heatmap row.

        Picks a time bin known to have spatial mass for every detector
        (so the CONDITIONAL_NON_LOCAL / CONDITIONAL_ON_SPATIAL paths
        don't fall back to ``zero_mass_fill`` and produce all-NaN).
        """
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        # Pick a time bin in the middle of the session — past the
        # initial-burn-in NaN rows and likely to have spatial mass
        # under every reduction strategy.
        n_time = ds.n_time
        t_idx = n_time // 2
        outputs: dict[str, np.ndarray] = {}
        for name in ("nl", "cf", "nsf", "dec"):
            ds.set_active_run(name)
            row = ds.slice_at_index(t_idx, which="posterior")
            assert row is not None
            model = PosteriorHeatmapModel(ds.active_run.detector)
            collapsed = model.update_window(row[np.newaxis, :])
            outputs[name] = collapsed.squeeze()
        # Different schemas → different reductions → different rows.
        # Compare on overlapping finite bins.
        names = list(outputs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = outputs[names[i]], outputs[names[j]]
                if a.shape != b.shape:
                    # Different shapes is itself a different output.
                    continue
                finite = np.isfinite(a) & np.isfinite(b)
                if not finite.any():
                    # Both rows fall to NaN-fill; can't compare. Try a
                    # different bin.
                    continue
                assert np.any(a[finite] != b[finite]), (
                    f"Outputs for {names[i]!r} and {names[j]!r} are "
                    "identical — schema swap did not change the row."
                )


def _window_slice_bounds(bundle: RunBundle, n_visible: int) -> tuple[int, int]:
    """Match ``_midwindow_slice``'s start/stop selection."""
    start = first_finite_row_index(_full_window_posterior(bundle))
    return start, start + n_visible
