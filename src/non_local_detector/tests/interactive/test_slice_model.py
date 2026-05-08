"""Tests for ``SliceModel`` per-bin slice behaviour against Track 0 fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.analysis.posterior import (
    PosteriorReduction,
    collapse_log_likelihood_to_position,
    collapse_posterior_to_position,
)
from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
    first_finite_row_index,
)
from non_local_detector.visualization.interactive.view_models.slice import (
    TOP_CURVE_LIKELIHOOD_LABEL,
    TOP_CURVE_POSTERIOR_FALLBACK_LABEL,
    SliceModel,
    collapse_log_likelihood_per_spatial_state,
)


def _t_idx_of_first_finite_loglik(results) -> int:
    """Return the first time index whose log-likelihood row is fully finite."""
    log_lik = results["log_likelihood"].values
    return first_finite_row_index(log_lik)


@pytest.mark.unit
class TestSliceModelDefaultReduction:
    def test_nl_picks_conditional_non_local(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        model = SliceModel(
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            time=nl_fitted.results["time"].values,
        )
        assert model.reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL

    def test_cf_picks_marginal(
        self, cf_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        model = SliceModel(
            detector=cf_fitted.detector,
            spike_times=sim_session.spike_times,
            time=cf_fitted.results["time"].values,
        )
        assert model.reduction is PosteriorReduction.MARGINAL


@pytest.mark.unit
class TestSliceModelTopCurve:
    def test_loglik_path_matches_helper_bit_identical(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        """Top curve via ``log_lik`` must match the analysis helper exactly."""
        results = nl_fitted.results
        time = results["time"].values
        t_idx = _t_idx_of_first_finite_loglik(results)
        log_lik_row = results["log_likelihood"].values[t_idx]
        posterior_row = results["acausal_posterior"].values[t_idx]

        model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
        payload = model.update_for_index(
            t_idx, posterior_row=posterior_row, log_lik_row=log_lik_row
        )

        expected = collapse_log_likelihood_to_position(log_lik_row, nl_fitted.detector)
        np.testing.assert_allclose(
            payload.top_curve, expected, atol=1e-14, equal_nan=True
        )
        expected_per_state = collapse_log_likelihood_per_spatial_state(
            log_lik_row, nl_fitted.detector
        )
        assert len(payload.top_curves) == len(expected_per_state)
        for actual, expected in zip(
            payload.top_curves, expected_per_state, strict=True
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)
        assert payload.top_curve_label == TOP_CURVE_LIKELIHOOD_LABEL

    def test_posterior_fallback_when_loglik_missing(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        """When ``log_lik_row=None`` the top curve falls back to collapsed posterior."""
        results = nl_fitted.results
        time = results["time"].values
        t_idx = _t_idx_of_first_finite_loglik(results)
        posterior_row = results["acausal_posterior"].values[t_idx]

        model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
        payload = model.update_for_index(
            t_idx, posterior_row=posterior_row, log_lik_row=None
        )

        expected = collapse_posterior_to_position(
            posterior_row, nl_fitted.detector, PosteriorReduction.CONDITIONAL_NON_LOCAL
        )
        np.testing.assert_allclose(
            payload.top_curve, expected, atol=1e-14, equal_nan=True
        )
        assert payload.top_curves == ()
        assert payload.top_curve_label == TOP_CURVE_POSTERIOR_FALLBACK_LABEL


@pytest.mark.unit
class TestSliceModelPredictiveOverlay:
    def test_predictive_path_matches_helper_bit_identical(
        self,
        run_bundles,
        sim_session: SimulatedSession,
    ) -> None:
        # ``predictive_posterior`` only appears in predict(return_outputs="all"),
        # which is the ``nl_all`` bundle variant (not the EM result on
        # ``nl_fitted``).
        bundle = run_bundles["nl_all"]
        results = bundle.results
        time = results["time"].values
        t_idx = _t_idx_of_first_finite_loglik(results)
        posterior_row = results["acausal_posterior"].values[t_idx]
        log_lik_row = results["log_likelihood"].values[t_idx]
        predictive_row = results["predictive_posterior"].values[t_idx]

        model = SliceModel(bundle.detector, sim_session.spike_times, time)
        payload = model.update_for_index(
            t_idx,
            posterior_row=posterior_row,
            log_lik_row=log_lik_row,
            predictive_row=predictive_row,
        )

        expected = collapse_posterior_to_position(
            predictive_row,
            bundle.detector,
            PosteriorReduction.MARGINAL,
        )
        assert payload.predictive_curve is not None
        np.testing.assert_allclose(
            payload.predictive_curve, expected, atol=1e-14, equal_nan=True
        )

    def test_predictive_hidden_when_missing(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        results = nl_fitted.results
        time = results["time"].values
        t_idx = _t_idx_of_first_finite_loglik(results)
        posterior_row = results["acausal_posterior"].values[t_idx]

        model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
        payload = model.update_for_index(
            t_idx, posterior_row=posterior_row, predictive_row=None
        )
        assert payload.predictive_curve is None


@pytest.mark.unit
class TestSliceModelPerCellRows:
    def test_cells_at_bin_with_real_spike(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        """Pick a real spike, find its bin, assert that cell appears in the slice."""
        time = nl_fitted.results["time"].values
        # Choose any cell with at least one spike inside the time window.
        cell_id = next(
            i
            for i, st in enumerate(sim_session.spike_times)
            if st.size and time[0] <= st[0] <= time[-1]
        )
        spike_t = float(sim_session.spike_times[cell_id][0])
        t_idx = int(np.searchsorted(time, spike_t, side="right") - 1)
        posterior_row = nl_fitted.results["acausal_posterior"].values[t_idx]

        model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
        payload = model.update_for_index(t_idx, posterior_row=posterior_row)
        active_ids = {c.cell_id for c in payload.cells}
        assert cell_id in active_ids
        # Place-field rows are peak-normalised (≤ 1, ≥ 0) and finite.
        for c in payload.cells:
            assert np.all(c.place_field_norm >= 0)
            assert np.all(c.place_field_norm <= 1.0 + 1e-12)
            assert np.all(np.isfinite(c.place_field_norm))

    def test_cells_at_empty_bin_returns_empty(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        """Bin with no spikes anywhere → empty cell tuple."""
        time = nl_fitted.results["time"].values
        # Build synthetic spike_times that have no events near the cursor.
        n_cells = len(sim_session.spike_times)
        far_off = float(time[-1] + 1000.0)
        empty_spikes = [np.array([far_off]) for _ in range(n_cells)]
        t_idx = len(time) // 2
        posterior_row = nl_fitted.results["acausal_posterior"].values[t_idx]

        model = SliceModel(nl_fitted.detector, empty_spikes, time)
        payload = model.update_for_index(t_idx, posterior_row=posterior_row)
        assert payload.cells == ()


@pytest.mark.unit
def test_set_active_run_rebinds_detector_and_spikes(
    nl_fitted: FittedDetector,
    cf_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``set_active_run`` swaps detector + spike_times + reduction default."""
    time = nl_fitted.results["time"].values
    model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
    assert model.reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL

    model.set_active_run(
        cf_fitted.detector,
        sim_session.spike_times,
        cf_fitted.results["time"].values,
    )
    assert model.detector is cf_fitted.detector
    assert model.reduction is PosteriorReduction.MARGINAL


@pytest.mark.unit
def test_update_for_index_validates_bounds(
    nl_fitted: FittedDetector, sim_session: SimulatedSession
) -> None:
    time = nl_fitted.results["time"].values
    model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
    posterior_row = nl_fitted.results["acausal_posterior"].values[0]
    with pytest.raises(IndexError):
        model.update_for_index(len(time), posterior_row=posterior_row)
    with pytest.raises(IndexError):
        model.update_for_index(-1, posterior_row=posterior_row)


@pytest.mark.unit
class TestSliceModelCellSlice:
    def test_cell_slice_returns_normalised_place_field(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        model = SliceModel(
            nl_fitted.detector,
            sim_session.spike_times,
            nl_fitted.results["time"].values,
        )
        cs = model.cell_slice(0)
        assert cs.cell_id == 0
        assert cs.spike_count == 0
        assert cs.place_field_norm.shape == (
            model._per_cell_pf_normalized.shape[1],  # type: ignore[has-type]
        )
        # Same row the active path would emit.
        np.testing.assert_array_equal(
            cs.place_field_norm,
            model._per_cell_pf_normalized[0],  # type: ignore[has-type]
        )

    def test_cell_slice_carries_explicit_spike_count(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        model = SliceModel(
            nl_fitted.detector,
            sim_session.spike_times,
            nl_fitted.results["time"].values,
        )
        cs = model.cell_slice(2, spike_count=7)
        assert cs.spike_count == 7

    def test_cell_slice_validates_bounds(
        self, nl_fitted: FittedDetector, sim_session: SimulatedSession
    ) -> None:
        model = SliceModel(
            nl_fitted.detector,
            sim_session.spike_times,
            nl_fitted.results["time"].values,
        )
        with pytest.raises(IndexError):
            model.cell_slice(-1)
        with pytest.raises(IndexError):
            model.cell_slice(model.n_cells)


@pytest.mark.unit
class TestPrecomputedBinIndex:
    """Per-bin cell-count lookup is built once at bind and matches the scan."""

    def test_cells_at_index_matches_scan_across_bins(
        self,
        nl_fitted: FittedDetector,
        sim_session: SimulatedSession,
    ) -> None:
        """``_cells_at_index(t_idx)`` produces the same cell set / counts as a
        bin-edge scan for every bin where data is available.

        The scan uses each bin's left-edge ``[t_lo, t_hi]`` edges
        (matching ``_bin_edges``) and counts spikes in ``[t_lo,
        t_hi]``. The new path uses ``searchsorted`` on precomputed
        edges, so the two methods should agree on every sampled bin.
        """
        from non_local_detector.visualization.interactive.view_models.slice import (
            SliceModel,
        )

        time = np.asarray(nl_fitted.results["time"].values)
        model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)

        # Spot-check across the session: head, tail, and a few middle
        # bins. Full-loop comparison is slow (n_bins ≈ 100k); a
        # representative sample catches a real drift.
        sample_indices = np.linspace(1, time.size - 2, num=50, dtype=int)
        for t_idx in sample_indices:
            t_lo, t_hi = model._bin_edges(int(t_idx))
            expected: dict[int, int] = {}
            for cell_id, st in enumerate(sim_session.spike_times):
                if st.size == 0:
                    continue
                count = int(np.count_nonzero((st >= t_lo) & (st < t_hi)))
                if count > 0:
                    expected[cell_id] = count
            actual = {
                c.cell_id: c.spike_count for c in model._cells_at_index(int(t_idx))
            }
            assert actual == expected, (
                f"bin {t_idx}: precomputed index disagrees with bin-edge scan. "
                f"expected={expected!r}, actual={actual!r}"
            )

    def test_event_index_rebuilt_after_active_run_swap(
        self,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
        sim_session: SimulatedSession,
    ) -> None:
        """``set_active_run`` rebuilds the run-local spike event index."""
        from non_local_detector.visualization.interactive.view_models.slice import (
            SliceModel,
        )

        time = np.asarray(nl_fitted.results["time"].values)
        model = SliceModel(nl_fitted.detector, sim_session.spike_times, time)
        nl_index_id = id(model._event_index)

        # Swap to CF (different detector — different state schema +
        # potentially different cell ordering).
        cf_time = np.asarray(cf_fitted.results["time"].values)
        model.set_active_run(cf_fitted.detector, sim_session.spike_times, cf_time)
        assert id(model._event_index) != nl_index_id
        assert model._event_index.time_indices.size > 0
        assert np.all(model._event_index.time_indices < cf_time.size)

    def test_left_edge_active_bin_convention(self, nl_fitted: FittedDetector) -> None:
        """Spikes at ``time[i]`` belong to bin ``i``, not the previous midpoint bin."""
        time = np.asarray(nl_fitted.results["time"].values[:5], dtype=float)
        spike_times = [
            np.array(
                [
                    time[1],
                    np.nextafter(time[2], time[1]),
                    time[2],
                ]
            )
        ]
        encoding_entry = next(iter(nl_fitted.detector.encoding_model_.values()))
        n_cells = encoding_entry["place_fields"].shape[0]
        spike_times.extend(np.array([], dtype=float) for _ in range(n_cells - 1))
        model = SliceModel(nl_fitted.detector, spike_times, time)

        assert {c.cell_id: c.spike_count for c in model._cells_at_index(1)} == {0: 2}
        assert {c.cell_id: c.spike_count for c in model._cells_at_index(2)} == {0: 1}
        assert model._bin_edges(1) == (pytest.approx(time[1]), pytest.approx(time[2]))
