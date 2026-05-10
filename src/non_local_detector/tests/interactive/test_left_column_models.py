"""Tests for ``LikelihoodHeatmapModel``, ``StateProbabilityModel``, and ``RasterModel``."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.analysis.posterior import collapse_log_likelihood_to_position
from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
    first_finite_row_index,
)
from non_local_detector.visualization.interactive.view_models.likelihood import (
    LikelihoodHeatmapModel,
)
from non_local_detector.visualization.interactive.view_models.raster import (
    RasterModel,
)
from non_local_detector.visualization.interactive.view_models.state_prob import (
    StateProbabilityModel,
)


def _n_pos(detector) -> int:
    return int(detector.environments[0].place_bin_centers_.shape[0])


# ---------------------------------------------------------------------------
# LikelihoodHeatmapModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLikelihoodHeatmapModel:
    def test_nl_window_matches_per_row_helper(self, nl_fitted: FittedDetector) -> None:
        log_lik = nl_fitted.results["log_likelihood"].values
        start = first_finite_row_index(log_lik)
        window = log_lik[start : start + 20]
        out = LikelihoodHeatmapModel(nl_fitted.detector).update_window(window)
        n_pos = _n_pos(nl_fitted.detector)
        assert out.shape == (window.shape[0], n_pos)
        # Each row bit-identical to the per-row helper.
        for i in range(window.shape[0]):
            expected = collapse_log_likelihood_to_position(
                window[i], nl_fitted.detector
            )
            np.testing.assert_allclose(out[i], expected, atol=1e-14, equal_nan=True)

    def test_cf_rectangular_shape(self, cf_fitted: FittedDetector) -> None:
        log_lik = cf_fitted.results["log_likelihood"].values
        start = first_finite_row_index(log_lik)
        window = log_lik[start : start + 12]
        out = LikelihoodHeatmapModel(cf_fitted.detector).update_window(window)
        assert out.shape == (12, _n_pos(cf_fitted.detector))

    def test_all_nan_window_returns_all_zero_rows(
        self, nl_fitted: FittedDetector
    ) -> None:
        n_state_bins = nl_fitted.detector.n_state_bins_
        n_pos = _n_pos(nl_fitted.detector)
        window = np.full((5, n_state_bins), np.nan)
        out = LikelihoodHeatmapModel(nl_fitted.detector).update_window(window)
        np.testing.assert_array_equal(out, np.zeros((5, n_pos)))

    def test_empty_window_preserves_n_pos_axis(self, nl_fitted: FittedDetector) -> None:
        n_state_bins = nl_fitted.detector.n_state_bins_
        n_pos = _n_pos(nl_fitted.detector)
        out = LikelihoodHeatmapModel(nl_fitted.detector).update_window(
            np.empty((0, n_state_bins))
        )
        assert out.shape == (0, n_pos)

    def test_collapse_at_matches_update_window_subset(
        self, nl_fitted: FittedDetector
    ) -> None:
        """``collapse_at(window, indices)`` equals ``update_window(window[indices])``.

        The 2D viewer pulls a single cursor row at a time via
        ``collapse_at``; the parity guarantee with ``update_window``
        means the at-cursor image stays consistent with what the
        full-window heatmap would have shown.
        """
        log_lik = nl_fitted.results["log_likelihood"].values
        start = first_finite_row_index(log_lik)
        window = log_lik[start : start + 20]
        model = LikelihoodHeatmapModel(nl_fitted.detector)

        full = model.update_window(window)
        indices = [0, 5, 12, 19]
        partial = model.collapse_at(window, indices)

        assert partial.shape == (len(indices), full.shape[1])
        np.testing.assert_array_equal(partial, full[indices])

    def test_collapse_at_empty_indices_preserves_n_pos_axis(
        self, nl_fitted: FittedDetector
    ) -> None:
        log_lik = nl_fitted.results["log_likelihood"].values
        start = first_finite_row_index(log_lik)
        window = log_lik[start : start + 5]
        out = LikelihoodHeatmapModel(nl_fitted.detector).collapse_at(window, [])
        assert out.shape == (0, _n_pos(nl_fitted.detector))


# ---------------------------------------------------------------------------
# StateProbabilityModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStateProbabilityModel:
    def test_state_names_from_detector(self, nl_fitted: FittedDetector) -> None:
        model = StateProbabilityModel(nl_fitted.detector)
        assert model.state_names == list(nl_fitted.detector.state_names)

    def test_window_passes_through(self, nl_fitted: FittedDetector) -> None:
        probs = nl_fitted.results["acausal_state_probabilities"].values
        window = probs[:30]
        model = StateProbabilityModel(nl_fitted.detector)
        out = model.update_window(window)
        np.testing.assert_array_equal(out, window)

    def test_set_active_run_updates_state_names(
        self,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
    ) -> None:
        model = StateProbabilityModel(nl_fitted.detector)
        assert len(model.state_names) == 4  # NL has 4 states
        model.set_active_run(cf_fitted.detector)
        assert len(model.state_names) == 2  # CF has 2 states
        assert model.state_names == list(cf_fitted.detector.state_names)

    def test_window_validates_state_count(self, nl_fitted: FittedDetector) -> None:
        model = StateProbabilityModel(nl_fitted.detector)
        # Wrong number of state columns.
        bad = np.zeros((10, 99))
        with pytest.raises(ValueError, match="states"):
            model.update_window(bad)


# ---------------------------------------------------------------------------
# RasterModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRasterModel:
    def test_sort_indices_match_static_plot_pattern(
        self,
        nl_fitted: FittedDetector,
        sim_session: SimulatedSession,
    ) -> None:
        """Sort order matches the static-plot place-field-peak sort."""
        from non_local_detector.analysis.place_fields import (
            extract_per_cell_place_fields,
        )

        env = nl_fitted.detector.environments[0]
        place_fields = extract_per_cell_place_fields(nl_fitted.detector)
        expected = np.argsort(
            env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
        )
        model = RasterModel(nl_fitted.detector, sim_session.spike_times)
        np.testing.assert_array_equal(model.sort_indices, expected)
        assert model.cell_label == "Neuron"

    def test_update_window_clips_per_cell(
        self,
        nl_fitted: FittedDetector,
        sim_session: SimulatedSession,
    ) -> None:
        model = RasterModel(nl_fitted.detector, sim_session.spike_times)
        # Pick a 1-second slice in the middle of the session.
        t = sim_session.time
        t_mid = float(t[len(t) // 2])
        t_start, t_stop = t_mid - 0.5, t_mid + 0.5
        payload = model.update_window(t_start, t_stop)

        # Per-cell spike arrays must all be within the window.
        for cell_spikes in payload.spike_times_per_cell:
            if cell_spikes.size:
                assert cell_spikes.min() >= t_start
                assert cell_spikes.max() <= t_stop

        # Sort order in the payload matches the model's sort.
        np.testing.assert_array_equal(payload.sort_indices, model.sort_indices)

    def test_set_active_run_with_new_detector(
        self,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
        sim_session: SimulatedSession,
    ) -> None:
        model = RasterModel(nl_fitted.detector, sim_session.spike_times)
        nl_sort = model.sort_indices.copy()
        # Both detectors share spike_times in this fixture, but the
        # encoding-model entries are independent fits → likely
        # different sort orders.
        model.set_active_run(cf_fitted.detector)
        assert model.cell_label == "Neuron"
        # Sort length unchanged (same n_neurons), but the peaks differ.
        assert len(model.sort_indices) == len(nl_sort)
