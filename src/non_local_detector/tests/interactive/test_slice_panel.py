"""Tests for ``QtSlicePanel`` — buffer + render + truncation behaviour."""

from __future__ import annotations

import os

import numpy as np
import pytest

# Force offscreen Qt platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from non_local_detector.analysis.posterior import (
    PosteriorReduction,
    collapse_log_likelihood_to_position,
    collapse_posterior_to_position,
)
from non_local_detector.tests._simulated_detectors import (
    SimulatedSession,
    first_finite_row_index,
)
from non_local_detector.visualization.interactive.view_models.base import (
    RunBundle,
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.slice import SliceModel

pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    from non_local_detector.visualization.interactive.viewer.qt import (
        _ensure_qapplication,
    )

    app = _ensure_qapplication()
    yield app


def _payload_for_results(results, posterior_only: bool = False) -> WindowPayload:
    """Build a WindowPayload covering the full session (window = whole results)."""
    n_time = results.sizes["time"]
    sl = slice(0, n_time)
    return WindowPayload(
        request_id=0,
        time=np.asarray(results["time"].values),
        indices=sl,
        posterior=np.asarray(results["acausal_posterior"].values),
        likelihood=(
            None
            if posterior_only
            else np.asarray(results["log_likelihood"].values)
            if "log_likelihood" in results.data_vars
            else None
        ),
        predictive=(
            np.asarray(results["predictive_posterior"].values)
            if "predictive_posterior" in results.data_vars and not posterior_only
            else None
        ),
        state_probabilities=np.asarray(
            results["acausal_state_probabilities"].values
        ),
    )


def _make_panel(qapp, model: SliceModel, position_centers: np.ndarray):
    from non_local_detector.visualization.interactive.panels.qt.slice import (
        QtSlicePanel,
    )

    return QtSlicePanel(model=model, position_centers=position_centers)


@pytest.mark.unit
def test_slice_panel_renders_top_curve_from_loglik(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """``set_window_buffer`` + ``update_for_index`` renders a top curve
    that matches the analysis helper bit-identically."""
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    payload = _payload_for_results(bundle.results)
    panel.set_window_buffer(payload)

    t_idx = first_finite_row_index(bundle.results["log_likelihood"].values)
    panel.update_for_index(t_idx)

    expected = collapse_log_likelihood_to_position(
        bundle.results["log_likelihood"].values[t_idx], detector
    )
    x_data, y_data = panel._top_curve_item.getData()
    np.testing.assert_array_equal(x_data, centers)
    np.testing.assert_allclose(y_data, expected, atol=1e-14, equal_nan=True)


@pytest.mark.unit
def test_slice_panel_pins_axes_and_hidden_row_footprint(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """Per-tick rendering should not trigger slice autorange/layout churn."""
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    model = SliceModel(detector, bundle.spike_times, bundle.results["time"].values)
    panel = _make_panel(qapp, model, centers)

    top_autorange = panel._top_plot.getViewBox().state["autoRange"]
    assert top_autorange == [False, False]
    for row in panel._per_cell_rows:
        row_autorange = row.plot.getViewBox().state["autoRange"]
        assert row_autorange == [False, False]
        assert row.container.sizePolicy().retainSizeWhenHidden()


@pytest.mark.unit
def test_slice_panel_renders_predictive_overlay(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    payload = _payload_for_results(bundle.results)
    panel.set_window_buffer(payload)

    t_idx = first_finite_row_index(bundle.results["log_likelihood"].values)
    panel.update_for_index(t_idx)

    expected = collapse_posterior_to_position(
        bundle.results["predictive_posterior"].values[t_idx],
        detector,
        PosteriorReduction.CONDITIONAL_NON_LOCAL,
    )
    _, y_data = panel._predictive_curve_item.getData()
    np.testing.assert_allclose(y_data, expected, atol=1e-14, equal_nan=True)


@pytest.mark.unit
def test_slice_panel_falls_back_to_posterior_when_loglik_missing(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """Default-predict bundle (no log_likelihood) renders posterior fallback."""
    bundle = run_bundles["nl_default"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    payload = _payload_for_results(bundle.results, posterior_only=True)
    panel.set_window_buffer(payload)

    t_idx = len(time) // 2
    panel.update_for_index(t_idx)

    expected = collapse_posterior_to_position(
        bundle.results["acausal_posterior"].values[t_idx],
        detector,
        PosteriorReduction.CONDITIONAL_NON_LOCAL,
    )
    _, y_data = panel._top_curve_item.getData()
    np.testing.assert_allclose(y_data, expected, atol=1e-14, equal_nan=True)
    # Predictive overlay hides — pyqtgraph maps empty/never-set to None.
    _, pred_y = panel._predictive_curve_item.getData()
    assert pred_y is None or pred_y.size == 0


@pytest.mark.unit
def test_slice_panel_out_of_buffer_index_is_no_op(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """``update_for_index`` outside the buffered window does NOT crash."""
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    # Buffer covers only first 50 time bins.
    sl = slice(0, 50)
    payload = WindowPayload(
        request_id=0,
        time=time[:50],
        indices=sl,
        posterior=bundle.results["acausal_posterior"].values[:50],
        likelihood=bundle.results["log_likelihood"].values[:50],
        predictive=bundle.results["predictive_posterior"].values[:50],
        state_probabilities=bundle.results["acausal_state_probabilities"].values[:50],
    )
    panel.set_window_buffer(payload)
    # Render at an in-buffer index first to populate state.
    panel.update_for_index(first_finite_row_index(payload.likelihood))
    in_buffer_y = panel._top_curve_item.getData()[1].copy()

    # Now request an out-of-buffer index — nothing should change.
    panel.update_for_index(500)
    after_y = panel._top_curve_item.getData()[1]
    np.testing.assert_array_equal(after_y, in_buffer_y)


@pytest.mark.unit
def test_slice_panel_per_cell_rows_show_active_cells(
    qapp,
    run_bundles: dict[str, RunBundle],
    sim_session: SimulatedSession,
) -> None:
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, sim_session.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    payload = _payload_for_results(bundle.results)
    panel.set_window_buffer(payload)

    # event_times is shape (n_events, 2). Scan inside the first event
    # window for a t_idx with >=1 spike — at 500 Hz with sparse firing
    # the midpoint bin often contains 0 spikes, so pick the first bin
    # that actually has activity instead of a fixed offset.
    event_start, event_end = sim_session.event_times[0]
    i_lo = int(np.searchsorted(time, event_start, side="left"))
    i_hi = int(np.searchsorted(time, event_end, side="right"))
    bin_dt = float(time[1] - time[0])
    t_idx = next(
        i
        for i in range(i_lo, i_hi)
        if any(
            np.any((st >= time[i] - bin_dt / 2) & (st <= time[i] + bin_dt / 2))
            for st in sim_session.spike_times
        )
    )
    panel.update_for_index(t_idx)

    visible_rows = [r for r in panel._per_cell_rows if not r.container.isHidden()]
    assert len(visible_rows) > 0
    for row in visible_rows:
        assert "#" in row.label.text()
        x, y = row.curve.getData()
        assert x.size == centers.size
        assert y.size == centers.size


@pytest.mark.unit
def test_slice_panel_truncation_indicator_when_more_than_pool(
    qapp,
    run_bundles: dict[str, RunBundle],
    sim_session: SimulatedSession,
) -> None:
    """When more than ``MAX_PER_CELL_PLOTS`` cells fire, label shows ``(+K more)``."""
    from non_local_detector.visualization.interactive.panels.qt.slice import (
        MAX_PER_CELL_PLOTS,
    )

    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values

    # Synthetic spikes: every cell fires at the cursor bin so the
    # active count exceeds MAX_PER_CELL_PLOTS.
    t_idx = len(time) // 2
    t_cursor = float(time[t_idx])
    n_cells = len(sim_session.spike_times)
    assert n_cells > MAX_PER_CELL_PLOTS  # otherwise the test is meaningless
    fake_spikes = [np.array([t_cursor]) for _ in range(n_cells)]

    model = SliceModel(detector, fake_spikes, time)
    panel = _make_panel(qapp, model, centers)
    panel.set_window_buffer(_payload_for_results(bundle.results))
    panel.update_for_index(t_idx)

    visible = [r for r in panel._per_cell_rows if not r.container.isHidden()]
    assert len(visible) == MAX_PER_CELL_PLOTS
    assert not panel._truncation_label.isHidden()
    assert f"+{n_cells - MAX_PER_CELL_PLOTS} more" in panel._truncation_label.text()


@pytest.mark.unit
class TestSlicePanelPinning:
    """Pin state on the panel — render order, rebind clears, validation."""

    def _setup(self, qapp, run_bundles, sim_session=None):
        bundle = run_bundles["nl_all"]
        detector = bundle.detector
        centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
        time = bundle.results["time"].values
        spike_times = (
            sim_session.spike_times if sim_session is not None else bundle.spike_times
        )
        model = SliceModel(detector, spike_times, time)
        panel = _make_panel(qapp, model, centers)
        panel.set_window_buffer(_payload_for_results(bundle.results))
        return bundle, panel

    def test_pin_unpin_toggle_updates_set(
        self, qapp, run_bundles: dict[str, RunBundle]
    ) -> None:
        _, panel = self._setup(qapp, run_bundles)
        panel.pin_cell(3)
        assert 3 in panel.pinned_cell_ids
        panel.pin_cell(7)
        assert {3, 7} <= set(panel.pinned_cell_ids)
        panel.unpin_cell(3)
        assert 3 not in panel.pinned_cell_ids
        panel.toggle_pin(7)  # was pinned → unpin
        assert 7 not in panel.pinned_cell_ids
        panel.toggle_pin(7)  # was unpinned → pin
        assert 7 in panel.pinned_cell_ids
        panel.clear_pins()
        assert panel.pinned_cell_ids == frozenset()

    def test_pin_validates_cell_id(
        self, qapp, run_bundles: dict[str, RunBundle]
    ) -> None:
        _, panel = self._setup(qapp, run_bundles)
        with pytest.raises(IndexError):
            panel.pin_cell(-1)
        with pytest.raises(IndexError):
            panel.pin_cell(panel.model.n_cells)

    def test_pinned_inactive_cell_renders_with_zero_count(
        self,
        qapp,
        run_bundles: dict[str, RunBundle],
    ) -> None:
        """Pin a cell that doesn't fire in the current bin → still rendered, count=0."""
        bundle, panel = self._setup(qapp, run_bundles)
        first_finite = first_finite_row_index(bundle.results["log_likelihood"].values)
        panel.update_for_index(first_finite)
        # Pin a specific cell.
        panel.pin_cell(11)
        # Re-render at the same bin via the auto-rerender path.
        visible_rows = [r for r in panel._per_cell_rows if not r.container.isHidden()]
        assert any("#11 ★" in r.label.text() for r in visible_rows)
        # And the count for the pinned-inactive case is 0.
        pinned_row_label = next(r.label.text() for r in visible_rows if "#11" in r.label.text())
        assert "(×0)" in pinned_row_label

    def test_pinned_active_cell_keeps_real_spike_count(
        self,
        qapp,
        run_bundles: dict[str, RunBundle],
        sim_session: SimulatedSession,
    ) -> None:
        """Pinned cell that *also* fires in the bin shows the real count."""
        bundle, panel = self._setup(qapp, run_bundles, sim_session)
        time = bundle.results["time"].values
        # Find a bin with at least one spike in cell_id=0.
        bin_dt = float(time[1] - time[0])
        st0 = sim_session.spike_times[0]
        first_in = next(s for s in st0 if time[0] <= s <= time[-1])
        t_idx = int(np.searchsorted(time, first_in, side="right") - 1)
        panel.update_for_index(t_idx)
        panel.pin_cell(0)
        visible_rows = [r for r in panel._per_cell_rows if not r.container.isHidden()]
        pinned_label = next(r.label.text() for r in visible_rows if "#0 " in r.label.text())
        assert "★" in pinned_label
        # Real count is whatever fell in the bin window — must be > 0.
        n_in_bin = int(
            np.count_nonzero(
                (st0 >= time[t_idx] - bin_dt / 2) & (st0 <= time[t_idx] + bin_dt / 2)
            )
        )
        assert n_in_bin > 0
        assert f"(×{n_in_bin})" in pinned_label

    def test_pin_render_order_pinned_first_then_active(
        self,
        qapp,
        run_bundles: dict[str, RunBundle],
        sim_session: SimulatedSession,
    ) -> None:
        """Pinned cells (sorted by cell_id) come before active not-already-pinned."""
        bundle, panel = self._setup(qapp, run_bundles, sim_session)
        time = bundle.results["time"].values
        bin_dt = float(time[1] - time[0])
        # Find an active bin.
        i_lo = int(np.searchsorted(time, sim_session.event_times[0, 0], side="left"))
        i_hi = int(np.searchsorted(time, sim_session.event_times[0, 1], side="right"))
        t_idx = next(
            i
            for i in range(i_lo, i_hi)
            if any(
                np.any((st >= time[i] - bin_dt / 2) & (st <= time[i] + bin_dt / 2))
                for st in sim_session.spike_times
            )
        )
        panel.update_for_index(t_idx)
        # Pin two cells that probably aren't active right now (high IDs).
        panel.pin_cell(panel.model.n_cells - 1)
        panel.pin_cell(panel.model.n_cells - 2)
        visible_rows = [r for r in panel._per_cell_rows if not r.container.isHidden()]
        labels = [r.label.text() for r in visible_rows]
        # Pinned cells appear first, sorted by cell_id ascending.
        assert "★" in labels[0]
        assert "★" in labels[1]
        # And the IDs go in ascending order.
        first_pinned_id = int(labels[0].split("#")[1].split(" ")[0])
        second_pinned_id = int(labels[1].split("#")[1].split(" ")[0])
        assert first_pinned_id < second_pinned_id

    def test_pin_dedupes_when_active_and_pinned(
        self,
        qapp,
        run_bundles: dict[str, RunBundle],
        sim_session: SimulatedSession,
    ) -> None:
        """A cell that's both pinned and active is rendered once, not twice."""
        bundle, panel = self._setup(qapp, run_bundles, sim_session)
        time = bundle.results["time"].values
        st0 = sim_session.spike_times[0]
        first_in = next(s for s in st0 if time[0] <= s <= time[-1])
        t_idx = int(np.searchsorted(time, first_in, side="right") - 1)
        panel.update_for_index(t_idx)
        panel.pin_cell(0)  # cell 0 is active at this bin
        visible_rows = [r for r in panel._per_cell_rows if not r.container.isHidden()]
        cell_0_rows = [r for r in visible_rows if "#0 " in r.label.text()]
        assert len(cell_0_rows) == 1

    def test_rebind_after_swap_clears_pins(
        self, qapp, run_bundles: dict[str, RunBundle]
    ) -> None:
        """Cell IDs are run-local; the swap must drop the pin set."""
        _, panel = self._setup(qapp, run_bundles)
        panel.pin_cell(3)
        panel.pin_cell(5)
        assert len(panel.pinned_cell_ids) == 2
        panel.rebind_after_swap()
        assert panel.pinned_cell_ids == frozenset()


@pytest.mark.unit
def test_slice_panel_overlay_mode_switches_overlay_source(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """``set_overlay_mode`` swaps between predictive and smoothed sources.

    On the same buffered payload, the dashed overlay curve must match
    the analysis-helper output for whichever row source the panel is
    currently configured to use, and ``"off"`` must drop the overlay
    entirely.
    """
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    payload = _payload_for_results(bundle.results)
    panel.set_window_buffer(payload)
    t_idx = first_finite_row_index(bundle.results["log_likelihood"].values)
    panel.update_for_index(t_idx)

    expected_predictive = collapse_posterior_to_position(
        bundle.results["predictive_posterior"].values[t_idx],
        detector,
        PosteriorReduction.CONDITIONAL_NON_LOCAL,
    )
    expected_smoothed = collapse_posterior_to_position(
        bundle.results["acausal_posterior"].values[t_idx],
        detector,
        PosteriorReduction.CONDITIONAL_NON_LOCAL,
    )

    # Default mode is "predictive".
    assert panel.overlay_mode == "predictive"
    _, y_pred = panel._predictive_curve_item.getData()
    np.testing.assert_allclose(y_pred, expected_predictive, atol=1e-14, equal_nan=True)

    # Switch to smoothed — overlay tracks the acausal-collapsed curve.
    panel.set_overlay_mode("smoothed")
    assert panel.overlay_mode == "smoothed"
    _, y_smooth = panel._predictive_curve_item.getData()
    np.testing.assert_allclose(y_smooth, expected_smoothed, atol=1e-14, equal_nan=True)

    # Switch to off — overlay hides.
    panel.set_overlay_mode("off")
    assert panel.overlay_mode == "off"
    _, y_off = panel._predictive_curve_item.getData()
    assert y_off is None or y_off.size == 0


@pytest.mark.unit
def test_slice_panel_overlay_combo_drives_set_overlay_mode(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """User-facing combo box selection routes through ``set_overlay_mode``."""
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    panel.set_window_buffer(_payload_for_results(bundle.results))
    t_idx = first_finite_row_index(bundle.results["log_likelihood"].values)
    panel.update_for_index(t_idx)

    smoothed_idx = next(
        j
        for j in range(panel._overlay_combo.count())
        if panel._overlay_combo.itemData(j) == "smoothed"
    )
    panel._overlay_combo.setCurrentIndex(smoothed_idx)
    assert panel.overlay_mode == "smoothed"
    expected_smoothed = collapse_posterior_to_position(
        bundle.results["acausal_posterior"].values[t_idx],
        detector,
        PosteriorReduction.CONDITIONAL_NON_LOCAL,
    )
    _, y = panel._predictive_curve_item.getData()
    np.testing.assert_allclose(y, expected_smoothed, atol=1e-14, equal_nan=True)


@pytest.mark.unit
def test_slice_panel_rebind_after_swap_clears_buffer(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """``rebind_after_swap`` drops the stale buffer + clears all rendered items."""
    bundle = run_bundles["nl_all"]
    detector = bundle.detector
    centers = np.asarray(detector.environments[0].place_bin_centers_).squeeze()
    time = bundle.results["time"].values
    model = SliceModel(detector, bundle.spike_times, time)
    panel = _make_panel(qapp, model, centers)

    panel.set_window_buffer(_payload_for_results(bundle.results))
    panel.update_for_index(first_finite_row_index(bundle.results["log_likelihood"].values))
    assert panel._top_curve_item.getData()[1].size > 0

    panel.rebind_after_swap()
    assert panel._buffered_payload is None
    top_y = panel._top_curve_item.getData()[1]
    assert top_y is None or top_y.size == 0
    assert panel._truncation_label.isHidden()
    for row in panel._per_cell_rows:
        assert row.container.isHidden()
    # An update_for_index after rebind without re-buffering is a no-op.
    panel.update_for_index(0)
    top_y = panel._top_curve_item.getData()[1]
    assert top_y is None or top_y.size == 0
