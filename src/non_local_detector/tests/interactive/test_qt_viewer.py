"""Headless Qt viewer smoke tests.

Marked ``@pytest.mark.gui`` because they construct an actual
``QApplication`` and ``QtViewer`` window. They run on the offscreen
platform plugin so they don't require a display server.

These do *not* assert pixel-perfect rendering — they verify that the
viewer launches, the model collapses correctly, and the slider /
backend wiring delivers a payload to the panel without crashing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Force offscreen Qt platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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

pytestmark = pytest.mark.gui


@pytest.mark.unit
def test_backend_worker_builds_linear_likelihood_for_filtered_overlay() -> None:
    """Linear likelihood is precomputed outside the slice widget path."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        _linear_likelihood_from_log,
    )

    log_lik = np.array(
        [
            [0.0, -1.0, -np.inf],
            [np.nan, np.nan, np.nan],
        ]
    )
    linear = _linear_likelihood_from_log(log_lik)

    assert linear is not None
    np.testing.assert_allclose(linear[0], [1.0, np.exp(-1.0), 0.0])
    np.testing.assert_array_equal(linear[1], [0.0, 0.0, 0.0])


@pytest.fixture
def qapp():
    """Provide a singleton QApplication for all GUI tests."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        _ensure_qapplication,
    )

    app = _ensure_qapplication()
    yield app


@pytest.mark.unit
def test_qt_panel_renders_collapsed_array(
    qapp,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``QtPosteriorHeatmapPanel`` accepts a real posterior window
    without raising.

    Numerically the rendered ImageItem array is the model's output
    (transposed for row-major). We assert shape + non-empty.
    """
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = nl_fitted.detector
    env = detector.environments[0]
    model = PosteriorHeatmapModel(detector)
    panel = QtPosteriorHeatmapPanel(
        model=model,
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )
    post = nl_fitted.results["acausal_posterior"].values
    start = first_finite_row_index(post)
    window = post[start : start + 50]
    panel.update_for_array(
        time=np.linspace(0.0, 1.0, window.shape[0]),
        posterior=window,
    )
    image = panel._image_item.image
    # ImageItem stores the transposed (n_pos, n_visible) array.
    n_pos = int(env.place_bin_centers_.shape[0])
    assert image.shape == (n_pos, window.shape[0])


@pytest.mark.unit
def test_qt_heatmap_rect_uses_left_edge_time_bounds(
    qapp,
    nl_fitted: FittedDetector,
) -> None:
    """Heatmap images span bin edges, not center-to-center sample times."""
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = nl_fitted.detector
    env = detector.environments[0]
    panel = QtPosteriorHeatmapPanel(
        model=PosteriorHeatmapModel(detector),
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )
    post = nl_fitted.results["acausal_posterior"].values
    start = first_finite_row_index(post)
    window = post[start : start + 3]
    panel.update_for_array(
        time=np.array([10.0, 10.1, 10.2]),
        posterior=window,
    )
    rect = panel._image_item.mapRectToParent(panel._image_item.boundingRect())

    assert rect.left() == pytest.approx(10.0)
    assert rect.right() == pytest.approx(10.3)


@pytest.mark.unit
def test_qt_viewer_launches_and_routes_payload(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``launch_qt(block=False)`` builds the window + keeps it alive.

    Regression: previously ``launch_qt`` returned without retaining a
    reference to the QtViewer, so PySide6 GC'd the window before the
    test could inspect it. The fix appends the viewer to a
    module-level ``_LIVE_VIEWERS`` registry; ``closeEvent`` removes it.

    Per-test ``_clear_qt_viewer_registry`` autouse fixture in
    ``conftest.py`` closes + clears the registry after the test, so
    later tests that count live windows aren't polluted by leaks
    from this one.
    """
    import gc

    from non_local_detector.visualization.interactive.viewer import qt as qt_mod
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtViewer,
        launch_qt,
    )

    code = launch_qt(multi_run_bundles, t_width=0.5, block=False)
    assert code == 0

    # Force a GC pass + drain the event queue. If the viewer
    # weren't retained, it would vanish here.
    gc.collect()
    qapp.processEvents()

    # The registry is the source of truth — `topLevelWidgets()`
    # can include closed-but-not-yet-deleted leftovers from prior
    # tests. The autouse cleanup fixture flushes those between tests
    # but defending against intra-test ordering matters too.
    live = [v for v in qt_mod._LIVE_VIEWERS if isinstance(v, QtViewer)]
    assert len(live) == 1
    viewer = live[0]
    # And it must also be in topLevelWidgets — i.e. PySide6 hasn't
    # GC'd it.
    assert viewer in qapp.topLevelWidgets()

    # closeEvent + WA_DeleteOnClose should pop the viewer off the
    # registry and ultimately delete the widget.
    viewer.close()
    qapp.processEvents()
    qapp.processEvents()
    assert viewer not in qt_mod._LIVE_VIEWERS


@pytest.mark.unit
def test_qt_viewer_set_active_run_via_core(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Construct viewer, then swap to a different run via ``core.set_active_run``."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer.core.active_run_name == "nl"
    viewer.core.set_active_run("cf")
    assert viewer.core.active_run_name == "cf"


@pytest.mark.unit
def test_event_overlay_mixin_renders_numpy_arrays(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Regression: ``set_event_overlays`` must accept multi-element NumPy arrays.

    The previous ``overlay.times or []`` fallback raised
    ``ValueError: ambiguous truth value`` because ``bool(np.array([1, 2]))``
    is ambiguous. The fix uses explicit ``is None`` comparisons.
    """
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = multi_run_bundles["nl"].detector
    env = detector.environments[0]
    panel = QtPosteriorHeatmapPanel(
        model=PosteriorHeatmapModel(detector),
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )

    overlays = [
        EventOverlay.points(
            name="multi-point",
            times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        EventOverlay.intervals(
            name="multi-interval",
            t_start=np.array([1.0, 5.0, 10.0]),
            t_end=np.array([2.0, 7.0, 12.0]),
        ),
    ]
    # Must not raise.
    panel.set_event_overlays(overlays)
    # 5 points + 3 intervals = 8 overlay items.
    assert len(panel._overlay_items) == 8

    # Idempotent: a fresh call replaces previously rendered items.
    panel.set_event_overlays([overlays[0]])
    assert len(panel._overlay_items) == 5

    # Empty list clears.
    panel.set_event_overlays([])
    assert len(panel._overlay_items) == 0


@pytest.mark.unit
def test_qt_viewer_swap_rebinds_panel_model(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Regression: swap from NL to CF must rebind PosteriorHeatmapModel.

    Before the fix, the panel held a model bound to the NL detector
    forever, so a CF posterior payload (n_state_bins=2*n_pos) would
    be collapsed under NL's state_ind_ (n_state_bins=n_pos+1+n_pos+n_pos)
    and produce garbage / crash.
    """
    from non_local_detector.analysis.posterior import PosteriorReduction
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    # Initial: NL → CONDITIONAL_NON_LOCAL.
    assert viewer._posterior_model.reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL
    assert viewer._posterior_model.detector is multi_run_bundles["nl"].detector

    viewer.core.set_active_run("cf")

    # After swap: CF → MARGINAL, model bound to the CF detector.
    assert viewer._posterior_model.reduction is PosteriorReduction.MARGINAL
    assert viewer._posterior_model.detector is multi_run_bundles["cf"].detector


@pytest.mark.unit
def test_qt_likelihood_panel_explains_missing_log_likelihood(
    qapp,
    nl_fitted: FittedDetector,
) -> None:
    """No-likelihood payload surfaces the re-``predict`` instruction in the title."""
    from non_local_detector.visualization.interactive.panels.qt.likelihood import (
        QtLikelihoodHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.likelihood import (
        LikelihoodHeatmapModel,
    )

    detector = nl_fitted.detector
    env = detector.environments[0]
    panel = QtLikelihoodHeatmapPanel(
        model=LikelihoodHeatmapModel(detector),
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )

    panel.update_window(
        WindowPayload(
            request_id=0,
            time=np.linspace(0.0, 1.0, 10),
            indices=slice(0, 10),
            likelihood=None,
        )
    )
    assert panel._title_message == panel.MISSING_DATA_MESSAGE
    assert "log_likelihood" in panel.MISSING_DATA_MESSAGE
    assert "predict" in panel.MISSING_DATA_MESSAGE
    # Mirror-only assertion would miss stale-text bugs: pg.setTitle(None)
    # hides the label without clearing cached text. Check the widget too.
    title_label = panel.plotItem.titleLabel
    assert title_label.text == panel.MISSING_DATA_MESSAGE
    assert title_label.isVisible()

    n_state_bins = detector.n_state_bins_
    log_lik = nl_fitted.results["log_likelihood"].values[
        first_finite_row_index(
            nl_fitted.results["log_likelihood"].values
        ) : first_finite_row_index(nl_fitted.results["log_likelihood"].values) + 10
    ]
    assert log_lik.shape == (10, n_state_bins)
    panel.update_window(
        WindowPayload(
            request_id=1,
            time=np.linspace(0.0, 1.0, 10),
            indices=slice(0, 10),
            likelihood=log_lik,
        )
    )
    assert panel._title_message is None
    assert title_label.text == ""
    assert not title_label.isVisible()


@pytest.mark.unit
def test_qt_viewer_constructs_full_left_column_stack(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """All four built-in left-column panels are constructed and laid out."""
    from non_local_detector.visualization.interactive.panels.qt.likelihood import (
        QtLikelihoodHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.panels.qt.raster import (
        QtRasterPanel,
    )
    from non_local_detector.visualization.interactive.panels.qt.state_prob import (
        QtStateProbabilityPanel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    builtin_types = [type(p) for p in viewer._builtin_panels]
    # Top-to-bottom visual order matches statespacecheck-paper-viewer:
    # heatmaps dominate; raster + state-prob are compact rows beneath.
    assert builtin_types == [
        QtPosteriorHeatmapPanel,
        QtLikelihoodHeatmapPanel,
        QtRasterPanel,
        QtStateProbabilityPanel,
    ]
    # Built-in panels live in the left column of the body's QHBoxLayout.
    left_col_layout = viewer._left_column_layout
    left_col_widgets = [
        left_col_layout.itemAt(i).widget() for i in range(left_col_layout.count())
    ]
    for panel in viewer._builtin_panels:
        assert panel in left_col_widgets
        assert panel in viewer._all_panels


@pytest.mark.unit
def test_qt_viewer_payload_routes_to_all_left_column_panels(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """A live load populates the render artifacts of every built-in panel."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    payload = viewer._backend._build_payload(viewer.core.current_view_state)

    assert payload.state_probabilities is not None
    n_visible = payload.state_probabilities.shape[0]
    assert payload.state_probabilities.shape == (
        n_visible,
        len(multi_run_bundles["nl"].detector.state_names),
    )
    assert payload.posterior is not None and payload.posterior.shape[0] == n_visible
    assert payload.likelihood is not None and payload.likelihood.shape[0] == n_visible

    viewer._on_window_loaded(payload)

    for line in viewer._state_prob_panel._lines:
        x, _ = line.getData()
        assert x.size == n_visible

    n_pos = int(
        multi_run_bundles["nl"].detector.environments[0].place_bin_centers_.shape[0]
    )
    assert viewer._likelihood_panel._image_item.image.shape == (n_pos, n_visible)
    assert viewer._panel._image_item.image.shape == (n_pos, n_visible)
    # ScatterPlotItem.getData returns (x, y); spike count may be 0 in
    # the window so just check API shape.
    assert len(viewer._raster_panel._scatter.getData()) == 2


@pytest.mark.unit
def test_qt_viewer_constructs_slice_panel_from_active_run(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """SliceModel + QtSlicePanel are built from ``data_source.active_run``."""
    from non_local_detector.visualization.interactive.panels.qt.slice import (
        QtSlicePanel,
    )
    from non_local_detector.visualization.interactive.view_models.slice import (
        SliceModel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    assert isinstance(viewer._slice_model, SliceModel)
    assert isinstance(viewer._slice_panel, QtSlicePanel)
    assert viewer._slice_model.detector is multi_run_bundles["nl"].detector
    # Slice panel sits inside the right-column wrapper widget under
    # the body splitter. (Right-column wrapper exists so
    # ``extra_bin_panels`` can stack below the slice panel.)
    splitter = viewer._body_splitter
    right_column = splitter.widget(1)
    right_column_widgets = [
        right_column.layout().itemAt(i).widget()
        for i in range(right_column.layout().count())
        if right_column.layout().itemAt(i).widget() is not None
    ]
    assert viewer._slice_panel in right_column_widgets


@pytest.mark.unit
def test_qt_viewer_slider_drives_slice_panel(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Slider tick reaches the slice panel via the synchronous per-tick path.

    Without this wiring, the slice panel only updates on async window
    loads, which adds ~16ms of perceived lag during scrubbing.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    # Prime the buffer first (the slice update is a no-op without it).
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)

    initial_y = viewer._slice_panel._top_curve_item.getData()[1]
    # Pick a target inside the buffered window so update_for_index
    # actually re-renders. The buffer covers ``payload.indices``;
    # offset by one bin from its start.
    sl = payload.indices
    target = sl.start + 1
    initial_t_idx = viewer._slider.value()
    assert target != initial_t_idx, "target must differ from initial slider value"
    viewer._on_slider_value_changed(target)
    after_y = viewer._slice_panel._top_curve_item.getData()[1]
    # Top curve changed (different t_idx → different row).
    assert initial_y is not None and after_y is not None
    assert not np.array_equal(initial_y, after_y)


@pytest.mark.unit
def test_qt_viewer_window_load_sets_slice_buffer(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``_on_window_loaded`` must populate ``slice_panel.set_window_buffer``."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer._slice_panel._buffered_payload is None

    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)
    assert viewer._slice_panel._buffered_payload is payload


@pytest.mark.unit
def test_qt_viewer_raster_spike_click_toggles_spike_pin(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Raster spike clicks pin by ``(cell_id, spike_time)`` identity."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    assert viewer._slice_panel.pinned_cell_ids == frozenset()
    viewer._raster_panel.spike_clicked.emit(3, 1.25)
    assert 3 in viewer._slice_panel.pinned_cell_ids
    assert viewer._pinned_spike_key == (3, 1.25)
    assert viewer._pinned_time == 1.25

    # Same spike → unpin.
    viewer._raster_panel.spike_clicked.emit(3, 1.25)
    assert 3 not in viewer._slice_panel.pinned_cell_ids
    assert viewer._pinned_spike_key is None

    # Another spike from the same cell repins/recenters to that event.
    viewer._raster_panel.spike_clicked.emit(3, 1.50)
    assert viewer._slice_panel.pinned_cell_ids == frozenset({3})
    assert viewer._pinned_spike_key == (3, 1.50)
    assert viewer.core.t_center == pytest.approx(1.50)


@pytest.mark.unit
def test_qt_viewer_raster_spike_click_updates_slice_to_spike_bin(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Clicking a visible spike renders that spike cell with a nonzero count."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    chosen: tuple[int, float, int] | None = None
    for cell_id, spike_times in enumerate(ds.active_run.spike_times):
        for spike_t in np.asarray(spike_times, dtype=float):
            t_idx = viewer._time_to_bin_index(float(spike_t))
            bucket = viewer._slice_model._per_bin_cell_counts[t_idx]
            if bucket.get(cell_id, 0) > 0:
                chosen = (cell_id, float(spike_t), t_idx)
                break
        if chosen is not None:
            break
    assert chosen is not None
    cell_id, spike_t, t_idx = chosen

    viewer.core.set_t_center(spike_t)
    viewer._on_window_loaded(
        viewer._backend._build_payload(viewer.core.current_view_state)
    )
    viewer._on_spike_clicked(cell_id, spike_t)

    assert viewer._slider.value() == t_idx
    visible_rows = [
        r for r in viewer._slice_panel._per_cell_rows if not r.container.isHidden()
    ]
    pinned_label = next(
        r.label.text() for r in visible_rows if f"#{cell_id}" in r.label.text()
    )
    expected_count = viewer._slice_model._per_bin_cell_counts[t_idx][cell_id]
    assert f"(×{expected_count})" in pinned_label
    assert "(×0)" not in pinned_label


@pytest.mark.unit
def test_qt_viewer_manual_navigation_clears_pins(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Manual scrub/step/reset interactions clear stale spike pins."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    viewer._on_spike_clicked(3, 1.25)
    assert viewer._slice_panel.pinned_cell_ids == frozenset({3})
    viewer._on_slider_value_changed(viewer._slider.value() + 1)
    assert viewer._slice_panel.pinned_cell_ids == frozenset()
    assert viewer._pinned_spike_key is None

    viewer._on_spike_clicked(3, 1.25)
    viewer._set_t_center_from_panel_click(1.75)
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    viewer._on_spike_clicked(3, 1.25)
    viewer._step_right()
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    viewer._on_spike_clicked(3, 1.25)
    viewer._step_window(+1)
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    viewer._on_spike_clicked(3, 1.25)
    viewer._reset_view()
    assert viewer._slice_panel.pinned_cell_ids == frozenset()


@pytest.mark.unit
def test_qt_viewer_esc_clears_pins(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``Esc`` shortcut routes to ``slice_panel.clear_pins()``."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._slice_panel.pin_cell(2)
    viewer._slice_panel.pin_cell(5)
    assert viewer._slice_panel.pinned_cell_ids == frozenset({2, 5})

    # Find the Esc shortcut and trigger it.
    from PySide6 import QtGui

    target_key = QtGui.QKeySequence("Escape")
    esc_shortcut = next(
        s
        for s in viewer.findChildren(QtGui.QShortcut)
        if s.key().toString() == target_key.toString()
    )
    esc_shortcut.activated.emit()
    assert viewer._slice_panel.pinned_cell_ids == frozenset()


@pytest.mark.unit
def test_qt_viewer_swap_rebinds_slice_model_and_clears_pins(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Swap rebinds the slice model from new active_run AND clears pins."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._slice_panel.pin_cell(0)
    assert 0 in viewer._slice_panel.pinned_cell_ids

    viewer.core.set_active_run("cf")
    cf = multi_run_bundles["cf"].detector

    assert viewer._slice_model.detector is cf
    # Cell IDs are run-local — pins must clear on swap.
    assert viewer._slice_panel.pinned_cell_ids == frozenset()


@pytest.mark.unit
def test_qt_viewer_swap_rebinds_built_in_panel_models(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Run swap rebinds every left-column model to the new schema."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    nl = multi_run_bundles["nl"].detector
    assert viewer._likelihood_model.detector is nl
    assert viewer._state_prob_model.detector is nl
    assert viewer._raster_model.detector is nl
    assert viewer._state_prob_model.state_names == list(nl.state_names)
    n_lines_nl = len(viewer._state_prob_panel._lines)

    viewer.core.set_active_run("cf")
    cf = multi_run_bundles["cf"].detector

    assert viewer._likelihood_model.detector is cf
    assert viewer._state_prob_model.detector is cf
    assert viewer._raster_model.detector is cf
    assert viewer._state_prob_model.state_names == list(cf.state_names)

    # State-prob panel must rebuild lines on swap (NL=4 states, CF=2);
    # without ``rebind_after_swap`` the line count would briefly
    # mismatch the new payload until the next ``_set_data`` rebuild.
    assert n_lines_nl == 4
    assert len(viewer._state_prob_panel._lines) == 2

    cf_centers = np.asarray(cf.environments[0].place_bin_centers_).squeeze()
    np.testing.assert_array_equal(viewer._panel.grid_layout.centers, cf_centers)
    np.testing.assert_array_equal(
        viewer._likelihood_panel.grid_layout.centers, cf_centers
    )

    # Smoke check: a fresh load under CF must complete without raising.
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)


# ---------------------------------------------------------------------------
# Layout parity (Chunk 1) — assertions on body splitter, stretch factors,
# slice top-alignment, and per-panel weights matching paper-viewer parity.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_qt_viewer_body_is_horizontal_qsplitter(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """The body container must be a QSplitter so the user can drag the divider.

    QHBoxLayout's stretch factors are static; QSplitter lets the user
    rebalance left vs right at runtime, matching the paper viewer.
    """
    from PySide6 import QtCore, QtWidgets

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    splitter = viewer._body_splitter
    assert isinstance(splitter, QtWidgets.QSplitter)
    assert splitter.orientation() == QtCore.Qt.Horizontal
    assert splitter.count() == 2
    # The splitter is the first child of the root QVBoxLayout; the
    # bottom controls bar now owns the center-time slider.
    root = viewer.centralWidget().layout()
    assert root.itemAt(0).widget() is splitter


@pytest.mark.unit
def test_qt_viewer_default_window_size_matches_paper_viewer(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Default window dimensions mirror statespacecheck-paper-viewer."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        _DEFAULT_WINDOW_HEIGHT,
        _DEFAULT_WINDOW_WIDTH,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    assert (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT) == (1200, 900)
    assert viewer.size().width() == _DEFAULT_WINDOW_WIDTH
    assert viewer.size().height() == _DEFAULT_WINDOW_HEIGHT


@pytest.mark.unit
def test_qt_viewer_body_splitter_stretch_factors_70_30(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Left:right body splitter weight ratio is roughly 70:30 (paper parity).

    QSplitter has no public getter for per-widget stretch factors, so
    we verify behaviour two ways: pin the module-level constants
    (a refactor that equalises them must update both places) and
    read ``splitter.sizes()`` post-construction (before the show
    event re-balances by Qt size policy).
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        _BODY_SPLITTER_LEFT_STRETCH,
        _BODY_SPLITTER_RIGHT_STRETCH,
        QtViewer,
    )

    assert _BODY_SPLITTER_LEFT_STRETCH > _BODY_SPLITTER_RIGHT_STRETCH
    constant_ratio = _BODY_SPLITTER_LEFT_STRETCH / _BODY_SPLITTER_RIGHT_STRETCH
    assert 2.0 <= constant_ratio <= 3.0, (
        f"left:right stretch constants ratio {constant_ratio:.2f} outside "
        "[2.0, 3.0] (target ~70:30)"
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    # ``QtViewer`` calls ``setSizes`` at construction. Read back to
    # confirm the call landed before the widget show event re-balances
    # by Qt size policy. The post-show balance varies by platform
    # (offscreen Qt sometimes equalises panes regardless of setSizes),
    # which is why this test reads the pre-show value rather than the
    # realised on-screen layout.
    sizes = viewer._body_splitter.sizes()
    assert sum(sizes) > 0, "splitter has no initial sizes set"
    initial_ratio = sizes[0] / max(sizes[1], 1)
    # Qt may pull the ratio toward 1:1 if the right column has a
    # large minimum size hint (the slice panel has per-cell rows
    # that demand vertical space). Accept anything clearly
    # left-dominant — a 50/50 regression would fail this.
    assert initial_ratio >= 1.5, (
        f"initial splitter sizes {sizes} ratio {initial_ratio:.2f} below 1.5 — "
        "viewer must call setSizes so the left column dominates"
    )


@pytest.mark.unit
def test_qt_viewer_left_column_stretch_heatmaps_dominate(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Posterior + likelihood heatmaps get a taller stretch than raster + state-prob.

    Mirrors the paper-viewer layout: the heatmaps are the primary
    visual context; raster and state-probability are compact rows.
    """
    from non_local_detector.visualization.interactive.panels.qt.likelihood import (
        QtLikelihoodHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.panels.qt.raster import (
        QtRasterPanel,
    )
    from non_local_detector.visualization.interactive.panels.qt.state_prob import (
        QtStateProbabilityPanel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    layout = viewer._left_column_layout

    stretch_by_type: dict[type, int] = {}
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is None:
            continue
        stretch_by_type[type(widget)] = layout.stretch(i)

    heatmap_stretch = min(
        stretch_by_type[QtPosteriorHeatmapPanel],
        stretch_by_type[QtLikelihoodHeatmapPanel],
    )
    compact_stretch = max(
        stretch_by_type[QtRasterPanel],
        stretch_by_type[QtStateProbabilityPanel],
    )
    assert heatmap_stretch > compact_stretch, (
        f"heatmap stretch ({heatmap_stretch}) must exceed compact stretch "
        f"({compact_stretch}); layout would otherwise distribute height equally."
    )


@pytest.mark.unit
def test_qt_viewer_right_column_top_aligned_with_trailing_stretch(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """The slice panel is top-aligned: trailing item is a stretch, not a widget.

    A trailing stretch keeps the slice panel aligned to the posterior
    heatmap's vertical extent instead of expanding to fill the full window.
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        _RIGHT_COLUMN_SLICE_STRETCH,
        _RIGHT_COLUMN_TRAILING_STRETCH,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    right_layout = viewer._right_column_layout
    n_items = right_layout.count()
    assert n_items >= 2
    # First item must be the slice panel.
    assert right_layout.itemAt(0).widget() is viewer._slice_panel
    assert right_layout.stretch(0) == _RIGHT_COLUMN_SLICE_STRETCH
    # Last item must be a spacer (stretch), not a widget.
    last_item = right_layout.itemAt(n_items - 1)
    assert last_item.widget() is None, (
        "trailing item must be a stretch, not a widget — slice panel "
        "would otherwise fill the full column height."
    )
    assert last_item.spacerItem() is not None
    assert right_layout.stretch(n_items - 1) == _RIGHT_COLUMN_TRAILING_STRETCH


@pytest.mark.unit
def test_qt_viewer_body_uses_tight_margins(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Outer margins + spacing are small so panels read as one figure.

    Default Qt margins (~9 px each side) produce visible gutters
    between panels. The paper viewer pulls them in tight.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    margins = viewer._left_column_layout.contentsMargins()
    assert margins.left() <= 4
    assert margins.top() <= 4
    assert margins.right() <= 4
    assert margins.bottom() <= 4
    assert viewer._left_column_layout.spacing() <= 4


# ---------------------------------------------------------------------------
# Position trace (Chunk 2) — white line on posterior + likelihood panels.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_window_payload_carries_position_for_visible_window(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """A live window load attaches the per-time-bin position to the payload."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    payload = viewer._backend._build_payload(viewer.core.current_view_state)

    assert payload.position is not None
    assert payload.position.ndim == 1
    assert payload.position.size == payload.time.size
    assert np.all(np.isfinite(payload.position))


def _expected_trace_y(panel, position: np.ndarray) -> np.ndarray:
    """Reference cm→pixel-y mapping (matches the panel's PositionTraceMixin).

    Bin centers sit at pixel centers (via the half-bin-padded
    ``setRect``), so plot a position cm at
    ``y0 + np.interp(cm, centers, arange) * uniform_step``. Equivalent
    to ``statespacecheck-paper-viewer``'s
    ``update_position_trajectory``.
    """
    return panel.grid_layout.cm_to_pixel_y(np.asarray(position))


@pytest.mark.unit
def test_posterior_panel_renders_white_position_trace(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``QtPosteriorHeatmapPanel`` draws a 1-px white trace mapped to pixel-y.

    The trace's y values are real cm interpolated through the
    panel's grid into the heatmap's pixel-center y so bin ``i`` of
    the trace lines up with row ``i`` of the image. On a uniform
    grid the mapping is identity in the interior, but values that
    exceed the grid endpoints clip to the last pixel center — the
    simulated fixture's position briefly hits 170 cm against a
    ~169 cm last-bin center, so we assert against the mapped value
    rather than raw cm.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)

    trace = viewer._panel._position_trace
    x, y = trace.getData()
    assert x is not None and y is not None
    assert x.size == payload.time.size
    np.testing.assert_array_equal(x, payload.time)
    expected = _expected_trace_y(viewer._panel, payload.position)
    np.testing.assert_allclose(y, expected, atol=1e-6)


@pytest.mark.unit
def test_likelihood_panel_renders_white_position_trace(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``QtLikelihoodHeatmapPanel`` mirrors the posterior panel's trace."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)

    trace = viewer._likelihood_panel._position_trace
    x, y = trace.getData()
    assert x.size == payload.time.size
    expected = _expected_trace_y(viewer._likelihood_panel, payload.position)
    np.testing.assert_allclose(y, expected, atol=1e-6)


@pytest.mark.unit
def test_position_trace_clears_when_payload_position_is_none(
    qapp,
    nl_fitted: FittedDetector,
) -> None:
    """Position trace empties when ``payload.position is None``.

    Direct ``update_window(None-position payload)`` exercises the
    branch panels take when a bundle ships without position (e.g.
    bundles built straight from ``predict()`` results without a
    behaviour stream).
    """
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = nl_fitted.detector
    env = detector.environments[0]
    panel = QtPosteriorHeatmapPanel(
        model=PosteriorHeatmapModel(detector),
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )
    posterior = nl_fitted.results["acausal_posterior"].values[:10]
    panel.update_window(
        WindowPayload(
            request_id=0,
            time=np.linspace(0.0, 1.0, 10),
            indices=slice(0, 10),
            posterior=posterior,
            position=None,
        )
    )
    x, y = panel._position_trace.getData()
    assert x is None or x.size == 0
    assert y is None or y.size == 0


@pytest.mark.unit
def test_position_trace_updates_after_active_run_swap(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """After an M-key swap, the next window load redraws the trace.

    The bundles share a session and therefore share position data,
    but the dispatch path must still re-render so any future bundle
    with distinct position picks up cleanly.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    payload_before = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload_before)

    next_run = next(name for name in ds.run_names if name != ds.active_run_name)
    viewer._core.set_active_run(next_run)
    payload_after = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload_after)

    x, y = viewer._panel._position_trace.getData()
    assert x is not None
    assert x.size == payload_after.time.size
    expected = _expected_trace_y(viewer._panel, payload_after.position)
    np.testing.assert_allclose(y, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Interaction-speed parity (Chunk 3) — debounce + one-in-flight coalescing
# in the backend adapter.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_backend_coalesces_burst_of_schedule_calls(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """A rapid burst of ``schedule_window_load`` calls submits exactly one job
    to the executor — the latest pending state.

    Replaces the previous "every schedule submits to executor" path
    where 30+ slider events per second swamped the executor with
    ~1 second of stale loads.
    """
    from PySide6 import QtCore

    from non_local_detector.visualization.interactive.viewer.core import (
        ViewerCore,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    backend = viewer._backend

    submitted: list[int] = []
    real_executor_submit = backend._executor.submit

    def _spy_submit(work):
        submitted.append(1)
        return real_executor_submit(work)

    backend._executor.submit = _spy_submit  # type: ignore[assignment]

    core = ViewerCore(ds, backend)
    # Stage three rapid t_center moves — each goes through schedule.
    for offset in (0.1, 0.2, 0.3):
        core.set_t_center(float(ds.time[ds.n_time // 2 + int(offset * 100)]))

    # Before the debounce timer fires, no executor submit yet.
    assert submitted == []
    assert backend._pending_state is not None

    # Drain the debounce timer (16 ms) — singleShot fires once on the
    # event loop. Process events until the timer fires.
    QtCore.QCoreApplication.processEvents()
    QtCore.QTest = None  # noqa: SLF001 — placeholder
    # Wait the debounce window deterministically.
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while submitted == [] and deadline.elapsed() < 200:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)

    # Exactly one executor submit despite three schedule calls.
    assert len(submitted) == 1, (
        f"debounce should coalesce burst into one executor submit; got {submitted}"
    )


@pytest.mark.unit
def test_backend_only_one_inflight_at_a_time(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """While a load is in-flight, new ``schedule_window_load`` calls park their
    state instead of submitting another executor job.

    Verified by holding the executor's worker on a sentinel until we
    enqueue several follow-up states; only one job runs at a time and
    the latest pending state is dispatched after the held job
    completes.
    """
    import threading

    from PySide6 import QtCore

    from non_local_detector.visualization.interactive.viewer.core import (
        ViewerCore,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    backend = viewer._backend
    submitted: list[int] = []
    release = threading.Event()

    real_executor_submit = backend._executor.submit
    real_build_payload = backend._build_payload

    def _slow_build_payload(state):
        # Block the worker until the test releases, simulating a
        # long-running window read.
        release.wait(timeout=2.0)
        return real_build_payload(state)

    backend._build_payload = _slow_build_payload  # type: ignore[assignment]

    def _spy_submit(work):
        submitted.append(1)
        return real_executor_submit(work)

    backend._executor.submit = _spy_submit  # type: ignore[assignment]

    core = ViewerCore(ds, backend)
    # First schedule: kicks the debounce.
    core.set_t_center(float(ds.time[ds.n_time // 2 + 10]))

    # Drain the debounce so the first job submits.
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while submitted == [] and deadline.elapsed() < 200:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)
    assert submitted == [1]
    assert backend._inflight is True

    # Now schedule several more while in-flight. None should submit.
    for offset in range(1, 4):
        core.set_t_center(float(ds.time[ds.n_time // 2 + 10 + offset]))
    QtCore.QCoreApplication.processEvents()
    assert submitted == [1], (
        "in-flight job must block new executor submits; "
        f"got {submitted} after parking states"
    )
    assert backend._pending_state is not None

    # Release the held job; the deliver path should re-arm the
    # debounce so the latest pending state runs next.
    release.set()
    deadline.restart()
    while len(submitted) < 2 and deadline.elapsed() < 1000:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)
    assert len(submitted) == 2, (
        f"latest pending state must run after the held job completes; got {submitted}"
    )


@pytest.mark.unit
def test_backend_recovers_when_worker_raises(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    caplog,
) -> None:
    """A ``_build_payload`` that raises must not freeze the backend.

    Bug repro: ``_flush_pending`` set ``_inflight=True`` before
    submitting; if the worker raised before emitting the done signal,
    ``_deliver_payload`` never ran and ``_inflight`` stayed True
    forever. Subsequent ``schedule_window_load`` calls only parked
    pending state, so the viewer silently stopped loading windows.
    """
    import logging

    from PySide6 import QtCore

    from non_local_detector.visualization.interactive.viewer.core import (
        ViewerCore,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    backend = viewer._backend

    # Phase 1: drive the first load through a raising _build_payload.
    real_build_payload = backend._build_payload
    raise_now = {"value": True}

    def _flaky_build_payload(state):
        if raise_now["value"]:
            raise RuntimeError("simulated worker failure")
        return real_build_payload(state)

    backend._build_payload = _flaky_build_payload  # type: ignore[assignment]

    core = ViewerCore(ds, backend)
    # Capture against this module's logger explicitly so the test
    # doesn't depend on propagation to root.
    caplog.set_level(
        logging.WARNING,
        logger="non_local_detector.visualization.interactive.viewer.qt",
    )
    core.set_t_center(float(ds.time[ds.n_time // 2 + 1]))

    # Drain the full ``debounce timer → worker → deliver`` cycle.
    # ``_inflight`` is False at the moment we return from
    # ``schedule_window_load`` (the debounce hasn't fired yet), so a
    # ``while _inflight`` loop would exit immediately. Wait until both
    # the pending-state slot and the inflight flag are cleared.
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while (
        backend._pending_state is not None
        or backend._inflight
        or backend._debounce_timer.isActive()
    ) and deadline.elapsed() < 500:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)

    assert backend._inflight is False, (
        "worker exception must clear _inflight via the delivery path"
    )
    # The error should have been logged through the deliver path so
    # the failure isn't silent.
    assert any(
        "window-load worker raised" in rec.getMessage() for rec in caplog.records
    ), (
        f"worker exception should be logged; got records: "
        f"{[rec.getMessage() for rec in caplog.records]!r}"
    )

    # Phase 2: stop raising; a fresh schedule must complete normally.
    raise_now["value"] = False
    delivered: list[int] = []
    core.on_window_loaded(lambda payload: delivered.append(payload.request_id))

    core.set_t_center(float(ds.time[ds.n_time // 2 + 2]))
    deadline.restart()
    while not delivered and deadline.elapsed() < 1000:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)
    assert delivered, (
        "schedule after worker exception must run to completion; "
        "_inflight likely never cleared"
    )


# ---------------------------------------------------------------------------
# Position trace registration on non-uniform grids — interpolate cm through
# the grid's pixel-y coords so the white trace lines up with image rows.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_position_trace_maps_through_non_uniform_grid(
    qapp,
    nl_fitted: FittedDetector,
) -> None:
    """Real-cm position is interpolated to the heatmap's uniform pixel-y.

    On a non-uniform position grid (e.g. linearised W-track), plotting
    raw cm on top of the image misregisters the trace against the
    image rows because ``ImageItem.setRect`` distributes rows
    uniformly between ``centers.min()`` and ``.max()``. The fix:
    interpolate cm through ``position_centers`` to a uniform-pixel
    space so a position falling in bin ``i`` is plotted at the y of
    image row ``i``.
    """
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = nl_fitted.detector
    # Synthetic non-uniform grid: pack the first half densely then
    # spread the second half. Real W-track linearisation looks like
    # this around junctions.
    n_pos = 10
    centers = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        dtype=np.float64,
    )
    assert centers.size == n_pos
    panel = QtPosteriorHeatmapPanel(
        model=PosteriorHeatmapModel(detector),
        position_centers=centers,
    )

    # Position vector targets bin 5 (cm == 5.0). On the non-uniform
    # grid that's row 5 of n_pos; the image row's y-center for a 10-
    # row image spanning [0, 80] is at y_min + (5.5 / 10) * 80 = 44.0.
    time = np.linspace(0.0, 1.0, 5)
    position = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    posterior = nl_fitted.results["acausal_posterior"].values[:5]
    panel.update_window(
        WindowPayload(
            request_id=0,
            time=time,
            indices=slice(0, 5),
            posterior=posterior,
            position=position,
        )
    )

    _, y = panel._position_trace.getData()
    # Expected: np.interp(5.0, centers, np.linspace(0, 80, 10))
    image_y = np.linspace(centers.min(), centers.max(), centers.size)
    expected = np.interp(position, centers, image_y)
    np.testing.assert_allclose(y, expected, atol=1e-12)
    # And critically, the mapped y is NOT the raw cm.
    assert not np.allclose(y, position), (
        "raw cm passed through unmapped — trace would misregister"
    )


@pytest.mark.unit
def test_position_trace_uniform_grid_round_trips(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """On a uniform grid, the cm→pixel-y mapping is the identity.

    Guards against the fix introducing drift on the common
    simulated-data case where position bins are evenly spaced.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)

    centers = viewer._panel.grid_layout.centers
    assert centers.size > 1
    # Verify uniformity — if this fixture ever moves to a non-uniform
    # grid the assertion will surface and we'll know to update the test.
    spacings = np.diff(centers)
    assert np.allclose(spacings, spacings[0], atol=1e-9), (
        "fixture position grid is no longer uniform; rewrite this test"
    )

    _, y = viewer._panel._position_trace.getData()
    expected = _expected_trace_y(viewer._panel, payload.position)
    np.testing.assert_allclose(y, expected, atol=1e-6)
    # On a uniform grid the mapping is identity for positions
    # *inside* the grid range. Ensure that's what the values
    # look like (with the simulated fixture's brief over-shoot
    # clipped to the last bin center, which is what the heatmap
    # itself does).
    in_range = (payload.position >= centers[0]) & (payload.position <= centers[-1])
    np.testing.assert_allclose(y[in_range], payload.position[in_range], atol=1e-6)
