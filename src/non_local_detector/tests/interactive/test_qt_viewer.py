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


@pytest.fixture
def qapp():
    """Provide a singleton QApplication for all GUI tests."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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
        first_finite_row_index(nl_fitted.results["log_likelihood"].values)
        : first_finite_row_index(nl_fitted.results["log_likelihood"].values) + 10
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
    assert builtin_types == [
        QtRasterPanel,
        QtStateProbabilityPanel,
        QtLikelihoodHeatmapPanel,
        QtPosteriorHeatmapPanel,
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

    n_pos = int(multi_run_bundles["nl"].detector.environments[0].place_bin_centers_.shape[0])
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
    # the body's QHBoxLayout split. (Right-column wrapper exists so
    # ``extra_bin_panels`` can stack below the slice panel.)
    body = viewer.centralWidget().layout().itemAt(1).widget()
    right_column = body.layout().itemAt(1).widget()
    right_column_widgets = [
        right_column.layout().itemAt(i).widget()
        for i in range(right_column.layout().count())
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
def test_qt_viewer_raster_click_toggles_slice_pin(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``raster.cell_clicked.emit(cell_id)`` toggles the slice panel's pin."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    assert viewer._slice_panel.pinned_cell_ids == frozenset()
    viewer._raster_panel.cell_clicked.emit(3)
    assert 3 in viewer._slice_panel.pinned_cell_ids
    # Click again → unpin (toggle semantics).
    viewer._raster_panel.cell_clicked.emit(3)
    assert 3 not in viewer._slice_panel.pinned_cell_ids


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
    np.testing.assert_array_equal(viewer._panel._position_centers, cf_centers)
    np.testing.assert_array_equal(viewer._likelihood_panel._position_centers, cf_centers)

    # Smoke check: a fresh load under CF must complete without raising.
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)
