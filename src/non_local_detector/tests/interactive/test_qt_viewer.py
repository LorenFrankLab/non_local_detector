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
    ViewState,
    WindowPayload,
)

pytestmark = pytest.mark.gui


@pytest.mark.unit
def test_slice_panel_linear_likelihood_row_helper() -> None:
    """Per-row peak-normalize-and-exp pins the filtered-overlay contract.

    The slice panel now derives the linear likelihood for the active
    bin only (was a full-window pass on the worker that scaled with
    ``n_visible`` and dominated wheel-resize latency). The numerical
    contract for one row is unchanged: finite entries get
    ``exp(log - row_max)``, all-non-finite rows return zeros.
    """
    from non_local_detector.visualization.interactive.panels.qt.slice import (
        _linear_likelihood_row,
    )

    finite_row = np.array([0.0, -1.0, -np.inf])
    np.testing.assert_allclose(
        _linear_likelihood_row(finite_row), [1.0, np.exp(-1.0), 0.0]
    )

    nan_row = np.array([np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(_linear_likelihood_row(nan_row), [0.0, 0.0, 0.0])

    assert _linear_likelihood_row(None) is None
    np.testing.assert_array_equal(
        _linear_likelihood_row(np.array([])), np.array([], dtype=np.float64)
    )


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
def test_qt_viewer_close_stops_autoscroll_timer(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Regression: closing the window mid-playback must stop the
    autoscroll timer before the widget enters its destruction path,
    otherwise the next ``_autoscroll_tick`` fires against a partially
    deleted Qt widget and segfaults.
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._start_autoscroll()
    assert viewer._autoscroll_timer is not None
    viewer._play_button.setChecked(True)
    qapp.processEvents()
    assert viewer._play_button.isChecked()

    viewer.close()
    qapp.processEvents()

    assert viewer._autoscroll_timer is None
    assert viewer._play_button.isChecked() is False


@pytest.mark.unit
def test_qt_viewer_help_text_covers_all_shortcuts(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``?`` opens a help dialog whose text covers every registered key."""
    from PySide6 import QtGui

    from non_local_detector.visualization.interactive.viewer.qt import (
        _SHORTCUT_TABLE,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    text = viewer._build_help_text()
    assert text  # non-empty
    for _, key, _, _ in _SHORTCUT_TABLE:
        assert key in text, f"shortcut {key!r} missing from help text"

    help_key = QtGui.QKeySequence("?").toString()
    bound = [
        s
        for s in viewer.findChildren(QtGui.QShortcut)
        if s.key().toString() == help_key
    ]
    assert bound, "? shortcut not registered"


@pytest.mark.unit
def test_qt_viewer_controls_bar_widgets_have_tooltips(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Every interactive controls-bar widget (button / combo / slider)
    has a non-empty tooltip — labels and frame separators are exempt."""
    from PySide6 import QtWidgets

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    bar = viewer._controls_bar
    interactive: list[QtWidgets.QWidget] = []
    for cls in (
        QtWidgets.QAbstractButton,
        QtWidgets.QComboBox,
        QtWidgets.QAbstractSlider,
    ):
        # ``findChildren`` is recursive; QComboBox popups contain
        # internal QScrollBar instances that aren't user-facing.
        for w in bar.findChildren(cls):
            if isinstance(w, QtWidgets.QScrollBar):
                continue
            interactive.append(w)

    assert interactive, "controls bar produced no interactive widgets"
    missing = [w for w in interactive if not w.toolTip()]
    assert not missing, (
        f"controls-bar widgets without tooltip: {[type(w).__name__ for w in missing]}"
    )


@pytest.mark.unit
def test_slice_overlay_combo_disables_unavailable_modes(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """When the active run lacks ``predictive_posterior``, the
    Predictive and Filtered combo items are disabled (Smoothed always
    enabled). When it has ``predictive_posterior`` but not
    ``log_likelihood``, only Filtered is disabled."""
    from dataclasses import replace

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    # No predictive, no log_likelihood → only Smoothed is available.
    bundle_default = run_bundles["nl_default"]
    ds = InMemoryDecoderDataSource.from_single(bundle_default)
    viewer = QtViewer(ds, t_width=0.5)
    combo = viewer._slice_overlay_combo
    model = combo.model()
    enabled_by_mode = {
        combo.itemData(i): model.item(i).isEnabled() for i in range(combo.count())
    }
    assert enabled_by_mode == {
        "predictive": False,
        "filtered": False,
        "smoothed": True,
    }
    viewer.close()

    # Predictive present, log_likelihood missing → Filtered disabled.
    full = run_bundles["nl_all"]
    bundle_no_loglik = replace(full, results=full.results.drop_vars("log_likelihood"))
    ds2 = InMemoryDecoderDataSource.from_single(bundle_no_loglik)
    viewer2 = QtViewer(ds2, t_width=0.5)
    combo2 = viewer2._slice_overlay_combo
    model2 = combo2.model()
    enabled_by_mode2 = {
        combo2.itemData(i): model2.item(i).isEnabled() for i in range(combo2.count())
    }
    assert enabled_by_mode2 == {
        "predictive": True,
        "filtered": False,
        "smoothed": True,
    }


@pytest.mark.unit
def test_slice_overlay_combo_disabled_items_carry_rebuild_tooltip(
    qapp,
    run_bundles: dict[str, RunBundle],
) -> None:
    """Disabled items expose the rebuild instruction via Qt.ToolTipRole
    so a hovering user gets actionable guidance, not silence."""
    from PySide6 import QtCore

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    bundle_default = run_bundles["nl_default"]
    ds = InMemoryDecoderDataSource.from_single(bundle_default)
    viewer = QtViewer(ds, t_width=0.5)
    combo = viewer._slice_overlay_combo
    model = combo.model()
    for i in range(combo.count()):
        if combo.itemData(i) == "predictive":
            tip = model.item(i).data(QtCore.Qt.ToolTipRole)
            assert tip and "predictive_posterior" in tip
        elif combo.itemData(i) == "filtered":
            tip = model.item(i).data(QtCore.Qt.ToolTipRole)
            assert tip and "predictive_posterior" in tip
            assert "log_likelihood" in tip


@pytest.mark.unit
def test_slice_overlay_falls_back_when_active_mode_unavailable(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    run_bundles: dict[str, RunBundle],
) -> None:
    """Swapping to a run that lacks the active overlay mode falls back
    to ``smoothed`` (always present) and refreshes the slice."""
    from dataclasses import replace

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    nl = run_bundles["nl_all"]
    bundles = {
        "nl": nl,
        "default": replace(nl, results=nl.results.drop_vars("predictive_posterior")),
    }
    ds = InMemoryDecoderDataSource(bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._slice_overlay_combo.setCurrentIndex(
        next(
            i
            for i in range(viewer._slice_overlay_combo.count())
            if viewer._slice_overlay_combo.itemData(i) == "predictive"
        )
    )
    qapp.processEvents()
    assert viewer._slice_panel.overlay_mode == "predictive"

    # Swap to the run missing predictive_posterior.
    viewer._core.set_active_run("default")
    qapp.processEvents()
    assert viewer._slice_panel.overlay_mode == "smoothed"
    assert viewer._slice_overlay_combo.currentData() == "smoothed"


@pytest.mark.unit
def test_qt_viewer_go_to_time_recenters_core(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    monkeypatch,
) -> None:
    """``g`` opens an input dialog whose accepted value drives
    ``core.set_t_center``."""
    from PySide6 import QtGui, QtWidgets

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    # Patch the dialog so the test doesn't block. Return (5.0, True).
    target_t = float(viewer._data_source.time[len(viewer._data_source.time) // 2])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        staticmethod(lambda *args, **kwargs: (target_t, True)),
    )

    viewer._show_go_to_time_dialog()
    assert viewer._core.t_center == target_t

    # ``g`` is registered as a shortcut.
    g_key = QtGui.QKeySequence("G").toString()
    matches = [
        s for s in viewer.findChildren(QtGui.QShortcut) if s.key().toString() == g_key
    ]
    assert matches, "g shortcut not registered"


@pytest.mark.unit
def test_qt_viewer_shortcut_handlers_match_table(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Every action string in ``_SHORTCUT_TABLE`` has a handler, and
    every handler maps to a row in the table — guards against silent
    drift on future table edits."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        _SHORTCUT_TABLE,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    actions_in_table = {row[2] for row in _SHORTCUT_TABLE}
    handlers_present = set(viewer._shortcut_handlers.keys())
    assert actions_in_table == handlers_present


@pytest.mark.unit
def test_scroll_keeps_x_view_range_fixed_and_absolute_readout_moves(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Phase 3.1 invariance: scrolling the t_center forward leaves the
    visible x-range and tick labels of every relative-time-axis panel
    unchanged, while the absolute-time readout in the controls bar
    tracks ``t_center``. Catches regressions that re-introduce
    absolute-axis churn."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.4)

    panels = viewer._builtin_panels
    initial_ranges = [panel.getViewBox().viewRange()[0] for panel in panels]
    initial_label = viewer._time_label.text()

    # Pick a different time bin so t_center actually changes.
    target_t = float(ds.time[len(ds.time) // 4])
    assert target_t != viewer._core.t_center, "fixture should give us motion"
    viewer._core.set_t_center(target_t)
    viewer._sync_control_labels()
    qapp.processEvents()

    after_ranges = [panel.getViewBox().viewRange()[0] for panel in panels]
    after_label = viewer._time_label.text()
    for before, after in zip(initial_ranges, after_ranges, strict=True):
        np.testing.assert_allclose(after, before, atol=1e-9)
    # Visible x-range should be exactly [-t_width/2, +t_width/2].
    for low, high in after_ranges:
        np.testing.assert_allclose([low, high], [-0.2, 0.2], atol=1e-9)
    # Absolute-time readout must have moved (it shows t_center).
    assert initial_label != after_label


@pytest.mark.unit
def test_t_width_change_relocks_x_view_range(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Changing ``t_width`` re-locks every panel's x-range to the new
    ``[-t_width/2, +t_width/2]`` bounds — the only event that should
    update the visible x-range under relative rendering."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.4)
    viewer._core.set_t_width(1.0)
    qapp.processEvents()
    for panel in viewer._builtin_panels:
        low, high = panel.getViewBox().viewRange()[0]
        np.testing.assert_allclose([low, high], [-0.5, 0.5], atol=1e-9)


@pytest.mark.unit
def test_wheel_resize_integrates_small_deltas(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """30 small touchpad events of ``delta=4`` produce approximately
    the same final ``t_width`` as one mouse-wheel detent of
    ``delta=120`` (within 1%) — proves
    ``factor = exp(-delta * RESIZE_GAIN)`` integrates additively."""
    from PySide6 import QtCore, QtGui

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    def _wheel_event(delta: int) -> QtGui.QWheelEvent:
        return QtGui.QWheelEvent(
            QtCore.QPointF(0, 0),
            QtCore.QPointF(0, 0),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, delta),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )

    ds = InMemoryDecoderDataSource(multi_run_bundles)

    # Path A: one big detent.
    viewer_a = QtViewer(ds, t_width=1.0)
    viewer_a.eventFilter(viewer_a, _wheel_event(120))
    final_a = viewer_a._core.t_width

    ds_b = InMemoryDecoderDataSource(multi_run_bundles)
    viewer_b = QtViewer(ds_b, t_width=1.0)
    for _ in range(30):
        viewer_b.eventFilter(viewer_b, _wheel_event(4))
    final_b = viewer_b._core.t_width

    np.testing.assert_allclose(final_b, final_a, rtol=0.01)


@pytest.mark.unit
def test_window_payload_carries_view_state(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``build_payload`` propagates the requested ``t_center`` and
    ``t_width`` so panels can render at relative coordinates against
    a fixed ``[-t_width/2, +t_width/2]`` x-range without re-deriving
    the view from ``time_start`` / ``time_stop``."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.4)
    target_t = float(ds.time[len(ds.time) // 2])
    viewer._core.set_t_center(target_t)

    payload = viewer._backend.build_payload(viewer._core.current_view_state)
    assert payload.t_center == target_t
    assert payload.t_width == 0.4


@pytest.mark.unit
def test_launch_qt_per_component_kwargs_constructs_run(
    qapp,
    nl_fitted,
    sim_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``launch_qt(detector=..., results=..., ...)`` builds the bundle
    internally so notebook callers don't need to import ``RunBundle``
    (Phase 6.1)."""
    from non_local_detector.visualization.interactive.viewer import qt as qt_mod
    from non_local_detector.visualization.interactive.viewer.qt import launch_qt

    seen: dict[str, object] = {}
    original = qt_mod.launch_qt_with_source

    def _capture(data_source, **kwargs):
        seen["data_source"] = data_source
        kwargs["block"] = False
        return original(data_source, **kwargs)

    monkeypatch.setattr(qt_mod, "launch_qt_with_source", _capture)

    code = launch_qt(
        detector=nl_fitted.detector,
        results=nl_fitted.results,
        spike_times=sim_session.spike_times,
        position=sim_session.position,
        position_time=sim_session.time,
        speed=sim_session.speed,
    )
    assert code == 0
    ds = seen["data_source"]
    assert ds.run_names == ["default"]


@pytest.mark.unit
def test_launch_qt_rejects_mixed_bundle_and_per_component(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    nl_fitted,
) -> None:
    """Mixing the bundle form with per-component kwargs raises
    ``ValueError`` naming the conflict — public-API silent-drop is a
    footgun (Phase 6.1 dispatch contract)."""
    from non_local_detector.visualization.interactive.viewer.qt import launch_qt

    with pytest.raises(ValueError, match="cannot mix"):
        launch_qt(
            multi_run_bundles,
            detector=nl_fitted.detector,
        )


@pytest.mark.unit
def test_launch_qt_rejects_per_component_missing_required(
    qapp,
    nl_fitted,
) -> None:
    """Per-component form without all required kwargs raises
    ``ValueError`` listing the missing arg(s)."""
    from non_local_detector.visualization.interactive.viewer.qt import launch_qt

    with pytest.raises(ValueError, match="missing"):
        launch_qt(
            detector=nl_fitted.detector,
            results=nl_fitted.results,
            # spike_times, position, position_time omitted
        )


@pytest.mark.unit
def test_launch_qt_bundle_form_still_works(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: the original bundle form still constructs a viewer."""
    from non_local_detector.visualization.interactive.viewer import qt as qt_mod
    from non_local_detector.visualization.interactive.viewer.qt import launch_qt

    original = qt_mod.launch_qt_with_source

    def _no_block(data_source, **kwargs):
        kwargs["block"] = False
        return original(data_source, **kwargs)

    monkeypatch.setattr(qt_mod, "launch_qt_with_source", _no_block)
    code = launch_qt(multi_run_bundles)
    assert code == 0


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
    # Missing-data message lives in a wrapping QLabel overlay so the
    # text isn't truncated by the panel-width-bound pyqtgraph title.
    # ``isVisible()`` returns False in offscreen Qt when the panel
    # isn't attached to a shown top-level — use the explicit
    # ``isHidden()`` flag instead (False ⇒ setVisible(True) was called).
    assert panel._missing_label.text() == panel.MISSING_DATA_MESSAGE
    assert not panel._missing_label.isHidden()
    assert "log_likelihood" in panel.MISSING_DATA_MESSAGE
    assert "predict" in panel.MISSING_DATA_MESSAGE
    # Title bar must not carry the missing-data text — that was the
    # bug the overlay refactor fixes.
    title_label = panel.plotItem.titleLabel
    assert title_label.text == ""

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
    assert panel._missing_label.isHidden()
    assert panel._missing_label.text() == ""


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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)

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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)
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

    payload = viewer._backend.build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)
    assert viewer._slice_panel._buffered_payload is payload


@pytest.mark.unit
def test_qt_viewer_raster_spike_click_toggles_spike_pin(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Raster spike clicks pin by stable event-id identity."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    first = ds.spike_event_at(0)
    same_cell_events = [
        event_id
        for event_id in range(ds.event_index.n_events)
        if ds.spike_event_at(event_id).cell_id == first.cell_id and event_id != 0
    ]
    second_event_id = same_cell_events[0] if same_cell_events else 1
    second = ds.spike_event_at(second_event_id)

    assert viewer._slice_panel.pinned_cell_ids == frozenset()
    viewer._raster_panel.event_clicked.emit(first.event_id)
    assert first.cell_id in viewer._slice_panel.pinned_cell_ids
    assert viewer._pinned_event_id == first.event_id
    assert viewer._pinned_time == first.time

    # Same event → unpin.
    viewer._raster_panel.event_clicked.emit(first.event_id)
    assert first.cell_id not in viewer._slice_panel.pinned_cell_ids
    assert viewer._pinned_event_id is None

    # Another spike from the same cell repins/recenters to that event.
    viewer._raster_panel.event_clicked.emit(second.event_id)
    assert viewer._slice_panel.pinned_cell_ids == frozenset({second.cell_id})
    assert viewer._pinned_event_id == second.event_id
    assert viewer.core.t_center == pytest.approx(second.time)
    # Phase 3.1d invariant: pin marker x lives in the panel's relative
    # frame; after recentering on ``second.time`` the pin should land
    # at relative 0 (i.e. inside the visible window), NOT off-screen
    # at the old ``second.time - first.time`` offset.
    pin_x = viewer._raster_panel._pin_line.value()
    assert abs(pin_x) < viewer.core.t_width, (
        f"pin marker x={pin_x} lives outside the visible window "
        f"[-t_width/2, +t_width/2]; refresh ordering regressed"
    )


@pytest.mark.unit
def test_qt_viewer_raster_spike_click_updates_slice_to_spike_bin(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Clicking a visible spike renders that spike cell with a nonzero count."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    event = ds.spike_event_at(0)
    cell_id = event.cell_id
    spike_t = event.time
    t_idx = event.time_index

    viewer.core.set_t_center(spike_t)
    viewer._on_window_loaded(
        viewer._backend.build_payload(viewer.core.current_view_state)
    )
    viewer._on_event_clicked(event.event_id)

    assert viewer._slider.value() == t_idx
    visible_rows = [
        r for r in viewer._slice_panel._per_cell_rows if not r.container.isHidden()
    ]
    pinned_label = next(
        r.label.text() for r in visible_rows if f"#{cell_id}" in r.label.text()
    )
    event_ids = viewer._slice_model.event_ids_at_bin(t_idx)
    expected_count = int(
        np.count_nonzero(ds.event_index.cell_ids[event_ids] == cell_id)
    )
    assert f"(×{expected_count})" in pinned_label
    assert "(×0)" not in pinned_label


@pytest.mark.unit
def test_qt_viewer_manual_navigation_clears_pins(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    monkeypatch,
) -> None:
    """Manual scrub/step/reset/go-to-time interactions clear stale spike pins."""
    from PySide6 import QtWidgets

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    event = ds.spike_event_at(0)
    viewer._on_event_clicked(event.event_id)
    assert viewer._slice_panel.pinned_cell_ids == frozenset({event.cell_id})
    viewer._on_slider_value_changed(viewer._slider.value() + 1)
    assert viewer._slice_panel.pinned_cell_ids == frozenset()
    assert viewer._pinned_event_id is None

    viewer._on_event_clicked(event.event_id)
    viewer._set_t_center_from_panel_click(1.75)
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    viewer._on_event_clicked(event.event_id)
    viewer._step_right()
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    viewer._on_event_clicked(event.event_id)
    viewer._step_window(+1)
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    viewer._on_event_clicked(event.event_id)
    viewer._reset_view()
    assert viewer._slice_panel.pinned_cell_ids == frozenset()

    # Go-to-time recenters via dialog; must clear pins like other manual nav.
    viewer._on_event_clicked(event.event_id)
    target_t = float(ds.time[len(ds.time) // 2])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        staticmethod(lambda *args, **kwargs: (target_t, True)),
    )
    viewer._show_go_to_time_dialog()
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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)
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
    assert _RIGHT_COLUMN_SLICE_STRETCH == 2
    assert _RIGHT_COLUMN_TRAILING_STRETCH == 4


@pytest.mark.unit
def test_qt_viewer_slice_overlay_combo_drives_slice_panel(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """The slice overlay mode is controlled from the bottom controls bar."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    assert not hasattr(viewer._slice_panel, "_overlay_combo")
    combo = viewer._slice_overlay_combo
    filtered_idx = next(
        i for i in range(combo.count()) if combo.itemData(i) == "filtered"
    )
    combo.setCurrentIndex(filtered_idx)

    assert viewer._slice_panel.overlay_mode == "filtered"


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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)

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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)

    trace = viewer._panel._position_trace
    x, y = trace.getData()
    assert x is not None and y is not None
    assert x.size == payload.time.size
    # Heatmap renders at relative coordinates (Phase 3.1).
    np.testing.assert_array_equal(x, payload.time - payload.t_center)
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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)
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
    payload_before = viewer._backend.build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload_before)

    next_run = next(name for name in ds.run_names if name != ds.active_run_name)
    viewer._core.set_active_run(next_run)
    payload_after = viewer._backend.build_payload(viewer.core.current_view_state)
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
def test_backend_uses_long_debounce_for_resize_burst(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """A wheel- or slider-driven resize burst sits on the long debounce.

    Center scrubbing fires per frame and benefits from the 16 ms
    debounce. Wheel-resize at 10 s windows would otherwise spawn a
    full per-pixel load that pins the CPU; the 100 ms trailing-edge
    debounce coalesces the entire drag into one final load.

    Drives the backend directly (not through ``ViewerCore``) so the
    test isn't entangled with ``QtViewer`` construction also scheduling
    initial loads — we want to exercise the per-call width-vs-last
    comparison in isolation.
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtBackendAdapter,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    backend = QtBackendAdapter(ds)
    t_center = float(ds.time[ds.n_time // 2])

    def _state(request_id: int, t_center: float, t_width: float) -> ViewState:
        return ViewState(request_id=request_id, t_center=t_center, t_width=t_width)

    def _no_op(_payload):
        return None

    # First-ever schedule: no prior width to compare against → scrub.
    backend.schedule_window_load(_state(0, t_center, 0.5), _no_op)
    assert backend._debounce_timer.interval() == backend.SCRUB_DEBOUNCE_MS
    assert not backend._in_resize_burst

    # Same-width center scrub stays on the short debounce.
    backend.schedule_window_load(_state(1, t_center + 0.1, 0.5), _no_op)
    assert backend._debounce_timer.interval() == backend.SCRUB_DEBOUNCE_MS
    assert not backend._in_resize_burst

    # Width change → resize burst → long debounce.
    backend.schedule_window_load(_state(2, t_center + 0.1, 0.6), _no_op)
    assert backend._in_resize_burst
    assert backend._debounce_timer.interval() == backend.RESIZE_DEBOUNCE_MS

    # Subsequent same-width center scrub mid-drag does NOT collapse
    # the long debounce — the burst flag persists until a flush.
    backend.schedule_window_load(_state(3, t_center + 0.2, 0.6), _no_op)
    assert backend._in_resize_burst
    assert backend._debounce_timer.interval() == backend.RESIZE_DEBOUNCE_MS

    # The drain path is decoupled from this test (it depends on the
    # executor and the Qt event loop). Pin only the state-transition
    # contract: ``_flush_pending`` must clear the burst flag so a
    # subsequent same-width tick can re-enter the short-debounce
    # path. The next live ``schedule_window_load`` after the executor
    # finishes will see ``_in_resize_burst=False`` + matching
    # ``_last_scheduled_t_width`` and pick ``SCRUB_DEBOUNCE_MS``.
    backend._pending_state = None
    backend._inflight = False
    backend._in_resize_burst = False

    backend.schedule_window_load(_state(4, t_center + 0.25, 0.6), _no_op)
    assert not backend._in_resize_burst
    assert backend._debounce_timer.interval() == backend.SCRUB_DEBOUNCE_MS


@pytest.mark.unit
def test_overlay_change_to_predictive_triggers_window_reload(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Switching from ``smoothed`` to ``predictive`` actually commits a load.

    The earlier (incorrect) fix called ``request_load`` which reuses
    the existing ``request_id``; once the initial window had committed
    the new payload was silently dropped by
    ``_handle_load_result``'s ``request_id <= latest_committed`` rule.
    This test pins the commit-side contract: after an overlay toggle
    the buffer must actually update with the new array (here,
    ``predictive`` becomes non-None).
    """
    from PySide6 import QtCore

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)

    # Drive an initial smoothed-mode load to completion so the
    # ``latest_committed_request_id`` reflects a real commit
    # (``QtViewer.__init__`` builds the core but doesn't fire a load).
    # Without this seed the ``request_load`` vs ``refresh`` distinction
    # wouldn't bite — the bug only fires *after* a prior commit.
    viewer._core.request_load()
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while viewer._core._latest_committed_request_id < 0 and deadline.elapsed() < 2000:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)
    assert viewer._core._latest_committed_request_id >= 0, (
        "initial load never committed within 2s — Qt event-loop seed failed"
    )
    initial_committed = viewer._core._latest_committed_request_id

    # Default overlay = ``"smoothed"`` — predictive was excluded.
    received: list[WindowPayload] = []
    viewer._core.on_window_loaded(received.append)

    overlay_index = next(
        i
        for i in range(viewer._slice_overlay_combo.count())
        if viewer._slice_overlay_combo.itemData(i) == "predictive"
    )
    viewer._slice_overlay_combo.setCurrentIndex(overlay_index)

    # Drain the event loop until the new load commits.
    deadline.start()
    while (
        viewer._core._latest_committed_request_id == initial_committed
        and deadline.elapsed() < 1500
    ):
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)
    assert viewer._core._latest_committed_request_id > initial_committed, (
        "overlay change must commit a fresh payload, not be dropped by "
        "the stale-result rule"
    )
    assert received, "on_window_loaded must fire for the refreshed payload"
    assert received[-1].predictive is not None


@pytest.mark.unit
def test_reset_view_round_trips_clamped_initial_t_width(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``R``-reset stores the *clamped* width, not the raw constructor arg.

    Pin: launching with ``--t-width 0`` (or any sub-floor value) must
    not blow up later when the user presses ``R``. Before the fix
    ``QtViewer`` stored the raw ``t_width`` value and ``_reset_view``
    fed it back into ``set_t_width``, which raises on non-positive
    values.
    """
    from non_local_detector.visualization.interactive.viewer.core import (
        MIN_T_WIDTH_SECONDS,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.0)
    # Constructor must clamp, not raw-store.
    assert viewer._initial_t_width == pytest.approx(MIN_T_WIDTH_SECONDS)
    assert viewer._core.t_width == pytest.approx(MIN_T_WIDTH_SECONDS)

    # Move the user away from the initial state, then reset — the
    # reset path must not raise.
    viewer._core.set_t_width(1.0)
    viewer._reset_view()
    assert viewer._core.t_width == pytest.approx(MIN_T_WIDTH_SECONDS)


@pytest.mark.unit
def test_backend_skips_predictive_when_slice_is_smoothed(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """The smoothed overlay doesn't read predictive — backend skips the load.

    At 10 s windows ``predictive_posterior`` is a
    ``(n_visible, n_state_bins)`` array that no panel renders when the
    slice is in ``"smoothed"`` mode. The QtViewer narrows the
    backend's required-outputs set on slice-overlay change so the
    worker doesn't materialize it.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    backend = viewer._backend

    # Default overlay mode is "smoothed" → predictive must be excluded.
    assert "predictive" not in backend._required_outputs

    state = ViewState(request_id=0, t_center=ds.time[ds.n_time // 2], t_width=0.5)
    payload = backend.build_payload(state)
    assert payload.predictive is None
    assert payload.posterior is not None  # heatmap still loaded
    assert payload.likelihood is not None

    # Switch to "predictive" → predictive must be loaded again.
    viewer._slice_panel.set_overlay_mode("predictive")
    viewer._sync_required_outputs_with_panels()
    assert "predictive" in backend._required_outputs
    payload = backend.build_payload(state)
    assert payload.predictive is not None


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
    realbuild_payload = backend.build_payload

    def _slowbuild_payload(state):
        # Block the worker until the test releases, simulating a
        # long-running window read.
        release.wait(timeout=2.0)
        return realbuild_payload(state)

    backend.build_payload = _slowbuild_payload  # type: ignore[assignment]

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
    """A ``build_payload`` that raises must not freeze the backend.

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

    # Phase 1: drive the first load through a raising build_payload.
    realbuild_payload = backend.build_payload
    raise_now = {"value": True}

    def _flakybuild_payload(state):
        if raise_now["value"]:
            raise RuntimeError("simulated worker failure")
        return realbuild_payload(state)

    backend.build_payload = _flakybuild_payload  # type: ignore[assignment]

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
    payload = viewer._backend.build_payload(viewer.core.current_view_state)
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
