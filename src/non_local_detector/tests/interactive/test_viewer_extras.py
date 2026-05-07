"""Tests for QtViewer's extra-panels + auto-construction + keyboard extras.

Marked ``@pytest.mark.gui`` because they construct an actual
``QtViewer``; ``[viewer]`` extra is required.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Force offscreen Qt platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from non_local_detector.tests._simulated_detectors import FittedDetector
from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.view_models.base import RunBundle
from non_local_detector.visualization.interactive.view_models.series import MetricSpec

pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    from non_local_detector.visualization.interactive.viewer.qt import (
        _ensure_qapplication,
    )

    app = _ensure_qapplication()
    yield app


@pytest.mark.unit
def test_qt_viewer_accepts_explicit_extra_panels(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``extra_panels=[...]`` are added below the posterior heatmap."""
    from non_local_detector.visualization.interactive.panels.qt.series import (
        LineSeriesPanel,
    )
    from non_local_detector.visualization.interactive.view_models.series import (
        LineSeriesModel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    extra = LineSeriesPanel(
        LineSeriesModel(name="dummy", t=np.arange(10.0), y=np.arange(10.0))
    )
    viewer = QtViewer(ds, t_width=0.5, extra_panels=[extra])
    assert viewer._extra_panels == [extra]
    assert extra in viewer._all_panels


@pytest.mark.unit
def test_qt_viewer_auto_builds_panels_from_extra_metrics(
    qapp,
    nl_fitted: FittedDetector,
    sim_session,
) -> None:
    """When ``extra_panels`` is unset, ``bundle.extra_metrics`` auto-renders.

    Three sized affordances test:
    - pd.Series → LineSeriesPanel.
    - MetricSpec.scatter → ScatterSeriesPanel.
    - MetricSpec.intervals → IntervalSeriesPanel.
    """
    from non_local_detector.visualization.interactive.panels.qt.series import (
        IntervalSeriesPanel,
        LineSeriesPanel,
        ScatterSeriesPanel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    bundle = RunBundle(
        results=nl_fitted.results,
        detector=nl_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
        extra_metrics={
            "replay": pd.Series(
                np.linspace(0, 1, 100),
                index=np.linspace(0.0, 10.0, 100),
            ),
            "spike_quality": MetricSpec.scatter(
                name="spike_quality",
                t=np.array([1.0, 2.0]),
                y=np.array([0.5, 0.9]),
            ),
            "events": MetricSpec.intervals(
                name="events",
                t_start=np.array([3.0]),
                t_end=np.array([4.0]),
            ),
        },
    )
    ds = InMemoryDecoderDataSource.from_single(bundle)
    viewer = QtViewer(ds, t_width=0.5)
    types = {type(p) for p in viewer._extra_panels}
    assert LineSeriesPanel in types
    assert ScatterSeriesPanel in types
    assert IntervalSeriesPanel in types


@pytest.mark.unit
def test_qt_viewer_swap_rebuilds_auto_extras(
    qapp,
    nl_fitted: FittedDetector,
    cf_fitted: FittedDetector,
    sim_session,
) -> None:
    """Auto-built extras rebuild against the new run's ``extra_metrics`` on swap."""
    from non_local_detector.visualization.interactive.panels.qt.series import (
        IntervalSeriesPanel,
        LineSeriesPanel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    # NL bundle ships a single line metric; CF bundle ships a different
    # set (one line + one intervals panel). After swap the panel type
    # set must reflect CF, not NL.
    nl_bundle = RunBundle(
        results=nl_fitted.results,
        detector=nl_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
        extra_metrics={
            "nl_only_metric": pd.Series(
                np.linspace(0.0, 1.0, 50),
                index=np.linspace(0.0, 5.0, 50),
            ),
        },
    )
    cf_bundle = RunBundle(
        results=cf_fitted.results,
        detector=cf_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
        extra_metrics={
            "cf_line": pd.Series(
                np.linspace(0.0, 0.5, 50),
                index=np.linspace(0.0, 5.0, 50),
            ),
            "cf_events": MetricSpec.intervals(
                name="cf_events",
                t_start=np.array([1.0, 2.0]),
                t_end=np.array([1.5, 2.5]),
            ),
        },
    )
    ds = InMemoryDecoderDataSource({"nl": nl_bundle, "cf": cf_bundle})
    viewer = QtViewer(ds, t_width=0.5)

    # Initial: NL extras → one LineSeriesPanel.
    assert [type(p) for p in viewer._extra_panels] == [LineSeriesPanel]
    initial_extras = list(viewer._extra_panels)

    viewer.core.set_active_run("cf")

    new_types = [type(p) for p in viewer._extra_panels]
    assert LineSeriesPanel in new_types
    assert IntervalSeriesPanel in new_types
    for old in initial_extras:
        assert old not in viewer._extra_panels
    for new_panel in viewer._extra_panels:
        assert new_panel in viewer._all_panels

    # Old extras' overlay callbacks must be unregistered from
    # ViewerCore — otherwise dispatch hits deleted Qt widgets.
    callbacks = viewer.core._on_overlays_changed_callbacks
    assert len(callbacks) == len(viewer._all_panels)
    callback_owners = [getattr(cb, "__self__", None) for cb in callbacks]
    for old in initial_extras:
        assert old not in callback_owners

    # Smoke check: dispatch after swap must not raise on deleted widgets.
    viewer.core.refresh_overlays()


@pytest.mark.unit
def test_qt_viewer_swap_preserves_user_supplied_extras(
    qapp,
    nl_fitted: FittedDetector,
    cf_fitted: FittedDetector,
    sim_session,
) -> None:
    """User-supplied ``extra_panels`` are not torn down on swap (caller-owned)."""
    from non_local_detector.visualization.interactive.panels.qt.series import (
        LineSeriesPanel,
    )
    from non_local_detector.visualization.interactive.view_models.series import (
        LineSeriesModel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    nl_bundle = RunBundle(
        results=nl_fitted.results,
        detector=nl_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
    )
    cf_bundle = RunBundle(
        results=cf_fitted.results,
        detector=cf_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
    )
    user_panel = LineSeriesPanel(
        LineSeriesModel(name="user", t=np.arange(10.0), y=np.arange(10.0))
    )
    ds = InMemoryDecoderDataSource({"nl": nl_bundle, "cf": cf_bundle})
    viewer = QtViewer(ds, t_width=0.5, extra_panels=[user_panel])

    assert viewer._extra_panels == [user_panel]
    viewer.core.set_active_run("cf")
    # Same user-supplied panel object — not torn down + rebuilt.
    assert viewer._extra_panels == [user_panel]


@pytest.mark.unit
def test_model_dropdown_visible_only_when_multi_run(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
    nl_fitted: FittedDetector,
    sim_session,
) -> None:
    """Model dropdown only appears when there are multiple runs."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    multi_ds = InMemoryDecoderDataSource(multi_run_bundles)
    multi_viewer = QtViewer(multi_ds, t_width=0.5)
    assert multi_viewer._model_combo is not None
    assert [
        multi_viewer._model_combo.itemText(i)
        for i in range(multi_viewer._model_combo.count())
    ] == ["nl", "cf", "nsf", "dec"]
    assert multi_viewer._model_combo.currentText() == "nl"

    single_bundle = RunBundle(
        results=nl_fitted.results,
        detector=nl_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
    )
    single_ds = InMemoryDecoderDataSource.from_single(single_bundle)
    single_viewer = QtViewer(single_ds, t_width=0.5)
    assert single_viewer._model_combo is None


@pytest.mark.unit
def test_model_combo_drives_set_active_run(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """User selecting a run in the dropdown swaps the active run."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer.core.active_run_name == "nl"

    # Pick the second run via the combo.
    cf_idx = viewer._model_combo.findText("cf")
    viewer._model_combo.setCurrentIndex(cf_idx)
    assert viewer.core.active_run_name == "cf"


@pytest.mark.unit
def test_m_key_cycles_through_runs(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``M`` cycles through the run list in dropdown order, wrapping at the end."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    sequence = []
    for _ in range(5):
        sequence.append(viewer.core.active_run_name)
        viewer._cycle_model()
    # nl → cf → nsf → dec → nl (wrap) → cf
    assert sequence == ["nl", "cf", "nsf", "dec", "nl"]


@pytest.mark.unit
def test_swap_via_core_syncs_model_combo(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Programmatic ``core.set_active_run`` updates the dropdown selection."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer._model_combo.currentText() == "nl"
    viewer.core.set_active_run("nsf")
    assert viewer._model_combo.currentText() == "nsf"


@pytest.mark.unit
def test_swap_preserves_view_state(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Swap preserves t_center, t_width, active_overlay_name."""
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    # Attach a shared overlay so set_active_overlay has something to bind to.
    original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
    try:
        shared = EventOverlay.points(name="swr", times=np.array([0.5, 1.5]))
        for bundle in multi_run_bundles.values():
            bundle.event_overlays.append(shared)
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        viewer = QtViewer(ds, t_width=0.7)

        # Manipulate view state.
        viewer.core.set_t_center(viewer.core.t_center + 0.123)
        viewer.core.set_t_width(0.42)
        viewer.core.set_active_overlay("swr")

        snap_t_center = viewer.core.t_center
        snap_t_width = viewer.core.t_width
        snap_overlay = viewer.core.active_overlay_name

        viewer.core.set_active_run("cf")
        assert viewer.core.t_center == pytest.approx(snap_t_center)
        assert viewer.core.t_width == pytest.approx(snap_t_width)
        assert viewer.core.active_overlay_name == snap_overlay
    finally:
        for n, b in multi_run_bundles.items():
            b.event_overlays[:] = original[n]


def _make_recording_bin_panel(*, with_rebind: bool = True):
    """Build a ``BinSyncedPanel`` test double that records calls.

    Defined as a factory because PySide6 is imported lazily; class
    definitions referencing ``QtWidgets.QWidget`` at module top
    level would fail when the ``[viewer]`` extra isn't installed.
    """
    from PySide6 import QtWidgets as _W

    class _RecordingBinPanel(_W.QWidget):
        def __init__(self) -> None:
            _W.QWidget.__init__(self)
            self.buffer_calls: list = []
            self.index_calls: list[int] = []
            self.rebind_calls: int = 0

        def set_window_buffer(self, payload) -> None:
            self.buffer_calls.append(payload)

        def update_for_index(self, t_idx: int) -> None:
            self.index_calls.append(t_idx)

        if with_rebind:

            def rebind_after_swap(self) -> None:
                self.rebind_calls += 1

    return _RecordingBinPanel()


@pytest.mark.unit
def test_extra_bin_panels_appear_under_slice_panel(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``extra_bin_panels`` are stacked below the built-in slice panel
    in the right column of the body layout."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    plugin = _make_recording_bin_panel()
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5, extra_bin_panels=[plugin])
    splitter = viewer._body_splitter
    right_column = splitter.widget(1)
    right_column_widgets = [
        right_column.layout().itemAt(i).widget()
        for i in range(right_column.layout().count())
        if right_column.layout().itemAt(i).widget() is not None
    ]
    # Slice panel sits at the top of the right column followed by
    # any bin plugins; the trailing item is a stretch (None widget).
    assert right_column_widgets[0] is viewer._slice_panel
    assert plugin in right_column_widgets


@pytest.mark.unit
def test_extra_bin_panels_get_window_buffer_on_load(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Each window load dispatches ``set_window_buffer`` to every bin plugin."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    plugin = _make_recording_bin_panel()
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5, extra_bin_panels=[plugin])
    payload = viewer._backend._build_payload(viewer.core.current_view_state)
    viewer._on_window_loaded(payload)
    assert plugin.buffer_calls == [payload]
    # Same load also drives an immediate update_for_index at the slider.
    assert plugin.index_calls == [viewer._slider.value()]


@pytest.mark.unit
def test_extra_bin_panels_get_update_for_index_on_slider_tick(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Slider ticks reach bin plugins synchronously off the main thread."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    plugin = _make_recording_bin_panel()
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5, extra_bin_panels=[plugin])
    plugin.index_calls.clear()
    target = viewer._slider.value() + 1
    viewer._on_slider_value_changed(target)
    assert plugin.index_calls == [target]


@pytest.mark.unit
def test_extra_bin_panels_rebind_after_swap_when_present(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Active-run swap calls ``rebind_after_swap`` when the plugin defines it."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    plugin = _make_recording_bin_panel()
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5, extra_bin_panels=[plugin])
    assert plugin.rebind_calls == 0
    viewer.core.set_active_run("cf")
    assert plugin.rebind_calls == 1


@pytest.mark.unit
def test_extra_bin_panels_swap_without_rebind_method_does_not_crash(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Plugins that omit ``rebind_after_swap`` are tolerated (optional hook)."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    plugin = _make_recording_bin_panel(with_rebind=False)
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5, extra_bin_panels=[plugin])
    viewer.core.set_active_run("cf")  # must not raise


@pytest.mark.unit
def test_extra_panels_remains_time_axis_only(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``extra_panels`` is left-column only — bin-only plugins must use
    ``extra_bin_panels`` instead.

    Passing a ``BinSyncedPanel``-shaped widget through ``extra_panels``
    fails at viewer-construction time inside ``_wire_panels`` because
    the time-axis path requires ``set_event_overlays``. That early
    failure is preferable to silently rendering nothing.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    bin_only_plugin = _make_recording_bin_panel()
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    with pytest.raises(AttributeError, match="set_event_overlays"):
        QtViewer(ds, t_width=0.5, extra_panels=[bin_only_plugin])


@pytest.mark.unit
def test_play_button_default_state_is_paused(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert not viewer._play_button.isChecked()
    assert viewer._play_button.text() == "▶"
    assert viewer._autoscroll_timer is None


@pytest.mark.unit
def test_speed_combo_default_is_005x(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Default playback speed matches the upstream ``AUTOSCROLL_DEFAULT_SPEED``."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        AUTOSCROLL_DEFAULT_SPEED,
        AUTOSCROLL_SPEED_OPTIONS,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer._autoscroll_rate == AUTOSCROLL_DEFAULT_SPEED
    assert viewer._speed_combo.itemData(viewer._speed_combo.currentIndex()) == (
        AUTOSCROLL_DEFAULT_SPEED
    )
    items = [
        viewer._speed_combo.itemData(i) for i in range(viewer._speed_combo.count())
    ]
    assert tuple(items) == AUTOSCROLL_SPEED_OPTIONS


@pytest.mark.unit
def test_toggle_play_starts_and_stops_timer(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Space (and clicking the play button) starts/stops the autoscroll timer."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._toggle_play()
    assert viewer._play_button.isChecked()
    assert viewer._play_button.text() == "⏸"
    assert viewer._autoscroll_timer is not None
    viewer._toggle_play()
    assert not viewer._play_button.isChecked()
    assert viewer._play_button.text() == "▶"
    assert viewer._autoscroll_timer is None


@pytest.mark.unit
def test_speed_combo_change_updates_rate(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    target_idx = viewer._speed_combo.findData(1.0)
    viewer._speed_combo.setCurrentIndex(target_idx)
    assert viewer._autoscroll_rate == 1.0


@pytest.mark.unit
def test_step_speed_advances_through_preset_list(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``,`` / ``.`` step through the preset list, clamped at the ends."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        AUTOSCROLL_SPEED_OPTIONS,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    initial = viewer._speed_combo.currentIndex()
    viewer._step_speed(+1)
    assert viewer._speed_combo.currentIndex() == initial + 1
    viewer._step_speed(-1)
    assert viewer._speed_combo.currentIndex() == initial
    # Clamp at lower bound.
    for _ in range(len(AUTOSCROLL_SPEED_OPTIONS)):
        viewer._step_speed(-1)
    assert viewer._speed_combo.currentIndex() == 0
    # Clamp at upper bound.
    for _ in range(len(AUTOSCROLL_SPEED_OPTIONS) * 2):
        viewer._step_speed(+1)
    assert viewer._speed_combo.currentIndex() == len(AUTOSCROLL_SPEED_OPTIONS) - 1


@pytest.mark.unit
def test_autoscroll_tick_advances_slider(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """One tick at a non-zero speed must move the slider forward."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    # Crank speed up so a single tick crosses many bins.
    high_idx = viewer._speed_combo.findData(8.0)
    viewer._speed_combo.setCurrentIndex(high_idx)
    viewer._toggle_play()  # initialise the float playback cursor
    initial = viewer._slider.value()
    viewer._autoscroll_tick()
    assert viewer._slider.value() > initial


@pytest.mark.unit
def test_autoscroll_accumulates_subbin_progress_at_default_speed(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Default 0.05× ticks must accumulate across bins (regression).

    Earlier the tick computed ``new_t = core.t_center + dt`` then
    quantized to a slider index. At default speed (0.05/30 ≈ 1.67ms)
    versus a 2ms simulated bin, the quantized index equalled the
    current slider, ``setValue`` was skipped, ``_core.t_center``
    never advanced, and playback froze. Fix: accumulate dt into a
    float playback cursor independent of slider quantization.
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        AUTOSCROLL_DEFAULT_SPEED,
        AUTOSCROLL_TICK_HZ,
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._toggle_play()  # initialise the float cursor
    initial_slider = viewer._slider.value()
    initial_cursor = viewer._autoscroll_cursor
    assert initial_cursor is not None

    # One tick advances the float cursor regardless of bin width.
    viewer._autoscroll_tick()
    assert viewer._autoscroll_cursor > initial_cursor

    # Enough ticks at default speed must cross at least one bin.
    time = viewer._data_source.time
    bin_dt = float(time[1] - time[0])
    per_tick = AUTOSCROLL_DEFAULT_SPEED / AUTOSCROLL_TICK_HZ
    n_ticks_to_cross = max(2, int(np.ceil(bin_dt / per_tick)))
    for _ in range(n_ticks_to_cross + 1):
        viewer._autoscroll_tick()
    assert viewer._slider.value() > initial_slider


@pytest.mark.unit
def test_manual_scrub_during_play_resyncs_cursor(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """User scrubbing the slider during playback resets the float cursor.

    Without resync, autoscroll would continue from the pre-scrub
    cursor position, snapping back on the next tick.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._toggle_play()
    target_idx = viewer._slider.value() + 100
    viewer._slider.setValue(target_idx)
    expected_cursor = float(viewer._data_source.time[target_idx])
    assert viewer._autoscroll_cursor == pytest.approx(expected_cursor)


@pytest.mark.unit
def test_core_set_t_center_during_play_resyncs_cursor(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Every navigation path that recenters during play must resync the cursor.

    Generalises the slider-drag test: any caller of
    ``core.set_t_center`` (Shift+Left/Right step-window, R reset,
    Left/Right step, click-to-recenter, N/Shift+N event navigator)
    routes through the same ``on_t_center_changed`` callback. Without
    this, the float cursor would lag the user's manual jumps and
    autoscroll would pull the view back on the next tick.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._toggle_play()
    target_t = float(
        viewer._data_source.time[viewer._slider.value() + 200]
    )
    viewer._core.set_t_center(target_t)
    assert viewer._autoscroll_cursor == pytest.approx(target_t)


@pytest.mark.unit
def test_cursor_markers_initial_dispatch(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Every built-in TimeAxisPanel has its center line + active-bin band
    placed at the initial t_center on construction."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    expected = viewer.core.t_center
    for panel in viewer._builtin_panels:
        assert panel._center_line.value() == pytest.approx(expected)
        lo, hi = panel._active_bin_band.getRegion()
        # Cursor lands inside its bin's [t_lo, t_hi] band.
        assert lo <= expected <= hi


@pytest.mark.unit
def test_cursor_markers_follow_t_center_changes(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Any ``core.set_t_center`` call updates every panel's markers."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    target_t = float(viewer._data_source.time[100])
    viewer.core.set_t_center(target_t)
    for panel in viewer._builtin_panels:
        assert panel._center_line.value() == pytest.approx(target_t)
        lo, hi = panel._active_bin_band.getRegion()
        assert lo <= target_t <= hi


@pytest.mark.unit
def test_cursor_markers_dispatched_to_extras_via_getattr(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """User-supplied ``extra_panels`` exposing ``set_cursor_markers``
    receive markers through the same dispatch (opt-in via ``getattr``)."""
    from PySide6 import QtWidgets as _W

    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    class _CursorRecordingPanel(_W.QWidget):
        cell_clicked = None  # marker not used here

        def __init__(self) -> None:
            _W.QWidget.__init__(self)
            self.cursor_calls: list[tuple[float, float, float]] = []

        # TimeAxisPanel surface (minimal — set_cursor_markers is opt-in).
        def update_window(self, payload) -> None:  # noqa: D401, ARG002
            return

        def x_link_target(self):
            return None

        def click_handler(self, callback) -> None:  # noqa: ARG002
            return

        def set_event_overlays(self, overlays) -> None:  # noqa: ARG002
            return

        def set_cursor_markers(
            self, t_center: float, t_lo: float, t_hi: float
        ) -> None:
            self.cursor_calls.append((t_center, t_lo, t_hi))

    plugin = _CursorRecordingPanel()
    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5, extra_panels=[plugin])
    # Initial dispatch should already have hit the plugin once.
    assert len(plugin.cursor_calls) >= 1
    target_t = float(viewer._data_source.time[200])
    viewer.core.set_t_center(target_t)
    assert plugin.cursor_calls[-1][0] == pytest.approx(target_t)


@pytest.mark.unit
def test_step_window_during_play_resyncs_cursor(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Shift+Left/Right (``_step_window``) re-anchors the cursor."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    viewer._toggle_play()
    initial_t_center = viewer._core.t_center
    viewer._step_window(+1)
    assert viewer._core.t_center > initial_t_center
    assert viewer._autoscroll_cursor == pytest.approx(viewer._core.t_center)


@pytest.mark.unit
def test_autoscroll_pauses_at_end_of_session(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Reaching the session's last bin auto-toggles play off."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    # Park the cursor at the very end first.
    viewer._slider.setValue(viewer._slider.maximum())
    viewer._toggle_play()
    assert viewer._play_button.isChecked()
    viewer._autoscroll_tick()
    # End-of-session → auto-pause.
    assert not viewer._play_button.isChecked()
    assert viewer._autoscroll_timer is None


@pytest.mark.unit
def test_qt_viewer_keyboard_window_scaling(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``[``  / ``]`` halve / double the window width; ``R`` resets."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=1.0)
    initial = viewer.core.t_width

    viewer._scale_t_width(0.5)
    assert viewer.core.t_width == pytest.approx(initial * 0.5)

    viewer._scale_t_width(2.0)
    assert viewer.core.t_width == pytest.approx(initial)

    # Step a full window.
    initial_center = viewer.core.t_center
    viewer._step_window(+1)
    assert viewer.core.t_center == pytest.approx(initial_center + viewer.core.t_width)

    # Reset.
    viewer._reset_view()
    assert viewer.core.t_width == pytest.approx(initial)
    assert viewer.core.t_center == pytest.approx(initial_center)


@pytest.mark.unit
def test_qt_viewer_controls_bar_always_visible(
    qapp,
    nl_fitted: FittedDetector,
    sim_session,
) -> None:
    """Controls bar is always visible — play/pause + speed are universal.

    Earlier the bar hid itself when neither overlays nor multi-run
    populated it. Auto-scroll added play + speed as universal
    affordances, so the bar is always present.
    """
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    bundle = RunBundle(
        results=nl_fitted.results,
        detector=nl_fitted.detector,
        spike_times=sim_session.spike_times,
        position_time=sim_session.time,
        position=sim_session.position,
        speed=sim_session.speed,
    )
    ds = InMemoryDecoderDataSource.from_single(bundle)
    viewer = QtViewer(ds, t_width=0.5)
    assert not viewer._controls_bar.isHidden()
    assert viewer._play_button is not None
    assert viewer._speed_combo is not None


@pytest.mark.unit
def test_qt_viewer_controls_bar_visible_when_multi_run_no_overlays(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Multi-run alone is enough to keep the controls bar visible
    (the model-swap dropdown lives there)."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert not viewer._controls_bar.isHidden()
    assert viewer._model_combo is not None


@pytest.mark.unit
def test_qt_viewer_controls_bar_with_overlays(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Attached overlays → dropdown lists them, checkbox toggles
    visibility, dropdown change updates active overlay name."""
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
    try:
        swr = EventOverlay.points(name="swr", times=np.array([1.0, 2.0]))
        theta = EventOverlay.points(name="theta", times=np.array([1.5]))
        for bundle in multi_run_bundles.values():
            bundle.event_overlays.append(swr)
            bundle.event_overlays.append(theta)
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        viewer = QtViewer(ds, t_width=0.5)

        # Bar visible + dropdown contains "(none)" + each overlay.
        assert not viewer._controls_bar.isHidden()
        items = [
            viewer._overlay_combo.itemText(i)
            for i in range(viewer._overlay_combo.count())
        ]
        assert items == ["(none)", "swr", "theta"]

        # Selecting "swr" updates core.active_overlay_name.
        swr_idx = viewer._overlay_combo.findText("swr")
        viewer._overlay_combo.setCurrentIndex(swr_idx)
        assert viewer.core.active_overlay_name == "swr"

        # Toggling theta's checkbox hides it.
        viewer._overlay_checkboxes["theta"].setChecked(False)
        assert viewer.core._overlay_visibility["theta"] is False
    finally:
        for n, b in multi_run_bundles.items():
            b.event_overlays[:] = original[n]
