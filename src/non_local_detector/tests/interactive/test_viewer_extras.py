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
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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
def test_qt_viewer_controls_bar_hidden_with_no_overlays(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """No overlays attached → controls bar is hidden (clean default UI)."""
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer._controls_bar.isHidden()


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
