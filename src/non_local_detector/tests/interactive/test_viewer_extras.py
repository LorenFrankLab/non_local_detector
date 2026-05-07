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
