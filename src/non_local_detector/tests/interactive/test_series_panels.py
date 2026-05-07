"""Construction-time tests for the four generic series panels.

These verify each panel actually wires up the constructor options
the user passes (fill_below adds a FillBetweenItem, thresholds add
dashed InfiniteLines, multi-line creates one PlotDataItem per
``ys`` entry with distinct pens). They run on the offscreen Qt
platform and don't assert pixel content.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Force offscreen Qt platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6 import QtWidgets

from non_local_detector.visualization.interactive.panels.qt.series import (
    IntervalSeriesPanel,
    LineSeriesPanel,
    MultiLineSeriesPanel,
    ScatterSeriesPanel,
)
from non_local_detector.visualization.interactive.view_models.base import WindowPayload
from non_local_detector.visualization.interactive.view_models.series import (
    IntervalSeriesModel,
    LineSeriesModel,
    MultiLineSeriesModel,
    ScatterSeriesModel,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _payload_for_window(t_start: float, t_stop: float, n: int = 20) -> WindowPayload:
    time = np.linspace(t_start, t_stop, n)
    return WindowPayload(request_id=0, time=time, indices=slice(0, n))


@pytest.mark.unit
def test_line_panel_plain_has_no_fill_no_thresholds(qapp) -> None:
    model = LineSeriesModel(
        name="m", t=np.linspace(0.0, 10.0, 100), y=np.linspace(0.0, 1.0, 100)
    )
    panel = LineSeriesPanel(model)
    assert panel._fill_curve is None
    threshold_lines = [
        item for item in panel.getPlotItem().items if isinstance(item, pg.InfiniteLine)
    ]
    assert threshold_lines == []


@pytest.mark.unit
def test_line_panel_fill_below_attaches_fill_between(qapp) -> None:
    model = LineSeriesModel(
        name="m",
        t=np.linspace(0.0, 10.0, 100),
        y=np.linspace(0.0, 1.0, 100),
        fill_below=True,
    )
    panel = LineSeriesPanel(model)
    assert isinstance(panel._fill_curve, pg.FillBetweenItem)


@pytest.mark.unit
def test_line_panel_thresholds_render_dashed_lines(qapp) -> None:
    model = LineSeriesModel(
        name="m",
        t=np.linspace(0.0, 10.0, 100),
        y=np.linspace(0.0, 1.0, 100),
        thresholds=(0.25, 0.75),
    )
    panel = LineSeriesPanel(model)
    threshold_lines = [
        item for item in panel.getPlotItem().items if isinstance(item, pg.InfiniteLine)
    ]
    assert len(threshold_lines) == 2
    positions = sorted(line.value() for line in threshold_lines)
    assert positions == [pytest.approx(0.25), pytest.approx(0.75)]


@pytest.mark.unit
def test_line_panel_window_renders_clipped_data(qapp) -> None:
    model = LineSeriesModel(name="m", t=np.arange(10.0), y=np.arange(10.0) * 2.0)
    panel = LineSeriesPanel(model)
    panel.update_window(_payload_for_window(2.0, 5.0, n=4))
    xs, ys = panel._line.getData()
    np.testing.assert_array_equal(xs, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(ys, [4.0, 6.0, 8.0, 10.0])


@pytest.mark.unit
def test_multi_line_panel_creates_one_line_per_entry_with_distinct_pens(qapp) -> None:
    t = np.linspace(0.0, 1.0, 10)
    ys = {"a": t, "b": -t, "c": 2 * t}
    model = MultiLineSeriesModel(name="multi", t=t, ys=ys)
    panel = MultiLineSeriesPanel(model)
    assert set(panel._lines) == {"a", "b", "c"}
    pens = [panel._lines[label].opts["pen"].color().name() for label in ("a", "b", "c")]
    # Three distinct colors from the default cycle.
    assert len(set(pens)) == 3


@pytest.mark.unit
def test_multi_line_panel_explicit_colors_overrides_palette(qapp) -> None:
    t = np.linspace(0.0, 1.0, 5)
    model = MultiLineSeriesModel(
        name="multi",
        t=t,
        ys={"a": t, "b": -t},
        colors={"a": "#aaaaaa", "b": "#bbbbbb"},
    )
    panel = MultiLineSeriesPanel(model)
    assert panel._lines["a"].opts["pen"].color().name() == "#aaaaaa"
    assert panel._lines["b"].opts["pen"].color().name() == "#bbbbbb"


@pytest.mark.unit
def test_multi_line_panel_window_renders_each_line(qapp) -> None:
    t = np.arange(10.0)
    model = MultiLineSeriesModel(
        name="multi",
        t=t,
        ys={"a": t * 2, "b": -t},
    )
    panel = MultiLineSeriesPanel(model)
    panel.update_window(_payload_for_window(2.0, 5.0, n=4))
    xs_a, ys_a = panel._lines["a"].getData()
    xs_b, ys_b = panel._lines["b"].getData()
    np.testing.assert_array_equal(xs_a, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(ys_a, [4.0, 6.0, 8.0, 10.0])
    np.testing.assert_array_equal(xs_b, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(ys_b, [-2.0, -3.0, -4.0, -5.0])


@pytest.mark.unit
def test_scatter_panel_renders_points_in_window(qapp) -> None:
    model = ScatterSeriesModel(
        name="s",
        t=np.array([0.5, 1.5, 2.5, 3.5]),
        y=np.array([10.0, 20.0, 30.0, 40.0]),
    )
    panel = ScatterSeriesPanel(model)
    panel.update_window(_payload_for_window(1.0, 3.0))
    pts = panel._scatter.points()
    xs = sorted(pt.pos().x() for pt in pts)
    np.testing.assert_array_equal(xs, [1.5, 2.5])


@pytest.mark.unit
def test_interval_panel_replaces_regions_per_window(qapp) -> None:
    """Each ``update_window`` call removes prior regions and adds new ones."""
    model = IntervalSeriesModel(
        name="ivl",
        t_start=np.array([0.0, 5.0, 10.0]),
        t_end=np.array([1.0, 6.0, 11.0]),
    )
    panel = IntervalSeriesPanel(model)
    panel.update_window(_payload_for_window(0.0, 7.0))
    assert len(panel._regions) == 2  # [0,1] and [5,6]

    panel.update_window(_payload_for_window(8.0, 12.0))
    assert len(panel._regions) == 1  # only [10, 11] now
