"""Tests for the generic series view-models."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.visualization.interactive.view_models.series import (
    IntervalSeriesModel,
    LineSeriesModel,
    MetricSpec,
    MultiLineSeriesModel,
    ScatterSeriesModel,
)


@pytest.mark.unit
class TestLineSeriesModel:
    def test_window_clips_to_range(self) -> None:
        t = np.linspace(0.0, 10.0, 11)
        y = t * 2.0
        model = LineSeriesModel(name="ramp", t=t, y=y)
        t_window, y_window = model.window(2.5, 6.5)
        np.testing.assert_array_equal(t_window, [3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(y_window, [6.0, 8.0, 10.0, 12.0])

    def test_from_metric_spec(self) -> None:
        spec = MetricSpec.line(
            name="m",
            t=np.array([0.0, 1.0]),
            y=np.array([10.0, 20.0]),
            fill_below=True,
            thresholds=(5.0, 15.0),
        )
        model = LineSeriesModel.from_metric_spec(spec)
        assert model.fill_below is True
        assert model.thresholds == (5.0, 15.0)
        np.testing.assert_array_equal(model.y, spec.y)

    def test_from_metric_spec_wrong_kind_raises(self) -> None:
        spec = MetricSpec.scatter(name="s", t=np.array([0.0]), y=np.array([1.0]))
        with pytest.raises(ValueError, match="kind='line'"):
            LineSeriesModel.from_metric_spec(spec)


@pytest.mark.unit
class TestMultiLineSeriesModel:
    def test_window_clips_each_line(self) -> None:
        t = np.arange(10, dtype=float)
        ys = {"a": t * 2, "b": -t}
        model = MultiLineSeriesModel(name="multi", t=t, ys=ys)
        t_window, ys_window = model.window(2.0, 5.0)
        np.testing.assert_array_equal(t_window, [2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(ys_window["a"], [4.0, 6.0, 8.0, 10.0])
        np.testing.assert_array_equal(ys_window["b"], [-2.0, -3.0, -4.0, -5.0])

    def test_mismatched_y_length_raises(self) -> None:
        t = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="ys"):
            MultiLineSeriesModel(name="multi", t=t, ys={"a": np.zeros(99)})


@pytest.mark.unit
class TestScatterSeriesModel:
    def test_default_click_recenters(self) -> None:
        model = ScatterSeriesModel(
            name="s", t=np.array([0.0, 1.0]), y=np.array([10.0, 20.0])
        )
        assert model.click_recenters is True

    def test_window_filters_points(self) -> None:
        t = np.array([0.5, 1.5, 2.5, 3.5])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        model = ScatterSeriesModel(name="s", t=t, y=y)
        t_window, y_window = model.window(1.0, 3.0)
        np.testing.assert_array_equal(t_window, [1.5, 2.5])
        np.testing.assert_array_equal(y_window, [20.0, 30.0])


@pytest.mark.unit
class TestIntervalSeriesModel:
    def test_window_returns_overlapping_intervals(self) -> None:
        starts = np.array([0.0, 5.0, 10.0])
        ends = np.array([1.0, 6.0, 11.0])
        model = IntervalSeriesModel(name="ivl", t_start=starts, t_end=ends)
        s_out, e_out = model.window(4.0, 8.0)
        np.testing.assert_array_equal(s_out, [5.0])
        np.testing.assert_array_equal(e_out, [6.0])

    def test_partial_overlap_included(self) -> None:
        """An interval whose start is before t_window but end is inside
        should be returned."""
        model = IntervalSeriesModel(
            name="ivl",
            t_start=np.array([0.0]),
            t_end=np.array([5.0]),
        )
        # Window [3, 10] overlaps the [0, 5] interval.
        s_out, e_out = model.window(3.0, 10.0)
        np.testing.assert_array_equal(s_out, [0.0])
        np.testing.assert_array_equal(e_out, [5.0])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="t_end"):
            IntervalSeriesModel(
                name="ivl",
                t_start=np.array([0.0, 1.0]),
                t_end=np.array([1.0]),
            )

    def test_from_metric_spec(self) -> None:
        spec = MetricSpec.intervals(
            name="ivl",
            t_start=np.array([0.0, 1.0]),
            t_end=np.array([0.5, 1.5]),
            color="red",
        )
        model = IntervalSeriesModel.from_metric_spec(spec)
        assert model.color == "red"
        np.testing.assert_array_equal(model.t_start, [0.0, 1.0])
