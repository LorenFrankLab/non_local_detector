"""Tests of ``align_tracking_to_results`` (tracking rows at decode-bin centers)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from non_local_detector.analysis import align_tracking_to_results

pytestmark = pytest.mark.unit


def _result_grid(start=0.0, stop=1.0, n_bins=500):
    edges = np.linspace(start, stop, n_bins + 1)
    return xr.Dataset(
        coords={
            "time": (edges[:-1] + edges[1:]) / 2,
            "time_bin_start": ("time", edges[:-1]),
            "time_bin_end": ("time", edges[1:]),
            "is_missing": ("time", np.zeros(n_bins, bool)),
        }
    )


def test_camera_tracking_alignment_interpolates_linear_motion_at_bin_centers():
    camera_time = np.arange(31) / 30
    tracking = pd.DataFrame(
        {"x": 100 * camera_time, "y": 0.0, "segment": 0, "angle": 0.0},
        index=camera_time,
    )
    results = _result_grid()
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x", "y"],
        valid_position_intervals=[[0.0, 1.0]],
        categorical_columns=["segment"],
        circular_columns=["angle"],
    )
    assert valid.all()
    np.testing.assert_array_equal(aligned.index, results.time)
    # x is linear in time, so linear interpolation is exact up to rounding.
    np.testing.assert_allclose(aligned["x"], 100 * results.time.values, atol=1e-10)
    np.testing.assert_array_equal(aligned["y"], 0.0)
    np.testing.assert_array_equal(aligned["segment"], 0)
    np.testing.assert_allclose(aligned["angle"], 0.0, atol=1e-12)


def test_alignment_preserves_gaps_missingness_and_endpoint_support():
    tracking = pd.DataFrame(
        {"x": [0.0, 1.0, np.nan, 3.0, 4.0, 100.0, 101.0]},
        index=[0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
    )
    results = _result_grid(-0.5, 11.5, 24)
    results["is_missing"][1] = True
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x"],
        valid_position_intervals=[[0.0, 4.0], [10.0, 11.0]],
    )
    expected = np.isin(np.arange(24), [2, 7, 8, 21, 22])
    np.testing.assert_array_equal(valid, expected)
    assert aligned.loc[~valid, "x"].isna().all()
    np.testing.assert_allclose(
        aligned.loc[valid, "x"], [0.75, 3.25, 3.75, 100.25, 100.75]
    )


def test_categorical_ownership_and_shortest_arc_stay_within_segments():
    tracking = pd.DataFrame(
        {
            "x": [0.0, 1.0, 100.0, 101.0],
            "label": ["a", "b", "c", "d"],
            "angle": [3.1, -3.1, 0.1, 0.3],
        },
        index=[0.0, 1.0, 2.0, 3.0],
    )
    results = _result_grid(0, 3, 6)
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x"],
        valid_position_intervals=[[0.0, 1.0], [2.0, 3.0]],
        categorical_columns=["label"],
        circular_columns=["angle"],
    )
    np.testing.assert_array_equal(valid, [True, True, False, False, True, True])
    assert aligned.loc[valid, "label"].tolist() == ["a", "a", "c", "c"]
    assert (np.abs(aligned.loc[valid, "angle"].to_numpy()[:2]) > 3.1).all()
    np.testing.assert_allclose(aligned.loc[valid, "x"], [0.25, 0.75, 100.25, 100.75])


def test_adjacent_tracking_intervals_do_not_interpolate_across_segment_boundaries():
    tracking = pd.DataFrame({"x": [0.0, 1.0, 100.0, 101.0]}, index=[0.0, 0.9, 1.1, 2.0])
    results = _result_grid(0, 2, 20)
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x"],
        valid_position_intervals=[[0, 1], [1, 2]],
    )
    assert not valid[9:11].any()
    assert aligned.iloc[9:11]["x"].isna().all()


def test_missing_covariates_do_not_interpolate_across_nan_runs():
    tracking = pd.DataFrame(
        {"x": np.arange(5.0), "speed": [1.0, 1.0, np.nan, 3.0, 3.0]},
        index=np.arange(5.0),
    )
    aligned, valid = align_tracking_to_results(
        tracking,
        _result_grid(0, 4, 8),
        position_columns=["x"],
        valid_position_intervals=[[0, 4]],
    )
    assert valid.all()
    assert aligned.iloc[2:6]["speed"].isna().all()
