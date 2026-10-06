"""Align measured tracking data to the actual bins of a decoder result."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from non_local_detector.exceptions import ValidationError


def _finite_runs(indices: np.ndarray, finite: np.ndarray):
    """Yield contiguous original sample rows, retaining missing sample breaks."""
    indices = indices[finite[indices]]
    if indices.size:
        yield from np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def align_tracking_to_results(
    position_info: pd.DataFrame,
    results: xr.Dataset,
    *,
    position_columns: Sequence[str],
    valid_position_intervals,
    categorical_columns: Sequence[str] = (),
    circular_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return tracking rows at decode centers and their measured-support mask.

    ``position_info`` keeps its original numeric timestamp index (seconds).
    ``results`` supplies ``time``, ``time_bin_start``, ``time_bin_end``, and,
    optionally, effective ``is_missing``. Each output row corresponds exactly
    to one result row, including after independent sequences are concatenated.

    Only bins entirely within a finite position sample span in a declared
    tracking interval are supported. NaN position samples split these spans;
    declared intervals stay separate even when adjacent. No endpoint position
    is held or extrapolated. Result missingness always excludes a row.

    An exact sample at an adjacent tracking boundary anchors both intervals'
    physical endpoints. This single indexed value cannot represent a jump
    between epochs; use separate measured spans or sequence-local tracking for
    discontinuous endpoints. This differs from spike boundary ownership.

    Numeric continuous columns are interpolated linearly within finite runs.
    Declare categorical columns explicitly: the preceding sample owns the
    center, including equality. Circular columns contain radians and follow
    the shortest arc, returned in [-pi, pi). Missing covariate samples also
    split that column's interpolation support. Non-numeric columns must be
    declared categorical.

    Returns
    -------
    aligned : pandas.DataFrame
        One row per result time. Unsupported rows contain NaNs. Individual
        covariates can additionally contain NaNs where their samples are missing.
    is_valid : numpy.ndarray of bool
        Whole-bin measured position support, excluding ``results.is_missing``.
        For an analysis, also require finite values in the covariates it uses.
    """
    if not isinstance(position_info, pd.DataFrame) or position_info.empty:
        raise ValidationError("position_info must be a non-empty DataFrame")
    try:
        sample_time = np.asarray(position_info.index, dtype=float)
        intervals = np.asarray(valid_position_intervals, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "Tracking timestamps and intervals must be numeric seconds"
        ) from error
    if not np.isfinite(sample_time).all() or np.any(np.diff(sample_time) <= 0):
        raise ValidationError(
            "Tracking timestamps must be finite and strictly increasing; resolve duplicate epochs upstream"
        )
    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
        or not len(intervals)
        or not np.isfinite(intervals).all()
        or np.any(intervals[:, 1] <= intervals[:, 0])
        or np.any(intervals[1:, 0] < intervals[:-1, 1])
    ):
        raise ValidationError(
            "valid_position_intervals must be ordered, non-overlapping [start, stop] intervals"
        )
    required = set(position_columns) | set(categorical_columns) | set(circular_columns)
    if not position_columns or not required.issubset(position_info.columns):
        raise ValidationError(
            "Declare existing position, categorical, and circular columns"
        )
    if set(categorical_columns) & (set(position_columns) | set(circular_columns)):
        raise ValidationError(
            "Position and circular columns must be numeric, not categorical"
        )
    for name in ("time", "time_bin_start", "time_bin_end"):
        if name not in results.coords:
            raise ValidationError(
                f"Result coordinates must include {name}; retain the saved bin bounds"
            )
    center = np.asarray(results.time, dtype=float)
    start = np.asarray(results.time_bin_start, dtype=float)
    stop = np.asarray(results.time_bin_end, dtype=float)
    if (
        center.ndim != 1
        or start.shape != center.shape
        or stop.shape != center.shape
        or not np.isfinite(np.concatenate([center, start, stop])).all()
        or np.any(start >= stop)
        or np.any(center < start)
        or np.any(center > stop)
    ):
        raise ValidationError(
            "Result times and bin bounds must be finite, aligned one-dimensional arrays"
        )
    missing = (
        np.asarray(results.is_missing, dtype=bool)
        if "is_missing" in results
        else np.zeros(len(center), bool)
    )
    if missing.shape != center.shape:
        raise ValidationError("Result is_missing must have one entry per bin")

    numeric = {}
    categorical = set(categorical_columns)
    for column in position_info:
        if column not in categorical:
            try:
                numeric[column] = position_info[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    f"Non-numeric column {column!r} must be declared categorical"
                ) from error
    finite_position = np.all(
        np.isfinite(np.column_stack([numeric[c] for c in position_columns])), axis=1
    )
    aligned = pd.DataFrame(index=pd.Index(center, name="time"))
    for column in position_info:
        aligned[column] = np.full(
            len(center), np.nan, dtype=object if column in categorical else float
        )
    valid = np.zeros(len(center), bool)
    for lower, upper in intervals:
        samples = np.flatnonzero((sample_time >= lower) & (sample_time <= upper))
        for run in _finite_runs(samples, finite_position):
            rows = (
                (start >= sample_time[run[0]])
                & (stop <= sample_time[run[-1]])
                & ~missing
            )
            valid |= rows
            if not rows.any():
                continue
            for column in position_info:
                values = (
                    position_info[column].to_numpy()
                    if column in categorical
                    else numeric[column]
                )
                finite_column = (
                    ~pd.isna(values) if column in categorical else np.isfinite(values)
                )
                for column_run in _finite_runs(run, finite_column):
                    column_rows = (
                        rows
                        & (start >= sample_time[column_run[0]])
                        & (stop <= sample_time[column_run[-1]])
                    )
                    if not column_rows.any():
                        continue
                    times, samples_at_time = sample_time[column_run], values[column_run]
                    if column in categorical:
                        owners = (
                            np.searchsorted(times, center[column_rows], side="right")
                            - 1
                        )
                        interpolated = samples_at_time[owners]
                    else:
                        if column in circular_columns:
                            samples_at_time = np.unwrap(samples_at_time)
                        interpolated = np.interp(
                            center[column_rows], times, samples_at_time
                        )
                        if column in circular_columns:
                            interpolated = (interpolated + np.pi) % (2 * np.pi) - np.pi
                    aligned.loc[column_rows, column] = interpolated
    return aligned, valid
