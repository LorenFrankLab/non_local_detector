"""Physical encoding support and integrated interpolation exposure.

Position timestamps are sample centers. Recording bounds and continuous
tracking intervals determine which times can contribute encoding evidence.
Event weights are dimensionless; integrated sample exposure is in seconds.
"""

import numpy as np

from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.time_edges import uniform_time_bin_width, validate_time_edges


class EncodingSupport:
    """Interpolation segments and integrals of their sample basis functions.

    ``encoding_time_range`` clips acquisition support without redefining the
    original interpolation basis. ``valid_position_intervals`` declares
    disconnected tracking segments; their endpoints are held constant.
    Uniform samples default to supported endpoint half-cells. NaN positions
    split the original timeline before any samples are removed.
    """

    def __init__(
        self,
        position_time,
        position,
        *,
        encoding_time_range=None,
        valid_position_intervals=None,
    ):
        time = np.asarray(position_time, dtype=np.float64)
        position = np.asarray(position)
        if time.size == 0:
            raise ValidationError(
                "Encoding has no position samples; position_time and position must be nonempty"
            )
        if (
            time.ndim != 1
            or time.size == 0
            or position.ndim not in (1, 2)
            or position.shape[0] != time.size
        ):
            raise ValidationError(
                "position_time and position need matching, nonempty sample rows"
            )
        position = position.reshape(len(time), -1)
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            raise DataError(
                "position_time must be finite and strictly increasing",
                data_name="position_time",
            )
        if np.any(np.isinf(position)):
            raise DataError("position contains infinite values", data_name="position")
        self.time = time
        self.exposure = np.zeros(time.size)
        self.segments = []
        finite = np.all(np.isfinite(position), axis=1)
        self.indices = np.flatnonzero(finite)
        intervals = None
        if valid_position_intervals is not None:
            intervals = np.asarray(valid_position_intervals, dtype=float)
            if intervals.ndim != 2 or intervals.shape[1] != 2:
                raise ValidationError(
                    "valid_position_intervals must have shape (n_intervals, 2)"
                )
            for interval in intervals:
                validate_time_edges(interval, name="valid_position_intervals")
            if len(intervals) > 1 and np.any(intervals[1:, 0] < intervals[:-1, 1]):
                raise ValidationError(
                    "valid_position_intervals must be ordered and non-overlapping"
                )
            for start, stop in intervals:
                if not np.any(finite & (time >= start) & (time <= stop)):
                    raise ValidationError(
                        "Each valid_position_intervals segment needs a finite position sample",
                        hint="Use encoding_time_range to clip an existing interpolation segment between samples.",
                    )
        if time.size > 1 and intervals is None:
            # Acquisition bounds do not establish where tracking was continuous.
            try:
                uniform_time_bin_width(np.asarray(position_time), name="position_time")
            except DataError as error:
                raise DataError(
                    "Irregular position_time requires explicit continuous tracking intervals",
                    data_name="position_time",
                    hint="Pass valid_position_intervals=[[start, stop], ...]. Acquisition bounds alone do not establish tracking continuity.",
                ) from error
        if encoding_time_range is not None:
            bounds = validate_time_edges(
                encoding_time_range, name="encoding_time_range"
            )
            if bounds.shape != (2,):
                raise ValidationError("encoding_time_range must be (start, stop)")
        elif intervals is not None and len(intervals):
            bounds = np.array([intervals[0, 0], intervals[-1, 1]])
        elif time.size < 2:
            raise ValidationError(
                "A single position sample requires encoding_time_range or valid_position_intervals"
            )
        else:
            width = uniform_time_bin_width(
                np.asarray(position_time), name="position_time"
            )
            bounds = np.array([time[0] - width / 2, time[-1] + width / 2])
        self.bounds = bounds
        if not self.indices.size:
            return
        runs = np.split(self.indices, np.flatnonzero(np.diff(self.indices) > 1) + 1)
        for run in runs:
            first, last = run[0], run[-1]
            lower = bounds[0] if first == 0 else (time[first - 1] + time[first]) / 2
            upper = (
                bounds[1]
                if last == time.size - 1
                else (time[last] + time[last + 1]) / 2
            )
            candidates = [(max(lower, bounds[0]), min(upper, bounds[1]), run)]
            if intervals is not None:
                candidates = []
                for start, stop in intervals:
                    indices = run[(time[run] >= start) & (time[run] <= stop)]
                    if not indices.size:
                        continue
                    candidates.append(
                        (
                            max(start, lower, bounds[0]),
                            min(stop, upper, bounds[1]),
                            indices,
                        )
                    )
            for start, stop, indices in candidates:
                if stop <= start or not indices.size:
                    continue
                self.segments.append((start, stop, indices))
                self._integrate(start, stop, indices)

    def _integrate(self, start, stop, indices):
        time = self.time[indices]
        self.exposure[indices[0]] += max(min(stop, time[0]) - start, 0)
        self.exposure[indices[-1]] += max(stop - max(start, time[-1]), 0)
        if time.size == 1:
            self.exposure[indices[0]] += max(
                min(stop, time[0]) - max(start, time[0]), 0
            )
            return
        left, right = time[:-1], time[1:]
        a, b = np.maximum(start, left), np.minimum(stop, right)
        active = b > a
        duration = np.where(active, b - a, 0)
        right_mass = np.where(
            active, ((b - left) ** 2 - (a - left) ** 2) / (2 * (right - left)), 0
        )
        np.add.at(self.exposure, indices[:-1], duration - right_mass)
        np.add.at(self.exposure, indices[1:], right_mass)

    def contains(self, times):
        times = np.asarray(times)
        mask = np.zeros(times.shape, dtype=bool)
        for start, stop, _ in self.segments:
            mask |= (times >= start) & (times <= stop)
        return mask

    def interpolate(self, values, times, *, fill_value=np.nan):
        values, times = np.asarray(values), np.asarray(times)
        shape = (len(times),) + values.shape[1:]
        result = np.full(shape, fill_value, dtype=float)
        for start, stop, indices in self.segments:
            mask = (times >= start) & (times <= stop)
            if values.ndim == 1:
                result[mask] = np.interp(
                    times[mask], self.time[indices], values[indices]
                )
            else:
                for column in range(values.shape[1]):
                    result[mask, column] = np.interp(
                        times[mask], self.time[indices], values[indices, column]
                    )
        return result

    def event_counts(self, spike_times, weights):
        """Dimensionless weighted events assigned to their interpolation basis."""
        spike_times, weights = np.asarray(spike_times, dtype=float), np.asarray(weights)
        counts = np.zeros(len(self.time))
        for segment_index, (start, stop, indices) in enumerate(self.segments):
            shared_stop = (
                segment_index + 1 < len(self.segments)
                and stop == self.segments[segment_index + 1][0]
            )
            owned = (spike_times >= start) & (
                spike_times < stop if shared_stop else spike_times <= stop
            )
            times = spike_times[owned]
            sample_times = self.time[indices]
            left = np.clip(
                np.searchsorted(sample_times, times, side="right") - 1,
                0,
                len(indices) - 1,
            )
            right = np.minimum(left + 1, len(indices) - 1)
            width = sample_times[right] - sample_times[left]
            fraction = np.clip(
                np.divide(
                    times - sample_times[left],
                    width,
                    out=np.zeros_like(times),
                    where=width > 0,
                ),
                0,
                1,
            )
            counts += np.bincount(
                indices[left],
                weights=(1 - fraction) * weights[indices[left]],
                minlength=len(counts),
            )
            counts += np.bincount(
                indices[right],
                weights=fraction * weights[indices[right]],
                minlength=len(counts),
            )
        return counts


def prepare_encoding_support(
    position_time,
    position,
    weights,
    *,
    encoding_time_range=None,
    valid_position_intervals=None,
    _encoding_support=None,
):
    """Prepare separate event weights and exposure weights for a backend."""
    from non_local_detector.likelihoods.common import validate_weights

    position = np.asarray(position)
    position = position[:, None] if position.ndim == 1 else position
    support = _encoding_support or EncodingSupport(
        position_time,
        position,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
    )
    weights = (
        np.ones(len(position))
        if weights is None
        else validate_weights(weights, len(position))
    )
    weights = np.where(np.all(np.isfinite(position), axis=1), weights, 0.0)
    exposure_weights = weights * support.exposure
    return support, weights, exposure_weights, np.nan_to_num(position)
