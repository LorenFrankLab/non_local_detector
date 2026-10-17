"""Physical encoding support and integrated interpolation exposure.

Position timestamps are sample centers. Recording bounds and continuous
tracking intervals determine which times can contribute encoding evidence.
Event weights are dimensionless; integrated sample exposure is in seconds.
"""

import warnings

import numpy as np

from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.time_edges import uniform_time_bin_width, validate_time_edges


class EncodingSupport:
    """Interpolation segments and integrals of their sample basis functions.

    ``encoding_time_range`` clips acquisition support without redefining the
    original interpolation basis. ``valid_position_intervals`` declares
    disconnected tracking segments; their endpoints are held constant.
    Uniform samples default to supported endpoint half-cells. NaN positions
    inside a tracking segment split its original timeline before acquisition
    clipping. Samples outside a declared segment cannot shorten its support.
    A sample on an adjacent segment's shared boundary anchors both bases and
    declares a shared physical endpoint. Discontinuous endpoints need separate
    measured sample spans or sequence-local tracking; one timestamp cannot
    represent two different positions. Spike ownership is a separate rule.
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
        # Establish each tracking segment before splitting on missing samples.
        # Acquisition bounds then clip these original interpolation bases;
        # they must not remove the bracketing sample centers from a basis.
        tracking_segments = (
            [(bounds[0], bounds[1], np.arange(time.size))]
            if intervals is None
            else [
                (start, stop, np.flatnonzero((time >= start) & (time <= stop)))
                for start, stop in intervals
            ]
        )
        for start, stop, segment_indices in tracking_segments:
            indices = segment_indices[finite[segment_indices]]
            if not indices.size:
                continue
            runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            for run in runs:
                first, last = run[0], run[-1]
                lower = (
                    start
                    if first == segment_indices[0]
                    else (time[first - 1] + time[first]) / 2
                )
                upper = (
                    stop
                    if last == segment_indices[-1]
                    else (time[last] + time[last + 1]) / 2
                )
                lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
                if upper <= lower:
                    continue
                self.segments.append((lower, upper, run))
                self._integrate(lower, upper, run)

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


def warn_if_no_encoding_exposure(
    exposure_weights, model: str, *, clusterless: bool, stacklevel: int = 3
) -> None:
    """Warn when an encoding fit has zero training exposure.

    A sample's exposure is its weight, zero for a NaN position, times the time
    its interpolation basis spends inside the encoding support (see
    :func:`prepare_encoding_support`). With none, rates are unidentified and
    the fit returns its floor model: rate-floor place fields (sorted spikes)
    or zero-rate electrodes (clusterless). That is usually a mask or label
    mistake.

    Parameters
    ----------
    exposure_weights : array_like, shape (n_samples,)
        Weighted sample exposure in seconds.
    model : str
        Name of the encoding model, for the message.
    clusterless : bool
        Whether the model is clusterless, which selects the stated consequence.
    stacklevel : int, optional
        Passed to ``warnings.warn``, by default 3 (the encoding fit's caller).
    """
    if not np.sum(exposure_weights) > 0.0:
        consequence = (
            "Every electrode is zero-rate, so its decode spikes are explained "
            "by the background rate alone."
            if clusterless
            else "Every place field is set to the rate floor, so the model "
            "carries no spatial information."
        )
        warnings.warn(
            f"{model} encoding has zero training exposure: no position sample "
            "has a positive weight, a finite position and time inside the "
            "encoding support, so the exposure weights sum to 0. "
            f"{consequence} Check the weights (or is_training, "
            "encoding_group_labels and environment_labels), NaN positions, and "
            "encoding_time_range / valid_position_intervals.",
            UserWarning,
            stacklevel=stacklevel,
        )


def largest_tracking_step(support: EncodingSupport, position) -> float:
    """Largest distance between consecutive tracking samples within a segment.

    Only samples that share an interpolation segment of ``support`` count, so
    jumps across declared tracking gaps or missing samples are ignored.

    Parameters
    ----------
    support : EncodingSupport
    position : array_like, shape (n_samples,) or (n_samples, n_position_dims)
        The samples ``support`` was built from, in the units of the bandwidth.

    Returns
    -------
    step : float
        Euclidean distance; 0 when no segment has two samples.
    """
    position = np.asarray(position, dtype=float).reshape(len(support.time), -1)
    steps = [
        np.linalg.norm(np.diff(position[indices], axis=0), axis=1)
        for _, _, indices in support.segments
        if indices.size > 1
    ]
    return float(max((step.max() for step in steps), default=0.0))


def warn_if_tracking_is_sparse(
    support: EncodingSupport, position, position_std, *, stacklevel: int = 3
) -> None:
    """Warn when tracking moves farther per sample than ``position_std``.

    Occupancy is built from the tracking samples, while spikes are placed on
    the interpolated path between them. For uniform motion with samples ``d``
    apart, a Gaussian occupancy of width ``sigma`` differs from that of the
    continuous path by at most about ``2 exp(-2 pi**2 sigma**2 / d**2)``:
    5e-9 at ``d = sigma``, 3e-4 at ``1.5 sigma`` and 1.4% at ``2 sigma``. Above
    ``sigma`` the place fields can be biased where spikes fall between samples.

    Parameters
    ----------
    support : EncodingSupport
    position : array_like, shape (n_samples,) or (n_samples, n_position_dims)
    position_std : float or array_like
        Position bandwidth; its smallest entry is used.
    stacklevel : int, optional
        Passed to ``warnings.warn``, by default 3 (the encoding fit's caller).
    """
    scale = float(np.min(position_std))
    step = largest_tracking_step(support, position)
    if step > scale:
        warnings.warn(
            f"Tracking moves up to {step:.3g} position units between samples, more "
            f"than position_std={scale:.3g}. Occupancy is estimated from the "
            "tracking samples while spikes are placed on the path between them, "
            "so place fields can be biased where spikes fall between distant "
            "samples. Use position sampled more often (or upsampled along the "
            "path), or a position_std at least as large as the largest step.",
            UserWarning,
            stacklevel=stacklevel,
        )
