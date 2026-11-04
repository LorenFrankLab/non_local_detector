"""Physical encoding support and integrated interpolation exposure.

Position timestamps are sample centers. Recording bounds and continuous
tracking intervals determine which times can contribute encoding evidence.
Event weights are dimensionless; integrated sample exposure is in seconds.
"""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import jax

from non_local_detector._validation import validate_weights
from non_local_detector._warnings import user_stacklevel
from non_local_detector.exceptions import DataError, DataQualityWarning, ValidationError
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

    Parameters
    ----------
    position_time : array_like, shape (n_samples,)
        Position sample times in seconds, finite and strictly increasing.
        Without ``valid_position_intervals`` they must be uniformly spaced.
    position : array_like, shape (n_samples,) or (n_samples, n_position_dims)
        Position at each sample. Rows with a NaN coordinate are treated as
        missing and split the interpolation segment.
    encoding_time_range : array_like, shape (2,), optional
        ``(start, stop)`` in seconds clipping the acquisition support. By
        default None, which uses the span of ``valid_position_intervals`` or,
        for uniform samples, the first and last sample extended by half a
        sample.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping ``(start, stop)`` intervals in seconds in
        which tracking was continuous. Required when ``position_time`` is
        irregularly sampled. By default None.

    Attributes
    ----------
    time : np.ndarray, shape (n_samples,)
        Position sample times in seconds.
    exposure : np.ndarray, shape (n_samples,)
        Integrated interpolation-basis exposure of each sample, in seconds.
    segments : list of tuple
        ``(start, stop, indices)`` for each continuous interpolation segment:
        its time bounds in seconds and the indices of its samples.
    bounds : np.ndarray, shape (2,)
        ``(start, stop)`` of the acquisition support, in seconds.
    indices : np.ndarray of int
        Indices of the samples with a finite position.
    """

    def __init__(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        *,
        encoding_time_range: ArrayLike | None = None,
        valid_position_intervals: ArrayLike | None = None,
    ) -> None:
        time = np.asarray(position_time, dtype=np.float64)
        position = np.asarray(position)
        if time.size == 0:
            msg = "Encoding has no position samples; position_time and position must be nonempty"
            raise ValidationError(msg)
        if (
            time.ndim != 1
            or time.size == 0
            or position.ndim not in (1, 2)
            or position.shape[0] != time.size
        ):
            msg = "position_time and position need matching, nonempty sample rows"
            raise ValidationError(msg)
        position = position.reshape(len(time), -1)
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            msg = "position_time must be finite and strictly increasing"
            raise DataError(
                msg,
                data_name="position_time",
            )
        if np.any(np.isinf(position)):
            msg = "position contains infinite values"
            raise DataError(msg, data_name="position")
        self.time = time
        self.exposure = np.zeros(time.size)
        self.segments: list[tuple[float, float, NDArray[np.intp]]] = []
        finite = np.all(np.isfinite(position), axis=1)
        self.indices = np.flatnonzero(finite)
        intervals = None
        if valid_position_intervals is not None:
            intervals = np.asarray(valid_position_intervals, dtype=float)
            if intervals.ndim != 2 or intervals.shape[1] != 2:
                msg = "valid_position_intervals must have shape (n_intervals, 2)"
                raise ValidationError(msg)
            for interval in intervals:
                validate_time_edges(interval, name="valid_position_intervals")
            if len(intervals) > 1 and np.any(intervals[1:, 0] < intervals[:-1, 1]):
                msg = "valid_position_intervals must be ordered and non-overlapping"
                raise ValidationError(msg)
            for start, stop in intervals:
                if not np.any(finite & (time >= start) & (time <= stop)):
                    msg = "Each valid_position_intervals segment needs a finite position sample"
                    raise ValidationError(
                        msg,
                        hint="Use encoding_time_range to clip an existing interpolation segment between samples.",
                    )
        if time.size > 1 and intervals is None:
            # Acquisition bounds do not establish where tracking was continuous.
            try:
                uniform_time_bin_width(np.asarray(position_time), name="position_time")
            except DataError as error:
                msg = "Irregular position_time requires explicit continuous tracking intervals"
                raise DataError(
                    msg,
                    data_name="position_time",
                    hint="Pass valid_position_intervals=[[start, stop], ...]. Acquisition bounds alone do not establish tracking continuity.",
                ) from error
        if encoding_time_range is not None:
            bounds = validate_time_edges(
                encoding_time_range, name="encoding_time_range"
            )
            if bounds.shape != (2,):
                msg = "encoding_time_range must be (start, stop)"
                raise ValidationError(msg)
        elif intervals is not None and len(intervals):
            bounds = np.array([intervals[0, 0], intervals[-1, 1]])
        elif time.size < 2:
            msg = "A single position sample requires encoding_time_range or valid_position_intervals"
            raise ValidationError(msg)
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

    def _integrate(self, start: float, stop: float, indices: NDArray[np.intp]) -> None:
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

    def contains(self, times: ArrayLike) -> NDArray[np.bool_]:
        """Return which times fall inside a supported interpolation segment.

        Parameters
        ----------
        times : array_like, shape (n_times,)
            Times in seconds.

        Returns
        -------
        is_supported : np.ndarray of bool, shape (n_times,)
            True where a time lies within the closed bounds of any segment.
        """
        times = np.asarray(times)
        mask = np.zeros(times.shape, dtype=bool)
        for start, stop, _ in self.segments:
            mask |= (times >= start) & (times <= stop)
        return mask

    def interpolate(
        self, values: ArrayLike, times: ArrayLike, *, fill_value: float = np.nan
    ) -> NDArray[np.float64]:
        """Linearly interpolate sample values within each supported segment.

        Parameters
        ----------
        values : array_like, shape (n_samples,) or (n_samples, n_columns)
            Values at the position samples.
        times : array_like, shape (n_times,)
            Times in seconds at which to interpolate.
        fill_value : float, optional
            Value for times outside every segment, by default NaN.

        Returns
        -------
        interpolated : np.ndarray, shape (n_times,) or (n_times, n_columns)
            Interpolated values; ``fill_value`` where a time is unsupported.
        """
        values, times = np.asarray(values), np.asarray(times)
        shape = (len(times), *values.shape[1:])
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

    def _segment_events(
        self, segment_index: int, spike_times: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.bool_],
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
    ]:
        """Spikes one segment owns, with their bracketing samples and fraction.

        Parameters
        ----------
        segment_index : int
            Index into ``self.segments``.
        spike_times : np.ndarray, shape (n_spikes,)
            Spike times in seconds.

        Returns
        -------
        owned : np.ndarray of bool, shape (n_spikes,)
        left, right : np.ndarray of int, shape (n_owned,)
            Sample indices bracketing each owned spike (equal at an endpoint).
        fraction : np.ndarray, shape (n_owned,)
            Weight of ``right``; ``left`` carries ``1 - fraction``.
        """
        start, stop, indices = self.segments[segment_index]
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
        return owned, indices[left], indices[right], fraction

    def event_basis(
        self, spike_times: ArrayLike
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
    ]:
        """Each supported spike's bracketing samples and interpolation fraction.

        A spike's unit event is split between its two bracketing samples: the
        ``left`` sample takes ``1 - fraction`` and the ``right`` one ``fraction``.

        Parameters
        ----------
        spike_times : array_like, shape (n_spikes,)
            Spike times in seconds.

        Returns
        -------
        spike_index : np.ndarray of int, shape (n_events,)
            Index into ``spike_times`` of each spike the support owns, in
            input order.
        left, right : np.ndarray of int, shape (n_events,)
            Sample indices bracketing each spike (equal at an endpoint).
        fraction : np.ndarray, shape (n_events,)
            Weight of ``right``, in [0, 1].
        """
        spike_times = np.asarray(spike_times, dtype=float)
        parts = [
            self._segment_events(segment_index, spike_times)
            for segment_index in range(len(self.segments))
        ]
        if not parts:
            empty = np.zeros(0, dtype=int)
            return empty, empty, empty, np.zeros(0)
        spike_index = np.concatenate([np.flatnonzero(part[0]) for part in parts])
        order = np.argsort(spike_index, kind="stable")
        left, right, fraction = (
            np.concatenate([part[column] for part in parts])[order]
            for column in (1, 2, 3)
        )
        return spike_index[order], left, right, fraction

    def event_counts(
        self, spike_times: ArrayLike, weights: ArrayLike
    ) -> NDArray[np.float64]:
        """Dimensionless weighted events assigned to their interpolation basis.

        Parameters
        ----------
        spike_times : array_like, shape (n_spikes,)
            Spike times in seconds.
        weights : array_like, shape (n_samples,)
            Dimensionless weight per position sample, applied to the samples
            bracketing each supported spike.

        Returns
        -------
        counts : np.ndarray, shape (n_samples,)
            Weighted event count assigned to each position sample.
        """
        spike_times, weights = np.asarray(spike_times, dtype=float), np.asarray(weights)
        counts = np.zeros(len(self.time))
        for segment_index in range(len(self.segments)):
            _, left, right, fraction = self._segment_events(segment_index, spike_times)
            counts += np.bincount(
                left, weights=(1 - fraction) * weights[left], minlength=len(counts)
            )
            counts += np.bincount(
                right, weights=fraction * weights[right], minlength=len(counts)
            )
        return counts


def prepare_encoding_support(
    position_time: ArrayLike,
    position: ArrayLike,
    weights: "NDArray[Any] | jax.Array | None",
    *,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> tuple[
    EncodingSupport, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Prepare separate event weights and exposure weights for a backend.

    Parameters
    ----------
    position_time : array_like, shape (n_samples,)
        Position sample times in seconds.
    position : array_like, shape (n_samples,) or (n_samples, n_position_dims)
        Position at each sample.
    weights : np.ndarray or jax.Array, shape (n_samples,), optional
        Dimensionless weight of each sample, or None for all ones. Weights of
        samples with a non-finite position are set to zero.
    encoding_time_range : array_like, shape (2,), optional
        ``(start, stop)`` in seconds clipping the acquisition support. See
        :class:`EncodingSupport`.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Continuous tracking intervals in seconds. See
        :class:`EncodingSupport`.
    _encoding_support : EncodingSupport, optional
        A prebuilt support to reuse instead of constructing one from
        ``position_time`` and the interval arguments.

    Returns
    -------
    support : EncodingSupport
        The interpolation support.
    weights : np.ndarray, shape (n_samples,)
        Validated event weights (dimensionless), zero for non-finite positions.
    exposure_weights : np.ndarray, shape (n_samples,)
        ``weights`` times the sample exposure, in seconds.
    position : np.ndarray, shape (n_samples, n_position_dims)
        Position with non-finite values replaced by zero.
    """
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
    exposure_weights: ArrayLike, model: str, *, clusterless: bool
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
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )


def largest_tracking_step(support: EncodingSupport, position: ArrayLike) -> float:
    """Largest distance between consecutive tracking samples within a segment.

    Only samples that share an interpolation segment of ``support`` count, so
    jumps across declared tracking gaps or missing samples are ignored.

    Parameters
    ----------
    support : EncodingSupport
        Interpolation support built from the position samples.
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
    support: EncodingSupport, position: ArrayLike, position_std: ArrayLike
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
        Interpolation support built from the position samples.
    position : array_like, shape (n_samples,) or (n_samples, n_position_dims)
        The position samples ``support`` was built from.
    position_std : float or array_like
        Position bandwidth; its smallest entry is used.
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
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )
