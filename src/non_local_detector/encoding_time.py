"""Physical encoding support and integrated interpolation exposure.

Position timestamps are sample centers. Recording bounds and continuous
tracking intervals determine which times can contribute encoding evidence.
Event weights are dimensionless; integrated sample exposure is in seconds.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.time_edges import uniform_time_bin_width, validate_time_edges

_QUADRATURE_CACHE_BYTES = 32 * 2**20
_QUADRATURE_MAX_BYTES = 512 * 2**20
_QUADRATURE_SOURCE_ROWS = 2048
_QUADRATURE_FILL_PARTS = 8192
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(4)


@dataclass(frozen=True)
class ExposureQuadrature:
    """Positive integration weights on the original continuous tracking basis.

    Spatial integration uses four-point Gauss--Legendre rules with subinterval
    displacement at most half a spatial scale. Histogram edge crossings are
    additional boundaries, making residence times exact for straight trajectories
    on rectangular grids. Weight interpolation remains linear and integrates
    exactly regardless of the spatial approximation or EM responsibilities.
    """

    times: np.ndarray
    positions: np.ndarray
    durations: np.ndarray
    left: np.ndarray
    right: np.ndarray
    fraction: np.ndarray

    def weighted_exposure(self, weights):
        result = np.empty(len(self.times))
        for rows, mass in self.iter_weighted_exposure(weights):
            result[rows] = mass
        return result

    def iter_weighted_exposure(self, weights):
        """Evaluate weights in bounded blocks without full-size gather temporaries."""
        weights = np.asarray(weights, dtype=float)
        for start in range(0, len(self.times), 32768):
            rows = slice(start, min(start + 32768, len(self.times)))
            fraction = self.fraction[rows]
            yield (
                rows,
                self.durations[rows]
                * (
                    (1.0 - fraction) * weights[self.left[rows]]
                    + fraction * weights[self.right[rows]]
                ),
            )

    @property
    def nbytes(self):
        return sum(value.nbytes for value in vars(self).values())


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
        self._quadrature_cache = None
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

    def quadrature(self, position, *, spatial_scale, bin_edges=None):
        """Integrate the supported trajectory without assigning time to samples.

        The latest small geometry is cached independently of sample weights, so
        complementary encoding groups and EM refits reuse identical nodes. The
        cache is capped at 32 MiB; larger requests are computed without retaining
        a second copy on the support. No recording-length neuron matrix is built.
        Endpoint holds, missing-data splits and clipping use ``self.segments``.
        """
        position = np.asarray(position, dtype=float).reshape(len(self.time), -1)
        scale = np.asarray(spatial_scale, dtype=float)
        if scale.size == 1:
            scale = np.full(position.shape[1], scale.item())
        elif scale.size != position.shape[1]:
            # Linearized-track bandwidths live in a different coordinate basis.
            scale = np.full(position.shape[1], np.min(scale))
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValidationError(
                "Trajectory integration needs a positive finite spatial scale"
            )
        edges = () if bin_edges is None else tuple(np.asarray(e) for e in bin_edges)
        if edges and len(edges) != position.shape[1]:
            raise ValidationError(
                "Histogram edges and trajectory dimensions must agree"
            )
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(position).tobytes())
        digest.update(scale.tobytes())
        for edge in edges:
            digest.update(np.asarray(edge.shape, dtype=np.intp).tobytes())
            digest.update(edge.tobytes())
        key = digest.digest()
        if self._quadrature_cache is not None and self._quadrature_cache[0] == key:
            return self._quadrature_cache[1]

        max_nodes = _QUADRATURE_MAX_BYTES // (8 * (position.shape[1] + 5))

        def resource_error():
            raise ValidationError(
                "Trajectory quadrature exceeds its 512 MiB geometry budget",
                hint="Use a coarser spatial bandwidth/grid or a shorter encoding period.",
            )

        def source_blocks():
            # No per-interval array objects are retained. Source metadata and
            # generation workspaces are bounded regardless of recording length.
            for start, stop, indices in self.segments:
                first, last = indices[0], indices[-1]
                if first == last:
                    yield (
                        np.array([start]),
                        np.array([stop]),
                        np.array([first]),
                        np.array([first]),
                    )
                    continue
                if start < min(stop, self.time[first]):
                    yield (
                        np.array([start]),
                        np.array([min(stop, self.time[first])]),
                        np.array([first]),
                        np.array([first]),
                    )
                for offset in range(0, len(indices) - 1, _QUADRATURE_SOURCE_ROWS):
                    left = indices[offset : offset + _QUADRATURE_SOURCE_ROWS]
                    right = indices[offset + 1 : offset + 1 + _QUADRATURE_SOURCE_ROWS]
                    left = left[: len(right)]
                    a = np.maximum(start, self.time[left])
                    b = np.minimum(stop, self.time[right])
                    active = b > a
                    if np.any(active):
                        yield a[active], b[active], left[active], right[active]
                if max(start, self.time[last]) < stop:
                    yield (
                        np.array([max(start, self.time[last])]),
                        np.array([stop]),
                        np.array([last]),
                        np.array([last]),
                    )

        def prepare_block(block):
            start, stop, left, right = block
            width = self.time[right] - self.time[left]
            moving = width > 0
            safe_width = np.where(moving, width, 1.0)
            a = np.where(moving, (start - self.time[left]) / safe_width, 0.0)
            b = np.where(moving, (stop - self.time[left]) / safe_width, 1.0)
            delta = position[right] - position[left]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                needed = np.maximum(
                    1.0,
                    np.ceil(
                        2 * np.max(np.abs((b - a)[:, None] * delta) / scale, axis=1)
                    ),
                )
            # Check before conversion to an integer or allocation: tiny scales
            # can request an unrepresentable subdivision count.
            if np.any(~np.isfinite(needed)) or np.any(needed > max_nodes // 4):
                resource_error()
            parts = needed.astype(np.intp)
            has_crossing = np.zeros(len(left), dtype=bool)
            crossing_ranges = []
            for dim, edge in enumerate(edges):
                p_start = position[left, dim] + a * delta[:, dim]
                p_stop = position[left, dim] + b * delta[:, dim]
                lo = np.searchsorted(edge, np.minimum(p_start, p_stop), side="right")
                hi = np.searchsorted(edge, np.maximum(p_start, p_stop), side="left")
                crossing_ranges.append((lo, hi))
                has_crossing |= (hi > lo) & (delta[:, dim] != 0)

            def boundaries(row):
                cuts = [np.linspace(a[row], b[row], int(parts[row]) + 1)]
                for dim, (edge, (lo, hi)) in enumerate(
                    zip(edges, crossing_ranges, strict=True)
                ):
                    if hi[row] > lo[row] and delta[row, dim] != 0:
                        cuts.append(
                            (edge[lo[row] : hi[row]] - position[left[row], dim])
                            / delta[row, dim]
                        )
                return np.unique(np.concatenate(cuts))

            counts = parts.copy()
            if crossing_ranges:
                upper_counts = parts + sum(
                    np.maximum(hi - lo, 0) for lo, hi in crossing_ranges
                )
                if np.any(upper_counts > max_nodes // 4):
                    resource_error()
            for row in np.flatnonzero(has_crossing):
                counts[row] = len(boundaries(row)) - 1
            return (
                start,
                stop,
                left,
                right,
                width,
                a,
                b,
                parts,
                has_crossing,
                counts,
                boundaries,
            )

        # Count first, then allocate the explicit node arrays once. Resident
        # geometry is four nodes per interval before spatial/grid subdivisions;
        # source/fill blocks do not grow with the complete recording.
        n_nodes = 0
        for block in source_blocks():
            counts = prepare_block(block)[-2]
            n_nodes += 4 * int(np.sum(counts))
            if n_nodes > max_nodes:
                resource_error()
        times = np.empty(n_nodes)
        durations = np.empty(n_nodes)
        left = np.empty(n_nodes, dtype=np.intp)
        right = np.empty(n_nodes, dtype=np.intp)
        fraction = np.empty(n_nodes)
        points = np.empty((n_nodes, position.shape[1]))

        def fill(
            destination, src_left, src_right, node_fraction, node_duration, node_time
        ):
            left[destination] = src_left
            right[destination] = src_right
            fraction[destination] = node_fraction
            durations[destination] = node_duration
            times[destination] = node_time
            points[destination] = (1 - node_fraction[:, None]) * position[
                src_left
            ] + node_fraction[:, None] * position[src_right]

        node_start = 0
        for block in source_blocks():
            (
                start,
                stop,
                src_left,
                src_right,
                width,
                a,
                b,
                parts,
                has_crossing,
                counts,
                boundaries,
            ) = prepare_block(block)
            row_offsets = np.r_[0, np.cumsum(counts)]
            ordinary = np.flatnonzero(~has_crossing)
            ordinary_offsets = np.r_[0, np.cumsum(parts[ordinary])]
            for begin in range(0, ordinary_offsets[-1], _QUADRATURE_FILL_PARTS):
                subparts = np.arange(
                    begin, min(begin + _QUADRATURE_FILL_PARTS, ordinary_offsets[-1])
                )
                row_index = (
                    np.searchsorted(ordinary_offsets, subparts, side="right") - 1
                )
                row = ordinary[row_index]
                subpart = subparts - ordinary_offsets[row_index]
                local = (subpart[:, None] + 0.5 * (1 + _QUADRATURE_NODES)) / parts[
                    row, None
                ]
                f = a[row, None] + (b[row] - a[row])[:, None] * local
                moving = width[row] > 0
                node_time = np.where(
                    moving[:, None],
                    self.time[src_left[row], None] + width[row, None] * f,
                    start[row, None] + (stop[row] - start[row])[:, None] * local,
                )
                f = np.where(moving[:, None], f, 0.0)
                node_duration = (
                    (stop[row] - start[row])[:, None]
                    / (2 * parts[row, None])
                    * _QUADRATURE_WEIGHTS
                )
                destination = (
                    node_start
                    + 4 * (row_offsets[row] + subpart)[:, None]
                    + np.arange(4)
                ).ravel()
                fill(
                    destination,
                    np.repeat(src_left[row], 4),
                    np.repeat(src_right[row], 4),
                    f.ravel(),
                    node_duration.ravel(),
                    node_time.ravel(),
                )
            for row in np.flatnonzero(has_crossing):
                cuts = boundaries(row)
                for begin in range(0, len(cuts) - 1, _QUADRATURE_FILL_PARTS):
                    lo = cuts[
                        begin : min(begin + _QUADRATURE_FILL_PARTS, len(cuts) - 1)
                    ]
                    hi = cuts[begin + 1 : begin + 1 + len(lo)]
                    half_width = (hi - lo) / 2
                    f = lo[:, None] + half_width[:, None] * (1 + _QUADRATURE_NODES)
                    node_duration = (
                        width[row] * half_width[:, None] * _QUADRATURE_WEIGHTS
                    )
                    node_time = self.time[src_left[row]] + width[row] * f
                    destination = (
                        node_start + 4 * (row_offsets[row] + begin) + np.arange(f.size)
                    )
                    fill(
                        destination,
                        np.full(f.size, src_left[row]),
                        np.full(f.size, src_right[row]),
                        f.ravel(),
                        node_duration.ravel(),
                        node_time.ravel(),
                    )
            node_start += 4 * int(np.sum(counts))
        result = ExposureQuadrature(times, points, durations, left, right, fraction)
        for value in vars(result).values():
            value.flags.writeable = False
        self._quadrature_cache = (
            (key, result) if result.nbytes <= _QUADRATURE_CACHE_BYTES else None
        )
        return result


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
