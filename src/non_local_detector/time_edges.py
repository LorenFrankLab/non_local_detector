"""Decode time edges: validation, uniformity, and grid construction.

Decoding bins are given by ``time_edges`` of shape ``(n_bins + 1,)``. Bin ``i``
covers ``[time_edges[i], time_edges[i + 1])``, except the final bin, which also
contains ``time_edges[-1]``. The bin centers are the observation coordinates
(position interpolation, local kernels, result coordinates) and
``np.diff(time_edges)`` are the bin durations.

Precision tolerances are stated in units in the last place (ulp) of the
largest edge magnitude. Correctly built float64 grids (``t0 + i * dt``,
``np.arange``, ``np.linspace``, cumulative sums, or edges derived from uniform
centers) keep every spacing within 1.7 ulp of the mean width at origins from 0
to Unix-epoch seconds. Grids can also carry rounding inherited from a larger
scale or a coarser dtype (epoch edges shifted to start at 0, or float32 edges
cast to float64); that stays well below a thousandth of a bin. The uniformity guard therefore allows the larger of
``_SPACING_TOLERANCE_ULPS`` ulp and ``_UNIFORMITY_RELATIVE_FLOOR`` of the bin
width, and rejects deviations above that larger bound. The separate precision
guard rejects grids whose ulp bound exceeds 1% of their smallest bin width.
"""

from dataclasses import dataclass
from functools import wraps

import numpy as np

from non_local_detector.exceptions import DataError, ValidationError

# Allowed deviation of a bin width from the mean width, in ulp of max|edge|.
_SPACING_TOLERANCE_ULPS = 4
# Edges whose spacing tolerance exceeds this fraction of a bin width cannot
# resolve the bins they describe.
_MAX_RELATIVE_SPACING_TOLERANCE = 1e-2
# Deviation from the mean width always allowed by the uniformity guard, as a
# fraction of the width: rounding inherited from a larger scale or dtype.
_UNIFORMITY_RELATIVE_FLOOR = 1e-3


def requires_time_edges(function):
    """Keep the explicit-edge signature and explain obsolete time arguments.

    This is an error adapter, not a compatibility conversion: old inputs are
    never interpreted as bin boundaries automatically.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        if "time" in kwargs or "time_edges" not in kwargs:
            raise TypeError(
                f"{function.__qualname__} requires keyword-only time_edges "
                "(n_bins + 1 bin boundaries). For uniformly spaced sample "
                "timestamps, use time_edges=time_edges_from_centers(time). "
                "This preserves row count but changes between-sample spike "
                "ownership. See docs/time_grid_migration.md for irregular "
                "tracking, masks, and independent sequences."
            )
        return function(*args, **kwargs)

    return wrapped


def _spacing_tolerance(time_edges: np.ndarray) -> float:
    """Timestamp representation tolerance of ``time_edges``, in seconds."""
    return float(_SPACING_TOLERANCE_ULPS * np.spacing(np.max(np.abs(time_edges))))


def _uniformity_tolerance(time_edges: np.ndarray, width: float) -> float:
    """Allowed deviation of a bin width from ``width``, in seconds."""
    return max(_spacing_tolerance(time_edges), _UNIFORMITY_RELATIVE_FLOOR * width)


def _validate_time_precision(
    time_edges: np.ndarray, smallest: float, name: str = "time_edges"
) -> None:
    """Require timestamps to resolve a requested width under the existing bound."""
    with np.errstate(over="ignore", invalid="ignore"):
        tolerance = _spacing_tolerance(time_edges)
    if not np.isfinite(tolerance) or (
        tolerance > _MAX_RELATIVE_SPACING_TOLERANCE * smallest
    ):
        raise DataError(
            f"{name} has insufficient precision for its bin widths",
            data_name=name,
            hint=f"dtype {time_edges.dtype} resolves timestamps near "
            f"{np.max(np.abs(time_edges)):.6g} only to about {tolerance:.3g} s, which "
            f"is more than {_MAX_RELATIVE_SPACING_TOLERANCE:.0%} of a "
            f"{smallest:.3g} s bin. Use float64 timestamps, or subtract a "
            "reference time before converting to a lower precision.",
        )


def validate_time_edges(time_edges, name: str = "time_edges") -> np.ndarray:
    """Validate decode bin edges and return them as a float array.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
        Bin boundaries in seconds. Any real numeric array-like is accepted;
        integers are converted to float64 and floating-point edges keep their
        dtype, so later checks judge them at the precision they carry.
    name : str, optional
        Name used in error messages, by default "time_edges".

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)
        The edges, as a floating-point array.

    Raises
    ------
    ValidationError
        If the edges are not a one-dimensional real numeric array with at least
        two entries.
    DataError
        If any edge is not finite, the edges are not strictly increasing, or
        their durations/span are not finite or their dtype cannot resolve the
        bin widths.
    """
    edges = np.asarray(time_edges)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValidationError(
            f"{name} must be a one-dimensional array of at least two bin edges",
            expected="array with shape (n_bins + 1,), n_bins >= 1",
            got=f"array with shape {edges.shape}",
            hint="Pass bin boundaries, not one timestamp per bin; see "
            "time_edges_from_centers to build edges from uniform sample times.",
            example=f"    {name} = np.arange(n_bins + 1) * 0.002 + t0",
        )
    if not (
        np.issubdtype(edges.dtype, np.integer)
        or np.issubdtype(edges.dtype, np.floating)
    ):
        raise ValidationError(
            f"{name} must contain real numbers",
            expected="an integer or floating-point array",
            got=f"dtype {edges.dtype}",
            hint="Convert timestamps to seconds as float64.",
        )
    if not np.all(np.isfinite(edges)):
        raise DataError(
            f"Found non-finite values in {name}",
            data_name=name,
            hint="Every bin edge must be a finite timestamp.",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        widths = np.diff(edges.astype(np.result_type(edges.dtype, np.float64)))
    span = float(edges[-1]) - float(edges[0])
    if not np.all(np.isfinite(widths)) or not np.isfinite(span):
        raise DataError(
            f"{name} must describe finite bin durations and a finite span",
            data_name=name,
            hint="Subtract a reference time or express timestamps in seconds "
            "at a scale whose differences remain finite.",
        )
    smallest = float(np.min(widths)) if np.all(widths > 0) else span / widths.size
    if span > 0:
        _validate_time_precision(edges, smallest, name)
    bad = np.flatnonzero(widths <= 0)
    if bad.size > 0:
        i = int(bad[0])
        raise DataError(
            f"{name} must be strictly increasing",
            data_name=name,
            hint=f"{name}[{i}] = {edges[i]!r} and {name}[{i + 1}] = "
            f"{edges[i + 1]!r}. Repeated edges give zero-width bins and "
            "decreasing edges misassign spikes.",
        )
    if np.issubdtype(edges.dtype, np.integer):
        return edges.astype(np.float64)
    return edges


@dataclass(frozen=True, init=False)
class _DecodeTimeGrid:
    """Validated edges shared only within one likelihood prediction.

    The caller must keep the edges unchanged while this context is in use.
    A new public prediction or standalone likelihood call validates its inputs
    again; this object must never be retained on a fitted detector.
    """

    edges: np.ndarray
    uniform_width: float | None
    uniform_tolerance: float | None

    def __init__(self, edges) -> None:
        object.__setattr__(self, "edges", validate_time_edges(edges))
        object.__setattr__(self, "uniform_width", None)
        object.__setattr__(self, "uniform_tolerance", None)

    @classmethod
    def from_validated_edges(
        cls, edges: np.ndarray, *, uniform_width: float | None = None
    ) -> "_DecodeTimeGrid":
        """Bind edges already validated at the private model boundary.

        ``uniform_width`` is supplied only after the detector's uniform-grid
        check. Raw contexts leave it unset so they cannot bypass that check.
        """
        result = object.__new__(cls)
        object.__setattr__(result, "edges", edges)
        object.__setattr__(result, "uniform_width", uniform_width)
        object.__setattr__(
            result,
            "uniform_tolerance",
            None
            if uniform_width is None
            else _uniformity_tolerance(edges, uniform_width),
        )
        return result

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def durations(self, row_start: int, row_stop: int) -> np.ndarray:
        """Allocate durations only for the requested, normalized row range."""
        return np.diff(self.edges[row_start : row_stop + 1])


def _resolve_time_grid(
    time_edges, time_grid: _DecodeTimeGrid | None = None
) -> _DecodeTimeGrid:
    """Reuse matching private preparation; validate fresh inputs otherwise."""
    if isinstance(time_grid, _DecodeTimeGrid) and time_grid.edges is time_edges:
        return time_grid
    return _DecodeTimeGrid(time_edges)


def uniform_time_bin_width(time_edges, name: str = "time_edges") -> float:
    """Validate that decode bins are uniform and return their width.

    See :func:`uniform_time_edges`, which also returns the validated edges.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
    name : str, optional
        Name used in error messages, by default "time_edges".

    Returns
    -------
    width : float
        Bin width in the units of ``time_edges`` (seconds).

    Raises
    ------
    ValidationError, DataError
        From :func:`uniform_time_edges`.
    """
    return uniform_time_edges(time_edges, name)[1]


def uniform_time_edges(
    time_edges, name: str = "time_edges"
) -> tuple[np.ndarray, float]:
    """Validate uniform decode bins in one pass; return the edges and width.

    The width is inferred from the edges as ``(edges[-1] - edges[0]) / n_bins``
    and every bin must match it within the larger of the timestamp
    representation tolerance and a thousandth of the width.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
    name : str, optional
        Name used in error messages, by default "time_edges".

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)
        The edges as a floating-point array (see :func:`validate_time_edges`).
    width : float
        Bin width in the units of ``time_edges`` (seconds).

    Raises
    ------
    ValidationError, DataError
        From :func:`validate_time_edges`, or `DataError` if any bin width
        differs from the mean by more than that tolerance.
    """
    edges = validate_time_edges(time_edges, name)
    n_bins = edges.shape[0] - 1
    width = (float(edges[-1]) - float(edges[0])) / n_bins
    deviation = np.abs(np.diff(edges) - width)
    tolerance = _uniformity_tolerance(edges, width)
    if np.any(deviation > tolerance):
        i = int(np.argmax(deviation))
        raise DataError(
            f"{name} must describe uniform bins for this detector",
            data_name=name,
            hint=f"Bin {i} is {edges[i + 1] - edges[i]:.9g} s wide but the mean "
            f"width is {width:.9g} s (tolerance {tolerance:.3g} s). The HMM "
            "applies one state transition per bin regardless of its duration, "
            "so nonuniform bins miscalibrate the posterior. Decode each "
            "uniformly sampled interval separately, or build a uniform grid "
            "with calculate_time_edges.",
        )
    return edges, float(width)


def time_edges_from_centers(time) -> np.ndarray:
    """Edges of uniform bins centered on uniformly spaced timestamps.

    This is the migration for code that decoded one row per position sample:
    each sample ``t[i]`` becomes the center of the bin
    ``[t[i] - dt / 2, t[i] + dt / 2)``, with ``dt`` inferred from the samples.

    Parameters
    ----------
    time : array_like, shape (n_bins,)
        Uniformly spaced bin centers, at least two.

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)

    Raises
    ------
    ValidationError
        If fewer than two centers are given (a single timestamp does not
        determine a bin duration) or the input is not one-dimensional.
    DataError
        If the centers are not finite, not strictly increasing, or not
        uniformly spaced.
    """
    centers = np.asarray(time)
    if centers.ndim == 1 and centers.shape[0] < 2:
        raise ValidationError(
            "time_edges_from_centers needs at least two timestamps",
            expected="array with shape (n_bins,), n_bins >= 2",
            got=f"array with shape {centers.shape}",
            hint="A single timestamp does not determine a bin duration; pass "
            "explicit edges such as np.array([t - dt / 2, t + dt / 2]).",
        )
    centers = validate_time_edges(centers, name="time")
    width = uniform_time_bin_width(centers, name="time")
    edges = np.concatenate(
        [
            [centers[0] - width / 2],
            centers[:-1] + 0.5 * np.diff(centers),
            [centers[-1] + width / 2],
        ]
    )
    # An outer half-cell can cross a floating-point exponent boundary. Judge
    # the resulting edges at their own precision, rather than only the centers.
    return uniform_time_edges(edges)[0]


def calculate_time_edges(
    time_range, sampling_frequency: float, trim: bool = False
) -> np.ndarray:
    """Uniform edges at ``1 / sampling_frequency`` spanning ``time_range``.

    Parameters
    ----------
    time_range : array_like, shape (2,)
        ``(start, stop)`` in seconds.
    sampling_frequency : float
        Bins per second.
    trim : bool, optional
        If False (default), the range must be a whole number of bins (within
        timestamp precision). If True, a partial final bin is dropped, so the
        last edge may fall before ``stop``.

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)

    Raises
    ------
    ValidationError
        If ``sampling_frequency`` is not positive and finite, ``time_range`` is
        not ``(start, stop)``, the range is not a whole number of bins and
        ``trim`` is False, or it contains no complete bin.
    DataError
        If ``start`` or ``stop`` is not finite, ``stop <= start``, or timestamp
        precision cannot resolve the requested width. Generated edges are
        validated by the same uniformity guard as detector inputs.
    """
    if not (np.isfinite(sampling_frequency) and sampling_frequency > 0):
        raise ValidationError(
            "sampling_frequency must be a positive finite number",
            expected="bins per second > 0",
            got=repr(sampling_frequency),
        )
    if np.shape(time_range) != (2,):
        raise ValidationError(
            "time_range must be (start, stop)",
            expected="array with shape (2,)",
            got=f"array with shape {np.shape(time_range)}",
        )
    start, stop = validate_time_edges(time_range, name="time_range")
    # The range can be resolvable while the requested bins are not. Generated
    # edges are float64; reject an impossible width before allocating them.
    _validate_time_precision(
        np.asarray([start, stop], dtype=np.float64), 1.0 / sampling_frequency
    )
    n_float = (stop - start) * sampling_frequency
    # Precision of the span in bins: the endpoints' representation plus the
    # rounding of the product.
    tolerance = _SPACING_TOLERANCE_ULPS * (
        float(np.spacing(max(abs(start), abs(stop)))) * sampling_frequency
        + float(np.finfo(np.float64).eps) * n_float
    )
    n_bins = int(np.round(n_float))
    if abs(n_float - n_bins) > tolerance:
        if not trim:
            raise ValidationError(
                "time_range is not a whole number of bins",
                expected=f"a duration that is a multiple of 1/{sampling_frequency} s",
                got=f"{stop - start!r} s = {n_float:.6f} bins",
                hint="Pass trim=True to drop the partial final bin, or adjust "
                "the range; a shorter final bin is never appended.",
            )
        n_bins = int(np.floor(n_float))
    if n_bins < 1:
        raise ValidationError(
            "time_range contains no complete bin",
            expected=f"a duration of at least 1/{sampling_frequency} s",
            got=f"{stop - start!r} s",
        )
    edges: np.ndarray = start + np.arange(n_bins + 1) / sampling_frequency
    return uniform_time_edges(edges)[0]
