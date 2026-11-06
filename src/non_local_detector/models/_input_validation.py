"""Input validation of the detectors.

The checks the detectors run on their constructor parameters and on the data
of their public methods: the one input step every public method takes
(``_InputValidationMixin._validated_inputs``, giving ``_ValidatedInputs``),
the decode time grid, the prediction and transition options,
``return_outputs``, and the fitted model's time contract. ``_DetectorBase`` inherits the methods of
``_InputValidationMixin``.

Methods that read the detector's attributes annotate ``self`` as
``_DetectorBase``, which mypy only accepts with ``type: ignore[misc]`` on a
mixin.
"""

import contextlib
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import non_local_detector
from non_local_detector import _validation as val
from non_local_detector._warnings import user_stacklevel
from non_local_detector.discrete_state_transitions import (
    DiscreteTransition,
    _refuse_untrusted_transition,
)
from non_local_detector.encoding_time import (
    EncodingSupport,
    parse_tracking_intervals,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import (
    DataError,
    DataQualityWarning,
    NotFittedError,
    ValidationError,
)
from non_local_detector.likelihoods.common import (
    resolve_row_slice,
    validate_spike_feature_population,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.time_edges import (
    _MAX_RELATIVE_SPACING_TOLERANCE,
    _DecodeTimeGrid,
    _uniformity_tolerance,
    decode_bin_centers,
    uniform_time_edges,
)
from non_local_detector.transition_operators import DEFAULT_MAX_DENSE_BYTES
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    FrozenTransitionRows,
    InferenceMode,
    OutputMode,
    ReturnOutputs,
    Stickiness,
    TransitionRepresentation,
)

if TYPE_CHECKING:
    import non_local_detector.models.base


# Units of a fitted encoding model. Every fit records a copy as
# ``time_contract_``; decoding refuses a model recorded with any other.
_TIME_CONTRACT: dict[str, object] = {
    "version": 1,
    "rate_units": "Hz",
    "encoding_time_units": "seconds",
}


# The optional outputs ``return_outputs`` can name; "all" names every one.
_OPTIONAL_OUTPUTS: frozenset[str] = frozenset(
    {"filter", "predictive", "predictive_posterior", "log_likelihood"}
)


VALID_OUTPUTS: frozenset[str] = _OPTIONAL_OUTPUTS | {"all"}


def _validate_transition_arguments(
    transition_representation: TransitionRepresentation, max_dense_transition_bytes: int
) -> None:
    """Reject unsupported transition settings before any fitted state changes."""
    if transition_representation not in get_args(TransitionRepresentation):
        msg = "transition_representation must be dense, structured, or auto"
        raise ValidationError(
            msg,
            expected=f"one of {get_args(TransitionRepresentation)}",
            got=repr(transition_representation),
        )
    if (
        isinstance(max_dense_transition_bytes, (bool, np.bool_))
        or not isinstance(max_dense_transition_bytes, (int, np.integer))  # type: ignore[redundant-expr]  # runtime check of caller input
        or max_dense_transition_bytes <= 0
    ):
        msg = "max_dense_transition_bytes must be a positive integer"
        raise ValidationError(msg)


def _validate_inference_arguments(
    inference_mode: InferenceMode,
    output_mode: OutputMode,
    *,
    cache_likelihood: bool,
    n_chunks: int,
    chunk_size: int | None,
    result_path: str | Path | None,
    checkpoint_dir: str | Path | None,
    selected_intervals: np.ndarray | None,
    replay_cache_bytes: int | None = None,
) -> None:
    """Reject prediction-mode settings that the chosen inference mode ignores.

    Raises
    ------
    ValidationError
        If either mode is unknown or an option does not apply to the mode.
    """
    if inference_mode not in get_args(InferenceMode):
        msg = "inference_mode must be dense or checkpointed"
        raise ValidationError(
            msg, expected=f"one of {get_args(InferenceMode)}", got=repr(inference_mode)
        )
    if output_mode not in get_args(OutputMode):
        msg = "output_mode must be compact or spatial"
        raise ValidationError(
            msg, expected=f"one of {get_args(OutputMode)}", got=repr(output_mode)
        )
    if inference_mode == "checkpointed" and (cache_likelihood or n_chunks != 1):
        msg = "Checkpointed prediction uses chunk_size; leave cache_likelihood=False and n_chunks=1"
        raise ValidationError(
            msg,
            got=f"cache_likelihood={cache_likelihood}, n_chunks={n_chunks}",
            hint="Checkpointed prediction never caches the likelihood and sets "
            "its chunk rows with chunk_size.",
        )
    dense_ignored = {
        "output_mode": output_mode != "spatial",
        "chunk_size": chunk_size is not None,
        "result_path": result_path is not None,
        "checkpoint_dir": checkpoint_dir is not None,
        "selected_intervals": selected_intervals is not None,
        "replay_cache_bytes": replay_cache_bytes is not None,
    }
    if inference_mode == "dense" and any(dense_ignored.values()):
        msg = "output selection/storage options require inference_mode='checkpointed'"
        raise ValidationError(
            msg,
            got=", ".join(name for name, given in dense_ignored.items() if given),
            hint="Pass inference_mode='checkpointed', or drop these options for "
            "dense prediction.",
        )


def _sample_times(values: ArrayLike) -> np.ndarray:
    """``position_time`` as a 1-D numeric array, shape (n_time_position,)."""
    times = np.asarray(values)
    if times.ndim != 1 or times.dtype.kind not in "iuf":
        msg = "position_time must be a one-dimensional array of times in seconds"
        raise ValidationError(
            msg,
            expected="numeric array with shape (n_time_position,)",
            got=f"{times.dtype} array with shape {times.shape}",
        )
    return times


def _sample_positions(values: ArrayLike, n_time: int) -> np.ndarray:
    """``position`` as a numeric array, shape (n_time,) or (n_time, n_position_dims)."""
    position = np.asarray(values)
    if position.ndim not in (1, 2) or position.dtype.kind not in "iuf":
        msg = "position must be a numeric array with one row per position sample"
        raise ValidationError(
            msg,
            expected="numeric array with shape (n_time_position,) or "
            "(n_time_position, n_position_dims)",
            got=f"{position.dtype} array with shape {position.shape}",
        )
    if position.shape[0] != n_time:
        msg = "position must have one row per position_time sample"
        raise ValidationError(
            msg,
            expected=f"{n_time} rows",
            got=f"{position.shape[0]} rows",
            hint="position and position_time describe the same samples.",
        )
    return position


def _sample_mask(
    values: ArrayLike,
    n_time: int,
    name: str,
    *,
    entry: str = "position sample",
    example: str | None = "speed > 4",
) -> np.ndarray:
    """Return a boolean mask with one entry per sample or bin, shape (n_time,).

    Integers equal to 0 or 1 are read as False or True. Any other integer, a
    float or another dtype is rejected rather than combined bitwise with the
    label masks, where ``2 & True`` is 0, or cast to bool, where 0.5, 2 and
    NaN are True.

    Parameters
    ----------
    values : array_like, shape (n_time,)
        The mask.
    n_time : int
        Number of entries required.
    name : str
        Argument name for the error messages.
    entry : str, optional
        What one entry describes, for the error messages, by default
        "position sample".
    example : str or None, optional
        Example expression for the mask in the error message, or None.

    Returns
    -------
    np.ndarray of bool, shape (n_time,)
        True where the entry is True or 1.

    Raises
    ------
    ValidationError
        If the mask does not have shape ``(n_time,)`` or is neither boolean nor
        integers 0 and 1.
    """
    mask = np.asarray(values)
    if mask.shape != (n_time,):
        msg = f"{name} must have one entry per {entry}"
        raise ValidationError(
            msg,
            expected=f"shape ({n_time},)",
            got=f"shape {mask.shape}",
            hint=f"A shorter or single-entry mask is not broadcast to every "
            f"{entry}; pass a 1-D array (e.g. {name}.ravel() for a column).",
        )
    if mask.dtype.kind in "iu" and np.isin(mask, (0, 1)).all():
        return mask.astype(bool)
    if mask.dtype != np.bool_:
        msg = f"{name} must be a boolean mask"
        raise ValidationError(
            msg,
            expected="dtype bool, or integers 0 and 1",
            got=f"dtype {mask.dtype}"
            + (
                f" with values {np.unique(mask)[:5].tolist()}"
                if mask.dtype.kind in "iu"
                else ""
            ),
            hint=f"Pass True/False per {entry}"
            + ("" if example is None else f", e.g. {name}={example}")
            + "; other integers and floats are not read as a mask.",
        )
    return mask


def _sample_labels(
    values: ArrayLike,
    n_time: int,
    name: str,
    *,
    kinds: str,
    contents: str,
    compared_with: str,
) -> np.ndarray:
    """Labels with one entry per position sample, shape (n_time,).

    ``kinds`` lists the accepted NumPy dtype kinds; ``contents`` and
    ``compared_with`` describe them for the error message.
    """
    labels = np.asarray(values)
    if labels.shape != (n_time,):
        msg = f"{name} must have one label per position sample"
        raise ValidationError(
            msg, expected=f"shape ({n_time},)", got=f"shape {labels.shape}"
        )
    if labels.dtype.kind not in kinds:
        msg = f"{name} must hold {contents}"
        raise ValidationError(
            msg,
            expected=contents,
            got=f"dtype {labels.dtype}",
            hint=f"Each label is compared with {compared_with}.",
        )
    return labels


def _require_labeled_environments(
    environment_labels: np.ndarray, names: Sequence[str]
) -> None:
    """Raise unless ``environment_labels`` label a sample with every environment.

    Parameters
    ----------
    environment_labels : np.ndarray, shape (n_time_position,)
        Label of each position sample.
    names : sequence of str
        The ``environment_name`` of each environment.

    Raises
    ------
    ValidationError
        If an environment labels no sample.
    """
    missing = [name for name in names if not np.any(environment_labels == name)]
    if not missing:
        return
    # str first: unlabeled samples (None, NaN) do not sort with names.
    labels = sorted({str(label) for label in environment_labels})
    unmatched = [label for label in labels if label not in names]
    msg = (
        f"No position sample is labeled with environment(s) {missing} "
        "in environment_labels, so their place grids cannot be fitted"
    )
    raise ValidationError(
        msg,
        expected=f"labels naming the environments {names}",
        got=f"labels {labels[:10]}"
        + (f" (of {len(labels)})" if len(labels) > 10 else ""),
        hint=(
            f"The labels {unmatched[:10]} match no environment_name. "
            if unmatched
            else ""
        )
        + "Label each position sample with the environment_name of "
        "the environment it was recorded in.",
    )


def _validate_spike_structure(
    spike_times: Sequence[Any],
    spike_waveform_features: Sequence[Any] | None,
    *,
    clusterless: bool,
) -> None:
    """Require one 1-D spike-time array per unit, and features to match.

    Reads only the arrays' shapes, so decoding repeats it for every chunk.

    Parameters
    ----------
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times of each neuron (sorted) or electrode (clusterless).
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features), or None
        Waveform features of each electrode; None for sorted spikes.
    clusterless : bool
        Whether the spikes are clusterless (electrodes with features).

    Raises
    ------
    ValidationError
        If ``spike_times`` is a numeric array of two or more dimensions, a
        unit's spike times are not 1-D, or (clusterless) the features do not
        have one row per spike of the same electrodes.
    """
    # Iterating a numeric (n_spikes, 1) array gives n_spikes units of one
    # spike each, which every per-unit check accepts. Only array containers
    # are inspected: converting a ragged list would fail.
    container_shape = getattr(spike_times, "shape", ())
    if len(container_shape) >= 2 and getattr(spike_times, "dtype", None) != np.dtype(
        object
    ):
        unit = "electrode" if clusterless else "neuron"
        msg = f"spike_times must be a list of 1-D arrays, one per {unit}"
        raise ValidationError(
            msg,
            expected=f"a list or tuple of arrays with shape (n_spikes,), one per {unit}",
            got=f"a numeric array of shape {tuple(container_shape)}",
            hint=f"Pass [times.ravel()] for one {unit}'s spike times, or "
            f"list(times) if each row is one {unit}'s spike times.",
        )
    if clusterless:
        validate_spike_feature_population(
            spike_times,
            spike_waveform_features,  # type: ignore[arg-type]  # a required argument of the clusterless methods
        )
        return
    for neuron, times in enumerate(spike_times):
        if len(np.shape(times)) != 1:
            msg = f"spike_times for neuron {neuron} must be 1-D"
            raise ValidationError(
                msg,
                expected="one array with shape (n_spikes,) per neuron",
                got=f"shape {np.shape(times)}",
                hint="Pass a list with one 1-D array of spike times (seconds) "
                "per neuron, e.g. [spike_times] for a single neuron.",
            )


def _numeric_spike_times(times: Any, unit: str, index: int) -> np.ndarray:
    """One unit's spike times as a numeric array, shape (n_spikes,).

    An object array of numbers (a row of ``np.array(spike_times,
    dtype=object)`` for equal-length units, or ``times.astype(object)``) is
    converted to float, as the spike binning reads it.

    Raises
    ------
    ValidationError
        If the times are not numbers.
    """
    array = np.asarray(times)
    if array.dtype == np.dtype(object):
        with contextlib.suppress(TypeError, ValueError):
            array = array.astype(float)
    if array.dtype.kind not in "iuf":
        msg = f"spike_times for {unit} {index} must be numeric"
        raise ValidationError(
            msg,
            expected="spike times in seconds",
            got=f"dtype {np.asarray(times).dtype}",
        )
    return array


# Spike times in ms against a position_time span in s that starts near 0
# overlap the span, so they are not far from it; instead almost none lie
# inside it (~0.1%) and they reach ~1000 spans past it. Spikes of a whole
# recording with position for part of it (one epoch of a day) keep far more
# than 1% inside and reach only a few spans past it. position_time in ms
# against spikes in s is the mirror image: all spikes overlap its start, and
# they span ~0.1% of it, where one epoch of a day spans several percent.
_MIN_FRACTION_INSIDE = 0.01
_MAX_SPANS_PAST = 10.0
_MIN_FRACTION_OF_SPAN = 0.01


def _warn_about_encoding_extent(
    arrays: list[np.ndarray], start: float, stop: float, first: float, last: float
) -> None:
    """Warn if the spikes and ``[start, stop]`` look like different units.

    Warns if under 1% of the spikes lie in the range and they reach over 10
    of its lengths past it, if none lies in it, or if they span under 1% of it.
    """
    span = stop - start
    if span <= 0:
        return
    n_spikes = sum(times.size for times in arrays)
    n_inside = sum(
        int(np.count_nonzero((times >= start) & (times <= stop))) for times in arrays
    )
    reach = max(last - stop, start - first)
    if n_inside < _MIN_FRACTION_INSIDE * n_spikes and reach > _MAX_SPANS_PAST * span:
        warnings.warn(
            f"Only {n_inside} of the {n_spikes} spike times "
            f"({n_inside / n_spikes:.2%}) lie within the position_time span "
            f"[{start:.6g}, {stop:.6g}] s, and they reach {reach / span:.3g} times "
            f"its length past it (spanning [{first:.6g}, {last:.6g}]). The spike "
            "times are probably in milliseconds (or other units) rather than "
            "seconds; the encoding model is fit to the few that fall inside. "
            "Convert them to seconds, or, if the position covers only a small part "
            "of the recording on purpose, pass the spikes of that part.",
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )
    elif n_inside == 0:
        warnings.warn(
            f"None of the {n_spikes} spike times lies inside the position_time span "
            f"[{start:.6g}, {stop:.6g}] s, so every encoding model is fit without "
            "spikes and its rates sit at their floor. Check that spike_times "
            "cover the tracked period.",
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )
    elif 0 < last - first < _MIN_FRACTION_OF_SPAN * span:
        warnings.warn(
            f"The spike times span {last - first:.3g} s, {(last - first) / span:.2%} "
            f"of the position_time span [{start:.6g}, {stop:.6g}] s. position_time "
            "is probably in milliseconds (or other units) rather than seconds, "
            "which also shrinks every firing rate; the encoding model counts the "
            "whole span as occupancy. Convert position_time to seconds, or, if "
            "the spikes cover only a small part of the tracking on purpose, pass "
            "the position of that part.",
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )


def _warn_if_times_cannot_resolve_bins(
    arrays: list[np.ndarray], bin_width: float
) -> bool:
    """Warn if a spike-time dtype is coarser than a decode bin near its values.

    Spikes then round onto a few representable times, away from the bins they
    fell in. Only ``time_edges`` have a dtype check of their own.

    Parameters
    ----------
    arrays : list of np.ndarray, each shape (n_spikes,)
        Finite spike times of each unit.
    bin_width : float
        Decode bin width, in seconds.

    Returns
    -------
    warned : bool
    """
    coarsest = max(
        (
            (float(np.spacing(np.max(np.abs(times)))), times)
            for times in arrays
            if times.size and times.dtype.kind == "f"
        ),
        key=lambda spacing_times: spacing_times[0],
        default=(0.0, None),
    )
    spacing, times = coarsest
    if times is None or spacing <= bin_width:
        return False
    warnings.warn(
        f"spike_times of dtype {times.dtype} resolve times near "
        f"{np.max(np.abs(times)):.6g} s only to {spacing:.3g} s, more than the "
        f"{bin_width:.3g} s decode bin, so spikes round onto a few values and "
        "away from the bins they fell in. Use float64 timestamps, or subtract a "
        "reference time before converting to a lower precision.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )
    return True


def _numeric_spike_times_list(
    spike_times: Sequence[Any], *, clusterless: bool
) -> list[np.ndarray]:
    """Every unit's spike times as a numeric array, shape (n_spikes,).

    ``_validate_spike_data`` checks these arrays, and fitting reads them, so
    object-dtype spike times are float in both, as in decoding.

    Raises
    ------
    ValidationError
        If any unit's times are not numbers.
    """
    unit = "electrode" if clusterless else "neuron"
    return [
        _numeric_spike_times(times, unit, index)
        for index, times in enumerate(spike_times)
    ]


def _validate_spike_data(
    spike_times: Sequence[Any],
    spike_waveform_features: Sequence[Any] | None = None,
    *,
    clusterless: bool,
    time_range: tuple[float, float] | None = None,
    range_name: str = "",
    check_extent: bool = False,
    bin_width: float | None = None,
) -> list[np.ndarray]:
    """Check spike data where it enters a public fitting or decoding method.

    Checks the structure (``_validate_spike_structure``), then that every
    spike time is a finite number. With ``time_range``, warns when the span
    of all the spike times, ``[min, max]`` over every unit, is disjoint from
    it and farther from it than ``max - min``: the spike times are then most
    likely in other units (ms rather than s) or on another clock than the
    times they are compared with. A range without spikes next to them (a
    quiet window before, between or after the spikes) and a unit whose spikes
    all lie outside it are not reported.

    Parameters
    ----------
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times of each neuron or electrode, in seconds.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features), or None, optional
        Waveform features of each electrode; None for sorted spikes.
    clusterless : bool
        Whether the spikes are clusterless (electrodes with features).
    time_range : tuple of float, optional
        ``(start, stop)`` in seconds the spikes are expected in, by default
        None (no warning).
    range_name : str, optional
        What ``time_range`` is, for the warning, e.g. "decode time_edges".
    check_extent : bool, optional
        Also warn when the spikes overlap ``time_range`` but look like other
        units than it (see ``_warn_about_encoding_extent``), by default False.
        For the encoding period only: a decode window is legitimately short
        against the recording its spikes come from.
    bin_width : float or None, optional
        Decode bin width in seconds the spike times' dtype must resolve, by
        default None (not checked). Times it cannot resolve warn, and the
        ``time_range`` warnings, which would read the rounded times, are
        skipped.

    Raises
    ------
    ValidationError
        If the structure is wrong or spike times are not numeric.
    DataError
        If any spike time is NaN or infinite.

    Warns
    -----
    DataQualityWarning
        If the spikes lie farther from ``time_range`` than they span, with
        ``check_extent`` look like other units than it, or with ``bin_width``
        have a dtype too coarse for the bins.

    Returns
    -------
    spike_times : list of np.ndarray
        Every unit's spike times as a numeric array, shape (n_spikes,).
    """
    _validate_spike_structure(
        spike_times, spike_waveform_features, clusterless=clusterless
    )
    unit = "electrode" if clusterless else "neuron"
    arrays = _numeric_spike_times_list(spike_times, clusterless=clusterless)
    n_non_finite = [int(np.count_nonzero(~np.isfinite(times))) for times in arrays]
    if any(n_non_finite):
        bad = [
            f"{unit} {index}: {count} of {arrays[index].size}"
            for index, count in enumerate(n_non_finite)
            if count
        ]
        msg = "spike_times contain NaN or infinite values"
        raise DataError(
            msg,
            data_name="spike_times",
            expected="finite spike times in seconds",
            got=", ".join(bad[:5])
            + (f" and {len(bad) - 5} more" if len(bad) > 5 else ""),
            hint="Non-finite spike times are not dropped: remove them (e.g. "
            "times[np.isfinite(times)], with the matching feature rows) or "
            "fix the spike export that produced them.",
        )
    if bin_width is not None and _warn_if_times_cannot_resolve_bins(arrays, bin_width):
        return arrays
    if time_range is None:
        return arrays
    start, stop = time_range
    n_spikes = sum(times.size for times in arrays)
    if n_spikes == 0:
        return arrays
    first = min(times.min() for times in arrays if times.size)
    last = max(times.max() for times in arrays if times.size)
    # A unit or clock mismatch (ms for s ~1000x, or spike times from 0
    # against Unix-epoch edges) puts the spikes farther from the range than
    # they span; a quiet window before, between or after the spikes is data.
    gap = max(first - stop, start - last)
    if gap <= last - first:
        if check_extent:
            _warn_about_encoding_extent(arrays, start, stop, first, last)
        return arrays
    warnings.warn(
        f"None of the {n_spikes} spike times lies within the {range_name} "
        f"[{start:.6g}, {stop:.6g}] s; they span [{first:.6g}, {last:.6g}]. "
        "The spike times are probably in other units (e.g. ms instead of s) "
        f"or on another clock than the {range_name}, so no spike is used. "
        "Convert them to seconds on the same clock.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )
    return arrays


_InputStage = Literal["fit", "encoding", "refit", "decode", "likelihood", "estimate"]


@dataclass(frozen=True)
class _ValidatedInputs:
    """Data of one public call, coerced and checked once.

    Built by ``_InputValidationMixin._validated_inputs``. An array the call's
    stage does not take stays None.

    Attributes
    ----------
    spike_data : tuple
        ``spike_times``, then ``spike_waveform_features`` when clusterless.
        ``spike_times`` are numeric arrays except when decoding, whose
        binning reads them as they are.
    position_time : np.ndarray, shape (n_time_position,), or None
    position : np.ndarray, shape (n_time_position, n_position_dims), or None
    is_training : np.ndarray of bool, shape (n_time_position,), or None
    encoding_group_labels : np.ndarray of int or str, shape (n_time_position,), or None
    environment_labels : np.ndarray of str, shape (n_time_position,), or None
    weights : np.ndarray of float, shape (n_time_position,), or None
    encoding_support : EncodingSupport or None
        Support of an encoding fit.
    time_edges : np.ndarray, shape (n_bins + 1,), or None
        Validated decode bin edges.
    time_centers : np.ndarray, shape (n_rows,), or None
        Centers of the requested decode bins.
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        The caller's mask combined with the bins without tracked position;
        those bins alone at ``likelihood``, whose mask covers the requested
        rows.
    """

    spike_data: tuple[Any, ...]
    position_time: np.ndarray | None = None
    position: np.ndarray | None = None
    is_training: np.ndarray | None = None
    encoding_group_labels: np.ndarray | None = None
    environment_labels: np.ndarray | None = None
    weights: np.ndarray | None = None
    encoding_support: EncodingSupport | None = None
    time_edges: np.ndarray | None = None
    time_centers: np.ndarray | None = None
    is_missing: np.ndarray | None = None

    @classmethod
    def from_encoding_arguments(
        cls,
        spike_data: tuple[Any, ...],
        position_time: ArrayLike,
        position: ArrayLike | None,
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> "_ValidatedInputs":
        """Coerce and check the arguments of ``fit``-like methods.

        Optional arrays stay None, so each fitting stage keeps its own
        default. ``position`` stays as the caller gave it, so errors report
        the caller's shape.

        Raises
        ------
        ValidationError
            If ``position`` is missing, an array has the wrong number of
            dimensions or samples, ``is_training`` is not a boolean or 0/1
            mask, a label array has the wrong dtype, or ``weights`` are not
            finite and non-negative.
        """
        if position is None:
            msg = "Missing required parameter: position"
            raise ValidationError(
                msg,
                expected="position array with shape (n_time, n_dims)",
                got="None",
                hint="Provide the animal's position data during training",
                example="    detector.fit(position_time, position, spike_times)",
            )
        times = _sample_times(position_time)
        n_time = len(times)
        return cls(
            spike_data,
            position_time=times,
            position=_sample_positions(position, n_time),
            is_training=None
            if is_training is None
            else _sample_mask(is_training, n_time, "is_training"),
            encoding_group_labels=None
            if encoding_group_labels is None
            else _sample_labels(
                encoding_group_labels,
                n_time,
                "encoding_group_labels",
                kinds="iuUSO",
                contents="integer or string encoding groups",
                compared_with="each observation model's encoding_group",
            ),
            environment_labels=None
            if environment_labels is None
            else _sample_labels(
                environment_labels,
                n_time,
                "environment_labels",
                kinds="USO",
                contents="environment names (strings)",
                compared_with="each environment's environment_name",
            ),
            weights=None if weights is None else val.validate_weights(weights, n_time),
        )

    @classmethod
    def from_decode_arguments(
        cls,
        spike_data: tuple[Any, ...],
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        *,
        example: str | None = None,
    ) -> "_ValidatedInputs":
        """Coerce and check the position arguments of a decoding method.

        Raises
        ------
        ValidationError
            If ``position`` is given without ``position_time``, or either
            array has the wrong number of dimensions or samples.
        """
        if position is not None and position_time is None:
            msg = "position_time is required when position is provided"
            raise ValidationError(
                msg,
                expected="position_time array with shape (n_time_position,)",
                got="None",
                hint="Provide position_time corresponding to the position samples",
                example=example,
            )
        times = None if position_time is None else _sample_times(position_time)
        if position is None or times is None:
            return cls(spike_data, times)
        position = _sample_positions(position, len(times))
        return cls(
            spike_data,
            times,
            position[:, np.newaxis] if position.ndim == 1 else position,
        )


def _require_sampling_frequency_width(
    time_edges: np.ndarray,
    width: float,
    sampling_frequency: float,
    *,
    tolerance: float | None = None,
) -> None:
    """Require uniform ``time_edges`` to be ``1 / sampling_frequency`` wide.

    The widths match within the tolerance that decides the edges are uniform
    (``time_edges._uniformity_tolerance``: the larger of 4 ulp of the largest
    edge and a thousandth of the width), so a grid accepted as uniform at a
    rate is also accepted as that rate.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Edges already checked uniform.
    width : float
        Their bin width, in seconds.
    sampling_frequency : float
        The detector's bins per second.
    tolerance : float or None, optional
        The uniformity tolerance already computed for these edges, or None to
        compute it.

    Raises
    ------
    ValidationError
        If the width differs from ``1 / sampling_frequency`` by more than the
        tolerance.
    """
    expected = 1.0 / sampling_frequency
    if tolerance is None:
        tolerance = _uniformity_tolerance(time_edges, width)
    if abs(width - expected) <= tolerance:
        return
    msg = "time_edges bin width differs from the detector's sampling_frequency"
    raise ValidationError(
        msg,
        expected=f"bin width 1 / sampling_frequency = {expected!r} s "
        f"(sampling_frequency={sampling_frequency!r} Hz)",
        got=f"bin width {width!r} s ({1.0 / width:.6g} Hz)",
        hint="The detector's transitions are per bin of 1 / sampling_frequency "
        f"seconds. Construct it with sampling_frequency={1.0 / width:.6g} to "
        "decode these edges (a fitted or loaded detector keeps its fit with "
        f"detector.set_params(sampling_frequency={1.0 / width:.6g})), or build "
        "the edges at the detector's rate with "
        "detector.calculate_time_edges(time_range).",
    )


def _decode_time_edges(
    time_edges: ArrayLike,
    transition_time_bin_width: float | None,
    transition_time_bin_width_tolerance: float = 0.0,
    *,
    sampling_frequency: float | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
    row_slice: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a detector decode grid and derive its bin centers.

    Detector bins must be uniform because the HMM applies one transition per
    bin. When transitions were learned by ``estimate_parameters``, the grid
    must also use the bin width they were learned at, and its width must be
    ``1 / sampling_frequency`` (see ``_require_sampling_frequency_width``).

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    transition_time_bin_width : float or None
        Bin width recorded with learned transitions, or None.
    transition_time_bin_width_tolerance : float, optional
        Uniformity tolerance of the grid the width was learned on. The widths
        match when they differ by no more than the larger of the two grids'
        tolerances.
    sampling_frequency : float or None, optional
        The detector's ``sampling_frequency`` (bins per second), or None to
        skip that comparison.
    row_slice : slice or None, optional
        Rows (bins) for which to return centers; None selects all bins.

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)
        The validated edges (integers converted to float64).
    time_centers : np.ndarray, shape (n_bins,)

    Raises
    ------
    ValidationError, DataError
        If the edges are invalid or not uniform, or their width differs from
        ``transition_time_bin_width`` by more than the timestamp precision or
        from ``1 / sampling_frequency``.
    """
    prepared_uniform_grid = (
        isinstance(_time_grid, _DecodeTimeGrid)
        and _time_grid.edges is time_edges
        and _time_grid.uniform_width is not None
    )
    if prepared_uniform_grid:
        assert _time_grid is not None
        assert _time_grid.uniform_width is not None
        time_edges = _time_grid.edges  # the same array, typed as ndarray
        width = _time_grid.uniform_width
        uniform_tolerance = _time_grid.uniform_tolerance
    else:
        time_edges, width = uniform_time_edges(time_edges)
        uniform_tolerance = None
    if (
        transition_time_bin_width is not None
        and (
            not np.isfinite(transition_time_bin_width) or transition_time_bin_width <= 0
        )
    ) or (
        not np.isfinite(transition_time_bin_width_tolerance)
        or transition_time_bin_width_tolerance < 0
        or (
            transition_time_bin_width is not None
            and transition_time_bin_width_tolerance
            > _MAX_RELATIVE_SPACING_TOLERANCE * transition_time_bin_width
        )
    ):
        msg = "Saved model has invalid transition clock provenance; refit before decoding."
        raise ValidationError(
            msg,
            hint="The learned bin width must be finite and positive, with a "
            "finite nonnegative timestamp tolerance within the existing "
            "precision bound. Call fit(...) or "
            "estimate_parameters(..., time_edges=...) with the original recording.",
        )
    if transition_time_bin_width is not None and abs(
        width - transition_time_bin_width
    ) > max(
        (
            uniform_tolerance
            if uniform_tolerance is not None
            else _uniformity_tolerance(time_edges, width)
        ),
        transition_time_bin_width_tolerance,
    ):
        msg = "time_edges bin width differs from the learned transitions"
        raise ValidationError(
            msg,
            expected=f"bin width {transition_time_bin_width!r} s",
            got=f"bin width {width!r} s",
            hint="The discrete transitions were estimated per bin at that width "
            "and are not valid at another rate. Decode at the same bin width, "
            "or refit the detector for the new width.",
        )
    if sampling_frequency is not None:
        _require_sampling_frequency_width(
            time_edges, width, sampling_frequency, tolerance=uniform_tolerance
        )
    start, stop = resolve_row_slice(row_slice, len(time_edges) - 1)
    return time_edges, decode_bin_centers(time_edges, start, stop)


def _missing_bins(
    time_edges: np.ndarray,
    is_missing: np.ndarray | None,
    position_time: np.ndarray | None,
    position: np.ndarray | None,
    *,
    reads_position: bool = True,
    valid_position_intervals: np.ndarray | None = None,
) -> np.ndarray | None:
    """Combine the caller's missing mask with bins that lack tracked position.

    Missing tracking is a break in support. Likelihoods read the position by
    linear interpolation (extrapolating from the end intervals) at the bin
    centers and, for local clusterless models, at every decoding spike time.
    A read is non-finite exactly when its time lies in ``[t[k - 1], t[k + 1])``
    for a non-finite sample ``k``: at ``t[k - 1]`` the sample still enters
    with weight zero (``0 * nan``), while at ``t[k + 1]`` it no longer does.
    The span extends to infinity for a sample in an end interval. A read more
    than half a sample before ``t[0]`` or after ``t[-1]`` is an extrapolation,
    not tracking, so it is missing too; the half sample keeps bins centered on
    the tracking samples, whose outer edges lie half a sample beyond them. The
    half sample is half the median sampling interval, plus the timestamp
    uniformity tolerance. A read more than half a sample outside every declared
    tracking interval is missing by the same reasoning: samples either side of
    a gap would otherwise be interpolated across it. A bin is decoded as
    missing when any time it owns (``[edges[i], edges[i + 1])``, plus
    ``edges[-1]`` for the final bin) lies in such a span, so the center and
    every spike of a non-missing bin have tracked, finite positions. These
    rules apply only when the decode likelihood reads position
    (``reads_position``); a model without Local states or a non-local position
    penalty decodes every bin. This concerns decoding only: encoding fits use
    position for their support.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    is_missing : np.ndarray, shape (n_bins,), optional
        Caller's boolean mask of missing bins.
    position_time : np.ndarray, shape (n_time_position,), optional
        Times of the position samples.
    position : np.ndarray, shape (n_time_position, n_position_dims), optional
        Position samples; non-finite rows mark untracked times.
    reads_position : bool, optional
        Whether the decode likelihood reads position, by default True. When
        False, position marks no bin missing; only ``is_missing`` does.
    valid_position_intervals : np.ndarray, shape (n_intervals, 2), optional
        Parsed spans of continuous tracking, or None if none were declared.

    Returns
    -------
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        None when nothing is missing and no mask was given.

    Warns
    -----
    DataQualityWarning
        If some bins extend outside the tracked period.

    Raises
    ------
    ValidationError
        If ``is_missing`` does not have one entry per bin, or ``position`` does
        not have one row per ``position_time`` sample.
    DataError
        If ``position_time`` is not finite and increasing, strictly when
        ``reads_position``.
    """
    n_bins = len(time_edges) - 1
    if is_missing is not None:
        is_missing = _sample_mask(
            is_missing, n_bins, "is_missing", entry="decode bin", example=None
        )
    if position is None or position_time is None:
        return is_missing
    position = np.asarray(position)
    position_time = np.asarray(position_time)
    val.ensure_all_finite(position_time, "position_time")
    # Strictly when the likelihood interpolates position in time, as fitting
    # requires.
    val.ensure_monotonic_increasing(
        position_time, "position_time", strict=reads_position
    )
    if not reads_position:
        return is_missing
    affected = _bins_outside_tracking(time_edges, position_time)
    if np.any(affected):
        warnings.warn(
            f"{int(affected.sum())} of {n_bins} decode bins extend more than half "
            "a position sample outside the tracked period "
            f"[{position_time[0]:.6g}, {position_time[-1]:.6g}] s. Position there "
            "could only be extrapolated from the first or last tracking interval, "
            "so these bins are decoded as missing, like bins with NaN position: "
            "their log likelihood is 0 and the posterior follows the transitions "
            "alone. Pass position covering the decode time_edges, or restrict "
            "time_edges to the tracked period.",
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )
    is_nan = ~np.all(np.isfinite(position.reshape(position.shape[0], -1)), axis=1)
    if np.any(is_nan):
        affected |= _bins_reading_nan_position(time_edges, position_time, is_nan)
    if valid_position_intervals is not None:
        affected |= _bins_outside_intervals(
            time_edges, position_time, valid_position_intervals
        )
    if not np.any(affected):
        return is_missing
    return affected if is_missing is None else is_missing | affected


def _half_sample(position_time: np.ndarray) -> float:
    """Half the median sampling interval, plus the uniformity tolerance.

    See :func:`_missing_bins` for the rule.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Finite and strictly increasing, at least two samples.

    Returns
    -------
    half_sample : float
    """
    sample_interval = float(np.median(np.diff(position_time)))
    return sample_interval / 2 + _uniformity_tolerance(position_time, sample_interval)


def _bins_outside_tracking(
    time_edges: np.ndarray, position_time: np.ndarray
) -> np.ndarray:
    """Bins owning a time more than half a sample outside the tracking.

    See :func:`_missing_bins` for the rule.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    position_time : np.ndarray, shape (n_time_position,)
        Finite and strictly increasing.

    Returns
    -------
    is_outside : np.ndarray of bool, shape (n_bins,)
    """
    n_bins = len(time_edges) - 1
    if position_time.size < 2:
        return np.zeros(n_bins, dtype=bool)
    half_sample = _half_sample(position_time)
    # Bin i owns [edges[i], edges[i + 1]) (the final bin also edges[-1]), so it
    # reaches before ``start`` iff edges[i] < start and after ``stop`` iff
    # edges[i + 1] > stop.
    start = position_time[0] - half_sample
    stop = position_time[-1] + half_sample
    return cast("np.ndarray", (time_edges[:-1] < start) | (time_edges[1:] > stop))


def _bins_reading_nan_position(
    time_edges: np.ndarray, position_time: np.ndarray, is_nan: np.ndarray
) -> np.ndarray:
    """Bins owning a time whose interpolated position is non-finite.

    See :func:`_missing_bins` for the rule.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    position_time : np.ndarray, shape (n_time_position,)
        Times of the position samples.
    is_nan : np.ndarray of bool, shape (n_time_position,)
        Samples with a non-finite position; at least one is True.

    Returns
    -------
    affected : np.ndarray of bool, shape (n_bins,)
    """
    nan_index = np.flatnonzero(is_nan)
    # Spans of consecutive NaN samples, widened to the neighbouring samples.
    # Positions before the first sample (after the last) are extrapolated from
    # the first (last) interval, so a NaN in an end interval reaches every bin
    # beyond that end.
    n_samples = position_time.size
    run_starts = nan_index[np.r_[True, np.diff(nan_index) > 1]]
    run_ends = nan_index[np.r_[np.diff(nan_index) > 1, True]]
    lower = np.where(
        run_starts > 1, position_time[np.maximum(run_starts - 1, 0)], -np.inf
    )
    upper = np.where(
        run_ends < n_samples - 2,
        position_time[np.minimum(run_ends + 1, n_samples - 1)],
        np.inf,
    )
    return _bins_meeting_spans(time_edges, lower, upper)


def _bins_outside_intervals(
    time_edges: np.ndarray, position_time: np.ndarray, intervals: np.ndarray
) -> np.ndarray:
    """Bins owning a time more than half a sample outside every interval.

    See :func:`_missing_bins` for the rule. Intervals closer than a sample
    leave no gap between them, so adjacent intervals keep the bin they share.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    position_time : np.ndarray, shape (n_time_position,)
        Finite and strictly increasing.
    intervals : np.ndarray, shape (n_intervals, 2)
        Ordered, non-overlapping spans of continuous tracking.

    Returns
    -------
    is_outside : np.ndarray of bool, shape (n_bins,)
    """
    half_sample = _half_sample(position_time) if position_time.size > 1 else 0.0
    gap_start = np.r_[-np.inf, intervals[:, 1] + half_sample]
    gap_stop = np.r_[intervals[:, 0] - half_sample, np.inf]
    is_gap = gap_start < gap_stop
    return _bins_meeting_spans(time_edges, gap_start[is_gap], gap_stop[is_gap])


def _bins_meeting_spans(
    time_edges: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Bins owning a time in any of the spans ``[lower, upper)``.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    lower, upper : np.ndarray, shape (n_spans,)
        Ends of the spans, which may be infinite.

    Returns
    -------
    affected : np.ndarray of bool, shape (n_bins,)
    """
    n_bins = len(time_edges) - 1
    # Bin i owns [edges[i], edges[i + 1]) and meets [lower, upper) iff
    # edges[i + 1] > lower and edges[i] < upper; the final bin also owns
    # edges[-1], so a read exactly there (lower == edges[-1]) marks it.
    first = np.searchsorted(time_edges[1:], lower, side="right")
    first = np.where(lower == time_edges[-1], n_bins - 1, first)
    stop = np.maximum(np.searchsorted(time_edges[:-1], upper, side="left"), first)
    change = np.zeros(n_bins + 1, dtype=int)
    np.add.at(change, first, 1)
    np.add.at(change, stop, -1)
    return np.cumsum(change[:-1]) > 0


# A step in position_time this many samples wide is a gap in the tracking.
# Narrower ones (a dropped frame, camera jitter) interpolate over no real
# movement. Fitting refuses any irregular step, but it can be told the intervals
# at no cost; a decode warning has to mean something on every call.
_TRACKING_GAP_SAMPLES = 5


def _unread_bins(
    n_bins: int,
    row_slice: slice | None,
    requested_is_missing: np.ndarray | None,
    position_is_missing: np.ndarray | None,
) -> np.ndarray:
    """Bins a ``compute_log_likelihood`` call does not read position for.

    Parameters
    ----------
    n_bins : int
        Bins of the decode grid.
    row_slice : slice or None
        Rows the call evaluates; None requests all.
    requested_is_missing : np.ndarray of bool, shape (n_rows,), or None
        The caller's mask of the requested rows.
    position_is_missing : np.ndarray of bool, shape (n_bins,), or None
        Bins without tracked position.

    Returns
    -------
    is_unread : np.ndarray of bool, shape (n_bins,)
    """
    row_start, row_stop = resolve_row_slice(row_slice, n_bins)
    is_unread = np.ones(n_bins, dtype=bool)
    is_unread[row_start:row_stop] = (
        False
        if requested_is_missing is None
        else _sample_mask(
            requested_is_missing,
            row_stop - row_start,
            "is_missing",
            entry="requested decode bin",
            example=None,
        )
    )
    return is_unread if position_is_missing is None else is_unread | position_is_missing


def _warn_about_tracking_coverage(
    time_edges: np.ndarray,
    position_time: np.ndarray,
    row_slice: slice | None,
    caller_is_missing: np.ndarray | None,
    is_missing: np.ndarray | None,
) -> None:
    """Warn about the tracking of a decode that reads position.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    position_time : np.ndarray, shape (n_time_position,)
        Finite and increasing.
    row_slice : slice or None
        Rows the call evaluates; None requests all.
    caller_is_missing : np.ndarray of bool, optional
        The caller's mask: of the requested rows with ``row_slice``.
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        Bins without tracked position, with the caller's where it covers the
        grid.
    """
    is_unread = _unread_bins(
        time_edges.size - 1, row_slice, caller_is_missing, is_missing
    )
    _warn_if_no_bin_is_tracked(
        time_edges,
        position_time,
        is_unread if row_slice is None else is_missing,
        caller_is_missing,
    )
    _warn_if_decoding_across_gaps(time_edges, position_time, is_unread)


def _warn_if_no_bin_is_tracked(
    time_edges: np.ndarray,
    position_time: np.ndarray,
    is_missing: np.ndarray | None,
    caller_is_missing: np.ndarray | None,
) -> None:
    """Warn when tracking, not the caller, leaves no bin to decode.

    The transitions' steady state looks like a decode, and only ``is_missing``
    would tell it apart. Bins outside the tracked period have their own
    warning.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    position_time : np.ndarray, shape (n_time_position,)
        Finite and increasing.
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        Bins decoded as missing.
    caller_is_missing : np.ndarray of bool, optional
        The caller's mask.

    Warns
    -----
    DataQualityWarning
        If every bin is missing, unless the caller marked every bin or all
        lie outside the tracked period.
    """
    if (
        is_missing is None
        or not np.all(is_missing)
        or (caller_is_missing is not None and np.all(caller_is_missing))
        or np.all(_bins_outside_tracking(time_edges, position_time))
    ):
        return
    warnings.warn(
        f"Every one of the {is_missing.size} decode bins is missing: over "
        "time_edges position is NaN or outside valid_position_intervals, so "
        "their log likelihood is 0 and the posterior follows the transitions "
        "alone. Pass the tracking intervals of the decoded period, in the units "
        "of position_time. To decode a period without tracking, use a model "
        "that reads no position (no Local state or non-local position penalty) "
        "and position=None.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )


def _warn_if_decoding_across_gaps(
    time_edges: np.ndarray, position_time: np.ndarray, is_unread: np.ndarray
) -> None:
    """Warn when a decode bin reads position across a tracking gap.

    Only the caller knows where tracking stopped, so these bins are not
    decoded as missing. Intervals that exclude a gap leave its bins missing;
    one that spans it, such as a session's bounds, does not.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Time bin edges of the decode grid.
    position_time : np.ndarray, shape (n_time_position,)
        Finite and strictly increasing.
    is_unread : np.ndarray of bool, shape (n_bins,)
        Bins that read no position, such as those decoded as missing.

    Warns
    -----
    DataQualityWarning
        If a bin not in ``is_unread`` reaches more than half a sample into a
        gap of at least ``_TRACKING_GAP_SAMPLES`` samples.
    """
    if position_time.size < 2:
        return
    steps = np.diff(position_time)
    is_gap = np.rint(steps / np.median(steps)) >= _TRACKING_GAP_SAMPLES
    if not np.any(is_gap):
        return
    half_sample = _half_sample(position_time)
    reads_gap = ~is_unread & _bins_meeting_spans(
        time_edges,
        position_time[:-1][is_gap] + half_sample,
        position_time[1:][is_gap] - half_sample,
    )
    if not np.any(reads_gap):
        return
    warnings.warn(
        f"{int(reads_gap.sum())} of {reads_gap.size} decode bins fall in gaps in "
        f"position_time (the widest is {float(steps.max()):.3g} s), so position "
        "is interpolated between the samples either side and these bins are "
        "decoded on that invented path: spikes that disagree with it can read "
        "as replay. Pass valid_position_intervals=[[start, stop], ...] that "
        "exclude the gaps to decode these bins as missing.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )


def _time_span(times: np.ndarray) -> tuple[float, float]:
    """First and last of validated, increasing times, as Python floats."""
    return float(times[0]), float(times[-1])


def _validate_penalty_params(
    non_local_position_penalty: float, non_local_penalty_std: float
) -> None:
    """Validate non-local position penalty parameters.

    Parameters
    ----------
    non_local_position_penalty : float
        Magnitude of the Gaussian penalty. Must be non-negative.
    non_local_penalty_std : float
        Standard deviation of the penalty Gaussian in position units.
        Must be strictly positive.

    Raises
    ------
    ValidationError
        If ``non_local_position_penalty`` is negative or NaN, or
        ``non_local_penalty_std`` is not positive (including NaN).
    """
    # Written as "not >=" / "not >" so NaN, which compares False, fails.
    if not non_local_position_penalty >= 0:
        msg = "non_local_position_penalty must be >= 0"
        raise ValidationError(
            msg,
            expected="non_local_position_penalty >= 0",
            got=f"{non_local_position_penalty}",
            hint="A negative penalty would reward non-local states near the animal's "
            "position instead of suppressing them. Use 0.0 to disable the penalty.",
        )
    if not non_local_penalty_std > 0:
        msg = "non_local_penalty_std must be > 0"
        raise ValidationError(
            msg,
            expected="non_local_penalty_std > 0",
            got=f"{non_local_penalty_std}",
            hint="Standard deviation controls the width of the Gaussian penalty "
            "and is used as a divisor. A value <= 0 would cause divide-by-zero "
            "or invalid behavior.",
        )


def _validate_prior_weight(prior_weight: Any, n_states: int) -> None:
    """Check ``discrete_transition_prior_weight`` against the number of states.

    The accepted shapes are those of the EM M-step (``_row_prior_weights``):
    one weight (a scalar or a one-element array) for every row, or one per
    row.

    Parameters
    ----------
    prior_weight : float or array_like, shape () or (n_states,)
        Data-adaptive Dirichlet prior weight.
    n_states : int
        Number of discrete states.

    Raises
    ------
    ValidationError
        If it is not numeric, not one weight or ``n_states`` weights, or holds
        a negative or non-finite weight.
    """
    hint = (
        "Use 0.0 for the fixed-count prior, a finite weight >= 0 for every "
        f"row, or an array of {n_states} such weights, one per row."
    )
    try:
        weights = np.asarray(prior_weight, dtype=float)
    except (TypeError, ValueError):
        msg = "discrete_transition_prior_weight must be numeric"
        raise ValidationError(
            msg, expected="a float or an array of floats", got=repr(prior_weight)
        ) from None
    if weights.size != 1 and weights.shape != (n_states,):
        msg = "discrete_transition_prior_weight must be one weight or one per state"
        raise ValidationError(
            msg,
            expected=f"a scalar or shape ({n_states},)",
            got=f"shape {weights.shape}",
            hint=hint,
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        msg = "discrete_transition_prior_weight must be finite and non-negative"
        raise ValidationError(
            msg, expected="finite weights >= 0", got=repr(prior_weight), hint=hint
        )


def _normalize_frozen_discrete_transition_rows(
    frozen_rows: FrozenTransitionRows,
    is_no_spike_state: np.ndarray,
) -> np.ndarray | None:
    """Normalize ``frozen_discrete_transition_rows`` to a boolean row mask.

    Accepts ``"no_spike"`` (the rows of the No-Spike states), a sequence of
    integer row indices, a boolean mask of shape ``(n_states,)``, or
    ``None``/empty (no frozen rows). Returns a boolean mask of shape
    ``(n_states,)`` where ``True`` marks rows to freeze, or ``None`` if no
    rows are frozen.

    Parameters
    ----------
    frozen_rows : {"no_spike"}, np.ndarray, list of int, tuple of int, or None
        Rows to freeze.
    is_no_spike_state : np.ndarray of bool, shape (n_states,)
        Which discrete states have a No-Spike observation model.

    Returns
    -------
    np.ndarray of bool, shape (n_states,), or None

    Raises
    ------
    ValidationError
        If the input is an unknown string, has the wrong shape or dtype, or
        has out-of-range indices.
    """
    n_states = len(is_no_spike_state)
    if isinstance(frozen_rows, str):
        if frozen_rows != "no_spike":
            msg = (  # type: ignore[unreachable]  # runtime validation of an arbitrary string
                "frozen_discrete_transition_rows must be 'no_spike', row "
                f"indices, a boolean mask or None, got {frozen_rows!r}"
            )
            raise ValidationError(msg)
        frozen_rows = np.asarray(is_no_spike_state, dtype=bool)
        return frozen_rows.copy() if frozen_rows.any() else None
    if frozen_rows is None:
        return None
    arr = np.asarray(frozen_rows)
    if arr.size == 0:
        return None
    if arr.dtype == bool:
        if arr.shape != (n_states,):
            msg = (
                f"frozen_discrete_transition_rows boolean mask must have "
                f"shape ({n_states},), got {arr.shape}"
            )
            raise ValidationError(msg)
        return arr.copy()
    if not np.issubdtype(arr.dtype, np.integer):
        msg = (
            f"frozen_discrete_transition_rows must be integer indices or "
            f"a boolean mask, got dtype {arr.dtype}"
        )
        raise ValidationError(msg)
    if arr.ndim != 1:
        msg = (
            f"frozen_discrete_transition_rows index array must be 1D, "
            f"got shape {arr.shape}"
        )
        raise ValidationError(msg)
    if np.any(arr < 0) or np.any(arr >= n_states):
        msg = (
            f"frozen_discrete_transition_rows indices must be in "
            f"[0, {n_states}), got {arr.tolist()}"
        )
        raise ValidationError(msg)
    mask = np.zeros(n_states, dtype=bool)
    mask[arr] = True
    return mask


def _position_columns(environment: Environment) -> int:
    """Return the number of position columns a fitted environment decodes.

    Parameters
    ----------
    environment : Environment
        A fitted environment.

    Returns
    -------
    n_columns : int
        The coordinates of the track graph's nodes for a track-graph
        environment, which linearizes raw positions, otherwise the
        dimensions of its place grid.
    """
    if environment.track_graph is not None:
        _, node_position = next(iter(environment.track_graph.nodes(data="pos")))
        return len(node_position)
    return int(environment.place_bin_centers_.shape[1])


# A position sample more than this many place bins outside a grid's bounding
# box is off the grid; fitting and decoding warn when more than this fraction
# of the finite samples are. Tracking noise near the edge stays within a bin,
# and a stray sample or two stays under the fraction.
_OFF_GRID_BINS = 1.0
_OFF_GRID_FRACTION = 0.01


def _off_grid(
    points: np.ndarray, environment: Environment
) -> tuple[np.ndarray, np.ndarray]:
    """Which points are off a fitted grid, and how far outside its bounding box.

    Parameters
    ----------
    points : np.ndarray, shape (n_points, n_position_dims)
        Finite positions in the grid's coordinates.
    environment : Environment
        A fitted environment without a track graph.

    Returns
    -------
    is_off : np.ndarray of bool, shape (n_points,)
        True for a point more than ``_OFF_GRID_BINS`` bins outside the
        bounding box of the grid's bins in some dimension.
    distance : np.ndarray, shape (n_points,)
        Euclidean distance to that bounding box, in position units.
    """
    edges = environment.edges_
    lower = np.array([dimension_edges[0] for dimension_edges in edges])
    upper = np.array([dimension_edges[-1] for dimension_edges in edges])
    bin_width = np.array(
        [np.max(np.diff(dimension_edges)) for dimension_edges in edges]
    )
    excess = np.maximum(lower - points, 0.0) + np.maximum(points - upper, 0.0)
    return (
        np.any(excess > _OFF_GRID_BINS * bin_width, axis=1),
        np.linalg.norm(excess, axis=1),
    )


def _warn_if_off_grid(
    position: np.ndarray, environments: Sequence[Environment]
) -> None:
    """Warn when many positions lie far outside the fitted grids.

    Likelihoods move a position outside the grid to the nearest interior bin
    (``snap_off_track_positions``), which hides positions in other units than
    the grid or outside its ``position_range``. A finite sample farther than
    ``_OFF_GRID_BINS`` bins outside the bounding box of every environment's
    grid is off the grid; more than ``_OFF_GRID_FRACTION`` of them warns.
    Interior holes lie inside the bounding box. A track graph projects every
    position onto its track, so nothing is checked when an environment has
    one: its samples cannot be told apart from the others'.

    Parameters
    ----------
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Positions, with NaN rows for untracked samples.
    environments : sequence of Environment
        The fitted environments the samples may lie in.

    Warns
    -----
    DataQualityWarning
        If more than ``_OFF_GRID_FRACTION`` of the finite samples are off the
        grid.
    """
    if any(environment.track_graph is not None for environment in environments):
        return
    points = position[np.all(np.isfinite(position), axis=1)]
    n_samples = len(points)
    if n_samples == 0:
        return
    is_off_each, distance_each = zip(
        *(_off_grid(points, environment) for environment in environments), strict=True
    )
    # A sample on any environment's grid is on the grid.
    is_off = np.all(is_off_each, axis=0)
    n_off = int(np.count_nonzero(is_off))
    if n_off <= _OFF_GRID_FRACTION * n_samples:
        return
    farthest = np.max(np.min(distance_each, axis=0)[is_off])
    names = ", ".join(
        repr(environment.environment_name) for environment in environments
    )
    warnings.warn(
        f"{n_off / n_samples:.1%} of the {n_samples} finite position samples "
        f"lie more than one place bin outside the fitted grid of environment "
        f"{names}, up to {farthest:.3g} position units beyond "
        "it. The likelihoods evaluate such a position at the nearest interior "
        "bin instead. Position may be in larger units than the grid (for "
        "example millimetres on a grid in centimetres), the environment's "
        "position_range may not match the data, or, when decoding, the "
        "positions may lie where the fitting positions never went (without "
        "position_range a grid spans only the positions it was fitted on). "
        "Check the units of position, the environment's position_range and "
        "place_bin_size, and that the fitted positions cover the decoded ones.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )


def _warn_if_range_passes_tracking(
    position_time: np.ndarray,
    encoding_time_range: ArrayLike | None,
    valid_position_intervals: ArrayLike | None,
) -> None:
    """Warn when ``encoding_time_range`` reaches well past the position samples.

    Encoding support then extends to the range, holding position at the first
    or last sample, so that one sample takes all the exposure and spikes of the
    extra time. Declared tracking intervals bound the support themselves.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Finite and strictly increasing (``EncodingSupport`` checked).
    encoding_time_range : array_like, shape (2,), or None
        ``(start, stop)`` in seconds.
    valid_position_intervals : array_like, shape (n_intervals, 2), or None
        Declared tracking intervals.

    Warns
    -----
    DataQualityWarning
        If the range passes the samples by over one sample interval.
    """
    if (
        encoding_time_range is None
        or valid_position_intervals is not None
        or position_time.size < 2
    ):
        return
    start, stop = np.asarray(encoding_time_range, dtype=float)
    excess = max(position_time[0] - start, stop - position_time[-1])
    sample_interval = float(np.median(np.diff(position_time)))
    if excess <= sample_interval + _uniformity_tolerance(
        position_time, sample_interval
    ):
        return
    warnings.warn(
        f"encoding_time_range [{start:.6g}, {stop:.6g}] s reaches {excess:.3g} s "
        f"past the position samples [{position_time[0]:.6g}, "
        f"{position_time[-1]:.6g}] s. Encoding support extends to the range, "
        "holding position at the first or last sample, so that sample takes all "
        "the exposure and spikes of the extra time. Pass the tracked period as "
        "encoding_time_range, or spikes and position of the same period.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )


def _require_position_columns(
    position: np.ndarray, environments: Sequence[Environment]
) -> None:
    """Require the number of position columns the fitted environments read.

    Parameters
    ----------
    position : np.ndarray, shape (n_time_position,) or (n_time_position, n_position_dims)
        Positions.
    environments : sequence of Environment
        Fitted environments.

    Raises
    ------
    ValidationError
        If an environment was fitted on another number of columns.
    """
    n_columns = position.shape[1]
    for environment in environments:
        expected = _position_columns(environment)
        if n_columns != expected:
            msg = (
                f"position has {n_columns} column(s), but environment "
                f"{environment.environment_name!r} was fitted on "
                f"positions with {expected} columns"
            )
            raise ValidationError(
                msg,
                expected=f"position of shape (n_time_position, {expected})",
                got=f"shape {position.shape}",
                hint="Use the same position columns, in the same "
                "order and units, as the detector was fitted with.",
            )


def _missing_covariates_error(
    *, reason: str, expected: str, hint: str, example: str
) -> ValidationError:
    """Return the error of a call without the covariates its transitions need.

    Parameters
    ----------
    reason : str
        Why the transitions need them.
    expected, hint, example : str
        What the call should pass, as ``ValidationError`` takes them.

    Returns
    -------
    error : ValidationError
    """
    msg = f"Missing required parameter: discrete_transition_covariate_data ({reason})"
    return ValidationError(
        msg, expected=expected, got="None", hint=hint, example=example
    )


def _validate_covariate_time_length(
    covariate_rows: np.ndarray, time: np.ndarray
) -> None:
    """Validate that covariates have one row per decode bin.

    Parameters
    ----------
    covariate_rows : np.ndarray, shape (n_covariate_time, ...)
        One row per covariate time step: the covariate-derived discrete
        transition matrices, shape (n_covariate_time, n_states, n_states), or
        one covariate.
    time : np.ndarray, shape (n_bins,)
        Decode bin centers.

    Raises
    ------
    ValidationError
        If the number of covariate time steps does not match decode time.
    """
    n_covariate_time = covariate_rows.shape[0]
    n_decode_time = len(time)
    if n_covariate_time != n_decode_time:
        msg = (
            f"Covariate data has {n_covariate_time} time steps but "
            f"decode time has {n_decode_time} time steps. "
            f"These must match."
        )
        raise ValidationError(
            msg,
            expected=f"{n_decode_time} rows of discrete_transition_covariate_data, "
            "one per decode bin",
            got=f"{n_covariate_time} rows",
            hint="Evaluate the covariates at the decode bin centers, "
            "(time_edges[:-1] + time_edges[1:]) / 2, e.g. by interpolating them "
            "from their own timeline. The row count cannot show that they are "
            "aligned: rebuild them for every new time_edges.",
        )


def _normalize_return_outputs(return_outputs: ReturnOutputs) -> set[str]:
    """Convert ``return_outputs`` to the set of optional outputs it names.

    A string names the same outputs as a one-element list: each name adds
    only its own output, and "all" adds every one.

    Parameters
    ----------
    return_outputs : str, collection of str, or None
        Output names from ``VALID_OUTPUTS``: one name, or any non-string
        collection of names (list, tuple, set, frozenset).

    Returns
    -------
    set of str
        Normalized set containing any of: 'filter', 'predictive',
        'predictive_posterior', 'log_likelihood'

    Raises
    ------
    ValidationError
        If return_outputs is not a string, a collection of strings or None,
        or contains invalid option names.
    """
    if return_outputs is None:
        return set()
    # Checked at runtime: callers may pass anything.
    value: object = return_outputs
    if isinstance(value, str):
        names: set[str] = {value}
    elif isinstance(value, Collection) and not isinstance(value, Mapping):
        wrong = [name for name in value if not isinstance(name, str)]
        if wrong:
            msg = "return_outputs entries must be strings"
            raise ValidationError(
                msg,
                expected=f"names from {sorted(VALID_OUTPUTS)}",
                got=f"entries of type {sorted({type(name).__name__ for name in wrong})}",
            )
        names = set(value)
    else:
        msg = "return_outputs must be a string, a collection of strings, or None"
        raise ValidationError(msg, got=type(value).__name__)
    invalid = names - VALID_OUTPUTS
    if invalid:
        msg = f"Invalid return_outputs: {sorted(invalid)}"
        raise ValidationError(
            msg,
            expected=f"names from {sorted(VALID_OUTPUTS)}",
            got=repr(return_outputs),
        )
    return set(_OPTIONAL_OUTPUTS) if "all" in names else names


class _InputValidationMixin:
    """Validation methods of ``_DetectorBase``."""

    def _validated_inputs(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        stage: _InputStage,
        spike_data: tuple[Any, ...],
        *,
        position_time: ArrayLike | None = None,
        position: ArrayLike | None = None,
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        encoding_time_range: ArrayLike | None = None,
        valid_position_intervals: ArrayLike | None = None,
        time_edges: ArrayLike | None = None,
        row_slice: slice | None = None,
        is_missing: np.ndarray | None = None,
        example: str | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        transition_representation: TransitionRepresentation = "dense",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> _ValidatedInputs:
        """Coerce and check the data of one public call.

        Every public method taking data goes through this before any work,
        once per stage it runs, so a new check is added here and applies to
        each stage that takes the argument. Warnings fire here and nowhere that a call can repeat
        (chunks, EM iterations). What differs between the stages:

        - ``fit``: the training arrays, the spikes, then what ``_fit`` needs
          before anything changes. Spikes are not compared with a time range
          here: ``fit`` goes on to ``fit_encoding_model``, whose stage warns.
        - ``encoding`` (``fit_encoding_model``): the spikes against the
          position span come first, then the training arrays, position against
          the grids, and the encoding support.
        - ``refit`` (EM's encoding refits, through ``_fit_encoding_model``):
          the arrays are the ones the public call checked, so this reads the
          iteration's ``weights`` and warns of nothing.
        - ``decode`` (``predict``, ``most_likely_sequence``): position is
          coerced first, then the fitted model, the time grid, the spikes
          against it, the position against the model, and the missing bins.
        - ``likelihood`` (``compute_log_likelihood``): as ``decode``, except
          that the fitted model comes first, and ``is_missing`` has one entry
          per requested row: it says which bins the gap check reads, and this
          stage returns the position's mask. ``row_slice`` selects the rows of
          ``time_centers``.
        - ``estimate`` (``estimate_parameters``): ``time_edges`` come checked
          with EM's options, and the model is not fitted yet, so the training
          arrays are read as in ``fit``, the spikes against the edges and the
          missing bins as in ``decode``. The internal ``fit`` then warns about
          the position span.

        Parameters
        ----------
        stage : {"fit", "encoding", "refit", "decode", "likelihood", "estimate"}
            The kind of public method.
        spike_data : tuple
            ``spike_times``, then ``spike_waveform_features`` when clusterless.
        position_time : array_like, shape (n_time_position,), optional
            Times of the position samples.
        position : array_like, shape (n_time_position,) or (n_time_position, n_position_dims), optional
            Position samples.
        is_training : array_like of bool, shape (n_time_position,), optional
            Samples to train the encoding model on.
        encoding_group_labels : array_like, shape (n_time_position,), optional
            Encoding group of each sample.
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment of each sample.
        weights : array_like, shape (n_time_position,), optional
            Training weights.
        encoding_time_range : array_like, shape (2,), optional
            See ``fit``.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            See ``fit`` and ``predict``.
        time_edges : array_like, shape (n_bins + 1,), optional
            Decode bin edges.
        row_slice : slice, optional
            Rows of ``compute_log_likelihood``.
        is_missing : array_like of bool, shape (n_bins,) or (n_rows,), optional
            The caller's missing bins, of the requested rows at ``likelihood``.
        example : str, optional
            Example call for the error of a missing ``position_time``.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            The covariates ``fit`` was given.
        transition_representation : {"dense", "structured", "auto"}, optional
            See ``fit``.
        max_dense_transition_bytes : int, optional
            See ``fit``.

        Returns
        -------
        inputs : _ValidatedInputs

        Raises
        ------
        ValidationError, DataError, NotFittedError
            From the checks of the stage.
        """
        decoding = stage in ("decode", "likelihood")
        if stage == "likelihood":
            self._require_decodable()
        if stage == "encoding":
            times = _sample_times(cast("ArrayLike", position_time))
            finite = times[np.isfinite(times)]
            spike_data = self._check_spike_data(
                spike_data,
                time_range=(float(finite.min()), float(finite.max()))
                if finite.size
                else None,
                range_name="position_time span",
                check_extent=True,
                bin_width=1.0 / self.sampling_frequency,
            )
        if decoding:
            inputs = _ValidatedInputs.from_decode_arguments(
                spike_data, position_time, position, example=example
            )
        else:
            inputs = _ValidatedInputs.from_encoding_arguments(
                spike_data,
                cast("ArrayLike", position_time),
                position,
                is_training,
                encoding_group_labels,
                environment_labels,
                weights,
            )
        position_time, position = inputs.position_time, inputs.position
        if stage == "decode":
            self._require_decodable()
        checked_edges = cast("np.ndarray | None", time_edges)
        time_centers = None
        if decoding:
            checked_edges, time_centers = _decode_time_edges(
                cast("ArrayLike", time_edges),
                *self._learned_transition_time_bin_width(),
                sampling_frequency=self.sampling_frequency,
                row_slice=row_slice,
            )
            self._check_spike_data(
                spike_data,
                time_range=_time_span(checked_edges),
                range_name="decode time_edges",
                bin_width=1.0 / self.sampling_frequency,
            )
            self._validate_decode_position(position)
        if stage == "estimate":
            # The dtype check is the internal fit's, which reads the same times.
            spike_data = self._check_spike_data(
                spike_data,
                time_range=_time_span(cast("np.ndarray", checked_edges)),
                range_name="decode time_edges",
            )
        if stage == "fit":
            spike_data = self._check_spike_data(spike_data)
            self._validate_fit_arguments(
                cast("np.ndarray", position),
                discrete_transition_covariate_data,
                transition_representation,
                max_dense_transition_bytes,
            )
        if position is not None and position.ndim == 1:
            position = position[:, np.newaxis]
        if stage == "encoding":
            self._validate_encoding_position(
                cast("np.ndarray", position), inputs.environment_labels
            )
        encoding_support = tracking_intervals = None
        if stage in ("encoding", "refit"):
            encoding_support = EncodingSupport(
                cast("ArrayLike", position_time),
                cast("ArrayLike", position),
                encoding_time_range=encoding_time_range,
                valid_position_intervals=valid_position_intervals,
            )
        if stage == "encoding":
            _warn_if_range_passes_tracking(
                cast("np.ndarray", position_time),
                encoding_time_range,
                valid_position_intervals,
            )
        if stage in ("decode", "likelihood", "estimate"):
            tracking_intervals = (
                None
                if valid_position_intervals is None
                else parse_tracking_intervals(valid_position_intervals)
            )
            reads_position = self._decode_reads_position(
                self.observation_models
                if stage == "estimate"
                else self.state_space_.observation_models
            )
            caller_is_missing = is_missing
            is_missing = _missing_bins(
                cast("np.ndarray", checked_edges),
                None if stage == "likelihood" else is_missing,
                position_time,
                position,
                reads_position=reads_position,
                valid_position_intervals=tracking_intervals,
            )
            edges = cast("np.ndarray", checked_edges)
            times = cast("np.ndarray", position_time)
            if decoding and reads_position:
                _warn_about_tracking_coverage(
                    edges, times, row_slice, caller_is_missing, is_missing
                )
            # Without intervals, the fit that comes next refuses a gap.
            if (
                stage == "estimate"
                and reads_position
                and tracking_intervals is not None
            ):
                _warn_if_decoding_across_gaps(
                    edges, times, _unread_bins(edges.size - 1, None, None, is_missing)
                )
        return replace(
            inputs,
            spike_data=spike_data,
            position=position,
            encoding_support=encoding_support,
            time_edges=checked_edges,
            time_centers=time_centers,
            is_missing=is_missing,
        )

    def _validate_fit_arguments(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        position: np.ndarray,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None,
        transition_representation: TransitionRepresentation,
        max_dense_transition_bytes: int,
    ) -> None:
        """Refuse a fit that cannot run, before anything changes.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position,) or (n_time_position, n_position_dims)
            As the caller gave it, so errors report the caller's shape.
        discrete_transition_covariate_data : dict or pd.DataFrame or None
            The fit's covariates.
        transition_representation : {"dense", "structured", "auto"}
            See ``fit``.
        max_dense_transition_bytes : int
            See ``fit``.

        Raises
        ------
        ValidationError
            If a transition setting is unsupported, no position sample is
            finite, position is 1-D on a track graph, the transition formula
            is untrusted, or covariate transitions have no covariates.
        DataError
            If position holds an infinite value.
        """
        _validate_transition_arguments(
            transition_representation, max_dense_transition_bytes
        )
        val.ensure_all_finite(position[~np.isnan(position)], "position")
        if not np.any(np.all(np.isfinite(position.reshape(len(position), -1)), axis=1)):
            msg = "At least one finite position sample is required for fitting"
            raise ValidationError(msg)

        # Raises ValidationError if position is 1D but env has track_graph.
        self._validate_position_dimensionality(position, context="fit")

        # Before anything changes, so a refused fit keeps the loaded model.
        _refuse_untrusted_transition(self.discrete_transition_type, "fit")
        if (
            not self.discrete_transition_type.is_stationary
            and discrete_transition_covariate_data is None
        ):
            raise _missing_covariates_error(
                reason=f"{type(self.discrete_transition_type).__name__} depends "
                "on covariates",
                expected="the training values of the covariates named in the "
                "transition formula",
                hint="Pass the covariates, or use a stationary transition such as "
                "DiscreteStationaryDiagonal.",
                example="    detector.fit(position_time, position, spike_times, "
                "discrete_transition_covariate_data={'speed': speed})",
            )

    def _require_decodable(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
    ) -> None:
        """Check the fitted model a decoding call reads, and warn if it changed."""
        self._validate_time_contract()
        self._warn_if_layout_parameters_changed()

    def _check_spike_data(
        self,
        spike_data: tuple[Any, ...],
        *,
        time_range: tuple[float, float] | None = None,
        range_name: str = "",
        check_extent: bool = False,
        bin_width: float | None = None,
    ) -> tuple[Any, ...]:
        """Check the spike arrays.

        See ``_validate_spike_data`` for what is checked.

        Parameters
        ----------
        spike_data : tuple
            ``spike_times``, then ``spike_waveform_features`` when clusterless.
        time_range : tuple of float, optional
            ``(start, stop)`` in seconds to warn against, by default None.
        range_name : str, optional
            What ``time_range`` is, for the warning.
        check_extent : bool, optional
            Also warn when the spikes overlap ``time_range`` but look like
            other units than it, by default False (encoding period only).
        bin_width : float, optional
            Bin width in seconds the spike times' dtype must resolve, by
            default None.

        Returns
        -------
        spike_data : tuple
            With ``spike_times`` as numeric arrays. Fitting reads them with
            ``np.interp``, which rejects the object-dtype arrays that decoding
            reads as float.
        """
        spike_times = _validate_spike_data(
            *spike_data,
            clusterless=len(spike_data) > 1,
            time_range=time_range,
            range_name=range_name,
            check_extent=check_extent,
            bin_width=bin_width,
        )
        return (spike_times, *spike_data[1:])

    def _validate_position_dimensionality(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        position: np.ndarray,
        context: str = "fit",
    ) -> None:
        """Check that position shape matches the environment type.

        When any environment has a ``track_graph``, positions must be
        raw 2D ``(x, y)`` coordinates — the detector linearizes
        internally via ``get_position_at_time``. Passing already-
        linearized 1D position in this case causes the internal
        linearization step to silently produce garbage coordinates,
        which mis-centers the local kernel and non-local penalty.

        This check raises a clear ``ValidationError`` when the shape
        does not match the environment type, rather than allowing
        silent numerical corruption downstream.

        Parameters
        ----------
        position : np.ndarray
            Position array to validate.
        context : str, optional
            Name of the calling method (``"fit"``, ``"fit_encoding_model"`` or
            ``"predict"``) for error messages. Fitting checks the configured
            ``environments``; ``predict`` checks the fitted ``environments_``,
            which it decodes on.
        """
        is_1d = position.ndim == 1 or (position.ndim == 2 and position.shape[1] == 1)
        if not is_1d:
            return

        environments = self.environments_ if context == "predict" else self.environments
        has_track_graph = any(env.track_graph is not None for env in environments)
        if has_track_graph:
            msg = (
                "Environment has a track_graph but position is 1D "
                f"(shape {position.shape}). Track-graph environments "
                "require raw 2D (x, y) position — the detector "
                "linearizes internally. Passing pre-linearized 1D "
                "position silently corrupts internal coordinates."
            )
            raise ValidationError(
                msg,
                expected="shape (n_time, 2) or (n_time, n_dims) with n_dims >= 2",
                got=f"shape {position.shape}",
                hint=(
                    "If you already called get_linearized_position() to "
                    "get 1D position, pass the raw 2D (x, y) array "
                    "instead — the detector linearizes via the track_graph "
                    "during both fit() and predict()."
                ),
                example=(
                    "    # Raw 2D position, not pre-linearized\n"
                    f"    detector.{context}(..., position=position_2d)"
                ),
            )

    def _validate_encoding_position(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        position: np.ndarray,
        environment_labels: np.ndarray | None,
    ) -> None:
        """Check position against the grids an encoding model is fit on.

        A grid is warned about here rather than where it is fit, so ``fit``
        warns once, after its fit succeeded (an EM fit that falls back to
        dense warns once), and ``fit_encoding_model`` warns like ``fit``.
        Without fitted environments there is nothing to compare with.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Positions, with NaN rows for untracked samples.
        environment_labels : np.ndarray, shape (n_time_position,), or None
            Environment of each sample.

        Raises
        ------
        ValidationError
            If position is 1-D while an environment has a track graph, or has
            another number of columns than a fitted environment's positions.

        Warns
        -----
        DataQualityWarning
            If many positions lie far outside a fitted grid (see
            ``_warn_if_off_grid``).
        """
        self._validate_position_dimensionality(position, context="fit_encoding_model")
        environments = getattr(self, "environments_", ())
        _require_position_columns(position, environments)
        # Each grid is fitted on its environment's samples.
        for environment in environments:
            _warn_if_off_grid(
                position
                if environment_labels is None
                else position[environment_labels == environment.environment_name],
                [environment],
            )

    def _validate_decode_position(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        position: np.ndarray | None,
    ) -> None:
        """Check that decoding has the position its likelihood reads.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims), or None
            Position supplied for decoding, or None if not supplied.

        Raises
        ------
        ValidationError
            If position is None but Local states or a non-local position
            penalty read it, position is 1-D (one column) while an
            environment has a track graph, which needs raw 2-D positions, or
            the likelihood reads position and it has another number of
            columns than a fitted environment's positions.

        Warns
        -----
        DataQualityWarning
            If the decode likelihood reads position and many positions lie
            far outside the fitted grids (see ``_warn_if_off_grid``).
        """
        observation_models = self.state_space_.observation_models
        if position is not None:
            self._validate_position_dimensionality(position, context="predict")
            if not self._decode_reads_position(observation_models):
                return
            _require_position_columns(position, self.environments_)
            _warn_if_off_grid(position, self.environments_)
            return
        if not self._decode_reads_position(observation_models):
            return
        reason = []
        if any(obs.is_local for obs in observation_models):
            reason.append("local observation models")
        if self._non_local_position_penalty() > 0:
            reason.append("non_local_position_penalty > 0")
        msg = f"Missing required parameter: position (needed for {', '.join(reason)})"
        raise ValidationError(
            msg,
            expected="position array with shape (n_time, n_dims)",
            got="None",
            hint="Provide position data or set non_local_position_penalty=0.0 to disable the penalty",
            example="    results = detector.predict(..., time_edges=time_edges, position=position, position_time=position_time)",
        )

    def _validate_initial_conditions(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        continuous_transition_types: ContinuousTransitions,
        discrete_transition_stickiness: Stickiness,
    ) -> None:
        """
        Validate the initial conditions.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.

        Raises
        ------
        ValidationError
            If the number of discrete initial conditions does not match the number of continuous initial conditions or transition types.
        """
        n_discrete = len(discrete_initial_conditions)
        n_continuous_init = len(continuous_initial_conditions_types)
        n_continuous_trans = len(continuous_transition_types)

        if n_discrete != n_continuous_init:
            msg = "Mismatch between discrete initial conditions and continuous initial conditions"
            raise ValidationError(
                msg,
                expected=f"{n_continuous_init} discrete initial condition(s) (one per continuous initial condition type)",
                got=f"{n_discrete} discrete initial condition(s)",
                hint="Each continuous initial condition type needs a corresponding discrete initial probability",
                example=f"    discrete_initial_conditions=np.array([{', '.join(['1.0/' + str(n_continuous_init)] * n_continuous_init)}])",
            )

        if n_discrete != n_continuous_trans:
            state_names_str = (
                f" ({self.state_names})"
                if hasattr(self, "state_names") and self.state_names
                else ""
            )
            msg = "Mismatch between discrete initial conditions and continuous transition types"
            raise ValidationError(
                msg,
                expected=f"{n_continuous_trans} discrete initial condition(s) (one per state{state_names_str})",
                got=f"{n_discrete} discrete initial condition(s)",
                hint="Each state needs an initial probability. Check that your discrete_initial_conditions array has the same length as continuous_transition_types list.",
                example=f"    # For {n_continuous_trans} states:\n    discrete_initial_conditions=np.array([{', '.join(['1.0/' + str(n_continuous_trans)] * n_continuous_trans)}])",
            )

        if not isinstance(discrete_transition_stickiness, float) and len(
            discrete_initial_conditions
        ) != len(discrete_transition_stickiness):
            msg = f"Discrete transition stickiness must be set for all {n_discrete} states or be a single float"
            raise ValidationError(
                msg,
                expected=f"Either a float or array of length {n_discrete}",
                got=f"Array of length {len(discrete_transition_stickiness)}",
                hint="Use a float for uniform stickiness across all states, or an array with one value per state",
                example=f"    discrete_transition_stickiness=0.0  # uniform\n    # OR\n    discrete_transition_stickiness=np.array([{', '.join(['0.0'] * n_discrete)}])  # per-state",
            )

    def _validate_probability_distributions(
        self, discrete_initial_conditions: np.ndarray
    ) -> None:
        """Validate that probability distributions sum to 1.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states

        Raises
        ------
        ValidationError
            If discrete_initial_conditions is not 1D, contains invalid values,
            or does not sum to 1
        DataError
            If array contains NaN or Inf values
        """
        # Check array is 1D
        val.ensure_array_1d(discrete_initial_conditions, "discrete_initial_conditions")

        # Check for NaN/Inf
        val.ensure_all_finite(
            discrete_initial_conditions, "discrete_initial_conditions"
        )

        # Check non-negative
        val.ensure_all_non_negative(
            discrete_initial_conditions, "discrete_initial_conditions"
        )

        # Check sums to 1
        val.ensure_probability_distribution(
            discrete_initial_conditions, "discrete_initial_conditions"
        )

    def _validate_numerical_parameters(
        self,
        discrete_transition_concentration: float,
        discrete_transition_regularization: float,
        sampling_frequency: float,
        no_spike_rate: float,
    ) -> None:
        """Validate numerical parameters are in valid ranges.

        Parameters
        ----------
        discrete_transition_concentration : float
            Concentration parameter (must be >= 1.0 for the MAP transition update).
            This ensures the pseudo-count update adds non-negative values.
        discrete_transition_regularization : float
            Regularization parameter (must be >= 0)
        sampling_frequency : float
            Sampling frequency in Hz (must be > 0)
        no_spike_rate : float
            No-spike rate (must be > 0)

        Raises
        ------
        ValidationError
            If any parameter is outside its valid range
        """
        val.ensure_positive_scalar(
            discrete_transition_concentration,
            "discrete_transition_concentration",
            minimum=1.0,
            strict=False,
        )

        val.ensure_positive_scalar(
            discrete_transition_regularization,
            "discrete_transition_regularization",
            minimum=0.0,
            strict=False,  # Can be exactly 0
        )

        val.ensure_positive_scalar(
            sampling_frequency,
            "sampling_frequency",
            minimum=0.0,
            strict=True,
        )

        val.ensure_positive_scalar(
            no_spike_rate,
            "no_spike_rate",
            minimum=0.0,
            strict=True,
        )

    def _validate_discrete_transition_type(
        self, discrete_transition_type: Any, n_states: int
    ) -> None:
        """Validate the discrete transition against the model.

        The transition validates its own values when it is constructed; this
        checks that it is a ``DiscreteTransition`` for this many states.

        Parameters
        ----------
        discrete_transition_type : DiscreteTransition
            Discrete transition type object
        n_states : int
            Number of states in the model

        Raises
        ------
        ValidationError
            If the object is not a ``DiscreteTransition`` or is sized for a
            different number of states.
        """
        if not isinstance(discrete_transition_type, DiscreteTransition):
            msg = "discrete_transition_type must be a DiscreteTransition"
            raise ValidationError(
                msg,
                expected=(
                    "an object with is_stationary, n_states, smoothness and "
                    "make_state_transition(covariate_data)"
                ),
                got=type(discrete_transition_type).__name__,
                hint=(
                    "Use DiscreteStationaryDiagonal, DiscreteStationaryCustom, "
                    "DiscreteNonStationaryDiagonal or DiscreteNonStationaryCustom."
                ),
            )
        if discrete_transition_type.n_states != n_states:
            msg = "Discrete transition matrix size must match number of states"
            raise ValidationError(
                msg,
                expected=f"a transition for {n_states} states",
                got=f"a transition for {discrete_transition_type.n_states} states",
                hint=(
                    f"Your model has {n_states} states, so the transition "
                    f"needs a {n_states}x{n_states} matrix (or {n_states} "
                    "diagonal values)."
                ),
            )

    @staticmethod
    def _validate_observation_models(observation_models: Any, n_states: int) -> None:
        """Require one ``ObservationModel`` per discrete state.

        The state roles (Local, No-Spike) are read from these models, so a
        non-model entry or a count that differs from the number of states
        would misassign them.

        Parameters
        ----------
        observation_models : list, tuple or np.ndarray of ObservationModel
            Output of ``_initialize_observation_models``.
        n_states : int
            Number of discrete states.

        Raises
        ------
        ValidationError
            If ``observation_models`` is not a list, tuple or array, holds
            anything but ``ObservationModel`` instances, or does not have
            ``n_states`` entries.
        """
        if not isinstance(observation_models, (list, tuple, np.ndarray)):
            msg = (
                "observation_models must be a list, tuple or array of "
                "ObservationModel instances"
            )
            raise ValidationError(
                msg,
                expected="an ObservationModel, a list, tuple or array of them, or None",
                got=type(observation_models).__name__,
            )
        wrong_types = sorted(
            {
                type(obs).__name__
                for obs in observation_models
                if not isinstance(obs, ObservationModel)
            }
        )
        if wrong_types:
            msg = "observation_models entries must be ObservationModel instances"
            raise ValidationError(
                msg,
                expected="ObservationModel entries",
                got=f"entries of type {', '.join(wrong_types)}",
            )
        if len(observation_models) != n_states:
            msg = (
                "observation_models must have one observation model per discrete state"
            )
            raise ValidationError(
                msg,
                expected=f"{n_states} observation models",
                got=f"{len(observation_models)}",
            )

    def _validate_time_contract(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
    ) -> None:
        """Reject an unfitted model, unknown encoding units or transition clocks.

        Public decoding methods call this once, before any work. Subclasses
        extend it with their own fitted state (the clusterless mark
        reference); a detector without an encoding model must override it.

        Raises
        ------
        NotFittedError
            If the detector has no encoding model yet.
        ValidationError
            If the transition formula is untrusted, the saved model no longer
            rebuilds as saved, or it has a legacy or unknown time contract
            (encoding units or transition clocks) and must be refit.
        """
        if not hasattr(self, "encoding_model_"):
            msg = "Detector is not fitted; call fit(...) before decoding."
            raise NotFittedError(msg)
        _refuse_untrusted_transition(self.discrete_transition_type, "decode")
        # A loaded detector's rebuilt structures must match the saved ones.
        rebuild_check = self.__dict__.get("_rebuild_check_")
        if rebuild_check is not None:
            rebuild_check.verify(self)
        if (
            getattr(self, "time_contract_", None) != _TIME_CONTRACT
            or not hasattr(self, "transition_time_bin_width_")
            or not isinstance(self.encoding_model_, Mapping)
        ):
            msg = "Saved model has a legacy or unknown time contract; refit before decoding."
            raise ValidationError(
                msg,
                hint="Load for inspection, then call fit(...) or estimate_parameters(..., time_edges=...) with the original recording. Old rates and learned transition clocks cannot be inferred safely.",
            )
        # Validate the complete dispatch set before any state is evaluated.
        # Scanning existing values alone misses a deleted active entry and
        # wrongly rejects unused entries, including No-Spike-only keys.
        required_keys = {
            (obs.environment_name, obs.encoding_group)
            for obs in self.state_space_.observation_models
            if not obs.is_no_spike
        }
        for key in required_keys:
            model = self.encoding_model_.get(key)
            units = model.get("rate_units") if isinstance(model, Mapping) else None
            if not isinstance(units, str) or units != "Hz":
                msg = f"Encoding entry {key!r} is missing or has unknown rate units; refit before decoding."
                raise ValidationError(
                    msg,
                    hint="Each environment/encoding-group used by a spike state "
                    "needs a fitted encoding dictionary with rate_units='Hz'. "
                    "Call fit(...) or estimate_parameters(..., time_edges=...) "
                    "with the original recording. No-Spike uses its constructor rate in Hz.",
                )
