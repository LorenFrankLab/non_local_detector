"""Input validation of the detectors.

The checks the detectors run on their constructor parameters and on the data
of their public methods: the coerced encoding and decoding inputs
(``_EncodingInputs``, ``_DecodeInputs``), the decode time grid, the
prediction and transition options, ``return_outputs``, and the fitted
model's time contract. ``_DetectorBase`` inherits the methods of
``_InputValidationMixin``.

Methods that read the detector's attributes annotate ``self`` as
``_DetectorBase``, which mypy only accepts with ``type: ignore[misc]`` on a
mixin.
"""

import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
from numpy.typing import ArrayLike

import non_local_detector
from non_local_detector import _validation as val
from non_local_detector._warnings import user_stacklevel
from non_local_detector.discrete_state_transitions import (
    DiscreteTransition,
    _refuse_untrusted_transition,
)
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


def _validate_spike_data(
    spike_times: Sequence[Any],
    spike_waveform_features: Sequence[Any] | None = None,
    *,
    clusterless: bool,
    time_range: tuple[float, float] | None = None,
    range_name: str = "",
) -> None:
    """Check spike data where it enters a public fitting or decoding method.

    Checks the structure (``_validate_spike_structure``), then that every
    spike time is a finite number. With ``time_range``, warns when every
    spike of every unit lies before it, or every one after it: the spike times
    are then most likely in other units (ms rather than s) or on another clock
    than the times they are compared with. A range without spikes between
    spikes (a quiet window) and a unit whose spikes all lie outside it are not
    reported.

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

    Raises
    ------
    ValidationError
        If the structure is wrong or spike times are not numeric.
    DataError
        If any spike time is NaN or infinite.

    Warns
    -----
    DataQualityWarning
        If there are spikes, all before or all after ``time_range``.
    """
    _validate_spike_structure(
        spike_times, spike_waveform_features, clusterless=clusterless
    )
    unit = "electrode" if clusterless else "neuron"
    arrays = [np.asarray(times) for times in spike_times]
    for index, times in enumerate(arrays):
        if times.dtype.kind not in "iuf":
            msg = f"spike_times for {unit} {index} must be numeric"
            raise ValidationError(
                msg, expected="spike times in seconds", got=f"dtype {times.dtype}"
            )
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
    if time_range is None:
        return
    start, stop = time_range
    n_spikes = sum(times.size for times in arrays)
    if n_spikes == 0:
        return
    first = min(times.min() for times in arrays if times.size)
    last = max(times.max() for times in arrays if times.size)
    # A quiet window between spikes is data, not a clock mismatch.
    if start <= last and first <= stop:
        return
    warnings.warn(
        f"None of the {n_spikes} spike times lies within the {range_name} "
        f"[{start:.6g}, {stop:.6g}] s; they span [{first:.6g}, {last:.6g}]. "
        "The spike times are probably in other units (e.g. ms instead of s) "
        f"or on another clock than the {range_name}, so no spike is used. "
        "Convert them to seconds on the same clock.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )


@dataclass(frozen=True)
class _EncodingInputs:
    """Encoding data of a public fitting method, coerced and checked once.

    Optional arrays stay None, so each fitting stage keeps its own default.

    Attributes
    ----------
    position_time : np.ndarray, shape (n_time_position,)
    position : np.ndarray, shape (n_time_position,) or (n_time_position, n_position_dims)
        As the caller gave it, so errors report the caller's shape.
    is_training : np.ndarray of bool, shape (n_time_position,), or None
    encoding_group_labels : np.ndarray of int or str, shape (n_time_position,), or None
    environment_labels : np.ndarray of str, shape (n_time_position,), or None
    weights : np.ndarray of float, shape (n_time_position,), or None
    """

    position_time: np.ndarray
    position: np.ndarray
    is_training: np.ndarray | None
    encoding_group_labels: np.ndarray | None
    environment_labels: np.ndarray | None
    weights: np.ndarray | None = None

    @classmethod
    def from_arguments(
        cls,
        position_time: ArrayLike,
        position: ArrayLike | None,
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> "_EncodingInputs":
        """Coerce and check the arguments of ``fit``-like methods.

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


@dataclass(frozen=True)
class _DecodeInputs:
    """Position of a public decoding method, coerced and checked once.

    Attributes
    ----------
    position_time : np.ndarray, shape (n_time_position,), or None
    position : np.ndarray, shape (n_time_position, n_position_dims), or None
    """

    position_time: np.ndarray | None
    position: np.ndarray | None

    @classmethod
    def from_arguments(
        cls,
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        *,
        example: str | None = None,
    ) -> "_DecodeInputs":
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
            return cls(times, None)
        position = _sample_positions(position, len(times))
        return cls(times, position[:, np.newaxis] if position.ndim == 1 else position)


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
        "decode these edges, or build the edges at the detector's rate with "
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


def _validate_covariate_time_length(
    predicted_transitions: np.ndarray, time: np.ndarray
) -> None:
    """Validate that covariate-derived transitions match decode time length.

    Parameters
    ----------
    predicted_transitions : np.ndarray, shape (n_covariate_time, n_states, n_states)
        Covariate-derived discrete transition matrices, one per time step.
    time : np.ndarray, shape (n_bins,)
        Decode bin centers.

    Raises
    ------
    ValidationError
        If the number of covariate time steps does not match decode time.
    """
    n_covariate_time = predicted_transitions.shape[0]
    n_decode_time = len(time)
    if n_covariate_time != n_decode_time:
        msg = (
            f"Covariate data has {n_covariate_time} time steps but "
            f"decode time has {n_decode_time} time steps. "
            f"These must match."
        )
        raise ValidationError(msg)


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
            Name of the calling context (``"fit"`` or ``"predict"``)
            for error messages. ``fit`` checks the configured
            ``environments``, which it is about to fit; ``predict`` checks
            the fitted ``environments_``, which it decodes on.
        """
        is_1d = position.ndim == 1 or (position.ndim == 2 and position.shape[1] == 1)
        if not is_1d:
            return

        environments = self.environments if context == "fit" else self.environments_
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
            penalty read it, or position is 1-D (one column) while an
            environment has a track graph, which needs raw 2-D positions.
        """
        if position is not None:
            self._validate_position_dimensionality(position, context="predict")
            return
        if not self._decode_reads_position():
            return
        reason = []
        if any(obs.is_local for obs in self.observation_models):
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
            for obs in self.observation_models
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
