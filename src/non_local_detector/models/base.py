"""Detector base class and the sorted-spikes and clusterless detectors.

This module defines the shared HMM machinery of the package: initializing the
environments, state space, initial conditions and state transitions; fitting
encoding models; computing log likelihoods; decoding with the forward-backward
algorithm (``predict``); estimating parameters with EM
(``estimate_parameters``); finding the most likely state sequence; and
saving and loading models and results. ``SortedSpikesDetector`` and
``ClusterlessDetector`` specialize it for sorted spike trains and for
unsorted spike waveform features. The model classes in the other modules of
``non_local_detector.models`` configure these two with specific states and
transitions.

``_DetectorBase`` inherits its input validation, checkpointed prediction, EM,
plotting, and saving and loading methods from mixins in the private modules
``_input_validation``, ``_checkpointed``, ``_em``, ``_plotting`` and
``_persistence``. The detector classes stay here: saved detectors name this
module.
"""

import abc
import copy
import inspect
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from patsy import DesignMatrix
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from non_local_detector import _validation as val
from non_local_detector._warnings import user_stacklevel
from non_local_detector.continuous_state_transitions import (
    TransitionContext,
    _check_block_environments,
    _check_continuous_transition,
    _make_transition_block,
    _multi_bin_local_transition,
)
from non_local_detector.core import (
    accepts_row_slice,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
    row_slice_aware,
)
from non_local_detector.discrete_state_transitions import (
    _check_initial_discrete_transition,
    _refuse_untrusted_transition,
    _row_smoothness,
    _smoothness_matrix,
    _transition_intervals,
    predict_discrete_state_transitions,
)
from non_local_detector.encoding_time import EncodingSupport
from non_local_detector.environment import Environment, find_environment_by_name
from non_local_detector.exceptions import (
    CompatibilityWarning,
    ConfigurationError,
    DataQualityWarning,
    NotFittedError,
    ValidationError,
)
from non_local_detector.graph_distances import DEFAULT_MAX_DENSE_DISTANCE_BYTES
from non_local_detector.initial_conditions import _check_state_initial_conditions
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    ClusterlessLikelihoodAlgorithm,
    ClusterlessPredict,
    LikelihoodAlgorithm,
    SortedSpikesFit,
    SortedSpikesPredict,
    _execution_settings,
    _resolve_fit_params,
)
from non_local_detector.likelihoods.common import (
    _SpikeTimeOrder,
    interpolate_weights_at_spike_times,
    resolve_row_slice,
    validate_population_lengths,
    validate_spike_feature_population,
)
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    fit_mark_references,
    mark_reference_log_likelihood,
)
from non_local_detector.likelihoods.no_spike import (
    no_spike_time_bin_sizes,
    predict_no_spike_log_likelihood,
)
from non_local_detector.models._checkpointed import (
    _CheckpointedPredictionMixin,
    _PosteriorOutputs,
)

# EM looks these names up in models._em: patch them there, not here.
from non_local_detector.models._em import (
    _EMMixin,
    _require_observed_bins,
    _resolve_em_options,
    _structured_em_options,
    _validate_encoding_update_damping,
    _validate_estimation_arguments,
)
from non_local_detector.models._input_validation import (
    _TIME_CONTRACT,
    _decode_time_edges,
    _DecodeInputs,
    _EncodingInputs,
    _InputValidationMixin,
    _normalize_frozen_discrete_transition_rows,
    _normalize_return_outputs,
    _require_sampling_frequency_width,
    _sample_mask,
    _sample_times,
    _validate_covariate_time_length,
    _validate_inference_arguments,
    _validate_penalty_params,  # noqa: F401 -- imported from base by tests
    _validate_prior_weight,
    _validate_spike_data,
    _validate_spike_structure,
    _validate_transition_arguments,
)
from non_local_detector.models._persistence import (
    _ModelUnpickler,  # noqa: F401 -- stored-artifact compatibility tests import it from base
    _PersistenceMixin,
    _upgrade_saved_state,
)
from non_local_detector.models._plotting import _PlottingMixin
from non_local_detector.models._results import (
    convert_results_to_xarray,
    convert_seq_to_df,
    has_place_bins,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.result_store import DEFAULT_MAX_READ_BYTES
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _uniform_time_grid,
    _uniformity_tolerance,
    calculate_time_edges,
    requires_time_edges,
)
from non_local_detector.transition_operators import (
    DEFAULT_MAX_DENSE_BYTES,
    _one_bin_block,
)
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    FrozenTransitionRows,
    InferenceMode,
    Observations,
    OutputMode,
    OutputName,
    ReturnOutputs,
    StateNames,
    Stickiness,
    TransitionRepresentation,
)

__all__ = [
    "ClusterlessDetector",
    "SortedSpikesDetector",
]

logger = getLogger(__name__)


@dataclass(frozen=True)
class _LikelihoodPositionContext:
    """Tracking preparation shared only during one likelihood prediction."""

    position: np.ndarray | None
    is_missing: np.ndarray | None
    all_missing: bool

    @classmethod
    def prepare(
        cls,
        time_edges: np.ndarray,
        position_time: np.ndarray | None,
        position: np.ndarray | None,
        *,
        reads_position: bool = True,
        warn_outside_tracking: bool = False,
    ) -> "_LikelihoodPositionContext":
        missing = _missing_bins(
            time_edges,
            None,
            position_time,
            position,
            reads_position=reads_position,
            warn_outside_tracking=warn_outside_tracking,
        )
        prepared = position
        all_missing = False
        if position is not None:
            prepared = np.asarray(position)
            prepared = prepared[:, None] if prepared.ndim == 1 else prepared
            finite = np.all(np.isfinite(prepared), axis=1)
            # A model that never reads position decodes even without any.
            all_missing = reads_position and not np.any(finite)
            if np.any(finite) and not np.all(finite):
                prepared = prepared.copy()
                prepared[~finite] = prepared[finite][0]
        return cls(prepared, missing, all_missing)


@dataclass(frozen=True)
class _DecodeContext:
    """One likelihood evaluation's validated inputs and shared preparation.

    A decoding method builds one per call, from its validated inputs, and
    every row request of that call (each chunk, or the whole recording)
    evaluates ``_DetectorBase._log_likelihood_rows`` with it. A direct
    ``compute_log_likelihood`` call builds its own. Nothing in it is stored
    on the detector or reused by a later call.

    Attributes
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Validated, uniform decode bin edges.
    time_grid : _DecodeTimeGrid
        ``time_edges`` with their uniform width, so row requests do not
        validate the full grid again.
    position_time : np.ndarray, shape (n_time_position,), or None
        Tracking times.
    spike_data : tuple
        The spike arrays named by ``_DetectorBase._SPIKE_DATA_NAMES``.
    position_context : _LikelihoodPositionContext
        Tracking positions prepared on ``time_edges``.
    spike_time_order : _SpikeTimeOrder
        Spike ordering shared across states and row requests.
    no_spike_time_bin_sizes : np.ndarray, shape (n_bins,), or None
        Bin durations shared by every No-Spike row request; None computes
        them per request.
    """

    time_edges: np.ndarray
    time_grid: _DecodeTimeGrid
    position_time: np.ndarray | None
    spike_data: tuple[Any, ...]
    position_context: _LikelihoodPositionContext
    spike_time_order: _SpikeTimeOrder
    no_spike_time_bin_sizes: np.ndarray | None


def _check_likelihood_rows(result: Any, n_rows: int) -> Any:
    """Return ``result`` if it has one row per requested decode bin."""
    if np.shape(result)[0] != n_rows:
        msg = "log likelihood callback returned the wrong number of rows"
        raise ValidationError(
            msg,
            expected=f"{n_rows} rows, one per requested decode bin",
            got=f"{np.shape(result)[0]} rows",
            hint="Likelihoods receive time_edges (n_bins + 1 edges) and "
            "return one row per bin, not one per edge.",
        )
    return result


def _custom_likelihood_callback(
    callback: Callable[..., Any], time_edges: np.ndarray
) -> Callable[..., Any]:
    """Adapt a replaced ``compute_log_likelihood`` to core's row callback contract.

    Core indexes observations by row and hands the callback its row
    coordinates (the bin centers). Detector likelihoods bin spikes against the
    decode edges instead, so the adapter replaces the coordinates with
    ``time_edges``:

    * A ``@row_slice_aware`` callback receives the full ``time_edges``, a
      global ``row_slice``, and an already sliced ``is_missing`` mask, and
      returns only the requested rows.
    * An unmarked callback receives the original full edges. Partial row
      requests are rejected: an edges-only callback cannot represent a chunk's
      open closing boundary without changing its physical duration and center.
      Custom callbacks used with chunking must declare ``row_slice_aware`` and
      forward the global row range to their likelihood implementation.

    Both adapters are marked ``row_slice_aware``, so core hands them the
    chunk's row range rather than only its row coordinates, and both check
    that the callback returned one row per requested bin: a callback written
    for one row per timestamp would otherwise be silently misaligned with the
    bins. The callback receives only ``compute_log_likelihood``'s public
    arguments; it does its own preparation.
    """
    parameters = inspect.signature(callback).parameters
    n_bins = len(time_edges) - 1
    edges_are_keyword_only = (
        "time_edges" in parameters
        and parameters["time_edges"].kind is inspect.Parameter.KEYWORD_ONLY
    )

    def evaluate(edges: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if edges_are_keyword_only:
            return callback(*args, time_edges=edges, **kwargs)
        return callback(edges, *args, **kwargs)

    if accepts_row_slice(callback):

        def row_callback(
            row_time: Any,  # noqa: ARG001 -- likelihood callback signature
            *args: Any,
            row_slice: slice | None = None,
            **kwargs: Any,
        ) -> Any:
            row_start, row_stop = resolve_row_slice(row_slice, n_bins)
            return _check_likelihood_rows(
                evaluate(time_edges, *args, row_slice=row_slice, **kwargs),
                row_stop - row_start,
            )

        row_slice_aware(row_callback)  # marks the function in place
        return row_callback

    def chunk_callback(
        row_time: Any,  # noqa: ARG001 -- likelihood callback signature
        *args: Any,
        row_slice: slice | None = None,
        **kwargs: Any,
    ) -> Any:
        row_start, row_stop = resolve_row_slice(row_slice, n_bins)
        if (row_start, row_stop) != (0, n_bins):
            msg = "Chunked custom likelihoods must be row_slice_aware"
            raise ValidationError(
                msg,
                hint="Decorate the callback with @row_slice_aware from "
                "non_local_detector.core and forward row_slice with the full "
                "time_edges. This preserves durations and counts boundary spikes once.",
            )
        return _check_likelihood_rows(evaluate(time_edges, *args, **kwargs), n_bins)

    row_slice_aware(chunk_callback)  # marks the function in place
    return chunk_callback


def _missing_bins(
    time_edges: np.ndarray,
    is_missing: np.ndarray | None,
    position_time: np.ndarray | None,
    position: np.ndarray | None,
    *,
    reads_position: bool = True,
    warn_outside_tracking: bool = False,
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
    uniformity tolerance. A bin is decoded as missing when any time it owns
    (``[edges[i], edges[i + 1])``, plus ``edges[-1]`` for the final bin) lies
    in such a span, so the center and every spike of a non-missing bin have
    tracked, finite positions. Both rules apply only when the decode
    likelihood reads position (``reads_position``); a model without Local
    states or a non-local position penalty decodes every bin. This concerns
    decoding only: encoding fits use position for their support.

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
    warn_outside_tracking : bool, optional
        Warn when bins are missing because they extend outside the tracked
        period, by default False. Public decode entry points set it, so one
        call warns once however many chunks or EM iterations it runs.

    Returns
    -------
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        None when nothing is missing and no mask was given.

    Warns
    -----
    DataQualityWarning
        If ``warn_outside_tracking`` and some bins extend outside the tracked
        period.

    Raises
    ------
    ValidationError
        If ``is_missing`` does not have one entry per bin, or ``position`` does
        not have one row per ``position_time`` sample.
    DataError
        If ``position_time`` is not finite and non-decreasing.
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
    if position.shape[0] != position_time.shape[0]:
        msg = "position must have one row per position_time sample"
        raise ValidationError(
            msg,
            expected=f"{position_time.shape[0]} rows",
            got=f"{position.shape[0]} rows",
        )
    val.ensure_all_finite(position_time, "position_time")
    val.ensure_monotonic_increasing(position_time, "position_time")
    if not reads_position:
        return is_missing
    affected = _bins_outside_tracking(time_edges, position_time)
    if warn_outside_tracking and np.any(affected):
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
    if not np.any(affected):
        return is_missing
    return affected if is_missing is None else is_missing | affected


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
        Finite and non-decreasing.

    Returns
    -------
    is_outside : np.ndarray of bool, shape (n_bins,)
    """
    n_bins = len(time_edges) - 1
    if position_time.size < 2:
        return np.zeros(n_bins, dtype=bool)
    sample_interval = float(np.median(np.diff(position_time)))
    half_sample = sample_interval / 2 + _uniformity_tolerance(
        position_time, sample_interval
    )
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
    n_bins = len(time_edges) - 1
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


def _group_spike_mask(
    spike_times: np.ndarray,
    position_time: np.ndarray,
    group_weights: np.ndarray,
    encoding_support: EncodingSupport | None = None,
) -> np.ndarray:
    """Select the spikes an encoding group owns.

    A spike belongs to a group exactly when the group's per-sample weights,
    linearly interpolated to the spike time on the full position timeline, are
    positive. Selection does not set the spike's weight: the encoding fit
    re-interpolates the same weights and applies each one once, so a spike near
    a mask transition keeps its fractional weight and complementary groups
    partition every spike. Spikes outside the position timeline are not
    selected, matching the encoding fits' own bounds.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Spike times.
    position_time : np.ndarray, shape (n_time_position,)
        Times of the position samples (the full position timeline).
    group_weights : np.ndarray, shape (n_time_position,)
        Per-sample weights of this group (mask times any EM weight).
    encoding_support : EncodingSupport, optional
        Support of the encoding fit: decides which spikes are in bounds and
        bounds the segments within which ``group_weights`` are interpolated to
        spike times; by default None, which uses the ends of
        ``position_time``.

    Returns
    -------
    is_group_spike : np.ndarray of bool, shape (n_spikes,)
    """
    spike_times = np.asarray(spike_times)
    in_bounds = (
        (spike_times >= position_time[0]) & (spike_times <= position_time[-1])
        if encoding_support is None
        else encoding_support.contains(spike_times)
    )
    spike_weights = interpolate_weights_at_spike_times(
        spike_times, position_time, group_weights, encoding_support=encoding_support
    )
    is_group_spike: np.ndarray = in_bounds & (spike_weights > 0.0)
    return is_group_spike


def _group_weights(
    is_group: np.ndarray,
    weights: np.ndarray | None,
    likelihood_name: tuple[Any, ...],
    encoding_support: EncodingSupport,
) -> np.ndarray:
    """Per-sample weights of one encoding group on the full position timeline.

    The group mask selects the model's samples: a sample outside it
    contributes no occupancy and owns no spikes. Warns when the group has zero
    exposure, i.e. no sample with positive weight also has a finite position
    and time inside the encoding support; positive weights only on NaN
    positions or outside ``encoding_time_range`` count as none. The existing
    rate floors then determine its likelihood, which is usually a mask or
    label mistake.

    Parameters
    ----------
    is_group : np.ndarray of bool, shape (n_time_position,)
        Mask of the samples belonging to the group.
    weights : np.ndarray, shape (n_time_position,), optional
        Extra per-sample weights (e.g. EM local-state weights).
    likelihood_name : tuple
        ``(environment_name, encoding_group)`` of the group.
    encoding_support : EncodingSupport
        Support of the encoding fit; its ``exposure`` is zero for samples with
        NaN position or outside the support.

    Returns
    -------
    group_weights : np.ndarray, shape (n_time_position,)
    """
    group_weights = is_group.astype(float) if weights is None else weights * is_group
    if not np.sum(group_weights * encoding_support.exposure) > 0.0:
        environment_name, encoding_group = likelihood_name
        warnings.warn(
            f"Encoding group {encoding_group!r} in environment "
            f"{environment_name!r} has no training samples with positive "
            "exposure (a positive weight, a finite position and time inside the "
            "encoding support), so its encoding model is fit with zero exposure. "
            "Its likelihood is then set by the rate floor (sorted spikes) or the "
            "background rate (clusterless). Check is_training, "
            "encoding_group_labels, environment_labels, NaN positions, and "
            "encoding_time_range / valid_position_intervals.",
            DataQualityWarning,
            stacklevel=user_stacklevel(),
        )
    return group_weights


class _DetectorBase(
    _InputValidationMixin,
    _CheckpointedPredictionMixin,
    _EMMixin,
    _PlottingMixin,
    _PersistenceMixin,
    BaseEstimator,  # type: ignore[misc]  # sklearn is untyped (BaseEstimator is Any)
    abc.ABC,
):
    """Base class for detector objects.

    Attributes
    ----------
    converged_ : bool
        True if EM converged within max_iter iterations and the final E-step,
        which evaluates the returned parameters, did not decrease the
        penalized log-likelihood. Set by ``estimate_parameters``.
    n_iter_ : int
        Number of EM iterations actually performed. Set by
        ``estimate_parameters``.
    em_monotonicity_violations_ : list[int]
        Iteration indices (1-based) where the penalized log-likelihood
        (marginal log-likelihood plus the discrete-transition log prior)
        decreased; the final E-step counts as iteration ``n_iter_ + 1``.
        Empty list when EM is well-behaved. Set by ``estimate_parameters``.
    degenerate_timesteps_ : np.ndarray
        Time indices (0-based) where the final E-step had all-``-inf``
        log-likelihoods (every state impossible) and the posterior fell back
        to the predicted distribution. Empty array when none occur. Set by
        ``estimate_parameters`` (reflects the training data); ``predict`` does
        not refresh it, though it still logs a degenerate-timestep warning for
        the data it is given.
    discrete_initial_conditions_ : np.ndarray, shape (n_states,)
        Fitted initial-condition distribution over discrete states. Set by
        ``estimate_parameters``; initially equal to the user-supplied
        ``discrete_initial_conditions`` constructor argument.
    """

    # Type annotations for attributes assigned during fit
    discrete_state_transitions_: np.ndarray
    discrete_transition_coefficients_: np.ndarray | None
    discrete_transition_design_matrix_: DesignMatrix | None
    discrete_transition_smoothness_matrix_: np.ndarray | None
    discrete_transition_smoothness_: np.ndarray | None
    discrete_transition_coefficient_covariance_: np.ndarray | None
    # np.ndarray, or a LazyDenseTransition (a budgeted NumPy-compatible view)
    continuous_state_transitions_: Any
    converged_: bool
    n_iter_: int
    em_monotonicity_violations_: list[int]
    degenerate_timesteps_: np.ndarray
    initial_conditions_: np.ndarray
    _max_dense_transition_bytes_: int

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ) -> None:
        """
        Initialize the _DetectorBase class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Rate in Hz of the decode bins; decoding ``time_edges`` must be
            ``1 / sampling_frequency`` wide. By default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Dimensionless weight for data-adaptive prior scaling. When > 0,
            the Dirichlet prior pseudo-counts are scaled by expected transition
            counts, making regularization strength approximately invariant to
            temporal resolution. Can be a scalar (same weight for all rows) or
            an array of shape ``(n_states,)`` for per-row control; each weight
            must be finite and >= 0. When 0.0 (default), the fixed-count prior
            from concentration and stickiness is used directly.
        frozen_discrete_transition_rows : {"no_spike"}, np.ndarray, list, tuple, or None, optional
            Rows of the discrete transition matrix to freeze during EM
            re-estimation. ``"no_spike"`` (default) freezes the rows of the
            states whose observation model has ``is_no_spike=True``, wherever
            they are listed, so a No-Spike self-transition cannot collapse to
            fit stray gaps between spikes; models without a No-Spike state
            freeze nothing. Also accepts integer row indices, a boolean mask
            of shape ``(n_states,)``, or ``None``/empty (no frozen rows).
            Frozen rows are snapshotted from the initial transition matrix
            and restored after every M-step, so they retain the values
            specified by ``discrete_transition_type`` regardless of what the
            data suggests. For covariate-dependent transitions, frozen rows
            keep their initial coefficients, so they stay at their initial
            (time-constant) values.
        local_position_std : float or None, optional
            Controls the Local state's spatial representation:

            - ``None`` (default): legacy behavior. Local state occupies a
              single bin; the likelihood is evaluated at the animal's
              exact interpolated position (continuous).
            - ``0.0``: delta-kernel multi-bin local. Local state spans all
              position bins, with all mass concentrated at the single bin
              containing the animal (one-hot). Likelihood is evaluated at
              bin centers (discrete).
            - ``> 0``: multi-bin local with a Gaussian kernel of standard
              deviation ``local_position_std`` (same units as ``position``,
              typically centimeters) on the shortest-path track-graph
              distance; models spatial uncertainty.
        """
        # A list or tuple is stored as an array, which fitting indexes; an
        # array is stored as given, as ``sklearn.clone`` requires.
        discrete_initial_conditions = np.asarray(discrete_initial_conditions)
        # Validate all parameters early
        self._validate_initial_conditions(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            continuous_transition_types,
            discrete_transition_stickiness,
        )
        self._validate_probability_distributions(discrete_initial_conditions)
        self._validate_numerical_parameters(
            discrete_transition_concentration,
            discrete_transition_regularization,
            sampling_frequency,
            no_spike_rate,
        )
        self._validate_discrete_transition_type(
            discrete_transition_type, len(discrete_initial_conditions)
        )
        _validate_prior_weight(
            discrete_transition_prior_weight, len(discrete_initial_conditions)
        )

        # Initial conditions parameters
        self.discrete_initial_conditions = discrete_initial_conditions
        self.continuous_initial_conditions_types = continuous_initial_conditions_types

        # Discrete state transition parameters
        self.discrete_transition_concentration = discrete_transition_concentration
        self.discrete_transition_stickiness = discrete_transition_stickiness
        self.discrete_transition_regularization = discrete_transition_regularization
        self.discrete_transition_type = discrete_transition_type
        self.discrete_transition_prior_weight = discrete_transition_prior_weight
        self.frozen_discrete_transition_rows = frozen_discrete_transition_rows

        # Continuous state transition parameters
        self.continuous_transition_types = continuous_transition_types

        # Environment parameters
        self.environments = self._initialize_environments(environments)
        self.infer_track_interior = infer_track_interior

        # Observation model parameters
        self.observation_models = self._initialize_observation_models(
            observation_models, continuous_transition_types, self.environments
        )
        self._validate_observation_models(
            self.observation_models, len(discrete_initial_conditions)
        )
        # Validate the frozen-row spec against the states up front (raises on
        # bad input). The normalized boolean mask itself is derived lazily via
        # the ``_frozen_discrete_transition_rows_mask_`` property rather than
        # stored as an instance attribute, so it stays out of ``vars()`` /
        # ``get_params()`` and does not break serialization round-trips that
        # reconstruct the model from its public parameters (e.g. Spyglass's
        # DecodingParameters).
        _ = self._frozen_discrete_transition_rows_mask_

        # State names
        self.state_names = self._initialize_state_names(
            state_names, discrete_initial_conditions
        )

        self.sampling_frequency = sampling_frequency
        self.no_spike_rate = no_spike_rate

        # Local position uncertainty parameter. Reject invalid widths up front:
        # non-finite (NaN/Inf slip past a bare ``< 0`` check), negative, or a
        # positive value whose float32 square underflows to 0 -- the kernel
        # evaluates ``-0.5 * d**2 / sigma**2`` in float32 and JAX FTZ flushes
        # subnormals to 0, a silent divide-by-zero.
        if local_position_std is not None:
            if not np.isfinite(local_position_std):
                msg = "local_position_std must be finite"
                raise ValidationError(
                    msg,
                    expected="finite float >= 0 or None",
                    got=str(local_position_std),
                    hint=(
                        "Use None for legacy single-bin local, 0.0 for a delta "
                        "kernel at the animal's bin, or a finite positive value "
                        "for the Gaussian anchor-kernel width."
                    ),
                )
            if local_position_std < 0:
                msg = "local_position_std must be non-negative"
                raise ValidationError(
                    msg,
                    expected="float >= 0 or None",
                    got=str(local_position_std),
                    hint=(
                        "Use None for legacy single-bin local (likelihood at "
                        "the animal's exact interpolated position), 0.0 for "
                        "multi-bin local with a delta kernel at the animal's "
                        "bin, or > 0 for a Gaussian kernel."
                    ),
                )
            f32_min_sigma = float(np.sqrt(np.finfo(np.float32).tiny))
            if 0 < float(local_position_std) < f32_min_sigma:
                msg = "local_position_std**2 underflows to 0 in float32"
                raise ValidationError(
                    msg,
                    expected=(
                        "0.0, or a positive value with a representable float32 "
                        f"square (>= sqrt(float32 tiny) ~{f32_min_sigma:.3e})"
                    ),
                    got=str(local_position_std),
                    hint=(
                        "Pass 0.0 explicitly for the delta kernel, or use a "
                        f"value >= {f32_min_sigma:.3e} so sigma**2 is "
                        "representable as a normal float32."
                    ),
                )
        self.local_position_std = local_position_std

    def set_params(self, **params: Any) -> "_DetectorBase":
        """Set constructor parameters, applying the constructor's rules.

        The merged parameters (current values updated with ``params``) are
        passed through the model's own constructor, so defaults resolution,
        normalization (``environments`` and ``observation_models`` become
        tuples, ``state_names=None`` becomes generated names) and validation are
        exactly those of ``__init__``. The new values are applied only if the
        constructor accepts them; on error the detector is unchanged. Fitted
        attributes, including the fitted environments in ``environments_``,
        are not touched. Changing ``environments``, ``state_names``,
        ``observation_models`` or any other parameter a fit depends on after
        fitting therefore requires refitting.

        As with the constructor, parameters that must agree (for example
        ``discrete_initial_conditions``, ``observation_models``,
        ``continuous_transition_types`` and ``state_names``, which each have
        one entry per discrete state) must be changed together.

        Parameters
        ----------
        **params : Any
            Constructor parameters to change. Nested ``component__parameter``
            keys are validated first, then handled by scikit-learn after the
            top-level ones.

        Returns
        -------
        self : _DetectorBase
            The detector with the updated parameters.

        Raises
        ------
        ValidationError
            If the resulting parameters are invalid, as in the constructor, or
            a parameter name is not a constructor parameter.
        """
        if not params:
            return self
        current = self.get_params(deep=False)
        top_level = {k: v for k, v in params.items() if "__" not in k}
        nested = {k: v for k, v in params.items() if "__" in k}
        # Validate every name before applying anything, so any error leaves the
        # detector unchanged.
        invalid = sorted(
            [k for k in top_level if k not in current]
            + [
                k
                for k in nested
                if not hasattr(current.get(k.split("__", 1)[0]), "set_params")
            ]
        )
        if invalid:
            msg = (
                f"Invalid parameter(s) {invalid!r} for estimator "
                f"{type(self).__name__}. Valid parameters are: "
                f"{sorted(current)!r}."
            )
            raise ValidationError(msg)
        if top_level:
            # Build a fresh detector first so a rejected value leaves ``self``
            # untouched.
            validated = type(self)(**{**current, **top_level})
            resolved = validated.get_params(deep=False)
            for name in current:
                setattr(self, name, resolved[name])
        if nested:
            super().set_params(**nested)
        return self

    @property
    def _frozen_discrete_transition_rows_mask_(self) -> np.ndarray | None:
        """Boolean mask of frozen discrete-transition rows (derived, lazy).

        Recomputed from ``frozen_discrete_transition_rows`` on each access
        rather than stored as an instance attribute, so it does not appear in
        ``vars()`` / ``get_params()``. This keeps serialization round-trips that
        reconstruct the model from its public parameters (e.g. Spyglass's
        ``DecodingParameters``) from passing this derived attribute back into
        ``__init__`` as an unexpected keyword argument.

        Returns
        -------
        np.ndarray of bool, shape (n_states,), or None
            ``None`` when no rows are frozen.
        """
        return _normalize_frozen_discrete_transition_rows(
            self.frozen_discrete_transition_rows,
            np.array([obs.is_no_spike for obs in self.observation_models]),
        )

    def _non_local_position_penalty(self) -> float:
        """Non-local position penalty, in nats; 0.0 for models without one.

        The non-local detectors override this to return their validated
        ``non_local_position_penalty``.

        Returns
        -------
        penalty : float
        """
        return 0.0

    def _decode_reads_position(self) -> bool:
        """Whether the decode likelihood reads the animal's position.

        Local observation models and a positive ``non_local_position_penalty``
        read it; other models ignore position at decode (``local_position_std``
        only shapes Local states). Fitting and EM encoding refits always use
        position for the encoding support, whatever this returns.

        Returns
        -------
        reads_position : bool
        """
        return bool(
            any(obs.is_local for obs in self.observation_models)
            or self._non_local_position_penalty() > 0
        )

    def _compute_non_local_position_penalty(
        self,
        time: np.ndarray,
        position_time: np.ndarray,
        position: np.ndarray,
        environment: "Environment",
    ) -> jax.Array:
        """Compute position-dependent penalty for non-local states.

        Returns a ``(n_time, n_interior_bins)`` array of log-likelihood
        penalties that suppress non-local likelihood near the animal's
        current position. The penalty is a negative Gaussian centered on
        the animal's position.

        Uses topology-aware distances via
        :meth:`Environment.get_distances_to_interior_bins`:

        - 1D track graph: shortest-path distance along the linearized
          track (handles multi-arm mazes correctly so the penalty at a
          bin reflects its graph distance, not its linearized coordinate
          difference).
        - N-D environment: shortest-path distance on ``track_graphDD``
          (respects holes and obstacles — the penalty follows the
          reachable interior, not straight-line Euclidean).
        - No precomputed distance matrix: Euclidean fallback.

        NaN/inf distances (off-track animal position, unreachable bins)
        are treated as infinite distance so the penalty at those bins is
        zero — consistent with the intuition "cannot compute → no
        penalty applied".

        Parameters
        ----------
        time : jax.Array, shape (n_bins,)
            Decode bin centers at which the animal's position is evaluated.
        position_time : jax.Array, shape (n_time_position,)
            Position sampling times.
        position : jax.Array, shape (n_time_position, n_position_dims)
            Position samples.
        environment : Environment
            The spatial environment with place_bin_centers_ and
            is_track_interior_.

        Returns
        -------
        penalty : jax.Array, shape (n_time, n_interior_bins)
            Non-positive penalty values (to be added to log-likelihood).
        """
        from non_local_detector.likelihoods.common import get_position_at_time

        animal_pos = get_position_at_time(position_time, position, time, environment)
        animal_pos_arr = np.asarray(animal_pos)

        # Detect NaN animal positions (tracking dropouts). get_bin_ind does
        # not guard against NaN — np.searchsorted(nan) returns an arbitrary
        # valid bin index, so a NaN position would produce a finite distance
        # and thus a spurious penalty at an unrelated bin. Mask those rows
        # to zero after computing the penalty.
        if animal_pos_arr.ndim == 1:
            nan_mask = jnp.isnan(jnp.asarray(animal_pos_arr))
        else:
            nan_mask = jnp.any(jnp.isnan(jnp.asarray(animal_pos_arr)), axis=-1)

        # Topology-aware distance (graph shortest path when available).
        # Replace NaN animal positions with 0 before the lookup to avoid
        # undefined searchsorted behavior; nan_mask masks those rows below.
        safe_animal_pos = np.nan_to_num(animal_pos_arr, nan=0.0)
        dist = environment.get_distances_to_interior_bins(safe_animal_pos)

        # NaN (off-track bin-to-bin lookups) and inf (unreachable bins) →
        # treat as infinitely far so the penalty evaluates to zero at those
        # cells, avoiding NaN propagation into the log-likelihood.
        sq_dist = jnp.nan_to_num(jnp.asarray(dist) ** 2, nan=jnp.inf, posinf=jnp.inf)

        penalty = -self.non_local_position_penalty * jnp.exp(
            -0.5 * sq_dist / (self.non_local_penalty_std**2)
        )

        # NaN animal positions produce zero penalty (no constraint during
        # tracking dropouts) rather than a spurious penalty at a searchsorted-
        # dependent bin.
        return cast("jax.Array", jnp.where(nan_mask[:, jnp.newaxis], 0.0, penalty))

    def _compute_local_position_kernel(
        self,
        time: jax.Array | np.ndarray,
        position_time: jax.Array | np.ndarray,
        position: jax.Array | np.ndarray,
        environment: "Environment",
    ) -> jax.Array:
        """Compute log position kernel for the local state.

        Dispatches to one of two paths based on ``self.local_position_std``:

        - ``== 0``: delta kernel. ``log(n_bins)`` at the animal's bin,
          ``-inf`` elsewhere (one-hot).
        - ``> 0``: Gaussian kernel over shortest-path track-graph
          distance (Euclidean fallback when no graph is fitted), then
          normalized so ``exp(log_kernel)`` sums to ``n_interior_bins``
          per time step. Unreachable bins on disconnected graph
          components get zero probability naturally via ``exp(-inf)``.

        Gap-bin positions (e.g. positions exactly on an arm-boundary
        edge of a linearized track) are snapped to the nearest interior
        bin by :meth:`Environment.get_bin_ind`, so the kernel is always
        defined on a real interior bin.

        NaN animal positions (tracking dropouts) fall back to a uniform
        kernel.

        Both paths satisfy the invariant ``exp(log_kernel).sum(axis=1)
        == n_bins``. Combined with the multi-bin local state's
        ``1/n_bins`` uniform continuous initial conditions, this makes
        the effective per-bin prior exactly ``exp(log_kernel)`` — so
        the total Local state mass after the HMM forward step equals
        ``p_local * P(spikes | animal_bin)`` in the delta limit.

        In the sharp-sigma limit, the Gaussian concentrates at the
        animal's bin and matches the delta-kernel behavior numerically.

        Parameters
        ----------
        time : jax.Array, shape (n_bins,)
            Decode bin centers at which the animal's position is evaluated.
        position_time : jax.Array, shape (n_time_position,)
            Position sampling times.
        position : jax.Array, shape (n_time_position, n_position_dims)
            Position samples.
        environment : Environment
            The spatial environment.

        Returns
        -------
        log_kernel : jax.Array, shape (n_time, n_interior_bins)
            Log kernel. exp(log_kernel) sums to n_interior_bins per
            time step (not 1).
        """
        from non_local_detector.likelihoods.common import get_position_at_time

        assert self.local_position_std is not None  # narrowing for type checker

        animal_pos = get_position_at_time(position_time, position, time, environment)

        n_bins = int(environment.is_track_interior_.sum())
        log_n_bins = jnp.log(jnp.array(n_bins, dtype=jnp.float32))

        # Detect NaN positions (from gaps in position data)
        if animal_pos.ndim == 1:
            nan_mask = jnp.isnan(animal_pos)
        else:
            nan_mask = jnp.any(jnp.isnan(animal_pos), axis=-1)

        # Delta kernel path: σ=0 concentrates the Local state on the
        # single bin containing the animal. Emits log(n_bins) at that bin
        # and -inf elsewhere, so after the multi-bin state's 1/n_bins
        # uniform continuous IC, the total Local mass equals p_local ×
        # P(spikes | animal_bin_center). Environment.get_bin_ind snaps
        # gap-bin assignments to the nearest interior bin, so the only
        # remaining case needing uniform fallback is a truly NaN animal
        # position (tracking dropout).
        if self.local_position_std == 0:
            interior_bin_indices = np.where(environment.is_track_interior_.ravel())[0]
            # Replace NaN animal positions with 0 before get_bin_ind (it
            # has no NaN guard); nan_mask below masks those rows.
            safe_animal_pos = np.nan_to_num(np.asarray(animal_pos), nan=0.0)
            animal_bin_inds = environment.get_bin_ind(safe_animal_pos)
            is_animal_bin = jnp.asarray(
                animal_bin_inds[:, np.newaxis] == interior_bin_indices[np.newaxis, :]
            )
            log_kernel = jnp.where(is_animal_bin, log_n_bins, -jnp.inf)
            return jnp.where(nan_mask[:, jnp.newaxis], 0.0, log_kernel)

        # Gaussian kernel path (σ > 0). Uses topology-aware distance via
        # Environment.get_distances_to_interior_bins(). After the
        # get_bin_ind snap, the animal bin is always interior, so the
        # distance-matrix row is always finite (no NaN rows). Unreachable
        # bins on disconnected graph components stay as inf, which through
        # exp(-0.5 * inf² / σ²) naturally yields zero probability there —
        # the Gaussian concentrates on the reachable interior.
        dist = environment.get_distances_to_interior_bins(np.asarray(animal_pos))
        sq_dist = jnp.asarray(dist) ** 2

        # Compute log-kernel with -inf at unreachable bins. jnp.where
        # avoids inf/NaN propagation through the arithmetic ops below.
        reachable = jnp.isfinite(sq_dist)
        log_kernel = jnp.where(
            reachable, -0.5 * sq_dist / (self.local_position_std**2), -jnp.inf
        )
        # Normalize to sum to n_bins (not 1) per time step. This compensates
        # for the 1/n_bins uniform continuous IC of the multi-bin local state
        # so the effective per-bin prior matches the kernel itself. Without
        # this scaling, non-local states dominate by a factor of n_bins.
        log_kernel = (
            log_kernel
            - jax.scipy.special.logsumexp(log_kernel, axis=1, keepdims=True)
            + log_n_bins
        )

        # NaN animal positions (tracking dropout) fall back to a uniform
        # kernel: each bin's exp value is 1, so log_kernel = 0 per bin.
        return jnp.where(nan_mask[:, jnp.newaxis], 0.0, log_kernel)

    def _initialize_environments(
        self, environments: Environments
    ) -> tuple[Environment, ...]:
        """
        Initialize environments.

        Parameters
        ----------
        environments : Environments
            Environments in which the detector operates.

        Returns
        -------
        tuple[Environment, ...]
            Initialized environments as a tuple. A tuple is returned as is, so
            the parameter round-trips through scikit-learn's ``clone``.
        """
        normalized: Any = environments
        if normalized is None:
            normalized = (Environment(),)
        if not hasattr(normalized, "__iter__"):
            normalized = (normalized,)
        if not isinstance(normalized, tuple):
            normalized = tuple(normalized)
        return cast("tuple[Environment, ...]", normalized)

    def _initialize_observation_models(
        self,
        observation_models: Observations,
        continuous_transition_types: ContinuousTransitions,
        environments: tuple[Environment, ...],
    ) -> tuple[ObservationModel, ...]:
        """
        Initialize observation models.

        Parameters
        ----------
        observation_models : Observations
            Observation models for the detector.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        environments : tuple[Environment, ...]
            Normalized environments in which the detector operates.

        Returns
        -------
        tuple[ObservationModel, ...]
            Initialized observation models as a tuple.
        """
        normalized: Any = observation_models
        if normalized is None:
            n_states = len(continuous_transition_types)
            env_name = environments[0].environment_name
            normalized = (ObservationModel(env_name),) * n_states
        elif isinstance(normalized, ObservationModel):
            normalized = (normalized,) * len(continuous_transition_types)
        return cast("tuple[ObservationModel, ...]", normalized)

    def _initialize_state_names(
        self, state_names: StateNames, discrete_initial_conditions: np.ndarray
    ) -> list[str]:
        """
        Initialize state names.

        Parameters
        ----------
        state_names : StateNames, optional
            Names of the states.
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states.

        Returns
        -------
        state_names : list[str]

        Raises
        ------
        ValueError
            If the number of state names does not match the number of discrete initial conditions.
        """
        if state_names is None:
            state_names = [
                f"state {state_ind}"
                for state_ind in range(len(discrete_initial_conditions))
            ]
        if len(state_names) != len(discrete_initial_conditions):
            msg = "Number of state names must match number of states"
            raise ValidationError(
                msg,
                expected=f"{len(discrete_initial_conditions)} state name(s)",
                got=f"{len(state_names)} state name(s)",
                hint="Provide one name per state in your model",
                example=f"    state_names={state_names[: len(discrete_initial_conditions)]}  # truncate to {len(discrete_initial_conditions)}",
            )
        return state_names

    def initialize_environments(
        self,
        position: np.ndarray,
        environment_labels: np.ndarray | None = None,
        *,
        compute_all_pairs_distances: bool = True,
        max_dense_distance_bytes: int = DEFAULT_MAX_DENSE_DISTANCE_BYTES,
    ) -> None:
        """
        Fit copies of the environments on the position data.

        Each environment is rebuilt from its configuration
        (``Environment.get_params``) and the copy is fitted, so the
        environments passed to the model, which other models may share, are
        never changed. ``self.environments`` keeps the configuration and
        ``self.environments_`` holds the fitted copies afterwards.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        environment_labels : np.ndarray, optional, shape (n_time,)
            Labels for each time points about which environment it corresponds to, by default None.
        compute_all_pairs_distances : bool, optional
            By default True, which builds each environment's all-pairs graph
            distances. False keeps an exact sparse query view for Cartesian
            environments; see ``Environment.fit_place_grid``.
        max_dense_distance_bytes : int, optional
            Byte limit for dense reads of the deferred distance view, by
            default 256 MiB. Used only when compute_all_pairs_distances=False.
        """
        fitted_environments = []
        for given_environment in self.environments:
            environment = type(given_environment)(
                **copy.deepcopy(given_environment.get_params())
            )
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name

            env_position = position[is_environment]
            env_position = environment.decoding_position(
                env_position[np.all(np.isfinite(env_position), axis=1)]
            )

            distance_options: dict[str, Any] = (
                {}
                if compute_all_pairs_distances
                else {
                    "compute_all_pairs_distances": False,
                    "max_dense_distance_bytes": max_dense_distance_bytes,
                }
            )
            environment.fit_place_grid(
                env_position,
                infer_track_interior=self.infer_track_interior,
                **distance_options,
            )
            fitted_environments.append(environment)
        self.environments_ = tuple(fitted_environments)

    def _initialize_state_index(self) -> None:
        """Initialize indices and parameters related to the combined state space.

        Determines the total number of bins across all discrete states (spatial
        bins for continuous states, 1 for discrete states like 'Local' or
        'No-Spike') and creates mappings between these combined bins and the
        original discrete states. Also identifies which combined bins correspond
        to the track interior.

        Attributes
        ----------
        n_discrete_states_ : int
            Total number of discrete states defined in the model.
        state_ind_ : np.ndarray, shape (n_total_bins,)
            Index mapping each combined bin to its corresponding discrete state index.
        n_state_bins_ : int
            Total number of bins across all states (sum of spatial bins for
            continuous states and 1 for discrete states). Referred to as
            `n_total_bins` in the shape description here for clarity.
        bin_sizes_ : np.ndarray, shape (n_discrete_states_,)
             Number of bins associated with each discrete state (e.g., number
             of place bins for spatial states, 1 for non-spatial states).
        is_track_interior_state_bins_ : np.ndarray, shape (n_total_bins,)
             Boolean array indicating if a combined state bin corresponds to the
             track interior. For non-spatial states, this is typically True.
        """
        self.n_discrete_states_ = len(self.state_names)
        bin_sizes = []
        state_ind = []
        is_track_interior = []
        for ind, obs in enumerate(self.observation_models):
            if not has_place_bins(self, obs):
                bin_sizes.append(1)
                state_ind.append(np.full(1, ind, dtype=int))
                is_track_interior.append(np.ones(1, dtype=bool))
            else:
                environment = self._get_environment_by_name(obs.environment_name)
                bin_sizes.append(environment.place_bin_centers_.shape[0])
                state_ind.append(np.full(bin_sizes[-1], ind, dtype=int))
                is_track_interior.append(environment.is_track_interior_flat_)

        self.state_ind_ = np.concatenate(state_ind)
        self.n_state_bins_ = len(self.state_ind_)
        self.bin_sizes_ = np.array(bin_sizes)
        self.is_track_interior_state_bins_ = np.concatenate(is_track_interior)

    def _initialize_initial_conditions(self) -> None:
        """Construct the initial probability for the state and each spatial bin.

        Attributes
        ----------
        continuous_initial_conditions_ : np.ndarray, shape (n_state_bins,)
            Initial probability distribution over the bins within each state.
        initial_conditions_ : np.ndarray, shape (n_state_bins,)
            Overall initial probability distribution across all state bins.
        """
        logger.info("Fitting initial conditions...")
        from dataclasses import replace

        ic_parts = []
        for state, (obs, cont_ic) in enumerate(
            zip(
                self.observation_models,
                self.continuous_initial_conditions_types,
                strict=False,
            )
        ):
            # Multi-bin local uses full spatial initial conditions
            effective_obs = obs
            if obs.is_local and self.local_position_std is not None:
                effective_obs = replace(obs, is_local=False)
            ic_parts.append(
                cont_ic.make_initial_conditions(effective_obs, self.environments_)
            )
            _check_state_initial_conditions(
                ic_parts[-1],
                state_name=self.state_names[state],
                initial_conditions=cont_ic,
                is_interior=self.is_track_interior_state_bins_[
                    self.state_ind_ == state
                ],
                discrete_weight=self.discrete_initial_conditions[state],
            )
        self.continuous_initial_conditions_ = np.concatenate(ic_parts)
        self.initial_conditions_ = (
            self.continuous_initial_conditions_
            * self.discrete_initial_conditions[self.state_ind_]
        )

    def _initialize_continuous_state_transition(
        self,
        continuous_transition_types: ContinuousTransitions,
        position: np.ndarray | None = None,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        *,
        transition_representation: TransitionRepresentation = "dense",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> None:
        """
        Construct the transition matrices for the continuous states.

        Parameters
        ----------
        continuous_transition_types : ContinuousTransitions
            Types of transition models.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        is_training : np.ndarray, optional, shape (n_time,)
            Boolean array that determines what data to train the place fields on, by default None.
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name, by default None.
        environment_labels : np.ndarray, shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None.
        transition_representation : {"dense", "structured", "auto"}, optional
            By default "dense", which builds the full matrix. "structured"
            requires exact supported state-block operators; "auto" permits
            budgeted dense fallback blocks.
        max_dense_transition_bytes : int, optional
            Byte limit for retained fallback blocks, temporary dense graph
            distances, and explicit reads of the lazy view, by default
            256 MiB. Does not limit "dense".

        Attributes
        ----------
        continuous_state_transitions_ : np.ndarray or LazyDenseTransition, shape (n_state_bins, n_state_bins)
            Probability of transitioning between bins, assuming a transition between the corresponding discrete states occurs.
            A budgeted lazy view of the operator unless transition_representation="dense".
        continuous_transition_types : ContinuousTransitions
            Stores the continuous transition types used.
        """
        logger.info("Fitting continuous state transition...")

        self.continuous_transition_types = continuous_transition_types

        _validate_transition_arguments(
            transition_representation, max_dense_transition_bytes
        )
        self._transition_representation_ = transition_representation
        self._max_dense_transition_bytes_ = int(max_dense_transition_bytes)
        self.__dict__.pop("_continuous_transition_operator_", None)
        context = TransitionContext(
            position, is_training, encoding_group_labels, environment_labels
        )
        if transition_representation != "dense":
            from non_local_detector.transition_operators import (
                LazyDenseTransition,
                build_transition_operator,
            )

            operator = build_transition_operator(
                self.continuous_transition_types,
                self.environments_,
                self.bin_sizes_,  # type: ignore[arg-type]  # typed Sequence[int] in transition_operators; an ndarray works
                observation_models=self.observation_models,
                local_position_std=self.local_position_std,
                allow_dense_fallback=transition_representation == "auto",
                max_dense_bytes=int(max_dense_transition_bytes),
                context=context,
            )
            self._continuous_transition_operator_ = operator
            self.continuous_state_transitions_ = LazyDenseTransition(
                operator, max_bytes=int(max_dense_transition_bytes)
            )
            return

        n_total_bins = len(self.state_ind_)
        self.continuous_state_transitions_ = np.zeros((n_total_bins, n_total_bins))

        for from_state, row in enumerate(self.continuous_transition_types):
            for to_state, transition in enumerate(row):
                _check_continuous_transition(transition)
                inds = np.ix_(
                    self.state_ind_ == from_state, self.state_ind_ == to_state
                )

                transition = _multi_bin_local_transition(
                    transition,
                    self.observation_models[from_state],
                    self.observation_models[to_state],
                    self.local_position_std,
                )

                n_row_bins = int(np.max(inds[0].shape))
                n_col_bins = int(np.max(inds[1].shape))

                # Blocks between a one-bin and a spatial state follow the
                # structured operator's rule, so both representations agree.
                _check_block_environments(
                    transition,
                    self.environments_,
                    self.observation_models[from_state].environment_name,
                    self.observation_models[to_state].environment_name,
                    n_row_bins,
                    n_col_bins,
                )
                one_bin_block = _one_bin_block(
                    transition, self.environments_, n_row_bins, n_col_bins
                )
                if one_bin_block is not None:
                    self.continuous_state_transitions_[inds] = one_bin_block.entries(
                        np.arange(n_row_bins)[:, np.newaxis],
                        np.arange(n_col_bins)[np.newaxis],
                    )
                else:
                    self.continuous_state_transitions_[inds] = _make_transition_block(
                        transition,
                        self.environments_,
                        context,
                        (n_row_bins, n_col_bins),
                    )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle without the patsy design matrix, which cannot be pickled.

        ``__setstate__`` rebuilds it from the fitted covariates.
        """
        state: dict[str, Any] = super().__getstate__()
        if state.get("discrete_transition_design_matrix_") is not None:
            state = dict(state)
            state["discrete_transition_design_matrix_"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        _upgrade_saved_state(self, state)

    def _initialize_discrete_state_transition(
        self, covariate_data: pd.DataFrame | dict[str, Any] | None = None
    ) -> None:
        """
        Construct the transition matrix for the discrete states.

        Parameters
        ----------
        covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Attributes
        ----------
        discrete_state_transitions_ : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
            Probability of transitioning between discrete states. Shape is (n_states, n_states) if no covariates, otherwise depends on covariate data.
        discrete_transition_coefficients_ : np.ndarray or None
            Fitted coefficients for covariate-dependent transitions.
        discrete_transition_design_matrix_ : patsy.DesignMatrix or None
            Design matrix information for covariate-dependent transitions.
        discrete_transition_smoothness_matrix_ : np.ndarray or None
            Curvature penalty of each of the formula's ``quantile_bs`` and
            ``cyclic_bs`` terms, shape (n_terms, n_coefficients,
            n_coefficients), for covariate-dependent transitions.
        discrete_transition_smoothness_ : np.ndarray or None
            Curvature-penalty strength of each row for each spline term,
            shape (n_states, n_terms), for covariate-dependent transitions.
            With ``smoothness="reml"`` the strengths REML chose in the last
            EM M-step.
        discrete_transition_coefficient_covariance_ : np.ndarray or None
            Laplace covariance of each row's flattened coefficients, shape
            (n_states, n_parameters, n_parameters), from the last EM M-step
            of a covariate-dependent transition; None before one.
        """
        logger.info("Fitting discrete state transition")
        transition_type = self.discrete_transition_type
        initial = transition_type.make_state_transition(covariate_data)
        _check_initial_discrete_transition(initial, transition_type)
        self.discrete_state_transitions_ = initial.transition_matrix
        self.discrete_transition_coefficients_ = initial.coefficients
        self.discrete_transition_design_matrix_ = initial.design_matrix
        # patsy design matrices cannot be pickled or deep-copied; keep a copy
        # of the covariates so ``__setstate__`` can rebuild the design matrix
        # with the fitted knots even if the caller later changes them.
        self._discrete_transition_covariate_data_ = (
            None if transition_type.is_stationary else copy.deepcopy(covariate_data)
        )
        # Curvature penalty of each of the formula's spline terms, the
        # strengths of each row's penalties (fixed, or REML's starting value until an
        # M-step chooses it) and, once an M-step has fit them, the Laplace
        # covariance of each row's coefficients.
        self.discrete_transition_coefficient_covariance_ = None
        if transition_type.is_stationary:
            self.discrete_transition_smoothness_matrix_ = None
            self.discrete_transition_smoothness_ = None
        else:
            # A covariate-dependent transition has a design matrix and
            # coefficients (checked above).
            assert self.discrete_transition_design_matrix_ is not None
            assert self.discrete_transition_coefficients_ is not None
            self.discrete_transition_smoothness_matrix_ = _smoothness_matrix(
                self.discrete_transition_design_matrix_.design_info
            )
            self.discrete_transition_smoothness_ = _row_smoothness(
                transition_type.smoothness,
                None,
                self.discrete_transition_smoothness_matrix_,
                self.discrete_transition_coefficients_.shape[1],
            )[0]

        # Snapshot frozen rows of a stationary matrix so the EM M-step can
        # restore them after every update. Covariate-dependent frozen rows
        # instead keep their coefficients: the M-step skips them.
        if (
            self._frozen_discrete_transition_rows_mask_ is not None
            and transition_type.is_stationary
        ):
            self._frozen_discrete_transition_rows_baseline_ = (
                self.discrete_state_transitions_[
                    self._frozen_discrete_transition_rows_mask_
                ].copy()
            )
        else:
            self._frozen_discrete_transition_rows_baseline_ = None

    def discrete_transition_intervals(
        self,
        covariate_data: pd.DataFrame | dict[str, Any],
        coverage: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fitted covariate-dependent transitions with pointwise intervals.

        Parameters
        ----------
        covariate_data : dict or pd.DataFrame
            Covariates to evaluate the transitions at, e.g.
            ``{"speed": np.linspace(0, 60, 100)}``.
        coverage : float, optional
            Pointwise coverage of the intervals, by default 0.95.

        Returns
        -------
        transitions, lower, upper : np.ndarray, shape (n_new_time, n_states, n_states)
            The fitted transition probabilities and their Laplace
            (delta-method) intervals. The intervals treat the last M-step's
            expected transitions and smoothness as known, so they leave out
            the uncertainty in which state the animal was in. Frozen rows are
            not estimated and have zero-width intervals; rows whose
            coefficients are not identified (e.g. a state without expected
            transitions) have NaN intervals. See
            :func:`~non_local_detector.discrete_state_transitions.predict_discrete_state_transition_intervals`.

        Raises
        ------
        ConfigurationError
            If the transitions do not depend on covariates.
        NotFittedError
            If EM has not estimated the transitions (no covariance yet).
        ValidationError
            If the transition formula was read from a saved model file and
            not trusted (see ``load_model``).
        """
        _refuse_untrusted_transition(
            self.discrete_transition_type, "compute transition intervals"
        )
        if self.discrete_transition_design_matrix_ is None:
            msg = "The discrete transitions do not depend on covariates."
            raise ConfigurationError(
                msg,
                hint=(
                    "Use a DiscreteNonStationaryCustom or "
                    "DiscreteNonStationaryDiagonal transition type."
                ),
            )
        if self.discrete_transition_coefficient_covariance_ is None:
            msg = "No transition covariance yet: it comes from an EM M-step."
            raise NotFittedError(
                msg,
                hint=(
                    "Run estimate_parameters with "
                    "estimate_discrete_transition=True and max_iter >= 1."
                ),
            )
        # The coefficients are set together with the design matrix.
        assert self.discrete_transition_coefficients_ is not None
        return _transition_intervals(
            self.discrete_transition_design_matrix_,
            self.discrete_transition_coefficients_,
            self.discrete_transition_coefficient_covariance_,
            covariate_data,
            coverage,
        )

    def _fit(
        self,
        position: np.ndarray,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        *,
        transition_representation: TransitionRepresentation = "dense",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> "_DetectorBase":
        """
        Fit the model to the data.

        The public ``fit`` methods coerce and check the arrays first (see
        ``_EncodingInputs``).

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        is_training : np.ndarray, optional, shape (n_time,)
            Boolean array that determines what data to train the place fields on, by default None.
        encoding_group_labels : np.ndarray, optional, shape (n_time,)
            If place fields should correspond to each state, label each time point with the group name, by default None.
        environment_labels : np.ndarray, optional, shape (n_time,)
            If there are multiple environments, label each time point with the environment name, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        transition_representation : {"dense", "structured", "auto"}, optional
            By default "dense", which builds the full matrix. "structured"
            requires exact supported state-block operators; "auto" permits
            budgeted dense fallback blocks.
        max_dense_transition_bytes : int, optional
            Byte limit for retained fallback blocks, temporary dense graph
            distances, and explicit reads of the lazy view, by default
            ``DEFAULT_MAX_DENSE_BYTES``. Does not limit "dense".

        Returns
        -------
        _DetectorBase
            Fitted model.
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

        position = position[:, np.newaxis] if position.ndim == 1 else position
        # Before anything changes, so a refused fit keeps the loaded model.
        _refuse_untrusted_transition(self.discrete_transition_type, "fit")
        self._invalidate_stored_log_likelihood()  # stale after a refit
        # A refit replaces every structure a loaded detector rebuilt.
        self.__dict__.pop("_rebuild_check_", None)
        self.initialize_environments(
            position=position,
            environment_labels=environment_labels,
            compute_all_pairs_distances=transition_representation == "dense",
            max_dense_distance_bytes=max_dense_transition_bytes,
        )
        self._initialize_state_index()
        self._initialize_initial_conditions()
        self._initialize_discrete_state_transition(
            covariate_data=discrete_transition_covariate_data
        )
        # The transitions are rebuilt from their configuration, not learned.
        self.transition_time_bin_width_: float | None = None
        self._transition_time_bin_width_tolerance = 0.0
        self._initialize_continuous_state_transition(
            continuous_transition_types=self.continuous_transition_types,
            transition_representation=transition_representation,
            max_dense_transition_bytes=max_dense_transition_bytes,
            position=position,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
        )

        return self

    @abc.abstractmethod
    @requires_time_edges
    def compute_log_likelihood(self) -> Any:
        """Compute the log likelihood. To be implemented by inheriting class."""

    def _validate_decode_population(
        self,
        spike_times: Any,
        spike_waveform_features: Any = None,
    ) -> None:
        """Validate active fitted populations before missing-data shortcuts."""
        is_clusterless = isinstance(self, ClusterlessDetector)
        active_keys = {
            (obs.environment_name, obs.encoding_group)
            for obs in self.observation_models
            if not obs.is_no_spike
        }
        # Each encoding model unpacks into its own algorithm's predict, which
        # the algorithm parameter selects; another predict would reject or
        # misread its keys.
        parameter = (
            "clusterless_algorithm" if is_clusterless else "sorted_spikes_algorithm"
        )
        current = getattr(self, parameter)
        for key in sorted(active_keys, key=repr):
            fitted = self.encoding_model_[key].get("algorithm")
            if fitted == current:
                continue
            if fitted is None:
                msg = (
                    f"Encoding entry {key!r} does not record the algorithm that "
                    "fitted it; refit before decoding."
                )
                hint = "Call fit(...) with the original recording."
                expected = "an encoding entry that records its algorithm"
                got = "no recorded algorithm"
            else:
                msg = (
                    f"Encoding entry {key!r} was fitted by {fitted!r}, but "
                    f"{parameter} is now {current!r}; refit before decoding."
                )
                hint = (
                    f"Call fit(...) again to fit {current!r}, or set "
                    f"{parameter}={fitted!r} to decode with the fitted model."
                )
                expected = f"{parameter}={fitted!r}"
                got = f"{parameter}={current!r}"
            raise ValidationError(msg, expected=expected, got=got, hint=hint)
        _validate_spike_structure(
            spike_times, spike_waveform_features, clusterless=is_clusterless
        )
        # The per-unit rates every fit of the family stores (see
        # ``SortedSpikesEncodingModel`` and ``ClusterlessEncodingModel``).
        field = "mean_rates" if is_clusterless else "place_fields"
        for key in active_keys:
            model = self.encoding_model_[key]
            validate_population_lengths(
                "electrode" if is_clusterless else "neuron",
                spike_times=spike_times,
                **{field: model[field]},
            )
            if is_clusterless:
                dimensions = model["mark_dimensions"]
                validate_population_lengths(
                    "electrode", spike_times=spike_times, mark_dimensions=dimensions
                )
                for electrode, (features, expected) in enumerate(
                    zip(spike_waveform_features, dimensions, strict=True)
                ):
                    if np.shape(features)[1] != expected:
                        msg = (
                            f"Waveform-feature dimensionality for electrode {electrode} "
                            "does not match the fitted encoding model"
                        )
                        raise ValidationError(
                            msg,
                            expected=f"{expected} waveform features",
                            got=f"{np.shape(features)[1]} waveform features",
                            hint="Use the same waveform-feature selection for encoding "
                            "and decoding, including when tracking is missing.",
                        )

    def _check_spike_data(
        self,
        spike_data: tuple[Any, ...],
        *,
        time_range: tuple[float, float] | None = None,
        range_name: str = "",
        check_extent: bool = False,
    ) -> None:
        """Check the spike arrays named by ``_SPIKE_DATA_NAMES``.

        See ``_validate_spike_data`` for what is checked.

        Parameters
        ----------
        spike_data : tuple
            The spike arrays, in the order of ``_SPIKE_DATA_NAMES``.
        time_range : tuple of float, optional
            ``(start, stop)`` in seconds to warn against, by default None.
        range_name : str, optional
            What ``time_range`` is, for the warning.
        check_extent : bool, optional
            Also warn when almost no spike lies in ``time_range`` and they
            reach far past it, by default False (encoding period only).
        """
        _validate_spike_data(
            *spike_data,
            clusterless=isinstance(self, ClusterlessDetector),
            time_range=time_range,
            range_name=range_name,
            check_extent=check_extent,
        )

    def _check_encoding_spike_data(
        self, position_time: ArrayLike, spike_data: tuple[Any, ...]
    ) -> None:
        """Check encoding spike data; warn if they look like other units or another clock.

        Warns if all lie far to one side of the position_time span, or if
        under 1% lie in it and they reach over 10 spans past it (ms for s on a
        clock starting near 0).

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Times of the position samples, in seconds.
        spike_data : tuple
            The spike arrays, in the order of ``_SPIKE_DATA_NAMES``.
        """
        times = _sample_times(position_time)
        finite = times[np.isfinite(times)]
        self._check_spike_data(
            spike_data,
            time_range=(float(finite.min()), float(finite.max()))
            if finite.size
            else None,
            range_name="position_time span",
            check_extent=True,
        )

    def _check_decode_spike_data(
        self, time_edges: np.ndarray, spike_data: tuple[Any, ...]
    ) -> None:
        """Check decoding spike data; warn if all lie on one side of ``time_edges``.

        Parameters
        ----------
        time_edges : np.ndarray, shape (n_bins + 1,)
            Validated decode edges, in seconds.
        spike_data : tuple
            The spike arrays, in the order of ``_SPIKE_DATA_NAMES``.
        """
        self._check_spike_data(
            spike_data,
            time_range=(float(time_edges[0]), float(time_edges[-1])),
            range_name="decode time_edges",
        )

    _SPIKE_DATA_NAMES: tuple[str, ...] = ()
    """Names of the spike-data arrays, in the order the decode methods take them."""
    _PREDICT_EXAMPLE: str = ""
    """Example ``predict`` call shown in validation errors."""

    def _likelihood_predict_function(self) -> SortedSpikesPredict | ClusterlessPredict:
        """Return the predict function of the detector's likelihood algorithm."""
        raise NotImplementedError

    def _likelihood_predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to every state's likelihood prediction."""
        return {}

    def _likelihood_execution_settings(self) -> dict[str, Any]:
        """Return the execution settings of the current algorithm params, over the fitted ones."""
        return {}

    def _no_spike_mark_log_likelihood(
        self,
        spike_data: tuple[Any, ...],  # noqa: ARG002 -- subclasses override and use them
        time_edges: np.ndarray,  # noqa: ARG002 -- subclasses override and use them
        row_slice: slice | None,  # noqa: ARG002 -- subclasses override and use them
        spike_time_order: _SpikeTimeOrder | None,  # noqa: ARG002 -- subclasses override and use them
    ) -> jax.Array | None:
        """Term added to the No-Spike log likelihood, or None when there is none."""
        return None

    def _compute_spike_log_likelihood(
        self,
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        spike_data: tuple[Any, ...],
        is_missing: np.ndarray | None,
        row_slice: slice | None,
        *,
        time_edges: ArrayLike,
    ) -> jax.Array:
        """Body of the detectors' ``compute_log_likelihood``.

        ``spike_data`` holds the spike arrays named by ``_SPIKE_DATA_NAMES``,
        and the other arguments are those of ``compute_log_likelihood``. It
        checks the model and inputs, as the public decoding methods do once
        per call, then evaluates the requested rows.
        """
        self._validate_time_contract()
        decode = _DecodeInputs.from_arguments(position_time, position)
        time_edges, _ = _decode_time_edges(
            time_edges,
            *self._learned_transition_time_bin_width(),
            sampling_frequency=self.sampling_frequency,
            row_slice=row_slice,
        )
        self._check_decode_spike_data(time_edges, spike_data)
        self._validate_decode_position(decode.position)
        context = self._decode_context(
            time_edges,
            (decode.position_time, decode.position, *spike_data),
            # A direct call warns like the other public decode entry points.
            warn_outside_tracking=True,
            share_no_spike_bin_sizes=False,
        )
        return self._log_likelihood_rows(context, row_slice, is_missing)

    def _decode_context(
        self,
        time_edges: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
        *,
        warn_outside_tracking: bool = False,
        share_no_spike_bin_sizes: bool = True,
    ) -> _DecodeContext:
        """Prepare one likelihood evaluation over validated ``time_edges``.

        Parameters
        ----------
        time_edges : np.ndarray, shape (n_bins + 1,)
            Edges already validated by ``_decode_time_edges``.
        log_likelihood_args : tuple
            ``(position_time, position, *spike_data)``, position coerced by
            ``_DecodeInputs``.
        warn_outside_tracking : bool, optional
            Warn about decode bins outside the tracked period, by default
            False (the decoding methods warned when they marked them missing).
        share_no_spike_bin_sizes : bool, optional
            Compute the No-Spike bin durations once for every row request, by
            default True.

        Returns
        -------
        context : _DecodeContext
        """
        position_time, position, *spike_data = log_likelihood_args
        has_no_spike = any(obs.is_no_spike for obs in self.observation_models)
        return _DecodeContext(
            time_edges=time_edges,
            time_grid=_uniform_time_grid(time_edges),
            position_time=position_time,
            spike_data=tuple(spike_data),
            position_context=_LikelihoodPositionContext.prepare(
                time_edges,
                position_time,
                position,
                reads_position=self._decode_reads_position(),
                warn_outside_tracking=warn_outside_tracking,
            ),
            spike_time_order=_SpikeTimeOrder(),
            no_spike_time_bin_sizes=(
                no_spike_time_bin_sizes(time_edges)
                if share_no_spike_bin_sizes and has_no_spike
                else None
            ),
        )

    def _replaced_log_likelihood(self) -> Callable[..., Any] | None:
        """``compute_log_likelihood`` when it is not the package's own.

        A replacement on the instance, in a subclass, or on a detector
        family class is returned. None when it is the family's own method,
        which decoding evaluates through ``_log_likelihood_rows`` with its
        prepared context.
        """
        callback = self.compute_log_likelihood
        if getattr(callback, "__func__", None) in _PACKAGE_LOG_LIKELIHOODS:
            return None
        return callback

    def _decode_likelihood_callback(
        self, time_edges: np.ndarray, log_likelihood_args: tuple[Any, ...]
    ) -> Callable[..., Any]:
        """Core's observation-row likelihood callback for one decoding call.

        The callback is ``row_slice_aware`` and checks that each request
        returns one row per requested bin. The detector's own likelihood
        evaluates the rows with one ``_DecodeContext`` prepared here; a
        replaced ``compute_log_likelihood`` is called with its public
        arguments and prepares its own (see ``_custom_likelihood_callback``).

        Parameters
        ----------
        time_edges : np.ndarray, shape (n_bins + 1,)
            Edges already validated by ``_decode_time_edges``.
        log_likelihood_args : tuple
            ``(position_time, position, *spike_data)``, as core passes them to
            the callback.
        """
        replaced = self._replaced_log_likelihood()
        if replaced is not None:
            return _custom_likelihood_callback(replaced, time_edges)
        context = self._decode_context(time_edges, log_likelihood_args)
        n_bins = len(time_edges) - 1

        @row_slice_aware
        def row_callback(
            row_time: Any,  # noqa: ARG001 -- likelihood callback signature
            *args: Any,  # noqa: ARG001 -- the context holds them
            row_slice: slice | None = None,
            is_missing: np.ndarray | None = None,
        ) -> Any:
            row_start, row_stop = resolve_row_slice(row_slice, n_bins)
            return _check_likelihood_rows(
                self._log_likelihood_rows(context, row_slice, is_missing),
                row_stop - row_start,
            )

        return row_callback

    def _log_likelihood_rows(
        self,
        context: _DecodeContext,
        row_slice: slice | None,
        is_missing: np.ndarray | None,
    ) -> jax.Array:
        """Log likelihood of the requested decode bins.

        Parameters
        ----------
        context : _DecodeContext
            The decoding call's inputs and preparation.
        row_slice : slice or None
            Contiguous rows of the full-time likelihood; None requests all.
        is_missing : np.ndarray of bool, shape (n_rows,), or None
            Missing bins among the requested rows.

        Returns
        -------
        log_likelihood : jax.Array, shape (n_rows, n_state_bins)
        """
        time_edges, row_time = _decode_time_edges(
            context.time_edges,
            *self._learned_transition_time_bin_width(),
            sampling_frequency=self.sampling_frequency,
            _time_grid=context.time_grid,
            row_slice=row_slice,
        )
        position_time, spike_data = context.position_time, context.spike_data
        self._validate_decode_population(*spike_data)
        position_missing = context.position_context.is_missing
        logger.info("Computing log likelihood...")
        non_local_penalty = self._non_local_position_penalty()

        row_start, row_stop = resolve_row_slice(row_slice, len(time_edges) - 1)
        n_rows = row_stop - row_start
        # Position interpolation, local kernels and the non-local penalty are
        # pointwise in the decoding time, so they are evaluated at the requested
        # bins' centers; the spike likelihoods bin against the full edges.
        if is_missing is None:
            is_missing = np.zeros(n_rows, dtype=bool)
        else:
            is_missing = _sample_mask(
                is_missing,
                n_rows,
                "is_missing",
                entry="requested decode bin",
                example=None,
            )
        if position_missing is not None:
            is_missing = is_missing | position_missing[row_start:row_stop]
        if context.position_context.all_missing:
            return jnp.zeros((n_rows, int(self.is_track_interior_state_bins_.sum())))
        position = context.position_context.position

        likelihood_func = self._likelihood_predict_function()
        execution_settings = self._likelihood_execution_settings()

        # Interior state bins per state. state_ind_ is built state by state, so
        # each state's bins are one contiguous block of columns.
        n_state_columns = np.bincount(
            self.state_ind_[self.is_track_interior_state_bins_],
            minlength=len(self.observation_models),
        )

        # Use dict for O(1) lookup instead of list
        computed_likelihoods: dict[Any, int] = {}

        # Compute all likelihoods first (stays in JAX/GPU if available)
        likelihood_results: dict[int, Any] = {}
        for state_id, obs in enumerate(self.observation_models):
            # Multi-bin local computes full spatial likelihood (is_local=False)
            # then adds position kernel afterward
            effective_is_local = obs.is_local and self.local_position_std is None

            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                effective_is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                no_spike_log_likelihood = predict_no_spike_log_likelihood(
                    spike_data[0],
                    self.no_spike_rate,
                    time_edges=time_edges,
                    row_slice=row_slice,
                    # The setting the other likelihoods receive. No-Spike may
                    # have no encoding entry; default to showing the bar.
                    disable_progress_bar=execution_settings.get(
                        "disable_progress_bar",
                        self.encoding_model_.get(likelihood_name[:2], {}).get(
                            "disable_progress_bar", False
                        ),
                    ),
                    _time_bin_sizes=context.no_spike_time_bin_sizes,
                    _spike_time_order=context.spike_time_order,
                    _time_grid=context.time_grid,
                )
                mark_log_likelihood = self._no_spike_mark_log_likelihood(
                    spike_data, time_edges, row_slice, context.spike_time_order
                )
                likelihood_results[state_id] = (
                    no_spike_log_likelihood
                    if mark_log_likelihood is None
                    else no_spike_log_likelihood + mark_log_likelihood
                )
            elif likelihood_name not in computed_likelihoods:
                likelihood_results[state_id] = likelihood_func(
                    position_time,
                    position,
                    *spike_data,
                    time_edges=time_edges,
                    **{
                        **self.encoding_model_[likelihood_name[:2]],
                        **execution_settings,
                    },
                    **self._likelihood_predict_kwargs(),
                    is_local=effective_is_local,
                    row_slice=row_slice,
                    _spike_time_order=context.spike_time_order,
                    _time_grid=context.time_grid,
                )
                computed_likelihoods[likelihood_name] = state_id
            else:
                # Will reuse previously computed likelihood
                likelihood_results[state_id] = computed_likelihoods[likelihood_name]

        # Assemble the per-state column blocks in one concatenation. As the
        # scatters into a float32 array did, each block is cast to float32 and
        # its penalty or local kernel added at the term's precision, then the
        # sum rounded to float32 once.
        env_penalties: dict[str, jax.Array] = {}
        env_kernels: dict[str, jax.Array] = {}
        blocks: list[jax.Array] = []
        for state_id, obs in enumerate(self.observation_models):
            result = likelihood_results[state_id]
            if isinstance(result, int):
                # Reuse the likelihood computed for an earlier state
                result = likelihood_results[result]
            block = jnp.broadcast_to(
                jnp.asarray(result, dtype=jnp.float32),
                (n_rows, int(n_state_columns[state_id])),
            )
            if non_local_penalty > 0 and not obs.is_local and not obs.is_no_spike:
                env_name = obs.environment_name
                if env_name not in env_penalties:
                    env = self._get_environment_by_name(env_name)
                    env_penalties[env_name] = self._compute_non_local_position_penalty(
                        row_time,
                        cast("np.ndarray", position_time),
                        cast("np.ndarray", position),
                        env,
                    )
                block = (block + jnp.asarray(env_penalties[env_name])).astype(
                    jnp.float32
                )
            if self.local_position_std is not None and obs.is_local:
                env_name = obs.environment_name
                if env_name not in env_kernels:
                    env = self._get_environment_by_name(env_name)
                    env_kernels[env_name] = self._compute_local_position_kernel(
                        row_time,
                        cast("np.ndarray", position_time),
                        cast("np.ndarray", position),
                        env,
                    )
                block = (block + jnp.asarray(env_kernels[env_name])).astype(jnp.float32)
            blocks.append(block)
        log_likelihood = jnp.concatenate(blocks, axis=1)

        # Apply missing data mask
        return jnp.where(jnp.asarray(is_missing)[:, jnp.newaxis], 0.0, log_likelihood)

    def _predict_spikes(
        self,
        spike_data: tuple[Any, ...],
        *,
        time_edges: ArrayLike,
        position: ArrayLike | None,
        position_time: ArrayLike | None,
        is_missing: np.ndarray | None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None,
        cache_likelihood: bool,
        n_chunks: int,
        inference_mode: InferenceMode,
        output_mode: OutputMode,
        chunk_size: int | None,
        result_path: str | Path | None,
        checkpoint_dir: str | Path | None,
        selected_intervals: np.ndarray | None,
        max_read_bytes: int,
        replay_cache_bytes: int | None,
        return_outputs: ReturnOutputs,
        save_log_likelihood_to_results: bool | None,
        save_causal_posterior_to_results: bool | None,
    ) -> xr.Dataset:
        """Body of the detectors' ``predict``; see ``_compute_spike_log_likelihood``."""
        decode = _DecodeInputs.from_arguments(
            position_time, position, example=self._PREDICT_EXAMPLE
        )
        position_time, position = decode.position_time, decode.position
        self._validate_time_contract()
        time_edges, time_centers = _decode_time_edges(
            time_edges,
            *self._learned_transition_time_bin_width(),
            sampling_frequency=self.sampling_frequency,
        )
        self._check_decode_spike_data(time_edges, spike_data)
        self._validate_decode_position(position)
        is_missing = _missing_bins(
            time_edges,
            is_missing,
            position_time,
            position,
            reads_position=self._decode_reads_position(),
            warn_outside_tracking=True,
        )

        # Handle deprecated boolean flags
        if (
            save_log_likelihood_to_results is not None
            or save_causal_posterior_to_results is not None
        ):
            warnings.warn(
                "save_log_likelihood_to_results and save_causal_posterior_to_results "
                "are deprecated. Use return_outputs parameter instead.",
                CompatibilityWarning,
                stacklevel=user_stacklevel(),
            )

            # Convert old flags to new format
            outputs_from_flags: set[OutputName] = set()
            if save_log_likelihood_to_results:
                outputs_from_flags.add("log_likelihood")
            if save_causal_posterior_to_results:
                outputs_from_flags.add("filter")

            if return_outputs is not None:
                msg = (
                    "Cannot specify both return_outputs and deprecated "
                    "save_*_to_results flags. Use return_outputs only."
                )
                raise ConfigurationError(
                    msg,
                    hint="Drop the save_*_to_results flags and pass return_outputs.",
                )
            return_outputs = outputs_from_flags or None

        # Normalize return_outputs to canonical set
        requested_outputs = _normalize_return_outputs(return_outputs)

        # The log likelihood has to be materialized when the caller asks for it.
        # Unchunked, that means caching the single full-time evaluation. Chunked,
        # caching is not available (``_predict`` disables it), so the chunked
        # path accumulates the per-chunk rows instead -- which allocates the
        # same full (n_time, n_state_bins) array, the documented exception to
        # per-chunk allocation.
        _validate_inference_arguments(
            inference_mode,
            output_mode,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            selected_intervals=selected_intervals,
            replay_cache_bytes=replay_cache_bytes,
        )
        return_log_likelihood = "log_likelihood" in requested_outputs
        if (
            inference_mode == "dense"
            and return_log_likelihood
            and not cache_likelihood
            and n_chunks == 1
        ):
            cache_likelihood = True

        predicted_transitions = None
        if discrete_transition_covariate_data is not None:
            if self.discrete_transition_coefficients_ is None:
                msg = "discrete_transition_coefficients_ is None but covariate data provided"
                raise ConfigurationError(
                    msg,
                    hint=(
                        "The discrete transition does not depend on covariates; "
                        "drop discrete_transition_covariate_data or use a "
                        "covariate-dependent transition type."
                    ),
                )
            predicted_transitions = predict_discrete_state_transitions(
                self.discrete_transition_design_matrix_,
                self.discrete_transition_coefficients_,
                discrete_transition_covariate_data,
            )
            _validate_covariate_time_length(predicted_transitions, time_centers)
        if inference_mode == "checkpointed":
            results, _ = self._predict_checkpointed(
                time_edges=time_edges,
                log_likelihood_args=(
                    position_time,
                    position,
                    *spike_data,
                ),
                is_missing=is_missing,
                discrete_transitions=predicted_transitions,
                output_mode=output_mode,
                chunk_size=chunk_size,
                result_path=result_path,
                checkpoint_dir=checkpoint_dir,
                selected_intervals=selected_intervals,
                requested_outputs=requested_outputs,
                max_read_bytes=max_read_bytes,
                replay_cache_bytes=replay_cache_bytes,
            )
            return results

        posterior = self._predict(
            time_edges=time_edges,
            log_likelihood_args=(
                position_time,
                position,
                *spike_data,
            ),
            is_missing=is_missing,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
            discrete_state_transitions=predicted_transitions,
            accumulate_log_likelihoods=return_log_likelihood,
        )

        return convert_results_to_xarray(
            self,
            time_centers,
            np.asarray(posterior.acausal_posterior),
            cast("np.ndarray", posterior.acausal_state_probabilities),
            posterior.marginal_log_likelihood,
            time_edges=time_edges,
            is_missing=is_missing,
            **posterior.requested_results(requested_outputs),
        )

    def _estimate_spike_parameters(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_data: tuple[Any, ...],
        *,
        time_edges: ArrayLike,
        is_missing: np.ndarray | None,
        is_training: ArrayLike | None,
        encoding_group_labels: ArrayLike | None,
        environment_labels: ArrayLike | None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None,
        estimate_initial_conditions: bool,
        estimate_discrete_transition: bool,
        estimate_encoding_model: bool,
        max_iter: int,
        tolerance: float,
        cache_likelihood: bool,
        store_log_likelihood: bool,
        n_chunks: int,
        return_outputs: ReturnOutputs,
        save_log_likelihood_to_results: bool | None,
        min_encoding_local_mass: float,
        min_encoding_local_ess: float,
        encoding_update_damping: float,
        encoding_time_range: np.ndarray | None,
        valid_position_intervals: np.ndarray | None,
        transition_representation: TransitionRepresentation,
        max_dense_transition_bytes: int,
        chunk_size: int | None,
        result_path: str | Path | None,
        replay_cache_bytes: int | None,
    ) -> xr.Dataset:
        """Body of the detectors' ``estimate_parameters``; see ``_compute_spike_log_likelihood``."""
        _validate_encoding_update_damping(encoding_update_damping)
        # Before the internal fit, which replaces the fitted state.
        transition_representation = _resolve_em_options(
            transition_representation,
            n_chunks,
            chunk_size,
            result_path,
            store_log_likelihood=store_log_likelihood,
            replay_cache_bytes=replay_cache_bytes,
        )
        time_edges, time_bin_width = _validate_estimation_arguments(
            time_edges,
            n_chunks,
            return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
        )
        _require_sampling_frequency_width(
            time_edges, time_bin_width, self.sampling_frequency
        )
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        position_time, position = inputs.position_time, inputs.position
        # The internal fit warns if every spike lies outside the position_time span.
        self._check_spike_data(spike_data)
        position = position[:, np.newaxis] if position.ndim == 1 else position
        # Mirror predict(): bins without tracked position are missing
        # observations in the E-step.
        is_missing = _missing_bins(
            time_edges,
            is_missing,
            position_time,
            position,
            reads_position=self._decode_reads_position(),
            warn_outside_tracking=True,
        )
        # Before the internal fit, so a refused estimate leaves the model as is.
        _require_observed_bins(is_missing)
        # Every run drops the stored likelihood of the previous fit. Drop it
        # before the snapshot, which would otherwise keep that
        # (n_time, n_state_bins) array alive through EM; a failed run leaves
        # it dropped too.
        self._invalidate_stored_log_likelihood()
        # EM is atomic: if the internal fit or any EM step raises, the detector
        # and its environments are left as they were before this call (see
        # _restore_state_on_error for the invariant the EM steps keep).
        with self._restore_state_on_error():
            self._fit_em_model(
                lambda representation: self.fit(
                    position_time,
                    position,
                    *spike_data,
                    is_training=inputs.is_training,
                    encoding_group_labels=inputs.encoding_group_labels,
                    environment_labels=inputs.environment_labels,
                    discrete_transition_covariate_data=discrete_transition_covariate_data,
                    encoding_time_range=encoding_time_range,
                    valid_position_intervals=valid_position_intervals,
                    transition_representation=representation,
                    max_dense_transition_bytes=max_dense_transition_bytes,
                ),
                transition_representation,
                len(time_edges) - 1,
                structured_required=bool(
                    _structured_em_options(chunk_size, result_path, replay_cache_bytes)
                ),
            )
            return self._run_em(
                time_edges=time_edges,
                log_likelihood_args=(
                    position_time,
                    position,
                    *spike_data,
                ),
                is_missing=is_missing,
                estimate_initial_conditions=estimate_initial_conditions,
                estimate_discrete_transition=estimate_discrete_transition,
                estimate_encoding_model=estimate_encoding_model,
                max_iter=max_iter,
                tolerance=tolerance,
                cache_likelihood=cache_likelihood,
                store_log_likelihood=store_log_likelihood,
                n_chunks=n_chunks,
                return_outputs=return_outputs,
                save_log_likelihood_to_results=save_log_likelihood_to_results,
                min_encoding_local_mass=min_encoding_local_mass,
                min_encoding_local_ess=min_encoding_local_ess,
                chunk_size=chunk_size,
                result_path=result_path,
                replay_cache_bytes=replay_cache_bytes,
                encoding_fit_arguments={
                    "position_time": position_time,
                    "position": position,
                    **dict(zip(self._SPIKE_DATA_NAMES, spike_data, strict=True)),
                    "is_training": inputs.is_training,
                    "encoding_group_labels": inputs.encoding_group_labels,
                    "environment_labels": inputs.environment_labels,
                    "encoding_time_range": encoding_time_range,
                    "valid_position_intervals": valid_position_intervals,
                },
            )

    def _most_likely_spike_sequence(
        self,
        spike_data: tuple[Any, ...],
        *,
        time_edges: ArrayLike,
        position: ArrayLike | None,
        position_time: ArrayLike | None,
        is_missing: np.ndarray | None,
        n_chunks: int,
    ) -> pd.DataFrame:
        """Body of the detectors' ``most_likely_sequence``; see ``_compute_spike_log_likelihood``."""
        decode = _DecodeInputs.from_arguments(position_time, position)
        position_time, position = decode.position_time, decode.position
        self._validate_time_contract()
        time_edges, _ = _decode_time_edges(
            time_edges,
            *self._learned_transition_time_bin_width(),
            sampling_frequency=self.sampling_frequency,
        )
        self._check_decode_spike_data(time_edges, spike_data)
        self._validate_decode_position(position)
        is_missing = _missing_bins(
            time_edges,
            is_missing,
            position_time,
            position,
            reads_position=self._decode_reads_position(),
            warn_outside_tracking=True,
        )
        return self._run_viterbi(
            time_edges=time_edges,
            log_likelihood_args=(
                position_time,
                position,
                *spike_data,
            ),
            is_missing=is_missing,
            n_chunks=n_chunks,
        )

    def _learned_transition_time_bin_width(self) -> tuple[float | None, float]:
        """Bin width the discrete transitions were learned at, and its precision.

        ``(None, 0.0)`` when the transitions were not learned by
        ``estimate_parameters``.
        """
        return (
            getattr(self, "transition_time_bin_width_", None),
            getattr(self, "_transition_time_bin_width_tolerance", 0.0),
        )

    def _predict(
        self,
        time_edges: np.ndarray,
        log_likelihood_args: tuple[Any, ...] = (),
        is_missing: np.ndarray | None = None,
        log_likelihoods: np.ndarray | None = None,
        *,
        cache_likelihood: bool = True,
        n_chunks: int = 1,
        discrete_state_transitions: np.ndarray | None = None,
        accumulate_log_likelihoods: bool = False,
    ) -> _PosteriorOutputs:
        """
        Compute the posterior probabilities.

        Parameters
        ----------
        time_edges : np.ndarray, shape (n_bins + 1,)
            Validated decode bin edges. Core receives one observation per bin,
            at the bin centers.
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default ()
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None
        log_likelihoods : np.ndarray, optional
            Precomputed log likelihoods, by default None
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default True
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        discrete_state_transitions : np.ndarray, shape (n_time, n_states, n_states) or None, optional
            Covariate-driven transition matrices to use instead of
            self.discrete_state_transitions_. When None, falls back to the
            fitted attribute. By default None.
        accumulate_log_likelihoods : bool, optional
            If True, the chunked (uncached) path copies into one preallocated
            host array so the returned log likelihoods cover every row in
            global order. This allocates the full (n_time, n_state_bins) array
            and is intended only for a caller that explicitly requested the log
            likelihood; when the likelihood is not requested the chunk arrays
            are not retained. By default False.

        Returns
        -------
        posterior : _PosteriorOutputs
            Every output. ``log_likelihood`` is None when the log likelihoods
            were neither given, cached nor accumulated. The three posteriors
            over state bins stay on the device: callers copy to the host only
            the ones they return or need there.
        """
        # The public methods have checked already; EM reaches here after its
        # own fit. Kept as the engine's guard for direct calls.
        self._validate_time_contract()
        time_edges, time_centers = _decode_time_edges(
            time_edges,
            *self._learned_transition_time_bin_width(),
            sampling_frequency=self.sampling_frequency,
        )
        # Disable caching when using multiple chunks (memory optimization)
        if n_chunks > 1 and cache_likelihood:
            logger.info("Disabling likelihood caching for chunked processing")
            cache_likelihood = False

        logger.info("Computing posterior...")
        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)
        state_ind = self.state_ind_[is_track_interior]

        # Use provided transitions or fall back to fitted attribute
        discrete_transitions = (
            discrete_state_transitions
            if discrete_state_transitions is not None
            else self.discrete_state_transitions_
        )
        if discrete_transitions.ndim == 3 and discrete_transitions.shape[0] != len(
            time_centers
        ):
            msg = "Predict covariate transitions must have one row per decode bin"
            raise ValidationError(
                msg,
                expected=f"{len(time_centers)} transition rows",
                got=f"{discrete_transitions.shape[0]} rows",
                hint="Pass discrete_transition_covariate_data aligned to the "
                "requested grid, or predict on the grid used by the fitted covariates.",
            )

        log_likelihood_func = self.compute_log_likelihood
        if log_likelihoods is None:
            log_likelihood_func = self._decode_likelihood_callback(
                time_edges, log_likelihood_args
            )

        if discrete_transitions.ndim == 2:
            result = chunked_filter_smoother(
                time=time_centers,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                transition_matrix=(
                    self.continuous_state_transitions_[cross_is_track_interior]
                    * discrete_transitions[np.ix_(state_ind, state_ind)]
                ),
                log_likelihood_func=log_likelihood_func,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=cache_likelihood,
                accumulate_log_likelihoods=accumulate_log_likelihoods,
                host_outputs=False,
            )
        else:
            result = chunked_filter_smoother_covariate_dependent(
                time=time_centers,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                discrete_transition_matrix=discrete_transitions,
                continuous_transition_matrix=self.continuous_state_transitions_[
                    cross_is_track_interior
                ],
                log_likelihood_func=log_likelihood_func,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=cache_likelihood,
                accumulate_log_likelihoods=accumulate_log_likelihoods,
                host_outputs=False,
            )
        return _PosteriorOutputs(
            marginal_log_likelihood=result.marginal_log_likelihood,
            degenerate_indices=np.asarray(result.degenerate_indices, dtype=int),
            acausal_posterior=result.acausal_posterior,
            acausal_state_probabilities=result.acausal_state_probabilities,
            causal_state_probabilities=result.causal_state_probabilities,
            predictive_state_probabilities=result.predictive_state_probabilities,
            log_likelihood=result.log_likelihoods,
            causal_posterior=result.causal_posterior,
            predictive_posterior=result.predictive_posterior,
        )

    @contextmanager
    def _restore_state_on_error(self) -> Iterator[None]:
        """Restore this detector if the block raises.

        ``fit`` rebuilds the environments, transitions, and encoding model in
        stages. If a later stage fails, the earlier stages must not leave the
        previously fitted detector paired with a refit grid. The detector
        rebinds attributes rather than edit arrays in place, and fits new
        environment copies rather than changing its environments, so a shallow
        snapshot of the detector's attributes is enough.

        ``estimate_parameters`` runs its internal fit and EM in one block. The
        EM steps rebind attributes, and edit in place only arrays the internal
        fit has just rebuilt (``initial_conditions_``, the frozen rows of the
        M-step's new ``discrete_state_transitions_``), so the snapshot still
        holds the previous values. An EM step must keep to this: an in-place
        edit of an array the internal fit does not rebuild could not be undone.

        The snapshot keeps every attribute it holds alive until the block
        exits; drop attributes the block always discards before entering it.
        """
        detector_state = dict(self.__dict__)
        try:
            yield
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(detector_state)
            raise

    def _invalidate_stored_log_likelihood(self) -> None:
        """Drop ``log_likelihood_``, which no longer describes this model.

        ``log_likelihood_`` is an OUTPUT of the run that produced it: it is tied
        to that run's spikes, position, missing-data mask and encoding model.
        Feeding it back into a later run decodes new data with an old likelihood
        and returns a plausible, wrong posterior, so it is dropped whenever the
        model or the data it describes can have changed -- at the start of every
        estimation run and on any refit.
        """
        with suppress(AttributeError):
            del self.log_likelihood_

    def _encoding_masks(
        self, inputs: _EncodingInputs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Training mask and labels of an encoding fit, with their defaults.

        Without them every sample trains encoding group 0 of the first
        environment.

        Returns
        -------
        is_training : np.ndarray of bool, shape (n_time_position,)
        encoding_group_labels : np.ndarray, shape (n_time_position,)
        environment_labels : np.ndarray, shape (n_time_position,)
        """
        n_time = len(inputs.position_time)
        return (
            np.ones((n_time,), dtype=bool)
            if inputs.is_training is None
            else inputs.is_training,
            np.zeros((n_time,), dtype=np.int32)
            if inputs.encoding_group_labels is None
            else inputs.encoding_group_labels,
            np.asarray([self.environments[0].environment_name] * n_time)
            if inputs.environment_labels is None
            else inputs.environment_labels,
        )

    def _fit_encoding_groups(
        self,
        fit_group: Callable[[Environment, np.ndarray], Any],
        *,
        is_training: np.ndarray,
        encoding_group_labels: np.ndarray,
        environment_labels: np.ndarray,
        weights: np.ndarray | None,
        support: EncodingSupport,
    ) -> dict[tuple[Any, ...], Any]:
        """Fit one encoding model per environment and encoding group.

        Parameters
        ----------
        fit_group : callable
            ``fit_group(environment, group_weights)`` fits the family's
            algorithm to one group's spikes and returns its encoding model.
        is_training : np.ndarray of bool, shape (n_time_position,)
            Samples the encoding models are trained on.
        encoding_group_labels : np.ndarray, shape (n_time_position,)
            Encoding group label of each sample.
        environment_labels : np.ndarray, shape (n_time_position,)
            From ``_encoding_masks``.
        weights : np.ndarray, shape (n_time_position,), or None
            Training weights.
        support : EncodingSupport
            Support of the encoding fit (exposure of each sample).

        Returns
        -------
        encoding_model : dict
            Encoding model of each ``(environment_name, encoding_group)``.
            Built here, so a failed refit keeps the previous encoding model
            and its stored log likelihood.
        """
        encoding_model: dict[tuple[Any, ...], Any] = {}

        # An encoding model is shared by ``(environment_name, encoding_group)``;
        # ``is_local`` / ``is_no_spike`` do not affect the fit. Since
        # ``ObservationModel`` equality now distinguishes those flags,
        # ``np.unique`` can return several entries that map to the same
        # ``likelihood_name`` — fit each key once to avoid redundant work.
        fitted_likelihood_names: set[tuple[Any, ...]] = set()
        for obs in np.unique(cast("Any", self.observation_models)):
            likelihood_name = (obs.environment_name, obs.encoding_group)
            if likelihood_name in fitted_likelihood_names:
                continue
            fitted_likelihood_names.add(likelihood_name)

            environment = self._get_environment_by_name(obs.environment_name)

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            is_group = is_training & is_encoding & is_environment
            # The mask is this model's exposure: a sample outside the selection
            # must contribute no occupancy, because only in-group spikes are
            # counted. Passing ``weights=None`` means "uniform over every
            # supplied sample", which is only true when the mask selects
            # everything -- otherwise occupancy accumulates over held-out
            # samples and other environments / encoding groups while the spike
            # count does not, biasing every place field low. The position array
            # stays full-length so it remains aligned with the weights and with
            # the grid used to interpolate per-spike weights.
            group_weights = _group_weights(is_group, weights, likelihood_name, support)
            encoding_model[likelihood_name] = fit_group(environment, group_weights)
        return encoding_model

    @abc.abstractmethod
    def fit_encoding_model(self, *args: Any, **kwargs: Any) -> None:
        """Fit the encoding model. To be implemented by inheriting class.

        Parameters
        ----------
        *args
            Positional arguments of the subclass implementation; see
            ``SortedSpikesDetector.fit_encoding_model`` and
            ``ClusterlessDetector.fit_encoding_model``.
        **kwargs
            Keyword arguments of the subclass implementation.
        """

    @abc.abstractmethod
    def _fit_encoding_model(self, *args: Any, **kwargs: Any) -> None:
        """Body of ``fit_encoding_model``, which EM encoding updates call.

        The public method is a thin wrapper, so overriding it does not change
        EM refits in either family.

        Parameters
        ----------
        *args
            Positional arguments of the subclass implementation.
        **kwargs
            Keyword arguments of the subclass implementation.
        """

    def _run_viterbi(
        self,
        time_edges: ArrayLike,
        log_likelihood_args: tuple[Any, ...] | None = None,
        is_missing: np.ndarray | None = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Viterbi engine: the most likely sequence of states.

        The detectors' ``most_likely_sequence`` validate the data, then call
        this with the ``compute_log_likelihood`` arguments. The likelihood is
        always recomputed from the data passed here; a stored
        ``log_likelihood_`` belongs to the run that produced it.

        Parameters
        ----------
        time_edges : array_like, shape (n_bins + 1,)
            Validated decode bin edges.
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default None.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
        n_chunks : int, optional
            Number of chunks, by default 1.

        Returns
        -------
        pd.DataFrame, shape (n_bins, n_columns)
            DataFrame containing the most likely sequence of states
            and corresponding positions/metadata at each time step.
        """
        # As in _predict, a guard for direct calls of the engine.
        self._validate_time_contract()
        time_edges, time_centers = _decode_time_edges(
            time_edges,
            *self._learned_transition_time_bin_width(),
            sampling_frequency=self.sampling_frequency,
        )
        # Validate parameters
        if log_likelihood_args is None:
            log_likelihood_args = ()

        if (
            self.discrete_state_transitions_.ndim == 3
            and self.discrete_state_transitions_.shape[0] != len(time_edges) - 1
        ):
            msg = "Viterbi covariate transitions must have one row per decode bin"
            raise ValidationError(
                msg,
                expected=f"{len(time_edges) - 1} transition rows",
                got=f"{self.discrete_state_transitions_.shape[0]} rows",
                hint="Viterbi uses the fitted covariate transitions. Fit or estimate with covariates aligned to this grid; predict(..., discrete_transition_covariate_data=...) accepts new aligned prediction covariates.",
            )

        log_likelihood_func = self._decode_likelihood_callback(
            time_edges, log_likelihood_args
        )
        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)
        state_ind = self.state_ind_[is_track_interior]
        if self.discrete_state_transitions_.ndim == 2:
            sequence_ind, _ = most_likely_sequence(
                time=time_centers,
                initial_distribution=self.initial_conditions_[is_track_interior],
                transition_matrix=(
                    self.continuous_state_transitions_[cross_is_track_interior]
                    * self.discrete_state_transitions_[np.ix_(state_ind, state_ind)]
                ),
                log_likelihood_func=log_likelihood_func,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
            )
        else:
            sequence_ind, _ = most_likely_sequence_covariate_dependent(
                time=time_centers,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                discrete_transition_matrix=self.discrete_state_transitions_,
                continuous_transition_matrix=self.continuous_state_transitions_[
                    cross_is_track_interior
                ],
                log_likelihood_func=log_likelihood_func,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
            )

        return convert_seq_to_df(self, sequence_ind, time_centers)

    def calculate_time_edges(
        self, time_range: np.ndarray, *, trim: bool = False
    ) -> np.ndarray:
        """Uniform decode edges at ``1 / sampling_frequency`` spanning a range.

        Parameters
        ----------
        time_range : np.ndarray, shape (2,)
            ``(start, stop)`` in seconds.
        trim : bool, optional
            If False (default), the range must be a whole number of bins. If
            True, a partial final bin is dropped.

        Returns
        -------
        time_edges : np.ndarray, shape (n_bins + 1,)
            Pass as ``time_edges`` to ``predict``.

        Raises
        ------
        ValidationError
            If the range is not a whole number of bins and ``trim`` is False,
            or holds no complete bin.
        DataError
            If ``start`` or ``stop`` is not finite, or ``stop <= start``.
        """
        return calculate_time_edges(time_range, self.sampling_frequency, trim=trim)

    def copy(self) -> "_DetectorBase":
        """
        Make a copy of the detector.

        Returns
        -------
        _DetectorBase
            Deep copy of the detector instance.
        """
        return copy.deepcopy(self)

    def _get_environment_by_name(self, environment_name: str) -> Environment:
        """
        Get one of this model's fitted environments by name.

        Parameters
        ----------
        environment_name : str
            Name of the environment to retrieve.

        Returns
        -------
        Environment
            The fitted copy in ``environments_``.

        Raises
        ------
        ValidationError
            If environment not found.
        """
        # Looked up in ``self.environments_`` on every call: each fit replaces
        # them with new fitted copies, so a stored lookup would go stale.
        return find_environment_by_name(self.environments_, environment_name)


class ClusterlessDetector(_DetectorBase):
    """
    Detector class for clusterless spikes.

    Decodes from spike waveform features (marks) without spike sorting. The
    constructor parameters are documented on ``__init__``.
    """

    mark_reference_: list[MarkReference | None]

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict[str, Any] | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ) -> None:
        """
        Initialize the ClusterlessDetector class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        clusterless_algorithm : str, optional
            Algorithm for clusterless spikes, by default "clusterless_kde".
        clusterless_algorithm_params : dict or None, optional
            Parameters for the clusterless algorithm. Stored as-is (``None``
            stays ``None``) to keep sklearn's ``clone()`` contract intact. If
            ``None`` (the default), the algorithm's registered
            ``default_params`` are copied into a fresh dict **at
            fit time**; a user-supplied dict replaces those defaults, is
            **stored by reference** and copied on use. At fit time a key the
            algorithm's fit does not accept raises ``ValidationError``.
            Mutations the caller makes to their own dict between construction
            and fit are therefore observed by the fit — pass
            ``dict(my_params)`` for an isolated copy.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Rate in Hz of the decode bins; decoding ``time_edges`` must be
            ``1 / sampling_frequency`` wide. By default 500.0.
        no_spike_rate : float, optional
            No-Spike firing rate per electrode in Hz, by default 1e-10. Its
            spikes' marks are scored with the training-fitted reference
            density ``mark_reference_``.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Data-adaptive prior weight, by default 0.0. See
            ``_DetectorBase`` for details.
        frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
            Rows of the discrete transition matrix to freeze during EM
            re-estimation, by default ``"no_spike"`` (the No-Spike states'
            rows). See ``_DetectorBase`` for details.
        local_position_std : float or None, optional
            Standard deviation of the position uncertainty kernel for the
            local state. See ``_DetectorBase`` for details.
        """
        super().__init__(
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=continuous_initial_conditions_types,
            discrete_transition_type=discrete_transition_type,
            discrete_transition_concentration=discrete_transition_concentration,
            discrete_transition_stickiness=discrete_transition_stickiness,
            discrete_transition_regularization=discrete_transition_regularization,
            continuous_transition_types=continuous_transition_types,
            observation_models=observation_models,
            environments=environments,
            infer_track_interior=infer_track_interior,
            state_names=state_names,
            sampling_frequency=sampling_frequency,
            no_spike_rate=no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        self.clusterless_algorithm = clusterless_algorithm
        # A removed name that stored parameter sets, such as Spyglass rows,
        # still hold: they construct, for inspection, and fitting raises.
        if clusterless_algorithm != "clusterless_kde_log":
            self._clusterless_algorithm()  # rejects an unregistered name
        # Stored as-is to keep sklearn's clone() contract intact; the dict
        # copy + default resolution happens lazily in
        # ``_resolve_clusterless_algorithm_params`` at fit time.
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def _refit_encoding_model(
        self, weights: np.ndarray, encoding_fit_arguments: dict[str, Any]
    ) -> None:
        """EM M-step: refit the encoding model with posterior ``weights``.

        Posterior weights update the encoding process, not the
        training-fitted ``mark_reference_``.

        Parameters
        ----------
        weights : np.ndarray, shape (n_time_position,)
            Posterior weight of each time sample in the refit.
        encoding_fit_arguments : dict
            The other keyword arguments of ``_fit_encoding_model``.
        """
        self._fit_encoding_model(
            **encoding_fit_arguments, weights=weights, refit_mark_reference=False
        )

    def _resolve_clusterless_algorithm_params(self) -> dict[str, Any]:
        """Return a fresh, validated copy of the clusterless algorithm params.

        Resolves a ``None`` user spec to the algorithm's registered
        ``default_params``; otherwise copies the user's dict.
        Raises ``ValidationError`` for a key the registered fit does not
        accept.
        """
        self._clusterless_algorithm()  # validates the algorithm name
        return _resolve_fit_params(
            _CLUSTERLESS_ALGORITHMS,
            self.clusterless_algorithm,
            self.clusterless_algorithm_params,
            "clusterless_algorithm_params",
        )

    def _clusterless_algorithm(self) -> ClusterlessLikelihoodAlgorithm:
        """Return the registered record of ``clusterless_algorithm``."""
        try:
            return _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]
        except KeyError:
            hint = (
                'Use "clusterless_kde": it is the same log-space KDE estimator.'
                if self.clusterless_algorithm == "clusterless_kde_log"
                else None
            )
            msg = f"Unknown clusterless_algorithm {self.clusterless_algorithm!r}"
            raise ValidationError(
                msg,
                expected=f"one of {sorted(_CLUSTERLESS_ALGORITHMS)}",
                got=repr(self.clusterless_algorithm),
                hint=hint,
            ) from None

    def _fit_mark_reference(
        self,
        position_time: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: np.ndarray,
        environment_labels: np.ndarray,
        weights: np.ndarray | None,
        support: EncodingSupport,
        algorithm_params: dict[str, Any],
    ) -> list[Any]:
        """Fit each electrode's mark reference from its training spikes.

        Training weights span every encoding group of the requested
        environments, are formed on the full tracking timeline, and are
        interpolated to spike times within the encoding support, as for the
        encoding models.
        """
        requested_environments = [
            obs.environment_name for obs in self.observation_models
        ]
        training_weights = np.asarray(is_training, dtype=float) * np.isin(
            environment_labels, requested_environments
        )
        if weights is not None:
            training_weights = training_weights * np.asarray(weights)

        # The reference adds the mark bandwidth, in waveform units, to each
        # feature's training spread.
        mark_reference_scale = self._clusterless_algorithm().mark_reference_scale
        kernel_scale = cast(
            "float | np.ndarray", mark_reference_scale(**algorithm_params)
        )
        spike_weights = [
            interpolate_weights_at_spike_times(
                times, position_time, training_weights, encoding_support=support
            )
            for times in spike_times
        ]
        return fit_mark_references(spike_waveform_features, kernel_scale, spike_weights)

    def fit_encoding_model(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        *,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
    ) -> None:
        """
        Fit the encoding model to the data.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray, len (n_electrodes,)
            One 1-D array of spike times per electrode, in seconds; NaN or
            infinite times raise ``DataError``.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_training : array_like of bool, shape (n_time_position,), optional
            True (or 1) for the position samples to train the encoding model
            on, one entry per sample, by default None (every sample).
        encoding_group_labels : array_like of int or str, shape (n_time_position,), optional
            Encoding group of each position sample, matched to the observation
            models' ``encoding_group``, by default None (group 0).
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment name of each position sample, by default None.
        weights : np.ndarray, optional, shape (n_time_position,)
            Weights for training data, by default None.

        encoding_time_range : array_like, shape (2,), optional
            Acquisition start/stop in seconds. Clips encoding support using the
            original interpolation basis. Uniform samples otherwise include endpoint
            half-cells; a singleton requires explicit bounds or a tracking interval.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            Ordered, non-overlapping continuous tracking intervals in seconds.
            Required for irregular timestamps; each interval needs a finite sample.
            Positions are held at segment endpoints and NaN rows split support.
            Encoding support does not automatically mark decode bins missing.

        Attributes
        ----------
        encoding_model_ : dict
            Dictionary holding the fitted encoding models for each unique observation model
            configuration (environment, encoding group).
            The values depend on the chosen `clusterless_algorithm`.
        mark_reference_ : list of MarkReference or None, length n_electrodes
            Normalized waveform density of each electrode: a product of
            per-feature Student-t densities (3 degrees of freedom) centred on
            the weighted mean of its training spikes in the requested
            environments (all encoding groups), with scale
            ``sqrt(training variance + h**2)`` for mark bandwidth ``h``.
            No-Spike scores its spikes' marks with it, and every active state
            adds a background intensity
            ``RATE_EPS_HZ * q(mark)``. None marks an electrode without such
            spikes. It is kept unchanged by posterior-weighted encoding updates.
        """
        self._check_encoding_spike_data(
            position_time, (spike_times, spike_waveform_features)
        )
        self._fit_encoding_model(
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_training,
            encoding_group_labels,
            environment_labels,
            weights,
            encoding_time_range=encoding_time_range,
            valid_position_intervals=valid_position_intervals,
            refit_mark_reference=True,
        )

    def _fit_encoding_model(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        *,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
        refit_mark_reference: bool,
    ) -> None:
        """Body of ``fit_encoding_model``.

        ``refit_mark_reference=False`` keeps the existing ``mark_reference_``,
        as posterior-weighted encoding updates do.
        """
        logger.info("Fitting clusterless spikes...")
        # Validate before any state changes and before the collections are
        # paired electrode by electrode (``fit`` also checks first, before
        # ``_fit`` rebuilds its own state).
        validate_spike_feature_population(spike_times, spike_waveform_features)
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            weights,
        )
        position_time, weights = inputs.position_time, inputs.weights
        position = (
            inputs.position[:, np.newaxis]
            if inputs.position.ndim == 1
            else inputs.position
        )
        is_training, encoding_group_labels, environment_labels = self._encoding_masks(
            inputs
        )

        support = EncodingSupport(
            position_time,
            position,
            encoding_time_range=encoding_time_range,
            valid_position_intervals=valid_position_intervals,
        )

        kwargs = self._resolve_clusterless_algorithm_params()
        encoding_algorithm = self._clusterless_algorithm().fit

        def fit_group(environment: Environment, group_weights: np.ndarray) -> Any:
            # The group's weights are its exposure on the full timeline: a
            # subset timeline would let spike-weight interpolation bridge the
            # mask's gaps.
            is_group_spike = [
                _group_spike_mask(times, position_time, group_weights, support)
                for times in spike_times
            ]
            return encoding_algorithm(
                position_time=position_time,
                position=position,
                spike_times=[
                    times[mask]
                    for times, mask in zip(spike_times, is_group_spike, strict=True)
                ],
                spike_waveform_features=[
                    features[mask]
                    for features, mask in zip(
                        spike_waveform_features, is_group_spike, strict=True
                    )
                ],
                environment=environment,
                _encoding_support=support,
                weights=group_weights,
                **kwargs,
            )

        encoding_model = self._fit_encoding_groups(
            fit_group,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            weights=weights,
            support=support,
        )

        mark_reference = (
            self._fit_mark_reference(
                position_time,
                spike_times,
                spike_waveform_features,
                is_training,
                environment_labels,
                weights,
                support,
                kwargs,
            )
            if refit_mark_reference
            else self.mark_reference_
        )

        self._invalidate_stored_log_likelihood()  # stale: encoding model replaced
        self.encoding_model_ = encoding_model
        self.mark_reference_ = mark_reference
        self.time_contract_ = dict(_TIME_CONTRACT)

    def fit(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        *,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
        transition_representation: TransitionRepresentation = "dense",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> "ClusterlessDetector":
        """
        Fit the detector to the data.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray, len (n_electrodes,)
            One 1-D array of spike times per electrode, in seconds; NaN or
            infinite times raise ``DataError``.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_training : array_like of bool, shape (n_time_position,), optional
            True (or 1) for the position samples to train the encoding model
            on, one entry per sample, by default None (every sample).
        encoding_group_labels : array_like of int or str, shape (n_time_position,), optional
            Encoding group of each position sample, matched to the observation
            models' ``encoding_group``, by default None (group 0).
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment name of each position sample, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        encoding_time_range : array_like, shape (2,), optional
            Acquisition start/stop in seconds. Clips encoding support using the
            original interpolation basis. Uniform samples otherwise include endpoint
            half-cells; a singleton requires explicit bounds or a tracking interval.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            Ordered, non-overlapping continuous tracking intervals in seconds.
            Required for irregular timestamps; each interval needs a finite sample.
            Positions are held at segment endpoints and NaN rows split support.
            Encoding support does not automatically mark decode bins missing.

        transition_representation : {"dense", "structured", "auto"}, optional
            By default "dense", which builds the full transition matrix.
            "structured" requires exact supported state-block operators;
            "auto" permits budgeted dense fallback blocks. Non-dense N-D
            environments defer all-pairs graph distances.
        max_dense_transition_bytes : int, optional
            Byte limit for retained fallback blocks, temporary dense graph
            distances, and explicit reads of the lazy transition and distance
            views, by default 256 MiB. Ignored when
            transition_representation="dense".

        Returns
        -------
        ClusterlessDetector
            Fitted detector instance.
        """
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        # Before ``_fit``; fit_encoding_model checks again and warns.
        self._check_spike_data((spike_times, spike_waveform_features))
        with self._restore_state_on_error():
            self._fit(
                inputs.position,
                inputs.is_training,
                inputs.encoding_group_labels,
                inputs.environment_labels,
                discrete_transition_covariate_data,
                transition_representation=transition_representation,
                max_dense_transition_bytes=max_dense_transition_bytes,
            )
            self.fit_encoding_model(
                inputs.position_time,
                inputs.position,
                spike_times,
                spike_waveform_features,
                inputs.is_training,
                inputs.encoding_group_labels,
                inputs.environment_labels,
                encoding_time_range=encoding_time_range,
                valid_position_intervals=valid_position_intervals,
            )
        return self

    _SPIKE_DATA_NAMES = ("spike_times", "spike_waveform_features")
    _PREDICT_EXAMPLE = (
        "    results = detector.predict(\n"
        "        spike_times=spike_times, spike_waveform_features=features,\n"
        "        time_edges=time_edges, position=position,\n"
        "        position_time=position_time\n"
        "    )"
    )

    def _likelihood_predict_function(self) -> ClusterlessPredict:
        return self._clusterless_algorithm().predict

    def _likelihood_execution_settings(self) -> dict[str, Any]:
        return _execution_settings(
            self._clusterless_algorithm(), self.clusterless_algorithm_params
        )

    def _likelihood_predict_kwargs(self) -> dict[str, Any]:
        # One reference for every state and No-Spike, so their
        # mark densities cancel consistently between states.
        return {"mark_reference": self.mark_reference_}

    def _no_spike_mark_log_likelihood(
        self,
        spike_data: tuple[Any, ...],
        time_edges: np.ndarray,
        row_slice: slice | None,
        spike_time_order: _SpikeTimeOrder | None,
    ) -> jax.Array:
        # Marked No-Spike intensity: no_spike_rate * q(mark).
        return mark_reference_log_likelihood(
            self.mark_reference_,
            *cast("tuple[Any, Any]", spike_data),  # (spike_times, features)
            time_edges,
            row_slice=row_slice,
            _spike_time_order=spike_time_order,
        )

    def _validate_time_contract(self) -> None:
        super()._validate_time_contract()
        if not hasattr(self, "mark_reference_"):
            msg = (
                "Saved clusterless model has no mark reference; refit before decoding."
            )
            raise NotFittedError(
                msg,
                hint="Clusterless states score waveform marks against a reference "
                "density fitted from training spikes. Call fit(...) with the "
                "original recording.",
            )

    @row_slice_aware
    @requires_time_edges
    def compute_log_likelihood(
        self,
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_missing: np.ndarray | None = None,
        row_slice: slice | None = None,
        *,
        time_edges: ArrayLike,
    ) -> jax.Array:
        """
        Compute the log likelihood for the given data.

        Notes
        -----
        When ``local_position_std`` is set, the per-bin value returned
        for local-state entries is not a pure observation likelihood.
        It combines the spatial spike likelihood with the position
        uncertainty kernel::

            log_likelihood[local, b, t] =
                log P(spikes_t | bin b)                   # spatial likelihood
              + log_kernel(b | animal_pos_t)              # Gaussian anchor
              + log(n_interior_bins)                      # mass-balance

        The kernel term is a time-varying state-specific spatial prior.
        Mathematically, injecting it into the likelihood is equivalent
        to injecting it into the transition matrix — both multiply into
        the HMM forward step — but avoids breaking the static-transition
        assumption used by ``jax.lax.scan``. The ``log(n_interior_bins)``
        constant compensates for the ``1/n_bins`` uniform continuous IC
        so the effective per-bin prior is ``exp(log_kernel)`` itself.

        The kernel is part of the likelihood model, not an ad-hoc
        post-processing step, so the posterior returned by ``predict()``
        is a valid probability distribution under the modified
        generative model. Users comparing ``local_position_std=None`` to
        a finite value will see different posteriors (the models
        differ); users reading the raw ``log_likelihood`` (via
        ``return_outputs='log_likelihood'``) should be aware that
        local-state entries are *not* pure ``log P(spikes | state, bin)``
        — they include the kernel and mass-balance terms.

        Parameters
        ----------
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        position_time : array_like, shape (n_time_position,), or None
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims), or None
            Position data.
        spike_times : list of np.ndarray, len (n_electrodes,)
            One 1-D array of spike times per electrode, in seconds; NaN or
            infinite times raise ``DataError``.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_missing : array_like of bool, shape (n_rows,), optional
            True (or 1) for the bins to decode as missing, one entry per
            requested decode bin, by default None.
        row_slice : slice | None, optional
            Contiguous range of rows of the full-time likelihood to compute, by
            default None (all rows). ``time_edges`` stay the FULL decoding edges,
            so a decoding spike is owned by its global row no matter how the
            rows were chunked, and the result equals the full-time likelihood
            sliced by ``row_slice``.

        Returns
        -------
        log_likelihood : jax.Array, shape (n_rows, n_state_bins)
            ``n_rows`` is ``n_time`` unless ``row_slice`` is given.
        """
        return self._compute_spike_log_likelihood(
            position_time,
            position,
            (spike_times, spike_waveform_features),
            is_missing,
            row_slice,
            time_edges=time_edges,
        )

    @requires_time_edges
    def predict(
        self,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        *,
        time_edges: ArrayLike,
        position: ArrayLike | None = None,
        position_time: ArrayLike | None = None,
        is_missing: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        cache_likelihood: bool = False,
        n_chunks: int = 1,
        inference_mode: InferenceMode = "dense",
        output_mode: OutputMode = "spatial",
        chunk_size: int | None = None,
        result_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        selected_intervals: np.ndarray | None = None,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        replay_cache_bytes: int | None = None,
        return_outputs: ReturnOutputs = None,
        save_log_likelihood_to_results: bool | None = None,
        save_causal_posterior_to_results: bool | None = None,
    ) -> xr.Dataset:
        """
        Predict the posterior probabilities for the given data.

        Parameters
        ----------
        spike_times : list of np.ndarray, len (n_electrodes,)
            One 1-D array of spike times per electrode, in seconds; NaN or
            infinite times raise ``DataError``.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        position : array_like, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : array_like, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : array_like of bool, shape (n_bins,), optional
            True (or 1) for the bins to decode as missing, one entry per
            decode bin, by default None.
        discrete_transition_covariate_data : dict-like, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default False.
            Caching is disabled automatically when ``n_chunks > 1``. Requesting
            ``return_outputs='log_likelihood'`` with ``n_chunks > 1`` instead
            accumulates the per-chunk rows into the full
            (n_time, n_state_bins) array -- the one exception to per-chunk
            likelihood allocation; when the log likelihood is not requested, no
            chunk array is retained.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1.
        return_outputs : str, collection of str, or None, optional
            Controls which optional outputs are returned.

            Options:
            - None: smoother only (default, minimal memory)
            - 'filter': filtered (causal) posterior and state probabilities
            - 'predictive': one-step-ahead (predictive) state probabilities
            - 'predictive_posterior': one-step-ahead posterior over state bins
            - 'log_likelihood': per-timepoint log likelihoods
            - 'all': all outputs above
            - List/set: e.g., ['filter', 'log_likelihood'] for multiple outputs

            A name adds the same outputs alone or in a list or set; ask for
            both predictive outputs with ['predictive', 'predictive_posterior'].
            acausal_state_probabilities and marginal_log_likelihood are always
            included; acausal_posterior is included except in checkpointed
            compact output. Compact output holds state probabilities only, so
            'log_likelihood' and 'predictive_posterior' (and 'all') require
            output_mode='spatial'.

            When to use each output:
            - 'filter': Online/causal decoding, debugging forward pass
            - 'predictive': Model evaluation, predictive checks of the states
            - 'predictive_posterior': Predictive checks over position
            - 'log_likelihood': Diagnostics, per-timepoint metrics, model comparison

            Memory warning: 'log_likelihood', 'filter' and 'predictive_posterior'
            hold (n_time, n_state_bins) arrays, which can be very large (~400 GB
            for 1M timepoints × 100k spatial bins). Use checkpointed compact mode
            to bound spatial memory.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead.
            Whether to save the log likelihood to the results, by default None.
        save_causal_posterior_to_results : bool, optional
            DEPRECATED. Use return_outputs='filter' instead.
            Whether to save the causal (filtered) posterior to the results, by default None.

        inference_mode : {"dense", "checkpointed"}, optional
            By default "dense", the in-memory driver. "checkpointed" stores
            boundary messages on disk and replays chunks for exact smoothing;
            it raises ValueError unless n_chunks=1 and cache_likelihood=False.
        output_mode : {"compact", "spatial"}, optional
            By default "spatial", the only output dense prediction accepts.
            Checkpointed "compact" returns state probabilities without
            full-session spatial arrays; checkpointed "spatial" writes lazy
            incremental arrays and requires result_path.
        chunk_size : int, optional
            Maximum checkpoint/replay rows (checkpointed mode only). By
            default, enough rows for one chunk of float32 likelihoods to take
            about 256 MiB, and at least 256. Larger chunks amortize per-chunk
            overhead; working memory grows with chunk rows, to several times
            one likelihood chunk. docs/hardware_settings.md lists measured
            memory and speed.
        result_path : str or Path, optional
            New directory for atomic incremental output. Existing destinations
            are never overwritten. Optional for compact, required for spatial.
        checkpoint_dir : str or Path, optional
            Parent directory for temporary boundary files, cleaned after the run.
        selected_intervals : array_like, shape (n_intervals, 2), optional
            Ordered non-overlapping start/stop selections. Emit only bins fully
            inside them while conditioning on the entire supplied recording.
        max_read_bytes : int, optional
            Maximum bytes in one explicit spatial read, by default 512 MiB.
            Select fewer rows/bins before converting a large lazy array.
        replay_cache_bytes : int, optional
            Checkpointed mode only: host memory for keeping forward-pass rows
            (filtered rows, or likelihoods when the likelihood or predictive
            outputs are requested) so the backward replay does not recompute
            them. Chunks beyond the budget are recomputed. By default None,
            which uses 16 GiB or half the available host memory, whichever is
            smaller.

        Returns
        -------
        xr.Dataset
            Dataset containing decoded posterior distributions.

            Compact mode omits spatial posteriors. Dense/spatial modes include:
            - acausal_posterior : (n_time, n_state_bins)
                Smoothed posterior over state bins
            - acausal_state_probabilities : (n_time, n_states)
                Smoothed discrete state probabilities
            - marginal_log_likelihoods : float (in attrs)
                Total log evidence for the model

            Conditionally included based on return_outputs:
            - causal_posterior : (n_time, n_state_bins) - if 'filter'
                Filtered (forward-only) posterior over state bins
            - causal_state_probabilities : (n_time, n_states) - if 'filter'
                Filtered discrete state probabilities
            - predictive_state_probabilities : (n_time, n_states) - if 'predictive'
                One-step-ahead predictive distributions over discrete states
            - predictive_posterior : (n_time, n_state_bins) - if 'predictive_posterior'
                One-step-ahead predictive distributions over state bins
                (Warning: can be very large, ~same size as causal_posterior)
            - log_likelihood : (n_time, n_state_bins) - if 'log_likelihood'
                Per-timepoint observation log likelihoods

        Examples
        --------
        Get only smoother (default, minimal memory):

        >>> results = model.predict(spike_times, time_edges=time_edges)
        >>> results.acausal_posterior.shape
        (10000, 50000)

        Include filtered posterior for online decoding:

        >>> results = model.predict(
        ...     spike_times, time_edges=time_edges, return_outputs='filter'
        ... )
        >>> results.causal_posterior.shape
        (10000, 50000)

        Get multiple outputs for analysis:

        >>> results = model.predict(
        ...     spike_times, time_edges=time_edges,
        ...     return_outputs=['filter', 'predictive']
        ... )

        Get everything for debugging:

        >>> results = model.predict(
        ...     spike_times, time_edges=time_edges, return_outputs='all'
        ... )
        """
        return self._predict_spikes(
            (spike_times, spike_waveform_features),
            time_edges=time_edges,
            position=position,
            position_time=position_time,
            is_missing=is_missing,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
            inference_mode=inference_mode,
            output_mode=output_mode,
            chunk_size=chunk_size,
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            selected_intervals=selected_intervals,
            max_read_bytes=max_read_bytes,
            replay_cache_bytes=replay_cache_bytes,
            return_outputs=return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            save_causal_posterior_to_results=save_causal_posterior_to_results,
        )

    @requires_time_edges
    def estimate_parameters(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        *,
        time_edges: ArrayLike,
        is_missing: np.ndarray | None = None,
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = False,
        max_iter: int = 20,
        tolerance: float = 1e-5,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: ReturnOutputs = None,
        save_log_likelihood_to_results: bool | None = None,
        min_encoding_local_mass: float = 1.0,
        min_encoding_local_ess: float = 1.0,
        encoding_update_damping: float = 0.0,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
        transition_representation: TransitionRepresentation = "auto",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
        chunk_size: int | None = None,
        result_path: str | Path | None = None,
        replay_cache_bytes: int | None = None,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray, len (n_electrodes,)
            One 1-D array of spike times per electrode, in seconds; NaN or
            infinite times raise ``DataError``.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : array_like of bool, shape (n_bins,), optional
            True (or 1) for the bins to decode as missing, one entry per
            decode bin, by default None.
        is_training : array_like of bool, shape (n_time_position,), optional
            True (or 1) for the position samples to train the encoding model
            on, one entry per sample, by default None (every sample).
        encoding_group_labels : array_like of int or str, shape (n_time_position,), optional
            Encoding group of each position sample, matched to the observation
            models' ``encoding_group``, by default None (group 0).
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment name of each position sample, by default None.
        discrete_transition_covariate_data : dict-like, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Re-fit the place fields in each M-step, weighted by the Local
            state's posterior, by default False. Every state shares the
            encoding model, so weighting it by the Local state alone is not
            an exact M-step: it can lower the penalized log-likelihood, which
            EM reports as a monotonicity violation and never counts as
            convergence. Off by default for that reason.
        max_iter : int, optional
            Maximum number of EM iterations, by default 20.
        tolerance : float, optional
            Convergence tolerance for the EM algorithm, in nats per time
            bin, by default 1e-5. EM stops when the penalized log-likelihood
            (marginal log-likelihood plus the discrete-transition log prior)
            rises by less than ``tolerance`` times the number of time bins in
            one iteration, without decreasing. With
            ``discrete_transition_prior_weight > 0`` or REML smoothness the
            last M-step must also change every transition probability by less
            than 1e-4.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
            Caching is disabled automatically when ``n_chunks > 1``; asking to
            keep the likelihood there (``store_log_likelihood=True`` or
            ``return_outputs='log_likelihood'``) makes the FINAL E-step
            accumulate the per-chunk rows into the full (n_time, n_state_bins)
            array instead. In-loop E-steps never retain a chunk array.
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihood_, by default False.
        n_chunks : int, optional
            Dense EM only: the number of row chunks the filter and smoother
            process, by default 1. Structured EM uses ``chunk_size`` (rows per
            chunk) instead.
        return_outputs : str, collection of str, or None, optional
            Controls which optional outputs are returned. See predict() for full
            documentation of options. By default None.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead. By default None.
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights to update encoding model. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size of local weights to update encoding. By default 1.0.
        encoding_update_damping : float, optional
            Must be 0.0 (the default). Damped encoding updates are not supported
            by any likelihood; a nonzero value raises ``ValidationError`` before
            any fitting. Use ``min_encoding_local_mass`` /
            ``min_encoding_local_ess`` to skip poorly supported updates.

        encoding_time_range : array_like, shape (2,), optional
            Acquisition start/stop in seconds. Clips encoding support using the
            original interpolation basis. Uniform samples otherwise include endpoint
            half-cells; a singleton requires explicit bounds or a tracking interval.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            Ordered, non-overlapping continuous tracking intervals in seconds.
            Required for irregular timestamps; each interval needs a finite sample.
            Positions are held at segment endpoints and NaN rows split support.
            Encoding support does not automatically mark decode bins missing.
        transition_representation : {"auto", "dense", "structured"}, optional
            Which EM to run, by default "auto"; keep the default unless you
            need one path. "dense" EM holds several (time bins x state bins)
            arrays, so its memory grows with the recording. "structured" EM
            runs on the checkpointed driver (see ``chunk_size``), whose device
            memory does not grow with the recording. "auto" chooses:

            - dense EM when it is faster and fits: the float32 interior
              transition is at most 16 MiB, the full dense transition fits
              ``max_dense_transition_bytes`` and dense EM's arrays fit in half
              the device memory; also when ``n_chunks`` is given, or when a
              transition has no bounded structured form;
            - structured EM otherwise, or when ``chunk_size``, ``result_path``
              or ``replay_cache_bytes`` is given.

            The results' ``em_transition_representation`` attribute records
            which ran. This is separate from ``fit``'s "auto", which only
            permits dense fallback blocks inside a structured fit.
        max_dense_transition_bytes : int, optional
            Byte budget for dense transition blocks or views of a structured
            fit, by default ``DEFAULT_MAX_DENSE_BYTES``.
        chunk_size : int, optional
            Rows per checkpoint chunk for structured-transition EM. Working
            memory scales with ``chunk_size`` times the state bins instead of
            the recording length, and each uncached chunk evaluates the
            likelihood twice (see ``replay_cache_bytes``; ``cache_likelihood``
            has no effect). The final E-step returns the same outputs as dense
            EM, held in memory unless ``result_path`` is given. Selects
            structured EM under "auto" and cannot be combined with
            ``n_chunks``. By default None, which sizes chunks so one chunk of
            float32 likelihoods takes about 256 MiB.
        result_path : str or Path, optional
            Structured-transition EM only: write the final outputs to an
            atomic directory store here and return them lazily, so host memory
            does not hold the full posterior. Cannot be combined with
            ``store_log_likelihood``.
        replay_cache_bytes : int, optional
            Structured-transition EM only: host memory for reusing each
            E-step's forward-pass rows in its backward replay instead of
            filtering and evaluating them again. Selects structured EM under
            "auto". By default None, which uses 16 GiB or half the available
            host memory (counting a container's limit), whichever is smaller.

        Returns
        -------
        results : xr.Dataset
            Results of the decoding.

        Notes
        -----
        If the call raises after its arguments are validated, the detector and
        its environments are restored to their state before the call, except
        that a stored ``log_likelihood_`` is dropped. This includes a warning
        promoted to an error (``-W error`` or
        ``warnings.simplefilter("error")``), such as EM's convergence or
        monotonicity warnings. A ``result_path`` store that the final E-step
        has written is not removed. Until the call returns, the detector keeps
        the previous fit (continuous and discrete transitions, initial
        conditions, encoding model and environment grids) for that restore,
        so EM holds this second copy; a dense continuous transition has
        ``n_state_bins ** 2`` float64 values.
        """
        return self._estimate_spike_parameters(
            position_time,
            position,
            (spike_times, spike_waveform_features),
            time_edges=time_edges,
            is_missing=is_missing,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            return_outputs=return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            encoding_update_damping=encoding_update_damping,
            encoding_time_range=encoding_time_range,
            valid_position_intervals=valid_position_intervals,
            transition_representation=transition_representation,
            max_dense_transition_bytes=max_dense_transition_bytes,
            chunk_size=chunk_size,
            result_path=result_path,
            replay_cache_bytes=replay_cache_bytes,
        )

    @requires_time_edges
    def most_likely_sequence(
        self,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        *,
        time_edges: ArrayLike,
        position: ArrayLike | None = None,
        position_time: ArrayLike | None = None,
        is_missing: np.ndarray | None = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Find the most likely sequence of states.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,), optional
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims), optional
            Position data.
        spike_times : list of np.ndarray, len (n_electrodes,)
            One 1-D array of spike times per electrode, in seconds; NaN or
            infinite times raise ``DataError``.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : array_like of bool, shape (n_bins,), optional
            True (or 1) for the bins to decode as missing, one entry per
            decode bin, by default None.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1.

        Returns
        -------
        most_likely_sequence : pd.DataFrame, shape (n_bins, n_cols)
            Most likely state and position at each decode bin, indexed by
            the bin center time.
        """
        return self._most_likely_spike_sequence(
            (spike_times, spike_waveform_features),
            time_edges=time_edges,
            position=position,
            position_time=position_time,
            is_missing=is_missing,
            n_chunks=n_chunks,
        )


class SortedSpikesDetector(_DetectorBase):
    """
    Detector class for sorted spikes.

    Decodes from the spike times of sorted units. The constructor parameters
    are documented on ``__init__``.
    """

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict[str, Any] | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ) -> None:
        """
        Initialize the SortedSpikesDetector class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        sorted_spikes_algorithm : str, optional
            Algorithm for sorted spikes, by default "sorted_spikes_kde".
        sorted_spikes_algorithm_params : dict or None, optional
            Parameters for the sorted spikes algorithm. Stored as-is (``None``
            stays ``None``) to keep sklearn's ``clone()`` contract intact. If
            ``None`` (the default), the algorithm's registered
            ``default_params`` are copied into a fresh dict **at
            fit time**; a user-supplied dict replaces those defaults, is
            **stored by reference** and copied on use. At fit time a key the
            algorithm's fit does not accept raises ``ValidationError``.
            Mutations the caller makes to their own dict between construction
            and fit are therefore observed by the fit — pass
            ``dict(my_params)`` for an isolated copy.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Rate in Hz of the decode bins; decoding ``time_edges`` must be
            ``1 / sampling_frequency`` wide. By default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Data-adaptive prior weight, by default 0.0. See
            ``_DetectorBase`` for details.
        frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
            Rows of the discrete transition matrix to freeze during EM
            re-estimation, by default ``"no_spike"`` (the No-Spike states'
            rows). See ``_DetectorBase`` for details.
        local_position_std : float or None, optional
            Standard deviation of the position uncertainty kernel for the
            local state. See ``_DetectorBase`` for details.
        """
        super().__init__(
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=continuous_initial_conditions_types,
            discrete_transition_type=discrete_transition_type,
            discrete_transition_concentration=discrete_transition_concentration,
            discrete_transition_stickiness=discrete_transition_stickiness,
            discrete_transition_regularization=discrete_transition_regularization,
            continuous_transition_types=continuous_transition_types,
            observation_models=observation_models,
            environments=environments,
            infer_track_interior=infer_track_interior,
            state_names=state_names,
            sampling_frequency=sampling_frequency,
            no_spike_rate=no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        self.sorted_spikes_algorithm = sorted_spikes_algorithm
        self._sorted_spikes_algorithm()  # rejects an unregistered name
        # Stored as-is to keep sklearn's clone() contract intact; the dict
        # copy + default resolution happens lazily in
        # ``_resolve_sorted_spikes_algorithm_params`` at fit time.
        self.sorted_spikes_algorithm_params = sorted_spikes_algorithm_params

    def _resolve_sorted_spikes_algorithm_params(self) -> dict[str, Any]:
        """Return a fresh, validated copy of the sorted-spikes algorithm params.

        Resolves a ``None`` user spec to the algorithm's registered
        ``default_params``; otherwise copies the user's dict.
        Raises ``ValidationError`` for a key the registered fit does not
        accept.
        """
        return _resolve_fit_params(
            _SORTED_SPIKES_ALGORITHMS,
            self.sorted_spikes_algorithm,
            self.sorted_spikes_algorithm_params,
            "sorted_spikes_algorithm_params",
        )

    def _sorted_spikes_algorithm(
        self,
    ) -> LikelihoodAlgorithm[SortedSpikesFit, SortedSpikesPredict]:
        """Return the registered record of ``sorted_spikes_algorithm``."""
        try:
            return _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm]
        except KeyError:
            msg = f"Unknown sorted_spikes_algorithm {self.sorted_spikes_algorithm!r}"
            raise ValidationError(
                msg,
                expected=f"one of {sorted(_SORTED_SPIKES_ALGORITHMS)}",
                got=repr(self.sorted_spikes_algorithm),
            ) from None

    def fit_encoding_model(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        *,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
    ) -> None:
        """
        Fit place fields to the data.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray, len (n_neurons,)
            One 1-D array of spike times per neuron, in seconds; NaN or
            infinite times raise ``DataError``.
        is_training : array_like of bool, shape (n_time_position,), optional
            True (or 1) for the position samples to train the encoding model
            on, one entry per sample, by default None (every sample).
        encoding_group_labels : array_like of int or str, shape (n_time_position,), optional
            Encoding group of each position sample, matched to the observation
            models' ``encoding_group``, by default None (group 0).
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment name of each position sample, by default None.
        weights : np.ndarray, optional, shape (n_time_position,)
            Weights for training data, by default None.

        encoding_time_range : array_like, shape (2,), optional
            Acquisition start/stop in seconds. Clips encoding support using the
            original interpolation basis. Uniform samples otherwise include endpoint
            half-cells; a singleton requires explicit bounds or a tracking interval.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            Ordered, non-overlapping continuous tracking intervals in seconds.
            Required for irregular timestamps; each interval needs a finite sample.
            Positions are held at segment endpoints and NaN rows split support.
            Encoding support does not automatically mark decode bins missing.

        Attributes
        ----------
        encoding_model_ : dict
            Dictionary holding the fitted encoding models (place fields) for each unique
            observation model configuration (environment, encoding group).
            The values depend on the chosen `sorted_spikes_algorithm`.
        """
        self._check_encoding_spike_data(position_time, (spike_times,))
        self._fit_encoding_model(
            position_time,
            position,
            spike_times,
            is_training,
            encoding_group_labels,
            environment_labels,
            weights,
            encoding_time_range=encoding_time_range,
            valid_position_intervals=valid_position_intervals,
        )

    def _fit_encoding_model(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        *,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
    ) -> None:
        """Body of ``fit_encoding_model``, which EM encoding updates call."""
        logger.info("Fitting place fields...")
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            weights,
        )
        position_time, weights = inputs.position_time, inputs.weights
        position = (
            inputs.position[:, np.newaxis]
            if inputs.position.ndim == 1
            else inputs.position
        )
        is_training, encoding_group_labels, environment_labels = self._encoding_masks(
            inputs
        )
        support = EncodingSupport(
            position_time,
            position,
            encoding_time_range=encoding_time_range,
            valid_position_intervals=valid_position_intervals,
        )

        kwargs = self._resolve_sorted_spikes_algorithm_params()
        encoding_algorithm = self._sorted_spikes_algorithm().fit

        def fit_group(environment: Environment, group_weights: np.ndarray) -> Any:
            return encoding_algorithm(
                position_time=position_time,
                position=position,
                spike_times=[
                    neuron_spike_times[
                        _group_spike_mask(
                            neuron_spike_times, position_time, group_weights, support
                        )
                    ]
                    for neuron_spike_times in spike_times
                ],
                environment=environment,
                _encoding_support=support,
                weights=group_weights,
                **kwargs,
            )

        encoding_model = self._fit_encoding_groups(
            fit_group,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            weights=weights,
            support=support,
        )

        self._invalidate_stored_log_likelihood()  # stale: encoding model replaced
        self.encoding_model_ = encoding_model
        self.time_contract_ = dict(_TIME_CONTRACT)

    def fit(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        *,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
        transition_representation: TransitionRepresentation = "dense",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> "SortedSpikesDetector":
        """
        Fit the detector to the data.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray, len (n_neurons,)
            One 1-D array of spike times per neuron, in seconds; NaN or
            infinite times raise ``DataError``.
        is_training : array_like of bool, shape (n_time_position,), optional
            True (or 1) for the position samples to train the encoding model
            on, one entry per sample, by default None (every sample).
        encoding_group_labels : array_like of int or str, shape (n_time_position,), optional
            Encoding group of each position sample, matched to the observation
            models' ``encoding_group``, by default None (group 0).
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment name of each position sample, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        encoding_time_range : array_like, shape (2,), optional
            Acquisition start/stop in seconds. Clips encoding support using the
            original interpolation basis. Uniform samples otherwise include endpoint
            half-cells; a singleton requires explicit bounds or a tracking interval.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            Ordered, non-overlapping continuous tracking intervals in seconds.
            Required for irregular timestamps; each interval needs a finite sample.
            Positions are held at segment endpoints and NaN rows split support.
            Encoding support does not automatically mark decode bins missing.

        transition_representation : {"dense", "structured", "auto"}, optional
            By default "dense", which builds the full transition matrix.
            "structured" requires exact supported state-block operators;
            "auto" permits budgeted dense fallback blocks. Non-dense N-D
            environments defer all-pairs graph distances.
        max_dense_transition_bytes : int, optional
            Byte limit for retained fallback blocks, temporary dense graph
            distances, and explicit reads of the lazy transition and distance
            views, by default 256 MiB. Ignored when
            transition_representation="dense".

        Returns
        -------
        SortedSpikesDetector
            Fitted detector instance.
        """
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        # Before ``_fit``; fit_encoding_model checks again and warns.
        self._check_spike_data((spike_times,))
        with self._restore_state_on_error():
            self._fit(
                inputs.position,
                inputs.is_training,
                inputs.encoding_group_labels,
                inputs.environment_labels,
                discrete_transition_covariate_data,
                transition_representation=transition_representation,
                max_dense_transition_bytes=max_dense_transition_bytes,
            )
            self.fit_encoding_model(
                inputs.position_time,
                inputs.position,
                spike_times,
                inputs.is_training,
                inputs.encoding_group_labels,
                inputs.environment_labels,
                encoding_time_range=encoding_time_range,
                valid_position_intervals=valid_position_intervals,
            )
        return self

    _SPIKE_DATA_NAMES = ("spike_times",)
    _PREDICT_EXAMPLE = (
        "    results = detector.predict(\n"
        "        spike_times=spike_times, time_edges=time_edges,\n"
        "        position=position, position_time=position_time\n"
        "    )"
    )

    def _likelihood_predict_function(self) -> SortedSpikesPredict:
        return self._sorted_spikes_algorithm().predict

    def _likelihood_execution_settings(self) -> dict[str, Any]:
        return _execution_settings(
            self._sorted_spikes_algorithm(), self.sorted_spikes_algorithm_params
        )

    @row_slice_aware
    @requires_time_edges
    def compute_log_likelihood(
        self,
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        spike_times: list[np.ndarray],
        is_missing: np.ndarray | None = None,
        row_slice: slice | None = None,
        *,
        time_edges: ArrayLike,
    ) -> jax.Array:
        """
        Compute the log likelihood for the given data.

        Notes
        -----
        When ``local_position_std`` is set, the per-bin value returned
        for local-state entries is not a pure observation likelihood.
        It combines the spatial spike likelihood with the position
        uncertainty kernel::

            log_likelihood[local, b, t] =
                log P(spikes_t | bin b)                   # spatial likelihood
              + log_kernel(b | animal_pos_t)              # Gaussian anchor
              + log(n_interior_bins)                      # mass-balance

        The kernel term is a time-varying state-specific spatial prior.
        Mathematically, injecting it into the likelihood is equivalent
        to injecting it into the transition matrix — both multiply into
        the HMM forward step — but avoids breaking the static-transition
        assumption used by ``jax.lax.scan``. The ``log(n_interior_bins)``
        constant compensates for the ``1/n_bins`` uniform continuous IC
        so the effective per-bin prior is ``exp(log_kernel)`` itself.

        The kernel is part of the likelihood model, not an ad-hoc
        post-processing step, so the posterior returned by ``predict()``
        is a valid probability distribution under the modified
        generative model. Users comparing ``local_position_std=None`` to
        a finite value will see different posteriors (the models
        differ); users reading the raw ``log_likelihood`` (via
        ``return_outputs='log_likelihood'``) should be aware that
        local-state entries are *not* pure ``log P(spikes | state, bin)``
        — they include the kernel and mass-balance terms.

        Parameters
        ----------
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        position_time : array_like, shape (n_time_position,), or None
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims), or None
            Position data.
        spike_times : list of np.ndarray, len (n_neurons,)
            One 1-D array of spike times per neuron, in seconds; NaN or
            infinite times raise ``DataError``.
        is_missing : array_like of bool, shape (n_rows,), optional
            True (or 1) for the bins to decode as missing, one entry per
            requested decode bin, by default None.
        row_slice : slice | None, optional
            Contiguous range of rows of the full-time likelihood to compute, by
            default None (all rows). ``time_edges`` stay the FULL decoding edges,
            so a decoding spike is owned by its global row no matter how the
            rows were chunked, and the result equals the full-time likelihood
            sliced by ``row_slice``.

        Returns
        -------
        log_likelihood : jax.Array, shape (n_rows, n_state_bins)
            ``n_rows`` is ``n_time`` unless ``row_slice`` is given.
        """
        return self._compute_spike_log_likelihood(
            position_time,
            position,
            (spike_times,),
            is_missing,
            row_slice,
            time_edges=time_edges,
        )

    @requires_time_edges
    def predict(
        self,
        spike_times: list[np.ndarray],
        *,
        time_edges: ArrayLike,
        position: ArrayLike | None = None,
        position_time: ArrayLike | None = None,
        is_missing: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        cache_likelihood: bool = False,
        n_chunks: int = 1,
        inference_mode: InferenceMode = "dense",
        output_mode: OutputMode = "spatial",
        chunk_size: int | None = None,
        result_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        selected_intervals: np.ndarray | None = None,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        replay_cache_bytes: int | None = None,
        return_outputs: ReturnOutputs = None,
        save_log_likelihood_to_results: bool | None = None,
        save_causal_posterior_to_results: bool | None = None,
    ) -> xr.Dataset:
        """
        Predict the posterior probabilities for the given data.

        Parameters
        ----------
        spike_times : list of np.ndarray, len (n_neurons,)
            One 1-D array of spike times per neuron, in seconds; NaN or
            infinite times raise ``DataError``.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        position : array_like, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : array_like, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : array_like of bool, shape (n_bins,), optional
            True (or 1) for the bins to decode as missing, one entry per
            decode bin, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default False.
            Caching is disabled automatically when ``n_chunks > 1``. Requesting
            ``return_outputs='log_likelihood'`` with ``n_chunks > 1`` instead
            accumulates the per-chunk rows into the full
            (n_time, n_state_bins) array -- the one exception to per-chunk
            likelihood allocation; when the log likelihood is not requested, no
            chunk array is retained.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1.
        return_outputs : str, collection of str, or None, optional
            Controls which optional outputs are returned. See ClusterlessDetector.predict
            for full documentation. By default None.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead. By default None.
        save_causal_posterior_to_results : bool, optional
            DEPRECATED. Use return_outputs='filter' instead. By default None.

        inference_mode : {"dense", "checkpointed"}, optional
            By default "dense", the in-memory driver. "checkpointed" stores
            boundary messages on disk and replays chunks for exact smoothing;
            it raises ValueError unless n_chunks=1 and cache_likelihood=False.
        output_mode : {"compact", "spatial"}, optional
            By default "spatial", the only output dense prediction accepts.
            Checkpointed "compact" returns state probabilities without
            full-session spatial arrays; checkpointed "spatial" writes lazy
            incremental arrays and requires result_path.
        chunk_size : int, optional
            Maximum checkpoint/replay rows (checkpointed mode only). By
            default, enough rows for one chunk of float32 likelihoods to take
            about 256 MiB, and at least 256. Larger chunks amortize per-chunk
            overhead; working memory grows with chunk rows, to several times
            one likelihood chunk. docs/hardware_settings.md lists measured
            memory and speed.
        result_path : str or Path, optional
            New directory for atomic incremental output. Existing destinations
            are never overwritten. Optional for compact, required for spatial.
        checkpoint_dir : str or Path, optional
            Parent directory for temporary boundary files, cleaned after the run.
        selected_intervals : array_like, shape (n_intervals, 2), optional
            Ordered non-overlapping start/stop selections. Emit only bins fully
            inside them while conditioning on the entire supplied recording.
        max_read_bytes : int, optional
            Maximum bytes in one explicit spatial read, by default 512 MiB.
            Select fewer rows/bins before converting a large lazy array.
        replay_cache_bytes : int, optional
            Checkpointed mode only: host memory for keeping forward-pass rows
            (filtered rows, or likelihoods when the likelihood or predictive
            outputs are requested) so the backward replay does not recompute
            them. Chunks beyond the budget are recomputed. By default None,
            which uses 16 GiB or half the available host memory, whichever is
            smaller.

        Returns
        -------
        xr.Dataset
            Predicted posterior probabilities.
        """
        return self._predict_spikes(
            (spike_times,),
            time_edges=time_edges,
            position=position,
            position_time=position_time,
            is_missing=is_missing,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
            inference_mode=inference_mode,
            output_mode=output_mode,
            chunk_size=chunk_size,
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            selected_intervals=selected_intervals,
            max_read_bytes=max_read_bytes,
            replay_cache_bytes=replay_cache_bytes,
            return_outputs=return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            save_causal_posterior_to_results=save_causal_posterior_to_results,
        )

    @requires_time_edges
    def estimate_parameters(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: list[np.ndarray],
        *,
        time_edges: ArrayLike,
        is_missing: np.ndarray | None = None,
        is_training: ArrayLike | None = None,
        encoding_group_labels: ArrayLike | None = None,
        environment_labels: ArrayLike | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict[str, Any] | None = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = False,
        max_iter: int = 20,
        tolerance: float = 1e-5,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: ReturnOutputs = None,
        save_log_likelihood_to_results: bool | None = None,
        min_encoding_local_mass: float = 1.0,
        min_encoding_local_ess: float = 1.0,
        encoding_update_damping: float = 0.0,
        encoding_time_range: np.ndarray | None = None,
        valid_position_intervals: np.ndarray | None = None,
        transition_representation: TransitionRepresentation = "auto",
        max_dense_transition_bytes: int = DEFAULT_MAX_DENSE_BYTES,
        chunk_size: int | None = None,
        result_path: str | Path | None = None,
        replay_cache_bytes: int | None = None,
    ) -> xr.Dataset:
        """Estimate the initial conditions and transition probabilities.

        Uses the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time of each position sample.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            One 1-D array of spike times per neuron, in seconds; NaN or
            infinite times raise ``DataError``.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : array_like of bool, shape (n_bins,), optional
            True (or 1) for the bins to decode as missing, one entry per
            decode bin, by default None.
        is_training : array_like of bool, shape (n_time_position,), optional
            True (or 1) for the position samples to train the encoding model
            on, one entry per sample, by default None (every sample).
        encoding_group_labels : array_like of int or str, shape (n_time_position,), optional
            Encoding group of each position sample, matched to the observation
            models' ``encoding_group``, by default None (group 0).
        environment_labels : array_like of str, shape (n_time_position,), optional
            Environment name of each position sample, by default None.
        discrete_transition_covariate_data : dict, optional
            Covariate data for a covariate dependent discrete transition.
            A dict-like object that can be used to look up variables, by default None.
        estimate_initial_conditions : bool, optional
            Estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Re-fit the place fields in each M-step, weighted by the Local
            state's posterior, by default False. Every state shares the
            encoding model, so weighting it by the Local state alone is not
            an exact M-step: it can lower the penalized log-likelihood, which
            EM reports as a monotonicity violation and never counts as
            convergence. Off by default for that reason.
        max_iter : int, optional
            Maximum number of EM iterations, by default 20.
        tolerance : float, optional
            Convergence tolerance for EM, in nats per time bin, by default
            1e-5. EM stops when the penalized log-likelihood (marginal
            log-likelihood plus the discrete-transition log prior) rises by
            less than ``tolerance`` times the number of time bins in one
            iteration, without decreasing. With
            ``discrete_transition_prior_weight > 0`` or REML smoothness the
            last M-step must also change every transition probability by less
            than 1e-4.
        cache_likelihood : bool, optional
            Store the likelihood for faster iterations, by default True
            Caching is disabled automatically when ``n_chunks > 1``; asking to
            keep the likelihood there (``store_log_likelihood=True`` or
            ``return_outputs='log_likelihood'``) makes the FINAL E-step
            accumulate the per-chunk rows into the full (n_time, n_state_bins)
            array instead. In-loop E-steps never retain a chunk array.
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihood_, by default False.
        n_chunks : int, optional
            Dense EM only: the number of row chunks the filter and smoother
            process, by default 1. Structured EM uses ``chunk_size`` (rows per
            chunk) instead.
        return_outputs : str, collection of str, or None, optional
            Controls which optional outputs are returned. See predict() for full
            documentation of options. By default None.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead. By default None.
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights to update encoding model. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size of local weights to update encoding. By default 1.0.
        encoding_update_damping : float, optional
            Must be 0.0 (the default). Damped encoding updates are not supported
            by any likelihood; a nonzero value raises ``ValidationError`` before
            any fitting. Use ``min_encoding_local_mass`` /
            ``min_encoding_local_ess`` to skip poorly supported updates.

        encoding_time_range : array_like, shape (2,), optional
            Acquisition start/stop in seconds. Clips encoding support using the
            original interpolation basis. Uniform samples otherwise include endpoint
            half-cells; a singleton requires explicit bounds or a tracking interval.
        valid_position_intervals : array_like, shape (n_intervals, 2), optional
            Ordered, non-overlapping continuous tracking intervals in seconds.
            Required for irregular timestamps; each interval needs a finite sample.
            Positions are held at segment endpoints and NaN rows split support.
            Encoding support does not automatically mark decode bins missing.
        transition_representation : {"auto", "dense", "structured"}, optional
            Which EM to run, by default "auto"; keep the default unless you
            need one path. "dense" EM holds several (time bins x state bins)
            arrays, so its memory grows with the recording. "structured" EM
            runs on the checkpointed driver (see ``chunk_size``), whose device
            memory does not grow with the recording. "auto" chooses:

            - dense EM when it is faster and fits: the float32 interior
              transition is at most 16 MiB, the full dense transition fits
              ``max_dense_transition_bytes`` and dense EM's arrays fit in half
              the device memory; also when ``n_chunks`` is given, or when a
              transition has no bounded structured form;
            - structured EM otherwise, or when ``chunk_size``, ``result_path``
              or ``replay_cache_bytes`` is given.

            The results' ``em_transition_representation`` attribute records
            which ran. This is separate from ``fit``'s "auto", which only
            permits dense fallback blocks inside a structured fit.
        max_dense_transition_bytes : int, optional
            Byte budget for dense transition blocks or views of a structured
            fit, by default ``DEFAULT_MAX_DENSE_BYTES``.
        chunk_size : int, optional
            Rows per checkpoint chunk for structured-transition EM. Working
            memory scales with ``chunk_size`` times the state bins instead of
            the recording length, and each uncached chunk evaluates the
            likelihood twice (see ``replay_cache_bytes``; ``cache_likelihood``
            has no effect). The final E-step returns the same outputs as dense
            EM, held in memory unless ``result_path`` is given. Selects
            structured EM under "auto" and cannot be combined with
            ``n_chunks``. By default None, which sizes chunks so one chunk of
            float32 likelihoods takes about 256 MiB.
        result_path : str or Path, optional
            Structured-transition EM only: write the final outputs to an
            atomic directory store here and return them lazily, so host memory
            does not hold the full posterior. Cannot be combined with
            ``store_log_likelihood``.
        replay_cache_bytes : int, optional
            Structured-transition EM only: host memory for reusing each
            E-step's forward-pass rows in its backward replay instead of
            filtering and evaluating them again. Selects structured EM under
            "auto". By default None, which uses 16 GiB or half the available
            host memory (counting a container's limit), whichever is smaller.

        Returns
        -------
        xr.Dataset
            Results of the decoding

        Notes
        -----
        If the call raises after its arguments are validated, the detector and
        its environments are restored to their state before the call, except
        that a stored ``log_likelihood_`` is dropped. This includes a warning
        promoted to an error (``-W error`` or
        ``warnings.simplefilter("error")``), such as EM's convergence or
        monotonicity warnings. A ``result_path`` store that the final E-step
        has written is not removed. Until the call returns, the detector keeps
        the previous fit (continuous and discrete transitions, initial
        conditions, encoding model and environment grids) for that restore,
        so EM holds this second copy; a dense continuous transition has
        ``n_state_bins ** 2`` float64 values.
        """
        return self._estimate_spike_parameters(
            position_time,
            position,
            (spike_times,),
            time_edges=time_edges,
            is_missing=is_missing,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            return_outputs=return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            encoding_update_damping=encoding_update_damping,
            encoding_time_range=encoding_time_range,
            valid_position_intervals=valid_position_intervals,
            transition_representation=transition_representation,
            max_dense_transition_bytes=max_dense_transition_bytes,
            chunk_size=chunk_size,
            result_path=result_path,
            replay_cache_bytes=replay_cache_bytes,
        )

    @requires_time_edges
    def most_likely_sequence(
        self,
        spike_times: list[np.ndarray],
        *,
        time_edges: ArrayLike,
        position: ArrayLike | None = None,
        position_time: ArrayLike | None = None,
        is_missing: np.ndarray | None = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Find the most likely sequence of states.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,), optional
            Time of each position sample.
        position : array_like, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            One 1-D array of spike times per neuron, in seconds; NaN or
            infinite times raise ``DataError``.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : array_like of bool, shape (n_bins,), optional
            True (or 1) for the bins to decode as missing, one entry per
            decode bin, by default None.
        n_chunks : int, optional
            Number of chunks for processing, by default 1.

        Returns
        -------
        most_likely_sequence : pd.DataFrame, shape (n_bins, n_cols)
            Most likely state and position at each decode bin, indexed by
            the bin center time.
        """
        return self._most_likely_spike_sequence(
            (spike_times,),
            time_edges=time_edges,
            position=position,
            position_time=position_time,
            is_missing=is_missing,
            n_chunks=n_chunks,
        )


# The families' own ``compute_log_likelihood``, recorded when the classes are
# defined so that a later replacement on a family class is told apart from it.
_PACKAGE_LOG_LIKELIHOODS: frozenset[Callable[..., Any]] = frozenset(
    {
        ClusterlessDetector.__dict__["compute_log_likelihood"],
        SortedSpikesDetector.__dict__["compute_log_likelihood"],
    }
)
