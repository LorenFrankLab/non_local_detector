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
"""

import abc
import copy
import inspect
import json
import pickle
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from logging import getLogger
from pathlib import Path
from typing import Any, cast, get_args

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from patsy import DesignMatrix, build_design_matrices
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from non_local_detector import _validation as val
from non_local_detector._position_dims import get_position_dim_names
from non_local_detector._warnings import user_stacklevel
from non_local_detector.checkpointed_inference import (
    CheckpointedResult,
    _available_host_bytes,
    default_chunk_size,
    default_replay_cache_bytes,
)
from non_local_detector.continuous_state_transitions import (
    TransitionContext,
    _make_transition_block,
    _multi_bin_local_transition,
    _scalar_to_spatial_destination,
)
from non_local_detector.core import (
    accepts_row_slice,
    check_converged,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
    row_slice_aware,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteStationaryCustom,
    _covariate_transition_covariance,
    _discrete_transition_from_statistics,
    _discrete_transition_statistics,
    _is_nonstationary,
    _max_transition_change,
    _row_smoothness,
    _smoothness_matrix,
    _transition_intervals,
    _transition_log_prior,
    centered_softmax_forward,
    predict_discrete_state_transitions,
)
from non_local_detector.encoding_time import EncodingSupport
from non_local_detector.environment import Environment, find_environment_by_name
from non_local_detector.exceptions import (
    DataError,
    ValidationError,
)
from non_local_detector.graph_distances import DEFAULT_MAX_DENSE_DISTANCE_BYTES
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    LikelihoodAlgorithm,
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
from non_local_detector.models._results import (
    convert_results_to_xarray,
    convert_seq_to_df,
    has_place_bins,
    label_checkpointed_result,
    state_bin_positions,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.result_store import DEFAULT_MAX_READ_BYTES
from non_local_detector.time_edges import (
    _MAX_RELATIVE_SPACING_TOLERANCE,
    _DecodeTimeGrid,
    _uniform_time_grid,
    _uniformity_tolerance,
    calculate_time_edges,
    decode_bin_centers,
    requires_time_edges,
    uniform_time_edges,
)
from non_local_detector.transition_operators import DEFAULT_MAX_DENSE_BYTES

try:
    from non_local_detector._version import __version__ as _PACKAGE_VERSION
except ImportError:
    _PACKAGE_VERSION = "unknown"
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

logger = getLogger(__name__)

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


@dataclass(frozen=True)
class _LikelihoodPositionContext:
    """Tracking preparation shared only during one likelihood prediction."""

    time_edges: np.ndarray
    source_time: np.ndarray | None
    source_position: np.ndarray | None
    position: np.ndarray | None
    is_missing: np.ndarray | None
    all_missing: bool
    reads_position: bool

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
        return cls(
            time_edges,
            position_time,
            position,
            prepared,
            missing,
            all_missing,
            reads_position,
        )

    def matches(
        self,
        time_edges: np.ndarray,
        position_time: np.ndarray | None,
        position: np.ndarray | None,
        reads_position: bool,
    ) -> bool:
        return (
            self.time_edges is time_edges
            and self.source_time is position_time
            and self.source_position is position
            and self.reads_position == reads_position
        )


def _prepare_likelihood_callback(
    callback: Callable[..., Any],
    time_edges: np.ndarray,
    *,
    has_no_spike: bool,
    log_likelihood_args: tuple[Any, ...] = (),
    _time_grid: _DecodeTimeGrid | None = None,
    reads_position: bool = True,
) -> Callable[..., Any]:
    """Adapt a detector likelihood to core's observation-row callback contract.

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
    bins.

    Declaring ``_spike_time_order`` or ``_no_spike_time_bin_sizes`` lets this
    adapter bind shared preparation. ``reads_position`` is the detector's
    decision whether its decode likelihood reads position; the bound position
    context marks bins missing for NaN or untracked position only when it
    does. Each call creates fresh preparation; nothing is stored on the
    detector or reused by a later prediction. Callers with a precomputed
    likelihood bypass this adapter entirely.
    """
    parameters = inspect.signature(callback).parameters
    prepared: dict[str, object] = {}
    if "_spike_time_order" in parameters:
        prepared["_spike_time_order"] = _SpikeTimeOrder()
    if "_time_grid" in parameters and _time_grid is not None:
        prepared["_time_grid"] = _time_grid
    if "_position_context" in parameters and len(log_likelihood_args) >= 2:
        prepared["_position_context"] = _LikelihoodPositionContext.prepare(
            time_edges, *log_likelihood_args[:2], reads_position=reads_position
        )
    if has_no_spike and "_no_spike_time_bin_sizes" in parameters:
        prepared["_no_spike_time_bin_sizes"] = no_spike_time_bin_sizes(time_edges)
    bound = partial(callback, **prepared) if prepared else callback

    n_bins = len(time_edges) - 1
    edges_are_keyword_only = (
        "time_edges" in parameters
        and parameters["time_edges"].kind is inspect.Parameter.KEYWORD_ONLY
    )

    def evaluate(edges: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if edges_are_keyword_only:
            return bound(*args, time_edges=edges, **kwargs)
        return bound(edges, *args, **kwargs)

    def checked(result: Any, n_rows: int) -> Any:
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

    if accepts_row_slice(callback):

        def row_callback(
            row_time: Any,  # noqa: ARG001 -- likelihood callback signature
            *args: Any,
            row_slice: slice | None = None,
            **kwargs: Any,
        ) -> Any:
            row_start, row_stop = resolve_row_slice(row_slice, n_bins)
            return checked(
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
        return checked(evaluate(time_edges, *args, **kwargs), n_bins)

    row_slice_aware(chunk_callback)  # marks the function in place
    return chunk_callback


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


def _validate_encoding_update_damping(encoding_update_damping: float) -> None:
    """Reject any nonzero ``encoding_update_damping``.

    Blending only ``place_fields`` cannot keep a refit encoding model
    self-consistent: the GLM and KDE local paths read other fitted state, the
    diffusion/MRF spike term reads ``interior_log_place_fields``, and the
    clusterless models have no ``place_fields`` at all. Estimation wrappers call
    this before their initial ``fit`` so a rejected value changes nothing.

    Parameters
    ----------
    encoding_update_damping : float

    Raises
    ------
    ValidationError
        If the value is not zero (including NaN).
    """
    if encoding_update_damping != 0.0:
        msg = "encoding_update_damping is not supported"
        raise ValidationError(
            msg,
            expected="encoding_update_damping=0.0",
            got=str(encoding_update_damping),
            hint=(
                "Damped encoding updates are not implemented for any likelihood: "
                "blending place fields leaves the other fitted encoding state "
                "stale. Use min_encoding_local_mass / min_encoding_local_ess to "
                "skip poorly supported encoding updates instead."
            ),
        )


# Dataset attribute in which ``save_results`` records integer dtypes.
_INTEGER_DTYPES_ATTR = "non_local_detector_integer_dtypes"

# EM with transition_representation="auto" runs dense EM up to this many bytes
# of float32 interior transition. A dense step reads the whole matrix, so its
# cost grows with this size; a structured step costs about the same at any
# size. Three warm EM iterations on a 23.6 min clusterless 2-D recording (A100)
# took 181/216 s dense/structured at 5.7 MB (1,190 interior state bins),
# 250/276 s at 13.5 MB, 348/255 s at 21.9 MB and 501/362 s at 43.5 MB (3,298
# bins); interpolating puts the crossover near 15 MB.
_AUTO_DENSE_TRANSITION_BYTES = 16 * 2**20
# Dense EM holds about this many float32 (time bins x state bins) arrays on the
# device (measured device peak 10.0 GB, about seven 709k x 498 arrays, on the
# clusterless track recording).
_DENSE_EM_WORKING_ARRAYS = 8


# With a prior that changes each E-step (data-adaptive pseudo-counts, REML
# smoothness), EM also requires the last M-step to move every transition
# probability by less than this. It is a probability, so it is separate from
# ``tolerance`` (nats per time bin).
_TRANSITION_CHANGE_TOLERANCE = 1e-4


def _device_memory_limit() -> int | None:
    """Bytes JAX may allocate on the default device, or None if unknown (CPU)."""
    stats = jax.local_devices()[0].memory_stats()
    return None if not stats else stats.get("bytes_limit")


def _structured_em_options(
    chunk_size: int | None,
    result_path: str | Path | None,
    replay_cache_bytes: int | None,
) -> list[str]:
    """Names of the structured-EM options that were given."""
    return [
        name
        for name, value in (
            ("chunk_size", chunk_size),
            ("result_path", result_path),
            ("replay_cache_bytes", replay_cache_bytes),
        )
        if value is not None
    ]


def _validate_em_options(
    structured: bool,
    n_chunks: int,
    chunk_size: int | None,
    result_path: str | Path | None = None,
    store_log_likelihood: bool = False,
    replay_cache_bytes: int | None = None,
) -> None:
    """Reject options that do not apply to the EM path.

    Parameters
    ----------
    structured : bool
        Whether EM runs on a structured transition operator.
    n_chunks : int
        Dense-EM row chunks.
    chunk_size : int or None
        Checkpoint chunk rows for structured EM.
    result_path : path-like or None
        Store for structured EM's final outputs.
    store_log_likelihood : bool
        Whether EM keeps the full likelihood in memory.
    replay_cache_bytes : int or None
        Structured EM's replay cache budget.
    """
    for name, value, minimum in (
        ("chunk_size", chunk_size, 1),
        ("replay_cache_bytes", replay_cache_bytes, 0),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))  # type: ignore[redundant-expr]  # runtime check of caller input
            or value < minimum
        ):
            msg = f"Invalid {name} for estimate_parameters"
            raise ValidationError(
                msg,
                expected=f"an integer >= {minimum}, or None for the default",
                got=f"{value!r} ({type(value).__name__})",
                hint=(
                    "Pass whole rows, e.g. chunk_size=20_000."
                    if name == "chunk_size"
                    else "Pass bytes as an integer, e.g. replay_cache_bytes=8 * 2**30."
                ),
            )
    if structured and n_chunks != 1:
        msg = "n_chunks applies only to dense EM"
        raise ValidationError(
            msg,
            got=f"n_chunks={n_chunks} with structured EM",
            hint=(
                "Structured EM already bounds memory with checkpoint chunks: set "
                "chunk_size (rows per chunk) instead, or pass "
                'transition_representation="dense" to estimate_parameters to '
                "run dense EM in n_chunks chunks."
            ),
        )
    if not structured:
        checkpoint_options = _structured_em_options(
            chunk_size, result_path, replay_cache_bytes
        )
        if checkpoint_options:
            msg = f"{', '.join(checkpoint_options)} applies only to structured EM"
            raise ValidationError(
                msg,
                got=f"{', '.join(checkpoint_options)} with dense EM",
                hint=(
                    'Pass transition_representation="structured" or "auto" to '
                    "estimate_parameters, or drop "
                    f"{', '.join(checkpoint_options)} for dense EM."
                ),
            )
    if result_path is not None and store_log_likelihood:
        msg = "store_log_likelihood cannot be combined with result_path"
        raise ValidationError(
            msg,
            got="store_log_likelihood=True and a result_path",
            hint=(
                "store_log_likelihood keeps the full likelihood in memory; with "
                "result_path, request return_outputs='log_likelihood' to write "
                "it to the store instead."
            ),
        )


def _dense_em_memory(n_time: int, n_bins: int) -> tuple[int, int | None, str]:
    """Bytes one-chunk dense EM holds, the memory it draws on, and where.

    ``_DENSE_EM_WORKING_ARRAYS`` float32 time-by-state-bin arrays, against the
    device limit, or the available host memory on CPU (None if unknown).

    Parameters
    ----------
    n_time : int
        Decode bins.
    n_bins : int
        Interior state bins.

    Returns
    -------
    needed : int
    limit : int or None
    where : {"device", "host"}
    """
    needed = _DENSE_EM_WORKING_ARRAYS * n_time * n_bins * 4
    limit = _device_memory_limit()
    if limit is not None:
        return needed, limit, "device"
    return needed, _available_host_bytes(), "host"


def _warn_if_dense_em_exceeds_memory(n_time: int, n_bins: int) -> None:
    """Warn before one-chunk dense EM whose arrays exceed half the memory.

    The line "auto" uses (see ``_dense_em_memory``). On the 23.6 min 2-D
    recording (A100) the estimate was 27.0 and 41.8 GB at 4 and 3 cm
    (measured peaks 23.7 and 36.6 GB) and 53.1 GB at 2.5 cm, which ran out of
    a 63.8 GB limit, so the full limit is not a safe line. The estimate is
    approximate, so this warns rather than raising.

    Parameters
    ----------
    n_time : int
        Decode bins.
    n_bins : int
        Interior state bins.
    """
    needed, limit, where = _dense_em_memory(n_time, n_bins)
    if limit is None or needed <= limit // 2:
        return
    warnings.warn(
        f"Dense EM needs about {needed / 1e9:.1f} GB of {where} memory for "
        f"{n_time} time bins x {n_bins} state bins, more than half of the "
        f"{limit / 1e9:.1f} GB available, and may run out of memory. "
        'Pass transition_representation="auto" (or "structured") to '
        "estimate_parameters to bound EM's memory, or n_chunks > 1 for dense EM.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def _resolve_em_options(
    transition_representation: TransitionRepresentation,
    n_chunks: int,
    chunk_size: int | None,
    result_path: str | Path | None,
    store_log_likelihood: bool,
    replay_cache_bytes: int | None,
) -> TransitionRepresentation:
    """Resolve "auto" with ``n_chunks`` to dense EM, then validate the options.

    ``n_chunks`` is a dense-EM option and ``chunk_size``, ``result_path`` and
    ``replay_cache_bytes`` are structured-EM options, so with "auto" the one
    given decides the path; both together are rejected.

    Returns
    -------
    transition_representation : str
    """
    checkpoint_options = _structured_em_options(
        chunk_size, result_path, replay_cache_bytes
    )
    if transition_representation == "auto" and n_chunks != 1:
        if checkpoint_options:
            msg = (
                "n_chunks (dense EM) cannot be combined with "
                f"{', '.join(checkpoint_options)} (structured EM)"
            )
            raise ValidationError(
                msg,
                got=f"n_chunks={n_chunks} and {', '.join(checkpoint_options)}",
                hint=(
                    "Drop n_chunks to run structured EM, or drop "
                    f"{', '.join(checkpoint_options)} to run dense EM in chunks."
                ),
            )
        transition_representation = "dense"
    _validate_em_options(
        transition_representation != "dense",
        n_chunks,
        chunk_size,
        result_path,
        store_log_likelihood,
        replay_cache_bytes,
    )
    return transition_representation


def _validate_estimation_arguments(
    time_edges: ArrayLike,
    n_chunks: int,
    return_outputs: ReturnOutputs,
    save_log_likelihood_to_results: bool | None,
    min_encoding_local_mass: float,
    min_encoding_local_ess: float,
) -> tuple[np.ndarray, float]:
    """Check ``estimate_parameters`` arguments that need no fitted state.

    The estimation wrappers call this before their initial ``fit`` so an invalid
    argument cannot leave a fitted detector partly rebuilt.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
    n_chunks : int
    return_outputs : str or list of str or set of str, optional
    save_log_likelihood_to_results : bool, optional
    min_encoding_local_mass : float
    min_encoding_local_ess : float

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)
        The validated edges (integers converted to float64).
    time_bin_width : float

    Raises
    ------
    ValidationError, DataError
        If ``time_edges`` are invalid or not uniform (see
        :func:`~non_local_detector.time_edges.uniform_time_edges`).
    ValueError
        If ``n_chunks`` exceeds the number of bins, ``return_outputs`` is
        invalid or conflicts with ``save_log_likelihood_to_results``, or an
        encoding-update threshold is negative.
    """
    time_edges, time_bin_width = uniform_time_edges(time_edges)
    n_bins = len(time_edges) - 1
    if n_chunks > n_bins:
        msg = (
            f"n_chunks ({n_chunks}) cannot exceed n_bins ({n_bins}). "
            f"Each chunk must contain at least one time bin."
        )
        raise ValueError(msg)
    if save_log_likelihood_to_results and return_outputs is not None:
        msg = (
            "Cannot specify both return_outputs and deprecated "
            "save_log_likelihood_to_results flag. "
            "Use return_outputs only."
        )
        raise ValueError(msg)
    _normalize_return_outputs(return_outputs)
    if min_encoding_local_mass < 0:
        msg = f"min_encoding_local_mass must be >= 0, got {min_encoding_local_mass}"
        raise ValueError(msg)
    if min_encoding_local_ess < 0:
        msg = f"min_encoding_local_ess must be >= 0, got {min_encoding_local_ess}"
        raise ValueError(msg)
    return time_edges, time_bin_width


@dataclass(frozen=True)
class _EMSettings:
    """Options of one ``_DetectorBase._run_em`` run, resolved before its loop.

    Fields not documented here are ``_run_em`` arguments, after validation and
    defaults.

    Attributes
    ----------
    time_bin_width : float
    time_centers : np.ndarray, shape (n_time,)
    n_observed_bins : int
        Decode bins not marked missing; the units of ``tolerance``.
    checkpointed : bool
        Whether EM runs on the structured transition operator.
    requested_outputs : set of str
        Canonical ``return_outputs``.
    em_attrs : dict
        EM path and chunking, recorded on the results.
    covariate_transitions : bool
        Whether the discrete transitions depend on covariates.
    transition_statistics : {None, "counts", "responses"}
        What structured E-steps accumulate for the discrete M-step.
    prior_tracks_counts : bool
        Whether the transition prior scales with each E-step's expected
        counts (``discrete_transition_prior_weight > 0``, stationary).
    prior_set_in_m_step : bool
        Whether REML sets the transition prior in each M-step.
    prior_changes : bool
        Either of the two above: the prior changes with each E-step.
    smoothness_setting : float or str
        ``discrete_transition_type.smoothness``.
    prior : tuple
        Concentration, stickiness, regularization and prior weight.
    frozen_rows, free_rows : np.ndarray, shape (n_states,), or None
        Discrete-transition rows pinned and estimated.
    interior_state_ind : np.ndarray, shape (n_interior_state_bins,), or None
        Dense EM with estimated transitions: state of each interior bin.
    interior_continuous_transition : np.ndarray or None
        Dense EM with estimated transitions: the continuous transition over
        the interior state bins.
    """

    time_edges: np.ndarray
    time_bin_width: float
    time_centers: np.ndarray
    log_likelihood_args: tuple[Any, ...]
    is_missing: np.ndarray | None
    n_observed_bins: int
    checkpointed: bool
    chunk_size: int | None
    replay_cache_bytes: int | None
    n_chunks: int
    cache_likelihood: bool
    store_log_likelihood: bool
    result_path: str | Path | None
    requested_outputs: set[str]
    em_attrs: dict[str, Any]
    estimate_initial_conditions: bool
    estimate_discrete_transition: bool
    estimate_encoding_model: bool
    max_iter: int
    tolerance: float
    min_encoding_local_mass: float
    min_encoding_local_ess: float
    covariate_transitions: bool
    transition_statistics: str | None
    prior_tracks_counts: bool
    prior_set_in_m_step: bool
    prior_changes: bool
    smoothness_setting: float | str
    prior: tuple[Any, Any, Any, Any]
    frozen_rows: np.ndarray | None
    free_rows: np.ndarray | None
    interior_state_ind: np.ndarray | None
    interior_continuous_transition: np.ndarray | None


@dataclass
class _EMState:
    """Values one EM iteration hands the next.

    EM monitors the marginal log-likelihood plus the log prior of the
    discrete transitions, the objective a MAP M-step never decreases; the
    likelihood alone can fall when the prior pulls the transitions. Each check
    compares two E-steps under the prior of the M-step between them
    (data-adaptive pseudo-counts change per E-step), so an iteration hands the
    next one its objective and the log prior of its M-step's output.

    Attributes
    ----------
    marginal_log_likelihoods : list of float
        Each in-loop E-step's marginal log-likelihood.
    objectives : list of float
        Each in-loop E-step's penalized log-likelihood.
    previous_objectives : list of float
        The objective each check compared ``objectives[k + 1]`` with.
    next_log_prior : float
        The log prior of the next E-step's parameters under the last M-step's
        prior; there is none before the first M-step.
    n_iter : int
    converged, objective_converged, is_increasing : bool
        The last check's outcome: converged (objective and, when the prior
        changes, transitions), objective within tolerance, and not decreasing.
    objective_change : float
        The last check's change in the objective.
    transition_change : float
        The last M-step's largest transition-probability change, tracked
        when the prior changes with each E-step.
    run_log_likelihoods : np.ndarray, shape (n_time, n_interior_state_bins), or None
        A likelihood computed in this run that the next E-step can reuse.
    transition_statistics : np.ndarray or None
        The last discrete M-step's expected transition statistics.
    """

    marginal_log_likelihoods: list[float] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    previous_objectives: list[float] = field(default_factory=list)
    next_log_prior: float = 0.0
    n_iter: int = 0
    converged: bool = False
    objective_converged: bool = False
    is_increasing: bool = True
    objective_change: float = np.inf
    transition_change: float = np.inf
    run_log_likelihoods: np.ndarray | None = None
    transition_statistics: np.ndarray | None = None

    def traces(self, final_log_likelihood: float) -> dict[str, list[float]]:
        """EM's traces, given the final E-step's marginal log-likelihood.

        The final E-step evaluates the returned parameters, the last M-step's
        output, under that M-step's prior (none with max_iter=0).
        """
        return {
            "marginal_log_likelihoods": [
                *self.marginal_log_likelihoods,
                final_log_likelihood,
            ],
            "em_objectives": [
                *self.objectives,
                final_log_likelihood + self.next_log_prior,
            ],
            "em_previous_objectives": list(self.previous_objectives),
        }


@dataclass(frozen=True)
class _EMEStep:
    """What an in-loop E-step hands the M-steps.

    Attributes
    ----------
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
    marginal_log_likelihood : float
    first_smoothed : np.ndarray, shape (n_interior_state_bins,), or None
        Smoothed posterior of the first bin (None when the checkpointed
        driver did not return it).
    transition_statistics : np.ndarray or None
        Expected discrete-transition statistics, when they are estimated.
    """

    acausal_state_probabilities: np.ndarray
    marginal_log_likelihood: float
    first_smoothed: np.ndarray | None
    transition_statistics: np.ndarray | None


@dataclass(frozen=True)
class _EMFinalEStep:
    """The final E-step's outputs.

    With ``result_path``, ``stored`` is the lazily loaded store and the arrays
    are None; otherwise outputs that were not computed are None.
    """

    marginal_log_likelihood: float
    stored: xr.Dataset | None = None
    acausal_posterior: Any = None
    acausal_state_probabilities: np.ndarray | None = None
    causal_state_probabilities: np.ndarray | None = None
    predictive_state_probabilities: np.ndarray | None = None
    log_likelihood: np.ndarray | None = None
    causal_posterior: Any = None
    predictive_posterior: Any = None


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


def _sample_mask(values: ArrayLike, n_time: int, name: str) -> np.ndarray:
    """A boolean mask with one entry per position sample, shape (n_time,).

    Integers equal to 0 or 1 are read as False or True. Any other integer, a
    float or another dtype is rejected rather than combined bitwise with the
    label masks, where ``2 & True`` is 0.
    """
    mask = np.asarray(values)
    if mask.shape != (n_time,):
        msg = f"{name} must have one entry per position sample"
        raise ValidationError(
            msg,
            expected=f"shape ({n_time},)",
            got=f"shape {mask.shape}",
            hint="A shorter or single-entry mask is not broadcast to every "
            f"sample; pass a 1-D array (e.g. {name}.ravel() for a column).",
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
            hint="Pass True/False per position sample, e.g. "
            f"{name}=speed > 4; other integers and floats are not read as a mask.",
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


def _decode_time_edges(
    time_edges: ArrayLike,
    transition_time_bin_width: float | None,
    transition_time_bin_width_tolerance: float = 0.0,
    *,
    _time_grid: _DecodeTimeGrid | None = None,
    row_slice: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a detector decode grid and derive its bin centers.

    Detector bins must be uniform because the HMM applies one transition per
    bin. When transitions were learned by ``estimate_parameters``, the grid
    must also use the bin width they were learned at.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
    transition_time_bin_width : float or None
        Bin width recorded with learned transitions, or None.
    transition_time_bin_width_tolerance : float, optional
        Uniformity tolerance of the grid the width was learned on. The widths
        match when they differ by no more than the larger of the two grids'
        tolerances.

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)
        The validated edges (integers converted to float64).
    time_centers : np.ndarray, shape (n_bins,)

    Raises
    ------
    ValidationError, DataError
        If the edges are invalid or not uniform, or their width differs from
        ``transition_time_bin_width`` by more than the timestamp precision.
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
    start, stop = resolve_row_slice(row_slice, len(time_edges) - 1)
    return time_edges, decode_bin_centers(time_edges, start, stop)


def _require_observed_bins(is_missing: np.ndarray | None) -> None:
    """Reject EM when every decode bin is missing: there is nothing to fit.

    Run anyway, EM moves the transitions on the prior dynamics alone (on a
    simulated recording every stay probability went to 1) and then reports
    non-convergence, pointing at ``max_iter`` instead of the mask.

    Parameters
    ----------
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        Missing bins from ``_missing_bins``.

    Raises
    ------
    DataError
        If every bin is missing.
    """
    if is_missing is not None and is_missing.size and bool(np.all(is_missing)):
        msg = (
            "Cannot estimate parameters: every decode time bin is missing, so "
            "EM has no observations to fit"
        )
        raise DataError(
            msg,
            data_name="is_missing",
            hint=(
                "Check that is_missing is True only for missing bins, and that "
                "position is tracked (finite) during time_edges, since bins "
                "without tracked position are missing too."
            ),
        )


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
    is_missing : np.ndarray, shape (n_bins,), optional
    position_time : np.ndarray, shape (n_time_position,), optional
    position : np.ndarray, shape (n_time_position, n_position_dims), optional
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
    UserWarning
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
        is_missing = np.asarray(is_missing, dtype=bool)
        if is_missing.shape != (n_bins,):
            msg = "is_missing must have one entry per decode bin"
            raise ValidationError(
                msg,
                expected=f"shape ({n_bins},)",
                got=f"shape {is_missing.shape}",
                hint="time_edges has n_bins + 1 entries; is_missing has n_bins.",
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
            UserWarning,
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
    position_time : np.ndarray, shape (n_time_position,)
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
    position_time : np.ndarray, shape (n_time_position,)
    group_weights : np.ndarray, shape (n_time_position,)
        Per-sample weights of this group (mask times any EM weight).

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
            UserWarning,
            stacklevel=user_stacklevel(),
        )
    return group_weights


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
    ValueError
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
            raise ValueError(msg)
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
            raise ValueError(msg)
        return arr.copy()
    if not np.issubdtype(arr.dtype, np.integer):
        msg = (
            f"frozen_discrete_transition_rows must be integer indices or "
            f"a boolean mask, got dtype {arr.dtype}"
        )
        raise ValueError(msg)
    if arr.ndim != 1:
        msg = (
            f"frozen_discrete_transition_rows index array must be 1D, "
            f"got shape {arr.shape}"
        )
        raise ValueError(msg)
    if np.any(arr < 0) or np.any(arr >= n_states):
        msg = (
            f"frozen_discrete_transition_rows indices must be in "
            f"[0, {n_states}), got {arr.tolist()}"
        )
        raise ValueError(msg)
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
    time : np.ndarray, shape (n_bins,)
        Decode bin centers.

    Raises
    ------
    ValueError
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
        raise ValueError(msg)


def _normalize_return_outputs(return_outputs: ReturnOutputs) -> set[str]:
    """Convert ``return_outputs`` to the set of optional outputs it names.

    A string names the same outputs as a one-element list: each name adds
    only its own output, and "all" adds every one.

    Parameters
    ----------
    return_outputs : str, list of str, set of str, or None
        Output names from ``VALID_OUTPUTS``.

    Returns
    -------
    set of str
        Normalized set containing any of: 'filter', 'predictive',
        'predictive_posterior', 'log_likelihood'

    Raises
    ------
    ValidationError
        If return_outputs contains invalid option names.
    TypeError
        If return_outputs is not str, list, set, or None.
    """
    if return_outputs is None:
        return set()
    if isinstance(return_outputs, str):
        names: set[str] = {return_outputs}
    elif isinstance(return_outputs, list | set):
        names = set(return_outputs)
    else:
        msg = (  # type: ignore[unreachable]  # runtime validation of an unsupported type
            f"return_outputs must be str, list of str, set of str, or None. "
            f"Got {type(return_outputs).__name__}"
        )
        raise TypeError(msg)
    invalid = names - VALID_OUTPUTS
    if invalid:
        msg = f"Invalid return_outputs: {sorted(invalid)}"
        raise ValidationError(
            msg,
            expected=f"names from {sorted(VALID_OUTPUTS)}",
            got=repr(return_outputs),
        )
    return set(_OPTIONAL_OUTPUTS) if "all" in names else names


class _DetectorBase(BaseEstimator, abc.ABC):  # type: ignore[misc]  # sklearn is untyped (BaseEstimator is Any)
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
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Dimensionless weight for data-adaptive prior scaling. When > 0,
            the Dirichlet prior pseudo-counts are scaled by expected transition
            counts, making regularization strength approximately invariant to
            temporal resolution. Can be a scalar (same weight for all rows) or
            an array of shape ``(n_states,)`` for per-row control. When 0.0
            (default), the fixed-count prior from concentration and stickiness
            is used directly.
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
        attributes are not touched. Changing ``environments``,
        ``state_names``, ``observation_models`` or any other parameter a fit
        depends on after fitting therefore requires refitting.

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
            If the resulting parameters are invalid, as in the constructor.
        ValueError
            If a parameter name is not a constructor parameter.
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
            raise ValueError(msg)
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

    def _validate_position_dimensionality(
        self, position: np.ndarray, context: str = "fit"
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
            for error messages.
        """
        is_1d = position.ndim == 1 or (position.ndim == 2 and position.shape[1] == 1)
        if not is_1d:
            return

        has_track_graph = any(env.track_graph is not None for env in self.environments)
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

    def _validate_decode_position(self, position: np.ndarray | None) -> None:
        """Check that decoding has the position its likelihood reads.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims), or None

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

        assert environment.is_track_interior_ is not None  # set by fit_place_grid
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

    def _validate_initial_conditions(
        self,
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
        """Validate discrete transition type configuration.

        Parameters
        ----------
        discrete_transition_type : DiscreteTransitions
            Discrete transition type object
        n_states : int
            Number of states in the model

        Raises
        ------
        ValidationError
            If transition matrix is invalid (wrong shape, doesn't sum to 1, etc.)
        DataError
            If transition matrix contains NaN or Inf
        """
        # If it's a custom transition matrix, validate it
        if isinstance(discrete_transition_type, DiscreteStationaryCustom):
            matrix = discrete_transition_type.values

            # Check it's a numpy array
            val.ensure_ndarray(matrix, "discrete_transition_type.values")

            # Check for NaN/Inf
            val.ensure_all_finite(matrix, "discrete_transition_type.values")

            # Check square
            val.ensure_square_matrix(matrix, "discrete_transition_type.values")

            # Check correct size
            if matrix.shape[0] != n_states:
                msg = "Discrete transition matrix size must match number of states"
                raise ValidationError(
                    msg,
                    expected=f"matrix with shape ({n_states}, {n_states})",
                    got=f"matrix with shape {matrix.shape}",
                    hint=f"Your model has {n_states} states, so transition matrix needs {n_states}x{n_states} shape",
                )

            # Check non-negative
            val.ensure_all_non_negative(matrix, "discrete_transition_type.values")

            # Check values in [0, 1]
            val.ensure_in_range(matrix, "discrete_transition_type.values", 0.0, 1.0)

            # Check row-stochastic (rows sum to 1)
            val.ensure_stochastic_matrix(matrix, "discrete_transition_type.values")

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
            Initialized environments as a tuple.
        """
        normalized: Any = environments
        if normalized is None:
            normalized = (Environment(),)
        if not hasattr(normalized, "__iter__"):
            normalized = (normalized,)
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
        never changed. ``self.environments`` holds the fitted copies afterwards.

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
        self.environments = tuple(fitted_environments)

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
                if environment.place_bin_centers_ is not None:
                    bin_sizes.append(environment.place_bin_centers_.shape[0])
                    state_ind.append(np.full(bin_sizes[-1], ind, dtype=int))
                else:
                    msg = "Environment place_bin_centers_ cannot be None"
                    raise ValueError(msg)
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
        for obs, cont_ic in zip(
            self.observation_models,
            self.continuous_initial_conditions_types,
            strict=False,
        ):
            # Multi-bin local uses full spatial initial conditions
            effective_obs = obs
            if obs.is_local and self.local_position_std is not None:
                effective_obs = replace(obs, is_local=False)
            ic_parts.append(
                cont_ic.make_initial_conditions(effective_obs, self.environments)
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
                self.environments,
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
                inds = np.ix_(
                    self.state_ind_ == from_state, self.state_ind_ == to_state
                )

                transition = _multi_bin_local_transition(
                    transition,
                    self.observation_models[from_state],
                    self.observation_models[to_state],
                    self.local_position_std,
                )

                n_row_bins = np.max(inds[0].shape)
                n_col_bins = np.max(inds[1].shape)

                if np.logical_and(n_row_bins == 1, n_col_bins > 1):
                    destination = _scalar_to_spatial_destination(
                        transition, self.environments
                    )
                    self.continuous_state_transitions_[inds] = (
                        destination / destination.sum()
                    )
                elif n_row_bins > 1 and n_col_bins == 1:
                    # Spatial to non-spatial: each source bin transitions
                    # to the single target bin with probability 1.
                    self.continuous_state_transitions_[inds] = np.ones((n_row_bins, 1))
                else:
                    self.continuous_state_transitions_[inds] = _make_transition_block(
                        transition, self.environments, context
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
        covariate_data = state.get("_discrete_transition_covariate_data_")
        if covariate_data is not None and (
            self.discrete_transition_design_matrix_ is None
        ):
            self.discrete_transition_design_matrix_ = (
                self.discrete_transition_type.make_state_transition(
                    covariate_data
                ).design_matrix
            )
        # Detectors fitted before transition smoothing existed: no penalty
        # strengths and no covariance, so EM and intervals see an unsmoothed
        # fit that has not estimated its uncertainty.
        if "discrete_transition_coefficient_covariance_" not in state:
            self.discrete_transition_coefficient_covariance_ = None
        if "discrete_transition_smoothness_matrix_" not in state:
            design = getattr(self, "discrete_transition_design_matrix_", None)
            if design is None:
                self.discrete_transition_smoothness_matrix_ = None
                self.discrete_transition_smoothness_ = None
            else:
                penalties = _smoothness_matrix(design.design_info)
                self.discrete_transition_smoothness_matrix_ = penalties
                # The design matrix is only set together with the coefficients.
                assert self.discrete_transition_coefficients_ is not None
                self.discrete_transition_smoothness_ = _row_smoothness(
                    0.0,
                    None,
                    penalties,
                    self.discrete_transition_coefficients_.shape[1],
                )[0]

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
        initial = self.discrete_transition_type.make_state_transition(covariate_data)
        self.discrete_state_transitions_ = initial.transition_matrix
        self.discrete_transition_coefficients_ = initial.coefficients
        self.discrete_transition_design_matrix_ = initial.design_matrix
        # patsy design matrices cannot be pickled or deep-copied; keep a copy
        # of the covariates so ``__setstate__`` can rebuild the design matrix
        # with the fitted knots even if the caller later changes them.
        self._discrete_transition_covariate_data_ = (
            None
            if self.discrete_transition_design_matrix_ is None
            else copy.deepcopy(covariate_data)
        )
        # Curvature penalty of each of the formula's spline terms, the
        # strengths of each row's penalties (fixed, or REML's starting value until an
        # M-step chooses it) and, once an M-step has fit them, the Laplace
        # covariance of each row's coefficients.
        self.discrete_transition_coefficient_covariance_ = None
        if self.discrete_transition_design_matrix_ is None:
            self.discrete_transition_smoothness_matrix_ = None
            self.discrete_transition_smoothness_ = None
        else:
            self.discrete_transition_smoothness_matrix_ = _smoothness_matrix(
                self.discrete_transition_design_matrix_.design_info
            )
            # The design matrix is only set together with the coefficients.
            assert self.discrete_transition_coefficients_ is not None
            self.discrete_transition_smoothness_ = _row_smoothness(
                getattr(self.discrete_transition_type, "smoothness", 0.0),
                None,
                self.discrete_transition_smoothness_matrix_,
                self.discrete_transition_coefficients_.shape[1],
            )[0]

        # Snapshot frozen rows of a stationary matrix so the EM M-step can
        # restore them after every update. Covariate-dependent frozen rows
        # instead keep their coefficients: the M-step skips them.
        if (
            self._frozen_discrete_transition_rows_mask_ is not None
            and self.discrete_state_transitions_.ndim == 2
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
        ValueError
            If the transitions do not depend on covariates, or were not
            estimated by EM (no covariance yet).
        """
        if self.discrete_transition_design_matrix_ is None:
            msg = (
                "The discrete transitions do not depend on covariates; use a "
                "DiscreteNonStationaryCustom or DiscreteNonStationaryDiagonal "
                "transition type."
            )
            raise ValueError(msg)
        if self.discrete_transition_coefficient_covariance_ is None:
            msg = (
                "No transition covariance yet: it comes from an EM M-step, so "
                "run estimate_parameters with estimate_discrete_transition=True "
                "and max_iter >= 1."
            )
            raise ValueError(msg)
        # The coefficients are set together with the design matrix.
        assert self.discrete_transition_coefficients_ is not None
        return _transition_intervals(
            self.discrete_transition_design_matrix_,
            self.discrete_transition_coefficients_,
            self.discrete_transition_coefficient_covariance_,
            covariate_data,
            coverage,
        )

    def plot_discrete_state_transition(
        self,
        cmap: str = "Oranges",
        ax: mpl.axes.Axes | None = None,
        convert_to_seconds: bool = False,
        sampling_frequency: int = 1,
        covariate_data: pd.DataFrame | dict[str, Any] | None = None,
    ) -> None:
        """
        Plot heatmap of discrete transition matrix.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap, by default "Oranges".
        ax : matplotlib.axes.Axes, optional
            Plotting axis, by default plots to current axis.
        convert_to_seconds : bool, optional
            Convert the probabilities of state to expected duration of state, by default False.
        sampling_frequency : int, optional
            Number of samples per second, by default 1.
        covariate_data : dict or pd.DataFrame, optional
            Dictionary or DataFrame of covariate data, by default None.
            Keys are covariate names and values are 1D arrays.
        """

        if self.discrete_state_transitions_.ndim == 2:
            if ax is None:
                ax = plt.gca()

            if convert_to_seconds:
                discrete_state_transition = (
                    1 / (1 - self.discrete_state_transitions_)
                ) / sampling_frequency
                vmin, vmax = 0.0, None
                label = "Seconds"
            else:
                discrete_state_transition = self.discrete_state_transitions_
                vmin, vmax = 0.0, 1.0
                label = "Probability"

            n_states = len(self.state_names)
            mesh = ax.pcolormesh(
                discrete_state_transition,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            # Row 0 at the top and tick labels centered on the cells.
            ax.set_xlim(0, n_states)
            ax.set_ylim(n_states, 0)
            ticks = np.arange(n_states) + 0.5
            ax.set_xticks(ticks, labels=self.state_names)
            ax.set_yticks(ticks, labels=self.state_names)
            cbar = ax.figure.colorbar(mesh, ax=ax, label=label)
            plt.setp(cbar.outline, visible=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Rotate tick labels that would overlap (as seaborn's heatmap did).
            ax.figure.canvas.draw()
            for axis in (ax.xaxis, ax.yaxis):
                boxes = [label.get_window_extent() for label in axis.get_ticklabels()]
                if any(a.overlaps(b) for a, b in pairwise(boxes)):
                    plt.setp(axis.get_ticklabels(), rotation="vertical")
            # Dark text on light cells, white text on dark cells.
            for (row, col), value, rgba in zip(
                np.ndindex(discrete_state_transition.shape),
                discrete_state_transition.ravel(),
                mesh.to_rgba(discrete_state_transition.ravel()),
                strict=True,
            ):
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    format(value, ".03f"),
                    ha="center",
                    va="center",
                    color=".15" if luminance > 0.408 else "w",
                )
            ax.set_ylabel("Previous State", fontsize=12)
            ax.set_xlabel("Current State", fontsize=12)
            ax.set_title("Discrete State Transition", fontsize=16)
        else:
            discrete_transition_design_matrix = self.discrete_transition_design_matrix_
            discrete_transition_coefficients = self.discrete_transition_coefficients_
            state_names = self.state_names

            if discrete_transition_design_matrix is None:
                msg = "discrete_transition_design_matrix_ is None for covariate-dependent prediction"
                raise ValueError(msg)

            predict_matrix = build_design_matrices(
                [discrete_transition_design_matrix.design_info],
                covariate_data,
            )[0]

            n_states = len(state_names)

            # The coefficients are set together with the design matrix.
            assert discrete_transition_coefficients is not None

            for covariate in covariate_data:  # type: ignore[union-attr]
                fig, axes = plt.subplots(
                    1, n_states, sharex=True, constrained_layout=True, figsize=(10, 5)
                )

                for from_state_ind, (ax, from_state) in enumerate(
                    zip(axes.flat, state_names, strict=False)
                ):
                    from_local_transition = centered_softmax_forward(
                        predict_matrix
                        @ discrete_transition_coefficients[:, from_state_ind]
                    )

                    ax.plot(covariate_data[covariate], from_local_transition)  # type: ignore[index]  # covariate_data is required on this path
                    ax.set_xlabel(covariate)
                    ax.set_ylabel("Prob.")
                    if from_state_ind == n_states - 1:
                        ax.legend(state_names)
                    ax.set_title(f"From {from_state}")
                fig.suptitle(f"Predicted transitions: {covariate}")

    def plot_continuous_state_transition(
        self, figsize_scaling: float = 1.5, vmax: float = 0.3
    ) -> None:
        """
        Plot heatmap of continuous state transition matrices.

        Parameters
        ----------
        figsize_scaling : float, optional
            Scaling factor for figure size, by default 1.5.
        vmax : float, optional
            Maximum value for color scale, by default 0.3.
        """
        GOLDEN_RATIO = 1.618

        if self.n_discrete_states_ == 1:
            # A single state has one panel, which needs room for its tick labels.
            _, ax = plt.subplots(
                constrained_layout=True,
                figsize=(4.0 * figsize_scaling, 3.5 * figsize_scaling),
            )
            ax.pcolormesh(self.continuous_state_transitions_, vmin=0.0, vmax=vmax)
            ax.set_xlabel("To position bin")
            ax.set_ylabel("From position bin")
            ax.set_title(self.state_names[0])
            return

        fig, axes = plt.subplots(
            self.n_discrete_states_,
            self.n_discrete_states_,
            gridspec_kw={
                "width_ratios": self.bin_sizes_,
                "height_ratios": self.bin_sizes_,
            },
            constrained_layout=True,
            figsize=(
                self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO,
                (self.n_discrete_states_ * figsize_scaling),
            ),
        )

        for from_state, ax_row in enumerate(axes):
            for to_state, ax in enumerate(ax_row):
                ind = np.ix_(self.state_ind_ == from_state, self.state_ind_ == to_state)
                ax.pcolormesh(
                    self.continuous_state_transitions_[ind], vmin=0.0, vmax=vmax
                )
                ax.set_xticks([])
                ax.set_yticks([])

                if to_state == 0:
                    ax.set_ylabel(
                        self.state_names[from_state],
                        rotation=0,
                        ha="right",
                        va="center",
                    )

                if from_state == self.n_discrete_states_ - 1:
                    ax.set_xlabel(
                        self.state_names[to_state],
                        rotation=45,
                        ha="right",
                        va="top",
                        labelpad=1.0,
                    )
        fig.supylabel("From State")
        fig.supxlabel("To State")

    def plot_initial_conditions(
        self, figsize_scaling: float = 1.5, vmax: float = 0.3
    ) -> None:
        """
        Plot heatmap of initial conditions.

        Parameters
        ----------
        figsize_scaling : float, optional
            Scaling factor for figure size, by default 1.5.
        vmax : float, optional
            Maximum value for color scale, by default 0.3.
        """
        GOLDEN_RATIO = 1.618
        fig, axes = plt.subplots(
            1,
            self.n_discrete_states_,
            gridspec_kw={"width_ratios": self.bin_sizes_},
            constrained_layout=True,
            figsize=(self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO, 1.1),
        )

        try:
            for state, ax in enumerate(axes):
                ind = self.state_ind_ == state
                ax.pcolormesh(
                    self.initial_conditions_[ind][:, np.newaxis], vmin=0.0, vmax=vmax
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(
                    self.state_names[state],
                    rotation=45,
                    ha="right",
                    va="top",
                    labelpad=0.0,
                )
        except TypeError:
            axes.pcolormesh(
                self.initial_conditions_[:, np.newaxis], vmin=0.0, vmax=vmax
            )
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel(
                self.state_names[0],
                rotation=45,
                ha="right",
                va="top",
                labelpad=0.0,
            )

        fig.suptitle("Initial Conditions")

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

        if position.ndim == 1 or (position.ndim == 2 and position.shape[1] == 1):
            has_track_graph = any(
                env.track_graph is not None for env in self.environments
            )
            if not has_track_graph:
                warnings.warn(
                    "Position data appears to be 1D but no track_graph was "
                    "provided. This will be treated as a 1D open field. If "
                    "you are working with a linear track, you likely want to "
                    "pass 2D position (x, y) along with a track_graph, "
                    "edge_order, and edge_spacing in your Environment so "
                    "that the track topology is properly linearized. See the "
                    "Environment docstring for details.",
                    UserWarning,
                    stacklevel=user_stacklevel(),
                )

        position = position[:, np.newaxis] if position.ndim == 1 else position
        self._invalidate_stored_log_likelihood()  # stale after a refit
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

    def _validate_time_contract(self) -> None:
        """Reject an unfitted model, unknown encoding units or transition clocks.

        Public decoding methods call this once, before any work. Subclasses
        extend it with their own fitted state (the clusterless mark
        reference); a detector without an encoding model must override it.
        """
        if not hasattr(self, "encoding_model_"):
            msg = "Detector is not fitted; call fit(...) before decoding."
            raise ValidationError(msg)
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

    def _validate_decode_population(
        self,
        spike_times: Any,
        spike_waveform_features: Any = None,
    ) -> None:
        """Validate active fitted populations before missing-data shortcuts."""
        is_clusterless = isinstance(self, ClusterlessDetector)
        if is_clusterless:
            validate_spike_feature_population(spike_times, spike_waveform_features)
        else:
            for neuron, times in enumerate(spike_times):
                if len(np.shape(times)) != 1:
                    msg = f"spike_times for neuron {neuron} must be 1-D"
                    raise ValidationError(
                        msg,
                        expected="one array with shape (n_spikes,) per neuron",
                        got=f"shape {np.shape(times)}",
                    )
        field = "mean_rates" if is_clusterless else "place_fields"
        for key in {
            (obs.environment_name, obs.encoding_group)
            for obs in self.observation_models
            if not obs.is_no_spike
        }:
            model = self.encoding_model_[key]
            validate_population_lengths(
                "electrode" if is_clusterless else "neuron",
                spike_times=spike_times,
                **{field: model.get(field)},
            )
            if is_clusterless:
                if "mark_dimensions" in model:
                    dimensions = model["mark_dimensions"]
                else:
                    encoded_marks = model.get(
                        "encoding_spike_waveform_features", model.get("encoding_marks")
                    )
                    dimensions = [np.shape(marks)[1] for marks in encoded_marks]
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

    _SPIKE_DATA_NAMES: tuple[str, ...] = ()
    """Names of the spike-data arrays, in the order the decode methods take them."""
    _PREDICT_EXAMPLE: str = ""
    """Example ``predict`` call shown in validation errors."""

    def _likelihood_predict_function(self) -> Callable[..., Any]:
        """The predict function of the detector's likelihood algorithm."""
        raise NotImplementedError

    def _likelihood_predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to every state's likelihood prediction."""
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
        _no_spike_time_bin_sizes: np.ndarray | None,
        _time_grid: _DecodeTimeGrid | None,
        _position_context: _LikelihoodPositionContext | None,
        _spike_time_order: _SpikeTimeOrder | None,
    ) -> jax.Array:
        """Body of the detectors' ``compute_log_likelihood``.

        ``spike_data`` holds the spike arrays named by ``_SPIKE_DATA_NAMES``,
        and the other arguments are those of ``compute_log_likelihood``. The
        decoding methods check the model and position once per call, then
        reach this through ``_prepare_likelihood_callback`` with a prepared
        ``_position_context``; a direct call has none and is checked here.
        """
        is_direct_call = _position_context is None
        if is_direct_call:
            self._validate_time_contract()
            decode = _DecodeInputs.from_arguments(position_time, position)
            position_time, position = decode.position_time, decode.position
        else:
            # A decoding method prepared the context from its coerced inputs.
            position_time = cast("np.ndarray | None", position_time)
            position = cast("np.ndarray | None", position)
        time_edges, row_time = _decode_time_edges(
            time_edges,
            *self._learned_transition_time_bin_width(),
            _time_grid=_time_grid,
            row_slice=row_slice,
        )
        _time_grid = _uniform_time_grid(time_edges, _time_grid)
        if is_direct_call:
            self._validate_decode_position(position)
        needs_position = self._decode_reads_position()
        if not isinstance(
            _position_context, _LikelihoodPositionContext
        ) or not _position_context.matches(
            time_edges, position_time, position, needs_position
        ):
            # Without a prepared context this is a direct call, so it warns
            # like the other public decode entry points.
            _position_context = _LikelihoodPositionContext.prepare(
                time_edges,
                position_time,
                position,
                reads_position=needs_position,
                warn_outside_tracking=is_direct_call,
            )
        self._validate_decode_population(*spike_data)
        position_missing = _position_context.is_missing
        logger.info("Computing log likelihood...")
        if _spike_time_order is None:
            _spike_time_order = _SpikeTimeOrder()
        non_local_penalty = self._non_local_position_penalty()

        row_start, row_stop = resolve_row_slice(row_slice, len(time_edges) - 1)
        n_rows = row_stop - row_start
        # Position interpolation, local kernels and the non-local penalty are
        # pointwise in the decoding time, so they are evaluated at the requested
        # bins' centers; the spike likelihoods bin against the full edges.
        if is_missing is None:
            is_missing = np.zeros(n_rows, dtype=bool)
        else:
            is_missing = np.asarray(is_missing, dtype=bool)
            if is_missing.shape != (n_rows,):
                msg = "is_missing must have one entry per requested decode bin"
                raise ValidationError(msg)
        if position_missing is not None:
            is_missing = is_missing | position_missing[row_start:row_stop]
        if _position_context.all_missing:
            return jnp.zeros((n_rows, int(self.is_track_interior_state_bins_.sum())))
        position = _position_context.position

        likelihood_func = self._likelihood_predict_function()

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
                    # The fitted setting, as the other likelihoods receive it.
                    # No-Spike may have no encoding entry; default to showing the bar.
                    disable_progress_bar=self.encoding_model_.get(
                        likelihood_name[:2], {}
                    ).get("disable_progress_bar", False),
                    _time_bin_sizes=_no_spike_time_bin_sizes,
                    _spike_time_order=_spike_time_order,
                    _time_grid=_time_grid,
                )
                mark_log_likelihood = self._no_spike_mark_log_likelihood(
                    spike_data, time_edges, row_slice, _spike_time_order
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
                    **self.encoding_model_[likelihood_name[:2]],
                    **self._likelihood_predict_kwargs(),
                    is_local=effective_is_local,
                    row_slice=row_slice,
                    _spike_time_order=_spike_time_order,
                    _time_grid=_time_grid,
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
            time_edges, *self._learned_transition_time_bin_width()
        )
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
                DeprecationWarning,
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
                raise ValueError(msg)
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
            cache_likelihood,
            n_chunks,
            chunk_size,
            result_path,
            checkpoint_dir,
            selected_intervals,
            replay_cache_bytes,
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
                raise ValueError(msg)
            predicted_transitions = predict_discrete_state_transitions(
                self.discrete_transition_design_matrix_,
                self.discrete_transition_coefficients_,
                discrete_transition_covariate_data,
            )
            _validate_covariate_time_length(predicted_transitions, time_centers)
        if inference_mode == "checkpointed":
            return self._predict_checkpointed(
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

        (
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            causal_state_probabilities,
            predictive_state_probabilities,
            log_likelihood,
            causal_posterior,
            predictive_posterior,
            _degenerate_indices,
        ) = self._predict(
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
            np.asarray(acausal_posterior),
            acausal_state_probabilities,
            marginal_log_likelihood,
            time_edges=time_edges,
            is_missing=is_missing,
            log_likelihood=(
                log_likelihood if "log_likelihood" in requested_outputs else None
            ),
            causal_posterior=(
                np.asarray(causal_posterior) if "filter" in requested_outputs else None
            ),
            causal_state_probabilities=(
                causal_state_probabilities if "filter" in requested_outputs else None
            ),
            predictive_state_probabilities=(
                predictive_state_probabilities
                if "predictive" in requested_outputs
                else None
            ),
            predictive_posterior=(
                np.asarray(predictive_posterior)
                if "predictive_posterior" in requested_outputs
                else None
            ),
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
            store_log_likelihood,
            replay_cache_bytes,
        )
        time_edges, _ = _validate_estimation_arguments(
            time_edges,
            n_chunks,
            return_outputs,
            save_log_likelihood_to_results,
            min_encoding_local_mass,
            min_encoding_local_ess,
        )
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        position_time, position = inputs.position_time, inputs.position
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
            self._encoding_model_data = {
                "position_time": position_time,
                "position": position,
                **dict(zip(self._SPIKE_DATA_NAMES, spike_data, strict=True)),
                "is_training": inputs.is_training,
                "encoding_group_labels": inputs.encoding_group_labels,
                "environment_labels": inputs.environment_labels,
                "encoding_time_range": encoding_time_range,
                "valid_position_intervals": valid_position_intervals,
            }
            try:
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
                )
            finally:
                # Drop the references to the caller's arrays.
                del self._encoding_model_data

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
            time_edges, *self._learned_transition_time_bin_width()
        )
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
        cache_likelihood: bool = True,
        n_chunks: int = 1,
        discrete_state_transitions: np.ndarray | None = None,
        accumulate_log_likelihoods: bool = False,
    ) -> tuple[
        jax.Array | np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        jax.Array | np.ndarray,
        jax.Array | np.ndarray,
        np.ndarray,
    ]:
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
        acausal_posterior : jax.Array, shape (n_time, n_state_bins)
        acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        marginal_log_likelihood : float
        causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        causal_posterior : jax.Array, shape (n_time, n_state_bins)
        predictive_posterior : jax.Array, shape (n_time, n_state_bins)
        degenerate_indices : np.ndarray of int, shape (n_degenerate,)

        The three posteriors stay on the device: callers copy to the host only
        the ones they return or need there.
        """
        # The public methods have checked already; EM reaches here after its
        # own fit. Kept as the engine's guard for direct calls.
        self._validate_time_contract()
        time_edges, time_centers = _decode_time_edges(
            time_edges, *self._learned_transition_time_bin_width()
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
            time_grid = _uniform_time_grid(time_edges)
            log_likelihood_func = _prepare_likelihood_callback(
                log_likelihood_func,
                time_edges,
                has_no_spike=any(obs.is_no_spike for obs in self.observation_models),
                log_likelihood_args=log_likelihood_args,
                reads_position=self._decode_reads_position(),
                _time_grid=time_grid,
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
        # Degenerate (all-impossible) time indices are found during the
        # forward pass, so they are reliable regardless of n_chunks/caching,
        # unlike the returned log-likelihoods (which are None when uncached,
        # e.g. for n_chunks > 1). Stored transiently for estimate_parameters
        # to copy into the public degenerate_timesteps_ attribute.
        self._degenerate_timesteps_ = np.asarray(result.degenerate_indices, dtype=int)
        return result

    def _checkpointed_likelihood(
        self,
        edges: np.ndarray,
        centers: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
    ) -> Callable[..., Any]:
        """Row-slice likelihood callback for the checkpointed driver."""
        time_grid = _uniform_time_grid(edges)
        prepared = _prepare_likelihood_callback(
            self.compute_log_likelihood,
            edges,
            has_no_spike=any(obs.is_no_spike for obs in self.observation_models),
            log_likelihood_args=log_likelihood_args,
            reads_position=self._decode_reads_position(),
            _time_grid=time_grid,
        )

        def likelihood(
            full_edges: np.ndarray,  # noqa: ARG001 -- likelihood callback signature
            *,
            row_slice: slice | None,
            is_missing: np.ndarray | None,
        ) -> Any:
            return prepared(
                centers,
                *log_likelihood_args,
                row_slice=row_slice,
                is_missing=is_missing,
            )

        return likelihood

    def _checkpointed_transition_args(
        self, discrete: np.ndarray, n_rows: int
    ) -> dict[str, Any]:
        """Checkpointed-driver transition arguments over the interior state bins.

        Structured fits pass the fused operator; dense fits pass the dense
        interior matrices. Covariate transitions pass their per-row discrete
        matrices separately and must have one row per decode bin.
        """
        if discrete.ndim == 3 and discrete.shape[0] != n_rows:
            msg = "Predict covariate transitions must have one row per decode bin"
            raise ValidationError(msg)
        interior = self.is_track_interior_state_bins_
        states = self.state_ind_[interior]
        transition_args: dict[str, Any] = {}
        operator = getattr(self, "_continuous_transition_operator_", None)
        if operator is not None:
            operator = operator.restricted(interior)
            if discrete.ndim == 2:
                operator = operator.bind_discrete(discrete)
            else:
                transition_args["discrete_transition_matrix"] = discrete
            transition_args["transition_operator"] = operator.fused(
                self._max_dense_transition_bytes_
            )
        else:
            continuous = self.continuous_state_transitions_[np.ix_(interior, interior)]
            if discrete.ndim == 2:
                transition_args["transition_matrix"] = (
                    continuous * discrete[np.ix_(states, states)]
                )
            else:
                transition_args.update(
                    continuous_transition_matrix=continuous,
                    discrete_transition_matrix=discrete,
                )
        return transition_args

    def _checkpointed_e_step(
        self,
        time_edges: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
        is_missing: np.ndarray | None,
        chunk_size: int,
        *,
        outputs: Iterable[str] = ("acausal_state_probabilities",),
        transition_statistics: str | None = None,
        replay_cache_bytes: int,
    ) -> CheckpointedResult:
        """One whole-recording E-step on the checkpointed driver.

        Parameters
        ----------
        time_edges : np.ndarray, shape (n_time + 1,)
        log_likelihood_args : tuple
        is_missing : np.ndarray, shape (n_time,), or None
        chunk_size : int
            Rows per checkpoint chunk.
        outputs : iterable of str
            Driver outputs. Spatial outputs are held in memory.
        transition_statistics : {None, "counts", "responses"}
        replay_cache_bytes : int
            Host bytes for forward-pass rows reused by the replay.

        Returns
        -------
        result : CheckpointedResult
        """
        from non_local_detector.checkpointed_inference import (
            _SPATIAL,
            checkpointed_forward_backward,
        )

        edges, centers = _decode_time_edges(
            time_edges, *self._learned_transition_time_bin_width()
        )
        interior = self.is_track_interior_state_bins_
        output_names = set(outputs)
        return checkpointed_forward_backward(
            edges,
            self.initial_conditions_[interior],
            self._checkpointed_likelihood(edges, centers, log_likelihood_args),
            state_ind=self.state_ind_[interior],
            n_states=len(self.state_names),
            is_missing=is_missing,
            chunk_size=chunk_size,
            output_mode="spatial" if output_names & _SPATIAL else "compact",
            return_outputs=output_names,
            dtype=np.float32,
            transition_statistics=transition_statistics,
            replay_cache_bytes=replay_cache_bytes,
            **self._checkpointed_transition_args(
                self.discrete_state_transitions_, len(centers)
            ),
        )

    def _predict_checkpointed(
        self,
        *,
        time_edges: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
        is_missing: np.ndarray | None,
        discrete_transitions: np.ndarray | None,
        output_mode: OutputMode,
        chunk_size: int | None,
        result_path: str | Path | None,
        checkpoint_dir: str | Path | None,
        selected_intervals: np.ndarray | None,
        requested_outputs: set[str],
        max_read_bytes: int,
        replay_cache_bytes: int | None,
        em_attrs: Callable[[float], dict[str, Any]] | None = None,
    ) -> xr.Dataset:
        """Decode with complete conditioning and bounded spatial working memory.

        ``replay_cache_bytes`` of None uses ``default_replay_cache_bytes()``.
        ``em_attrs`` maps this pass's marginal log-likelihood to EM's
        attributes, which a result store keeps with the rest so that
        ``load_results`` returns them.
        """
        from non_local_detector.checkpointed_inference import (
            checkpointed_forward_backward,
        )

        if replay_cache_bytes is None:
            replay_cache_bytes = default_replay_cache_bytes()

        if output_mode == "spatial" and result_path is None:
            msg = "Checkpointed spatial prediction requires result_path"
            raise ValidationError(
                msg,
                hint="Pass a new directory as result_path, or use output_mode='compact'.",
            )
        if output_mode == "compact" and requested_outputs & {
            "log_likelihood",
            "predictive_posterior",
        }:
            msg = "Spatial outputs (including return_outputs='all') require output_mode='spatial'"
            raise ValidationError(
                msg,
                got=f"return_outputs requesting {sorted(requested_outputs)}",
                hint="Compact output holds state probabilities only; request "
                "'filter' and 'predictive', or use output_mode='spatial' with a result_path.",
            )
        edges, centers = _decode_time_edges(
            time_edges, *self._learned_transition_time_bin_width()
        )
        interior = self.is_track_interior_state_bins_
        states = self.state_ind_[interior]
        if chunk_size is None:
            chunk_size = default_chunk_size(
                cast("int", np.count_nonzero(interior)), np.float32
            )
        discrete = (
            self.discrete_state_transitions_
            if discrete_transitions is None
            else discrete_transitions
        )
        transition_args = self._checkpointed_transition_args(discrete, len(centers))
        likelihood = self._checkpointed_likelihood(edges, centers, log_likelihood_args)
        selected: np.ndarray | None = None
        if selected_intervals is not None:
            intervals = np.asarray(selected_intervals, dtype=float)
            if (
                intervals.ndim != 2
                or intervals.shape[1] != 2
                or not np.isfinite(intervals).all()
                or np.any(intervals[:, 0] >= intervals[:, 1])
                or np.any(intervals[1:, 0] < intervals[:-1, 1])
            ):
                msg = "selected_intervals must be ordered non-overlapping start/stop pairs"
                raise ValidationError(msg)
            mask = np.zeros(len(centers), dtype=bool)
            # The tolerances are under 1% of a bin width, so bins more than one
            # bin outside an interval can never be selected; apply the rule only
            # from one bin before each start to one bin after each stop.
            firsts = np.maximum(np.searchsorted(edges[:-1], intervals[:, 0]) - 1, 0)
            lasts = np.minimum(
                np.searchsorted(edges[1:], intervals[:, 1], side="right") + 1,
                len(centers),
            )
            for (start, stop), first, last in zip(
                intervals, firsts, lasts, strict=True
            ):
                lower, upper = edges[first:last], edges[first + 1 : last + 1]
                width = upper - lower
                lower_tol = np.minimum(
                    4 * np.maximum(np.spacing(abs(start)), np.spacing(abs(lower))),
                    width * 0.01,
                )
                upper_tol = np.minimum(
                    4 * np.maximum(np.spacing(abs(stop)), np.spacing(abs(upper))),
                    width * 0.01,
                )
                mask[first:last] |= (lower >= start - lower_tol) & (
                    upper <= stop + upper_tol
                )
            selected = np.flatnonzero(mask)
            if selected.size == 0:
                msg = "selected_intervals select no complete decode bins"
                raise ValidationError(
                    msg,
                    got=f"{len(intervals)} interval(s)",
                    hint=(
                        "Intervals are in seconds and select bins whose start and "
                        f"stop edges both lie inside one; the recording spans "
                        f"[{edges[0]}, {edges[-1]}] in bins of {edges[1] - edges[0]}."
                    ),
                )
        outputs = {"acausal_state_probabilities"}
        if output_mode == "spatial":
            outputs.add("acausal_posterior")
        if "filter" in requested_outputs:
            outputs.add("causal_state_probabilities")
            if output_mode == "spatial":
                outputs.add("causal_posterior")
        if "predictive" in requested_outputs:
            outputs.add("predictive_state_probabilities")
        if "predictive_posterior" in requested_outputs:
            outputs.add("predictive_posterior")
        if "log_likelihood" in requested_outputs:
            outputs.add("log_likelihood")
        position = np.concatenate(state_bin_positions(self))
        metadata: dict[str, Any] = {
            "state_names": list(self.state_names),
            "environment_names": [
                obs.environment_name for obs in self.observation_models
            ],
            "encoding_groups": [
                obs.encoding_group.item()
                if isinstance(obs.encoding_group, np.generic)
                else obs.encoding_group
                for obs in self.observation_models
            ],
            "position": position.tolist(),
            "position_names": get_position_dim_names(position.shape[1]),
            "state_ind": self.state_ind_.tolist(),
            "interior_mask": interior.tolist(),
            "max_read_bytes": int(max_read_bytes),
        }
        attrs: dict[str, Any] = {
            "non_local_detector_version": _PACKAGE_VERSION,
            "time_units": "seconds",
            "time_bin_convention": "left_closed_right_open_final_closed",
            "conditioning": "whole_recording",
            "_native_metadata": metadata,
            "recording_rows": len(centers),
            "transition_representation": getattr(
                self, "_transition_representation_", "dense"
            ),
        }
        result = checkpointed_forward_backward(
            edges,
            self.initial_conditions_[interior],
            likelihood,
            state_ind=states,
            n_states=len(self.state_names),
            is_missing=is_missing,
            chunk_size=chunk_size,
            output_mode=output_mode,
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            return_outputs=outputs,
            selected_rows=selected,
            dtype=np.float32,
            max_read_bytes=max_read_bytes,
            result_attrs=(
                attrs
                if em_attrs is None
                else lambda marginal: {**attrs, **em_attrs(marginal)}
            ),
            replay_cache_bytes=replay_cache_bytes,
            **transition_args,
        )
        dataset = result.dataset
        self._degenerate_timesteps_ = np.asarray(
            result.diagnostics.get("degenerate_indices", []), dtype=int
        )
        return label_checkpointed_result(dataset, metadata)

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

    def _refit_encoding_model(self, weights: np.ndarray) -> None:
        """EM M-step: refit the encoding model with posterior ``weights``.

        Parameters
        ----------
        weights : np.ndarray, shape (n_time_position,)
        """
        self.fit_encoding_model(**self._encoding_model_data, weights=weights)

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

    def _fit_em_model(
        self,
        fit: Callable[[str], Any],
        transition_representation: TransitionRepresentation,
        n_time: int,
        structured_required: bool,
    ) -> None:
        """Run EM's internal fit and, for "auto", choose dense or structured EM.

        "auto" cannot build some transitions in bounded form that the dense
        constructor accepts (opaque custom constructors with deferred graph
        distances, or dense fallback blocks over ``max_dense_transition_bytes``).
        Unless a structured-EM option requires the operator, those run dense EM;
        otherwise ``_choose_em_representation`` picks the path.

        Parameters
        ----------
        fit : callable
            ``fit(transition_representation)`` runs the detector's ``fit``.
        transition_representation : str
            Resolved by ``_resolve_em_options``.
        n_time : int
            Decode bins.
        structured_required : bool
            Whether ``chunk_size``, ``result_path`` or ``replay_cache_bytes``
            was given.
        """
        from non_local_detector.transition_operators import (
            DenseTransitionBudgetError,
            UnsupportedTransitionError,
        )

        try:
            fit(transition_representation)
        except (UnsupportedTransitionError, DenseTransitionBudgetError) as error:
            if transition_representation != "auto" or structured_required:
                raise
            logger.info("EM with transition_representation='auto': dense (%s)", error)
            fit("dense")
            return
        if transition_representation == "auto" and not structured_required:
            self._choose_em_representation(n_time)

    def _choose_em_representation(self, n_time: int) -> None:
        """Run EM dense when it is faster and fits; otherwise keep the operator.

        Called after an ``"auto"`` fit. Dense EM is chosen when the float32
        interior transition is at most ``_AUTO_DENSE_TRANSITION_BYTES``, the
        full dense transition it stores (every state bin, float64, as the
        dense constructor builds it) fits ``max_dense_transition_bytes``, and
        dense EM's arrays fit in half the device memory, or half the available
        host memory on CPU (see ``_dense_em_memory``; unchecked when unknown).
        The dense matrix comes from the fitted operator, which matches the
        dense constructor. When memory alone rules dense EM out, it warns:
        structured EM recomputes the likelihood on every E-step, which made EM
        about 9x slower per E-step on a 567k-bin clusterless track decode.

        Parameters
        ----------
        n_time : int
            Decode bins.
        """
        n_bins = int(np.count_nonzero(self.is_track_interior_state_bins_))
        working, limit, where = _dense_em_memory(n_time, n_bins)
        n_total = int(self.state_ind_.shape[0])
        dense_preferred = (
            n_bins * n_bins * 4 <= _AUTO_DENSE_TRANSITION_BYTES
            and n_total * n_total * 8 <= self._max_dense_transition_bytes_
        )
        dense_fits = limit is None or working <= limit // 2
        use_dense = dense_preferred and dense_fits
        if dense_preferred and not dense_fits:
            assert limit is not None  # dense_fits is True whenever limit is None
            warnings.warn(
                'EM with transition_representation="auto" runs structured EM '
                f"because dense EM needs about {working / 1e9:.1f} GB of {where} "
                f"memory for {n_time} time bins x {n_bins} state bins, more than "
                f"half of the {limit / 1e9:.1f} GB available. Structured EM "
                "bounds memory by recomputing the likelihood on every E-step, "
                "which can make EM several times slower. The estimate is "
                "approximate: if dense EM fits, pass "
                'transition_representation="dense" to estimate_parameters.',
                UserWarning,
                stacklevel=user_stacklevel(),
            )
        logger.info(
            "EM with transition_representation='auto': %s (%d interior state "
            "bins, %d time bins)",
            "dense" if use_dense else "structured",
            n_bins,
            n_time,
        )
        if use_dense:
            self.continuous_state_transitions_ = np.asarray(
                self.continuous_state_transitions_
            )
            self.__dict__.pop("_continuous_transition_operator_", None)
            self._transition_representation_ = "dense"

    def _run_em(
        self,
        time_edges: ArrayLike,
        log_likelihood_args: tuple[Any, ...] | None = None,
        is_missing: np.ndarray | None = None,
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
        chunk_size: int | None = None,
        result_path: str | Path | None = None,
        replay_cache_bytes: int | None = None,
    ) -> xr.Dataset:
        """EM engine: estimate the parameters from a likelihood's arguments.

        The detectors' ``estimate_parameters`` validate and fit the data, then
        call this with the ``compute_log_likelihood`` arguments. It estimates
        the initial conditions, the discrete transitions and, optionally, the
        encoding model with the Expectation-Maximization algorithm.

        Parameters
        ----------
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the bin
            centers, and the bin width is recorded with the learned transitions
            as ``transition_time_bin_width_``.
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default None.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Re-fit the place fields in each M-step, weighted by the summed
            posterior of observations with ``is_local=True``, independently
            of their state names, by default False. Without local observations
            this update is skipped. Every state shares the encoding model, so
            weighting it by local observations alone is not
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
        return_outputs : str, list of str, set of str, or None, optional
            Controls which optional outputs are returned.

            Options:
            - 'filter' : causal (filtered) posterior and state probabilities
            - 'predictive' : predictive state probabilities
            - 'predictive_posterior' : predictive posterior over state bins
            - 'log_likelihood' : log likelihoods
            - 'all' : all of the above

            A name adds the same outputs alone or in a list or set.

            By default None (only smoother posterior and state probabilities).
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead.
            Whether to save the log likelihood to the results, by default None.
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights required to update the encoding
            model. If the total mass is below this threshold, the encoding
            M-step is skipped for that iteration. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size (ESS) of local state weights required
            to update the encoding model. ESS = sum(w)^2 / sum(w^2). If ESS
            is below this threshold, the encoding M-step is skipped. By default 1.0.
        chunk_size : int, optional
            Rows per checkpoint chunk when the model was fit with structured
            transitions (``transition_representation="structured"`` or
            ``"auto"``). Such models run every E-step on the checkpointed
            driver: one forward pass that keeps a checkpoint per chunk and a
            backward replay, so working memory scales with ``chunk_size``
            times the state bins instead of the recording length, and the
            M-steps use the expected transition statistics accumulated in
            the replay. The final E-step returns the same outputs as dense
            EM, held in memory unless ``result_path`` is given. ``n_chunks``
            must then be 1 and ``cache_likelihood`` has no effect: each
            uncached chunk evaluates the likelihood twice (see
            ``replay_cache_bytes``). By default None, which sizes chunks so one
            chunk of float32 likelihoods takes about 256 MiB.
        result_path : str or Path, optional
            Structured-transition EM only: write the final E-step's outputs
            (the requested ``return_outputs`` plus the acausal posterior) to
            an atomic directory store here, as checkpointed ``predict`` does,
            and return it lazily, so host memory does not hold the full
            posterior. ``marginal_log_likelihoods`` in the returned attributes
            lists every iteration; the store itself records the final one.
            ``store_log_likelihood`` cannot be combined with it.
        replay_cache_bytes : int, optional
            Structured-transition EM only: host memory for keeping each
            E-step's forward-pass rows (filtered rows, or likelihoods when the
            final outputs need them), so the backward replay does not filter
            and evaluate them again. Chunks beyond the budget are recomputed.
            By default None, which uses 16 GiB or half the available host
            memory (counting a container's limit), whichever is smaller.

        Returns
        -------
        results : xr.Dataset
            Results of the decoding, including posterior probabilities.
            ``attrs["marginal_log_likelihoods"]`` and ``attrs["em_objectives"]``
            give each E-step's marginal log-likelihood and penalized
            log-likelihood (plus the discrete-transition log prior), the
            quantity EM's convergence and monotonicity checks use. Each check
            compares two E-steps under the prior of the M-step between them:
            ``check_converged(em_objectives[k + 1], em_previous_objectives[k],
            tolerance, n_time_bins=n_time_bins)`` reproduces it, with
            ``attrs["em_previous_objectives"]`` the earlier E-step's objective
            under that prior and ``n_time_bins`` the number of decode bins
            not marked missing. With
            ``discrete_transition_prior_weight > 0`` or REML smoothness the
            prior changes with each E-step, so ``em_previous_objectives[k]``
            can differ from ``em_objectives[k]``. With ``max_iter=0`` there is no M-step
            prior; the sole objective is the final marginal log-likelihood
            and ``em_previous_objectives`` is empty. A ``result_path`` store
            keeps these attributes, so ``load_results`` returns them too.
        """
        settings = self._em_settings(
            time_edges,
            log_likelihood_args,
            is_missing,
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
        )
        self._em_reset_fitted_attributes()
        state = _EMState()

        while not state.converged and (state.n_iter < settings.max_iter):
            e_step = self._em_e_step(settings, state)

            # Maximization step
            logger.info("Maximization step...")
            if settings.estimate_encoding_model:
                self._em_update_encoding_model(settings, state, e_step)
            if settings.estimate_discrete_transition:
                log_prior_before, log_prior_after = (
                    self._em_update_discrete_transitions(settings, state, e_step)
                )
            else:
                log_prior_before = log_prior_after = 0.0
            if settings.estimate_initial_conditions:
                self._em_update_initial_conditions(e_step)

            self._em_record_iteration(
                settings,
                state,
                e_step.marginal_log_likelihood,
                log_prior_before,
                log_prior_after,
            )

        if (
            settings.estimate_discrete_transition
            and settings.covariate_transitions
            and state.n_iter > 0
        ):
            self._em_update_transition_covariance(settings, state)

        # converged_ is set after the final E-step, which evaluates the
        # returned parameters and can still find a decrease.
        self.n_iter_ = int(state.n_iter)

        if not state.converged:
            self._em_warn_not_converged(settings, state)

        final = self._em_final_e_step(settings, state)
        traces = state.traces(final.marginal_log_likelihood)

        # The preceding final E-step's _predict recorded the degenerate
        # (all-impossible) timesteps during its forward pass; expose them as a
        # fitted attribute. Sourced from the forward-pass collector rather than
        # the returned log-likelihoods, so it is reliable even when caching is
        # disabled (n_chunks > 1), where the returned log-likelihoods are None.
        self.degenerate_timesteps_ = self._degenerate_timesteps_

        self._em_check_final_e_step(settings, state, traces["em_objectives"][-1])

        if settings.store_log_likelihood:
            self.log_likelihood_ = final.log_likelihood

        if settings.result_path is not None:
            # The store already holds the EM attributes and traces.
            assert final.stored is not None
            return final.stored
        return self._em_results(settings, final, traces)

    def _em_settings(
        self,
        time_edges: ArrayLike,
        log_likelihood_args: tuple[Any, ...] | None,
        is_missing: np.ndarray | None,
        *,
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
        chunk_size: int | None,
        result_path: str | Path | None,
        replay_cache_bytes: int | None,
    ) -> _EMSettings:
        """Validate ``_run_em``'s arguments and resolve its options.

        Sets no attribute, so a validation failure leaves the detector as it
        was. Parameters are those of ``_run_em``.

        Returns
        -------
        settings : _EMSettings
        """
        time_edges, time_bin_width = _validate_estimation_arguments(
            time_edges,
            n_chunks,
            return_outputs,
            save_log_likelihood_to_results,
            min_encoding_local_mass,
            min_encoding_local_ess,
        )
        checkpointed = (
            getattr(self, "_continuous_transition_operator_", None) is not None
        )
        _validate_em_options(
            checkpointed,
            n_chunks,
            chunk_size,
            result_path,
            store_log_likelihood,
            replay_cache_bytes,
        )
        # Resolved once and recorded on the results, so users can tell which
        # EM path ran and with what chunks.
        em_attrs: dict[str, Any]
        if checkpointed:
            if chunk_size is None:
                chunk_size = default_chunk_size(
                    cast("int", np.count_nonzero(self.is_track_interior_state_bins_)),
                    np.float32,
                )
            if replay_cache_bytes is None:
                replay_cache_bytes = default_replay_cache_bytes()
            em_attrs = {
                "em_transition_representation": "structured",
                "em_chunk_size": int(chunk_size),
                "em_replay_cache_bytes": int(replay_cache_bytes),
            }
        else:
            em_attrs = {
                "em_transition_representation": "dense",
                "em_n_chunks": n_chunks,
            }
            if n_chunks == 1:
                _warn_if_dense_em_exceeds_memory(
                    len(time_edges) - 1,
                    int(np.count_nonzero(self.is_track_interior_state_bins_)),
                )
        # Subclasses have already combined ``is_missing`` with the bins
        # without tracked position via ``_missing_bins``.
        time_centers = decode_bin_centers(time_edges, 0, len(time_edges) - 1)
        # Units of ``tolerance``. Subclasses refuse to estimate with every bin
        # missing (``_require_observed_bins``), so at least one is observed.
        n_missing = 0 if is_missing is None else int(np.count_nonzero(is_missing))
        n_observed_bins = len(time_centers) - n_missing

        if log_likelihood_args is None:
            log_likelihood_args = ()

        # Handle deprecated boolean flag
        if save_log_likelihood_to_results is not None:
            warnings.warn(
                "save_log_likelihood_to_results is deprecated. "
                "Use return_outputs='log_likelihood' instead.",
                DeprecationWarning,
                stacklevel=user_stacklevel(),
            )
            if save_log_likelihood_to_results:
                return_outputs = "log_likelihood"

        # Normalize return_outputs to canonical set
        requested_outputs = _normalize_return_outputs(return_outputs)

        # A requested log likelihood has to be materialized. Unchunked that
        # means caching the full-time evaluation; with n_chunks > 1 caching is
        # disabled by _predict and the final E-step accumulates the per-chunk
        # rows instead (see _em_final_e_step).
        if (
            "log_likelihood" in requested_outputs
            and not cache_likelihood
            and n_chunks == 1
        ):
            cache_likelihood = True

        covariate_transitions = _is_nonstationary(
            self.discrete_transition_coefficients_,
            self.discrete_transition_design_matrix_,
        )
        # Structured EM accumulates the discrete M-step's statistics in its
        # replay: per-row responses for covariate transitions, else counts.
        transition_statistics = None
        if checkpointed and estimate_discrete_transition:
            transition_statistics = "responses" if covariate_transitions else "counts"
        # Data-adaptive pseudo-counts (prior_weight > 0 on a stationary
        # transition) scale with each E-step's expected counts. Any other prior
        # is fixed, so an M-step's input has the log prior the previous M-step
        # computed for its output.
        prior_tracks_counts = (
            estimate_discrete_transition
            and not covariate_transitions
            and bool(np.any(np.asarray(self.discrete_transition_prior_weight) > 0))
        )
        # REML chooses each covariate row's smoothness in its M-step, so that
        # M-step's prior is known only after it, and EM checks the M-step's
        # input against its output under it.
        smoothness_setting = getattr(self.discrete_transition_type, "smoothness", 0.0)
        prior_set_in_m_step = (
            estimate_discrete_transition
            and covariate_transitions
            and isinstance(smoothness_setting, str)
        )
        prior_changes = prior_tracks_counts or prior_set_in_m_step
        interior_state_ind: np.ndarray | None = None
        interior_continuous_transition: np.ndarray | None = None
        if estimate_discrete_transition and not checkpointed:
            interior_state_bins = self.is_track_interior_state_bins_
            interior_state_ind = self.state_ind_[interior_state_bins]
            interior_continuous_transition = self.continuous_state_transitions_[
                np.ix_(interior_state_bins, interior_state_bins)
            ]
        prior = (
            self.discrete_transition_concentration,
            self.discrete_transition_stickiness,
            self.discrete_transition_regularization,
            self.discrete_transition_prior_weight,
        )
        # Frozen rows are not estimated (covariate-dependent rows are skipped,
        # stationary rows restored after each M-step), so they do not enter the
        # objective it maximizes.
        frozen_rows = self._frozen_discrete_transition_rows_mask_
        free_rows = None if frozen_rows is None else ~frozen_rows
        return _EMSettings(
            time_edges=time_edges,
            time_bin_width=time_bin_width,
            time_centers=time_centers,
            log_likelihood_args=log_likelihood_args,
            is_missing=is_missing,
            n_observed_bins=n_observed_bins,
            checkpointed=checkpointed,
            chunk_size=chunk_size,
            replay_cache_bytes=replay_cache_bytes,
            n_chunks=n_chunks,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            result_path=result_path,
            requested_outputs=requested_outputs,
            em_attrs=em_attrs,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            covariate_transitions=covariate_transitions,
            transition_statistics=transition_statistics,
            prior_tracks_counts=prior_tracks_counts,
            prior_set_in_m_step=prior_set_in_m_step,
            prior_changes=prior_changes,
            smoothness_setting=smoothness_setting,
            prior=prior,
            frozen_rows=frozen_rows,
            free_rows=free_rows,
            interior_state_ind=interior_state_ind,
            interior_continuous_transition=interior_continuous_transition,
        )

    def _em_reset_fitted_attributes(self) -> None:
        """Start EM's result attributes and drop an earlier run's likelihood."""
        # Initialize the EM-result attributes together, after validation, so a
        # validation failure leaves NONE of them set (all-or-nothing).
        # converged_/n_iter_ are updated to their final values after the loop;
        # degenerate_timesteps_ is filled from the final E-step's likelihoods.
        self.converged_ = False
        self.n_iter_ = 0
        self.em_monotonicity_violations_ = []
        self.degenerate_timesteps_ = np.array([], dtype=int)
        # Any likelihood stored by an EARLIER run describes that run's data and
        # encoding model, not this one's. Drop it now, and reuse within this run
        # only a likelihood computed inside this run
        # (``_EMState.run_log_likelihoods``).
        self._invalidate_stored_log_likelihood()
        # Fitted initial-condition distribution starts at the user-supplied
        # constructor argument; overwritten with the fitted distribution in
        # each M-step when ``estimate_initial_conditions`` is True.
        self.discrete_initial_conditions_ = np.asarray(self.discrete_initial_conditions)

    def _em_transition_log_prior(
        self,
        settings: _EMSettings,
        statistics: np.ndarray,
        coefficients: np.ndarray | None = None,
    ) -> float:
        """Log prior of the current transitions (or covariate
        ``coefficients``) under one M-step's prior."""
        return _transition_log_prior(
            statistics,
            self.discrete_state_transitions_,
            (
                self.discrete_transition_coefficients_
                if coefficients is None
                else coefficients
            ),
            self.discrete_transition_design_matrix_,
            *settings.prior,
            free_rows=settings.free_rows,
            smoothness=(
                0.0
                if self.discrete_transition_smoothness_ is None
                else self.discrete_transition_smoothness_
            ),
            smoothness_matrix=self.discrete_transition_smoothness_matrix_,
        )

    def _em_e_step(self, settings: _EMSettings, state: _EMState) -> _EMEStep:
        """One in-loop E-step, dense or checkpointed.

        Updates ``state.run_log_likelihoods`` with a likelihood the next
        E-step can reuse.
        """
        logger.info("Expectation step...")
        if settings.checkpointed:
            e_step = self._checkpointed_e_step(
                settings.time_edges,
                settings.log_likelihood_args,
                settings.is_missing,
                cast("int", settings.chunk_size),  # resolved when checkpointed
                transition_statistics=settings.transition_statistics,
                replay_cache_bytes=cast("int", settings.replay_cache_bytes),
            )
            acausal_state_probabilities = e_step.dataset[
                "acausal_state_probabilities"
            ].values
            marginal_log_likelihood = e_step.marginal_log_likelihood
            first_smoothed = e_step.first_smoothed
            log_likelihood = None
            transition_statistics = e_step.transition_statistics
        else:
            # Device arrays.
            acausal_posterior: Any
            causal_posterior: Any
            predictive_posterior: Any
            (
                acausal_posterior,
                acausal_state_probabilities,
                marginal_log_likelihood,
                _causal_state_probabilities,
                _predictive_state_probabilities,
                log_likelihood,
                causal_posterior,
                predictive_posterior,
                _degenerate_indices,
            ) = self._predict(
                time_edges=settings.time_edges,
                log_likelihood_args=settings.log_likelihood_args,
                is_missing=settings.is_missing,
                cache_likelihood=settings.cache_likelihood,
                log_likelihoods=state.run_log_likelihoods,
                n_chunks=settings.n_chunks,
            )
            first_smoothed = np.asarray(acausal_posterior[0])
            # The transition statistics need only these posteriors, so
            # take them now; the device arrays are released on return, before
            # the M-step (an encoding refit allocates on the device too).
            transition_statistics = (
                _discrete_transition_statistics(
                    causal_posterior,
                    predictive_posterior,
                    acausal_posterior,
                    cast("np.ndarray", settings.interior_continuous_transition),
                    cast("np.ndarray", settings.interior_state_ind),
                    self.discrete_state_transitions_,
                    self.discrete_transition_coefficients_,
                    self.discrete_transition_design_matrix_,
                )
                if settings.estimate_discrete_transition
                else None
            )
        # Reusable by the next iteration: the likelihood depends only on the
        # data and the encoding model, and the encoding M-step clears this when
        # it updates the encoding model. ``None`` when the likelihood was not
        # materialized (n_chunks > 1 disables caching), i.e. recompute.
        if settings.cache_likelihood:
            state.run_log_likelihoods = log_likelihood
        return _EMEStep(
            acausal_state_probabilities=acausal_state_probabilities,
            marginal_log_likelihood=marginal_log_likelihood,
            first_smoothed=first_smoothed,
            transition_statistics=transition_statistics,
        )

    def _em_update_encoding_model(
        self, settings: _EMSettings, state: _EMState, e_step: _EMEStep
    ) -> None:
        """Encoding M-step: refit with the local observations' posterior.

        Skipped without local observations, or when their posterior mass or
        effective sample size is below the thresholds.
        """
        local_state_indices = [
            index
            for index, observation in enumerate(self.observation_models)
            if observation.is_local
        ]

        if local_state_indices:
            logger.info("Estimating encoding model...")
            local_state_weights = e_step.acausal_state_probabilities[
                :, local_state_indices
            ].sum(axis=1)
            # Weights live on the decode bins; the encoding model is
            # fit on the position samples. Interpolate from the bin
            # centers even when the counts match: equal lengths do not
            # mean equal coordinates. Samples outside the decoded edges
            # have no posterior and get no weight.
            position_time = self._encoding_model_data.get("position_time")
            if position_time is not None:
                position_time = np.asarray(position_time)
                local_state_weights = np.where(
                    (position_time >= settings.time_edges[0])
                    & (position_time <= settings.time_edges[-1]),
                    np.interp(
                        position_time, settings.time_centers, local_state_weights
                    ),
                    0.0,
                )

            # ESS / mass guard: skip encoding update when local
            # state has effectively disappeared
            mass = float(np.sum(local_state_weights))
            sum_sq = float(np.sum(local_state_weights**2))
            ess = (mass**2 / sum_sq) if sum_sq > 0 else 0.0

            if (
                mass < settings.min_encoding_local_mass
                or ess < settings.min_encoding_local_ess
            ):
                logger.info(
                    "Skipping encoding update: local mass=%.4g, "
                    "ESS=%.4g (thresholds: mass>=%.4g, ESS>=%.4g)",
                    mass,
                    ess,
                    settings.min_encoding_local_mass,
                    settings.min_encoding_local_ess,
                )
            else:
                # Re-fit the encoding model using the posterior weights.
                # Known limitation (issue #31): the encoding model (the
                # emission) is shared across states but is re-fit here
                # using only the Local-state marginal as weights, so this
                # M-step does not strictly maximize the full-model
                # expected complete-data log-likelihood and can break the
                # EM monotonicity guarantee. The monotonicity-violation /
                # final-E-step warnings now surface this when it occurs.
                self._refit_encoding_model(local_state_weights)

                # The encoding model changed, so the likelihood
                # computed from the previous one is stale.
                # (``fit_encoding_model`` also drops any stored
                # ``log_likelihood_``.)
                state.run_log_likelihoods = None

    def _em_update_discrete_transitions(
        self, settings: _EMSettings, state: _EMState, e_step: _EMEStep
    ) -> tuple[float, float]:
        """Discrete-transition M-step.

        Records the step's statistics in ``state.transition_statistics`` and,
        when the prior changes with each E-step, how far the transitions moved
        in ``state.transition_change``.

        Returns
        -------
        log_prior_before : float
            Log prior of the M-step's input under its prior.
        log_prior_after : float
            Log prior of the M-step's output under its prior.
        """
        statistics = e_step.transition_statistics
        # Requested from the E-step whenever the transition is estimated.
        assert statistics is not None
        state.transition_statistics = statistics
        log_prior_before = (
            self._em_transition_log_prior(settings, statistics)
            if state.n_iter == 0 or settings.prior_tracks_counts
            else state.next_log_prior
        )
        if settings.prior_changes:
            coefficients_before = self.discrete_transition_coefficients_
            # Covariate transitions are compared from their
            # coefficients, without copying (n_time, n_states, n_states).
            if not settings.covariate_transitions:
                transitions_before = np.array(self.discrete_state_transitions_)
        m_step = _discrete_transition_from_statistics(
            statistics,
            self.discrete_transition_coefficients_,
            self.discrete_transition_design_matrix_,
            *settings.prior,
            frozen_rows=settings.frozen_rows,
            smoothness=settings.smoothness_setting,
            smoothness_matrix=self.discrete_transition_smoothness_matrix_,
            initial_smoothness=self.discrete_transition_smoothness_,
        )
        self.discrete_state_transitions_ = m_step.transition
        self.discrete_transition_coefficients_ = m_step.coefficients
        if settings.covariate_transitions:
            self.discrete_transition_smoothness_ = m_step.smoothness
        if settings.prior_set_in_m_step:
            log_prior_before = self._em_transition_log_prior(
                settings, statistics, coefficients_before
            )
        # The learned transitions are per bin at this width.
        self.transition_time_bin_width_ = settings.time_bin_width
        self._transition_time_bin_width_tolerance = _uniformity_tolerance(
            settings.time_edges, settings.time_bin_width
        )
        # Restore frozen rows: overwrite the M-step result with
        # the initial snapshot for rows the user wants pinned.
        # This is a constrained M-step; EM converges to the
        # constrained fixed point rather than the unconstrained
        # maximizer, which is the desired behavior here.
        if self._frozen_discrete_transition_rows_baseline_ is not None:
            self.discrete_state_transitions_[
                self._frozen_discrete_transition_rows_mask_
            ] = self._frozen_discrete_transition_rows_baseline_
        log_prior_after = self._em_transition_log_prior(settings, statistics)
        if settings.prior_changes and settings.covariate_transitions:
            # Covariate transitions have a design matrix and coefficients.
            assert self.discrete_transition_design_matrix_ is not None
            assert coefficients_before is not None
            assert self.discrete_transition_coefficients_ is not None
            state.transition_change = _max_transition_change(
                self.discrete_transition_design_matrix_,
                coefficients_before,
                self.discrete_transition_coefficients_,
            )
        elif settings.prior_changes:
            moved = self.discrete_state_transitions_ - transitions_before
            state.transition_change = float(np.max(np.abs(moved)))
        return log_prior_before, log_prior_after

    def _em_update_initial_conditions(self, e_step: _EMEStep) -> None:
        """Initial-condition M-step: the first bin's smoothed posterior."""
        self.initial_conditions_[self.is_track_interior_state_bins_] = (
            e_step.first_smoothed
        )
        self.discrete_initial_conditions_ = np.asarray(
            e_step.acausal_state_probabilities[0]
        )

        expanded_discrete_ic = e_step.acausal_state_probabilities[0][self.state_ind_]
        is_zero = np.isclose(expanded_discrete_ic, 0.0)
        safe_discrete_ic = np.where(is_zero, 1.0, expanded_discrete_ic)
        self.continuous_initial_conditions_ = np.where(
            is_zero,
            0.0,
            self.initial_conditions_ / safe_discrete_ic,
        )

    def _em_record_iteration(
        self,
        settings: _EMSettings,
        state: _EMState,
        marginal_log_likelihood: float,
        log_prior_before: float,
        log_prior_after: float,
    ) -> None:
        """Record an iteration's objective and check convergence.

        Warns and records the iteration in ``em_monotonicity_violations_``
        when the penalized log-likelihood decreased.

        Parameters
        ----------
        settings : _EMSettings
        state : _EMState
        marginal_log_likelihood : float
            This iteration's E-step marginal log-likelihood.
        log_prior_before, log_prior_after : float
            Log prior of this iteration's M-step input and output under its
            prior (0.0 when the transitions are not estimated).
        """
        logger.info("Computing stats..")
        state.n_iter += 1

        state.marginal_log_likelihoods.append(marginal_log_likelihood)
        # This E-step evaluated the last M-step's output, under that
        # M-step's prior; the first E-step has only its own M-step's.
        state.objectives.append(
            marginal_log_likelihood
            + (state.next_log_prior if state.n_iter > 1 else log_prior_before)
        )
        if state.n_iter > 1:
            state.objective_change = (
                state.objectives[-1] - state.previous_objectives[-1]
            )
            state.objective_converged, state.is_increasing = check_converged(
                state.objectives[-1],
                state.previous_objectives[-1],
                settings.tolerance,
                n_time_bins=settings.n_observed_bins,
            )
            # A data-adaptive prior (or REML smoothness) changes with
            # every E-step, so a flat objective alone does not mean the
            # transitions have settled.
            state.converged = state.objective_converged and (
                not settings.prior_changes
                or state.transition_change < _TRANSITION_CHANGE_TOLERANCE
            )

            if not state.is_increasing:
                # Surfaced via warnings.warn(UserWarning) to match the
                # other EM/GMM/GLM fit-quality diagnostics (decode-time
                # degenerate-timestep warnings use logger.warning instead;
                # see core._warn_degenerate_and_nan_timesteps).
                warnings.warn(
                    f"EM iteration {state.n_iter}: penalized log-likelihood "
                    f"decreased from {state.previous_objectives[-1]:.6f} to "
                    f"{state.objectives[-1]:.6f} "
                    f"(change={state.objective_change:.6e}; "
                    f"marginal log-likelihood + transition log prior). "
                    f"A MAP M-step cannot lower it; re-fitting the encoding "
                    f"model inside EM, a covariate transition fit stopping "
                    f"early, or a numerical instability can.",
                    UserWarning,
                    stacklevel=user_stacklevel(),
                )
                self.em_monotonicity_violations_.append(state.n_iter)

            logger.info(
                "iteration %s, likelihood: %s, penalized: %s, change: %s%s",
                state.n_iter,
                state.marginal_log_likelihoods[-1],
                state.objectives[-1],
                state.objective_change,
                (
                    f", transition change: {state.transition_change}"
                    if settings.prior_changes
                    else ""
                ),
            )
        else:
            logger.info(
                "iteration %s, likelihood: %s",
                state.n_iter,
                state.marginal_log_likelihoods[-1],
            )
        # The next E-step evaluates this M-step's output; it is checked
        # against this E-step, both under this M-step's prior.
        state.previous_objectives.append(marginal_log_likelihood + log_prior_before)
        state.next_log_prior = log_prior_after

    def _em_update_transition_covariance(
        self, settings: _EMSettings, state: _EMState
    ) -> None:
        """Covariance of the covariate transition coefficients after EM."""
        # The last M-step's statistics and output (covariate transitions
        # have a design matrix, coefficients and smoothness).
        assert self.discrete_transition_coefficients_ is not None
        assert self.discrete_transition_design_matrix_ is not None
        assert self.discrete_transition_smoothness_ is not None
        self.discrete_transition_coefficient_covariance_ = (
            _covariate_transition_covariance(
                self.discrete_transition_coefficients_,
                self.discrete_transition_design_matrix_,
                np.asarray(state.transition_statistics),
                self.discrete_transition_concentration,
                self.discrete_transition_stickiness,
                self.discrete_transition_regularization,
                self.discrete_transition_smoothness_,
                self.discrete_transition_smoothness_matrix_,
                settings.frozen_rows,
            )
        )

    def _em_warn_not_converged(self, settings: _EMSettings, state: _EMState) -> None:
        """Warn that EM stopped at ``max_iter``, with advice for the cause."""
        if not state.is_increasing:
            advice = (
                "The penalized log-likelihood decreased in the last "
                "iteration, and EM never counts a decrease as convergence, "
                "so a looser tolerance will not help; see the monotonicity "
                "warnings above for likely causes."
            )
        elif state.n_iter > 1 and state.objectives[-1] == -np.inf:
            advice = (
                "The penalized log-likelihood is -inf, so its change cannot "
                "be judged; this usually means degenerate timesteps (every "
                "state impossible), listed in degenerate_timesteps_ after "
                "the fit."
            )
        elif state.objective_converged:
            advice = (
                "The penalized log-likelihood changed by less than "
                "tolerance, but with discrete_transition_prior_weight > 0 "
                "or REML smoothness the last M-step must also move every "
                "transition probability by less than "
                f"{_TRANSITION_CHANGE_TOLERANCE:g}, and it "
                f"moved one by {state.transition_change:.3e}; consider "
                "increasing max_iter."
            )
        else:
            advice = "Consider increasing max_iter or relaxing tolerance."
        warnings.warn(
            f"EM did not converge after max_iter={settings.max_iter} iterations "
            f"(final penalized log-likelihood change per time bin="
            f"{state.objective_change / settings.n_observed_bins:.3e}, "
            f"tolerance={settings.tolerance:.3e}). Fitted parameters may be biased. "
            f"{advice}",
            UserWarning,
            stacklevel=user_stacklevel(),
        )

    def _em_final_e_step(self, settings: _EMSettings, state: _EMState) -> _EMFinalEStep:
        """E-step on the fitted parameters, with the requested outputs.

        Run after the last M-step so the returned posterior reflects the final
        fitted parameters. Without it, the returned arrays are the E-step
        computed *before* the last M-step and disagree with predict() on the
        same data.
        """
        logger.info("Final E-step after EM...")
        requested_outputs = settings.requested_outputs
        if settings.result_path is not None:
            em_attrs = settings.em_attrs
            stored = self._predict_checkpointed(
                time_edges=settings.time_edges,
                log_likelihood_args=settings.log_likelihood_args,
                is_missing=settings.is_missing,
                discrete_transitions=None,
                output_mode="spatial",
                chunk_size=settings.chunk_size,
                result_path=settings.result_path,
                checkpoint_dir=None,
                selected_intervals=None,
                requested_outputs=requested_outputs,
                max_read_bytes=DEFAULT_MAX_READ_BYTES,
                replay_cache_bytes=settings.replay_cache_bytes,
                # The store is complete before its marginal log-likelihood
                # reaches this method, so it writes the traces itself.
                em_attrs=lambda marginal: {**em_attrs, **state.traces(marginal)},
            )
            return _EMFinalEStep(
                marginal_log_likelihood=float(stored.attrs["marginal_log_likelihood"]),
                stored=stored,
            )
        if settings.checkpointed:
            outputs = {"acausal_posterior", "acausal_state_probabilities"}
            if "filter" in requested_outputs:
                outputs |= {"causal_posterior", "causal_state_probabilities"}
            if "predictive" in requested_outputs:
                outputs.add("predictive_state_probabilities")
            if "predictive_posterior" in requested_outputs:
                outputs.add("predictive_posterior")
            if settings.store_log_likelihood or "log_likelihood" in requested_outputs:
                outputs.add("log_likelihood")
            final = self._checkpointed_e_step(
                settings.time_edges,
                settings.log_likelihood_args,
                settings.is_missing,
                cast("int", settings.chunk_size),  # resolved when checkpointed
                outputs=outputs,
                replay_cache_bytes=cast("int", settings.replay_cache_bytes),
            )
            values = {name: final.dataset[name].values for name in final.dataset}
            self._degenerate_timesteps_ = final.diagnostics["degenerate_indices"]
            return _EMFinalEStep(
                marginal_log_likelihood=final.marginal_log_likelihood,
                acausal_posterior=values["acausal_posterior"],
                acausal_state_probabilities=values["acausal_state_probabilities"],
                causal_state_probabilities=values.get("causal_state_probabilities"),
                predictive_state_probabilities=values.get(
                    "predictive_state_probabilities"
                ),
                log_likelihood=values.get("log_likelihood"),
                causal_posterior=values.get("causal_posterior"),
                predictive_posterior=values.get("predictive_posterior"),
            )
        (
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            causal_state_probabilities,
            predictive_state_probabilities,
            log_likelihood,
            causal_posterior,
            predictive_posterior,
            _degenerate_indices,
        ) = self._predict(
            time_edges=settings.time_edges,
            log_likelihood_args=settings.log_likelihood_args,
            is_missing=settings.is_missing,
            cache_likelihood=settings.cache_likelihood,
            log_likelihoods=state.run_log_likelihoods,
            n_chunks=settings.n_chunks,
            # Final E-step only: this is the likelihood the caller can ask to
            # keep. With n_chunks > 1 caching is unavailable, so the chunked
            # path accumulates the per-chunk rows (the documented T x N
            # exception) instead of returning None. The in-loop E-steps never
            # need the array, so they keep the per-chunk allocation.
            accumulate_log_likelihoods=(
                settings.store_log_likelihood or "log_likelihood" in requested_outputs
            ),
        )
        return _EMFinalEStep(
            marginal_log_likelihood=marginal_log_likelihood,
            acausal_posterior=acausal_posterior,
            acausal_state_probabilities=acausal_state_probabilities,
            causal_state_probabilities=causal_state_probabilities,
            predictive_state_probabilities=predictive_state_probabilities,
            log_likelihood=log_likelihood,
            causal_posterior=causal_posterior,
            predictive_posterior=predictive_posterior,
        )

    def _em_check_final_e_step(
        self, settings: _EMSettings, state: _EMState, final_objective: float
    ) -> None:
        """Check the final E-step like an in-loop iteration and set ``converged_``.

        There is no check when max_iter=0. The final E-step counts as
        iteration n_iter + 1, and a decrease here means the fit has not
        converged.
        """
        final_is_increasing = True
        if state.n_iter > 0:
            _, final_is_increasing = check_converged(
                final_objective,
                state.previous_objectives[-1],
                settings.tolerance,
                n_time_bins=settings.n_observed_bins,
            )
            if not final_is_increasing:
                self.em_monotonicity_violations_.append(state.n_iter + 1)
                decrease = state.previous_objectives[-1] - final_objective
                warnings.warn(
                    f"Final E-step penalized log-likelihood (marginal log-likelihood "
                    f"+ transition log prior) decreased by {decrease:.3e}. This "
                    f"indicates an inconsistency between the E-step and the M-step "
                    f"output and is unusual; the returned posterior may not reflect "
                    f"the fitted parameters. converged_ is False and iteration "
                    f"{state.n_iter + 1} is recorded in em_monotonicity_violations_.",
                    UserWarning,
                    stacklevel=user_stacklevel(),
                )
        self.converged_ = bool(state.converged) and final_is_increasing

    def _em_results(
        self,
        settings: _EMSettings,
        final: _EMFinalEStep,
        traces: dict[str, list[float]],
    ) -> xr.Dataset:
        """The in-memory results of EM: the final E-step and EM's traces."""
        requested_outputs = settings.requested_outputs
        # Every final E-step without a result_path returns them.
        assert final.acausal_state_probabilities is not None
        em_attrs = dict(settings.em_attrs)
        em_attrs["em_objectives"] = np.asarray(traces["em_objectives"])
        em_attrs["em_previous_objectives"] = np.asarray(
            traces["em_previous_objectives"]
        )
        results = convert_results_to_xarray(
            self,
            settings.time_centers,
            np.asarray(final.acausal_posterior),
            final.acausal_state_probabilities,
            traces["marginal_log_likelihoods"],
            time_edges=settings.time_edges,
            is_missing=settings.is_missing,
            log_likelihood=(
                final.log_likelihood if "log_likelihood" in requested_outputs else None
            ),
            causal_posterior=(
                np.asarray(final.causal_posterior)
                if "filter" in requested_outputs
                else None
            ),
            causal_state_probabilities=(
                final.causal_state_probabilities
                if "filter" in requested_outputs
                else None
            ),
            predictive_state_probabilities=(
                final.predictive_state_probabilities
                if "predictive" in requested_outputs
                else None
            ),
            predictive_posterior=(
                np.asarray(final.predictive_posterior)
                if "predictive_posterior" in requested_outputs
                else None
            ),
        )
        results.attrs.update(em_attrs)
        return results

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
            time_edges, *self._learned_transition_time_bin_width()
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

        log_likelihood_func = _prepare_likelihood_callback(
            self.compute_log_likelihood,
            time_edges,
            has_no_spike=any(obs.is_no_spike for obs in self.observation_models),
            log_likelihood_args=log_likelihood_args,
            reads_position=self._decode_reads_position(),
            _time_grid=_uniform_time_grid(time_edges),
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
        self, time_range: np.ndarray, trim: bool = False
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

    def save_model(self, filename: str = "model.pkl") -> None:
        """
        Save the detector to a pickled file.

        Parameters
        ----------
        filename : str, optional
            Filename to save the model, by default "model.pkl".
        """
        with Path(filename).open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str = "model.pkl") -> "_DetectorBase":
        """
        Load the detector from a file.

        .. warning::

            Only load models from trusted sources. Pickle files can execute
            arbitrary code during deserialization.

        Parameters
        ----------
        filename : str, optional
            Filename to load the model from, by default "model.pkl".

        Returns
        -------
        _DetectorBase
            Loaded detector instance
        """
        with Path(filename).open("rb") as f:
            return cast("_DetectorBase", pickle.load(f))

    @staticmethod
    def save_results(results: xr.Dataset, filename: str = "results.nc") -> None:
        """
        Save the results to a netCDF file.

        ``state_bins`` is a MultiIndex, which netCDF does not support, so it
        is converted before saving. Boolean attributes are stored as integers
        and the internal ``_native_metadata`` attribute is dropped. The dtype
        of each integer variable and coordinate is recorded so that
        ``load_results`` restores it: netCDF3, written when neither netCDF4
        nor h5netcdf is installed, stores int64 as int32.

        Parameters
        ----------
        results : xr.Dataset
            Decoding results.
        filename : str, optional
            Name to save, by default "results.nc".
        """
        export = (
            results.reset_index("state_bins")
            if "state_bins" in results.indexes
            else results.copy(deep=False)
        )
        export.attrs = {
            name: int(value) if isinstance(value, (bool, np.bool_)) else value
            for name, value in export.attrs.items()
            if name != "_native_metadata"
        }
        export.attrs[_INTEGER_DTYPES_ATTR] = json.dumps(
            {
                str(name): variable.dtype.name
                for name, variable in export.variables.items()
                if variable.dtype.kind in "iu"
            }
        )
        export.to_netcdf(filename)

    @staticmethod
    def load_results(
        filename: str | Path = "results.nc",
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> xr.Dataset:
        """
        Load results saved by ``save_results`` or a checkpointed result_path.

        A netCDF file is read lazily and gets its ``state_bins`` MultiIndex
        back and its integer variables and coordinates the dtypes they were
        saved with; closing the returned dataset closes the file. A directory
        is opened as a checkpointed result store with lazy arrays.

        Parameters
        ----------
        filename : str or Path, optional
            A netCDF file or checkpointed result directory, by default
            "results.nc".
        max_read_bytes : int, optional
            Maximum bytes in one explicit read of a result directory's lazy
            arrays, by default 512 MiB. Ignored for netCDF files.

        Returns
        -------
        xr.Dataset
            Decoding results
        """
        if Path(filename).is_dir():
            from non_local_detector.result_store import open_result_store

            results = open_result_store(filename, max_read_bytes=max_read_bytes)
            metadata = results.attrs.get("_native_metadata")
            if metadata:
                metadata = {**metadata, "max_read_bytes": max_read_bytes}
                return label_checkpointed_result(results, metadata)
            return results
        opened = xr.open_dataset(filename)
        results = opened
        if _INTEGER_DTYPES_ATTR in results.attrs:
            integer_dtypes = json.loads(results.attrs.pop(_INTEGER_DTYPES_ATTR))
        else:
            # Files saved before the dtypes were recorded: restore state_ind.
            integer_dtypes = {"state_ind": np.dtype(int).name}
        # Only a still-integer variable is restored: one edited to another
        # dtype after saving (say, float with NaN) keeps its values.
        restored = {
            name: results.variables[name].astype(dtype)
            for name, dtype in integer_dtypes.items()
            if name in results.variables
            and results.variables[name].dtype.kind in "iu"
            and results.variables[name].dtype != dtype
        }
        results = results.assign_coords(
            {name: value for name, value in restored.items() if name in results.coords}
        ).assign(
            {
                name: value
                for name, value in restored.items()
                if name in results.data_vars
            }
        )
        if "state_bins" in results.dims:
            if "state_ind" in results.dims:
                # Earlier versions saved state_ind on a dimension of its own.
                results = results.drop_vars("state_ind").assign_coords(
                    state_ind=("state_bins", results["state_ind"].values)
                )
            # Scalar state/environment coordinates are also attached to this
            # array, but only the original one-dimensional levels form its
            # spatial index; state_ind is a plain coordinate along it.
            coord_names = [
                name
                for name, coordinate in results.coords.items()
                if name not in {"state_bins", "state_ind"}
                and coordinate.dims == ("state_bins",)
            ]
            results = results.set_index(state_bins=coord_names)
        # The transformations above drop the file's close handle; closing the
        # returned results (or leaving a ``with`` block) must close the file.
        results.set_close(opened.close)
        return results

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
        Get one of this model's environments by name.

        Parameters
        ----------
        environment_name : str
            Name of the environment to retrieve.

        Returns
        -------
        Environment
            The environment object.

        Raises
        ------
        ValidationError
            If environment not found.
        """
        # Looked up in ``self.environments`` on every call: each fit replaces
        # them with new fitted copies, so a stored lookup would go stale.
        return find_environment_by_name(self.environments, environment_name)


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
            Sampling frequency, by default 500.0.
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
        # Stored as-is to keep sklearn's clone() contract intact; the dict
        # copy + default resolution happens lazily in
        # ``_resolve_clusterless_algorithm_params`` at fit time.
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def _refit_encoding_model(self, weights: np.ndarray) -> None:
        """EM M-step: refit the encoding model with posterior ``weights``.

        Posterior weights update the encoding process, not the
        training-fitted ``mark_reference_``.

        Parameters
        ----------
        weights : np.ndarray, shape (n_time_position,)
        """
        self.fit_encoding_model(
            **self._encoding_model_data, weights=weights, _refit_mark_reference=False
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

    def _clusterless_algorithm(self) -> LikelihoodAlgorithm:
        """The registered record of ``clusterless_algorithm``."""
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
        if mark_reference_scale is None:
            msg = (
                f"clusterless_algorithm {self.clusterless_algorithm!r} is "
                "registered without a mark_reference_scale"
            )
            raise ValidationError(
                msg,
                hint="Register it as LikelihoodAlgorithm(fit, predict, "
                "mark_reference_scale=...): the mark reference that scores "
                "No-Spike marks needs the algorithm's mark-kernel width.",
            )
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
        _refit_mark_reference: bool = True,
    ) -> None:
        """
        Fit the encoding model to the data.

        Parameters
        ----------
        position_time : array_like, shape (n_time_position,)
            Time points for position data.
        position : array_like, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
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
        _refit_mark_reference : bool, optional
            Internal. If False, keep the existing ``mark_reference_`` instead
            of refitting it, as for posterior-weighted encoding updates. By
            default True.

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

        # Build the new models locally so a failed refit keeps the previous
        # encoding model and its stored log likelihood.
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

            encoding_algorithm = self._clusterless_algorithm().fit
            is_group = is_training & is_encoding & is_environment
            # The mask is this model's exposure on the full timeline, as for
            # sorted spikes: a subset timeline would let spike-weight
            # interpolation bridge the mask's gaps.
            group_weights = _group_weights(is_group, weights, likelihood_name, support)
            is_group_spike = [
                _group_spike_mask(times, position_time, group_weights, support)
                for times in spike_times
            ]
            group_spike_times = [
                times[mask]
                for times, mask in zip(spike_times, is_group_spike, strict=True)
            ]
            group_spike_waveform_features = [
                features[mask]
                for features, mask in zip(
                    spike_waveform_features, is_group_spike, strict=True
                )
            ]
            encoding_model[likelihood_name] = encoding_algorithm(
                position_time=position_time,
                position=position,
                spike_times=group_spike_times,
                spike_waveform_features=group_spike_waveform_features,
                environment=environment,
                _encoding_support=support,
                weights=group_weights,
                **kwargs,
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
            if _refit_mark_reference
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
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
        validate_spike_feature_population(spike_times, spike_waveform_features)
        inputs = _EncodingInputs.from_arguments(
            position_time,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
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

    def _likelihood_predict_function(self) -> Callable[..., Any]:
        return self._clusterless_algorithm().predict

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
            raise ValidationError(
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
        _no_spike_time_bin_sizes: np.ndarray | None = None,
        _time_grid: _DecodeTimeGrid | None = None,
        _position_context: _LikelihoodPositionContext | None = None,
        _spike_time_order: _SpikeTimeOrder | None = None,
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_missing : np.ndarray, shape (n_rows,), optional
            Boolean array indicating missing data for the requested rows, by
            default None.
        row_slice : slice | None, optional
            Contiguous range of rows of the full-time likelihood to compute, by
            default None (all rows). ``time_edges`` stay the FULL decoding edges,
            so a decoding spike is owned by its global row no matter how the
            rows were chunked, and the result equals the full-time likelihood
            sliced by ``row_slice``.
        _no_spike_time_bin_sizes : np.ndarray | None, optional
            Internal full-timeline array of bin durations, prepared once by ``_predict``
            and reused for No-Spike in every chunk. None computes it on demand.
        _time_grid : _DecodeTimeGrid | None, optional
            Internal validated time edges shared within one prediction. None
            validates ``time_edges`` for this likelihood call.
        _position_context : _LikelihoodPositionContext | None, optional
            Internal tracking preparation shared within one prediction. None
            prepares it for this likelihood call.
        _spike_time_order : _SpikeTimeOrder | None, optional
            Internal ordering preparation shared across states and chunks. None
            creates fresh preparation for this likelihood call.

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
            _no_spike_time_bin_sizes=_no_spike_time_bin_sizes,
            _time_grid=_time_grid,
            _position_context=_position_context,
            _spike_time_order=_spike_time_order,
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        position : array_like, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : array_like, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
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
        return_outputs : str, list of str, set of str, or None, optional
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
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
        return_outputs : str, list of str, set of str, or None, optional
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
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
            Sampling frequency, by default 500.0.
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
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

        # Build the new models locally so a failed refit keeps the previous
        # encoding model and its stored log likelihood.
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
            encoding_algorithm = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ].fit
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

            encoding_model[likelihood_name] = encoding_algorithm(
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
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

    def _likelihood_predict_function(self) -> Callable[..., Any]:
        return _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm].predict

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
        _no_spike_time_bin_sizes: np.ndarray | None = None,
        _time_grid: _DecodeTimeGrid | None = None,
        _position_context: _LikelihoodPositionContext | None = None,
        _spike_time_order: _SpikeTimeOrder | None = None,
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_missing : np.ndarray, shape (n_rows,), optional
            Boolean array indicating missing data for the requested rows, by
            default None.
        row_slice : slice | None, optional
            Contiguous range of rows of the full-time likelihood to compute, by
            default None (all rows). ``time_edges`` stay the FULL decoding edges,
            so a decoding spike is owned by its global row no matter how the
            rows were chunked, and the result equals the full-time likelihood
            sliced by ``row_slice``.
        _no_spike_time_bin_sizes : np.ndarray | None, optional
            Internal full-timeline array of bin durations, prepared once by ``_predict``
            and reused for No-Spike in every chunk. None computes it on demand.
        _time_grid : _DecodeTimeGrid | None, optional
            Internal validated time edges shared within one prediction. None
            validates ``time_edges`` for this likelihood call.
        _position_context : _LikelihoodPositionContext | None, optional
            Internal tracking preparation shared within one prediction. None
            prepares it for this likelihood call.
        _spike_time_order : _SpikeTimeOrder | None, optional
            Internal ordering preparation shared across states and chunks. None
            creates fresh preparation for this likelihood call.

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
            _no_spike_time_bin_sizes=_no_spike_time_bin_sizes,
            _time_grid=_time_grid,
            _position_context=_position_context,
            _spike_time_order=_spike_time_order,
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
        spike_times : list of np.ndarray
            Spike times for each neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        position : array_like, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : array_like, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
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
        return_outputs : str, list of str, set of str, or None, optional
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
        """
        Estimate the initial conditions and transition probabilities
         using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time of each position sample.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            Each element of the list is an array of spike times for a neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : np.ndarray, shape (n_bins,), optional
            Denote missing samples, None includes all samples, by default None.
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
        return_outputs : str, list of str, set of str, or None, optional
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
            Each element of the list is an array of spike times for a neuron.
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the
            bin centers.
        is_missing : np.ndarray, shape (n_bins,), optional
            Denote missing samples, None includes all samples, by default None.
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
