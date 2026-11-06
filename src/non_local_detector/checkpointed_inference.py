"""Exact single-sequence forward/backward replay with disk boundary checkpoints.

At most two likelihood/forward/smoothed chunks are live: host work for one
chunk overlaps device work for the next. Selected output rows
remain conditioned on the complete recording. First-pass evidence increments
are summed in host float64 by default, preserving the state-probability dtype
and arithmetic. Explicit reference mode retains the original cumulative carry.
The host/disk driver is not a JAX
transformation target; existing differentiable pure-core APIs remain unchanged.
Likelihood callbacks must be deterministic and honor global ``row_slice``.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike, NDArray

from non_local_detector.core import (
    TransitionOperator,
    TransitionProducts,
    _call_log_likelihood_chunk,
    _check_operator_rows,
    _check_shape,
    _CovariateTransition,
    _degenerate_and_nan_masks_device,
    _filter_scan,
    _float64_evidence,
    _report_degenerate_and_nan_timesteps,
    _require_row_slice_aware,
    _smoother_scan,
    _StationaryTransition,
    _validate_hmm_inputs,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.result_store import (
    DEFAULT_MAX_READ_BYTES,
    IncrementalResultWriter,
    _validate_read_budget,
    open_result_store,
)
from non_local_detector.time_edges import decode_bin_centers, uniform_time_edges
from non_local_detector.types import (
    EvidenceAccumulation,
    OutputMode,
    TransitionStatistics,
)

_SPATIAL = {
    "causal_posterior",
    "predictive_posterior",
    "acausal_posterior",
    "log_likelihood",
}
_MARGINAL = {
    "causal_state_probabilities",
    "predictive_state_probabilities",
    "acausal_state_probabilities",
}


# Detector predictions size checkpoint chunks so one chunk of likelihoods is
# about this many bytes.
DEFAULT_CHUNK_BYTES = 256 * 1024**2
MIN_DEFAULT_CHUNK_SIZE = 256


# Largest default host memory for keeping forward-pass rows for the replay:
# enough for a 23.6 min recording on a 2 cm 2-D grid (about 14 GB of host
# memory in all) or about 4.5 hours on a 514-bin track at 500 Hz.
DEFAULT_REPLAY_CACHE_BYTES = 16 * 2**30


# cgroup v1 reports an unlimited memory limit as a value near 2**63.
_CGROUP_UNLIMITED = 2**60


def _read_cgroup_value(path: Path) -> int | None:
    """Return a cgroup memory file as an int, or None if absent, unlimited or unreadable."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value >= _CGROUP_UNLIMITED else value


def _cgroup_remaining_bytes(
    root: Path = Path("/sys/fs/cgroup"), membership: str | None = None
) -> int | None:
    """Memory left under this process's cgroup limits, or None if unlimited.

    A limit on any ancestor cgroup also binds the process, so this walks the
    memory controller's path from the process's cgroup up to the mount root
    and returns the smallest remaining allowance (limit minus that cgroup's
    usage). It reads cgroup v2 (``memory.max`` and ``memory.current``) and v1
    (``memory.limit_in_bytes`` and ``memory.usage_in_bytes`` under
    ``memory/``); path components not mounted, as in a container that sees its
    own cgroup at the root, are skipped. Linux host memory figures ignore a
    container's limit, so this caps them.

    Parameters
    ----------
    root : Path, optional
        Mount point of the cgroup file system.
    membership : str, optional
        Contents of ``/proc/self/cgroup``; read from there when None.

    Returns
    -------
    remaining : int or None
    """
    if membership is None:
        try:
            membership = Path("/proc/self/cgroup").read_text()
        except OSError:
            membership = ""
    hierarchies = []
    for line in membership.splitlines():
        hierarchy_id, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy_id == "0" and not controllers:
            hierarchies.append((root, path, "memory.max", "memory.current"))
        elif "memory" in controllers.split(","):
            hierarchies.append(
                (
                    root / "memory",
                    path,
                    "memory.limit_in_bytes",
                    "memory.usage_in_bytes",
                )
            )
    remaining = None
    for mount, path, limit_name, usage_name in hierarchies:
        directory = mount / path.strip("/")
        while True:
            limit = _read_cgroup_value(directory / limit_name)
            if limit is not None:
                usage = _read_cgroup_value(directory / usage_name) or 0
                allowance = max(limit - usage, 0)
                remaining = (
                    allowance if remaining is None else min(remaining, allowance)
                )
            if directory == mount:
                break
            directory = directory.parent
    return remaining


def _available_host_bytes() -> int | None:
    """Host memory available to this process, or None if it cannot be found.

    Uses psutil when installed, then ``MemAvailable`` from ``/proc/meminfo``
    (Linux), then half the physical memory, capped by the remaining memory
    allowance of a limited cgroup (a container).
    """
    host = None
    try:
        import psutil

        host = int(psutil.virtual_memory().available)
    except ImportError:
        try:
            with Path("/proc/meminfo").open() as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        host = int(line.split()[1]) * 1024
        except OSError:
            pass
        if host is None:
            try:
                host = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2
            except (ValueError, OSError, AttributeError):
                host = None
    container = _cgroup_remaining_bytes()
    known = [value for value in (host, container) if value is not None]
    return min(known) if known else None


def default_replay_cache_bytes() -> int:
    """Return the default replay cache size, in bytes.

    This is ``DEFAULT_REPLAY_CACHE_BYTES``, capped at half the available host
    memory, so the cache cannot exhaust a smaller machine.

    Returns
    -------
    replay_cache_bytes : int
        Replay cache budget in bytes.
    """
    available = _available_host_bytes()
    if available is None:
        return DEFAULT_REPLAY_CACHE_BYTES
    return int(min(DEFAULT_REPLAY_CACHE_BYTES, available // 2))


def default_chunk_size(n_bins: int, dtype: DTypeLike = np.float32) -> int:
    """Return rows per chunk so a likelihood chunk is ``DEFAULT_CHUNK_BYTES``.

    Parameters
    ----------
    n_bins : int
        Hidden state bins per row.
    dtype : numpy dtype, optional
        Likelihood dtype, by default ``np.float32``.

    Returns
    -------
    chunk_size : int
        At least ``MIN_DEFAULT_CHUNK_SIZE``. Larger chunks amortize per-chunk
        overhead; device working memory grows in proportion to chunk rows.
    """
    row_bytes = max(int(n_bins), 1) * np.dtype(dtype).itemsize
    return max(MIN_DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_BYTES // row_bytes)


# The chunk kernels are core's filter and smoother scan bodies, compiled
# without donation: the driver reuses its inputs.
_forward_chunk = jax.jit(
    _filter_scan, static_argnames=("keep_rows", "keep_predictive", "keep_increments")
)
_backward_chunk = jax.jit(_smoother_scan, static_argnames=("pair_statistics",))


@jax.jit
def _sum_state_rows(values: jax.Array, columns: jax.Array) -> jax.Array:
    """Use a tree reduction instead of sequential indexed accumulation.

    Only one state's columns from the current chunk are gathered at a time.
    Large float32 spatial grids otherwise accumulate appreciable roundoff in
    ``segment_sum`` despite the underlying spatial probabilities summing to one.
    """
    return jnp.sum(values[:, columns], axis=-1)


@dataclass
class CheckpointedResult:
    """A compact in-memory or lazy disk result plus whole-recording diagnostics.

    ``first_smoothed`` is the smoothed posterior of the first recording row,
    shape (n_bins,). ``transition_statistics`` holds the requested expected
    discrete transition counts, shape (n_states, n_states), or per-row
    responses, shape (n_time - 1, n_states, n_states).

    Attributes
    ----------
    dataset : xarray.Dataset
        Requested outputs for the selected rows, in memory or lazily read
        from ``result_path``, with the run's diagnostics in ``attrs``.
    marginal_log_likelihood : float
        Log marginal likelihood of the whole recording.
    diagnostics : dict
        Whole-recording diagnostics, such as counts and indices of degenerate
        rows and the number of likelihood evaluations.
    result_path : Path or None
        Directory of the disk result store, or None for an in-memory result.
    first_smoothed : np.ndarray or None
        Smoothed posterior of the first recording row, shape (n_bins,).
    transition_statistics : np.ndarray or None
        Expected discrete transition counts, shape (n_states, n_states), or
        per-row responses, shape (n_time - 1, n_states, n_states); None unless
        requested.
    """

    dataset: xr.Dataset
    marginal_log_likelihood: float
    diagnostics: dict[str, Any]
    result_path: Path | None = None
    first_smoothed: np.ndarray | None = None
    transition_statistics: np.ndarray | None = None


def _selected_rows(selected: ArrayLike | None, n_time: int) -> NDArray[np.intp]:
    if selected is None:
        return np.arange(n_time)
    selected = np.asarray(selected)
    if selected.dtype.kind == "b":
        _check_shape(
            "selected_rows Boolean mask",
            selected,
            (n_time,),
            "A Boolean mask selects rows of the whole recording, one entry per "
            "decode bin.",
        )
        selected = np.flatnonzero(selected)
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or np.any(selected < 0)
        or np.any(selected >= n_time)
        or np.any(selected[1:] <= selected[:-1])
    ):
        msg = "selected_rows must be unique, increasing recording row indices"
        raise ValidationError(
            msg,
            expected=f"increasing integers in [0, {n_time})",
            got=f"shape {selected.shape}, dtype {selected.dtype}",
            hint="Pass sorted unique row indices, e.g. np.unique(rows), or a "
            "Boolean mask over the decode bins.",
        )
    return selected.astype(np.intp, copy=False)


def _sum_evidence(chunks: Iterable[float]) -> float:
    """Float64 sum of per-chunk evidence with IEEE nonfinite propagation.

    Holds one float per chunk. Exceptions raised while producing chunks
    propagate unchanged; only ``math.fsum``'s own overflow is handled.
    """
    values = [float(value) for value in chunks]
    try:
        total = math.fsum(value for value in values if math.isfinite(value))
    except OverflowError:
        # Extreme float64 inputs can overflow fsum's intermediate partials;
        # preserve the scalar IEEE behavior of sequential addition.
        total = 0.0
        for value in values:
            total += value
        return total
    has_positive, has_negative = math.inf in values, -math.inf in values
    if any(math.isnan(value) for value in values) or (has_positive and has_negative):
        return math.nan
    if has_positive:
        return math.inf
    if has_negative:
        return -math.inf
    return total


def _mix32(bits: jax.Array) -> jax.Array:
    """Bijective 32-bit integer hash (lowbias32) in wrapping uint32 arithmetic."""
    bits = bits ^ (bits >> 16)
    bits = bits * jnp.uint32(0x7FEB352D)
    bits = bits ^ (bits >> 15)
    bits = bits * jnp.uint32(0x846CA68B)
    return bits ^ (bits >> 16)


@jax.jit
def _prepare_chunk(
    values: jax.Array, missing: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Neutralize missing rows; return diagnostic masks and a replay checksum.

    The checksum holds two wrapping uint32 sums over the value bits: one
    weighted by odd position weights, and one of a bijective hash of each
    element's bits mixed with its position. Integer addition is associative,
    so parallel reductions give identical bits on every backend. Any single
    changed element changes both sums, and the hashed sum catches structured
    multi-element changes, such as two sign flips, that cancel in a linear
    sum. It detects accidental changes but is not cryptographic.

    Parameters
    ----------
    values : jax.Array, shape (n_rows, n_bins)
        Chunk of values; rows flagged in ``missing`` are zeroed before the
        masks and checksum are computed.
    missing : jax.Array of bool, shape (n_rows,)
        True for rows to neutralize.

    Returns
    -------
    values : jax.Array, shape (n_rows, n_bins)
    degenerate, nan : jax.Array of bool, shape (n_rows,)
        Same semantics as ``core._degenerate_and_nan_masks``.
    checksum : jax.Array of uint32, shape (2,)
    """
    values = jnp.where(missing[:, None], jnp.zeros((), values.dtype), values)
    degenerate, nan = _degenerate_and_nan_masks_device(values)
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32).ravel()
    positions = jnp.arange(bits.size, dtype=jnp.uint32)
    checksum = jnp.stack(
        [
            jnp.sum(bits * (2 * positions + 1), dtype=jnp.uint32),
            jnp.sum(
                _mix32(bits ^ _mix32(positions + jnp.uint32(0x9E3779B9))),
                dtype=jnp.uint32,
            ),
        ]
    )
    return values, degenerate, nan, checksum


def checkpointed_forward_backward(
    time_edges: ArrayLike,
    initial_distribution: ArrayLike,
    log_likelihood_func: Callable[..., Any],
    *,
    log_likelihood_args: tuple[Any, ...] = (),
    transition_matrix: ArrayLike | None = None,
    transition_operator: TransitionProducts | None = None,
    discrete_transition_matrix: ArrayLike | None = None,
    continuous_transition_matrix: ArrayLike | None = None,
    state_ind: ArrayLike | None = None,
    n_states: int | None = None,
    is_missing: ArrayLike | None = None,
    chunk_size: int = 2000,
    output_mode: OutputMode = "compact",
    result_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    return_outputs: str | Iterable[str] | None = None,
    selected_rows: ArrayLike | None = None,
    dtype: DTypeLike = np.float32,
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    result_attrs: dict[str, Any] | Callable[[float], dict[str, Any]] | None = None,
    evidence_accumulation: EvidenceAccumulation = "stable",
    transition_statistics: TransitionStatistics | None = None,
    replay_cache_bytes: int = 0,
) -> CheckpointedResult:
    """Filter once, replay each chunk in reverse, and emit exact requested rows.

    The likelihood callback is called as ``log_likelihood_func(
    *log_likelihood_args, time_edges=time_edges, is_missing=chunk_mask,
    row_slice=global_slice)``, as core's drivers call it; with more than one
    chunk it must be marked :func:`~non_local_detector.core.row_slice_aware`,
    and an unmarked callback is called without ``row_slice`` for the single
    chunk. Counts must retain global edge/final-edge ownership.
    Every request spans ``min(chunk_size, n_time)`` rows: a short last chunk is
    requested over the final ``chunk_size`` rows and its earlier rows are
    discarded, so compiled likelihood kernels see one row count.
    This driver applies neutral likelihoods to missing rows and checks replay
    determinism. ``transition_operator`` is a registered JAX pytree implementing
    :class:`~non_local_detector.core.TransitionProducts` (``forward`` and
    ``backward``), or :class:`~non_local_detector.core.TransitionOperator` when
    transition statistics are requested; ``weight`` is None or the global
    discrete row.
    Alternatively supply a dense stationary matrix, or continuous/discrete
    factors and ``state_ind`` for the covariate path.

    Compact outputs are discrete-state marginals. Spatial outputs with a disk
    ``result_path`` remain lazy, without full-session spatial allocations;
    without one they are held in memory, ``n_time * n_bins`` values each.
    All inference runs over all recording rows regardless of ``selected_rows``.
    Initial/final conditions, impossible observations, NaNs, transition indices,
    filtering arithmetic and per-bin evidence increments follow the original
    core. Stable mode sums increments in host float64; reference mode retains
    the original cumulative evidence carry. Checkpoints are deleted on
    success/failure. Memory scales with chunk rows and hidden bins, excluding
    resident inputs/model, recording-sized metadata/marginals, compiler allocator
    reservations, and filesystem page cache. EM/Viterbi and global gradients are
    outside this API; it is not resumable after a crash.

    ``transition_statistics="counts"`` also accumulates the expected discrete
    transition counts between consecutive rows, as the stationary EM M-step
    uses them; ``"responses"`` keeps them per row, as the covariate M-step
    does. They are computed in the replay from the operator's per-block
    products (``pair_products``), so no spatial posterior is retained. Counts
    are summed per chunk on the device and across chunks in host float64;
    responses cost ``n_time * n_states**2`` values.

    ``result_attrs`` are attributes of the returned dataset, and of the result
    store's metadata; a callable is given the pass's marginal log-likelihood
    and returns them, for attributes that depend on it.

    ``replay_cache_bytes`` keeps each forward-pass chunk that still fits in the
    remaining budget in host memory, and the replay uses those instead of
    recomputing. When the outputs need only filtered rows (no likelihood or
    predictive output), the filtered rows are kept, so a cached chunk skips both
    its second likelihood evaluation and its replayed forward pass; otherwise
    the prepared likelihood is kept. Either costs ``n_bins`` values per row. By
    default (0) every chunk is filtered and evaluated twice.

    Global diagnostic masks
    cost two bytes per recording row while computing; exact NumPy index arrays
    cost up to two intp entries per row. Persisted masks use packed bits, avoiding
    per-row Python objects or large JSON index lists.

    Parameters
    ----------
    time_edges : array_like, shape (n_time + 1,)
        Uniform decoding bin boundaries in seconds.
    initial_distribution : array_like, shape (n_bins,)
        Initial probability of each hidden state bin: nonnegative, summing
        to 1.
    log_likelihood_func : callable
        Callback ``log_likelihood_func(*log_likelihood_args,
        time_edges=time_edges, is_missing=..., row_slice=...)`` returning the
        log likelihood of the requested global rows, shape (n_rows, n_bins).
        Marked by :func:`~non_local_detector.core.row_slice_aware` unless
        ``chunk_size`` covers every row.
    log_likelihood_args : tuple, optional
        Positional arguments passed to ``log_likelihood_func``; by default
        none.
    transition_matrix : array_like, shape (n_bins, n_bins), optional
        Dense stationary transition matrix, nonnegative with each row summing
        to at most 1. Used when neither ``transition_operator`` nor
        ``discrete_transition_matrix`` is given.
    transition_operator : TransitionProducts, optional
        Registered JAX pytree used instead of a dense matrix. Filtering and
        smoothing need only :class:`~non_local_detector.core.TransitionProducts`
        (``forward`` and ``backward``); ``transition_statistics`` needs a
        :class:`~non_local_detector.core.TransitionOperator`, which adds
        ``pair_products`` and ``backward_and_pairs`` (inherited only by
        subclassing ``TransitionOperator``). Its rows are checked through
        their sums ``backward(1)``, as ``transition_matrix``'s rows are: at
        most 1. With ``discrete_transition_matrix``, each bin's row is
        checked per discrete state, as ``continuous_transition_matrix``'s
        is, from the product with each weight whose column ``k`` is one.
    discrete_transition_matrix : array_like, shape (n_time, n_states, n_states), optional
        Per-row discrete state transition weights for the covariate path,
        combined with ``continuous_transition_matrix``: nonnegative, each row
        summing to 1.
    continuous_transition_matrix : array_like, shape (n_bins, n_bins), optional
        Dense continuous transition matrix used with
        ``discrete_transition_matrix``: nonnegative, each bin's row summing to
        at most 1 over the bins of each discrete state.
    state_ind : array_like of int, shape (n_bins,), optional
        Discrete state index of each hidden bin. Required for
        ``output_mode="compact"``; by default each bin is its own state.
    n_states : int, optional
        Number of discrete states; by default the last axis of
        ``discrete_transition_matrix``, or one more than the largest
        ``state_ind``.
    is_missing : array_like of bool, shape (n_time,), optional
        Rows without observations, which get neutral likelihoods. By default
        no row is missing.
    chunk_size : int, optional
        Number of rows per checkpoint chunk, by default 2000.
    output_mode : {"compact", "spatial"}, optional
        ``"compact"`` returns discrete-state probabilities; ``"spatial"``
        returns per-bin posteriors. By default "compact".
    result_path : str or Path, optional
        If given, write the outputs to this new directory and return a lazy
        view; otherwise outputs are held in memory.
    checkpoint_dir : str or Path, optional
        Parent directory for temporary checkpoints, by default the parent of
        ``result_path`` (or the system temporary directory).
    return_outputs : str or iterable of str, optional
        Names of the outputs to return, such as ``"acausal_posterior"``,
        ``"causal_posterior"``, ``"predictive_posterior"``, ``"log_likelihood"``
        or the ``*_state_probabilities`` marginals. By default the acausal
        state probabilities (compact) or posterior (spatial).
    selected_rows : array_like, optional
        Increasing row indices, or a boolean mask of shape (n_time,), of the
        rows to return. Inference still uses all rows. By default all rows.
    dtype : numpy dtype, optional
        State-probability dtype, ``float32`` or ``float64`` (the latter
        requires ``JAX_ENABLE_X64=1``), by default ``np.float32``.
    max_read_bytes : int, optional
        Largest single explicit read from a disk result, in bytes, by default
        512 MiB.
    result_attrs : dict or callable, optional
        Attributes added to the returned dataset and store metadata, or a
        callable mapping the marginal log likelihood to such a dict.
    evidence_accumulation : {"stable", "reference"}, optional
        ``"stable"`` sums evidence increments in host float64; ``"reference"``
        keeps the original cumulative carry. By default "stable".
    transition_statistics : {"counts", "responses"}, optional
        Also compute expected discrete transition counts or per-row
        responses, from the operator's ``backward_and_pairs``. By default
        None.
    replay_cache_bytes : int, optional
        Host memory budget for keeping forward-pass chunks to avoid
        recomputation in the replay, by default 0 (no cache).

    Returns
    -------
    result : CheckpointedResult
        Dataset of the requested outputs, the marginal log likelihood and
        diagnostics.

    Raises
    ------
    ValidationError
        If an argument is invalid, such as a non-positive ``chunk_size``, an
        unrecognized output, an input that breaks the shape, dtype or
        probability contract of the dense drivers (float64 requires JAX's
        64-bit mode), a ``transition_operator`` whose row sums break it or
        that cannot be applied to ``n_bins`` state bins, or a likelihood
        callback that returns the wrong shape, changes between passes, or is
        not ``row_slice_aware`` although ``chunk_size`` splits the rows.
    DataError
        If a log likelihood outside the bins marked missing is NaN or ``+inf``.
    TypeError
        If ``transition_operator`` is not a registered JAX pytree implementing
        :class:`~non_local_detector.core.TransitionProducts`, or, with
        ``transition_statistics``, :class:`~non_local_detector.core.TransitionOperator`.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))  # type: ignore[redundant-expr]  # runtime check of caller input
        or chunk_size <= 0
    ):
        msg = "chunk_size must be a positive integer"
        raise ValidationError(
            msg,
            expected="an integer of at least 1",
            got=repr(chunk_size),
            hint="Pass the number of decode bins per checkpoint chunk.",
        )
    if evidence_accumulation not in ("reference", "stable"):
        msg = "evidence_accumulation must be reference or stable"
        raise ValidationError(
            msg,
            expected="'stable' or 'reference'",
            got=repr(evidence_accumulation),
            hint="Use 'stable' (float64 host sums) unless reproducing the "
            "original cumulative carry.",
        )
    if transition_statistics not in (None, "counts", "responses"):
        msg = "transition_statistics must be None, counts or responses"
        raise ValidationError(
            msg,
            expected="None, 'counts' or 'responses'",
            got=repr(transition_statistics),
            hint="'counts' sums the expected discrete transitions over the "
            "recording; 'responses' keeps them per row.",
        )
    if (
        isinstance(replay_cache_bytes, bool)
        or not isinstance(replay_cache_bytes, (int, np.integer))  # type: ignore[redundant-expr]  # runtime check of caller input
        or replay_cache_bytes < 0
    ):
        msg = "replay_cache_bytes must be a nonnegative integer"
        raise ValidationError(
            msg,
            expected="an integer number of bytes, at least 0",
            got=repr(replay_cache_bytes),
            hint="Pass 0 to recompute every chunk in the replay.",
        )
    if output_mode not in ("compact", "spatial"):
        msg = "output_mode must be compact or spatial"
        raise ValidationError(
            msg,
            expected="'compact' or 'spatial'",
            got=repr(output_mode),
            hint="'compact' returns discrete-state probabilities; 'spatial' "
            "returns per-bin posteriors.",
        )
    max_read_bytes = _validate_read_budget(max_read_bytes)
    edges, _ = uniform_time_edges(time_edges)
    n_time = len(edges) - 1
    if state_ind is None:
        if output_mode == "compact":
            msg = "Compact state marginals require state_ind"
            raise ValidationError(
                msg,
                expected="state_ind with output_mode='compact'",
                got="state_ind=None",
                hint="Pass the discrete state of each hidden bin, or use "
                "output_mode='spatial'.",
            )
        state_ind = np.arange(np.size(initial_distribution))
        if discrete_transition_matrix is not None and np.shape(
            discrete_transition_matrix
        )[-1:] != (len(state_ind),):
            msg = (
                "discrete_transition_matrix needs one discrete state per hidden "
                "bin without state_ind"
            )
            raise ValidationError(
                msg,
                expected=f"shape ({n_time}, {len(state_ind)}, {len(state_ind)})",
                got=f"shape {np.shape(discrete_transition_matrix)}",
                hint="Pass state_ind, the discrete state of each hidden bin, to "
                "use fewer discrete states than hidden bins.",
            )
    covariate = discrete_transition_matrix is not None
    dense = transition_operator is None
    dtype, validated_states = _validate_hmm_inputs(
        n_time=n_time,
        initial_distribution=initial_distribution,
        dtype=np.dtype(dtype),
        state_ind=state_ind,
        n_states=n_states,
        transition_matrix=transition_matrix if dense and not covariate else None,
        discrete_transition_matrix=discrete_transition_matrix,
        continuous_transition_matrix=(
            continuous_transition_matrix if dense and covariate else None
        ),
        is_missing=is_missing,
    )
    assert validated_states is not None  # state_ind is always set here
    n_states = validated_states
    initial = np.asarray(initial_distribution, dtype=dtype)
    n_bins = len(initial)
    state_ind = np.asarray(state_ind)
    missing = (
        np.zeros(n_time, bool)
        if is_missing is None
        else np.asarray(is_missing, dtype=bool)
    )
    outputs = (
        (
            ("acausal_state_probabilities",)
            if output_mode == "compact"
            else ("acausal_posterior",)
        )
        if return_outputs is None
        else return_outputs
    )
    outputs = {outputs} if isinstance(outputs, str) else set(outputs)
    if not outputs or outputs - (_SPATIAL | _MARGINAL):
        msg = "Unrecognized or empty return_outputs"
        raise ValidationError(
            msg,
            expected=f"a nonempty subset of {sorted(_SPATIAL | _MARGINAL)}",
            got=repr(sorted(outputs)),
            hint="Name each output to return.",
        )
    if output_mode == "compact" and outputs & _SPATIAL:
        msg = "Compact mode only retains state probabilities"
        raise ValidationError(
            msg,
            expected=f"outputs among {sorted(_MARGINAL)}",
            got=repr(sorted(outputs & _SPATIAL)),
            hint="Use output_mode='spatial' for per-bin outputs.",
        )
    state_columns = (
        tuple(
            jnp.asarray(np.flatnonzero(state_ind == state)) for state in range(n_states)
        )
        if outputs & _MARGINAL
        else ()
    )
    selected = _selected_rows(selected_rows, n_time)
    weights = (
        None if not covariate else np.asarray(discrete_transition_matrix, dtype=dtype)
    )
    if transition_operator is None:
        matrix = continuous_transition_matrix if covariate else transition_matrix
        if matrix is None:
            needed = (
                "continuous_transition_matrix" if covariate else "transition_matrix"
            )
            msg = "checkpointed_forward_backward needs a transition"
            raise ValidationError(
                msg,
                expected="transition_matrix, transition_operator, or "
                "continuous_transition_matrix with discrete_transition_matrix",
                got=f"no {needed}",
                hint=f"Pass {needed} or a transition_operator.",
            )
        # The covariate operator applies each row's discrete matrix per
        # discrete state, as core does, without building the joint matrix.
        transition_operator = (
            _CovariateTransition.create(
                jnp.asarray(matrix, dtype=dtype), state_ind, n_states
            )
            if covariate
            else _StationaryTransition(
                jnp.asarray(matrix, dtype=dtype),
                jnp.asarray(state_ind),
                n_states,
            )
        )
    else:
        leaves, _ = jax.tree_util.tree_flatten(transition_operator)
        if len(leaves) == 1 and leaves[0] is transition_operator:
            # An unregistered object would be a static jit argument: its arrays
            # would be baked into the compiled chunks and later changes ignored.
            msg = (
                "transition_operator must be a registered JAX pytree (for "
                "example via jax.tree_util.register_pytree_node_class), got "
                f"{type(transition_operator).__name__}."
            )
            raise TypeError(msg)
        protocol, required = (
            (TransitionProducts, ("forward", "backward"))
            if transition_statistics is None
            else (
                TransitionOperator,
                ("forward", "backward", "pair_products", "backward_and_pairs"),
            )
        )
        if not isinstance(transition_operator, protocol):
            missing_methods = [  # type: ignore[unreachable]  # runtime check of caller input
                name
                for name in required
                if not callable(getattr(transition_operator, name, None))
            ]
            msg = (
                f"transition_operator must implement "
                f"non_local_detector.core.{protocol.__name__}"
                f"{'' if transition_statistics is None else ' for transition_statistics'}; "
                f"{type(transition_operator).__name__} lacks "
                f"{', '.join(missing_methods)}. Subclassing TransitionOperator "
                "provides backward_and_pairs."
            )
            raise TypeError(msg)
        # Transfer array leaves once; each jitted chunk call would otherwise
        # convert host leaves again.
        transition_operator = jax.device_put(transition_operator)
        _check_operator_rows(
            transition_operator,
            n_bins,
            dtype,
            n_states if covariate else None,
            pair_states=None if transition_statistics is None else n_states,
        )
    dims = {
        name: ("time", "state_bins" if name in _SPATIAL else "states")
        for name in outputs
    }
    shapes = {
        name: (len(selected), n_bins if name in _SPATIAL else n_states)
        for name in outputs
    }
    coordinates = {
        "time": (("time",), decode_bin_centers(edges, 0, n_time)[selected]),
        "time_bin_start": (("time",), edges[:-1][selected]),
        "time_bin_end": (("time",), edges[1:][selected]),
        "time_bin_width": (("time",), np.diff(edges)[selected]),
        "source_row": (("time",), selected),
        "is_missing": (("time",), missing[selected]),
        "states": (("states",), np.arange(n_states)),
    }
    if outputs & _SPATIAL:
        coordinates["state_bins"] = (("state_bins",), np.arange(n_bins))
    arrays: dict[str, NDArray[Any]] | None = (
        {name: np.empty(shapes[name], dtype=dtype) for name in outputs}
        if result_path is None
        else None
    )
    checkpoint_parent = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else Path(result_path).parent
        if result_path is not None
        else None
    )
    if checkpoint_parent is not None:
        checkpoint_parent.mkdir(parents=True, exist_ok=True)
    writer = (
        None
        if result_path is None
        else IncrementalResultWriter(
            result_path,
            {name: (dims[name], shapes[name], dtype) for name in outputs},
            coordinates,
        )
    )
    boundaries = range(0, n_time, int(chunk_size))
    _require_row_slice_aware(
        log_likelihood_func,
        len(boundaries),
        chunking=f"chunk_size={chunk_size} for {n_time} rows",
        hint="Mark the callback, or pass a chunk_size of at least the number of rows.",
    )
    diagnostics: dict[str, Any] = {
        "n_degenerate": 0,
        "max_chunk_rows": 0,
        "likelihood_evaluations": 0,
        "checkpoint_bytes": 0,
        "checkpoint_cache_entries": 1,
        "replay_cache_bytes": 0,
        "global_gradients_supported": False,
    }

    degenerate_rows = np.zeros(n_time, dtype=bool)
    n_nan = 0
    pair_statistics = transition_statistics is not None
    statistics = (
        np.zeros((n_states, n_states))
        if transition_statistics == "counts"
        else np.empty((max(n_time - 1, 0), n_states, n_states), dtype=dtype)
        if transition_statistics == "responses"
        else None
    )

    def likelihood(
        start: int, stop: int
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return device ``(values, degenerate, nan, checksum)`` for one chunk."""
        # A short last chunk is evaluated over the final chunk_size rows and
        # trimmed, so every callback has one shape and reuses compiled kernels.
        first = max(min(start, n_time - int(chunk_size)), 0)
        values = jnp.asarray(
            _call_log_likelihood_chunk(
                log_likelihood_func,
                time_edges,
                log_likelihood_args,
                slice(first, stop),
                missing[first:stop],
            ),
            dtype=dtype,
        )
        _check_shape(
            "log_likelihood_func output",
            values,
            (stop - first, n_bins),
            "The likelihood callback must return exactly the requested global "
            "rows (row_slice) and one column per hidden bin.",
        )
        diagnostics["likelihood_evaluations"] += 1
        diagnostics["max_chunk_rows"] = max(diagnostics["max_chunk_rows"], stop - first)
        return cast(
            "tuple[jax.Array, jax.Array, jax.Array, jax.Array]",
            _prepare_chunk(values[start - first :], jnp.asarray(missing[start:stop])),
        )

    def emit(name: str, start: int, stop: int, values: Any) -> None:
        first, last = np.searchsorted(selected, [start, stop])
        if first == last:
            return
        rows = selected[first:last] - start
        values = np.asarray(values)
        # Contiguous rows (the usual case) are a view rather than a copy.
        values = (
            values[rows[0] : rows[-1] + 1]
            if rows[-1] - rows[0] + 1 == len(rows)
            else values[rows]
        )
        if writer is None:
            assert arrays is not None  # arrays is built whenever result_path is None
            arrays[name][first:last] = values
        else:
            writer.write(name, int(first), values)

    def emit_chunk(
        start: int,
        stop: int,
        chunk_outputs: dict[str, Any],
        chunk_statistics: Any = None,
    ) -> None:
        for name, value in chunk_outputs.items():
            emit(name, start, stop, value)
        if chunk_statistics is None:
            return
        assert statistics is not None  # chunk statistics exist only when requested
        if transition_statistics == "counts":
            statistics[...] += np.asarray(chunk_statistics, np.float64)
        else:
            statistics[start : min(stop, n_time - 1)] = np.asarray(chunk_statistics)

    try:
        with tempfile.TemporaryDirectory(
            prefix="checkpoints-", dir=checkpoint_parent
        ) as folder_name:
            folder = Path(folder_name)
            predicted, evidence = jnp.asarray(initial), jnp.asarray(0, dtype=dtype)

            # Filtered rows suffice unless an output needs the likelihood or the
            # predictive rows, which only a replayed forward pass gives.
            cache_rows = not outputs & {
                "log_likelihood",
                "predictive_posterior",
                "predictive_state_probabilities",
            }
            cached_rows: dict[int, NDArray[Any]] = {}
            cached_likelihoods: dict[int, NDArray[Any]] = {}

            def finish_forward_chunk(
                start: int,
                stop: int,
                predicted: jax.Array,
                evidence: jax.Array,
                degenerate: Any,
                nan: Any,
                checksum: jax.Array,
                cached: jax.Array | None,
                increments: Any,
            ) -> Any:
                nonlocal n_nan
                """Pull one dispatched chunk's results to the host and checkpoint it."""
                if cached is not None:
                    (cached_rows if cache_rows else cached_likelihoods)[start] = (
                        np.asarray(cached)
                    )
                    diagnostics["replay_cache_bytes"] += cached.nbytes
                checkpoint = folder / f"{start}.npz"
                np.savez(
                    checkpoint,
                    predicted=np.asarray(predicted),
                    evidence=np.asarray(evidence),
                    checksum=np.asarray(checksum),
                )
                diagnostics["checkpoint_bytes"] += checkpoint.stat().st_size
                degenerate = np.asarray(degenerate)
                degenerate_rows[start:stop] = degenerate
                diagnostics["n_degenerate"] += int(degenerate.sum())
                n_nan += int(np.asarray(nan).sum())
                return (
                    _float64_evidence(increments)
                    if evidence_accumulation == "stable"
                    else 0.0
                )

            def forward_increment_sums() -> Iterator[Any]:
                nonlocal predicted, evidence
                # Each chunk's host work runs after the next chunk is dispatched,
                # so device computation overlaps host transfers and disk writes.
                pending = None
                planned = 0
                for start in boundaries:
                    stop = min(start + chunk_size, n_time)
                    nbytes = (stop - start) * n_bins * dtype.itemsize
                    cache = planned + nbytes <= replay_cache_bytes
                    planned += nbytes if cache else 0
                    values, degenerate, nan, checksum = likelihood(start, stop)
                    chunk = (
                        start,
                        stop,
                        predicted,
                        evidence,
                        degenerate,
                        nan,
                        checksum,
                    )
                    (evidence, predicted), out = _forward_chunk(
                        predicted,
                        evidence,
                        values,
                        None if weights is None else jnp.asarray(weights[start:stop]),
                        transition_operator,
                        keep_rows=cache and cache_rows,
                        keep_predictive=False,
                        keep_increments=evidence_accumulation == "stable",
                    )
                    if cache and cache_rows:
                        cached, _, increments = out
                    else:
                        cached, increments = (values if cache else None), out
                    if pending is not None:
                        yield finish_forward_chunk(*pending)
                    # Kept until the chunk finishes and is copied to the host.
                    pending = (*chunk, cached, increments)
                    del values, cached
                if pending is not None:
                    yield finish_forward_chunk(*pending)

            if evidence_accumulation == "stable":
                marginal = _sum_evidence(forward_increment_sums())
            else:
                for _ in forward_increment_sums():
                    pass
                marginal = float(evidence)
            _report_degenerate_and_nan_timesteps(
                diagnostics["n_degenerate"],
                n_nan,
                n_time,
                marginal_log_likelihood=marginal,
            )
            next_smoothed = jnp.asarray(initial)
            # Outputs are emitted after the next chunk is dispatched; each chunk's
            # checksum is verified before its outputs are emitted.
            pending = None
            for start in reversed(boundaries):
                stop = min(start + chunk_size, n_time)
                with np.load(folder / f"{start}.npz", allow_pickle=False) as checkpoint:
                    expected_checksum = checkpoint["checksum"]
                    prior, start_evidence = (
                        jnp.asarray(checkpoint["predicted"]),
                        jnp.asarray(checkpoint["evidence"]),
                    )
                chunk_weights = (
                    None if weights is None else jnp.asarray(weights[start:stop])
                )
                if start in cached_rows:
                    # No output needs the likelihood or predictive rows here.
                    filtered = jnp.asarray(cached_rows.pop(start))
                    values = predictive = None
                    checksum = expected_checksum
                else:
                    if start in cached_likelihoods:
                        values = jnp.asarray(cached_likelihoods.pop(start))
                        checksum = expected_checksum
                    else:
                        values, _, _, checksum = likelihood(start, stop)
                    _, (filtered, predictive, _) = _forward_chunk(
                        prior,
                        start_evidence,
                        values,
                        chunk_weights,
                        transition_operator,
                        keep_rows=True,
                        keep_predictive=bool(
                            outputs
                            & {"predictive_posterior", "predictive_state_probabilities"}
                        ),
                    )
                if stop == n_time:
                    next_smoothed = filtered[-1]
                next_smoothed, smoothed = _backward_chunk(
                    filtered,
                    chunk_weights,
                    transition_operator,
                    next_smoothed,
                    jnp.arange(start, stop),
                    n_time,
                    pair_statistics=pair_statistics,
                )
                chunk_statistics = None
                if pair_statistics:
                    smoothed, pairs = smoothed
                    # Kept on the device and copied with the chunk's outputs,
                    # after the next chunk is dispatched. The last recording
                    # row has no successor.
                    chunk_statistics = (
                        jnp.sum(pairs, axis=0)
                        if transition_statistics == "counts"
                        else pairs[: min(stop, n_time - 1) - start]
                    )
                    del pairs
                chunk_outputs = {}
                for name in outputs:
                    value = (
                        values
                        if name == "log_likelihood"
                        else filtered
                        if name.startswith("causal")
                        else predictive
                        if name.startswith("predictive")
                        else smoothed
                    )
                    if name in _MARGINAL:
                        value = jnp.stack(
                            [
                                _sum_state_rows(value, columns)
                                for columns in state_columns
                            ],
                            axis=1,
                        )
                    chunk_outputs[name] = value
                # Only the requested outputs and the smoothed first-row carry
                # survive the next iteration.
                del values, filtered, predictive, smoothed, chunk_weights, prior
                if pending is not None:
                    emit_chunk(*pending)
                if not np.array_equal(np.asarray(checksum), expected_checksum):
                    msg = (
                        "Likelihood callback changed during checkpoint replay; "
                        "deterministic inputs are required"
                    )
                    raise ValidationError(
                        msg,
                        expected="the same values for a row_slice on every call",
                        got=f"different values for rows {start} to {stop}",
                        hint="Make the likelihood callback deterministic: no "
                        "randomness or state that changes between the forward "
                        "pass and the replay.",
                    )
                pending = (start, stop, chunk_outputs, chunk_statistics)
            if pending is not None:
                emit_chunk(*pending)
        diagnostics["degenerate_indices"] = np.flatnonzero(degenerate_rows)
        packed_masks = {
            "degenerate_row_mask_hex": np.packbits(degenerate_rows, bitorder="little")
            .tobytes()
            .hex(),
            "diagnostic_row_mask_encoding": "packbits_little_hex",
            "diagnostics_are_global": True,
        }
        if callable(result_attrs):
            result_attrs = result_attrs(marginal)
        attrs = {
            **(result_attrs or {}),
            "marginal_log_likelihood": marginal,
            "evidence_accumulation": evidence_accumulation,
            "evidence_dtype": "float64"
            if evidence_accumulation == "stable"
            else dtype.name,
            "state_probability_dtype": dtype.name,
            "inference_mode": "checkpointed",
            "output_mode": output_mode,
            "n_recording_bins": n_time,
            "dtype": dtype.name,
            "whole_recording_conditioning": True,
            **{
                key: value
                for key, value in diagnostics.items()
                if not key.endswith("_indices")
            },
            **packed_masks,
        }
        if writer is None:
            assert arrays is not None  # arrays is built whenever result_path is None
            dataset = xr.Dataset(
                {name: (dims[name], arrays[name]) for name in outputs},
                coords=coordinates,
                attrs=attrs,
            )
        else:
            writer.complete(attrs)
            assert result_path is not None  # a writer exists only with result_path
            dataset = open_result_store(result_path, max_read_bytes=max_read_bytes)
        return CheckpointedResult(
            dataset,
            marginal,
            diagnostics,
            None if result_path is None else Path(result_path),
            first_smoothed=np.asarray(next_smoothed),
            transition_statistics=statistics,
        )
    finally:
        if writer is not None:
            writer.abort()
