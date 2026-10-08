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
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from non_local_detector.core import (
    _condition_on,
    _divide_safe,
    _normalize,
    _warn_degenerate_and_nan_timesteps,
)
from non_local_detector.likelihoods.common import decode_bin_centers
from non_local_detector.result_store import (
    IncrementalResultWriter,
    _validate_read_budget,
    open_result_store,
)
from non_local_detector.time_edges import uniform_time_edges

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


@jax.tree_util.register_pytree_node_class
class _DenseTransition:
    def __init__(self, matrix, state_ind=None):
        self.matrix, self.state_ind = matrix, state_ind

    def _matrix(self, discrete_weights):
        if discrete_weights is None:
            return self.matrix
        return self.matrix * discrete_weights[jnp.ix_(self.state_ind, self.state_ind)]

    def forward(self, probabilities, discrete_weights=None):
        return jnp.matmul(
            probabilities,
            self._matrix(discrete_weights),
            precision=jax.lax.Precision.HIGHEST,
        )

    def backward(self, values, discrete_weights=None):
        return jnp.matmul(
            self._matrix(discrete_weights),
            values,
            precision=jax.lax.Precision.HIGHEST,
        )

    def tree_flatten(self):
        return (self.matrix, self.state_ind), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class _CallableTransition:
    """Use a non-pytree protocol object as a static pair of JAX functions."""

    def __init__(self, operator):
        self.forward, self.backward = operator.forward, operator.backward

    def tree_flatten(self):
        return (), (self.forward, self.backward)

    @classmethod
    def tree_unflatten(cls, aux, children):
        result = object.__new__(cls)
        result.forward, result.backward = aux
        return result


@partial(jax.jit, static_argnames=("keep_rows", "keep_predictive", "keep_increments"))
def _forward_chunk(
    initial,
    evidence,
    likelihoods,
    weights,
    operator,
    *,
    keep_rows,
    keep_predictive,
    keep_increments=False,
):
    def step(carry, args):
        total, predicted = carry
        ll, weight = args
        filtered, increment = _condition_on(predicted, ll)
        out = (
            (filtered, predicted if keep_predictive else None)
            if keep_rows
            else increment
            if keep_increments
            else None
        )
        return (total + increment, operator.forward(filtered, weight)), out

    return jax.lax.scan(step, (evidence, initial), (likelihoods, weights))


@jax.jit
def _backward_chunk(filtered, weights, operator, next_smoothed, rows, n_time):
    def step(following, args):
        current, weight, row = args
        predicted = operator.forward(current, weight)
        # Keep the core's arithmetic, including terminal and NaN semantics.
        smoothed = current * operator.backward(
            _divide_safe(following, predicted), weight
        ) * (row < n_time - 1) + current * (row == n_time - 1)
        smoothed, _ = _normalize(smoothed, axis=-1)
        return smoothed, smoothed

    return jax.lax.scan(step, next_smoothed, (filtered, weights, rows), reverse=True)


@jax.jit
def _sum_state_rows(values, columns):
    """Use a tree reduction instead of sequential indexed accumulation.

    Only one state's columns from the current chunk are gathered at a time.
    Large float32 spatial grids otherwise accumulate appreciable roundoff in
    ``segment_sum`` despite the underlying spatial probabilities summing to one.
    """
    return jnp.sum(values[:, columns], axis=-1)


@dataclass
class CheckpointedResult:
    """A compact in-memory or lazy disk result plus whole-recording diagnostics."""

    dataset: xr.Dataset
    marginal_log_likelihood: float
    diagnostics: dict
    result_path: Path | None = None


def _selected_rows(selected, n_time):
    if selected is None:
        return np.arange(n_time)
    selected = np.asarray(selected)
    if selected.dtype.kind == "b":
        if selected.shape != (n_time,):
            raise ValueError("selected_rows Boolean mask must match the recording")
        selected = np.flatnonzero(selected)
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or np.any(selected < 0)
        or np.any(selected >= n_time)
        or np.any(selected[1:] <= selected[:-1])
    ):
        raise ValueError(
            "selected_rows must be unique, increasing recording row indices"
        )
    return selected.astype(np.intp, copy=False)


def _sum_evidence(chunks):
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


@jax.jit
def _prepare_chunk(values, missing):
    """Neutralize missing rows; return diagnostic masks and a replay checksum.

    The checksum sums the value bits, plain and with odd position weights, in
    wrapping uint32 arithmetic. Integer addition is associative, so parallel
    reductions give identical bits on every backend. Any single changed element
    changes both sums; it detects accidental changes but is not cryptographic.

    Parameters
    ----------
    values : jax.Array, shape (n_rows, n_bins)
    missing : jax.Array of bool, shape (n_rows,)

    Returns
    -------
    values : jax.Array, shape (n_rows, n_bins)
    degenerate, nan : jax.Array of bool, shape (n_rows,)
        Same semantics as ``core._degenerate_and_nan_masks``.
    checksum : jax.Array of uint32, shape (2,)
    """
    values = jnp.where(missing[:, None], jnp.zeros((), values.dtype), values)
    degenerate = values.max(axis=-1) == -jnp.inf
    nan = jnp.any(jnp.isnan(values), axis=-1)
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32).ravel()
    position_weights = 2 * jnp.arange(bits.size, dtype=jnp.uint32) + 1
    checksum = jnp.stack(
        [
            jnp.sum(bits, dtype=jnp.uint32),
            jnp.sum(bits * position_weights, dtype=jnp.uint32),
        ]
    )
    return values, degenerate, nan, checksum


def checkpointed_forward_backward(
    time_edges,
    initial_distribution,
    log_likelihood_func,
    *,
    transition_matrix=None,
    transition_operator=None,
    discrete_transition_matrix=None,
    continuous_transition_matrix=None,
    state_ind=None,
    n_states=None,
    is_missing=None,
    chunk_size=2000,
    output_mode="compact",
    result_path=None,
    checkpoint_dir=None,
    return_outputs=None,
    selected_rows=None,
    dtype=np.float32,
    max_read_bytes=512 * 1024**2,
    result_attrs=None,
    evidence_accumulation="stable",
):
    """Filter once, replay each chunk in reverse, and emit exact requested rows.

    The likelihood callback receives ``(full_edges, row_slice=global_slice,
    is_missing=chunk_mask)``. Counts must retain global edge/final-edge ownership.
    This driver applies neutral likelihoods to missing rows and checks replay
    determinism. ``transition_operator`` implements pure-JAX ``forward(p, weight)``
    and ``backward(v, weight)``; ``weight`` is None or the global discrete row.
    Alternatively supply a dense stationary matrix, or continuous/discrete
    factors and ``state_ind`` for the reference covariate path.

    Compact outputs are discrete-state marginals. Spatial outputs require a disk
    ``result_path`` and remain lazy, without full-session spatial allocations.
    All inference runs over all recording rows regardless of ``selected_rows``.
    Initial/final conditions, impossible observations, NaNs, transition indices,
    filtering arithmetic and per-bin evidence increments follow the original
    core. Stable mode sums increments in host float64; reference mode retains
    the original cumulative evidence carry. Checkpoints are deleted on
    success/failure. Memory scales with chunk rows and hidden bins, excluding
    resident inputs/model, recording-sized metadata/marginals, compiler allocator
    reservations, and filesystem page cache. EM/Viterbi and global gradients are
    outside this API; it is not resumable after a crash. Global diagnostic masks
    cost two bytes per recording row while computing; exact NumPy index arrays
    cost up to two intp entries per row. Persisted masks use packed bits, avoiding
    per-row Python objects or large JSON index lists.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if evidence_accumulation not in ("reference", "stable"):
        raise ValueError("evidence_accumulation must be reference or stable")
    max_read_bytes = _validate_read_budget(max_read_bytes)
    edges, _ = uniform_time_edges(time_edges)
    n_time, dtype = len(edges) - 1, np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError(
            "float64 requires JAX_ENABLE_X64=1; precision is not silently downgraded"
        )
    initial = np.asarray(initial_distribution, dtype=dtype)
    if initial.ndim != 1 or not len(initial):
        raise ValueError("initial_distribution must be a nonempty vector")
    n_bins = len(initial)
    if state_ind is None:
        if output_mode == "compact":
            raise ValueError("Compact state marginals require state_ind")
        state_ind = np.arange(n_bins)
    state_ind = np.asarray(state_ind)
    if (
        state_ind.shape != (n_bins,)
        or state_ind.dtype.kind not in "iu"
        or np.any(state_ind < 0)
    ):
        raise ValueError(
            "state_ind must contain one nonnegative integer per hidden bin"
        )
    minimum_states = int(state_ind.max()) + 1
    n_states = minimum_states if n_states is None else n_states
    if (
        isinstance(n_states, bool)
        or not isinstance(n_states, (int, np.integer))
        or n_states < minimum_states
    ):
        raise ValueError("n_states must include every declared state_ind value")
    n_states = int(n_states)
    missing = (
        np.zeros(n_time, bool)
        if is_missing is None
        else np.asarray(is_missing, dtype=bool)
    )
    if missing.shape != (n_time,):
        raise ValueError("is_missing must contain one value per decode bin")
    if output_mode not in ("compact", "spatial"):
        raise ValueError("output_mode must be compact or spatial")
    if output_mode == "spatial" and result_path is None:
        raise ValueError("Spatial output requires result_path")
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
        raise ValueError("Unrecognized or empty return_outputs")
    if output_mode == "compact" and outputs & _SPATIAL:
        raise ValueError("Compact mode only retains state probabilities")
    state_columns = (
        tuple(
            jnp.asarray(np.flatnonzero(state_ind == state)) for state in range(n_states)
        )
        if outputs & _MARGINAL
        else ()
    )
    selected = _selected_rows(selected_rows, n_time)
    covariate = discrete_transition_matrix is not None
    weights = (
        None if not covariate else np.asarray(discrete_transition_matrix, dtype=dtype)
    )
    if covariate and weights.shape != (n_time, n_states, n_states):
        raise ValueError(
            "Discrete transition rows must match the full recording/state dimensions"
        )
    if transition_operator is None:
        matrix = continuous_transition_matrix if covariate else transition_matrix
        if matrix is None or np.shape(matrix) != (n_bins, n_bins):
            raise ValueError(
                "A matching dense matrix or transition_operator is required"
            )
        transition_operator = _DenseTransition(
            jnp.asarray(matrix, dtype=dtype), jnp.asarray(state_ind)
        )
    else:
        leaves, _ = jax.tree_util.tree_flatten(transition_operator)
        if len(leaves) == 1 and leaves[0] is transition_operator:
            transition_operator = _CallableTransition(transition_operator)
        else:
            # Transfer array leaves once; each jitted chunk call would otherwise
            # convert host leaves again.
            transition_operator = jax.device_put(transition_operator)
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
    arrays = (
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
    diagnostics = {
        "n_degenerate": 0,
        "n_nan": 0,
        "max_chunk_rows": 0,
        "likelihood_evaluations": 0,
        "checkpoint_bytes": 0,
        "checkpoint_cache_entries": 1,
        "global_gradients_supported": False,
    }

    degenerate_rows = np.zeros(n_time, dtype=bool)
    nan_rows = np.zeros(n_time, dtype=bool)

    def likelihood(start, stop):
        """Return device ``(values, degenerate, nan, checksum)`` for one chunk."""
        values = jnp.asarray(
            log_likelihood_func(
                time_edges, row_slice=slice(start, stop), is_missing=missing[start:stop]
            ),
            dtype=dtype,
        )
        if values.shape != (stop - start, n_bins):
            raise ValueError(
                "Likelihood callback must return exactly the requested global rows and hidden bins"
            )
        diagnostics["likelihood_evaluations"] += 1
        diagnostics["max_chunk_rows"] = max(diagnostics["max_chunk_rows"], stop - start)
        return _prepare_chunk(values, jnp.asarray(missing[start:stop]))

    def emit(name, start, stop, values):
        first, last = np.searchsorted(selected, [start, stop])
        if first == last:
            return
        rows = selected[first:last] - start
        values = np.asarray(values)[rows]
        if writer is None:
            arrays[name][first:last] = values
        else:
            writer.write(name, int(first), values)

    def emit_chunk(start, stop, chunk_outputs):
        for name, value in chunk_outputs.items():
            emit(name, start, stop, value)

    try:
        with tempfile.TemporaryDirectory(
            prefix="checkpoints-", dir=checkpoint_parent
        ) as folder:
            folder = Path(folder)
            predicted, evidence = jnp.asarray(initial), jnp.asarray(0, dtype=dtype)

            def finish_forward_chunk(
                start, stop, predicted, evidence, degenerate, nan, checksum, increments
            ):
                """Pull one dispatched chunk's results to the host and checkpoint it."""
                checkpoint = folder / f"{start}.npz"
                np.savez(
                    checkpoint,
                    predicted=np.asarray(predicted),
                    evidence=np.asarray(evidence),
                    checksum=np.asarray(checksum),
                )
                diagnostics["checkpoint_bytes"] += checkpoint.stat().st_size
                degenerate, nan = np.asarray(degenerate), np.asarray(nan)
                degenerate_rows[start:stop] = degenerate
                nan_rows[start:stop] = nan
                diagnostics["n_degenerate"] += int(degenerate.sum())
                diagnostics["n_nan"] += int(nan.sum())
                return (
                    float(np.sum(np.asarray(increments), dtype=np.float64))
                    if evidence_accumulation == "stable"
                    else 0.0
                )

            def forward_increment_sums():
                nonlocal predicted, evidence
                # Each chunk's host work runs after the next chunk is dispatched,
                # so device computation overlaps host transfers and disk writes.
                pending = None
                for start in boundaries:
                    stop = min(start + chunk_size, n_time)
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
                    (evidence, predicted), increments = _forward_chunk(
                        predicted,
                        evidence,
                        values,
                        None if weights is None else jnp.asarray(weights[start:stop]),
                        transition_operator,
                        keep_rows=False,
                        keep_predictive=False,
                        keep_increments=evidence_accumulation == "stable",
                    )
                    del values
                    if pending is not None:
                        yield finish_forward_chunk(*pending)
                    pending = (*chunk, increments)
                if pending is not None:
                    yield finish_forward_chunk(*pending)

            if evidence_accumulation == "stable":
                marginal = _sum_evidence(forward_increment_sums())
            else:
                for _ in forward_increment_sums():
                    pass
                marginal = float(evidence)
            next_smoothed = jnp.asarray(initial)
            # Outputs are emitted after the next chunk is dispatched; each chunk's
            # checksum is verified before its outputs are emitted.
            pending = None
            for start in reversed(boundaries):
                stop = min(start + chunk_size, n_time)
                values, _, _, checksum = likelihood(start, stop)
                with np.load(folder / f"{start}.npz", allow_pickle=False) as checkpoint:
                    expected_checksum = checkpoint["checksum"]
                    prior, start_evidence = (
                        jnp.asarray(checkpoint["predicted"]),
                        jnp.asarray(checkpoint["evidence"]),
                    )
                chunk_weights = (
                    None if weights is None else jnp.asarray(weights[start:stop])
                )
                _, (filtered, predictive) = _forward_chunk(
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
                )
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
                    raise ValueError(
                        "Likelihood callback changed during checkpoint replay; deterministic inputs are required"
                    )
                pending = (start, stop, chunk_outputs)
            if pending is not None:
                emit_chunk(*pending)
        _warn_degenerate_and_nan_timesteps(
            diagnostics["n_degenerate"],
            diagnostics["n_nan"],
            n_time,
            marginal_log_likelihood=marginal,
        )
        diagnostics["degenerate_indices"] = np.flatnonzero(degenerate_rows)
        diagnostics["nan_indices"] = np.flatnonzero(nan_rows)
        packed_masks = {
            "degenerate_row_mask_hex": np.packbits(degenerate_rows, bitorder="little")
            .tobytes()
            .hex(),
            "nan_row_mask_hex": np.packbits(nan_rows, bitorder="little")
            .tobytes()
            .hex(),
            "diagnostic_row_mask_encoding": "packbits_little_hex",
            "diagnostics_are_global": True,
        }
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
            dataset = xr.Dataset(
                {name: (dims[name], arrays[name]) for name in outputs},
                coords=coordinates,
                attrs=attrs,
            )
        else:
            writer.complete(attrs)
            dataset = open_result_store(result_path, max_read_bytes=max_read_bytes)
        return CheckpointedResult(
            dataset,
            marginal,
            diagnostics,
            None if result_path is None else Path(result_path),
        )
    finally:
        if writer is not None:
            writer.abort()
