"""Poisson emission terms of sorted spike counts and the bin-duration evidence."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

from non_local_detector.likelihoods.common.spike_rows import (
    SpikeTimeOrder,
    get_spikecount_per_time_bin,
    prepare_decode_rows,
    spike_counts_matrix,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
)


def log_bin_duration_evidence(
    spike_times: Sequence[np.ndarray | jax.Array],
    time_edges: NDArray[np.float64],
    row_slice: slice | None,
    spike_time_order: "SpikeTimeOrder | None",
    *,
    intensity_time_scale: float = 1.0,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Compute the marked-process event term ``N_bin * log(duration / intensity_time_scale)``.

    ``duration`` is each decode bin's width in seconds; ``intensity_time_scale``
    (seconds, by default 1) is the time unit of the caller's intensities.

    Parameters
    ----------
    spike_times : sequence of np.ndarray, each shape (n_spikes,)
        Decoding spike times for each electrode, in seconds.
    time_edges : np.ndarray, shape (n_bins + 1,)
        FULL decoding bin edges, in seconds.
    row_slice : slice | None
        Contiguous range of decode bins to evaluate, or None for all bins.
    spike_time_order : SpikeTimeOrder | None
        Internal ordering preparation shared across states and chunks; None
        verifies the spike-time ordering on this call.
    intensity_time_scale : float, optional
        Time unit of the caller's intensities, in seconds, by default 1.0.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    evidence : jax.Array, shape (n_rows,)
        ``N_bin * log(duration / intensity_time_scale)`` per requested bin, where
        ``N_bin`` is the number of spikes of all electrodes in the bin.
    """
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, spike_time_order)
    counts = np.zeros(rows.stop - rows.start, dtype=int)
    for times in spike_times:
        counts += get_spikecount_per_time_bin(
            times,
            time_edges=rows.time_edges,
            row_slice=row_slice,
            _spike_time_order=rows.spike_time_order,
        )
    return jnp.asarray(counts) * jnp.log(
        jnp.asarray(rows.durations() / intensity_time_scale)
    )


# Rows per non-local Poisson emission block. Every block has this shape, and a
# row's position in its block is fixed by the decode grid.
_EMISSION_BLOCK_ROWS = 64
# Bytes of one emission block's float32 (rows x bins) product. Blocks are a
# multiple of ``_EMISSION_BLOCK_ROWS`` rows set by the bin count alone, so a
# row keeps its block position for every row range, and large blocks keep the
# product one efficient matrix multiplication.
_EMISSION_BLOCK_BYTES = 4 * 1024**2


def _emission_block_rows(n_bins: int) -> int:
    """Rows per emission block: a multiple of 64 within ``_EMISSION_BLOCK_BYTES``."""
    rows = _EMISSION_BLOCK_BYTES // (4 * max(n_bins, 1))
    return max(
        _EMISSION_BLOCK_ROWS, rows // _EMISSION_BLOCK_ROWS * _EMISSION_BLOCK_ROWS
    )


def _matrix_emission_is_safe(
    rates: jax.Array, durations: jax.Array, accumulator_dtype: Any
) -> jax.Array:
    """Whether the Poisson emission can use the log-rate matrix product.

    Parameters
    ----------
    rates : jax.Array, shape (n_neurons, n_bins)
        Place-field rates.
    durations : jax.Array, shape (n_rows,)
        Only their dtype is used.
    accumulator_dtype : dtype
        The emission's accumulator dtype.

    Returns
    -------
    is_safe : jax.Array, shape ()
        True when every rate is positive and finite and the inputs already
        have the accumulator dtype. No duration value enters, so every row
        range of a decode grid takes the same path.
    """
    return cast(
        jax.Array,
        jnp.all(jnp.isfinite(rates) & (rates > 0))
        & (jnp.result_type(rates, durations) == accumulator_dtype),
    )


@partial(jax.jit, static_argnames=("block_rows",))
def _poisson_nonlocal_log_likelihood(
    counts: jax.Array,
    rates: jax.Array,
    durations: jax.Array,
    summed_rates: jax.Array,
    lead: int = 0,
    *,
    block_rows: int = _EMISSION_BLOCK_ROWS,
) -> jax.Array:
    """Batched Poisson emissions with fixed arithmetic across row partitions.

    A matrix kernel over fixed blocks of ``block_rows`` rows multiplies counts by the log rates, taken
    once per call, and adds each row's count-weighted log duration. Highest
    dot precision excludes a TF32 approximation. Exact zeros and non-finite
    rates retain xlogy.
    Padding is internal and never contributes an observation or returned row.

    Parameters
    ----------
    counts : jax.Array, shape (n_rows, n_neurons)
        Spike count of each neuron in each row.
    rates : jax.Array, shape (n_neurons, n_bins)
        Place-field rates.
    durations : jax.Array, shape (n_rows,)
        Bin durations.
    summed_rates : jax.Array, shape (n_bins,)
        Population rate summed over neurons.
    lead : int, optional
        Rows of the first block before the first row, in ``[0, block_rows)``;
        0 by default. Passing the first row's decode-grid index modulo
        ``block_rows`` keeps each row at one position in one block shape,
        whatever row range is requested.
    block_rows : int, optional, keyword-only
        Rows per emission block, by default 64 (see ``_emission_block_rows``).

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_bins)
    """
    n_rows, n_neurons = counts.shape
    n_bins = rates.shape[1]
    accumulator_dtype = jnp.result_type(rates, durations, jnp.zeros(()))
    if not n_rows or not n_neurons or not n_bins:
        return (
            jnp.zeros((n_rows, n_bins), dtype=accumulator_dtype)
            - durations[:, None] * summed_rates
        )
    # Enough blocks for any ``lead``, so the shape depends only on n_rows.
    n_blocks = (n_rows + 2 * block_rows - 2) // block_rows
    # Block rows outside the request have no counts and a unit duration.
    row_durations = jax.lax.dynamic_update_slice(
        jnp.ones(n_blocks * block_rows, dtype=durations.dtype), durations, (lead,)
    )
    matrix_is_safe = _matrix_emission_is_safe(rates, durations, accumulator_dtype)
    matrix_dtype = jnp.result_type(rates, durations)
    # Once per call rather than per block. Unsafe rates (which never take the
    # matrix path) are replaced so their unused log has a finite derivative.
    usable_rates = jnp.isfinite(rates) & (rates > 0)
    log_rates = jnp.log(jnp.where(usable_rates, rates, 1).astype(matrix_dtype))

    def emit_block(block_counts: jax.Array, block_durations: jax.Array) -> jax.Array:
        def matrix_accumulation(_: None) -> jax.Array:
            block_counts_float = block_counts.astype(matrix_dtype)
            n_spikes = block_counts_float.sum(axis=1, keepdims=True)
            # log(rate * duration) = log(rate) + log(duration): one product
            # with the per-call log rates, plus each row's own duration term,
            # which is 0 for a row without spikes (as xlogy gives).
            duration_term = jnp.where(
                n_spikes == 0,
                0,
                n_spikes * jnp.log(block_durations.astype(matrix_dtype))[:, None],
            )
            values = (
                jnp.matmul(
                    block_counts_float, log_rates, precision=jax.lax.Precision.HIGHEST
                )
                + duration_term
            )
            return values.astype(accumulator_dtype)

        def xlogy_accumulation(_: None) -> jax.Array:
            def add_neuron(
                likelihood: jax.Array, neuron: tuple[jax.Array, jax.Array]
            ) -> tuple[jax.Array, None]:
                neuron_counts, neuron_rates = neuron
                product = jax.lax.optimization_barrier(  # type: ignore[no-untyped-call]  # untyped in JAX
                    neuron_rates[None, :] * block_durations[:, None]
                )
                # Floating counts give integer inputs a zero tangent instead of
                # float0, so rate and duration derivatives pass through.
                contribution = jax.lax.optimization_barrier(  # type: ignore[no-untyped-call]  # untyped in JAX
                    jax.scipy.special.xlogy(
                        neuron_counts.astype(jnp.result_type(neuron_counts, product))[
                            :, None
                        ],
                        product,
                    )
                )
                return jax.lax.optimization_barrier(likelihood + contribution), None  # type: ignore[no-untyped-call]  # untyped in JAX

            likelihood, _ = jax.lax.scan(
                add_neuron,
                jnp.zeros((block_rows, n_bins), dtype=accumulator_dtype),
                (block_counts.T, rates),
            )
            return likelihood

        return cast(
            jax.Array,
            jax.lax.cond(matrix_is_safe, matrix_accumulation, xlogy_accumulation, None),
        )

    def durations_at(first: jax.Array | int) -> jax.Array:
        return jax.lax.dynamic_slice(row_durations, (first,), (block_rows,))

    def update_edge_block(number: int, result: jax.Array) -> jax.Array:
        first = number * block_rows
        # Indexing the requested rows directly, with zero counts outside them
        # and those rows dropped from the result, copies neither the counts
        # nor the result. Negative rows would otherwise wrap.
        rows = first - lead + jnp.arange(block_rows)
        rows = jnp.where(rows < 0, n_rows, rows)
        block = emit_block(
            counts.at[rows].get(mode="fill", fill_value=0), durations_at(first)
        )
        return result.at[rows].set(block, mode="drop")

    def update_interior_block(number: jax.Array, result: jax.Array) -> jax.Array:
        first = number * block_rows
        block = emit_block(
            jax.lax.dynamic_slice(counts, (first - lead, 0), (block_rows, n_neurons)),
            durations_at(first),
        )
        return jax.lax.dynamic_update_slice(result, block, (first - lead, 0))

    # Blocks 1 through n_blocks - 3 lie within the rows for any ``lead``; only
    # the first and last two can extend past them.
    likelihood = jnp.zeros((n_rows, n_bins), dtype=accumulator_dtype)
    for number in sorted({0, max(n_blocks - 2, 0), n_blocks - 1}):
        likelihood = update_edge_block(number, likelihood)
    if n_blocks > 3:
        likelihood = jax.lax.fori_loop(
            1, n_blocks - 2, update_interior_block, likelihood
        )
    return likelihood - durations[:, None] * summed_rates


@jax.jit
def poisson_row_log_likelihood(counts: Any, expected_counts: Any) -> jax.Array:
    """Retain neuron addition order inside one compiled row-bounded scan."""
    expected_counts = jnp.broadcast_to(expected_counts, counts.shape)
    # xlogy's forward pass already promotes integer counts to this dtype.
    # Explicit promotion also avoids float0 tangents in its rate derivative.
    counts = counts.astype(expected_counts.dtype)

    def add_unit(total: jax.Array, inputs: tuple[Any, Any]) -> tuple[jax.Array, None]:
        count, expected = inputs
        safe_expected = jnp.where(count == 0, 1.0, expected)
        event = jax.lax.optimization_barrier(  # type: ignore[no-untyped-call]  # jax.lax.optimization_barrier is untyped
            jax.scipy.special.xlogy(count, safe_expected)
        )
        term = jax.lax.optimization_barrier(event - expected)  # type: ignore[no-untyped-call]  # jax.lax.optimization_barrier is untyped
        return total + term, None

    return jax.lax.scan(
        add_unit, jnp.zeros((counts.shape[0],)), (counts.T, expected_counts.T)
    )[0]


# Host bytes for one block of int64 spike counts, so full-grid requests never
# build a rows-by-population count matrix. The non-local emission blocks are
# fixed to the decode grid, so blocked and unblocked requests are
# bitwise identical.
COUNT_BLOCK_BYTES = 32 * 1024**2


def _count_block_rows(n_neurons: int) -> int:
    """Rows per spike-count block within ``COUNT_BLOCK_BYTES``, a multiple of 64."""
    return max(64, COUNT_BLOCK_BYTES // (8 * max(n_neurons, 1)) // 64 * 64)


def concatenate_row_blocks(
    row_start: int,
    row_stop: int,
    n_neurons: int,
    evaluate: Callable[[int, int], jax.Array],
    desc: str,
    disable_progress_bar: bool,
) -> jax.Array:
    """Concatenate ``evaluate(start, stop)`` over spike-count row blocks.

    Parameters
    ----------
    row_start, row_stop : int
        Global rows to evaluate.
    n_neurons : int
        Population size, which sets the block rows.
    evaluate : callable
        ``evaluate(start, stop)`` returns the result for global rows
        ``[start, stop)``, rows on the first axis.
    desc : str
        Progress bar label.
    disable_progress_bar : bool
        Whether to hide the progress bar.

    Returns
    -------
    result : jax.Array, shape (row_stop - row_start, ...)
        One-dimensional zeros when no rows are requested.
    """
    block_rows = _count_block_rows(n_neurons)
    blocks = [
        evaluate(start, min(start + block_rows, row_stop))
        for start in tqdm(
            range(row_start, row_stop, block_rows),
            unit="block",
            desc=desc,
            disable=disable_progress_bar,
        )
    ]
    if not blocks:
        return jnp.zeros((0,))
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks)


def blocked_nonlocal_poisson_log_likelihood(
    spike_times: list[np.ndarray],
    time_edges: np.ndarray,
    row_start: int,
    row_stop: int,
    rates: jax.Array,
    durations: jax.Array,
    summed_rates: jax.Array,
    *,
    disable_progress_bar: bool = True,
    _spike_time_order: "SpikeTimeOrder | None" = None,
) -> jax.Array:
    """Non-local Poisson emission computed over bounded host count blocks.

    Parameters
    ----------
    spike_times : list of np.ndarray
        One spike-time array per neuron.
    time_edges : np.ndarray, shape (n_time_bins + 1,)
        The complete decode grid.
    row_start, row_stop : int
        Global rows to evaluate.
    rates : jax.Array, shape (n_neurons, n_bins)
        Interior place fields in Hz.
    durations : jax.Array, shape (row_stop - row_start,)
        Bin durations in seconds for the requested rows.
    summed_rates : jax.Array, shape (n_bins,)
        Population rate summed over neurons.
    disable_progress_bar : bool
        Whether to hide the progress bar.
    _spike_time_order : SpikeTimeOrder, optional
        Shared per-prediction spike ordering cache.

    Returns
    -------
    log_likelihood : jax.Array, shape (row_stop - row_start, n_bins)

    Notes
    -----
    Host counts are bounded by ``COUNT_BLOCK_BYTES``. Concatenating
    the block outputs briefly holds about twice the returned array. The
    progress bar advances per block rather than per neuron.
    """
    if _spike_time_order is None:
        # Verify each neuron's spike ordering once, not once per block.
        _spike_time_order = SpikeTimeOrder()
    block_rows = _emission_block_rows(rates.shape[1])
    if row_start == row_stop:
        return cast(
            jax.Array,
            _poisson_nonlocal_log_likelihood(
                jnp.zeros((0, len(spike_times))), rates, durations, summed_rates
            ),
        )

    def evaluate(start: int, stop: int) -> jax.Array:
        counts = spike_counts_matrix(
            spike_times,
            time_edges,
            "Position-Bin Likelihood",
            True,
            slice(start, stop),
            _spike_time_order=_spike_time_order,
        )
        # Emission blocks start at multiples of block_rows decode rows, so a
        # row sits at the same position of the same block shape in every row
        # range.
        return cast(
            jax.Array,
            _poisson_nonlocal_log_likelihood(
                jnp.asarray(counts),
                rates,
                durations[start - row_start : stop - row_start],
                summed_rates,
                start % block_rows,
                block_rows=block_rows,
            ),
        )

    return concatenate_row_blocks(
        row_start,
        row_stop,
        len(spike_times),
        evaluate,
        "Position-Bin Likelihood",
        disable_progress_bar,
    )
