from collections.abc import Sequence, Sized
from dataclasses import dataclass, field
from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from jax.nn import logsumexp
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
    validate_time_edges,
)

# JAX exposes this exception only privately. Older supported releases predate
# explicit sharding and may lack the type; an empty tuple catches nothing.
try:
    from jax._src.core import ShardingTypeError as _JaxShardingTypeError
except ImportError:
    _JAX_SHARDING_ERRORS: tuple[type[Exception], ...] = ()
else:
    _JAX_SHARDING_ERRORS = (_JaxShardingTypeError,)

EPS = 1e-15
LOG_EPS = np.log(EPS)
RATE_REFERENCE_SECONDS = 0.002
LOG_RATE_EPS_HZ = np.log(EPS / RATE_REFERENCE_SECONDS)
RATE_EPS_HZ = (
    EPS / RATE_REFERENCE_SECONDS
)  # Historical safeguard at the 2 ms reference, now Hz.


def log_bin_duration_evidence(
    spike_times,
    time_edges,
    row_slice,
    spike_time_order,
    *,
    intensity_time_scale=1.0,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """The marked-process event term ``N_bin * log(duration_seconds)``."""
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if spike_time_order is None:
        spike_time_order = _SpikeTimeOrder()
    start, stop = resolve_row_slice(row_slice, len(time_edges) - 1)
    counts = np.zeros(stop - start, dtype=int)
    for times in spike_times:
        counts += get_spikecount_per_time_bin(
            times,
            time_edges=time_edges,
            row_slice=row_slice,
            _spike_time_order=spike_time_order,
        )
    return jnp.asarray(counts) * jnp.log(
        jnp.asarray(_time_grid.durations(start, stop) / intensity_time_scale)
    )


def as_std_array(std: "jnp.ndarray | float | int", n_dims: int) -> jnp.ndarray:
    """Broadcast a scalar kernel bandwidth to a per-dimension array.

    A single ``jnp.ndim`` check covers Python ``int``/``float``, NumPy scalars
    (including ``np.float32``), 0-d arrays, and JAX scalars uniformly, avoiding
    the ``isinstance(std, int | float)`` pattern that silently misses
    ``np.float32`` and 0-d arrays. Array-valued ``std`` (``ndim >= 1``) is
    returned via ``jnp.asarray`` (a no-op for JAX arrays; lists/NumPy arrays are
    converted to a JAX array).

    Parameters
    ----------
    std : jnp.ndarray or float or int
        Scalar (broadcast to ``n_dims``) or per-dimension standard deviation(s).
    n_dims : int
        Number of dimensions to broadcast a scalar to.

    Returns
    -------
    std_array : jnp.ndarray
        Shape ``(n_dims,)`` for scalar input; the original array otherwise.
    """
    if jnp.ndim(std) == 0:
        return jnp.full((n_dims,), std)
    return jnp.asarray(std)


def validate_weights(weights: np.ndarray, n_time: int) -> np.ndarray:
    """Validate per-sample weights shared by the sorted-spikes and clusterless fits.

    Weights feed both the occupancy field and the spike counts, so an invalid
    array corrupts the whole encoding silently; validate once at the fit entry.

    Parameters
    ----------
    weights : np.ndarray, shape (n_time,)
        Per-sample weights (e.g. posterior state probabilities during EM).
    n_time : int
        Expected number of samples (``position.shape[0]``).

    Returns
    -------
    weights : np.ndarray, shape (n_time,)
        The weights as a float array.

    Raises
    ------
    ValidationError
        If ``weights`` is not 1-D of length ``n_time``, or contains non-finite or
        negative values.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.shape != (n_time,):
        raise ValidationError(
            f"weights must have shape ({n_time},), got {weights.shape}"
        )

    if not np.all(np.isfinite(weights)):
        raise ValidationError("weights must contain only finite values")

    if np.any(weights < 0):
        raise ValidationError("weights must be non-negative")

    return weights


def validate_finite(array: np.ndarray | jnp.ndarray, name: str) -> None:
    """Raise if ``array`` contains any non-finite (NaN/inf) value.

    Shared fit-input finiteness check for the encoding fits: a non-finite
    position or spike waveform feature is invalid encoding data that would
    otherwise propagate into the density estimate as a silent NaN. Fail fast at
    the fit entry, consistent with ``validate_weights`` and the GMM/MRF fits
    (which already reject non-finite inputs).

    Parameters
    ----------
    array : np.ndarray | jnp.ndarray
        The array to check.
    name : str
        Name used in the error message (e.g. ``"position"``).

    Raises
    ------
    ValidationError
        If ``array`` contains any non-finite value.
    """
    # Reduce on the array's own device: for a JAX array this keeps the full
    # (potentially GPU-resident) array on-device and transfers only the scalar
    # result, avoiding a full device->host copy per electrode on every refit.
    if isinstance(array, jnp.ndarray):
        all_finite = bool(jnp.all(jnp.isfinite(array)))
    else:
        all_finite = bool(np.all(np.isfinite(np.asarray(array))))
    if not all_finite:
        raise ValidationError(f"{name} must contain only finite values")


def validate_population_lengths(unit_name: str, **populations: Sized | None) -> int:
    """Require parallel population collections to contain the same number of units.

    Likelihood predictors combine observed spike trains with fitted per-unit models.
    Plain ``zip`` silently truncates when one collection is shorter, producing a
    plausible likelihood from only part of the recorded population. Validate the
    collections once at the entry point and return the common length.

    Parameters
    ----------
    unit_name : str
        Human-readable population unit, such as ``"electrode"`` or ``"neuron"``.
    **populations
        Named sized collections that should be parallel. ``None`` marks an
        optional collection that was not supplied and is skipped.

    Returns
    -------
    n_units : int
        Shared collection length, or 0 when no collections are supplied.

    Raises
    ------
    ValidationError
        If the collection lengths differ.
    """
    lengths = {
        name: len(values) for name, values in populations.items() if values is not None
    }
    if not lengths:
        return 0
    expected = next(iter(lengths.values()))
    if any(length != expected for length in lengths.values()):
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValidationError(
            f"{unit_name} population lengths do not match",
            expected=f"all per-{unit_name} collections to have length {expected}",
            got=details,
            hint=(
                f"Provide one entry in every collection for each {unit_name}; do not "
                "drop empty units."
            ),
        )
    return expected


def validate_spike_feature_pair(
    spike_times: np.ndarray | jnp.ndarray,
    spike_features: np.ndarray | jnp.ndarray,
    electrode: int,
) -> None:
    """Validate one electrode's parallel spike/mark arrays without copying them.

    Parameters
    ----------
    spike_times : np.ndarray or jnp.ndarray, shape (n_spikes,)
    spike_features : np.ndarray or jnp.ndarray, shape (n_spikes, n_features)
    electrode : int
        Electrode index, used in the error message.

    Raises
    ------
    ValidationError
        If either array has the wrong rank or their row counts differ.
    """
    times_shape = np.shape(spike_times)
    features_shape = np.shape(spike_features)
    if len(times_shape) != 1:
        raise ValidationError(
            f"spike_times for electrode {electrode} must be 1-D",
            expected="shape (n_spikes,)",
            got=f"shape {times_shape}",
        )
    if len(features_shape) != 2:
        raise ValidationError(
            f"spike_waveform_features for electrode {electrode} must be 2-D",
            expected="shape (n_spikes, n_features)",
            got=f"shape {features_shape}",
        )
    if features_shape[0] != times_shape[0]:
        raise ValidationError(
            f"spike times and waveform features disagree for electrode {electrode}",
            expected=f"{times_shape[0]} waveform-feature rows",
            got=f"{features_shape[0]} rows",
            hint="Provide exactly one waveform-feature row for every spike time.",
        )


def validate_spike_feature_population(
    spike_times: Sequence[np.ndarray | jnp.ndarray],
    spike_waveform_features: Sequence[np.ndarray | jnp.ndarray],
) -> int:
    """Require clusterless spike times and features to describe the same electrodes.

    Checks the electrode count and each electrode's spike/feature row alignment
    before anything pairs the collections, so a mismatch cannot silently drop an
    electrode or misalign features with spike times.

    Parameters
    ----------
    spike_times : sequence of arrays, each shape (n_spikes,)
    spike_waveform_features : sequence of arrays, each shape (n_spikes, n_features)

    Returns
    -------
    n_electrodes : int

    Raises
    ------
    ValidationError
        If the electrode counts differ or any electrode's arrays disagree.
    """
    n_electrodes = validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
    )
    for electrode, (times, features) in enumerate(
        zip(spike_times, spike_waveform_features, strict=True)
    ):
        validate_spike_feature_pair(times, features, electrode)
    return n_electrodes


def interpolate_weights_at_spike_times(
    spike_times: np.ndarray,
    position_time: np.ndarray,
    weights: np.ndarray,
    *,
    encoding_support=None,
) -> np.ndarray:
    """Per-spike weights: the per-sample ``weights`` linearly interpolated onto times.

    Spike times are assumed already clipped to ``[position_time[0], position_time[-1]]``,
    so 1-D ``np.interp`` matches ``interpn`` without building an interpolator (and does
    not extrapolate). Shared by the clusterless and sorted-spikes encoding fits.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    position_time : np.ndarray, shape (n_time_position,)
    weights : np.ndarray, shape (n_time_position,)

    Returns
    -------
    spike_weights : np.ndarray, shape (n_spikes,)
    """
    if encoding_support is not None:
        return encoding_support.interpolate(weights, spike_times, fill_value=0.0)
    return np.interp(
        np.asarray(spike_times), np.asarray(position_time), np.asarray(weights)
    )


def weighted_mean_rate(spike_weights: np.ndarray, weight_sum: float) -> float:
    """Weighted mean firing rate: weighted spike count / weighted occupancy time.

    Returns 0.0 when ``weight_sum`` is not positive (no effective training data), so a
    zero-weight electrode/group contributes nothing rather than dividing by zero.

    Parameters
    ----------
    spike_weights : np.ndarray, shape (n_spikes,)
        Per-spike weights (see :func:`interpolate_weights_at_spike_times`).
    weight_sum : float
        Sum of weighted sample exposure in seconds (not sample count).

    Returns
    -------
    mean_rate : float
    """
    return float(np.sum(spike_weights) / weight_sum) if weight_sum > 0 else 0.0


@jax.jit
def _poisson_nonlocal_log_likelihood(
    counts: jnp.ndarray,
    rates: jnp.ndarray,
    durations: jnp.ndarray,
    summed_rates: jnp.ndarray,
) -> jnp.ndarray:
    """Batched Poisson emissions with fixed arithmetic across row partitions.

    A fixed 64-row matrix kernel uses the same rate-duration products as the
    per-neuron reference. Highest dot precision excludes a TF32 approximation.
    Exact zeros, non-finite products and nonuniform durations retain xlogy.
    Padding is internal and never contributes an observation or returned row.
    """
    n_rows, n_neurons = counts.shape
    n_bins = rates.shape[1]
    accumulator_dtype = jnp.result_type(rates, durations, jnp.zeros(()))
    if not n_rows or not n_neurons or not n_bins:
        return (
            jnp.zeros((n_rows, n_bins), dtype=accumulator_dtype)
            - durations[:, None] * summed_rates
        )
    block_rows = 64
    n_full, remainder = divmod(n_rows, block_rows)
    padded_rows = n_rows + (-n_rows) % block_rows
    padded_counts = jnp.pad(counts, ((0, padded_rows - n_rows), (0, 0)))
    padded_durations = jnp.pad(durations, (0, padded_rows - n_rows), mode="edge")

    def emit_block(block_counts, block_durations):
        smallest = jnp.min(rates) * jnp.min(block_durations)
        largest = jnp.max(rates) * jnp.max(block_durations)
        matrix_is_safe = (
            jnp.all(jnp.isfinite(rates))
            & jnp.all(jnp.isfinite(block_durations))
            & jnp.all(block_durations == block_durations[0])
            & (block_durations[0] > 0)
            & (smallest > 0)
            & jnp.isfinite(largest)
            & (jnp.result_type(rates, block_durations) == accumulator_dtype)
        )

        def matrix_accumulation(_):
            dtype = jnp.result_type(rates, block_durations)
            block_counts_float = block_counts.astype(dtype)
            # Values use the block's common duration. The zero-valued term
            # (exactly 0 for these finite durations) restores each row's own
            # duration derivative, which the shared duration would collapse.
            log_durations = jnp.log(block_durations.astype(dtype))
            duration_tangent = (
                block_counts_float.sum(axis=1, keepdims=True)
                * (log_durations - jax.lax.stop_gradient(log_durations))[:, None]
            )
            return (
                jnp.matmul(
                    block_counts_float,
                    jnp.log(
                        rates.astype(dtype) * jax.lax.stop_gradient(block_durations[0])
                    ),
                    precision=jax.lax.Precision.HIGHEST,
                )
                + duration_tangent
            ).astype(accumulator_dtype)

        def xlogy_accumulation(_):
            def add_neuron(likelihood, neuron):
                neuron_counts, neuron_rates = neuron
                product = jax.lax.optimization_barrier(
                    neuron_rates[None, :] * block_durations[:, None]
                )
                # Floating counts give integer inputs a zero tangent instead of
                # float0, so rate and duration derivatives pass through.
                contribution = jax.lax.optimization_barrier(
                    jax.scipy.special.xlogy(
                        neuron_counts.astype(jnp.result_type(neuron_counts, product))[
                            :, None
                        ],
                        product,
                    )
                )
                return jax.lax.optimization_barrier(likelihood + contribution), None

            likelihood, _ = jax.lax.scan(
                add_neuron,
                jnp.zeros((block_rows, n_bins), dtype=accumulator_dtype),
                (block_counts.T, rates),
            )
            return likelihood

        return jax.lax.cond(
            matrix_is_safe, matrix_accumulation, xlogy_accumulation, None
        )

    likelihood = jnp.zeros((n_rows, n_bins), dtype=accumulator_dtype)
    if n_full:

        def update_block(number, result):
            first = number * block_rows
            block_counts = jax.lax.dynamic_slice(
                padded_counts, (first, 0), (block_rows, n_neurons)
            )
            block_durations = jax.lax.dynamic_slice(
                padded_durations, (first,), (block_rows,)
            )
            return jax.lax.dynamic_update_slice(
                result, emit_block(block_counts, block_durations), (first, 0)
            )

        likelihood = jax.lax.fori_loop(0, n_full, update_block, likelihood)
    if remainder:
        first = n_full * block_rows
        tail = emit_block(padded_counts[first:], padded_durations[first:])[:remainder]
        likelihood = likelihood.at[first:].set(tail)
    return likelihood - durations[:, None] * summed_rates


# Requests above this many rows use per-neuron accumulation, so full-grid
# callers never build a rows-by-population count buffer for local emissions.
COMPILED_ROW_LIMIT = 256

# Host bytes for one block of int64 spike counts. Block rows are a multiple of
# the emission's fixed 64-row kernel, so blocked and unblocked requests are
# bitwise identical.
NONLOCAL_COUNT_BLOCK_BYTES = 32 * 1024**2


def _blocked_nonlocal_poisson_log_likelihood(
    spike_times: list[np.ndarray],
    time_edges: np.ndarray,
    row_start: int,
    row_stop: int,
    rates: jnp.ndarray,
    durations: jnp.ndarray,
    summed_rates: jnp.ndarray,
    *,
    disable_progress_bar: bool = True,
    _spike_time_order: "_SpikeTimeOrder | None" = None,
) -> jnp.ndarray:
    """Non-local Poisson emission computed over bounded host count blocks.

    Parameters
    ----------
    spike_times : list of np.ndarray
        One spike-time array per neuron.
    time_edges : np.ndarray, shape (n_time_bins + 1,)
        The complete decode grid.
    row_start, row_stop : int
        Global rows to evaluate.
    rates : jnp.ndarray, shape (n_neurons, n_bins)
        Interior place fields in Hz.
    durations : jnp.ndarray, shape (row_stop - row_start,)
        Bin durations in seconds for the requested rows.
    summed_rates : jnp.ndarray, shape (n_bins,)
        Population rate summed over neurons.
    disable_progress_bar : bool
    _spike_time_order : _SpikeTimeOrder, optional
        Shared per-prediction spike ordering cache.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (row_stop - row_start, n_bins)

    Notes
    -----
    Host counts are bounded by ``NONLOCAL_COUNT_BLOCK_BYTES``. Concatenating
    the block outputs briefly holds about twice the returned array. The
    progress bar advances per block rather than per neuron.
    """
    if _spike_time_order is None:
        # Verify each neuron's spike ordering once, not once per block.
        _spike_time_order = _SpikeTimeOrder()
    block_rows = max(
        64, NONLOCAL_COUNT_BLOCK_BYTES // (8 * max(len(spike_times), 1)) // 64 * 64
    )
    blocks = []
    for start in tqdm(
        range(row_start, row_stop, block_rows),
        unit="block",
        desc="Non-Local Likelihood",
        disable=disable_progress_bar,
    ):
        stop = min(start + block_rows, row_stop)
        counts = _spike_counts_matrix(
            spike_times,
            time_edges,
            "Non-Local Likelihood",
            True,
            slice(start, stop),
            _spike_time_order=_spike_time_order,
        )
        blocks.append(
            _poisson_nonlocal_log_likelihood(
                jnp.asarray(counts),
                rates,
                durations[start - row_start : stop - row_start],
                summed_rates,
            )
        )
    if not blocks:
        return _poisson_nonlocal_log_likelihood(
            jnp.zeros((0, len(spike_times))), rates, durations, summed_rates
        )
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks)


def get_position_at_time(
    time: np.ndarray | jnp.ndarray,
    position: jnp.ndarray,
    spike_times: np.ndarray | jnp.ndarray,
    env: Environment | None = None,
    *,
    encoding_support=None,
) -> np.ndarray:
    """Get the position at the time of each spike.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
    position : jnp.ndarray, shape (n_time_position, n_dims_position)
    spike_times : jnp.ndarray, shape (n_spikes,)
    env : Environment | None, optional
        The spatial environment, by default None

    Returns
    -------
    position_at_spike_times : np.ndarray, shape (n_spikes, n_dims_position)
    """
    if encoding_support is None:
        position_at_spike_times = scipy.interpolate.interpn(
            (time,), position, spike_times, bounds_error=False, fill_value=None
        )
    else:
        position_at_spike_times = encoding_support.interpolate(position, spike_times)
    if env is not None and env.track_graph is not None:
        if position_at_spike_times.shape[0] > 0:
            position_at_spike_times = get_linearized_position(
                position_at_spike_times,
                env.track_graph,
                edge_order=env.edge_order,
                edge_spacing=env.edge_spacing,
            ).linear_position.to_numpy()[:, None]
        else:
            position_at_spike_times = np.array([])[:, None]

    return position_at_spike_times


def resolve_row_slice(row_slice: slice | None, n_bins: int) -> tuple[int, int]:
    """Normalize a likelihood row request against the full decoding timeline.

    A backend's ``row_slice`` selects which rows of the FULL-time likelihood to
    return; ``time_edges`` always stay the full decoding edges so that
    spike-to-row ownership is independent of how the rows were chunked.

    Parameters
    ----------
    row_slice : slice | None
        Contiguous (unit-step) range of output rows, or None for all rows.
    n_bins : int
        Number of decode bins, ``len(time_edges) - 1``.

    Returns
    -------
    row_start : int
    row_stop : int
        Half-open row range with ``0 <= row_start <= row_stop <= n_bins``.
    """
    if row_slice is None:
        return 0, n_bins
    if row_slice.step not in (None, 1):
        raise ValidationError(
            "row_slice must select a contiguous range of rows",
            expected="a slice with step None or 1",
            got=f"step={row_slice.step}",
            hint="Chunked prediction only requests contiguous row ranges.",
            example="    predict_..._log_likelihood(time_edges, ..., row_slice=slice(0, 100))",
        )
    row_start, row_stop, _ = row_slice.indices(n_bins)
    return row_start, max(row_start, row_stop)


def _spikes_are_ascending(spike_times: np.ndarray) -> bool:
    """Verify non-decreasing times without sorting or changing precision."""
    return spike_times.size < 2 or bool(np.all(spike_times[1:] >= spike_times[:-1]))


class _SpikeTimeOrder:
    """Host spike times and verified ordering, scoped to one prediction.

    Each original input is converted and checked on first use, then reused by
    every observation state and chunk. Strong references prevent identity reuse.
    Inputs must remain unchanged during the prediction; a fresh instance on the
    next call rechecks even arrays modified in place. Never store this object on
    a detector or pass it into a JIT kernel.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[object, np.ndarray, bool]] = {}

    def get(self, spike_times) -> tuple[np.ndarray, bool]:
        """Return the original input's host values and established ordering."""
        key = id(spike_times)
        if key not in self._entries:
            values = np.asarray(spike_times)
            self._entries[key] = (spike_times, values, _spikes_are_ascending(values))
        _, values, ascending = self._entries[key]
        return values, ascending


@dataclass(frozen=True)
class SpikeSelection:
    """Host metadata for spikes belonging to one likelihood row range.

    ``n_spikes`` is the original train length, used to validate paired arrays
    before slicing. ``n_rows`` is the requested row count, the segment count
    that pairs with the LOCAL rows in ``bin_ind``; ``sum_spikes_into_rows``
    reads it from here so a reduction cannot be paired with another request's
    row count. ``indices_are_sorted`` records the verified ordering of
    ``bin_ind`` independently of how ``indexer`` represents the selection.
    This object stays outside JIT kernels.
    """

    indexer: slice | np.ndarray
    bin_ind: np.ndarray
    indices_are_sorted: bool
    n_spikes: int
    n_rows: int


@requires_time_edges
def select_spikes_in_rows(
    spike_times: np.ndarray,
    row_start: int,
    row_stop: int,
    *,
    time_edges: np.ndarray,
    _spike_time_order: _SpikeTimeOrder | None = None,
) -> SpikeSelection:
    """Select the spikes owned by the decode bins ``[row_start, row_stop)``.

    Bin ``i`` owns ``time_edges[i] <= t < time_edges[i + 1]``; the final bin
    also owns ``t == time_edges[-1]``. Spikes outside
    ``[time_edges[0], time_edges[-1]]`` belong to no bin.

    Owning a bin ``>= row_start`` is exactly ``t >= time_edges[row_start]`` and
    owning a bin ``< row_stop`` is exactly ``t < time_edges[row_stop]``
    (inclusive when ``row_stop == n_bins``, i.e. the range reaches the final
    bin), so the selection is a
    contiguous range in time, found with ``np.searchsorted`` when the spike
    times are ascending. Only the selected subset is binned, so no spike is
    re-binned for every chunk.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Decoding spike times for one neuron/electrode.
    time_edges : np.ndarray, shape (n_bins + 1,)
        FULL decoding bin edges (not the chunk's).
    row_start : int
    row_stop : int
        Half-open range of requested bins, as returned by ``resolve_row_slice``.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal prediction-local preparation shared across states and chunks.
        Direct callers omit it and verify ordering on each call.

    Returns
    -------
    selection : SpikeSelection
        ``indexer`` selects owned spikes in their original order and is applied
        identically to waveform features by ``select_spike_rows``. ``bin_ind``
        contains LOCAL row indices (``global_row - row_start``) and ``n_rows``
        their segment count, ``row_stop - row_start``. ``indices_are_sorted`` is the
        verified JAX reduction hint; callers must use it rather than infer
        ordering from the indexer's type. ``n_spikes`` records the original
        length so paired arrays can be validated even for empty requests.
    """
    time_edges = (
        validate_time_edges(time_edges)
        if _spike_time_order is None
        else np.asarray(time_edges)
    )
    n_bins = time_edges.shape[0] - 1
    n_spikes = len(spike_times)
    n_rows = row_stop - row_start
    if n_spikes == 0 or n_rows <= 0:
        return SpikeSelection(
            slice(0, 0), np.zeros((0,), dtype=int), True, n_spikes, n_rows
        )

    # Direct callers get a fresh preparation: converted and checked once here.
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    spike_times, is_ascending = _spike_time_order.get(spike_times)

    lower = time_edges[row_start]
    upper = time_edges[row_stop]
    # The final bin is closed so a spike at the last edge is counted.
    upper_is_inclusive = row_stop == n_bins

    # The ordering the range lookup needs was established, never assumed.
    if is_ascending:
        start = int(np.searchsorted(spike_times, lower, side="left"))
        stop = int(
            np.searchsorted(
                spike_times, upper, side="right" if upper_is_inclusive else "left"
            )
        )
        indexer: slice | np.ndarray = slice(start, max(start, stop))
        selected = spike_times[indexer]
    else:
        in_rows = spike_times >= lower
        in_rows &= spike_times <= upper if upper_is_inclusive else spike_times < upper
        indexer = in_rows
        selected = spike_times[in_rows]

    # Selection above establishes global ownership. Only the interior edges of
    # the requested bins are needed for the local bin index; the full edge
    # array would make digitize's monotonicity check scan the whole recording
    # for every unit and every chunk. A spike at the closing edge of the final
    # bin lands past the last interior edge, in the final local bin.
    interior_edges = time_edges[row_start + 1 : row_stop]
    return SpikeSelection(
        indexer, np.digitize(selected, interior_edges), is_ascending, n_spikes, n_rows
    )


def _segmented_add(left, right):
    """Associative segmented sum: a segment start in ``right`` resets the total."""
    left_start, left_value = left
    right_start, right_value = right
    keep = right_start.reshape(
        right_start.shape + (1,) * (right_value.ndim - right_start.ndim)
    )
    return left_start | right_start, jnp.where(
        keep, right_value, left_value + right_value
    )


@partial(jax.jit, static_argnames=("num_segments", "indices_are_sorted"))
def deterministic_segment_sum(
    values: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    indices_are_sorted: bool = False,
) -> jnp.ndarray:
    """Sum rows into segments with a fixed reduction tree and no scatters.

    Parameters
    ----------
    values : jnp.ndarray, shape (n, ...)
    segment_ids : jnp.ndarray, shape (n,), integer
        Ids outside ``[0, num_segments)`` are dropped, including unsigned or
        64-bit ids too large for the canonical index dtype.
    num_segments : int
        Must be below 2**31.
    indices_are_sorted : bool
        True only when ``segment_ids`` is verified nondecreasing.

    Returns
    -------
    sums : jnp.ndarray, shape (num_segments, ...)
        Repeated evaluation of the same shapes is bitwise identical on every
        backend; a different ``n`` changes the tree and can change rounding.

    Notes
    -----
    The scan keeps ``O(n)`` temporaries the size of ``values`` (about 1.5x
    for sorted ids and 2.5x unsorted on XLA:CPU), unlike a scatter.
    """
    n = values.shape[0]
    shape = (num_segments, *values.shape[1:])
    if n == 0 or num_segments == 0:
        return jnp.zeros(shape, values.dtype)
    # A canonical signed index dtype keeps -1 representable for unsigned and
    # narrow ids; values that wrap negative are invalid and dropped. Clipping
    # keeps sorted ids sorted: low invalid ids first, high ones last.
    index_dtype = jax.dtypes.canonicalize_dtype(jnp.int64)
    ids = jnp.clip(jnp.asarray(segment_ids).astype(index_dtype), -1, num_segments)
    if not indices_are_sorted:
        order = jnp.argsort(ids, stable=True)
        ids, values = ids[order], values[order]
    starts = jnp.concatenate([jnp.ones(1, bool), ids[1:] != ids[:-1]])
    _, prefix = jax.lax.associative_scan(_segmented_add, (starts, values))
    # Each segment's total is its last prefix element: gather, never scatter.
    segments = jnp.arange(num_segments, dtype=ids.dtype)
    last = jnp.clip(jnp.searchsorted(ids, segments, side="right") - 1, 0, n - 1)
    present = (ids[last] == segments).reshape(
        (num_segments,) + (1,) * (values.ndim - 1)
    )
    return jnp.where(present, prefix[last], jnp.zeros((), values.dtype))


def _reduces_sequentially(values) -> bool:
    """Whether ``values`` live on CPU, where XLA scatter adds updates in order.

    Traced values carry no device, so the default backend decides; a CPU
    default with work explicitly placed on an accelerator would therefore use
    the scatter. The package never places work that way.
    """
    if isinstance(values, jax.core.Tracer):
        return jax.default_backend() == "cpu"
    return all(device.platform == "cpu" for device in values.devices())


def _bucketed_size(n: int) -> int:
    """Next power of two, so varying spike counts share few executables."""
    return 1 << max(n - 1, 0).bit_length()


def deterministic_row_sum(
    values: jnp.ndarray,
    row_ids: jnp.ndarray,
    n_rows: int,
    *,
    indices_are_sorted: bool = False,
) -> jnp.ndarray:
    """Deterministic row sums: sequential scatter on CPU, segmented scan elsewhere.

    Parameters
    ----------
    values : jnp.ndarray, shape (n, ...)
    row_ids : array-like, shape (n,), integer
        Ids outside ``[0, n_rows)`` are dropped.
    n_rows : int
    indices_are_sorted : bool
        True only when ``row_ids`` is verified nondecreasing.

    Returns
    -------
    sums : jnp.ndarray, shape (n_rows, ...)

    Notes
    -----
    XLA:CPU scatter applies updates sequentially, so it is deterministic and
    fastest there. Other backends avoid floating-point scatter atomics, which
    change results between otherwise identical evaluations. Concrete
    off-CPU calls pad the spike axis to a power of two with dropped ids, so
    chunks with varying spike counts reuse a bounded set of executables.
    """
    if not isinstance(values, jax.core.Tracer):
        values = jnp.asarray(values)
    if _reduces_sequentially(values):
        return jax.ops.segment_sum(
            values,
            row_ids,
            num_segments=n_rows,
            indices_are_sorted=indices_are_sorted,
        )
    row_ids = jnp.asarray(row_ids)
    if not isinstance(values, jax.core.Tracer):
        padding = _bucketed_size(values.shape[0]) - values.shape[0]
        if padding:
            # Trailing n_rows ids are dropped and keep sorted ids sorted.
            values = jnp.concatenate(
                [values, jnp.zeros((padding, *values.shape[1:]), values.dtype)]
            )
            row_ids = jnp.concatenate(
                [row_ids, jnp.full(padding, n_rows, dtype=row_ids.dtype)]
            )
    return deterministic_segment_sum(values, row_ids, n_rows, indices_are_sorted)


def deterministic_row_add(
    output: jnp.ndarray,
    values: jnp.ndarray,
    row_ids: jnp.ndarray,
    column_start=0,
    *,
    indices_are_sorted: bool = False,
) -> jnp.ndarray:
    """Add per-spike column-tile vectors into ``output[:, column_start:...]``.

    Parameters
    ----------
    output : jnp.ndarray, shape (n_rows, n_columns)
    values : jnp.ndarray, shape (n, tile_columns)
    row_ids : array-like, shape (n,), integer
    column_start : int or scalar array
        First output column of the tile.
    indices_are_sorted : bool

    Returns
    -------
    output : jnp.ndarray, shape (n_rows, n_columns)
        The tile holds ``initial + sum(contributions)``. Off CPU the reduction
        also keeps scan temporaries the size of ``values`` (see
        ``deterministic_segment_sum``). As with ``dynamic_slice``, a tile that
        would run past the last column is shifted left, so callers must pass
        tiles that fit.
    """
    values = jnp.asarray(values).astype(output.dtype)
    if output.shape[0] == 0 or values.shape[0] == 0 or values.size == 0:
        return output
    tile = deterministic_row_sum(
        values, row_ids, output.shape[0], indices_are_sorted=indices_are_sorted
    )
    # dynamic_slice needs one index dtype; x64 makes a literal 0 int64.
    column_start = jnp.asarray(column_start)
    start = (jnp.zeros((), column_start.dtype), column_start)
    current = jax.lax.dynamic_slice(output, start, tile.shape)
    return jax.lax.dynamic_update_slice(output, current + tile, start)


def sum_spikes_into_rows(values: jnp.ndarray, selection: SpikeSelection) -> jnp.ndarray:
    """Sum one value per selected spike into the local row that owns it.

    Parameters
    ----------
    values : jnp.ndarray, shape (n_selected, ...)
        One entry per spike in ``selection``, in selection order.
    selection : SpikeSelection
        The selection from ``select_spikes_in_rows``; supplies the local row of
        each spike, the row count, and the verified sorted-index hint.

    Returns
    -------
    row_sums : jnp.ndarray, shape (selection.n_rows, ...)
        Rows owning no selected spike sum to zero.

    Notes
    -----
    Uses ``deterministic_row_sum``: CPU keeps the sequential segment reduction
    and its verified ordering hint; other backends use a segmented scan with a
    fixed reduction tree, so repeated evaluations are bitwise identical.
    Invalid row indices are dropped.
    """
    values = jnp.asarray(values)
    if values.dtype == jnp.bool_:
        raise TypeError(
            "Spike row sums require numeric values; Boolean sums are unsupported"
        )
    indices = np.asarray(selection.bin_ind)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("Spike row indices must be a one-dimensional integer array")
    if values.ndim == 0 or values.shape[0] != len(indices):
        raise ValueError("Values must contain one entry per selected spike")
    # Normalize invalid host metadata before JAX dtype canonicalization, so a
    # huge unsigned/out-of-range index cannot truncate into a valid row.
    valid = (indices >= 0) & (indices < selection.n_rows)
    safe_indices = np.full(indices.shape, -1, dtype=np.intp)
    safe_indices[indices >= selection.n_rows] = selection.n_rows
    safe_indices[valid] = indices[valid]
    return deterministic_row_sum(
        values,
        safe_indices,
        selection.n_rows,
        indices_are_sorted=selection.indices_are_sorted,
    )


@jax.jit
def log_gaussian_pdf(
    x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log of the Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples, n_dims)
        Input data.
    mean : jnp.ndarray, shape (n_dims,)
        Mean of the Gaussian.
    sigma : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian.

    Returns
    -------
    log_pdf : jnp.ndarray, shape (n_samples,)
    """
    # For singleton-shaped tiles XLA:CPU can rewrite division by a broadcast
    # scalar into multiplication by a rounded reciprocal, so tiled and untiled
    # KDEs disagree by about 2e-6 relative in Gaussian tails. Materializing the
    # divisor (only a vector here) keeps true division. Preserve true-division
    # promotion and the original normalization formula.
    divisor = sigma
    if x.size == 1 or mean.size == 1:
        shape = jnp.broadcast_shapes(x.shape, mean.shape, jnp.shape(sigma))
        divisor = jax.lax.optimization_barrier(jnp.broadcast_to(sigma, shape))
    return -0.5 * ((x - mean) / divisor) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def gaussian_pdf(x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples, n_dims)
        Input data.
    mean : jnp.ndarray, shape (n_dims,)
        Mean of the Gaussian.
    sigma : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian.

    Returns
    -------
    pdf : jnp.ndarray, shape (n_samples,)
    """
    return jnp.exp(log_gaussian_pdf(x, mean, sigma))


def _log_kernel_matrix(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Log Gaussian kernel matrix summed over dimensions.

    Shared by :func:`kde` and :func:`log_kde` so both build the kernel the same
    way: accumulate per-dimension log densities, defer the single ``exp`` (or
    ``logsumexp``) to the caller.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std : jnp.ndarray, shape (n_dims,)

    Returns
    -------
    log_kernel : jnp.ndarray, shape (n_samples, n_eval_points)
        ``log_kernel[i, j] = sum_d log N(eval_points[j, d] | samples[i, d], std[d])``.
    """
    log_kernel = jnp.zeros((samples.shape[0], eval_points.shape[0]))
    for dim_eval, dim_samp, dim_std in zip(eval_points.T, samples.T, std, strict=True):
        log_kernel += log_gaussian_pdf(
            jnp.expand_dims(dim_eval, axis=0),
            jnp.expand_dims(dim_samp, axis=1),
            dim_std,
        )
    return log_kernel


@jax.jit
def kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Kernel density estimation.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    weights : jnp.ndarray, shape (n_samples,)
        Weights for each sample.

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)

    Notes
    -----
    The kernel is accumulated in log-space and exponentiated once, so per-sample
    densities only underflow to 0 in float32 at the final value (where 0 is
    typically the correct result). :func:`log_kde` (or
    :meth:`KDEModel.predict_log`) remains the fully underflow-safe path when the
    log-density itself is needed.
    """
    distance = jnp.exp(_log_kernel_matrix(eval_points, samples, std))
    # Double-where pattern: substitute safe denominator, then select result.
    # This avoids NaN in both forward pass and gradients.
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)
    return jnp.where(weight_sum > 0, (weights @ distance) / safe_weight_sum, 0.0)


def block_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Kernel density estimation split into blocks.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Size of blocks to do computation over, by default 100
    weights : jnp.ndarray, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)

    Notes
    -----
    Wraps :func:`kde`, whose underflow is deferred to the final density value;
    prefer :func:`block_log_kde` when the log-density itself is needed.
    """
    n_eval_points = eval_points.shape[0]

    if n_eval_points == 0:
        return jnp.zeros((n_eval_points,))

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block densities and concatenate once. The blocks tile
    # [0, n_eval_points) exactly, so this is identical to filling a preallocated
    # array but avoids the O(n_eval * n_blocks) copies of a per-block
    # dynamic_update_slice.
    blocks = [
        kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval_points, block_size)
    ]
    return jnp.concatenate(blocks)


@jax.jit
def log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Log kernel density estimate:
        log p(x) = logsumexp_i [ log w_i + sum_d log N(x_d | s_{i,d}, std_d) ] - logsumexp_i [log w_i]
    Shapes:
      eval_points: (n_eval, n_dims)
      samples:     (n_samp, n_dims)
      std:         (n_dims,)
      weights:     (n_samp,)
    Returns: (n_eval,)
    """
    if eval_points.ndim == 1:
        eval_points = jnp.expand_dims(eval_points, axis=1)

    # log-kernel matrix K_log with shape (n_samp, n_eval)
    K_log = _log_kernel_matrix(eval_points, samples, std)

    # True log-weight: a zero weight (log = -inf) drops the sample from both the
    # numerator and the denominator, exactly like the linear ``kde``. ``safe_log``
    # would floor a zero weight to LOG_EPS and leak that sample's kernel back in
    # (visible at an eval point near a zero-weight sample but far from the rest).
    safe_weights = jnp.where(weights > 0, weights, 1.0)
    log_w = jnp.where(weights > 0, jnp.log(safe_weights), -jnp.inf)  # (n_samp,)
    log_num = logsumexp(log_w[:, None] + K_log, axis=0)  # (n_eval,)
    log_den = logsumexp(log_w)  # scalar; -inf only when every weight is 0
    # All-zero weights -> no support anywhere. Return LOG_EPS (matching the linear
    # kde's 0.0, which callers floor) instead of NaN from -inf - (-inf).
    return jnp.where(jnp.isneginf(log_den), LOG_EPS, log_num - log_den)


def block_log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Log kernel density estimation split into blocks over eval points.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Size of blocks to do computation over, by default 100
    weights : jnp.ndarray, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    log_density_estimate : jnp.ndarray, shape (n_eval_points,)
    """
    n_eval = eval_points.shape[0]

    if n_eval == 0:
        return jnp.full((n_eval,), LOG_EPS)

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block log-densities and concatenate once (see block_kde).
    blocks = [
        log_kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval, block_size)
    ]
    return jnp.concatenate(blocks)


_Samples = TypeVar("_Samples")
_Weights = TypeVar("_Weights")


def drop_zero_weight_samples(
    samples: _Samples, weights: _Weights
) -> tuple[_Samples, _Weights]:
    """Remove samples that carry no weight from a weighted KDE's training set.

    A zero-weight sample adds nothing to a weighted kernel density, so removing
    it leaves the density unchanged while every later evaluation stops paying
    for it. Detectors pass the full position timeline with the group mask as
    weights, so an occupancy model would otherwise evaluate every out-of-group
    sample. When no weight is positive the inputs are returned unchanged, which
    keeps the zero-exposure behavior of the fits.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, ...)
    weights : np.ndarray, shape (n_samples,)
        Non-negative weights.

    Returns
    -------
    samples, weights
        The rows with positive weight.
    """
    keep = np.asarray(weights) > 0.0
    if not np.any(keep) or np.all(keep):
        return samples, weights
    return samples[keep], weights[keep]  # type: ignore[index]


@dataclass
class KDEModel:
    std: jnp.ndarray
    block_size: int | None = None
    samples_: jnp.ndarray | None = field(init=False, default=None)
    weights_: jnp.ndarray | None = field(init=False, default=None)

    def fit(
        self, samples: jnp.ndarray, weights: jnp.ndarray | None = None
    ) -> "KDEModel":
        """Fit the model.

        Parameters
        ----------
        samples : jnp.ndarray, shape (n_samples, n_dims)
            Training samples.

        Returns
        -------
        self : KDEModel
        """
        samples = jnp.asarray(samples)
        if samples.ndim == 1:
            samples = jnp.expand_dims(samples, axis=1)
        self.samples_ = samples
        if weights is None:
            self.weights_ = jnp.ones((samples.shape[0],))
        else:
            self.weights_ = jnp.asarray(weights)

        return self

    def predict(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Predict the density at the evaluation points.

        Parameters
        ----------
        eval_points : jnp.ndarray, shape (n_eval_points, n_dims)

        Returns
        -------
        density : jnp.ndarray, shape (n_eval_points,)
        """
        if self.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size, self.weights_)

    def predict_log(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """
        Log-density version of predict(). Same inputs, returns log p(eval_points).
        """
        if self.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )
        return block_log_kde(eval_points, self.samples_, std, block_size, self.weights_)


def select_spike_rows(
    array: "np.ndarray | jnp.ndarray", selection: SpikeSelection
) -> np.ndarray:
    """Take the rows an indexer names without materializing the whole array.

    ``select_spikes_in_rows`` returns one selection per unit/electrode, applied to
    the spike times and to the per-spike waveform features. Those arrays are
    recording-length; the selection is one chunk's worth. Converting the array
    before selecting therefore pays for the whole recording on every chunk call.
    For a ``jax.Array`` it is worse than a copy: ``np.asarray`` gathers the array
    to the host **and caches that host copy on the array object**
    (``jax.Array._npy_value``), so the memory is retained for as long as the
    caller holds its inputs.

    Indexing on device avoids both. It is not always available: under a mesh with
    ``AxisType.Explicit`` axes JAX refuses a gather whose output sharding it
    cannot infer, so that case falls back to gather-then-slice — the behaviour
    every backend had before, correct but paying the host copy.

    Parameters
    ----------
    array : np.ndarray or jnp.ndarray, shape (n_spikes, ...)
        Per-spike array for one unit/electrode.
    selection : SpikeSelection
        The selection from ``select_spikes_in_rows``, including the original
        spike count. Every paired array must have exactly that many rows,
        whether the requested selection uses a slice or a mask.

    Returns
    -------
    selected : np.ndarray, shape (n_selected, ...)
        The selected rows, always on the host. A selection that kept a sharded
        input's sharding would carry it into the jitted kernels downstream, which
        also take single-device encoding-model arrays -- JAX rejects that mix.
        The selection is request-sized, so this costs the chunk, not the
        recording; consumers that want a device array convert it themselves.

    Raises
    ------
    ValidationError
        If the array's row count differs from the original spike count. Checking
        before selection catches both extra and missing feature rows, including
        mismatches outside the requested chunk, without reading array values.
    """
    if array.shape[0] != selection.n_spikes:
        raise ValidationError(
            "spike selection does not match the length of a per-spike array",
            expected=f"{selection.n_spikes} rows (one per spike time)",
            got=f"{array.shape[0]} rows",
            hint="Spike times and waveform features must be paired row for row.",
        )

    indexer = selection.indexer
    if not isinstance(array, jax.Array):
        return array[indexer]

    try:
        if isinstance(indexer, slice):
            selected = array[indexer]
        else:
            # A boolean mask becomes a gather of explicit integer positions,
            # which is the form JAX can shard; the mask itself is a host array,
            # so its positions are known here. The length check above is what
            # keeps ``take``'s default ``mode='fill'`` from turning a mismatch
            # into NaN rows.
            selected = jnp.take(array, jnp.asarray(np.flatnonzero(indexer)), axis=0)
    except _JAX_SHARDING_ERRORS:
        # Explicitly sharded indexing may have ambiguous output sharding.
        # Preserve its host fallback, but let unrelated indexing, device and
        # memory errors propagate without copying the full recording.
        return np.asarray(array)[indexer]

    return np.asarray(selected)


@requires_time_edges
def get_spikecount_per_time_bin(
    spike_times: np.ndarray,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    _spike_time_order: _SpikeTimeOrder | None = None,
) -> np.ndarray:
    """Get the number of spikes in each requested decode bin.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    time_edges : np.ndarray, shape (n_bins + 1,)
        FULL decoding bin edges, which define the bin a spike belongs to (see
        ``select_spikes_in_rows``). Direct calls validate these edges. Internal
        calls with prepared spike ordering reuse the detector's validated grid.
    row_slice : slice | None, optional
        Contiguous range of bins to count, by default None (all bins).
        Counting bins ``[a, b)`` of the full timeline gives the same values as
        counting all bins and slicing ``[a:b]``, so concatenating the chunks of
        a partition reproduces the full-time counts exactly.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    count : np.ndarray, shape (n_rows,)
        ``n_rows`` is ``n_bins`` by default, else the length of ``row_slice``.
    """
    time_edges = (
        validate_time_edges(time_edges)
        if _spike_time_order is None
        else np.asarray(time_edges)
    )
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    selection = select_spikes_in_rows(
        spike_times,
        row_start,
        row_stop,
        time_edges=time_edges,
        _spike_time_order=_spike_time_order,
    )
    return np.bincount(selection.bin_ind, minlength=selection.n_rows)


def _spike_counts_matrix(
    spike_times: list[np.ndarray],
    time_edges: np.ndarray,
    desc: str,
    disable_progress_bar: bool,
    row_slice: slice | None = None,
    *,
    _spike_time_order: _SpikeTimeOrder | None = None,
) -> np.ndarray:
    """Stack per-neuron spike counts into a ``(n_rows, n_neurons)`` matrix.

    ``get_spikecount_per_time_bin`` bins spikes against the full ``time_edges`` and
    selects those owned by ``row_slice`` internally, so no explicit pre-masking
    is needed here. ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    counts = [
        get_spikecount_per_time_bin(
            neuron_spike_times,
            time_edges=time_edges,
            row_slice=row_slice,
            _spike_time_order=_spike_time_order,
        )
        for neuron_spike_times in tqdm(
            spike_times, unit="cell", desc=desc, disable=disable_progress_bar
        )
    ]
    if not counts:  # zero neurons
        return np.zeros((row_stop - row_start, 0))
    return np.stack(counts, axis=1)


def decode_bin_centers(
    time_edges: np.ndarray, row_start: int, row_stop: int
) -> np.ndarray:
    """Centers of the decode bins ``[row_start, row_stop)``.

    Local-position kernels, the non-local penalty, and local likelihoods
    evaluate the animal's position at these observation coordinates.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
    row_start : int
    row_stop : int

    Returns
    -------
    centers : np.ndarray, shape (row_stop - row_start,)
    """
    edges = np.asarray(time_edges)[row_start : row_stop + 1]
    return edges[:-1] + 0.5 * np.diff(edges)


def safe_divide(numerator, denominator, eps=EPS, condition=None):
    """Safely divide two arrays, avoiding division by zero.

    Parameters
    ----------
    numerator : jnp.ndarray
        Numerator array of any shape.
    denominator : jnp.ndarray
        Denominator array, must be broadcastable with numerator.
    eps : float, optional
        Small value to avoid division by zero, by default 1e-8.
    condition : jnp.ndarray, optional
        Boolean condition array to apply the division, by default None.
        If None, condition is computed as abs(denominator) < eps.
        Useful if pre-computing the condition is more efficient.

    Returns
    -------
    result : jnp.ndarray
        Result of safe division with same shape as broadcast of inputs.
        Where condition is True, returns eps instead of dividing.
    """
    if condition is None:
        condition = jnp.abs(denominator) < eps

    # Double-where: substitute safe denominator first, then select result.
    # This avoids NaN in both forward pass and gradients.
    safe_denominator = jnp.where(condition, 1.0, denominator)
    return jnp.where(condition, eps, numerator / safe_denominator)


def safe_log(x, eps=EPS, condition=None):
    """Safely compute the logarithm of an array, avoiding log(0).

    Parameters
    ----------
    x : jnp.ndarray
        Input array of any shape.
    eps : float, optional
        Small value to avoid log(0), by default 1e-8.
    condition : jnp.ndarray, optional
        Boolean condition array to apply the logarithm, by default None.
        If None, condition is computed as abs(x) < eps.
        Useful if pre-computing the condition is more efficient.

    Returns
    -------
    result : jnp.ndarray
        Logarithm of input array with same shape as x.
        Where condition is True, returns log(eps) instead of log(0).
    """
    if condition is None:
        condition = jnp.abs(x) < eps

    return jnp.log(jnp.where(condition, eps, x))
