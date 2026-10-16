from collections import OrderedDict
from collections.abc import Callable, Sequence, Sized
from dataclasses import dataclass, field
from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from jax.nn import logsumexp
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

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


def validate_bandwidth(std, n_dims: int, name: str) -> jnp.ndarray:
    """Kernel bandwidths as a per-dimension array, rejecting unusable values.

    A zero, negative or non-finite bandwidth would fit a degenerate kernel that
    yields plausible-looking floor rates instead of an error.

    Parameters
    ----------
    std : float or array-like, shape () or (n_dims,)
        Python, NumPy or JAX scalar, or one bandwidth per dimension.
    n_dims : int
        Number of kernel dimensions.
    name : str
        Parameter name for error messages.

    Returns
    -------
    std : jnp.ndarray, shape (n_dims,)
        In JAX's default floating dtype.

    Raises
    ------
    ValidationError
        If a value is not finite and positive, the shape is neither scalar nor
        ``(n_dims,)``, or a value overflows or underflows to zero in JAX's
        default floating dtype.
    """
    values = np.asarray(std, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValidationError(
            f"{name} must contain only finite positive values",
            expected="finite bandwidths greater than zero",
            got=f"{name}={std!r}",
            hint="Kernel bandwidths are standard deviations in the data's units.",
        )
    if values.ndim == 0:
        values = np.full(n_dims, values.item())
    if values.shape != (n_dims,):
        raise ValidationError(
            f"{name} has the wrong shape",
            expected=f"a scalar or shape ({n_dims},)",
            got=f"shape {values.shape}",
        )
    dtype = jax.dtypes.canonicalize_dtype(values.dtype)
    with np.errstate(over="ignore", under="ignore"):
        values = values.astype(dtype)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValidationError(
            f"{name} is not representable as finite positive {dtype} values",
            got=f"{name}={std!r}",
            hint="Rescale the data or enable JAX float64 (jax_enable_x64).",
        )
    return jnp.asarray(values)


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


def _block_duration_values(block_durations):
    """The block's first duration and its first different one (or the first).

    Parameters
    ----------
    block_durations : jnp.ndarray, shape (block_rows,)

    Returns
    -------
    first, other : jnp.ndarray, shape ()
    """
    first = block_durations[0]
    return first, block_durations[jnp.argmax(block_durations != first)]


def _matrix_emission_is_safe(rates, block_durations, accumulator_dtype):
    """Whether one block's Poisson emission can use the rate-matrix product.

    Parameters
    ----------
    rates : jnp.ndarray, shape (n_neurons, n_bins)
    block_durations : jnp.ndarray, shape (block_rows,)
    accumulator_dtype : dtype
        The emission's accumulator dtype.

    Returns
    -------
    is_safe : jnp.ndarray, shape ()
        True when every rate-duration product is positive and finite, the block
        has at most two distinct durations (a uniform grid whose timestamps
        round to two values), and the inputs already have the accumulator dtype.
    """
    first, other = _block_duration_values(block_durations)
    smallest = jnp.min(rates) * jnp.min(block_durations)
    largest = jnp.max(rates) * jnp.max(block_durations)
    return (
        jnp.all(jnp.isfinite(rates))
        & jnp.all(jnp.isfinite(block_durations))
        & jnp.all((block_durations == first) | (block_durations == other))
        & (jnp.min(block_durations) > 0)
        & (smallest > 0)
        & jnp.isfinite(largest)
        & (jnp.result_type(rates, block_durations) == accumulator_dtype)
    )


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
    Exact zeros, non-finite products and blocks with more than two distinct
    durations retain xlogy.
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
        matrix_is_safe = _matrix_emission_is_safe(
            rates, block_durations, accumulator_dtype
        )

        def matrix_accumulation(_):
            dtype = jnp.result_type(rates, block_durations)
            block_counts_float = block_counts.astype(dtype)
            # Each row uses the product for its own duration, one of the
            # block's at most two values, so a row's value does not depend on
            # where a row partition starts. The zero-valued term (exactly 0 for
            # these finite durations) restores each row's own duration
            # derivative, which the stopped gradients would collapse.
            log_durations = jnp.log(block_durations.astype(dtype))
            duration_tangent = (
                block_counts_float.sum(axis=1, keepdims=True)
                * (log_durations - jax.lax.stop_gradient(log_durations))[:, None]
            )
            first, other = _block_duration_values(block_durations)

            def emission(duration):
                return jnp.matmul(
                    block_counts_float,
                    jnp.log(rates.astype(dtype) * jax.lax.stop_gradient(duration)),
                    precision=jax.lax.Precision.HIGHEST,
                )

            # A block of equal durations (the usual uniform grid) needs one
            # product; ``where`` alone would compute both.
            values = jax.lax.cond(
                first == other,
                lambda: emission(first),
                lambda: jnp.where(
                    (block_durations == first)[:, None],
                    emission(first),
                    emission(other),
                ),
            )
            return (values + duration_tangent).astype(accumulator_dtype)

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


# Host bytes for one block of int64 spike counts, so full-grid requests never
# build a rows-by-population count matrix. Block rows are a multiple of the
# non-local emission's fixed 64-row kernel, so blocked and unblocked requests
# are bitwise identical.
COUNT_BLOCK_BYTES = 32 * 1024**2


def _count_block_rows(n_neurons: int) -> int:
    """Rows per spike-count block within ``COUNT_BLOCK_BYTES``, a multiple of 64."""
    return max(64, COUNT_BLOCK_BYTES // (8 * max(n_neurons, 1)) // 64 * 64)


def _concatenate_row_blocks(
    row_start: int,
    row_stop: int,
    n_neurons: int,
    evaluate,
    desc: str,
    disable_progress_bar: bool,
) -> jnp.ndarray:
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
    disable_progress_bar : bool

    Returns
    -------
    result : jnp.ndarray, shape (row_stop - row_start, ...)
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
    Host counts are bounded by ``COUNT_BLOCK_BYTES``. Concatenating
    the block outputs briefly holds about twice the returned array. The
    progress bar advances per block rather than per neuron.
    """
    if _spike_time_order is None:
        # Verify each neuron's spike ordering once, not once per block.
        _spike_time_order = _SpikeTimeOrder()
    if row_start == row_stop:
        return _poisson_nonlocal_log_likelihood(
            jnp.zeros((0, len(spike_times))), rates, durations, summed_rates
        )

    def evaluate(start, stop):
        counts = _spike_counts_matrix(
            spike_times,
            time_edges,
            "Non-Local Likelihood",
            True,
            slice(start, stop),
            _spike_time_order=_spike_time_order,
        )
        return _poisson_nonlocal_log_likelihood(
            jnp.asarray(counts),
            rates,
            durations[start - row_start : stop - row_start],
            summed_rates,
        )

    return _concatenate_row_blocks(
        row_start,
        row_stop,
        len(spike_times),
        evaluate,
        "Non-Local Likelihood",
        disable_progress_bar,
    )


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
            position_at_spike_times = env._linear_position(position_at_spike_times)[
                :, None
            ]
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
    """Host spike times, verified ordering and other preparation for one prediction.

    Each original input is converted and checked on first use, then reused by
    every observation state and chunk. Strong references prevent identity reuse.
    Inputs must remain unchanged during the prediction; a fresh instance on the
    next call rechecks even arrays modified in place. Never store this object on
    a detector or pass it into a JIT kernel.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[object, np.ndarray, bool]] = {}
        self._prepared: dict[tuple, tuple[tuple, object]] = {}
        self._kde_prepared: OrderedDict[tuple, tuple[tuple, object, int]] = (
            OrderedDict()
        )
        self._kde_prepared_bytes = 0

    def memo(self, name: str, sources: tuple, build):
        """Return ``build()``, computed once per ``name`` and ``sources`` identities.

        ``sources`` are kept alive so their ids cannot be reused during the
        prediction.
        """
        key = (name, *map(id, sources))
        if key not in self._prepared:
            self._prepared[key] = (sources, build())
        return self._prepared[key][1]

    def memo_kde(self, name: str, sources: tuple, build):
        """Prediction-scoped preparation, bounding values and retained source arrays.

        Counting source arrays conservatively includes already resident inputs.
        It also bounds old fitted leaves retained after a model is refitted.
        """
        key = (name, *map(id, sources))
        if key in self._kde_prepared:
            self._kde_prepared.move_to_end(key)
            return self._kde_prepared[key][1]
        value = build()
        size = sum(
            int(getattr(leaf, "nbytes", 0))
            for leaf in jax.tree_util.tree_leaves((sources, value))
        )
        if size <= _KDE_PREPARATION_BYTES:
            while self._kde_prepared and (
                self._kde_prepared_bytes + size > _KDE_PREPARATION_BYTES
                or len(self._kde_prepared) >= _KDE_PREPARATION_ENTRIES
            ):
                _, (_, _, old_size) = self._kde_prepared.popitem(last=False)
                self._kde_prepared_bytes -= old_size
            self._kde_prepared[key] = (sources, value, size)
            self._kde_prepared_bytes += size
        return value

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
        Ids outside ``[0, num_segments)`` are dropped. With x64 disabled,
        64-bit ids wrap to 32 bits first; normalize them on the host (see
        ``spike_row_ids``).
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
    Unlike a scatter, the scan keeps ``O(n)`` temporaries: about 1.5x the
    size of ``values`` for sorted ids and 2.5x unsorted on XLA:CPU, and about
    0.05-0.4x on an A100.
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
        Ids outside ``[0, n_rows)`` are dropped. With x64 disabled, 64-bit
        ids wrap to 32 bits first; normalize them on the host (see
        ``spike_row_ids``).
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
    if values.ndim == 0 or values.shape[0] != np.shape(row_ids)[0]:
        raise ValueError("values must contain one row per row id")
    if _reduces_sequentially(values):
        return jax.ops.segment_sum(
            values,
            row_ids,
            num_segments=n_rows,
            indices_are_sorted=indices_are_sorted,
        )
    # Widen before padding so the n_rows sentinel cannot wrap in narrow dtypes.
    row_ids = jnp.asarray(row_ids).astype(jax.dtypes.canonicalize_dtype(jnp.int64))
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
        The tile holds ``initial + sum(contributions)``. Each call builds an
        ``(n_rows, tile_columns)`` row-sum temporary. Off CPU the reduction
        also keeps segmented-scan temporaries (see ``deterministic_segment_sum``;
        about 0.05-0.4x of ``values`` measured on an A100). As with
        ``dynamic_slice``, a tile that
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
    return deterministic_row_sum(
        values,
        spike_row_ids(selection),
        selection.n_rows,
        indices_are_sorted=selection.indices_are_sorted,
    )


def _normalized_row_ids(
    indices: np.ndarray, n_rows: int, n_padded: int | None = None
) -> np.ndarray:
    """Host row ids for ``deterministic_row_sum``, optionally padded.

    Parameters
    ----------
    indices : array-like, shape (n_selected,), integer
    n_rows : int
    n_padded : int, optional
        Length to pad to. Padded entries use the dropped id ``n_rows``, which
        keeps sorted ids sorted.

    Returns
    -------
    row_ids : np.ndarray of intp, shape (n_padded or n_selected,)
        Negative ids become -1 and ids at or above ``n_rows`` become ``n_rows``
        before JAX dtype canonicalization, so a huge unsigned or out-of-range
        index cannot truncate into a valid row.
    """
    indices = np.asarray(indices)
    n_ids = len(indices) if n_padded is None else n_padded
    safe_indices = np.full(n_ids, n_rows, dtype=np.intp)
    valid = (indices >= 0) & (indices < n_rows)
    safe_indices[: len(indices)][indices < 0] = -1
    safe_indices[: len(indices)][valid] = indices[valid]
    return safe_indices


def spike_row_ids(selection: SpikeSelection, n_padded: int | None = None) -> np.ndarray:
    """``_normalized_row_ids`` for a selection's spikes and row count."""
    return _normalized_row_ids(selection.bin_ind, selection.n_rows, n_padded)


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
        Evaluation rows per block, by default 100. Reduced when needed so one
        block's (samples x rows) kernel matrix fits ``_KDE_BLOCK_KERNEL_BYTES``.
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
    block_size = _kde_block_rows(block_size, eval_points, samples, std, weights)
    if block_size > 0 and -(-n_eval_points // block_size) > _UNROLLED_KDE_BLOCKS:
        # Many small blocks (large sample sets) run as one compiled loop
        # instead of one dispatch per block.
        return _compiled_block_kde(eval_points, samples, std, block_size, weights)
    blocks = [
        kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval_points, block_size)
    ]
    return jnp.concatenate(blocks)


def _octave_size(n: int, log2_sizes: int) -> int:
    """Smallest of ``2**log2_sizes`` sizes per octave at least ``n`` (``n >= 1``)."""
    step = 1 << max((n - 1).bit_length() - 1 - log2_sizes, 0)
    return -(-n // step) * step


def _quarter_octave(n: int) -> int:
    """Smallest of four sizes per octave that is at least ``n`` (``n >= 1``)."""
    return _octave_size(n, 2)


def _padded_sample_count(n_samples: int) -> int:
    """Encoding samples per electrode: four sizes per octave, at most 25% padding."""
    return 16 if n_samples <= 16 else _quarter_octave(n_samples)


def _padded_occupancy_count(n_samples: int) -> int:
    """Occupancy KDE samples: 64 sizes per octave, at most 1/64 padding.

    Occupancy samples grow with the recording and each one is evaluated at
    every spike, so they are padded more finely than encoding samples. EM
    refits them from decode bins, one fewer than the position samples of the
    first fit; both counts then usually share a size, so the local kernels
    do not recompile.
    """
    return _octave_size(n_samples, 6)


def _padded_spike_count(n_spikes: int, block_size: int, *, fine: bool = False) -> int:
    """Decoding spikes per electrode kernel call, from a few sizes.

    Up to one block: powers of two (at least 16), or with ``fine`` four sizes
    per octave (at most 25% padding) for kernels whose per-spike cost is
    high. Above one block: whole blocks, their count padded to four sizes
    per octave, so a large request grows by at most a quarter plus one
    block. Chunks with varying spike counts then reuse few compiled kernels.

    Parameters
    ----------
    n_spikes : int
    block_size : int
    fine : bool, optional

    Returns
    -------
    n_padded : int
    """
    if n_spikes <= block_size:
        size = (
            _padded_sample_count(n_spikes)
            if fine
            else 1 << max(n_spikes - 1, 15).bit_length()
        )
        return min(block_size, size)
    return block_size * _quarter_octave(-(-n_spikes // block_size))


def _pad_rows(array, n_rows: int) -> np.ndarray:
    """Zero rows appended on the host to reach ``n_rows``."""
    array = np.asarray(array)
    padding = np.zeros((n_rows - array.shape[0], *array.shape[1:]), array.dtype)
    return np.concatenate([array, padding])


def _padded_samples(
    spike_time_order: "_SpikeTimeOrder",
    samples: np.ndarray | jnp.ndarray,
    weights: np.ndarray | jnp.ndarray | None,
    count: Callable[[int], int] = _padded_sample_count,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Device KDE samples and weights padded with zero-weight rows, once per prediction.

    Zero-weight samples add nothing to weighted kernel sums, so electrodes with
    nearby encoding sizes share compiled kernels.

    Parameters
    ----------
    spike_time_order : _SpikeTimeOrder
        The prediction's preparation cache; results are reused per source arrays.
    samples : array, shape (n_samples, n_dims)
    weights : array, shape (n_samples,), or None
        None means uniform weights.
    count : callable, optional
        Padded size for a sample count, by default ``_padded_sample_count``.

    Returns
    -------
    samples, weights : jnp.ndarray, shape (n_padded, n_dims) and (n_padded,)
    """

    def build():
        n_samples = count(np.shape(samples)[0])
        sample_weights = (
            np.ones(np.shape(samples)[0]) if weights is None else np.asarray(weights)
        )
        return (
            jnp.asarray(_pad_rows(samples, n_samples)),
            jnp.asarray(_pad_rows(sample_weights, n_samples)),
        )

    return spike_time_order.memo(
        f"padded_samples:{count.__name__}", (samples, weights), build
    )


# Up to this many evaluation blocks are traced individually: faster on CPU
# than a loop, and the compiled graph stays small.
_UNROLLED_KDE_BLOCKS = 8

# Bytes for one evaluation block's (samples x rows) kernel matrix. Samples such
# as an occupancy model's position samples grow with the recording, so a fixed
# row count alone does not bound the workspace.
_KDE_BLOCK_KERNEL_BYTES = 256 * 1024**2


def _kde_block_rows(block_size: int, eval_points, samples, std, weights) -> int:
    """Evaluation rows per block: ``block_size``, reduced to the kernel budget.

    Parameters
    ----------
    block_size : int
        Requested rows per block.
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std, weights : array-like
        The other kernel operands. With the points, samples and the kernel's
        accumulator (the default float, float64 under x64) they set the
        kernel's dtype.

    Returns
    -------
    rows : int
        ``min(block_size, rows whose kernel matrix fits _KDE_BLOCK_KERNEL_BYTES)``,
        at least one row when ``block_size`` is positive.
    """
    # _log_kernel_matrix accumulates into the default float type.
    accumulator = jax.dtypes.canonicalize_dtype(jnp.float64)
    itemsize = jnp.result_type(eval_points, samples, std, weights, accumulator).itemsize
    budget_rows = _KDE_BLOCK_KERNEL_BYTES // (max(samples.shape[0], 1) * itemsize)
    return min(block_size, max(1, budget_rows))


def _traced_blocks(blocked, kernel, points, samples, std, block_size, weights):
    """Blocked KDE inside a trace, with a bounded graph for any row count.

    Rows per block are reduced so one block's kernel matrix fits
    ``_KDE_BLOCK_KERNEL_BYTES``. Up to ``_UNROLLED_KDE_BLOCKS`` blocks are traced
    as ``blocked`` does. More blocks run ``kernel`` in a loop over the same block
    size, with a zero-padded final block evaluated and trimmed.
    """
    n_points = points.shape[0]
    block_size = _kde_block_rows(block_size, points, samples, std, weights)
    n_blocks = -(-n_points // block_size)
    if n_blocks <= _UNROLLED_KDE_BLOCKS:
        return blocked(points, samples, std, block_size, weights)
    padded = jnp.pad(points, ((0, n_blocks * block_size - n_points), (0, 0)))
    values = jax.lax.map(
        lambda block: kernel(block, samples, std, weights),
        padded.reshape(n_blocks, block_size, points.shape[1]),
    )
    return values.reshape(-1)[:n_points]


def _traced_block_kde(points, samples, std, block_size, weights):
    """``block_kde`` inside a trace, with a bounded graph for any row count.

    Each point's density depends only on that point, so the unrolled and
    looped forms compute the same values; compiled forms can round
    differently (XLA may multiply by a hoisted reciprocal of the bandwidth
    instead of dividing).

    Parameters
    ----------
    points : jnp.ndarray, shape (n_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std : jnp.ndarray, shape (n_dims,)
    block_size : int
    weights : jnp.ndarray, shape (n_samples,)

    Returns
    -------
    density : jnp.ndarray, shape (n_points,)
    """
    return _traced_blocks(block_kde, kde, points, samples, std, block_size, weights)


def _traced_block_log_kde(
    points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """``block_log_kde`` inside a trace, blocked like :func:`_traced_block_kde`.

    Parameters
    ----------
    points : jnp.ndarray, shape (n_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std : jnp.ndarray, shape (n_dims,)
    block_size : int
    weights : jnp.ndarray, shape (n_samples,)

    Returns
    -------
    log_density : jnp.ndarray, shape (n_points,)
    """
    return _traced_blocks(
        block_log_kde, log_kde, points, samples, std, block_size, weights
    )


# Eager block_kde/block_log_kde with more blocks than _UNROLLED_KDE_BLOCKS run
# these: one compiled loop over the blocks instead of one dispatch per block.
_compiled_block_kde = jax.jit(_traced_block_kde, static_argnames=("block_size",))
_compiled_block_log_kde = jax.jit(
    _traced_block_log_kde, static_argnames=("block_size",)
)


def _relative_weights(weights: jnp.ndarray):
    """Weights relative to the largest, their logs, and their sum.

    Parameters
    ----------
    weights : jnp.ndarray, shape (n_samples,)
        Weights that are not positive (including NaN) carry no mass.

    Returns
    -------
    relative_weights : jnp.ndarray, shape (n_samples,)
        ``weights / max(weights)``; ratios below ``tiny`` flush to zero.
    log_weights : jnp.ndarray, shape (n_samples,)
        Their logs, ``-inf`` without mass. A ratio that would flush takes its
        log from the weights themselves, so no positive weight is discarded.
    total : jnp.ndarray, shape ()
        ``sum(relative_weights)``, at least one with any mass. Flushed ratios
        add less than ``n_samples * tiny`` to it.
    """
    positive = weights > 0
    positive_weights = jnp.where(positive, weights, 0.0)
    max_weight = jnp.max(positive_weights, initial=0.0)
    safe_max_weight = jnp.where(max_weight > 0, max_weight, 1.0)
    weight_scale = safe_max_weight
    # XLA can lower this division to multiplication by a reciprocal. For
    # weights near the dtype maximum that reciprocal is subnormal and may
    # flush to zero. A power-of-two prescale keeps both operands representable
    # without changing ordinary-weight arithmetic; the barrier prevents
    # cancellation of this protective scale during compiler reassociation.
    positive_weights, weight_scale = jax.lax.cond(
        weight_scale > 1.0 / jnp.finfo(positive_weights.dtype).tiny,
        lambda pair: jax.lax.optimization_barrier((pair[0] * 0.25, pair[1] * 0.25)),
        lambda pair: pair,
        (positive_weights, weight_scale),
    )
    relative_weights = positive_weights / weight_scale
    representable = relative_weights >= jnp.finfo(relative_weights.dtype).tiny
    log_weights = jnp.where(
        representable,
        jnp.log(jnp.where(representable, relative_weights, 1.0)),
        jnp.log(jnp.where(positive, weights, 1.0)) - jnp.log(safe_max_weight),
    )
    return (
        relative_weights,
        jnp.where(positive, log_weights, -jnp.inf),
        jnp.sum(relative_weights),
    )


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

    if samples.shape[0] == 0:
        return jnp.full(eval_points.shape[0], LOG_EPS)

    # log-kernel matrix K_log with shape (n_samp, n_eval)
    K_log = _log_kernel_matrix(eval_points, samples, std)

    _, log_weights, total = _relative_weights(weights)
    log_total = jnp.log(jnp.where(total > 0, total, 1.0))
    positive = weights > 0

    def supported(_):
        # Offset each point by its largest weighted term, so weights of any
        # representable ratio contribute. Each kernel is taken relative to the
        # offset before its log weight is added: adding a log weight to a huge
        # negative kernel first would round away small terms before they can
        # cancel, and identical sources must retain their single kernel.
        offset = jax.lax.stop_gradient(
            jnp.max(
                jnp.where(positive[:, None], K_log + log_weights[:, None], -jnp.inf),
                axis=0,
            )
        )
        safe_offset = jnp.where(jnp.isfinite(offset), offset, 0.0)
        relative = jnp.where(
            positive[:, None],
            (K_log - safe_offset[None, :]) + log_weights[:, None],
            -jnp.inf,
        )
        # A positive HIGHEST dot avoids shape-dependent reassociation of the
        # sum across full and looped query batches. Its operand uses the same
        # bounded kernel matrix already covered by KDE blocking.
        mass = jnp.matmul(
            jnp.ones(relative.shape[0], relative.dtype),
            jnp.exp(relative),
            precision=jax.lax.Precision.HIGHEST,
        )
        log_mass = jnp.log(jnp.where(mass > 0, mass, 1.0))
        values = jnp.where(mass > 0, safe_offset + (log_mass - log_total), -jnp.inf)
        values = jnp.where(jnp.isnan(mass), jnp.nan, values)
        # Invalid sources retain the established diagnostic even at zero mass.
        return jnp.where(jnp.any(jnp.isnan(K_log), axis=0), jnp.nan, values)

    # Bypass undefined empty reductions, including their derivatives.
    return jax.lax.cond(
        jnp.any(positive),
        supported,
        lambda _: jnp.full(eval_points.shape[0], LOG_EPS),
        None,
    )


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
        Evaluation rows per block, by default 100. Reduced when needed so one
        block's (samples x rows) kernel matrix fits ``_KDE_BLOCK_KERNEL_BYTES``.
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
    block_size = _kde_block_rows(block_size, eval_points, samples, std, weights)
    if block_size > 0 and -(-n_eval // block_size) > _UNROLLED_KDE_BLOCKS:
        return _compiled_block_log_kde(eval_points, samples, std, block_size, weights)
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


def kde_log_density(model: KDEModel, points: jnp.ndarray) -> jnp.ndarray:
    """True log density, with zero total fitted weight represented as zero mass.

    Runs outside jit. One compiled pass scales each point by an upper bound
    on its largest log kernel, exact for 1D points; points whose pass could
    have lost mass are then evaluated in log space (see
    :func:`_scaled_kde_log_density`).
    """
    if type(model) is KDEModel:
        if model.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        points = jnp.asarray(points)
        if points.ndim == 1:
            points = points[:, None]
        return _true_kde_log_density(
            points,
            model.samples_,
            as_std_array(model.std, points.shape[1]),
            points.shape[0] if model.block_size is None else model.block_size,
            model.weights_,
            _sorted_sample_keys(points, model.samples_, model.weights_),
        )
    # Preserve custom KDE subclasses' predict_log implementations.
    values = model.predict_log(points)
    total = jnp.sum(model.weights_)
    return jnp.where(
        jnp.isnan(total), jnp.nan, jnp.where(total > 0.0, values, -jnp.inf)
    )


def _true_kde_log_density(points, samples, std, block_size, weights, keys=None):
    """True log density of fitted KDE leaves, outside jit.

    One compiled scaled pass (see :func:`_scaled_kde_log_density`), then the
    flagged points in log space. Without ``keys`` points are scaled by the
    kernel's peak, which suits points expected near the samples.

    Parameters
    ----------
    points : jnp.ndarray, shape (n_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std : jnp.ndarray, shape (n_dims,)
    block_size : int
    weights : jnp.ndarray, shape (n_samples,)
    keys : jnp.ndarray, shape (n_samples,), optional
        From :func:`_sorted_sample_keys`.

    Returns
    -------
    log_density : jnp.ndarray, shape (n_points,)
    """
    values, needs_log_space = _compiled_scaled_kde_log_density(
        points, samples, std, block_size, weights, keys
    )
    return _repair_in_log_space(
        values, np.asarray(needs_log_space), points, samples, std, block_size, weights
    )


def _repair_in_log_space(
    values, needs_log_space, points, samples, std, block_size, weights
):
    """Replace the flagged points of a scaled pass with log-space values.

    Parameters
    ----------
    values : jnp.ndarray, shape (n_points,)
    needs_log_space : np.ndarray of bool, shape (n_points,)
    points, samples, std, block_size, weights
        The pass's inputs.

    Returns
    -------
    log_density : jnp.ndarray, shape (n_points,)
    """
    if not needs_log_space.any():
        return values
    index = np.flatnonzero(needs_log_space)
    exact = _compiled_log_space_kde_log_density(
        points[_padded_index(index)], samples, std, block_size, weights
    )
    return values.at[index].set(exact[: index.size])


def _kde_log_densities(models, points) -> list:
    """True log densities of several fitted KDEs at the same points.

    ``kde_log_density`` for each model, with fewer compiled kernels: models
    whose padded sample counts (four sizes per octave, zero-weight rows that
    add no mass) and block sizes match share one pass, mapped over models in
    batches (see :func:`_batched_kde_rows`). Runs outside jit; custom
    subclasses use ``kde_log_density``.

    Parameters
    ----------
    models : sequence of KDEModel
    points : jnp.ndarray, shape (n_points, n_dims) or (n_points,)

    Returns
    -------
    log_densities : list of jnp.ndarray, shape (n_points,)
        In the order of ``models``.
    """
    points = jnp.asarray(points)
    if points.ndim == 1:
        points = points[:, None]
    results = [None] * len(models)
    groups: dict = {}
    for index, model in enumerate(models):
        if type(model) is not KDEModel:
            results[index] = kde_log_density(model, points)
            continue
        if model.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        std = np.asarray(as_std_array(model.std, points.shape[1]))
        key = (
            _padded_sample_count(model.samples_.shape[0]),
            points.shape[0] if model.block_size is None else model.block_size,
            std.tobytes(),
            np.dtype(model.samples_.dtype).str,
            np.dtype(model.weights_.dtype).str,
        )
        groups.setdefault(key, []).append(index)
    for (n_padded, block_size, _, _, _), members in groups.items():
        samples = np.stack([_pad_rows(models[i].samples_, n_padded) for i in members])
        weights = np.stack([_pad_rows(models[i].weights_, n_padded) for i in members])
        std = as_std_array(models[members[0]].std, points.shape[1])
        keys = None
        if points.shape[1] == 1:
            keys = jnp.asarray(
                np.sort(np.where(weights > 0, samples[..., 0], np.inf), axis=1).astype(
                    samples.dtype, copy=False
                )
            )
        batch_size, batch_block_size = _batched_kde_rows(
            block_size, len(members), points, samples[0], std, weights[0]
        )
        values, needs_log_space = _compiled_batched_scaled_kde_log_density(
            points,
            jnp.asarray(samples),
            std,
            batch_block_size,
            jnp.asarray(weights),
            keys,
            batch_size=batch_size,
        )
        # Rows are split on the host: indexing the device array compiles a
        # slice for every row.
        values, needs_log_space = np.asarray(values), np.asarray(needs_log_space)
        for row, index in enumerate(members):
            results[index] = jnp.asarray(values[row])
            if needs_log_space[row].any():
                results[index] = _repair_in_log_space(
                    results[index],
                    needs_log_space[row],
                    points,
                    jnp.asarray(samples[row]),
                    std,
                    block_size,
                    jnp.asarray(weights[row]),
                )
    return results


def _padded_index(index: np.ndarray) -> np.ndarray:
    """Nonempty ``index`` padded to four sizes per octave (at least 16).

    Padding repeats the last entry, so a scatter through the padded index
    writes only indexed rows. Kernels evaluated on the indexed subset then
    compile for few sizes; gathering callers keep the first ``index.size``
    results.
    """
    return np.pad(
        index, (0, _quarter_octave(max(index.size, 16)) - index.size), mode="edge"
    )


def _log_space_kde_log_density(points, samples, std, block_size, weights):
    """True log density evaluated entirely in log space.

    Callers that map over batches use this directly: under ``vmap`` the
    log-space repair in :func:`traced_kde_log_density` becomes a select that
    evaluates both forms.
    """
    if points.shape[0] == 0:
        return jnp.full((0,), LOG_EPS)
    values = _traced_block_log_kde(points, samples, std, block_size, weights)
    total = jnp.sum(weights)
    return jnp.where(
        jnp.isnan(total), jnp.nan, jnp.where(total > 0.0, values, -jnp.inf)
    )


def _sorted_sample_keys(points, samples, weights):
    """Sorted 1D sample positions for :func:`_kde_log_scale`, or None.

    Samples without positive weight become ``inf``, which sorts last. The
    sort runs on the host: for 4k-20k elements XLA's GPU sort took 0.7-1.6 s
    to compile for each size (A100, JAX 0.9.0).

    Parameters
    ----------
    points : array, shape (n_points, n_dims)
    samples : array, shape (n_samples, n_dims)
    weights : array, shape (n_samples,)

    Returns
    -------
    keys : jnp.ndarray, shape (n_samples,), or None
        None unless the points are 1D and there are samples.
    """
    if points.shape[1] != 1 or samples.shape[0] == 0:
        return None
    samples = np.asarray(samples)
    keys = np.where(np.asarray(weights) > 0, samples[:, 0], np.inf)
    return jnp.asarray(np.sort(keys.astype(samples.dtype, copy=False)))


def _nearest_sample_log_kernel(points, keys, std):
    """Each 1D point's largest log kernel, at its nearest key by binary search.

    Parameters
    ----------
    points : jnp.ndarray, shape (n_points, 1)
    keys : jnp.ndarray, shape (n_samples,)
        Sorted positions of the samples with positive weight, ``inf`` for
        the rest.
    std : jnp.ndarray, shape (1,)

    Returns
    -------
    log_kernel : jnp.ndarray, shape (n_points,)
        ``-inf`` where no key is finite.
    """
    index = jnp.searchsorted(keys, points[:, 0])
    last = keys.shape[0] - 1
    left = keys[jnp.clip(index - 1, 0, last)]
    right = keys[jnp.clip(index, 0, last)]
    nearest = jnp.where(
        jnp.abs(points[:, 0] - left) <= jnp.abs(points[:, 0] - right), left, right
    )
    return log_gaussian_pdf(points[:, 0], nearest, std[0])


def _kde_log_scale(points, std, keys=None):
    """Per-point upper bound on the largest log kernel over positive weights.

    With sorted 1D ``keys`` (:func:`_sorted_sample_keys`) the nearest
    positive-weight sample gives the exact maximum. Otherwise, and for points
    with no finite bound (no positive weights, infinite coordinates), the
    bound is the kernel's peak.

    Parameters
    ----------
    points : jnp.ndarray, shape (n_points, n_dims)
    std : jnp.ndarray, shape (n_dims,)
    keys : jnp.ndarray, shape (n_samples,), optional

    Returns
    -------
    log_scale : jnp.ndarray, shape (n_points,)
    """
    peak = sum(
        log_gaussian_pdf(jnp.zeros(()), jnp.zeros(()), dim_std) for dim_std in std
    )
    if keys is None:
        return jnp.full(points.shape[0], peak)
    scale = _nearest_sample_log_kernel(points, keys, std)
    return jax.lax.stop_gradient(jnp.where(jnp.isfinite(scale), scale, peak))


def _scaled_kde_log_density(
    points, samples, std, block_size, weights, keys=None, *, repair_in_graph
):
    """True log density in one pass per block, each point scaled first.

    Each point's log kernels are shifted by :func:`_kde_log_scale` before
    exponentiating, then summed by a weighted product as the probability-space
    KDE does. Weights enter relative to the largest, so neither weights nor
    their sum overflow. A term is lost only if it flushes below ``tiny``;
    points whose scaled sum is small enough for that to matter are flagged.
    Log space (:func:`log_kde`) offsets each point by its largest weighted
    term, which recovers what either scale loses.

    Parameters
    ----------
    points : jnp.ndarray, shape (n_points, n_dims)
    samples : jnp.ndarray, shape (n_samples, n_dims)
    std : jnp.ndarray, shape (n_dims,)
    block_size : int
    weights : jnp.ndarray, shape (n_samples,)
    keys : jnp.ndarray, shape (n_samples,), optional
        Sorted 1D keys for an exact per-point scale; None scales by the
        kernel's peak.
    repair_in_graph : bool
        Evaluate flagged points in log space inside the graph, under a
        ``lax.cond`` per block.

    Returns
    -------
    log_density : jnp.ndarray, shape (n_points,)
    needs_log_space : jnp.ndarray, shape (n_points,)
        The flagged points, already repaired when ``repair_in_graph``.
    """
    n_points = points.shape[0]
    if n_points == 0:
        return jnp.full((0,), LOG_EPS), jnp.zeros((0,), bool)
    if samples.shape[0] == 0:
        return jnp.full(n_points, -jnp.inf), jnp.zeros(n_points, bool)
    std = as_std_array(std, points.shape[1])
    active = weights > 0
    relative_weights, _, relative_total = _relative_weights(weights)
    safe_relative_total = jnp.where(relative_total > 0, relative_total, 1.0)

    scale = _kde_log_scale(points, std, keys)

    def block_values(block):
        block_points, block_scale = block
        log_kernel = _log_kernel_matrix(block_points, samples, std)
        scaled = jnp.matmul(
            relative_weights,
            jnp.exp(
                jnp.where(active[:, None], log_kernel - block_scale[None, :], -jnp.inf)
            ),
            precision=jax.lax.Precision.HIGHEST,
        )
        # Normalize before adding the scale, as the tiled kernel does. A NaN
        # sum (NaN points or samples) stays NaN.
        average = scaled / safe_relative_total
        defined = (average > 0) | jnp.isnan(average)
        values = jnp.where(
            defined,
            block_scale + jnp.log(jnp.where(defined, average, 1.0)),
            -jnp.inf,
        )
        lost = (
            samples.shape[0]
            * jnp.finfo(scaled.dtype).tiny
            / (4 * jnp.finfo(scaled.dtype).eps)
        )
        needs_log_space = scaled < lost
        if repair_in_graph:
            values = jax.lax.cond(
                jnp.any(needs_log_space),
                lambda values: jnp.where(
                    needs_log_space,
                    log_kde(block_points, samples, std, weights),
                    values,
                ),
                lambda values: values,
                values,
            )
        return values, needs_log_space

    rows = min(_kde_block_rows(block_size, points, samples, std, weights), n_points)
    n_blocks = -(-n_points // rows)
    padding = n_blocks * rows - n_points
    # Edge padding repeats the last point, so padded rows flag nothing new.
    values, needs_log_space = jax.lax.map(
        block_values,
        (
            jnp.pad(points, ((0, padding), (0, 0)), mode="edge").reshape(
                n_blocks, rows, -1
            ),
            jnp.pad(scale, (0, padding), mode="edge").reshape(n_blocks, rows),
        ),
    )
    total = jnp.sum(weights)
    values = jnp.where(
        jnp.isnan(total),
        jnp.nan,
        jnp.where(total > 0.0, values.reshape(-1)[:n_points], -jnp.inf),
    )
    return values, needs_log_space.reshape(-1)[:n_points] & (total > 0.0)


def traced_kde_log_density(points, samples, std, block_size, weights):
    """Traceable true-density counterpart of :func:`kde_log_density`.

    Points are scaled by the kernel's peak, so each block's pass matches the
    probability-space KDE; a block holding points in a distant tail evaluates
    them in log space under a ``lax.cond``. Under ``vmap`` that becomes a
    select evaluating both forms; callers mapped in batches use
    :func:`_log_space_kde_log_density`. Without a sort in the graph, 1D tails
    are not scaled per point; callers expecting many tail points should use
    :func:`kde_log_density`.
    """
    return _scaled_kde_log_density(
        points, samples, std, block_size, weights, repair_in_graph=True
    )[0]


@partial(jax.jit, static_argnames=("block_size",))
def _compiled_scaled_kde_log_density(points, samples, std, block_size, weights, keys):
    """The scaled pass for :func:`kde_log_density`, which repairs on the host."""
    return _scaled_kde_log_density(
        points, samples, std, block_size, weights, keys, repair_in_graph=False
    )


def _batched_kde_rows(
    block_size: int, n_models: int, points, samples, std, weights
) -> tuple[int, int]:
    """Models per batch and query rows per block for batched KDE passes.

    A batch evaluates its models' blocks together, so their kernel matrices
    share one model's ``_KDE_BLOCK_KERNEL_BYTES`` budget: at most 16 models,
    each with an equal share of the rows one model alone would take.

    Parameters
    ----------
    block_size : int
        Requested rows per block.
    n_models : int
    points : array, shape (n_points, n_dims)
    samples : array, shape (n_samples, n_dims)
        One model's (padded) samples.
    std, weights : array-like
        One model's other kernel operands.

    Returns
    -------
    batch_size, rows : int
        ``batch_size * rows`` is at most the rows of one model's budget.
    """
    budget_rows = _kde_block_rows(block_size, points, samples, std, weights)
    batch_size = max(1, min(16, n_models, budget_rows))
    return batch_size, max(1, budget_rows // batch_size)


@partial(jax.jit, static_argnames=("block_size", "batch_size"))
def _compiled_batched_scaled_kde_log_density(
    points, samples, std, block_size, weights, keys, *, batch_size
):
    """The scaled pass for each of a stack of models, ``batch_size`` at a time.

    ``samples`` has shape ``(n_models, n_samples, n_dims)`` and ``weights``
    and ``keys`` (or None) ``(n_models, n_samples)``. Returns log densities and
    flags of shape ``(n_models, n_points)``.
    """

    def one_model(leaf):
        model_samples, model_weights, model_keys = leaf
        return _scaled_kde_log_density(
            points,
            model_samples,
            std,
            block_size,
            model_weights,
            model_keys,
            repair_in_graph=False,
        )

    return jax.lax.map(one_model, (samples, weights, keys), batch_size=batch_size)


_compiled_log_space_kde_log_density = jax.jit(
    _log_space_kde_log_density, static_argnames=("block_size",)
)
_KDE_PREPARATION_BYTES = 32 * 1024**2
_KDE_PREPARATION_ENTRIES = 256


def _prepared_kde_log_density(
    model, points, spike_time_order, *, predict_log=None, policy=""
):
    """Cache true density for immutable points and fitted leaves during prediction."""

    def predict():
        if predict_log is None:
            return kde_log_density(model, points)
        return predict_log(model, points)

    return spike_time_order.memo_kde(
        f"kde_log_density:{model.block_size}:{policy}",
        (model, model.samples_, model.weights_, model.std, points),
        predict,
    )


def _prepare_kde_local_inputs(
    model,
    position_time,
    position,
    environment,
    time_grid,
    row_start,
    row_stop,
    spike_time_order,
    *,
    predict_log=None,
    policy="",
):
    """Interpolate one row range and prepare its occupancy density once.

    The cache belongs to one immutable prediction. A new prediction rechecks
    original inputs, and replacing fitted leaves invalidates density entries.
    Keys retain strong source references; row chunks share a 32 MiB LRU budget.
    """
    points = spike_time_order.memo_kde(
        f"kde_local_position:{row_start}:{row_stop}",
        (position_time, position, environment, time_grid),
        lambda: get_position_at_time(
            position_time,
            position,
            decode_bin_centers(time_grid.edges, row_start, row_stop),
            environment,
        ),
    )
    return points, _prepared_kde_log_density(
        model, points, spike_time_order, predict_log=predict_log, policy=policy
    )


_compiled_position_log_kernel = jax.jit(_log_kernel_matrix)


def _prepared_kde_nonlocal_points(environment, spike_time_order):
    """Device spatial centers shared by states and chunks of this prediction."""
    return spike_time_order.memo_kde(
        "kde_nonlocal_points",
        (environment.place_bin_centers_, environment.is_track_interior_),
        lambda: jnp.asarray(
            environment.place_bin_centers_[environment.is_track_interior_.ravel()]
        ),
    )


def _prepared_kde_position_log_kernel(spike_time_order, points, samples, std):
    """Reuse small spatial kernels; larger ones stay inside the bounded scorer.

    One kernel may occupy at most one quarter of the prediction's preparation
    budget so multiple electrodes do not evict all local geometry/densities.
    Returning None retains the scorer's original compiled computation.
    """
    accumulator = jax.dtypes.canonicalize_dtype(jnp.float64)
    std_dtype = getattr(std, "dtype", None)
    if std_dtype is None:
        std_dtype = np.asarray(std).dtype
    itemsize = jnp.result_type(points, samples, std_dtype, accumulator).itemsize
    size = int(points.shape[0]) * int(samples.shape[0]) * itemsize
    if size > _KDE_PREPARATION_BYTES // 4:
        return None
    return spike_time_order.memo_kde(
        "kde_position_log_kernel",
        (points, samples, std),
        lambda: _compiled_position_log_kernel(
            points, samples, as_std_array(std, points.shape[1])
        ),
    )


def log_density_intensity(mean_rate, log_density, log_occupancy):
    """Log(rate * density / occupancy), with density ratios formed before rates.

    An unsupported occupancy or numerator gives zero signal, rather than NaN
    from subtracting two negative infinities. Finite tails and NaN diagnostics
    remain visible. Callers apply their final intensity floor after this ratio.
    """
    safe_occupancy = jnp.where(jnp.isneginf(log_occupancy), 0.0, log_occupancy)
    safe_rate = jnp.where(mean_rate == 0.0, 1.0, mean_rate)
    log_rate = jnp.where(mean_rate == 0.0, -jnp.inf, jnp.log(safe_rate))
    result = log_rate + (log_density - safe_occupancy)
    no_support = (
        jnp.isneginf(log_occupancy) | jnp.isneginf(log_density) | (mean_rate == 0.0)
    ) & ~jnp.isnan(result)
    return jnp.where(no_support, -jnp.inf, result)


def local_kde_log_ratio(
    positions,
    marks,
    encoding_positions,
    encoding_marks,
    position_std,
    mark_std,
    log_occupancy,
    weights,
    block_size,
):
    """Joint KDE/occupancy ratio, dividing position kernels before adding marks.

    Keeping the mark term separate preserves its normalizer when position log
    densities are large enough that adding it would round it away in float32.
    Evaluation blocking retains the shared KDE kernel-workspace limit.
    """
    n_position_dims = positions.shape[1]
    points = jnp.concatenate((positions, marks, log_occupancy[:, None]), axis=1)
    samples = jnp.concatenate((encoding_positions, encoding_marks), axis=1)
    std = jnp.concatenate((position_std, mark_std))

    def kernel(query, samples, std, weights):
        def supported(_):
            log_occ = query[:, -1]
            safe_occ = jnp.where(jnp.isneginf(log_occ), 0.0, log_occ)
            log_position = (
                _log_kernel_matrix(
                    query[:, :n_position_dims],
                    samples[:, :n_position_dims],
                    std[:n_position_dims],
                )
                - safe_occ[None, :]
            )
            log_marks = _log_kernel_matrix(
                query[:, n_position_dims:-1],
                samples[:, n_position_dims:],
                std[n_position_dims:],
            )
            safe_weights = jnp.where(weights > 0.0, weights, 1.0)
            log_weights = jnp.where(weights > 0.0, jnp.log(safe_weights), -jnp.inf)
            ratio = logsumexp(
                log_position + log_marks + log_weights[:, None], axis=0
            ) - jnp.log(jnp.sum(weights))
            return jnp.where(jnp.isneginf(log_occ), -jnp.inf, ratio)

        return jax.lax.cond(
            jnp.sum(weights) > 0.0,
            supported,
            lambda _: jnp.full(query.shape[0], -jnp.inf),
            None,
        )

    def blocked(query, samples, std, rows, weights):
        return jnp.concatenate(
            [
                kernel(query[start : start + rows], samples, std, weights)
                for start in range(0, query.shape[0], rows)
            ]
        )

    if points.shape[0] == 0:
        return jnp.zeros(0)
    return _traced_blocks(blocked, kernel, points, samples, std, block_size, weights)


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
    """Per-neuron spike counts as a ``(n_rows, n_neurons)`` matrix.

    Spikes are binned against the full ``time_edges`` and selected by
    ``row_slice`` as in ``get_spikecount_per_time_bin``, so the values equal its
    per-neuron counts. One ``bincount`` over ``row * n_neurons + neuron`` fills
    the row-major matrix in a single pass instead of stacking columns. ``n_rows``
    is ``n_bins`` unless ``row_slice`` is given.
    """
    if _spike_time_order is None:
        time_edges = validate_time_edges(time_edges)
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_rows, n_neurons = row_stop - row_start, len(spike_times)
    if not n_neurons:
        return np.zeros((n_rows, 0))
    flat_index = [
        select_spikes_in_rows(
            neuron_spike_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        ).bin_ind.astype(np.int64)
        * n_neurons
        + neuron
        for neuron, neuron_spike_times in enumerate(
            tqdm(spike_times, unit="cell", desc=desc, disable=disable_progress_bar)
        )
    ]
    return np.bincount(
        np.concatenate(flat_index), minlength=n_rows * n_neurons
    ).reshape(n_rows, n_neurons)


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


def log_nonnegative(x: jnp.ndarray) -> jnp.ndarray:
    """Log of a non-negative array: ``-inf`` at zero, ``NaN`` kept.

    Unlike ``safe_log`` it applies no floor, so a zero intensity stays an
    impossible event for the caller to combine (for example with a background
    by ``logaddexp``). The gradient is zero at zero instead of ``NaN``.

    Parameters
    ----------
    x : jnp.ndarray
        Non-negative values.

    Returns
    -------
    log_x : jnp.ndarray
        Same shape as ``x``.
    """
    positive = x > 0
    return jnp.where(
        positive,
        jnp.log(jnp.where(positive, x, 1.0)),
        jnp.where(jnp.isnan(x), jnp.nan, -jnp.inf),
    )


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
