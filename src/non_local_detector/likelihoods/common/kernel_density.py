"""Weighted Gaussian kernel density estimation."""

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import logsumexp
from numpy.typing import NDArray

from non_local_detector._jax_typing import finfo, optimization_barrier
from non_local_detector._validation import validate_weights
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common.numerics import LOG_EPS
from non_local_detector.likelihoods.common.padding import (
    pad_rows,
    padded_index,
    padded_repair_count,
    padded_sample_count,
    padded_tile_count,
)
from non_local_detector.likelihoods.common.position import (
    get_position_at_time,
    snap_off_track_positions,
)
from non_local_detector.likelihoods.common.reductions import (
    deterministic_row_add,
    deterministic_row_sum,
)
from non_local_detector.likelihoods.common.spike_rows import SpikeTimeOrder
from non_local_detector.likelihoods.common.validation import (
    as_std_array,
    validate_bandwidth,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    decode_bin_centers,
)


@jax.jit
def log_gaussian_pdf(x: jax.Array, mean: jax.Array, sigma: jax.Array) -> jax.Array:
    """Compute the log Gaussian probability density at ``x``.

    Elementwise and univariate: the arguments broadcast against each other and
    nothing is summed over dimensions. Kernel builders pass one dimension at a
    time, e.g. ``x`` of shape ``(1, n_eval)`` against ``mean`` of shape
    ``(n_samples, 1)`` for a ``(n_samples, n_eval)`` result.

    Parameters
    ----------
    x : jax.Array
        Input data, broadcastable with ``mean`` and ``sigma``.
    mean : jax.Array
        Mean of the Gaussian.
    sigma : jax.Array
        Standard deviation of the Gaussian.

    Returns
    -------
    log_pdf : jax.Array
        Shape ``broadcast_shapes(x.shape, mean.shape, sigma.shape)``.
    """
    # For singleton-shaped tiles XLA:CPU can rewrite division by a broadcast
    # scalar into multiplication by a rounded reciprocal, so tiled and untiled
    # KDEs disagree by about 2e-6 relative in Gaussian tails. Materializing the
    # divisor (only a vector here) keeps true division. Preserve true-division
    # promotion and the original normalization formula.
    divisor = sigma
    if x.size == 1 or mean.size == 1:
        shape = jnp.broadcast_shapes(x.shape, mean.shape, jnp.shape(sigma))
        divisor = optimization_barrier(jnp.broadcast_to(sigma, shape))
    return -0.5 * ((x - mean) / divisor) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def gaussian_pdf(x: jax.Array, mean: jax.Array, sigma: jax.Array) -> jax.Array:
    """Compute the Gaussian probability density at ``x``.

    Elementwise with broadcasting, like :func:`log_gaussian_pdf`; nothing is
    summed over dimensions.

    Parameters
    ----------
    x : jax.Array
        Input data, broadcastable with ``mean`` and ``sigma``.
    mean : jax.Array
        Mean of the Gaussian.
    sigma : jax.Array
        Standard deviation of the Gaussian.

    Returns
    -------
    pdf : jax.Array
        Shape ``broadcast_shapes(x.shape, mean.shape, sigma.shape)``.
    """
    return jnp.exp(log_gaussian_pdf(x, mean, sigma))


def _log_kernel_matrix(
    eval_points: jax.Array, samples: jax.Array, std: jax.Array
) -> jax.Array:
    """Log Gaussian kernel matrix summed over dimensions.

    Shared by :func:`kde` and :func:`log_kde` so both build the kernel the same
    way: accumulate per-dimension log densities, defer the single ``exp`` (or
    ``logsumexp``) to the caller.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval_points, n_dims)
        Points at which the kernels are evaluated.
    samples : jax.Array, shape (n_samples, n_dims)
        Kernel centers.
    std : jax.Array, shape (n_dims,)
        Kernel standard deviation in each dimension.

    Returns
    -------
    log_kernel : jax.Array, shape (n_samples, n_eval_points)
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
    eval_points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Kernel density estimation.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        Training samples.
    std : jax.Array, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    weights : jax.Array, shape (n_samples,)
        Weights for each sample.

    Returns
    -------
    density_estimate : jax.Array, shape (n_eval_points,)
        Weighted kernel density estimate at each evaluation point.

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
    eval_points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int = 100,
    weights: jax.Array | None = None,
) -> jax.Array:
    """Kernel density estimation split into blocks.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        Training samples.
    std : jax.Array, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Evaluation rows per block, by default 100. Reduced when needed so one
        block's (samples x rows) kernel matrix fits ``_KDE_BLOCK_KERNEL_BYTES``.
    weights : jax.Array, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    density_estimate : jax.Array, shape (n_eval_points,)
        Weighted kernel density estimate at each evaluation point.

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
        return cast(
            jax.Array,
            _compiled_block_kde(eval_points, samples, std, block_size, weights),
        )
    blocks = [
        kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval_points, block_size)
    ]
    return jnp.concatenate(blocks)


def padded_samples(
    spike_time_order: "SpikeTimeOrder",
    samples: np.ndarray | jax.Array,
    weights: np.ndarray | jax.Array | None,
    count: Callable[[int], int] = padded_sample_count,
) -> tuple[jax.Array, jax.Array]:
    """Device KDE samples and weights padded with zero-weight rows, once per prediction.

    Zero-weight samples add nothing to weighted kernel sums, so electrodes with
    nearby encoding sizes share compiled kernels.

    Parameters
    ----------
    spike_time_order : SpikeTimeOrder
        The prediction's preparation cache; results are reused per source arrays.
    samples : array, shape (n_samples, n_dims)
        Samples to pad: positions or waveform features.
    weights : array, shape (n_samples,), or None
        None means uniform weights.
    count : callable, optional
        Padded size for a sample count, by default ``padded_sample_count``.

    Returns
    -------
    samples, weights : jax.Array, shape (n_padded, n_dims) and (n_padded,)
    """

    def build() -> tuple[jax.Array, jax.Array]:
        n_samples = count(np.shape(samples)[0])
        sample_weights = (
            np.ones(np.shape(samples)[0]) if weights is None else np.asarray(weights)
        )
        return (
            jnp.asarray(pad_rows(samples, n_samples)),
            jnp.asarray(pad_rows(sample_weights, n_samples)),
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


def _kde_block_rows(
    block_size: int,
    eval_points: jax.Array | np.ndarray,
    samples: jax.Array | np.ndarray,
    std: jax.Array,
    weights: jax.Array | np.ndarray,
) -> int:
    """Return the evaluation rows per block: ``block_size``, reduced to the kernel budget.

    Parameters
    ----------
    block_size : int
        Requested rows per block.
    eval_points : jax.Array, shape (n_eval_points, n_dims)
        Points at which the kernels are evaluated.
    samples : jax.Array, shape (n_samples, n_dims)
        KDE sample positions.
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
    itemsize = _kernel_itemsize(eval_points, samples, std, weights)
    budget_rows = _KDE_BLOCK_KERNEL_BYTES // (max(samples.shape[0], 1) * itemsize)
    return min(block_size, max(1, budget_rows))


def _kernel_itemsize(*operands: Any) -> int:
    """Bytes per kernel matrix entry for these KDE operands.

    With the kernel's accumulator, the default float type (float64 under
    x64) that ``_log_kernel_matrix`` accumulates into, the operands set the
    kernel's dtype.
    """
    accumulator = jax.dtypes.canonicalize_dtype(jnp.float64)
    itemsize: int = jnp.result_type(*operands, accumulator).itemsize
    return itemsize


def _traced_blocks(
    blocked: Callable[..., jax.Array],
    kernel: Callable[..., jax.Array],
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
) -> jax.Array:
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
    return cast(jax.Array, values.reshape(-1)[:n_points])


def traced_block_kde(
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
) -> jax.Array:
    """``block_kde`` inside a trace, with a bounded graph for any row count.

    Each point's density depends only on that point, so the unrolled and
    looped forms compute the same values; compiled forms can round
    differently (XLA may multiply by a hoisted reciprocal of the bandwidth
    instead of dividing).

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        KDE sample positions.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth in each dimension.
    block_size : int
        Evaluation points per block.
    weights : jax.Array, shape (n_samples,)
        Sample weights.

    Returns
    -------
    density : jax.Array, shape (n_points,)
    """
    return _traced_blocks(block_kde, kde, points, samples, std, block_size, weights)


def _traced_block_log_kde(
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
) -> jax.Array:
    """``block_log_kde`` inside a trace, blocked like :func:`traced_block_kde`.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        KDE sample positions.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth in each dimension.
    block_size : int
        Evaluation points per block.
    weights : jax.Array, shape (n_samples,)
        Sample weights.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
    """
    return _traced_blocks(
        block_log_kde, log_kde, points, samples, std, block_size, weights
    )


# Eager block_kde/block_log_kde with more blocks than _UNROLLED_KDE_BLOCKS run
# these: one compiled loop over the blocks instead of one dispatch per block.
_compiled_block_kde = jax.jit(traced_block_kde, static_argnames=("block_size",))


_compiled_block_log_kde = jax.jit(
    _traced_block_log_kde, static_argnames=("block_size",)
)


def _relative_weights(
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weights relative to the largest, their logs, and their sum.

    Parameters
    ----------
    weights : jax.Array, shape (n_samples,)
        Weights that are not positive (including NaN) carry no mass.

    Returns
    -------
    relative_weights : jax.Array, shape (n_samples,)
        ``weights / max(weights)``; ratios below ``tiny`` flush to zero.
    log_weights : jax.Array, shape (n_samples,)
        Their logs, ``-inf`` without mass. A ratio that would flush takes its
        log from the weights themselves, so no positive weight is discarded.
    total : jax.Array, shape ()
        ``sum(relative_weights)``, at least one with any mass. Flushed ratios
        add less than ``n_samples * tiny`` to it.
    """
    max_weight = jnp.max(jnp.where(weights > 0, weights, 0.0), initial=0.0)
    relative_weights, log_weights = _weights_relative_to(weights, max_weight)
    return relative_weights, log_weights, jnp.sum(relative_weights)


def _weights_relative_to(
    weights: jax.Array, max_weight: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Weights relative to ``max_weight``, the largest positive weight, and their logs.

    The relative weights and logs of :func:`_relative_weights`, for a
    largest weight found elsewhere, such as over a model's other tiles.

    Parameters
    ----------
    weights : jax.Array, shape (n_samples,)
        Weights that are not positive (including NaN) carry no mass.
    max_weight : jax.Array, shape ()
        The largest positive weight of the model ``weights`` belong to, 0
        without one.

    Returns
    -------
    relative_weights, log_weights : jax.Array, shape (n_samples,)
    """
    positive = weights > 0
    positive_weights = jnp.where(positive, weights, 0.0)
    safe_max_weight = jnp.where(max_weight > 0, max_weight, 1.0)
    weight_scale = safe_max_weight
    # XLA can lower this division to multiplication by a reciprocal. For
    # weights near the dtype maximum that reciprocal is subnormal and may
    # flush to zero. A power-of-two prescale keeps both operands representable
    # without changing ordinary-weight arithmetic; the barrier prevents
    # cancellation of this protective scale during compiler reassociation.
    positive_weights, weight_scale = jax.lax.cond(
        weight_scale > 1.0 / finfo(positive_weights.dtype).tiny,
        lambda pair: optimization_barrier((pair[0] * 0.25, pair[1] * 0.25)),
        lambda pair: pair,
        (positive_weights, weight_scale),
    )
    relative_weights = positive_weights / weight_scale
    representable = relative_weights >= finfo(relative_weights.dtype).tiny
    log_weights = jnp.where(
        representable,
        jnp.log(jnp.where(representable, relative_weights, 1.0)),
        jnp.log(jnp.where(positive, weights, 1.0)) - jnp.log(safe_max_weight),
    )
    return relative_weights, jnp.where(positive, log_weights, -jnp.inf)


@jax.jit
def log_kde(
    eval_points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Compute the log kernel density estimate.

    The estimate is ``log p(x) = logsumexp_i [ log w_i + sum_d log N(x_d | s_{i,d},
    std_d) ] - logsumexp_i [log w_i]``.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samp, n_dims)
        Training samples.
    std : jax.Array, shape (n_dims,)
        Standard deviation of the Gaussian kernel in each dimension.
    weights : jax.Array, shape (n_samp,)
        Weight of each sample.

    Returns
    -------
    log_density : jax.Array, shape (n_eval,)
        Log kernel density estimate at each evaluation point.

    Notes
    -----
    With no samples or no positive weight it returns the finite floor
    ``LOG_EPS`` at every point (``kde_log_density`` returns ``-inf`` there). A
    point whose weighted kernel mass underflows to zero gets ``-inf``.
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

    def supported(_: None) -> jax.Array:
        """Evaluate the log density when the total weight is positive."""
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
    return cast(
        jax.Array,
        jax.lax.cond(
            jnp.any(positive),
            supported,
            lambda _: jnp.full(eval_points.shape[0], LOG_EPS),
            None,
        ),
    )


def block_log_kde(
    eval_points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int = 100,
    weights: jax.Array | None = None,
) -> jax.Array:
    """Log kernel density estimation split into blocks over eval points.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        Training samples.
    std : jax.Array, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Evaluation rows per block, by default 100. Reduced when needed so one
        block's (samples x rows) kernel matrix fits ``_KDE_BLOCK_KERNEL_BYTES``.
    weights : jax.Array, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    log_density_estimate : jax.Array, shape (n_eval_points,)
        Log of the weighted kernel density estimate at each evaluation point.
    """
    n_eval = eval_points.shape[0]

    if n_eval == 0:
        return jnp.full((n_eval,), LOG_EPS)

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block log-densities and concatenate once (see block_kde).
    block_size = _kde_block_rows(block_size, eval_points, samples, std, weights)
    if block_size > 0 and -(-n_eval // block_size) > _UNROLLED_KDE_BLOCKS:
        return cast(
            jax.Array,
            _compiled_block_log_kde(eval_points, samples, std, block_size, weights),
        )
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
        Training samples.
    weights : np.ndarray, shape (n_samples,)
        Non-negative weights.

    Returns
    -------
    samples : np.ndarray, shape (n_kept, ...)
        The samples with positive weight.
    weights : np.ndarray, shape (n_kept,)
        The positive weights.
    """
    keep = np.asarray(weights) > 0.0
    if not np.any(keep) or np.all(keep):
        return samples, weights
    return samples[keep], weights[keep]  # type: ignore[index]


@dataclass
class KDEModel:
    """Weighted Gaussian kernel density estimate with a diagonal bandwidth.

    Attributes
    ----------
    std : jax.Array, shape (n_dims,) or scalar
        Kernel standard deviation per dimension; a scalar applies to every
        dimension.
    block_size : int, optional
        Evaluation points per block; None (default) evaluates all at once.
    samples_ : jax.Array, shape (n_samples, n_dims)
        Training samples, set by :meth:`fit`.
    weights_ : jax.Array, shape (n_samples,)
        Sample weights, set by :meth:`fit` (ones when none are given).
    """

    std: jax.Array
    block_size: int | None = None
    samples_: jax.Array | None = field(init=False, default=None)
    weights_: jax.Array | None = field(init=False, default=None)

    def fit(
        self,
        samples: jax.Array | np.ndarray,
        weights: jax.Array | np.ndarray | None = None,
    ) -> "KDEModel":
        """Fit the model.

        Parameters
        ----------
        samples : jax.Array, shape (n_samples, n_dims)
            Training samples.
        weights : jax.Array, shape (n_samples,), optional
            Finite, non-negative sample weights, by default None (uniform).
            The density is normalized by their sum.

        Returns
        -------
        self : KDEModel
            The fitted model.

        Raises
        ------
        ValidationError
            If ``std`` is not finite and positive with shape ``()`` or
            ``(n_dims,)``, or ``weights`` is not finite, non-negative and of
            length ``n_samples``.
        """
        samples = jnp.asarray(samples)
        if samples.ndim == 1:
            samples = jnp.expand_dims(samples, axis=1)
        validate_bandwidth(self.std, samples.shape[1], "std")
        self.samples_ = samples
        if weights is None:
            self.weights_ = jnp.ones((samples.shape[0],))
        else:
            validate_weights(weights, samples.shape[0])
            self.weights_ = jnp.asarray(weights)

        return self

    def predict(self, eval_points: jax.Array) -> jax.Array:
        """Predict the density at the evaluation points.

        Parameters
        ----------
        eval_points : jax.Array, shape (n_eval_points, n_dims)
            Points at which to evaluate the density.

        Returns
        -------
        density : jax.Array, shape (n_eval_points,)
            Kernel density estimate at each evaluation point.
        """
        if self.samples_ is None:
            msg = "This KDE instance is not fitted yet."
            raise RuntimeError(msg)
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size, self.weights_)

    def predict_log(self, eval_points: jax.Array) -> jax.Array:
        """Predict the log density at the evaluation points.

        Log-density version of :meth:`predict`. A model without positive weight
        has zero density, so ``-inf``, as ``predict`` returns 0 and
        :func:`kde_log_density` returns ``-inf``.

        Parameters
        ----------
        eval_points : jax.Array, shape (n_eval_points, n_dims)
            Points at which to evaluate the log density.

        Returns
        -------
        log_density : jax.Array, shape (n_eval_points,)
            Log kernel density estimate at each evaluation point.
        """
        if self.samples_ is None:
            msg = "This KDE instance is not fitted yet."
            raise RuntimeError(msg)
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )
        values = block_log_kde(
            eval_points, self.samples_, std, block_size, self.weights_
        )
        assert self.weights_ is not None  # set together with samples_ in fit
        return jnp.where(jnp.sum(self.weights_) > 0.0, values, -jnp.inf)


# Pickles name a class by its module. Detectors saved before likelihoods.common
# became a package name ``non_local_detector.likelihoods.common.KDEModel``;
# keeping that name lets releases before the split load detectors saved now.
# The package re-exports the class, so the name resolves here too.
KDEModel.__module__ = "non_local_detector.likelihoods.common"


def kde_log_density(model: KDEModel, points: jax.Array | np.ndarray) -> jax.Array:
    """Compute the true log density, with zero total fitted weight as zero mass.

    No samples or zero total weight gives ``-inf`` at every point, unlike
    :func:`log_kde`, which returns the finite floor ``LOG_EPS``.

    Runs outside jit. One compiled pass scales each point by an upper bound
    on its largest log kernel, exact for 1D points; points whose pass could
    have lost mass are then evaluated in log space (see
    :func:`_scaled_kde_log_density`).

    Parameters
    ----------
    model : KDEModel
        A fitted model.
    points : jax.Array, shape (n_points, n_dims)
        Points at which to evaluate the log density.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
        Log density at each point.
    """
    if type(model) is KDEModel:
        if model.samples_ is None:
            msg = "This KDE instance is not fitted yet."
            raise RuntimeError(msg)
        assert model.weights_ is not None  # set together with samples_ in fit
        points = jnp.asarray(points)
        if points.ndim == 1:
            points = points[:, None]
        return true_kde_log_density(
            points,
            model.samples_,
            as_std_array(model.std, points.shape[1]),
            points.shape[0] if model.block_size is None else model.block_size,
            model.weights_,
            _sorted_sample_keys(points, model.samples_, model.weights_),
        )
    # Preserve custom KDE subclasses' predict_log implementations.
    values = model.predict_log(cast(jax.Array, points))
    total = jnp.sum(cast(jax.Array, model.weights_))
    return jnp.where(
        jnp.isnan(total), jnp.nan, jnp.where(total > 0.0, values, -jnp.inf)
    )


def true_kde_log_density(
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
    keys: jax.Array | None = None,
    *,
    n_valid: int | None = None,
) -> jax.Array:
    """Return the true log density of fitted KDE leaves, outside jit.

    One compiled scaled pass (see :func:`_scaled_kde_log_density`), then the
    flagged points in log space. Without ``keys`` points are scaled by the
    kernel's peak, which suits points expected near the samples.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        KDE sample positions.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth in each dimension.
    block_size : int
        Evaluation points per block.
    weights : jax.Array, shape (n_samples,)
        Sample weights.
    keys : jax.Array, shape (n_samples,), optional
        From :func:`_sorted_sample_keys`.
    n_valid : int, optional
        Number of leading rows of ``points`` that are evaluation points. The
        rows after them are padding: they keep their scaled-pass values,
        which may be wrong or ``-inf``, and are never re-evaluated in log
        space. By default every row is an evaluation point.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
    """
    values, needs_log_space = _compiled_scaled_kde_log_density(
        points, samples, std, block_size, weights, keys
    )
    needs_log_space = np.asarray(needs_log_space)
    if n_valid is not None:
        needs_log_space = needs_log_space & (np.arange(needs_log_space.size) < n_valid)
    return _repair_in_log_space(
        values, needs_log_space, points, samples, std, block_size, weights
    )


def _repair_in_log_space(
    values: jax.Array,
    needs_log_space: NDArray[np.bool_],
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
) -> jax.Array:
    """Replace the flagged points of a scaled pass with log-space values.

    The flagged points are gathered and written back through an index padded
    to few sizes (see ``padded_index``), so repairs of different numbers of
    points share compiled kernels; the padding's writes are dropped.

    Parameters
    ----------
    values : jax.Array, shape (n_points,)
        Log densities from the scaled pass.
    needs_log_space : np.ndarray of bool, shape (n_points,)
        Points to recompute in log space.
    points, samples, std, block_size, weights
        The pass's inputs.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
    """
    if not needs_log_space.any():
        return values
    index = np.flatnonzero(needs_log_space)
    gather = padded_index(index)
    exact = _compiled_log_space_kde_log_density(
        points[gather], samples, std, block_size, weights
    )
    # Out-of-range targets drop the padding's writes.
    scatter = np.where(np.arange(gather.size) < index.size, gather, values.shape[0])
    return values.at[scatter].set(exact, mode="drop")


def kde_log_densities(
    models: Sequence["KDEModel"], points: jax.Array | np.ndarray
) -> list[jax.Array]:
    """Return the true log densities of several fitted KDEs at the same points.

    ``kde_log_density`` for each model, with few compiled kernels: models
    with the same block size and bandwidth share one pass over tiles of
    ``_KDE_SAMPLE_TILE`` samples (zero-weight padding rows add no mass), so
    the kernels compile for the total number of tiles, not for each model's
    sample count; flagged points are then evaluated in log space in one
    more tiled pass (see :func:`_tiled_kde_log_densities`). Values match
    ``kde_log_density`` to rounding. Runs outside jit; custom subclasses use
    ``kde_log_density``.

    Parameters
    ----------
    models : sequence of KDEModel
        Fitted KDE models.
    points : jax.Array, shape (n_points, n_dims) or (n_points,)
        Points at which every model is evaluated.

    Returns
    -------
    log_densities : list of jax.Array, shape (n_points,)
        In the order of ``models``.
    """
    return [
        jnp.asarray(row) for row in _kde_log_density_list(models, jnp.asarray(points))
    ]


# Per-model and per-point terms of a log-density floor, added: see
# kde_log_density_rows.
LogFloor = tuple[np.ndarray, np.ndarray]


def kde_log_density_rows(
    models: Sequence["KDEModel"],
    points: jax.Array | np.ndarray,
    *,
    log_floor: LogFloor | None = None,
    tiles: dict[tuple[int, ...], "_GroupTiles"] | None = None,
) -> np.ndarray:
    """:func:`kde_log_densities` as one host array, one row per model.

    Parameters
    ----------
    models : sequence of KDEModel
        Fitted KDE models.
    points : array, shape (n_points, n_dims) or (n_points,)
        Points at which every model is evaluated.
    log_floor : (np.ndarray, np.ndarray), shapes (n_models,) and (n_points,), optional
        Per-model and per-point terms whose sum is the log density below
        which the caller does not need a model's value at a point. A flagged
        point is not evaluated in log space when its density is certainly
        below its floor (see :func:`_tiled_scaled_kde_log_densities`): it
        keeps its scaled-pass value, which is then below the floor too,
        possibly ``-inf``. A point term of ``inf`` marks a padding point; a
        NaN floor (``inf`` plus ``-inf``) leaves the point flagged. By
        default every flagged point is evaluated in log space. Models that
        are not exactly ``KDEModel`` ignore it.
    tiles : dict, optional
        The caller's store of the models' sample tiles and sorted 1D keys,
        filled on first use: calls that pass the same dict with the same
        ``models`` and point dimensions build and move them to the device
        once. By default each call builds its own.

    Returns
    -------
    log_densities : np.ndarray, shape (n_models, n_points)
        In the order of ``models``, in their common dtype.
    """
    points = jnp.asarray(points)
    if points.ndim == 1:
        points = points[:, None]
    results = _kde_log_density_list(models, points, log_floor, tiles)
    if not results:
        return np.empty(
            (0, points.shape[0]), jax.dtypes.canonicalize_dtype(jnp.float64)
        )
    return np.stack([np.asarray(row) for row in results])


# (model, point) arrays the tiled pass holds at once, in units of the float
# itemsize. 1-D: the per-point scale and two more temporaries, the two
# bracketing keys, the values and the flags (measured 6.3 on CPU in float32
# and float64); 2-D and up: one temporary, the values and the flags
# (measured 2.3).
_PASS_POINT_ARRAYS_1D = 7
_PASS_POINT_ARRAYS = 3

# The pass's smaller temporaries beside its tile buffers (measured 1 to 6 MiB).
_PASS_SLACK_BYTES = _KDE_BLOCK_KERNEL_BYTES // 32


def _tiled_pass_bytes(
    n_points: int,
    block_size: int,
    n_models: int,
    n_dims: int,
    n_tiles: int,
    widest: int,
    std: jax.Array,
    sample_dtype: np.dtype,
    weight_dtype: np.dtype,
) -> tuple[int, int]:
    """Bytes the tiled pass of ``n_models`` holds at ``n_points``, and its point block.

    Counts its tile buffers as :func:`_tiled_kde_rows` sizes them (the
    per-tile sums of a block of points and one step's kernel matrix), its
    tile arguments, its ``(n_models, n_points)`` arrays and
    ``_PASS_SLACK_BYTES``.
    """
    # Zero-stride views: only shapes and dtypes are read.
    points = np.broadcast_to(
        np.zeros((), jax.dtypes.canonicalize_dtype(jnp.float64)), (n_points, n_dims)
    )
    tile_samples = np.zeros((1, _KDE_SAMPLE_TILE, n_dims), sample_dtype)
    tile_weights = np.zeros((1, _KDE_SAMPLE_TILE), weight_dtype)
    rows, batch_size, n_pass_tiles = _tiled_kde_rows(
        block_size, points, n_tiles, std, tile_samples, tile_weights, widest
    )
    itemsize = _kernel_itemsize(points, tile_samples, std, tile_weights)
    point_arrays = _PASS_POINT_ARRAYS_1D if n_dims == 1 else _PASS_POINT_ARRAYS
    n_bytes = (
        itemsize * rows * (n_pass_tiles + batch_size * _KDE_SAMPLE_TILE)
        + itemsize * n_pass_tiles * _KDE_SAMPLE_TILE * (n_dims + 1)
        + 4 * n_pass_tiles
        + itemsize * point_arrays * n_models * n_points
        + _PASS_SLACK_BYTES
    )
    return n_bytes, rows


def tiled_kde_block(
    models: Sequence["KDEModel"], n_dims: int, max_points: int
) -> tuple[int, int]:
    """Points per call, and per point block, at which :func:`kde_log_density_rows` fits the budget.

    The call's points take what ``_KDE_BLOCK_KERNEL_BYTES`` leaves beside the
    pass's own tile buffers and arguments, counting every
    ``(n_models, n_points)`` array the pass holds (see
    :func:`_tiled_pass_bytes`). The point count is then split into whole
    point blocks: ``n_points = m * block`` with ``m`` blocks of at most the
    pass's own block, which loses fewer than ``m`` points of the largest
    count that fits. Evaluated with block size ``block``, the pass pads no
    points. Exact when the models form one group (bandwidth, block size and
    dtypes), as one fit's do; otherwise it is computed as if they did.

    Parameters
    ----------
    models : sequence of KDEModel
        Fitted KDE models, with the bandwidth and block size of the first.
    n_dims : int
        Point dimensions.
    max_points : int
        Most points per call.

    Returns
    -------
    n_points, block : int
        ``n_points <= max_points`` is a multiple of ``block``; at least one
        point when even one exceeds the budget.
    """
    if not models:
        return max_points, max_points
    model = models[0]
    assert model.samples_ is not None  # fitted
    assert model.weights_ is not None  # set with samples_
    model_tiles = [
        max(1, -(-np.shape(cast(jax.Array, m.samples_))[0] // _KDE_SAMPLE_TILE))
        for m in models
    ]

    sample_dtype, weight_dtype = model.samples_.dtype, model.weights_.dtype
    std = as_std_array(model.std, n_dims)

    def pass_bytes(n_points: int) -> tuple[int, int]:
        return _tiled_pass_bytes(
            n_points,
            n_points if model.block_size is None else model.block_size,
            len(models),
            n_dims,
            padded_tile_count(sum(model_tiles)),
            max(model_tiles),
            std,
            sample_dtype,
            weight_dtype,
        )

    # The bytes grow with the points: the largest count that fits, by
    # bisection.
    fits, too_many = 1, max_points + 1
    if pass_bytes(max_points)[0] <= _KDE_BLOCK_KERNEL_BYTES:
        fits = max_points
    else:
        while too_many - fits > 1:
            middle = (fits + too_many) // 2
            if pass_bytes(middle)[0] <= _KDE_BLOCK_KERNEL_BYTES:
                fits = middle
            else:
                too_many = middle
    point_block = pass_bytes(fits)[1]
    n_blocks = -(-fits // point_block)
    block = fits // n_blocks
    return n_blocks * block, block


def _kde_log_density_list(
    models: Sequence["KDEModel"],
    points: jax.Array,
    log_floor: LogFloor | None = None,
    tiles: dict[tuple[int, ...], "_GroupTiles"] | None = None,
) -> list[np.ndarray | jax.Array]:
    """Each model's log densities in its own dtype; see :func:`kde_log_density_rows`."""
    if points.ndim == 1:
        points = points[:, None]
    results: list[np.ndarray | jax.Array] = [None] * len(models)  # type: ignore[list-item]  # every slot is filled before return
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, model in enumerate(models):
        if type(model) is not KDEModel:
            results[index] = kde_log_density(model, points)
            continue
        if model.samples_ is None:
            msg = "This KDE instance is not fitted yet."
            raise RuntimeError(msg)
        assert model.weights_ is not None  # set together with samples_ in fit
        std = np.asarray(as_std_array(model.std, points.shape[1]))
        key = (
            points.shape[0] if model.block_size is None else model.block_size,
            std.dtype.str,
            std.tobytes(),
            np.dtype(model.samples_.dtype).str,
            np.dtype(model.weights_.dtype).str,
        )
        groups.setdefault(key, []).append(index)
    if tiles is None:
        tiles = {}
    for (block_size, *_), members in groups.items():
        tiles_key = (points.shape[1], *members)
        if tiles_key not in tiles:
            tiles[tiles_key] = _GroupTiles.build(
                [np.asarray(models[i].samples_) for i in members],
                [np.asarray(models[i].weights_) for i in members],
                one_dimensional=points.shape[1] == 1,
            )
        group = tiles[tiles_key]
        values = _tiled_kde_log_densities(
            points,
            group,
            as_std_array(models[members[0]].std, points.shape[1]),
            block_size,
            log_floor=None
            if log_floor is None
            else (np.asarray(log_floor[0])[members], log_floor[1]),
        )
        for row, index in enumerate(members):
            results[index] = values[row]
    return results


def log_space_kde_log_density(
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
) -> jax.Array:
    """Return the true log density evaluated entirely in log space.

    Re-evaluates the points a scaled pass flags; traced callers use it
    directly.
    """
    if points.shape[0] == 0:
        return jnp.full((0,), LOG_EPS)
    values = _traced_block_log_kde(points, samples, std, block_size, weights)
    total = jnp.sum(weights)
    return jnp.where(
        jnp.isnan(total), jnp.nan, jnp.where(total > 0.0, values, -jnp.inf)
    )


def _sorted_sample_keys(
    points: jax.Array, samples: jax.Array | np.ndarray, weights: jax.Array
) -> jax.Array | None:
    """Sorted 1D sample positions for :func:`_kde_log_scale`, or None.

    Samples without positive weight become ``inf``, which sorts last. The
    sort runs on the host: for 4k-20k elements XLA's GPU sort took 0.7-1.6 s
    to compile for each size (A100, JAX 0.9.0).

    Parameters
    ----------
    points : array, shape (n_points, n_dims)
        Evaluation points; only their dimensionality is used.
    samples : array, shape (n_samples, n_dims)
        KDE sample positions.
    weights : array, shape (n_samples,)
        Sample weights; samples without positive weight sort last.

    Returns
    -------
    keys : jax.Array, shape (n_samples,), or None
        None unless the points are 1D and there are samples.
    """
    if points.shape[1] != 1 or samples.shape[0] == 0:
        return None
    samples = np.asarray(samples)
    keys = np.where(np.asarray(weights) > 0, samples[:, 0], np.inf)
    return jnp.asarray(np.sort(keys.astype(samples.dtype, copy=False)))


def _nearest_sample_log_kernel(
    points: jax.Array, keys: jax.Array, std: jax.Array
) -> jax.Array:
    """Each 1D point's largest log kernel, at its nearest key by binary search.

    Parameters
    ----------
    points : jax.Array, shape (n_points, 1)
        Evaluation points.
    keys : jax.Array, shape (n_samples,)
        Sorted positions of the samples with positive weight, ``inf`` for
        the rest.
    std : jax.Array, shape (1,)
        Kernel standard deviation.

    Returns
    -------
    log_kernel : jax.Array, shape (n_points,)
        ``-inf`` where no key is finite.
    """
    index = jnp.searchsorted(keys, points[:, 0])
    last = keys.shape[0] - 1
    left = keys[jnp.clip(index - 1, 0, last)]
    right = keys[jnp.clip(index, 0, last)]
    return _nearer_key_log_kernel(points[:, 0], left, right, std)


def _nearer_key_log_kernel(
    points: jax.Array, left: jax.Array, right: jax.Array, std: jax.Array
) -> jax.Array:
    """Log kernel of each 1D point at the nearer of its two bracketing keys.

    Parameters
    ----------
    points : jax.Array, shape (..., n_points)
        Evaluation coordinates.
    left, right : jax.Array, shape (..., n_points)
        The keys just below and at or above each point, as
        :func:`_nearest_sample_log_kernel` finds them; ties go left.
    std : jax.Array, shape (1,)
        Kernel standard deviation.

    Returns
    -------
    log_kernel : jax.Array, shape (..., n_points)
    """
    nearest = jnp.where(jnp.abs(points - left) <= jnp.abs(points - right), left, right)
    return cast(jax.Array, log_gaussian_pdf(points, nearest, std[0]))


def _sorted_model_keys(
    samples: Sequence[np.ndarray], weights: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Each model's sorted 1D keys for :func:`_bracketing_keys`, on the host.

    As :func:`_sorted_sample_keys` per model, in the samples' common dtype (a
    floating one for integer samples): samples without positive weight are
    ``inf``.

    Parameters
    ----------
    samples : sequence of np.ndarray, each shape (n_samples, 1)
        Each model's samples.
    weights : sequence of np.ndarray, each shape (n_samples,)
        Each model's weights.

    Returns
    -------
    keys : list of np.ndarray, each shape (n_samples,)
    """
    key_dtype = np.result_type(*(np.asarray(s).dtype for s in samples), np.float16)
    return [
        np.sort(
            np.where(model_weights > 0, model_samples[:, 0], np.inf).astype(
                key_dtype, copy=False
            )
        )
        for model_samples, model_weights in zip(samples, weights, strict=True)
    ]


@jax.jit
def _bracketing_keys(
    sorted_points: jax.Array,
    order: jax.Array,
    keys: jax.Array,
    key_model: jax.Array,
    key_start: jax.Array,
    key_count: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Each model's sorted 1D keys just below and at or above each point, on the device.

    The neighbours :func:`_nearest_sample_log_kernel` finds, for several
    models at once. A point's index among a model's sorted keys is the
    number of its keys below the point: a running count, over the sorted
    points, of where each key falls among them (one search per key, and an
    integer count, so the same keys as a search per point; NaN points sort
    last and count every key, as in NumPy).

    Parameters
    ----------
    sorted_points : jax.Array, shape (n_points,)
        Evaluation coordinates in ascending order (NumPy's order: NaN last).
    order : jax.Array of int32, shape (n_points,)
        The position of each sorted point among the points.
    keys : jax.Array, shape (n_keys,)
        Every model's sorted keys (see :func:`_sorted_model_keys`), model
        after model; at least one.
    key_model : jax.Array of int32, shape (n_keys,)
        Each key's model.
    key_start, key_count : jax.Array of int32, shape (n_models,)
        Each model's first key and number of keys.

    Returns
    -------
    left, right : jax.Array, shape (n_models, n_points)
        In the points' order; ``inf`` for a model without keys.
    """
    n_models, n_points = key_start.shape[0], sorted_points.shape[0]
    where = jnp.searchsorted(sorted_points, keys, side="right")
    below = (
        jnp.zeros((n_models, n_points + 1), jnp.int32)
        .at[key_model, where]
        .add(1, indices_are_sorted=True)
    )
    index = jnp.cumsum(below, axis=1)[:, :n_points]
    last = (key_count - 1)[:, None]
    empty = (key_count == 0)[:, None]

    def neighbour(position: jax.Array) -> jax.Array:
        found = keys[key_start[:, None] + jnp.clip(position, 0, last)]
        in_sorted_order = jnp.where(empty, jnp.inf, found)
        return jnp.zeros_like(in_sorted_order).at[:, order].set(in_sorted_order)

    return neighbour(index - 1), neighbour(index)


def _kde_log_scale(
    points: jax.Array, std: jax.Array, keys: jax.Array | None = None
) -> jax.Array:
    """Per-point upper bound on the largest log kernel over positive weights.

    With sorted 1D ``keys`` (:func:`_sorted_sample_keys`) the nearest
    positive-weight sample gives the exact maximum. Otherwise, and for points
    with no finite bound (no positive weights, infinite coordinates), the
    bound is the kernel's peak.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims)
        Evaluation points.
    std : jax.Array, shape (n_dims,)
        Kernel standard deviation in each dimension.
    keys : jax.Array, shape (n_samples,), optional
        Sorted 1D sample keys; None bounds by the kernel's peak.

    Returns
    -------
    log_scale : jax.Array, shape (n_points,)
    """
    peak = _kernel_log_peak(std)
    if keys is None:
        return jnp.full(points.shape[0], peak)
    return _bounded_log_scale(_nearest_sample_log_kernel(points, keys, std), peak)


def _kernel_log_peak(std: jax.Array) -> jax.Array:
    """Log of the kernel's peak, its value at zero distance; ``std`` shape (n_dims,)."""
    return cast(
        jax.Array,
        sum(log_gaussian_pdf(jnp.zeros(()), jnp.zeros(()), dim_std) for dim_std in std),
    )


def _bounded_log_scale(scale: jax.Array, peak: jax.Array) -> jax.Array:
    """``scale`` where finite, else the kernel's log ``peak``; no gradient."""
    return jax.lax.stop_gradient(jnp.where(jnp.isfinite(scale), scale, peak))


def _scaled_kde_log_density(
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
    keys: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the true log density in one pass per block, each point scaled first.

    Each point's log kernels are shifted by :func:`_kde_log_scale` before
    exponentiating, then summed by a weighted product as the probability-space
    KDE does. Weights enter relative to the largest, so neither weights nor
    their sum overflow. A term is lost only if it flushes below ``tiny``;
    points whose scaled sum is small enough for that to matter are flagged.
    Log space (:func:`log_kde`) offsets each point by its largest weighted
    term, which recovers what either scale loses.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        KDE sample positions.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth in each dimension.
    block_size : int
        Evaluation points per block.
    weights : jax.Array, shape (n_samples,)
        Sample weights.
    keys : jax.Array, shape (n_samples,), optional
        Sorted 1D keys for an exact per-point scale; None scales by the
        kernel's peak.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
    needs_log_space : jax.Array, shape (n_points,)
        The flagged points, which callers re-evaluate in log space.
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

    def block_values(block: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
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
            samples.shape[0] * finfo(scaled.dtype).tiny / (4 * finfo(scaled.dtype).eps)
        )
        needs_log_space = scaled < lost
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


@partial(jax.jit, static_argnames=("block_size",))
def _compiled_scaled_kde_log_density(
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
    keys: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Compute the scaled pass for :func:`kde_log_density`, which repairs on the host."""
    return _scaled_kde_log_density(points, samples, std, block_size, weights, keys)


# Samples per tile of a batched pass (see kde_log_densities). Each model's
# samples are cut into tiles of this many rows, the last padded with
# zero-weight rows, so a pass over models of any mix of sample counts has one
# shape per number of tiles.
_KDE_SAMPLE_TILE = 256


def _sample_tiles(
    samples: Sequence[np.ndarray], weights: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut each model's samples into zero-weight padded tiles, on the host.

    Parameters
    ----------
    samples : sequence of np.ndarray, each shape (n_samples, n_dims)
        Each model's samples.
    weights : sequence of np.ndarray, each shape (n_samples,)
        Each model's weights.

    Returns
    -------
    tile_samples : np.ndarray, shape (n_tiles, _KDE_SAMPLE_TILE, n_dims)
    tile_weights : np.ndarray, shape (n_tiles, _KDE_SAMPLE_TILE)
        Zero on padding rows.
    tile_model : np.ndarray of int32, shape (n_tiles,)
        The model of each tile, nondecreasing. A model without samples has
        one tile of zero weight.
    """
    size = _KDE_SAMPLE_TILE
    n_tiles = [max(1, -(-np.shape(s)[0] // size)) for s in samples]
    n_dims = np.shape(samples[0])[1]
    tile_samples = np.concatenate(
        [pad_rows(s, n * size) for s, n in zip(samples, n_tiles, strict=True)]
    ).reshape(sum(n_tiles), size, n_dims)
    tile_weights = np.concatenate(
        [pad_rows(w, n * size) for w, n in zip(weights, n_tiles, strict=True)]
    ).reshape(sum(n_tiles), size)
    tile_model = np.repeat(np.arange(len(samples), dtype=np.int32), n_tiles)
    return tile_samples, tile_weights, tile_model


def _bucketed_tiles(
    tile_samples: np.ndarray,
    tile_weights: np.ndarray,
    tile_model: np.ndarray,
    n_padded: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Move every model's tiles to the device, padded to ``n_padded`` tiles.

    Padding tiles have zero weight and the model id ``n_models``, which the
    per-model reductions drop, so passes over different sample counts share
    compiled kernels.

    Parameters
    ----------
    tile_samples, tile_weights, tile_model : np.ndarray
        From :func:`_sample_tiles`.
    n_padded : int
        Number of tiles returned, at least ``tile_model.size``.

    Returns
    -------
    tile_samples, tile_weights, tile_model : jax.Array
        ``n_padded`` tiles; ``tile_model`` is nondecreasing.
    """
    n_models = int(tile_model[-1]) + 1
    padding = n_padded - tile_model.size
    return (
        jnp.asarray(pad_rows(tile_samples, n_padded)),
        jnp.asarray(pad_rows(tile_weights, n_padded)),
        jnp.asarray(
            np.concatenate([tile_model, np.full(padding, n_models, tile_model.dtype)])
        ),
    )


def _tiled_kde_rows(
    block_size: int,
    points: jax.Array | np.ndarray,
    n_tiles: int,
    std: jax.Array,
    tile_samples: jax.Array | np.ndarray,
    tile_weights: jax.Array | np.ndarray,
    widest_tiles: int,
) -> tuple[int, int, int]:
    """Query rows per block, tiles per step and tiles of the tiled scaled pass.

    Each block of ``rows`` points holds per-tile sums of shape
    ``(n_tiles, rows)``, which take at most a quarter of
    ``_KDE_BLOCK_KERNEL_BYTES``. One step evaluates ``batch_size`` tiles at
    these points. Its kernel matrix takes at most half the budget and has no
    more entries than the widest model's own block, but at least one tile's.
    The compiled workspace then holds a few such matrices, as a single-model
    pass of the widest model does. The pass's
    tiles are a whole number of equal steps: a last, smaller step would
    compile a second batched program beside the first, with its own
    workspace. The steps are evened out, so this adds fewer zero-weight
    tiles than there are steps.

    Parameters
    ----------
    block_size : int
        Requested rows per block.
    points : array, shape (n_points, n_dims)
        Evaluation points.
    n_tiles : int
        Tiles to evaluate, padding included.
    std : jax.Array
        Kernel bandwidth.
    tile_samples, tile_weights : array
        From :func:`_sample_tiles`; only their dtypes are used.
    widest_tiles : int
        Tiles of the model with the most.

    Returns
    -------
    rows, batch_size : int
        At least one each.
    n_tiles : int
        ``n_tiles`` rounded up to a multiple of ``batch_size``, by fewer
        zero-weight tiles than the ``n_tiles / batch_size`` steps.
    """
    budget = _KDE_BLOCK_KERNEL_BYTES // _kernel_itemsize(
        points, tile_samples, std, tile_weights
    )
    size = _KDE_SAMPLE_TILE
    rows = max(1, min(block_size, points.shape[0], budget // (4 * max(n_tiles, size))))
    step = min(budget // 2, widest_tiles * size * rows)
    largest_batch = max(1, min(n_tiles, step // (size * rows)))
    n_steps = -(-n_tiles // largest_batch)
    batch_size = -(-n_tiles // n_steps)
    return rows, batch_size, n_steps * batch_size


def _tiled_model_reduce(
    tile_terms: Callable[..., Any],
    tiles: tuple[jax.Array, ...],
    tile_model: jax.Array,
    n_models: int,
    points: jax.Array,
    model_values: jax.Array | None,
    combine: Callable[[jax.Array, jax.Array, int], jax.Array],
    rows: int,
    batch_size: int,
) -> Any:
    """Reduce per-tile terms over each model's tiles, block by block of points.

    Parameters
    ----------
    tile_terms : callable
        ``tile_terms(tile, block_points, block_values)`` returns a pytree of
        arrays of shape ``(rows,)`` for one tile, where ``tile`` holds one
        tile of each of ``tiles`` and ``block_values`` is the tile's model's
        row of ``model_values`` (None without them).
    tiles : tuple of jax.Array, each shape (n_tiles, ...)
        Per-tile operands.
    tile_model : jax.Array, shape (n_tiles,)
        Each tile's model, nondecreasing.
    n_models : int
        Number of models.
    points : jax.Array, shape (n_points, n_dims), n_points >= 1
        Evaluation points; the last block repeats the last point.
    model_values : jax.Array, shape (n_models, n_points), or None
        Per-model values at each point that ``tile_terms`` reads.
    combine : callable
        ``combine(terms, tile_model, n_models)`` reduces ``(n_tiles, rows)``
        tile terms to ``(n_models, rows)``.
    rows, batch_size : int
        From :func:`_tiled_kde_rows`.

    Returns
    -------
    reduced : pytree of jax.Array, each shape (n_models, n_points)
    """
    n_points = points.shape[0]
    n_blocks = -(-n_points // rows)
    padding = n_blocks * rows - n_points
    block_points = jnp.pad(points, ((0, padding), (0, 0)), mode="edge").reshape(
        n_blocks, rows, -1
    )
    block_values = (
        None
        if model_values is None
        else jnp.pad(model_values, ((0, 0), (0, padding)), mode="edge")
        .reshape(n_models, n_blocks, rows)
        .transpose(1, 0, 2)
    )

    def block(xs: tuple[jax.Array, jax.Array | None]) -> Any:
        points_in_block, values_in_block = xs

        def one_tile(tile: tuple[jax.Array, ...]) -> Any:
            *operands, model = tile
            return tile_terms(
                tuple(operands),
                points_in_block,
                None if values_in_block is None else values_in_block[model],
            )

        terms = jax.lax.map(one_tile, (*tiles, tile_model), batch_size=batch_size)
        return jax.tree_util.tree_map(
            lambda term: combine(term, tile_model, n_models), terms
        )

    reduced = jax.lax.map(block, (block_points, block_values))
    return jax.tree_util.tree_map(
        lambda values: values.transpose(1, 0, 2).reshape(n_models, -1)[:, :n_points],
        reduced,
    )


def _sum_over_tiles(
    terms: jax.Array, tile_model: jax.Array, n_models: int
) -> jax.Array:
    """Each model's sum of its tiles' terms, in a fixed order on every backend."""
    return deterministic_row_sum(terms, tile_model, n_models, indices_are_sorted=True)


def _max_over_tiles(
    terms: jax.Array, tile_model: jax.Array, n_models: int
) -> jax.Array:
    """Each model's largest tile term; a maximum does not depend on order."""
    return jax.ops.segment_max(
        terms, tile_model, num_segments=n_models, indices_are_sorted=True
    )


def _tile_weight_terms(
    tile_weights: jax.Array, tile_model: jax.Array, n_models: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute :func:`_relative_weights` of each model from its tiles.

    Returns
    -------
    relative_weights, log_weights : jax.Array, shape (n_tiles, _KDE_SAMPLE_TILE)
        Relative to the model's largest positive weight.
    relative_total, total : jax.Array, shape (n_models,)
        Each model's sum of relative and of given weights.
    """
    tile_max = jnp.max(jnp.where(tile_weights > 0, tile_weights, 0.0), axis=1)
    max_weight = _max_over_tiles(tile_max, tile_model, n_models)
    relative_weights, log_weights = jax.vmap(_weights_relative_to)(
        tile_weights, max_weight[tile_model]
    )
    return (
        relative_weights,
        log_weights,
        _sum_over_tiles(jnp.sum(relative_weights, axis=1), tile_model, n_models),
        _sum_over_tiles(jnp.sum(tile_weights, axis=1), tile_model, n_models),
    )


@partial(jax.jit, static_argnames=("rows", "batch_size"))
def _tiled_scaled_kde_log_densities(
    points: jax.Array,
    tile_samples: jax.Array,
    tile_weights: jax.Array,
    tile_model: jax.Array,
    counts: jax.Array,
    std: jax.Array,
    bracketing_keys: tuple[jax.Array, jax.Array] | None,
    log_floor: tuple[jax.Array, jax.Array] | None = None,
    *,
    rows: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Run the scaled pass of :func:`_scaled_kde_log_density` for every model by tiles.

    Each model's weighted kernel sum adds its tiles' sums in a fixed order
    (see :func:`_sum_over_tiles`), so it matches the single-model pass to
    rounding.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims), n_points >= 1
        Evaluation points.
    tile_samples, tile_weights, tile_model
        From :func:`_sample_tiles`.
    counts : jax.Array, shape (n_models,)
        Each model's number of samples, which sets the loss bound of its
        sum.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth.
    bracketing_keys : (left, right) of jax.Array, each shape (n_models, n_points), or None
        From :func:`_bracketing_keys` for 1D points: the exact per-point
        scale. None scales by the kernel's peak.
    log_floor : (jax.Array, jax.Array), shapes (n_models,) and (n_points,), optional
        Per-model and per-point terms of a log-density floor, added. A
        flagged point is left unflagged when its true log density is
        certainly below its floor. A flagged scaled sum is below the loss
        bound ``lost``. Each sample's term that flushed below ``tiny`` lost
        less than ``tiny``, and rounding at most halves the sum of the rest
        and doubles its normalizer (true while ``n_tiles + 256`` additions
        lose less than a factor of two: below about 10 million tiles, 2.6e9
        samples, per model). So the true density is below
        ``exp(scale) * (4 * lost + 2 * n * tiny) / relative_total``. NaN
        floors leave points flagged.
    rows, batch_size : int
        From :func:`_tiled_kde_rows`.

    Returns
    -------
    log_density, needs_log_space : jax.Array, shape (n_models, n_points)
    """
    n_models = counts.shape[0]
    peak = _kernel_log_peak(std)
    # Without keys every point is scaled by the kernel's peak.
    scale = None
    if bracketing_keys is not None:
        left, right = bracketing_keys
        scale = _bounded_log_scale(
            _nearer_key_log_kernel(points[:, 0][None, :], left, right, std), peak
        )
    relative_weights, _, relative_total, total = _tile_weight_terms(
        tile_weights, tile_model, n_models
    )

    def tile_sum(
        tile: tuple[jax.Array, ...],
        block_points: jax.Array,
        block_scale: jax.Array | None,
    ) -> jax.Array:
        samples, weights, active = tile
        log_kernel = _log_kernel_matrix(block_points, samples, std)
        shift = peak if block_scale is None else block_scale[None, :]
        return jnp.matmul(
            weights,
            jnp.exp(jnp.where(active[:, None], log_kernel - shift, -jnp.inf)),
            precision=jax.lax.Precision.HIGHEST,
        )

    scaled = _tiled_model_reduce(
        tile_sum,
        (tile_samples, relative_weights, tile_weights > 0),
        tile_model,
        n_models,
        points,
        scale,
        _sum_over_tiles,
        rows,
        batch_size,
    )
    # As _scaled_kde_log_density, per model.
    average = scaled / jnp.where(relative_total > 0, relative_total, 1.0)[:, None]
    defined = (average > 0) | jnp.isnan(average)
    values = jnp.where(
        defined,
        (peak if scale is None else scale) + jnp.log(jnp.where(defined, average, 1.0)),
        -jnp.inf,
    )
    lost = (
        counts.astype(scaled.dtype)
        * finfo(scaled.dtype).tiny
        / (4 * finfo(scaled.dtype).eps)
    )
    has_mass = (total > 0.0)[:, None]
    values = jnp.where(
        jnp.isnan(total)[:, None], jnp.nan, jnp.where(has_mass, values, -jnp.inf)
    )
    needs_log_space = (scaled < lost[:, None]) & has_mass
    if log_floor is not None:
        bound = 4 * lost + 2 * counts.astype(scaled.dtype) * finfo(scaled.dtype).tiny
        upper = (peak if scale is None else scale) + jnp.log(
            bound / jnp.where(relative_total > 0, relative_total, 1.0)
        )[:, None]
        model_floor, point_floor = log_floor
        floor = model_floor[:, None] + point_floor[None, :]
        needs_log_space = needs_log_space & ~(upper < floor)
    return values, needs_log_space


# Points per block of a model's own flagged points in the log-space repair.
_KDE_REPAIR_POINTS = 16


def _repair_sizes(
    n_items: int,
    block_size: int,
    points: jax.Array | np.ndarray,
    std: jax.Array,
    tile_samples: jax.Array | np.ndarray,
    tile_weights: jax.Array | np.ndarray,
    widest_tiles: int,
) -> tuple[int, int, int]:
    """Items per step and per chunk, and padded items, of the log-space repair.

    An item is one tile at one block of ``_KDE_REPAIR_POINTS`` of its model's
    flagged points. One step's kernel matrix (``batch_size`` items) takes at
    most an eighth of ``_KDE_BLOCK_KERNEL_BYTES``, and has no more entries
    than a quarter of the widest model's own block at ``block_size`` points,
    but at least one item's: the batched repair holds about four temporaries
    per entry, against about two in the single-model pass. A chunk's
    per-item terms, two buffers of ``(chunk, _KDE_REPAIR_POINTS)``, take at
    most a sixteenth of the budget. Chunks and steps are evened out, so the
    padding is fewer items than one step per chunk, plus one step.

    Parameters
    ----------
    n_items : int
        Items to evaluate, padding included.
    block_size : int
        Requested points per block of the single-model pass.
    points, std, tile_samples, tile_weights : array
        Only their dtypes are used.
    widest_tiles : int
        Tiles of the model with the most.

    Returns
    -------
    batch_size, chunk : int
        ``chunk`` is a multiple of ``batch_size``.
    n_items : int
        ``n_items`` rounded up to a whole number of chunks.
    """
    budget = _KDE_BLOCK_KERNEL_BYTES // _kernel_itemsize(
        points, tile_samples, std, tile_weights
    )
    entries = _KDE_SAMPLE_TILE * _KDE_REPAIR_POINTS
    step = min(budget // 8, widest_tiles * _KDE_SAMPLE_TILE * block_size // 4)
    batch_size = max(1, min(n_items, step // entries))
    largest_chunk = max(1, budget // (32 * _KDE_REPAIR_POINTS) // batch_size)
    n_steps = -(-n_items // batch_size)
    n_chunks = -(-n_steps // largest_chunk)
    chunk = -(-n_steps // n_chunks) * batch_size
    return batch_size, chunk, n_chunks * chunk


@partial(jax.jit, static_argnames=("n_models", "n_chunks", "chunk", "batch_size"))
def _paired_log_space_kde_log_densities(
    block_points: jax.Array,
    block_model: jax.Array,
    block_first_item: jax.Array,
    block_first_tile: jax.Array,
    tile_samples: jax.Array,
    tile_weights: jax.Array,
    tile_model: jax.Array,
    std: jax.Array,
    *,
    n_models: int,
    n_chunks: int,
    chunk: int,
    batch_size: int,
) -> jax.Array:
    """:func:`log_kde` of each model at blocks of its own points, by tile and block.

    Each item evaluates one tile at one block: block ``b`` holds items
    ``block_first_item[b]`` to ``block_first_item[b + 1]``, one per tile of
    its model from ``block_first_tile[b]``. Each chunk finds its items' blocks
    and tiles from these starts, so no per-item index array is stored. Each
    block's offset is its model's largest weighted log kernel over the
    block's items (a maximum, independent of order); the items' sums are
    then added per block in a fixed order on each backend (see
    ``deterministic_row_add``), chunk by chunk.

    Parameters
    ----------
    block_points : jax.Array, shape (n_blocks, _KDE_REPAIR_POINTS, n_dims)
        Each block's points.
    block_model : jax.Array, shape (n_blocks,)
        Each block's model.
    block_first_item : jax.Array of int32, shape (n_blocks + 1,)
        Each block's first item, nondecreasing, then the number of items.
        Padding blocks hold no items; the items past the last, up to
        ``n_chunks * chunk``, fall in block ``n_blocks``, which the
        reductions drop.
    block_first_tile : jax.Array of int32, shape (n_blocks,)
        The first tile of each block's model.
    tile_samples, tile_weights, tile_model
        Every model's tiles, from :func:`_bucketed_tiles`.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth.
    n_models : int
        Number of models.
    n_chunks, chunk, batch_size : int
        From :func:`_repair_sizes`: ``n_chunks * chunk`` items, padding
        included.

    Returns
    -------
    log_density : jax.Array, shape (n_blocks, _KDE_REPAIR_POINTS)
        Defined for blocks of models with positive total weight, which are
        the only ones a scaled pass flags.
    """
    _, log_weights, relative_total, _ = _tile_weight_terms(
        tile_weights, tile_model, n_models
    )
    positive = tile_weights > 0
    n_blocks = block_points.shape[0]
    last_tile = tile_samples.shape[0] - 1

    def chunk_items(index: jax.Array) -> tuple[jax.Array, jax.Array]:
        item = index * chunk + jnp.arange(chunk, dtype=jnp.int32)
        block = jnp.searchsorted(block_first_item, item, side="right") - 1
        known = jnp.clip(block, 0, n_blocks - 1)
        tile = block_first_tile[known] + (item - block_first_item[known])
        return jnp.clip(tile, 0, last_tile), block.astype(jnp.int32)

    def log_kernel(tile: jax.Array, block: jax.Array) -> jax.Array:
        return _log_kernel_matrix(block_points[block], tile_samples[tile], std)

    def offsets(
        carry: tuple[jax.Array, jax.Array], index: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        tiles, blocks = chunk_items(index)

        def one_item(item: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            tile, block = item
            kernel = log_kernel(tile, block)
            return (
                jnp.max(
                    jnp.where(
                        positive[tile][:, None],
                        kernel + log_weights[tile][:, None],
                        -jnp.inf,
                    ),
                    axis=0,
                ),
                jnp.any(jnp.isnan(kernel), axis=0).astype(kernel.dtype),
            )

        offset, has_nan = jax.lax.map(one_item, (tiles, blocks), batch_size=batch_size)
        carry_offset, carry_nan = carry
        return (
            carry_offset.at[blocks].max(offset, mode="drop", indices_are_sorted=True),
            carry_nan.at[blocks].max(has_nan, mode="drop", indices_are_sorted=True),
        ), None

    dtype = jnp.result_type(block_points, tile_samples, std, jnp.float64)
    shape = (n_blocks, _KDE_REPAIR_POINTS)
    (offset, has_nan), _ = jax.lax.scan(
        offsets,
        (jnp.full(shape, -jnp.inf, dtype), jnp.zeros(shape, dtype)),
        jnp.arange(n_chunks, dtype=jnp.int32),
    )
    offset = jnp.where(jnp.isfinite(offset), offset, 0.0)

    def masses(carry: jax.Array, index: jax.Array) -> tuple[jax.Array, None]:
        tiles, blocks = chunk_items(index)

        def one_item(item: tuple[jax.Array, jax.Array]) -> jax.Array:
            tile, block = item
            relative = jnp.where(
                positive[tile][:, None],
                (log_kernel(tile, block) - offset[block][None, :])
                + log_weights[tile][:, None],
                -jnp.inf,
            )
            return jnp.matmul(
                jnp.ones(relative.shape[0], relative.dtype),
                jnp.exp(relative),
                precision=jax.lax.Precision.HIGHEST,
            )

        mass = jax.lax.map(one_item, (tiles, blocks), batch_size=batch_size)
        return deterministic_row_add(carry, mass, blocks, indices_are_sorted=True), None

    mass, _ = jax.lax.scan(
        masses, jnp.zeros(shape, dtype), jnp.arange(n_chunks, dtype=jnp.int32)
    )
    # As log_kde, per block.
    log_total = jnp.log(jnp.where(relative_total > 0, relative_total, 1.0))[
        block_model
    ][:, None]
    log_mass = jnp.log(jnp.where(mass > 0, mass, 1.0))
    values = jnp.where(mass > 0, offset + (log_mass - log_total), -jnp.inf)
    values = jnp.where(jnp.isnan(mass), jnp.nan, values)
    return jnp.where(has_nan > 0, jnp.nan, values)


@dataclass
class _GroupTiles:
    """A group of models' tiles and sorted 1D keys, built once per prediction.

    Attributes
    ----------
    host : (np.ndarray, np.ndarray, np.ndarray)
        From :func:`_sample_tiles`.
    widest : int
        Tiles of the model with the most.
    counts : np.ndarray, shape (n_models,)
        Each model's number of samples.
    keys : list of np.ndarray or None
        From :func:`_sorted_model_keys`, for 1D points.
    device_keys : tuple of jax.Array or None
        The keys moved to the device once, for :func:`_bracketing_keys`:
        every model's keys in one array, each key's model, and each model's
        first key and key count; None without keys or before first use.
    device : dict
        The tiles on the device, by padded tile count (see
        :func:`_bucketed_tiles`).
    """

    host: tuple[np.ndarray, np.ndarray, np.ndarray]
    widest: int
    counts: np.ndarray
    keys: list[np.ndarray] | None
    device: dict[int, tuple[jax.Array, jax.Array, jax.Array]] = field(
        default_factory=dict
    )
    device_keys: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None

    @classmethod
    def build(
        cls,
        samples: Sequence[np.ndarray],
        weights: Sequence[np.ndarray],
        *,
        one_dimensional: bool,
    ) -> "_GroupTiles":
        """Cut the models' samples into tiles and, for 1D points, sort their keys."""
        host = _sample_tiles(samples, weights)
        return cls(
            host=host,
            widest=int(np.bincount(host[2]).max()),
            counts=np.asarray([np.shape(s)[0] for s in samples]),
            keys=_sorted_model_keys(samples, weights) if one_dimensional else None,
        )

    def keys_on_device(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return the sorted 1D keys as :func:`_bracketing_keys` takes them, moved once."""
        assert self.keys is not None  # 1D points with at least one key
        if self.device_keys is None:
            count = np.asarray([k.size for k in self.keys], np.int32)
            start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int32)
            self.device_keys = (
                jnp.asarray(np.concatenate(self.keys)),
                jnp.asarray(np.repeat(np.arange(count.size, dtype=np.int32), count)),
                jnp.asarray(start),
                jnp.asarray(count),
            )
        return self.device_keys

    def on_device(self, n_padded: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the tiles padded to ``n_padded``, moved to the device once."""
        if n_padded not in self.device:
            self.device[n_padded] = _bucketed_tiles(*self.host, n_padded)
        return self.device[n_padded]


def _tiled_kde_log_densities(
    points: jax.Array,
    group: _GroupTiles,
    std: jax.Array,
    block_size: int,
    *,
    log_floor: LogFloor | None = None,
) -> np.ndarray:
    """Return the true log densities of KDEs sharing a bandwidth, on the host.

    One tiled scaled pass for every model, then one log-space pass that
    evaluates each flagged model at its own flagged points only (see
    :func:`_paired_log_space_kde_log_densities`). Tile counts are padded to
    four sizes per octave and repair block and item counts to powers of two,
    so the kernels compile for those sizes, not for each model's sample
    count or each flagged set.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims), n_points >= 1
        Evaluation points.
    group : _GroupTiles
        The models' tiles and, for 1D points, sorted keys.
    std : jax.Array, shape (n_dims,)
        Kernel bandwidth.
    block_size : int
        Evaluation points per block.
    log_floor : (np.ndarray, np.ndarray), shapes (n_models,) and (n_points,), optional
        See :func:`kde_log_density_rows`.

    Returns
    -------
    log_densities : np.ndarray, shape (n_models, n_points)
    """
    n_models = group.counts.size
    if points.shape[0] == 0:
        return np.full((n_models, 0), LOG_EPS)
    host_points = np.asarray(points)
    host_tiles = group.host
    widest = group.widest
    n_tiles = padded_tile_count(host_tiles[2].size)
    rows, batch_size, n_pass_tiles = _tiled_kde_rows(
        block_size, points, n_tiles, std, *host_tiles[:2], widest
    )
    tiles = group.on_device(n_pass_tiles)
    bracketing_keys = None
    if group.keys is not None:
        if sum(k.size for k in group.keys) == 0:
            no_keys = jnp.full((n_models, points.shape[0]), jnp.inf, points.dtype)
            bracketing_keys = (no_keys, no_keys)
        else:
            # Sorted on the host (an XLA sort compiles slowly on GPU); the
            # search runs on the device.
            order = np.argsort(host_points[:, 0], kind="stable").astype(np.int32)
            bracketing_keys = _bracketing_keys(
                jnp.asarray(host_points[order, 0]),
                jnp.asarray(order),
                *group.keys_on_device(),
            )
    values, needs_log_space = _tiled_scaled_kde_log_densities(
        points,
        *tiles,
        jnp.asarray(group.counts),
        std,
        bracketing_keys,
        None
        if log_floor is None
        else (jnp.asarray(log_floor[0]), jnp.asarray(log_floor[1])),
        rows=rows,
        batch_size=batch_size,
    )
    # Copied to the host: device indexing compiles a kernel per shape.
    values, needs_log_space = np.array(values), np.asarray(needs_log_space)
    if not needs_log_space.any():
        return values
    # Each flagged model is evaluated at its own flagged points only, in
    # blocks of _KDE_REPAIR_POINTS (the last repeats its last point): one
    # item per tile and block. Blocks and items are padded to powers of two,
    # so repairs of different flagged sets (chunks, refits) share kernels.
    size = _KDE_REPAIR_POINTS
    tile_start = np.searchsorted(host_tiles[2], np.arange(n_models + 1))
    flagged_models = np.flatnonzero(needs_log_space.any(axis=1))
    flagged_points = [np.flatnonzero(needs_log_space[m]) for m in flagged_models]
    n_model_blocks = [-(-indices.size // size) for indices in flagged_points]
    n_blocks = sum(n_model_blocks)
    gather = np.concatenate(
        [
            np.pad(indices, (0, n * size - indices.size), mode="edge")
            for indices, n in zip(flagged_points, n_model_blocks, strict=True)
        ]
    ).reshape(n_blocks, size)
    block_model = np.repeat(flagged_models, n_model_blocks).astype(np.int32)
    first_block = np.cumsum([0, *n_model_blocks])
    # Each block has one item per tile of its model; padding blocks none.
    n_padded_blocks = padded_repair_count(n_blocks)
    block_items = pad_rows(np.diff(tile_start)[block_model], n_padded_blocks)
    block_first_item = np.concatenate([[0], np.cumsum(block_items)])
    # Item indices are int32 on the device; a wrapped index would write into
    # the last block instead of being dropped.
    if padded_repair_count(int(block_first_item[-1])) >= 2**31:
        msg = (
            f"The log-space KDE repair needs {int(block_first_item[-1]):,} "
            "(tile, block) pairs, more than int32 indexing allows; decode in "
            "more chunks."
        )
        raise RuntimeError(msg)
    block_first_item = block_first_item.astype(np.int32)
    batch_size, chunk, n_items = _repair_sizes(
        padded_repair_count(int(block_first_item[-1])),
        block_size,
        host_points,
        std,
        *host_tiles[:2],
        widest,
    )
    exact = np.asarray(
        _paired_log_space_kde_log_densities(
            jnp.asarray(host_points[pad_rows(gather, n_padded_blocks)]),
            jnp.asarray(pad_rows(block_model, n_padded_blocks)),
            jnp.asarray(block_first_item),
            jnp.asarray(
                pad_rows(tile_start[block_model].astype(np.int32), n_padded_blocks)
            ),
            *tiles,
            std,
            n_models=n_models,
            n_chunks=n_items // chunk,
            chunk=chunk,
            batch_size=batch_size,
        )
    )
    for i, (model, indices) in enumerate(
        zip(flagged_models, flagged_points, strict=True)
    ):
        block_values = exact[first_block[i] : first_block[i + 1]].reshape(-1)
        values[model, indices] = block_values[: indices.size]
    return values


_compiled_log_space_kde_log_density = jax.jit(
    log_space_kde_log_density, static_argnames=("block_size",)
)


def prepared_kde_log_density(
    model: KDEModel,
    points: np.ndarray,
    spike_time_order: SpikeTimeOrder,
    *,
    predict_log: Callable[[KDEModel, Any], jax.Array] | None = None,
    policy: str = "",
) -> jax.Array:
    """Cache true density for immutable points and fitted leaves during prediction."""

    def predict() -> jax.Array:
        if predict_log is None:
            return kde_log_density(model, points)
        return predict_log(model, points)

    return spike_time_order.memo_kde(
        f"kde_log_density:{model.block_size}:{policy}",
        (model, model.samples_, model.weights_, model.std, points),
        predict,
    )


def prepared_kde_log_densities(
    models: Sequence[KDEModel],
    points: np.ndarray,
    spike_time_order: SpikeTimeOrder,
) -> list[jax.Array]:
    """Cache :func:`kde_log_densities` at immutable points during a prediction.

    The several-model form of :func:`prepared_kde_log_density`: one entry
    for all ``models``, which replacing any fitted leaf invalidates.
    """
    leaves = [
        leaf
        for model in models
        for leaf in (model, model.samples_, model.weights_, model.std)
    ]
    return spike_time_order.memo_kde(
        f"kde_log_densities:{tuple(model.block_size for model in models)}",
        (*leaves, points),
        lambda: kde_log_densities(models, points),
    )


def kde_model_with_block_size(
    model: KDEModel, block_size: int | None, spike_time_order: SpikeTimeOrder
) -> KDEModel:
    """``model`` evaluated in blocks of ``block_size`` points, without changing it.

    A prediction may evaluate fitted models in other blocks than the fit did.
    The copy is made once per prediction, so its cached densities are shared
    across states and chunks; the model itself is returned when its block size
    already matches.

    Parameters
    ----------
    model : KDEModel
        A fitted model.
    block_size : int or None
        Evaluation points per block; None evaluates all at once.
    spike_time_order : SpikeTimeOrder
        The prediction's preparation cache.

    Returns
    -------
    model : KDEModel
        ``model``, or a shallow copy with ``block_size`` set.
    """
    if model.block_size == block_size:
        return model

    def build() -> KDEModel:
        blocked = copy.copy(model)
        blocked.block_size = block_size
        return blocked

    return spike_time_order.memo(f"kde_block_size:{block_size}", (model,), build)


def prepare_kde_local_inputs(
    model: KDEModel,
    position_time: np.ndarray | jax.Array,
    position: np.ndarray | jax.Array,
    environment: Environment,
    time_grid: _DecodeTimeGrid,
    row_start: int,
    row_stop: int,
    spike_time_order: SpikeTimeOrder,
    *,
    predict_log: Callable[[KDEModel, Any], jax.Array] | None = None,
    policy: str = "",
) -> tuple[np.ndarray, jax.Array]:
    """Interpolate one row range and prepare its occupancy density once.

    The cache belongs to one immutable prediction. A new prediction rechecks
    original inputs, and replacing fitted leaves invalidates density entries.
    Keys retain strong source references; row chunks share a 32 MiB LRU budget.
    """
    points = spike_time_order.memo_kde(
        f"kde_local_position:{row_start}:{row_stop}",
        (position_time, position, environment, time_grid),
        lambda: snap_off_track_positions(
            get_position_at_time(
                position_time,
                position,
                decode_bin_centers(time_grid.edges, row_start, row_stop),
                environment,
            ),
            environment,
        ),
    )
    return points, prepared_kde_log_density(
        model, points, spike_time_order, predict_log=predict_log, policy=policy
    )


def log_density_intensity(
    mean_rate: jax.Array | float,
    log_density: jax.Array,
    log_occupancy: jax.Array,
) -> jax.Array:
    """Compute ``log(rate * density / occupancy)``, forming density ratios before rates.

    An unsupported occupancy or numerator gives zero signal, rather than NaN
    from subtracting two negative infinities. Finite tails and NaN diagnostics
    remain visible. Clusterless callers combine the result with their background.

    Parameters
    ----------
    mean_rate : jax.Array or float
        Mean firing rate, broadcastable with ``log_density``, in the same time
        unit as the caller's intensities.
    log_density : jax.Array
        Log of the joint (or ground-process) density.
    log_occupancy : jax.Array
        Log occupancy density, broadcastable with ``log_density``.

    Returns
    -------
    log_intensity : jax.Array
        ``log(mean_rate) + log_density - log_occupancy``, or ``-inf`` where the
        rate, density or occupancy has no support.
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
    positions: jax.Array,
    marks: jax.Array,
    encoding_positions: jax.Array,
    encoding_marks: jax.Array,
    position_std: jax.Array,
    mark_std: jax.Array,
    log_occupancy: jax.Array,
    weights: jax.Array,
    block_size: int,
) -> jax.Array:
    """Compute the joint KDE/occupancy log ratio, dividing position kernels before adding marks.

    Keeping the mark term separate preserves its normalizer when position log
    densities are large enough that adding it would round it away in float32.
    Evaluation blocking retains the shared KDE kernel-workspace limit.

    Parameters
    ----------
    positions : jax.Array, shape (n_eval, n_position_dims)
        Positions at which to evaluate the ratio.
    marks : jax.Array, shape (n_eval, n_features)
        Waveform features (marks) at the evaluation points.
    encoding_positions : jax.Array, shape (n_encoding, n_position_dims)
        Positions of the encoding spikes.
    encoding_marks : jax.Array, shape (n_encoding, n_features)
        Waveform features (marks) of the encoding spikes.
    position_std : jax.Array, shape (n_position_dims,)
        Position kernel standard deviation.
    mark_std : jax.Array, shape (n_features,)
        Waveform-feature kernel standard deviation.
    log_occupancy : jax.Array, shape (n_eval,)
        Log occupancy density at ``positions``.
    weights : jax.Array, shape (n_encoding,)
        Weight of each encoding spike; zero weights drop the spike.
    block_size : int
        Maximum number of evaluation points per block.

    Returns
    -------
    log_ratio : jax.Array, shape (n_eval,)
        Log of the weighted joint KDE divided by the occupancy density at each
        evaluation point, normalized by the sum of ``weights``; ``-inf`` where
        the occupancy is zero or the weights sum to zero.
    """
    n_position_dims = positions.shape[1]
    points = jnp.concatenate((positions, marks, log_occupancy[:, None]), axis=1)
    samples = jnp.concatenate((encoding_positions, encoding_marks), axis=1)
    std = jnp.concatenate((position_std, mark_std))

    def kernel(
        query: jax.Array, samples: jax.Array, std: jax.Array, weights: jax.Array
    ) -> jax.Array:
        """Evaluate the log ratio of one block of query rows."""

        def supported(_: None) -> jax.Array:
            """Evaluate the ratio for a query with positive total weight."""
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

        return cast(
            jax.Array,
            jax.lax.cond(
                jnp.sum(weights) > 0.0,
                supported,
                lambda _: jnp.full(query.shape[0], -jnp.inf),
                None,
            ),
        )

    def blocked(
        query: jax.Array,
        samples: jax.Array,
        std: jax.Array,
        rows: int,
        weights: jax.Array,
    ) -> jax.Array:
        """Evaluate the log ratio of ``query`` in blocks of ``rows`` rows."""
        return jnp.concatenate(
            [
                kernel(query[start : start + rows], samples, std, weights)
                for start in range(0, query.shape[0], rows)
            ]
        )

    if points.shape[0] == 0:
        return jnp.zeros(0)
    return _traced_blocks(blocked, kernel, points, samples, std, block_size, weights)
