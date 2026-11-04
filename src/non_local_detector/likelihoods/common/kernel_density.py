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

from non_local_detector._validation import validate_weights
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common.numerics import LOG_EPS
from non_local_detector.likelihoods.common.padding import (
    pad_rows,
    padded_index,
    padded_sample_count,
)
from non_local_detector.likelihoods.common.position import (
    get_position_at_time,
    snap_off_track_positions,
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
        divisor = jax.lax.optimization_barrier(jnp.broadcast_to(sigma, shape))  # type: ignore[no-untyped-call]  # untyped in JAX
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
    samples : jax.Array, shape (n_samples, n_dims)
    std : jax.Array, shape (n_dims,)

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
    samples : jax.Array, shape (n_samples, n_dims)
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
    itemsize: int = jnp.result_type(
        eval_points, samples, std, weights, accumulator
    ).itemsize
    budget_rows = _KDE_BLOCK_KERNEL_BYTES // (max(samples.shape[0], 1) * itemsize)
    return min(block_size, max(1, budget_rows))


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
    samples : jax.Array, shape (n_samples, n_dims)
    std : jax.Array, shape (n_dims,)
    block_size : int
    weights : jax.Array, shape (n_samples,)

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
    samples : jax.Array, shape (n_samples, n_dims)
    std : jax.Array, shape (n_dims,)
    block_size : int
    weights : jax.Array, shape (n_samples,)

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
        weight_scale > 1.0 / jnp.finfo(positive_weights.dtype).tiny,  # type: ignore[no-untyped-call]  # untyped in JAX
        lambda pair: jax.lax.optimization_barrier((pair[0] * 0.25, pair[1] * 0.25)),  # type: ignore[no-untyped-call]  # untyped in JAX
        lambda pair: pair,
        (positive_weights, weight_scale),
    )
    relative_weights = positive_weights / weight_scale
    representable = relative_weights >= jnp.finfo(relative_weights.dtype).tiny  # type: ignore[no-untyped-call]  # untyped in JAX
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
) -> jax.Array:
    """Return the true log density of fitted KDE leaves, outside jit.

    One compiled scaled pass (see :func:`_scaled_kde_log_density`), then the
    flagged points in log space. Without ``keys`` points are scaled by the
    kernel's peak, which suits points expected near the samples.

    Parameters
    ----------
    points : jax.Array, shape (n_points, n_dims)
    samples : jax.Array, shape (n_samples, n_dims)
    std : jax.Array, shape (n_dims,)
    block_size : int
    weights : jax.Array, shape (n_samples,)
    keys : jax.Array, shape (n_samples,), optional
        From :func:`_sorted_sample_keys`.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
    """
    values, needs_log_space = _compiled_scaled_kde_log_density(
        points, samples, std, block_size, weights, keys
    )
    return _repair_in_log_space(
        values, np.asarray(needs_log_space), points, samples, std, block_size, weights
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

    Parameters
    ----------
    values : jax.Array, shape (n_points,)
    needs_log_space : np.ndarray of bool, shape (n_points,)
    points, samples, std, block_size, weights
        The pass's inputs.

    Returns
    -------
    log_density : jax.Array, shape (n_points,)
    """
    if not needs_log_space.any():
        return values
    index = np.flatnonzero(needs_log_space)
    exact = _compiled_log_space_kde_log_density(
        points[padded_index(index)], samples, std, block_size, weights
    )
    return values.at[index].set(exact[: index.size])


def kde_log_densities(
    models: Sequence["KDEModel"], points: jax.Array | np.ndarray
) -> list[jax.Array]:
    """Return the true log densities of several fitted KDEs at the same points.

    ``kde_log_density`` for each model, with fewer compiled kernels: models
    whose padded sample counts (four sizes per octave, zero-weight rows that
    add no mass) and block sizes match share one pass, mapped over models in
    batches (see :func:`_batched_kde_rows`). Runs outside jit; custom
    subclasses use ``kde_log_density``.

    Parameters
    ----------
    models : sequence of KDEModel
    points : jax.Array, shape (n_points, n_dims) or (n_points,)

    Returns
    -------
    log_densities : list of jax.Array, shape (n_points,)
        In the order of ``models``.
    """
    points = jnp.asarray(points)
    if points.ndim == 1:
        points = points[:, None]
    results: list[jax.Array] = [None] * len(models)  # type: ignore[list-item]  # every slot is filled before return
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, model in enumerate(models):
        if type(model) is not KDEModel:
            results[index] = kde_log_density(model, points)
            continue
        if model.samples_ is None:
            msg = "This KDE instance is not fitted yet."
            raise RuntimeError(msg)
        assert model.weights_ is not None  # set together with samples_ in fit
        std_bytes = np.asarray(as_std_array(model.std, points.shape[1])).tobytes()
        key = (
            padded_sample_count(model.samples_.shape[0]),
            points.shape[0] if model.block_size is None else model.block_size,
            std_bytes,
            np.dtype(model.samples_.dtype).str,
            np.dtype(model.weights_.dtype).str,
        )
        groups.setdefault(key, []).append(index)
    for (n_padded, block_size, _, _, _), members in groups.items():
        samples = np.stack(
            [pad_rows(cast(jax.Array, models[i].samples_), n_padded) for i in members]
        )
        weights = np.stack(
            [pad_rows(cast(jax.Array, models[i].weights_), n_padded) for i in members]
        )
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
    samples : array, shape (n_samples, n_dims)
    weights : array, shape (n_samples,)

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
    keys : jax.Array, shape (n_samples,)
        Sorted positions of the samples with positive weight, ``inf`` for
        the rest.
    std : jax.Array, shape (1,)

    Returns
    -------
    log_kernel : jax.Array, shape (n_points,)
        ``-inf`` where no key is finite.
    """
    index = jnp.searchsorted(keys, points[:, 0])
    last = keys.shape[0] - 1
    left = keys[jnp.clip(index - 1, 0, last)]
    right = keys[jnp.clip(index, 0, last)]
    nearest = jnp.where(
        jnp.abs(points[:, 0] - left) <= jnp.abs(points[:, 0] - right), left, right
    )
    return cast(jax.Array, log_gaussian_pdf(points[:, 0], nearest, std[0]))


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
    std : jax.Array, shape (n_dims,)
    keys : jax.Array, shape (n_samples,), optional

    Returns
    -------
    log_scale : jax.Array, shape (n_points,)
    """
    peak = sum(
        log_gaussian_pdf(jnp.zeros(()), jnp.zeros(()), dim_std) for dim_std in std
    )
    if keys is None:
        return jnp.full(points.shape[0], peak)
    scale = _nearest_sample_log_kernel(points, keys, std)
    return cast(
        jax.Array, jax.lax.stop_gradient(jnp.where(jnp.isfinite(scale), scale, peak))
    )


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
    samples : jax.Array, shape (n_samples, n_dims)
    std : jax.Array, shape (n_dims,)
    block_size : int
    weights : jax.Array, shape (n_samples,)
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
            samples.shape[0]
            * jnp.finfo(scaled.dtype).tiny  # type: ignore[no-untyped-call]  # untyped in JAX
            / (4 * jnp.finfo(scaled.dtype).eps)  # type: ignore[no-untyped-call]  # untyped in JAX
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


def _batched_kde_rows(
    block_size: int,
    n_models: int,
    points: jax.Array,
    samples: jax.Array | np.ndarray,
    std: jax.Array,
    weights: jax.Array | np.ndarray,
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
    points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
    block_size: int,
    weights: jax.Array,
    keys: jax.Array | None,
    *,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Compute the scaled pass for each of a stack of models, ``batch_size`` at a time.

    ``samples`` has shape ``(n_models, n_samples, n_dims)`` and ``weights``
    and ``keys`` (or None) ``(n_models, n_samples)``. Returns log densities and
    flags of shape ``(n_models, n_points)``.
    """

    def one_model(
        leaf: tuple[jax.Array, jax.Array, jax.Array | None],
    ) -> tuple[jax.Array, jax.Array]:
        model_samples, model_weights, model_keys = leaf
        return _scaled_kde_log_density(
            points,
            model_samples,
            std,
            block_size,
            model_weights,
            model_keys,
        )

    return cast(
        tuple[jax.Array, jax.Array],
        jax.lax.map(one_model, (samples, weights, keys), batch_size=batch_size),
    )


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
