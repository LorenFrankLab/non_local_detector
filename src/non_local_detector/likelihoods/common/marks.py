"""Mark kernels and masked log-space sums shared by the clusterless likelihoods."""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp

from non_local_detector.likelihoods.common.numerics import EPS

# Flagged indices per block repaired on a short path that visits only them;
# more visit every index (``repair_flagged``) or every group of flagged
# indices (``_repair_flagged_in_groups``).
EXACT_REPAIR_SLOTS = 32


def masked_logsumexp(values: jax.Array) -> jax.Array:
    """Log-sum over encoding rows with a constant empty-support derivative."""
    supported = jnp.any(~jnp.isneginf(values), axis=0)
    safe_values = jnp.where(supported[None, :], values, 0.0)
    reduced = jax.nn.logsumexp(safe_values, axis=0)
    return jnp.where(supported, reduced, -jnp.inf)


def masked_logaddexp(left: jax.Array, right: jax.Array) -> jax.Array:
    """Preserve the empty accumulation identity without -inf/-inf tangents."""
    empty = jnp.isneginf(left) & jnp.isneginf(right)
    reduced = jnp.logaddexp(jnp.where(empty, 0.0, left), jnp.where(empty, 0.0, right))
    return jnp.where(empty, -jnp.inf, reduced)


def zero_if_neginf(log_max: jax.Array) -> jax.Array:
    """Neutralize ``-inf`` maxima before subtraction; ``NaN`` stays visible."""
    return jnp.where(jnp.isneginf(log_max), 0.0, log_max)


def repair_flagged(
    values: jax.Array,
    needs_exact: jax.Array,
    repaired: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Apply ``repaired(values, index, flag)`` at each flagged index.

    At most ``EXACT_REPAIR_SLOTS`` flagged indices visit only those; more visit
    every index. Static loop bounds keep the repair differentiable.

    Parameters
    ----------
    values : jax.Array
        Estimates, updated in place by ``repaired``.
    needs_exact : jax.Array of bool, shape (n,)
        True at the indices to repair.
    repaired : callable
        ``repaired(values, index, flag) -> values``; leaves ``values`` unchanged
        when ``flag`` is False.

    Returns
    -------
    values : jax.Array
    """
    n = needs_exact.shape[0]
    n_flagged = jnp.sum(needs_exact)
    flagged = jnp.nonzero(needs_exact, size=n, fill_value=0)[0]
    n_slots = min(n, EXACT_REPAIR_SLOTS)

    def repair(n_visits: int, current: jax.Array) -> jax.Array:
        return cast(
            jax.Array,
            jax.lax.fori_loop(
                0,
                n_visits,
                lambda slot, current: repaired(
                    current, flagged[slot], slot < n_flagged
                ),
                current,
            ),
        )

    return cast(
        jax.Array,
        jax.lax.cond(
            n_flagged == 0,
            lambda current: current,
            lambda current: jax.lax.cond(
                n_flagged <= n_slots,
                partial(repair, n_slots),
                partial(repair, n),
                current,
            ),
            values,
        ),
    )


def compute_log_mark_kernel_gemm(
    decoding_features: jax.Array,
    encoding_features: jax.Array,
    waveform_stds: jax.Array,
) -> jax.Array:
    """Stable log Gaussian mark kernel for the GEMM marginal paths.

    Subtract each pair's features before scaling and squaring. Expanding the
    squared distance as ``||x||^2 + ||y||^2 - 2*x@y`` loses nearby distances
    when features have large offsets, even at HIGHEST matmul precision. A
    shared centering point cannot prevent this for widely separated clusters.

    Accumulate one feature at a time, with only an ``(n_enc, n_dec)`` array
    per feature rather than an ``(n_enc, n_dec, n_features)`` temporary. XLA
    can fuse these elementwise operations. The historical function name is
    retained; the marginal reduction applies GEMM to the exponentiated kernel.

    Parameters
    ----------
    decoding_features : jax.Array, shape (n_decoding_spikes, n_features)
        Waveform features for decoding spikes.
    encoding_features : jax.Array, shape (n_encoding_spikes, n_features)
        Waveform features for encoding spikes.
    waveform_stds : jax.Array, shape (n_features,)
        Standard deviations for each feature dimension.

    Returns
    -------
    logK_mark : jax.Array, shape (n_encoding_spikes, n_decoding_spikes)
        Log kernel matrix K[i, j] = log(Gaussian kernel between encoding spike i and decoding spike j).
    """
    n_features = encoding_features.shape[1]

    # Preserve the shared bandwidth floor and Gaussian normalizer. Standalone
    # callers can supply a length-one bandwidth for all features.
    waveform_stds = jnp.clip(waveform_stds, min=EPS)
    waveform_stds = jnp.broadcast_to(waveform_stds, (n_features,))

    # Log normalization constant: -0.5 * (D * log(2π) + 2 * sum(log(sigma)))
    # Factor of 2 because we have sum of log(sigma), not log(sigma^2)
    log_norm_const = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )

    feature_dtype = jnp.result_type(
        decoding_features, encoding_features, waveform_stds, jnp.float32
    )
    # Raw ADC marks can be signed/unsigned integers; subtraction must not wrap.
    decoding_features = decoding_features.astype(feature_dtype)
    encoding_features = encoding_features.astype(feature_dtype)

    # Halved features cannot overflow their subtraction, and a broad sigma can
    # make a finite distance from features whose plain difference overflows.
    # Scaling by a power of two commutes with rounding, so 4 * sum(half^2)
    # equals sum(difference^2) bit for bit wherever the difference is finite
    # and no scaled term is subnormal (those terms are below 1e-37).
    # One branch-free expression keeps the loop fused with the exponentials
    # that consume it and is traced once.
    half_sq_dist = jnp.zeros(
        (encoding_features.shape[0], decoding_features.shape[0]),
        dtype=feature_dtype,
    )
    for decoding_dim, encoding_dim, sigma in zip(
        0.5 * decoding_features.T, 0.5 * encoding_features.T, waveform_stds, strict=True
    ):
        half_difference = (encoding_dim[:, None] - decoding_dim[None, :]) / sigma
        half_sq_dist += half_difference**2
    return log_norm_const - 2.0 * half_sq_dist  # (n_enc, n_dec)
