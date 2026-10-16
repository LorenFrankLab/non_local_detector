"""Clusterless KDE likelihood.

The marked intensity is a Gaussian KDE over position and waveform features divided
by the occupancy density, evaluated in log space. Each spike's term is
``logaddexp(log intensity, log background)`` with a background
``RATE_EPS_HZ * q(mark)`` (see ``mark_reference``), so every intensity that can
change that term must be exact even far below float32's probability range. A
scaled GEMM keeps the reduction fast; spikes whose relevant range it cannot cover
are recomputed with an exact logsumexp.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    KDEModel,
    _kde_log_densities,
    _pad_rows,
    _padded_occupancy_count,
    _padded_samples,
    _padded_spike_count,
    _prepare_kde_local_inputs,
    _prepared_kde_log_density,
    _SpikeTimeOrder,
    _true_kde_log_density,
    as_std_array,
    deterministic_row_sum,
    drop_zero_weight_samples,
    get_position_at_time,
    interpolate_weights_at_spike_times,
    kde_log_density,
    local_kde_log_ratio,
    log_bin_duration_evidence,
    log_density_intensity,
    log_gaussian_pdf,
    log_nonnegative,
    resolve_row_slice,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    sum_spikes_into_rows,
    traced_kde_log_density,
    validate_bandwidth,
    validate_finite,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
    fit_mark_reference,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)

# Encoding rows per reduction tile. Each tile is one GEMM over the whole query
# block, so temporaries scale with the tile rather than the encoding count.
_LOG_REDUCTION_ENCODING_ROWS = 2048
# Flagged spikes per block whose exact reduction visits only those rows; a block
# with more flagged spikes visits every row.
_EXACT_REPAIR_SLOTS = 32


def _masked_logsumexp(values: jnp.ndarray) -> jnp.ndarray:
    """Log-sum over encoding rows with a constant empty-support derivative."""
    supported = jnp.any(~jnp.isneginf(values), axis=0)
    safe_values = jnp.where(supported[None, :], values, 0.0)
    reduced = jax.nn.logsumexp(safe_values, axis=0)
    return jnp.where(supported, reduced, -jnp.inf)


def _masked_logaddexp(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    """Preserve the empty accumulation identity without -inf/-inf tangents."""
    empty = jnp.isneginf(left) & jnp.isneginf(right)
    reduced = jnp.logaddexp(jnp.where(empty, 0.0, left), jnp.where(empty, 0.0, right))
    return jnp.where(empty, -jnp.inf, reduced)


def _zero_if_neginf(log_max: jnp.ndarray) -> jnp.ndarray:
    """Neutralize ``-inf`` maxima before subtraction; ``NaN`` stays visible."""
    return jnp.where(jnp.isneginf(log_max), 0.0, log_max)


def _log_marginal_floor(mean_rate: float, occupancy: jnp.ndarray) -> jnp.ndarray:
    """Per-bin part of the smallest log marginal that can change a spike's term.

    Each spike's term is ``logaddexp(log_joint, log_background)``. A joint below
    ``log_background + log(eps)`` leaves that term unchanged in the working
    precision, so it need not be exact. That threshold on the joint is, on the
    marginal, the spike's threshold plus this per-bin offset, which mirrors
    ``_log_joint_from_log_marginal`` through the same ``log_density_intensity``.
    A zero rate or zero occupancy gives ``inf``: no marginal reaches the
    threshold.

    Parameters
    ----------
    mean_rate : float
    occupancy : jnp.ndarray, shape (n_position_bins,)

    Returns
    -------
    position_floor : jnp.ndarray, shape (n_position_bins,)
    """
    log_occupancy = jnp.log(occupancy)
    return -log_density_intensity(
        mean_rate, jnp.zeros_like(log_occupancy), log_occupancy
    )


def _scaled_gemm_shift(n_rows: int, dtype) -> float:
    """Exponent shift that keeps an ``n_rows``-term scaled sum below 1 / (2 tiny).

    The cap keeps both the sum and its reciprocal, the derivative of its
    logarithm, normal numbers.
    """
    dtype_info = jnp.finfo(dtype)
    return -0.5 * float(np.log(2.0 * n_rows * dtype_info.tiny))


def _scaled_gemm_window(n_rows: int, dtype) -> float:
    """Nats below its factor maxima within which a scaled-GEMM entry is exact.

    A sum of ``n_rows`` factors, each scaled by its maximum and shifted up by
    ``_scaled_gemm_shift``, loses at most ``n_rows * tiny * exp(shift)`` per
    entry in scaled units. An entry whose ``mark_max + position_max``
    exceeds its floor by no more than this window therefore omits less than
    four dtype epsilons of any value at or above the floor: about 104.7 nats
    for 2048 float32 rows.
    """
    dtype_info = jnp.finfo(dtype)
    return float(
        np.log(4 * dtype_info.eps)
        - np.log(n_rows * dtype_info.tiny)
        + _scaled_gemm_shift(n_rows, dtype)
    )


def _log_joint_from_log_marginal(
    log_marginal: jnp.ndarray, mean_rate: float, occupancy: jnp.ndarray
) -> jnp.ndarray:
    """Combine a log marginal density with mean rate and occupancy.

    Computes ``log(mean_rate * marginal / occupancy)`` in log space with one
    degeneracy contract shared by the GEMM joint-intensity paths (untiled, encoding-
    tiled, and streaming) and by the local at-position likelihood: a bin
    with zero occupancy, a fully underflowed marginal (``-inf``, true zero mass), or a
    zero mean rate (a fully de-weighted electrode) has zero intensity, ``-inf``.
    Callers combine it with the spike's background, which then explains the spike.
    A ``NaN`` marginal is deliberately kept (the zero-occupancy mask is
    ``~isnan``-gated) so a broken computation reaches ``core.py``'s NaN
    diagnostics instead of being laundered into a finite value.

    Parameters
    ----------
    log_marginal : jnp.ndarray, shape (n_rows, n_position_bins)
        Log marginal mark/position density. ``n_rows`` is the number of decoding
        spikes for the non-local paths, or ``1`` for the per-spike local path (which
        passes its ``(n_spikes,)`` marginal as a single row so the per-bin broadcast
        below becomes a per-spike elementwise combine).
    mean_rate : float
        Mean firing rate for this electrode.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        Occupancy density at the position bins (per decoding spike for the local
        path).

    Returns
    -------
    log_joint : jnp.ndarray, shape (n_rows, n_position_bins)
    """
    log_occ = jnp.log(occupancy)
    log_joint = log_density_intensity(mean_rate, log_marginal, log_occ[None, :])
    # Zero occupancy (no support) has zero intensity; NaN stays visible.
    unsupported = (occupancy[None, :] <= 0.0) & ~jnp.isnan(log_joint)
    return jnp.where(unsupported, -jnp.inf, log_joint)


@jax.jit
def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized KDE distance computed via log-space for numerical stability.

    Computes the product of per-dimension Gaussian kernels by summing
    log-Gaussian values and exponentiating, avoiding underflow that occurs
    when multiplying many small per-dimension PDFs directly.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel for each dimension.

    Returns
    -------
    distance : jnp.ndarray, shape (n_samples, n_eval_points)
        Product of per-dimension Gaussian PDF values.

    Notes
    -----
    Inputs are assumed to have matching dimensionality (not validated, to keep
    JIT compatibility). ``std`` is clamped to ``[EPS, inf)`` inside the delegated
    ``log_kde_distance`` call, so a zero bandwidth yields a heavily peaked kernel
    rather than NaN.
    """
    return jnp.exp(log_kde_distance(eval_points, samples, std))


@jax.jit
def log_kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized log-distance (log kernel product) using vmap.

    Computes:
        log_distance[i, j] = sum_d log N(eval_points[j, d] | samples[i, d], std[d])

    Uses jax.vmap to eliminate Python for-loop over dimensions, enabling full parallelization.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Per-dimension kernel std.

    Returns
    -------
    log_distance : jnp.ndarray, shape (n_samples, n_eval_points)
        Log of the product of per-dimension Gaussian kernels.

    Notes
    -----
    ``std`` is clamped to ``[EPS, inf)`` so a zero bandwidth yields a finite
    (heavily peaked) kernel instead of NaN, matching
    ``log_kde_distance_streaming`` at the edges.
    """
    # Clamp std to avoid division by zero (mirrors log_kde_distance_streaming).
    std = jnp.clip(std, EPS, jnp.inf)

    def log_gaussian_per_dim(eval_dim, sample_dim, sigma):
        return log_gaussian_pdf(
            eval_dim[None, :],  # shape (1, n_eval)
            sample_dim[:, None],  # shape (n_samples, 1)
            sigma,
        )

    # vmap over dimensions: produces (n_dims, n_samples, n_eval)
    per_dim_log_distances = jax.vmap(log_gaussian_per_dim)(
        eval_points.T, samples.T, std
    )

    # Sum over dimensions: (n_samples, n_eval)
    return jnp.sum(per_dim_log_distances, axis=0)


@jax.jit
def log_kde_distance_streaming(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log KDE distance in streaming fashion to avoid D×n_samp×n_eval intermediate.

    This is mathematically equivalent to log_kde_distance but uses a fori_loop over
    dimensions instead of vmap. This avoids materializing a (n_dims, n_samples, n_eval)
    intermediate array, reducing peak memory from O(D×n_samp×n_eval) to O(n_samp×n_eval).

    For large D (many position dimensions), this can significantly reduce memory usage.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval, n_dims)
        Points at which to evaluate the KDE.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Sample points from which to build the KDE.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation for each dimension.

    Returns
    -------
    log_distance : jnp.ndarray, shape (n_samples, n_eval)
        Log of the Gaussian kernel distance for each sample-evaluation pair.

    Notes
    -----
    This function is JIT-compiled and will be specialized for each unique combination
    of input shapes. The shape dimensions (n_dims, n_samp, n_eval) are traced during
    compilation, so different shapes will result in separate compiled versions.

    Memory usage:
    - log_kde_distance (vmap): O(D×n_samp×n_eval) peak
    - log_kde_distance_streaming (fori_loop): O(n_samp×n_eval) peak

    For D=10, n_samp=1000, n_eval=100: 10× memory reduction
    """
    n_dims = eval_points.shape[1]
    n_samp = samples.shape[0]
    n_eval = eval_points.shape[0]

    # Clamp std to avoid division by zero
    std = jnp.clip(std, EPS, jnp.inf)

    # Initialize accumulator: (n_samp, n_eval)
    log_distance_acc = jnp.zeros((n_samp, n_eval))

    def accumulate_dim(dim_idx: int, acc: jnp.ndarray) -> jnp.ndarray:
        """Accumulate log-Gaussian contribution from one dimension."""
        # Extract 1D slices for this dimension
        eval_d = jax.lax.dynamic_slice_in_dim(eval_points, dim_idx, 1, axis=1).squeeze(
            axis=1
        )  # (n_eval,)
        samp_d = jax.lax.dynamic_slice_in_dim(samples, dim_idx, 1, axis=1).squeeze(
            axis=1
        )  # (n_samp,)

        # Compute log Gaussian for this dimension: (n_samp, n_eval)
        logp_d = log_gaussian_pdf(
            eval_d[None, :],  # (1, n_eval) -> broadcast to (n_samp, n_eval)
            samp_d[:, None],  # (n_samp, 1) -> broadcast to (n_samp, n_eval)
            std[dim_idx],
        )

        # Accumulate (sum in log-space is just addition)
        return acc + logp_d

    # Loop over dimensions, accumulating log-distance contributions
    return jax.lax.fori_loop(0, n_dims, accumulate_dim, log_distance_acc)


def _compute_log_mark_kernel_gemm(
    decoding_features: jnp.ndarray,
    encoding_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
) -> jnp.ndarray:
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
    decoding_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
        Waveform features for decoding spikes.
    encoding_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
        Waveform features for encoding spikes.
    waveform_stds : jnp.ndarray, shape (n_features,)
        Standard deviations for each feature dimension.

    Returns
    -------
    logK_mark : jnp.ndarray, shape (n_encoding_spikes, n_decoding_spikes)
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

    def squared_distances(guard_overflow: bool) -> jnp.ndarray:
        sq_dist = jnp.zeros(
            (encoding_features.shape[0], decoding_features.shape[0]),
            dtype=feature_dtype,
        )
        for decoding_dim, encoding_dim, sigma in zip(
            decoding_features.T, encoding_features.T, waveform_stds, strict=True
        ):
            encoding_column, decoding_row = encoding_dim[:, None], decoding_dim[None, :]
            difference = encoding_column - decoding_row
            if guard_overflow:
                overflow = (
                    jnp.isinf(difference)
                    & jnp.isfinite(encoding_column)
                    & jnp.isfinite(decoding_row)
                )
                # A broad sigma can make a finite distance even if subtraction
                # overflows. Only those pairs subtract scaled features. Mask
                # before division to keep inactive terms out of derivatives.
                scaled_difference = jnp.where(overflow, 0.0, difference) / sigma
                scaled_difference += (
                    jnp.where(overflow, encoding_column, 0.0) / sigma
                    - jnp.where(overflow, decoding_row, 0.0) / sigma
                )
            else:
                scaled_difference = difference / sigma
            sq_dist += scaled_difference**2
        return sq_dist

    # No pair can overflow subtraction when both operands are within half the
    # dtype limit. Check the feature arrays once, so ordinary waveforms avoid
    # per-pair overflow masks and the downstream logsumexp does not fuse those
    # masks across every spatial bin.
    half_limit = jnp.finfo(feature_dtype).max / 2
    can_overflow = jnp.any(jnp.abs(encoding_features) > half_limit) | jnp.any(
        jnp.abs(decoding_features) > half_limit
    )
    sq_dist: jnp.ndarray = jax.lax.cond(
        can_overflow,
        lambda _: squared_distances(True),
        lambda _: squared_distances(False),
        None,
    )
    logK_mark = log_norm_const - 0.5 * sq_dist  # (n_enc, n_dec)

    return logK_mark


def _encoding_log_weights(encoding_weights, n_encoding_spikes: int) -> jnp.ndarray:
    """Per-encoding-spike log weight ``log(w_e / sum w)``.

    A zero weight gives ``-inf``, dropping the spike out of every maximum and
    product; a negative or NaN weight gives NaN. ``None`` weights uniformly.

    Parameters
    ----------
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,), or None
    n_encoding_spikes : int

    Returns
    -------
    log_w : jnp.ndarray, shape (n_encoding_spikes,)
    """
    if encoding_weights is None:
        # Use max(n, 1) to avoid log(0); when n=0 the result is unused.
        safe_n = jnp.where(n_encoding_spikes > 0, float(n_encoding_spikes), 1.0)
        return jnp.full((n_encoding_spikes,), -jnp.log(safe_n))
    weight_total = jnp.sum(encoding_weights)
    safe_weight_total = jnp.where(weight_total > 0, weight_total, 1.0)
    safe_weights = jnp.where(encoding_weights > 0, encoding_weights, 1.0)
    log_w = jnp.where(
        encoding_weights == 0,
        -jnp.inf,
        jnp.log(safe_weights) - jnp.log(safe_weight_total),
    )
    return jnp.where(
        (encoding_weights < 0) | jnp.isnan(encoding_weights), jnp.nan, log_w
    )


def _occupancy_normalizer(log_occupancy):
    """Occupancy to divide position kernels by before mark log densities are added.

    Dividing first preserves small conditional mark terms even in very distant
    tails.

    Parameters
    ----------
    log_occupancy : jnp.ndarray, shape (n_position_bins,)

    Returns
    -------
    normalizer : jnp.ndarray, shape (n_position_bins,)
        Log occupancy to subtract from position kernels; 0 where it is zero.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        Occupancy left after the division: 1 where supported, 0 where it is
        zero, NaN where it is NaN.
    """
    normalizer = jnp.where(jnp.isneginf(log_occupancy), 0.0, log_occupancy)
    occupancy = jnp.where(
        jnp.isneginf(log_occupancy),
        0.0,
        jnp.where(jnp.isnan(log_occupancy), jnp.nan, 1.0),
    )
    return normalizer, occupancy


def _tiled_position_peak(log_position_distance, log_w, enc_tile_size, dtype):
    """Largest weighted position kernel per bin, reduced over encoding tiles.

    No encoding x position temporary is formed. The overlapping final tile only
    repeats rows, which a max ignores.
    """
    n_enc, n_pos = log_position_distance.shape

    def tile_peak(enc_idx, peak):
        enc_start = jnp.minimum(enc_idx * enc_tile_size, n_enc - enc_tile_size)
        tile = (
            jax.lax.dynamic_slice_in_dim(
                log_position_distance, enc_start, enc_tile_size
            )
            + jax.lax.dynamic_slice_in_dim(log_w, enc_start, enc_tile_size)[:, None]
        )
        return jnp.maximum(peak, jnp.max(tile, axis=0))

    n_chunks = (n_enc + enc_tile_size - 1) // enc_tile_size
    return jax.lax.fori_loop(
        0, n_chunks, tile_peak, jnp.full((n_pos,), -jnp.inf, dtype)
    )


def _position_scale_terms(log_position_distance, log_w, dtype):
    """Position peak and scaled position factor of the linear reduction.

    They depend on no decoding spike, so a caller scoring many blocks of spikes
    against one encoding model computes them once (see
    ``_estimate_with_enc_chunking``'s ``position_terms``).

    Returns
    -------
    position_peak : jnp.ndarray, shape (n_position_bins,)
    position_factor : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
        ``exp(log_position + log_w - peak + shift)``, as the linear reduction
        evaluates it.
    """
    n_enc = log_position_distance.shape[0]
    log_w = jnp.asarray(log_w, dtype)
    position_peak = _tiled_position_peak(
        log_position_distance,
        log_w,
        min(n_enc, _LOG_REDUCTION_ENCODING_ROWS),
        dtype,
    )
    shift = _scaled_gemm_shift(n_enc, jnp.float32)
    factor = jnp.exp(
        log_position_distance
        + log_w[:, None]
        - _zero_if_neginf(position_peak)[None, :]
        + shift
    )
    return position_peak, factor


def _estimate_with_enc_chunking(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None,
    log_w: jnp.ndarray,
    enc_tile_size: int,
    pos_tile_size: int | None,
    log_background: jnp.ndarray,
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
    use_streaming: bool = False,
    position_log_normalizer: jnp.ndarray | None = None,
    position_terms: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Log joint mark intensity, reduced over bounded encoding tiles.

    Each encoding tile is one scaled GEMM over every query in the block and
    each position tile, so temporaries scale with the tile and the query
    block rather than the encoding count. Streaming builds only an encoding x
    position tile rather than a full spatial kernel.

    Only entries that can change ``logaddexp(log_joint, log_background)`` must
    be exact: those above each spike's floor ``log_background + log(eps)``.
    Every tile is scaled by fixed, kernel-free peaks, so their products add
    linearly. A kernel-free bound flags each spike whose floor may lie below
    the scaled GEMM's window of those peaks (extreme mark or position
    normalizations, or a background far below the kernel peaks). A flagged
    spike is recomputed with a direct logsumexp over each encoding x position
    tile.

    Returns
    -------
    log_joint : jnp.ndarray, shape (n_dec, n_pos)
        Log joint mark intensity, exact above each spike's floor; ``-inf``
        where it is zero.
    """
    n_enc = encoding_spike_waveform_features.shape[0]
    n_dec = decoding_spike_waveform_features.shape[0]
    if use_streaming:
        if (
            position_eval_points is None
            or encoding_positions is None
            or position_std is None
        ):
            raise ValueError(
                "use_streaming=True requires encoding_positions, position_eval_points, and position_std"
            )
        n_pos = position_eval_points.shape[0]
    else:
        if log_position_distance is None:
            raise ValueError("use_streaming=False requires log_position_distance")
        n_pos = log_position_distance.shape[1]
    enc_tile_size = min(n_enc, enc_tile_size, _LOG_REDUCTION_ENCODING_ROWS)
    pos_tile_size = min(n_pos, pos_tile_size or n_pos)
    n_enc_chunks = (n_enc + enc_tile_size - 1) // enc_tile_size
    n_pos_chunks = (n_pos + pos_tile_size - 1) // pos_tile_size
    # Tile kernels promote every input with float32; explicitly typed carries
    # keep the loops' dtypes fixed when x64 is enabled.
    position_inputs = (
        (encoding_positions, position_eval_points, position_std)
        if use_streaming
        else (log_position_distance,)
    )
    if position_log_normalizer is not None:
        position_inputs += (position_log_normalizer,)
    dtype = jnp.result_type(
        jnp.float32,
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
        log_w,
        *position_inputs,
    )
    # A weakly typed weight would turn strong inside the loop bodies.
    log_w = jnp.asarray(log_w, dtype)
    query_floor = jnp.asarray(log_background, dtype) + np.log(jnp.finfo(dtype).eps)
    position_floor = _log_marginal_floor(mean_rate, occupancy)

    # Every log kernel is at most its peak: the Gaussian normalizer for marks
    # (zero distance) and, per position, the largest weighted position kernel.
    # Scaling by these peaks loses under exp(peak - window) per entry, so if
    # that stays below four epsilons of the floor everywhere, no entry above
    # the floor loses mass. float32 is the narrowest GEMM dtype, so its window
    # is the conservative one.
    n_features = encoding_spike_waveform_features.shape[1]
    origin = jnp.zeros((1, n_features), dtype)
    mark_peak = _compute_log_mark_kernel_gemm(origin, origin, waveform_stds)[0, 0]
    if use_streaming:
        position_origin = jnp.zeros((1, encoding_positions.shape[1]), dtype)
        position_peak = log_kde_distance_streaming(
            position_origin, position_origin, position_std
        )[0, 0].astype(dtype) + jnp.max(log_w)
        if position_log_normalizer is not None:
            position_peak = position_peak - position_log_normalizer.astype(dtype)
    elif position_terms is not None:
        # Precomputed once per encoding model by the caller.
        position_peak = position_terms[0]
    else:
        position_peak = _tiled_position_peak(
            log_position_distance, log_w, enc_tile_size, dtype
        )
    # A spike's linear estimate is exact above its floor when every kernel peak
    # lies within the scaled GEMM's window of that floor; other spikes are
    # recomputed exactly, one at a time.
    needs_exact_row = (
        jnp.max(
            mark_peak
            + position_peak
            - _scaled_gemm_window(n_enc, jnp.float32)
            - position_floor
        )
        > query_floor
    )

    def encoding_tile(enc_idx, queries):
        requested_enc_start = enc_idx * enc_tile_size
        enc_start = jnp.minimum(requested_enc_start, n_enc - enc_tile_size)
        marks = jax.lax.dynamic_slice_in_dim(
            encoding_spike_waveform_features, enc_start, enc_tile_size
        )
        log_mark = _compute_log_mark_kernel_gemm(queries, marks, waveform_stds)
        weights = jax.lax.dynamic_slice_in_dim(log_w, enc_start, enc_tile_size)
        # Slice an overlapping final tile and mask its prefix, avoiding a
        # padded copy of the potentially very large precomputed matrix.
        weights = jnp.where(
            enc_start + jnp.arange(enc_tile_size) >= requested_enc_start,
            weights,
            -jnp.inf,
        )
        positions = (
            jax.lax.dynamic_slice_in_dim(encoding_positions, enc_start, enc_tile_size)
            if use_streaming
            else None
        )
        return enc_start, log_mark, weights, positions

    def position_tile(pos_idx, enc_start, positions):
        requested_pos_start = pos_idx * pos_tile_size
        pos_start = jnp.minimum(requested_pos_start, n_pos - pos_tile_size)
        if use_streaming:
            eval_points = jax.lax.dynamic_slice_in_dim(
                position_eval_points, pos_start, pos_tile_size
            )
            # Its accumulator has the default float type; keep the tile dtype.
            log_position = log_kde_distance_streaming(
                eval_points, positions, position_std
            ).astype(dtype)
            if position_log_normalizer is not None:
                log_position = (
                    log_position
                    - jax.lax.dynamic_slice_in_dim(
                        position_log_normalizer.astype(dtype), pos_start, pos_tile_size
                    )[None, :]
                )
        else:
            log_position = jax.lax.dynamic_slice(
                log_position_distance,
                (enc_start, pos_start),
                (enc_tile_size, pos_tile_size),
            )
        # An overlapping final tile repeats earlier positions; only the rest
        # are new.
        is_new = pos_start + jnp.arange(pos_tile_size) >= requested_pos_start
        return pos_start, log_position, is_new

    def reduce_linearly():
        # Every tile shares the kernel-free peaks as fixed scales, so their
        # products add linearly. The scales depend on no decoding spike. The
        # float32 shift matches the gate's window and cannot overflow any
        # wider dtype the operands may use.
        shift = _scaled_gemm_shift(n_enc, jnp.float32)
        safe_position_peak = jnp.broadcast_to(_zero_if_neginf(position_peak), (n_pos,))

        def process_enc_chunk(enc_idx, numerator):
            enc_start, log_mark, weights, positions = encoding_tile(
                enc_idx, decoding_spike_waveform_features
            )
            # Zero-weight rows have zero position factors; NaN marks propagate.
            mark_factor = jnp.exp(log_mark - _zero_if_neginf(mark_peak) + shift)

            def process_position_chunk(pos_idx, numerator):
                pos_start, log_position, is_new = position_tile(
                    pos_idx, enc_start, positions
                )
                if position_terms is None:
                    peak = jax.lax.dynamic_slice_in_dim(
                        safe_position_peak, pos_start, pos_tile_size
                    )
                    position_factor = jnp.exp(
                        log_position + weights[:, None] - peak + shift
                    )
                else:
                    # The same factor, evaluated once per encoding model.
                    position_factor = jax.lax.dynamic_slice(
                        position_terms[1],
                        (enc_start, pos_start),
                        (enc_tile_size, pos_tile_size),
                    )
                    if n_enc % enc_tile_size:
                        # The overlapping final tile's repeated rows (weight
                        # -inf here) must contribute nothing.
                        position_factor = jnp.where(
                            jnp.isneginf(weights)[:, None], 0.0, position_factor
                        )
                product = jnp.matmul(
                    mark_factor.T,
                    position_factor,
                    precision=jax.lax.Precision.HIGHEST,
                )
                current = jax.lax.dynamic_slice_in_dim(
                    numerator, pos_start, pos_tile_size, axis=1
                )
                updated = jnp.where(is_new[None, :], current + product, current)
                return jax.lax.dynamic_update_slice(numerator, updated, (0, pos_start))

            return jax.lax.fori_loop(0, n_pos_chunks, process_position_chunk, numerator)

        numerator = jax.lax.fori_loop(
            0, n_enc_chunks, process_enc_chunk, jnp.zeros((n_dec, n_pos), dtype)
        )
        # The fixed peaks cannot carry a NaN from the marks; keep the product's.
        return (
            log_nonnegative(numerator)
            + _zero_if_neginf(mark_peak)
            + safe_position_peak[None, :]
            - 2.0 * shift
        )

    def exact_row(row):
        # One spike's tile is only encoding x position, so it is reduced by a
        # direct logsumexp, with no scaling to validate.
        query = jax.lax.dynamic_slice_in_dim(decoding_spike_waveform_features, row, 1)

        def process_enc_chunk(enc_idx, marginal):
            enc_start, log_mark, weights, positions = encoding_tile(enc_idx, query)

            def process_position_chunk(pos_idx, marginal):
                pos_start, log_position, is_new = position_tile(
                    pos_idx, enc_start, positions
                )
                contribution = _masked_logsumexp(
                    log_mark + log_position + weights[:, None]
                )
                current = jax.lax.dynamic_slice_in_dim(
                    marginal, pos_start, pos_tile_size, axis=1
                )
                combined = jnp.where(
                    is_new[None, :],
                    _masked_logaddexp(current, contribution[None, :]),
                    current,
                )
                return jax.lax.dynamic_update_slice(marginal, combined, (0, pos_start))

            return jax.lax.fori_loop(0, n_pos_chunks, process_position_chunk, marginal)

        return jax.lax.fori_loop(
            0, n_enc_chunks, process_enc_chunk, jnp.full((1, n_pos), -jnp.inf, dtype)
        )

    def repaired(marginal, row, needs_exact):
        # A scalar cond on one row; the full estimate is updated in place.
        values = jax.lax.cond(
            needs_exact,
            lambda: exact_row(row),
            lambda: jax.lax.dynamic_slice_in_dim(marginal, row, 1),
        )
        return jax.lax.dynamic_update_slice_in_dim(marginal, values, row, 0)

    n_flagged = jnp.sum(needs_exact_row)
    flagged_rows = jnp.nonzero(needs_exact_row, size=n_dec, fill_value=0)[0]
    n_slots = min(n_dec, _EXACT_REPAIR_SLOTS)

    def repair(n_visits, marginal):
        # Visit the flagged rows; static bounds keep the loop differentiable.
        return jax.lax.fori_loop(
            0,
            n_visits,
            lambda slot, current: repaired(
                current, flagged_rows[slot], slot < n_flagged
            ),
            marginal,
        )

    marginal = jax.lax.cond(
        n_flagged == 0,
        lambda current: current,
        lambda current: jax.lax.cond(
            n_flagged <= n_slots,
            partial(repair, n_slots),
            partial(repair, n_dec),
            current,
        ),
        reduce_linearly(),
    )
    return _log_joint_from_log_marginal(marginal, mean_rate, occupancy)


def _encoding_supported(log_w, encoding_features, waveform_stds, *position_inputs):
    """Whether an encoding model can give a nonzero or NaN intensity.

    With every weight zero the intensity is exactly zero, so the reduction can be
    skipped; NaN inputs must still reach it so that they propagate.
    """
    supported = (
        jnp.any(jnp.isfinite(log_w))
        | jnp.any(jnp.isnan(log_w))
        | jnp.any(jnp.isnan(encoding_features))
        | jnp.any(jnp.isnan(waveform_stds))
    )
    for position_input in position_inputs:
        supported |= jnp.any(jnp.isnan(position_input))
    return supported


def _score_spikes(
    supported,
    decoding_features,
    encoding_features,
    waveform_stds,
    occupancy,
    mean_rate,
    log_position_distance,
    log_w,
    enc_tile_size,
    pos_tile_size,
    log_background,
    **reduction_options,
):
    """Each spike's term ``logaddexp(log intensity, log_background)``.

    Arguments are those of ``_estimate_with_enc_chunking`` after
    ``supported`` (see ``_encoding_supported``); an unsupported model scores
    the background without reducing.

    Returns
    -------
    log_terms : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)
    """
    log_joint = jax.lax.cond(
        supported | jnp.any(jnp.isnan(decoding_features)),
        lambda: _estimate_with_enc_chunking(
            decoding_features,
            encoding_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            log_w,
            enc_tile_size,
            pos_tile_size,
            log_background,
            **reduction_options,
        ),
        lambda: _log_joint_from_log_marginal(
            jnp.full((decoding_features.shape[0], occupancy.shape[0]), -jnp.inf),
            mean_rate,
            occupancy,
        ),
    )
    return jnp.logaddexp(log_joint, log_background[:, None])


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None = None,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
    encoding_weights: jnp.ndarray | None = None,
    log_occupancy: jnp.ndarray | None = None,
    *,
    log_background: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    log_position_distance : jnp.ndarray | None, shape (n_encoding_spikes, n_position_bins)
        Log-space position kernel (output of log_kde_distance). Required if use_streaming=False.
        Using log-space prevents underflow from multi-dimensional Gaussian products.
    use_gemm : bool, optional
        If True (default), reduce bounded encoding tiles with scaled GEMMs (see
        Notes). If False, sum kernel products in linear space, a reference
        that underflows in the far tails.
    pos_tile_size : int | None, optional
        If provided, split each encoding tile's GEMM over position chunks of this
        size (only for use_gemm=True). By default all positions share one GEMM.
    enc_tile_size : int | None, optional
        Encoding rows per tile (only for use_gemm=True). Tiles never exceed 2048
        rows, so peak memory is O(tile * (n_decoding_spikes + n_position_bins))
        plus the output, independent of the encoding count.
    use_streaming : bool, optional, default=False
        If True, compute position kernel on-the-fly per encoding chunk (streaming mode).
        Avoids materializing full (n_enc × n_pos) position distance matrix.
        Requires encoding_positions, position_eval_points, and position_std.
        Only valid with enc_tile_size (requires chunking).
    encoding_positions : jnp.ndarray | None, shape (n_encoding_spikes, n_position_dims)
        Encoding spike positions. Required if use_streaming=True.
    position_eval_points : jnp.ndarray | None, shape (n_position_bins, n_position_dims)
        Position evaluation points (e.g., interior_place_bin_centers). Required if use_streaming=True.
    position_std : jnp.ndarray | None, shape (n_position_dims,)
        Position standard deviations. Required if use_streaming=True.
    encoding_weights : jnp.ndarray | None, shape (n_encoding_spikes,), optional
        Per-encoding-spike weights (the posterior weight at each spike time). When
        None (default), every spike is weighted uniformly (``1 / n``). When given,
        each spike contributes ``w_e / sum_w``; a zero weight drops that spike out
        of the reduction entirely. Applied identically on every numerical path.
    log_background : jnp.ndarray, shape (n_decoding_spikes,), keyword-only
        Log background intensity of each decoding spike, in the units of
        ``mean_rate``: ``log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS) + log q(mark)``
        for a mark reference ``q``.

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)
        ``logaddexp(log intensity, log_background)`` on both paths. A zero
        intensity (no support, zero rate) leaves the background. Values are exact
        to float32 roundoff regardless of the other decoding spikes in the call;
        ``NaN`` inputs propagate.

    Notes
    -----
    The GEMM path scales mark and position factors by kernel-free peaks (the
    mark normalizer and each bin's largest weighted position kernel), shifts
    them to use the exponent range, and adds encoding tiles linearly. An
    intensity below ``log_background + log(eps)`` cannot change the result, so
    only intensities above that floor must be exact. When the peaks exceed the
    floor by more than the GEMM's window of about 100 nats, which needs an
    extreme normalization such as tiny bandwidths in many dimensions or a
    background far below the peaks, that spike is recomputed with a direct
    logsumexp. Under ``vmap`` the repair runs for every spike, so map over
    blocks with ``jax.lax.map`` instead.

    The module-level ``estimate_log_joint_mark_intensity`` name is rebound just
    below its definition to a jitted version with
    ``static_argnames=('use_gemm', 'pos_tile_size', 'enc_tile_size',
    'use_streaming')`` — so callers get JIT automatically and do not need to
    wrap it themselves. The undecorated function object remains accessible via
    ``estimate_log_joint_mark_intensity.__wrapped__`` if an un-jitted version is
    needed.
    """
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]
    position_log_normalizer = None

    # Reject unsupported combinations before any array work so the failure is a
    # clear message rather than a downstream exp(None) TypeError.
    if use_streaming and not use_gemm:
        raise ValueError(
            "use_streaming=True is not supported with use_gemm=False "
            "(the linear-space reference path). Use use_gemm=True for streaming."
        )

    if not use_gemm:
        # Linear-space reference: convert log position back to linear for the
        # matrix multiply.
        if log_occupancy is not None:
            normalizer, occupancy = _occupancy_normalizer(log_occupancy)
            log_position_distance = log_position_distance - normalizer[None, :]
        position_distance = jnp.exp(log_position_distance)

        spike_waveform_feature_distance = kde_distance(
            decoding_spike_waveform_features,
            encoding_spike_waveform_features,
            waveform_stds,
        )  # shape (n_encoding_spikes, n_decoding_spikes)

        if encoding_weights is None:
            # Uniform weights: divide the summed kernel by the spike count.
            # Double-where: substitute safe denominator, then select result.
            safe_n = jnp.where(n_encoding_spikes > 0, n_encoding_spikes, 1)
            marginal_density = jnp.where(
                n_encoding_spikes > 0,
                jnp.matmul(
                    spike_waveform_feature_distance.T,
                    position_distance,
                    precision=jax.lax.Precision.HIGHEST,
                )
                / safe_n,
                0.0,
            )  # shape (n_decoding_spikes, n_position_bins)
        else:
            # Weighted average: weight each encoding spike's outer product by its
            # per-spike weight, normalized by the total weight (uniform weights
            # recover the /n form above).
            weight_total = jnp.sum(encoding_weights)
            safe_weight_total = jnp.where(weight_total > 0, weight_total, 1.0)
            marginal_density = jnp.where(
                weight_total > 0,
                jnp.matmul(
                    spike_waveform_feature_distance.T,
                    encoding_weights[:, None] * position_distance,
                    precision=jax.lax.Precision.HIGHEST,
                )
                / safe_weight_total,
                0.0,
            )  # shape (n_decoding_spikes, n_position_bins)
        log_joint = log_nonnegative(
            mean_rate
            * jnp.where(
                occupancy > 0.0,
                marginal_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                0.0,
            )
        )
        return jnp.logaddexp(log_joint, log_background[:, None])

    # Validation: streaming requires chunking configuration and inputs.
    if use_streaming:
        if enc_tile_size is None:
            raise ValueError(
                "use_streaming=True requires enc_tile_size to be specified"
            )
        if (
            encoding_positions is None
            or position_eval_points is None
            or position_std is None
        ):
            raise ValueError(
                "use_streaming=True requires encoding_positions, position_eval_points, "
                "and position_std to be specified"
            )
        if enc_tile_size >= n_encoding_spikes:
            # This group has fewer encoding spikes than the tile, so there is
            # nothing to chunk and streaming's memory savings are moot at this
            # size. Materialize the (small) position kernel and fall back to the
            # non-chunked path instead of raising — a single small group must
            # not be able to abort a whole prediction over heterogeneous groups.
            log_position_distance = log_kde_distance(
                position_eval_points, encoding_positions, position_std
            )
            use_streaming = False

    # Log-space computation with GEMM optimization
    if use_streaming:
        # When streaming, we don't use log_position_distance at all
        n_pos = position_eval_points.shape[0]
    else:
        n_pos = log_position_distance.shape[1]
    n_dec = decoding_spike_waveform_features.shape[0]

    if log_occupancy is not None:
        normalizer, occupancy = _occupancy_normalizer(log_occupancy)
        if use_streaming:
            position_log_normalizer = normalizer
        else:
            log_position_distance = log_position_distance - normalizer[None, :]

    if n_encoding_spikes == 0 or n_dec == 0 or n_pos == 0:
        return jnp.logaddexp(
            _log_joint_from_log_marginal(
                jnp.full((n_dec, n_pos), -jnp.inf), mean_rate, occupancy
            ),
            log_background[:, None],
        )

    # Per-encoding-spike log weight, added to the position kernel (not folded into the
    # mark kernel).
    log_w = _encoding_log_weights(encoding_weights, n_encoding_spikes)

    # The encoding tile is capped inside the reduction; positions are split only
    # when the caller asks.
    reduction_enc_rows = (
        max(n_encoding_spikes, 1) if enc_tile_size is None else enc_tile_size
    )
    position_inputs = (
        (encoding_positions, position_eval_points, position_std)
        if use_streaming
        else (log_position_distance,)
    )
    if position_log_normalizer is not None:
        position_inputs += (position_log_normalizer,)
    return _score_spikes(
        _encoding_supported(
            log_w, encoding_spike_waveform_features, waveform_stds, *position_inputs
        ),
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        log_w,
        reduction_enc_rows,
        pos_tile_size,
        log_background,
        encoding_positions=encoding_positions,
        position_eval_points=position_eval_points,
        position_std=position_std,
        use_streaming=use_streaming,
        position_log_normalizer=position_log_normalizer,
    )


# JIT-compile with static arguments for performance
# This allows JAX to specialize the function for different tile sizes and modes
estimate_log_joint_mark_intensity = jax.jit(
    estimate_log_joint_mark_intensity,
    static_argnames=("use_gemm", "pos_tile_size", "enc_tile_size", "use_streaming"),
)


def _validate_tile_sizes(enc_tile_size, pos_tile_size) -> None:
    """Reject tile sizes that are not None or a positive integer."""
    for name, value in (
        ("enc_tile_size", enc_tile_size),
        ("pos_tile_size", pos_tile_size),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValidationError(
                f"{name} must be None or a positive integer", got=repr(value)
            )


def _pad_with_last_row(array, n_rows: int) -> np.ndarray:
    """Repeat the last row on the host to reach ``n_rows`` (keeps block maxima)."""
    array = np.asarray(array)
    return np.concatenate([array, np.repeat(array[-1:], n_rows - len(array), axis=0)])


@partial(
    jax.jit,
    static_argnames=("block_size", "pos_tile_size", "indices_are_sorted"),
    donate_argnums=0,
)
def _add_electrode_log_mark_intensities(
    total: jnp.ndarray,
    decoding_features: jnp.ndarray,
    row_ids: jnp.ndarray,
    encoding_features: jnp.ndarray,
    encoding_positions: jnp.ndarray,
    encoding_weights: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    position_std: jnp.ndarray,
    place_bin_centers: jnp.ndarray,
    mean_rate: float,
    log_occupancy: jnp.ndarray,
    log_background: jnp.ndarray,
    *,
    block_size: int,
    pos_tile_size: int | None,
    indices_are_sorted: bool,
) -> jnp.ndarray:
    """Add one electrode's non-local log marked intensities to their rows.

    Parameters
    ----------
    total : jnp.ndarray, shape (n_rows, n_bins)
        Running sum over electrodes; donated.
    decoding_features : jnp.ndarray, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows. Padding repeats the last spike; each
        spike is reduced independently, so padding cannot change real rows.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    encoding_features : jnp.ndarray, shape (n_padded_samples, n_features)
    encoding_positions : jnp.ndarray, shape (n_padded_samples, n_position_dims)
    encoding_weights : jnp.ndarray, shape (n_padded_samples,)
        Zero for padding samples, which then drop out of every reduction.
    waveform_stds : jnp.ndarray, shape (n_features,)
    position_std : jnp.ndarray, shape (n_position_dims,)
    place_bin_centers : jnp.ndarray, shape (n_bins, n_position_dims)
    mean_rate : float
        Mean rate times ``RATE_REFERENCE_SECONDS``.
    log_occupancy : jnp.ndarray, shape (n_bins,)
    log_background : jnp.ndarray, shape (n_padded_spikes,)
        Log background intensity of each spike, in the units of ``mean_rate``.

    Returns
    -------
    total : jnp.ndarray, shape (n_rows, n_bins)
    """
    log_position_distance = log_kde_distance(
        place_bin_centers, encoding_positions, std=position_std
    )
    # Everything that depends on no decoding spike is prepared once for the
    # electrode, not once per block of spikes.
    normalizer, occupancy = _occupancy_normalizer(log_occupancy)
    log_position_distance = log_position_distance - normalizer[None, :]
    n_encoding = encoding_features.shape[0]
    n_bins = occupancy.shape[0]
    log_w = _encoding_log_weights(encoding_weights, n_encoding)
    dtype = jnp.result_type(
        jnp.float32,
        decoding_features,
        encoding_features,
        waveform_stds,
        log_w,
        log_position_distance,
    )
    position_terms = _position_scale_terms(log_position_distance, log_w, dtype)
    supported = _encoding_supported(
        log_w, encoding_features, waveform_stds, log_position_distance
    )

    def score_block(block):
        features, background = block
        return _score_spikes(
            supported,
            features,
            encoding_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            log_w,
            max(n_encoding, 1),
            pos_tile_size,
            background,
            position_terms=position_terms,
        )

    n_blocks = decoding_features.shape[0] // block_size
    intensities = jax.lax.map(
        score_block,
        (
            decoding_features.reshape(n_blocks, block_size, -1),
            log_background.reshape(n_blocks, block_size),
        ),
    ).reshape(-1, n_bins)
    return total + deterministic_row_sum(
        intensities,
        row_ids,
        total.shape[0],
        indices_are_sorted=indices_are_sorted,
    )


@partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "gpi_block_size",
        "indices_are_sorted",
    ),
    donate_argnums=(0, 1),
)
def _add_electrode_local_terms(
    log_likelihood: jnp.ndarray,
    expected_counts: jnp.ndarray,
    spike_positions: jnp.ndarray,
    decoding_features: jnp.ndarray,
    row_ids: jnp.ndarray,
    encoding_positions: jnp.ndarray,
    encoding_features: jnp.ndarray,
    encoding_weights: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    log_occupancy_at_spikes: jnp.ndarray,
    gpi_samples: jnp.ndarray,
    gpi_weights: jnp.ndarray,
    gpi_std: jnp.ndarray,
    positions: jnp.ndarray,
    log_occupancy: jnp.ndarray,
    scaled_rate: float,
    mean_rate: float,
    log_background: jnp.ndarray,
    *,
    block_size: int,
    gpi_block_size: int,
    indices_are_sorted: bool,
    log_gpi_density: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add one electrode's local spike terms and expected count rate.

    Parameters
    ----------
    log_likelihood, expected_counts : jnp.ndarray, shape (n_rows,)
        Running sums over electrodes; donated.
    spike_positions : jnp.ndarray, shape (n_padded_spikes, n_position_dims)
        Position at each decoding spike; padding rows are zero.
    decoding_features : jnp.ndarray, shape (n_padded_spikes, n_features)
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    encoding_positions, encoding_features, encoding_weights
        Padded joint KDE samples and weights (zero for padding).
    position_std, waveform_stds : jnp.ndarray
    log_occupancy_at_spikes : jnp.ndarray, shape (n_padded_spikes,)
        True log occupancy density at ``spike_positions``.
    gpi_samples, gpi_weights, gpi_std
        This electrode's padded ground-process KDE leaves.
    positions : jnp.ndarray, shape (n_rows, n_position_dims)
        Animal position at each row.
    log_occupancy : jnp.ndarray, shape (n_rows,)
        Log occupancy density at ``positions``.
    scaled_rate : float
        Mean rate times ``RATE_REFERENCE_SECONDS``.
    mean_rate : float
        Mean rate in Hz.
    log_background : jnp.ndarray, shape (n_padded_spikes,)
        Log background intensity of each spike, in the units of ``scaled_rate``.

    Returns
    -------
    log_likelihood, expected_counts : jnp.ndarray, shape (n_rows,)
    """
    # See compute_local_log_likelihood for the per-spike combine.
    log_ratio = local_kde_log_ratio(
        spike_positions,
        decoding_features,
        encoding_positions,
        encoding_features,
        position_std,
        waveform_stds,
        log_occupancy_at_spikes,
        encoding_weights,
        block_size,
    )
    spike_contribution = jnp.logaddexp(
        log_density_intensity(scaled_rate, log_ratio, jnp.zeros_like(log_ratio)),
        log_background,
    )
    log_likelihood = log_likelihood + deterministic_row_sum(
        spike_contribution,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )
    if log_gpi_density is None:
        log_gpi_density = traced_kde_log_density(
            positions, gpi_samples, gpi_std, gpi_block_size, gpi_weights
        )
    expected_counts = expected_counts + jnp.exp(
        log_density_intensity(mean_rate, log_gpi_density, log_occupancy)
    )
    return log_likelihood, expected_counts


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None = None,
    block_size: int = 100,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
    encoding_weights: jnp.ndarray | None = None,
    log_occupancy: jnp.ndarray | None = None,
    *,
    log_background: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    log_position_distance : jnp.ndarray | None, shape (n_encoding_spikes, n_position_bins)
        Log-space position kernel. Prevents underflow in multi-dimensional position spaces.
        Can be None if use_streaming=True.
    block_size : int, optional
        Number of decoding spikes to process per block.
    use_gemm : bool, optional
        If True (default), use the scaled-GEMM reduction.
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension.
    enc_tile_size : int | None, optional
        If provided, tile computation over encoding spikes dimension for memory efficiency.
    use_streaming : bool, optional
        If True, compute position kernel on-the-fly using streaming log_kde_distance.
        Requires enc_tile_size, encoding_positions, position_eval_points, position_std.
    encoding_positions : jnp.ndarray | None, shape (n_encoding_spikes, n_position_dims)
        Required when use_streaming=True. Positions where encoding spikes occurred.
    position_eval_points : jnp.ndarray | None, shape (n_position_bins, n_position_dims)
        Required when use_streaming=True. Positions to evaluate (e.g., place bin centers).
    position_std : jnp.ndarray | None, shape (n_position_dims,)
        Required when use_streaming=True. Position kernel bandwidth per dimension.
    encoding_weights : jnp.ndarray | None, shape (n_encoding_spikes,), optional
        Per-encoding-spike weights forwarded to the per-block estimator. None
        (default) weights every encoding spike uniformly.
    log_background : jnp.ndarray, shape (n_decoding_spikes,), keyword-only
        Log background intensity of each decoding spike (see
        ``estimate_log_joint_mark_intensity``).

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)
        ``logaddexp(log intensity, log_background)``.
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]
    log_background = jnp.asarray(log_background)

    if n_decoding_spikes == 0:
        return jnp.zeros((0, n_position_bins))
    # A large configured block must not pad a sparse chunk up to its capacity.
    block_size = min(block_size, n_decoding_spikes)

    # Process decoding spikes in fixed-size blocks. Each decoding spike is
    # computed independently, so the final partial block is padded up to
    # block_size (and its extra rows sliced off) to keep every call to the
    # jitted estimator at the same shape — avoiding a recompile for the
    # leftover block. Results are collected and concatenated once instead of
    # O(n_blocks) full-array dynamic_update_slice copies.
    #
    # Pad by repeating the last real row (mode="edge"). The reduction's scales
    # never depend on other decoding spikes, so padding cannot change real rows.
    block_results = []
    for start_ind in range(0, n_decoding_spikes, block_size):
        block = decoding_spike_waveform_features[start_ind : start_ind + block_size]
        background = log_background[start_ind : start_ind + block_size]
        actual_len = block.shape[0]
        if actual_len < block_size:
            block = jnp.pad(block, ((0, block_size - actual_len), (0, 0)), mode="edge")
            background = jnp.pad(background, (0, block_size - actual_len), mode="edge")
        block_result = estimate_log_joint_mark_intensity(
            block,
            encoding_spike_waveform_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=use_gemm,
            pos_tile_size=pos_tile_size,
            enc_tile_size=enc_tile_size,
            use_streaming=use_streaming,
            encoding_positions=encoding_positions,
            position_eval_points=position_eval_points,
            position_std=position_std,
            encoding_weights=encoding_weights,
            log_occupancy=log_occupancy,
            log_background=background,
        )
        block_results.append(block_result[:actual_len])

    return jnp.concatenate(block_results, axis=0)


def fit_clusterless_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    weights: jnp.ndarray | None = None,
    position_std: float = np.sqrt(12.5),
    waveform_std: float = 24.0,
    block_size: int = 100,
    enc_tile_size: int | None = None,
    pos_tile_size: int | None = None,
    use_streaming: bool = False,
    disable_progress_bar: bool = False,
    *,
    encoding_time_range=None,
    valid_position_intervals=None,
    _encoding_support=None,
) -> dict:
    """Fit the clusterless KDE encoding model.

    Parameters
    ----------
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Spike waveform features for each electrode.
    environment : Environment
        The spatial environment.
    weights : jnp.ndarray | None, shape (n_time_position,), optional
        Per-position-sample weights (e.g. the EM posterior for this encoding
        group), by default None (uniform). They weight the occupancy KDE, the
        per-electrode ground-process KDE, and the mean rate, and are interpolated
        onto each spike's time to form the ``encoding_weights`` used at decode time.
    position_std : float, optional
        Gaussian smoothing standard deviation for position, by default sqrt(12.5)
    waveform_std : float, optional
        Gaussian smoothing standard deviation for waveform, by default 24.0
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds. Clips encoding support using the
        original interpolation basis. Uniform samples otherwise include endpoint
        half-cells; a singleton requires explicit bounds or a tracking interval.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
        Required for irregular timestamps; each interval needs a finite sample.
        Positions are held at segment endpoints and NaN rows split support.
        Encoding support does not automatically mark decode bins missing.

    Returns
    -------
    encoding_model : dict

    Notes
    -----
    Fitted mean rates and ground-process intensities are Hz. Occupancy weights
    integrate the interpolation basis over valid encoding support in seconds;
    event weights are dimensionless. Decode bins may differ from tracking samples.
    """
    _validate_tile_sizes(enc_tile_size, pos_tile_size)
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    if environment.place_bin_centers_ is None:
        raise ValueError(
            "Environment must be fitted with place_bin_centers_. "
            "Call environment.fit_place_grid() first."
        )

    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = validate_weights(np.asarray(weights), position.shape[0])
    # Weighted occupancy "time": the sum of per-sample weights (uniform weights
    # recover the training-sample count). Gaps from is_training / encoding-group
    # masks carry weight 0 and are not charged as occupancy time.
    weight_sum = float(exposure_weights.sum())
    # A track graph with multi-dim position linearizes occupancy to 1D, so the
    # bandwidth is a single dimension there; otherwise one per position column.
    n_std_dims = (
        1
        if (environment.track_graph is not None and position.shape[1] > 1)
        else position.shape[1]
    )
    # Validate bandwidths once, host-side, at fit time: a zero, negative or
    # non-finite std would otherwise fit a degenerate kernel. waveform_std stays
    # scalar or per-feature and is expanded per electrode at predict time.
    position_std = validate_bandwidth(position_std, n_std_dims, "position_std")
    for features in spike_waveform_features:
        validate_bandwidth(waveform_std, features.shape[1], "waveform_std")

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    occupancy_samples, occupancy_weights = drop_zero_weight_samples(
        position, exposure_weights
    )
    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        occupancy_samples = environment._linear_position(occupancy_samples)[:, None]
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
        occupancy_samples, weights=jnp.asarray(occupancy_weights)
    )

    log_occupancy = kde_log_density(occupancy_model, interior_place_bin_centers)
    # The probability density is the exponential of the true log density.
    occupancy = jnp.exp(log_occupancy)
    encoding_positions = []
    encoding_weights = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)
    bounded_spike_waveform_features = []
    mark_reference: list[MarkReference | None] = []

    for electrode, (
        electrode_spike_waveform_features,
        electrode_spike_times,
    ) in enumerate(
        zip(
            tqdm(
                spike_waveform_features,
                desc="Encoding models",
                unit="electrode",
                disable=disable_progress_bar,
            ),
            spike_times,
            strict=True,
        )
    ):
        is_in_bounds = support.contains(electrode_spike_times)
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        # Validate only the in-window spikes that actually enter the fit; a
        # non-finite feature on a spike outside the encoding interval is
        # discarded here anyway and must not abort fitting.
        bounded_features = electrode_spike_waveform_features[is_in_bounds]
        validate_finite(bounded_features, "spike_waveform_features")
        bounded_spike_waveform_features.append(bounded_features)
        # Weight each encoding spike by the posterior weight at its spike time.
        electrode_weights_host = interpolate_weights_at_spike_times(
            electrode_spike_times,
            position_time,
            np.asarray(weights),
            encoding_support=support,
        )
        electrode_weights = jnp.asarray(electrode_weights_host)
        encoding_weights.append(electrode_weights)
        # Weighted mean rate: weighted spike count / weighted occupancy time. Sum on the
        # host array (as the GMM fit does) to avoid a device->host sync per electrode.
        mean_rates.append(weighted_mean_rate(electrode_weights_host, weight_sum))
        mark_reference.append(
            fit_mark_reference(
                np.asarray(bounded_features),
                electrode_weights_host,
                waveform_std,
                electrode,
            )
        )
        encoding_positions.append(
            get_position_at_time(
                position_time,
                position,
                electrode_spike_times,
                environment,
                encoding_support=support,
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1], weights=electrode_weights
        )
        gpi_models.append(gpi_model)

    # One batched pass per group of similar sample counts, instead of a
    # compiled kernel for every electrode's sample count.
    for mean_rate, log_gpi_density in zip(
        mean_rates,
        _kde_log_densities(gpi_models, interior_place_bin_centers),
        strict=True,
    ):
        summed_ground_process_intensity += jnp.exp(
            log_density_intensity(mean_rate, log_gpi_density, log_occupancy)
        )

    # Every electrode, including zero-rate ones, adds its background rate.
    summed_ground_process_intensity = (
        summed_ground_process_intensity + len(spike_times) * RATE_EPS_HZ
    )

    return {
        "occupancy": occupancy,
        "log_occupancy": log_occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
        "encoding_weights": encoding_weights,
        "environment": environment,
        "mean_rates": mean_rates,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "position_std": position_std,
        "waveform_std": waveform_std,
        "block_size": block_size,
        "disable_progress_bar": disable_progress_bar,
        "enc_tile_size": enc_tile_size,
        "pos_tile_size": pos_tile_size,
        "use_streaming": use_streaming,
        "mark_reference": mark_reference,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


@requires_time_edges
def predict_clusterless_kde_log_likelihood(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy: jnp.ndarray,
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: list[jnp.ndarray],
    environment: Environment,
    mean_rates: jnp.ndarray,
    summed_ground_process_intensity: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    is_local: bool = False,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    enc_tile_size: int | None = None,
    pos_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_weights: list[jnp.ndarray] | None = None,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,
    log_occupancy: jnp.ndarray | None = None,
    mark_reference: list[MarkReference | None] | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Predict the log likelihood of the clusterless KDE model.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample (used only by the local path; accepted for
        signature parity when ``is_local`` is False).
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples (used only by the local path; accepted for signature
        parity when ``is_local`` is False).
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Waveform features for each electrode.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        How much time is spent in each position bin by the animal.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jnp.ndarray]
        Spike waveform features for each electrode used for encoding.
    encoding_positions : list[jnp.ndarray]
        Per-electrode encoding positions, each of shape
        (n_encoding_spikes, n_position_dims).
    environment : Environment
        The spatial environment
    mean_rates : jnp.ndarray, shape (n_electrodes,)
        Mean firing rate for each electrode.
    summed_ground_process_intensity : jnp.ndarray, shape (n_position_bins,)
        Summed ground process intensity for all electrodes.
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    is_local : bool, optional
        If True, compute the log likelihood at the animal's position, by default False
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False
    enc_tile_size : int | None, optional
        Encoding samples are always reduced in tiles of at most 2048 rows; a
        smaller value lowers peak memory further and selects the block-wise path.
        By default None.
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension to reduce memory.
        By default None (no tiling).
    use_streaming : bool, optional
        If True, compute position kernel on-the-fly per encoding chunk (streaming mode).
        Avoids materializing full (n_enc × n_pos) position distance matrix.
        Provides D× memory reduction where D is position dimensionality.
        Requires enc_tile_size to be specified and < n_enc. By default False.
    encoding_weights : list[jnp.ndarray] | None, optional
        Per-electrode encoding-spike weights (the ``encoding_weights`` returned by
        the fit). Each entry has shape (n_encoding_spikes,) and weights that
        electrode's joint mark intensity. By default None (uniform weights).
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges: spikes are binned
        against them and only those owned by the requested rows are evaluated, so
        the result equals the full-time result sliced by ``row_slice`` while the
        spatial workspaces scale with the requested rows and selected spikes.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    mark_reference : list of MarkReference or None, keyword-only
        Normalized reference mark density per electrode, taken from the encoding
        model. Each spike's intensity gains ``RATE_EPS_HZ * q(mark)`` and each
        electrode's expected count ``RATE_EPS_HZ``, so likelihood ratios between
        states do not depend on the waveform units. None for an electrode means
        no training marks; its background is the bare rate.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, 1) or (n_rows, n_position_bins)
        Shape depends on whether local or non-local decoding, respectively.
        ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    if rate_units != "Hz":
        raise ValidationError(
            "Encoding rates must be in Hz; refit legacy encoding models before decoding."
        )
    _validate_tile_sizes(enc_tile_size, pos_tile_size)
    if mark_reference is None:
        raise ValidationError(
            "Encoding model has no mark reference; refit before decoding.",
            hint="Refit with fit_clusterless_kde_encoding_model.",
        )
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        gpi_models=gpi_models,
        encoding_spike_waveform_features=encoding_spike_waveform_features,
        encoding_positions=encoding_positions,
        mean_rates=mean_rates,
        encoding_weights=encoding_weights,
        mark_reference=mark_reference,
    )
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    # Uniform (None) weights per electrode when the caller passes none, so the loop
    # and the local path can zip a weight per electrode uniformly.
    per_electrode_weights = (
        encoding_weights
        if encoding_weights is not None
        else [None] * len(encoding_spike_waveform_features)
    )

    if is_local:
        log_likelihood = compute_local_log_likelihood(
            time_edges,
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            occupancy_model,
            gpi_models,
            encoding_spike_waveform_features,
            encoding_positions,
            environment,
            mean_rates,
            position_std,
            waveform_std,
            block_size,
            disable_progress_bar,
            per_electrode_weights,
            row_slice=row_slice,
            mark_reference=mark_reference,
            _spike_time_order=_spike_time_order,
            _time_grid=_time_grid,
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]
        if log_occupancy is None:
            log_occupancy = kde_log_density(occupancy_model, interior_place_bin_centers)

        log_likelihood = (
            -jnp.asarray(_time_grid.durations(row_start, row_stop))[:, None]
            * summed_ground_process_intensity
        )

        for (
            electrode_encoding_spike_waveform_features,
            electrode_encoding_positions,
            electrode_mean_rate,
            electrode_decoding_spike_waveform_features,
            electrode_spike_times,
            electrode_encoding_weights,
            electrode_mark_reference,
        ) in zip(
            tqdm(
                encoding_spike_waveform_features,
                unit="electrode",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_positions,
            mean_rates,
            spike_waveform_features,
            spike_times,
            per_electrode_weights,
            mark_reference,
            strict=True,
        ):
            selection = select_spikes_in_rows(
                electrode_spike_times,
                row_start,
                row_stop,
                time_edges=time_edges,
                _spike_time_order=_spike_time_order,
            )
            electrode_decoding_spike_waveform_features = select_spike_rows(
                electrode_decoding_spike_waveform_features, selection
            )
            log_background = background_log_intensity(
                electrode_mark_reference,
                np.asarray(electrode_decoding_spike_waveform_features),
                LOG_EPS,  # log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS)
            )
            # Expand waveform_std to match this electrode's feature count if scalar
            n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]
            electrode_waveform_std = as_std_array(waveform_std, n_waveform_features)

            if not use_streaming and enc_tile_size is None:
                n_spikes = electrode_decoding_spike_waveform_features.shape[0]
                if n_spikes == 0:
                    continue
                # Padded spikes and zero-weight encoding samples let electrodes
                # and chunks with nearby sizes share one compiled kernel.
                n_padded = _padded_spike_count(n_spikes, block_size)
                padded_features, padded_weights = _padded_samples(
                    _spike_time_order,
                    electrode_encoding_spike_waveform_features,
                    electrode_encoding_weights,
                )
                padded_positions, _ = _padded_samples(
                    _spike_time_order,
                    electrode_encoding_positions,
                    electrode_encoding_weights,
                )
                log_likelihood = _add_electrode_log_mark_intensities(
                    log_likelihood,
                    jnp.asarray(
                        _pad_with_last_row(
                            electrode_decoding_spike_waveform_features, n_padded
                        )
                    ),
                    jnp.asarray(spike_row_ids(selection, n_padded)),
                    padded_features,
                    padded_positions,
                    padded_weights,
                    electrode_waveform_std,
                    jnp.asarray(position_std),
                    interior_place_bin_centers,
                    electrode_mean_rate * RATE_REFERENCE_SECONDS,
                    log_occupancy,
                    jnp.asarray(_pad_with_last_row(log_background, n_padded)),
                    block_size=min(block_size, n_padded),
                    pos_tile_size=pos_tile_size,
                    indices_are_sorted=selection.indices_are_sorted,
                )
                continue

            # Compute position kernel in log-space to prevent underflow
            # (Skip if using streaming mode - computed on-the-fly)
            if use_streaming:
                log_position_distance = None
            else:
                log_position_distance = log_kde_distance(
                    interior_place_bin_centers,
                    electrode_encoding_positions,
                    std=position_std,
                )

            log_likelihood += sum_spikes_into_rows(
                block_estimate_log_joint_mark_intensity(
                    electrode_decoding_spike_waveform_features,
                    electrode_encoding_spike_waveform_features,
                    electrode_waveform_std,
                    occupancy,
                    electrode_mean_rate * RATE_REFERENCE_SECONDS,
                    log_position_distance,
                    block_size=block_size,
                    enc_tile_size=enc_tile_size,
                    pos_tile_size=pos_tile_size,
                    use_streaming=use_streaming,
                    encoding_positions=(
                        electrode_encoding_positions if use_streaming else None
                    ),
                    position_eval_points=(
                        interior_place_bin_centers if use_streaming else None
                    ),
                    position_std=position_std if use_streaming else None,
                    encoding_weights=electrode_encoding_weights,
                    log_occupancy=log_occupancy,
                    log_background=log_background,
                ),
                selection,
            )

    return (
        log_likelihood
        + log_bin_duration_evidence(
            spike_times,
            time_edges,
            row_slice,
            _spike_time_order,
            intensity_time_scale=RATE_REFERENCE_SECONDS,
            _time_grid=_time_grid,
        )[:, None]
    )


def compute_local_log_likelihood(
    time_edges: np.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: list[jnp.ndarray],
    environment: Environment,
    mean_rates: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    encoding_weights: list[jnp.ndarray] | None = None,
    row_slice: slice | None = None,
    *,
    mark_reference: list[MarkReference | None],
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Compute the log likelihood at the animal's position.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        List of spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        List of spike waveform features for each electrode.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        List of KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jnp.ndarray]
        List of spike waveform features for each electrode used for encoding.
    encoding_positions : list[jnp.ndarray]
        Per-electrode encoding positions, each of shape
        (n_encoding_spikes, n_position_dims).
    environment : Environment
        The spatial environment.
    mean_rates : jnp.ndarray
        Mean firing rate for each electrode.
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False
    encoding_weights : list[jnp.ndarray] | None, optional
        Per-electrode encoding-spike weights, each shape (n_encoding_spikes,). By
        default None (uniform); weight the local marginal density KDE per electrode.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` stay the FULL decoding edges (see
        ``predict_clusterless_kde_log_likelihood``).
    mark_reference : list of MarkReference or None
        Reference mark density per electrode for the background intensity (see
        ``predict_clusterless_kde_log_likelihood``).
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, 1)
    """
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_rows = row_stop - row_start

    # Normalize to a per-electrode list; None -> uniform weights for each electrode.
    if encoding_weights is None:
        encoding_weights = [None] * len(encoding_positions)

    interpolated_position, log_occupancy = _prepare_kde_local_inputs(
        occupancy_model,
        position_time,
        position,
        environment,
        _time_grid,
        row_start,
        row_stop,
        _spike_time_order,
    )

    log_likelihood = jnp.zeros((n_rows,))
    summed_expected_counts = jnp.zeros((n_rows,))
    positions = jnp.asarray(interpolated_position)
    n_position_dims = positions.shape[1]
    position_std_array = jnp.asarray(position_std)
    occupancy_std = as_std_array(occupancy_model.std, n_position_dims)
    for (
        electrode_encoding_spike_waveform_features,
        electrode_encoding_positions,
        electrode_encoding_weights,
        electrode_mean_rate,
        electrode_gpi_model,
        electrode_decoding_spike_waveform_features,
        electrode_spike_times,
        electrode_mark_reference,
    ) in zip(
        tqdm(
            encoding_spike_waveform_features,
            unit="electrode",
            desc="Local Likelihood",
            disable=disable_progress_bar,
        ),
        encoding_positions,
        encoding_weights,
        mean_rates,
        gpi_models,
        spike_waveform_features,
        spike_times,
        mark_reference,
        strict=True,
    ):
        selection = select_spikes_in_rows(
            electrode_spike_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        position_at_spike_time = get_position_at_time(
            position_time,
            position,
            select_spike_rows(electrode_spike_times, selection),
            environment,
        )
        electrode_decoding_spike_waveform_features = select_spike_rows(
            electrode_decoding_spike_waveform_features, selection
        )
        # Expand waveform_std to match this electrode's feature count if scalar
        n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]

        # Padded spikes and zero-weight samples let electrodes and chunks with
        # nearby sizes share one compiled kernel. The marginal density is
        # formed in log space; the spike contribution is
        # logaddexp(log(rate * density / occupancy), log background).
        n_padded = _padded_spike_count(position_at_spike_time.shape[0], block_size)
        padded_positions, padded_weights = _padded_samples(
            _spike_time_order,
            electrode_encoding_positions,
            electrode_encoding_weights,
        )
        padded_features, _ = _padded_samples(
            _spike_time_order,
            electrode_encoding_spike_waveform_features,
            electrode_encoding_weights,
        )
        gpi_samples, gpi_weights = _padded_samples(
            _spike_time_order,
            electrode_gpi_model.samples_,
            electrode_gpi_model.weights_,
        )
        occupancy_samples, occupancy_weights = _padded_samples(
            _spike_time_order,
            occupancy_model.samples_,
            occupancy_model.weights_,
            count=_padded_occupancy_count,
        )
        padded_spike_positions = jnp.asarray(
            _pad_rows(position_at_spike_time, n_padded)
        )
        log_likelihood, summed_expected_counts = _add_electrode_local_terms(
            log_likelihood,
            summed_expected_counts,
            padded_spike_positions,
            jnp.asarray(
                _pad_rows(electrode_decoding_spike_waveform_features, n_padded)
            ),
            jnp.asarray(spike_row_ids(selection, n_padded)),
            padded_positions,
            padded_features,
            padded_weights,
            position_std_array,
            as_std_array(waveform_std, n_waveform_features),
            # Evaluated outside the kernel, which then compiles no density
            # repair for it.
            _true_kde_log_density(
                padded_spike_positions,
                occupancy_samples,
                occupancy_std,
                occupancy_model.block_size or n_padded,
                occupancy_weights,
            ),
            gpi_samples,
            gpi_weights,
            as_std_array(electrode_gpi_model.std, n_position_dims),
            positions,
            log_occupancy,
            electrode_mean_rate * RATE_REFERENCE_SECONDS,
            electrode_mean_rate,
            jnp.asarray(
                _pad_rows(
                    background_log_intensity(
                        electrode_mark_reference,
                        np.asarray(electrode_decoding_spike_waveform_features),
                        LOG_EPS,  # log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS)
                    ),
                    n_padded,
                )
            ),
            block_size=min(block_size, n_padded),
            gpi_block_size=electrode_gpi_model.block_size or n_rows,
            indices_are_sorted=selection.indices_are_sorted,
            log_gpi_density=_prepared_kde_log_density(
                electrode_gpi_model, interpolated_position, _spike_time_order
            ),
        )

    # Every electrode, including zero-rate ones, adds its background rate, as in
    # fit_clusterless_kde_encoding_model's summed_ground_process_intensity.
    log_likelihood -= jnp.asarray(_time_grid.durations(row_start, row_stop)) * (
        summed_expected_counts + len(spike_times) * RATE_EPS_HZ
    )
    return log_likelihood[:, jnp.newaxis]
