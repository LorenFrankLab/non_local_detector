from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    KDEModel,
    _pad_rows,
    _padded_samples,
    _padded_spike_count,
    _SpikeTimeOrder,
    _traced_block_kde,
    _traced_block_log_kde,
    as_std_array,
    decode_bin_centers,
    deterministic_row_sum,
    drop_zero_weight_samples,
    get_position_at_time,
    interpolate_weights_at_spike_times,
    log_bin_duration_evidence,
    log_gaussian_pdf,
    resolve_row_slice,
    safe_log,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    sum_spikes_into_rows,
    validate_finite,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)

# Maximum waveform feature dimensions for the compensated-linear fast path.
# Above this threshold, mark kernel underflow causes accuracy degradation
# and the logsumexp path is used instead. Empirically validated: ≤8 dims
# gives <8e-6 max absolute error vs logsumexp across 10 random seeds.
#
# Caveat: the threshold bounds only the *mark* kernel's dynamic range. After
# the sqrt-scale split both stabilized factors can underflow independently, so
# the *position* kernel (many bins far from an encoding spike, small
# position_std, or long linearized tracks) also contributes to underflow. In
# float32 each factor floors near 1e-38, so joint terms far below the row max
# are dropped regardless of mark dimensionality. Note x64 is not enabled
# anywhere in this package, so float32 is the normal regime. The empirical
# validation above was run with realistic position kernels; if you rely on a
# much wider position dynamic range, prefer the logsumexp path.
_COMPENSATED_LINEAR_MAX_FEATURES = 8


def _log_joint_from_log_marginal(
    log_marginal: jnp.ndarray, mean_rate: float, occupancy: jnp.ndarray
) -> jnp.ndarray:
    """Combine a log marginal density with mean rate and occupancy.

    Computes ``log(mean_rate * marginal / occupancy)`` in log space and applies one
    degeneracy contract shared by the GEMM joint-intensity paths (compensated-linear /
    logsumexp / chunked / streaming) and by the local at-position likelihood: a bin
    with zero occupancy, a fully underflowed marginal (``-inf``, true zero mass), or a
    zero mean rate (a fully de-weighted electrode) collapses to ``LOG_EPS`` ("no
    support here"). A ``NaN`` marginal is deliberately *not* floored (``isneginf``
    catches only ``-inf``, and the zero-occupancy mask is ``~isnan``-gated) so a broken
    computation reaches ``core.py``'s NaN diagnostics instead of being laundered into a
    finite value. (The ``use_gemm=False`` reference path in
    ``estimate_log_joint_mark_intensity`` floors independently via ``safe_log`` and
    does not route through here.)

    A positive mean rate is folded in via ``safe_log`` (rates below ``EPS`` floor to
    ``LOG_EPS``, exactly as the previous ``safe_log(mean_rate)`` did); a rate of exactly
    0 uses ``-inf`` so the whole term floors to ``LOG_EPS``. A zero rate always
    coincides with a ``-inf`` marginal on the non-local paths (so this is unchanged
    there) and is what the local path needs to floor a de-weighted electrode.

    A supported bin with a legitimately tiny (finite, below-``LOG_EPS``) marginal is
    *not* floored here; the caller applies that floor —
    ``block_estimate_log_joint_mark_intensity``'s clamp for the non-local paths, an
    explicit ``jnp.maximum`` for the local path (which does not re-clamp).

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
    # A positive rate uses safe_log (rates below EPS floor to LOG_EPS, exactly as the
    # previous safe_log(mean_rate) did); a rate of exactly 0 forces -inf so the whole
    # term floors to LOG_EPS (a zero rate always coincides with a -inf marginal on the
    # non-local paths, so this is unchanged there, and it is what the local path needs).
    log_mean_rate = jnp.where(mean_rate > 0.0, safe_log(mean_rate, eps=EPS), -jnp.inf)
    log_occ = safe_log(occupancy, eps=EPS)
    log_joint = log_mean_rate + log_marginal - log_occ[None, :]
    # Floor degenerate bins -- zero occupancy (no support) or a true zero-mass marginal
    # (-inf, incl. a zero mean rate) -- to LOG_EPS, but exclude NaN (`& ~isnan`) so a
    # broken computation still reaches core.py's diagnostics instead of being masked.
    floor = ((occupancy[None, :] <= 0.0) | jnp.isneginf(log_joint)) & ~jnp.isnan(
        log_joint
    )
    return jnp.where(floor, LOG_EPS, log_joint)


def _zero_if_neginf(log_max: jnp.ndarray) -> jnp.ndarray:
    """Neutralize ``-inf`` maxima before a max-subtraction stabilization.

    The compensated-linear paths stabilize by subtracting a maximum (per kernel
    row, or the running/global offset) from log-space values. An *empty* row or
    prefix -- one carrying no mass at all, because every kernel entry or the
    weight is ``-inf`` -- has a ``-inf`` maximum, and ``-inf - (-inf)`` is
    ``NaN``. Substituting ``0.0`` for that maximum *before* the subtraction
    leaves the entries at ``-inf``, so ``exp(-inf - 0) == 0``: the empty row
    contributes nothing, which is the correct answer, and the derivative stays
    finite. A ``jnp.where`` applied *after* the ``NaN`` arithmetic would repair
    the value but still poison autodiff, because the untaken branch's ``NaN``
    propagates back through the reverse-mode sum.

    A ``-inf`` maximum means zero mass whatever produced it -- a zero encoding
    weight, a fully underflowed kernel row, or a padded row. ``NaN`` inputs are
    deliberately left visible (``isneginf`` does not match them), so a broken
    computation reaches ``core.py``'s NaN diagnostics instead of being treated
    as an empty row.

    Parameters
    ----------
    log_max : jnp.ndarray
        Log-space maximum: either a per-row array, shape ``(n_rows,)``, or a
        scalar running/global maximum.

    Returns
    -------
    safe_log_max : jnp.ndarray, same shape as ``log_max``
        ``log_max`` with every ``-inf`` entry replaced by ``0.0``.
    """
    return jnp.where(jnp.isneginf(log_max), 0.0, log_max)


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
    """Compute log mark kernel using GEMM (matrix multiplication) instead of per-dimension loop.

    This is mathematically equivalent to the loop-based approach but much faster for
    multi-dimensional features. The Gaussian kernel in log-space:

        log K(x, y) = -0.5 * sum_d [(x_d - y_d)^2 / sigma_d^2] - log_norm_const
                    = -0.5 * sum_d [(x_d/sigma_d)^2 + (y_d/sigma_d)^2 - 2*(x_d/sigma_d)*(y_d/sigma_d)] - log_norm_const
                    = -0.5 * (||x_scaled||^2 + ||y_scaled||^2 - 2 * x_scaled @ y_scaled^T) - log_norm_const

    The cross term x_scaled @ y_scaled^T is a single matrix multiply (GEMM).

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
    n_features = waveform_stds.shape[0]

    # Precompute inverse standard deviations and normalization constant
    # Clip to avoid division by zero for degenerate feature dimensions
    waveform_stds = jnp.clip(waveform_stds, min=EPS)
    inv_sigma = 1.0 / waveform_stds  # (n_features,)

    # Log normalization constant: -0.5 * (D * log(2π) + 2 * sum(log(sigma)))
    # Factor of 2 because we have sum of log(sigma), not log(sigma^2)
    log_norm_const = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )

    # Scale features by inverse standard deviations
    Y = encoding_features * inv_sigma[None, :]  # (n_enc, n_features)
    X = decoding_features * inv_sigma[None, :]  # (n_dec, n_features)

    # Compute squared norms
    y2 = jnp.sum(Y**2, axis=1)  # (n_enc,)
    x2 = jnp.sum(X**2, axis=1)  # (n_dec,)

    # GEMM: compute cross terms X @ Y^T = (n_dec, n_features) @ (n_features, n_enc)
    cross_term = jnp.matmul(
        X, Y.T, precision=jax.lax.Precision.HIGHEST
    )  # (n_dec, n_enc)

    # Combine: log K[i,j] = -0.5 * (y2[i] + x2[j] - 2*cross_term[j,i]) + log_norm_const
    # Note: We need (n_enc, n_dec) output, so transpose the cross term
    # Clamp squared distances to non-negative to avoid catastrophic cancellation
    # when x ≈ y (the expanded GEMM form can produce small negative values).
    sq_dist = jnp.maximum(y2[:, None] + x2[None, :] - 2.0 * cross_term.T, 0.0)
    logK_mark = log_norm_const - 0.5 * sq_dist  # (n_enc, n_dec)

    return logK_mark


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
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
    use_streaming: bool = False,
) -> jnp.ndarray:
    """Compute log joint mark intensity with encoding spike chunking.

    Uses online logsumexp to accumulate across encoding chunks, reducing
    peak memory from O(n_enc * n_pos) to O(enc_tile_size * n_pos).

    Supports two modes:
    1. Precomputed: Uses precomputed log_position_distance matrix
    2. Streaming: Computes position distances on-the-fly per chunk (saves memory)

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_dec, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_enc, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_pos,)
    mean_rate : float
    log_position_distance : jnp.ndarray | None, shape (n_enc, n_pos)
        Precomputed log position distances. Required if use_streaming=False.
    log_w : jnp.ndarray, shape (n_enc,)
        Per-encoding-spike log weight (uniform ``-log(n_enc)`` or ``log(w_e / sum_w)``).
        Sliced per chunk; a ``-inf`` entry drops that spike out of the chunk's logsumexp.
    enc_tile_size : int
        Number of encoding spikes to process in each chunk
    pos_tile_size : int | None
        If provided, also tile over positions
    encoding_positions : jnp.ndarray | None, shape (n_enc, n_pos_dims)
        Encoding positions. Required if use_streaming=True.
    position_eval_points : jnp.ndarray | None, shape (n_pos, n_pos_dims)
        Position evaluation points (e.g., interior_place_bin_centers). Required if use_streaming=True.
    position_std : jnp.ndarray | None, shape (n_pos_dims,)
        Position standard deviations. Required if use_streaming=True.
    use_streaming : bool, default=False
        If True, compute position distances on-the-fly. Reduces memory but adds computation.

    Returns
    -------
    log_joint : jnp.ndarray, shape (n_dec, n_pos)
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

    # Pad encoding arrays to be divisible by enc_tile_size (required for dynamic_slice)
    n_enc_chunks = (n_enc + enc_tile_size - 1) // enc_tile_size
    n_enc_padded = n_enc_chunks * enc_tile_size
    pad_enc = n_enc_padded - n_enc

    # Create validity mask for encoding spikes (used in streaming mode)
    if use_streaming:
        enc_valid_mask = jnp.arange(n_enc_padded) < n_enc  # Shape: (n_enc_padded,)

    if pad_enc > 0:
        # Pad waveform features with zeros
        encoding_spike_waveform_features = jnp.pad(
            encoding_spike_waveform_features,
            ((0, pad_enc), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        if use_streaming:
            # Streaming mode: pad encoding positions with zeros (will be masked with -inf later)
            encoding_positions = jnp.pad(
                encoding_positions,
                ((0, pad_enc), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        else:
            # Precomputed mode: pad log_position_distance with -inf
            log_position_distance = jnp.pad(
                log_position_distance,
                ((0, pad_enc), (0, 0)),
                mode="constant",
                constant_values=-jnp.inf,
            )
        # Pad the per-spike log weight to match; the padded rows already have -inf
        # log position distance, so the added weight is inert.
        log_w = jnp.pad(log_w, (0, pad_enc), mode="constant", constant_values=-jnp.inf)

    # Define vmapped function once (outside loop) for efficiency
    def compute_for_one_spike(
        log_pos_chunk: jnp.ndarray, y_col: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute logsumexp for one decoding spike across encoding chunk.

        Parameters
        ----------
        log_pos_chunk : jnp.ndarray, shape (enc_tile_size, n_pos)
            Already weighted: the per-spike log weight has been added per row.
        y_col : jnp.ndarray, shape (enc_tile_size,)

        Returns
        -------
        jnp.ndarray, shape (n_pos,)
        """
        return jax.nn.logsumexp(log_pos_chunk + y_col[:, None], axis=0)

    # Predefine vmapped function for position tiling
    def compute_for_one_spike_tile(
        log_pos_tile: jnp.ndarray, y_col: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute logsumexp for one decoding spike over position tile.

        Parameters
        ----------
        log_pos_tile : jnp.ndarray, shape (enc_tile_size, tile_size)
            Already weighted: the per-spike log weight has been added per row.
        y_col : jnp.ndarray, shape (enc_tile_size,)

        Returns
        -------
        jnp.ndarray, shape (tile_size,)
        """
        return jax.nn.logsumexp(log_pos_tile + y_col[:, None], axis=0)

    def process_enc_chunk(chunk_idx: int, log_marginal: jnp.ndarray) -> jnp.ndarray:
        """Process one encoding chunk and accumulate with online logsumexp."""
        enc_start = chunk_idx * enc_tile_size

        # Extract encoding chunk (always enc_tile_size, possibly padded)
        enc_chunk_features = jax.lax.dynamic_slice(
            encoding_spike_waveform_features,
            (enc_start, 0),
            (enc_tile_size, encoding_spike_waveform_features.shape[1]),
        )

        # Get position distance for this encoding chunk: (enc_tile_size, n_pos)
        if use_streaming:
            # Streaming mode: compute position kernel on-the-fly
            # Extract encoding positions for this chunk
            enc_chunk_positions = jax.lax.dynamic_slice(
                encoding_positions,
                (enc_start, 0),
                (enc_tile_size, encoding_positions.shape[1]),
            )

            # Compute position distances using streaming (avoids D×n_enc×n_pos intermediate)
            log_pos_chunk = log_kde_distance_streaming(
                position_eval_points,  # (n_pos, n_pos_dims)
                enc_chunk_positions,  # (enc_tile_size, n_pos_dims)
                position_std,  # (n_pos_dims,)
            )
            # Returns: (enc_tile_size, n_pos)

            # Mask padded entries (beyond n_enc) with -inf
            # Extract validity mask for this chunk
            chunk_valid_mask = jax.lax.dynamic_slice(
                enc_valid_mask,
                (enc_start,),
                (enc_tile_size,),
            )
            # Apply mask: invalid entries → -inf
            log_pos_chunk = jnp.where(
                chunk_valid_mask[:, None], log_pos_chunk, -jnp.inf
            )
        else:
            # Precomputed mode: slice from full matrix
            log_pos_chunk = jax.lax.dynamic_slice(
                log_position_distance,
                (enc_start, 0),
                (enc_tile_size, n_pos),
            )

        # Fold this chunk's per-spike log weight into the position kernel so the
        # logsumexp below is weighted. Added (not folded into the mark kernel) so a
        # zero-weight spike (-inf) simply exps to 0 instead of poisoning a row max.
        log_w_chunk = jax.lax.dynamic_slice(log_w, (enc_start,), (enc_tile_size,))
        log_pos_chunk = log_pos_chunk + log_w_chunk[:, None]

        # Compute log mark kernel for this encoding chunk: (enc_tile_size, n_dec)
        logK_mark_chunk = _compute_log_mark_kernel_gemm(
            decoding_spike_waveform_features,
            enc_chunk_features,
            waveform_stds,
        )

        # No need to mask logK_mark_chunk - padded entries already have -inf in log_pos_chunk

        if pos_tile_size is None or pos_tile_size >= n_pos:
            # No position tiling: process all positions at once
            # vmap over decoding spikes: (n_dec, n_pos)
            log_marginal_chunk = jax.vmap(compute_for_one_spike, in_axes=(None, 0))(
                log_pos_chunk, logK_mark_chunk.T
            )
        else:
            # Position tiling: use lax.fori_loop for JIT compilation
            # Pad positions to be divisible by pos_tile_size
            n_pos_chunks = (n_pos + pos_tile_size - 1) // pos_tile_size
            n_pos_padded = n_pos_chunks * pos_tile_size
            pad_pos = n_pos_padded - n_pos

            if pad_pos > 0:
                # Pad log_pos_chunk with -inf so they don't contribute
                log_pos_chunk = jnp.pad(
                    log_pos_chunk,
                    ((0, 0), (0, pad_pos)),
                    mode="constant",
                    constant_values=-jnp.inf,
                )

            def process_pos_tile(
                pos_chunk_idx: int, log_marginal_chunk: jnp.ndarray
            ) -> jnp.ndarray:
                """Process one position tile within encoding chunk."""
                pos_start = pos_chunk_idx * pos_tile_size

                log_pos_tile = jax.lax.dynamic_slice(
                    log_pos_chunk,
                    (0, pos_start),
                    (enc_tile_size, pos_tile_size),
                )

                # vmap over decoding spikes for this position tile -> (n_dec, pos_tile_size)
                log_marginal_tile = jax.vmap(
                    compute_for_one_spike_tile, in_axes=(None, 0)
                )(log_pos_tile, logK_mark_chunk.T)

                # Update output with this tile
                # fori_loop handles in-place updates efficiently when possible
                return jax.lax.dynamic_update_slice(
                    log_marginal_chunk, log_marginal_tile, (0, pos_start)
                )

            # Initialize chunk accumulator with -inf (logsumexp identity)
            log_marginal_chunk = jnp.full((n_dec, n_pos_padded), -jnp.inf)
            log_marginal_chunk = jax.lax.fori_loop(
                0, n_pos_chunks, process_pos_tile, log_marginal_chunk
            )

            # Trim back to original size
            log_marginal_chunk = log_marginal_chunk[:, :n_pos]

        # Online logsumexp: accumulate this chunk into running total
        return jnp.logaddexp(log_marginal, log_marginal_chunk)

    # Initialize accumulator with -inf (identity for logsumexp)
    log_marginal = jnp.full((n_dec, n_pos), -jnp.inf)

    # Use lax.fori_loop instead of Python for-loop for JIT compilation
    log_marginal = jax.lax.fori_loop(0, n_enc_chunks, process_enc_chunk, log_marginal)

    # Fully masked positions accumulate to -inf; the shared combiner floors
    # those (and zero-occupancy bins) to LOG_EPS.
    return _log_joint_from_log_marginal(log_marginal, mean_rate, occupancy)


def _compensated_linear_marginal(
    logK_mark: jnp.ndarray,
    log_position_distance: jnp.ndarray,
    log_w: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
) -> jnp.ndarray:
    """Compute log joint mark intensity using compensated-linear matmul.

    Computes ``logsumexp_e(log_w + logK_mark[e,d] + logK_pos[e,p])`` by
    stabilizing both kernel matrices via per-row max subtraction, absorbing
    a shared scale factor into each matrix, and reducing with a single BLAS
    matmul.  This avoids materializing the ``(n_enc, n_dec, n_pos)`` 3D
    tensor that logsumexp requires and is 15-50x faster on CPU.

    Numerically safe for waveform feature dimensions ≤ 8 (mark kernel
    underflow < ~13%).  For higher dimensions, use the logsumexp path.

    Parameters
    ----------
    logK_mark : jnp.ndarray, shape (n_enc, n_dec)
        Log mark (waveform) kernel matrix.
    log_position_distance : jnp.ndarray, shape (n_enc, n_pos)
        Log position kernel matrix.
    log_w : jnp.ndarray, shape (n_enc,)
        Per-encoding-spike log weight (``-log(n_enc)`` for each spike when uniform,
        ``log(w_e / sum_w)`` when weighted). A ``-inf`` entry (zero weight) makes that
        row's ``sqrt_scale`` zero, dropping the spike out of the matmul.
    occupancy : jnp.ndarray, shape (n_pos,)
        Occupancy density at position bins.
    mean_rate : float
        Mean firing rate for this electrode.

    Returns
    -------
    log_joint : jnp.ndarray, shape (n_dec, n_pos)
        Log joint mark intensity.
    """
    # Per-encoding-spike row maxima for numerical stabilization
    max_pos = jnp.max(log_position_distance, axis=1)  # (n_enc,)
    max_wf = jnp.max(logK_mark, axis=1)  # (n_enc,)

    # Global offset for the entire sum
    total_max_per_enc = max_pos + max_wf + log_w  # (n_enc,)
    global_max = jnp.max(total_max_per_enc)

    # Stable per-row scale: all values in (-inf, 0]
    # An electrode with no mass at all has global_max == -inf; see _zero_if_neginf.
    safe_global_max = _zero_if_neginf(global_max)
    log_scale = total_max_per_enc - safe_global_max  # (n_enc,)

    # Stabilized kernels: all values in [0, 1]
    # Empty kernel rows have -inf maxima; see _zero_if_neginf.
    safe_max_pos = _zero_if_neginf(max_pos)
    safe_max_wf = _zero_if_neginf(max_wf)
    K_pos_stable = jnp.exp(log_position_distance - safe_max_pos[:, None])
    K_wf_stable = jnp.exp(logK_mark - safe_max_wf[:, None])

    # Absorb sqrt(scale) into each factor so the matmul carries the weight.
    # Identity: sum_e scale[e] * K_wf[e,d] * K_pos[e,p]
    #         = sum_e sqrt_scale[e]^2 * K_wf[e,d] * K_pos[e,p]
    #         = (W.T @ P)[d,p]   where W[e,d] = K_wf[e,d]*sqrt_scale[e]
    sqrt_scale = jnp.exp(0.5 * log_scale)  # (n_enc,)
    W = K_wf_stable * sqrt_scale[:, None]  # (n_enc, n_dec)
    P = K_pos_stable * sqrt_scale[:, None]  # (n_enc, n_pos)

    # Single BLAS matmul: (n_dec, n_enc) @ (n_enc, n_pos) -> (n_dec, n_pos)
    marginal_scaled = jnp.matmul(W.T, P, precision=jax.lax.Precision.HIGHEST)

    # Back to log space.  Use double-where (safe_marginal avoids log(0) in the
    # untaken branch) and encode a zero matmul result as -inf so the shared
    # combiner floors it to LOG_EPS, identically to the logsumexp path. A NaN
    # marginal (from non-finite inputs) is propagated, not turned into -inf:
    # `NaN > 0.0` is False, so without the isnan branch a broken computation
    # would be silently laundered to LOG_EPS instead of surfacing at core.py's
    # NaN diagnostics (the module's contract, matched by the logsumexp path).
    safe_marginal = jnp.where(marginal_scaled > 0.0, marginal_scaled, 1.0)
    log_marginal = jnp.where(
        marginal_scaled > 0.0,
        jnp.log(safe_marginal) + global_max,
        jnp.where(jnp.isnan(marginal_scaled), jnp.nan, -jnp.inf),
    )
    return _log_joint_from_log_marginal(log_marginal, mean_rate, occupancy)


def _compensated_linear_marginal_chunked(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None,
    log_w: jnp.ndarray,
    enc_tile_size: int,
    use_streaming: bool = False,
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Chunked compensated-linear marginal with online max tracking.

    Single-pass algorithm that accumulates matmul results across encoding
    chunks while maintaining numerical stability via online max rescaling.
    Analogous to online logsumexp but uses BLAS matmul for the reduction.

    Here "compensated" refers to the running-max compensation that keeps the
    accumulator in a stable range (as in ``_compensated_linear_marginal``); it
    is *not* Kahan/compensated summation. The cross-chunk accumulation is a
    plain ``running_sum + W.T @ P``, so with a very large number of chunks in
    float32 the low-order bits of early chunks can be lost. This is acceptable
    for KDE densities at the tolerances used here; add a compensation term to
    the scan carry if chunk counts ever grow large enough to matter.

    Memory: O(enc_tile_size × max(n_dec, n_pos)) per chunk — independent
    of total n_enc.  Supports both precomputed and streaming position kernels.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_dec, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_enc, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_pos,)
    mean_rate : float
    log_position_distance : jnp.ndarray | None, shape (n_enc, n_pos)
        Precomputed log position distances. None if use_streaming=True.
    log_w : jnp.ndarray, shape (n_enc,)
        Per-encoding-spike log weight (uniform ``-log(n_enc)`` or ``log(w_e / sum_w)``).
        Sliced per chunk; a ``-inf`` entry drops that spike out of the chunk's matmul.
    enc_tile_size : int
        Number of encoding spikes per chunk.
    use_streaming : bool
        If True, compute position kernel on-the-fly per chunk.
    encoding_positions : jnp.ndarray | None, shape (n_enc, n_pos_dims)
        Required if use_streaming=True.
    position_eval_points : jnp.ndarray | None, shape (n_pos, n_pos_dims)
        Required if use_streaming=True.
    position_std : jnp.ndarray | None, shape (n_pos_dims,)
        Required if use_streaming=True.

    Returns
    -------
    log_joint : jnp.ndarray, shape (n_dec, n_pos)
    """
    n_enc = encoding_spike_waveform_features.shape[0]
    n_dec = decoding_spike_waveform_features.shape[0]

    if use_streaming:
        n_pos = position_eval_points.shape[0]
    else:
        n_pos = log_position_distance.shape[1]

    # Pad encoding arrays to be divisible by enc_tile_size
    n_chunks = (n_enc + enc_tile_size - 1) // enc_tile_size
    n_enc_padded = n_chunks * enc_tile_size
    pad_enc = n_enc_padded - n_enc

    if pad_enc > 0:
        encoding_spike_waveform_features = jnp.pad(
            encoding_spike_waveform_features,
            ((0, pad_enc), (0, 0)),
            constant_values=0.0,
        )
        if use_streaming:
            encoding_positions = jnp.pad(
                encoding_positions,
                ((0, pad_enc), (0, 0)),
                constant_values=0.0,
            )
        else:
            log_position_distance = jnp.pad(
                log_position_distance,
                ((0, pad_enc), (0, 0)),
                constant_values=-jnp.inf,
            )
        # Padded rows have zero mass through their -inf log weight.
        log_w = jnp.pad(log_w, (0, pad_enc), constant_values=-jnp.inf)

    # Validity mask for padded entries
    enc_valid = jnp.arange(n_enc_padded) < n_enc  # (n_enc_padded,)

    def process_chunk(carry, chunk_idx):
        """Process one encoding chunk with online max rescaling."""
        running_sum, running_max = carry
        enc_start = chunk_idx * enc_tile_size
        log_w_chunk = jax.lax.dynamic_slice(log_w, (enc_start,), (enc_tile_size,))

        # Extract encoding features for this chunk
        enc_chunk = jax.lax.dynamic_slice(
            encoding_spike_waveform_features,
            (enc_start, 0),
            (enc_tile_size, encoding_spike_waveform_features.shape[1]),
        )

        # Get position kernel for this chunk
        if use_streaming:
            enc_pos_chunk = jax.lax.dynamic_slice(
                encoding_positions,
                (enc_start, 0),
                (enc_tile_size, encoding_positions.shape[1]),
            )
            logK_pos_chunk = log_kde_distance_streaming(
                position_eval_points,
                enc_pos_chunk,
                position_std,
            )
        else:
            logK_pos_chunk = jax.lax.dynamic_slice(
                log_position_distance,
                (enc_start, 0),
                (enc_tile_size, n_pos),
            )

        # Compute mark kernel for this chunk: (enc_tile, n_dec)
        logK_mark_chunk = _compute_log_mark_kernel_gemm(
            decoding_spike_waveform_features,
            enc_chunk,
            waveform_stds,
        )

        # Mask padded entries
        chunk_valid = jax.lax.dynamic_slice(enc_valid, (enc_start,), (enc_tile_size,))
        logK_pos_chunk = jnp.where(chunk_valid[:, None], logK_pos_chunk, -jnp.inf)
        logK_mark_chunk = jnp.where(chunk_valid[:, None], logK_mark_chunk, -jnp.inf)

        # Preserve empty rows' -inf total, but use safe maxima in the kernel
        # subtractions below. This covers both padding and real zero-mass rows.
        max_pos = jnp.max(logK_pos_chunk, axis=1)  # (tile,)
        max_wf = jnp.max(logK_mark_chunk, axis=1)  # (tile,)
        chunk_total = jnp.where(chunk_valid, max_pos + max_wf + log_w_chunk, -jnp.inf)
        chunk_max = jnp.max(chunk_total)

        # Online max update: rescale running_sum if new max is larger.
        # If chunk_max <= running_max, exp(running_max - new_max) = 1 (no-op).
        new_max = jnp.maximum(running_max, chunk_max)
        # An empty prefix has new_max == -inf; safe_new_max is reused in BOTH
        # subtractions below (here and in log_scale). See _zero_if_neginf.
        safe_new_max = _zero_if_neginf(new_max)
        running_sum = running_sum * jnp.exp(running_max - safe_new_max)

        # Stabilize this chunk's kernels.  Invalid rows get 0 because
        # logK - 0 is still -inf, and exp(-inf) = 0.  See _zero_if_neginf.
        safe_max_pos = _zero_if_neginf(max_pos)
        safe_max_wf = _zero_if_neginf(max_wf)
        K_pos_stable = jnp.exp(logK_pos_chunk - safe_max_pos[:, None])
        K_wf_stable = jnp.exp(logK_mark_chunk - safe_max_wf[:, None])

        # Scale factors relative to current global max
        log_scale = chunk_total - safe_new_max
        sqrt_scale = jnp.exp(0.5 * log_scale)
        W = K_wf_stable * sqrt_scale[:, None]  # (tile, n_dec)
        P = K_pos_stable * sqrt_scale[:, None]  # (tile, n_pos)

        # Accumulate: matmul adds to running sum
        running_sum = running_sum + jnp.matmul(
            W.T, P, precision=jax.lax.Precision.HIGHEST
        )  # (n_dec, n_pos)

        return (running_sum, new_max), None

    # Initialize: zero sum, -inf max
    init_sum = jnp.zeros((n_dec, n_pos))
    init_max = jnp.array(-jnp.inf)

    (final_sum, final_max), _ = jax.lax.scan(
        process_chunk,
        (init_sum, init_max),
        jnp.arange(n_chunks),
    )

    # Back to log space (double-where; zero mass -> -inf so the shared combiner
    # floors it to LOG_EPS, matching the logsumexp path). A NaN sum (from
    # non-finite inputs) is propagated rather than laundered to -inf/LOG_EPS,
    # since `NaN > 0.0` is False (see _compensated_linear_marginal).
    safe_sum = jnp.where(final_sum > 0.0, final_sum, 1.0)
    log_marginal = jnp.where(
        final_sum > 0.0,
        jnp.log(safe_sum) + final_max,
        jnp.where(jnp.isnan(final_sum), jnp.nan, -jnp.inf),
    )
    return _log_joint_from_log_marginal(log_marginal, mean_rate, occupancy)


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
        If True (default), use GEMM-based log-space computation (faster for multi-dimensional features).
        If False, use linear-space computation (matches reference exactly).
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension in chunks (only for use_gemm=True).
    enc_tile_size : int | None, optional
        If provided, tile computation over encoding spikes dimension to reduce memory.
        Uses online logsumexp to accumulate across encoding chunks. Reduces peak memory
        from O(n_enc * n_pos) to O(enc_tile_size * n_pos). Only for use_gemm=True.
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

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    Notes
    -----
    The module-level ``estimate_log_joint_mark_intensity`` name is rebound just
    below its definition to a jitted version with
    ``static_argnames=('use_gemm', 'pos_tile_size', 'enc_tile_size',
    'use_streaming')`` — so callers get JIT automatically and do not need to
    wrap it themselves. The undecorated function object remains accessible via
    ``estimate_log_joint_mark_intensity.__wrapped__`` if an un-jitted version is
    needed.
    """
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]

    # Reject unsupported combinations before any array work so the failure is a
    # clear message rather than a downstream exp(None) TypeError.
    if use_streaming and not use_gemm:
        raise ValueError(
            "use_streaming=True is not supported with use_gemm=False "
            "(the linear-space reference path). Use use_gemm=True for streaming."
        )

    if not use_gemm:
        # Linear-space computation (matches reference exactly)
        # Convert log position back to linear for matrix multiply
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
            # per-spike weight, normalized by the total weight (matches the linear
            # clusterless_kde path; uniform weights recover the /n form above).
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
        # Use safe_log to avoid -inf from zero marginal_density or mean_rate
        return safe_log(
            mean_rate
            * jnp.where(
                occupancy > 0.0,
                marginal_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                EPS,
            ),
            eps=EPS,
        )

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

    # Per-encoding-spike log weight, added (not folded into the mark kernel) so the
    # kernel row-maxima used for stabilization stay finite even when a spike has zero
    # weight (log_w = -inf); a -inf term then cleanly zeroes that spike's contribution
    # (sqrt_scale = exp(-inf) = 0, or it drops out of the logsumexp) instead of
    # producing NaN from -inf - (-inf) in the max subtraction.
    if encoding_weights is None:
        # Uniform weights: log(1/n) for each encoding spike.
        # Use max(n, 1) to avoid log(0); when n=0 the result is unused.
        safe_n = jnp.where(n_encoding_spikes > 0, float(n_encoding_spikes), 1.0)
        log_w = jnp.full((n_encoding_spikes,), -jnp.log(safe_n))
    else:
        # log(w_e / sum_w); a zero weight -> -inf drops the spike out of the sum.
        weight_total = jnp.sum(encoding_weights)
        safe_weight_total = jnp.where(weight_total > 0, weight_total, 1.0)
        log_w = jnp.log(encoding_weights) - jnp.log(safe_weight_total)

    # If enc_tile_size specified, chunk over encoding spikes
    if enc_tile_size is not None and enc_tile_size < n_encoding_spikes:
        n_features = waveform_stds.shape[0]
        if n_features <= _COMPENSATED_LINEAR_MAX_FEATURES:
            # Chunked compensated-linear: matmul speed with bounded memory.
            # Uses online max tracking to accumulate across chunks.
            return _compensated_linear_marginal_chunked(
                decoding_spike_waveform_features,
                encoding_spike_waveform_features,
                waveform_stds,
                occupancy,
                mean_rate,
                log_position_distance,
                log_w,
                enc_tile_size,
                use_streaming=use_streaming,
                encoding_positions=encoding_positions,
                position_eval_points=position_eval_points,
                position_std=position_std,
            )
        # >8D features: fall back to logsumexp tiling
        return _estimate_with_enc_chunking(
            decoding_spike_waveform_features,
            encoding_spike_waveform_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            log_w,
            enc_tile_size,
            pos_tile_size,
            encoding_positions=encoding_positions,
            position_eval_points=position_eval_points,
            position_std=position_std,
            use_streaming=use_streaming,
        )

    # No encoding chunking: compute full logK_mark matrix
    # Build log-kernel matrix for marks: (n_enc, n_dec)
    logK_mark = _compute_log_mark_kernel_gemm(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
    )

    # Fast path: compensated-linear matmul for low-dimensional features.
    # Uses a single BLAS matmul instead of logsumexp, giving 15-50x speedup.
    # Safe for ≤ _COMPENSATED_LINEAR_MAX_FEATURES waveform dimensions.
    # Guard: n_encoding_spikes > 0 to avoid NaN from jnp.max on empty arrays.
    # Note: n_features is a static shape known at JAX trace time, so this
    # branch is resolved at compilation — JAX compiles the taken path only.
    n_features = waveform_stds.shape[0]
    if (
        n_features <= _COMPENSATED_LINEAR_MAX_FEATURES
        and not use_streaming
        and log_position_distance is not None
        and n_encoding_spikes > 0
    ):
        return _compensated_linear_marginal(
            logK_mark, log_position_distance, log_w, occupancy, mean_rate
        )

    # Define vmapped function once for efficiency (avoids closure creation per iteration)
    def compute_for_one_spike_full(y_col: jnp.ndarray) -> jnp.ndarray:
        """Compute logsumexp for one decoding spike across all positions.

        Parameters
        ----------
        y_col : jnp.ndarray, shape (n_enc,)
            Column of logK_mark for one decoding spike

        Returns
        -------
        jnp.ndarray, shape (n_pos,)
            Log-space marginal for this spike across all positions
        """
        return jax.nn.logsumexp(
            log_w[:, None] + log_position_distance + y_col[:, None], axis=0
        )

    def compute_for_one_spike_tile(
        log_pos_tile: jnp.ndarray, y_col: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute logsumexp for one decoding spike over position tile.

        Parameters
        ----------
        log_pos_tile : jnp.ndarray, shape (n_enc, tile_size)
        y_col : jnp.ndarray, shape (n_enc,)

        Returns
        -------
        jnp.ndarray, shape (tile_size,)
        """
        return jax.nn.logsumexp(log_w[:, None] + log_pos_tile + y_col[:, None], axis=0)

    # Use vmap for full parallelization over decoding spikes
    if pos_tile_size is None or pos_tile_size >= n_pos:
        # No tiling: process all positions at once (default, fastest)
        # vmap over decoding spikes' columns -> (n_dec, n_pos)
        log_marginal = jax.vmap(compute_for_one_spike_full)(logK_mark.T)
    else:
        # Tiled: process positions in chunks to reduce peak memory
        # Use lax.fori_loop instead of Python for-loop for JIT compilation
        n_pos_chunks = (n_pos + pos_tile_size - 1) // pos_tile_size
        n_pos_padded = n_pos_chunks * pos_tile_size
        pad_pos = n_pos_padded - n_pos

        # Pad log_position_distance if needed
        if pad_pos > 0:
            log_position_distance = jnp.pad(
                log_position_distance,
                ((0, 0), (0, pad_pos)),
                mode="constant",
                constant_values=-jnp.inf,
            )

        def process_pos_tile(
            pos_chunk_idx: int, log_marginal: jnp.ndarray
        ) -> jnp.ndarray:
            """Process one position tile."""
            pos_start = pos_chunk_idx * pos_tile_size

            # Tile: slice of log_position_distance for this chunk of positions
            log_pos_tile = jax.lax.dynamic_slice(
                log_position_distance,
                (0, pos_start),
                (n_encoding_spikes, pos_tile_size),
            )

            # vmap over decoding spikes for this position tile -> (n_dec, pos_tile_size)
            log_marginal_tile = jax.vmap(compute_for_one_spike_tile, in_axes=(None, 0))(
                log_pos_tile, logK_mark.T
            )

            # Update output with this tile
            return jax.lax.dynamic_update_slice(
                log_marginal, log_marginal_tile, (0, pos_start)
            )

        # Initialize with -inf (logsumexp identity, consistent with encoding chunking)
        log_marginal = jnp.full((n_dec, n_pos_padded), -jnp.inf)
        log_marginal = jax.lax.fori_loop(
            0, n_pos_chunks, process_pos_tile, log_marginal
        )

        # Trim back to original size
        log_marginal = log_marginal[:, :n_pos]

    # Result: log(mean_rate * marginal / occupancy). Fully masked positions are
    # -inf; the shared combiner floors those and zero-occupancy bins to LOG_EPS.
    return _log_joint_from_log_marginal(log_marginal, mean_rate, occupancy)


# JIT-compile with static arguments for performance
# This allows JAX to specialize the function for different tile sizes and modes
estimate_log_joint_mark_intensity = jax.jit(
    estimate_log_joint_mark_intensity,
    static_argnames=("use_gemm", "pos_tile_size", "enc_tile_size", "use_streaming"),
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
    occupancy: jnp.ndarray,
    mean_rate: float,
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
        A multiple of ``block_size`` rows. Padding repeats the last spike, so
        the stabilizing maxima over each block's spikes are unchanged.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    encoding_features : jnp.ndarray, shape (n_padded_samples, n_features)
    encoding_positions : jnp.ndarray, shape (n_padded_samples, n_position_dims)
    encoding_weights : jnp.ndarray, shape (n_padded_samples,)
        Zero for padding samples, which then drop out of every reduction.
    waveform_stds : jnp.ndarray, shape (n_features,)
    position_std : jnp.ndarray, shape (n_position_dims,)
    place_bin_centers : jnp.ndarray, shape (n_bins, n_position_dims)
    occupancy : jnp.ndarray, shape (n_bins,)
    mean_rate : float
        Mean rate times ``RATE_REFERENCE_SECONDS``.

    Returns
    -------
    total : jnp.ndarray, shape (n_rows, n_bins)
    """
    log_position_distance = log_kde_distance(
        place_bin_centers, encoding_positions, std=position_std
    )
    n_blocks = decoding_features.shape[0] // block_size
    intensities = jax.lax.map(
        lambda block: estimate_log_joint_mark_intensity(
            block,
            encoding_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            pos_tile_size=pos_tile_size,
            encoding_weights=encoding_weights,
        ),
        decoding_features.reshape(n_blocks, block_size, -1),
    ).reshape(-1, occupancy.shape[0])
    # Floor at LOG_EPS as block_estimate_log_joint_mark_intensity does.
    return total + deterministic_row_sum(
        jnp.clip(intensities, min=LOG_EPS, max=None),
        row_ids,
        total.shape[0],
        indices_are_sorted=indices_are_sorted,
    )


@partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "occupancy_block_size",
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
    occupancy_samples: jnp.ndarray,
    occupancy_weights: jnp.ndarray,
    occupancy_std: jnp.ndarray,
    gpi_samples: jnp.ndarray,
    gpi_weights: jnp.ndarray,
    gpi_std: jnp.ndarray,
    positions: jnp.ndarray,
    occupancy: jnp.ndarray,
    scaled_rate: float,
    mean_rate: float,
    *,
    block_size: int,
    occupancy_block_size: int,
    gpi_block_size: int,
    indices_are_sorted: bool,
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
    occupancy_samples, occupancy_weights, occupancy_std
        Fitted occupancy KDE leaves.
    gpi_samples, gpi_weights, gpi_std
        This electrode's padded ground-process KDE leaves.
    positions : jnp.ndarray, shape (n_rows, n_position_dims)
        Animal position at each row.
    occupancy : jnp.ndarray, shape (n_rows,)
        Occupancy density at ``positions``.
    scaled_rate : float
        Mean rate times ``RATE_REFERENCE_SECONDS``.
    mean_rate : float
        Mean rate in Hz.

    Returns
    -------
    log_likelihood, expected_counts : jnp.ndarray, shape (n_rows,)
    """
    log_marginal_density = _traced_block_log_kde(
        jnp.concatenate((spike_positions, decoding_features), axis=1),
        jnp.concatenate((encoding_positions, encoding_features), axis=1),
        jnp.concatenate((position_std, waveform_stds)),
        block_size,
        encoding_weights,
    )
    occupancy_at_spikes = _traced_block_kde(
        spike_positions,
        occupancy_samples,
        occupancy_std,
        occupancy_block_size,
        occupancy_weights,
    )
    # See compute_local_log_likelihood for the per-spike combine and floor.
    spike_contribution = jnp.maximum(
        _log_joint_from_log_marginal(
            log_marginal_density[None, :], scaled_rate, occupancy_at_spikes
        )[0],
        LOG_EPS,
    )
    log_likelihood = log_likelihood + deterministic_row_sum(
        spike_contribution,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )
    gpi_density = _traced_block_kde(
        positions, gpi_samples, gpi_std, gpi_block_size, gpi_weights
    )
    expected_counts = expected_counts + mean_rate * jnp.where(
        occupancy > 0.0,
        gpi_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
        0.0,
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
        If True (default), use GEMM-based log-space computation.
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

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    if n_decoding_spikes == 0:
        return jnp.full((0, n_position_bins), LOG_EPS)

    # Process decoding spikes in fixed-size blocks. Each decoding spike is
    # computed independently, so the final partial block is padded up to
    # block_size (and its extra rows sliced off) to keep every call to the
    # jitted estimator at the same shape — avoiding a recompile for the
    # leftover block. Results are collected and concatenated once instead of
    # O(n_blocks) full-array dynamic_update_slice copies.
    #
    # Pad by repeating the last real row (mode="edge"), not with zeros: the
    # compensated-linear path takes a max over the decoding axis, so a zero row
    # could perturb that max (and thus the shared stabilization) for real rows.
    # A duplicated real row is already accounted for in the max, so padding is
    # provably inert and the kept rows are bit-identical to the unpadded result.
    block_results = []
    for start_ind in range(0, n_decoding_spikes, block_size):
        block = decoding_spike_waveform_features[start_ind : start_ind + block_size]
        actual_len = block.shape[0]
        if actual_len < block_size:
            block = jnp.pad(block, ((0, block_size - actual_len), (0, 0)), mode="edge")
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
        )
        block_results.append(block_result[:actual_len])

    # Floor the assembled result at LOG_EPS. _log_joint_from_log_marginal
    # already reconciles the *degenerate* (-inf / zero-occupancy) bins across
    # paths, so this clip no longer has to do that -- but it is still
    # load-bearing: a supported bin with a legitimately tiny marginal comes back
    # below LOG_EPS and is floored here. (NaN, if any, is preserved by clip and
    # surfaces downstream.)
    return jnp.clip(jnp.concatenate(block_results, axis=0), min=LOG_EPS, max=None)


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
    position_std = as_std_array(position_std, n_std_dims)
    # Keep waveform_std as-is (scalar or array) - will be expanded per-electrode at predict time

    # Validate bandwidths once, host-side, at fit time. A zero/negative std is a
    # config error; catching it here (a clear ValueError, cannot break JIT) is
    # better than the near-delta kernel the downstream EPS clamp would silently
    # produce.
    if not np.all(np.asarray(position_std) > 0.0):
        raise ValueError(f"position_std must be positive, got {position_std}")
    if not np.all(np.asarray(waveform_std) > 0.0):
        raise ValueError(f"waveform_std must be positive, got {waveform_std}")

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    occupancy_samples, occupancy_weights = drop_zero_weight_samples(
        position, exposure_weights
    )
    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        occupancy_samples = get_linearized_position(
            occupancy_samples,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
        occupancy_samples, weights=jnp.asarray(occupancy_weights)
    )

    occupancy = occupancy_model.predict(interior_place_bin_centers)
    encoding_positions = []
    encoding_weights = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)
    bounded_spike_waveform_features = []

    for electrode_spike_waveform_features, electrode_spike_times in zip(
        tqdm(
            spike_waveform_features,
            desc="Encoding models",
            unit="electrode",
            disable=disable_progress_bar,
        ),
        spike_times,
        strict=True,
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

        gpi_density = gpi_model.predict(interior_place_bin_centers)
        summed_ground_process_intensity += mean_rates[-1] * jnp.where(
            occupancy > 0.0,
            gpi_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
            EPS,
        )

    # Clip the summed intensity once (not per electrode) so an empty bin gets a
    # single EPS floor rather than accumulating n_electrodes * EPS.
    summed_ground_process_intensity = jnp.clip(
        summed_ground_process_intensity, min=RATE_EPS_HZ, max=None
    )

    return {
        "occupancy": occupancy,
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
        If provided, tile computation over encoding spikes dimension to reduce memory.
        Uses online logsumexp accumulation. Reduces peak memory from O(n_enc * n_pos)
        to O(enc_tile_size * n_pos). By default None (no tiling).
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
            _spike_time_order=_spike_time_order,
            _time_grid=_time_grid,
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

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
                    occupancy,
                    electrode_mean_rate * RATE_REFERENCE_SECONDS,
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

    # Need to interpolate position at the requested rows only
    interpolated_position = get_position_at_time(
        position_time,
        position,
        decode_bin_centers(time_edges, row_start, row_stop),
        environment,
    )
    occupancy = occupancy_model.predict(interpolated_position)

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
        # log(rate * density / occupancy) through the shared combiner, so the
        # local path keeps the non-local degeneracy contract (rate, occupancy
        # or zero mass -> LOG_EPS), then floors tiny finite values to LOG_EPS
        # as the linear path's per-spike safe_log does.
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
        log_likelihood, summed_expected_counts = _add_electrode_local_terms(
            log_likelihood,
            summed_expected_counts,
            jnp.asarray(_pad_rows(position_at_spike_time, n_padded)),
            jnp.asarray(
                _pad_rows(electrode_decoding_spike_waveform_features, n_padded)
            ),
            jnp.asarray(spike_row_ids(selection, n_padded)),
            padded_positions,
            padded_features,
            padded_weights,
            position_std_array,
            as_std_array(waveform_std, n_waveform_features),
            occupancy_model.samples_,
            occupancy_model.weights_,
            occupancy_std,
            gpi_samples,
            gpi_weights,
            as_std_array(electrode_gpi_model.std, n_position_dims),
            positions,
            occupancy,
            electrode_mean_rate * RATE_REFERENCE_SECONDS,
            electrode_mean_rate,
            block_size=min(block_size, n_padded),
            occupancy_block_size=occupancy_model.block_size or n_padded,
            gpi_block_size=electrode_gpi_model.block_size or n_rows,
            indices_are_sorted=selection.indices_are_sorted,
        )

    # Subtract the summed ground-process intensity once, floored at EPS to
    # mirror fit_clusterless_kde_encoding_model's summed_ground_process_intensity
    # (a single EPS floor, not n_electrodes * EPS).
    log_likelihood -= jnp.asarray(_time_grid.durations(row_start, row_stop)) * jnp.clip(
        summed_expected_counts, min=RATE_EPS_HZ
    )
    return log_likelihood[:, jnp.newaxis]
