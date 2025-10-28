import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    KDEModel,
    block_kde,
    gaussian_pdf,
    get_position_at_time,
    get_spike_time_bin_ind,
    log_gaussian_pdf,
    safe_log,
)


@jax.jit
def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized KDE distance using vmap (optimized version).

    Computes the product of per-dimension Gaussian kernels using vectorization
    instead of Python for-loop, enabling full parallelization.

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
    This function assumes inputs are valid (same dimensionality, positive std).
    No validation is performed here to maintain JIT compatibility.
    """

    def gaussian_per_dim(eval_dim, sample_dim, sigma):
        return gaussian_pdf(
            eval_dim[None, :],  # shape (1, n_eval)
            sample_dim[:, None],  # shape (n_samples, 1)
            sigma,
        )

    # vmap over dimensions: produces (n_dims, n_samples, n_eval)
    per_dim_distances = jax.vmap(gaussian_per_dim)(eval_points.T, samples.T, std)

    # Product over dimensions: (n_samples, n_eval)
    return jnp.prod(per_dim_distances, axis=0)


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
    This function assumes inputs are valid (same dimensionality, positive std).
    No validation is performed here to maintain JIT compatibility.
    """

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
    cross_term = X @ Y.T  # (n_dec, n_enc)

    # Combine: log K[i,j] = -0.5 * (y2[i] + x2[j] - 2*cross_term[j,i]) + log_norm_const
    # Note: We need (n_enc, n_dec) output, so transpose the cross term
    logK_mark = log_norm_const - 0.5 * (
        y2[:, None] + x2[None, :] - 2.0 * cross_term.T
    )  # (n_enc, n_dec)

    return logK_mark


def _estimate_with_enc_chunking(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None,
    log_w: float,
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
    log_w : float
        log(1/n_enc) weight for each encoding spike
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

    # Define vmapped function once (outside loop) for efficiency
    def compute_for_one_spike(
        log_pos_chunk: jnp.ndarray, y_col: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute logsumexp for one decoding spike across encoding chunk.

        Parameters
        ----------
        log_pos_chunk : jnp.ndarray, shape (enc_tile_size, n_pos)
        y_col : jnp.ndarray, shape (enc_tile_size,)

        Returns
        -------
        jnp.ndarray, shape (n_pos,)
        """
        return jax.nn.logsumexp(log_w + log_pos_chunk + y_col[:, None], axis=0)

    # Predefine vmapped function for position tiling
    def compute_for_one_spike_tile(
        log_pos_tile: jnp.ndarray, y_col: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute logsumexp for one decoding spike over position tile.

        Parameters
        ----------
        log_pos_tile : jnp.ndarray, shape (enc_tile_size, tile_size)
        y_col : jnp.ndarray, shape (enc_tile_size,)

        Returns
        -------
        jnp.ndarray, shape (tile_size,)
        """
        return jax.nn.logsumexp(log_w + log_pos_tile + y_col[:, None], axis=0)

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

    # Add mean rate and subtract occupancy (in log)
    log_mean_rate = safe_log(mean_rate, eps=EPS)
    log_occ = safe_log(occupancy, eps=EPS)

    log_joint = jnp.where(
        occupancy[None, :] > 0.0,
        log_mean_rate + log_marginal - log_occ[None, :],
        jnp.log(0.0),  # -inf for zero occupancy (intentional masking)
    )

    return log_joint


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

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    Notes
    -----
    This function is JIT-compiled automatically when called from higher-level functions.
    For manual JIT compilation with custom settings, use:

        jitted_fn = jax.jit(
            estimate_log_joint_mark_intensity,
            static_argnames=('use_gemm', 'pos_tile_size', 'enc_tile_size', 'use_streaming')
        )

    Buffer donation can further reduce memory usage for the _update_block helper (already applied).

    """
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]

    if not use_gemm:
        # Linear-space computation (matches reference exactly)
        # Convert log position back to linear for matrix multiply
        position_distance = jnp.exp(log_position_distance)

        spike_waveform_feature_distance = kde_distance(
            decoding_spike_waveform_features,
            encoding_spike_waveform_features,
            waveform_stds,
        )  # shape (n_encoding_spikes, n_decoding_spikes)

        marginal_density = (
            spike_waveform_feature_distance.T @ position_distance / n_encoding_spikes
        )  # shape (n_decoding_spikes, n_position_bins)
        # Use safe_log to avoid -inf from zero marginal_density or mean_rate
        return safe_log(
            mean_rate * jnp.where(occupancy > 0.0, marginal_density / occupancy, EPS),
            eps=EPS,
        )

    # Validation: streaming requires chunking and streaming parameters
    if use_streaming:
        if enc_tile_size is None:
            raise ValueError(
                "use_streaming=True requires enc_tile_size to be specified"
            )
        if enc_tile_size >= n_encoding_spikes:
            raise ValueError(
                f"use_streaming=True requires enc_tile_size < n_encoding_spikes "
                f"(got enc_tile_size={enc_tile_size}, n_encoding_spikes={n_encoding_spikes}). "
                f"Streaming is only beneficial when chunking encoding spikes."
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

    # Log-space computation with GEMM optimization
    if use_streaming:
        # When streaming, we don't use log_position_distance at all
        n_pos = position_eval_points.shape[0]
    else:
        n_pos = log_position_distance.shape[1]
    n_dec = decoding_spike_waveform_features.shape[0]

    # Uniform weights: log(1/n) for each encoding spike
    log_w = -jnp.log(float(n_encoding_spikes))  # n_encoding_spikes always > 0

    # If enc_tile_size specified, chunk over encoding spikes
    if enc_tile_size is not None and enc_tile_size < n_encoding_spikes:
        # Use online logsumexp to accumulate across encoding chunks
        # This reduces peak memory from O(n_enc * n_pos) to O(enc_tile_size * n_pos)
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
        return jax.nn.logsumexp(log_w + log_position_distance + y_col[:, None], axis=0)

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
        return jax.nn.logsumexp(log_w + log_pos_tile + y_col[:, None], axis=0)

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

    # Add mean rate and subtract occupancy (in log)
    # safe_log clamps to LOG_EPS instead of producing -inf for zero values
    log_mean_rate = safe_log(mean_rate, eps=EPS)
    log_occ = safe_log(occupancy, eps=EPS)

    # Result: log(mean_rate * marginal / occupancy)
    # Use where to handle occupancy = 0 cases (still set to -inf explicitly)
    log_joint = jnp.where(
        occupancy[None, :] > 0.0,
        log_mean_rate + log_marginal - log_occ[None, :],
        jnp.log(0.0),  # -inf for zero occupancy (intentional masking)
    )

    return log_joint


# JIT-compile with static arguments for performance
# This allows JAX to specialize the function for different tile sizes and modes
estimate_log_joint_mark_intensity = jax.jit(
    estimate_log_joint_mark_intensity,
    static_argnames=("use_gemm", "pos_tile_size", "enc_tile_size", "use_streaming"),
)


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

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    if n_decoding_spikes == 0:
        return jnp.full((0, n_position_bins), LOG_EPS)

    out = jnp.zeros((n_decoding_spikes, n_position_bins))
    for start_ind in range(0, n_decoding_spikes, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        block_result = estimate_log_joint_mark_intensity(
            decoding_spike_waveform_features[block_inds],
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
        )
        out = jax.lax.dynamic_update_slice(out, block_result, (start_ind, 0))

    return jnp.clip(out, a_min=LOG_EPS, a_max=None)


def fit_clusterless_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    sampling_frequency: int = 500,
    weights: jnp.ndarray | None = None,
    position_std: float = np.sqrt(12.5),
    waveform_std: float = 24.0,
    block_size: int = 100,
    enc_tile_size: int | None = None,
    pos_tile_size: int | None = None,
    use_streaming: bool = False,
    disable_progress_bar: bool = False,
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
    sampling_frequency : int, optional
        Samples per second, by default 500
    position_std : float, optional
        Gaussian smoothing standard deviation for position, by default sqrt(12.5)
    waveform_std : float, optional
        Gaussian smoothing standard deviation for waveform, by default 24.0
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    Returns
    -------
    encoding_model : dict
    """
    if environment.place_bin_centers_ is None:
        raise ValueError(
            "Environment must be fitted with place_bin_centers_. "
            "Call environment.fit_place_grid() first."
        )

    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if isinstance(position_std, int | float):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])
    if isinstance(waveform_std, int | float):
        waveform_std = jnp.array([waveform_std] * spike_waveform_features[0].shape[1])

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        position1D = get_linearized_position(
            position,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position1D
        )
    else:
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position
        )

    occupancy = occupancy_model.predict(interior_place_bin_centers)
    encoding_positions = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)

    n_time_bins = int((position_time[-1] - position_time[0]) * sampling_frequency)
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
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= position_time[0],
            electrode_spike_times <= position_time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        bounded_spike_waveform_features.append(
            electrode_spike_waveform_features[is_in_bounds]
        )
        mean_rates.append(len(electrode_spike_times) / n_time_bins)
        encoding_positions.append(
            get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1]
        )
        gpi_models.append(gpi_model)

        gpi_density = gpi_model.predict(interior_place_bin_centers)
        summed_ground_process_intensity += jnp.clip(
            mean_rates[-1] * jnp.where(occupancy > 0.0, gpi_density / occupancy, EPS),
            a_min=EPS,
            a_max=None,
        )

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
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
    }


def predict_clusterless_kde_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy: jnp.ndarray,
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: jnp.ndarray,
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
) -> jnp.ndarray:
    """Predict the log likelihood of the clusterless KDE model.

    Parameters
    ----------
    time : jnp.ndarray
        Decoding time bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
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
    encoding_positions : jnp.ndarray, shape (n_encoding_spikes, n_position_dims)
        Position samples used for encoding.
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

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1) or (n_time, n_position_bins)
        Shape depends on whether local or non-local decoding, respectively.
    """
    n_time = len(time)

    if is_local:
        log_likelihood = compute_local_log_likelihood(
            time,
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
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

        log_likelihood = -1.0 * summed_ground_process_intensity * jnp.ones((n_time, 1))

        for (
            electrode_encoding_spike_waveform_features,
            electrode_encoding_positions,
            electrode_mean_rate,
            electrode_decoding_spike_waveform_features,
            electrode_spike_times,
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
            strict=True,
        ):
            is_in_bounds = jnp.logical_and(
                electrode_spike_times >= time[0],
                electrode_spike_times <= time[-1],
            )
            electrode_spike_times = electrode_spike_times[is_in_bounds]
            electrode_decoding_spike_waveform_features = (
                electrode_decoding_spike_waveform_features[is_in_bounds]
            )
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

            log_likelihood += jax.ops.segment_sum(
                block_estimate_log_joint_mark_intensity(
                    electrode_decoding_spike_waveform_features,
                    electrode_encoding_spike_waveform_features,
                    waveform_std,
                    occupancy,
                    electrode_mean_rate,
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
                ),
                get_spike_time_bin_ind(electrode_spike_times, time),
                indices_are_sorted=True,
                num_segments=n_time,
            )

    return log_likelihood


def compute_local_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: jnp.ndarray,
    environment: Environment,
    mean_rates: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    block_size: int = 100,
    disable_progress_bar: bool = False,
) -> jnp.ndarray:
    """Compute the log likelihood at the animal's position.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
        Time bins for decoding.
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
    encoding_positions : jnp.ndarray
        Position samples used for encoding.
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

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
    """

    # Need to interpolate position
    interpolated_position = get_position_at_time(
        position_time, position, time, environment
    )
    occupancy = occupancy_model.predict(interpolated_position)

    # Pre-compute all spike positions and occupancies to avoid repeated computation
    # This is more efficient than computing occupancy per electrode
    all_spike_positions = []
    all_spike_position_offsets = [0]  # Track where each electrode's spikes start

    for electrode_spike_times in spike_times:
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= time[0],
            electrode_spike_times <= time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]

        position_at_spike_time = get_position_at_time(
            position_time, position, electrode_spike_times, environment
        )
        all_spike_positions.append(position_at_spike_time)
        all_spike_position_offsets.append(
            all_spike_position_offsets[-1] + len(position_at_spike_time)
        )

    # Compute occupancy once for all spike positions
    if all_spike_positions:
        all_spike_positions_concat = jnp.concatenate(all_spike_positions, axis=0)
        all_occupancies = occupancy_model.predict(all_spike_positions_concat)
    else:
        all_occupancies = jnp.array([])

    n_time = len(time)
    log_likelihood = jnp.zeros((n_time,))

    for electrode_idx, (
        electrode_encoding_spike_waveform_features,
        electrode_encoding_positions,
        electrode_mean_rate,
        electrode_gpi_model,
        electrode_decoding_spike_waveform_features,
        electrode_spike_times,
    ) in enumerate(
        zip(
            tqdm(
                encoding_spike_waveform_features,
                unit="electrode",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_positions,
            mean_rates,
            gpi_models,
            spike_waveform_features,
            spike_times,
            strict=True,
        )
    ):
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= time[0],
            electrode_spike_times <= time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        electrode_decoding_spike_waveform_features = (
            electrode_decoding_spike_waveform_features[is_in_bounds]
        )

        # Get pre-computed position and occupancy for this electrode
        start_idx = all_spike_position_offsets[electrode_idx]
        end_idx = all_spike_position_offsets[electrode_idx + 1]
        position_at_spike_time = all_spike_positions[electrode_idx]
        occupancy_at_spike_time = all_occupancies[start_idx:end_idx]

        marginal_density = block_kde(
            eval_points=jnp.concatenate(
                (
                    position_at_spike_time,
                    electrode_decoding_spike_waveform_features,
                ),
                axis=1,
            ),
            samples=jnp.concatenate(
                (
                    electrode_encoding_positions,
                    electrode_encoding_spike_waveform_features,
                ),
                axis=1,
            ),
            std=jnp.concatenate((position_std, waveform_std)),
            block_size=block_size,
        )

        # Use safe_log to avoid -inf from zero marginal_density or occupancy
        # The where still protects against division by zero occupancy
        log_likelihood += jax.ops.segment_sum(
            safe_log(
                electrode_mean_rate
                * jnp.where(
                    occupancy_at_spike_time > 0.0,
                    marginal_density / occupancy_at_spike_time,
                    EPS,  # Use EPS instead of 0 to avoid log(0)
                ),
                eps=EPS,
            ),
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * jnp.where(
            occupancy > 0.0,
            electrode_gpi_model.predict(interpolated_position) / occupancy,
            0.0,
        )
    return log_likelihood[:, jnp.newaxis]
