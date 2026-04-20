from typing import Literal

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
    block_log_kde,
    gaussian_pdf,
    get_position_at_time,
    get_spike_time_bin_ind,
    log_gaussian_pdf,
    safe_log,
)

# Private-impl / public-JIT-alias pattern used throughout this module:
#
#     def _foo_impl(...): ...                # un-JITted body, plain Python
#     foo = jax.jit(_foo_impl, static_argnames=...)   # public name
#
# The ``_impl`` suffix marks the un-JITted body, distinct from the
# public ``jax.jit``-wrapped alias of the same name without the suffix.
# The wrapped alias is what callers use; the ``_impl`` is what scan /
# fori_loop bodies and other JIT-traced contexts call directly so JAX
# can trace through the function body without re-entering a JIT
# boundary.  Mirrors the convention in ``core.py``.

# Maximum waveform feature dimensions for the compensated-linear fast path.
# Above this threshold, mark kernel underflow causes accuracy degradation
# and the logsumexp path is used instead. Empirically validated: ≤8 dims
# gives <8e-6 max absolute error vs logsumexp across 10 random seeds.
_COMPENSATED_LINEAR_MAX_FEATURES = 8


# Fraction of the GPU's reported ``bytes_limit`` that
# :func:`auto_select_tile_sizes` treats as the usable budget for
# mark-kernel and position-kernel intermediates.  The remainder is
# reserved for XLA workspaces, other JIT regions, scan-carry buffers,
# and per-iteration activations.  25% is conservative; if users push
# single dense workloads they can pass an explicit
# ``memory_budget_bytes`` to raise the target.
_AUTO_TILE_MEMORY_BUDGET_FRACTION = 0.25

# Default memory budget when neither the device reports a memory limit
# nor the caller supplies one — 2 GB is a safe lower bound that fits
# a consumer GPU (8 GB with 25% usable) and a CPU run.
_AUTO_TILE_DEFAULT_BUDGET_BYTES = int(2e9)

# Minimum usable encoding-tile size.  Tiles smaller than 256 defeat the
# GEMM vectorization and aren't worth the overhead, so the heuristic
# clamps upward rather than splitting further.
_AUTO_TILE_MIN_ENC_TILE = 256

# Safety factor applied to the (n_enc × block_size) and (n_enc × n_pos)
# budget calculations — accounts for the ~3 intermediate copies of each
# tensor that live simultaneously during the compensated-linear matmul
# (``logK_mark``, ``K_wf_stable``, ``W = K_wf_stable * sqrt_scale``).
_AUTO_TILE_INTERMEDIATE_COPIES = 3


def auto_select_tile_sizes(
    n_enc: int,
    n_dec: int,
    n_pos: int,
    n_wf: int = 4,  # noqa: ARG001 — reserved for future heuristics
    memory_budget_bytes: int | None = None,
) -> dict[str, int | None]:
    """Heuristic tile sizes for ``predict_clusterless_kde_log_likelihood``.

    Picks ``block_size`` (decoding-spike block inside the ``fori_loop``
    body — see :func:`block_estimate_log_joint_mark_intensity`) and
    ``enc_tile_size`` (encoding-spike tile inside
    :func:`_estimate_with_enc_chunking`) so that the mark-kernel and
    position-kernel intermediates fit inside a target memory budget.

    The default budget is ``0.25 *`` the GPU's reported ``bytes_limit``
    (or 2 GB if the device doesn't expose memory stats — e.g. CPU).
    Users can override with ``memory_budget_bytes``.

    Parameters
    ----------
    n_enc : int
        Number of encoding spikes for a single electrode.
    n_dec : int
        Number of decoding spikes for a single electrode.  Caps
        ``block_size`` from above — larger blocks than ``n_dec`` are
        wasted shape padding.
    n_pos : int
        Number of position bins.
    n_wf : int, default 4
        Number of waveform features.  Currently unused; reserved for
        future heuristics that choose between the compensated-linear
        and logsumexp paths based on ``n_wf``.
    memory_budget_bytes : int | None, default None
        Target usable memory budget.  When ``None``, queries
        ``jax.devices()[0].memory_stats()["bytes_limit"]`` and applies
        :data:`_AUTO_TILE_MEMORY_BUDGET_FRACTION`; falls back to 2 GB
        if the query fails (CPU, older JAX versions).

    Returns
    -------
    dict with keys

    * ``block_size`` — int, size of each decoding-spike block in the
      ``fori_loop``.  Always ``>= 1`` and ``<= n_dec``.
    * ``enc_tile_size`` — int or None.  ``None`` signals "no encoding
      tiling needed" (the full mark kernel fits).  When not ``None``,
      always ``>= 256`` (tiles smaller than that defeat GEMM
      vectorization).

    Notes
    -----
    This helper is a starting point, not a precise cost model.  Runtime
    tuning against a representative workload is still the gold standard.
    The heuristic ignores ``n_wf`` (the factor-of-3 intermediate
    estimate already dominates) and the ``(n_time, n_pos)``
    accumulator (constant across tile choices).  Resolved values
    become static JIT arguments — each distinct value triggers a
    recompile, so cache the result per electrode-shape combination
    rather than calling on every ``predict`` invocation.
    """
    if memory_budget_bytes is None:
        memory_budget_bytes = _default_memory_budget_bytes()

    n_enc = max(int(n_enc), 1)
    n_dec = max(int(n_dec), 1)
    n_pos = max(int(n_pos), 1)
    f32 = 4

    # block_size bound: fit (n_enc × block_size × f32) × intermediate_copies
    # into the budget.
    max_block = max(
        memory_budget_bytes // (_AUTO_TILE_INTERMEDIATE_COPIES * n_enc * f32),
        1,
    )
    block_size = int(min(max_block, n_dec))

    # enc_tile_size: only needed when the (n_enc × n_pos) position kernel
    # would overflow the budget allowing for the ~3 simultaneous live
    # tensors (``logK_pos_chunk``, ``K_pos_stable``, ``P``).  Using the
    # same intermediate-copies factor as the block_size formula keeps the
    # trigger consistent and prevents a boundary case where ``n_enc *
    # n_pos * 4`` just barely fits but the 3-tensor working set doesn't.
    if n_enc * n_pos * f32 * _AUTO_TILE_INTERMEDIATE_COPIES > memory_budget_bytes:
        enc_tile_size: int | None = int(
            max(
                memory_budget_bytes // (_AUTO_TILE_INTERMEDIATE_COPIES * n_pos * f32),
                _AUTO_TILE_MIN_ENC_TILE,
            )
        )
    else:
        enc_tile_size = None

    return {"block_size": block_size, "enc_tile_size": enc_tile_size}


def _default_memory_budget_bytes() -> int:
    """Query JAX device memory and apply the auto-tile usable fraction."""
    try:
        device = jax.devices()[0]
        stats = device.memory_stats() or {}
        bytes_limit = int(stats.get("bytes_limit", 0))
        if bytes_limit > 0:
            return int(bytes_limit * _AUTO_TILE_MEMORY_BUDGET_FRACTION)
    except (AttributeError, KeyError, RuntimeError, NotImplementedError):
        # TPU / Metal / older CPU backends may raise NotImplementedError
        # from memory_stats(); treat as "no budget reported" and fall back.
        pass
    return _AUTO_TILE_DEFAULT_BUDGET_BYTES


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


def _log_kde_distance_impl(
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


log_kde_distance = jax.jit(_log_kde_distance_impl)


def _log_kde_distance_streaming_impl(
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


log_kde_distance_streaming = jax.jit(_log_kde_distance_streaming_impl)


def _precompute_encoding_gemm_quantities(
    encoding_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute the encoding-side GEMM quantities used by the log mark kernel.

    These quantities depend only on the per-electrode encoding spikes and
    waveform bandwidths — they are constant across decoding-spike blocks
    and can be hoisted outside any loop that iterates over decoding-spike
    blocks (e.g. the block ``fori_loop`` in
    ``_block_estimate_log_joint_mark_intensity_impl``).

    Pairs with :func:`_compute_log_mark_kernel_from_precomputed`.

    Parameters
    ----------
    encoding_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)

    Returns
    -------
    inv_sigma : jnp.ndarray, shape (n_features,)
        ``1 / clip(waveform_stds, EPS)``.
    log_norm_const : jnp.ndarray, shape ()
        0-d JAX scalar equal to ``-0.5 * (D * log(2π) + 2 * sum(log(sigma)))``.
        Kept as a traced scalar (not a Python float) so the tuple is a
        homogeneous JAX pytree — safe to carry through a ``scan`` without
        recompilation on different ``waveform_stds`` values.
    Y : jnp.ndarray, shape (n_encoding_spikes, n_features)
        Encoding features scaled by ``inv_sigma``.
    y2 : jnp.ndarray, shape (n_encoding_spikes,)
        Row-wise squared norms ``sum(Y**2, axis=1)``.

    Notes
    -----
    The return is a flat tuple rather than a dict for cheaper JAX pytree
    handling inside ``scan``/``fori_loop`` carries.
    """
    n_features = waveform_stds.shape[0]
    # Clip to avoid division by zero for degenerate feature dimensions.
    waveform_stds = jnp.clip(waveform_stds, min=EPS)
    inv_sigma = 1.0 / waveform_stds
    # Factor of 2 because we have sum(log(sigma)), not log(sigma**2).
    log_norm_const = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )
    Y = encoding_features * inv_sigma[None, :]
    y2 = jnp.sum(Y**2, axis=1)
    return inv_sigma, log_norm_const, Y, y2


def _compute_log_mark_kernel_from_precomputed(
    decoding_features: jnp.ndarray,
    precomp: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Compute the log mark kernel from decoding features + precomputed encoding quantities.

    Pairs with :func:`_precompute_encoding_gemm_quantities`.  When called
    inside a loop that iterates over decoding-spike blocks, the caller
    computes ``precomp`` once outside the loop so the encoding-side
    scaling and row-wise sum-of-squares are never repeated.

    Parameters
    ----------
    decoding_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    precomp : tuple
        Output of :func:`_precompute_encoding_gemm_quantities`.

    Returns
    -------
    logK_mark : jnp.ndarray, shape (n_encoding_spikes, n_decoding_spikes)
    """
    inv_sigma, log_norm_const, Y, y2 = precomp
    X = decoding_features * inv_sigma[None, :]
    x2 = jnp.sum(X**2, axis=1)
    # GEMM: (n_dec, n_features) @ (n_features, n_enc) -> (n_dec, n_enc).
    cross_term = X @ Y.T
    # Clamp sq_dist to non-negative to absorb catastrophic cancellation in
    # the expanded GEMM form when x ≈ y.
    sq_dist = jnp.maximum(y2[:, None] + x2[None, :] - 2.0 * cross_term.T, 0.0)
    return log_norm_const - 0.5 * sq_dist


def _precompute_decoding_gemm_quantities(
    decoding_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute the decoding-side GEMM quantities used by the log mark kernel.

    Mirror of :func:`_precompute_encoding_gemm_quantities` for the loops
    that iterate over *encoding*-spike chunks (see
    :func:`_estimate_with_enc_chunking` and
    :func:`_compensated_linear_marginal_chunked`).  In those loops the
    decoding features are constant across chunks, so the scaled copy and
    its row-wise sum-of-squares can be hoisted outside the loop body.

    Returns
    -------
    inv_sigma : jnp.ndarray, shape (n_features,)
    log_norm_const : jnp.ndarray, shape ()
    X : jnp.ndarray, shape (n_decoding_spikes, n_features)
        Decoding features scaled by ``inv_sigma``.
    x2 : jnp.ndarray, shape (n_decoding_spikes,)
        Row-wise squared norms ``sum(X**2, axis=1)``.
    """
    n_features = waveform_stds.shape[0]
    waveform_stds = jnp.clip(waveform_stds, min=EPS)
    inv_sigma = 1.0 / waveform_stds
    log_norm_const = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )
    X = decoding_features * inv_sigma[None, :]
    x2 = jnp.sum(X**2, axis=1)
    return inv_sigma, log_norm_const, X, x2


def _log_mark_kernel_chunk_from_decoding_precomputed(
    encoding_chunk_features: jnp.ndarray,
    dec_precomp: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Log mark kernel for one encoding chunk against all decoding spikes.

    Pairs with :func:`_precompute_decoding_gemm_quantities`.  Intended to
    be called inside the encoding-chunk ``fori_loop`` body of
    :func:`_estimate_with_enc_chunking` and
    :func:`_compensated_linear_marginal_chunked` — the scale factors
    ``inv_sigma`` / ``log_norm_const`` and the decoding-side scaled
    features ``X`` / row-norms ``x2`` are loop-invariant and are
    precomputed once outside.

    Returns
    -------
    logK_mark_chunk : jnp.ndarray, shape (enc_tile_size, n_decoding_spikes)
    """
    inv_sigma, log_norm_const, X, x2 = dec_precomp
    Y_chunk = encoding_chunk_features * inv_sigma[None, :]
    y2_chunk = jnp.sum(Y_chunk**2, axis=1)
    # GEMM: (n_dec, n_features) @ (n_features, tile) -> (n_dec, tile).
    cross_term = X @ Y_chunk.T
    sq_dist = jnp.maximum(y2_chunk[:, None] + x2[None, :] - 2.0 * cross_term.T, 0.0)
    return log_norm_const - 0.5 * sq_dist


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

    Notes
    -----
    Thin wrapper around :func:`_precompute_encoding_gemm_quantities` +
    :func:`_compute_log_mark_kernel_from_precomputed`.  Prefer calling
    those two directly when you need to reuse the encoding-side
    quantities across multiple decoding-spike batches (e.g. the
    ``fori_loop`` body in ``_block_estimate_log_joint_mark_intensity_impl``).

    Currently still used by the encoding-chunked streaming paths
    (``_estimate_with_enc_chunking`` and ``_compensated_linear_marginal_chunked``),
    which call this per chunk.  Those paths could be refactored to hoist
    the chunk-invariant ``inv_sigma`` / ``log_norm_const`` outside the
    per-chunk loop — small potential gain, deferred from this PR.
    """
    precomp = _precompute_encoding_gemm_quantities(encoding_features, waveform_stds)
    return _compute_log_mark_kernel_from_precomputed(decoding_features, precomp)


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

    # Hoist chunk-invariant decoding-side GEMM quantities (inv_sigma,
    # log_norm_const, X_scaled, x2) outside the fori_loop so each
    # encoding chunk reuses them instead of recomputing.  See
    # :func:`_precompute_decoding_gemm_quantities`.
    dec_gemm_precomp = _precompute_decoding_gemm_quantities(
        decoding_spike_waveform_features, waveform_stds
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

        # Compute log mark kernel for this encoding chunk: (enc_tile_size, n_dec).
        # ``dec_gemm_precomp`` (inv_sigma, log_norm_const, X_scaled, x2) is
        # hoisted outside the fori_loop; this helper does only the per-chunk
        # enc-side scaling + GEMM + completion of the squared distance.
        logK_mark_chunk = _log_mark_kernel_chunk_from_decoding_precomputed(
            enc_chunk_features, dec_gemm_precomp
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


def _compensated_linear_marginal(
    logK_mark: jnp.ndarray,
    log_position_distance: jnp.ndarray,
    log_w: float,
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
    log_w : float
        Log uniform weight, typically ``-log(n_enc)``.
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
    log_scale = total_max_per_enc - global_max  # (n_enc,)

    # Stabilized kernels: all values in [0, 1]
    K_pos_stable = jnp.exp(log_position_distance - max_pos[:, None])  # (n_enc, n_pos)
    K_wf_stable = jnp.exp(logK_mark - max_wf[:, None])  # (n_enc, n_dec)

    # Absorb sqrt(scale) into each factor so the matmul carries the weight.
    # Identity: sum_e scale[e] * K_wf[e,d] * K_pos[e,p]
    #         = sum_e sqrt_scale[e]^2 * K_wf[e,d] * K_pos[e,p]
    #         = (W.T @ P)[d,p]   where W[e,d] = K_wf[e,d]*sqrt_scale[e]
    sqrt_scale = jnp.exp(0.5 * log_scale)  # (n_enc,)
    W = K_wf_stable * sqrt_scale[:, None]  # (n_enc, n_dec)
    P = K_pos_stable * sqrt_scale[:, None]  # (n_enc, n_pos)

    # Single BLAS matmul: (n_dec, n_enc) @ (n_enc, n_pos) -> (n_dec, n_pos)
    marginal_scaled = W.T @ P

    # Back to log space.  Use double-where to produce LOG_EPS (not NaN)
    # when the matmul result is zero, matching the logsumexp path's
    # contract via block_estimate_log_joint_mark_intensity's LOG_EPS clamp.
    safe_marginal = jnp.where(marginal_scaled > 0.0, marginal_scaled, 1.0)
    log_marginal = jnp.where(
        marginal_scaled > 0.0,
        jnp.log(safe_marginal) + global_max,
        LOG_EPS,
    )

    # Add mean rate and subtract occupancy (in log)
    log_mean_rate = safe_log(mean_rate, eps=EPS)
    log_occ = safe_log(occupancy, eps=EPS)

    return jnp.where(
        occupancy[None, :] > 0.0,
        log_mean_rate + log_marginal - log_occ[None, :],
        jnp.log(0.0),  # -inf for zero occupancy
    )


def _compensated_linear_marginal_chunked(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None,
    log_w: float,
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
    log_w : float
        Log uniform weight, typically ``-log(n_enc)``.
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

    # Validity mask for padded entries
    enc_valid = jnp.arange(n_enc_padded) < n_enc  # (n_enc_padded,)

    # Hoist chunk-invariant decoding-side GEMM quantities outside the
    # chunk scan.  See :func:`_precompute_decoding_gemm_quantities`.
    dec_gemm_precomp = _precompute_decoding_gemm_quantities(
        decoding_spike_waveform_features, waveform_stds
    )

    def process_chunk(carry, chunk_idx):
        """Process one encoding chunk with online max rescaling."""
        running_sum, running_max = carry
        enc_start = chunk_idx * enc_tile_size

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

        # Compute mark kernel for this chunk: (enc_tile, n_dec).
        # ``dec_gemm_precomp`` is hoisted outside the scan.
        logK_mark_chunk = _log_mark_kernel_chunk_from_decoding_precomputed(
            enc_chunk, dec_gemm_precomp
        )

        # Mask padded entries
        chunk_valid = jax.lax.dynamic_slice(enc_valid, (enc_start,), (enc_tile_size,))
        logK_pos_chunk = jnp.where(chunk_valid[:, None], logK_pos_chunk, -jnp.inf)
        logK_mark_chunk = jnp.where(chunk_valid[:, None], logK_mark_chunk, -jnp.inf)

        # Per-row maxima.  Invalid (all -inf) rows get -inf, which would
        # produce NaN in the stabilization step (-inf - (-inf) = NaN).
        # Replace with 0.0 so those rows exp to 0 and contribute nothing.
        max_pos = jnp.max(logK_pos_chunk, axis=1)  # (tile,)
        max_wf = jnp.max(logK_mark_chunk, axis=1)  # (tile,)
        row_valid = chunk_valid  # rows with real data
        max_pos = jnp.where(row_valid, max_pos, 0.0)
        max_wf = jnp.where(row_valid, max_wf, 0.0)
        chunk_total = jnp.where(row_valid, max_pos + max_wf + log_w, -jnp.inf)
        chunk_max = jnp.max(chunk_total)

        # Online max update: rescale running_sum if new max is larger.
        # If chunk_max <= running_max, exp(running_max - new_max) = 1 (no-op).
        new_max = jnp.maximum(running_max, chunk_max)
        running_sum = running_sum * jnp.exp(running_max - new_max)

        # Stabilize this chunk's kernels.  Invalid rows get 0 because
        # logK - 0 is still -inf, and exp(-inf) = 0.
        K_pos_stable = jnp.exp(logK_pos_chunk - max_pos[:, None])
        K_wf_stable = jnp.exp(logK_mark_chunk - max_wf[:, None])

        # Scale factors relative to current global max
        log_scale = chunk_total - new_max
        sqrt_scale = jnp.exp(0.5 * log_scale)
        W = K_wf_stable * sqrt_scale[:, None]  # (tile, n_dec)
        P = K_pos_stable * sqrt_scale[:, None]  # (tile, n_pos)

        # Accumulate: matmul adds to running sum
        running_sum = running_sum + W.T @ P  # (n_dec, n_pos)

        return (running_sum, new_max), None

    # Initialize: zero sum, -inf max
    init_sum = jnp.zeros((n_dec, n_pos))
    init_max = jnp.array(-jnp.inf)

    (final_sum, final_max), _ = jax.lax.scan(
        process_chunk,
        (init_sum, init_max),
        jnp.arange(n_chunks),
    )

    # Back to log space (double-where for -inf contract)
    safe_sum = jnp.where(final_sum > 0.0, final_sum, 1.0)
    log_marginal = jnp.where(
        final_sum > 0.0,
        jnp.log(safe_sum) + final_max,
        LOG_EPS,
    )

    # Add mean rate and subtract occupancy
    log_mean_rate = safe_log(mean_rate, eps=EPS)
    log_occ = safe_log(occupancy, eps=EPS)

    return jnp.where(
        occupancy[None, :] > 0.0,
        log_mean_rate + log_marginal - log_occ[None, :],
        jnp.log(0.0),
    )


def _estimate_log_joint_mark_intensity_impl(
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
    precomputed_enc_gemm: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | None = None,
    n_real_encoding_spikes: jnp.ndarray | None = None,
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
            _estimate_log_joint_mark_intensity_impl,
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

        # When ``n_real_encoding_spikes`` is provided (electrode-scan batch
        # padding), use it for the 1/n normalization instead of the padded
        # shape.  Keep the ``n_encoding_spikes > 0`` guard on the PADDED
        # shape — inside the scan, max_n_enc >= 1 always, so the guard
        # passes and the real-count normalization applies.
        if n_real_encoding_spikes is not None:
            safe_n = jnp.maximum(n_real_encoding_spikes.astype(jnp.float32), 1.0)
            has_real_enc = n_real_encoding_spikes > 0
        else:
            # Double-where: substitute safe denominator, then select result
            safe_n = jnp.where(n_encoding_spikes > 0, n_encoding_spikes, 1)
            has_real_enc = n_encoding_spikes > 0
        marginal_density = jnp.where(
            has_real_enc,
            spike_waveform_feature_distance.T @ position_distance / safe_n,
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

    # Uniform weights: log(1/n) for each encoding spike.  When
    # ``n_real_encoding_spikes`` is provided (electrode-scan batch padding),
    # use the real count instead of the padded shape — padded encoding
    # slots are masked to ``-inf`` in ``log_position_distance`` upstream
    # so they contribute nothing to the matmul, but the normalization
    # ``1/n`` must still reflect the real count.
    if n_real_encoding_spikes is not None:
        safe_n = jnp.maximum(n_real_encoding_spikes.astype(jnp.float32), 1.0)
    else:
        # Use max(n, 1) to avoid log(0); when n=0 the result is unused.
        safe_n = jnp.where(n_encoding_spikes > 0, float(n_encoding_spikes), 1.0)
    log_w = -jnp.log(safe_n)

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
    # Build log-kernel matrix for marks: (n_enc, n_dec).
    # When precomputed_enc_gemm is provided (e.g. by the block fori_loop,
    # which hoists the encoding-side scaling/sum-of-squares outside the
    # loop body), reuse those quantities instead of recomputing them per
    # decoding-spike block.
    if precomputed_enc_gemm is not None:
        logK_mark = _compute_log_mark_kernel_from_precomputed(
            decoding_spike_waveform_features, precomputed_enc_gemm
        )
    else:
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
    _estimate_log_joint_mark_intensity_impl,
    static_argnames=("use_gemm", "pos_tile_size", "enc_tile_size", "use_streaming"),
)


def _block_estimate_log_joint_mark_intensity_impl(
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
    """JIT-traceable block-loop body for block_estimate_log_joint_mark_intensity.

    Replaces the Python for-loop over decoding spike blocks with a
    ``jax.lax.fori_loop`` so the entire block iteration compiles to a single
    JAX program (one kernel dispatch sequence on GPU rather than one per block).

    Assumes ``decoding_spike_waveform_features.shape[0] > 0``. The empty-array
    edge case is handled by the public ``block_estimate_log_joint_mark_intensity``
    wrapper before entering JIT.
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_features = decoding_spike_waveform_features.shape[1]
    n_position_bins = occupancy.shape[0]

    # Pad to a multiple of block_size so each fori_loop iteration can
    # use a static-size dynamic_slice.  Edge-replication (not zero-fill)
    # is required: padded rows are byte-identical copies of the last
    # real row, so the per-encoding-spike reduction
    # ``max(logK_mark, axis=1)`` in ``_compensated_linear_marginal`` is
    # uncontaminated.  Zero-fill would seed "fake spikes" near the
    # encoding cloud's center that frequently win the max.
    n_blocks = (n_decoding_spikes + block_size - 1) // block_size
    n_padded = n_blocks * block_size
    pad_amount = n_padded - n_decoding_spikes
    if pad_amount > 0:
        decoding_spike_waveform_features = jnp.pad(
            decoding_spike_waveform_features,
            ((0, pad_amount), (0, 0)),
            mode="edge",
        )

    # Hoist the encoding-side GEMM quantities outside the fori_loop when
    # the downstream non-chunked-GEMM path is taken (the only branch
    # that reads ``precomputed_enc_gemm``).  All gating conditions are
    # static at trace time, so JAX traces only the taken arm.  The
    # ``enc_tile_size >= n_encoding_spikes`` case routes through
    # non-chunked too — the chunked sub-paths only fire when the tile
    # is strictly smaller.
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]
    use_precomputed_gemm = (
        use_gemm
        and not use_streaming
        and (enc_tile_size is None or enc_tile_size >= n_encoding_spikes)
    )
    if use_precomputed_gemm:
        precomp = _precompute_encoding_gemm_quantities(
            encoding_spike_waveform_features, waveform_stds
        )
    else:
        precomp = None

    def process_block(i: int, out: jnp.ndarray) -> jnp.ndarray:
        start = i * block_size
        block_features = jax.lax.dynamic_slice(
            decoding_spike_waveform_features,
            (start, 0),
            (block_size, n_features),
        )
        block_result = _estimate_log_joint_mark_intensity_impl(
            block_features,
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
            precomputed_enc_gemm=precomp,
        )
        return jax.lax.dynamic_update_slice(out, block_result, (start, 0))

    out = jnp.zeros((n_padded, n_position_bins))
    out = jax.lax.fori_loop(0, n_blocks, process_block, out)
    return jnp.clip(out[:n_decoding_spikes], min=LOG_EPS, max=None)


_block_estimate_log_joint_mark_intensity_jit = jax.jit(
    _block_estimate_log_joint_mark_intensity_impl,
    static_argnames=(
        "block_size",
        "use_gemm",
        "pos_tile_size",
        "enc_tile_size",
        "use_streaming",
    ),
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

    Notes
    -----
    The empty-decoding-spike case is handled here before dispatching to JIT.
    Non-empty inputs are forwarded to the JIT-compiled
    ``_block_estimate_log_joint_mark_intensity_impl`` which replaces the
    Python block-loop with a ``jax.lax.fori_loop`` so the full iteration
    compiles to a single JAX program.
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    if n_decoding_spikes == 0:
        return jnp.full((0, n_position_bins), LOG_EPS)

    return _block_estimate_log_joint_mark_intensity_jit(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        block_size=block_size,
        use_gemm=use_gemm,
        pos_tile_size=pos_tile_size,
        enc_tile_size=enc_tile_size,
        use_streaming=use_streaming,
        encoding_positions=encoding_positions,
        position_eval_points=position_eval_points,
        position_std=position_std,
    )


def _block_estimate_with_segment_sum_impl(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None,
    spike_time_bin_ind: jnp.ndarray,
    n_time: int,
    block_size: int = 100,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
    n_real_encoding_spikes: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Block-estimate fused with segment_sum; accumulates ``(n_time, n_pos)`` directly.

    Equivalent to running
    :func:`block_estimate_log_joint_mark_intensity` then
    ``jax.ops.segment_sum(..., num_segments=n_time, indices_are_sorted=True)``,
    but with the segment_sum performed per-block inside the
    ``fori_loop`` so the full ``(n_dec, n_pos)`` mark-intensity matrix
    is never materialized.

    Padded decoding-spike slots are assigned segment id ``n_time``;
    ``segment_sum(num_segments=n_time)`` drops indices ``>= num_segments``,
    so padded slots contribute nothing.  Assumes
    ``decoding_spike_waveform_features.shape[0] > 0``; the empty-array
    case is handled by the public wrapper.

    Notes
    -----
    Within the electrode-scan path the fused-segment_sum design
    avoids carrying the ``(n_dec, n_pos)`` block output alongside
    the outer ``(n_time, n_pos)`` accumulator.  See
    :func:`_predict_nonlocal_electrode_scan_impl` for the
    accumulator's memory profile.
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_features = decoding_spike_waveform_features.shape[1]
    n_position_bins = occupancy.shape[0]

    # Pad decoding features (edge-replication — see
    # _block_estimate_log_joint_mark_intensity_impl for rationale) so the
    # fori_loop body can use static-size dynamic_slice.  Pad spike_time_bin_ind
    # with the sentinel segment id ``n_time`` so padded spikes contribute
    # nothing to the segment_sum accumulator (``num_segments=n_time`` ignores
    # indices >= num_segments).
    n_blocks = (n_decoding_spikes + block_size - 1) // block_size
    n_padded = n_blocks * block_size
    pad_amount = n_padded - n_decoding_spikes
    if pad_amount > 0:
        decoding_spike_waveform_features = jnp.pad(
            decoding_spike_waveform_features,
            ((0, pad_amount), (0, 0)),
            mode="edge",
        )
        spike_time_bin_ind = jnp.pad(
            spike_time_bin_ind,
            (0, pad_amount),
            mode="constant",
            constant_values=n_time,
        )

    # Hoist the encoding-side GEMM quantities outside the fori_loop whenever
    # the downstream path is the non-chunked use_gemm GEMM (see
    # _block_estimate_log_joint_mark_intensity_impl for the full rationale).
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]
    use_precomputed_gemm = (
        use_gemm
        and not use_streaming
        and (enc_tile_size is None or enc_tile_size >= n_encoding_spikes)
    )
    if use_precomputed_gemm:
        precomp = _precompute_encoding_gemm_quantities(
            encoding_spike_waveform_features, waveform_stds
        )
    else:
        precomp = None

    def process_block(i: int, out: jnp.ndarray) -> jnp.ndarray:
        start = i * block_size
        block_features = jax.lax.dynamic_slice(
            decoding_spike_waveform_features,
            (start, 0),
            (block_size, n_features),
        )
        block_seg_ids = jax.lax.dynamic_slice(
            spike_time_bin_ind, (start,), (block_size,)
        )
        block_mark = _estimate_log_joint_mark_intensity_impl(
            block_features,
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
            precomputed_enc_gemm=precomp,
            n_real_encoding_spikes=n_real_encoding_spikes,
        )
        # Apply LOG_EPS floor so this fused output matches the separate
        # (block_estimate -> segment_sum) pipeline byte-for-byte.
        block_mark = jnp.clip(block_mark, min=LOG_EPS, max=None)
        # ``indices_are_sorted=True`` is valid: caller passes sorted
        # ``spike_time_bin_ind`` and our sentinel ``n_time`` is ≥ all real ids.
        block_contribution = jax.ops.segment_sum(
            block_mark,
            block_seg_ids,
            num_segments=n_time,
            indices_are_sorted=True,
        )
        return out + block_contribution

    out = jnp.zeros((n_time, n_position_bins))
    return jax.lax.fori_loop(0, n_blocks, process_block, out)


_block_estimate_with_segment_sum_jit = jax.jit(
    _block_estimate_with_segment_sum_impl,
    static_argnames=(
        "n_time",
        "block_size",
        "use_gemm",
        "pos_tile_size",
        "enc_tile_size",
        "use_streaming",
    ),
)


def block_estimate_with_segment_sum_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray | None,
    spike_time_bin_ind: jnp.ndarray,
    n_time: int,
    block_size: int = 100,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_positions: jnp.ndarray | None = None,
    position_eval_points: jnp.ndarray | None = None,
    position_std: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Fused block_estimate + segment_sum.

    Public wrapper around :func:`_block_estimate_with_segment_sum_impl`
    that handles the empty-decoding-spikes case before dispatching to
    JIT.  Output is identical to running
    :func:`block_estimate_log_joint_mark_intensity` then
    ``jax.ops.segment_sum(..., num_segments=n_time, indices_are_sorted=True)``,
    but the full ``(n_decoding_spikes, n_position_bins)`` mark-intensity
    matrix is never materialized.

    Parameters
    ----------
    spike_time_bin_ind : jnp.ndarray of int, shape (n_decoding_spikes,)
        Time-bin index for each decoding spike, sorted ascending.  All
        values must lie in ``[0, n_time)``; out-of-range values are
        silently dropped (same sentinel mechanism used for padded slots).
    n_time : int
        Number of decoding time bins (``len(time_edges)``).  Static
        JIT argument — each distinct value triggers a recompilation.

    Other parameters match :func:`block_estimate_log_joint_mark_intensity`.

    Returns
    -------
    log_likelihood_contribution : jnp.ndarray, shape (n_time, n_position_bins)
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    if n_decoding_spikes == 0:
        return jnp.zeros((n_time, n_position_bins))

    return _block_estimate_with_segment_sum_jit(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        spike_time_bin_ind,
        n_time,
        block_size=block_size,
        use_gemm=use_gemm,
        pos_tile_size=pos_tile_size,
        enc_tile_size=enc_tile_size,
        use_streaming=use_streaming,
        encoding_positions=encoding_positions,
        position_eval_points=position_eval_points,
        position_std=position_std,
    )


# Encoding-position padding sentinel.  Padded encoding rows take this
# value so that ``log_kde_distance`` returns a value far enough below
# zero that ``exp(...)`` underflows — zeroing those rows' contribution
# to the matmul without explicit masking.
#
# Float32 exp underflows below ~-87, float64 below ~-708.  At
# ``log_pos ≈ -0.5 * (pad / std)^2``, a sentinel of 1e10 stays in the
# safe regime for any ``std`` up to ~1e7 in float32 (giving
# ``log_pos ≈ -5e5``) and any ``std`` up to ~1e3 in float64 (giving
# ``log_pos ≈ -5e13``) — well beyond plausible spatial-bandwidth
# values.  Using ``jnp.inf`` instead would propagate ``inf - inf =
# NaN`` through the GEMM expansion ``y2 + x2 - 2*cross``.
_ELECTRODE_SCAN_POS_PAD_VALUE = 1.0e10


def _group_electrodes_by_n_wf(
    encoding_spike_waveform_features: list[jnp.ndarray],
) -> dict[int, list[int]]:
    """Return a mapping from ``n_wf`` to the electrode indices with that count.

    In the common case (all tetrodes have 4 features) this returns a
    single group; if a bad tetrode wire reduces ``n_wf`` for some
    electrodes, we run one JIT-compiled scan per group.

    Notes
    -----
    The alternative is to pad ragged ``waveform_features`` to a common
    ``max_n_wf`` and scan all electrodes once.  Padding shifts each
    mark-kernel entry by ``-log(σ)`` per padded dim; after segment_sum
    that becomes a per-time-bin spike-count-weighted shift, constant
    across position bins, which cancels in the HMM ``_condition_on``
    normalization but biases the absolute log-likelihood.  Grouping is
    exact and is therefore the default; the padding alternative
    (``pad_waveform_features=True``) is sketched in the optimization
    plan and not implemented here.
    """
    groups: dict[int, list[int]] = {}
    for i, enc_wf in enumerate(encoding_spike_waveform_features):
        if enc_wf.ndim != 2:
            raise ValueError(
                f"electrode {i}: encoding_spike_waveform_features must be 2-D; "
                f"got shape {enc_wf.shape}"
            )
        n_wf = int(enc_wf.shape[1])
        groups.setdefault(n_wf, []).append(i)
    return groups


def _bucket_by_size(
    electrode_indices: list[int],
    encoding_spike_waveform_features: list[jnp.ndarray],
    decoding_spike_times: list[jnp.ndarray],
    time: jnp.ndarray,
    max_buckets: int = 4,
) -> list[list[int]]:
    """Partition same-``n_wf`` electrodes into size-buckets.

    Padding all electrodes in a group to the group's max encoding-spike
    count inflates per-electrode compute proportionally to
    ``max_n_enc / mean_n_enc``.  Bucketing electrodes of similar size
    together bounds the inflation per bucket at the cost of one extra
    JIT compilation per bucket.

    Bucketing uses quantiles of ``max(n_enc, n_dec_in_bounds)`` per
    electrode.  Returns a list of electrode-index lists, each suitable
    to hand to :func:`_prepare_electrode_scan_group`.  Order is
    preserved within each bucket.

    Notes
    -----
    The effective bucket count is ``min(max_buckets, len(electrode_indices))``
    — single-element groups and groups smaller than ``max_buckets`` get
    fewer splits automatically, avoiding empty buckets.  Empty buckets
    that arise from quantile ties are also dropped from the returned
    list.
    """
    if len(electrode_indices) <= 1:
        return [electrode_indices]

    time_np = np.asarray(time)
    t0, t1 = float(time_np[0]), float(time_np[-1])
    sizes: list[int] = []
    for i in electrode_indices:
        n_enc = int(encoding_spike_waveform_features[i].shape[0])
        dec_t = np.asarray(decoding_spike_times[i])
        n_dec = int(np.sum((dec_t >= t0) & (dec_t <= t1)))
        sizes.append(max(n_enc, n_dec))

    n_buckets = min(max_buckets, len(electrode_indices))
    # Use quantile edges so each bucket has ~equal cardinality.
    qs = np.quantile(
        np.asarray(sizes, dtype=float),
        np.linspace(0.0, 1.0, n_buckets + 1)[1:-1],
    )
    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    for idx, size in zip(electrode_indices, sizes, strict=True):
        b = int(np.searchsorted(qs, size, side="right"))
        buckets[b].append(idx)
    # Drop empty buckets (can happen if all electrodes tie on a quantile).
    return [b for b in buckets if b]


def _prepare_electrode_scan_group(
    electrode_indices: list[int],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: list[jnp.ndarray],
    decoding_spike_waveform_features: list[jnp.ndarray],
    decoding_spike_times: list[jnp.ndarray],
    mean_rates: jnp.ndarray,
    time: jnp.ndarray,
    waveform_std: jnp.ndarray | float,
) -> dict:
    """Pad and stack a group of same-``n_wf`` electrodes into batched arrays.

    Prepares the per-group input to ``_predict_nonlocal_electrode_scan_jit``.
    All electrodes in ``electrode_indices`` must share the same
    ``n_wf``.  Within the group, encoding and decoding arrays are padded
    to the max count across electrodes; encoding features use
    ``mode='edge'`` (copy last real row), encoding positions use the
    ``_ELECTRODE_SCAN_POS_PAD_VALUE`` sentinel (so padded rows'
    ``log_kde_distance`` underflows to 0 downstream), decoding features
    use ``mode='edge'``, and decoding segment ids use the ``n_time``
    sentinel (ignored by ``segment_sum``).  Decoding spikes outside the
    decoding time window ``[time[0], time[-1]]`` are filtered out before
    padding.  An electrode with zero real decoding spikes has
    all-sentinel seg ids and contributes nothing to the accumulator
    regardless of feature values.

    Parameters
    ----------
    electrode_indices : list[int]
        Indices (into the outer per-electrode lists) of the electrodes
        in this group.
    encoding_spike_waveform_features : list[jnp.ndarray]
        One per-electrode array, shape ``(n_enc_e, n_wf)``.
    encoding_positions : list[jnp.ndarray]
        One per-electrode array, shape ``(n_enc_e, n_pos_dim)``.
    decoding_spike_waveform_features : list[jnp.ndarray]
        One per-electrode array, shape ``(n_dec_e, n_wf)``.
    decoding_spike_times : list[jnp.ndarray]
        One per-electrode array, shape ``(n_dec_e,)``.  Expected sorted
        ascending.
    mean_rates : jnp.ndarray, shape (n_electrodes_total,)
    time : jnp.ndarray, shape (n_time_edges,)
        Decoding-bin edges.  ``n_time = int(time.shape[0])`` is the
        ``num_segments`` value used for segment_sum.
    waveform_std : jnp.ndarray | float
        Scalar or per-feature waveform bandwidth.

    Returns
    -------
    dict with keys

    * ``enc_wf_batch`` — jnp.ndarray, shape ``(n_electrodes, max_n_enc, n_wf)``
      encoding waveform features, edge-padded.
    * ``enc_pos_batch`` — jnp.ndarray, shape ``(n_electrodes, max_n_enc, n_pos_dim)``
      encoding positions, padded with ``_ELECTRODE_SCAN_POS_PAD_VALUE``.
    * ``dec_wf_batch`` — jnp.ndarray, shape ``(n_electrodes, max_n_dec, n_wf)``
      decoding waveform features, edge-padded.
    * ``dec_seg_ids_batch`` — jnp.ndarray of int32, shape
      ``(n_electrodes, max_n_dec)``; values in ``[0, n_time)`` for real
      spikes, ``n_time`` (sentinel) for padded slots.
    * ``mean_rates_batch`` — jnp.ndarray, shape ``(n_electrodes,)``
    * ``n_real_enc_batch`` — jnp.ndarray of int32, shape
      ``(n_electrodes,)``; true encoding-spike count per electrode
      (used to compute ``log_w = -log(n_real)``).
    * ``waveform_stds`` — jnp.ndarray, shape ``(n_wf,)``
    * ``n_time`` — int, static (`len(time)` convention)
    * ``n_wf`` — int
    * ``max_n_enc``, ``max_n_dec`` — int, padding targets
    * ``n_electrodes`` — int, ``len(electrode_indices)``
    """
    # n_time matches the len(time) convention used elsewhere in this
    # module (``len(time_edges)``, one larger than the number of bins).
    # The accumulator returned by the scan has shape (n_time, n_pos) and
    # is summed with the ``-ground_process_intensity * ones((n_time, 1))``
    # baseline in predict_clusterless_kde_log_likelihood.  Valid segment
    # ids after ``np.digitize(...)`` are in ``[0, n_time)``; the sentinel
    # ``n_time`` is out of range for ``segment_sum(num_segments=n_time)``
    # and therefore ignored.
    if not electrode_indices:
        raise ValueError(
            "_prepare_electrode_scan_group requires a non-empty electrode "
            "list; the caller (predict_clusterless_kde_log_likelihood) "
            "should drop empty buckets before calling."
        )
    n_time = int(time.shape[0])
    time_np = np.asarray(time)  # hoisted: shared across electrodes

    # Per-electrode raw arrays after in-bounds filtering.
    per_enc_wf: list[np.ndarray] = []
    per_enc_pos: list[np.ndarray] = []
    per_dec_wf: list[np.ndarray] = []
    per_dec_seg_ids: list[np.ndarray] = []
    per_n_real_enc: list[int] = []
    per_n_real_dec: list[int] = []

    for i in electrode_indices:
        enc_wf_i = np.asarray(encoding_spike_waveform_features[i])
        enc_pos_i = np.asarray(encoding_positions[i])
        dec_wf_i = np.asarray(decoding_spike_waveform_features[i])
        dec_times_i = np.asarray(decoding_spike_times[i])

        in_bounds = (dec_times_i >= time_np[0]) & (dec_times_i <= time_np[-1])
        dec_times_i = dec_times_i[in_bounds]
        dec_wf_i = dec_wf_i[in_bounds]
        seg_ids_i = np.digitize(dec_times_i, time_np[1:-1]).astype(np.int32)

        per_enc_wf.append(enc_wf_i)
        per_enc_pos.append(enc_pos_i)
        per_dec_wf.append(dec_wf_i)
        per_dec_seg_ids.append(seg_ids_i)
        per_n_real_enc.append(int(enc_wf_i.shape[0]))
        per_n_real_dec.append(int(dec_wf_i.shape[0]))

    n_wf = int(per_enc_wf[0].shape[1])
    n_pos_dim = int(per_enc_pos[0].shape[1])
    # ``max_n_enc`` and ``max_n_dec`` are floored at 1 so padded shapes are
    # valid for ``dynamic_slice`` even when every electrode in the bucket
    # has zero real spikes (e.g., a silent-during-encode tetrode batch).
    # All-zero real counts produce all-sentinel seg_ids, so segment_sum
    # discards the entire bucket's contribution regardless of feature
    # values — the shape floor is for JIT tracing, not numerics.
    max_n_enc = max(per_n_real_enc + [1])
    max_n_dec = max(per_n_real_dec + [1])

    n_electrodes = len(electrode_indices)

    # Stacked, padded arrays.
    enc_wf_batch = np.zeros((n_electrodes, max_n_enc, n_wf), dtype=np.float32)
    enc_pos_batch = np.zeros((n_electrodes, max_n_enc, n_pos_dim), dtype=np.float32)
    dec_wf_batch = np.zeros((n_electrodes, max_n_dec, n_wf), dtype=np.float32)
    dec_seg_ids_batch = np.full((n_electrodes, max_n_dec), n_time, dtype=np.int32)

    for k, (enc_wf_i, enc_pos_i, dec_wf_i, seg_ids_i) in enumerate(
        zip(
            per_enc_wf,
            per_enc_pos,
            per_dec_wf,
            per_dec_seg_ids,
            strict=True,
        )
    ):
        n_enc_i = enc_wf_i.shape[0]
        n_dec_i = dec_wf_i.shape[0]

        # Edge-pad encoding features (rationale in
        # _block_estimate_log_joint_mark_intensity_impl); far-pad encoding
        # positions so padded encoding rows give hugely-negative log_pos
        # that underflow downstream exp(), zeroing their matmul contribution.
        if n_enc_i > 0:
            enc_wf_batch[k, :n_enc_i] = enc_wf_i
            if n_enc_i < max_n_enc:
                enc_wf_batch[k, n_enc_i:] = enc_wf_i[-1]
            enc_pos_batch[k, :n_enc_i] = enc_pos_i
            if n_enc_i < max_n_enc:
                enc_pos_batch[k, n_enc_i:] = _ELECTRODE_SCAN_POS_PAD_VALUE
        else:
            enc_wf_batch[k, :] = 0.0  # electrode with 0 real enc spikes
            enc_pos_batch[k, :] = _ELECTRODE_SCAN_POS_PAD_VALUE

        # Edge-pad decoding features; seg_ids already sentinel-padded to n_time.
        if n_dec_i > 0:
            dec_wf_batch[k, :n_dec_i] = dec_wf_i
            if n_dec_i < max_n_dec:
                dec_wf_batch[k, n_dec_i:] = dec_wf_i[-1]
            dec_seg_ids_batch[k, :n_dec_i] = seg_ids_i
        else:
            # Electrode with 0 real decoding spikes: leave dec_wf as zeros
            # (seg_ids are all sentinel, so segment_sum ignores this row
            # regardless of feature values).  We CANNOT edge-pad because
            # there is no last-real to copy; the sentinel seg_ids are the
            # safety net.
            pass

    # Expand waveform_std to per-feature for this group.
    if isinstance(waveform_std, (int, float)) or (
        hasattr(waveform_std, "ndim") and np.asarray(waveform_std).ndim == 0
    ):
        wf_stds = np.full(n_wf, float(np.asarray(waveform_std)), dtype=np.float32)
    else:
        wf_stds = np.asarray(waveform_std, dtype=np.float32)

    mean_rates_np = np.asarray(mean_rates, dtype=np.float32)
    mean_rates_batch = np.array(
        [mean_rates_np[i] for i in electrode_indices], dtype=np.float32
    )
    n_real_enc_batch = np.asarray(per_n_real_enc, dtype=np.int32)

    return {
        "enc_wf_batch": jnp.asarray(enc_wf_batch),
        "enc_pos_batch": jnp.asarray(enc_pos_batch),
        "dec_wf_batch": jnp.asarray(dec_wf_batch),
        "dec_seg_ids_batch": jnp.asarray(dec_seg_ids_batch),
        "mean_rates_batch": jnp.asarray(mean_rates_batch),
        "n_real_enc_batch": jnp.asarray(n_real_enc_batch),
        "waveform_stds": jnp.asarray(wf_stds),
        "n_time": n_time,
        "n_wf": n_wf,
        "max_n_enc": max_n_enc,
        "max_n_dec": max_n_dec,
        "n_electrodes": n_electrodes,
    }


def _predict_nonlocal_electrode_scan_impl(
    enc_wf_batch: jnp.ndarray,
    enc_pos_batch: jnp.ndarray,
    dec_wf_batch: jnp.ndarray,
    dec_seg_ids_batch: jnp.ndarray,
    mean_rates_batch: jnp.ndarray,
    n_real_enc_batch: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    interior_place_bin_centers: jnp.ndarray,
    position_std: jnp.ndarray,
    occupancy: jnp.ndarray,
    n_time: int,
    block_size: int,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
) -> jnp.ndarray:
    """Scan over electrodes: one iteration computes one electrode's ``(n_time, n_pos)`` contribution.

    The accumulator shape is ``(n_time, n_position_bins)``, initialized
    to zeros; the caller adds the ``-ground_process_intensity``
    baseline.  Returns the accumulated non-local log-likelihood across
    all electrodes in the group.

    Notes
    -----
    The scan carry persists for the full scan duration and scales as
    ``(n_time, n_position_bins) * 4 bytes`` — ~147 MB at the
    documented production scale, ~500 MB for 2-hour sessions.  For
    long recordings on consumer GPUs, either split into time windows
    or pass ``enc_tile_size`` / ``use_streaming=True`` to use the
    Python-loop fallback.  Per-iteration intermediates
    (``log_position_distance``, per-block mark-intensity) are
    released between iterations.  See :func:`auto_select_tile_sizes`
    for memory-aware ``block_size`` selection.
    """
    n_position_bins = occupancy.shape[0]

    def electrode_body(carry: jnp.ndarray, electrode):
        (enc_wf_e, enc_pos_e, dec_wf_e, seg_ids_e, mean_rate_e, n_real_enc_e) = (
            electrode
        )
        # Compute the position kernel inside scan (per-electrode scale)
        # rather than batching, to keep peak memory at single-electrode
        # size.  Padded encoding rows carry the
        # ``_ELECTRODE_SCAN_POS_PAD_VALUE`` sentinel so their kernel
        # rows underflow downstream — no explicit mask needed.
        log_pos_e = _log_kde_distance_impl(
            interior_place_bin_centers, enc_pos_e, position_std
        )
        contribution = _block_estimate_with_segment_sum_impl(
            dec_wf_e,
            enc_wf_e,
            waveform_stds,
            occupancy,
            mean_rate_e,
            log_pos_e,
            seg_ids_e,
            n_time,
            block_size=block_size,
            use_gemm=use_gemm,
            pos_tile_size=pos_tile_size,
            enc_tile_size=enc_tile_size,
            use_streaming=False,
            n_real_encoding_spikes=n_real_enc_e,
        )
        return carry + contribution, None

    init = jnp.zeros((n_time, n_position_bins))
    result, _ = jax.lax.scan(
        electrode_body,
        init,
        (
            enc_wf_batch,
            enc_pos_batch,
            dec_wf_batch,
            dec_seg_ids_batch,
            mean_rates_batch,
            n_real_enc_batch,
        ),
    )
    return result


_predict_nonlocal_electrode_scan_jit = jax.jit(
    _predict_nonlocal_electrode_scan_impl,
    static_argnames=(
        "n_time",
        "block_size",
        "use_gemm",
        "pos_tile_size",
        "enc_tile_size",
    ),
)


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
    # Keep waveform_std as-is (scalar or array) - will be expanded per-electrode at predict time

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

    # Count actual training samples (not the wall-clock span) so that gaps
    # introduced by is_training / encoding-group masks are not charged as
    # occupancy time.
    n_time_bins = max(len(position_time), 1)
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
        is_in_bounds = np.logical_and(
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
            mean_rates[-1]
            * jnp.where(
                occupancy > 0.0,
                gpi_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                EPS,
            ),
            min=EPS,
            max=None,
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


def _validate_block_size_argument(block_size: int | Literal["auto"]) -> None:
    """Validate ``block_size`` is a positive int or the string ``"auto"``.

    Raises ``ValueError`` on anything else.  ``bool`` is explicitly
    rejected before the int-positivity check because ``bool`` is an
    ``int`` subclass in Python — without the guard, ``True`` / ``False``
    would be silently coerced to 1 / 0.  Does NOT resolve ``"auto"``
    (no dependency on workload dims); see :func:`_resolve_block_size`
    for that.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int | str):
        raise ValueError(
            f"block_size must be a positive int or the string 'auto'; "
            f"got {type(block_size).__name__} {block_size!r}."
        )
    if isinstance(block_size, int):
        if block_size < 1:
            raise ValueError(
                f"block_size must be ≥ 1 when passed as an int; got {block_size}."
            )
        return
    if block_size != "auto":
        raise ValueError(
            f"block_size string must be 'auto' (got {block_size!r}); "
            f"pass an int for a fixed block size."
        )


def _resolve_block_size(
    block_size: int | Literal["auto"],
    *,
    n_enc: int,
    n_dec: int,
    n_pos: int,
    n_wf: int,
) -> int:
    """Resolve ``block_size`` to a concrete int, expanding ``"auto"``.

    Validates first (via :func:`_validate_block_size_argument`), then
    either returns a fixed int unchanged (the common case — no-op for
    the default ``block_size=100``) or calls
    :func:`auto_select_tile_sizes` to resolve ``"auto"`` against the
    current memory budget.  Queries ``jax.devices()[0].memory_stats()``
    only on the auto branch.
    """
    _validate_block_size_argument(block_size)
    if isinstance(block_size, int):
        return block_size
    # At this point block_size == "auto" (validator rejected everything else).
    return int(
        auto_select_tile_sizes(n_enc=n_enc, n_dec=n_dec, n_pos=n_pos, n_wf=n_wf)[
            "block_size"
        ]
    )


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
    block_size: int | Literal["auto"] = 100,
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
    block_size : int | Literal["auto"], optional
        Decoding-spike block size for the fori_loop inside each scan
        iteration (or fallback-path call).  Default 100.  When
        ``"auto"``, :func:`auto_select_tile_sizes` picks a
        memory-aware value per bucket (scan path) or per electrode
        (fallback path).  Each distinct resolved value is a static
        JIT argument and triggers a recompile, so prefer a fixed int
        when running many shape-different workloads in one process.
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
        # Validate the argument's type/range even on the local path so
        # invalid values (e.g., 0, -5, True, 1.5, None) fail loudly at
        # the public API boundary rather than producing a confusing
        # downstream error inside compute_local_log_likelihood.
        _validate_block_size_argument(block_size)
        if isinstance(block_size, str):
            # ``auto`` resolution for the local path is out of scope for
            # this optimization branch (the local likelihood uses
            # block_size only inside KDEModel.predict batching, not in
            # the Task 1 fori_loop path this PR added ``auto`` for).
            # Pass an int or omit the argument for is_local=True.
            raise ValueError(
                f"block_size={block_size!r} is not supported for is_local=True; "
                f"pass an explicit int (default 100) instead."
            )
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

        # Scan-based path: group electrodes by waveform-feature count and run
        # a JIT-compiled ``jax.lax.scan`` per group, eliminating the Python
        # dispatch round-trip between electrodes.  Streaming and encoding-
        # chunking paths use on-the-fly position kernel / chunked logsumexp
        # code that isn't plumbed through the scan body yet — those modes
        # fall back to the per-electrode Python loop (same as Task 4).
        use_scan_path = not use_streaming and enc_tile_size is None
        if use_scan_path:
            # Group electrodes by (n_wf, size-bucket).  Same n_wf is required
            # for the mark kernel's per-feature sigma broadcast; size-bucketing
            # within a feature group bounds the worst-case padding overhead
            # (pad-to-max within a bucket, not across all electrodes).
            groups = _group_electrodes_by_n_wf(encoding_spike_waveform_features)
            scan_batches: list[list[int]] = []
            for _n_wf, electrode_indices in sorted(
                groups.items(), key=lambda kv: kv[0]
            ):
                scan_batches.extend(
                    _bucket_by_size(
                        electrode_indices,
                        encoding_spike_waveform_features,
                        spike_times,
                        time,
                    )
                )
            for batch_indices in tqdm(
                scan_batches,
                unit="batch",
                desc="Non-Local Likelihood (scan)",
                disable=disable_progress_bar,
            ):
                batch = _prepare_electrode_scan_group(
                    batch_indices,
                    encoding_spike_waveform_features,
                    encoding_positions,
                    spike_waveform_features,
                    spike_times,
                    mean_rates,
                    time,
                    waveform_std,
                )
                # No-op for fixed-int ``block_size``; resolves "auto"
                # per-bucket using each bucket's max shapes.
                resolved_block_size = _resolve_block_size(
                    block_size,
                    n_enc=batch["max_n_enc"],
                    n_dec=batch["max_n_dec"],
                    n_pos=occupancy.shape[0],
                    n_wf=batch["n_wf"],
                )
                log_likelihood = log_likelihood + _predict_nonlocal_electrode_scan_jit(
                    batch["enc_wf_batch"],
                    batch["enc_pos_batch"],
                    batch["dec_wf_batch"],
                    batch["dec_seg_ids_batch"],
                    batch["mean_rates_batch"],
                    batch["n_real_enc_batch"],
                    batch["waveform_stds"],
                    interior_place_bin_centers,
                    position_std,
                    occupancy,
                    n_time=batch["n_time"],
                    block_size=resolved_block_size,
                    pos_tile_size=pos_tile_size,
                    enc_tile_size=None,
                )
        else:
            # Fallback Python loop for streaming / enc-chunking modes.
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

                # Expand waveform_std to match this electrode's feature count if scalar
                n_waveform_features = electrode_encoding_spike_waveform_features.shape[
                    1
                ]
                if isinstance(waveform_std, (int, float)) or (
                    hasattr(waveform_std, "ndim") and waveform_std.ndim == 0
                ):
                    electrode_waveform_std = jnp.full(n_waveform_features, waveform_std)
                else:
                    electrode_waveform_std = waveform_std

                # No-op for fixed-int ``block_size``; resolves "auto"
                # per-electrode using post-in-bounds-filter counts.
                resolved_block_size = _resolve_block_size(
                    block_size,
                    n_enc=int(electrode_encoding_spike_waveform_features.shape[0]),
                    n_dec=int(electrode_decoding_spike_waveform_features.shape[0]),
                    n_pos=occupancy.shape[0],
                    n_wf=n_waveform_features,
                )
                # NOTE: n_time is a static argument to the fused function — each
                # distinct value triggers a JIT recompilation.
                log_likelihood += (
                    block_estimate_with_segment_sum_log_joint_mark_intensity(
                        electrode_decoding_spike_waveform_features,
                        electrode_encoding_spike_waveform_features,
                        electrode_waveform_std,
                        occupancy,
                        electrode_mean_rate,
                        log_position_distance,
                        get_spike_time_bin_ind(electrode_spike_times, time),
                        n_time,
                        block_size=resolved_block_size,
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
                    )
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

        # Expand waveform_std to match this electrode's feature count if scalar
        n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]
        if isinstance(waveform_std, (int, float)) or (
            hasattr(waveform_std, "ndim") and waveform_std.ndim == 0
        ):
            electrode_waveform_std = jnp.full(n_waveform_features, waveform_std)
        else:
            electrode_waveform_std = waveform_std

        # Compute marginal density in log-space for numerical stability
        log_marginal_density = block_log_kde(
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
            std=jnp.concatenate((position_std, electrode_waveform_std)),
            block_size=block_size,
        )

        # Compute spike contribution in log-space:
        # log(rate * density / occupancy) = log(rate) + log(density) - log(occupancy)
        log_mean_rate = safe_log(electrode_mean_rate, eps=EPS)
        log_occupancy = safe_log(occupancy_at_spike_time, eps=EPS)

        # Spike contribution: sum over spikes in each time bin
        spike_contribution = log_mean_rate + log_marginal_density - log_occupancy

        log_likelihood += jax.ops.segment_sum(
            spike_contribution,
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * jnp.where(
            occupancy > 0.0,
            electrode_gpi_model.predict(interpolated_position)
            / jnp.where(occupancy > 0.0, occupancy, 1.0),
            0.0,
        )
    return log_likelihood[:, jnp.newaxis]
