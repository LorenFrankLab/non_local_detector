"""Clusterless KDE likelihood.

The marked intensity is a Gaussian KDE over position and waveform features divided
by the occupancy density, evaluated in log space. Each spike's term is
``logaddexp(log intensity, log background)`` with a background
``RATE_EPS_HZ * q(mark)`` (see ``mark_reference``), so every intensity that can
change that term must be exact even far below float32's probability range. A
scaled GEMM keeps the reduction fast; spikes whose relevant range it cannot cover
are recomputed with an exact logsumexp.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

from non_local_detector._jax_typing import finfo
from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
    warn_if_tracking_is_sparse,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_spikes import (
    ElectrodeSpikes,
    add_electrode_spike_terms,
)
from non_local_detector.likelihoods.common import (
    _KDE_BLOCK_KERNEL_BYTES,
    _KDE_SAMPLE_TILE,
    DEFAULT_WAVEFORM_STD,
    EPS,
    EXACT_REPAIR_SLOTS,
    LOG_EPS,
    RATE_REFERENCE_SECONDS,
    DecodeRows,
    KDEModel,
    SpikeTimeOrder,
    _log_kernel_matrix,
    as_array,
    as_arrays,
    as_std_array,
    compute_log_mark_kernel_gemm,
    deterministic_row_accumulate,
    drop_zero_weight_samples,
    environment_grid_fingerprint,
    get_position_at_time,
    kde_log_densities,
    kde_log_density,
    kde_model_with_block_size,
    log_bin_duration_evidence,
    log_density_intensity,
    log_gaussian_pdf,
    log_nonnegative,
    masked_logaddexp,
    masked_logsumexp,
    pad_rows,
    padded_block_count,
    padded_occupancy_count,
    padded_samples,
    padded_spike_count,
    padded_tile_count,
    place_encoding_spikes,
    prepare_decode_rows,
    prepare_kde_local_inputs,
    prepared_kde_log_densities,
    require_position,
    select_decode_marks,
    select_spike_rows,
    select_spikes_in_rows,
    snap_off_track_positions,
    spike_row_ids,
    sum_spikes_into_rows,
    true_kde_log_density,
    validate_bandwidth,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_positive_int,
    weighted_mean_rate,
    zero_if_neginf,
)
from non_local_detector.likelihoods.common.kernel_density import (
    _bucketed_tiles,
    _sample_tiles,
    _sum_over_tiles,
)
from non_local_detector.likelihoods.encoding_model import ClusterlessEncodingModel
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
    background_rate,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    requires_time_edges,
)

__all__ = [
    "ClusterlessKDEEncodingModel",
    "fit_clusterless_kde_encoding_model",
    "predict_clusterless_kde_log_likelihood",
]


# Encoding rows per reduction tile. Each tile is one GEMM over the whole query
# block, so temporaries scale with the tile rather than the encoding count.
_LOG_REDUCTION_ENCODING_ROWS = 2048
# Target elements of one exact-repair temporary (encoding tile x flagged spikes
# x position tile), 16 MiB in float32. It sets how many flagged spikes are
# reduced together, at least one, so larger tiles exceed it.
_EXACT_REPAIR_ELEMENTS = 1 << 22


def _repair_flagged_in_groups(
    values: jax.Array,
    needs_exact: jax.Array,
    exact_rows: Callable[[jax.Array], jax.Array],
    group_size: int,
) -> jax.Array:
    """Replace each flagged row of ``values`` by ``exact_rows`` of it.

    The indices of flagged rows are listed in order and split into groups of
    ``group_size``, each recomputed by one ``exact_rows`` call. No repair runs
    when no row is flagged. When the flagged rows fit in the groups covering
    ``EXACT_REPAIR_SLOTS`` slots, only those groups are visited; otherwise
    every group is visited, and groups past the flagged count are skipped.
    Static loop bounds keep the repair differentiable.

    Parameters
    ----------
    values : jax.Array, shape (n, ...)
        Estimates whose flagged rows are replaced.
    needs_exact : jax.Array of bool, shape (n,)
        Rows to replace.
    exact_rows : callable
        ``exact_rows(rows) -> (group_size, ...)`` exact values of the given rows.
        The last group's unused slots pass row 0, whose values are dropped.
    group_size : int
        Flagged rows recomputed per call (capped at ``n``).

    Returns
    -------
    values : jax.Array, shape (n, ...)
    """
    n = needs_exact.shape[0]
    group_size = min(n, group_size)
    n_groups = -(-n // group_size)
    n_flagged = jnp.sum(needs_exact)
    flagged = jnp.nonzero(needs_exact, size=n_groups * group_size, fill_value=0)[0]

    def repair_group(group: jax.Array, current: jax.Array) -> jax.Array:
        start = group * group_size
        rows = jax.lax.dynamic_slice_in_dim(flagged, start, group_size)
        is_flagged = start + jnp.arange(group_size) < n_flagged
        # Unflagged slots write past the end, where the scatter drops them.
        targets = jnp.where(is_flagged, rows, n)
        return cast(
            jax.Array,
            jax.lax.cond(
                start < n_flagged,
                lambda current: current.at[targets].set(exact_rows(rows), mode="drop"),
                lambda current: current,
                current,
            ),
        )

    def repair(n_visits: int, current: jax.Array) -> jax.Array:
        return cast(jax.Array, jax.lax.fori_loop(0, n_visits, repair_group, current))

    n_slot_groups = min(n_groups, -(-EXACT_REPAIR_SLOTS // group_size))
    return cast(
        jax.Array,
        jax.lax.cond(
            n_flagged == 0,
            lambda current: current,
            lambda current: jax.lax.cond(
                n_flagged <= n_slot_groups * group_size,
                partial(repair, n_slot_groups),
                partial(repair, n_groups),
                current,
            ),
            values,
        ),
    )


def _log_marginal_floor(
    mean_rate: float | jax.Array, occupancy: jax.Array
) -> jax.Array:
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
    mean_rate : float or jax.Array, scalar
        Electrode's mean rate.
    occupancy : jax.Array, shape (n_position_bins,)
        Occupancy density at each position bin.

    Returns
    -------
    position_floor : jax.Array, shape (n_position_bins,)
    """
    log_occupancy = jnp.log(occupancy)
    return -log_density_intensity(
        mean_rate, jnp.zeros_like(log_occupancy), log_occupancy
    )


def _scaled_gemm_shift(n_rows: int, dtype: DTypeLike) -> float:
    """Exponent shift that keeps an ``n_rows``-term scaled sum below 1 / (2 tiny).

    The cap keeps both the sum and its reciprocal, the derivative of its
    logarithm, normal numbers.
    """
    dtype_info = finfo(dtype)
    return -0.5 * float(np.log(2.0 * n_rows * dtype_info.tiny))


def _scaled_gemm_window(n_rows: int, dtype: DTypeLike) -> float:
    """Nats below its factor maxima within which a scaled-GEMM entry is exact.

    A sum of ``n_rows`` factors, each scaled by its maximum and shifted up by
    ``_scaled_gemm_shift``, loses at most ``n_rows * tiny * exp(shift)`` per
    entry in scaled units. An entry whose ``mark_max + position_max``
    exceeds its floor by no more than this window therefore omits less than
    four dtype epsilons of any value at or above the floor: about 104.7 nats
    for 2048 float32 rows.
    """
    dtype_info = finfo(dtype)
    return float(
        np.log(4 * dtype_info.eps)
        - np.log(n_rows * dtype_info.tiny)
        + _scaled_gemm_shift(n_rows, dtype)
    )


def _log_joint_from_log_marginal(
    log_marginal: jax.Array, mean_rate: float | jax.Array, occupancy: jax.Array
) -> jax.Array:
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
    log_marginal : jax.Array, shape (n_rows, n_position_bins)
        Log marginal mark/position density. ``n_rows`` is the number of decoding
        spikes for the non-local paths, or ``1`` for the per-spike local path (which
        passes its ``(n_spikes,)`` marginal as a single row so the per-bin broadcast
        below becomes a per-spike elementwise combine).
    mean_rate : float or jax.Array, scalar
        Mean firing rate for this electrode.
    occupancy : jax.Array, shape (n_position_bins,)
        Occupancy density at the position bins (per decoding spike for the local
        path).

    Returns
    -------
    log_joint : jax.Array, shape (n_rows, n_position_bins)
    """
    log_occ = jnp.log(occupancy)
    log_joint = log_density_intensity(mean_rate, log_marginal, log_occ[None, :])
    # Zero occupancy (no support) has zero intensity; NaN stays visible.
    unsupported = (occupancy[None, :] <= 0.0) & ~jnp.isnan(log_joint)
    return jnp.where(unsupported, -jnp.inf, log_joint)


@jax.jit
def log_kde_distance(
    eval_points: jax.Array, samples: jax.Array, std: jax.Array
) -> jax.Array:
    """Vectorized log-distance (log kernel product) using vmap.

    Computes:
        log_distance[i, j] = sum_d log N(eval_points[j, d] | samples[i, d], std[d])

    Uses jax.vmap to eliminate Python for-loop over dimensions, enabling full parallelization.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jax.Array, shape (n_samples, n_dims)
        Training samples.
    std : jax.Array, shape (n_dims,)
        Per-dimension kernel std.

    Returns
    -------
    log_distance : jax.Array, shape (n_samples, n_eval_points)
        Log of the product of per-dimension Gaussian kernels.

    Notes
    -----
    ``std`` is clamped to ``[EPS, inf)`` so a zero bandwidth yields a finite
    (heavily peaked) kernel instead of NaN, matching
    ``log_kde_distance_streaming`` at the edges.
    """
    # Clamp std to avoid division by zero (mirrors log_kde_distance_streaming).
    std = jnp.clip(std, EPS, jnp.inf)

    def log_gaussian_per_dim(
        eval_dim: jax.Array, sample_dim: jax.Array, sigma: jax.Array
    ) -> jax.Array:
        """Log Gaussian kernel of one dimension, shape (n_samples, n_eval)."""
        return cast(
            jax.Array,
            log_gaussian_pdf(
                eval_dim[None, :],  # shape (1, n_eval)
                sample_dim[:, None],  # shape (n_samples, 1)
                sigma,
            ),
        )

    # vmap over dimensions: produces (n_dims, n_samples, n_eval)
    per_dim_log_distances = jax.vmap(log_gaussian_per_dim)(
        eval_points.T, samples.T, std
    )

    # Sum over dimensions: (n_samples, n_eval)
    return jnp.sum(per_dim_log_distances, axis=0)


@jax.jit
def log_kde_distance_streaming(
    eval_points: jax.Array,
    samples: jax.Array,
    std: jax.Array,
) -> jax.Array:
    """Compute log KDE distance in streaming fashion to avoid D×n_samp×n_eval intermediate.

    This is mathematically equivalent to log_kde_distance but uses a fori_loop over
    dimensions instead of vmap. This avoids materializing a (n_dims, n_samples, n_eval)
    intermediate array, reducing peak memory from O(D×n_samp×n_eval) to O(n_samp×n_eval).

    For large D (many position dimensions), this can significantly reduce memory usage.

    Parameters
    ----------
    eval_points : jax.Array, shape (n_eval, n_dims)
        Points at which to evaluate the KDE.
    samples : jax.Array, shape (n_samples, n_dims)
        Sample points from which to build the KDE.
    std : jax.Array, shape (n_dims,)
        Standard deviation for each dimension.

    Returns
    -------
    log_distance : jax.Array, shape (n_samples, n_eval)
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

    def accumulate_dim(dim_idx: int, acc: jax.Array) -> jax.Array:
        """Accumulate the log-Gaussian contribution of one dimension.

        Parameters
        ----------
        dim_idx : int
            Index of the dimension to add.
        acc : jax.Array, shape (n_samp, n_eval)
            Running sum of the log-Gaussian kernels of the earlier dimensions.

        Returns
        -------
        acc : jax.Array, shape (n_samp, n_eval)
            ``acc`` plus the log-Gaussian kernel of dimension ``dim_idx``.
        """
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
        return cast(jax.Array, acc + logp_d)

    # Loop over dimensions, accumulating log-distance contributions
    return cast(
        jax.Array,
        jax.lax.fori_loop(0, n_dims, accumulate_dim, log_distance_acc),
    )


def _encoding_log_weights(
    encoding_weights: jax.Array | None, n_encoding_spikes: int
) -> jax.Array:
    """Per-encoding-spike log weight ``log(w_e / sum w)``.

    A zero weight gives ``-inf``, dropping the spike out of every maximum and
    product; a negative or NaN weight gives NaN. ``None`` weights uniformly.

    Parameters
    ----------
    encoding_weights : jax.Array, shape (n_encoding_spikes,), or None
        Encoding spike weights; None weights all spikes equally.
    n_encoding_spikes : int
        Number of encoding spikes.

    Returns
    -------
    log_w : jax.Array, shape (n_encoding_spikes,)
    """
    if encoding_weights is None:
        # Use max(n, 1) to avoid log(0); when n=0 the result is unused.
        safe_n = jnp.where(n_encoding_spikes > 0, float(n_encoding_spikes), 1.0)
        return jnp.full((n_encoding_spikes,), -jnp.log(safe_n))
    return _log_weights_given_total(encoding_weights, jnp.sum(encoding_weights))


def _log_weights_given_total(
    encoding_weights: jax.Array, weight_total: jax.Array
) -> jax.Array:
    """``log(w_e / weight_total)`` with the semantics of ``_encoding_log_weights``.

    Parameters
    ----------
    encoding_weights : jax.Array, shape (n_encoding_spikes,)
        Encoding spike weights.
    weight_total : jax.Array, scalar
        The sum of every weight of the encoding model, which may hold more
        samples than ``encoding_weights``.

    Returns
    -------
    log_w : jax.Array, shape (n_encoding_spikes,)
    """
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


def _occupancy_normalizer(log_occupancy: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Occupancy to divide position kernels by before mark log densities are added.

    Dividing first preserves small conditional mark terms even in very distant
    tails.

    Parameters
    ----------
    log_occupancy : jax.Array, shape (n_position_bins,)
        Log occupancy density at each position bin.

    Returns
    -------
    normalizer : jax.Array, shape (n_position_bins,)
        Log occupancy to subtract from position kernels; 0 where it is zero.
    occupancy : jax.Array, shape (n_position_bins,)
        Occupancy left after the division: 1 where supported, 0 where it is
        zero, NaN where it is NaN.
    """
    normalizer = zero_if_neginf(log_occupancy)
    occupancy = jnp.where(
        jnp.isneginf(log_occupancy),
        0.0,
        jnp.where(jnp.isnan(log_occupancy), jnp.nan, 1.0),
    )
    return normalizer, occupancy


def _tiled_position_peak(
    log_position_distance: jax.Array,
    log_w: jax.Array,
    enc_tile_size: int,
    dtype: DTypeLike,
) -> jax.Array:
    """Largest weighted position kernel per bin, reduced over encoding tiles.

    No encoding x position temporary is formed. The overlapping final tile only
    repeats rows, which a max ignores.
    """
    n_enc, n_pos = log_position_distance.shape

    def tile_peak(enc_idx: jax.Array, peak: jax.Array) -> jax.Array:
        enc_start = jnp.minimum(enc_idx * enc_tile_size, n_enc - enc_tile_size)
        tile = (
            jax.lax.dynamic_slice_in_dim(
                log_position_distance, enc_start, enc_tile_size
            )
            + jax.lax.dynamic_slice_in_dim(log_w, enc_start, enc_tile_size)[:, None]
        )
        return jnp.maximum(peak, jnp.max(tile, axis=0))

    n_chunks = (n_enc + enc_tile_size - 1) // enc_tile_size
    return cast(
        jax.Array,
        jax.lax.fori_loop(0, n_chunks, tile_peak, jnp.full((n_pos,), -jnp.inf, dtype)),
    )


def _scaled_factor(
    log_values: jax.Array, safe_peak: jax.Array, shift: Any
) -> jax.Array:
    """``exp(log_values - safe_peak + shift)``: a factor of the scaled GEMM.

    Mark factors are scaled by the mark kernel's peak and position factors
    (weighted log position kernels) by each bin's peak, both shifted up by
    ``_scaled_gemm_shift``.
    """
    return jnp.exp(log_values - safe_peak + shift)


def _linear_log_marginal(
    numerator: jax.Array,
    mark_peak: jax.Array,
    safe_position_peak: jax.Array,
    shift: Any,
) -> jax.Array:
    """Log marginal from a scaled GEMM numerator, undoing its scales.

    The fixed peaks cannot carry a NaN from the marks; the product's is kept.
    """
    return cast(
        jax.Array,
        log_nonnegative(numerator)
        + zero_if_neginf(mark_peak)
        + safe_position_peak[None, :]
        - 2.0 * shift,
    )


def _exact_threshold(
    mark_peak: jax.Array,
    position_peak: jax.Array,
    window: Any,
    position_floor: jax.Array,
) -> jax.Array:
    """Return the query floor above which a spike's scaled estimate may lose mass.

    A spike's linear estimate is exact above its floor when every kernel peak
    lies within the scaled GEMM's window of that floor; a spike whose floor
    is below this threshold is recomputed exactly.
    """
    return jnp.max(mark_peak + position_peak - window - position_floor)


def _repair_group_size(encoding_rows: int, pos_tile_size: int) -> int:
    """Return the flagged spikes per exact-repair call, which bounds its temporary."""
    return max(
        1,
        min(
            EXACT_REPAIR_SLOTS,
            _EXACT_REPAIR_ELEMENTS // (encoding_rows * pos_tile_size),
        ),
    )


def _fold_position_tiles(
    update: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array],
    n_bins: int,
    pos_tile_size: int,
    accumulated: jax.Array,
) -> jax.Array:
    """Fold ``update(accumulated, pos_start, is_new)`` over tiles of place bins.

    A final tile that would run past the last bin starts earlier and repeats
    bins of the previous tile; ``is_new`` marks the bins it adds. With one
    tile ``is_new`` is None.
    """
    if pos_tile_size >= n_bins:
        return update(accumulated, jnp.asarray(0), None)

    def position_chunk(pos_idx: jax.Array, accumulated: jax.Array) -> jax.Array:
        requested = pos_idx * pos_tile_size
        pos_start = jnp.minimum(requested, n_bins - pos_tile_size)
        is_new = pos_start + jnp.arange(pos_tile_size) >= requested
        return update(accumulated, pos_start, is_new)

    return cast(
        jax.Array,
        jax.lax.fori_loop(0, -(-n_bins // pos_tile_size), position_chunk, accumulated),
    )


def _combine_tile_columns(
    accumulated: jax.Array,
    values: jax.Array,
    pos_start: jax.Array,
    is_new: jax.Array | None,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """``combine`` a tile's ``values`` into its new columns of ``accumulated``."""
    if is_new is None:
        return combine(accumulated, values)
    width = values.shape[1]
    current = jax.lax.dynamic_slice_in_dim(accumulated, pos_start, width, axis=1)
    updated = jnp.where(is_new[None, :], combine(current, values), current)
    return jax.lax.dynamic_update_slice(
        accumulated, updated, (jnp.zeros((), pos_start.dtype), pos_start)
    )


def _add_scaled_tile(
    numerator: jax.Array,
    queries: jax.Array,
    encoding_marks: jax.Array,
    waveform_stds: jax.Array,
    mark_peak: jax.Array,
    shift: Any,
    position_factor: Callable[[jax.Array], jax.Array],
    pos_tile_size: int,
    feature_mask: jax.Array | None = None,
) -> jax.Array:
    """Add one encoding tile's scaled GEMM to the numerator, by position tiles.

    Parameters
    ----------
    numerator : jax.Array, shape (n_queries, n_bins)
        Running scaled sums.
    queries : jax.Array, shape (n_queries, n_features)
        Decoding marks.
    encoding_marks : jax.Array, shape (n_samples, n_features)
        The tile's encoding marks.
    waveform_stds : jax.Array, shape (n_features,)
        Mark bandwidths.
    mark_peak : jax.Array, scalar
        The mark kernel's peak.
    shift : float or jax.Array
        ``_scaled_gemm_shift`` of the encoding model.
    position_factor : callable
        ``position_factor(pos_start)`` gives the tile's scaled position
        factors at ``pos_tile_size`` bins from ``pos_start``, shape
        ``(n_samples, pos_tile_size)``.
    pos_tile_size : int
        Place bins per tile.
    feature_mask : jax.Array of bool, shape (n_features,), optional
        See ``compute_log_mark_kernel_gemm``.

    Returns
    -------
    numerator : jax.Array, shape (n_queries, n_bins)
    """
    # Zero-weight rows have zero position factors; NaN marks propagate.
    mark_factor = _scaled_factor(
        compute_log_mark_kernel_gemm(
            queries, encoding_marks, waveform_stds, feature_mask
        ),
        zero_if_neginf(mark_peak),
        shift,
    )

    def update(
        numerator: jax.Array, pos_start: jax.Array, is_new: jax.Array | None
    ) -> jax.Array:
        product = jnp.matmul(
            mark_factor.T,
            position_factor(pos_start),
            precision=jax.lax.Precision.HIGHEST,
        )
        return _combine_tile_columns(numerator, product, pos_start, is_new, jnp.add)

    return _fold_position_tiles(update, numerator.shape[1], pos_tile_size, numerator)


def _add_exact_tile(
    marginal: jax.Array,
    queries: jax.Array,
    encoding_marks: jax.Array,
    waveform_stds: jax.Array,
    log_w: jax.Array,
    log_position: Callable[[jax.Array], jax.Array],
    pos_tile_size: int,
    feature_mask: jax.Array | None = None,
) -> jax.Array:
    """Add one encoding tile to an exact log marginal by a direct logsumexp.

    ``log_position(pos_start)`` gives the tile's log position kernels at
    ``pos_tile_size`` bins, shape ``(n_samples, pos_tile_size)``; ``log_w``
    is each sample's log weight, ``-inf`` for samples to leave out. A group
    of queries shares one (samples, queries, bins) temporary.
    """
    log_mark = compute_log_mark_kernel_gemm(
        queries, encoding_marks, waveform_stds, feature_mask
    )

    def update(
        marginal: jax.Array, pos_start: jax.Array, is_new: jax.Array | None
    ) -> jax.Array:
        contribution = masked_logsumexp(
            log_mark[:, :, None]
            + log_position(pos_start)[:, None, :]
            + log_w[:, None, None]
        )
        return _combine_tile_columns(
            marginal, contribution, pos_start, is_new, masked_logaddexp
        )

    return _fold_position_tiles(update, marginal.shape[1], pos_tile_size, marginal)


def _estimate_with_enc_chunking(
    decoding_spike_waveform_features: jax.Array,
    encoding_spike_waveform_features: jax.Array,
    waveform_stds: jax.Array,
    occupancy: jax.Array,
    mean_rate: float,
    log_position_distance: jax.Array | None,
    log_w: jax.Array,
    enc_tile_size: int,
    pos_tile_size: int | None,
    log_background: jax.Array,
    encoding_positions: jax.Array | None = None,
    position_eval_points: jax.Array | None = None,
    position_std: jax.Array | None = None,
    *,
    use_streaming: bool = False,
    position_log_normalizer: jax.Array | None = None,
) -> jax.Array:
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
    log_joint : jax.Array, shape (n_dec, n_pos)
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
            msg = "use_streaming=True requires encoding_positions, position_eval_points, and position_std"
            raise ValueError(msg)
        n_pos = position_eval_points.shape[0]
    else:
        if log_position_distance is None:
            msg = "use_streaming=False requires log_position_distance"
            raise ValueError(msg)
        n_pos = log_position_distance.shape[1]
    enc_tile_size = min(n_enc, enc_tile_size, _LOG_REDUCTION_ENCODING_ROWS)
    pos_tile_size = min(n_pos, pos_tile_size or n_pos)
    n_enc_chunks = (n_enc + enc_tile_size - 1) // enc_tile_size
    # Tile kernels promote every input with float32; explicitly typed carries
    # keep the loops' dtypes fixed when x64 is enabled.
    position_inputs: tuple[jax.Array | None, ...] = (
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
    query_floor = jnp.asarray(log_background, dtype) + np.log(finfo(dtype).eps)
    position_floor = _log_marginal_floor(mean_rate, occupancy)

    # Every log kernel is at most its peak: the Gaussian normalizer for marks
    # (zero distance) and, per position, the largest weighted position kernel.
    # Scaling by these peaks loses under exp(peak - window) per entry, so if
    # that stays below four epsilons of the floor everywhere, no entry above
    # the floor loses mass. float32 is the narrowest GEMM dtype, so its window
    # is the conservative one.
    n_features = encoding_spike_waveform_features.shape[1]
    origin = jnp.zeros((1, n_features), dtype)
    mark_peak = compute_log_mark_kernel_gemm(origin, origin, waveform_stds)[0, 0]
    if use_streaming:
        position_origin = jnp.zeros(
            (1, cast(jax.Array, encoding_positions).shape[1]), dtype
        )
        position_peak = log_kde_distance_streaming(
            position_origin, position_origin, position_std
        )[0, 0].astype(dtype) + jnp.max(log_w)
        if position_log_normalizer is not None:
            position_peak = position_peak - position_log_normalizer.astype(dtype)
    else:
        position_peak = _tiled_position_peak(
            cast(jax.Array, log_position_distance), log_w, enc_tile_size, dtype
        )
    needs_exact_row = (
        _exact_threshold(
            mark_peak,
            position_peak,
            _scaled_gemm_window(n_enc, jnp.float32),
            position_floor,
        )
        > query_floor
    )

    def encoding_tile(
        enc_idx: jax.Array | int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        requested_enc_start = enc_idx * enc_tile_size
        enc_start = jnp.minimum(requested_enc_start, n_enc - enc_tile_size)
        marks = jax.lax.dynamic_slice_in_dim(
            encoding_spike_waveform_features, enc_start, enc_tile_size
        )
        weights = jax.lax.dynamic_slice_in_dim(log_w, enc_start, enc_tile_size)
        # Slice an overlapping final tile and mask its prefix, avoiding a
        # padded copy of the potentially very large precomputed matrix.
        weights = jnp.where(
            enc_start + jnp.arange(enc_tile_size) >= requested_enc_start,
            weights,
            -jnp.inf,
        )
        positions = (
            jax.lax.dynamic_slice_in_dim(
                cast(jax.Array, encoding_positions), enc_start, enc_tile_size
            )
            if use_streaming
            else None
        )
        return enc_start, marks, weights, positions

    def log_position_at(
        pos_start: jax.Array, enc_start: jax.Array, positions: jax.Array | None
    ) -> jax.Array:
        if use_streaming:
            eval_points = jax.lax.dynamic_slice_in_dim(
                cast(jax.Array, position_eval_points), pos_start, pos_tile_size
            )
            # Its accumulator has the default float type; keep the tile dtype.
            log_position = log_kde_distance_streaming(
                eval_points, cast(jax.Array, positions), cast(jax.Array, position_std)
            ).astype(dtype)
            if position_log_normalizer is not None:
                log_position = (
                    log_position
                    - jax.lax.dynamic_slice_in_dim(
                        position_log_normalizer.astype(dtype), pos_start, pos_tile_size
                    )[None, :]
                )
            return cast(jax.Array, log_position)
        return jax.lax.dynamic_slice(
            cast(jax.Array, log_position_distance),
            (enc_start, pos_start),
            (enc_tile_size, pos_tile_size),
        )

    # Every tile shares the kernel-free peaks as fixed scales, so their
    # products add linearly. The scales depend on no decoding spike. The
    # float32 shift matches the gate's window and cannot overflow any wider
    # dtype the operands may use.
    shift = _scaled_gemm_shift(n_enc, jnp.float32)
    safe_position_peak = jnp.broadcast_to(zero_if_neginf(position_peak), (n_pos,))

    def reduce_linearly() -> jax.Array:
        def process_enc_chunk(enc_idx: jax.Array, numerator: jax.Array) -> jax.Array:
            enc_start, marks, weights, positions = encoding_tile(enc_idx)

            def position_factor(pos_start: jax.Array) -> jax.Array:
                return _scaled_factor(
                    log_position_at(pos_start, enc_start, positions) + weights[:, None],
                    jax.lax.dynamic_slice_in_dim(
                        safe_position_peak, pos_start, pos_tile_size
                    ),
                    shift,
                )

            return _add_scaled_tile(
                numerator,
                decoding_spike_waveform_features,
                marks,
                waveform_stds,
                mark_peak,
                shift,
                position_factor,
                pos_tile_size,
            )

        numerator = jax.lax.fori_loop(
            0, n_enc_chunks, process_enc_chunk, jnp.zeros((n_dec, n_pos), dtype)
        )
        return _linear_log_marginal(numerator, mark_peak, safe_position_peak, shift)

    def exact_rows(rows: jax.Array) -> jax.Array:
        # Each flagged spike is reduced by a direct logsumexp over encoding
        # rows, with no scaling to validate.
        queries = decoding_spike_waveform_features[rows]

        def process_enc_chunk(enc_idx: jax.Array, marginal: jax.Array) -> jax.Array:
            enc_start, marks, weights, positions = encoding_tile(enc_idx)
            return _add_exact_tile(
                marginal,
                queries,
                marks,
                waveform_stds,
                weights,
                lambda pos_start: log_position_at(pos_start, enc_start, positions),
                pos_tile_size,
            )

        return cast(
            jax.Array,
            jax.lax.fori_loop(
                0,
                n_enc_chunks,
                process_enc_chunk,
                jnp.full((rows.shape[0], n_pos), -jnp.inf, dtype),
            ),
        )

    marginal = _repair_flagged_in_groups(
        reduce_linearly(),
        needs_exact_row,
        exact_rows,
        _repair_group_size(enc_tile_size, pos_tile_size),
    )
    return _log_joint_from_log_marginal(marginal, mean_rate, occupancy)


def _encoding_supported(
    log_w: jax.Array,
    encoding_features: jax.Array,
    waveform_stds: jax.Array,
    *position_inputs: jax.Array | None,
) -> jax.Array:
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
        supported |= jnp.any(jnp.isnan(cast(jax.Array, position_input)))
    return supported


def _score_spikes(
    supported: jax.Array,
    decoding_features: jax.Array,
    encoding_features: jax.Array,
    waveform_stds: jax.Array,
    occupancy: jax.Array,
    mean_rate: float,
    log_position_distance: jax.Array | None,
    log_w: jax.Array,
    enc_tile_size: int,
    pos_tile_size: int | None,
    log_background: jax.Array,
    **reduction_options: Any,
) -> jax.Array:
    """Each spike's term ``logaddexp(log intensity, log_background)``.

    Arguments are those of ``_estimate_with_enc_chunking`` after
    ``supported`` (see ``_encoding_supported``); an unsupported model scores
    the background without reducing.

    Returns
    -------
    log_terms : jax.Array, shape (n_decoding_spikes, n_position_bins)
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
    decoding_spike_waveform_features: jax.Array,
    encoding_spike_waveform_features: jax.Array,
    waveform_stds: jax.Array,
    occupancy: jax.Array,
    mean_rate: float,
    log_position_distance: jax.Array | None = None,
    *,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_positions: jax.Array | None = None,
    position_eval_points: jax.Array | None = None,
    position_std: jax.Array | None = None,
    encoding_weights: jax.Array | None = None,
    log_occupancy: jax.Array | None = None,
    log_background: jax.Array,
) -> jax.Array:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jax.Array, shape (n_decoding_spikes, n_features)
        Waveform features (marks) of the spikes to score.
    encoding_spike_waveform_features : jax.Array, shape (n_encoding_spikes, n_features)
        Waveform features (marks) of the encoding spikes that define the kernel
        density.
    waveform_stds : jax.Array, shape (n_features,)
        Gaussian kernel standard deviation of each waveform feature.
    occupancy : jax.Array, shape (n_position_bins,)
        Occupancy density at each position bin. The intensity is divided by it
        unless ``log_occupancy`` is given.
    mean_rate : float
        Mean firing rate of the electrode, in Hz.
    log_position_distance : jax.Array | None, shape (n_encoding_spikes, n_position_bins)
        Log-space position kernel (output of log_kde_distance). Required if use_streaming=False.
        Using log-space prevents underflow from multi-dimensional Gaussian products.
    pos_tile_size : int | None, optional
        If provided, split each encoding tile's GEMM over position chunks of this
        size. By default all positions share one GEMM.
    enc_tile_size : int | None, optional
        Encoding rows per tile. Tiles never exceed 2048
        rows, so peak memory is O(tile * (n_decoding_spikes + n_position_bins))
        plus the output, independent of the encoding count.
    use_streaming : bool, optional, default=False
        If True, compute position kernel on-the-fly per encoding chunk (streaming mode).
        Avoids materializing full (n_enc × n_pos) position distance matrix.
        Requires encoding_positions, position_eval_points, and position_std.
        Only valid with enc_tile_size (requires chunking).
    encoding_positions : jax.Array | None, shape (n_encoding_spikes, n_position_dims)
        Encoding spike positions. Required if use_streaming=True.
    position_eval_points : jax.Array | None, shape (n_position_bins, n_position_dims)
        Position evaluation points (e.g., interior_place_bin_centers). Required if use_streaming=True.
    position_std : jax.Array | None, shape (n_position_dims,)
        Position standard deviations. Required if use_streaming=True.
    encoding_weights : jax.Array | None, shape (n_encoding_spikes,), optional
        Per-encoding-spike weights (the posterior weight at each spike time). When
        None (default), every spike is weighted uniformly (``1 / n``). When given,
        each spike contributes ``w_e / sum_w``; a zero weight drops that spike out
        of the reduction entirely. Applied identically on every numerical path.
    log_occupancy : jax.Array | None, shape (n_position_bins,), optional
        Log occupancy density. When given it replaces ``occupancy``: the
        position kernel is divided by it in log space before the mark kernel is
        added, which keeps small tail intensities representable. Bins where it
        is ``-inf`` have zero intensity. By default None (use ``occupancy``).
    log_background : jax.Array, shape (n_decoding_spikes,), keyword-only
        Log background intensity of each decoding spike, in the units of
        ``mean_rate``: ``log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS) + log q(mark)``
        for a mark reference ``q``.

    Returns
    -------
    log_joint_mark_intensity : jax.Array, shape (n_decoding_spikes, n_position_bins)
        ``logaddexp(log intensity, log_background)``. A zero
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
    ``static_argnames=('pos_tile_size', 'enc_tile_size', 'use_streaming')`` —
    so callers get JIT automatically and do not need to
    wrap it themselves. The undecorated function object remains accessible via
    ``estimate_log_joint_mark_intensity.__wrapped__`` if an un-jitted version is
    needed.
    """
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]
    position_log_normalizer = None

    # Validation: streaming requires chunking configuration and inputs.
    if use_streaming:
        if enc_tile_size is None:
            msg = "use_streaming=True requires enc_tile_size to be specified"
            raise ValueError(msg)
        if (
            encoding_positions is None
            or position_eval_points is None
            or position_std is None
        ):
            msg = (
                "use_streaming=True requires encoding_positions, position_eval_points, "
                "and position_std to be specified"
            )
            raise ValueError(msg)
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
        n_pos = cast(jax.Array, position_eval_points).shape[0]
    else:
        n_pos = cast(jax.Array, log_position_distance).shape[1]
    n_dec = decoding_spike_waveform_features.shape[0]

    if log_occupancy is not None:
        normalizer, occupancy = _occupancy_normalizer(log_occupancy)
        if use_streaming:
            position_log_normalizer = normalizer
        else:
            log_position_distance = (
                cast(jax.Array, log_position_distance) - normalizer[None, :]
            )

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
    position_inputs: tuple[jax.Array | None, ...] = (
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
    static_argnames=("pos_tile_size", "enc_tile_size", "use_streaming"),
)


def _validate_execution_settings(
    block_size: Any, enc_tile_size: Any, pos_tile_size: Any, use_streaming: Any
) -> None:
    """Reject block and tile settings the fit and the prediction cannot use."""
    validate_positive_int("block_size", block_size)
    validate_positive_int("enc_tile_size", enc_tile_size, allow_none=True)
    validate_positive_int("pos_tile_size", pos_tile_size, allow_none=True)
    if not isinstance(use_streaming, (bool, np.bool_)):
        msg = "use_streaming must be True or False"
        raise ValidationError(msg, got=repr(use_streaming))
    if use_streaming and enc_tile_size is None:
        msg = "use_streaming=True requires enc_tile_size"
        raise ValidationError(
            msg,
            expected="an enc_tile_size with use_streaming=True",
            got="enc_tile_size=None",
            hint="Set enc_tile_size, for example 2048, or use_streaming=False.",
        )


def _pad_with_last_row(array: ArrayLike, n_rows: int) -> np.ndarray:
    """Repeat the last row on the host to reach ``n_rows`` (keeps block maxima)."""
    array = np.asarray(array)
    return np.concatenate([array, np.repeat(array[-1:], n_rows - len(array), axis=0)])


# Float buffers per spike that one GEMM of the batched non-local pass is
# counted as holding against _KDE_BLOCK_KERNEL_BYTES: this many per place bin
# (numerator, marginal, joint, terms) and per encoding sample of the GEMM
# (log mark kernel, mark factor). XLA:CPU holds about three per bin, or six
# when the mark factor dominates, and one per sample.
_NONLOCAL_BIN_BUFFERS = 4
_NONLOCAL_SAMPLE_BUFFERS = 2
# Float buffers per (encoding sample, place bin) that a group of tiles is
# counted as holding: its log position kernels, their weighted sums and the
# slice of scaled factors its GEMM reads.
_NONLOCAL_TILE_BUFFERS = 3
# Float buffers per (spike, encoding sample) that a group of blocks of the
# Local pass is counted as holding: XLA:CPU holds about seven.
_LOCAL_TILE_BUFFERS = 8
# The most blocks of one electrode that the batched passes reduce together.
# Groups of 8, 4, 2 and 1 blocks give an electrode's spikes operations of at
# least half their rows, up to 8 blocks, for four traced copies of the
# reduction; few rows per operation are slow on CPU.
_GROUP_BLOCKS = 8


class _EncodingTiles(NamedTuple):
    """Every electrode's encoding samples, cut into tiles of ``_KDE_SAMPLE_TILE``.

    Attributes
    ----------
    positions : jax.Array, shape (n_tiles, _KDE_SAMPLE_TILE, n_position_dims)
    marks : jax.Array, shape (n_tiles, _KDE_SAMPLE_TILE, n_features)
        Zero on padded features, which electrodes with fewer features have.
    weights : jax.Array, shape (n_tiles, _KDE_SAMPLE_TILE)
        Zero on padding rows.
    electrode : jax.Array of int32, shape (n_tiles,)
        Each tile's electrode, nondecreasing; ``n_electrodes`` on padding
        tiles.
    first_tile : jax.Array of int32, shape (n_electrodes + 1,)
        Electrode ``e``'s tiles are ``first_tile[e]`` to ``first_tile[e + 1]``.
    """

    positions: jax.Array
    marks: jax.Array
    weights: jax.Array
    electrode: jax.Array
    first_tile: jax.Array


class _ElectrodeTerms(NamedTuple):
    """Per-electrode kernel settings of the batched passes.

    Attributes
    ----------
    waveform_stds : jax.Array, shape (n_electrodes, n_features)
        Mark bandwidths; 1 on padded features.
    feature_mask : jax.Array of bool, shape (n_electrodes, n_features)
        The electrode's own features.
    gemm_shift, gemm_window : jax.Array, shape (n_electrodes,)
        ``_scaled_gemm_shift`` and ``_scaled_gemm_window`` of the electrode's
        sample count for a float32 GEMM.
    """

    waveform_stds: jax.Array
    feature_mask: jax.Array
    gemm_shift: jax.Array
    gemm_window: jax.Array


class _SpikeBlocks(NamedTuple):
    """Every electrode's decoding spikes, cut into blocks of one electrode each.

    Attributes
    ----------
    marks : jax.Array, shape (n_blocks, block_rows, n_features)
        Zero on padded features.
    log_background : jax.Array, shape (n_blocks, block_rows)
    rows : jax.Array, shape (n_blocks, block_rows)
        Local row of each spike; ``n_rows`` on padding rows, which the row
        sums drop.
    first_block : jax.Array of int32, shape (n_electrodes + 1,)
        Electrode ``e``'s blocks are ``first_block[e]`` to
        ``first_block[e + 1]``; the padding blocks after the last are never
        reduced.
    positions : jax.Array, shape (n_blocks, block_rows, n_position_dims), or None
        Position at each spike (the Local pass only).
    log_occupancy : jax.Array, shape (n_blocks, block_rows), or None
        Log occupancy density at each spike's position (the Local pass only).
    """

    marks: jax.Array
    log_background: jax.Array
    rows: jax.Array
    first_block: jax.Array
    positions: jax.Array | None = None
    log_occupancy: jax.Array | None = None


def _encoding_tiles(
    spike_time_order: SpikeTimeOrder,
    encoding_features: Sequence[ArrayLike],
    encoding_positions: Sequence[ArrayLike],
    encoding_weights: Sequence[ArrayLike | None],
    waveform_std: jax.Array | np.ndarray | float,
) -> tuple[_EncodingTiles, _ElectrodeTerms, np.ndarray]:
    """Tile every electrode's encoding samples once per prediction.

    Each electrode's samples are cut into tiles of ``_KDE_SAMPLE_TILE`` rows,
    the last padded with zero-weight rows (an electrode without samples has
    one), and its marks are padded with zero features to the most any
    electrode has. The tile count is padded to four sizes per octave with
    zero-weight tiles of electrode ``n_electrodes``. One compiled pass then
    covers electrodes of any sample and feature counts.

    Parameters
    ----------
    spike_time_order : SpikeTimeOrder
        The prediction's preparation cache; the tiles are reused across its
        chunks.
    encoding_features : sequence of array, each shape (n_samples, n_features)
        Each electrode's encoding waveform features.
    encoding_positions : sequence of array, each shape (n_samples, n_position_dims)
        Each electrode's encoding positions.
    encoding_weights : sequence of array or None, each shape (n_samples,)
        None weights an electrode's samples uniformly.
    waveform_std : float or array, shape (n_features,)
        Mark bandwidth, shared by every electrode.

    Returns
    -------
    tiles : _EncodingTiles
    electrodes : _ElectrodeTerms
    n_tiles : np.ndarray of int, shape (n_electrodes,)
        Each electrode's tiles.
    """

    def build() -> tuple[_EncodingTiles, _ElectrodeTerms, np.ndarray]:
        n_features = [int(np.shape(f)[1]) for f in encoding_features]
        width = max(n_features)
        marks = [
            np.pad(np.asarray(f), ((0, 0), (0, width - n)))
            for f, n in zip(encoding_features, n_features, strict=True)
        ]
        weights = [
            np.ones(np.shape(f)[0]) if w is None else np.asarray(w)
            for f, w in zip(encoding_features, encoding_weights, strict=True)
        ]
        tile_positions, tile_weights, tile_electrode = _sample_tiles(
            [np.asarray(p) for p in encoding_positions], weights
        )
        tile_marks, _, _ = _sample_tiles(marks, weights)
        n_electrodes = len(n_features)
        first_tile = np.searchsorted(tile_electrode, np.arange(n_electrodes + 1))
        positions, device_weights, device_electrode = _bucketed_tiles(
            tile_positions,
            tile_weights,
            tile_electrode,
            padded_tile_count(tile_electrode.size),
        )
        electrode_stds = [np.asarray(as_std_array(waveform_std, n)) for n in n_features]
        stds = np.ones(
            (n_electrodes, width), np.result_type(*electrode_stds, np.float32)
        )
        for electrode, (n, std) in enumerate(
            zip(n_features, electrode_stds, strict=True)
        ):
            stds[electrode, :n] = std
        # The bound of each electrode's scaled sum counts its own samples:
        # padding rows have zero weight and add nothing.
        n_samples = [max(np.shape(f)[0], 1) for f in encoding_features]
        return (
            _EncodingTiles(
                positions,
                jnp.asarray(pad_rows(tile_marks, device_electrode.shape[0])),
                device_weights,
                device_electrode,
                jnp.asarray(first_tile.astype(np.int32)),
            ),
            _ElectrodeTerms(
                jnp.asarray(stds),
                jnp.asarray(np.arange(width) < np.asarray(n_features)[:, None]),
                jnp.asarray(
                    np.array([_scaled_gemm_shift(n, jnp.float32) for n in n_samples])
                ),
                jnp.asarray(
                    np.array([_scaled_gemm_window(n, jnp.float32) for n in n_samples])
                ),
            ),
            np.diff(first_tile),
        )

    return spike_time_order.memo(
        "clusterless_kde_encoding_tiles",
        (
            *encoding_features,
            *encoding_positions,
            *encoding_weights,
            waveform_std,
        ),
        build,
    )


def _power_of_two_below(n: int) -> int:
    """Return the largest power of two at most ``max(n, 1)``."""
    return 1 << (max(n, 1).bit_length() - 1)


def _nonlocal_sizes(
    n_spikes: Sequence[int],
    n_tiles: Sequence[int],
    block_size: int,
    n_bins: int,
    itemsize: int,
    widest_tiles: int,
    expected: tuple[Sequence[int], Sequence[int]],
) -> tuple[int, int, int]:
    """Sizes of the batched non-local pass.

    One GEMM reduces at most ``group_blocks`` blocks of one electrode
    against at most ``group_tiles`` of its tiles. A group of tiles holds
    buffers counted as ``_NONLOCAL_TILE_BUFFERS`` floats per sample per
    place bin, within half of ``_KDE_BLOCK_KERNEL_BYTES`` (one tile at
    least). The GEMM's rows, at most ``block_size``, hold buffers counted as
    ``_NONLOCAL_BIN_BUFFERS`` floats per place bin and
    ``_NONLOCAL_SAMPLE_BUFFERS`` per sample of the group of tiles, within the
    rest of the budget (16 rows at least).

    Parameters
    ----------
    n_spikes : sequence of int
        Spike count of each electrode with spikes in the rows.
    n_tiles : sequence of int
        Encoding tiles of the same electrodes.
    block_size : int
        The requested spikes per block.
    n_bins : int
        Place bins.
    itemsize : int
        Bytes per float of the pass.
    widest_tiles : int
        Tiles of the electrode with the most, spikes or not.
    expected : (sequence of int, sequence of int)
        Expected spike and tile counts for the chunk (see ``_block_rows``).

    Returns
    -------
    block_rows, group_blocks, group_tiles : int
    """
    tile_bytes = _NONLOCAL_TILE_BUFFERS * _KDE_SAMPLE_TILE * n_bins * itemsize
    group_tiles = _power_of_two_below(
        min(
            _LOG_REDUCTION_ENCODING_ROWS // _KDE_SAMPLE_TILE,
            widest_tiles,
            _KDE_BLOCK_KERNEL_BYTES // 2 // tile_bytes,
        )
    )
    row_bytes = itemsize * (
        _NONLOCAL_BIN_BUFFERS * n_bins
        + _NONLOCAL_SAMPLE_BUFFERS * group_tiles * _KDE_SAMPLE_TILE
    )
    row_limit = max(
        16, (_KDE_BLOCK_KERNEL_BYTES - group_tiles * tile_bytes) // row_bytes
    )
    return _group_sizes(n_spikes, n_tiles, block_size, row_limit, group_tiles, expected)


def _local_sizes(
    n_spikes: Sequence[int],
    n_tiles: Sequence[int],
    block_size: int,
    widest_tiles: int,
    expected: tuple[Sequence[int], Sequence[int]],
) -> tuple[int, int, int]:
    """Sizes of the Local pass.

    At most ``group_blocks`` blocks of one electrode are reduced against at
    most ``group_tiles`` of its tiles at once. Their (samples x spikes)
    buffers, counted as ``_LOCAL_TILE_BUFFERS`` floats per entry of the
    default float type, fit ``_KDE_BLOCK_KERNEL_BYTES``.

    Parameters
    ----------
    n_spikes : sequence of int
        Spike count of each electrode with spikes in the rows.
    n_tiles : sequence of int
        Encoding tiles of the same electrodes.
    block_size : int
        The requested spikes per block.
    widest_tiles : int
        Tiles of the electrode with the most, spikes or not.
    expected : (sequence of int, sequence of int)
        Expected spike and tile counts for the chunk (see ``_block_rows``).

    Returns
    -------
    block_rows, group_blocks, group_tiles : int
    """
    itemsize = jnp.dtype(jax.dtypes.canonicalize_dtype(jnp.float64)).itemsize
    group_tiles = _power_of_two_below(
        min(_LOG_REDUCTION_ENCODING_ROWS // _KDE_SAMPLE_TILE, widest_tiles)
    )
    row_limit = max(
        1,
        _KDE_BLOCK_KERNEL_BYTES
        // (_LOCAL_TILE_BUFFERS * group_tiles * _KDE_SAMPLE_TILE * itemsize),
    )
    return _group_sizes(n_spikes, n_tiles, block_size, row_limit, group_tiles, expected)


def _group_sizes(
    n_spikes: Sequence[int],
    n_tiles: Sequence[int],
    block_size: int,
    row_limit: int,
    group_tiles: int,
    expected: tuple[Sequence[int], Sequence[int]],
) -> tuple[int, int, int]:
    """Block rows and the largest groups of blocks and tiles of a batched pass.

    A group of blocks holds at most ``min(block_size, row_limit)`` spikes
    and ``_GROUP_BLOCKS`` blocks, at most the padded block count; both group
    sizes are powers of two.
    """
    block_rows = _block_rows(n_spikes, n_tiles, block_size, row_limit, expected)
    n_blocks = padded_block_count(sum(-(-n // block_rows) for n in n_spikes))
    group_blocks = _power_of_two_below(
        min(_GROUP_BLOCKS, min(block_size, row_limit) // block_rows, n_blocks)
    )
    return block_rows, group_blocks, group_tiles


def _block_rows(
    n_spikes: Sequence[int],
    n_tiles: Sequence[int],
    block_size: int,
    row_limit: int,
    expected: tuple[Sequence[int], Sequence[int]],
) -> int:
    """Spikes per block of a batched pass, shared by every electrode.

    Each electrode's spikes are padded to whole blocks, and each padded
    spike is reduced against all of its electrode's tiles. The rows are the
    largest of 16, 64, 256, ... (powers of four) below
    ``min(block_size, row_limit)``, or that cap itself, that adds at most a
    quarter to the (spike, tile) pairs the pass reduces; the smallest of them
    when none does. The size the same rule picks for the ``expected`` spike
    counts of a chunk this long is preferred when it meets the bound, so
    chunks of one decode share compiled passes. The choice depends on the
    chunk alone, not on the chunks decoded before it.

    Parameters
    ----------
    n_spikes : sequence of int
        Each electrode's spike count, at least one each.
    n_tiles : sequence of int
        Each electrode's encoding tiles.
    block_size : int
        The requested spikes per block.
    row_limit : int
        The most rows whose buffers fit the pass's budget.
    expected : (sequence of int, sequence of int)
        Spike and tile counts of the electrodes for a chunk of the same
        number of rows at the decode's mean rates (see
        ``_expected_chunk_spikes``).

    Returns
    -------
    rows : int
    """
    cap = max(1, min(block_size, row_limit))
    sizes = [
        size for size in (16 << 2 * k for k in range(cap.bit_length())) if size < cap
    ]
    smallest = (sizes or [cap])[0]

    def meeting(spikes: Sequence[int], tiles: Sequence[int]) -> list[int]:
        pairs = sum(n * t for n, t in zip(spikes, tiles, strict=True))
        return [
            size
            for size in (*sizes, cap)
            if 4
            * sum(-(-n // size) * size * t for n, t in zip(spikes, tiles, strict=True))
            <= 5 * pairs
        ]

    own = meeting(n_spikes, n_tiles)
    preferred = max(meeting(*expected) or [smallest])
    if preferred in own:
        return preferred
    return max(own or [smallest])


def _expected_chunk_spikes(
    rows: DecodeRows,
    spike_times: Sequence[ArrayLike],
    n_tiles: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Spike and tile counts of the electrodes for a chunk of ``rows``'s length.

    Each electrode's spikes over the whole decode, scaled to the chunk's
    share of its rows and rounded up; electrodes without decoding spikes are
    left out. They set a preferred block size (see ``_block_rows``) from the
    decode and the chunk's length alone.

    Parameters
    ----------
    rows : DecodeRows
        The chunk's rows within the full decode grid.
    spike_times : sequence of array, each shape (n_spikes,)
        Every electrode's decoding spike times.
    n_tiles : np.ndarray of int, shape (n_electrodes,)
        Each electrode's encoding tiles.

    Returns
    -------
    spikes, tiles : list of int
    """
    edges = rows.time_edges

    def build() -> np.ndarray:
        counts = []
        for times in spike_times:
            values, _ = rows.spike_time_order.get(times)
            counts.append(
                int(np.count_nonzero((values >= edges[0]) & (values <= edges[-1])))
            )
        return np.asarray(counts)

    totals = rows.spike_time_order.memo(
        "clusterless_kde_decode_spike_counts", (*spike_times, edges), build
    )
    share = (rows.stop - rows.start) / max(len(edges) - 1, 1)
    keep = np.flatnonzero(totals)
    return (
        [max(1, int(np.ceil(totals[e] * share))) for e in keep],
        [int(n_tiles[e]) for e in keep],
    )


def _spike_blocks(
    spikes: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    n_rows: int,
    n_features: Sequence[int],
    block_rows: int,
    *,
    pad_with_last: bool,
    per_spike: Sequence[Sequence[np.ndarray]] = (),
) -> tuple[_SpikeBlocks, tuple[np.ndarray, ...]]:
    """Cut every electrode's spikes into blocks of ``block_rows``, on the host.

    Parameters
    ----------
    spikes : sequence of (electrode, marks, log_background, rows)
        Each electrode's index, marks ``(n_spikes, n_features)``, log
        background ``(n_spikes,)`` and local rows ``(n_spikes,)``.
    n_rows : int
        Number of output rows; the id of padding rows.
    n_features : sequence of int
        Each electrode's encoding feature count, which its marks must have.
        Marks are padded with zero features to the largest.
    block_rows : int
        Spikes per block.
    pad_with_last : bool
        Pad an electrode's last block with its last spike, rather than zeros.
    per_spike : sequence of sequences of np.ndarray
        Further per-spike arrays, one sequence per array and one array per
        electrode in ``spikes``, padded like the marks.

    Returns
    -------
    blocks : _SpikeBlocks
        The number of blocks is padded to a power of four.
    per_spike : tuple of np.ndarray, each shape (n_blocks, block_rows, ...)

    Raises
    ------
    ValidationError
        If an electrode's marks have another feature count than its encoding
        marks.
    """
    for electrode, marks, _, _ in spikes:
        if marks.shape[1] != n_features[electrode]:
            msg = (
                f"waveform features of electrode {electrode} do not match its encoding"
            )
            raise ValidationError(
                msg,
                expected=f"{n_features[electrode]} features per spike",
                got=f"{marks.shape[1]} features per spike",
            )
    n_electrodes, width = len(n_features), max(n_features)

    def pad(array: np.ndarray, n: int) -> np.ndarray:
        return _pad_with_last_row(array, n) if pad_with_last else pad_rows(array, n)

    n_blocks = [-(-marks.shape[0] // block_rows) for _, marks, _, _ in spikes]
    n_padded = padded_block_count(sum(n_blocks))

    def blocked(arrays: list[np.ndarray], fill: float) -> np.ndarray:
        padded = np.concatenate(
            [pad(a, n * block_rows) for a, n in zip(arrays, n_blocks, strict=True)]
        )
        tail = np.full(
            ((n_padded - sum(n_blocks)) * block_rows, *padded.shape[1:]),
            fill,
            padded.dtype,
        )
        return np.concatenate([padded, tail]).reshape(
            n_padded, block_rows, *padded.shape[1:]
        )

    marks = blocked(
        [np.pad(m, ((0, 0), (0, width - m.shape[1]))) for _, m, _, _ in spikes], 0
    )
    # Rows pad with the dropped id n_rows, which keeps sorted rows sorted.
    rows = blocked(
        [
            np.concatenate([r, np.full(n * block_rows - r.size, n_rows, r.dtype)])
            for (_, _, _, r), n in zip(spikes, n_blocks, strict=True)
        ],
        n_rows,
    )
    electrodes = [e for e, _, _, _ in spikes]
    if np.any(np.diff(electrodes) <= 0):
        msg = "electrodes must be listed once each, in increasing order"
        raise ValueError(msg)
    first_block = np.searchsorted(
        np.repeat(electrodes, n_blocks), np.arange(n_electrodes + 1)
    )
    return (
        _SpikeBlocks(
            jnp.asarray(marks),
            jnp.asarray(blocked([b for _, _, b, _ in spikes], 0)),
            jnp.asarray(rows),
            jnp.asarray(first_block.astype(np.int32)),
        ),
        tuple(blocked(list(arrays), 0) for arrays in per_spike),
    )


def _fold_groups(
    add: Callable[[Any, jax.Array, int], Any],
    largest: int,
    begin: jax.Array,
    end: jax.Array,
    accumulated: Any,
) -> Any:
    """Fold ``add(accumulated, index, size)`` over ``begin`` to ``end`` in groups.

    As many groups of ``largest`` (a power of two) as fit, then at most one
    group each of half as many, and so on to one. Larger groups make larger
    operations; each size is traced once.

    Parameters
    ----------
    add : callable
        ``add(accumulated, index, size)`` adds the group of ``size`` items
        from ``index``.
    largest : int
        Items in the largest group, a power of two.
    begin, end : jax.Array, scalar int
        The items to fold.
    accumulated : pytree of jax.Array
        Initial value.

    Returns
    -------
    accumulated : pytree of jax.Array
    """
    index = jnp.asarray(begin)
    for size in (1 << k for k in reversed(range(largest.bit_length()))):
        n_groups = (end - index) // size
        accumulated = jax.lax.fori_loop(
            0,
            n_groups,
            lambda group, accumulated, index=index, size=size: add(
                accumulated, index + group * size, size
            ),
            accumulated,
        )
        index = index + n_groups * size
    return accumulated


def _electrode_weight_totals(tiles: _EncodingTiles, n_electrodes: int) -> jax.Array:
    """Each electrode's total sample weight, summed in a fixed order."""
    return _sum_over_tiles(
        jnp.sum(tiles.weights, axis=1), tiles.electrode, n_electrodes + 1
    )


@partial(
    jax.jit,
    static_argnames=(
        "widest_tiles",
        "group_blocks",
        "group_tiles",
        "pos_tile_size",
        "indices_are_sorted",
    ),
    donate_argnums=0,
)
def _add_log_mark_intensities(
    total: jax.Array,
    blocks: _SpikeBlocks,
    tiles: _EncodingTiles,
    electrodes: _ElectrodeTerms,
    mean_rates: jax.Array,
    position_std: jax.Array,
    place_bin_centers: jax.Array,
    log_occupancy: jax.Array,
    *,
    widest_tiles: int,
    group_blocks: int,
    group_tiles: int,
    pos_tile_size: int | None,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add every electrode's non-local log marked intensities to their rows.

    Electrode by electrode, the scaled position factors of its tiles are
    formed once, then each of its blocks of spikes is reduced over them with
    the electrode's own weights, bandwidths, mean rate and background, as
    ``_estimate_with_enc_chunking`` reduces one electrode: a scaled GEMM,
    exact above each spike's floor, and a direct logsumexp over its tiles for
    the spikes it flags. Each block's terms are then added to their rows
    (see ``deterministic_row_accumulate``).

    Parameters
    ----------
    total : jax.Array, shape (n_rows, n_bins)
        Running log likelihood; donated.
    blocks : _SpikeBlocks
        Decoding spikes, with ``first_block``; padding repeats each
        electrode's last spike, which is reduced like it and dropped.
    tiles : _EncodingTiles
        Every electrode's encoding samples.
    electrodes : _ElectrodeTerms
        Every electrode's mark bandwidths and scaled-sum bounds.
    mean_rates : jax.Array, shape (n_electrodes,)
        Mean rates times ``RATE_REFERENCE_SECONDS``.
    position_std : jax.Array, shape (n_position_dims,)
        Position kernel bandwidths.
    place_bin_centers : jax.Array, shape (n_bins, n_position_dims)
        Place bin centers.
    log_occupancy : jax.Array, shape (n_bins,)
        Log occupancy density at each bin.
    widest_tiles : int
        The most tiles any electrode has, which sizes the position factors
        of one electrode.
    group_blocks, group_tiles : int
        Powers of two: the most blocks and tiles of one
        electrode reduced in one GEMM (see ``_fold_groups``). Larger GEMMs
        are faster, and the padding stays within one block and one tile per
        electrode.
    pos_tile_size : int or None
        Place bins per tile of the reduction; None for no tiling.
    indices_are_sorted : bool
        Whether every block's rows are nondecreasing.

    Returns
    -------
    total : jax.Array, shape (n_rows, n_bins)
    """
    n_electrodes = mean_rates.shape[0]
    n_bins = place_bin_centers.shape[0]
    block_rows = blocks.marks.shape[1]
    tile_rows = tiles.weights.shape[1]
    first_block = blocks.first_block
    pos_tile_size = min(n_bins, pos_tile_size or n_bins)
    normalizer, occupancy = _occupancy_normalizer(log_occupancy)
    dtype = jnp.result_type(
        jnp.float32,
        blocks.marks,
        tiles.marks,
        electrodes.waveform_stds,
        tiles.weights,
        tiles.positions,
        place_bin_centers,
        position_std,
        log_occupancy,
    )
    weight_total = _electrode_weight_totals(tiles, n_electrodes)
    mean_rates = jnp.asarray(mean_rates, dtype)
    n_features = electrodes.waveform_stds.shape[1]
    origin = jnp.zeros((1, n_features), dtype)
    group_size = _repair_group_size(tile_rows, pos_tile_size)

    def log_position(tile: jax.Array, pos_start: jax.Array | int) -> jax.Array:
        """Log position kernel of one tile at ``pos_tile_size`` bins, over occupancy."""
        bins = jax.lax.dynamic_slice_in_dim(place_bin_centers, pos_start, pos_tile_size)
        return cast(
            jax.Array,
            log_kde_distance(bins, tiles.positions[tile], std=position_std)
            - jax.lax.dynamic_slice_in_dim(normalizer, pos_start, pos_tile_size)[
                None, :
            ],
        )

    def log_weights(tile: jax.Array, electrode: jax.Array) -> jax.Array:
        return jnp.asarray(
            _log_weights_given_total(tiles.weights[tile], weight_total[electrode]),
            dtype,
        )

    gemm_shift = jnp.asarray(electrodes.gemm_shift, dtype)

    def add_electrode(
        carry: tuple[jax.Array, jax.Array], electrode: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        total, factors = carry
        first = tiles.first_tile[electrode]
        # An electrode without spikes in the rows needs no factors.
        has_spikes = first_block[electrode + 1] > first_block[electrode]
        stop = jnp.where(has_spikes, tiles.first_tile[electrode + 1], first)
        stds = electrodes.waveform_stds[electrode]
        mask = electrodes.feature_mask[electrode]
        shift = gemm_shift[electrode]
        mark_peak = compute_log_mark_kernel_gemm(origin, origin, stds, mask)[0, 0]

        # The electrode's weighted log position kernels, their largest value
        # per bin and whether they can give a nonzero or NaN intensity (see
        # _encoding_supported), then its scaled factors, indexed from its
        # first tile.
        def weighted_tiles(
            carry: tuple[jax.Array, jax.Array, jax.Array],
            tile: jax.Array,
            n_tiles: int,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            factors, peak, supported = carry
            log_w = jnp.asarray(
                _log_weights_given_total(
                    jax.lax.dynamic_slice_in_dim(tiles.weights, tile, n_tiles),
                    weight_total[electrode],
                ),
                dtype,
            )
            log_position_distance = (
                jax.vmap(
                    lambda positions: log_kde_distance(
                        place_bin_centers, positions, std=position_std
                    )
                )(jax.lax.dynamic_slice_in_dim(tiles.positions, tile, n_tiles))
                - (normalizer[None, None, :])
            )
            weighted = (log_position_distance + log_w[:, :, None]).astype(dtype)
            supported |= (
                jnp.any(jnp.isfinite(log_w))
                | jnp.any(jnp.isnan(log_w))
                | jnp.any(
                    jnp.isnan(jax.lax.dynamic_slice_in_dim(tiles.marks, tile, n_tiles))
                )
                | jnp.any(jnp.isnan(log_position_distance))
            )
            return (
                jax.lax.dynamic_update_slice_in_dim(factors, weighted, tile - first, 0),
                jnp.maximum(peak, jnp.max(weighted, axis=(0, 1))),
                supported,
            )

        factors, peak, supported = _fold_groups(
            weighted_tiles,
            group_tiles,
            first,
            stop,
            (factors, jnp.full(n_bins, -jnp.inf, dtype), jnp.any(jnp.isnan(stds))),
        )

        safe_peak = zero_if_neginf(peak)

        # The scaled position factor of each tile.
        def scaled_tiles(
            factors: jax.Array, index: jax.Array, n_tiles: int
        ) -> jax.Array:
            weighted = jax.lax.dynamic_slice_in_dim(factors, index, n_tiles)
            return jax.lax.dynamic_update_slice_in_dim(
                factors,
                _scaled_factor(weighted, safe_peak[None, None, :], shift),
                index,
                0,
            )

        factors = _fold_groups(
            scaled_tiles, group_tiles, jnp.zeros((), first.dtype), stop - first, factors
        )
        exact_threshold = _exact_threshold(
            mark_peak,
            peak,
            jnp.asarray(electrodes.gemm_window[electrode], dtype),
            _log_marginal_floor(mean_rates[electrode], occupancy),
        )

        start, end = first_block[electrode], first_block[electrode + 1]

        def spike_terms(marks: jax.Array, background: jax.Array) -> jax.Array:
            """``logaddexp(log intensity, background)`` of some of its spikes."""
            n_spikes = marks.shape[0]

            def reduce_linearly() -> jax.Array:
                def add_tile_group(
                    numerator: jax.Array, tile: jax.Array, n_tiles: int
                ) -> jax.Array:
                    """Add the scaled GEMM over ``n_tiles`` tiles from ``tile``."""
                    position_factors = jax.lax.dynamic_slice_in_dim(
                        factors, tile - first, n_tiles
                    ).reshape(n_tiles * tile_rows, n_bins)
                    return _add_scaled_tile(
                        numerator,
                        marks,
                        jax.lax.dynamic_slice_in_dim(
                            tiles.marks, tile, n_tiles
                        ).reshape(n_tiles * tile_rows, n_features),
                        stds,
                        mark_peak,
                        shift,
                        lambda pos_start: jax.lax.dynamic_slice_in_dim(
                            position_factors, pos_start, pos_tile_size, axis=1
                        ),
                        pos_tile_size,
                        mask,
                    )

                numerator = _fold_groups(
                    add_tile_group,
                    group_tiles,
                    first,
                    stop,
                    jnp.zeros((n_spikes, n_bins), dtype),
                )
                return _linear_log_marginal(numerator, mark_peak, safe_peak, shift)

            def exact_rows(rows: jax.Array) -> jax.Array:
                queries = marks[rows]
                return cast(
                    jax.Array,
                    jax.lax.fori_loop(
                        first,
                        stop,
                        lambda tile, marginal: _add_exact_tile(
                            marginal,
                            queries,
                            tiles.marks[tile],
                            stds,
                            log_weights(tile, electrode),
                            lambda pos_start: log_position(tile, pos_start),
                            pos_tile_size,
                            mask,
                        ),
                        jnp.full((rows.shape[0], n_bins), -jnp.inf, dtype),
                    ),
                )

            def estimate() -> jax.Array:
                query_floor = jnp.asarray(background, dtype) + np.log(finfo(dtype).eps)
                marginal = _repair_flagged_in_groups(
                    reduce_linearly(),
                    exact_threshold > query_floor,
                    exact_rows,
                    group_size,
                )
                return _log_joint_from_log_marginal(
                    marginal, mean_rates[electrode], occupancy
                )

            # As _score_spikes: spikes of an electrode that cannot give a
            # nonzero or NaN intensity have the background alone.
            log_joint = jax.lax.cond(
                supported | jnp.any(jnp.isnan(marks)),
                estimate,
                lambda: _log_joint_from_log_marginal(
                    jnp.full((n_spikes, n_bins), -jnp.inf, dtype),
                    mean_rates[electrode],
                    occupancy,
                ),
            )
            return jnp.logaddexp(log_joint, background[:, None])

        def add_blocks(total: jax.Array, block: jax.Array, n_blocks: int) -> jax.Array:
            """Add the terms of ``n_blocks`` blocks from ``block`` to their rows."""

            def spikes(values: jax.Array) -> jax.Array:
                return jax.lax.dynamic_slice_in_dim(values, block, n_blocks).reshape(
                    n_blocks * block_rows, *values.shape[2:]
                )

            return deterministic_row_accumulate(
                total,
                spike_terms(spikes(blocks.marks), spikes(blocks.log_background)),
                spikes(blocks.rows),
                indices_are_sorted=indices_are_sorted,
            )

        # Groups of group_blocks blocks are reduced together, then at most one
        # group each of half as many, and so on to one block; likewise each
        # GEMM's tiles.
        total = _fold_groups(add_blocks, group_blocks, start, end, total)
        return (total, factors), None

    (total, _), _ = jax.lax.scan(
        add_electrode,
        (total, jnp.zeros((widest_tiles, tile_rows, n_bins), dtype)),
        jnp.arange(n_electrodes),
    )
    return total


@partial(
    jax.jit,
    static_argnames=("group_blocks", "group_tiles", "indices_are_sorted"),
    donate_argnums=0,
)
def _add_local_terms(
    log_likelihood: jax.Array,
    blocks: _SpikeBlocks,
    tiles: _EncodingTiles,
    electrodes: _ElectrodeTerms,
    scaled_rates: jax.Array,
    position_std: jax.Array,
    *,
    group_blocks: int,
    group_tiles: int,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add every electrode's local spike terms to their rows.

    Electrode by electrode, its blocks of spikes are reduced over its tiles
    only. The joint KDE over occupancy at each spike's position and mark is
    a logsumexp over the electrode's samples, carried tile by tile with a
    running maximum. The position kernels are divided by the occupancy
    before the mark kernels are added, which keeps the mark normalizer when
    position log densities are large enough to round it away in float32.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows,)
        Running log likelihood; donated.
    blocks : _SpikeBlocks
        Decoding spikes with ``positions`` and ``log_occupancy``; padding
        rows are zero and dropped.
    tiles : _EncodingTiles
        Every electrode's encoding samples.
    electrodes : _ElectrodeTerms
        Every electrode's mark bandwidths.
    scaled_rates : jax.Array, shape (n_electrodes,)
        Mean rates times ``RATE_REFERENCE_SECONDS``.
    position_std : jax.Array, shape (n_position_dims,)
        Position kernel bandwidths.
    group_blocks, group_tiles : int
        Powers of two: the most blocks and tiles of one
        electrode reduced together (see ``_fold_groups``).
    indices_are_sorted : bool
        Whether every block's rows are nondecreasing.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows,)
    """
    n_electrodes = scaled_rates.shape[0]
    positions = cast(jax.Array, blocks.positions)
    block_log_occupancy = cast(jax.Array, blocks.log_occupancy)
    block_rows = blocks.marks.shape[1]
    tile_rows = tiles.weights.shape[1]
    n_features = blocks.marks.shape[2]
    n_position_dims = positions.shape[2]
    weight_total = _electrode_weight_totals(tiles, n_electrodes)
    # Spike positions, marks and occupancy share one float type, as do the
    # samples' positions and marks; the kernels accumulate in the default
    # float.
    dtype = jnp.result_type(
        jax.dtypes.canonicalize_dtype(jnp.float64),
        positions,
        blocks.marks,
        block_log_occupancy,
        tiles.positions,
        tiles.marks,
        position_std,
        electrodes.waveform_stds,
    )

    def add_electrode(
        log_likelihood: jax.Array, electrode: jax.Array
    ) -> tuple[jax.Array, None]:
        stds = electrodes.waveform_stds[electrode]
        mask = electrodes.feature_mask[electrode]
        own_total = weight_total[electrode]
        first, stop = tiles.first_tile[electrode], tiles.first_tile[electrode + 1]

        def add_blocks(
            log_likelihood: jax.Array, block: jax.Array, n_blocks: int
        ) -> jax.Array:
            """Add the terms of ``n_blocks`` blocks from ``block`` to their rows."""

            def spikes(values: jax.Array) -> jax.Array:
                return jax.lax.dynamic_slice_in_dim(values, block, n_blocks).reshape(
                    n_blocks * block_rows, *values.shape[2:]
                )

            n_spikes = n_blocks * block_rows
            log_occ = spikes(block_log_occupancy).astype(dtype)
            safe_occ = jnp.where(jnp.isneginf(log_occ), 0.0, log_occ)
            spike_positions = spikes(positions).astype(dtype)
            spike_marks = spikes(blocks.marks).astype(dtype)

            def log_ratio() -> jax.Array:
                def add_tiles(
                    carry: tuple[jax.Array, jax.Array], tile: jax.Array, n_tiles: int
                ) -> tuple[jax.Array, jax.Array]:
                    """Fold ``n_tiles`` tiles from ``tile`` into the logsumexp."""
                    peak, mass = carry

                    def samples(values: jax.Array, width: int) -> jax.Array:
                        return jax.lax.dynamic_slice_in_dim(
                            values, tile, n_tiles
                        ).reshape(n_tiles * tile_rows, width)

                    weights = jax.lax.dynamic_slice_in_dim(
                        tiles.weights, tile, n_tiles
                    ).reshape(-1)
                    safe_weights = jnp.where(weights > 0.0, weights, 1.0)
                    log_weights = jnp.where(
                        weights > 0.0, jnp.log(safe_weights), -jnp.inf
                    )
                    values = (
                        (
                            _log_kernel_matrix(
                                spike_positions,
                                samples(tiles.positions, n_position_dims).astype(dtype),
                                position_std,
                            )
                            - safe_occ[None, :]
                        )
                        + _log_kernel_matrix(
                            spike_marks,
                            samples(tiles.marks, n_features).astype(dtype),
                            stds,
                            mask,
                        )
                        + log_weights[:, None]
                    )
                    new_peak = jnp.maximum(peak, jnp.max(values, axis=0))
                    scale = jnp.where(jnp.isfinite(new_peak), new_peak, 0.0)
                    mass = mass * jnp.exp(peak - scale) + jnp.sum(
                        jnp.exp(values - scale[None, :]), axis=0
                    )
                    return new_peak, mass

                peak, mass = _fold_groups(
                    add_tiles,
                    group_tiles,
                    first,
                    stop,
                    (
                        jnp.full(n_spikes, -jnp.inf, dtype),
                        jnp.zeros(n_spikes, dtype),
                    ),
                )
                ratio = (
                    jnp.log(mass)
                    + jnp.where(jnp.isfinite(peak), peak, 0.0)
                    - jnp.log(own_total)
                )
                return cast(
                    jax.Array, jnp.where(jnp.isneginf(log_occ), -jnp.inf, ratio)
                )

            ratio = jax.lax.cond(
                own_total > 0.0,
                log_ratio,
                lambda: jnp.full(n_spikes, -jnp.inf, dtype),
            )
            return deterministic_row_accumulate(
                log_likelihood,
                jnp.logaddexp(
                    log_density_intensity(
                        scaled_rates[electrode], ratio, jnp.zeros_like(ratio)
                    ),
                    spikes(blocks.log_background),
                ),
                spikes(blocks.rows),
                indices_are_sorted=indices_are_sorted,
            )

        return _fold_groups(
            add_blocks,
            group_blocks,
            blocks.first_block[electrode],
            blocks.first_block[electrode + 1],
            log_likelihood,
        ), None

    log_likelihood, _ = jax.lax.scan(
        add_electrode, log_likelihood, jnp.arange(n_electrodes)
    )
    return log_likelihood


@jax.jit
def _summed_expected_counts(
    mean_rates: jax.Array, log_gpi_densities: jax.Array, log_occupancy: jax.Array
) -> jax.Array:
    """Every electrode's expected count rate at the rows, added in electrode order.

    Parameters
    ----------
    mean_rates : jax.Array, shape (n_electrodes,)
        Mean rates in Hz.
    log_gpi_densities : jax.Array, shape (n_electrodes, n_rows)
        Each electrode's ground-process log density at the animal's position.
    log_occupancy : jax.Array, shape (n_rows,)
        Log occupancy density at the animal's position.

    Returns
    -------
    expected_counts : jax.Array, shape (n_rows,)
    """

    def add(
        total: jax.Array, electrode: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        rate, log_gpi_density = electrode
        return total + jnp.exp(
            log_density_intensity(rate, log_gpi_density, log_occupancy)
        ), None

    total, _ = jax.lax.scan(
        add,
        jnp.zeros(log_occupancy.shape, log_gpi_densities.dtype),
        (mean_rates, log_gpi_densities),
    )
    return total


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jax.Array | np.ndarray,
    encoding_spike_waveform_features: jax.Array,
    waveform_stds: jax.Array,
    occupancy: jax.Array,
    mean_rate: float,
    log_position_distance: jax.Array | None = None,
    block_size: int = 10_000,
    *,
    pos_tile_size: int | None = None,
    enc_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_positions: jax.Array | None = None,
    position_eval_points: jax.Array | None = None,
    position_std: jax.Array | None = None,
    encoding_weights: jax.Array | None = None,
    log_occupancy: jax.Array | None = None,
    log_background: jax.Array | np.ndarray,
) -> jax.Array:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jax.Array, shape (n_decoding_spikes, n_features)
        Waveform features (marks) of the spikes to score.
    encoding_spike_waveform_features : jax.Array, shape (n_encoding_spikes, n_features)
        Waveform features (marks) of the encoding spikes that define the kernel
        density.
    waveform_stds : jax.Array, shape (n_features,)
        Gaussian kernel standard deviation of each waveform feature.
    occupancy : jax.Array, shape (n_position_bins,)
        Occupancy density at each position bin. The intensity is divided by it
        unless ``log_occupancy`` is given.
    mean_rate : float
        Mean firing rate of the electrode, in Hz.
    log_position_distance : jax.Array | None, shape (n_encoding_spikes, n_position_bins)
        Log-space position kernel. Prevents underflow in multi-dimensional position spaces.
        Can be None if use_streaming=True.
    block_size : int, optional
        Number of decoding spikes to process per block, by default 10,000.
        Fewer spikes are padded to a power of two (at least 16), so calls
        with nearby spike counts share one compiled reduction.
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension.
    enc_tile_size : int | None, optional
        If provided, tile computation over encoding spikes dimension for memory efficiency.
    use_streaming : bool, optional
        If True, compute position kernel on-the-fly using streaming log_kde_distance.
        Requires enc_tile_size, encoding_positions, position_eval_points, position_std.
    encoding_positions : jax.Array | None, shape (n_encoding_spikes, n_position_dims)
        Required when use_streaming=True. Positions where encoding spikes occurred.
    position_eval_points : jax.Array | None, shape (n_position_bins, n_position_dims)
        Required when use_streaming=True. Positions to evaluate (e.g., place bin centers).
    position_std : jax.Array | None, shape (n_position_dims,)
        Required when use_streaming=True. Position kernel bandwidth per dimension.
    encoding_weights : jax.Array | None, shape (n_encoding_spikes,), optional
        Per-encoding-spike weights forwarded to the per-block estimator. None
        (default) weights every encoding spike uniformly.
    log_occupancy : jax.Array | None, shape (n_position_bins,), optional
        Log occupancy density forwarded to the per-block estimator, where it
        replaces ``occupancy`` (see ``estimate_log_joint_mark_intensity``).
    log_background : jax.Array, shape (n_decoding_spikes,), keyword-only
        Log background intensity of each decoding spike (see
        ``estimate_log_joint_mark_intensity``).

    Returns
    -------
    log_joint_mark_intensity : jax.Array, shape (n_decoding_spikes, n_position_bins)
        ``logaddexp(log intensity, log_background)``.
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]
    log_background = jnp.asarray(log_background)

    if n_decoding_spikes == 0:
        return jnp.zeros((0, n_position_bins))
    # A large configured block must not pad a sparse chunk up to its capacity;
    # a few padded sizes let chunks with nearby spike counts share a kernel.
    block_size = min(block_size, padded_spike_count(n_decoding_spikes, block_size))

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
            cast(jax.Array, block),
            encoding_spike_waveform_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
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


def mark_reference_scale(
    waveform_std: float | np.ndarray = DEFAULT_WAVEFORM_STD, **_fit_params: object
) -> float | np.ndarray:
    """Mark-kernel width the reference adds in quadrature to each feature's spread.

    Parameters
    ----------
    waveform_std : float or array_like, shape (n_features,), optional
        The fit's mark bandwidth.
    **_fit_params
        Other fit parameters, ignored.

    Returns
    -------
    scale : float or array_like
        ``waveform_std``.
    """
    return waveform_std


class ClusterlessKDEEncodingModel(ClusterlessEncodingModel):
    """The encoding model ``fit_clusterless_kde_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_clusterless_kde_log_likelihood``, documented there.
    """

    occupancy: np.ndarray
    log_occupancy: jax.Array
    occupancy_model: KDEModel
    gpi_models: list[KDEModel]
    encoding_spike_waveform_features: list[np.ndarray]
    encoding_positions: list[np.ndarray]
    encoding_weights: list[jax.Array]
    summed_ground_process_intensity: np.ndarray
    position_std: jax.Array
    waveform_std: float
    block_size: int
    enc_tile_size: int | None
    pos_tile_size: int | None
    use_streaming: bool


def fit_clusterless_kde_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    position_std: float | jax.Array = np.sqrt(12.5),
    waveform_std: float = DEFAULT_WAVEFORM_STD,
    block_size: int = 10_000,
    enc_tile_size: int | None = None,
    pos_tile_size: int | None = None,
    use_streaming: bool = False,
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> ClusterlessKDEEncodingModel:
    """Fit the clusterless KDE encoding model.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Time of each position sample.
    position : array_like, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times for each electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Spike waveform features for each electrode.
    environment : Environment
        The spatial environment.
    weights : array_like, shape (n_time_position,), optional
        Per-position-sample weights (e.g. the EM posterior for this encoding
        group), by default None (uniform). They weight the occupancy KDE, the
        per-electrode ground-process KDE, and the mean rate, and are interpolated
        onto each spike's time to form the ``encoding_weights`` used at decode
        time. A spike sits at its interpolated position, except where its two
        bracketing samples differ in weight: there it sits at the mean of their
        positions weighted by its share of each sample's weight, so a spike
        next to an unweighted sample sits on its weighted sample, where that
        interval's occupancy is.
    position_std : float, optional
        Gaussian smoothing standard deviation for position, by default sqrt(12.5).
    waveform_std : float, optional
        Gaussian smoothing standard deviation for waveform, by default 24.0.
    block_size : int, optional
        Most decoding spikes per block, by default 10,000, matching the
        detectors' default. It is stored with the model and used at
        prediction (see ``predict_clusterless_kde_log_likelihood``).
    enc_tile_size : int, optional
        Encoding samples per reduction tile at predict time (capped at 2048),
        bounding the workspace by the tile rather than the encoding count. By
        default None (one tile of up to 2048 samples).
    pos_tile_size : int, optional
        Position bins per tile at predict time. By default None (all bins).
    use_streaming : bool, optional
        Compute each encoding tile's position kernel when it is used instead of
        the full ``(n_encoding, n_position_bins)`` kernel; requires
        ``enc_tile_size``. By default False.
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False.

    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds, replacing the outer bounds of the
        encoding support. A range inside the samples clips it, using the
        original interpolation basis; a range past them extends it, holding
        position at the first or last sample. Uniform samples otherwise include
        endpoint half-cells; a singleton requires explicit bounds or a tracking
        interval.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
        Required for irregular timestamps; each interval needs a finite sample.
        Positions are held at segment endpoints and NaN rows split support.
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support. Direct callers
        omit it and the support is built from the other arguments.

    Returns
    -------
    encoding_model : ClusterlessKDEEncodingModel
        Keyword arguments for ``predict_clusterless_kde_log_likelihood``,
        including ``algorithm`` (``"clusterless_kde"``), ``rate_units``
        (``"Hz"``), ``encoding_exposure_seconds`` (weighted encoding occupancy
        time in seconds) and ``mark_dimensions`` (each electrode's
        waveform-feature count).

    Notes
    -----
    Fitted mean rates and ground-process intensities are Hz. Occupancy weights
    integrate the interpolation basis over valid encoding support in seconds;
    event weights are dimensionless. Decode bins may differ from tracking samples.
    """
    _validate_execution_settings(
        block_size, enc_tile_size, pos_tile_size, use_streaming
    )
    position_time = as_array(position_time)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    # Weighted occupancy time in seconds: each sample's weight times the
    # tracking time it covers (its integrated interpolation exposure). Gaps from
    # is_training / encoding-group masks carry weight 0 and are not charged as
    # occupancy time.
    weight_sum = float(exposure_weights.sum())
    warn_if_no_encoding_exposure(exposure_weights, "Clusterless KDE", clusterless=True)
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
    warn_if_tracking_is_sparse(support, position, position_std)
    for features in spike_waveform_features:
        validate_bandwidth(waveform_std, features.shape[1], "waveform_std")

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    occupancy_samples, occupancy_weights = drop_zero_weight_samples(
        position, exposure_weights
    )
    occupancy_samples = environment.decoding_position(occupancy_samples)
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
        # Each encoding spike, weighted by the posterior weight at its spike
        # time, sits at its interpolated position or, where its bracketing
        # samples differ in weight, at their share-weighted mean.
        spike_positions, electrode_weights_host = place_encoding_spikes(
            position_time,
            position,
            electrode_spike_times,
            weights,
            environment,
            support,
        )
        electrode_weights = jnp.asarray(electrode_weights_host)
        encoding_weights.append(electrode_weights)
        # Weighted mean rate: weighted spike count / weighted occupancy time. Sum on the
        # host array (as the GMM fit does) to avoid a device->host sync per electrode.
        mean_rates.append(weighted_mean_rate(electrode_weights_host, weight_sum))
        encoding_positions.append(spike_positions)

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1], weights=electrode_weights
        )
        gpi_models.append(gpi_model)

    # One tiled pass for every electrode, whose kernels compile for the total
    # number of sample tiles, not for each electrode's sample count.
    for mean_rate, log_gpi_density in zip(
        mean_rates,
        kde_log_densities(gpi_models, interior_place_bin_centers),
        strict=True,
    ):
        summed_ground_process_intensity += jnp.exp(
            log_density_intensity(mean_rate, log_gpi_density, log_occupancy)
        )

    return {
        "algorithm": "clusterless_kde",
        "occupancy": np.asarray(occupancy),
        "log_occupancy": log_occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "mark_dimensions": [
            int(np.shape(features)[1]) for features in bounded_spike_waveform_features
        ],
        "encoding_positions": encoding_positions,
        "encoding_weights": encoding_weights,
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "mean_rates": np.asarray(mean_rates),
        "summed_ground_process_intensity": np.asarray(summed_ground_process_intensity),
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
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    *,
    occupancy: ArrayLike,
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jax.Array],
    encoding_positions: list[jax.Array],
    environment: Environment,
    mean_rates: np.ndarray,
    summed_ground_process_intensity: np.ndarray,
    position_std: jax.Array,
    waveform_std: jax.Array,
    is_local: bool = False,
    block_size: int = 10_000,
    disable_progress_bar: bool = False,
    enc_tile_size: int | None = None,
    pos_tile_size: int | None = None,
    use_streaming: bool = False,
    encoding_weights: list[jax.Array] | None = None,
    row_slice: slice | None = None,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    mark_dimensions: list[int] | None = None,  # noqa: ARG001 -- read by the detector
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    log_occupancy: jax.Array | None = None,
    environment_grid: dict[str, Any] | None = None,
    mark_reference: list[MarkReference | None],
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Predict the log likelihood of the clusterless KDE model.

    Parameters
    ----------
    position_time : array_like or None, shape (n_time_position,)
        Time of each position sample (used only by the local path; accepted for
        signature parity when ``is_local`` is False).
        May be None when ``is_local`` is False.
    position : array_like or None, shape (n_time_position, n_position_dims)
        Position samples (used only by the local path; accepted for signature
        parity when ``is_local`` is False).
        May be None when ``is_local`` is False.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times for each electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Waveform features for each electrode.
    occupancy : array_like, shape (n_interior_bins,)
        Occupancy density at the interior place bins (the fit's
        ``exp(log_occupancy)``). The non-local path reads ``log_occupancy``
        instead and uses this only for the bin count; the local path ignores it.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jax.Array]
        Spike waveform features for each electrode used for encoding.
    encoding_positions : list[jax.Array]
        Per-electrode encoding positions, each of shape
        (n_encoding_spikes, n_position_dims).
    environment : Environment
        The spatial environment.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Mean firing rate for each electrode.
    summed_ground_process_intensity : np.ndarray, shape (n_interior_bins,)
        Summed ground process intensity of the encoding model at the interior
        bins, in Hz; predict adds each electrode's background rate.
    position_std : jax.Array
        Gaussian smoothing standard deviation for position.
    waveform_std : jax.Array
        Gaussian smoothing standard deviation for waveform.
    is_local : bool, optional
        If True, compute the log likelihood at the animal's position, by default False.
    block_size : int, optional
        Most decoding spikes per block, by default 10,000, matching the
        detectors' default. Without ``enc_tile_size``, one pass reduces every
        electrode in blocks of one size: fewer spikes when padding each
        electrode's spikes to whole blocks would add more than a quarter, or
        when a block's buffers would exceed the kernel budget.
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False.
    enc_tile_size : int | None, optional
        Encoding samples are always reduced in tiles of at most 2048 rows; a
        smaller value lowers peak memory further and selects the block-wise path.
        By default None.
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension to reduce memory.
        By default None (no tiling).
    use_streaming : bool, optional
        If True, compute each encoding tile's position kernel when it is used
        instead of materializing the full ``(n_encoding_spikes, n_interior_bins)``
        position kernel. Requires ``enc_tile_size``; an electrode with at most
        ``enc_tile_size`` encoding spikes materializes its (small) kernel
        instead. By default False.
    encoding_weights : list[jax.Array] | None, optional
        Per-electrode encoding-spike weights (the ``encoding_weights`` returned by
        the fit). Each entry has shape (n_encoding_spikes,) and weights that
        electrode's joint mark intensity. By default None (uniform weights).
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges: spikes are binned
        against them and only those owned by the requested rows are evaluated, so
        the result equals the full-time result sliced by ``row_slice`` while the
        spatial workspaces scale with the requested rows and selected spikes.
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    algorithm : str, optional
        The algorithm the fit recorded. Accepted so the encoding dictionary
        can be passed as keywords; unused.
    mark_dimensions : list of int, optional
        Each electrode's fitted waveform-feature count, which the detector
        checks the decoded features against; unused here.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the encoding fit; anything other than ``"Hz"``
        raises ``ValidationError`` (refit legacy encoding models).
    encoding_exposure_seconds : float, optional, keyword-only
        Weighted encoding occupancy time in seconds, recorded by the fit.
        Accepted so the encoding dictionary can be passed as keywords; unused.
    log_occupancy : jax.Array, shape (n_interior_bins,), optional, keyword-only
        Log occupancy density at the interior bins, as returned by the fit.
        Used by the non-local path, which computes it from ``occupancy_model``
        when None; the local path ignores it.
    environment_grid : dict | None, optional
        The fit's record of the environment's grid (see
        ``environment_grid_fingerprint``), by default None, which skips the
        check (a model fitted before the grid was recorded). Prediction raises
        if the environment has since been refit to another grid.
    mark_reference : list of MarkReference or None, keyword-only
        Normalized reference mark density per electrode, from
        ``fit_mark_references`` (a detector passes its ``mark_reference_``).
        Each spike's intensity gains ``RATE_EPS_HZ * q(mark)`` and each
        electrode's expected count ``RATE_EPS_HZ``, so likelihood ratios between
        states do not depend on the waveform units. None for an electrode means
        no training marks; its background is the bare rate.
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, 1) or (n_rows, n_interior_bins)
        Shape depends on whether local or non-local decoding, respectively.
        ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    _validate_execution_settings(
        block_size, enc_tile_size, pos_tile_size, use_streaming
    )
    # The mark-intensity kernels below are JAX-side code that takes a device
    # array; convert the fit's NumPy occupancy once here rather than in each.
    occupancy = jnp.asarray(occupancy)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    # The fitted density models are evaluated in the requested blocks.
    occupancy_model = kde_model_with_block_size(
        occupancy_model, block_size, rows.spike_time_order
    )
    gpi_models = [
        kde_model_with_block_size(model, block_size, rows.spike_time_order)
        for model in gpi_models
    ]
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
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
    # Uniform (None) weights per electrode when the caller passes none, so the loop
    # and the local path can zip a weight per electrode uniformly.
    per_electrode_weights: Sequence[jax.Array | None] = (
        encoding_weights
        if encoding_weights is not None
        else cast(
            Sequence[jax.Array | None], [None] * len(encoding_spike_waveform_features)
        )
    )

    if is_local:
        position_time, position = require_position(
            position_time, position, "local clusterless KDE likelihood"
        )
        log_likelihood = compute_local_log_likelihood(
            rows.time_edges,
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
            disable_progress_bar=disable_progress_bar,
            encoding_weights=per_electrode_weights,
            row_slice=row_slice,
            mark_reference=mark_reference,
            _spike_time_order=rows.spike_time_order,
            _time_grid=rows.time_grid,
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]
        if log_occupancy is None:
            log_occupancy = kde_log_density(occupancy_model, interior_place_bin_centers)

        batched = not use_streaming and enc_tile_size is None
        collected: list[ElectrodeSpikes] = []

        def add_electrode(
            log_likelihood: jax.Array,
            spikes: ElectrodeSpikes,
            model: tuple[jax.Array, jax.Array, Any, jax.Array | None],
        ) -> jax.Array:
            if batched:
                # Every electrode is reduced in one pass below.
                collected.append(spikes)
                return log_likelihood
            encoding_features, encoding_positions, mean_rate, encoding_weights = model
            # Expand waveform_std to match this electrode's feature count if scalar
            electrode_waveform_std = as_std_array(
                waveform_std, encoding_features.shape[1]
            )
            # Padded zero-weight encoding samples let electrodes and chunks
            # with nearby sizes share one compiled kernel.
            padded_features, padded_weights = padded_samples(
                rows.spike_time_order, encoding_features, encoding_weights
            )
            padded_positions, _ = padded_samples(
                rows.spike_time_order, encoding_positions, encoding_weights
            )
            # Compute position kernel in log-space to prevent underflow
            # (Skip if using streaming mode - computed on-the-fly)
            if use_streaming:
                log_position_distance = None
            else:
                log_position_distance = log_kde_distance(
                    interior_place_bin_centers,
                    padded_positions,
                    std=position_std,
                )

            return log_likelihood + sum_spikes_into_rows(
                block_estimate_log_joint_mark_intensity(
                    spikes.marks,
                    padded_features,
                    electrode_waveform_std,
                    occupancy,
                    mean_rate * RATE_REFERENCE_SECONDS,
                    log_position_distance,
                    block_size=block_size,
                    enc_tile_size=enc_tile_size,
                    pos_tile_size=pos_tile_size,
                    use_streaming=use_streaming,
                    encoding_positions=padded_positions if use_streaming else None,
                    position_eval_points=(
                        interior_place_bin_centers if use_streaming else None
                    ),
                    position_std=position_std if use_streaming else None,
                    encoding_weights=padded_weights,
                    log_occupancy=log_occupancy,
                    log_background=spikes.log_background,
                ),
                spikes.selection,
            )

        log_likelihood = add_electrode_spike_terms(
            -jnp.asarray(rows.durations())[:, None]
            * (summed_ground_process_intensity + background_rate(len(spike_times))),
            spike_times,
            spike_waveform_features,
            mark_reference,
            list(
                zip(
                    encoding_spike_waveform_features,
                    encoding_positions,
                    mean_rates,
                    per_electrode_weights,
                    strict=True,
                )
            ),
            add_electrode,
            rows=rows,
            log_background_floor=LOG_EPS,  # log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS)
            desc="Position-Bin Likelihood",
            disable_progress_bar=disable_progress_bar,
        )
        if collected:
            tiles, electrodes, n_tiles = _encoding_tiles(
                rows.spike_time_order,
                encoding_spike_waveform_features,
                encoding_positions,
                per_electrode_weights,
                waveform_std,
            )
            widest_tiles = int(n_tiles.max())
            block_rows, group_blocks, group_tiles = _nonlocal_sizes(
                [spikes.n_spikes for spikes in collected],
                [int(n_tiles[spikes.electrode]) for spikes in collected],
                block_size,
                interior_place_bin_centers.shape[0],
                log_likelihood.dtype.itemsize,
                widest_tiles,
                _expected_chunk_spikes(rows, spike_times, n_tiles),
            )
            blocks, _ = _spike_blocks(
                [
                    (
                        spikes.electrode,
                        spikes.marks,
                        spikes.log_background,
                        spike_row_ids(spikes.selection),
                    )
                    for spikes in collected
                ],
                log_likelihood.shape[0],
                [np.shape(f)[1] for f in encoding_spike_waveform_features],
                block_rows,
                pad_with_last=True,
            )
            log_likelihood = _add_log_mark_intensities(
                log_likelihood,
                blocks,
                tiles,
                electrodes,
                jnp.asarray(np.asarray(mean_rates, float) * RATE_REFERENCE_SECONDS),
                jnp.asarray(position_std),
                jnp.asarray(interior_place_bin_centers),
                jnp.asarray(log_occupancy),
                widest_tiles=widest_tiles,
                group_blocks=group_blocks,
                group_tiles=group_tiles,
                pos_tile_size=pos_tile_size,
                indices_are_sorted=all(
                    spikes.selection.indices_are_sorted for spikes in collected
                ),
            )

    return (
        log_likelihood
        + log_bin_duration_evidence(
            spike_times,
            rows.time_edges,
            row_slice,
            rows.spike_time_order,
            intensity_time_scale=RATE_REFERENCE_SECONDS,
            _time_grid=rows.time_grid,
        )[:, None]
    )


def _log_occupancy_at_spikes(
    positions: Sequence[np.ndarray],
    occupancy_samples: jax.Array,
    occupancy_std: jax.Array,
    occupancy_block_size: int | None,
    occupancy_weights: jax.Array,
    block_size: int,
) -> list[np.ndarray]:
    """Evaluate the true log occupancy density at every electrode's spikes in one pass.

    The electrodes' spike positions are evaluated together, padded to few
    sizes (see ``padded_spike_count``), so the density kernel compiles for
    the chunk's total spike count, not for each electrode's.

    Parameters
    ----------
    positions : sequence of np.ndarray, each shape (n_spikes, n_position_dims)
        Each electrode's position at its decoding spikes.
    occupancy_samples, occupancy_weights : jax.Array
        Padded occupancy samples and weights (see ``padded_samples``).
    occupancy_std : jax.Array, shape (n_position_dims,)
        Occupancy kernel bandwidth.
    occupancy_block_size : int or None
        The occupancy model's evaluation points per block; None evaluates
        the padded spikes in one block.
    block_size : int
        Decoding spikes per block, which sets the padded sizes.

    Returns
    -------
    log_occupancy : list of np.ndarray, each shape (n_spikes,)
        In the order of ``positions``.
    """
    if not positions:
        return []
    n_spikes = [electrode_positions.shape[0] for electrode_positions in positions]
    n_valid = sum(n_spikes)
    n_padded = padded_spike_count(n_valid, block_size)
    values = np.asarray(
        true_kde_log_density(
            jnp.asarray(pad_rows(np.concatenate(positions), n_padded)),
            occupancy_samples,
            occupancy_std,
            occupancy_block_size or n_padded,
            occupancy_weights,
            # Zero padding rows can lie far from every occupancy sample;
            # their values are dropped, so they are not repaired.
            n_valid=n_valid,
        )
    )
    return np.split(values[:n_valid], np.cumsum(n_spikes)[:-1])


def compute_local_log_likelihood(
    time_edges: np.ndarray,
    position_time: np.ndarray | jax.Array,
    position: np.ndarray | jax.Array,
    spike_times: Sequence[np.ndarray | jax.Array],
    spike_waveform_features: Sequence[np.ndarray | jax.Array],
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jax.Array],
    encoding_positions: list[jax.Array],
    environment: Environment,
    mean_rates: np.ndarray,
    position_std: jax.Array,
    waveform_std: jax.Array,
    block_size: int = 10_000,
    *,
    disable_progress_bar: bool = False,
    encoding_weights: Sequence[jax.Array | None] | None = None,
    row_slice: slice | None = None,
    mark_reference: list[MarkReference | None],
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Compute the log likelihood at the animal's position.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : np.ndarray or jax.Array, shape (n_time_position,)
        Time of each position sample.
    position : np.ndarray or jax.Array, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of np.ndarray or jax.Array
        Spike times for each electrode.
    spike_waveform_features : sequence of np.ndarray or jax.Array
        Spike waveform features for each electrode.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        List of KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jax.Array]
        List of spike waveform features for each electrode used for encoding.
    encoding_positions : list[jax.Array]
        Per-electrode encoding positions, each of shape
        (n_encoding_spikes, n_position_dims).
    environment : Environment
        The spatial environment.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Mean firing rate for each electrode.
    position_std : jax.Array
        Gaussian smoothing standard deviation for position.
    waveform_std : jax.Array
        Gaussian smoothing standard deviation for waveform.
    block_size : int, optional
        Most decoding spikes per block, by default 10,000, matching the
        detectors' default (see ``predict_clusterless_kde_log_likelihood``).
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False.
    encoding_weights : list[jax.Array] | None, optional
        Per-electrode encoding-spike weights, each shape (n_encoding_spikes,). By
        default None (uniform); weight the local marginal density KDE per electrode.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` stay the FULL decoding edges (see
        ``predict_clusterless_kde_log_likelihood``).
    mark_reference : list of MarkReference or None
        Reference mark density per electrode for the background intensity (see
        ``predict_clusterless_kde_log_likelihood``).
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, 1)
        Log likelihood at the animal's position for each requested decoding bin.
    """
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    n_rows = rows.stop - rows.start

    # Normalize to a per-electrode list; None -> uniform weights for each electrode.
    if encoding_weights is None:
        encoding_weights = [None] * len(encoding_positions)

    position = position if position.ndim > 1 else position[:, np.newaxis]
    interpolated_position, log_occupancy = prepare_kde_local_inputs(
        occupancy_model,
        position_time,
        position,
        environment,
        rows.time_grid,
        rows.start,
        rows.stop,
        rows.spike_time_order,
    )

    n_position_dims = np.shape(interpolated_position)[1]
    occupancy_std = as_std_array(occupancy_model.std, n_position_dims)
    occupancy_samples, occupancy_weights = padded_samples(
        rows.spike_time_order,
        cast(jax.Array, occupancy_model.samples_),
        occupancy_model.weights_,
        count=padded_occupancy_count,
    )
    selections = [
        select_spikes_in_rows(
            electrode_spike_times,
            rows.start,
            rows.stop,
            time_edges=rows.time_edges,
            _spike_time_order=rows.spike_time_order,
        )
        for electrode_spike_times in spike_times
    ]
    positions_at_spike_times = [
        np.asarray(
            snap_off_track_positions(
                get_position_at_time(
                    position_time,
                    position,
                    select_spike_rows(electrode_spike_times, selection),
                    environment,
                ),
                environment,
            )
        )
        for electrode_spike_times, selection in zip(
            spike_times, selections, strict=True
        )
    ]
    # Every electrode's occupancy at its spikes, ground-process density at the
    # rows and spike terms, each in one pass: their kernels compile for the
    # chunk, not for each electrode's spike, encoding or feature count.
    occupancy_at_spikes = _log_occupancy_at_spikes(
        positions_at_spike_times,
        occupancy_samples,
        occupancy_std,
        occupancy_model.block_size,
        occupancy_weights,
        block_size,
    )
    log_gpi_densities = prepared_kde_log_densities(
        gpi_models, interpolated_position, rows.spike_time_order
    )
    spikes = []
    for electrode, (
        decoding_features,
        selection,
        electrode_mark_reference,
    ) in enumerate(
        zip(
            tqdm(
                spike_waveform_features,
                unit="electrode",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            selections,
            mark_reference,
            strict=True,
        )
    ):
        marks = select_decode_marks(decoding_features, selection, electrode)
        if marks.shape[0]:
            spikes.append(
                (
                    electrode,
                    marks,
                    background_log_intensity(
                        electrode_mark_reference,
                        marks,
                        LOG_EPS,  # log(RATE_EPS_HZ * RATE_REFERENCE_SECONDS)
                    ),
                    spike_row_ids(selection),
                )
            )
    log_likelihood = jnp.zeros((n_rows,))
    if spikes:
        tiles, electrodes, n_tiles = _encoding_tiles(
            rows.spike_time_order,
            encoding_spike_waveform_features,
            encoding_positions,
            encoding_weights,
            waveform_std,
        )
        block_rows, group_blocks, group_tiles = _local_sizes(
            [marks.shape[0] for _, marks, _, _ in spikes],
            [int(n_tiles[electrode]) for electrode, *_ in spikes],
            block_size,
            int(n_tiles.max()),
            _expected_chunk_spikes(rows, spike_times, n_tiles),
        )
        blocks, (block_positions, block_log_occupancy) = _spike_blocks(
            spikes,
            n_rows,
            [np.shape(f)[1] for f in encoding_spike_waveform_features],
            block_rows,
            pad_with_last=False,
            per_spike=(
                [positions_at_spike_times[electrode] for electrode, *_ in spikes],
                [occupancy_at_spikes[electrode] for electrode, *_ in spikes],
            ),
        )
        log_likelihood = _add_local_terms(
            log_likelihood,
            blocks._replace(
                positions=jnp.asarray(block_positions),
                log_occupancy=jnp.asarray(block_log_occupancy),
            ),
            tiles,
            electrodes,
            jnp.asarray(np.asarray(mean_rates, float) * RATE_REFERENCE_SECONDS),
            jnp.asarray(position_std),
            group_blocks=group_blocks,
            group_tiles=group_tiles,
            indices_are_sorted=all(
                selection.indices_are_sorted for selection in selections
            ),
        )
    summed_expected_counts = _summed_expected_counts(
        jnp.asarray(np.asarray(mean_rates, float)),
        jnp.stack([jnp.asarray(density) for density in log_gpi_densities]),
        jnp.asarray(log_occupancy),
    )
    log_likelihood -= jnp.asarray(rows.durations()) * (
        summed_expected_counts + background_rate(len(spike_times))
    )
    return cast(jax.Array, log_likelihood[:, jnp.newaxis])
