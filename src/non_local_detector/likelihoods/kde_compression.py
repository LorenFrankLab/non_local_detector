"""Smaller weighted encoding dictionaries for the clusterless KDE.

A clusterless KDE places one joint (position, mark) kernel on every encoding
spike, so its prediction cost grows with the recording. These functions
replace one electrode's encoding spikes with a smaller weighted dictionary
whose weighted kernel sum estimates the full one:

- ``"uniform"``: a uniform random subset, each kept spike weighted by the
  inverse of its inclusion probability.
- ``"stratified"``: Pareto πps sampling [2]_ whose inclusion
  probabilities fall as the inverse square root of the spike's stratum size,
  so rare marks and places with few spikes are kept more often than a uniform
  subset keeps them. Strata are position cells of two position bandwidths
  crossed with k-means cells of the bandwidth-scaled marks. Every spike keeps
  a nonzero inclusion probability, and the kept spikes are weighted by its
  inverse, which keeps the weighted sum approximately unbiased.
- ``"merge"``: the kernel compression of Sodkomkham et al. [1]_. Spikes are
  visited once in a seeded random order. Each joins the nearest existing
  kernel when its Mahalanobis distance to that kernel, in the kernel's own
  bandwidths over position and marks, is at most ``compression_threshold``;
  otherwise it starts a kernel of the base bandwidths. A merge replaces the
  two kernels by one with their summed weight and their combined mean and
  per-dimension variance (moment matching), so every kernel ends up with its
  own diagonal bandwidth. Larger thresholds give fewer kernels.

  With ``compression_rank`` r > 0, an addition to the paper's method, each
  kernel instead keeps the part of its moment-matched joint (position and
  mark) covariance in excess of the base bandwidths as a diagonal plus a
  rank-r factor, ``diag(base + e) + U U^T``. After every merge the excess is
  projected back to rank r and the dropped direction's variance moves to the
  diagonal, so each dimension's variance is kept exactly. A merge adds one
  direction, so the joint covariance is tracked exactly only at a rank of at
  least ``n_position_dims + n_features``. Distances use that covariance. The
  kernel to merge into is the nearest by full Mahalanobis distance among the
  ``_LOW_RANK_SHORTLIST`` nearest by diagonal distance. The returned kernels
  keep the mark covariance and each kernel's position variance; the
  position-mark covariance is dropped, so position factors out of the kernel
  as for the diagonal merge.

The dictionary leaves the electrode's mean rate and the encoding exposure to
the caller, which computes them from every spike: a smaller dictionary must
not look like a shorter recording.

References
----------
.. [1] Sodkomkham, D., Ciliberti, D., Wilson, M. A., Fukui, K., Moriyama, K.,
   Numao, M., & Kloosterman, F. (2016). Kernel density compression for
   real-time Bayesian encoding/decoding of unsorted hippocampal spikes.
   Knowledge-Based Systems, 94, 1-12.
   https://doi.org/10.1016/j.knosys.2015.09.013
.. [2] Rosén, B. (1997). On sampling with probability proportional to size.
   Journal of Statistical Planning and Inference, 62(2), 159-191.
"""

from functools import partial
from typing import Literal, NamedTuple, get_args

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans

from non_local_detector.exceptions import ValidationError

SubsamplingMethod = Literal["uniform", "stratified"]
CompressionMethod = Literal["uniform", "stratified", "merge"]
COMPRESSION_METHODS: tuple[str, ...] = get_args(CompressionMethod)
# Merge threshold when compression_threshold is None.
DEFAULT_COMPRESSION_THRESHOLD = 1.0

# k-means cells over the bandwidth-scaled marks, one stratification axis.
_MARK_STRATA = 32
# Position strata are cells this many position bandwidths wide.
_POSITION_STRATUM_WIDTH = 2.0
# Low-rank merge: kernels nearest by diagonal distance whose full Mahalanobis
# distance to a new spike is evaluated.
_LOW_RANK_SHORTLIST = 32


def validate_compression(
    encoding_compression: str | None,
    n_encoding_kernels: object,
    compression_threshold: object,
    compression_rank: object = 0,
) -> None:
    """Raise ``ValidationError`` for an unknown method or invalid setting.

    A setting the method would ignore is invalid, so a forgotten
    ``encoding_compression`` cannot silently fit the full dictionary.

    Parameters
    ----------
    encoding_compression : str or None
    n_encoding_kernels : object
        A positive integer for ``"uniform"`` and ``"stratified"``; None
        otherwise (``"merge"`` sizes its dictionary by the threshold).
    compression_threshold : object
        None, or for ``"merge"`` a finite, non-negative number.
    compression_rank : object, optional
        A non-negative integer; nonzero only for ``"merge"``. By default 0.
        Its upper bound depends on each electrode's dimensions, which the fit
        checks.
    """
    if (
        isinstance(compression_rank, (bool, np.bool_))
        or not isinstance(compression_rank, (int, np.integer))
        or compression_rank < 0
    ):
        raise ValidationError(
            "compression_rank must be a non-negative integer",
            got=repr(compression_rank),
        )
    if compression_rank > 0 and encoding_compression != "merge":
        raise ValidationError(
            "compression_rank applies to merged kernels only",
            expected="encoding_compression='merge'",
            got=repr(encoding_compression),
        )
    if compression_threshold is not None and encoding_compression != "merge":
        raise ValidationError(
            "compression_threshold applies to merged kernels only",
            expected="encoding_compression='merge'",
            got=repr(encoding_compression),
        )
    if encoding_compression is None:
        if n_encoding_kernels is not None:
            raise ValidationError(
                "n_encoding_kernels needs a subsampling method",
                expected="encoding_compression='uniform' or 'stratified'",
                got="encoding_compression=None",
            )
        return
    if encoding_compression not in COMPRESSION_METHODS:
        raise ValidationError(
            "Unknown encoding_compression",
            expected=f"None or one of {COMPRESSION_METHODS}",
            got=repr(encoding_compression),
        )
    if encoding_compression == "merge":
        if n_encoding_kernels is not None:
            raise ValidationError(
                "encoding_compression='merge' sets its size by compression_threshold",
                expected="n_encoding_kernels=None",
                got=repr(n_encoding_kernels),
            )
        if compression_threshold is not None and (
            isinstance(compression_threshold, (bool, np.bool_))
            or not isinstance(compression_threshold, (int, float, np.number))
            or not np.isfinite(compression_threshold)
            or compression_threshold < 0
        ):
            raise ValidationError(
                "compression_threshold must be finite and non-negative",
                got=repr(compression_threshold),
            )
        return
    if (
        isinstance(n_encoding_kernels, (bool, np.bool_))
        or not isinstance(n_encoding_kernels, (int, np.integer))
        or n_encoding_kernels < 1
    ):
        raise ValidationError(
            f"encoding_compression={encoding_compression!r} needs a dictionary size",
            expected="n_encoding_kernels, a positive integer",
            got=repr(n_encoding_kernels),
        )


def _inclusion_probabilities(sizes: np.ndarray, n_kernels: int) -> np.ndarray:
    """Probabilities ``min(1, c * sizes**-0.5)`` summing to ``n_kernels``.

    Parameters
    ----------
    sizes : np.ndarray, shape (n_spikes,)
        Size of each spike's stratum.
    n_kernels : int
        Less than ``n_spikes``.

    Returns
    -------
    probabilities : np.ndarray, shape (n_spikes,)
    """
    scores = 1.0 / np.sqrt(sizes)
    # sum(min(1, c * scores)) increases with c; bisect in log c.
    low, high = np.log(n_kernels / scores.sum()), np.log(n_kernels / scores.min())
    for _ in range(100):
        middle = 0.5 * (low + high)
        if np.minimum(1.0, np.exp(middle) * scores).sum() < n_kernels:
            low = middle
        else:
            high = middle
    return np.minimum(1.0, np.exp(high) * scores)


def _strata(
    positions: np.ndarray,
    marks: np.ndarray,
    position_std: np.ndarray,
    waveform_std: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Stratum of each spike: position cell crossed with mark k-means cell.

    Returns
    -------
    strata : np.ndarray of int, shape (n_spikes,)
    """
    position_cells = np.floor(
        (positions - positions.min(axis=0)) / (_POSITION_STRATUM_WIDTH * position_std)
    ).astype(np.int64)
    n_clusters = min(_MARK_STRATA, marks.shape[0])
    mark_cells = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit_predict(
        marks / waveform_std
    )
    return np.unique(
        np.column_stack([position_cells, mark_cells]), axis=0, return_inverse=True
    )[1].ravel()


def compress_encoding_spikes(
    positions: np.ndarray,
    marks: np.ndarray,
    weights: np.ndarray,
    method: SubsamplingMethod,
    n_kernels: int,
    position_std: np.ndarray,
    waveform_std: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted dictionary standing in for one electrode's encoding spikes.

    Zero-weight spikes carry no kernel mass and are dropped first. If at most
    ``n_kernels`` spikes remain they are all kept, unchanged.

    Parameters
    ----------
    positions : np.ndarray, shape (n_spikes, n_position_dims)
    marks : np.ndarray, shape (n_spikes, n_features)
    weights : np.ndarray, shape (n_spikes,)
        Non-negative spike weights.
    method : {"uniform", "stratified"}
    n_kernels : int
        Dictionary size.
    position_std : np.ndarray, shape (n_position_dims,)
    waveform_std : np.ndarray, shape (n_features,)
        Kernel bandwidths; they scale the strata.
    seed : int, optional
        Seed of the sampling and of the mark k-means, by default 0.

    Returns
    -------
    positions : np.ndarray, shape (n_kept, n_position_dims)
    marks : np.ndarray, shape (n_kept, n_features)
    weights : np.ndarray, shape (n_kept,)
        Each kept spike's weight divided by its inclusion probability. Kept
        spikes stay in their original order.
    """
    if method not in get_args(SubsamplingMethod):
        raise ValidationError(
            "Unknown subsampling method",
            expected=f"one of {get_args(SubsamplingMethod)}",
            got=repr(method),
        )
    if n_kernels < 1:
        raise ValidationError(
            "The dictionary needs at least one kernel",
            expected="n_kernels >= 1",
            got=repr(n_kernels),
        )
    positions, marks, weights = (np.asarray(a) for a in (positions, marks, weights))
    keep = np.flatnonzero(weights > 0)
    if keep.size > n_kernels:
        rng = np.random.default_rng(seed)
        if method == "uniform":
            probabilities = np.full(keep.size, n_kernels / keep.size)
            kept = rng.choice(keep.size, n_kernels, replace=False)
        else:  # stratified
            strata = _strata(
                positions[keep], marks[keep], position_std, waveform_std, seed
            )
            probabilities = _inclusion_probabilities(
                np.bincount(strata)[strata], n_kernels
            )
            # Pareto sampling: the n_kernels smallest ranking variables.
            # Probabilities are positive and uniform < 1: no division by zero.
            uniform = rng.random(keep.size)
            ranking = (uniform * (1.0 - probabilities)) / (
                (1.0 - uniform) * probabilities
            )
            kept = np.argpartition(ranking, n_kernels - 1)[:n_kernels]
        kept = np.sort(kept)
        weights = weights.copy()
        weights[keep[kept]] /= probabilities[kept]
        keep = keep[kept]
    return positions[keep], marks[keep], weights[keep]


class MergedKernels(NamedTuple):
    """A merged dictionary: kernel centres, weights and per-kernel bandwidths.

    Attributes
    ----------
    positions : np.ndarray, shape (n_kernels, n_position_dims)
    marks : np.ndarray, shape (n_kernels, n_features)
    weights : np.ndarray, shape (n_kernels,)
    position_stds : np.ndarray, shape (n_kernels, n_position_dims)
        Square roots of each kernel's position variances; for a low-rank merge
        they include the factor's position part.
    mark_stds : np.ndarray, shape (n_kernels, n_features)
        Square roots of the diagonal of each kernel's mark covariance, less the
        low-rank part.
    mark_factors : np.ndarray, shape (n_kernels, n_features, rank), or None
        Low-rank merges only: each kernel's mark covariance is
        ``diag(mark_stds**2) + mark_factors @ mark_factors.T``.
    """

    positions: np.ndarray
    marks: np.ndarray
    weights: np.ndarray
    position_stds: np.ndarray
    mark_stds: np.ndarray
    mark_factors: np.ndarray | None = None


@jax.jit
def _merge_scan(samples, weights, base_variance, threshold):
    """Online moment-matching merge over a fixed-capacity kernel buffer.

    Parameters
    ----------
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Kernel centres in visiting order.
    weights : jnp.ndarray, shape (n_samples,)
        Positive weights.
    base_variance : jnp.ndarray, shape (n_dims,)
        Squared base bandwidths: the variance of each new kernel.
    threshold : float
        Largest Mahalanobis distance that merges.

    Returns
    -------
    means, variances : jnp.ndarray, shape (n_samples, n_dims)
    totals : jnp.ndarray, shape (n_samples,)
    n_kernels : jnp.ndarray, shape ()
        The first ``n_kernels`` rows are the dictionary.
    """
    n_samples = samples.shape[0]
    slots = jnp.arange(n_samples)

    def step(carry, sample):
        means, variances, totals, n_kernels = carry
        x, weight = sample
        distance_squared = jnp.sum((x - means) ** 2 / variances, axis=1)
        distance_squared = jnp.where(slots < n_kernels, distance_squared, jnp.inf)
        nearest = jnp.argmin(distance_squared)
        merges = distance_squared[nearest] <= threshold**2
        total = totals[nearest] + weight
        offset = x - means[nearest]
        # Moment matching in a form free of cancellation:
        # var = (w_a var_a + w_b var_b) / w + (w_a w_b / w**2) * offset**2.
        merged_mean = means[nearest] + (weight / total) * offset
        merged_variance = (
            totals[nearest] * variances[nearest] + weight * base_variance
        ) / total + (totals[nearest] * weight / total**2) * offset**2
        slot = jnp.where(merges, nearest, n_kernels)
        means = means.at[slot].set(jnp.where(merges, merged_mean, x))
        variances = variances.at[slot].set(
            jnp.where(merges, merged_variance, base_variance)
        )
        totals = totals.at[slot].set(jnp.where(merges, total, weight))
        return (means, variances, totals, n_kernels + jnp.where(merges, 0, 1)), None

    initial = (
        jnp.zeros_like(samples),
        jnp.ones_like(samples),  # unused slots; never divide by zero
        jnp.zeros_like(weights),
        jnp.asarray(0),
    )
    (means, variances, totals, n_kernels), _ = jax.lax.scan(
        step, initial, (samples, weights)
    )
    return means, variances, totals, n_kernels


def _low_rank_mahalanobis_squared(offsets, factors, variances):
    """Squared Mahalanobis distances to kernels of covariance diag(d) + U U^T.

    With ``z = offset / sqrt(d)`` and ``W = U / sqrt(d)``, the distance
    ``z^T (I + W W^T)^-1 z`` is the minimum over ``y`` of
    ``|z - W y|**2 + |y|**2``, reached at ``y = (I + W^T W)^-1 W^T z``. Both
    terms are non-negative, so nothing cancels; Woodbury's
    ``|z|**2 - |C^-1 W^T z|**2`` cancels for elongated kernels and rounds in
    float32 to small or negative distances that merge distant spikes.

    Parameters
    ----------
    offsets : jnp.ndarray, shape (n_kernels, n_dims)
    factors : jnp.ndarray, shape (n_kernels, n_dims, rank)
    variances : jnp.ndarray, shape (n_kernels, n_dims)
        The diagonal ``d`` of each kernel.

    Returns
    -------
    distance_squared : jnp.ndarray, shape (n_kernels,)
    """
    scale = jnp.sqrt(variances)
    z = offsets / scale
    whitened = factors / scale[..., None]
    inner = jnp.eye(factors.shape[-1], dtype=offsets.dtype) + jnp.einsum(
        "cdr,cds->crs", whitened, whitened
    )
    y = jnp.linalg.solve(inner, jnp.einsum("cdr,cd->cr", whitened, z)[..., None])
    residual = z - jnp.einsum("cdr,cr->cd", whitened, y[..., 0])
    return jnp.sum(residual**2, axis=-1) + jnp.sum(y[..., 0] ** 2, axis=-1)


@partial(jax.jit, static_argnames=("rank", "shortlist"))
def _low_rank_merge_scan(samples, weights, base_variance, threshold, rank, shortlist):
    """Online merge keeping each kernel's excess covariance as diag(e) + U U^T.

    Parameters
    ----------
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Kernel centres in visiting order.
    weights : jnp.ndarray, shape (n_samples,)
        Positive weights.
    base_variance : jnp.ndarray, shape (n_dims,)
        Squared base bandwidths: the covariance of each new kernel is
        ``diag(base_variance)``.
    threshold : float
        Largest Mahalanobis distance that merges.
    rank : int
        Columns of each kernel's factor ``U``.
    shortlist : int
        Kernels nearest by diagonal distance whose full distance is evaluated.

    Returns
    -------
    means : jnp.ndarray, shape (n_samples, n_dims)
    factors : jnp.ndarray, shape (n_samples, n_dims, rank)
    excess : jnp.ndarray, shape (n_samples, n_dims)
        Kernel covariance is ``diag(base_variance + excess) + U U^T``.
    totals : jnp.ndarray, shape (n_samples,)
    n_kernels : jnp.ndarray, shape ()
        The first ``n_kernels`` rows are the dictionary.
    """
    n_samples, n_dims = samples.shape
    slots = jnp.arange(n_samples)
    shortlist = min(shortlist, n_samples)
    # A factor of rank >= n_dims keeps every direction of the update.
    n_kept = min(rank, n_dims)

    def step(carry, sample):
        means, factors, excess, marginal_variances, totals, n_kernels = carry
        x, weight = sample
        diagonal_distance = jnp.sum((x - means) ** 2 / marginal_variances, axis=1)
        diagonal_distance = jnp.where(slots < n_kernels, diagonal_distance, jnp.inf)
        _, candidates = jax.lax.top_k(-diagonal_distance, shortlist)
        distance_squared = _low_rank_mahalanobis_squared(
            x - means[candidates],
            factors[candidates],
            base_variance[None] + excess[candidates],
        )
        distance_squared = jnp.where(
            jnp.isfinite(diagonal_distance[candidates]), distance_squared, jnp.inf
        )
        best = jnp.argmin(distance_squared)
        nearest = candidates[best]
        merges = distance_squared[best] <= threshold**2
        total = totals[nearest] + weight
        kept_share = totals[nearest] / total
        cross_share = totals[nearest] * weight / total**2
        offset = x - means[nearest]
        # The moment-matched excess is kept_share (diag(e) + U U^T)
        # + cross_share offset offset^T: a rank-(r + 1) update of the factor,
        # projected back to rank r with the dropped direction moved to the
        # diagonal, so the diagonal is kept exactly.
        stacked = jnp.concatenate(
            [
                jnp.sqrt(kept_share) * factors[nearest],
                jnp.sqrt(cross_share) * offset[:, None],
            ],
            axis=1,
        )
        left, singular, _ = jnp.linalg.svd(stacked, full_matrices=False)
        merged_factor = jnp.zeros((n_dims, rank), samples.dtype)
        merged_factor = merged_factor.at[:, :n_kept].set(
            left[:, :n_kept] * singular[:n_kept]
        )
        dropped = (
            (left[:, rank] * singular[rank]) ** 2
            if rank < singular.shape[0]
            else jnp.zeros(n_dims, samples.dtype)
        )
        merged_excess = kept_share * excess[nearest] + dropped
        merged_mean = means[nearest] + (weight / total) * offset
        slot = jnp.where(merges, nearest, n_kernels)
        slot_factor = jnp.where(merges, merged_factor, 0.0)
        slot_excess = jnp.where(merges, merged_excess, 0.0)
        means = means.at[slot].set(jnp.where(merges, merged_mean, x))
        factors = factors.at[slot].set(slot_factor)
        excess = excess.at[slot].set(slot_excess)
        marginal_variances = marginal_variances.at[slot].set(
            base_variance + slot_excess + jnp.sum(slot_factor**2, axis=1)
        )
        totals = totals.at[slot].set(jnp.where(merges, total, weight))
        carry = (
            means,
            factors,
            excess,
            marginal_variances,
            totals,
            n_kernels + jnp.where(merges, 0, 1),
        )
        return carry, None

    initial = (
        jnp.zeros_like(samples),
        jnp.zeros((n_samples, n_dims, rank), samples.dtype),
        jnp.zeros_like(samples),
        jnp.ones_like(samples),  # unused slots; never divide by zero
        jnp.zeros_like(weights),
        jnp.asarray(0),
    )
    (means, factors, excess, _, totals, n_kernels), _ = jax.lax.scan(
        step, initial, (samples, weights)
    )
    return means, factors, excess, totals, n_kernels


def merge_encoding_spikes(
    positions: np.ndarray,
    marks: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    position_std: np.ndarray,
    waveform_std: np.ndarray,
    seed: int = 0,
    rank: int = 0,
) -> MergedKernels:
    """Merge one electrode's joint kernels (Sodkomkham et al. 2016; see the module).

    Zero-weight spikes carry no kernel mass and are dropped first. The merged
    dictionary keeps the total weight and, for every dimension, the weighted
    mean and second moment of the original kernels.

    Parameters
    ----------
    positions : np.ndarray, shape (n_spikes, n_position_dims)
    marks : np.ndarray, shape (n_spikes, n_features)
    weights : np.ndarray, shape (n_spikes,)
        Non-negative spike weights.
    threshold : float
        Largest Mahalanobis distance, in the nearest kernel's own covariance
        (its bandwidths, plus its factor for ``rank`` > 0), at which a spike
        merges into it; 0 merges only coincident spikes.
    position_std : np.ndarray, shape (n_position_dims,)
    waveform_std : np.ndarray, shape (n_features,)
        Base bandwidths of each spike's kernel.
    seed : int, optional
        Seed of the visiting order, by default 0.
    rank : int, optional
        0 (the default) for diagonal kernels; otherwise the rank of each
        kernel's excess covariance factor (see the module docstring), at most
        ``n_position_dims + n_features``. At that rank the merges track each
        kernel's joint covariance exactly; the returned kernels still drop the
        position-mark covariance.

    Returns
    -------
    merged : MergedKernels
        ``mark_factors`` is None for ``rank=0``.
    """
    positions, marks, weights = (np.asarray(a) for a in (positions, marks, weights))
    keep = np.flatnonzero(weights > 0)
    n_position_dims, n_features = positions.shape[1], marks.shape[1]
    # The default float: float32, or float64 with x64 enabled.
    dtype = jax.dtypes.canonicalize_dtype(np.float64)
    if keep.size == 0:
        # No kernel mass: an empty dictionary, like the other methods give.
        return MergedKernels(
            positions=np.zeros((0, n_position_dims), dtype),
            marks=np.zeros((0, n_features), dtype),
            weights=np.zeros(0, dtype),
            position_stds=np.ones((0, n_position_dims), dtype),
            mark_stds=np.ones((0, n_features), dtype),
            mark_factors=None if rank == 0 else np.zeros((0, n_features, rank), dtype),
        )
    order = keep[np.random.default_rng(seed).permutation(keep.size)]
    samples = np.concatenate([positions[order], marks[order]], axis=1)
    base_std = np.concatenate(
        [
            np.broadcast_to(position_std, (n_position_dims,)),
            np.broadcast_to(waveform_std, (marks.shape[1],)),
        ]
    )
    if rank > 0:
        means, factors, excess, totals, n_kernels = _low_rank_merge_scan(
            jnp.asarray(samples, dtype),
            jnp.asarray(weights[order], dtype),
            jnp.asarray(base_std**2, dtype),
            threshold,
            rank,
            _LOW_RANK_SHORTLIST,
        )
        n_kernels = int(n_kernels)
        means = np.asarray(means[:n_kernels])
        factors = np.asarray(factors[:n_kernels])
        variances = np.asarray(base_std**2, dtype) + np.asarray(excess[:n_kernels])
        # Position keeps its marginal variance, factor included; the
        # position-mark covariance is dropped.
        position_variances = variances[:, :n_position_dims] + np.sum(
            factors[:, :n_position_dims] ** 2, axis=2
        )
        return MergedKernels(
            positions=means[:, :n_position_dims],
            marks=means[:, n_position_dims:],
            weights=np.asarray(totals[:n_kernels]),
            position_stds=np.sqrt(position_variances),
            mark_stds=np.sqrt(variances[:, n_position_dims:]),
            mark_factors=factors[:, n_position_dims:],
        )
    means, variances, totals, n_kernels = _merge_scan(
        jnp.asarray(samples, dtype),
        jnp.asarray(weights[order], dtype),
        jnp.asarray(base_std**2, dtype),
        threshold,
    )
    n_kernels = int(n_kernels)
    means = np.asarray(means[:n_kernels])
    stds = np.sqrt(np.asarray(variances[:n_kernels]))
    return MergedKernels(
        positions=means[:, :n_position_dims],
        marks=means[:, n_position_dims:],
        weights=np.asarray(totals[:n_kernels]),
        position_stds=stds[:, :n_position_dims],
        mark_stds=stds[:, n_position_dims:],
    )
