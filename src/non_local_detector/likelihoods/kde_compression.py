"""Smaller weighted encoding dictionaries for the clusterless KDE.

A clusterless KDE places one joint (position, mark) kernel on every encoding
spike, so its prediction cost grows with the recording. These functions
replace one electrode's encoding spikes with a smaller weighted dictionary
whose weighted kernel sum estimates the full one:

- ``"uniform"``: a uniform random subset, each kept spike weighted by the
  inverse of its inclusion probability.
- ``"stratified"``: Pareto πps sampling (Rosén 1997) whose inclusion
  probabilities fall as the inverse square root of the spike's stratum size,
  so rare marks and rarely visited places are kept more often than a uniform
  subset keeps them. Strata are position cells of two position bandwidths
  crossed with k-means cells of the bandwidth-scaled marks. Every spike keeps
  a nonzero inclusion probability, and the kept spikes are weighted by its
  inverse, which keeps the weighted sum approximately unbiased.

The dictionary leaves the electrode's mean rate and the encoding exposure to
the caller, which computes them from every spike: a smaller dictionary must
not look like a shorter recording.
"""

from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from non_local_detector.exceptions import ValidationError

CompressionMethod = Literal["uniform", "stratified"]
COMPRESSION_METHODS: tuple[str, ...] = ("uniform", "stratified")

# k-means cells over the bandwidth-scaled marks, one stratification axis.
_MARK_STRATA = 32
# Position strata are cells this many position bandwidths wide.
_POSITION_STRATUM_WIDTH = 2.0


def validate_compression(
    encoding_compression: str | None, n_encoding_kernels: object
) -> None:
    """Raise ``ValidationError`` for an unknown method or dictionary size.

    Parameters
    ----------
    encoding_compression : str or None
    n_encoding_kernels : object
        Must be a positive integer when ``encoding_compression`` is given.
    """
    if encoding_compression is None:
        return
    if encoding_compression not in COMPRESSION_METHODS:
        raise ValidationError(
            "Unknown encoding_compression",
            expected=f"None or one of {COMPRESSION_METHODS}",
            got=repr(encoding_compression),
        )
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
    method: CompressionMethod,
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
    positions, marks, weights = (np.asarray(a) for a in (positions, marks, weights))
    keep = np.flatnonzero(weights > 0)
    if keep.size > n_kernels:
        rng = np.random.default_rng(seed)
        if method == "uniform":
            probabilities = np.full(keep.size, n_kernels / keep.size)
            kept = rng.choice(keep.size, n_kernels, replace=False)
        else:
            strata = _strata(
                positions[keep], marks[keep], position_std, waveform_std, seed
            )
            probabilities = _inclusion_probabilities(
                np.bincount(strata)[strata], n_kernels
            )
            # Pareto sampling: the n_kernels smallest ranking variables.
            uniform = rng.random(keep.size)
            with np.errstate(divide="ignore"):
                ranking = (uniform * (1.0 - probabilities)) / (
                    (1.0 - uniform) * probabilities
                )
            kept = np.argpartition(ranking, n_kernels - 1)[:n_kernels]
        kept = np.sort(kept)
        weights = weights.copy()
        weights[keep[kept]] /= probabilities[kept]
        keep = keep[kept]
    return positions[keep], marks[keep], weights[keep]
