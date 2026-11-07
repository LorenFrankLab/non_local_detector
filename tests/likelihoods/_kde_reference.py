"""Float64 references for the clusterless KDE joint mark intensity.

They use NumPy and SciPy only, never the package's kernels, so a test that
compares ``estimate_log_joint_mark_intensity`` (or its blocked, tiled and
streaming forms) against them checks the reduction itself.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp
from scipy.stats import norm


def reference_log_position_kernel(
    eval_points: ArrayLike, encoding_positions: ArrayLike, position_std: ArrayLike
) -> NDArray[np.float64]:
    """Log Gaussian position kernel of each encoding spike at each bin.

    Parameters
    ----------
    eval_points : array_like, shape (n_position_bins, n_position_dims)
    encoding_positions : array_like, shape (n_encoding_spikes, n_position_dims)
    position_std : array_like, shape (n_position_dims,)

    Returns
    -------
    log_position_kernel : np.ndarray, shape (n_encoding_spikes, n_position_bins)
    """
    return norm.logpdf(
        np.asarray(eval_points, np.float64)[None],
        np.asarray(encoding_positions, np.float64)[:, None],
        np.asarray(position_std, np.float64),
    ).sum(axis=-1)


def _log_terms(
    decoding_features: ArrayLike,
    encoding_features: ArrayLike,
    waveform_stds: ArrayLike,
    log_position_distance: ArrayLike,
    encoding_weights: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Log mark kernels, weighted log position kernels and ``(m_e - m) / std**2``.

    Shapes ``(n_encoding, n_decoding)``, ``(n_encoding, n_bins)`` and
    ``(n_encoding, n_decoding, n_features)``.
    """
    decoding = np.asarray(decoding_features, np.float64)
    encoding = np.asarray(encoding_features, np.float64)
    stds = np.asarray(waveform_stds, np.float64)
    log_mark = norm.logpdf(decoding[None], encoding[:, None], stds).sum(axis=-1)
    weights = (
        np.ones(encoding.shape[0])
        if encoding_weights is None
        else np.asarray(encoding_weights, np.float64)
    )
    with np.errstate(divide="ignore"):  # zero weights drop out as -inf
        log_weights = np.log(weights) - np.log(weights.sum() or 1.0)
    log_position = np.asarray(log_position_distance, np.float64) + log_weights[:, None]
    return log_mark, log_position, (encoding[:, None] - decoding[None]) / stds**2


def reference_log_joint_mark_intensity(
    decoding_features: ArrayLike,
    encoding_features: ArrayLike,
    waveform_stds: ArrayLike,
    occupancy: ArrayLike,
    mean_rate: float,
    log_position_distance: ArrayLike,
    encoding_weights: ArrayLike | None = None,
    log_background: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Brute-force float64 log joint mark intensity by logsumexp.

    ``log(mean_rate * sum_e w_e K_mark(m, m_e) K_pos(e, x) / sum_e w_e /
    occupancy(x))`` with Gaussian mark kernels; uniform weights by default.
    No encoding spikes, a zero total weight or a zero occupancy give ``-inf``.

    Parameters
    ----------
    decoding_features : array_like, shape (n_decoding_spikes, n_features)
    encoding_features : array_like, shape (n_encoding_spikes, n_features)
    waveform_stds : array_like, shape (n_features,)
    occupancy : array_like, shape (n_position_bins,)
    mean_rate : float
    log_position_distance : array_like, shape (n_encoding_spikes, n_position_bins)
        Log position kernel of each encoding spike at each bin.
    encoding_weights : array_like, shape (n_encoding_spikes,), optional
    log_background : array_like, shape (n_decoding_spikes,), optional
        When given, return ``logaddexp(log intensity, log_background)``, the
        estimator's output.

    Returns
    -------
    log_joint : np.ndarray, shape (n_decoding_spikes, n_position_bins)
    """
    n_decoding = np.shape(decoding_features)[0]
    occupancy = np.asarray(occupancy, np.float64)
    log_marginal = np.full((n_decoding, occupancy.shape[0]), -np.inf)
    if np.shape(encoding_features)[0] > 0:
        log_mark, log_position, _ = _log_terms(
            decoding_features,
            encoding_features,
            waveform_stds,
            log_position_distance,
            encoding_weights,
        )
        # One decoding spike at a time bounds memory to (n_encoding, n_bins).
        for spike in range(n_decoding):
            log_marginal[spike] = logsumexp(
                log_mark[:, spike, None] + log_position, axis=0
            )
    # Zero occupancy has no intensity.
    supported = occupancy > 0
    log_joint = np.where(
        supported[None],
        np.log(mean_rate) + log_marginal - np.log(np.where(supported, occupancy, 1.0)),
        -np.inf,
    )
    if log_background is None:
        return log_joint
    return np.logaddexp(log_joint, np.asarray(log_background, np.float64)[:, None])


def reference_log_joint_mark_intensity_gradient(
    decoding_features: ArrayLike,
    encoding_features: ArrayLike,
    waveform_stds: ArrayLike,
    occupancy: ArrayLike,
    mean_rate: float,
    log_position_distance: ArrayLike,
    log_background: ArrayLike,
    encoding_weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Gradient of the summed estimator output over the decoding marks.

    ``d logaddexp(log I, b) / dm = I / (I + B) * d log I / dm``, where
    ``d log I / dm`` is the posterior mean over encoding spikes of
    ``(m_e - m) / std**2``; zero where the intensity is zero.

    Parameters
    ----------
    Same as ``reference_log_joint_mark_intensity``; ``log_background`` is
    required.

    Returns
    -------
    gradient : np.ndarray, shape (n_decoding_spikes, n_features)
    """
    args = (
        decoding_features,
        encoding_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        encoding_weights,
    )
    log_intensity = reference_log_joint_mark_intensity(*args)
    log_mark, log_position, mark_score = _log_terms(*args[:3], *args[5:])
    # (n_encoding, n_decoding, n_bins) log terms of each intensity.
    log_terms = log_mark[:, :, None] + log_position[:, None, :]
    with np.errstate(invalid="ignore"):  # all -inf terms: no intensity
        posterior = np.exp(log_terms - logsumexp(log_terms, axis=0))
        d_log_intensity = np.einsum("erp,erd->rpd", posterior, mark_score)
    share = np.exp(
        log_intensity
        - np.logaddexp(log_intensity, np.asarray(log_background, np.float64)[:, None])
    )
    contribution = np.where(
        (share > 0)[..., None], share[..., None] * d_log_intensity, 0.0
    )
    return contribution.sum(axis=1)
