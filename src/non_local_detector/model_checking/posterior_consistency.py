"""Posterior consistency tests for state space model goodness of fit.

This module provides functions to assess the consistency between posterior
distributions and their component likelihood distributions in Bayesian
state space models. These tests help identify issues with prior specification
and model assumptions.
"""

import numpy as np
from scipy.stats import entropy  # type: ignore[import-untyped]

from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
)


def posterior_consistency_kl_divergence(
    posterior: np.ndarray, likelihood: np.ndarray
) -> np.ndarray:
    """Compute Kullback-Leibler divergence between posterior and likelihood distributions.

    Measures the information divergence between the posterior and likelihood
    distributions at each time point. Large divergences may indicate issues
    with the prior specification or model assumptions.

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior probability distributions over position at each time point.
        Must be properly normalized probability distributions.
    likelihood : np.ndarray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Likelihood distributions at each time point. Must have same shape
        as posterior and be properly normalized.

    Returns
    -------
    kl_divergence : np.ndarray, shape (n_time,)
        Kullback-Leibler divergence D_KL(posterior || likelihood) at each
        time point. Values are non-negative, with 0 indicating identical
        distributions.

    Notes
    -----
    The KL divergence is computed using scipy.stats.entropy with the formula:
    D_KL(P || Q) = sum(P * log(P / Q))
    where P is the posterior and Q is the likelihood.
    """
    return entropy(posterior, likelihood, axis=-1)


def posterior_consistency_hpd_overlap(
    posterior: np.ndarray, likelihood: np.ndarray, coverage: float = 0.95
) -> np.ndarray:
    """Compute overlap between HPD regions of posterior and likelihood distributions.

    Measures the spatial overlap between the highest posterior density regions
    of the posterior and likelihood distributions. High overlap suggests
    consistency between the likelihood and prior contributions to the posterior.

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior probability distributions over position at each time point.
        Must be properly normalized probability distributions.
    likelihood : np.ndarray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Likelihood distributions at each time point. Must have same shape
        as posterior and be properly normalized.
    coverage : float, optional
        Coverage probability for the HPD regions. Must be between 0 and 1.
        Default is 0.95 for 95% HPD regions.

    Returns
    -------
    hpd_overlap : np.ndarray, shape (n_time,)
        Proportion of overlap between the HPD regions of posterior and
        likelihood at each time point. Values range from 0 (no overlap)
        to 1 (complete overlap).

    Notes
    -----
    The overlap is computed as the intersection of the HPD regions divided
    by the minimum of the two HPD region sizes to normalize for different
    region sizes.
    """
    posterior = np.asarray(posterior)
    likelihood = np.asarray(likelihood)

    posterior_threshold = get_highest_posterior_threshold(posterior, coverage=coverage)
    likelihood_threshold = get_highest_posterior_threshold(
        likelihood, coverage=coverage
    )
    isin_posterior_hpd = posterior >= posterior_threshold[:, None]
    isin_likelihood_hpd = likelihood >= likelihood_threshold[:, None]

    denom = np.min(
        np.stack(
            (isin_posterior_hpd.sum(axis=1), isin_likelihood_hpd.sum(axis=1)), axis=1
        ),
        axis=1,
    )
    # Avoid division by zero
    denom = np.clip(denom, 1, None)

    return (isin_posterior_hpd & isin_likelihood_hpd).sum(axis=1) / denom
