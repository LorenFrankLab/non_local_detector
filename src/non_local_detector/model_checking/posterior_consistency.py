"""State space goodness of fit for """

import numpy as np
from scipy.stats import entropy

from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
)


def posterior_consistency_kl_divergence(posterior, likelihood):
    """Measure of the divergence between the posterior and likelihood distributions.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior distribution of the model.
    likelihood : xr.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Likelihood distribution of the model.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence between the posterior and likelihood distributions.
    """
    return entropy(posterior, likelihood, axis=1)


def posterior_consistency_hpd_overlap(posterior, likelihood, coverage: float = 0.95):
    """Measure of the overlap between the highest posterior density (HPD) regions of
    the posterior and likelihood distributions.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior distribution of the model.
    likelihood : xr.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Likelihood distribution of the model.
    coverage : float, optional
        Highest posterior density coverage threshold, by default 0.95

    Returns
    -------
    hpd_overlap : float
        Overlap between the HPD regions of the posterior and likelihood distributions.
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
