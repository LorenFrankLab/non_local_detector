import numpy as np


def get_highest_posterior_threshold(posterior, coverage=0.95):
    """Estimate of the posterior spread that can account for multimodal
    distributions.

    https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
    coverage : float, optional

    Returns
    -------
    threshold : ndarray, shape (n_time,)

    """
    # Stack 2D positions into one dimension
    try:
        posterior = (
            posterior.stack(position=["x_position", "y_position"])
            .dropna("position")
            .values
        )
    except KeyError:
        posterior = posterior.dropna("position").values
    const = np.sum(posterior, axis=1, keepdims=True)
    sorted_norm_posterior = np.sort(posterior, axis=1)[:, ::-1] / const
    posterior_less_than_coverage = np.cumsum(sorted_norm_posterior, axis=1) >= coverage
    crit_ind = np.argmax(posterior_less_than_coverage, axis=1)
    # Handle case when there are no points in the posterior less than coverage
    crit_ind[posterior_less_than_coverage.sum(axis=1) == 0] = posterior.shape[1] - 1

    n_time = posterior.shape[0]
    threshold = sorted_norm_posterior[(np.arange(n_time), crit_ind)] * const.squeeze()
    return threshold


def get_HPD_spatial_coverage(posterior, hpd_threshold):
    """Total area of the environment covered by the higest posterior values.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
    hpd_threshold : numpy.ndarray, shape (n_time,)


    Returns
    -------
    spatial_coverage : float
        Amount of the environment covered by the higest posterior values.
    """
    isin_hpd = posterior >= hpd_threshold[:, np.newaxis]
    return (isin_hpd * np.diff(posterior.position)[0]).sum("position").values
