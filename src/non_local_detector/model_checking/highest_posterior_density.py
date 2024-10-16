import numpy as np
import xarray as xr


def get_highest_posterior_threshold(
    posterior: xr.DataArray, coverage: float = 0.95
) -> np.ndarray:
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
    # Reshape non-time dimensions into a single dimension
    n_time = posterior.shape[0]
    posterior = np.asarray(posterior).reshape((n_time, -1))
    # Remove NaN values from the posterior
    posterior = posterior[:, ~np.any(np.isnan(posterior), axis=0)]

    # Sort the posterior values in descending order
    const = np.sum(posterior, axis=1, keepdims=True)
    sorted_norm_posterior = np.sort(posterior, axis=1)[:, ::-1] / const

    # Find the threshold that corresponds to the coverage
    # by finding the first index where the cumulative sum is greater than the coverage
    posterior_less_than_coverage = np.cumsum(sorted_norm_posterior, axis=1) >= coverage
    crit_ind = np.argmax(posterior_less_than_coverage, axis=1)

    # Handle case when there are no points in the posterior less than coverage
    crit_ind[posterior_less_than_coverage.sum(axis=1) == 0] = posterior.shape[1] - 1

    return sorted_norm_posterior[(np.arange(n_time), crit_ind)] * const.squeeze()


def get_HPD_spatial_coverage(
    posterior: xr.DataArray, hpd_threshold: np.ndarray
) -> np.ndarray:
    """Total area of the environment covered by the higest posterior values.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
    hpd_threshold : numpy.ndarray, shape (n_time,)


    Returns
    -------
    spatial_coverage : np.ndarray, shape (n_time,)
        Amount of the environment covered by the higest posterior values.
    """
    isin_hpd = posterior >= hpd_threshold[:, np.newaxis]
    return (isin_hpd * np.diff(posterior.position)[0]).sum("position").values
