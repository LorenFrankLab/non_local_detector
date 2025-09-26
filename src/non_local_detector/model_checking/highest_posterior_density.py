import numpy as np
import xarray as xr


def get_highest_posterior_threshold(
    posterior: xr.DataArray, coverage: float = 0.95
) -> np.ndarray:
    """Estimate the posterior threshold for highest posterior density (HPD) regions.

    Computes the threshold values that define the highest posterior density
    regions containing a specified coverage probability. This approach can
    handle multimodal distributions by selecting regions with highest density
    rather than requiring a single contiguous interval.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior probability distributions over position at each time point.
        The non-time dimensions are flattened for processing.
    coverage : float, optional
        Desired coverage probability for the HPD region. Must be between 0 and 1.
        Default is 0.95 for 95% coverage.

    Returns
    -------
    threshold : np.ndarray, shape (n_time,)
        Threshold values for each time point. Posterior values at or above
        this threshold define the HPD region with the specified coverage.

    Notes
    -----
    The algorithm sorts posterior values in descending order and finds the
    threshold where the cumulative probability first exceeds the desired coverage.
    For reference see: https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    """
    # Reshape non-time dimensions into a single dimension
    n_time = posterior.shape[0]
    posterior_array = np.asarray(posterior).reshape((n_time, -1))
    # Remove NaN values from the posterior
    posterior_array = posterior_array[:, ~np.any(np.isnan(posterior_array), axis=0)]

    # Sort the posterior values in descending order
    const = np.sum(posterior_array, axis=1, keepdims=True)
    sorted_norm_posterior = np.sort(posterior_array, axis=1)[:, ::-1] / const

    # Find the threshold that corresponds to the coverage
    # by finding the first index where the cumulative sum is greater than the coverage
    posterior_less_than_coverage = np.cumsum(sorted_norm_posterior, axis=1) >= coverage
    crit_ind = np.argmax(posterior_less_than_coverage, axis=1)

    # Handle case when there are no points in the posterior less than coverage
    # Use the last valid index instead of accessing shape directly
    crit_ind[posterior_less_than_coverage.sum(axis=1) == 0] = (
        posterior_array.shape[1] - 1
    )  # type: ignore[misc]

    threshold_values = (
        sorted_norm_posterior[np.arange(n_time), crit_ind] * const.squeeze()
    )
    return np.asarray(threshold_values)


def get_HPD_spatial_coverage(
    posterior: xr.DataArray, hpd_threshold: np.ndarray
) -> np.ndarray:
    """Compute total spatial area covered by highest posterior density regions.

    Calculates the total area of the environment covered by posterior values
    that exceed the HPD threshold at each time point. This provides a measure
    of spatial uncertainty in the posterior distribution.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
        Posterior probability distributions over spatial position at each
        time point. Must have a 'position' coordinate for spatial integration.
    hpd_threshold : np.ndarray, shape (n_time,)
        HPD threshold values for each time point, typically obtained from
        `get_highest_posterior_threshold`.

    Returns
    -------
    spatial_coverage : np.ndarray, shape (n_time,)
        Total spatial area covered by the highest posterior density regions
        at each time point. Units depend on the spatial coordinate system
        of the posterior (e.g., cm², m²).

    Notes
    -----
    The function assumes uniform spatial bin spacing and uses the first
    difference of the position coordinate to determine bin width for
    area calculation.
    """
    isin_hpd = posterior >= hpd_threshold[:, np.newaxis]
    return np.asarray(
        (isin_hpd * np.diff(posterior.position)[0]).sum("position").values
    )
