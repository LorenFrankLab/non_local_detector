import numpy as np
import xarray as xr
from scipy.stats import rv_histogram  # type: ignore[import-untyped]


def maximum_a_posteriori_estimate(posterior: xr.DataArray) -> np.ndarray:
    """Find the most likely position from the posterior distribution.

    Computes the maximum a posteriori (MAP) estimate by finding the position
    bin with the highest probability at each time point. Handles both 1D
    and 2D position representations.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins) or (n_time, n_x_bins, n_y_bins)
        Posterior probability distribution over position bins. For 1D tracks,
        dimensions are (time, position). For 2D environments, dimensions are
        (time, x_position, y_position).

    Returns
    -------
    map_estimate : np.ndarray, shape (n_time, n_spatial_dims)
        Most likely position coordinates at each time point. For 1D tracks,
        shape is (n_time, 1). For 2D environments, shape is (n_time, 2).

    """
    try:
        stacked_posterior = posterior.stack(z=["x_position", "y_position"])
        map_estimate = stacked_posterior.z[stacked_posterior.argmax("z")]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior.position[np.log(posterior).argmax("position")]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def sample_posterior(
    posterior: xr.DataArray, place_bin_edges: np.ndarray, n_samples: int = 1000
) -> np.ndarray:
    """Generate random samples from the posterior distribution.

    Treats the posterior as a probability mass function and generates random
    samples from it at each time point using scipy's rv_histogram. Useful
    for uncertainty quantification and Bayesian inference.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins) or (n_time, n_x_bins, n_y_bins)
        Posterior probability distribution over position bins. For 2D environments,
        the spatial dimensions are automatically flattened.
    place_bin_edges : np.ndarray, shape (n_position_bins + 1,)
        Bin edges defining the boundaries of the position bins. Must have one
        more element than the number of position bins.
    n_samples : int, optional
        Number of random samples to generate at each time point, by default 1000.

    Returns
    -------
    posterior_samples : np.ndarray, shape (n_time, n_samples)
        Random samples drawn from the posterior distribution at each time point.
        Each row contains samples for one time point.

    """
    # Stack 2D positions into one dimension
    try:
        posterior = posterior.stack(z=["x_position", "y_position"]).values
    except (KeyError, AttributeError):
        posterior = np.asarray(posterior)

    place_bin_edges = place_bin_edges.squeeze()
    n_time = posterior.shape[0]

    posterior_samples = [
        rv_histogram((posterior[time_ind], place_bin_edges)).rvs(size=n_samples)
        for time_ind in range(n_time)
    ]

    return np.asarray(posterior_samples)
