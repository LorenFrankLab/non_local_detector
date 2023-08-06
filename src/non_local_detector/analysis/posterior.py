import numpy as np
from scipy.stats import rv_histogram


def maximum_a_posteriori_estimate(posterior):
    """Finds the most likely position of the posterior (the posterior mode)

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    """
    try:
        stacked_posterior = np.log(posterior.stack(z=["x_position", "y_position"]))
        map_estimate = stacked_posterior.z[stacked_posterior.argmax("z")]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior.position[np.log(posterior).argmax("position")]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def sample_posterior(posterior, place_bin_edges, n_samples=1000):
    """Samples the posterior positions.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    posterior_samples : numpy.ndarray, shape (n_time, n_samples)

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
