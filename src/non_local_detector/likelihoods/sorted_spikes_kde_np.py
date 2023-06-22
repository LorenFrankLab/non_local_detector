import numpy as np
import scipy.interpolate
import scipy.stats
from tqdm.auto import tqdm

from non_local_detector.core import atleast_2d

EPS = 1e-15


def gaussian_pdf(x: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma."""
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def estimate_position_distance(
    place_bin_centers: np.ndarray, positions: np.ndarray, position_std: np.ndarray
) -> np.ndarray:
    """Estimates the Euclidean distance between positions and position bins.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins, n_position_dims)
    positions : np.ndarray, shape (n_time, n_position_dims)
    position_std : array_like, shape (n_position_dims,)

    Returns
    -------
    position_distance : np.ndarray, shape (n_time, n_position_bins)

    """
    n_time, n_position_dims = positions.shape
    n_position_bins = place_bin_centers.shape[0]

    position_distance = np.ones((n_time, n_position_bins), dtype=np.float32)

    if isinstance(position_std, (int, float)):
        position_std = [position_std] * n_position_dims

    for position_ind, std in enumerate(position_std):
        position_distance *= gaussian_pdf(
            np.expand_dims(place_bin_centers[:, position_ind], axis=0),
            np.expand_dims(positions[:, position_ind], axis=1),
            std,
        )

    return position_distance


def estimate_position_density(
    place_bin_centers: np.ndarray,
    positions: np.ndarray,
    position_std: float | np.ndarray,
    block_size: int = 100,
) -> np.ndarray:
    """Estimates a kernel density estimate over position bins using
    Euclidean distances.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins, n_position_dims)
    positions : np.ndarray, shape (n_time, n_position_dims)
    position_std : float or array_like, shape (n_position_dims,)
    block_size : int

    Returns
    -------
    position_density : np.ndarray, shape (n_position_bins,)

    """
    n_time = positions.shape[0]
    n_position_bins = place_bin_centers.shape[0]

    if block_size is None:
        block_size = n_time

    position_density = np.empty((n_position_bins,))
    for start_ind in range(0, n_position_bins, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        position_density[block_inds] = np.mean(
            estimate_position_distance(
                place_bin_centers[block_inds], positions, position_std
            ),
            axis=0,
        )
    return position_density


def get_firing_rate(
    is_spike: np.ndarray,
    position: np.ndarray,
    place_bin_centers: np.ndarray,
    is_track_interior: np.ndarray,
    not_nan_position: np.ndarray,
    occupancy: np.ndarray,
    position_std: np.ndarray,
    block_size: int | None = None,
) -> np.ndarray:
    if is_spike.sum() > 0:
        mean_rate = is_spike.mean()
        marginal_density = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)

        marginal_density[is_track_interior] = estimate_position_density(
            place_bin_centers[is_track_interior],
            np.asarray(position[is_spike & not_nan_position], dtype=np.float32),
            position_std,
            block_size=block_size,
        )
        return np.exp(np.log(mean_rate) + np.log(marginal_density) - np.log(occupancy))
    else:
        return np.zeros_like(occupancy)


def estimate_place_fields_kde(
    position: np.ndarray,
    spikes: np.ndarray,
    place_bin_centers: np.ndarray,
    position_std: np.ndarray,
    is_track_boundary: np.ndarray | None = None,
    is_track_interior: np.ndarray | None = None,
    edges: list[np.ndarray] | None = None,
    place_bin_edges: np.ndarray | None = None,
    use_diffusion: bool = False,
    block_size: int | None = None,
) -> np.ndarray:
    """Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
    spikes : np.ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    place_bin_centers : np.ndarray, shape (n_bins, n_position_dims)
    position_std : float or array_like, shape (n_position_dims,)
        Amount of smoothing for position.  Standard deviation of kernel.
    is_track_boundary : None or np.ndarray, shape (n_bins,)
    is_track_interior : None or np.ndarray, shape (n_bins,)
    edges : None or list of np.ndarray
    place_bin_edges : np.ndarray, shape (n_bins + 1, n_position_dims)
    use_diffusion : bool
        Respect track geometry by using diffusion distances
    block_size : int
        Size of data to process in chunks

    Returns
    -------
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """

    position = atleast_2d(position).astype(np.float32)
    place_bin_centers = atleast_2d(place_bin_centers).astype(np.float32)
    not_nan_position = np.all(~np.isnan(position), axis=1)

    occupancy = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)
    occupancy[is_track_interior.ravel(order="F")] = estimate_position_density(
        place_bin_centers[is_track_interior.ravel(order="F")],
        position[not_nan_position],
        position_std,
        block_size=block_size,
    )
    place_fields = np.stack(
        [
            get_firing_rate(
                is_spike,
                position,
                place_bin_centers,
                is_track_interior.ravel(order="F"),
                not_nan_position,
                occupancy,
                position_std,
            )
            for is_spike in np.asarray(spikes, dtype=bool).T
        ],
        axis=1,
    )

    return place_fields


def combined_likelihood(
    spikes: np.ndarray, conditional_intensity: np.ndarray
) -> np.ndarray:
    """

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(tqdm(spikes.T), conditional_intensity.T):
        log_likelihood += scipy.stats.poisson.logpmf(is_spike, ci)

    return log_likelihood


def estimate_spiking_likelihood_kde(
    spikes: np.ndarray,
    conditional_intensity: np.ndarray,
    is_track_interior: np.ndarray | None = None,
) -> np.ndarray:
    """

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)
    is_track_interior : None or np.ndarray, optional, shape (n_x_position_bins,
                                                             n_y_position_bins)
    Returns
    -------
    likelihood : np.ndarray, shape (n_time, n_bins)

    """
    spikes = np.asarray(spikes, dtype=np.float32)

    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order="F")
    else:
        n_bins = conditional_intensity.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=np.bool)

    log_likelihood = combined_likelihood(spikes, conditional_intensity)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask


def fit_sorted_spikes_kde_np_encoding_model(
    position,
    spikes,
    place_bin_centers,
    place_bin_edges,
    edges,
    is_track_interior,
    is_track_boundary,
    position_std=8.0,
    use_diffusion=False,
    block_size=100,
):
    place_fields = estimate_place_fields_kde(
        position,
        spikes,
        place_bin_centers,
        position_std,
        is_track_boundary,
        is_track_interior,
        edges,
        place_bin_edges,
        use_diffusion,
        block_size,
    )
    return {
        "place_fields": place_fields,
        "place_bin_centers": place_bin_centers,
        "is_track_interior": is_track_interior,
    }


def predict_sorted_spikes_kde_np_log_likelihood(
    position,
    spikes,
    place_fields,
    place_bin_centers,
    is_track_interior,
    is_local: bool = False,
):
    n_time = spikes.shape[0]
    if is_local:
        log_likelihood = np.zeros((n_time, 1))
        for neuron_spikes, non_local_rate in zip(tqdm(spikes.T), place_fields.T):
            local_rate = scipy.interpolate.griddata(
                place_bin_centers, non_local_rate, position, method="nearest"
            )
            local_rate = np.clip(local_rate, a_min=EPS, a_max=None)
            log_likelihood += scipy.stats.poisson.logpmf(
                neuron_spikes[:, np.newaxis], local_rate
            )
    else:
        log_likelihood = np.zeros((n_time, len(place_bin_centers)))
        for neuron_spikes, non_local_rate in zip(tqdm(spikes.T), place_fields.T):
            non_local_rate[~is_track_interior] = EPS
            non_local_rate = np.clip(non_local_rate, a_min=EPS, a_max=None)
            log_likelihood += scipy.stats.poisson.logpmf(
                neuron_spikes[:, np.newaxis], non_local_rate[np.newaxis]
            )
        log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood
