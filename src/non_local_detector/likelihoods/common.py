from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate
from track_linearization import get_linearized_position

from non_local_detector.environment import Environment

EPS = 1e-15
LOG_EPS = np.log(EPS)


def get_position_at_time(
    time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: jnp.ndarray,
    env: Optional[Environment] = None,
) -> jnp.ndarray:
    """Get the position at the time of each spike.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
    position : jnp.ndarray, shape (n_time_position, n_dims_position)
    spike_times : jnp.ndarray, shape (n_spikes,)
    env : Optional[Environment], optional
        The spatial environment, by default None

    Returns
    -------
    position_at_spike_times : jnp.ndarray, shape (n_spikes, n_dims_position)
    """
    position_at_spike_times = scipy.interpolate.interpn(
        (time,), position, spike_times, bounds_error=False, fill_value=None
    )
    if env is not None and env.track_graph is not None:
        if position_at_spike_times.shape[0] > 0:
            position_at_spike_times = get_linearized_position(
                position_at_spike_times,
                env.track_graph,
                edge_order=env.edge_order,
                edge_spacing=env.edge_spacing,
            ).linear_position.to_numpy()[:, None]
        else:
            position_at_spike_times = jnp.array([])[:, None]

    return position_at_spike_times


@jax.jit
def log_gaussian_pdf(
    x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log of the Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples, n_dims)
        Input data.
    mean : jnp.ndarray, shape (n_dims,)
        Mean of the Gaussian.
    sigma : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian.

    Returns
    -------
    log_pdf : jnp.ndarray, shape (n_samples,)
    """
    return -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def gaussian_pdf(x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples, n_dims)
        Input data.
    mean : jnp.ndarray, shape (n_dims,)
        Mean of the Gaussian.
    sigma : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian.

    Returns
    -------
    pdf : jnp.ndarray, shape (n_samples,)
    """
    return jnp.exp(log_gaussian_pdf(x, mean, sigma))


def kde(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Kernel density estimation.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)
    """
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))

    for dim_eval_points, dim_samples, dim_std in zip(eval_points.T, samples.T, std):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return jnp.mean(distance, axis=0)


def block_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
) -> jnp.ndarray:
    """Kernel density estimation split into blocks.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.
    block_size : int, optional
        Size of blocks to do computation over, by default 100

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)
    """
    n_eval_points = eval_points.shape[0]
    density = jnp.zeros((n_eval_points,))
    for start_ind in range(0, n_eval_points, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        density = jax.lax.dynamic_update_slice(
            density,
            kde(eval_points[block_inds], samples, std),
            (start_ind,),
        )

    return density


@dataclass
class KDEModel:
    std: jnp.ndarray
    block_size: Optional[int] = None

    def fit(self, samples: jnp.ndarray) -> "KDEModel":
        """Fit the model.

        Parameters
        ----------
        samples : jnp.ndarray, shape (n_samples, n_dims)
            Training samples.

        Returns
        -------
        self : KDEModel
        """
        samples = jnp.asarray(samples)
        if samples.ndim == 1:
            samples = jnp.expand_dims(samples, axis=1)
        self.samples_ = samples

        return self

    def predict(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Predict the density at the evaluation points.

        Parameters
        ----------
        eval_points : jnp.ndarray, shape (n_eval_points, n_dims)

        Returns
        -------
        density : jnp.ndarray, shape (n_eval_points,)
        """
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = (
            jnp.array([self.std] * eval_points.shape[1])
            if isinstance(self.std, (int, float))
            else self.std
        )
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size)


def get_spikecount_per_time_bin(
    spike_times: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Get the number of spikes in each time bin.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    time : np.ndarray, shape (n_time,)

    Returns
    -------
    count : np.ndarray, shape (n_time,)
    """
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    return np.bincount(
        np.digitize(spike_times, time[1:-1]),
        minlength=time.shape[0],
    )
