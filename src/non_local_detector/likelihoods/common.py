from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from jax.nn import logsumexp
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment

EPS = 1e-15
LOG_EPS = np.log(EPS)


def get_position_at_time(
    time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: jnp.ndarray,
    env: Environment | None = None,
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


def get_spike_time_bin_ind(spike_times: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Get the index of the time bin for each spike time.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    time : np.ndarray, shape (n_time_bins,)
        Bin edges.

    Returns
    -------
    ind : np.ndarray, shape (n_spikes,)
    """
    return np.digitize(spike_times, time[1:-1])


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
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    weights: jnp.ndarray,
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
    weights : jnp.ndarray, shape (n_samples,)
        Weights for each sample.

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)
    """
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))

    for dim_eval_points, dim_samples, dim_std in zip(
        eval_points.T, samples.T, std, strict=False
    ):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return (weights @ distance) / jnp.sum(weights)


def block_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
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

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    for start_ind in range(0, n_eval_points, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        density = jax.lax.dynamic_update_slice(
            density,
            kde(eval_points[block_inds], samples, std, weights),
            (start_ind,),
        )

    return density


@jax.jit
def log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Log kernel density estimate:
        log p(x) = logsumexp_i [ log w_i + sum_d log N(x_d | s_{i,d}, std_d) ] - logsumexp_i [log w_i]
    Shapes:
      eval_points: (n_eval, n_dims)
      samples:     (n_samp, n_dims)
      std:         (n_dims,)
      weights:     (n_samp,)
    Returns: (n_eval,)
    """
    if eval_points.ndim == 1:
        eval_points = jnp.expand_dims(eval_points, axis=1)

    # build log-kernel matrix K_log with shape (n_samp, n_eval)
    K_log = jnp.zeros((samples.shape[0], eval_points.shape[0]))
    for dim_eval, dim_samp, dim_std in zip(eval_points.T, samples.T, std, strict=False):
        K_log += log_gaussian_pdf(
            jnp.expand_dims(dim_eval, axis=0),
            jnp.expand_dims(dim_samp, axis=1),
            dim_std,
        )

    log_w = safe_log(weights)  # (n_samp,)
    log_num = logsumexp(log_w[:, None] + K_log, axis=0)  # (n_eval,)
    log_den = logsumexp(log_w)  # scalar
    return log_num - log_den


def block_log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Log KDE split into blocks over eval points. Returns (n_eval,)
    """
    n_eval = eval_points.shape[0]
    out = jnp.full((n_eval,), LOG_EPS)

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    for start in range(0, n_eval, block_size):
        sl = slice(start, start + block_size)
        block_vals = log_kde(eval_points[sl], samples, std, weights)
        out = jax.lax.dynamic_update_slice(out, block_vals, (start,))
    return out


@dataclass
class KDEModel:
    std: jnp.ndarray
    block_size: int | None = None

    def fit(
        self, samples: jnp.ndarray, weights: jnp.ndarray | None = None
    ) -> "KDEModel":
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
        if weights is None:
            self.weights_ = jnp.ones((samples.shape[0],))
        else:
            self.weights_ = jnp.asarray(weights)

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
            if isinstance(self.std, int | float)
            else self.std
        )
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size, self.weights_)

    def predict_log(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """
        Log-density version of predict(). Same inputs, returns log p(eval_points).
        """
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = (
            jnp.array([self.std] * eval_points.shape[1])
            if isinstance(self.std, int | float)
            else self.std
        )
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )
        return block_log_kde(eval_points, self.samples_, std, block_size, self.weights_)


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


def safe_divide(numerator, denominator, eps=EPS, condition=None):
    """Safely divide two arrays, avoiding division by zero.

    Parameters
    ----------
    numerator : jnp.ndarray
        Numerator array of any shape.
    denominator : jnp.ndarray
        Denominator array, must be broadcastable with numerator.
    eps : float, optional
        Small value to avoid division by zero, by default 1e-8.
    condition : jnp.ndarray, optional
        Boolean condition array to apply the division, by default None.
        If None, condition is computed as abs(denominator) < eps.
        Useful if pre-computing the condition is more efficient.

    Returns
    -------
    result : jnp.ndarray
        Result of safe division with same shape as broadcast of inputs.
        Where condition is True, returns eps instead of dividing.
    """
    if condition is None:
        condition = jnp.abs(denominator) < eps

    return jnp.where(condition, eps, numerator / denominator)


def safe_log(x, eps=EPS, condition=None):
    """Safely compute the logarithm of an array, avoiding log(0).

    Parameters
    ----------
    x : jnp.ndarray
        Input array of any shape.
    eps : float, optional
        Small value to avoid log(0), by default 1e-8.
    condition : jnp.ndarray, optional
        Boolean condition array to apply the logarithm, by default None.
        If None, condition is computed as abs(x) < eps.
        Useful if pre-computing the condition is more efficient.

    Returns
    -------
    result : jnp.ndarray
        Logarithm of input array with same shape as x.
        Where condition is True, returns log(eps) instead of log(0).
    """
    if condition is None:
        condition = jnp.abs(x) < eps

    return jnp.log(jnp.where(condition, eps, x))
