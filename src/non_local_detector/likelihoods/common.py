from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from jax.nn import logsumexp
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError

EPS = 1e-15
LOG_EPS = np.log(EPS)


def as_std_array(std: "jnp.ndarray | float | int", n_dims: int) -> jnp.ndarray:
    """Broadcast a scalar kernel bandwidth to a per-dimension array.

    A single ``jnp.ndim`` check covers Python ``int``/``float``, NumPy scalars
    (including ``np.float32``), 0-d arrays, and JAX scalars uniformly, avoiding
    the ``isinstance(std, int | float)`` pattern that silently misses
    ``np.float32`` and 0-d arrays. Array-valued ``std`` (``ndim >= 1``) is
    returned via ``jnp.asarray`` (a no-op for JAX arrays; lists/NumPy arrays are
    converted to a JAX array).

    Parameters
    ----------
    std : jnp.ndarray or float or int
        Scalar (broadcast to ``n_dims``) or per-dimension standard deviation(s).
    n_dims : int
        Number of dimensions to broadcast a scalar to.

    Returns
    -------
    std_array : jnp.ndarray
        Shape ``(n_dims,)`` for scalar input; the original array otherwise.
    """
    if jnp.ndim(std) == 0:
        return jnp.full((n_dims,), std)
    return jnp.asarray(std)


def validate_weights(weights: np.ndarray, n_time: int) -> np.ndarray:
    """Validate per-sample weights shared by the sorted-spikes and clusterless fits.

    Weights feed both the occupancy field and the spike counts, so an invalid
    array corrupts the whole encoding silently; validate once at the fit entry.

    Parameters
    ----------
    weights : np.ndarray, shape (n_time,)
        Per-sample weights (e.g. posterior state probabilities during EM).
    n_time : int
        Expected number of samples (``position.shape[0]``).

    Returns
    -------
    weights : np.ndarray, shape (n_time,)
        The weights as a float array.

    Raises
    ------
    ValidationError
        If ``weights`` is not 1-D of length ``n_time``, or contains non-finite or
        negative values.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.shape != (n_time,):
        raise ValidationError(
            f"weights must have shape ({n_time},), got {weights.shape}"
        )

    if not np.all(np.isfinite(weights)):
        raise ValidationError("weights must contain only finite values")

    if np.any(weights < 0):
        raise ValidationError("weights must be non-negative")

    return weights


def interpolate_weights_at_spike_times(
    spike_times: np.ndarray, position_time: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Per-spike weights: the per-sample ``weights`` linearly interpolated onto times.

    Spike times are assumed already clipped to ``[position_time[0], position_time[-1]]``,
    so 1-D ``np.interp`` matches ``interpn`` without building an interpolator (and does
    not extrapolate). Shared by the clusterless and sorted-spikes encoding fits.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    position_time : np.ndarray, shape (n_time_position,)
    weights : np.ndarray, shape (n_time_position,)

    Returns
    -------
    spike_weights : np.ndarray, shape (n_spikes,)
    """
    return np.interp(
        np.asarray(spike_times), np.asarray(position_time), np.asarray(weights)
    )


def weighted_mean_rate(spike_weights: np.ndarray, weight_sum: float) -> float:
    """Weighted mean firing rate: weighted spike count / weighted occupancy time.

    Returns 0.0 when ``weight_sum`` is not positive (no effective training data), so a
    zero-weight electrode/group contributes nothing rather than dividing by zero.

    Parameters
    ----------
    spike_weights : np.ndarray, shape (n_spikes,)
        Per-spike weights (see :func:`interpolate_weights_at_spike_times`).
    weight_sum : float
        Sum of the per-sample weights (the weighted occupancy time).

    Returns
    -------
    mean_rate : float
    """
    return float(np.sum(spike_weights) / weight_sum) if weight_sum > 0 else 0.0


def get_position_at_time(
    time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: jnp.ndarray,
    env: Environment | None = None,
) -> np.ndarray:
    """Get the position at the time of each spike.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
    position : jnp.ndarray, shape (n_time_position, n_dims_position)
    spike_times : jnp.ndarray, shape (n_spikes,)
    env : Environment | None, optional
        The spatial environment, by default None

    Returns
    -------
    position_at_spike_times : np.ndarray, shape (n_spikes, n_dims_position)
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
            position_at_spike_times = np.array([])[:, None]

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

    Notes
    -----
    The per-dimension kernel product accumulates in linear space and underflows
    to 0 in float32 beyond a few dimensions or with tight bandwidths. Prefer
    :func:`log_kde` (or :meth:`KDEModel.predict_log`) when that regime matters.
    """
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))

    for dim_eval_points, dim_samples, dim_std in zip(
        eval_points.T, samples.T, std, strict=True
    ):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    # Double-where pattern: substitute safe denominator, then select result.
    # This avoids NaN in both forward pass and gradients.
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)
    return jnp.where(weight_sum > 0, (weights @ distance) / safe_weight_sum, 0.0)


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
    weights : jnp.ndarray, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    density_estimate : jnp.ndarray, shape (n_eval_points,)

    Notes
    -----
    Shares the linear-space underflow behaviour of :func:`kde`; prefer
    :func:`block_log_kde` when the density can drop below the float32 range.
    """
    n_eval_points = eval_points.shape[0]

    if n_eval_points == 0:
        return jnp.zeros((n_eval_points,))

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block densities and concatenate once. The blocks tile
    # [0, n_eval_points) exactly, so this is identical to filling a preallocated
    # array but avoids the O(n_eval * n_blocks) copies of a per-block
    # dynamic_update_slice.
    blocks = [
        kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval_points, block_size)
    ]
    return jnp.concatenate(blocks)


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
    for dim_eval, dim_samp, dim_std in zip(eval_points.T, samples.T, std, strict=True):
        K_log += log_gaussian_pdf(
            jnp.expand_dims(dim_eval, axis=0),
            jnp.expand_dims(dim_samp, axis=1),
            dim_std,
        )

    # True log-weight: a zero weight (log = -inf) drops the sample from both the
    # numerator and the denominator, exactly like the linear ``kde``. ``safe_log``
    # would floor a zero weight to LOG_EPS and leak that sample's kernel back in
    # (visible at an eval point near a zero-weight sample but far from the rest).
    safe_weights = jnp.where(weights > 0, weights, 1.0)
    log_w = jnp.where(weights > 0, jnp.log(safe_weights), -jnp.inf)  # (n_samp,)
    log_num = logsumexp(log_w[:, None] + K_log, axis=0)  # (n_eval,)
    log_den = logsumexp(log_w)  # scalar; -inf only when every weight is 0
    # All-zero weights -> no support anywhere. Return LOG_EPS (matching the linear
    # kde's 0.0, which callers floor) instead of NaN from -inf - (-inf).
    return jnp.where(jnp.isneginf(log_den), LOG_EPS, log_num - log_den)


def block_log_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Log kernel density estimation split into blocks over eval points.

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
    weights : jnp.ndarray, shape (n_samples,), optional
        Weights for each sample, by default None (uniform weights).

    Returns
    -------
    log_density_estimate : jnp.ndarray, shape (n_eval_points,)
    """
    n_eval = eval_points.shape[0]

    if n_eval == 0:
        return jnp.full((n_eval,), LOG_EPS)

    if weights is None:
        weights = jnp.ones((samples.shape[0],))

    # Collect per-block log-densities and concatenate once (see block_kde).
    blocks = [
        log_kde(eval_points[start : start + block_size], samples, std, weights)
        for start in range(0, n_eval, block_size)
    ]
    return jnp.concatenate(blocks)


@dataclass
class KDEModel:
    std: jnp.ndarray
    block_size: int | None = None
    samples_: jnp.ndarray | None = field(init=False, default=None)
    weights_: jnp.ndarray | None = field(init=False, default=None)

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
        if self.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size, self.weights_)

    def predict_log(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """
        Log-density version of predict(). Same inputs, returns log p(eval_points).
        """
        if self.samples_ is None:
            raise RuntimeError("This KDE instance is not fitted yet.")
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = as_std_array(self.std, eval_points.shape[1])
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

    # Double-where: substitute safe denominator first, then select result.
    # This avoids NaN in both forward pass and gradients.
    safe_denominator = jnp.where(condition, 1.0, denominator)
    return jnp.where(condition, eps, numerator / safe_denominator)


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
