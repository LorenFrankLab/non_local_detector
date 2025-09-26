"""Log-likelihood computation for a very low firing rate Poisson firing model.

This module provides a function to calculate the log-likelihood of observed spike
counts under a simple baseline model. This model is intended to quiescent
times when the population of neurons is not firing, or is firing at a very low rate.
Examples of this are times when the animal is immobile and the hippocampus has
burst like activity.

The primary function `predict_no_spike_log_likelihood` computes this value
for specified time bins based on the provided spike times and baseline rate.
It utilizes JAX for efficient computation.
"""

import jax.numpy as jnp
import jax.scipy
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.likelihoods.common import get_spikecount_per_time_bin


def predict_no_spike_log_likelihood(
    time: np.ndarray,
    spike_times: list[list[float]],
    no_spike_rate: float = 1e-10,
) -> jnp.ndarray:
    """Return the log likelihood of low spike rate for each time bin.

    This function computes the log-likelihood under a Poisson model with
    very low firing rates, typically used during quiescent periods when
    neural activity is minimal or during immobility periods.

    Parameters
    ----------
    time : np.ndarray, shape (n_time + 1,)
        Time bin edges for likelihood computation. The number of bins
        is len(time) - 1.
    spike_times : list[list[float]]
        Nested list where each inner list contains spike times for one neuron.
        Length equals number of neurons in the population.
    no_spike_rate : float, default=1e-10
        Expected firing rate during no-spike periods in Hz. Should be very
        small to represent baseline/quiescent activity levels.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
        Log-likelihood values for each time bin under the no-spike model.

    Notes
    -----
    The model assumes Poisson firing with rate `no_spike_rate` scaled by
    the time bin duration. This is appropriate for modeling background
    activity during periods of behavioral quiescence such as slow-wave sleep
    or immobile periods when place cells show minimal spatial selectivity.

    The log-likelihood is computed as:

    .. math::
        \\log P(n|\\lambda) = n \\log(\\lambda \\Delta t) - \\lambda \\Delta t

    where n is the spike count, λ is the firing rate, and Δt is the bin duration.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 100)  # 99 time bins
    >>> spike_times = [[] for _ in range(5)]  # 5 neurons, no spikes
    >>> log_lik = predict_no_spike_log_likelihood(time, spike_times)
    >>> log_lik.shape
    (99, 1)

    >>> # With some sparse spikes
    >>> spike_times = [[1.0, 5.0], [], [8.5], [], []]
    >>> log_lik = predict_no_spike_log_likelihood(time, spike_times, no_spike_rate=1e-8)
    >>> log_lik.shape
    (99, 1)
    """
    no_spike_rates = no_spike_rate * np.median(np.diff(time))
    no_spike_log_likelihood = jnp.zeros((time.shape[0],))

    for neuron_spike_times in tqdm(
        spike_times, unit="cell", desc="No Spike Likelihood"
    ):
        no_spike_log_likelihood += (
            jax.scipy.special.xlogy(
                get_spikecount_per_time_bin(neuron_spike_times, time), no_spike_rates
            )
            - no_spike_rates
        )

    return no_spike_log_likelihood[:, None]
