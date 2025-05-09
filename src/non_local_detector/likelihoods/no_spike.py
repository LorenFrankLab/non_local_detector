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
from tqdm.autonotebook import tqdm

from non_local_detector.likelihoods.common import get_spikecount_per_time_bin


def predict_no_spike_log_likelihood(
    time: np.ndarray,
    spike_times: list[list[float]],
    no_spike_rate: float = 1e-10,
) -> jnp.ndarray:
    """Return the log likelihood of low spike rate for each time bin.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time bins.
    spike_times : list[list[float]]
        Spike times for each neuron.
    no_spike_rate : float, optional
        Rate of low spiking process, by default 1e-10

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
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
