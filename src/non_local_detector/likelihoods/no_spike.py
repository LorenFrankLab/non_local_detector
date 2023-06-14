import numpy as np
import scipy.stats


def predict_no_spike_log_likelihood(
    spikes,
    no_spike_rate: float = 1e-10,
    sampling_frequency: float = 500.0,
):
    n_neurons = spikes.shape[1]
    no_spike_rates = np.ones((n_neurons,)) * no_spike_rate / sampling_frequency
    return np.sum(
        scipy.stats.poisson.logpmf(spikes, no_spike_rates), axis=-1, keepdims=True
    )
