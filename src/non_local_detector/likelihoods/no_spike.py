import numpy as np
import scipy.stats


def predict_no_spike_log_likelihood(
    time,
    spike_times: np.ndarray,
    no_spike_rate: float = 1e-10,
    sampling_frequency: float = 500.0,
):
    n_neurons = len(spike_times)
    no_spike_rates = np.ones((n_neurons,)) * no_spike_rate / sampling_frequency
    spike_count_per_time_bin = np.stack(
        [
            np.bincount(
                np.digitize(neuron_spike_times, time[1:-1]),
                minlength=time.shape[0],
            )
            for neuron_spike_times in spike_times
        ],
        axis=1,
    )
    return np.sum(
        scipy.stats.poisson.logpmf(spike_count_per_time_bin, no_spike_rates),
        axis=-1,
        keepdims=True,
    )
