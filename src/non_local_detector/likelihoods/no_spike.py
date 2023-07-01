import jax.numpy as jnp
import jax.scipy


def get_spikecount_per_time_bin(spike_times, time):
    spike_times = spike_times[
        jnp.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    return jnp.bincount(
        jnp.digitize(spike_times, time[1:-1]),
        minlength=time.shape[0],
    )


def predict_no_spike_log_likelihood(
    time,
    spike_times: jnp.ndarray,
    no_spike_rate: float = 1e-10,
    sampling_frequency: float = 500.0,
):
    no_spike_rates = no_spike_rate / sampling_frequency
    spike_count_per_time_bin = jnp.stack(
        [
            get_spikecount_per_time_bin(neuron_spike_times, time)
            for neuron_spike_times in spike_times
        ],
        axis=1,
    )
    return jnp.sum(
        jax.scipy.special.xlogy(spike_count_per_time_bin, no_spike_rates)
        - no_spike_rates,
        axis=-1,
        keepdims=True,
    )
