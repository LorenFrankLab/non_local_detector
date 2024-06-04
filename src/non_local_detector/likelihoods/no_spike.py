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
