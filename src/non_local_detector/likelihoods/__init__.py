from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood
from non_local_detector.likelihoods.sorted_spikes_glm_jax import (
    fit_sorted_spikes_glm_jax_encoding_model,
    predict_sorted_spikes_glm_jax_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde_jax import (
    fit_sorted_spikes_kde_jax_encoding_model,
    predict_sorted_spikes_kde_jax_log_likelihood,
)

_SORTED_SPIKES_ALGORITHMS = {
    "no_spike": (None, predict_no_spike_log_likelihood),
    "sorted_spikes_glm_jax": (
        fit_sorted_spikes_glm_jax_encoding_model,
        predict_sorted_spikes_glm_jax_log_likelihood,
    ),
    "sorted_spikes_kde_jax": (
        fit_sorted_spikes_kde_jax_encoding_model,
        predict_sorted_spikes_kde_jax_log_likelihood,
    ),
}
_CLUSTERLESS_ALGORITHMS = {
    "no_spike": (None, predict_no_spike_log_likelihood),
    "clusterless_kde_jax": (None, None),
}
