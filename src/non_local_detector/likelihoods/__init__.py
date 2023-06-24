from non_local_detector.likelihoods.clusterless_kde_jax import (
    fit_clusterless_kde_jax_encoding_model,
    predict_clusterless_kde_jax_log_likelihood,
)
from non_local_detector.likelihoods.no_spike import (
    predict_no_spike_log_likelihood,
)  # noqa
from non_local_detector.likelihoods.sorted_spikes_glm_jax import (
    fit_sorted_spikes_glm_jax_encoding_model,
    predict_sorted_spikes_glm_jax_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde_jax import (
    fit_sorted_spikes_kde_jax_encoding_model,
    predict_sorted_spikes_kde_jax_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde_np import (
    fit_sorted_spikes_kde_np_encoding_model,
    predict_sorted_spikes_kde_np_log_likelihood,
)

_SORTED_SPIKES_ALGORITHMS = {
    "sorted_spikes_glm_jax": (
        fit_sorted_spikes_glm_jax_encoding_model,
        predict_sorted_spikes_glm_jax_log_likelihood,
    ),
    "sorted_spikes_kde_np": (
        fit_sorted_spikes_kde_np_encoding_model,
        predict_sorted_spikes_kde_np_log_likelihood,
    ),
    "sorted_spikes_kde_jax": (
        fit_sorted_spikes_kde_jax_encoding_model,
        predict_sorted_spikes_kde_jax_log_likelihood,
    ),
}
_CLUSTERLESS_ALGORITHMS = {
    "clusterless_kde_jax": (
        fit_clusterless_kde_jax_encoding_model,
        predict_clusterless_kde_jax_log_likelihood,
    ),
}
