from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.no_spike import (
    predict_no_spike_log_likelihood,
)  # noqa
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

_SORTED_SPIKES_ALGORITHMS = {
    "sorted_spikes_glm": (
        fit_sorted_spikes_glm_encoding_model,
        predict_sorted_spikes_glm_log_likelihood,
    ),
    "sorted_spikes_kde": (
        fit_sorted_spikes_kde_encoding_model,
        predict_sorted_spikes_kde_log_likelihood,
    ),
}
_CLUSTERLESS_ALGORITHMS = {
    "clusterless_kde": (
        fit_clusterless_kde_encoding_model,
        predict_clusterless_kde_log_likelihood,
    ),
}
