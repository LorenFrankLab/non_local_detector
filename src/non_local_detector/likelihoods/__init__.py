from collections.abc import Callable

from non_local_detector.likelihoods.clusterless_gmm import (  # noqa
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde_log import (  # noqa
    fit_clusterless_kde_encoding_model as fit_clusterless_kde_log_encoding_model,
)
from non_local_detector.likelihoods.clusterless_kde_log import (  # noqa
    predict_clusterless_kde_log_likelihood as predict_clusterless_kde_log_log_likelihood,
)
from non_local_detector.likelihoods.no_spike import (  # noqa
    predict_no_spike_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

_SORTED_SPIKES_ALGORITHMS: dict[str, tuple[Callable, Callable]] = {
    "sorted_spikes_glm": (
        fit_sorted_spikes_glm_encoding_model,
        predict_sorted_spikes_glm_log_likelihood,
    ),
    "sorted_spikes_kde": (
        fit_sorted_spikes_kde_encoding_model,
        predict_sorted_spikes_kde_log_likelihood,
    ),
}
_CLUSTERLESS_ALGORITHMS: dict[str, tuple[Callable, Callable]] = {
    "clusterless_kde": (
        fit_clusterless_kde_encoding_model,
        predict_clusterless_kde_log_likelihood,
    ),
    "clusterless_kde_log": (
        fit_clusterless_kde_log_encoding_model,
        predict_clusterless_kde_log_log_likelihood,
    ),
    "clusterless_gmm": (
        fit_clusterless_gmm_encoding_model,
        predict_clusterless_gmm_log_likelihood,
    ),
}
