from collections.abc import Callable

from non_local_detector.likelihoods import (
    clusterless_diffusion,
    clusterless_gmm,
    clusterless_kde,
    clusterless_mark_glm,
)
from non_local_detector.likelihoods.clusterless_diffusion import (
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_gmm import (  # noqa
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_mark_glm import (
    fit_clusterless_mark_glm_encoding_model,
    predict_clusterless_mark_glm_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import (  # noqa
    fit_mark_references,
)
from non_local_detector.likelihoods.no_spike import (  # noqa
    predict_no_spike_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    fit_sorted_spikes_diffusion_encoding_model,
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    fit_sorted_spikes_mrf_encoding_model,
    predict_sorted_spikes_mrf_log_likelihood,
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
    "sorted_spikes_diffusion": (
        fit_sorted_spikes_diffusion_encoding_model,
        predict_sorted_spikes_diffusion_log_likelihood,
    ),
    "sorted_spikes_mrf": (
        fit_sorted_spikes_mrf_encoding_model,
        predict_sorted_spikes_mrf_log_likelihood,
    ),
}
_CLUSTERLESS_ALGORITHMS: dict[str, tuple[Callable, Callable]] = {
    "clusterless_kde": (
        fit_clusterless_kde_encoding_model,
        predict_clusterless_kde_log_likelihood,
    ),
    "clusterless_gmm": (
        fit_clusterless_gmm_encoding_model,
        predict_clusterless_gmm_log_likelihood,
    ),
    "clusterless_diffusion": (
        fit_clusterless_diffusion_encoding_model,
        predict_clusterless_diffusion_log_likelihood,
    ),
    "clusterless_mark_glm": (
        fit_clusterless_mark_glm_encoding_model,
        predict_clusterless_mark_glm_log_likelihood,
    ),
}
# Each clusterless algorithm's mark-kernel width, from its fit parameters, which
# the mark reference adds in quadrature to each feature's training spread (see
# ``fit_mark_references``).
_CLUSTERLESS_MARK_REFERENCE_SCALES: dict[str, Callable[..., object]] = {
    "clusterless_kde": clusterless_kde.mark_reference_scale,
    "clusterless_gmm": clusterless_gmm.mark_reference_scale,
    "clusterless_diffusion": clusterless_diffusion.mark_reference_scale,
    "clusterless_mark_glm": clusterless_mark_glm.mark_reference_scale,
}
