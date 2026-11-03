import inspect
from collections.abc import Callable
from typing import Any

from non_local_detector.exceptions import ValidationError
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
from non_local_detector.likelihoods.clusterless_gmm import (
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
from non_local_detector.likelihoods.mark_reference import (
    fit_mark_references,
)
from non_local_detector.likelihoods.no_spike import (
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

_SORTED_SPIKES_ALGORITHMS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
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
_CLUSTERLESS_ALGORITHMS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
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
# The parameters each algorithm's fit receives when a detector's
# ``*_algorithm_params`` is None, beyond the fit's own defaults; a user dict
# replaces them. Bandwidths come from the fits (position_std sqrt(12.5)).
_SORTED_SPIKES_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "sorted_spikes_glm": {},
    "sorted_spikes_kde": {"block_size": 10_000},
    "sorted_spikes_diffusion": {},
    "sorted_spikes_mrf": {},
}
_CLUSTERLESS_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "clusterless_kde": {},
    "clusterless_gmm": {},
    "clusterless_diffusion": {},
    "clusterless_mark_glm": {},
}


def _resolve_fit_params(
    algorithms: dict[str, tuple[Callable[..., Any], Callable[..., Any]]],
    default_params: dict[str, dict[str, Any]],
    algorithm: str,
    params: dict[str, Any] | None,
    params_name: str,
) -> dict[str, Any]:
    """Return a fresh dict of the fit parameters for a detector's algorithm.

    Parameters
    ----------
    algorithms : dict
        The registry holding ``algorithm``'s ``(fit, predict)`` pair.
    default_params : dict
        The registry's per-algorithm defaults.
    algorithm : str
        A key of ``algorithms``.
    params : dict or None
        The detector's algorithm params. None selects the algorithm's
        defaults; a dict replaces them.
    params_name : str
        The detector argument ``params`` came from, for the error message.

    Returns
    -------
    params : dict
        A copy, so mutating it changes neither the defaults nor the user's dict.

    Raises
    ------
    ValidationError
        If ``algorithm`` is not registered, or a key is not a fit parameter
        the user may set: one with a default,
        other than ``weights`` and the private ``_``-prefixed ones. The
        detector supplies the fit's data, environment and weights itself. A
        fit that takes ``**kwargs`` is not checked.
    """
    if algorithm not in algorithms:
        algorithm_name = params_name.removesuffix("_params")
        msg = f"Unknown {algorithm_name} {algorithm!r}"
        raise ValidationError(
            msg,
            expected=f"one of {sorted(algorithms)}",
            got=repr(algorithm),
        )
    resolved = dict(default_params[algorithm] if params is None else params)
    parameters = inspect.signature(algorithms[algorithm][0]).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return resolved  # a fit taking ``**kwargs`` accepts any name
    accepted = [
        name
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and name != "weights"
        and not name.startswith("_")
    ]
    unknown = sorted(set(resolved) - set(accepted))
    if unknown:
        msg = f"unknown {algorithm} parameter(s) in {params_name}: {', '.join(unknown)}"
        raise ValidationError(
            msg,
            expected=f"parameters among: {', '.join(accepted)}",
            got=", ".join(f"{name}={resolved[name]!r}" for name in unknown),
            hint="Check the spelling, and that the parameter belongs to "
            f"{algorithm}: each algorithm accepts only its own fit parameters.",
        )
    return resolved


# Each clusterless algorithm's mark-kernel width, from its fit parameters, which
# the mark reference adds in quadrature to each feature's training spread (see
# ``fit_mark_references``).
_CLUSTERLESS_MARK_REFERENCE_SCALES: dict[str, Callable[..., object]] = {
    "clusterless_kde": clusterless_kde.mark_reference_scale,
    "clusterless_gmm": clusterless_gmm.mark_reference_scale,
    "clusterless_diffusion": clusterless_diffusion.mark_reference_scale,
    "clusterless_mark_glm": clusterless_mark_glm.mark_reference_scale,
}
