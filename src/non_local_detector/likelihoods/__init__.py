"""Likelihood algorithms and the registries that map algorithm names to them."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
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


@dataclass(frozen=True)
class LikelihoodAlgorithm:
    """A registered likelihood algorithm: its fit, predict and defaults.

    Attributes
    ----------
    fit : Callable[..., dict[str, Any]]
        Fits the encoding model and returns it as a dict of keyword arguments
        for ``predict``.
    predict : Callable[..., jax.Array]
        Returns the log likelihood, float32 with shape
        ``(n_rows, n_interior_bins)``, or ``(n_rows, 1)`` when ``is_local``.
    default_params : Mapping[str, Any]
        The parameters ``fit`` receives when a detector's
        ``*_algorithm_params`` is None, beyond the fit's own defaults; a user
        dict replaces them. Stored as a read-only copy. By default empty.
    mark_reference_scale : Callable[..., object] or None
        Clusterless algorithms only: the mark-kernel width, from the fit
        parameters, which the mark reference adds in quadrature to each
        feature's training spread (see ``fit_mark_references``). By default
        None.

    Notes
    -----
    A detector stores the dict ``fit`` returns as its encoding model and calls
    ``predict(position_time, position, *spike_data, **encoding_model,
    **detector_keywords)``, where ``spike_data`` is ``spike_times`` (and
    ``spike_waveform_features`` for clusterless data) and the keywords are
    ``_PREDICT_KEYWORDS`` (``_CLUSTERLESS_PREDICT_KEYWORDS``). So the encoding
    model must hold ``_ENCODING_MODEL_KEYS``, must not hold a data or keyword
    name, and must provide every required ``predict`` parameter; unless
    ``predict`` takes ``**kwargs``, each of its keys must be a ``predict``
    parameter. ``tests/likelihoods/test_algorithm_contract.py`` checks every
    registered algorithm.
    """

    fit: Callable[..., dict[str, Any]]
    predict: Callable[..., Any]
    default_params: Mapping[str, Any] = field(default_factory=dict)
    mark_reference_scale: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        # A read-only copy, so a caller can mutate neither the registered
        # defaults nor, through them, every later fit.
        object.__setattr__(
            self, "default_params", MappingProxyType(dict(self.default_params))
        )


# The keyword arguments a detector passes to every predict besides the fit's
# encoding model (see ``LikelihoodAlgorithm``), and those of a clusterless
# predict, which also scores marks against the detector's mark reference.
_PREDICT_KEYWORDS: frozenset[str] = frozenset(
    {"time_edges", "is_local", "row_slice", "_spike_time_order", "_time_grid"}
)
_CLUSTERLESS_PREDICT_KEYWORDS: frozenset[str] = _PREDICT_KEYWORDS | {"mark_reference"}
# The keys every fit returns. Each predict checks ``rate_units`` and
# ``environment_grid`` against ``environment`` (``validate_encoding_contract``);
# the detectors read ``rate_units`` and ``disable_progress_bar``, and model
# checking ``environment``. ``encoding_exposure_seconds`` is the fit's weighted
# occupancy time, for diagnostics.
_ENCODING_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "environment",
        "environment_grid",
        "rate_units",
        "encoding_exposure_seconds",
        "disable_progress_bar",
    }
)

# The algorithms a detector's ``*_algorithm`` names. Only sorted KDE has default
# params; bandwidths come from the fits (position_std sqrt(12.5)).
_SORTED_SPIKES_ALGORITHMS: dict[str, LikelihoodAlgorithm] = {
    "sorted_spikes_glm": LikelihoodAlgorithm(
        fit_sorted_spikes_glm_encoding_model,
        predict_sorted_spikes_glm_log_likelihood,
    ),
    "sorted_spikes_kde": LikelihoodAlgorithm(
        fit_sorted_spikes_kde_encoding_model,
        predict_sorted_spikes_kde_log_likelihood,
        default_params={"block_size": 10_000},
    ),
    "sorted_spikes_diffusion": LikelihoodAlgorithm(
        fit_sorted_spikes_diffusion_encoding_model,
        predict_sorted_spikes_diffusion_log_likelihood,
    ),
    "sorted_spikes_mrf": LikelihoodAlgorithm(
        fit_sorted_spikes_mrf_encoding_model,
        predict_sorted_spikes_mrf_log_likelihood,
    ),
}
_CLUSTERLESS_ALGORITHMS: dict[str, LikelihoodAlgorithm] = {
    "clusterless_kde": LikelihoodAlgorithm(
        fit_clusterless_kde_encoding_model,
        predict_clusterless_kde_log_likelihood,
        mark_reference_scale=clusterless_kde.mark_reference_scale,
    ),
    "clusterless_gmm": LikelihoodAlgorithm(
        fit_clusterless_gmm_encoding_model,
        predict_clusterless_gmm_log_likelihood,
        mark_reference_scale=clusterless_gmm.mark_reference_scale,
    ),
    "clusterless_diffusion": LikelihoodAlgorithm(
        fit_clusterless_diffusion_encoding_model,
        predict_clusterless_diffusion_log_likelihood,
        mark_reference_scale=clusterless_diffusion.mark_reference_scale,
    ),
    "clusterless_mark_glm": LikelihoodAlgorithm(
        fit_clusterless_mark_glm_encoding_model,
        predict_clusterless_mark_glm_log_likelihood,
        mark_reference_scale=clusterless_mark_glm.mark_reference_scale,
    ),
}


def _resolve_fit_params(
    algorithms: dict[str, LikelihoodAlgorithm],
    algorithm: str,
    params: dict[str, Any] | None,
    params_name: str,
) -> dict[str, Any]:
    """Return a fresh dict of the fit parameters for a detector's algorithm.

    Parameters
    ----------
    algorithms : dict[str, LikelihoodAlgorithm]
        The registry holding ``algorithm``.
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
    record = algorithms[algorithm]
    resolved = dict(record.default_params if params is None else params)
    parameters = inspect.signature(record.fit).parameters
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
