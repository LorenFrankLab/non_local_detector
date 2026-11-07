"""Likelihood algorithms and the registries that map algorithm names to them."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

import jax
import numpy as np
from numpy.typing import ArrayLike

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    clusterless_diffusion,
    clusterless_gmm,
    clusterless_kde,
    clusterless_mark_glm,
)
from non_local_detector.likelihoods.clusterless_diffusion import (
    ClusterlessDiffusionEncodingModel,
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_gmm import (
    ClusterlessGMMEncodingModel,
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    ClusterlessKDEEncodingModel,
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_mark_glm import (
    ClusterlessMarkGLMEncodingModel,
    fit_clusterless_mark_glm_encoding_model,
    predict_clusterless_mark_glm_log_likelihood,
)
from non_local_detector.likelihoods.encoding_model import (
    ClusterlessEncodingModel,
    EncodingModel,
    SortedSpikesEncodingModel,
)
from non_local_detector.likelihoods.mark_reference import (
    fit_mark_references,
)
from non_local_detector.likelihoods.no_spike import (
    predict_no_spike_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    SortedSpikesDiffusionEncodingModel,
    fit_sorted_spikes_diffusion_encoding_model,
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    SortedSpikesGLMEncodingModel,
    fit_sorted_spikes_glm_encoding_model,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    SortedSpikesKDEEncodingModel,
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    SortedSpikesMRFEncodingModel,
    fit_sorted_spikes_mrf_encoding_model,
    predict_sorted_spikes_mrf_log_likelihood,
)


# The protocols type the data arguments a detector passes. The parameters after
# them, the fit's own and, for predict, the encoding model's keys and the
# detector's keywords, are left to mypy unchecked (``*args: Any, **kwargs:
# Any``) and checked against each signature by the contract tests instead. A
# detector passes predict its data positionally, and a fit's by keyword.
class SortedSpikesFit(Protocol):
    """The fit of a sorted-spikes algorithm, as a detector calls it.

    A detector passes NumPy arrays and lists of them; a fit accepts any
    array-likes.
    """

    def __call__(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: Sequence[ArrayLike],
        environment: Environment,
        weights: ArrayLike | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SortedSpikesEncodingModel:
        """Call the fit; see the class docstring."""
        ...


class ClusterlessFit(Protocol):
    """The fit of a clusterless algorithm, as a detector calls it.

    A detector passes NumPy arrays and lists of them; a fit accepts any
    array-likes.
    """

    def __call__(
        self,
        position_time: ArrayLike,
        position: ArrayLike,
        spike_times: Sequence[ArrayLike],
        spike_waveform_features: Sequence[ArrayLike],
        environment: Environment,
        weights: ArrayLike | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> ClusterlessEncodingModel:
        """Call the fit; see the class docstring."""
        ...


class SortedSpikesPredict(Protocol):
    """The predict of a sorted-spikes algorithm, as a detector calls it.

    ``position_time`` and ``position`` are None when no state reads position
    (no Local state and no non-local position penalty); a detector then
    calls predict with ``is_local=False`` only.
    """

    def __call__(
        self,
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        spike_times: Sequence[ArrayLike],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        """Call the predict; see the class docstring."""
        ...


class ClusterlessPredict(Protocol):
    """The predict of a clusterless algorithm, as a detector calls it.

    ``position_time`` and ``position`` are None when no state reads position
    (no Local state and no non-local position penalty); a detector then
    calls predict with ``is_local=False`` only.
    """

    def __call__(
        self,
        position_time: ArrayLike | None,
        position: ArrayLike | None,
        spike_times: Sequence[ArrayLike],
        spike_waveform_features: Sequence[ArrayLike],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        """Call the predict; see the class docstring."""
        ...


# A record's fit and predict protocols: ``SortedSpikesFit`` and
# ``SortedSpikesPredict``, or ``ClusterlessFit`` and ``ClusterlessPredict``.
FitT = TypeVar("FitT")
PredictT = TypeVar("PredictT")


@dataclass(frozen=True)
class LikelihoodAlgorithm(Generic[FitT, PredictT]):
    """A registered likelihood algorithm: its fit, predict and defaults.

    Attributes
    ----------
    fit : SortedSpikesFit or ClusterlessFit
        Fits the encoding model and returns it as a dict of keyword arguments
        for ``predict``.
    predict : SortedSpikesPredict or ClusterlessPredict
        Returns the log likelihood, float32 with shape
        ``(n_rows, n_interior_bins)``, or ``(n_rows, 1)`` when ``is_local``.
        It must accept ``position_time=None`` and ``position=None`` when
        ``is_local`` is False: a detector passes None for both when no state
        reads position.
    encoding_model_type : type
        The ``TypedDict`` that annotates ``fit``'s return and describes the
        dict it returns. It derives from ``SortedSpikesEncodingModel`` or
        ``ClusterlessEncodingModel``.
    default_params : Mapping[str, Any]
        The parameters ``fit`` receives when a detector's
        ``*_algorithm_params`` is None, beyond the fit's own defaults; a user
        dict replaces them. Stored as a read-only copy. By default empty.
    execution_settings : frozenset of str
        Fit parameters that set how ``predict`` runs (memory budgets, block
        and tile sizes, progress bars) rather than what is fitted. The fit
        records each in the encoding model, and a detector passes ``predict``
        the value its ``*_algorithm_params`` sets now, if any, so these can
        change without refitting (see ``_execution_settings``). By default
        ``{"disable_progress_bar"}``.

    Notes
    -----
    A detector stores the dict ``fit`` returns as its encoding model and calls
    ``predict(position_time, position, *spike_data, **encoding_model,
    **detector_keywords)``, where ``spike_data`` is ``spike_times`` (and
    ``spike_waveform_features`` for clusterless data) and the keywords are
    ``_PREDICT_KEYWORDS`` (``_CLUSTERLESS_PREDICT_KEYWORDS``). So the
    encoding model must hold exactly the keys of ``encoding_model_type``,
    with ``algorithm`` set to the registered name; must not hold a data or
    keyword name; must provide every named ``predict`` parameter the detector
    does not, so none is left at its default; and each of its keys must be a
    ``predict`` parameter (predict takes no ``**kwargs``). The data arguments
    are NumPy arrays (lists of them for spike data), and a fit or predict
    takes them as array-likes (see ``SortedSpikesFit`` and the other
    protocols, which type them for mypy). Records pickle and deep-copy.
    ``tests/likelihoods/test_algorithm_contract.py`` checks every registered
    algorithm.

    A detector refuses a ``predict`` of the wrong shape, and raises
    ``DataError`` on a NaN or ``+inf`` once a pass has seen it. No code can
    check the rest of what the model needs:

    - The log likelihood is absolute, ``sum n log(rate dt) - rate dt`` over
      the neurons with rates in Hz, not relative to a position-free rate or
      to the row's maximum. No-Spike is compared against that reference, so a
      row correct only up to a constant moves probability between states.
    - Values are finite. ``-inf`` means the state is impossible, a row of all
      ``-inf`` is only logged, and a pass under ``jax.jit`` cannot raise.
    - ``fit`` calls ``prepare_encoding_support``. Spike counts use the
      ``weights`` it returns and occupancy the ``exposure_weights``, which
      multiply each sample's weight by its exposure in seconds, so a fit that
      ignores them counts time the detector left out.
    """

    fit: FitT
    predict: PredictT
    encoding_model_type: type
    default_params: Mapping[str, Any] = field(default_factory=dict)
    execution_settings: frozenset[str] = frozenset({"disable_progress_bar"})

    def __post_init__(self) -> None:
        # A read-only copy, so a caller can mutate neither the registered
        # defaults nor, through them, every later fit.
        object.__setattr__(
            self, "default_params", MappingProxyType(dict(self.default_params))
        )

    def __getstate__(self) -> dict[str, Any]:
        # A mapping proxy cannot be pickled, so store a plain dict;
        # ``copy.deepcopy`` uses this too.
        return {**vars(self), "default_params": dict(self.default_params)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self.__post_init__()


@dataclass(frozen=True, kw_only=True)
class ClusterlessLikelihoodAlgorithm(
    LikelihoodAlgorithm[ClusterlessFit, ClusterlessPredict]
):
    """A registered clusterless likelihood algorithm.

    Attributes
    ----------
    mark_reference_scale : Callable[..., object]
        The mark-kernel width, from the fit parameters, which the mark
        reference adds in quadrature to each feature's training spread (see
        ``fit_mark_references``).
    electrode_intensity : str or None
        Model checking only: the name of the class in
        ``non_local_detector.model_checking._clusterless_models`` that builds
        one electrode's fitted mark intensity from the encoding model and the
        electrode index. None, the default, if model checking does not
        support the algorithm. A name rather than the class, so the
        likelihoods do not import model checking, which imports them.
    """

    mark_reference_scale: Callable[..., object]
    electrode_intensity: str | None = None


# The keyword arguments a detector passes to every predict besides the fit's
# encoding model (see ``LikelihoodAlgorithm``), and those of a clusterless
# predict, which also scores marks against the detector's mark reference.
_PREDICT_KEYWORDS: frozenset[str] = frozenset(
    {"time_edges", "is_local", "row_slice", "_spike_time_order", "_time_grid"}
)
_CLUSTERLESS_PREDICT_KEYWORDS: frozenset[str] = _PREDICT_KEYWORDS | {"mark_reference"}
# The keys every fit returns (see ``EncodingModel``). Each predict checks
# ``rate_units`` and ``environment_grid`` against ``environment``
# (``validate_encoding_contract``); the detectors read ``algorithm``,
# ``rate_units`` and ``disable_progress_bar``, and model checking
# ``algorithm`` and ``environment``. ``encoding_exposure_seconds`` is the
# fit's weighted occupancy time, for diagnostics.
_ENCODING_MODEL_KEYS: frozenset[str] = frozenset(EncodingModel.__required_keys__)

# The algorithms a detector's ``*_algorithm`` names. Only sorted KDE has default
# params; bandwidths come from the fits (position_std sqrt(12.5)).
_SORTED_SPIKES_ALGORITHMS: dict[
    str, LikelihoodAlgorithm[SortedSpikesFit, SortedSpikesPredict]
] = {
    "sorted_spikes_glm": LikelihoodAlgorithm(
        fit_sorted_spikes_glm_encoding_model,
        predict_sorted_spikes_glm_log_likelihood,
        SortedSpikesGLMEncodingModel,
    ),
    "sorted_spikes_kde": LikelihoodAlgorithm(
        fit_sorted_spikes_kde_encoding_model,
        predict_sorted_spikes_kde_log_likelihood,
        SortedSpikesKDEEncodingModel,
        default_params={"block_size": 10_000},
    ),
    "sorted_spikes_diffusion": LikelihoodAlgorithm(
        fit_sorted_spikes_diffusion_encoding_model,
        predict_sorted_spikes_diffusion_log_likelihood,
        SortedSpikesDiffusionEncodingModel,
    ),
    "sorted_spikes_mrf": LikelihoodAlgorithm(
        fit_sorted_spikes_mrf_encoding_model,
        predict_sorted_spikes_mrf_log_likelihood,
        SortedSpikesMRFEncodingModel,
    ),
}
_CLUSTERLESS_ALGORITHMS: dict[str, ClusterlessLikelihoodAlgorithm] = {
    "clusterless_kde": ClusterlessLikelihoodAlgorithm(
        fit_clusterless_kde_encoding_model,
        predict_clusterless_kde_log_likelihood,
        ClusterlessKDEEncodingModel,
        execution_settings=frozenset(
            {
                "block_size",
                "enc_tile_size",
                "pos_tile_size",
                "use_streaming",
                "disable_progress_bar",
            }
        ),
        mark_reference_scale=clusterless_kde.mark_reference_scale,
        electrode_intensity="KDEElectrode",
    ),
    "clusterless_gmm": ClusterlessLikelihoodAlgorithm(
        fit_clusterless_gmm_encoding_model,
        predict_clusterless_gmm_log_likelihood,
        ClusterlessGMMEncodingModel,
        execution_settings=frozenset(
            {"spike_block_size", "bin_tile_size", "disable_progress_bar"}
        ),
        mark_reference_scale=clusterless_gmm.mark_reference_scale,
        electrode_intensity="GMMElectrode",
    ),
    "clusterless_diffusion": ClusterlessLikelihoodAlgorithm(
        fit_clusterless_diffusion_encoding_model,
        predict_clusterless_diffusion_log_likelihood,
        ClusterlessDiffusionEncodingModel,
        execution_settings=frozenset(
            {"block_size", "memory_budget", "disable_progress_bar"}
        ),
        mark_reference_scale=clusterless_diffusion.mark_reference_scale,
        electrode_intensity="DiffusionElectrode",
    ),
    # Model checking does not implement the mark GLM's intensity.
    "clusterless_mark_glm": ClusterlessLikelihoodAlgorithm(
        fit_clusterless_mark_glm_encoding_model,
        predict_clusterless_mark_glm_log_likelihood,
        ClusterlessMarkGLMEncodingModel,
        execution_settings=frozenset({"spike_block_size", "disable_progress_bar"}),
        mark_reference_scale=clusterless_mark_glm.mark_reference_scale,
    ),
}


# Fit parameters a detector sets from its own ``fit`` arguments. Each fit
# accepts them for direct calls, but a detector passes it the encoding support
# it built from them (``_encoding_support``), which the fit uses instead, so in
# ``*_algorithm_params`` they would be ignored.
_DETECTOR_FIT_PARAMETERS: frozenset[str] = frozenset(
    {"encoding_time_range", "valid_position_intervals"}
)


def _resolve_fit_params(
    algorithms: Mapping[str, LikelihoodAlgorithm[Any, Any]],
    algorithm: str,
    params: dict[str, Any] | None,
    params_name: str,
) -> dict[str, Any]:
    """Return a fresh dict of the fit parameters for a detector's algorithm.

    Parameters
    ----------
    algorithms : Mapping[str, LikelihoodAlgorithm]
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
        the user may set: one with a default, other than ``weights``, the
        encoding-support parameters (``_DETECTOR_FIT_PARAMETERS``) and the
        private ``_``-prefixed ones. The detector supplies the fit's data,
        environment, weights and encoding support itself. A fit that takes
        ``**kwargs`` is not checked for other names.
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
    if misplaced := sorted(set(resolved) & _DETECTOR_FIT_PARAMETERS):
        msg = (
            f"{', '.join(misplaced)} in {params_name} would be ignored: the "
            "detector builds the encoding support from its own fit arguments "
            "and passes that to the algorithm"
        )
        raise ValidationError(
            msg,
            got=", ".join(f"{name}={resolved[name]!r}" for name in misplaced),
            hint=f"Remove it from {params_name} and pass it to the detector "
            "instead: "
            + "; ".join(
                f"detector.fit(..., {name}={resolved[name]!r})" for name in misplaced
            )
            + " (estimate_parameters takes it too).",
        )
    parameters = inspect.signature(record.fit).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return resolved  # a fit taking ``**kwargs`` accepts any name
    accepted = [
        name
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and name != "weights"
        and name not in _DETECTOR_FIT_PARAMETERS
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


def _execution_settings(
    record: LikelihoodAlgorithm[Any, Any], params: dict[str, Any] | None
) -> dict[str, Any]:
    """Return the execution settings a detector's algorithm params set, for predict.

    A detector passes these to ``predict`` in place of the values its encoding
    model recorded at fit, so memory, tiling and progress options can change
    after fitting. A setting the params do not name keeps its fitted value.
    Other keys are left to the next fit, which validates them, so a detector
    saved with keys its algorithm no longer accepts still decodes.

    Parameters
    ----------
    record : LikelihoodAlgorithm
        The detector's registered algorithm.
    params : dict or None
        The detector's current algorithm params; None (the defaults the fit
        used) sets none.

    Returns
    -------
    settings : dict
        Each of the algorithm's ``execution_settings`` that ``params`` names,
        with its value.
    """
    if params is None:
        return {}
    return {
        name: value
        for name, value in params.items()
        if name in record.execution_settings
    }


# The keys of each algorithm's encoding model as main saved them before
# encoding models recorded their algorithm (main at 66047901 and d5f85823,
# 2026-11-03). History, not derived from the current types: a key added to a
# type later must not be added here, or those saved models stop matching.
# ``mark_dimensions`` was saved by clusterless_gmm and clusterless_mark_glm only.
_PRE_ALGORITHM_ENCODING_KEYS: dict[str, frozenset[str]] = {
    "sorted_spikes_glm": frozenset(
        {
            "coefficients",
            "disable_progress_bar",
            "emission_design_info",
            "encoding_exposure_seconds",
            "environment",
            "environment_grid",
            "is_track_interior",
            "no_spike_part_log_likelihood",
            "place_fields",
            "rate_units",
        }
    ),
    "sorted_spikes_kde": frozenset(
        {
            "disable_progress_bar",
            "encoding_exposure_seconds",
            "environment",
            "environment_grid",
            "is_track_interior",
            "log_occupancy",
            "marginal_models",
            "mean_rates",
            "no_spike_part_log_likelihood",
            "occupancy",
            "occupancy_model",
            "place_fields",
            "rate_units",
        }
    ),
    "sorted_spikes_diffusion": frozenset(
        {
            "bin_sizes",
            "disable_progress_bar",
            "encoding_exposure_seconds",
            "environment",
            "environment_grid",
            "interior_log_place_fields",
            "is_track_interior",
            "local_interpolation",
            "mean_rates",
            "no_spike_part_log_likelihood",
            "node_order",
            "occupancy",
            "place_fields",
            "rate_units",
        }
    ),
    "sorted_spikes_mrf": frozenset(
        {
            "bin_sizes",
            "disable_progress_bar",
            "encoding_exposure_seconds",
            "environment",
            "environment_grid",
            "interior_log_place_fields",
            "is_track_interior",
            "local_interpolation",
            "mean_rates",
            "mrf_coefficients",
            "mrf_converged",
            "mrf_log_penalty_bounds",
            "mrf_max_step",
            "mrf_n_iter",
            "mrf_penalty",
            "mrf_penalty_selected_by_reml",
            "mrf_penalty_weights",
            "mrf_rank",
            "mrf_reml_objective",
            "no_spike_part_log_likelihood",
            "node_order",
            "occupancy",
            "place_fields",
            "rate_units",
        }
    ),
    "clusterless_kde": frozenset(
        {
            "block_size",
            "disable_progress_bar",
            "enc_tile_size",
            "encoding_exposure_seconds",
            "encoding_positions",
            "encoding_spike_waveform_features",
            "encoding_weights",
            "environment",
            "environment_grid",
            "gpi_models",
            "log_occupancy",
            "mean_rates",
            "occupancy",
            "occupancy_model",
            "pos_tile_size",
            "position_std",
            "rate_units",
            "summed_ground_process_intensity",
            "use_streaming",
            "waveform_std",
        }
    ),
    "clusterless_gmm": frozenset(
        {
            "bin_tile_size",
            "disable_progress_bar",
            "encoding_exposure_seconds",
            "environment",
            "environment_grid",
            "gmm_effective_components",
            "gmm_requested_components",
            "gpi_models",
            "interior_place_bin_centers",
            "joint_models",
            "log_occupancy",
            "mark_dimensions",
            "mean_rates",
            "occupancy_model",
            "rate_units",
            "spike_block_size",
            "summed_ground_process_intensity",
        }
    ),
    "clusterless_diffusion": frozenset(
        {
            "bin_sizes",
            "block_size",
            "disable_progress_bar",
            "encoding_bin_indices",
            "encoding_exposure_seconds",
            "encoding_marks",
            "encoding_weights",
            "environment",
            "environment_grid",
            "mean_rates",
            "memory_budget",
            "node_order",
            "occupancy",
            "position_std",
            "rate_units",
            "resolved_rank",
            "summed_ground_process_intensity",
            "waveform_std",
            "weight_total",
        }
    ),
    "clusterless_mark_glm": frozenset(
        {
            "disable_progress_bar",
            "emission_design_info",
            "encoding_exposure_seconds",
            "environment",
            "environment_grid",
            "log_rate_maps",
            "mark_dimensions",
            "mark_mixtures",
            "mean_rates",
            "rate_coefficients",
            "rate_units",
            "spike_block_size",
            "summed_ground_process_intensity",
        }
    ),
}

# Where the encoding models of the clusterless algorithms that did not record
# ``mark_dimensions`` before they recorded their algorithm keep each
# electrode's encoding waveform features, shape (n_spikes, n_features).
_LEGACY_ENCODING_MARKS: dict[str, str] = {
    "clusterless_kde": "encoding_spike_waveform_features",
    "clusterless_diffusion": "encoding_marks",
}


def _record_legacy_encoding_algorithm(encoding_model: object) -> None:
    """Record the algorithm of an encoding model saved before models recorded it.

    The algorithm is the one whose saved keys
    (``_PRE_ALGORITHM_ENCODING_KEYS``) are exactly the model's; no two
    algorithms saved the same keys. ``mark_dimensions``, which
    ``clusterless_kde`` and ``clusterless_diffusion`` did not save, is derived
    from the stored encoding waveform features. A model matching none, such
    as one fitted by a 0.6 release, is left unchanged, so decoding with it
    asks for a refit. The model is updated in place.

    Parameters
    ----------
    encoding_model : object
        One entry of a detector's ``encoding_model_``.
    """
    if not isinstance(encoding_model, dict) or "algorithm" in encoding_model:
        return
    keys = set(encoding_model)
    matches = [
        name for name, saved in _PRE_ALGORITHM_ENCODING_KEYS.items() if keys == saved
    ]
    if len(matches) != 1:
        return
    (algorithm,) = matches
    encoding_model["algorithm"] = algorithm
    if algorithm in _LEGACY_ENCODING_MARKS:
        encoding_model["mark_dimensions"] = [
            int(np.shape(features)[1])
            for features in encoding_model[_LEGACY_ENCODING_MARKS[algorithm]]
        ]


# Keys several algorithms store, whose fits returned lists of floats or JAX
# arrays before each became an ``np.ndarray`` (``EncodingModel`` and the
# algorithms' types declare it).
_ARRAY_ENCODING_KEYS: tuple[str, ...] = (
    "mean_rates",
    "occupancy",
    "summed_ground_process_intensity",
)

# Keys that fits stored and no prediction reads any more, which stored
# encoding models still carry; loading drops them.
_REMOVED_ENCODING_KEYS: frozenset[str] = frozenset(
    {
        # sorted_spikes_diffusion and sorted_spikes_mrf; prediction takes the
        # log of place_fields itself.
        "interior_log_place_fields",
    }
)


def _unify_stored_encoding_values(encoding_model: object) -> None:
    """Give a stored encoding model the keys and value types fits now return.

    Each value of ``_ARRAY_ENCODING_KEYS`` that is not an ``np.ndarray`` (a list
    of floats or a JAX array, as saved before) becomes ``np.asarray`` of it,
    with the dtype it had (float64 for Python floats), and each key of
    ``_REMOVED_ENCODING_KEYS`` is dropped. The algorithms read the remaining
    values as before, so decoding is unchanged. The model is updated in place.

    Parameters
    ----------
    encoding_model : object
        One entry of a detector's ``encoding_model_``; anything but a dict is
        left alone.
    """
    if not isinstance(encoding_model, dict):
        return
    for key in _ARRAY_ENCODING_KEYS:
        value = encoding_model.get(key)
        if value is not None and not isinstance(value, np.ndarray):
            encoding_model[key] = np.asarray(value)
    for key in _REMOVED_ENCODING_KEYS:
        encoding_model.pop(key, None)
