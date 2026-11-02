"""Docstrings shared by the sorted-spikes and clusterless model classes.

Each model family (decoder, continuous/fragmented, ...) has a sorted-spikes
and a clusterless class with the same states and defaults. Their docstring is
written once, as a template, and filled for each spike-data type by
:func:`document_model`. Parameters documented the same way in every family
are written once here.

Templates use :class:`string.Template` placeholders:

- ``$<parameter name>`` for a shared parameter's entry (``_SHARED_PARAMETERS``);
- ``$data``, ``$algorithm_parameters``, ``$no_spike_rate``, ``$fit_example``
  and ``$predict_example`` for the spike-data type (``_DATA_TYPES``);
- ``$cls`` for the class name and ``$default_<parameter name>`` for a
  parameter's default, read from the class signature.
"""

import inspect
from collections.abc import Callable
from string import Template
from typing import TypeVar

_ModelClass = TypeVar("_ModelClass", bound=type)

_SHARED_PARAMETERS = {
    "discrete_transition_concentration": """\
discrete_transition_concentration : float, optional
    Dirichlet prior parameter of every entry of each discrete transition row
    during EM re-estimation; ``discrete_transition_stickiness`` is added to
    the self-transition. By default $default_discrete_transition_concentration.""",
    "discrete_transition_regularization": """\
discrete_transition_regularization : float, optional
    L2 penalty on the non-intercept coefficients of covariate-dependent
    discrete transitions, equivalent to a Gaussian prior with standard
    deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
    Stationary transitions do not use it.""",
    "infer_track_interior": """\
infer_track_interior : bool, optional
    Whether to infer the track interior from the positions visited during
    fitting, by default True. Ignored for environments with a track graph.""",
    "sampling_frequency": """\
sampling_frequency : float, optional
    Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
    time range; ``predict`` and ``estimate_parameters`` use the
    ``time_edges`` they are given. Transition probabilities are per decode
    bin. By default $default_sampling_frequency.""",
    "discrete_transition_prior_weight": """\
discrete_transition_prior_weight : float or np.ndarray, optional
    Dimensionless weight for data-adaptive Dirichlet prior scaling during
    EM re-estimation of discrete transitions. When > 0, the prior
    pseudo-counts are scaled by expected transition counts; a scalar
    applies to every row, an array of shape ``(n_states,)`` per row. By
    default 0.0 (the fixed-count prior from concentration and
    stickiness).""",
    "frozen_discrete_transition_rows": """\
frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
    Discrete transition rows held at their initial values during EM
    re-estimation. By default ``"no_spike"``: the rows of the states whose
    observation model has ``is_no_spike=True``, wherever they are listed, so
    a No-Spike self-transition cannot collapse to fit stray gaps between
    spikes. Models without a No-Spike state freeze nothing. Row indices or
    a boolean mask of shape ``(n_states,)`` select rows by position;
    ``None`` freezes nothing.""",
}

_DATA_TYPES = {
    "sorted_spikes": {
        "data": "sorted spike data",
        "algorithm_parameters": """\
sorted_spikes_algorithm : str, optional
    Sorted-spikes likelihood algorithm, a key of
    ``_SORTED_SPIKES_ALGORITHMS``, by default "sorted_spikes_kde".
sorted_spikes_algorithm_params : dict or None, optional
    Parameters for the sorted-spikes algorithm. If ``None`` (the default),
    the module-level ``_DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS`` values are
    copied into a fresh dict at fit time. User-supplied dicts are stored
    as-is for sklearn ``clone()`` compatibility and copied on use.""",
        "no_spike_rate": """\
no_spike_rate : float, optional
    Firing rate in Hz of each neuron in a No-Spike state, by default
    1e-10.""",
        "fit_example": "position_time, position, spike_times",
        "predict_example": "spike_times",
    },
    "clusterless": {
        "data": "clusterless spike data",
        "algorithm_parameters": """\
clusterless_algorithm : str, optional
    Clusterless likelihood algorithm, a key of ``_CLUSTERLESS_ALGORITHMS``,
    by default "clusterless_kde".
clusterless_algorithm_params : dict or None, optional
    Parameters for the clusterless algorithm. If ``None`` (the default),
    the module-level ``_DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS`` values are
    copied into a fresh dict at fit time. User-supplied dicts are stored
    as-is for sklearn ``clone()`` compatibility and copied on use.""",
        "no_spike_rate": """\
no_spike_rate : float, optional
    Firing rate in Hz of each electrode in a No-Spike state, by default
    1e-10. The marks of its spikes are scored with the training-fitted
    reference density ``mark_reference_``.""",
        "fit_example": (
            "position_time, position, spike_times, spike_waveform_features"
        ),
        "predict_example": "spike_times, spike_waveform_features",
    },
}


def document_model(
    template: str, data_type: str
) -> Callable[[_ModelClass], _ModelClass]:
    """Class decorator that fills a model-family docstring template.

    Parameters
    ----------
    template : str
        Docstring template; see the module docstring for its placeholders.
    data_type : {"sorted_spikes", "clusterless"}
        Spike-data type of the decorated class.

    Returns
    -------
    decorate : callable
        Sets the class's ``__doc__`` and returns the class unchanged, with
        its type preserved for type checkers.
    """

    def decorate(cls: _ModelClass) -> _ModelClass:
        defaults = {
            f"default_{name}": repr(parameter.default)
            for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        sections = Template(template).safe_substitute(
            _SHARED_PARAMETERS, **_DATA_TYPES[data_type]
        )
        cls.__doc__ = Template(sections).substitute(defaults, cls=cls.__name__)
        return cls

    return decorate
