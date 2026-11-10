"""Classifiers of which environment the neural activity represents.

Two-state models, for sorted spikes and clusterless data, with one state per
environment.
"""

from typing import Any

import numpy as np

from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
from non_local_detector.models.base import (
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    FrozenTransitionRows,
    Observations,
    StateNames,
    Stickiness,
)

__all__ = [
    "MultiEnvironmentClusterlessClassifier",
    "MultiEnvironmentSortedSpikesClassifier",
]


class MultiEnvironmentSortedSpikesClassifier(SortedSpikesDetector):
    """Classifier of which of two environments sorted spike data represents.

    A two-state model, one state per environment. Within an environment the
    decoded position follows a random walk; switching environment places it
    uniformly in the other one.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (2,), optional
        Initial probabilities of [env1, env2], by default one transition after
        env1, the env1 row of the default transition matrix
        (``[0.999, 0.001]``). A state that starts at zero stays at zero through
        EM.
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial position distribution of each state, by default uniform for
        both.
    discrete_transition_type : DiscreteTransitions, optional
        Discrete transition model, by default ``DiscreteStationaryDiagonal``
        with self-transition probabilities ``[0.999, 0.98]``.
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[200.0, 30.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``RandomWalk`` within an environment and ``Uniform`` between
        them.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``ObservationModel(environment_name="env1")`` and
        ``ObservationModel(environment_name="env2")``.
    environments : Environments, optional
        The two spatial environments, by default
        ``Environment(environment_name="env1")`` and
        ``Environment(environment_name="env2")``.
    sorted_spikes_algorithm : str, optional
        Sorted-spikes likelihood algorithm, a key of
        ``_SORTED_SPIKES_ALGORITHMS``, by default "sorted_spikes_kde".
    sorted_spikes_algorithm_params : dict or None, optional
        Parameters for the sorted-spikes algorithm's fit. If ``None`` (the
        default), the algorithm's registered ``default_params`` are
        copied into a fresh dict at fit time; a user dict replaces those
        defaults. A key the fit does not accept raises ``ValidationError``.
        User-supplied dicts are stored as-is for sklearn ``clone()``
        compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer the track interior from the positions visited during
        fitting, by default True. Ignored for environments with a track graph.
    state_names : StateNames, optional
        Names of the states, by default ``["env1", "env2"]``.
    sampling_frequency : float, optional
        Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
        time range. The ``time_edges`` given to ``predict``,
        ``estimate_parameters``, ``most_likely_sequence`` and
        ``compute_log_likelihood`` must be ``1 / sampling_frequency`` wide, or
        they raise ``ValidationError``. Transition probabilities are per decode
        bin. By default 500.0.
    no_spike_rate : float, optional
        Firing rate in Hz of each neuron in a No-Spike state, by default
        1e-10.
    discrete_transition_prior_weight : float or np.ndarray, optional
        Dimensionless weight for data-adaptive Dirichlet prior scaling during
        EM re-estimation of discrete transitions. When > 0, the prior
        pseudo-counts are scaled by expected transition counts; a scalar
        applies to every row, an array of shape ``(n_states,)`` per row. By
        default 0.0 (the fixed-count prior from concentration and
        stickiness).
    frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
        Discrete transition rows held at their initial values during EM
        re-estimation. By default ``"no_spike"``: the rows of the states whose
        observation model has ``is_no_spike=True``, wherever they are listed, so
        a No-Spike self-transition cannot collapse to fit stray gaps between
        spikes. Models without a No-Spike state freeze nothing. Row indices or
        a boolean mask of shape ``(n_states,)`` select rows by position;
        ``None`` freezes nothing.

    Examples
    --------
    >>> from non_local_detector import (
    ...     MultiEnvironmentSortedSpikesClassifier,
    ...     time_edges_from_centers,
    ... )
    >>> classifier = MultiEnvironmentSortedSpikesClassifier()
    >>> # environment_labels: ("env1" or "env2") for each position sample
    >>> classifier.fit(
    ...     position_time, position, spike_times,
    ...     environment_labels=environment_labels,
    ... )
    >>> time_edges = time_edges_from_centers(position_time)
    >>> results = classifier.predict(
    ...     spike_times,
    ...     time_edges=time_edges,
    ... )
    >>> results.acausal_state_probabilities  # (n_time, n_states)
    """

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 0.1,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict[str, Any] | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
    ):
        params = _initialize_params(
            _ModelDefaults.multi_environment_defaults(),
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=continuous_initial_conditions_types,
            discrete_transition_type=discrete_transition_type,
            discrete_transition_stickiness=discrete_transition_stickiness,
            observation_models=observation_models,
            environments=environments,
            state_names=state_names,
            continuous_transition_types=continuous_transition_types,
        )

        super().__init__(
            discrete_initial_conditions=params["discrete_initial_conditions"],
            continuous_initial_conditions_types=params[
                "continuous_initial_conditions_types"
            ],
            discrete_transition_type=params["discrete_transition_type"],
            discrete_transition_concentration=discrete_transition_concentration,
            discrete_transition_stickiness=params["discrete_transition_stickiness"],
            discrete_transition_regularization=discrete_transition_regularization,
            continuous_transition_types=params["continuous_transition_types"],
            observation_models=params["observation_models"],
            environments=params["environments"],
            sorted_spikes_algorithm=sorted_spikes_algorithm,
            sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
            infer_track_interior=infer_track_interior,
            state_names=params["state_names"],
            sampling_frequency=sampling_frequency,
            no_spike_rate=no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
        )


class MultiEnvironmentClusterlessClassifier(ClusterlessDetector):
    """Classifier of which of two environments clusterless spike data represents.

    A two-state model, one state per environment. Within an environment the
    decoded position follows a random walk; switching environment places it
    uniformly in the other one.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (2,), optional
        Initial probabilities of [env1, env2], by default one transition after
        env1, the env1 row of the default transition matrix
        (``[0.999, 0.001]``). A state that starts at zero stays at zero through
        EM.
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial position distribution of each state, by default uniform for
        both.
    discrete_transition_type : DiscreteTransitions, optional
        Discrete transition model, by default ``DiscreteStationaryDiagonal``
        with self-transition probabilities ``[0.999, 0.98]``.
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[200.0, 30.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``RandomWalk`` within an environment and ``Uniform`` between
        them.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``ObservationModel(environment_name="env1")`` and
        ``ObservationModel(environment_name="env2")``.
    environments : Environments, optional
        The two spatial environments, by default
        ``Environment(environment_name="env1")`` and
        ``Environment(environment_name="env2")``.
    clusterless_algorithm : str, optional
        Clusterless likelihood algorithm, a key of ``_CLUSTERLESS_ALGORITHMS``,
        by default "clusterless_kde".
    clusterless_algorithm_params : dict or None, optional
        Parameters for the clusterless algorithm's fit. If ``None`` (the
        default), the algorithm's registered ``default_params`` are
        copied into a fresh dict at fit time; a user dict replaces those
        defaults. A key the fit does not accept raises ``ValidationError``.
        User-supplied dicts are stored as-is for sklearn ``clone()``
        compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer the track interior from the positions visited during
        fitting, by default True. Ignored for environments with a track graph.
    state_names : StateNames, optional
        Names of the states, by default ``["env1", "env2"]``.
    sampling_frequency : float, optional
        Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
        time range. The ``time_edges`` given to ``predict``,
        ``estimate_parameters``, ``most_likely_sequence`` and
        ``compute_log_likelihood`` must be ``1 / sampling_frequency`` wide, or
        they raise ``ValidationError``. Transition probabilities are per decode
        bin. By default 500.0.
    no_spike_rate : float, optional
        Firing rate in Hz of each electrode in a No-Spike state, by default
        1e-10. The marks of its spikes are scored with the training-fitted
        reference density ``mark_reference_``.
    discrete_transition_prior_weight : float or np.ndarray, optional
        Dimensionless weight for data-adaptive Dirichlet prior scaling during
        EM re-estimation of discrete transitions. When > 0, the prior
        pseudo-counts are scaled by expected transition counts; a scalar
        applies to every row, an array of shape ``(n_states,)`` per row. By
        default 0.0 (the fixed-count prior from concentration and
        stickiness).
    frozen_discrete_transition_rows : {"no_spike"}, array-like, or None, optional
        Discrete transition rows held at their initial values during EM
        re-estimation. By default ``"no_spike"``: the rows of the states whose
        observation model has ``is_no_spike=True``, wherever they are listed, so
        a No-Spike self-transition cannot collapse to fit stray gaps between
        spikes. Models without a No-Spike state freeze nothing. Row indices or
        a boolean mask of shape ``(n_states,)`` select rows by position;
        ``None`` freezes nothing.

    Examples
    --------
    >>> from non_local_detector import (
    ...     MultiEnvironmentClusterlessClassifier,
    ...     time_edges_from_centers,
    ... )
    >>> classifier = MultiEnvironmentClusterlessClassifier()
    >>> # environment_labels: ("env1" or "env2") for each position sample
    >>> classifier.fit(
    ...     position_time, position, spike_times, spike_waveform_features,
    ...     environment_labels=environment_labels,
    ... )
    >>> time_edges = time_edges_from_centers(position_time)
    >>> results = classifier.predict(
    ...     spike_times, spike_waveform_features,
    ...     time_edges=time_edges,
    ... )
    >>> results.acausal_state_probabilities  # (n_time, n_states)
    """

    def __init__(
        self,
        *,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 0.1,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict[str, Any] | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
    ):
        params = _initialize_params(
            _ModelDefaults.multi_environment_defaults(),
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=continuous_initial_conditions_types,
            discrete_transition_type=discrete_transition_type,
            discrete_transition_stickiness=discrete_transition_stickiness,
            observation_models=observation_models,
            environments=environments,
            state_names=state_names,
            continuous_transition_types=continuous_transition_types,
        )

        super().__init__(
            discrete_initial_conditions=params["discrete_initial_conditions"],
            continuous_initial_conditions_types=params[
                "continuous_initial_conditions_types"
            ],
            discrete_transition_type=params["discrete_transition_type"],
            discrete_transition_concentration=discrete_transition_concentration,
            discrete_transition_stickiness=params["discrete_transition_stickiness"],
            discrete_transition_regularization=discrete_transition_regularization,
            continuous_transition_types=params["continuous_transition_types"],
            observation_models=params["observation_models"],
            environments=params["environments"],
            clusterless_algorithm=clusterless_algorithm,
            clusterless_algorithm_params=clusterless_algorithm_params,
            infer_track_interior=infer_track_interior,
            state_names=params["state_names"],
            sampling_frequency=sampling_frequency,
            no_spike_rate=no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
        )
