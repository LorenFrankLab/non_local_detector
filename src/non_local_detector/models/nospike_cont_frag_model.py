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


class NoSpikeContFragSortedSpikesClassifier(SortedSpikesDetector):
    """No-Spike, continuous and fragmented classifier for sorted spike data.

    A three-state model. The No-Spike state is a near-silent firing model
    without position; in the Continuous state the decoded position follows a
    random walk, and in the Fragmented state it jumps uniformly over the track.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (3,), optional
        Initial probabilities of [No-Spike, Continuous, Fragmented], by default
        ``[1/3, 1/3, 1/3]``.
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial position distribution of each state, by default uniform for
        all three.
    discrete_transition_type : DiscreteTransitions, optional
        Discrete transition model, by default ``DiscreteStationaryCustom`` with
        self-transition probabilities 0.99 (No-Spike), 0.9 (Continuous) and
        0.98 (Fragmented); Continuous and Fragmented enter No-Spike with
        probability 1e-5.
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.1.
    discrete_transition_stickiness : Stickiness, shape (3,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[100000.0, 30.0, 200.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``[[Discrete(), Uniform(), Uniform()], [Discrete(), RandomWalk(),
        Uniform()], [Discrete(), Uniform(), Uniform()]]``.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``ObservationModel(is_no_spike=True)`` for No-Spike and
        ``ObservationModel()`` for the others.
    environments : Environments, optional
        Spatial environment, by default ``Environment(environment_name="")``.
    sorted_spikes_algorithm : str, optional
        Sorted-spikes likelihood algorithm, a key of
        ``_SORTED_SPIKES_ALGORITHMS``, by default "sorted_spikes_kde".
    sorted_spikes_algorithm_params : dict or None, optional
        Parameters for the sorted-spikes algorithm's fit. If ``None`` (the
        default), the algorithm's entry of ``_SORTED_SPIKES_DEFAULT_PARAMS`` is
        copied into a fresh dict at fit time; a user dict replaces those
        defaults. A key the fit does not accept raises ``ValidationError``.
        User-supplied dicts are stored as-is for sklearn ``clone()``
        compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer the track interior from the positions visited during
        fitting, by default True. Ignored for environments with a track graph.
    state_names : StateNames, optional
        Names of the states, by default
        ``["No-Spike", "Continuous", "Fragmented"]``.
    sampling_frequency : float, optional
        Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
        time range; ``predict`` and ``estimate_parameters`` use the
        ``time_edges`` they are given. Transition probabilities are per decode
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
    ...     NoSpikeContFragSortedSpikesClassifier,
    ...     time_edges_from_centers,
    ... )
    >>> classifier = NoSpikeContFragSortedSpikesClassifier()
    >>> classifier.fit(position_time, position, spike_times)
    >>> time_edges = time_edges_from_centers(position_time)
    >>> results = classifier.predict(
    ...     spike_times,
    ...     time_edges=time_edges,
    ... )
    >>> results.acausal_state_probabilities  # (n_time, n_states)
    """

    def __init__(
        self,
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
            _ModelDefaults.nospike_cont_frag_defaults(),
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
            params["discrete_initial_conditions"],
            params["continuous_initial_conditions_types"],
            params["discrete_transition_type"],
            discrete_transition_concentration,
            params["discrete_transition_stickiness"],
            discrete_transition_regularization,
            params["continuous_transition_types"],
            params["observation_models"],
            params["environments"],
            sorted_spikes_algorithm,
            sorted_spikes_algorithm_params,
            infer_track_interior,
            params["state_names"],
            sampling_frequency,
            no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
        )


class NoSpikeContFragClusterlessClassifier(ClusterlessDetector):
    """No-Spike, continuous and fragmented classifier for clusterless spike data.

    A three-state model. The No-Spike state is a near-silent firing model
    without position; in the Continuous state the decoded position follows a
    random walk, and in the Fragmented state it jumps uniformly over the track.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (3,), optional
        Initial probabilities of [No-Spike, Continuous, Fragmented], by default
        ``[1/3, 1/3, 1/3]``.
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial position distribution of each state, by default uniform for
        all three.
    discrete_transition_type : DiscreteTransitions, optional
        Discrete transition model, by default ``DiscreteStationaryCustom`` with
        self-transition probabilities 0.99 (No-Spike), 0.9 (Continuous) and
        0.98 (Fragmented); Continuous and Fragmented enter No-Spike with
        probability 1e-5.
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.1.
    discrete_transition_stickiness : Stickiness, shape (3,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[100000.0, 30.0, 200.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``[[Discrete(), Uniform(), Uniform()], [Discrete(), RandomWalk(),
        Uniform()], [Discrete(), Uniform(), Uniform()]]``.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``ObservationModel(is_no_spike=True)`` for No-Spike and
        ``ObservationModel()`` for the others.
    environments : Environments, optional
        Spatial environment, by default ``Environment(environment_name="")``.
    clusterless_algorithm : str, optional
        Clusterless likelihood algorithm, a key of ``_CLUSTERLESS_ALGORITHMS``,
        by default "clusterless_kde".
    clusterless_algorithm_params : dict or None, optional
        Parameters for the clusterless algorithm's fit. If ``None`` (the
        default), the algorithm's entry of ``_CLUSTERLESS_DEFAULT_PARAMS`` is
        copied into a fresh dict at fit time; a user dict replaces those
        defaults. A key the fit does not accept raises ``ValidationError``.
        User-supplied dicts are stored as-is for sklearn ``clone()``
        compatibility and copied on use.
    infer_track_interior : bool, optional
        Whether to infer the track interior from the positions visited during
        fitting, by default True. Ignored for environments with a track graph.
    state_names : StateNames, optional
        Names of the states, by default
        ``["No-Spike", "Continuous", "Fragmented"]``.
    sampling_frequency : float, optional
        Rate in Hz of the decode bins that ``calculate_time_edges`` builds from a
        time range; ``predict`` and ``estimate_parameters`` use the
        ``time_edges`` they are given. Transition probabilities are per decode
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
    ...     NoSpikeContFragClusterlessClassifier,
    ...     time_edges_from_centers,
    ... )
    >>> classifier = NoSpikeContFragClusterlessClassifier()
    >>> classifier.fit(position_time, position, spike_times, spike_waveform_features)
    >>> time_edges = time_edges_from_centers(position_time)
    >>> results = classifier.predict(
    ...     spike_times, spike_waveform_features,
    ...     time_edges=time_edges,
    ... )
    >>> results.acausal_state_probabilities  # (n_time, n_states)
    """

    def __init__(
        self,
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
            _ModelDefaults.nospike_cont_frag_defaults(),
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
            params["discrete_initial_conditions"],
            params["continuous_initial_conditions_types"],
            params["discrete_transition_type"],
            discrete_transition_concentration,
            params["discrete_transition_stickiness"],
            discrete_transition_regularization,
            params["continuous_transition_types"],
            params["observation_models"],
            params["environments"],
            clusterless_algorithm,
            clusterless_algorithm_params,
            infer_track_interior,
            params["state_names"],
            sampling_frequency,
            no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
        )
