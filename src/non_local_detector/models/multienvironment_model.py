import numpy as np

from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
from non_local_detector.models._docstrings import document_model
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

_MULTI_ENVIRONMENT_DOC = """Classifier of which of two environments $data represents.

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
$discrete_transition_concentration
discrete_transition_stickiness : Stickiness, shape (2,), optional
    Added to each state's self-transition in the Dirichlet prior, by default
    ``[200.0, 30.0]``.
$discrete_transition_regularization
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
$algorithm_parameters
$infer_track_interior
state_names : StateNames, optional
    Names of the states, by default ``["env1", "env2"]``.
$sampling_frequency
$no_spike_rate
$discrete_transition_prior_weight
$frozen_discrete_transition_rows

Examples
--------
>>> from non_local_detector import $cls, time_edges_from_centers
>>> classifier = $cls()
>>> # environment_labels: ("env1" or "env2") for each position sample
>>> classifier.fit(
...     $fit_example,
...     environment_labels=environment_labels,
... )
>>> time_edges = time_edges_from_centers(position_time)
>>> results = classifier.predict($predict_example, time_edges=time_edges)
>>> results.acausal_state_probabilities  # (n_time, n_states)
"""


@document_model(_MULTI_ENVIRONMENT_DOC, "sorted_spikes")
class MultiEnvironmentSortedSpikesClassifier(SortedSpikesDetector):
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
        sorted_spikes_algorithm_params: dict | None = None,
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


@document_model(_MULTI_ENVIRONMENT_DOC, "clusterless")
class MultiEnvironmentClusterlessClassifier(ClusterlessDetector):
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
        clusterless_algorithm_params: dict | None = None,
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
