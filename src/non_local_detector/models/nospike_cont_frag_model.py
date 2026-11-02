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

_NO_SPIKE_CONT_FRAG_DOC = """Classifier of No-Spike, continuous and fragmented periods from $data.

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
$discrete_transition_concentration
discrete_transition_stickiness : Stickiness, shape (3,), optional
    Added to each state's self-transition in the Dirichlet prior, by default
    ``[100000.0, 30.0, 200.0]``.
$discrete_transition_regularization
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
$algorithm_parameters
$infer_track_interior
state_names : StateNames, optional
    Names of the states, by default
    ``["No-Spike", "Continuous", "Fragmented"]``.
$sampling_frequency
$no_spike_rate
$discrete_transition_prior_weight
$frozen_discrete_transition_rows

Examples
--------
>>> from non_local_detector import $cls, time_edges_from_centers
>>> classifier = $cls()
>>> classifier.fit($fit_example)
>>> time_edges = time_edges_from_centers(position_time)
>>> results = classifier.predict($predict_example, time_edges=time_edges)
>>> results.acausal_state_probabilities  # (n_time, n_states)
"""


@document_model(_NO_SPIKE_CONT_FRAG_DOC, "sorted_spikes")
class NoSpikeContFragSortedSpikesClassifier(SortedSpikesDetector):
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


@document_model(_NO_SPIKE_CONT_FRAG_DOC, "clusterless")
class NoSpikeContFragClusterlessClassifier(ClusterlessDetector):
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
