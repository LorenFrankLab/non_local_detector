import numpy as np

from non_local_detector.continuous_state_transitions import (
    Discrete,
    RandomWalk,
    Uniform,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteStationaryCustom,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models.base import (
    _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
    _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    Observations,
    StateNames,
    Stickiness,
)

environment = Environment(environment_name="")

observation_models = [
    ObservationModel(is_no_spike=True),
    ObservationModel(),
    ObservationModel(),
]

discrete_initial_conditions = np.ones((3,)) / 3
continuous_initial_conditions = [
    UniformInitialConditions(),
    UniformInitialConditions(),
    UniformInitialConditions(),
]

discrete_transition_stickiness = np.array([100_000.0, 30.0, 200.0])

# transition probability to no spike state
no_spike_trans_prob = 1e-5
# probability of staying in continuous non-local state
cont_non_local_prob = 0.9
# probability of staying in non-local fragmented state
non_local_frag_prob = 0.98
# probability of staying in no-spike state
no_spike_prob = 0.99

discrete_transition_matrix_values = np.array(
    [
        [
            no_spike_prob,
            (1 - no_spike_prob) / 2,
            (1 - no_spike_prob) / 2,
        ],
        [
            no_spike_trans_prob,
            cont_non_local_prob,
            (1 - cont_non_local_prob - no_spike_trans_prob),
        ],
        [
            no_spike_trans_prob,
            (1 - non_local_frag_prob - no_spike_trans_prob),
            non_local_frag_prob,
        ],
    ]
)

discrete_transition_type = DiscreteStationaryCustom(
    values=discrete_transition_matrix_values
)

non_stationary_discrete_transition_type = DiscreteNonStationaryCustom(
    values=discrete_transition_matrix_values
)

no_spike_rate = 1e-10

state_names = [
    "No-Spike",
    "Continuous",
    "Fragmented",
]


class NoSpikeContFragSortedSpikesClassifier(SortedSpikesDetector):
    """Classifier for no-spike, continuous, and fragmented states using sorted spike data.

    This classifier uses a three-state Hidden Markov Model to distinguish between
    no-spike periods, continuous non-local activity, and fragmented non-local
    activity using sorted spike data. This model explicitly handles periods of
    neural silence alongside different types of replay events.

    The three states are:
    - No-Spike: Periods with minimal neural activity (uses Discrete transitions)
    - Continuous: Smooth spatial replay trajectories (uses RandomWalk)
    - Fragmented: Disjointed spatial representations (uses Uniform)

    The model uses custom transition matrices with high self-transition
    probabilities to maintain state persistence, especially for no-spike periods.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (3,), optional
        Initial probabilities for [No-Spike, Continuous, Fragmented] states,
        by default np.ones((3,)) / 3 (uniform distribution).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial condition types for continuous states, by default uniform
        initial conditions for all three states.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transitions, by default custom stationary
        transition matrix with specified probabilities.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, shape (3,), optional
        Stickiness parameters for state persistence, by default
        [100000.0, 30.0, 200.0] with very high no-spike persistence.
    discrete_transition_regularization : float, optional
        Regularization parameter for transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions or None, optional
        Transition types for continuous states. If None, uses default:
        [[Discrete(), Uniform(), Uniform()], [Discrete(), RandomWalk(), Uniform()],
        [Discrete(), Uniform(), Uniform()]].
    observation_models : Observations, optional
        Observation models for likelihood computation, by default includes
        a no-spike observation model.
    environments : Environments, optional
        Environment specifications, by default empty Environment.
    sorted_spikes_algorithm : str, optional
        Algorithm for sorted spikes likelihood, by default 'sorted_spikes_kde'.
    sorted_spikes_algorithm_params : dict, optional
        Parameters for the sorted spikes algorithm, by default uses package defaults.
    infer_track_interior : bool, optional
        Whether to infer track interior during decoding, by default True.
    state_names : StateNames, optional
        Names for the states, by default ['No-Spike', 'Continuous', 'Fragmented'].
    sampling_frequency : float, optional
        Data sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate parameter for no-spike periods, by default 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from non_local_detector.models.nospike_cont_frag_model import NoSpikeContFragSortedSpikesClassifier
    >>> # Create classifier with default parameters
    >>> classifier = NoSpikeContFragSortedSpikesClassifier()
    >>> # Fit to training data
    >>> classifier.fit(position, spikes)
    >>> # Classify test data
    >>> results = classifier.predict(test_spikes)
    >>> # Extract state probabilities
    >>> state_probs = results.acausal_posterior.sum('position')
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ):
        if discrete_initial_conditions is None:
            discrete_initial_conditions = np.ones((3,)) / 3
        if continuous_initial_conditions_types is None:
            continuous_initial_conditions_types = [
                UniformInitialConditions(),
                UniformInitialConditions(),
                UniformInitialConditions(),
            ]
        if discrete_transition_type is None:
            discrete_transition_type = DiscreteStationaryCustom(
                values=np.array(
                    [
                        [
                            0.99,
                            (1 - 0.99) / 2,
                            (1 - 0.99) / 2,
                        ],
                        [
                            1e-5,
                            0.9,
                            (1 - 0.9 - 1e-5),
                        ],
                        [
                            1e-5,
                            (1 - 0.98 - 1e-5),
                            0.98,
                        ],
                    ]
                )
            )
        if discrete_transition_stickiness is None:
            discrete_transition_stickiness = np.array([100_000.0, 30.0, 200.0])
        if observation_models is None:
            observation_models = [
                ObservationModel(is_no_spike=True),
                ObservationModel(),
                ObservationModel(),
            ]
        if environments is None:
            environments = Environment(environment_name="")
        if state_names is None:
            state_names = [
                "No-Spike",
                "Continuous",
                "Fragmented",
            ]

        if continuous_transition_types is None:
            continuous_transition_types = [
                [Discrete(), Uniform(), Uniform()],
                [Discrete(), RandomWalk(), Uniform()],
                [Discrete(), Uniform(), Uniform()],
            ]
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            sorted_spikes_algorithm,
            sorted_spikes_algorithm_params,
            infer_track_interior,
            state_names,
            sampling_frequency,
            no_spike_rate,
        )


class NoSpikeContFragClusterlessClassifier(ClusterlessDetector):
    """Classifier for no-spike, continuous, and fragmented states using clusterless data.

    This classifier uses a three-state Hidden Markov Model to distinguish between
    no-spike periods, continuous non-local activity, and fragmented non-local
    activity using clusterless (continuous) spike data. This model explicitly
    handles periods of neural silence alongside different types of replay events.

    The three states are:
    - No-Spike: Periods with minimal neural activity
    - Continuous: Smooth spatial replay trajectories (uses RandomWalk)
    - Fragmented: Disjointed spatial representations (uses Uniform)

    Note: This implementation uses a simplified two-state continuous transition
    model despite having three discrete states, which may be appropriate for
    specific clusterless data analysis scenarios.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (3,), optional
        Initial probabilities for [No-Spike, Continuous, Fragmented] states,
        by default np.ones((3,)) / 3 (uniform distribution).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial condition types for continuous states, by default uniform
        initial conditions for all three states.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transitions, by default custom stationary
        transition matrix with specified probabilities.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, shape (3,), optional
        Stickiness parameters for state persistence, by default
        [100000.0, 30.0, 200.0] with very high no-spike persistence.
    discrete_transition_regularization : float, optional
        Regularization parameter for transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions or None, optional
        Transition types for continuous states. If None, uses simplified
        two-state model: [[RandomWalk(), Uniform()], [Uniform(), Uniform()]].
    observation_models : Observations, optional
        Observation models for likelihood computation, by default includes
        a no-spike observation model.
    environments : Environments, optional
        Environment specifications, by default empty Environment.
    clusterless_algorithm : str, optional
        Algorithm for clusterless data likelihood, by default 'clusterless_kde'.
    clusterless_algorithm_params : dict, optional
        Parameters for the clusterless algorithm, by default uses package defaults.
    infer_track_interior : bool, optional
        Whether to infer track interior during decoding, by default True.
    state_names : StateNames, optional
        Names for the states, by default ['No-Spike', 'Continuous', 'Fragmented'].
    sampling_frequency : float, optional
        Data sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate parameter for no-spike periods, by default 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from non_local_detector.models.nospike_cont_frag_model import NoSpikeContFragClusterlessClassifier
    >>> # Create classifier with default parameters
    >>> classifier = NoSpikeContFragClusterlessClassifier()
    >>> # Fit to training data
    >>> classifier.fit(position, clusterless_spikes)
    >>> # Classify test data
    >>> results = classifier.predict(test_clusterless_spikes)
    >>> # Extract state probabilities
    >>> state_probs = results.acausal_posterior.sum('position')
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ):
        if discrete_initial_conditions is None:
            discrete_initial_conditions = np.ones((3,)) / 3
        if continuous_initial_conditions_types is None:
            continuous_initial_conditions_types = [
                UniformInitialConditions(),
                UniformInitialConditions(),
                UniformInitialConditions(),
            ]
        if discrete_transition_type is None:
            discrete_transition_type = DiscreteStationaryCustom(
                values=np.array(
                    [
                        [
                            0.99,
                            (1 - 0.99) / 2,
                            (1 - 0.99) / 2,
                        ],
                        [
                            1e-5,
                            0.9,
                            (1 - 0.9 - 1e-5),
                        ],
                        [
                            1e-5,
                            (1 - 0.98 - 1e-5),
                            0.98,
                        ],
                    ]
                )
            )
        if discrete_transition_stickiness is None:
            discrete_transition_stickiness = np.array([100_000.0, 30.0, 200.0])
        if observation_models is None:
            observation_models = [
                ObservationModel(is_no_spike=True),
                ObservationModel(),
                ObservationModel(),
            ]
        if environments is None:
            environments = Environment(environment_name="")
        if state_names is None:
            state_names = [
                "No-Spike",
                "Continuous",
                "Fragmented",
            ]

        if continuous_transition_types is None:
            continuous_transition_types = [
                [Discrete(), Uniform(), Uniform()],
                [Discrete(), RandomWalk(), Uniform()],
                [Discrete(), Uniform(), Uniform()],
            ]
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            clusterless_algorithm,
            clusterless_algorithm_params,
            infer_track_interior,
            state_names,
            sampling_frequency,
            no_spike_rate,
        )
