import numpy as np

from non_local_detector.continuous_state_transitions import RandomWalk, Uniform
from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal
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

environments = [
    Environment(environment_name="env1"),
    Environment(environment_name="env2"),
]

observation_models = [
    ObservationModel(environment_name="env1"),
    ObservationModel(environment_name="env2"),
]

discrete_initial_conditions = np.array([1.0, 0.0])
continuous_initial_conditions = [
    UniformInitialConditions(),
    UniformInitialConditions(),
]

discrete_transition_stickiness = np.array([200.0, 30.0])

discrete_transition_type = DiscreteStationaryDiagonal(
    diagonal_values=np.array([0.999, 0.98])
)

state_names = [
    "env1",
    "env2",
]


class MultiEnvironmentSortedSpikesClassifier(SortedSpikesDetector):
    """Classifier for multi-environment neural activity using sorted spike data.

    This classifier uses a two-state Hidden Markov Model to distinguish between
    neural activity corresponding to two different spatial environments using
    sorted spike data. Each state represents neural activity associated with
    a specific environment, allowing classification of which environment the
    animal is mentally representing.

    The model transitions:
    - Environment 1: Uses RandomWalk within env1, Uniform transition to env2
    - Environment 2: Uses Uniform transition from env1, RandomWalk within env2

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (2,), optional
        Initial probabilities for [env1, env2] states, by default
        np.array([1.0, 0.0]) (starts in environment 1).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial condition types for continuous states, by default uniform
        initial conditions for both environments.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transitions, by default stationary diagonal
        with high self-transition probabilities [99.9%, 98%].
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Stickiness parameters for state persistence, by default [200.0, 30.0]
        favoring environment 1 persistence.
    discrete_transition_regularization : float, optional
        Regularization parameter for transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions or None, optional
        Transition types for continuous states. If None, uses environment-specific
        transitions with RandomWalk within environments and Uniform between.
    observation_models : Observations, optional
        Observation models for each environment, by default environment-specific
        ObservationModel instances.
    environments : Environments, optional
        Environment specifications, by default [env1, env2] Environment instances.
    sorted_spikes_algorithm : str, optional
        Algorithm for sorted spikes likelihood, by default 'sorted_spikes_kde'.
    sorted_spikes_algorithm_params : dict, optional
        Parameters for the sorted spikes algorithm, by default uses package defaults.
    infer_track_interior : bool, optional
        Whether to infer track interior during decoding, by default True.
    state_names : StateNames, optional
        Names for the environments, by default ['env1', 'env2'].
    sampling_frequency : float, optional
        Data sampling frequency in Hz, by default 500.
    no_spike_rate : float, optional
        Rate parameter for no-spike periods, by default 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from non_local_detector.models.multienvironment_model import MultiEnvironmentSortedSpikesClassifier
    >>> # Create classifier with default parameters
    >>> classifier = MultiEnvironmentSortedSpikesClassifier()
    >>> # Fit to training data from both environments
    >>> classifier.fit(position, spikes)
    >>> # Classify test data
    >>> results = classifier.predict(test_spikes)
    >>> # Extract environment probabilities
    >>> env_probs = results.acausal_posterior.sum('position')
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = discrete_initial_conditions,
        continuous_initial_conditions_types: ContinuousInitialConditions = continuous_initial_conditions,
        discrete_transition_type: DiscreteTransitions = discrete_transition_type,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = discrete_transition_stickiness,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations = observation_models,
        environments: Environments = environments,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = state_names,
        sampling_frequency: float = 500,
        no_spike_rate: float = 1e-10,
    ):
        if continuous_transition_types is None:
            continuous_transition_types = [
                [
                    RandomWalk(environment_name="env1"),
                    Uniform(environment_name="env1", environment2_name="env2"),
                ],
                [
                    Uniform(environment_name="env2", environment2_name="env1"),
                    RandomWalk(environment_name="env2"),
                ],
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


class MultiEnvironmentClusterlessClassifier(ClusterlessDetector):
    """Classifier for multi-environment neural activity using clusterless data.

    This classifier uses a two-state Hidden Markov Model to distinguish between
    neural activity corresponding to two different spatial environments using
    clusterless (continuous) spike data. Each state represents neural activity
    associated with a specific environment, allowing classification of which
    environment the animal is mentally representing.

    The model uses simplified transitions with RandomWalk dynamics, suitable
    for clusterless data analysis in multi-environment contexts.

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (2,), optional
        Initial probabilities for [env1, env2] states, by default
        np.array([1.0, 0.0]) (starts in environment 1).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial condition types for continuous states, by default uniform
        initial conditions for both environments.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transitions, by default stationary diagonal
        with high self-transition probabilities [99.9%, 98%].
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Stickiness parameters for state persistence, by default [200.0, 30.0]
        favoring environment 1 persistence.
    discrete_transition_regularization : float, optional
        Regularization parameter for transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions or None, optional
        Transition types for continuous states. If None, uses RandomWalk
        dynamics for clusterless data.
    observation_models : Observations, optional
        Observation models for each environment, by default environment-specific
        ObservationModel instances.
    environments : Environments, optional
        Environment specifications, by default [env1, env2] Environment instances.
    clusterless_algorithm : str, optional
        Algorithm for clusterless data likelihood, by default 'clusterless_kde'.
    clusterless_algorithm_params : dict, optional
        Parameters for the clusterless algorithm, by default uses package defaults.
    infer_track_interior : bool, optional
        Whether to infer track interior during decoding, by default True.
    state_names : StateNames, optional
        Names for the environments, by default ['env1', 'env2'].
    sampling_frequency : float, optional
        Data sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate parameter for no-spike periods, by default 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from non_local_detector.models.multienvironment_model import MultiEnvironmentClusterlessClassifier
    >>> # Create classifier with default parameters
    >>> classifier = MultiEnvironmentClusterlessClassifier()
    >>> # Fit to training data from both environments
    >>> classifier.fit(position, clusterless_spikes)
    >>> # Classify test data
    >>> results = classifier.predict(test_clusterless_spikes)
    >>> # Extract environment probabilities
    >>> env_probs = results.acausal_posterior.sum('position')
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray = discrete_initial_conditions,
        continuous_initial_conditions_types: ContinuousInitialConditions = continuous_initial_conditions,
        discrete_transition_type: DiscreteTransitions = discrete_transition_type,
        discrete_transition_concentration: float = 1.1,
        discrete_transition_stickiness: Stickiness = discrete_transition_stickiness,
        discrete_transition_regularization: float = 1e-10,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations = observation_models,
        environments: Environments = environments,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = state_names,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
    ):
        if continuous_transition_types is None:
            continuous_transition_types = [
                [
                    RandomWalk(environment_name="env1"),
                    Uniform(environment_name="env1", environment2_name="env2"),
                ],
                [
                    Uniform(environment_name="env2", environment2_name="env1"),
                    RandomWalk(environment_name="env2"),
                ],
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
