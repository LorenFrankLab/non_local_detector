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

_DECODER_DOC = """Decoder of position from $data with a single continuous state.

The position follows a random walk and starts uniform over the track.

Parameters
----------
discrete_initial_conditions : np.ndarray, shape (1,), optional
    Initial probability of the state, by default ``[1.0]``.
continuous_initial_conditions_types : ContinuousInitialConditions, optional
    Initial position distribution, by default ``[UniformInitialConditions()]``.
discrete_transition_type : DiscreteTransitions, optional
    Discrete transition model, by default
    ``DiscreteStationaryDiagonal(diagonal_values=[1.0])``.
$discrete_transition_concentration
discrete_transition_stickiness : Stickiness, optional
    Added to the self-transition of the Dirichlet prior, by default ``[0.0]``.
$discrete_transition_regularization
continuous_transition_types : ContinuousTransitions, optional
    Position dynamics, by default ``[[RandomWalk()]]``.
observation_models : Observations, optional
    Observation model of the state, by default ``[ObservationModel()]``.
environments : Environments, optional
    Spatial environment, by default ``Environment(environment_name="")``.
$algorithm_parameters
$infer_track_interior
state_names : StateNames, optional
    Name of the state, by default ``["Continuous"]``.
$sampling_frequency
$no_spike_rate
$discrete_transition_prior_weight
$frozen_discrete_transition_rows

Examples
--------
>>> from non_local_detector import $cls, time_edges_from_centers
>>> decoder = $cls()
>>> decoder.fit($fit_example)
>>> time_edges = time_edges_from_centers(position_time)
>>> results = decoder.predict($predict_example, time_edges=time_edges)
>>> results.acausal_posterior  # (n_time, n_position_bins)
"""


@document_model(_DECODER_DOC, "sorted_spikes")
class SortedSpikesDecoder(SortedSpikesDetector):
    def __init__(
        self,
        discrete_initial_conditions: np.ndarray | None = None,
        continuous_initial_conditions_types: ContinuousInitialConditions | None = None,
        discrete_transition_type: DiscreteTransitions | None = None,
        discrete_transition_concentration: float = 1.0,
        discrete_transition_stickiness: Stickiness | None = None,
        discrete_transition_regularization: float = 0.1,
        continuous_transition_types: ContinuousTransitions | None = None,
        observation_models: Observations | None = None,
        environments: Environments | None = None,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
    ):
        params = _initialize_params(
            _ModelDefaults.decoder_defaults(),
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


@document_model(_DECODER_DOC, "clusterless")
class ClusterlessDecoder(ClusterlessDetector):
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
            _ModelDefaults.decoder_defaults(),
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
