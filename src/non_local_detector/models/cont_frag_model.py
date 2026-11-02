import numpy as np
import xarray as xr

from non_local_detector._position_dims import get_position_dims
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


def _state_probabilities_from_results(results: xr.Dataset) -> xr.DataArray:
    """Marginalize the acausal posterior over position to per-state probabilities.

    Unstacks the ``state_bins`` MultiIndex and sums over every position
    dimension, leaving ``(time, state)``. Position dims are those named
    ``position`` (1D) or ending in ``_position`` (``x_position``/``y_position``
    for 2D, ``z_position`` and beyond, or ``dim{i}_position`` past six
    dimensions), matching the names ``_convert_results_to_xarray`` assigns.

    Raises
    ------
    ValueError
        If no position dimension is present (``unstacked.sum([])`` would
        otherwise silently return the full per-bin array).
    """
    unstacked = results.acausal_posterior.unstack("state_bins")
    position_dims = get_position_dims(unstacked)
    if not position_dims:
        raise ValueError(
            "get_posterior found no position dimension to marginalize over; "
            f"acausal_posterior unstacked to dims {tuple(unstacked.dims)}. "
            "Expected a dim named 'position' or ending in '_position'."
        )
    return unstacked.sum(position_dims)


_CONT_FRAG_DOC = """Classifier of continuous vs. fragmented replay from $data.

A two-state model. In the Continuous state the decoded position follows a
random walk; in the Fragmented state it jumps uniformly over the track, and
switching state places it uniformly too.

Parameters
----------
discrete_initial_conditions : np.ndarray, shape (2,), optional
    Initial probabilities of [Continuous, Fragmented], by default
    ``[0.5, 0.5]``.
continuous_initial_conditions_types : ContinuousInitialConditions, optional
    Initial position distribution of each state, by default uniform for
    both.
discrete_transition_type : DiscreteTransitions, optional
    Discrete transition model, by default ``DiscreteStationaryDiagonal``
    with self-transition probability 0.98 for both states.
$discrete_transition_concentration
discrete_transition_stickiness : Stickiness, shape (2,), optional
    Added to each state's self-transition in the Dirichlet prior, by default
    ``[0.0, 0.0]``.
$discrete_transition_regularization
continuous_transition_types : ContinuousTransitions, optional
    Position dynamics from each state (row) to each state (column), by
    default ``[[RandomWalk(), Uniform()], [Uniform(), Uniform()]]``.
observation_models : Observations, optional
    Observation model of each state, by default
    ``[ObservationModel(), ObservationModel()]``.
environments : Environments, optional
    Spatial environment, by default ``Environment(environment_name="")``.
$algorithm_parameters
$infer_track_interior
state_names : StateNames, optional
    Names of the states, by default ``["Continuous", "Fragmented"]``.
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
>>> state_probs = classifier.get_posterior(results)  # (n_time, n_states)
"""


class _ContFragResultsMixin:
    """Results helpers shared by the continuous/fragmented classifiers."""

    @staticmethod
    def get_posterior(results: xr.Dataset) -> xr.DataArray:
        """Extract posterior probabilities for continuous vs fragmented states.

        This method extracts and sums the posterior probabilities across all
        position bins to get the overall probability of each state (continuous
        vs fragmented) at each time point. Every position dimension is
        collapsed: the dim named ``position`` (1D) plus any dim whose name ends
        in ``_position`` (``x_position``/``y_position`` for 2D, ``z_position``
        and beyond for 3-6D, or ``dim{i}_position`` past six dimensions).

        Parameters
        ----------
        results : xr.Dataset
            Results dataset from model prediction containing acausal_posterior
            with dimensions (time, state_bins) where state_bins combines
            discrete states and position bins.

        Returns
        -------
        xr.DataArray, shape (n_time, n_states)
            Posterior probabilities for each state at each time point,
            with dimensions (time, state) where states are
            [Continuous, Fragmented]. The state dim is named ``state``
            (singular), matching the MultiIndex level name on
            ``acausal_posterior``.

        Examples
        --------
        >>> results = classifier.predict(...)
        >>> state_probs = classifier.get_posterior(results)
        >>> continuous_prob = state_probs.sel(state='Continuous')
        """
        return _state_probabilities_from_results(results)


@document_model(_CONT_FRAG_DOC, "sorted_spikes")
class ContFragSortedSpikesClassifier(_ContFragResultsMixin, SortedSpikesDetector):
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
        sampling_frequency: float = 500,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
    ):
        params = _initialize_params(
            _ModelDefaults.cont_frag_defaults(),
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


@document_model(_CONT_FRAG_DOC, "clusterless")
class ContFragClusterlessClassifier(_ContFragResultsMixin, ClusterlessDetector):
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
            _ModelDefaults.cont_frag_defaults(),
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
