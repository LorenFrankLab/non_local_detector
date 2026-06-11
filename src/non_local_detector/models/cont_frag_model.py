import numpy as np
import xarray as xr

from non_local_detector._position_dims import get_position_dims
from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
from non_local_detector.models.base import (
    _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
    _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
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


class ContFragSortedSpikesClassifier(SortedSpikesDetector):
    """Classifier for continuous vs fragmented non-local activity using sorted spike data.

    This classifier uses a two-state Hidden Markov Model to distinguish between
    continuous and fragmented non-local neural replay events in sorted spike data.
    The continuous state models smooth spatial trajectories during replay, while
    the fragmented state captures disjointed spatial representations.

    The model transitions:
    - Continuous state: Uses RandomWalk for within-state and Uniform for between-state transitions
    - Fragmented state: Uses Uniform transitions for both within and between states

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (2,), optional
        Initial probabilities for [Continuous, Fragmented] states, by default
        np.ones((2,)) / 2 (uniform distribution).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial condition types for continuous states, by default uniform
        initial conditions for both states.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transitions, by default stationary diagonal
        with 98% self-transition probability.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Stickiness parameters for state persistence, by default [0.0, 0.0].
    discrete_transition_regularization : float, optional
        Regularization parameter for transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions or None, optional
        Transition types for continuous states. If None, uses default:
        [[RandomWalk(), Uniform()], [Uniform(), Uniform()]].
    observation_models : Observations, optional
        Observation models for likelihood computation, by default empty
        ObservationModel instances.
    environments : Environments, optional
        Environment specifications, by default empty Environment.
    sorted_spikes_algorithm : str, optional
        Algorithm for sorted spikes likelihood, by default 'sorted_spikes_kde'.
    sorted_spikes_algorithm_params : dict, optional
        Parameters for the sorted spikes algorithm, by default uses package defaults.
    infer_track_interior : bool, optional
        Whether to infer track interior during decoding, by default True.
    state_names : StateNames, optional
        Names for the states, by default ['Continuous', 'Fragmented'].
    sampling_frequency : float, optional
        Data sampling frequency in Hz, by default 500.
    no_spike_rate : float, optional
        Rate parameter for no-spike periods, by default 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from non_local_detector.models.cont_frag_model import ContFragSortedSpikesClassifier
    >>> # Create classifier with default parameters
    >>> classifier = ContFragSortedSpikesClassifier()
    >>> # Fit to training data
    >>> classifier.fit(position, spikes)
    >>> # Classify test data
    >>> results = classifier.predict(test_spikes)
    >>> # Get state probabilities
    >>> state_probs = classifier.get_posterior(results)
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
        sampling_frequency: float = 500,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = None,
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
        >>> results = classifier.predict(test_spikes)
        >>> state_probs = classifier.get_posterior(results)
        >>> continuous_prob = state_probs.sel(state='Continuous')
        """
        return _state_probabilities_from_results(results)


class ContFragClusterlessClassifier(ClusterlessDetector):
    """Classifier for continuous vs fragmented non-local activity using clusterless data.

    This classifier uses a two-state Hidden Markov Model to distinguish between
    continuous and fragmented non-local neural replay events in clusterless
    (continuous) spike data. The continuous state models smooth spatial trajectories
    during replay, while the fragmented state captures disjointed spatial
    representations.

    The model transitions:
    - Continuous state: Uses RandomWalk for within-state and Uniform for between-state transitions
    - Fragmented state: Uses Uniform transitions for both within and between states

    Parameters
    ----------
    discrete_initial_conditions : np.ndarray, shape (2,), optional
        Initial probabilities for [Continuous, Fragmented] states, by default
        np.ones((2,)) / 2 (uniform distribution).
    continuous_initial_conditions_types : ContinuousInitialConditions, optional
        Initial condition types for continuous states, by default uniform
        initial conditions for both states.
    discrete_transition_type : DiscreteTransitions, optional
        Type of discrete state transitions, by default stationary diagonal
        with 98% self-transition probability.
    discrete_transition_concentration : float, optional
        Concentration parameter for discrete transitions, by default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Stickiness parameters for state persistence, by default [0.0, 0.0].
    discrete_transition_regularization : float, optional
        Regularization parameter for transitions, by default 1e-10.
    continuous_transition_types : ContinuousTransitions or None, optional
        Transition types for continuous states. If None, uses default:
        [[RandomWalk(), Uniform()], [Uniform(), Uniform()]].
    observation_models : Observations, optional
        Observation models for likelihood computation, by default empty
        ObservationModel instances.
    environments : Environments, optional
        Environment specifications, by default empty Environment.
    clusterless_algorithm : str, optional
        Algorithm for clusterless data likelihood, by default 'clusterless_kde'.
    clusterless_algorithm_params : dict, optional
        Parameters for the clusterless algorithm, by default uses package defaults.
    infer_track_interior : bool, optional
        Whether to infer track interior during decoding, by default True.
    state_names : StateNames, optional
        Names for the states, by default ['Continuous', 'Fragmented'].
    sampling_frequency : float, optional
        Data sampling frequency in Hz, by default 500.0.
    no_spike_rate : float, optional
        Rate parameter for no-spike periods, by default 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from non_local_detector.models.cont_frag_model import ContFragClusterlessClassifier
    >>> # Create classifier with default parameters
    >>> classifier = ContFragClusterlessClassifier()
    >>> # Fit to training data
    >>> classifier.fit(position, clusterless_spikes)
    >>> # Classify test data
    >>> results = classifier.predict(test_clusterless_spikes)
    >>> # Get state probabilities
    >>> state_probs = classifier.get_posterior(results)
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
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = None,
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
        >>> results = classifier.predict(test_clusterless_spikes)
        >>> state_probs = classifier.get_posterior(results)
        >>> continuous_prob = state_probs.sel(state='Continuous')
        """
        return _state_probabilities_from_results(results)
