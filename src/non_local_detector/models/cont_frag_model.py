"""Continuous vs. fragmented replay classifiers.

Two-state models, for sorted spikes and clusterless data, that distinguish
replay that moves continuously through space from fragmented replay.
"""

from typing import Any, cast

import numpy as np
import xarray as xr

from non_local_detector._position_dims import get_position_dims
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
        msg = (
            "get_posterior found no position dimension to marginalize over; "
            f"acausal_posterior unstacked to dims {tuple(unstacked.dims)}. "
            "Expected a dim named 'position' or ending in '_position'."
        )
        raise ValueError(msg)
    return cast("xr.DataArray", unstacked.sum(position_dims))


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


class ContFragSortedSpikesClassifier(_ContFragResultsMixin, SortedSpikesDetector):
    """Classifier of continuous vs. fragmented replay from sorted spike data.

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
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[0.0, 0.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``[[RandomWalk(), Uniform()], [Uniform(), Uniform()]]``.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``[ObservationModel(), ObservationModel()]``.
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
        Names of the states, by default ``["Continuous", "Fragmented"]``.
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
    ...     ContFragSortedSpikesClassifier,
    ...     time_edges_from_centers,
    ... )
    >>> classifier = ContFragSortedSpikesClassifier()
    >>> classifier.fit(position_time, position, spike_times)
    >>> time_edges = time_edges_from_centers(position_time)
    >>> results = classifier.predict(
    ...     spike_times,
    ...     time_edges=time_edges,
    ... )
    >>> state_probs = classifier.get_posterior(results)  # (n_time, n_states)
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


class ContFragClusterlessClassifier(_ContFragResultsMixin, ClusterlessDetector):
    """Classifier of continuous vs. fragmented replay from clusterless spike data.

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
    discrete_transition_concentration : float, optional
        Dirichlet prior parameter of every entry of each discrete transition row
        during EM re-estimation; ``discrete_transition_stickiness`` is added to
        the self-transition. By default 1.1.
    discrete_transition_stickiness : Stickiness, shape (2,), optional
        Added to each state's self-transition in the Dirichlet prior, by default
        ``[0.0, 0.0]``.
    discrete_transition_regularization : float, optional
        L2 penalty on the non-intercept coefficients of covariate-dependent
        discrete transitions, equivalent to a Gaussian prior with standard
        deviation ``1 / sqrt(2 * regularization)`` on each, by default 0.1.
        Stationary transitions do not use it.
    continuous_transition_types : ContinuousTransitions, optional
        Position dynamics from each state (row) to each state (column), by
        default ``[[RandomWalk(), Uniform()], [Uniform(), Uniform()]]``.
    observation_models : Observations, optional
        Observation model of each state, by default
        ``[ObservationModel(), ObservationModel()]``.
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
        Names of the states, by default ``["Continuous", "Fragmented"]``.
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
    ...     ContFragClusterlessClassifier,
    ...     time_edges_from_centers,
    ... )
    >>> classifier = ContFragClusterlessClassifier()
    >>> classifier.fit(position_time, position, spike_times, spike_waveform_features)
    >>> time_edges = time_edges_from_centers(position_time)
    >>> results = classifier.predict(
    ...     spike_times, spike_waveform_features,
    ...     time_edges=time_edges,
    ... )
    >>> state_probs = classifier.get_posterior(results)  # (n_time, n_states)
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
