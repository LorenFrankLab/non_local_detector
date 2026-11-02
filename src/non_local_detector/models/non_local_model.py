import numpy as np
import xarray as xr

from non_local_detector.exceptions import ValidationError
from non_local_detector.models._defaults import (
    _initialize_params,
    _ModelDefaults,
)
from non_local_detector.models._docstrings import document_model
from non_local_detector.models.base import (
    ClusterlessDetector,
    SortedSpikesDetector,
)
from non_local_detector.observation_models import ObservationModel
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


def _get_conditional_non_local_posterior(
    results: xr.Dataset,
    state_names: list[str],
    observation_models: list[ObservationModel],
) -> xr.DataArray:
    """Posterior over position given that activity is non-local.

    Sums the posterior of every non-local state -- each state whose
    observation model is neither Local (``is_local``) nor No-Spike
    (``is_no_spike``) -- and normalizes across position.

    Parameters
    ----------
    results : xr.Dataset
        Results from the detector's ``predict`` method, containing
        ``acausal_posterior`` with a ``state`` coordinate.
    state_names : list of str, length n_states
        The detector's state names, as labelled in ``results``.
    observation_models : list of ObservationModel, length n_states
        The detector's observation models, in the same order.

    Returns
    -------
    conditional_posterior : xr.DataArray
        Posterior over position given a non-local state. Shape is
        ``(n_time, n_position_bins)`` for 1-D environments and
        ``(n_time, n_x_bins, n_y_bins)`` for 2-D. Rows where the non-local
        states have no posterior mass are NaN.

    Raises
    ------
    ValidationError
        If the detector has no non-local states, or ``results`` lacks them
        (it came from a different detector).
    """
    non_local_states = list(
        dict.fromkeys(
            name
            for name, obs in zip(state_names, observation_models, strict=True)
            if not obs.is_local and not obs.is_no_spike
        )
    )
    if not non_local_states:
        raise ValidationError(
            "The detector has no non-local states",
            expected="at least one observation model with is_local=False "
            "and is_no_spike=False",
            got=f"states {list(state_names)}",
        )
    result_states = set(results.acausal_posterior.coords["state"].values)
    missing = [name for name in non_local_states if name not in result_states]
    if missing:
        raise ValidationError(
            f"results lack the detector's non-local states {missing}",
            expected=f"results with states {list(state_names)}",
            got=f"states {sorted(result_states)}",
            hint="Pass results from this detector's predict method.",
        )

    acausal_posterior = sum(
        results.acausal_posterior.sel(state=name) for name in non_local_states
    )

    # Marginalize over position. After .sel(state=...), the position bins stay
    # stacked under the single "state_bins" dim (2D+) or appear as the unstacked
    # "position" dim (1D), so summing every non-"time" dim handles both. NB: this
    # is the stacked layout, not the named-position-dim convention used
    # elsewhere -- do not substitute _position_dims.get_position_dims here; it
    # would return [] for the stacked "state_bins" case and break normalization.
    position_dims = [d for d in acausal_posterior.dims if d != "time"]
    return acausal_posterior / acausal_posterior.sum(position_dims)


def _validate_penalty_params(
    non_local_position_penalty: float, non_local_penalty_std: float
) -> None:
    """Validate non-local position penalty parameters.

    Parameters
    ----------
    non_local_position_penalty : float
        Magnitude of the Gaussian penalty. Must be non-negative.
    non_local_penalty_std : float
        Standard deviation of the penalty Gaussian in position units.
        Must be strictly positive.

    Raises
    ------
    ValidationError
        If ``non_local_position_penalty < 0`` or
        ``non_local_penalty_std <= 0``.
    """
    if non_local_position_penalty < 0:
        raise ValidationError(
            "non_local_position_penalty must be >= 0",
            expected="non_local_position_penalty >= 0",
            got=f"{non_local_position_penalty}",
            hint="A negative penalty would reward non-local states near the animal's "
            "position instead of suppressing them. Use 0.0 to disable the penalty.",
        )
    if non_local_penalty_std <= 0:
        raise ValidationError(
            "non_local_penalty_std must be > 0",
            expected="non_local_penalty_std > 0",
            got=f"{non_local_penalty_std}",
            hint="Standard deviation controls the width of the Gaussian penalty "
            "and is used as a divisor. A value <= 0 would cause divide-by-zero "
            "or invalid behavior.",
        )


_NON_LOCAL_DOC = """Detector of non-local activity, such as replay, from $data.

A four-state model:

- Local: activity represents the animal's current position.
- No-Spike: a near-silent firing model without position.
- Non-Local Continuous: the represented position follows a random walk.
- Non-Local Fragmented: the represented position jumps uniformly over the
  track.

Parameters
----------
discrete_initial_conditions : np.ndarray, shape (4,), optional
    Initial probabilities of the four states, by default
    ``[1.0, 0.0, 0.0, 0.0]``.
continuous_initial_conditions_types : ContinuousInitialConditions, optional
    Initial position distribution of each state, by default uniform for
    all four.
discrete_transition_type : DiscreteTransitions, optional
    Discrete transition model, by default ``DiscreteStationaryCustom`` with
    self-transition probabilities 0.99 (Local), 0.98 (No-Spike) and 0.96
    (each Non-Local state): expected dwell times of 200, 100 and 50 ms in
    2 ms bins. Local and the Non-Local states enter No-Spike with
    probability 0.005.
$discrete_transition_concentration
discrete_transition_stickiness : Stickiness, shape (4,), optional
    Added to each state's self-transition in the Dirichlet prior, by default
    ``[100000.0, 100000.0, 300.0, 300.0]``.
$discrete_transition_regularization
continuous_transition_types : ContinuousTransitions, optional
    Position dynamics from each state (row) to each state (column), by
    default ``Discrete()`` into Local and No-Spike, ``RandomWalk()`` within
    Non-Local Continuous, and ``Uniform()`` otherwise.
observation_models : Observations, optional
    Observation model of each state, by default
    ``ObservationModel(is_local=True)``,
    ``ObservationModel(is_no_spike=True)`` and ``ObservationModel()`` for
    each Non-Local state. The states' roles are read from these flags.
environments : Environments, optional
    Spatial environment, by default ``Environment(environment_name="")``.
$algorithm_parameters
$infer_track_interior
state_names : StateNames, optional
    Names of the states, by default ``["Local", "No-Spike",
    "Non-Local Continuous", "Non-Local Fragmented"]``.
$sampling_frequency
$no_spike_rate
non_local_position_penalty : float, optional
    Magnitude, in nats, of the Gaussian penalty subtracted from the
    Non-Local states' log-likelihood near the animal's position. Suppresses
    non-local detection near the current location. By default 0.0 (no
    penalty). Must be >= 0.
non_local_penalty_std : float, optional
    Standard deviation of the penalty Gaussian, in the units of
    ``position`` (typically cm), by default 1.0. Distances are shortest
    paths on the track graph. Must be > 0.
$discrete_transition_prior_weight
$frozen_discrete_transition_rows
local_position_std : float or None, optional
    Spatial representation of the Local state:

    - ``None`` (default): Local occupies a single bin; its likelihood is
      evaluated at the animal's interpolated position.
    - ``0.0``: Local spans all position bins, with all mass at the bin
      containing the animal; the likelihood is evaluated at bin centers.
    - ``> 0``: Local spans all position bins with a Gaussian kernel of this
      standard deviation (units of ``position``, typically cm) on the
      shortest-path track-graph distance, modelling spatial uncertainty.

Examples
--------
>>> from non_local_detector import $cls, time_edges_from_centers
>>> detector = $cls()
>>> detector.fit($fit_example)
>>> results = detector.predict(
...     $predict_example,
...     time_edges=time_edges_from_centers(position_time),
...     position=position,
...     position_time=position_time,
... )
>>> non_local_posterior = detector.get_conditional_non_local_posterior(results)
"""


@document_model(_NON_LOCAL_DOC, "sorted_spikes")
class NonLocalSortedSpikesDetector(SortedSpikesDetector):
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
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        non_local_position_penalty: float = 0.0,
        non_local_penalty_std: float = 1.0,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ):
        params = _initialize_params(
            _ModelDefaults.non_local_defaults(),
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
            local_position_std=local_position_std,
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_std)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_std = non_local_penalty_std

    def get_conditional_non_local_posterior(self, results: xr.Dataset) -> xr.DataArray:
        """Posterior over position given that activity is non-local.

        The non-local states are those whose observation model is neither
        Local nor No-Spike, wherever they are listed. See
        :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(
            results, self.state_names, self.observation_models
        )


@document_model(_NON_LOCAL_DOC, "clusterless")
class NonLocalClusterlessDetector(ClusterlessDetector):
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
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict | None = None,
        infer_track_interior: bool = True,
        state_names: StateNames | None = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        non_local_position_penalty: float = 0.0,
        non_local_penalty_std: float = 1.0,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: FrozenTransitionRows = "no_spike",
        local_position_std: float | None = None,
    ):
        params = _initialize_params(
            _ModelDefaults.non_local_defaults(),
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
            local_position_std=local_position_std,
        )
        _validate_penalty_params(non_local_position_penalty, non_local_penalty_std)
        self.non_local_position_penalty = non_local_position_penalty
        self.non_local_penalty_std = non_local_penalty_std

    def get_conditional_non_local_posterior(self, results: xr.Dataset) -> xr.DataArray:
        """Posterior over position given that activity is non-local.

        The non-local states are those whose observation model is neither
        Local nor No-Spike, wherever they are listed. See
        :func:`_get_conditional_non_local_posterior` for full documentation.
        """
        return _get_conditional_non_local_posterior(
            results, self.state_names, self.observation_models
        )
