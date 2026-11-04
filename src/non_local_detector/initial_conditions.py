"""Initial conditions for Hidden Markov Models in non-local neural decoding.

This module provides classes and functions for setting up initial state distributions
in HMM-based neural decoders, including uniform distributions and EM-based estimation
from posterior distributions.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np

from non_local_detector._validation import PROBABILITY_SUM_TOLERANCE
from non_local_detector.environment import Environment, find_environment_by_name
from non_local_detector.exceptions import ValidationError
from non_local_detector.observation_models import ObservationModel

__all__ = [
    "UniformInitialConditions",
]


@dataclass
class UniformInitialConditions:
    """Initial conditions where all position bins are equally likely.

    This class creates uniform initial state distributions for HMM decoders,
    ensuring all spatial positions have equal prior probability at the start
    of decoding. The initial conditions are automatically adjusted based on
    the observation model type and environmental constraints.

    Examples
    --------
    >>> import numpy as np
    >>> initial_conditions = UniformInitialConditions()
    >>> obs_model = ObservationModel(environment_name="track1")
    >>> environment = Environment(environment_name="track1", place_bin_size=10.0)
    >>> envs = (environment.fit_place_grid(np.linspace(0, 100, 50)[:, None]),)
    >>> ic_array = initial_conditions.make_initial_conditions(obs_model, envs)
    """

    def make_initial_conditions(
        self,
        observation_model: ObservationModel,
        environments: tuple[Environment, ...],
    ) -> np.ndarray:
        """Create the initial conditions array for the specified observation model.

        Generates a uniform probability distribution over position bins,
        with special handling for local and no-spike observation models.

        Parameters
        ----------
        observation_model : ObservationModel
            The observation model specifying environment and decoding type.
        environments : tuple[Environment, ...]
            Tuple of available environments to match against the observation model.

        Returns
        -------
        initial_conditions : np.ndarray, shape (n_place_bins,)
            Normalized initial probability distribution. For local/no-spike models,
            returns shape (1,). For spatial models, one value per place bin of
            the environment: uniform over its interior bins, 0 elsewhere.

        Raises
        ------
        ValidationError
            If the observation model's environment is not fitted or has no
            interior bins.
        """
        if observation_model.is_local or observation_model.is_no_spike:
            return np.ones((1,), dtype=np.float32)
        environment = find_environment_by_name(
            environments, observation_model.environment_name
        )
        name = environment.environment_name
        if not environment._is_fitted:
            msg = f"Environment {name!r} is not fitted, so it has no place bins"
            raise ValidationError(
                msg,
                hint=(
                    "Fit the detector (or call environment.fit_place_grid) "
                    "before making initial conditions."
                ),
            )
        initial_conditions = environment.is_track_interior_.ravel().astype(np.float32)
        total = initial_conditions.sum()
        if total == 0:
            msg = f"Environment {name!r} has no interior bins"
            raise ValidationError(
                msg,
                got=f"{initial_conditions.size} place bins, none on the track",
                hint=(
                    "Check is_track_interior, or the position data the "
                    "environment's track interior is inferred from."
                ),
            )
        initial_conditions /= total
        return initial_conditions


def _check_state_initial_conditions(
    values: np.ndarray,
    *,
    state_name: str,
    initial_conditions: object,
    is_interior: np.ndarray,
    discrete_weight: float,
) -> None:
    """Raise unless ``values`` is a distribution over one state's interior bins.

    The detectors weight each state's continuous initial conditions by the
    state's discrete initial probability and decode over the interior bins,
    so the joint initial distribution sums to 1 only if each state's values
    over its interior bins do. Checking each state names the state and the
    initial-condition class that broke this, instead of the joint
    distribution at decoding. A state of discrete weight 0 adds nothing to
    the joint distribution, so its sum is not checked.

    Parameters
    ----------
    values : np.ndarray
        Output of ``initial_conditions.make_initial_conditions`` for the state.
    state_name : str
        Name of the discrete state.
    initial_conditions : object
        The state's initial-condition object, named in the error.
    is_interior : np.ndarray of bool, shape (n_state_bins,)
        Which of the state's bins are on the track; one True for a state
        without place bins.
    discrete_weight : float
        The state's discrete initial probability.

    Raises
    ------
    ValidationError
        If ``values`` does not have one entry per bin of the state, has a
        negative or non-finite entry, or, for a state of nonzero
        ``discrete_weight``, does not sum to 1 (within
        ``PROBABILITY_SUM_TOLERANCE``) over the interior bins.
    """
    label = (
        f"Initial conditions of state {state_name!r} from "
        f"{type(initial_conditions).__name__}.make_initial_conditions"
    )
    values = np.asarray(values)
    if values.shape != is_interior.shape:
        msg = f"{label} have the wrong shape"
        raise ValidationError(
            msg,
            expected=f"shape {is_interior.shape}, one value per bin of the state",
            got=f"shape {values.shape}",
            hint="Return one value per place bin of the state's environment, "
            "or shape (1,) for a state without place bins (Local, No-Spike).",
        )
    interior_sum = values[is_interior].sum(dtype=np.float64)
    got = None
    if not np.all(np.isfinite(values)):
        got = "non-finite entries (nan or inf)"
    elif values.min() < 0:
        got = f"minimum entry {values.min():.6g} at bin {int(np.argmin(values))}"
    elif (
        discrete_weight != 0 and not abs(interior_sum - 1) <= PROBABILITY_SUM_TOLERANCE
    ):
        got = f"sums to {interior_sum:.6g} over its {int(is_interior.sum())} interior bins"
    if got is not None:
        msg = f"{label} must be a probability distribution over the state's bins"
        raise ValidationError(
            msg,
            expected="nonnegative, finite values summing to 1 (within "
            f"{PROBABILITY_SUM_TOLERANCE:g}) over the interior bins",
            got=got,
            hint="Return values summing to 1 over the environment's interior "
            "bins (is_track_interior_flat_) and 0 elsewhere, as "
            "UniformInitialConditions does; discrete_initial_conditions "
            "weights the states.",
        )


def estimate_initial_conditions(acausal_posterior: np.ndarray) -> np.ndarray:
    """Estimate initial conditions from acausal posterior distribution via EM algorithm.

    Extracts the initial state distribution by taking the first time point
    from the smoothed (acausal) posterior distribution. This is used in
    Expectation-Maximization parameter estimation for HMMs.

    Parameters
    ----------
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Acausal (smoothed) posterior distribution over states and time,
        typically obtained from forward-backward algorithm.

    Returns
    -------
    initial_conditions : np.ndarray, shape (n_state_bins,)
        Estimated initial state distribution at t=0.

    Notes
    -----
    This function implements the M-step update for initial conditions
    in the EM algorithm for HMM parameter estimation, as described
    in standard HMM literature (e.g., Rabiner, 1989).

    Examples
    --------
    >>> import numpy as np
    >>> posterior = np.random.default_rng(0).random((100, 50))  # 100 time points, 50 position bins
    >>> initial_conds = estimate_initial_conditions(posterior)
    >>> initial_conds.shape
    (50,)
    """
    return cast(np.ndarray, acausal_posterior[0])
