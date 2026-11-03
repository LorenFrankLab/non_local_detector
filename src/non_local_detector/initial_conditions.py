"""Initial conditions for Hidden Markov Models in non-local neural decoding.

This module provides classes and functions for setting up initial state distributions
in HMM-based neural decoders, including uniform distributions and EM-based estimation
from posterior distributions.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np

from non_local_detector.environment import Environment, find_environment_by_name
from non_local_detector.exceptions import ValidationError
from non_local_detector.observation_models import ObservationModel


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
        if environment.is_track_interior_ is None:
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
