"""Initial conditions for Hidden Markov Models in non-local neural decoding.

This module provides classes and functions for setting up initial state distributions
in HMM-based neural decoders, including uniform distributions and EM-based estimation
from posterior distributions.
"""

from dataclasses import dataclass

import numpy as np

from non_local_detector.environment import Environment
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
    >>> initial_conditions = UniformInitialConditions()
    >>> obs_model = ObservationModel(environment_name="track1")
    >>> envs = (Environment(environment_name="track1"),)
    >>> ic_array = initial_conditions.make_initial_conditions(obs_model, envs)
    """

    def make_initial_conditions(
        self,
        observation_model: ObservationModel,
        environments: tuple[Environment, ...],
    ) -> np.ndarray:
        """Creates initial conditions array for the specified observation model.

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
            returns shape (1,). For spatial models, shape matches the number of
            interior track position bins.
        """

        if observation_model.is_local or observation_model.is_no_spike:
            initial_conditions = np.ones((1,), dtype=np.float32)
        else:
            environment = environments[
                environments.index(observation_model.environment_name)
            ]
            if environment.is_track_interior_ is not None:
                initial_conditions = environment.is_track_interior_.ravel().astype(
                    np.float32
                )
            else:
                initial_conditions = np.ones((1,), dtype=np.float32)

        initial_conditions /= initial_conditions.sum()

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
    >>> posterior = np.random.rand(100, 50)  # 100 time points, 50 position bins
    >>> initial_conds = estimate_initial_conditions(posterior)
    >>> initial_conds.shape
    (50,)
    """
    return acausal_posterior[0]
