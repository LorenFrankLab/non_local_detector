from dataclasses import dataclass

import numpy as np

from non_local_detector.environment.environment import Environment
from non_local_detector.observation_models import ObservationModel


@dataclass
class UniformInitialConditions:
    """Initial conditions where all position bins are
    equally likely."""

    def make_initial_conditions(
        self,
        observation_model: ObservationModel,
        environments: list[Environment],
    ) -> np.ndarray:
        """Creates initial conditions array

        Parameters
        ----------
        observation_model : ObservationModel
        environments : list[Environment]

        Returns
        -------
        initial_conditions : np.ndarray, shape (n_place_bins,)
        """

        if observation_model.is_local or observation_model.is_no_spike:
            initial_conditions = np.ones((1,), dtype=np.float32)
        else:
            environment = environments[
                environments.index(observation_model.environment_name)
            ]
            initial_conditions = environment.is_track_interior_.ravel().astype(
                np.float32
            )

        initial_conditions /= initial_conditions.sum()

        return initial_conditions


def estimate_initial_conditions(acausal_posterior: np.ndarray) -> np.ndarray:
    """Estimate initial conditions from acausal posterior distribution via EM algorithm

    Parameters
    ----------
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Acausal posterior distribution

    Returns
    -------
    initial_conditions : np.ndarray, shape (n_state_bins,)
        Estimated initial conditions

    """
    return acausal_posterior[0]
