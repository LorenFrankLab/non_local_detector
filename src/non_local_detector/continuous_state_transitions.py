import numpy as np
from replay_trajectory_classification.continuous_state_transitions import (  # noqa
    EmpiricalMovement,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)


class Discrete:
    pass

    def make_state_transition(self, *args, **kwargs):
        """Creates a continuous transition matrix for a discrete state space.

        Essentially, there is no continuous state space, so the transition matrix is just an identity matrix.

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        return np.ones((1, 1))
