from typing import Union

import numpy as np


def get_dirichlet_prior(
    concentration: float, stickiness: Union[float, np.ndarray], n_states: int
) -> np.ndarray:
    """Creates a Dirichlet prior for the transition matrix rows.

    Parameters
    ----------
    concentration : float
        The concentration parameter for the Dirichlet prior.
    stickiness : float or np.ndarray
        The stickiness parameter, which can be a single float value or an array
        specifying the stickiness for each state.
    n_states : int
        The number of states in the transition model.

    Returns
    -------
    np.ndarray, shape (n_states, n_states)
        A 2D array representing the prior for each state.
    """
    if isinstance(stickiness, (int, float)):
        stickiness_arr = stickiness * np.eye(n_states)
    else:
        # Assume stickiness provided per state
        stickiness_arr = np.diag(stickiness)
    return np.maximum(concentration * np.ones((n_states,)) + stickiness_arr, 1.0)
