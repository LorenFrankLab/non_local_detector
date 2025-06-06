from typing import Union

import numpy as np

from .kernels.stationary import Stationary


def _make_transition_from_diag(diag: np.ndarray) -> np.ndarray:
    """Make a transition matrix where diagonal probabilities are `diag`,
    and off-diagonal probabilities are uniform for the remaining probability.

    Parameters
    ----------
    diag : np.ndarray, shape (n_states,)
        The desired diagonal probabilities of the transition matrix.

    Returns
    -------
    transition_matrix : np.ndarray, shape (n_states, n_states)
        The constructed transition matrix.
    """
    n_states = len(diag)
    transition_matrix = diag * np.eye(n_states)
    if n_states == 1:
        off_diag = 1.0
    else:
        off_diag = ((1.0 - diag) / (n_states - 1.0))[:, np.newaxis]
    transition_matrix += np.ones((n_states, n_states)) * off_diag - off_diag * np.eye(
        n_states
    )

    return transition_matrix


def diag_stickiness(
    diag_probs: Union[list, tuple, np.ndarray],
    *,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
) -> Stationary:
    """Quick helper for the common 'only supply diagonals' case.

    Parameters
    ----------
    diag_probs : list, tuple, or np.ndarray
        The diagonal probabilities for the transition matrix.
    concentration : float, optional
        The concentration parameter for the Dirichlet prior, default is 1.0.
    stickiness : float or np.ndarray, optional
        The stickiness parameter, which can be a single float value or an array
        specifying the stickiness for each state, default is 0.0.

    Returns
    -------
    Stationary
        An instance of the Stationary class with the constructed transition matrix.
    """
    if diag_probs is np.isscalar(diag_probs):
        raise ValueError(
            "diag_probs must be a list, tuple, or numpy array, not a scalar"
        )
    if isinstance(diag_probs, (list, tuple)):
        diag_probs = np.array(diag_probs)
    elif not isinstance(diag_probs, np.ndarray):
        raise ValueError("diag_probs must be a list, tuple, or numpy array")
    if diag_probs.ndim != 1:
        raise ValueError(f"diag_probs must be 1D, got shape {diag_probs.shape}")

    if np.any(diag_probs < 0) or np.any(diag_probs > 1):
        raise ValueError("diag_probs must be in the range [0, 1]")

    transition_matrix = _make_transition_from_diag(diag_probs)
    return Stationary(
        transition_matrix, concentration=concentration, stickiness=stickiness
    )
