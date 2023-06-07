import numpy as np


def set_initial_conditions(
    state_ind: np.ndarray, state_names: list, local_state_name: str = "local"
) -> np.ndarray:
    """Set initial conditions for the causal algorithm assuming the first time bin is in the local state

    Parameters
    ----------
    state_ind : np.ndarray, shape (n_state_bins,)
    state_names : list, len (n_states)
    local_state_name : str, optional
        by default "local"

    Returns
    -------
    initial_conditions : np.ndarray, shape (n_state_bins,)

    """
    initial_conditions = np.zeros((len(state_ind),))
    initial_conditions[state_ind == state_names.index(local_state_name)] = 1.0

    return initial_conditions


def estimate_initial_conditions(acausal_posterior: np.ndarray) -> np.ndarray:
    """Estimate initial conditions from acausal posterior distribution via EM algorithm

    Parameters
    ----------
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution

    Returns
    -------
    initial_conditions : np.ndarray, shape (n_states,)
        Estimated initial conditions

    """
    return acausal_posterior[0]
