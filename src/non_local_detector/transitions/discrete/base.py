from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DiscreteTransitionModel(Protocol):
    """Return P(z_{t+1} | z_t, covariates) as
    (n_states, n_states) or (n_time, n_states, n_states) NumPy/JAX arrays."""

    n_states: int

    # ---------- forward pass ----------
    def matrix(
        self, *, covariate_data: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray: ...

    # ---------- EM helper  ----------
    def update_parameters(
        self,
        *,
        causal_posterior: np.ndarray,
        predictive_distribution: np.ndarray,
        acausal_posterior: np.ndarray,
    ) -> np.ndarray: ...

    """Return the transition matrix after updating parameters.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        Predictive distribution
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution

    Returns
    -------
    np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
        The updated transition matrix, which is a square matrix of shape
        (n_states, n_states) or a 3D array of shape (n_time, n_states, n_states).
        If the model is time-dependent, the first dimension will be the time
        dimension, and each slice along this dimension will be a transition matrix
        for that time step.
    """
