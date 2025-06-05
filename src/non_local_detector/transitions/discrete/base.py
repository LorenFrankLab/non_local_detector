from typing import Dict, Optional, Protocol

import numpy as np


class DiscreteTransitionModel(Protocol):
    """Return P(z_{t+1} | z_t, covariates) as
    (n_states, n_states) or (n_time, n_states, n_states) NumPy/JAX arrays."""

    n_states: int

    # ---------- forward pass ----------
    def matrix(
        self, *, covars: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray: ...

    # ---------- EM helper  ----------
    def update_parameters(
        self, *, causal: np.ndarray, predictive: np.ndarray, acausal: np.ndarray
    ) -> None: ...
