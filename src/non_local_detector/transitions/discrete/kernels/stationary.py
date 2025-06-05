from dataclasses import dataclass

import numpy as np

from ..base import DiscreteTransitionModel
from ..estimation import estimate_stationary_state_transition
from ..registry import register_discrete_transition


@dataclass
@register_discrete_transition("stationary")
class Stationary(DiscreteTransitionModel):
    """P(z_{t+1}|z_t) = constant matrix.

    Attributes
    ----------
    initial_transition_matrix : np.ndarray
        The initial transition matrix to be used as a starting point for the
        stationary state transition estimation.
    concentration : float
        The concentration parameter for the Dirichlet prior, which influences
        the strength of the prior on the transition probabilities.
    stickiness : float or np.ndarray
        The stickiness parameter, which can be a single float value or an array
        specifying the stickiness for each state. It represents the tendency of
        the system to remain in the same state.
    """

    initial_transition_matrix: np.ndarray  # (n_states, n_states)
    concentration: float = 1.0
    stickiness: float | np.ndarray = 0.0

    # cache
    _matrix: np.ndarray = None

    @property
    def n_states(self) -> int:
        return self.initial_transition_matrix.shape[0]

    def __post_init__(self):
        if self.initial_transition_matrix.ndim != 2:
            raise ValueError("initial_transition_matrix must be a 2D array.")
        if (
            self.initial_transition_matrix.shape[0]
            != self.initial_transition_matrix.shape[1]
        ):
            raise ValueError("initial_transition_matrix must be square.")
        if (
            np.testing.assert_array_almost_equal(
                self.initial_transition_matrix.sum(axis=1), 1.0
            )
            is not None
        ):
            raise ValueError(
                "initial_transition_matrix rows must sum to 1.0 (probability distribution)."
            )
        if self.concentration <= 0:
            raise ValueError("concentration must be a positive float.")
        if isinstance(self.stickiness, (int, float)):
            self.stickiness = np.full(self.n_states, self.stickiness)
        elif isinstance(self.stickiness, np.ndarray):
            if self.stickiness.ndim != 1 or len(self.stickiness) != self.n_states:
                raise ValueError("stickiness must be a 1D array of length n_states.")

    # --- API impl. --------------------------------------------------
    def matrix(self, *, covariate_data=None) -> np.ndarray:
        """
        Returns the transition matrix for the current instance.

        If the transition matrix has not been computed yet, it initializes it using
        the `initial_transition_matrix` attribute. Optionally accepts covariate data,
        though it is not used in the current implementation.

        Parameters
        ----------
        covariate_data : None
            Optional covariate data, not used in this implementation.

        Returns
        -------
        transition_matrix : np.ndarray, shape (n_states, n_states)
            The transition matrix representing the probabilities of transitioning
            from one state to another.
        """
        if self._matrix is None:
            self._matrix = self.initial_transition_matrix
        return self._matrix

    def update_parameters(
        self, *, causal_posterior, predictive_distribution, acausal_posterior
    ) -> None:
        """
        Updates the internal transition matrix parameters using the
        provided causal, predictive, and acausal data.

        This method estimates a new stationary state transition matrix
        based on the given inputs and updates the internal `_matrix` attribute.
        If the current transition matrix is not set, it uses the initial
        transition matrix as a starting point.

        Parameters
        ----------
        causal_posterior : np.ndarray, shape (n_time, n_states,)
            Causal data representing the influence of previous states.
        predictive_distribution : np.ndarray, shape (n_time, n_states,)
            Predictive data representing the expected future states.
        acausal_posterior : np.ndarray, shape (n_time, n_states,)
            Acausal data representing the influence of future states.

        Returns
        -------
            None
        """

        transition_matrix = (
            self._matrix if self._matrix is not None else self.initial_transition_matrix
        )
        self._matrix = estimate_stationary_state_transition(
            causal_posterior=causal_posterior,
            predictive_distribution=predictive_distribution,
            transition_matrix=transition_matrix,
            acausal_posterior=acausal_posterior,
            concentration=self.concentration,
            stickiness=self.stickiness,
        )
