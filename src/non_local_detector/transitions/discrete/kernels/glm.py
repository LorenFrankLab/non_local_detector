"""
transitions/discrete/glm.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Covariate-dependent discrete-state transition model implemented as a
categorical (soft-max) GLM.  Conforms to the `TransitionModel` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from patsy import DesignMatrix, build_design_matrices, dmatrix

from ..base import DiscreteTransitionModel
from ..estimation import (
    estimate_non_stationary_state_transition,
    predict_discrete_state_transitions,
)
from ..registry import register_discrete_transition

# --------------------------------------------------------------------------- #
#  Type aliases                                                               #
# --------------------------------------------------------------------------- #
Array = np.ndarray
CovariateDict = Dict[str, Array]
Covariates = Union[pd.DataFrame, CovariateDict]


# --------------------------------------------------------------------------- #
#  Concrete implementation                                                    #
# --------------------------------------------------------------------------- #
@dataclass
@register_discrete_transition("glm")
class CategoricalGLM(DiscreteTransitionModel):
    """
    Categorical-GLM transition model
    :math:`P\\,(z_{t+1}\\mid z_t, \\text{covariates}_t)`.

    Parameters
    ----------
    n_states : int
        Number of discrete latent states *D*.
    formula : str
        **patsy** formula string (for example ``"1 + bs(speed, df=5)"``).
    concentration : float
        Dirichlet prior strength for each transition row (pseudo-count).
    stickiness : float or np.ndarray
        Additive “stay probability” prior (diagonal reinforcement).
    l2_penalty : float
        Ridge regularisation strength on the coefficient tensor.
    """

    n_states: int
    formula: str

    concentration: float = 1.0
    stickiness: Union[float, Array] = 0.0
    l2_penalty: float = 1e-5

    # Populated by `initialize_parameters`
    _design_matrix: DesignMatrix | None = field(init=False, default=None, repr=False)
    _coefficients: Array | None = field(init=False, default=None, repr=False)

    # --------------------------------------------------------------------- #
    #  Lifecycle helper                                                     #
    # --------------------------------------------------------------------- #
    def initialize_parameters(
        self,
        covariate_data: Covariates,
        *,
        intercept_matrix: Optional[Array] = None,
    ) -> "CategoricalGLM":
        """
        Build and cache a patsy design matrix, then create initial coefficients.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict[str, np.ndarray]
            Covariate data for the design matrix.
        intercept_matrix : np.ndarray, optional
            Initial intercept matrix of shape `(n_states, n_states - 1)`.
            If provided, the first row of the coefficient tensor will be set
            to this matrix, representing the initial bias for each state.

        Returns
        -------
        CategoricalGLM
            The instance itself, with design matrix and coefficients initialized.

        Raises
        ------
        ValueError
            If `intercept_matrix` is provided but does not match the expected shape.
        """
        design_matrix: DesignMatrix = dmatrix(self.formula, covariate_data)
        n_features: int = design_matrix.shape[1]

        self._design_matrix = design_matrix

        # Tensor: (n_features, n_states, n_states-1)
        self._coefficients = np.zeros(
            (n_features, self.n_states, self.n_states - 1),
            dtype=design_matrix.dtype,
        )
        if intercept_matrix is not None:
            expected_shape = (self.n_states, self.n_states - 1)
            if intercept_matrix.shape != expected_shape:
                raise ValueError(
                    "`intercept_matrix` must have shape "
                    f"{expected_shape}, got {intercept_matrix.shape}"
                )
            self._coefficients[0] = intercept_matrix  # first row is the bias

        return self

    # --------------------------------------------------------------------- #
    #  E-step helper                                                        #
    # --------------------------------------------------------------------- #
    def matrix(
        self,
        covariate_data: Optional[Covariates] = None,
    ) -> Array:
        """
        Return a time-varying transition tensor of shape
        `(n_time_points, n_states, n_states)`.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict[str, np.ndarray], optional
            Covariate data for the design matrix. If not provided, uses the
            design matrix initialized in `initialize_parameters()`.

        Returns
        -------
        Array : np.ndarray, shape (n_time_points, n_states, n_states)
            Transition tensor representing the probabilities of transitioning
            from one state to another, conditioned on the covariate data.
        """
        if self._design_matrix is None or self._coefficients is None:
            raise RuntimeError("Call `initialize_parameters()` before `matrix()`.")

        return predict_discrete_state_transitions(
            self._design_matrix, self._coefficients, covariate_data
        )

    # --------------------------------------------------------------------- #
    #  M-step helper                                                        #
    # --------------------------------------------------------------------- #
    def update_parameters(
        self,
        *,
        causal_posterior: Array,
        predictive_distribution: Array,
        acausal_posterior: Array,
        covariate_data: Covariates,
        transition_tensor: Optional[Array] = None,
        **override_hyperparams: Any,
    ) -> None:
        """
        In-place MAP update of the coefficient tensor using EM posteriors.

        Parameters
        ----------
        causal_posterior : np.ndarray, shape (n_time, n_states)
            Causal posterior probabilities for each state at each time point.
        predictive_distribution : np.ndarray, shape (n_time, n_states, n_states)
            Predictive distribution of the next state given the current state
            and covariates.
        acausal_posterior : np.ndarray, shape (n_time, n_states)
            Acausal posterior probabilities for each state at each time point.
        covariate_data : pd.DataFrame or dict[str, np.ndarray]
            Covariate data for the design matrix, used to build the design matrix
            for the current update.
        transition_tensor : np.ndarray, optional
            Pre-computed transition tensor of shape `(n_time, n_states, n_states)`.
            If not provided, it will be computed from the design matrix and coefficients.
        override_hyperparams : dict, optional
            Hyperparameters to override the default values:
            - `concentration`: Dirichlet prior strength (default: `self.concentration`).
            - `stickiness`: Additive "stay probability" prior (default: `self.stickiness`).
            - `l2_penalty`: Ridge regularization strength (default: `self.l2_penalty`).

        Raises
        -------
        RuntimeError
            If the model has not been initialized with `initialize_parameters()`.
        """
        if self._coefficients is None or self._design_matrix is None:
            raise RuntimeError(
                "Model must be initialised before `update_parameters()`."
            )

        design_matrix = build_design_matrices(
            [self._design_matrix.design_info],
            covariate_data,
        )[0]

        concentration: float = override_hyperparams.get(
            "concentration", self.concentration
        )
        stickiness: Union[float, Array] = override_hyperparams.get(
            "stickiness", self.stickiness
        )
        l2_penalty: float = override_hyperparams.get("l2_penalty", self.l2_penalty)

        if transition_tensor is None:
            transition_tensor = predict_discrete_state_transitions(
                design_matrix, self._coefficients
            )

        self._coefficients, _ = estimate_non_stationary_state_transition(
            transition_coefficients=self._coefficients,
            design_matrix=design_matrix,
            causal_posterior=causal_posterior,
            predictive_distribution=predictive_distribution,
            transition_matrix=transition_tensor,
            acausal_posterior=acausal_posterior,
            concentration=concentration,
            stickiness=stickiness,
            transition_regularization=l2_penalty,
        )

    # --------------------------------------------------------------------- #
    #  Debug representation                                                 #
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        n_features: str = (
            str(self._design_matrix.shape[1])
            if self._design_matrix is not None
            else "∅"
        )
        return (
            "CategoricalGLM("
            f"n_states={self.n_states}, "
            f"n_features={n_features}, "
            f"formula={self.formula!r}), "
            f"concentration={self.concentration}, "
            f"stickiness={self.stickiness}, "
            f"l2_penalty={self.l2_penalty})"
        )
