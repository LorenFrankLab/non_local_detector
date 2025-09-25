from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from jax.nn import log_softmax
from patsy import DesignMatrix, build_design_matrices, dmatrix  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import softmax  # type: ignore[import-untyped]


def centered_softmax_forward(y: np.ndarray) -> np.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Parameters
    ----------
    y : np.ndarray, shape (..., n_states)
        The input values. Can have leading dimensions.
        The last dimension is the state dimension.

    Returns
    -------
    softmax : np.ndarray, shape (..., n_states + 1)
        The softmax of the input values

    Example
    -------
    >>> y = np.log([2, 3, 4])
    >>> np.allclose(centered_softmax_forward(y), [0.2, 0.3, 0.4, 0.1])
    True
    """
    if y.ndim == 1:
        y = np.append(y, 0)
    else:
        y = np.column_stack((y, np.zeros((y.shape[0],))))

    return softmax(y, axis=-1)


def centered_softmax_inverse(y: np.ndarray) -> np.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Parameters
    ----------
    y : np.ndarray, shape (..., n_states + 1)
        The softmax values. Can have leading dimensions.

    Returns
    -------
    inverse : np.ndarray, shape (..., n_states)
        The inverse of the softmax values

    Example
    -------
    >>> y = np.asarray([0.2, 0.3, 0.4, 0.1])
    >>> np.allclose(np.exp(centered_softmax_inverse(y)), np.asarray([2,3,4]))
    True
    """
    return np.log(y[..., :-1]) - np.log(y[..., [-1]])


def estimate_joint_distribution(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
) -> np.ndarray:
    """Estimate the joint_distribution of latents given the observations

    p(x_t, x_{t+1} | O_{1:T})

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution P(z_t | x_{1:t})
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution P(z_{t+1} | x_{1:t})
    transition_matrix : np.ndarray, shape (n_time, n_states, n_states) or shape (n_states, n_states)
        Current estimate of the transition matrix P(z_{t+1} | z_t)
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution P(z_{t+1} | x_{1:T})

    Returns
    -------
    joint_distribution : np.ndarray, shape (n_time - 1, n_states, n_states)

    """
    relative_distribution = np.where(
        np.isclose(predictive_distribution[1:], 0.0),
        0.0,
        acausal_posterior[1:] / predictive_distribution[1:],
    )[:, np.newaxis]

    if transition_matrix.ndim == 2:
        # Add a singleton dimension for the time axis
        # if the transition matrix is stationary
        # shape (1, n_states, n_states)
        joint_distribution = (
            transition_matrix[np.newaxis]
            * causal_posterior[:-1, :, np.newaxis]
            * relative_distribution
        )
    else:
        # shape (n_time - 1, n_states, n_states)
        joint_distribution = (
            transition_matrix[:-1]
            * causal_posterior[:-1, :, np.newaxis]
            * relative_distribution
        )

    return joint_distribution


@jax.jit
def jax_centered_log_softmax_forward(y: jnp.ndarray) -> jnp.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Parameters
    ----------
    y : jnp.ndarray, shape (..., n_states)
        The input values

    Returns
    -------
    log_softmax : jnp.ndarray, shape (..., n_states + 1)
        The log softmax of the input values

    Example
    -------
    >>> y = jnp.log(jnp.array([2, 3, 4]))
    >>> log_p = jax_centered_log_softmax_forward(y)
    >>> np.allclose(jnp.exp(log_p), jnp.array([0.2, 0.3, 0.4, 0.1]))
    True
    """
    if y.ndim == 1:
        y = jnp.append(y, 0)
    else:
        y = jnp.column_stack((y, jnp.zeros((y.shape[0],))))

    return log_softmax(y, axis=-1)


@jax.jit
def multinomial_neg_log_likelihood(
    coefficients: jnp.ndarray,
    design_matrix: jnp.ndarray,
    response: jnp.ndarray,
    l2_penalty: float = 1e-10,
) -> float:
    """Negative expected complete log likelihood of the transition model.

    Parameters
    ----------
    coefficients : jnp.ndarray, shape (n_coefficients * n_states - 1)
        Flattened coefficients.
    design_matrix : jnp.ndarray, shape (n_samples, n_coefficients)
    response : jnp.ndarray, shape (n_samples, n_states)
        Expected counts or probabilities for each state transition.

    Returns
    -------
    negative_expected_complete_log_likelihood : float

    """
    # Reshape flattened coefficients to shape (n_coefficients, n_states - 1)
    n_coefficients = design_matrix.shape[1]
    coefficients = coefficients.reshape((n_coefficients, -1))

    # The last state probability can be inferred from the other state probabilities
    # since the probabilities must sum to 1
    # shape (n_samples, n_states)
    log_probs = jax_centered_log_softmax_forward(design_matrix @ coefficients)

    # Average cross entropy over samples
    # Average is used instead of sum to make the negative log likelihood
    # invariant to the number of samples
    n_samples = response.shape[0]
    neg_log_likelihood = -1.0 * jnp.sum(response * log_probs) / n_samples

    # Penalize the size (squared magnitude) of the coefficients
    # Don't penalize the intercept for identifiability
    l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)

    return neg_log_likelihood + l2_penalty_term


multinomial_gradient = jax.grad(multinomial_neg_log_likelihood)
multinomial_hessian = jax.hessian(multinomial_neg_log_likelihood)


def get_transition_prior(
    concentration: float, stickiness: float | np.ndarray, n_states: int
) -> np.ndarray:
    """Creates a Dirichlet prior for the transition matrix rows."""
    if isinstance(stickiness, (int, float)):
        stickiness_arr = stickiness * np.eye(n_states)
    else:
        # Assume stickiness provided per state
        stickiness_arr = np.diag(stickiness)
    return np.maximum(concentration * np.ones((n_states,)) + stickiness_arr, 1.0)


def estimate_non_stationary_state_transition(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 1e-5,
    optimization_method: str = "Newton-CG",
    maxiter: int | None = 100,
    disp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the non-stationary state transition model using Dirichlet likelihood.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Initial estimate of the transition coefficients.
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Filtered posterior P(z_t | x_{1:t}).
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One-step predictive distribution P(z_{t+1} | x_{1:t}).
    transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Current estimate of the transition matrix P(z_{t+1} | z_t).
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Smoothed posterior P(z_{t+1} | x_{1:T}).
    concentration : float, optional
        Dirichlet prior concentration parameter (uniform part), by default 1.0.
    stickiness : float or np.ndarray, optional
        Dirichlet prior stickiness parameter (diagonal enhancement), by default 0.0.
    transition_regularization : float, optional
        L2 penalty on coefficients (excluding intercept), by default 1e-5.
    optimization_method : str, optional
        Optimization method for `scipy.optimize.minimize`, by default "Newton-CG".
    maxiter : int, optional
        Maximum iterations for optimizer, by default 100.
    disp : bool, optional
        Display optimizer convergence messages, by default False.

    Returns
    -------
    estimated_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Optimized transition coefficients.
    estimated_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Resulting non-stationary transition matrix.
    """
    # p(x_t, x_{t+1} | O_{1:T})
    joint_distribution = estimate_joint_distribution(
        causal_posterior,
        predictive_distribution,
        transition_matrix,
        acausal_posterior,
    )

    n_coefficients, n_states = transition_coefficients.shape[:2]
    estimated_transition_coefficients = np.zeros(
        (n_coefficients, n_states, (n_states - 1))
    )

    n_time = design_matrix.shape[0]
    estimated_transition_matrix = np.zeros((n_time, n_states, n_states))

    alpha = get_transition_prior(concentration, stickiness, n_states)

    # Estimate the transition coefficients for each state
    for from_state, row_alpha in enumerate(alpha):
        result = minimize(
            dirichlet_neg_log_likelihood,
            x0=transition_coefficients[:, from_state].ravel(),
            method=optimization_method,
            jac=dirichlet_gradient,
            hess=dirichlet_hessian,
            args=(
                design_matrix[:-1],
                joint_distribution[:, from_state, :],
                row_alpha,
                transition_regularization,
            ),
            options={"disp": disp, "maxiter": maxiter},
        )

        estimated_transition_coefficients[:, from_state, :] = result.x.reshape(
            (n_coefficients, n_states - 1)
        )

        linear_predictor = (
            design_matrix @ estimated_transition_coefficients[:, from_state, :]
        )
        estimated_transition_matrix[:, from_state, :] = jnp.exp(
            jax_centered_log_softmax_forward(linear_predictor)
        )

    # # if any is zero, set to small number
    # estimated_transition_matrix = np.clip(
    #     estimated_transition_matrix, 1e-16, 1.0 - 1e-16
    # )

    return estimated_transition_coefficients, estimated_transition_matrix


def estimate_stationary_state_transition(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    stickiness: float = 0.0,
    concentration: float = 1.0,
) -> np.ndarray:
    """Estimate the stationary state transition model.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
    predictive_distribution : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)
    acausal_posterior : np.ndarray, shape (n_time, n_states)
    stickiness : float, optional
    concentration : float, optional

    Returns
    -------
    new_transition_matrix : np.ndarray, shape (n_states, n_states)
    """
    # p(x_t, x_{t+1} | O_{1:T})
    joint_distribution = estimate_joint_distribution(
        causal_posterior,
        predictive_distribution,
        transition_matrix,
        acausal_posterior,
    )

    # Dirichlet prior for transition probabilities
    n_states = acausal_posterior.shape[1]
    alpha = get_transition_prior(concentration, stickiness, n_states)

    new_transition_matrix = joint_distribution.sum(axis=0) + alpha - 1.0
    new_transition_matrix /= new_transition_matrix.sum(axis=-1, keepdims=True)

    # if any is zero, set to small number
    # new_transition_matrix = np.clip(
    #     new_transition_matrix, min=1e-16, max=1.0 - 1e-16
    # )

    return new_transition_matrix


@jax.jit
def dirichlet_neg_log_likelihood(
    coefficients: jnp.ndarray,
    design_matrix: jnp.ndarray,
    response: jnp.ndarray,
    alpha: float | jnp.ndarray = 1.0,
    l2_penalty: float = 1e-5,
) -> float:
    """Negative expected complete log likelihood for Dirichlet-Multinomial model.

    Parameters
    ----------
    coefficients : jnp.ndarray, shape (n_coefficients * (n_states - 1),)
        Flattened regression coefficients.
    design_matrix : jnp.ndarray, shape (n_samples, n_coefficients)
        Covariate design matrix.
    response : jnp.ndarray, shape (n_samples, n_states)
        Expected counts or probabilities for each state transition.
    alpha : Union[float, jnp.ndarray], shape (n_states,), optional
        Dirichlet prior parameters for this row of the transition matrix.
        If float, assumed uniform. Defaults to 1.0 (Laplace smoothing).
    l2_penalty : float, optional
        L2 regularization penalty on coefficients (excluding intercept).
        Defaults to 1e-5.

    Returns
    -------
    negative_expected_complete_log_likelihood : float
        The loss value to be minimized.
    """
    n_coefficients = design_matrix.shape[1]

    # shape (n_coefficients, n_states - 1)
    coefficients = coefficients.reshape((n_coefficients, -1))

    # shape (n_samples, n_states)
    log_probs = jax_centered_log_softmax_forward(design_matrix @ coefficients)

    # Dirichlet prior
    n_samples = response.shape[0]
    prior = (alpha - 1.0) / n_samples

    neg_log_likelihood = -1.0 * jnp.sum((response + prior) * log_probs) / n_samples
    l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)

    return neg_log_likelihood + l2_penalty_term


dirichlet_gradient = jax.grad(dirichlet_neg_log_likelihood)
dirichlet_hessian = jax.hessian(dirichlet_neg_log_likelihood)


def make_transition_from_diag(diag: np.ndarray) -> np.ndarray:
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


def set_initial_discrete_transition(
    speed: np.ndarray,
    speed_knots: np.ndarray | None = None,
    is_stationary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set the initial discrete transition matrix for the local/non-local model

    Parameters
    ----------
    speed : np.ndarray, shape (n_time,), optional
        Required if `is_stationary` is False.
    speed_knots : np.ndarray, optional
        Used if `is_stationary` is False and `formula` includes knots.
    is_stationary : bool, optional
        If True, return a stationary matrix. If False, return non-stationary.
    diag : np.ndarray, optional
        Diagonal values for the initial stationary matrix.
        Defaults to [0.90, 0.90, 0.90, 0.98].
    formula : str, optional
        Patsy formula for non-stationary transitions.
        Defaults to "1 + bs(speed, knots=[1.0, 4.0, 16.0, 32.0, 64.0])".

    Returns
    -------
    discrete_transition : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
        The initial transition matrix.
    discrete_transition_coefficients : Optional[np.ndarray], shape (n_coefficients, n_states, n_states - 1)
        Initial coefficients (only if non-stationary).
    discrete_transition_design_matrix : Optional[patsy.DesignMatrix]
        Design matrix (only if non-stationary).

    Raises
    ------
    ValueError
        If `is_stationary` is False but `speed` is None.
    """
    state_names = [
        "local",
        "no_spike",
        "non-local continuous",
        "non-local fragmented",
    ]
    n_states = len(state_names)

    if is_stationary:
        diag = np.array([0.90, 0.90, 0.90, 0.98])

        discrete_transition = make_transition_from_diag(diag)

        discrete_transition_coefficients = None
        discrete_transition_design_matrix = None
    else:
        diag = np.array([0.90, 0.90, 0.90, 0.98])
        discrete_transition = make_transition_from_diag(diag)

        if speed_knots is None:
            speed_knots = [1.0, 4.0, 16.0, 32.0, 64.0]

        formula = f"1 + bs(speed, knots={speed_knots})"
        data = {"speed": np.concatenate(([0.0], speed[:-1]))}  # lagged speed
        discrete_transition_design_matrix = dmatrix(formula, data)

        n_time, n_coefficients = discrete_transition_design_matrix.shape

        discrete_transition_coefficients = np.zeros(
            (n_coefficients, n_states, n_states - 1)
        )
        discrete_transition_coefficients[0] = centered_softmax_inverse(
            discrete_transition
        )

        discrete_transition = discrete_transition[np.newaxis] * np.ones(
            (n_time, n_states, n_states)
        )

    return (
        discrete_transition,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
    )


def _estimate_discrete_transition(
    causal_state_probabilities: np.ndarray,
    predictive_state_probabilities: np.ndarray,
    acausal_state_probabilities: np.ndarray,
    discrete_transition: np.ndarray,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
    transition_concentration: float,
    transition_stickiness: float | np.ndarray,
    transition_regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the discrete transition matrix (stationary or non-stationary).

    Parameters
    ----------
    causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        P(z_t | x_{1:t})
    predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        P(z_{t+1} | x_{1:t})
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        P(z_{t+1} | x_{1:T})
    discrete_transition : np.ndarray, shape (n_time, n_states, n_states) or (n_states, n_states)
        Current transition matrix estimate.
    discrete_transition_coefficients : Optional[np.ndarray], shape (n_coefficients, n_states, n_states - 1)
        Current coefficient estimate (if non-stationary).
    discrete_transition_design_matrix : Optional[patsy.DesignMatrix]
        Design matrix (if non-stationary).
    transition_concentration : float
        Dirichlet prior concentration.
    transition_stickiness : float or np.ndarray
        Dirichlet prior stickiness.
    transition_regularization : float
        L2 penalty for non-stationary coefficients.

    Returns
    -------
    estimated_discrete_transition : np.ndarray
        Updated transition matrix.
    estimated_discrete_transition_coefficients : Optional[np.ndarray]
        Updated coefficients (if non-stationary).
    """

    if (
        discrete_transition_coefficients is not None
        and discrete_transition_design_matrix is not None
    ):
        (
            discrete_transition_coefficients,
            discrete_transition,
        ) = estimate_non_stationary_state_transition(
            discrete_transition_coefficients,
            discrete_transition_design_matrix,
            causal_state_probabilities,
            predictive_state_probabilities,
            discrete_transition,
            acausal_state_probabilities,
            concentration=transition_concentration,
            stickiness=transition_stickiness,
            transition_regularization=transition_regularization,
        )

    else:
        discrete_transition = estimate_stationary_state_transition(
            causal_state_probabilities,
            predictive_state_probabilities,
            discrete_transition,
            acausal_state_probabilities,
            concentration=transition_concentration,
            stickiness=transition_stickiness,
        )

    return (
        discrete_transition,
        discrete_transition_coefficients,
    )


@dataclass
class DiscreteStationaryDiagonal:
    """Diagonal values are placed on the diagonal.

    Off-diagonals are probability: (1 - `diagonal_value`) / (`n_states` - 1)

    Attributes
    ----------
    diagonal_values : np.ndarray, shape (n_states,)
        The diagonal of the transition matrix.

    """

    diagonal_values: np.ndarray

    def make_state_transition(self, *args, **kwargs) -> tuple[np.ndarray, None, None]:
        """Constructs the initial discrete transition matrix.

        Returns
        -------
        discrete_transition : np.ndarray, shape (n_states, n_states)
            The initial discrete transition matrix.
        discrete_transition_coefficients : None
            The coefficients for the non-stationary transition matrix.
            It is None here because the transition matrix is stationary.
        discrete_transition_design_matrix : None
            The design matrix for the non-stationary transition matrix.
            It is None here because the transition matrix is stationary.

        """
        diag = np.asarray(self.diagonal_values)
        return make_transition_from_diag(diag), None, None


@dataclass
class DiscreteStationaryCustom:
    """Creates a custom discrete transition matrix.


    Attributes
    ----------
    values : np.ndarray, shape (n_states, n_states)
        The transition matrix values. Rows must sum to 1.

    """

    values: np.ndarray

    def make_state_transition(self, *args, **kwargs) -> tuple[np.ndarray, None, None]:
        """Constructs the initial discrete transition matrix.

        Returns
        -------
        discrete_transition : np.ndarray, shape (n_states, n_states)
            The initial discrete transition matrix.
        discrete_transition_coefficients : None
            The coefficients for the non-stationary transition matrix.
            It is None here because the transition matrix is stationary.
        discrete_transition_design_matrix : None
            The design matrix for the non-stationary transition matrix.
            It is None here because the transition matrix is stationary.

        """
        return np.asarray(self.values), None, None


@dataclass
class DiscreteNonStationaryDiagonal:
    """Non-stationary transition matrix driven by covariates.

    Initialized with a stationary diagonal matrix, then coefficients are estimated.
    Off-diagonals are uniform based on the diagonal value at each time step.

    Attributes
    ----------
    diagonal_values : np.ndarray, shape (n_states,)
        Initial diagonal probabilities used to set intercept coefficients.
    formula : str, optional
        Patsy formula defining the relationship between covariates and transitions.
        Defaults to a spline based on 'speed'.
    """

    diagonal_values: np.ndarray
    formula: str = "1 + bs(speed, knots=[1.0, 4.0, 16.0, 32.0, 64.0])"

    def make_state_transition(
        self, covariate_data: pd.DataFrame | dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constructs the initial non-stationary discrete transition structures.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict
            Data containing covariates specified in the formula. Must have
            length matching the number of time steps.

        Returns
        -------
        initial_discrete_transition : np.ndarray, shape (n_time, n_states, n_states)
            Initial guess for the transition matrix at each time step (based on intercept).
        initial_discrete_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
            Initial coefficients, with intercepts set based on `diagonal_values`.
        discrete_transition_design_matrix : patsy.DesignMatrix
            The design matrix derived from the formula and covariate data.
        """

        n_states = len(self.diagonal_values)
        discrete_transition = make_transition_from_diag(self.diagonal_values)

        discrete_transition_design_matrix = dmatrix(self.formula, covariate_data)

        n_time, n_coefficients = discrete_transition_design_matrix.shape

        discrete_transition_coefficients = np.zeros(
            (n_coefficients, n_states, n_states - 1)
        )
        discrete_transition_coefficients[0] = centered_softmax_inverse(
            discrete_transition
        )

        discrete_transition = discrete_transition[np.newaxis] * np.ones(
            (n_time, n_states, n_states)
        )

        return (
            discrete_transition,
            discrete_transition_coefficients,
            discrete_transition_design_matrix,
        )


@dataclass
class DiscreteNonStationaryCustom:
    """Non-stationary transition matrix driven by covariates, with custom initial values.

    Initialized with a custom stationary matrix, then coefficients are estimated.

    Attributes
    ----------
    values : np.ndarray, shape (n_states, n_states)
        Initial stationary transition matrix used to set intercept coefficients.
        Rows must sum to 1.
    formula : str, optional
        Patsy formula defining the relationship between covariates and transitions.
        Defaults to a spline based on 'speed'.
    """

    values: np.ndarray
    formula: str = "1 + bs(speed, knots=[1.0, 4.0, 16.0, 32.0, 64.0])"

    def make_state_transition(
        self, covariate_data: tuple[pd.DataFrame, dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constructs the initial non-stationary discrete transition structures.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict
            Data containing covariates specified in the formula. Must have
            length matching the number of time steps.

        Returns
        -------
        initial_discrete_transition : np.ndarray, shape (n_time, n_states, n_states)
            Initial guess for the transition matrix at each time step (based on intercept).
        initial_discrete_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
            Initial coefficients, with intercepts set based on `values`.
        discrete_transition_design_matrix : patsy.DesignMatrix
            The design matrix derived from the formula and covariate data.
        """

        n_states = len(self.values)
        discrete_transition = self.values

        discrete_transition_design_matrix = dmatrix(self.formula, covariate_data)
        if discrete_transition_design_matrix.shape[0] == 0:
            raise ValueError(
                "No covariate data provided for transition matrix or NaNs are present in the covariate data."
            )

        n_time, n_coefficients = discrete_transition_design_matrix.shape

        discrete_transition_coefficients = np.zeros(
            (n_coefficients, n_states, n_states - 1)
        )
        discrete_transition_coefficients[0] = centered_softmax_inverse(
            discrete_transition
        )

        discrete_transition = discrete_transition[np.newaxis] * np.ones(
            (n_time, n_states, n_states)
        )

        return (
            discrete_transition,
            discrete_transition_coefficients,
            discrete_transition_design_matrix,
        )


def predict_discrete_state_transitions(
    discrete_transition_design_matrix: DesignMatrix,
    discrete_transition_coefficients: np.ndarray,
    discrete_transition_covariate_data: pd.DataFrame | dict,
) -> np.ndarray:
    """Predict the discrete state transitions based on new covariate data.

    Parameters
    ----------
    discrete_transition_design_matrix : patsy.DesignMatrix
        Original design matrix used for fitting (contains design_info).
    discrete_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Fitted regression coefficients.
    discrete_transition_covariate_data : pd.DataFrame or dict
        New covariate data for prediction.

    Returns
    -------
    discrete_state_transitions : np.ndarray, shape (n_new_time, n_states, n_states)
        Predicted transition matrices for the new covariate data.
    """
    design_matrix = build_design_matrices(
        [discrete_transition_design_matrix.design_info],
        discrete_transition_covariate_data,
    )[0]

    n_time = design_matrix.shape[0]
    n_states = discrete_transition_coefficients.shape[1]

    discrete_state_transitions = jnp.zeros((n_time, n_states, n_states))
    for from_state in range(n_states):
        discrete_state_transitions[:, from_state, :] = jnp.exp(
            jax_centered_log_softmax_forward(
                design_matrix @ discrete_transition_coefficients[:, from_state, :]
            )
        )
    return discrete_state_transitions
