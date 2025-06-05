from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.nn import log_softmax
from jax.scipy.special import logsumexp
from patsy import DesignMatrix, build_design_matrices, dmatrix
from scipy.optimize import minimize
from scipy.special import softmax

from .priors import get_dirichlet_prior


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
    EPS = 1e-16
    relative_distribution = np.where(
        np.isclose(predictive_distribution[1:], 0.0),
        EPS,
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


def estimate_non_stationary_state_transition(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    concentration: float = 1.0,
    stickiness: Union[float, np.ndarray] = 0.0,
    transition_regularization: float = 1e-5,
    optimization_method: str = "Newton-CG",
    maxiter: Optional[int] = 100,
    disp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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

    alpha = get_dirichlet_prior(concentration, stickiness, n_states)

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
    EPS = 1e-16
    estimated_transition_matrix = np.clip(estimated_transition_matrix, EPS, 1.0 - EPS)

    return estimated_transition_coefficients, estimated_transition_matrix


def estimate_stationary_state_transition(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    concentration: float = 1.0,
    stickiness: float = 0.0,
) -> np.ndarray:
    """Estimate the stationary state transition model.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
    predictive_distribution : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)
    acausal_posterior : np.ndarray, shape (n_time, n_states)
    concentration : float, optional
    stickiness : float, optional

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
    alpha = get_dirichlet_prior(concentration, stickiness, n_states)

    new_transition_matrix = joint_distribution.sum(axis=0) + alpha - 1.0
    new_transition_matrix /= new_transition_matrix.sum(axis=-1, keepdims=True)

    # if any is zero, set to small number
    EPS = 1e-16
    new_transition_matrix = np.clip(new_transition_matrix, a_min=EPS, a_max=1.0 - EPS)

    return new_transition_matrix


@jax.jit
def dirichlet_neg_log_likelihood(
    coefficients: jnp.ndarray,
    design_matrix: jnp.ndarray,
    response: jnp.ndarray,
    alpha: Union[float, jnp.ndarray] = 1.0,
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


def predict_discrete_state_transitions(
    fitted_design_matrix: DesignMatrix,
    coefficients: np.ndarray,  # (n_features, n_states, n_states-1)
    covariate_data: Union[pd.DataFrame, dict],
) -> jnp.ndarray:
    """
    Generate time-varying discrete-state transition tensors from new covariates.

    Parameters
    ----------
    fitted_design_matrix : patsy.DesignMatrix, shape (n_time, n_features)
        The *original* patsy DesignMatrix used during fitting (only its
        ``design_info`` is accessed, so a slice works fine).
    coefficients : np.ndarray, shape (n_features, n_states, n_states - 1)
        Regression coefficients with shape
        ``(n_features, n_states, n_states - 1)``.  The last destination
        category in each row is the reference class.
    covariate_data : Union[pd.DataFrame, dict]
        New covariates—either a pandas DataFrame or a
        ``{column_name: ndarray}`` dictionary.

    Returns
    -------
    transition_tensor : jnp.ndarray, shape (n_time, n_states, n_states)
    """
    design_matrix = build_design_matrices(
        [fitted_design_matrix.design_info],
        covariate_data,
    )[0]

    # tensordot → (n_time, n_states, n_states - 1)
    linear_term = jnp.tensordot(
        design_matrix,  # (n_time, n_features)
        coefficients,  # (n_features, n_states, n_states - 1)
        axes=([1], [0]),
    )

    # Add reference category (implicitly zero) and soft-max
    #  logits_full: (n_time, n_states, n_states)
    logits_full = jnp.concatenate(
        [linear_term, jnp.zeros((*linear_term.shape[:2], 1), dtype=linear_term.dtype)],
        axis=-1,
    )

    log_probabilities = logits_full - logsumexp(logits_full, axis=-1, keepdims=True)

    return jnp.exp(log_probabilities)
