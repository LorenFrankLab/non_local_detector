import logging
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from jax.nn import log_softmax
from patsy import (  # type: ignore[import-untyped]
    DesignMatrix,
    build_design_matrices,
    dmatrix,
)
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import softmax  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


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
    EPS = np.finfo(y.dtype).tiny
    y_safe = np.clip(y, EPS, None)
    return np.log(y_safe[..., :-1]) - np.log(y_safe[..., [-1]])


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
    pred = predictive_distribution[1:]
    safe_pred = np.where(np.isclose(pred, 0.0), 1.0, pred)
    relative_distribution = np.where(
        np.isclose(pred, 0.0),
        0.0,
        acausal_posterior[1:] / safe_pred,
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


def _state_aggregation_matrix(state_ind: np.ndarray) -> np.ndarray:
    """Return a bin-to-discrete-state aggregation matrix."""
    state_ind = np.asarray(state_ind, dtype=int)
    n_state_bins = state_ind.size
    n_states = int(state_ind.max()) + 1
    aggregation = np.zeros((n_state_bins, n_states))
    aggregation[np.arange(n_state_bins), state_ind] = 1.0
    return aggregation


def _n_states_from_state_ind(state_ind: np.ndarray) -> int:
    """Return the number of discrete states represented by expanded bins."""
    return int(np.max(state_ind)) + 1


def _aggregate_xi_by_state_jax(
    xi: jnp.ndarray, state_ind: jnp.ndarray, n_states: int
) -> jnp.ndarray:
    """Aggregate expanded-bin pair probabilities to discrete-state pairs."""
    # Equivalent to A.T @ xi @ A without materializing the bin-to-state matrix A.
    target_sum = jax.ops.segment_sum(
        xi.T,
        state_ind,
        num_segments=n_states,
    ).T
    return jax.ops.segment_sum(
        target_sum,
        state_ind,
        num_segments=n_states,
    )


def _aggregate_factorized_xi_by_state_jax(
    causal_t: jnp.ndarray,
    ratio: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    n_states: int,
) -> jnp.ndarray:
    """Aggregate factorized expanded-bin pair probabilities by state."""
    # For each source bin k and target state q, sum C[k, l] * ratio[l]
    # over target bins l in q. This avoids forming
    # C[k, l] * D[state(k), state(l)] or xi[k, l].
    target_sum = jax.ops.segment_sum(
        (continuous_transition_matrix * ratio[jnp.newaxis, :]).T,
        state_ind,
        num_segments=n_states,
    ).T
    source_sum = jax.ops.segment_sum(
        causal_t[:, jnp.newaxis] * target_sum,
        state_ind,
        num_segments=n_states,
    )
    return source_sum * discrete_transition_matrix


def _safe_ratio_jax(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Return numerator / denominator with zero output near zero denominators."""
    is_zero = jnp.isclose(denominator, 0.0)
    safe_denominator = jnp.where(is_zero, 1.0, denominator)
    return jnp.where(is_zero, 0.0, numerator / safe_denominator)


def _assert_map_compatible_alpha(alpha: np.ndarray) -> None:
    if np.any(alpha < 1.0):
        raise ValueError(
            "concentration and stickiness must produce prior parameters >= 1.0 "
            "for this MAP transition update"
        )


@partial(
    jax.jit,
    static_argnames=(
        "n_states",
        "is_factorized",
        "is_stationary",
        "return_time_series",
    ),
)
def _transition_pair_stats_jax(
    causal_posterior: jnp.ndarray,
    predictive_posterior: jnp.ndarray,
    acausal_posterior: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    n_states: int,
    *,
    is_factorized: bool,
    is_stationary: bool,
    return_time_series: bool,
) -> jnp.ndarray:
    """JAX scan kernel for exact discrete transition responses or counts."""

    def aggregate(causal_t, predictive_next, acausal_next, transition_t):
        ratio = _safe_ratio_jax(acausal_next, predictive_next)
        if is_factorized:
            return _aggregate_factorized_xi_by_state_jax(
                causal_t,
                ratio,
                continuous_transition_matrix,
                transition_t,
                state_ind,
                n_states,
            )

        xi = causal_t[:, jnp.newaxis] * transition_t * ratio[jnp.newaxis, :]
        return _aggregate_xi_by_state_jax(xi, state_ind, n_states)

    if return_time_series:
        if is_stationary:

            def step(_, inputs):
                causal_t, predictive_next, acausal_next = inputs
                return (
                    None,
                    aggregate(
                        causal_t,
                        predictive_next,
                        acausal_next,
                        transition_matrix,
                    ),
                )

            _, response = jax.lax.scan(
                step,
                None,
                (
                    causal_posterior[:-1],
                    predictive_posterior[1:],
                    acausal_posterior[1:],
                ),
            )
        else:

            def step(_, inputs):
                causal_t, predictive_next, acausal_next, transition_t = inputs
                return (
                    None,
                    aggregate(
                        causal_t,
                        predictive_next,
                        acausal_next,
                        transition_t,
                    ),
                )

            _, response = jax.lax.scan(
                step,
                None,
                (
                    causal_posterior[:-1],
                    predictive_posterior[1:],
                    acausal_posterior[1:],
                    transition_matrix[:-1],
                ),
            )

        return response

    initial_counts = jnp.zeros(
        (n_states, n_states),
        dtype=jnp.result_type(
            causal_posterior,
            predictive_posterior,
            acausal_posterior,
            transition_matrix,
            continuous_transition_matrix,
        ),
    )

    if is_stationary:

        def step(joint_sum, inputs):
            causal_t, predictive_next, acausal_next = inputs
            counts_t = aggregate(
                causal_t,
                predictive_next,
                acausal_next,
                transition_matrix,
            )
            return joint_sum + counts_t, None

        joint_sum, _ = jax.lax.scan(
            step,
            initial_counts,
            (
                causal_posterior[:-1],
                predictive_posterior[1:],
                acausal_posterior[1:],
            ),
        )
    else:

        def step(joint_sum, inputs):
            causal_t, predictive_next, acausal_next, transition_t = inputs
            counts_t = aggregate(
                causal_t,
                predictive_next,
                acausal_next,
                transition_t,
            )
            return joint_sum + counts_t, None

        joint_sum, _ = jax.lax.scan(
            step,
            initial_counts,
            (
                causal_posterior[:-1],
                predictive_posterior[1:],
                acausal_posterior[1:],
                transition_matrix[:-1],
            ),
        )

    return joint_sum


def estimate_discrete_transition_responses_from_expanded_posteriors(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Return exact expected discrete transition responses.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Filtered posterior over expanded state bins.
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        One-step predictive posterior over expanded state bins.
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Smoothed posterior over expanded state bins.
    transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins) or
        (n_time, n_state_bins, n_state_bins)
        Full expanded transition matrix used by filtering.
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.

    Returns
    -------
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Exact expected transition counts from each discrete state to each
        discrete state at each time.
    """
    state_ind = np.asarray(state_ind, dtype=int)
    n_states = _n_states_from_state_ind(state_ind)
    transition_matrix = jnp.asarray(transition_matrix)
    response = _transition_pair_stats_jax(
        jnp.asarray(causal_posterior),
        jnp.asarray(predictive_posterior),
        jnp.asarray(acausal_posterior),
        transition_matrix,
        jnp.empty((0, 0), dtype=transition_matrix.dtype),
        jnp.asarray(state_ind),
        n_states,
        is_factorized=False,
        is_stationary=transition_matrix.ndim == 2,
        return_time_series=True,
    )

    return np.asarray(response)


def estimate_discrete_transition_counts_from_expanded_posteriors(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Return exact expected discrete transition counts.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Filtered posterior over expanded state bins.
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        One-step predictive posterior over expanded state bins.
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Smoothed posterior over expanded state bins.
    transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins) or
        (n_time, n_state_bins, n_state_bins)
        Full expanded transition matrix used by filtering.
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.

    Returns
    -------
    joint_sum : np.ndarray, shape (n_states, n_states)
        Exact expected transition counts from each discrete state to each
        discrete state.
    """
    state_ind = np.asarray(state_ind, dtype=int)
    n_states = _n_states_from_state_ind(state_ind)
    transition_matrix = jnp.asarray(transition_matrix)
    joint_sum = _transition_pair_stats_jax(
        jnp.asarray(causal_posterior),
        jnp.asarray(predictive_posterior),
        jnp.asarray(acausal_posterior),
        transition_matrix,
        jnp.empty((0, 0), dtype=transition_matrix.dtype),
        jnp.asarray(state_ind),
        n_states,
        is_factorized=False,
        is_stationary=transition_matrix.ndim == 2,
        return_time_series=False,
    )

    return np.asarray(joint_sum)


def estimate_discrete_transition_responses_from_factorized_posteriors(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Return exact discrete transition responses from factorized transitions.

    This is the streaming equivalent of first constructing
    ``continuous_transition_matrix * discrete_transition_matrix[:, state_ind, state_ind]``
    and then calling
    `estimate_discrete_transition_responses_from_expanded_posteriors`.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Filtered posterior over expanded state bins.
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        One-step predictive posterior over expanded state bins.
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Smoothed posterior over expanded state bins.
    continuous_transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
        Continuous transition matrix over expanded state bins.
    discrete_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Time-varying discrete transition matrix.
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.

    Returns
    -------
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Exact expected transition counts from each discrete state to each
        discrete state at each time.
    """
    state_ind = np.asarray(state_ind, dtype=int)
    n_states = _n_states_from_state_ind(state_ind)
    continuous_transition_matrix = jnp.asarray(continuous_transition_matrix)
    response = _transition_pair_stats_jax(
        jnp.asarray(causal_posterior),
        jnp.asarray(predictive_posterior),
        jnp.asarray(acausal_posterior),
        jnp.asarray(discrete_transition_matrix),
        continuous_transition_matrix,
        jnp.asarray(state_ind),
        n_states,
        is_factorized=True,
        is_stationary=False,
        return_time_series=True,
    )
    return np.asarray(response)


def estimate_discrete_transition_counts_from_factorized_posteriors(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Return exact counts from stationary factorized transitions.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Filtered posterior over expanded state bins.
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        One-step predictive posterior over expanded state bins.
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Smoothed posterior over expanded state bins.
    continuous_transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
        Stationary continuous transition matrix over expanded state bins.
    discrete_transition_matrix : np.ndarray, shape (n_states, n_states)
        Stationary discrete transition matrix.
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.

    Returns
    -------
    joint_sum : np.ndarray, shape (n_states, n_states)
        Exact expected transition counts from each discrete state to each
        discrete state.

    Raises
    ------
    ValueError
        If ``discrete_transition_matrix`` is not stationary with shape
        ``(n_states, n_states)``.
    """
    if discrete_transition_matrix.ndim != 2:
        raise ValueError(
            "discrete_transition_matrix must be stationary with shape (n_states, n_states)."
        )

    state_ind = np.asarray(state_ind, dtype=int)
    n_states = _n_states_from_state_ind(state_ind)
    continuous_transition_matrix = jnp.asarray(continuous_transition_matrix)
    joint_sum = _transition_pair_stats_jax(
        jnp.asarray(causal_posterior),
        jnp.asarray(predictive_posterior),
        jnp.asarray(acausal_posterior),
        jnp.asarray(discrete_transition_matrix),
        continuous_transition_matrix,
        jnp.asarray(state_ind),
        n_states,
        is_factorized=True,
        is_stationary=True,
        return_time_series=False,
    )
    return np.asarray(joint_sum)


@jax.jit
def jax_centered_log_softmax_forward(y: jnp.ndarray) -> jnp.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    The 1D and 2D input branches compile as separate JAX specializations.

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
    """Creates a Dirichlet prior for the transition matrix rows.

    Constructs prior parameters for a Dirichlet distribution over transition
    matrix rows, incorporating base concentration and state-specific stickiness.

    Parameters
    ----------
    concentration : float
        Base concentration parameter for the Dirichlet prior.
    stickiness : float or np.ndarray, shape (n_states,)
        Stickiness parameter(s) for diagonal entries. If float, same value
        is applied to all states. If array, per-state stickiness values.
    n_states : int
        Number of discrete states.

    Returns
    -------
    prior_params : np.ndarray, shape (n_states, n_states)
        Dirichlet prior parameters for each transition matrix row.
    """
    if isinstance(stickiness, int | float):
        stickiness_arr = stickiness * np.eye(n_states)
    else:
        # Assume stickiness provided per state
        stickiness_arr = np.diag(stickiness)
    # Dirichlet parameters must be strictly positive. MAP transition estimators
    # that add alpha - 1 as pseudo-counts separately reject alpha < 1.
    return np.maximum(concentration * np.ones((n_states,)) + stickiness_arr, 1e-10)


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

    return estimate_non_stationary_state_transition_from_responses(
        transition_coefficients,
        design_matrix,
        joint_distribution,
        concentration=concentration,
        stickiness=stickiness,
        transition_regularization=transition_regularization,
        optimization_method=optimization_method,
        maxiter=maxiter,
        disp=disp,
    )


def estimate_non_stationary_state_transition_from_responses(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 1e-5,
    optimization_method: str = "Newton-CG",
    maxiter: int | None = 100,
    disp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a non-stationary transition model from expected counts.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Initial estimate of the transition coefficients.
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Expected transition counts for each time and transition row.
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
    n_coefficients, n_states = transition_coefficients.shape[:2]
    estimated_transition_coefficients = np.zeros(
        (n_coefficients, n_states, (n_states - 1))
    )

    n_time = design_matrix.shape[0]
    estimated_transition_matrix = np.zeros((n_time, n_states, n_states))

    alpha = get_transition_prior(concentration, stickiness, n_states)
    _assert_map_compatible_alpha(alpha)

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
                response[:, from_state, :],
                row_alpha,
                transition_regularization,
            ),
            options={"disp": disp, "maxiter": maxiter},
        )

        if not result.success:
            logger.warning(
                "Transition optimization did not converge for state %d: %s. "
                "Keeping previous coefficients.",
                from_state,
                result.message,
            )
            # Keep previous coefficients for this row
            estimated_transition_coefficients[:, from_state, :] = (
                transition_coefficients[:, from_state, :]
            )
        else:
            estimated_transition_coefficients[:, from_state, :] = result.x.reshape(
                (n_coefficients, n_states - 1)
            )

        linear_predictor = (
            design_matrix @ estimated_transition_coefficients[:, from_state, :]
        )
        estimated_transition_matrix[:, from_state, :] = jnp.exp(
            jax_centered_log_softmax_forward(linear_predictor)
        )

    return estimated_transition_coefficients, estimated_transition_matrix


def estimate_stationary_state_transition(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    stickiness: float | np.ndarray = 0.0,
    concentration: float = 1.0,
    prior_weight: float | np.ndarray = 0.0,
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
    prior_weight : float or np.ndarray, shape (n_states,), optional
        Dimensionless weight for data-adaptive prior scaling. When > 0,
        the effective prior pseudo-counts for that row are scaled by the
        expected transition count N_i, making the prior influence
        approximately invariant to the number of time bins.

        Can be a scalar (same weight for all rows) or an array with one
        value per state. Rows with ``prior_weight[i] == 0`` use the legacy
        fixed-count prior from ``concentration`` and ``stickiness``
        directly. This allows mixing frozen rows (e.g., structural states
        with very high stickiness) with data-adaptive rows in the same
        call. When 0.0 (default), all rows use legacy behavior.

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

    joint_sum = joint_distribution.sum(axis=0)  # (n_states, n_states)

    return estimate_stationary_state_transition_from_counts(
        joint_sum,
        stickiness=stickiness,
        concentration=concentration,
        prior_weight=prior_weight,
    )


def estimate_stationary_state_transition_from_counts(
    joint_sum: np.ndarray,
    stickiness: float | np.ndarray = 0.0,
    concentration: float = 1.0,
    prior_weight: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Estimate a stationary transition matrix from expected transition counts.

    Parameters
    ----------
    joint_sum : np.ndarray, shape (n_states, n_states)
        Expected transition counts from each source state to each target state.
    stickiness : float, optional
        Diagonal stickiness parameter, by default 0.0. If array-like, one
        stickiness value per state.
    concentration : float, optional
        Dirichlet prior concentration parameter, by default 1.0.
    prior_weight : float or np.ndarray, shape (n_states,), optional
        Dimensionless data-adaptive prior weight. See
        `estimate_stationary_state_transition`.

    Returns
    -------
    new_transition_matrix : np.ndarray, shape (n_states, n_states)
    """
    n_states = joint_sum.shape[0]

    # Normalize prior_weight to shape (n_states,)
    prior_weight_arr = np.atleast_1d(np.asarray(prior_weight, dtype=float))
    if prior_weight_arr.size == 1:
        prior_weight_arr = np.full(n_states, prior_weight_arr.item())
    if prior_weight_arr.shape != (n_states,):
        raise ValueError(
            f"prior_weight must be a scalar or array of shape ({n_states},), "
            f"got shape {prior_weight_arr.shape}"
        )
    if np.any(prior_weight_arr < 0):
        raise ValueError(f"prior_weight must be non-negative, got {prior_weight_arr}")

    alpha = get_transition_prior(concentration, stickiness, n_states)
    _assert_map_compatible_alpha(alpha)

    # Legacy prior for rows with prior_weight[i] == 0
    legacy_prior = alpha - 1.0  # (n_states, n_states)

    if np.any(prior_weight_arr > 0):
        # Data-adaptive prior: scale pseudo-counts by expected transitions
        # so regularization strength is invariant to temporal resolution.
        alpha_shape = alpha - 1.0  # (n_states, n_states)

        # Normalize to get prior direction per row
        row_sums = alpha_shape.sum(axis=-1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
        tilde_alpha = np.where(row_sums == 0, 0.0, alpha_shape / safe_row_sums)

        # N_i = expected transitions from state i
        N_i = joint_sum.sum(axis=-1, keepdims=True)  # (n_states, 1)

        # Per-row adaptive prior: prior_weight[i] * N_i * tilde_alpha[i]
        adaptive_prior = prior_weight_arr[:, np.newaxis] * N_i * tilde_alpha

        # Use adaptive prior for rows where prior_weight > 0, legacy otherwise
        use_adaptive = prior_weight_arr > 0  # (n_states,)
        effective_prior = np.where(
            use_adaptive[:, np.newaxis],
            adaptive_prior,
            legacy_prior,
        )
    else:
        # All rows use legacy fixed-count prior
        effective_prior = legacy_prior

    new_transition_matrix = joint_sum + effective_prior

    # Normalize rows to get transition probabilities.
    # Guard against all-zero rows (e.g., unvisited states) to avoid NaN.
    # Fall back to normalized Dirichlet prior direction for unvisited states,
    # preserving stickiness semantics. If prior is also uniform, this gives 1/n.
    prior_fallback = alpha / alpha.sum(axis=-1, keepdims=True)
    row_totals = new_transition_matrix.sum(axis=-1, keepdims=True)
    safe_row_totals = np.where(row_totals == 0, 1.0, row_totals)
    new_transition_matrix = np.where(
        row_totals == 0,
        prior_fallback,
        new_transition_matrix / safe_row_totals,
    )

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
    alpha : float | jnp.ndarray, shape (n_states,), optional
        Dirichlet prior parameters for this row of the transition matrix.
        If float, assumed uniform. Defaults to 1.0 (no prior effect).
        The non-stationary objective adds ``alpha - 1`` to each time sample's
        response before averaging over samples. For stationary transitions,
        the estimator instead adds ``alpha - 1`` once to the summed joint
        distribution.
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

    # Dirichlet prior as a per-time response offset. The objective is averaged
    # over samples, so this behaves as per-time regularization rather than the
    # stationary estimator's once-per-summed-row pseudo-count.
    n_samples = response.shape[0]
    prior = alpha - 1.0

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
    discrete_transition_coefficients : np.ndarray | None, shape (n_coefficients, n_states, n_states - 1)
        Initial coefficients (only if non-stationary).
    discrete_transition_design_matrix : patsy.DesignMatrix | None
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
    transition_prior_weight: float | np.ndarray = 0.0,
    causal_posterior: np.ndarray | None = None,
    predictive_posterior: np.ndarray | None = None,
    acausal_posterior: np.ndarray | None = None,
    continuous_transition: np.ndarray | None = None,
    state_ind: np.ndarray | None = None,
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
    discrete_transition_coefficients : np.ndarray | None, shape (n_coefficients, n_states, n_states - 1)
        Current coefficient estimate (if non-stationary).
    discrete_transition_design_matrix : patsy.DesignMatrix | None
        Design matrix (if non-stationary).
    transition_concentration : float
        Dirichlet prior concentration.
    transition_stickiness : float or np.ndarray
        Dirichlet prior stickiness.
    transition_regularization : float
        L2 penalty for non-stationary coefficients.
    transition_prior_weight : float, optional
        Data-adaptive prior weight for stationary transitions. When > 0,
        pseudo-counts scale with expected transition counts. Only applies
        to the stationary path. By default 0.0.
    causal_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
        Expanded-bin filtered posterior. If supplied with the other expanded
        arguments, the M-step uses exact expanded-state transition counts.
    predictive_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
        Expanded-bin one-step predictive posterior.
    acausal_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
        Expanded-bin smoothed posterior.
    continuous_transition : np.ndarray, optional, shape (n_state_bins, n_state_bins)
        Continuous transition matrix over expanded state bins.
    state_ind : np.ndarray, optional, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.

    Returns
    -------
    estimated_discrete_transition : np.ndarray
        Updated transition matrix.
    estimated_discrete_transition_coefficients : np.ndarray | None
        Updated coefficients (if non-stationary).
    """
    use_expanded_counts = all(
        arg is not None
        for arg in (
            causal_posterior,
            predictive_posterior,
            acausal_posterior,
            continuous_transition,
            state_ind,
        )
    )

    if (
        discrete_transition_coefficients is not None
        and discrete_transition_design_matrix is not None
    ):
        if use_expanded_counts:
            response = (
                estimate_discrete_transition_responses_from_factorized_posteriors(
                    causal_posterior,
                    predictive_posterior,
                    acausal_posterior,
                    continuous_transition,
                    discrete_transition,
                    state_ind,
                )
            )
            (
                discrete_transition_coefficients,
                discrete_transition,
            ) = estimate_non_stationary_state_transition_from_responses(
                discrete_transition_coefficients,
                discrete_transition_design_matrix,
                response,
                concentration=transition_concentration,
                stickiness=transition_stickiness,
                transition_regularization=transition_regularization,
            )
        else:
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
        # Convert stickiness to float if needed
        if (
            isinstance(transition_stickiness, np.ndarray)
            and transition_stickiness.size == 1
        ):
            stickiness_value = float(transition_stickiness.item())
        else:
            stickiness_value = transition_stickiness

        if use_expanded_counts:
            joint_sum = estimate_discrete_transition_counts_from_factorized_posteriors(
                causal_posterior,
                predictive_posterior,
                acausal_posterior,
                continuous_transition,
                discrete_transition,
                state_ind,
            )
            discrete_transition = estimate_stationary_state_transition_from_counts(
                joint_sum,
                concentration=transition_concentration,
                stickiness=stickiness_value,
                prior_weight=transition_prior_weight,
            )
        else:
            discrete_transition = estimate_stationary_state_transition(
                causal_state_probabilities,
                predictive_state_probabilities,
                discrete_transition,
                acausal_state_probabilities,
                concentration=transition_concentration,
                stickiness=stickiness_value,
                prior_weight=transition_prior_weight,
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
    ) -> tuple[np.ndarray, np.ndarray, DesignMatrix]:
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
        self, covariate_data: pd.DataFrame | dict | None
    ) -> tuple[np.ndarray, np.ndarray, DesignMatrix]:
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

    n_states = discrete_transition_coefficients.shape[1]

    rows = []
    for from_state in range(n_states):
        rows.append(
            jnp.exp(
                jax_centered_log_softmax_forward(
                    design_matrix @ discrete_transition_coefficients[:, from_state, :]
                )
            )
        )
    # rows[i] has shape (n_time, n_states), stack along axis=1 to get (n_time, n_states, n_states)
    return jnp.stack(rows, axis=1)
