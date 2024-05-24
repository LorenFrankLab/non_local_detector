from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


def convert_to_state_probability(
    causal_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    state_ind: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the causal, acausal, and predictive distributions to state probabilities.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        The causal posterior distribution.
    acausal_posterior : np.ndarray, shape (n_time, n_states_bins)
        The acausal posterior distribution.
    predictive_distribution : np.ndarray, shape (n_time, n_state_bins)
        The predictive distribution.
    state_ind : np.ndarray, shape (n_time,)
        The state indices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the causal, acausal, and predictive state probabilities.
        causal_state_probabilities : np.ndarray, shape (n_time, n_states)
            The causal state probabilities.
        acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
            The acausal state probabilities.
        predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
            The predictive state probabilities.
    """
    n_states = np.unique(state_ind).size
    n_time = causal_posterior.shape[0]

    causal_state_probabilities = np.zeros((n_time, n_states))
    acausal_state_probabilities = np.zeros((n_time, n_states))
    predictive_state_probabilities = np.zeros((n_time, n_states))

    acausal_posterior = np.asarray(acausal_posterior)
    causal_posterior = np.asarray(causal_posterior)
    predictive_distribution = np.asarray(predictive_distribution)

    for ind in range(n_states):
        is_state = state_ind == ind
        causal_state_probabilities[:, ind] = causal_posterior[:, is_state].sum(axis=1)
        acausal_state_probabilities[:, ind] = acausal_posterior[:, is_state].sum(axis=1)
        predictive_state_probabilities[:, ind] = predictive_distribution[
            :, is_state
        ].sum(axis=1)

    return (
        causal_state_probabilities,
        acausal_state_probabilities,
        predictive_state_probabilities,
    )


## NOTE: adapted from dynamax: https://github.com/probml/dynamax/ with modifications ##
def _normalize(u, axis=0, eps=1e-15):
    """Normalizes the values within the axis in a way that they sum up to 1.

    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.

    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = jnp.clip(u, a_min=eps, a_max=None)
    c = u.sum(axis=axis)
    return u / c, c


# Helper functions for the two key filtering steps
def _condition_on(probs, ll):
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.

    Args:
        probs(k): prior for state k
        ll(k): log likelihood for state k

    Returns:
        probs(k): posterior for state k
    """
    ll_max = ll.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    new_probs, norm = _normalize(probs * jnp.exp(ll - ll_max))
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _divide_safe(a, b):
    """Divides two arrays, while setting the result to 0.0
    if the denominator is 0.0."""
    return jnp.where(b == 0.0, 0.0, a / b)


@jax.jit
def filter(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
):
    def _step(carry, ll):
        log_normalizer, predicted_probs = carry

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = filtered_probs @ transition_matrix

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    (log_normalizer, _), (filtered_probs, predicted_probs) = jax.lax.scan(
        _step, (0.0, initial_distribution), log_likelihoods
    )

    return (
        log_normalizer,
        filtered_probs,
        predicted_probs,
    )


@jax.jit
def smoother(
    transition_matrix,
    filtered_probs,
):
    n_time = filtered_probs.shape[0]

    def _step(smoothed_probs_next, args):
        filtered_probs_t, t = args

        smoothed_probs = filtered_probs_t * (
            transition_matrix
            @ _divide_safe(smoothed_probs_next, filtered_probs_t @ transition_matrix)
        ) * (t < n_time - 1) + filtered_probs_t * (t == n_time - 1)
        smoothed_probs /= smoothed_probs.sum(keepdims=True)

        return smoothed_probs, smoothed_probs

    _, smoothed_probs = jax.lax.scan(
        _step,
        filtered_probs[-1],
        (filtered_probs, jnp.arange(n_time)),
        reverse=True,
    )

    return smoothed_probs


def filter_smoother(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
):
    (
        marginal_loglik,
        filtered_probs,
        predicted_probs,
    ) = filter(
        initial_distribution,
        transition_matrix,
        log_likelihoods,
    )

    smoothed_probs = smoother(
        transition_matrix,
        filtered_probs,
    )

    return (
        marginal_loglik,
        filtered_probs,
        predicted_probs,
        smoothed_probs,
    )


def _get_transition_matrix(
    discrete_transition_matrix_t,
    continuous_transition_matrix,
    state_ind,
):
    return (
        continuous_transition_matrix
        * discrete_transition_matrix_t[jnp.ix_(state_ind, state_ind)]
    )


@jax.jit
def filter_covariate_dependent(
    initial_distribution,
    discrete_transition_matrix,
    continuous_transition_matrix,
    state_ind,
    log_likelihoods,
):
    def _step(carry, args):
        log_normalizer, predicted_probs = carry
        discrete_transition_matrix_t, ll = args

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = filtered_probs @ _get_transition_matrix(
            discrete_transition_matrix_t, continuous_transition_matrix, state_ind
        )

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    (log_normalizer, _), (filtered_probs, predicted_probs) = jax.lax.scan(
        _step,
        (0.0, initial_distribution),
        (log_likelihoods, discrete_transition_matrix),
    )

    return (
        log_normalizer,
        filtered_probs,
        predicted_probs,
    )


@jax.jit
def smoother_covariate_dependent(
    discrete_transition_matrix,
    continuous_transition_matrix,
    state_ind,
    filtered_probs,
):
    n_time = filtered_probs.shape[0]

    def _step(smoothed_probs_next, args):
        filtered_probs_t, discrete_transition_matrix_t, t = args

        smoothed_probs = filtered_probs_t * (
            _get_transition_matrix(
                discrete_transition_matrix_t, continuous_transition_matrix, state_ind
            )
            @ _divide_safe(
                smoothed_probs_next,
                filtered_probs_t
                @ _get_transition_matrix(
                    discrete_transition_matrix_t,
                    continuous_transition_matrix,
                    state_ind,
                ),
            )
        ) * (t < n_time - 1) + filtered_probs_t * (t == n_time - 1)
        smoothed_probs /= smoothed_probs.sum(keepdims=True)

        return smoothed_probs, smoothed_probs

    _, smoothed_probs = jax.lax.scan(
        _step,
        filtered_probs[-1],
        (filtered_probs, discrete_transition_matrix, jnp.arange(n_time)),
        reverse=True,
    )

    return smoothed_probs


def filter_smoother_covariate_dependent(
    initial_distribution,
    discrete_transition_matrix,
    continuous_transition_matrix,
    state_ind,
    log_likelihoods,
):
    (
        marginal_loglik,
        filtered_probs,
        predicted_probs,
    ) = filter(
        initial_distribution,
        discrete_transition_matrix,
        continuous_transition_matrix,
        state_ind,
        log_likelihoods,
    )

    smoothed_probs = smoother(
        discrete_transition_matrix,
        continuous_transition_matrix,
        state_ind,
        filtered_probs,
    )

    return (
        marginal_loglik,
        filtered_probs,
        predicted_probs,
        smoothed_probs,
    )


def check_converged(
    log_likelihood: np.ndarray,
    previous_log_likelihood: np.ndarray,
    tolerance: float = 1e-4,
) -> Tuple[bool, bool]:
    """We have converged if the slope of the log-likelihood function falls below 'tolerance',

    i.e., |f(t) - f(t-1)| / avg < tolerance,
    where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.

    Parameters
    ----------
    log_likelihood : np.ndarray
        Current log likelihood
    previous_log_likelihood : np.ndarray
        Previous log likelihood
    tolerance : float, optional
        threshold for similarity, by default 1e-4

    Returns
    -------
    is_converged : bool
    is_increasing : bool

    """
    delta_log_likelihood = abs(log_likelihood - previous_log_likelihood)
    avg_log_likelihood = (
        abs(log_likelihood) + abs(previous_log_likelihood) + np.spacing(1)
    ) / 2

    is_increasing = log_likelihood - previous_log_likelihood >= -1e-3
    is_converged = (delta_log_likelihood / avg_log_likelihood) < tolerance

    return is_converged, is_increasing
