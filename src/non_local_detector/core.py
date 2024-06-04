from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


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

    return jax.lax.scan(_step, (0.0, initial_distribution), log_likelihoods)


@jax.jit
def smoother(
    transition_matrix,
    filtered_probs,
    initial=None,
    ind=None,
    n_time=None,
):
    if n_time is None:
        n_time = filtered_probs.shape[0]
    if ind is None:
        ind = jnp.arange(n_time)
    if initial is None:
        initial = filtered_probs[-1]

    def _step(smoothed_probs_next, args):
        filtered_probs_t, t = args

        smoothed_probs = filtered_probs_t * (
            transition_matrix
            @ _divide_safe(smoothed_probs_next, filtered_probs_t @ transition_matrix)
        ) * (t < n_time - 1) + filtered_probs_t * (t == n_time - 1)
        smoothed_probs /= smoothed_probs.sum(keepdims=True)

        return smoothed_probs, smoothed_probs

    return jax.lax.scan(
        _step,
        initial,
        (filtered_probs, ind),
        reverse=True,
    )[1]


def chunked_filter_smoother(
    time: np.array,
    state_ind: np.array,
    initial_distribution: np.array,
    transition_matrix: np.array,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: Optional[np.array] = None,
    n_chunks: int = 1,
    log_likelihoods: Optional[np.array] = None,
    cache_log_likelihoods: bool = True,
):
    causal_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    marginal_likelihood = 0.0

    n_time = len(time)
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    n_states = len(np.unique(state_ind))
    state_mask = np.identity(n_states, dtype=np.float32)[
        state_ind
    ]  # shape (n_state_inds, n_states)

    if cache_log_likelihoods and log_likelihoods is None:
        log_likelihoods = log_likelihood_func(
            time,
            *log_likelihood_args,
            is_missing=is_missing,
        )

    for chunk_id, time_inds in enumerate(time_chunks):
        if log_likelihoods is not None:
            log_likelihood_chunk = log_likelihoods[time_inds]
        else:
            is_missing_chunk = is_missing[time_inds] if is_missing is not None else None
            log_likelihood_chunk = log_likelihood_func(
                time[time_inds],
                *log_likelihood_args,
                is_missing=is_missing_chunk,
            )

        (
            (marginal_likelihood_chunk, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk),
        ) = filter(
            initial_distribution=(
                initial_distribution if chunk_id == 0 else predicted_probs_next
            ),
            transition_matrix=transition_matrix,
            log_likelihoods=log_likelihood_chunk,
        )

        causal_posterior_chunk = np.asarray(causal_posterior_chunk)
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(causal_posterior_chunk @ state_mask)
        predictive_state_probabilities.append(
            np.asarray(predicted_probs_chunk) @ state_mask
        )

        marginal_likelihood += marginal_likelihood_chunk

    causal_posterior = np.concatenate(causal_posterior)
    causal_state_probabilities = np.concatenate(causal_state_probabilities)
    predictive_state_probabilities = np.concatenate(predictive_state_probabilities)

    for chunk_id, time_inds in enumerate(reversed(time_chunks)):
        acausal_posterior_chunk = smoother(
            transition_matrix=transition_matrix,
            filtered_probs=causal_posterior[time_inds],
            initial=(
                causal_posterior[-1] if chunk_id == 0 else acausal_posterior_chunk[0]
            ),
            ind=time_inds,
            n_time=n_time,
        )
        acausal_posterior_chunk = np.asarray(acausal_posterior_chunk)
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(acausal_posterior_chunk @ state_mask)

    acausal_posterior = np.concatenate(acausal_posterior[::-1])
    acausal_state_probabilities = np.concatenate(acausal_state_probabilities[::-1])

    return (
        acausal_posterior,
        acausal_state_probabilities,
        marginal_likelihood,
        causal_state_probabilities,
        predictive_state_probabilities,
        log_likelihoods,
    )


## Covariate dependent filtering and smoothing ##
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

    return jax.lax.scan(
        _step,
        (0.0, initial_distribution),
        (log_likelihoods, discrete_transition_matrix),
    )


@jax.jit
def smoother_covariate_dependent(
    discrete_transition_matrix,
    continuous_transition_matrix,
    state_ind,
    filtered_probs,
    initial=None,
    ind=None,
    n_time=None,
):
    if n_time is None:
        n_time = filtered_probs.shape[0]
    if ind is None:
        ind = jnp.arange(n_time)
    if initial is None:
        initial = filtered_probs[-1]

    def _step(smoothed_probs_next, args):
        filtered_probs_t, discrete_transition_matrix_t, t = args

        transition_matrix = _get_transition_matrix(
            discrete_transition_matrix_t,
            continuous_transition_matrix,
            state_ind,
        )

        smoothed_probs = filtered_probs_t * (
            transition_matrix
            @ _divide_safe(
                smoothed_probs_next,
                filtered_probs_t @ transition_matrix,
            )
        ) * (t < n_time - 1) + filtered_probs_t * (t == n_time - 1)
        smoothed_probs /= smoothed_probs.sum(keepdims=True)

        return smoothed_probs, smoothed_probs

    return jax.lax.scan(
        _step,
        initial,
        (filtered_probs, discrete_transition_matrix, ind),
        reverse=True,
    )[1]


def chunked_filter_smoother_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: Optional[np.array] = None,
    n_chunks: int = 1,
    log_likelihoods: Optional[np.array] = None,
    cache_log_likelihoods: bool = True,
):
    causal_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    marginal_likelihood = 0.0

    n_time = len(time)
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    n_states = len(np.unique(state_ind))
    state_mask = np.identity(n_states, dtype=np.float32)[
        state_ind
    ]  # shape (n_state_inds, n_states)

    if cache_log_likelihoods and log_likelihoods is None:
        log_likelihoods = log_likelihood_func(
            time,
            *log_likelihood_args,
            is_missing=is_missing,
        )

    for chunk_id, time_inds in enumerate(time_chunks):
        if log_likelihoods is not None:
            log_likelihood_chunk = log_likelihoods[time_inds]
        else:
            is_missing_chunk = is_missing[time_inds] if is_missing is not None else None
            log_likelihood_chunk = log_likelihood_func(
                time[time_inds],
                *log_likelihood_args,
                is_missing=is_missing_chunk,
            )

        (
            (marginal_likelihood_chunk, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk),
        ) = filter_covariate_dependent(
            initial_distribution=(
                initial_distribution if chunk_id == 0 else predicted_probs_next
            ),
            discrete_transition_matrix=discrete_transition_matrix[time_inds],
            continuous_transition_matrix=continuous_transition_matrix,
            state_ind=state_ind,
            log_likelihoods=log_likelihood_chunk,
        )
        causal_posterior_chunk = np.asarray(causal_posterior_chunk)
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(causal_posterior_chunk @ state_mask)
        predictive_state_probabilities.append(
            np.asarray(predicted_probs_chunk) @ state_mask
        )

        marginal_likelihood += marginal_likelihood_chunk

    causal_posterior = np.concatenate(causal_posterior)
    causal_state_probabilities = np.concatenate(causal_state_probabilities)
    predictive_state_probabilities = np.concatenate(predictive_state_probabilities)

    for chunk_id, time_inds in enumerate(reversed(time_chunks)):
        acausal_posterior_chunk = smoother_covariate_dependent(
            discrete_transition_matrix=discrete_transition_matrix[time_inds],
            continuous_transition_matrix=continuous_transition_matrix,
            state_ind=state_ind,
            filtered_probs=causal_posterior[time_inds],
            initial=(
                causal_posterior[-1] if chunk_id == 0 else acausal_posterior_chunk[0]
            ),
            ind=time_inds,
            n_time=n_time,
        )
        acausal_posterior_chunk = np.asarray(acausal_posterior_chunk)
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(acausal_posterior_chunk @ state_mask)

    acausal_posterior = np.concatenate(acausal_posterior[::-1])
    acausal_state_probabilities = np.concatenate(acausal_state_probabilities[::-1])

    return (
        acausal_posterior,
        acausal_state_probabilities,
        marginal_likelihood,
        causal_state_probabilities,
        predictive_state_probabilities,
        log_likelihoods,
    )


## Convergence check ##
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
