from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


## NOTE: adapted from dynamax: https://github.com/probml/dynamax/ with modifications ##
def _normalize(
    u: jnp.ndarray, axis: int = 0, eps: float = 1e-15
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalizes the values within the axis in a way that they sum up to 1.

    Parameters
    ----------
    u : jnp.ndarray
        Array to normalize
    axis : int, optional
        Axis to normalize, by default 0
    eps : float, optional
        Small value to avoid division by zero, by default 1e-15

    Returns
    -------
    normalized_u : jnp.ndarray
        Normalized array
    c : jnp.ndarray
        Normalization constant
    """
    u = jnp.clip(u, a_min=eps, a_max=None)
    c = u.sum(axis=axis)
    return u / c, c


# Helper functions for the two key filtering steps
def _condition_on(probs: jnp.ndarray, ll: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.

    Parameters
    ----------
    probs : jnp.ndarray
        Current state probabilities
    ll : jnp.ndarray
        Log likelihoods for each state

    Returns
    -------
    new_probs : jnp.ndarray
        Updated state probabilities
    log_norm : float
        Log normalization constant
    """
    ll_max = ll.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    new_probs, norm = _normalize(probs * jnp.exp(ll - ll_max))
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _divide_safe(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    """Divides two arrays, while setting the result to 0.0
    if the denominator is 0.0.

    Parameters
    ----------
    numerator : jnp.ndarray
    denominator : jnp.ndarray
    """
    return jnp.where(denominator == 0.0, 0.0, numerator / denominator)


@jax.jit
def filter(
    initial_distribution: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> Tuple[Tuple[float, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Filtering step.

    Parameters
    ----------
    initial_distribution : jnp.ndarray, shape (n_states,)
        Initial state distribution
    transition_matrix : jnp.ndarray, shape (n_states, n_states)
        Transition matrix
    log_likelihoods : jnp.ndarray, shape (n_time, n_states)
        Log likelihoods for each state at each time point

    Returns
    -------
    marginal_likelihood : float
    predicted_probs_next : jnp.ndarray, shape (n_states,)
        Next one-step-ahead predicted state probabilities
    causal_posterior : jnp.ndarray, shape (n_time, n_states)
        Filtered state probabilities
    predicted_probs : jnp.ndarray, shape (n_time, n_states)
        One-step-ahead predicted state probabilities
    """

    def _step(carry, ll):
        log_normalizer, predicted_probs = carry

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = filtered_probs @ transition_matrix

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    return jax.lax.scan(_step, (0.0, initial_distribution), log_likelihoods)


@jax.jit
def smoother(
    transition_matrix: jnp.ndarray,
    filtered_probs: jnp.ndarray,
    initial: Optional[jnp.ndarray] = None,
    ind: Optional[jnp.ndarray] = None,
    n_time: Optional[int] = None,
) -> jnp.ndarray:
    """Smoother step.

    Parameters
    ----------
    transition_matrix : jnp.ndarray, shape (n_states, n_states)
        Transition matrix
    filtered_probs : jnp.ndarray, shape (n_time, n_states)
        Filtered state probabilities
    initial : jnp.ndarray, optional
        Initial state distribution, by default None
    ind : jnp.ndarray, optional
        Time indices, by default None
    n_time : int, optional
        Number of time points, by default None

    Returns
    -------
    smoothed_probs : jnp.ndarray, shape (n_time, n_states)
        Smoothed state probabilities
    """
    if n_time is None:
        n_time = filtered_probs.shape[0]
    if ind is None:
        ind = jnp.arange(n_time)
    if initial is None:
        initial = filtered_probs[-1]

    def _step(
        smoothed_probs_next: jnp.ndarray, args: Tuple[jnp.ndarray, int]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: Optional[np.ndarray] = None,
    n_chunks: int = 1,
    log_likelihoods: Optional[np.ndarray] = None,
    cache_log_likelihoods: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Filter and smooth the state probabilities in chunks.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    state_ind : np.ndarray, shape (n_state_bins,)
        Connects the discrete states to the state space
    initial_distribution : np.ndarray, shape (n_state_bins,)
    transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
    log_likelihood_func : callable
    log_likelihood_args : tuple
    is_missing : np.ndarray, shape (n_time,), optional
    n_chunks : int, optional
        Number of chunks to split the data into, by default 1
    log_likelihoods : Optional[np.ndarray], optional
    cache_log_likelihoods : bool, optional
        If True, log likelihoods are cached, by default True

    Returns
    -------
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Smoothed state probabilities
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Smoothed state probabilities for each state index
    marginal_likelihood : float
    causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Filtered state probabilities for each state index
    predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        One-step-ahead predicted state probabilities for each state index
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        Log likelihoods for each state at each time point
    """
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


@jax.jit
def viterbi(
    initial_distribution: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Parameters
    ----------
    initial_distribution : jnp.ndarray, shape (n_states,)
        Initial state distribution
    transition_matrix : jnp.ndarray, shape (n_states, n_states)
        Transition matrix
    log_likelihoods : jnp.ndarray, shape (n_time, n_states)
        Log likelihoods for each state at each time point

    Returns
    -------
    most_likely_state_sequence : jnp.ndarray, shape (n_time,)

    """

    # Run the backward pass
    def _backward_pass(best_next_score, t):
        scores = jnp.log(transition_matrix) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, best_next_states = jax.lax.scan(
        _backward_pass,
        jnp.zeros(num_states),
        jnp.arange(num_timesteps - 1),
        reverse=True,
    )

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


def most_likely_sequence(
    time: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: Optional[np.ndarray] = None,
    log_likelihoods: Optional[np.ndarray] = None,
    n_chunks: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:

    if n_chunks > 1:
        raise NotImplementedError("Chunked Viterbi is not yet implemented.")

    if log_likelihoods is None:
        log_likelihoods = log_likelihood_func(
            time,
            *log_likelihood_args,
            is_missing=is_missing,
        )
    return viterbi(
        initial_distribution=initial_distribution,
        transition_matrix=transition_matrix,
        log_likelihoods=log_likelihoods,
    )


## Covariate dependent filtering and smoothing ##
def _get_transition_matrix(
    discrete_transition_matrix_t: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
) -> jnp.ndarray:
    """Get the transition matrix for the current time point.

    Combines the discrete and continuous transition matrices.

    Parameters
    ----------
    discrete_transition_matrix_t : jnp.ndarray, shape (n_states, n_states)
    continuous_transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : jnp.ndarray, shape (n_state_bins,)

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    """
    return (
        continuous_transition_matrix
        * discrete_transition_matrix_t[jnp.ix_(state_ind, state_ind)]
    )


@jax.jit
def filter_covariate_dependent(
    initial_distribution: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> Tuple[Tuple[float, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Filtering step with covariate dependent transitions.

    Parameters
    ----------
    initial_distribution : jnp.ndarray, shape (n_state_bins,)
    discrete_transition_matrix : jnp.ndarray, shape (n_time, n_states, n_states)
    continuous_transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : jnp.ndarray, shape (n_state_bins,)
    log_likelihoods : jnp.ndarray, shape (n_time, n_states)

    Returns
    -------
    marginal_likelihood : float
    predicted_probs_next : jnp.ndarray, shape (n_state_bins,)
        Next one-step-ahead predicted state probabilities
    causal_posterior : jnp.ndarray, shape (n_time, n_state_bins)
        Filtered state probabilities
    predicted_probs : jnp.ndarray, shape (n_time, n_state_bins)
        One-step-ahead predicted state probabilities
    """

    def _step(
        carry: Tuple[float, jnp.ndarray], args: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Tuple[float, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        log_normalizer, predicted_probs = carry
        ll, discrete_transition_matrix_t = args

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
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    filtered_probs: jnp.ndarray,
    initial: Optional[jnp.ndarray] = None,
    ind: Optional[jnp.ndarray] = None,
    n_time: Optional[int] = None,
) -> jnp.ndarray:
    """Smoother step with covariate dependent transitions.

    Parameters
    ----------
    discrete_transition_matrix : jnp.ndarray, shape (n_time, n_states, n_states)
    continuous_transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : jnp.ndarray, shape (n_state_bins,)
    filtered_probs : jnp.ndarray, shape (n_time, n_state_bins)
    initial : jnp.ndarray, shape (n_state_bins,), optional
    ind : jnp.ndarray, shape (n_time,), optional
    n_time : int, optional

    Returns
    -------
    smoothed_probs : jnp.ndarray, shape (n_time, n_state_bins)
    """
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
    """Filter and smooth the state probabilities in chunks with covariate dependent transitions.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    state_ind : np.ndarray, shape (n_state_bins,)
    initial_distribution : np.ndarray, shape (n_state_bins,)
    discrete_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
    continuous_transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
    log_likelihood_func : callable
    log_likelihood_args : tuple
    is_missing : np.ndarray, shape (n_time,), optional
    n_chunks : int, optional
        Number of chunks to split the data into, by default 1
    log_likelihoods : np.ndarray, optional
    cache_log_likelihoods : bool, optional
        If True, log likelihoods are cached, by default True

    Returns
    -------
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Smoothed state probabilities
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Smoothed state probabilities for each state index
    marginal_likelihood : float
    causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Filtered state probabilities for each state index
    predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        One-step-ahead predicted state probabilities for each state index
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        Log likelihoods for each state at each time point
    """
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


@jax.jit
def viterbi_covariate_dependent(
    initial_distribution: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Parameters
    ----------
    initial_distribution : jnp.ndarray, shape (n_states,)
        Initial state distribution
    transition_matrix : jnp.ndarray, shape (n_states, n_states)
        Transition matrix
    log_likelihoods : jnp.ndarray, shape (n_time, n_states)
        Log likelihoods for each state at each time point

    Returns
    -------
    most_likely_state_sequence : jnp.ndarray, shape (n_time,)

    """

    # Run the backward pass
    def _backward_pass(best_next_score, args):
        t, discrete_transition_matrix_t = args
        transition_matrix = _get_transition_matrix(
            discrete_transition_matrix_t,
            continuous_transition_matrix,
            state_ind,
        )
        scores = jnp.log(transition_matrix) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, best_next_states = jax.lax.scan(
        _backward_pass,
        jnp.zeros(num_states),
        (jnp.arange(num_timesteps - 1), discrete_transition_matrix[:-1]),
        reverse=True,
    )

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


def most_likely_sequence_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: Optional[np.ndarray] = None,
    log_likelihoods: Optional[np.ndarray] = None,
    n_chunks: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_chunks > 1:
        raise NotImplementedError("Chunked Viterbi is not yet implemented.")

    if log_likelihoods is None:
        log_likelihoods = log_likelihood_func(
            time,
            *log_likelihood_args,
            is_missing=is_missing,
        )
    return viterbi_covariate_dependent(
        initial_distribution=initial_distribution,
        discrete_transition_matrix=discrete_transition_matrix,
        continuous_transition_matrix=continuous_transition_matrix,
        state_ind=state_ind,
        log_likelihoods=log_likelihoods,
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
