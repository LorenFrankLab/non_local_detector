import jax
import jax.numpy as jnp
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


## NOTE: adapted from dynamax: https://github.com/probml/dynamax/ with modifications ##
def _normalize(
    u: jnp.ndarray, axis: int = 0, eps: float = 1e-15
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    u = jnp.clip(u, min=eps, max=None)
    c = u.sum(axis=axis)
    return u / c, c


# Helper functions for the two key filtering steps
def _condition_on(probs: jnp.ndarray, ll: jnp.ndarray) -> tuple[jnp.ndarray, float]:
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
        Numerator array.
    denominator : jnp.ndarray
        Denominator array.

    Returns
    -------
    result : jnp.ndarray
        Element-wise division result with safe handling of zero denominators.
    """
    # Use jnp.where for conditional division - XLA can optimize this well
    return jnp.where(denominator != 0.0, numerator / denominator, 0.0)


def filter(
    initial_distribution: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> tuple[tuple[float, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Performs the forward pass of the forward-backward algorithm (filtering).

    Computes the filtered state probabilities P(z_t | x_{1:t}) and the
    one-step-ahead predicted probabilities P(z_t | x_{1:t-1}) iteratively.

    Parameters
    ----------
    initial_distribution : jnp.ndarray, shape (n_states,)
        Probability distribution of the initial state, P(z_1).
    transition_matrix : jnp.ndarray, shape (n_states, n_states)
        State transition probability matrix, P(z_t | z_{t-1}).
    log_likelihoods : jnp.ndarray, shape (n_time, n_states)
        Log likelihood of the observations for each state at each time step,
        log P(x_t | z_t).

    Returns
    -------
    carry : tuple[float, jnp.ndarray]
        A tuple containing:
            - marginal_likelihood : float
                Log probability of the observations log P(x_{1:T}).
            - predicted_probs_next : jnp.ndarray, shape (n_states,)
                The final one-step-ahead prediction P(z_{T+1} | x_{1:T}).
    outputs : tuple[jnp.ndarray, jnp.ndarray]
         A tuple containing:
            - causal_posterior : jnp.ndarray, shape (n_time, n_states)
                Filtered state probabilities P(z_t | x_{1:t}) for t=1...T.
            - predicted_probs : jnp.ndarray, shape (n_time, n_states)
                One-step-ahead predicted state probabilities P(z_t | x_{1:t-1})
                for t=1...T. Note that `predicted_probs[0]` is equivalent
                to `initial_distribution`.
    """

    def _step(carry, ll):
        log_normalizer, predicted_probs = carry

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = filtered_probs @ transition_matrix

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    return jax.lax.scan(_step, (0.0, initial_distribution), log_likelihoods)


# Apply JIT without donation - tests reuse inputs
filter = jax.jit(filter)


# Internal version with buffer donation for use in chunked drivers
# This is safe because chunked drivers control the data flow and don't reuse donated arrays
_filter_internal = jax.jit(filter.__wrapped__, donate_argnums=(0, 2))


def smoother(
    transition_matrix: jnp.ndarray,
    filtered_probs: jnp.ndarray,
    initial: jnp.ndarray | None = None,
    ind: jnp.ndarray | None = None,
    n_time: int | None = None,
) -> jnp.ndarray:
    """Smoother step.

    Parameters
    ----------
    transition_matrix : jnp.ndarray, shape (n_states, n_states)
        Transition matrix
    filtered_probs : jnp.ndarray, shape (n_time, n_states)
        Filtered state probabilities.
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
        smoothed_probs_next: jnp.ndarray, args: tuple[jnp.ndarray, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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


# Apply JIT without donation - tests reuse inputs
smoother = jax.jit(smoother)


# Internal version with buffer donation for use in chunked drivers
_smoother_internal = jax.jit(smoother.__wrapped__, donate_argnums=(1,))


def chunked_filter_smoother(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    n_chunks: int = 1,
    log_likelihoods: np.ndarray | None = None,
    cache_log_likelihoods: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Filtered state probabilities
    """
    causal_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    marginal_likelihood = 0.0

    n_time = len(time)
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    # Convert inputs to JAX arrays once at the top - keep data on device
    state_ind_jax = jnp.asarray(state_ind)
    initial_distribution_jax = jnp.asarray(initial_distribution)
    transition_matrix_jax = jnp.asarray(transition_matrix)

    # Initialize variables that may be referenced in loop conditionals
    predicted_probs_next = None
    acausal_posterior_chunk = None

    # Precompute or convert log_likelihoods to JAX once
    if cache_log_likelihoods and log_likelihoods is None:
        log_likelihoods = log_likelihood_func(
            time,
            *log_likelihood_args,
            is_missing=is_missing,
        )
    log_likelihoods_jax = (
        jnp.asarray(log_likelihoods) if log_likelihoods is not None else None
    )

    # Forward pass: accumulate JAX arrays
    for chunk_id, time_inds in enumerate(time_chunks):
        if log_likelihoods_jax is not None:
            log_likelihood_chunk = log_likelihoods_jax[time_inds]
        else:
            is_missing_chunk = is_missing[time_inds] if is_missing is not None else None
            log_likelihood_chunk = log_likelihood_func(
                time[time_inds],
                *log_likelihood_args,
                is_missing=is_missing_chunk,
            )
            log_likelihood_chunk = jnp.asarray(log_likelihood_chunk)

        # Use internal version with buffer donation for memory efficiency
        # Safe because log_likelihood_chunk is created fresh each iteration
        (
            (marginal_likelihood_chunk, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk),
        ) = _filter_internal(
            initial_distribution=(
                initial_distribution_jax if chunk_id == 0 else predicted_probs_next
            ),
            transition_matrix=transition_matrix_jax,
            log_likelihoods=log_likelihood_chunk,
        )

        # Keep as JAX arrays - no conversion to NumPy yet
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(causal_posterior_chunk[:, state_ind_jax])
        predictive_state_probabilities.append(predicted_probs_chunk[:, state_ind_jax])

        marginal_likelihood += marginal_likelihood_chunk

    # Concatenate JAX arrays on device
    causal_posterior_jax = jnp.concatenate(causal_posterior)
    causal_state_probabilities_jax = jnp.concatenate(causal_state_probabilities)
    predictive_state_probabilities_jax = jnp.concatenate(predictive_state_probabilities)

    # Backward pass: accumulate JAX arrays
    for chunk_id, time_inds in enumerate(reversed(time_chunks)):
        # Use internal version with buffer donation
        # Safe because causal_posterior_jax[time_inds] creates a slice (copy)
        acausal_posterior_chunk = _smoother_internal(
            transition_matrix=transition_matrix_jax,
            filtered_probs=causal_posterior_jax[time_inds],
            initial=(
                causal_posterior_jax[-1]
                if chunk_id == 0
                else acausal_posterior_chunk[0]
            ),
            ind=jnp.asarray(time_inds),
            n_time=n_time,
        )
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(acausal_posterior_chunk[:, state_ind_jax])

    # Concatenate JAX arrays on device
    acausal_posterior_jax = jnp.concatenate(acausal_posterior[::-1])
    acausal_state_probabilities_jax = jnp.concatenate(acausal_state_probabilities[::-1])

    # Convert to NumPy only at the very end for API compatibility
    return (
        np.asarray(acausal_posterior_jax),
        np.asarray(acausal_state_probabilities_jax),
        float(marginal_likelihood),
        np.asarray(causal_state_probabilities_jax),
        np.asarray(predictive_state_probabilities_jax),
        log_likelihoods,  # Keep as original (may be None or NumPy)
        np.asarray(causal_posterior_jax),
    )


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
        Log likelihoods for each state at each time point.

    Returns
    -------
    most_likely_state_sequence : jnp.ndarray, shape (n_time,)

    """
    # Precompute logs once outside the scan loop
    log_transition_matrix = jnp.log(transition_matrix)
    log_initial_distribution = jnp.log(initial_distribution)

    # Run the backward pass
    def _backward_pass(best_next_score, t):
        scores = log_transition_matrix + best_next_score + log_likelihoods[t + 1]
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
        log_initial_distribution + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


# Apply JIT without donation - tests reuse inputs
viterbi = jax.jit(viterbi)


def most_likely_sequence(
    time: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    log_likelihoods: np.ndarray | None = None,
    n_chunks: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the most likely state sequence using the Viterbi algorithm.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time indices.
    initial_distribution : np.ndarray, shape (n_states,)
        Initial state distribution.
    transition_matrix : np.ndarray, shape (n_states, n_states)
        State transition probability matrix.
    log_likelihood_func : callable
        Function to compute log likelihoods.
    log_likelihood_args : tuple
        Arguments to pass to log_likelihood_func.
    is_missing : np.ndarray, shape (n_time,), optional
        Boolean array indicating missing observations, by default None.
    log_likelihoods : np.ndarray, shape (n_time, n_states), optional
        Pre-computed log likelihoods, by default None.
    n_chunks : int, optional
        Number of chunks (not yet implemented for > 1), by default 1.

    Returns
    -------
    most_likely_sequence : np.ndarray, shape (n_time,)
        Most likely state sequence.
    log_likelihoods : np.ndarray, shape (n_time, n_states)
        Log likelihoods for each state at each time point.
    """
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


def filter_covariate_dependent(
    initial_distribution: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> tuple[tuple[float, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
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
        carry: tuple[float, jnp.ndarray], args: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[float, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
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


# Apply JIT without donation - tests reuse inputs
filter_covariate_dependent = jax.jit(filter_covariate_dependent)


# Internal version with buffer donation for use in chunked drivers
_filter_covariate_dependent_internal = jax.jit(
    filter_covariate_dependent.__wrapped__, donate_argnums=(0, 4)
)


def smoother_covariate_dependent(
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    filtered_probs: jnp.ndarray,
    initial: jnp.ndarray | None = None,
    ind: jnp.ndarray | None = None,
    n_time: int | None = None,
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


# Apply JIT without donation - tests reuse inputs
smoother_covariate_dependent = jax.jit(smoother_covariate_dependent)


# Internal version with buffer donation for use in chunked drivers
_smoother_covariate_dependent_internal = jax.jit(
    smoother_covariate_dependent.__wrapped__, donate_argnums=(3,)
)


def chunked_filter_smoother_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    n_chunks: int = 1,
    log_likelihoods: np.ndarray | None = None,
    cache_log_likelihoods: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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
        If True, log likelihoods are cached instead of recomputed for each chunk, by default True

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
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Filtered state probabilities
    """
    causal_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    marginal_likelihood = 0.0

    n_time = len(time)
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    # Convert inputs to JAX arrays once at the top - keep data on device
    state_ind_jax = jnp.asarray(state_ind)
    initial_distribution_jax = jnp.asarray(initial_distribution)
    discrete_transition_matrix_jax = jnp.asarray(discrete_transition_matrix)
    continuous_transition_matrix_jax = jnp.asarray(continuous_transition_matrix)

    # Initialize variables that may be referenced in loop conditionals
    predicted_probs_next = None
    acausal_posterior_chunk = None

    # Precompute or convert log_likelihoods to JAX once
    if cache_log_likelihoods and log_likelihoods is None:
        log_likelihoods = log_likelihood_func(
            time,
            *log_likelihood_args,
            is_missing=is_missing,
        )
    log_likelihoods_jax = (
        jnp.asarray(log_likelihoods) if log_likelihoods is not None else None
    )

    # Forward pass: accumulate JAX arrays
    for chunk_id, time_inds in enumerate(time_chunks):
        if log_likelihoods_jax is not None:
            log_likelihood_chunk = log_likelihoods_jax[time_inds]
        else:
            is_missing_chunk = is_missing[time_inds] if is_missing is not None else None
            log_likelihood_chunk = log_likelihood_func(
                time[time_inds],
                *log_likelihood_args,
                is_missing=is_missing_chunk,
            )
            log_likelihood_chunk = jnp.asarray(log_likelihood_chunk)

        # Use internal version with buffer donation for memory efficiency
        # Safe because log_likelihood_chunk is created fresh each iteration
        (
            (marginal_likelihood_chunk, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk),
        ) = _filter_covariate_dependent_internal(
            initial_distribution=(
                initial_distribution_jax if chunk_id == 0 else predicted_probs_next
            ),
            discrete_transition_matrix=discrete_transition_matrix_jax[time_inds],
            continuous_transition_matrix=continuous_transition_matrix_jax,
            state_ind=state_ind_jax,
            log_likelihoods=log_likelihood_chunk,
        )

        # Keep as JAX arrays - no conversion to NumPy yet
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(causal_posterior_chunk[:, state_ind_jax])
        predictive_state_probabilities.append(predicted_probs_chunk[:, state_ind_jax])

        marginal_likelihood += marginal_likelihood_chunk

    # Concatenate JAX arrays on device
    causal_posterior_jax = jnp.concatenate(causal_posterior)
    causal_state_probabilities_jax = jnp.concatenate(causal_state_probabilities)
    predictive_state_probabilities_jax = jnp.concatenate(predictive_state_probabilities)

    # Backward pass: accumulate JAX arrays
    for chunk_id, time_inds in enumerate(reversed(time_chunks)):
        # Use internal version with buffer donation
        # Safe because causal_posterior_jax[time_inds] creates a slice (copy)
        acausal_posterior_chunk = _smoother_covariate_dependent_internal(
            discrete_transition_matrix=discrete_transition_matrix_jax[time_inds],
            continuous_transition_matrix=continuous_transition_matrix_jax,
            state_ind=state_ind_jax,
            filtered_probs=causal_posterior_jax[time_inds],
            initial=(
                causal_posterior_jax[-1]
                if chunk_id == 0
                else acausal_posterior_chunk[0]
            ),
            ind=jnp.asarray(time_inds),
            n_time=n_time,
        )
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(acausal_posterior_chunk[:, state_ind_jax])

    # Concatenate JAX arrays on device
    acausal_posterior_jax = jnp.concatenate(acausal_posterior[::-1])
    acausal_state_probabilities_jax = jnp.concatenate(acausal_state_probabilities[::-1])

    # Convert to NumPy only at the very end for API compatibility
    return (
        np.asarray(acausal_posterior_jax),
        np.asarray(acausal_state_probabilities_jax),
        float(marginal_likelihood),
        np.asarray(causal_state_probabilities_jax),
        np.asarray(predictive_state_probabilities_jax),
        log_likelihoods,  # Keep as original (may be None or NumPy)
        np.asarray(causal_posterior_jax),
    )


def viterbi_covariate_dependent(
    initial_distribution: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    state_ind: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the most likely state sequence with covariate dependent transitions.

    Uses the Viterbi algorithm with time-varying transition matrices.

    Parameters
    ----------
    initial_distribution : jnp.ndarray, shape (n_state_bins,)
        Initial state distribution.
    discrete_transition_matrix : jnp.ndarray, shape (n_time, n_states, n_states)
        Time-varying discrete transition matrices.
    continuous_transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
        Continuous transition matrix.
    state_ind : jnp.ndarray, shape (n_state_bins,)
        State indices connecting discrete states to state space bins.
    log_likelihoods : jnp.ndarray, shape (n_time, n_state_bins)
        Log likelihoods for each state bin at each time point.

    Returns
    -------
    most_likely_state_sequence : jnp.ndarray, shape (n_time,)
        Most likely state sequence.
    """
    # Precompute logs once outside the scan loop
    log_continuous_transition_matrix = jnp.log(continuous_transition_matrix)
    log_initial_distribution = jnp.log(initial_distribution)

    # Run the backward pass
    def _backward_pass(best_next_score, args):
        t, discrete_transition_matrix_t = args
        # Build log-space transition matrix: log(A * B) = log(A) + log(B)
        log_transition_matrix = log_continuous_transition_matrix + jnp.log(
            discrete_transition_matrix_t[jnp.ix_(state_ind, state_ind)]
        )
        scores = log_transition_matrix + best_next_score + log_likelihoods[t + 1]
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
        log_initial_distribution + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


# Apply JIT without donation - tests reuse inputs
viterbi_covariate_dependent = jax.jit(viterbi_covariate_dependent)


def most_likely_sequence_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: callable,
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    log_likelihoods: np.ndarray | None = None,
    n_chunks: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the most likely state sequence with covariate dependent transitions.

    Wrapper function for Viterbi algorithm with time-varying transition matrices.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time indices.
    state_ind : np.ndarray, shape (n_state_bins,)
        State indices connecting discrete states to state space bins.
    initial_distribution : np.ndarray, shape (n_state_bins,)
        Initial state distribution.
    discrete_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Time-varying discrete transition matrices.
    continuous_transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
        Continuous transition matrix.
    log_likelihood_func : callable
        Function to compute log likelihoods.
    log_likelihood_args : tuple
        Arguments to pass to log_likelihood_func.
    is_missing : np.ndarray, shape (n_time,), optional
        Boolean array indicating missing observations, by default None.
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins), optional
        Pre-computed log likelihoods, by default None.
    n_chunks : int, optional
        Number of chunks (not yet implemented for > 1), by default 1.

    Returns
    -------
    most_likely_sequence : np.ndarray, shape (n_time,)
        Most likely state sequence.
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        Log likelihoods for each state bin at each time point.
    """
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
) -> tuple[bool, bool]:
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
