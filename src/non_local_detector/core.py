import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

# Note: This only affects NumPy operations, not JAX operations
# Most computation is in JAX, so this has minimal effect
np.seterr(divide="ignore", invalid="ignore")


## NOTE: adapted from dynamax: https://github.com/probml/dynamax/ with modifications ##
def _normalize(
    u: ArrayLike, axis: int = 0, eps: float = 1e-15
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Normalizes the values within the axis in a way that they sum up to 1.

    Avoids clipping to preserve probability mass. Adds eps to denominator
    for numerical stability instead of clipping the input.

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
        Normalization constant (squeezed along normalized axis)
    """
    c = u.sum(axis=axis, keepdims=True)
    u = u / (c + eps)  # Add eps to denominator, don't clip input
    return u, c.squeeze(axis)


# Helper functions for the two key filtering steps
def _condition_on(probs: ArrayLike, ll: ArrayLike) -> tuple[jnp.ndarray, float]:
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


def _safe_log(p: ArrayLike) -> jnp.ndarray:
    """Compute log of probabilities safely, handling zeros.

    Returns -inf for zero probabilities (valid in log space) instead of NaN.

    Parameters
    ----------
    p : jnp.ndarray
        Probability array (may contain zeros)

    Returns
    -------
    log_p : jnp.ndarray
        Log probabilities with -inf for zeros
    """
    return jnp.where(p > 0, jnp.log(p), -jnp.inf)


def _divide_safe(numerator: ArrayLike, denominator: ArrayLike) -> jnp.ndarray:
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
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
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
    transition_matrix: ArrayLike,
    filtered_probs: ArrayLike,
    initial: ArrayLike | None = None,
    ind: ArrayLike | None = None,
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

        # Compute predicted probs once and reuse (avoids redundant matmul)
        predicted_probs = filtered_probs_t @ transition_matrix

        smoothed_probs = filtered_probs_t * (
            transition_matrix @ _divide_safe(smoothed_probs_next, predicted_probs)
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
# Donates filtered_probs (arg 1) and initial (arg 2) - both are single-use in chunked loops
_smoother_internal = jax.jit(smoother.__wrapped__, donate_argnums=(1, 2))


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
    dtype: jnp.dtype = jnp.float32,
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
    dtype : jnp.dtype, optional
        Data type for computations (jnp.float32 or jnp.float64), by default jnp.float32.
        Use float64 for numerically challenging problems.
        Note: Requires JAX_ENABLE_X64=1 environment variable for float64.

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
    # Cast to desired dtype for precision control
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    transition_matrix_jax = jnp.asarray(transition_matrix, dtype=dtype)

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
        jnp.asarray(log_likelihoods, dtype=dtype)
        if log_likelihoods is not None
        else None
    )

    # Forward pass: accumulate JAX arrays
    for chunk_id, time_inds_np in enumerate(time_chunks):
        # Convert time_inds to JAX once to avoid repeated host↔device syncs
        time_inds = jnp.asarray(time_inds_np)

        if log_likelihoods_jax is not None:
            log_likelihood_chunk = log_likelihoods_jax[time_inds]
        else:
            is_missing_chunk = (
                is_missing[time_inds_np] if is_missing is not None else None
            )
            log_likelihood_chunk = log_likelihood_func(
                time[time_inds_np],
                *log_likelihood_args,
                is_missing=is_missing_chunk,
            )
            log_likelihood_chunk = jnp.asarray(log_likelihood_chunk, dtype=dtype)

        # Donated: log_likelihood_chunk (created fresh), initial_distribution
        # Do not read these after the call - they are consumed by the JIT function
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
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
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

    Notes
    -----
    Base case at t=T: terminal cost is 0, we accumulate from t=T-1 down.
    """
    # Precompute logs once outside the scan loop using safe log
    # Handles zero probabilities correctly (returns -inf, not NaN)
    log_transition_matrix = _safe_log(transition_matrix)
    log_initial_distribution = _safe_log(initial_distribution)

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
    discrete_transition_matrix_t: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike | None = None,
) -> jnp.ndarray:
    """Get the transition matrix for the current time point.

    Combines the discrete and continuous transition matrices.

    Parameters
    ----------
    discrete_transition_matrix_t : jnp.ndarray, shape (n_state_bins, n_state_bins) or (n_states, n_states)
        If state_ind is provided, expected shape is (n_states, n_states) and will be indexed.
        If state_ind is None, expected shape is (n_state_bins, n_state_bins) (pre-indexed).
    continuous_transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : jnp.ndarray, shape (n_state_bins,), optional
        If provided, used to index discrete_transition_matrix_t.
        If None, assumes discrete_transition_matrix_t is already indexed.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    """
    if state_ind is not None:
        # Need to index - used by public API functions
        discrete_indexed = discrete_transition_matrix_t[jnp.ix_(state_ind, state_ind)]
    else:
        # Already indexed - used by optimized chunked functions
        discrete_indexed = discrete_transition_matrix_t

    return continuous_transition_matrix * discrete_indexed


def filter_covariate_dependent(
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
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
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    filtered_probs: ArrayLike,
    initial: ArrayLike | None = None,
    ind: ArrayLike | None = None,
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

        # Compute predicted probs once and reuse (avoids redundant matmul)
        predicted_probs = filtered_probs_t @ transition_matrix

        smoothed_probs = filtered_probs_t * (
            transition_matrix @ _divide_safe(smoothed_probs_next, predicted_probs)
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
# Donates filtered_probs (arg 3) and initial (arg 4) - both are single-use in chunked loops
_smoother_covariate_dependent_internal = jax.jit(
    smoother_covariate_dependent.__wrapped__, donate_argnums=(3, 4)
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
    dtype: jnp.dtype = jnp.float32,
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
    dtype : jnp.dtype, optional
        Data type for computations (jnp.float32 or jnp.float64), by default jnp.float32.
        Use float64 for numerically challenging problems.
        Note: Requires JAX_ENABLE_X64=1 environment variable for float64.

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
    # Cast to desired dtype for precision control
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    continuous_transition_matrix_jax = jnp.asarray(
        continuous_transition_matrix, dtype=dtype
    )

    # Precompute indexed discrete blocks to reduce repeated advanced indexing in scans
    # This slices discrete_transition_matrix[:, state_ind][:, :, state_ind] once
    discrete_transition_matrix_jax = jnp.asarray(
        discrete_transition_matrix, dtype=dtype
    )
    discrete_transition_matrix_indexed = discrete_transition_matrix_jax[
        :, state_ind_jax[:, None], state_ind_jax
    ]

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
        jnp.asarray(log_likelihoods, dtype=dtype)
        if log_likelihoods is not None
        else None
    )

    # Forward pass: accumulate JAX arrays
    for chunk_id, time_inds_np in enumerate(time_chunks):
        # Convert time_inds to JAX once to avoid repeated host↔device syncs
        time_inds = jnp.asarray(time_inds_np)

        if log_likelihoods_jax is not None:
            log_likelihood_chunk = log_likelihoods_jax[time_inds]
        else:
            is_missing_chunk = (
                is_missing[time_inds_np] if is_missing is not None else None
            )
            log_likelihood_chunk = log_likelihood_func(
                time[time_inds_np],
                *log_likelihood_args,
                is_missing=is_missing_chunk,
            )
            log_likelihood_chunk = jnp.asarray(log_likelihood_chunk, dtype=dtype)

        # Donated: log_likelihood_chunk (created fresh), initial_distribution
        # Do not read these after the call - they are consumed by the JIT function
        (
            (marginal_likelihood_chunk, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk),
        ) = _filter_covariate_dependent_internal(
            initial_distribution=(
                initial_distribution_jax if chunk_id == 0 else predicted_probs_next
            ),
            discrete_transition_matrix=discrete_transition_matrix_indexed[time_inds],
            continuous_transition_matrix=continuous_transition_matrix_jax,
            state_ind=None,  # Pre-indexed, don't index again
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
            discrete_transition_matrix=discrete_transition_matrix_indexed[time_inds],
            continuous_transition_matrix=continuous_transition_matrix_jax,
            state_ind=None,  # Pre-indexed, don't index again
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
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
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

    Notes
    -----
    Base case at t=T: terminal cost is 0, we accumulate from t=T-1 down.
    """
    # Precompute logs once outside the scan loop using safe log
    # Handles zero probabilities correctly (returns -inf, not NaN)
    log_continuous_transition_matrix = _safe_log(continuous_transition_matrix)
    log_initial_distribution = _safe_log(initial_distribution)

    # Pre-index and precompute log of discrete transitions to avoid repeated operations in scan
    # This eliminates both the expensive gather operation AND the repeated log computation
    discrete_indexed = discrete_transition_matrix[:, state_ind[:, None], state_ind]
    log_discrete_transition_matrix = _safe_log(discrete_indexed)

    # Run the backward pass
    def _backward_pass(best_next_score, args):
        t, log_discrete_t = args
        # Build log-space transition matrix: log(A * B) = log(A) + log(B)
        # Both operands are already in log space, just add them
        log_transition_matrix = log_continuous_transition_matrix + log_discrete_t
        scores = log_transition_matrix + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, best_next_states = jax.lax.scan(
        _backward_pass,
        jnp.zeros(num_states),
        (jnp.arange(num_timesteps - 1), log_discrete_transition_matrix[:-1]),
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
