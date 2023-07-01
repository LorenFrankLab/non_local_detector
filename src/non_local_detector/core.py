from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


def convert_to_state_probability(
    causal_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    state_ind: np.ndarray,
):
    n_states = np.unique(state_ind).size
    n_time = causal_posterior.shape[0]

    causal_state_probabilities = np.zeros((n_time, n_states))
    acausal_state_probabilities = np.zeros((n_time, n_states))
    predictive_state_probabilities = np.zeros((n_time, n_states))

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


## NOTE: copied from dynamax: https://github.com/probml/dynamax/ ##
def get_trans_mat(transition_matrix, transition_fn, t):
    if transition_fn is not None:
        return transition_fn(t)
    else:
        if transition_matrix.ndim == 3:  # (T,K,K)
            return transition_matrix[t]
        else:
            return transition_matrix


def _normalize(u, axis=0, eps=1e-15):
    """Normalizes the values within the axis in a way that they sum up to 1.

    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.

    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
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
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(probs, A):
    return A.T @ probs


@partial(jax.jit, static_argnames=["transition_fn"])
def hmm_filter(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
    transition_fn=None,
):
    r"""Forwards filtering

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        filtered posterior distribution

    """
    num_timesteps = log_likelihoods.shape[0]

    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        A = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t]

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, A)

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = jax.lax.scan(
        _step, carry, jnp.arange(num_timesteps)
    )

    return (
        log_normalizer,
        filtered_probs,
        predicted_probs,
    )


@partial(jax.jit, static_argnames=["transition_fn"])
def hmm_smoother(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
    transition_fn=None,
):
    r"""Computed the smoothed state probabilities using a general
    Bayesian smoother.

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    *Note: This is the discrete SSM analog of the RTS smoother for linear Gaussian SSMs.*

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        posterior distribution

    """
    num_timesteps = log_likelihoods.shape[0]

    # Run the HMM filter
    (marginal_loglik, filtered_probs, predicted_probs) = hmm_filter(
        initial_distribution, transition_matrix, log_likelihoods, transition_fn
    )

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs, predicted_probs_next = args

        A = get_trans_mat(transition_matrix, transition_fn, t)

        # Fold in the next state (Eq. 8.2 of Saarka, 2013)
        # If hard 0. in predicted_probs_next, set relative_probs_next as 0. to avoid NaN values
        relative_probs_next = jnp.where(
            jnp.isclose(predicted_probs_next, 0.0),
            0.0,
            smoothed_probs_next / predicted_probs_next,
        )
        smoothed_probs = filtered_probs * (A @ relative_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        return smoothed_probs, smoothed_probs

    # Run the HMM smoother
    carry = filtered_probs[-1]
    args = (
        jnp.arange(num_timesteps - 2, -1, -1),
        filtered_probs[:-1][::-1],
        predicted_probs[1:][::-1],
    )
    _, rev_smoothed_probs = jax.lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = jnp.row_stack([rev_smoothed_probs[::-1], filtered_probs[-1]])

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
