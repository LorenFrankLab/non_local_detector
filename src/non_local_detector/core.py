import numpy as np
from replay_trajectory_classification.core import atleast_2d, check_converged  # noqa


def get_transition_matrix(
    continuous_state_transitions, discrete_state_transitions, state_ind, t
):
    if discrete_state_transitions.ndim == 2:
        # could consider caching this
        return (
            continuous_state_transitions
            * discrete_state_transitions[np.ix_(state_ind, state_ind)]
        )
    else:
        return (
            continuous_state_transitions
            * discrete_state_transitions[t][np.ix_(state_ind, state_ind)]
        )


def forward(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    discrete_state_transitions: np.ndarray,
    continuous_state_transitions: np.ndarray,
    state_ind: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Causal algorithm for computing the posterior distribution of the hidden states of a switching model

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    discrete_state_transitions : np.ndarray, shape (n_time, n_states, n_states) or (n_states, n_states)
    continuous_state_transitions : np.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : np.ndarray, shape (n_state_bins,)

    Returns
    -------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    marginal_likelihood : float

    """
    n_time = log_likelihood.shape[0]

    predictive_distribution = np.zeros_like(log_likelihood)
    causal_posterior = np.zeros_like(log_likelihood)
    max_log_likelihood = np.nanmax(log_likelihood, axis=1, keepdims=True)
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood = np.clip(
        likelihood, a_min=np.nextafter(0.0, 1.0, dtype=np.float32), a_max=1.0
    )

    predictive_distribution[0] = initial_conditions
    causal_posterior[0] = initial_conditions * likelihood[0]
    norm = np.nansum(causal_posterior[0])
    marginal_likelihood = np.log(norm)
    causal_posterior[0] /= norm

    for t in range(1, n_time):
        # Predict
        predictive_distribution[t] = (
            get_transition_matrix(
                continuous_state_transitions, discrete_state_transitions, state_ind, t
            ).T
            @ causal_posterior[t - 1]
        )
        # Update
        causal_posterior[t] = predictive_distribution[t] * likelihood[t]
        # Normalize
        norm = np.nansum(causal_posterior[t])
        marginal_likelihood += np.log(norm)
        causal_posterior[t] /= norm

    marginal_likelihood += np.sum(max_log_likelihood)

    return causal_posterior, predictive_distribution, marginal_likelihood


def smoother(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    discrete_state_transitions: np.ndarray,
    continuous_state_transitions: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Acausal algorithm for computing the posterior distribution of the hidden states of a switching model

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    acausal_posterior, np.ndarray, shape (n_time, n_states)

    """
    n_time = causal_posterior.shape[0]

    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1]

    for t in range(n_time - 2, -1, -1):
        # Handle divide by zero
        relative_distribution = np.where(
            np.isclose(predictive_distribution[t + 1], 0.0),
            0.0,
            acausal_posterior[t + 1] / predictive_distribution[t + 1],
        )
        acausal_posterior[t] = causal_posterior[t] * (
            get_transition_matrix(
                continuous_state_transitions, discrete_state_transitions, state_ind, t
            )
            @ relative_distribution
        )
        acausal_posterior[t] /= acausal_posterior[t].sum()

    return acausal_posterior
