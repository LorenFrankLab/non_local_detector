"""Reference computations of EM's discrete-transition statistics and M-step.

The package computes the statistics in JAX: stationary counts from the
posteriors, and covariate responses in the smoother's scan.

``expanded_responses``, and ``expanded_counts`` for a time-varying transition,
are independent of that code: a NumPy loop over rows, so they can check it.
``expanded_counts`` for a stationary transition, ``dirichlet_gradient`` and
``dirichlet_hessian`` call package code. They give tests the package's
computation through a convenient signature and are not independent references;
compare them with finite differences or ``expanded_responses``.
"""

import numpy as np

from non_local_detector._transition_estimation import (
    _blocked_hessian,
    _blocked_loss_and_gradient,
    _penalty_matrix,
    _row_problem,
    _transition_blocks,
)
from non_local_detector.discrete_state_transitions import (
    _float64_jax,
    estimate_discrete_transition_counts_from_factorized_posteriors,
)


def expanded_responses(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Expected discrete transitions between consecutive rows, one row at a time.

    Parameters
    ----------
    causal_posterior, predictive_posterior, acausal_posterior : np.ndarray
        Shape (n_time, n_state_bins).
    transition_matrix : np.ndarray
        The joint transition between state bins, shape
        (n_state_bins, n_state_bins) or one per row,
        (n_time, n_state_bins, n_state_bins).
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete state of each state bin.

    Returns
    -------
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
    """
    aggregation = np.eye(int(np.max(state_ind)) + 1)[np.asarray(state_ind)]
    n_time = causal_posterior.shape[0]
    n_states = aggregation.shape[1]
    response = np.zeros((n_time - 1, n_states, n_states))

    for t in range(n_time - 1):
        ratio = np.divide(
            acausal_posterior[t + 1],
            predictive_posterior[t + 1],
            out=np.zeros_like(acausal_posterior[t + 1]),
            where=predictive_posterior[t + 1] != 0.0,
        )
        transition_t = (
            transition_matrix if transition_matrix.ndim == 2 else transition_matrix[t]
        )
        xi = causal_posterior[t, :, np.newaxis] * transition_t * ratio[np.newaxis, :]
        response[t] = aggregation.T @ xi @ aggregation

    return response


def expanded_counts(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Expected discrete transition counts for a joint transition between bins.

    A stationary joint transition goes through the package's count
    contraction, as a factorized transition whose discrete part is all ones;
    a time-varying one is summed from :func:`expanded_responses`.

    Returns
    -------
    counts : np.ndarray, shape (n_states, n_states)
    """
    if np.ndim(transition_matrix) == 3:
        return expanded_responses(
            causal_posterior,
            predictive_posterior,
            acausal_posterior,
            transition_matrix,
            state_ind,
        ).sum(axis=0)
    n_states = int(np.max(state_ind)) + 1
    return estimate_discrete_transition_counts_from_factorized_posteriors(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        transition_matrix,
        np.ones((n_states, n_states), dtype=np.asarray(transition_matrix).dtype),
        state_ind,
    )


def dirichlet_gradient(
    coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    alpha: float | np.ndarray = 1.0,
    l2_penalty: float = 0.1,
    smoothness: float = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of ``dirichlet_neg_log_likelihood``, as the M-step computes it."""
    with _float64_jax():
        return _blocked_loss_and_gradient(
            coefficients,
            *_row_problem(_transition_blocks(design_matrix), response, alpha),
            _penalty_matrix(
                np.shape(design_matrix)[1], l2_penalty, smoothness, smoothness_matrix
            ),
        )[1]


def dirichlet_hessian(
    coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    alpha: float | np.ndarray = 1.0,
    l2_penalty: float = 0.1,
    smoothness: float = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Hessian of ``dirichlet_neg_log_likelihood``, as the M-step computes it.

    Returns
    -------
    hessian : np.ndarray, shape (n_parameters, n_parameters)
        ``n_parameters = n_coefficients * (n_states - 1)``, in the order of
        the flattened coefficients.
    """
    with _float64_jax():
        return _blocked_hessian(
            coefficients,
            *_row_problem(_transition_blocks(design_matrix), response, alpha),
            _penalty_matrix(
                np.shape(design_matrix)[1], l2_penalty, smoothness, smoothness_matrix
            ),
        )
