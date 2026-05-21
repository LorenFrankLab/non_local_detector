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


def _validate_expanded_posterior_shapes(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    state_ind: np.ndarray,
) -> tuple[int, int]:
    """Validate expanded posterior shapes and return ``(n_time, n_state_bins)``."""
    if causal_posterior.ndim != 2:
        raise ValueError(
            "causal_posterior must have shape (n_time, n_state_bins), "
            f"got shape {causal_posterior.shape}"
        )

    n_time, n_state_bins = causal_posterior.shape
    expected_posterior_shape = (n_time, n_state_bins)
    for name, posterior in (
        ("predictive_posterior", predictive_posterior),
        ("acausal_posterior", acausal_posterior),
    ):
        if posterior.shape != expected_posterior_shape:
            raise ValueError(
                f"{name} must have shape {expected_posterior_shape}, "
                f"got shape {posterior.shape}"
            )

    if state_ind.shape != (n_state_bins,):
        raise ValueError(
            f"state_ind must have shape ({n_state_bins},), got shape {state_ind.shape}"
        )

    return n_time, n_state_bins


def _validate_expanded_transition_shape(
    transition_matrix: np.ndarray,
    n_time: int,
    n_state_bins: int,
) -> bool:
    """Validate expanded transition shape and return whether it is stationary."""
    if transition_matrix.ndim == 2:
        expected_shape = (n_state_bins, n_state_bins)
    elif transition_matrix.ndim == 3:
        expected_shape = (n_time, n_state_bins, n_state_bins)
    else:
        raise ValueError(
            "transition_matrix must have shape (n_state_bins, n_state_bins) or "
            "(n_time, n_state_bins, n_state_bins), "
            f"got shape {transition_matrix.shape}"
        )

    if transition_matrix.shape != expected_shape:
        raise ValueError(
            f"transition_matrix must have shape {expected_shape}, "
            f"got shape {transition_matrix.shape}"
        )

    return transition_matrix.ndim == 2


def _validate_continuous_transition_shape(
    continuous_transition_matrix: np.ndarray,
    n_state_bins: int,
) -> None:
    """Validate stationary continuous transition shape."""
    expected_shape = (n_state_bins, n_state_bins)
    if continuous_transition_matrix.shape != expected_shape:
        raise ValueError(
            f"continuous_transition_matrix must have shape {expected_shape}, "
            f"got shape {continuous_transition_matrix.shape}"
        )


def _validate_factorized_discrete_transition_shape(
    discrete_transition_matrix: np.ndarray,
    n_time: int,
    n_states: int,
    *,
    require_stationary: bool,
) -> bool:
    """Validate discrete transition shape and return whether it is stationary."""
    stationary_shape = (n_states, n_states)
    nonstationary_shape = (n_time, n_states, n_states)

    if require_stationary:
        if discrete_transition_matrix.shape != stationary_shape:
            raise ValueError(
                "discrete_transition_matrix must be stationary with shape "
                f"{stationary_shape}, got shape {discrete_transition_matrix.shape}"
            )
        return True

    if discrete_transition_matrix.ndim == 2:
        expected_shape = stationary_shape
    elif discrete_transition_matrix.ndim == 3:
        expected_shape = nonstationary_shape
    else:
        raise ValueError(
            "discrete_transition_matrix must have shape (n_states, n_states) or "
            "(n_time, n_states, n_states), "
            f"got shape {discrete_transition_matrix.shape}"
        )

    if discrete_transition_matrix.shape != expected_shape:
        raise ValueError(
            f"discrete_transition_matrix must have shape {expected_shape}, "
            f"got shape {discrete_transition_matrix.shape}"
        )

    return discrete_transition_matrix.ndim == 2


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
    target_masks: jnp.ndarray,
    source_state_ind: jnp.ndarray,
    n_states: int,
) -> jnp.ndarray:
    """Aggregate factorized expanded-bin pair probabilities by state."""
    # For each target state q, compute C @ (ratio * 1[state == q]).
    # This follows the filter/smoother pattern of applying the transition as a
    # dense operator to vectors, avoiding n_bins x n_bins pair-posterior
    # temporaries while preserving exact dense-C semantics.
    target_sum = continuous_transition_matrix @ (target_masks * ratio[jnp.newaxis, :]).T
    source_sum = jax.ops.segment_sum(
        causal_t[:, jnp.newaxis] * target_sum,
        source_state_ind,
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
    if is_factorized:
        target_masks = jax.nn.one_hot(
            state_ind,
            n_states,
            dtype=continuous_transition_matrix.dtype,
        ).T

    def aggregate(causal_t, predictive_next, acausal_next, transition_t):
        ratio = _safe_ratio_jax(acausal_next, predictive_next)
        if is_factorized:
            return _aggregate_factorized_xi_by_state_jax(
                causal_t,
                ratio,
                continuous_transition_matrix,
                transition_t,
                target_masks,
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
    causal_posterior = np.asarray(causal_posterior)
    predictive_posterior = np.asarray(predictive_posterior)
    acausal_posterior = np.asarray(acausal_posterior)
    state_ind = np.asarray(state_ind, dtype=int)
    n_time, n_state_bins = _validate_expanded_posterior_shapes(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        state_ind,
    )
    transition_matrix = np.asarray(transition_matrix)
    is_stationary = _validate_expanded_transition_shape(
        transition_matrix,
        n_time,
        n_state_bins,
    )
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
        is_stationary=is_stationary,
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
    causal_posterior = np.asarray(causal_posterior)
    predictive_posterior = np.asarray(predictive_posterior)
    acausal_posterior = np.asarray(acausal_posterior)
    state_ind = np.asarray(state_ind, dtype=int)
    n_time, n_state_bins = _validate_expanded_posterior_shapes(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        state_ind,
    )
    transition_matrix = np.asarray(transition_matrix)
    is_stationary = _validate_expanded_transition_shape(
        transition_matrix,
        n_time,
        n_state_bins,
    )
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
        is_stationary=is_stationary,
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

    This is the streaming equivalent of first constructing the full expanded
    transition from the stationary or time-varying discrete transition matrix
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
    discrete_transition_matrix : np.ndarray, shape (n_states, n_states) or
        (n_time, n_states, n_states)
        Stationary or time-varying discrete transition matrix.
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.

    Returns
    -------
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Exact expected transition counts from each discrete state to each
        discrete state at each time.
    """
    causal_posterior = np.asarray(causal_posterior)
    predictive_posterior = np.asarray(predictive_posterior)
    acausal_posterior = np.asarray(acausal_posterior)
    state_ind = np.asarray(state_ind, dtype=int)
    n_time, n_state_bins = _validate_expanded_posterior_shapes(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        state_ind,
    )
    continuous_transition_matrix = np.asarray(continuous_transition_matrix)
    _validate_continuous_transition_shape(
        continuous_transition_matrix,
        n_state_bins,
    )
    n_states = _n_states_from_state_ind(state_ind)
    discrete_transition_matrix = np.asarray(discrete_transition_matrix)
    is_stationary = _validate_factorized_discrete_transition_shape(
        discrete_transition_matrix,
        n_time,
        n_states,
        require_stationary=False,
    )
    continuous_transition_matrix = jnp.asarray(continuous_transition_matrix)
    discrete_transition_matrix = jnp.asarray(discrete_transition_matrix)
    response = _transition_pair_stats_jax(
        jnp.asarray(causal_posterior),
        jnp.asarray(predictive_posterior),
        jnp.asarray(acausal_posterior),
        discrete_transition_matrix,
        continuous_transition_matrix,
        jnp.asarray(state_ind),
        n_states,
        is_factorized=True,
        is_stationary=is_stationary,
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
    causal_posterior = np.asarray(causal_posterior)
    predictive_posterior = np.asarray(predictive_posterior)
    acausal_posterior = np.asarray(acausal_posterior)
    state_ind = np.asarray(state_ind, dtype=int)
    n_time, n_state_bins = _validate_expanded_posterior_shapes(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        state_ind,
    )
    continuous_transition_matrix = np.asarray(continuous_transition_matrix)
    _validate_continuous_transition_shape(
        continuous_transition_matrix,
        n_state_bins,
    )
    n_states = _n_states_from_state_ind(state_ind)
    discrete_transition_matrix = np.asarray(discrete_transition_matrix)
    _validate_factorized_discrete_transition_shape(
        discrete_transition_matrix,
        n_time,
        n_states,
        require_stationary=True,
    )
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


def estimate_non_stationary_state_transition_from_responses(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 1e-5,
    optimization_method: str = "L-BFGS-B",
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
        Optimization method for `scipy.optimize.minimize`, by default "L-BFGS-B".
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
        objective_args = (
            design_matrix[:-1],
            response[:, from_state, :],
            row_alpha,
            transition_regularization,
        )
        options = {"maxiter": maxiter}
        if optimization_method != "L-BFGS-B":
            options["disp"] = disp
        minimize_kwargs = {
            "fun": dirichlet_neg_log_likelihood,
            "x0": transition_coefficients[:, from_state].ravel(),
            "method": optimization_method,
            "jac": dirichlet_gradient,
            "args": objective_args,
            "options": options,
        }
        if optimization_method in {"Newton-CG", "trust-ncg", "dogleg", "trust-exact"}:
            minimize_kwargs["hess"] = dirichlet_hessian

        result = minimize(**minimize_kwargs)

        use_result = result.success
        if not result.success and hasattr(result, "fun"):
            initial_loss = float(
                dirichlet_neg_log_likelihood(
                    transition_coefficients[:, from_state].ravel(),
                    *objective_args,
                )
            )
            result_loss = float(result.fun)
            use_result = np.isfinite(result_loss) and result_loss < initial_loss

        if not use_result:
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
            if not result.success:
                logger.warning(
                    "Transition optimization did not report convergence for state %d: "
                    "%s. Using improved coefficients.",
                    from_state,
                    result.message,
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

    return estimated_transition_coefficients, estimated_transition_matrix


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
        Dimensionless weight for data-adaptive prior scaling. When > 0,
        the effective prior pseudo-counts for that row are scaled by the
        expected transition count ``N_i = joint_sum[i].sum()``, making the
        prior influence approximately invariant to the number of time bins.

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


def _estimate_discrete_transition(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    continuous_transition: np.ndarray,
    state_ind: np.ndarray,
    discrete_transition: np.ndarray,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
    transition_concentration: float,
    transition_stickiness: float | np.ndarray,
    transition_regularization: float,
    transition_prior_weight: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the discrete transition matrix (stationary or non-stationary).

    Always uses the exact expanded-state M-step that aggregates bin-level
    pair posteriors into discrete-state transition counts/responses.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Expanded-bin filtered posterior P(X_t | y_{1:t}).
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        Expanded-bin one-step predictive posterior P(X_{t+1} | y_{1:t}).
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        Expanded-bin smoothed posterior P(X_t | y_{1:T}).
    continuous_transition : np.ndarray, shape (n_state_bins, n_state_bins)
        Continuous transition matrix over expanded state bins.
    state_ind : np.ndarray, shape (n_state_bins,)
        Discrete-state index for each expanded state bin.
    discrete_transition : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
        Current discrete transition matrix.
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

    Returns
    -------
    estimated_discrete_transition : np.ndarray
        Updated transition matrix.
    estimated_discrete_transition_coefficients : np.ndarray | None
        Updated coefficients (if non-stationary).
    """
    is_nonstationary = (
        discrete_transition_coefficients is not None
        and discrete_transition_design_matrix is not None
    )

    if is_nonstationary:
        response = estimate_discrete_transition_responses_from_factorized_posteriors(
            causal_posterior,
            predictive_posterior,
            acausal_posterior,
            continuous_transition,
            discrete_transition,
            state_ind,
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
        if (
            isinstance(transition_stickiness, np.ndarray)
            and transition_stickiness.size == 1
        ):
            stickiness_value = float(transition_stickiness.item())
        else:
            stickiness_value = transition_stickiness

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

    return discrete_transition, discrete_transition_coefficients


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
