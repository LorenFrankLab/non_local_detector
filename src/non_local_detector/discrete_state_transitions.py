"""Discrete state transition models and their estimation.

This module defines the classes that specify the initial discrete-state
transition matrix of a detector (stationary diagonal or custom matrices, and
non-stationary matrices that depend on covariates through a design-matrix
formula) together with the M-step routines that estimate them from expected
transition counts, including Dirichlet priors, stickiness, and penalized
(optionally spline-smoothed) multinomial regression on covariates.
"""

import logging
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Protocol, TypeAlias, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from jax.nn import log_softmax
from jax.typing import ArrayLike
from patsy import (
    DesignInfo,
    DesignMatrix,
    build_design_matrices,
    dmatrix,
    stateful_transform,
)
from patsy.splines import BS
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, softmax, xlogy
from scipy.stats import norm

from non_local_detector import _validation as val
from non_local_detector.exceptions import ValidationError

__all__ = [
    "DiscreteNonStationaryCustom",
    "DiscreteNonStationaryDiagonal",
    "DiscreteStationaryCustom",
    "DiscreteStationaryDiagonal",
    "DiscreteTransition",
    "InitialDiscreteTransition",
    "cyclic_bs",
    "quantile_bs",
]


logger = logging.getLogger(__name__)

# ``(design, response)`` blocks of one transition row; see ``_row_problem``.
_Blocks: TypeAlias = list[tuple[jax.Array, jax.Array]]


def _float64_jax() -> AbstractContextManager[Any]:
    """A context in which JAX computes in float64.

    The scoped switch is ``jax.enable_x64`` in current JAX and
    ``jax.experimental.enable_x64`` in older releases such as 0.6.
    """
    if hasattr(jax, "enable_x64"):
        return cast(AbstractContextManager[Any], jax.enable_x64(True))
    from jax.experimental import enable_x64

    return cast(AbstractContextManager[Any], enable_x64(True))


def centered_softmax_forward(y: np.ndarray) -> np.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Parameters
    ----------
    y : np.ndarray, shape (n_states,) or (n_samples, n_states)
        The input values. Only 1-D and 2-D inputs are supported;
        the last dimension is the state dimension.

    Returns
    -------
    softmax : np.ndarray, shape (n_states + 1,) or (n_samples, n_states + 1)
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

    return cast(np.ndarray, softmax(y, axis=-1))


def centered_softmax_inverse(y: np.ndarray) -> np.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Parameters
    ----------
    y : np.ndarray, shape (n_states + 1,) or (n_samples, n_states + 1)
        The softmax values.

    Returns
    -------
    inverse : np.ndarray, shape (n_states,) or (n_samples, n_states)
        The inverse of the softmax values

    Notes
    -----
    The last state is the reference category of the centered parameterization.
    If it has (near) zero probability, ``log(tiny) ≈ -708`` produces intercepts
    near +708 that can stall an L-BFGS warm start; keep nonzero mass there.

    Example
    -------
    >>> y = np.asarray([0.2, 0.3, 0.4, 0.1])
    >>> np.allclose(np.exp(centered_softmax_inverse(y)), np.asarray([2,3,4]))
    True
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        # np.finfo requires a floating dtype; promote integer/other inputs.
        y = y.astype(float)
    EPS = np.finfo(y.dtype).tiny
    y_safe = np.clip(y, EPS, None)
    return cast(np.ndarray, np.log(y_safe[..., :-1]) - np.log(y_safe[..., [-1]]))


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


def _host_or_device(posterior: ArrayLike) -> np.ndarray | jax.Array:
    """A posterior as an array, leaving a device array on the device.

    Converting a JAX array with ``np.asarray`` would copy it to the host only
    for the estimators to upload it again.
    """
    return posterior if isinstance(posterior, jax.Array) else np.asarray(posterior)


def _validate_expanded_posterior_shapes(
    causal_posterior: np.ndarray | jax.Array,
    predictive_posterior: np.ndarray | jax.Array,
    acausal_posterior: np.ndarray | jax.Array | jax.Array,
    state_ind: np.ndarray,
) -> tuple[int, int]:
    """Validate expanded posterior shapes and return ``(n_time, n_state_bins)``."""
    if causal_posterior.ndim != 2:
        msg = (
            "causal_posterior must have shape (n_time, n_state_bins), "
            f"got shape {causal_posterior.shape}"
        )
        raise ValueError(msg)

    n_time, n_state_bins = causal_posterior.shape
    expected_posterior_shape = (n_time, n_state_bins)
    for name, posterior in (
        ("predictive_posterior", predictive_posterior),
        ("acausal_posterior", acausal_posterior),
    ):
        if posterior.shape != expected_posterior_shape:
            msg = (
                f"{name} must have shape {expected_posterior_shape}, "
                f"got shape {posterior.shape}"
            )
            raise ValueError(msg)

    if state_ind.shape != (n_state_bins,):
        msg = (
            f"state_ind must have shape ({n_state_bins},), got shape {state_ind.shape}"
        )
        raise ValueError(msg)

    return n_time, n_state_bins


def _validate_expanded_transition_shape(
    transition_matrix: np.ndarray,
    n_time: int,
    n_state_bins: int,
) -> bool:
    """Validate expanded transition shape and return whether it is stationary."""
    expected_shape: tuple[int, ...]
    if transition_matrix.ndim == 2:
        expected_shape = (n_state_bins, n_state_bins)
    elif transition_matrix.ndim == 3:
        expected_shape = (n_time, n_state_bins, n_state_bins)
    else:
        msg = (
            "transition_matrix must have shape (n_state_bins, n_state_bins) or "
            "(n_time, n_state_bins, n_state_bins), "
            f"got shape {transition_matrix.shape}"
        )
        raise ValueError(msg)

    if transition_matrix.shape != expected_shape:
        msg = (
            f"transition_matrix must have shape {expected_shape}, "
            f"got shape {transition_matrix.shape}"
        )
        raise ValueError(msg)

    return transition_matrix.ndim == 2


def _validate_continuous_transition_shape(
    continuous_transition_matrix: np.ndarray,
    n_state_bins: int,
) -> None:
    """Validate stationary continuous transition shape."""
    expected_shape = (n_state_bins, n_state_bins)
    if continuous_transition_matrix.shape != expected_shape:
        msg = (
            f"continuous_transition_matrix must have shape {expected_shape}, "
            f"got shape {continuous_transition_matrix.shape}"
        )
        raise ValueError(msg)


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
            msg = (
                "discrete_transition_matrix must be stationary with shape "
                f"{stationary_shape}, got shape {discrete_transition_matrix.shape}"
            )
            raise ValueError(msg)
        return True

    expected_shape: tuple[int, ...]
    if discrete_transition_matrix.ndim == 2:
        expected_shape = stationary_shape
    elif discrete_transition_matrix.ndim == 3:
        expected_shape = nonstationary_shape
    else:
        msg = (
            "discrete_transition_matrix must have shape (n_states, n_states) or "
            "(n_time, n_states, n_states), "
            f"got shape {discrete_transition_matrix.shape}"
        )
        raise ValueError(msg)

    if discrete_transition_matrix.shape != expected_shape:
        msg = (
            f"discrete_transition_matrix must have shape {expected_shape}, "
            f"got shape {discrete_transition_matrix.shape}"
        )
        raise ValueError(msg)

    return discrete_transition_matrix.ndim == 2


def _aggregate_xi_by_state_jax(
    xi: jax.Array, state_ind: jax.Array, n_states: int
) -> jax.Array:
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
    causal_t: jax.Array,
    ratio: jax.Array,
    continuous_transition_matrix: jax.Array,
    discrete_transition_matrix: jax.Array,
    target_masks: jax.Array,
    source_state_ind: jax.Array,
    n_states: int,
) -> jax.Array:
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


def _safe_ratio_jax(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Return numerator / denominator with zero output at exactly zero denominators."""
    # Match core._divide_safe: only exact zeros are treated as unreachable bins.
    # An `isclose` threshold would zero legitimately small predictive
    # probabilities whose acausal/predictive ratio (the smoother backward
    # message) is O(1), silently dropping real transitions from the M-step.
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    return jnp.where(denominator != 0.0, numerator / safe_denominator, 0.0)


def _assert_map_compatible_alpha(alpha: np.ndarray) -> None:
    if np.any(alpha < 1.0):
        msg = (
            "concentration and stickiness must produce prior parameters >= 1.0 "
            "for this MAP transition update"
        )
        raise ValueError(msg)


# Bytes of each posterior row block copied to the device for stationary
# transition counts, so device memory does not grow with recording length.
_PAIR_SUM_BLOCK_BYTES = 256 * 1024**2


# Rows summed per partial product of the count contraction. A single product
# over many similar terms accumulates float32 rounding in proportion to the
# rows (5.7e-4 relative over 50,000 rows); short partial sums added together
# keep it near the single-sum bound for this many rows.
_CONTRACTION_SUB_ROWS = 256


# Transitions per block of the covariate M-step loss and its log prior, so
# their float64 temporaries stay bounded however long the recording is.
_COVARIATE_BLOCK_ROWS = 65_536


def _blocked_contraction(left: jax.Array, right: jax.Array) -> jax.Array:
    """``left.T @ right`` summed over the rows in short partial products.

    Parameters
    ----------
    left, right : jax.Array, shape (n_rows, n_bins)

    Returns
    -------
    product : jax.Array, shape (n_bins, n_bins)
    """
    n_rows, n_bins = left.shape
    padding = -n_rows % _CONTRACTION_SUB_ROWS
    left = jnp.pad(left, ((0, padding), (0, 0)))
    right = jnp.pad(right, ((0, padding), (0, 0)))

    def add(block: jax.Array, total: jax.Array) -> jax.Array:
        start = block * _CONTRACTION_SUB_ROWS
        return total + jnp.matmul(
            jax.lax.dynamic_slice_in_dim(left, start, _CONTRACTION_SUB_ROWS).T,
            jax.lax.dynamic_slice_in_dim(right, start, _CONTRACTION_SUB_ROWS),
            precision=jax.lax.Precision.HIGHEST,
        )

    return cast(
        jax.Array,
        jax.lax.fori_loop(
            0,
            (n_rows + padding) // _CONTRACTION_SUB_ROWS,
            add,
            jnp.zeros((n_bins, n_bins), jnp.result_type(left, right)),
        ),
    )


@partial(jax.jit, static_argnames=("n_states",))
def _stationary_counts_block(
    causal_posterior: jax.Array,
    predictive_next: jax.Array,
    acausal_next: jax.Array,
    transition_matrix: jax.Array,
    state_ind: jax.Array,
    n_states: int,
) -> jax.Array:
    """Expected discrete transition counts over a block of consecutive rows.

    A stationary transition factors out of the sum over rows:
    ``sum_t causal_t[i] T[i, j] ratio_{t+1}[j] = T[i, j] * pair_sum[i, j]``,
    so the block needs one contraction over rows instead of a scan with one
    step per row. The ratios ``acausal / predictive`` can approach
    ``1 / T[i, j]``, so their sum over rows can overflow although each weighted
    term is at most one. A ratio column whose sum could exceed the dtype's range
    is divided by the smallest power of two that keeps it within range, and
    multiplied back only after the transition weights it. Powers of two divide
    exactly, and only columns near overflow are scaled, so a rare transition
    into one bin does not scale, or underflow, the ratios of other bins, as a
    single global scale would.

    Parameters
    ----------
    causal_posterior : jax.Array, shape (n_rows, n_state_bins)
        Filtered posterior at times ``t``.
    predictive_next, acausal_next : jax.Array, shape (n_rows, n_state_bins)
        Predictive and smoothed posteriors at times ``t + 1``.
    transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        The expanded stationary transition.
    state_ind : jax.Array, shape (n_state_bins,)
    n_states : int

    Returns
    -------
    counts : jax.Array, shape (n_states, n_states)
    """
    ratio = _safe_ratio_jax(acausal_next, predictive_next)
    maximum = jnp.max(ratio, axis=0, initial=0.0)
    # The column sum is at most n_rows * maximum; keep it 256x below the
    # largest finite value.
    headroom = jnp.finfo(ratio.dtype).max / 256  # type: ignore[no-untyped-call]  # jax.numpy.finfo is untyped
    exponent = jnp.ceil(
        jnp.log2(jnp.maximum(maximum / headroom * causal_posterior.shape[0], 1.0))
    )
    # Non-finite ratios propagate unscaled, as in an unscaled sum.
    exponent = jnp.where(jnp.isfinite(exponent), exponent, 0.0)
    scale = jnp.ldexp(jnp.ones((), ratio.dtype), exponent.astype(jnp.int32))
    contraction = _blocked_contraction(causal_posterior, ratio / scale)
    # Weight first: the weighted terms are bounded, the unweighted sums are not.
    weighted = jax.lax.optimization_barrier(transition_matrix * contraction)  # type: ignore[no-untyped-call]  # jax.lax.optimization_barrier is untyped
    return _aggregate_xi_by_state_jax(weighted * scale, state_ind, n_states)


def _stationary_counts(
    causal_posterior: np.ndarray | jax.Array,
    predictive_posterior: np.ndarray | jax.Array,
    acausal_posterior: np.ndarray | jax.Array | jax.Array,
    transition_matrix: jax.Array,
    state_ind: np.ndarray | jax.Array,
    n_states: int,
) -> jax.Array:
    """``_stationary_counts_block`` over all consecutive rows, in byte-bounded blocks.

    Parameters
    ----------
    causal_posterior, predictive_posterior, acausal_posterior : np.ndarray or jax.Array
        Shape (n_time, n_state_bins). Host arrays are uploaded block by block.
    transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        The expanded stationary transition.
    state_ind : np.ndarray, shape (n_state_bins,)
    n_states : int

    Returns
    -------
    counts : jax.Array, shape (n_states, n_states)
    """
    n_time, n_state_bins = causal_posterior.shape
    itemsize = np.result_type(causal_posterior.dtype, acausal_posterior.dtype).itemsize
    rows = max(1, _PAIR_SUM_BLOCK_BYTES // (max(n_state_bins, 1) * itemsize))
    state_ind = jnp.asarray(state_ind)
    counts = jnp.zeros(
        (n_states, n_states),
        dtype=jnp.result_type(causal_posterior, acausal_posterior, transition_matrix),
    )
    for start in range(0, n_time - 1, rows):
        stop = min(start + rows, n_time - 1)
        counts = counts + _stationary_counts_block(
            jnp.asarray(causal_posterior[start:stop]),
            jnp.asarray(predictive_posterior[start + 1 : stop + 1]),
            jnp.asarray(acausal_posterior[start + 1 : stop + 1]),
            transition_matrix,
            state_ind,
            n_states,
        )
    return counts


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
    causal_posterior: jax.Array,
    predictive_posterior: jax.Array,
    acausal_posterior: jax.Array,
    transition_matrix: jax.Array,
    continuous_transition_matrix: jax.Array,
    state_ind: jax.Array,
    n_states: int,
    *,
    is_factorized: bool,
    is_stationary: bool,
    return_time_series: bool,
) -> jax.Array:
    """JAX scan kernel for per-row responses or non-stationary counts.

    Stationary counts use :func:`_stationary_counts` instead.
    """
    if is_stationary and not return_time_series:
        msg = "stationary counts use _stationary_counts, not this scan"
        raise ValueError(msg)
    if is_factorized:
        target_masks = jax.nn.one_hot(
            state_ind,
            n_states,
            dtype=continuous_transition_matrix.dtype,
        ).T

    def aggregate(
        causal_t: jax.Array,
        predictive_next: jax.Array,
        acausal_next: jax.Array,
        transition_t: jax.Array,
    ) -> jax.Array:
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

            def step(_: Any, inputs: tuple[Any, ...]) -> tuple[None, jax.Array]:
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

            def step(_: Any, inputs: tuple[Any, ...]) -> tuple[None, jax.Array]:
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

    def accumulate(
        joint_sum: jax.Array, inputs: tuple[Any, ...]
    ) -> tuple[jax.Array, None]:
        causal_t, predictive_next, acausal_next, transition_t = inputs
        counts_t = aggregate(
            causal_t,
            predictive_next,
            acausal_next,
            transition_t,
        )
        return joint_sum + counts_t, None

    joint_sum, _ = jax.lax.scan(
        accumulate,
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
    transition_matrix: np.ndarray | jax.Array,
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
    transition_matrix: np.ndarray | jax.Array,
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
    if is_stationary:
        joint_sum = _stationary_counts(
            causal_posterior,
            predictive_posterior,
            acausal_posterior,
            transition_matrix,
            state_ind,
            n_states,
        )
        return np.asarray(joint_sum)
    joint_sum = _transition_pair_stats_jax(
        jnp.asarray(causal_posterior),
        jnp.asarray(predictive_posterior),
        jnp.asarray(acausal_posterior),
        transition_matrix,
        jnp.empty((0, 0), dtype=transition_matrix.dtype),
        jnp.asarray(state_ind),
        n_states,
        is_factorized=False,
        is_stationary=False,
        return_time_series=False,
    )

    return np.asarray(joint_sum)


def estimate_discrete_transition_responses_from_factorized_posteriors(
    causal_posterior: np.ndarray | jax.Array,
    predictive_posterior: np.ndarray | jax.Array,
    acausal_posterior: np.ndarray | jax.Array,
    continuous_transition_matrix: np.ndarray | jax.Array,
    discrete_transition_matrix: np.ndarray | jax.Array,
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
    causal_posterior = _host_or_device(causal_posterior)
    predictive_posterior = _host_or_device(predictive_posterior)
    acausal_posterior = _host_or_device(acausal_posterior)
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
    causal_posterior: np.ndarray | jax.Array,
    predictive_posterior: np.ndarray | jax.Array,
    acausal_posterior: np.ndarray | jax.Array,
    continuous_transition_matrix: np.ndarray | jax.Array,
    discrete_transition_matrix: np.ndarray | jax.Array,
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
    causal_posterior = _host_or_device(causal_posterior)
    predictive_posterior = _host_or_device(predictive_posterior)
    acausal_posterior = _host_or_device(acausal_posterior)
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
    joint_sum = _stationary_counts(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        jnp.asarray(continuous_transition_matrix)
        * jnp.asarray(discrete_transition_matrix)[jnp.ix_(state_ind, state_ind)],
        state_ind,
        n_states,
    )
    return np.asarray(joint_sum)


@jax.jit
def jax_centered_log_softmax_forward(y: jax.Array) -> jax.Array:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    The 1D and 2D input branches compile as separate JAX specializations.
    Only 1-D and 2-D inputs are supported.

    Parameters
    ----------
    y : jax.Array, shape (n_states,) or (n_samples, n_states)
        The input values.

    Returns
    -------
    log_softmax : jax.Array, shape (n_states + 1,) or (n_samples, n_states + 1)
        The log softmax of the input values.

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
    """Create a Dirichlet prior for the transition matrix rows.

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
    if np.ndim(stickiness) == 0:
        # Scalar (Python or NumPy, including 0-d arrays): same value all states.
        stickiness_arr = float(stickiness) * np.eye(n_states)
    else:
        # Assume stickiness provided per state
        stickiness_arr = np.diag(stickiness)
    # Dirichlet parameters must be strictly positive. MAP transition estimators
    # that add alpha - 1 as pseudo-counts separately reject alpha < 1.
    return np.maximum(concentration * np.ones((n_states,)) + stickiness_arr, 1e-10)


class CovariateTransitionFit(NamedTuple):
    """A covariate-dependent discrete transition fit by
    :func:`_fit_covariate_transition`.

    Attributes
    ----------
    coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
    transition : np.ndarray, shape (n_time, n_states, n_states)
    smoothness : np.ndarray, shape (n_states, n_terms)
        Curvature-penalty strength of each row for each spline term.
    covariance : np.ndarray, shape (n_states, n_parameters, n_parameters)
        Laplace (posterior) covariance of each row's flattened coefficients,
        ``n_parameters = n_coefficients * (n_states - 1)``: the inverse
        Hessian of the row's penalized loss at the fit. Zero for frozen rows;
        NaN for a row that is not identified (no transitions, or a singular
        Hessian).
    """

    coefficients: np.ndarray
    transition: np.ndarray
    smoothness: np.ndarray
    covariance: np.ndarray | None


# REML searches the curvature-penalty strength over this grid of log10
# strengths, then refines the best grid point within one decade to this
# log10 tolerance. Above 1e6 the fitted curve is already its linear limit
# (transitions move by < 0.002) and the LAML is flat (within 4e-4 nats to
# 1e7 in one test). From about 3e7 the inner fit stops converging (its
# Hessian's condition number passes 1e14), so the LAML it gives is wrong:
# by 0.2 nats at 3e8 and 37 at 1e10 in that test.
# The grid reaches 1e-12 because the penalty is on the covariate rescaled
# between the spline's boundaries: a boundary far beyond the data (e.g. set
# by tracking glitches) moves the useful strengths down by decades, and the
# LAML can have maxima there and near 1 (0.2 nats apart in one test).
# Per-term searches around a current value scan down to the floor too.
_REML_LOG10_FLOOR = -12.0
_REML_LOG10_GRID = np.arange(_REML_LOG10_FLOOR, 6.5, 1.0)
_REML_LOG10_TOLERANCE = 0.02
# With several spline terms, REML then refines each term's strength in turn,
# in at most this many sweeps over the terms.
_REML_MAX_SWEEPS = 5


def estimate_non_stationary_state_transition_from_responses(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 0.1,
    optimization_method: str = "trust-exact",
    maxiter: int | None = 100,
    disp: bool = False,
    frozen_rows: np.ndarray | None = None,
    smoothness: float | np.ndarray | str = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a non-stationary transition model from expected counts.

    Each row maximizes its expected complete-data log-likelihood, summed over
    time bins, plus a Dirichlet prior counted once per row (as for stationary
    transitions, so ``stickiness`` means the same pseudo-counts for both), an
    L2 penalty and an optional curvature penalty, in float64. See
    :func:`dirichlet_neg_log_likelihood`.

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
        L2 penalty on coefficients (excluding intercept), by default 0.1, as
        for the detectors' ``discrete_transition_regularization``.
    optimization_method : str, optional
        Optimization method for `scipy.optimize.minimize`, by default
        "trust-exact": Newton steps with the exact float64 Hessian, which
        converge in a few iterations on spline designs whose Hessians have
        condition numbers of 1e5 to 1e7, where "L-BFGS-B" stops short.
    maxiter : int, optional
        Maximum iterations for optimizer, by default 100.
    disp : bool, optional
        Display optimizer convergence messages, by default False.
    frozen_rows : np.ndarray of bool, shape (n_states,), optional
        Rows that keep their ``transition_coefficients`` instead of being
        estimated. None (default) estimates every row.
    smoothness : float, np.ndarray, or "reml", optional
        Strength of the curvature penalty on each ``quantile_bs`` and
        ``cyclic_bs`` term: one for every row and term, one per row (shape
        (n_states,)), or one per row and term (shape (n_states, n_terms)); by
        default 0.0 (none). "reml" chooses each row's strength for each term
        by maximizing the row's Laplace approximate marginal likelihood.
    smoothness_matrix : np.ndarray, optional
        Curvature penalties from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients), needed when
        ``smoothness`` is not 0.

    Returns
    -------
    estimated_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Optimized transition coefficients.
    estimated_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Resulting non-stationary transition matrix.
    """
    fit = _fit_covariate_transition(
        transition_coefficients,
        design_matrix,
        response,
        concentration=concentration,
        stickiness=stickiness,
        transition_regularization=transition_regularization,
        smoothness=smoothness,
        smoothness_matrix=smoothness_matrix,
        optimization_method=optimization_method,
        maxiter=maxiter,
        disp=disp,
        frozen_rows=frozen_rows,
        compute_covariance=False,
    )
    return fit.coefficients, fit.transition


def _term_penalties(
    smoothness_matrix: np.ndarray | None, n_coefficients: int
) -> np.ndarray:
    """Curvature penalties as a stack, one per spline term.

    Parameters
    ----------
    smoothness_matrix : np.ndarray or None
        A stack from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients); one term's matrix, shape
        (n_coefficients, n_coefficients); or None (no spline terms).
    n_coefficients : int

    Returns
    -------
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
    """
    if smoothness_matrix is None:
        return np.zeros((0, n_coefficients, n_coefficients))
    return np.reshape(
        np.asarray(smoothness_matrix, dtype=np.float64),
        (-1, n_coefficients, n_coefficients),
    )


def _row_strengths(
    smoothness: float | np.ndarray, n_states: int, n_terms: int
) -> np.ndarray:
    """Curvature-penalty strengths as (n_states, n_terms): a scalar for every
    row and term, shape (n_states,) per row, or (n_states, n_terms)."""
    values = np.asarray(smoothness, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return np.broadcast_to(values, (n_states, n_terms)).astype(np.float64)


def _validate_smoothness(smoothness: float | np.ndarray | str) -> bool:
    """Check a smoothness value: strengths that are finite and >= 0, or
    ``"reml"``. Returns whether REML estimates the strengths."""
    if isinstance(smoothness, str):
        if smoothness != "reml":
            msg = f'smoothness must be a float >= 0 or "reml", got {smoothness!r}.'
            raise ValueError(msg)
        return True
    if np.any(~np.isfinite(smoothness) | (np.asarray(smoothness) < 0.0)):
        msg = f"smoothness must be finite and >= 0, got {smoothness}."
        raise ValueError(msg)
    return False


def _require_spline_term(
    smoothness: float | np.ndarray | str, smoothness_matrix: np.ndarray | None
) -> None:
    """Check that a smoothness penalty (REML, or a positive strength) has a
    spline term to penalize."""
    penalized = isinstance(smoothness, str) or np.any(np.asarray(smoothness) > 0.0)
    if penalized and (smoothness_matrix is None or not np.any(smoothness_matrix)):
        msg = (
            "A smoothness penalty needs a quantile_bs or cyclic_bs term in the "
            "transition formula, e.g. '1 + quantile_bs(speed)'."
        )
        raise ValueError(msg)


def _row_smoothness(
    smoothness: float | np.ndarray | str,
    initial_smoothness: np.ndarray | None,
    smoothness_matrix: np.ndarray | None,
    n_states: int,
) -> tuple[np.ndarray, bool]:
    """Validated per-row, per-term curvature-penalty strengths to start from.

    Returns
    -------
    smoothness : np.ndarray, shape (n_states, n_terms)
        The fixed strengths, or the REML starting strengths
        (``initial_smoothness``, else 1).
    estimate : bool
        Whether REML estimates the strengths.
    """
    estimate = _validate_smoothness(smoothness)
    _require_spline_term(smoothness, smoothness_matrix)
    if estimate:
        values = 1.0 if initial_smoothness is None else initial_smoothness
        _validate_smoothness(values)
    else:
        values = cast(float | np.ndarray, smoothness)
    n_terms = (
        0
        if smoothness_matrix is None
        else _term_penalties(smoothness_matrix, np.shape(smoothness_matrix)[-1]).shape[
            0
        ]
    )
    return _row_strengths(values, n_states, n_terms), estimate


def _check_smoothness_design(
    smoothness: float | str, design_matrix: DesignMatrix
) -> None:
    """Check that a transition formula supports its smoothness penalty.

    A penalty needs a spline term, and an intercept as the first column: the
    L2 penalty leaves the first column unpenalized and REML counts the
    penalized directions assuming it is the intercept.
    """
    _validate_smoothness(smoothness)
    _require_spline_term(smoothness, _smoothness_matrix(design_matrix.design_info))
    if (isinstance(smoothness, str) or smoothness != 0.0) and (
        design_matrix.design_info.column_names[0] != "Intercept"
    ):
        msg = (
            "A smoothness penalty needs an intercept as the first column of the "
            "transition formula, e.g. '1 + quantile_bs(speed)', got columns "
            f"{design_matrix.design_info.column_names}."
        )
        raise ValueError(msg)


def _fit_covariate_transition(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 0.1,
    smoothness: float | np.ndarray | str = 0.0,
    smoothness_matrix: np.ndarray | None = None,
    initial_smoothness: np.ndarray | None = None,
    optimization_method: str = "trust-exact",
    maxiter: int | None = 100,
    disp: bool = False,
    frozen_rows: np.ndarray | None = None,
    compute_covariance: bool = True,
) -> CovariateTransitionFit:
    """Fit a covariate-dependent transition, its smoothness and covariance.

    With ``smoothness="reml"``, each row's curvature-penalty strengths, one
    per spline term, maximize the row's Laplace approximate marginal
    likelihood (LAML, the
    REML criterion of Wood, 2011, J. R. Stat. Soc. B 73:3 for non-Gaussian
    models) ``-L(beta) + log|Q|_+ / 2 - log|A| / 2``, with ``beta`` the
    penalized fit, ``L`` the penalized loss, ``Q`` the prior precision (twice
    the penalty) and ``A`` the Hessian of ``L``, given the E-step's expected
    transitions. The LAML can have more than one local maximum (the L2
    penalty and the curvature penalty both shrink the curve), so the
    strength is searched over a grid of decades and then refined. The L2
    penalty stays fixed and keeps rare transitions from running off along the
    curvature penalty's unpenalized linear trends. See :func:`_reml_row` for
    the search over several terms' strengths.

    Parameters
    ----------
    transition_coefficients, design_matrix, response, concentration, stickiness,
    transition_regularization, optimization_method, maxiter, disp, frozen_rows,
    smoothness, smoothness_matrix
        As in :func:`estimate_non_stationary_state_transition_from_responses`.
    initial_smoothness : np.ndarray, shape (n_states, n_terms), optional
        Strengths kept by rows REML does not fit (frozen rows), e.g. the
        previous EM iteration's; by default 1. REML searches its full range
        for every fitted row. Ignored for fixed ``smoothness``.
    compute_covariance : bool, optional
        Whether to compute the fit's covariance, by default True. EM skips
        it in every M-step and computes it once, after its last, with
        :func:`_covariate_transition_covariance`.

    Returns
    -------
    fit : CovariateTransitionFit
        ``covariance`` is None without ``compute_covariance``.
    """
    n_coefficients, n_states = transition_coefficients.shape[:2]
    estimated_transition_coefficients = np.zeros(
        (n_coefficients, n_states, (n_states - 1))
    )

    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    n_time = design_matrix.shape[0]
    estimated_transition_matrix = np.zeros((n_time, n_states, n_states))

    alpha = get_transition_prior(concentration, stickiness, n_states)
    _assert_map_compatible_alpha(alpha)
    row_smoothness, estimate_smoothness = _row_smoothness(
        smoothness, initial_smoothness, smoothness_matrix, n_states
    )

    options: dict[str, Any] = {}
    if maxiter is not None:
        options["maxiter"] = maxiter
    # scipy deprecated `disp`/`iprint` for L-BFGS-B (removal in 1.18), so
    # only forward `disp` to methods that still accept it.
    if optimization_method != "L-BFGS-B":
        options["disp"] = disp
    else:
        # If L-BFGS-B is requested, stop on the projected gradient, not on
        # the relative change of the summed (large) objective.
        options["ftol"] = 0.0
        options["gtol"] = 1e-7

    penalties = _term_penalties(smoothness_matrix, n_coefficients)

    def penalty(strengths: np.ndarray) -> np.ndarray:
        return _penalty_matrix(
            n_coefficients, transition_regularization, strengths, penalties
        )

    with _float64_jax():
        # Every row's fit shares the design matrix's blocks.
        design_blocks = _transition_blocks(design_matrix[:-1])
        for from_state, row_alpha in enumerate(alpha):
            row_coefficients = transition_coefficients[:, from_state, :]
            if frozen_rows is None or not frozen_rows[from_state]:
                blocks, prior_per_bin = _row_problem(
                    design_blocks, response[:, from_state, :], row_alpha
                )

                def fit_row(
                    start: np.ndarray,
                    strengths: np.ndarray,
                    blocks: _Blocks = blocks,
                    prior_per_bin: jax.Array = prior_per_bin,
                    from_state: int = from_state,
                    warn: bool = True,
                ) -> np.ndarray:
                    return _fit_covariate_row(
                        start,
                        blocks,
                        prior_per_bin,
                        penalty(strengths),
                        optimization_method,
                        options,
                        from_state,
                        warn,
                    )

                strength = row_smoothness[from_state]
                if estimate_smoothness and blocks:
                    fitted, strength = _reml_row(
                        row_coefficients,
                        fit_row,
                        blocks,
                        prior_per_bin,
                        penalty,
                        penalties,
                        transition_regularization,
                    )
                    # The search's fits each start from a neighbouring one,
                    # so check the result against the row's start too.
                    if _blocked_loss(
                        fitted, blocks, prior_per_bin, penalty(strength)
                    ) > _blocked_loss(
                        row_coefficients, blocks, prior_per_bin, penalty(strength)
                    ):
                        logger.warning(
                            "Transition optimization did not lower the loss for "
                            "state %d at its REML smoothness. Keeping previous "
                            "coefficients.",
                            from_state,
                        )
                        fitted = row_coefficients
                    elif (
                        _newton_decrement(
                            np.ravel(fitted), blocks, prior_per_bin, penalty(strength)
                        )
                        > 1e-6
                    ):
                        logger.warning(
                            "Transition optimization did not converge for state %d "
                            "at its REML smoothness. Using improved coefficients.",
                            from_state,
                        )
                else:
                    fitted = fit_row(row_coefficients, strength)
                row_coefficients = np.reshape(fitted, row_coefficients.shape)
                row_smoothness[from_state] = strength
            estimated_transition_coefficients[:, from_state, :] = row_coefficients
            device_coefficients = jnp.asarray(row_coefficients, dtype=jnp.float64)
            for start in range(0, n_time, _COVARIATE_BLOCK_ROWS):
                rows = slice(start, start + _COVARIATE_BLOCK_ROWS)
                estimated_transition_matrix[rows, from_state, :] = jnp.exp(
                    jax_centered_log_softmax_forward(
                        jnp.asarray(design_matrix[rows], dtype=jnp.float64)
                        @ device_coefficients
                    )
                )

    return CovariateTransitionFit(
        estimated_transition_coefficients,
        estimated_transition_matrix,
        row_smoothness,
        _covariate_transition_covariance(
            estimated_transition_coefficients,
            design_matrix,
            response,
            concentration,
            stickiness,
            transition_regularization,
            row_smoothness,
            smoothness_matrix,
            frozen_rows,
        )
        if compute_covariance
        else None,
    )


def _covariate_transition_covariance(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float,
    stickiness: float | np.ndarray,
    transition_regularization: float,
    smoothness: np.ndarray,
    smoothness_matrix: np.ndarray | None,
    frozen_rows: np.ndarray | None,
) -> np.ndarray:
    """Laplace covariance of each row's coefficients at a fit: the inverse
    Hessian of the row's penalized loss.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        The fitted coefficients.
    design_matrix, response, concentration, stickiness,
    transition_regularization, smoothness_matrix, frozen_rows
        As given to :func:`_fit_covariate_transition`.
    smoothness : np.ndarray, shape (n_states, n_terms)
        The fit's strengths, :attr:`CovariateTransitionFit.smoothness`.

    Returns
    -------
    covariance : np.ndarray, shape (n_states, n_parameters, n_parameters)
        As :attr:`CovariateTransitionFit.covariance`.
    """
    n_coefficients, n_states, n_free = transition_coefficients.shape
    n_parameters = n_coefficients * n_free
    covariance = np.zeros((n_states, n_parameters, n_parameters))
    alpha = get_transition_prior(concentration, stickiness, n_states)
    penalties = _term_penalties(smoothness_matrix, n_coefficients)
    with _float64_jax():
        design_blocks = _transition_blocks(
            np.asarray(design_matrix, dtype=np.float64)[:-1]
        )
        for from_state, row_alpha in enumerate(alpha):
            if frozen_rows is not None and frozen_rows[from_state]:
                continue
            blocks, prior_per_bin = _row_problem(
                design_blocks, response[:, from_state, :], row_alpha
            )
            # Without transitions (a single time bin) the row is not
            # identified.
            covariance[from_state] = (
                _inverse_symmetric(
                    _blocked_hessian(
                        transition_coefficients[:, from_state, :],
                        blocks,
                        prior_per_bin,
                        _penalty_matrix(
                            n_coefficients,
                            transition_regularization,
                            smoothness[from_state],
                            penalties,
                        ),
                    )
                )
                if blocks
                else np.nan
            )
    return covariance


def _max_transition_change(
    design_matrix: np.ndarray,
    old_coefficients: np.ndarray,
    new_coefficients: np.ndarray,
) -> float:
    """Largest change of any covariate-dependent transition probability
    between two sets of coefficients.

    Evaluated a block of time bins at a time, so no copy of the
    (n_time, n_states, n_states) transitions is held.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    old_coefficients, new_coefficients : np.ndarray
        Shape (n_coefficients, n_states, n_states - 1).

    Returns
    -------
    change : float
    """
    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    change = 0.0
    for start in range(0, design_matrix.shape[0], _COVARIATE_BLOCK_ROWS):
        design = design_matrix[start : start + _COVARIATE_BLOCK_ROWS]
        moved = _transitions_on_design(design, new_coefficients) - (
            _transitions_on_design(design, old_coefficients)
        )
        change = max(change, float(np.max(np.abs(moved))))
    return change


def _inverse_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive definite matrix; NaN if it is
    singular.

    A singular penalized Hessian leaves some direction of the coefficients
    unidentified (e.g. a row without expected transitions and with a flat
    prior), so its covariance is unknown; a pseudo-inverse would give that
    direction zero variance and the intervals false certainty.
    """
    if matrix.shape[0] == 0:
        # A single state has no free coefficients (scipy < 1.13's LAPACK
        # wrappers reject empty matrices).
        return np.zeros_like(matrix)
    try:
        factor = cho_factor(matrix)
    except np.linalg.LinAlgError:
        return np.full_like(matrix, np.nan)
    return cast(np.ndarray, cho_solve(factor, np.eye(matrix.shape[0])))


def _precision_log_pdet(
    penalties: np.ndarray, strengths: np.ndarray, l2_penalty: float
) -> float:
    """Log pseudo-determinant of one logit's prior precision.

    The precision is twice :func:`_penalty_matrix`; the pseudo-determinant
    is over its penalized directions, whose number the strengths do not
    change: every coefficient but the intercept with an L2 penalty, else the
    curvature penalties' ranges. The spline terms sit on disjoint columns, so
    the precision is block diagonal and its determinant factors by term.
    Taking the largest eigenvalues of the summed precision instead would let
    one strongly penalized term's round-off null directions outrank a weakly
    penalized term's genuine ones.

    Parameters
    ----------
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
        Each spline term's curvature penalty.
    strengths : np.ndarray, shape (n_terms,)
        Positive strength of each term.
    l2_penalty : float
        The L2 penalty on every coefficient but the intercept.

    Returns
    -------
    log_pdet : float
        Without an L2 penalty, up to a constant that does not depend on the
        strengths.
    """
    n_coefficients = penalties.shape[-1]
    log_pdet = 0.0
    n_term_columns = 0
    for strength, term in zip(strengths, penalties, strict=True):
        columns = np.flatnonzero(np.any(term != 0.0, axis=0))
        n_term_columns += columns.size
        if l2_penalty > 0.0:
            eigenvalues = np.linalg.eigvalsh(term[np.ix_(columns, columns)])
            log_pdet += np.sum(
                np.log(2.0 * (l2_penalty + strength * np.maximum(eigenvalues, 0.0)))
            )
        else:
            # A range stretched by outliers puts genuine eigenvalues of the
            # penalty at 1e-16 of its largest, among its null directions'
            # round-off. Scaled by its diagonal (a congruence, which keeps the
            # rank) they sit near 1e-2, far above round-off. The log
            # pseudo-determinant of the scaled penalty plus that of the
            # scaling is exact for a full-rank penalty and otherwise off by a
            # constant in the strength, which the LAML's comparisons cancel.
            block = term[np.ix_(columns, columns)]
            diagonal = np.diag(block)
            scaled = block / np.sqrt(np.outer(diagonal, diagonal))
            rank = np.linalg.matrix_rank(scaled, hermitian=True)
            scaled_eigenvalues = np.linalg.eigvalsh(scaled)
            log_pdet += (
                rank * np.log(2.0 * strength)
                + np.sum(np.log(scaled_eigenvalues[scaled_eigenvalues.size - rank :]))
                + np.sum(np.log(diagonal))
            )
    if l2_penalty > 0.0:
        # Penalized columns outside every spline term carry the L2 penalty.
        log_pdet += (n_coefficients - 1 - n_term_columns) * np.log(2.0 * l2_penalty)
    return float(log_pdet)


def _reml_row(
    coefficients: np.ndarray,
    fit_row: Callable[..., np.ndarray],
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: Callable[[np.ndarray], np.ndarray],
    penalties: np.ndarray,
    l2_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One row's curvature-penalty strengths maximizing its LAML.

    The strengths of all spline terms are first searched together, as one
    shared strength; with several terms each is then searched in turn,
    around its current value with the others fixed, until none moves by more
    than the search tolerance. A search keeps its starting value unless it
    finds a higher LAML, so the LAML never decreases. Call with JAX in
    float64. See :func:`_fit_covariate_transition`.

    Parameters
    ----------
    coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        Starting coefficients.
    fit_row : callable
        ``fit_row(start, strengths, warn=False)`` fits the row's coefficients
        without logging.
    blocks, prior_per_bin
        From :func:`_row_problem`.
    penalty : callable
        ``penalty(strengths)``, the row's :func:`_penalty_matrix`.
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
        Each spline term's curvature penalty.
    l2_penalty : float
        The L2 penalty on every coefficient but the intercept.

    Returns
    -------
    coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        The fit at the returned strengths.
    strengths : np.ndarray, shape (n_terms,)
    """
    n_terms, n_coefficients = penalties.shape[:2]
    n_free = np.size(coefficients) // n_coefficients
    fits: dict[tuple[float, ...], tuple[np.ndarray, float]] = {}

    def laml(log10_strengths: np.ndarray, start: np.ndarray) -> float:
        key = tuple(np.asarray(log10_strengths, dtype=np.float64))
        if key in fits:
            return fits[key][1]
        strengths = 10.0 ** np.asarray(log10_strengths, dtype=np.float64)
        beta = np.reshape(
            fit_row(start, strengths, warn=False), (n_coefficients, n_free)
        )
        row_penalty = penalty(strengths)
        log_pdet = n_free * _precision_log_pdet(penalties, strengths, l2_penalty)
        # A singular Hessian is handled below; slogdet need not warn about it.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_det = np.linalg.slogdet(
                _blocked_hessian(beta, blocks, prior_per_bin, row_penalty)
            )
        value: float = (
            -_blocked_loss(beta, blocks, prior_per_bin, row_penalty)
            + 0.5 * log_pdet
            - 0.5 * log_det
        )
        # A singular or indefinite Hessian (e.g. a row without expected
        # transitions) has no Laplace approximation.
        if sign <= 0 or not np.isfinite(value):
            value = -np.inf
        fits[key] = (beta, value)
        return value

    def search(
        point: Callable[[float], np.ndarray], grid: np.ndarray, start: np.ndarray
    ) -> float:
        """The log10 strength ``x`` maximizing ``laml(point(x))``: a grid,
        strongest first with each fit starting from the next-stronger one,
        then a bounded refinement within a decade of the best."""
        scores: dict[float, tuple[float, ...]] = {}

        def score(x: float, start: np.ndarray) -> float:
            value = laml(point(x), start)
            scores[float(x)] = tuple(point(x))
            return value

        for x in np.sort(grid)[::-1]:
            score(x, start)
            start = fits[scores[float(x)]][0]

        def best() -> float:
            return max(scores, key=lambda x: fits[scores[x]][1])

        if not np.isfinite(fits[scores[best()]][1]):
            # No strength has a Laplace approximation: keep the strongest
            # (searched first), whose curve is flattest.
            return best()

        # When the weakest strength searched scores best, scan every decade
        # to the floor: the LAML can dip and rise again (a 0.2-nat dip hid a
        # maximum 13 nats higher three decades down).
        lowest = float(np.min(grid))
        if best() == lowest:
            while lowest > _REML_LOG10_FLOOR:
                above, lowest = lowest, lowest - 1.0
                score(lowest, fits[scores[above]][0])
        best_start = fits[scores[best()]][0]
        minimize_scalar(
            lambda x: -score(x, best_start),
            bounds=(max(best() - 1.0, lowest), min(best() + 1.0, float(np.max(grid)))),
            method="bounded",
            options={"xatol": _REML_LOG10_TOLERANCE},
        )
        return best()

    start = np.reshape(coefficients, (n_coefficients, n_free))
    shared = search(lambda x: np.full(n_terms, x), _REML_LOG10_GRID, start)
    log10_strengths = np.full(n_terms, shared)
    if n_terms > 1:
        for _ in range(_REML_MAX_SWEEPS):
            previous = log10_strengths.copy()
            for term in range(n_terms):

                def point(
                    x: float,
                    term: int = term,
                    base: np.ndarray = log10_strengths.copy(),
                ) -> np.ndarray:
                    moved = base.copy()
                    moved[term] = x
                    return moved

                current = log10_strengths[term]
                # Within two decades of the current value, which is on the
                # grid so the search never returns a lower LAML.
                grid = np.unique(
                    np.clip(
                        current + np.arange(-2.0, 3.0),
                        _REML_LOG10_FLOOR,
                        _REML_LOG10_GRID[-1],
                    )
                )
                log10_strengths = point(
                    search(point, grid, fits[tuple(log10_strengths)][0])
                )
            if np.max(np.abs(log10_strengths - previous)) < _REML_LOG10_TOLERANCE:
                break
    return fits[tuple(log10_strengths)][0], 10.0**log10_strengths


def _fit_covariate_row(
    initial_coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
    optimization_method: str,
    options: dict[str, Any],
    from_state: int,
    warn: bool = True,
) -> np.ndarray:
    """Minimize one row's :func:`dirichlet_neg_log_likelihood` over its
    blocks of transitions. Call with JAX in float64.

    Parameters
    ----------
    initial_coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
    blocks, prior_per_bin
        From :func:`_row_problem`.
    penalty : np.ndarray, shape (n_coefficients, n_coefficients)
        From :func:`_penalty_matrix`.
    optimization_method : str
    options : dict
        Options for `scipy.optimize.minimize`.
    from_state : int
        The row, for log messages.
    warn : bool, optional
        Log when the optimizer does not lower the loss or stops short, by
        default True. REML's candidate fits are not logged; the chosen fit is
        checked once.

    Returns
    -------
    coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        The initial coefficients when there are no transitions or no
        coefficients to fit (one state, whose only transition is certain), or
        when the optimizer does not lower the loss.
    """
    x0 = np.asarray(initial_coefficients, dtype=np.float64).ravel()
    if not blocks or x0.size == 0:
        return initial_coefficients
    args = (blocks, prior_per_bin, penalty)
    minimize_kwargs = {
        "fun": _blocked_loss_and_gradient,
        "x0": x0,
        "method": optimization_method,
        "jac": True,
        "args": args,
        "options": options,
    }
    if optimization_method in {"Newton-CG", "trust-ncg", "dogleg", "trust-exact"}:
        minimize_kwargs["hess"] = _blocked_hessian
    try:
        # scipy 1.15's trust-exact takes the square root of a negative bound
        # product on some Hessians and warns; the loss check below guards the
        # result whatever the optimizer did.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "invalid value encountered in sqrt",
                RuntimeWarning,
                r"scipy\.optimize\._trustregion_exact",
            )
            result = minimize(**minimize_kwargs)
    except (ValueError, np.linalg.LinAlgError) as error:
        # trust-exact's subproblem can fail on a badly conditioned Hessian
        # (seen with scipy 1.12 at strong curvature penalties): fall back to
        # gradient steps, which the loss check below still guards.
        if warn:
            logger.warning(
                "%s failed for state %d (%s); retrying with L-BFGS-B.",
                optimization_method,
                from_state,
                error,
            )
        minimize_kwargs.pop("hess", None)
        result = minimize(
            **{
                **minimize_kwargs,
                "method": "L-BFGS-B",
                "options": {
                    "maxiter": options.get("maxiter", 100),
                    "ftol": 0.0,
                    "gtol": 1e-7,
                },
            }
        )

    # EM never lowers its penalized log-likelihood only if no M-step raises
    # its own loss, so the optimizer's coefficients replace the previous ones
    # only when they do not, whatever the optimizer reports.
    result_loss = _blocked_loss(result.x, *args)
    if not (np.isfinite(result_loss) and result_loss <= _blocked_loss(x0, *args)):
        if not warn:
            return initial_coefficients
        logger.warning(
            "Transition optimization did not lower the loss for state %d: %s. "
            "Keeping previous coefficients.",
            from_state,
            result.message,
        )
        return initial_coefficients
    if warn and not result.success and _newton_decrement(result.x, *args) > 1e-6:
        logger.warning(
            "Transition optimization did not report convergence for state %d: "
            "%s. Using improved coefficients.",
            from_state,
            result.message,
        )
    return cast(np.ndarray, result.x.reshape(np.shape(initial_coefficients)))


def _newton_decrement(coefficients: np.ndarray, *args: Any) -> float:
    """Loss decrease a Newton step predicts, ``g @ inv(H) @ g / 2``.

    Near the optimum of a badly conditioned row (strong penalties give
    condition numbers of 1e10) the gradient stalls well above scipy's
    tolerance while the remaining decrease is negligible; this measures the
    latter. ``inf`` if the Hessian is singular. Call with JAX in float64.
    """
    _, gradient = _blocked_loss_and_gradient(coefficients, *args)
    try:
        step = np.linalg.solve(_blocked_hessian(coefficients, *args), gradient)
    except np.linalg.LinAlgError:
        return np.inf
    return float(gradient @ step) / 2.0


def _row_prior_weights(prior_weight: float | np.ndarray, n_states: int) -> np.ndarray:
    """Validate ``prior_weight`` and broadcast it to one weight per row.

    Parameters
    ----------
    prior_weight : float or np.ndarray, shape (n_states,)
    n_states : int

    Returns
    -------
    prior_weight : np.ndarray, shape (n_states,)
    """
    prior_weight_arr = np.atleast_1d(np.asarray(prior_weight, dtype=float))
    if prior_weight_arr.size == 1:
        prior_weight_arr = np.full(n_states, prior_weight_arr.item())
    if prior_weight_arr.shape != (n_states,):
        msg = (
            f"prior_weight must be a scalar or array of shape ({n_states},), "
            f"got shape {prior_weight_arr.shape}"
        )
        raise ValueError(msg)
    if np.any(prior_weight_arr < 0):
        msg = f"prior_weight must be non-negative, got {prior_weight_arr}"
        raise ValueError(msg)
    return prior_weight_arr


def _stationary_pseudo_counts(
    joint_sum: np.ndarray, alpha: np.ndarray, prior_weight: np.ndarray
) -> np.ndarray:
    """Dirichlet pseudo-counts the stationary M-step adds to the expected counts.

    Rows with ``prior_weight[i] == 0`` use the fixed ``alpha - 1``. Rows with
    ``prior_weight[i] > 0`` keep the direction of ``alpha - 1`` but scale its
    total to ``prior_weight[i]`` times the expected transitions out of state
    ``i``, so these pseudo-counts change with the E-step.

    Parameters
    ----------
    joint_sum : np.ndarray, shape (n_states, n_states)
        Expected transition counts.
    alpha : np.ndarray, shape (n_states, n_states)
        Dirichlet parameters from :func:`get_transition_prior`.
    prior_weight : np.ndarray, shape (n_states,)
        Validated per-row weights from :func:`_row_prior_weights`.

    Returns
    -------
    pseudo_counts : np.ndarray, shape (n_states, n_states)
    """
    legacy_prior = alpha - 1.0
    if not np.any(prior_weight > 0):
        return legacy_prior
    # Normalize to get prior direction per row
    row_sums = legacy_prior.sum(axis=-1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
    tilde_alpha = np.where(row_sums == 0, 0.0, legacy_prior / safe_row_sums)
    # N_i = expected transitions from state i
    n_i = joint_sum.sum(axis=-1, keepdims=True)
    adaptive_prior = prior_weight[:, np.newaxis] * n_i * tilde_alpha
    return np.where((prior_weight > 0)[:, np.newaxis], adaptive_prior, legacy_prior)


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
        Row-stochastic transition matrix estimated from the counts.
    """
    n_states = joint_sum.shape[0]
    alpha = get_transition_prior(concentration, stickiness, n_states)
    _assert_map_compatible_alpha(alpha)
    prior_weight_arr = _row_prior_weights(prior_weight, n_states)

    # Warn when an adaptive row has a uniform prior direction (alpha - 1 == 0,
    # e.g. concentration=1.0 and stickiness=0.0): prior_weight then has no
    # effect because there is nothing to scale.
    no_op_rows = (prior_weight_arr > 0) & ((alpha - 1.0).sum(axis=-1) == 0)
    if np.any(no_op_rows):
        logger.warning(
            "prior_weight > 0 has no effect for states %s because the prior "
            "direction is uniform (alpha - 1 == 0). Increase concentration or "
            "stickiness to regularize these rows.",
            np.flatnonzero(no_op_rows).tolist(),
        )

    effective_prior = _stationary_pseudo_counts(joint_sum, alpha, prior_weight_arr)
    new_transition_matrix = joint_sum + effective_prior

    # Normalize rows to get transition probabilities.
    # Guard against all-zero rows (e.g., unvisited states) to avoid NaN.
    # Fall back to normalized Dirichlet prior direction for unvisited states,
    # preserving stickiness semantics. If prior is also uniform, this gives 1/n.
    prior_fallback = alpha / alpha.sum(axis=-1, keepdims=True)
    row_totals = new_transition_matrix.sum(axis=-1, keepdims=True)
    safe_row_totals = np.where(row_totals == 0, 1.0, row_totals)
    return cast(
        np.ndarray,
        np.where(
            row_totals == 0,
            prior_fallback,
            new_transition_matrix / safe_row_totals,
        ),
    )


def _block_loss(
    coefficients: jax.Array,
    design_matrix: jax.Array,
    response: jax.Array,
    prior_per_bin: jax.Array,
) -> jax.Array:
    """``-sum_t (r_t + prior_per_bin) . log p_t`` over one block of time bins.

    Parameters
    ----------
    coefficients : jax.Array, shape (n_coefficients * (n_states - 1),)
    design_matrix : jax.Array, shape (n_block, n_coefficients)
    response : jax.Array, shape (n_block, n_states)
    prior_per_bin : jax.Array, shape (n_states,)
        The row's Dirichlet pseudo-counts ``alpha - 1`` divided by the number
        of time bins of the whole row, so that the prior counts once.

    Returns
    -------
    loss : jax.Array, shape ()
    """
    log_probabilities = jax_centered_log_softmax_forward(
        design_matrix @ coefficients.reshape((design_matrix.shape[1], -1))
    )
    return -jnp.sum((response + prior_per_bin) * log_probabilities)


@jax.jit
def _block_hessian(
    coefficients: jax.Array,
    design_matrix: jax.Array,
    response: jax.Array,
    prior_per_bin: jax.Array,
) -> jax.Array:
    """Hessian of :func:`_block_loss` in closed form.

    With weights ``w_t = r_t + prior_per_bin`` summing to ``W_t`` and free
    probabilities ``p_t`` (all but the last state), the loss's Hessian in
    the logits ``z_t`` is ``W_t (diag(p_t) - p_t p_t')``, so the Hessian in
    the coefficients is ``sum_t x_t x_t' (x) W_t (diag(p_t) - p_t p_t')``.
    About 4x faster than ``jax.hessian``, which differentiates the gradient
    once per coefficient.

    Parameters
    ----------
    coefficients : jax.Array, shape (n_coefficients * (n_states - 1),)
        Ordered (coefficient, logit).
    design_matrix : jax.Array, shape (n_block, n_coefficients)
    response : jax.Array, shape (n_block, n_states)
    prior_per_bin : jax.Array, shape (n_states,)

    Returns
    -------
    hessian : jax.Array, shape (n_coefficients * (n_states - 1),) * 2
    """
    n_coefficients = design_matrix.shape[1]
    free = jnp.exp(
        jax_centered_log_softmax_forward(
            design_matrix @ coefficients.reshape((n_coefficients, -1))
        )
    )[:, :-1]
    n_free = free.shape[1]
    weight = jnp.sum(response + prior_per_bin, axis=1)
    curvature = weight[:, None, None] * (
        free[:, :, None] * jnp.eye(n_free) - free[:, :, None] * free[:, None, :]
    )
    hessian = jnp.einsum("tc,tjk,td->cjdk", design_matrix, curvature, design_matrix)
    return hessian.reshape((n_coefficients * n_free, n_coefficients * n_free))


_jit_block_loss = jax.jit(_block_loss)
_block_loss_and_gradient = jax.jit(jax.value_and_grad(_block_loss))


def _transition_blocks(
    values: np.ndarray, dtype: type | None = np.float64
) -> list[jax.Array]:
    """Device blocks of ``_COVARIATE_BLOCK_ROWS`` rows of ``values``.

    Call with JAX in float64; ``dtype=None`` keeps the values' own type.
    """
    return [
        jnp.asarray(values[start : start + _COVARIATE_BLOCK_ROWS], dtype=dtype)
        for start in range(0, len(values), _COVARIATE_BLOCK_ROWS)
    ]


def _row_problem(
    design_blocks: list[jax.Array],
    response: np.ndarray,
    alpha: float | np.ndarray,
) -> tuple[_Blocks, jax.Array]:
    """One row's blocks of transitions and its prior pseudo-counts per bin.

    Call with JAX in float64.

    Parameters
    ----------
    design_blocks : list of jax.Array
        :func:`_transition_blocks` of the design matrix's transition rows.
    response : np.ndarray, shape (n_samples, n_states)
    alpha : float or np.ndarray, shape (n_states,)

    Returns
    -------
    blocks : list of (jax.Array, jax.Array)
        ``(design, response)`` blocks; empty without transitions.
    prior_per_bin : jax.Array, shape (n_states,)
        ``(alpha - 1) / n_samples``, zero without transitions.
    """
    n_samples, n_states = np.shape(response)
    blocks = list(
        zip(design_blocks, _transition_blocks(response, dtype=None), strict=True)
    )
    pseudo_counts = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n_states,))
    prior_per_bin = (pseudo_counts - 1.0) / max(n_samples, 1)
    return blocks, jnp.asarray(prior_per_bin, dtype=jnp.float64)


def _penalty_matrix(
    n_coefficients: int,
    l2_penalty: float,
    smoothness: float | np.ndarray = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Quadratic penalty on one row's coefficients, per free logit.

    ``l2_penalty`` on every coefficient but the intercept (the first) plus
    each spline term's curvature penalty times its strength.

    Parameters
    ----------
    n_coefficients : int
    l2_penalty : float
    smoothness : float or np.ndarray, shape (n_terms,), optional
        One strength for every term, or one per term.
    smoothness_matrix : np.ndarray, optional
        From :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients), or one term's
        (n_coefficients, n_coefficients); None adds no curvature penalty.

    Returns
    -------
    penalty : np.ndarray, shape (n_coefficients, n_coefficients)
        The row's penalty is ``sum_k beta[:, k] @ penalty @ beta[:, k]`` over
        its ``n_states - 1`` logits.
    """
    penalty = np.diag(np.r_[0.0, np.full(n_coefficients - 1, float(l2_penalty))])
    penalties = _term_penalties(smoothness_matrix, n_coefficients)
    strengths = np.broadcast_to(
        np.asarray(smoothness, dtype=np.float64), (penalties.shape[0],)
    )
    if np.any(strengths != 0.0):
        penalty = penalty + np.tensordot(strengths, penalties, axes=1)
    return penalty


def _penalty_value(coefficients: np.ndarray, penalty: np.ndarray) -> float:
    """``sum_k beta[:, k] @ penalty @ beta[:, k]`` of flattened coefficients."""
    beta = np.reshape(coefficients, (penalty.shape[0], -1))
    return float(np.sum(beta * (penalty @ beta)))


def _blocked_loss(
    coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
) -> float:
    """:func:`dirichlet_neg_log_likelihood` over blocks of transitions.

    Call with JAX in float64. ``penalty`` is from :func:`_penalty_matrix`.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    # Summed on the device, so the blocks are not each copied to the host.
    device_coefficients = jnp.asarray(coefficients)
    loss = sum(
        _jit_block_loss(device_coefficients, design, response, prior_per_bin)
        for design, response in blocks
    )
    return float(loss) + _penalty_value(coefficients, penalty)


def _blocked_loss_and_gradient(
    coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
) -> tuple[float, np.ndarray]:
    """:func:`dirichlet_neg_log_likelihood` and its gradient over blocks of
    transitions, for scipy. Call with JAX in float64."""
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    # Summed on the device, so the blocks are not each copied to the host.
    device_coefficients = jnp.asarray(coefficients)
    device_loss, device_gradient = jnp.zeros(()), jnp.zeros(coefficients.shape)
    for design, response in blocks:
        block_loss, block_gradient = _block_loss_and_gradient(
            device_coefficients, design, response, prior_per_bin
        )
        device_loss = device_loss + block_loss
        device_gradient = device_gradient + block_gradient
    gradient = np.array(device_gradient, dtype=np.float64)
    loss = float(device_loss) + _penalty_value(coefficients, penalty)
    beta = coefficients.reshape((penalty.shape[0], -1))
    gradient += (2.0 * penalty @ beta).ravel()
    return loss, gradient


def _blocked_hessian(
    coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
) -> np.ndarray:
    """:func:`dirichlet_neg_log_likelihood`'s Hessian over blocks of
    transitions, for scipy. Call with JAX in float64."""
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    # Summed on the device, so the blocks are not each copied to the host.
    device_coefficients = jnp.asarray(coefficients)
    device_hessian = jnp.zeros((coefficients.size, coefficients.size))
    for design, response in blocks:
        device_hessian = device_hessian + _block_hessian(
            device_coefficients, design, response, prior_per_bin
        )
    hessian = np.array(device_hessian, dtype=np.float64)
    # Flattened coefficients are ordered (coefficient, logit), so each logit
    # gets its own copy of the penalty.
    n_free = coefficients.size // penalty.shape[0]
    return hessian + np.kron(2.0 * penalty, np.eye(n_free))


def dirichlet_neg_log_likelihood(
    coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    alpha: float | np.ndarray = 1.0,
    l2_penalty: float = 0.1,
    smoothness: float = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> float:
    """Negative log posterior of one covariate-dependent transition row.

    ``-sum_t (r_t + (alpha - 1) / n) . log p_t + l2_penalty * |beta[1:]|^2
    + sum_j smoothness_j * sum_k beta[:, k] @ S_j @ beta[:, k]``, summed over the ``n``
    time bins: the expected complete-data log-likelihood of the row plus a
    Dirichlet prior counted once, as for a stationary row, an L2 penalty and
    a curvature penalty ``S_j`` on each spline term. Evaluated with
    JAX in float64 over blocks of ``_COVARIATE_BLOCK_ROWS`` time bins, as the
    M-step evaluates it.

    Parameters
    ----------
    coefficients : np.ndarray, shape (n_coefficients * (n_states - 1),)
        Flattened regression coefficients.
    design_matrix : np.ndarray, shape (n_samples, n_coefficients)
        Covariate design matrix.
    response : np.ndarray, shape (n_samples, n_states)
        Expected transition counts from this row at each time bin.
    alpha : float or np.ndarray, shape (n_states,), optional
        Dirichlet prior parameters for this row. Defaults to 1.0 (no prior).
    l2_penalty : float, optional
        L2 penalty on coefficients (excluding intercept), by default 0.1, as
        for the detectors' ``discrete_transition_regularization``.
        The intercept is assumed to be the first coefficient row
        (``coefficients[0]``), as produced by patsy ``"1 + ..."`` formulas.
        A ``"0 + ..."`` formula or hand-built design matrix without a leading
        intercept column would penalize the wrong rows.
    smoothness : float or np.ndarray, shape (n_terms,), optional
        Strength of each spline term's curvature penalty (one value for
        every term), by default 0.0.
    smoothness_matrix : np.ndarray, optional
        Curvature penalties from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients), or one term's
        (n_coefficients, n_coefficients); None (default) adds none.

    Returns
    -------
    negative_log_posterior : float
        Value of the penalized negative log posterior.
    """
    with _float64_jax():
        return _blocked_loss(
            coefficients,
            *_row_problem(_transition_blocks(design_matrix), response, alpha),
            _penalty_matrix(
                np.shape(design_matrix)[1], l2_penalty, smoothness, smoothness_matrix
            ),
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
    """Gradient of :func:`dirichlet_neg_log_likelihood`.

    Parameters
    ----------
    coefficients, design_matrix, response, alpha, l2_penalty, smoothness, smoothness_matrix
        As in :func:`dirichlet_neg_log_likelihood`.

    Returns
    -------
    gradient : np.ndarray, shape (n_coefficients * (n_states - 1),)
        Gradient of the negative log posterior with respect to the
        flattened coefficients.
    """
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
    """Hessian of :func:`dirichlet_neg_log_likelihood`.

    Parameters
    ----------
    coefficients, design_matrix, response, alpha, l2_penalty, smoothness, smoothness_matrix
        As in :func:`dirichlet_neg_log_likelihood`.

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
    if n_states == 1:
        # A single state has nowhere else to go; the only stochastic matrix is
        # [[1.0]] regardless of the requested diagonal value.
        return np.ones((1, 1))
    transition_matrix = diag * np.eye(n_states)
    off_diag = ((1.0 - diag) / (n_states - 1.0))[:, np.newaxis]
    transition_matrix += np.ones((n_states, n_states)) * off_diag - off_diag * np.eye(
        n_states
    )

    return transition_matrix


def _discrete_transition_statistics(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    continuous_transition: np.ndarray,
    state_ind: np.ndarray,
    discrete_transition: np.ndarray,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
) -> np.ndarray:
    """Expected discrete transition statistics from the expanded posteriors.

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

    Returns
    -------
    statistics : np.ndarray
        Expected transition counts, shape (n_states, n_states), for a
        stationary transition; per-row responses, shape
        (n_time - 1, n_states, n_states), for a covariate transition.
    """
    return (
        estimate_discrete_transition_responses_from_factorized_posteriors
        if _is_nonstationary(
            discrete_transition_coefficients, discrete_transition_design_matrix
        )
        else estimate_discrete_transition_counts_from_factorized_posteriors
    )(
        causal_posterior,
        predictive_posterior,
        acausal_posterior,
        continuous_transition,
        discrete_transition,
        state_ind,
    )


def _transition_log_prior(
    statistics: np.ndarray,
    discrete_transition: np.ndarray | None,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | np.ndarray | None,
    transition_concentration: float,
    transition_stickiness: float | np.ndarray,
    transition_regularization: float,
    transition_prior_weight: float | np.ndarray = 0.0,
    free_rows: np.ndarray | None = None,
    smoothness: float | np.ndarray = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> float:
    """Log prior of discrete transition parameters, as the M-step weighs it.

    The discrete M-step maximizes the expected complete-data log-likelihood
    plus this term, so EM never decreases the marginal log-likelihood plus
    this term. Both are evaluated with the prior of one M-step (the
    data-adaptive ``prior_weight`` pseudo-counts depend on ``statistics``).

    Stationary: ``sum_ij c_ij log A_ij`` with the pseudo-counts ``c`` the
    stationary M-step adds once per row. Covariate-dependent:
    ``sum_ij (alpha_ij - 1) mean_t log A_t,ij - lambda * |beta_i[1:]|^2``
    ``- sum_j smoothness_ij * sum_k beta_i[:, k] @ S_j @ beta_i[:, k]`` over the
    ``n_time - 1`` transitions, the prior and penalty terms of
    :func:`dirichlet_neg_log_likelihood`, negated.

    Parameters
    ----------
    statistics : np.ndarray
        Expected transition counts, shape (n_states, n_states), or per-row
        responses, shape (n_time - 1, n_states, n_states), as passed to
        :func:`_discrete_transition_from_statistics`.
    discrete_transition : np.ndarray, shape (n_states, n_states), or None
        Stationary transition matrix to evaluate. Ignored for a
        covariate-dependent transition, which is evaluated from its
        coefficients.
    discrete_transition_coefficients : np.ndarray or None
        Shape (n_coefficients, n_states, n_states - 1) for a
        covariate-dependent transition.
    discrete_transition_design_matrix : DesignMatrix, np.ndarray or None
        Shape (n_time, n_coefficients) for a covariate-dependent transition.
    transition_concentration, transition_stickiness, transition_regularization,
    transition_prior_weight
        As in :func:`_discrete_transition_from_statistics`.
    free_rows : np.ndarray of bool, shape (n_states,), optional
        Rows the M-step estimates. Frozen rows are constant and left out.
        All rows by default.
    smoothness : float or np.ndarray, optional
        Curvature-penalty strength of each covariate-dependent row and spline
        term, shape (n_states, n_terms), or one per row, shape (n_states,);
        by default 0.0.
    smoothness_matrix : np.ndarray, optional
        From :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients); None adds no curvature
        penalty.

    Returns
    -------
    log_prior : float
        ``-inf`` if the prior puts pseudo-counts on a zero-probability entry.
    """
    statistics = np.asarray(statistics)
    n_states = statistics.shape[-1]
    alpha = get_transition_prior(
        transition_concentration, transition_stickiness, n_states
    )
    rows = np.ones(n_states, dtype=bool) if free_rows is None else free_rows
    if _is_nonstationary(
        discrete_transition_coefficients, discrete_transition_design_matrix
    ):
        n_samples = statistics.shape[0]
        if n_samples == 0:
            # A recording without transitions has no covariate prior, and the
            # M-step keeps its coefficients.
            return 0.0
        coefficients = np.asarray(discrete_transition_coefficients, dtype=np.float64)
        design_matrix = np.asarray(discrete_transition_design_matrix)
        free_states = np.flatnonzero(rows)
        prior_per_bin = (alpha - 1.0) / n_samples
        # Minus the M-step's loss with no expected transitions: its prior and
        # penalties alone, with one block of the design matrix on the device
        # at a time.
        penalties = _term_penalties(smoothness_matrix, coefficients.shape[0])
        row_smoothness = _row_strengths(smoothness, n_states, penalties.shape[0])
        log_prior = -sum(
            _penalty_value(
                coefficients[:, row],
                _penalty_matrix(
                    coefficients.shape[0],
                    transition_regularization,
                    row_smoothness[row],
                    penalties,
                ),
            )
            for row in free_states
        )
        with _float64_jax():
            for start in range(0, n_samples, _COVARIATE_BLOCK_ROWS):
                stop = min(start + _COVARIATE_BLOCK_ROWS, n_samples)
                design = jnp.asarray(design_matrix[start:stop], dtype=jnp.float64)
                no_counts = jnp.zeros((stop - start, n_states))
                for row in free_states:
                    log_prior -= float(
                        _jit_block_loss(
                            coefficients[:, row].ravel(),
                            design,
                            no_counts,
                            jnp.asarray(prior_per_bin[row]),
                        )
                    )
        return log_prior

    pseudo_counts = _stationary_pseudo_counts(
        statistics, alpha, _row_prior_weights(transition_prior_weight, n_states)
    )
    # xlogy gives 0 * log(0) = 0: an impossible transition costs nothing
    # without pseudo-counts, and -inf with them.
    terms = xlogy(pseudo_counts, np.asarray(discrete_transition, dtype=np.float64))
    return float(np.sum(terms[rows]))


def _is_nonstationary(
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
) -> bool:
    """Whether the discrete transition depends on covariates."""
    return (
        discrete_transition_coefficients is not None
        and discrete_transition_design_matrix is not None
    )


def _discrete_transition_from_statistics(
    statistics: np.ndarray,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
    transition_concentration: float,
    transition_stickiness: float | np.ndarray,
    transition_regularization: float,
    transition_prior_weight: float | np.ndarray = 0.0,
    frozen_rows: np.ndarray | None = None,
    smoothness: float | np.ndarray | str = 0.0,
    smoothness_matrix: np.ndarray | None = None,
    initial_smoothness: np.ndarray | None = None,
) -> "_DiscreteTransitionMStep":
    """Discrete transition M-step from expected transition statistics.

    Parameters
    ----------
    statistics : np.ndarray
        Expected transition counts, shape (n_states, n_states), for a
        stationary transition; per-row responses, shape
        (n_time - 1, n_states, n_states), for a covariate transition.
    discrete_transition_coefficients, discrete_transition_design_matrix
        As in :func:`_discrete_transition_statistics`.
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
    frozen_rows : np.ndarray of bool, shape (n_states,), optional
        Covariate-dependent rows that keep their coefficients. Stationary
        frozen rows are restored by the caller.
    smoothness, smoothness_matrix, initial_smoothness
        Curvature penalty of a covariate transition, as in
        :func:`_fit_covariate_transition`. Ignored for a stationary one. The
        covariance is not computed: EM computes it once, after its last
        M-step, with :func:`_covariate_transition_covariance`.

    Returns
    -------
    m_step : _DiscreteTransitionMStep

    Raises
    ------
    ValueError
        If any statistic is NaN or infinite, which would otherwise give NaN
        transitions without an error.
    """
    statistics = np.asarray(statistics)
    bad = ~np.isfinite(statistics)
    if bad.any():
        pairs = sorted(
            {tuple(int(i) for i in index[-2:]) for index in np.argwhere(bad)}
        )
        msg = (
            "Expected discrete transition statistics are not finite for state "
            f"pairs {pairs}. This usually follows NaN or impossible likelihood "
            "rows; see the degenerate/NaN timestep warnings."
        )
        raise ValueError(msg)
    if _is_nonstationary(
        discrete_transition_coefficients, discrete_transition_design_matrix
    ):
        fit = _fit_covariate_transition(
            cast(np.ndarray, discrete_transition_coefficients),
            cast(np.ndarray, discrete_transition_design_matrix),
            statistics,
            concentration=transition_concentration,
            stickiness=transition_stickiness,
            transition_regularization=transition_regularization,
            smoothness=smoothness,
            smoothness_matrix=smoothness_matrix,
            initial_smoothness=initial_smoothness,
            frozen_rows=frozen_rows,
            compute_covariance=False,
        )
        return _DiscreteTransitionMStep(
            fit.transition, fit.coefficients, fit.smoothness
        )
    stickiness_value: float | np.ndarray
    if (
        isinstance(transition_stickiness, np.ndarray)
        and transition_stickiness.size == 1
    ):
        stickiness_value = float(transition_stickiness.item())
    else:
        stickiness_value = transition_stickiness
    discrete_transition = estimate_stationary_state_transition_from_counts(
        statistics,
        concentration=transition_concentration,
        stickiness=stickiness_value,
        prior_weight=transition_prior_weight,
    )

    return _DiscreteTransitionMStep(
        discrete_transition, discrete_transition_coefficients
    )


class _DiscreteTransitionMStep(NamedTuple):
    """Output of :func:`_discrete_transition_from_statistics`.

    ``smoothness`` is that of :class:`CovariateTransitionFit`, and None for
    a stationary transition.
    """

    transition: np.ndarray
    coefficients: np.ndarray | None
    smoothness: np.ndarray | None = None


class _QuantileBSpline:
    """Patsy stateful transform: a B-spline with knots at data quantiles.

    The boundaries sit at the ``lower_quantile`` and ``upper_quantile`` of
    the fitted values (by default their range), and interior knots at
    quantiles of the distinct fitted values between them, so ties (e.g. many
    zero speeds at rest) do not stack knots and every basis column has data.
    Values beyond the boundaries, fitted or new, are clipped to them: they
    take the boundary's value instead of raising. An upper boundary below the
    maximum keeps a sparse tail (e.g. the fastest 1% of speeds) from bending
    the last spline segment.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def memorize_chunk(
        self,
        x: ArrayLike,
        n_knots: int = 5,
        degree: int = 3,
        lower_quantile: float = 0.0,
        upper_quantile: float = 1.0,
    ) -> None:
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            msg = (
                "quantile_bs needs 0 <= lower_quantile < upper_quantile <= 1, got "
                f"lower_quantile={lower_quantile}, upper_quantile={upper_quantile}."
            )
            raise ValueError(msg)
        self._n_knots, self._degree = n_knots, degree
        self._quantiles = (lower_quantile, upper_quantile)
        self._chunks.append(np.ravel(np.asarray(x, dtype=float)))

    def memorize_finish(self) -> None:
        values = np.concatenate(self._chunks)
        del self._chunks
        # Missing values are left to the formula's NA_action.
        values = values[np.isfinite(values)]
        self._lower, self._upper = np.quantile(values, self._quantiles)
        distinct = np.unique(values[(values >= self._lower) & (values <= self._upper)])
        if distinct.size < 2:
            msg = "quantile_bs needs at least two distinct values."
            raise ValueError(msg)
        knots = np.unique(
            np.quantile(distinct, np.linspace(0, 1, self._n_knots + 2)[1:-1])
        )
        self._interior_knots = knots[(knots > self._lower) & (knots < self._upper)]
        self._spline = BS()
        self._spline.memorize_chunk(
            distinct,
            knots=self._interior_knots,
            degree=self._degree,
            lower_bound=self._lower,
            upper_bound=self._upper,
        )
        self._spline.memorize_finish()

    def transform(
        self,
        x: ArrayLike,
        n_knots: int = 5,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
        degree: int = 3,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
        lower_quantile: float = 0.0,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
        upper_quantile: float = 1.0,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
    ) -> np.ndarray:
        return cast(
            np.ndarray,
            self._spline.transform(
                np.clip(np.asarray(x, dtype=float), self._lower, self._upper)
            ),
        )

    def curvature_penalty(self) -> np.ndarray:
        """Integrated squared second derivative of the basis.

        ``beta @ penalty @ beta`` is ``integral_0^1 f''(u)^2 du`` for the
        spline ``f`` with coefficients ``beta`` on this transform's columns,
        with the covariate rescaled to ``u`` in [0, 1] between the
        boundaries, so the penalty does not depend on the covariate's units.

        Returns
        -------
        penalty : np.ndarray, shape (n_columns, n_columns)
        """
        knots = np.asarray(self._spline._all_knots, dtype=np.float64)
        degree = self._degree
        n_basis = knots.size - degree - 1
        width = self._upper - self._lower
        if degree < 2:
            return np.zeros((n_basis - 1, n_basis - 1))
        second = BSpline(knots, np.eye(n_basis), degree).derivative(2)
        penalty = _curvature_gram(second, np.unique(knots), degree)
        # d2f/du2 = width**2 d2f/dx2 and du = dx / width. patsy's bs drops the
        # first basis column, whose effect the intercept absorbs.
        return cast(np.ndarray, width**3 * penalty[1:, 1:])


def _curvature_gram(
    second: Callable[[np.ndarray], np.ndarray], breaks: np.ndarray, degree: int
) -> np.ndarray:
    """``integral second(x).T @ second(x) dx`` between the first and last
    break, for basis second derivatives ``second(x)`` of shape
    (n_points, n_basis) that are polynomials between breaks.

    The second derivatives of a degree-``degree`` spline are polynomials of
    degree ``degree - 2`` between knots, so ``degree`` Gauss-Legendre points
    per interval integrate their products exactly.
    """
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    left, right = breaks[:-1], breaks[1:]
    points = (left[:, None] + right[:, None]) / 2 + (right - left)[:, None] / 2 * nodes
    point_weights = (right - left)[:, None] / 2 * weights
    values = second(points.ravel())
    return cast(np.ndarray, (values * point_weights.ravel()[:, None]).T @ values)


class _CyclicBSpline:
    """Patsy stateful transform: a periodic cubic B-spline with knots at
    data quantiles, for cyclic covariates such as phase or head direction.

    The period is ``[lower, upper)``, given explicitly because the fitted
    values need not span it. Values outside it wrap. ``n_knots`` knots sit at
    quantiles of the distinct wrapped values, and the curve and its first
    ``degree - 1`` derivatives join smoothly across ``lower`` / ``upper``.
    The periodic basis sums to 1, so its first column, whose effect the
    intercept absorbs, is dropped.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def memorize_chunk(
        self,
        x: ArrayLike,
        lower: float,
        upper: float,
        n_knots: int = 8,
        degree: int = 3,
    ) -> None:
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            msg = (
                "cyclic_bs needs finite lower < upper bounds of the period, got "
                f"lower={lower}, upper={upper}."
            )
            raise ValueError(msg)
        if n_knots < degree + 1:
            msg = (
                f"cyclic_bs needs n_knots >= degree + 1 = {degree + 1}, got "
                f"n_knots={n_knots}."
            )
            raise ValueError(msg)
        self._lower, self._upper = float(lower), float(upper)
        self._n_knots, self._degree = n_knots, degree
        self._chunks.append(np.ravel(np.asarray(x, dtype=float)))

    def memorize_finish(self) -> None:
        values = np.concatenate(self._chunks)
        del self._chunks
        # Missing values are left to the formula's NA_action.
        distinct = np.unique(self._wrap(values[np.isfinite(values)]))
        if distinct.size < self._n_knots:
            msg = (
                f"cyclic_bs needs at least n_knots={self._n_knots} distinct "
                f"values, got {distinct.size}."
            )
            raise ValueError(msg)
        self._knots = np.quantile(
            distinct, (np.arange(self._n_knots) + 0.5) / self._n_knots
        )

    def _wrap(self, x: np.ndarray) -> np.ndarray:
        return self._lower + np.mod(x - self._lower, self._upper - self._lower)

    def _extended(self) -> tuple[np.ndarray, float]:
        """Knots extended by ``degree`` periods' worth on each side, so the
        open B-spline basis over one period from the first knot can be
        folded into the periodic one."""
        period, knots, degree = self._upper - self._lower, self._knots, self._degree
        extended = np.concatenate(
            [knots[-degree:] - period, knots, knots[: degree + 1] + period]
        )
        return extended, knots[0]

    def _fold(self, basis: np.ndarray) -> np.ndarray:
        """The periodic basis: open basis function ``j + n_knots`` is
        function ``j`` one period later."""
        periodic = basis[:, : self._n_knots].copy()
        periodic[:, : self._degree] += basis[:, self._n_knots :]
        return periodic

    def transform(
        self,
        x: ArrayLike,
        lower: float,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
        upper: float,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
        n_knots: int = 8,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
        degree: int = 3,  # noqa: ARG002 -- patsy passes memorize_chunk's arguments
    ) -> np.ndarray:
        extended, start = self._extended()
        period = self._upper - self._lower
        values = np.ravel(np.asarray(x, dtype=float))
        # Missing values stay missing, for the formula's NA_action.
        finite = np.isfinite(values)
        columns = np.full((values.size, self._n_knots - 1), np.nan)
        # One period starting at the first knot, where the open basis lives.
        shifted = start + np.mod(values[finite] - start, period)
        basis = BSpline.design_matrix(shifted, extended, self._degree)
        columns[finite] = self._fold(basis.toarray())[:, 1:]
        return columns.reshape((*np.shape(x), -1))

    def curvature_penalty(self) -> np.ndarray:
        """Integrated squared second derivative of the basis over one period.

        ``beta @ penalty @ beta`` is ``integral_0^1 f''(u)^2 du`` with the
        covariate rescaled to ``u`` in [0, 1) over the period, as for
        :meth:`_QuantileBSpline.curvature_penalty`.

        Returns
        -------
        penalty : np.ndarray, shape (n_knots - 1, n_knots - 1)
        """
        if self._degree < 2:
            return np.zeros((self._n_knots - 1, self._n_knots - 1))
        extended, start = self._extended()
        period = self._upper - self._lower
        n_open = extended.size - self._degree - 1
        second = BSpline(extended, np.eye(n_open), self._degree).derivative(2)
        breaks = np.append(self._knots, start + period)
        penalty = _curvature_gram(lambda x: self._fold(second(x)), breaks, self._degree)
        return period**3 * penalty[1:, 1:]


def _smoothness_matrix(design_info: DesignInfo) -> np.ndarray:
    """Curvature penalty of each spline term of a design.

    Parameters
    ----------
    design_info : patsy.DesignInfo
        ``design_info`` of a transition design matrix.

    Returns
    -------
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
        One matrix per ``quantile_bs`` or ``cyclic_bs`` main-effect term: its
        ``curvature_penalty`` on the term's columns, zero elsewhere.
        Interactions are not penalized. ``n_terms`` is 0 without such terms.
    """
    n_columns = len(design_info.column_names)
    penalties = []
    for term, columns in design_info.term_slices.items():
        if len(term.factors) != 1:
            continue
        state = getattr(design_info.factor_infos[term.factors[0]], "state", {})
        splines = [
            transform
            for transform in state.get("transforms", {}).values()
            if isinstance(transform, _QuantileBSpline | _CyclicBSpline)
        ]
        if len(splines) != 1:
            continue
        term_penalty = splines[0].curvature_penalty()
        if term_penalty.shape == (columns.stop - columns.start,) * 2:
            penalty = np.zeros((n_columns, n_columns))
            penalty[columns, columns] = term_penalty
            penalties.append(penalty)
    return np.reshape(np.asarray(penalties), (len(penalties), n_columns, n_columns))


# Usable in transition formulas, e.g. "1 + quantile_bs(speed, n_knots=5)".
quantile_bs = stateful_transform(_QuantileBSpline)
# Usable in transition formulas for cyclic covariates, e.g.
# "1 + cyclic_bs(head_direction, lower=0, upper=360)".
cyclic_bs = stateful_transform(_CyclicBSpline)


class InitialDiscreteTransition(NamedTuple):
    """The discrete transition a ``make_state_transition`` starts EM from.

    Attributes
    ----------
    transition_matrix : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
        Probability of moving from each state to each state; one matrix per
        time step for a covariate-dependent transition.
    coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1), or None
        Regression coefficients of a covariate-dependent transition, with
        only the intercepts set; None for a stationary transition.
    design_matrix : patsy.DesignMatrix or None
        Design matrix of the covariates; None for a stationary transition.
    """

    transition_matrix: np.ndarray
    coefficients: np.ndarray | None
    design_matrix: DesignMatrix | None


@runtime_checkable
class DiscreteTransition(Protocol):
    """A model of the transitions between discrete states.

    The detectors read ``is_stationary``, ``n_states`` and ``smoothness``
    and call ``make_state_transition(covariate_data)`` at fit. A
    user-defined class with these members can be used in place of the
    package's ``DiscreteStationaryDiagonal``, ``DiscreteStationaryCustom``,
    ``DiscreteNonStationaryDiagonal`` and ``DiscreteNonStationaryCustom``.
    Spyglass stores a transition as ``vars(obj)`` and rebuilds it with
    ``type(obj)(**vars(obj))``, so every instance attribute should be an
    ``__init__`` argument.
    """

    @property
    def is_stationary(self) -> bool:
        """Whether the transition is the same at every time step.

        A stationary transition returns a ``(n_states, n_states)`` matrix and
        no coefficients or design matrix; a covariate-dependent one returns
        one matrix per time step, its coefficients and its design matrix.
        """
        ...

    @property
    def n_states(self) -> int:
        """Number of discrete states."""
        ...

    @property
    def smoothness(self) -> float | str:
        """Curvature-penalty strength of covariate-dependent rows (or
        ``"reml"``); 0.0 for a stationary transition."""
        ...

    def make_state_transition(
        self, covariate_data: pd.DataFrame | dict[str, Any] | None
    ) -> InitialDiscreteTransition:
        """The transition EM starts from.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict or None
            Covariates, one row per time step, for a covariate-dependent
            transition; ignored by a stationary one.

        Returns
        -------
        initial : InitialDiscreteTransition
        """
        ...


def _check_initial_discrete_transition(
    initial: InitialDiscreteTransition, transition: DiscreteTransition
) -> None:
    """Check that ``make_state_transition``'s result matches the transition.

    Parameters
    ----------
    initial : InitialDiscreteTransition
        Result of ``transition.make_state_transition``.
    transition : DiscreteTransition

    Raises
    ------
    ValidationError
        If the result's shape does not match ``n_states``, or its
        coefficients, design matrix or number of dimensions do not match
        ``is_stationary``.
    """
    name = type(transition).__name__
    n_states = transition.n_states
    matrix = np.asarray(initial.transition_matrix)
    has_covariates = (
        initial.coefficients is not None or initial.design_matrix is not None
    )
    if transition.is_stationary:
        consistent = matrix.ndim == 2 and not has_covariates
        expected = (
            "a (n_states, n_states) transition_matrix and None coefficients "
            "and design_matrix"
        )
    else:
        consistent = (
            matrix.ndim == 3
            and initial.coefficients is not None
            and initial.design_matrix is not None
        )
        expected = (
            "a (n_time, n_states, n_states) transition_matrix with its "
            "coefficients and design_matrix"
        )
    if not consistent:
        msg = (
            f"{name}.make_state_transition does not match "
            f"{name}.is_stationary={transition.is_stationary}"
        )
        raise ValidationError(
            msg,
            expected=expected,
            got=(
                f"transition_matrix with shape {matrix.shape}, coefficients "
                f"{'set' if initial.coefficients is not None else 'None'}, "
                "design_matrix "
                f"{'set' if initial.design_matrix is not None else 'None'}"
            ),
        )
    if matrix.shape[-2:] != (n_states, n_states):
        msg = f"{name}.make_state_transition returned the wrong number of states"
        raise ValidationError(
            msg,
            expected=f"{n_states} x {n_states} transition matrices",
            got=f"transition_matrix with shape {matrix.shape}",
        )


def _validate_diagonal_values(diagonal_values: npt.ArrayLike, name: str) -> None:
    """Check that diagonal values are one probability per state.

    Raises
    ------
    ValidationError
        If the values are not one-dimensional or not in [0, 1].
    DataError
        If any value is NaN or infinite.
    """
    values = np.asarray(diagonal_values, dtype=float)
    val.ensure_array_1d(values, name)
    val.ensure_all_finite(values, name)
    val.ensure_in_range(values, name, 0.0, 1.0)


def _validate_transition_values(values: npt.ArrayLike, name: str) -> None:
    """Check that a custom matrix is a square row-stochastic matrix.

    Raises
    ------
    ValidationError
        If the matrix is not square, has negative values or values above 1,
        or a row that does not sum to 1.
    DataError
        If any value is NaN or infinite.
    """
    matrix = np.asarray(values, dtype=float)
    val.ensure_all_finite(matrix, name)
    val.ensure_square_matrix(matrix, name)
    val.ensure_all_non_negative(matrix, name)
    val.ensure_in_range(matrix, name, 0.0, 1.0)
    val.ensure_stochastic_matrix(matrix, name)


@dataclass(eq=False)
class DiscreteStationaryDiagonal:
    """Diagonal values are placed on the diagonal.

    Off-diagonals are probability: (1 - `diagonal_value`) / (`n_states` - 1)

    Attributes
    ----------
    diagonal_values : np.ndarray, shape (n_states,)
        The diagonal of the transition matrix, each in [0, 1].
    """

    diagonal_values: np.ndarray

    def __post_init__(self) -> None:
        _validate_diagonal_values(
            self.diagonal_values, "DiscreteStationaryDiagonal.diagonal_values"
        )

    @property
    def is_stationary(self) -> bool:
        """True: the transition does not depend on covariates."""
        return True

    @property
    def n_states(self) -> int:
        """Number of discrete states, the length of ``diagonal_values``."""
        return len(np.asarray(self.diagonal_values))

    @property
    def smoothness(self) -> float:
        """0.0: a stationary transition has no curvature penalty."""
        return 0.0

    def make_state_transition(
        self,
        covariate_data: pd.DataFrame | dict[str, Any] | None = None,  # noqa: ARG002 -- shared make_state_transition signature
    ) -> InitialDiscreteTransition:
        """Construct the initial discrete transition matrix.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict, optional
            Ignored; a stationary transition has no covariates.

        Returns
        -------
        initial : InitialDiscreteTransition
            ``transition_matrix`` of shape (n_states, n_states), and None
            ``coefficients`` and ``design_matrix`` because the transition is
            stationary.
        """
        diag = np.asarray(self.diagonal_values)
        return InitialDiscreteTransition(make_transition_from_diag(diag), None, None)


@dataclass(eq=False)
class DiscreteStationaryCustom:
    """Create a custom discrete transition matrix.

    Attributes
    ----------
    values : np.ndarray, shape (n_states, n_states)
        The transition matrix values: non-negative, and each row sums to 1.
    """

    values: np.ndarray

    def __post_init__(self) -> None:
        _validate_transition_values(self.values, "DiscreteStationaryCustom.values")

    @property
    def is_stationary(self) -> bool:
        """True: the transition does not depend on covariates."""
        return True

    @property
    def n_states(self) -> int:
        """Number of discrete states, the rows of ``values``."""
        return len(np.asarray(self.values))

    @property
    def smoothness(self) -> float:
        """0.0: a stationary transition has no curvature penalty."""
        return 0.0

    def make_state_transition(
        self,
        covariate_data: pd.DataFrame | dict[str, Any] | None = None,  # noqa: ARG002 -- shared make_state_transition signature
    ) -> InitialDiscreteTransition:
        """Construct the initial discrete transition matrix.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict, optional
            Ignored; a stationary transition has no covariates.

        Returns
        -------
        initial : InitialDiscreteTransition
            ``transition_matrix`` of shape (n_states, n_states), and None
            ``coefficients`` and ``design_matrix`` because the transition is
            stationary.
        """
        return InitialDiscreteTransition(np.asarray(self.values), None, None)


def _initial_nonstationary_transition(
    discrete_transition: np.ndarray,
    formula: str,
    covariate_data: pd.DataFrame | dict[str, Any] | None,
    smoothness: float | str,
) -> InitialDiscreteTransition:
    """Initial structures of a covariate-driven transition from its intercept matrix.

    Parameters
    ----------
    discrete_transition : np.ndarray, shape (n_states, n_states)
        Transition matrix the intercept coefficients reproduce.
    formula : str
        Patsy formula for the transition design matrix.
    covariate_data : pd.DataFrame or dict
        Covariates named in ``formula``, one row per time step.
    smoothness : float or str
        The model's ``smoothness`` setting, checked against the design.

    Returns
    -------
    initial : InitialDiscreteTransition
        ``transition_matrix`` of shape (n_time, n_states, n_states),
        ``coefficients`` of shape (n_coefficients, n_states, n_states - 1)
        and the patsy ``design_matrix``.
    """
    n_states = len(discrete_transition)

    # NA_action="raise" prevents patsy from silently dropping rows with NaN
    # covariates, which would misalign the design matrix (n_time) against the
    # full-length posteriors/responses in the M-step.
    discrete_transition_design_matrix = dmatrix(
        formula, covariate_data, NA_action="raise"
    )
    if discrete_transition_design_matrix.shape[0] == 0:
        msg = "No covariate data provided for the transition matrix."
        raise ValueError(msg)
    _check_smoothness_design(smoothness, discrete_transition_design_matrix)

    n_time, n_coefficients = discrete_transition_design_matrix.shape

    discrete_transition_coefficients = np.zeros(
        (n_coefficients, n_states, n_states - 1)
    )
    discrete_transition_coefficients[0] = centered_softmax_inverse(discrete_transition)

    discrete_transition = discrete_transition[np.newaxis] * np.ones(
        (n_time, n_states, n_states)
    )

    return InitialDiscreteTransition(
        discrete_transition,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
    )


@dataclass(eq=False)
class DiscreteNonStationaryDiagonal:
    """Non-stationary transition matrix driven by covariates.

    Initialized with a stationary diagonal matrix, then coefficients are estimated.
    Off-diagonals are uniform based on the diagonal value at each time step.

    Attributes
    ----------
    diagonal_values : np.ndarray, shape (n_states,)
        Initial diagonal probabilities, each in [0, 1], used to set intercept
        coefficients.
    formula : str, optional
        Patsy formula defining the relationship between covariates and
        transitions. Defaults to a cubic B-spline of ``speed`` with five
        interior knots at quantiles of the fitted speeds
        (``quantile_bs``); speeds beyond the fitted range take the
        boundary's value. Pass explicit knots with e.g.
        ``"1 + bs(speed, knots=[1.0, 4.0, 16.0])"``.
    smoothness : float or "reml", optional
        Strength of a penalty on the curvature of every ``quantile_bs`` and
        ``cyclic_bs`` term, ``integral f''(u)^2 du`` with the covariate
        rescaled to [0, 1] between the spline's boundaries (over the period
        for ``cyclic_bs``); by default 0.0 (none). "reml" chooses each row's
        strength for each term in every EM M-step by maximizing the row's
        Laplace approximate marginal likelihood. A smoothness penalty needs a
        ``quantile_bs`` or ``cyclic_bs`` term in ``formula``; use
        ``cyclic_bs(x, lower=..., upper=...)`` for a cyclic covariate such as
        phase or head direction.
    """

    diagonal_values: np.ndarray
    formula: str = "1 + quantile_bs(speed)"
    smoothness: float | str = 0.0

    def __post_init__(self) -> None:
        _validate_diagonal_values(
            self.diagonal_values, "DiscreteNonStationaryDiagonal.diagonal_values"
        )
        _validate_smoothness(self.smoothness)

    @property
    def is_stationary(self) -> bool:
        """False: the transition depends on covariates."""
        return False

    @property
    def n_states(self) -> int:
        """Number of discrete states, the length of ``diagonal_values``."""
        return len(np.asarray(self.diagonal_values))

    def make_state_transition(
        self, covariate_data: pd.DataFrame | dict[str, Any] | None
    ) -> InitialDiscreteTransition:
        """Construct the initial non-stationary discrete transition structures.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict
            Data containing covariates specified in the formula. Must have
            length matching the number of time steps.

        Returns
        -------
        initial : InitialDiscreteTransition
            ``transition_matrix``, shape (n_time, n_states, n_states): the
            initial guess at each time step (based on the intercept).
            ``coefficients``, shape (n_coefficients, n_states, n_states - 1):
            initial coefficients, with intercepts set based on `diagonal_values`.
            ``design_matrix``: the patsy design matrix derived from the
            formula and covariate data.
        """
        return _initial_nonstationary_transition(
            make_transition_from_diag(self.diagonal_values),
            self.formula,
            covariate_data,
            self.smoothness,
        )


@dataclass(eq=False)
class DiscreteNonStationaryCustom:
    """Non-stationary transition matrix driven by covariates, with custom initial values.

    Initialized with a custom stationary matrix, then coefficients are estimated.

    Attributes
    ----------
    values : np.ndarray, shape (n_states, n_states)
        Initial stationary transition matrix used to set intercept coefficients:
        non-negative, and each row sums to 1.
    formula : str, optional
        Patsy formula defining the relationship between covariates and
        transitions. Defaults to a cubic B-spline of ``speed`` with five
        interior knots at quantiles of the fitted speeds
        (``quantile_bs``); speeds beyond the fitted range take the
        boundary's value. Pass explicit knots with e.g.
        ``"1 + bs(speed, knots=[1.0, 4.0, 16.0])"``.
    smoothness : float or "reml", optional
        Strength of a penalty on the curvature of every ``quantile_bs`` and
        ``cyclic_bs`` term, ``integral f''(u)^2 du`` with the covariate
        rescaled to [0, 1] between the spline's boundaries (over the period
        for ``cyclic_bs``); by default 0.0 (none). "reml" chooses each row's
        strength for each term in every EM M-step by maximizing the row's
        Laplace approximate marginal likelihood. A smoothness penalty needs a
        ``quantile_bs`` or ``cyclic_bs`` term in ``formula``; use
        ``cyclic_bs(x, lower=..., upper=...)`` for a cyclic covariate such as
        phase or head direction.
    """

    values: np.ndarray
    formula: str = "1 + quantile_bs(speed)"
    smoothness: float | str = 0.0

    def __post_init__(self) -> None:
        _validate_transition_values(self.values, "DiscreteNonStationaryCustom.values")
        _validate_smoothness(self.smoothness)

    @property
    def is_stationary(self) -> bool:
        """False: the transition depends on covariates."""
        return False

    @property
    def n_states(self) -> int:
        """Number of discrete states, the rows of ``values``."""
        return len(np.asarray(self.values))

    def make_state_transition(
        self, covariate_data: pd.DataFrame | dict[str, Any] | None
    ) -> InitialDiscreteTransition:
        """Construct the initial non-stationary discrete transition structures.

        Parameters
        ----------
        covariate_data : pd.DataFrame or dict
            Data containing covariates specified in the formula. Must have
            length matching the number of time steps.

        Returns
        -------
        initial : InitialDiscreteTransition
            ``transition_matrix``, shape (n_time, n_states, n_states): the
            initial guess at each time step (based on the intercept).
            ``coefficients``, shape (n_coefficients, n_states, n_states - 1):
            initial coefficients, with intercepts set based on `values`.
            ``design_matrix``: the patsy design matrix derived from the
            formula and covariate data.
        """
        return _initial_nonstationary_transition(
            self.values, self.formula, covariate_data, self.smoothness
        )


def predict_discrete_state_transitions(
    discrete_transition_design_matrix: DesignMatrix,
    discrete_transition_coefficients: np.ndarray,
    discrete_transition_covariate_data: pd.DataFrame | dict[str, Any],
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
    return _transitions_on_design(design_matrix, discrete_transition_coefficients)


def _transitions_on_design(
    design_matrix: np.ndarray | jax.Array, coefficients: np.ndarray | jax.Array
) -> np.ndarray:
    """Transition probabilities at the rows of a built design matrix.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)

    Returns
    -------
    transitions : np.ndarray, shape (n_time, n_states, n_states)
    """
    n_states = coefficients.shape[1]

    # The float64 probabilities the covariate M-step fits; each row has shape
    # (n_time, n_states), stacked to (n_time, n_states, n_states).
    with _float64_jax():
        design_matrix = jnp.asarray(np.asarray(design_matrix, dtype=np.float64))
        coefficients = jnp.asarray(np.asarray(coefficients, dtype=np.float64))
        rows = [
            jnp.exp(
                jax_centered_log_softmax_forward(
                    design_matrix @ coefficients[:, from_state, :]
                )
            )
            for from_state in range(n_states)
        ]
        return np.asarray(jnp.stack(rows, axis=1))


def predict_discrete_state_transition_intervals(
    discrete_transition_design_matrix: DesignMatrix,
    discrete_transition_coefficients: np.ndarray,
    discrete_transition_coefficient_covariance: np.ndarray,
    discrete_transition_covariate_data: pd.DataFrame | dict[str, Any],
    coverage: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise intervals of covariate-dependent transition probabilities.

    Each probability's interval is the delta-method normal interval of its
    log-odds, mapped back, so it stays inside (0, 1). The covariance is the
    Laplace covariance of the last M-step, which treats that M-step's
    expected transitions as known, so the intervals leave out the
    uncertainty in which state the animal was in.

    Parameters
    ----------
    discrete_transition_design_matrix : patsy.DesignMatrix
        Original design matrix used for fitting (contains design_info).
    discrete_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Fitted regression coefficients.
    discrete_transition_coefficient_covariance : np.ndarray, shape (n_states, n_parameters, n_parameters)
        Covariance of each row's flattened coefficients,
        ``n_parameters = n_coefficients * (n_states - 1)``, e.g. a detector's
        ``discrete_transition_coefficient_covariance_``.
    discrete_transition_covariate_data : pd.DataFrame or dict
        Covariate data to evaluate the intervals at.
    coverage : float, optional
        Pointwise coverage, by default 0.95.

    Returns
    -------
    lower, upper : np.ndarray, shape (n_new_time, n_states, n_states)
        NaN for rows whose covariance is NaN: their coefficients are not
        identified (e.g. a state without expected transitions and with a flat
        prior), so their uncertainty is unknown.
    """
    _, lower, upper = _transition_intervals(
        discrete_transition_design_matrix,
        discrete_transition_coefficients,
        discrete_transition_coefficient_covariance,
        discrete_transition_covariate_data,
        coverage,
    )
    return lower, upper


def _transition_intervals(
    discrete_transition_design_matrix: DesignMatrix,
    discrete_transition_coefficients: np.ndarray,
    discrete_transition_coefficient_covariance: np.ndarray,
    discrete_transition_covariate_data: pd.DataFrame | dict[str, Any],
    coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`predict_discrete_state_transition_intervals`, with the
    transition probabilities it centres them on.

    Returns
    -------
    transitions, lower, upper : np.ndarray, shape (n_new_time, n_states, n_states)
    """
    if not 0.0 < coverage < 1.0:
        msg = f"coverage must be in (0, 1), got {coverage}."
        raise ValueError(msg)
    design_matrix = np.asarray(
        build_design_matrices(
            [discrete_transition_design_matrix.design_info],
            discrete_transition_covariate_data,
        )[0],
        dtype=np.float64,
    )
    coefficients = np.asarray(discrete_transition_coefficients, dtype=np.float64)
    n_coefficients, n_states, n_free = coefficients.shape
    transitions = _transitions_on_design(design_matrix, coefficients)
    tiny = np.finfo(np.float64).tiny
    probabilities = np.clip(transitions, tiny, 1.0 - np.finfo(np.float64).eps)
    covariance = np.asarray(
        discrete_transition_coefficient_covariance, dtype=np.float64
    ).reshape(n_states, n_coefficients, n_free, n_coefficients, n_free)
    # Covariance of each row's free logits at each time.
    logit_covariance = np.einsum(
        "tc,scjdk,td->tsjk", design_matrix, covariance, design_matrix
    )
    # d logit(p_k) / d z_j = (1[k == j] - p_j) / (1 - p_k); the last state's
    # logit is fixed at 0.
    indicator = np.eye(n_states)[:, :n_free]
    gradient = (indicator - probabilities[:, :, np.newaxis, :n_free]) / (
        1.0 - probabilities[..., np.newaxis]
    )
    variance = np.einsum("tskj,tsjl,tskl->tsk", gradient, logit_covariance, gradient)
    half_width = norm.ppf(0.5 + coverage / 2.0) * np.sqrt(np.maximum(variance, 0.0))
    center = logit(probabilities)
    return transitions, expit(center - half_width), expit(center + half_width)
