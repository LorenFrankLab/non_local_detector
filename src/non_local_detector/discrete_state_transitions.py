"""Discrete state transition models.

This module defines the classes that specify the initial discrete-state
transition matrix of a detector (stationary diagonal or custom matrices, and
non-stationary matrices that depend on covariates through a design-matrix
formula), the patsy spline transforms such formulas use, and the post-hoc
stationary transition counts. The M-step that estimates a transition from
expected counts lives in :mod:`non_local_detector._transition_estimation`.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Protocol, cast, runtime_checkable

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
from scipy.special import expit, logit, softmax
from scipy.stats import norm

from non_local_detector import _validation as val
from non_local_detector._formula_grammar import (
    LoadedFormula,
    UntrustedFormula,
    check_formula,
    refuse_untrusted,
)
from non_local_detector._jax_typing import finfo, optimization_barrier
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


def _float64_jax() -> AbstractContextManager[Any]:
    """Provide a context in which JAX computes in float64.

    The scoped switch is ``jax.enable_x64`` in current JAX and
    ``jax.experimental.enable_x64`` in older releases such as 0.6.
    """
    if hasattr(jax, "enable_x64"):
        return cast(AbstractContextManager[Any], jax.enable_x64(new_val=True))
    from jax.experimental import enable_x64

    return cast(AbstractContextManager[Any], enable_x64(True))  # noqa: FBT003 -- jax.experimental.enable_x64 in older JAX


def centered_softmax_forward(y: np.ndarray) -> np.ndarray:
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate.

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
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate.

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


def _n_states_from_state_ind(state_ind: np.ndarray) -> int:
    """Return the number of discrete states represented by expanded bins."""
    return int(np.max(state_ind)) + 1


def _host_or_device(posterior: ArrayLike) -> np.ndarray | jax.Array:
    """Return a posterior as an array, leaving a device array on the device.

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


def _validate_stationary_discrete_transition_shape(
    discrete_transition_matrix: np.ndarray, n_states: int
) -> None:
    """Require a stationary discrete transition of shape (n_states, n_states)."""
    expected_shape = (n_states, n_states)
    if discrete_transition_matrix.shape != expected_shape:
        msg = (
            "discrete_transition_matrix must be stationary with shape "
            f"{expected_shape}, got shape {discrete_transition_matrix.shape}"
        )
        raise ValueError(msg)


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


def _safe_ratio_jax(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Return numerator / denominator with zero output at exactly zero denominators."""
    # Match core._divide_safe: only exact zeros are treated as unreachable bins.
    # An `isclose` threshold would zero legitimately small predictive
    # probabilities whose acausal/predictive ratio (the smoother backward
    # message) is O(1), silently dropping real transitions from the M-step.
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    return jnp.where(denominator != 0.0, numerator / safe_denominator, 0.0)


# Bytes of each posterior row block copied to the device for stationary
# transition counts, so device memory does not grow with recording length.
_PAIR_SUM_BLOCK_BYTES = 256 * 1024**2


# Rows summed per partial product of the count contraction. A single product
# over many similar terms accumulates float32 rounding in proportion to the
# rows (5.7e-4 relative over 50,000 rows); short partial sums added together
# keep it near the single-sum bound for this many rows.
_CONTRACTION_SUB_ROWS = 256


def _blocked_contraction(left: jax.Array, right: jax.Array) -> jax.Array:
    """``left.T @ right`` summed over the rows in short partial products.

    Parameters
    ----------
    left, right : jax.Array, shape (n_rows, n_bins)
        Matrices whose rows are contracted: the product is ``left.T @ right``.

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
    """Return expected discrete transition counts over a block of consecutive rows.

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
        Maps each state bin to its discrete state.
    n_states : int
        Number of discrete states.

    Returns
    -------
    counts : jax.Array, shape (n_states, n_states)
    """
    ratio = _safe_ratio_jax(acausal_next, predictive_next)
    maximum = jnp.max(ratio, axis=0, initial=0.0)
    # The column sum is at most n_rows * maximum; keep it 256x below the
    # largest finite value.
    headroom = finfo(ratio.dtype).max / 256
    exponent = jnp.ceil(
        jnp.log2(jnp.maximum(maximum / headroom * causal_posterior.shape[0], 1.0))
    )
    # Non-finite ratios propagate unscaled, as in an unscaled sum.
    exponent = jnp.where(jnp.isfinite(exponent), exponent, 0.0)
    scale = jnp.ldexp(jnp.ones((), ratio.dtype), exponent.astype(jnp.int32))
    contraction = _blocked_contraction(causal_posterior, ratio / scale)
    # Weight first: the weighted terms are bounded, the unweighted sums are not.
    weighted = optimization_barrier(transition_matrix * contraction)
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
        Maps each state bin to its discrete state.
    n_states : int
        Number of discrete states.

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
    _, n_state_bins = _validate_expanded_posterior_shapes(
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
    _validate_stationary_discrete_transition_shape(discrete_transition_matrix, n_states)
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
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate.

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


def _validate_smoothness(smoothness: float | np.ndarray | str) -> bool:
    """Check a smoothness value and return whether REML estimates the strengths.

    Valid values are strengths that are finite and >= 0, or ``"reml"``.
    """
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
    """Check that a smoothness penalty has a spline term to penalize.

    The penalty is REML or a positive strength.
    """
    penalized = isinstance(smoothness, str) or np.any(np.asarray(smoothness) > 0.0)
    if penalized and (smoothness_matrix is None or not np.any(smoothness_matrix)):
        msg = (
            "A smoothness penalty needs a quantile_bs or cyclic_bs term in the "
            "transition formula, e.g. '1 + quantile_bs(speed)'."
        )
        raise ValueError(msg)


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


def make_transition_from_diag(diag: np.ndarray) -> np.ndarray:
    """Make a transition matrix with diagonal probabilities `diag`.

    Off-diagonal probabilities are uniform for the remaining probability.

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
    """Integrate ``second(x).T @ second(x) dx`` between the first and last break.

    ``second(x)`` are basis second derivatives of shape (n_points, n_basis)
    that are polynomials between breaks.

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
    """Patsy stateful transform for a periodic cubic B-spline.

    Knots sit at data quantiles, for cyclic covariates such as phase or head
    direction.

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
        """Return knots extended by ``degree`` periods' worth on each side.

        The open B-spline basis over one period from the first knot can then
        be folded into the periodic one.
        """
        period, knots, degree = self._upper - self._lower, self._knots, self._degree
        extended = np.concatenate(
            [knots[-degree:] - period, knots, knots[: degree + 1] + period]
        )
        return extended, knots[0]

    def _fold(self, basis: np.ndarray) -> np.ndarray:
        """Fold the open basis into the periodic basis.

        Open basis function ``j + n_knots`` is function ``j`` one period later.
        """
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
        """Curvature-penalty strength of covariate-dependent rows.

        May be ``"reml"``; 0.0 for a stationary transition.
        """
        ...

    def make_state_transition(
        self, covariate_data: pd.DataFrame | dict[str, Any] | None
    ) -> InitialDiscreteTransition:
        """Return the transition EM starts from.

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
        Transition whose declared ``n_states`` and ``is_stationary`` the result
        is checked against.

    Raises
    ------
    ValidationError
        If the result's shape does not match ``n_states``, its coefficients,
        design matrix or number of dimensions do not match ``is_stationary``,
        or the matrix has a negative value or a row that does not sum to 1
        (within ``PROBABILITY_SUM_TOLERANCE``).
    DataError
        If the matrix has NaN or infinite values.
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
    label = f"{name}.make_state_transition(...).transition_matrix"
    val.ensure_all_finite(matrix, label)
    val.ensure_all_non_negative(matrix, label)
    val.ensure_stochastic_matrix(matrix.reshape(-1, n_states), label)


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
    """Return the initial structures of a covariate-driven transition from its intercept matrix.

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

    # A formula read from a saved model file (see load_model's
    # trust_formulas) is evaluated only within the grammar, naming only the
    # covariates; patsy would resolve any other name in this frame.
    if isinstance(formula, UntrustedFormula):
        refuse_untrusted(formula, "build the transition design")
    if isinstance(formula, LoadedFormula):
        check_formula(formula, [] if covariate_data is None else list(covariate_data))
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
        ``"1 + bs(speed, knots=[1.0, 4.0, 16.0])"``. patsy evaluates the
        formula as Python; ``load_model`` checks a formula read from a saved
        model file before evaluating it (see its ``trust_formulas``).
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
        ``"1 + bs(speed, knots=[1.0, 4.0, 16.0])"``. patsy evaluates the
        formula as Python; ``load_model`` checks a formula read from a saved
        model file before evaluating it (see its ``trust_formulas``).
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
    design_matrix = _design_on(
        discrete_transition_design_matrix, discrete_transition_covariate_data
    )
    return _transitions_on_design(design_matrix, discrete_transition_coefficients)


def _design_on(
    discrete_transition_design_matrix: DesignMatrix,
    covariate_data: pd.DataFrame | dict[str, Any],
) -> DesignMatrix:
    """Evaluate a fitted transition design on new covariates.

    Parameters
    ----------
    discrete_transition_design_matrix : patsy.DesignMatrix
        Design matrix used for fitting (contains design_info).
    covariate_data : pd.DataFrame or dict
        Covariates named in the design's formula, one row per time step.

    Returns
    -------
    design_matrix : patsy.DesignMatrix, shape (n_new_time, n_coefficients)
    """
    design_info = discrete_transition_design_matrix.design_info
    return cast(DesignMatrix, build_design_matrices([design_info], covariate_data)[0])


def _refuse_untrusted_transition(transition: object, action: str) -> None:
    """Raise if ``transition``'s formula was read from a file and not trusted.

    Parameters
    ----------
    transition : object
        A discrete transition; one without a ``formula`` passes.
    action : str
        What would evaluate the formula, for the message.

    Raises
    ------
    ValidationError
        If the formula is an ``UntrustedFormula``.
    """
    formula = getattr(transition, "formula", None)
    if isinstance(formula, UntrustedFormula):
        refuse_untrusted(formula, action)


def _transitions_on_design(
    design_matrix: np.ndarray | jax.Array, coefficients: np.ndarray | jax.Array
) -> np.ndarray:
    """Transition probabilities at the rows of a built design matrix.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Transition coefficients.

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
    """Return the intervals of :func:`predict_discrete_state_transition_intervals`.

    Also returns the transition probabilities they are centred on.

    Returns
    -------
    transitions, lower, upper : np.ndarray, shape (n_new_time, n_states, n_states)
    """
    if not 0.0 < coverage < 1.0:
        msg = f"coverage must be in (0, 1), got {coverage}."
        raise ValueError(msg)
    design_matrix = np.asarray(
        _design_on(
            discrete_transition_design_matrix, discrete_transition_covariate_data
        ),
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
