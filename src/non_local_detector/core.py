"""Core HMM forward-backward algorithms in JAX.

This module implements filtering (the forward pass), smoothing (the backward
pass) and Viterbi decoding for hidden Markov models whose states are discrete
states crossed with spatial bins, in log-space for numerical stability. It
includes chunked drivers that bound memory on long recordings, variants with
covariate-dependent discrete transitions, and the EM convergence check.
Adapted from dynamax (https://github.com/probml/dynamax) with modifications.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, TypeAlias, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from non_local_detector._validation import PROBABILITY_SUM_TOLERANCE
from non_local_detector.exceptions import ValidationError

# Internal module; only the decorator custom likelihoods use is public
# (docs/time_grid_migration.md).
__all__ = ["row_slice_aware"]


logger = logging.getLogger(__name__)

# Concrete array inputs: the functions below call array methods (``.sum``,
# ``.dtype``, indexing), which the scalar members of ``jax.typing.ArrayLike``
# do not provide.
ArrayLike: TypeAlias = jax.Array | np.ndarray


## NOTE: adapted from dynamax: https://github.com/probml/dynamax/ with modifications ##
def _normalize(u: ArrayLike, axis: int = 0) -> tuple[jax.Array, jax.Array]:
    """Normalize the values within the axis in a way that they sum up to 1.

    Divides by the exact sum so any positive total mass — down to the smallest
    normal value of the dtype — normalizes to one. A slice whose sum is exactly
    zero is returned unchanged (all zeros stay all zeros) rather than producing
    NaN; a NaN sum propagates.

    Parameters
    ----------
    u : jax.Array
        Array to normalize
    axis : int, optional
        Axis to normalize, by default 0

    Returns
    -------
    normalized_u : jax.Array
        Normalized array
    c : jax.Array
        Normalization constant (squeezed along normalized axis)
    """
    c = u.sum(axis=axis, keepdims=True)
    u = u / jnp.where(c == 0.0, 1.0, c)
    return cast(jax.Array, u), cast(jax.Array, c.squeeze(axis))


# Helper functions for the two key filtering steps
def _condition_on_primal(
    probs: ArrayLike, ll: ArrayLike
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Stable Bayes update plus the shifts its derivative rule reuses.

    This is the forward pass of :func:`_condition_on`. Do not differentiate it
    directly: its ``probs > 0`` masks make plain autodiff wrong at zero prior
    mass. ``_condition_on_jvp`` supplies the derivatives and calls this only
    for the auxiliary values, which share the forward calculation so ordinary
    filtering never evaluates the derivative rule.

    Parameters
    ----------
    probs : jax.Array, shape (n_states,)
        Predicted state probabilities (sum to 1).
    ll : jax.Array, shape (n_states,)
        Log-likelihood at this time step.

    Returns
    -------
    result : tuple[jax.Array, jax.Array]
        ``(new_probs, log_norm)`` exactly as :func:`_condition_on` returns them.
    auxiliaries : tuple[jax.Array, jax.Array, jax.Array]
        ``(ll_shift, centered_log_norm, is_zero_norm)``: the reachable-state
        log-likelihood shift, ``log_norm - ll_shift`` (kept separate so the
        derivative rule can form ``ll - log Z`` without cancellation at large
        likelihood offsets), and the scalar zero-normalizer flag.
    """
    reachable = probs > 0
    ll_max = jnp.where(reachable, ll, -jnp.inf).max()
    ll_shift = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    ll_shifted = ll - ll_shift
    log_joint_max = jnp.where(reachable, _safe_log(probs) + ll_shifted, -jnp.inf).max()
    joint_shift = jnp.where(jnp.isfinite(log_joint_max), log_joint_max, 0.0)
    # Floor the second shift so exp(-joint_shift) <= 1 / tiny cannot overflow
    # when the dominant prior mass is itself subnormal. (XLA CPU flushes
    # subnormals to zero, so there they are simply unreachable; the floor
    # matters on backends that preserve them.)
    joint_shift = jnp.maximum(
        joint_shift,
        jnp.log(jnp.finfo(probs.dtype).tiny),  # type: ignore[no-untyped-call]  # jax.numpy.finfo is untyped
    )

    # The first shift excludes unreachable maxima; the second rescales tiny
    # prior mass before exponentiation. Multiply the prior in probability
    # space to retain accuracy when its logarithm would lose significant bits.
    # Finite unreachable likelihoods cannot affect the forward result. Mask
    # them before exp, testing the ORIGINAL likelihood so subtraction overflow
    # is not mistaken for invalid input. Actual NaN/+inf inputs stay visible.
    exponent = jnp.where(reachable | ~jnp.isfinite(ll), ll_shifted - joint_shift, 0.0)
    posterior, norm = _normalize(probs * jnp.exp(exponent))
    centered_log_norm = jnp.log(norm) + joint_shift
    is_zero_norm = norm == 0.0
    new_probs = jnp.where(is_zero_norm, probs, posterior)
    log_norm = jnp.where(is_zero_norm, -jnp.inf, centered_log_norm + ll_shift)
    return (new_probs, log_norm), (ll_shift, centered_log_norm, is_zero_norm)


@jax.custom_jvp
def _condition_on(probs: ArrayLike, ll: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Condition on new emissions without numerical underflow.

    The emissions are given as log likelihoods for each discrete state.

    Parameters
    ----------
    probs : jax.Array
        Predicted state probabilities (sum to 1).
    ll : jax.Array
        Log-likelihood at this time step. Same shape as ``probs``.

    Returns
    -------
    new_probs : jax.Array
        Updated state probabilities. When the Bayes normalizer is exactly zero
        — either every entry of ``ll`` is ``-inf`` (all states impossible) or
        the predicted ``probs`` place zero mass on every state with finite
        likelihood (a prediction/likelihood support mismatch) — the predicted
        probabilities are returned unchanged and ``log_norm`` is set to
        ``-inf`` to mark the step in the marginal likelihood, rather than
        propagating an all-zero (invalid) posterior forward. A ``NaN`` in
        ``ll`` (a likelihood-computation bug, not impossible data) is NOT
        masked: it makes the normalizer ``NaN`` (not zero), so it propagates
        into ``new_probs`` and stays visible to the caller. A ``+inf`` in
        ``ll`` (also invalid) likewise yields non-finite output.
    log_norm : float
        Log normalization constant, or ``-inf`` when the normalizer is zero.

    Notes
    -----
    Derivatives come from the explicit rule in ``_condition_on_jvp`` (this is
    a :func:`jax.custom_jvp`), so they are exact at zero prior mass, at exactly
    tied likelihoods, and under large common likelihood offsets; the plain
    autodiff of the forward pass is not.

    A subnormal predicted probability counts as zero on backends that flush
    subnormals (XLA CPU does), so the same inputs can differ across devices at
    the smallest-normal boundary.
    """
    return _condition_on_primal(probs, ll)[0]


@_condition_on.defjvp
def _condition_on_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Define the derivative rule for :func:`_condition_on`.

    With ``r = exp(ll) / Z`` and posterior ``q``, the tangent of the normalized
    weight ``w / Z`` (``Z`` held fixed) is ``r * d(probs) + q * d(ll)``. Its
    sum is ``d(log Z)``; subtracting ``q * d(log Z)`` gives ``d(q)``. This
    holds at zero prior mass and at exactly tied likelihoods, where the plain
    autodiff of the masked forward pass does not.

    Parameters
    ----------
    primals : tuple[jax.Array, jax.Array]
        ``(probs, ll)``, each of shape ``(n_states,)``.
    tangents : tuple[jax.Array, jax.Array]
        ``(d(probs), d(ll))``; ``d(ll)`` may be a ``float0`` zero for integer
        likelihoods.

    Returns
    -------
    primal_out : tuple[jax.Array, jax.Array]
        ``(new_probs, log_norm)``.
    tangent_out : tuple[jax.Array, jax.Array]
        ``(d(new_probs), d(log_norm))``. At the zero-normalizer fallback the
        forward output is ``(probs, -inf)``, so the tangent is ``(d(probs),
        0)``; as with any selected branch, a NaN in the unselected ``d(ll)``
        is dropped there. A derivative ``r_i`` too large for the dtype is
        saturated at the dtype maximum per state, so a zero tangent stays zero
        instead of ``0 * inf``; a nonzero tangent along such a state is then
        finite but understated, and two saturated states in one tangent
        direction can still sum past the dtype's range.
    """
    probs, ll = primals
    probs_dot, ll_dot = tangents
    if ll_dot.dtype == jax.dtypes.float0:
        ll_dot = jnp.zeros_like(ll, dtype=probs.dtype)
    # Call the decorated function, not ``_condition_on_primal``: under
    # higher-order differentiation this recursion is what routes d(log Z)
    # through the rule again, so second derivatives at zero prior mass stay
    # correct. Reusing the primal's result here silently corrupts them.
    posterior, log_norm = _condition_on(probs, ll)
    _, (ll_shift, centered_value, is_zero_norm) = _condition_on_primal(probs, ll)

    # Keep the centered value to avoid cancellation in ll - log Z at large
    # likelihood offsets. The zero-valued term carries the derivative of the
    # decorated evidence output, so higher derivatives also include zero-prior
    # states. Numerical shifts are constants for differentiation.
    finite_log_norm = jnp.where(jnp.isfinite(log_norm), log_norm, 0.0)
    centered_log_norm = jax.lax.stop_gradient(centered_value) + (
        finite_log_norm - jax.lax.stop_gradient(finite_log_norm)
    )
    exponent = (ll - jax.lax.stop_gradient(ll_shift)) - centered_log_norm
    # At a genuine zero normalizer, -inf - (-inf) can be NaN. Make the unused
    # exponential finite before selecting the fallback tangent, including in
    # reverse mode, where a zero cotangent times NaN would still be NaN.
    exponent = jnp.where(is_zero_norm, 0.0, exponent)
    # Saturate a normalized derivative that cannot fit the dtype so zero
    # tangents stay finite (see the docstring for what this costs nonzero
    # ones). An invalid +inf/NaN input already made ``exponent`` NaN above and
    # is never repaired here. A strict ``where`` rather than ``jnp.minimum``:
    # at an exact tie ``minimum`` splits its derivative between operands and
    # would halve second derivatives of a derivative that sits on the bound.
    finfo = jnp.finfo(probs.dtype)  # type: ignore[no-untyped-call]  # jax.numpy.finfo is untyped
    max_exponent = jnp.nextafter(jnp.log(finfo.max), jnp.asarray(0.0, probs.dtype))
    exponent = jnp.where(exponent > max_exponent, max_exponent, exponent)
    weight_dot = jnp.exp(exponent) * probs_dot + posterior * ll_dot
    log_norm_dot = weight_dot.sum()
    posterior_dot = weight_dot - posterior * log_norm_dot

    # The selected operational fallback returns the prior and constant -inf.
    return (posterior, log_norm), (
        jnp.where(is_zero_norm, probs_dot, posterior_dot),
        jnp.where(is_zero_norm, 0.0, log_norm_dot),
    )


def _degenerate_and_nan_masks(
    log_likelihoods: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep boolean masks for degenerate and NaN log-likelihoods.

    Computes both masks here (a ``max(axis=-1)`` reduction and an ``isnan``
    reduction) and pulls them to the host in one place, so callers derive
    counts *and* indices from a single host sync point per call instead of the
    previous three separate helper calls (each with its own reduction and
    transfer).

    Parameters
    ----------
    log_likelihoods : array-like
        Log-likelihood array. The last axis is the state axis; all leading
        axes are treated as time-step axes.

    Returns
    -------
    degenerate_mask : np.ndarray of bool
        True where every state is ``-inf`` (impossible data). Uses
        ``max(axis=-1) == -inf`` rather than ``~isfinite`` so NaN steps are
        excluded (NaN propagates to the max, and ``NaN == -inf`` is False).
    nan_mask : np.ndarray of bool
        True where any state is ``NaN`` — a likelihood-computation bug, distinct
        from impossible data. ``_condition_on`` lets the NaN reach the posterior
        so it stays visible.
    """
    degenerate_mask, nan_mask = jax.device_get(
        _degenerate_and_nan_masks_device(jnp.asarray(log_likelihoods))
    )
    return degenerate_mask, nan_mask


@jax.jit
def _degenerate_and_nan_masks_device(
    log_likelihoods: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Device-side masks of ``_degenerate_and_nan_masks`` in one fused pass."""
    return (
        log_likelihoods.max(axis=-1) == -jnp.inf,
        jnp.any(jnp.isnan(log_likelihoods), axis=-1),
    )


def _accumulate_chunk_degeneracy(
    log_likelihood_chunk: ArrayLike,
    time_inds: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Find a chunk's degenerate and NaN steps at one host sync point.

    Returns the *global* indices of the chunk's all-``-inf`` steps
    (``time_inds`` maps chunk positions to global time indices) and the
    chunk's number of NaN steps.
    """
    degenerate_mask, nan_mask = _degenerate_and_nan_masks(log_likelihood_chunk)
    return np.asarray(time_inds)[degenerate_mask], int(nan_mask.sum())


# Per-algorithm phrasing for the degeneracy diagnostics. Filtering and Viterbi
# react differently to the same degenerate log-likelihoods, so the consequence
# clause is selected by ``context`` (a filter-worded message is misleading for a
# Viterbi decode, which has no posterior and no predicted-distribution fallback).
_ALGO_LABEL = {"filter": "filtering", "viterbi": "Viterbi decoding"}
_DEGENERATE_CONSEQUENCE = {
    "filter": "the posterior at those steps fell back to the predicted distribution",
    "viterbi": (
        "the most likely state there is arbitrary (argmax over an all-(-inf) "
        "column returns state 0)"
    ),
}
_NAN_CONSEQUENCE = {
    "filter": "the posterior is NaN at those steps and propagates to every later step",
    "viterbi": "the most likely sequence is undefined from those steps onward",
}


def _warn_degenerate_and_nan_timesteps(
    n_degenerate: int,
    n_nan: int,
    n_total: int,
    *,
    marginal_log_likelihood: float | None = None,
    context: str = "filter",
) -> None:
    """Emit host-side warnings summarizing degenerate and NaN time steps.

    These diagnostics use ``logger.warning`` (not ``warnings.warn``) because
    they are per-invocation decoding diagnostics — they should fire on every
    affected call rather than be deduplicated by the warnings filter. The
    fit-time convergence diagnostics use ``warnings.warn(NumericalWarning)`` instead
    (see ``models.base`` / the likelihood models).

    Parameters
    ----------
    n_degenerate, n_nan, n_total : int
        Counts of all-``-inf`` steps, NaN steps, and total steps.
    marginal_log_likelihood : float, optional
        The marginal log-likelihood, for filtering paths only. When it is
        ``-inf`` yet no step was all-``-inf`` (``n_degenerate == 0``), a zero
        Bayes normalizer arose from a prediction/likelihood *support mismatch*
        or underflow (finite log-likelihoods, but the predicted state
        distribution placed ~no mass on the supported states). ``_condition_on``
        marks such steps ``-inf`` in the marginal but they are not all-``-inf``,
        so this is the only place they are surfaced. Viterbi has no marginal and
        passes ``None``.
    context : {"filter", "viterbi"}, optional
        Selects algorithm-appropriate consequence wording.
    """
    label = _ALGO_LABEL[context]
    if n_degenerate > 0:
        logger.warning(
            "HMM %s encountered %d/%d time step(s) with all-impossible (-inf) "
            "log-likelihoods; %s.",
            label,
            n_degenerate,
            n_total,
            _DEGENERATE_CONSEQUENCE[context],
        )
    if n_nan > 0:
        logger.warning(
            "HMM %s encountered %d/%d time step(s) with NaN log-likelihoods; %s. "
            "This indicates a bug in the likelihood computation (e.g. a "
            "non-converged or degenerate encoding model), not impossible data. "
            "Check the log-likelihood inputs.",
            label,
            n_nan,
            n_total,
            _NAN_CONSEQUENCE[context],
        )
    if (
        marginal_log_likelihood is not None
        and n_degenerate == 0
        and np.isneginf(marginal_log_likelihood)
    ):
        logger.warning(
            "HMM %s produced a -inf marginal log-likelihood with no all-(-inf) "
            "time step: a zero Bayes normalizer arose from a prediction/likelihood "
            "support mismatch or underflow (finite log-likelihoods, but the "
            "predicted state distribution placed ~no mass on the supported "
            "states). The affected step(s) are marked -inf in the marginal "
            "likelihood.",
            label,
        )


def _is_traced(*values: object) -> bool:
    """Whether any value is a JAX tracer (inside ``jit``, ``vmap`` or ``grad``).

    Host-side checks and diagnostics need concrete values, so they skip
    traced inputs, keeping the public passes JAX-transformable.
    """
    return any(isinstance(value, jax.core.Tracer) for value in values)


def _warn_if_degenerate_timesteps(
    log_likelihoods: ArrayLike,
    *,
    marginal_log_likelihood: ArrayLike | None = None,
    context: str = "filter",
) -> None:
    """Emit host-side warnings if any time step is all-``-inf`` or has a NaN.

    Parameters
    ----------
    log_likelihoods : array-like
        Log-likelihood array. The last axis is the state axis.
    marginal_log_likelihood : array-like, optional
        Marginal log-likelihood (filtering paths only); forwarded to
        ``_warn_degenerate_and_nan_timesteps`` to surface support-mismatch /
        underflow steps. Viterbi drivers omit it.
    context : {"filter", "viterbi"}, optional
        Selects algorithm-appropriate consequence wording.

    Notes
    -----
    Skipped when ``log_likelihoods`` is a JAX tracer (i.e. the wrappers are
    being run inside ``jit``/``vmap``/``scan``). The diagnostics need concrete
    values (``int(jnp.sum(...))``) and warnings cannot be emitted during
    tracing, so skipping keeps the public wrappers JAX-transformable; the pure
    computation still runs.
    """
    ll_array = jnp.asarray(log_likelihoods)
    if _is_traced(ll_array):
        return
    n_total = int(np.prod(ll_array.shape[:-1])) if ll_array.ndim > 1 else 1
    degenerate_mask, nan_mask = _degenerate_and_nan_masks(ll_array)
    marginal = (
        None if marginal_log_likelihood is None else float(marginal_log_likelihood)
    )
    _warn_degenerate_and_nan_timesteps(
        int(degenerate_mask.sum()),
        int(nan_mask.sum()),
        n_total,
        marginal_log_likelihood=marginal,
        context=context,
    )


def _safe_log(p: ArrayLike) -> jax.Array:
    """Compute log of probabilities safely, handling zeros.

    Returns -inf for zero probabilities (valid in log space) instead of NaN.

    Parameters
    ----------
    p : jax.Array
        Probability array (may contain zeros)

    Returns
    -------
    log_p : jax.Array
        Log probabilities with -inf for zeros
    """
    # Double-where: substitute safe value into operand first
    safe_p = jnp.where(p > 0, p, 1.0)
    return jnp.where(p > 0, jnp.log(safe_p), -jnp.inf)


def _divide_safe(numerator: ArrayLike, denominator: ArrayLike) -> jax.Array:
    """Divide two arrays, setting the result to 0.0 where the denominator is 0.0.

    Parameters
    ----------
    numerator : jax.Array
        Numerator array.
    denominator : jax.Array
        Denominator array.

    Returns
    -------
    result : jax.Array
        Element-wise division result with safe handling of zero denominators.
    """
    # Double-where: substitute safe denominator first, then select result.
    # This avoids NaN in both forward pass and gradients.
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    return jnp.where(denominator != 0.0, numerator / safe_denominator, 0.0)


# Time steps per scan iteration in the dense filter and smoother. On GPU each
# step is a few small kernels, so the per-iteration loop overhead dominates
# small state spaces: unrolling 8 steps made 514-bin scans 1.7x faster on an
# A100 (21.7 -> 12.6 us per filter step) with no change on CPU. Larger unrolls
# compile more slowly without further gain.
_HMM_SCAN_UNROLL = 8


## Transition operators and the shared passes ##
# Every HMM pass (filter, smoother, Viterbi) has one scan body below. The
# transition model enters only through an operator: the filter and smoother
# need ``TransitionProducts``, and the smoother's transition statistics need a
# full ``TransitionOperator``. Viterbi uses the dense operators' ``log()``
# twin, whose ``scores`` gives the max-plus terms.


@runtime_checkable
class TransitionProducts(Protocol):
    """Products with a joint transition ``T`` between state bins.

    The minimum interface of an operator for filtering and smoothing: the
    passes see the transition model only through these products, so a
    structured operator never has to build ``T``. An operator is a registered
    JAX pytree whose methods are pure JAX. ``weight`` is None for a stationary
    operator or the row's (n_states, n_states) discrete transition matrix,
    which scales ``T[i, j]`` by ``weight[s_i, s_j]`` (``s`` the discrete state
    of each bin); the checkpointed driver checks the rows on that convention.
    Expected discrete transition counts need a :class:`TransitionOperator`.
    """

    def forward(
        self, probabilities: jax.Array, weight: jax.Array | None, /
    ) -> jax.Array:
        """``probabilities @ T``, shape (n_state_bins,)."""
        ...

    def backward(self, values: jax.Array, weight: jax.Array | None, /) -> jax.Array:
        """``T @ values``, shape (n_state_bins,)."""
        ...


@runtime_checkable
class TransitionOperator(TransitionProducts, Protocol):
    """Transition products plus the per-state-pair terms of EM's statistics.

    The interface an operator needs when the smoother also returns expected
    discrete transition counts. An implementer must define ``forward``,
    ``backward`` and ``pair_products``. ``backward_and_pairs`` is inherited,
    unfused, only by a class that subclasses this protocol; a class that
    satisfies it structurally, without subclassing, must define it too. An
    operator that shares work between the two products overrides it.

    Implementers are the dense ``_StationaryTransition`` and
    ``_CovariateTransition`` here and the structured
    ``BlockTransitionOperator`` and ``FusedBlockTransition`` in
    ``transition_operators``.
    """

    def pair_products(
        self, filtered: jax.Array, ratio: jax.Array, weight: jax.Array | None, /
    ) -> jax.Array:
        """Per discrete-state-pair terms of ``filtered @ T * ratio``.

        With a filtered row and the smoother ratio ``smoothed_{t+1} /
        predicted_{t+1}`` these are one step's expected discrete transition
        counts, shape (n_states, n_states).
        """
        ...

    def backward_and_pairs(
        self, values: jax.Array, filtered: jax.Array, weight: jax.Array | None, /
    ) -> tuple[jax.Array, jax.Array]:
        """``backward(values)`` and ``pair_products(filtered, values)``.

        Returns
        -------
        backward : jax.Array, shape (n_state_bins,)
        pairs : jax.Array, shape (n_states, n_states)
        """
        return self.backward(values, weight), self.pair_products(
            filtered, values, weight
        )


def _state_one_hot(state_ind: ArrayLike, n_states: int, dtype: Any) -> jax.Array:
    """One-hot discrete-state membership of each state bin.

    Parameters
    ----------
    state_ind : jax.Array, shape (n_state_bins,)
        Maps each state bin to its discrete state.
    n_states : int
        Number of discrete states.
    dtype : dtype
        Dtype of the returned one-hot matrix.

    Returns
    -------
    one_hot : jax.Array, shape (n_states, n_state_bins)
    """
    return (state_ind[None, :] == jnp.arange(n_states)[:, None]).astype(dtype)


def _covariate_forward_product(
    probs: ArrayLike,
    discrete_transition_matrix_t: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    one_hot: ArrayLike,
) -> jax.Array:
    """Propagate probabilities forward through one covariate-dependent step.

    Computes ``probs @ T`` for the joint transition
    ``T[i, j] = C[i, j] * D[s_i, s_j]`` without building ``T``: grouping the
    source bins by discrete state ``k`` gives
    ``out[j] = sum_k D[k, s_j] * (probs * (s == k)) @ C[:, j]``, so each step
    reads ``C`` once in a single (n_states, n_state_bins) matrix product.

    Parameters
    ----------
    probs : jax.Array, shape (n_state_bins,)
        Probability of each state bin before the step.
    discrete_transition_matrix_t : jax.Array, shape (n_states, n_states)
        Discrete transition matrix ``D`` of this step.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Continuous transition matrix ``C`` of this step.
    state_ind : jax.Array, shape (n_state_bins,)
        Maps each state bin to its discrete state.
    one_hot : jax.Array, shape (n_states, n_state_bins)
        ``_state_one_hot(state_ind, n_states, dtype)``.

    Returns
    -------
    propagated : jax.Array, shape (n_state_bins,)
    """
    # HIGHEST: the default CUDA precision (TF32) would round these products.
    per_state = jnp.matmul(
        one_hot * probs,
        continuous_transition_matrix,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sum(discrete_transition_matrix_t[:, state_ind] * per_state, axis=0)


def _covariate_backward_product(
    values: ArrayLike,
    discrete_transition_matrix_t: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    one_hot: ArrayLike,
) -> jax.Array:
    """Pull values backward through one covariate-dependent step.

    Computes ``T @ values`` for ``T[i, j] = C[i, j] * D[s_i, s_j]`` without
    building ``T``: ``out[i] = sum_k D[s_i, k] * C[i, :] @ (values * (s == k))``.

    Parameters
    ----------
    values : jax.Array, shape (n_state_bins,)
        Value of each state bin at the next time step.
    discrete_transition_matrix_t : jax.Array, shape (n_states, n_states)
        Discrete transition matrix ``D`` of this step.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Continuous transition matrix ``C`` of this step.
    state_ind : jax.Array, shape (n_state_bins,)
        Maps each state bin to its discrete state.
    one_hot : jax.Array, shape (n_states, n_state_bins)
        ``_state_one_hot(state_ind, n_states, dtype)``.

    Returns
    -------
    pulled : jax.Array, shape (n_state_bins,)
    """
    per_state = jnp.matmul(
        continuous_transition_matrix,
        (one_hot * values).T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sum(discrete_transition_matrix_t[state_ind, :] * per_state, axis=1)


class _LogStationaryTransition(NamedTuple):
    """Log-domain twin of :class:`_StationaryTransition` for Viterbi.

    Attributes
    ----------
    log_matrix : jax.Array, shape (n_state_bins, n_state_bins)
    """

    log_matrix: jax.Array

    def scores(
        self,
        best_next_score: jax.Array,
        next_log_likelihood: jax.Array,
        log_weight: None,  # noqa: ARG002 -- operator interface
    ) -> jax.Array:
        """Max-plus terms ``log T[i, j] + best_next[j] + ll_next[j]``.

        Returns
        -------
        scores : jax.Array, shape (n_state_bins, n_state_bins)
        """
        return self.log_matrix + best_next_score + next_log_likelihood


class _LogCovariateTransition(NamedTuple):
    """Log-domain twin of :class:`_CovariateTransition` for Viterbi.

    Attributes
    ----------
    log_continuous : jax.Array, shape (n_state_bins, n_state_bins)
    state_ind : jax.Array, shape (n_state_bins,)
    """

    log_continuous: jax.Array
    state_ind: jax.Array

    def scores(
        self,
        best_next_score: jax.Array,
        next_log_likelihood: jax.Array,
        log_weight: jax.Array,
    ) -> jax.Array:
        """Max-plus terms ``log C[i, j] + log D[s_i, s_j] + next[j]``.

        ``log D[s_i, s_j] + next[j]`` depends on ``i`` only through ``s_i``,
        so it is built per source state, (n_states, n_state_bins), and whole
        rows are gathered: a row gather is ~2x faster on CPU than the
        elementwise ``D[s_i, s_j]`` gather.

        Returns
        -------
        scores : jax.Array, shape (n_state_bins, n_state_bins)
        """
        next_score_by_source_state = (
            log_weight[:, self.state_ind] + best_next_score + next_log_likelihood
        )
        return self.log_continuous + next_score_by_source_state[self.state_ind]


@jax.tree_util.register_pytree_node_class
class _StationaryTransition(TransitionOperator):
    """Time-invariant joint transition between state bins, ``T[i, j]``.

    Parameters
    ----------
    matrix : jax.Array, shape (n_state_bins, n_state_bins)
    state_ind : jax.Array, shape (n_state_bins,), optional
        Discrete state of each bin; only ``pair_products`` needs it.
    n_states : int, optional
        Number of discrete states; only ``pair_products`` needs it.

    Notes
    -----
    The products use ``Precision.HIGHEST``, as the covariate and structured
    operators do: the default CUDA precision (TF32) would round them, so the
    paths would disagree on GPU. On CPU the precision has no effect.
    """

    def __init__(
        self,
        matrix: ArrayLike,
        state_ind: jax.Array | None = None,
        n_states: int | None = None,
    ) -> None:
        self.matrix, self.state_ind, self.n_states = matrix, state_ind, n_states

    def forward(
        self, probabilities: jax.Array, weight: jax.Array | None = None
    ) -> jax.Array:
        """``probabilities @ T``, shape (n_state_bins,)."""
        _reject_weight(weight)
        return jnp.matmul(
            probabilities, self.matrix, precision=jax.lax.Precision.HIGHEST
        )

    def backward(self, values: jax.Array, weight: jax.Array | None = None) -> jax.Array:
        """``T @ values``, shape (n_state_bins,)."""
        _reject_weight(weight)
        return jnp.matmul(self.matrix, values, precision=jax.lax.Precision.HIGHEST)

    def pair_products(
        self, filtered: jax.Array, ratio: jax.Array, weight: jax.Array | None = None
    ) -> jax.Array:
        """Per-state-pair terms of ``filtered @ T * ratio``, (n_states, n_states)."""
        _reject_weight(weight)
        if self.state_ind is None or self.n_states is None:
            msg = "pair_products needs state_ind and n_states"
            raise ValueError(msg)
        states = jax.nn.one_hot(self.state_ind, self.n_states, dtype=filtered.dtype)
        precision = jax.lax.Precision.HIGHEST
        source_rows = jnp.matmul(
            (states * filtered[:, None]).T, self.matrix, precision=precision
        )
        return jnp.matmul(source_rows, states * ratio[:, None], precision=precision)

    def log(self) -> _LogStationaryTransition:
        """Log-domain twin for Viterbi."""
        return _LogStationaryTransition(_safe_log(self.matrix))

    def tree_flatten(
        self,
    ) -> tuple[tuple[ArrayLike, jax.Array | None], int | None]:
        return (self.matrix, self.state_ind), self.n_states

    @classmethod
    def tree_unflatten(
        cls,
        aux: int | None,
        children: tuple[ArrayLike, jax.Array | None],
    ) -> "_StationaryTransition":
        return cls(*children, aux)


def _reject_weight(weight: object) -> None:
    """Trace-time guard: a stationary operator has no per-row weights."""
    if weight is not None:
        msg = "A stationary transition takes no per-row discrete weights"
        raise TypeError(msg)


def _require_weight(weight: jax.Array | None) -> jax.Array:
    """Trace-time guard: a covariate operator needs each row's discrete weights."""
    if weight is None:
        msg = "A covariate-dependent transition needs each row's discrete weights"
        raise TypeError(msg)
    return weight


@jax.tree_util.register_pytree_node_class
class _CovariateTransition(TransitionOperator):
    """Joint transition ``T[i, j] = C[i, j] * D_t[s_i, s_j]``, never built.

    Each product applies the row's discrete matrix ``D_t`` (the ``weight``)
    per discrete state on top of the continuous transition ``C``, so neither
    the per-step (n_state_bins, n_state_bins) joint transition nor the
    (n_time, n_state_bins, n_state_bins) series exists. Build it with
    :meth:`create`.

    Parameters
    ----------
    continuous : jax.Array, shape (n_state_bins, n_state_bins)
    state_ind : jax.Array, shape (n_state_bins,)
    one_hot : jax.Array, shape (n_states, n_state_bins)
        ``_state_one_hot(state_ind, n_states, continuous.dtype)``.
    """

    def __init__(
        self, continuous: jax.Array, state_ind: jax.Array, one_hot: jax.Array
    ) -> None:
        self.continuous, self.state_ind, self.one_hot = continuous, state_ind, one_hot

    @classmethod
    def create(
        cls, continuous: ArrayLike, state_ind: ArrayLike, n_states: int
    ) -> "_CovariateTransition":
        """Operator for ``continuous`` with ``n_states`` discrete states."""
        continuous, state_ind = jnp.asarray(continuous), jnp.asarray(state_ind)
        return cls(
            continuous,
            state_ind,
            _state_one_hot(state_ind, n_states, continuous.dtype),
        )

    def forward(self, probabilities: jax.Array, weight: jax.Array | None) -> jax.Array:
        """``probabilities @ T``, shape (n_state_bins,)."""
        return _covariate_forward_product(
            probabilities,
            _require_weight(weight),
            self.continuous,
            self.state_ind,
            self.one_hot,
        )

    def backward(self, values: jax.Array, weight: jax.Array | None) -> jax.Array:
        """``T @ values``, shape (n_state_bins,)."""
        return _covariate_backward_product(
            values,
            _require_weight(weight),
            self.continuous,
            self.state_ind,
            self.one_hot,
        )

    def pair_products(
        self, filtered: jax.Array, ratio: jax.Array, weight: jax.Array | None
    ) -> jax.Array:
        """Per-state-pair terms of ``filtered @ T * ratio``, (n_states, n_states).

        ``sum_{i in k, j in l} f_i C[i, j] D[k, l] r_j = D[k, l] *
        ((f * (s == k)) @ C @ (r * (s == l)))``.
        """
        precision = jax.lax.Precision.HIGHEST
        per_source = jnp.matmul(
            self.one_hot * filtered, self.continuous, precision=precision
        )
        return _require_weight(weight) * jnp.matmul(
            per_source, (self.one_hot * ratio).T, precision=precision
        )

    def log(self) -> _LogCovariateTransition:
        """Log-domain twin for Viterbi."""
        return _LogCovariateTransition(_safe_log(self.continuous), self.state_ind)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        return (self.continuous, self.state_ind, self.one_hot), None

    @classmethod
    def tree_unflatten(
        cls,
        aux: None,  # noqa: ARG003 -- pytree protocol
        children: tuple[jax.Array, jax.Array, jax.Array],
    ) -> "_CovariateTransition":
        return cls(*children)


def _filter_scan(
    initial: ArrayLike,
    evidence: ArrayLike | float,
    log_likelihoods: ArrayLike,
    weights: ArrayLike | None,
    operator: TransitionProducts,
    *,
    keep_rows: bool,
    keep_predictive: bool,
    keep_increments: bool = False,
) -> tuple[tuple[jax.Array, jax.Array], Any]:
    """Forward pass (filter) over consecutive rows: the one filter scan body.

    Parameters
    ----------
    initial : array, shape (n_state_bins,)
        Predicted distribution of the first row.
    evidence : scalar
        Evidence accumulated before the first row (the carried total starts here).
    log_likelihoods : array, shape (n_rows, n_state_bins)
        Log likelihood of each row's observations in each state bin.
    weights : array, shape (n_rows, n_states, n_states), or None
        Per-row discrete transitions handed to ``operator`` (None for a
        stationary operator).
    operator : TransitionProducts
        Transition products used to propagate the predicted distribution.
    keep_rows, keep_predictive, keep_increments : bool
        Which per-row outputs to emit (static under jit).

    Returns
    -------
    carry : tuple
        ``(evidence, predicted)``: the running evidence total in the working
        dtype and the predicted distribution of the row after the last.
    outputs : tuple, array or None
        With ``keep_rows``: ``(filtered, predicted or None, increments or
        None)``, each with a leading (n_rows,) axis. Otherwise the increments
        (each row's log normalizer) or None.
    """

    def step(carry: tuple[jax.Array, jax.Array], args: tuple[Any, Any]) -> Any:
        total, predicted = carry
        ll, weight = args
        filtered, increment = _condition_on(predicted, ll)
        out: Any
        if keep_rows:
            out = (
                filtered,
                predicted if keep_predictive else None,
                increment if keep_increments else None,
            )
        else:
            out = increment if keep_increments else None
        return (total + increment, operator.forward(filtered, weight)), out

    return jax.lax.scan(
        step,
        (evidence, initial),  # type: ignore[arg-type]  # carry may start as a Python float
        (log_likelihoods, weights),
        unroll=_HMM_SCAN_UNROLL,
    )


def _smoother_scan(
    filtered: ArrayLike,
    weights: ArrayLike | None,
    operator: TransitionProducts,
    next_smoothed: ArrayLike,
    rows: ArrayLike,
    n_time: int,
    *,
    pair_statistics: bool = False,
) -> tuple[jax.Array, Any]:
    """Backward pass (smoother) over consecutive rows: the one smoother scan body.

    Parameters
    ----------
    filtered : array, shape (n_rows, n_state_bins)
        Filtered distribution of each row.
    weights : array, shape (n_rows, n_states, n_states), or None
        Per-row discrete transitions handed to ``operator`` (None for a
        stationary operator).
    operator : TransitionProducts
        Transition products used to pull the smoothed row backward. A
        :class:`TransitionOperator` when ``pair_statistics`` is set.
    next_smoothed : array, shape (n_state_bins,)
        Smoothed row after the last row (any row of the right shape for the
        recording's final row, which is its own smoothed row).
    rows : array of int, shape (n_rows,)
        Recording row indices; the row ``n_time - 1`` is terminal.
    n_time : int
        Rows in the whole recording.
    pair_statistics : bool
        Also return each row's expected discrete transition counts to the
        next row, ``operator.pair_products`` of the filtered row and
        ``smoothed_{t+1} / predicted_{t+1}`` (zero on the terminal row),
        computed with ``operator.backward_and_pairs``.

    Returns
    -------
    first_smoothed : array, shape (n_state_bins,)
    smoothed : array, shape (n_rows, n_state_bins)
        Or ``(smoothed, pairs)`` with ``pair_statistics``, pairs of shape
        (n_rows, n_states, n_states).
    """
    paired: TransitionOperator | None = None
    if pair_statistics:
        if not isinstance(operator, TransitionOperator):
            msg = (
                "Transition statistics need a TransitionOperator; "
                f"{type(operator).__name__} has only the transition products"
            )
            raise TypeError(msg)
        paired = operator

    def step(following: jax.Array, args: tuple[Any, Any, Any]) -> Any:
        current, weight, row = args
        predicted = operator.forward(current, weight)
        ratio = _divide_safe(following, predicted)
        if paired is not None:
            backward, pairs = paired.backward_and_pairs(ratio, current, weight)
        else:
            backward = operator.backward(ratio, weight)
        smoothed = current * backward * (row < n_time - 1) + current * (
            row == n_time - 1
        )
        smoothed, _ = _normalize(smoothed, axis=-1)
        if not pair_statistics:
            return smoothed, smoothed
        pairs = jnp.where(row < n_time - 1, pairs, 0)
        return smoothed, (smoothed, pairs)

    return jax.lax.scan(
        step,
        next_smoothed,  # type: ignore[arg-type]  # NumPy input is converted
        (filtered, weights, rows),
        reverse=True,
        unroll=_HMM_SCAN_UNROLL,
    )


def _viterbi_scan(
    initial_distribution: ArrayLike,
    log_likelihoods: ArrayLike,
    log_weights: ArrayLike | None,
    log_operator: Any,
) -> jax.Array:
    """Most likely state-bin sequence: the one Viterbi scan body.

    Parameters
    ----------
    initial_distribution : array, shape (n_state_bins,)
        Initial probability of each state bin.
    log_likelihoods : array, shape (n_time, n_state_bins)
        Log likelihood of each time bin's observations in each state bin.
    log_weights : array, shape (n_time, n_states, n_states), or None
        Log per-row discrete transitions (None for a stationary operator).
    log_operator : log-domain operator
        ``operator.log()``: its ``scores(best_next, ll_next, log_weight)``
        returns the (n_state_bins, n_state_bins) max-plus terms.

    Returns
    -------
    most_likely_state_sequence : jax.Array, shape (n_time,)

    Notes
    -----
    Base case at t=T: terminal cost is 0, we accumulate from t=T-1 down.
    """
    log_initial_distribution = _safe_log(initial_distribution)

    def _backward_step(
        best_next_score: jax.Array, args: tuple[jax.Array, Any]
    ) -> tuple[jax.Array, jax.Array]:
        t, log_weight = args
        scores = log_operator.scores(
            best_next_score, log_likelihoods[t + 1], log_weight
        )
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    # The terminal cost takes the scores' dtype. A default-dtype zeros carry
    # is float64 under x64, which promoted every step of a float32 scan.
    score_dtype = jnp.result_type(
        log_likelihoods, *jax.tree_util.tree_leaves((log_operator, log_weights))
    )
    best_second_score, best_next_states = jax.lax.scan(
        _backward_step,
        jnp.zeros(num_states, dtype=score_dtype),
        (
            jnp.arange(num_timesteps - 1),
            None if log_weights is None else log_weights[:-1],
        ),
        reverse=True,
    )

    def _forward_step(
        state: jax.Array, best_next_state: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        log_initial_distribution + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_step, first_state, best_next_states)

    return jnp.concatenate([first_state[None], states])


def _filter_impl(
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """JIT-compiled forward pass of the forward-backward algorithm.

    See :func:`filter` for the user-facing wrapper that adds host-side
    diagnostics. This raw implementation runs :func:`_filter_scan` and is
    wrapped by both the non-donating ``_filter_jit`` (used by the public
    ``filter``) and the donating ``_filter_internal`` (used by the chunked
    drivers).

    The outputs are the filtered and predicted rows and each row's log
    normalizer (evidence increment). The carried total is a running sum in
    the working dtype, which the jit-composable :func:`filter` returns; the
    chunked drivers report :func:`_float64_evidence` of the increments.
    """
    return _filter_scan(
        initial_distribution,
        0.0,
        log_likelihoods,
        None,
        _StationaryTransition(transition_matrix),
        keep_rows=True,
        keep_predictive=True,
        keep_increments=True,
    )


def _float64_evidence(increments: ArrayLike) -> float:
    """Sum per-row evidence increments in float64.

    A float32 running total over a long recording drifts by many units (713
    over a million rows of about -17), enough to mislead EM's convergence and
    monotonicity checks; the increments themselves are float32-accurate.
    Infinities and NaNs propagate as in sequential addition.

    Parameters
    ----------
    increments : array-like, shape (n_time,)
        Per-row log normalizers (log evidence increments).

    Returns
    -------
    evidence : float
    """
    return float(np.sum(np.asarray(increments), dtype=np.float64))


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_filter_jit = jax.jit(_filter_impl)


# Internal version with buffer donation for use in chunked drivers
# This is safe because chunked drivers control the data flow and don't reuse donated arrays.
# All call sites pass arguments by keyword, so donation is specified by name to make
# intent unambiguous across JAX versions (positional donate_argnums + keyword calls is
# fragile on older JAX).
_filter_internal = jax.jit(
    _filter_impl, donate_argnames=("initial_distribution", "log_likelihoods")
)


def filter(
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
) -> tuple[tuple[float, jax.Array], tuple[jax.Array, jax.Array]]:
    """Perform the forward pass of the forward-backward algorithm (filtering).

    Computes the filtered state probabilities P(z_t | x_{1:t}) and the
    one-step-ahead predicted probabilities P(z_t | x_{1:t-1}) iteratively.

    Parameters
    ----------
    initial_distribution : jax.Array, shape (n_states,)
        Probability distribution of the initial state, P(z_1): nonnegative,
        summing to at most 1. A sum below 1 is accepted so that a previous
        call's ``predicted_probs_next`` can start the next call even when
        some transition rows lose mass.
    transition_matrix : jax.Array, shape (n_states, n_states)
        State transition probability matrix, P(z_t | z_{t-1}): nonnegative,
        each row summing to at most 1.
    log_likelihoods : jax.Array, shape (n_time, n_states)
        Log likelihood of the observations for each state at each time step,
        log P(x_t | z_t).

    Returns
    -------
    carry : tuple[float, jax.Array]
        A tuple containing:
            - marginal_likelihood : float
                Log probability of the observations log P(x_{1:T}).
            - predicted_probs_next : jax.Array, shape (n_states,)
                The final one-step-ahead prediction P(z_{T+1} | x_{1:T}).
    outputs : tuple[jax.Array, jax.Array]
         A tuple containing:
            - causal_posterior : jax.Array, shape (n_time, n_states)
                Filtered state probabilities P(z_t | x_{1:t}) for t=1...T.
            - predicted_probs : jax.Array, shape (n_time, n_states)
                One-step-ahead predicted state probabilities P(z_t | x_{1:t-1})
                for t=1...T. Note that `predicted_probs[0]` is equivalent
                to `initial_distribution`.

    Raises
    ------
    ValidationError
        If the inputs are concrete and their shapes disagree or they break
        the probability contract above.

    Notes
    -----
    The inputs are checked only when all of them are concrete arrays. Under
    ``jax.jit``, ``vmap`` or ``grad`` they are tracers, and the checks are
    skipped so the pass stays composable with no host transfer; check such
    inputs before the transformation.

    If any time step has all-``-inf`` log-likelihoods, ``_condition_on``
    falls back to the predicted distribution at that step. A ``NaN`` at a
    timestep is left to propagate (it signals a likelihood-computation bug, not
    impossible data): because the filter carries state forward, a single NaN
    contaminates the posterior at that step and every step after it. A
    host-side ``logger.warning`` summarizing the count of degenerate and/or NaN
    steps is emitted when any are found.

    ``_condition_on`` also falls back (and marks the step ``-inf`` in the
    marginal) when the Bayes normalizer is zero for a *finite*-likelihood step —
    a prediction/likelihood support mismatch or an underflow. Such a step is not
    all-``-inf``, so it is surfaced separately: when the returned marginal
    log-likelihood is ``-inf`` but no step was all-``-inf``, an additional
    ``logger.warning`` is emitted.
    """
    if not _is_traced(initial_distribution, transition_matrix, log_likelihoods):
        _check_hmm_arrays(
            n_time=np.shape(log_likelihoods)[0],
            initial_distribution=initial_distribution,
            transition_matrix=transition_matrix,
            log_likelihoods=log_likelihoods,
            initial_mass_loss=True,
        )
    (marginal, predicted_probs_next), (filtered, predicted, _) = _filter_jit(
        initial_distribution, transition_matrix, log_likelihoods
    )
    # Forwarding the marginal surfaces support-mismatch / underflow steps (a
    # -inf marginal with no all-(-inf) step) that the per-step degeneracy
    # check alone cannot see.
    _warn_if_degenerate_timesteps(log_likelihoods, marginal_log_likelihood=marginal)
    return (marginal, predicted_probs_next), (filtered, predicted)


def _smoother_impl(
    transition_matrix: ArrayLike,
    filtered_probs: ArrayLike,
    initial: ArrayLike | None = None,
    ind: ArrayLike | None = None,
    n_time: int | None = None,
) -> jax.Array:
    """Smoother step.

    Parameters
    ----------
    transition_matrix : jax.Array, shape (n_states, n_states)
        Transition matrix
    filtered_probs : jax.Array, shape (n_time, n_states)
        Filtered state probabilities.
    initial : jax.Array, optional
        Initial state distribution, by default None
    ind : jax.Array, optional
        Time indices, by default None
    n_time : int, optional
        Number of time points, by default None

    Returns
    -------
    smoothed_probs : jax.Array, shape (n_time, n_states)
        Smoothed state probabilities
    """
    if n_time is None:
        n_time = filtered_probs.shape[0]
    if ind is None:
        ind = jnp.arange(n_time)
    if initial is None:
        initial = filtered_probs[-1]
    return cast(
        jax.Array,
        _smoother_scan(
            filtered_probs,
            None,
            _StationaryTransition(transition_matrix),
            initial,
            ind,
            n_time,
        )[1],
    )


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_smoother_jit = jax.jit(_smoother_impl)


def smoother(
    transition_matrix: ArrayLike,
    filtered_probs: ArrayLike,
    initial: ArrayLike | None = None,
    ind: ArrayLike | None = None,
    n_time: int | None = None,
) -> jax.Array:
    """Perform the backward pass of the forward-backward algorithm (smoothing).

    Parameters
    ----------
    transition_matrix : jax.Array, shape (n_states, n_states)
        State transition probability matrix: nonnegative, each row summing
        to at most 1.
    filtered_probs : jax.Array, shape (n_time, n_states)
        Filtered state probabilities.
    initial : jax.Array, shape (n_states,), optional
        Smoothed row after the last row; by default the last filtered row.
    ind : jax.Array, shape (n_time,), optional
        Row indices of the recording; by default ``arange(n_time)``.
    n_time : int, optional
        Rows in the whole recording; by default the length of
        ``filtered_probs``.

    Returns
    -------
    smoothed_probs : jax.Array, shape (n_time, n_states)
        Smoothed state probabilities.

    Raises
    ------
    ValidationError
        If ``transition_matrix`` and ``filtered_probs`` are concrete and the
        matrix does not have shape (n_states, n_states) or breaks the
        probability contract above.

    Notes
    -----
    The inputs are checked only when they are concrete arrays. Under
    ``jax.jit``, ``vmap`` or ``grad`` they are tracers, and the checks are
    skipped so the pass stays composable with no host transfer; check such
    inputs before the transformation.
    """
    if not _is_traced(transition_matrix, filtered_probs):
        _check_hmm_arrays(
            n_time=np.shape(filtered_probs)[0],
            n_bins=np.shape(filtered_probs)[-1],
            transition_matrix=transition_matrix,
        )
    return cast(
        jax.Array,
        _smoother_jit(transition_matrix, filtered_probs, initial, ind, n_time),
    )


# Internal version with buffer donation for use in chunked drivers
# Donates filtered_probs, which is single-use in chunked loops and has the
# output's shape. ``initial`` is not donated: XLA can only reuse a donated buffer
# for an output of the same shape and dtype, and no output is (n_states,), so
# donating it only made JAX warn "Some donated buffers were not usable".
# Donation is specified by name (call sites pass by keyword).
_smoother_internal = jax.jit(_smoother_impl, donate_argnames=("filtered_probs",))


ROW_SLICE_ATTRIBUTE = "accepts_row_slice"


def row_slice_aware(
    log_likelihood_func: Callable[..., ArrayLike],
) -> Callable[..., ArrayLike]:
    """Mark a likelihood callback as accepting the ``row_slice`` keyword.

    A marked callback bins the spikes against the FULL decoding ``time`` it is
    handed and returns only the rows named by ``row_slice``, so chunked
    prediction owns every spike exactly once. The chunked drivers require the
    mark whenever they evaluate the likelihood chunk by chunk; an unmarked
    callback is only ever called once, with the full ``time`` and no
    ``row_slice``.

    Parameters
    ----------
    log_likelihood_func : callable
        Function (or unbound method) accepting ``is_missing=None`` and
        ``row_slice=None``. When a row range is requested, ``time`` has shape
        ``(n_time,)`` but ``is_missing`` is already sliced to ``(n_rows,)``;
        the callback must not slice the missing-data mask again. It returns
        ``(n_rows, n_state_bins)``. With ``row_slice=None``, both inputs and
        the result cover the full timeline.

    Returns
    -------
    log_likelihood_func : callable
        The same object, marked.
    """
    setattr(log_likelihood_func, ROW_SLICE_ATTRIBUTE, True)
    return log_likelihood_func


def accepts_row_slice(log_likelihood_func: Callable[..., ArrayLike]) -> bool:
    """Whether a likelihood callback was marked by :func:`row_slice_aware`.

    Bound methods forward the lookup to their underlying function, so marking
    the method in the class body is enough. ``functools.partial`` objects and
    decorators that only record their wrappee in ``__wrapped__`` do not forward
    attribute lookups, so both are unwrapped here: a marked callback hidden
    behind one would otherwise be rejected for chunked evaluation.

    Parameters
    ----------
    log_likelihood_func : callable
        The callback a chunked driver was given.

    Returns
    -------
    accepts_row_slice : bool
        True if the callback (or anything it wraps) was marked by
        :func:`row_slice_aware`, in which case it is called with the full
        ``time`` and a ``row_slice``.
    """
    func: object = log_likelihood_func
    seen: set[int] = set()
    while id(func) not in seen:
        if getattr(func, ROW_SLICE_ATTRIBUTE, False):
            return True
        seen.add(id(func))
        if isinstance(func, functools.partial):
            func = func.func
        elif hasattr(func, "__wrapped__"):
            func = func.__wrapped__
        else:
            break
    return False


def _require_row_slice_aware(
    log_likelihood_func: Callable[..., ArrayLike], n_chunks: int
) -> None:
    """Reject chunk-by-chunk evaluation of an unmarked likelihood callback.

    Handing a chunk only ``time[chunk]`` clips a callback that bins spikes
    against the timeline it is given to that chunk, dropping every spike
    between two chunks while still returning plausible numbers. Only a
    ``row_slice_aware`` callback can be evaluated per chunk.

    Raises
    ------
    ValidationError
        If ``n_chunks > 1`` and the callback is not ``row_slice_aware``.
    """
    if n_chunks > 1 and not accepts_row_slice(log_likelihood_func):
        msg = (
            f"n_chunks={n_chunks} without cached log likelihoods evaluates the "
            "likelihood callback chunk by chunk, which needs a callback marked "
            "with non_local_detector.core.row_slice_aware"
        )
        raise ValidationError(
            msg,
            expected="a row_slice_aware callback that bins against the full "
            "time and returns the rows of row_slice",
            got=f"unmarked {type(log_likelihood_func).__name__}",
            hint="Mark the callback, pass log_likelihoods or "
            "cache_log_likelihoods=True, or use n_chunks=1.",
        )


_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

_CALLBACK_SHAPE_HINT = (
    "log_likelihood_func must return one row per requested time bin and one "
    "column per state bin (len(initial_distribution))."
)


def _check_shape(
    name: str, value: Any, expected: tuple[int, ...], hint: str | None = None
) -> None:
    """Raise ``ValidationError`` unless ``value`` (an array) has shape ``expected``."""
    if np.shape(value) != expected:
        msg = f"{name} has the wrong shape"
        raise ValidationError(
            msg,
            expected=f"shape {expected}",
            got=f"shape {np.shape(value)}",
            hint=hint,
        )


def _validate_dtype(dtype: DTypeLike) -> np.dtype[Any]:
    """Return the working dtype: float32, or float64 with JAX's 64-bit mode on."""
    dtype = np.dtype(dtype)
    if dtype not in _SUPPORTED_DTYPES:
        msg = "dtype must be float32 or float64"
        raise ValidationError(
            msg,
            expected="float32 or float64",
            got=str(dtype),
            hint="Pass dtype=np.float32 (the default) or np.float64; other "
            "dtypes are rejected, not converted.",
        )
    if dtype == np.float64 and not jax.config.x64_enabled:  # type: ignore[attr-defined]  # jax.config attributes are defined dynamically
        msg = "float64 requires JAX's 64-bit mode; precision is not silently downgraded"
        raise ValidationError(
            msg,
            expected="JAX_ENABLE_X64=1 for dtype=float64",
            got="64-bit mode disabled",
            hint="Set the environment variable JAX_ENABLE_X64=1 before "
            "importing JAX, or call jax.config.update('jax_enable_x64', True) "
            "at startup; otherwise pass dtype=np.float32.",
        )
    return dtype


def _validate_time(time: Any) -> int:
    """Return the number of time bins of a dense driver's nonempty ``time`` vector."""
    if np.ndim(time) != 1 or np.shape(time)[0] == 0:
        msg = "time must be a nonempty vector"
        raise ValidationError(
            msg,
            expected="shape (n_time,) with n_time >= 1",
            got=f"shape {np.shape(time)}",
            hint="Pass the decoding time bin centers as a 1-D array.",
        )
    return int(np.shape(time)[0])


def _bad_entries(
    values: np.ndarray, sums: np.ndarray, describe: Callable[[tuple[int, ...]], str]
) -> str | None:
    """Describe a non-finite or negative entry of ``values``, or return None.

    ``sums`` are the row sums already computed, which carry any infinity.
    """
    minimum = values.min()
    if not (np.isfinite(minimum) and np.all(np.isfinite(sums))):
        return "non-finite entries (nan or inf)"
    if minimum < 0:
        where = np.unravel_index(int(np.argmin(values)), values.shape)
        return f"minimum entry {minimum:.6g} at {describe(tuple(map(int, where)))}"
    return None


def _check_distribution(
    name: str, values: np.ndarray, *, allow_mass_loss: bool = False
) -> None:
    """Raise unless ``values`` is nonnegative, finite and sums to 1.

    With ``allow_mass_loss``, a sum below 1 is accepted too: the final
    prediction of a filter whose transition loses mass (a row below 1)
    sums below 1 and starts the next pass over the recording.
    """
    total = values.sum(dtype=np.float64)
    got = _bad_entries(values, total, lambda where: f"state bin {where[0]}")
    lowest = -np.inf if allow_mass_loss else 1 - PROBABILITY_SUM_TOLERANCE
    if got is None and not lowest <= total <= 1 + PROBABILITY_SUM_TOLERANCE:
        got = f"sums to {total:.6g}"
    if got is not None:
        msg = f"{name} must be a probability distribution over state bins"
        raise ValidationError(
            msg,
            expected="nonnegative, finite entries summing to "
            + ("at most " if allow_mass_loss else "")
            + f"1 (within {PROBABILITY_SUM_TOLERANCE:g})"
            + (
                "; a sum below 1 is mass a previous pass's transitions lost"
                if allow_mass_loss
                else ""
            ),
            got=got,
            hint=f"Pass nonnegative probabilities normalized to sum to 1, e.g. "
            f"{name} / {name}.sum().",
        )


def _check_transition_rows(name: str, matrix: np.ndarray) -> None:
    """Raise unless no row of the joint transition ``matrix`` creates mass.

    A row may sum to less than 1: the detectors build all-zero continuous
    rows for bins nothing can leave, such as an ``EmpiricalMovement`` bin
    never visited, so the joint row keeps only its other discrete states'
    mass. Such a row lowers the evidence; a row above 1 would inflate it.
    """
    row_sums = matrix.sum(axis=1, dtype=np.float64)
    got = _bad_entries(
        matrix, row_sums, lambda where: f"row {where[0]}, column {where[1]}"
    )
    if got is None:
        above = ~(row_sums <= 1 + PROBABILITY_SUM_TOLERANCE)
        if above.any():
            first = int(np.argmax(above))
            got = (
                f"{int(above.sum())} of {above.size} rows sum above 1 "
                f"(row {first} sums to {row_sums[first]:.6g})"
            )
    if got is not None:
        msg = f"{name} rows must be probability distributions"
        raise ValidationError(
            msg,
            expected="nonnegative, finite entries with each row summing to at "
            f"most 1 (within {PROBABILITY_SUM_TOLERANCE:g}); a row below 1 "
            "loses probability, as for a bin with no outgoing transition",
            got=got,
            hint=f"Normalize each row: {name} / {name}.sum(axis=1, "
            "keepdims=True), leaving all-zero rows for bins nothing can leave.",
        )


def _check_discrete_rows(discrete_transition_matrix: np.ndarray) -> None:
    """Raise unless every row of every time bin is a probability distribution."""
    name = "discrete_transition_matrix"
    row_sums = discrete_transition_matrix.sum(axis=-1, dtype=np.float64)
    got = _bad_entries(
        discrete_transition_matrix,
        row_sums,
        lambda where: f"time {where[0]}, from state {where[1]} to state {where[2]}",
    )
    if got is None:
        off = ~(np.abs(row_sums - 1) <= PROBABILITY_SUM_TOLERANCE)
        if off.any():
            time_ind, state = np.unravel_index(int(np.argmax(off)), off.shape)
            got = (
                f"{int(off.sum())} of {off.size} rows sum to other than 1 (time "
                f"{time_ind}, from state {state}, sums to "
                f"{row_sums[time_ind, state]:.6g})"
            )
    if got is not None:
        msg = f"{name} rows must be probability distributions"
        raise ValidationError(
            msg,
            expected="nonnegative, finite entries with each row of each time "
            f"bin summing to 1 (within {PROBABILITY_SUM_TOLERANCE:g})",
            got=got,
            hint=f"Normalize each row: {name} / {name}.sum(axis=-1, keepdims=True).",
        )


def _check_continuous_block_rows(
    continuous_transition_matrix: np.ndarray, state_ind: np.ndarray, n_states: int
) -> None:
    """Raise unless each bin's row sums to at most 1 within each target state.

    The covariate-dependent joint transition is ``C[i, j] * D_t[s_i, s_j]``,
    whose row ``i`` sums to ``sum_k D_t[s_i, k] * R[i, k]`` with ``R[i, k]``
    the sum of ``C[i]`` over the bins of state ``k``. With every ``D_t`` row a
    distribution, ``R <= 1`` is exactly what keeps every joint row at most 1
    for every ``D_t``, without building a joint matrix.
    """
    name = "continuous_transition_matrix"
    states = np.zeros(
        (len(state_ind), n_states),
        dtype=np.result_type(continuous_transition_matrix.dtype, np.float32),
    )
    states[np.arange(len(state_ind)), state_ind] = 1
    block_sums = continuous_transition_matrix @ states
    got = _bad_entries(
        continuous_transition_matrix,
        block_sums,
        lambda where: f"row {where[0]}, column {where[1]}",
    )
    if got is None:
        above = ~(block_sums <= 1 + PROBABILITY_SUM_TOLERANCE)
        if above.any():
            row, state = np.unravel_index(int(np.argmax(above)), above.shape)
            got = (
                f"{int(above.sum())} of {above.size} (bin, target state) rows "
                f"sum above 1 (bin {row} to discrete state {state} sums to "
                f"{block_sums[row, state]:.6g})"
            )
    if got is not None:
        msg = f"{name} rows must be probability distributions"
        raise ValidationError(
            msg,
            expected="nonnegative, finite entries with each bin's row summing "
            f"to at most 1 (within {PROBABILITY_SUM_TOLERANCE:g}) over the bins "
            "of each discrete state",
            got=got,
            hint="Each block of continuous_transition_matrix between the bins "
            "of two discrete states is a conditional transition: normalize its "
            "rows to 1, or leave them zero for bins nothing can leave.",
        )


@functools.partial(jax.jit, static_argnames=("n_bins", "dtype", "n_states"))
def _operator_row_sums(
    operator: TransitionProducts,
    n_bins: int,
    dtype: np.dtype[Any],
    n_states: int | None,
) -> jax.Array:
    """Row sums ``T @ 1`` of an operator, or its per-target-state block sums.

    Jitted: dispatching the products' operations one by one compiled each
    of them, about 1 s for a 7,202-bin structured operator, against 0.1 s
    for one compilation.

    Parameters
    ----------
    operator : TransitionProducts
        Transition operator.
    n_bins : int
        Number of state bins.
    dtype : np.dtype
        Dtype of the all-ones vector and the indicator weights.
    n_states : int or None
        Number of discrete states of per-row weights; None for a stationary
        operator.

    Returns
    -------
    sums : jax.Array, shape (n_bins,) or (n_bins, n_states)
        ``operator.backward(1, None)`` for a stationary operator. With
        weights, column ``k`` is the product with the weight whose column
        ``k`` is one: ``R[i, k]``, the sum of ``T[i, j]`` over the bins ``j``
        of state ``k``, for an operator that scales ``T[i, j]`` by
        ``weight[s_i, s_j]``.
    """
    ones = jnp.ones(n_bins, dtype=dtype)
    if n_states is None:
        return operator.backward(ones, None)
    indicator = jnp.zeros((n_states, n_states), dtype=dtype)
    return jnp.stack(
        [operator.backward(ones, indicator.at[:, k].set(1)) for k in range(n_states)],
        axis=-1,
    )


def _check_operator_rows(
    operator: TransitionProducts,
    n_bins: int,
    dtype: DTypeLike,
    n_states: int | None = None,
) -> None:
    """Raise unless no row of ``operator``'s transition creates mass.

    An operator exposes only its products, so its rows are checked through
    their sums ``T @ 1 = operator.backward(1, weight)``, with the dense
    checks' semantics: as :func:`_check_transition_rows` for a stationary
    operator, and as :func:`_check_continuous_block_rows`, per target state,
    for one given per-row discrete weights (``n_states``), which assumes the
    operator scales ``T[i, j]`` by ``weight[s_i, s_j]`` as every operator in
    the package does. A negative entry that its row's other entries
    outweigh is not visible in the sums. The check costs one product
    (``n_states`` with weights) and its compilation, measured at 0.2 ms and
    0.1 s for a 7,202-bin structured operator, against seconds for the
    checkpointed passes over 10,000 rows.

    Parameters
    ----------
    operator : TransitionProducts
        Transition operator, applied to all-ones vectors.
    n_bins : int
        Number of state bins.
    dtype : dtype
        Working dtype of the products.
    n_states : int, optional
        Number of discrete states of the per-row weights; None for a
        stationary operator.

    Raises
    ------
    ValidationError
        If applying the operator to an all-ones vector raises ``TypeError``
        or ``ValueError``, or the row sums do not have one entry per state
        bin, are not finite, are negative, or exceed 1 (per target state
        with weights) beyond ``PROBABILITY_SUM_TOLERANCE``.
    """
    name = "transition_operator"
    weight = (
        "None"
        if n_states is None
        else f"each ({n_states}, {n_states}) column-indicator weight"
    )
    try:
        sums = np.asarray(
            _operator_row_sums(operator, n_bins, np.dtype(dtype), n_states),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        msg = f"{name} could not be applied to an all-ones vector"
        raise ValidationError(
            msg,
            expected=f"backward(ones({n_bins}), {weight}) to return row sums",
            got=f"{type(error).__name__}: {error}",
            hint="The operator must act on the state bins of "
            "initial_distribution, and take per-row weights exactly when "
            "discrete_transition_matrix is given.",
        ) from error
    if sums.shape[:1] != (n_bins,):
        msg = f"{name} gives row sums of the wrong shape"
        raise ValidationError(
            msg,
            expected=f"shape ({n_bins},), one sum per state bin",
            got=f"shape {sums.shape[:1]}",
            hint="The operator's state bins must be those of "
            "initial_distribution and state_ind.",
        )
    got = None
    if not np.all(np.isfinite(sums)):
        got = "non-finite row sums (nan or inf)"
    elif sums.min() < 0:
        where = np.unravel_index(int(np.argmin(sums)), sums.shape)
        got = f"negative row sum {sums.min():.6g} at row {where[0]}"
    else:
        above = ~(sums <= 1 + PROBABILITY_SUM_TOLERANCE)
        if above.any():
            where = np.unravel_index(int(np.argmax(above)), above.shape)
            got = (
                f"{int(above.sum())} of {above.size} rows sum above 1 (row "
                f"{where[0]} sums to {sums[where]:.6g})"
                if n_states is None
                else f"{int(above.sum())} of {above.size} (bin, target state) "
                f"rows sum above 1 (bin {where[0]} to discrete state "
                f"{where[1]} sums to {sums[where]:.6g})"
            )
    if got is not None:
        msg = f"{name} rows must be probability distributions"
        raise ValidationError(
            msg,
            expected="rows summing to at most 1 (within "
            f"{PROBABILITY_SUM_TOLERANCE:g})"
            + ("" if n_states is None else " over the bins of each discrete state")
            + "; a row below 1 loses probability, as for a bin with no "
            "outgoing transition",
            got=got,
            hint="Normalize the operator's conditional transitions so each "
            "row sums to 1, leaving zero rows for bins nothing can leave. A "
            "detector's operator holds its continuous transitions' blocks, "
            "such as a custom transition's structured_block.",
        )


def _validate_hmm_inputs(
    *,
    n_time: int,
    initial_distribution: Any,
    dtype: DTypeLike,
    state_ind: Any = None,
    n_states: Any = None,
    transition_matrix: Any = None,
    discrete_transition_matrix: Any = None,
    continuous_transition_matrix: Any = None,
    is_missing: Any = None,
    log_likelihoods: Any = None,
) -> tuple[np.dtype[Any], int | None]:
    """Check the inputs of a forward-backward or Viterbi driver at its entry.

    The one input contract of the dense and checkpointed drivers. It runs on
    the host before any likelihood is evaluated, and reads each transition
    once (a minimum and a float64 row sum), cheap next to the scans that
    apply it every time bin.

    Parameters
    ----------
    n_time : int
        Number of decoding time bins.
    initial_distribution : array_like, shape (n_state_bins,)
        Nonnegative, summing to 1.
    dtype : dtype
        Working dtype; see :func:`_validate_dtype`.
    state_ind : array_like of int, shape (n_state_bins,), optional
        Nonnegative discrete state of each state bin.
    n_states : int, optional
        Number of discrete states; by default the last axis of
        ``discrete_transition_matrix``, or one more than the largest
        ``state_ind``.
    transition_matrix : array_like, shape (n_state_bins, n_state_bins), optional
        Stationary joint transition; rows sum to at most 1.
    discrete_transition_matrix : array_like, shape (n_time, n_states, n_states), optional
        Per-time discrete transitions; rows sum to 1. Needs ``state_ind``.
    continuous_transition_matrix : array_like, shape (n_state_bins, n_state_bins), optional
        Covariate-path continuous transition; each bin's row sums to at most
        1 over the bins of each discrete state.
    is_missing : array_like, shape (n_time,), optional
        True for time bins without observations; checked for shape only.
    log_likelihoods : array_like, shape (n_time, n_state_bins), optional
        Precomputed log likelihoods; checked for shape only.

    Returns
    -------
    dtype : np.dtype
        The validated working dtype.
    n_states : int or None
        The number of discrete states, or None without ``state_ind``.

    Raises
    ------
    ValidationError
        If any input breaks the contract above.
    """
    dtype = _validate_dtype(dtype)
    return dtype, _check_hmm_arrays(
        n_time=n_time,
        initial_distribution=initial_distribution,
        state_ind=state_ind,
        n_states=n_states,
        transition_matrix=transition_matrix,
        discrete_transition_matrix=discrete_transition_matrix,
        continuous_transition_matrix=continuous_transition_matrix,
        is_missing=is_missing,
        log_likelihoods=log_likelihoods,
    )


def _check_hmm_arrays(
    *,
    n_time: int,
    initial_distribution: Any = None,
    n_bins: int | None = None,
    state_ind: Any = None,
    n_states: Any = None,
    transition_matrix: Any = None,
    discrete_transition_matrix: Any = None,
    continuous_transition_matrix: Any = None,
    is_missing: Any = None,
    log_likelihoods: Any = None,
    initial_mass_loss: bool = False,
) -> int | None:
    """Check the shapes and probabilities of :func:`_validate_hmm_inputs`.

    Everything that function checks except the dtype, for the passes that
    compute in their inputs' dtype. The smoothers have no
    ``initial_distribution`` and pass ``n_bins`` instead. The public passes
    set ``initial_mass_loss``, accepting an initial distribution that sums
    below 1, such as a previous call's final prediction.

    Parameters
    ----------
    n_time : int
        Number of time bins.
    initial_distribution : array_like, shape (n_state_bins,), optional
        Nonnegative, summing to 1; sets ``n_state_bins``.
    n_bins : int, optional
        Number of state bins, without ``initial_distribution``.
    state_ind, n_states, transition_matrix, discrete_transition_matrix, continuous_transition_matrix, is_missing, log_likelihoods : optional
        As in :func:`_validate_hmm_inputs`.
    initial_mass_loss : bool, optional
        Accept an ``initial_distribution`` summing below 1. By default False.

    Returns
    -------
    n_states : int or None
        The number of discrete states, or None without ``state_ind``.

    Raises
    ------
    ValidationError
        If any input breaks the contract of :func:`_validate_hmm_inputs`.
    """
    if initial_distribution is None:
        bins_source = "filtered_probs.shape[1]"
    else:
        initial = np.asarray(initial_distribution)
        if initial.ndim != 1 or initial.size == 0:
            msg = "initial_distribution must be a nonempty vector over state bins"
            raise ValidationError(
                msg,
                expected="shape (n_state_bins,) with n_state_bins >= 1",
                got=f"shape {initial.shape}",
                hint="Pass one initial probability per state bin.",
            )
        n_bins, bins_source = initial.size, "len(initial_distribution)"
        _check_distribution(
            "initial_distribution", initial, allow_mass_loss=initial_mass_loss
        )
    assert n_bins is not None  # given directly or by initial_distribution
    shape_hint = (
        f"n_time = {n_time} decoding time bins and n_state_bins = {n_bins} "
        f"({bins_source}) set every input's shape."
    )
    if is_missing is not None:
        _check_shape("is_missing", is_missing, (n_time,), shape_hint)
    if log_likelihoods is not None:
        _check_shape("log_likelihoods", log_likelihoods, (n_time, n_bins), shape_hint)
    if transition_matrix is not None:
        _check_shape(
            "transition_matrix", transition_matrix, (n_bins, n_bins), shape_hint
        )
        _check_transition_rows("transition_matrix", np.asarray(transition_matrix))
    if continuous_transition_matrix is not None:
        _check_shape(
            "continuous_transition_matrix",
            continuous_transition_matrix,
            (n_bins, n_bins),
            shape_hint,
        )
    if state_ind is None:
        return None

    state_ind = np.asarray(state_ind)
    _check_shape("state_ind", state_ind, (n_bins,), shape_hint)
    if state_ind.dtype.kind not in "iu" or state_ind.min() < 0:
        msg = "state_ind must hold nonnegative integers"
        raise ValidationError(
            msg,
            expected="the integer discrete state of each state bin",
            got=f"dtype {state_ind.dtype}"
            if state_ind.dtype.kind not in "iu"
            else f"minimum {state_ind.min()}",
            hint="Pass state_ind as a nonnegative integer array, e.g. "
            "np.asarray(state_ind, dtype=int).",
        )
    source = "n_states"
    if n_states is None and discrete_transition_matrix is not None:
        n_states, source = (
            np.shape(discrete_transition_matrix)[-1],
            "discrete_transition_matrix",
        )
    minimum_states = int(state_ind.max()) + 1
    if n_states is None:
        n_states = minimum_states
    if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)):
        msg = "n_states must be an integer"
        raise ValidationError(
            msg,
            expected=f"an integer of at least {minimum_states}",
            got=repr(n_states),
            hint="Pass the number of discrete states, or None to infer it.",
        )
    n_states = int(n_states)
    if n_states < minimum_states:
        msg = "state_ind refers to a discrete state the transitions do not have"
        raise ValidationError(
            msg,
            expected=f"state_ind values below {n_states}",
            got=f"discrete state {minimum_states - 1} with {source} of "
            f"{n_states} states",
            hint="Every state bin's discrete state needs a row and column of "
            "the discrete transitions.",
        )
    if discrete_transition_matrix is not None:
        _check_shape(
            "discrete_transition_matrix",
            discrete_transition_matrix,
            (n_time, n_states, n_states),
            shape_hint,
        )
        _check_discrete_rows(np.asarray(discrete_transition_matrix))
        if continuous_transition_matrix is not None:
            _check_continuous_block_rows(
                np.asarray(continuous_transition_matrix), state_ind, n_states
            )
    return cast(int, n_states)


def _store_chunk_rows(
    buffer: np.ndarray | None,
    chunk: jax.Array,
    time_inds: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """Copy one chunk's rows into the host log-likelihood buffer, allocating it first.

    Assignment copies into the final host buffer before the chunk is donated
    to the filter. Retaining a NumPy view would alias the donated buffer on
    CPU; retaining separate chunk copies would double host storage.
    """
    if buffer is None:
        buffer = np.empty((n_time, chunk.shape[1]), dtype=chunk.dtype)
    buffer[int(time_inds[0]) : int(time_inds[-1]) + 1] = np.asarray(chunk)
    return buffer


def _call_log_likelihood_chunk(
    log_likelihood_func: Callable[..., ArrayLike],
    time: np.ndarray,
    log_likelihood_args: tuple[Any, ...],
    time_inds: np.ndarray,
    is_missing_chunk: np.ndarray | None,
) -> ArrayLike:
    """Evaluate one chunk's log likelihood with global spike binning.

    A ``row_slice_aware`` callback receives the FULL ``time`` plus the chunk's
    global row range, so a spike lands in the row that owns it on the full
    timeline. An unmarked callback reaches here only as the single chunk (see
    :func:`_require_row_slice_aware`) and is called on the full timeline.
    """
    if not accepts_row_slice(log_likelihood_func):
        return log_likelihood_func(
            time, *log_likelihood_args, is_missing=is_missing_chunk
        )
    return log_likelihood_func(
        time,
        *log_likelihood_args,
        is_missing=is_missing_chunk,
        row_slice=slice(int(time_inds[0]), int(time_inds[-1]) + 1),
    )


def _state_marginals(posterior: jax.Array, aggregation: jax.Array) -> jax.Array:
    """Sum state-bin probabilities into discrete states at full float32 precision.

    Parameters
    ----------
    posterior : jax.Array, shape (n_time, n_state_bins)
        Probability of each state bin at each time bin.
    aggregation : jax.Array, shape (n_state_bins, n_states)
        One-hot state membership.

    Returns
    -------
    state_probabilities : jax.Array, shape (n_time, n_states)
    """
    # The default CUDA precision (TF32) visibly changes these sums.
    return jnp.matmul(posterior, aggregation, precision=jax.lax.Precision.HIGHEST)


class FilterSmootherResult(NamedTuple):
    """Outputs of the chunked forward-backward drivers.

    Returned by :func:`chunked_filter_smoother` and
    :func:`chunked_filter_smoother_covariate_dependent`. Its first eight
    fields unpack like the positional tuple those drivers returned before.

    Attributes
    ----------
    acausal_posterior : np.ndarray or jax.Array, shape (n_time, n_state_bins)
        Smoothed state probabilities. A JAX array with ``host_outputs=False``.
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Smoothed state probabilities for each state index.
    marginal_log_likelihood : float
        Log marginal likelihood of the observations, summed in float64.
    causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        Filtered state probabilities for each state index.
    predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        One-step-ahead predicted state probabilities for each state index.
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins), or None
        Log likelihoods for each state bin at each time point; None when they
        were neither given, cached nor accumulated.
    causal_posterior : np.ndarray or jax.Array, shape (n_time, n_state_bins)
        Filtered state probabilities. A JAX array with ``host_outputs=False``.
    predictive_posterior : np.ndarray or jax.Array, shape (n_time, n_state_bins)
        One-step-ahead predicted state probabilities over state bins. A JAX
        array with ``host_outputs=False``.
    degenerate_indices : np.ndarray of int, shape (n_degenerate,)
        Increasing time indices whose log likelihoods are ``-inf`` in every
        state bin. Found during the forward pass, so they are available even
        when the log likelihoods are not cached (``log_likelihoods`` is None).
    """

    acausal_posterior: np.ndarray | jax.Array
    acausal_state_probabilities: np.ndarray
    marginal_log_likelihood: float
    causal_state_probabilities: np.ndarray
    predictive_state_probabilities: np.ndarray
    log_likelihoods: np.ndarray | None
    causal_posterior: np.ndarray | jax.Array
    predictive_posterior: np.ndarray | jax.Array
    degenerate_indices: np.ndarray


def _chunked_filter_smoother(
    filter_chunk: Callable[..., tuple[Any, ...]],
    smoother_chunk: Callable[..., jax.Array],
    *,
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple[Any, ...],
    is_missing: np.ndarray | None,
    n_chunks: int,
    log_likelihoods: np.ndarray | None,
    cache_log_likelihoods: bool,
    dtype: DTypeLike,
    accumulate_log_likelihoods: bool,
    host_outputs: bool,
) -> FilterSmootherResult:
    """Chunked forward-backward shared by the stationary and covariate drivers.

    The transition model enters only through the two per-chunk steps, which
    close over the device transition arrays. Both run donating JIT functions:
    they must not be handed an array the driver reads again.

    Parameters
    ----------
    filter_chunk : callable
        ``filter_chunk(initial_distribution, log_likelihoods, time_inds)``
        filters one chunk from the predicted distribution of its first row,
        returning ``((_, predicted_probs_next), (causal_posterior,
        predicted_probs, increments))`` as :func:`_filter_impl` does.
        ``time_inds`` is the chunk's global row indices (device array).
    smoother_chunk : callable
        ``smoother_chunk(filtered_probs, initial, time_inds, n_time)`` smooths
        one chunk backward from ``initial``, the smoothed row after the chunk
        (the last filtered row for the final chunk), as :func:`_smoother_impl`
        does.

    time : np.ndarray, shape (n_time,)
        Time of each decoding bin; its length sets the number of time bins.
    state_ind : np.ndarray, shape (n_state_bins,)
        Connects the discrete states to the state space.
    initial_distribution : np.ndarray, shape (n_state_bins,)
        Initial probability of each state bin.
    log_likelihood_func : callable
        Called as ``log_likelihood_func(time, *log_likelihood_args,
        is_missing=...)``; a callable marked by :func:`row_slice_aware` also
        receives ``row_slice=slice(start, stop)`` per chunk.
    log_likelihood_args : tuple
        Extra positional arguments passed to ``log_likelihood_func``.
    is_missing : np.ndarray, shape (n_time,), or None
        True for time bins without observations.
    n_chunks : int
        Number of chunks to split the time bins into.
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins), or None
        Precomputed log likelihoods used instead of calling
        ``log_likelihood_func``.
    cache_log_likelihoods : bool
        If True, compute and cache the log likelihoods before the loop.
    dtype : dtype
        Working dtype, float32 or float64.
    accumulate_log_likelihoods : bool
        If True and the log likelihoods are not cached, copy each chunk's rows
        into one host array that is returned.
    host_outputs : bool
        If True, return the posteriors as NumPy arrays; otherwise keep them on
        the device as JAX arrays.

    The returns are those of :func:`chunked_filter_smoother`.
    """
    causal_posterior = []
    predictive_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    # Device increments, summed in float64 after the loop so the host does not
    # wait for each chunk's filter.
    chunk_increments = []
    degenerate_indices = []
    n_nan_total = 0

    n_time = len(time)
    # Validate n_chunks: ensure it doesn't exceed n_time to avoid empty chunks
    if n_chunks > n_time:
        msg = f"n_chunks ({n_chunks}) cannot exceed n_time ({n_time})"
        raise ValidationError(
            msg,
            expected=f"1 <= n_chunks <= {n_time}",
            got=f"n_chunks={n_chunks}",
            hint="Each chunk must contain at least one time bin; use fewer chunks.",
        )
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    # Convert inputs to JAX arrays once at the top - keep data on device
    # Cast to desired dtype for precision control
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)

    # Create aggregation matrix for marginalizing state bins -> discrete states
    # This is reused across all chunks
    n_discrete_states = int(jnp.max(state_ind_jax)) + 1
    state_aggregation_matrix = jax.nn.one_hot(
        state_ind_jax, n_discrete_states
    )  # (n_state_bins, n_states)

    # Initialize variables that may be referenced in loop conditionals
    predicted_probs_next: jax.Array | None = None
    acausal_posterior_chunk: jax.Array | None = None

    # Precompute or convert log_likelihoods to JAX once
    if cache_log_likelihoods and log_likelihoods is None:
        log_likelihoods = cast(
            np.ndarray,
            log_likelihood_func(time, *log_likelihood_args, is_missing=is_missing),
        )
        _check_shape(
            "log_likelihood_func output",
            log_likelihoods,
            (n_time, len(initial_distribution)),
            _CALLBACK_SHAPE_HINT,
        )
    log_likelihoods_jax = (
        jnp.asarray(log_likelihoods, dtype=dtype)
        if log_likelihoods is not None
        else None
    )
    accumulated_log_likelihoods: np.ndarray | None = None
    if log_likelihoods_jax is None:
        _require_row_slice_aware(log_likelihood_func, n_chunks)

    # Forward pass: accumulate JAX arrays
    for chunk_id, time_inds_np in enumerate(time_chunks):
        # Convert time_inds to JAX once to avoid repeated host↔device syncs
        time_inds = jnp.asarray(time_inds_np)

        if log_likelihoods_jax is not None:
            log_likelihood_chunk: ArrayLike = log_likelihoods_jax[time_inds]
        else:
            is_missing_chunk = (
                is_missing[time_inds_np] if is_missing is not None else None
            )
            log_likelihood_chunk = _call_log_likelihood_chunk(
                log_likelihood_func,
                time,
                log_likelihood_args,
                time_inds_np,
                is_missing_chunk,
            )
            _check_shape(
                "log_likelihood_func output",
                log_likelihood_chunk,
                (len(time_inds_np), len(initial_distribution)),
                _CALLBACK_SHAPE_HINT,
            )
            log_likelihood_chunk = jnp.asarray(log_likelihood_chunk, dtype=dtype)
            if accumulate_log_likelihoods:
                accumulated_log_likelihoods = _store_chunk_rows(
                    accumulated_log_likelihoods,
                    log_likelihood_chunk,
                    time_inds_np,
                    n_time,
                )

        # Tally degenerate (all -inf) and NaN timesteps at one host sync point
        # before the array is donated to the JIT call. Accumulated across chunks
        # and reported once after the forward pass.
        chunk_degenerate, n_nan = _accumulate_chunk_degeneracy(
            log_likelihood_chunk, time_inds_np
        )
        degenerate_indices.append(chunk_degenerate)
        n_nan_total += n_nan

        # Donated: log_likelihood_chunk (created fresh), initial_distribution
        # Do not read these after the call - they are consumed by the JIT function
        # Note: On first iteration, initial_distribution_jax may trigger a donation warning
        # if the array is small (< ~1 KB). This is benign - JAX skips donation for small
        # arrays where the overhead exceeds the benefit. The optimization works correctly
        # for large production workloads.
        (
            (_, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk, increments),
        ) = filter_chunk(
            initial_distribution=(
                initial_distribution_jax if chunk_id == 0 else predicted_probs_next
            ),
            log_likelihoods=log_likelihood_chunk,
            time_inds=time_inds,
        )

        # Keep as JAX arrays - no conversion to NumPy yet
        # Marginalize state bins -> discrete states by summing probabilities
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(
            _state_marginals(causal_posterior_chunk, state_aggregation_matrix)
        )
        predictive_posterior.append(predicted_probs_chunk)
        predictive_state_probabilities.append(
            _state_marginals(predicted_probs_chunk, state_aggregation_matrix)
        )

        chunk_increments.append(increments)

    marginal_likelihood = 0.0
    for increments in chunk_increments:
        marginal_likelihood += _float64_evidence(increments)
    del chunk_increments
    degenerate_time_indices = np.concatenate(degenerate_indices)
    _warn_degenerate_and_nan_timesteps(
        len(degenerate_time_indices),
        n_nan_total,
        n_time,
        marginal_log_likelihood=float(marginal_likelihood),
    )

    if accumulated_log_likelihoods is not None:
        log_likelihoods = accumulated_log_likelihoods

    # Concatenate JAX arrays on device
    causal_posterior_jax = jnp.concatenate(causal_posterior)
    causal_state_probabilities_jax = jnp.concatenate(causal_state_probabilities)
    predictive_posterior_jax = jnp.concatenate(predictive_posterior)
    predictive_state_probabilities_jax = jnp.concatenate(predictive_state_probabilities)
    # Release the per-chunk copies so they don't coexist with the concatenations
    del causal_posterior, predictive_posterior

    # Backward pass: accumulate JAX arrays
    for chunk_id, time_inds_np in enumerate(reversed(time_chunks)):
        time_inds = jnp.asarray(time_inds_np)

        # Donated: the filtered_probs slice (a copy) and the initial boundary
        # row are single-use per iteration.
        # Note: Small arrays may trigger benign donation warnings in tests
        acausal_posterior_chunk = smoother_chunk(
            filtered_probs=causal_posterior_jax[time_inds],
            initial=(
                causal_posterior_jax[-1]
                if chunk_id == 0
                else cast(jax.Array, acausal_posterior_chunk)[0]
            ),
            time_inds=time_inds,
            n_time=n_time,
        )
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(
            _state_marginals(acausal_posterior_chunk, state_aggregation_matrix)
        )

    # Concatenate JAX arrays on device
    acausal_posterior_jax = jnp.concatenate(acausal_posterior[::-1])
    acausal_state_probabilities_jax = jnp.concatenate(acausal_state_probabilities[::-1])
    del acausal_posterior

    # The state probabilities are small and always go to the host; the
    # (n_time, n_state_bins) posteriors only when the caller wants host arrays.
    spatial = np.asarray if host_outputs else jnp.asarray
    return FilterSmootherResult(
        acausal_posterior=spatial(acausal_posterior_jax),
        acausal_state_probabilities=np.asarray(acausal_state_probabilities_jax),
        marginal_log_likelihood=float(marginal_likelihood),
        causal_state_probabilities=np.asarray(causal_state_probabilities_jax),
        predictive_state_probabilities=np.asarray(predictive_state_probabilities_jax),
        # Keep as original (may be None or NumPy)
        log_likelihoods=log_likelihoods,
        causal_posterior=spatial(causal_posterior_jax),
        predictive_posterior=spatial(predictive_posterior_jax),
        degenerate_indices=degenerate_time_indices,
    )


def chunked_filter_smoother(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple[Any, ...],
    is_missing: np.ndarray | None = None,
    n_chunks: int = 1,
    log_likelihoods: np.ndarray | None = None,
    # positional: ``dtype`` after it is public positional API (a test pins it)
    cache_log_likelihoods: bool = True,  # noqa: FBT001, FBT002
    dtype: DTypeLike = jnp.float32,
    *,
    accumulate_log_likelihoods: bool = False,
    host_outputs: bool = True,
) -> FilterSmootherResult:
    """Filter and smooth the state probabilities in chunks.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time of each decoding bin (bin centers, in seconds), passed to
        ``log_likelihood_func``. Its length sets the number of time bins.
    state_ind : np.ndarray, shape (n_state_bins,)
        Connects the discrete states to the state space.
    initial_distribution : np.ndarray, shape (n_state_bins,)
        Initial probability of each state bin: nonnegative, summing to 1
        (within ``PROBABILITY_SUM_TOLERANCE``).
    transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
        Stationary transition probability between state bins: nonnegative,
        each row summing to at most 1. A row below 1 loses probability, as an
        all-zero row does for a bin nothing can leave.
    log_likelihood_func : callable
        Called as ``log_likelihood_func(time, *log_likelihood_args,
        is_missing=...)`` with the FULL ``time``. Without cached or given log
        likelihoods, a callable marked by :func:`row_slice_aware` is called
        per chunk and also receives ``row_slice=slice(start, stop)`` and the
        chunk's ``is_missing`` rows, so each decoding spike is binned once
        against the whole timeline and lands in the row that owns it. An
        unmarked callable is called once for the whole timeline; with
        ``n_chunks > 1`` and no cached likelihoods it raises ``ValidationError``.
    log_likelihood_args : tuple
        Extra positional arguments passed to ``log_likelihood_func`` after
        ``time``.
    is_missing : np.ndarray, shape (n_time,), optional
        True for time bins without observations, which get neutral
        likelihoods. By default None (none missing).
    n_chunks : int, optional
        Number of chunks to split the data into, by default 1.
    log_likelihoods : np.ndarray | None, optional
        Precomputed log likelihoods, shape (n_time, n_state_bins), used
        instead of calling ``log_likelihood_func``. By default None.
    cache_log_likelihoods : bool, optional
        If True, log likelihoods are cached, by default True.
    dtype : dtype, optional
        Working dtype, float32 (the default) or float64. Use float64 for
        numerically challenging problems; it requires JAX's 64-bit mode
        (``JAX_ENABLE_X64=1``) and raises ``ValidationError`` without it.
    accumulate_log_likelihoods : bool, optional, keyword-only
        If True and the log likelihoods are not cached, each chunk's rows are
        copied into a single preallocated host array so the returned
        ``log_likelihoods`` covers every row in global order instead of being
        ``None``. This allocates the full (n_time, n_state_bins) array, so set
        it only when the caller explicitly asked for the log likelihoods.
        Keyword-only, and last, so that it cannot capture a positional argument
        aimed at a parameter that predates it (``dtype``). By default False.
    host_outputs : bool, optional, keyword-only
        If True (the default), the three (n_time, n_state_bins) posteriors are
        returned as NumPy arrays. If False, they stay on the device as JAX
        arrays, so a caller that needs only some of them, or computes on them
        on the device, does not copy all three to the host.

    Returns
    -------
    result : FilterSmootherResult
        Named tuple of ``acausal_posterior``, ``acausal_state_probabilities``,
        ``marginal_log_likelihood``, ``causal_state_probabilities``,
        ``predictive_state_probabilities``, ``log_likelihoods``,
        ``causal_posterior``, ``predictive_posterior`` and
        ``degenerate_indices``; see :class:`FilterSmootherResult` for shapes.

    Raises
    ------
    ValidationError
        If ``dtype`` is neither float32 nor float64 with JAX's 64-bit mode
        on; an input's shape does not match ``n_time = len(time)`` and
        ``n_state_bins = len(initial_distribution)``; an input breaks the
        probability contract of its parameter description; or
        ``log_likelihood_func`` returns the wrong shape.
    """
    _validate_hmm_inputs(
        n_time=_validate_time(time),
        initial_distribution=initial_distribution,
        dtype=dtype,
        state_ind=state_ind,
        transition_matrix=transition_matrix,
        is_missing=is_missing,
        log_likelihoods=log_likelihoods,
    )
    transition_matrix_jax = jnp.asarray(transition_matrix, dtype=dtype)

    def filter_chunk(
        initial_distribution: jax.Array,
        log_likelihoods: jax.Array,
        time_inds: jax.Array,  # noqa: ARG001 -- chunked-filter callback signature
    ) -> tuple[Any, ...]:
        return cast(
            tuple[Any, ...],
            _filter_internal(
                initial_distribution=initial_distribution,
                transition_matrix=transition_matrix_jax,
                log_likelihoods=log_likelihoods,
            ),
        )

    def smoother_chunk(
        filtered_probs: jax.Array,
        initial: jax.Array,
        time_inds: jax.Array,
        n_time: int,
    ) -> jax.Array:
        return cast(
            jax.Array,
            _smoother_internal(
                transition_matrix=transition_matrix_jax,
                filtered_probs=filtered_probs,
                initial=initial,
                ind=time_inds,
                n_time=n_time,
            ),
        )

    return _chunked_filter_smoother(
        filter_chunk,
        smoother_chunk,
        time=time,
        state_ind=state_ind,
        initial_distribution=initial_distribution,
        log_likelihood_func=log_likelihood_func,
        log_likelihood_args=log_likelihood_args,
        is_missing=is_missing,
        n_chunks=n_chunks,
        log_likelihoods=log_likelihoods,
        cache_log_likelihoods=cache_log_likelihoods,
        dtype=dtype,
        accumulate_log_likelihoods=accumulate_log_likelihoods,
        host_outputs=host_outputs,
    )


def _viterbi_impl(
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
) -> jax.Array:
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Parameters
    ----------
    initial_distribution : jax.Array, shape (n_states,)
        Initial state distribution
    transition_matrix : jax.Array, shape (n_states, n_states)
        Transition matrix
    log_likelihoods : jax.Array, shape (n_time, n_states)
        Log likelihoods for each state at each time point.

    Returns
    -------
    most_likely_state_sequence : jax.Array, shape (n_time,)

    Notes
    -----
    Base case at t=T: terminal cost is 0, we accumulate from t=T-1 down.
    """
    # The log transition is computed once, outside the scan; _safe_log maps
    # zero probabilities to -inf, not NaN.
    return _viterbi_scan(
        initial_distribution,
        log_likelihoods,
        None,
        _StationaryTransition(transition_matrix).log(),
    )


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_viterbi_jit = jax.jit(_viterbi_impl)


def viterbi(
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
) -> jax.Array:
    """Compute the most likely state sequence with the Viterbi algorithm.

    Parameters
    ----------
    initial_distribution : jax.Array, shape (n_states,)
        Initial state distribution: nonnegative, summing to at most 1, as
        for :func:`filter`.
    transition_matrix : jax.Array, shape (n_states, n_states)
        State transition probability matrix: nonnegative, each row summing
        to at most 1.
    log_likelihoods : jax.Array, shape (n_time, n_states)
        Log likelihoods for each state at each time point.

    Returns
    -------
    most_likely_state_sequence : jax.Array, shape (n_time,)
        Most likely state sequence.

    Raises
    ------
    ValidationError
        If the inputs are concrete and their shapes disagree or they break
        the probability contract above.

    Notes
    -----
    The inputs are checked only when all of them are concrete arrays. Under
    ``jax.jit``, ``vmap`` or ``grad`` they are tracers, and the checks are
    skipped so the pass stays composable with no host transfer; check such
    inputs before the transformation.
    """
    if not _is_traced(initial_distribution, transition_matrix, log_likelihoods):
        _check_hmm_arrays(
            n_time=np.shape(log_likelihoods)[0],
            initial_distribution=initial_distribution,
            transition_matrix=transition_matrix,
            log_likelihoods=log_likelihoods,
            initial_mass_loss=True,
        )
    return cast(
        jax.Array,
        _viterbi_jit(initial_distribution, transition_matrix, log_likelihoods),
    )


def most_likely_sequence(
    time: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple[Any, ...],
    is_missing: np.ndarray | None = None,
    log_likelihoods: np.ndarray | None = None,
    n_chunks: int = 1,
    dtype: DTypeLike = jnp.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the most likely state sequence using the Viterbi algorithm.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time indices.
    initial_distribution : np.ndarray, shape (n_states,)
        Initial state distribution: nonnegative, summing to 1.
    transition_matrix : np.ndarray, shape (n_states, n_states)
        State transition probability matrix: nonnegative, each row summing
        to at most 1.
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
    dtype : dtype, optional
        Working dtype, float32 (the default) or float64. float64 requires
        JAX's 64-bit mode (``JAX_ENABLE_X64=1``) and raises
        ``ValidationError`` without it.

    Returns
    -------
    most_likely_sequence : np.ndarray, shape (n_time,)
        Most likely state sequence.
    log_likelihoods : np.ndarray, shape (n_time, n_states)
        Log likelihoods for each state at each time point.

    Raises
    ------
    ValidationError
        If ``dtype`` is neither float32 nor float64 with JAX's 64-bit mode
        on; an input's shape does not match ``n_time = len(time)`` and
        ``n_state_bins = len(initial_distribution)``; an input breaks the
        probability contract of its parameter description; or
        ``log_likelihood_func`` returns the wrong shape.
    """
    if n_chunks > 1:
        msg = "Chunked Viterbi is not yet implemented."
        raise NotImplementedError(msg)
    _validate_hmm_inputs(
        n_time=_validate_time(time),
        initial_distribution=initial_distribution,
        dtype=dtype,
        transition_matrix=transition_matrix,
        is_missing=is_missing,
        log_likelihoods=log_likelihoods,
    )

    if log_likelihoods is None:
        log_likelihoods = cast(
            np.ndarray,
            log_likelihood_func(time, *log_likelihood_args, is_missing=is_missing),
        )
        _check_shape(
            "log_likelihood_func output",
            log_likelihoods,
            (len(time), len(initial_distribution)),
            _CALLBACK_SHAPE_HINT,
        )

    # Warn if any time step is all-impossible (-inf) or contains a NaN, so a
    # Viterbi-only workflow gets the same diagnostics as the filter/smoother
    # drivers. Note ``argmax`` over an all-``-inf`` column silently returns
    # state 0, which this warning surfaces. context="viterbi" selects
    # Viterbi-appropriate wording (no posterior / marginal here).
    _warn_if_degenerate_timesteps(log_likelihoods, context="viterbi")

    # Cast to desired dtype for precision control
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    transition_matrix_jax = jnp.asarray(transition_matrix, dtype=dtype)
    log_likelihoods_jax = jnp.asarray(log_likelihoods, dtype=dtype)

    most_likely_states = _viterbi_jit(
        initial_distribution=initial_distribution_jax,
        transition_matrix=transition_matrix_jax,
        log_likelihoods=log_likelihoods_jax,
    )

    # Return NumPy array for API consistency
    return np.asarray(most_likely_states), log_likelihoods


## Covariate dependent filtering and smoothing ##
def _filter_covariate_dependent_impl(
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """JIT-compiled covariate-dependent forward pass.

    See :func:`filter_covariate_dependent` for the user-facing wrapper that
    adds host-side diagnostics. Outputs as in :func:`_filter_impl`.
    """
    return _filter_scan(
        initial_distribution,
        0.0,
        log_likelihoods,
        discrete_transition_matrix,
        _CovariateTransition.create(
            continuous_transition_matrix,
            state_ind,
            discrete_transition_matrix.shape[-1],
        ),
        keep_rows=True,
        keep_predictive=True,
        keep_increments=True,
    )


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_filter_covariate_dependent_jit = jax.jit(_filter_covariate_dependent_impl)


# Internal version with buffer donation for use in chunked drivers
# Donates initial_distribution and log_likelihoods, which have outputs of their
# shape to reuse. The (n_time, n_states, n_states) discrete_transition_matrix has
# no such output, so donating it only made JAX warn "Some donated buffers were
# not usable". Donation is specified by name because call sites pass by keyword.
_filter_covariate_dependent_internal = jax.jit(
    _filter_covariate_dependent_impl,
    donate_argnames=("initial_distribution", "log_likelihoods"),
)


def filter_covariate_dependent(
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
) -> tuple[tuple[float, jax.Array], tuple[jax.Array, jax.Array]]:
    """Run a filtering step with covariate dependent transitions.

    Parameters
    ----------
    initial_distribution : jax.Array, shape (n_state_bins,)
        Initial probability of each state bin: nonnegative, summing to at
        most 1. A sum below 1 is accepted so that a previous call's
        ``predicted_probs_next`` can start the next call even when some
        transition rows lose mass.
    discrete_transition_matrix : jax.Array, shape (n_time, n_states, n_states)
        Discrete state transition probabilities at each time bin:
        nonnegative, each row summing to 1.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Transition probability between state bins within the continuous
        state space: nonnegative, each bin's row summing to at most 1 over
        the bins of each discrete state.
    state_ind : jax.Array, shape (n_state_bins,)
        Connects the discrete states to the state space.
    log_likelihoods : jax.Array, shape (n_time, n_state_bins)
        Log likelihood of the observations for each state bin at each time.

    Returns
    -------
    marginal_likelihood : float
        Log marginal likelihood of the observations.
    predicted_probs_next : jax.Array, shape (n_state_bins,)
        Next one-step-ahead predicted state probabilities.
    causal_posterior : jax.Array, shape (n_time, n_state_bins)
        Filtered state probabilities.
    predicted_probs : jax.Array, shape (n_time, n_state_bins)
        One-step-ahead predicted state probabilities.

    Raises
    ------
    ValidationError
        If the inputs are concrete and their shapes disagree or they break
        the probability contract above.

    Notes
    -----
    The inputs are checked only when all of them are concrete arrays. Under
    ``jax.jit``, ``vmap`` or ``grad`` they are tracers, and the checks are
    skipped so the pass stays composable with no host transfer; check such
    inputs before the transformation.

    If any time step has all-``-inf`` log-likelihoods, ``_condition_on``
    falls back to the predicted distribution at that step. A ``NaN`` at a
    timestep is left to propagate (it signals a likelihood-computation bug, not
    impossible data): because the filter carries state forward, a single NaN
    contaminates the posterior at that step and every step after it. A
    host-side ``logger.warning`` summarizing the count of degenerate and/or NaN
    steps is emitted when any are found.

    ``_condition_on`` also falls back (and marks the step ``-inf`` in the
    marginal) when the Bayes normalizer is zero for a *finite*-likelihood step —
    a prediction/likelihood support mismatch or an underflow. Such a step is not
    all-``-inf``, so it is surfaced separately: when the returned marginal
    log-likelihood is ``-inf`` but no step was all-``-inf``, an additional
    ``logger.warning`` is emitted.
    """
    if not _is_traced(
        initial_distribution,
        discrete_transition_matrix,
        continuous_transition_matrix,
        state_ind,
        log_likelihoods,
    ):
        _check_hmm_arrays(
            n_time=np.shape(log_likelihoods)[0],
            initial_distribution=initial_distribution,
            state_ind=state_ind,
            discrete_transition_matrix=discrete_transition_matrix,
            continuous_transition_matrix=continuous_transition_matrix,
            log_likelihoods=log_likelihoods,
            initial_mass_loss=True,
        )
    (marginal, predicted_probs_next), (filtered, predicted, _) = (
        _filter_covariate_dependent_jit(
            initial_distribution,
            discrete_transition_matrix,
            continuous_transition_matrix,
            state_ind,
            log_likelihoods,
        )
    )
    # Forwarding the marginal surfaces support-mismatch / underflow steps that
    # are not all-(-inf).
    _warn_if_degenerate_timesteps(log_likelihoods, marginal_log_likelihood=marginal)
    return (marginal, predicted_probs_next), (filtered, predicted)


def _smoother_covariate_dependent_impl(
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    filtered_probs: ArrayLike,
    initial: ArrayLike | None = None,
    ind: ArrayLike | None = None,
    n_time: int | None = None,
) -> jax.Array:
    """Smoother step with covariate dependent transitions.

    Parameters
    ----------
    discrete_transition_matrix : jax.Array, shape (n_time, n_states, n_states)
        Per-time discrete transition matrices.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Continuous transition matrix.
    state_ind : jax.Array, shape (n_state_bins,)
        Maps each state bin to its discrete state.
    filtered_probs : jax.Array, shape (n_time, n_state_bins)
        Filtered distribution of each time bin.
    initial : jax.Array, shape (n_state_bins,), optional
        Smoothed row after the last row; by default the last filtered row.
    ind : jax.Array, shape (n_time,), optional
        Row indices of the recording; by default ``arange(n_time)``.
    n_time : int, optional
        Rows in the whole recording; by default the length of
        ``filtered_probs``.

    Returns
    -------
    smoothed_probs : jax.Array, shape (n_time, n_state_bins)
    """
    if n_time is None:
        n_time = filtered_probs.shape[0]
    if ind is None:
        ind = jnp.arange(n_time)
    if initial is None:
        initial = filtered_probs[-1]
    return cast(
        jax.Array,
        _smoother_scan(
            filtered_probs,
            discrete_transition_matrix,
            _CovariateTransition.create(
                continuous_transition_matrix,
                state_ind,
                discrete_transition_matrix.shape[-1],
            ),
            initial,
            ind,
            n_time,
        )[1],
    )


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_smoother_covariate_dependent_jit = jax.jit(_smoother_covariate_dependent_impl)


def smoother_covariate_dependent(
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    filtered_probs: ArrayLike,
    initial: ArrayLike | None = None,
    ind: ArrayLike | None = None,
    n_time: int | None = None,
) -> jax.Array:
    """Run the smoother with covariate dependent transitions.

    Parameters
    ----------
    discrete_transition_matrix : jax.Array, shape (n_rows, n_states, n_states)
        Discrete transition matrix of each row of ``filtered_probs``:
        nonnegative, each row summing to 1.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Continuous transition matrix: nonnegative, each bin's row summing to
        at most 1 over the bins of each discrete state.
    state_ind : jax.Array, shape (n_state_bins,)
        Maps each state bin to its discrete state.
    filtered_probs : jax.Array, shape (n_rows, n_state_bins)
        Filtered distribution of each time bin.
    initial : jax.Array, shape (n_state_bins,), optional
        Smoothed row after the last row; by default the last filtered row.
    ind : jax.Array, shape (n_rows,), optional
        Row indices of the recording; by default ``arange(n_rows)``.
    n_time : int, optional
        Rows in the whole recording; by default the length of
        ``filtered_probs``.

    Returns
    -------
    smoothed_probs : jax.Array, shape (n_rows, n_state_bins)
        Smoothed state probabilities.

    Raises
    ------
    ValidationError
        If the inputs are concrete and their shapes disagree or they break
        the probability contract above.

    Notes
    -----
    The inputs are checked only when all of them are concrete arrays. Under
    ``jax.jit``, ``vmap`` or ``grad`` they are tracers, and the checks are
    skipped so the pass stays composable with no host transfer; check such
    inputs before the transformation.
    """
    if not _is_traced(
        discrete_transition_matrix,
        continuous_transition_matrix,
        state_ind,
        filtered_probs,
    ):
        _check_hmm_arrays(
            n_time=np.shape(filtered_probs)[0],
            n_bins=np.shape(filtered_probs)[-1],
            state_ind=state_ind,
            discrete_transition_matrix=discrete_transition_matrix,
            continuous_transition_matrix=continuous_transition_matrix,
        )
    return cast(
        jax.Array,
        _smoother_covariate_dependent_jit(
            discrete_transition_matrix,
            continuous_transition_matrix,
            state_ind,
            filtered_probs,
            initial,
            ind,
            n_time,
        ),
    )


# Internal version with buffer donation for use in chunked drivers
# Donates filtered_probs, the only input with an output of its shape to reuse
# (a donated buffer without one makes JAX warn "Some donated buffers were not
# usable"). Donation is specified by name because call sites pass by keyword.
_smoother_covariate_dependent_internal = jax.jit(
    _smoother_covariate_dependent_impl, donate_argnames=("filtered_probs",)
)


def chunked_filter_smoother_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple[Any, ...],
    is_missing: np.ndarray | None = None,
    n_chunks: int = 1,
    log_likelihoods: np.ndarray | None = None,
    # positional: ``dtype`` after it is public positional API (a test pins it)
    cache_log_likelihoods: bool = True,  # noqa: FBT001, FBT002
    dtype: DTypeLike = jnp.float32,
    *,
    accumulate_log_likelihoods: bool = False,
    host_outputs: bool = True,
) -> FilterSmootherResult:
    """Filter and smooth the state probabilities in chunks with covariate dependent transitions.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time of each decoding bin (bin centers, in seconds), passed to
        ``log_likelihood_func``. Its length sets the number of time bins.
    state_ind : np.ndarray, shape (n_state_bins,)
        Connects the discrete states to the state space.
    initial_distribution : np.ndarray, shape (n_state_bins,)
        Initial probability of each state bin: nonnegative, summing to 1.
    discrete_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Discrete state transition probabilities at each time bin: nonnegative,
        each row summing to 1.
    continuous_transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
        Transition probability between state bins within the continuous
        state space: nonnegative, each bin's row summing to at most 1 over
        the bins of each discrete state.
    log_likelihood_func : callable
        Called as ``log_likelihood_func(time, *log_likelihood_args,
        is_missing=...)`` with the FULL ``time``. Without cached or given log
        likelihoods, a callable marked by :func:`row_slice_aware` is called
        per chunk and also receives ``row_slice=slice(start, stop)`` and the
        chunk's ``is_missing`` rows, so each decoding spike is binned once
        against the whole timeline and lands in the row that owns it. An
        unmarked callable is called once for the whole timeline; with
        ``n_chunks > 1`` and no cached likelihoods it raises ``ValidationError``.
    log_likelihood_args : tuple
        Extra positional arguments passed to ``log_likelihood_func`` after
        ``time``.
    is_missing : np.ndarray, shape (n_time,), optional
        True for time bins without observations, which get neutral
        likelihoods. By default None (none missing).
    n_chunks : int, optional
        Number of chunks to split the data into, by default 1.
        Each step applies the discrete transition per discrete state on top of
        the continuous transition, so no joint (n_state_bins, n_state_bins)
        transition is built per step and the (n_time, n_state_bins,
        n_state_bins) one never exists; ``n_chunks=1`` is feasible even on long
        sessions.
    log_likelihoods : np.ndarray, optional
        Precomputed log likelihoods, shape (n_time, n_state_bins), used
        instead of calling ``log_likelihood_func``. By default None.
    cache_log_likelihoods : bool, optional
        If True, log likelihoods are cached instead of recomputed for each
        chunk, by default True.
    dtype : dtype, optional
        Working dtype, float32 (the default) or float64. Use float64 for
        numerically challenging problems; it requires JAX's 64-bit mode
        (``JAX_ENABLE_X64=1``) and raises ``ValidationError`` without it.
    accumulate_log_likelihoods : bool, optional, keyword-only
        If True and the log likelihoods are not cached, each chunk's rows are
        copied into a single preallocated host array so the returned
        ``log_likelihoods`` covers every row in global order instead of being
        ``None``. This allocates the full (n_time, n_state_bins) array, so set
        it only when the caller explicitly asked for the log likelihoods.
        Keyword-only, and last, so that it cannot capture a positional argument
        aimed at a parameter that predates it (``dtype``). By default False.
    host_outputs : bool, optional, keyword-only
        If True (the default), the three (n_time, n_state_bins) posteriors are
        returned as NumPy arrays. If False, they stay on the device as JAX
        arrays, so a caller that needs only some of them, or computes on them
        on the device, does not copy all three to the host.

    Returns
    -------
    result : FilterSmootherResult
        Named tuple of ``acausal_posterior``, ``acausal_state_probabilities``,
        ``marginal_log_likelihood``, ``causal_state_probabilities``,
        ``predictive_state_probabilities``, ``log_likelihoods``,
        ``causal_posterior``, ``predictive_posterior`` and
        ``degenerate_indices``; see :class:`FilterSmootherResult` for shapes.

    Raises
    ------
    ValidationError
        If ``dtype`` is neither float32 nor float64 with JAX's 64-bit mode
        on; an input's shape does not match ``n_time = len(time)`` and
        ``n_state_bins = len(initial_distribution)``; an input breaks the
        probability contract of its parameter description; or
        ``log_likelihood_func`` returns the wrong shape.
    """
    _validate_hmm_inputs(
        n_time=_validate_time(time),
        initial_distribution=initial_distribution,
        dtype=dtype,
        state_ind=state_ind,
        discrete_transition_matrix=discrete_transition_matrix,
        continuous_transition_matrix=continuous_transition_matrix,
        is_missing=is_missing,
        log_likelihoods=log_likelihoods,
    )
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    continuous_transition_matrix_jax = jnp.asarray(
        continuous_transition_matrix, dtype=dtype
    )
    # Convert the discrete transitions but do not expand them to state bins:
    # each chunk takes its time slice, and the scan body applies them per
    # discrete state, so no joint (n_state_bins, n_state_bins) transition is
    # materialized.
    discrete_transition_matrix_jax = jnp.asarray(
        discrete_transition_matrix, dtype=dtype
    )

    def filter_chunk(
        initial_distribution: jax.Array,
        log_likelihoods: jax.Array,
        time_inds: jax.Array,
    ) -> tuple[Any, ...]:
        # The time slice is not donated: no output has its shape to reuse.
        return cast(
            tuple[Any, ...],
            _filter_covariate_dependent_internal(
                initial_distribution=initial_distribution,
                discrete_transition_matrix=discrete_transition_matrix_jax[time_inds],
                continuous_transition_matrix=continuous_transition_matrix_jax,
                state_ind=state_ind_jax,
                log_likelihoods=log_likelihoods,
            ),
        )

    def smoother_chunk(
        filtered_probs: jax.Array,
        initial: jax.Array,
        time_inds: jax.Array,
        n_time: int,
    ) -> jax.Array:
        # The time slice is not donated: no output has its shape to reuse.
        return cast(
            jax.Array,
            _smoother_covariate_dependent_internal(
                discrete_transition_matrix=discrete_transition_matrix_jax[time_inds],
                continuous_transition_matrix=continuous_transition_matrix_jax,
                state_ind=state_ind_jax,
                filtered_probs=filtered_probs,
                initial=initial,
                ind=time_inds,
                n_time=n_time,
            ),
        )

    return _chunked_filter_smoother(
        filter_chunk,
        smoother_chunk,
        time=time,
        state_ind=state_ind,
        initial_distribution=initial_distribution,
        log_likelihood_func=log_likelihood_func,
        log_likelihood_args=log_likelihood_args,
        is_missing=is_missing,
        n_chunks=n_chunks,
        log_likelihoods=log_likelihoods,
        cache_log_likelihoods=cache_log_likelihoods,
        dtype=dtype,
        accumulate_log_likelihoods=accumulate_log_likelihoods,
        host_outputs=host_outputs,
    )


def _viterbi_covariate_dependent_impl(
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
) -> jax.Array:
    r"""Compute the most likely state sequence with covariate dependent transitions.

    Uses the Viterbi algorithm with time-varying transition matrices.

    Parameters
    ----------
    initial_distribution : jax.Array, shape (n_state_bins,)
        Initial state distribution.
    discrete_transition_matrix : jax.Array, shape (n_time, n_states, n_states)
        Time-varying discrete transition matrices.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Continuous transition matrix.
    state_ind : jax.Array, shape (n_state_bins,)
        State indices connecting discrete states to state space bins.
    log_likelihoods : jax.Array, shape (n_time, n_state_bins)
        Log likelihoods for each state bin at each time point.

    Returns
    -------
    most_likely_state_sequence : jax.Array, shape (n_time,)
        Most likely state sequence.

    Notes
    -----
    Base case at t=T: terminal cost is 0, we accumulate from t=T-1 down.
    """
    # The logs are computed once, outside the scan; _safe_log maps zero
    # probabilities to -inf, not NaN. The (n_time, n_states, n_states)
    # discrete matrices are logged, not the expanded (n_state_bins,
    # n_state_bins) ones: same values, n_states**2 logs per step, and the
    # discrete transitions are indexed per step, so no (n_time, n_state_bins,
    # n_state_bins) array is materialized.
    return _viterbi_scan(
        initial_distribution,
        log_likelihoods,
        _safe_log(discrete_transition_matrix),
        _CovariateTransition.create(
            continuous_transition_matrix,
            state_ind,
            discrete_transition_matrix.shape[-1],
        ).log(),
    )


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_viterbi_covariate_dependent_jit = jax.jit(_viterbi_covariate_dependent_impl)


def viterbi_covariate_dependent(
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
) -> jax.Array:
    """Compute the most likely state sequence with covariate dependent transitions.

    Parameters
    ----------
    initial_distribution : jax.Array, shape (n_state_bins,)
        Initial state distribution: nonnegative, summing to at most 1, as
        for :func:`filter_covariate_dependent`.
    discrete_transition_matrix : jax.Array, shape (n_time, n_states, n_states)
        Time-varying discrete transition matrices: nonnegative, each row
        summing to 1.
    continuous_transition_matrix : jax.Array, shape (n_state_bins, n_state_bins)
        Continuous transition matrix: nonnegative, each bin's row summing to
        at most 1 over the bins of each discrete state.
    state_ind : jax.Array, shape (n_state_bins,)
        State indices connecting discrete states to state space bins.
    log_likelihoods : jax.Array, shape (n_time, n_state_bins)
        Log likelihoods for each state bin at each time point.

    Returns
    -------
    most_likely_state_sequence : jax.Array, shape (n_time,)
        Most likely state sequence.

    Raises
    ------
    ValidationError
        If the inputs are concrete and their shapes disagree or they break
        the probability contract above.

    Notes
    -----
    The inputs are checked only when all of them are concrete arrays. Under
    ``jax.jit``, ``vmap`` or ``grad`` they are tracers, and the checks are
    skipped so the pass stays composable with no host transfer; check such
    inputs before the transformation.
    """
    if not _is_traced(
        initial_distribution,
        discrete_transition_matrix,
        continuous_transition_matrix,
        state_ind,
        log_likelihoods,
    ):
        _check_hmm_arrays(
            n_time=np.shape(log_likelihoods)[0],
            initial_distribution=initial_distribution,
            state_ind=state_ind,
            discrete_transition_matrix=discrete_transition_matrix,
            continuous_transition_matrix=continuous_transition_matrix,
            log_likelihoods=log_likelihoods,
            initial_mass_loss=True,
        )
    return cast(
        jax.Array,
        _viterbi_covariate_dependent_jit(
            initial_distribution,
            discrete_transition_matrix,
            continuous_transition_matrix,
            state_ind,
            log_likelihoods,
        ),
    )


def most_likely_sequence_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple[Any, ...],
    is_missing: np.ndarray | None = None,
    log_likelihoods: np.ndarray | None = None,
    n_chunks: int = 1,
    dtype: DTypeLike = jnp.float32,
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
        Initial state distribution: nonnegative, summing to 1.
    discrete_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Time-varying discrete transition matrices: nonnegative, each row
        summing to 1.
    continuous_transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
        Continuous transition matrix: nonnegative, each bin's row summing to
        at most 1 over the bins of each discrete state.
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
    dtype : dtype, optional
        Working dtype, float32 (the default) or float64. float64 requires
        JAX's 64-bit mode (``JAX_ENABLE_X64=1``) and raises
        ``ValidationError`` without it.

    Returns
    -------
    most_likely_sequence : np.ndarray, shape (n_time,)
        Most likely state sequence.
    log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        Log likelihoods for each state bin at each time point.

    Raises
    ------
    ValidationError
        If ``dtype`` is neither float32 nor float64 with JAX's 64-bit mode
        on; an input's shape does not match ``n_time = len(time)`` and
        ``n_state_bins = len(initial_distribution)``; an input breaks the
        probability contract of its parameter description; or
        ``log_likelihood_func`` returns the wrong shape.
    """
    if n_chunks > 1:
        msg = "Chunked Viterbi is not yet implemented."
        raise NotImplementedError(msg)
    _validate_hmm_inputs(
        n_time=_validate_time(time),
        initial_distribution=initial_distribution,
        dtype=dtype,
        state_ind=state_ind,
        discrete_transition_matrix=discrete_transition_matrix,
        continuous_transition_matrix=continuous_transition_matrix,
        is_missing=is_missing,
        log_likelihoods=log_likelihoods,
    )

    if log_likelihoods is None:
        log_likelihoods = cast(
            np.ndarray,
            log_likelihood_func(time, *log_likelihood_args, is_missing=is_missing),
        )
        _check_shape(
            "log_likelihood_func output",
            log_likelihoods,
            (len(time), len(initial_distribution)),
            _CALLBACK_SHAPE_HINT,
        )

    # Warn if any time step is all-impossible (-inf) or contains a NaN, so a
    # Viterbi-only workflow gets the same diagnostics as the filter/smoother
    # drivers. Note ``argmax`` over an all-``-inf`` column silently returns
    # state 0, which this warning surfaces. context="viterbi" selects
    # Viterbi-appropriate wording (no posterior / marginal here).
    _warn_if_degenerate_timesteps(log_likelihoods, context="viterbi")

    # Cast to desired dtype for precision control
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    discrete_transition_matrix_jax = jnp.asarray(
        discrete_transition_matrix, dtype=dtype
    )
    continuous_transition_matrix_jax = jnp.asarray(
        continuous_transition_matrix, dtype=dtype
    )
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    log_likelihoods_jax = jnp.asarray(log_likelihoods, dtype=dtype)

    most_likely_states = _viterbi_covariate_dependent_jit(
        initial_distribution=initial_distribution_jax,
        discrete_transition_matrix=discrete_transition_matrix_jax,
        continuous_transition_matrix=continuous_transition_matrix_jax,
        state_ind=state_ind_jax,
        log_likelihoods=log_likelihoods_jax,
    )

    # Return NumPy array for API consistency
    return np.asarray(most_likely_states), log_likelihoods


## Convergence check ##
# A relative decrease this small is float rounding, not a monotonicity
# violation: the per-row evidence increments are float32-accurate (their sum
# is float64), so two evaluations of the same objective can differ by a few
# float32 ulps relative to its magnitude.
EM_MONOTONICITY_SLACK = 10 * float(np.finfo(np.float32).eps)


def check_converged(
    log_likelihood: float,
    previous_log_likelihood: float,
    tolerance: float = 1e-5,
    *,
    n_time_bins: int,
) -> tuple[bool, bool]:
    """Check EM convergence and monotonicity from two objective values.

    EM has converged when the objective no longer decreases and its change
    per time bin, ``|f(t) - f(t-1)| / n_time_bins``, falls below
    ``tolerance`` nats. A decrease larger than float rounding
    (``EM_MONOTONICITY_SLACK`` relative to the objective's magnitude) is a
    monotonicity violation and never counts as convergence, however small it
    is.

    Parameters
    ----------
    log_likelihood : float
        Current objective (marginal log-likelihood plus any log prior the
        M-step maximizes).
    previous_log_likelihood : float
        Previous objective.
    tolerance : float, optional
        Convergence threshold on the objective's gain, in nats per time bin,
        by default 1e-5.
    n_time_bins : int
        Number of observed (non-missing) time bins the objective sums over.
        Missing bins have a log-likelihood of zero and add nothing to the
        objective, so they do not count.

    Returns
    -------
    is_converged : bool
        True if the objective did not decrease and its change per time bin
        is below ``tolerance``. False while the objective stays ``-inf``.
    is_increasing : bool
        True if the objective did not decrease by more than float rounding.
    """
    log_likelihood = float(log_likelihood)
    previous_log_likelihood = float(previous_log_likelihood)
    if not (np.isfinite(log_likelihood) and np.isfinite(previous_log_likelihood)):
        # A -inf objective (an impossible timestep, or a prior on a zero
        # transition) has no finite change: compare the values directly,
        # without a NaN ratio. -inf after -inf is no decrease, but leaves no
        # change to judge convergence by; NaN is never an increase.
        return False, bool(log_likelihood >= previous_log_likelihood)

    change = log_likelihood - previous_log_likelihood
    # Rounding error scales with the objective's magnitude, so the
    # monotonicity slack is relative to it.
    avg_log_likelihood = (
        abs(log_likelihood) + abs(previous_log_likelihood) + np.spacing(1)
    ) / 2
    is_increasing = bool(change / avg_log_likelihood >= -EM_MONOTONICITY_SLACK)
    # Convergence is judged per time bin, not relative to the objective: the
    # objective's magnitude is set by the spike or mark likelihood (and shifts
    # with mark or rate units), not by how much EM is still improving the fit,
    # while the gain per bin does not depend on recording length.
    is_converged = is_increasing and bool(abs(change) / n_time_bins < tolerance)

    return is_converged, is_increasing
