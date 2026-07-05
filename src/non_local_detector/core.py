import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

logger = logging.getLogger(__name__)


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
        Predicted state probabilities (sum to 1).
    ll : jnp.ndarray
        Log-likelihood at this time step. Same shape as ``probs``.

    Returns
    -------
    new_probs : jnp.ndarray
        Updated state probabilities. When the Bayes normalizer is exactly zero
        — either every entry of ``ll`` is ``-inf`` (all states impossible) or
        the predicted ``probs`` place zero mass on every state with finite
        likelihood (a prediction/likelihood support mismatch) — the predicted
        probabilities are returned unchanged and ``log_norm`` is set to
        ``-inf`` to mark the step in the marginal likelihood, rather than
        propagating an all-zero (invalid) posterior forward. A ``NaN`` in
        ``ll`` (a likelihood-computation bug, not impossible data) is NOT
        masked: it makes the normalizer ``NaN`` (not zero), so it propagates
        into ``new_probs`` and stays visible to the caller.
    log_norm : float
        Log normalization constant, or ``-inf`` when the normalizer is zero.
    """
    ll_max = ll.max()
    # Shift by the max for numerical stability. When the max is non-finite
    # (all -inf, or a NaN present) use 0.0 so any finite states are not
    # corrupted by an inf - inf subtraction.
    ll_max_safe = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    new_probs_normal, norm = _normalize(probs * jnp.exp(ll - ll_max_safe))
    log_norm = jnp.log(norm) + ll_max_safe

    # Fallback when the normalizer is exactly zero. Two distinct situations
    # produce a 0/0 Bayes update:
    #   1. every state is -inf (all-impossible data), or
    #   2. the predicted distribution places zero mass on every state with
    #      finite likelihood (a prediction/likelihood support mismatch).
    # In both, return the predicted distribution unchanged and mark log_norm as
    # -inf so the step is recorded in the marginal likelihood instead of
    # silently propagating an all-zero (invalid) posterior. A NaN in ``ll``
    # makes ``norm`` NaN (not 0), so ``norm == 0`` is False and the NaN flows
    # through the normal branch into new_probs, keeping the upstream bug visible
    # rather than laundering it into the predicted prior.
    is_zero_norm = norm == 0.0
    new_probs = jnp.where(is_zero_norm, probs, new_probs_normal)
    log_norm = jnp.where(is_zero_norm, -jnp.inf, log_norm)
    return new_probs, log_norm


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
    ll_array = jnp.asarray(log_likelihoods)
    degenerate_mask = np.asarray(ll_array.max(axis=-1) == -jnp.inf)
    nan_mask = np.asarray(jnp.any(jnp.isnan(ll_array), axis=-1))
    return degenerate_mask, nan_mask


def _accumulate_chunk_degeneracy(
    log_likelihood_chunk: ArrayLike,
    time_inds: np.ndarray,
    degenerate_indices_out: list | None,
) -> tuple[int, int]:
    """Tally a chunk's degenerate/NaN steps at one host sync point.

    Returns ``(n_degenerate, n_nan)`` for the chunk and, when
    ``degenerate_indices_out`` is provided, appends the *global* indices of the
    chunk's all-``-inf`` steps (``time_inds`` maps chunk positions to global
    time indices).
    """
    degenerate_mask, nan_mask = _degenerate_and_nan_masks(log_likelihood_chunk)
    if degenerate_indices_out is not None:
        degenerate_indices_out.extend(np.asarray(time_inds)[degenerate_mask].tolist())
    return int(degenerate_mask.sum()), int(nan_mask.sum())


def _warn_degenerate_and_nan_timesteps(
    n_degenerate: int, n_nan: int, n_total: int
) -> None:
    """Emit host-side warnings summarizing degenerate and NaN time steps.

    These diagnostics use ``logger.warning`` (not ``warnings.warn``) because
    they are per-invocation decoding diagnostics — they should fire on every
    affected filter call rather than be deduplicated by the warnings filter.
    The fit-time convergence diagnostics use ``warnings.warn(UserWarning)``
    instead (see ``models.base`` / the likelihood models).
    """
    if n_degenerate > 0:
        logger.warning(
            "HMM filter encountered %d/%d time step(s) with all-impossible "
            "(-inf) log-likelihoods; the posterior at those steps fell back to "
            "the predicted distribution.",
            n_degenerate,
            n_total,
        )
    if n_nan > 0:
        logger.warning(
            "HMM filter encountered %d/%d time step(s) with NaN log-likelihoods; "
            "the posterior is NaN at those steps and propagates to every later "
            "step. This indicates a bug in the likelihood computation (e.g. a "
            "non-converged or degenerate encoding model), not impossible data. "
            "Check the log-likelihood inputs.",
            n_nan,
            n_total,
        )


def _warn_if_degenerate_timesteps(log_likelihoods: ArrayLike) -> None:
    """Emit host-side warnings if any time step is all-``-inf`` or has a NaN.

    Parameters
    ----------
    log_likelihoods : array-like
        Log-likelihood array. The last axis is the state axis.

    Notes
    -----
    Skipped when ``log_likelihoods`` is a JAX tracer (i.e. ``filter`` /
    ``filter_covariate_dependent`` are being run inside ``jit``/``vmap``/
    ``scan``). The diagnostics need concrete values (``int(jnp.sum(...))``)
    and warnings cannot be emitted during tracing, so skipping keeps the
    public filter wrappers JAX-transformable; the pure filtering computation
    still runs.
    """
    ll_array = jnp.asarray(log_likelihoods)
    if isinstance(ll_array, jax.core.Tracer):
        return
    n_total = int(np.prod(ll_array.shape[:-1])) if ll_array.ndim > 1 else 1
    degenerate_mask, nan_mask = _degenerate_and_nan_masks(ll_array)
    _warn_degenerate_and_nan_timesteps(
        int(degenerate_mask.sum()), int(nan_mask.sum()), n_total
    )


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
    # Double-where: substitute safe value into operand first
    safe_p = jnp.where(p > 0, p, 1.0)
    return jnp.where(p > 0, jnp.log(safe_p), -jnp.inf)


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
    # Double-where: substitute safe denominator first, then select result.
    # This avoids NaN in both forward pass and gradients.
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    return jnp.where(denominator != 0.0, numerator / safe_denominator, 0.0)


def _filter_impl(
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    log_likelihoods: ArrayLike,
) -> tuple[tuple[float, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """JIT-compiled forward pass of the forward-backward algorithm.

    See :func:`filter` for the user-facing wrapper that adds host-side
    diagnostics. This raw implementation runs ``jax.lax.scan`` and is wrapped
    by both the non-donating ``_filter_jit`` (used by the public ``filter``)
    and the donating ``_filter_internal`` (used by the chunked drivers).
    """

    def _step(carry, ll):
        log_normalizer, predicted_probs = carry

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = filtered_probs @ transition_matrix

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    return jax.lax.scan(_step, (0.0, initial_distribution), log_likelihoods)


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

    Notes
    -----
    If any time step has all-``-inf`` log-likelihoods, ``_condition_on``
    falls back to the predicted distribution at that step. A ``NaN`` at a
    timestep is left to propagate (it signals a likelihood-computation bug, not
    impossible data): because the filter carries state forward, a single NaN
    contaminates the posterior at that step and every step after it. A
    host-side ``logger.warning`` summarizing the count of degenerate and/or NaN
    steps is emitted when any are found.
    """
    result = _filter_jit(initial_distribution, transition_matrix, log_likelihoods)
    _warn_if_degenerate_timesteps(log_likelihoods)
    return result


def _smoother_impl(
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
        smoothed_probs, _ = _normalize(smoothed_probs, axis=-1)

        return smoothed_probs, smoothed_probs

    return jax.lax.scan(
        _step,
        initial,
        (filtered_probs, ind),
        reverse=True,
    )[1]


# Apply JIT without donation - tests reuse inputs
smoother = jax.jit(_smoother_impl)


# Internal version with buffer donation for use in chunked drivers
# Donates filtered_probs and initial - both are single-use in chunked loops.
# Donation is specified by name (call sites pass by keyword).
_smoother_internal = jax.jit(
    _smoother_impl, donate_argnames=("filtered_probs", "initial")
)


def chunked_filter_smoother(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    n_chunks: int = 1,
    log_likelihoods: np.ndarray | None = None,
    cache_log_likelihoods: bool = True,
    dtype: jnp.dtype = jnp.float32,
    degenerate_indices_out: list | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
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
    log_likelihoods : np.ndarray | None, optional
    cache_log_likelihoods : bool, optional
        If True, log likelihoods are cached, by default True
    dtype : jnp.dtype, optional
        Data type for computations (jnp.float32 or jnp.float64), by default jnp.float32.
        Use float64 for numerically challenging problems.
        Note: Requires JAX_ENABLE_X64=1 environment variable for float64.
    degenerate_indices_out : list | None, optional
        If provided, the global time indices of all-``-inf`` (degenerate)
        timesteps are appended to this list during the forward pass. This is
        the only way to recover those indices when likelihoods are not cached
        (the returned ``log_likelihoods`` is then ``None``). By default None.

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
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        One-step-ahead predicted state probabilities over state bins
    """
    causal_posterior = []
    predictive_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    marginal_likelihood = 0.0
    n_degenerate_total = 0
    n_nan_total = 0

    n_time = len(time)
    # Validate n_chunks: ensure it doesn't exceed n_time to avoid empty chunks
    if n_chunks > n_time:
        raise ValueError(
            f"n_chunks ({n_chunks}) cannot exceed n_time ({n_time}). "
            f"Each chunk must contain at least one time point."
        )
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    # Convert inputs to JAX arrays once at the top - keep data on device
    # Cast to desired dtype for precision control
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    transition_matrix_jax = jnp.asarray(transition_matrix, dtype=dtype)

    # Create aggregation matrix for marginalizing state bins -> discrete states
    # This is reused across all chunks
    n_discrete_states = int(jnp.max(state_ind_jax)) + 1
    state_aggregation_matrix = jax.nn.one_hot(
        state_ind_jax, n_discrete_states
    )  # (n_state_bins, n_states)

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

        # Tally degenerate (all -inf) and NaN timesteps at one host sync point
        # before the array is donated to the JIT call. Accumulated across chunks
        # and reported once after the forward pass.
        n_degen, n_nan = _accumulate_chunk_degeneracy(
            log_likelihood_chunk, time_inds_np, degenerate_indices_out
        )
        n_degenerate_total += n_degen
        n_nan_total += n_nan

        # Donated: log_likelihood_chunk (created fresh), initial_distribution
        # Do not read these after the call - they are consumed by the JIT function
        # Note: On first iteration, initial_distribution_jax may trigger a donation warning
        # if the array is small (< ~1 KB). This is benign - JAX skips donation for small
        # arrays where the overhead exceeds the benefit. The optimization works correctly
        # for large production workloads.
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
        # Marginalize state bins -> discrete states by summing probabilities
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(
            causal_posterior_chunk @ state_aggregation_matrix
        )
        predictive_posterior.append(predicted_probs_chunk)
        predictive_state_probabilities.append(
            predicted_probs_chunk @ state_aggregation_matrix
        )

        marginal_likelihood += marginal_likelihood_chunk

    _warn_degenerate_and_nan_timesteps(n_degenerate_total, n_nan_total, n_time)

    # Concatenate JAX arrays on device
    causal_posterior_jax = jnp.concatenate(causal_posterior)
    causal_state_probabilities_jax = jnp.concatenate(causal_state_probabilities)
    predictive_posterior_jax = jnp.concatenate(predictive_posterior)
    predictive_state_probabilities_jax = jnp.concatenate(predictive_state_probabilities)

    # Backward pass: accumulate JAX arrays
    for chunk_id, time_inds_np in enumerate(reversed(time_chunks)):
        time_inds = jnp.asarray(time_inds_np)

        # Use internal version with buffer donation
        # Safe because causal_posterior_jax[time_inds] creates a slice (copy)
        # Note: Small arrays may trigger benign donation warnings in tests
        acausal_posterior_chunk = _smoother_internal(
            transition_matrix=transition_matrix_jax,
            filtered_probs=causal_posterior_jax[time_inds],
            initial=(
                causal_posterior_jax[-1]
                if chunk_id == 0
                else acausal_posterior_chunk[0]
            ),
            ind=time_inds,
            n_time=n_time,
        )
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(
            acausal_posterior_chunk @ state_aggregation_matrix
        )

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
        np.asarray(predictive_posterior_jax),
    )


def _viterbi_impl(
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

    return jnp.concatenate([first_state[None], states])


# Apply JIT without donation - tests reuse inputs
viterbi = jax.jit(_viterbi_impl)


def most_likely_sequence(
    time: np.ndarray,
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    log_likelihoods: np.ndarray | None = None,
    n_chunks: int = 1,
    dtype: jnp.dtype = jnp.float32,
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
    dtype : jnp.dtype, optional
        Data type for computations (jnp.float32 or jnp.float64), by default jnp.float32.
        Note: Requires JAX_ENABLE_X64=1 environment variable for float64.

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

    # Warn if any time step is all-impossible (-inf) or contains a NaN, so a
    # Viterbi-only workflow gets the same diagnostics as the filter/smoother
    # drivers. Note ``argmax`` over an all-``-inf`` column silently returns
    # state 0, which this warning surfaces.
    _warn_if_degenerate_timesteps(log_likelihoods)

    # Cast to desired dtype for precision control
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    transition_matrix_jax = jnp.asarray(transition_matrix, dtype=dtype)
    log_likelihoods_jax = jnp.asarray(log_likelihoods, dtype=dtype)

    most_likely_states = viterbi(
        initial_distribution=initial_distribution_jax,
        transition_matrix=transition_matrix_jax,
        log_likelihoods=log_likelihoods_jax,
    )

    # Return NumPy array for API consistency
    return np.asarray(most_likely_states), log_likelihoods


## Covariate dependent filtering and smoothing ##
def _index_discrete_transition(
    discrete_transition_matrix_t: ArrayLike,
    state_ind: ArrayLike,
) -> jnp.ndarray:
    """Expand a discrete transition to the joint state-bin space.

    Parameters
    ----------
    discrete_transition_matrix_t : jnp.ndarray, shape (n_states, n_states)
    state_ind : jnp.ndarray, shape (n_state_bins,)
        Maps each state bin to its discrete state.

    Returns
    -------
    discrete_indexed : jnp.ndarray, shape (n_state_bins, n_state_bins)
    """
    return discrete_transition_matrix_t[jnp.ix_(state_ind, state_ind)]


def _get_transition_matrix(
    discrete_transition_matrix_t: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
) -> jnp.ndarray:
    """Get the transition matrix for the current time point.

    Combines the discrete and continuous transition matrices.

    Parameters
    ----------
    discrete_transition_matrix_t : jnp.ndarray, shape (n_states, n_states)
    continuous_transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : jnp.ndarray, shape (n_state_bins,)
        Maps each state bin to its discrete state.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (n_state_bins, n_state_bins)
    """
    discrete_indexed = _index_discrete_transition(
        discrete_transition_matrix_t, state_ind
    )
    return continuous_transition_matrix * discrete_indexed


def _filter_covariate_dependent_impl(
    initial_distribution: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    continuous_transition_matrix: ArrayLike,
    state_ind: ArrayLike,
    log_likelihoods: ArrayLike,
) -> tuple[tuple[float, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """JIT-compiled covariate-dependent forward pass.

    See :func:`filter_covariate_dependent` for the user-facing wrapper that
    adds host-side diagnostics.
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


# JIT-compiled non-donating implementation (safe to reuse inputs across calls)
_filter_covariate_dependent_jit = jax.jit(_filter_covariate_dependent_impl)


# Internal version with buffer donation for use in chunked drivers
# Donates initial_distribution, discrete_transition_matrix, and log_likelihoods.
# dtm is chunked per-iteration so safe to donate. Donation is specified by name
# because call sites pass by keyword.
_filter_covariate_dependent_internal = jax.jit(
    _filter_covariate_dependent_impl,
    donate_argnames=(
        "initial_distribution",
        "discrete_transition_matrix",
        "log_likelihoods",
    ),
)


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
    log_likelihoods : jnp.ndarray, shape (n_time, n_state_bins)

    Returns
    -------
    marginal_likelihood : float
    predicted_probs_next : jnp.ndarray, shape (n_state_bins,)
        Next one-step-ahead predicted state probabilities
    causal_posterior : jnp.ndarray, shape (n_time, n_state_bins)
        Filtered state probabilities
    predicted_probs : jnp.ndarray, shape (n_time, n_state_bins)
        One-step-ahead predicted state probabilities

    Notes
    -----
    If any time step has all-``-inf`` log-likelihoods, ``_condition_on``
    falls back to the predicted distribution at that step. A ``NaN`` at a
    timestep is left to propagate (it signals a likelihood-computation bug, not
    impossible data): because the filter carries state forward, a single NaN
    contaminates the posterior at that step and every step after it. A
    host-side ``logger.warning`` summarizing the count of degenerate and/or NaN
    steps is emitted when any are found.
    """
    result = _filter_covariate_dependent_jit(
        initial_distribution,
        discrete_transition_matrix,
        continuous_transition_matrix,
        state_ind,
        log_likelihoods,
    )
    _warn_if_degenerate_timesteps(log_likelihoods)
    return result


def _smoother_covariate_dependent_impl(
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
        Maps each state bin to its discrete state.
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
        smoothed_probs, _ = _normalize(smoothed_probs, axis=-1)

        return smoothed_probs, smoothed_probs

    return jax.lax.scan(
        _step,
        initial,
        (filtered_probs, discrete_transition_matrix, ind),
        reverse=True,
    )[1]


# Apply JIT without donation - tests reuse inputs
smoother_covariate_dependent = jax.jit(_smoother_covariate_dependent_impl)


# Internal version with buffer donation for use in chunked drivers
# Donates discrete_transition_matrix, filtered_probs, and initial.
# dtm is chunked per-iteration, all are single-use. Donation is specified by name
# because call sites pass by keyword.
_smoother_covariate_dependent_internal = jax.jit(
    _smoother_covariate_dependent_impl,
    donate_argnames=("discrete_transition_matrix", "filtered_probs", "initial"),
)


def chunked_filter_smoother_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    n_chunks: int = 1,
    log_likelihoods: np.ndarray | None = None,
    cache_log_likelihoods: bool = True,
    dtype: jnp.dtype = jnp.float32,
    degenerate_indices_out: list | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
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
        Number of chunks to split the data into, by default 1.
        The discrete transition is expanded to the joint
        (n_state_bins, n_state_bins) transition one time step at a time inside
        the scan, so the (n_time, n_state_bins, n_state_bins) joint transition
        is never materialized; ``n_chunks=1`` is feasible even on long sessions.
    log_likelihoods : np.ndarray, optional
    cache_log_likelihoods : bool, optional
        If True, log likelihoods are cached instead of recomputed for each chunk, by default True
    dtype : jnp.dtype, optional
        Data type for computations (jnp.float32 or jnp.float64), by default jnp.float32.
        Use float64 for numerically challenging problems.
        Note: Requires JAX_ENABLE_X64=1 environment variable for float64.
    degenerate_indices_out : list | None, optional
        If provided, the global time indices of all-``-inf`` (degenerate)
        timesteps are appended to this list during the forward pass. This is
        the only way to recover those indices when likelihoods are not cached
        (the returned ``log_likelihoods`` is then ``None``). By default None.

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
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        One-step-ahead predicted state probabilities over state bins
    """
    causal_posterior = []
    predictive_posterior = []
    predictive_state_probabilities = []
    causal_state_probabilities = []
    acausal_posterior = []
    acausal_state_probabilities = []
    marginal_likelihood = 0.0
    n_degenerate_total = 0
    n_nan_total = 0

    n_time = len(time)
    # Validate n_chunks: ensure it doesn't exceed n_time to avoid empty chunks
    if n_chunks > n_time:
        raise ValueError(
            f"n_chunks ({n_chunks}) cannot exceed n_time ({n_time}). "
            f"Each chunk must contain at least one time point."
        )
    time_chunks = np.array_split(np.arange(n_time), n_chunks)

    # Convert inputs to JAX arrays once at the top - keep data on device
    # Cast to desired dtype for precision control
    state_ind_jax = jnp.asarray(state_ind)  # Keep as integer
    initial_distribution_jax = jnp.asarray(initial_distribution, dtype=dtype)
    continuous_transition_matrix_jax = jnp.asarray(
        continuous_transition_matrix, dtype=dtype
    )
    # Convert discrete transition matrix but don't pre-index to avoid T×N×N materialization
    # Index per-chunk instead (see forward/backward loops)
    discrete_transition_matrix_jax = jnp.asarray(
        discrete_transition_matrix, dtype=dtype
    )

    # Create aggregation matrix for marginalizing state bins -> discrete states
    # This is reused across all chunks
    n_discrete_states = int(jnp.max(state_ind_jax)) + 1
    state_aggregation_matrix = jax.nn.one_hot(
        state_ind_jax, n_discrete_states
    )  # (n_state_bins, n_states)

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

        # Tally degenerate (all -inf) and NaN timesteps at one host sync point
        # before the array is donated to the JIT call. Accumulated across chunks
        # and reported once after the forward pass.
        n_degen, n_nan = _accumulate_chunk_degeneracy(
            log_likelihood_chunk, time_inds_np, degenerate_indices_out
        )
        n_degenerate_total += n_degen
        n_nan_total += n_nan

        # Time-slice only: the scan body indexes states per step, so the joint
        # (chunk, n_state_bins, n_state_bins) transition is never materialized.
        dtm_chunk = discrete_transition_matrix_jax[time_inds]

        # Donated: log_likelihood_chunk, dtm_chunk, initial_distribution
        # All are single-use per iteration - safe to donate
        # Note: Small arrays may trigger benign donation warnings in tests
        (
            (marginal_likelihood_chunk, predicted_probs_next),
            (causal_posterior_chunk, predicted_probs_chunk),
        ) = _filter_covariate_dependent_internal(
            initial_distribution=(
                initial_distribution_jax if chunk_id == 0 else predicted_probs_next
            ),
            discrete_transition_matrix=dtm_chunk,
            continuous_transition_matrix=continuous_transition_matrix_jax,
            state_ind=state_ind_jax,
            log_likelihoods=log_likelihood_chunk,
        )

        # Keep as JAX arrays - no conversion to NumPy yet
        # Marginalize state bins -> discrete states by summing probabilities
        causal_posterior.append(causal_posterior_chunk)
        causal_state_probabilities.append(
            causal_posterior_chunk @ state_aggregation_matrix
        )
        predictive_posterior.append(predicted_probs_chunk)
        predictive_state_probabilities.append(
            predicted_probs_chunk @ state_aggregation_matrix
        )

        marginal_likelihood += marginal_likelihood_chunk

    _warn_degenerate_and_nan_timesteps(n_degenerate_total, n_nan_total, n_time)

    # Concatenate JAX arrays on device
    causal_posterior_jax = jnp.concatenate(causal_posterior)
    causal_state_probabilities_jax = jnp.concatenate(causal_state_probabilities)
    predictive_posterior_jax = jnp.concatenate(predictive_posterior)
    predictive_state_probabilities_jax = jnp.concatenate(predictive_state_probabilities)

    # Backward pass: accumulate JAX arrays
    for chunk_id, time_inds_np in enumerate(reversed(time_chunks)):
        time_inds = jnp.asarray(time_inds_np)

        # Time-slice only; states indexed per step in the scan (see forward).
        dtm_chunk = discrete_transition_matrix_jax[time_inds]

        # Donated: dtm_chunk, filtered_probs slice, initial boundary
        # All are single-use per iteration
        # Note: Small arrays may trigger benign donation warnings in tests
        acausal_posterior_chunk = _smoother_covariate_dependent_internal(
            discrete_transition_matrix=dtm_chunk,
            continuous_transition_matrix=continuous_transition_matrix_jax,
            state_ind=state_ind_jax,
            filtered_probs=causal_posterior_jax[time_inds],
            initial=(
                causal_posterior_jax[-1]
                if chunk_id == 0
                else acausal_posterior_chunk[0]
            ),
            ind=time_inds,
            n_time=n_time,
        )
        acausal_posterior.append(acausal_posterior_chunk)
        acausal_state_probabilities.append(
            acausal_posterior_chunk @ state_aggregation_matrix
        )

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
        np.asarray(predictive_posterior_jax),
    )


def _viterbi_covariate_dependent_impl(
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

    # Run the backward pass
    # Index discrete transitions per-step to avoid materializing T×N_bins×N_bins array
    def _backward_pass(best_next_score, args):
        t, discrete_transition_matrix_t = args
        # Index states and compute log per-step (avoids T×N×N materialization)
        discrete_t_indexed = _index_discrete_transition(
            discrete_transition_matrix_t, state_ind
        )
        log_discrete_t = _safe_log(discrete_t_indexed)

        # Build log-space transition matrix: log(A * B) = log(A) + log(B)
        log_transition_matrix = log_continuous_transition_matrix + log_discrete_t
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

    return jnp.concatenate([first_state[None], states])


# Apply JIT without donation - tests reuse inputs
viterbi_covariate_dependent = jax.jit(_viterbi_covariate_dependent_impl)


def most_likely_sequence_covariate_dependent(
    time: np.ndarray,
    state_ind: np.ndarray,
    initial_distribution: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    log_likelihood_func: Callable[..., ArrayLike],
    log_likelihood_args: tuple,
    is_missing: np.ndarray | None = None,
    log_likelihoods: np.ndarray | None = None,
    n_chunks: int = 1,
    dtype: jnp.dtype = jnp.float32,
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
    dtype : jnp.dtype, optional
        Data type for computations (jnp.float32 or jnp.float64), by default jnp.float32.
        Note: Requires JAX_ENABLE_X64=1 environment variable for float64.

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

    # Warn if any time step is all-impossible (-inf) or contains a NaN, so a
    # Viterbi-only workflow gets the same diagnostics as the filter/smoother
    # drivers. Note ``argmax`` over an all-``-inf`` column silently returns
    # state 0, which this warning surfaces.
    _warn_if_degenerate_timesteps(log_likelihoods)

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

    most_likely_states = viterbi_covariate_dependent(
        initial_distribution=initial_distribution_jax,
        discrete_transition_matrix=discrete_transition_matrix_jax,
        continuous_transition_matrix=continuous_transition_matrix_jax,
        state_ind=state_ind_jax,
        log_likelihoods=log_likelihoods_jax,
    )

    # Return NumPy array for API consistency
    return np.asarray(most_likely_states), log_likelihoods


## Convergence check ##
def check_converged(
    log_likelihood: float,
    previous_log_likelihood: float,
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
        True if the relative change in log-likelihood is below ``tolerance``.
    is_increasing : bool
        True if the log-likelihood did not decrease by more than ``tolerance``.
        A decrease smaller than ``tolerance`` is treated as noise rather than a
        true monotonicity violation, matching the slack used by the final-E-step
        consistency check.

    """
    delta_log_likelihood = abs(log_likelihood - previous_log_likelihood)
    avg_log_likelihood = (
        abs(log_likelihood) + abs(previous_log_likelihood) + np.spacing(1)
    ) / 2

    # Compare the *relative* change against the tolerance (matching
    # ``is_converged`` below), not an absolute difference. The marginal
    # log-likelihood is a sum over timesteps, so its magnitude scales with
    # dataset size; an absolute slack would spuriously flag negligible
    # floating-point decreases on large datasets as monotonicity violations.
    relative_change = (log_likelihood - previous_log_likelihood) / avg_log_likelihood
    is_increasing = relative_change >= -tolerance
    is_converged = (delta_log_likelihood / avg_log_likelihood) < tolerance

    return is_converged, is_increasing
