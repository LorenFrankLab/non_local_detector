"""Sorted-spikes MRF-GAM (penalized-Poisson) encoding/decoding.

A second estimator on the same spectral engine as ``sorted_spikes_diffusion``, but
more principled statistically. It fits a penalized-Poisson generalized additive
model whose reduced-rank design matrix ``B`` is the ``rank`` smoothest eigenmodes of
the environment's graph Laplacian and whose quadratic penalty is those eigenvalues
``d``::

    n_ik ~ Poisson( o_i * exp((B gamma_k)_i) )
    penalty  (lambda / 2) * gamma_k^T diag(d) gamma_k

This is the *same operator* as the diffusion smoother viewed as a GAM: penalizing
``gamma^T diag(d) gamma`` in the eigenbasis is penalizing the diffusion energy
``g^T L g``. Occupancy ``o`` enters as a log-offset (exposure), never a denominator,
so low-occupancy bins produce finite rates. The smoothing parameter ``lambda`` is
chosen by REML (Wood 2011). The whole population is fit at once (the design matrix
``B`` and offset ``o`` are shared; only the spike counts ``n_k`` and coefficients
``gamma_k`` differ), vectorized over neurons.

Implementation. The Newton/IRLS fit and the REML score run in JAX (jit + GPU-ready):
the whole loop compiles to batched linear algebra over the neuron axis, so the fit
scales linearly in neurons and accelerates on a GPU. Computation is float32 (the
package regime -- ``core.py`` is float32-explicit, so no global x64 is enabled);
rate error versus a float64 reference is ~1e-7, negligible for a Poisson rate. The
eigenbasis is still built once on CPU with SciPy ``eigsh`` (no JAX sparse eig).
Cost is dominated by the reduced-rank ``rank`` (Hessian ~``rank**2``, solve
~``rank**3``), so pass a modest ``rank`` for large populations.

Relationship to mgcv (R). The REML objective and the penalized-IRLS fit (with
step-halving) mirror mgcv's ``gam.fit3``, and the Laplacian-eigenmode penalty is
mgcv's ``bs="mrf"`` Markov-random-field smoother. Two deliberate departures: (1) a
**single shared** ``lambda`` is selected for the whole population (pooled smoothing),
whereas mgcv fits each response separately with its own smoothing parameter -- pooling
regularizes sparse cells but can over/under-smooth a cell whose roughness differs from
the population; (2) the penalty is the **weighted** finite-difference Laplacian
(``w = 1/distance**2``, for grid-size-independent bandwidth), not mgcv's default
unweighted neighbour Laplacian (degree on the diagonal, ``-1`` off) -- in mgcv terms
this corresponds to a user-supplied ``xt$penalty``.

Place fields ``exp(eta)`` are stored FULL-GRID exactly like ``sorted_spikes_kde`` /
``sorted_spikes_diffusion``, so the Poisson prediction is shared with the diffusion
likelihood (:func:`predict_sorted_spikes_mrf_log_likelihood` is that function).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import lax

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import validate_weights
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    environment_graph,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    _assemble_place_fields,
    _validate_local_interpolation,
    pixellate_interior_fields,
    predict_sorted_spikes_diffusion_log_likelihood,
)

# The MRF place-field / Poisson-prediction contract is identical to the diffusion
# smoother's (both store FULL-GRID rate maps), so the prediction is shared.
predict_sorted_spikes_mrf_log_likelihood = (
    predict_sorted_spikes_diffusion_log_likelihood
)

# The Newton/IRLS fit and REML score run in JAX (jit + GPU-ready), batched over
# neurons. float32 is the package regime (core.py is float32-explicit, so no global
# x64 is needed); rate error vs the float64 reference is ~1e-7, well within tests.
_FIT_DTYPE = jnp.float32
# Clip the linear predictor before exponentiating to avoid overflow during IRLS.
_ETA_CLIP = 30.0
# Max Newton step-halvings per iteration (mgcv gam.fit3 style monotone-descent guard).
_MAX_STEP_HALVINGS = 30
# Ridge added to the Hessian diagonal for numerical stability.
_HESSIAN_JITTER = 1e-10
# float32-safe floors: the coefficient-step convergence tol and the step-halving
# "did the objective increase" threshold must sit above float32 rounding noise
# (~1e-7), or convergence never triggers and halving fires on noise.
_FIT_TOL_FLOOR = 1e-6
_DESCENT_TOL = 1e-5
# Search bounds for log(lambda) during REML selection.
_LOG_PENALTY_BOUNDS = (-8.0, 20.0)
# Default cap on the reduced-rank basis when ``rank`` is None: the full-rank fit
# forms and solves a dense per-neuron Hessian each Newton step, which is impractical
# on large grids, and REML tunes smoothness within the basis anyway. This cap is a
# performance choice of ours, not an mgcv default (mgcv fits full-rank by default).
_DEFAULT_MAX_RANK = 250


def _as_positive_int(name: str, value: int) -> int:
    """Validate positive integer controls used by the Newton solver."""
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a positive integer")

    try:
        int_value = int(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"{name} must be a positive integer") from err

    if int_value != value or int_value < 1:
        raise ValidationError(f"{name} must be a positive integer")

    return int_value


def _as_positive_float(name: str, value: float) -> float:
    """Validate positive floating-point controls."""
    try:
        float_value = float(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"{name} must be a positive finite value") from err

    if not np.isfinite(float_value) or float_value <= 0:
        raise ValidationError(f"{name} must be a positive finite value")

    return float_value


def _as_nonnegative_float(name: str, value: float) -> float:
    """Validate non-negative scalar model parameters."""
    try:
        float_value = float(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"{name} must be a non-negative finite value") from err

    if not np.isfinite(float_value) or float_value < 0:
        raise ValidationError(f"{name} must be a non-negative finite value")

    return float_value


def _validate_log_penalty_bounds(
    log_penalty_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Validate REML search bounds in log-penalty space."""
    try:
        lower, upper = tuple(float(bound) for bound in log_penalty_bounds)
    except (TypeError, ValueError) as err:
        raise ValidationError(
            "log_penalty_bounds must contain two finite values with lower < upper"
        ) from err

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValidationError(
            "log_penalty_bounds must contain two finite values with lower < upper"
        )

    return lower, upper


def _validate_mrf_problem(
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
    penalty: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    """Validate arrays and solver controls shared by MRF fit and REML."""
    counts = np.asarray(counts, dtype=float)
    occupancy = np.asarray(occupancy, dtype=float)
    basis = np.asarray(basis, dtype=float)
    penalty_weights = np.asarray(penalty_weights, dtype=float)
    penalty = _as_nonnegative_float("penalty", penalty)
    max_iter = _as_positive_int("max_iter", max_iter)
    tol = _as_positive_float("tol", tol)

    if counts.ndim != 2:
        raise ValidationError("counts must be a 2D array of shape (n_bins, n_neurons)")

    if occupancy.ndim != 1:
        raise ValidationError("occupancy must be a 1D array")

    if basis.ndim != 2:
        raise ValidationError("basis must be a 2D array")

    if penalty_weights.ndim != 1:
        raise ValidationError("penalty_weights must be a 1D array")

    if occupancy.shape[0] != counts.shape[0]:
        raise ValidationError(
            "occupancy and counts must have the same number of spatial bins"
        )

    if basis.shape[0] != counts.shape[0]:
        raise ValidationError(
            "basis and counts must have the same number of spatial bins"
        )

    if penalty_weights.shape[0] != basis.shape[1]:
        raise ValidationError(
            "penalty_weights must have one entry for each basis vector"
        )

    if not np.all(np.isfinite(counts)):
        raise ValidationError("counts must contain only finite values")

    if not np.all(np.isfinite(occupancy)):
        raise ValidationError("occupancy must contain only finite values")

    if not np.all(np.isfinite(basis)):
        raise ValidationError("basis must contain only finite values")

    if not np.all(np.isfinite(penalty_weights)):
        raise ValidationError("penalty_weights must contain only finite values")

    if np.any(counts < 0):
        raise ValidationError("counts must be non-negative")

    if np.any(occupancy < 0):
        raise ValidationError("occupancy must be non-negative")

    if np.any(penalty_weights < 0):
        raise ValidationError("penalty_weights must be non-negative")

    return counts, occupancy, basis, penalty_weights, penalty, max_iter, tol


def _penalized_hessian(basis: jnp.ndarray, mu: jnp.ndarray, penalty_diag: jnp.ndarray):
    """Batched penalized Hessian ``Bᵀ diag(mu_k) B + diag(penalty_diag)`` per neuron.

    The 3-operand einsum is compiled by XLA to an efficient batched matmul (unlike
    NumPy's einsum path). Returns shape ``(n_neurons, rank, rank)``.
    """
    hessian = jnp.einsum("br,bk,bs->krs", basis, mu, basis)
    return hessian + jnp.eye(basis.shape[1], dtype=basis.dtype) * (
        penalty_diag + _HESSIAN_JITTER
    )


@partial(jax.jit, static_argnames=("max_iter",))
def _newton_fit_jax(
    counts: jnp.ndarray,
    occupancy: jnp.ndarray,
    basis: jnp.ndarray,
    penalty_diag: jnp.ndarray,
    max_iter: int,
    tol: jnp.ndarray,
):
    """Batched penalized-Poisson Newton/IRLS with per-neuron step-halving (jit).

    All inputs are ``_FIT_DTYPE`` and the whole loop compiles to batched linear
    algebra (GPU-ready). Returns ``(coeffs, eta, mu, n_iter, max_step, converged)``.
    """
    n_bins = basis.shape[0]
    n_neurons = counts.shape[1]
    tol = jnp.maximum(tol, _FIT_TOL_FLOOR)  # float32-safe convergence floor

    def penalized_neg_loglik(coeffs, eta, mu):
        loglik = jnp.sum(counts * eta - mu, axis=0)  # (n_neurons,)
        penalty_term = 0.5 * jnp.sum(penalty_diag[:, None] * coeffs**2, axis=0)
        return -loglik + penalty_term

    # Warm start each neuron from a constant log-rate (fast, convex problem).
    total_occupancy = jnp.maximum(occupancy.sum(), 1e-9)
    eta0 = jnp.log(jnp.clip(counts.sum(0) / total_occupancy, 1e-6, None))
    basis_pinv_ones = jnp.linalg.lstsq(basis, jnp.ones(n_bins, basis.dtype))[0]
    coeffs0 = basis_pinv_ones[:, None] * eta0[None, :]  # (rank, n_neurons)

    def newton_cond(state):
        _, iteration, max_step = state
        return (iteration < max_iter) & (max_step >= tol)

    def newton_body(state):
        coeffs, iteration, _ = state
        eta = basis @ coeffs
        mu = occupancy[:, None] * jnp.exp(jnp.clip(eta, -_ETA_CLIP, _ETA_CLIP))
        grad = basis.T @ (counts - mu) - penalty_diag[:, None] * coeffs
        hessian = _penalized_hessian(basis, mu, penalty_diag)
        step = jnp.linalg.solve(hessian, grad.T[..., None])[..., 0]  # (n_neurons, rank)

        objective = penalized_neg_loglik(coeffs, eta, mu)

        def is_worse(scale):
            trial = coeffs + scale[None, :] * step.T
            trial_eta = basis @ trial
            trial_mu = occupancy[:, None] * jnp.exp(
                jnp.clip(trial_eta, -_ETA_CLIP, _ETA_CLIP)
            )
            trial_objective = penalized_neg_loglik(trial, trial_eta, trial_mu)
            # NaN or a non-noise increase (float32-safe threshold) -> halve.
            return ~(
                trial_objective <= objective + _DESCENT_TOL * (1.0 + jnp.abs(objective))
            )

        def halving_cond(hstate):
            scale, halvings = hstate
            return (halvings < _MAX_STEP_HALVINGS) & jnp.any(is_worse(scale))

        def halving_body(hstate):
            scale, halvings = hstate
            return jnp.where(is_worse(scale), 0.5 * scale, scale), halvings + 1

        scale, _ = lax.while_loop(
            halving_cond, halving_body, (jnp.ones(n_neurons, basis.dtype), 0)
        )
        accepted_step = scale[None, :] * step.T
        # initial=0.0 so an empty neuron axis (n_neurons == 0) reduces to 0 (converged)
        # rather than raising -- keeps the public REML helpers safe on zero neurons.
        max_step = jnp.max(jnp.abs(accepted_step), initial=0.0)
        return coeffs + accepted_step, iteration + 1, max_step

    coeffs, n_iter, max_step = lax.while_loop(
        newton_cond,
        newton_body,
        (coeffs0, jnp.array(0), jnp.array(jnp.inf, basis.dtype)),
    )
    # Clip eta so any rate the caller derives via exp(eta) stays finite even if a
    # low-penalty / near-zero-occupancy fit drove eta large (mirrors mgcv).
    eta = jnp.clip(basis @ coeffs, -_ETA_CLIP, _ETA_CLIP)
    mu = occupancy[:, None] * jnp.exp(eta)
    return coeffs, eta, mu, n_iter, max_step, max_step < tol


def mrf_penalized_poisson_fit(
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
    penalty: float,
    max_iter: int = 100,
    tol: float = 1e-10,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | bool]]:
    """Fit the population penalized-Poisson GAM by vectorized Newton/IRLS.

    Fits, for every neuron ``k`` at once,
    ``counts[:, k] ~ Poisson(occupancy * exp(basis @ coeffs[:, k]))`` with penalty
    ``penalty * coeffs[:, k]^T diag(penalty_weights) coeffs[:, k]``. The design
    matrix, occupancy offset, penalty, and penalty weights are shared across
    neurons; only the response and coefficients differ, so the Newton step is
    batched over the neuron axis (no per-neuron Python loop).

    Parameters
    ----------
    counts : np.ndarray, shape (n_bins, n_neurons)
        Spike counts per interior bin per neuron.
    occupancy : np.ndarray, shape (n_bins,)
        Occupancy (exposure) per interior bin, shared across neurons.
    basis : np.ndarray, shape (n_bins, rank)
        Reduced-rank design matrix (the smoothest Laplacian eigenmodes).
    penalty_weights : np.ndarray, shape (rank,)
        Per-mode penalty weights (the Laplacian eigenvalues).
    penalty : float
        The smoothing parameter ``lambda``.
    max_iter : int, optional
        Maximum Newton iterations, by default 100.
    tol : float, optional
        Convergence tolerance on the max coefficient step, by default 1e-10.
    validate : bool, optional
        Validate inputs via :func:`_validate_mrf_problem`, by default True. The REML
        search passes False to skip re-validation on every candidate fit (the caller
        has already validated the shared, loop-invariant arrays once).

    Returns
    -------
    coeffs : np.ndarray, shape (rank, n_neurons)
        Fitted coefficients.
    eta : np.ndarray, shape (n_bins, n_neurons)
        Linear predictor ``basis @ coeffs``.
    mu : np.ndarray, shape (n_bins, n_neurons)
        Fitted mean ``occupancy[:, None] * exp(eta)``.
    diagnostics : dict
        Solver diagnostics with ``n_iter``, ``converged``, and ``max_step``.
    """
    if validate:
        counts, occupancy, basis, penalty_weights, penalty, max_iter, tol = (
            _validate_mrf_problem(
                counts, occupancy, basis, penalty_weights, penalty, max_iter, tol
            )
        )
    n_bins, rank = basis.shape
    n_neurons = counts.shape[1]
    if n_neurons == 0:
        # No neurons to fit: jnp reductions over the empty neuron axis are ill-defined,
        # so short-circuit with empty arrays consistent with the batched contract.
        empty = (np.zeros((rank, 0)), np.zeros((n_bins, 0)), np.zeros((n_bins, 0)))
        return (*empty, {"n_iter": 0, "converged": True, "max_step": 0.0})

    coeffs, eta, mu, n_iter, max_step, converged = _newton_fit_jax(
        jnp.asarray(counts, _FIT_DTYPE),
        jnp.asarray(occupancy, _FIT_DTYPE),
        jnp.asarray(basis, _FIT_DTYPE),
        jnp.asarray(penalty * penalty_weights, _FIT_DTYPE),  # penalty_diag
        int(max_iter),
        _FIT_DTYPE(tol),
    )
    diagnostics = {
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "max_step": float(max_step),
    }
    return np.asarray(coeffs), np.asarray(eta), np.asarray(mu), diagnostics


def _penalty_rank(penalty_weights: np.ndarray) -> int:
    """Number of penalized directions (positive eigenvalues) for the REML df term."""
    largest = penalty_weights.max()
    if largest <= 0:
        return 0
    return int(np.sum(penalty_weights > 1e-12 * largest))


@partial(jax.jit, static_argnames=("max_iter",))
def _reml_score_jax(
    log_penalty: jnp.ndarray,
    counts: jnp.ndarray,
    occupancy: jnp.ndarray,
    basis: jnp.ndarray,
    penalty_weights: jnp.ndarray,
    penalty_rank: jnp.ndarray,
    max_iter: int,
    tol: jnp.ndarray,
) -> jnp.ndarray:
    """Negative Laplace REML for ``lambda = exp(log_penalty)``, summed over neurons.

    On-device scalar (the fit stays on device across REML candidates). Returns +inf
    for any lambda whose per-neuron Hessian is not positive-definite, so the search
    never optimizes over an invalid log-determinant.
    """
    penalty = jnp.exp(log_penalty)
    penalty_diag = penalty * penalty_weights
    coeffs, eta, mu, _, _, _ = _newton_fit_jax(
        counts, occupancy, basis, penalty_diag, max_iter, tol
    )
    loglik = jnp.sum(counts * eta - mu, axis=0)
    penalty_term = 0.5 * penalty * jnp.sum(penalty_weights[:, None] * coeffs**2, axis=0)
    # log|H| via Cholesky: logdet = 2 * sum(log(diag(L))), more float32-stable than the
    # LU-based slogdet (and the SPD structure is exact here). A non-positive-definite
    # Hessian yields a NaN Cholesky, so the finiteness check both validates and rejects.
    chol = jnp.linalg.cholesky(_penalized_hessian(basis, mu, penalty_diag))
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    reml = -loglik + penalty_term - 0.5 * penalty_rank * log_penalty + 0.5 * logdet
    return jnp.where(jnp.all(jnp.isfinite(logdet)), jnp.sum(reml), jnp.inf)


def mrf_reml_objective(
    log_penalty: float,
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
    validate: bool = True,
) -> float:
    """Negative Laplace REML (Wood 2011) for a single shared ``lambda``, all neurons.

    The population shares ``lambda``, so the restricted marginal likelihood factorizes
    over neurons and the objective is the sum of the per-neuron terms
    ``-loglik + penalty - 0.5 * penalty_rank * log(lambda) + 0.5 * logdet(H)``.
    Minimized over ``log_penalty``. Pass ``validate=False`` to skip input validation
    when the caller (the REML search) has already validated the shared arrays.
    """
    log_penalty = float(log_penalty)
    if not np.isfinite(log_penalty):
        return np.inf
    if validate:
        counts, occupancy, basis, penalty_weights, _, max_iter, tol = (
            _validate_mrf_problem(
                counts,
                occupancy,
                basis,
                penalty_weights,
                np.exp(log_penalty),
                max_iter,
                tol,
            )
        )
    score = _reml_score_jax(
        _FIT_DTYPE(log_penalty),
        jnp.asarray(counts, _FIT_DTYPE),
        jnp.asarray(occupancy, _FIT_DTYPE),
        jnp.asarray(basis, _FIT_DTYPE),
        jnp.asarray(penalty_weights, _FIT_DTYPE),
        _FIT_DTYPE(_penalty_rank(penalty_weights)),
        int(max_iter),
        _FIT_DTYPE(tol),
    )
    return float(score)


def select_penalty_by_reml(
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
    log_penalty_bounds: tuple[float, float] = _LOG_PENALTY_BOUNDS,
    reml_xatol: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """Select a single shared smoothing parameter ``lambda`` by REML.

    Minimizes :func:`mrf_reml_objective` over ``log(lambda)`` on a bounded interval
    (deterministic; no random state). Returns ``(lambda, reml_objective)``.
    """
    log_penalty_bounds = _validate_log_penalty_bounds(log_penalty_bounds)
    reml_xatol = _as_positive_float("reml_xatol", reml_xatol)
    # Validate the problem arrays and solver controls once here: the bounded optimizer
    # evaluates the objective many times on the same shared, loop-invariant inputs, so
    # re-validating inside every candidate fit is pure overhead. The placeholder
    # penalty (1.0) is only for this validation; each candidate supplies its own
    # penalty via exp(log_penalty).
    counts, occupancy, basis, penalty_weights, _, max_iter, tol = _validate_mrf_problem(
        counts, occupancy, basis, penalty_weights, 1.0, max_iter, tol
    )

    # Move the loop-invariant arrays to the device once, not on every objective call:
    # only log(lambda) varies across the bounded search, so re-sending the (n_bins, rank)
    # basis and the count/occupancy arrays each evaluation is wasted host->device
    # transfer (undercuts "stays on device" on GPU). The scalar objective then calls the
    # jitted REML score directly with these device arrays.
    counts_dev = jnp.asarray(counts, _FIT_DTYPE)
    occupancy_dev = jnp.asarray(occupancy, _FIT_DTYPE)
    basis_dev = jnp.asarray(basis, _FIT_DTYPE)
    penalty_weights_dev = jnp.asarray(penalty_weights, _FIT_DTYPE)
    penalty_rank = _FIT_DTYPE(_penalty_rank(penalty_weights))
    tol_dev = _FIT_DTYPE(tol)
    max_iter = int(max_iter)

    def objective(log_penalty: float) -> float:
        log_penalty = float(log_penalty)
        if not np.isfinite(log_penalty):
            return np.inf
        score = _reml_score_jax(
            _FIT_DTYPE(log_penalty),
            counts_dev,
            occupancy_dev,
            basis_dev,
            penalty_weights_dev,
            penalty_rank,
            max_iter,
            tol_dev,
        )
        return float(score)

    result = scipy.optimize.minimize_scalar(
        objective,
        bounds=log_penalty_bounds,
        method="bounded",
        options={"xatol": reml_xatol},
    )
    # The objective returns +inf for any lambda whose per-neuron Hessian is not
    # positive-definite. If no candidate had a finite objective, minimize_scalar
    # still returns an arbitrary point; reject it rather than fitting with a
    # meaningless penalty.
    if not result.success or not np.isfinite(result.fun):
        raise ValidationError(
            "REML failed to find a valid smoothing parameter",
            expected="a finite REML objective for some lambda in the search interval",
            got=str(result.message)
            if not result.success
            else "a non-positive-definite Hessian at every candidate lambda",
            hint=(
                "The reduced-rank basis is too large relative to the data, or too "
                "many interior bins have zero occupancy. Reduce `rank`, or provide a "
                "denser/longer training trajectory."
            ),
        )
    return float(np.exp(result.x)), float(result.fun)


def fit_sorted_spikes_mrf_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    weights: np.ndarray | None = None,
    sampling_frequency: int = 500,
    rank: int | None = None,
    penalty: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    log_penalty_bounds: tuple[float, float] = _LOG_PENALTY_BOUNDS,
    reml_xatol: float = 1e-3,
    block_size: int = 100,
    local_interpolation: str = "linear",
    disable_progress_bar: bool = False,
) -> dict:
    """Fit a population MRF-GAM encoding model for sorted spikes.

    Pixellates spike counts and occupancy onto interior bins, uses the ``rank``
    smoothest Laplacian eigenmodes (from the cached engine) as the reduced-rank
    basis with their eigenvalues as the penalty, and fits the whole population's
    penalized-Poisson GAM at once with occupancy as an exposure offset. The
    smoothing parameter ``lambda`` is chosen by REML unless ``penalty`` is given.
    Place fields ``exp(eta)`` are stored FULL-GRID, so the encoding-dict contract
    and Poisson prediction are identical to ``sorted_spikes_diffusion``.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
    position : np.ndarray, shape (n_time_position, n_position_dims)
    spike_times : list[np.ndarray]
        Spike times for each neuron.
    environment : Environment
        The spatial environment (must be fitted).
    weights : np.ndarray, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM). If None,
        uniform weights are used.
    sampling_frequency : int, optional
        Accepted for signature compatibility; not used by the MRF fit.
    rank : int or None, optional
        Number of smoothest eigenmodes used as the basis. None (default) caps at
        ``min(n_interior_bins, 250)`` -- a performance bound on the dense per-neuron
        Hessian, with REML tuning smoothness within the basis. Note this differs from
        mgcv's ``bs="mrf"`` default, which uses the full rank (all regions) and only
        truncates when ``k`` is set below the region count; the 250 cap silently drops
        the highest-frequency modes on large grids (pass an explicit ``rank`` to
        override). Must be an explicit parameter so a user-supplied ``rank`` is not
        dropped by the base class's signature filter.
    penalty : float or None, optional
        The smoothing parameter ``lambda``. None (default) selects it by REML. Pass
        ``0.0`` for an unpenalized (saturated) fit.
    max_iter : int, optional
        Maximum Newton iterations for both REML candidate fits and the final fit.
    tol : float, optional
        Convergence tolerance on the max coefficient step.
    log_penalty_bounds : tuple[float, float], optional
        REML search interval for ``log(lambda)`` when ``penalty`` is None.
    reml_xatol : float, optional
        Scalar optimizer tolerance for REML penalty selection.
    block_size : int, optional
        Accepted for signature compatibility with sorted-spikes likelihood defaults;
        unused by the MRF fit.
    local_interpolation : {"linear", "nearest"}, optional
        How local likelihood evaluates full-grid rate maps at the animal's position.
        ``"linear"`` interpolates within connected interior stencils and falls back
        to nearest-bin lookup otherwise. ``"nearest"`` preserves the historical
        bin lookup.
    disable_progress_bar : bool, optional

    Returns
    -------
    encoding_model : dict
        The same keys as ``sorted_spikes_diffusion`` (``environment``, ``occupancy``,
        ``mean_rates``, ``place_fields`` [FULL-GRID], ``no_spike_part_log_likelihood``,
        ``is_track_interior``, ``node_order``, ``bin_sizes``,
        ``local_interpolation``, ``disable_progress_bar``), plus MRF diagnostics
        (``mrf_penalty``, ``mrf_rank``, ``mrf_coefficients``,
        ``mrf_penalty_weights``, ``mrf_reml_objective``, ``mrf_n_iter``,
        ``mrf_converged``, ``mrf_max_step``, ``mrf_log_penalty_bounds``,
        ``mrf_penalty_selected_by_reml``). Note
        ``occupancy`` here is the raw exposure field (weighted interior-bin counts),
        NOT the integral-one density ``sorted_spikes_diffusion`` stores; it is carried
        only for contract parity and is unused in prediction.
    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = validate_weights(weights, position.shape[0])
    max_iter = _as_positive_int("max_iter", max_iter)
    tol = _as_positive_float("tol", tol)
    log_penalty_bounds = _validate_log_penalty_bounds(log_penalty_bounds)
    reml_xatol = _as_positive_float("reml_xatol", reml_xatol)
    local_interpolation = _validate_local_interpolation(local_interpolation)
    if penalty is not None:
        penalty = _as_nonnegative_float("penalty", penalty)
    if rank is not None:
        rank = _as_positive_int("rank", rank)

    _, node_order, bin_sizes = environment_graph(environment)
    # Cap the default basis size to bound the dense per-neuron Hessian cost (a
    # performance choice; mgcv's mrf default is full rank -- see the `rank` docstring).
    requested_rank = (
        rank if rank is not None else min(node_order.shape[0], _DEFAULT_MAX_RANK)
    )
    penalty_weights, basis = cached_eigenbasis(environment, requested_rank)
    # cached_eigenbasis caps at the number of available modes, so the true basis rank
    # can be below the request; report what was actually used, not what was asked for.
    effective_rank = basis.shape[1]

    assert environment.is_track_interior_ is not None
    is_track_interior = environment.is_track_interior_.ravel()
    n_total_bins = is_track_interior.shape[0]

    occupancy_field, spike_fields, mean_rates = pixellate_interior_fields(
        position_time,
        position,
        spike_times,
        environment,
        node_order,
        weights,
        disable_progress_bar,
    )
    if spike_fields:
        counts = np.stack(spike_fields, axis=1)  # (n_interior, n_neurons)
    else:
        counts = np.zeros((node_order.shape[0], 0))

    penalty_selected_by_reml = penalty is None and counts.shape[1] > 0
    if penalty_selected_by_reml:
        penalty, reml_objective = select_penalty_by_reml(
            counts,
            occupancy_field,
            basis,
            penalty_weights,
            log_penalty_bounds=log_penalty_bounds,
            reml_xatol=reml_xatol,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        reml_objective = np.nan

    # `penalty is None` only survives here for the zero-neuron case; a caller's
    # explicit penalty=0.0 (unpenalized) must be respected, not treated as falsy.
    fit_penalty = penalty if penalty is not None else 1.0
    coeffs, eta, _, diagnostics = mrf_penalized_poisson_fit(
        counts,
        occupancy_field,
        basis,
        penalty_weights,
        fit_penalty,
        max_iter=max_iter,
        tol=tol,
    )
    rate_interior = np.exp(eta)  # (n_interior, n_neurons); eta is clipped in the fit
    place_fields, no_spike_part_log_likelihood = _assemble_place_fields(
        rate_interior, node_order, n_total_bins
    )

    return {
        "environment": environment,
        "occupancy": occupancy_field,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "node_order": node_order,
        "bin_sizes": bin_sizes,
        "local_interpolation": local_interpolation,
        "disable_progress_bar": disable_progress_bar,
        "mrf_penalty": fit_penalty,
        "mrf_rank": effective_rank,
        "mrf_coefficients": coeffs,
        "mrf_penalty_weights": penalty_weights,
        "mrf_reml_objective": reml_objective,
        "mrf_n_iter": diagnostics["n_iter"],
        "mrf_converged": diagnostics["converged"],
        "mrf_max_step": diagnostics["max_step"],
        "mrf_log_penalty_bounds": log_penalty_bounds,
        "mrf_penalty_selected_by_reml": penalty_selected_by_reml,
    }
