"""JAX-based Gaussian Mixture Model implementation for clusterless neural decoding.

This module provides a high-performance implementation of Gaussian Mixture Models
using JAX for automatic differentiation and GPU acceleration. It's specifically
designed for modeling clusterless (continuous) spike waveform features in neural
decoding applications.

The main class GaussianMixtureModel follows scikit-learn conventions while
leveraging JAX for efficient computation on both CPU and GPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------
Array = jnp.ndarray
Params = tuple[Array, Array, Array]  # (weights, means, covariances)
CovType = Literal["full", "tied", "diag", "spherical"]


def _has_negative_or_nonfinite(arr: Array) -> bool:
    """True if ``arr`` has any NaN/Inf or negative entry (as a concrete Python bool).

    Used for initial mixture weights: a NaN/negative weight becomes
    ``jnp.log(w) = NaN`` in EM and is otherwise misreported as a singular covariance.
    """
    return not bool(jnp.all(jnp.isfinite(arr))) or bool(jnp.any(arr < 0))


# ---------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------
def _estimate_gaussian_covariances_full(
    resp: Array,
    X: Array,
    nk: Array,
    means: Array,
    reg_covar: float,
) -> Array:
    """
    Estimate full covariance matrices.

    Parameters
    ----------
    resp : Array, shape (n_samples, n_components)
        Responsibilities (already multiplied by sample weights if provided).
    X : Array, shape (n_samples, n_features)
        Input data.
    nk : Array, shape (n_components,)
        Effective component counts (sum of responsibilities per component).
    means : Array, shape (n_components, n_features)
        Component means.
    reg_covar : float
        Diagonal regularization added to each covariance.

    Returns
    -------
    covariances : Array, shape (n_components, n_features, n_features)
        Full covariance matrices for each component.
    """
    _, n_features = means.shape

    # Map over components sequentially so only one (N, D) centered-data block is
    # live at a time. A vmap here would batch the subtraction and force XLA to
    # materialize the full (K, N, D) array; lax.map keeps the peak at (N, D).
    # The centered form (X - mu) is retained (not the second-moment identity)
    # to preserve numerical stability.
    def per_component(args: tuple[Array, Array, Array]) -> Array:
        mu, r, n = args
        d = X - mu  # (N, D)
        return ((d.T * r) @ d) / n

    covariances = jax.lax.map(per_component, (means, resp.T, nk))
    # Symmetrize covariances for numerical stability (ensure exact symmetry)
    covariances = (covariances + jnp.swapaxes(covariances, -2, -1)) * 0.5
    covariances += jnp.eye(n_features, dtype=X.dtype) * reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(
    resp: Array,
    X: Array,
    nk: Array,
    means: Array,
    reg_covar: float,
) -> Array:
    """
    Estimate a single tied covariance matrix shared by all components.

    Parameters
    ----------
    resp : Array, shape (n_samples, n_components)
        Responsibilities (already multiplied by sample weights if provided).
    X : Array, shape (n_samples, n_features)
        Input data.
    nk : Array, shape (n_components,)
        Effective component counts (unused directly here but included for parity).
    means : Array, shape (n_components, n_features)
        Component means.
    reg_covar : float
        Diagonal regularization added to the tied covariance.

    Returns
    -------
    covariance : Array, shape (n_features, n_features)
        Tied covariance matrix.
    """
    _, n_features = X.shape
    sum_w = resp.sum()

    def add_component(k: Array, covariance_sum: Array) -> Array:
        centered = X - means[k]
        return covariance_sum + (centered.T * resp[:, k]) @ centered

    # Center before summing: the second-moment form E[xx^T] - mu mu^T cancels
    # catastrophically when |mu| >> sigma. The loop keeps the workspace at
    # O(unroll * N * D) (a centered block and its weighted copy per unrolled
    # step) plus a (D, D) accumulator, instead of a (K, N, D) batch.
    # A small unroll lets XLA overlap the per-component matmuls, with
    # bitwise-identical output: measured 1.35x faster at D=8 and 1.09x at D=32
    # (N=100k, K=32, float32 CPU). The scratch buffer scales exactly with the
    # unroll factor (and linearly with N), so the factor is capped at 2 -- twice
    # the serial loop's O(N D) rather than the 8x that `unroll=8` costs and the
    # Kx that `unroll=True` costs. At N=1M, D=32 that is the difference between
    # 512 MB and 2 GB of transient, which decides whether a large clusterless
    # fit survives on GPU; the speed difference (1.16x vs 1.24x there) does not.
    covariance_sum: Array = jax.lax.fori_loop(
        0,
        means.shape[0],
        add_component,
        jnp.zeros((n_features, n_features), dtype=jnp.result_type(X, means, resp)),
        unroll=2,
    )

    # Guard division by zero with a tiny floor rather than 1.0: for legitimate
    # fractional sample weights whose total is < 1, dividing by 1.0 would bias
    # the covariance downward; eps only intervenes when there is effectively no
    # weight at all.
    eps = 10 * jnp.finfo(X.dtype).eps
    covariance = covariance_sum / jnp.maximum(sum_w, eps)
    # Symmetrize covariance for numerical stability (ensure exact symmetry)
    covariance = (covariance + covariance.T) * 0.5
    covariance += jnp.eye(n_features, dtype=X.dtype) * reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(
    resp: Array,
    X: Array,
    nk: Array,
    means: Array,
    reg_covar: float,
) -> Array:
    """
    Estimate diagonal covariance vectors.

    Parameters
    ----------
    resp : Array, shape (n_samples, n_components)
        Responsibilities (already multiplied by sample weights if provided).
    X : Array, shape (n_samples, n_features)
        Input data.
    nk : Array, shape (n_components,)
        Effective component counts.
    means : Array, shape (n_components, n_features)
        Component means.
    reg_covar : float
        Diagonal regularization added to each diagonal.

    Returns
    -------
    covariances : Array, shape (n_components, n_features)
        Diagonal covariance for each component.
    """

    _, n_features = X.shape

    def add_dimension(d: Array, second_moment: Array) -> Array:
        centered = X[:, d, None] - means[None, :, d]  # (N, K)
        return second_moment.at[:, d].set(jnp.sum(resp * centered * centered, axis=0))

    # Center per component before squaring: the second-moment form
    # E[x^2] - mu^2 cancels catastrophically when |mu| >> sigma. Looping over
    # features (rather than components) writes straight into the required
    # (K, D) output and keeps the transient at O(unroll * N * K), so neither a
    # (K, N, D) batch nor a transposed copy of ``resp`` is ever materialized.
    # Note the transient is O(N K) per step, independent of D, where the
    # per-component map this replaces was O(N D). As in the tied loop the unroll
    # is capped at 2, since the scratch scales exactly with it: at unroll=2 the
    # buffer matches the old map's at D=32 and is 1.6x it at D=8 (N=100k, K=32,
    # float32 CPU: 26 MB vs 26/16 MB), while being 1.0-1.6x faster. unroll=1
    # would be cheaper still (13 MB) but is ~8% slower than the old map at D=32.
    second_moment: Array = jax.lax.fori_loop(
        0,
        n_features,
        add_dimension,
        jnp.zeros(means.shape, dtype=jnp.result_type(X, means, resp)),
        unroll=2,
    )
    return second_moment / nk[:, None] + reg_covar


def _estimate_gaussian_covariances_spherical(
    resp: Array,
    X: Array,
    nk: Array,
    means: Array,
    reg_covar: float,
) -> Array:
    """
    Estimate spherical variances (single scalar variance per component).

    Parameters
    ----------
    resp : Array, shape (n_samples, n_components)
        Responsibilities (already multiplied by sample weights if provided).
    X : Array, shape (n_samples, n_features)
        Input data.
    nk : Array, shape (n_components,)
        Effective component counts.
    means : Array, shape (n_components, n_features)
        Component means.
    reg_covar : float
        Variance floor added to each spherical variance.

    Returns
    -------
    variances : Array, shape (n_components,)
        Spherical variances per component.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(
        axis=1
    )


# ---------------------------------------------------------------------
# Precision Cholesky and log-probability utilities
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=("covariance_type",))
def _compute_precision_cholesky(
    covariances: Array,
    covariance_type: CovType,
) -> Array:
    """
    Compute Cholesky factors of precision matrices.

    For 'full'/'tied', we compute:
        L = chol(Sigma) (lower)
        chol(Precision) = solve_triangular(L, I, lower=True).T

    Parameters
    ----------
    covariances : Array
        For 'full': shape (n_components, n_features, n_features)
        For 'tied': shape (n_features, n_features)
        For 'diag': shape (n_components, n_features)
        For 'spherical': shape (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Covariance parameterization.

    Returns
    -------
    precisions_chol : Array
        For 'full': shape (n_components, n_features, n_features)
        For 'tied': shape (n_features, n_features)
        For 'diag': shape (n_components, n_features)
        For 'spherical': shape (n_components,)
    """

    def single_inv_chol(cov: Array) -> Array:
        cov_chol = jnp.linalg.cholesky(cov)
        ident = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        return jax.scipy.linalg.solve_triangular(cov_chol, ident, lower=True).T

    if covariance_type == "full":
        return jax.vmap(single_inv_chol)(covariances)
    if covariance_type == "tied":
        return single_inv_chol(covariances)
    # diag / spherical
    # Clip to a positive floor so the reciprocal-sqrt precision stays finite
    # inside JIT (a Python `if`/raise on a traced value is not allowed here).
    # NOTE: this deliberately diverges from sklearn, which *raises* when a
    # variance is <= 0. Here a collapsed component (variance clipped up from
    # <=0) yields a large-but-finite precision (~1e5) instead of an error, so
    # the diag/spherical paths stay robust; increase reg_covar if a component
    # is genuinely collapsing. The full/tied paths surface singular covariances
    # as a NaN lower bound, which `fit` reports with an actionable error.
    covariances = jnp.clip(covariances, min=1e-10, max=None)
    return 1.0 / jnp.sqrt(covariances)


def _compute_log_det_cholesky(
    matrix_chol: Array,
    covariance_type: CovType,
    n_features: int,
) -> Array:
    """
    Compute log-determinant of precision matrices from their Cholesky factors.

    Parameters
    ----------
    matrix_chol : Array
        For 'full': shape (n_components, n_features, n_features)
        For 'tied': shape (n_features, n_features)
        For 'diag': shape (n_components, n_features)
        For 'spherical': shape (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Covariance parameterization.
    n_features : int
        Number of features (D).

    Returns
    -------
    log_det_precision : Array, shape (n_components,)
        Per-component log|Precision|.
        For 'tied', the scalar log-det is broadcast conceptually.
    """
    if covariance_type == "full":
        return jnp.sum(jnp.log(jnp.diagonal(matrix_chol, axis1=-2, axis2=-1)), axis=1)
    if covariance_type == "tied":
        return jnp.sum(jnp.log(jnp.diag(matrix_chol)))
    if covariance_type == "diag":
        return jnp.sum(jnp.log(matrix_chol), axis=1)
    # spherical
    return n_features * jnp.log(matrix_chol)


def _estimate_gaussian_parameters(
    X: Array,
    resp_unweighted: Array,
    reg_covar: float,
    covariance_type: CovType,
    sample_weight: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Compute (nk, means, covariances) given responsibilities and optional sample weights.

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
        Input data.
    resp_unweighted : Array, shape (n_samples, n_components)
        Responsibilities before applying per-sample weights.
    reg_covar : float
        Diagonal regularization added to covariance(s).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Covariance parameterization.
    sample_weight : Array | None, shape (n_samples,), default=None
        Nonnegative per-sample weights. If provided, effective responsibilities
        become r'[i,k] = sample_weight[i] * resp_unweighted[i,k].

    Returns
    -------
    nk : Array, shape (n_components,)
        Effective component counts after weighting.
    means : Array, shape (n_components, n_features)
        Component means.
    covariances : Array
        Covariances with shape depending on `covariance_type`:
            'full'      -> (n_components, n_features, n_features)
            'tied'      -> (n_features, n_features)
            'diag'      -> (n_components, n_features)
            'spherical' -> (n_components,)
    """
    if sample_weight is None:
        resp = resp_unweighted
    else:
        resp = resp_unweighted * sample_weight[:, None]

    eps = 10 * jnp.finfo(X.dtype).eps
    nk = resp.sum(axis=0) + eps  # (K,)
    means = (resp.T @ X) / nk[:, jnp.newaxis]  # (K, D)

    estimator = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type]
    covariances = estimator(resp, X, nk, means, reg_covar)
    return nk, means, covariances


@partial(jax.jit, static_argnames=("covariance_type",))
def _estimate_log_gaussian_prob(
    X: Array,
    means: Array,
    precisions_chol: Array,
    covariance_type: CovType,
) -> Array:
    """
    Log Gaussian probabilities for all samples/components.

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
    means : Array, shape (n_components, n_features)
    precisions_chol : Array
        For 'full': shape (n_components, n_features, n_features)
        For 'tied': shape (n_features, n_features)
        For 'diag': shape (n_components, n_features)
        For 'spherical': shape (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : Array, shape (n_samples, n_components)
    """
    if X.ndim != 2 or means.ndim != 2 or X.shape[1] != means.shape[1]:
        raise ValueError("X and means must be 2-D with matching feature dimensions")
    n_components, n_features = means.shape
    # Validate every parameterization here: 'full'/'tied' would otherwise fail
    # deep inside solve_triangular with a shape message that names neither the
    # covariance type nor the expected layout.
    expected_precision_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }[covariance_type]
    if tuple(precisions_chol.shape) != expected_precision_shape:
        raise ValueError(
            f"precisions_chol for covariance_type='{covariance_type}' must have "
            f"shape {expected_precision_shape}, got {tuple(precisions_chol.shape)}"
        )

    if covariance_type == "full":

        def comp_full(args: tuple[Array, Array]) -> Array:
            # Sequential over components so only one (N, D) block is live; a
            # vmap here would batch (X - mu) and materialize the full (K, N, D)
            # array. This is the E-step and runs on every fit iteration and
            # every predict/score call, so the peak matters most here.
            # Benchmarked: vmap is only faster on small data; at D=32, N=500k it is
            # ~3x slower (the multi-GB (K, N, D) intermediate) and OOMs on GPU. Keep
            # lax.map.
            mu, R = args
            Y = (X - mu) @ R  # (N, D)
            return jnp.sum(Y * Y, axis=1)  # (N,)

        maha = jax.lax.map(comp_full, (means, precisions_chol)).T
    elif covariance_type == "tied":

        def comp_tied(mu: Array) -> Array:
            # Sequential over components as in the 'full' branch above; a
            # batched (X[None] - means[:, None]) would materialize (K, N, D).
            Yk = (X - mu) @ precisions_chol  # (N, D)
            return jnp.sum(Yk * Yk, axis=1)  # (N,)

        maha = jax.lax.map(comp_tied, means).T
    else:  # diag / spherical
        # Center on each component's own mean before squaring: the matmul
        # expansion |x|^2 - 2 x.mu + |mu|^2 cancels catastrophically whenever
        # |mu| >> sigma, and centering on a shared point only removes a common
        # offset, not the separation between tight, distant components. XLA
        # fuses the broadcast into the minor-axis reduction, so no (N, K, D)
        # buffer is allocated (0 bytes of compiled scratch; N=100k, K=32, CPU).
        centered = X[:, None, :] - means[None, :, :]  # (N, K, D), fused
        if covariance_type == "diag":
            # precisions_chol holds per-feature inverse standard deviations.
            centered = centered * precisions_chol[None, :, :]
            maha = jnp.sum(centered * centered, axis=-1)  # (N, K)
        else:
            # One inverse standard deviation per component: scale after the
            # reduction (1.2-1.5x faster than inside it).
            maha = jnp.sum(centered * centered, axis=-1) * precisions_chol**2

    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)
    return -0.5 * (n_features * jnp.log(2.0 * jnp.pi) + maha) + log_det


def _estimate_weighted_log_prob(
    X: Array,
    means: Array,
    precisions_chol: Array,
    covariance_type: CovType,
    weights: Array,
) -> Array:
    """
    Weighted log-probabilities (log p(x|z=k) + log pi_k).

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
    means : Array, shape (n_components, n_features)
    precisions_chol : Array
        See `_estimate_log_gaussian_prob`.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    weights : Array, shape (n_components,)
        Mixture weights summing to 1.

    Returns
    -------
    weighted_log_prob : Array, shape (n_samples, n_components)
    """
    return _estimate_log_gaussian_prob(
        X, means, precisions_chol, covariance_type
    ) + jnp.log(weights)


@partial(jax.jit, static_argnames=("covariance_type",))
def _estimate_log_prob_resp(
    X: Array,
    means: Array,
    precisions_chol: Array,
    covariance_type: CovType,
    weights: Array,
) -> tuple[Array, Array]:
    """
    Compute per-sample log p(x) and log responsibilities.

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
    means : Array, shape (n_components, n_features)
    precisions_chol : Array
        See `_estimate_log_gaussian_prob`.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    weights : Array, shape (n_components,)

    Returns
    -------
    log_prob_norm : Array, shape (n_samples,)
        Log marginal likelihood per sample.
    log_responsibilities : Array, shape (n_samples, n_components)
        Log responsibilities per sample/component.
    """
    wlp = _estimate_weighted_log_prob(
        X, means, precisions_chol, covariance_type, weights
    )
    log_prob_norm = jax.nn.logsumexp(wlp, axis=1)
    log_resp = wlp - log_prob_norm[:, jnp.newaxis]
    return log_prob_norm, log_resp


# ---------------------------------------------------------------------
# EM helpers
# ---------------------------------------------------------------------
def _e_step_func(
    X: Array, params: Params, covariance_type: CovType
) -> tuple[Array, Array]:
    """
    E-step helper.

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
    params : tuple(weights, means, covariances)
        weights : Array, shape (n_components,)
        means : Array, shape (n_components, n_features)
        covariances : Array, shape depends on covariance_type
    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob_norm : Array, shape (n_samples,)
    log_resp : Array, shape (n_samples, n_components)
    """
    weights, means, covariances = params
    precisions_chol = _compute_precision_cholesky(covariances, covariance_type)
    return _estimate_log_prob_resp(X, means, precisions_chol, covariance_type, weights)


def _m_step_func(
    X: Array,
    log_resp: Array,
    reg_covar: float,
    covariance_type: CovType,
    sample_weight: Array | None = None,
) -> Params:
    """
    M-step helper.

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
    log_resp : Array, shape (n_samples, n_components)
        Log responsibilities.
    reg_covar : float
        Covariance regularization.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    sample_weight : Array | None, shape (n_samples,), default=None

    Returns
    -------
    params : tuple(weights, means, covariances)
        weights : Array, shape (n_components,)
        means : Array, shape (n_components, n_features)
        covariances : Array, shape depends on covariance_type
    """
    nk, means, covariances = _estimate_gaussian_parameters(
        X, jnp.exp(log_resp), reg_covar, covariance_type, sample_weight
    )
    # nk carries the small empty-component guard; dividing by the sum of the
    # guarded counts keeps the mixture weights summing to exactly one.
    weights = nk / jnp.sum(nk)
    return (weights, means, covariances)


@partial(jax.jit, static_argnames=("covariance_type",))
def _em_fit_while_loop(
    X: Array,
    init_params: Params,
    tol: float,
    max_iter: int,
    reg_covar: float,
    covariance_type: CovType,
    sample_weight: Array | None = None,
) -> tuple[Params, Array, Array, Array]:
    """
    Run EM with a while_loop until convergence.

    Parameters
    ----------
    X : Array, shape (n_samples, n_features)
    init_params : tuple(weights, means, covariances)
    tol : float
        Convergence tolerance on improvement of average lower bound.
    max_iter : int
        Maximum EM iterations.
    reg_covar : float
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    sample_weight : Array | None, shape (n_samples,), default=None

    Returns
    -------
    final_params : tuple(weights, means, covariances)
    final_lb : Array, shape ()
        Final average log likelihood (weighted when sample_weight is given),
        evaluated before the final M-step, not at final_params.
    final_i : Array, shape ()
        Number of iterations run.
    converged : Array, shape ()
        Boolean scalar (True if converged).
    """
    tol = jnp.asarray(tol, dtype=X.dtype)
    # Loop-invariant: normalize once here rather than on every iteration. The
    # weights are not assumed to average one because this is a public jitted
    # entry point that callers may reach with arbitrary positive weights.
    normalized_weight = (
        None if sample_weight is None else sample_weight / jnp.sum(sample_weight)
    )

    state0 = (
        init_params,  # params
        jnp.array(-jnp.inf, dtype=X.dtype),  # prev lower bound
        jnp.array(jnp.inf, dtype=X.dtype),  # delta
        jnp.array(0, dtype=jnp.int32),  # iter
    )

    def cond_fun(state):
        _, _, delta, i = state
        return jnp.logical_and(i < max_iter, delta > tol)

    def body_fun(state):
        params, prev_lb, _, i = state
        log_prob_norm, log_resp = _e_step_func(X, params, covariance_type)
        new_params = _m_step_func(
            X, log_resp, reg_covar, covariance_type, sample_weight
        )
        # The objective describes params at the E-step, before this M-step.
        # Use the same sample weights as fitting for stopping and restart choice.
        lb = (
            jnp.mean(log_prob_norm)
            if normalized_weight is None
            else jnp.sum(log_prob_norm * normalized_weight)
        )
        delta = jnp.abs(lb - prev_lb)
        return (new_params, lb, delta, i + 1)

    final_params, final_lb, final_delta, final_i = jax.lax.while_loop(
        cond_fun, body_fun, state0
    )
    # The loop exits only when delta <= tol (converged) or i >= max_iter
    # (exhausted). Basing `converged` on the delta condition alone correctly
    # reports convergence even when it is reached on the final allowed
    # iteration (final_i == max_iter); the prior `final_i < max_iter` test
    # mislabeled that boundary case as non-converged.
    converged = final_delta <= tol
    return final_params, final_lb, final_i, converged


def _effective_weight_mask(
    sample_weight: np.ndarray, dtype: np.typing.DTypeLike
) -> np.ndarray:
    """
    Rows whose weight is large enough for ``fit`` to treat as data.

    ``fit`` rescales the positive weights to mean one, and the M-step adds a
    ``10 * eps`` guard (eps of ``dtype``) to every component count. A row whose
    rescaled weight is below that guard contributes less to any component than
    the guard itself: a component seeded by it is effectively empty, and its
    mean collapses toward the origin. Such rows are treated as zero weight.
    Together they hold less than ``10 * eps`` of the total weight (each is
    below ``10 * eps`` and the rescaled weights sum to the number of positive
    rows), so dropping them is below the dtype's resolution of the fit.

    Parameters
    ----------
    sample_weight : np.ndarray, shape (n_samples,)
        Nonnegative, finite per-sample weights.
    dtype : np.typing.DTypeLike
        Floating dtype the weights are fitted in (the feature dtype of ``X``).

    Returns
    -------
    keep : np.ndarray, shape (n_samples,)
        Boolean mask of the rows ``fit`` trains on. All False when no weight is
        positive; the largest weight is always kept otherwise.
    """
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    maximum = float(np.max(sample_weight, initial=0.0))
    if maximum <= 0.0:
        return np.zeros(sample_weight.shape, dtype=bool)
    # Divide by the maximum first so the sum cannot overflow for huge weights.
    ratio = sample_weight / maximum
    mean_one = ratio * (np.count_nonzero(ratio) / np.sum(ratio))
    return np.asarray(mean_one >= 10 * np.finfo(dtype).eps, dtype=bool)


def _effective_sample_count(
    X: Array | np.ndarray, sample_weight: np.ndarray | None
) -> int:
    """
    Number of rows ``GaussianMixtureModel.fit`` trains on for this ``X``.

    Callers that size ``n_components`` before fitting must count with the dtype
    of the array actually fitted, since the zero-weight cutoff depends on it.

    Parameters
    ----------
    X : Array | np.ndarray, shape (n_samples, n_features)
        The exact array that will be passed to ``fit``.
    sample_weight : np.ndarray | None, shape (n_samples,)
        Nonnegative, finite per-sample weights, or None for an unweighted fit.

    Returns
    -------
    n_effective : int
        ``n_samples`` when ``sample_weight`` is None, otherwise the number of
        rows kept by the zero-weight cutoff.
    """
    if sample_weight is None:
        return int(X.shape[0])
    dtype = jax.dtypes.canonicalize_dtype(X.dtype)
    return int(np.count_nonzero(_effective_weight_mask(sample_weight, dtype)))


# ---------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------
@dataclass
class GaussianMixtureModel:
    """
    Gaussian Mixture Model (GMM) with EM in JAX.

    Parameters
    ----------
    n_components : int
        Number of mixture components (K).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Covariance parameterization.
    tol : float, default=1e-3
        Convergence threshold on average lower bound improvement.
    reg_covar : float, default=1e-6
        Nonnegative regularization added to diagonal(s) of covariance(s).
    max_iter : int, default=100
        Maximum number of EM iterations.
    n_init : int, default=1
        Number of EM restarts; best run by average lower bound is kept.
    init_params : {'kmeans', 'random'}, default='kmeans'
        Initialization strategy.
        - 'kmeans' runs sklearn.KMeans and converts labels to responsibilities.
        - 'random' draws soft random responsibilities.
    kmeans_init : {'k-means++', 'random'}, default='k-means++'
        KMeans center init scheme (used if init_params='kmeans').
    kmeans_n_init : int, default=1
        KMeans restarts (used if init_params='kmeans').
    random_state : int | None, default=None
        Random seed used in KMeans/k-means++ and random responsibilities.
    weights_init : Array | None, shape (n_components,), default=None
        User-provided initial mixture weights.
    means_init : Array | None, shape (n_components, n_features), default=None
        User-provided initial means.
    covariances_init : Array | None, shape depends on covariance_type, default=None
        User-provided initial covariances.

    Attributes
    ----------
    weights_ : Array, shape (n_components,)
        Learned mixture weights.
    means_ : Array, shape (n_components, n_features)
        Learned component means.
    covariances_ : Array
        Learned covariances:
            'full'      -> (n_components, n_features, n_features)
            'tied'      -> (n_features, n_features)
            'diag'      -> (n_components, n_features)
            'spherical' -> (n_components,)
    precisions_chol_ : Array
        Cholesky of precision matrices with matching shapes.
    converged_ : bool
        True if convergence criteria were met.
    n_iter_ : int
        Number of EM iterations run in the best restart.
    lower_bound_ : float
        Best average log likelihood across restarts, weighted when fitting with
        sample_weight. Evaluated before the best restart's final M-step.
    """

    n_components: int
    covariance_type: CovType = "full"
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: Literal["kmeans", "random"] = "kmeans"
    kmeans_init: Literal["k-means++", "random"] = "k-means++"
    kmeans_n_init: int = 1
    random_state: int | None = None

    weights_init: Array | None = None
    means_init: Array | None = None
    covariances_init: Array | None = None

    weights_: Array | None = field(init=False, default=None)
    means_: Array | None = field(init=False, default=None)
    covariances_: Array | None = field(init=False, default=None)
    precisions_chol_: Array | None = field(init=False, default=None)
    converged_: bool | None = field(init=False, default=None)
    n_iter_: int | None = field(init=False, default=None)
    lower_bound_: float | None = field(init=False, default=None)

    # ----------------- Validation & conversion -----------------
    def __post_init__(self) -> None:
        # Import locally to avoid circular dependency:
        # gmm.py -> exceptions.py (if at top level, would create cycle through likelihoods/__init__.py)
        from non_local_detector._validation import ensure_probability_distribution
        from non_local_detector.exceptions import ValidationError

        if self.n_components <= 0:
            raise ValidationError(
                "Number of GMM components must be positive",
                expected="n_components > 0",
                got=f"n_components = {self.n_components}",
                hint="Use at least 1 component for the Gaussian mixture",
                example="    GaussianMixture(n_components=3, ...)",
            )
        if self.covariance_type not in ["full", "tied", "diag", "spherical"]:
            raise ValidationError(
                "Invalid covariance type for GMM",
                expected="One of: 'full', 'tied', 'diag', 'spherical'",
                got=f"covariance_type = '{self.covariance_type}'",
                hint="Use 'full' for full covariance matrices, 'tied' for shared covariance, 'diag' for diagonal, or 'spherical' for single variance",
                example="    GaussianMixture(covariance_type='full', ...)",
            )
        if self.tol < 0:
            raise ValidationError(
                "Convergence tolerance must be non-negative",
                expected="tol >= 0",
                got=f"tol = {self.tol}",
                hint="Use a small positive value like 1e-4 for stricter convergence",
                example="    GaussianMixture(tol=1e-4, ...)",
            )
        if self.reg_covar < 0:
            raise ValidationError(
                "Covariance regularization must be non-negative",
                expected="reg_covar >= 0",
                got=f"reg_covar = {self.reg_covar}",
                hint="Use a small positive value like 1e-6 to prevent singular covariance matrices",
                example="    GaussianMixture(reg_covar=1e-6, ...)",
            )
        if self.max_iter <= 0:
            raise ValidationError(
                "Maximum iterations must be positive",
                expected="max_iter > 0",
                got=f"max_iter = {self.max_iter}",
                hint="Increase max_iter to allow more time for convergence",
                example="    GaussianMixture(max_iter=1000, ...)",
            )
        if self.n_init <= 0:
            raise ValidationError(
                "Number of initializations must be positive",
                expected="n_init > 0",
                got=f"n_init = {self.n_init}",
                hint="Use multiple initializations (e.g., 10) to find better solutions",
                example="    GaussianMixture(n_init=10, ...)",
            )

        if self.weights_init is not None:
            self.weights_init = jnp.asarray(self.weights_init)
            if self.weights_init.shape != (self.n_components,):
                raise ValidationError(
                    "weights_init has the wrong shape",
                    expected=f"shape ({self.n_components},)",
                    got=f"shape {tuple(self.weights_init.shape)}",
                    hint="Provide one initial mixture weight per component",
                    example="    GaussianMixture(n_components=3, weights_init=[0.2, 0.3, 0.5])",
                )
            if _has_negative_or_nonfinite(self.weights_init):
                # Caught here so a bad weight is diagnosed accurately: a negative
                # or NaN weight would otherwise produce jnp.log(weights) = NaN
                # deep in EM and be misreported as a singular covariance.
                raise ValidationError(
                    "weights_init must be finite and nonnegative",
                    expected="all weights_init >= 0 and finite",
                    got=f"weights_init = {np.asarray(self.weights_init)}",
                    hint="Mixture weights are probabilities; they cannot be negative or NaN",
                    example="    GaussianMixture(n_components=3, weights_init=[0.2, 0.3, 0.5])",
                )
            ensure_probability_distribution(self.weights_init, "weights_init")
        if self.means_init is not None:
            self.means_init = jnp.asarray(self.means_init)
            if (
                self.means_init.ndim != 2
                or self.means_init.shape[0] != self.n_components
            ):
                raise ValidationError(
                    "means_init has the wrong shape",
                    expected=f"shape ({self.n_components}, n_features)",
                    got=f"shape {tuple(self.means_init.shape)}",
                    hint="Provide one initial mean vector per component",
                    example="    means_init = np.zeros((n_components, n_features))",
                )
        if self.covariances_init is not None:
            # Feature-dependent shape is validated at fit time (needs n_features).
            self.covariances_init = jnp.asarray(self.covariances_init)

    def _supplied_params(self) -> tuple[Array, Array, Array] | None:
        """(weights, means, covariances) if all are user-initialized, else None."""
        if (
            self.weights_init is not None
            and self.means_init is not None
            and self.covariances_init is not None
        ):
            return self.weights_init, self.means_init, self.covariances_init
        return None

    # ----------------- Public API -----------------
    def _validate_fit_inputs(
        self, X: Array, sample_weight: Array | np.ndarray | None
    ) -> np.ndarray | None:
        """
        Validate fit inputs whose constraints depend on ``X``, and select rows.

        Checks that ``sample_weight`` (if given) is finite, nonnegative, and has
        one entry per sample, that ``X`` is 2-D and that the rows ``fit`` will
        train on are finite, that those rows still cover every component, and
        that any user-provided init array agrees with ``X``'s number of
        features. Construction-time constraints (component counts,
        ``weights_init`` sum) are enforced in ``__post_init__``.

        The weights are validated before ``X``'s finiteness because ``fit``
        trains only on the rows this method keeps: a NaN in a row that carries
        no usable weight is not part of the fitted data and must not reject it.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        sample_weight : Array | np.ndarray | None, shape (n_samples,)

        Returns
        -------
        keep : np.ndarray | None, shape (n_samples,)
            Boolean mask of the rows ``fit`` trains on, or None when
            ``sample_weight`` is None (every row is kept).

        Raises
        ------
        ValidationError
            If any input is inconsistent.
        """
        from non_local_detector.exceptions import ValidationError

        if X.ndim != 2:
            raise ValidationError(
                "X must be a 2-D (n_samples, n_features) array",
                expected="X.ndim == 2",
                got=f"X.ndim = {X.ndim}, shape {tuple(X.shape)}",
                hint="Reshape X so each row is one sample's feature vector",
                example="    X = X.reshape(n_samples, n_features)",
            )
        n_samples, n_features = X.shape

        keep = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight)
            if sw.shape != (n_samples,):
                raise ValidationError(
                    "sample_weight has the wrong length",
                    expected=f"shape ({n_samples},) matching X's rows",
                    got=f"shape {tuple(sw.shape)}",
                    hint="Provide one nonnegative weight per sample in X",
                    example="    model.fit(X, key, sample_weight=np.ones(len(X)))",
                )
            if not np.all(np.isfinite(sw)) or np.any(sw < 0):
                raise ValidationError(
                    "sample_weight must be finite and nonnegative",
                    expected="all sample_weight >= 0 and finite",
                    got=f"min = {float(np.min(sw)):.6g}, "
                    f"all_finite = {bool(np.all(np.isfinite(sw)))}",
                    hint="Remove or zero out negative/NaN weights before fitting",
                    example="    sample_weight = np.clip(sample_weight, 0.0, None)",
                )
            if not np.any(sw > 0):
                # An all-zero sample_weight has no effective data: dropping the
                # zero-weight rows in ``fit`` would leave no samples, and
                # KMeans/EM would then fail opaquely.
                raise ValidationError(
                    "sample_weight sums to 0 (no effective training data)",
                    expected="sum(sample_weight) > 0",
                    got="sum = 0",
                    hint="At least one sample must carry positive weight",
                    example="    sample_weight = np.ones(len(X))",
                )
            keep = _effective_weight_mask(sw, jax.dtypes.canonicalize_dtype(X.dtype))

        # Only the kept rows are fitted, so only they must be finite. Caught
        # here for a clear message; otherwise NaN/Inf surfaces either as an
        # opaque sklearn KMeans ValueError or as a NaN lower bound misreported
        # as a singular covariance.
        # A per-row flag, not a gathered copy of X, so a partly de-weighted fit
        # does not copy the kept rows twice (``fit`` gathers them once).
        if keep is None:
            all_finite = bool(jnp.all(jnp.isfinite(X)))
        else:
            finite_rows = np.asarray(jnp.all(jnp.isfinite(X), axis=1))
            all_finite = bool(np.all(finite_rows[keep]))
        if not all_finite:
            raise ValidationError(
                "X contains non-finite values",
                expected="all entries of X with positive weight are finite",
                got="X has NaN or Inf entries",
                hint="Remove or impute NaN/Inf samples before fitting",
                example="    X = X[np.all(np.isfinite(X), axis=1)]",
            )

        # ``fit`` drops the rows this method excluded before initialization and
        # EM, so the rows that remain must still cover every component. Fewer
        # rows than components would otherwise reach sklearn KMeans as a raw
        # ValueError naming a sample count the caller never passed, or let
        # random initialization fit duplicate components silently. A full
        # warm start (every init array supplied) initializes from no data, so
        # EM may start from it on any nonempty batch.
        n_effective = n_samples if keep is None else int(np.count_nonzero(keep))
        if n_effective < self.n_components and self._supplied_params() is None:
            if keep is None:
                raise ValidationError(
                    "fewer samples than mixture components",
                    expected=f"at least n_components = {self.n_components} samples",
                    got=f"{n_samples} samples",
                    hint="Reduce n_components or supply more samples",
                    example="    model = GaussianMixtureModel(n_components=1)",
                )
            raise ValidationError(
                "fewer positive-weight samples than mixture components",
                expected=f"at least n_components = {self.n_components} samples "
                "with positive weight",
                got=f"{n_effective} of {n_samples} samples have usable positive "
                "weight (a weight below 10 * eps of the feature dtype, after "
                "rescaling the weights to mean one, counts as zero)",
                hint="Reduce n_components or supply more positive-weight samples",
                example="    model = GaussianMixtureModel(n_components=1)",
            )

        if self.means_init is not None and self.means_init.shape[1] != n_features:
            raise ValidationError(
                "means_init feature dimension does not match X",
                expected=f"means_init.shape == ({self.n_components}, {n_features})",
                got=f"shape {tuple(self.means_init.shape)}",
                hint="Each initial mean needs one entry per feature in X",
                example=f"    means_init has shape ({self.n_components}, {n_features})",
            )

        if self.covariances_init is not None:
            expected_shapes = {
                "full": (self.n_components, n_features, n_features),
                "tied": (n_features, n_features),
                "diag": (self.n_components, n_features),
                "spherical": (self.n_components,),
            }
            expected = expected_shapes[self.covariance_type]
            if tuple(self.covariances_init.shape) != expected:
                raise ValidationError(
                    "covariances_init shape is inconsistent with covariance_type and X",
                    expected=f"shape {expected} for covariance_type='{self.covariance_type}'",
                    got=f"shape {tuple(self.covariances_init.shape)}",
                    hint="Match covariances_init to the covariance parameterization and n_features",
                    example=f"    covariance_type='{self.covariance_type}' expects {expected}",
                )

        return keep

    def fit(
        self, X: Array, key: jax.Array, sample_weight: Array | np.ndarray | None = None
    ) -> GaussianMixtureModel:
        """
        Fit the model by EM.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
            Training data.
        key : jax.Array
            PRNG key; if `random_state` is provided, it overrides JAX randomness in inits.
        sample_weight : Array | np.ndarray | None, shape (n_samples,), default=None
            Nonnegative per-sample weights. They are rescaled to mean one
            before initialization and EM, so their common scale does not change
            the fit. A weight whose rescaled value is below ``10 * eps`` of the
            feature dtype (about 1.2e-6 for float32) is treated as zero: it
            weighs less than the empty-component guard, and such rows together
            carry under ``10 * eps`` of the total weight. Zero-weight rows are
            excluded from initialization and fitting, and only the kept rows
            are required to be finite. The convergence objective and restart
            ranking use the weighted average log likelihood.

        Returns
        -------
        self : GaussianMixtureModel

        Raises
        ------
        ValidationError
            If ``X`` is not 2-D, a fitted row of ``X`` is non-finite,
            ``sample_weight`` is negative, all-zero or the wrong length, fewer
            rows are fitted than there are components (with or without
            ``sample_weight``) and not every init array is supplied, or any
            user-provided init array is inconsistent with ``X``.
        RuntimeError
            If every restart fails to produce a finite lower bound (e.g. a
            component's covariance became singular). The message names the
            likely cause and remedy.
        """
        # Validate and normalize on the host before JAX arithmetic can flush
        # subnormal weights to zero or narrow finite float64 weights to float32.
        # ``keep`` marks the rows EM will see; the count guards were applied to
        # exactly those rows, so the guards and the fitted data agree.
        host_weights = None if sample_weight is None else np.asarray(sample_weight)
        keep = self._validate_fit_inputs(X, host_weights)

        best_lower_bound = -jnp.inf
        init_keys = jax.random.split(key, self.n_init)

        sw = None
        if keep is not None:  # sample_weight was given
            weights = np.asarray(host_weights, dtype=np.float64)
            if not keep.all():
                X, weights = X[keep], weights[keep]
            # Divide by the maximum first, in float64: the rescaled sum is then
            # bounded by the number of samples, so even huge float64 weights
            # cannot overflow the sum used for the mean-one rescale. The
            # division also makes each weight's own magnitude irrelevant -- only
            # its ratio to the largest weight, which ``keep`` has already
            # bounded below. A new array, so the caller's weights are untouched.
            weights = weights / np.max(weights)
            weights *= X.shape[0] / np.sum(weights)
            sw = jnp.asarray(weights, dtype=X.dtype)

        # Reset all fitted attributes so a refit cannot silently retain the
        # previous fit's parameters. If every init returns final_lb == -inf/NaN,
        # the `if final_lb > best_lower_bound` block below never fires; because
        # weights_ is now None the guard raises rather than reporting the stale
        # means/covariances from an earlier successful fit with
        # converged_=False, n_iter_=0.
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_chol_ = None
        self.lower_bound_ = None
        self.converged_ = False
        self.n_iter_ = 0

        saw_nan = False
        for i in range(self.n_init):
            params = self._initialize_parameters(
                X, init_keys[i], sample_weight=sw, init_index=i
            )
            (
                final_w,
                final_m,
                final_S,
                final_lb,
                n_iter,
                final_converged,
            ) = self._fit_single(X, params, sample_weight=sw)

            # A singular covariance makes the Cholesky (and hence the lower
            # bound) NaN; NaN never beats best_lower_bound, so such a restart is
            # discarded. Remember it to give an actionable error if all fail.
            saw_nan = saw_nan or bool(np.isnan(final_lb))

            if final_lb > best_lower_bound:
                best_lower_bound = final_lb
                self.weights_ = final_w
                self.means_ = final_m
                self.covariances_ = final_S
                self.precisions_chol_ = _compute_precision_cholesky(
                    self.covariances_, self.covariance_type
                )
                self.n_iter_ = n_iter
                self.converged_ = bool(final_converged)

        self.lower_bound_ = float(best_lower_bound)
        if self.weights_ is None:
            if saw_nan:
                raise RuntimeError(
                    "GMM fitting failed: a component's covariance became "
                    "singular, giving a non-finite log-likelihood. Increase "
                    "reg_covar (e.g. to 1e-4), reduce n_components, or remove "
                    "duplicate/degenerate samples, then refit."
                )
            raise RuntimeError(
                "GMM fitting failed: no restart produced a finite lower bound. "
                "Check X for NaN/Inf values or try a different initialization."
            )

        if not self.converged_:
            warnings.warn(
                f"GaussianMixture did not converge within max_iter={self.max_iter} "
                f"iterations (best restart ran {self.n_iter_}). Final lower-bound "
                f"delta exceeded tol={self.tol:.3e}.",
                UserWarning,
                stacklevel=2,
            )

        # diag/spherical variances are clamped up to a 1e-10 floor for a finite
        # precision (rather than raising like full/tied on a singular
        # covariance). Surface it so a collapsed component is not silent.
        if self.covariance_type in ("diag", "spherical") and (
            self.covariances_ is not None
            and bool(jnp.any(jnp.asarray(self.covariances_) <= 1e-10))
        ):
            warnings.warn(
                "GaussianMixture: a diag/spherical component variance hit the "
                "1e-10 floor (a collapsed/near-singular component); its precision "
                "was clamped rather than raised. Increase reg_covar or reduce "
                "n_components.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def _check_fitted(self) -> None:
        """Raise if the model is not fitted (predict/score need fitted params)."""
        if (
            self.weights_ is None
            or self.means_ is None
            or self.precisions_chol_ is None
        ):
            raise RuntimeError("This GMM instance is not fitted yet.")

    def predict(self, X: Array) -> Array:
        """
        Predict labels by maximum a posteriori component.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)

        Returns
        -------
        labels : Array, shape (n_samples,)
        """
        return jnp.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: Array) -> Array:
        """
        Posterior probabilities (responsibilities) per sample and component.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)

        Returns
        -------
        probs : Array, shape (n_samples, n_components)
        """
        self._check_fitted()
        _, log_resp = _estimate_log_prob_resp(
            X, self.means_, self.precisions_chol_, self.covariance_type, self.weights_
        )
        return jnp.exp(log_resp)

    def score_samples(self, X: Array) -> Array:
        """
        Per-sample log-likelihood log p(x).

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)

        Returns
        -------
        log_likelihoods : Array, shape (n_samples,)
        """
        self._check_fitted()
        log_likelihoods, _ = _estimate_log_prob_resp(
            X, self.means_, self.precisions_chol_, self.covariance_type, self.weights_
        )
        return log_likelihoods

    # ----------------- Initialization helpers -----------------
    def _initialize_random(self, X: Array, key: jax.Array) -> Array:
        """
        Random soft responsibilities.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        key : jax.Array

        Returns
        -------
        resp : Array, shape (n_samples, n_components)
        """
        if self.random_state is not None:
            key = jax.random.fold_in(key, self.random_state)
        n_samples, _ = X.shape
        rand_resp = jax.random.uniform(
            key, shape=(n_samples, self.n_components), dtype=X.dtype
        )
        return rand_resp / rand_resp.sum(axis=1, keepdims=True)

    def _initialize_kmeans_resp(
        self,
        X: Array,
        key: jax.Array,
        init_index: int = 0,
        *,
        sample_weight: Array | None = None,
    ) -> Array:
        """
        One-hot responsibilities from sklearn KMeans.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        key : jax.Array
            Per-restart PRNG key. Used to derive a distinct KMeans seed for
            restarts beyond the first when ``random_state`` is set.
        init_index : int, default=0
            Index of the current restart within ``n_init``.
        sample_weight : Array | None, optional
            Relative sample weights for KMeans centers and initialization.

        Returns
        -------
        resp : Array, shape (n_samples, n_components)

        Notes
        -----
        Seeding rules keep single-init fits reproducible and byte-compatible
        while making multi-restart runs actually explore:

        - ``random_state is None``: pass ``None`` to KMeans, which draws from
          NumPy's global RNG and so already varies across restarts.
        - ``random_state`` set, ``init_index == 0``: use ``random_state``
          directly (the reproducible primary init).
        - ``random_state`` set, ``init_index > 0``: derive a distinct,
          reproducible seed from ``key`` so ``n_init`` restarts differ instead
          of repeating the primary init.
        """
        n_samples, _ = X.shape
        X_np = np.asarray(X)
        if self.random_state is None:
            seed: int | None = None
        elif init_index == 0:
            seed = self.random_state
        else:
            seed = int(
                jax.random.randint(
                    jax.random.fold_in(key, self.random_state),
                    (),
                    0,
                    jnp.iinfo(jnp.int32).max,
                )
            )
        km = KMeans(
            n_clusters=self.n_components,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            random_state=seed,
        ).fit(
            X_np,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )
        labels = km.labels_
        resp_np = np.zeros((n_samples, self.n_components), dtype=X_np.dtype)
        resp_np[np.arange(n_samples), labels] = 1.0
        return jnp.asarray(resp_np, dtype=X.dtype)

    def _initialize_parameters(
        self,
        X: Array,
        key: jax.Array,
        sample_weight: Array | None = None,
        init_index: int = 0,
    ) -> Params:
        """
        Construct initial (weights, means, covariances).

        Responsibilities are built by the chosen init strategy and the
        (weights, means, covariances) estimated from them. Any user-provided
        init array overrides the corresponding estimate; the pieces the user
        does *not* supply keep their responsibility-based values (sklearn
        parity). Filling missing means/covariances from the resp-based init —
        rather than tiling the global mean / an identity — avoids a degenerate
        symmetry trap where all components start identical and EM can never
        separate them.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        key : jax.Array
        sample_weight : Array | None, shape (n_samples,), default=None
        init_index : int, default=0
            Index of the current restart within ``n_init``; forwarded to
            ``_initialize_kmeans_resp`` so restarts beyond the first get
            distinct KMeans seeds.

        Returns
        -------
        weights : Array, shape (n_components,)
        means : Array, shape (n_components, n_features)
        covariances : Array
            Shape depends on `covariance_type`.
        """
        # When every piece is user-supplied there is nothing to fill, so skip
        # the responsibility-based estimate (and the needless KMeans run).
        supplied = self._supplied_params()
        if supplied is not None:
            return supplied

        if self.init_params == "kmeans":
            resp = self._initialize_kmeans_resp(
                X, key, init_index, sample_weight=sample_weight
            )
        elif self.init_params == "random":
            resp = self._initialize_random(X, key)
        else:
            raise ValueError(f"Unknown init_params: {self.init_params}")

        nk, means_est, cov_est = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type, sample_weight
        )
        weights_est = nk / jnp.sum(nk)

        # Override only the pieces the user supplied; the rest keep their
        # resp-based values (sklearn parity, avoids the symmetry trap).
        weights = self.weights_init if self.weights_init is not None else weights_est
        means = self.means_init if self.means_init is not None else means_est
        covariances = (
            self.covariances_init if self.covariances_init is not None else cov_est
        )
        return weights, means, covariances

    # ----------------- Single EM run -----------------
    def _fit_single(
        self,
        X: Array,
        params: Params,
        sample_weight: Array | None = None,
    ) -> tuple[Array, Array, Array, float, int, bool]:
        """
        Run a single EM fit from given initial parameters.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        params : tuple(weights, means, covariances)
        sample_weight : Array | None, shape (n_samples,), default=None

        Returns
        -------
        weights : Array, shape (n_components,)
        means : Array, shape (n_components, n_features)
        covariances : Array
            Shape depends on `covariance_type`.
        lower_bound : float
            Final average log likelihood (weighted when sample_weight is given),
            evaluated before the final M-step.
        n_iter : int
            Number of EM iterations performed.
        converged : bool
            True if the EM lower-bound delta dropped below ``tol`` before
            hitting ``max_iter``.
        """
        final_params, final_lb, n_iter, final_converged = _em_fit_while_loop(
            X=X,
            init_params=params,
            tol=self.tol,
            max_iter=self.max_iter,
            reg_covar=self.reg_covar,
            covariance_type=self.covariance_type,
            sample_weight=sample_weight,
        )
        final_weights, final_means, final_covariances = final_params
        return (
            final_weights,
            final_means,
            final_covariances,
            float(final_lb),
            int(n_iter),
            bool(final_converged),
        )
