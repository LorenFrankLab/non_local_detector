"""JAX-based Gaussian Mixture Model implementation for clusterless neural decoding.

This module provides a high-performance implementation of Gaussian Mixture Models
using JAX for automatic differentiation and GPU acceleration. It's specifically
designed for modeling clusterless (continuous) spike waveform features in neural
decoding applications.

The main class GaussianMixtureModel follows scikit-learn conventions while
leveraging JAX for efficient computation on both CPU and GPU.
"""

from __future__ import annotations

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
    diff = X[jnp.newaxis, :, :] - means[:, jnp.newaxis, :]  # (K, N, D)
    covariances = jax.vmap(lambda r, d, n: ((d.T * r) @ d) / n)(resp.T, diff, nk)
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
    n_samples, n_features = X.shape
    # Per-sample total weights: w_i = sum_k resp[i,k]
    w = resp.sum(axis=1, keepdims=True)  # (N, 1)
    sum_w = jnp.sum(w)  # scalar (total weighted sample size)

    # Second moment: sum_i w_i x_i x_i^T
    XTX_w = X.T @ (X * w)  # (D, D)

    # Mean term: sum_k nk_k mu_k mu_k^T == (nk * means^T) @ means
    avg_means2 = (nk * means.T) @ means  # (D, D)

    covariance = (XTX_w - avg_means2) / jnp.maximum(sum_w, 1.0)
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
    avg_X2 = resp.T @ (X * X) / nk[:, jnp.newaxis]
    avg_means2 = means**2
    return avg_X2 - avg_means2 + reg_covar


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
    msg = (
        "Fitting failed: ill-defined empirical covariance. "
        "Decrease n_components, increase reg_covar, or scale the data."
    )

    def single_inv_chol(cov: Array) -> Array:
        cov_chol = jnp.linalg.cholesky(cov)
        ident = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        return jax.scipy.linalg.solve_triangular(cov_chol, ident, lower=True).T

    if covariance_type == "full":
        return jax.vmap(single_inv_chol)(covariances)
    if covariance_type == "tied":
        return single_inv_chol(covariances)
    # diag / spherical
    if jnp.any(covariances <= 0.0):
        raise ValueError(msg)
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
    sample_weight : Optional[Array], shape (n_samples,), default=None
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
    _, n_features = X.shape

    if covariance_type == "full":

        def comp(mu, R):
            Y = (X - mu) @ R
            return jnp.sum(Y * Y, axis=1)

        maha = jax.vmap(comp)(means, precisions_chol).T
    elif covariance_type == "tied":
        Y = (X[None, :, :] - means[:, None, :]) @ precisions_chol  # (K, N, D)
        maha = jnp.sum(Y * Y, axis=2).T
    elif covariance_type == "diag":
        precisions = precisions_chol**2
        maha = (
            jnp.sum(means**2 * precisions, axis=1)
            - 2.0 * X @ (means * precisions).T
            + (X**2) @ precisions.T
        )
    else:  # spherical
        precisions = precisions_chol**2
        maha = (
            jnp.sum(X**2, axis=1)[:, None] * precisions[None, :]
            - 2 * X @ (means.T * precisions)
            + jnp.sum(means**2, axis=1) * precisions
        )

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
    sample_weight : Optional[Array], shape (n_samples,), default=None

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
    total_weight = X.shape[0] if sample_weight is None else jnp.sum(sample_weight)
    weights = nk / total_weight
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
    sample_weight : Optional[Array], shape (n_samples,), default=None

    Returns
    -------
    final_params : tuple(weights, means, covariances)
    final_lb : Array, shape ()
        Final average lower bound (scalar).
    final_i : Array, shape ()
        Number of iterations run.
    converged : Array, shape ()
        Boolean scalar (True if converged).
    """
    tol = jnp.asarray(tol, dtype=X.dtype)

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
        lb = jnp.mean(log_prob_norm)
        delta = jnp.abs(lb - prev_lb)
        return (new_params, lb, delta, i + 1)

    final_params, final_lb, final_delta, final_i = jax.lax.while_loop(
        cond_fun, body_fun, state0
    )
    converged = jnp.logical_and(final_i < max_iter, final_delta <= tol)
    return final_params, final_lb, final_i, converged


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
    random_state : Optional[int], default=None
        Random seed used in KMeans/k-means++ and random responsibilities.
    weights_init : Optional[Array], shape (n_components,), default=None
        User-provided initial mixture weights.
    means_init : Optional[Array], shape (n_components, n_features), default=None
        User-provided initial means.
    covariances_init : Optional[Array], shape depends on covariance_type, default=None
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
        Best average lower bound across restarts.
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
        if self.n_components <= 0:
            raise ValueError("n_components must be > 0")
        if self.covariance_type not in ["full", "tied", "diag", "spherical"]:
            raise ValueError(f"Invalid covariance_type: {self.covariance_type}")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.n_init <= 0:
            raise ValueError("n_init must be > 0")

        if self.weights_init is not None:
            self.weights_init = jnp.asarray(self.weights_init)
        if self.means_init is not None:
            self.means_init = jnp.asarray(self.means_init)
        if self.covariances_init is not None:
            self.covariances_init = jnp.asarray(self.covariances_init)

    # ----------------- Public API -----------------
    def fit(
        self, X: Array, key: jax.Array, sample_weight: Array | None = None
    ) -> GaussianMixtureModel:
        """
        Fit the model by EM.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
            Training data.
        key : jax.Array
            PRNG key; if `random_state` is provided, it overrides JAX randomness in inits.
        sample_weight : Optional[Array], shape (n_samples,), default=None
            Nonnegative per-sample weights.

        Returns
        -------
        self : GaussianMixtureModel
        """
        best_lower_bound = -jnp.inf
        init_keys = jax.random.split(key, self.n_init)

        sw = (
            None if sample_weight is None else jnp.asarray(sample_weight, dtype=X.dtype)
        )

        for i in range(self.n_init):
            params = self._initialize_parameters(X, init_keys[i], sample_weight=sw)
            final_w, final_m, final_S, final_lb, n_iter = self._fit_single(
                X, params, sample_weight=sw
            )

            if final_lb > best_lower_bound:
                best_lower_bound = final_lb
                self.weights_ = final_w
                self.means_ = final_m
                self.covariances_ = final_S
                self.precisions_chol_ = _compute_precision_cholesky(
                    self.covariances_, self.covariance_type
                )
                self.n_iter_ = n_iter
                self.converged_ = n_iter < self.max_iter

        self.lower_bound_ = float(best_lower_bound)
        if self.weights_ is None:
            raise RuntimeError("Fitting failed.")
        return self

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
        if (
            self.weights_ is None
            or self.means_ is None
            or self.precisions_chol_ is None
        ):
            raise RuntimeError("This GMM instance is not fitted yet.")
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
        if (
            self.weights_ is None
            or self.means_ is None
            or self.precisions_chol_ is None
        ):
            raise RuntimeError("This GMM instance is not fitted yet.")
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

    def _initialize_kmeans_resp(self, X: Array) -> Array:
        """
        One-hot responsibilities from sklearn KMeans.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)

        Returns
        -------
        resp : Array, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        X_np = np.asarray(X)
        km = KMeans(
            n_clusters=self.n_components,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            random_state=self.random_state,
        ).fit(X_np)
        labels = km.labels_
        resp_np = np.zeros((n_samples, self.n_components), dtype=X_np.dtype)
        resp_np[np.arange(n_samples), labels] = 1.0
        return jnp.asarray(resp_np, dtype=X.dtype)

    def _initialize_parameters(
        self,
        X: Array,
        key: jax.Array,
        sample_weight: Array | None = None,
    ) -> Params:
        """
        Construct initial (weights, means, covariances).

        If user-provided inits exist, they are used (with basic shape checks).
        Otherwise, responsibilities are built by the chosen init strategy,
        then means/weights/covariances are computed from those responsibilities.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        key : jax.Array
        sample_weight : Optional[Array], shape (n_samples,), default=None

        Returns
        -------
        weights : Array, shape (n_components,)
        means : Array, shape (n_components, n_features)
        covariances : Array
            Shape depends on `covariance_type`.
        """
        _, n_features = X.shape

        # Use user-provided inits if any
        if (
            (self.means_init is not None)
            or (self.weights_init is not None)
            or (self.covariances_init is not None)
        ):
            means = (
                self.means_init
                if self.means_init is not None
                else jnp.tile(jnp.mean(X, axis=0), (self.n_components, 1))
            )
            weights = (
                self.weights_init
                if self.weights_init is not None
                else jnp.full(self.n_components, 1 / self.n_components, dtype=X.dtype)
            )
            if self.covariances_init is not None:
                covariances = self.covariances_init
            else:
                eye = jnp.eye(n_features, dtype=X.dtype)
                if self.covariance_type == "full":
                    covariances = jnp.tile(eye, (self.n_components, 1, 1))
                elif self.covariance_type == "tied":
                    covariances = eye
                elif self.covariance_type == "diag":
                    covariances = jnp.ones(
                        (self.n_components, n_features), dtype=X.dtype
                    )
                else:  # spherical
                    covariances = jnp.ones((self.n_components,), dtype=X.dtype)
            return weights, means, covariances

        # Otherwise, build responsibilities from the chosen init
        if self.init_params == "kmeans":
            resp = self._initialize_kmeans_resp(X)
        elif self.init_params == "random":
            resp = self._initialize_random(X, key)
        else:
            raise ValueError(f"Unknown init_params: {self.init_params}")

        # Compute initial params FROM responsibilities (sklearn parity)
        nk, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type, sample_weight
        )
        total_weight = X.shape[0] if sample_weight is None else jnp.sum(sample_weight)
        weights = nk / total_weight
        return weights, means, covariances

    # ----------------- Single EM run -----------------
    def _fit_single(
        self,
        X: Array,
        params: Params,
        sample_weight: Array | None = None,
    ) -> tuple[Array, Array, Array, float, int]:
        """
        Run a single EM fit from given initial parameters.

        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
        params : tuple(weights, means, covariances)
        sample_weight : Optional[Array], shape (n_samples,), default=None

        Returns
        -------
        weights : Array, shape (n_components,)
        means : Array, shape (n_components, n_features)
        covariances : Array
            Shape depends on `covariance_type`.
        lower_bound : float
            Final average lower bound.
        n_iter : int
            Number of EM iterations performed.
        """
        final_params, final_lb, n_iter, _ = _em_fit_while_loop(
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
        )
