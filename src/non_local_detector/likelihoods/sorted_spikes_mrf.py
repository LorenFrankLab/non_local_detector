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

Place fields ``exp(eta)`` are stored FULL-GRID exactly like ``sorted_spikes_kde`` /
``sorted_spikes_diffusion``, so the Poisson prediction is shared with the diffusion
likelihood (:func:`predict_sorted_spikes_mrf_log_likelihood` is that function).
"""

import jax.numpy as jnp
import numpy as np
import scipy.optimize

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import EPS
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    environment_graph,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    pixellate_interior_fields,
    predict_sorted_spikes_diffusion_log_likelihood,
)

# The MRF place-field / Poisson-prediction contract is identical to the diffusion
# smoother's (both store FULL-GRID rate maps), so the prediction is shared.
predict_sorted_spikes_mrf_log_likelihood = (
    predict_sorted_spikes_diffusion_log_likelihood
)

# Clip the linear predictor before exponentiating to avoid overflow during IRLS.
_ETA_CLIP = 30.0
# Ridge added to the Hessian diagonal for numerical stability.
_HESSIAN_JITTER = 1e-10
# Search bounds for log(lambda) during REML selection.
_LOG_PENALTY_BOUNDS = (-8.0, 20.0)
# Default cap on the reduced-rank basis when ``rank`` is None (mgcv-style): the
# full-rank fit forms and solves a dense per-neuron Hessian each Newton step, which
# is impractical on large grids, and REML tunes smoothness within the basis anyway.
_DEFAULT_MAX_RANK = 250


def _assemble_hessian(
    basis: np.ndarray, mu: np.ndarray, penalty_diag: np.ndarray
) -> np.ndarray:
    """Batched penalized Hessian ``Bᵀ diag(mu_k) B + diag(penalty_diag)`` per neuron.

    Returns shape ``(n_neurons, rank, rank)``.
    """
    hessian = np.einsum("ir,ik,is->krs", basis, mu, basis, optimize=True)
    diag = np.arange(basis.shape[1])
    hessian[:, diag, diag] += penalty_diag + _HESSIAN_JITTER
    return hessian


def mrf_penalized_poisson_fit(
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
    penalty: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    coeffs : np.ndarray, shape (rank, n_neurons)
        Fitted coefficients.
    eta : np.ndarray, shape (n_bins, n_neurons)
        Linear predictor ``basis @ coeffs``.
    mu : np.ndarray, shape (n_bins, n_neurons)
        Fitted mean ``occupancy[:, None] * exp(eta)``.
    """
    n_bins = basis.shape[0]
    penalty_diag = penalty * penalty_weights  # (rank,)

    # Warm start each neuron from a constant log-rate (fast, convex problem).
    total_occupancy = max(float(occupancy.sum()), 1e-9)
    eta0 = np.log(np.clip(counts.sum(axis=0) / total_occupancy, 1e-6, None))
    basis_pinv_ones = np.linalg.lstsq(basis, np.ones(n_bins), rcond=None)[0]
    coeffs = basis_pinv_ones[:, None] * eta0[None, :]  # (rank, n_neurons)

    # Fixed penalized-IRLS iterations (like the mgcv reference). No convergence
    # warning: for a silent cell (all-zero counts) or a near-zero-occupancy bin the
    # linear predictor legitimately drifts toward a clip boundary rather than a
    # tol-sized step, so a global "did not converge" signal would misfire on common
    # data; the eta clip below guarantees a finite result regardless.
    for _ in range(max_iter):
        eta = basis @ coeffs
        mu = occupancy[:, None] * np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))
        grad = basis.T @ (counts - mu) - penalty_diag[:, None] * coeffs
        hessian = _assemble_hessian(basis, mu, penalty_diag)
        # Batched solve H_k step_k = grad_k over the neuron axis.
        step = np.linalg.solve(hessian, grad.T[..., None])[..., 0]  # (n_neurons, rank)
        coeffs = coeffs + step.T
        # step.size == 0 covers the zero-neuron case (np.max would raise on it).
        if step.size == 0 or np.max(np.abs(step)) < tol:
            break

    # Clip the returned linear predictor so the mean and any rate the caller derives
    # via exp(eta) stay finite even if a low-penalty / near-zero-occupancy fit drove
    # eta very large (mirrors the mgcv reference, which clips at this point too).
    eta = np.clip(basis @ coeffs, -_ETA_CLIP, _ETA_CLIP)
    mu = occupancy[:, None] * np.exp(eta)
    return coeffs, eta, mu


def _penalty_rank(penalty_weights: np.ndarray) -> int:
    """Number of penalized directions (positive eigenvalues) for the REML df term."""
    largest = penalty_weights.max()
    if largest <= 0:
        return 0
    return int(np.sum(penalty_weights > 1e-12 * largest))


def mrf_reml_objective(
    log_penalty: float,
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
) -> float:
    """Negative Laplace REML (Wood 2011) for a single shared ``lambda``, all neurons.

    The population shares ``lambda``, so the restricted marginal likelihood factorizes
    over neurons and the objective is the sum of the per-neuron terms
    ``-loglik + penalty - 0.5 * penalty_rank * log(lambda) + 0.5 * logdet(H)``.
    Minimized over ``log_penalty``.
    """
    penalty = np.exp(log_penalty)
    coeffs, eta, mu = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty
    )
    loglik = np.sum(counts * eta - mu, axis=0)  # (n_neurons,)
    penalty_term = 0.5 * penalty * np.sum(penalty_weights[:, None] * coeffs**2, axis=0)

    hessian = _assemble_hessian(basis, mu, penalty * penalty_weights)
    _, logdet_hessian = np.linalg.slogdet(hessian)  # (n_neurons,)

    penalty_rank = _penalty_rank(penalty_weights)
    reml = (
        -loglik + penalty_term - 0.5 * penalty_rank * log_penalty + 0.5 * logdet_hessian
    )
    return float(reml.sum())


def select_penalty_by_reml(
    counts: np.ndarray,
    occupancy: np.ndarray,
    basis: np.ndarray,
    penalty_weights: np.ndarray,
) -> float:
    """Select a single shared smoothing parameter ``lambda`` by REML.

    Minimizes :func:`mrf_reml_objective` over ``log(lambda)`` on a bounded interval
    (deterministic; no random state). Returns the selected ``lambda``.
    """
    result = scipy.optimize.minimize_scalar(
        mrf_reml_objective,
        args=(counts, occupancy, basis, penalty_weights),
        bounds=_LOG_PENALTY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(np.exp(result.x))


def fit_sorted_spikes_mrf_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    weights: np.ndarray | None = None,
    sampling_frequency: int = 500,
    rank: int | None = None,
    penalty: float | None = None,
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
        Number of smoothest eigenmodes used as the basis. None (default) uses the
        mgcv-style reduced-rank regime ``min(n_interior_bins, 250)`` (REML tunes
        smoothness within the basis); pass an explicit int to override. Must be an
        explicit parameter so a user-supplied ``rank`` is not dropped by the base
        class's signature filter.
    penalty : float or None, optional
        The smoothing parameter ``lambda``. None (default) selects it by REML. Pass
        ``0.0`` for an unpenalized (saturated) fit.
    disable_progress_bar : bool, optional

    Returns
    -------
    encoding_model : dict
        The same keys as ``sorted_spikes_diffusion`` (``environment``, ``occupancy``,
        ``mean_rates``, ``place_fields`` [FULL-GRID], ``no_spike_part_log_likelihood``,
        ``is_track_interior``, ``node_order``, ``bin_sizes``, ``disable_progress_bar``).
    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = np.asarray(weights)

    _, node_order, bin_sizes = environment_graph(environment)
    # Default to the mgcv-style reduced-rank regime rather than a full dense fit.
    effective_rank = (
        rank if rank is not None else min(node_order.shape[0], _DEFAULT_MAX_RANK)
    )
    penalty_weights, basis = cached_eigenbasis(environment, effective_rank)

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

    if penalty is None and counts.shape[1] > 0:
        penalty = select_penalty_by_reml(
            counts, occupancy_field, basis, penalty_weights
        )
    # `penalty is None` only survives here for the zero-neuron case; a caller's
    # explicit penalty=0.0 (unpenalized) must be respected, not treated as falsy.
    _, eta, _ = mrf_penalized_poisson_fit(
        counts,
        occupancy_field,
        basis,
        penalty_weights,
        penalty if penalty is not None else 1.0,
    )
    rate_interior = np.exp(eta)  # (n_interior, n_neurons); eta is clipped in the fit

    place_fields = np.zeros((counts.shape[1], n_total_bins))
    for neuron in range(counts.shape[1]):
        place_fields[neuron, node_order] = np.clip(rate_interior[:, neuron], EPS, None)

    place_fields = jnp.asarray(place_fields)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "environment": environment,
        "occupancy": occupancy_field,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "node_order": node_order,
        "bin_sizes": bin_sizes,
        "disable_progress_bar": disable_progress_bar,
    }
