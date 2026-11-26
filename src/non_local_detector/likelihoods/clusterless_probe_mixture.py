"""Clusterless probe-geometry structured mixture likelihood.

Designed for high-density multielectrode probes (polymer shanks, silicon
probes, Neuropixels) while falling back cleanly to dense groups (tetrodes) when
contact coordinates are not supplied.

Each electrode's joint mark intensity is a Gaussian mixture over marks with a
Poisson GLM spatial rate map per component:

    lambda_e(x, m) = sum_k lambda_ek(x) N(m; mu_ek, Sigma_ek) + beta q_e(m),

where every component ``k`` has:

- a local contact support ``S_k`` of at most ``max_local_channels`` contacts
  within ``neighborhood_radius`` of its peak contact (inferred automatically
  from contact pitch when ``contact_positions`` is given; all channels when
  ``contact_positions`` is None);
- a factor-analysis-plus-noise covariance ``Sigma_ek = Psi_e + U_ek U_ek^T``
  with ``U_ek`` of rank at most ``covariance_rank`` supported strictly on
  ``S_k`` (using a symmetric ZCA whitener when measured ``noise_covariances``
  are supplied, and estimating a shared per-channel diagonal residual variance
  ``Psi_e`` otherwise; ``covariance_rank=None`` uses full covariance on
  ``S_k``);
- a Poisson GLM rate map ``lambda_ek(x)`` on ``sorted_spikes_glm``'s spline
  basis (branch splines on track graphs, picklable tensor splines in 1-D/2-D).

In ``Psi_e``-standardized coordinates, each component's covariance on ``S_k``
is stored in orthonormal eigenbasis form ``(I - Q_k Q_k^T) + Q_k diag(lam_k)
Q_k^T``, which evaluates the exact full-channel Mahalanobis distance in
``O(|S_k| * r)`` time as a pure sum of squares without catastrophic float32
cancellation, while preserving exact normalization over all ``C`` channels so
``mark_reference`` and ``No-Spike`` remain unit-consistent.

Fitting uses probe-tiled local initialization merged into one whole-shank
mixture followed by warm-started generalized EM (joint position-mark
responsibilities, exact monotone factor-analysis mark M-step, and warm-started
damped Newton GLM rate-map updates on a cached position-cell basis), with
blocked inner-validation early stopping and exact iteration-0 fallback. At
prediction time, an optional ``max_candidate_components`` execution setting
builds a static per-peak-contact lookup table for sub-linear scoring on
Neuropixels-scale component counts.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from non_local_detector._warnings import user_stacklevel
from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import DataQualityWarning, ValidationError
from non_local_detector.likelihoods.clusterless_gmm import DEFAULT_GMM_REG_COVAR
from non_local_detector.likelihoods.clusterless_mark_glm import (
    _MIN_COMPONENT_SPIKES,
    MarkMixture,
    fit_mark_mixture,
)
from non_local_detector.likelihoods.clusterless_spikes import (
    ElectrodeSpikes,
    add_background_only,
    add_electrode_spike_terms,
)
from non_local_detector.likelihoods.common import (
    LOG_RATE_EPS_HZ,
    SpikeTimeOrder,
    as_array,
    as_arrays,
    deterministic_row_add,
    deterministic_row_sum,
    environment_grid_fingerprint,
    get_position_at_time,
    log_bin_duration_evidence,
    pad_rows,
    padded_spike_count,
    prepare_decode_rows,
    require_position,
    select_spike_rows,
    snap_off_track_positions,
    spike_row_ids,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_positive_int,
    validate_spike_feature_pair,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.encoding_model import ClusterlessEncodingModel
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_rate,
)
from non_local_detector.likelihoods.probe_geometry import (
    DEFAULT_MAX_LOCAL_CHANNELS,
    ProbeGeometry,
    build_probe_tiles,
    compute_local_neighborhoods,
    compute_zca_whitener,
    resolve_probe_geometry,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    GLM_LOG_RATE_PRIOR_STD,
    TensorSplineBasis,
    TrackGraphBasis,
    fit_prepared_rate_maps,
    glm_rates,
    make_spline_predict_matrix,
    prepare_rate_map_basis,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    decode_bin_centers,
    requires_time_edges,
)

__all__ = [
    "ClusterlessProbeMixtureEncodingModel",
    "ProbeGeometry",
    "ProbeMarkMixture",
    "fit_clusterless_probe_mixture_encoding_model",
    "predict_clusterless_probe_mixture_log_likelihood",
]

# Spikes processed per block during EM E-steps to bound device workspace.
_E_STEP_BLOCK = 4096
# Target maximum elements in the (block, k_cand, n_bins) candidate gather tensor
# (~16 MiB in float32) so 2-D grids stay memory-bounded.
_CANDIDATE_GATHER_ELEMENTS = 1 << 22
# Minimum responsibility mass before a component keeps its previous parameters
# during intermediate EM iterations.
_MIN_RESPONSIBILITY_MASS = 1e-8
# Default low-rank covariance rank per component on local support.
DEFAULT_COVARIANCE_RANK = 4


def _power_of_two(n: int) -> int:
    """Smallest power of two >= max(n, 1)."""
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def _validate_nonnegative_int(name: str, value: Any) -> int:
    """Reject values that are not non-negative integers."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        msg = f"{name} must be a non-negative integer"
        raise ValidationError(msg, expected="int >= 0", got=repr(value))
    return int(value)


def _validate_execution_settings(
    spike_block_size: Any, max_candidate_components: Any
) -> None:
    """Validate prediction execution settings at both fit and predict."""
    validate_positive_int("spike_block_size", spike_block_size)
    validate_positive_int(
        "max_candidate_components", max_candidate_components, allow_none=True
    )


def mark_reference_scale(
    mark_reg_covar: float = DEFAULT_GMM_REG_COVAR, **_fit_params: object
) -> float:
    """Mark-kernel width the reference adds in quadrature to each feature's spread.

    Parameters
    ----------
    mark_reg_covar : float, optional
        Covariance regularization floor in squared mark units.
    **_fit_params
        Other fit parameters, ignored.

    Returns
    -------
    scale : float
        ``sqrt(mark_reg_covar)``.
    """
    return float(np.sqrt(mark_reg_covar))


@dataclass(frozen=True, eq=False)
class ProbeMarkMixture:
    """A fitted probe-localized low-rank-plus-noise Gaussian mark mixture.

    Each component ``k`` lives on local channels ``support_indices[k]`` (masked
    by ``support_mask[k]``) in noise-whitened coordinates ``y = m @ W`` (or
    ``y = m`` when ``noise_whitener`` is None). Outside ``S_k``, component ``k``
    has mean ``background_mean`` and diagonal variance ``noise_variance``. On
    ``S_k``, its ``noise_variance``-standardized covariance is
    ``(I - Q_k Q_k^T) + Q_k diag(basis_variances[k]) Q_k^T``, where ``Q_k =
    basis[k]`` has orthonormal columns on the valid entries of ``S_k``.

    Attributes
    ----------
    weights : np.ndarray, shape (n_components,)
        Component mixing weights summing to 1.
    means : np.ndarray, shape (n_components, c_loc)
        Local component means in noise-whitened coordinates (0 on padded slots).
    basis : np.ndarray, shape (n_components, c_loc, rank)
        Orthonormal local covariance eigenvectors in ``noise_variance``-
        standardized coordinates (0 on padded channels and unused rank slots).
    basis_variances : np.ndarray, shape (n_components, rank)
        Positive eigenvalues along ``basis`` (1.0 on unused rank slots).
    support_indices : np.ndarray, shape (n_components, c_loc)
        int32 channel indices for each component's local support (0 on padded).
    support_mask : np.ndarray, shape (n_components, c_loc)
        Boolean mask; True for real channels in ``S_k``, False for padding.
    noise_variance : np.ndarray, shape (n_channels,)
        Shared per-channel residual variance in noise-whitened coordinates.
    background_mean : np.ndarray, shape (n_channels,)
        Silent-channel mean in noise-whitened coordinates.
    noise_whitener : np.ndarray or None, shape (n_channels, n_channels)
        Symmetric ZCA whitener ``Sigma_noise^{-1/2}``, or None.
    log_det_whitener : float
        ``log |det noise_whitener|`` (0.0 when ``noise_whitener`` is None).
    contact_positions : np.ndarray or None, shape (n_channels, n_probe_dims)
        Physical contact coordinates if supplied, for candidate lookup tables.
    component_peak_channels : np.ndarray, shape (n_components,)
        int32 peak contact index of each component, for candidate lookup tables.
    """

    weights: np.ndarray
    means: np.ndarray
    basis: np.ndarray
    basis_variances: np.ndarray
    support_indices: np.ndarray
    support_mask: np.ndarray
    noise_variance: np.ndarray
    background_mean: np.ndarray
    noise_whitener: np.ndarray | None
    log_det_whitener: float
    contact_positions: np.ndarray | None
    component_peak_channels: np.ndarray

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return len(self.weights)

    @property
    def n_channels(self) -> int:
        """Total number of channels (features) modeled."""
        return len(self.noise_variance)

    @property
    def local_width(self) -> int:
        """Padded local support width ``C_loc``."""
        return int(self.support_indices.shape[1])

    @property
    def basis_rank(self) -> int:
        """Stored basis rank ``R``."""
        return int(self.basis.shape[2])

    @property
    def all_full_support(self) -> bool:
        """Whether every component covers every channel without truncation."""
        return bool(
            self.support_indices.shape[1] == len(self.noise_variance)
            and np.all(self.support_mask)
        )

    def whiten_marks(self, marks: np.ndarray) -> np.ndarray:
        """Transform raw marks into noise-whitened float64 coordinates."""
        arr = np.asarray(marks, dtype=np.float64)
        if self.noise_whitener is None:
            return arr
        return cast(np.ndarray, arr @ self.noise_whitener)

    def peak_channels(self, marks: np.ndarray) -> np.ndarray:
        """Peak contact index per spike relative to silent background mean."""
        y = self.whiten_marks(marks)
        return cast(
            np.ndarray,
            np.argmax(np.abs(y - self.background_mean[None, :]), axis=1).astype(
                np.int32
            ),
        )

    def candidate_table(self, max_candidates: int) -> tuple[np.ndarray, np.ndarray]:
        """Build static per-contact candidate component indices and validity mask.

        Parameters
        ----------
        max_candidates : int
            Maximum candidate components scored per spike.

        Returns
        -------
        candidates : np.ndarray of int32, shape (n_channels, k_cand)
            Component indices ranked closest to each contact (or by descending
            weight when ``contact_positions`` is None).
        candidate_mask : np.ndarray of bool, shape (n_channels, k_cand)
            True for real components.
        """
        n_comp = self.n_components
        n_ch = self.n_channels
        k_cand = min(int(max_candidates), n_comp)
        if self.contact_positions is not None:
            comp_pos = self.contact_positions[self.component_peak_channels]
            diff = self.contact_positions[:, None, :] - comp_pos[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=-1))  # (C, K)
            candidates = np.empty((n_ch, k_cand), dtype=np.int32)
            neg_weights = -self.weights
            indices = np.arange(n_comp)
            for c in range(n_ch):
                order = np.lexsort((indices, neg_weights, dist[c]))
                candidates[c] = order[:k_cand]
        else:
            order = np.lexsort((np.arange(n_comp), -self.weights))[:k_cand]
            candidates = np.broadcast_to(
                order.astype(np.int32)[None, :], (n_ch, k_cand)
            ).copy()
        return candidates, np.ones((n_ch, k_cand), dtype=bool)

    def log_component_density(self, marks: np.ndarray) -> np.ndarray:
        """Evaluate ``log N(m; mu_k, Sigma_k)`` per spike and component in float64.

        Parameters
        ----------
        marks : np.ndarray, shape (n_spikes, n_channels)
            Spike waveform marks in original units.

        Returns
        -------
        log_density : np.ndarray, shape (n_spikes, n_components)
            Exact Gaussian log-density for each spike and component.
        """
        y = self.whiten_marks(marks)
        n_spikes, n_ch = y.shape
        n_comp = self.n_components
        inv_std = 1.0 / np.sqrt(self.noise_variance)
        log_norm = (
            -0.5
            * (n_ch * np.log(2.0 * np.pi) + float(np.sum(np.log(self.noise_variance))))
            + self.log_det_whitener
        )
        log_det_basis = np.sum(np.log(self.basis_variances), axis=1)
        inv_basis_var = 1.0 / self.basis_variances

        all_full = self.all_full_support
        if not all_full:
            bg_std = (y - self.background_mean[None, :]) * inv_std[None, :]
        else:
            bg_std = np.empty((0, 0), dtype=np.float64)

        out = np.empty((n_spikes, n_comp), dtype=np.float64)
        for k in range(n_comp):
            mask_k = self.support_mask[k]
            ch_k = self.support_indices[k, mask_k]
            mu_k = self.means[k, mask_k]
            q_k = self.basis[k, mask_k, :]
            delta = (y[:, ch_k] - mu_k[None, :]) * inv_std[None, ch_k]
            proj = delta @ q_k
            resid = delta - proj @ q_k.T
            d2_local = np.sum(resid * resid, axis=1) + np.sum(
                (proj * proj) * inv_basis_var[k][None, :], axis=1
            )
            if all_full:
                d2_out = 0.0
            else:
                out_mask = np.ones(n_ch, dtype=bool)
                out_mask[ch_k] = False
                d2_out = np.sum(bg_std[:, out_mask] ** 2, axis=1)
            out[:, k] = log_norm - 0.5 * (log_det_basis[k] + d2_local + d2_out)
        return out

    def responsibilities(self, marks: np.ndarray) -> np.ndarray:
        """Posterior component responsibilities ``p(k | m)``, rows summing to 1."""
        log_joint = (
            self.log_component_density(marks)
            + np.log(np.maximum(self.weights, 1e-300))[None, :]
        )
        return cast(
            np.ndarray,
            np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True)),
        )


def _eigendecompose_local_covariance(
    cov: np.ndarray,
    local_noise_var: np.ndarray,
    *,
    covariance_rank: int | None,
    max_rank: int,
    full_rank_floor: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor a local covariance ``cov`` into orthonormal ``(Q, lam)`` in noise units.

    Parameters
    ----------
    cov : np.ndarray, shape (m, m)
        Symmetric positive semi-definite sample covariance on the ``m`` valid
        channels of component ``k``.
    local_noise_var : np.ndarray, shape (m,)
        Diagonal residual noise variance ``Psi_{S_k}`` on those channels.
    covariance_rank : int or None
        Requested factor rank; None (or ``>= m``) keeps all ``m`` eigenmodes.
    max_rank : int
        Padded output rank ``R >= min(m, covariance_rank or m)``.
    full_rank_floor : float or None
        When not None and full-rank mode applies (``covariance_rank is None``
        or ``covariance_rank >= m``), adds ``full_rank_floor * I`` to ``cov``
        and allows arbitrary positive eigenvalues ``lam > 0`` (exact unconstrained
        full covariance). Otherwise applies the Tipping-Bishop factor-analysis
        floor ``lam >= 1.0`` (i.e. ``Sigma_k = Psi + U_k U_k^T``).

    Returns
    -------
    q_padded : np.ndarray, shape (m, max_rank)
        Orthonormal columns in ``local_noise_var``-standardized coordinates.
    lam_padded : np.ndarray, shape (max_rank,)
        Positive eigenvalues (1.0 on unused columns).
    """
    m = cov.shape[0]
    q_out = np.zeros((m, max_rank), dtype=np.float64)
    lam_out = np.ones(max_rank, dtype=np.float64)
    is_full = covariance_rank is None or covariance_rank >= m
    inv_sqrt = 1.0 / np.sqrt(local_noise_var)

    if is_full and full_rank_floor is not None:
        # Exact constrained Gaussian MLE under Sigma >= full_rank_floor * I:
        # clamp eigenvalues in the coordinate system where the floor is defined
        # rather than shifting eigenvalues that already exceed the floor.
        sym = 0.5 * (cov + cov.T)
        raw_vals, raw_vecs = np.linalg.eigh(sym)
        clamped_vals = np.maximum(raw_vals, float(full_rank_floor))
        if np.allclose(local_noise_var, 1.0, rtol=0.0, atol=1e-12):
            order = np.argsort(clamped_vals)[::-1]
            q_out[:, :m] = raw_vecs[:, order]
            lam_out[:m] = clamped_vals[order]
            return q_out, lam_out
        sym_clamped = (raw_vecs * clamped_vals[None, :]) @ raw_vecs.T
        std_cov = (inv_sqrt[:, None] * sym_clamped) * inv_sqrt[None, :]
        std_cov = 0.5 * (std_cov + std_cov.T)
        eigvals, eigvecs = np.linalg.eigh(std_cov)
        order = np.argsort(eigvals)[::-1]
        q_out[:, :m] = eigvecs[:, order]
        lam_out[:m] = np.maximum(eigvals[order], 1e-12)
        return q_out, lam_out

    r_eff = m if covariance_rank is None else min(int(covariance_rank), m)
    sym = 0.5 * (cov + cov.T)
    std_cov = (inv_sqrt[:, None] * sym) * inv_sqrt[None, :]
    std_cov = 0.5 * (std_cov + std_cov.T)
    eigvals, eigvecs = np.linalg.eigh(std_cov)
    order = np.argsort(eigvals)[::-1][:r_eff]
    top_vals = np.maximum(eigvals[order], 1.0)
    top_vecs = eigvecs[:, order]
    q_out[:, :r_eff] = top_vecs
    lam_out[:r_eff] = top_vals
    return q_out, lam_out


def _mark_mixture_to_probe_mixture(
    mixture: MarkMixture,
    *,
    n_channels: int,
    support_indices: np.ndarray,
    support_mask: np.ndarray,
    covariance_rank: int | None,
    mark_reg_covar: float,
    noise_whitener: np.ndarray | None,
    log_det_whitener: float,
    contact_positions: np.ndarray | None,
    marks_whitened: np.ndarray,
    spike_weights: np.ndarray,
    initial_background: np.ndarray | None = None,
) -> ProbeMarkMixture:
    """Convert a single-group ``MarkMixture`` to a ``ProbeMarkMixture``."""
    n_comp = mixture.n_components
    c_loc = support_indices.shape[1]
    all_full = bool(c_loc == n_channels and np.all(support_mask))
    is_unconstrained_full = all_full and (
        covariance_rank is None or covariance_rank >= n_channels
    )
    init_bg = (
        np.zeros(n_channels, dtype=np.float64)
        if initial_background is None
        else np.asarray(initial_background, dtype=np.float64)
    )

    if is_unconstrained_full:
        # Exact full-covariance conversion matching MarkMixture without inversion
        # via eigendecomposition of the precision matrix P_k = L_k @ L_k.T.
        # Reorder mean and eigenvector rows to match each component's
        # support_indices[k] channel ordering (which geometry orders by distance).
        rank = n_channels
        noise_var = np.ones(n_channels, dtype=np.float64)
        bg_mean = np.zeros(n_channels, dtype=np.float64)
        means = np.zeros((n_comp, c_loc), dtype=np.float64)
        basis = np.zeros((n_comp, c_loc, rank), dtype=np.float64)
        basis_variances = np.ones((n_comp, rank), dtype=np.float64)
        peaks = np.zeros(n_comp, dtype=np.int32)
        for k in range(n_comp):
            ch_k = support_indices[k]
            means[k] = mixture.means[k, ch_k]
            peaks[k] = int(np.argmax(np.abs(mixture.means[k] - init_bg)))
            prec_chol = mixture.precisions_cholesky[k]
            precision = prec_chol @ prec_chol.T
            precision = 0.5 * (precision + precision.T)
            tau, eigvecs = np.linalg.eigh(precision)
            # Smallest precision eigenvalue = largest variance eigenvalue.
            lam = 1.0 / np.maximum(tau, 1e-30)
            if noise_whitener is not None:
                lam = np.maximum(lam, 1.0)
            order = np.argsort(lam)[::-1]
            basis[k] = eigvecs[ch_k][:, order]
            basis_variances[k] = lam[order]
        return ProbeMarkMixture(
            weights=np.array(mixture.weights, dtype=np.float64),
            means=means,
            basis=basis,
            basis_variances=basis_variances,
            support_indices=support_indices,
            support_mask=support_mask,
            noise_variance=noise_var,
            background_mean=bg_mean,
            noise_whitener=noise_whitener,
            log_det_whitener=log_det_whitener,
            contact_positions=contact_positions,
            component_peak_channels=peaks,
        )

    # Low-rank factor-analysis (or local-support) initialization from fitted
    # covariances. Reconstruct each component covariance from precision Cholesky.
    covs: list[np.ndarray] = []
    for k in range(n_comp):
        prec_chol = mixture.precisions_cholesky[k]
        precision = 0.5 * (prec_chol @ prec_chol.T + (prec_chol @ prec_chol.T).T)
        tau, eigvecs = np.linalg.eigh(precision)
        lam = 1.0 / np.maximum(tau, 1e-30)
        cov_k = (eigvecs * lam[None, :]) @ eigvecs.T
        covs.append(0.5 * (cov_k + cov_k.T))

    rank = c_loc if covariance_rank is None else min(int(covariance_rank), c_loc)
    effective_reg = 1.0 if noise_whitener is not None else mark_reg_covar
    bg_mean = init_bg.copy()
    if noise_whitener is not None:
        noise_var = np.ones(n_channels, dtype=np.float64)
    else:
        # Estimate initial shared per-channel residual variance from component
        # tail variances (or global weighted variance), floored by mark_reg_covar.
        w_sum = max(float(np.sum(spike_weights)), 1e-12)
        diag_accum = np.zeros(n_channels, dtype=np.float64)
        for k in range(n_comp):
            eigs = np.linalg.eigvalsh(covs[k])
            tail_count = max(1, n_channels - rank)
            tail_var = float(np.mean(eigs[:tail_count]))
            diag_accum += mixture.weights[k] * np.minimum(np.diag(covs[k]), tail_var)
        if not np.all(np.isfinite(diag_accum)) or w_sum <= 0.0:
            noise_var = np.full(n_channels, mark_reg_covar, dtype=np.float64)
        else:
            noise_var = np.maximum(diag_accum, mark_reg_covar)

    means = np.zeros((n_comp, c_loc), dtype=np.float64)
    basis = np.zeros((n_comp, c_loc, rank), dtype=np.float64)
    basis_variances = np.ones((n_comp, rank), dtype=np.float64)
    peaks = np.zeros(n_comp, dtype=np.int32)
    full_floor = effective_reg if covariance_rank is None else None

    for k in range(n_comp):
        mask_k = support_mask[k]
        ch_k = support_indices[k, mask_k]
        means[k, mask_k] = mixture.means[k, ch_k]
        peaks[k] = int(np.argmax(np.abs(mixture.means[k] - bg_mean)))
        q_k, lam_k = _eigendecompose_local_covariance(
            covs[k][np.ix_(ch_k, ch_k)],
            noise_var[ch_k],
            covariance_rank=covariance_rank,
            max_rank=rank,
            full_rank_floor=full_floor,
        )
        basis[k, mask_k, :] = q_k
        basis_variances[k, :] = lam_k

    # For local-support models, initialize silent-channel background mean and
    # noise variance directly from the data outside each component's support.
    if not all_full and len(marks_whitened) > 0:
        total_w = max(float(np.sum(spike_weights)), 1e-12)
        # Rough assignment by nearest component peak channel (baseline-relative).
        spike_peaks = np.argmax(np.abs(marks_whitened - bg_mean[None, :]), axis=1)
        if contact_positions is not None:
            comp_pos = contact_positions[peaks]
            d_sc = np.sum(
                (contact_positions[spike_peaks][:, None, :] - comp_pos[None, :, :])
                ** 2,
                axis=-1,
            )
            assign = np.argmin(d_sc, axis=1)
        else:
            assign = np.zeros(len(marks_whitened), dtype=np.int32)
        in_support = np.zeros((n_comp, n_channels), dtype=bool)
        for k in range(n_comp):
            in_support[k, support_indices[k, support_mask[k]]] = True
        silent = ~in_support[assign]  # (N, C)
        silent_w = spike_weights[:, None] * silent
        silent_sum = np.sum(silent_w, axis=0)
        has_silent = silent_sum > _MIN_RESPONSIBILITY_MASS
        bg_mean = np.where(
            has_silent,
            np.sum(silent_w * marks_whitened, axis=0)
            / np.where(has_silent, silent_sum, 1.0),
            bg_mean,
        )
        if noise_whitener is None:
            silent_var = np.sum(
                silent_w * (marks_whitened - bg_mean[None, :]) ** 2, axis=0
            )
            active_frac = np.maximum(0.0, 1.0 - silent_sum / total_w)
            combined_var = silent_var / total_w + active_frac * noise_var
            noise_var = np.maximum(combined_var, mark_reg_covar)

    return ProbeMarkMixture(
        weights=np.array(mixture.weights, dtype=np.float64),
        means=means,
        basis=basis,
        basis_variances=basis_variances,
        support_indices=support_indices,
        support_mask=support_mask,
        noise_variance=noise_var,
        background_mean=bg_mean,
        noise_whitener=noise_whitener,
        log_det_whitener=log_det_whitener,
        contact_positions=contact_positions,
        component_peak_channels=peaks,
    )


def _allocate_tile_components(
    tile_weights: np.ndarray, total_components: int
) -> np.ndarray:
    """Budget ``total_components`` across active tiles proportionally to weight.

    Every active tile receives at least 1 component so no populated region of
    the shank is dropped. When ``total_components >= len(tile_weights)``, the
    remaining budget is distributed by largest-remainder (Hamilton) allocation
    so the sum equals ``total_components`` exactly.
    """
    n_tiles = len(tile_weights)
    if n_tiles == 0:
        return np.zeros(0, dtype=np.int64)
    if total_components <= n_tiles:
        return np.ones(n_tiles, dtype=np.int64)
    rem = int(total_components) - n_tiles
    total_w = float(np.sum(tile_weights))
    shares = (
        tile_weights / total_w * rem
        if total_w > 0.0
        else np.full(n_tiles, rem / n_tiles, dtype=np.float64)
    )
    floors = np.floor(shares).astype(np.int64)
    leftover = rem - int(np.sum(floors))
    if leftover > 0:
        remainders = shares - floors
        order = np.lexsort((np.arange(n_tiles), -tile_weights, -remainders))
        floors[order[:leftover]] += 1
    return np.ones(n_tiles, dtype=np.int64) + floors


def _estimate_channel_baseline(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Estimate per-channel silent baseline robust to DC offsets and 2-unit splits.

    For unimodal or majority-silent channels (including constant DC offsets on
    silent contacts and dense probes with >= 3 units), returns the weighted
    median. For strongly bimodal channels (e.g., a 2-unit shank where one unit
    accounts for ~45-55% of spikes), selects the tighter-spread (or closer-to-
    zero) mode rather than landing between modes or on the firing mode.
    """
    n_spikes, n_channels = y.shape
    if n_spikes == 0:
        return np.zeros(n_channels, dtype=np.float64)
    if n_spikes == 1:
        return np.zeros(n_channels, dtype=np.float64)
    total_w = float(np.sum(weights))
    if total_w <= 0.0:
        return np.zeros(n_channels, dtype=np.float64)
    probs = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], dtype=np.float64)
    baseline = np.empty(n_channels, dtype=np.float64)
    for c in range(n_channels):
        vals = y[:, c]
        order = np.argsort(vals, kind="mergesort")
        sv = vals[order]
        sw = weights[order]
        cdf = (np.cumsum(sw) - 0.5 * sw) / total_w
        q12, q25, q37, q50, q62, q75, q87 = np.interp(probs, cdf, sv)
        iqr_low = float(q37 - q12)
        iqr_high = float(q87 - q62)
        if float(q75 - q25) > 2.5 * max(iqr_low, iqr_high, 1e-12):
            if iqr_low < 0.85 * iqr_high:
                baseline[c] = float(q25)
            elif iqr_high < 0.85 * iqr_low:
                baseline[c] = float(q75)
            else:
                baseline[c] = float(q25) if abs(q25) <= abs(q75) else float(q75)
        else:
            baseline[c] = float(q50)
    return baseline


def _initialize_probe_mixture(
    marks: np.ndarray,
    weights: np.ndarray,
    *,
    contact_positions: np.ndarray | None,
    noise_covariance: np.ndarray | None,
    neighborhood_radius: float | None,
    max_local_channels: int | None,
    covariance_rank: int | None,
    n_mark_components: int,
    mark_weight_concentration: float | None,
    mark_reg_covar: float,
    mark_fit_max_spikes: int,
    mark_max_iter: int,
    mark_tol: float,
    mark_random_state: int,
) -> ProbeMarkMixture:
    """Initialize a ``ProbeMarkMixture`` for one electrode (tiled or single-group)."""
    marks = np.asarray(marks, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights > 0.0
    marks, weights = marks[positive], weights[positive]
    n_spikes, n_channels = marks.shape

    if noise_covariance is not None:
        whitener, log_det_whitener = compute_zca_whitener(noise_covariance)
        y = marks @ whitener
        # In ZCA-whitened coordinates, background noise covariance is I_C (variance 1.0).
        effective_reg_covar = 1.0
    else:
        whitener = None
        log_det_whitener = 0.0
        y = marks
        effective_reg_covar = mark_reg_covar

    if contact_positions is not None:
        nbr_indices, nbr_mask, radius = compute_local_neighborhoods(
            contact_positions,
            neighborhood_radius=neighborhood_radius,
            max_local_channels=max_local_channels,
        )
        c_loc = int(nbr_indices.shape[1])
        tiles, tile_supports = build_probe_tiles(
            contact_positions, radius=radius, max_local_channels=max_local_channels
        )
    else:
        nbr_indices = np.broadcast_to(
            np.arange(n_channels, dtype=np.int32)[None, :], (n_channels, n_channels)
        ).copy()
        nbr_mask = np.ones((n_channels, n_channels), dtype=bool)
        c_loc = n_channels
        tiles = [np.arange(n_channels, dtype=np.int32)]
        tile_supports = [np.arange(n_channels, dtype=np.int32)]

    has_silent_channels = not (c_loc == n_channels and np.all(nbr_mask))

    # Single-spike electrode: one component centered on its mark with covariance
    # effective_reg_covar * I (matching clusterless_mark_glm when unwhitened).
    if n_spikes == 1:
        peak = int(np.argmax(np.abs(y[0])))
        sup_idx = nbr_indices[peak : peak + 1].copy()
        sup_mask = nbr_mask[peak : peak + 1].copy()
        rank = c_loc if covariance_rank is None else min(int(covariance_rank), c_loc)
        means = np.zeros((1, c_loc), dtype=np.float64)
        means[0, sup_mask[0]] = y[0, sup_idx[0, sup_mask[0]]]
        bg_mean = np.zeros(n_channels, dtype=np.float64)
        if not (c_loc == n_channels and np.all(sup_mask)):
            bg_mean = y[0].copy()
        return ProbeMarkMixture(
            weights=np.ones(1, dtype=np.float64),
            means=means,
            basis=np.zeros((1, c_loc, rank), dtype=np.float64),
            basis_variances=np.ones((1, rank), dtype=np.float64),
            support_indices=sup_idx,
            support_mask=sup_mask,
            noise_variance=np.full(n_channels, effective_reg_covar, dtype=np.float64),
            background_mean=bg_mean,
            noise_whitener=whitener,
            log_det_whitener=log_det_whitener,
            contact_positions=contact_positions,
            component_peak_channels=np.array([peak], dtype=np.int32),
        )

    init_bg = (
        _estimate_channel_baseline(y, weights)
        if has_silent_channels
        else np.zeros(n_channels, dtype=np.float64)
    )

    # Single-tile path (no geometry, or electrode fits in one local neighborhood).
    if len(tiles) == 1:
        base_mixture = fit_mark_mixture(
            y,
            weights,
            n_components=n_mark_components,
            weight_concentration_prior=mark_weight_concentration,
            reg_covar=effective_reg_covar,
            max_spikes=mark_fit_max_spikes,
            max_iter=mark_max_iter,
            tol=mark_tol,
            random_state=mark_random_state,
        )
        peaks = np.argmax(np.abs(base_mixture.means - init_bg[None, :]), axis=1).astype(
            np.int32
        )
        if contact_positions is not None:
            sup_idx = nbr_indices[peaks].copy()
            sup_mask = nbr_mask[peaks].copy()
        else:
            sup_idx = np.broadcast_to(
                np.arange(n_channels, dtype=np.int32)[None, :],
                (base_mixture.n_components, n_channels),
            ).copy()
            sup_mask = np.ones((base_mixture.n_components, n_channels), dtype=bool)
        return _mark_mixture_to_probe_mixture(
            base_mixture,
            n_channels=n_channels,
            support_indices=sup_idx,
            support_mask=sup_mask,
            covariance_rank=covariance_rank,
            mark_reg_covar=mark_reg_covar,
            noise_whitener=whitener,
            log_det_whitener=log_det_whitener,
            contact_positions=contact_positions,
            marks_whitened=y,
            spike_weights=weights,
            initial_background=init_bg,
        )

    # Multi-tile dense shank: route spikes by baseline-relative peak contact to
    # disjoint tiles, budget n_mark_components across active tiles proportionally
    # to spike weight, fit local DP-VB mixtures on each tile's contact support,
    # and merge into one unified whole-shank ProbeMarkMixture.
    contact_to_tile = np.empty(n_channels, dtype=np.int32)
    for tile_idx, members in enumerate(tiles):
        contact_to_tile[members] = tile_idx

    spike_peaks = np.argmax(np.abs(y - init_bg[None, :]), axis=1)
    spike_tiles = contact_to_tile[spike_peaks]

    total_weight = float(np.sum(weights))
    active_tiles: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for tile_idx, support_ch in enumerate(tile_supports):
        in_tile = spike_tiles == tile_idx
        if not np.any(in_tile):
            continue
        tile_w = weights[in_tile]
        tile_weight_sum = float(np.sum(tile_w))
        if tile_weight_sum <= 0.0:
            continue
        active_tiles.append(
            (tile_idx, support_ch, y[in_tile][:, support_ch], tile_w, tile_weight_sum)
        )

    tile_budgets = _allocate_tile_components(
        np.asarray([entry[4] for entry in active_tiles], dtype=np.float64),
        n_mark_components,
    )

    comp_weights: list[float] = []
    comp_peaks: list[int] = []
    comp_means_local: list[np.ndarray] = []
    comp_covs_local: list[np.ndarray] = []
    comp_sup_idx: list[np.ndarray] = []
    comp_sup_mask: list[np.ndarray] = []

    for (tile_idx, support_ch, tile_y, tile_w, tile_weight_sum), tile_k in zip(
        active_tiles, tile_budgets, strict=True
    ):
        tile_mix = fit_mark_mixture(
            tile_y,
            tile_w,
            n_components=int(tile_k),
            weight_concentration_prior=mark_weight_concentration,
            reg_covar=effective_reg_covar,
            max_spikes=mark_fit_max_spikes,
            max_iter=mark_max_iter,
            tol=mark_tol,
            random_state=mark_random_state + tile_idx,
        )
        # Map tile-support index -> channel index, and look up permanent S_k.
        support_lookup = {int(ch): pos for pos, ch in enumerate(support_ch)}
        tile_bg = init_bg[support_ch]
        for k in range(tile_mix.n_components):
            mean_tile = tile_mix.means[k]
            peak_pos = int(np.argmax(np.abs(mean_tile - tile_bg)))
            peak_ch = int(support_ch[peak_pos])
            s_idx = nbr_indices[peak_ch].copy()
            s_mask = nbr_mask[peak_ch].copy()

            # Reconstruct covariance on support_ch from precision Cholesky.
            prec_chol = tile_mix.precisions_cholesky[k]
            precision = prec_chol @ prec_chol.T
            precision = 0.5 * (precision + precision.T)
            tau, eigvecs = np.linalg.eigh(precision)
            cov_tile = (eigvecs * (1.0 / np.maximum(tau, 1e-30))[None, :]) @ eigvecs.T
            cov_tile = 0.5 * (cov_tile + cov_tile.T)

            # Extract/project component mean & covariance onto its permanent S_k.
            valid_pos = np.flatnonzero(s_mask)
            n_valid = len(valid_pos)
            mu_loc = np.zeros(c_loc, dtype=np.float64)
            mu_loc[valid_pos] = init_bg[s_idx[valid_pos]]
            cov_loc = np.eye(n_valid, dtype=np.float64) * effective_reg_covar
            # Identify which channels in S_k were in support_ch.
            mapped_local: list[int] = []
            mapped_tile: list[int] = []
            for j, pos_idx in enumerate(valid_pos):
                ch = int(s_idx[pos_idx])
                if ch in support_lookup:
                    mapped_local.append(j)
                    mapped_tile.append(support_lookup[ch])
                    mu_loc[pos_idx] = mean_tile[support_lookup[ch]]
            if mapped_local:
                ml = np.asarray(mapped_local, dtype=int)
                mt = np.asarray(mapped_tile, dtype=int)
                cov_loc[np.ix_(ml, ml)] = cov_tile[np.ix_(mt, mt)]

            comp_weights.append(
                float(tile_mix.weights[k]) * (tile_weight_sum / total_weight)
            )
            comp_peaks.append(peak_ch)
            comp_means_local.append(mu_loc)
            comp_covs_local.append(cov_loc)
            comp_sup_idx.append(s_idx)
            comp_sup_mask.append(s_mask)

    n_comp = len(comp_weights)
    w_arr = np.asarray(comp_weights, dtype=np.float64)
    w_arr = w_arr / np.sum(w_arr)
    sup_idx_arr = np.stack(comp_sup_idx, axis=0)
    sup_mask_arr = np.stack(comp_sup_mask, axis=0)
    means_arr = np.stack(comp_means_local, axis=0)
    peaks_arr = np.asarray(comp_peaks, dtype=np.int32)

    rank = c_loc if covariance_rank is None else min(int(covariance_rank), c_loc)
    # Initialize silent background mean & shared noise variance from spikes
    # outside each spike's own peak neighborhood.
    silent_mask = np.ones((n_channels, n_channels), dtype=bool)
    for c in range(n_channels):
        silent_mask[c, nbr_indices[c, nbr_mask[c]]] = False
    is_silent = silent_mask[spike_peaks]  # (N, C)
    silent_w = weights[:, None] * is_silent
    silent_sum = np.sum(silent_w, axis=0)
    has_silent = silent_sum > _MIN_RESPONSIBILITY_MASS
    bg_mean = np.where(
        has_silent,
        np.sum(silent_w * y, axis=0) / np.where(has_silent, silent_sum, 1.0),
        init_bg,
    )
    if whitener is not None:
        noise_var = np.ones(n_channels, dtype=np.float64)
    else:
        silent_var = np.where(
            has_silent,
            np.sum(silent_w * (y - bg_mean[None, :]) ** 2, axis=0)
            / np.where(has_silent, silent_sum, 1.0),
            mark_reg_covar,
        )
        noise_var = np.maximum(silent_var, mark_reg_covar)

    basis = np.zeros((n_comp, c_loc, rank), dtype=np.float64)
    basis_variances = np.ones((n_comp, rank), dtype=np.float64)
    full_floor = effective_reg_covar if covariance_rank is None else None
    for k in range(n_comp):
        mask_k = sup_mask_arr[k]
        ch_k = sup_idx_arr[k, mask_k]
        q_k, lam_k = _eigendecompose_local_covariance(
            comp_covs_local[k],
            noise_var[ch_k],
            covariance_rank=covariance_rank,
            max_rank=rank,
            full_rank_floor=full_floor,
        )
        basis[k, mask_k, :] = q_k
        basis_variances[k, :] = lam_k

    return ProbeMarkMixture(
        weights=w_arr,
        means=means_arr,
        basis=basis,
        basis_variances=basis_variances,
        support_indices=sup_idx_arr,
        support_mask=sup_mask_arr,
        noise_variance=noise_var,
        background_mean=bg_mean,
        noise_whitener=whitener,
        log_det_whitener=log_det_whitener,
        contact_positions=contact_positions,
        component_peak_channels=peaks_arr,
    )


def _outside_support_mask(
    support_indices: jax.Array,
    support_mask: jax.Array,
    n_channels: int,
    dtype: Any,
) -> jax.Array:
    """Return ``(K, C)`` mask with 1.0 on channels outside ``S_k`` and 0.0 inside."""
    n_comp = support_indices.shape[0]
    in_sup = jnp.zeros((n_comp, n_channels + 1), dtype=dtype)
    idx = jnp.where(support_mask, support_indices, n_channels)
    k_idx = jnp.arange(n_comp)[:, None]
    in_sup = in_sup.at[k_idx, idx].set(jnp.asarray(1.0, dtype=dtype))
    return jnp.asarray(1.0, dtype=dtype) - in_sup[:, :n_channels]


def _woodbury_log_density(
    features: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    *,
    log_det_whitener: float = 0.0,
    all_full_support: bool = False,
) -> jax.Array:
    """Evaluate exact Gaussian log-densities via cancellation-free orthonormal Woodbury.

    Parameters
    ----------
    features : jax.Array, shape (n_spikes, n_channels)
        Noise-whitened spike waveform features.
    means : jax.Array, shape (n_components, c_loc)
        Local component means on ``support_indices``.
    basis : jax.Array, shape (n_components, c_loc, rank)
        Orthonormal local covariance basis in ``noise_variance``-standardized
        coordinates (zero on masked contacts and padded rank columns).
    basis_variances : jax.Array, shape (n_components, rank)
        Positive eigenvalues along ``basis`` (1.0 on padded columns).
    support_indices : jax.Array, shape (n_components, c_loc)
        int32 local channel indices.
    support_mask : jax.Array, shape (n_components, c_loc)
        Boolean local support mask.
    noise_variance : jax.Array, shape (n_channels,)
        Per-channel residual variance ``Psi``.
    background_mean : jax.Array, shape (n_channels,)
        Per-channel silent mean ``mu_bg``.
    is_component : jax.Array, shape (n_components,)
        Boolean validity mask (False for padded components).
    log_det_whitener : float, optional
        ``log |det noise_whitener|``, by default 0.0.
    all_full_support : bool, optional
        If True, skips the outside-support quadratic term (identically zero).

    Returns
    -------
    log_density : jax.Array, shape (n_spikes, n_components)
        Log-density per spike and component; ``-inf`` for padded components or
        overflowing marks.
    """
    dtype = features.dtype
    n_channels = features.shape[1]
    inv_std = jnp.reciprocal(jnp.sqrt(noise_variance))
    local_inv_std = inv_std[support_indices]

    y_local = features[:, support_indices]  # (B, K, C_loc)
    delta = jnp.where(
        support_mask[None, :, :],
        (y_local - means[None, :, :]) * local_inv_std[None, :, :],
        jnp.asarray(0.0, dtype=dtype),
    )
    proj = jnp.einsum("bkc,kcr->bkr", delta, basis, precision=jax.lax.Precision.HIGHEST)
    resid = delta - jnp.einsum(
        "bkr,kcr->bkc", proj, basis, precision=jax.lax.Precision.HIGHEST
    )
    inv_basis_var = jnp.reciprocal(basis_variances)
    d2_local = jnp.sum(resid * resid, axis=2) + jnp.sum(
        (proj * proj) * inv_basis_var[None, :, :], axis=2
    )

    if all_full_support:
        d2_outside: jax.Array | float = jnp.asarray(0.0, dtype=dtype)
    else:
        bg_std = (features - background_mean[None, :]) * inv_std[None, :]
        bg_sq = bg_std * bg_std  # (B, C)
        outside_mask = _outside_support_mask(
            support_indices, support_mask, n_channels, dtype
        )  # (K, C)
        safe_bg_sq = jnp.where(
            jnp.isfinite(bg_sq), bg_sq, jnp.asarray(0.0, dtype=dtype)
        )
        has_inf_out = (
            jnp.matmul(
                (~jnp.isfinite(bg_sq)).astype(dtype),
                outside_mask.T,
                precision=jax.lax.Precision.HIGHEST,
            )
            > 0.0
        )
        d2_outside = jnp.where(
            has_inf_out,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.matmul(safe_bg_sq, outside_mask.T, precision=jax.lax.Precision.HIGHEST),
        )

    log_norm = jnp.asarray(
        -0.5 * n_channels * np.log(2.0 * np.pi) + log_det_whitener, dtype=dtype
    ) - jnp.asarray(0.5, dtype=dtype) * jnp.sum(jnp.log(noise_variance))
    log_det_basis = jnp.sum(jnp.log(basis_variances), axis=1)
    log_density = log_norm - jnp.asarray(0.5, dtype=dtype) * (
        log_det_basis[None, :] + d2_local + d2_outside
    )
    valid = is_component[None, :] & jnp.isfinite(log_density)
    return jnp.where(valid, log_density, jnp.asarray(-jnp.inf, dtype=dtype))


def _woodbury_candidate_log_density(
    features: jax.Array,
    candidate_indices: jax.Array,
    candidate_mask: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    *,
    log_det_whitener: float = 0.0,
    all_full_support: bool = False,
) -> jax.Array:
    """Evaluate exact Woodbury log-densities only on per-spike candidate components.

    Parameters
    ----------
    features : jax.Array, shape (n_spikes, n_channels)
        Noise-whitened spike marks.
    candidate_indices : jax.Array, shape (n_spikes, k_cand)
        Candidate component indices per spike.
    candidate_mask : jax.Array, shape (n_spikes, k_cand)
        Boolean validity mask for candidate slots.
    means, basis, basis_variances, support_indices, support_mask, noise_variance, background_mean, is_component
        Padded component arrays (see :func:`_woodbury_log_density`).
    log_det_whitener : float, optional
        ``log |det noise_whitener|``.
    all_full_support : bool, optional
        Whether every component covers every channel.

    Returns
    -------
    log_density : jax.Array, shape (n_spikes, k_cand)
    """
    dtype = features.dtype
    n_channels = features.shape[1]
    inv_std = jnp.reciprocal(jnp.sqrt(noise_variance))

    c_means = means[candidate_indices]  # (B, k_cand, C_loc)
    c_basis = basis[candidate_indices]  # (B, k_cand, C_loc, R)
    c_var = basis_variances[candidate_indices]  # (B, k_cand, R)
    c_sup = support_indices[candidate_indices]  # (B, k_cand, C_loc)
    c_mask = support_mask[candidate_indices]  # (B, k_cand, C_loc)
    c_valid = is_component[candidate_indices] & candidate_mask  # (B, k_cand)
    c_inv_std = inv_std[c_sup]  # (B, k_cand, C_loc)

    y_local = jnp.take_along_axis(
        features[:, None, :], c_sup, axis=2
    )  # (B, k_cand, C_loc)
    delta = jnp.where(
        c_mask, (y_local - c_means) * c_inv_std, jnp.asarray(0.0, dtype=dtype)
    )
    proj = jnp.einsum(
        "bjc,bjcr->bjr", delta, c_basis, precision=jax.lax.Precision.HIGHEST
    )
    resid = delta - jnp.einsum(
        "bjr,bjcr->bjc", proj, c_basis, precision=jax.lax.Precision.HIGHEST
    )
    d2_local = jnp.sum(resid * resid, axis=2) + jnp.sum(
        (proj * proj) * jnp.reciprocal(c_var), axis=2
    )

    if all_full_support:
        d2_outside: jax.Array | float = jnp.asarray(0.0, dtype=dtype)
    else:
        bg_std = (features - background_mean[None, :]) * inv_std[None, :]
        bg_sq = bg_std * bg_std  # (B, C)
        outside_mask = _outside_support_mask(
            support_indices, support_mask, n_channels, dtype
        )
        c_outside = outside_mask[candidate_indices]  # (B, k_cand, C)
        safe_bg_sq = jnp.where(
            jnp.isfinite(bg_sq), bg_sq, jnp.asarray(0.0, dtype=dtype)
        )
        has_inf_out = (
            jnp.einsum(
                "bc,bjc->bj",
                (~jnp.isfinite(bg_sq)).astype(dtype),
                c_outside,
                precision=jax.lax.Precision.HIGHEST,
            )
            > 0.0
        )
        d2_outside = jnp.where(
            has_inf_out,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.einsum(
                "bc,bjc->bj", safe_bg_sq, c_outside, precision=jax.lax.Precision.HIGHEST
            ),
        )

    log_norm = jnp.asarray(
        -0.5 * n_channels * np.log(2.0 * np.pi) + log_det_whitener, dtype=dtype
    ) - jnp.asarray(0.5, dtype=dtype) * jnp.sum(jnp.log(noise_variance))
    log_det_basis = jnp.sum(jnp.log(c_var), axis=2)
    log_density = log_norm - jnp.asarray(0.5, dtype=dtype) * (
        log_det_basis + d2_local + d2_outside
    )
    return jnp.where(
        c_valid & jnp.isfinite(log_density),
        log_density,
        jnp.asarray(-jnp.inf, dtype=dtype),
    )


@partial(jax.jit, static_argnames=("all_full_support",))
def _e_step_block(
    features: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    log_rates_left: jax.Array,
    log_rates_right: jax.Array,
    left_share: jax.Array,
    right_share: jax.Array,
    log_det_whitener: jax.Array,
    *,
    all_full_support: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split-position responsibilities and weighted spike log-likelihood for one block."""
    dtype = features.dtype
    log_dens = _woodbury_log_density(
        features,
        means,
        basis,
        basis_variances,
        support_indices,
        support_mask,
        noise_variance,
        background_mean,
        is_component,
        log_det_whitener=0.0,
        all_full_support=all_full_support,
    ) + log_det_whitener.astype(dtype)
    log_dens = jnp.where(
        is_component[None, :] & jnp.isfinite(log_dens),
        log_dens,
        jnp.asarray(-jnp.inf, dtype=dtype),
    )
    joint_l = log_rates_left + log_dens
    log_tot_l = jax.scipy.special.logsumexp(joint_l, axis=1)
    safe_l = jnp.where(
        jnp.isfinite(log_tot_l), log_tot_l, jnp.asarray(0.0, dtype=dtype)
    )
    resp_l = jnp.where(
        jnp.isfinite(log_tot_l)[:, None],
        jnp.exp(joint_l - safe_l[:, None]),
        jnp.asarray(0.0, dtype=dtype),
    )

    joint_r = log_rates_right + log_dens
    log_tot_r = jax.scipy.special.logsumexp(joint_r, axis=1)
    safe_r = jnp.where(
        jnp.isfinite(log_tot_r), log_tot_r, jnp.asarray(0.0, dtype=dtype)
    )
    resp_r = jnp.where(
        jnp.isfinite(log_tot_r)[:, None],
        jnp.exp(joint_r - safe_r[:, None]),
        jnp.asarray(0.0, dtype=dtype),
    )

    weighted_ll = jnp.sum(
        jnp.where((left_share > 0) & jnp.isfinite(log_tot_l), left_share * safe_l, 0.0)
        + jnp.where(
            (right_share > 0) & jnp.isfinite(log_tot_r), right_share * safe_r, 0.0
        )
    )
    return resp_l, resp_r, weighted_ll


def _e_step_electrode(
    marks_whitened: np.ndarray,
    mixture: ProbeMarkMixture,
    log_rates_left: np.ndarray,
    log_rates_right: np.ndarray,
    left_share: np.ndarray,
    right_share: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact split-position responsibilities (float64) and weighted spike LL."""
    n_spikes = len(marks_whitened)
    n_comp = mixture.n_components
    all_full = mixture.all_full_support
    means = jnp.asarray(mixture.means, dtype=jnp.float32)
    basis = jnp.asarray(mixture.basis, dtype=jnp.float32)
    basis_var = jnp.asarray(mixture.basis_variances, dtype=jnp.float32)
    sup_idx = jnp.asarray(mixture.support_indices, dtype=jnp.int32)
    sup_mask = jnp.asarray(mixture.support_mask)
    noise_var = jnp.asarray(mixture.noise_variance, dtype=jnp.float32)
    bg_mean = jnp.asarray(mixture.background_mean, dtype=jnp.float32)
    is_comp = jnp.ones((n_comp,), dtype=bool)
    log_det_w = jnp.asarray(mixture.log_det_whitener, dtype=jnp.float32)

    resp_l_out = np.empty((n_spikes, n_comp), dtype=np.float64)
    resp_r_out = np.empty((n_spikes, n_comp), dtype=np.float64)
    total_ll = 0.0
    block = min(_E_STEP_BLOCK, max(n_spikes, 1))
    for start in range(0, n_spikes, block):
        stop = min(start + block, n_spikes)
        n_cur = stop - start
        n_pad = max(n_cur, block)
        y_blk = jnp.asarray(
            pad_rows(marks_whitened[start:stop], n_pad), dtype=jnp.float32
        )
        lrl_blk = jnp.asarray(
            pad_rows(log_rates_left[start:stop], n_pad), dtype=jnp.float32
        )
        lrr_blk = jnp.asarray(
            pad_rows(log_rates_right[start:stop], n_pad), dtype=jnp.float32
        )
        ls_blk = jnp.asarray(pad_rows(left_share[start:stop], n_pad), dtype=jnp.float32)
        rs_blk = jnp.asarray(
            pad_rows(right_share[start:stop], n_pad), dtype=jnp.float32
        )
        rl_blk, rr_blk, ll_blk = _e_step_block(
            y_blk,
            means,
            basis,
            basis_var,
            sup_idx,
            sup_mask,
            noise_var,
            bg_mean,
            is_comp,
            lrl_blk,
            lrr_blk,
            ls_blk,
            rs_blk,
            log_det_w,
            all_full_support=all_full,
        )
        resp_l_out[start:stop] = np.asarray(rl_blk[:n_cur], dtype=np.float64)
        resp_r_out[start:stop] = np.asarray(rr_blk[:n_cur], dtype=np.float64)
        total_ll += float(ll_blk)
    return resp_l_out, resp_r_out, total_ll


def _mark_m_step(
    marks_whitened: np.ndarray,
    weights: np.ndarray,
    responsibilities: np.ndarray,
    previous: ProbeMarkMixture,
    *,
    covariance_rank: int | None,
    mark_reg_covar: float,
) -> ProbeMarkMixture:
    """Exact monotone ECM mark M-step for fixed local supports ``S_k``.

    Updates silent ``background_mean``, local means ``mu_{k, S_k}``, shared
    diagonal noise ``Psi`` (when ``noise_whitener`` is None, via exact
    constrained FA-EM), and orthonormal basis factors ``(Q_k, lam_k)`` via
    Tipping-Bishop eigendecomposition (or full covariance when unconstrained).
    """
    n_comp = previous.n_components
    n_ch = previous.n_channels
    c_loc = previous.local_width
    rank = previous.basis_rank
    all_full = previous.all_full_support
    is_unconstrained_full = all_full and (
        covariance_rank is None or covariance_rank >= n_ch
    )

    gamma = weights[:, None] * responsibilities  # (N, K)
    comp_mass = np.sum(gamma, axis=0)  # (K,)
    total_mass = max(float(np.sum(comp_mass)), _MIN_RESPONSIBILITY_MASS)
    new_weights = np.maximum(comp_mass, 1e-12)
    new_weights = new_weights / np.sum(new_weights)

    # 1. Update silent-channel background mean (exact closed form) when C_loc < C.
    if all_full:
        bg_mean = previous.background_mean.copy()
        silent_weight_per_spike = np.empty((0, 0), dtype=np.float64)
    else:
        in_support = np.zeros((n_comp, n_ch), dtype=np.float64)
        for k in range(n_comp):
            in_support[k, previous.support_indices[k, previous.support_mask[k]]] = 1.0
        active_w = gamma @ in_support  # (N, C)
        silent_weight_per_spike = np.maximum(0.0, weights[:, None] - active_w)
        silent_sum = np.sum(silent_weight_per_spike, axis=0)
        has_silent = silent_sum > _MIN_RESPONSIBILITY_MASS
        bg_mean = np.where(
            has_silent,
            np.sum(silent_weight_per_spike * marks_whitened, axis=0)
            / np.where(has_silent, silent_sum, 1.0),
            previous.background_mean,
        )

    # 2. Update local component means and sample covariances C_k on S_k.
    means = previous.means.copy()
    sample_covs: list[np.ndarray | None] = []
    for k in range(n_comp):
        mask_k = previous.support_mask[k]
        ch_k = previous.support_indices[k, mask_k]
        n_k = float(comp_mass[k])
        if n_k <= _MIN_RESPONSIBILITY_MASS:
            sample_covs.append(None)
            continue
        g_k = gamma[:, k]
        y_k = marks_whitened[:, ch_k]
        mu_k = (g_k @ y_k) / n_k
        means[k, mask_k] = mu_k
        diff = y_k - mu_k[None, :]
        cov_k = (diff * g_k[:, None]).T @ diff / n_k
        sample_covs.append(0.5 * (cov_k + cov_k.T))

    # 3. Unconstrained full-covariance case (all channels, full rank).
    effective_reg = 1.0 if previous.noise_whitener is not None else mark_reg_covar
    if is_unconstrained_full:
        noise_var = np.ones(n_ch, dtype=np.float64)
        basis = np.zeros((n_comp, c_loc, rank), dtype=np.float64)
        basis_var = np.ones((n_comp, rank), dtype=np.float64)
        for k in range(n_comp):
            cov_k = sample_covs[k]
            if cov_k is None:
                basis[k] = previous.basis[k]
                basis_var[k] = previous.basis_variances[k]
                continue
            q_k, lam_k = _eigendecompose_local_covariance(
                cov_k,
                noise_var,
                covariance_rank=None,
                max_rank=rank,
                full_rank_floor=effective_reg,
            )
            basis[k] = q_k
            basis_var[k] = lam_k
        return ProbeMarkMixture(
            weights=new_weights,
            means=means,
            basis=basis,
            basis_variances=basis_var,
            support_indices=previous.support_indices,
            support_mask=previous.support_mask,
            noise_variance=noise_var,
            background_mean=bg_mean,
            noise_whitener=previous.noise_whitener,
            log_det_whitener=previous.log_det_whitener,
            contact_positions=previous.contact_positions,
            component_peak_channels=previous.component_peak_channels,
        )

    # 4. Factor-analysis / local-support ECM step.
    # Step 4a: Tipping-Bishop update at old Psi (guarantees exact FA-EM residual).
    psi_old = previous.noise_variance
    full_floor = effective_reg if covariance_rank is None else None
    basis = np.zeros((n_comp, c_loc, rank), dtype=np.float64)
    basis_var = np.ones((n_comp, rank), dtype=np.float64)
    for k in range(n_comp):
        mask_k = previous.support_mask[k]
        ch_k = previous.support_indices[k, mask_k]
        cov_k = sample_covs[k]
        if cov_k is None:
            basis[k] = previous.basis[k]
            basis_var[k] = previous.basis_variances[k]
            continue
        q_k, lam_k = _eigendecompose_local_covariance(
            cov_k,
            psi_old[ch_k],
            covariance_rank=covariance_rank,
            max_rank=rank,
            full_rank_floor=full_floor,
        )
        basis[k, mask_k, :] = q_k
        basis_var[k, :] = lam_k

    # Step 4b: Constrained FA-EM update for shared diagonal Psi >= mark_reg_covar
    # when noise_whitener is None (and covariance_rank is finite, or C_loc < C).
    if previous.noise_whitener is None:
        residual_ss = np.zeros(n_ch, dtype=np.float64)
        if not all_full:
            residual_ss += np.sum(
                silent_weight_per_spike * (marks_whitened - bg_mean[None, :]) ** 2,
                axis=0,
            )
        for k in range(n_comp):
            mask_k = previous.support_mask[k]
            ch_k = previous.support_indices[k, mask_k]
            cov_k = sample_covs[k]
            n_k = float(comp_mass[k])
            if cov_k is None or n_k <= _MIN_RESPONSIBILITY_MASS:
                continue
            if full_floor is not None:
                # Local full-rank components have arbitrary covariance on S_k;
                # only silent channels outside S_k inform Psi.
                residual_ss[ch_k] += n_k * psi_old[ch_k]
                continue
            q_k = basis[k, mask_k, :]
            lam_k = basis_var[k, :]
            psi_k = psi_old[ch_k]
            # Exact FA-EM expected squared residual at the Tipping-Bishop optimum:
            # e_j = psi_j * ((C_std)_jj - sum_a (lam_a - 1) Q_{j, a}^2) >= 0.
            c_std_diag = np.diag(cov_k) / psi_k
            explained = np.sum((lam_k - 1.0)[None, :] * (q_k * q_k), axis=1)
            e_k = psi_k * np.maximum(0.0, c_std_diag - explained)
            residual_ss[ch_k] += n_k * e_k

        psi_new = np.maximum(residual_ss / total_mass, mark_reg_covar)

        # Step 4c: Re-project factors via Tipping-Bishop at Psi_new (monotone).
        for k in range(n_comp):
            mask_k = previous.support_mask[k]
            ch_k = previous.support_indices[k, mask_k]
            cov_k = sample_covs[k]
            if cov_k is None:
                continue
            q_k, lam_k = _eigendecompose_local_covariance(
                cov_k,
                psi_new[ch_k],
                covariance_rank=covariance_rank,
                max_rank=rank,
                full_rank_floor=full_floor,
            )
            basis[k, mask_k, :] = q_k
            basis_var[k, :] = lam_k
    else:
        psi_new = psi_old

    return ProbeMarkMixture(
        weights=new_weights,
        means=means,
        basis=basis,
        basis_variances=basis_var,
        support_indices=previous.support_indices,
        support_mask=previous.support_mask,
        noise_variance=psi_new,
        background_mean=bg_mean,
        noise_whitener=previous.noise_whitener,
        log_det_whitener=previous.log_det_whitener,
        contact_positions=previous.contact_positions,
        component_peak_channels=previous.component_peak_channels,
    )


class _ElectrodeEvents(NamedTuple):
    """Positive-weight encoding spikes for one electrode."""

    marks_whitened: np.ndarray
    left: np.ndarray
    right: np.ndarray
    left_share: np.ndarray
    right_share: np.ndarray
    total_weight: np.ndarray
    primary_sample: np.ndarray


def _collect_electrode_events(
    support: EncodingSupport,
    weights: np.ndarray,
    support_position: np.ndarray,
    electrode_times: np.ndarray,
    electrode_marks: np.ndarray,
    *,
    allowed_samples: np.ndarray | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
):
    """Extract positive-weight encoding spikes and their bracketing shares.

    When ``allowed_samples`` is supplied (for inner validation folds), each
    spike is first assigned to a single owning tracking sample (the endpoint
    with the larger full-data share, left on ties) and kept only when that
    owner lies in ``allowed_samples``. Any cross-boundary minor share on the
    non-allowed side is folded onto the owning endpoint so total spike weight
    is preserved and no spike can enter both training and validation.
    """
    spike_index, left, right, fraction = support.event_basis(electrode_times)
    validate_finite(electrode_marks[spike_index], "spike_waveform_features")
    left_share = (1.0 - fraction) * weights[left]
    right_share = fraction * weights[right]
    spike_weight = left_share + right_share
    positive = spike_weight > 0.0
    if not np.any(positive):
        return None
    sp_idx = spike_index[positive]
    l_orig = left[positive]
    r_orig = right[positive]
    l_sh = left_share[positive]
    r_sh = right_share[positive]
    owner_orig = np.where(
        (r_sh <= 0.0) | ((l_sh >= r_sh) & (l_sh > 0.0)), l_orig, r_orig
    )

    if allowed_samples is not None:
        owned = allowed_samples[owner_orig]
        if not np.any(owned):
            return None
        sp_idx = sp_idx[owned]
        l_orig = l_orig[owned]
        r_orig = r_orig[owned]
        l_sh = l_sh[owned]
        r_sh = r_sh[owned]
        l_in = allowed_samples[l_orig]
        r_in = allowed_samples[r_orig]
        total_sh = l_sh + r_sh
        l_sh = np.where(l_in & r_in, l_sh, np.where(l_in, total_sh, 0.0))
        r_sh = np.where(l_in & r_in, r_sh, np.where(r_in, total_sh, 0.0))

    l_pos = support_position[l_orig]
    r_pos = support_position[r_orig]
    primary = np.where((r_pos < 0) | ((l_pos >= 0) & (l_sh >= r_sh)), l_pos, r_pos)
    return (
        electrode_marks[sp_idx],
        l_pos,
        r_pos,
        l_sh,
        r_sh,
        primary,
    )


def _soft_cell_counts(
    events_list: list[_ElectrodeEvents],
    responsibilities_list: list[tuple[np.ndarray, np.ndarray]],
    sample_cells: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Sum split-position responsibility-weighted spike shares into position cells."""
    rows: list[np.ndarray] = []
    for events, (resp_l, resp_r) in zip(
        events_list, responsibilities_list, strict=True
    ):
        n_comp = resp_l.shape[1]
        counts = np.zeros(n_cells * n_comp, dtype=np.float64)
        for sample, share, resp in (
            (events.left, events.left_share, resp_l),
            (events.right, events.right_share, resp_r),
        ):
            inside = (sample >= 0) & (share > 0.0)
            if not np.any(inside):
                continue
            index = (
                sample_cells[sample[inside]][:, None] * n_comp
                + np.arange(n_comp)[None, :]
            )
            counts += np.bincount(
                index.ravel(),
                weights=(share[inside, None] * resp[inside]).ravel(),
                minlength=n_cells * n_comp,
            )
        rows.append(counts.reshape(n_cells, n_comp).T)
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, n_cells))


def _contiguous_exposure_blocks(
    exposure_weights: np.ndarray, n_blocks: int
) -> list[np.ndarray]:
    """Split positive-exposure samples into ``n_blocks`` contiguous equal-exposure masks."""
    positive_idx = np.flatnonzero(exposure_weights > 0.0)
    if positive_idx.size < n_blocks or n_blocks < 2:
        return []
    cum = np.cumsum(exposure_weights[positive_idx])
    total = float(cum[-1])
    if total <= 0.0:
        return []
    block_id = np.minimum(np.floor(cum / total * n_blocks).astype(int), n_blocks - 1)
    masks: list[np.ndarray] = []
    for b in range(n_blocks):
        mask = np.zeros(len(exposure_weights), dtype=bool)
        mask[positive_idx[block_id == b]] = True
        if np.any(mask):
            masks.append(mask)
    return masks if len(masks) >= 2 else []


class _EMRun:
    """Lockstep generalized-EM state for one full or inner-validation fit."""

    def __init__(
        self,
        *,
        environment: Environment,
        interior_place_bin_centers: jax.Array,
        support: EncodingSupport,
        support_position: np.ndarray,
        encoding_positions: np.ndarray,
        spike_times: list[np.ndarray],
        spike_marks: list[np.ndarray],
        contact_positions: list[np.ndarray | None],
        noise_covariances: list[np.ndarray | None],
        weights: np.ndarray,
        exposure_weights: np.ndarray,
        init_kwargs: dict[str, Any],
        emission_knot_spacing: float,
        log_rate_prior_std: float,
        covariance_rank: int | None,
        mark_reg_covar: float,
        warn_empty: bool,
        train_sample_mask: np.ndarray | None = None,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.covariance_rank = covariance_rank
        self.mark_reg_covar = mark_reg_covar
        weight_sum = float(exposure_weights.sum())
        self.prepared = prepare_rate_map_basis(
            environment,
            environment.place_bin_edges_,
            interior_place_bin_centers,
            encoding_positions,
            exposure_weights[support.indices],
            emission_knot_spacing=emission_knot_spacing,
            log_rate_prior_std=log_rate_prior_std,
        )

        self.mark_mixtures: list[ProbeMarkMixture | None] = []
        self.mean_rates: list[float] = []
        self.fitted_indices: list[int] = []
        self.fitted: list[ProbeMarkMixture] = []
        self.events: list[_ElectrodeEvents] = []
        self.spike_cells_left: list[np.ndarray] = []
        self.spike_cells_right: list[np.ndarray] = []

        for electrode, (times, marks, pos_c, noise_c) in enumerate(
            zip(
                spike_times,
                spike_marks,
                contact_positions,
                noise_covariances,
                strict=True,
            )
        ):
            raw = _collect_electrode_events(
                support,
                weights,
                support_position,
                times,
                marks,
                allowed_samples=train_sample_mask,
            )
            if raw is None:
                if warn_empty:
                    warnings.warn(
                        f"Clusterless probe mixture: electrode {electrode} has no "
                        "effective encoding spikes (empty or zero total weight); it "
                        "is treated as zero-rate -- its observed decode spikes are "
                        "explained by the background rate.",
                        DataQualityWarning,
                        stacklevel=user_stacklevel(),
                    )
                self.mark_mixtures.append(None)
                self.mean_rates.append(0.0)
                continue
            e_marks, left, right, l_sh, r_sh, primary = raw
            e_weight = l_sh + r_sh
            self.mean_rates.append(weighted_mean_rate(e_weight, weight_sum))
            mixture = _initialize_probe_mixture(
                e_marks,
                e_weight,
                contact_positions=pos_c,
                noise_covariance=noise_c,
                covariance_rank=covariance_rank,
                mark_reg_covar=mark_reg_covar,
                **init_kwargs,
            )
            self.mark_mixtures.append(mixture)
            self.fitted_indices.append(electrode)
            self.fitted.append(mixture)
            self.events.append(
                _ElectrodeEvents(
                    marks_whitened=mixture.whiten_marks(e_marks),
                    left=left,
                    right=right,
                    left_share=l_sh,
                    right_share=r_sh,
                    total_weight=e_weight,
                    primary_sample=primary,
                )
            )
            if self.prepared.sample_cells.size:
                self.spike_cells_left.append(
                    self.prepared.sample_cells[np.maximum(left, 0)]
                )
                self.spike_cells_right.append(
                    self.prepared.sample_cells[np.maximum(right, 0)]
                )
            else:
                zeros = np.zeros(len(primary), dtype=np.int64)
                self.spike_cells_left.append(zeros)
                self.spike_cells_right.append(zeros)

        self.total_spike_weight = sum(
            float(np.sum(ev.total_weight)) for ev in self.events
        )
        # Iteration 0: exact two-stage fit (position-free mark responsibilities,
        # identical on both bracketing endpoints).
        init_resp: list[tuple[np.ndarray, np.ndarray]] = []
        for mixture, ev in zip(self.fitted, self.events, strict=True):
            r0 = (
                mixture.responsibilities(ev.marks_whitened)
                if mixture.noise_whitener is None
                else _responsibilities_whitened(mixture, ev.marks_whitened)
            )
            init_resp.append((r0, r0))
        self._fit_rate_maps(init_resp, warm_start=False, max_steps=None, warn=True)
        self.train_log_likelihood: list[float] = []
        self.validation_nats: list[float] = []
        self.validation_log_likelihood: list[float] = []
        self._responsibilities: list[tuple[np.ndarray, np.ndarray]] = init_resp

        # Prepare validation spikes & validation cell exposure if requested.
        self._val_data: dict[str, Any] | None = None
        if validation is not None and self.fitted:
            val_w, val_exp, val_sample_mask = validation
            val_events: list[
                tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ] = []
            val_spikes = 0.0
            for fit_idx, electrode in enumerate(self.fitted_indices):
                raw_v = _collect_electrode_events(
                    support,
                    val_w,
                    support_position,
                    spike_times[electrode],
                    spike_marks[electrode],
                    allowed_samples=val_sample_mask,
                )
                if raw_v is None:
                    continue
                v_marks, v_left, v_right, v_l_sh, v_r_sh, _ = raw_v
                val_spikes += float(np.sum(v_l_sh + v_r_sh))
                if self.prepared.sample_cells.size:
                    v_cells_l = self.prepared.sample_cells[np.maximum(v_left, 0)]
                    v_cells_r = self.prepared.sample_cells[np.maximum(v_right, 0)]
                else:
                    v_cells_l = np.zeros(len(v_marks), dtype=np.int64)
                    v_cells_r = v_cells_l
                val_events.append(
                    (fit_idx, v_marks, v_cells_l, v_cells_r, v_l_sh, v_r_sh)
                )
            val_exp_support = val_exp[support.indices]
            val_exp_cells = np.bincount(
                self.prepared.sample_cells,
                weights=val_exp_support,
                minlength=self.prepared.n_position_cells,
            )
            self._val_data = {
                "events": val_events,
                "cell_exposure": val_exp_cells,
                "spikes": val_spikes,
            }

    def _fit_rate_maps(
        self,
        responsibilities: list[tuple[np.ndarray, np.ndarray]],
        *,
        warm_start: bool,
        max_steps: int | None,
        warn: bool,
    ) -> None:
        if not self.fitted:
            n_coef = int(self.prepared.emission_predict_matrix.shape[1])
            n_bins = int(self.prepared.emission_predict_matrix.shape[0])
            self.coefficients = np.zeros((0, n_coef), dtype=np.float32)
            self.interior_rates = np.zeros((n_bins, 0), dtype=np.float32)
            self.cell_rates = np.zeros(
                (self.prepared.n_position_cells, 0), dtype=np.float32
            )
            return
        cell_counts = _soft_cell_counts(
            self.events,
            responsibilities,
            self.prepared.sample_cells,
            self.prepared.n_position_cells,
        )
        init_coef = self.coefficients if warm_start else None
        coef, interior_rates = fit_prepared_rate_maps(
            self.prepared,
            cell_counts,
            initial_coefficients=init_coef,
            max_steps=max_steps,
            warn_on_nonconvergence=warn,
        )
        self.coefficients = np.asarray(coef)
        self.interior_rates = np.asarray(interior_rates)
        self.cell_rates = np.asarray(
            glm_rates(self.prepared.emission_design_matrix, coef)
        )

    def evaluate(self) -> None:
        """Run exact split-position E-step and record penalized & validation LL."""
        if not self.fitted:
            return
        cell_log_rates = np.log(self.cell_rates)  # (n_cells, total_K)
        responsibilities: list[tuple[np.ndarray, np.ndarray]] = []
        spike_ll = 0.0
        first = 0
        for mixture, ev, cells_l, cells_r in zip(
            self.fitted,
            self.events,
            self.spike_cells_left,
            self.spike_cells_right,
            strict=True,
        ):
            k_count = mixture.n_components
            maps = slice(first, first + k_count)
            first += k_count
            lr_left = cell_log_rates[cells_l, maps]
            lr_right = cell_log_rates[cells_r, maps]
            resp_l, resp_r, e_ll = _e_step_electrode(
                ev.marks_whitened,
                mixture,
                lr_left,
                lr_right,
                ev.left_share,
                ev.right_share,
            )
            responsibilities.append((resp_l, resp_r))
            spike_ll += e_ll
        self._responsibilities = responsibilities

        expected_count = float(
            np.asarray(self.cell_rates, dtype=np.float64).sum(axis=1)
            @ np.asarray(self.prepared.cell_exposure, dtype=np.float64)
        )
        l2_term = self.prepared.l2_penalty * float(
            np.sum(np.asarray(self.coefficients[:, 1:], dtype=np.float64) ** 2)
        )
        penalized_ll = spike_ll - expected_count - l2_term
        self.train_log_likelihood.append(
            penalized_ll / max(self.total_spike_weight, 1.0)
        )

        if self._val_data is not None:
            val_ll = 0.0
            offsets: list[slice] = []
            offset = 0
            for mixture in self.fitted:
                offsets.append(slice(offset, offset + mixture.n_components))
                offset += mixture.n_components
            for (
                fit_idx,
                v_marks,
                v_cells_l,
                v_cells_r,
                v_l_sh,
                v_r_sh,
            ) in self._val_data["events"]:
                mixture = self.fitted[fit_idx]
                maps = offsets[fit_idx]
                _, _, e_vll = _e_step_electrode(
                    mixture.whiten_marks(v_marks),
                    mixture,
                    cell_log_rates[v_cells_l, maps],
                    cell_log_rates[v_cells_r, maps],
                    v_l_sh,
                    v_r_sh,
                )
                val_ll += e_vll
            val_expected = float(
                np.asarray(self.cell_rates, dtype=np.float64).sum(axis=1)
                @ np.asarray(self._val_data["cell_exposure"], dtype=np.float64)
            )
            nats = val_ll - val_expected
            self.validation_nats.append(nats)
            self.validation_log_likelihood.append(
                nats / max(float(self._val_data["spikes"]), 1.0)
            )

    def step(self, *, max_glm_steps: int | None, warn_glm: bool) -> None:
        """Apply one generalized-EM M-step (marks + warm-started GLM rate maps)."""
        if not self.fitted:
            return
        updated_fitted: list[ProbeMarkMixture] = []
        for ev, (resp_l, resp_r), mixture in zip(
            self.events, self._responsibilities, self.fitted, strict=True
        ):
            # Combine split-position responsibilities using exact endpoint shares.
            gamma = ev.left_share[:, None] * resp_l + ev.right_share[:, None] * resp_r
            resp_mark = gamma / np.maximum(ev.total_weight[:, None], 1e-300)
            updated_fitted.append(
                _mark_m_step(
                    ev.marks_whitened,
                    ev.total_weight,
                    resp_mark,
                    mixture,
                    covariance_rank=self.covariance_rank,
                    mark_reg_covar=self.mark_reg_covar,
                )
            )
        self.fitted = updated_fitted
        for fit_idx, electrode in enumerate(self.fitted_indices):
            self.mark_mixtures[electrode] = self.fitted[fit_idx]
        self._fit_rate_maps(
            self._responsibilities,
            warm_start=True,
            max_steps=max_glm_steps,
            warn=warn_glm,
        )


def _responsibilities_whitened(
    mixture: ProbeMarkMixture, marks_whitened: np.ndarray
) -> np.ndarray:
    """Responsibilities directly from already-whitened marks (avoids re-whitening)."""
    n_spikes, n_ch = marks_whitened.shape
    n_comp = mixture.n_components
    inv_std = 1.0 / np.sqrt(mixture.noise_variance)
    log_norm = (
        -0.5
        * (n_ch * np.log(2.0 * np.pi) + float(np.sum(np.log(mixture.noise_variance))))
        + mixture.log_det_whitener
    )
    log_det_basis = np.sum(np.log(mixture.basis_variances), axis=1)
    inv_basis_var = 1.0 / mixture.basis_variances
    all_full = mixture.all_full_support
    if not all_full:
        bg_std = (marks_whitened - mixture.background_mean[None, :]) * inv_std[None, :]
    else:
        bg_std = np.empty((0, 0), dtype=np.float64)
    log_dens = np.empty((n_spikes, n_comp), dtype=np.float64)
    for k in range(n_comp):
        mask_k = mixture.support_mask[k]
        ch_k = mixture.support_indices[k, mask_k]
        delta = (marks_whitened[:, ch_k] - mixture.means[k, mask_k][None, :]) * inv_std[
            None, ch_k
        ]
        q_k = mixture.basis[k, mask_k, :]
        proj = delta @ q_k
        resid = delta - proj @ q_k.T
        d2_loc = np.sum(resid * resid, axis=1) + np.sum(
            (proj * proj) * inv_basis_var[k][None, :], axis=1
        )
        if all_full:
            d2_out = 0.0
        else:
            out_mask = np.ones(n_ch, dtype=bool)
            out_mask[ch_k] = False
            d2_out = np.sum(bg_std[:, out_mask] ** 2, axis=1)
        log_dens[:, k] = log_norm - 0.5 * (log_det_basis[k] + d2_loc + d2_out)
    log_joint = log_dens + np.log(np.maximum(mixture.weights, 1e-300))[None, :]
    return cast(
        np.ndarray,
        np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True)),
    )


class ClusterlessProbeMixtureEncodingModel(ClusterlessEncodingModel):
    """The encoding model ``fit_clusterless_probe_mixture_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_clusterless_probe_mixture_log_likelihood``, documented there.
    """

    emission_design_info: TrackGraphBasis | TensorSplineBasis
    mark_mixtures: list[ProbeMarkMixture | None]
    rate_coefficients: list[np.ndarray | jax.Array | None]
    log_rate_maps: list[np.ndarray | None]
    summed_ground_process_intensity: np.ndarray
    spike_block_size: int
    max_candidate_components: int | None
    em_diagnostics: dict[str, Any]


def fit_clusterless_probe_mixture_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    contact_positions: Sequence[ArrayLike | None] | ProbeGeometry | None = None,
    noise_covariances: Sequence[ArrayLike | None] | None = None,
    neighborhood_radius: float | None = None,
    max_local_channels: int | None = DEFAULT_MAX_LOCAL_CHANNELS,
    covariance_rank: int | None = DEFAULT_COVARIANCE_RANK,
    n_mark_components: int = 32,
    mark_weight_concentration: float | None = None,
    mark_reg_covar: float = DEFAULT_GMM_REG_COVAR,
    mark_fit_max_spikes: int = 20_000,
    mark_max_iter: int = 500,
    mark_tol: float = 1e-4,
    mark_random_state: int = 0,
    n_em_iterations: int = 100,
    em_validation_blocks: int = 4,
    em_patience: int = 5,
    em_tol: float = 1e-4,
    em_glm_steps: int = 2,
    emission_knot_spacing: float = np.sqrt(12.5) * 2,
    log_rate_prior_std: float = GLM_LOG_RATE_PRIOR_STD,
    spike_block_size: int = 1000,
    max_candidate_components: int | None = None,
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> ClusterlessProbeMixtureEncodingModel:
    """Fit the probe-localized structured mark-mixture GLM encoding model.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Tracking timestamps in seconds.
    position : array_like, shape (n_time_position, n_position_dims)
        Animal position at each tracking sample.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Encoding spike times per electrode/shank.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_channels)
        Encoding spike waveform features (e.g. peak amplitudes) per electrode.
    environment : Environment
        Fitted spatial environment.
    weights : array_like, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM), by
        default None (uniform).
    contact_positions : sequence of array_like or None, ProbeGeometry, or None, optional
        Per-electrode physical contact coordinates (each shape
        ``(n_channels, n_probe_dims)``), a ``ProbeGeometry`` to split by shank,
        or None (dense full-group support).
    noise_covariances : sequence of array_like or None, or None, optional
        Optional per-electrode measured channel noise covariance matrices (each
        shape ``(n_channels, n_channels)`` SPD). Applied via symmetric ZCA
        whitening.
    neighborhood_radius : float or None, optional
        Physical contact radius for each component's local support ``S_k``.
        By default None (auto-inferred as 3x median nearest-neighbor pitch).
    max_local_channels : int or None, optional
        Maximum contacts in each component's local support ``S_k``, by default
        16. None keeps all contacts within ``neighborhood_radius``.
    covariance_rank : int or None, optional
        Low-rank factor dimension ``r`` per component on ``S_k``, by default 4.
        Automatically uses full covariance whenever ``covariance_rank is None``
        or ``covariance_rank >= len(S_k)`` (e.g. on tetrodes).
    n_mark_components : int, optional
        Upper bound on initial components per electrode (distributed
        proportionally across active probe tiles by spike weight when tiled,
        with at least 1 component per active tile), by default 32.
    mark_weight_concentration : float or None, optional
        Dirichlet-process concentration for initialization, by default None.
    mark_reg_covar : float, optional
        Minimum residual noise variance floor in squared mark units when
        ``noise_covariances`` is not supplied (in ZCA-whitened coordinates the
        physical noise floor is fixed at 1.0), by default
        ``DEFAULT_GMM_REG_COVAR`` (1.0).
    mark_fit_max_spikes : int, optional
        Maximum spikes per electrode/tile used to initialize the mark mixture,
        by default 20,000. All spikes enter EM and rate maps.
    mark_max_iter : int, optional
        Maximum variational iterations during initialization, by default 500.
    mark_tol : float, optional
        Variational bound tolerance per weighted spike during initialization,
        by default 1e-4.
    mark_random_state : int, optional
        Seed for initialization subsampling and k-means, by default 0.
    n_em_iterations : int, optional
        Maximum joint generalized-EM iterations after iteration 0, by default
        100. Set 0 for pure two-stage fitting.
    em_validation_blocks : int, optional
        Number of contiguous equal-exposure inner validation blocks used to
        choose the EM iteration count, by default 4. Set 0 to run directly on
        all encoding data (useful for matched flat refits).
    em_patience : int, optional
        Stop inner validation after this many consecutive iterations without
        validation improvement, by default 5.
    em_tol : float, optional
        Stopping tolerance on penalized training log-likelihood gain per
        weighted spike (nats), by default 1e-4.
    em_glm_steps : int, optional
        Warm-started damped Newton steps per intermediate EM iteration, by
        default 2. Iteration 0 and the final EM iteration solve to full GLM
        convergence.
    emission_knot_spacing : float, optional
        Requested spacing of rate-map spline knots in position units, by
        default ``sqrt(12.5) * 2``.
    log_rate_prior_std : float, optional
        Prior SD (nats) of each component's log-rate deviation from baseline,
        by default ``GLM_LOG_RATE_PRIOR_STD``.
    spike_block_size : int, optional
        Decode spikes scored per block at predict, by default 1000.
    max_candidate_components : int or None, optional
        Maximum candidate components scored per decode spike using a static
        per-peak-contact lookup table, by default None (evaluate all K).
    disable_progress_bar : bool, optional
        Stored for prediction's progress bars, by default False.
    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds, replacing outer bounds of support.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
    _encoding_support : EncodingSupport, optional
        Internal precomputed encoding support from detector.

    Returns
    -------
    encoding_model : ClusterlessProbeMixtureEncodingModel
        Fitted encoding model dict accepted by
        :func:`predict_clusterless_probe_mixture_log_likelihood`.
    """
    _validate_execution_settings(spike_block_size, max_candidate_components)
    validate_positive_int("n_mark_components", n_mark_components)
    validate_positive_int("mark_fit_max_spikes", mark_fit_max_spikes)
    validate_positive_int("mark_max_iter", mark_max_iter)
    validate_positive_int("em_patience", em_patience)
    validate_positive_int("em_glm_steps", em_glm_steps)
    validate_positive_int("covariance_rank", covariance_rank, allow_none=True)
    validate_positive_int("max_local_channels", max_local_channels, allow_none=True)
    n_em_iterations = _validate_nonnegative_int("n_em_iterations", n_em_iterations)
    em_validation_blocks = _validate_nonnegative_int(
        "em_validation_blocks", em_validation_blocks
    )
    if not (np.isfinite(mark_reg_covar) and mark_reg_covar > 0.0):
        msg = "mark_reg_covar must be finite and positive"
        raise ValidationError(msg, expected="> 0", got=repr(mark_reg_covar))
    if not (np.isfinite(mark_tol) and mark_tol > 0.0):
        msg = "mark_tol must be finite and positive"
        raise ValidationError(msg, expected="> 0", got=repr(mark_tol))
    if np.isnan(em_tol):
        msg = "em_tol must not be NaN"
        raise ValidationError(msg, got=repr(em_tol))

    position_time = as_array(position_time)
    spike_times_raw = as_arrays(spike_times)
    spike_marks_raw = as_arrays(spike_waveform_features)
    support, weights_arr, exposure_weights, position_arr = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    validate_population_lengths(
        "electrode",
        spike_times=spike_times_raw,
        spike_waveform_features=spike_marks_raw,
    )
    warn_if_no_encoding_exposure(
        exposure_weights, "Clusterless probe mixture", clusterless=True
    )

    host_times: list[np.ndarray] = []
    host_marks: list[np.ndarray] = []
    mark_dimensions: list[int] = []
    for electrode, (t_e, m_e) in enumerate(
        zip(spike_times_raw, spike_marks_raw, strict=True)
    ):
        validate_spike_feature_pair(t_e, m_e, electrode)
        t_np = np.asarray(t_e, dtype=np.float64)
        m_np = np.asarray(m_e, dtype=np.float64)
        host_times.append(t_np)
        host_marks.append(m_np)
        mark_dimensions.append(int(m_np.shape[1]))

    resolved_positions, resolved_covariances = resolve_probe_geometry(
        contact_positions, noise_covariances, mark_dimensions
    )

    is_track_interior = environment.is_track_interior_flat_
    interior_place_bin_centers = jnp.asarray(
        environment.place_bin_centers_[is_track_interior]
    )
    encoding_positions = environment.decoding_position(position_arr[support.indices])
    support_position = np.full(len(position_arr), -1, dtype=np.int64)
    support_position[support.indices] = np.arange(len(support.indices), dtype=np.int64)

    init_kwargs: dict[str, Any] = {
        "neighborhood_radius": neighborhood_radius,
        "max_local_channels": max_local_channels,
        "n_mark_components": n_mark_components,
        "mark_weight_concentration": mark_weight_concentration,
        "mark_fit_max_spikes": mark_fit_max_spikes,
        "mark_max_iter": mark_max_iter,
        "mark_tol": mark_tol,
        "mark_random_state": mark_random_state,
    }

    run_common: dict[str, Any] = {
        "environment": environment,
        "interior_place_bin_centers": interior_place_bin_centers,
        "support": support,
        "support_position": support_position,
        "encoding_positions": encoding_positions,
        "spike_times": host_times,
        "spike_marks": host_marks,
        "contact_positions": resolved_positions,
        "noise_covariances": resolved_covariances,
        "init_kwargs": init_kwargs,
        "emission_knot_spacing": emission_knot_spacing,
        "log_rate_prior_std": log_rate_prior_std,
        "covariance_rank": covariance_rank,
        "mark_reg_covar": mark_reg_covar,
    }

    # Choose EM iteration count by blocked inner validation when requested.
    validation_trace: list[float] = []
    chosen_iterations = n_em_iterations
    if n_em_iterations > 0 and em_validation_blocks >= 2:
        block_masks = _contiguous_exposure_blocks(
            exposure_weights, em_validation_blocks
        )
        if len(block_masks) >= 2:
            val_runs = [
                _EMRun(
                    **run_common,
                    weights=weights_arr,
                    exposure_weights=np.where(mask, 0.0, exposure_weights),
                    warn_empty=False,
                    train_sample_mask=~mask,
                    validation=(
                        weights_arr,
                        np.where(mask, exposure_weights, 0.0),
                        mask,
                    ),
                )
                for mask in block_masks
            ]
            active_runs = [r for r in val_runs if r.fitted]
            if active_runs:
                for run in active_runs:
                    run.evaluate()
                pooled_spikes = max(
                    sum(
                        float(r._val_data["spikes"])
                        for r in active_runs
                        if r._val_data is not None
                    ),
                    1.0,
                )
                pooled0 = sum(r.validation_nats[-1] for r in active_runs)
                validation_trace.append(pooled0 / pooled_spikes)
                best_iter = 0
                best_val = pooled0
                stale = 0
                for iteration in range(1, n_em_iterations + 1):
                    for run in active_runs:
                        run.step(max_glm_steps=em_glm_steps, warn_glm=False)
                        run.evaluate()
                    pooled = sum(r.validation_nats[-1] for r in active_runs)
                    validation_trace.append(pooled / pooled_spikes)
                    if pooled > best_val + 1e-9:
                        best_val = pooled
                        best_iter = iteration
                        stale = 0
                    else:
                        stale += 1
                    gains = [
                        r.train_log_likelihood[-1] - r.train_log_likelihood[-2]
                        for r in active_runs
                        if len(r.train_log_likelihood) >= 2
                    ]
                    if stale >= em_patience or (
                        gains and all(g < em_tol for g in gains)
                    ):
                        break
                chosen_iterations = best_iter
            else:
                chosen_iterations = 0

    # Full-data fit for ``chosen_iterations`` generalized-EM iterations.
    full_run = _EMRun(
        **run_common,
        weights=weights_arr,
        exposure_weights=exposure_weights,
        warn_empty=True,
        validation=None,
    )
    actual_iterations = 0
    if full_run.fitted:
        full_run.evaluate()
        for iteration in range(1, chosen_iterations + 1):
            is_last = iteration == chosen_iterations
            full_run.step(
                max_glm_steps=None if is_last else em_glm_steps,
                warn_glm=is_last,
            )
            full_run.evaluate()
            actual_iterations = iteration
            if (
                em_validation_blocks == 0
                and len(full_run.train_log_likelihood) >= 2
                and (
                    full_run.train_log_likelihood[-1]
                    - full_run.train_log_likelihood[-2]
                )
                < em_tol
            ):
                if not is_last:
                    # Polish rate maps to full GLM convergence on early exit.
                    full_run._fit_rate_maps(
                        full_run._responsibilities,
                        warm_start=True,
                        max_steps=None,
                        warn=True,
                    )
                break

    # Prune dead components supported by fewer than _MIN_COMPONENT_SPIKES
    # (always retaining at least the strongest component per non-empty electrode).
    final_mixtures: list[ProbeMarkMixture | None] = []
    rate_coefficients: list[np.ndarray | jax.Array | None] = []
    log_rate_maps: list[np.ndarray | None] = []
    first = 0
    fit_pos = 0
    for mixture in full_run.mark_mixtures:
        if mixture is None:
            final_mixtures.append(None)
            rate_coefficients.append(None)
            log_rate_maps.append(None)
            continue
        k_count = mixture.n_components
        maps = slice(first, first + k_count)
        first += k_count
        ev = full_run.events[fit_pos]
        resp_l, resp_r = full_run._responsibilities[fit_pos]
        fit_pos += 1
        counts = ev.left_share @ resp_l + ev.right_share @ resp_r
        keep = counts >= min(_MIN_COMPONENT_SPIKES, float(np.max(counts)))
        if not np.all(keep):
            w_kept = mixture.weights[keep]
            w_kept = w_kept / np.sum(w_kept)
            mixture = ProbeMarkMixture(
                weights=w_kept,
                means=mixture.means[keep],
                basis=mixture.basis[keep],
                basis_variances=mixture.basis_variances[keep],
                support_indices=mixture.support_indices[keep],
                support_mask=mixture.support_mask[keep],
                noise_variance=mixture.noise_variance,
                background_mean=mixture.background_mean,
                noise_whitener=mixture.noise_whitener,
                log_det_whitener=mixture.log_det_whitener,
                contact_positions=mixture.contact_positions,
                component_peak_channels=mixture.component_peak_channels[keep],
            )
            coef_e = full_run.coefficients[maps][keep]
            rates_e = full_run.interior_rates[:, maps][:, keep]
        else:
            coef_e = full_run.coefficients[maps]
            rates_e = full_run.interior_rates[:, maps]
        final_mixtures.append(mixture)
        rate_coefficients.append(np.asarray(coef_e))
        log_rate_maps.append(np.log(np.asarray(rates_e)))

    kept_rate_arrays = [
        np.exp(lr) for lr in log_rate_maps if lr is not None and lr.size
    ]
    if kept_rate_arrays:
        summed_gpi = np.sum([rates.sum(axis=1) for rates in kept_rate_arrays], axis=0)
    else:
        summed_gpi = np.zeros(len(interior_place_bin_centers), dtype=np.float32)

    return {
        "algorithm": "clusterless_probe_mixture",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "emission_design_info": full_run.prepared.emission_design_info,
        "mark_mixtures": final_mixtures,
        "rate_coefficients": rate_coefficients,
        "log_rate_maps": log_rate_maps,
        "summed_ground_process_intensity": np.asarray(summed_gpi),
        "mean_rates": np.asarray(full_run.mean_rates, dtype=np.float32),
        "mark_dimensions": mark_dimensions,
        "spike_block_size": spike_block_size,
        "max_candidate_components": max_candidate_components,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
        "em_diagnostics": {
            "n_iterations": int(actual_iterations),
            "train_log_likelihood": [float(v) for v in full_run.train_log_likelihood],
            "validation_log_likelihood": [float(v) for v in validation_trace],
        },
    }


class _PaddedProbeElectrode(NamedTuple):
    """Padded per-electrode arrays for shared compiled predict kernels."""

    mixture: ProbeMarkMixture
    means: jax.Array
    basis: jax.Array
    basis_variances: jax.Array
    support_indices: jax.Array
    support_mask: jax.Array
    noise_variance: jax.Array
    background_mean: jax.Array
    is_component: jax.Array
    coefficients: jax.Array
    log_rates: np.ndarray | None
    n_components: int
    all_full_support: bool


def _pad_probe_mixture(
    mixture: ProbeMarkMixture, n_comp_pad: int, c_loc_pad: int, rank_pad: int
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Pad a ``ProbeMarkMixture`` to static ``(n_comp_pad, c_loc_pad, rank_pad)``."""
    k = mixture.n_components
    c_loc = mixture.local_width
    r = mixture.basis_rank
    dk = n_comp_pad - k
    dc = c_loc_pad - c_loc
    dr = rank_pad - r

    means = np.pad(mixture.means, ((0, dk), (0, dc)), constant_values=0.0)
    basis = np.pad(mixture.basis, ((0, dk), (0, dc), (0, dr)), constant_values=0.0)
    basis_var = np.pad(mixture.basis_variances, ((0, dk), (0, dr)), constant_values=1.0)
    sup_idx = np.pad(
        mixture.support_indices, ((0, dk), (0, dc)), constant_values=0
    ).astype(np.int32)
    sup_mask = np.pad(mixture.support_mask, ((0, dk), (0, dc)), constant_values=False)
    is_comp = np.arange(n_comp_pad) < k
    return (
        jnp.asarray(means, dtype=jnp.float32),
        jnp.asarray(basis, dtype=jnp.float32),
        jnp.asarray(basis_var, dtype=jnp.float32),
        jnp.asarray(sup_idx, dtype=jnp.int32),
        jnp.asarray(sup_mask),
        jnp.asarray(mixture.noise_variance, dtype=jnp.float32),
        jnp.asarray(mixture.background_mean, dtype=jnp.float32),
        jnp.asarray(is_comp),
    )


def _device_marks(marks: np.ndarray, n_rows: int) -> jax.Array:
    """Pad marks to ``n_rows`` on device in float32, ignoring float32 overflow."""
    with np.errstate(over="ignore"):
        return jnp.asarray(pad_rows(marks, n_rows), dtype=jnp.float32)


@partial(
    jax.jit,
    static_argnames=("block_size", "indices_are_sorted", "all_full_support"),
    donate_argnums=0,
)
def _add_electrode_intensities(
    log_likelihood: jax.Array,
    features: jax.Array,
    row_ids: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    log_det_whitener: jax.Array,
    scaled_rates: jax.Array,
    log_rate_scale: jax.Array,
    log_background: jax.Array,
    *,
    block_size: int,
    indices_are_sorted: bool,
    all_full_support: bool,
) -> jax.Array:
    """Add one electrode's non-local log intensities across all K components."""

    def add_block(number: int, ll: jax.Array) -> jax.Array:
        first = number * block_size
        block = jax.lax.dynamic_slice_in_dim(features, first, block_size)
        rows = jax.lax.dynamic_slice_in_dim(row_ids, first, block_size)
        background = jax.lax.dynamic_slice_in_dim(log_background, first, block_size)
        log_density = _woodbury_log_density(
            block,
            means,
            basis,
            basis_variances,
            support_indices,
            support_mask,
            noise_variance,
            background_mean,
            is_component,
            log_det_whitener=0.0,
            all_full_support=all_full_support,
        ) + log_det_whitener.astype(block.dtype)
        log_density = jnp.where(
            is_component[None, :] & jnp.isfinite(log_density),
            log_density,
            jnp.asarray(-jnp.inf, dtype=block.dtype),
        )
        log_scale = jnp.max(log_density, axis=1)
        log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
        mixed = jnp.matmul(
            jnp.exp(log_density - log_scale[:, None]),
            scaled_rates,
            precision=jax.lax.Precision.HIGHEST,
        )
        log_intensity = jnp.log(mixed) + log_scale[:, None] + log_rate_scale[None]
        return deterministic_row_add(
            ll,
            jnp.logaddexp(log_intensity, background[:, None]),
            rows,
            indices_are_sorted=indices_are_sorted,
        )

    return cast(
        jax.Array,
        jax.lax.fori_loop(
            0, features.shape[0] // block_size, add_block, log_likelihood
        ),
    )


@partial(
    jax.jit,
    static_argnames=("block_size", "indices_are_sorted", "all_full_support"),
    donate_argnums=0,
)
def _add_electrode_candidate_intensities(
    log_likelihood: jax.Array,
    features: jax.Array,
    peak_channels: jax.Array,
    row_ids: jax.Array,
    contact_candidates: jax.Array,
    contact_candidate_mask: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    log_det_whitener: jax.Array,
    scaled_rates: jax.Array,
    log_rate_scale: jax.Array,
    log_background: jax.Array,
    *,
    block_size: int,
    indices_are_sorted: bool,
    all_full_support: bool,
) -> jax.Array:
    """Add one electrode's non-local log intensities via per-contact candidate LUT."""

    def add_block(number: int, ll: jax.Array) -> jax.Array:
        first = number * block_size
        block = jax.lax.dynamic_slice_in_dim(features, first, block_size)
        peaks = jax.lax.dynamic_slice_in_dim(peak_channels, first, block_size)
        rows = jax.lax.dynamic_slice_in_dim(row_ids, first, block_size)
        background = jax.lax.dynamic_slice_in_dim(log_background, first, block_size)
        cand_k = contact_candidates[peaks]  # (block, k_cand)
        cand_mask = contact_candidate_mask[peaks]  # (block, k_cand)
        log_density = _woodbury_candidate_log_density(
            block,
            cand_k,
            cand_mask,
            means,
            basis,
            basis_variances,
            support_indices,
            support_mask,
            noise_variance,
            background_mean,
            is_component,
            log_det_whitener=0.0,
            all_full_support=all_full_support,
        ) + log_det_whitener.astype(block.dtype)
        log_density = jnp.where(
            cand_mask & jnp.isfinite(log_density),
            log_density,
            jnp.asarray(-jnp.inf, dtype=block.dtype),
        )
        log_scale = jnp.max(log_density, axis=1)
        log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
        exp_dens = jnp.exp(log_density - log_scale[:, None])
        cand_rates = scaled_rates[cand_k]  # (block, k_cand, n_bins)
        mixed = jnp.einsum(
            "bj,bjn->bn", exp_dens, cand_rates, precision=jax.lax.Precision.HIGHEST
        )
        log_intensity = jnp.log(mixed) + log_scale[:, None] + log_rate_scale[None]
        return deterministic_row_add(
            ll,
            jnp.logaddexp(log_intensity, background[:, None]),
            rows,
            indices_are_sorted=indices_are_sorted,
        )

    return cast(
        jax.Array,
        jax.lax.fori_loop(
            0, features.shape[0] // block_size, add_block, log_likelihood
        ),
    )


@partial(
    jax.jit,
    static_argnames=("indices_are_sorted", "all_full_support"),
    donate_argnums=0,
)
def _add_electrode_local_terms(
    log_likelihood: jax.Array,
    features: jax.Array,
    row_ids: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    log_det_whitener: jax.Array,
    spike_log_rates: jax.Array,
    log_background: jax.Array,
    *,
    indices_are_sorted: bool,
    all_full_support: bool,
) -> jax.Array:
    """Add one electrode's local spike terms across all K components."""
    log_density = _woodbury_log_density(
        features,
        means,
        basis,
        basis_variances,
        support_indices,
        support_mask,
        noise_variance,
        background_mean,
        is_component,
        log_det_whitener=0.0,
        all_full_support=all_full_support,
    ) + log_det_whitener.astype(features.dtype)
    log_density = jnp.where(
        is_component[None, :] & jnp.isfinite(log_density),
        log_density,
        jnp.asarray(-jnp.inf, dtype=features.dtype),
    )
    terms = jnp.logaddexp(
        jax.scipy.special.logsumexp(log_density + spike_log_rates, axis=1),
        log_background,
    )
    return log_likelihood + deterministic_row_sum(
        terms,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )


@partial(
    jax.jit,
    static_argnames=("indices_are_sorted", "all_full_support"),
    donate_argnums=0,
)
def _add_electrode_local_candidate_terms(
    log_likelihood: jax.Array,
    features: jax.Array,
    peak_channels: jax.Array,
    row_ids: jax.Array,
    contact_candidates: jax.Array,
    contact_candidate_mask: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    support_indices: jax.Array,
    support_mask: jax.Array,
    noise_variance: jax.Array,
    background_mean: jax.Array,
    is_component: jax.Array,
    log_det_whitener: jax.Array,
    spike_log_rates: jax.Array,
    log_background: jax.Array,
    *,
    indices_are_sorted: bool,
    all_full_support: bool,
) -> jax.Array:
    """Add one electrode's local spike terms via per-contact candidate LUT."""
    cand_k = contact_candidates[peak_channels]
    cand_mask = contact_candidate_mask[peak_channels]
    log_density = _woodbury_candidate_log_density(
        features,
        cand_k,
        cand_mask,
        means,
        basis,
        basis_variances,
        support_indices,
        support_mask,
        noise_variance,
        background_mean,
        is_component,
        log_det_whitener=0.0,
        all_full_support=all_full_support,
    ) + log_det_whitener.astype(features.dtype)
    log_density = jnp.where(
        cand_mask & jnp.isfinite(log_density),
        log_density,
        jnp.asarray(-jnp.inf, dtype=features.dtype),
    )
    cand_log_rates = jnp.take_along_axis(spike_log_rates, cand_k, axis=1)
    terms = jnp.logaddexp(
        jax.scipy.special.logsumexp(log_density + cand_log_rates, axis=1),
        log_background,
    )
    return log_likelihood + deterministic_row_sum(
        terms,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )


@requires_time_edges
def predict_clusterless_probe_mixture_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    *,
    environment: Environment,
    emission_design_info: TrackGraphBasis | TensorSplineBasis,
    mark_mixtures: list[ProbeMarkMixture | None],
    rate_coefficients: list[np.ndarray | None],
    log_rate_maps: list[np.ndarray | None],
    summed_ground_process_intensity: np.ndarray,
    mean_rates: np.ndarray,
    is_local: bool = False,
    spike_block_size: int = 1000,
    max_candidate_components: int | None = None,
    disable_progress_bar: bool = False,
    row_slice: slice | None = None,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    mark_dimensions: list[int],
    mark_reference: list[MarkReference | None],
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    environment_grid: dict[str, Any] | None = None,
    em_diagnostics: dict[str, Any] | None = None,  # noqa: ARG001 -- fit diagnostic
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Predict the non-local or local log likelihood of the probe mixture model.

    Parameters
    ----------
    position_time : array_like or None, shape (n_time_position,)
        Tracking timestamps in seconds; read only when ``is_local`` is True.
    position : array_like or None, shape (n_time_position, n_position_dims)
        Animal position; read only when ``is_local`` is True.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Decoding spike times per electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_channels)
        Decoding spike marks per electrode.
    environment : Environment
        Environment the model was fit on.
    emission_design_info : TrackGraphBasis or TensorSplineBasis
        Rate-map spline basis, from the fit.
    mark_mixtures : list of ProbeMarkMixture or None
        Fitted probe mixtures per electrode; None for zero-rate electrodes.
    rate_coefficients : list of np.ndarray or None
        Per electrode, shape ``(n_components, n_coefficients)``, from the fit.
    log_rate_maps : list of np.ndarray or None
        Per electrode, shape ``(n_interior_bins, n_components)``, from the fit.
    summed_ground_process_intensity : np.ndarray, shape (n_interior_bins,)
        Summed component rates in Hz, from the fit.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Mean rate per electrode in Hz, from the fit.
    is_local : bool, optional
        If True, evaluate at the animal's interpolated position. By default False.
    spike_block_size : int, optional
        Decode spikes scored per block, by default 1000.
    max_candidate_components : int or None, optional
        Maximum candidate components scored per decode spike using a static
        per-peak-contact lookup table. By default None (evaluate all K).
    disable_progress_bar : bool, optional
        Hide per-electrode progress bar, by default False.
    row_slice : slice or None, optional
        Contiguous output bin slice, by default None (all bins).
    time_edges : np.ndarray, shape (n_bins + 1,), keyword-only
        Full decoding bin edges.
    algorithm : str, optional
        Registered algorithm name recorded by the fit; unused.
    mark_dimensions : list of int, keyword-only
        Mark feature dimension per electrode, from the fit.
    mark_reference : list of MarkReference or None, keyword-only
        Reference mark density per electrode for the background intensity.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the fit; must be ``"Hz"``.
    encoding_exposure_seconds : float, optional, keyword-only
        Recorded by the fit; unused.
    environment_grid : dict or None, optional
        Fit-time environment grid fingerprint.
    em_diagnostics : dict or None, optional
        Fit-time EM convergence trace; unused.
    _spike_time_order : SpikeTimeOrder, optional
        Internal precomputed spike ordering.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode time grid.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_interior_bins) or (n_rows, 1)
        Float32 log likelihood per requested time bin.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    _validate_execution_settings(spike_block_size, max_candidate_components)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        mark_mixtures=mark_mixtures,
        mean_rates=mean_rates,
        mark_dimensions=mark_dimensions,
        mark_reference=mark_reference,
    )
    for electrode, (electrode_times, electrode_marks, expected) in enumerate(
        zip(spike_times, spike_waveform_features, mark_dimensions, strict=True)
    ):
        validate_spike_feature_pair(electrode_times, electrode_marks, electrode)
        if np.shape(electrode_marks)[1] != expected:
            msg = f"waveform feature dimension changed for electrode {electrode}"
            raise ValidationError(
                msg,
                expected=f"{expected} features per spike",
                got=f"{np.shape(electrode_marks)[1]} features",
                hint="Use the same waveform feature representation at fit and predict.",
            )

    n_rows = rows.stop - rows.start
    durations = jnp.asarray(rows.durations(), dtype=jnp.float32)
    if is_local:
        local_position_time, local_position = require_position(
            position_time, position, "local clusterless probe mixture likelihood"
        )
        local_position_time = np.asarray(local_position_time)
        local_position = np.asarray(local_position)
        local_position = (
            local_position if local_position.ndim > 1 else local_position[:, None]
        )
        row_position = snap_off_track_positions(
            get_position_at_time(
                local_position_time,
                local_position,
                decode_bin_centers(rows.time_edges, rows.start, rows.stop),
                environment,
            ),
            environment,
        )
        row_design = jnp.asarray(
            make_spline_predict_matrix(emission_design_info, row_position),
            dtype=jnp.float32,
        )
        expected_rate = jnp.zeros((n_rows,), dtype=jnp.float32)
        log_likelihood = jnp.zeros((n_rows,), dtype=jnp.float32)
    else:
        log_likelihood = -durations[:, None] * (
            jnp.asarray(summed_ground_process_intensity, dtype=jnp.float32)[None]
            + background_rate(len(spike_times))
        )

    fitted_mixtures = [m for m in mark_mixtures if m is not None]
    n_comp_pad = _power_of_two(
        max((m.n_components for m in fitted_mixtures), default=1)
    )
    c_loc_pad = _power_of_two(max((m.local_width for m in fitted_mixtures), default=1))
    rank_pad = _power_of_two(max((m.basis_rank for m in fitted_mixtures), default=1))

    electrode_models: list[_PaddedProbeElectrode | None] = []
    for mixture, coefficients, log_rates in zip(
        mark_mixtures, rate_coefficients, log_rate_maps, strict=True
    ):
        if mixture is None:
            electrode_models.append(None)
            continue
        (
            means,
            basis,
            basis_var,
            sup_idx,
            sup_mask,
            noise_var,
            bg_mean,
            is_comp,
        ) = _pad_probe_mixture(mixture, n_comp_pad, c_loc_pad, rank_pad)
        padded_coef = jnp.asarray(
            pad_rows(cast(np.ndarray, coefficients), n_comp_pad), dtype=jnp.float32
        )
        if is_local:
            expected_rate = expected_rate + glm_rates(
                row_design, padded_coef
            ) @ is_comp.astype(row_design.dtype)
        electrode_models.append(
            _PaddedProbeElectrode(
                mixture=mixture,
                means=means,
                basis=basis,
                basis_variances=basis_var,
                support_indices=sup_idx,
                support_mask=sup_mask,
                noise_variance=noise_var,
                background_mean=bg_mean,
                is_component=is_comp,
                coefficients=padded_coef,
                log_rates=log_rates,
                n_components=mixture.n_components,
                all_full_support=mixture.all_full_support,
            )
        )

    def add_electrode(
        ll: jax.Array,
        spikes: ElectrodeSpikes,
        model: _PaddedProbeElectrode | None,
    ) -> jax.Array:
        if model is None:
            return add_background_only(ll, spikes)
        # Whiten marks in float64 on host so offset digits are preserved and
        # whitened/unwhitened electrodes share the exact same compiled kernel.
        with np.errstate(over="ignore", invalid="ignore"):
            whitened_marks = model.mixture.whiten_marks(spikes.marks)
        log_det_w = jnp.asarray(model.mixture.log_det_whitener, dtype=jnp.float32)
        use_candidates = max_candidate_components is not None

        if is_local:
            selected_times = select_spike_rows(spikes.times, spikes.selection)
            raw_row_ids = spike_row_ids(spikes.selection)
            if use_candidates:
                assert max_candidate_components is not None
                cand_tbl, cand_msk = model.mixture.candidate_table(
                    max_candidate_components
                )
                cand_tbl_j = jnp.asarray(cand_tbl, dtype=jnp.int32)
                cand_msk_j = jnp.asarray(cand_msk)

            for start in range(0, spikes.n_spikes, spike_block_size):
                stop = min(start + spike_block_size, spikes.n_spikes)
                n_cur = stop - start
                n_pad = padded_spike_count(n_cur, spike_block_size)
                blk_position = snap_off_track_positions(
                    get_position_at_time(
                        local_position_time,
                        local_position,
                        selected_times[start:stop],
                        environment,
                    ),
                    environment,
                )
                blk_log_rates = jnp.log(
                    glm_rates(
                        jnp.asarray(
                            pad_rows(
                                make_spline_predict_matrix(
                                    emission_design_info, blk_position
                                ),
                                n_pad,
                            ),
                            dtype=jnp.float32,
                        ),
                        model.coefficients,
                    )
                )
                blk_marks = _device_marks(whitened_marks[start:stop], n_pad)
                blk_ids_host = np.full((n_pad,), n_rows, dtype=raw_row_ids.dtype)
                blk_ids_host[:n_cur] = raw_row_ids[start:stop]
                blk_row_ids = jnp.asarray(blk_ids_host)
                blk_bg = jnp.asarray(
                    pad_rows(spikes.log_background[start:stop], n_pad),
                    dtype=jnp.float32,
                )
                if use_candidates:
                    blk_w = whitened_marks[start:stop]
                    with np.errstate(over="ignore", invalid="ignore"):
                        blk_peaks = np.argmax(
                            np.abs(
                                np.where(
                                    np.isfinite(blk_w),
                                    blk_w - model.mixture.background_mean[None, :],
                                    0.0,
                                )
                            ),
                            axis=1,
                        ).astype(np.int32)
                    ll = cast(
                        jax.Array,
                        _add_electrode_local_candidate_terms(
                            ll,
                            blk_marks,
                            jnp.asarray(pad_rows(blk_peaks, n_pad), dtype=jnp.int32),
                            blk_row_ids,
                            cand_tbl_j,
                            cand_msk_j,
                            model.means,
                            model.basis,
                            model.basis_variances,
                            model.support_indices,
                            model.support_mask,
                            model.noise_variance,
                            model.background_mean,
                            model.is_component,
                            log_det_w,
                            blk_log_rates,
                            blk_bg,
                            indices_are_sorted=spikes.selection.indices_are_sorted,
                            all_full_support=model.all_full_support,
                        ),
                    )
                else:
                    ll = cast(
                        jax.Array,
                        _add_electrode_local_terms(
                            ll,
                            blk_marks,
                            blk_row_ids,
                            model.means,
                            model.basis,
                            model.basis_variances,
                            model.support_indices,
                            model.support_mask,
                            model.noise_variance,
                            model.background_mean,
                            model.is_component,
                            log_det_w,
                            blk_log_rates,
                            blk_bg,
                            indices_are_sorted=spikes.selection.indices_are_sorted,
                            all_full_support=model.all_full_support,
                        ),
                    )
            return ll

        log_rates = jnp.asarray(model.log_rates, dtype=jnp.float32)
        log_rate_scale = jnp.max(log_rates, axis=1)
        scaled_rates = jnp.pad(
            jnp.exp(log_rates - log_rate_scale[:, None]).T,
            ((0, n_comp_pad - model.n_components), (0, 0)),
        )
        if use_candidates:
            assert max_candidate_components is not None
            cand_tbl, cand_msk = model.mixture.candidate_table(max_candidate_components)
            k_cand = cand_tbl.shape[1]
            n_bins = int(log_rates.shape[0])
            eff_block = max(
                1,
                min(
                    spike_block_size,
                    _CANDIDATE_GATHER_ELEMENTS // max(k_cand * max(n_bins, 1), 1),
                ),
            )
            n_padded = padded_spike_count(spikes.n_spikes, eff_block, fine=True)
            block_size = min(eff_block, n_padded)
            with np.errstate(over="ignore", invalid="ignore"):
                peaks = np.argmax(
                    np.abs(
                        np.where(
                            np.isfinite(whitened_marks),
                            whitened_marks - model.mixture.background_mean[None, :],
                            0.0,
                        )
                    ),
                    axis=1,
                ).astype(np.int32)
            return cast(
                jax.Array,
                _add_electrode_candidate_intensities(
                    ll,
                    _device_marks(whitened_marks, n_padded),
                    jnp.asarray(pad_rows(peaks, n_padded), dtype=jnp.int32),
                    jnp.asarray(spike_row_ids(spikes.selection, n_padded)),
                    jnp.asarray(cand_tbl, dtype=jnp.int32),
                    jnp.asarray(cand_msk),
                    model.means,
                    model.basis,
                    model.basis_variances,
                    model.support_indices,
                    model.support_mask,
                    model.noise_variance,
                    model.background_mean,
                    model.is_component,
                    log_det_w,
                    scaled_rates,
                    log_rate_scale,
                    jnp.asarray(
                        pad_rows(spikes.log_background, n_padded), dtype=jnp.float32
                    ),
                    block_size=block_size,
                    indices_are_sorted=spikes.selection.indices_are_sorted,
                    all_full_support=model.all_full_support,
                ),
            )

        n_padded = padded_spike_count(spikes.n_spikes, spike_block_size, fine=True)
        return cast(
            jax.Array,
            _add_electrode_intensities(
                ll,
                _device_marks(whitened_marks, n_padded),
                jnp.asarray(spike_row_ids(spikes.selection, n_padded)),
                model.means,
                model.basis,
                model.basis_variances,
                model.support_indices,
                model.support_mask,
                model.noise_variance,
                model.background_mean,
                model.is_component,
                log_det_w,
                scaled_rates,
                log_rate_scale,
                jnp.asarray(
                    pad_rows(spikes.log_background, n_padded), dtype=jnp.float32
                ),
                block_size=min(spike_block_size, n_padded),
                indices_are_sorted=spikes.selection.indices_are_sorted,
                all_full_support=model.all_full_support,
            ),
        )

    log_likelihood = add_electrode_spike_terms(
        log_likelihood,
        spike_times,
        spike_waveform_features,
        mark_reference,
        electrode_models,
        add_electrode,
        rows=rows,
        log_background_floor=LOG_RATE_EPS_HZ,
        desc=f"{'Local' if is_local else 'Position-Bin'} Likelihood (probe mixture)",
        disable_progress_bar=disable_progress_bar,
    )

    duration_evidence = log_bin_duration_evidence(
        spike_times,
        rows.time_edges,
        row_slice,
        rows.spike_time_order,
        _time_grid=rows.time_grid,
    )
    if is_local:
        log_likelihood = log_likelihood - durations * (
            expected_rate + background_rate(len(spike_times))
        )
        return (log_likelihood + duration_evidence)[:, None]
    return log_likelihood + duration_evidence[:, None]
