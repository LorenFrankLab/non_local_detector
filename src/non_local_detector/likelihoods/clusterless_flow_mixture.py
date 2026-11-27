"""Clusterless flow-warped mark-mixture likelihood.

Models each electrode's marked intensity by warping waveform features through
an invertible normalizing flow ``z = f_theta(m)`` into a Gaussian-friendly
latent space, paired with a low-rank-plus-diagonal Gaussian mixture and
Poisson GLM spatial rate maps:

    lambda_e(x, m) = |det J_{f_theta}(m)| * sum_k lambda_ek(x) N(f_theta(m); mu_ek, Sigma_ek)
                     + beta * q_e(m),

where:

- **Layer 0 (standardization / optional ZCA whitening)** centers and scales
  marks into dimensionless coordinates (or applies symmetric ZCA whitening
  ``Sigma_noise^{-1/2}`` followed by per-channel scaling when measured
  ``noise_covariances`` are supplied), preserving exact equivariance under
  uniform scalar unit changes (``m -> a*m + b`` with ``a > 0`` and
  ``noise_covariances -> a^2 * noise_covariances`` or
  ``mark_reg_covar -> a^2 * mark_reg_covar``);
- **Bijector ``g_theta``** is a stack of Rational-Quadratic Neural Spline
  (RQS) coupling layers (Durkan et al. 2019) with linear identity tails outside
  ``[-B, B]`` (slope 1, log-determinant 0), zero-initialized so iteration 0 is
  the exact identity map. Identity tails guarantee exponential Gaussian decay
  along every ray in ``R^D``, preserving ``mark_reference``'s Student-t
  (``nu = 3``) polynomial tail dominance for ``No-Spike``. In dense mode
  (``contact_positions=None``), zero-initialized Cayley orthogonal layers
  ``Q = (I - A)(I + A)^{-1}`` mix channels with exact ``log|det Q| = 0`` and
  exact norm preservation. When physical ``contact_positions`` are provided,
  each transformed contact conditions only on physically nearby contacts from
  the complementary spatial partition;
- **Latent mixture** uses low-rank-plus-diagonal covariances
  ``Sigma_ek = diag(psi_e) + U_ek U_ek^T`` (auto-full when ``covariance_rank``
  is None or ``>= D``), stored in cancellation-free orthonormal eigenbasis form
  for ``O(K D r)`` scoring;
- **Spatial rate maps** reuse ``sorted_spikes_glm``'s spline basis with cached
  position-cell aggregation and warm-started damped Newton M-steps.

Because ``f_theta`` is a global ``C^1`` diffeomorphism,
``integral lambda_e(x, m) dm = sum_k lambda_ek(x) + beta`` holds analytically
with no Monte Carlo normalization, and ``f_theta(m)`` runs once per spike at
prediction time before the ``K x n_bins`` spatial reduction.

Training optimizes the flow with ``optax`` (optional extra
``non_local_detector[neural]``) under either the joint marked-Poisson objective
(``flow_objective="joint"``, default) or position-free marginal mark density
(``flow_objective="marginal"``), alternating flow AdamW steps, exact monotone
factor-analysis mixture M-steps in latent space, and warm-started GLM rate-map
M-steps, with blocked inner-validation early stopping and identity epoch-0
fallback. Prediction and ``.nld`` loading run in pure JAX without ``optax``.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, cast

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
    RATE_EPS_HZ,
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
from non_local_detector.likelihoods.sorted_spikes_glm import (
    GLM_LOG_RATE_PRIOR_STD,
    PreparedRateMapBasis,
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
    "ClusterlessFlowMixtureEncodingModel",
    "FlowMarkMixture",
    "fit_clusterless_flow_mixture_encoding_model",
    "predict_clusterless_flow_mixture_log_likelihood",
]

# Valid flow training objectives.
FLOW_OBJECTIVES = ("joint", "marginal")
# Valid serialized flow execution modes.
FLOW_MODES = ("identity", "elementwise", "dense", "local")
# Default low-rank latent covariance rank.
DEFAULT_COVARIANCE_RANK = 4
# Default spline bijector hyperparameters.
DEFAULT_N_FLOW_LAYERS = 4
DEFAULT_N_SPLINE_BINS = 8
DEFAULT_SPLINE_BOUND = 5.0
DEFAULT_FLOW_HIDDEN_UNITS = 64
# Local shared conditioner hidden width cap so O(n_trans * C_loc * H_loc) stays compact.
_MAX_LOCAL_HIDDEN_UNITS = 32
# Default physical neighborhood radius multiplier and neighbor cap for local mode.
_DEFAULT_PITCH_RADIUS_MULTIPLIER = 3.0
_DEFAULT_MAX_LOCAL_CHANNELS = 16
# Rational-quadratic spline numerical floors.
_MIN_BIN_FRACTION = 1e-3
_MIN_SPLINE_DERIVATIVE = 1e-3
# Dimensionless floor on standardized latent diagonal variances.
_MIN_LATENT_VARIANCE = 1e-6
# Minimum responsibility mass before keeping previous component parameters.
_MIN_RESPONSIBILITY_MASS = 1e-8
# Spikes per block for host-batched E-steps and flow evaluation.
_E_STEP_BLOCK = 1024
# Default mini-batch size for optax flow optimization steps.
DEFAULT_FLOW_BATCH_SIZE = 512
# Module-level cache of compiled optax flow-step kernels keyed by hyperparameters.
_FLOW_STEP_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _require_optax() -> Any:
    """Import optax or raise a clear installation error."""
    try:
        return importlib.import_module("optax")
    except ImportError as error:
        if error.name != "optax":
            raise
        msg = (
            "clusterless_flow_mixture trains its normalizing flow with optax, "
            "which is not installed. Install it with "
            "`pip install non_local_detector[neural]`."
        )
        raise ImportError(msg) from error


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


def _validate_positive_float(name: str, value: Any) -> float:
    """Reject values that are not finite strictly positive numbers."""
    if isinstance(value, (bool, np.bool_)):
        msg = f"{name} must be a positive float"
        raise ValidationError(msg, expected="float > 0", got=repr(value))
    try:
        val = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a positive float"
        raise ValidationError(msg, expected="float > 0", got=repr(value)) from error
    if not (np.isfinite(val) and val > 0.0):
        msg = f"{name} must be finite and positive"
        raise ValidationError(msg, expected="finite float > 0", got=repr(value))
    return val


def _validate_nonnegative_float(name: str, value: Any) -> float:
    """Reject values that are not finite non-negative numbers."""
    if isinstance(value, (bool, np.bool_)):
        msg = f"{name} must be a non-negative float"
        raise ValidationError(msg, expected="float >= 0", got=repr(value))
    try:
        val = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a non-negative float"
        raise ValidationError(msg, expected="float >= 0", got=repr(value)) from error
    if not (np.isfinite(val) and val >= 0.0):
        msg = f"{name} must be finite and non-negative"
        raise ValidationError(msg, expected="finite float >= 0", got=repr(value))
    return val


def _read_only_copy(array: np.ndarray, *, dtype: Any = None) -> np.ndarray:
    """Return a read-only contiguous NumPy copy."""
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


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


# ---------------------------------------------------------------------------
# Optional probe geometry & ZCA noise-whitening helpers (self-contained)
# ---------------------------------------------------------------------------


def _validate_contact_positions(
    contact_positions: ArrayLike,
    *,
    expected_channels: int,
    electrode: int,
) -> np.ndarray:
    """Validate one electrode's physical contact coordinates."""
    positions = np.asarray(contact_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] < 1:
        msg = (
            f"contact_positions of electrode {electrode} must be a non-empty 2-D array"
        )
        raise ValidationError(
            msg,
            expected="shape (n_contacts, n_probe_dims)",
            got=f"shape {positions.shape}",
        )
    if not np.all(np.isfinite(positions)):
        msg = f"contact_positions of electrode {electrode} must contain only finite values"
        raise ValidationError(msg)
    if positions.shape[0] != expected_channels:
        msg = (
            f"contact_positions of electrode {electrode} channel count does not "
            "match waveform features"
        )
        raise ValidationError(
            msg,
            expected=f"{expected_channels} contacts",
            got=f"{positions.shape[0]} contacts",
        )
    return positions


def _validate_noise_covariance(
    noise_covariance: ArrayLike,
    n_channels: int,
    *,
    electrode: int,
) -> np.ndarray:
    """Validate a symmetric positive-definite channel noise covariance."""
    cov = np.asarray(noise_covariance, dtype=np.float64)
    if cov.shape != (n_channels, n_channels):
        msg = f"noise_covariance of electrode {electrode} shape does not match channel count"
        raise ValidationError(
            msg,
            expected=f"shape ({n_channels}, {n_channels})",
            got=f"shape {cov.shape}",
        )
    if not np.all(np.isfinite(cov)):
        msg = (
            f"noise_covariance of electrode {electrode} must contain only finite values"
        )
        raise ValidationError(msg)
    if not np.allclose(cov, cov.T, rtol=1e-5, atol=1e-8):
        msg = f"noise_covariance of electrode {electrode} must be symmetric"
        raise ValidationError(msg)
    sym = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(sym)
    if not (np.all(np.isfinite(eigvals)) and float(np.min(eigvals)) > 0.0):
        msg = f"noise_covariance of electrode {electrode} must be symmetric positive definite"
        raise ValidationError(
            msg,
            expected="strictly positive eigenvalues",
            got=f"min eigenvalue {float(np.min(eigvals)):.3e}",
        )
    return sym


def _compute_zca_whitener(noise_covariance: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric ZCA whitener ``Sigma^{-1/2}`` and its log-determinant."""
    eigvals, eigvecs = np.linalg.eigh(noise_covariance)
    inv_sqrt = 1.0 / np.sqrt(eigvals)
    whitener = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T
    whitener = 0.5 * (whitener + whitener.T)
    log_det = -0.5 * float(np.sum(np.log(eigvals)))
    return whitener, log_det


def _resolve_electrode_geometry(
    contact_positions: Sequence[ArrayLike | None] | None,
    noise_covariances: Sequence[ArrayLike | None] | None,
    mark_dimensions: Sequence[int],
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Resolve optional per-electrode contact positions and noise covariances."""
    n_electrodes = len(mark_dimensions)
    if contact_positions is None:
        raw_pos: list[ArrayLike | None] = [None] * n_electrodes
    else:
        if len(contact_positions) != n_electrodes:
            msg = "contact_positions length does not match electrode count"
            raise ValidationError(
                msg,
                expected=f"{n_electrodes} entries",
                got=f"{len(contact_positions)} entries",
            )
        raw_pos = list(contact_positions)

    if noise_covariances is None:
        raw_cov: list[ArrayLike | None] = [None] * n_electrodes
    else:
        if len(noise_covariances) != n_electrodes:
            msg = "noise_covariances length does not match electrode count"
            raise ValidationError(
                msg,
                expected=f"{n_electrodes} entries",
                got=f"{len(noise_covariances)} entries",
            )
        raw_cov = list(noise_covariances)

    resolved_pos: list[np.ndarray | None] = []
    resolved_cov: list[np.ndarray | None] = []
    for electrode, (n_ch, pos, cov) in enumerate(
        zip(mark_dimensions, raw_pos, raw_cov, strict=True)
    ):
        resolved_pos.append(
            None
            if pos is None
            else _validate_contact_positions(
                pos, expected_channels=n_ch, electrode=electrode
            )
        )
        resolved_cov.append(
            None
            if cov is None
            else _validate_noise_covariance(cov, n_ch, electrode=electrode)
        )
    return resolved_pos, resolved_cov


def _build_local_coupling_topology(
    contact_positions: np.ndarray,
    n_flow_layers: int,
    *,
    neighborhood_radius: float | None = None,
    max_local_channels: int | None = _DEFAULT_MAX_LOCAL_CHANNELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build spatially alternating partitions and neighbor masks for local RQS layers.

    Each contact ``c`` in the transformed half ``B_l`` conditions strictly on
    its physical neighbors in the complementary half ``A_l`` within
    ``neighborhood_radius`` (capped at ``max_local_channels``). Because
    ``A_l`` and ``B_l`` are disjoint, ``c`` never conditions on itself and the
    coupling Jacobian is strictly triangular. Only the ``n_trans = ceil(D / 2)``
    transformed contacts are evaluated per layer, using a weight-shared
    conditioner with static per-contact geometry context.

    Parameters
    ----------
    contact_positions : np.ndarray, shape (n_contacts, n_probe_dims)
        Physical coordinates of each contact.
    n_flow_layers : int
        Number of coupling layers.
    neighborhood_radius : float or None, optional
        Physical radius; None auto-infers ``3.0 * median_nearest_neighbor_pitch``.
    max_local_channels : int or None, optional
        Maximum conditioning neighbors retained per contact.

    Returns
    -------
    cond_nbr_indices : np.ndarray of int32, shape (n_flow_layers, n_trans, c_loc)
        Conditioning neighbor channel indices per layer and transformed slot.
    cond_nbr_masks : np.ndarray of bool, shape (n_flow_layers, n_trans, c_loc)
        True for real conditioning neighbors in ``A_l``.
    trans_indices : np.ndarray of int32, shape (n_flow_layers, n_trans)
        Contact indices in ``B_l`` (padded slots hold ``n_contacts``, pointing
        to a harmless dummy column during scatter).
    trans_valid : np.ndarray of bool, shape (n_flow_layers, n_trans)
        True for real transformed contacts, False for odd-channel padding slots.
    cond_geom : np.ndarray of float64, shape (n_flow_layers, n_trans, c_loc + n_pos)
        Static geometry context per transformed slot (neighbor presence mask
        plus normalized physical coordinates in ``[-1, 1]``).
    """
    n_contacts, n_probe_dims = contact_positions.shape
    diff = contact_positions[:, None, :] - contact_positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    if neighborhood_radius is None:
        if n_contacts <= 1:
            resolved_radius = np.inf
        else:
            off_diag = np.where(np.eye(n_contacts, dtype=bool), np.inf, dist)
            nn_dist = np.min(off_diag, axis=1)
            positive_nn = nn_dist[nn_dist > 0.0]
            if positive_nn.size == 0:
                resolved_radius = np.inf
            else:
                resolved_radius = _DEFAULT_PITCH_RADIUS_MULTIPLIER * float(
                    np.median(positive_nn)
                )
    else:
        resolved_radius = _validate_positive_float(
            "neighborhood_radius", neighborhood_radius
        )

    # Sort contacts along highest-variance spatial axis (breaking ties by
    # remaining coordinates) for a balanced spatial checkerboard partition.
    axis_order = np.argsort(-np.var(contact_positions, axis=0), kind="mergesort")
    keys = tuple(contact_positions[:, ax] for ax in reversed(axis_order))
    spatial_order = np.lexsort(keys)
    parity = np.zeros(n_contacts, dtype=np.int32)
    parity[spatial_order] = np.arange(n_contacts, dtype=np.int32) % 2

    # Normalized physical contact coordinates in [-1, 1] (up to 3 spatial axes).
    n_pos_feats = min(n_probe_dims, 3)
    pos_cols = contact_positions[:, axis_order[:n_pos_feats]]
    p_min = np.min(pos_cols, axis=0)
    p_max = np.max(pos_cols, axis=0)
    span = p_max - p_min
    norm_pos = np.where(
        span[None, :] > 0.0,
        2.0 * (pos_cols - p_min[None, :]) / np.where(span > 0.0, span, 1.0)[None, :]
        - 1.0,
        0.0,
    )

    trans_lists: list[np.ndarray] = []
    raw_lists: list[list[np.ndarray]] = []
    max_found = 1
    max_trans = 1
    for layer in range(n_flow_layers):
        is_cond = parity == (layer % 2)
        t_contacts = np.flatnonzero(~is_cond).astype(np.int32)
        trans_lists.append(t_contacts)
        max_trans = max(max_trans, int(t_contacts.size))
        layer_nbrs: list[np.ndarray] = []
        for c in t_contacts:
            cand = np.flatnonzero(is_cond & (dist[c] <= resolved_radius + 1e-12))
            if cand.size > 0:
                # Break distance ties canonically by signed displacement along
                # principal probe axes so shared-weight neighbor slots align.
                tie_keys = tuple(
                    contact_positions[cand, ax] - contact_positions[c, ax]
                    for ax in reversed(axis_order)
                )
                order = np.lexsort((*tie_keys, np.round(dist[c, cand], decimals=9)))
                cand = cand[order]
                if max_local_channels is not None and cand.size > max_local_channels:
                    cand = cand[:max_local_channels]
                max_found = max(max_found, int(cand.size))
            layer_nbrs.append(cand.astype(np.int32))
        raw_lists.append(layer_nbrs)

    c_loc = max(max_found, 1)
    n_trans = max(max_trans, 1)
    cond_nbr_indices = np.zeros((n_flow_layers, n_trans, c_loc), dtype=np.int32)
    cond_nbr_masks = np.zeros((n_flow_layers, n_trans, c_loc), dtype=bool)
    trans_indices = np.full((n_flow_layers, n_trans), n_contacts, dtype=np.int32)
    trans_valid = np.zeros((n_flow_layers, n_trans), dtype=bool)
    cond_geom = np.zeros(
        (n_flow_layers, n_trans, c_loc + n_pos_feats), dtype=np.float64
    )
    for layer in range(n_flow_layers):
        t_contacts = trans_lists[layer]
        for slot, c in enumerate(t_contacts):
            trans_indices[layer, slot] = int(c)
            trans_valid[layer, slot] = True
            nbrs = raw_lists[layer][slot]
            k = nbrs.size
            if k > 0:
                cond_nbr_indices[layer, slot, :k] = nbrs
                cond_nbr_masks[layer, slot, :k] = True
                cond_geom[layer, slot, :k] = 1.0
            cond_geom[layer, slot, c_loc:] = norm_pos[int(c)]
    return cond_nbr_indices, cond_nbr_masks, trans_indices, trans_valid, cond_geom


# ---------------------------------------------------------------------------
# Rational-Quadratic Neural Spline (RQS) bijector kernels (pure JAX)
# ---------------------------------------------------------------------------


def _rqs_spline_knots(
    raw_params: jax.Array,
    *,
    n_bins: int,
    bound: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert unconstrained ``(..., 3*K - 1)`` parameters into valid RQS knots.

    Returns ``(x_knots, widths, y_knots, heights, derivatives)`` with exact
    endpoints ``[-bound, bound]`` and boundary derivatives equal to ``1.0``.
    Zero input parameters produce the exact identity map.
    """
    dtype = raw_params.dtype
    raw_w = raw_params[..., :n_bins]
    raw_h = raw_params[..., n_bins : 2 * n_bins]
    raw_d = raw_params[..., 2 * n_bins :]

    min_frac = min(_MIN_BIN_FRACTION, 0.5 / float(n_bins))
    span = jnp.asarray(2.0 * bound, dtype=dtype)
    low = jnp.asarray(-bound, dtype=dtype)
    high = jnp.asarray(bound, dtype=dtype)

    w_frac = min_frac + (1.0 - n_bins * min_frac) * jax.nn.softmax(raw_w, axis=-1)
    h_frac = min_frac + (1.0 - n_bins * min_frac) * jax.nn.softmax(raw_h, axis=-1)

    cum_w = jnp.cumsum(w_frac * span, axis=-1)
    cum_h = jnp.cumsum(h_frac * span, axis=-1)

    lead_shape = (*raw_params.shape[:-1], 1)
    x_knots = jnp.concatenate(
        [
            jnp.full(lead_shape, low, dtype=dtype),
            low + cum_w[..., :-1],
            jnp.full(lead_shape, high, dtype=dtype),
        ],
        axis=-1,
    )
    y_knots = jnp.concatenate(
        [
            jnp.full(lead_shape, low, dtype=dtype),
            low + cum_h[..., :-1],
            jnp.full(lead_shape, high, dtype=dtype),
        ],
        axis=-1,
    )
    widths = x_knots[..., 1:] - x_knots[..., :-1]
    heights = y_knots[..., 1:] - y_knots[..., :-1]

    # Constant shift so softplus(0 + shift) + min_deriv == 1.0 identically.
    deriv_shift = float(np.log(np.expm1(1.0 - _MIN_SPLINE_DERIVATIVE)))
    inner_d = _MIN_SPLINE_DERIVATIVE + jax.nn.softplus(raw_d + deriv_shift)
    ones = jnp.ones(lead_shape, dtype=dtype)
    derivatives = jnp.concatenate([ones, inner_d, ones], axis=-1)
    return x_knots, widths, y_knots, heights, derivatives


def _rqs_forward_1d(
    x: jax.Array,
    raw_params: jax.Array,
    *,
    n_bins: int,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    """Elementwise forward Rational-Quadratic Spline with identity tails.

    Parameters
    ----------
    x : jax.Array, shape (...,)
        Input coordinates.
    raw_params : jax.Array, shape (..., 3 * n_bins - 1)
        Unconstrained spline parameters matching ``x.shape``.
    n_bins : int
        Number of spline segments on ``[-bound, bound]``.
    bound : float
        Half-width ``B > 0`` of the nonlinear spline box; identity outside.

    Returns
    -------
    y : jax.Array, shape (...,)
        Transformed coordinates.
    log_deriv : jax.Array, shape (...,)
        Log absolute derivative ``log |dy / dx|`` (0 outside ``[-bound, bound]``).
    """
    inside = (x >= -bound) & (x <= bound)
    # Double-where guard prevents NaN gradients when x is far outside [-B, B].
    safe_x = jnp.where(inside, x, jnp.zeros_like(x))

    x_knots, widths, y_knots, heights, derivs = _rqs_spline_knots(
        raw_params, n_bins=n_bins, bound=bound
    )
    bin_idx = jnp.sum(safe_x[..., None] >= x_knots[..., 1:-1], axis=-1)
    idx = bin_idx[..., None]
    idx_next = (bin_idx + 1)[..., None]

    x_k = jnp.take_along_axis(x_knots, idx, axis=-1)[..., 0]
    w_k = jnp.take_along_axis(widths, idx, axis=-1)[..., 0]
    y_k = jnp.take_along_axis(y_knots, idx, axis=-1)[..., 0]
    h_k = jnp.take_along_axis(heights, idx, axis=-1)[..., 0]
    d_k = jnp.take_along_axis(derivs, idx, axis=-1)[..., 0]
    d_k1 = jnp.take_along_axis(derivs, idx_next, axis=-1)[..., 0]

    s_k = h_k / w_k
    xi = jnp.clip((safe_x - x_k) / w_k, 0.0, 1.0)
    xi_1m = xi * (1.0 - xi)
    one_m_xi = 1.0 - xi

    num = h_k * (s_k * xi * xi + d_k * xi_1m)
    den = s_k + (d_k1 + d_k - 2.0 * s_k) * xi_1m
    y_in = y_k + num / den

    deriv_num = (s_k * s_k) * (
        d_k1 * xi * xi + 2.0 * s_k * xi_1m + d_k * one_m_xi * one_m_xi
    )
    log_deriv_in = jnp.log(deriv_num) - 2.0 * jnp.log(den)

    y = jnp.where(inside, y_in, x)
    log_deriv = jnp.where(inside, log_deriv_in, jnp.zeros_like(x))
    return y, log_deriv


def _rqs_inverse_1d(
    y: jax.Array,
    raw_params: jax.Array,
    *,
    n_bins: int,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    """Exact analytical inverse of ``_rqs_forward_1d``."""
    inside = (y >= -bound) & (y <= bound)
    safe_y = jnp.where(inside, y, jnp.zeros_like(y))

    x_knots, widths, y_knots, heights, derivs = _rqs_spline_knots(
        raw_params, n_bins=n_bins, bound=bound
    )
    bin_idx = jnp.sum(safe_y[..., None] >= y_knots[..., 1:-1], axis=-1)
    idx = bin_idx[..., None]
    idx_next = (bin_idx + 1)[..., None]

    x_k = jnp.take_along_axis(x_knots, idx, axis=-1)[..., 0]
    w_k = jnp.take_along_axis(widths, idx, axis=-1)[..., 0]
    y_k = jnp.take_along_axis(y_knots, idx, axis=-1)[..., 0]
    h_k = jnp.take_along_axis(heights, idx, axis=-1)[..., 0]
    d_k = jnp.take_along_axis(derivs, idx, axis=-1)[..., 0]
    d_k1 = jnp.take_along_axis(derivs, idx_next, axis=-1)[..., 0]

    s_k = h_k / w_k
    dy = jnp.clip(safe_y - y_k, 0.0, h_k)
    d_delta = d_k1 + d_k - 2.0 * s_k

    a = h_k * (s_k - d_k) + dy * d_delta
    b = h_k * d_k - dy * d_delta
    c = -s_k * dy
    disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    sqrt_disc = jnp.sqrt(disc)
    # Cancellation-free quadratic root in [0, 1]: when b >= 0 use 2c / (-b - sqrt_disc)
    # (exact even at a == 0); when b < 0, a >= -b > 0 holds strictly, so
    # (-b + sqrt_disc) / (2a) avoids subtracting two large nearly equal numbers.
    xi_b_pos = (2.0 * c) / jnp.where(b >= 0.0, -b - sqrt_disc, -1.0)
    xi_b_neg = (-b + sqrt_disc) / jnp.where(b < 0.0, 2.0 * a, 1.0)
    xi = jnp.clip(jnp.where(b >= 0.0, xi_b_pos, xi_b_neg), 0.0, 1.0)
    x_in = x_k + xi * w_k

    xi_1m = xi * (1.0 - xi)
    one_m_xi = 1.0 - xi
    den = s_k + d_delta * xi_1m
    deriv_num = (s_k * s_k) * (
        d_k1 * xi * xi + 2.0 * s_k * xi_1m + d_k * one_m_xi * one_m_xi
    )
    log_deriv_in = -(jnp.log(deriv_num) - 2.0 * jnp.log(den))

    x = jnp.where(inside, x_in, y)
    log_deriv = jnp.where(inside, log_deriv_in, jnp.zeros_like(y))
    return x, log_deriv


def _cayley_orthogonal(raw_matrix: jax.Array) -> jax.Array:
    """Cayley orthogonal matrix ``(I - A)(I + A)^{-1}`` from skew ``A``.

    Exact identity at ``raw_matrix = 0``; ``det Q = +1`` and ``Q @ Q.T = I``
    for every finite input.
    """
    n_dims = raw_matrix.shape[-1]
    skew = 0.5 * (raw_matrix - jnp.swapaxes(raw_matrix, -1, -2))
    eye = jnp.eye(n_dims, dtype=raw_matrix.dtype)
    return cast(jax.Array, jnp.linalg.solve(eye + skew, eye - skew))


def _dense_coupling_layer_forward(
    u: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
    b2: jax.Array,
    w_out: jax.Array,
    b_out: jax.Array,
    q_mat: jax.Array,
    cond_mask: jax.Array,
    *,
    n_bins: int,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    """One dense RQS coupling layer preceded by orthogonal mixing ``Q``."""
    u_rot = jnp.matmul(u, q_mat.T, precision=jax.lax.Precision.HIGHEST)
    cond_f = cond_mask.astype(u.dtype)
    trans_f = 1.0 - cond_f
    # Smoothly bound conditioning inputs so extreme outliers cannot overflow MLP.
    u_cond = bound * jnp.tanh((u_rot * cond_f[None, :]) / bound)
    h1 = jax.nn.silu(
        jnp.matmul(u_cond, w1, precision=jax.lax.Precision.HIGHEST) + b1[None, :]
    )
    h2 = jax.nn.silu(
        jnp.matmul(h1, w2, precision=jax.lax.Precision.HIGHEST) + b2[None, :]
    )
    raw = jnp.matmul(h2, w_out, precision=jax.lax.Precision.HIGHEST) + b_out[None, :]
    n_dims = u.shape[1]
    raw = raw.reshape((u.shape[0], n_dims, 3 * n_bins - 1))
    y_all, log_d_all = _rqs_forward_1d(u_rot, raw, n_bins=n_bins, bound=bound)
    z = jnp.where(cond_mask[None, :], u_rot, y_all)
    log_det = jnp.sum(log_d_all * trans_f[None, :], axis=1)
    return z, log_det


def _local_coupling_layer_forward(
    u: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
    b2: jax.Array,
    w_out: jax.Array,
    b_out: jax.Array,
    cond_nbr_idx: jax.Array,
    cond_nbr_mask: jax.Array,
    trans_idx: jax.Array,
    trans_valid: jax.Array,
    cond_geom: jax.Array,
    *,
    n_bins: int,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    """One weight-shared physically neighbor-masked RQS coupling layer."""
    n_dims = u.shape[1]
    c_loc = cond_nbr_idx.shape[1]
    # Pad one dummy column at index D so odd-channel padding slots write safely.
    u_ext = jnp.pad(u, ((0, 0), (0, 1)))
    u_trans = u_ext[:, trans_idx]  # (N, n_trans)
    u_nbr = u[:, cond_nbr_idx]  # (N, n_trans, C_loc)
    v_nbr = bound * jnp.tanh(u_nbr / bound) * cond_nbr_mask[None, :, :].astype(u.dtype)
    # Split w1 into amplitude (:c_loc) and static geometry bias (c_loc:) to
    # avoid broadcasting cond_geom over the spike batch dimension.
    geom_bias = (
        jnp.matmul(
            cond_geom.astype(u.dtype), w1[c_loc:], precision=jax.lax.Precision.HIGHEST
        )
        + b1[None, :]
    )
    h1 = jax.nn.silu(
        jnp.matmul(v_nbr, w1[:c_loc], precision=jax.lax.Precision.HIGHEST)
        + geom_bias[None, :, :]
    )
    h2 = jax.nn.silu(
        jnp.matmul(h1, w2, precision=jax.lax.Precision.HIGHEST) + b2[None, None, :]
    )
    raw = (
        jnp.matmul(h2, w_out, precision=jax.lax.Precision.HIGHEST)
        + b_out[None, None, :]
    )
    y_trans, log_d_trans = _rqs_forward_1d(u_trans, raw, n_bins=n_bins, bound=bound)
    z = u_ext.at[:, trans_idx].set(y_trans)[:, :n_dims]
    log_det = jnp.sum(log_d_trans * trans_valid[None, :].astype(u.dtype), axis=1)
    return z, log_det


def _apply_flow_forward_jax(
    u: jax.Array,
    flow_params: dict[str, jax.Array],
    *,
    flow_mode: str,
    n_bins: int,
    bound: float,
    cond_masks: jax.Array | None = None,
    cond_nbr_indices: jax.Array | None = None,
    cond_nbr_masks: jax.Array | None = None,
    trans_indices: jax.Array | None = None,
    trans_valid: jax.Array | None = None,
    cond_geom: jax.Array | None = None,
    precomputed_cayley: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply all layers of the RQS flow in JAX via ``lax.scan``.

    Returns ``(z, total_log_det_j)`` with shapes ``(N, D)`` and ``(N,)``.
    """
    n_spikes = u.shape[0]
    if flow_mode == "identity":
        return u, jnp.zeros((n_spikes,), dtype=u.dtype)

    if flow_mode == "elementwise":
        ew = flow_params["elementwise_params"]  # (L, 1, 3K - 1)

        def _ew_step(
            carry: tuple[jax.Array, jax.Array],
            layer_param: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            cur, acc_ld = carry
            raw = jnp.broadcast_to(
                layer_param[None, :, :], (cur.shape[0], 1, layer_param.shape[-1])
            )
            nxt, ld = _rqs_forward_1d(cur, raw, n_bins=n_bins, bound=bound)
            return (nxt, acc_ld + ld[:, 0]), None

        (z, total_ld), _ = jax.lax.scan(
            _ew_step, (u, jnp.zeros((n_spikes,), dtype=u.dtype)), ew
        )
        return z, total_ld

    if flow_mode == "dense":
        assert cond_masks is not None
        if precomputed_cayley is not None:
            q_mats = precomputed_cayley
        else:
            q_mats = jax.vmap(_cayley_orthogonal)(flow_params["cayley_raw"])
        xs = (
            flow_params["w1"],
            flow_params["b1"],
            flow_params["w2"],
            flow_params["b2"],
            flow_params["w_out"],
            flow_params["b_out"],
            q_mats,
            cond_masks,
        )

        def _dense_step(
            carry: tuple[jax.Array, jax.Array],
            layer: tuple[jax.Array, ...],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            cur, acc_ld = carry
            w1, b1, w2, b2, w_out, b_out, q_mat, c_mask = layer
            nxt, ld = _dense_coupling_layer_forward(
                cur,
                w1,
                b1,
                w2,
                b2,
                w_out,
                b_out,
                q_mat,
                c_mask,
                n_bins=n_bins,
                bound=bound,
            )
            return (nxt, acc_ld + ld), None

        (z, total_ld), _ = jax.lax.scan(
            _dense_step, (u, jnp.zeros((n_spikes,), dtype=u.dtype)), xs
        )
        return z, total_ld

    if flow_mode == "local":
        assert cond_nbr_indices is not None
        assert cond_nbr_masks is not None
        assert trans_indices is not None
        assert trans_valid is not None
        assert cond_geom is not None
        xs_loc = (
            flow_params["w1"],
            flow_params["b1"],
            flow_params["w2"],
            flow_params["b2"],
            flow_params["w_out"],
            flow_params["b_out"],
            cond_nbr_indices,
            cond_nbr_masks,
            trans_indices,
            trans_valid,
            cond_geom,
        )

        def _local_step(
            carry: tuple[jax.Array, jax.Array],
            layer: tuple[jax.Array, ...],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            cur, acc_ld = carry
            (
                w1,
                b1,
                w2,
                b2,
                w_out,
                b_out,
                nbr_i,
                nbr_m,
                t_idx,
                t_val,
                geom,
            ) = layer
            nxt, ld = _local_coupling_layer_forward(
                cur,
                w1,
                b1,
                w2,
                b2,
                w_out,
                b_out,
                nbr_i,
                nbr_m,
                t_idx,
                t_val,
                geom,
                n_bins=n_bins,
                bound=bound,
            )
            return (nxt, acc_ld + ld), None

        (z, total_ld), _ = jax.lax.scan(
            _local_step, (u, jnp.zeros((n_spikes,), dtype=u.dtype)), xs_loc
        )
        return z, total_ld

    msg = f"Unknown flow_mode {flow_mode!r}"
    raise ValidationError(msg)


# ---------------------------------------------------------------------------
# Latent low-rank-plus-diagonal Gaussian mixture kernels (cancellation-free)
# ---------------------------------------------------------------------------


def _latent_woodbury_log_density_jax(
    z: jax.Array,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    noise_variance: jax.Array,
    is_component: jax.Array,
) -> jax.Array:
    """Exact cancellation-free latent Gaussian log-density ``(N, K)``.

    Uses orthonormal eigenbasis representation ``(I - Q_k Q_k^T) + Q_k diag(lam_k) Q_k^T``
    in ``noise_variance``-standardized coordinates so the Mahalanobis distance
    is a pure sum of non-negative squares with zero float32 subtraction
    cancellation.
    """
    dtype = z.dtype
    n_dims = z.shape[1]
    inv_sqrt_psi = jax.lax.rsqrt(noise_variance)  # (D,)
    log_det_psi = jnp.sum(jnp.log(noise_variance))  # scalar

    # Standardized difference: (N, K, D).
    z_std = z * inv_sqrt_psi[None, :]  # (N, D)
    mu_std = means * inv_sqrt_psi[None, :]  # (K, D)
    delta = z_std[:, None, :] - mu_std[None, :, :]  # (N, K, D)

    # Projection onto orthonormal columns Q_k: (N, K, R).
    proj = jnp.einsum("nkd,kdr->nkr", delta, basis, precision=jax.lax.Precision.HIGHEST)
    # Orthogonal complement residual: delta - Q_k @ proj.
    recon = jnp.einsum("nkr,kdr->nkd", proj, basis, precision=jax.lax.Precision.HIGHEST)
    orth_sq = jnp.sum((delta - recon) ** 2, axis=-1)  # (N, K)
    par_sq = jnp.sum((proj * proj) / basis_variances[None, :, :], axis=-1)  # (N, K)

    log_det_k = log_det_psi + jnp.sum(jnp.log(basis_variances), axis=-1)  # (K,)
    log_const = -0.5 * (
        jnp.asarray(float(n_dims) * np.log(2.0 * np.pi), dtype=dtype) + log_det_k
    )
    log_dens = log_const[None, :] - 0.5 * (orth_sq + par_sq)
    return jnp.where(is_component[None, :], log_dens, -jnp.inf)


def _latent_woodbury_log_density_f64(
    z: np.ndarray,
    means: np.ndarray,
    basis: np.ndarray,
    basis_variances: np.ndarray,
    noise_variance: np.ndarray,
) -> np.ndarray:
    """Exact float64 latent Gaussian log-density ``(N, K)`` on host in bounded blocks."""
    n_spikes, n_dims = z.shape
    n_comp = means.shape[0]
    if n_spikes == 0:
        return np.empty((0, n_comp), dtype=np.float64)
    inv_sqrt_psi = 1.0 / np.sqrt(noise_variance)
    log_det_psi = float(np.sum(np.log(noise_variance)))
    mu_std = means * inv_sqrt_psi[None, :]
    log_det_k = log_det_psi + np.sum(np.log(basis_variances), axis=-1)
    log_const = -0.5 * (float(n_dims) * np.log(2.0 * np.pi) + log_det_k)
    out = np.empty((n_spikes, n_comp), dtype=np.float64)
    for start in range(0, n_spikes, _E_STEP_BLOCK):
        stop = min(start + _E_STEP_BLOCK, n_spikes)
        z_std = z[start:stop] * inv_sqrt_psi[None, :]
        delta = z_std[:, None, :] - mu_std[None, :, :]
        proj = np.einsum("nkd,kdr->nkr", delta, basis)
        recon = np.einsum("nkr,kdr->nkd", proj, basis)
        orth_sq = np.sum((delta - recon) ** 2, axis=-1)
        par_sq = np.sum((proj * proj) / basis_variances[None, :, :], axis=-1)
        out[start:stop] = log_const[None, :] - 0.5 * (orth_sq + par_sq)
    return out


def _project_covariances_to_orthonormal_basis(
    covariances: np.ndarray,
    noise_variance: np.ndarray,
    rank: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent ``(K, D, D)`` PSD covariances in ``noise_variance``-standardized eigenbasis.

    When ``rank is None`` or ``rank >= D``, retains all ``D`` eigenpairs (exact
    full covariance). When ``rank < D``, retains the top ``rank`` eigenvalues
    above ``1.0`` (Tipping-Bishop factor-analysis optimum at fixed ``Psi``).
    """
    n_comp, n_dims, _ = covariances.shape
    stored_rank = n_dims if (rank is None or rank >= n_dims) else int(rank)
    is_full = stored_rank >= n_dims
    inv_sqrt_psi = 1.0 / np.sqrt(noise_variance)

    basis = np.zeros((n_comp, n_dims, stored_rank), dtype=np.float64)
    basis_variances = np.ones((n_comp, stored_rank), dtype=np.float64)

    for k in range(n_comp):
        c_std = covariances[k] * inv_sqrt_psi[:, None] * inv_sqrt_psi[None, :]
        c_std = 0.5 * (c_std + c_std.T)
        eigvals, eigvecs = np.linalg.eigh(c_std)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if is_full:
            basis[k] = eigvecs[:, :n_dims]
            # Exact constrained MLE under Sigma >= diag(noise_variance): clamp
            # standardized eigenvalues at 1.0 only where the floor binds.
            basis_variances[k] = np.maximum(eigvals[:n_dims], 1.0)
        else:
            top_vals = eigvals[:stored_rank]
            top_vecs = eigvecs[:, :stored_rank]
            active = top_vals > 1.0
            if np.any(active):
                n_act = int(np.sum(active))
                basis[k, :, :n_act] = top_vecs[:, active]
                basis_variances[k, :n_act] = top_vals[active]
    return basis, basis_variances


def _latent_mark_m_step(
    z: np.ndarray,
    responsibilities: np.ndarray,
    spike_weights: np.ndarray,
    prev_means: np.ndarray,
    prev_basis: np.ndarray,
    prev_basis_variances: np.ndarray,
    prev_noise_variance: np.ndarray,
    *,
    covariance_rank: int | None,
    min_noise_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provably monotone ECM mark M-step in warped latent space ``z``.

    Updates component weights, means, shared diagonal residual variance ``Psi``
    (exact Tipping-Bishop / FA-EM update floored at ``min_noise_variance``),
    and orthonormal low-rank eigenbasis ``(basis, basis_variances)``.
    """
    n_comp = responsibilities.shape[1]
    n_dims = z.shape[1]
    is_full = covariance_rank is None or covariance_rank >= n_dims

    weighted_resp = responsibilities * spike_weights[:, None]  # (N, K)
    mass = np.sum(weighted_resp, axis=0)  # (K,)
    total_mass = float(np.sum(mass))
    weights = (
        mass / total_mass
        if total_mass > 0.0
        else np.full(n_comp, 1.0 / n_comp, dtype=np.float64)
    )

    means = np.array(prev_means, dtype=np.float64, copy=True)
    covariances = np.empty((n_comp, n_dims, n_dims), dtype=np.float64)
    active_comp = mass > _MIN_RESPONSIBILITY_MASS

    for k in range(n_comp):
        if active_comp[k]:
            wk = weighted_resp[:, k]
            mu_k = (wk @ z) / mass[k]
            means[k] = mu_k
            diff = z - mu_k[None, :]
            cov_k = (diff.T * wk[None, :]) @ diff / mass[k]
            cov_k = 0.5 * (cov_k + cov_k.T)
            covariances[k] = cov_k
        else:
            # Keep previous covariance reconstructed from previous basis.
            sqrt_psi = np.sqrt(prev_noise_variance)
            q_k = prev_basis[k]
            lam_m1 = prev_basis_variances[k] - 1.0
            c_std = np.eye(n_dims) + (q_k * lam_m1[None, :]) @ q_k.T
            covariances[k] = c_std * sqrt_psi[:, None] * sqrt_psi[None, :]

    if is_full:
        # Exact constrained Gaussian MLE under Sigma >= diag(min_noise_variance):
        # clamp standardized eigenvalues at 1.0 rather than shifting all modes.
        noise_variance = np.array(min_noise_variance, dtype=np.float64, copy=True)
        basis, basis_variances = _project_covariances_to_orthonormal_basis(
            covariances, noise_variance, rank=None
        )
        return weights, means, basis, basis_variances, noise_variance

    # Low-rank factor-analysis ECM update:
    # Step 1: exact Tipping-Bishop optimal loadings at prev_noise_variance.
    basis_old, var_old = _project_covariances_to_orthonormal_basis(
        covariances, prev_noise_variance, rank=covariance_rank
    )
    # Step 2: exact expected residual variance per channel under FA-EM:
    # e_{k, d} = psi_old_d * (C_std_{dd} - sum_a (lam_{ka} - 1) Q_{k, d, a}^2).
    inv_psi_old = 1.0 / prev_noise_variance
    resid_num = np.zeros(n_dims, dtype=np.float64)
    resid_den = 0.0
    for k in range(n_comp):
        if not active_comp[k]:
            continue
        c_diag_std = np.diag(covariances[k]) * inv_psi_old
        explained_std = np.sum(
            (var_old[k] - 1.0)[None, :] * (basis_old[k] ** 2), axis=1
        )
        e_k = np.maximum(prev_noise_variance * (c_diag_std - explained_std), 0.0)
        resid_num += mass[k] * e_k
        resid_den += mass[k]

    if resid_den > 0.0:
        noise_variance = np.maximum(resid_num / resid_den, min_noise_variance)
    else:
        noise_variance = np.maximum(prev_noise_variance, min_noise_variance)

    # Step 3: exact Tipping-Bishop projection at updated noise_variance.
    basis, basis_variances = _project_covariances_to_orthonormal_basis(
        covariances, noise_variance, rank=covariance_rank
    )
    return weights, means, basis, basis_variances, noise_variance


# ---------------------------------------------------------------------------
# Serialized FlowMarkMixture model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FlowMarkMixture:
    """A fitted normalizing-flow-warped low-rank-plus-diagonal mark mixture.

    Transforms raw marks ``m`` through Layer 0 (per-feature standardization or
    symmetric ZCA whitening) and ``n_flow_layers`` Rational-Quadratic Neural
    Spline (RQS) coupling layers into latent coordinates ``z = f_theta(m)``,
    then evaluates an orthonormal low-rank-plus-diagonal Gaussian mixture in
    ``z`` plus the exact ``log |det J_{f_theta}(m)|``.

    Stored exclusively as read-only NumPy arrays and Python scalars so it
    pickles cleanly, round-trips through ``.nld``, and runs prediction in pure
    JAX without requiring ``optax``.

    Attributes
    ----------
    weights : np.ndarray, shape (n_components,)
        Latent mixture component weights summing to 1.
    means : np.ndarray, shape (n_components, n_features)
        Component means in warped latent space ``z``.
    basis : np.ndarray, shape (n_components, n_features, rank)
        Orthonormal covariance eigenvectors in ``noise_variance``-standardized
        latent coordinates.
    basis_variances : np.ndarray, shape (n_components, rank)
        Positive eigenvalues along ``basis``.
    noise_variance : np.ndarray, shape (n_features,)
        Shared per-dimension residual variance in latent space ``z``.
    input_mean : np.ndarray, shape (n_features,)
        Layer 0 centering offset in raw mark units.
    input_scale : np.ndarray, shape (n_features,)
        Layer 0 positive per-feature scale (applied after optional
        ``noise_whitener`` to keep high-SNR peaks inside ``[-spline_bound,
        spline_bound]``).
    noise_whitener : np.ndarray or None, shape (n_features, n_features)
        Symmetric ZCA whitener ``Sigma_noise^{-1/2}``, or None.
    log_det_input : float
        Log Jacobian determinant of Layer 0.
    flow_mode : str
        One of ``"identity"``, ``"elementwise"``, ``"dense"``, ``"local"``.
    n_flow_layers : int
        Number of RQS layers.
    n_spline_bins : int
        Number of RQS segments on ``[-spline_bound, spline_bound]``.
    spline_bound : float
        Half-width ``B > 0`` of the RQS nonlinear interval.
    elementwise_params : np.ndarray or None
        Stacked ``(L, 1, 3*K - 1)`` parameters for ``flow_mode="elementwise"``.
    w1, b1, w2, b2, w_out, b_out : np.ndarray or None
        Stacked per-layer MLP conditioner parameters for ``"dense"`` or
        ``"local"`` modes.
    cayley_matrices : np.ndarray or None
        Precomputed ``(L, D, D)`` orthogonal mixing matrices for ``"dense"``.
    cond_masks : np.ndarray or None
        Boolean ``(L, D)`` conditioning masks for ``"dense"``.
    cond_nbr_indices : np.ndarray or None
        ``(L, n_trans, C_loc)`` int32 neighbor indices for ``"local"``.
    cond_nbr_masks : np.ndarray or None
        ``(L, n_trans, C_loc)`` boolean neighbor masks for ``"local"``.
    trans_indices : np.ndarray or None
        ``(L, n_trans)`` int32 transformed contact indices (``D`` on padded
        odd-channel slots) for ``"local"``.
    trans_valid : np.ndarray or None
        ``(L, n_trans)`` boolean validity mask for ``trans_indices``.
    cond_geom : np.ndarray or None
        ``(L, n_trans, F_geo)`` static geometry context for ``"local"``.
    """

    weights: np.ndarray
    means: np.ndarray
    basis: np.ndarray
    basis_variances: np.ndarray
    noise_variance: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    noise_whitener: np.ndarray | None
    log_det_input: float
    flow_mode: str
    n_flow_layers: int
    n_spline_bins: int
    spline_bound: float
    elementwise_params: np.ndarray | None = None
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: np.ndarray | None = None
    w_out: np.ndarray | None = None
    b_out: np.ndarray | None = None
    cayley_matrices: np.ndarray | None = None
    cond_masks: np.ndarray | None = None
    cond_nbr_indices: np.ndarray | None = None
    cond_nbr_masks: np.ndarray | None = None
    trans_indices: np.ndarray | None = None
    trans_valid: np.ndarray | None = None
    cond_geom: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.flow_mode not in FLOW_MODES:
            msg = f"flow_mode must be one of {FLOW_MODES}"
            raise ValidationError(msg, got=repr(self.flow_mode))
        for field_name in (
            "weights",
            "means",
            "basis",
            "basis_variances",
            "noise_variance",
            "input_mean",
            "input_scale",
        ):
            arr = _read_only_copy(getattr(self, field_name), dtype=np.float64)
            object.__setattr__(self, field_name, arr)

        for opt_float in (
            "noise_whitener",
            "elementwise_params",
            "w1",
            "b1",
            "w2",
            "b2",
            "w_out",
            "b_out",
            "cayley_matrices",
            "cond_geom",
        ):
            val = getattr(self, opt_float)
            if val is not None:
                object.__setattr__(
                    self, opt_float, _read_only_copy(val, dtype=np.float64)
                )

        for opt_int in ("cond_nbr_indices", "trans_indices"):
            val = getattr(self, opt_int)
            if val is not None:
                object.__setattr__(self, opt_int, _read_only_copy(val, dtype=np.int32))
        for opt_bool in ("cond_masks", "cond_nbr_masks", "trans_valid"):
            val = getattr(self, opt_bool)
            if val is not None:
                object.__setattr__(self, opt_bool, _read_only_copy(val, dtype=bool))

        object.__setattr__(self, "log_det_input", float(self.log_det_input))
        object.__setattr__(self, "n_flow_layers", int(self.n_flow_layers))
        object.__setattr__(self, "n_spline_bins", int(self.n_spline_bins))
        object.__setattr__(self, "spline_bound", float(self.spline_bound))

    @property
    def n_components(self) -> int:
        """Number of latent Gaussian components."""
        return len(self.weights)

    @property
    def n_features(self) -> int:
        """Number of mark dimensions modeled."""
        return len(self.input_mean)

    @property
    def basis_rank(self) -> int:
        """Stored latent covariance rank."""
        return int(self.basis.shape[2])

    def standardize_marks(self, marks: np.ndarray) -> np.ndarray:
        """Apply Layer 0 centering + optional ZCA whitening + scaling in float64."""
        return _standardize_host(
            np.asarray(marks, dtype=np.float64),
            self.input_mean,
            self.input_scale,
            self.noise_whitener,
        )

    def unstandardize_marks(self, u: np.ndarray) -> np.ndarray:
        """Exact inverse of ``standardize_marks`` in float64."""
        arr = np.asarray(u, dtype=np.float64)
        scaled = arr * self.input_scale[None, :]
        if self.noise_whitener is not None:
            centered = np.linalg.solve(self.noise_whitener, scaled.T).T
        else:
            centered = scaled
        return cast(np.ndarray, centered + self.input_mean[None, :])

    def _jax_flow_params(self, dtype: Any) -> dict[str, jax.Array]:
        """Pack stacked flow parameters as JAX arrays of ``dtype``."""
        if self.flow_mode == "identity":
            return {}
        if self.flow_mode == "elementwise":
            assert self.elementwise_params is not None
            return {
                "elementwise_params": jnp.asarray(self.elementwise_params, dtype=dtype)
            }
        assert self.w1 is not None
        assert self.b1 is not None
        assert self.w2 is not None
        assert self.b2 is not None
        assert self.w_out is not None
        assert self.b_out is not None
        return {
            "w1": jnp.asarray(self.w1, dtype=dtype),
            "b1": jnp.asarray(self.b1, dtype=dtype),
            "w2": jnp.asarray(self.w2, dtype=dtype),
            "b2": jnp.asarray(self.b2, dtype=dtype),
            "w_out": jnp.asarray(self.w_out, dtype=dtype),
            "b_out": jnp.asarray(self.b_out, dtype=dtype),
        }

    def forward(self, marks: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Transform raw marks ``m -> (z, log |det J_{f_theta}(m)|)`` in float64.

        Parameters
        ----------
        marks : array_like, shape (n_spikes, n_features)
            Waveform features in raw mark units.

        Returns
        -------
        z : np.ndarray, shape (n_spikes, n_features)
            Latent coordinates in float64.
        log_det_j : np.ndarray, shape (n_spikes,)
            Exact log Jacobian determinant ``log |det df/dm|`` per spike.
        """
        u = self.standardize_marks(np.asarray(marks, dtype=np.float64))
        n_spikes = u.shape[0]
        if n_spikes == 0 or self.flow_mode == "identity":
            return u, np.full((n_spikes,), self.log_det_input, dtype=np.float64)

        # Evaluate in float64 via numpy-backed loop for exact diagnostics/tests.
        cur = u
        acc_ld = np.full((n_spikes,), self.log_det_input, dtype=np.float64)
        for layer in range(self.n_flow_layers):
            cur, ld = self._forward_layer_f64(cur, layer)
            acc_ld = acc_ld + ld
        return cur, acc_ld

    def inverse(self, z: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Exact analytical inverse ``z -> (m, log |det J_{f^{-1}}(z)|)`` in float64.

        Parameters
        ----------
        z : array_like, shape (n_spikes, n_features)
            Latent coordinates.

        Returns
        -------
        marks : np.ndarray, shape (n_spikes, n_features)
            Recovered raw waveform features.
        log_det_inv : np.ndarray, shape (n_spikes,)
            Exact log Jacobian determinant of the inverse map.
        """
        cur = np.asarray(z, dtype=np.float64)
        n_spikes = cur.shape[0]
        acc_ld = np.full((n_spikes,), -self.log_det_input, dtype=np.float64)
        if n_spikes > 0 and self.flow_mode != "identity":
            for layer in range(self.n_flow_layers - 1, -1, -1):
                cur, ld = self._inverse_layer_f64(cur, layer)
                acc_ld = acc_ld + ld
        return self.unstandardize_marks(cur), acc_ld

    def _forward_layer_f64(
        self, u: np.ndarray, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact float64 forward pass of one flow layer on host."""
        k_bins = self.n_spline_bins
        bound = self.spline_bound
        if self.flow_mode == "elementwise":
            assert self.elementwise_params is not None
            raw = np.broadcast_to(
                self.elementwise_params[layer][None, :, :],
                (u.shape[0], 1, 3 * k_bins - 1),
            )
            y, ld = _rqs_forward_1d_np(u, raw, n_bins=k_bins, bound=bound)
            return y, ld[:, 0]

        if self.flow_mode == "dense":
            assert self.w1 is not None
            assert self.b1 is not None
            assert self.w2 is not None
            assert self.b2 is not None
            assert self.w_out is not None
            assert self.b_out is not None
            assert self.cayley_matrices is not None
            assert self.cond_masks is not None
            u_rot = u @ self.cayley_matrices[layer].T
            c_mask = self.cond_masks[layer]
            raw = _dense_conditioner_np(
                u_rot,
                c_mask,
                self.w1[layer],
                self.b1[layer],
                self.w2[layer],
                self.b2[layer],
                self.w_out[layer],
                self.b_out[layer],
                n_bins=k_bins,
                bound=bound,
            )
            y_all, ld_all = _rqs_forward_1d_np(u_rot, raw, n_bins=k_bins, bound=bound)
            z = np.where(c_mask[None, :], u_rot, y_all)
            ld = np.sum(ld_all * (~c_mask)[None, :], axis=1)
            return z, ld

        assert self.flow_mode == "local"
        assert self.w1 is not None
        assert self.b1 is not None
        assert self.w2 is not None
        assert self.b2 is not None
        assert self.w_out is not None
        assert self.b_out is not None
        assert self.cond_nbr_indices is not None
        assert self.cond_nbr_masks is not None
        assert self.trans_indices is not None
        assert self.trans_valid is not None
        assert self.cond_geom is not None
        n_dims = u.shape[1]
        t_idx = self.trans_indices[layer]
        t_valid = self.trans_valid[layer]
        u_ext = np.pad(u, ((0, 0), (0, 1)))
        u_trans = u_ext[:, t_idx]
        raw = _local_conditioner_np(
            u,
            self.cond_nbr_indices[layer],
            self.cond_nbr_masks[layer],
            self.cond_geom[layer],
            self.w1[layer],
            self.b1[layer],
            self.w2[layer],
            self.b2[layer],
            self.w_out[layer],
            self.b_out[layer],
            bound=bound,
        )
        y_trans, ld_trans = _rqs_forward_1d_np(u_trans, raw, n_bins=k_bins, bound=bound)
        u_ext[:, t_idx] = y_trans
        z = u_ext[:, :n_dims]
        ld = np.sum(ld_trans * t_valid[None, :], axis=1)
        return z, ld

    def _inverse_layer_f64(
        self, z: np.ndarray, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact float64 inverse pass of one flow layer on host."""
        k_bins = self.n_spline_bins
        bound = self.spline_bound
        if self.flow_mode == "elementwise":
            assert self.elementwise_params is not None
            raw = np.broadcast_to(
                self.elementwise_params[layer][None, :, :],
                (z.shape[0], 1, 3 * k_bins - 1),
            )
            x, ld = _rqs_inverse_1d_np(z, raw, n_bins=k_bins, bound=bound)
            return x, ld[:, 0]

        if self.flow_mode == "dense":
            assert self.w1 is not None
            assert self.b1 is not None
            assert self.w2 is not None
            assert self.b2 is not None
            assert self.w_out is not None
            assert self.b_out is not None
            assert self.cayley_matrices is not None
            assert self.cond_masks is not None
            c_mask = self.cond_masks[layer]
            raw = _dense_conditioner_np(
                z,
                c_mask,
                self.w1[layer],
                self.b1[layer],
                self.w2[layer],
                self.b2[layer],
                self.w_out[layer],
                self.b_out[layer],
                n_bins=k_bins,
                bound=bound,
            )
            x_all, ld_all = _rqs_inverse_1d_np(z, raw, n_bins=k_bins, bound=bound)
            u_rot = np.where(c_mask[None, :], z, x_all)
            u = u_rot @ self.cayley_matrices[layer]  # Q^{-1} = Q^T -> row @ Q
            ld = np.sum(ld_all * (~c_mask)[None, :], axis=1)
            return u, ld

        assert self.flow_mode == "local"
        assert self.w1 is not None
        assert self.b1 is not None
        assert self.w2 is not None
        assert self.b2 is not None
        assert self.w_out is not None
        assert self.b_out is not None
        assert self.cond_nbr_indices is not None
        assert self.cond_nbr_masks is not None
        assert self.trans_indices is not None
        assert self.trans_valid is not None
        assert self.cond_geom is not None
        n_dims = z.shape[1]
        t_idx = self.trans_indices[layer]
        t_valid = self.trans_valid[layer]
        z_ext = np.pad(z, ((0, 0), (0, 1)))
        z_trans = z_ext[:, t_idx]
        raw = _local_conditioner_np(
            z,
            self.cond_nbr_indices[layer],
            self.cond_nbr_masks[layer],
            self.cond_geom[layer],
            self.w1[layer],
            self.b1[layer],
            self.w2[layer],
            self.b2[layer],
            self.w_out[layer],
            self.b_out[layer],
            bound=bound,
        )
        x_trans, ld_trans = _rqs_inverse_1d_np(z_trans, raw, n_bins=k_bins, bound=bound)
        z_ext[:, t_idx] = x_trans
        u = z_ext[:, :n_dims]
        ld = np.sum(ld_trans * t_valid[None, :], axis=1)
        return u, ld

    def log_component_density(self, marks: ArrayLike) -> np.ndarray:
        """Log density ``log p_k(m)`` for each spike and component in float64.

        Parameters
        ----------
        marks : array_like, shape (n_spikes, n_features)
            Waveform features in raw mark units.

        Returns
        -------
        log_density : np.ndarray, shape (n_spikes, n_components)
            ``log |det J_{f_theta}(m)| + log N(f_theta(m); mu_k, Sigma_k)``.
        """
        arr = np.asarray(marks, dtype=np.float64)
        if arr.shape[0] == 0:
            return np.empty((0, self.n_components), dtype=np.float64)
        z, log_det = self.forward(arr)
        log_gauss = _latent_woodbury_log_density_f64(
            z, self.means, self.basis, self.basis_variances, self.noise_variance
        )
        return cast(np.ndarray, log_det[:, None] + log_gauss)

    def responsibilities(self, marks: ArrayLike) -> np.ndarray:
        """Position-free component responsibilities ``r_ik`` in float64."""
        log_dens = self.log_component_density(marks)
        if log_dens.shape[0] == 0:
            return np.empty((0, self.n_components), dtype=np.float64)
        log_weighted = log_dens + np.log(np.maximum(self.weights, 1e-300))[None, :]
        log_norm = logsumexp(log_weighted, axis=1, keepdims=True)
        return cast(np.ndarray, np.exp(log_weighted - log_norm))


# ---------------------------------------------------------------------------
# Pure NumPy float64 RQS helpers (exact host diagnostics & inverse checks)
# ---------------------------------------------------------------------------


def _silu_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable SiLU ``x * sigmoid(x)`` in float64."""
    clipped = np.clip(x, -500.0, 500.0)
    return cast(np.ndarray, x / (1.0 + np.exp(-clipped)))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable last-axis softmax in float64."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return cast(np.ndarray, exp_x / np.sum(exp_x, axis=-1, keepdims=True))


def _softplus_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus ``log(1 + exp(x))`` in float64."""
    return cast(np.ndarray, np.logaddexp(0.0, x))


def _rqs_knots_np(
    raw_params: np.ndarray, *, n_bins: int, bound: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Float64 NumPy mirror of ``_rqs_spline_knots``."""
    raw_w = raw_params[..., :n_bins]
    raw_h = raw_params[..., n_bins : 2 * n_bins]
    raw_d = raw_params[..., 2 * n_bins :]

    min_frac = min(_MIN_BIN_FRACTION, 0.5 / float(n_bins))
    span = 2.0 * bound
    w_frac = min_frac + (1.0 - n_bins * min_frac) * _softmax_np(raw_w)
    h_frac = min_frac + (1.0 - n_bins * min_frac) * _softmax_np(raw_h)
    cum_w = np.cumsum(w_frac * span, axis=-1)
    cum_h = np.cumsum(h_frac * span, axis=-1)

    lead = (*raw_params.shape[:-1], 1)
    x_knots = np.concatenate(
        [np.full(lead, -bound), -bound + cum_w[..., :-1], np.full(lead, bound)],
        axis=-1,
    )
    y_knots = np.concatenate(
        [np.full(lead, -bound), -bound + cum_h[..., :-1], np.full(lead, bound)],
        axis=-1,
    )
    widths = x_knots[..., 1:] - x_knots[..., :-1]
    heights = y_knots[..., 1:] - y_knots[..., :-1]

    deriv_shift = float(np.log(np.expm1(1.0 - _MIN_SPLINE_DERIVATIVE)))
    inner_d = _MIN_SPLINE_DERIVATIVE + _softplus_np(raw_d + deriv_shift)
    ones = np.ones(lead, dtype=np.float64)
    derivs = np.concatenate([ones, inner_d, ones], axis=-1)
    return x_knots, widths, y_knots, heights, derivs


def _rqs_forward_1d_np(
    x: np.ndarray, raw_params: np.ndarray, *, n_bins: int, bound: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact float64 forward RQS on host."""
    inside = (x >= -bound) & (x <= bound)
    safe_x = np.where(inside, x, 0.0)
    x_knots, widths, y_knots, heights, derivs = _rqs_knots_np(
        raw_params, n_bins=n_bins, bound=bound
    )
    bin_idx = np.sum(safe_x[..., None] >= x_knots[..., 1:-1], axis=-1)
    idx = bin_idx[..., None]
    idx_next = (bin_idx + 1)[..., None]

    x_k = np.take_along_axis(x_knots, idx, axis=-1)[..., 0]
    w_k = np.take_along_axis(widths, idx, axis=-1)[..., 0]
    y_k = np.take_along_axis(y_knots, idx, axis=-1)[..., 0]
    h_k = np.take_along_axis(heights, idx, axis=-1)[..., 0]
    d_k = np.take_along_axis(derivs, idx, axis=-1)[..., 0]
    d_k1 = np.take_along_axis(derivs, idx_next, axis=-1)[..., 0]

    s_k = h_k / w_k
    xi = np.clip((safe_x - x_k) / w_k, 0.0, 1.0)
    xi_1m = xi * (1.0 - xi)
    one_m_xi = 1.0 - xi
    num = h_k * (s_k * xi * xi + d_k * xi_1m)
    den = s_k + (d_k1 + d_k - 2.0 * s_k) * xi_1m
    y_in = y_k + num / den
    deriv_num = (s_k * s_k) * (
        d_k1 * xi * xi + 2.0 * s_k * xi_1m + d_k * one_m_xi * one_m_xi
    )
    log_d_in = np.log(deriv_num) - 2.0 * np.log(den)
    return np.where(inside, y_in, x), np.where(inside, log_d_in, 0.0)


def _rqs_inverse_1d_np(
    y: np.ndarray, raw_params: np.ndarray, *, n_bins: int, bound: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact float64 analytical inverse RQS on host."""
    inside = (y >= -bound) & (y <= bound)
    safe_y = np.where(inside, y, 0.0)
    x_knots, widths, y_knots, heights, derivs = _rqs_knots_np(
        raw_params, n_bins=n_bins, bound=bound
    )
    bin_idx = np.sum(safe_y[..., None] >= y_knots[..., 1:-1], axis=-1)
    idx = bin_idx[..., None]
    idx_next = (bin_idx + 1)[..., None]

    x_k = np.take_along_axis(x_knots, idx, axis=-1)[..., 0]
    w_k = np.take_along_axis(widths, idx, axis=-1)[..., 0]
    y_k = np.take_along_axis(y_knots, idx, axis=-1)[..., 0]
    h_k = np.take_along_axis(heights, idx, axis=-1)[..., 0]
    d_k = np.take_along_axis(derivs, idx, axis=-1)[..., 0]
    d_k1 = np.take_along_axis(derivs, idx_next, axis=-1)[..., 0]

    s_k = h_k / w_k
    dy = np.clip(safe_y - y_k, 0.0, h_k)
    d_delta = d_k1 + d_k - 2.0 * s_k
    a = h_k * (s_k - d_k) + dy * d_delta
    b = h_k * d_k - dy * d_delta
    c = -s_k * dy
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    sqrt_disc = np.sqrt(disc)
    xi_b_pos = (2.0 * c) / np.where(b >= 0.0, -b - sqrt_disc, -1.0)
    xi_b_neg = (-b + sqrt_disc) / np.where(b < 0.0, 2.0 * a, 1.0)
    xi = np.clip(np.where(b >= 0.0, xi_b_pos, xi_b_neg), 0.0, 1.0)
    x_in = x_k + xi * w_k

    xi_1m = xi * (1.0 - xi)
    one_m_xi = 1.0 - xi
    den = s_k + d_delta * xi_1m
    deriv_num = (s_k * s_k) * (
        d_k1 * xi * xi + 2.0 * s_k * xi_1m + d_k * one_m_xi * one_m_xi
    )
    log_d_in = -(np.log(deriv_num) - 2.0 * np.log(den))
    return np.where(inside, x_in, y), np.where(inside, log_d_in, 0.0)


def _dense_conditioner_np(
    u: np.ndarray,
    cond_mask: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w_out: np.ndarray,
    b_out: np.ndarray,
    *,
    n_bins: int,
    bound: float,
) -> np.ndarray:
    """Host float64 mirror of dense MLP conditioner."""
    u_cond = bound * np.tanh((u * cond_mask[None, :]) / bound)
    h1 = _silu_np(u_cond @ w1 + b1[None, :])
    h2 = _silu_np(h1 @ w2 + b2[None, :])
    raw = h2 @ w_out + b_out[None, :]
    return cast(np.ndarray, raw.reshape((u.shape[0], u.shape[1], 3 * n_bins - 1)))


def _local_conditioner_np(
    u: np.ndarray,
    cond_nbr_idx: np.ndarray,
    cond_nbr_mask: np.ndarray,
    cond_geom: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w_out: np.ndarray,
    b_out: np.ndarray,
    *,
    bound: float,
) -> np.ndarray:
    """Host float64 mirror of weight-shared physically neighbor-masked conditioner."""
    c_loc = cond_nbr_idx.shape[1]
    v_nbr = bound * np.tanh(u[:, cond_nbr_idx] / bound) * cond_nbr_mask[None, :, :]
    geom_bias = cond_geom @ w1[c_loc:] + b1[None, :]
    h1 = _silu_np(v_nbr @ w1[:c_loc] + geom_bias[None, :, :])
    h2 = _silu_np(h1 @ w2 + b2[None, None, :])
    return cast(np.ndarray, h2 @ w_out + b_out[None, None, :])


# ---------------------------------------------------------------------------
# Flow initialization and optax-based training step
# ---------------------------------------------------------------------------


class _FlowTopology(NamedTuple):
    """Static architecture metadata shared across folds of one electrode."""

    flow_mode: str
    n_flow_layers: int
    n_spline_bins: int
    spline_bound: float
    cond_masks: np.ndarray | None
    cond_nbr_indices: np.ndarray | None
    cond_nbr_masks: np.ndarray | None
    trans_indices: np.ndarray | None
    trans_valid: np.ndarray | None
    cond_geom: np.ndarray | None


def _build_flow_topology(
    n_features: int,
    contact_positions: np.ndarray | None,
    *,
    n_flow_layers: int,
    n_spline_bins: int,
    spline_bound: float,
    neighborhood_radius: float | None,
    max_local_channels: int | None,
) -> _FlowTopology:
    """Resolve flow mode and static coupling masks from dimension and geometry."""
    if n_flow_layers == 0:
        return _FlowTopology(
            flow_mode="identity",
            n_flow_layers=0,
            n_spline_bins=n_spline_bins,
            spline_bound=spline_bound,
            cond_masks=None,
            cond_nbr_indices=None,
            cond_nbr_masks=None,
            trans_indices=None,
            trans_valid=None,
            cond_geom=None,
        )
    if n_features == 1:
        return _FlowTopology(
            flow_mode="elementwise",
            n_flow_layers=n_flow_layers,
            n_spline_bins=n_spline_bins,
            spline_bound=spline_bound,
            cond_masks=None,
            cond_nbr_indices=None,
            cond_nbr_masks=None,
            trans_indices=None,
            trans_valid=None,
            cond_geom=None,
        )
    if contact_positions is None:
        cond_masks = np.stack(
            [
                (np.arange(n_features) % 2) == (layer % 2)
                for layer in range(n_flow_layers)
            ],
            axis=0,
        )
        return _FlowTopology(
            flow_mode="dense",
            n_flow_layers=n_flow_layers,
            n_spline_bins=n_spline_bins,
            spline_bound=spline_bound,
            cond_masks=cond_masks,
            cond_nbr_indices=None,
            cond_nbr_masks=None,
            trans_indices=None,
            trans_valid=None,
            cond_geom=None,
        )
    (
        cond_nbr_indices,
        cond_nbr_masks,
        trans_indices,
        trans_valid,
        cond_geom,
    ) = _build_local_coupling_topology(
        contact_positions,
        n_flow_layers,
        neighborhood_radius=neighborhood_radius,
        max_local_channels=max_local_channels,
    )
    return _FlowTopology(
        flow_mode="local",
        n_flow_layers=n_flow_layers,
        n_spline_bins=n_spline_bins,
        spline_bound=spline_bound,
        cond_masks=None,
        cond_nbr_indices=cond_nbr_indices,
        cond_nbr_masks=cond_nbr_masks,
        trans_indices=trans_indices,
        trans_valid=trans_valid,
        cond_geom=cond_geom,
    )


def _init_identity_flow_params(
    topology: _FlowTopology,
    n_features: int,
    flow_hidden_units: int,
    rng: np.random.Generator,
) -> dict[str, jax.Array]:
    """Initialize flow parameters so the map is the exact identity at epoch 0."""
    n_layers = topology.n_flow_layers
    p_dim = 3 * topology.n_spline_bins - 1
    if topology.flow_mode == "identity":
        return {}
    if topology.flow_mode == "elementwise":
        return {
            "elementwise_params": jnp.zeros((n_layers, 1, p_dim), dtype=jnp.float32)
        }
    if topology.flow_mode == "dense":
        h = int(flow_hidden_units)
        s1 = float(np.sqrt(2.0 / (n_features + h)))
        s2 = float(np.sqrt(2.0 / (h + h)))
        return {
            "w1": jnp.asarray(
                rng.normal(0.0, s1, size=(n_layers, n_features, h)), dtype=jnp.float32
            ),
            "b1": jnp.zeros((n_layers, h), dtype=jnp.float32),
            "w2": jnp.asarray(
                rng.normal(0.0, s2, size=(n_layers, h, h)), dtype=jnp.float32
            ),
            "b2": jnp.zeros((n_layers, h), dtype=jnp.float32),
            "w_out": jnp.zeros((n_layers, h, n_features * p_dim), dtype=jnp.float32),
            "b_out": jnp.zeros((n_layers, n_features * p_dim), dtype=jnp.float32),
            "cayley_raw": jnp.zeros(
                (n_layers, n_features, n_features), dtype=jnp.float32
            ),
        }

    assert topology.flow_mode == "local"
    assert topology.cond_nbr_indices is not None
    assert topology.cond_geom is not None
    c_loc = int(topology.cond_nbr_indices.shape[2])
    f_geo = int(topology.cond_geom.shape[2])
    f_in = c_loc + f_geo
    h_loc = min(int(flow_hidden_units), _MAX_LOCAL_HIDDEN_UNITS)
    s1 = float(np.sqrt(2.0 / (f_in + h_loc)))
    s2 = float(np.sqrt(2.0 / (h_loc + h_loc)))
    return {
        "w1": jnp.asarray(
            rng.normal(0.0, s1, size=(n_layers, f_in, h_loc)),
            dtype=jnp.float32,
        ),
        "b1": jnp.zeros((n_layers, h_loc), dtype=jnp.float32),
        "w2": jnp.asarray(
            rng.normal(0.0, s2, size=(n_layers, h_loc, h_loc)),
            dtype=jnp.float32,
        ),
        "b2": jnp.zeros((n_layers, h_loc), dtype=jnp.float32),
        "w_out": jnp.zeros((n_layers, h_loc, p_dim), dtype=jnp.float32),
        "b_out": jnp.zeros((n_layers, p_dim), dtype=jnp.float32),
    }


def _make_flow_step_kernel(
    optimizer: Any,
    *,
    flow_mode: str,
    n_bins: int,
    bound: float,
) -> Any:
    """Build one static-shape JIT-compiled ``lax.scan`` flow-optimization kernel.

    Compiles once per ``(flow_mode, n_bins, bound, optimizer_config)`` and
    runs ``n_steps`` optax updates over pre-batched ``(n_steps, batch, ...)``
    arrays, avoiding recompilation across folds and electrodes.
    """

    @jax.jit
    def _run_steps(
        params: dict[str, jax.Array],
        opt_state: Any,
        u_batches: jax.Array,
        w_batches: jax.Array,
        log_weight_batches: jax.Array,
        means: jax.Array,
        basis: jax.Array,
        basis_variances: jax.Array,
        noise_variance: jax.Array,
        is_component: jax.Array,
        cond_masks: jax.Array | None,
        cond_nbr_indices: jax.Array | None,
        cond_nbr_masks: jax.Array | None,
        trans_indices: jax.Array | None,
        trans_valid: jax.Array | None,
        cond_geom: jax.Array | None,
    ) -> tuple[dict[str, jax.Array], Any]:
        def _loss(
            p: dict[str, jax.Array],
            u_b: jax.Array,
            w_b: jax.Array,
            lw_b: jax.Array,
        ) -> jax.Array:
            z, log_det = _apply_flow_forward_jax(
                u_b,
                p,
                flow_mode=flow_mode,
                n_bins=n_bins,
                bound=bound,
                cond_masks=cond_masks,
                cond_nbr_indices=cond_nbr_indices,
                cond_nbr_masks=cond_nbr_masks,
                trans_indices=trans_indices,
                trans_valid=trans_valid,
                cond_geom=cond_geom,
            )
            log_gauss = _latent_woodbury_log_density_jax(
                z, means, basis, basis_variances, noise_variance, is_component
            )
            # Exact marginal over mixture components: log sum_k w_ik * N(z_i; mu_k, Sigma_k).
            # Zero out padded components before logsumexp.
            log_terms = jnp.where(is_component[None, :], lw_b + log_gauss, -jnp.inf)
            log_mix = jax.nn.logsumexp(log_terms, axis=1)
            per_spike = log_det + log_mix
            # Guard padded rows (weight == 0) so -inf * 0 never yields NaN.
            safe_per_spike = jnp.where(w_b > 0.0, per_spike, 0.0)
            denom = jnp.maximum(jnp.sum(w_b), 1e-12)
            return -jnp.sum(w_b * safe_per_spike) / denom

        val_and_grad = jax.value_and_grad(_loss)

        def _scan_body(
            carry: tuple[dict[str, jax.Array], Any],
            batch: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, jax.Array], Any], None]:
            cur_params, cur_opt = carry
            u_b, w_b, lw_b = batch
            _, grads = val_and_grad(cur_params, u_b, w_b, lw_b)
            # Guard any non-finite gradient step by skipping its update.
            grad_finite = jnp.all(
                jnp.asarray([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)])
            )
            updates, next_opt = optimizer.update(grads, cur_opt, cur_params)
            next_params = jax.tree.map(lambda p, u: p + u, cur_params, updates)
            out_params = jax.tree.map(
                lambda n, o: jnp.where(grad_finite, n, o), next_params, cur_params
            )
            out_opt = jax.tree.map(
                lambda n, o: jnp.where(grad_finite, n, o), next_opt, cur_opt
            )
            return (out_params, out_opt), None

        (final_params, final_opt), _ = jax.lax.scan(
            _scan_body,
            (params, opt_state),
            (u_batches, w_batches, log_weight_batches),
        )
        return final_params, final_opt

    return _run_steps


def _pack_flow_mixture(
    *,
    weights: np.ndarray,
    means: np.ndarray,
    basis: np.ndarray,
    basis_variances: np.ndarray,
    noise_variance: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    noise_whitener: np.ndarray | None,
    log_det_input: float,
    topology: _FlowTopology,
    flow_params: dict[str, jax.Array],
) -> FlowMarkMixture:
    """Freeze training JAX arrays into a serializable ``FlowMarkMixture``."""
    mode = topology.flow_mode
    if mode == "identity":
        return FlowMarkMixture(
            weights=weights,
            means=means,
            basis=basis,
            basis_variances=basis_variances,
            noise_variance=noise_variance,
            input_mean=input_mean,
            input_scale=input_scale,
            noise_whitener=noise_whitener,
            log_det_input=log_det_input,
            flow_mode="identity",
            n_flow_layers=0,
            n_spline_bins=topology.n_spline_bins,
            spline_bound=topology.spline_bound,
        )
    if mode == "elementwise":
        return FlowMarkMixture(
            weights=weights,
            means=means,
            basis=basis,
            basis_variances=basis_variances,
            noise_variance=noise_variance,
            input_mean=input_mean,
            input_scale=input_scale,
            noise_whitener=noise_whitener,
            log_det_input=log_det_input,
            flow_mode="elementwise",
            n_flow_layers=topology.n_flow_layers,
            n_spline_bins=topology.n_spline_bins,
            spline_bound=topology.spline_bound,
            elementwise_params=np.asarray(
                flow_params["elementwise_params"], dtype=np.float64
            ),
        )
    if mode == "dense":
        raw_cayley = np.asarray(flow_params["cayley_raw"], dtype=np.float64)
        n_layers, n_dims, _ = raw_cayley.shape
        eye = np.eye(n_dims, dtype=np.float64)
        q_mats = np.empty_like(raw_cayley)
        for layer in range(n_layers):
            skew = 0.5 * (raw_cayley[layer] - raw_cayley[layer].T)
            q_mats[layer] = np.linalg.solve(eye + skew, eye - skew)
        return FlowMarkMixture(
            weights=weights,
            means=means,
            basis=basis,
            basis_variances=basis_variances,
            noise_variance=noise_variance,
            input_mean=input_mean,
            input_scale=input_scale,
            noise_whitener=noise_whitener,
            log_det_input=log_det_input,
            flow_mode="dense",
            n_flow_layers=topology.n_flow_layers,
            n_spline_bins=topology.n_spline_bins,
            spline_bound=topology.spline_bound,
            w1=np.asarray(flow_params["w1"], dtype=np.float64),
            b1=np.asarray(flow_params["b1"], dtype=np.float64),
            w2=np.asarray(flow_params["w2"], dtype=np.float64),
            b2=np.asarray(flow_params["b2"], dtype=np.float64),
            w_out=np.asarray(flow_params["w_out"], dtype=np.float64),
            b_out=np.asarray(flow_params["b_out"], dtype=np.float64),
            cayley_matrices=q_mats,
            cond_masks=topology.cond_masks,
        )

    assert mode == "local"
    return FlowMarkMixture(
        weights=weights,
        means=means,
        basis=basis,
        basis_variances=basis_variances,
        noise_variance=noise_variance,
        input_mean=input_mean,
        input_scale=input_scale,
        noise_whitener=noise_whitener,
        log_det_input=log_det_input,
        flow_mode="local",
        n_flow_layers=topology.n_flow_layers,
        n_spline_bins=topology.n_spline_bins,
        spline_bound=topology.spline_bound,
        w1=np.asarray(flow_params["w1"], dtype=np.float64),
        b1=np.asarray(flow_params["b1"], dtype=np.float64),
        w2=np.asarray(flow_params["w2"], dtype=np.float64),
        b2=np.asarray(flow_params["b2"], dtype=np.float64),
        w_out=np.asarray(flow_params["w_out"], dtype=np.float64),
        b_out=np.asarray(flow_params["b_out"], dtype=np.float64),
        cond_nbr_indices=topology.cond_nbr_indices,
        cond_nbr_masks=topology.cond_nbr_masks,
        trans_indices=topology.trans_indices,
        trans_valid=topology.trans_valid,
        cond_geom=topology.cond_geom,
    )


# ---------------------------------------------------------------------------
# Layer 0 standardization and initial latent Gaussian mixture
# ---------------------------------------------------------------------------


def _standardize_host(
    marks: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    noise_whitener: np.ndarray | None,
) -> np.ndarray:
    """Apply Layer 0 centering, optional ZCA whitening, and per-feature scaling."""
    centered = marks - input_mean[None, :]
    if noise_whitener is not None:
        centered = centered @ noise_whitener
    return cast(np.ndarray, centered / input_scale[None, :])


def _fit_input_standardization(
    marks: np.ndarray,
    spike_weights: np.ndarray,
    *,
    mark_reg_covar: float,
    noise_covariance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, np.ndarray]:
    """Compute Layer 0 centering + optional ZCA whitening + scaling and variance floor.

    Uniform scalar unit changes (``m -> a*m + b`` with ``a > 0``, together with
    ``noise_covariance -> a^2 * noise_covariance`` or
    ``mark_reg_covar -> a^2 * mark_reg_covar``) leave the standardized
    coordinates ``u`` invariant and shift ``log_det_input`` by ``-D * log(a)``.
    With correlated ``noise_covariance``, anisotropic per-channel rescaling
    ``A`` does not commute with ``Sigma`` (``(A Sigma A)^{-1/2} = Q Sigma^{-1/2} A^{-1}``
    with an orthogonal rotation ``Q != I``), which changes the coordinate-aligned
    post-whitening scale and coupling layers.

    Returns ``(input_mean, input_scale, noise_whitener, log_det_input, min_latent_var)``.
    """
    total_w = float(np.sum(spike_weights))
    norm_w = spike_weights / total_w
    input_mean = norm_w @ marks  # (D,)
    diff = marks - input_mean[None, :]

    if noise_covariance is not None:
        whitener, log_det_w = _compute_zca_whitener(noise_covariance)
        # Chain per-channel data scaling after ZCA whitening so high-SNR spike
        # peaks land inside [-spline_bound, spline_bound] while background noise
        # remains diagonal with exact variance 1.0 / input_scale**2 per channel.
        y = diff @ whitener
        var_y = norm_w @ (y * y)
        scale_sq = var_y + 1.0
        input_scale = np.sqrt(scale_sq)
        log_det_input = log_det_w - float(np.sum(np.log(input_scale)))
        min_latent_var = np.maximum(1.0 / scale_sq, _MIN_LATENT_VARIANCE)
        return input_mean, input_scale, whitener, log_det_input, min_latent_var

    var = norm_w @ (diff * diff)
    # Scale matches mark_reference's sqrt(var + reg) so uniform scaling a*m+b
    # with reg -> a^2*reg leaves standardized u invariant identically.
    scale_sq = var + float(mark_reg_covar)
    input_scale = np.sqrt(scale_sq)
    log_det_input = -float(np.sum(np.log(input_scale)))
    min_latent_var = np.maximum(float(mark_reg_covar) / scale_sq, _MIN_LATENT_VARIANCE)
    return input_mean, input_scale, None, log_det_input, min_latent_var


def _initialize_latent_mixture(
    u_train: np.ndarray,
    spike_weights: np.ndarray,
    *,
    n_mark_components: int,
    covariance_rank: int | None,
    min_latent_var: np.ndarray,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize latent Gaussian mixture on standardized marks ``u_train``."""
    n_spikes, n_dims = u_train.shape
    total_w = float(np.sum(spike_weights))
    if n_spikes == 1 or total_w <= _MIN_COMPONENT_SPIKES:
        norm_w = spike_weights / total_w
        mean = (norm_w @ u_train)[None, :]
        noise_var = np.array(min_latent_var, dtype=np.float64)
        diff = u_train - mean
        cov = ((diff.T * norm_w[None, :]) @ diff)[None, :, :]
        basis, basis_var = _project_covariances_to_orthonormal_basis(
            cov, noise_var, rank=covariance_rank
        )
        return (
            np.array([1.0], dtype=np.float64),
            mean,
            basis,
            basis_var,
            noise_var,
        )

    reg_scalar = float(np.median(min_latent_var))
    init_gmm: MarkMixture = fit_mark_mixture(
        u_train,
        spike_weights,
        n_components=n_mark_components,
        reg_covar=reg_scalar,
        random_state=0 if random_state is None else int(random_state),
    )
    n_comp = init_gmm.n_components
    # Recover component covariances from MarkMixture's precision Cholesky:
    # precision = L @ L.T => cov = (L^{-1}).T @ L^{-1}.
    covariances = np.empty((n_comp, n_dims, n_dims), dtype=np.float64)
    eye = np.eye(n_dims, dtype=np.float64)
    for k in range(n_comp):
        inv_l = np.linalg.solve(init_gmm.precisions_cholesky[k], eye)
        cov_k = inv_l.T @ inv_l
        covariances[k] = 0.5 * (cov_k + cov_k.T)

    is_full = covariance_rank is None or covariance_rank >= n_dims
    if is_full:
        # Standardized eigenvalues are clamped at 1.0 in _project_covariances_to_orthonormal_basis,
        # enforcing Sigma >= diag(min_latent_var) without shifting modes above the floor.
        noise_var = np.array(min_latent_var, dtype=np.float64)
        basis, basis_var = _project_covariances_to_orthonormal_basis(
            covariances, noise_var, rank=None
        )
        return (
            np.array(init_gmm.weights, dtype=np.float64),
            np.array(init_gmm.means, dtype=np.float64),
            basis,
            basis_var,
            noise_var,
        )

    # Low-rank: initialize shared Psi from weighted component diagonal variance,
    # then refine once with monotone FA projection.
    diag_vars = np.diagonal(covariances, axis1=1, axis2=2)  # (K, D)
    init_psi = np.maximum(init_gmm.weights @ diag_vars * 0.5, min_latent_var)
    basis, basis_var = _project_covariances_to_orthonormal_basis(
        covariances, init_psi, rank=covariance_rank
    )
    return (
        np.array(init_gmm.weights, dtype=np.float64),
        np.array(init_gmm.means, dtype=np.float64),
        basis,
        basis_var,
        init_psi,
    )


# ---------------------------------------------------------------------------
# Blocked inner validation and alternating Flow-EM driver
# ---------------------------------------------------------------------------


def _contiguous_exposure_blocks(
    exposure_weights: np.ndarray, n_blocks: int
) -> np.ndarray:
    """Assign each position sample to one of ``n_blocks`` equal-exposure blocks."""
    cum = np.cumsum(exposure_weights)
    total = float(cum[-1]) if cum.size > 0 else 0.0
    if total <= 0.0 or n_blocks <= 1:
        return np.zeros(exposure_weights.shape[0], dtype=np.int32)
    edges = np.linspace(0.0, total, n_blocks + 1)[1:-1]
    return np.searchsorted(edges, cum, side="left").astype(np.int32)


@partial(jax.jit, static_argnames=("flow_mode", "n_bins", "bound"))
def _evaluate_flow_block_jax(
    u_block: jax.Array,
    flow_params: dict[str, jax.Array],
    cond_masks: jax.Array | None,
    cond_nbr_indices: jax.Array | None,
    cond_nbr_masks: jax.Array | None,
    trans_indices: jax.Array | None,
    trans_valid: jax.Array | None,
    cond_geom: jax.Array | None,
    *,
    flow_mode: str,
    n_bins: int,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    """Compiled static-shape forward flow evaluation for E-steps and validation."""
    return _apply_flow_forward_jax(
        u_block,
        flow_params,
        flow_mode=flow_mode,
        n_bins=n_bins,
        bound=bound,
        cond_masks=cond_masks,
        cond_nbr_indices=cond_nbr_indices,
        cond_nbr_masks=cond_nbr_masks,
        trans_indices=trans_indices,
        trans_valid=trans_valid,
        cond_geom=cond_geom,
    )


def _evaluate_flow_batched(
    u: np.ndarray,
    flow_params: dict[str, jax.Array],
    topology: _FlowTopology,
    *,
    cond_masks_j: jax.Array | None = None,
    nbr_idx_j: jax.Array | None = None,
    nbr_mask_j: jax.Array | None = None,
    trans_idx_j: jax.Array | None = None,
    trans_val_j: jax.Array | None = None,
    cond_geom_j: jax.Array | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``u -> (z, log_det_g)`` via compiled padded blocks."""
    n_spikes = u.shape[0]
    if n_spikes == 0 or topology.flow_mode == "identity":
        return u, np.zeros((n_spikes,), dtype=np.float64)

    if cond_masks_j is None and topology.cond_masks is not None:
        cond_masks_j = jnp.asarray(topology.cond_masks)
    if nbr_idx_j is None and topology.cond_nbr_indices is not None:
        nbr_idx_j = jnp.asarray(topology.cond_nbr_indices)
    if nbr_mask_j is None and topology.cond_nbr_masks is not None:
        nbr_mask_j = jnp.asarray(topology.cond_nbr_masks)
    if trans_idx_j is None and topology.trans_indices is not None:
        trans_idx_j = jnp.asarray(topology.trans_indices)
    if trans_val_j is None and topology.trans_valid is not None:
        trans_val_j = jnp.asarray(topology.trans_valid)
    if cond_geom_j is None and topology.cond_geom is not None:
        cond_geom_j = jnp.asarray(topology.cond_geom, dtype=jnp.float32)

    z_out = np.empty_like(u, dtype=np.float64)
    ld_out = np.empty((n_spikes,), dtype=np.float64)
    for start in range(0, n_spikes, _E_STEP_BLOCK):
        stop = min(start + _E_STEP_BLOCK, n_spikes)
        n_cur = stop - start
        n_pad = padded_spike_count(n_cur, _E_STEP_BLOCK)
        u_blk = jnp.asarray(pad_rows(u[start:stop], n_pad), dtype=jnp.float32)
        z_blk, ld_blk = _evaluate_flow_block_jax(
            u_blk,
            flow_params,
            cond_masks_j,
            nbr_idx_j,
            nbr_mask_j,
            trans_idx_j,
            trans_val_j,
            cond_geom_j,
            flow_mode=topology.flow_mode,
            n_bins=topology.n_spline_bins,
            bound=topology.spline_bound,
        )
        z_out[start:stop] = np.asarray(z_blk[:n_cur], dtype=np.float64)
        ld_out[start:stop] = np.asarray(ld_blk[:n_cur], dtype=np.float64)
    return z_out, ld_out


def _position_free_responsibilities_f64(
    z: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    basis: np.ndarray,
    basis_variances: np.ndarray,
    noise_variance: np.ndarray,
) -> np.ndarray:
    """Compute initial position-free responsibilities ``(N, K)`` in bounded blocks."""
    n_spikes = z.shape[0]
    n_comp = means.shape[0]
    resp = np.empty((n_spikes, n_comp), dtype=np.float64)
    log_w = np.log(np.maximum(weights, 1e-300))[None, :]
    for start in range(0, n_spikes, _E_STEP_BLOCK):
        stop = min(start + _E_STEP_BLOCK, n_spikes)
        log_gauss = _latent_woodbury_log_density_f64(
            z[start:stop], means, basis, basis_variances, noise_variance
        )
        lw = log_gauss + log_w
        resp[start:stop] = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
    return resp


def _e_step_latent_split(
    z: np.ndarray,
    log_det_total: np.ndarray,
    log_rates_left: np.ndarray,
    log_rates_right: np.ndarray,
    means: np.ndarray,
    basis: np.ndarray,
    basis_variances: np.ndarray,
    noise_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact split-position responsibilities and log intensities ``(N, K)``/``(N,)``."""
    n_spikes = z.shape[0]
    n_comp = means.shape[0]
    resp_l = np.empty((n_spikes, n_comp), dtype=np.float64)
    resp_r = np.empty((n_spikes, n_comp), dtype=np.float64)
    log_int_l = np.empty((n_spikes,), dtype=np.float64)
    log_int_r = np.empty((n_spikes,), dtype=np.float64)
    for start in range(0, n_spikes, _E_STEP_BLOCK):
        stop = min(start + _E_STEP_BLOCK, n_spikes)
        log_gauss = _latent_woodbury_log_density_f64(
            z[start:stop], means, basis, basis_variances, noise_variance
        )
        ld_blk = log_det_total[start:stop]
        joint_l = log_rates_left[start:stop] + log_gauss
        lse_l = logsumexp(joint_l, axis=1)
        resp_l[start:stop] = np.exp(joint_l - lse_l[:, None])
        log_int_l[start:stop] = ld_blk + lse_l

        joint_r = log_rates_right[start:stop] + log_gauss
        lse_r = logsumexp(joint_r, axis=1)
        resp_r[start:stop] = np.exp(joint_r - lse_r[:, None])
        log_int_r[start:stop] = ld_blk + lse_r
    return resp_l, resp_r, log_int_l, log_int_r


def _partition_fold_shares(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    left_share: np.ndarray,
    right_share: np.ndarray,
    allowed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each spike to a single owning fold sample and fold boundary shares.

    Returns ``(owned_mask, fold_left_share, fold_right_share)`` over the owned
    subset so no boundary spike can enter both training and validation and
    total spike weight is conserved on allowed samples.
    """
    safe_l = np.maximum(left_pos, 0)
    safe_r = np.maximum(right_pos, 0)
    l_valid = (left_pos >= 0) & (left_share > 0.0)
    r_valid = (right_pos >= 0) & (right_share > 0.0)
    owner = np.where(
        (~r_valid) | (l_valid & (left_share >= right_share)), safe_l, safe_r
    )
    owned = allowed_mask[owner]
    l_in = allowed_mask[safe_l[owned]] & (left_pos[owned] >= 0)
    r_in = allowed_mask[safe_r[owned]] & (right_pos[owned] >= 0)
    ls = left_share[owned]
    rs = right_share[owned]
    tot = ls + rs
    fold_ls = np.where(l_in & r_in, ls, np.where(l_in, tot, 0.0))
    fold_rs = np.where(l_in & r_in, rs, np.where(r_in, tot, 0.0))
    return owned, fold_ls, fold_rs


def _sample_flow_batches(
    u: np.ndarray,
    spike_weights: np.ndarray,
    log_comp_weights: np.ndarray,
    *,
    n_steps: int,
    batch_size: int,
    k_pad: int,
    rng: np.random.Generator,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build static-shape ``(n_steps, b_pad, ...)`` batches for optax scan.

    When mini-batching (``n_spikes > batch_size``), draws indices proportionally
    to ``spike_weights`` and assigns uniform unit batch weights so the
    per-batch mean in ``_loss`` is an exact unbiased estimator of the
    weight-normalized objective at every ``batch_size >= 1``.
    """
    n_spikes, n_dims = u.shape
    n_comp = log_comp_weights.shape[1]
    use_full = n_spikes <= batch_size
    b_pad = _power_of_two(n_spikes) if use_full else int(batch_size)

    lw_padded = np.full((n_spikes, k_pad), -1e30, dtype=np.float32)
    lw_padded[:, :n_comp] = log_comp_weights.astype(np.float32)

    if use_full:
        u_one = np.zeros((b_pad, n_dims), dtype=np.float32)
        u_one[:n_spikes] = u.astype(np.float32)
        w_one = np.zeros((b_pad,), dtype=np.float32)
        w_one[:n_spikes] = spike_weights.astype(np.float32)
        lw_one = np.full((b_pad, k_pad), -1e30, dtype=np.float32)
        lw_one[:n_spikes] = lw_padded
        return (
            jnp.broadcast_to(jnp.asarray(u_one)[None, :, :], (n_steps, b_pad, n_dims)),
            jnp.broadcast_to(jnp.asarray(w_one)[None, :], (n_steps, b_pad)),
            jnp.broadcast_to(jnp.asarray(lw_one)[None, :, :], (n_steps, b_pad, k_pad)),
        )

    w_pos = np.maximum(np.asarray(spike_weights, dtype=np.float64), 0.0)
    nz = np.flatnonzero(w_pos > 0.0)
    if nz.size == 0:
        return (
            jnp.zeros((n_steps, b_pad, n_dims), dtype=jnp.float32),
            jnp.zeros((n_steps, b_pad), dtype=jnp.float32),
            jnp.full((n_steps, b_pad, k_pad), -1e30, dtype=jnp.float32),
        )
    cdf = np.cumsum(w_pos[nz])
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    pick = np.searchsorted(cdf, rng.random((n_steps, b_pad)), side="right")
    idx = nz[np.minimum(pick, nz.size - 1)]
    u_f32 = u.astype(np.float32)
    return (
        jnp.asarray(u_f32[idx]),
        jnp.ones((n_steps, b_pad), dtype=jnp.float32),
        jnp.asarray(lw_padded[idx]),
    )


def _pad_latent_params_for_flow(
    means: np.ndarray,
    basis: np.ndarray,
    basis_variances: np.ndarray,
    noise_variance: np.ndarray,
    k_pad: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pad latent mixture arrays to static component count ``k_pad``."""
    n_comp, n_dims = means.shape
    rank = basis.shape[2]
    n_pad = k_pad - n_comp
    means_p = np.concatenate([means, np.zeros((n_pad, n_dims))], axis=0)
    basis_p = np.concatenate([basis, np.zeros((n_pad, n_dims, rank))], axis=0)
    var_p = np.concatenate([basis_variances, np.ones((n_pad, rank))], axis=0)
    is_comp = np.arange(k_pad) < n_comp
    return (
        jnp.asarray(means_p, dtype=jnp.float32),
        jnp.asarray(basis_p, dtype=jnp.float32),
        jnp.asarray(var_p, dtype=jnp.float32),
        jnp.asarray(noise_variance, dtype=jnp.float32),
        jnp.asarray(is_comp),
    )


class _ElectrodeFoldState:
    """Mutable state of one electrode on one training fold during Flow-EM."""

    def __init__(
        self,
        *,
        marks: np.ndarray,
        spike_weights: np.ndarray,
        left_pos: np.ndarray,
        right_pos: np.ndarray,
        left_share: np.ndarray,
        right_share: np.ndarray,
        left_cells: np.ndarray,
        right_cells: np.ndarray,
        train_prepared: PreparedRateMapBasis,
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        noise_whitener: np.ndarray | None,
        log_det_input: float,
        min_latent_var: np.ndarray,
        topology: _FlowTopology,
        flow_params: dict[str, jax.Array],
        opt_state: Any,
        weights: np.ndarray,
        means: np.ndarray,
        basis: np.ndarray,
        basis_variances: np.ndarray,
        noise_variance: np.ndarray,
        coefficients: np.ndarray,
        val_marks: np.ndarray | None = None,
        val_left_share: np.ndarray | None = None,
        val_right_share: np.ndarray | None = None,
        val_left_cells: np.ndarray | None = None,
        val_right_cells: np.ndarray | None = None,
        val_reference_log_bg: np.ndarray | None = None,
    ) -> None:
        self.marks = marks
        self.spike_weights = spike_weights
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.left_share = left_share
        self.right_share = right_share
        self.left_cells = left_cells
        self.right_cells = right_cells
        self.train_prepared = train_prepared
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.noise_whitener = noise_whitener
        self.log_det_input = log_det_input
        self.min_latent_var = min_latent_var
        self.topology = topology
        self.flow_params = flow_params
        self.opt_state = opt_state
        self.weights = weights
        self.means = means
        self.basis = basis
        self.basis_variances = basis_variances
        self.noise_variance = noise_variance
        self.coefficients = coefficients  # shape (K, n_basis)
        self.val_marks = val_marks
        self.val_left_share = val_left_share
        self.val_right_share = val_right_share
        self.val_left_cells = val_left_cells
        self.val_right_cells = val_right_cells
        self.val_reference_log_bg = val_reference_log_bg

        self.u_train = _standardize_host(marks, input_mean, input_scale, noise_whitener)
        if val_marks is not None and val_marks.shape[0] > 0:
            self.u_val: np.ndarray | None = _standardize_host(
                val_marks, input_mean, input_scale, noise_whitener
            )
        else:
            self.u_val = None

    def to_mixture(self) -> FlowMarkMixture:
        """Freeze current state into a ``FlowMarkMixture``."""
        return _pack_flow_mixture(
            weights=self.weights,
            means=self.means,
            basis=self.basis,
            basis_variances=self.basis_variances,
            noise_variance=self.noise_variance,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            noise_whitener=self.noise_whitener,
            log_det_input=self.log_det_input,
            topology=self.topology,
            flow_params=self.flow_params,
        )


def _fold_mark_reference(
    marks: np.ndarray, weights: np.ndarray, mark_reg_covar: float
) -> MarkReference:
    """Fit a training-fold Student-t mark reference for validation background."""
    norm_w = weights / float(np.sum(weights))
    loc = norm_w @ marks
    var = norm_w @ ((marks - loc[None, :]) ** 2)
    return MarkReference(loc, np.sqrt(var + float(mark_reg_covar)))


def _accumulate_cell_counts(
    sample_cells: np.ndarray,
    n_cells: int,
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    left_share: np.ndarray,
    right_share: np.ndarray,
    resp_left: np.ndarray,
    resp_right: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate split-position soft spike counts onto cells, shape ``(K, n_cells)``."""
    resp_r = resp_left if resp_right is None else resp_right
    n_comp = resp_left.shape[1]
    counts = np.zeros((n_comp, n_cells), dtype=np.float64)
    if resp_left.shape[0] == 0:
        return counts
    l_ok = (left_pos >= 0) & (left_share > 0.0)
    r_ok = (right_pos >= 0) & (right_share > 0.0)
    left_cells = sample_cells[left_pos[l_ok]]
    right_cells = sample_cells[right_pos[r_ok]]
    ls = left_share[l_ok]
    rs = right_share[r_ok]
    for k in range(n_comp):
        c_k = np.zeros(n_cells, dtype=np.float64)
        if left_cells.size:
            c_k += np.bincount(
                left_cells, weights=ls * resp_left[l_ok, k], minlength=n_cells
            )
        if right_cells.size:
            c_k += np.bincount(
                right_cells, weights=rs * resp_r[r_ok, k], minlength=n_cells
            )
        counts[k] = c_k
    return counts


def _fit_single_electrode_flow_em(
    *,
    electrode: int,
    marks_eff: np.ndarray,
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    left_share: np.ndarray,
    right_share: np.ndarray,
    encoding_positions: np.ndarray,
    full_prepared: PreparedRateMapBasis,
    fold_specs: list[tuple[np.ndarray, np.ndarray, PreparedRateMapBasis, np.ndarray]],
    contact_positions: np.ndarray | None,
    noise_covariance: np.ndarray | None,
    n_mark_components: int,
    covariance_rank: int | None,
    mark_reg_covar: float,
    n_flow_layers: int,
    n_spline_bins: int,
    spline_bound: float,
    flow_hidden_units: int,
    neighborhood_radius: float | None,
    max_local_channels: int | None,
    flow_objective: Literal["joint", "marginal"],
    n_em_iterations: int,
    flow_steps_per_em: int,
    flow_learning_rate: float,
    flow_weight_decay: float,
    flow_batch_size: int,
    em_glm_steps: int,
    em_patience: int,
    random_state: int | None,
    optax_mod: Any,
    flow_kernel_cache: dict[tuple[Any, ...], Any],
) -> tuple[FlowMarkMixture, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one active electrode's ``FlowMarkMixture``, GLM coefficients, and log rates."""
    n_features = marks_eff.shape[1]
    sample_cells = full_prepared.sample_cells
    n_cells = full_prepared.n_position_cells
    cell_design_j = full_prepared.emission_design_matrix

    topology = _build_flow_topology(
        n_features,
        contact_positions,
        n_flow_layers=n_flow_layers,
        n_spline_bins=n_spline_bins,
        spline_bound=spline_bound,
        neighborhood_radius=neighborhood_radius,
        max_local_channels=max_local_channels,
    )

    optimizer = optax_mod.chain(
        optax_mod.clip_by_global_norm(1.0),
        optax_mod.adamw(
            learning_rate=flow_learning_rate,
            weight_decay=flow_weight_decay,
        ),
    )

    kernel_key = (
        topology.flow_mode,
        topology.n_spline_bins,
        topology.spline_bound,
        flow_learning_rate,
        flow_weight_decay,
    )
    if kernel_key not in flow_kernel_cache and topology.flow_mode != "identity":
        flow_kernel_cache[kernel_key] = _make_flow_step_kernel(
            optimizer,
            flow_mode=topology.flow_mode,
            n_bins=topology.n_spline_bins,
            bound=topology.spline_bound,
        )
    flow_step_fn = flow_kernel_cache.get(kernel_key)

    cond_masks_j = (
        None if topology.cond_masks is None else jnp.asarray(topology.cond_masks)
    )
    nbr_idx_j = (
        None
        if topology.cond_nbr_indices is None
        else jnp.asarray(topology.cond_nbr_indices)
    )
    nbr_mask_j = (
        None
        if topology.cond_nbr_masks is None
        else jnp.asarray(topology.cond_nbr_masks)
    )
    trans_idx_j = (
        None if topology.trans_indices is None else jnp.asarray(topology.trans_indices)
    )
    trans_val_j = (
        None if topology.trans_valid is None else jnp.asarray(topology.trans_valid)
    )
    cond_geom_j = (
        None
        if topology.cond_geom is None
        else jnp.asarray(topology.cond_geom, dtype=jnp.float32)
    )

    base_seed = 0 if random_state is None else int(random_state)

    def _build_fold_state(
        fold_idx: int,
        train_mask_sup: np.ndarray,
        val_mask_sup: np.ndarray | None,
        train_prepared: PreparedRateMapBasis,
    ) -> _ElectrodeFoldState | None:
        sp_train, fold_ls, fold_rs = _partition_fold_shares(
            left_pos, right_pos, left_share, right_share, train_mask_sup
        )
        fold_w = fold_ls + fold_rs
        if not np.any(sp_train) or float(np.sum(train_prepared.cell_exposure)) <= 0.0:
            return None

        fold_marks = marks_eff[sp_train]
        fold_l_pos = left_pos[sp_train]
        fold_r_pos = right_pos[sp_train]
        fold_l_cells = sample_cells[np.maximum(fold_l_pos, 0)]
        fold_r_cells = sample_cells[np.maximum(fold_r_pos, 0)]

        in_mean, in_scale, whitener, ld_in, min_lvar = _fit_input_standardization(
            fold_marks,
            fold_w,
            mark_reg_covar=mark_reg_covar,
            noise_covariance=noise_covariance,
        )
        u_tr = _standardize_host(fold_marks, in_mean, in_scale, whitener)

        rng_init = np.random.default_rng(base_seed * 1009 + electrode * 37 + fold_idx)
        weights_0, means_0, basis_0, bvar_0, nvar_0 = _initialize_latent_mixture(
            u_tr,
            fold_w,
            n_mark_components=n_mark_components,
            covariance_rank=covariance_rank,
            min_latent_var=min_lvar,
            random_state=int(rng_init.integers(0, 2**31 - 1)),
        )
        flow_params_0 = _init_identity_flow_params(
            topology, n_features, flow_hidden_units, rng_init
        )
        opt_state_0 = (
            optimizer.init(flow_params_0) if topology.flow_mode != "identity" else None
        )

        # Initial position-free responsibilities at identity flow (iteration 0),
        # evaluated in bounded ``_E_STEP_BLOCK`` chunks to avoid full (N, K, D) RAM spikes.
        resp_0 = _position_free_responsibilities_f64(
            u_tr, weights_0, means_0, basis_0, bvar_0, nvar_0
        )

        cell_counts_0 = _accumulate_cell_counts(
            sample_cells,
            n_cells,
            fold_l_pos,
            fold_r_pos,
            fold_ls,
            fold_rs,
            resp_0,
            resp_0,
        )
        coeffs_0_j, _ = fit_prepared_rate_maps(
            train_prepared,
            cell_counts_0,
            warn_on_nonconvergence=False,
        )
        coeffs_0 = np.asarray(coeffs_0_j, dtype=np.float64)

        val_m = val_ls = val_rs = val_lc = val_rc = val_ref_bg = None
        if val_mask_sup is not None:
            sp_val, v_ls, v_rs = _partition_fold_shares(
                left_pos, right_pos, left_share, right_share, val_mask_sup
            )
            if np.any(sp_val):
                val_m = marks_eff[sp_val]
                val_ls = v_ls
                val_rs = v_rs
                val_lc = sample_cells[np.maximum(left_pos[sp_val], 0)]
                val_rc = sample_cells[np.maximum(right_pos[sp_val], 0)]
                val_ref = _fold_mark_reference(fold_marks, fold_w, mark_reg_covar)
                val_ref_bg = LOG_RATE_EPS_HZ + val_ref.log_density(val_m)

        return _ElectrodeFoldState(
            marks=fold_marks,
            spike_weights=fold_w,
            left_pos=fold_l_pos,
            right_pos=fold_r_pos,
            left_share=fold_ls,
            right_share=fold_rs,
            left_cells=fold_l_cells,
            right_cells=fold_r_cells,
            train_prepared=train_prepared,
            input_mean=in_mean,
            input_scale=in_scale,
            noise_whitener=whitener,
            log_det_input=ld_in,
            min_latent_var=min_lvar,
            topology=topology,
            flow_params=flow_params_0,
            opt_state=opt_state_0,
            weights=weights_0,
            means=means_0,
            basis=basis_0,
            basis_variances=bvar_0,
            noise_variance=nvar_0,
            coefficients=coeffs_0,
            val_marks=val_m,
            val_left_share=val_ls,
            val_right_share=val_rs,
            val_left_cells=val_lc,
            val_right_cells=val_rc,
            val_reference_log_bg=val_ref_bg,
        )

    def _eval_flow(
        u_arr: np.ndarray, params: dict[str, jax.Array]
    ) -> tuple[np.ndarray, np.ndarray]:
        return _evaluate_flow_batched(
            u_arr,
            params,
            topology,
            cond_masks_j=cond_masks_j,
            nbr_idx_j=nbr_idx_j,
            nbr_mask_j=nbr_mask_j,
            trans_idx_j=trans_idx_j,
            trans_val_j=trans_val_j,
            cond_geom_j=cond_geom_j,
        )

    def _score_fold_val(
        state: _ElectrodeFoldState, val_cell_exposure: np.ndarray
    ) -> float:
        rates_cells = np.asarray(
            glm_rates(cell_design_j, jnp.asarray(state.coefficients)), dtype=np.float64
        )
        gpi_cells = np.sum(rates_cells, axis=1) + RATE_EPS_HZ
        expected_count = float(np.dot(gpi_cells, val_cell_exposure))

        if (
            state.u_val is None
            or state.val_left_share is None
            or state.val_right_share is None
            or state.val_left_cells is None
            or state.val_right_cells is None
        ):
            return -expected_count

        z_val, ld_flow = _eval_flow(state.u_val, state.flow_params)
        log_det_tot = state.log_det_input + ld_flow
        cell_log_rates = np.log(rates_cells)
        _, _, log_int_l, log_int_r = _e_step_latent_split(
            z_val,
            log_det_tot,
            cell_log_rates[state.val_left_cells],
            cell_log_rates[state.val_right_cells],
            state.means,
            state.basis,
            state.basis_variances,
            state.noise_variance,
        )
        if state.val_reference_log_bg is not None:
            log_int_l = np.logaddexp(log_int_l, state.val_reference_log_bg)
            log_int_r = np.logaddexp(log_int_r, state.val_reference_log_bg)
        return (
            float(
                np.dot(state.val_left_share, log_int_l)
                + np.dot(state.val_right_share, log_int_r)
            )
            - expected_count
        )

    def _score_fold_train(state: _ElectrodeFoldState) -> float:
        rates_cells = np.asarray(
            glm_rates(cell_design_j, jnp.asarray(state.coefficients)), dtype=np.float64
        )
        expected_count = float(
            np.dot(np.sum(rates_cells, axis=1), state.train_prepared.cell_exposure)
        )
        z_tr, ld_flow = _eval_flow(state.u_train, state.flow_params)
        log_det_tot = state.log_det_input + ld_flow
        cell_log_rates = np.log(rates_cells)
        _, _, log_int_l, log_int_r = _e_step_latent_split(
            z_tr,
            log_det_tot,
            cell_log_rates[state.left_cells],
            cell_log_rates[state.right_cells],
            state.means,
            state.basis,
            state.basis_variances,
            state.noise_variance,
        )
        l2_term = state.train_prepared.l2_penalty * float(
            np.sum(state.coefficients[:, 1:] ** 2)
        )
        return (
            float(
                np.dot(state.left_share, log_int_l)
                + np.dot(state.right_share, log_int_r)
            )
            - expected_count
            - l2_term
        )

    def _run_one_em_iteration(
        state: _ElectrodeFoldState,
        step_rng: np.random.Generator,
    ) -> None:
        n_comp = state.means.shape[0]
        k_pad = _power_of_two(n_comp)
        cell_log_rates = np.log(
            np.asarray(
                glm_rates(cell_design_j, jnp.asarray(state.coefficients)),
                dtype=np.float64,
            )
        )

        # 1. Flow optimization step (skipped if identity mode or < 2 spikes).
        if (
            flow_step_fn is not None
            and flow_steps_per_em > 0
            and state.u_train.shape[0] >= 2
        ):
            if flow_objective == "joint":
                same_cell = state.left_cells == state.right_cells
                has_l = (~same_cell) & (state.left_share > 0.0)
                has_r = (~same_cell) & (state.right_share > 0.0)
                u_flow = np.concatenate(
                    [
                        state.u_train[same_cell],
                        state.u_train[has_l],
                        state.u_train[has_r],
                    ],
                    axis=0,
                )
                w_flow = np.concatenate(
                    [
                        state.spike_weights[same_cell],
                        state.left_share[has_l],
                        state.right_share[has_r],
                    ]
                )
                log_comp_w = np.concatenate(
                    [
                        cell_log_rates[state.left_cells[same_cell]],
                        cell_log_rates[state.left_cells[has_l]],
                        cell_log_rates[state.right_cells[has_r]],
                    ],
                    axis=0,
                )
            else:
                u_flow = state.u_train
                w_flow = state.spike_weights
                log_pi = np.log(np.maximum(state.weights, 1e-300))
                log_comp_w = np.broadcast_to(
                    log_pi[None, :], (state.u_train.shape[0], n_comp)
                )
            u_b, w_b, lw_b = _sample_flow_batches(
                u_flow,
                w_flow,
                log_comp_w,
                n_steps=flow_steps_per_em,
                batch_size=flow_batch_size,
                k_pad=k_pad,
                rng=step_rng,
            )
            m_p, b_p, v_p, nv_p, is_c = _pad_latent_params_for_flow(
                state.means,
                state.basis,
                state.basis_variances,
                state.noise_variance,
                k_pad,
            )
            state.flow_params, state.opt_state = flow_step_fn(
                state.flow_params,
                state.opt_state,
                u_b,
                w_b,
                lw_b,
                m_p,
                b_p,
                v_p,
                nv_p,
                is_c,
                cond_masks_j,
                nbr_idx_j,
                nbr_mask_j,
                trans_idx_j,
                trans_val_j,
                cond_geom_j,
            )

        # 2. Exact split-position E-step in updated warped coordinates z = g_theta(u).
        z_tr, ld_flow = _eval_flow(state.u_train, state.flow_params)
        log_det_tot = state.log_det_input + ld_flow
        resp_l, resp_r, _, _ = _e_step_latent_split(
            z_tr,
            log_det_tot,
            cell_log_rates[state.left_cells],
            cell_log_rates[state.right_cells],
            state.means,
            state.basis,
            state.basis_variances,
            state.noise_variance,
        )

        # 3. Monotone FA-EM latent mixture M-step using exact combined shares.
        gamma = state.left_share[:, None] * resp_l + state.right_share[:, None] * resp_r
        resp_mark = gamma / np.maximum(state.spike_weights[:, None], 1e-300)
        (
            state.weights,
            state.means,
            state.basis,
            state.basis_variances,
            state.noise_variance,
        ) = _latent_mark_m_step(
            z_tr,
            resp_mark,
            state.spike_weights,
            state.means,
            state.basis,
            state.basis_variances,
            state.noise_variance,
            covariance_rank=covariance_rank,
            min_noise_variance=state.min_latent_var,
        )

        # 4. Warm-started damped Newton spatial rate-map M-step.
        cell_counts = _accumulate_cell_counts(
            sample_cells,
            n_cells,
            state.left_pos,
            state.right_pos,
            state.left_share,
            state.right_share,
            resp_l,
            resp_r,
        )
        coeffs_j, _ = fit_prepared_rate_maps(
            state.train_prepared,
            cell_counts,
            initial_coefficients=state.coefficients,
            max_steps=em_glm_steps,
            warn_on_nonconvergence=False,
        )
        state.coefficients = np.asarray(coeffs_j, dtype=np.float64)

    # Determine iteration count via blocked inner validation (when requested).
    val_trace: list[float] = []
    if n_em_iterations > 0 and fold_specs and marks_eff.shape[0] >= 4:
        fold_states: list[tuple[_ElectrodeFoldState, np.ndarray]] = []
        for f_idx, (tr_mask, val_mask, tr_prep, val_cell_exp) in enumerate(fold_specs):
            st = _build_fold_state(f_idx + 1, tr_mask, val_mask, tr_prep)
            if st is not None:
                fold_states.append((st, val_cell_exp))

        if fold_states:
            best_iter = 0
            best_val = sum(_score_fold_val(st, ve) for st, ve in fold_states)
            val_trace.append(best_val)
            stale = 0
            for it in range(1, n_em_iterations + 1):
                for f_idx, (st, _) in enumerate(fold_states):
                    step_rng = np.random.default_rng(
                        base_seed * 10007 + electrode * 97 + (f_idx + 1) * 1009 + it
                    )
                    _run_one_em_iteration(st, step_rng)
                cur_val = sum(_score_fold_val(st, ve) for st, ve in fold_states)
                val_trace.append(cur_val)
                if cur_val > best_val + 1e-9:
                    best_val = cur_val
                    best_iter = it
                    stale = 0
                else:
                    stale += 1
                    if stale >= em_patience:
                        break
            chosen_iterations = best_iter
        else:
            chosen_iterations = 0
    else:
        chosen_iterations = int(n_em_iterations)

    # Final fit on full training data for chosen_iterations.
    all_sup_mask = np.ones(len(encoding_positions), dtype=bool)
    full_state = _build_fold_state(0, all_sup_mask, None, full_prepared)
    assert full_state is not None
    train_trace: list[float] = [_score_fold_train(full_state)]

    for it in range(1, chosen_iterations + 1):
        step_rng = np.random.default_rng(base_seed * 10007 + electrode * 97 + it)
        _run_one_em_iteration(full_state, step_rng)
        train_trace.append(_score_fold_train(full_state))

    # Polish final GLM coefficients to convergence after flow/latent updates.
    if chosen_iterations > 0:
        z_final, ld_flow = _eval_flow(full_state.u_train, full_state.flow_params)
        final_cell_lr = np.log(
            np.asarray(
                glm_rates(cell_design_j, jnp.asarray(full_state.coefficients)),
                dtype=np.float64,
            )
        )
        resp_l_final, resp_r_final, _, _ = _e_step_latent_split(
            z_final,
            full_state.log_det_input + ld_flow,
            final_cell_lr[full_state.left_cells],
            final_cell_lr[full_state.right_cells],
            full_state.means,
            full_state.basis,
            full_state.basis_variances,
            full_state.noise_variance,
        )
        cell_counts_final = _accumulate_cell_counts(
            sample_cells,
            n_cells,
            full_state.left_pos,
            full_state.right_pos,
            full_state.left_share,
            full_state.right_share,
            resp_l_final,
            resp_r_final,
        )
        coeffs_j, interior_rates_j = fit_prepared_rate_maps(
            full_prepared,
            cell_counts_final,
            initial_coefficients=full_state.coefficients,
            warn_on_nonconvergence=False,
        )
        full_state.coefficients = np.asarray(coeffs_j, dtype=np.float64)
        interior_rates = np.asarray(interior_rates_j, dtype=np.float64)
    else:
        interior_rates = np.asarray(
            glm_rates(
                full_prepared.emission_predict_matrix,
                jnp.asarray(full_state.coefficients),
            ),
            dtype=np.float64,
        )

    diagnostics = {
        "n_iterations": int(chosen_iterations),
        "train_log_likelihood": train_trace,
        "validation_log_likelihood": val_trace,
    }
    return full_state.to_mixture(), full_state.coefficients, interior_rates, diagnostics


# ---------------------------------------------------------------------------
# Encoding model TypedDict & Public Fit Function
# ---------------------------------------------------------------------------


class ClusterlessFlowMixtureEncodingModel(ClusterlessEncodingModel):
    """Encoding model returned by ``fit_clusterless_flow_mixture_encoding_model``."""

    emission_design_info: TrackGraphBasis | TensorSplineBasis
    mark_mixtures: list[FlowMarkMixture | None]
    rate_coefficients: list[np.ndarray | jax.Array | None]
    log_rate_maps: list[np.ndarray | None]
    summed_ground_process_intensity: np.ndarray
    spike_block_size: int
    em_diagnostics: dict[str, Any]


def fit_clusterless_flow_mixture_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    n_mark_components: int = 32,
    covariance_rank: int | None = DEFAULT_COVARIANCE_RANK,
    mark_reg_covar: float = DEFAULT_GMM_REG_COVAR,
    contact_positions: Sequence[ArrayLike | None] | None = None,
    noise_covariances: Sequence[ArrayLike | None] | None = None,
    neighborhood_radius: float | None = None,
    max_local_channels: int | None = _DEFAULT_MAX_LOCAL_CHANNELS,
    n_flow_layers: int = DEFAULT_N_FLOW_LAYERS,
    n_spline_bins: int = DEFAULT_N_SPLINE_BINS,
    spline_bound: float = DEFAULT_SPLINE_BOUND,
    flow_hidden_units: int = DEFAULT_FLOW_HIDDEN_UNITS,
    flow_objective: Literal["joint", "marginal"] = "joint",
    n_em_iterations: int = 30,
    flow_steps_per_em: int = 20,
    flow_learning_rate: float = 1e-3,
    flow_weight_decay: float = 1e-4,
    flow_batch_size: int = DEFAULT_FLOW_BATCH_SIZE,
    em_glm_steps: int = 2,
    em_validation_blocks: int = 4,
    em_patience: int = 5,
    emission_knot_spacing: float = np.sqrt(12.5) * 2,
    log_rate_prior_std: float = GLM_LOG_RATE_PRIOR_STD,
    spike_block_size: int = 1000,
    disable_progress_bar: bool = False,
    random_state: int | None = 0,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> ClusterlessFlowMixtureEncodingModel:
    """Fit a flow-warped low-rank-plus-diagonal mark-mixture encoding model.

    Parameters
    ----------
    position_time : array_like, shape (n_time,)
        Timestamps of position samples in seconds.
    position : array_like, shape (n_time,) or (n_time, n_position_dims)
        Animal position at ``position_time``.
    spike_times : sequence of array_like, length n_electrodes
        One 1-D array of spike times (seconds) per electrode.
    spike_waveform_features : sequence of array_like, length n_electrodes
        One 2-D array ``(n_spikes, n_features)`` of waveform features per electrode.
    environment : Environment
        Fitted spatial environment defining the place grid (and track graph).
    weights : array_like, shape (n_time,), optional
        Non-negative encoding sample weights.
    n_mark_components : int, optional
        Upper bound on latent Gaussian components per electrode, by default 32.
    covariance_rank : int or None, optional
        Rank of the low-rank factor-analysis covariance in latent space
        (``None`` or ``>= D`` uses exact full covariance), by default 4.
    mark_reg_covar : float, optional
        Covariance regularization floor in squared mark units, by default ``1.0``.
    contact_positions : sequence of array_like or None, optional
        Optional per-electrode contact coordinates ``(n_channels, n_probe_dims)``.
        When provided for an electrode, the flow uses physically neighbor-masked
        spatial coupling layers instead of dense MLP coupling.
    noise_covariances : sequence of array_like or None, optional
        Optional per-electrode SPD channel noise covariances ``(n_channels, n_channels)``,
        applied via symmetric ZCA whitening followed by per-channel scaling in
        Layer 0. Preserves exact equivariance under uniform scalar unit changes
        (``m -> a*m + b`` with ``a > 0`` and ``noise_covariances -> a^2 * noise_covariances``);
        anisotropic per-channel rescaling with correlated noise rotates the ZCA
        basis.
    neighborhood_radius : float or None, optional
        Physical radius for neighbor-masked coupling layers when
        ``contact_positions`` is given. None auto-infers ``3 * median_pitch``.
    max_local_channels : int or None, optional
        Maximum conditioning neighbors per contact in local coupling mode, by
        default 16.
    n_flow_layers : int, optional
        Number of Rational-Quadratic Neural Spline (RQS) coupling layers (0
        disables warping), by default 4.
    n_spline_bins : int, optional
        Number of RQS segments on ``[-spline_bound, spline_bound]``, by default 8.
    spline_bound : float, optional
        Half-width ``B > 0`` of the RQS nonlinear interval (identity outside),
        by default 5.0.
    flow_hidden_units : int, optional
        Hidden width of the dense coupling conditioner MLP (capped at 32 in
        weight-shared local coupling mode), by default 64.
    flow_objective : {"joint", "marginal"}, optional
        ``"joint"`` (default) trains the flow on the marked Poisson spatial-mark
        log-likelihood; ``"marginal"`` trains it on position-free mark density.
    n_em_iterations : int, optional
        Maximum alternating Flow-EM outer iterations (0 = identity flow), by
        default 30.
    flow_steps_per_em : int, optional
        Number of ``optax`` AdamW steps per outer EM iteration, by default 20.
    flow_learning_rate : float, optional
        AdamW learning rate for flow parameters, by default ``1e-3``.
    flow_weight_decay : float, optional
        AdamW decoupled weight decay for flow parameters, by default ``1e-4``.
    flow_batch_size : int, optional
        Mini-batch size for flow gradient steps (full-batch when
        ``n_spikes <= flow_batch_size``), by default 512.
    em_glm_steps : int, optional
        Warm-started damped Newton steps for the GLM rate-map M-step per outer
        iteration, by default 2.
    em_validation_blocks : int, optional
        Number of contiguous equal-exposure inner validation blocks for early
        stopping (0 or 1 disables validation and runs all ``n_em_iterations``),
        by default 4.
    em_patience : int, optional
        Consecutive non-improving validation iterations before early stopping,
        by default 5.
    emission_knot_spacing : float, optional
        Spline knot spacing in environment coordinate units, by default
        ``2 * sqrt(12.5)``.
    log_rate_prior_std : float, optional
        Prior standard deviation on GLM spline coefficients, by default
        ``GLM_LOG_RATE_PRIOR_STD``.
    spike_block_size : int, optional
        Prediction spike block size (execution setting), by default 1000.
    disable_progress_bar : bool, optional
        Whether to hide progress bars during prediction, by default False.
    random_state : int or None, optional
        Random seed for deterministic initialization and mini-batching, by
        default 0.
    encoding_time_range : array_like, shape (2,), optional
        Explicit acquisition time bounds in seconds.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Continuous tracking intervals in seconds.
    _encoding_support : EncodingSupport, optional
        Internal precomputed encoding support passed by detectors.

    Returns
    -------
    encoding_model : ClusterlessFlowMixtureEncodingModel
        Fitted encoding model dictionary.
    """
    optax_mod = _require_optax()

    validate_positive_int("n_mark_components", n_mark_components)
    validate_positive_int("covariance_rank", covariance_rank, allow_none=True)
    mark_reg_covar = _validate_positive_float("mark_reg_covar", mark_reg_covar)
    n_flow_layers = _validate_nonnegative_int("n_flow_layers", n_flow_layers)
    validate_positive_int("n_spline_bins", n_spline_bins)
    if n_spline_bins < 2:
        msg = "n_spline_bins must be at least 2"
        raise ValidationError(msg, expected="int >= 2", got=repr(n_spline_bins))
    spline_bound = _validate_positive_float("spline_bound", spline_bound)
    validate_positive_int("flow_hidden_units", flow_hidden_units)
    validate_positive_int("max_local_channels", max_local_channels, allow_none=True)
    if neighborhood_radius is not None:
        neighborhood_radius = _validate_positive_float(
            "neighborhood_radius", neighborhood_radius
        )
    if flow_objective not in FLOW_OBJECTIVES:
        msg = f"flow_objective must be one of {FLOW_OBJECTIVES}"
        raise ValidationError(msg, got=repr(flow_objective))
    n_em_iterations = _validate_nonnegative_int("n_em_iterations", n_em_iterations)
    flow_steps_per_em = _validate_nonnegative_int(
        "flow_steps_per_em", flow_steps_per_em
    )
    flow_learning_rate = _validate_positive_float(
        "flow_learning_rate", flow_learning_rate
    )
    flow_weight_decay = _validate_nonnegative_float(
        "flow_weight_decay", flow_weight_decay
    )
    validate_positive_int("flow_batch_size", flow_batch_size)
    validate_positive_int("em_glm_steps", em_glm_steps)
    em_validation_blocks = _validate_nonnegative_int(
        "em_validation_blocks", em_validation_blocks
    )
    validate_positive_int("em_patience", em_patience)
    emission_knot_spacing = _validate_positive_float(
        "emission_knot_spacing", emission_knot_spacing
    )
    log_rate_prior_std = _validate_positive_float(
        "log_rate_prior_std", log_rate_prior_std
    )
    validate_positive_int("spike_block_size", spike_block_size)

    position_time_arr = as_array(position_time)
    spike_times_arr = as_arrays(spike_times)
    marks_arr = as_arrays(spike_waveform_features)
    support, unitless_weights, exposure_weights, pos_arr = prepare_encoding_support(
        position_time_arr,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    n_electrodes = validate_population_lengths(
        "electrode",
        spike_times=spike_times_arr,
        spike_waveform_features=marks_arr,
    )
    mark_dimensions: list[int] = []
    for electrode, (t_e, m_e) in enumerate(
        zip(spike_times_arr, marks_arr, strict=True)
    ):
        validate_spike_feature_pair(t_e, m_e, electrode)
        mark_dimensions.append(int(np.shape(m_e)[1]))

    resolved_pos, resolved_cov = _resolve_electrode_geometry(
        contact_positions, noise_covariances, mark_dimensions
    )

    weight_sum = float(exposure_weights.sum())
    warn_if_no_encoding_exposure(
        exposure_weights, "Clusterless flow mixture", clusterless=True
    )

    is_track_interior = environment.is_track_interior_flat_
    interior_place_bin_centers = jnp.asarray(
        environment.place_bin_centers_[is_track_interior]
    )
    encoding_positions = environment.decoding_position(pos_arr[support.indices])
    support_exposure = np.asarray(exposure_weights[support.indices], dtype=np.float64)

    full_prepared = prepare_rate_map_basis(
        environment,
        environment.place_bin_edges_,
        interior_place_bin_centers,
        encoding_positions,
        support_exposure,
        emission_knot_spacing=emission_knot_spacing,
        log_rate_prior_std=log_rate_prior_std,
    )

    # Precompute fold cell exposures once across all electrodes.
    fold_specs: list[
        tuple[np.ndarray, np.ndarray, PreparedRateMapBasis, np.ndarray]
    ] = []
    if n_em_iterations > 0 and em_validation_blocks >= 2 and weight_sum > 0.0:
        block_ids = _contiguous_exposure_blocks(support_exposure, em_validation_blocks)
        for b in range(em_validation_blocks):
            val_mask = block_ids == b
            train_mask = ~val_mask
            tr_cell_exp = np.bincount(
                full_prepared.sample_cells,
                weights=np.where(train_mask, support_exposure, 0.0),
                minlength=full_prepared.n_position_cells,
            )
            val_cell_exp = np.bincount(
                full_prepared.sample_cells,
                weights=np.where(val_mask, support_exposure, 0.0),
                minlength=full_prepared.n_position_cells,
            )
            if float(np.sum(tr_cell_exp)) > 0.0 and float(np.sum(val_cell_exp)) > 0.0:
                tr_prep = PreparedRateMapBasis(
                    emission_design_info=full_prepared.emission_design_info,
                    sample_cells=full_prepared.sample_cells,
                    n_position_cells=full_prepared.n_position_cells,
                    cell_exposure=tr_cell_exp,
                    emission_design_matrix=full_prepared.emission_design_matrix,
                    emission_predict_matrix=full_prepared.emission_predict_matrix,
                    l2_penalty=full_prepared.l2_penalty,
                )
                fold_specs.append((train_mask, val_mask, tr_prep, val_cell_exp))

    support_position = np.full(len(pos_arr), -1, dtype=np.int64)
    support_position[support.indices] = np.arange(len(support.indices), dtype=np.int64)

    n_interior = len(interior_place_bin_centers)
    mean_rates = np.zeros(n_electrodes, dtype=np.float64)
    mark_mixtures: list[FlowMarkMixture | None] = [None] * n_electrodes
    rate_coefficients: list[np.ndarray | jax.Array | None] = [None] * n_electrodes
    log_rate_maps: list[np.ndarray | None] = [None] * n_electrodes
    summed_gpi = np.zeros(n_interior, dtype=np.float64)
    per_electrode_diag: list[dict[str, Any]] = []
    flow_kernel_cache = _FLOW_STEP_KERNEL_CACHE

    for electrode, (t_e, m_e) in enumerate(
        zip(spike_times_arr, marks_arr, strict=True)
    ):
        t_np = np.asarray(t_e, dtype=np.float64)
        m_np = np.asarray(m_e, dtype=np.float64)
        spike_index, left, right, fraction = support.event_basis(t_np)
        validate_finite(m_np[spike_index], "spike_waveform_features")
        left_share = (1.0 - fraction) * unitless_weights[left]
        right_share = fraction * unitless_weights[right]
        spike_weight = left_share + right_share
        positive = spike_weight > 0.0
        if not np.any(positive) or weight_sum <= 0.0:
            warnings.warn(
                f"Clusterless flow mixture: electrode {electrode} has no effective "
                "encoding spikes (empty or zero total weight); it is treated as "
                "zero-rate -- its observed decode spikes are explained by the "
                "background rate.",
                DataQualityWarning,
                stacklevel=user_stacklevel(),
            )
            per_electrode_diag.append(
                {
                    "n_iterations": 0,
                    "train_log_likelihood": [],
                    "validation_log_likelihood": [],
                }
            )
            continue

        mean_rates[electrode] = weighted_mean_rate(spike_weight, weight_sum)
        marks_eff = m_np[spike_index[positive]]
        mixture, coeffs, interior_rates, diag = _fit_single_electrode_flow_em(
            electrode=electrode,
            marks_eff=marks_eff,
            left_pos=support_position[left[positive]],
            right_pos=support_position[right[positive]],
            left_share=left_share[positive],
            right_share=right_share[positive],
            encoding_positions=encoding_positions,
            full_prepared=full_prepared,
            fold_specs=fold_specs,
            contact_positions=resolved_pos[electrode],
            noise_covariance=resolved_cov[electrode],
            n_mark_components=n_mark_components,
            covariance_rank=covariance_rank,
            mark_reg_covar=mark_reg_covar,
            n_flow_layers=n_flow_layers,
            n_spline_bins=n_spline_bins,
            spline_bound=spline_bound,
            flow_hidden_units=flow_hidden_units,
            neighborhood_radius=neighborhood_radius,
            max_local_channels=max_local_channels,
            flow_objective=flow_objective,
            n_em_iterations=n_em_iterations,
            flow_steps_per_em=flow_steps_per_em,
            flow_learning_rate=flow_learning_rate,
            flow_weight_decay=flow_weight_decay,
            flow_batch_size=flow_batch_size,
            em_glm_steps=em_glm_steps,
            em_patience=em_patience,
            random_state=random_state,
            optax_mod=optax_mod,
            flow_kernel_cache=flow_kernel_cache,
        )
        mark_mixtures[electrode] = mixture
        rate_coefficients[electrode] = coeffs
        log_rate_maps[electrode] = np.log(interior_rates)
        summed_gpi += np.sum(interior_rates, axis=1)
        per_electrode_diag.append(diag)

    max_chosen = max((d["n_iterations"] for d in per_electrode_diag), default=0)
    em_diagnostics: dict[str, Any] = {
        "n_iterations": int(max_chosen),
        "per_electrode": per_electrode_diag,
        "flow_objective": flow_objective,
    }

    return {
        "algorithm": "clusterless_flow_mixture",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "emission_design_info": full_prepared.emission_design_info,
        "mark_mixtures": mark_mixtures,
        "rate_coefficients": rate_coefficients,
        "log_rate_maps": log_rate_maps,
        "summed_ground_process_intensity": summed_gpi,
        "mean_rates": mean_rates,
        "mark_dimensions": mark_dimensions,
        "spike_block_size": int(spike_block_size),
        "disable_progress_bar": bool(disable_progress_bar),
        "rate_units": "Hz",
        "encoding_exposure_seconds": weight_sum,
        "em_diagnostics": em_diagnostics,
    }


# ---------------------------------------------------------------------------
# Compiled JAX prediction kernels (pure JAX — no optax dependency)
# ---------------------------------------------------------------------------


def _device_u_block(u_slice: np.ndarray, n_pad: int) -> jax.Array:
    """Pad standardized marks to ``n_pad`` on device without float32 overflow warnings."""
    with np.errstate(over="ignore"):
        return jnp.asarray(pad_rows(u_slice, n_pad), dtype=jnp.float32)


@partial(jax.jit, static_argnames=("flow_mode", "n_bins", "bound"))
def _predict_nonlocal_block_jax(
    u_block: jax.Array,
    log_background: jax.Array,
    log_det_input: jax.Array,
    flow_params: dict[str, jax.Array],
    cayley_matrices: jax.Array | None,
    cond_masks: jax.Array | None,
    cond_nbr_indices: jax.Array | None,
    cond_nbr_masks: jax.Array | None,
    trans_indices: jax.Array | None,
    trans_valid: jax.Array | None,
    cond_geom: jax.Array | None,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    noise_variance: jax.Array,
    is_component: jax.Array,
    scaled_rates: jax.Array,
    log_rate_scale: jax.Array,
    *,
    flow_mode: str,
    n_bins: int,
    bound: float,
) -> jax.Array:
    """Compute per-spike non-local log intensities ``(block, n_bins)`` in float32."""
    # Guard marks that overflowed float32 to inf so they cleanly hit background.
    row_finite = jnp.all(jnp.isfinite(u_block), axis=1)
    safe_u = jnp.where(row_finite[:, None], u_block, 0.0)
    z, log_det_flow = _apply_flow_forward_jax(
        safe_u,
        flow_params,
        flow_mode=flow_mode,
        n_bins=n_bins,
        bound=bound,
        cond_masks=cond_masks,
        cond_nbr_indices=cond_nbr_indices,
        cond_nbr_masks=cond_nbr_masks,
        trans_indices=trans_indices,
        trans_valid=trans_valid,
        cond_geom=cond_geom,
        precomputed_cayley=cayley_matrices,
    )
    log_gauss = _latent_woodbury_log_density_jax(
        z, means, basis, basis_variances, noise_variance, is_component
    )
    log_comp = log_det_input + log_det_flow[:, None] + log_gauss
    log_comp = jnp.where(
        row_finite[:, None] & is_component[None, :] & ~jnp.isnan(log_comp),
        log_comp,
        -jnp.inf,
    )

    comp_max = jnp.max(log_comp, axis=1)
    safe_comp_max = jnp.where(jnp.isfinite(comp_max), comp_max, 0.0)
    scaled_comp = jnp.exp(log_comp - safe_comp_max[:, None])
    mixed = jnp.matmul(
        scaled_comp, scaled_rates, precision=jax.lax.Precision.HIGHEST
    )  # (block, n_bins)
    log_active = jnp.where(
        mixed > 0.0,
        jnp.log(jnp.maximum(mixed, 1e-38))
        + safe_comp_max[:, None]
        + log_rate_scale[None, :],
        -jnp.inf,
    )
    return jnp.logaddexp(log_active, log_background[:, None])


@partial(jax.jit, static_argnames=("flow_mode", "n_bins", "bound"))
def _predict_local_block_jax(
    u_block: jax.Array,
    log_background: jax.Array,
    log_det_input: jax.Array,
    flow_params: dict[str, jax.Array],
    cayley_matrices: jax.Array | None,
    cond_masks: jax.Array | None,
    cond_nbr_indices: jax.Array | None,
    cond_nbr_masks: jax.Array | None,
    trans_indices: jax.Array | None,
    trans_valid: jax.Array | None,
    cond_geom: jax.Array | None,
    means: jax.Array,
    basis: jax.Array,
    basis_variances: jax.Array,
    noise_variance: jax.Array,
    is_component: jax.Array,
    log_rates_at_spikes: jax.Array,
    *,
    flow_mode: str,
    n_bins: int,
    bound: float,
) -> jax.Array:
    """Compute per-spike local log intensities ``(block,)`` in float32."""
    row_finite = jnp.all(jnp.isfinite(u_block), axis=1)
    safe_u = jnp.where(row_finite[:, None], u_block, 0.0)
    z, log_det_flow = _apply_flow_forward_jax(
        safe_u,
        flow_params,
        flow_mode=flow_mode,
        n_bins=n_bins,
        bound=bound,
        cond_masks=cond_masks,
        cond_nbr_indices=cond_nbr_indices,
        cond_nbr_masks=cond_nbr_masks,
        trans_indices=trans_indices,
        trans_valid=trans_valid,
        cond_geom=cond_geom,
        precomputed_cayley=cayley_matrices,
    )
    log_gauss = _latent_woodbury_log_density_jax(
        z, means, basis, basis_variances, noise_variance, is_component
    )
    log_terms = log_det_input + log_det_flow[:, None] + log_gauss + log_rates_at_spikes
    log_terms = jnp.where(
        row_finite[:, None] & is_component[None, :] & ~jnp.isnan(log_terms),
        log_terms,
        -jnp.inf,
    )
    has_any = jnp.any(~jnp.isneginf(log_terms), axis=1)
    safe_terms = jnp.where(has_any[:, None], log_terms, 0.0)
    log_active = jnp.where(has_any, jax.nn.logsumexp(safe_terms, axis=1), -jnp.inf)
    return jnp.logaddexp(log_active, log_background)


def _pad_mixture_for_predict(
    mixture: FlowMarkMixture,
    coefficients: np.ndarray | jax.Array,
    log_rates: np.ndarray | None,
    k_pad: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
]:
    """Pad one electrode's latent mixture and rate terms to ``k_pad``."""
    n_comp = mixture.n_components
    n_dims = mixture.n_features
    rank = mixture.basis_rank
    n_pad = k_pad - n_comp

    means_p = np.concatenate([mixture.means, np.zeros((n_pad, n_dims))], axis=0)
    basis_p = np.concatenate([mixture.basis, np.zeros((n_pad, n_dims, rank))], axis=0)
    var_p = np.concatenate([mixture.basis_variances, np.ones((n_pad, rank))], axis=0)
    is_comp = np.arange(k_pad) < n_comp

    # coefficients shape: (K, n_basis) -> padded (k_pad, n_basis).
    coeffs_p = jnp.asarray(pad_rows(coefficients, k_pad))

    scaled_rates_j: jax.Array | None = None
    log_rate_scale_j: jax.Array | None = None
    if log_rates is not None:
        lr = jnp.asarray(log_rates, dtype=jnp.float32)  # (n_bins, K)
        lr_scale = jnp.max(lr, axis=1)  # (n_bins,)
        scaled_rates_j = jnp.pad(
            jnp.exp(lr - lr_scale[:, None]).T,  # (K, n_bins)
            ((0, n_pad), (0, 0)),
        )
        log_rate_scale_j = lr_scale

    return (
        jnp.asarray(means_p, dtype=jnp.float32),
        jnp.asarray(basis_p, dtype=jnp.float32),
        jnp.asarray(var_p, dtype=jnp.float32),
        jnp.asarray(mixture.noise_variance, dtype=jnp.float32),
        jnp.asarray(is_comp),
        coeffs_p,
        scaled_rates_j,
        log_rate_scale_j,
    )


@requires_time_edges
def predict_clusterless_flow_mixture_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    *,
    environment: Environment,
    emission_design_info: TrackGraphBasis | TensorSplineBasis,
    mark_mixtures: Sequence[FlowMarkMixture | None],
    rate_coefficients: Sequence[np.ndarray | jax.Array | None],
    log_rate_maps: Sequence[np.ndarray | None],
    summed_ground_process_intensity: ArrayLike,
    mean_rates: ArrayLike,
    mark_dimensions: Sequence[int],
    mark_reference: Sequence[MarkReference | None],
    time_edges: ArrayLike | None = None,
    is_local: bool = False,
    spike_block_size: int = 1000,
    disable_progress_bar: bool = False,
    row_slice: slice | None = None,
    algorithm: str | None = None,
    environment_grid: dict[str, Any] | None = None,
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,
    em_diagnostics: dict[str, Any] | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Evaluate marked-point-process log likelihoods for ``clusterless_flow_mixture``.

    Parameters
    ----------
    position_time : array_like or None
        Timestamps of decoding position samples (required when ``is_local=True``).
    position : array_like or None
        Decoding animal position (required when ``is_local=True``).
    spike_times : sequence of array_like, length n_electrodes
        Decoding spike times per electrode, in seconds.
    spike_waveform_features : sequence of array_like, length n_electrodes
        Decoding waveform features per electrode, shape ``(n_spikes, n_features)``.
    environment : Environment
        Fitted environment matching the encoding fit.
    emission_design_info : TrackGraphBasis or TensorSplineBasis
        Fitted spatial spline basis.
    mark_mixtures : sequence of FlowMarkMixture or None
        Fitted flow-warped mark mixtures per electrode.
    rate_coefficients : sequence of np.ndarray or None
        Fitted GLM spline coefficients per electrode, shape ``(K, n_basis)``.
    log_rate_maps : sequence of np.ndarray or None
        Precomputed ``(n_interior_bins, K)`` log rates per electrode.
    summed_ground_process_intensity : array_like, shape (n_interior_bins,)
        Sum of ground process intensities across active electrodes (Hz).
    mean_rates : array_like, shape (n_electrodes,)
        Mean encoding firing rates per electrode (Hz).
    mark_dimensions : sequence of int, length n_electrodes
        Expected mark feature count per electrode.
    mark_reference : sequence of MarkReference or None, length n_electrodes
        Normalized Student-t reference mark densities per electrode.
    time_edges : array_like, shape (n_time + 1,), optional
        Decoding time-bin edges in seconds.
    is_local : bool, optional
        If True, evaluate local at-position likelihood ``(n_rows, 1)``; if
        False, evaluate full interior-bin likelihood ``(n_rows, n_interior)``.
    spike_block_size : int, optional
        Maximum spikes scored per device block, by default 1000.
    disable_progress_bar : bool, optional
        Whether to suppress electrode progress bars, by default False.
    row_slice : slice, optional
        Optional subset of decode rows to evaluate.
    algorithm : str, optional
        Recorded algorithm name (validated against ``"clusterless_flow_mixture"``).
    environment_grid : dict, optional
        Recorded environment grid fingerprint for contract checking.
    rate_units : str, default="Hz", keyword-only
        Recorded rate units (validated to be ``"Hz"``).
    encoding_exposure_seconds : float, optional
        Recorded weighted encoding exposure (ignored at predict).
    em_diagnostics : dict, optional
        Recorded EM diagnostics (ignored at predict).
    _spike_time_order : SpikeTimeOrder, optional
        Internal precomputed spike ordering.
    _time_grid : _DecodeTimeGrid, optional
        Internal precomputed decode time grid.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, 1) or (n_rows, n_interior_bins)
        Per-bin log likelihoods in nats.
    """
    del algorithm, encoding_exposure_seconds, em_diagnostics
    validate_encoding_contract(rate_units, environment, environment_grid)
    validate_positive_int("spike_block_size", spike_block_size)

    rows = prepare_decode_rows(
        cast(ArrayLike, time_edges), row_slice, _time_grid, _spike_time_order
    )
    spike_times_arr = as_arrays(spike_times)
    features_arr = as_arrays(spike_waveform_features)
    mean_rates_arr = np.asarray(mean_rates)
    n_electrodes = validate_population_lengths(
        "electrode",
        spike_times=spike_times_arr,
        spike_waveform_features=features_arr,
        mark_mixtures=mark_mixtures,
        rate_coefficients=rate_coefficients,
        log_rate_maps=log_rate_maps,
        mean_rates=mean_rates_arr,
        mark_dimensions=mark_dimensions,
        mark_reference=mark_reference,
    )
    for electrode, (t_e, f_e, dim) in enumerate(
        zip(spike_times_arr, features_arr, mark_dimensions, strict=True)
    ):
        validate_spike_feature_pair(t_e, f_e, electrode)
        if np.shape(f_e)[1] != int(dim):
            msg = f"waveform feature dimension changed for electrode {electrode}"
            raise ValidationError(
                msg,
                expected=f"{dim} features per spike",
                got=f"{np.shape(f_e)[1]} features",
                hint="Use the same waveform feature representation at fit and predict.",
            )

    n_rows = rows.stop - rows.start
    durations = jnp.asarray(rows.durations())
    bg_rate_total = background_rate(n_electrodes)

    if is_local:
        pos_time_req, pos_req = require_position(
            position_time, position, "local clusterless flow mixture likelihood"
        )
        local_pos_time = np.asarray(pos_time_req)
        local_pos = np.asarray(pos_req)
        local_pos = local_pos if local_pos.ndim > 1 else local_pos[:, None]
        row_position = snap_off_track_positions(
            get_position_at_time(
                local_pos_time,
                local_pos,
                decode_bin_centers(rows.time_edges, rows.start, rows.stop),
                environment,
            ),
            environment,
        )
        row_design = jnp.asarray(
            make_spline_predict_matrix(emission_design_info, row_position)
        )
        expected_rate = jnp.zeros((n_rows,))
        log_likelihood = jnp.zeros((n_rows,))
    else:
        log_likelihood = -durations[:, None] * (
            jnp.asarray(summed_ground_process_intensity)[None] + bg_rate_total
        )

    max_k = max(
        (m.n_components for m in mark_mixtures if m is not None),
        default=1,
    )
    k_pad = _power_of_two(max_k)

    padded_models: list[
        tuple[
            FlowMarkMixture,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array | None,
            jax.Array | None,
        ]
        | None
    ] = []
    for mixture, coeffs, lr_map in zip(
        mark_mixtures, rate_coefficients, log_rate_maps, strict=True
    ):
        if mixture is None or coeffs is None:
            padded_models.append(None)
            continue
        means_j, basis_j, var_j, nvar_j, is_c, coeffs_p, scaled_r_j, lr_scale_j = (
            _pad_mixture_for_predict(mixture, coeffs, lr_map, k_pad)
        )
        if is_local:
            expected_rate = expected_rate + glm_rates(
                row_design, coeffs_p
            ) @ is_c.astype(row_design.dtype)
        padded_models.append(
            (
                mixture,
                means_j,
                basis_j,
                var_j,
                nvar_j,
                is_c,
                coeffs_p,
                scaled_r_j,
                lr_scale_j,
            )
        )

    def _add_electrode(
        running_ll: jax.Array,
        spikes: ElectrodeSpikes,
        model: tuple[
            FlowMarkMixture,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array | None,
            jax.Array | None,
        ]
        | None,
    ) -> jax.Array:
        if model is None:
            return add_background_only(running_ll, spikes)

        (
            mixture,
            means_j,
            basis_j,
            var_j,
            nvar_j,
            is_c,
            coeffs_p,
            scaled_r_j,
            lr_scale_j,
        ) = model

        # Standardize in float64 on host so large offsets never lose digits
        # before float32 device execution.
        u_host = mixture.standardize_marks(spikes.marks)
        n_sp = u_host.shape[0]
        raw_row_ids = np.asarray(spikes.selection.bin_ind, dtype=np.int32)
        log_bg_host = np.asarray(spikes.log_background, dtype=np.float64)
        is_sorted = spikes.selection.indices_are_sorted

        flow_params_j = mixture._jax_flow_params(jnp.float32)
        cayley_j = (
            None
            if mixture.cayley_matrices is None
            else jnp.asarray(mixture.cayley_matrices, dtype=jnp.float32)
        )
        cond_m_j = (
            None if mixture.cond_masks is None else jnp.asarray(mixture.cond_masks)
        )
        nbr_i_j = (
            None
            if mixture.cond_nbr_indices is None
            else jnp.asarray(mixture.cond_nbr_indices)
        )
        nbr_m_j = (
            None
            if mixture.cond_nbr_masks is None
            else jnp.asarray(mixture.cond_nbr_masks)
        )
        trans_i_j = (
            None
            if mixture.trans_indices is None
            else jnp.asarray(mixture.trans_indices)
        )
        trans_v_j = (
            None if mixture.trans_valid is None else jnp.asarray(mixture.trans_valid)
        )
        cond_g_j = (
            None
            if mixture.cond_geom is None
            else jnp.asarray(mixture.cond_geom, dtype=jnp.float32)
        )
        log_det_in_j = jnp.asarray(mixture.log_det_input, dtype=jnp.float32)

        selected_times = (
            select_spike_rows(spikes.times, spikes.selection) if is_local else None
        )

        for start in range(0, n_sp, spike_block_size):
            stop = min(start + spike_block_size, n_sp)
            n_cur = stop - start
            n_pad = padded_spike_count(n_cur, spike_block_size)

            u_blk = _device_u_block(u_host[start:stop], n_pad)
            bg_blk = jnp.asarray(
                pad_rows(log_bg_host[start:stop], n_pad), dtype=jnp.float32
            )
            # Padding rows carry row id == n_rows, which deterministic_row_* drops.
            ids_pad = np.full((n_pad,), n_rows, dtype=np.int32)
            ids_pad[:n_cur] = raw_row_ids[start:stop]
            ids_j = jnp.asarray(ids_pad)

            if is_local:
                assert selected_times is not None
                blk_pos = snap_off_track_positions(
                    get_position_at_time(
                        local_pos_time,
                        local_pos,
                        selected_times[start:stop],
                        environment,
                    ),
                    environment,
                )
                blk_design = jnp.asarray(
                    pad_rows(
                        make_spline_predict_matrix(emission_design_info, blk_pos),
                        n_pad,
                    ),
                    dtype=jnp.float32,
                )
                lr_blk = jnp.log(glm_rates(blk_design, coeffs_p))
                terms = _predict_local_block_jax(
                    u_blk,
                    bg_blk,
                    log_det_in_j,
                    flow_params_j,
                    cayley_j,
                    cond_m_j,
                    nbr_i_j,
                    nbr_m_j,
                    trans_i_j,
                    trans_v_j,
                    cond_g_j,
                    means_j,
                    basis_j,
                    var_j,
                    nvar_j,
                    is_c,
                    lr_blk,
                    flow_mode=mixture.flow_mode,
                    n_bins=mixture.n_spline_bins,
                    bound=mixture.spline_bound,
                )
                running_ll = running_ll + deterministic_row_sum(
                    terms, ids_j, n_rows, indices_are_sorted=is_sorted
                )
            else:
                assert scaled_r_j is not None
                assert lr_scale_j is not None
                terms_2d = _predict_nonlocal_block_jax(
                    u_blk,
                    bg_blk,
                    log_det_in_j,
                    flow_params_j,
                    cayley_j,
                    cond_m_j,
                    nbr_i_j,
                    nbr_m_j,
                    trans_i_j,
                    trans_v_j,
                    cond_g_j,
                    means_j,
                    basis_j,
                    var_j,
                    nvar_j,
                    is_c,
                    scaled_r_j,
                    lr_scale_j,
                    flow_mode=mixture.flow_mode,
                    n_bins=mixture.n_spline_bins,
                    bound=mixture.spline_bound,
                )
                running_ll = deterministic_row_add(
                    running_ll, terms_2d, ids_j, indices_are_sorted=is_sorted
                )

        return running_ll

    log_likelihood = add_electrode_spike_terms(
        log_likelihood,
        spike_times_arr,
        features_arr,
        mark_reference,
        padded_models,
        _add_electrode,
        rows=rows,
        log_background_floor=LOG_RATE_EPS_HZ,
        desc=f"{'Local' if is_local else 'Position-Bin'} Likelihood (flow mixture)",
        disable_progress_bar=disable_progress_bar,
    )

    duration_evidence = log_bin_duration_evidence(
        spike_times_arr,
        rows.time_edges,
        row_slice,
        rows.spike_time_order,
        _time_grid=rows.time_grid,
    )
    if is_local:
        log_likelihood = log_likelihood - durations * (expected_rate + bg_rate_total)
        return (log_likelihood + duration_evidence)[:, None]
    return log_likelihood + duration_evidence[:, None]
