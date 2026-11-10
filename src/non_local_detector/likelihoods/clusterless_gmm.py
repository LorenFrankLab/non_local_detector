"""
Clusterless decoding using Gaussian Mixture Models (GMM).

Spike log intensities are the raw ``log(rate) + log p(pos, mark) - log
p_occ(pos)`` and the ground process is ``rate * p_joint(pos) / p_occ(pos)``,
where ``p_joint(pos)`` is the analytical position marginal of the same joint
GMM. Both are formed in log space. Each spike's intensity is combined with the
electrode's background ``RATE_EPS_HZ * q(mark)`` by ``logaddexp`` (``q`` is the
electrode's reference mark density, see ``mark_reference``), and each
electrode's expected count gains ``RATE_EPS_HZ``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

from non_local_detector._warnings import user_stacklevel
from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import (
    DataQualityWarning,
    ParameterWarning,
    ValidationError,
)
from non_local_detector.likelihoods.clusterless_spikes import (
    ElectrodeSpikes,
    add_background_only,
    add_electrode_spike_terms,
)
from non_local_detector.likelihoods.common import (
    LOG_RATE_EPS_HZ,
    SpikeTimeOrder,
    as_arrays,
    deterministic_row_add,
    deterministic_row_sum,
    environment_grid_fingerprint,
    get_position_at_time,
    interpolate_weights_at_spike_times,
    log_bin_duration_evidence,
    log_nonnegative,
    pad_rows,
    padded_spike_count,
    prepare_decode_rows,
    require_position,
    select_decode_marks,
    select_spike_rows,
    select_spikes_in_rows,
    snap_off_track_positions,
    spike_row_ids,
    sum_spikes_into_rows,
    validate_encoding_contract,
    validate_population_lengths,
    validate_positive_int,
    validate_spike_feature_pair,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.encoding_model import ClusterlessEncodingModel
from non_local_detector.likelihoods.gmm import (
    CovType,
    GaussianMixtureModel,
    as_floating,
    compute_precision_cholesky,
    effective_sample_count,
    estimate_log_prob_resp,
)
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
    background_rate,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    decode_bin_centers,
    requires_time_edges,
)

__all__ = [
    "ClusterlessGMMEncodingModel",
    "fit_clusterless_gmm_encoding_model",
    "predict_clusterless_gmm_log_likelihood",
]


# Default covariance regularization, in squared feature units.
DEFAULT_GMM_REG_COVAR = 1.0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _as_jnp(x: Any) -> jax.Array:
    """Input as a JAX array, with integer or boolean features as floats.

    Parameters
    ----------
    x : array-like
        Input to convert to JAX array.

    Returns
    -------
    jax.Array
        JAX array version of input, in a floating dtype for integer input.
    """
    return as_floating(x)


def _gmm_logp(gmm: GaussianMixtureModel, X: jax.Array | np.ndarray) -> jax.Array:
    """Log density under a fitted GMM.

    Parameters
    ----------
    gmm : GaussianMixtureModel
        Fitted Gaussian mixture model.
    X : jax.Array, shape (n_samples, n_features)
        Input samples to evaluate.

    Returns
    -------
    log_density : jax.Array, shape (n_samples,)
        Log probability density for each sample.
    """
    return gmm.score_samples(cast(jax.Array, X))


@partial(jax.jit, donate_argnums=(0,))
def _accumulate_log_likelihood_block(
    log_likelihood: jax.Array,
    joint_logp: jax.Array,
    segment_ids: jax.Array,
    bin_ids: jax.Array,
    log_rate: jax.Array,
    log_occupancy: jax.Array,
    log_background: jax.Array,
) -> jax.Array:
    """Scatter one spike block into its observed time rows and spatial columns.

    The density difference is formed first: adding ``log_rate`` to a huge log
    density before subtracting the other would round the rate term away. Each
    spike's intensity then gains its background ``log_background``.
    """
    log_contribution = jnp.logaddexp(
        log_rate + (joint_logp - log_occupancy), log_background[:, None]
    )
    if not bin_ids.shape[0]:
        return log_likelihood
    # Callers provide contiguous, unique spatial tile columns. Keep the initial
    # ground-process term and add the deterministically reduced contributions
    # without a full-row temporary or concurrent floating-point scatters.
    return deterministic_row_add(
        log_likelihood, log_contribution, segment_ids, bin_ids[0]
    )


# Local spike terms are padded like one non-local block of this size.
_LOCAL_SPIKE_BLOCK = 1000


def _gmm_parameters(gmm: GaussianMixtureModel) -> tuple[jax.Array, ...]:
    """Fitted means, Cholesky precisions and weights, for compiled scoring."""
    gmm._check_fitted()
    return cast(tuple[jax.Array, ...], (gmm.means_, gmm.precisions_chol_, gmm.weights_))


def _gmm_score(
    points: jax.Array, parameters: tuple[jax.Array, ...], covariance_type: CovType
) -> jax.Array:
    """Traceable ``GaussianMixtureModel.score_samples`` on fitted parameters."""
    means, precisions_chol, weights = parameters
    return cast(
        jax.Array,
        estimate_log_prob_resp(
            points, means, precisions_chol, covariance_type, weights
        )[0],
    )


@partial(
    jax.jit,
    static_argnames=("covariance_type", "block_size", "indices_are_sorted"),
    donate_argnums=0,
)
def _add_electrode_gmm_intensities(
    log_likelihood: jax.Array,
    features: jax.Array,
    row_ids: jax.Array,
    bin_centers: jax.Array,
    parameters: tuple[jax.Array, ...],
    log_rate: jax.Array,
    log_occupancy: jax.Array,
    log_background: jax.Array,
    *,
    covariance_type: CovType,
    block_size: int,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add one electrode's non-local log intensities to their rows.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows, n_bins)
        Running sum over electrodes; donated.
    features : jax.Array, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jax.Array, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    bin_centers : jax.Array, shape (n_bins, n_position_dims)
        Place bin centers.
    parameters : tuple of jax.Array
        The joint GMM's fitted means, Cholesky precisions and weights.
    log_rate : jax.Array, scalar
        Log of the electrode's mean rate.
    log_occupancy : jax.Array, shape (n_bins,)
        Log occupancy density at each bin.
    log_background : jax.Array, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.
    covariance_type : {"full", "tied", "diag", "spherical"}
        Covariance structure of the joint GMM.
    block_size : int
        Decoding spikes per block.
    indices_are_sorted : bool
        Whether ``row_ids`` are sorted in nondecreasing order.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_bins)
    """
    n_bins = bin_centers.shape[0]
    tiled_bins = jnp.tile(bin_centers, (block_size, 1))

    def add_block(number: jax.Array, log_likelihood: jax.Array) -> jax.Array:
        first = number * block_size
        block = jax.lax.dynamic_slice_in_dim(features, first, block_size)
        rows = jax.lax.dynamic_slice_in_dim(row_ids, first, block_size)
        background = jax.lax.dynamic_slice_in_dim(log_background, first, block_size)
        points = jnp.concatenate(
            [tiled_bins, jnp.repeat(block, n_bins, axis=0)], axis=1
        )
        log_density = _gmm_score(points, parameters, covariance_type).reshape(
            block_size, n_bins
        )
        # Density difference first (see _accumulate_log_likelihood_block).
        return deterministic_row_add(
            log_likelihood,
            jnp.logaddexp(
                log_rate + (log_density - log_occupancy), background[:, None]
            ),
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
    static_argnames=("joint_type", "occupancy_type", "indices_are_sorted"),
    donate_argnums=0,
)
def _add_electrode_gmm_spike_terms(
    log_likelihood: jax.Array,
    spike_positions: jax.Array,
    features: jax.Array,
    row_ids: jax.Array,
    joint: tuple[jax.Array, ...],
    occupancy: tuple[jax.Array, ...],
    mean_rate: jax.Array,
    log_background: jax.Array,
    *,
    joint_type: CovType,
    occupancy_type: CovType,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add one electrode's local spike terms to their rows.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_rows,)
        Running sum over electrodes; donated.
    spike_positions : jax.Array, shape (n_padded_spikes, n_position_dims)
        Position at each decoding spike; padding rows are zero.
    features : jax.Array, shape (n_padded_spikes, n_features)
        Waveform features of each decoding spike.
    row_ids : jax.Array, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    joint, occupancy : tuple of jax.Array
        Fitted means, Cholesky precisions and weights of each GMM.
    mean_rate : jax.Array, scalar
        Electrode's mean rate.
    log_background : jax.Array, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.
    joint_type : {"full", "tied", "diag", "spherical"}
        Covariance structure of the joint GMM.
    occupancy_type : {"full", "tied", "diag", "spherical"}
        Covariance structure of the occupancy GMM.
    indices_are_sorted : bool
        Whether ``row_ids`` are sorted in nondecreasing order.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows,)
    """
    joint_logp = _gmm_score(
        jnp.concatenate([spike_positions, features], axis=1), joint, joint_type
    )
    occupancy_logp = _gmm_score(spike_positions, occupancy, occupancy_type)
    # Density difference first (see _accumulate_log_likelihood_block).
    terms = jnp.logaddexp(
        log_nonnegative(mean_rate) + (joint_logp - occupancy_logp),
        log_background,
    )
    return log_likelihood + deterministic_row_sum(
        terms,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )


@partial(jax.jit, static_argnames=("gpi_type",), donate_argnums=0)
def _add_gmm_expected_counts(
    expected_counts: jax.Array,
    gpi: tuple[jax.Array, ...],
    positions: jax.Array,
    log_occupancy: jax.Array,
    mean_rate: jax.Array,
    *,
    gpi_type: CovType,
) -> jax.Array:
    """Add one electrode's expected count rate at the animal's position.

    Electrodes with and without spikes in a chunk use this same compiled
    function, so their expected counts round identically.

    Parameters
    ----------
    expected_counts : jax.Array, shape (n_rows,)
        Running sum over electrodes; donated.
    gpi : tuple of jax.Array
        Fitted means, Cholesky precisions and weights of the electrode's GPI GMM.
    positions : jax.Array, shape (n_rows, n_position_dims)
        Animal position at each row.
    log_occupancy : jax.Array, shape (n_rows,)
        Log occupancy density at ``positions``.
    mean_rate : jax.Array, scalar
        Electrode's mean rate.
    gpi_type : {"full", "tied", "diag", "spherical"}
        Covariance structure of the ground-process-intensity GMM.

    Returns
    -------
    expected_counts : jax.Array, shape (n_rows,)
    """
    return expected_counts + _ground_process_intensity(
        mean_rate, _gmm_score(positions, gpi, gpi_type), log_occupancy
    )


def _ground_process_intensity(
    mean_rate: jax.Array, gpi_logp: jax.Array, log_occupancy: jax.Array
) -> jax.Array:
    """Return expected spike intensity ``rate * p_gpi(x) / p_occ(x)``, formed in log space.

    Parameters
    ----------
    mean_rate : jax.Array, scalar
        Fitted mean firing rate of the electrode (positive).
    gpi_logp : jax.Array, shape (n_points,)
        Log density of the electrode's spike positions at the evaluation points.
    log_occupancy : jax.Array, shape (n_points,)
        Log occupancy density at the same points.

    Returns
    -------
    intensity : jax.Array, shape (n_points,)
        ``exp(log(rate) + gpi_logp - log_occupancy)``. Keeping every factor in
        the exponent preserves a finite ratio of two densities that each
        underflow (``exp(-800) / exp(-790)`` is ``0 / 0`` in float32; the
        ratio is ``e^-10``), and a representable product such as
        ``1e-15 * exp(95)`` whose separate ``exp`` overflows. A NaN in either
        density propagates; a ratio beyond the dtype's range overflows to
        ``inf`` (a ``-inf`` log likelihood) rather than being substituted.
        Both the fit-time bin intensities and the local expected counts use
        this function.
    """
    # Difference first: two nearly equal large log densities subtract exactly,
    # whereas adding log(rate) to one of them first rounds at their magnitude.
    return jnp.exp(jnp.log(mean_rate) + (gpi_logp - log_occupancy))


def _fit_gmm_density(
    X: jax.Array,
    weights: jax.Array | np.ndarray | None,
    n_components: int,
    random_state: int | None,
    covariance_type: CovType = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> GaussianMixtureModel:
    """Fit a GMM density model on samples X (optionally weighted).

    Parameters
    ----------
    X : jax.Array, shape (n_samples, n_features)
        Input samples for fitting.
    weights : jax.Array, shape (n_samples,), optional
        Sample weights, by default None (uniform weights). ``fit`` validates
        them and drops the zero-weight samples itself, so they influence
        neither the KMeans initialization nor EM and the fit equals a hard
        subset fit.
    n_components : int
        Number of Gaussian components in the mixture.
    random_state : int, optional
        Random seed for reproducible initialization, by default None.
    covariance_type : {"full", "tied", "diag", "spherical"}, optional
        Covariance parameterization {'full', 'tied', 'diag', 'spherical'},
        by default "full".
    reg_covar : float, optional
        Regularization term added to diagonal of covariance matrices,
        by default 1e-6.
    max_iter : int, optional
        Maximum EM iterations, by default 200.
    tol : float, optional
        Convergence threshold, by default 1e-3.

    Returns
    -------
    gmm : GaussianMixtureModel
        Fitted Gaussian mixture model.
    """
    X = _as_jnp(X)
    key = jax.random.PRNGKey(0 if random_state is None else random_state)
    gmm = GaussianMixtureModel(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        init_params="kmeans",
        kmeans_init="k-means++",
        kmeans_n_init=1,
        random_state=random_state,
    )
    gmm.fit(X, key, sample_weight=weights)
    return gmm


def _position_marginal(
    joint: GaussianMixtureModel, n_position_dims: int
) -> GaussianMixtureModel:
    """Derive a position-only GMM without fitting a second distribution.

    Parameters
    ----------
    joint : GaussianMixtureModel
        Fitted mixture over leading position coordinates followed by marks.
    n_position_dims : int
        Number of leading position coordinates to retain.

    Returns
    -------
    marginal : GaussianMixtureModel
        Same weights and component count, with the position mean and covariance
        blocks. Precision factors are recomputed: a precision subblock would
        instead describe a conditional Gaussian. Solver settings and convergence
        metadata describe the source joint fit; no marginal EM fit is performed
        and ``lower_bound_`` is None.
    """
    assert joint.means_ is not None
    assert joint.covariances_ is not None
    covariance = joint.covariances_
    if joint.covariance_type == "full":
        covariance = covariance[:, :n_position_dims, :n_position_dims]
    elif joint.covariance_type == "tied":
        covariance = covariance[:n_position_dims, :n_position_dims]
    elif joint.covariance_type == "diag":
        covariance = covariance[:, :n_position_dims]
    marginal = replace(joint)
    marginal.weights_ = joint.weights_
    marginal.means_ = joint.means_[:, :n_position_dims]
    marginal.covariances_ = covariance
    marginal.precisions_chol_ = compute_precision_cholesky(
        covariance, joint.covariance_type
    )
    marginal.converged_ = joint.converged_
    marginal.n_iter_ = joint.n_iter_
    return marginal


# ---------------------------------------------------------------------
# Encoding (fit) — GMM
# ---------------------------------------------------------------------


def _gmm_sample_weight(
    weights: np.ndarray, *, weights_was_none: bool
) -> np.ndarray | None:
    """sample_weight for a GMM EM fit, or None to take the unweighted path.

    Returns None when the caller passed no weights (keeps a direct unweighted
    fit byte-identical; detectors always pass their group mask as weights) or
    when the weights sum to 0; otherwise the weights. An
    all-zero ``sample_weight`` is rejected by ``GaussianMixtureModel.fit`` (it
    leaves no effective training data), so a fully de-weighted encoding group
    deliberately degrades to an unweighted occupancy fit after the caller's
    zero-exposure warning rather than raising; detectors pass the full
    position timeline, so that fallback occupancy covers every supplied sample,
    not only the group's. Because the unweighted
    fallback keeps every position row, the caller's ``occupancy_n_samples < 1``
    check below is then reachable only for a zero-row position array.
    """
    if weights_was_none or float(np.sum(weights)) == 0.0:
        return None
    return weights


def _validate_execution_settings(spike_block_size: Any, bin_tile_size: Any) -> None:
    """Reject block and tile settings the fit and the prediction cannot use."""
    validate_positive_int("spike_block_size", spike_block_size)
    validate_positive_int("bin_tile_size", bin_tile_size, allow_none=True)


def mark_reference_scale(
    gmm_reg_covar: float = DEFAULT_GMM_REG_COVAR, **_fit_params: object
) -> float:
    """Mark-kernel width the reference adds in quadrature to each feature's spread.

    Parameters
    ----------
    gmm_reg_covar : float, optional
        The fit's covariance regularization, a variance in squared feature
        units.
    **_fit_params
        Other fit parameters, ignored.

    Returns
    -------
    scale : float
        ``sqrt(gmm_reg_covar)``.
    """
    return float(np.sqrt(gmm_reg_covar))


class ClusterlessGMMEncodingModel(ClusterlessEncodingModel):
    """The encoding model ``fit_clusterless_gmm_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_clusterless_gmm_log_likelihood``, documented there. The
    component counts (``gmm_requested_components``,
    ``gmm_effective_components``) are fit diagnostics that predict ignores.
    """

    occupancy_model: GaussianMixtureModel
    interior_place_bin_centers: np.ndarray
    log_occupancy: jax.Array
    gpi_models: list[GaussianMixtureModel | None]
    joint_models: list[GaussianMixtureModel | None]
    summed_ground_process_intensity: np.ndarray
    spike_block_size: int
    bin_tile_size: int | None
    gmm_requested_components: dict[str, int]
    gmm_effective_components: dict[str, int | list[int]]


def fit_clusterless_gmm_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    gmm_components_occupancy: int = 32,
    gmm_components_joint: int = 64,
    gmm_covariance_type_occupancy: CovType = "full",
    gmm_covariance_type_joint: CovType = "full",
    gmm_random_state: int | None = 0,
    gmm_reg_covar: float = DEFAULT_GMM_REG_COVAR,
    gmm_max_iter: int = 200,
    gmm_tol: float = 0.001,
    spike_block_size: int = 1000,
    bin_tile_size: int | None = None,
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> ClusterlessGMMEncodingModel:
    """
    Fit the clusterless encoding model using GMMs.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Time of each position sample, in seconds.
    position : array_like, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Encoding spike times per electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Encoding spike waveform features per electrode.
    environment : Environment
        The spatial environment (must be fitted).
    weights : array_like, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM). None means
        uniform. Weights the occupancy and joint (position+mark) GMMs (via
        ``sample_weight``), and the mean firing rate. GPI is the joint marginal.
    gmm_components_occupancy : int, default=32
        Number of mixture components for occupancy GMM.
    gmm_components_joint : int, default=64
        Number of mixture components for joint (position+mark) GMM.
    gmm_covariance_type_occupancy : {"full", "tied", "diag", "spherical"}, default="full"
        Covariance type for occupancy GMM. Options: "full", "tied", "diag", "spherical".
    gmm_covariance_type_joint : {"full", "tied", "diag", "spherical"}, default="full"
        Covariance type for joint GMM. Options: "full", "tied", "diag", "spherical".
    gmm_random_state : int | None, default=0
        Random state for reproducibility.
    gmm_reg_covar : float, default=1.0
        Nonnegative covariance regularization used by the occupancy and joint
        GMMs, added to each covariance diagonal in squared feature units
        (cm^2 for positions, squared mark units for waveform features). It must
        exceed the float32 rounding of the covariances, about
        ``feature_magnitude**2 * 1e-7``: positions repeated during a pause 500 cm
        from the origin need more than about 0.02 cm^2. Scale it with the
        features if they are not in cm and microvolts.
    gmm_max_iter : int, default=200
        Maximum EM iterations for every fitted GMM.
    gmm_tol : float, default=1e-3
        Convergence tolerance for every fitted GMM.
    spike_block_size : int, default=1000
        Decode spikes scored per block at predict. Stored in the encoding model,
        which is how a detector's ``clusterless_algorithm_params`` reach
        ``predict_clusterless_gmm_log_likelihood``.
    bin_tile_size : int | None, default=None
        Position bins scored per tile at predict (None: all bins at once).
        Stored in the encoding model like ``spike_block_size``.
    disable_progress_bar : bool, default=False
        If True, disable progress bar.

    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds, replacing the outer bounds of the
        encoding support. A range inside the samples clips it, using the
        original interpolation basis; a range past them extends it, holding
        position at the first or last sample. Uniform samples otherwise include
        endpoint half-cells; a singleton requires explicit bounds or a tracking
        interval.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
        Required for irregular timestamps; each interval needs a finite sample.
        Positions are held at segment endpoints and NaN rows split support.
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support. Direct callers
        omit it and the support is built from the other arguments.

    Returns
    -------
    encoding_model : ClusterlessGMMEncodingModel
        Dictionary containing the fitted encoding model with keys:
        - algorithm ("clusterless_gmm")
        - environment
        - occupancy_model
        - interior_place_bin_centers
        - log_occupancy
        - gpi_models (analytical position marginals of joint_models)
        - joint_models
        - mean_rates
        - summed_ground_process_intensity
        - disable_progress_bar
        - spike_block_size
        - bin_tile_size
        - gmm_requested_components
        - gmm_effective_components
        - mark_dimensions
        - rate_units ("Hz")
        - encoding_exposure_seconds (weighted encoding occupancy time, seconds)
    """
    _validate_execution_settings(spike_block_size, bin_tile_size)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    weights_was_none = weights is None
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
    )
    position = _as_jnp(position)
    # NOTE: Do NOT convert position_time to JAX! It causes float64→float32 precision loss
    # with large timestamp values (e.g., Unix timestamps), creating apparent duplicates.
    # Keep as numpy for interpolation (scipy.interpolate.interpn requires numpy anyway).
    position_time = np.asarray(position_time)

    # Weighted occupancy time in seconds: per-sample weight times the tracking
    # time each sample covers (its integrated interpolation exposure).
    weight_sum = float(exposure_weights.sum())
    warn_if_no_encoding_exposure(exposure_weights, "Clusterless GMM", clusterless=True)
    occupancy_sample_weight = _gmm_sample_weight(
        exposure_weights, weights_was_none=False
    )

    is_track_interior = environment.is_track_interior_flat_

    # Occupancy/GPI position: linear position on a track graph.
    pos_for_occ = _as_jnp(environment.decoding_position(position))

    # Count the rows the GMM will actually fit on, using the array it fits: its
    # zero-weight cutoff depends on that array's dtype, so counting every
    # positive weight here could reserve more components than there are rows.
    occupancy_n_samples = effective_sample_count(pos_for_occ, occupancy_sample_weight)
    if occupancy_n_samples < 1:
        msg = "clusterless GMM has no position samples to fit the occupancy density"
        raise ValidationError(
            msg,
            expected="at least one position sample in the encoding period",
            got=f"{occupancy_n_samples} effective position samples",
            hint="Provide position data covering the encoding period.",
        )
    effective_occupancy_components = min(gmm_components_occupancy, occupancy_n_samples)
    if effective_occupancy_components < gmm_components_occupancy:
        warnings.warn(
            "Clusterless GMM: reduced occupancy components from "
            f"{gmm_components_occupancy} to {effective_occupancy_components} "
            f"because only {occupancy_n_samples} effective position samples are "
            "available.",
            ParameterWarning,
            stacklevel=user_stacklevel(),
        )

    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    # Fit occupancy GMM and precompute per-bin terms
    occupancy_model = _fit_gmm_density(
        X=pos_for_occ,
        weights=occupancy_sample_weight,
        n_components=effective_occupancy_components,
        random_state=gmm_random_state,
        covariance_type=gmm_covariance_type_occupancy,
        reg_covar=gmm_reg_covar,
        max_iter=gmm_max_iter,
        tol=gmm_tol,
    )
    log_occupancy = _gmm_logp(occupancy_model, interior_place_bin_centers)

    gpi_models: list[GaussianMixtureModel | None] = []
    joint_models: list[GaussianMixtureModel | None] = []
    mean_rates: list[float | jax.Array] = []
    effective_gpi_components: list[int] = []
    effective_joint_components: list[int] = []
    mark_dimensions: list[int] = []

    summed_ground_process_intensity = jnp.zeros_like(log_occupancy)

    # Fit per-electrode models
    for electrode, (elect_feats, elect_times) in enumerate(
        tqdm(
            zip(spike_waveform_features, spike_times, strict=True),
            desc="Encoding models (GMM)",
            unit="electrode",
            disable=disable_progress_bar,
        )
    ):
        validate_spike_feature_pair(elect_times, elect_feats, electrode)
        mark_dimensions.append(elect_feats.shape[1])
        # Clip to encoding window
        in_bounds = support.contains(elect_times)
        elect_times = elect_times[in_bounds]
        elect_feats = _as_jnp(elect_feats[in_bounds])
        # Weight each encoding spike by the posterior weight at its spike time.
        elect_weights = interpolate_weights_at_spike_times(
            elect_times, position_time, np.asarray(weights), encoding_support=support
        )

        # An electrode with no effective encoding spikes has no data to fit, so
        # its rate is zero. This covers two cases: the electrode had no spikes in
        # the encoding window, or every supplied weight for it is zero. Fitting a
        # GMM on zero (or fully de-weighted) spikes would both fail and, for the
        # de-weighted case, leak the excluded spatial pattern into decoding via
        # the joint density. Mark it zero-rate (mean rate 0, None GPI/joint
        # sentinels) and add no ground-process intensity (the rate is zero).
        # Predict explains each of its observed decode spikes by the background
        # alone -- the marked-point-process likelihood of a spike under a
        # zero-rate model -- rather than skipping the electrode (which would drop
        # that negative evidence), matching the KDE path. An electrode that has
        # spikes and no supplied weights still takes the normal unweighted fit
        # below.
        if not np.any(elect_weights > 0.0):
            warnings.warn(
                f"Clusterless GMM: electrode {electrode} has no effective encoding "
                "spikes (empty or zero total weight); it is treated as "
                "zero-rate -- its observed decode spikes are explained by the background rate "
                "rather than fit on the de-weighted spikes.",
                DataQualityWarning,
                stacklevel=user_stacklevel(),
            )
            mean_rates.append(0.0)
            gpi_models.append(None)
            joint_models.append(None)
            effective_gpi_components.append(0)
            effective_joint_components.append(0)
            continue

        # Weighted mean firing rate: weighted spike count / weighted occupancy time.
        mean_rate = jnp.asarray(weighted_mean_rate(elect_weights, weight_sum))
        mean_rates.append(mean_rate)

        # Positions at spike times
        enc_pos = get_position_at_time(
            position_time, position, elect_times, environment, encoding_support=support
        )

        joint_samples = jnp.concatenate([enc_pos, elect_feats], axis=1)
        elect_sample_weight = None if weights_was_none else elect_weights
        n_joint_spikes = effective_sample_count(joint_samples, elect_sample_weight)
        n_joint_components = min(gmm_components_joint, n_joint_spikes)
        effective_gpi_components.append(n_joint_components)
        effective_joint_components.append(n_joint_components)
        if n_joint_components < gmm_components_joint:
            warnings.warn(
                f"Clusterless GMM: electrode {electrode} has {n_joint_spikes} "
                "effective encoding spikes; reduced "
                f"joint components {gmm_components_joint}->{n_joint_components}.",
                ParameterWarning,
                stacklevel=user_stacklevel(),
            )

        # Joint GMM over [position, waveform]
        joint_gmm = _fit_gmm_density(
            X=joint_samples,
            weights=elect_sample_weight,
            n_components=n_joint_components,
            random_state=gmm_random_state,
            covariance_type=gmm_covariance_type_joint,
            reg_covar=gmm_reg_covar,
            max_iter=gmm_max_iter,
            tol=gmm_tol,
        )
        joint_models.append(joint_gmm)
        gpi_gmm = _position_marginal(joint_gmm, enc_pos.shape[1])
        gpi_models.append(gpi_gmm)

        # Expected-counts term at bins: mean_rate * (gpi / occupancy), in log space
        summed_ground_process_intensity += _ground_process_intensity(
            mean_rate,
            _gmm_logp(gpi_gmm, interior_place_bin_centers),
            log_occupancy,
        )

    return {
        "algorithm": "clusterless_gmm",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "occupancy_model": occupancy_model,
        "interior_place_bin_centers": interior_place_bin_centers,
        "log_occupancy": log_occupancy,
        "gpi_models": gpi_models,
        "joint_models": joint_models,
        "mean_rates": np.asarray(jnp.asarray(mean_rates)),
        "summed_ground_process_intensity": np.asarray(summed_ground_process_intensity),
        "disable_progress_bar": disable_progress_bar,
        "spike_block_size": spike_block_size,
        "bin_tile_size": bin_tile_size,
        "gmm_requested_components": {
            "occupancy": gmm_components_occupancy,
            "gpi": gmm_components_joint,
            "joint": gmm_components_joint,
        },
        "gmm_effective_components": {
            "occupancy": effective_occupancy_components,
            "gpi": effective_gpi_components,
            "joint": effective_joint_components,
        },
        "mark_dimensions": mark_dimensions,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


# ---------------------------------------------------------------------
# Decoding (non-local + local) — GMM
# ---------------------------------------------------------------------


@requires_time_edges
def predict_clusterless_gmm_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    *,
    environment: Environment,
    occupancy_model: GaussianMixtureModel,
    interior_place_bin_centers: jax.Array,
    log_occupancy: jax.Array,
    gpi_models: list[GaussianMixtureModel | None],
    joint_models: list[GaussianMixtureModel | None],
    mean_rates: np.ndarray,
    summed_ground_process_intensity: np.ndarray,
    is_local: bool = False,
    spike_block_size: int = 1000,
    bin_tile_size: int | None = None,
    disable_progress_bar: bool = False,
    row_slice: slice | None = None,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    gmm_requested_components: dict[str, Any] | None = None,  # noqa: ARG001 -- fit diagnostic, unused
    gmm_effective_components: dict[str, Any] | None = None,  # noqa: ARG001 -- fit diagnostic, unused
    mark_dimensions: list[int],
    mark_reference: list[MarkReference | None],
    environment_grid: dict[str, Any] | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """
    Predict the (non-local or local) log likelihood using the fitted GMM model.

    Parameters
    ----------
    position_time : array_like or None, shape (n_time_position,)
        Time of each position sample for the decoding period (used only by the
        local path; accepted for signature parity when ``is_local`` is False).
        May be None when ``is_local`` is False.
    position : array_like or None, shape (n_time_position, n_position_dims)
        Position samples during the decoding period (used only by the local
        path; accepted for signature parity when ``is_local`` is False).
        May be None when ``is_local`` is False.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Decoding spike times per electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Decoding spike waveform features per electrode.
    environment : Environment
        Fitted environment. Its grid must be the one the model was fit on
        (see ``environment_grid``). The local path uses its track graph, if
        any, to linearize the animal's position, and moves positions off the
        track to the nearest interior bin.
    occupancy_model : GaussianMixtureModel
        Occupancy density over position (1-D linearized position when the
        environment has a track graph and position is multi-dimensional).
    interior_place_bin_centers : jax.Array, shape (n_interior_bins, n_position_dims)
        Centers of the environment's interior place bins.
    log_occupancy : jax.Array, shape (n_interior_bins,)
        Log occupancy density at ``interior_place_bin_centers``.
    gpi_models : list of GaussianMixtureModel or None, length n_electrodes
        Position marginal of each electrode's joint GMM; None for a zero-rate
        electrode.
    joint_models : list of GaussianMixtureModel or None, length n_electrodes
        Joint (position, mark) GMM per electrode; None for a zero-rate electrode.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Weighted mean firing rate per electrode, in Hz.
    summed_ground_process_intensity : np.ndarray, shape (n_interior_bins,)
        Sum over electrodes of ``rate * p_gpi(x) / p_occ(x)`` at the interior
        bins, in Hz (without the background rate).
    is_local : bool, default=False
        If True, compute local likelihood at the animal's position.
        Else compute non-local likelihood across interior bins.
    spike_block_size : int, default=1000
        Process spikes in blocks of this size to reduce peak memory.
        Reduces memory from O(n_spikes × n_bins) to O(spike_block_size × n_bins).
        A fitted encoding model supplies it and ``bin_tile_size`` (both are
        fit parameters).
    bin_tile_size : int | None, default=None
        If provided, tile computation over position bins in chunks of this size.
        Reduces memory from O(spike_block_size × n_bins) to O(spike_block_size × bin_tile_size).
        Useful for very large position grids (> 2000 bins).
    disable_progress_bar : bool, default=False
        Turn off the progress bar.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges: spikes are binned
        against them and only those owned by the requested rows are evaluated, so
        the result equals the full-time result sliced by ``row_slice`` while the
        spatial workspaces scale with the requested rows and selected spikes.
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    algorithm : str, optional
        The algorithm the fit recorded. Accepted so the encoding dictionary
        can be passed as keywords; unused.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the encoding fit; anything other than ``"Hz"``
        raises ``ValidationError`` (refit legacy encoding models).
    encoding_exposure_seconds : float, optional, keyword-only
        Weighted encoding occupancy time in seconds, recorded by the fit.
        Accepted so the encoding dictionary can be passed as keywords; unused.
    gmm_requested_components, gmm_effective_components : dict, optional
        The fit's requested and fitted component counts, kept in the encoding
        model as diagnostics; unused.
    mark_dimensions : list[int], keyword-only
        Fitted waveform-feature count per electrode, taken from the encoding
        model. Predict rejects decode spikes whose feature dimension differs, so
        every electrode (including zero-rate ones) must have a recorded count.
    mark_reference : list of MarkReference or None, keyword-only
        Normalized reference mark density per electrode, from
        ``fit_mark_references`` (a detector passes its ``mark_reference_``).
        Each spike's intensity gains ``RATE_EPS_HZ * q(mark)`` and each
        electrode's expected count ``RATE_EPS_HZ``, so likelihood ratios between
        states do not depend on the waveform units. None for an electrode means
        no training marks; its background is the bare rate.
    environment_grid : dict | None, optional
        The fit's record of the environment's grid (see
        ``environment_grid_fingerprint``), by default None, which skips the
        check (a model fitted before the grid was recorded). Prediction raises
        if the environment has since been refit to another grid.
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood :
        If non-local: jax.Array, shape (n_rows, n_interior_bins)
        If local    : jax.Array, shape (n_rows, 1)
        ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    _validate_execution_settings(spike_block_size, bin_tile_size)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        gpi_models=gpi_models,
        joint_models=joint_models,
        mean_rates=mean_rates,
        mark_dimensions=mark_dimensions,
        mark_reference=mark_reference,
    )

    for electrode, (elect_times, elect_feats, expected_mark_dims) in enumerate(
        zip(spike_times, spike_waveform_features, mark_dimensions, strict=True)
    ):
        validate_spike_feature_pair(elect_times, elect_feats, electrode)
        if np.shape(elect_feats)[1] != expected_mark_dims:
            msg = f"waveform feature dimension changed for electrode {electrode}"
            raise ValidationError(
                msg,
                expected=f"{expected_mark_dims} features per spike",
                got=f"{np.shape(elect_feats)[1]} features",
                hint="Use the same waveform feature representation at fit and predict.",
            )

    # The position arrays are read only by compute_local_log_likelihood, which
    # keeps them on the host. Converting them to device arrays here would copy
    # the FULL-recording position on every call -- once per chunk under chunked
    # prediction -- for a non-local likelihood that never reads it.
    if is_local:
        position_time, position = require_position(
            position_time, position, "local clusterless GMM likelihood"
        )
        local = compute_local_log_likelihood(
            time_edges=rows.time_edges,
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_waveform_features,
            environment=environment,
            occupancy_model=occupancy_model,
            gpi_models=gpi_models,
            joint_models=joint_models,
            mean_rates=mean_rates,
            mark_reference=mark_reference,
            disable_progress_bar=disable_progress_bar,
            row_slice=row_slice,
            _spike_time_order=rows.spike_time_order,
            _time_grid=rows.time_grid,
        )

        return (
            local
            + log_bin_duration_evidence(
                spike_times,
                rows.time_edges,
                row_slice,
                rows.spike_time_order,
                _time_grid=rows.time_grid,
            )[:, None]
        )

    n_bins = interior_place_bin_centers.shape[0]

    def add_electrode(
        log_likelihood: jax.Array,
        spikes: ElectrodeSpikes,
        model: tuple[GaussianMixtureModel | None, Any],
    ) -> jax.Array:
        joint_gmm, mean_rate = model
        # A None model marks a zero-rate electrode: it has no effective encoding
        # spikes, so no density was fit and each observed spike is explained by
        # the background alone, uniformly across position bins. This is *not*
        # the same as skipping the electrode: skipping would drop the negative
        # evidence an observed spike carries (e.g. against a state whose
        # encoding de-weighted this electrode).
        if joint_gmm is None:
            return add_background_only(log_likelihood, spikes)

        log_mean_rate = log_nonnegative(mean_rate)
        n_spikes = spikes.n_spikes

        if bin_tile_size is None or bin_tile_size >= n_bins:
            # Spikes padded to a few counts (padding rows are dropped from the
            # row sums) let electrodes and chunks share one compiled kernel.
            # Each spike is scored at every bin, so pad finely.
            n_padded = padded_spike_count(n_spikes, spike_block_size, fine=True)
            return cast(
                jax.Array,
                _add_electrode_gmm_intensities(
                    log_likelihood,
                    jnp.asarray(pad_rows(spikes.marks, n_padded)),
                    jnp.asarray(spike_row_ids(spikes.selection, n_padded)),
                    interior_place_bin_centers,
                    _gmm_parameters(joint_gmm),
                    log_mean_rate,
                    log_occupancy,
                    jnp.asarray(pad_rows(spikes.log_background, n_padded)),
                    covariance_type=joint_gmm.covariance_type,
                    block_size=min(spike_block_size, n_padded),
                    indices_are_sorted=spikes.selection.indices_are_sorted,
                ),
            )

        # Bin tiling: accumulate per spike block and position tile directly.
        # Memory: O(spike_block_size × tile_size), with no full block×bins array.
        # Select FIRST, then convert: converting the whole array would
        # device-copy every decoding spike in the recording on every chunk call.
        elect_feats = jnp.asarray(spikes.marks)
        log_background = jnp.asarray(spikes.log_background)
        for spike_start in range(0, n_spikes, spike_block_size):
            spike_end = min(spike_start + spike_block_size, n_spikes)
            block_feats = elect_feats[spike_start:spike_end]
            block_seg_ids = spikes.selection.bin_ind[spike_start:spike_end]
            block_size = block_feats.shape[0]
            for bin_start in range(0, n_bins, bin_tile_size):
                bin_end = min(bin_start + bin_tile_size, n_bins)
                n_tile = bin_end - bin_start

                # Build eval points for this tile
                tiled_bins_tile = jnp.tile(
                    interior_place_bin_centers[bin_start:bin_end], (block_size, 1)
                )
                repeated_feats_tile = jnp.repeat(block_feats, n_tile, axis=0)
                eval_points_tile = jnp.concatenate(
                    [tiled_bins_tile, repeated_feats_tile], axis=1
                )

                # GMM evaluation, dispatched per tile
                joint_logp_tile = _gmm_logp(joint_gmm, eval_points_tile).reshape(
                    block_size, n_tile
                )

                # Scatter this block directly into the tile columns. The
                # working arrays remain O(block_size × tile_size); the only
                # recording-length array is the returned likelihood itself.
                log_likelihood = _accumulate_log_likelihood_block(
                    log_likelihood,
                    joint_logp_tile,
                    block_seg_ids,
                    jnp.arange(bin_start, bin_end),
                    log_mean_rate,
                    log_occupancy[bin_start:bin_end],
                    log_background[spike_start:spike_end],
                )
        return log_likelihood

    # Start with the expected-counts (ground process) term, broadcast over
    # time, then add each electrode's spikes in log space.
    log_likelihood = add_electrode_spike_terms(
        -jnp.asarray(rows.durations())[:, None]
        * (summed_ground_process_intensity + background_rate(len(spike_times))),
        spike_times,
        spike_waveform_features,
        mark_reference,
        list(zip(joint_models, mean_rates, strict=True)),
        add_electrode,
        rows=rows,
        log_background_floor=LOG_RATE_EPS_HZ,
        desc="Position-Bin Likelihood (GMM, log-space)",
        disable_progress_bar=disable_progress_bar,
    )

    return (
        log_likelihood
        + log_bin_duration_evidence(
            spike_times,
            rows.time_edges,
            row_slice,
            rows.spike_time_order,
            _time_grid=rows.time_grid,
        )[:, None]
    )


def compute_local_log_likelihood(
    time_edges: np.ndarray,
    position_time: jax.Array | np.ndarray,
    position: jax.Array | np.ndarray,
    spike_times: Sequence[np.ndarray | jax.Array],
    spike_waveform_features: Sequence[np.ndarray | jax.Array],
    environment: Environment,
    occupancy_model: GaussianMixtureModel,
    gpi_models: list[GaussianMixtureModel | None],
    joint_models: list[GaussianMixtureModel | None],
    mean_rates: np.ndarray,
    *,
    disable_progress_bar: bool = False,
    row_slice: slice | None = None,
    mark_reference: list[MarkReference | None],
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Local log-likelihood at the animal's interpolated position.

    Computes the likelihood of observing spikes at the animal's true position
    at each time bin, using the fitted GMM encoding model.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : np.ndarray or jax.Array, shape (n_time_position,)
        Timestamps for position samples.
    position : np.ndarray or jax.Array, shape (n_time_position, n_position_dims)
        Position samples during decoding period.
    spike_times : sequence of np.ndarray or jax.Array
        Spike times per electrode during decoding.
    spike_waveform_features : sequence of np.ndarray or jax.Array
        Spike waveform features per electrode during decoding.
    environment : Environment
        Fitted environment; its track graph, if any, linearizes the position.
    occupancy_model : GaussianMixtureModel
        Occupancy density over (possibly linearized) position.
    gpi_models : list of GaussianMixtureModel or None, length n_electrodes
        Position marginal of each electrode's joint GMM; None for a zero-rate
        electrode.
    joint_models : list of GaussianMixtureModel or None, length n_electrodes
        Joint (position, mark) GMM per electrode; None for a zero-rate electrode.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Weighted mean firing rate per electrode, in Hz.
    disable_progress_bar : bool, optional
        Turn off progress bar display, by default False.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` stay the FULL decoding edges (see
        ``predict_clusterless_gmm_log_likelihood``).
    mark_reference : list of MarkReference or None
        Reference mark density per electrode for the background intensity (see
        ``predict_clusterless_gmm_log_likelihood``).
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, 1)
        Log likelihood at the animal's position for each requested time bin.
    """
    # NOTE: Keep position_time and position as numpy to avoid float64→float32
    # precision loss. Every consumer below is host-side (``get_position_at_time``
    # interpolates with scipy), so converting the recording-length position to a
    # device array would copy the whole recording on every chunk call for an
    # array this path never evaluates on device. The working dtype is the one a
    # full conversion would have produced: the fit's floating dtype (integer
    # positions as floats; float64 as float32 unless x64 is enabled).
    position_time = np.asarray(position_time)
    position = np.asarray(position)
    position = position if position.ndim > 1 else position[:, None]
    working_dtype = _as_jnp(position[:0]).dtype

    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    n_rows = rows.stop - rows.start

    # Interpolate position at the requested rows' bin times (use bin centers)

    interp_pos = snap_off_track_positions(
        get_position_at_time(
            position_time,
            position,
            decode_bin_centers(rows.time_edges, rows.start, rows.stop),
            environment,
        ),
        environment,
    )  # (n_rows, pos_dims)

    # Occupancy density and its log at the animal's position
    log_occ_at_pos = _gmm_logp(occupancy_model, interp_pos)  # (n_rows,)

    log_likelihood = jnp.zeros((n_rows,), dtype=working_dtype)
    summed_expected_counts = jnp.zeros((n_rows,), dtype=working_dtype)

    for electrode, (
        elect_feats,
        elect_times,
        joint_gmm,
        gpi_gmm,
        mean_rate,
        reference,
    ) in enumerate(
        tqdm(
            zip(
                spike_waveform_features,
                spike_times,
                joint_models,
                gpi_models,
                mean_rates,
                mark_reference,
                strict=True,
            ),
            desc="Local Likelihood (GMM, log-space)",
            unit="electrode",
            disable=disable_progress_bar,
        )
    ):
        # None marks a zero-rate electrode with no effective encoding spikes: each
        # observed spike is explained by the background alone. Do not skip -- an
        # observed spike is negative evidence, not "no data".
        selection = select_spikes_in_rows(
            elect_times,
            rows.start,
            rows.stop,
            time_edges=rows.time_edges,
            _spike_time_order=rows.spike_time_order,
        )
        n_spikes = selection.bin_ind.shape[0]
        if n_spikes:
            selected_feats = select_decode_marks(elect_feats, selection, electrode)
            log_background = background_log_intensity(
                reference, selected_feats, LOG_RATE_EPS_HZ
            )
        if joint_gmm is None:
            if n_spikes:
                log_likelihood = log_likelihood + sum_spikes_into_rows(
                    np.asarray(log_background, dtype=working_dtype), selection
                )
            continue

        if n_spikes:
            # Spikes padded to a few counts (padding rows are dropped from the
            # row sums) let electrodes and chunks share one compiled kernel.
            n_padded = padded_spike_count(n_spikes, _LOCAL_SPIKE_BLOCK)
            spike_positions = snap_off_track_positions(
                get_position_at_time(
                    position_time,
                    position,
                    select_spike_rows(elect_times, selection),
                    environment,
                ),
                environment,
            )
            log_likelihood = _add_electrode_gmm_spike_terms(
                log_likelihood,
                jnp.asarray(pad_rows(spike_positions, n_padded), dtype=working_dtype),
                jnp.asarray(pad_rows(selected_feats, n_padded)),
                jnp.asarray(spike_row_ids(selection, n_padded)),
                _gmm_parameters(joint_gmm),
                _gmm_parameters(occupancy_model),
                mean_rate,
                jnp.asarray(pad_rows(log_background, n_padded), dtype=working_dtype),
                joint_type=joint_gmm.covariance_type,
                occupancy_type=occupancy_model.covariance_type,
                indices_are_sorted=selection.indices_are_sorted,
            )
        summed_expected_counts = _add_gmm_expected_counts(
            summed_expected_counts,
            _gmm_parameters(gpi_gmm),
            interp_pos,
            log_occ_at_pos,
            mean_rate,
            gpi_type=gpi_gmm.covariance_type,
        )

    log_likelihood = log_likelihood - jnp.asarray(rows.durations()) * (
        summed_expected_counts + background_rate(len(spike_times))
    )

    return log_likelihood[:, None]
