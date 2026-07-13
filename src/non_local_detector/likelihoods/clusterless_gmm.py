"""
Clusterless decoding using Gaussian Mixture Models (GMM)
"""

from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    get_position_at_time,
    get_spike_time_bin_ind,
    interpolate_weights_at_spike_times,
    safe_log,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.gmm import GaussianMixtureModel

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _as_jnp(x) -> jnp.ndarray:
    """Convert input to JAX array if not already.

    Parameters
    ----------
    x : array-like
        Input to convert to JAX array.

    Returns
    -------
    jnp.ndarray
        JAX array version of input.
    """
    return x if isinstance(x, jnp.ndarray) else jnp.asarray(x)


def _gmm_logp(gmm: GaussianMixtureModel, X: jnp.ndarray) -> jnp.ndarray:
    """Log density under a fitted GMM.

    Parameters
    ----------
    gmm : GaussianMixtureModel
        Fitted Gaussian mixture model.
    X : jnp.ndarray, shape (n_samples, n_features)
        Input samples to evaluate.

    Returns
    -------
    log_density : jnp.ndarray, shape (n_samples,)
        Log probability density for each sample.
    """
    return gmm.score_samples(X)


def _gmm_density(gmm: GaussianMixtureModel, X: jnp.ndarray) -> jnp.ndarray:
    """Density under a fitted GMM.

    Parameters
    ----------
    gmm : GaussianMixtureModel
        Fitted Gaussian mixture model.
    X : jnp.ndarray, shape (n_samples, n_features)
        Input samples to evaluate.

    Returns
    -------
    density : jnp.ndarray, shape (n_samples,)
        Probability density for each sample.
    """
    return jnp.exp(gmm.score_samples(X))


@partial(jax.jit, donate_argnums=(0,))
def _accumulate_log_likelihood_block(
    log_likelihood: jnp.ndarray,
    joint_logp: jnp.ndarray,
    segment_ids: jnp.ndarray,
    bin_ids: jnp.ndarray,
    log_rate: jnp.ndarray,
    log_occupancy: jnp.ndarray,
) -> jnp.ndarray:
    """Scatter one spike block into its observed time rows and spatial columns."""
    log_contribution = log_rate + joint_logp - log_occupancy
    return log_likelihood.at[segment_ids[:, None], bin_ids[None, :]].add(
        log_contribution
    )


def _fit_gmm_density(
    X: jnp.ndarray,
    weights: jnp.ndarray | None,
    n_components: int,
    random_state: int | None,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> GaussianMixtureModel:
    """Fit a GMM density model on samples X (optionally weighted).

    Parameters
    ----------
    X : jnp.ndarray, shape (n_samples, n_features)
        Input samples for fitting.
    weights : jnp.ndarray, shape (n_samples,), optional
        Sample weights, by default None (uniform weights). Zero-weight samples
        are dropped before fitting so they influence neither the KMeans
        initialization nor EM, keeping the fit equal to a hard subset.
    n_components : int
        Number of Gaussian components in the mixture.
    random_state : int, optional
        Random seed for reproducible initialization, by default None.
    covariance_type : str, optional
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
    sample_weight = None
    if weights is not None:
        # A zero-weight sample is absent from the weighted density. Remove it
        # before KMeans initialization as well as EM; sklearn's unweighted
        # KMeans would otherwise let mathematically excluded samples choose the
        # initial component centers and break binary-weight/subset equivalence.
        weights_np = np.asarray(weights)
        positive_weight = weights_np > 0.0
        if np.all(positive_weight):
            # Posterior weights used during EM are normally all positive. Avoid
            # gathering millions of rows through an all-True mask on every
            # occupancy, GPI, and joint refit.
            sample_weight = _as_jnp(weights_np)
        else:
            X = X[positive_weight]
            sample_weight = _as_jnp(weights_np[positive_weight])

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
    gmm.fit(X, key, sample_weight=sample_weight)
    return gmm


# ---------------------------------------------------------------------
# Encoding (fit) — GMM
# ---------------------------------------------------------------------


def _gmm_sample_weight(weights: np.ndarray, weights_was_none: bool):
    """sample_weight for a GMM EM fit, or None to take the unweighted path.

    Returns None when the caller passed no weights (keeps the unweighted fit
    byte-identical) or when the weights sum to 0 (an all-zero ``sample_weight`` makes
    the EM M-step divide by 0 and raise "Fitting failed"); otherwise the weights.
    """
    if weights_was_none or float(np.sum(weights)) == 0.0:
        return None
    return weights


def _validate_spike_feature_pair(
    spike_times: np.ndarray | jnp.ndarray,
    spike_features: np.ndarray | jnp.ndarray,
    electrode: int,
) -> tuple[np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray]:
    """Validate one electrode's parallel spike/mark arrays without copying them."""
    times_shape = np.shape(spike_times)
    features_shape = np.shape(spike_features)
    if len(times_shape) != 1:
        raise ValidationError(
            f"spike_times for electrode {electrode} must be 1-D",
            expected="shape (n_spikes,)",
            got=f"shape {times_shape}",
        )
    if len(features_shape) != 2:
        raise ValidationError(
            f"spike_waveform_features for electrode {electrode} must be 2-D",
            expected="shape (n_spikes, n_features)",
            got=f"shape {features_shape}",
        )
    if features_shape[0] != times_shape[0]:
        raise ValidationError(
            f"spike times and waveform features disagree for electrode {electrode}",
            expected=f"{times_shape[0]} waveform-feature rows",
            got=f"{features_shape[0]} rows",
            hint="Provide exactly one waveform-feature row for every spike time.",
        )
    return spike_times, spike_features


def fit_clusterless_gmm_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    sampling_frequency: int = 500,
    weights: jnp.ndarray | None = None,
    *,
    gmm_components_occupancy: int = 32,
    gmm_components_gpi: int = 32,
    gmm_components_joint: int = 64,
    gmm_covariance_type_occupancy: str = "full",
    gmm_covariance_type_gpi: str = "full",
    gmm_covariance_type_joint: str = "full",
    gmm_random_state: int | None = 0,
    gmm_reg_covar: float = 1e-6,
    gmm_max_iter: int = 200,
    gmm_tol: float = 1e-3,
    disable_progress_bar: bool = False,
    **kwargs,  # Accept but ignore KDE-specific parameters for API compatibility
) -> dict:
    """
    Fit the clusterless encoding model using GMMs.

    Parameters
    ----------
    position_time : jnp.ndarray, shape (n_time_position,)
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
    spike_times : list[jnp.ndarray]
        Encoding spike times per electrode.
    spike_waveform_features : list[jnp.ndarray]
        Encoding spike waveform features per electrode.
    environment : Environment
    weights : jnp.ndarray | None, shape (n_time_position,), default=None
        Per-sample weights (e.g. posterior state probabilities during EM). None means
        uniform. Weights the occupancy, per-electrode GPI and joint (position+mark)
        GMMs (via ``sample_weight``), and the mean firing rate.
    gmm_components_occupancy : int, default=32
        Number of mixture components for occupancy GMM.
    gmm_components_gpi : int, default=32
        Number of mixture components for GPI (position-only) GMM.
    gmm_components_joint : int, default=64
        Number of mixture components for joint (position+mark) GMM.
    gmm_covariance_type_occupancy : str, default="full"
        Covariance type for occupancy GMM. Options: "full", "tied", "diag", "spherical".
    gmm_covariance_type_gpi : str, default="full"
        Covariance type for GPI GMM. Options: "full", "tied", "diag", "spherical".
    gmm_covariance_type_joint : str, default="full"
        Covariance type for joint GMM. Options: "full", "tied", "diag", "spherical".
        Note: "diag" may offer speedups but currently has JIT compatibility issues.
    gmm_random_state : int | None, default=0
        Random state for reproducibility.
    gmm_reg_covar : float, default=1e-6
        Nonnegative covariance regularization used by every occupancy, GPI, and
        joint GMM.
    gmm_max_iter : int, default=200
        Maximum EM iterations for every fitted GMM.
    gmm_tol : float, default=1e-3
        Convergence tolerance for every fitted GMM.
    disable_progress_bar : bool, default=False
        If True, disable progress bar.

    Returns
    -------
    encoding_model : dict
        Dictionary containing the fitted encoding model with keys:
        - environment
        - occupancy_model
        - interior_place_bin_centers
        - log_occupancy
        - gpi_models
        - joint_models
        - mean_rates
        - summed_ground_process_intensity
        - disable_progress_bar
        - gmm_requested_components
        - gmm_effective_components
        - mark_dimensions
    """
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
    )
    position = _as_jnp(position if position.ndim > 1 else position[:, None])
    # NOTE: Do NOT convert position_time to JAX! It causes float64→float32 precision loss
    # with large timestamp values (e.g., Unix timestamps), creating apparent duplicates.
    # Keep as numpy for interpolation (scipy.interpolate.interpn requires numpy anyway).
    position_time = np.asarray(position_time)

    weights_was_none = weights is None
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = validate_weights(weights, position.shape[0])
    # Weighted occupancy "time": sum of per-sample weights (uniform -> sample count).
    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
        warnings.warn(
            "clusterless GMM encoding weights sum to 0 (no effective training data "
            "for this environment / encoding group); the fit will be degenerate.",
            UserWarning,
            stacklevel=2,
        )
    occupancy_sample_weight = _gmm_sample_weight(weights, weights_was_none)
    occupancy_n_samples = (
        position.shape[0]
        if occupancy_sample_weight is None
        else int(np.count_nonzero(occupancy_sample_weight > 0.0))
    )
    if occupancy_n_samples < 1:
        raise ValidationError(
            "clusterless GMM has no position samples to fit the occupancy density",
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
            UserWarning,
            stacklevel=2,
        )

    # Interior bins (cached)
    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        if environment.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ is required when is_track_interior_ is None"
            )
        is_track_interior = jnp.ones(len(environment.place_bin_centers_), dtype=bool)

    # If environment has a graph and positions are 2D+, linearize to 1D for occupancy/GPI
    if environment.track_graph is not None and position.shape[1] > 1:
        position1D = get_linearized_position(
            np.asarray(position),
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        pos_for_occ = _as_jnp(position1D)
    else:
        pos_for_occ = position

    is_track_interior = environment.is_track_interior_.ravel()
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
    mean_rates: list[float] = []
    effective_gpi_components: list[int] = []
    effective_joint_components: list[int] = []
    mark_dimensions: list[int] = []

    occupancy = jnp.exp(log_occupancy)
    summed_ground_process_intensity = jnp.zeros_like(occupancy)

    # Fit per-electrode models
    for electrode, (elect_feats, elect_times) in enumerate(
        tqdm(
            zip(spike_waveform_features, spike_times, strict=True),
            desc="Encoding models (GMM)",
            unit="electrode",
            disable=disable_progress_bar,
        )
    ):
        elect_times, elect_feats = _validate_spike_feature_pair(
            elect_times, elect_feats, electrode
        )
        mark_dimensions.append(elect_feats.shape[1])
        # Clip to encoding window
        in_bounds = np.logical_and(
            elect_times >= position_time[0], elect_times <= position_time[-1]
        )
        elect_times = elect_times[in_bounds]
        elect_feats = _as_jnp(elect_feats[in_bounds])
        # Weight each encoding spike by the posterior weight at its spike time.
        elect_weights = interpolate_weights_at_spike_times(
            elect_times, position_time, weights
        )

        # An electrode with no effective encoding spikes has no data to fit, so
        # its rate is zero. This covers two cases: the electrode had no spikes in
        # the encoding window, or every supplied weight for it is zero. Fitting a
        # GMM on zero (or fully de-weighted) spikes would both fail and, for the
        # de-weighted case, leak the excluded spatial pattern into decoding via
        # the joint density. Mark it zero-rate (mean rate 0, None GPI/joint
        # sentinels) and add no ground-process intensity (the rate is zero).
        # Predict floors each of its observed decode spikes to LOG_EPS -- the
        # marked-point-process likelihood of a spike under a zero-rate model --
        # rather than skipping the electrode (which would drop that negative
        # evidence), matching the KDE path. An electrode that has spikes and no
        # supplied weights still takes the normal unweighted fit below.
        effective_spike_count = int(np.count_nonzero(elect_weights > 0.0))
        if effective_spike_count == 0:
            warnings.warn(
                f"Clusterless GMM: electrode {electrode} has no effective encoding "
                "spikes (empty or zero total weight); it is treated as "
                "zero-rate -- its observed decode spikes are floored to LOG_EPS "
                "rather than fit on the de-weighted spikes.",
                UserWarning,
                stacklevel=2,
            )
            mean_rates.append(0.0)  # zero rate, not the EPS floor below
            gpi_models.append(None)
            joint_models.append(None)
            effective_gpi_components.append(0)
            effective_joint_components.append(0)
            continue

        # Weighted mean firing rate: weighted spike count / weighted occupancy time.
        mean_rate = jnp.clip(
            weighted_mean_rate(elect_weights, weight_sum), min=EPS
        )  # avoid 0 rate
        mean_rates.append(mean_rate)

        # Positions at spike times
        enc_pos = get_position_at_time(
            position_time, position, elect_times, environment
        )

        elect_sample_weight = None if weights_was_none else elect_weights
        n_gpi_components = min(gmm_components_gpi, effective_spike_count)
        n_joint_components = min(gmm_components_joint, effective_spike_count)
        effective_gpi_components.append(n_gpi_components)
        effective_joint_components.append(n_joint_components)
        if (
            n_gpi_components < gmm_components_gpi
            or n_joint_components < gmm_components_joint
        ):
            warnings.warn(
                f"Clusterless GMM: electrode {electrode} has "
                f"{effective_spike_count} effective encoding spikes; reduced "
                f"GPI components {gmm_components_gpi}->{n_gpi_components} and "
                f"joint components {gmm_components_joint}->{n_joint_components}.",
                UserWarning,
                stacklevel=2,
            )

        # GPI GMM (position only)
        gpi_gmm = _fit_gmm_density(
            X=enc_pos,
            weights=elect_sample_weight,
            n_components=n_gpi_components,
            random_state=gmm_random_state,
            covariance_type=gmm_covariance_type_gpi,
            reg_covar=gmm_reg_covar,
            max_iter=gmm_max_iter,
            tol=gmm_tol,
        )
        gpi_models.append(gpi_gmm)

        # Joint GMM over [position, waveform]
        joint_samples = jnp.concatenate([enc_pos, elect_feats], axis=1)
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

        # Expected-counts term at bins: mean_rate * (gpi / occupancy)
        gpi_density = _gmm_density(gpi_gmm, interior_place_bin_centers)
        summed_ground_process_intensity += mean_rate * jnp.where(
            occupancy > 0.0,
            gpi_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
            EPS,
        )

    # Clip the summed intensity once (not per electrode) so an empty bin gets a
    # single EPS floor rather than accumulating n_electrodes * EPS.
    summed_ground_process_intensity = jnp.clip(
        summed_ground_process_intensity, min=EPS, max=None
    )

    return {
        "environment": environment,
        "occupancy_model": occupancy_model,
        "interior_place_bin_centers": interior_place_bin_centers,
        "log_occupancy": log_occupancy,
        "gpi_models": gpi_models,
        "joint_models": joint_models,
        "mean_rates": jnp.asarray(mean_rates),
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "disable_progress_bar": disable_progress_bar,
        "gmm_requested_components": {
            "occupancy": gmm_components_occupancy,
            "gpi": gmm_components_gpi,
            "joint": gmm_components_joint,
        },
        "gmm_effective_components": {
            "occupancy": effective_occupancy_components,
            "gpi": effective_gpi_components,
            "joint": effective_joint_components,
        },
        "mark_dimensions": mark_dimensions,
    }


# ---------------------------------------------------------------------
# Decoding (non-local + local) — GMM
# ---------------------------------------------------------------------


def predict_clusterless_gmm_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    occupancy_model: GaussianMixtureModel,
    interior_place_bin_centers: jnp.ndarray,
    log_occupancy: jnp.ndarray,
    gpi_models: list[GaussianMixtureModel | None],
    joint_models: list[GaussianMixtureModel | None],
    mean_rates: jnp.ndarray,
    summed_ground_process_intensity: jnp.ndarray,
    is_local: bool = False,
    spike_block_size: int = 1000,
    bin_tile_size: int | None = None,
    disable_progress_bar: bool = False,
    *,
    mark_dimensions: list[int],
    **kwargs,  # Accept and ignore extra kwargs for compatibility with model interface
) -> jnp.ndarray:
    """
    Predict the (non-local or local) log likelihood using the fitted GMM model.

    Parameters
    ----------
    time : jnp.ndarray
        Decoding time bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample for the decoding period.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples during the decoding period (used for local decoding).
    spike_times : list[jnp.ndarray]
        Decoding spike times per electrode.
    spike_waveform_features : list[jnp.ndarray]
        Decoding spike waveform features per electrode.
    encoding_model : dict
        Output of fit_clusterless_gmm_encoding_model.
    is_local : bool, default=False
        If True, compute local likelihood at the animal's position.
        Else compute non-local likelihood across interior bins.
    spike_block_size : int, default=1000
        Process spikes in blocks of this size to reduce peak memory.
        Reduces memory from O(n_spikes × n_bins) to O(spike_block_size × n_bins).
    bin_tile_size : int | None, default=None
        If provided, tile computation over position bins in chunks of this size.
        Reduces memory from O(spike_block_size × n_bins) to O(spike_block_size × bin_tile_size).
        Useful for very large position grids (> 2000 bins).
    disable_progress_bar : bool, default=False
    mark_dimensions : list[int], keyword-only
        Fitted waveform-feature count per electrode, taken from the encoding
        model. Predict rejects decode spikes whose feature dimension differs, so
        every electrode (including zero-rate ones) must have a recorded count.

    Returns
    -------
    log_likelihood :
        If non-local: jnp.ndarray, shape (n_time, n_bins)
        If local    : jnp.ndarray, shape (n_time, 1)
    """
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        gpi_models=gpi_models,
        joint_models=joint_models,
        mean_rates=mean_rates,
        mark_dimensions=mark_dimensions,
    )

    for electrode, (elect_times, elect_feats, expected_mark_dims) in enumerate(
        zip(spike_times, spike_waveform_features, mark_dimensions, strict=True)
    ):
        _, features = _validate_spike_feature_pair(elect_times, elect_feats, electrode)
        if features.shape[1] != expected_mark_dims:
            raise ValidationError(
                f"waveform feature dimension changed for electrode {electrode}",
                expected=f"{expected_mark_dims} features per spike",
                got=f"{features.shape[1]} features",
                hint="Use the same waveform feature representation at fit and predict.",
            )

    # NOTE: Keep position_time as numpy to avoid float64→float32 precision loss
    position_time = np.asarray(position_time)
    position = _as_jnp(position if position.ndim > 1 else position[:, None])

    if is_local:
        return compute_local_log_likelihood(
            time=time,
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_waveform_features,
            environment=environment,
            occupancy_model=occupancy_model,
            gpi_models=gpi_models,
            joint_models=joint_models,
            mean_rates=mean_rates,
            disable_progress_bar=disable_progress_bar,
        )

    n_time = time.shape[0]
    n_bins = interior_place_bin_centers.shape[0]
    all_bin_ids = jnp.arange(n_bins)

    # Start with the expected-counts (ground process) term, broadcast over time
    # log_likelihood = (
    #     (-summed_ground_process_intensity).reshape(1, -1).repeat(n_time, axis=0)
    # )  # (n_time, n_bins)
    log_likelihood = -1.0 * summed_ground_process_intensity * jnp.ones((n_time, 1))

    # Per-electrode contributions in log-space
    for elect_feats, elect_times, joint_gmm, mean_rate in tqdm(
        zip(
            spike_waveform_features, spike_times, joint_models, mean_rates, strict=True
        ),
        desc="Non-Local Likelihood (GMM, log-space)",
        unit="electrode",
        disable=disable_progress_bar,
    ):
        # A None model marks a zero-rate electrode: it has no effective encoding
        # spikes, so no density was fit. The fitted rate is zero, so its
        # ground-process (integral) term is zero (already omitted at fit) and
        # every observed decoding spike is near-impossible under this model.
        # Floor each such spike's log-intensity to LOG_EPS, added uniformly
        # across position bins -- matching the KDE path and the marked-point-
        # process likelihood. This is *not* the same as skipping the electrode:
        # skipping would drop the negative evidence an observed spike carries
        # (e.g. against a state whose encoding de-weighted this electrode). With
        # no in-window spikes the scatter-add contributes zero.
        if joint_gmm is None:
            in_bounds = np.logical_and(elect_times >= time[0], elect_times <= time[-1])
            seg_ids = get_spike_time_bin_ind(elect_times[in_bounds], time)
            spikes_per_bin = jnp.zeros(n_time).at[seg_ids].add(1.0)  # (n_time,)
            log_likelihood = log_likelihood + LOG_EPS * spikes_per_bin[:, None]
            continue

        # Clip to decoding window
        in_bounds = np.logical_and(elect_times >= time[0], elect_times <= time[-1])
        elect_times = elect_times[in_bounds]
        elect_feats = _as_jnp(elect_feats[in_bounds])

        # Bin spikes
        seg_ids = get_spike_time_bin_ind(elect_times, time)  # (n_spikes,)

        # Process spikes in blocks to reduce peak memory
        # Memory: O(spike_block_size × n_bins) instead of O(n_spikes × n_bins)
        n_spikes = elect_feats.shape[0]

        # Precompute log(mean_rate) outside loop
        log_mean_rate = safe_log(mean_rate, eps=EPS)

        # Process spikes in blocks
        for spike_start in range(0, n_spikes, spike_block_size):
            spike_end = min(spike_start + spike_block_size, n_spikes)
            block_feats = elect_feats[spike_start:spike_end]
            block_seg_ids = seg_ids[spike_start:spike_end]
            block_size = block_feats.shape[0]

            if bin_tile_size is None or bin_tile_size >= n_bins:
                # No bin tiling: process all bins at once (default)
                tiled_bins = jnp.tile(interior_place_bin_centers, (block_size, 1))
                repeated_feats = jnp.repeat(block_feats, n_bins, axis=0)
                eval_points = jnp.concatenate([tiled_bins, repeated_feats], axis=1)

                # GMM evaluation (not JIT-able)
                joint_logp_flat = _gmm_logp(joint_gmm, eval_points)
                joint_logp_block = jnp.clip(
                    joint_logp_flat.reshape(block_size, n_bins), min=LOG_EPS
                )

                # Scatter only the block's observed time rows. A segment_sum with
                # num_segments=n_time would allocate an output-sized temporary for
                # every block, which is prohibitive for hour-long recordings.
                log_likelihood = _accumulate_log_likelihood_block(
                    log_likelihood,
                    joint_logp_block,
                    block_seg_ids,
                    all_bin_ids,
                    log_mean_rate,
                    log_occupancy,
                )
            else:
                # Bin tiling: accumulate per-tile directly (no full block×bins array)
                # Memory: O(block_size × tile_size) instead of O(block_size × n_bins)
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

                    # GMM evaluation (not JIT-able)
                    joint_logp_tile = jnp.clip(
                        _gmm_logp(joint_gmm, eval_points_tile).reshape(
                            block_size, n_tile
                        ),
                        min=LOG_EPS,
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
                    )

    return log_likelihood


def compute_local_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    occupancy_model: GaussianMixtureModel,
    gpi_models: list[GaussianMixtureModel | None],
    joint_models: list[GaussianMixtureModel | None],
    mean_rates: jnp.ndarray,
    disable_progress_bar: bool = False,
) -> jnp.ndarray:
    """Local log-likelihood at the animal's interpolated position.

    Computes the likelihood of observing spikes at the animal's true position
    at each time bin, using the fitted GMM encoding model.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time + 1,)
        Time bin edges for decoding.
    position_time : jnp.ndarray, shape (n_time_position,)
        Timestamps for position samples.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples during decoding period.
    spike_times : list[jnp.ndarray]
        Spike times per electrode during decoding.
    spike_waveform_features : list[jnp.ndarray]
        Spike waveform features per electrode during decoding.
    encoding_model : dict
        Fitted encoding model containing GMM components.
    disable_progress_bar : bool, optional
        Turn off progress bar display, by default False.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
        Log likelihood at the animal's position for each time bin.
    """
    # NOTE: Keep position_time as numpy to avoid float64→float32 precision loss
    position_time = np.asarray(position_time)
    position = _as_jnp(position if position.ndim > 1 else position[:, None])

    n_time = time.shape[0]

    # Interpolate position at bin times (use bin centers)

    interp_pos = get_position_at_time(
        position_time, np.asarray(position), time, environment
    )  # (n_time, pos_dims)

    # Occupancy density and its log at the animal's position
    log_occ_at_pos = _gmm_logp(occupancy_model, interp_pos)  # (n_time,)

    log_likelihood = jnp.zeros((n_time,), dtype=position.dtype)
    summed_expected_counts = jnp.zeros((n_time,), dtype=position.dtype)

    for elect_feats, elect_times, joint_gmm, gpi_gmm, mean_rate in tqdm(
        zip(
            spike_waveform_features,
            spike_times,
            joint_models,
            gpi_models,
            mean_rates,
            strict=True,
        ),
        desc="Local Likelihood (GMM, log-space)",
        unit="electrode",
        disable=disable_progress_bar,
    ):
        # None marks a zero-rate electrode with no effective encoding spikes, so
        # floor each observed decoding spike's
        # log-intensity to LOG_EPS (added to its time bin), matching the KDE path
        # and the marked-point-process likelihood. The integral term is zero. Do
        # not skip -- an observed spike is negative evidence, not "no data".
        if joint_gmm is None:
            in_bounds = jnp.logical_and(elect_times >= time[0], elect_times <= time[-1])
            bounded_times = elect_times[in_bounds]
            if bounded_times.shape[0] > 0:
                seg_ids = get_spike_time_bin_ind(bounded_times, time)
                spikes_per_bin = jnp.zeros(n_time).at[seg_ids].add(1.0)  # (n_time,)
                log_likelihood = log_likelihood + LOG_EPS * spikes_per_bin
            continue

        elect_feats = _as_jnp(elect_feats)

        # Clip to decoding window
        in_bounds = jnp.logical_and(elect_times >= time[0], elect_times <= time[-1])
        elect_times = elect_times[in_bounds]
        elect_feats = elect_feats[in_bounds]

        # Spike contributions at their true positions
        if elect_times.shape[0] > 0:
            pos_at_spike_time = get_position_at_time(
                position_time, np.asarray(position), elect_times, environment
            )  # (n_spikes, pos_dims)
            eval_points = jnp.concatenate(
                [pos_at_spike_time, elect_feats], axis=1
            )  # (n_spikes, P+M)
            joint_logp = jnp.clip(
                _gmm_logp(joint_gmm, eval_points), min=LOG_EPS
            )  # (n_spikes,)
            # log term: log(mean_rate) + log p(pos_t, mark_t) - log occupancy(pos_t)
            log_occ_at_spike_pos = _gmm_logp(
                occupancy_model, pos_at_spike_time
            )  # (n_spikes,)
            terms = (
                safe_log(mean_rate, eps=EPS) + joint_logp - log_occ_at_spike_pos
            )  # (n_spikes,)

            seg_ids = get_spike_time_bin_ind(elect_times, time)  # (n_spikes,)
            log_likelihood = (
                log_likelihood
                + segment_sum(
                    terms[:, None],
                    seg_ids,
                    num_segments=n_time,
                    indices_are_sorted=True,
                ).ravel()
            )

        # Expected counts term at the animal's position (linear space):
        # mean_rate * (gpi / occupancy) evaluated at interpolated positions.
        gpi_logp_at_pos = _gmm_logp(gpi_gmm, interp_pos)
        expected_counts = mean_rate * jnp.exp(
            gpi_logp_at_pos - log_occ_at_pos
        )  # (n_time,)
        summed_expected_counts = summed_expected_counts + expected_counts

    # Subtract the summed ground-process intensity once, floored at EPS to
    # mirror fit_clusterless_gmm_encoding_model's summed_ground_process_intensity
    # (a single EPS floor, not n_electrodes * EPS).
    log_likelihood = log_likelihood - jnp.clip(summed_expected_counts, min=EPS)

    return log_likelihood[:, None]
