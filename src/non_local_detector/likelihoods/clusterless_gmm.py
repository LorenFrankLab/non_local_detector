"""
Clusterless decoding using Gaussian Mixture Models (GMM)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import EPS, get_position_at_time, safe_divide
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


def get_spike_time_bin_ind(
    spike_times: jnp.ndarray, time_bin_edges: jnp.ndarray
) -> jnp.ndarray:
    """Map spike times to decoding time-bin indices on device.

    Parameters
    ----------
    spike_times : jnp.ndarray, shape (n_spikes,)
        Array of spike times to map to bins.
    time_bin_edges : jnp.ndarray, shape (n_bins + 1,)
        Edges of time bins defining intervals [t0, t1), ..., [t_{n-1}, tn].

    Returns
    -------
    bin_indices : jnp.ndarray, shape (n_spikes,)
        Index of time bin for each spike (0 to n_bins-1).
    """
    # Right-closed bins [t_i, t_{i+1}), except the last edge which is included
    # Use JAX ops to keep everything on device
    inds = jnp.searchsorted(time_bin_edges, spike_times, side="right") - 1
    last = jnp.isclose(spike_times, time_bin_edges[-1])
    return jnp.where(last, time_bin_edges.shape[0] - 2, inds).astype(jnp.int32)


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
        Sample weights, by default None (uniform weights).
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
    gmm.fit(
        _as_jnp(X), key, sample_weight=None if weights is None else _as_jnp(weights)
    )
    return gmm


# ---------------------------------------------------------------------
# Encoded model container (DEPRECATED: Use dictionary for now)
# ---------------------------------------------------------------------
# NOTE: EncodingModel dataclass is kept for future migration but not currently used.
# All functions now use dictionary-based encoding models for compatibility with
# existing architecture. Eventually, all likelihood models should transition to
# using dataclasses for better type safety and IDE support.


@dataclass
class EncodingModel:
    """
    Container for everything the decoder needs (precomputed & cached).

    DEPRECATED: This dataclass is not currently used. Use dictionary format instead.
    Kept for future migration.

    Attributes
    ----------
    environment : Environment
    occupancy_model : GaussianMixtureModel
        GMM over position for occupancy.
    interior_place_bin_centers : jnp.ndarray, shape (n_bins, n_pos_dims)
        Interior bin centers.
    occupancy_bins : jnp.ndarray, shape (n_bins,)
        Occupancy density at interior bin centers.
    log_occupancy_bins : jnp.ndarray, shape (n_bins,)
        Log occupancy density at interior bin centers.
    gpi_models : list[GaussianMixtureModel]
        Per-electrode GMMs over position (spike ground process intensity).
    joint_models : list[GaussianMixtureModel]
        Per-electrode GMMs over [position, waveform].
    mean_rates : jnp.ndarray, shape (n_electrodes,)
        Mean firing rate per electrode during encoding.
    summed_ground_process_intensity : jnp.ndarray, shape (n_bins,)
        Sum over electrodes of mean_rate * (gpi / occupancy) at interior bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Position timestamps used in encoding (needed for interpolation in local decoding).
    """

    environment: Environment
    occupancy_model: GaussianMixtureModel
    interior_place_bin_centers: jnp.ndarray
    occupancy_bins: jnp.ndarray
    log_occupancy_bins: jnp.ndarray
    gpi_models: list[GaussianMixtureModel]
    joint_models: list[GaussianMixtureModel]
    mean_rates: jnp.ndarray
    summed_ground_process_intensity: jnp.ndarray
    position_time: jnp.ndarray


# ---------------------------------------------------------------------
# Encoding (fit) — GMM
# ---------------------------------------------------------------------


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
    disable_progress_bar: bool = False,
) -> EncodingModel:
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
    weights : Optional[jnp.ndarray], shape (n_time_position,), default=None
        Per-position sample weights for occupancy.
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
    gmm_random_state : Optional[int], default=0
        Random state for reproducibility.
    disable_progress_bar : bool, default=False
        If True, disable progress bar.

    Returns
    -------
    encoding_model : dict
        Dictionary containing the fitted encoding model with keys:
        - environment
        - occupancy_model
        - interior_place_bin_centers
        - occupancy_bins
        - log_occupancy_bins
        - gpi_models
        - joint_models
        - mean_rates
        - summed_ground_process_intensity
        - position_time
        - gmm_components_occupancy
        - gmm_components_gpi
        - gmm_components_joint
        - gmm_covariance_type_occupancy
        - gmm_covariance_type_gpi
        - gmm_covariance_type_joint
        - gmm_random_state
        - disable_progress_bar
    """
    position = _as_jnp(position if position.ndim > 1 else position[:, None])
    position_time = _as_jnp(position_time)

    # Interior bins (cached)
    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        if environment.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ is required when is_track_interior_ is None"
            )
        is_track_interior = jnp.ones(len(environment.place_bin_centers_), dtype=bool)

    # Occupancy weights over trajectory
    if weights is None:
        weights = jnp.ones((position.shape[0],), dtype=position.dtype)
    else:
        weights = _as_jnp(weights)
    total_weight = float(jnp.sum(weights))

    # If environment has a graph and positions are 2D+, linearize to 1D for occupancy/GPI
    if environment.track_graph is not None and position.shape[1] > 1:
        position1D = get_linearized_position(
            np.asarray(position),
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        pos_for_occ = _as_jnp(position1D)

        # CRITICAL: Linearize bin centers too (must match occupancy space)
        raw_bin_centers = environment.place_bin_centers_[is_track_interior]
        lin_bins = get_linearized_position(
            np.asarray(raw_bin_centers),
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        interior_place_bin_centers = _as_jnp(lin_bins)
    else:
        pos_for_occ = position
        interior_place_bin_centers = _as_jnp(
            environment.place_bin_centers_[is_track_interior]
        )

    # Fit occupancy GMM and precompute per-bin terms
    occupancy_model = _fit_gmm_density(
        X=pos_for_occ,
        weights=weights,
        n_components=gmm_components_occupancy,
        random_state=gmm_random_state,
        covariance_type=gmm_covariance_type_occupancy,
    )
    occupancy_bins = _gmm_density(
        occupancy_model, interior_place_bin_centers
    )  # (n_bins,)
    log_occupancy_bins = _gmm_logp(
        occupancy_model, interior_place_bin_centers
    )  # (n_bins,)

    gpi_models: list[GaussianMixtureModel] = []
    joint_models: list[GaussianMixtureModel] = []
    mean_rates: list[float] = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy_bins)

    # Fit per-electrode models
    for elect_feats, elect_times in tqdm(
        zip(spike_waveform_features, spike_times, strict=False),
        desc="Encoding models (GMM)",
        unit="electrode",
        disable=disable_progress_bar,
    ):
        elect_times = _as_jnp(elect_times)
        elect_feats = _as_jnp(elect_feats)

        # Clip to encoding window
        in_bounds = jnp.logical_and(
            elect_times >= position_time[0], elect_times <= position_time[-1]
        )
        elect_times = elect_times[in_bounds]
        elect_feats = elect_feats[in_bounds]

        # Interpolate position weights at spike times
        elect_weights = jnp.interp(elect_times, position_time, weights)

        # Mean rate contribution
        mean_rate = float(jnp.sum(elect_weights) / total_weight)
        mean_rate = jnp.clip(mean_rate, a_min=EPS)  # avoid 0 rate
        mean_rates.append(mean_rate)

        # Positions at spike times
        enc_pos = get_position_at_time(
            position_time, position, elect_times, environment
        )

        # GPI GMM (position only)
        gpi_gmm = _fit_gmm_density(
            X=enc_pos,
            weights=elect_weights,
            n_components=gmm_components_gpi,
            random_state=gmm_random_state,
            covariance_type=gmm_covariance_type_gpi,
        )
        gpi_models.append(gpi_gmm)

        # Joint GMM over [position, waveform]
        joint_samples = jnp.concatenate([enc_pos, elect_feats], axis=1)
        joint_gmm = _fit_gmm_density(
            X=joint_samples,
            weights=elect_weights,
            n_components=gmm_components_joint,
            random_state=gmm_random_state,
            covariance_type=gmm_covariance_type_joint,
        )
        joint_models.append(joint_gmm)

        # Expected-counts term at bins: mean_rate * (gpi / occupancy)
        gpi_bins = _gmm_density(gpi_gmm, interior_place_bin_centers)  # (n_bins,)
        summed_ground_process_intensity = summed_ground_process_intensity + jnp.clip(
            mean_rate * safe_divide(gpi_bins, occupancy_bins), a_min=EPS
        )

    return {
        "environment": environment,
        "occupancy_model": occupancy_model,
        "interior_place_bin_centers": interior_place_bin_centers,
        "occupancy_bins": occupancy_bins,
        "log_occupancy_bins": log_occupancy_bins,
        "gpi_models": gpi_models,
        "joint_models": joint_models,
        "mean_rates": jnp.asarray(mean_rates),
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "position_time": position_time,
        "gmm_components_occupancy": gmm_components_occupancy,
        "gmm_components_gpi": gmm_components_gpi,
        "gmm_components_joint": gmm_components_joint,
        "gmm_covariance_type_occupancy": gmm_covariance_type_occupancy,
        "gmm_covariance_type_gpi": gmm_covariance_type_gpi,
        "gmm_covariance_type_joint": gmm_covariance_type_joint,
        "gmm_random_state": gmm_random_state,
        "disable_progress_bar": disable_progress_bar,
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
    encoding_model: dict,
    is_local: bool = False,
    spike_block_size: int = 1000,
    bin_tile_size: int | None = None,
    disable_progress_bar: bool = False,
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

    Returns
    -------
    log_likelihood :
        If non-local: jnp.ndarray, shape (n_time, n_bins)
        If local    : jnp.ndarray, shape (n_time, 1)
    """
    time = _as_jnp(time)
    position_time = _as_jnp(position_time)
    position = _as_jnp(position if position.ndim > 1 else position[:, None])

    if is_local:
        return compute_local_log_likelihood(
            time=time,
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=spike_waveform_features,
            encoding_model=encoding_model,
            disable_progress_bar=disable_progress_bar,
        )

    bin_centers = encoding_model["interior_place_bin_centers"]
    log_occ_bins = encoding_model["log_occupancy_bins"]  # log density
    mean_rates = encoding_model["mean_rates"]
    joint_models = encoding_model["joint_models"]
    summed_ground = encoding_model["summed_ground_process_intensity"]

    n_time = time.shape[0]
    n_bins = bin_centers.shape[0]

    # Start with the expected-counts (ground process) term, broadcast over time
    log_likelihood = (
        (-summed_ground).reshape(1, -1).repeat(n_time, axis=0)
    )  # (n_time, n_bins)

    # Per-electrode contributions in log-space
    for elect_feats, elect_times, joint_gmm, mean_rate in tqdm(
        zip(
            spike_waveform_features, spike_times, joint_models, mean_rates, strict=False
        ),
        desc="Non-Local Likelihood (GMM, log-space)",
        unit="electrode",
        disable=disable_progress_bar,
    ):
        elect_times = _as_jnp(elect_times)
        elect_feats = _as_jnp(elect_feats)

        # Clip to decoding window
        in_bounds = jnp.logical_and(elect_times >= time[0], elect_times <= time[-1])
        elect_times = elect_times[in_bounds]
        elect_feats = elect_feats[in_bounds]

        if elect_times.shape[0] == 0:
            continue

        # Bin spikes
        seg_ids = get_spike_time_bin_ind(elect_times, time)  # (n_spikes,)

        # Process spikes in blocks to reduce peak memory
        # Memory: O(spike_block_size × n_bins) instead of O(n_spikes × n_bins)
        n_spikes = elect_feats.shape[0]

        # Precompute log(mean_rate) outside loop
        log_mean_rate = jnp.log(mean_rate)

        # Larger JIT boundary: include log contribution computation and accumulation
        # This reduces dispatch overhead and helps XLA fuse operations
        def _update_block_all_bins(
            log_lik_array,
            joint_logp_block,
            block_seg_ids,
            log_rate,
            log_occ,
            num_segments,
        ):
            """Update likelihood with block contributions (all bins, no tiling)."""
            # Compute log contributions: log(rate) + joint_logp - log_occ
            log_contrib = log_rate + joint_logp_block - log_occ

            # Accumulate per time bin
            block_contrib = segment_sum(
                log_contrib,
                block_seg_ids,
                num_segments=num_segments,
                indices_are_sorted=True,
            )
            return log_lik_array + block_contrib

        update_block_all_bins = jax.jit(
            _update_block_all_bins,
            donate_argnums=(0,),
            static_argnames=("num_segments",),
        )

        # For tiled path: accumulate per-tile contributions directly
        # This avoids materializing a full (block_size × n_bins) array
        def _update_block_one_tile(
            log_lik_array,
            joint_logp_tile,
            block_seg_ids,
            log_rate,
            log_occ_tile,
            num_segments,
        ):
            """Update likelihood with one tile's contributions."""
            log_contrib_tile = log_rate + joint_logp_tile - log_occ_tile

            block_contrib = segment_sum(
                log_contrib_tile,
                block_seg_ids,
                num_segments=num_segments,
                indices_are_sorted=True,
            )
            return log_lik_array + block_contrib

        update_block_one_tile = jax.jit(
            _update_block_one_tile,
            donate_argnums=(0,),
            static_argnames=("num_segments",),
        )

        # Process spikes in blocks
        for spike_start in range(0, n_spikes, spike_block_size):
            spike_end = min(spike_start + spike_block_size, n_spikes)
            block_feats = elect_feats[spike_start:spike_end]
            block_seg_ids = seg_ids[spike_start:spike_end]
            block_size = block_feats.shape[0]

            if bin_tile_size is None or bin_tile_size >= n_bins:
                # No bin tiling: process all bins at once (default)
                tiled_bins = jnp.tile(bin_centers, (block_size, 1))
                repeated_feats = jnp.repeat(block_feats, n_bins, axis=0)
                eval_points = jnp.concatenate([tiled_bins, repeated_feats], axis=1)

                # GMM evaluation (not JIT-able)
                joint_logp_flat = _gmm_logp(joint_gmm, eval_points)
                joint_logp_block = joint_logp_flat.reshape(block_size, n_bins)

                # JIT-compiled update with larger boundary (better fusion)
                log_likelihood = update_block_all_bins(
                    log_likelihood,
                    joint_logp_block,
                    block_seg_ids,
                    log_mean_rate,
                    log_occ_bins,
                    n_time,
                )
            else:
                # Bin tiling: accumulate per-tile directly (no full block×bins array)
                # Memory: O(block_size × tile_size) instead of O(block_size × n_bins)
                for bin_start in range(0, n_bins, bin_tile_size):
                    bin_end = min(bin_start + bin_tile_size, n_bins)
                    n_tile = bin_end - bin_start

                    # Build eval points for this tile
                    tiled_bins_tile = jnp.tile(
                        bin_centers[bin_start:bin_end], (block_size, 1)
                    )
                    repeated_feats_tile = jnp.repeat(block_feats, n_tile, axis=0)
                    eval_points_tile = jnp.concatenate(
                        [tiled_bins_tile, repeated_feats_tile], axis=1
                    )

                    # GMM evaluation (not JIT-able)
                    joint_logp_tile = _gmm_logp(joint_gmm, eval_points_tile).reshape(
                        block_size, n_tile
                    )

                    # Accumulate this tile directly (no intermediate full array)
                    log_likelihood = update_block_one_tile(
                        log_likelihood,
                        joint_logp_tile,
                        block_seg_ids,
                        log_mean_rate,
                        log_occ_bins[bin_start:bin_end],
                        n_time,
                    )

    return log_likelihood


def compute_local_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    encoding_model: dict,
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
    time = _as_jnp(time)
    position_time = _as_jnp(position_time)
    position = _as_jnp(position if position.ndim > 1 else position[:, None])

    env = encoding_model["environment"]
    occupancy_model = encoding_model["occupancy_model"]
    gpi_models = encoding_model["gpi_models"]
    joint_models = encoding_model["joint_models"]
    mean_rates = encoding_model["mean_rates"]

    n_time = time.shape[0] - 1

    # Interpolate position at bin times (use bin centers)
    # We'll take the midpoints as "time of interest" for local evaluation
    t_centers = 0.5 * (time[:-1] + time[1:])
    interp_pos = get_position_at_time(
        position_time, position, t_centers, env
    )  # (n_time, pos_dims)

    # Occupancy density and its log at the animal's position
    log_occ_at_pos = _gmm_logp(occupancy_model, interp_pos)  # (n_time,)

    log_likelihood = jnp.zeros((n_time,), dtype=position.dtype)

    for elect_feats, elect_times, joint_gmm, gpi_gmm, mean_rate in tqdm(
        zip(
            spike_waveform_features,
            spike_times,
            joint_models,
            gpi_models,
            mean_rates,
            strict=False,
        ),
        desc="Local Likelihood (GMM, log-space)",
        unit="electrode",
        disable=disable_progress_bar,
    ):
        elect_times = _as_jnp(elect_times)
        elect_feats = _as_jnp(elect_feats)

        # Clip to decoding window
        in_bounds = jnp.logical_and(elect_times >= time[0], elect_times <= time[-1])
        elect_times = elect_times[in_bounds]
        elect_feats = elect_feats[in_bounds]

        # Spike contributions at their true positions
        if elect_times.shape[0] > 0:
            pos_at_spike_time = get_position_at_time(
                position_time, position, elect_times, env
            )  # (n_spikes, pos_dims)
            eval_points = jnp.concatenate(
                [pos_at_spike_time, elect_feats], axis=1
            )  # (n_spikes, P+M)
            joint_logp = _gmm_logp(joint_gmm, eval_points)  # (n_spikes,)
            # log term: log(mean_rate) + log p(pos_t, mark_t) - log occupancy(pos_t)
            log_occ_at_spike_pos = _gmm_logp(
                occupancy_model, pos_at_spike_time
            )  # (n_spikes,)
            terms = (
                jnp.log(mean_rate) + joint_logp - log_occ_at_spike_pos
            )  # (n_spikes,)

            seg_ids = get_spike_time_bin_ind(elect_times, time)  # (n_spikes,)
            log_likelihood = (
                log_likelihood
                + segment_sum(
                    terms[:, None],
                    seg_ids,
                    n_time,
                    indices_are_sorted=True,
                    num_segments=n_time,
                ).ravel()
            )

        # Subtract expected counts term at the animal's position (linear space)
        # mean_rate * (gpi / occupancy) evaluated at interpolated positions
        gpi_logp_at_pos = _gmm_logp(gpi_gmm, interp_pos)
        expected_counts = mean_rate * jnp.exp(
            gpi_logp_at_pos - log_occ_at_pos
        )  # (n_time,)
        log_likelihood = log_likelihood - expected_counts

    return log_likelihood[:, None]
