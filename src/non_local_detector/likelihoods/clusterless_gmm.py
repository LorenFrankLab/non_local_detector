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
    return x if isinstance(x, jnp.ndarray) else jnp.asarray(x)


def get_spike_time_bin_ind(
    spike_times: jnp.ndarray, time_bin_edges: jnp.ndarray
) -> jnp.ndarray:
    """
    Map spike times to decoding time-bin indices on device.

    Parameters
    ----------
    spike_times : jnp.ndarray, shape (n_spikes,)
    time_bin_edges : jnp.ndarray, shape (n_bins + 1,)

    Returns
    -------
    bin_indices : jnp.ndarray, shape (n_spikes,)
    """
    # Right-closed bins [t_i, t_{i+1}), except the last edge which is included
    inds = np.searchsorted(time_bin_edges, spike_times, side="right") - 1
    last = np.isclose(spike_times, time_bin_edges[-1])
    return jnp.asarray(
        np.where(last, time_bin_edges.shape[0] - 2, inds), dtype=jnp.int32
    )


def _gmm_logp(gmm: GaussianMixtureModel, X: jnp.ndarray) -> jnp.ndarray:
    """Log density under a fitted GMM."""
    return gmm.score_samples(X)


def _gmm_density(gmm: GaussianMixtureModel, X: jnp.ndarray) -> jnp.ndarray:
    """Density under a fitted GMM."""
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
    """Fit a GMM density model on samples X (optionally weighted)."""
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
# Encoded model container
# ---------------------------------------------------------------------


@dataclass
class EncodingModel:
    """
    Container for everything the decoder needs (precomputed & cached).

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
    weights: jnp.ndarray | None = None,
    *,
    gmm_components_occupancy: int = 32,
    gmm_components_gpi: int = 32,
    gmm_components_joint: int = 64,
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
    gmm_components_gpi : int, default=32
    gmm_components_joint : int, default=64
    gmm_random_state : Optional[int], default=0
    disable_progress_bar : bool, default=False

    Returns
    -------
    model : EncodingModel
    """
    position = _as_jnp(position if position.ndim > 1 else position[:, None])
    position_time = _as_jnp(position_time)

    # Interior bins (cached)
    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        is_track_interior = jnp.ones(len(environment.place_bin_centers_), dtype=bool)
    interior_place_bin_centers = _as_jnp(
        environment.place_bin_centers_[is_track_interior]
    )

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
    else:
        pos_for_occ = position

    # Fit occupancy GMM and precompute per-bin terms
    occupancy_model = _fit_gmm_density(
        X=pos_for_occ,
        weights=weights,
        n_components=gmm_components_occupancy,
        random_state=gmm_random_state,
        covariance_type="full",
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
            covariance_type="full",
        )
        gpi_models.append(gpi_gmm)

        # Joint GMM over [position, waveform]
        joint_samples = jnp.concatenate([enc_pos, elect_feats], axis=1)
        joint_gmm = _fit_gmm_density(
            X=joint_samples,
            weights=elect_weights,
            n_components=gmm_components_joint,
            random_state=gmm_random_state,
            covariance_type="full",
        )
        joint_models.append(joint_gmm)

        # Expected-counts term at bins: mean_rate * (gpi / occupancy)
        gpi_bins = _gmm_density(gpi_gmm, interior_place_bin_centers)  # (n_bins,)
        summed_ground_process_intensity = summed_ground_process_intensity + jnp.clip(
            mean_rate * safe_divide(gpi_bins, occupancy_bins), a_min=EPS
        )

    return EncodingModel(
        environment=environment,
        occupancy_model=occupancy_model,
        interior_place_bin_centers=interior_place_bin_centers,
        occupancy_bins=occupancy_bins,
        log_occupancy_bins=log_occupancy_bins,
        gpi_models=gpi_models,
        joint_models=joint_models,
        mean_rates=jnp.asarray(mean_rates),
        summed_ground_process_intensity=summed_ground_process_intensity,
        position_time=position_time,
    )


# ---------------------------------------------------------------------
# Decoding (non-local + local) — GMM
# ---------------------------------------------------------------------


def predict_clusterless_gmm_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    encoding_model: EncodingModel,
    is_local: bool = False,
    disable_progress_bar: bool = False,
) -> jnp.ndarray:
    """
    Predict the (non-local or local) log likelihood using the fitted GMM model.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time + 1,)
        Decoding time bin edges.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample for the decoding period.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples during the decoding period (used for local decoding).
    spike_times : list[jnp.ndarray]
        Decoding spike times per electrode.
    spike_waveform_features : list[jnp.ndarray]
        Decoding spike waveform features per electrode.
    encoding_model : EncodingModel
        Output of fit_clusterless_gmm_encoding_model.
    is_local : bool, default=False
        If True, compute local likelihood at the animal's position.
        Else compute non-local likelihood across interior bins.
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

    bin_centers = encoding_model.interior_place_bin_centers
    log_occ_bins = encoding_model.log_occupancy_bins  # log density
    mean_rates = encoding_model.mean_rates
    joint_models = encoding_model.joint_models
    summed_ground = encoding_model.summed_ground_process_intensity

    n_time = time.shape[0] - 1
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

        # Joint logp for each mark vs all bins — VMAP for lower memory
        def mark_logp(mark: jnp.ndarray, gmm_model: object) -> jnp.ndarray:
            # Build [bin_centers, mark] without forming (B*n_bins, ...) at once
            tiled_mark = jnp.repeat(mark[None, :], n_bins, axis=0)  # (n_bins, M)
            eval_points = jnp.concatenate(
                [bin_centers, tiled_mark], axis=1
            )  # (n_bins, P+M)
            return _gmm_logp(gmm_model, eval_points)  # (n_bins,)

        joint_logp = jax.vmap(mark_logp, in_axes=(0, None))(
            elect_feats, joint_gmm
        )  # (n_spikes, n_bins)

        # log contribution: log(mean_rate) + log p(pos, mark) - log occupancy(pos)
        log_contrib = (
            jnp.log(mean_rate) + joint_logp - log_occ_bins
        )  # (n_spikes, n_bins)

        # Sum per time bin
        log_likelihood = log_likelihood + segment_sum(
            log_contrib,
            seg_ids,
            n_time,
            indices_are_sorted=True,
            num_segments=n_time,
        )

    return log_likelihood


def compute_local_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    encoding_model: EncodingModel,
    disable_progress_bar: bool = False,
) -> jnp.ndarray:
    """
    Local log-likelihood at the animal's interpolated position.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
    """
    time = _as_jnp(time)
    position_time = _as_jnp(position_time)
    position = _as_jnp(position if position.ndim > 1 else position[:, None])

    env = encoding_model.environment
    occupancy_model = encoding_model.occupancy_model
    gpi_models = encoding_model.gpi_models
    joint_models = encoding_model.joint_models
    mean_rates = encoding_model.mean_rates

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
