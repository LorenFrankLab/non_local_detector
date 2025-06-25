# clusterless_diffusion_kde.py

from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy
import jax.scipy.stats
import numpy as np
from scipy.spatial import KDTree
from tqdm.autonotebook import tqdm

from non_local_detector.diffusion_kernels import compute_diffusion_kernels
from non_local_detector.environment.environment import Environment
from non_local_detector.likelihoods.common import EPS, LOG_EPS, get_position_at_time


# Re-add helper function for Gaussian KDE logpdf calculation
@jax.jit
def gaussian_kde_logpdf(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Computes log PDF of Gaussian KDE.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval, n_features)
    samples : jnp.ndarray, shape (n_samples, n_features)
    std : jnp.ndarray, shape (n_features,)

    Returns
    -------
    log_pdf : jnp.ndarray, shape (n_eval,)
    """
    n_samples = samples.shape[0]
    if n_samples == 0:
        # If there are no encoding samples, the probability is effectively zero everywhere.
        # Return a very small log probability.
        return jnp.full(eval_points.shape[0], LOG_EPS)

    # Calculate log probability for each sample component-wise
    # Shape: (n_samples, n_eval, n_features)
    log_probs_each = jax.scipy.stats.norm.logpdf(
        jnp.expand_dims(eval_points, axis=0),  # (1, n_eval, n_features)
        jnp.expand_dims(samples, axis=1),  # (n_samples, 1, n_features)
        std,  # (n_features,) - broadcasted
    )

    # Sum log probs over features -> log prob for each sample-eval pair
    # Shape: (n_samples, n_eval)
    log_prob_sum_features = jnp.sum(log_probs_each, axis=2)

    # LogSumExp over samples -> log of summed probabilities
    # Shape: (n_eval,)
    log_sum_exp_samples = jax.scipy.special.logsumexp(log_prob_sum_features, axis=0)

    # Normalize by number of samples (log domain)
    log_pdf = log_sum_exp_samples - jnp.log(n_samples)

    # Clip result to avoid -inf if density is exactly zero somewhere
    return jnp.clip(log_pdf, a_min=LOG_EPS)


def fit_clusterless_diffusion_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: List[np.ndarray],
    spike_waveform_features: List[np.ndarray],
    environment: Environment,
    diffusion_bandwidth_sigma: float,
    feature_bandwidth_sigma: Union[float, np.ndarray],
    diffusion_coeff: float = 0.5,
    weights: Optional[np.ndarray] = None,
    # kde_block_size: Optional[int] = None, # No longer needed here
    disable_progress_bar: bool = False,
) -> Dict:
    """Fits a clusterless encoding model using spatial diffusion and stores feature samples.

    Estimates the joint intensity of spikes based on position (smoothed with
    diffusion kernels) and stores waveform features for on-the-fly KDE during prediction.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Timestamps for position samples.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Position samples (typically 2D).
    spike_times : list[np.ndarray]
        List where each element is an array of spike times for an electrode.
    spike_waveform_features : list[np.ndarray]
        List where each element is shape (n_spikes, n_features) for an electrode.
    environment : Environment
        A *fitted* Environment object
    diffusion_bandwidth_sigma : float
        Spatial bandwidth (std dev) for the diffusion kernel.
    feature_bandwidth_sigma : float or np.ndarray, shape (n_features,)
        Bandwidth (std dev) for Gaussian KDE applied to waveform features during prediction.
    diffusion_coeff : float, optional
        Diffusion coefficient for kernel computation. Defaults to 0.5.
    weights : np.ndarray, shape (n_time_position,), optional
        Weights for position samples (e.g., duration). Defaults to uniform time.
    disable_progress_bar : bool, optional
        If True, suppresses progress bars. Defaults to False.

    Returns
    -------
    encoding_model : dict
        Contains fitted components: 'smoothed_spatial_spike_density', 'feature_samples',
        'feature_bandwidth_sigma', 'smoothed_occupancy', 'mean_rates',
        'interior_bin_indices_flat', 'diffusion_bandwidth_sigma', 'environment',
        'summed_spatial_intensity', 'disable_progress_bar'.
    """
    # --- Input Validation and Setup ---
    if not environment._is_fitted:
        raise ValueError("Environment object must be fitted first.")
    if environment.is_1d:
        raise ValueError("Diffusion model requires N-D grid environment.")
    if environment.bin_centers.shape[1] != 2:
        raise ValueError("Diffusion model currently requires a 2D environment.")

    if len(spike_times) != len(spike_waveform_features):
        raise ValueError(
            "Mismatch between spike_times and spike_waveform_features lists."
        )

    # Ensure feature bandwidth is an array
    n_features = spike_waveform_features[0].shape[1]
    if isinstance(feature_bandwidth_sigma, (int, float)):
        feature_bandwidth_sigma = np.full((n_features,), float(feature_bandwidth_sigma))
    # Store as JAX array for use in prediction
    feature_bandwidth_sigma_jax = jnp.asarray(feature_bandwidth_sigma)

    position = position if position.ndim > 1 else np.expand_dims(position, axis=1)
    n_electrodes = len(spike_times)

    # --- Precompute Diffusion Kernels ---
    print("Precomputing diffusion kernels...")
    kernel_matrix = compute_diffusion_kernels(
        track_graph_nd=environment.connectivity,
        bandwidth_sigma=diffusion_bandwidth_sigma,
        diffusion_coeff=diffusion_coeff,
    )
    print(f"Computed {kernel_matrix.shape} kernel matrix.")

    # --- Compute Smoothed Occupancy (Density) ---
    print("Calculating smoothed occupancy...")
    if weights is None:
        weights = np.ones_like(position_time)

    is_nan_pos = np.any(np.isnan(position), axis=1)
    valid_pos_mask = ~is_nan_pos
    valid_positions = position[valid_pos_mask]
    time_weights = weights[valid_pos_mask]

    if valid_positions.shape[0] > 0:
        position_bin_inds_valid = environment.bin_at(valid_positions)
    else:
        position_bin_inds_valid = np.array([], dtype=int)

    full_occupancy_hist = np.zeros((environment.n_bins,))
    if position_bin_inds_valid.size > 0:
        np.add.at(full_occupancy_hist, position_bin_inds_valid, time_weights)

    interior_occupancy_hist = jnp.asarray(full_occupancy_hist)
    smoothed_interior_occupancy = kernel_matrix @ interior_occupancy_hist
    total_interior_time = smoothed_interior_occupancy.sum()

    if total_interior_time > EPS:
        smoothed_interior_occupancy_density = (
            smoothed_interior_occupancy / total_interior_time
        )
    else:
        print("Warning: Total interior occupancy time is close to zero.")
        smoothed_interior_occupancy_density = jnp.zeros_like(
            smoothed_interior_occupancy
        )

    smoothed_occupancy_full = smoothed_interior_occupancy_density

    # --- Compute Smoothed Spatial Spike Density and Store Feature Samples (Per Electrode) ---
    smoothed_spatial_spike_densities = []
    encoding_feature_samples = []  # Store raw feature samples
    mean_rates = []
    total_encoding_time = weights.sum()  # Total time from weights

    n_total_bins = environment.n_bins

    for elec_id, (elec_spike_times, elec_features) in enumerate(
        tqdm(
            zip(spike_times, spike_waveform_features),
            total=n_electrodes,
            unit="electrode",
            desc="Encoding models (Clusterless Diffusion)",
            disable=disable_progress_bar,
        )
    ):
        # Filter spikes outside position time range
        is_valid_time = np.logical_and(
            elec_spike_times >= position_time[0],
            elec_spike_times <= position_time[-1],
        )
        elec_spike_times = elec_spike_times[is_valid_time]
        elec_features = elec_features[is_valid_time]
        n_spikes = len(elec_spike_times)

        # Calculate mean rate
        mean_rates.append(
            n_spikes / total_encoding_time if total_encoding_time > 0 else 0.0
        )

        # Store features for KDE prediction later
        encoding_feature_samples.append(jnp.asarray(elec_features))

        # Calculate spatial density (same as before)
        if n_spikes > 0:
            pos_at_spike = get_position_at_time(
                position_time, position, elec_spike_times, environment
            )
            is_nan_spike_pos = np.any(np.isnan(pos_at_spike), axis=1)
            valid_spike_pos_mask = ~is_nan_spike_pos
            valid_pos_at_spike = pos_at_spike[valid_spike_pos_mask]

            if valid_pos_at_spike.shape[0] > 0:
                spike_bin_inds_valid = environment.bin_at(valid_pos_at_spike)
            else:
                spike_bin_inds_valid = np.array([], dtype=int)

            full_spike_hist = np.zeros(n_total_bins)
            if spike_bin_inds_valid.size > 0:
                np.add.at(full_spike_hist, spike_bin_inds_valid, 1)

            interior_spike_hist = jnp.asarray(full_spike_hist)
            smoothed_interior_spike_counts = kernel_matrix @ interior_spike_hist
            total_interior_spikes = smoothed_interior_spike_counts.sum()
            if total_interior_spikes > EPS:
                smoothed_interior_spike_density = (
                    smoothed_interior_spike_counts / total_interior_spikes
                )
            else:
                smoothed_interior_spike_density = jnp.zeros_like(
                    smoothed_interior_spike_counts
                )

            smoothed_spatial_density_full = smoothed_interior_spike_density
        else:
            smoothed_spatial_density_full = jnp.zeros(n_total_bins)

        smoothed_spatial_spike_densities.append(smoothed_spatial_density_full)

    # --- Calculate Summed Spatial Intensity (for prediction integral term) ---
    # (Code remains the same)
    summed_spatial_intensity = jnp.zeros_like(smoothed_occupancy_full)
    for mean_rate, spatial_density in zip(mean_rates, smoothed_spatial_spike_densities):
        spatial_intensity = mean_rate * jnp.where(
            smoothed_occupancy_full > EPS,
            spatial_density / smoothed_occupancy_full,
            0.0,
        )
        summed_spatial_intensity += jnp.clip(spatial_intensity, a_min=0.0)

    # --- Finalize and Return ---
    return {
        "smoothed_spatial_spike_density": jnp.stack(
            smoothed_spatial_spike_densities, axis=0
        ),
        "feature_samples": encoding_feature_samples,  # Store raw samples
        "feature_bandwidth_sigma": feature_bandwidth_sigma_jax,  # Store bandwidth
        # Removed 'feature_kde_models'
        "smoothed_occupancy": smoothed_occupancy_full,
        "mean_rates": mean_rates,
        "diffusion_bandwidth_sigma": diffusion_bandwidth_sigma,
        "environment": environment,
        "summed_spatial_intensity": summed_spatial_intensity,
        "disable_progress_bar": disable_progress_bar,
        # Removed 'kde_block_size' as it's not relevant here
    }


def predict_clusterless_diffusion_log_likelihood(
    time: np.ndarray,
    spike_times: List[np.ndarray],
    spike_waveform_features: List[np.ndarray],
    encoding_model: Dict,
    is_local: bool = False,
    position_time: Optional[np.ndarray] = None,  # Needed if is_local=True
    position: Optional[np.ndarray] = None,  # Needed if is_local=True
) -> jnp.ndarray:
    """Predict the log likelihood for clusterless spikes using diffusion models.

    Calculates the log likelihood based on the fitted spatial diffusion model
    and on-the-fly feature KDE calculation.

    Parameters
    ----------
    time : np.ndarray, shape (n_time + 1,)
        Edges of the decoding time bins.
    spike_times : list[np.ndarray]
        Spike times for each electrode during decoding.
    spike_waveform_features : list[np.ndarray]
        Waveform features for each electrode during decoding.
    encoding_model : dict
        The fitted model dictionary from `fit_clusterless_diffusion_encoding_model`.
        Must contain 'feature_samples' and 'feature_bandwidth_sigma'.
    is_local : bool, optional
        If True, compute log likelihood at the animal's position. Requires
        `position_time` and `position`. Uses nearest interior bin. Defaults to False.
    position_time : np.ndarray, shape (n_pos_time,), optional
        Timestamps for position data, required if `is_local` is True.
    position : np.ndarray, shape (n_pos_time, n_dims), optional
        Position data, required if `is_local` is True.

    Returns
    -------
    log_likelihood : jnp.ndarray
        Shape (n_time, n_interior_bins) if `is_local` is False.
        Shape (n_time, 1) if `is_local` is True.
    """
    # --- Extract components from encoding_model ---
    smoothed_spatial_spike_density = encoding_model["smoothed_spatial_spike_density"]
    feature_samples = encoding_model["feature_samples"]  # Get list of feature samples
    feature_bw = encoding_model["feature_bandwidth_sigma"]  # Get feature bandwidth
    smoothed_occupancy = encoding_model["smoothed_occupancy"]
    mean_rates = encoding_model["mean_rates"]
    interior_indices = encoding_model["interior_bin_indices_flat"]
    summed_spatial_intensity = encoding_model["summed_spatial_intensity"]
    environment = encoding_model["environment"]
    disable_progress_bar = encoding_model["disable_progress_bar"]

    n_time = len(time)
    if n_time <= 0:
        raise ValueError("`time` array must contain at least two elements.")
    n_electrodes = len(spike_times)
    n_total_bins = smoothed_occupancy.shape[0]
    n_interior_bins = interior_indices.shape[0]
    log_likelihood = None  # Initialize

    # Precompute spatial intensity part (same as before)
    clipped_occupancy = jnp.clip(smoothed_occupancy, a_min=EPS)
    spatial_log_intensity_part = (
        jnp.log(jnp.asarray(mean_rates)[:, None])
        + jnp.log(smoothed_spatial_spike_density + EPS)
        - jnp.log(clipped_occupancy[None, :] + EPS)
    )
    spatial_log_intensity_part = jnp.where(
        smoothed_occupancy[None, :] > EPS, spatial_log_intensity_part, LOG_EPS
    )

    if is_local:
        # --- Local Log Likelihood ---
        if position_time is None or position is None:
            raise ValueError(
                "'position_time' and 'position' are required for local=True."
            )
        if not environment._is_fitted:
            raise ValueError("Environment object must be fitted.")

        print("Calculating local clusterless likelihood (nearest interior bin)...")
        time_bin_centers = time[:-1] + np.diff(time) / 2
        interpolated_position = get_position_at_time(
            position_time, position, time_bin_centers, environment
        )
        is_nan_interp_pos = np.any(np.isnan(interpolated_position), axis=1)
        valid_interp_pos = interpolated_position[~is_nan_interp_pos]

        nearest_interior_indices = np.full(n_time, -1, dtype=int)
        nearest_flat_bin_indices = np.full(n_time, -1, dtype=int)

        if valid_interp_pos.shape[0] > 0:
            interior_centers = environment.bin_centers[interior_indices]
            if interior_centers.shape[0] == 0:
                raise ValueError("No interior bin centers found.")
            tree = KDTree(interior_centers)
            _, nn_indices_for_valid = tree.query(valid_interp_pos)
            nearest_interior_indices[~is_nan_interp_pos] = nn_indices_for_valid
            valid_flat_indices = interior_indices[jnp.asarray(nn_indices_for_valid)]
            nearest_flat_bin_indices[~is_nan_interp_pos] = valid_flat_indices

        log_likelihood_local = jnp.zeros((n_time,))

        valid_time_bins_mask_local = nearest_flat_bin_indices != -1
        valid_flat_indices_local = nearest_flat_bin_indices[valid_time_bins_mask_local]
        if valid_flat_indices_local.size > 0:
            integral_term_values = -summed_spatial_intensity[
                jnp.asarray(valid_flat_indices_local)
            ]
            log_likelihood_local = log_likelihood_local.at[
                valid_time_bins_mask_local
            ].add(integral_term_values)

        for elec_id, (elec_spikes, elec_features) in enumerate(
            tqdm(
                zip(spike_times, spike_waveform_features),
                total=n_electrodes,
                unit="electrode",
                desc="Local Clusterless Likelihood",
                disable=disable_progress_bar,
            )
        ):
            valid_time_mask = (elec_spikes >= time[0]) & (elec_spikes < time[-1])
            spikes_in_window = elec_spikes[valid_time_mask]
            features_in_window = jnp.asarray(elec_features[valid_time_mask])

            if spikes_in_window.size == 0:
                continue

            spike_bin_inds = get_spike_time_bin_ind(spikes_in_window, time)
            flat_bin_for_spikes = nearest_flat_bin_indices[spike_bin_inds]
            valid_spike_bin_mask = flat_bin_for_spikes != -1

            if not np.any(valid_spike_bin_mask):
                continue

            valid_spike_bin_inds = spike_bin_inds[valid_spike_bin_mask]
            valid_features = features_in_window[valid_spike_bin_mask]
            valid_flat_bins = flat_bin_for_spikes[valid_spike_bin_mask]

            # Calculate feature log pdf using helper function
            if valid_features.shape[0] > 0:
                feature_log_pdf = gaussian_kde_logpdf(
                    valid_features, feature_samples[elec_id], feature_bw
                )
            else:
                feature_log_pdf = jnp.full(valid_features.shape[0], LOG_EPS)

            spatial_part = spatial_log_intensity_part[elec_id][
                jnp.asarray(valid_flat_bins)
            ]
            spike_log_likelihood = spatial_part + feature_log_pdf
            spike_log_likelihood = jnp.clip(spike_log_likelihood, a_min=LOG_EPS)

            # Use segment_sum for potentially better performance/readability even in local case
            # log_likelihood_local = log_likelihood_local.at[valid_spike_bin_inds].add(spike_log_likelihood)
            # Create temporary array for segment sum
            spike_contributions = jnp.zeros_like(log_likelihood_local)
            spike_contributions = spike_contributions.at[valid_spike_bin_inds].set(
                spike_log_likelihood
            )
            log_likelihood_local += (
                spike_contributions  # Add contributions for this electrode
            )

        log_likelihood = log_likelihood_local[:, jnp.newaxis]

    else:
        # --- Non-Local Log Likelihood ---
        print("Calculating non-local clusterless likelihood...")
        if n_interior_bins == 0:
            print("Warning: No interior bins found. Returning empty likelihood array.")
            return jnp.zeros((n_time, 0))

        integral_term = -summed_spatial_intensity[interior_indices]
        log_likelihood_nonlocal = jnp.tile(integral_term[jnp.newaxis, :], (n_time, 1))

        spatial_log_intensity_interior = spatial_log_intensity_part[:, interior_indices]

        for elec_id, (elec_spikes, elec_features) in enumerate(
            tqdm(
                zip(spike_times, spike_waveform_features),
                total=n_electrodes,
                unit="electrode",
                desc="Non-Local Clusterless Likelihood",
                disable=disable_progress_bar,
            )
        ):
            valid_time_mask = (elec_spikes >= time[0]) & (elec_spikes < time[-1])
            spikes_in_window = elec_spikes[valid_time_mask]
            features_in_window = jnp.asarray(elec_features[valid_time_mask])

            if spikes_in_window.size == 0:
                continue

            spike_bin_inds = get_spike_time_bin_ind(spikes_in_window, time)

            # Calculate feature log pdf using helper function
            if features_in_window.shape[0] > 0:
                feature_log_pdf = gaussian_kde_logpdf(
                    features_in_window, feature_samples[elec_id], feature_bw
                )
            else:
                feature_log_pdf = jnp.full(features_in_window.shape[0], LOG_EPS)

            spatial_part = spatial_log_intensity_interior[elec_id][jnp.newaxis, :]
            feature_part = feature_log_pdf[:, jnp.newaxis]
            spike_log_likelihood = spatial_part + feature_part
            spike_log_likelihood = jnp.clip(spike_log_likelihood, a_min=LOG_EPS)

            # Sum spike contributions into the correct time bins using segment_sum
            log_likelihood_nonlocal = log_likelihood_nonlocal + jax.ops.segment_sum(
                spike_log_likelihood,
                spike_bin_inds,
                num_segments=n_time,
                indices_are_sorted=False,
            )

        log_likelihood = log_likelihood_nonlocal

    if log_likelihood is None:
        raise RuntimeError("Log likelihood calculation failed.")

    return log_likelihood
