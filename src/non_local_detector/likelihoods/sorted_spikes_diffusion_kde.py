from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
from scipy.spatial import KDTree
from tqdm.autonotebook import tqdm

from non_local_detector.diffusion_kernels import compute_diffusion_kernels
from non_local_detector.environment.environment import Environment
from non_local_detector.likelihoods.common import (
    EPS,
    get_position_at_time,
    get_spikecount_per_time_bin,
)


def fit_sorted_spikes_diffusion_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    environment: Environment,
    weights: Optional[jnp.ndarray] = None,
    sampling_frequency: int = 500,
    position_std: float = np.sqrt(12.5),
    block_size: int = 100,
    disable_progress_bar: bool = False,
) -> Dict:
    """Fits an encoding model for sorted spikes using diffusion kernel smoothing.

    Estimates spatial firing rate (place fields) by smoothing occupancy and
    spike histograms using precomputed diffusion kernels based on the
    environment's geometry.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Timestamps for position samples.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Position samples (typically 2D for this diffusion implementation).
    spike_times : list[np.ndarray]
        List where each element is an array of spike times for a single neuron.
    environment : Environment
        A *fitted* Environment object defining the spatial grid, interior,
        and connectivity graph (track_graph_nd_). Must be 2D.
    position_std : float
        Spatial bandwidth (standard deviation) for the diffusion kernel. Controls
        the amount of smoothing. Units should match environment units.
    weights : np.ndarray, shape (n_time_position,), optional
        Weights for each position sample (e.g., duration). If None, assumes
        uniform sampling time.
    disable_progress_bar : bool, optional
        If True, suppresses progress bars. Defaults to False.

    Returns
    -------
    encoding_model : dict
        A dictionary containing the fitted model components:
        'place_fields' : jnp.ndarray, shape (n_neurons, n_total_bins)
            Smoothed firing rate maps for each neuron across all grid bins.
        'mean_rates' : list[float]
            Average firing rate for each neuron.
        'smoothed_occupancy' : jnp.ndarray, shape (n_total_bins,)
            Smoothed occupancy map across all grid bins.
        'interior_bin_indices_flat' : jnp.ndarray, shape (n_interior_bins,)
            Flat indices of the bins considered part of the track interior.
        'diffusion_bandwidth_sigma' : float
            The bandwidth parameter used for diffusion smoothing.
        'environment' : Environment
            The Environment object used for fitting.
        'no_spike_part_log_likelihood': jnp.ndarray, shape (n_total_bins,)
            Sum of place fields, used in likelihood calculation.
        'is_track_interior' : jnp.ndarray, shape (n_total_bins,)
             Flattened boolean mask of track interior bins.
        'disable_progress_bar': bool
    """
    # --- Input Validation and Setup ---
    if not environment._is_fitted:
        raise ValueError("Environment object must be fitted first.")

    track_graph_nd_ = environment.get_fitted_track_graph()

    position = position if position.ndim > 1 else np.expand_dims(position, axis=1)
    n_total_bins = environment.place_bin_centers_.shape[0]
    interior_mask = environment.is_track_interior_
    interior_mask_flat = interior_mask.ravel()
    interior_indices = jnp.where(interior_mask_flat)[0]

    # --- Diffusion Kernels ---
    print("Computing diffusion kernels...")
    kernel_matrix = compute_diffusion_kernels(
        track_graph=track_graph_nd_,
        interior_mask=interior_mask,
        bandwidth_sigma=position_std,
    )
    print(f"Computed {kernel_matrix.shape} kernel matrix.")

    # --- Compute Smoothed Occupancy ---
    print("Calculating smoothed occupancy...")
    if weights is None:
        weights = np.ones_like(position_time)

    # Filter out NaN positions BEFORE binning
    is_nan_pos = np.any(np.isnan(position), axis=1)
    valid_pos_mask = ~is_nan_pos
    valid_positions = position[valid_pos_mask]
    weights = weights[valid_pos_mask]

    # Bin valid positions using the Environment method
    if valid_positions.shape[0] > 0:
        position_bin_inds_valid = environment.bin_at(valid_positions)
    else:
        position_bin_inds_valid = np.array([], dtype=int)

    # Histogram of time spent on the full grid
    full_occupancy_hist = np.zeros(n_total_bins)
    np.add.at(full_occupancy_hist, position_bin_inds_valid, weights)

    # Filter to interior bins
    interior_occupancy_hist = jnp.asarray(full_occupancy_hist[interior_indices])

    # Smooth using the kernel matrix
    smoothed_interior_occupancy = kernel_matrix @ interior_occupancy_hist

    # Normalize to density (divide by total time spent in interior)
    total_interior_time = smoothed_interior_occupancy.sum()
    if total_interior_time > EPS:
        smoothed_interior_occupancy_density = (
            smoothed_interior_occupancy / total_interior_time
        )
    else:
        print("Warning: Total occupancy time in interior is close to zero.")
        smoothed_interior_occupancy_density = jnp.zeros_like(
            smoothed_interior_occupancy
        )

    # Map back to full grid (zero elsewhere)
    smoothed_occupancy_full = (
        jnp.zeros(n_total_bins)
        .at[interior_indices]
        .set(smoothed_interior_occupancy_density)
    )

    # --- Compute Smoothed Spike Marginals and Place Fields (Per Neuron) ---
    place_fields = []
    mean_rates = []

    total_encoding_time = weights.sum()  # More accurate total time from weights

    for neuron_id, neuron_spike_times in enumerate(
        tqdm(
            spike_times,
            unit="neuron",
            desc="Encoding models (Diffusion)",
            disable=disable_progress_bar,
        )
    ):
        # Filter spikes outside position time range
        is_valid_spike = np.logical_and(
            neuron_spike_times >= position_time[0],
            neuron_spike_times <= position_time[-1],
        )
        neuron_spike_times = neuron_spike_times[is_valid_spike]
        n_spikes = len(neuron_spike_times)

        # Calculate mean rate
        mean_rates.append(
            n_spikes / total_encoding_time if total_encoding_time > 0 else 0.0
        )

        if n_spikes > 0:
            # Find position at each spike time
            pos_at_spike = get_position_at_time(
                position_time, position, neuron_spike_times, environment
            )
            time_weights_at_spikes = get_position_at_time(
                position_time, weights, neuron_spike_times, environment
            )

            # Filter out NaN spike positions BEFORE binning
            is_nan_spike_pos = np.any(np.isnan(pos_at_spike), axis=1)
            valid_spike_pos_mask = ~is_nan_spike_pos
            valid_pos_at_spike = pos_at_spike[valid_spike_pos_mask]
            time_weights_at_spikes = time_weights_at_spikes[valid_spike_pos_mask]

            # Bin valid spike positions using the Environment method
            if valid_pos_at_spike.shape[0] > 0:
                spike_bin_inds_valid = environment.bin_at(valid_pos_at_spike)
            else:
                spike_bin_inds_valid = np.array([], dtype=int)

            # Histogram of spikes on the full grid
            full_spike_hist = np.zeros(n_total_bins)
            if spike_bin_inds_valid.size > 0:
                np.add.at(full_spike_hist, spike_bin_inds_valid, time_weights_at_spikes)

            # Filter to interior bins
            interior_spike_hist = jnp.asarray(full_spike_hist[interior_indices])

            # Smooth using the kernel matrix
            smoothed_interior_spike_density = kernel_matrix @ interior_spike_hist

            # Calculate smoothed marginal density (normalized)
            total_interior_spikes = smoothed_interior_spike_density.sum()
            if total_interior_spikes > EPS:
                smoothed_interior_marginal_density = (
                    smoothed_interior_spike_density / total_interior_spikes
                )
            else:
                smoothed_interior_marginal_density = jnp.zeros_like(
                    smoothed_interior_spike_density
                )

            # Calculate place field value for interior bins
            place_field_interior = mean_rates[-1] * jnp.where(
                smoothed_interior_occupancy_density > EPS,
                smoothed_interior_marginal_density
                / smoothed_interior_occupancy_density,
                0.0,  # Rate is zero if occupancy is zero
            )
            # Clip to avoid negative values (shouldn't happen) and ensure minimum rate
            place_field_interior = jnp.clip(place_field_interior, a_min=EPS)

            # Map back to full grid
            neuron_place_field = (
                jnp.zeros(n_total_bins).at[interior_indices].set(place_field_interior)
            )

        else:
            # No spikes, place field is EPS everywhere
            neuron_place_field = jnp.full((n_total_bins,), EPS)

        place_fields.append(neuron_place_field)

    # --- Finalize and Return ---
    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)  # Sum of rates lambda

    return {
        "place_fields": place_fields,
        "mean_rates": mean_rates,
        "environment": environment,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "interior_bin_indices_flat": interior_indices,
        "is_track_interior": interior_mask_flat,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_sorted_spikes_diffusion_kde_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[np.ndarray],
    place_fields: jnp.ndarray,
    mean_rates: List[float],
    no_spike_part_log_likelihood: jnp.ndarray,
    environment: Environment,
    interior_bin_indices_flat: jnp.ndarray,
    is_track_interior: jnp.ndarray,  # Flattened boolean mask
    disable_progress_bar: bool = False,
    is_local: bool = False,
) -> jnp.ndarray:
    """Predict the log likelihood of sorted spikes using diffusion encoding models.

    Calculates the log likelihood based on Poisson statistics, where the rate
    parameter is determined by the diffusion-smoothed place fields.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Decoding time bins' edges or centers. Edges are expected for binning spikes.
    spike_times : list[np.ndarray]
        List where each element is an array of spike times for a single neuron
        during the decoding period.
    place_fields : jnp.ndarray, shape (n_neurons, n_total_bins)
        Smoothed firing rate maps (lambda) from the fitted diffusion model.
    no_spike_part_log_likelihood : jnp.ndarray, shape (n_total_bins,)
        Sum of place fields (sum of lambda) across neurons for each bin.
    environment : Environment
        The *fitted* Environment object used during encoding.
    interior_bin_indices_flat : jnp.ndarray, shape (n_interior_bins,)
        Flat indices of the bins considered part of the track interior.
    is_track_interior : jnp.ndarray, shape (n_total_bins,)
        Flattened boolean mask of track interior bins.
    disable_progress_bar : bool, optional
        If True, suppresses progress bars. Defaults to False.
    is_local : bool, optional
        If True, compute the log likelihood at the animal's position (requires
        `position_time` and `position`). Uses nearest interior bin approximation.
        If False, compute across all interior bins. Defaults to False.
    position_time : np.ndarray, shape (n_pos_time,), optional
        Timestamps for position data, required if `is_local` is True.
    position : np.ndarray, shape (n_pos_time, n_dims), optional
        Position data, required if `is_local` is True.


    Returns
    -------
    log_likelihood : jnp.ndarray
        Shape (n_time, n_interior_bins) if `is_local` is False.
        Shape (n_time, 1) if `is_local` is True.

    Notes
    -----
    The local likelihood calculation (`is_local=True`) uses a nearest-neighbor
    approach, mapping the animal's continuous position to the closest interior
    bin center and using the rate precomputed for that bin. This simplifies
    computation but ignores sub-bin spatial information.
    Expects `time` to be bin edges for `get_spikecount_per_time_bin`.
    """
    # Check if time likely represents edges or centers. Assume edges if length > 1.
    n_time = len(time)

    log_likelihood = None  # Initialize

    if is_local:
        # --- Local Likelihood Calculation (Nearest Interior Bin) ---
        if position_time is None or position is None:
            raise ValueError(
                "'position_time' and 'position' are required when is_local=True."
            )
        if not environment._is_fitted:
            raise ValueError("Environment object must be fitted.")

        print("Calculating local likelihood (nearest interior bin)...")
        # 1. Get centers of time bins for interpolation
        # time_bin_centers = time[:-1] + np.diff(time) / 2

        # 2. Interpolate position to time bin centers
        interpolated_position = get_position_at_time(
            position_time, position, time, environment
        )  # Shape (n_time, n_dims)

        # Filter out NaN interpolated positions
        is_nan_interp_pos = np.any(np.isnan(interpolated_position), axis=1)
        valid_interp_pos = interpolated_position[~is_nan_interp_pos]

        # 3. Find nearest *interior* bin center for each *valid* interpolated position
        nearest_interior_indices = np.full(
            n_time, -1, dtype=int
        )  # Initialize with invalid index
        if valid_interp_pos.shape[0] > 0:
            interior_centers = environment.place_bin_centers_[interior_bin_indices_flat]
            if interior_centers.shape[0] == 0:
                raise ValueError("No interior bin centers found in environment.")
            # Use KDTree for efficient nearest neighbor search
            tree = KDTree(interior_centers)
            _, nn_indices_for_valid = tree.query(valid_interp_pos)
            nearest_interior_indices[~is_nan_interp_pos] = nn_indices_for_valid
        # nearest_interior_indices shape: (n_time,). Values are indices *within the interior subset* or -1.

        # 4. Calculate log likelihood summed across neurons
        log_likelihood_local = jnp.zeros((n_time,))

        for neuron_id, neuron_spike_times in enumerate(
            tqdm(
                spike_times,
                unit="neuron",
                desc="Local Likelihood (Diffusion)",
                disable=disable_progress_bar,
            )
        ):
            # Count spikes in decoding time bins
            spike_counts_per_bin = get_spikecount_per_time_bin(
                neuron_spike_times, time
            )  # Shape (n_time,)

            # Get the rate for this neuron at the nearest interior bin for each time point
            neuron_place_field_interior = place_fields[neuron_id][
                interior_bin_indices_flat
            ]

            # Initialize local rates with EPS (for invalid positions or bins)
            local_rates = jnp.full((n_time,), EPS)

            # Get rates only for time bins with valid nearest neighbors
            valid_time_bins_mask = nearest_interior_indices != -1
            valid_nn_indices = nearest_interior_indices[valid_time_bins_mask]

            if valid_nn_indices.size > 0:
                rates_at_valid_bins = neuron_place_field_interior[
                    jnp.asarray(valid_nn_indices)
                ]
                local_rates = local_rates.at[valid_time_bins_mask].set(
                    rates_at_valid_bins
                )

            local_rates = jnp.clip(local_rates, a_min=EPS)  # Ensure rate > 0

            # Calculate Poisson log likelihood component for this neuron
            log_likelihood_local += (
                jax.scipy.special.xlogy(spike_counts_per_bin, local_rates) - local_rates
            )

        return log_likelihood_local[:, jnp.newaxis]  # Reshape to (n_time, 1)

    else:
        # --- Non-Local Likelihood Calculation ---
        print("Calculating non-local likelihood...")
        n_interior_bins = interior_bin_indices_flat.shape[0]
        if n_interior_bins == 0:
            print("Warning: No interior bins found. Returning empty likelihood array.")
            return jnp.zeros((n_time, 0))

        log_likelihood_nonlocal = jnp.zeros((n_time, n_interior_bins))

        # Pre-filter place fields and summed rate for interior bins only
        place_fields_interior = place_fields[
            :, interior_bin_indices_flat
        ]  # (n_neurons, n_interior_bins)
        no_spike_ll_interior = no_spike_part_log_likelihood[
            interior_bin_indices_flat
        ]  # (n_interior_bins,)

        for neuron_id, neuron_spike_times in enumerate(
            tqdm(
                spike_times,
                unit="neuron",
                desc="Non-Local Likelihood (Diffusion)",
                disable=disable_progress_bar,
            )
        ):
            # Count spikes in decoding time bins
            spike_counts_per_bin = get_spikecount_per_time_bin(
                neuron_spike_times, time
            )  # Shape (n_time,)

            # Get rates (lambda) for this neuron across all interior bins
            rates_interior = place_fields_interior[
                neuron_id
            ]  # Shape (n_interior_bins,)
            rates_interior = jnp.clip(rates_interior, a_min=EPS)

            # Calculate k * log(lambda) part using broadcasting
            log_likelihood_nonlocal += jax.scipy.special.xlogy(
                spike_counts_per_bin[:, jnp.newaxis], rates_interior[jnp.newaxis, :]
            )

        # Subtract the summed rate term (lambda)
        log_likelihood_nonlocal -= no_spike_ll_interior[jnp.newaxis, :]

        return log_likelihood_nonlocal  # Shape (n_time, n_interior_bins)
