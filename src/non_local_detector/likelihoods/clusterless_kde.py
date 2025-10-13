"""Clusterless decoding using Kernel Density Estimation (KDE).

This module implements a clusterless decoding algorithm based on Kernel Density
Estimation (KDE). Unlike traditional methods that rely on pre-sorted neural
units (clusters), this approach uses the waveform features of detected spikes
directly, treating them as "marks" in a marked point process.

The core idea is to model the joint distribution of spike times, spike waveform
features, and the animal's position during an "encoding" period (typically
when the animal is actively exploring the environment). This model is then used
during a "decoding" period (e.g., during sleep or quiet wakefulness) to compute
the likelihood of observing spikes with specific waveform features occurring at
various positions.

Key components:
1.  **Encoding Model Fitting (`fit_clusterless_kde_encoding_model`):**
    - Takes position data, spike times, and spike waveform features from the
      encoding period.
    - Uses KDE with Gaussian kernels to estimate:
        - Spatial occupancy (how much time the animal spends where).
        - Ground process intensity (spatial firing rate density, marginalizing
          over features) for each electrode.
        - Implicitly models the joint distribution of position and waveform
          features for spikes on each electrode.
    - Handles both 2D environments and 1D linearized tracks.
    - Returns a dictionary containing the fitted models (occupancy, GPI models),
      processed encoding data, and parameters.

2.  **Log-Likelihood Prediction (`predict_clusterless_kde_log_likelihood`):**
    - Takes time bins for decoding, new spike data (times and features), and
      the fitted encoding model.
    - Calculates the log-likelihood of the observed spikes under the model for
      each time bin.
    - Can compute either:
        - **Non-local likelihood:** Log-likelihood across all spatial bins,
          suitable for estimating a posterior probability distribution over
          position.
        - **Local likelihood:** Log-likelihood specifically at the animal's
          actual (interpolated) position at each time bin.
    - Leverages JAX for efficient computation, particularly the KDE steps,
      often using blocking (`block_estimate_log_joint_mark_intensity`) to
      manage memory for large datasets.

Helper functions are included for tasks like mapping spike times to time bins
(`get_spike_time_bin_ind`) and computing KDE distances (`kde_distance`).
Constants like `EPS` and `LOG_EPS` are used for numerical stability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    KDEModel,
    block_kde,
    gaussian_pdf,
    get_position_at_time,
    safe_divide,
)


def get_spike_time_bin_ind(
    spike_times: np.ndarray, time_bin_edges: np.ndarray
) -> np.ndarray:
    """Gets the index of the time bin for each spike time.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Times of occurrences (spikes).
    time_bin_edges : np.ndarray, shape (n_bins + 1,)
        Sorted array defining the edges of the time bins [t0, t1, ..., tN].
        Defines n_bins intervals: [t0, t1), ..., [t_{N-1}, tN].

    Returns
    -------
    bin_indices : np.ndarray, shape (n_spikes,)
        Index of the bin for each spike. Bins are indexed 0 to n_bins-1.
        Doesn't handle out of bounds spikes.
    """

    bin_indices = np.searchsorted(time_bin_edges, spike_times, side="right") - 1
    is_last_bin = np.isclose(
        spike_times,
        time_bin_edges[-1],
    )
    bin_indices[is_last_bin] = len(time_bin_edges) - 2

    return bin_indices


@jax.jit
def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Distance between evaluation points and samples using Gaussian kernel density

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    distance : jnp.ndarray, shape (n_samples, n_eval_points)

    """
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))
    for dim_eval_points, dim_samples, dim_std in zip(
        eval_points.T, samples.T, std, strict=False
    ):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return distance


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    encoding_weights: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    pos_tile_size: int | None = None,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension in chunks of this size.
        Reduces peak memory from O(n_enc * n_pos) to O(n_enc * pos_tile_size).
        If None (default), process all positions at once (fastest but more memory).

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    spike_waveform_feature_distance = kde_distance(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
    )  # shape (n_encoding_spikes, n_decoding_spikes)

    n_encoding_spikes = jnp.sum(encoding_weights)
    n_pos = position_distance.shape[1]
    n_dec = spike_waveform_feature_distance.shape[1]

    if pos_tile_size is None or pos_tile_size >= n_pos:
        # No tiling: process all positions at once (default)
        marginal_density = (
            spike_waveform_feature_distance.T
            @ (encoding_weights[:, None] * position_distance)
            / n_encoding_spikes
        )  # shape (n_decoding_spikes, n_position_bins)
    else:
        # Tiled: process positions in chunks to reduce peak memory
        marginal_density = jnp.zeros((n_dec, n_pos))

        for pos_start in range(0, n_pos, pos_tile_size):
            pos_end = min(pos_start + pos_tile_size, n_pos)
            pos_slice = slice(pos_start, pos_end)

            # Compute for this tile
            marginal_density_tile = (
                spike_waveform_feature_distance.T
                @ (encoding_weights[:, None] * position_distance[:, pos_slice])
                / n_encoding_spikes
            )  # shape (n_decoding_spikes, tile_size)

            # Update output
            marginal_density = marginal_density.at[:, pos_slice].set(marginal_density_tile)

    return jnp.log(mean_rate * safe_divide(marginal_density, occupancy))


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    encoding_weights: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    block_size: int = 100,
    pos_tile_size: int | None = None,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
    block_size : int, optional
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension. Passed to estimate_log_joint_mark_intensity.

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]
    if n_decoding_spikes == 0:
        return jnp.full(
            (0, n_position_bins), LOG_EPS
        )  # Return empty if no decoding spikes

    log_joint_mark_intensity = jnp.zeros((n_decoding_spikes, n_position_bins))

    for start_ind in range(0, n_decoding_spikes, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        log_joint_mark_intensity = jax.lax.dynamic_update_slice(
            log_joint_mark_intensity,
            estimate_log_joint_mark_intensity(
                decoding_spike_waveform_features[block_inds],
                encoding_spike_waveform_features,
                encoding_weights,
                waveform_stds,
                occupancy,
                mean_rate,
                position_distance,
                pos_tile_size=pos_tile_size,
            ),
            (start_ind, 0),
        )

    return jnp.clip(log_joint_mark_intensity, min=LOG_EPS, max=None)


def fit_clusterless_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    weights: jnp.ndarray | None = None,
    sampling_frequency: int = 500,
    position_std: float | jnp.ndarray = np.sqrt(12.5),
    waveform_std: float | jnp.ndarray = 24.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
) -> dict:
    """Fit the clusterless KDE encoding model.

    Parameters
    ----------
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Spike waveform features for each electrode.
    environment : Environment
        The spatial environment.
    weights : jnp.ndarray, shape (n_time_position,), optional
        Sample weights for each position time point, by default None.
        If None, uniform weights are used.
    sampling_frequency : int, optional
        Samples per second, by default 500
    position_std : float, optional
        Gaussian smoothing standard deviation for position, by default sqrt(12.5)
    waveform_std : float, optional
        Gaussian smoothing standard deviation for waveform, by default 24.0
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    Returns
    -------
    encoding_model : dict
        Dictionary containing fitted encoding model components:
        - 'occupancy': Occupancy density at interior place bins
        - 'occupancy_model': Fitted KDE model for occupancy
        - 'gpi_models': Ground process intensity KDE models per electrode
        - 'encoding_spike_waveform_features': Bounded spike features per electrode
        - 'encoding_positions': Position at spike times per electrode
        - 'encoding_spike_weights': Weights at spike times per electrode
        - 'environment': The spatial environment
        - 'mean_rates': Mean firing rates per electrode
        - 'summed_ground_process_intensity': Summed GPI across electrodes
        - 'position_std': Position kernel standard deviations
        - 'waveform_std': Waveform kernel standard deviations
        - 'block_size': Block size used for computation
        - 'disable_progress_bar': Progress bar setting
    """
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if isinstance(position_std, (int, float)):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])
    # Ensure position_std is a JAX array for KDEModel
    assert isinstance(position_std, jnp.ndarray)

    if isinstance(waveform_std, (int, float)):
        waveform_std = jnp.array([waveform_std] * spike_waveform_features[0].shape[1])
    # Ensure waveform_std is a JAX array for KDEModel
    assert isinstance(waveform_std, jnp.ndarray)

    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        if environment.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ is required when is_track_interior_ is None"
            )
        is_track_interior = jnp.ones(
            environment.place_bin_centers_.shape[0], dtype=bool
        )
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    if weights is None:
        weights = jnp.ones((position.shape[0],))
    total_weight = np.sum(weights)

    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        position1D = get_linearized_position(
            position,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position1D, weights=weights
        )
    else:
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position, weights=weights
        )

    occupancy = occupancy_model.predict(interior_place_bin_centers)
    encoding_positions = []
    encoding_spike_weights = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)

    bounded_spike_waveform_features = []

    for electrode_spike_waveform_features, electrode_spike_times in zip(
        tqdm(
            spike_waveform_features,
            desc="Encoding models",
            unit="electrode",
            disable=disable_progress_bar,
        ),
        spike_times,
        strict=False,
    ):
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= position_time[0],
            electrode_spike_times <= position_time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        bounded_spike_waveform_features.append(
            electrode_spike_waveform_features[is_in_bounds]
        )

        electrode_weights_at_spike_times = np.interp(
            electrode_spike_times, position_time, weights
        )
        encoding_spike_weights.append(electrode_weights_at_spike_times)
        mean_rates.append(np.sum(electrode_weights_at_spike_times) / total_weight)
        encoding_positions.append(
            get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1], weights=jnp.array(electrode_weights_at_spike_times)
        )
        gpi_models.append(gpi_model)

        summed_ground_process_intensity += jnp.clip(
            mean_rates[-1]
            * safe_divide(gpi_model.predict(interior_place_bin_centers), occupancy),
            min=EPS,
            max=None,
        )

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
        "encoding_spike_weights": encoding_spike_weights,
        "environment": environment,
        "mean_rates": mean_rates,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "position_std": position_std,
        "waveform_std": waveform_std,
        "block_size": block_size,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_clusterless_kde_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy: jnp.ndarray,
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: list[jnp.ndarray],
    encoding_spike_weights: jnp.ndarray,
    environment: Environment,
    mean_rates: jnp.ndarray,
    summed_ground_process_intensity: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    is_local: bool = False,
    block_size: int = 100,
    disable_progress_bar: bool = False,
) -> jnp.ndarray:
    """Predict the log likelihood of the clusterless KDE model.

    Parameters
    ----------
    time : jnp.ndarray
        Decoding time bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Waveform features for each electrode.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        How much time is spent in each position bin by the animal.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jnp.ndarray]
        Spike waveform features for each electrode used for encoding.
    encoding_positions : list[jnp.ndarray], shape (n_encoding_spikes, n_position_dims)
        Position samples used for encoding.
    environment : Environment
        The spatial environment
    mean_rates : jnp.ndarray, shape (n_electrodes,)
        Mean firing rate for each electrode.
    summed_ground_process_intensity : jnp.ndarray, shape (n_position_bins,)
        Summed ground process intensity for all electrodes.
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    is_local : bool, optional
        If True, compute the log likelihood at the animal's position, by default False
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1) or (n_time, n_position_bins)
        Shape depends on whether local or non-local decoding, respectively.
    """
    n_time = len(time)

    if is_local:
        log_likelihood = compute_local_log_likelihood(
            time,
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            occupancy_model,
            gpi_models,
            encoding_spike_waveform_features,
            encoding_positions,
            encoding_spike_weights,
            environment,
            mean_rates,
            position_std,
            waveform_std,
            block_size,
            disable_progress_bar,
        )
    else:
        if environment.is_track_interior_ is not None:
            is_track_interior = environment.is_track_interior_.ravel()
        else:
            if environment.place_bin_centers_ is None:
                raise ValueError(
                    "place_bin_centers_ is required when is_track_interior_ is None"
                )
            is_track_interior = jnp.ones(
                environment.place_bin_centers_.shape[0], dtype=bool
            )
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

        log_likelihood = -1.0 * summed_ground_process_intensity * jnp.ones((n_time, 1))

        for (
            electrode_encoding_spike_waveform_features,
            electrode_encoding_positions,
            electrode_encoding_weights,
            electrode_mean_rate,
            electrode_decoding_spike_waveform_features,
            electrode_spike_times,
        ) in zip(
            tqdm(
                encoding_spike_waveform_features,
                unit="electrode",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_positions,
            encoding_spike_weights,
            mean_rates,
            spike_waveform_features,
            spike_times,
            strict=False,
        ):
            is_in_bounds = jnp.logical_and(
                electrode_spike_times >= time[0],
                electrode_spike_times <= time[-1],
            )
            electrode_spike_times = electrode_spike_times[is_in_bounds]
            electrode_decoding_spike_waveform_features = (
                electrode_decoding_spike_waveform_features[is_in_bounds]
            )
            position_distance = kde_distance(
                interior_place_bin_centers,
                electrode_encoding_positions,
                std=position_std,
            )

            log_likelihood += jax.ops.segment_sum(
                block_estimate_log_joint_mark_intensity(
                    electrode_decoding_spike_waveform_features,
                    electrode_encoding_spike_waveform_features,
                    electrode_encoding_weights,
                    waveform_std,
                    occupancy,
                    electrode_mean_rate,
                    position_distance,
                    block_size,
                ),
                get_spike_time_bin_ind(electrode_spike_times, time),
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
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: jnp.ndarray,
    encoding_spike_weights: list[jnp.ndarray],
    environment: Environment,
    mean_rates: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    block_size: int = 100,
    disable_progress_bar: bool = False,
) -> jnp.ndarray:
    """Compute the log likelihood at the animal's position.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
        Time bins for decoding.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        List of spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        List of spike waveform features for each electrode.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        List of KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jnp.ndarray]
        List of spike waveform features for each electrode used for encoding.
    encoding_positions : jnp.ndarray
        Position samples used for encoding.
    environment : Environment
        The spatial environment.
    mean_rates : jnp.ndarray
        Mean firing rate for each electrode.
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    weights : jnp.ndarray, shape (n_time_position,), optional
        Sample weights for position, by default None.
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
    """

    # Need to interpolate position
    interpolated_position = get_position_at_time(
        position_time, position, time, environment
    )
    occupancy = occupancy_model.predict(interpolated_position)

    n_time = len(time)
    log_likelihood = jnp.zeros((n_time,))
    for (
        electrode_encoding_spike_waveform_features,
        electrode_encoding_positions,
        electrode_encoding_weights,
        electrode_mean_rate,
        electrode_gpi_model,
        electrode_decoding_spike_waveform_features,
        electrode_spike_times,
    ) in zip(
        tqdm(
            encoding_spike_waveform_features,
            unit="electrode",
            desc="Local Likelihood",
            disable=disable_progress_bar,
        ),
        encoding_positions,
        encoding_spike_weights,
        mean_rates,
        gpi_models,
        spike_waveform_features,
        spike_times,
        strict=False,
    ):
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= time[0],
            electrode_spike_times <= time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        electrode_decoding_spike_waveform_features = (
            electrode_decoding_spike_waveform_features[is_in_bounds]
        )

        position_at_spike_time = get_position_at_time(
            position_time, position, electrode_spike_times, environment
        )

        marginal_density = block_kde(
            eval_points=jnp.concatenate(
                (
                    position_at_spike_time,
                    electrode_decoding_spike_waveform_features,
                ),
                axis=1,
            ),
            samples=jnp.concatenate(
                (
                    electrode_encoding_positions,
                    electrode_encoding_spike_waveform_features,
                ),
                axis=1,
            ),
            std=jnp.concatenate((position_std, waveform_std)),
            block_size=block_size,
            weights=electrode_encoding_weights,
        )
        occupancy_at_spike_time = occupancy_model.predict(position_at_spike_time)

        log_likelihood += jax.ops.segment_sum(
            jnp.log(
                electrode_mean_rate
                * safe_divide(marginal_density, occupancy_at_spike_time)
            ),
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * safe_divide(
            electrode_gpi_model.predict(interpolated_position), occupancy
        )

    return log_likelihood[:, jnp.newaxis]
