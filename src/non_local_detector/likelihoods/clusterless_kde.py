import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm
from track_linearization import get_linearized_position

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import (
    LOG_EPS,
    EPS,
    KDEModel,
    block_kde,
    gaussian_pdf,
    get_position_at_time,
)


def get_spike_time_bin_ind(spike_times, time):
    return np.digitize(spike_times, time[1:-1])


def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))
    for dim_eval_points, dim_samples, dim_std in zip(eval_points.T, samples.T, std):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return distance


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    spike_waveform_feature_distance = kde_distance(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
    )

    n_encoding_spikes = encoding_spike_waveform_features.shape[0]
    marginal_density = (
        spike_waveform_feature_distance.T @ position_distance / n_encoding_spikes
    )
    return jnp.log(
        mean_rate * jnp.where(occupancy > 0.0, marginal_density / occupancy, 0.0)
    )


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    block_size: int = 100,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
    block_size : int, optional

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    log_joint_mark_intensity = jnp.zeros((n_decoding_spikes, n_position_bins))

    for start_ind in range(0, n_decoding_spikes, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        log_joint_mark_intensity = jax.lax.dynamic_update_slice(
            log_joint_mark_intensity,
            estimate_log_joint_mark_intensity(
                decoding_spike_waveform_features[block_inds],
                encoding_spike_waveform_features,
                waveform_stds,
                occupancy,
                mean_rate,
                position_distance,
            ).squeeze(),
            (start_ind, 0),
        )

    return jnp.clip(log_joint_mark_intensity, a_min=LOG_EPS, a_max=None)


def fit_clusterless_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    sampling_frequency: int = 500,
    position_std: float = 6.0,
    waveform_std: float = 24.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
):
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if isinstance(position_std, (int, float)):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])
    if isinstance(waveform_std, (int, float)):
        waveform_std = jnp.array([waveform_std] * spike_waveform_features[0].shape[1])

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        position1D = get_linearized_position(
            position,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position1D
        )
    else:
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position
        )

    occupancy = occupancy_model.predict(interior_place_bin_centers)
    encoding_positions = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)

    n_time_bins = int((position_time[-1] - position_time[0]) * sampling_frequency)
    bounded_spike_waveform_features = []

    for electrode_spike_waveform_features, electrode_spike_times in zip(
        tqdm(
            spike_waveform_features,
            desc="Encoding models",
            unit="electrode",
            disable=disable_progress_bar,
        ),
        spike_times,
    ):
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= position_time[0],
            electrode_spike_times <= position_time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        bounded_spike_waveform_features.append(
            electrode_spike_waveform_features[is_in_bounds]
        )
        mean_rates.append(len(electrode_spike_times) / n_time_bins)
        encoding_positions.append(
            get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1]
        )
        gpi_models.append(gpi_model)

        summed_ground_process_intensity += jnp.clip(
            mean_rates[-1] * gpi_model.predict(interior_place_bin_centers) / occupancy,
            a_min=EPS,
            a_max=None,
        )

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
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
    occupancy,
    occupancy_model,
    gpi_models,
    encoding_spike_waveform_features,
    encoding_positions,
    environment,
    mean_rates,
    summed_ground_process_intensity,
    position_std,
    waveform_std,
    is_local: bool = False,
    block_size: int = 100,
    disable_progress_bar: bool = False,
):
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
            environment,
            mean_rates,
            position_std,
            waveform_std,
            block_size,
            disable_progress_bar,
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

        log_likelihood = -1.0 * summed_ground_process_intensity * jnp.ones((n_time, 1))

        for (
            electrode_encoding_spike_waveform_features,
            electrode_encoding_positions,
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
            mean_rates,
            spike_waveform_features,
            spike_times,
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
    occupancy_model,
    gpi_models,
    encoding_spike_waveform_features,
    encoding_positions,
    environment,
    mean_rates,
    position_std,
    waveform_std,
    block_size: int = 100,
    disable_progress_bar: bool = False,
):
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
        mean_rates,
        gpi_models,
        spike_waveform_features,
        spike_times,
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
        )
        occupancy_at_spike_time = occupancy_model.predict(position_at_spike_time)

        log_likelihood += jax.ops.segment_sum(
            jnp.log(
                electrode_mean_rate
                * jnp.where(
                    occupancy_at_spike_time > 0.0,
                    marginal_density / occupancy_at_spike_time,
                    0.0,
                )
            ),
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * jnp.where(
            occupancy > 0.0,
            electrode_gpi_model.predict(interpolated_position) / occupancy,
            0.0,
        )
    return log_likelihood[:, jnp.newaxis]
