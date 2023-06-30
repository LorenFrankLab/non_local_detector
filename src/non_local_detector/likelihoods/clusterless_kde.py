from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import scipy.interpolate
from tqdm.autonotebook import tqdm
from track_linearization import get_linearized_position

from non_local_detector.environment import Environment

EPS = 1e-15


def get_spike_time_bin_ind(spike_times, time):
    return jnp.digitize(spike_times, time[1:-1])


@jax.jit
def gaussian_pdf(x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma."""
    return jnp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def kde(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))

    for dim_ind, std in enumerate(std):
        distance *= gaussian_pdf(
            jnp.expand_dims(eval_points[:, dim_ind], axis=0),
            jnp.expand_dims(samples[:, dim_ind], axis=1),
            std,
        )
    return jnp.mean(distance, axis=0).squeeze()


@partial(jax.jit, static_argnums=(3,))
def block_kde(
    eval_points: jnp.ndarray,
    samples: jnp.ndarray,
    std: jnp.ndarray,
    block_size: int = 100,
) -> jnp.ndarray:
    n_eval_points = eval_points.shape[0]
    density = jnp.zeros((n_eval_points,))
    for start_ind in range(0, n_eval_points, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        density = jax.lax.dynamic_update_slice(
            density,
            kde(eval_points[block_inds], samples, std).squeeze(),
            (start_ind,),
        )

    return density


@jax.jit
def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))

    for dim_ind, std in enumerate(std):
        distance *= gaussian_pdf(
            jnp.expand_dims(eval_points[:, dim_ind], axis=0),
            jnp.expand_dims(samples[:, dim_ind], axis=1),
            std,
        )
    return distance


@jax.jit
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


@partial(jax.jit, static_argnums=(6,))
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

    return log_joint_mark_intensity


@dataclass
class KDEModel:
    std: jnp.ndarray
    block_size: int | None = None

    def fit(self, samples: jnp.ndarray):
        samples = jnp.asarray(samples)
        if samples.ndim == 1:
            samples = jnp.expand_dims(samples, axis=1)
        self.samples_ = samples

        return self

    def predict(self, eval_points: jnp.ndarray):
        if eval_points.ndim == 1:
            eval_points = jnp.expand_dims(eval_points, axis=1)
        std = (
            jnp.array([self.std] * eval_points.shape[1])
            if isinstance(self.std, (int, float))
            else self.std
        )
        block_size = (
            eval_points.shape[0] if self.block_size is None else self.block_size
        )

        return block_kde(eval_points, self.samples_, std, block_size)


def get_position_at_time(
    time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: jnp.ndarray,
    env: Environment | None = None,
):
    position_at_spike_times = scipy.interpolate.interpn(
        (time,), position, spike_times, bounds_error=False, fill_value=None
    )
    if env is not None and env.track_graph is not None:
        position_at_spike_times = get_linearized_position(
            position_at_spike_times,
            env.track_graph,
            edge_order=env.edge_order,
            edge_spacing=env.edge_spacing,
        ).linear_position.to_numpy()

    return position_at_spike_times


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
    if isinstance(position_std, (int, float)):
        position_std = jnp.array([position_std] * position.shape[1])
    if isinstance(waveform_std, (int, float)):
        waveform_std = jnp.array([waveform_std] * spike_waveform_features[0].shape[1])

    is_track_interior = environment.is_track_interior_.ravel(order="F")
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    if environment.track_graph is not None:
        # convert to 1D
        position1D = get_linearized_position(
            position,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()
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

    for electrode_spike_times in spike_times:
        electrode_spike_times = electrode_spike_times[
            jnp.logical_and(
                electrode_spike_times >= position_time[0],
                electrode_spike_times <= position_time[-1],
            )
        ]
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

        summed_ground_process_intensity = summed_ground_process_intensity.at[
            is_track_interior
        ].add(gpi_model.predict(interior_place_bin_centers))

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": spike_waveform_features,
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
        # Need to interpolate position
        interpolated_position = get_position_at_time(
            position_time, position, time, environment
        )
        occupancy = occupancy_model.predict(interpolated_position)

        log_likelihood = jnp.zeros((n_time, 1))
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
            position_distance = kde_distance(
                interpolated_position,
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
                indices_are_sorted=False,
                num_segments=n_time,
            )
            log_likelihood -= electrode_gpi_model.predict(interpolated_position)
    else:
        is_track_interior = environment.is_track_interior_.ravel(order="F")
        interior_occupancy = occupancy[is_track_interior]
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

        log_likelihood = -summed_ground_process_intensity * jnp.ones(
            (n_time, 1),
        )

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
            log_likelihood = log_likelihood.at[:, is_track_interior].add(
                jax.ops.segment_sum(
                    block_estimate_log_joint_mark_intensity(
                        electrode_decoding_spike_waveform_features,
                        electrode_encoding_spike_waveform_features,
                        waveform_std,
                        interior_occupancy,
                        electrode_mean_rate,
                        position_distance,
                        block_size,
                    ),
                    get_spike_time_bin_ind(electrode_spike_times, time),
                    indices_are_sorted=False,
                    num_segments=n_time,
                )
            )

    return log_likelihood
