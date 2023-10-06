import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm
from track_linearization import get_linearized_position

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import EPS, KDEModel, get_position_at_time


def fit_sorted_spikes_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    environment: Environment,
    sampling_frequency: int = 500,
    position_std: float = 6.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
):
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if isinstance(position_std, (int, float)):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])

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

    mean_rates = []
    place_fields = []
    marginal_models = []

    n_time_bins = int((position_time[-1] - position_time[0]) * sampling_frequency)

    for neuron_spike_times in tqdm(
        spike_times,
        unit="cell",
        desc="Encoding models",
        disable=disable_progress_bar,
    ):
        neuron_spike_times = neuron_spike_times[
            jnp.logical_and(
                neuron_spike_times >= position_time[0],
                neuron_spike_times <= position_time[-1],
            )
        ]
        mean_rates.append(len(neuron_spike_times) / n_time_bins)
        neuron_marginal_model = KDEModel(std=position_std, block_size=block_size).fit(
            get_position_at_time(
                position_time, position, neuron_spike_times, environment
            )
        )
        marginal_models.append(neuron_marginal_model)
        marginal_density = neuron_marginal_model.predict(interior_place_bin_centers)
        marginal_density = jnp.where(jnp.isnan(marginal_density), 0.0, marginal_density)
        place_fields.append(
            jnp.zeros((is_track_interior.shape[0],))
            .at[is_track_interior]
            .set(
                jnp.clip(
                    mean_rates[-1]
                    * jnp.where(occupancy > 0.0, marginal_density / occupancy, EPS),
                    a_min=EPS,
                    a_max=None,
                )
            )
        )

    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "environment": environment,
        "marginal_models": marginal_models,
        "occupancy_model": occupancy_model,
        "occupancy": occupancy,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_sorted_spikes_kde_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    marginal_models: list[KDEModel],
    occupancy_model: KDEModel,
    occupancy: jnp.ndarray,
    mean_rates: jnp.ndarray,
    place_fields: jnp.ndarray,
    no_spike_part_log_likelihood: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    disable_progress_bar: bool = False,
    is_local: bool = False,
):
    n_time = time.shape[0]
    if is_local:
        log_likelihood = jnp.zeros((n_time,))

        # Need to interpolate position
        interpolated_position = get_position_at_time(
            position_time, position, time, environment
        )
        occupancy = occupancy_model.predict(interpolated_position)

        for neuron_spike_times, neuron_marginal_model, neuron_mean_rate in zip(
            tqdm(
                spike_times,
                unit="cell",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            marginal_models,
            mean_rates,
        ):
            neuron_spike_times = neuron_spike_times[
                np.logical_and(
                    neuron_spike_times >= time[0],
                    neuron_spike_times <= time[-1],
                )
            ]
            spike_count_per_time_bin = np.bincount(
                np.digitize(neuron_spike_times, time[1:-1]),
                minlength=time.shape[0],
            )
            marginal_density = neuron_marginal_model.predict(interpolated_position)
            marginal_density = jnp.where(
                jnp.isnan(marginal_density), 0.0, marginal_density
            )
            local_rate = neuron_mean_rate * jnp.where(
                occupancy > 0.0, marginal_density / occupancy, EPS
            )
            local_rate = jnp.clip(local_rate, a_min=EPS, a_max=None)
            log_likelihood += (
                jax.scipy.special.xlogy(spike_count_per_time_bin, local_rate)
                - local_rate
            )

        log_likelihood = jnp.expand_dims(log_likelihood, axis=1)
    else:
        n_interior_bins = is_track_interior.sum()
        log_likelihood = jnp.zeros((n_time, n_interior_bins))
        for neuron_spike_times, place_field in zip(
            tqdm(
                spike_times,
                unit="cell",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            place_fields,
        ):
            neuron_spike_times = neuron_spike_times[
                np.logical_and(
                    neuron_spike_times >= time[0],
                    neuron_spike_times <= time[-1],
                )
            ]
            spike_count_per_time_bin = np.bincount(
                np.digitize(neuron_spike_times, time[1:-1]),
                minlength=time.shape[0],
            )
            log_likelihood += jax.scipy.special.xlogy(
                np.expand_dims(spike_count_per_time_bin, axis=1),
                jnp.expand_dims(place_field[is_track_interior], axis=0),
            )

        log_likelihood -= no_spike_part_log_likelihood[is_track_interior]

    return log_likelihood
