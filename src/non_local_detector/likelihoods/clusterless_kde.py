from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from tqdm.autonotebook import tqdm

EPS = 1e-15


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


def fit_clusterless_kde_encoding_model(
    position: jnp.ndarray,
    multiunits: jnp.ndarray,
    place_bin_centers: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    *args,
    position_std: float = 6.0,
    waveform_std: float = 24.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    **kwargs,
):
    if isinstance(position_std, (int, float)):
        position_std = jnp.array([position_std] * position.shape[1])
    if isinstance(waveform_std, (int, float)):
        waveform_std = jnp.array([waveform_std] * multiunits.shape[1])

    occupancy = jnp.zeros((place_bin_centers.shape[0],))
    occupancy_model = KDEModel(position_std, block_size=block_size).fit(position)
    occupancy = occupancy.at[is_track_interior].set(
        occupancy_model.predict(place_bin_centers[is_track_interior])
    )

    encoding_spike_waveform_features_electrodes = []
    encoding_position_electrodes = []
    mean_rates = []
    kde_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)

    for electrode_multiunit in jnp.moveaxis(multiunits, 2, 0):
        is_encoding_spike = jnp.any(~jnp.isnan(electrode_multiunit), axis=1)

        encoding_spike_waveform_features_electrodes.append(
            electrode_multiunit[is_encoding_spike]
        )
        mean_rates.append(jnp.mean(is_encoding_spike))
        encoding_position_electrodes.append(position[is_encoding_spike])

        kde_model = KDEModel(std=position_std, block_size=block_size).fit(
            position[is_encoding_spike]
        )
        kde_models.append(kde_model)

        summed_ground_process_intensity = summed_ground_process_intensity.at[
            is_track_interior
        ].add(kde_model.predict(place_bin_centers[is_track_interior]))

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "kde_models": kde_models,
        "encoding_spike_waveform_features_electrodes": encoding_spike_waveform_features_electrodes,
        "encoding_position_electrodes": encoding_position_electrodes,
        "mean_rates": mean_rates,
        "place_bin_centers": place_bin_centers,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "position_std": position_std,
        "waveform_std": waveform_std,
        "is_track_interior": is_track_interior,
        "block_size": block_size,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_clusterless_kde_log_likelihood(
    position: jnp.ndarray,
    multiunits: jnp.ndarray,
    occupancy,
    occupancy_model,
    kde_models,
    encoding_spike_waveform_features_electrodes,
    encoding_position_electrodes,
    mean_rates,
    place_bin_centers,
    summed_ground_process_intensity,
    position_std,
    waveform_std,
    is_track_interior: jnp.ndarray,
    is_local: bool = False,
    block_size: int = 100,
    disable_progress_bar: bool = False,
):
    n_time = multiunits.shape[0]

    if is_local:
        occupancy = occupancy_model.predict(position)
        log_likelihood = jnp.zeros((n_time, 1))
        for (
            encoding_spike_waveform_features,
            encoding_positions,
            mean_rate,
            kde_model,
            decoding_multiunits,
        ) in zip(
            tqdm(
                encoding_spike_waveform_features_electrodes,
                unit="electrode",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_position_electrodes,
            mean_rates,
            kde_models,
            jnp.moveaxis(multiunits, 2, 0),
        ):
            is_decoding_spike = jnp.any(~jnp.isnan(decoding_multiunits), axis=1)
            decoding_spike_waveform_features = decoding_multiunits[is_decoding_spike]

            position_distance = kde_distance(
                position,
                encoding_positions,
                std=position_std,
            )

            log_likelihood = log_likelihood.at[is_decoding_spike].set(
                block_estimate_log_joint_mark_intensity(
                    decoding_spike_waveform_features,
                    encoding_spike_waveform_features,
                    waveform_std,
                    interior_occupancy,
                    mean_rate,
                    position_distance,
                    block_size,
                )
            )
            log_likelihood -= kde_model.predict(position)
    else:
        interior_occupancy = occupancy[is_track_interior]
        interior_place_bin_centers = place_bin_centers[is_track_interior]
        is_track_interior_ind = jnp.nonzero(is_track_interior)[0]

        log_likelihood = -summed_ground_process_intensity * jnp.ones(
            (n_time, 1),
        )

        for (
            encoding_spike_waveform_features,
            encoding_positions,
            mean_rate,
            decoding_multiunits,
        ) in zip(
            tqdm(
                encoding_spike_waveform_features_electrodes,
                unit="electrode",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_position_electrodes,
            mean_rates,
            jnp.moveaxis(multiunits, 2, 0),
        ):
            decoding_spike_ind = jnp.nonzero(
                jnp.any(~jnp.isnan(decoding_multiunits), axis=1)
            )[0]
            decoding_spike_waveform_features = decoding_multiunits[decoding_spike_ind]

            position_distance = kde_distance(
                interior_place_bin_centers,
                encoding_positions,
                std=position_std,
            )

            log_likelihood = log_likelihood.at[
                jnp.ix_(decoding_spike_ind, is_track_interior_ind)
            ].set(
                block_estimate_log_joint_mark_intensity(
                    decoding_spike_waveform_features,
                    encoding_spike_waveform_features,
                    waveform_std,
                    interior_occupancy,
                    mean_rate,
                    position_distance,
                    block_size,
                )
            )

    return log_likelihood
