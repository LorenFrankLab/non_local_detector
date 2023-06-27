from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import scipy.interpolate
from tqdm.autonotebook import tqdm

EPS = 1e-15


@jax.jit
def gaussian_pdf(x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma."""
    return jnp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def kde(eval_points, samples, std):
    def _gaussian_distance(distance, x):
        eval, samp, std = x
        distance *= gaussian_pdf(
            jnp.expand_dims(eval, axis=0),
            jnp.expand_dims(samp, axis=1),
            std,
        )
        return (distance, None)

    distance, _ = jax.lax.scan(
        _gaussian_distance,
        init=jnp.ones((samples.shape[0], eval_points.shape[0])),
        xs=(eval_points.T, samples.T, std),
    )

    return jnp.mean(distance, axis=0)


@partial(jax.jit, static_argnums=(1,))
def reshape_to_blocks(eval_points, block_size=100):
    n_points, n_dim = eval_points.shape
    remainder = n_points % block_size
    n_blocks = n_points // block_size

    if remainder > 0:
        n_pad = block_size - remainder
        n_blocks += 1
        return jnp.concatenate(
            (eval_points, jnp.zeros((n_pad, n_dim))), axis=0
        ).reshape(n_blocks, block_size, n_dim)
    else:
        return eval_points.reshape(n_blocks, block_size, n_dim)


@partial(jax.jit, static_argnums=(3,))
def block_kde(eval_points, samples, std, block_size=100):
    return jax.lax.map(
        jax.vmap(partial(kde, samples=samples, std=std)),
        reshape_to_blocks(eval_points, block_size=block_size),
    ).reshape((-1,))[: eval_points.shape[0]]


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


def fit_sorted_spikes_kde_encoding_model2(
    position: jnp.ndarray,
    spikes: jnp.ndarray,
    place_bin_centers: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    *args,
    position_std: float = 5.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    **kwargs,
):
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(position)
    occupancy = occupancy_model.predict(place_bin_centers[is_track_interior])
    mean_rates = jnp.mean(spikes, axis=0).squeeze()

    place_fields = []
    kde_models = []

    for neuron_spikes, mean_rate in zip(
        tqdm(
            spikes.T.astype(bool),
            unit="cell",
            desc="Encoding models",
            disable=disable_progress_bar,
        ),
        mean_rates,
    ):
        kde_model = KDEModel(std=position_std, block_size=block_size).fit(
            position[neuron_spikes]
        )
        kde_models.append(kde_model)
        marginal_density = kde_model.predict(place_bin_centers[is_track_interior])
        place_field = jnp.zeros((is_track_interior.shape[0],))
        place_fields.append(
            place_field.at[is_track_interior].set(
                jnp.clip(
                    mean_rate
                    * jnp.where(occupancy > 0.0, marginal_density / occupancy, EPS),
                    a_min=EPS,
                    a_max=None,
                )
            )
        )

    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "kde_models": kde_models,
        "occupancy_model": occupancy_model,
        "occupancy": occupancy,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "place_bin_centers": place_bin_centers,
        "is_track_interior": is_track_interior,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_sorted_spikes_kde_log_likelihood2(
    position: jnp.ndarray,
    spikes: jnp.ndarray,
    kde_models: list[KDEModel],
    occupancy_model: KDEModel,
    occupancy: jnp.ndarray,
    mean_rates: jnp.ndarray,
    place_fields: jnp.ndarray,
    no_spike_part_log_likelihood: jnp.ndarray,
    place_bin_centers: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    disable_progress_bar: bool = False,
    is_local: bool = False,
    interpolate_local: bool = False,
):
    n_time = spikes.shape[0]
    if is_local:
        log_likelihood = jnp.zeros((n_time,))

        if interpolate_local:
            for neuron_spikes, non_local_rate in zip(tqdm(spikes.T), place_fields.T):
                local_rate = scipy.interpolate.griddata(
                    place_bin_centers, non_local_rate, position, method="nearest"
                )
                local_rate = jnp.clip(local_rate, a_min=EPS, a_max=None)
                log_likelihood += (
                    jax.scipy.special.xlogy(neuron_spikes, local_rate) - local_rate
                )
        else:
            occupancy = occupancy_model.predict(position)

            for neuron_spikes, kde_model, mean_rate in zip(
                tqdm(
                    spikes.T,
                    unit="cell",
                    desc="Local Likelihood",
                    disable=disable_progress_bar,
                ),
                kde_models,
                mean_rates,
            ):
                marginal_density = kde_model.predict(position)
                local_rate = mean_rate * jnp.where(
                    occupancy > 0.0, marginal_density / occupancy, EPS
                )
                local_rate = jnp.clip(local_rate, a_min=EPS, a_max=None)
                log_likelihood += (
                    jax.scipy.special.xlogy(neuron_spikes, local_rate) - local_rate
                )

        log_likelihood = jnp.expand_dims(log_likelihood, axis=1)
    else:
        log_likelihood = jnp.zeros((n_time, place_fields.shape[1]))
        for neuron_spikes, place_field in zip(
            tqdm(
                spikes.T,
                unit="cell",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            place_fields,
        ):
            log_likelihood += jax.scipy.special.xlogy(
                neuron_spikes[:, jnp.newaxis], place_field[jnp.newaxis]
            )
        log_likelihood -= no_spike_part_log_likelihood[jnp.newaxis]
        log_likelihood = jnp.where(
            is_track_interior[jnp.newaxis, :], log_likelihood, EPS
        )

    return log_likelihood
