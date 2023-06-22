import jax.numpy as jnp
import jax
from tqdm.autonotebook import tqdm
from functools import partial
from dataclasses import dataclass


EPS = 1e-15


@jax.jit
def gaussian_pdf_jax(
    x: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma."""
    return jnp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))


@jax.jit
def kde_jax(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    return jnp.mean(
        jnp.prod(
            gaussian_pdf_jax(
                jnp.expand_dims(eval_points, axis=0),
                jnp.expand_dims(samples, axis=1),
                std,
            ),
            axis=-1,
        ),
        axis=0,
    ).squeeze()


@partial(jax.jit, static_argnums=(3,))
def block_kde_jax(
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
            kde_jax(eval_points[block_inds], samples, std).squeeze(),
            (start_ind,),
        )

    return density


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

        return block_kde_jax(eval_points, self.samples_, std, block_size)


def fit_sorted_spikes_kde_jax_encoding_model(
    position: jnp.ndarray,
    spikes: jnp.ndarray,
    place_bin_centers: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    *args,
    position_std: float = 5.0,
    block_size: int = 100,
    **kwargs,
):
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(position)
    occupancy = occupancy_model.predict(place_bin_centers)
    mean_rates = jnp.mean(spikes, axis=0).squeeze()

    place_fields = []
    kde_models = []

    for mean_rate, neuron_spikes in zip(mean_rates, tqdm(spikes.T.astype(bool))):
        kde_model = KDEModel(std=position_std, block_size=block_size).fit(
            position[neuron_spikes]
        )
        kde_models.append(kde_model)
        marginal_density = kde_model.predict(place_bin_centers)
        place_field = mean_rate * jnp.where(
            occupancy > 0.0, marginal_density / occupancy, 0.0
        )
        place_field = jnp.clip(place_field, a_min=EPS, a_max=None)
        place_fields.append(place_field)

    place_fields = jnp.stack(place_fields, axis=0)

    return {
        "kde_models": kde_models,
        "occupancy_model": occupancy_model,
        "occupancy": occupancy,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "is_track_interior": is_track_interior,
    }


def predict_sorted_spikes_kde_jax_log_likelihood(
    position: jnp.ndarray,
    spikes: jnp.ndarray,
    kde_models,
    occupancy_model,
    occupancy,
    mean_rates,
    place_fields,
    is_track_interior: jnp.ndarray,
    is_local: bool = False,
):
    n_time = spikes.shape[0]
    if is_local:
        log_likelihood = jnp.zeros((n_time,))

        occupancy = occupancy_model.predict(position)

        for neuron_spikes, kde_model, mean_rate in zip(
            tqdm(spikes.T), kde_models, mean_rates
        ):
            marginal_density = kde_model.predict(position)
            local_rate = mean_rate * jnp.where(
                occupancy > 0.0, marginal_density / occupancy, 0.0
            )
            local_rate = jnp.clip(local_rate, a_min=EPS, a_max=None)
            log_likelihood += jax.scipy.stats.poisson.logpmf(neuron_spikes, local_rate)

        log_likelihood = jnp.expand_dims(log_likelihood, axis=1)
    else:
        log_likelihood = jnp.zeros((n_time, place_fields.shape[1]))
        for neuron_spikes, place_field in zip(tqdm(spikes.T), place_fields):
            log_likelihood += jax.scipy.stats.poisson.logpmf(
                neuron_spikes[:, jnp.newaxis], place_field[jnp.newaxis]
            )
        log_likelihood.at[:, ~is_track_interior].set(jnp.nan)

    return log_likelihood
