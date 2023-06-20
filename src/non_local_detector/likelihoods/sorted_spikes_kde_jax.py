import jax.numpy as jnp
import jax.scipy
from tqdm.autonotebook import tqdm

EPS = 1e-15


def fit_sorted_spikes_kde_jax_encoding_model(
    position: jnp.array,
    spikes: jnp.array,
    place_bin_centers: jnp.array,
    position_std: float = 5.0,
):
    log_occupancy = jax.scipy.stats.gaussian_kde(
        position.squeeze(), bw_method=position_std
    ).logpdf(position.squeeze())
    log_mean_rate = jnp.log(jnp.mean(spikes, axis=0, keepdims=True).T)

    place_fields = []
    kde_models = []

    for neuron_ind, is_spike in enumerate(spikes.T.astype(bool)):
        kde_models.append(
            jax.scipy.stats.gaussian_kde(
                position[is_spike].squeeze(), bw_method=position_std
            )
        )
        log_marginal_density = kde_models[-1].pdf(place_bin_centers.squeeze())
        place_fields.append(
            jnp.exp(log_mean_rate[neuron_ind] + log_marginal_density - log_occupancy)
        )

    place_fields = jnp.stack(place_fields, axis=0)

    return {
        "kde_models": kde_models,
        "log_occupancy": log_occupancy,
        "log_mean_rate": log_mean_rate,
        "place_fields": place_fields,
    }


def get_firing_rate(position, kde_models, log_occupancy, log_mean_rate):
    log_marginal_density = jnp.stack(
        [kde_model.logpdf(position.squeeze()) for kde_model in kde_models],
        axis=0,
    )

    return log_mean_rate + log_marginal_density - log_occupancy


def predict_sorted_spikes_kde_jax_log_likelihood(
    position: jnp.ndarray,
    spikes: jnp.ndarray,
    kde_models,
    log_occupancy,
    log_mean_rate,
    place_fields,
    is_track_interior: jnp.ndarray,
    is_local: bool = False,
):
    n_time = spikes.shape[0]
    if is_local:
        log_likelihood = jnp.zeros((n_time,))
        for neuron_ind, (neuron_spikes, kde_model) in enumerate(
            zip(tqdm(spikes.T), kde_models)
        ):
            log_marginal_density = kde_model.logpdf(position.squeeze())
            local_rate = (
                log_mean_rate[neuron_ind] + log_marginal_density - log_occupancy
            )
            local_rate = jnp.clip(local_rate, a_min=EPS, a_max=None)
            log_likelihood += jax.scipy.stats.poisson.logpmf(neuron_spikes, local_rate)

        log_likelihood = log_likelihood[:, jnp.newaxis]
    else:
        log_likelihood = jnp.zeros((n_time, place_fields.shape[1]))
        for neuron_spikes, place_field in zip(tqdm(spikes.T), place_fields):
            log_likelihood += jax.scipy.stats.poisson.logpmf(
                neuron_spikes[:, jnp.newaxis], place_field[jnp.newaxis]
            )
        log_likelihood[:, ~is_track_interior] = jnp.nan

    return log_likelihood
