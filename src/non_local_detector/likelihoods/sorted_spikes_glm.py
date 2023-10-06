import jax
import jax.numpy as jnp
import numpy as np
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from tqdm.autonotebook import tqdm

from non_local_detector.environment import Environment, get_n_bins
from non_local_detector.likelihoods.common import (
    EPS,
    get_position_at_time,
    get_spikecount_per_time_bin,
)


def make_spline_design_matrix(
    position: np.ndarray, place_bin_edges: np.ndarray, knot_spacing: float = 10.0
):
    position = position if position.ndim > 1 else position[:, np.newaxis]
    inner_knots = []
    for pos, edges in zip(position.T, place_bin_edges.T):
        n_points = get_n_bins(edges, bin_size=knot_spacing)
        knots = np.linspace(edges.min(), edges.max(), n_points)[1:-1]
        knots = knots[(knots > pos.min()) & (knots < pos.max())]
        inner_knots.append(knots)

    inner_knots = np.meshgrid(*inner_knots, indexing="ij")

    data = {}
    formula = "1 + te("
    for ind in range(position.shape[1]):
        formula += f"cr(x{ind}, knots=inner_knots[{ind}])"
        formula += ", "
        data[f"x{ind}"] = position[:, ind]

    formula += 'constraints="center")'
    return dmatrix(formula, data)


def make_spline_predict_matrix(design_info, position: jnp.ndarray) -> jnp.ndarray:
    position = jnp.asarray(position)
    is_nan = jnp.any(jnp.isnan(position), axis=1)
    position = jnp.where(is_nan[:, jnp.newaxis], 0.0, position)

    predict_data = {}
    for ind in range(position.shape[1]):
        predict_data[f"x{ind}"] = position[:, ind]

    design_matrix = build_design_matrices([design_info], predict_data)[0]
    design_matrix[is_nan] = np.nan

    return jnp.asarray(design_matrix)


def fit_poisson_regression(
    design_matrix: np.ndarray,
    spikes: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float = 1e-7,
) -> jnp.ndarray:
    @jax.jit
    def neglogp(
        coefficients, spikes=spikes, design_matrix=design_matrix, weights=weights
    ):
        conditional_intensity = jnp.exp(design_matrix @ coefficients)
        conditional_intensity = jnp.clip(conditional_intensity, a_min=EPS, a_max=None)
        negative_log_likelihood = -1.0 * jnp.mean(
            weights * jax.scipy.stats.poisson.logpmf(spikes, conditional_intensity)
        )
        l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)
        return negative_log_likelihood + l2_penalty_term

    dlike = jax.grad(neglogp)

    initial_condition = jnp.asarray([jnp.log(jnp.average(spikes, weights=weights))])
    initial_condition = jnp.concatenate(
        [initial_condition, jnp.zeros(design_matrix.shape[1] - 1)]
    )

    res = minimize(
        neglogp,
        x0=initial_condition,
        method="BFGS",
        jac=dlike,
    )

    return jnp.asarray(res.x)


def fit_sorted_spikes_glm_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    environment: Environment,
    place_bin_edges: np.ndarray,
    edges: np.ndarray,
    is_track_interior: np.ndarray,
    is_track_boundary: np.ndarray,
    emission_knot_spacing: float = 10.0,
    l2_penalty: float = 1e-3,
    disable_progress_bar: bool = False,
    sampling_frequency: float = 500.0,
):
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    time_range = (position_time[0], position_time[-1])
    n_time_bins = int(np.ceil((time_range[-1] - time_range[0]) * sampling_frequency))
    time = time_range[0] + np.arange(n_time_bins) / sampling_frequency

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = jnp.asarray(
        environment.place_bin_centers_[is_track_interior]
    )

    emission_design_matrix = make_spline_design_matrix(
        position, place_bin_edges, knot_spacing=emission_knot_spacing
    )
    emission_design_info = emission_design_matrix.design_info
    emission_design_matrix = jnp.asarray(emission_design_matrix)

    emission_predict_matrix = make_spline_predict_matrix(
        emission_design_info, interior_place_bin_centers
    )

    weights = jnp.ones((position_time.shape[0],))

    coefficients = []
    place_fields = []

    for neuron_spike_times in tqdm(
        spike_times,
        unit="cell",
        desc="Encoding models",
        disable=disable_progress_bar,
    ):
        spike_count_per_time_bin = get_spikecount_per_time_bin(neuron_spike_times, time)
        coef = fit_poisson_regression(
            emission_design_matrix,
            spike_count_per_time_bin,
            weights,
            l2_penalty=l2_penalty,
        )
        coefficients.append(coef)
        place_field = jnp.zeros((is_track_interior.shape[0],))
        place_fields.append(
            place_field.at[is_track_interior].set(
                jnp.clip(
                    jnp.exp(emission_predict_matrix @ coef),
                    a_min=EPS,
                    a_max=None,
                )
            )
        )

    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "coefficients": jnp.stack(coefficients, axis=0),
        "emission_design_info": emission_design_info,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_sorted_spikes_glm_log_likelihood(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    coefficients: jnp.ndarray,
    emission_design_info,
    place_fields: jnp.ndarray,
    no_spike_part_log_likelihood: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    disable_progress_bar: bool = False,
    is_local: bool = False,
):
    n_time = time.shape[0]

    if is_local:
        log_likelihood = jnp.zeros((n_time,))
        log_likelihood = jnp.zeros((n_time,))

        # Need to interpolate position
        interpolated_position = get_position_at_time(
            position_time, position, time, environment
        )
        emission_predict_matrix = make_spline_predict_matrix(
            emission_design_info, interpolated_position
        )
        for neuron_spike_times, coef in zip(
            tqdm(
                spike_times.T,
                unit="cell",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            coefficients,
        ):
            spike_count_per_time_bin = get_spikecount_per_time_bin(spike_times, time)
            local_rate = jnp.exp(emission_predict_matrix @ coef)
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
