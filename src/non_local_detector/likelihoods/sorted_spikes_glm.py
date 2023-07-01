import jax
import jax.numpy as jnp
import numpy as np
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from tqdm.autonotebook import tqdm

from non_local_detector.environment import get_n_bins

EPS = 1e-15


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

    inner_knots = np.meshgrid(*inner_knots)

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
    position: np.ndarray,
    spikes: np.ndarray,
    place_bin_centers: np.ndarray,
    place_bin_edges: np.ndarray,
    edges: np.ndarray,
    is_track_interior: np.ndarray,
    is_track_boundary: np.ndarray,
    emission_knot_spacing: float = 10.0,
    l2_penalty: float = 1e-3,
    disable_progress_bar: bool = False,
):
    emission_design_matrix = make_spline_design_matrix(
        position, place_bin_edges, knot_spacing=emission_knot_spacing
    )
    emission_design_info = emission_design_matrix.design_info
    emission_design_matrix = jnp.asarray(emission_design_matrix)

    emission_predict_matrix = make_spline_predict_matrix(
        emission_design_info, place_bin_centers
    )

    position = jnp.asarray(position)
    spikes = jnp.asarray(spikes)
    place_bin_centers = jnp.asarray(place_bin_centers[is_track_interior])
    weights = jnp.ones((spikes.shape[0],))

    coefficients = []
    place_fields = []

    for neuron_spikes in tqdm(
        spikes.T, unit="cell", desc="Encoding models", disable=disable_progress_bar
    ):
        coef = fit_poisson_regression(
            emission_design_matrix,
            neuron_spikes,
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
    position: np.ndarray,
    spikes: np.ndarray,
    coefficients: jnp.ndarray,
    emission_design_info,
    place_fields: jnp.ndarray,
    no_spike_part_log_likelihood: jnp.ndarray,
    is_track_interior: np.ndarray,
    disable_progress_bar: bool = False,
    is_local: bool = False,
):
    n_time = spikes.shape[0]
    position = jnp.asarray(position)
    spikes = jnp.asarray(spikes)

    if is_local:
        log_likelihood = jnp.zeros((n_time,))
        emission_predict_matrix = make_spline_predict_matrix(
            emission_design_info, position
        )
        for neuron_spikes, coef in zip(
            tqdm(
                spikes.T,
                unit="cell",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            coefficients,
        ):
            local_rate = jnp.exp(emission_predict_matrix @ coef)
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
            is_track_interior[jnp.newaxis, :], log_likelihood, jnp.log(EPS)
        )

    return log_likelihood
