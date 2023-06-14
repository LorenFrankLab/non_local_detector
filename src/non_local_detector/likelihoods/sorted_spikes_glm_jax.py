import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from tqdm.autonotebook import tqdm

from non_local_detector.core import atleast_2d
from non_local_detector.environment import get_n_bins

EPS = 1e-15


def make_spline_design_matrix(position, place_bin_edges, knot_spacing=10):
    position = atleast_2d(position)
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


def make_spline_predict_matrix(design_info, position):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position[is_nan] = 0.0

    predict_data = {}
    for ind in range(position.shape[1]):
        predict_data[f"x{ind}"] = position[:, ind]

    design_matrix = build_design_matrices([design_info], predict_data)[0]
    design_matrix[is_nan] = np.nan

    return design_matrix


def fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=0.0):
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

    initial_condition = np.array([np.log(np.average(spikes, weights=weights))])
    initial_condition = np.concatenate(
        [initial_condition, np.zeros(design_matrix.shape[1] - 1)]
    )

    res = minimize(
        neglogp,
        x0=initial_condition,
        method="BFGS",
        jac=dlike,
    )

    return res.x


def fit_sorted_spikes_glm_jax_encoding_model(
    position,
    spikes,
    place_bin_centers,
    place_bin_edges,
    edges,
    is_track_interior,
    is_track_boundary,
    emission_knot_spacing=10,
    l2_penalty=1e-3,
):
    emission_design_matrix = make_spline_design_matrix(
        position, place_bin_edges, knot_spacing=emission_knot_spacing
    )

    weights = np.ones((spikes.shape[0],), dtype=np.float32)

    coefficients = []
    for neuron_spikes in tqdm(spikes.T):
        coef = fit_poisson_regression(
            emission_design_matrix,
            neuron_spikes,
            weights,
            l2_penalty=l2_penalty,
        )
        coefficients.append(coef)

    return {
        "coefficients": np.stack(coefficients, axis=0),
        "emission_design_matrix": emission_design_matrix,
        "place_bin_centers": place_bin_centers,
        "is_track_interior": is_track_interior,
    }


def predict_sorted_spikes_glm_jax_log_likelihood(
    position,
    spikes,
    coefficients,
    emission_design_matrix,
    place_bin_centers,
    is_track_interior,
    is_local: bool = False,
):
    n_time = spikes.shape[0]
    if is_local:
        log_likelihood = np.zeros((n_time,))
        emission_predict_matrix = make_spline_predict_matrix(
            emission_design_matrix.design_info, position
        )
        for neuron_spikes, coef in zip(tqdm(spikes.T), coefficients):
            local_rate = np.exp(emission_predict_matrix @ coef)
            local_rate = np.clip(local_rate, a_min=EPS, a_max=None)
            log_likelihood += scipy.stats.poisson.logpmf(neuron_spikes, local_rate)

        log_likelihood = log_likelihood[:, np.newaxis]
    else:
        emission_predict_matrix = make_spline_predict_matrix(
            emission_design_matrix.design_info, place_bin_centers
        )
        log_likelihood = np.zeros((n_time, len(place_bin_centers)))
        for neuron_spikes, coef in zip(tqdm(spikes.T), coefficients):
            non_local_rate = np.exp(emission_predict_matrix @ coef)
            non_local_rate[~is_track_interior] = EPS
            non_local_rate = np.clip(non_local_rate, a_min=EPS, a_max=None)
            log_likelihood += scipy.stats.poisson.logpmf(
                neuron_spikes[:, np.newaxis], non_local_rate[np.newaxis]
            )
        log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood
