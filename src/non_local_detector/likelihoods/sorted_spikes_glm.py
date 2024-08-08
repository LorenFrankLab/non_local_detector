import jax
import jax.numpy as jnp
import numpy as np
from patsy import build_design_matrices, dmatrix
from patsy.design_info import DesignInfo
from scipy.optimize import minimize
from tqdm.autonotebook import tqdm

from non_local_detector.environment import Environment, get_n_bins
from non_local_detector.likelihoods.common import (
    EPS,
    get_position_at_time,
    get_spikecount_per_time_bin,
)


def make_spline_design_matrix(
    position: np.ndarray,
    place_bin_edges: np.ndarray,
    knot_spacing: float = np.sqrt(12.5) * 2,
) -> np.ndarray:
    """Create a design matrix for a spline basis.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
    place_bin_edges : np.ndarray, shape (n_bins,)
    knot_spacing : float, shape (n_bins,), optional
        Spacing of spline knots, by default 10.0

    Returns
    -------
    design_matrix : np.ndarray, shape (n_time, n_spline_basis)
    """
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


def make_spline_predict_matrix(
    design_info: DesignInfo, position: jnp.ndarray
) -> jnp.ndarray:
    """Create a prediction matrix for a spline basis.

    Parameters
    ----------
    design_info : patsy.design_info.DesignInfo
    position : jnp.ndarray, shape (n_position_bins, n_position_dims)

    Returns
    -------
    jnp.ndarray, shape (n_position_bins, n_spline_basis)
    """
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
    """Fit a Poisson regression model.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    spikes : np.ndarray, shape (n_time,)
    weights : np.ndarray, shape (n_time,)
    l2_penalty : float, optional
        L2 regression penalty, by default 1e-7

    Returns
    -------
    coefficients : jnp.ndarray, shape (n_coefficients,)
    """

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
    emission_knot_spacing: float = np.sqrt(12.5) * 2,
    l2_penalty: float = 1e-3,
    disable_progress_bar: bool = False,
    sampling_frequency: float = 500.0,
) -> dict:
    """Fit a GLM encoding model

    Parameters
    ----------
    position_time : jnp.ndarray, shape (n_time_position,)
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
    spike_times : list[jnp.ndarray]
        Spike times for each neuron.
    environment : Environment
        The spatial environment.
    place_bin_edges : np.ndarray, shape (n_bins + 1,)
        The edges of the place bins.
    edges : np.ndarray, shape (n_edges, 2)
        The edges of the place bins.
    is_track_interior : np.ndarray, shape (n_position_bins,)
        Identifies if the bin is on the track interior.
    is_track_boundary : np.ndarray, shape (n_position_bins,)
        Identifies if the bin is on the track boundary.
    emission_knot_spacing : float, optional
        Knots over position, by default 10.0
    l2_penalty : float, optional
        L2 penalty for regression, by default 1e-3
    disable_progress_bar : bool, optional
        Turn off the progress bars, by default False
    sampling_frequency : float, optional
        Samples per second, by default 500.0

    Returns
    -------
    encoding_model : dict
    """
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
    emission_design_info: DesignInfo,
    place_fields: jnp.ndarray,
    no_spike_part_log_likelihood: jnp.ndarray,
    is_track_interior: jnp.ndarray,
    disable_progress_bar: bool = False,
    is_local: bool = False,
) -> jnp.ndarray:
    """Predict the log likelihood of spikes given the model.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
        Decoded time bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time bins for position.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position data.
    spike_times : list[np.ndarray]
        Spike times for each neuron.
    environment : Environment
        The spatial environment.
    coefficients : jnp.ndarray, shape (n_neurons, n_coefficients)
        Coefficients for each neuron.
    emission_design_info : patsy.design_info.DesignInfo
        _description_
    place_fields : jnp.ndarray
        _description_
    no_spike_part_log_likelihood : jnp.ndarray
        _description_
    is_track_interior : jnp.ndarray
        _description_
    disable_progress_bar : bool, optional
        _description_, by default False
    is_local : bool, optional
        _description_, by default False

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, n_bins)
    """
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
