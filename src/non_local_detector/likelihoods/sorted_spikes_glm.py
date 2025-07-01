"""Poisson Generalized Linear Model (GLM) for Sorted Spikes using Spatial Splines.

This module provides functions to fit and predict neural activity using a
Generalized Linear Model (GLM) assuming Poisson firing statistics. It is
specifically designed for **sorted spikes**, where each spike train corresponds
to a distinct, pre-identified neural unit.

The core idea is to model the firing rate of each neuron as a function of the
animal's position. This relationship (the place field) is captured flexibly
using B-splines defined over the spatial dimensions. The model assumes:
  `spike_count ~ Poisson(rate)`
  `log(rate) = design_matrix @ coefficients`
where the `design_matrix` is constructed using spatial spline basis functions
derived from the animal's position, and `coefficients` are parameters fitted
to the data.

Key functionalities include:
1.  **Spline Basis Generation:**
    - `make_spline_design_matrix`: Creates the design matrix for fitting using
      `patsy`, defining cyclic cubic regression splines based on position data
      and specified knot spacing.
    - `make_spline_predict_matrix`: Generates the corresponding matrix for new
      positions based on the fitted model's design information, ensuring
      consistent basis functions for prediction.

2.  **Model Fitting:**
    - `fit_poisson_regression`: Fits the Poisson GLM coefficients for a *single*
      neuron using its spike counts and the design matrix. It employs L2
      regularization and optimizes the Poisson log-likelihood using SciPy's
      BFGS optimizer, leveraging JAX for automatic differentiation.
    - `fit_sorted_spikes_glm_encoding_model`: Orchestrates the fitting process
      for *all neurons*. It iterates through each neuron's spike train, calls
      `fit_poisson_regression`, computes the resulting place field (expected
      firing rate across space), and aggregates the results.

3.  **Likelihood Prediction:**
    - `predict_sorted_spikes_glm_log_likelihood`: Calculates the log-likelihood
      of observing spike trains during a *decoding* period, given the fitted
      GLM. It uses the Poisson log-likelihood formula:
      `sum_{neurons} [ k * log(lambda) - lambda ]`
      where `k` is the observed spike count and `lambda` is the predicted rate
      from the model. It supports both:
        - **Non-local decoding:** Computing likelihood across all spatial bins.
        - **Local decoding:** Computing likelihood only at the animal's
          interpolated position at each time point.

This module integrates with the `non_local_detector.environment` for spatial
context and uses common helper functions for spike counting and position
interpolation.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from patsy import build_design_matrices, dmatrix
from patsy.design_info import DesignInfo
from scipy.optimize import minimize
from tqdm.autonotebook import tqdm

from non_local_detector.environment.environment import Environment
from non_local_detector.environment.layout.helpers.utils import get_n_bins
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
    knot_spacing : float, optional
        Spacing of spline knots

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

    n_time = design_matrix.shape[0]

    @jax.jit
    def neglogp(
        coefficients, spikes=spikes, design_matrix=design_matrix, weights=weights
    ):
        conditional_intensity = jnp.exp(design_matrix @ coefficients)
        conditional_intensity = jnp.clip(conditional_intensity, a_min=EPS, a_max=None)
        log_likelihood_term = weights * (
            jax.scipy.special.xlogy(spikes, conditional_intensity)
            - conditional_intensity
        )
        mean_neg_log_likelihood = -jnp.sum(log_likelihood_term) / n_time
        l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)
        return mean_neg_log_likelihood + l2_penalty_term

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
        tol=1e-5,  # Added tolerance for potentially better convergence
    )

    return jnp.asarray(res.x)


def fit_sorted_spikes_glm_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    environment: Environment,
    place_bin_edges: np.ndarray,
    edges: np.ndarray,
    is_track_boundary: np.ndarray,
    weights: Optional[np.ndarray] = None,
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
        coefficients : jnp.ndarray, shape (n_neurons, n_coefficients)
            Fitted coefficients for each neuron.
        emission_design_info : patsy.design_info.DesignInfo
            DesignInfo object for the spline basis.
        place_fields : jnp.ndarray, shape (n_neurons, n_bins)
            Place fields for each neuron.
        no_spike_part_log_likelihood : jnp.ndarray, shape (n_bins,)
            Contribution to the log likelihood from no spikes.
        disable_progress_bar : bool
            If True, suppresses the progress bar display.

    """
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    time_range = (position_time[0], position_time[-1])
    n_time_bins = int(np.ceil((time_range[-1] - time_range[0]) * sampling_frequency))
    time = time_range[0] + np.arange(n_time_bins) / sampling_frequency

    if environment.is_1d:
        # convert to 1D
        interior_bin_centers = environment.to_linear(environment.bin_centers)[:, None]
    else:
        interior_bin_centers = environment.bin_centers

    interior_bin_centers = jnp.asarray(interior_bin_centers)

    emission_design_matrix = make_spline_design_matrix(
        position, place_bin_edges, knot_spacing=emission_knot_spacing
    )
    emission_design_info = emission_design_matrix.design_info
    emission_design_matrix = jnp.asarray(emission_design_matrix)

    emission_predict_matrix = make_spline_predict_matrix(
        emission_design_info, interior_bin_centers
    )
    if weights is None:
        weights = jnp.ones((position.shape[0],))

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
        place_fields.append(
            jnp.clip(
                jnp.exp(emission_predict_matrix @ coef),
                a_min=EPS,
                a_max=None,
            )
        )

    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "coefficients": jnp.stack(coefficients, axis=0),
        "emission_design_info": emission_design_info,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
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
    disable_progress_bar: bool = False,
    is_local: bool = False,
) -> jnp.ndarray:
    """Predict the log likelihood of spikes given a fitted GLM encoding model.

    Calculates the log likelihood of observing the given spike times under
    either a non-local (over spatial bins) or local (at the animal's current
    position) GLM model.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
        Time bins for decoding likelihood.
    position_time : jnp.ndarray, shape (n_time_position,)
        Timestamps corresponding to the position data.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position data of the animal.
    spike_times : list[np.ndarray]
        List where each element is an array of spike times for a single neuron.
    environment : Environment
        The spatial environment object containing track geometry information.
    coefficients : jnp.ndarray, shape (n_neurons, n_coefficients)
        Fitted GLM coefficients for each neuron.
    emission_design_info : patsy.design_info.DesignInfo
        Patsy DesignInfo object used for creating the spline design matrix
        during encoding, needed for prediction.
    place_fields : jnp.ndarray, shape (n_neurons, n_position_bins)
        Expected firing rate for each neuron in each position bin, derived
        from the fitted GLM (`exp(predict_matrix @ coefficients)`).
    no_spike_part_log_likelihood : jnp.ndarray, shape (n_position_bins,)
        The contribution to the log likelihood from the possibility of no spikes
        occurring in a time bin, summed across neurons (`sum(place_fields)`).
    disable_progress_bar : bool, optional
        If True, suppresses the progress bar display. By default False.
    is_local : bool, optional
        If True, compute the log likelihood only at the animal's current
        interpolated position (local decoding). If False, compute the log
        likelihood across all position bins (non-local decoding).
        By default False.

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
                spike_times,
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
        n_interior_bins = environment.n_bins
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
                jnp.expand_dims(place_field, axis=0),
            )

        log_likelihood -= no_spike_part_log_likelihood

    return log_likelihood
