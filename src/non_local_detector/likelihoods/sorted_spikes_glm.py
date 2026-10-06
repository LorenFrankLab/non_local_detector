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
      neuron using its weighted spike counts, per-sample exposure weights, and
      the design matrix. It employs L2
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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from patsy import build_design_matrices, dmatrix  # type: ignore[import-untyped]
from patsy.design_info import DesignInfo  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment, get_n_bins
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    RATE_EPS_HZ,
    _SpikeTimeOrder,
    decode_bin_centers,
    get_position_at_time,
    get_spikecount_per_time_bin,
    resolve_row_slice,
    validate_population_lengths,
    validate_weights,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
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
    varying_dimensions = np.flatnonzero(np.ptp(position, axis=0) > 0)
    if not len(varying_dimensions):
        return dmatrix("1", pd.DataFrame(index=np.arange(len(position))))
    inner_knots = []
    for ind in varying_dimensions:
        pos, edges = position[:, ind], place_bin_edges[:, ind]
        n_points = get_n_bins(edges, bin_size=knot_spacing)
        knots = np.linspace(edges.min(), edges.max(), n_points)[1:-1]
        knots = knots[(knots > pos.min()) & (knots < pos.max())]
        inner_knots.append(knots)

    inner_knots = np.meshgrid(*inner_knots, indexing="ij")

    data = {}
    formula = "1 + te("
    for knot_ind, ind in enumerate(varying_dimensions):
        formula += f"cr(x{ind}, knots=inner_knots[{knot_ind}])"
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

    data = predict_data if design_info.factor_infos else pd.DataFrame(predict_data)
    design_matrix = build_design_matrices([design_info], data)[0]
    design_matrix[is_nan] = np.nan

    return jnp.asarray(design_matrix)


# Gradient inf-norm above which a BFGS fit is treated as genuinely
# non-converged. Set well above the optimizer's own gtol (1e-5) so that the
# common benign "precision loss" exit (success=False with a near-zero
# gradient) does not warn.
GLM_CONVERGENCE_GRAD_TOL = 1e-3


def fit_poisson_regression(
    design_matrix: np.ndarray,
    spikes: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float = 1e-7,
    *,
    rate_floor: float = EPS,
) -> jnp.ndarray:
    """Fit a weighted Poisson regression model.

    Maximizes ``sum_i [spikes_i * log(lambda_i) - weights_i * lambda_i]``
    (normalized by ``sum(weights)``) minus the L2 penalty. The event and
    exposure terms are weighted separately: ``spikes`` already carries each
    event's weight, so it is not multiplied by ``weights`` again.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    spikes : np.ndarray, shape (n_time,)
        Event mass per row: the weighted spike counts from
        ``weighted_spike_counts``, not multiplied by ``weights`` again.
    weights : np.ndarray, shape (n_time,)
        Per-row exposure weight.
    l2_penalty : float, optional
        L2 regression penalty, by default 1e-7

    Returns
    -------
    coefficients : jnp.ndarray, shape (n_coefficients,)
    """

    spikes = jnp.asarray(spikes, dtype=jnp.float32)
    design_matrix = jnp.asarray(design_matrix, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)

    @jax.jit
    def neglogp(
        coefficients, spikes=spikes, design_matrix=design_matrix, weights=weights
    ):
        conditional_intensity = jnp.exp(design_matrix @ coefficients)
        conditional_intensity = jnp.clip(
            conditional_intensity, min=rate_floor, max=None
        )
        log_likelihood_term = (
            jax.scipy.special.xlogy(spikes, conditional_intensity)
            - weights * conditional_intensity
        )
        # Normalize by sum of weights (not n_time) so the data term
        # scales correctly when local state mass is small, keeping
        # the regularizer balanced relative to the effective data.
        weight_sum = jnp.maximum(jnp.sum(weights), EPS)
        mean_neg_log_likelihood = -jnp.sum(log_likelihood_term) / weight_sum
        l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)
        return mean_neg_log_likelihood + l2_penalty_term

    dlike = jax.grad(neglogp)

    # Zero total exposure means this group has no training coverage at all, so
    # the rate is unidentified and ``sum(spikes) / sum(weights)`` is 0/0 = NaN. Return
    # an intercept-only model at the EPS floor, which is exactly what a unit with
    # real exposure but no spikes already gets from the ``maximum(avg_rate, EPS)``
    # guard below -- the two zero-rate cases agree. A group with small-but-
    # positive exposure (a low-mass EM update) is a different case and still
    # fits normally.
    if float(jnp.sum(weights)) <= 0.0:
        return jnp.concatenate(
            [
                jnp.asarray([jnp.log(rate_floor)]),
                jnp.zeros(design_matrix.shape[1] - 1),
            ]
        )

    avg_rate = jnp.sum(spikes) / jnp.sum(weights)
    # Guard against zero spikes: use EPS to avoid log(0) = -inf
    initial_condition = jnp.asarray([jnp.log(jnp.maximum(avg_rate, rate_floor))])
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

    # Judge convergence by the actual gradient inf-norm rather than SciPy's
    # boolean ``success`` flag: BFGS frequently reports ``success=False``
    # ("Desired error not necessarily achieved due to precision loss") even at
    # a good minimum, so warning on ``not res.success`` is routinely spurious.
    # Warn when the gradient is meaningfully far from zero, OR when the fit
    # diverged to a non-finite loss/gradient: for a diverged fit ``grad_norm``
    # is ``NaN`` and ``NaN > tol`` is ``False``, so a gradient-only check would
    # silently pass a broken fit.
    grad_norm = float(np.max(np.abs(res.jac))) if res.jac is not None else float("inf")
    final_loss = float(res.fun)
    if (
        not np.isfinite(grad_norm)
        or not np.isfinite(final_loss)
        or grad_norm > GLM_CONVERGENCE_GRAD_TOL
    ):
        warnings.warn(
            f"GLM Poisson regression may not have converged: final gradient "
            f"inf-norm {grad_norm:.3e} vs tolerance {GLM_CONVERGENCE_GRAD_TOL:.1e}, "
            f"final loss {final_loss:.6f} (a non-finite gradient or loss means the "
            f"fit diverged). Place-field coefficients may be unreliable. "
            f"(scipy message: {res.message}, n_iter={res.nit})",
            UserWarning,
            stacklevel=2,
        )

    return jnp.asarray(res.x)


def weighted_spike_counts(
    spike_times: np.ndarray, position_time: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Split each spike's interpolated weight between its bracketing samples.

    A spike a fraction ``a`` of the way from sample ``i`` to sample ``i + 1``
    adds ``(1 - a) * weights[i]`` to row ``i`` and ``a * weights[i + 1]`` to
    row ``i + 1``. The two parts sum to ``weights`` linearly interpolated to the
    spike time, so each spike keeps its canonical weight and complementary
    groups still partition it. Each part is carried by the sample that supplies
    it, so a row receives event mass only where its own exposure weight is
    positive; a row with events but no exposure would let the fitted rate at
    its position diverge. For a locally constant rate and uniform sampling, an
    interior row's expected event mass is ``rate * weights[i] * dt``,
    proportional to its exposure term. Spikes outside
    ``[position_time[0], position_time[-1]]`` are dropped.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    position_time : np.ndarray, shape (n_time_position,)
    weights : np.ndarray, shape (n_time_position,)

    Returns
    -------
    counts : np.ndarray, shape (n_time_position,)
        Event mass per position row.
    """
    position_time = np.asarray(position_time)
    weights = np.asarray(weights)
    spike_times = np.asarray(spike_times)
    n_time = position_time.shape[0]
    times = spike_times[
        (spike_times >= position_time[0]) & (spike_times <= position_time[-1])
    ]
    # Encoding rows are position samples, not decode bins: the left sample is
    # the last one at or before the spike, never the final sample, so a spike
    # at position_time[-1] splits into the final interval.
    left = np.searchsorted(position_time[1:-1], times, side="right")
    right = np.minimum(left + 1, n_time - 1)
    interval = position_time[right] - position_time[left]
    safe_interval = np.where(interval > 0.0, interval, 1.0)
    # A zero-length interval (a repeated final timestamp, or a single sample)
    # gives the spike the right sample's weight, as np.interp does.
    fraction = np.where(
        interval > 0.0, (times - position_time[left]) / safe_interval, 1.0
    )
    return np.bincount(
        left, weights=(1.0 - fraction) * weights[left], minlength=n_time
    ) + np.bincount(right, weights=fraction * weights[right], minlength=n_time)


def fit_sorted_spikes_glm_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    environment: Environment,
    place_bin_edges: np.ndarray,
    edges: np.ndarray,
    is_track_interior: np.ndarray,
    is_track_boundary: np.ndarray,
    weights: np.ndarray | None = None,
    emission_knot_spacing: float = np.sqrt(12.5) * 2,
    l2_penalty: float = 0.5,
    disable_progress_bar: bool = False,
    *,
    encoding_time_range=None,
    valid_position_intervals=None,
    _encoding_support=None,
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
    weights : np.ndarray, shape (n_time_position,), optional
        Sample weights for each position time point, by default None.
        If None, uniform weights are used.
    emission_knot_spacing : float, optional
        Knots over position, by default 10.0
    l2_penalty : float, optional
        L2 penalty per second, by default 0.5 (0.001 at the 500 Hz reference)
    disable_progress_bar : bool, optional
        Turn off the progress bars, by default False

    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds. Clips encoding support using the
        original interpolation basis. Uniform samples otherwise include endpoint
        half-cells; a singleton requires explicit bounds or a tracking interval.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
        Required for irregular timestamps; each interval needs a finite sample.
        Positions are held at segment endpoints and NaN rows split support.
        Encoding support does not automatically mark decode bins missing.

    Returns
    -------
    encoding_model : dict
        coefficients : jnp.ndarray, shape (n_neurons, n_coefficients)
            Fitted coefficients for each neuron.
        emission_design_info : patsy.design_info.DesignInfo
            DesignInfo object for the spline basis.
        place_fields : jnp.ndarray, shape (n_neurons, n_bins)
            Spatial firing rates in Hz for each neuron.
        no_spike_part_log_likelihood : jnp.ndarray, shape (n_bins,)
            Sum of Hz rates across neurons; multiply by duration before scoring.
        is_track_interior : jnp.ndarray, shape (n_bins,)
            Boolean array indicating track interior.
        disable_progress_bar : bool
            If True, suppresses the progress bar display.

    """
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    # Use position_time directly so spike counts, design matrix, and
    # weights all share the same time grid.

    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        if environment.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ is required when is_track_interior_ is None"
            )
        is_track_interior = jnp.ones(len(environment.place_bin_centers_), dtype=bool)
    interior_place_bin_centers = jnp.asarray(
        environment.place_bin_centers_[is_track_interior]
    )

    encoding_positions = position[support.indices]
    # A known recording duration can expose a stationary/singleton sample.
    # Define its spline on the environment, then retain only actual encoding
    # rows for the event and exposure terms. A one-center grid uses an intercept.
    basis_positions = (
        encoding_positions
        if len(encoding_positions) and np.all(np.ptp(encoding_positions, axis=0) > 0)
        else np.asarray(interior_place_bin_centers)
    )
    emission_design_matrix = make_spline_design_matrix(
        np.asarray(basis_positions),
        place_bin_edges,
        knot_spacing=emission_knot_spacing,
    )
    emission_design_info = emission_design_matrix.design_info
    emission_design_matrix = (
        jnp.asarray(emission_design_matrix)
        if basis_positions is encoding_positions
        else make_spline_predict_matrix(
            emission_design_info,
            encoding_positions
            if len(encoding_positions)
            else interior_place_bin_centers,
        )
    )

    emission_predict_matrix = make_spline_predict_matrix(
        emission_design_info, interior_place_bin_centers
    )
    if weights is None:
        weights = jnp.ones((position.shape[0],))
    else:
        weights = validate_weights(weights, position.shape[0])

    # Ensure weights is not None for type checking
    assert weights is not None

    coefficients = []
    place_fields = []

    for neuron_spike_times in tqdm(
        spike_times,
        unit="cell",
        desc="Encoding models",
        disable=disable_progress_bar,
    ):
        coef = fit_poisson_regression(
            emission_design_matrix,
            support.event_counts(neuron_spike_times, weights)[support.indices]
            if len(support.indices)
            else np.zeros(len(emission_design_matrix)),
            exposure_weights[support.indices]
            if len(support.indices)
            else np.zeros(len(emission_design_matrix)),
            l2_penalty=l2_penalty,
            rate_floor=RATE_EPS_HZ,
        )
        coefficients.append(coef)
        place_field = jnp.zeros((is_track_interior.shape[0],))
        place_fields.append(
            place_field.at[is_track_interior].set(
                jnp.clip(
                    jnp.exp(emission_predict_matrix @ coef),
                    min=RATE_EPS_HZ,
                    max=None,
                )
            )
        )

    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "environment": environment,
        "coefficients": jnp.stack(coefficients, axis=0),
        "emission_design_info": emission_design_info,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


@requires_time_edges
def predict_sorted_spikes_glm_log_likelihood(
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
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Predict the log likelihood of spikes given a fitted GLM encoding model.

    Calculates the log likelihood of observing the given spike times under
    either a non-local (over spatial bins) or local (at the animal's current
    position) GLM model.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
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
        Expected firing rate in Hz for each neuron in each position bin, derived
        from the fitted GLM (`exp(predict_matrix @ coefficients)`).
    no_spike_part_log_likelihood : jnp.ndarray, shape (n_position_bins,)
        Sum of Hz rates across neurons (`sum(place_fields)`), despite the
        historical key name. Multiply by bin duration and subtract to score
        no spikes.
    is_track_interior : jnp.ndarray, shape (n_position_bins,)
        Boolean array indicating which position bins are part of the valid
        track area.
    disable_progress_bar : bool, optional
        If True, suppresses the progress bar display. By default False.
    is_local : bool, optional
        If True, compute the log likelihood only at the animal's current
        interpolated position (local decoding). If False, compute the log
        likelihood across all position bins (non-local decoding).
        By default False.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges, so spikes are binned
        against them and only those owned by the requested rows are counted; the
        result equals the full-time result sliced by ``row_slice``.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, n_place_bins)
        ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    if rate_units != "Hz":
        raise ValidationError(
            "Encoding rates must be in Hz; refit legacy encoding models before decoding."
        )
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    validate_population_lengths(
        "neuron",
        spike_times=spike_times,
        coefficients=coefficients,
        place_fields=place_fields,
    )
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    durations = jnp.asarray(_time_grid.durations(row_start, row_stop))
    n_rows = row_stop - row_start
    row_time = decode_bin_centers(time_edges, row_start, row_stop)

    if is_local:
        log_likelihood = jnp.zeros((n_rows,))

        # Need to interpolate position
        interpolated_position = get_position_at_time(
            position_time, position, row_time, environment
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
            strict=True,
        ):
            spike_count_per_time_bin = get_spikecount_per_time_bin(
                neuron_spike_times,
                time_edges=time_edges,
                row_slice=row_slice,
                _spike_time_order=_spike_time_order,
            )
            local_rate = jnp.exp(emission_predict_matrix @ coef)
            local_rate = jnp.clip(local_rate, min=RATE_EPS_HZ, max=None)
            local_rate = local_rate * durations
            log_likelihood += (
                jax.scipy.special.xlogy(spike_count_per_time_bin, local_rate)
                - local_rate
            )

        log_likelihood = jnp.expand_dims(log_likelihood, axis=1)
    else:
        n_interior_bins = is_track_interior.sum()
        log_likelihood = jnp.zeros((n_rows, n_interior_bins))
        for neuron_spike_times, place_field in zip(
            tqdm(
                spike_times,
                unit="cell",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            place_fields,
            strict=True,
        ):
            spike_count_per_time_bin = get_spikecount_per_time_bin(
                neuron_spike_times,
                time_edges=time_edges,
                row_slice=row_slice,
                _spike_time_order=_spike_time_order,
            )
            log_likelihood += jax.scipy.special.xlogy(
                np.expand_dims(spike_count_per_time_bin, axis=1),
                jnp.expand_dims(place_field[is_track_interior], axis=0)
                * durations[:, None],
            )

        log_likelihood -= (
            durations[:, None] * no_spike_part_log_likelihood[is_track_interior]
        )

    return log_likelihood
