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
    - `fit_poisson_regressions`: Fits the L2-regularized Poisson GLM
      coefficients of many neurons that share one design matrix and exposure,
      from their weighted spike counts, with damped Newton steps taken
      together (`fit_poisson_regression` fits a single neuron the same way).
    - `fit_sorted_spikes_glm_encoding_model`: Orchestrates the fitting process
      for *all neurons*: it builds every neuron's weighted spike counts, calls
      `fit_poisson_regressions` once, and computes the resulting place fields
      (expected firing rate across space).

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

import logging
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from patsy import build_design_matrices, dmatrix  # type: ignore[import-untyped]
from patsy.design_info import DesignInfo  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment, get_n_bins
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    RATE_EPS_HZ,
    _blocked_nonlocal_poisson_log_likelihood,
    _concatenate_row_blocks,
    _spike_counts_matrix,
    _SpikeTimeOrder,
    decode_bin_centers,
    get_position_at_time,
    resolve_row_slice,
    validate_population_lengths,
    validate_weights,
)
from non_local_detector.likelihoods.no_spike import _poisson_row_log_likelihood
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)

logger = logging.getLogger(__name__)


@jax.jit
def _local_glm_log_likelihood(design, coefficients, counts, durations):
    if counts.shape[1] == 0:
        return jnp.zeros((counts.shape[0],))
    rate = jax.lax.optimization_barrier(
        jnp.exp(jnp.matmul(design, coefficients.T, precision=jax.lax.Precision.HIGHEST))
    )
    rate = jax.lax.optimization_barrier(jnp.clip(rate, min=RATE_EPS_HZ, max=None))
    expected = jax.lax.optimization_barrier(rate * durations[:, None])
    return _poisson_row_log_likelihood(counts, expected)


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


# Gradient inf-norm above which a fit is treated as genuinely non-converged.
# Set well above GLM_GRADIENT_TOL, which float32 sums over long recordings
# cannot always reach, so stopping at the precision limit does not warn.
GLM_CONVERGENCE_GRAD_TOL = 1e-3
# A fit stops once its gradient inf-norm is below this, or once no step
# lowers its objective.
GLM_GRADIENT_TOL = 1e-5
GLM_MAX_NEWTON_STEPS = 50
_GLM_MAX_STEP_HALVINGS = 30
# Bounds the per-block (n_cells, block_rows, n_coefficients) Hessian product.
_GLM_BLOCK_BYTES = 256 * 2**20


@partial(jax.jit, static_argnames=("block_rows", "derivatives"))
def _poisson_regression_sums(
    design_matrix,
    spikes,
    weights,
    coefficients,
    rate_floor,
    *,
    block_rows,
    derivatives,
):
    """Per-cell sums over rows of the Poisson data term and its derivatives.

    Parameters
    ----------
    design_matrix : jnp.ndarray, shape (n_rows, n_coefficients)
        ``n_rows`` is a multiple of ``block_rows``; padding rows have zero
        weight and spikes.
    spikes : jnp.ndarray, shape (n_cells, n_rows)
    weights : jnp.ndarray, shape (n_rows,)
    coefficients : jnp.ndarray, shape (n_cells, n_coefficients)
    rate_floor : float
    block_rows : int
    derivatives : bool
        Also return the gradient and Hessian of the data term.

    Returns
    -------
    data : jnp.ndarray, shape (n_cells,)
        ``sum_i [spikes_i * log(rate_i) - weights_i * rate_i]`` with
        ``rate = max(exp(design_matrix @ coefficients), rate_floor)``.
    gradient : jnp.ndarray, shape (n_cells, n_coefficients)
        Only if ``derivatives``.
    hessian : jnp.ndarray, shape (n_cells, n_coefficients, n_coefficients)
        Only if ``derivatives``: ``sum_i weights_i * rate_i * x_i x_i^T`` over
        unfloored rows, the negated Hessian of the data term.
    """
    highest = jax.lax.Precision.HIGHEST
    n_cells, n_coefficients = coefficients.shape

    def add_block(block, sums):
        start = block * block_rows
        X = jax.lax.dynamic_slice_in_dim(design_matrix, start, block_rows)
        y = jax.lax.dynamic_slice_in_dim(spikes, start, block_rows, axis=1)
        w = jax.lax.dynamic_slice_in_dim(weights, start, block_rows)
        unfloored = jnp.exp(jnp.matmul(coefficients, X.T, precision=highest))
        above_floor = unfloored > rate_floor
        rate = jnp.where(above_floor, unfloored, rate_floor)
        data = jnp.sum(jax.scipy.special.xlogy(y, rate) - w * rate, axis=1)
        if not derivatives:
            return (sums[0] + data,)
        residual = jnp.where(above_floor, y - w * rate, 0.0)
        curvature = jnp.where(above_floor, w * rate, 0.0)
        return (
            sums[0] + data,
            sums[1] + jnp.matmul(residual, X, precision=highest),
            sums[2] + jnp.einsum("cb,bp,bq->cpq", curvature, X, X, precision=highest),
        )

    dtype = design_matrix.dtype
    sums = (jnp.zeros(n_cells, dtype),)
    if derivatives:
        sums += (
            jnp.zeros((n_cells, n_coefficients), dtype),
            jnp.zeros((n_cells, n_coefficients, n_coefficients), dtype),
        )
    return jax.lax.fori_loop(0, design_matrix.shape[0] // block_rows, add_block, sums)


def fit_poisson_regressions(
    design_matrix: np.ndarray,
    spikes: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float = 1e-7,
    *,
    rate_floor: float = EPS,
) -> jnp.ndarray:
    """Fit weighted Poisson regressions for many cells sharing one design.

    Each cell maximizes ``sum_i [spikes_i * log(lambda_i) - weights_i *
    lambda_i]`` (normalized by ``sum(weights)``) minus ``l2_penalty`` times the
    squared non-intercept coefficients, with ``lambda = max(exp(design_matrix
    @ coefficients), rate_floor)``. The event and exposure terms are weighted
    separately: ``spikes`` already carries each event's weight, so it is not
    multiplied by ``weights`` again.

    All cells take damped Newton steps together. The sums over rows run
    compiled on the device in row blocks; each cell's small Newton system is
    solved in float64 on the host, so basis functions with almost no weighted
    support are not moved by float32 rounding. A cell stops once its gradient
    inf-norm is below ``GLM_GRADIENT_TOL`` or no step halving lowers its
    objective.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    spikes : np.ndarray, shape (n_cells, n_time)
        Event mass per row and cell: the weighted spike counts from
        ``weighted_spike_counts``, not multiplied by ``weights`` again.
    weights : np.ndarray, shape (n_time,)
        Per-row exposure weight.
    l2_penalty : float, optional
        L2 regression penalty, by default 1e-7
    rate_floor : float, optional
        Lower bound on the fitted rate, by default ``EPS``.

    Returns
    -------
    coefficients : jnp.ndarray, shape (n_cells, n_coefficients)
    """
    design_matrix = np.asarray(design_matrix, dtype=np.float32)
    spikes = np.asarray(spikes, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    n_cells = spikes.shape[0]
    n_time, n_coefficients = design_matrix.shape

    # Zero total exposure means no training coverage at all, so the rate is
    # unidentified and ``sum(spikes) / sum(weights)`` is 0/0. Return an
    # intercept-only model at the rate floor, which is exactly what a unit
    # with real exposure but no spikes gets from the initial condition below.
    # A small-but-positive exposure (a low-mass EM update) still fits normally.
    total_exposure = float(np.sum(weights, dtype=np.float64))
    coefficients = np.zeros((n_cells, n_coefficients))
    if total_exposure <= 0.0:
        coefficients[:, 0] = np.log(rate_floor)
        return jnp.asarray(coefficients)
    weight_sum = max(total_exposure, EPS)
    average_rate = np.sum(spikes, axis=1, dtype=np.float64) / total_exposure
    coefficients[:, 0] = np.log(np.maximum(average_rate, rate_floor))

    block_rows = int(
        np.clip(_GLM_BLOCK_BYTES // (4 * max(n_cells, 1) * n_coefficients), 1, n_time)
    )
    n_padded = -(-n_time // block_rows) * block_rows
    padding = n_padded - n_time
    design = jnp.asarray(np.pad(design_matrix, ((0, padding), (0, 0))))
    counts = jnp.asarray(np.pad(spikes, ((0, 0), (0, padding))))
    exposure = jnp.asarray(np.pad(weights, (0, padding)))
    penalized = np.ones(n_coefficients)
    penalized[0] = 0.0

    def evaluate(coefficients, derivatives):
        sums = _poisson_regression_sums(
            design,
            counts,
            exposure,
            jnp.asarray(coefficients, dtype=jnp.float32),
            rate_floor,
            block_rows=block_rows,
            derivatives=derivatives,
        )
        sums = [np.asarray(value, dtype=np.float64) for value in sums]
        sums[0] = -sums[0] / weight_sum + l2_penalty * np.sum(
            penalized * coefficients**2, axis=1
        )
        if derivatives:
            sums[1] = -sums[1] / weight_sum + 2 * l2_penalty * penalized * coefficients
            sums[2] = sums[2] / weight_sum + 2 * l2_penalty * np.diag(penalized)
        return sums

    loss, gradient, hessian = evaluate(coefficients, derivatives=True)
    active = np.ones(n_cells, dtype=bool)
    for _ in range(GLM_MAX_NEWTON_STEPS):
        gradient_norm = np.max(np.abs(gradient), axis=1)
        active &= np.isfinite(loss) & (gradient_norm > GLM_GRADIENT_TOL)
        if not active.any():
            break
        try:
            direction = -np.linalg.solve(hessian[active], gradient[active][..., None])[
                ..., 0
            ]
        except np.linalg.LinAlgError:
            logger.debug(
                "GLM Newton system singular for a cell; using least squares for "
                "cells %s",
                np.flatnonzero(active).tolist(),
            )
            direction = -np.stack(
                [
                    np.linalg.lstsq(h, g, rcond=None)[0]
                    for h, g in zip(hessian[active], gradient[active], strict=True)
                ]
            )
        slope = np.sum(gradient[active] * direction, axis=1)
        step = np.ones(direction.shape[0])
        accepted = np.zeros(direction.shape[0], dtype=bool)
        for _ in range(_GLM_MAX_STEP_HALVINGS):
            candidate = coefficients.copy()
            pending = ~accepted
            rows = np.flatnonzero(active)[pending]
            candidate[rows] += step[pending, None] * direction[pending]
            (candidate_loss,) = evaluate(candidate, derivatives=False)
            # Armijo sufficient decrease, per cell.
            decreased = np.isfinite(candidate_loss[rows]) & (
                candidate_loss[rows]
                <= loss[rows] + 1e-4 * step[pending] * slope[pending]
            )
            coefficients[rows[decreased]] = candidate[rows[decreased]]
            accepted[np.flatnonzero(pending)[decreased]] = True
            if accepted.all():
                break
            step[~accepted] *= 0.5
        # A cell no step can improve has reached the precision limit.
        active[np.flatnonzero(active)[~accepted]] = False
        loss, gradient, hessian = evaluate(coefficients, derivatives=True)

    # Judge convergence by the gradient inf-norm, and also warn when a fit
    # diverged to a non-finite loss or gradient: ``NaN > tol`` is ``False``,
    # so a gradient-only check would silently pass a broken fit.
    gradient_norm = np.max(np.abs(gradient), axis=1)
    unconverged = (
        ~np.isfinite(gradient_norm)
        | ~np.isfinite(loss)
        | (gradient_norm > GLM_CONVERGENCE_GRAD_TOL)
    )
    if unconverged.any():
        worst = float(
            np.max(
                np.where(np.isfinite(gradient_norm), gradient_norm, np.inf)[unconverged]
            )
        )
        warnings.warn(
            f"GLM Poisson regression may not have converged for "
            f"{int(unconverged.sum())} of {n_cells} cells (indices "
            f"{np.flatnonzero(unconverged)[:20].tolist()}): largest final "
            f"gradient inf-norm {worst:.3e} vs tolerance "
            f"{GLM_CONVERGENCE_GRAD_TOL:.1e} (a non-finite gradient or loss means "
            f"the fit diverged). Place-field coefficients may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    return jnp.asarray(coefficients)


def fit_poisson_regression(
    design_matrix: np.ndarray,
    spikes: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float = 1e-7,
    *,
    rate_floor: float = EPS,
) -> jnp.ndarray:
    """Fit a weighted Poisson regression model for one cell.

    See :func:`fit_poisson_regressions`, which this calls with one cell.

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
    rate_floor : float, optional
        Lower bound on the fitted rate, by default ``EPS``.

    Returns
    -------
    coefficients : jnp.ndarray, shape (n_coefficients,)
    """
    return fit_poisson_regressions(
        design_matrix,
        np.asarray(spikes)[None],
        weights,
        l2_penalty,
        rate_floor=rate_floor,
    )[0]


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
        Stored in the encoding model for prediction's progress bars; the fit
        has none. By default False.

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
    if (
        environment.track_graph is not None
        and encoding_positions.shape[1] > 1
        and len(encoding_positions)
    ):
        # The spline is defined on the track's 1-D bin edges, as the KDE fits
        # and the prediction path use linear position.
        encoding_positions = environment._linear_position(
            np.asarray(encoding_positions)
        )[:, None]
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

    if len(support.indices):
        event_counts = np.stack(
            [
                np.asarray(support.event_counts(neuron_spike_times, weights))[
                    support.indices
                ]
                for neuron_spike_times in spike_times
            ]
        ).reshape(len(spike_times), -1)
        exposure = exposure_weights[support.indices]
    else:
        event_counts = np.zeros((len(spike_times), len(emission_design_matrix)))
        exposure = np.zeros(len(emission_design_matrix))
    coefficients = fit_poisson_regressions(
        emission_design_matrix,
        event_counts,
        exposure,
        l2_penalty=l2_penalty,
        rate_floor=RATE_EPS_HZ,
    )
    interior_rates = jnp.clip(
        jnp.exp(emission_predict_matrix @ coefficients.T), min=RATE_EPS_HZ, max=None
    )  # (n_interior_bins, n_cells)
    place_fields = (
        jnp.zeros((len(spike_times), is_track_interior.shape[0]))
        .at[:, is_track_interior]
        .set(interior_rates.T)
    )
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    return {
        "environment": environment,
        "coefficients": coefficients,
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
    row_time = decode_bin_centers(time_edges, row_start, row_stop)

    if is_local:
        # Need to interpolate position
        interpolated_position = get_position_at_time(
            position_time, position, row_time, environment
        )
        emission_predict_matrix = make_spline_predict_matrix(
            emission_design_info, interpolated_position
        )
        design = jnp.asarray(emission_predict_matrix)
        coefficients = jnp.asarray(coefficients)

        def evaluate(start, stop):
            rows = slice(start - row_start, stop - row_start)
            counts = _spike_counts_matrix(
                spike_times,
                time_edges,
                "Local Likelihood",
                True,
                slice(start, stop),
                _spike_time_order=_spike_time_order,
            )
            return _local_glm_log_likelihood(
                design[rows], coefficients, jnp.asarray(counts), durations[rows]
            )

        log_likelihood = _concatenate_row_blocks(
            row_start,
            row_stop,
            len(spike_times),
            evaluate,
            "Local Likelihood",
            disable_progress_bar,
        )[:, None]
    else:
        log_likelihood = _blocked_nonlocal_poisson_log_likelihood(
            spike_times,
            time_edges,
            row_start,
            row_stop,
            jnp.asarray(place_fields)[:, is_track_interior],
            durations,
            no_spike_part_log_likelihood[is_track_interior],
            disable_progress_bar=disable_progress_bar,
            _spike_time_order=_spike_time_order,
        )

    return log_likelihood
