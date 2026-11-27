"""Kernel Density Estimation (KDE) based encoding/decoding for sorted spikes.

This module implements a non-parametric method for modeling neural firing rates
(place fields) and decoding position based on **sorted spike** data, using
Kernel Density Estimation (KDE).

Unlike parametric models like GLMs, this approach directly estimates density
functions from the data. The core principle is to estimate:
1.  The spatial density of the animal's occupancy (how much time is spent where).
2.  The spatial density of each neuron's spikes (where a neuron tends to fire).

The place field, or firing rate map (`rate(x)`), for each neuron is then
calculated as:
  `rate(x) = mean_firing_rate * (spike_density(x) / occupancy_density(x))`
Both `spike_density` and `occupancy_density` are estimated using KDE with
Gaussian kernels. The model assumes Poisson firing statistics, where the spike
counts are driven by this estimated rate `rate(x)`.

Key functionalities:
1.  **Encoding Model Fitting (`fit_sorted_spikes_kde_encoding_model`):**
    - Takes position data, sorted spike times (one list per neuron), and
      environment information.
    - Fits KDE models (using the shared `KDEModel` class) to estimate the
      spatial occupancy density and the marginal spatial density for each
      neuron's spikes.
    - Calculates the mean firing rate for each neuron.
    - Derives the place field (rate map) for each neuron using the formula above.
    - Handles both 2D and 1D linearized environments.
    - Returns a dictionary containing the fitted KDE models, mean rates,
      derived place fields, and occupancy information.

2.  **Log-Likelihood Prediction (`predict_sorted_spikes_kde_log_likelihood`):**
    - Calculates the log-likelihood of observing spike trains during a
      *decoding* period, given the fitted KDE-based encoding model.
    - Assumes Poisson statistics and uses the log-likelihood formula:
      `sum_{neurons} [ k * log(lambda) - lambda ]`
      where `k` is the observed spike count and `lambda` is the predicted rate
      (`rate(x)`) derived from the KDE place fields.
    - Supports both:
        - **Non-local decoding:** Computing likelihood across all spatial bins.
        - **Local decoding:** Computing likelihood only at the animal's
          interpolated position.

This module provides a non-parametric alternative to GLM-based methods for
sorted spikes, relying on density estimation rather than fitted coefficients.
It utilizes JAX and SciPy for efficient computation and interpolation.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

from non_local_detector._warnings import user_stacklevel
from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
    warn_if_tracking_is_sparse,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import NumericalWarning
from non_local_detector.likelihoods.common import (
    _KDE_BLOCK_KERNEL_BYTES,
    RATE_EPS_HZ,
    KDEModel,
    SpikeTimeOrder,
    _count_block_rows,
    _quarter_octave,
    as_array,
    as_arrays,
    blocked_nonlocal_poisson_log_likelihood,
    concatenate_row_blocks,
    drop_zero_weight_samples,
    environment_grid_fingerprint,
    kde_log_densities,
    kde_log_density,
    kde_log_density_rows,
    kde_model_with_block_size,
    log_density_intensity,
    pad_rows,
    padded_sample_count,
    place_encoding_spikes,
    poisson_row_log_likelihood,
    prepare_decode_rows,
    prepare_kde_local_inputs,
    require_position,
    spike_counts_matrix,
    tiled_kde_block,
    validate_bandwidth,
    validate_encoding_contract,
    validate_population_lengths,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.encoding_model import SortedSpikesEncodingModel
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    requires_time_edges,
)

__all__ = [
    "SortedSpikesKDEEncodingModel",
    "fit_sorted_spikes_kde_encoding_model",
    "predict_sorted_spikes_kde_log_likelihood",
]


# (n_units, n_rows) arrays the Poisson terms hold at once: two temporaries,
# the log densities and the counts, and one to spare.
_POISSON_UNIT_ARRAYS = 5


def _local_block_rows(models: list[KDEModel], n_dims: int) -> tuple[int, int]:
    """Rows of a full Local block, and the density pass's point block for it.

    At most the spike-count block, the rows whose Poisson arrays fit
    ``_KDE_BLOCK_KERNEL_BYTES``, and the rows whose density pass fits it
    (see ``tiled_kde_block``); a whole number of point blocks.
    """
    itemsize = jax.dtypes.canonicalize_dtype(jnp.float64).itemsize
    n_units = max(len(models), 1)
    limit = min(
        _count_block_rows(n_units),
        _KDE_BLOCK_KERNEL_BYTES // (_POISSON_UNIT_ARRAYS * n_units * itemsize),
    )
    return tiled_kde_block(models, n_dims, max(1, limit))


def _padded_block_rows(n_rows: int, max_rows: int, point_rows: int) -> int:
    """Rows a Local block of ``n_rows`` is evaluated at, so chunks share kernels.

    Up to one point block, one of four sizes per octave (at most 25% more);
    above it, a number of point blocks of four sizes per octave (at most 25%
    more blocks, plus under one block); never more than ``max_rows``.
    """
    if n_rows <= point_rows:
        return min(max_rows, point_rows, padded_sample_count(n_rows))
    n_blocks = _quarter_octave(-(-n_rows // point_rows))
    return min(max_rows, point_rows * n_blocks)


# Every place-field and Local rate is clipped up to this many Hz.
_RATE_FLOOR_HZ = RATE_EPS_HZ


def _kde_rates(
    mean_rates: jax.Array | float,
    log_marginals: jax.Array,
    log_occupancy: jax.Array,
) -> jax.Array:
    """Return firing rates in Hz from true log densities, floored at ``RATE_EPS_HZ``.

    The place fields at fit and the Local rows at prediction both use it, so
    both turn densities into rates the same way.

    Parameters
    ----------
    mean_rates : jax.Array or float
        Mean firing rates in Hz, broadcastable with ``log_marginals``.
    log_marginals : jax.Array
        True log densities of the neurons' spike positions.
    log_occupancy : jax.Array
        True log occupancy density, broadcastable with ``log_marginals``.

    Returns
    -------
    rates : jax.Array
        ``mean_rate * marginal / occupancy``, formed from log ratios (see
        ``log_density_intensity``). NaN densities give NaN rates.
    """
    return jnp.clip(
        jnp.exp(log_density_intensity(mean_rates, log_marginals, log_occupancy)),
        min=_RATE_FLOOR_HZ,
        max=None,
    )


def _unit_log_density_floor(mean_rates: np.ndarray) -> np.ndarray:
    """Per-unit term of the log density below which a rate is surely clipped.

    A density below ``log(_RATE_FLOOR_HZ / 2) - log(mean_rate) + log_occupancy``
    gives a rate below half of the clip in :func:`_kde_rates`, so its exact
    value does not change the result; the factor 2 leaves room for the
    rounding of the floor and of the log ratio.

    Parameters
    ----------
    mean_rates : np.ndarray, shape (n_units,)
        Mean firing rates in Hz.

    Returns
    -------
    log_floor : np.ndarray, shape (n_units,)
        ``inf`` for silent units, whose rate is the clip whatever the density.
    """
    with np.errstate(divide="ignore"):
        log_mean_rates: np.ndarray = np.log(np.asarray(mean_rates, float))
    log_floor: np.ndarray = np.log(_RATE_FLOOR_HZ / 2) - log_mean_rates
    return log_floor


@jax.jit
def _local_kde_log_likelihood(
    log_marginals: jax.Array,
    log_occupancy: jax.Array,
    counts: jax.Array,
    mean_rates: jax.Array,
    durations: jax.Array,
) -> jax.Array:
    """Poisson log likelihood of each row's spike counts at the animal's position.

    Parameters
    ----------
    log_marginals : jax.Array, shape (n_neurons, n_rows)
        True log density of each neuron's spike positions at each row's
        position.
    log_occupancy : jax.Array, shape (n_rows,)
        True log occupancy density at each row's position.
    counts : jax.Array, shape (n_rows, n_neurons)
        Spike count of each neuron in each row.
    mean_rates : jax.Array, shape (n_neurons,)
        Mean firing rate of each neuron, in Hz.
    durations : jax.Array, shape (n_rows,)
        Duration of each row in seconds.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows,)
        Neuron terms are added in neuron order.
    """
    rates = _kde_rates(mean_rates[:, None], log_marginals, log_occupancy[None, :])
    return poisson_row_log_likelihood(counts, rates.T * durations[:, None])  # type: ignore[no-any-return]  # jit-wrapped


class SortedSpikesKDEEncodingModel(SortedSpikesEncodingModel):
    """The encoding model ``fit_sorted_spikes_kde_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_sorted_spikes_kde_log_likelihood``, documented there.
    """

    marginal_models: list[KDEModel]
    occupancy_model: KDEModel
    occupancy: np.ndarray
    log_occupancy: jax.Array
    mean_rates: np.ndarray
    no_spike_part_log_likelihood: jax.Array
    is_track_interior: np.ndarray


def fit_sorted_spikes_kde_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    position_std: float | jax.Array = np.sqrt(12.5),
    block_size: int = 100,
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> SortedSpikesKDEEncodingModel:
    """Fit a KDE encoding model for sorted spikes.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Sampling times for the position.
    position : array_like, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times for each neuron.
    environment : Environment
        The spatial environment.
    weights : array_like, shape (n_time_position,), optional
        Sample weights for each position time point, by default None.
        If None, uniform weights are used. Each spike takes the weight
        interpolated to its time. It sits at its interpolated position, except
        where its two bracketing samples differ in weight: there it sits at the
        mean of their positions weighted by its share of each sample's weight,
        so a spike next to an unweighted sample sits on its weighted sample,
        where that interval's occupancy is.
    position_std : float, optional
        Gaussian kernel standard deviation for position, by default sqrt(12.5).
    block_size : int, optional
        Size of blocks for KDE computation, by default 100.
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False.

    encoding_time_range : array_like, shape (2,), optional
        Acquisition start/stop in seconds, replacing the outer bounds of the
        encoding support. A range inside the samples clips it, using the
        original interpolation basis; a range past them extends it, holding
        position at the first or last sample. Uniform samples otherwise include
        endpoint half-cells; a singleton requires explicit bounds or a tracking
        interval.
    valid_position_intervals : array_like, shape (n_intervals, 2), optional
        Ordered, non-overlapping continuous tracking intervals in seconds.
        Required for irregular timestamps; each interval needs a finite sample.
        Positions are held at segment endpoints and NaN rows split support.
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support. Direct callers
        omit it and the support is built from the other arguments.

    Returns
    -------
    encoding_model : SortedSpikesKDEEncodingModel
        Dictionary containing fitted encoding model components:
        - 'algorithm': ``"sorted_spikes_kde"``
        - 'environment': The spatial environment
        - 'marginal_models': KDE models for each neuron's spatial firing
        - 'occupancy_model': KDE model for spatial occupancy
        - 'occupancy': Occupancy density at interior place bins
        - 'log_occupancy': Log occupancy density at interior place bins
        - 'mean_rates': Mean firing rates per neuron
        - 'place_fields': Derived place fields (firing rates) per neuron
        - 'no_spike_part_log_likelihood': Sum of Hz rates across neurons
        - 'is_track_interior': Boolean mask for interior track bins
        - 'disable_progress_bar': Progress bar setting
        - 'rate_units': ``"Hz"``
        - 'encoding_exposure_seconds': Weighted encoding occupancy time, seconds
    """
    position_time = as_array(position_time)
    spike_times = as_arrays(spike_times)
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    n_dims = 1 if environment.track_graph is not None else position.shape[1]
    position_std = validate_bandwidth(position_std, n_dims, "position_std")
    warn_if_tracking_is_sparse(support, position, position_std)
    warn_if_no_encoding_exposure(
        exposure_weights, "Sorted-spikes KDE", clusterless=False
    )

    is_track_interior = environment.is_track_interior_flat_
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]
    occupancy_samples, occupancy_weights = drop_zero_weight_samples(
        position, exposure_weights
    )
    occupancy_samples = environment.decoding_position(occupancy_samples)
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
        occupancy_samples, weights=occupancy_weights
    )

    log_occupancy = kde_log_density(occupancy_model, interior_place_bin_centers)
    # The probability density is the exponential of the true log density.
    occupancy = jnp.exp(log_occupancy)

    mean_rates = []
    place_fields: Any = []  # list of per-neuron fields, restacked into one array below
    marginal_models = []
    # Accumulate the NaN-density count on-device (jnp) and materialize it to a
    # Python int once after the loop; int(jnp.sum(...)) inside the loop forces a
    # host sync per neuron and would serialize KDE fitting on GPU.
    nan_density_count = jnp.array(0, dtype=jnp.int32)

    for neuron_spike_times in tqdm(
        spike_times,
        unit="cell",
        desc="Encoding models",
        disable=disable_progress_bar,
    ):
        neuron_spike_times = neuron_spike_times[support.contains(neuron_spike_times)]
        # Each spike, weighted by the weight at its spike time, sits at its
        # interpolated position or, where its bracketing samples differ in
        # weight, at their share-weighted mean (see place_encoding_spikes).
        spike_positions, weights_at_spike_times = place_encoding_spikes(
            position_time,
            position,
            neuron_spike_times,
            weights,
            environment,
            support,
        )

        weight_sum = exposure_weights.sum()
        mean_rates.append(weighted_mean_rate(weights_at_spike_times, weight_sum))
        neuron_marginal_model = KDEModel(std=position_std, block_size=block_size).fit(
            spike_positions, weights=weights_at_spike_times
        )
        marginal_models.append(neuron_marginal_model)

    # One tiled pass for every neuron, whose kernels compile for the total
    # number of sample tiles, not for each neuron's sample count.
    for mean_rate, log_marginal in zip(
        mean_rates,
        kde_log_densities(marginal_models, interior_place_bin_centers),
        strict=True,
    ):
        # NaN here means a degenerate KDE (e.g., zero-variance spike
        # features), not "no density" — track the count so it can be
        # surfaced rather than silently zeroed.
        nan_density_count = nan_density_count + jnp.sum(jnp.isnan(log_marginal))
        place_fields.append(
            jnp.zeros((is_track_interior.shape[0],))
            .at[is_track_interior]
            .set(_kde_rates(mean_rate, log_marginal, log_occupancy))
        )

    place_fields = (
        jnp.stack(place_fields, axis=0)
        if place_fields
        else jnp.zeros((0, len(is_track_interior)))
    )
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    n_nan_density_bins = int(nan_density_count)
    if n_nan_density_bins > 0:
        warnings.warn(
            f"KDE marginal density was NaN at {n_nan_density_bins} "
            f"(neuron, bin) location(s); the invalid likelihood remains visible. NaN density "
            f"usually indicates a degenerate KDE (zero-variance or identical "
            f"spike features, or position_std=0). Inspect the encoding-spike "
            f"feature distribution.",
            NumericalWarning,
            stacklevel=user_stacklevel(),
        )

    return {
        "algorithm": "sorted_spikes_kde",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "marginal_models": marginal_models,
        "occupancy_model": occupancy_model,
        "occupancy": np.asarray(occupancy),
        "log_occupancy": log_occupancy,
        "mean_rates": np.asarray(mean_rates),
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


@requires_time_edges
def predict_sorted_spikes_kde_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    *,
    environment: Environment,
    marginal_models: list[KDEModel],
    occupancy_model: KDEModel,
    occupancy: ArrayLike,  # noqa: ARG001 -- shared likelihood-algorithm signature
    mean_rates: np.ndarray,
    place_fields: jax.Array,
    no_spike_part_log_likelihood: jax.Array,
    is_track_interior: jax.Array,
    disable_progress_bar: bool = False,
    is_local: bool = False,
    row_slice: slice | None = None,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    log_occupancy: jax.Array | None = None,
    environment_grid: dict[str, Any] | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Predict the log likelihood of sorted spikes using KDE encoding models.

    Parameters
    ----------
    position_time : array_like or None, shape (n_time_position,)
        Sampling times for the position.
        May be None when ``is_local`` is False.
    position : array_like or None, shape (n_time_position, n_position_dims)
        Position samples.
        May be None when ``is_local`` is False.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times for each neuron.
    environment : Environment
        The spatial environment.
    marginal_models : list[KDEModel]
        Marginal models for each neuron.
    occupancy_model : KDEModel
        Occupancy model.
    occupancy : array_like, shape (n_interior_bins,)
        Occupancy density at the interior bins, from the fit. Unused: the
        non-local path reads ``place_fields`` and the local path evaluates
        ``occupancy_model`` at the animal's position.
    mean_rates : np.ndarray, shape (n_neurons,)
        Mean firing rates in Hz for each neuron.
    place_fields : jax.Array, shape (n_neurons, n_place_bins)
        Spatial firing rates in Hz for each neuron.
    no_spike_part_log_likelihood : jax.Array, shape (n_place_bins,)
        Sum of Hz rates across neurons, despite the historical key name.
        Multiply by each bin's duration and subtract to score no spikes.
    is_track_interior : jax.Array, shape (n_place_bins,)
        Boolean mask for track interior.
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False.
    is_local : bool, optional
        Compute the log likelihood at the animal's position, by default False.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges, so spikes are binned
        against them and only those owned by the requested rows are counted; the
        result equals the full-time result sliced by ``row_slice``.
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    algorithm : str, optional
        The algorithm the fit recorded. Accepted so the encoding dictionary
        can be passed as keywords; unused.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the encoding fit; anything other than ``"Hz"``
        raises ``ValidationError`` (refit legacy encoding models).
    encoding_exposure_seconds : float, optional, keyword-only
        Weighted encoding occupancy time in seconds, recorded by the fit.
        Accepted so the encoding dictionary can be passed as keywords; unused.
    log_occupancy : jax.Array, shape (n_interior_bins,), optional, keyword-only
        Log occupancy density at the interior bins, from the fit. Unused, like
        ``occupancy``.
    environment_grid : dict | None, optional
        The fit's record of the environment's grid (see
        ``environment_grid_fingerprint``), by default None, which skips the
        check (a model fitted before the grid was recorded). Prediction raises
        if the environment has since been refit to another grid.
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_interior_bins) or (n_rows, 1)
        The log likelihood of the spikes at each requested time bin. ``n_rows``
        is ``n_bins`` unless ``row_slice`` is given. The shape is
        (n_rows, n_interior_bins) if is_local is False (interior bins only),
        otherwise (n_rows, 1).
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    spike_times = as_arrays(spike_times)
    validate_population_lengths(
        "neuron",
        spike_times=spike_times,
        marginal_models=marginal_models,
        mean_rates=mean_rates,
        place_fields=place_fields,
    )
    if is_local:
        position_time, position = require_position(
            position_time, position, "local sorted-spikes KDE likelihood"
        )
        interpolated_position, log_occupancy = prepare_kde_local_inputs(
            occupancy_model,
            position_time,
            position,
            environment,
            rows.time_grid,
            rows.start,
            rows.stop,
            rows.spike_time_order,
        )
        # Fitted KDE models run compiled over spike-count row blocks, so the
        # host count matrix and compiled graph stay bounded for any request.
        for model in marginal_models:
            if not isinstance(model, KDEModel):
                msg = (  # type: ignore[unreachable]  # runtime guard on a typed list element
                    "Local sorted-spikes KDE likelihood needs KDEModel marginal "
                    f"models from the fitted encoding model, got {type(model).__name__}."
                )
                raise TypeError(msg)
            if model.samples_ is None:
                msg = "This KDE instance is not fitted yet."
                raise RuntimeError(msg)
        points = np.asarray(interpolated_position)
        if points.ndim == 1:
            points = points[:, None]
        host_log_occupancy = np.asarray(log_occupancy)
        host_durations = np.asarray(rows.durations())
        rates = jnp.asarray(mean_rates)
        unit_log_floor = _unit_log_density_floor(mean_rates)
        # Full blocks keep their rows; a shorter block is padded (see
        # _padded_block_rows), so chunks share compiled kernels.
        max_rows, point_rows = _local_block_rows(marginal_models, points.shape[1])
        # The density pass evaluates the models in blocks of point_rows, from
        # tiles built once for every block of this call.
        tiles: dict[tuple[int, ...], Any] = {}
        block_models = [
            kde_model_with_block_size(model, point_rows, rows.spike_time_order)
            for model in marginal_models
        ]

        def evaluate(start: int, stop: int) -> jax.Array:
            block = slice(start - rows.start, stop - rows.start)
            counts = spike_counts_matrix(
                spike_times,
                rows.time_edges,
                "Local Likelihood",
                disable_progress_bar=True,
                row_slice=slice(start, stop),
                _spike_time_order=rows.spike_time_order,
            )
            n_rows = stop - start
            n_padded = _padded_block_rows(n_rows, max_rows, point_rows)

            def pad(values: np.ndarray) -> np.ndarray:
                widths = [(0, n_padded - n_rows)] + [(0, 0)] * (values.ndim - 1)
                return np.pad(values, widths, mode="edge")

            # Every neuron's true log density at the rows' positions, in one
            # tiled pass whose kernels compile for the total number of sample
            # tiles, not for each neuron's sample count. Densities whose rate
            # is surely clipped need no log-space evaluation; their floor is
            # the unit's term plus the row's log occupancy, added on the
            # device. Padding rows repeat the last row, are never evaluated
            # in log space (an infinite floor) and are dropped.
            block_log_occupancy = pad(host_log_occupancy[block])
            row_log_floor = block_log_occupancy.copy()
            row_log_floor[n_rows:] = np.inf
            log_marginals = kde_log_density_rows(
                block_models,
                pad(points[block]),
                log_floor=(unit_log_floor, row_log_floor),
                tiles=tiles,
            )
            log_likelihood = _local_kde_log_likelihood(
                log_marginals,
                block_log_occupancy,
                pad_rows(counts, n_padded),
                rates,
                pad(host_durations[block]),
            )
            # jit-wrapped kernels return Any.
            return log_likelihood if n_padded == n_rows else log_likelihood[:n_rows]  # type: ignore[no-any-return]

        return concatenate_row_blocks(
            rows.start,
            rows.stop,
            len(spike_times),
            evaluate,
            "Local Likelihood",
            disable_progress_bar=disable_progress_bar,
            block_rows=max_rows,
        )[:, None]
    return blocked_nonlocal_poisson_log_likelihood(
        spike_times,
        rows.time_edges,
        rows.start,
        rows.stop,
        jnp.asarray(place_fields)[:, is_track_interior],
        jnp.asarray(rows.durations()),
        no_spike_part_log_likelihood[is_track_interior],
        disable_progress_bar=disable_progress_bar,
        _spike_time_order=rows.spike_time_order,
    )
