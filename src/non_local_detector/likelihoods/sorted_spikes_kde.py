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

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    RATE_EPS_HZ,
    KDEModel,
    _SpikeTimeOrder,
    decode_bin_centers,
    drop_zero_weight_samples,
    get_position_at_time,
    get_spikecount_per_time_bin,
    resolve_row_slice,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)


def fit_sorted_spikes_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    environment: Environment,
    weights: jnp.ndarray | None = None,
    position_std: float = np.sqrt(12.5),
    block_size: int = 100,
    disable_progress_bar: bool = False,
    *,
    encoding_time_range=None,
    valid_position_intervals=None,
    _encoding_support=None,
) -> dict:
    """Fit a KDE encoding model for sorted spikes.

    Parameters
    ----------
    position_time : jnp.ndarray, shape (n_time_position,)
        Sampling times for the position.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        Spike times for each neuron.
    environment : Environment
        The spatial environment.
    weights : jnp.ndarray, shape (n_time_position,), optional
        Sample weights for each position time point, by default None.
        If None, uniform weights are used.
    position_std : float, optional
        Gaussian kernel standard deviation for position, by default sqrt(12.5)
    block_size : int, optional
        Size of blocks for KDE computation, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

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
        Dictionary containing fitted encoding model components:
        - 'environment': The spatial environment
        - 'marginal_models': KDE models for each neuron's spatial firing
        - 'occupancy_model': KDE model for spatial occupancy
        - 'occupancy': Occupancy density at interior place bins
        - 'mean_rates': Mean firing rates per neuron
        - 'place_fields': Derived place fields (firing rates) per neuron
        - 'no_spike_part_log_likelihood': Sum of Hz rates across neurons
        - 'is_track_interior': Boolean mask for interior track bins
        - 'disable_progress_bar': Progress bar setting
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
    if isinstance(position_std, int | float):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])

    # Ensure position_std is a JAX array for KDEModel
    assert isinstance(position_std, jnp.ndarray)

    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        if environment.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ is required when is_track_interior_ is None"
            )
        is_track_interior = jnp.ones(len(environment.place_bin_centers_), dtype=bool)
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]
    if weights is None:
        weights = jnp.ones((position.shape[0],))
    else:
        weights = validate_weights(weights, position.shape[0])

    occupancy_samples, occupancy_weights = drop_zero_weight_samples(
        position, exposure_weights
    )
    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        occupancy_samples = get_linearized_position(
            occupancy_samples,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
        occupancy_samples, weights=occupancy_weights
    )

    occupancy = occupancy_model.predict(interior_place_bin_centers)

    mean_rates = []
    place_fields = []
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
        weights_at_spike_times = support.interpolate(
            weights, neuron_spike_times, fill_value=0.0
        )

        weight_sum = exposure_weights.sum()
        mean_rates.append(weighted_mean_rate(weights_at_spike_times, weight_sum))
        neuron_marginal_model = KDEModel(std=position_std, block_size=block_size).fit(
            get_position_at_time(
                position_time,
                position,
                neuron_spike_times,
                environment,
                encoding_support=support,
            ),
            weights=weights_at_spike_times,
        )
        marginal_models.append(neuron_marginal_model)
        marginal_density = neuron_marginal_model.predict(interior_place_bin_centers)
        # NaN here means a degenerate KDE (e.g., zero-variance spike
        # features), not "no density" — track the count so it can be
        # surfaced rather than silently zeroed.
        nan_mask = jnp.isnan(marginal_density)
        nan_density_count = nan_density_count + jnp.sum(nan_mask)
        marginal_density = jnp.where(nan_mask, 0.0, marginal_density)
        place_fields.append(
            jnp.zeros((is_track_interior.shape[0],))
            .at[is_track_interior]
            .set(
                jnp.clip(
                    mean_rates[-1]
                    * jnp.where(
                        occupancy > 0.0,
                        marginal_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                        EPS,
                    ),
                    min=RATE_EPS_HZ,
                    max=None,
                )
            )
        )

    place_fields = jnp.stack(place_fields, axis=0)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

    n_nan_density_bins = int(nan_density_count)
    if n_nan_density_bins > 0:
        warnings.warn(
            f"KDE marginal density was NaN at {n_nan_density_bins} "
            f"(neuron, bin) location(s); these were set to zero. NaN density "
            f"usually indicates a degenerate KDE (zero-variance or identical "
            f"spike features, or position_std=0). Inspect the encoding-spike "
            f"feature distribution.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "environment": environment,
        "marginal_models": marginal_models,
        "occupancy_model": occupancy_model,
        "occupancy": occupancy,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


@requires_time_edges
def predict_sorted_spikes_kde_log_likelihood(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    marginal_models: list[KDEModel],
    occupancy_model: KDEModel,
    occupancy: jnp.ndarray,
    mean_rates: jnp.ndarray,
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
    """Predict the log likelihood of sorted spikes using KDE encoding models.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : jnp.ndarray, shape (n_time_position,)
        Sampling times for the position.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[np.ndarray]
        Spike times for each neuron.
    environment : Environment
        The spatial environment.
    marginal_models : list[KDEModel]
        Marginal models for each neuron.
    occupancy_model : KDEModel
        Occupancy model.
    occupancy : jnp.ndarray, shape (n_place_bins,)
        Occupancy for each place bin.
    mean_rates : jnp.ndarray, shape (n_neurons,)
        Mean firing rates in Hz for each neuron.
    place_fields : jnp.ndarray, shape (n_neurons, n_place_bins)
        Spatial firing rates in Hz for each neuron.
    no_spike_part_log_likelihood : jnp.ndarray, shape (n_place_bins,)
        Sum of Hz rates across neurons, despite the historical key name.
        Multiply by each bin's duration and subtract to score no spikes.
    is_track_interior : jnp.ndarray, shape (n_place_bins,)
        Boolean mask for track interior.
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False
    is_local : bool, optional
        Compute the log likelihood at the animal's position, by default False
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
    log_likelihood : jnp.ndarray, shape (n_rows, n_place_bins) or (n_rows, 1)
        The log likelihood of the spikes at each requested time bin. ``n_rows``
        is ``n_bins`` unless ``row_slice`` is given. The shape is
        (n_rows, n_place_bins) if is_local is False, otherwise (n_rows, 1).

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
        marginal_models=marginal_models,
        mean_rates=mean_rates,
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
        occupancy = occupancy_model.predict(interpolated_position)

        for neuron_spike_times, neuron_marginal_model, neuron_mean_rate in zip(
            tqdm(
                spike_times,
                unit="cell",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            marginal_models,
            mean_rates,
            strict=True,
        ):
            spike_count_per_time_bin = get_spikecount_per_time_bin(
                neuron_spike_times,
                time_edges=time_edges,
                row_slice=row_slice,
                _spike_time_order=_spike_time_order,
            )
            marginal_density = neuron_marginal_model.predict(interpolated_position)
            # A NaN marginal at decode means a NaN interpolated position
            # (dropped tracking / out-of-bounds) -- an expected decode-time gap,
            # not a degenerate encoding model (that is surfaced at fit time and,
            # after input validation, cannot occur here). Zero-fill quietly; a
            # per-timestep warning would just be noise.
            marginal_density = jnp.where(
                jnp.isnan(marginal_density), 0.0, marginal_density
            )
            local_rate = neuron_mean_rate * jnp.where(
                occupancy > 0.0,
                marginal_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                EPS,
            )
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
