from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    LOG_EPS,
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    KDEModel,
    _kde_log_densities,
    _kde_log_scale,
    _log_kernel_matrix,
    _pad_rows,
    _padded_index,
    _padded_occupancy_count,
    _padded_samples,
    _padded_spike_count,
    _prepare_kde_local_inputs,
    _prepared_kde_log_density,
    _prepared_kde_nonlocal_points,
    _prepared_kde_position_log_kernel,
    _relative_weights,
    _sorted_sample_keys,
    _SpikeTimeOrder,
    _traced_block_kde,
    _true_kde_log_density,
    as_std_array,
    deterministic_row_sum,
    drop_zero_weight_samples,
    get_position_at_time,
    interpolate_weights_at_spike_times,
    kde_log_density,
    local_kde_log_ratio,
    log_bin_duration_evidence,
    log_density_intensity,
    resolve_row_slice,
    safe_log,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    sum_spikes_into_rows,
    traced_kde_log_density,
    validate_bandwidth,
    validate_finite,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.streamed_kde import (
    _sample_tiled_density,
    _sample_tiled_log_ratio,
    _streamed_joint_mark_row_sums,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)


def _validate_workspace_limits(encoding_block_size, position_block_size):
    limits = []
    for name, value in (
        ("encoding_block_size", encoding_block_size),
        ("position_block_size", position_block_size),
    ):
        if value is not None:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValidationError(f"{name} must be None or a positive integer")
            value = int(value)
        limits.append(value)
    return tuple(limits)


def _tile_size(limit, size):
    """Keep configured capacity from padding a smaller or empty input."""
    size = max(1, size)
    return size if limit is None else min(limit, size)


def _predict_kde_log_density(model, points, encoding_block_size, position_block_size):
    """Use true log densities while retaining an optional fitted tile policy.

    Tiled evaluation scales each point by an upper bound on its largest log
    kernel, so sample tiles add in one pass. Points whose scaled sum could
    have lost mass to flushed terms are re-evaluated with a running maximum.
    Runs outside jit.
    """
    if encoding_block_size is None and position_block_size is None:
        return kde_log_density(model, points)
    points = jnp.asarray(points)
    if points.ndim == 1:
        points = points[:, None]
    std = as_std_array(model.std, points.shape[1])
    n_samples = model.samples_.shape[0]
    sample_tile_size = _tile_size(encoding_block_size, n_samples)
    eval_block = (
        position_block_size if position_block_size is not None else model.block_size
    )

    def tiled(points, log_scale):
        return _sample_tiled_log_ratio(
            points,
            model.samples_,
            std,
            model.weights_,
            jnp.zeros(points.shape[0]),
            log_scale,
            n_position_dims=0,
            sample_tile_size=sample_tile_size,
            eval_tile_size=_tile_size(eval_block, points.shape[0]),
        )

    log_scale = _kde_log_scale(
        points, std, _sorted_sample_keys(points, model.samples_, model.weights_)
    )
    log_density = tiled(points, log_scale)
    _, _, relative_total = _relative_weights(model.weights_)
    relative_total = float(relative_total)
    if not relative_total > 0.0:
        return log_density
    # Weights enter relative to the largest, so each flushed term is below
    # tiny relative to the scale; the tiled sum is log_density - log_scale
    # plus the log of the relative total.
    dtype_info = jnp.finfo(log_density.dtype)
    log_lost = np.log(
        n_samples * dtype_info.tiny / (4 * dtype_info.eps * relative_total)
    )
    needs_running_max = np.asarray(
        (log_density - log_scale < log_lost) & ~jnp.isnan(log_density)
    )
    if not needs_running_max.any():
        return log_density
    index = np.flatnonzero(needs_running_max)
    return log_density.at[index].set(
        tiled(points[_padded_index(index)], None)[: index.size]
    )


def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Distance between evaluation points and samples using Gaussian kernel density.

    Computed via log-space (sum of log-Gaussian PDFs) to avoid underflow when
    multiplying many small per-dimension Gaussian PDFs directly.

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    distance : jnp.ndarray, shape (n_samples, n_eval_points)

    """
    return jnp.exp(_log_kernel_matrix(eval_points, samples, std))


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    encoding_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,), optional
        Per-encoding-spike weights, by default None (uniform).

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    spike_waveform_feature_distance = kde_distance(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
    )  # shape (n_encoding_spikes, n_decoding_spikes)

    n_encoding_spikes = encoding_spike_waveform_features.shape[0]
    if encoding_weights is None:
        encoding_weights = jnp.ones((n_encoding_spikes,))
    # Weighted average over encoding spikes (each contributes w_e), normalized by the
    # weight total. Uniform weights recover the plain 1/n_encoding_spikes average.
    # Double-where: substitute a safe denominator, then select the result.
    weight_total = jnp.sum(encoding_weights)
    safe_weight_total = jnp.where(weight_total > 0, weight_total, 1.0)
    marginal_density = jnp.where(
        weight_total > 0,
        jnp.matmul(
            spike_waveform_feature_distance.T,
            encoding_weights[:, None] * position_distance,
            precision=jax.lax.Precision.HIGHEST,
        )
        / safe_weight_total,
        0.0,
    )  # shape (n_decoding_spikes, n_position_bins)
    return safe_log(
        mean_rate
        * jnp.where(
            occupancy > 0.0,
            marginal_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
            0.0,
        )
    )


@partial(jax.jit, static_argnames=("block_size",))
def _blocked_joint_mark_intensity(
    decoding_features,
    encoding_features,
    waveform_stds,
    occupancy,
    mean_rate,
    position_distance,
    encoding_weights,
    block_size,
    log_position_distance=None,
    log_occupancy=None,
):
    """Hoist the weighted kernel and update output inside one compiled loop."""
    weighted_position = encoding_weights[:, None] * position_distance
    weight_total = jnp.sum(encoding_weights)
    safe_weight_total = jnp.where(weight_total > 0, weight_total, 1.0)

    def update_block(number, output):
        first = number * block_size
        features = jax.lax.dynamic_slice(
            decoding_features, (first, 0), (block_size, decoding_features.shape[1])
        )
        mark_distance = kde_distance(features, encoding_features, waveform_stds)
        marginal_density = jnp.where(
            weight_total > 0,
            jnp.matmul(
                mark_distance.T, weighted_position, precision=jax.lax.Precision.HIGHEST
            )
            / safe_weight_total,
            0.0,
        )
        intensity = safe_log(
            mean_rate
            * jnp.where(
                occupancy > 0.0,
                marginal_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                0.0,
            )
        )
        if log_occupancy is not None:
            from non_local_detector.likelihoods.clusterless_kde_log import (
                estimate_log_joint_mark_intensity as log_estimate,
            )

            tail = (log_occupancy < LOG_EPS) | jnp.isnan(log_occupancy)
            intensity = jax.lax.cond(
                jnp.any(tail),
                lambda original: jnp.where(
                    tail[None, :],
                    log_estimate(
                        features,
                        encoding_features,
                        waveform_stds,
                        occupancy,
                        mean_rate,
                        log_position_distance,
                        encoding_weights=encoding_weights,
                        log_occupancy=log_occupancy,
                    ),
                    original,
                ),
                lambda original: original,
                intensity,
            )
        return jax.lax.dynamic_update_slice(output, intensity, (first, 0))

    output = jax.lax.fori_loop(
        0,
        decoding_features.shape[0] // block_size,
        update_block,
        jnp.zeros((decoding_features.shape[0], occupancy.shape[0])),
    )
    return jnp.clip(output, min=LOG_EPS, max=None)


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    block_size: int = 100,
    encoding_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
    waveform_stds : jnp.ndarray, shape (n_features,)
    occupancy : jnp.ndarray, shape (n_position_bins,)
    mean_rate : float
    position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,), optional
        Per-encoding-spike weights, by default None (uniform).
    block_size : int, optional

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    if n_decoding_spikes == 0:
        return jnp.zeros((0, n_position_bins))
    # A large encoding block setting must not inflate a sparse decode chunk's
    # spatial output to that requested capacity. No bucket policy is imposed.
    block_size = min(block_size, n_decoding_spikes)
    if encoding_weights is None:
        encoding_weights = jnp.ones((encoding_spike_waveform_features.shape[0],))
    # Decoded padding is internal and removed from the public result. Encoding
    # samples and normalization retain exactly the original caller's support.
    padding = (-n_decoding_spikes) % block_size
    features = jnp.pad(decoding_spike_waveform_features, ((0, padding), (0, 0)))
    return _blocked_joint_mark_intensity(
        features,
        encoding_spike_waveform_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
        encoding_weights,
        block_size,
    )[:n_decoding_spikes]


@partial(
    jax.jit, static_argnames=("block_size", "indices_are_sorted"), donate_argnums=0
)
def _add_electrode_mark_intensities(
    total,
    decoding_features,
    row_ids,
    encoding_features,
    encoding_positions,
    encoding_weights,
    waveform_stds,
    position_std,
    place_bin_centers,
    occupancy,
    mean_rate,
    log_occupancy,
    *,
    block_size,
    indices_are_sorted,
    log_position_distance=None,
):
    """Add one electrode's non-local log marked intensities to their rows.

    Parameters
    ----------
    total : jnp.ndarray, shape (n_rows, n_bins)
        Running sum over electrodes; donated.
    decoding_features : jnp.ndarray, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    encoding_features : jnp.ndarray, shape (n_padded_samples, n_features)
    encoding_positions : jnp.ndarray, shape (n_padded_samples, n_position_dims)
    encoding_weights : jnp.ndarray, shape (n_padded_samples,)
        Zero for padding samples.
    waveform_stds : jnp.ndarray, shape (n_features,)
    position_std : jnp.ndarray, shape (n_position_dims,)
    place_bin_centers : jnp.ndarray, shape (n_bins, n_position_dims)
    occupancy : jnp.ndarray, shape (n_bins,)
    mean_rate : float
        Mean rate times ``RATE_REFERENCE_SECONDS``.
    log_occupancy : jnp.ndarray, shape (n_bins,), or None
        Log occupancy at the bins. Bins in its tail (below ``LOG_EPS`` or
        NaN) take a log-space mark estimate; None skips that fallback.
    log_position_distance : jnp.ndarray, shape (n_padded_samples, n_bins), optional
        Log position kernel between encoding samples and bins; computed when
        None.

    Returns
    -------
    total : jnp.ndarray, shape (n_rows, n_bins)
    """
    if log_position_distance is None:
        log_position_distance = _log_kernel_matrix(
            place_bin_centers, encoding_positions, position_std
        )
    position_distance = jnp.exp(log_position_distance)
    intensities = _blocked_joint_mark_intensity(
        decoding_features,
        encoding_features,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
        encoding_weights,
        block_size,
        log_position_distance,
        log_occupancy,
    )
    return total + deterministic_row_sum(
        intensities, row_ids, total.shape[0], indices_are_sorted=indices_are_sorted
    )


@partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "gpi_block_size",
        "indices_are_sorted",
        "has_occupancy_tails",
    ),
    donate_argnums=(0, 1),
)
def _add_electrode_local_terms(
    log_likelihood,
    expected_counts,
    spike_positions,
    decoding_features,
    row_ids,
    encoding_positions,
    encoding_features,
    encoding_weights,
    position_std,
    waveform_stds,
    log_occupancy_at_spikes,
    gpi_samples,
    gpi_weights,
    gpi_std,
    positions,
    log_occupancy,
    scaled_rate,
    mean_rate,
    *,
    block_size,
    gpi_block_size,
    indices_are_sorted,
    has_occupancy_tails,
    log_gpi_density=None,
):
    """Add one electrode's local spike terms and expected count rate.

    Parameters
    ----------
    log_likelihood, expected_counts : jnp.ndarray, shape (n_rows,)
        Running sums over electrodes; donated.
    spike_positions : jnp.ndarray, shape (n_padded_spikes, n_position_dims)
        Position at each decoding spike; padding rows are zero.
    decoding_features : jnp.ndarray, shape (n_padded_spikes, n_features)
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    encoding_positions, encoding_features, encoding_weights
        Padded joint KDE samples and weights (zero for padding).
    position_std, waveform_stds : jnp.ndarray
    log_occupancy_at_spikes : jnp.ndarray, shape (n_padded_spikes,)
        True log occupancy density at ``spike_positions``.
    gpi_samples, gpi_weights, gpi_std
        This electrode's padded ground-process KDE leaves.
    positions : jnp.ndarray, shape (n_rows, n_position_dims)
        Animal position at each row.
    log_occupancy : jnp.ndarray, shape (n_rows,)
        Log occupancy density at ``positions``.
    scaled_rate : float
        Mean rate times ``RATE_REFERENCE_SECONDS``.
    mean_rate : float
        Mean rate in Hz.
    has_occupancy_tails : bool
        Whether some decoding spike's log occupancy is below ``LOG_EPS``.
        Those spikes take the log-space joint ratio; without any, it is not
        compiled.

    Returns
    -------
    log_likelihood, expected_counts : jnp.ndarray, shape (n_rows,)
    """
    marginal_density = _traced_block_kde(
        jnp.concatenate((spike_positions, decoding_features), axis=1),
        jnp.concatenate((encoding_positions, encoding_features), axis=1),
        jnp.concatenate((position_std, waveform_stds)),
        block_size,
        encoding_weights,
    )
    occupancy_at_spikes = jnp.exp(log_occupancy_at_spikes)
    spike_terms = safe_log(
        scaled_rate
        * jnp.where(
            occupancy_at_spikes > 0.0,
            marginal_density
            / jnp.where(occupancy_at_spikes > 0.0, occupancy_at_spikes, 1.0),
            0.0,
        )
    )
    if has_occupancy_tails:
        tail = log_occupancy_at_spikes < LOG_EPS
        spike_terms = jax.lax.cond(
            jnp.any(tail),
            lambda original: jnp.where(
                tail,
                jnp.maximum(
                    log_density_intensity(
                        scaled_rate,
                        local_kde_log_ratio(
                            spike_positions,
                            decoding_features,
                            encoding_positions,
                            encoding_features,
                            position_std,
                            waveform_stds,
                            log_occupancy_at_spikes,
                            encoding_weights,
                            block_size,
                        ),
                        jnp.zeros_like(log_occupancy_at_spikes),
                    ),
                    LOG_EPS,
                ),
                original,
            ),
            lambda original: original,
            spike_terms,
        )
    log_likelihood = log_likelihood + deterministic_row_sum(
        spike_terms,
        row_ids,
        log_likelihood.shape[0],
        indices_are_sorted=indices_are_sorted,
    )
    if log_gpi_density is None:
        log_gpi_density = traced_kde_log_density(
            positions, gpi_samples, gpi_std, gpi_block_size, gpi_weights
        )
    expected_counts = expected_counts + jnp.exp(
        log_density_intensity(mean_rate, log_gpi_density, log_occupancy)
    )
    return log_likelihood, expected_counts


def fit_clusterless_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    weights: jnp.ndarray | None = None,
    position_std: float = np.sqrt(12.5),
    waveform_std: float = 24.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    *,
    encoding_block_size: int | None = None,
    position_block_size: int | None = None,
    encoding_time_range=None,
    valid_position_intervals=None,
    _encoding_support=None,
) -> dict:
    """Fit the clusterless KDE encoding model.

    Parameters
    ----------
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Spike waveform features for each electrode.
    environment : Environment
        The spatial environment.
    weights : jnp.ndarray, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM), by default
        None (uniform). Weights the occupancy, per-electrode ground-process, and mean-
        rate fits, and are carried per encoding spike (``encoding_weights``) so the
        decode-time joint mark intensity is weighted too.
    position_std : float, optional
        Gaussian smoothing standard deviation for position, by default sqrt(12.5)
    waveform_std : float, optional
        Gaussian smoothing standard deviation for waveform, by default 24.0
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False
    encoding_block_size, position_block_size : int | None, optional
        Opt-in limits on sample and evaluation kernel tiles. Fitted samples and
        weights retain their original support. Either positive limit enables
        tiled evaluation during fit and prediction; both None keep the legacy
        path. The fitted model carries the selected policy to prediction.

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
    """
    encoding_block_size, position_block_size = _validate_workspace_limits(
        encoding_block_size, position_block_size
    )
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    if environment.place_bin_centers_ is None:
        raise ValueError(
            "Environment must be fitted with place_bin_centers_. "
            "Call environment.fit_place_grid() first."
        )

    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = validate_weights(np.asarray(weights), position.shape[0])
    # Weighted occupancy "time": the sum of per-sample weights (uniform weights recover
    # the training-sample count). Gaps from is_training / encoding-group masks are not
    # charged as occupancy time.
    weight_sum = float(exposure_weights.sum())
    # A track graph with multi-dim position linearizes occupancy to 1D, so the
    # bandwidth is a single dimension there; otherwise one per position column.
    n_std_dims = (
        1
        if (environment.track_graph is not None and position.shape[1] > 1)
        else position.shape[1]
    )
    position_std = validate_bandwidth(position_std, n_std_dims, "position_std")
    # Keep waveform_std as-is (scalar or array) - will be expanded per-electrode at predict time

    # Validate bandwidths once, host-side, at fit time. A zero/negative std is a
    # config error; catch it here (clear ValueError, cannot break JIT) rather
    # than silently producing a near-delta kernel downstream.
    for features in spike_waveform_features:
        validate_bandwidth(waveform_std, features.shape[1], "waveform_std")

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    occupancy_samples, occupancy_weights = drop_zero_weight_samples(
        position, exposure_weights
    )
    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        occupancy_samples = environment._linear_position(occupancy_samples)[:, None]
    occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
        occupancy_samples, weights=jnp.asarray(occupancy_weights)
    )

    log_occupancy = _predict_kde_log_density(
        occupancy_model,
        interior_place_bin_centers,
        encoding_block_size,
        position_block_size,
    )
    # The probability density is the exponential of the true log density.
    occupancy = jnp.exp(log_occupancy)
    encoding_positions = []
    encoding_weights = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)
    bounded_spike_waveform_features = []

    for electrode_spike_waveform_features, electrode_spike_times in zip(
        tqdm(
            spike_waveform_features,
            desc="Encoding models",
            unit="electrode",
            disable=disable_progress_bar,
        ),
        spike_times,
        strict=True,
    ):
        is_in_bounds = support.contains(electrode_spike_times)
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        # Validate only the in-window spikes that actually enter the fit; a
        # non-finite feature on a spike outside the encoding interval is
        # discarded here anyway and must not abort fitting.
        bounded_features = electrode_spike_waveform_features[is_in_bounds]
        validate_finite(bounded_features, "spike_waveform_features")
        bounded_spike_waveform_features.append(bounded_features)
        # Weight each encoding spike by the posterior weight at its spike time (linear
        # interpolation of the per-sample weights onto the spike times).
        electrode_weights_host = interpolate_weights_at_spike_times(
            electrode_spike_times,
            position_time,
            np.asarray(weights),
            encoding_support=support,
        )
        electrode_weights = jnp.asarray(electrode_weights_host)
        encoding_weights.append(electrode_weights)
        # Weighted mean rate: weighted spike count / weighted occupancy time. Sum on the
        # host array (as the GMM fit does) to avoid a device->host sync per electrode.
        mean_rates.append(weighted_mean_rate(electrode_weights_host, weight_sum))
        encoding_positions.append(
            get_position_at_time(
                position_time,
                position,
                electrode_spike_times,
                environment,
                encoding_support=support,
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1], weights=electrode_weights
        )
        gpi_models.append(gpi_model)

    if encoding_block_size is None and position_block_size is None:
        # One batched pass per group of similar sample counts, instead of a
        # compiled kernel for every electrode's sample count.
        log_gpi_densities = _kde_log_densities(gpi_models, interior_place_bin_centers)
    else:
        log_gpi_densities = [
            _predict_kde_log_density(
                gpi_model,
                interior_place_bin_centers,
                encoding_block_size,
                position_block_size,
            )
            for gpi_model in gpi_models
        ]
    for mean_rate, log_gpi_density in zip(mean_rates, log_gpi_densities, strict=True):
        summed_ground_process_intensity += jnp.exp(
            log_density_intensity(mean_rate, log_gpi_density, log_occupancy)
        )

    # Clip the summed intensity once (not per electrode) so an empty bin gets a
    # single EPS floor rather than accumulating n_electrodes * EPS.
    summed_ground_process_intensity = jnp.clip(
        summed_ground_process_intensity, min=RATE_EPS_HZ, max=None
    )

    encoding_model = {
        "occupancy": occupancy,
        "log_occupancy": log_occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
        "encoding_weights": encoding_weights,
        "environment": environment,
        "mean_rates": mean_rates,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "position_std": position_std,
        "waveform_std": waveform_std,
        "block_size": block_size,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }
    if encoding_block_size is not None or position_block_size is not None:
        encoding_model.update(
            encoding_block_size=encoding_block_size,
            position_block_size=position_block_size,
        )
    return encoding_model


@requires_time_edges
def predict_clusterless_kde_log_likelihood(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy: jnp.ndarray,
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: jnp.ndarray,
    environment: Environment,
    mean_rates: jnp.ndarray,
    summed_ground_process_intensity: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    is_local: bool = False,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    encoding_weights: list[jnp.ndarray] | None = None,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,
    log_occupancy: jnp.ndarray | None = None,
    encoding_block_size: int | None = None,
    position_block_size: int | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Predict the log likelihood of the clusterless KDE model.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample (used only by the local path; accepted for
        signature parity when ``is_local`` is False).
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples (used only by the local path; accepted for signature
        parity when ``is_local`` is False).
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Waveform features for each electrode.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        How much time is spent in each position bin by the animal.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jnp.ndarray]
        Spike waveform features for each electrode used for encoding.
    encoding_positions : jnp.ndarray, shape (n_encoding_spikes, n_position_dims)
        Position samples used for encoding.
    encoding_weights : list[jnp.ndarray], optional
        Per-encoding-spike weights for each electrode, by default None (uniform). Weights
        the joint mark intensity so a posterior-weighted (EM) encoding decodes correctly.
    environment : Environment
        The spatial environment
    mean_rates : jnp.ndarray, shape (n_electrodes,)
        Mean firing rate for each electrode.
    summed_ground_process_intensity : jnp.ndarray, shape (n_position_bins,)
        Summed ground process intensity for all electrodes.
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    is_local : bool, optional
        If True, compute the log likelihood at the animal's position, by default False
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False
    encoding_block_size, position_block_size : int | None, optional
        Optional fitted workspace policy. Encoding and spatial kernels are
        tiled independently and finished marked intensities accumulate directly
        into output rows. Both None preserve the legacy prediction path.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges: spikes are binned
        against them and only those owned by the requested rows are evaluated, so
        the result equals the full-time result sliced by ``row_slice`` while the
        spatial workspaces scale with the requested rows and selected spikes.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, 1) or (n_rows, n_position_bins)
        Shape depends on whether local or non-local decoding, respectively.
        ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    encoding_block_size, position_block_size = _validate_workspace_limits(
        encoding_block_size, position_block_size
    )
    if rate_units != "Hz":
        raise ValidationError(
            "Encoding rates must be in Hz; refit legacy encoding models before decoding."
        )
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        gpi_models=gpi_models,
        encoding_spike_waveform_features=encoding_spike_waveform_features,
        encoding_positions=encoding_positions,
        mean_rates=mean_rates,
        encoding_weights=encoding_weights,
    )
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    # Normalize to a per-electrode list; None -> uniform weights for each electrode.
    if encoding_weights is None:
        encoding_weights = [None] * len(encoding_positions)

    if is_local:
        log_likelihood = compute_local_log_likelihood(
            time_edges,
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            occupancy_model,
            gpi_models,
            encoding_spike_waveform_features,
            encoding_positions,
            environment,
            mean_rates,
            position_std,
            waveform_std,
            block_size,
            disable_progress_bar,
            encoding_weights=encoding_weights,
            row_slice=row_slice,
            _spike_time_order=_spike_time_order,
            _time_grid=_time_grid,
            encoding_block_size=encoding_block_size,
            position_block_size=position_block_size,
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]
        if log_occupancy is None:
            log_occupancy = _predict_kde_log_density(
                occupancy_model,
                interior_place_bin_centers,
                encoding_block_size,
                position_block_size,
            )
        # Bins in occupancy tails take a log-space mark estimate; without any,
        # passing None keeps that fallback out of every electrode's kernel.
        host_log_occupancy = np.asarray(log_occupancy)
        tail_log_occupancy = (
            log_occupancy
            if np.any((host_log_occupancy < LOG_EPS) | np.isnan(host_log_occupancy))
            else None
        )

        log_likelihood = (
            -jnp.asarray(_time_grid.durations(row_start, row_stop))[:, None]
            * summed_ground_process_intensity
        )
        place_bin_centers = _prepared_kde_nonlocal_points(
            environment, _spike_time_order
        )
        position_std_array = jnp.asarray(position_std)

        for (
            electrode_encoding_spike_waveform_features,
            electrode_encoding_positions,
            electrode_encoding_weights,
            electrode_mean_rate,
            electrode_decoding_spike_waveform_features,
            electrode_spike_times,
        ) in zip(
            tqdm(
                encoding_spike_waveform_features,
                unit="electrode",
                desc="Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_positions,
            encoding_weights,
            mean_rates,
            spike_waveform_features,
            spike_times,
            strict=True,
        ):
            selection = select_spikes_in_rows(
                electrode_spike_times,
                row_start,
                row_stop,
                time_edges=time_edges,
                _spike_time_order=_spike_time_order,
            )
            electrode_decoding_spike_waveform_features = select_spike_rows(
                electrode_decoding_spike_waveform_features, selection
            )
            if encoding_block_size is not None or position_block_size is not None:
                # Accumulate completed mark tiles into rows without retaining
                # an encoding-by-position or all-marks-by-position matrix.
                n_spikes = electrode_decoding_spike_waveform_features.shape[0]
                if n_spikes == 0:
                    continue
                # Padded spikes (dropped from the row sums) let chunks with
                # nearby spike counts share one compiled kernel.
                n_padded = _padded_spike_count(n_spikes, block_size)
                n_waveform_features = electrode_encoding_spike_waveform_features.shape[
                    1
                ]
                electrode_waveform_std = as_std_array(waveform_std, n_waveform_features)
                log_likelihood += _streamed_joint_mark_row_sums(
                    _pad_rows(electrode_decoding_spike_waveform_features, n_padded),
                    electrode_encoding_spike_waveform_features,
                    electrode_encoding_positions,
                    interior_place_bin_centers,
                    electrode_waveform_std,
                    position_std,
                    occupancy,
                    electrode_mean_rate * RATE_REFERENCE_SECONDS,
                    electrode_encoding_weights,
                    row_indices=spike_row_ids(selection, n_padded),
                    n_rows=selection.n_rows,
                    encoding_tile_size=_tile_size(
                        encoding_block_size,
                        electrode_encoding_spike_waveform_features.shape[0],
                    ),
                    position_tile_size=_tile_size(
                        position_block_size, interior_place_bin_centers.shape[0]
                    ),
                    decoding_tile_size=_tile_size(block_size, n_padded),
                    log_occupancy=tail_log_occupancy,
                )
                continue
            n_spikes = electrode_decoding_spike_waveform_features.shape[0]
            if n_spikes == 0:
                continue
            # Padded spikes and zero-weight encoding samples let electrodes and
            # chunks with nearby sizes share one compiled kernel.
            n_padded = _padded_spike_count(n_spikes, block_size)
            padded_features, padded_weights = _padded_samples(
                _spike_time_order,
                electrode_encoding_spike_waveform_features,
                electrode_encoding_weights,
            )
            padded_positions, _ = _padded_samples(
                _spike_time_order,
                electrode_encoding_positions,
                electrode_encoding_weights,
            )
            # Expand waveform_std to match this electrode's feature count if scalar
            n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]
            log_likelihood = _add_electrode_mark_intensities(
                log_likelihood,
                jnp.asarray(
                    _pad_rows(electrode_decoding_spike_waveform_features, n_padded)
                ),
                jnp.asarray(spike_row_ids(selection, n_padded)),
                padded_features,
                padded_positions,
                padded_weights,
                as_std_array(waveform_std, n_waveform_features),
                position_std_array,
                place_bin_centers,
                occupancy,
                electrode_mean_rate * RATE_REFERENCE_SECONDS,
                tail_log_occupancy,
                block_size=min(block_size, n_padded),
                indices_are_sorted=selection.indices_are_sorted,
                log_position_distance=_prepared_kde_position_log_kernel(
                    _spike_time_order, place_bin_centers, padded_positions, position_std
                ),
            )

    return (
        log_likelihood
        + log_bin_duration_evidence(
            spike_times,
            time_edges,
            row_slice,
            _spike_time_order,
            intensity_time_scale=RATE_REFERENCE_SECONDS,
            _time_grid=_time_grid,
        )[:, None]
    )


def compute_local_log_likelihood(
    time_edges: np.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    occupancy_model: KDEModel,
    gpi_models: list[KDEModel],
    encoding_spike_waveform_features: list[jnp.ndarray],
    encoding_positions: jnp.ndarray,
    environment: Environment,
    mean_rates: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    block_size: int = 100,
    disable_progress_bar: bool = False,
    encoding_weights: list[jnp.ndarray] | None = None,
    row_slice: slice | None = None,
    *,
    encoding_block_size: int | None = None,
    position_block_size: int | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jnp.ndarray:
    """Compute the log likelihood at the animal's position.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        List of spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        List of spike waveform features for each electrode.
    occupancy_model : KDEModel
        KDE model for occupancy.
    gpi_models : list[KDEModel]
        List of KDE models for the ground process intensity.
    encoding_spike_waveform_features : list[jnp.ndarray]
        List of spike waveform features for each electrode used for encoding.
    encoding_positions : jnp.ndarray
        Position samples used for encoding.
    environment : Environment
        The spatial environment.
    mean_rates : jnp.ndarray
        Mean firing rate for each electrode.
    encoding_weights : list[jnp.ndarray], optional
        Per-encoding-spike weights for each electrode, by default None (uniform).
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False
    encoding_block_size, position_block_size : int | None, optional
        Positive sample/evaluation tile limits for the fitted linear KDE leaves.
        Both None retain the original local kernels.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` stay the FULL decoding edges (see
        ``predict_clusterless_kde_log_likelihood``).
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, 1)
    """
    encoding_block_size, position_block_size = _validate_workspace_limits(
        encoding_block_size, position_block_size
    )
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_rows = row_stop - row_start

    density_policy = f"{encoding_block_size}:{position_block_size}"
    predict_density = partial(
        _predict_kde_log_density,
        encoding_block_size=encoding_block_size,
        position_block_size=position_block_size,
    )
    interpolated_position, log_occupancy = _prepare_kde_local_inputs(
        occupancy_model,
        position_time,
        position,
        environment,
        _time_grid,
        row_start,
        row_stop,
        _spike_time_order,
        predict_log=predict_density,
        policy=density_policy,
    )

    if encoding_weights is None:
        encoding_weights = [None] * len(encoding_positions)
    log_likelihood = jnp.zeros((n_rows,))
    summed_expected_counts = jnp.zeros((n_rows,))
    is_streamed = encoding_block_size is not None or position_block_size is not None
    if not is_streamed:
        positions = jnp.asarray(interpolated_position)
        n_position_dims = positions.shape[1]
        position_std_array = jnp.asarray(position_std)
        occupancy_std = as_std_array(occupancy_model.std, n_position_dims)
    for (
        electrode_encoding_spike_waveform_features,
        electrode_encoding_positions,
        electrode_encoding_weights,
        electrode_mean_rate,
        electrode_gpi_model,
        electrode_decoding_spike_waveform_features,
        electrode_spike_times,
    ) in zip(
        tqdm(
            encoding_spike_waveform_features,
            unit="electrode",
            desc="Local Likelihood",
            disable=disable_progress_bar,
        ),
        encoding_positions,
        encoding_weights,
        mean_rates,
        gpi_models,
        spike_waveform_features,
        spike_times,
        strict=True,
    ):
        selection = select_spikes_in_rows(
            electrode_spike_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        electrode_spike_times = select_spike_rows(electrode_spike_times, selection)
        electrode_decoding_spike_waveform_features = select_spike_rows(
            electrode_decoding_spike_waveform_features, selection
        )

        position_at_spike_time = get_position_at_time(
            position_time, position, electrode_spike_times, environment
        )

        # Expand waveform_std to match this electrode's feature count if scalar
        n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]
        electrode_waveform_std = as_std_array(waveform_std, n_waveform_features)

        if not is_streamed:
            # Padded spikes and zero-weight samples let electrodes and chunks
            # with nearby sizes share one compiled kernel.
            n_padded = _padded_spike_count(position_at_spike_time.shape[0], block_size)
            padded_positions, padded_weights = _padded_samples(
                _spike_time_order,
                electrode_encoding_positions,
                electrode_encoding_weights,
            )
            padded_features, _ = _padded_samples(
                _spike_time_order,
                electrode_encoding_spike_waveform_features,
                electrode_encoding_weights,
            )
            gpi_samples, gpi_weights = _padded_samples(
                _spike_time_order,
                electrode_gpi_model.samples_,
                electrode_gpi_model.weights_,
            )
            occupancy_samples, occupancy_weights = _padded_samples(
                _spike_time_order,
                occupancy_model.samples_,
                occupancy_model.weights_,
                count=_padded_occupancy_count,
            )
            padded_spike_positions = jnp.asarray(
                _pad_rows(position_at_spike_time, n_padded)
            )
            # Occupancy at the spikes is evaluated before the kernel, so the
            # kernel compiles its tail fallback only when some spike needs it.
            log_occupancy_at_spikes = _true_kde_log_density(
                padded_spike_positions,
                occupancy_samples,
                occupancy_std,
                occupancy_model.block_size or n_padded,
                occupancy_weights,
            )
            # Padding rows are dropped from the row sums.
            has_occupancy_tails = bool(
                np.any(
                    np.asarray(log_occupancy_at_spikes)[
                        : position_at_spike_time.shape[0]
                    ]
                    < LOG_EPS
                )
            )
            log_likelihood, summed_expected_counts = _add_electrode_local_terms(
                log_likelihood,
                summed_expected_counts,
                padded_spike_positions,
                jnp.asarray(
                    _pad_rows(electrode_decoding_spike_waveform_features, n_padded)
                ),
                jnp.asarray(spike_row_ids(selection, n_padded)),
                padded_positions,
                padded_features,
                padded_weights,
                position_std_array,
                electrode_waveform_std,
                log_occupancy_at_spikes,
                gpi_samples,
                gpi_weights,
                as_std_array(electrode_gpi_model.std, n_position_dims),
                positions,
                log_occupancy,
                electrode_mean_rate * RATE_REFERENCE_SECONDS,
                electrode_mean_rate,
                block_size=min(block_size, n_padded),
                gpi_block_size=electrode_gpi_model.block_size or n_rows,
                indices_are_sorted=selection.indices_are_sorted,
                has_occupancy_tails=has_occupancy_tails,
                log_gpi_density=_prepared_kde_log_density(
                    electrode_gpi_model,
                    interpolated_position,
                    _spike_time_order,
                    predict_log=predict_density,
                    policy=density_policy,
                ),
            )
            continue

        marginal_density = _sample_tiled_density(
            eval_points=jnp.concatenate(
                (
                    position_at_spike_time,
                    electrode_decoding_spike_waveform_features,
                ),
                axis=1,
            ),
            samples=jnp.concatenate(
                (
                    electrode_encoding_positions,
                    electrode_encoding_spike_waveform_features,
                ),
                axis=1,
            ),
            std=jnp.concatenate((position_std, electrode_waveform_std)),
            weights=electrode_encoding_weights,
            sample_tile_size=_tile_size(
                encoding_block_size, electrode_encoding_positions.shape[0]
            ),
            eval_tile_size=_tile_size(
                position_block_size if position_block_size is not None else block_size,
                position_at_spike_time.shape[0],
            ),
        )
        log_occ_at_spikes = _predict_kde_log_density(
            occupancy_model,
            position_at_spike_time,
            encoding_block_size,
            position_block_size,
        )
        occupancy_at_spike_time = jnp.exp(log_occ_at_spikes)
        spike_terms = safe_log(
            (electrode_mean_rate * RATE_REFERENCE_SECONDS)
            * jnp.where(
                occupancy_at_spike_time > 0.0,
                marginal_density
                / jnp.where(
                    occupancy_at_spike_time > 0.0, occupancy_at_spike_time, 1.0
                ),
                0.0,
            )
        )
        # Only spikes in occupancy tails need the log-space joint ratio.
        tail = np.asarray((log_occ_at_spikes < LOG_EPS) | jnp.isnan(log_occ_at_spikes))
        if tail.any():
            log_ratio = _sample_tiled_log_ratio(
                jnp.concatenate(
                    (
                        position_at_spike_time,
                        electrode_decoding_spike_waveform_features,
                    ),
                    axis=1,
                )[tail],
                jnp.concatenate(
                    (
                        electrode_encoding_positions,
                        electrode_encoding_spike_waveform_features,
                    ),
                    axis=1,
                ),
                jnp.concatenate((position_std, electrode_waveform_std)),
                jnp.ones(electrode_encoding_positions.shape[0])
                if electrode_encoding_weights is None
                else electrode_encoding_weights,
                log_occ_at_spikes[tail],
                n_position_dims=position_at_spike_time.shape[1],
                sample_tile_size=_tile_size(
                    encoding_block_size, electrode_encoding_positions.shape[0]
                ),
                eval_tile_size=_tile_size(
                    position_block_size
                    if position_block_size is not None
                    else block_size,
                    int(tail.sum()),
                ),
            )
            spike_terms = spike_terms.at[np.flatnonzero(tail)].set(
                jnp.maximum(
                    log_density_intensity(
                        electrode_mean_rate * RATE_REFERENCE_SECONDS,
                        log_ratio,
                        jnp.zeros_like(log_ratio),
                    ),
                    LOG_EPS,
                )
            )

        log_likelihood += sum_spikes_into_rows(spike_terms, selection)

        summed_expected_counts += jnp.exp(
            log_density_intensity(
                electrode_mean_rate,
                _prepared_kde_log_density(
                    electrode_gpi_model,
                    interpolated_position,
                    _spike_time_order,
                    predict_log=predict_density,
                    policy=density_policy,
                ),
                log_occupancy,
            )
        )

    # Subtract the summed ground-process intensity once, floored at EPS to
    # mirror fit_clusterless_kde_encoding_model's summed_ground_process_intensity
    # (a single EPS floor, not n_electrodes * EPS).
    log_likelihood -= jnp.asarray(_time_grid.durations(row_start, row_stop)) * jnp.clip(
        summed_expected_counts, min=RATE_EPS_HZ
    )
    return log_likelihood[:, jnp.newaxis]
