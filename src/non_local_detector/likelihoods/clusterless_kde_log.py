import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    KDEModel,
    block_kde,
    gaussian_pdf,
    get_position_at_time,
    log_gaussian_pdf,
)


def get_spike_time_bin_ind(spike_times: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Get the index of the time bin for each spike time.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
    time : np.ndarray, shape (n_time_bins,)
        Bin edges.

    Returns
    -------
    ind : np.ndarray, shape (n_spikes,)
    """
    return np.digitize(spike_times, time[1:-1])


def kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Distance between evaluation points and samples using Gaussian kernel density

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
    distance = jnp.ones((samples.shape[0], eval_points.shape[0]))
    for dim_eval_points, dim_samples, dim_std in zip(
        eval_points.T, samples.T, std, strict=False
    ):
        distance *= gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return distance


@jax.jit
def log_kde_distance(
    eval_points: jnp.ndarray, samples: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Log-distance (log kernel product) between eval points and samples using Gaussian kernels.

    Computes:
        log_distance[i, j] = sum_d log N(eval_points[j, d] | samples[i, d], std[d])

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (n_eval_points, n_dims)
        Evaluation points.
    samples : jnp.ndarray, shape (n_samples, n_dims)
        Training samples.
    std : jnp.ndarray, shape (n_dims,)
        Per-dimension kernel std.

    Returns
    -------
    log_distance : jnp.ndarray, shape (n_samples, n_eval_points)
        Log of the product of per-dimension Gaussian kernels.
    """
    log_dist = jnp.zeros((samples.shape[0], eval_points.shape[0]))
    for dim_eval, dim_samp, dim_std in zip(eval_points.T, samples.T, std, strict=False):
        log_dist += log_gaussian_pdf(
            jnp.expand_dims(dim_eval, axis=0),  # (1, n_eval)
            jnp.expand_dims(dim_samp, axis=1),  # (n_samples, 1)
            dim_std,
        )
    return log_dist


def _compute_log_mark_kernel_gemm(
    decoding_features: jnp.ndarray,
    encoding_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log mark kernel using GEMM (matrix multiplication) instead of per-dimension loop.

    This is mathematically equivalent to the loop-based approach but much faster for
    multi-dimensional features. The Gaussian kernel in log-space:

        log K(x, y) = -0.5 * sum_d [(x_d - y_d)^2 / sigma_d^2] - log_norm_const
                    = -0.5 * sum_d [(x_d/sigma_d)^2 + (y_d/sigma_d)^2 - 2*(x_d/sigma_d)*(y_d/sigma_d)] - log_norm_const
                    = -0.5 * (||x_scaled||^2 + ||y_scaled||^2 - 2 * x_scaled @ y_scaled^T) - log_norm_const

    The cross term x_scaled @ y_scaled^T is a single matrix multiply (GEMM).

    Parameters
    ----------
    decoding_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
        Waveform features for decoding spikes.
    encoding_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
        Waveform features for encoding spikes.
    waveform_stds : jnp.ndarray, shape (n_features,)
        Standard deviations for each feature dimension.

    Returns
    -------
    logK_mark : jnp.ndarray, shape (n_encoding_spikes, n_decoding_spikes)
        Log kernel matrix K[i, j] = log(Gaussian kernel between encoding spike i and decoding spike j).
    """
    n_features = waveform_stds.shape[0]

    # Precompute inverse standard deviations and normalization constant
    inv_sigma = 1.0 / waveform_stds  # (n_features,)

    # Log normalization constant: -0.5 * (D * log(2π) + 2 * sum(log(sigma)))
    # Factor of 2 because we have sum of log(sigma), not log(sigma^2)
    log_norm_const = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )

    # Scale features by inverse standard deviations
    Y = encoding_features * inv_sigma[None, :]  # (n_enc, n_features)
    X = decoding_features * inv_sigma[None, :]  # (n_dec, n_features)

    # Compute squared norms
    y2 = jnp.sum(Y**2, axis=1)  # (n_enc,)
    x2 = jnp.sum(X**2, axis=1)  # (n_dec,)

    # GEMM: compute cross terms X @ Y^T = (n_dec, n_features) @ (n_features, n_enc)
    cross_term = X @ Y.T  # (n_dec, n_enc)

    # Combine: log K[i,j] = -0.5 * (y2[i] + x2[j] - 2*cross_term[j,i]) + log_norm_const
    # Note: We need (n_enc, n_dec) output, so transpose the cross term
    logK_mark = log_norm_const - 0.5 * (
        y2[:, None] + x2[None, :] - 2.0 * cross_term.T
    )  # (n_enc, n_dec)

    return logK_mark


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
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
    use_gemm : bool, optional
        If True (default), use GEMM-based log-space computation (faster for multi-dimensional features).
        If False, use linear-space computation (matches reference exactly).
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension in chunks (only for use_gemm=True).

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_encoding_spikes = encoding_spike_waveform_features.shape[0]

    if not use_gemm:
        # Linear-space computation (matches reference exactly)
        spike_waveform_feature_distance = kde_distance(
            decoding_spike_waveform_features,
            encoding_spike_waveform_features,
            waveform_stds,
        )  # shape (n_encoding_spikes, n_decoding_spikes)

        marginal_density = (
            spike_waveform_feature_distance.T @ position_distance / n_encoding_spikes
        )  # shape (n_decoding_spikes, n_position_bins)
        return jnp.log(
            mean_rate * jnp.where(occupancy > 0.0, marginal_density / occupancy, 0.0)
        )

    # Log-space computation with GEMM optimization
    # Build log-kernel matrix for marks: (n_enc, n_dec)
    logK_mark = _compute_log_mark_kernel_gemm(
        decoding_spike_waveform_features,
        encoding_spike_waveform_features,
        waveform_stds,
    )

    # Convert position_distance to log-space
    log_position_distance = jnp.log(position_distance)

    # Uniform weights: log(1/n) for each encoding spike
    log_w = -jnp.log(float(n_encoding_spikes))

    # Use scan to avoid materializing (n_enc × n_dec × n_pos) array
    n_pos = log_position_distance.shape[1]
    n_dec = logK_mark.shape[1]

    if pos_tile_size is None or pos_tile_size >= n_pos:
        # No tiling: process all positions at once (default, fastest)
        def scan_over_dec(carry, y_col: jnp.ndarray) -> tuple[None, jnp.ndarray]:
            # y_col: (n_enc,), the column of logK_mark for one decoding spike
            # returns: (n_pos,), logsumexp over enc dimension
            result = jax.nn.logsumexp(
                log_w + log_position_distance + y_col[:, None], axis=0
            )
            return None, result

        # scan over decoding spikes' columns -> (n_dec, n_pos)
        _, log_marginal = jax.lax.scan(scan_over_dec, None, logK_mark.T)
    else:
        # Tiled: process positions in chunks to reduce peak memory
        log_marginal = jnp.zeros((n_dec, n_pos))

        for pos_start in range(0, n_pos, pos_tile_size):
            pos_end = min(pos_start + pos_tile_size, n_pos)
            pos_slice = slice(pos_start, pos_end)

            # Tile: slice of log_position_distance for this chunk of positions
            log_pos_tile = log_position_distance[:, pos_slice]  # (n_enc, tile_size)

            # Create closure to capture log_pos_tile properly
            def make_scan_fn(tile):
                def scan_over_dec_tile(
                    carry, y_col: jnp.ndarray
                ) -> tuple[None, jnp.ndarray]:
                    # y_col: (n_enc,)
                    # returns: (tile_size,), logsumexp over enc dimension
                    result = jax.nn.logsumexp(log_w + tile + y_col[:, None], axis=0)
                    return None, result

                return scan_over_dec_tile

            # scan over decoding spikes for this position tile -> (n_dec, tile_size)
            _, log_marginal_tile = jax.lax.scan(
                make_scan_fn(log_pos_tile), None, logK_mark.T
            )

            # Update output with this tile
            log_marginal = log_marginal.at[:, pos_slice].set(log_marginal_tile)

    # Add mean rate and subtract occupancy (in log)
    log_mean_rate = jnp.log(mean_rate)
    log_occ = jnp.log(jnp.where(occupancy > 0.0, occupancy, 1.0))  # avoid log(0)

    # Result: log(mean_rate * marginal / occupancy)
    # Use where to handle occupancy = 0 cases
    log_joint = jnp.where(
        occupancy[None, :] > 0.0,
        log_mean_rate + log_marginal - log_occ[None, :],
        jnp.log(0.0),  # -inf for zero occupancy
    )

    return log_joint


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    block_size: int = 100,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
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
    block_size : int, optional
    use_gemm : bool, optional
        If True (default), use GEMM-based log-space computation.
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension.

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    if n_decoding_spikes == 0:
        return jnp.full((0, n_position_bins), LOG_EPS)

    # Use JIT-compiled update with buffer donation for memory efficiency
    # Donate the accumulator buffer (arg 0) so it can be reused in-place
    @jax.jit
    def _update_block(out_array, block_result, start_idx):
        return jax.lax.dynamic_update_slice(out_array, block_result, (start_idx, 0))

    out = jnp.zeros((n_decoding_spikes, n_position_bins))
    for start_ind in range(0, n_decoding_spikes, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        block_result = estimate_log_joint_mark_intensity(
            decoding_spike_waveform_features[block_inds],
            encoding_spike_waveform_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
            use_gemm=use_gemm,
            pos_tile_size=pos_tile_size,
        )
        out = _update_block(out, block_result, start_ind)

    return jnp.clip(out, a_min=LOG_EPS, a_max=None)


def fit_clusterless_kde_encoding_model(
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    sampling_frequency: int = 500,
    weights: jnp.ndarray | None = None,
    position_std: float = np.sqrt(12.5),
    waveform_std: float = 24.0,
    block_size: int = 100,
    disable_progress_bar: bool = False,
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
    sampling_frequency : int, optional
        Samples per second, by default 500
    position_std : float, optional
        Gaussian smoothing standard deviation for position, by default sqrt(12.5)
    waveform_std : float, optional
        Gaussian smoothing standard deviation for waveform, by default 24.0
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    Returns
    -------
    encoding_model : dict
    """
    if environment.place_bin_centers_ is None:
        raise ValueError(
            "Environment must be fitted with place_bin_centers_. "
            "Call environment.fit_place_grid() first."
        )

    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if isinstance(position_std, (int, float)):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])
    if isinstance(waveform_std, (int, float)):
        waveform_std = jnp.array([waveform_std] * spike_waveform_features[0].shape[1])

    is_track_interior = environment.is_track_interior_.ravel()
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        position1D = get_linearized_position(
            position,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position1D
        )
    else:
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position
        )

    occupancy = occupancy_model.predict(interior_place_bin_centers)
    encoding_positions = []
    mean_rates = []
    gpi_models = []
    summed_ground_process_intensity = jnp.zeros_like(occupancy)

    n_time_bins = int((position_time[-1] - position_time[0]) * sampling_frequency)
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
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= position_time[0],
            electrode_spike_times <= position_time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        bounded_spike_waveform_features.append(
            electrode_spike_waveform_features[is_in_bounds]
        )
        mean_rates.append(len(electrode_spike_times) / n_time_bins)
        encoding_positions.append(
            get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1]
        )
        gpi_models.append(gpi_model)

        gpi_density = gpi_model.predict(interior_place_bin_centers)
        summed_ground_process_intensity += jnp.clip(
            mean_rates[-1] * jnp.where(occupancy > 0.0, gpi_density / occupancy, EPS),
            a_min=EPS,
            a_max=None,
        )

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
        "environment": environment,
        "mean_rates": mean_rates,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "position_std": position_std,
        "waveform_std": waveform_std,
        "block_size": block_size,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_clusterless_kde_log_likelihood(
    time: jnp.ndarray,
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
) -> jnp.ndarray:
    """Predict the log likelihood of the clusterless KDE model.

    Parameters
    ----------
    time : jnp.ndarray
        Decoding time bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
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

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1) or (n_time, n_position_bins)
        Shape depends on whether local or non-local decoding, respectively.
    """
    n_time = len(time)

    if is_local:
        log_likelihood = compute_local_log_likelihood(
            time,
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
        )
    else:
        is_track_interior = environment.is_track_interior_.ravel()
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

        log_likelihood = -1.0 * summed_ground_process_intensity * jnp.ones((n_time, 1))

        for (
            electrode_encoding_spike_waveform_features,
            electrode_encoding_positions,
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
            mean_rates,
            spike_waveform_features,
            spike_times,
            strict=True,
        ):
            is_in_bounds = jnp.logical_and(
                electrode_spike_times >= time[0],
                electrode_spike_times <= time[-1],
            )
            electrode_spike_times = electrode_spike_times[is_in_bounds]
            electrode_decoding_spike_waveform_features = (
                electrode_decoding_spike_waveform_features[is_in_bounds]
            )
            position_distance = kde_distance(
                interior_place_bin_centers,
                electrode_encoding_positions,
                std=position_std,
            )
            log_likelihood += jax.ops.segment_sum(
                block_estimate_log_joint_mark_intensity(
                    electrode_decoding_spike_waveform_features,
                    electrode_encoding_spike_waveform_features,
                    waveform_std,
                    occupancy,
                    electrode_mean_rate,
                    position_distance,
                    block_size,
                ),
                get_spike_time_bin_ind(electrode_spike_times, time),
                indices_are_sorted=True,
                num_segments=n_time,
            )

    return log_likelihood


def compute_local_log_likelihood(
    time: jnp.ndarray,
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
) -> jnp.ndarray:
    """Compute the log likelihood at the animal's position.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
        Time bins for decoding.
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
    position_std : jnp.ndarray
        Gaussian smoothing standard deviation for position.
    waveform_std : jnp.ndarray
        Gaussian smoothing standard deviation for waveform.
    block_size : int, optional
        Divide computation into blocks, by default 100
    disable_progress_bar : bool, optional
        Turn off progress bar, by default False

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, 1)
    """

    # Need to interpolate position
    interpolated_position = get_position_at_time(
        position_time, position, time, environment
    )
    occupancy = occupancy_model.predict(interpolated_position)

    n_time = len(time)
    log_likelihood = jnp.zeros((n_time,))
    for (
        electrode_encoding_spike_waveform_features,
        electrode_encoding_positions,
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
        mean_rates,
        gpi_models,
        spike_waveform_features,
        spike_times,
        strict=False,
    ):
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= time[0],
            electrode_spike_times <= time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        electrode_decoding_spike_waveform_features = (
            electrode_decoding_spike_waveform_features[is_in_bounds]
        )

        position_at_spike_time = get_position_at_time(
            position_time, position, electrode_spike_times, environment
        )

        marginal_density = block_kde(
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
            std=jnp.concatenate((position_std, waveform_std)),
            block_size=block_size,
        )
        occupancy_at_spike_time = occupancy_model.predict(position_at_spike_time)

        log_likelihood += jax.ops.segment_sum(
            jnp.log(
                electrode_mean_rate
                * jnp.where(
                    occupancy_at_spike_time > 0.0,
                    marginal_density / occupancy_at_spike_time,
                    0.0,
                )
            ),
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * jnp.where(
            occupancy > 0.0,
            electrode_gpi_model.predict(interpolated_position) / occupancy,
            0.0,
        )
    return log_likelihood[:, jnp.newaxis]
