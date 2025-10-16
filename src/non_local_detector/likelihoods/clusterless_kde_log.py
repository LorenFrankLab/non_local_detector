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
    block_log_kde,
    get_position_at_time,
    log_gaussian_pdf,
    safe_divide,
    safe_log,
)


def get_spike_time_bin_ind(
    spike_times: np.ndarray, time_bin_edges: np.ndarray
) -> np.ndarray:
    """Gets the index of the time bin for each spike time.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Times of occurrences (spikes).
    time_bin_edges : np.ndarray, shape (n_bins + 1,)
        Sorted array defining the edges of the time bins [t0, t1, ..., tN].
        Defines n_bins intervals: [t0, t1), ..., [t_{N-1}, tN].

    Returns
    -------
    bin_indices : np.ndarray, shape (n_spikes,)
        Index of the bin for each spike. Bins are indexed 0 to n_bins-1.
        Doesn't handle out of bounds spikes.
    """

    bin_indices = np.searchsorted(time_bin_edges, spike_times, side="right") - 1
    is_last_bin = np.isclose(
        spike_times,
        time_bin_edges[-1],
    )
    bin_indices[is_last_bin] = len(time_bin_edges) - 2

    return bin_indices


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


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    encoding_weights: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity of decoding spikes and spike waveforms (log-space).

    Computes the log of the joint intensity λ(mark, pos) for marked point processes
    using kernel density estimation in log space for numerical stability.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
        Waveform features for spikes during decoding period.
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
        Waveform features for spikes during encoding period.
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,)
        Weights for each encoding spike.
    waveform_stds : jnp.ndarray, shape (n_features,)
        Standard deviations for Gaussian kernels over waveform dimensions.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        Spatial occupancy density.
    mean_rate : float
        Mean firing rate for the electrode.
    log_position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
        Log of position-based kernel distances between encoding spikes and position bins.
    use_gemm : bool, optional
        If True (default), use GEMM-based computation for mark kernel (faster for multi-dimensional features).
        If False, use per-dimension loop (slower but equivalent).
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension in chunks of this size.
        Reduces peak memory from O(n_enc * n_pos) to O(n_enc * pos_tile_size).
        If None (default), process all positions at once (fastest but more memory).
        Useful for very large position grids (> 2000 bins).

    Returns
    -------
    log_joint_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)
        Log joint mark intensity λ(mark, pos) for each decoding spike at each position.
    """

    # 2) Build log-kernel matrix for marks: (n_enc, n_dec)
    #    Two equivalent approaches:
    #    - GEMM: Single matrix multiply (faster, especially for many features)
    #    - Loop: Sum log-Gaussians over mark dimensions (slower but simpler)

    if use_gemm:
        # GEMM approach: O(n_enc * n_dec * n_features) via single matmul
        logK_mark = _compute_log_mark_kernel_gemm(
            decoding_spike_waveform_features,
            encoding_spike_waveform_features,
            waveform_stds,
        )
    else:
        # Loop approach: O(n_enc * n_dec * n_features) via n_features separate operations
        n_enc = encoding_spike_waveform_features.shape[0]
        n_dec = decoding_spike_waveform_features.shape[0]
        logK_mark = jnp.zeros((n_enc, n_dec))
        for dec_dim, enc_dim, std_d in zip(
            decoding_spike_waveform_features.T,
            encoding_spike_waveform_features.T,
            waveform_stds,
            strict=False,
        ):
            # broadcast to (n_enc, n_dec): each column is a decoding spike, rows are encoding spikes
            logK_mark += log_gaussian_pdf(
                x=jnp.expand_dims(dec_dim, axis=0),  # (1, n_dec)
                mean=jnp.expand_dims(enc_dim, axis=1),  # (n_enc, 1)
                sigma=std_d,
            )

    # 3) Weighted log-sum-exp across encoding spikes for each (decoding, position) pair
    #    log sum_i [ w_i * exp(logK_mark[i,dec]) * exp(logK_pos[i,pos]) ]
    #  = logsumexp_i ( log w_i + logK_mark[i,dec] + logK_pos[i,pos] )
    log_w = safe_log(encoding_weights)  # (n_enc,)
    # Use logsumexp for denominator (more numerically stable when weights vary by orders of magnitude)
    log_den = jax.nn.logsumexp(log_w)  # scalar

    # Use scan instead of vmap to avoid materializing (n_enc × n_dec × n_pos) array
    # This reduces memory from O(n_enc * n_dec * n_pos) to O(n_enc * n_pos)

    n_pos = log_position_distance.shape[1]
    n_dec = logK_mark.shape[1]

    if pos_tile_size is None or pos_tile_size >= n_pos:
        # No tiling: process all positions at once (default, fastest)
        def scan_over_dec(carry, y_col: jnp.ndarray) -> tuple[None, jnp.ndarray]:
            # y_col: (n_enc,), the column of logK_mark for one decoding spike
            # returns: (n_pos,), logsumexp over enc dimension
            result = jax.nn.logsumexp(
                log_w[:, None] + log_position_distance + y_col[:, None], axis=0
            )
            return None, result

        # scan over decoding spikes' columns -> (n_dec, n_pos)
        _, log_num = jax.lax.scan(scan_over_dec, None, logK_mark.T)
    else:
        # Tiled: process positions in chunks to reduce peak memory
        # Memory: O(n_enc * pos_tile_size) instead of O(n_enc * n_pos)
        log_num = jnp.zeros((n_dec, n_pos))

        for pos_start in range(0, n_pos, pos_tile_size):
            pos_end = min(pos_start + pos_tile_size, n_pos)
            pos_slice = slice(pos_start, pos_end)

            # Tile: slice of log_position_distance for this chunk of positions
            log_pos_tile = log_position_distance[:, pos_slice]  # (n_enc, tile_size)

            def scan_over_dec_tile(
                log_pos_tile_arg: jnp.ndarray,
            ) -> tuple[None, jnp.ndarray]:
                def inner_scan(carry, y_col: jnp.ndarray) -> tuple[None, jnp.ndarray]:
                    # y_col: (n_enc,)
                    # returns: (tile_size,), logsumexp over enc dimension
                    result = jax.nn.logsumexp(
                        log_w[:, None] + log_pos_tile_arg + y_col[:, None], axis=0
                    )
                    return None, result

                return inner_scan

            # scan over decoding spikes for this position tile -> (n_dec, tile_size)
            _, log_num_tile = jax.lax.scan(
                scan_over_dec_tile(log_pos_tile), None, logK_mark.T
            )

            # Update output with this tile
            log_num = log_num.at[:, pos_slice].set(log_num_tile)

    # normalize by total weight sum (same as dividing by n_encoding_spikes in linear space)
    log_marginal = log_num - log_den  # (n_dec, n_pos)

    # 4) Add mean rate and subtract occupancy (in log)
    log_mean_rate = safe_log(mean_rate)
    log_occ = safe_log(occupancy)[None, :]  # (1, n_pos)

    log_joint = log_mean_rate + log_marginal - log_occ  # (n_dec, n_pos)
    return log_joint


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    encoding_weights: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    log_position_distance: jnp.ndarray,
    block_size: int = 100,
    use_gemm: bool = True,
    pos_tile_size: int | None = None,
) -> jnp.ndarray:
    """Estimate the log joint mark intensity in blocks over decoding spikes (log-space).

    Processes decoding spikes in blocks to manage memory usage when computing
    joint mark intensity. Calls estimate_log_joint_mark_intensity for each block.

    Parameters
    ----------
    decoding_spike_waveform_features : jnp.ndarray, shape (n_decoding_spikes, n_features)
        Waveform features for spikes during decoding period.
    encoding_spike_waveform_features : jnp.ndarray, shape (n_encoding_spikes, n_features)
        Waveform features for spikes during encoding period.
    encoding_weights : jnp.ndarray, shape (n_encoding_spikes,)
        Weights for each encoding spike.
    waveform_stds : jnp.ndarray, shape (n_features,)
        Standard deviations for Gaussian kernels over waveform dimensions.
    occupancy : jnp.ndarray, shape (n_position_bins,)
        Spatial occupancy density.
    mean_rate : float
        Mean firing rate for the electrode.
    log_position_distance : jnp.ndarray, shape (n_encoding_spikes, n_position_bins)
        Log of position-based kernel distances between encoding spikes and position bins.
    block_size : int, optional
        Number of decoding spikes to process per block, by default 100.
    use_gemm : bool, optional
        If True (default), use GEMM-based computation for mark kernel (faster for multi-dimensional features).
        If False, use per-dimension loop (slower but equivalent).
    pos_tile_size : int | None, optional
        If provided, tile computation over position dimension. Passed to estimate_log_joint_mark_intensity.

    Returns
    -------
    log_joint_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)
        Log joint mark intensity for all decoding spikes, clipped to LOG_EPS minimum.
    """

    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]
    if n_decoding_spikes == 0:
        return jnp.full((0, n_position_bins), LOG_EPS)

    # Use JIT-compiled update with buffer donation for memory efficiency
    # Donate the accumulator buffer (arg 0) so it can be reused in-place
    def _update_block(out_array, block_result, start_idx):
        return jax.lax.dynamic_update_slice(out_array, block_result, (start_idx, 0))

    update_block = jax.jit(_update_block, donate_argnums=(0,))

    out = jnp.zeros((n_decoding_spikes, n_position_bins))
    for start in range(0, n_decoding_spikes, block_size):
        sl = slice(start, start + block_size)
        block_result = estimate_log_joint_mark_intensity(
            decoding_spike_waveform_features[sl],
            encoding_spike_waveform_features,
            encoding_weights,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=use_gemm,
            pos_tile_size=pos_tile_size,
        )
        out = update_block(out, block_result, start)

    return jnp.clip(out, min=LOG_EPS, max=None)


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
    weights : jnp.ndarray, shape (n_time_position,), optional
        Sample weights for each position time point, by default None.
        If None, uniform weights are used.
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
        Dictionary containing fitted encoding model components:
        - 'occupancy': Occupancy density at interior place bins
        - 'occupancy_model': Fitted KDE model for occupancy
        - 'gpi_models': Ground process intensity KDE models per electrode
        - 'encoding_spike_waveform_features': Bounded spike features per electrode
        - 'encoding_positions': Position at spike times per electrode
        - 'encoding_spike_weights': Weights at spike times per electrode
        - 'environment': The spatial environment
        - 'mean_rates': Mean firing rates per electrode
        - 'summed_ground_process_intensity': Summed GPI across electrodes
        - 'position_std': Position kernel standard deviations
        - 'waveform_std': Waveform kernel standard deviations
        - 'block_size': Block size used for computation
        - 'disable_progress_bar': Progress bar setting
    """
    position = position if position.ndim > 1 else jnp.expand_dims(position, axis=1)
    if isinstance(position_std, (int, float)):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])
    if isinstance(waveform_std, (int, float)):
        waveform_std = jnp.array([waveform_std] * spike_waveform_features[0].shape[1])

    # Ensure std parameters are JAX arrays for KDEModel
    assert isinstance(position_std, jnp.ndarray)
    assert isinstance(waveform_std, jnp.ndarray)

    if environment.is_track_interior_ is not None:
        is_track_interior = environment.is_track_interior_.ravel()
    else:
        if environment.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ is required when is_track_interior_ is None"
            )
        is_track_interior = jnp.ones(
            environment.place_bin_centers_.shape[0], dtype=bool
        )
    interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

    if weights is None:
        weights = jnp.ones((position.shape[0],))
    total_weight = np.sum(weights)

    if environment.track_graph is not None and position.shape[1] > 1:
        # convert to 1D
        position1D = get_linearized_position(
            position,
            environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        ).linear_position.to_numpy()[:, None]
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position1D, weights=weights
        )
    else:
        occupancy_model = KDEModel(std=position_std, block_size=block_size).fit(
            position, weights=weights
        )

    occupancy = occupancy_model.predict(interior_place_bin_centers)
    encoding_positions = []
    encoding_spike_weights = []
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
        strict=False,
    ):
        is_in_bounds = jnp.logical_and(
            electrode_spike_times >= position_time[0],
            electrode_spike_times <= position_time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        bounded_spike_waveform_features.append(
            electrode_spike_waveform_features[is_in_bounds]
        )

        electrode_weights_at_spike_times = np.interp(
            electrode_spike_times, position_time, weights
        )
        encoding_spike_weights.append(electrode_weights_at_spike_times)
        mean_rates.append(np.sum(electrode_weights_at_spike_times) / total_weight)
        encoding_positions.append(
            get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
        )

        gpi_model = KDEModel(std=position_std, block_size=block_size).fit(
            encoding_positions[-1], weights=electrode_weights_at_spike_times
        )
        gpi_models.append(gpi_model)

        summed_ground_process_intensity += jnp.clip(
            mean_rates[-1]
            * safe_divide(gpi_model.predict(interior_place_bin_centers), occupancy),
            min=EPS,
            max=None,
        )

    return {
        "occupancy": occupancy,
        "occupancy_model": occupancy_model,
        "gpi_models": gpi_models,
        "encoding_spike_waveform_features": bounded_spike_waveform_features,
        "encoding_positions": encoding_positions,
        "encoding_spike_weights": encoding_spike_weights,
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
    encoding_positions: list[jnp.ndarray],
    encoding_spike_weights: list[jnp.ndarray],
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
    encoding_positions : list[jnp.ndarray], shape (n_encoding_spikes, n_position_dims)
        Position samples used for encoding.
    encoding_spike_weights : list[jnp.ndarray]
        Weights for encoding spikes per electrode.
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
            encoding_spike_weights,
            environment,
            mean_rates,
            position_std,
            waveform_std,
            block_size,
            disable_progress_bar,
        )
    else:
        if environment.is_track_interior_ is not None:
            is_track_interior = environment.is_track_interior_.ravel()
        else:
            if environment.place_bin_centers_ is None:
                raise ValueError(
                    "place_bin_centers_ is required when is_track_interior_ is None"
                )
            is_track_interior = jnp.ones(
                environment.place_bin_centers_.shape[0], dtype=bool
            )
        interior_place_bin_centers = environment.place_bin_centers_[is_track_interior]

        log_likelihood = -1.0 * summed_ground_process_intensity * jnp.ones((n_time, 1))

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
            encoding_spike_weights,
            mean_rates,
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
            log_position_distance = log_kde_distance(
                interior_place_bin_centers,
                electrode_encoding_positions,
                std=position_std,
            )

            log_likelihood += jax.ops.segment_sum(
                block_estimate_log_joint_mark_intensity(
                    electrode_decoding_spike_waveform_features,
                    electrode_encoding_spike_waveform_features,
                    electrode_encoding_weights,
                    waveform_std,
                    occupancy,
                    electrode_mean_rate,
                    log_position_distance,
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
    encoding_positions: list[jnp.ndarray],
    encoding_spike_weights: list[jnp.ndarray],
    environment: Environment,
    mean_rates: jnp.ndarray,
    position_std: jnp.ndarray,
    waveform_std: jnp.ndarray,
    weights: jnp.ndarray | None = None,
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
        encoding_spike_weights,
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

        eval_points = jnp.concatenate(
            (position_at_spike_time, electrode_decoding_spike_waveform_features), axis=1
        )
        samples = jnp.concatenate(
            (electrode_encoding_positions, electrode_encoding_spike_waveform_features),
            axis=1,
        )
        std = jnp.concatenate((position_std, waveform_std))

        log_marginal = block_log_kde(
            eval_points=eval_points,
            samples=samples,
            std=std,
            block_size=block_size,
            weights=electrode_encoding_weights,
        )
        occupancy_at_spike_time_log = occupancy_model.predict_log(
            position_at_spike_time
        )
        log_spike_term = (
            safe_log(electrode_mean_rate) + log_marginal - occupancy_at_spike_time_log
        )

        log_likelihood += jax.ops.segment_sum(
            log_spike_term,
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * safe_divide(
            electrode_gpi_model.predict(interpolated_position), occupancy
        )

    return log_likelihood[:, jnp.newaxis]
