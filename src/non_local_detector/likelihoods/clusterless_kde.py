import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm
from track_linearization import get_linearized_position

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    KDEModel,
    block_kde,
    get_position_at_time,
    get_spike_time_bin_ind,
    log_gaussian_pdf,
    safe_log,
)


def _estimate_predict_peak_bytes(
    *,
    n_time: int,
    n_state_bins: int,
    n_pos: int,
    n_encoding_spikes_max: int,
    n_decoding_spikes_max: int,
    n_waveform_features: int = 0,  # noqa: ARG001 — reserved for future tiling knobs
    n_chunks: int = 1,
    block_size: int = 100,
    enc_tile_size: int | None = None,
    pos_tile_size: int | None = None,
    dtype_bytes: int = 4,
) -> int:
    """Estimate peak GPU bytes for a clusterless-KDE predict call.

    Models the dominant tensors live simultaneously at peak memory in
    the serial-per-electrode loop inside
    :func:`predict_clusterless_kde_log_likelihood`:

    - ``likelihood_slab``: ``(chunk_size, n_state_bins)`` — per-chunk
      accumulator summed across electrodes.
    - ``position_distance``: ``(enc_dim, pos_dim)`` — per-electrode,
      live during one electrode's processing.
    - ``per_electrode_output``: ``(n_dec_chunk_max, pos_dim)`` — the
      log joint mark intensity for one electrode.
    - ``mark_kernel_tmp``: ``(enc_dim, block_size)`` — per-block
      decoding-spike vs. all-encoding kernel.
    - ``block_output_tmp``: ``(block_size, pos_dim)`` — per-block
      output fragment.
    - ``transition``: ``(n_state_bins, n_state_bins)`` — persistent.
    - ``fixed_scratch``: 2 GB for XLA autotuning workspace + transient
      scratch (empirical; refined in Task 6 of the streaming plan).

    Then applies a multiplicative safety factor of 2.0 to absorb model
    imprecision around JAX/XLA intermediate allocations.  Observed
    peak at 22-tet/2D HPC was ~8× the bare likelihood slab (33 GB vs
    4.1 GB); this model with the 2× multiplier captures the
    non-trivial tensors explicitly and uses the multiplier for
    untracked transients.

    Parameters
    ----------
    n_time
        Total decoding time bins (pre-chunking).
    n_state_bins
        Interior state bin count — typically
        ``detector.is_track_interior_state_bins_.sum()``.
    n_pos
        Interior position bin count.
    n_encoding_spikes_max
        Max encoding-spike count over electrodes.
    n_decoding_spikes_max
        Max decoding-spike count over electrodes across the full time
        window.
    n_waveform_features
        Waveform feature dim.  Reserved for future tiling knobs; not
        used by the current model.
    n_chunks
        Streaming chunk count (Task 1 knob).
    block_size
        Decoding-spike block-loop size.
    enc_tile_size
        Encoding-spike tile size.  None disables encoding tiling.
    pos_tile_size
        Position-bin tile size.  None disables position tiling.
    dtype_bytes
        Bytes per element.  4 for fp32 (default), 8 for fp64.

    Returns
    -------
    int
        Estimated peak bytes.
    """
    chunk_size = (n_time + n_chunks - 1) // n_chunks
    n_dec_per_chunk_max = (n_decoding_spikes_max + n_chunks - 1) // n_chunks
    enc_dim = (
        enc_tile_size if enc_tile_size is not None else n_encoding_spikes_max
    )
    pos_dim = pos_tile_size if pos_tile_size is not None else n_pos

    likelihood_slab = chunk_size * n_state_bins * dtype_bytes
    position_distance = enc_dim * pos_dim * dtype_bytes
    per_electrode_output = n_dec_per_chunk_max * pos_dim * dtype_bytes
    mark_kernel_tmp = enc_dim * block_size * dtype_bytes
    block_output_tmp = block_size * pos_dim * dtype_bytes
    transition = n_state_bins * n_state_bins * dtype_bytes
    fixed_scratch = 2 * 2**30  # 2 GB empirical XLA scratch

    per_electrode_live = (
        position_distance
        + per_electrode_output
        + mark_kernel_tmp
        + block_output_tmp
    )
    persistent = likelihood_slab + transition + fixed_scratch

    # Safety multiplier calibrated against observed peak memory.
    #
    # Empirical from Task 8 real-data validation (22-tet 2D HPC ContFrag):
    # - Per-tensor model predicts ~7.5 GB at n_chunks=1, block_size=10000.
    # - Observed JAX ``peak_bytes_in_use`` at the same config: ~33 GB.
    # - Ratio ~4.4× — the unmodeled overhead is primarily JAX async dispatch
    #   keeping all 22 electrodes' ``position_distance`` +
    #   ``per_electrode_output`` intermediates alive until the filter
    #   consumes the chunk's likelihood.  At ``n_electrodes`` up to ~64 in
    #   production this ratio can climb further, so we round up to 5×.
    #
    # Note: sorted-spikes peak/slab ratio is much smaller (~2×) because
    # there's no per-electrode mark kernel; see sorted_spikes_kde /
    # sorted_spikes_glm estimators for the algorithm-appropriate value.
    _SAFETY_MULTIPLIER = 5.0
    return int(_SAFETY_MULTIPLIER * (per_electrode_live + persistent))


def _estimate_fit_peak_bytes(
    *,
    n_time_pos: int,
    n_pos: int,
    n_encoding_spikes_max: int,
    n_electrodes: int = 1,  # noqa: ARG001 — electrodes iterate serially
    n_waveform_features: int = 0,  # noqa: ARG001
    fit_block_size: int = 10_000,
    dtype_bytes: int = 4,
) -> int:
    """Estimate peak GPU bytes for ``fit_clusterless_kde_encoding_model``.

    Models the dominant allocations during fit:

    - ``occupancy_fit_peak``: ``(fit_block_size, n_pos)`` — per-block
      occupancy KDE eval.
    - ``per_electrode_kde``: ``(n_enc_max, n_pos)`` — one electrode's
      encoding-positions KDE eval (serial per-electrode).
    - ``occupancy_output``: ``(n_pos,)`` — final occupancy density.
    - ``fixed_scratch``: 0.5 GB (fit is less scratch-heavy than predict).

    Multiplicative safety factor of 2.0.  ``occupancy_fit_peak`` and
    ``per_electrode_kde`` don't live simultaneously (occupancy finishes
    before per-electrode starts), so we take the ``max`` of the two.

    Returns
    -------
    int
        Estimated peak bytes during fit.
    """
    occupancy_fit_peak = fit_block_size * n_pos * dtype_bytes
    per_electrode_kde = n_encoding_spikes_max * n_pos * dtype_bytes
    occupancy_output = n_pos * dtype_bytes
    fixed_scratch = 512 * 2**20  # 0.5 GB

    _SAFETY_MULTIPLIER = 2.0
    return int(
        _SAFETY_MULTIPLIER
        * (
            max(occupancy_fit_peak, per_electrode_kde)
            + occupancy_output
            + fixed_scratch
        )
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
    log_distance = jnp.zeros((samples.shape[0], eval_points.shape[0]))
    for dim_eval_points, dim_samples, dim_std in zip(
        eval_points.T, samples.T, std, strict=True
    ):
        log_distance += log_gaussian_pdf(
            jnp.expand_dims(dim_eval_points, axis=0),
            jnp.expand_dims(dim_samples, axis=1),
            dim_std,
        )
    return jnp.exp(log_distance)


def estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
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
    # Double-where: substitute safe denominator, then select result
    safe_n = jnp.where(n_encoding_spikes > 0, n_encoding_spikes, 1)
    marginal_density = jnp.where(
        n_encoding_spikes > 0,
        spike_waveform_feature_distance.T @ position_distance / safe_n,
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


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features: jnp.ndarray,
    encoding_spike_waveform_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
    occupancy: jnp.ndarray,
    mean_rate: float,
    position_distance: jnp.ndarray,
    block_size: int = 100,
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

    Returns
    -------
    log_joint_mark_intensity : jnp.ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    log_joint_mark_intensity = jnp.zeros((n_decoding_spikes, n_position_bins))

    for start_ind in range(0, n_decoding_spikes, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        log_joint_mark_intensity = jax.lax.dynamic_update_slice(
            log_joint_mark_intensity,
            estimate_log_joint_mark_intensity(
                decoding_spike_waveform_features[block_inds],
                encoding_spike_waveform_features,
                waveform_stds,
                occupancy,
                mean_rate,
                position_distance,
            ),
            (start_ind, 0),
        )

    return jnp.clip(log_joint_mark_intensity, min=LOG_EPS, max=None)


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
    if isinstance(position_std, int | float):
        if environment.track_graph is not None and position.shape[1] > 1:
            position_std = jnp.array([position_std])
        else:
            position_std = jnp.array([position_std] * position.shape[1])
    # Keep waveform_std as-is (scalar or array) - will be expanded per-electrode at predict time

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

    # Count actual training samples (not the wall-clock span) so that gaps
    # introduced by is_training / encoding-group masks are not charged as
    # occupancy time.
    n_time_bins = max(len(position_time), 1)
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
            mean_rates[-1]
            * jnp.where(
                occupancy > 0.0,
                gpi_density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                EPS,
            ),
            min=EPS,
            max=None,
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
            # Expand waveform_std to match this electrode's feature count if scalar
            n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]
            if isinstance(waveform_std, (int, float)) or (
                hasattr(waveform_std, "ndim") and waveform_std.ndim == 0
            ):
                electrode_waveform_std = jnp.full(n_waveform_features, waveform_std)
            else:
                electrode_waveform_std = waveform_std
            log_likelihood += jax.ops.segment_sum(
                block_estimate_log_joint_mark_intensity(
                    electrode_decoding_spike_waveform_features,
                    electrode_encoding_spike_waveform_features,
                    electrode_waveform_std,
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

        position_at_spike_time = get_position_at_time(
            position_time, position, electrode_spike_times, environment
        )

        # Expand waveform_std to match this electrode's feature count if scalar
        n_waveform_features = electrode_encoding_spike_waveform_features.shape[1]
        if isinstance(waveform_std, (int, float)) or (
            hasattr(waveform_std, "ndim") and waveform_std.ndim == 0
        ):
            electrode_waveform_std = jnp.full(n_waveform_features, waveform_std)
        else:
            electrode_waveform_std = waveform_std

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
            std=jnp.concatenate((position_std, electrode_waveform_std)),
            block_size=block_size,
        )
        occupancy_at_spike_time = occupancy_model.predict(position_at_spike_time)

        log_likelihood += jax.ops.segment_sum(
            safe_log(
                electrode_mean_rate
                * jnp.where(
                    occupancy_at_spike_time > 0.0,
                    marginal_density
                    / jnp.where(
                        occupancy_at_spike_time > 0.0, occupancy_at_spike_time, 1.0
                    ),
                    0.0,
                )
            ),
            get_spike_time_bin_ind(electrode_spike_times, time),
            indices_are_sorted=True,
            num_segments=n_time,
        )

        log_likelihood -= electrode_mean_rate * jnp.where(
            occupancy > 0.0,
            electrode_gpi_model.predict(interpolated_position)
            / jnp.where(occupancy > 0.0, occupancy, 1.0),
            0.0,
        )
    return log_likelihood[:, jnp.newaxis]
