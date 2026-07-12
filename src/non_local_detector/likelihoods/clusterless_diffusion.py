"""Graph-diffusion clusterless (marked-point-process) likelihood.

An opt-in clusterless likelihood, selected via
``clusterless_algorithm="clusterless_diffusion"`` on
:class:`~non_local_detector.models.base.ClusterlessDetector` and its subclasses. It
estimates the same mark intensity as :mod:`clusterless_kde` but replaces the Gaussian
*position* kernel with the environment's graph heat kernel ``exp(-t L)`` (``t =
position_std**2 / 2``). The *mark* kernel stays a Gaussian over waveform features
(via ``kde_distance``): ``position_std`` is the physical heat-kernel bandwidth (in
coordinate units) and ``waveform_std`` is the mark bandwidth, exactly as in
``clusterless_kde``. This is the clusterless analog of :mod:`sorted_spikes_diffusion`.

Two goals motivate it over KDE:

- **Goal A -- geometry-respecting spatial smoothing.** The heat kernel follows the
  environment's track topology (diffusion distance on the graph), not Euclidean
  distance, so it does not leak across walls, holes, or junctions the way a Gaussian
  KDE smoothed in coordinate space can.
- **Goal B -- speed.** Occupancy and each electrode's ground-process field are
  diffused once at fit; predict applies one cached low-rank heat-kernel matmul per
  decode-spike block instead of KDE's pairwise ``O(n_bins * n_enc * n_decode)``
  position kernel. The win grows with encoding/decode-spike count.

This is purely additive: the default clusterless likelihood remains
``clusterless_kde``.

For electrode ``e`` the joint mark density on interior bins is

    D_e[:, j]   = sum_i w_i * K_mark(m_j, m_i) * onehot(bin(x_i))
    p_e(x, m_j) = (H_t D_e)[x, j] / ( (sum_i w_i) * dV(x) )

where ``w_i`` are the per-encoding-spike EM weights, ``K_mark`` is the Gaussian
mark kernel (its normalizer already included by ``kde_distance``), ``H_t`` is the
mass-conserving heat kernel (:func:`heat_kernel_apply`), ``sum_i w_i`` the weighted
encoding count, and ``dV(x) = bin_sizes`` the per-bin measure. Dividing by
``(sum_i w_i) * dV(x)`` -- rather than normalizing each diffused column to unit
integral -- makes ``p_e`` a proper joint density whose spatial integral recovers
the weighted mark marginal.

The likelihood follows the shared marked-point-process contract::

    log L_e = sum_{j: decode spikes on e} log lambda_e(x, m_j) - integral lambda_bar_e(x) dt
    lambda_e(x, m)  = mean_rate_e * p_e(x, m) / occupancy(x)
    lambda_bar_e(x) = mean_rate_e * p_gpi_e(x) / occupancy(x)

Occupancy and the mark-marginalized ground-process field are diffused once at fit;
the only per-decode-spike work is building ``D_e`` (weighted mark kernel + scatter)
and one low-rank diffusion. Prob-space throughout, with ``safe_log`` flooring to
``LOG_EPS``.
"""

import logging
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_kde import kde_distance
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_EPS,
    as_std_array,
    get_position_at_time,
    get_spike_time_bin_ind,
    interpolate_weights_at_spike_times,
    safe_log,
    validate_finite,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    cached_heat_kernel_eigenbasis,
    environment_graph,
    get_device_basis,
    heat_kernel_apply,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    _full_to_local,
    _interior_bin_indices,
)

logger = logging.getLogger(__name__)

# float32 working dtype for the diffusion matmul; its item size drives the memory
# policy (spec sec 3) -- do not hardcode 4 elsewhere.
_WORKING_ITEMSIZE = np.dtype(np.float32).itemsize


def _validate_diffusion_params(
    position_std: float,
    waveform_std,
    heat_kernel_rank: int | None,
    memory_budget: int,
    block_size: int,
) -> float:
    """Tier-1 validation for the diffusion-specific parameters (spec sec 4).

    Returns ``position_std`` coerced to a plain float (it becomes part of the
    bandwidth-keyed eigenbasis cache key, so a JAX scalar/array would be
    unhashable; the graph heat kernel is isotropic in graph distance, so a
    per-dimension bandwidth is not supported).
    """
    position_std_arr = np.asarray(position_std, dtype=float)
    if position_std_arr.size != 1:
        raise ValidationError(
            "position_std must be a scalar; the graph heat kernel is isotropic in "
            "graph distance, so per-dimension bandwidths are not supported.",
            expected="a scalar position_std",
            got=f"array of shape {position_std_arr.shape}",
        )
    position_std = float(position_std_arr.item())
    if not (position_std > 0.0 and np.isfinite(position_std)):
        raise ValidationError(
            "position_std must be a finite positive number",
            expected="0 < position_std < inf",
            got=f"position_std = {position_std}",
        )
    waveform_std_arr = np.asarray(waveform_std, dtype=float)
    if not (np.all(waveform_std_arr > 0.0) and np.all(np.isfinite(waveform_std_arr))):
        raise ValidationError(
            "waveform_std must be finite and positive",
            expected="0 < waveform_std < inf",
            got=f"waveform_std = {waveform_std}",
        )

    if heat_kernel_rank is not None:
        # Reject float/bool/zero/negative before it reaches SciPy / cache slicing.
        if isinstance(heat_kernel_rank, bool) or not isinstance(heat_kernel_rank, int):
            raise ValidationError(
                "heat_kernel_rank must be None or a positive integer",
                expected="None or a positive int",
                got=f"{heat_kernel_rank!r} (type {type(heat_kernel_rank).__name__})",
            )
        if heat_kernel_rank < 1:
            raise ValidationError(
                "heat_kernel_rank must be a positive integer",
                expected="heat_kernel_rank >= 1",
                got=f"heat_kernel_rank = {heat_kernel_rank}",
            )

    if isinstance(memory_budget, bool) or not isinstance(memory_budget, int):
        raise ValidationError(
            "memory_budget must be a positive integer number of bytes",
            expected="a positive int",
            got=f"{memory_budget!r} (type {type(memory_budget).__name__})",
        )
    if memory_budget < 1:
        raise ValidationError(
            "memory_budget must be a positive integer number of bytes",
            expected="memory_budget >= 1",
            got=f"memory_budget = {memory_budget}",
        )

    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValidationError(
            "block_size must be a positive integer",
            expected="a positive int >= 1",
            got=f"{block_size!r}",
        )

    return position_std


def _effective_block(
    memory_budget: int,
    n_enc: int,
    rank: int,
    n_bins: int,
    block_size: int,
) -> int:
    """Decode-spike block size that fits the per-block live set in ``memory_budget``.

    Per decode-spike column the live set is dominated by ``K`` (``n_enc``), the
    spectral projection (``rank``), and ~4 ``n_bins``-scaled arrays (``D``, ``P``,
    ``lc`` and the clip/rescale temporaries), all in the float32 working dtype. A
    safety factor of 2 leaves headroom for XLA workspace (spec sec 3).
    """
    bytes_per_col = _WORKING_ITEMSIZE * (n_enc + rank + 4 * n_bins)
    raw = memory_budget // (bytes_per_col * 2)
    return int(np.clip(raw, 1, block_size))


def fit_clusterless_diffusion_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    *,
    sampling_frequency: int = 500,
    position_std: float = float(np.sqrt(12.5)),
    waveform_std: float = 24.0,
    weights: np.ndarray | None = None,
    heat_kernel_rank: int | None = None,
    block_size: int = 10_000,
    memory_budget: int = 536_870_912,
    disable_progress_bar: bool = False,
    **kwargs: object,
) -> dict:
    """Fit the clusterless graph-diffusion encoding model.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Time of each position sample.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[jnp.ndarray]
        Spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Spike waveform features (marks) for each electrode.
    environment : Environment
        The spatial environment (must be fitted).
    sampling_frequency : int, optional
        Samples per second, by default 500. Accepted for signature compatibility;
        not used by the diffusion smoother.
    position_std : float, optional
        Heat-kernel smoothing standard deviation (the physical bandwidth) in
        coordinate units, by default ``sqrt(12.5)``. Scalar only: the heat kernel is
        isotropic in graph distance.
    waveform_std : float, optional
        Gaussian smoothing standard deviation for the mark kernel, by default 24.0.
    weights : np.ndarray, shape (n_time_position,), optional
        Per-sample EM weights, by default None (uniform). Weight the occupancy,
        per-electrode ground-process, mean rate, and the per-spike joint estimator.
    heat_kernel_rank : int or None, optional
        Eigenbasis truncation rank, by default None (bandwidth-aware auto-selection).
    block_size : int, optional
        Requested cap on the decode-spike block size, by default 10_000. The
        effective block is the memory-budget-resolved value (spec sec 3).
    memory_budget : int, optional
        Per-block byte budget driving the effective block size, by default
        536_870_912 (512 MiB).
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.

    Returns
    -------
    encoding_model : dict
        Keys consumed by :func:`predict_clusterless_diffusion_log_likelihood`:
        ``environment``, ``occupancy`` (pi, ``(n_interior,)``),
        ``summed_ground_process_intensity`` (``(n_interior,)``),
        ``encoding_bin_indices`` (list per electrode), ``encoding_marks`` (list),
        ``encoding_weights`` (list), ``weight_total`` (list of float),
        ``mean_rates`` (list), ``resolved_rank`` (int), ``node_order``,
        ``bin_sizes`` (dV per interior bin), ``position_std``, ``waveform_std``,
        ``block_size``, ``memory_budget``, ``disable_progress_bar``.
    """
    if environment.place_bin_centers_ is None:
        raise ValueError(
            "Environment must be fitted with place_bin_centers_. "
            "Call environment.fit_place_grid() first."
        )

    position = position if np.ndim(position) > 1 else np.asarray(position)[:, None]
    validate_finite(position, "position")
    n_time_position = position.shape[0]
    if weights is None:
        weights = np.ones((n_time_position,))
    weights = validate_weights(weights, n_time_position)
    position_std = _validate_diffusion_params(
        position_std, waveform_std, heat_kernel_rank, memory_budget, block_size
    )

    # dV = per-interior-bin measure (uniform grid -> constant; linearized track ->
    # per-bin width), aligned with node_order.
    _, node_order, bin_sizes = environment_graph(environment)
    bin_sizes = np.asarray(bin_sizes)
    n_interior = node_order.shape[0]
    n_total_bins = environment.is_track_interior_.ravel().shape[0]
    full_to_local = _full_to_local(node_order, n_total_bins)

    # Resolve the eigenbasis ONCE and record its rank; predict retrieves the device
    # basis via get_device_basis(environment, resolved_rank).
    if heat_kernel_rank is None:
        _, eigvecs = cached_heat_kernel_eigenbasis(environment, position_std)
    else:
        _, eigvecs = cached_eigenbasis(environment, heat_kernel_rank)
    resolved_rank = int(eigvecs.shape[1])

    Lam, Q, labels, n_components = get_device_basis(environment, resolved_rank)

    weight_sum = float(weights.sum())  # weighted occupancy time (sum w_pos)

    # ---- occupancy pi = H O / (sum w_pos * dV), floored at EPS ----
    occupancy_positions = get_position_at_time(
        position_time, position, position_time, environment
    )
    occupancy_field = np.bincount(
        _interior_bin_indices(environment, occupancy_positions, full_to_local),
        weights=weights,
        minlength=n_interior,
    )
    occupancy_hat = np.asarray(
        heat_kernel_apply(
            Lam,
            Q,
            position_std,
            jnp.asarray(occupancy_field[:, None], dtype=jnp.float32),
            labels,
            n_components=n_components,
        )
    )[:, 0]
    safe_weight_sum = weight_sum if weight_sum > 0 else 1.0
    occupancy = np.clip(occupancy_hat / (safe_weight_sum * bin_sizes), EPS, None)
    if weight_sum == 0:
        warnings.warn(
            "occupancy weights sum to 0; the diffusion occupancy is degenerate and a "
            "safe denominator is used. Every electrode will be zero-rate.",
            UserWarning,
            stacklevel=2,
        )

    # ---- per electrode: marks, bin indices, weights, mean rate, ground process ----
    encoding_bin_indices: list[np.ndarray] = []
    encoding_marks: list[jnp.ndarray] = []
    encoding_weights: list[jnp.ndarray] = []
    weight_total: list[float] = []
    mean_rates: list[float] = []
    summed_ground_process_intensity = np.zeros((n_interior,))

    for electrode, (electrode_spike_times, electrode_features) in enumerate(
        zip(
            tqdm(
                spike_times,
                desc="Encoding models",
                unit="electrode",
                disable=disable_progress_bar,
            ),
            spike_waveform_features,
            strict=True,
        )
    ):
        electrode_spike_times = np.asarray(electrode_spike_times)
        is_in_bounds = np.logical_and(
            electrode_spike_times >= position_time[0],
            electrode_spike_times <= position_time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        # Validate only the in-window features that actually enter the fit.
        bounded_features = np.asarray(electrode_features)[is_in_bounds]
        validate_finite(bounded_features, "spike_waveform_features")

        spike_weights = interpolate_weights_at_spike_times(
            electrode_spike_times, position_time, weights
        )
        w_total = float(spike_weights.sum())

        spike_positions = get_position_at_time(
            position_time, position, electrode_spike_times, environment
        )
        bins = _interior_bin_indices(environment, spike_positions, full_to_local)

        encoding_bin_indices.append(np.asarray(bins))
        encoding_marks.append(jnp.asarray(bounded_features))
        encoding_weights.append(jnp.asarray(spike_weights))
        weight_total.append(w_total)

        if w_total == 0:
            # Zero-rate electrode: no fitted density, no ground-process contribution.
            mean_rates.append(0.0)
            warnings.warn(
                f"electrode {electrode} has zero total encoding weight (zero-rate): "
                "no effective encoding spikes. It contributes no ground-process "
                "intensity and its decode spikes floor to LOG_EPS.",
                UserWarning,
                stacklevel=2,
            )
            continue

        mean_rate = weighted_mean_rate(spike_weights, weight_sum)
        mean_rates.append(mean_rate)
        spike_field = np.bincount(bins, weights=spike_weights, minlength=n_interior)
        spike_field_hat = np.asarray(
            heat_kernel_apply(
                Lam,
                Q,
                position_std,
                jnp.asarray(spike_field[:, None], dtype=jnp.float32),
                labels,
                n_components=n_components,
            )
        )[:, 0]
        p_gpi = spike_field_hat / (w_total * bin_sizes)
        summed_ground_process_intensity += mean_rate * p_gpi / occupancy

    summed_ground_process_intensity = np.clip(
        summed_ground_process_intensity, EPS, None
    )

    return {
        "environment": environment,
        "occupancy": occupancy,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "encoding_bin_indices": encoding_bin_indices,
        "encoding_marks": encoding_marks,
        "encoding_weights": encoding_weights,
        "weight_total": weight_total,
        "mean_rates": mean_rates,
        "resolved_rank": resolved_rank,
        "node_order": node_order,
        "bin_sizes": bin_sizes,
        "position_std": position_std,
        "waveform_std": waveform_std,
        "block_size": block_size,
        "memory_budget": memory_budget,
        "disable_progress_bar": disable_progress_bar,
    }


def predict_clusterless_diffusion_log_likelihood(
    time: np.ndarray,
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    *,
    is_local: bool = False,
    **encoding_model: object,
) -> jnp.ndarray:
    """Predict the clusterless graph-diffusion log likelihood.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Decoding time bins.
    position_time : np.ndarray, shape (n_time_position,)
        Time of each position sample (used only by the local path; accepted for
        signature parity when ``is_local`` is False).
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Position samples (used only by the local path; accepted for signature
        parity when ``is_local`` is False).
    spike_times : list[jnp.ndarray]
        Decode spike times for each electrode.
    spike_waveform_features : list[jnp.ndarray]
        Decode spike waveform features (marks) for each electrode.
    is_local : bool, optional
        If True, evaluate the likelihood at the animal's position (nearest interior
        bin; the ``D_e`` construction and ``heat_kernel_apply`` call are identical to
        the non-local path -- only the readout differs). By default False.
    **encoding_model
        The dict returned by :func:`fit_clusterless_diffusion_encoding_model`.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_time, n_interior_bins) if ``is_local`` is
        False, else (n_time, 1).
    """
    environment: Environment = encoding_model["environment"]  # type: ignore[assignment]
    occupancy = jnp.asarray(encoding_model["occupancy"])
    summed_ground_process_intensity = jnp.asarray(
        encoding_model["summed_ground_process_intensity"]
    )
    encoding_bin_indices = encoding_model["encoding_bin_indices"]
    encoding_marks = encoding_model["encoding_marks"]
    encoding_weights = encoding_model["encoding_weights"]
    weight_total = encoding_model["weight_total"]
    mean_rates = encoding_model["mean_rates"]
    resolved_rank = int(encoding_model["resolved_rank"])  # type: ignore[call-overload]
    node_order = np.asarray(encoding_model["node_order"])
    bin_sizes = jnp.asarray(encoding_model["bin_sizes"])
    position_std = float(encoding_model["position_std"])  # type: ignore[arg-type]
    waveform_std = encoding_model["waveform_std"]
    block_size = int(encoding_model["block_size"])  # type: ignore[call-overload]
    memory_budget = int(encoding_model["memory_budget"])  # type: ignore[call-overload]
    disable_progress_bar = bool(encoding_model.get("disable_progress_bar", False))

    time = np.asarray(time)
    validate_finite(time, "time")
    n_time = len(time)
    n_bins = occupancy.shape[0]

    Lam, Q, labels, n_components = get_device_basis(environment, resolved_rank)

    if is_local:
        # The local path reads the animal's position; non-local ignores it and the
        # base API permits position=None (as clusterless_kde does), so position is
        # only required and validated here (Tier 1, spec sec 4).
        validate_finite(position, "position")
        # Reconstruct the interior-bin mapping from node_order (not stored, per
        # spec sec 3) and evaluate the diffused column at the animal's nearest
        # interior bin instead of returning the whole column.
        n_total_bins = environment.is_track_interior_.ravel().shape[0]
        full_to_local = _full_to_local(node_order, n_total_bins)

        interpolated_position = get_position_at_time(
            position_time, position, time, environment
        )
        animal_time_bins = _interior_bin_indices(
            environment, interpolated_position, full_to_local
        )
        # Ground-process term evaluated at the animal's position per decode time
        # bin (mirrors clusterless_kde's local summed_expected_counts).
        local_ground_process_intensity = summed_ground_process_intensity[
            animal_time_bins
        ]  # (n_time,)
        log_likelihood = -local_ground_process_intensity

        for (
            electrode_bins,
            electrode_marks,
            electrode_weights,
            electrode_weight_total,
            electrode_mean_rate,
            electrode_decode_features,
            electrode_spike_times,
        ) in zip(
            tqdm(
                encoding_bin_indices,
                unit="electrode",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_marks,
            encoding_weights,
            weight_total,
            mean_rates,
            spike_waveform_features,
            spike_times,
            strict=True,
        ):
            electrode_spike_times = np.asarray(electrode_spike_times)
            is_in_bounds = np.logical_and(
                electrode_spike_times >= time[0],
                electrode_spike_times <= time[-1],
            )
            electrode_spike_times = electrode_spike_times[is_in_bounds]
            # Validate only the in-window decode features that actually enter the
            # likelihood (mirrors fit's post-clip validation); an out-of-window
            # spike's feature must not raise.
            decode_features = jnp.asarray(electrode_decode_features)[
                jnp.asarray(is_in_bounds)
            ]
            validate_finite(decode_features, "spike_waveform_features")
            n_decode = electrode_spike_times.shape[0]
            if n_decode == 0:
                continue
            seg = jnp.asarray(get_spike_time_bin_ind(electrode_spike_times, time))

            # Zero-rate electrode: floor every observed decode spike to LOG_EPS,
            # identical contract to the non-local zero-rate guard.
            if electrode_weight_total == 0:
                log_likelihood += jax.ops.segment_sum(
                    jnp.full((n_decode,), LOG_EPS),
                    seg,
                    indices_are_sorted=True,
                    num_segments=n_time,
                )
                continue

            enc_bins = jnp.asarray(electrode_bins)
            enc_marks = jnp.asarray(electrode_marks)
            enc_weights = jnp.asarray(electrode_weights)
            n_enc = enc_marks.shape[0]
            n_features = enc_marks.shape[1]
            electrode_waveform_std = as_std_array(waveform_std, n_features)
            safe_weight_total = (
                electrode_weight_total if electrode_weight_total > 0 else 1.0
            )

            # The animal's interior bin at each decode spike's own time (nearest-bin
            # interpolation; linear is a documented follow-up, not this task).
            position_at_spike_time = get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
            spike_bins = jnp.asarray(
                _interior_bin_indices(
                    environment, position_at_spike_time, full_to_local
                )
            )

            effective_block = _effective_block(
                memory_budget, n_enc, resolved_rank, n_bins, block_size
            )

            for start in range(0, n_decode, effective_block):
                block = slice(start, start + effective_block)
                decode_block = decode_features[block]
                seg_block = seg[block]
                spike_bins_block = spike_bins[block]
                # Same D_e / heat_kernel_apply as non-local -- the clip+mass-rescale
                # is a per-column nonlinearity, so the local value must come from
                # the SAME diffused column, just indexed at the animal's bin.
                mark_kernel = kde_distance(
                    decode_block, enc_marks, electrode_waveform_std
                )
                weighted_kernel = enc_weights[:, None] * mark_kernel
                D = (
                    jnp.zeros((n_bins, decode_block.shape[0]))
                    .at[enc_bins]
                    .add(weighted_kernel)
                )
                P = heat_kernel_apply(
                    Lam, Q, position_std, D, labels, n_components=n_components
                )
                column_ind = jnp.arange(decode_block.shape[0])
                p_e_at_animal = P[spike_bins_block, column_ind] / (
                    safe_weight_total * bin_sizes[spike_bins_block]
                )
                lc = safe_log(
                    electrode_mean_rate * p_e_at_animal / occupancy[spike_bins_block]
                )  # (n_block,)
                log_likelihood += jax.ops.segment_sum(
                    lc,
                    seg_block,
                    indices_are_sorted=True,
                    num_segments=n_time,
                )

        return log_likelihood[:, None]

    occupancy_col = occupancy[:, None]

    log_likelihood = -summed_ground_process_intensity[None, :] * jnp.ones((n_time, 1))

    for (
        electrode_bins,
        electrode_marks,
        electrode_weights,
        electrode_weight_total,
        electrode_mean_rate,
        electrode_decode_features,
        electrode_spike_times,
    ) in zip(
        tqdm(
            encoding_bin_indices,
            unit="electrode",
            desc="Non-Local Likelihood",
            disable=disable_progress_bar,
        ),
        encoding_marks,
        encoding_weights,
        weight_total,
        mean_rates,
        spike_waveform_features,
        spike_times,
        strict=True,
    ):
        electrode_spike_times = np.asarray(electrode_spike_times)
        is_in_bounds = np.logical_and(
            electrode_spike_times >= time[0],
            electrode_spike_times <= time[-1],
        )
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        # Validate only the in-window decode features that actually enter the
        # likelihood (mirrors fit's post-clip validation); an out-of-window spike's
        # feature must not raise.
        decode_features = jnp.asarray(electrode_decode_features)[
            jnp.asarray(is_in_bounds)
        ]
        validate_finite(decode_features, "spike_waveform_features")
        n_decode = electrode_spike_times.shape[0]
        if n_decode == 0:
            continue
        seg = jnp.asarray(get_spike_time_bin_ind(electrode_spike_times, time))

        # Zero-rate electrode: floor every observed decode spike to LOG_EPS BEFORE any
        # division (D_e == 0 and P_e / weight_total_e is 0/0). Not skipped. Each
        # time bin's contribution is (spikes in that bin) * LOG_EPS, identical across
        # bins -- accumulate it from per-time-bin counts rather than materializing an
        # (n_decode, n_bins) array (which would be gigabytes at millions of spikes).
        if electrode_weight_total == 0:
            spike_counts = jax.ops.segment_sum(
                jnp.ones(n_decode),
                seg,
                indices_are_sorted=True,
                num_segments=n_time,
            )  # (n_time,)
            log_likelihood += spike_counts[:, None] * LOG_EPS
            continue

        enc_bins = jnp.asarray(electrode_bins)
        enc_marks = jnp.asarray(electrode_marks)
        enc_weights = jnp.asarray(electrode_weights)
        n_enc = enc_marks.shape[0]
        n_features = enc_marks.shape[1]
        electrode_waveform_std = as_std_array(waveform_std, n_features)
        # weight_total_e > 0 here; belt-and-suspenders safe denominator.
        safe_weight_total = (
            electrode_weight_total if electrode_weight_total > 0 else 1.0
        )

        effective_block = _effective_block(
            memory_budget, n_enc, resolved_rank, n_bins, block_size
        )
        logger.debug(
            "clusterless_diffusion predict: effective_block=%d "
            "(requested block_size=%d, memory_budget=%d bytes, n_enc=%d, rank=%d, "
            "n_bins=%d)",
            effective_block,
            block_size,
            memory_budget,
            n_enc,
            resolved_rank,
            n_bins,
        )

        for start in range(0, n_decode, effective_block):
            block = slice(start, start + effective_block)
            decode_block = decode_features[block]
            seg_block = seg[block]
            # (n_enc, n_block) mark kernel; its Gaussian normalizer is already in kde_distance.
            mark_kernel = kde_distance(decode_block, enc_marks, electrode_waveform_std)
            # D_e[:, j] = sum_i w_i K(m_j, m_i) onehot(bin(x_i))  -> (n_bins, n_block)
            weighted_kernel = enc_weights[:, None] * mark_kernel
            D = (
                jnp.zeros((n_bins, decode_block.shape[0]))
                .at[enc_bins]
                .add(weighted_kernel)
            )
            P = heat_kernel_apply(
                Lam, Q, position_std, D, labels, n_components=n_components
            )
            # bin_sizes (dV) makes p_e a proper joint density (recovers the mark
            # marginal, guarded by test_mark_marginal_recovery); it cancels in the
            # p_e / occupancy ratio below, so it does not change the likelihood.
            p_e = P / (safe_weight_total * bin_sizes[:, None])
            log_intensity = safe_log(
                electrode_mean_rate * p_e / occupancy_col
            )  # (n_bins, n_block)
            log_likelihood += jax.ops.segment_sum(
                log_intensity.T,
                seg_block,
                indices_are_sorted=True,
                num_segments=n_time,
            )

    return log_likelihood
