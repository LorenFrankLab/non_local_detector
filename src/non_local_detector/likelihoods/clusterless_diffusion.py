"""Graph-diffusion clusterless (marked-point-process) likelihood.

An opt-in clusterless likelihood, selected via
``clusterless_algorithm="clusterless_diffusion"`` on
:class:`~non_local_detector.models.base.ClusterlessDetector` and its subclasses. It
estimates the same mark intensity as :mod:`clusterless_kde` but replaces the Gaussian
*position* kernel with the environment's graph heat kernel ``exp(-t L)`` (``t =
position_std**2 / 2``). The *mark* kernel stays a Gaussian over waveform features
(computed in log space): ``position_std`` is the physical heat-kernel bandwidth (in
coordinate units) and ``waveform_std`` is the mark bandwidth, exactly as in
``clusterless_kde``. This is the clusterless analog of :mod:`sorted_spikes_diffusion`.

Two properties motivate it over KDE:

- **Geometry-respecting spatial smoothing.** The heat kernel follows the
  environment's track topology (diffusion distance on the graph), not Euclidean
  distance, so it does not leak across walls, holes, or junctions the way a Gaussian
  KDE smoothed in coordinate space can.
- **Speed.** Occupancy and each electrode's ground-process field are diffused once
  at fit; predict combines each block's marks per source bin and applies the
  spatial kernel's columns in bounded tiles (``O(n_bins * n_sources * n_decode)``
  instead of KDE's pairwise ``O(n_bins * n_enc * n_decode)`` position kernel).

This is purely additive: the default clusterless likelihood remains
``clusterless_kde``.

For electrode ``e`` the joint mark density on interior bins is

    D_e[:, j]   = sum_i w_i * K_mark(m_j, m_i) * onehot(bin(x_i))
    p_e(x, m_j) = (H D_e)[x, j] / ( (sum_i w_i) * dV(x) )

where ``i`` runs over encoding entries: each encoding spike's EM weight is split
between its two bracketing tracking samples by the interpolation fraction and
placed at their bins, as the samples' exposure is (one entry when both samples
share a bin, else two with the spike's mark). ``K_mark`` is the Gaussian
mark kernel with its normalizer, ``H`` is the fixed spatial kernel
(:func:`_kernel_columns`: heat-kernel columns clipped, kept within their
connected component, cut below ``_FIELD_RESOLUTION`` of their peak and
normalized to unit mass) that also diffuses occupancy and the ground process,
``sum_i w_i`` the weighted encoding count, and ``dV(x) = bin_sizes`` the per-bin
measure. Because ``H``'s columns sum to one, dividing by ``(sum_i w_i) * dV(x)``
makes ``p_e`` a proper joint density whose spatial integral recovers the
weighted mark marginal, and the marked intensity integrates over marks to
exactly the expected count.

The likelihood follows the shared marked-point-process contract::

    log L_e = sum_{j: decode spikes on e} log(lambda_e(x, m_j) + beta * q_e(m_j))
              - integral (lambda_bar_e(x) + beta) dt
    lambda_e(x, m)  = mean_rate_e * p_e(x, m) / occupancy(x)
    lambda_bar_e(x) = mean_rate_e * p_gpi_e(x) / occupancy(x)

where ``beta = RATE_EPS_HZ`` is the background rate and ``q_e`` the electrode's
reference mark density (the per-spike sum is formed with ``logaddexp``).

Occupancy and the mark-marginalized ground-process field are diffused once at fit.
Per block of decode spikes, each spike's mark kernels are combined per source bin
in log space and applied through ``H`` relative to the spike's strongest source;
a spike whose weaker sources could underflow there yet reach its background is
recomputed exactly in log space. ``H`` is built once per prediction when it fits
half of ``memory_budget`` and otherwise in tiles of columns.
"""

import logging
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.encoding_time import (
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_kde import (
    _compute_log_mark_kernel_gemm,
    _masked_logaddexp,
    _masked_logsumexp,
    _repair_flagged,
    _zero_if_neginf,
)
from non_local_detector.likelihoods.clusterless_kde import (
    mark_reference_scale as mark_reference_scale,
)
from non_local_detector.likelihoods.common import (
    DEFAULT_WAVEFORM_STD,
    EPS,
    LOG_RATE_EPS_HZ,
    _pad_rows,
    _padded_sample_count,
    _padded_spike_count,
    _SpikeTimeOrder,
    as_std_array,
    decode_bin_centers,
    deterministic_row_sum,
    environment_grid_fingerprint,
    get_position_at_time,
    log_bin_duration_evidence,
    log_density_intensity,
    log_nonnegative,
    resolve_row_slice,
    select_decode_marks,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    sum_spikes_into_rows,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.diffusion import (
    _FIELD_RESOLUTION,
    _resolved_occupancy,
    cached_eigenbasis,
    cached_heat_kernel_eigenbasis,
    check_smoothing_bandwidth,
    environment_graph,
    get_device_basis,
    heat_kernel_apply,
)
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_log_intensity,
    background_rate,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    _full_to_local,
    _interior_bin_indices,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
)

logger = logging.getLogger(__name__)

# float32 working dtype for the diffusion matmul; its item size drives the memory
# policy -- do not hardcode 4 elsewhere.
_WORKING_ITEMSIZE = np.dtype(np.float32).itemsize


def _validate_diffusion_params(
    position_std: float,
    waveform_std,
    heat_kernel_rank: int | None,
    memory_budget: int,
    block_size: int,
) -> float:
    """Raise-on-invalid validation for the diffusion-specific parameters.

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


def _validate_waveform_std_shape(waveform_std, n_features: int, electrode: int) -> None:
    """Reject a ``waveform_std`` that cannot broadcast to this electrode's marks.

    ``waveform_std`` must be a scalar (broadcast to every feature) or a 1-D array
    of length ``n_features``. Fit never evaluates the mark kernel, so an ill-shaped
    ``waveform_std`` (empty, or a length mismatching the waveform-feature dimension)
    would otherwise pass fit silently and fail later in the mark kernel at
    predict. Caught here per electrode instead.
    """
    waveform_std_arr = np.asarray(waveform_std, dtype=float)
    if waveform_std_arr.ndim == 0:
        return  # a scalar bandwidth broadcasts to any feature dimension
    if waveform_std_arr.ndim != 1 or waveform_std_arr.shape[0] != n_features:
        raise ValidationError(
            "waveform_std must be a scalar or a 1-D array of length n_features "
            "(matching each electrode's waveform-feature dimension)",
            expected=f"scalar or shape ({n_features},) for electrode {electrode}",
            got=f"waveform_std of shape {waveform_std_arr.shape}",
        )


def _effective_block(
    memory_budget: int,
    n_enc: int,
    n_sources: int,
    n_bins: int,
    block_size: int,
) -> int:
    """Decode-spike block size that fits the per-block live set in ``memory_budget``.

    Per decode-spike column the live set is dominated by the weighted log mark
    kernels (``n_enc``), the per-source log and scaled mark densities
    (``2 n_sources``) and ~3 ``n_bins``-scaled arrays (the field, its log and the
    log intensity), all in the float32 working dtype; kernel tiles are sized
    separately (:func:`_source_tile_size`). A safety factor of 2 leaves headroom
    for XLA workspace. The fitting size is rounded down to a power of two so
    electrodes with nearby encoding counts share compiled kernels.
    """
    bytes_per_col = _WORKING_ITEMSIZE * (n_enc + 2 * n_sources + 3 * n_bins)
    raw = max(memory_budget // (bytes_per_col * 2), 1)
    return min(1 << (raw.bit_length() - 1), block_size)


def fit_clusterless_diffusion_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    *,
    position_std: float = float(np.sqrt(12.5)),
    waveform_std: float = DEFAULT_WAVEFORM_STD,
    weights: np.ndarray | None = None,
    heat_kernel_rank: int | None = None,
    block_size: int = 10000,
    memory_budget: int = 536870912,
    disable_progress_bar: bool = False,
    encoding_time_range=None,
    valid_position_intervals=None,
    _encoding_support=None,
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
        effective block is the memory-budget-resolved value.
    memory_budget : int, optional
        Byte budget, by default 536_870_912 (512 MiB). It sets the decode-spike
        block size and the spatial kernel's tile size (at fit and predict), and
        whether predict builds the whole kernel once (when it fits half the
        budget) or streams it in tiles.
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.

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
        Keys consumed by :func:`predict_clusterless_diffusion_log_likelihood`:
        ``environment``, ``occupancy`` (pi, ``(n_interior,)``),
        ``summed_ground_process_intensity`` (``(n_interior,)``),
        ``encoding_bin_indices`` (list per electrode), ``encoding_marks`` (list),
        ``encoding_weights`` (list): one bin-sorted entry per encoding spike, or
        two when its bracketing tracking samples lie in different bins, each
        with its share of the spike's weight; ``weight_total`` (list of float),
        ``mean_rates`` (list), ``resolved_rank`` (int), ``node_order``,
        ``bin_sizes`` (dV per interior bin), ``position_std``, ``waveform_std``,
        ``block_size``, ``memory_budget``, ``disable_progress_bar``,
        ``rate_units`` (``"Hz"``), ``encoding_exposure_seconds`` (weighted
        encoding occupancy time in seconds).
    """
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
    graph, node_order, bin_sizes = environment_graph(environment)
    # Warn if position_std under-smooths relative to the grid spacing (the smoother
    # barely spreads past one bin -> near-raw histograms), matching sorted_spikes_diffusion.
    check_smoothing_bandwidth(position_std, graph)
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

    weight_sum = float(exposure_weights.sum())  # weighted occupancy seconds

    # Exposure and encoding spikes share the samples' interpolation basis: each
    # sample's exposure goes to its bin, and each spike's weight is split between
    # its two bracketing samples and binned with them, so spikes land only where
    # exposure does even when the animal moves more than a bin per sample.
    sample_bins = np.full(n_time_position, -1)
    if support.indices.size:
        sample_bins[support.indices] = _interior_bin_indices(
            environment,
            get_position_at_time(
                position_time, position, position_time[support.indices], environment
            ),
            full_to_local,
        )

    # ---- occupancy pi = H O / (sum w_pos * dV), floored at EPS ----
    active = exposure_weights > 0
    occupancy_field = np.bincount(
        sample_bins[active], weights=exposure_weights[active], minlength=n_interior
    )
    warn_if_no_encoding_exposure(
        exposure_weights, "Clusterless diffusion", clusterless=True
    )

    # ---- per electrode: marks, bin indices, weights, mean rate, ground process ----
    encoding_bin_indices: list[np.ndarray] = []
    encoding_marks: list[jnp.ndarray] = []
    encoding_weights: list[jnp.ndarray] = []
    weight_total: list[float] = []
    mean_rates: list[float] = []
    # Count fields to diffuse together: occupancy, then each nonzero-rate electrode.
    fields = [occupancy_field]

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
        spike_index, left, right, fraction = support.event_basis(electrode_spike_times)
        # Validate only the in-window features that actually enter the fit.
        bounded_features = np.asarray(electrode_features)[spike_index]
        validate_finite(bounded_features, "spike_waveform_features")
        _validate_waveform_std_shape(waveform_std, bounded_features.shape[1], electrode)

        # One entry per spike whose bracketing samples share a bin; otherwise a
        # second entry carries the mark into the right sample's bin.
        left_weight = (1.0 - fraction) * weights[left]
        right_weight = fraction * weights[right]
        left_bins, right_bins = sample_bins[left], sample_bins[right]
        split = (left_bins != right_bins) & (right_weight > 0)
        bins = np.concatenate([left_bins, right_bins[split]])
        spike_weights = np.concatenate(
            [
                np.where(split, left_weight, left_weight + right_weight),
                right_weight[split],
            ]
        )
        entry_features = np.concatenate([bounded_features, bounded_features[split]])
        w_total = float(spike_weights.sum())

        # Store encoding entries sorted by bin: predict sums them by bin with
        # indices_are_sorted=True, which requires nondecreasing ids.
        order = np.argsort(bins, kind="stable")

        encoding_bin_indices.append(bins[order])
        encoding_marks.append(jnp.asarray(entry_features[order]))
        encoding_weights.append(jnp.asarray(spike_weights[order]))
        weight_total.append(w_total)

        if w_total == 0:
            # Zero-rate electrode: no fitted density, no ground-process contribution.
            mean_rates.append(0.0)
            warnings.warn(
                f"electrode {electrode} has zero total encoding weight (zero-rate): "
                "no effective encoding spikes. It contributes no ground-process "
                "intensity and its decode spikes are explained by the background rate.",
                UserWarning,
                stacklevel=2,
            )
            continue

        mean_rates.append(weighted_mean_rate(spike_weights, weight_sum))
        fields.append(np.bincount(bins, weights=spike_weights, minlength=n_interior))

    diffused = _diffuse_fields(
        np.stack(fields, axis=1),
        environment,
        resolved_rank,
        position_std,
        memory_budget,
    )
    safe_weight_sum = weight_sum if weight_sum > 0 else 1.0
    occupancy = np.clip(diffused[:, 0] / (safe_weight_sum * bin_sizes), EPS, None)
    summed_ground_process_intensity = np.zeros((n_interior,))
    spike_fields_hat = iter(diffused[:, 1:].T)
    for mean_rate, w_total in zip(mean_rates, weight_total, strict=True):
        if w_total == 0:
            continue
        summed_ground_process_intensity += next(spike_fields_hat) * np.exp(
            np.asarray(_log_intensity_scale(occupancy, bin_sizes, mean_rate, w_total))
        )

    return {
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
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
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


def _pad_sources(sources, n_padded: int) -> np.ndarray:
    """Source bins padded to ``n_padded`` with repeats of the last (bin 0 if none)."""
    sources = np.asarray(sources, dtype=np.int32)
    if sources.size == 0:
        return np.zeros(n_padded, dtype=np.int32)
    return np.pad(sources, (0, n_padded - sources.size), mode="edge")


# The scaled spatial product multiplies exp(log_source - peak) by kernel
# entries; a product below float32's smallest normal number may flush to zero.
_LOG_TINY = float(np.log(np.finfo(np.float32).tiny))


@partial(jax.jit, static_argnames=("position_std", "n_components"))
def _kernel_columns(eigvals, eigvecs, labels, sources, *, position_std, n_components):
    """Diffusion of a unit mass from each source bin, shape (n_bins, n_sources).

    Column ``k`` is the heat kernel applied to bin ``sources[k]``: clipped to be
    non-negative, zero outside that bin's connected component, cut below
    ``_FIELD_RESOLUTION`` of its peak (heat-kernel noise) and normalized to unit
    mass. Every column depends only on its own source, so the model's one
    spatial kernel ``H`` can be built in tiles of sources. Occupancy, the ground
    process and every spike's marked field are the same linear map ``H`` of
    their sources, so the marked intensity integrates over marks to exactly the
    expected count, and disconnected components stay exactly zero.

    Parameters
    ----------
    eigvals, eigvecs, labels : jnp.ndarray
        Device heat-kernel basis and component labels (``get_device_basis``).
    sources : jnp.ndarray, shape (n_sources,)
        Interior bin indices.
    position_std : float
        Heat-kernel bandwidth in position units.
    n_components : int

    Returns
    -------
    columns : jnp.ndarray, shape (n_bins, n_sources)
    """
    n_bins = eigvecs.shape[0]
    onehot = (
        jnp.zeros((n_bins, sources.shape[0]), eigvecs.dtype)
        .at[sources, jnp.arange(sources.shape[0])]
        .set(1.0)
    )
    columns = heat_kernel_apply(
        eigvals, eigvecs, position_std, onehot, labels, n_components=n_components
    )
    columns = jnp.where(
        columns >= _FIELD_RESOLUTION * jnp.max(columns, axis=0, keepdims=True),
        columns,
        0.0,
    )
    return columns / jnp.sum(columns, axis=0, keepdims=True)


def spatial_kernel_columns(environment, resolved_rank, position_std, sources):
    """The model's spatial kernel ``H[:, sources]``, for a fitted environment.

    Parameters
    ----------
    environment : Environment
    resolved_rank : int
        The fitted model's ``resolved_rank``.
    position_std : float
    sources : array_like of int, shape (n_sources,)
        Interior bin indices.

    Returns
    -------
    columns : jnp.ndarray, shape (n_bins, n_sources)
        Non-negative columns that each sum to one: the diffusion of a unit mass
        from each source bin, zero outside its connected component and below
        ``_FIELD_RESOLUTION`` of its peak. ``resolved_rank`` and
        ``position_std`` must be the fitted model's.
    """
    eigvals, eigvecs, labels, n_components = get_device_basis(
        environment, resolved_rank
    )
    sources = np.asarray(sources, dtype=np.int32)
    # Pad to few sizes so callers with nearby source counts share one executable.
    padded = _pad_sources(sources, _padded_sample_count(max(sources.size, 1)))
    return _kernel_columns(
        eigvals,
        eigvecs,
        labels,
        jnp.asarray(padded),
        position_std=float(position_std),
        n_components=n_components,
    )[:, : sources.size]


def _dense_kernel_fits(n_bins: int, memory_budget: int) -> bool:
    """Whether the whole ``(n_bins, n_bins)`` kernel fits half of ``memory_budget``."""
    return 2 * _WORKING_ITEMSIZE * n_bins * n_bins <= memory_budget


def _dense_sources(n_bins: int, tile_budget: int):
    """Every bin as a source, padded to whole tiles with repeats of the last.

    Returns
    -------
    sources : np.ndarray, shape (n_padded_sources,)
    tile : int
    """
    tile = min(tile_budget, n_bins)
    return _pad_sources(np.arange(n_bins), -(-n_bins // tile) * tile), tile


@partial(jax.jit, static_argnames=("tile", "position_std", "n_components"))
def _dense_kernel(
    eigvals, eigvecs, labels, sources, *, tile, position_std, n_components
):
    """Kernel columns of every (padded) dense source, shape (n_bins, n_sources).

    Built once per prediction in tiles; see :func:`_dense_sources`.
    """
    tiles = jax.lax.map(
        lambda start: _kernel_columns(
            eigvals,
            eigvecs,
            labels,
            jax.lax.dynamic_slice_in_dim(sources, start, tile),
            position_std=position_std,
            n_components=n_components,
        ),
        jnp.arange(sources.shape[0] // tile) * tile,
    )  # (n_tiles, n_bins, tile)
    return jnp.moveaxis(tiles, 0, 1).reshape(tiles.shape[1], -1)


def _source_tile_size(memory_budget: int, n_bins: int, rank: int) -> int:
    """Kernel columns per tile, so a tile and its spectral temporaries fit half
    of ``memory_budget`` (a power of two, for shared compiled kernels)."""
    raw = max(memory_budget // (2 * _WORKING_ITEMSIZE * (3 * n_bins + rank)), 1)
    return 1 << (raw.bit_length() - 1)


def _diffuse_fields(
    fields: np.ndarray,
    environment,
    resolved_rank: int,
    position_std: float,
    memory_budget: int,
) -> np.ndarray:
    """``H @ fields`` for ``(n_interior, n_fields)`` count fields, on host.

    The occupancy and per-electrode ground-process fields are diffused together
    at fit time, by the same linear map as every spike's marked field. Only the
    bins with mass in some field contribute, in tiles of kernel columns.
    """
    eigvals, eigvecs, labels, n_components = get_device_basis(
        environment, resolved_rank
    )
    fields = np.asarray(fields, dtype=np.float64)
    sources = np.flatnonzero(np.any(fields != 0, axis=1))
    out = np.zeros(fields.shape)
    tile = _source_tile_size(memory_budget, fields.shape[0], int(eigvecs.shape[1]))
    for start in range(0, sources.size, tile):
        chunk = sources[start : start + tile]
        columns = np.asarray(
            _kernel_columns(
                eigvals,
                eigvecs,
                labels,
                jnp.asarray(_pad_sources(chunk, tile)),
                position_std=float(position_std),
                n_components=n_components,
            ),
            dtype=np.float64,
        )[:, : chunk.size]
        out += columns @ fields[chunk]
    return out


def _log_intensity_scale(occupancy, bin_sizes, mean_rate, weight_total):
    """Per-bin log of ``mean_rate / (sum_i w_i * dV(x) * occupancy(x))``.

    Multiplying a diffused source field by this gives the intensity. Bins
    whose occupancy the heat kernel does not resolve (see
    :func:`_resolved_occupancy`) carry no encoding information: their scale is
    ``-inf``, so their marked and ground intensities are zero and their spikes
    are explained by the background alone. Shared by decoding and model
    checking.

    Parameters
    ----------
    occupancy, bin_sizes : array, shape (n_bins,)
    mean_rate, weight_total : float

    Returns
    -------
    log_scale : array, shape (n_bins,)
    """
    occupancy = jnp.asarray(occupancy)
    return jnp.where(
        _resolved_occupancy(occupancy),
        log_density_intensity(
            mean_rate,
            -jnp.log(weight_total * jnp.asarray(bin_sizes)),
            jnp.log(occupancy),
        ),
        -jnp.inf,
    )


def _source_layout(bins: np.ndarray, n_bins: int, tile_budget: int):
    """Source bins of padded, bin-sorted encoding samples, padded to few sizes.

    Parameters
    ----------
    bins : np.ndarray, shape (n_padded_samples,)
        Bin of each encoding sample; padding samples use ``n_bins``.
    n_bins : int
    tile_budget : int
        Largest kernel tile (see :func:`_source_tile_size`).

    Returns
    -------
    sources : np.ndarray, shape (n_padded_sources,)
        Distinct bins holding encoding samples, then repeats of the last; a
        multiple of ``tile`` long.
    source_index : np.ndarray, shape (n_padded_samples,)
        Each sample's position in ``sources``, ascending; padding samples use
        the last slot (their weight is zero).
    tile : int
        Kernel columns per tile.
    """
    real = bins < n_bins
    distinct, inverse = np.unique(bins[real], return_inverse=True)
    n_padded = _padded_sample_count(max(distinct.size, 1))
    tile = min(tile_budget, n_padded)
    n_padded = -(-n_padded // tile) * tile
    source_index = np.full(bins.size, n_padded - 1, dtype=np.int32)
    source_index[real] = inverse
    return _pad_sources(distinct, n_padded), source_index, tile


def _electrode_layout(
    bins, marks, weights, n_bins: int, tile_budget: int, dense: bool = False
):
    """Padded, bin-sorted encoding samples with their source layout, for one electrode.

    Models fitted before encoding spikes were stored bin-sorted are sorted here
    (stably). Padding samples take the dropped bin ``n_bins``, zero marks and
    zero weight, so electrodes with nearby encoding counts share compiled
    kernels. With a ``dense`` kernel (one shared array of every bin's column, see
    :func:`_dense_kernel`) every bin is a source, padded to whole tiles, so
    all electrodes share one layout shape; otherwise sources are the bins
    holding encoding samples. Either way the per-source bounds and the exact
    repair reduce the kernel one tile of columns at a time.

    Returns
    -------
    marks : jnp.ndarray, shape (n_padded_samples, n_features)
    weights : jnp.ndarray, shape (n_padded_samples,)
        Zero for padding samples.
    sources : jnp.ndarray, shape (n_padded_sources,)
        Source bins, a whole number of tiles long.
    source_index : jnp.ndarray, shape (n_padded_samples,)
        Each sample's position in ``sources``, ascending.
    tile : int
        Kernel columns per tile.
    """
    bins, marks, weights = np.asarray(bins), np.asarray(marks), np.asarray(weights)
    if np.any(bins[1:] < bins[:-1]):
        order = np.argsort(bins, kind="stable")
        bins, marks, weights = bins[order], marks[order], weights[order]
    n_padded = _padded_sample_count(bins.shape[0])
    padded_bins = np.full(n_padded, n_bins, dtype=bins.dtype)
    padded_bins[: bins.shape[0]] = bins
    if dense:
        sources, tile = _dense_sources(n_bins, tile_budget)
        source_index = np.where(padded_bins < n_bins, padded_bins, sources.size - 1)
    else:
        sources, source_index, tile = _source_layout(padded_bins, n_bins, tile_budget)
    return (
        jnp.asarray(_pad_rows(marks, n_padded)),
        jnp.asarray(_pad_rows(weights, n_padded)),
        jnp.asarray(sources),
        jnp.asarray(source_index, dtype=jnp.int32),
        tile,
    )


def _log_source_marks(
    decode_features,
    encoding_marks,
    encoding_weights,
    source_index,
    waveform_std,
    n_sources,
):
    """Log of ``sum_i w_i K_mark(m_j, m_i)`` over each source bin's samples.

    Reduced in log space within each source, so no source is lost however far
    its marks lie from another's.

    Returns
    -------
    log_source : jnp.ndarray, shape (n_sources, n_block)
        ``-inf`` for a source without positively weighted samples.
    """
    weighted = (
        _compute_log_mark_kernel_gemm(decode_features, encoding_marks, waveform_std)
        + log_nonnegative(encoding_weights)[:, None]
    )  # (n_samples, n_block)
    peak = jax.ops.segment_max(
        weighted, source_index, num_segments=n_sources, indices_are_sorted=True
    )
    safe_peak = _zero_if_neginf(peak)
    total = deterministic_row_sum(
        jnp.exp(weighted - safe_peak[source_index]),
        source_index,
        n_sources,
        indices_are_sorted=True,
    )
    return jnp.where(jnp.isneginf(peak), -jnp.inf, log_nonnegative(total) + peak)


def _tile_columns(basis, sources, cached, start, *, position_std, n_components, tile):
    """Kernel columns of ``sources[start:start + tile]``, shape (n_bins, tile).

    Sliced from ``cached`` (columns aligned with ``sources``) when given, else
    built from the heat-kernel basis.
    """
    if cached is not None:
        return jax.lax.dynamic_slice_in_dim(cached, start, tile, axis=1)
    eigvals, eigvecs, labels = basis
    return _kernel_columns(
        eigvals,
        eigvecs,
        labels,
        jax.lax.dynamic_slice_in_dim(sources, start, tile),
        position_std=position_std,
        n_components=n_components,
    )


@partial(jax.jit, static_argnames=("tile", "position_std", "n_components"))
def _kernel_bounds(
    eigvals,
    eigvecs,
    labels,
    kernel,
    sources,
    log_scale,
    *,
    tile,
    position_std,
    n_components,
):
    """Per-source bounds for the exact-repair gate, one tile of columns at a time.

    Computed once per prediction for every bin, with the scale of a unit rate
    and weight; an electrode's gain adds its ``log(mean_rate / weight_total)``.

    Parameters
    ----------
    eigvals, eigvecs, labels : jnp.ndarray
        Device heat-kernel basis and component labels.
    kernel : jnp.ndarray, shape (n_bins, n_sources), or None
        The kernel columns of ``sources`` when already built.
    sources : jnp.ndarray, shape (n_sources,)
        A whole number of tiles long.
    log_scale : jnp.ndarray, shape (n_bins,)
        :func:`_log_intensity_scale` for a unit rate and weight.
    tile : int
    position_std : float
    n_components : int

    Returns
    -------
    gain : jnp.ndarray, shape (n_sources,)
        Largest ``log_scale(x) + log H[x, s]`` over bins: how much a source can
        add to any bin's log intensity per unit of its mark density.
    floor : jnp.ndarray, shape (n_sources,)
        Smallest positive ``log H[x, s]`` over bins: a source's scaled mark
        density times this may underflow first.
    """
    columns_of = partial(
        _tile_columns,
        (eigvals, eigvecs, labels),
        sources,
        kernel,
        position_std=position_std,
        n_components=n_components,
        tile=tile,
    )

    def tile_bounds(start):
        columns = columns_of(start)
        log_columns = log_nonnegative(columns)
        return (
            jnp.max(log_scale[:, None] + log_columns, axis=0),
            jnp.min(jnp.where(columns > 0, log_columns, jnp.inf), axis=0),
        )

    starts = jnp.arange(sources.shape[0] // tile) * tile
    gain, floor = jax.lax.map(tile_bounds, starts)
    return gain.reshape(-1), floor.reshape(-1)


def _log_marked_field(
    log_source, rows, columns_of, n_tiles, tile, bounds, log_background
):
    """``log(H D)`` for one block of spikes, at every bin or at each spike's own.

    The scaled product ``H exp(log_source - peak)`` is applied in tiles of
    kernel columns. A source's scaled mark density times its smallest kernel
    entry may fall below float32's smallest normal number and flush to zero; if
    such a source could reach the spike's background floor at any bin, that
    spike is recomputed by an exact log-space reduction over the sources.

    Parameters
    ----------
    log_source : jnp.ndarray, shape (n_sources, n_block)
    rows : jnp.ndarray, shape (n_block,), or None
        Each spike's own bin (local), or None to read every bin (non-local).
    columns_of : callable
        ``columns_of(start)`` gives the kernel columns of sources
        ``start:start + tile``, shape (n_bins, tile).
    n_tiles, tile : int
    bounds : tuple of jnp.ndarray, each shape (n_sources,)
        ``(gain, floor)`` of each source (see :func:`_kernel_bounds`).
    log_background : jnp.ndarray, shape (n_block,)

    Returns
    -------
    log_field : jnp.ndarray, shape (n_bins, n_block), or (n_block,) when local
    """
    local = rows is not None
    peak = jnp.max(log_source, axis=0)
    safe_peak = _zero_if_neginf(peak)
    scaled = jnp.exp(log_source - safe_peak[None, :])
    n_bins = jax.eval_shape(columns_of, 0).shape[0]

    def read(columns, at):
        return columns if at is None else columns[at]

    def add_tile(index, field):
        start = index * tile
        columns = read(columns_of(start), rows)  # (n_bins or n_block, tile)
        part = jax.lax.dynamic_slice_in_dim(scaled, start, tile, axis=0)
        if local:
            return field + jnp.sum(columns * part.T, axis=1)
        return field + jnp.matmul(columns, part, precision=jax.lax.Precision.HIGHEST)

    shape = peak.shape if local else (n_bins, peak.shape[0])
    field = jax.lax.fori_loop(0, n_tiles, add_tile, jnp.zeros(shape, scaled.dtype))
    log_field = log_nonnegative(field) + (peak if local else peak[None, :])

    # Sources the scaled product may have lost, and whether any could matter.
    gain, floor = bounds
    dropped = log_source - safe_peak[None, :] + floor[:, None] < _LOG_TINY + 1.0
    lost_bound = jnp.max(
        jnp.where(dropped, log_source + gain[:, None], -jnp.inf), axis=0
    )
    needs_exact = lost_bound > log_background + np.log(np.finfo(np.float32).eps)

    def exact(j, at):
        # log sum_s H[at, s] exp(log_source[s, j]), reduced in log space.
        def add_exact_tile(index, value):
            start = index * tile
            log_columns = log_nonnegative(read(columns_of(start), at))
            part = jax.lax.dynamic_slice_in_dim(log_source[:, j], start, tile)
            terms = jnp.where(
                jnp.isneginf(part)[None, :], -jnp.inf, log_columns + part[None, :]
            )
            return _masked_logaddexp(value, _masked_logsumexp(terms.T))

        n_read = n_bins if at is None else at.shape[0]
        return jax.lax.fori_loop(
            0, n_tiles, add_exact_tile, jnp.full((n_read,), -jnp.inf)
        )

    def repaired(values, j, flag):
        if local:
            fixed = jax.lax.cond(
                flag, lambda: exact(j, rows[j][None])[0], lambda: values[j]
            )
            return values.at[j].set(fixed)
        fixed = jax.lax.cond(flag, lambda: exact(j, None), lambda: values[:, j])
        return values.at[:, j].set(fixed)

    # Flagged spikes are rare (none of 75,458 on a real recording). Each one
    # reduces every kernel tile in log space; streamed tiles are rebuilt from
    # the basis per spike, so many flags in one block are slow but exact.
    return _repair_flagged(log_field, needs_exact, repaired)


@partial(
    jax.jit,
    static_argnames=(
        "position_std",
        "n_components",
        "tile",
        "block_size",
        "indices_are_sorted",
    ),
    donate_argnums=0,
)
def _add_electrode_diffusion_terms(
    total,
    decode_features,
    row_ids,
    spike_bins,
    encoding_marks,
    encoding_weights,
    source_index,
    sources,
    waveform_std,
    eigvals,
    eigvecs,
    labels,
    kernel,
    log_scale,
    bounds,
    log_background,
    *,
    position_std,
    n_components,
    tile,
    block_size,
    indices_are_sorted,
):
    """Add one electrode's log mark intensities to their rows.

    Non-local intensities cover every bin; local ones read the same kernel
    columns at the animal's interior bin.

    Parameters
    ----------
    total : jnp.ndarray, shape (n_rows, n_bins), or (n_rows,) when local
        Running sum over electrodes; donated.
    decode_features : jnp.ndarray, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    spike_bins : jnp.ndarray, shape (n_padded_spikes,), or None
        Animal's interior bin at each decoding spike (padding rows are zero)
        for local intensities; None for non-local.
    encoding_marks, encoding_weights, source_index, sources : jnp.ndarray
        From :func:`_electrode_layout`.
    waveform_std : jnp.ndarray, shape (n_features,)
    eigvals, eigvecs, labels : jnp.ndarray
        Device heat-kernel basis and component labels.
    kernel : jnp.ndarray, shape (n_bins, n_sources), or None
        The whole spatial kernel when it fits the memory budget
        (:func:`_dense_kernel`); then ``sources`` are every bin. Otherwise
        kernel columns are built from the basis, tile by tile.
    log_scale : jnp.ndarray, shape (n_bins,)
        From :func:`_log_intensity_scale`.
    bounds : tuple of jnp.ndarray, each shape (n_sources,)
        This electrode's ``(gain, floor)`` of ``sources`` (see
        :func:`_kernel_bounds`).
    log_background : jnp.ndarray, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.
    position_std : float
    n_components : int
    tile : int
        Kernel columns per tile (divides ``sources``' length).
    block_size : int
        Decoding spikes per block.
    indices_are_sorted : bool

    Returns
    -------
    total : jnp.ndarray, shape (n_rows, n_bins), or (n_rows,) when local
    """
    columns_of = partial(
        _tile_columns,
        (eigvals, eigvecs, labels),
        sources,
        kernel,
        position_std=position_std,
        n_components=n_components,
        tile=tile,
    )
    n_tiles = sources.shape[0] // tile

    def add_block(block, total):
        start = block * block_size
        background = jax.lax.dynamic_slice_in_dim(log_background, start, block_size)
        rows = (
            None
            if spike_bins is None
            else jax.lax.dynamic_slice_in_dim(spike_bins, start, block_size)
        )
        log_field = _log_marked_field(
            _log_source_marks(
                jax.lax.dynamic_slice_in_dim(decode_features, start, block_size),
                encoding_marks,
                encoding_weights,
                source_index,
                waveform_std,
                sources.shape[0],
            ),
            rows,
            columns_of,
            n_tiles,
            tile,
            bounds,
            background,
        )
        if rows is None:
            log_intensity = jnp.logaddexp(
                log_scale[:, None] + log_field, background[None, :]
            ).T  # (n_block, n_bins)
        else:
            log_intensity = jnp.logaddexp(log_scale[rows] + log_field, background)
        return total + deterministic_row_sum(
            log_intensity,
            jax.lax.dynamic_slice_in_dim(row_ids, start, block_size),
            total.shape[0],
            indices_are_sorted=indices_are_sorted,
        )

    n_blocks = decode_features.shape[0] // block_size
    return jax.lax.fori_loop(0, n_blocks, add_block, total)


@requires_time_edges
def predict_clusterless_diffusion_log_likelihood(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    *,
    time_edges: np.ndarray,
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,
    is_local: bool = False,
    row_slice: slice | None = None,
    mark_reference: list[MarkReference | None],
    environment_grid: dict | None = None,
    _spike_time_order: _SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
    **encoding_model: object,
) -> jnp.ndarray:
    """Predict the clusterless graph-diffusion log likelihood.

    Parameters
    ----------
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
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
        bin; the same kernel columns as the non-local path, read at the animal's
        bin). By default False.
    row_slice : slice | None, optional
        Contiguous range of output rows to compute, by default None (all rows).
        ``time_edges`` always stay the FULL decoding edges: spikes are binned
        against them and only those owned by the requested rows are evaluated, so
        the result equals the full-time result sliced by ``row_slice`` while the
        spatial workspaces scale with the requested rows and selected spikes.
    mark_reference : list of MarkReference or None, keyword-only
        Normalized reference mark density per electrode, from
        ``fit_mark_references`` (a detector passes its ``mark_reference_``).
        Each spike's intensity gains ``RATE_EPS_HZ * q(mark)`` and each
        electrode's expected count ``RATE_EPS_HZ``, so likelihood ratios between
        states do not depend on the waveform units. None for an electrode means
        no training marks; its background is the bare rate.
    environment_grid : dict | None, optional
        The fit's record of the environment's grid (see
        ``environment_grid_fingerprint``), by default None, which skips the
        check (a model fitted before the grid was recorded). Prediction raises
        if the environment has since been refit to another grid.
    _spike_time_order : _SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the encoding fit; anything other than ``"Hz"``
        raises ``ValidationError`` (refit legacy encoding models).
    encoding_exposure_seconds : float, optional, keyword-only
        Weighted encoding occupancy time in seconds, recorded by the fit.
        Accepted so the encoding dictionary can be passed as keywords; unused.
    **encoding_model
        The dict returned by :func:`fit_clusterless_diffusion_encoding_model`.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, n_interior_bins) if ``is_local`` is
        False, else (n_rows, 1). ``n_rows`` is ``n_bins`` unless ``row_slice``
        is given.
    """
    environment: Environment = encoding_model["environment"]  # type: ignore[assignment]
    validate_encoding_contract(rate_units, environment, environment_grid)
    occupancy = jnp.asarray(encoding_model["occupancy"])
    summed_ground_process_intensity = jnp.asarray(
        encoding_model["summed_ground_process_intensity"]
    ) + background_rate(len(spike_times))
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

    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        encoding_bin_indices=encoding_bin_indices,  # type: ignore[arg-type]
        encoding_marks=encoding_marks,  # type: ignore[arg-type]
        encoding_weights=encoding_weights,  # type: ignore[arg-type]
        weight_total=weight_total,  # type: ignore[arg-type]
        mean_rates=mean_rates,  # type: ignore[arg-type]
        mark_reference=mark_reference,  # type: ignore[arg-type]
    )
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_bins = occupancy.shape[0]

    eigvals, eigvecs, labels, n_components = get_device_basis(
        environment, resolved_rank
    )
    tile_budget = _source_tile_size(memory_budget, n_bins, resolved_rank)
    dense = _dense_kernel_fits(n_bins, memory_budget)
    every_bin, every_bin_tile = _dense_sources(n_bins, tile_budget)
    every_bin = jnp.asarray(every_bin)
    basis_static = {"position_std": position_std, "n_components": n_components}
    # One kernel per prediction, shared by every electrode, state and chunk.
    kernel = (
        _spike_time_order.memo(
            f"diffusion_kernel:{position_std!r}:{every_bin_tile}",
            (eigvecs, labels),
            partial(
                _dense_kernel,
                eigvals,
                eigvecs,
                labels,
                every_bin,
                tile=every_bin_tile,
                **basis_static,
            ),
        )
        if dense
        else None
    )
    # Every bin's repair-gate bounds for a unit rate and weight, once per model.
    kernel_gain, kernel_floor = _spike_time_order.memo(
        f"diffusion_kernel_bounds:{position_std!r}:{every_bin_tile}:{dense}",
        (eigvecs, labels, encoding_model["occupancy"]),
        lambda: _kernel_bounds(
            eigvals,
            eigvecs,
            labels,
            kernel,
            every_bin,
            _log_intensity_scale(occupancy, bin_sizes, 1.0, 1.0),
            tile=every_bin_tile,
            **basis_static,
        ),
    )

    durations = jnp.asarray(_time_grid.durations(row_start, row_stop))
    if is_local:
        # The local path reads the animal's position; non-local ignores it and the
        # base API permits position=None (as clusterless_kde does), so position is
        # only required and validated here.
        validate_finite(position, "position")
        # Reconstruct the interior-bin mapping from node_order (derived here at
        # predict, not stored) and evaluate the diffused column at the animal's nearest
        # interior bin instead of returning the whole column.
        n_total_bins = environment.is_track_interior_.ravel().shape[0]
        full_to_local = _full_to_local(node_order, n_total_bins)

        interpolated_position = get_position_at_time(
            position_time,
            position,
            decode_bin_centers(time_edges, row_start, row_stop),
            environment,
        )
        animal_time_bins = _interior_bin_indices(
            environment, interpolated_position, full_to_local
        )
        # Ground-process term evaluated at the animal's position per decode time
        # bin (mirrors clusterless_kde's local summed_expected_counts).
        log_likelihood = (
            -durations * summed_ground_process_intensity[animal_time_bins]
        )  # (n_time,)
    else:
        log_likelihood = -durations[:, None] * summed_ground_process_intensity[None, :]
    # Kernel outputs are committed to the basis's device; committing the
    # starting sum too lets every electrode reuse one executable.
    log_likelihood = jax.device_put(log_likelihood, eigvecs.sharding)

    for electrode, (
        electrode_bins,
        electrode_marks,
        electrode_weights,
        electrode_weight_total,
        electrode_mean_rate,
        electrode_decode_features,
        electrode_spike_times,
        electrode_mark_reference,
    ) in enumerate(
        zip(
            tqdm(
                encoding_bin_indices,
                unit="electrode",
                desc="Local Likelihood" if is_local else "Non-Local Likelihood",
                disable=disable_progress_bar,
            ),
            encoding_marks,
            encoding_weights,
            weight_total,
            mean_rates,
            spike_waveform_features,
            spike_times,
            mark_reference,
            strict=True,
        )
    ):
        selection = select_spikes_in_rows(
            electrode_spike_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        # Select FIRST, convert after: converting the whole array would cost
        # every decoding spike in the recording on every chunk call.
        decode_features = select_decode_marks(
            electrode_decode_features, selection, electrode
        )
        n_decode = decode_features.shape[0]
        if n_decode == 0:
            continue
        log_background = background_log_intensity(
            electrode_mark_reference, decode_features, LOG_RATE_EPS_HZ
        )

        # Zero-rate electrode: the background explains every observed spike,
        # BEFORE any division (D_e == 0 and P_e / weight_total_e is 0/0). Its
        # contribution is identical across bins -- accumulate it per time bin
        # rather than materializing an (n_decode, n_bins) array.
        if electrode_weight_total == 0:
            background_sum = sum_spikes_into_rows(log_background, selection)
            log_likelihood += background_sum if is_local else background_sum[:, None]
            continue

        enc_marks, enc_weights, sources, source_index, tile = _spike_time_order.memo(
            f"diffusion_electrode_layout:{n_bins}:{tile_budget}:{dense}",
            (electrode_bins, electrode_marks, electrode_weights),
            partial(
                _electrode_layout,
                electrode_bins,
                electrode_marks,
                electrode_weights,
                n_bins,
                tile_budget,
                dense,
            ),
        )
        effective_block = _effective_block(
            memory_budget, enc_marks.shape[0], sources.shape[0], n_bins, block_size
        )
        logger.debug(
            "clusterless_diffusion predict: effective_block=%d "
            "(requested block_size=%d, memory_budget=%d bytes, n_enc=%d, "
            "n_sources=%d, tile=%d, n_bins=%d)",
            effective_block,
            block_size,
            memory_budget,
            enc_marks.shape[0],
            sources.shape[0],
            tile,
            n_bins,
        )
        # Padded spikes (dropped from the row sums) let chunks with nearby
        # spike counts share one compiled kernel.
        n_padded = _padded_spike_count(n_decode, effective_block)
        spike_bins = None
        if is_local:
            # The animal's interior bin at each decode spike's own time
            # (nearest-bin interpolation).
            spike_bins = jnp.asarray(
                _pad_rows(
                    _interior_bin_indices(
                        environment,
                        get_position_at_time(
                            position_time,
                            position,
                            select_spike_rows(electrode_spike_times, selection),
                            environment,
                        ),
                        full_to_local,
                    ),
                    n_padded,
                )
            )
        rate_offset = float(np.log(electrode_mean_rate / electrode_weight_total))
        log_likelihood = _add_electrode_diffusion_terms(
            log_likelihood,
            jnp.asarray(_pad_rows(decode_features, n_padded)),
            jnp.asarray(spike_row_ids(selection, n_padded)),
            spike_bins,
            enc_marks,
            enc_weights,
            source_index,
            sources,
            as_std_array(waveform_std, enc_marks.shape[1]),
            eigvals,
            eigvecs,
            labels,
            kernel,
            _log_intensity_scale(
                occupancy, bin_sizes, electrode_mean_rate, electrode_weight_total
            ),
            (kernel_gain[sources] + rate_offset, kernel_floor[sources]),
            jnp.asarray(_pad_rows(log_background, n_padded)),
            tile=tile,
            block_size=min(effective_block, n_padded),
            indices_are_sorted=selection.indices_are_sorted,
            **basis_static,
        )

    evidence = log_bin_duration_evidence(
        spike_times, time_edges, row_slice, _spike_time_order, _time_grid=_time_grid
    )
    if is_local:
        return (log_likelihood + evidence)[:, None]
    return log_likelihood + evidence[:, None]
