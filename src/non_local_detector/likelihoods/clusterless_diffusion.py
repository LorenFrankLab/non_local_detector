"""Graph-diffusion clusterless (marked-point-process) likelihood.

An opt-in clusterless likelihood, selected via
``clusterless_algorithm="clusterless_diffusion"`` on
:class:`~non_local_detector.models.base.ClusterlessDetector` and its subclasses. It
estimates the same mark intensity as :mod:`clusterless_kde` but replaces the Gaussian
*position* kernel with fixed positive, mass-preserving source columns derived
from the graph heat kernel ``exp(-t L)`` (``t = position_std**2 / 2``).
The *mark* kernel stays a Gaussian over waveform features
(via ``kde_distance``): ``position_std`` is the physical heat-kernel bandwidth (in
coordinate units) and ``waveform_std`` is the mark bandwidth, exactly as in
``clusterless_kde``. This is the clusterless analog of :mod:`sorted_spikes_diffusion`.

Two properties motivate it over KDE:

- **Geometry-respecting spatial smoothing.** The heat kernel follows the
  environment's track topology (diffusion distance on the graph), not Euclidean
  distance, so it does not leak across walls, holes, or junctions the way a Gaussian
  KDE smoothed in coordinate space can.
- **Reuse.** Occupancy and each electrode's ground-process field are smoothed once
  at fit. A bounded source-operator cache is shared across encoding groups and
  EM refits. Larger grids generate source columns in memory-bounded tiles.

This is purely additive: the default clusterless likelihood remains
``clusterless_kde``.

For electrode ``e`` the joint mark density on interior bins is

    D_e[:, j]   = sum_i w_i * K_mark(m_j, m_i) * onehot(bin(x_i))
    p_e(x, m_j) = (H_t D_e)[x, j] / ( (sum_i w_i) * dV(x) )

Here ``H_t`` denotes the fixed operator whose individual heat-kernel source
columns are clipped to non-negative values and normalized within their source
component. Applying this correction before mixing sources keeps smoothing
linear, so integrating the joint mark density recovers its fitted GPI even with
a truncated eigenbasis.

``w_i`` are the per-encoding-spike EM weights, ``K_mark`` is the Gaussian
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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_kde import kde_distance
from non_local_detector.likelihoods.common import (
    EPS,
    LOG_RATE_EPS_HZ,
    RATE_EPS_HZ,
    _pad_rows,
    _padded_sample_count,
    _padded_spike_count,
    _SpikeTimeOrder,
    as_std_array,
    decode_bin_centers,
    deterministic_row_sum,
    get_position_at_time,
    interpolate_weights_at_spike_times,
    log_bin_duration_evidence,
    resolve_row_slice,
    safe_log,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    sum_spikes_into_rows,
    validate_finite,
    validate_population_lengths,
    validate_weights,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    cached_heat_kernel_eigenbasis,
    check_smoothing_bandwidth,
    environment_graph,
    get_device_basis,
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
_SOURCE_CACHE_MAX_BYTES = 64 * 1024 * 1024


def _validate_diffusion_params(
    position_std: float,
    waveform_std,
    heat_kernel_rank: int | None,
    memory_budget: int,
    block_size: int,
) -> float:
    """Tier-1 (raise-on-invalid) validation for the diffusion-specific parameters.

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
    would otherwise pass fit silently and fail later inside ``kde_distance`` at
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
    rank: int,
    n_bins: int,
    block_size: int,
) -> int:
    """Decode-spike block size that fits the per-block live set in ``memory_budget``.

    Per decode-spike column the live set is dominated by ``K`` (``n_enc``), the
    spectral projection (``rank``), and ~4 ``n_bins``-scaled arrays (``D``, ``P``,
    ``lc`` and the clip/rescale temporaries), all in the float32 working dtype. A
    safety factor of 2 leaves headroom for XLA workspace. The fitting size is
    rounded down to a power of two so electrodes with nearby encoding counts
    share compiled kernels.
    """
    bytes_per_col = _WORKING_ITEMSIZE * (n_enc + rank + 4 * n_bins)
    raw = max(memory_budget // (bytes_per_col * 2), 1)
    return min(1 << (raw.bit_length() - 1), block_size)


@partial(jax.jit, static_argnames=("position_std",))
def _positive_source_columns(eigvals, eigvecs, labels, sources, *, position_std):
    """Fixed positive, mass-preserving columns of the approximate heat operator.

    Correct each point-source column before mixing fields. Correcting an entire
    mark-dependent field instead is nonlinear and its mark integral does not
    recover the separately diffused ground-process field.
    """
    coefficients = jnp.exp(-(position_std**2 / 2.0) * eigvals)
    columns = jnp.matmul(
        eigvecs,
        coefficients[:, None] * eigvecs[sources].T,
        precision=jax.lax.Precision.HIGHEST,
    )
    columns = jnp.where(labels[:, None] == labels[sources][None, :], columns, 0.0)
    columns = jnp.maximum(columns, 0.0)
    mass = jnp.sum(columns, axis=0)
    return columns / jnp.where(mass > 0.0, mass, 1.0)[None, :]


def _source_tile_size(memory_budget, n_bins, rank):
    """Bound source-column workspace; one column is the minimum useful tile."""
    capacity = max(memory_budget // (8 * _WORKING_ITEMSIZE * (n_bins + rank)), 1)
    return min(256, n_bins, 1 << (capacity.bit_length() - 1))


def _source_cache_budget(memory_budget):
    return min(_SOURCE_CACHE_MAX_BYTES, memory_budget // 4)


def _cached_source_kernel(
    eigvals,
    eigvecs,
    labels,
    position_std,
    memory_budget,
    environment=None,
    *,
    sources=None,
):
    """Resolve requested occupied columns from one bounded environment cache.

    Encoding models keep source indices rather than strong matrix references.
    Related encoding groups share a union of columns when it fits; otherwise
    the current requested support replaces the previous cache. Only the returned
    subset is a transient prediction or fitting buffer.
    """
    n_bins, rank = eigvecs.shape
    sources = (
        np.arange(n_bins, dtype=np.int32)
        if sources is None
        else np.asarray(sources, dtype=np.int32)
    )
    budget = _source_cache_budget(memory_budget)
    if _WORKING_ITEMSIZE * n_bins * sources.size + sources.nbytes > budget:
        if environment is not None and hasattr(
            environment, "_diffusion_source_kernel_"
        ):
            del environment._diffusion_source_kernel_
        return jnp.zeros((n_bins, 0), dtype=jnp.float32)
    cached = getattr(environment, "_diffusion_source_kernel_", None)
    compatible = (
        cached is not None
        and len(cached) == 4
        and cached[0] is eigvecs
        and cached[1] == position_std
        and cached[2].nbytes + cached[3].nbytes <= budget
    )
    cache_sources = sources
    if compatible:
        if np.array_equal(sources, cached[2]):
            return cached[3]
        contained = np.all(np.isin(sources, cached[2], assume_unique=True))
        if contained:
            return cached[3][:, np.searchsorted(cached[2], sources)]
        union = np.union1d(cached[2], sources).astype(np.int32)
        if _WORKING_ITEMSIZE * n_bins * union.size + union.nbytes <= budget:
            cache_sources = union
        else:
            compatible = False
    kernel = np.empty((n_bins, cache_sources.size), dtype=np.float32)
    missing_sources = cache_sources
    if compatible:
        # Preserve already computed columns exactly rather than rebuilding them
        # with a different tile width, which changes float32 GEMM rounding.
        kernel[:, np.searchsorted(cache_sources, cached[2])] = np.asarray(cached[3])
        missing_sources = cache_sources[
            ~np.isin(cache_sources, cached[2], assume_unique=True)
        ]
    tile_size = _source_tile_size(memory_budget, n_bins, rank)
    for start in range(0, missing_sources.size, tile_size):
        block_sources = missing_sources[start : start + tile_size]
        kernel[:, np.searchsorted(cache_sources, block_sources)] = np.asarray(
            _positive_source_columns(
                eigvals,
                eigvecs,
                labels,
                jnp.asarray(block_sources),
                position_std=position_std,
            )
        )
    kernel = jnp.asarray(kernel)
    if environment is not None:
        frozen_sources = cache_sources.copy()
        frozen_sources.setflags(write=False)
        environment._diffusion_source_kernel_ = (
            eigvecs,
            position_std,
            frozen_sources,
            kernel,
        )
    if np.array_equal(cache_sources, sources):
        return kernel
    return kernel[:, np.searchsorted(cache_sources, sources)]


def _prediction_source_operator(model, eigvecs):
    """Resolve the fitted operator cache under the prediction's memory budget."""
    _check_diffusion_grid_revision(model)
    if model.get("spatial_operator") != "positive_source_columns":
        raise ValidationError(
            "Encoding model uses the legacy nonlinear diffusion normalization; "
            "refit the encoding model before decoding."
        )
    memory_budget = int(model["memory_budget"])
    n_bins, rank = eigvecs.shape
    tile_size = _source_tile_size(memory_budget, n_bins, rank)
    kernel = jnp.asarray(model["source_kernel"])
    source_indices = model.get("source_bin_indices")
    if model.get("source_kernel_cache") == "environment" and not kernel.shape[1]:
        eigvals, _, labels, _ = get_device_basis(
            model["environment"], model["resolved_rank"]
        )
        kernel = _cached_source_kernel(
            eigvals,
            eigvecs,
            labels,
            model["position_std"],
            memory_budget,
            model["environment"],
            sources=source_indices,
        )
    support_bytes = 0 if source_indices is None else source_indices.nbytes
    if kernel.size * _WORKING_ITEMSIZE + support_bytes > _source_cache_budget(
        memory_budget
    ):
        kernel = jnp.zeros((n_bins, 0), dtype=jnp.float32)
    # The cached path does not use tiles; keep its executable independent of the
    # workspace budget whenever the same bounded kernel cache fits.
    return kernel, 1 if kernel.shape[1] else tile_size


def _check_diffusion_grid_revision(model):
    if model.get("grid_revision") != getattr(
        model["environment"], "_place_grid_revision_", 0
    ):
        raise ValidationError(
            "encoding model is stale: the environment's physical grid was refit; "
            "refit the encoding model before decoding."
        )


@partial(jax.jit, static_argnames=("position_std", "source_tile_size"))
def _linear_heat_apply(
    eigvals,
    eigvecs,
    labels,
    fields,
    source_kernel,
    *,
    position_std,
    source_tile_size,
    source_bins=None,
):
    """Apply positive columns to fields on the supplied compact source support.

    Column normalization always covers all destination bins, even when a caller
    later selects only an animal's local destination. Empty source support maps
    to a zero field on all destinations.
    """
    if source_kernel.shape[1]:
        return jnp.matmul(source_kernel, fields, precision=jax.lax.Precision.HIGHEST)
    n_bins, n_sources = eigvecs.shape[0], fields.shape[0]
    result = jnp.zeros((n_bins, fields.shape[1]), dtype=fields.dtype)
    if n_sources == 0:
        return result
    if source_bins is None:
        source_bins = jnp.arange(n_sources)
    n_tiles = (n_sources + source_tile_size - 1) // source_tile_size
    padding = n_tiles * source_tile_size - n_sources
    padded_fields = jnp.pad(fields, ((0, padding), (0, 0)))
    padded_sources = jnp.pad(source_bins, (0, padding), mode="edge")

    def add_tile(tile, result):
        first = tile * source_tile_size
        sources = jax.lax.dynamic_slice_in_dim(padded_sources, first, source_tile_size)
        columns = _positive_source_columns(
            eigvals, eigvecs, labels, sources, position_std=position_std
        )
        values = jax.lax.dynamic_slice_in_dim(padded_fields, first, source_tile_size)
        return result + jnp.matmul(columns, values, precision=jax.lax.Precision.HIGHEST)

    return jax.lax.fori_loop(0, n_tiles, add_tile, result)


def _diffuse_field(
    field: np.ndarray,
    Lam: jnp.ndarray,
    Q: jnp.ndarray,
    position_std: float,
    labels: jnp.ndarray,
    n_components: int,
    source_kernel,
    source_tile_size,
    source_bins=None,
) -> np.ndarray:
    """Diffuse a single ``(n_interior,)`` count field, returned as a host array.

    Shared by the occupancy and per-electrode ground-process fields at fit time:
    both promote to a single-column field, apply the same fixed source operator,
    then squeeze back to ``(n_interior,)`` on host.
    """
    if source_bins is not None:
        # Occupancy can have support beyond the retained encoding spike bins.
        # Generate only its occupied columns, without retaining a second cache.
        occupied = np.flatnonzero(field)
        if not np.all(np.isin(occupied, source_bins, assume_unique=True)):
            source_bins = occupied
            source_kernel = jnp.zeros((Q.shape[0], 0), dtype=jnp.float32)
        field = field[source_bins]
    field_hat = _linear_heat_apply(
        Lam,
        Q,
        labels,
        jnp.asarray(field[:, None], dtype=jnp.float32),
        source_kernel,
        position_std=position_std,
        source_tile_size=source_tile_size,
        source_bins=None if source_bins is None else jnp.asarray(source_bins),
    )
    return np.asarray(field_hat)[:, 0]


def fit_clusterless_diffusion_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[jnp.ndarray],
    spike_waveform_features: list[jnp.ndarray],
    environment: Environment,
    *,
    position_std: float = float(np.sqrt(12.5)),
    waveform_std: float = 24.0,
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
        Per-block byte budget driving the effective block size, by default
        536_870_912 (512 MiB).
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
        ``encoding_weights`` (list), ``weight_total`` (list of float),
        ``mean_rates`` (list), ``resolved_rank`` (int), ``node_order``,
        ``bin_sizes`` (dV per interior bin), ``position_std``, ``waveform_std``,
        ``block_size``, ``memory_budget``, ``disable_progress_bar``.
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

    Lam, Q, labels, n_components = get_device_basis(environment, resolved_rank)
    source_tile_size = _source_tile_size(memory_budget, n_interior, resolved_rank)

    weight_sum = float(exposure_weights.sum())  # weighted occupancy seconds

    # ---- occupancy pi = H O / (sum w_pos * dV), floored at EPS ----
    occupancy_field: np.ndarray
    active = exposure_weights > 0
    if np.any(active):
        occupancy_positions = get_position_at_time(
            position_time, position, position_time[active], environment
        )
        occupancy_field = np.bincount(
            _interior_bin_indices(environment, occupancy_positions, full_to_local),
            weights=exposure_weights[active],
            minlength=n_interior,
        )
    else:
        occupancy_field = np.zeros(n_interior)
    # ---- per electrode: marks, bin indices, weights, mean rate, ground process ----
    encoding_bin_indices: list[np.ndarray] = []
    encoding_marks: list[jnp.ndarray] = []
    encoding_weights: list[jnp.ndarray] = []
    encoding_fit_weights: list[np.ndarray] = []
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
        is_in_bounds = support.contains(electrode_spike_times)
        electrode_spike_times = electrode_spike_times[is_in_bounds]
        # Validate only the in-window features that actually enter the fit.
        bounded_features = np.asarray(electrode_features)[is_in_bounds]
        validate_finite(bounded_features, "spike_waveform_features")
        _validate_waveform_std_shape(waveform_std, bounded_features.shape[1], electrode)

        spike_weights = interpolate_weights_at_spike_times(
            electrode_spike_times, position_time, weights, encoding_support=support
        )
        w_total = float(spike_weights.sum())

        spike_positions = get_position_at_time(
            position_time,
            position,
            electrode_spike_times,
            environment,
            encoding_support=support,
        )
        bins = np.asarray(
            _interior_bin_indices(environment, spike_positions, full_to_local)
        )
        # Store encoding spikes sorted by bin: predict sums them by bin with
        # indices_are_sorted=True, which requires nondecreasing ids.
        order = np.argsort(bins, kind="stable")

        encoding_bin_indices.append(bins[order])
        encoding_marks.append(jnp.asarray(np.asarray(bounded_features)[order]))
        encoding_weights.append(jnp.asarray(np.asarray(spike_weights)[order]))
        encoding_fit_weights.append(np.asarray(spike_weights)[order])
        weight_total.append(w_total)

        if w_total == 0:
            # Zero-rate electrode: no fitted density, no ground-process contribution.
            mean_rates.append(0.0)
            warnings.warn(
                f"electrode {electrode} has zero total encoding weight (zero-rate): "
                "no effective encoding spikes. It contributes no ground-process "
                "intensity and its decode spikes use the rate floor in Hz.",
                UserWarning,
                stacklevel=2,
            )
            continue

        mean_rates.append(weighted_mean_rate(spike_weights, weight_sum))

    # Retain every encoding bin, including spikes whose current EM weight is
    # zero. The support therefore remains valid when a refit activates them.
    source_bin_indices = np.unique(
        np.concatenate(encoding_bin_indices)
        if encoding_bin_indices
        else np.empty(0, dtype=np.int32)
    ).astype(np.int32)
    source_bin_indices.setflags(write=False)
    encoding_source_indices = [
        np.searchsorted(source_bin_indices, bins).astype(np.int32)
        for bins in encoding_bin_indices
    ]
    for source_indices in encoding_source_indices:
        source_indices.setflags(write=False)
    source_kernel = _cached_source_kernel(
        Lam,
        Q,
        labels,
        position_std,
        memory_budget,
        environment,
        sources=source_bin_indices,
    )

    occupancy_hat = _diffuse_field(
        occupancy_field,
        Lam,
        Q,
        position_std,
        labels,
        n_components,
        source_kernel,
        source_tile_size,
        source_bin_indices,
    )
    safe_weight_sum = weight_sum if weight_sum > 0 else 1.0
    occupancy = np.clip(occupancy_hat / (safe_weight_sum * bin_sizes), EPS, None)
    if weight_sum == 0:
        warnings.warn(
            "occupancy weights sum to 0; the diffusion occupancy is degenerate and a "
            "safe denominator is used. Every electrode will be zero-rate.",
            UserWarning,
            stacklevel=2,
        )

    for bins, spike_weights, w_total, mean_rate in zip(
        encoding_bin_indices,
        encoding_fit_weights,
        weight_total,
        mean_rates,
        strict=True,
    ):
        if w_total == 0:
            continue
        spike_field = np.bincount(
            bins, weights=np.asarray(spike_weights), minlength=n_interior
        )
        spike_field_hat = _diffuse_field(
            spike_field,
            Lam,
            Q,
            position_std,
            labels,
            n_components,
            source_kernel,
            source_tile_size,
            source_bin_indices,
        )
        p_gpi = spike_field_hat / (w_total * bin_sizes)
        summed_ground_process_intensity += mean_rate * p_gpi / occupancy

    unfloored_ground_process_intensity = summed_ground_process_intensity
    summed_ground_process_intensity = np.clip(
        summed_ground_process_intensity, RATE_EPS_HZ, None
    )

    return {
        "environment": environment,
        "occupancy": occupancy,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "unfloored_ground_process_intensity": unfloored_ground_process_intensity,
        "encoding_bin_indices": encoding_bin_indices,
        "encoding_source_indices": encoding_source_indices,
        "source_bin_indices": source_bin_indices,
        "encoding_marks": encoding_marks,
        "encoding_weights": encoding_weights,
        "weight_total": weight_total,
        "mean_rates": mean_rates,
        "resolved_rank": resolved_rank,
        # The environment owns the only persistent matrix. Keeping this fitted
        # model alive cannot prevent an evicted group's cache from being freed.
        "source_kernel": jnp.zeros((n_interior, 0), dtype=jnp.float32),
        "source_kernel_cache": "environment",
        "spatial_operator": "positive_source_columns",
        "grid_revision": getattr(environment, "_place_grid_revision_", 0),
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


def _padded_encoding(bins, marks, weights, n_bins: int, source_bins=None):
    """Device encoding bins, marks and weights in bin order, padded with dropped rows.

    Predict sums by bin with ``indices_are_sorted=True``. Models fitted before
    encoding spikes were stored bin-sorted are sorted here (stably). Padding
    rows take the dropped bin ``n_bins`` (keeping the bins sorted), zero marks
    and zero weight, so electrodes with nearby encoding counts share compiled
    kernels.

    Returns
    -------
    bins : jnp.ndarray, shape (n_padded,)
    marks : jnp.ndarray, shape (n_padded, n_features)
    weights : jnp.ndarray, shape (n_padded,)
    """
    bins, marks, weights = np.asarray(bins), np.asarray(marks), np.asarray(weights)
    if np.any(bins[1:] < bins[:-1]):
        order = np.argsort(bins, kind="stable")
        bins, marks, weights = bins[order], marks[order], weights[order]
    if source_bins is not None:
        sources = np.asarray(source_bins)
        compact_bins = np.searchsorted(sources, bins)
        if np.any(compact_bins >= sources.size) or np.any(
            sources[compact_bins] != bins
        ):
            raise ValidationError(
                "encoding bin support changed; refit the encoding model before decoding."
            )
        bins = compact_bins.astype(np.int32)
    n_padded = _padded_sample_count(bins.shape[0])
    padded_bins = np.full(n_padded, n_bins, dtype=bins.dtype)
    padded_bins[: bins.shape[0]] = bins
    return (
        jnp.asarray(padded_bins),
        jnp.asarray(_pad_rows(marks, n_padded)),
        jnp.asarray(_pad_rows(weights, n_padded)),
    )


def _diffused_mark_field(
    decode_features,
    encoding_bins,
    encoding_marks,
    encoding_weights,
    waveform_std,
    eigvals,
    eigvecs,
    labels,
    position_std: float,
    n_components: int,
    source_kernel,
    source_tile_size,
    source_bins=None,
):
    """``H_t D_e`` for one block of decoding spikes, shape (n_bins, n_block).

    ``D_e[:, j] = sum_i w_i K_mark(m_j, m_i) onehot(bin(x_i))``; the mark
    kernel's Gaussian normalizer is already in ``kde_distance``.
    """
    mark_kernel = kde_distance(decode_features, encoding_marks, waveform_std)
    D = deterministic_row_sum(
        encoding_weights[:, None] * mark_kernel,
        encoding_bins,
        eigvecs.shape[0] if source_bins is None else source_bins.shape[0],
        indices_are_sorted=True,
    )
    return _linear_heat_apply(
        eigvals,
        eigvecs,
        labels,
        D,
        source_kernel,
        position_std=position_std,
        source_tile_size=source_tile_size,
        source_bins=source_bins,
    )


@partial(
    jax.jit,
    static_argnames=(
        "position_std",
        "n_components",
        "block_size",
        "indices_are_sorted",
        "source_tile_size",
    ),
    donate_argnums=0,
)
def _add_electrode_diffusion_intensities(
    total,
    decode_features,
    row_ids,
    encoding_bins,
    encoding_marks,
    encoding_weights,
    waveform_std,
    eigvals,
    eigvecs,
    labels,
    occupancy,
    bin_sizes,
    mean_rate,
    weight_total,
    source_kernel,
    source_bins=None,
    *,
    position_std,
    n_components,
    block_size,
    indices_are_sorted,
    source_tile_size,
):
    """Add one electrode's non-local log mark intensities to their rows.

    Parameters
    ----------
    total : jnp.ndarray, shape (n_rows, n_bins)
        Running sum over electrodes; donated.
    decode_features : jnp.ndarray, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    encoding_bins, encoding_marks, encoding_weights : jnp.ndarray
        Padded encoding samples from :func:`_padded_encoding`.
    waveform_std : jnp.ndarray, shape (n_features,)
    eigvals, eigvecs, labels : jnp.ndarray
        Device heat-kernel basis and component labels.
    occupancy, bin_sizes : jnp.ndarray, shape (n_bins,)
    mean_rate, weight_total : float
    position_std : float
    n_components : int
    block_size : int
        Decoding spikes per diffused block.
    indices_are_sorted : bool

    Returns
    -------
    total : jnp.ndarray, shape (n_rows, n_bins)
    """

    def add_block(block, total):
        start = block * block_size
        P = _diffused_mark_field(
            jax.lax.dynamic_slice_in_dim(decode_features, start, block_size),
            encoding_bins,
            encoding_marks,
            encoding_weights,
            waveform_std,
            eigvals,
            eigvecs,
            labels,
            position_std,
            n_components,
            source_kernel,
            source_tile_size,
            source_bins,
        )
        # bin_sizes (dV) makes p_e a proper joint density (recovers the mark
        # marginal, guarded by test_mark_marginal_recovery). It cancels in the
        # p_e / occupancy ratio below wherever occupancy carries the same dV
        # factor -- i.e. every bin except those where occupancy hit its EPS floor,
        # which carry unreliable near-zero evidence anyway.
        p_e = P / (weight_total * bin_sizes[:, None])
        log_intensity = safe_log(
            mean_rate * p_e / occupancy[:, None], eps=RATE_EPS_HZ
        )  # (n_bins, n_block)
        return total + deterministic_row_sum(
            log_intensity.T,
            jax.lax.dynamic_slice_in_dim(row_ids, start, block_size),
            total.shape[0],
            indices_are_sorted=indices_are_sorted,
        )

    n_blocks = decode_features.shape[0] // block_size
    return jax.lax.fori_loop(0, n_blocks, add_block, total)


@partial(
    jax.jit,
    static_argnames=(
        "position_std",
        "n_components",
        "block_size",
        "indices_are_sorted",
        "source_tile_size",
    ),
    donate_argnums=0,
)
def _add_electrode_local_diffusion_terms(
    total,
    decode_features,
    row_ids,
    spike_bins,
    encoding_bins,
    encoding_marks,
    encoding_weights,
    waveform_std,
    eigvals,
    eigvecs,
    labels,
    occupancy,
    bin_sizes,
    mean_rate,
    weight_total,
    source_kernel,
    source_bins=None,
    *,
    position_std,
    n_components,
    block_size,
    indices_are_sorted,
    source_tile_size,
):
    """Add one electrode's local log mark intensities to their rows.

    The same fixed positive source operator is used, so the local
    value is read from the same smoothed column as the non-local one, at the
    animal's interior bin.

    Parameters
    ----------
    total : jnp.ndarray, shape (n_rows,)
        Running sum over electrodes; donated.
    decode_features : jnp.ndarray, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jnp.ndarray, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    spike_bins : jnp.ndarray, shape (n_padded_spikes,)
        Animal's interior bin at each decoding spike; padding rows are zero.
    encoding_bins, encoding_marks, encoding_weights, waveform_std, eigvals,
    eigvecs, labels, occupancy, bin_sizes, mean_rate, weight_total,
    position_std, n_components, block_size, indices_are_sorted
        As in :func:`_add_electrode_diffusion_intensities`.

    Returns
    -------
    total : jnp.ndarray, shape (n_rows,)
    """

    def add_block(block, total):
        start = block * block_size
        bins = jax.lax.dynamic_slice_in_dim(spike_bins, start, block_size)
        P = _diffused_mark_field(
            jax.lax.dynamic_slice_in_dim(decode_features, start, block_size),
            encoding_bins,
            encoding_marks,
            encoding_weights,
            waveform_std,
            eigvals,
            eigvecs,
            labels,
            position_std,
            n_components,
            source_kernel,
            source_tile_size,
            source_bins,
        )
        p_e_at_animal = P[bins, jnp.arange(block_size)] / (
            weight_total * bin_sizes[bins]
        )
        log_intensity = safe_log(
            mean_rate * p_e_at_animal / occupancy[bins], eps=RATE_EPS_HZ
        )  # (n_block,)
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
        bin; the ``D_e`` construction and source operator are identical to
        the non-local path -- only the readout differs). By default False.
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
    **encoding_model
        The dict returned by :func:`fit_clusterless_diffusion_encoding_model`.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (n_rows, n_interior_bins) if ``is_local`` is
        False, else (n_rows, 1). ``n_rows`` is ``n_bins`` unless ``row_slice``
        is given.
    """
    if rate_units != "Hz":
        raise ValidationError(
            "Encoding rates must be in Hz; refit legacy encoding models before decoding."
        )
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

    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        encoding_bin_indices=encoding_bin_indices,  # type: ignore[arg-type]
        encoding_marks=encoding_marks,  # type: ignore[arg-type]
        encoding_weights=encoding_weights,  # type: ignore[arg-type]
        weight_total=weight_total,  # type: ignore[arg-type]
        mean_rates=mean_rates,  # type: ignore[arg-type]
    )
    _time_grid = _resolve_time_grid(time_edges, _time_grid)
    time_edges = _time_grid.edges
    if _spike_time_order is None:
        _spike_time_order = _SpikeTimeOrder()
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_bins = occupancy.shape[0]

    # The encoding model is grid-bound: its occupancy, node_order, per-spike bin
    # indices, and ground-process field all describe the interior-bin layout at fit
    # time. If the shared Environment was refit (fit_place_grid) to a different grid
    # afterwards, those stale arrays no longer match the environment's current graph
    # -- and when the interior-bin count happens to be unchanged, the shape checks in
    # the diffusion matmul do not catch it, silently misattributing the likelihood to
    # the wrong bins. Compare the fit-time node_order against the environment's
    # current one and fail loudly instead. Re-fit the encoding model after any refit.
    _check_diffusion_grid_revision(encoding_model)
    current_node_order = environment_graph(environment)[1]
    if not np.array_equal(np.asarray(node_order), np.asarray(current_node_order)):
        raise ValidationError(
            "encoding model is stale: the environment's grid was refit after this "
            "encoding model was fit, so its interior-bin layout no longer matches.",
            expected="an encoding model fit against the environment's current grid",
            got=(
                f"fit-time interior-bin count {np.asarray(node_order).shape[0]} vs "
                f"current {np.asarray(current_node_order).shape[0]}"
            ),
            hint="Re-fit the encoding model (fit_clusterless_diffusion_encoding_model) "
            "against the refit environment before predicting.",
        )

    Lam, Q, labels, n_components = get_device_basis(environment, resolved_rank)

    source_kernel, source_tile_size = _prediction_source_operator(encoding_model, Q)
    source_bin_indices = encoding_model.get("source_bin_indices")
    source_bins = (
        None if source_bin_indices is None else jnp.asarray(source_bin_indices)
    )
    n_sources = n_bins if source_bins is None else source_bins.shape[0]

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
        local_ground_process_intensity = summed_ground_process_intensity[
            animal_time_bins
        ]  # (n_time,)
        log_likelihood = (
            -jnp.asarray(_time_grid.durations(row_start, row_stop))
            * local_ground_process_intensity
        )
        # Kernel outputs are committed to the basis's device; committing the
        # starting sum too lets every electrode reuse one executable.
        log_likelihood = jax.device_put(log_likelihood, Q.sharding)

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
            selection = select_spikes_in_rows(
                electrode_spike_times,
                row_start,
                row_stop,
                time_edges=time_edges,
                _spike_time_order=_spike_time_order,
            )
            electrode_spike_times = select_spike_rows(electrode_spike_times, selection)
            # Validate only the in-window decode features that actually enter the
            # likelihood (mirrors fit's post-clip validation); an out-of-window
            # spike's feature must not raise.
            # Select FIRST, convert after: converting the whole array would
            # cost every decoding spike in the recording on every chunk call
            # (and, for a jax.Array input, retain that host copy on the array).
            decode_features = jnp.asarray(
                select_spike_rows(electrode_decode_features, selection)
            )
            validate_finite(decode_features, "spike_waveform_features")
            n_decode = electrode_spike_times.shape[0]
            if n_decode == 0:
                continue

            # Zero-rate electrode: floor every observed decode spike to LOG_RATE_EPS_HZ,
            # identical contract to the non-local zero-rate guard.
            if electrode_weight_total == 0:
                log_likelihood += sum_spikes_into_rows(
                    jnp.full((n_decode,), LOG_RATE_EPS_HZ), selection
                )
                continue

            enc_bins, enc_marks, enc_weights = _spike_time_order.memo(
                "padded_diffusion_encoding",
                (
                    electrode_bins,
                    electrode_marks,
                    electrode_weights,
                    source_bin_indices,
                ),
                partial(
                    _padded_encoding,
                    electrode_bins,
                    electrode_marks,
                    electrode_weights,
                    n_sources,
                    source_bin_indices,
                ),
            )
            safe_weight_total = (
                electrode_weight_total if electrode_weight_total > 0 else 1.0
            )

            # The animal's interior bin at each decode spike's own time (nearest-bin
            # interpolation; linear is a documented follow-up, not this task).
            position_at_spike_time = get_position_at_time(
                position_time, position, electrode_spike_times, environment
            )
            spike_bins = _interior_bin_indices(
                environment, position_at_spike_time, full_to_local
            )

            effective_block = _effective_block(
                memory_budget, enc_marks.shape[0], resolved_rank, n_bins, block_size
            )
            # Padded spikes (dropped from the row sums) let chunks with nearby
            # spike counts share one compiled kernel.
            n_padded = _padded_spike_count(n_decode, effective_block)
            log_likelihood = _add_electrode_local_diffusion_terms(
                log_likelihood,
                jnp.asarray(_pad_rows(decode_features, n_padded)),
                jnp.asarray(spike_row_ids(selection, n_padded)),
                jnp.asarray(_pad_rows(spike_bins, n_padded)),
                enc_bins,
                enc_marks,
                enc_weights,
                as_std_array(waveform_std, enc_marks.shape[1]),
                Lam,
                Q,
                labels,
                occupancy,
                bin_sizes,
                electrode_mean_rate,
                safe_weight_total,
                source_kernel,
                source_bins,
                position_std=position_std,
                n_components=n_components,
                block_size=min(effective_block, n_padded),
                indices_are_sorted=selection.indices_are_sorted,
                source_tile_size=source_tile_size,
            )

        return (
            log_likelihood
            + log_bin_duration_evidence(
                spike_times,
                time_edges,
                row_slice,
                _spike_time_order,
                _time_grid=_time_grid,
            )
        )[:, None]

    log_likelihood = (
        -jnp.asarray(_time_grid.durations(row_start, row_stop))[:, None]
        * summed_ground_process_intensity[None, :]
    )
    # Committed like the kernel outputs (see the local branch above).
    log_likelihood = jax.device_put(log_likelihood, Q.sharding)

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
        selection = select_spikes_in_rows(
            electrode_spike_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        )
        # Validate only the in-window decode features that actually enter the
        # likelihood (mirrors fit's post-clip validation); an out-of-window spike's
        # feature must not raise.
        # Select FIRST, convert after (see the local branch above).
        decode_features = jnp.asarray(
            select_spike_rows(electrode_decode_features, selection)
        )
        validate_finite(decode_features, "spike_waveform_features")
        n_decode = decode_features.shape[0]
        if n_decode == 0:
            continue

        # Zero-rate electrode: floor every observed decode spike to LOG_RATE_EPS_HZ BEFORE any
        # division (D_e == 0 and P_e / weight_total_e is 0/0). Not skipped. Each
        # time bin's contribution is (spikes in that bin) * LOG_RATE_EPS_HZ, identical across
        # bins -- accumulate it from per-time-bin counts rather than materializing an
        # (n_decode, n_bins) array (which would be gigabytes at millions of spikes).
        if electrode_weight_total == 0:
            spike_counts = sum_spikes_into_rows(
                jnp.ones(n_decode), selection
            )  # (n_rows,)
            log_likelihood += spike_counts[:, None] * LOG_RATE_EPS_HZ
            continue

        enc_bins, enc_marks, enc_weights = _spike_time_order.memo(
            "padded_diffusion_encoding",
            (electrode_bins, electrode_marks, electrode_weights, source_bin_indices),
            partial(
                _padded_encoding,
                electrode_bins,
                electrode_marks,
                electrode_weights,
                n_sources,
                source_bin_indices,
            ),
        )
        # weight_total_e > 0 here; belt-and-suspenders safe denominator.
        safe_weight_total = (
            electrode_weight_total if electrode_weight_total > 0 else 1.0
        )

        effective_block = _effective_block(
            memory_budget, enc_marks.shape[0], resolved_rank, n_bins, block_size
        )
        logger.debug(
            "clusterless_diffusion predict: effective_block=%d "
            "(requested block_size=%d, memory_budget=%d bytes, n_enc=%d, rank=%d, "
            "n_bins=%d)",
            effective_block,
            block_size,
            memory_budget,
            enc_marks.shape[0],
            resolved_rank,
            n_bins,
        )
        # Padded spikes (dropped from the row sums) let chunks with nearby
        # spike counts share one compiled kernel.
        n_padded = _padded_spike_count(n_decode, effective_block)
        log_likelihood = _add_electrode_diffusion_intensities(
            log_likelihood,
            jnp.asarray(_pad_rows(decode_features, n_padded)),
            jnp.asarray(spike_row_ids(selection, n_padded)),
            enc_bins,
            enc_marks,
            enc_weights,
            as_std_array(waveform_std, enc_marks.shape[1]),
            Lam,
            Q,
            labels,
            occupancy,
            bin_sizes,
            electrode_mean_rate,
            safe_weight_total,
            source_kernel,
            source_bins,
            position_std=position_std,
            n_components=n_components,
            block_size=min(effective_block, n_padded),
            indices_are_sorted=selection.indices_are_sorted,
            source_tile_size=source_tile_size,
        )

    return (
        log_likelihood
        + log_bin_duration_evidence(
            spike_times, time_edges, row_slice, _spike_time_order, _time_grid=_time_grid
        )[:, None]
    )
