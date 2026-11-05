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
connected component, cut below ``FIELD_RESOLUTION`` of their peak and
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
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.autonotebook import tqdm

from non_local_detector._warnings import user_stacklevel
from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import DataQualityWarning, ValidationError
from non_local_detector.likelihoods.clusterless_kde import (
    mark_reference_scale as mark_reference_scale,
)
from non_local_detector.likelihoods.clusterless_spikes import (
    ElectrodeSpikes,
    add_background_only,
    add_electrode_spike_terms,
)
from non_local_detector.likelihoods.common import (
    DEFAULT_WAVEFORM_STD,
    EPS,
    LOG_RATE_EPS_HZ,
    SpikeTimeOrder,
    as_array,
    as_arrays,
    as_std_array,
    compute_log_mark_kernel_gemm,
    deterministic_row_sum,
    environment_grid_fingerprint,
    full_to_local_index,
    get_position_at_time,
    interior_bin_indices,
    log_bin_duration_evidence,
    log_density_intensity,
    log_nonnegative,
    masked_logaddexp,
    masked_logsumexp,
    pad_rows,
    padded_sample_count,
    padded_spike_count,
    prepare_decode_rows,
    repair_flagged,
    require_position,
    select_spike_rows,
    spike_row_ids,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_positive_int,
    weighted_mean_rate,
    zero_if_neginf,
)
from non_local_detector.likelihoods.diffusion import (
    FIELD_RESOLUTION,
    cached_eigenbasis,
    cached_heat_kernel_eigenbasis,
    check_smoothing_bandwidth,
    environment_graph,
    get_device_basis,
    heat_kernel_apply,
    resolved_occupancy,
)
from non_local_detector.likelihoods.encoding_model import ClusterlessEncodingModel
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    background_rate,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    decode_bin_centers,
    requires_time_edges,
)

__all__ = [
    "ClusterlessDiffusionEncodingModel",
    "fit_clusterless_diffusion_encoding_model",
    "predict_clusterless_diffusion_log_likelihood",
]


logger = logging.getLogger(__name__)

# float32 working dtype for the diffusion matmul; its item size drives the memory
# policy -- do not hardcode 4 elsewhere.
_WORKING_ITEMSIZE = np.dtype(np.float32).itemsize


def _validate_execution_settings(memory_budget: Any, block_size: Any) -> None:
    """Reject a memory budget (bytes) or block size the fit and prediction cannot use."""
    validate_positive_int("memory_budget", memory_budget)
    validate_positive_int("block_size", block_size)


def _validate_diffusion_params(
    position_std: float,
    waveform_std: float | np.ndarray | jax.Array,
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
        msg = (
            "position_std must be a scalar; the graph heat kernel is isotropic in "
            "graph distance, so per-dimension bandwidths are not supported."
        )
        raise ValidationError(
            msg,
            expected="a scalar position_std",
            got=f"array of shape {position_std_arr.shape}",
        )
    position_std = float(position_std_arr.item())
    if not (position_std > 0.0 and np.isfinite(position_std)):
        msg = "position_std must be a finite positive number"
        raise ValidationError(
            msg,
            expected="0 < position_std < inf",
            got=f"position_std = {position_std}",
        )
    waveform_std_arr = np.asarray(waveform_std, dtype=float)
    if not (np.all(waveform_std_arr > 0.0) and np.all(np.isfinite(waveform_std_arr))):
        msg = "waveform_std must be finite and positive"
        raise ValidationError(
            msg,
            expected="0 < waveform_std < inf",
            got=f"waveform_std = {waveform_std}",
        )

    if heat_kernel_rank is not None:
        # Reject float/bool/zero/negative before it reaches SciPy / cache slicing.
        if isinstance(heat_kernel_rank, bool) or not isinstance(heat_kernel_rank, int):
            msg = "heat_kernel_rank must be None or a positive integer"
            raise ValidationError(
                msg,
                expected="None or a positive int",
                got=f"{heat_kernel_rank!r} (type {type(heat_kernel_rank).__name__})",
            )
        if heat_kernel_rank < 1:
            msg = "heat_kernel_rank must be a positive integer"
            raise ValidationError(
                msg,
                expected="heat_kernel_rank >= 1",
                got=f"heat_kernel_rank = {heat_kernel_rank}",
            )

    _validate_execution_settings(memory_budget, block_size)

    return position_std


def _validate_waveform_std_shape(
    waveform_std: float | np.ndarray | jax.Array, n_features: int, electrode: int
) -> None:
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
        msg = (
            "waveform_std must be a scalar or a 1-D array of length n_features "
            "(matching each electrode's waveform-feature dimension)"
        )
        raise ValidationError(
            msg,
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


class ClusterlessDiffusionEncodingModel(ClusterlessEncodingModel):
    """The encoding model ``fit_clusterless_diffusion_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_clusterless_diffusion_log_likelihood``, documented there.
    """

    occupancy: np.ndarray
    summed_ground_process_intensity: np.ndarray
    encoding_bin_indices: list[np.ndarray]
    encoding_marks: list[jax.Array]
    encoding_weights: list[jax.Array]
    weight_total: list[float]
    resolved_rank: int
    node_order: np.ndarray
    bin_sizes: np.ndarray
    position_std: float
    waveform_std: float | np.ndarray | jax.Array
    block_size: int
    memory_budget: int


def fit_clusterless_diffusion_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    position_std: float = float(np.sqrt(12.5)),
    waveform_std: float | np.ndarray | jax.Array = DEFAULT_WAVEFORM_STD,
    heat_kernel_rank: int | None = None,
    block_size: int = 10000,
    memory_budget: int = 536870912,
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> ClusterlessDiffusionEncodingModel:
    """Fit the clusterless graph-diffusion encoding model.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Time of each position sample.
    position : array_like, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times for each electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Spike waveform features (marks) for each electrode.
    environment : Environment
        The spatial environment (must be fitted).
    weights : array_like, shape (n_time_position,), optional
        Per-sample EM weights, by default None (uniform). Weight the occupancy,
        per-electrode ground-process, mean rate, and the per-spike joint estimator.
    position_std : float, optional
        Heat-kernel smoothing standard deviation (the physical bandwidth) in
        coordinate units, by default ``sqrt(12.5)``. Scalar only: the heat kernel is
        isotropic in graph distance.
    waveform_std : float, optional
        Gaussian smoothing standard deviation for the mark kernel, by default 24.0.
    heat_kernel_rank : int or None, optional
        Eigenbasis truncation rank, by default None (bandwidth-aware auto-selection).
    block_size : int, optional
        Requested cap on the decode-spike block size, by default 10_000. The
        effective block is the memory-budget-resolved value.
    memory_budget : int, optional
        Byte budget, by default 536_870_912 (512 MiB). It sets the decode-spike
        block size and the spatial kernel's tile size (at fit and predict), and
        whether predict builds the whole kernel once (when it fits half the
        budget) or streams it in tiles. At bins where the field is close to
        its float32 resolution floor (kernel values near 4e-7 of the column's
        peak), the log likelihood is set by rounding, so it can differ between
        budgets, and between platforms, by about 1e-4 relative; posteriors
        change by about 5e-6 or less.
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.

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
        Encoding support does not automatically mark decode bins missing.
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support. Direct callers
        omit it and the support is built from the other arguments.

    Returns
    -------
    encoding_model : ClusterlessDiffusionEncodingModel
        Keys consumed by :func:`predict_clusterless_diffusion_log_likelihood`:
        ``algorithm`` (``"clusterless_diffusion"``), ``environment``,
        ``occupancy`` (pi, ``(n_interior,)``),
        ``summed_ground_process_intensity`` (``(n_interior,)``),
        ``encoding_bin_indices`` (list per electrode), ``encoding_marks`` (list),
        ``encoding_weights`` (list): one bin-sorted entry per encoding spike, or
        two when its bracketing tracking samples lie in different bins, each
        with its share of the spike's weight; ``weight_total`` (list of float),
        ``mean_rates`` (``(n_electrodes,)``), ``resolved_rank`` (int), ``node_order``,
        ``bin_sizes`` (dV per interior bin), ``position_std``, ``waveform_std``,
        ``block_size``, ``memory_budget``, ``disable_progress_bar``,
        ``rate_units`` (``"Hz"``), ``encoding_exposure_seconds`` (weighted
        encoding occupancy time in seconds), ``mark_dimensions`` (each
        electrode's waveform-feature count).
    """
    position_time = as_array(position_time)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        encoding_time_range=encoding_time_range,
        valid_position_intervals=valid_position_intervals,
        _encoding_support=_encoding_support,
    )
    n_time_position = position.shape[0]
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
    full_to_local = full_to_local_index(node_order, n_total_bins)

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
        sample_bins[support.indices] = interior_bin_indices(
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
    encoding_marks: list[jax.Array] = []
    encoding_weights: list[jax.Array] = []
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
                DataQualityWarning,
                stacklevel=user_stacklevel(),
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
            np.asarray(log_intensity_scale(occupancy, bin_sizes, mean_rate, w_total))
        )

    return {
        "algorithm": "clusterless_diffusion",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "occupancy": occupancy,
        "summed_ground_process_intensity": summed_ground_process_intensity,
        "encoding_bin_indices": encoding_bin_indices,
        "encoding_marks": encoding_marks,
        "mark_dimensions": [int(np.shape(marks)[1]) for marks in encoding_marks],
        "encoding_weights": encoding_weights,
        "weight_total": weight_total,
        "mean_rates": np.asarray(mean_rates),
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


def _pad_sources(sources: ArrayLike, n_padded: int) -> NDArray[np.int32]:
    """Source bins padded to ``n_padded`` with repeats of the last (bin 0 if none)."""
    sources = np.asarray(sources, dtype=np.int32)
    if sources.size == 0:
        return np.zeros(n_padded, dtype=np.int32)
    return np.pad(sources, (0, n_padded - sources.size), mode="edge")


# The scaled spatial product multiplies exp(log_source - peak) by kernel
# entries; a product below float32's smallest normal number may flush to zero.
_LOG_TINY = float(np.log(np.finfo(np.float32).tiny))


@partial(jax.jit, static_argnames=("position_std", "n_components"))
def _kernel_columns(
    eigvals: jax.Array,
    eigvecs: jax.Array,
    labels: jax.Array,
    sources: jax.Array,
    *,
    position_std: float,
    n_components: int,
) -> jax.Array:
    """Diffusion of a unit mass from each source bin, shape (n_bins, n_sources).

    Column ``k`` is the heat kernel applied to bin ``sources[k]``: clipped to be
    non-negative, zero outside that bin's connected component, cut below
    ``FIELD_RESOLUTION`` of its peak (heat-kernel noise) and normalized to unit
    mass. Every column depends only on its own source, so the model's one
    spatial kernel ``H`` can be built in tiles of sources. Occupancy, the ground
    process and every spike's marked field are the same linear map ``H`` of
    their sources, so the marked intensity integrates over marks to exactly the
    expected count, and disconnected components stay exactly zero.

    Parameters
    ----------
    eigvals, eigvecs, labels : jax.Array
        Device heat-kernel basis and component labels (``get_device_basis``).
    sources : jax.Array, shape (n_sources,)
        Interior bin indices.
    position_std : float
        Heat-kernel bandwidth in position units.
    n_components : int
        Static number of connected components in the heat-kernel basis.

    Returns
    -------
    columns : jax.Array, shape (n_bins, n_sources)
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
        columns >= FIELD_RESOLUTION * jnp.max(columns, axis=0, keepdims=True),
        columns,
        0.0,
    )
    return columns / jnp.sum(columns, axis=0, keepdims=True)


def spatial_kernel_columns(
    environment: Environment,
    resolved_rank: int,
    position_std: float,
    sources: ArrayLike,
) -> jax.Array:
    """Return the model's spatial kernel ``H[:, sources]``, for a fitted environment.

    Parameters
    ----------
    environment : Environment
        The fitted environment.
    resolved_rank : int
        The fitted model's ``resolved_rank``.
    position_std : float
        The fitted model's heat-kernel smoothing standard deviation, in
        coordinate units.
    sources : array_like of int, shape (n_sources,)
        Interior bin indices.

    Returns
    -------
    columns : jax.Array, shape (n_bins, n_sources)
        Non-negative columns that each sum to one: the diffusion of a unit mass
        from each source bin, zero outside its connected component and below
        ``FIELD_RESOLUTION`` of its peak. ``resolved_rank`` and
        ``position_std`` must be the fitted model's.
    """
    eigvals, eigvecs, labels, n_components = get_device_basis(
        environment, resolved_rank
    )
    sources = np.asarray(sources, dtype=np.int32)
    # Pad to few sizes so callers with nearby source counts share one executable.
    padded = _pad_sources(sources, padded_sample_count(max(sources.size, 1)))
    return cast(
        jax.Array,
        _kernel_columns(
            eigvals,
            eigvecs,
            labels,
            jnp.asarray(padded),
            position_std=float(position_std),
            n_components=n_components,
        )[:, : sources.size],
    )


def _dense_kernel_fits(n_bins: int, memory_budget: int) -> bool:
    """Whether the whole ``(n_bins, n_bins)`` kernel fits half of ``memory_budget``."""
    return 2 * _WORKING_ITEMSIZE * n_bins * n_bins <= memory_budget


def _dense_sources(n_bins: int, tile_budget: int) -> tuple[NDArray[np.int32], int]:
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
    eigvals: jax.Array,
    eigvecs: jax.Array,
    labels: jax.Array,
    sources: jax.Array,
    *,
    tile: int,
    position_std: float,
    n_components: int,
) -> jax.Array:
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
    """Return the kernel columns per tile.

    A tile and its spectral temporaries fit half of ``memory_budget``. The
    count is a power of two, for shared compiled kernels.
    """
    raw = max(memory_budget // (2 * _WORKING_ITEMSIZE * (3 * n_bins + rank)), 1)
    return 1 << (raw.bit_length() - 1)


def _diffuse_fields(
    fields: np.ndarray,
    environment: Environment,
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


def log_intensity_scale(
    occupancy: jax.Array | np.ndarray,
    bin_sizes: jax.Array | np.ndarray,
    mean_rate: jax.Array | float,
    weight_total: jax.Array | float,
) -> jax.Array:
    """Per-bin log of ``mean_rate / (sum_i w_i * dV(x) * occupancy(x))``.

    Multiplying a diffused source field by this gives the intensity. Bins
    whose occupancy the heat kernel does not resolve (see
    :func:`resolved_occupancy`) carry no encoding information: their scale is
    ``-inf``, so their marked and ground intensities are zero and their spikes
    are explained by the background alone. Shared by decoding and model
    checking.

    Parameters
    ----------
    occupancy, bin_sizes : array, shape (n_bins,)
        Occupancy and size of each place bin.
    mean_rate, weight_total : float
        Mean rate and total encoding weight.

    Returns
    -------
    log_scale : array, shape (n_bins,)
    """
    occupancy = jnp.asarray(occupancy)
    return jnp.where(
        resolved_occupancy(occupancy),
        log_density_intensity(
            mean_rate,
            -jnp.log(weight_total * jnp.asarray(bin_sizes)),
            jnp.log(occupancy),
        ),
        -jnp.inf,
    )


def _source_layout(
    bins: np.ndarray, n_bins: int, tile_budget: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], int]:
    """Source bins of padded, bin-sorted encoding samples, padded to few sizes.

    Parameters
    ----------
    bins : np.ndarray, shape (n_padded_samples,)
        Bin of each encoding sample; padding samples use ``n_bins``.
    n_bins : int
        Number of bins; the bin index used for padding samples.
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
    n_padded = padded_sample_count(max(distinct.size, 1))
    tile = min(tile_budget, n_padded)
    n_padded = -(-n_padded // tile) * tile
    source_index = np.full(bins.size, n_padded - 1, dtype=np.int32)
    source_index[real] = inverse
    return _pad_sources(distinct, n_padded), source_index, tile


def _electrode_layout(
    bins: ArrayLike,
    marks: ArrayLike,
    weights: ArrayLike,
    n_bins: int,
    tile_budget: int,
    *,
    dense: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """Return padded, bin-sorted encoding samples with their source layout, for one electrode.

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
    marks : jax.Array, shape (n_padded_samples, n_features)
    weights : jax.Array, shape (n_padded_samples,)
        Zero for padding samples.
    sources : jax.Array, shape (n_padded_sources,)
        Source bins, a whole number of tiles long.
    source_index : jax.Array, shape (n_padded_samples,)
        Each sample's position in ``sources``, ascending.
    tile : int
        Kernel columns per tile.
    """
    bins, marks, weights = np.asarray(bins), np.asarray(marks), np.asarray(weights)
    if np.any(bins[1:] < bins[:-1]):
        order = np.argsort(bins, kind="stable")
        bins, marks, weights = bins[order], marks[order], weights[order]
    n_padded = padded_sample_count(bins.shape[0])
    padded_bins = np.full(n_padded, n_bins, dtype=bins.dtype)
    padded_bins[: bins.shape[0]] = bins
    if dense:
        sources, tile = _dense_sources(n_bins, tile_budget)
        source_index = np.where(padded_bins < n_bins, padded_bins, sources.size - 1)
    else:
        sources, source_index, tile = _source_layout(padded_bins, n_bins, tile_budget)
    return (
        jnp.asarray(pad_rows(marks, n_padded)),
        jnp.asarray(pad_rows(weights, n_padded)),
        jnp.asarray(sources),
        jnp.asarray(source_index, dtype=jnp.int32),
        tile,
    )


def _log_source_marks(
    decode_features: jax.Array,
    encoding_marks: jax.Array,
    encoding_weights: jax.Array,
    source_index: jax.Array,
    waveform_std: jax.Array,
    n_sources: int,
) -> jax.Array:
    """Log of ``sum_i w_i K_mark(m_j, m_i)`` over each source bin's samples.

    Reduced in log space within each source, so no source is lost however far
    its marks lie from another's.

    Returns
    -------
    log_source : jax.Array, shape (n_sources, n_block)
        ``-inf`` for a source without positively weighted samples.
    """
    weighted = (
        compute_log_mark_kernel_gemm(decode_features, encoding_marks, waveform_std)
        + log_nonnegative(encoding_weights)[:, None]
    )  # (n_samples, n_block)
    peak = jax.ops.segment_max(
        weighted, source_index, num_segments=n_sources, indices_are_sorted=True
    )
    safe_peak = zero_if_neginf(peak)
    total = deterministic_row_sum(
        jnp.exp(weighted - safe_peak[source_index]),
        source_index,
        n_sources,
        indices_are_sorted=True,
    )
    return jnp.where(jnp.isneginf(peak), -jnp.inf, log_nonnegative(total) + peak)


def _tile_columns(
    basis: tuple[jax.Array, jax.Array, jax.Array],
    sources: jax.Array,
    cached: jax.Array | None,
    start: jax.Array | int,
    *,
    position_std: float,
    n_components: int,
    tile: int,
) -> jax.Array:
    """Kernel columns of ``sources[start:start + tile]``, shape (n_bins, tile).

    Sliced from ``cached`` (columns aligned with ``sources``) when given, else
    built from the heat-kernel basis.
    """
    if cached is not None:
        return jax.lax.dynamic_slice_in_dim(cached, start, tile, axis=1)
    eigvals, eigvecs, labels = basis
    return cast(
        jax.Array,
        _kernel_columns(
            eigvals,
            eigvecs,
            labels,
            jax.lax.dynamic_slice_in_dim(sources, start, tile),
            position_std=position_std,
            n_components=n_components,
        ),
    )


@partial(jax.jit, static_argnames=("tile", "position_std", "n_components"))
def _kernel_bounds(
    eigvals: jax.Array,
    eigvecs: jax.Array,
    labels: jax.Array,
    kernel: jax.Array | None,
    sources: jax.Array,
    log_scale: jax.Array,
    *,
    tile: int,
    position_std: float,
    n_components: int,
) -> tuple[jax.Array, jax.Array]:
    """Per-source bounds for the exact-repair gate, one tile of columns at a time.

    Computed once per prediction for every bin, with the scale of a unit rate
    and weight; an electrode's gain adds its ``log(mean_rate / weight_total)``.

    Parameters
    ----------
    eigvals, eigvecs, labels : jax.Array
        Device heat-kernel basis and component labels.
    kernel : jax.Array, shape (n_bins, n_sources), or None
        The kernel columns of ``sources`` when already built.
    sources : jax.Array, shape (n_sources,)
        A whole number of tiles long.
    log_scale : jax.Array, shape (n_bins,)
        :func:`log_intensity_scale` for a unit rate and weight.
    tile : int
        Kernel columns per tile.
    position_std : float
        Heat-kernel bandwidth in position units.
    n_components : int
        Static number of connected components in the heat-kernel basis.

    Returns
    -------
    gain : jax.Array, shape (n_sources,)
        Largest ``log_scale(x) + log H[x, s]`` over bins: how much a source can
        add to any bin's log intensity per unit of its mark density.
    floor : jax.Array, shape (n_sources,)
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

    def tile_bounds(start: jax.Array) -> tuple[jax.Array, jax.Array]:
        columns = columns_of(start)
        log_columns = log_nonnegative(columns)
        return (
            jnp.max(log_scale[:, None] + log_columns, axis=0),
            jnp.min(jnp.where(columns > 0, log_columns, jnp.inf), axis=0),
        )

    starts = jnp.arange(sources.shape[0] // tile) * tile
    gain, floor = jax.lax.map(tile_bounds, starts)
    return cast(jax.Array, gain.reshape(-1)), cast(jax.Array, floor.reshape(-1))


def _log_marked_field(
    log_source: jax.Array,
    rows: jax.Array | None,
    columns_of: Callable[[jax.Array | int], jax.Array],
    n_tiles: int,
    tile: int,
    bounds: tuple[jax.Array, jax.Array],
    log_background: jax.Array,
) -> jax.Array:
    """``log(H D)`` for one block of spikes, at every bin or at each spike's own.

    The scaled product ``H exp(log_source - peak)`` is applied in tiles of
    kernel columns. A source's scaled mark density times its smallest kernel
    entry may fall below float32's smallest normal number and flush to zero; if
    such a source could reach the spike's background floor at any bin, that
    spike is recomputed by an exact log-space reduction over the sources.

    Parameters
    ----------
    log_source : jax.Array, shape (n_sources, n_block)
        Log of each source's unscaled mark density for each spike
        (``log sum_i w_i K_mark``); the peak scaling happens here.
    rows : jax.Array, shape (n_block,), or None
        Each spike's own bin (local), or None to read every bin (non-local).
    columns_of : callable
        ``columns_of(start)`` gives the kernel columns of sources
        ``start:start + tile``, shape (n_bins, tile).
    n_tiles, tile : int
        Number of tiles of kernel columns and columns per tile.
    bounds : tuple of jax.Array, each shape (n_sources,)
        ``(gain, floor)`` of each source (see :func:`_kernel_bounds`).
    log_background : jax.Array, shape (n_block,)
        Log background intensity of each spike.

    Returns
    -------
    log_field : jax.Array, shape (n_bins, n_block), or (n_block,) when local
    """
    local = rows is not None
    peak = jnp.max(log_source, axis=0)
    safe_peak = zero_if_neginf(peak)
    scaled = jnp.exp(log_source - safe_peak[None, :])
    n_bins = jax.eval_shape(columns_of, 0).shape[0]

    def read(columns: jax.Array, at: jax.Array | None) -> jax.Array:
        return columns if at is None else columns[at]

    def add_tile(index: jax.Array, field: jax.Array) -> jax.Array:
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

    def exact(j: jax.Array, at: jax.Array | None) -> jax.Array:
        # log sum_s H[at, s] exp(log_source[s, j]), reduced in log space.
        def add_exact_tile(index: jax.Array, value: jax.Array) -> jax.Array:
            start = index * tile
            log_columns = log_nonnegative(read(columns_of(start), at))
            part = jax.lax.dynamic_slice_in_dim(log_source[:, j], start, tile)
            terms = jnp.where(
                jnp.isneginf(part)[None, :], -jnp.inf, log_columns + part[None, :]
            )
            return masked_logaddexp(value, masked_logsumexp(terms.T))

        n_read = n_bins if at is None else at.shape[0]
        return cast(
            jax.Array,
            jax.lax.fori_loop(
                0, n_tiles, add_exact_tile, jnp.full((n_read,), -jnp.inf)
            ),
        )

    def repaired(values: jax.Array, j: jax.Array, flag: jax.Array) -> jax.Array:
        if local:
            fixed = jax.lax.cond(
                flag,
                lambda: exact(j, cast(jax.Array, rows)[j][None])[0],
                lambda: values[j],
            )
            return values.at[j].set(fixed)
        fixed = jax.lax.cond(flag, lambda: exact(j, None), lambda: values[:, j])
        return values.at[:, j].set(fixed)

    # Flagged spikes are rare (none of 75,458 on a real recording). Each one
    # reduces every kernel tile in log space; streamed tiles are rebuilt from
    # the basis per spike, so many flags in one block are slow but exact.
    return repair_flagged(log_field, needs_exact, repaired)


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
    total: jax.Array,
    decode_features: jax.Array,
    row_ids: jax.Array,
    spike_bins: jax.Array | None,
    encoding_marks: jax.Array,
    encoding_weights: jax.Array,
    source_index: jax.Array,
    sources: jax.Array,
    waveform_std: jax.Array,
    eigvals: jax.Array,
    eigvecs: jax.Array,
    labels: jax.Array,
    kernel: jax.Array | None,
    log_scale: jax.Array,
    bounds: tuple[jax.Array, jax.Array],
    log_background: jax.Array,
    *,
    position_std: float,
    n_components: int,
    tile: int,
    block_size: int,
    indices_are_sorted: bool,
) -> jax.Array:
    """Add one electrode's log mark intensities to their rows.

    Non-local intensities cover every bin; local ones read the same kernel
    columns at the animal's interior bin.

    Parameters
    ----------
    total : jax.Array, shape (n_rows, n_bins), or (n_rows,) when local
        Running sum over electrodes; donated.
    decode_features : jax.Array, shape (n_padded_spikes, n_features)
        A multiple of ``block_size`` rows; padding rows are zero.
    row_ids : jax.Array, shape (n_padded_spikes,)
        Local rows; padding uses ``n_rows``, which the row sum drops.
    spike_bins : jax.Array, shape (n_padded_spikes,), or None
        Animal's interior bin at each decoding spike (padding rows are zero)
        for local intensities; None for non-local.
    encoding_marks, encoding_weights, source_index, sources : jax.Array
        From :func:`_electrode_layout`.
    waveform_std : jax.Array, shape (n_features,)
        Waveform-feature kernel bandwidths.
    eigvals, eigvecs, labels : jax.Array
        Device heat-kernel basis and component labels.
    kernel : jax.Array, shape (n_bins, n_sources), or None
        The whole spatial kernel when it fits the memory budget
        (:func:`_dense_kernel`); then ``sources`` are every bin. Otherwise
        kernel columns are built from the basis, tile by tile.
    log_scale : jax.Array, shape (n_bins,)
        From :func:`log_intensity_scale`.
    bounds : tuple of jax.Array, each shape (n_sources,)
        This electrode's ``(gain, floor)`` of ``sources`` (see
        :func:`_kernel_bounds`).
    log_background : jax.Array, shape (n_padded_spikes,)
        Log background intensity of each spike, in Hz.
    position_std : float
        Heat-kernel bandwidth in position units.
    n_components : int
        Static number of connected components in the heat-kernel basis.
    tile : int
        Kernel columns per tile (divides ``sources``' length).
    block_size : int
        Decoding spikes per block.
    indices_are_sorted : bool
        Whether ``row_ids`` are sorted in nondecreasing order.

    Returns
    -------
    total : jax.Array, shape (n_rows, n_bins), or (n_rows,) when local
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

    def add_block(block: jax.Array, total: jax.Array) -> jax.Array:
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
    return cast(jax.Array, jax.lax.fori_loop(0, n_blocks, add_block, total))


@requires_time_edges
def predict_clusterless_diffusion_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    spike_waveform_features: Sequence[ArrayLike],
    *,
    environment: Environment,
    occupancy: np.ndarray | jax.Array,
    summed_ground_process_intensity: np.ndarray | jax.Array,
    encoding_bin_indices: list[np.ndarray],
    encoding_marks: list[jax.Array],
    encoding_weights: list[jax.Array],
    weight_total: list[float],
    mean_rates: np.ndarray,
    resolved_rank: int,
    node_order: np.ndarray,
    bin_sizes: np.ndarray | jax.Array,
    position_std: float,
    waveform_std: float | np.ndarray | jax.Array,
    block_size: int,
    memory_budget: int,
    disable_progress_bar: bool = False,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    mark_dimensions: list[int] | None = None,  # noqa: ARG001 -- read by the detector
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    is_local: bool = False,
    row_slice: slice | None = None,
    mark_reference: list[MarkReference | None],
    environment_grid: dict[str, Any] | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Predict the clusterless graph-diffusion log likelihood.

    Parameters
    ----------
    position_time : array_like or None, shape (n_time_position,)
        Time of each position sample (used only by the local path; accepted for
        signature parity when ``is_local`` is False).
        May be None when ``is_local`` is False.
    position : array_like or None, shape (n_time_position, n_position_dims)
        Position samples (used only by the local path; accepted for signature
        parity when ``is_local`` is False).
        May be None when ``is_local`` is False.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Decode spike times for each electrode.
    spike_waveform_features : sequence of array_like, each shape (n_spikes, n_features)
        Decode spike waveform features (marks) for each electrode.
    environment : Environment
        The fitted environment.
    occupancy : np.ndarray, shape (n_interior_bins,)
        Diffused occupancy density at the interior bins.
    summed_ground_process_intensity : np.ndarray, shape (n_interior_bins,)
        Sum of the electrodes' ground intensities in Hz, without the
        background rate.
    encoding_bin_indices : list of np.ndarray, length n_electrodes
        Interior bin of each encoding entry, sorted by bin.
    encoding_marks : list of jax.Array, length n_electrodes
        Marks of each encoding entry, shape (n_entries, n_features).
    encoding_weights : list of jax.Array, length n_electrodes
        Weight of each encoding entry.
    weight_total : list of float, length n_electrodes
        Total encoding weight of each electrode.
    mean_rates : np.ndarray, shape (n_electrodes,)
        Mean rate of each electrode in Hz.
    resolved_rank : int
        Eigenbasis rank of the heat kernel.
    node_order : np.ndarray, shape (n_interior_bins,)
        Interior flat-bin indices in graph order.
    bin_sizes : np.ndarray, shape (n_interior_bins,)
        Volume of each interior bin.
    position_std : float
        Heat-kernel bandwidth in position units.
    waveform_std : float or array
        Mark-kernel bandwidth, scalar or per feature.
    block_size : int
        Requested cap on the decode-spike block size.
    memory_budget : int
        Byte budget for the spike blocks and kernel tiles.
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.
    time_edges : np.ndarray, shape (n_bins + 1,)
        Decoding bin edges.
    algorithm : str, optional
        The algorithm the fit recorded. Accepted so the encoding dictionary
        can be passed as keywords; unused.
    mark_dimensions : list of int, optional
        Each electrode's fitted waveform-feature count, which the detector
        checks the decoded features against; unused here.
    rate_units : str, default="Hz", keyword-only
        Rate units recorded by the encoding fit; anything other than ``"Hz"``
        raises ``ValidationError`` (refit legacy encoding models).
    encoding_exposure_seconds : float, optional, keyword-only
        Weighted encoding occupancy time in seconds, recorded by the fit.
        Accepted so the encoding dictionary can be passed as keywords; unused.
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
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.
    _time_grid : _DecodeTimeGrid, optional
        Internal prepared decode grid shared across a detector's states and
        chunks; direct callers omit it.

    Returns
    -------
    log_likelihood : jax.Array, shape (n_rows, n_interior_bins) if ``is_local`` is
        False, else (n_rows, 1). ``n_rows`` is ``n_bins`` unless ``row_slice``
        is given.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    _validate_execution_settings(memory_budget, block_size)
    spike_times = as_arrays(spike_times)
    spike_waveform_features = as_arrays(spike_waveform_features)
    fitted_occupancy = occupancy
    occupancy = jnp.asarray(occupancy)
    summed_ground_process_intensity = jnp.asarray(
        summed_ground_process_intensity
    ) + background_rate(len(spike_times))
    resolved_rank = int(resolved_rank)
    node_order = np.asarray(node_order)
    bin_sizes = jnp.asarray(bin_sizes)
    position_std = float(position_std)
    block_size = int(block_size)
    memory_budget = int(memory_budget)
    disable_progress_bar = bool(disable_progress_bar)

    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        encoding_bin_indices=encoding_bin_indices,
        encoding_marks=encoding_marks,
        encoding_weights=encoding_weights,
        weight_total=weight_total,
        mean_rates=mean_rates,
        mark_reference=mark_reference,
    )
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    n_bins = occupancy.shape[0]

    eigvals, eigvecs, labels, n_components = get_device_basis(
        environment, resolved_rank
    )
    tile_budget = _source_tile_size(memory_budget, n_bins, resolved_rank)
    dense = _dense_kernel_fits(n_bins, memory_budget)
    every_bin_host, every_bin_tile = _dense_sources(n_bins, tile_budget)
    every_bin = jnp.asarray(every_bin_host)
    basis_static: dict[str, Any] = {
        "position_std": position_std,
        "n_components": n_components,
    }
    # One kernel per prediction, shared by every electrode, state and chunk.
    kernel = (
        rows.spike_time_order.memo(
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
    kernel_gain, kernel_floor = rows.spike_time_order.memo(
        f"diffusion_kernel_bounds:{position_std!r}:{every_bin_tile}:{dense}",
        (eigvecs, labels, fitted_occupancy),
        lambda: _kernel_bounds(
            eigvals,
            eigvecs,
            labels,
            kernel,
            every_bin,
            log_intensity_scale(occupancy, bin_sizes, 1.0, 1.0),
            tile=every_bin_tile,
            **basis_static,
        ),
    )

    durations = jnp.asarray(rows.durations())
    if is_local:
        # The local path reads the animal's position; non-local ignores it and the
        # base API permits position=None (as clusterless_kde does), so position is
        # only required and validated here. add_electrode below reads these
        # names on this path only.
        local_position_time, local_position = require_position(
            position_time, position, "local clusterless diffusion likelihood"
        )
        validate_finite(local_position, "position")
        # Reconstruct the interior-bin mapping from node_order (derived here at
        # predict, not stored) and evaluate the diffused column at the animal's nearest
        # interior bin instead of returning the whole column.
        n_total_bins = environment.is_track_interior_.ravel().shape[0]
        full_to_local = full_to_local_index(node_order, n_total_bins)

        interpolated_position = get_position_at_time(
            local_position_time,
            local_position,
            decode_bin_centers(rows.time_edges, rows.start, rows.stop),
            environment,
        )
        animal_time_bins = interior_bin_indices(
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
    log_likelihood = cast(jax.Array, jax.device_put(log_likelihood, eigvecs.sharding))

    def add_electrode(
        log_likelihood: jax.Array,
        spikes: ElectrodeSpikes,
        model: tuple[Any, Any, Any, Any, Any],
    ) -> jax.Array:
        (
            electrode_bins,
            electrode_marks,
            electrode_weights,
            electrode_weight_total,
            electrode_mean_rate,
        ) = model
        # Zero-rate electrode: the background explains every observed spike,
        # BEFORE any division (D_e == 0 and P_e / weight_total_e is 0/0). Its
        # contribution is identical across bins -- accumulate it per time bin
        # rather than materializing an (n_decode, n_bins) array.
        if electrode_weight_total == 0:
            return add_background_only(log_likelihood, spikes)

        enc_marks, enc_weights, sources, source_index, tile = (
            rows.spike_time_order.memo(
                f"diffusion_electrode_layout:{n_bins}:{tile_budget}:{dense}",
                (electrode_bins, electrode_marks, electrode_weights),
                partial(
                    _electrode_layout,
                    electrode_bins,
                    electrode_marks,
                    electrode_weights,
                    n_bins,
                    tile_budget,
                    dense=dense,
                ),
            )
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
        n_padded = padded_spike_count(spikes.n_spikes, effective_block)
        spike_bins = None
        if is_local:
            # The animal's interior bin at each decode spike's own time
            # (nearest-bin interpolation).
            spike_bins = jnp.asarray(
                pad_rows(
                    interior_bin_indices(
                        environment,
                        get_position_at_time(
                            local_position_time,
                            local_position,
                            select_spike_rows(spikes.times, spikes.selection),
                            environment,
                        ),
                        full_to_local,
                    ),
                    n_padded,
                )
            )
        rate_offset = float(np.log(electrode_mean_rate / electrode_weight_total))
        return cast(
            jax.Array,
            _add_electrode_diffusion_terms(
                log_likelihood,
                jnp.asarray(pad_rows(spikes.marks, n_padded)),
                jnp.asarray(spike_row_ids(spikes.selection, n_padded)),
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
                log_intensity_scale(
                    occupancy, bin_sizes, electrode_mean_rate, electrode_weight_total
                ),
                (kernel_gain[sources] + rate_offset, kernel_floor[sources]),
                jnp.asarray(pad_rows(spikes.log_background, n_padded)),
                tile=tile,
                block_size=min(effective_block, n_padded),
                indices_are_sorted=spikes.selection.indices_are_sorted,
                **basis_static,
            ),
        )

    log_likelihood = add_electrode_spike_terms(
        log_likelihood,
        spike_times,
        spike_waveform_features,
        mark_reference,
        list(
            zip(
                encoding_bin_indices,
                encoding_marks,
                encoding_weights,
                weight_total,
                mean_rates,
                strict=True,
            )
        ),
        add_electrode,
        rows=rows,
        log_background_floor=LOG_RATE_EPS_HZ,
        desc="Local Likelihood" if is_local else "Position-Bin Likelihood",
        disable_progress_bar=disable_progress_bar,
    )

    evidence = log_bin_duration_evidence(
        spike_times,
        rows.time_edges,
        row_slice,
        rows.spike_time_order,
        _time_grid=rows.time_grid,
    )
    if is_local:
        return (log_likelihood + evidence)[:, None]
    return log_likelihood + evidence[:, None]
