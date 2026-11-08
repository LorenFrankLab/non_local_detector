"""Graph-diffusion encoding/decoding for sorted spikes.

A model-level drop-in for :mod:`sorted_spikes_kde`: it estimates the same Poisson
place fields ``rate(x) = mean_rate * spike_density(x) / occupancy_density(x)`` and
uses the same log-likelihood, but the spatial densities are smoothed on the
environment's manifold graph with a heat kernel ``exp(-t L)`` (``t = position_std**2
/ 2``) instead of a Gaussian KDE. Graph diffusion respects track geometry — walls,
holes, and junctions — so place fields do not smear across barriers in 1D or N-D.

The occupancy field and each neuron's spike field are pixellated (histogrammed,
optionally weighted) onto the environment's interior bins, diffused in a single
batched matmul through the cached Laplacian eigenbasis, normalized to integrate to
one, and combined into the rate map. Place fields are stored FULL-GRID (like KDE),
so the non-local likelihood slices interior bins. The local likelihood evaluates the
rate map at the animal's continuous position: by default (``local_interpolation=
"linear"``) it interpolates within a connected interior stencil, falling back to
nearest-bin lookup where interpolation would cross a barrier or invalid bin;
``"nearest"`` restores the plain bin lookup.

The smoothing operator is provided by :mod:`non_local_detector.likelihoods.diffusion`
and cached on the ``Environment``, so it is built once and reused across neurons and
EM refits.
"""

from collections.abc import Sequence
from typing import Any, Literal, cast, get_args

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import scipy.interpolate
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

from non_local_detector.encoding_time import (
    EncodingSupport,
    prepare_encoding_support,
    warn_if_no_encoding_exposure,
)
from non_local_detector.environment import Environment, get_centers
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    RATE_EPS_HZ,
    SpikeTimeOrder,
    as_array,
    as_arrays,
    blocked_nonlocal_poisson_log_likelihood,
    concatenate_row_blocks,
    environment_grid_fingerprint,
    full_to_local_index,
    get_position_at_time,
    interior_bin_indices,
    prepare_decode_rows,
    require_position,
    spike_counts_matrix,
    validate_encoding_contract,
    validate_population_lengths,
    weighted_mean_rate,
)
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    cached_heat_kernel_eigenbasis,
    check_smoothing_bandwidth,
    connected_component_labels,
    diffuse,
    environment_graph,
    resolved_occupancy,
    to_density,
)
from non_local_detector.likelihoods.encoding_model import SortedSpikesEncodingModel
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    decode_bin_centers,
    requires_time_edges,
)

__all__ = [
    "SortedSpikesDiffusionEncodingModel",
    "fit_sorted_spikes_diffusion_encoding_model",
    "predict_sorted_spikes_diffusion_log_likelihood",
]


LocalInterpolation = Literal["linear", "nearest"]
"""How local place fields are read at the animal's position: interpolated
between bin centers (``"linear"``) or from the nearest bin (``"nearest"``)."""

_LOCAL_INTERPOLATION_MODES = set(get_args(LocalInterpolation))


def validate_local_interpolation(local_interpolation: str) -> LocalInterpolation:
    """Validate the local place-field lookup mode."""
    if local_interpolation not in _LOCAL_INTERPOLATION_MODES:
        msg = (
            "local_interpolation must be one of "
            f"{sorted(_LOCAL_INTERPOLATION_MODES)}, got {local_interpolation!r}"
        )
        raise ValidationError(msg)
    return cast("LocalInterpolation", local_interpolation)


def _grid_axes(environment: Environment) -> tuple[np.ndarray, ...]:
    """Sorted per-dimension coordinate axes of the environment's full place grid.

    ``get_centers(edge)`` gives each dimension's bin centers directly; the full grid
    is their tensor product, so these equal ``np.unique(place_bin_centers_[:, d])``.
    """
    return tuple(get_centers(edge) for edge in environment.edges_)


def _interpolation_corner_indices(
    position: np.ndarray,
    axes: tuple[np.ndarray, ...],
    centers_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Full-grid flat indices touched by multilinear interpolation at one position."""
    if np.any(~np.isfinite(position)):
        return None

    candidates: list[list[int]] = []
    for value, axis in zip(position, axes, strict=True):
        if value < axis[0] or value > axis[-1]:
            return None

        upper = int(np.searchsorted(axis, value, side="left"))
        if upper == 0:
            candidates.append([0])
        elif upper == axis.size:
            candidates.append([axis.size - 1])
        elif np.isclose(value, axis[upper]):
            candidates.append([upper])
        else:
            candidates.append([upper - 1, upper])

    corners = np.array(np.meshgrid(*candidates, indexing="ij")).reshape(len(axes), -1)
    return cast(np.ndarray | None, np.ravel_multi_index(corners, centers_shape))


def _linear_interpolation_safe_mask(
    environment: Environment,
    positions: np.ndarray,
    is_track_interior: np.ndarray,
    node_order: np.ndarray | None,
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Rows where local interpolation cannot mix across invalid or disconnected bins."""
    centers_shape = tuple(environment.centers_shape_)
    is_interior = np.asarray(is_track_interior, dtype=bool).ravel()
    if int(np.prod(centers_shape)) != is_interior.shape[0]:
        return np.zeros((positions.shape[0],), dtype=bool)

    graph, graph_node_order, _ = environment_graph(environment)
    node_order = graph_node_order if node_order is None else np.asarray(node_order)
    full_to_local = full_to_local_index(node_order, is_interior.shape[0])

    def _stencil_is_safe(corners: np.ndarray | None) -> bool:
        if corners is None or not np.all(is_interior[corners]):
            return False
        local_nodes = full_to_local[corners]
        if np.any(local_nodes < 0):
            return False
        unique_nodes = np.unique(local_nodes)
        return bool(
            unique_nodes.size <= 1 or nx.is_connected(graph.subgraph(unique_nodes))
        )

    # The verdict depends only on which interpolation stencil a position falls in --
    # fully determined by the per-dim upper-node index, an on-node flag, and whether
    # the position is out of bounds (the branches in _interpolation_corner_indices).
    # The animal dwells within a handful of cells relative to the decode length, so
    # compute that descriptor vectorized, then evaluate the corners + O(cells)
    # connectivity check once per distinct stencil instead of once per time point.
    upper = np.empty(positions.shape, dtype=np.int64)
    on_node = np.zeros(positions.shape, dtype=np.int64)
    out_of_bounds = np.zeros(positions.shape[0], dtype=bool)
    for dim, axis in enumerate(axes):
        column = positions[:, dim]
        out_of_bounds |= ~np.isfinite(column) | (column < axis[0]) | (column > axis[-1])
        dim_upper = np.searchsorted(axis, column, side="left")
        upper[:, dim] = dim_upper
        clamped = np.clip(dim_upper, 0, axis.size - 1)
        on_node[:, dim] = np.isclose(column, axis[clamped])

    # Collapse every out-of-bounds row to a single descriptor (verdict is always False).
    descriptor = np.column_stack(
        [
            np.where(out_of_bounds[:, None], -1, upper),
            np.where(out_of_bounds[:, None], 0, on_node),
            out_of_bounds.astype(np.int64)[:, None],
        ]
    )
    unique_desc, first_ind, inverse = np.unique(
        descriptor, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()

    verdicts = np.zeros(unique_desc.shape[0], dtype=bool)
    for stencil_ind in range(unique_desc.shape[0]):
        if unique_desc[stencil_ind, -1]:  # out-of-bounds group
            continue
        corners = _interpolation_corner_indices(
            positions[first_ind[stencil_ind]], axes, centers_shape
        )
        verdicts[stencil_ind] = _stencil_is_safe(corners)

    return verdicts[inverse]


def _local_place_field_rates(
    environment: Environment,
    positions: np.ndarray,
    place_fields: jax.Array,
    is_track_interior: np.ndarray,
    node_order: np.ndarray | None,
    local_interpolation: LocalInterpolation,
) -> jax.Array:
    """Evaluate full-grid place fields at local positions.

    ``nearest`` is the historical bin lookup. ``linear`` uses multilinear
    interpolation only where the interpolation stencil is interior and connected;
    every unsafe row falls back to the nearest-bin value.
    """
    local_interpolation = validate_local_interpolation(local_interpolation)
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]

    place_fields_np = np.asarray(place_fields)
    n_neurons = place_fields_np.shape[0]
    if n_neurons == 0:
        return jnp.zeros((positions.shape[0], 0))

    safe_positions = np.nan_to_num(positions, nan=0.0)
    bin_inds = environment.get_bin_ind(safe_positions)
    nearest_rates = place_fields_np[:, bin_inds].T

    if local_interpolation == "nearest":
        return jnp.clip(jnp.asarray(nearest_rates), min=RATE_EPS_HZ, max=None)

    axes = _grid_axes(environment)
    values_grid = np.moveaxis(
        place_fields_np.reshape((n_neurons, *[axis.size for axis in axes])),
        0,
        -1,
    )
    interpolated_rates = scipy.interpolate.interpn(
        axes,
        values_grid,
        positions,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    safe = _linear_interpolation_safe_mask(
        environment, positions, is_track_interior, node_order, axes
    )
    safe = safe & np.all(np.isfinite(interpolated_rates), axis=1)
    rates = np.where(safe[:, np.newaxis], interpolated_rates, nearest_rates)
    return jnp.clip(jnp.asarray(rates), min=RATE_EPS_HZ, max=None)


def pixellate_interior_fields(
    position_time: np.ndarray | jax.Array,
    position: np.ndarray | jax.Array,
    spike_times: Sequence[np.ndarray | jax.Array],
    environment: Environment,
    node_order: np.ndarray,
    weights: np.ndarray,
    *,
    disable_progress_bar: bool = False,
    encoding_support: EncodingSupport | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Histogram weighted occupancy and per-neuron spike counts onto interior bins.

    Shared by the diffusion smoother and the MRF-GAM: both need the same weighted
    count fields (in ``node_order`` order) and per-neuron mean rates before they
    diverge (diffusion smooths + normalizes; the MRF fits a penalized-Poisson GAM).

    Exposure and spikes follow one rule, the encoding support's sample basis:
    each tracking sample's integrated exposure goes to the bin of its position,
    and each spike's weight is split between its two bracketing samples by the
    linear-interpolation fraction (:meth:`EncodingSupport.event_counts`) and goes
    to their bins. A bin then holds spikes only where it holds exposure, in
    proportion to it, even when the animal moves more than a bin per sample.
    Placing spikes at their interpolated positions instead puts them in bins the
    samples skip, which have little or no exposure.

    Parameters
    ----------
    position_time : np.ndarray or jax.Array, shape (n_time_position,)
        Sampling times for the position.
    position : np.ndarray or jax.Array, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of np.ndarray or jax.Array
        Spike times for each neuron.
    environment : Environment
        The spatial environment (must be fitted).
    node_order : np.ndarray, shape (n_interior,)
        Interior flat-bin indices from :func:`environment_graph`.
    weights : np.ndarray, shape (n_time_position,)
        Per-sample weights (must be a 1-D array).
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.
    encoding_support : EncodingSupport, optional
        Precomputed encoding support, by default None (built from
        ``position_time``, ``position`` and ``weights``).

    Returns
    -------
    occupancy_field : np.ndarray, shape (n_interior,)
        Weighted exposure in seconds per interior bin.
    spike_fields : list[np.ndarray]
        Weighted spike count per interior bin (each array shape ``(n_interior,)``), one
        array per neuron.
    mean_rates : list[float]
        Weighted event count divided by weighted exposure seconds per neuron (Hz).
    """
    support, weights, exposure_weights, position = prepare_encoding_support(
        position_time,
        position,
        weights,
        _encoding_support=encoding_support,
    )
    n_total_bins = environment.is_track_interior_.ravel().shape[0]
    n_interior = node_order.shape[0]

    full_to_local = full_to_local_index(node_order, n_total_bins)

    # A sample that receives event mass brackets a supported spike, so its basis
    # overlaps the support: binning the exposed samples covers every event.
    active = exposure_weights > 0
    sample_bins = (
        interior_bin_indices(
            environment,
            get_position_at_time(
                position_time, position, position_time[active], environment
            ),
            full_to_local,
        )
        if np.any(active)
        else np.zeros(0, dtype=int)
    )
    occupancy_field = np.bincount(
        sample_bins, weights=exposure_weights[active], minlength=n_interior
    )

    weight_sum = exposure_weights.sum()
    mean_rates: list[float] = []
    spike_fields: list[np.ndarray] = []
    for neuron_spike_times in tqdm(
        spike_times,
        unit="cell",
        desc="Encoding models",
        disable=disable_progress_bar,
    ):
        event_counts = support.event_counts(neuron_spike_times, weights)[active]
        mean_rates.append(weighted_mean_rate(event_counts, weight_sum))
        spike_fields.append(
            np.bincount(sample_bins, weights=event_counts, minlength=n_interior)
        )

    return occupancy_field, spike_fields, mean_rates


def assemble_place_fields(
    rate_interior: np.ndarray, node_order: np.ndarray, n_total_bins: int
) -> tuple[jax.Array, jax.Array]:
    """Scatter per-neuron interior rates onto FULL-GRID place fields (0 off-track).

    Shared by the diffusion and MRF fits so the EPS floor, node-order scatter and
    no-spike term are computed once.

    Parameters
    ----------
    rate_interior : np.ndarray, shape (n_interior, n_neurons)
        Per-interior-bin firing rate in Hz for each neuron.
    node_order : np.ndarray, shape (n_interior,)
        Interior flat-bin indices in graph order.
    n_total_bins : int
        Number of bins in the full grid (interior and off-track).

    Returns
    -------
    place_fields : jax.Array, shape (n_neurons, n_total_bins)
        Floored at ``RATE_EPS_HZ`` on interior bins, exactly 0 off-track.
    no_spike_part_log_likelihood : jax.Array, shape (n_total_bins,)
        Sum of Hz rates across neurons; multiply by duration before scoring.
    """
    clipped_interior = np.clip(
        rate_interior.T, RATE_EPS_HZ, None
    )  # (n_neurons, n_interior)
    place_fields: np.ndarray | jax.Array = np.zeros(
        (rate_interior.shape[1], n_total_bins)
    )
    place_fields[:, node_order] = clipped_interior
    place_fields = jnp.asarray(place_fields)
    return place_fields, jnp.sum(place_fields, axis=0)


class SortedSpikesDiffusionEncodingModel(SortedSpikesEncodingModel):
    """The encoding model ``fit_sorted_spikes_diffusion_encoding_model`` returns.

    Its keys are the encoding-model parameters of
    ``predict_sorted_spikes_diffusion_log_likelihood``, documented there.
    """

    occupancy: np.ndarray
    mean_rates: np.ndarray
    no_spike_part_log_likelihood: jax.Array
    is_track_interior: np.ndarray
    node_order: np.ndarray
    bin_sizes: np.ndarray
    local_interpolation: LocalInterpolation


def fit_sorted_spikes_diffusion_encoding_model(
    position_time: ArrayLike,
    position: ArrayLike,
    spike_times: Sequence[ArrayLike],
    environment: Environment,
    weights: ArrayLike | None = None,
    *,
    position_std: float = float(np.sqrt(12.5)),
    rank: int | None = None,
    local_interpolation: LocalInterpolation = "linear",
    disable_progress_bar: bool = False,
    encoding_time_range: ArrayLike | None = None,
    valid_position_intervals: ArrayLike | None = None,
    _encoding_support: EncodingSupport | None = None,
) -> SortedSpikesDiffusionEncodingModel:
    """Fit a graph-diffusion encoding model for sorted spikes.

    Parameters
    ----------
    position_time : array_like, shape (n_time_position,)
        Sampling times for the position.
    position : array_like, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : sequence of array_like, each shape (n_spikes,)
        Spike times for each neuron.
    environment : Environment
        The spatial environment (must be fitted).
    weights : array_like, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM), by default
        None. If None, uniform weights are used.
    position_std : float, optional
        Heat-kernel smoothing standard deviation in coordinate units (the physical
        bandwidth), by default sqrt(12.5). Scalar only: the heat kernel is isotropic in
        graph distance, so (unlike the KDE likelihood) a per-dimension ``[sx, sy]`` is
        not supported.
    rank : int or None, optional
        Number of Laplacian eigenmodes to use. None (default) auto-selects the rank
        from ``position_std`` via ``diffusion.heat_kernel_rank`` — a
        near-lossless truncation that keeps only the modes the heat kernel needs at
        this bandwidth (``~ area / position_std**2`` modes, independent of grid
        fineness), so large grids avoid a full dense eigendecomposition. Pass an
        explicit integer to force a fixed rank, or a value ``>= n_interior_bins`` for
        the exact full basis.
    local_interpolation : {"linear", "nearest"}, optional
        How local likelihood evaluates full-grid rate maps at the animal's position, by
        default "linear". ``"linear"`` interpolates within connected interior stencils
        and falls back to nearest-bin lookup otherwise. ``"nearest"`` preserves the
        historical bin lookup.
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
    _encoding_support : EncodingSupport, optional
        Internal: the detector's precomputed encoding support. Direct callers
        omit it and the support is built from the other arguments.

    Returns
    -------
    encoding_model : SortedSpikesDiffusionEncodingModel
        - 'algorithm': ``"sorted_spikes_diffusion"``
        - 'environment': the spatial environment
        - 'occupancy': occupancy density at interior place bins
        - 'mean_rates': mean firing rate per neuron
        - 'place_fields': FULL-GRID place fields, shape (n_neurons, n_total_bins)
        - 'no_spike_part_log_likelihood': summed place fields, shape (n_total_bins,)
        - 'is_track_interior': interior-bin mask, shape (n_total_bins,)
        - 'node_order': interior flat-bin indices in graph order, shape (n_interior,)
        - 'bin_sizes': per-interior-bin volume, shape (n_interior,)
        - 'local_interpolation': local likelihood interpolation mode
        - 'disable_progress_bar': progress-bar setting
        - 'rate_units': ``"Hz"``
        - 'encoding_exposure_seconds': weighted encoding occupancy time, seconds

    Raises
    ------
    ValidationError
        If ``position_std`` is not a scalar (per-dimension bandwidths are not
        supported), or if ``weights`` is not a 1-D array of length ``n_time_position``
        with finite, non-negative values.
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
    local_interpolation = validate_local_interpolation(local_interpolation)
    # Coerce position_std to a plain float: it becomes part of the bandwidth-keyed rank
    # cache key, so a JAX scalar/array (unhashable) would raise. The heat kernel is
    # isotropic in graph distance, so per-dimension bandwidths are not supported.
    position_std_arr = np.asarray(position_std, dtype=float)
    if position_std_arr.size != 1:
        msg = (
            "position_std must be a scalar; the graph heat kernel is isotropic in graph "
            "distance, so per-dimension bandwidths are not supported."
        )
        raise ValidationError(
            msg,
            expected="a scalar position_std",
            got=f"array of shape {position_std_arr.shape}",
        )
    position_std = float(position_std_arr.item())

    graph, node_order, bin_sizes = environment_graph(environment)
    check_smoothing_bandwidth(position_std, graph)
    warn_if_no_encoding_exposure(
        exposure_weights, "Sorted-spikes diffusion", clusterless=False
    )
    if rank is None:
        # Auto-truncate: keep only the modes the heat kernel needs at this bandwidth
        # (near-lossless), so large grids avoid a full dense eigendecomposition. Resolving
        # the rank and building the basis share one eigensolve.
        eigvals, eigvecs = cached_heat_kernel_eigenbasis(environment, position_std)
    else:
        eigvals, eigvecs = cached_eigenbasis(environment, rank)

    is_track_interior = environment.is_track_interior_.ravel()
    n_total_bins = is_track_interior.shape[0]

    occupancy_field, spike_fields, mean_rates = pixellate_interior_fields(
        position_time,
        position,
        spike_times,
        environment,
        node_order,
        weights,
        disable_progress_bar=disable_progress_bar,
        encoding_support=support,
    )

    # Diffuse occupancy + all neuron fields in a single batched matmul, then
    # convert each column of per-bin masses to an integral-one density. Bin volumes
    # cancel in marginal / occupancy, so the rate is smoothed spike count over
    # smoothed exposure seconds even on unequal bins. Per-component labels keep
    # truncated-rank smoothing from moving mass across disconnected components.
    fields = np.column_stack([occupancy_field, *spike_fields])
    density = to_density(
        diffuse(
            eigvals,
            eigvecs,
            position_std,
            fields,
            component_labels=connected_component_labels(graph),
        ),
        bin_sizes,
    )
    occupancy = density[:, 0]
    marginals = density[:, 1:]

    # rate(x) = mean_rate * marginal / occupancy on resolved bins, EPS elsewhere.
    occupied = resolved_occupancy(occupancy)
    safe_occupancy = np.where(occupied, occupancy, 1.0)
    rate_interior = np.asarray(mean_rates)[np.newaxis, :] * np.where(
        occupied[:, np.newaxis], marginals / safe_occupancy[:, np.newaxis], EPS
    )
    place_fields, no_spike_part_log_likelihood = assemble_place_fields(
        rate_interior, node_order, n_total_bins
    )

    return {
        "algorithm": "sorted_spikes_diffusion",
        "environment": environment,
        "environment_grid": environment_grid_fingerprint(environment),
        "occupancy": occupancy,
        "mean_rates": np.asarray(mean_rates),
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "node_order": node_order,
        "bin_sizes": bin_sizes,
        "local_interpolation": local_interpolation,
        "disable_progress_bar": disable_progress_bar,
        "rate_units": "Hz",
        "encoding_exposure_seconds": float(exposure_weights.sum()),
    }


@requires_time_edges
def predict_sorted_spikes_diffusion_log_likelihood(
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: Sequence[ArrayLike],
    *,
    environment: Environment,
    occupancy: np.ndarray,  # noqa: ARG001 -- shared likelihood-algorithm signature
    mean_rates: np.ndarray,
    place_fields: jax.Array,
    no_spike_part_log_likelihood: jax.Array,
    is_track_interior: np.ndarray,
    node_order: np.ndarray | None = None,
    bin_sizes: np.ndarray | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    local_interpolation: LocalInterpolation = "linear",
    disable_progress_bar: bool = False,
    is_local: bool = False,
    row_slice: slice | None = None,
    time_edges: np.ndarray,
    algorithm: str | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    rate_units: str = "Hz",
    encoding_exposure_seconds: float | None = None,  # noqa: ARG001 -- shared likelihood-algorithm signature
    environment_grid: dict[str, Any] | None = None,
    _spike_time_order: SpikeTimeOrder | None = None,
    _time_grid: _DecodeTimeGrid | None = None,
) -> jax.Array:
    """Predict the Poisson log-likelihood of sorted spikes under the diffusion model.

    Dedicated function (not the KDE predict): the base class splats the whole encoding
    dict as keyword arguments. ``occupancy``, ``mean_rates``, and ``bin_sizes`` are
    carried for encoding-dict parity but unused here (the rate is already baked into
    ``place_fields``); ``node_order`` is used only to guard local interpolation against
    crossing invalid graph stencils. The MRF-GAM predicts with this function too
    (``predict_sorted_spikes_mrf_log_likelihood``), after its own fit diagnostics.

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
        The spatial environment (must be fitted).
    occupancy : np.ndarray, shape (n_interior_bins,)
        Occupancy density (unused; carried for KDE parity).
    mean_rates : np.ndarray, shape (n_neurons,)
        Mean firing rate in Hz per neuron (unused; carried for KDE parity).
    place_fields : jax.Array, shape (n_neurons, n_total_bins)
        FULL-GRID spatial firing rates in Hz.
    no_spike_part_log_likelihood : jax.Array, shape (n_total_bins,)
        Sum of FULL-GRID Hz rates across neurons, despite the historical key
        name. Multiply by bin duration and subtract to score no spikes.
    is_track_interior : np.ndarray, shape (n_total_bins,)
        Interior-bin mask.
    node_order : np.ndarray, shape (n_interior,), optional
        Interior flat-bin indices in graph order, by default None. Used only to guard
        local interpolation against crossing invalid graph stencils.
    bin_sizes : np.ndarray, shape (n_interior,), optional
        Per-interior-bin volume, by default None. Carried for encoding-dict parity;
        unused here.
    local_interpolation : {"linear", "nearest"}, optional
        How local likelihood evaluates ``place_fields`` at the animal's position, by
        default "linear".
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.
    is_local : bool, optional
        Compute the likelihood at the animal's position, by default False.
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
    log_likelihood : jax.Array
        Shape (n_rows, n_interior_bins) when ``is_local`` is False, else
        (n_rows, 1). ``n_rows`` is ``n_bins`` unless ``row_slice`` is given.
    """
    validate_encoding_contract(rate_units, environment, environment_grid)
    rows = prepare_decode_rows(time_edges, row_slice, _time_grid, _spike_time_order)
    # Both paths broadcast or contract over neurons, so a population mismatch would
    # otherwise surface as a JAX shape error or, on the local path, broadcast one
    # neuron's rates across the observed spike trains.
    spike_times = as_arrays(spike_times)
    validate_population_lengths(
        "neuron",
        spike_times=spike_times,
        mean_rates=mean_rates,
        place_fields=place_fields,
    )
    durations = jnp.asarray(rows.durations())
    if is_local:
        position_time, position = require_position(
            position_time, position, "local sorted-spikes diffusion likelihood"
        )
        interpolated_position = get_position_at_time(
            position_time,
            position,
            decode_bin_centers(rows.time_edges, rows.start, rows.stop),
            environment,
        )
        local_rates = _local_place_field_rates(
            environment,
            np.asarray(interpolated_position),
            place_fields,
            is_track_interior,
            node_order,
            local_interpolation,
        )
        # local_rates is (n_time, n_neurons). The per-neuron loop summed
        # xlogy(count_n, rate_n) - rate_n over neurons; vectorize as one elementwise
        # reduction. Local rates are floored at RATE_EPS_HZ, so xlogy equals
        # count * log(rate) here.
        local_rates = local_rates * durations[:, None]

        def evaluate_local(start: int, stop: int) -> jax.Array:
            """Return the log likelihood of rows ``start`` to ``stop``, shape (stop - start,)."""
            block_rates = local_rates[start - rows.start : stop - rows.start]
            spike_counts = jnp.asarray(
                spike_counts_matrix(
                    spike_times,
                    rows.time_edges,
                    "Local Likelihood",
                    disable_progress_bar=True,
                    row_slice=slice(start, stop),
                    _spike_time_order=rows.spike_time_order,
                ),
                dtype=block_rates.dtype,
            )
            return (
                jax.scipy.special.xlogy(spike_counts, block_rates) - block_rates
            ).sum(axis=1)

        if rows.start == rows.stop:
            return evaluate_local(rows.start, rows.stop)[:, None]
        # Bounded count blocks, as on the non-local path.
        return concatenate_row_blocks(
            rows.start,
            rows.stop,
            len(spike_times),
            evaluate_local,
            "Local Likelihood",
            disable_progress_bar=disable_progress_bar,
        )[:, None]

    # The shared non-local emission, as for the sorted KDE and GLM: interior
    # fields are floored at RATE_EPS_HZ (see assemble_place_fields), so it takes
    # its count-matrix @ log-fields product in fixed 64-row blocks.
    return blocked_nonlocal_poisson_log_likelihood(
        spike_times,
        rows.time_edges,
        rows.start,
        rows.stop,
        jnp.asarray(place_fields)[:, is_track_interior],
        durations,
        no_spike_part_log_likelihood[is_track_interior],
        disable_progress_bar=disable_progress_bar,
        _spike_time_order=rows.spike_time_order,
    )
