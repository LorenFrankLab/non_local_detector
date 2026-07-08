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

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.environment import Environment, get_centers
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    get_position_at_time,
    get_spikecount_per_time_bin,
    validate_weights,
)
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    check_smoothing_bandwidth,
    connected_component_labels,
    diffuse,
    environment_graph,
    to_density,
)

_LOCAL_INTERPOLATION_MODES = {"nearest", "linear"}


def _interior_bin_indices(
    environment: Environment, positions: np.ndarray, full_to_local: np.ndarray
) -> np.ndarray:
    """Map sample positions to local interior-bin indices (``node_order`` order)."""
    return full_to_local[environment.get_bin_ind(positions)]


def _validate_local_interpolation(local_interpolation: str) -> str:
    """Validate the local place-field lookup mode."""
    if local_interpolation not in _LOCAL_INTERPOLATION_MODES:
        raise ValidationError(
            "local_interpolation must be one of "
            f"{sorted(_LOCAL_INTERPOLATION_MODES)}, got {local_interpolation!r}"
        )
    return local_interpolation


def _grid_axes(environment: Environment) -> tuple[np.ndarray, ...]:
    """Sorted per-dimension coordinate axes of the environment's full place grid.

    ``get_centers(edge)`` gives each dimension's bin centers directly; the full grid
    is their tensor product, so these equal ``np.unique(place_bin_centers_[:, d])``.
    """
    if environment.edges_ is None:
        raise ValueError("environment must be fitted (edges_ is None)")
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
    return np.ravel_multi_index(corners, centers_shape)


def _linear_interpolation_safe_mask(
    environment: Environment,
    positions: np.ndarray,
    is_track_interior: np.ndarray,
    node_order: np.ndarray | None,
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Rows where local interpolation cannot mix across invalid or disconnected bins."""
    if environment.centers_shape_ is None:
        return np.zeros((positions.shape[0],), dtype=bool)

    centers_shape = tuple(environment.centers_shape_)
    is_interior = np.asarray(is_track_interior, dtype=bool).ravel()
    if int(np.prod(centers_shape)) != is_interior.shape[0]:
        return np.zeros((positions.shape[0],), dtype=bool)

    graph, graph_node_order, _ = environment_graph(environment)
    node_order = graph_node_order if node_order is None else np.asarray(node_order)
    full_to_local = np.full(is_interior.shape[0], -1, dtype=int)
    full_to_local[node_order] = np.arange(node_order.shape[0])

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
    place_fields: jnp.ndarray,
    is_track_interior: np.ndarray,
    node_order: np.ndarray | None,
    local_interpolation: str,
) -> jnp.ndarray:
    """Evaluate full-grid place fields at local positions.

    ``nearest`` is the historical bin lookup. ``linear`` uses multilinear
    interpolation only where the interpolation stencil is interior and connected;
    every unsafe row falls back to the nearest-bin value.
    """
    local_interpolation = _validate_local_interpolation(local_interpolation)
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
        return jnp.clip(jnp.asarray(nearest_rates), min=EPS, max=None)

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
    return jnp.clip(jnp.asarray(rates), min=EPS, max=None)


def pixellate_interior_fields(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    node_order: np.ndarray,
    weights: np.ndarray,
    disable_progress_bar: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Histogram weighted occupancy and per-neuron spike counts onto interior bins.

    Shared by the diffusion smoother and the MRF-GAM: both need the same weighted
    count fields (in ``node_order`` order) and per-neuron mean rates before they
    diverge (diffusion smooths + normalizes; the MRF fits a penalized-Poisson GAM).

    Parameters
    ----------
    position_time, position, spike_times, environment, weights, disable_progress_bar
        As in the fit functions; ``weights`` must be a 1-D array.
    node_order : np.ndarray, shape (n_interior,)
        Interior flat-bin indices from :func:`environment_graph`.

    Returns
    -------
    occupancy_field : np.ndarray, shape (n_interior,)
        Weighted position-sample count per interior bin.
    spike_fields : list[np.ndarray]
        Weighted spike count per interior bin, one array per neuron.
    mean_rates : list[float]
        ``weights_at_spike_times.sum() / weights.sum()`` per neuron.
    """
    assert environment.is_track_interior_ is not None
    n_total_bins = environment.is_track_interior_.ravel().shape[0]
    n_interior = node_order.shape[0]

    # Full-grid interior flat index -> local interior index (node_order order).
    full_to_local = np.full(n_total_bins, -1, dtype=int)
    full_to_local[node_order] = np.arange(n_interior)

    occupancy_positions = get_position_at_time(
        position_time, position, position_time, environment
    )
    occupancy_field = np.bincount(
        _interior_bin_indices(environment, occupancy_positions, full_to_local),
        weights=weights,
        minlength=n_interior,
    )

    weight_sum = weights.sum()
    mean_rates: list[float] = []
    spike_fields: list[np.ndarray] = []
    for neuron_spike_times in tqdm(
        spike_times,
        unit="cell",
        desc="Encoding models",
        disable=disable_progress_bar,
    ):
        neuron_spike_times = neuron_spike_times[
            np.logical_and(
                neuron_spike_times >= position_time[0],
                neuron_spike_times <= position_time[-1],
            )
        ]
        # Spike times are already clipped to [position_time[0], position_time[-1]],
        # so 1-D linear np.interp matches interpn without building an interpolator.
        weights_at_spike_times = np.interp(neuron_spike_times, position_time, weights)

        mean_rates.append(
            float(weights_at_spike_times.sum() / weight_sum) if weight_sum > 0 else 0.0
        )

        if neuron_spike_times.shape[0] > 0:
            spike_positions = get_position_at_time(
                position_time, position, neuron_spike_times, environment
            )
            spike_field = np.bincount(
                _interior_bin_indices(environment, spike_positions, full_to_local),
                weights=weights_at_spike_times,
                minlength=n_interior,
            )
        else:
            spike_field = np.zeros((n_interior,))
        spike_fields.append(spike_field)

    return occupancy_field, spike_fields, mean_rates


def _assemble_place_fields(
    rate_interior: np.ndarray, node_order: np.ndarray, n_total_bins: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scatter per-neuron interior rates onto FULL-GRID place fields (0 off-track).

    Shared by the diffusion and MRF fits so the EPS floor, node-order scatter, and
    no-spike term are computed once.

    Parameters
    ----------
    rate_interior : np.ndarray, shape (n_interior, n_neurons)
        Per-interior-bin firing rate for each neuron.
    node_order : np.ndarray, shape (n_interior,)
        Interior flat-bin indices in graph order.
    n_total_bins : int

    Returns
    -------
    place_fields : jnp.ndarray, shape (n_neurons, n_total_bins)
        EPS-floored on interior bins, exactly 0 off-track.
    no_spike_part_log_likelihood : jnp.ndarray, shape (n_total_bins,)
        Summed place fields.
    """
    place_fields = np.zeros((rate_interior.shape[1], n_total_bins))
    place_fields[:, node_order] = np.clip(rate_interior.T, EPS, None)
    place_fields = jnp.asarray(place_fields)
    return place_fields, jnp.sum(place_fields, axis=0)


def fit_sorted_spikes_diffusion_encoding_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    weights: np.ndarray | None = None,
    sampling_frequency: int = 500,
    position_std: float = float(np.sqrt(12.5)),
    rank: int | None = None,
    block_size: int = 100,
    local_interpolation: str = "linear",
    disable_progress_bar: bool = False,
) -> dict:
    """Fit a graph-diffusion encoding model for sorted spikes.

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Sampling times for the position.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : list[np.ndarray]
        Spike times for each neuron.
    environment : Environment
        The spatial environment (must be fitted).
    weights : np.ndarray, shape (n_time_position,), optional
        Per-sample weights (e.g. posterior state probabilities during EM). If None,
        uniform weights are used.
    sampling_frequency : int, optional
        Samples per second, by default 500. Accepted for signature compatibility;
        not used by the diffusion smoother.
    position_std : float, optional
        Heat-kernel smoothing standard deviation in coordinate units (the physical
        bandwidth), by default sqrt(12.5).
    rank : int or None, optional
        Number of Laplacian eigenmodes to use. None (default) uses the full basis;
        a smaller rank is a low-pass approximation for large grids. Must be an
        explicit parameter so ``sorted_spikes_algorithm_params={"rank": ...}`` is
        not dropped by the base class's signature filter.
    block_size : int, optional
        Accepted for signature compatibility with the shared sorted-spikes params
        (the default params include ``block_size``); not used by the diffusion
        smoother.
    local_interpolation : {"linear", "nearest"}, optional
        How local likelihood evaluates full-grid rate maps at the animal's position.
        ``"linear"`` interpolates within connected interior stencils and falls back
        to nearest-bin lookup otherwise. ``"nearest"`` preserves the historical
        bin lookup.
    disable_progress_bar : bool, optional
        Turn off the progress bar, by default False.

    Returns
    -------
    encoding_model : dict
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
    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = validate_weights(weights, position.shape[0])
    local_interpolation = _validate_local_interpolation(local_interpolation)

    graph, node_order, bin_sizes = environment_graph(environment)
    check_smoothing_bandwidth(position_std, graph)
    eigvals, eigvecs = cached_eigenbasis(environment, rank)

    # environment_graph raises if the environment is unfitted, so is_track_interior_
    # is guaranteed set here (this also narrows the type for static checkers).
    assert environment.is_track_interior_ is not None
    is_track_interior = environment.is_track_interior_.ravel()
    n_total_bins = is_track_interior.shape[0]

    occupancy_field, spike_fields, mean_rates = pixellate_interior_fields(
        position_time,
        position,
        spike_times,
        environment,
        node_order,
        weights,
        disable_progress_bar,
    )

    # Diffuse occupancy + all neuron fields in a single batched matmul, then
    # normalize each column to an integral-one density. Per-component labels keep
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

    # rate(x) = mean_rate * marginal / occupancy on occupied bins, EPS elsewhere.
    occupied = occupancy > 0.0
    safe_occupancy = np.where(occupied, occupancy, 1.0)
    rate_interior = np.asarray(mean_rates)[np.newaxis, :] * np.where(
        occupied[:, np.newaxis], marginals / safe_occupancy[:, np.newaxis], EPS
    )
    place_fields, no_spike_part_log_likelihood = _assemble_place_fields(
        rate_interior, node_order, n_total_bins
    )

    return {
        "environment": environment,
        "occupancy": occupancy,
        "mean_rates": mean_rates,
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike_part_log_likelihood,
        "is_track_interior": is_track_interior,
        "node_order": node_order,
        "bin_sizes": bin_sizes,
        "local_interpolation": local_interpolation,
        "disable_progress_bar": disable_progress_bar,
    }


def _spike_counts_matrix(
    spike_times: list[np.ndarray],
    time: np.ndarray,
    desc: str,
    disable_progress_bar: bool,
) -> np.ndarray:
    """Stack per-neuron spike counts into a ``(n_time, n_neurons)`` matrix.

    ``get_spikecount_per_time_bin`` masks spikes to ``time`` internally, so no
    explicit pre-masking is needed here.
    """
    counts = [
        get_spikecount_per_time_bin(neuron_spike_times, time)
        for neuron_spike_times in tqdm(
            spike_times, unit="cell", desc=desc, disable=disable_progress_bar
        )
    ]
    if not counts:  # zero neurons
        return np.zeros((time.shape[0], 0))
    return np.stack(counts, axis=1)


def predict_sorted_spikes_diffusion_log_likelihood(
    time: np.ndarray,
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    environment: Environment,
    occupancy: np.ndarray,
    mean_rates: list[float],
    place_fields: jnp.ndarray,
    no_spike_part_log_likelihood: jnp.ndarray,
    is_track_interior: np.ndarray,
    node_order: np.ndarray | None = None,
    bin_sizes: np.ndarray | None = None,
    local_interpolation: str = "linear",
    disable_progress_bar: bool = False,
    is_local: bool = False,
    **_encoding_extras: object,
) -> jnp.ndarray:
    """Predict the Poisson log-likelihood of sorted spikes under the diffusion model.

    Dedicated function (not the KDE predict): the base class splats the whole encoding
    dict as keyword arguments. ``occupancy``, ``mean_rates``, and ``bin_sizes`` are
    carried for encoding-dict parity but unused here (the rate is already baked into
    ``place_fields``); ``node_order`` is used only to guard local interpolation against
    crossing invalid graph stencils. ``**_encoding_extras`` absorbs any extra keys a
    reusing estimator adds to its dict -- e.g. the MRF-GAM's ``mrf_*`` diagnostics,
    since ``predict_sorted_spikes_mrf_log_likelihood`` is this same function -- so the
    diffusion signature need not enumerate another estimator's fields.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Decoding time bins.
    position_time : np.ndarray, shape (n_time_position,)
    position : np.ndarray, shape (n_time_position, n_position_dims)
    spike_times : list[np.ndarray]
        Spike times for each neuron.
    environment : Environment
    occupancy : np.ndarray, shape (n_interior_bins,)
        Occupancy density (unused; carried for KDE parity).
    mean_rates : list[float]
        Mean firing rate per neuron (unused; carried for KDE parity).
    place_fields : jnp.ndarray, shape (n_neurons, n_total_bins)
        FULL-GRID place fields.
    no_spike_part_log_likelihood : jnp.ndarray, shape (n_total_bins,)
    is_track_interior : np.ndarray, shape (n_total_bins,)
    local_interpolation : {"linear", "nearest"}, optional
        How local likelihood evaluates ``place_fields`` at the animal's position.
    disable_progress_bar : bool, optional
    is_local : bool, optional
        Compute the likelihood at the animal's position, by default False.
    **_encoding_extras
        Extra encoding-dict keys a reusing estimator carries (e.g. the MRF's
        ``mrf_*`` diagnostics); absorbed and unused.

    Returns
    -------
    log_likelihood : jnp.ndarray
        Shape (n_time, n_interior_bins) when ``is_local`` is False, else (n_time, 1).
    """
    if is_local:
        interpolated_position = get_position_at_time(
            position_time, position, time, environment
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
        # reduction. xlogy is kept (local rates are not guaranteed EPS-floored).
        spike_counts = jnp.asarray(
            _spike_counts_matrix(
                spike_times, time, "Local Likelihood", disable_progress_bar
            ),
            dtype=local_rates.dtype,
        )
        log_likelihood = (
            jax.scipy.special.xlogy(spike_counts, local_rates) - local_rates
        ).sum(axis=1)
        return jnp.expand_dims(log_likelihood, axis=1)

    # Slice all neurons' fields to interior bins once. Interior fields are EPS-floored
    # (see _assemble_place_fields), so log is finite and xlogy(count, field) reduces to
    # count * log(field). The per-neuron loop accumulated sum_n count_n[:, None] *
    # log(field_n)[None, :]; vectorize it as a single (n_time, n_neurons) @
    # (n_neurons, n_interior_bins) matmul instead of N (n_time, n_interior_bins) terms.
    interior_place_fields = jnp.asarray(place_fields)[:, is_track_interior]
    spike_counts = jnp.asarray(
        _spike_counts_matrix(
            spike_times, time, "Non-Local Likelihood", disable_progress_bar
        ),
        dtype=interior_place_fields.dtype,
    )
    log_likelihood = spike_counts @ jnp.log(interior_place_fields)
    log_likelihood -= no_spike_part_log_likelihood[is_track_interior]

    return log_likelihood
