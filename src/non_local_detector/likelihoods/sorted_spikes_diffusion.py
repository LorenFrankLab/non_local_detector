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

from non_local_detector.environment import Environment
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
    """Return sorted coordinate axes for the environment's full place grid."""
    if environment.place_bin_centers_ is None:
        raise ValueError("environment must have place_bin_centers_ set")

    place_bin_centers = np.asarray(environment.place_bin_centers_)
    if place_bin_centers.ndim == 1:
        place_bin_centers = place_bin_centers[:, np.newaxis]
    return tuple(
        np.unique(place_bin_centers[:, dimension])
        for dimension in range(place_bin_centers.shape[1])
    )


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
        weights_at_spike_times = scipy.interpolate.interpn(
            (position_time,),
            weights,
            neuron_spike_times,
            bounds_error=False,
            fill_value=None,
        )
        if weights_at_spike_times.ndim > 1:
            weights_at_spike_times = weights_at_spike_times.squeeze(axis=1)

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
    # normalize each column to an integral-one density.
    fields = np.column_stack([occupancy_field, *spike_fields])
    density = to_density(diffuse(eigvals, eigvecs, position_std, fields), bin_sizes)
    occupancy = density[:, 0]
    marginals = density[:, 1:]

    place_fields = np.zeros((len(spike_fields), n_total_bins))
    for neuron, mean_rate in enumerate(mean_rates):
        rate_interior = mean_rate * np.where(
            occupancy > 0.0,
            marginals[:, neuron] / np.where(occupancy > 0.0, occupancy, 1.0),
            EPS,
        )
        place_fields[neuron, node_order] = np.clip(rate_interior, EPS, None)

    place_fields = jnp.asarray(place_fields)
    no_spike_part_log_likelihood = jnp.sum(place_fields, axis=0)

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
    mrf_penalty: float | None = None,
    mrf_rank: int | None = None,
    mrf_coefficients: np.ndarray | None = None,
    mrf_penalty_weights: np.ndarray | None = None,
    mrf_reml_objective: float | None = None,
    mrf_n_iter: int | None = None,
    mrf_converged: bool | None = None,
    mrf_max_step: float | None = None,
    mrf_log_penalty_bounds: tuple[float, float] | None = None,
    mrf_penalty_selected_by_reml: bool | None = None,
    disable_progress_bar: bool = False,
    is_local: bool = False,
) -> jnp.ndarray:
    """Predict the Poisson log-likelihood of sorted spikes under the diffusion model.

    Dedicated function (not the KDE predict): the base class splats the whole
    encoding dict as keyword arguments, so every dict key is a parameter here.
    ``occupancy``, ``mean_rates``, ``node_order``, ``bin_sizes``, and the optional
    ``mrf_*`` diagnostics are carried in encoding dicts (shared contract) but mostly
    unused here; the rate is already baked into ``place_fields``. ``node_order`` is
    used only to guard local interpolation against crossing invalid graph stencils.

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
    mrf_* : optional
        MRF-GAM fit diagnostics accepted so the MRF encoding dict can be passed to
        prediction with ``**encoding_model``.
    disable_progress_bar : bool, optional
    is_local : bool, optional
        Compute the likelihood at the animal's position, by default False.

    Returns
    -------
    log_likelihood : jnp.ndarray
        Shape (n_time, n_interior_bins) when ``is_local`` is False, else (n_time, 1).
    """
    n_time = time.shape[0]

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
        log_likelihood = jnp.zeros((n_time,))
        for neuron_spike_times, local_rate in zip(
            tqdm(
                spike_times,
                unit="cell",
                desc="Local Likelihood",
                disable=disable_progress_bar,
            ),
            local_rates.T,
            strict=False,
        ):
            neuron_spike_times = neuron_spike_times[
                np.logical_and(
                    neuron_spike_times >= time[0],
                    neuron_spike_times <= time[-1],
                )
            ]
            spike_count_per_time_bin = get_spikecount_per_time_bin(
                neuron_spike_times, time
            )
            log_likelihood += (
                jax.scipy.special.xlogy(spike_count_per_time_bin, local_rate)
                - local_rate
            )
        return jnp.expand_dims(log_likelihood, axis=1)

    n_interior_bins = int(is_track_interior.sum())
    log_likelihood = jnp.zeros((n_time, n_interior_bins))
    for neuron_spike_times, place_field in zip(
        tqdm(
            spike_times,
            unit="cell",
            desc="Non-Local Likelihood",
            disable=disable_progress_bar,
        ),
        place_fields,
        strict=False,
    ):
        neuron_spike_times = neuron_spike_times[
            np.logical_and(
                neuron_spike_times >= time[0],
                neuron_spike_times <= time[-1],
            )
        ]
        spike_count_per_time_bin = get_spikecount_per_time_bin(neuron_spike_times, time)
        log_likelihood += jax.scipy.special.xlogy(
            np.expand_dims(spike_count_per_time_bin, axis=1),
            jnp.expand_dims(place_field[is_track_interior], axis=0),
        )

    log_likelihood -= no_spike_part_log_likelihood[is_track_interior]

    return log_likelihood
