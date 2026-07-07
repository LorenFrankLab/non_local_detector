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
so the non-local likelihood slices interior bins and the local likelihood indexes
the animal's bin directly.

The smoothing operator is provided by :mod:`non_local_detector.likelihoods.diffusion`
and cached on the ``Environment``, so it is built once and reused across neurons and
EM refits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate  # type: ignore[import-untyped]
from tqdm.autonotebook import tqdm  # type: ignore[import-untyped]

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import (
    EPS,
    get_position_at_time,
    get_spikecount_per_time_bin,
)
from non_local_detector.likelihoods.diffusion import (
    cached_eigenbasis,
    check_smoothing_bandwidth,
    diffuse,
    environment_graph,
    to_density,
)


def _interior_bin_indices(
    environment: Environment, positions: np.ndarray, full_to_local: np.ndarray
) -> np.ndarray:
    """Map sample positions to local interior-bin indices (``node_order`` order)."""
    return full_to_local[environment.get_bin_ind(positions)]


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
        - 'disable_progress_bar': progress-bar setting
    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    if weights is None:
        weights = np.ones((position.shape[0],))
    weights = np.asarray(weights)

    graph, node_order, bin_sizes = environment_graph(environment)
    check_smoothing_bandwidth(position_std, graph)
    eigvals, eigvecs = cached_eigenbasis(environment, rank)

    # environment_graph raises if the environment is unfitted, so is_track_interior_
    # is guaranteed set here (this also narrows the type for static checkers).
    assert environment.is_track_interior_ is not None
    is_track_interior = environment.is_track_interior_.ravel()
    n_total_bins = is_track_interior.shape[0]
    n_interior = node_order.shape[0]

    # Full-grid interior flat index -> local interior index (node_order order).
    full_to_local = np.full(n_total_bins, -1, dtype=int)
    full_to_local[node_order] = np.arange(n_interior)

    # Weighted occupancy count field on interior bins.
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
    disable_progress_bar: bool = False,
    is_local: bool = False,
) -> jnp.ndarray:
    """Predict the Poisson log-likelihood of sorted spikes under the diffusion model.

    Dedicated function (not the KDE predict): the base class splats the whole
    encoding dict as keyword arguments, so every dict key is a parameter here.
    ``occupancy``, ``mean_rates``, ``node_order``, and ``bin_sizes`` are carried in
    the encoding dict (shared contract) but unused here — the rate is already baked
    into ``place_fields`` and prediction indexes bins via ``environment.get_bin_ind``.

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
        bin_inds = environment.get_bin_ind(interpolated_position)
        log_likelihood = jnp.zeros((n_time,))
        for neuron_spike_times, place_field in zip(
            tqdm(
                spike_times,
                unit="cell",
                desc="Local Likelihood",
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
            spike_count_per_time_bin = get_spikecount_per_time_bin(
                neuron_spike_times, time
            )
            local_rate = jnp.clip(place_field[bin_inds], min=EPS, max=None)
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
