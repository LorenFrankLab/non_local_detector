"""Position at spike times, snapping of off-track positions and interior bins."""

import itertools
from typing import TYPE_CHECKING, Literal, cast

import jax
import numpy as np
import scipy.interpolate
import scipy.spatial

from non_local_detector.environment import Environment

if TYPE_CHECKING:
    from non_local_detector.encoding_time import EncodingSupport


def get_position_at_time(
    time: np.ndarray | jax.Array,
    position: np.ndarray | jax.Array,
    spike_times: np.ndarray | jax.Array,
    env: Environment | None = None,
    *,
    encoding_support: "EncodingSupport | None" = None,
) -> np.ndarray:
    """Get the position at the time of each spike.

    Parameters
    ----------
    time : jax.Array, shape (n_time,)
        Time of each position sample, in seconds.
    position : jax.Array, shape (n_time_position, n_dims_position)
        Position samples.
    spike_times : jax.Array, shape (n_spikes,)
        Spike times, in seconds.
    env : Environment | None, optional
        The spatial environment, by default None. On a track graph,
        multi-dimensional position is linearized after interpolation and
        one-column position is taken as linear (see
        ``Environment.decoding_position``).
    encoding_support : EncodingSupport, optional
        Encoding support whose segments bound the interpolation, by default
        None (interpolate over all of ``time``, extrapolating outside it).

    Returns
    -------
    position_at_spike_times : np.ndarray, shape (n_spikes, n_decoding_dims)
        Position interpolated at each spike time.
    """
    if encoding_support is None:
        position_at_spike_times = scipy.interpolate.interpn(
            (time,), position, spike_times, bounds_error=False, fill_value=None
        )
    else:
        position_at_spike_times = encoding_support.interpolate(position, spike_times)
    if env is not None:
        position_at_spike_times = env.decoding_position(position_at_spike_times)

    return cast(np.ndarray, position_at_spike_times)


def place_encoding_spikes(
    position_time: np.ndarray | jax.Array,
    position: np.ndarray,
    spike_times: np.ndarray | jax.Array,
    weights: np.ndarray,
    environment: Environment,
    encoding_support: "EncodingSupport",
) -> tuple[np.ndarray, np.ndarray]:
    """Position and weight of each encoding spike of a weighted KDE fit.

    A KDE fit puts each weighted tracking sample's exposure at that sample's
    position. A spike between two samples of equal weight keeps its
    interpolated position and weight. Where the weights differ (a mask edge,
    or posterior weights), the spike's weight is split between its samples as
    its time is, ``(1 - fraction) * weights[left]`` and
    ``fraction * weights[right]``, and the spike sits at the mean of the two
    sample positions weighted by those shares, with their sum as its weight.
    So a spike next to an unweighted sample sits exactly on its weighted
    sample and never past a mask edge. Either way the spike's weight is the
    weight interpolated to its time, and each spike is one entry.

    Parameters
    ----------
    position_time : np.ndarray or jax.Array, shape (n_time_position,)
        Time of each position sample, in seconds.
    position : np.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    spike_times : np.ndarray or jax.Array, shape (n_spikes,)
        Spike times inside ``encoding_support``, in seconds.
    weights : np.ndarray, shape (n_time_position,)
        Non-negative weight of each position sample.
    environment : Environment
        The spatial environment; positions are returned in its decoding
        coordinates (see ``Environment.decoding_position``).
    encoding_support : EncodingSupport
        Encoding support whose segments bound the interpolation.

    Returns
    -------
    spike_positions : np.ndarray, shape (n_spikes, n_decoding_dims)
        Position of each spike.
    spike_weights : np.ndarray, shape (n_spikes,)
        Weight of each spike.
    """
    spike_positions = get_position_at_time(
        position_time,
        position,
        spike_times,
        environment,
        encoding_support=encoding_support,
    )
    spike_weights = encoding_support.interpolate(weights, spike_times, fill_value=0.0)
    spike_index, left, right, fraction = encoding_support.event_basis(spike_times)
    is_transition = weights[left] != weights[right]
    if not np.any(is_transition):
        return spike_positions, spike_weights

    spikes = spike_index[is_transition]
    left, right = left[is_transition], right[is_transition]
    fraction = fraction[is_transition]
    left_share = (1.0 - fraction) * weights[left]
    right_share = fraction * weights[right]
    total = left_share + right_share
    left_position = position[left].astype(float)
    right_position = position[right].astype(float)
    # Exactly the weighted sample's position when the other is unweighted;
    # between two weighted samples, the share-weighted mean of the two.
    mean_position = np.where(
        (weights[left] > 0)[:, None], left_position, right_position
    )
    both = (weights[left] > 0) & (weights[right] > 0)
    mean_position[both] += (right_share[both] / total[both])[:, None] * (
        right_position[both] - left_position[both]
    )
    spike_positions[spikes] = environment.decoding_position(mean_position)
    spike_weights[spikes] = total
    return spike_positions, spike_weights


def snap_off_track_positions(
    position: np.ndarray, environment: Environment
) -> np.ndarray:
    """Move positions outside the track to the nearest interior bin center.

    A local likelihood evaluates its encoding model at the animal's position,
    but the non-local likelihood only scores interior bins. A position beyond
    the grid or in a non-interior bin (a hole, a boundary ring or an
    inter-arm gap) is moved to the nearest interior bin center by Euclidean
    distance, so the local likelihood there equals the non-local one at that
    bin, and no model is evaluated where it extrapolates. Positions inside an
    interior bin, including on its closed boundary, and non-finite positions
    are returned unchanged.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims) or (n_time,)
        Positions in the environment's binning coordinates (linear position
        on a track graph), as ``get_position_at_time`` returns them.
    environment : Environment
        The fitted environment.

    Returns
    -------
    np.ndarray
        Shape and dtype of the input; the input itself when no position is
        off the track.
    """
    position = np.asarray(position)
    points = position if position.ndim > 1 else position[:, np.newaxis]
    if points.shape[0] == 0:
        return position
    edges = environment.edges_
    n_bins = [len(dimension_edges) - 1 for dimension_edges in edges]
    is_interior = environment.is_track_interior_flat_
    finite = np.all(np.isfinite(points), axis=1)
    in_grid = finite & np.all(
        [
            (points[:, dim] >= dimension_edges[0])
            & (points[:, dim] <= dimension_edges[-1])
            for dim, dimension_edges in enumerate(edges)
        ],
        axis=0,
    )
    # A position on an edge between bins belongs to every bin sharing it, so
    # the end of a track arm next to an inter-arm gap, or the corner of an
    # interior bin, is on the track. Per dimension, "left" gives the lower
    # and "right" the upper of the bins sharing an edge (the same bin off an
    # edge); a position is on the track if any combination is interior.
    candidate_bins = [
        [
            np.clip(
                np.searchsorted(dimension_edges, points[:, dim], side=side) - 1,
                0,
                n_bins[dim] - 1,
            )
            for side in cast(tuple[Literal["left", "right"], ...], ("left", "right"))
        ]
        for dim, dimension_edges in enumerate(edges)
    ]
    in_interior_bin = np.zeros(points.shape[0], dtype=bool)
    for bins in itertools.product(*candidate_bins):
        in_interior_bin |= is_interior[np.ravel_multi_index(bins, n_bins)]
    off_track = finite & ~(in_grid & in_interior_bin)
    if not off_track.any():
        return position
    interior_centers = np.asarray(environment.place_bin_centers_)[is_interior]
    _, nearest = scipy.spatial.cKDTree(interior_centers).query(points[off_track])
    snapped = points.copy()
    snapped[off_track] = interior_centers[nearest]
    return snapped.reshape(position.shape)


def full_to_local_index(node_order: np.ndarray, n_total_bins: int) -> np.ndarray:
    """Inverse permutation full-grid flat-bin index -> local interior index.

    ``node_order`` maps local interior index -> full-grid flat index; this returns its
    inverse, with ``-1`` for off-track (non-interior) bins.
    """
    full_to_local = np.full(n_total_bins, -1, dtype=int)
    full_to_local[node_order] = np.arange(node_order.shape[0])
    return full_to_local


def interior_bin_indices(
    environment: Environment, positions: np.ndarray, full_to_local: np.ndarray
) -> np.ndarray:
    """Map sample positions to local interior-bin indices (``node_order`` order)."""
    return cast(np.ndarray, full_to_local[environment.get_bin_ind(positions)])
