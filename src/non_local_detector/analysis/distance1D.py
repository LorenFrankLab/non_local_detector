"""Distances between decoded and actual positions along a 1D track graph."""

import itertools
from collections.abc import Hashable
from typing import Any, cast

import networkx as nx
import numpy as np
import xarray as xr

from non_local_detector.analysis.posterior import get_position_posterior
from non_local_detector.environment import (
    _segment_index_of_edge_id,
    find_environment_by_name,
)
from non_local_detector.environment import gaussian_smooth as _gaussian_smooth
from non_local_detector.exceptions import ValidationError
from non_local_detector.models.base import _DetectorBase

__all__ = [
    "get_ahead_behind_distance",
    "get_map_speed",
    "get_trajectory_data",
]


def _most_likely_bins(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Most likely position bin of each time, and where one exists.

    A row with no position mass, such as the first row of a detector that
    starts in Local, has no most likely bin. It holds the next row's, so it
    adds no jump to what is computed from these bins.

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time, n_position_bins)
        Posterior over position; NaN bins hold no mass.

    Returns
    -------
    map_bins : np.ndarray of int, shape (n_time,)
    has_position : np.ndarray of bool, shape (n_time,)
        False for rows with no position mass.
    """
    posterior = np.nan_to_num(posterior)
    has_position = posterior.sum(axis=1) > 0
    map_bins = posterior.argmax(axis=1)
    defined = np.flatnonzero(has_position)
    if defined.size:
        next_defined = np.searchsorted(defined, np.arange(len(map_bins)))
        map_bins = map_bins[defined[np.minimum(next_defined, defined.size - 1)]]
    return map_bins, has_position


def _get_MAP_estimate_2d_position_edges(
    posterior: xr.DataArray,
    track_graph: nx.Graph,
    decoder: _DetectorBase,
    environment_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MAP estimate 2D positions and corresponding track graph edges.

    Computes the maximum a posteriori (MAP) estimate of position from the
    posterior distribution and identifies which edges of the track graph
    these positions correspond to.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins)
        Decoded posterior probability distribution over position bins, or a
        raw ``results.acausal_posterior`` with a ``state_bins`` dimension.
    track_graph : nx.Graph
        The environment's ``track_graph``, whose edges carry ``edge_id``.
    decoder : _DetectorBase
        Fitted decoder model containing environment information and place
        bin mappings.
    environment_name : str, optional
        Name of the specific environment to use, by default "".
        If empty, uses the decoder's default environment.

    Returns
    -------
    mental_position_2d : np.ndarray, shape (n_time, 2)
        Most likely decoded 2D position coordinates for each time point, NaN
        at a time with no position mass.
    mental_position_edges : np.ndarray, shape (n_time, 2)
        Corresponding track graph edges (node pairs) for each MAP position.
        A time with no position mass holds the next time's edge.

    """
    try:
        environments = decoder.environments_
        env = find_environment_by_name(environments, environment_name)
    except AttributeError:
        try:
            env = decoder.environment
        except AttributeError:
            env = decoder

    # Get 2D position on track from decoder MAP estimate
    map_position_ind, has_position = _most_likely_bins(
        get_position_posterior(posterior)
        .where(env.is_track_interior_)
        .transpose("time", "position")
        .values
    )
    try:
        place_bin_center_2D_position = env.place_bin_center_2D_position_  # type: ignore[attr-defined]  # legacy attribute; AttributeError fallback below
    except AttributeError:
        place_bin_center_2D_position = np.asarray(
            env.place_bin_centers_nodes_df_.loc[:, ["x_position", "y_position"]]  # type: ignore[union-attr]
        )

    mental_position_2d = place_bin_center_2D_position[map_position_ind]
    mental_position_2d[~has_position] = np.nan

    # Figure out which track segment it belongs to
    try:
        # Legacy attribute: the bin's position in track_graph.edges.
        segment_index = env.place_bin_center_ind_to_edge_id_  # type: ignore[attr-defined]  # legacy attribute; AttributeError fallback below
    except AttributeError:
        missing = [
            edge
            for *edge, edge_id in track_graph.edges(data="edge_id")
            if edge_id is None
        ]
        if missing:
            msg = "track_graph edges are missing the 'edge_id' attribute"
            raise ValidationError(
                msg,
                expected="the environment's track_graph, whose edges carry 'edge_id'",
                got=f"{len(missing)} edge(s) without 'edge_id', e.g. {tuple(missing[0])}",
                hint=(
                    "Pass the decoder's environment track_graph, or build the graph "
                    "with track_linearization.make_track_graph, which sets edge_id."
                ),
            ) from None
        edge_id = np.asarray(env.place_bin_centers_nodes_df_.edge_id)  # type: ignore[union-attr]
        segment_index = _segment_index_of_edge_id(track_graph, edge_id)

    track_segment_id = segment_index[map_position_ind]
    mental_position_edges = np.asarray(list(track_graph.edges))[track_segment_id]

    return mental_position_2d, mental_position_edges


def _points_toward_node(
    track_graph: nx.Graph, edge: np.ndarray, head_direction: float | np.ndarray
) -> Any:
    """Determine which node of an edge the head direction vector points toward.

    Given an edge defined by two nodes and a head direction angle, computes
    which node the head is pointing toward based on the dot product of the
    edge vector and head direction vector.

    Parameters
    ----------
    track_graph : nx.Graph
        Graph containing node position information.
    edge : np.ndarray, shape (2,)
        Array containing two node identifiers defining the edge.
    head_direction : np.ndarray, shape ()
        Head orientation angle in radians.

    Returns
    -------
    node : object
        Node identifier of the node that the head direction points toward.

    """
    edge = np.asarray(edge)
    node1_pos = np.asarray(track_graph.nodes[edge[0]]["pos"])
    node2_pos = np.asarray(track_graph.nodes[edge[1]]["pos"])
    edge_vector = node2_pos - node1_pos
    head_vector = np.asarray([np.cos(head_direction), np.sin(head_direction)])

    return edge[(edge_vector @ head_vector >= 0).astype(int)]


def _get_distance_between_nodes(
    track_graph: nx.Graph, node1: Hashable, node2: Hashable
) -> float:
    """Calculate Euclidean distance between two nodes in the track graph.

    Computes the straight-line distance between two nodes using their
    2D position coordinates stored in the graph.

    Parameters
    ----------
    track_graph : nx.Graph
        Graph containing node position data in 'pos' attribute.
    node1 : Hashable
        Identifier for the first node.
    node2 : Hashable
        Identifier for the second node.

    Returns
    -------
    distance : float
        Euclidean distance between the two nodes.

    """
    node1_pos = np.asarray(track_graph.nodes[node1]["pos"])
    node2_pos = np.asarray(track_graph.nodes[node2]["pos"])
    return cast("float", np.sqrt(np.sum((node1_pos - node2_pos) ** 2)))


def _setup_track_graph(
    track_graph: nx.Graph,
    actual_pos: np.ndarray,
    actual_edge: np.ndarray,
    head_direction: float,
    mental_pos: np.ndarray,
    mental_edge: np.ndarray,
) -> nx.Graph:
    """Add temporary nodes and edges for actual and mental positions.

    Modifies the track graph by inserting temporary nodes representing
    the animal's actual position, head position, and mental (decoded)
    position. Connects these nodes to the existing track structure with
    appropriate edge weights based on distances.

    Parameters
    ----------
    track_graph : nx.Graph
        Original track graph to be modified.
    actual_pos : np.ndarray, shape (2,)
        2D coordinates of animal's actual position.
    actual_edge : np.ndarray, shape (2,)
        Node identifiers defining the edge where actual position lies.
    head_direction : float
        Head orientation angle in radians.
    mental_pos : np.ndarray, shape (2,)
        2D coordinates of decoded mental position.
    mental_edge : np.ndarray, shape (2,)
        Node identifiers defining the edge where mental position lies.

    Returns
    -------
    track_graph : nx.Graph
        Modified graph with temporary nodes and connecting edges added.

    """
    track_graph.add_node("actual_position", pos=actual_pos)
    track_graph.add_node("head", pos=actual_pos)
    track_graph.add_node("mental_position", pos=mental_pos)

    # determine which node head is pointing towards
    node_ahead = _points_toward_node(track_graph, actual_edge, head_direction)
    node_behind = actual_edge[~np.isin(actual_edge, node_ahead)][0]

    # insert edges between nodes
    if np.all(actual_edge == mental_edge):  # actual and mental on same edge
        actual_pos_distance = _get_distance_between_nodes(
            track_graph, "actual_position", node_ahead
        )
        mental_pos_distance = _get_distance_between_nodes(
            track_graph, "mental_position", node_ahead
        )

        if actual_pos_distance < mental_pos_distance:
            node_order = [
                node_ahead,
                "head",
                "actual_position",
                "mental_position",
                node_behind,
            ]
        else:
            node_order = [
                node_ahead,
                "mental_position",
                "head",
                "actual_position",
                node_behind,
            ]
    else:  # actual and mental are on different edges
        node_order = [node_ahead, "head", "actual_position", node_behind]

        distance = _get_distance_between_nodes(
            track_graph, mental_edge[0], "mental_position"
        )
        track_graph.add_edge(mental_edge[0], "mental_position", distance=distance)

        distance = _get_distance_between_nodes(
            track_graph, "mental_position", mental_edge[1]
        )
        track_graph.add_edge("mental_position", mental_edge[1], distance=distance)

    for node1, node2 in itertools.pairwise(node_order):
        distance = _get_distance_between_nodes(track_graph, node1, node2)
        track_graph.add_edge(node1, node2, distance=distance)

    return track_graph


def _calculate_ahead_behind(
    track_graph: nx.Graph,
    source: Hashable = "actual_position",
    target: Hashable = "mental_position",
) -> int:
    """Determine if target position is ahead or behind source along track.

    Calculates the directional relationship between two positions by finding
    the shortest path and checking if the head direction node lies on this path.
    If the head node is on the path, the target is ahead; otherwise, it's behind.

    Parameters
    ----------
    track_graph : nx.Graph
        Track graph containing temporary nodes for actual position, mental
        position, and head direction.
    source : Hashable, optional
        Starting node identifier, by default "actual_position".
    target : Hashable, optional
        Target node identifier, by default "mental_position".

    Returns
    -------
    sign : int
        Direction indicator: 1 if target is ahead of source,
        -1 if target is behind source.

    """
    path = nx.shortest_path(
        track_graph,
        source=source,
        target=target,
        weight="distance",
    )

    return 1 if "head" in path else -1


def _calculate_distance(
    track_graph: nx.Graph,
    source: Hashable = "actual_position",
    target: Hashable = "mental_position",
) -> float:
    """Calculate shortest path distance between two nodes in track graph.

    Computes the weighted shortest path distance between source and target
    nodes using the 'distance' edge attribute as weights.

    Parameters
    ----------
    track_graph : nx.Graph
        Track graph with distance-weighted edges.
    source : Hashable, optional
        Starting node identifier, by default "actual_position".
    target : Hashable, optional
        Target node identifier, by default "mental_position".

    Returns
    -------
    shortest_path_distance : float
        Total distance along the shortest path from source to target.

    """
    return cast(
        "float",
        nx.shortest_path_length(
            track_graph, source=source, target=target, weight="distance"
        ),
    )


def get_trajectory_data(
    posterior: xr.DataArray,
    track_graph: nx.Graph,
    decoder: _DetectorBase,
    actual_projected_position: np.ndarray,
    track_segment_id: np.ndarray,
    actual_orientation: np.ndarray,
    environment_name: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the MAP decoded position and the animal's actual position.

    Convenience function returning the most likely decoded position and the
    actual position, edge and orientation of the animal.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins)
        Decoded probability of position, or a raw
        ``results.acausal_posterior`` with a ``state_bins`` dimension.
    track_graph : networkx.Graph
        The environment's ``track_graph``; the decoded position's edge is
        found by its ``edge_id``.
    decoder : SortedSpikesDecoder, ClusterlessDecoder, SortedSpikesClassifier, or ClusterlessClassifier
        Model used to decode the data.
    actual_projected_position : numpy.ndarray, shape (n_time, 2)
        2D position of the animal projected onto the `track_graph`.
    track_segment_id : numpy.ndarray, shape (n_time,)
        Index of the `track_graph` edge (in `track_graph.edges` order) that the
        animal is on at each time point.
    actual_orientation : numpy.ndarray, shape (n_time,)
        Orientation of the animal in radians.
    environment_name : str, optional
        Name of the environment, by default "".

    Returns
    -------
    actual_projected_position : numpy.ndarray, shape (n_time, 2)
        2D position of the animal projected onto the `track_graph`.
    actual_edges : numpy.ndarray, shape (n_time, 2)
        Edge of the `track_graph` (pair of node IDs) that the animal is
        currently on.
    actual_orientation : numpy.ndarray, shape (n_time,)
        Orientation of the animal in radians.
    mental_position_2d : numpy.ndarray, shape (n_time, 2)
        Most likely decoded position, NaN at a time with no position mass.
    mental_position_edges : numpy.ndarray, shape (n_time, 2)
        Edge of the `track_graph` that most likely decoded position corresponds
        to. A time with no position mass holds the next time's edge.
    """
    (mental_position_2d, mental_position_edges) = _get_MAP_estimate_2d_position_edges(
        posterior, track_graph, decoder, environment_name
    )
    actual_projected_position = np.asarray(actual_projected_position)
    track_segment_id = np.asarray(track_segment_id).astype(int).squeeze()
    actual_edges = np.asarray(list(track_graph.edges))[track_segment_id]
    actual_orientation = np.asarray(actual_orientation)

    return (
        actual_projected_position,
        actual_edges,
        actual_orientation,
        mental_position_2d,
        mental_position_edges,
    )


def get_ahead_behind_distance(
    track_graph: nx.Graph,
    actual_projected_position: np.ndarray,
    actual_edges: np.ndarray,
    actual_orientation: np.ndarray,
    mental_position_2d: np.ndarray,
    mental_position_edges: np.ndarray,
    source: Hashable = "actual_position",
) -> np.ndarray:
    """Calculate signed distance from animal's position to mental position along the track graph.

    Computes the shortest path distance along the track graph between the
    animal's actual projected position and the estimated mental position (MAP).
    The sign indicates direction relative to the animal's orientation:
    positive means the mental position is ahead, negative means behind.

    Parameters
    ----------
    track_graph : networkx.Graph
        Graph representation of the environment, potentially including temporary
        nodes for actual/mental positions.
    actual_projected_position : np.ndarray, shape (n_time, 2)
        2D position of the animal projected onto the `track_graph`.
    actual_edges : np.ndarray, shape (n_time, 2)
        Tuple of node IDs representing the edge of the `track_graph` that the
        animal is currently on at each time point.
    actual_orientation : np.ndarray, shape (n_time,)
        Orientation of the animal in radians at each time point. NaN
        orientations give NaN distances.
    mental_position_2d : np.ndarray, shape (n_time, 2)
        Most likely decoded 2D position (MAP estimate) at each time point.
    mental_position_edges : np.ndarray, shape (n_time, 2)
        Tuple of node IDs representing the edge of the `track_graph` that the
        most likely decoded position corresponds to at each time point.
    source : Hashable, optional
        Node ID representing the starting point for distance calculation,
        typically the temporary node "actual_position".
        By default "actual_position".

    Returns
    -------
    ahead_behind_distance : np.ndarray, shape (n_time,)
        The shortest path distance along the track graph. Positive if the
        mental position is ahead of the animal based on its orientation,
        negative if behind, NaN where the orientation is NaN. Units are the
        same as the graph edge distances.

    Notes
    -----
    This function iteratively modifies a copy of the input `track_graph`
    by adding temporary nodes for the actual and mental positions for each
    time step.
    """
    copy_graph = track_graph.copy()
    ahead_behind_distance = []

    for actual_pos, actual_edge, orientation, map_pos, map_edge in zip(
        actual_projected_position,
        actual_edges,
        actual_orientation,
        mental_position_2d,
        mental_position_edges,
        strict=False,
    ):
        if np.isnan(orientation):
            # Ahead or behind is undefined without a heading.
            ahead_behind_distance.append(np.nan)
            continue

        # Insert nodes for actual position, mental position, head
        copy_graph = _setup_track_graph(
            copy_graph, actual_pos, actual_edge, orientation, map_pos, map_edge
        )

        # Get metrics
        distance = _calculate_distance(
            copy_graph, source=source, target="mental_position"
        )
        ahead_behind = _calculate_ahead_behind(
            copy_graph, source=source, target="mental_position"
        )
        ahead_behind_distance.append(ahead_behind * distance)

        # Cleanup: remove inserted nodes
        copy_graph.remove_node("actual_position")
        copy_graph.remove_node("head")
        copy_graph.remove_node("mental_position")

    return np.asarray(ahead_behind_distance)


def get_map_speed(
    posterior: xr.DataArray | np.ndarray,
    track_graph_with_bin_centers_edges: nx.Graph,
    place_bin_center_ind_to_node: np.ndarray,
    sampling_frequency: float = 500.0,
    smooth_sigma: float = 0.0025,
) -> np.ndarray:
    """Get the speed of the most likely decoded position.

    Parameters
    ----------
    posterior : xr.DataArray or np.ndarray, shape (n_time, n_position_bins)
        Decoded posterior probability of position. A DataArray may also be a
        raw ``results.acausal_posterior`` with a ``state_bins`` dimension; see
        :func:`~non_local_detector.analysis.get_position_posterior`.
    track_graph_with_bin_centers_edges : nx.Graph
        Track graph with bin centers as nodes and edges.
    place_bin_center_ind_to_node : np.ndarray, shape (n_position_bins,)
        Mapping of place bin center index to node ID.
    sampling_frequency : float, optional
        Samples per second, by default 500.0.
    smooth_sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel applied to the
        speed, in seconds, by default 0.0025. Near the first and last times
        the kernel is renormalized over the samples that exist.

    Returns
    -------
    map_speed : np.ndarray
        Speed of the most likely decoded position, NaN at a time with no
        position mass, such as the first time of a detector that starts in
        Local.
    """
    dt = 1 / sampling_frequency
    if isinstance(posterior, xr.DataArray) and (
        "state_bins" in posterior.dims or posterior.ndim > 2
    ):
        posterior = get_position_posterior(posterior)
    map_position_ind, has_position = _most_likely_bins(np.asarray(posterior))
    node_ids = place_bin_center_ind_to_node[map_position_ind]
    n_time = len(node_ids)

    if n_time == 1:
        return np.asarray([np.nan])
    if n_time == 2:
        speed = (
            np.asarray(
                [
                    nx.shortest_path_length(
                        track_graph_with_bin_centers_edges,
                        source=node_ids[0],
                        target=node_ids[1],
                        weight="distance",
                    ),
                    nx.shortest_path_length(
                        track_graph_with_bin_centers_edges,
                        source=node_ids[-2],
                        target=node_ids[-1],
                        weight="distance",
                    ),
                ]
            )
            / dt
        )
    else:
        shortest_path_lengths = dict(
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges, weight="distance"
            )
        )
        speed = np.array(
            [
                shortest_path_lengths[node1][node2]
                for node1, node2 in zip(node_ids[:-2], node_ids[2:], strict=False)
            ]
        ) / (2.0 * dt)
        speed = np.insert(
            speed,
            0,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[0],
                target=node_ids[1],
                weight="distance",
            )
            / dt,
        )
        speed = np.append(
            speed,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[-2],
                target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    # Divide by the kernel mass inside the array: zero padding alone would pull
    # the first and last samples toward zero.
    speed = cast(
        "np.ndarray",
        _gaussian_smooth(np.abs(speed), smooth_sigma, sampling_frequency)
        / _gaussian_smooth(np.ones_like(speed), smooth_sigma, sampling_frequency),
    )
    speed[~has_position] = np.nan
    return speed
