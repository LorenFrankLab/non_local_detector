from typing import Hashable

import networkx as nx
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from non_local_detector.models.base import _DetectorBase


def _gaussian_smooth(
    data: np.ndarray,
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: int = 8,
) -> np.ndarray:
    """1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def _get_MAP_estimate_2d_position_edges(
    posterior: xr.DataArray,
    track_graph: nx.Graph,
    decoder: _DetectorBase,
    environment_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Get the most likely position of the posterior position and the edges
    of the track graph that the position corresponds to.

    Parameters
    ----------
    posterior : xarray.DataArray, shape (n_time, n_position_bins)
        Decoded probability of position
    track_graph : networkx.Graph
        Graph representation of the environment
    decoder : SortedSpikesDecoder, ClusterlessDecoder, SortedSpikesClassifier,
                ClusterlessClassifier
            Model used to decode the data
    environment_name : str, optional
        Name of the environment

    Returns
    -------
    mental_position_2d : numpy.ndarray, shape (n_time, 2)
        Most likely decoded position
    mental_position_edges : numpy.ndarray, shape (n_time,)
        Edge of the `track_graph` that most likely decoded position corresponds
    """
    try:
        environments = decoder.environments
        env = environments[environments.index(environment_name)]
    except AttributeError:
        try:
            env = decoder.environment
        except AttributeError:
            env = decoder

    # Get 2D position on track from decoder MAP estimate
    map_position_ind = (
        posterior.where(env.is_track_interior_).argmax("position", skipna=True).values
    )
    try:
        place_bin_center_2D_position = env.place_bin_center_2D_position_
    except AttributeError:
        place_bin_center_2D_position = np.asarray(
            env.place_bin_centers_nodes_df_.loc[:, ["x_position", "y_position"]]
        )

    mental_position_2d = place_bin_center_2D_position[map_position_ind]

    # Figure out which track segment it belongs to
    try:
        edge_id = env.place_bin_center_ind_to_edge_id_
    except AttributeError:
        edge_id = np.asarray(env.place_bin_centers_nodes_df_.edge_id)

    track_segment_id = edge_id[map_position_ind]
    mental_position_edges = np.asarray(list(track_graph.edges))[track_segment_id]

    return mental_position_2d, mental_position_edges


def _points_toward_node(
    track_graph: nx.Graph, edge: np.ndarray, head_direction: np.ndarray
) -> object:
    """Given an edge, determine the node the head is pointed toward

    Parameters
    ----------
    track_graph : networkx.Graph
    edge : array-like, shape (2,)
    head_direction : array-like
        Angle of head in radians

    Returns
    -------
    node : object

    """
    edge = np.asarray(edge)
    node1_pos = np.asarray(track_graph.nodes[edge[0]]["pos"])
    node2_pos = np.asarray(track_graph.nodes[edge[1]]["pos"])
    edge_vector = node2_pos - node1_pos
    head_vector = np.asarray([np.cos(head_direction), np.sin(head_direction)])

    return edge[(edge_vector @ head_vector >= 0).astype(int)]


def _get_distance_between_nodes(
    track_graph: nx.Graph, node1: Hashable, node2: Hashable
):
    """Get the distance between two nodes in the track graph

    Parameters
    ----------
    track_graph : nx.Graph
    node1 : Hashable
    node2 : Hashable

    Returns
    -------
    distance : float
    """
    node1_pos = np.asarray(track_graph.nodes[node1]["pos"])
    node2_pos = np.asarray(track_graph.nodes[node2]["pos"])
    return np.sqrt(np.sum((node1_pos - node2_pos) ** 2))


def _setup_track_graph(
    track_graph: nx.Graph,
    actual_pos: np.ndarray,
    actual_edge: np.ndarray,
    head_direction: float,
    mental_pos: np.ndarray,
    mental_edge: np.ndarray,
) -> nx.Graph:
    """Takes the track graph and add nodes for the animal's actual position,
    mental position, and head direction.

    Parameters
    ----------
    track_graph : nx.Graph
    actual_pos : array-like, shape (2,)
    actual_edge : array-like, shape (2,)
    head_direction : float
    mental_pos : array-like, shape (2,)
    mental_edge : array-like, shape (2,)

    Returns
    -------
    track_graph : nx.Graph

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

    for node1, node2 in zip(node_order[:-1], node_order[1:]):
        distance = _get_distance_between_nodes(track_graph, node1, node2)
        track_graph.add_edge(node1, node2, distance=distance)

    return track_graph


def _calculate_ahead_behind(
    track_graph: nx.Graph,
    source: Hashable = "actual_position",
    target: Hashable = "mental_position",
) -> int:
    """Calculate whether the animal is ahead or behind the mental position

    Parameters
    ----------
    track_graph : nx.Graph
    source : Hashable, optional
        Node ID, by default "actual_position"
    target : Hashable, optional
        Node ID, by default "mental_position"

    Returns
    -------
    sign : int
        1 if ahead, -1 if behind
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
    """Calculate the distance between two nodes in the track graph

    Parameters
    ----------
    track_graph : nx.Graph
    source : Hashable, optional
        Node ID, by default "actual_position"
    target : Hashable, optional
        Node ID, by default "mental_position"

    Returns
    -------
    shortest_path_distance : float
    """
    return nx.shortest_path_length(
        track_graph, source=source, target=target, weight="distance"
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
    """Convenience function for getting the most likely position of the
    posterior position and actual position/direction of the animal.

    Parameters
    ----------
    posterior: xarray.DataArray, shape (n_time, n_position_bins)
        Decoded probability of position
    track_graph : networkx.Graph
        Graph representation of the environment
    decoder : SortedSpikesDecoder, ClusterlessDecoder, SortedSpikesClassifier,
              ClusterlessClassifier
        Model used to decode the data
    actual_projected_position : numpy.ndarray, shape (n_time, 2)
    track_segment_id : numpy.ndarray, shape (n_time,)
    actual_orientation : numpy.ndarray, shape (n_time,)

    Returns
    -------
    actual_projected_position : numpy.ndarray, shape (n_time, 2)
        2D position of the animal projected onto the `track_graph`.
    actual_edges : numpy.ndarray, shape (n_time,)
        Edge of the `track_graph` that the animal is currently on.
    actual_orientation : numpy.ndarray, shape (n_time,)
        Orientation of the animal in radians.
    mental_position_2d : numpy.ndarray, shape (n_time, 2)
        Most likely decoded position
    mental_position_edges : numpy.ndarray, shape (n_time,)
        Edge of the `track_graph` that most likely decoded position corresponds
        to.

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
        Orientation of the animal in radians at each time point.
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
        negative if behind. Units are the same as the graph edge distances.

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
    ):
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
    posterior: xr.DataArray,
    track_graph_with_bin_centers_edges: nx.Graph,
    place_bin_center_ind_to_node: np.ndarray,
    sampling_frequency: float = 500.0,
    smooth_sigma: float = 0.0025,
) -> np.ndarray:
    """Get the speed of the most likely decoded position.

    Parameters
    ----------
    posterior : xr.DataArray
    track_graph_with_bin_centers_edges : nx.Graph
        Track graph with bin centers as nodes and edges
    place_bin_center_ind_to_node : np.ndarray
        Mapping of place bin center index to node ID
    sampling_frequency : float, optional
        Samples per second, by default 500.0
    smooth_sigma : float, optional
        Gaussian smoothing parameter, by default 0.0025

    Returns
    -------
    map_speed : np.ndarray
        Speed of the most likely decoded position.
    """
    dt = 1 / sampling_frequency
    posterior = np.asarray(posterior)
    map_position_ind = np.argmax(posterior, axis=1)
    node_ids = place_bin_center_ind_to_node[map_position_ind]
    n_time = len(node_ids)

    if n_time == 1:
        return np.asarray([np.nan])
    elif n_time == 2:
        speed = np.asarray([])
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
        speed = np.insert(
            speed,
            -1,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[-2],
                target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    else:
        speed = []
        for node1, node2 in zip(node_ids[:-2], node_ids[2:]):
            speed.append(
                nx.shortest_path_length(
                    track_graph_with_bin_centers_edges,
                    source=node1,
                    target=node2,
                    weight="distance",
                )
                / (2.0 * dt)
            )
        speed = np.asarray(speed)
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
        speed = np.insert(
            speed,
            -1,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[-2],
                target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    return _gaussian_smooth(np.abs(speed), smooth_sigma, sampling_frequency)
