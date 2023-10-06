import networkx as nx
import numpy as np
from non_local_detector.environment import Environment
from scipy.ndimage import gaussian_filter1d


def make_2D_track_graph_from_environment(
    environment: Environment,
) -> nx.Graph:
    """Creates a graph of the position where on track nodes are
    connected by edges.

    Parameters
    ----------
    environment : Environment

    Returns
    -------
    track_graph : nx.Graph
    """

    track_graph = nx.Graph()

    for node_id, (node_position, is_interior) in enumerate(
        zip(
            environment.place_bin_centers_,
            environment.is_track_interior_.ravel(),
        )
    ):
        track_graph.add_node(
            node_id, pos=tuple(node_position), is_track_interior=is_interior
        )

    edges = []
    for x_ind, y_ind in zip(*np.nonzero(environment.is_track_interior_)):
        x_inds, y_inds = np.meshgrid(
            x_ind + np.asarray([-1, 0, 1]),
            y_ind + np.asarray([-1, 0, 1]),
            indexing="ij",
        )
        adj_edges = environment.is_track_interior_[x_inds, y_inds]
        adj_edges[1, 1] = False

        node_id = np.ravel_multi_index((x_ind, y_ind), environment.centers_shape_)
        adj_node_ids = np.ravel_multi_index(
            (x_inds[adj_edges], y_inds[adj_edges]),
            environment.centers_shape_,
        )
        edges.append(
            np.concatenate(np.meshgrid(node_id, adj_node_ids, indexing="ij"), axis=1)
        )

    edges = np.concatenate(edges)

    for node1, node2 in edges:
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


def get_bin_ind(sample: np.ndarray, edges: list) -> np.ndarray:
    """Find the indices of the bins to which each value in input array belongs.

    Parameters
    ----------
    sample : np.ndarray, shape (n_time, n_dim)
    edges : list of np.ndarray, shape (n_bins,)

    Returns
    -------
    bin_ind : np.ndarray, shape (n_time,)

    """
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, np.intp)
    for i in range(D):
        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side="right")
        for i in range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    return np.ravel_multi_index(
        Ncount,
        nbin,
    )


def get_map_estimate_direction_from_track_graph(
    head_position: np.ndarray,
    map_estimate: np.ndarray,
    track_graph: nx.Graph,
    edges: list,
    precomputed_distance: bool = False,
) -> np.ndarray:
    """Get the direction of the MAP estimate of the decoded position from the
    animal's head position.

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
    map_estimate : np.ndarray, shape (n_time, 2)
    track_graph : nx.Graph
    edges : list

    Returns
    -------
    map_estimate_direction : np.ndarray, shape (n_time,)

    """
    node_positions = nx.get_node_attributes(track_graph, "pos")

    map_estimate_direction = np.zeros((head_position.shape[0],))

    # remove outer boundary edge
    bin_edges = [e[1:-1] for e in edges]
    if precomputed_distance:
        node_positions = np.asarray(list(node_positions.values()))
        n_nodes = node_positions.shape[0]
        bin_ind1 = get_bin_ind(head_position, bin_edges)
        bin_ind2 = get_bin_ind(map_estimate, bin_edges)

        first_node_on_path = np.full((n_nodes, n_nodes), -1, dtype=int)
        for from_node_id, to_node_data in nx.shortest_path(
            track_graph,
            weight="distance",
        ).items():
            for to_node_id, path in to_node_data.items():
                try:
                    first_node_on_path[from_node_id, to_node_id] = path[1]
                except IndexError:
                    first_node_on_path[from_node_id, to_node_id] = path[0]
        head_position_node_pos = node_positions[bin_ind1]
        first_node_on_path_pos = node_positions[first_node_on_path[bin_ind1, bin_ind2]]
        map_estimate_direction = np.arctan2(
            first_node_on_path_pos[:, 1] - head_position_node_pos[:, 1],
            first_node_on_path_pos[:, 0] - head_position_node_pos[:, 0],
        )
    else:
        node_ids = np.asarray(list(node_positions.keys()))
        head_position_nodes = node_ids[get_bin_ind(head_position, bin_edges)]
        map_estimate_nodes = node_ids[get_bin_ind(map_estimate, bin_edges)]

        for i, (head_position_node, map_estimate_node) in enumerate(
            zip(head_position_nodes, map_estimate_nodes)
        ):
            try:
                first_node_on_path = nx.shortest_path(
                    track_graph,
                    source=head_position_node,
                    target=map_estimate_node,
                    weight="distance",
                )[1]
            except IndexError:
                # head_position_node and map_estimate_node are the same
                first_node_on_path = map_estimate_node

            head_position_node_pos = node_positions[head_position_node]
            first_node_on_path_pos = node_positions[first_node_on_path]

            map_estimate_direction[i] = np.arctan2(
                first_node_on_path_pos[1] - head_position_node_pos[1],
                first_node_on_path_pos[0] - head_position_node_pos[0],
            )

    return map_estimate_direction


def get_2D_distance(
    position1: np.ndarray,
    position2: np.ndarray,
    track_graph: nx.Graph = None,
    edges: list = None,
    precomputed_distance: bool = True,
) -> np.ndarray:
    """Distance of two points along the graph of the track.

    Parameters
    ----------
    position1 : np.ndarray, shape (n_time, 2)
    position2 : np.ndarray, shape (n_time, 2)
    track_graph : nx.Graph or None
    edges : list or None
    precomputed_distance : bool, optional

    Returns
    -------
    distance : np.ndarray, shape (n_time,)

    """
    position1 = np.asarray(position1)
    position2 = np.asarray(position2)

    if position1.ndim < 2:
        position1 = position1[np.newaxis]
    if position2.ndim < 2:
        position2 = position2[np.newaxis]

    if track_graph is None:
        distance = np.linalg.norm(position1 - position2, axis=1)
    else:
        node_positions = nx.get_node_attributes(track_graph, "pos")
        node_ids = np.asarray(list(node_positions.keys()))
        node_positions = np.asarray(list(node_positions.values()))

        # remove outer boundary edge
        bin_edges = [e[1:-1] for e in edges]

        if precomputed_distance:
            bin_ind1 = get_bin_ind(position1, bin_edges)
            bin_ind2 = get_bin_ind(position2, bin_edges)
            distance = np.full((len(node_ids), len(node_ids)), np.inf)
            for to_node_id, from_node_id in nx.shortest_path_length(
                track_graph,
                weight="distance",
            ):
                distance[to_node_id, list(from_node_id.keys())] = list(
                    from_node_id.values()
                )
            distance = distance[bin_ind1, bin_ind2]
        else:
            node_ids1 = node_ids[get_bin_ind(position1, bin_edges)]
            node_ids2 = node_ids[get_bin_ind(position2, bin_edges)]
            distance = np.full((position1.shape[0]), np.inf)
            for i in range(position1.shape[0]):
                try:
                    distance[i] = nx.shortest_path_length(
                        track_graph,
                        source=node_ids1[i],
                        target=node_ids2[i],
                        weight="distance",
                    )
                except nx.NetworkXNoPath:
                    print(f"No path between {node_ids1[i]} and {node_ids2[i]}")

    return distance


def head_direction_simliarity(
    head_position: np.ndarray,
    head_direction: np.ndarray,
    map_estimate: np.ndarray,
    track_graph: nx.Graph = None,
    edges: list = None,
    precomputed_distance: bool = False,
) -> np.ndarray:
    """Cosine similarity of the head direction vector with the vector from the
    animal's head to MAP estimate of the decoded position.

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
    head_direction : np.ndarray, shape (n_time, 2)
    map_estimate : np.ndarray, shape (n_time, 2)
    track_graph : nx.Graph or None
    edges : list or None
    precomputed_distance : bool, optional

    Returns
    -------
    cosine_similarity : np.ndarray, shape (n_time,)

    """
    head_position = np.asarray(head_position)
    head_direction = np.asarray(head_direction)
    map_estimate = np.asarray(map_estimate)

    head_direction = head_direction.squeeze()

    if head_position.ndim < 2:
        head_position = head_position[np.newaxis]
    if map_estimate.ndim < 2:
        map_estimate = map_estimate[np.newaxis]

    if track_graph is None:
        map_estimate_direction = np.arctan2(
            map_estimate[:, 1] - head_position[:, 1],
            map_estimate[:, 0] - head_position[:, 0],
        )
    else:
        map_estimate_direction = get_map_estimate_direction_from_track_graph(
            head_position, map_estimate, track_graph, edges, precomputed_distance
        )

    return np.cos(head_direction - map_estimate_direction)


def get_ahead_behind_distance2D(
    head_position: np.ndarray,
    head_direction: np.ndarray,
    map_position: np.ndarray,
    track_graph: nx.Graph = None,
    edges: list = None,
    precomputed_distance: bool = False,
) -> np.ndarray:
    """Distance of the MAP decoded position to the animal's head position where
     the sign indicates if the decoded position is in front of the
     head (positive) or behind (negative).

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
    head_direction : np.ndarray, shape (n_time, 2)
    map_position : np.ndarray, shape (n_time, 2)
    track_graph : nx.Graph or None
    edges : list or None
    precomputed_distance : bool, optional

    Returns
    -------
    ahead_behind_distance : np.ndarray, shape (n_time,)

    """

    distance = get_2D_distance(
        head_position, map_position, track_graph, edges, precomputed_distance
    )

    direction_similarity = head_direction_simliarity(
        head_position,
        head_direction,
        map_position,
        track_graph,
        edges,
        precomputed_distance,
    )
    ahead_behind = np.sign(direction_similarity)

    # If there is no direction same point, arbitrarily set to positive
    ahead_behind[np.isclose(ahead_behind, 0.0)] = 1.0

    ahead_behind_distance = ahead_behind * distance

    return ahead_behind_distance


def _gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
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


def get_velocity(position, time=None, sigma=0.0025, sampling_frequency=500):
    if time is None:
        time = np.arange(position.shape[0])

    return _gaussian_smooth(
        np.gradient(position, time, axis=0),
        sigma,
        sampling_frequency,
        axis=0,
        truncate=8,
    )


def get_speed(velocity):
    return np.sqrt(np.sum(velocity**2, axis=1))
