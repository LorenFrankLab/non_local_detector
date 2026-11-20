"""Distances, directions and speeds of decoded positions in 2D environments."""

import logging
import warnings
from collections.abc import Sequence
from typing import cast

import networkx as nx
import numpy as np

from non_local_detector._warnings import user_stacklevel
from non_local_detector.environment import (
    Environment,
    _bin_ind_before_snap,
    make_nD_track_graph_from_environment,
)
from non_local_detector.environment import gaussian_smooth as _gaussian_smooth
from non_local_detector.graph_distances import LazyGraphDistances

__all__ = [
    "get_2D_distance",
    "get_ahead_behind_distance2D",
    "get_map_estimate_direction_from_track_graph",
    "get_speed",
    "get_velocity",
    "head_direction_similarity",
    "make_2D_track_graph_from_environment",
]


logger = logging.getLogger(__name__)


def make_2D_track_graph_from_environment(
    environment: Environment,
) -> nx.Graph:
    """Create a graph of the environment's place bins.

    Every place bin is a node (its id is the flat bin index), and the
    ``is_track_interior`` node attribute marks the on-track bins. Edges connect
    adjacent on-track bins, including diagonal neighbors.

    Parameters
    ----------
    environment : Environment
        Fitted environment whose place bins define the graph.

    Returns
    -------
    track_graph : nx.Graph
        Graph with all place bins as nodes (bin center in the ``pos``
        attribute, on-track flag in ``is_track_interior``) and edges between
        adjacent on-track bins carrying ``distance`` and ``edge_id``
        attributes.
    """
    return make_nD_track_graph_from_environment(environment)


def _warn_precomputed_distance(*, precomputed_distance: bool | None) -> None:
    """Warn that ``precomputed_distance`` was passed: it has no effect.

    Parameters
    ----------
    precomputed_distance : bool or None
        The value the caller passed; None (left out) does not warn.
    """
    if precomputed_distance is not None:
        warnings.warn(
            "precomputed_distance has no effect and will be removed; leave it "
            "out. Graph distances come from the environment's distance table, "
            "or from shortest paths computed once per distinct MAP bin.",
            FutureWarning,
            stacklevel=user_stacklevel(),
        )


def _graph_and_distances(
    track_graph: nx.Graph | None,
    edges: Sequence[np.ndarray] | None,
    environment: Environment | None,
) -> tuple[nx.Graph, Sequence[np.ndarray], np.ndarray | LazyGraphDistances]:
    """Return the bin graph, bin edges and graph-distance table of a 2-D lookup.

    Parameters
    ----------
    track_graph : nx.Graph or None
        Graph of the place bins, with nodes ``0`` through ``n_bins - 1``.
    edges : list of np.ndarray or None
        Bin edges for each spatial dimension of the grid of `track_graph`.
    environment : Environment or None
        Fitted N-D environment, used in place of `track_graph` and `edges`.

    Returns
    -------
    track_graph : nx.Graph
        Graph of the place bins.
    edges : list of np.ndarray
        Bin edges for each spatial dimension.
    distances : np.ndarray or LazyGraphDistances, shape (n_bins, n_bins)
        The environment's ``distance_between_nodes_``, or shortest-path
        lengths of `track_graph` computed for the source rows an index
        selects.

    Raises
    ------
    ValueError
        If neither or both of `environment` and `track_graph` (or `edges`)
        are given, `track_graph` comes without `edges`, or `environment` has
        no N-D bin graph.
    """
    if environment is not None:
        if track_graph is not None or edges is not None:
            msg = "Pass either environment, or track_graph and edges, not both"
            raise ValueError(msg)
        if environment.track_graphDD is None:
            msg = (
                "environment has no track_graphDD: pass an N-D environment, "
                "fitted without a track_graph"
            )
            raise ValueError(msg)
        return (
            environment.track_graphDD,
            environment.edges_,
            cast(
                "np.ndarray | LazyGraphDistances", environment.distance_between_nodes_
            ),
        )
    if track_graph is None:
        msg = "Pass track_graph and edges, or environment"
        raise ValueError(msg)
    if edges is None:
        msg = "edges parameter is required when track_graph is provided"
        raise ValueError(msg)
    # A query computes each source row once, so a row cache would only hold
    # copies until this call returns.
    return (
        track_graph,
        edges,
        LazyGraphDistances.from_graph(track_graph, max_cache_bytes=0),
    )


def _sorted_neighbors(
    track_graph: nx.Graph, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each node's neighbors in ascending node id, and the edges' distances.

    Parameters
    ----------
    track_graph : nx.Graph
        Graph whose edges have a ``distance`` attribute.
    nodes : np.ndarray of int, shape (n_nodes,)
        Nodes of `track_graph`.

    Returns
    -------
    neighbors : np.ndarray of int, shape (n_nodes, max_degree)
        The neighbors of ``nodes[i]`` in row ``i``, padded with ``nodes[i]``.
        ``max_degree`` is the largest number of neighbors, at least 1.
    edge_distance : np.ndarray, shape (n_nodes, max_degree)
        Distance of the edge to each neighbor; inf in the padding.
    """
    adjacency = [sorted(track_graph.adj[node].items()) for node in nodes.tolist()]
    degree = np.array([len(items) for items in adjacency], dtype=np.intp)
    neighbors = np.repeat(
        nodes[:, np.newaxis], max(1, int(degree.max(initial=0))), axis=1
    )
    edge_distance = np.full(neighbors.shape, np.inf)
    is_neighbor = np.arange(neighbors.shape[1]) < degree[:, np.newaxis]
    neighbors[is_neighbor] = [node for items in adjacency for node, _ in items]
    edge_distance[is_neighbor] = [
        data["distance"] for items in adjacency for _, data in items
    ]
    return neighbors, edge_distance


def _paired_distances(
    distances: np.ndarray | LazyGraphDistances,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Entries ``distances[rows[i], columns[i, j]]`` of a distance table.

    A `LazyGraphDistances` is read with ``cross_distances``, whole rows for a
    batch of distinct `rows` at a time, as many as its Dijkstra workspace
    holds. The result belongs to the caller, so the table's dense budget does
    not limit it.

    Parameters
    ----------
    distances : np.ndarray or LazyGraphDistances, shape (n_bins, n_bins)
        Distance table.
    rows : np.ndarray of int, shape (n_pairs,)
        Row (source bin) of each output row.
    columns : np.ndarray of int, shape (n_pairs, n_columns)
        Columns (destination bins) read from that row.

    Returns
    -------
    entries : np.ndarray, shape (n_pairs, n_columns)
    """
    if not isinstance(distances, LazyGraphDistances):
        return np.asarray(distances[rows[:, np.newaxis], columns])
    n_bins = distances.shape[1]
    sources, source_ind = np.unique(rows, return_inverse=True)
    batch_size = max(1, distances.max_workspace_bytes // (8 * n_bins))
    entries = np.empty(columns.shape)
    for start in range(0, len(sources), batch_size):
        block = distances.cross_distances(
            sources[start : start + batch_size], np.arange(n_bins)
        )
        in_batch = (source_ind >= start) & (source_ind < start + batch_size)
        entries[in_batch] = block[
            (source_ind[in_batch] - start)[:, np.newaxis], columns[in_batch]
        ]
    return entries


def _graph_distance_and_direction(
    position1: np.ndarray,
    position2: np.ndarray,
    track_graph: nx.Graph | None,
    edges: Sequence[np.ndarray] | None,
    environment: Environment | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Graph distance and first-step direction from `position1` toward `position2`.

    Each distinct pair of bins is looked up once. For start bin ``a`` and
    target bin ``b`` the distance is ``D[b, a]`` of the distance table ``D``,
    and the first step is the neighbor ``n`` of ``a`` that minimizes
    ``w(a, n) + D[b, n]``, ``w`` being the edge's ``distance``: the first
    exact minimum, with the neighbors in ascending node id.

    Parameters
    ----------
    position1 : np.ndarray, shape (n_time, 2)
        Start positions.
    position2 : np.ndarray, shape (n_time, 2)
        Target positions.
    track_graph : nx.Graph or None
        Graph of the place bins (see `_graph_and_distances`).
    edges : list of np.ndarray or None
        Bin edges for each spatial dimension of the grid of `track_graph`.
    environment : Environment or None
        Fitted N-D environment, used in place of `track_graph` and `edges`.

    Returns
    -------
    distance : np.ndarray, shape (n_time,)
        Shortest path length between the two bins; inf without a path.
    direction : np.ndarray, shape (n_time,)
        Direction in radians of the first step; 0 in the same bin, NaN
        without a path. Both outputs are NaN where either position has a
        non-finite coordinate.
    """
    track_graph, edges, distances = _graph_and_distances(
        track_graph, edges, environment
    )
    position1 = np.atleast_2d(position1)
    position2 = np.atleast_2d(position2)
    # A non-finite position has no bin (searchsorted would put NaN in the last
    # one), so its rows stay NaN.
    is_finite = np.all(np.isfinite(position1), axis=1) & np.all(
        np.isfinite(position2), axis=1
    )
    distance = np.full(is_finite.shape, np.nan)
    direction = np.full(is_finite.shape, np.nan)
    if not np.any(is_finite):
        return distance, direction

    # Bins of the grid that _bin_ind_before_snap indexes, so each pair has its
    # own code.
    n_bins = int(np.prod([len(edge) - 1 for edge in edges]))
    pairs, pair_ind = np.unique(
        _bin_ind_before_snap(edges, position1[is_finite]) * n_bins
        + _bin_ind_before_snap(edges, position2[is_finite]),
        return_inverse=True,
    )
    start_bin, target_bin = np.divmod(pairs, n_bins)

    neighbors, edge_distance = _sorted_neighbors(track_graph, start_bin)
    # Distances from each target bin to the start bin (column 0) and to the
    # start bin's neighbors: one shortest-path row per distinct target bin.
    to_target = _paired_distances(
        distances, target_bin, np.column_stack([start_bin, neighbors])
    )
    pair_distance = to_target[:, 0]
    first_step = neighbors[
        np.arange(len(pairs)), np.argmin(edge_distance + to_target[:, 1:], axis=1)
    ]
    is_same_bin = start_bin == target_bin
    first_step[is_same_bin] = start_bin[is_same_bin]

    is_reachable = np.isfinite(pair_distance)
    n_unreachable = int(np.sum(~is_reachable[pair_ind]))
    if n_unreachable:
        logger.warning(
            "%d time point(s) have no path on the track graph between their "
            "two positions' bins (disconnected track graph); their distance "
            "is inf and their direction NaN.",
            n_unreachable,
        )

    nodes, node_ind = np.unique(
        np.concatenate([start_bin, first_step]), return_inverse=True
    )
    node_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in nodes.tolist()]
    )
    step = node_position[node_ind[len(pairs) :]] - node_position[node_ind[: len(pairs)]]
    pair_direction = np.where(is_reachable, np.arctan2(step[:, 1], step[:, 0]), np.nan)

    distance[is_finite] = pair_distance[pair_ind]
    direction[is_finite] = pair_direction[pair_ind]
    return distance, direction


def get_map_estimate_direction_from_track_graph(
    head_position: np.ndarray,
    map_estimate: np.ndarray,
    track_graph: nx.Graph | None = None,
    edges: Sequence[np.ndarray] | None = None,
    *,
    precomputed_distance: bool | None = None,
    environment: Environment | None = None,
) -> np.ndarray:
    """Get the direction of the MAP estimate from the animal's head position.

    The direction is the heading of the first step along a shortest path on
    the track graph, from the head position's bin toward the MAP estimate's
    bin.

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
        Position of the animal's head.
    map_estimate : np.ndarray, shape (n_time, 2)
        MAP decoded position.
    track_graph : nx.Graph or None, optional
        Graph of the environment's place bins (``Environment.track_graphDD``),
        whose nodes are the flat bin indices. Give it with `edges`, or give
        `environment` instead. By default None.
    edges : list of np.ndarray or None, optional
        Bin edges for each spatial dimension (``Environment.edges_``) of the
        grid whose bins are the `track_graph` nodes. Required with
        `track_graph`. By default None.
    precomputed_distance : bool or None, optional
        Deprecated, and has no effect: any value other than None warns with a
        ``FutureWarning``. It will be removed. By default None.
    environment : Environment or None, optional
        Fitted N-D environment (fitted without a ``track_graph``). Its bin
        graph, bin edges and distance table (``distance_between_nodes_``) are
        used in place of `track_graph` and `edges`. By default None.

    Returns
    -------
    map_estimate_direction : np.ndarray, shape (n_time,)
        Direction in radians of the first step along the shortest path from the
        head position to the MAP estimate. 0 where the head position and the MAP
        estimate are in the same bin. NaN where the track graph has no path
        between their bins (a warning is logged), and where either position
        has a non-finite coordinate.

    Raises
    ------
    ValueError
        If neither or both of `environment` and `track_graph` are given,
        `track_graph` comes without `edges`, or `environment` has no N-D bin
        graph (it was fitted on a track graph).

    Notes
    -----
    For head bin ``a`` and MAP bin ``b``, the first step is the neighbor
    ``n`` of ``a`` that minimizes ``w(a, n) + D(b, n)``, where ``w`` is the
    edge's ``distance`` and ``D`` the shortest path length. ``D`` is read from
    the environment's table; with `track_graph`, it is computed once from
    each distinct MAP bin (scipy Dijkstra). When several neighbors give the
    same smallest value (two shortest paths), the step goes to the one with
    the lowest node id: the first exact minimum, with the neighbors in
    ascending node id.
    """
    _warn_precomputed_distance(precomputed_distance=precomputed_distance)
    return _graph_distance_and_direction(
        head_position, map_estimate, track_graph, edges, environment
    )[1]


def get_2D_distance(
    position1: np.ndarray,
    position2: np.ndarray,
    track_graph: nx.Graph = None,
    edges: Sequence[np.ndarray] | None = None,
    *,
    precomputed_distance: bool | None = None,
    environment: Environment | None = None,
) -> np.ndarray:
    """Distance of two points along the graph of the track.

    Parameters
    ----------
    position1 : np.ndarray, shape (n_time, 2)
        First set of positions.
    position2 : np.ndarray, shape (n_time, 2)
        Second set of positions.
    track_graph : nx.Graph or None
        Graph of the environment's place bins (e.g.
        ``Environment.track_graphDD``). If None, and `environment` is None,
        distances are Euclidean.
    edges : list of np.ndarray or None
        Bin edges for each spatial dimension (``Environment.edges_``) of the
        grid whose bins are the ``track_graph`` nodes. Required when
        ``track_graph`` is given; ignored otherwise.
    precomputed_distance : bool or None, optional
        Deprecated, and has no effect: any value other than None warns with a
        ``FutureWarning``. It will be removed. By default None.
    environment : Environment or None, optional
        Fitted N-D environment (fitted without a ``track_graph``). Its bin
        graph, bin edges and distance table (``distance_between_nodes_``) are
        used in place of `track_graph` and `edges`. By default None.

    Returns
    -------
    distance : np.ndarray, shape (n_time,)
        Distance between the two positions at each time point. Euclidean
        without a graph, otherwise the shortest path length along the graph
        between their bins (``inf`` if no path exists): read from the
        environment's table, or with `track_graph` computed once from each
        distinct bin of `position2`. With a graph, NaN where either position
        has a non-finite coordinate.
    """
    _warn_precomputed_distance(precomputed_distance=precomputed_distance)
    position1 = np.asarray(position1)
    position2 = np.asarray(position2)

    if position1.ndim < 2:
        position1 = position1[np.newaxis]
    if position2.ndim < 2:
        position2 = position2[np.newaxis]

    if track_graph is None and environment is None:
        distance = np.linalg.norm(position1 - position2, axis=1)
    else:
        distance, _ = _graph_distance_and_direction(
            position1, position2, track_graph, edges, environment
        )

    return cast("np.ndarray", distance)


def head_direction_similarity(
    head_position: np.ndarray,
    head_direction: np.ndarray,
    map_estimate: np.ndarray,
    track_graph: nx.Graph = None,
    edges: Sequence[np.ndarray] | None = None,
    *,
    precomputed_distance: bool | None = None,
    environment: Environment | None = None,
) -> np.ndarray:
    """Cosine similarity of the head direction with the direction to the MAP.

    Compares the head direction with the vector from the animal's head to the
    MAP estimate of the decoded position, or with a graph, with the first
    step of a shortest path toward it (see
    `get_map_estimate_direction_from_track_graph`).

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
        Position of the animal's head.
    head_direction : np.ndarray, shape (n_time,)
        Heading angle in radians (e.g. ``np.arctan2(v[:, 1], v[:, 0])`` for a
        heading vector ``v``), not a 2-D vector.
    map_estimate : np.ndarray, shape (n_time, 2)
        MAP decoded position.
    track_graph : nx.Graph or None
        Graph of the environment's place bins (e.g.
        ``Environment.track_graphDD``). If None, and `environment` is None,
        the direction to the MAP estimate is a straight line.
    edges : list of np.ndarray or None
        Bin edges for each spatial dimension (``Environment.edges_``) of the
        grid whose bins are the ``track_graph`` nodes. Required when
        ``track_graph`` is given; ignored otherwise.
    precomputed_distance : bool or None, optional
        Deprecated, and has no effect: any value other than None warns with a
        ``FutureWarning``. It will be removed. By default None.
    environment : Environment or None, optional
        Fitted N-D environment (fitted without a ``track_graph``). Its bin
        graph, bin edges and distance table (``distance_between_nodes_``) are
        used in place of `track_graph` and `edges`. By default None.

    Returns
    -------
    cosine_similarity : np.ndarray, shape (n_time,)
        Cosine of the angle between the head direction and the direction to the
        MAP estimate; 1 means directly ahead, -1 directly behind.
    """
    _warn_precomputed_distance(precomputed_distance=precomputed_distance)
    head_position = np.asarray(head_position)
    head_direction = np.asarray(head_direction)
    map_estimate = np.asarray(map_estimate)

    head_direction = head_direction.squeeze()

    if head_position.ndim < 2:
        head_position = head_position[np.newaxis]
    if map_estimate.ndim < 2:
        map_estimate = map_estimate[np.newaxis]

    if track_graph is None and environment is None:
        map_estimate_direction = np.arctan2(
            map_estimate[:, 1] - head_position[:, 1],
            map_estimate[:, 0] - head_position[:, 0],
        )
    else:
        map_estimate_direction = get_map_estimate_direction_from_track_graph(
            head_position,
            map_estimate,
            track_graph,
            edges,
            environment=environment,
        )

    return cast("np.ndarray", np.cos(head_direction - map_estimate_direction))


def get_ahead_behind_distance2D(
    head_position: np.ndarray,
    head_direction: np.ndarray,
    map_position: np.ndarray,
    track_graph: nx.Graph = None,
    edges: Sequence[np.ndarray] | None = None,
    *,
    precomputed_distance: bool | None = None,
    environment: Environment | None = None,
) -> np.ndarray:
    """Signed distance of the MAP decoded position from the animal's head.

    The sign indicates if the decoded position is in front of the head
    (positive) or behind (negative).

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
        Position of the animal's head.
    head_direction : np.ndarray, shape (n_time,)
        Heading angle in radians (e.g. ``np.arctan2(v[:, 1], v[:, 0])`` for a
        heading vector ``v``), not a 2-D vector.
    map_position : np.ndarray, shape (n_time, 2)
        MAP decoded position.
    track_graph : nx.Graph or None
        Graph of the environment's place bins (e.g.
        ``Environment.track_graphDD``). If None, and `environment` is None,
        distances are Euclidean.
    edges : list of np.ndarray or None
        Bin edges for each spatial dimension (``Environment.edges_``) of the
        grid whose bins are the ``track_graph`` nodes. Required when
        ``track_graph`` is given; ignored otherwise.
    precomputed_distance : bool or None, optional
        Deprecated, and has no effect: any value other than None warns with a
        ``FutureWarning``. It will be removed. By default None.
    environment : Environment or None, optional
        Fitted N-D environment (fitted without a ``track_graph``). Its bin
        graph, bin edges and distance table (``distance_between_nodes_``) are
        used in place of `track_graph` and `edges`. By default None.

    Returns
    -------
    ahead_behind_distance : np.ndarray, shape (n_time,)
        Signed distance (positive ahead of the head, negative behind), in the
        units of the position. With a graph, the distance is that of
        `get_2D_distance` and the sign compares the heading with the first
        step of a shortest path (see
        `get_map_estimate_direction_from_track_graph`), both from one lookup;
        NaN where either position has a non-finite coordinate or the graph
        has no path between the two bins.
    """
    _warn_precomputed_distance(precomputed_distance=precomputed_distance)
    if track_graph is None and environment is None:
        distance = get_2D_distance(head_position, map_position)
        direction_similarity = head_direction_similarity(
            head_position, head_direction, map_position
        )
    else:
        distance, map_direction = _graph_distance_and_direction(
            np.asarray(head_position),
            np.asarray(map_position),
            track_graph,
            edges,
            environment,
        )
        direction_similarity = np.cos(
            np.asarray(head_direction).squeeze() - map_direction
        )
    ahead_behind = np.sign(direction_similarity)

    # If there is no direction same point, arbitrarily set to positive
    ahead_behind[np.isclose(ahead_behind, 0.0)] = 1.0

    return cast("np.ndarray", ahead_behind * distance)


def get_velocity(
    position: np.ndarray,
    time: np.ndarray | None = None,
    sigma: float = 0.0025,
    sampling_frequency: float = 500,
) -> np.ndarray:
    """Estimate the velocity of the decode.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
        MAP decoded position.
    time : np.ndarray, shape (n_time,), optional
        Time of each sample used to take the gradient. If None, sample indices
        are used and the velocity is in position units per sample; otherwise it
        is in position units per unit of ``time``. By default None.
    sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel in seconds,
        by default 0.0025. Near the first and last samples the kernel is
        renormalized over the samples that exist.
    sampling_frequency : float, optional
        Sampling frequency of the data in Hz, by default 500.

    Returns
    -------
    decoded_velocity : np.ndarray, shape (n_time, n_position_dims)
        Smoothed velocity of the decoded position.
    """
    if time is None:
        time = np.arange(position.shape[0])

    velocity = np.gradient(position, time, axis=0)
    # Divide by the kernel mass inside the array: zero padding alone would pull
    # the first and last samples toward zero.
    return cast(
        "np.ndarray",
        _gaussian_smooth(velocity, sigma, sampling_frequency, axis=0, truncate=8)
        / _gaussian_smooth(
            np.ones_like(velocity), sigma, sampling_frequency, axis=0, truncate=8
        ),
    )


def get_speed(velocity: np.ndarray) -> np.ndarray:
    """Estimate the speed of the decoded position.

    Parameters
    ----------
    velocity : np.ndarray, shape (n_time, n_position_dims)
        Velocity of the decoded position.

    Returns
    -------
    speed : np.ndarray, shape (n_time,)
        Euclidean norm of the velocity at each time point.
    """
    return cast("np.ndarray", np.sqrt(np.sum(velocity**2, axis=1)))
