"""
Utility functions for graph-based (linearized track) layouts. 📐

This module provides helper functions specifically for environments where the
spatial layout is defined by a graph structure that is subsequently linearized,
such as an animal's track in an experiment. These functions handle:

- Discretizing a linearized graph track into 1D bins, accounting for edge
  lengths and specified spacing between segments (`_get_graph_bins`).
- Creating a connectivity graph from these binned segments, where nodes
  represent active 1D bin centers and edges connect adjacent bins
  (`_create_graph_layout_connectivity_graph`).
- Projecting 1D linearized positions back to their corresponding N-D
  coordinates on the original track graph (`_project_1d_to_2d`).
- Finding the appropriate 1D bin index for a given linear position
  (`_find_bin_for_linear_position`).

These utilities are primarily used by the `GraphLayout` engine.
"""

import math
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np

from non_local_detector.environment.utils import get_centers

Edge = Tuple[Any, Any]


def _get_graph_bins(
    graph: nx.Graph,
    edge_order: List[Tuple[object, object]],
    edge_spacing: Union[float, List[float]],
    bin_size: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Discretize a graph into 1D bins along a specified path, accounting for spacing.

    Calculates 1D bin centers and edges for a track defined by `edge_order`
    from the `graph`. It considers the length of each edge and any specified
    `edge_spacing` (gaps) between consecutive edges in the path.

    Parameters
    ----------
    graph : networkx.Graph
        The graph defining the track. Nodes are expected to have a 'pos'
        attribute (e.g., `graph.nodes[node_id]['pos'] = (x, y)`).
    edge_order : list[tuple[object, object]]
        Ordered sequence of edge tuples (node_id_1, node_id_2) defining
        the linearized track segments.
    edge_spacing : float | list[float]
        Spacing (gap length) between consecutive edges in `edge_order`.
        If a float, applies uniformly to all gaps.
        If a list, specifies spacing for each gap; its length must be
        `len(edge_order) - 1`.
    bin_size : float
        Desired length of each bin along the linearized track. Must be positive.

    Returns
    -------
    bin_centers_1d : np.ndarray, shape (n_total_bins,)
        1D coordinates of the center of each bin along the linearized track,
        including bins within gaps (which will be marked inactive).
    bin_edges_1d_tuple : tuple[np.ndarray, ...]
        A tuple containing a single 1D NumPy array of bin edge coordinates
        (shape `(n_total_bins + 1,)`). This structure matches `np.histogramdd` output.
    active_mask_1d : np.ndarray, shape (n_total_bins,), dtype=bool
        Boolean mask indicating active bins (True, on an actual edge segment)
        vs. inactive bins (False, within a gap).
    edge_ids_for_active_bins : np.ndarray, shape (n_active_bins,), dtype=int
        Integer IDs corresponding to the original graph edges (from `graph.edges()`)
        for each *active* bin. `n_active_bins = np.sum(active_mask_1d)`.

    Raises
    ------
    ValueError
        If any edge in `edge_order` contains nodes not in `graph`.
        If `bin_size` is not positive.
        If `edge_spacing` (if a list) has an incorrect length.
    """
    # Verify the nodes are in the graph
    for edge in edge_order:
        if not graph.has_node(edge[0]) or not graph.has_node(edge[1]):
            raise ValueError(f"Edge {edge} contains nodes not in the graph")

    # Verify the bin size is positive
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    # Figure out the gap
    n_edges = len(edge_order)
    if isinstance(edge_spacing, (int, float)):
        # Ensure gaps are float, handles n_edges=0 or 1 correctly
        gaps = np.full(max(0, n_edges - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
        if gaps.size != max(0, n_edges - 1):
            raise ValueError(f"edge_spacing list length must be {max(0, n_edges - 1)}")

    node_positions = nx.get_node_attributes(graph, "pos")
    # Create a mapping from edge tuple to a unique integer ID
    # This handles undirected nature: (u,v) and (v,u) map to the same ID if
    # graph.edges stores them consistently or if we normalize.
    # For simplicity, assuming graph.edges provides unique edge representations.
    edge_id_map: Dict[Tuple[object, object], int] = {
        edge: i for i, edge in enumerate(graph.edges())
    }
    bin_edges = []
    active_mask = []
    edge_ids = []
    linear_start_pos = 0.0
    for i, (start_node, end_node) in enumerate(edge_order):
        start_node_pos = np.asarray(node_positions[start_node])
        end_node_pos = np.asarray(node_positions[end_node])
        edge_length = np.linalg.norm(end_node_pos - start_node_pos)

        n_bins = max(1, np.ceil(edge_length / bin_size).astype(int))
        bin_edges.extend(
            np.linspace(
                linear_start_pos,
                linear_start_pos + edge_length,
                n_bins + 1,
                endpoint=True,
            )
        )
        try:
            edge_ids.extend([edge_id_map[(start_node, end_node)]] * n_bins)
        except KeyError:
            edge_ids.extend([edge_id_map[(end_node, start_node)]] * n_bins)

        active_mask.extend([True] * n_bins)
        linear_start_pos += edge_length
        # Add the gap
        if i < n_edges - 1:
            if gaps[i] > 0:
                linear_start_pos += gaps[i]
                active_mask.append(False)

    # Add the last bin edge
    bin_edges.append(linear_start_pos)
    # Convert to numpy array
    bin_edges = np.array(bin_edges)
    bin_edges = (np.unique(bin_edges),)
    bin_centers = get_centers(bin_edges[0])
    active_mask = np.array(active_mask, dtype=bool)
    edge_ids = np.array(edge_ids, dtype=int)

    return bin_centers, bin_edges, active_mask, edge_ids


def _create_graph_layout_connectivity_graph(
    graph: nx.Graph,
    bin_centers_nd: np.ndarray,
    linear_bin_centers: np.ndarray,
    original_edge_ids: np.ndarray,
    active_mask: np.ndarray,
    edge_order: List[Tuple[object, object]],
) -> nx.Graph:
    """Create a connectivity graph from binned graph segments.

    Nodes in the connectivity graph represent centers of active bins.
    Edges connect adjacent bins, both within the same original graph edge
    (intra-segment) and between bins at the a_junction of original graph edges
    (inter-segment).

    Parameters
    ----------
    graph : networkx.Graph
        The original graph from which bins were derived.
    bin_centers_nd : np.ndarray, shape (n_active_bins, 2)
        n-D coordinates of the centers of active bins.
    linear_bin_centers : np.ndarray, shape (n_total_bins,)
        1D coordinates of all bin centers (active and inactive).
    original_edge_ids : np.ndarray, shape (n_active_bins,)
        Integer IDs of the original graph edge for each active bin.
    active_mask : np.ndarray, shape (n_total_bins,), dtype=bool
        Mask indicating which bins in `linear_bin_centers` are active.
    edge_order : list of tuples
        The ordered sequence of edges from the original graph that defines
        the linearized track layout. Used for inter-segment connections.

    Returns
    -------
    networkx.Graph
        A new graph where nodes are indexed `0` to `n_active_bins - 1`.
        Node attributes:
        - 'pos': tuple (x, y, ...), N-D position of the active bin center.
        - 'source_grid_flat_index': int, Original flat index of this bin in
          the `linear_bin_centers_all` array (the 1D "grid").
        - 'original_grid_nd_index': tuple (int,), N-D (here, 1D) index in
          the `linear_bin_centers_all` array.
        - 'pos_1D': float, 1D linearized position of the active bin center.
        - 'source_edge_id': int, ID of the original graph edge this bin is on.
        Edge attributes:
        - 'distance': float, Euclidean distance between connected N-D bin centers.
        - 'weight': float, Typically same as 'distance'.
        - 'vector': tuple, Displacement vector in N-D.
        - 'angle_2d': float, (For 2D) angle of the displacement vector.
        - 'edge_id': int, Unique ID for this edge within the connectivity graph.
    """
    # Create a new graph
    connectivity_graph = nx.Graph()

    # --- 1. Add Nodes (representing active bin centers) ---
    nodes_to_add = []

    bin_ind = np.arange(len(linear_bin_centers))

    # Add active bin centers to the graph
    for node_id, (center_2D, center_1D, original_edge_id, b_ind) in enumerate(
        zip(
            bin_centers_nd[active_mask],
            linear_bin_centers[active_mask],
            original_edge_ids,
            bin_ind[active_mask],
        )
    ):
        nodes_to_add.append(
            (
                int(node_id),
                {
                    "pos": tuple(center_2D),
                    "source_grid_flat_index": b_ind,
                    "original_grid_nd_index": np.unravel_index(
                        b_ind, linear_bin_centers.shape
                    ),
                    "pos_1D": center_1D,
                    "source_edge_id": original_edge_id,
                },
            )
        )
    connectivity_graph.add_nodes_from(nodes_to_add)

    # --- 2. Add Intra-Segment Edges (connecting bins on the same original edge) ---
    edges_to_add = []
    _, sort_ind = np.unique(original_edge_ids, return_index=True)
    unsorted_unique_edge_ids = [int(original_edge_ids[ind]) for ind in sorted(sort_ind)]
    bin_edge_order = []
    for original_edge_id in unsorted_unique_edge_ids:
        edge_active_bin_ind = np.where(np.isin(original_edge_ids, original_edge_id))[0]

        for bin_ind1, bin_ind2 in zip(
            edge_active_bin_ind[:-1], edge_active_bin_ind[1:]
        ):
            displacement_vector = (
                bin_centers_nd[active_mask][bin_ind1]
                - bin_centers_nd[active_mask][bin_ind2]
            )
            dist = float(np.linalg.norm(displacement_vector))
            weight = 1 / dist if dist > 0 else np.inf

            edges_to_add.append(
                (
                    int(bin_ind1),
                    int(bin_ind2),
                    {
                        "distance": dist,
                        "weight": float(weight),
                        "vector": tuple(displacement_vector.tolist()),
                        "angle_2d": math.atan2(
                            displacement_vector[1], displacement_vector[0]
                        ),
                    },
                )
            )

        bin_edge_order.append(
            (int(edge_active_bin_ind[0]), int(edge_active_bin_ind[-1]))
        )
    connectivity_graph.add_edges_from(edges_to_add)
    bin_edge_order = np.asarray(bin_edge_order)

    # --- 3. Add Inter-Segment Edges (connecting end/start bins of adjacent original edges) ---
    edge_order_array = np.asarray(edge_order)
    bins_to_connect = []
    for node_id in graph.nodes:
        connections = bin_edge_order[edge_order_array == node_id]
        if len(connections) > 1:
            for i in range(len(connections) - 1):
                displacement_vector = (
                    bin_centers_nd[active_mask][connections[i]]
                    - bin_centers_nd[active_mask][connections[i + 1]]
                )
                bins_to_connect.append(
                    (
                        int(connections[i]),
                        int(connections[i + 1]),
                        {
                            "distance": float(np.linalg.norm(displacement_vector)),
                            "vector": tuple(displacement_vector.tolist()),
                            "angle_2d": math.atan2(
                                displacement_vector[1], displacement_vector[0]
                            ),
                        },
                    )
                )
    connectivity_graph.add_edges_from(bins_to_connect)

    # Add edge IDs to the graph
    # This is a unique ID for each edge in the graph, starting from 0
    # and incrementing by 1 for each edge
    for edge_id_counter, (u, v) in enumerate(connectivity_graph.edges()):
        connectivity_graph.edges[u, v]["edge_id"] = edge_id_counter

    return connectivity_graph


def _project_1d_to_2d(
    linear_position: np.ndarray,
    graph: nx.Graph,
    edge_order: List[Edge],
    edge_spacing: Union[float, List[float]] = 0.0,
) -> np.ndarray:
    """
    Map 1D linear positions back to N-D coordinates on the track graph.

    Projects points from a 1D linearized representation of the track (defined
    by `graph`, `edge_order`, and `edge_spacing`) back to their original
    N-dimensional spatial coordinates.

    Parameters
    ----------
    linear_position : np.ndarray, shape (n_points,)
        1D positions along the linearized track.
    graph : networkx.Graph
        The original graph. Nodes must have a 'pos' attribute (N-D coordinates)
        and edges must have a 'distance' attribute (pre-calculated length).
    edge_order : list[tuple[node, node]]
        Ordered sequence of edges defining the linearization path.
    edge_spacing : float or list[float], optional, default=0.0
        Spacing (gaps) between track segments used during linearization.
        If float, applied uniformly. If list, length must be `len(edge_order) - 1`.

    Returns
    -------
    coords_nd : np.ndarray, shape (n_points, n_dims)
        N-dimensional coordinates corresponding to each input `linear_position`.
        Positions falling into gaps or beyond track ends are typically
        projected onto the nearest valid track segment endpoint.
        NaNs in `linear_position` propagate to rows of NaNs in output.

    Raises
    ------
    ValueError
        If `linear_position` is not 1D.
        If `edge_spacing` (if a list) has an incorrect length.
    """
    linear_position = np.asarray(linear_position, dtype=float)
    if linear_position.ndim == 0:
        linear_position = linear_position.reshape((1,))
    elif linear_position.ndim > 1:
        # Attempt to squeeze if it's like (N,1) or (1,N)
        squeezed_pos = np.squeeze(linear_position)
        if squeezed_pos.ndim == 1:
            linear_position = squeezed_pos
        elif squeezed_pos.ndim == 0 and linear_position.size == 1:  # e.g. from [[5.]]
            linear_position = squeezed_pos.reshape((1,))
        else:  # If it's still not 1D (e.g. (M,N) where M,N > 1)
            raise ValueError("linear_position must be convertible to a 1D array.")
    if linear_position.ndim != 1:
        raise ValueError("linear_position must be a 1D array.")
    n_edges = len(edge_order)

    # --- edge lengths & spacing ------------------------------------------------
    edge_lengths = np.array(
        [graph.edges[e]["distance"] for e in edge_order], dtype=float
    )

    if isinstance(edge_spacing, (int, float)):
        gaps = np.full(max(0, n_edges - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
        if gaps.size != max(0, n_edges - 1):
            raise ValueError("edge_spacing length must be len(edge_order)‑1")

    # cumulative start position of each edge
    cumulative_edge_start_position = np.concatenate(
        [[0.0], np.cumsum(edge_lengths[:-1] + gaps)]
    )  # shape (n_edges,)

    edge_ind = (
        np.searchsorted(cumulative_edge_start_position, linear_position, side="right")
        - 1
    )
    edge_ind = np.clip(edge_ind, 0, n_edges - 1)  # clamp to valid edge index

    nan_mask = ~np.isfinite(linear_position).squeeze()
    edge_ind[nan_mask] = 0  # dummy index, will overwrite later

    # param along each chosen edge
    normalized_edge_position = (
        linear_position - cumulative_edge_start_position[edge_ind]
    ) / edge_lengths[edge_ind]
    normalized_edge_position = np.clip(
        normalized_edge_position, 0.0, 1.0
    )  # project extremes onto endpoints

    # gather endpoint coordinates
    node_position_2D = nx.get_node_attributes(graph, "pos")
    start_node_position_2D = np.array(
        [node_position_2D[edge_order[int(i)][0]] for i in edge_ind]
    )
    end_node_position_2D = np.array(
        [node_position_2D[edge_order[int(i)][1]] for i in edge_ind]
    )

    # Linear interpolation between endpoints
    position_2D = (
        1.0 - normalized_edge_position[:, None]
    ) * start_node_position_2D + normalized_edge_position[
        :, None
    ] * end_node_position_2D

    # propagate NaNs from the input
    position_2D[nan_mask] = np.nan

    return position_2D


def _find_bin_for_linear_position(
    linear_positions: Union[float, np.ndarray],
    bin_edges: np.ndarray,
    active_mask: np.ndarray = None,
) -> Union[int, np.ndarray]:
    """
    Find the 1D bin index for each given linear position.

    Assigns each position in `linear_positions` to a bin defined by
    `bin_edges_1d`. If `active_mask_1d` is provided, positions falling into
    inactive bins (gaps) can be handled (e.g., by adjusting to a nearby
    active bin or returning an invalid index, depending on implementation details).
    The current implementation maps to bins based on `np.digitize` and handles
    positions on edges that might fall into inactive bins by shifting them to
    the preceding active bin.

    Parameters
    ----------
    linear_positions : float | np.ndarray, shape (n_points,)
        Linear position(s) to map to bin indices.
    bin_edges_1d : np.ndarray, shape (n_total_bins + 1,)
        1D array of sorted bin edge coordinates for the entire linearized track
        (including gaps).
    active_mask_1d : Optional[np.ndarray], shape (n_total_bins,), dtype=bool, optional
        Boolean mask indicating which bins in the `bin_edges_1d` definition
        are active (on an edge segment) vs. inactive (in a gap). If provided,
        the function ensures returned indices correspond to active bins or
        handles points in gaps.

    Returns
    -------
    bin_indices : int | np.ndarray, shape (n_points,)
        0-based bin index (relative to the full set of bins defined by
        `bin_edges_1d`) for each input linear position.
        Returns -1 if a position falls outside the range of `bin_edges_1d`.
        If `active_mask_1d` is used, logic attempts to assign to an active bin.

    Raises
    ------
    ValueError
        If `active_mask_1d` (if provided) length doesn't match the number of bins.
        If, after processing, a position maps to an index marked as invalid by
        `active_mask_1d` (indicating an issue with gap handling logic or input).
    """
    was_scalar = np.isscalar(linear_positions)
    linear_positions = np.atleast_1d(np.asarray(linear_positions, dtype=float))
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:
        # No bins defined
        return -1 if was_scalar else np.full(linear_positions.shape, -1, dtype=int)

    if active_mask is not None and len(active_mask) != n_bins:
        raise ValueError(
            f"active_mask length ({len(active_mask)}) must match number of bins ({n_bins})."
        )

    bin_ind = np.digitize(linear_positions, bin_edges, right=False) - 1
    out_of_bounds_mask = (bin_ind < 0) | (bin_ind >= n_bins)
    bin_ind[out_of_bounds_mask] = -1

    if active_mask is not None:
        invalid_bin_ind = np.nonzero(~active_mask)[0]
    else:
        invalid_bin_ind = np.array([])

    in_gap_ind = np.nonzero(np.isin(bin_ind, invalid_bin_ind))[0]
    is_bin_edge = np.asarray(
        [
            np.any(np.isclose(bin_edges[1:], lin_pos))
            for lin_pos in linear_positions[in_gap_ind]
        ],
        dtype=bool,
    )
    bin_ind[in_gap_ind[is_bin_edge]] = bin_ind[in_gap_ind[is_bin_edge]] - 1

    bin_ind[in_gap_ind[~is_bin_edge]] = -1
    return int(bin_ind[0]) if was_scalar else bin_ind.astype(int)
