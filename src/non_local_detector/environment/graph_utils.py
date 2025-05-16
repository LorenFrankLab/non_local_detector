import math
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from track_linearization.core import _calculate_linear_position

from non_local_detector.environment.utils import get_centers

Edge = Tuple[Any, Any]


def _get_graph_bins(
    graph: nx.Graph,
    edge_order: List[Tuple[object, object]],
    edge_spacing: float | List[float],
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the 1D bins of a graph accounting for edge spacing.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be binned. Node positions are expected to be stored
        as a 'pos' attribute, e.g., graph.nodes[node_id]['pos'] = (x, y).
    edge_order : list of tuples
        The ordered sequence of edges (node_id_1, node_id_2) defining the
        linearized track.
    edge_spacing : float or list of float
        If float, the spacing between all consecutive edges in `edge_order`.
        If list, the spacing between each pair of consecutive edges.
        Length must be `len(edge_order) - 1`.
    bin_size : float
        The desired size of each bin along the track.

    Returns
    -------
    bin_centers : np.ndarray, shape (N_total_bins,)
        1D coordinates of the center of each bin (including active and inactive bins).
    bin_edges : np.ndarray, shape (N_total_bins + 1,)
        1D coordinates of the edges of each bin.
    active_mask : np.ndarray, shape (N_total_bins,), dtype=bool
        Boolean mask indicating active bins (True, on an edge) vs.
        inactive bins (False, in a gap).
    edge_ids : np.ndarray, shape (N_active_bins,), dtype=int
        Integer IDs corresponding to the original graph edges for each *active* bin.
        `N_active_bins = np.sum(active_mask)`.
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
    bin_edges = [np.unique(bin_edges)]
    bin_centers = get_centers(bin_edges[0])
    active_mask = np.array(active_mask, dtype=bool)
    edge_ids = np.array(edge_ids, dtype=int)

    return bin_centers, bin_edges, active_mask, edge_ids


def _create_graph_layout_connectivity_graph(
    graph: nx.Graph,
    bin_centers_2D: np.ndarray,
    bin_centers_1D: np.ndarray,
    edge_ids: np.ndarray,
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
    bin_centers_2D : np.ndarray, shape (N_sum_active_mask, 2)
        2D coordinates of the centers of active bins.
        Obtained by `projection_function(bin_centers_1D[active_mask])`.
    bin_centers_1D : np.ndarray, shape (N_total_bins,)
        1D coordinates of all bin centers (active and inactive).
    edge_ids : np.ndarray, shape (N_sum_active_mask,)
        Integer IDs of the original graph edge for each active bin.
    active_mask : np.ndarray, shape (N_total_bins,), dtype=bool
        Mask indicating which bins in `bin_centers_1D` are active.
    edge_order : list of tuples
        The ordered sequence of edges from the original graph that defines
        the linearized track layout. Used for inter-segment connections.

    Returns
    -------
    networkx.Graph
        A new graph where nodes are active bin centers and edges show connectivity.
        Node attributes include:
        - 'pos': tuple (x, y), 2D position of the bin center.
        - 'bin_ind': int, original index of the bin in `bin_centers_1D`.
        - 'bin_ind_flat': int, same as `bin_ind`.
        - 'pos_1D': float, 1D position of the bin center.
        - 'edge_id': int, ID of the original graph edge this bin belongs to.
        Edge attributes include:
        - 'dist': float, Euclidean distance between connected bin centers.
        - 'edge_id': int, (for intra-segment edges) ID of the original edge.
    """
    # Create a new graph
    connectivity_graph = nx.Graph()

    # --- 1. Add Nodes (representing active bin centers) ---
    nodes_to_add = []

    bin_ind = np.arange(len(bin_centers_1D))

    # Add active bin centers to the graph
    for node_id, (center_2D, center_1D, edge_id, b_ind) in enumerate(
        zip(
            bin_centers_2D[active_mask],
            bin_centers_1D[active_mask],
            edge_ids,
            bin_ind[active_mask],
        )
    ):
        nodes_to_add.append(
            (
                int(node_id),
                {
                    "pos": tuple(center_2D),
                    "bin_ind": b_ind,
                    "bin_ind_flat": b_ind,
                    "pos_1D": center_1D,
                    "edge_id": edge_id,
                },
            )
        )
    connectivity_graph.add_nodes_from(nodes_to_add)

    # --- 2. Add Intra-Segment Edges (connecting bins on the same original edge) ---
    edges_to_add = []
    _, sort_ind = np.unique(edge_ids, return_index=True)
    unsorted_unique_edge_ids = [int(edge_ids[ind]) for ind in sorted(sort_ind)]
    bin_edge_order = []
    for edge_id in unsorted_unique_edge_ids:
        edge_active_bin_ind = np.where(np.isin(edge_ids, edge_id))[0]
        for bin_ind1, bin_ind2 in zip(
            edge_active_bin_ind[:-1], edge_active_bin_ind[1:]
        ):
            displacement_vector = (
                bin_centers_2D[active_mask][bin_ind1]
                - bin_centers_2D[active_mask][bin_ind2]
            )

            edges_to_add.append(
                (
                    int(bin_ind1),
                    int(bin_ind2),
                    {
                        "distance": float(np.linalg.norm(displacement_vector)),
                        "vector": tuple(displacement_vector.tolist()),
                        "edge_id": int(edge_id),
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
                    bin_centers_2D[active_mask][connections[i]]
                    - bin_centers_2D[active_mask][connections[i + 1]]
                )
                bins_to_connect.append(
                    (
                        int(connections[i]),
                        int(connections[i + 1]),
                        {
                            "distance": float(np.linalg.norm(displacement_vector)),
                            "vector": tuple(displacement_vector.tolist()),
                            "edge_id": int(edge_id),
                            "angle_2d": math.atan2(
                                displacement_vector[1], displacement_vector[0]
                            ),
                        },
                    )
                )
    connectivity_graph.add_edges_from(bins_to_connect)

    return connectivity_graph


def _project_1d_to_2d(
    linear_position: np.ndarray,
    graph: nx.Graph,
    edge_order: List[Edge],
    edge_spacing: Union[float, List[float]] = 0.0,
) -> np.ndarray:
    """
    Map 1-D linear positions back to 2-D coordinates on the track graph.

    Parameters
    ----------
    linear_position : np.ndarray, shape (n_time,)
    graph : networkx.Graph
        Same graph you passed to `get_linearized_position`.
        Nodes must have `"pos"`; edges must have `"distance"`.
    edge_order : list[tuple(node, node)]
        Same order you used for linearization.
    edge_spacing : float or list of float, optional
        Controls the spacing between track segments in 1D position.
        If float, applied uniformly. If list, length must be `len(edge_order) - 1`.

    Returns
    -------
    coords : np.ndarray, shape (n_time, n_space)
        2-D (or 3-D) coordinates corresponding to each 1-D input.
        Positions that fall beyond the last edge are clipped to the last node.
        NaNs in `linear_position` propagate to rows of NaNs.
    """
    linear_position = np.asarray(linear_position, dtype=float)
    try:
        linear_position = np.squeeze(linear_position, axis=-1)
    except ValueError:  # If linear_position is already 1D, this will raise an error
        pass
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
    """Find the bin index for each linear position.

    Parameters
    ----------
    linear_positions : float or np.ndarray
        Linear positions to find the corresponding bin index.
    bin_edges : np.ndarray
        Array of bin edges.
    active_mask : np.ndarray, optional, shape (n_bins,)
        Mask to filter the bins. If provided, only bins where the mask is True
        will be considered for the search. The mask should be of the same length
        as the number of bins.
        If not provided, all bins will be considered.

    Returns
    -------
    bin_indices : int or np.ndarray
        Bin indices corresponding to the input linear positions.
        If a position falls outside the range of bin edges, returns -1.
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
        ]
    )
    bin_ind[in_gap_ind[is_bin_edge]] = bin_ind[in_gap_ind[is_bin_edge]] - 1

    if np.any(np.isin(bin_ind, invalid_bin_ind)):
        raise ValueError(
            "Some bin indices are invalid. Check the active_mask or bin_edges."
        )
    if was_scalar:
        return int(bin_ind[0]) if bin_ind.size > 0 else -1
    else:
        # Return the bin indices as an array
        return bin_ind.astype(int)
