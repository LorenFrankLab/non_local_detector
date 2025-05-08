"""Defines spatial environments using discrete grids and graph representations.

This module provides the `Environment` class and associated helper functions to
represent spatial environments commonly used in neuroscience experiments, such
as open fields or linear tracks. The core idea is to discretize the continuous
space into a grid of bins and potentially represent the connectivity or topology
of the valid space using a graph.

The module supports two main types of environments:

1.  **N-Dimensional Environments (e.g., Open Field, W-Track):**
    - Discretizes the space into a regular N-D grid based on specified bin sizes
      or position data ranges.
    - Can automatically infer the "track interior" (the portion of the grid
      actually occupied by the animal) from position data using histogramming
      and optional morphological operations (filling holes, dilation).
    - Constructs a `networkx` graph (`track_graph_nd`) where nodes represent
      the centers of *interior* bins, and edges connect adjacent interior bins.
      This graph captures the connectivity of the valid space.
    - Can compute shortest-path distances between all pairs of interior bins
      on this graph (`distance_between_bins`).
    - Provides methods to find the bin index for a given position (`get_bin_ind`),
      calculate manifold distances between positions (`get_manifold_distances`),
      and determine movement direction relative to the track center (`get_direction`).

2.  **1-Dimensional Environments (Linear Tracks, W-Tracks):**
    - Requires a `networkx.Graph` (`track_graph`) defining the track's topology
      (nodes, edges, and their positions) along with edge ordering and spacing.
    - Linearizes the track based on the provided graph structure.
    - Creates bins along this linearized track.
    - Computes shortest-path distances between all nodes in this augmented graph
      (`distance_between_bins`).

The central component is the `Environment` dataclass, which holds the parameters
defining the environment and stores the results of the fitting process (grid
details, graphs, distances) as attributes. The primary method `fit`
is used to perform the discretization and graph construction based on the input
parameters and optional position data.
"""

import pickle
from dataclasses import MISSING, dataclass, field, fields
from functools import cached_property
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from track_linearization import get_linearized_position, plot_graph_as_1D
from track_linearization.core import _calculate_linear_position

# --- Helper Functions ---


def get_centers(bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculates the center of each bin given its edges.

    Parameters
    ----------
    bin_edges : NDArray[np.float64], shape (n_edges,)
        The edges defining the bins.

    Returns
    -------
    bin_centers : NDArray[np.float64], shape (n_edges - 1,)
        The center of each bin.
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def get_n_bins(
    position: NDArray[np.float64],
    bin_size: Union[float, Sequence[float]],
    position_range: Optional[Sequence[Tuple[float, float]]] = None,
) -> NDArray[np.int_]:
    """Calculates the number of bins needed for each dimension.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_dims)
        Position data to determine range if `position_range` is not given.
    bin_size : float or Sequence[float]
        The desired size(s) of the bins.
    position_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit range [(min_dim1, max_dim1), ...] for each dimension.
        If None, range is calculated from `position`. Defaults to None.

    Returns
    -------
    n_bins : NDArray[np.int_], shape (n_dims,)
        Number of bins required for each dimension.
    """
    if position_range is not None:
        # Ensure position_range is numpy array for consistent processing
        pr = np.asarray(position_range)
        if pr.shape[1] != 2:
            raise ValueError("position_range must be sequence of (min, max) pairs.")
        extent = np.diff(pr, axis=1).squeeze(axis=1)
    else:
        # Ignore NaNs when calculating range from data
        extent = np.nanmax(position, axis=0) - np.nanmin(position, axis=0)

    # Ensure bin_size is positive
    if np.any(bin_size <= 0):
        raise ValueError("bin_size must be positive.")

    # Calculate number of bins, ensuring at least 1 bin even if extent is 0
    n_bins = np.ceil(extent / bin_size).astype(np.int32)
    n_bins[n_bins == 0] = 1  # Handle zero extent case

    return n_bins


def _create_grid(
    position: Optional[NDArray[np.float64]] = None,
    bin_size: Union[float, Sequence[float]] = 2.0,
    position_range: Optional[Sequence[Tuple[float, float]]] = None,
    add_boundary_bins: bool = True,
) -> Tuple[
    Tuple[NDArray[np.float64], ...],  # edges_tuple
    NDArray[np.float64],  # place_bin_edges_flat
    NDArray[np.float64],  # place_bin_centers_flat
    Tuple[int, ...],  # centers_shape
]:
    """Calculates bin edges and centers for a spatial grid.

    Creates a grid based on provided position data or range. Handles multiple
    position dimensions and optionally adds boundary bins around the core grid.

    Parameters
    ----------
    position : Optional[NDArray[np.float64]], shape (n_time, n_dims), optional
        Position data. Used to determine grid extent if `position_range`
        is None. NaNs are ignored. Required if `position_range` is None.
        Defaults to None.
    bin_size : Union[float, Sequence[float]], optional
        Desired approximate size of bins in each dimension. If a sequence,
        must match the number of dimensions. Defaults to 2.0.
    position_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit grid boundaries [(min_dim1, max_dim1), ...]. If None,
        boundaries are derived from `position`. Defaults to None.
    add_boundary_bins : bool, optional
        If True, add one bin on each side of the grid in each dimension,
        extending the range. Defaults to True.

    Returns
    -------
    edges : Tuple[NDArray[np.float64], ...]
        Tuple containing bin edges for each dimension (shape (n_bins_d + 1,)).
        Includes boundary bins if `add_boundary_bins` is True.
    place_bin_edges_flat : np.ndarray, shape (n_total_bins, n_position_dims)
        The edges corresponding to each bin in the flattened grid.
    place_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Center coordinates of each bin in the flattened grid.
    centers_shape : Tuple[int, ...]
        Shape of the grid (number of bins in each dimension).

    Raises
    ------
    ValueError
        If both `position` and `position_range` are None.
        If `bin_size` sequence length doesn't match dimensions.
        If `position_range` sequence length doesn't match dimensions.
    """
    if position is None and position_range is None:
        raise ValueError("Either `position` or `position_range` must be provided.")
    if position is not None:
        pos_nd = np.atleast_2d(position)
        n_dims = pos_nd.shape[1]
        pos_clean = pos_nd[~np.any(np.isnan(pos_nd), axis=1)]
        if pos_clean.shape[0] == 0 and position_range is None:
            raise ValueError(
                "Position data contains only NaNs and no position_range provided."
            )
    elif position_range is not None:
        n_dims = len(position_range)
        pos_clean = None  # No position data needed if range is given
    else:  # Should be unreachable due to first check, but added for safety
        raise ValueError("Cannot determine number of dimensions.")

    # Validate and process bin_size
    if isinstance(bin_size, (float, int)):
        bin_sizes = np.array([float(bin_size)] * n_dims)
    elif len(bin_size) == n_dims:
        bin_sizes = np.array(bin_size, dtype=float)
    else:
        raise ValueError(
            f"`bin_size` sequence length ({len(bin_size)}) must match "
            f"number of dimensions ({n_dims})."
        )
    if np.any(bin_sizes <= 0):
        raise ValueError("All elements in `bin_size` must be positive.")

    # Determine histogram range
    hist_range = position_range
    if hist_range is None and pos_clean is not None:
        hist_range = [
            (np.nanmin(pos_clean[:, dim]), np.nanmax(pos_clean[:, dim]))
            for dim in range(n_dims)
        ]
        # Handle case where min == max in a dimension
        hist_range = [
            (
                (r[0], r[1])
                if r[0] < r[1]
                else (r[0] - 0.5 * bin_sizes[i], r[0] + 0.5 * bin_sizes[i])
            )
            for i, r in enumerate(hist_range)
        ]

    # Validate position_range dimensions if provided
    if position_range is not None and len(position_range) != n_dims:
        raise ValueError(
            f"`position_range` length ({len(position_range)}) must match "
            f"number of dimensions ({n_dims})."
        )

    # Calculate number of bins for the core range
    n_bins_core = get_n_bins(pos_clean, bin_sizes, hist_range)  # Pass array bin_sizes

    # Calculate core edges using histogramdd (even if position is None, to get uniform bins)
    # Need dummy data if no position provided
    dummy_data = (
        np.array([[(r[0] + r[1]) / 2] for r in hist_range]).T
        if pos_clean is None
        else pos_clean
    )
    _, core_edges_list = np.histogramdd(dummy_data, bins=n_bins_core, range=hist_range)

    if add_boundary_bins:
        # Add boundary bins by extending edges
        final_edges_list = []
        for edges_dim in core_edges_list:
            step = np.diff(edges_dim)[0]  # Assume uniform bins from histogramdd
            extended_edges = np.insert(
                edges_dim,
                [0, len(edges_dim)],
                [edges_dim[0] - step, edges_dim[-1] + step],
            )
            final_edges_list.append(extended_edges)
    else:
        final_edges_list = list(core_edges_list)  # Ensure it's a list of arrays

    # Calculate centers and shape
    centers_list = [get_centers(edge_dim) for edge_dim in final_edges_list]
    centers_shape = tuple(len(c) for c in centers_list)

    # Create meshgrid of centers and flatten
    mesh_centers = np.meshgrid(*centers_list, indexing="ij")
    place_bin_centers_flat = np.stack(
        [center.ravel() for center in mesh_centers], axis=1
    )

    # Create meshgrid of edges and flatten
    mesh_edges = np.meshgrid(*final_edges_list, indexing="ij")
    place_bin_edges_flat = np.stack([edge.ravel() for edge in mesh_edges], axis=1)

    edges_tuple: Tuple[NDArray[np.float64], ...] = tuple(final_edges_list)

    return edges_tuple, place_bin_edges_flat, place_bin_centers_flat, centers_shape


def _infer_track_interior(
    position: NDArray[np.float64],
    edges: Tuple[NDArray[np.float64], ...],
    close_gaps: bool = False,
    fill_holes: bool = False,
    dilate: bool = False,
    bin_count_threshold: int = 0,
    boundary_exists: bool = True,
) -> NDArray[np.bool_]:
    """Infers the interior bins of the track based on position density.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_dims)
        Position data. NaNs are ignored.
    edges : Tuple[NDArray[np.float64], ...]
        Bin edges for each dimension, as returned by `create_grid`.
    fill_holes : bool, optional
        Fill holes within the inferred occupied area using binary closing
        and filling. Defaults to False.
    dilate : bool, optional
        Expand the boundary of the inferred occupied area using binary
        dilation. Defaults to False.
    bin_count_threshold : int, optional
        Minimum samples in a bin for it to be considered part of the track.
        Defaults to 0 (any occupancy counts).
    boundary_exists : bool, optional
        If True, the last bin in each dimension is not considered part of
        the track. Defaults to False.

    Returns
    -------
    is_track_interior : NDArray[np.bool_], shape (n_bins_dim1, n_bins_dim2, ...)
        Boolean array indicating which bins are considered part of the track.
    """
    pos_clean = position[~np.any(np.isnan(position), axis=1)]
    if pos_clean.shape[0] == 0:
        # Handle case with no valid positions
        grid_shape = tuple(len(e) - 1 for e in edges)
        return np.zeros(grid_shape, dtype=bool)

    bin_counts, _ = np.histogramdd(pos_clean, bins=edges)
    is_track_interior = bin_counts > bin_count_threshold

    n_dims = position.shape[1]
    if n_dims > 1:
        # Use connectivity=1 for 4-neighbor (2D) or 6-neighbor (3D) etc.
        structure = ndimage.generate_binary_structure(n_dims, connectivity=1)

        if close_gaps:
            # Closing operation first (dilation then erosion) to close small gaps
            is_track_interior = ndimage.binary_closing(
                is_track_interior, structure=structure
            )

        if fill_holes:
            # Fill larger holes enclosed by occupied bins
            is_track_interior = ndimage.binary_fill_holes(
                is_track_interior, structure=structure
            )

        if dilate:
            # Expand the occupied area
            is_track_interior = ndimage.binary_dilation(
                is_track_interior, structure=structure
            )

        if boundary_exists:
            is_track_interior[-1] = False
            is_track_interior[:, -1] = False

    return is_track_interior.astype(bool)


def _get_track_boundary(
    is_track_interior: NDArray[np.bool_], connectivity: int = 1
) -> NDArray[np.bool_]:
    """Identifies boundary bins adjacent to the track interior.

    The boundary bins themselves are *not* part of the track interior.

    Parameters
    ----------
    is_track_interior : NDArray[np.bool_], shape (n_bins_dim1, ...)
        Boolean array indicating track interior bins.
    connectivity : int, optional
        Defines neighborhood for dilation (1 for direct orthogonal neighbors,
        higher for diagonals). Defaults to 1.

    Returns
    -------
    is_track_boundary : NDArray[np.bool_], shape (n_bins_dim1, ...)
        Boolean array indicating bins adjacent to the track interior.
    """
    n_dims = is_track_interior.ndim
    if n_dims == 0:  # Handle scalar case
        return np.array(False, dtype=bool)
    structure = ndimage.generate_binary_structure(
        rank=n_dims, connectivity=connectivity
    )
    # Dilate the interior and XOR with original interior to find the boundary shell
    return (
        ndimage.binary_dilation(is_track_interior, structure=structure)
        ^ is_track_interior
    )


def _make_nd_track_graph(
    place_bin_centers: NDArray[np.float64],
    is_track_interior: NDArray[np.bool_],
    centers_shape: Tuple[int, ...],
) -> nx.Graph:
    """Creates a NetworkX graph connecting centers of adjacent interior bins in N-D.

    Parameters
    ----------
    place_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Coordinates of the center of each bin (flattened).
    is_track_interior : NDArray[np.bool_], shape (n_bins_dim1, ...)
        Boolean grid indicating which bins are part of the track.
    centers_shape : Tuple[int, ...]
        Shape of the bin grid (bins per dimension).

    Returns
    -------
    track_graph_nd : nx.Graph
        Graph where nodes are indices of interior bins, 'pos' attribute stores
        coordinates, and edges connect adjacent interior bins with 'distance'.
    """

    track_graph_nd = nx.Graph()
    axis_offsets = [-1, 0, 1]

    # Enumerate over nodes
    for node_id, (node_position, is_interior) in enumerate(
        zip(
            place_bin_centers,
            is_track_interior.ravel(),
        )
    ):
        track_graph_nd.add_node(
            node_id,
            pos=tuple(node_position),
            is_track_interior=is_interior,
            bin_ind=tuple(np.unravel_index(node_id, centers_shape)),
            bin_ind_flat=node_id,
        )

    edges = []
    # Enumerate over nodes in the track interior
    for ind in zip(*np.nonzero(is_track_interior)):
        ind = np.array(ind)
        # Indices of adjacent nodes
        adj_inds = np.meshgrid(*[axis_offsets + i for i in ind], indexing="ij")
        # Remove out of bounds indices
        adj_inds = [
            inds[np.logical_and(inds >= 0, inds < dim_size)]
            for inds, dim_size in zip(adj_inds, centers_shape)
        ]

        # Is the adjacent node on the track?
        adj_on_track_inds = is_track_interior[tuple(adj_inds)]

        # Remove the center node
        center_idx = [n // 2 for n in adj_on_track_inds.shape]
        adj_on_track_inds[tuple(center_idx)] = False

        # Get the node ids of the center node
        node_id = np.ravel_multi_index(ind, centers_shape)

        # Get the node ids of the adjacent nodes on the track
        adj_node_ids = np.ravel_multi_index(
            [inds[adj_on_track_inds] for inds in adj_inds],
            centers_shape,
        )

        # Collect the edges for the graph
        edges.append(
            np.concatenate(
                np.meshgrid([node_id], adj_node_ids, indexing="ij"), axis=0
            ).T
        )

    edges = np.concatenate(edges)

    # Add edges to the graph with distance
    for node1, node2 in edges:
        pos1 = np.asarray(track_graph_nd.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph_nd.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph_nd.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph_nd.edges):
        track_graph_nd.edges[edge]["edge_id"] = edge_id

    return track_graph_nd


def _get_distance_between_nodes(track_graph_nd: nx.Graph) -> np.ndarray:
    """Calculates the shortest path distances between nodes in a graph.

    Parameters
    ----------
    track_graph_nd : nx.Graph
        Graph where nodes are indices of interior bins, 'pos' attribute stores
        coordinates, and edges connect adjacent interior bins with 'distance'.

    Returns
    -------
    distance : np.ndarray, shape (n_nodes, n_nodes)
        Matrix of shortest path distances between all pairs of nodes.
    """

    node_to_bin_ind_flat = nx.get_node_attributes(track_graph_nd, "bin_ind_flat")

    node_positions = nx.get_node_attributes(track_graph_nd, "pos")
    node_positions = np.asarray(list(node_positions.values()))
    n_bins = len(node_positions)

    distance = np.full((n_bins, n_bins), np.inf)
    for to_node_id, from_node_id in nx.shortest_path_length(
        track_graph_nd,
        weight="distance",
    ):
        to_bin_ind = node_to_bin_ind_flat[to_node_id]
        from_bin_inds = [node_to_bin_ind_flat[node_id] for node_id in from_node_id]
        distance[to_bin_ind, from_bin_inds] = list(from_node_id.values())

    return distance


def make_track_graph_bin_centers_edges(
    track_graph: nx.Graph, place_bin_size: float
) -> nx.Graph:
    """Insert the bin center and bin edge positions as nodes in the track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    place_bin_size : float

    Returns
    -------
    track_graph_bin_centers_edges : nx.Graph

    """
    track_graph_bin_centers_edges = track_graph.copy()
    n_nodes = len(track_graph.nodes)

    for edge_ind, (node1, node2) in enumerate(track_graph.edges):
        node1_x_pos, node1_y_pos = track_graph.nodes[node1]["pos"]
        node2_x_pos, node2_y_pos = track_graph.nodes[node2]["pos"]

        edge_size = np.linalg.norm(
            [(node2_x_pos - node1_x_pos), (node2_y_pos - node1_y_pos)]
        )
        n_bins = 2 * np.ceil(edge_size / place_bin_size).astype(np.int32) + 1
        if ~np.isclose(node1_x_pos, node2_x_pos):
            f = interp1d((node1_x_pos, node2_x_pos), (node1_y_pos, node2_y_pos))
            xnew = np.linspace(node1_x_pos, node2_x_pos, num=n_bins, endpoint=True)
            xy = np.stack((xnew, f(xnew)), axis=1)
        else:
            ynew = np.linspace(node1_y_pos, node2_y_pos, num=n_bins, endpoint=True)
            xnew = np.ones_like(ynew) * node1_x_pos
            xy = np.stack((xnew, ynew), axis=1)
        dist_between_nodes = np.linalg.norm(np.diff(xy, axis=0), axis=1)

        new_node_ids = n_nodes + np.arange(len(dist_between_nodes) + 1)
        nx.add_path(
            track_graph_bin_centers_edges,
            [*new_node_ids],
            distance=dist_between_nodes[0],
        )
        nx.add_path(track_graph_bin_centers_edges, [node1, new_node_ids[0]], distance=0)
        nx.add_path(
            track_graph_bin_centers_edges, [node2, new_node_ids[-1]], distance=0
        )
        track_graph_bin_centers_edges.remove_edge(node1, node2)
        for ind, (node_id, pos) in enumerate(zip(new_node_ids, xy)):
            track_graph_bin_centers_edges.nodes[node_id]["pos"] = pos
            track_graph_bin_centers_edges.nodes[node_id]["edge_id"] = edge_ind
            if ind % 2:
                track_graph_bin_centers_edges.nodes[node_id]["is_bin_edge"] = False
            else:
                track_graph_bin_centers_edges.nodes[node_id]["is_bin_edge"] = True
        track_graph_bin_centers_edges.nodes[node1]["edge_id"] = edge_ind
        track_graph_bin_centers_edges.nodes[node2]["edge_id"] = edge_ind
        track_graph_bin_centers_edges.nodes[node1]["is_bin_edge"] = True
        track_graph_bin_centers_edges.nodes[node2]["is_bin_edge"] = True
        n_nodes = len(track_graph_bin_centers_edges.nodes)

    return track_graph_bin_centers_edges


def extract_bin_info_from_track_graph(
    track_graph: nx.Graph,
    track_graph_bin_centers_edges: nx.Graph,
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
) -> pd.DataFrame:
    """For each node, find edge_id, is_bin_edge, x_position, y_position, and
    linear_position.

    Parameters
    ----------
    track_graph : nx.Graph
    track_graph_bin_centers_edges : nx.Graph
    edge_order : list of 2-tuples
    edge_spacing : list, len n_edges - 1

    Returns
    -------
    nodes_df : pd.DataFrame
        Collect information about each bin

    """
    nodes_df = (
        pd.DataFrame.from_dict(
            dict(track_graph_bin_centers_edges.nodes(data=True)), orient="index"
        )
        .assign(x_position=lambda df: np.asarray(list(df.pos))[:, 0])
        .assign(y_position=lambda df: np.asarray(list(df.pos))[:, 1])
        .drop(columns="pos")
    )
    node_linear_position, _, _ = _calculate_linear_position(
        track_graph,
        np.asarray(nodes_df.loc[:, ["x_position", "y_position"]]),
        np.asarray(nodes_df.edge_id),
        edge_order,
        edge_spacing,
    )
    nodes_df["linear_position"] = node_linear_position
    nodes_df = nodes_df.rename_axis(index="node_id")
    edge_avg_linear_position = (
        nodes_df.groupby("edge_id")
        .linear_position.mean()
        .rename("edge_avg_linear_position")
    )

    nodes_df = (
        pd.merge(nodes_df.reset_index(), edge_avg_linear_position, on="edge_id")
        .sort_values(by=["edge_avg_linear_position", "linear_position"], axis="rows")
        .set_index("node_id")
        .drop(columns="edge_avg_linear_position")
    )

    return nodes_df


def _create_1d_track_grid_data(
    track_graph: nx.Graph,
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
    place_bin_size: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
    tuple,
    tuple,
    nx.Graph,
]:
    """Figures out 1D spatial bins given a track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    edge_order : list of 2-tuples
    edge_spacing : list, len n_edges - 1
    place_bin_size : float

    Returns
    -------
    place_bin_centers : np.ndarray, shape (n_bins, n_position_dims)
    place_bin_edges : np.ndarray, shape (n_bins + n_position_dims, n_position_dims)
    is_track_interior : np.ndarray, shape (n_bins, n_position_dim)
    distance_between_nodes : dict
    centers_shape : tuple
    edges : tuple of np.ndarray
    track_graph_bin_centers : nx.Graph
    """
    track_graph_bin_centers_edges = make_track_graph_bin_centers_edges(
        track_graph, place_bin_size
    )
    nodes_df = extract_bin_info_from_track_graph(
        track_graph, track_graph_bin_centers_edges, edge_order, edge_spacing
    )

    # Dataframe with nodes from track graph only
    original_nodes = list(track_graph.nodes)
    original_nodes_df = nodes_df.loc[original_nodes].reset_index()

    # Dataframe with only added edge nodes
    place_bin_edges_nodes_df = nodes_df.loc[
        ~nodes_df.index.isin(original_nodes) & nodes_df.is_bin_edge
    ].reset_index()

    # Dataframe with only added center nodes
    place_bin_centers_nodes_df = nodes_df.loc[~nodes_df.is_bin_edge].reset_index()

    # Determine place bin edges and centers.
    # Make sure to remove duplicate nodes from bins with no gaps
    is_duplicate_edge = np.isclose(
        np.diff(np.asarray(place_bin_edges_nodes_df.linear_position)), 0.0
    )
    is_duplicate_edge = np.append(is_duplicate_edge, False)
    no_duplicate_place_bin_edges_nodes_df = place_bin_edges_nodes_df.iloc[
        ~is_duplicate_edge
    ]
    place_bin_edges = np.asarray(no_duplicate_place_bin_edges_nodes_df.linear_position)
    place_bin_centers = get_centers(place_bin_edges)

    # Figure out which points are on the track and not just gaps
    change_edge_ind = np.nonzero(
        np.diff(no_duplicate_place_bin_edges_nodes_df.edge_id)
    )[0]

    if isinstance(edge_spacing, (int, float)):
        n_edges = len(edge_order)
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    is_track_interior = np.ones_like(place_bin_centers, dtype=bool)
    not_track = change_edge_ind[np.asarray(edge_spacing) > 0]
    is_track_interior[not_track] = False

    # Add information about bin centers not on track
    place_bin_centers_nodes_df = (
        pd.concat(
            (
                place_bin_centers_nodes_df,
                pd.DataFrame(
                    {
                        "linear_position": place_bin_centers[~is_track_interior],
                        "node_id": -1,
                        "edge_id": -1,
                        "is_bin_edge": False,
                    }
                ),
            )
        ).sort_values(by=["linear_position"], axis="rows")
    ).reset_index(drop=True)
    place_bin_centers_nodes_df["bin_ind"] = np.arange(
        len(place_bin_centers_nodes_df), dtype=np.int32
    )
    place_bin_centers_nodes_df["bin_ind_flat"] = place_bin_centers_nodes_df["bin_ind"]

    track_graph_bin_centers = _make_track_graph_bin_centers(
        place_bin_centers_nodes_df,
        track_graph_bin_centers_edges,
        original_nodes_df,
    )

    # Other needed information
    edges = (place_bin_edges,)
    centers_shape = (place_bin_centers.size,)

    return (
        place_bin_centers[:, np.newaxis],
        place_bin_edges[:, np.newaxis],
        is_track_interior,
        centers_shape,
        edges,
        track_graph_bin_centers,
    )


def _order_boundary(boundary: np.ndarray) -> np.ndarray:
    """Given boundary bin centers, orders them in a way to make a continuous line.

    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line

    Parameters
    ----------
    boundary : np.ndarray, shape (n_boundary_points, n_position_dims)

    Returns
    -------
    ordered_boundary : np.ndarray, shape (n_boundary_points, n_position_dims)

    """
    n_points = boundary.shape[0]
    clf = NearestNeighbors(n_neighbors=2).fit(boundary)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(n_points)]
    min_idx, min_dist = 0, np.inf

    for idx, path in enumerate(paths):
        ordered = boundary[path]  # ordered nodes
        cost = np.sum(np.diff(ordered) ** 2)
        if cost < min_dist:
            min_idx, min_dist = idx, cost

    opt_order = paths[min_idx]
    return boundary[opt_order][:-1]


def get_track_boundary_points(
    is_track_interior: np.ndarray, edges: list[np.ndarray], connectivity: int = 1
) -> np.ndarray:
    """

    Parameters
    ----------
    is_track_interior : np.ndarray, shape (n_x_bins, n_y_bins)
    edges : list of ndarray

    Returns
    -------
    boundary_points : np.ndarray, shape (n_boundary_points, n_position_dims)

    """
    boundary = _get_track_boundary(is_track_interior, connectivity=connectivity)

    inds = np.nonzero(boundary)
    centers = [get_centers(x) for x in edges]
    boundary = np.stack([center[ind] for center, ind in zip(centers, inds)], axis=1)
    return _order_boundary(boundary)


def gaussian_smooth(
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
    return ndimage.gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def add_distance_weight_to_edges(
    track_graph: nx.Graph,
) -> nx.Graph:
    """Adds a distance weight to each edge in the track graph.

    Parameters
    ----------
    track_graph : nx.Graph
        The input track graph.

    Returns
    -------
    track_graph : nx.Graph
        The modified track graph with distance weights added to edges.
    """
    new_attribute_name = "distance_weight"
    for u, v, data in track_graph.edges(data=True):
        try:
            distance = data["distance"]
            # Avoid division by zero
            computed_value = 1.0 / (distance + 1e-9) if distance > -1e9 else np.inf
            track_graph.edges[u, v][new_attribute_name] = computed_value
        except KeyError:
            # If the distance attribute is not present, skip this edge
            continue


def _get_node_pos(graph: nx.Graph, node_id: Any) -> np.ndarray:
    """Helper to get node position as a numpy array."""
    if node_id not in graph or "pos" not in graph.nodes[node_id]:
        raise KeyError(f"Node {node_id} or its 'pos' attribute not found in graph.")
    return np.asarray(graph.nodes[node_id]["pos"])


def _make_track_graph_bin_centers(
    place_bin_centers_nodes_df: pd.DataFrame,
    track_graph_bin_centers_edges: nx.Graph,
    original_nodes_df: pd.DataFrame,
) -> nx.Graph:
    """Creates a graph connecting bin centers sequentially along the track.

    Builds a graph using only bin center nodes. Connects adjacent centers
    within segments based on linear position. Links segment endpoints meeting
    at original track graph nodes by looking via intermediate nodes in the
    augmented graph.

    Parameters
    ----------
    place_bin_centers_nodes_df : pd.DataFrame
        DataFrame containing bin center nodes with columns:
        - node_id: Unique identifier for each node.
        - x_position: X-coordinate of the node.
        - y_position: Y-coordinate of the node.
        - edge_id: Identifier for the edge this node belongs to.
        - linear_position: Linear position along the edge.
    track_graph_bin_centers_edges : nx.Graph
        Graph with bin centers as nodes, linked sequentially and at junctions.
    original_nodes_df : pd.DataFrame
        DataFrame containing original nodes with columns:
        - node_id: Unique identifier for each node.
        - x_position: X-coordinate of the node.
        - y_position: Y-coordinate of the node.
        - edge_id: Identifier for the edge this node belongs to.
        - linear_position: Linear position along the edge.
        - is_bin_edge: Boolean indicating if the node is a bin edge.
        - is_track_interior: Boolean indicating if the node is part of the track.
        - bin_ind: Index of the bin in the grid.
        - bin_ind_flat: Flattened index of the bin in the grid.
        - edge_avg_linear_position: Average linear position of the edge.
        - distance: Distance to the next node.

    Returns
    -------
    track_graph_bin_centers : nx.Graph
        Graph with bin centers as nodes, linked sequentially and at junctions.

    Raises
    ------
    KeyError
        If a node's position is not found in the graph.

    """
    track_graph_bin_centers = nx.Graph()
    centers_df = place_bin_centers_nodes_df.copy()

    # --- 1. Add Nodes ---
    nodes_to_add = []
    valid_centers_df = centers_df[centers_df["node_id"] != -1]
    for _, row in valid_centers_df.iterrows():
        nodes_to_add.append(
            (
                row["node_id"],
                {
                    "pos": (row["x_position"], row["y_position"]),
                    "edge_id": int(row["edge_id"]),
                    "bin_ind": row["bin_ind"],
                    "bin_ind_flat": row["bin_ind_flat"],
                    "is_track_interior": True,
                },
            )
        )
    track_graph_bin_centers.add_nodes_from(nodes_to_add)

    for ind, (_, row) in enumerate(centers_df[centers_df["node_id"] == -1].iterrows()):
        nodes_to_add.append(
            (
                -ind - 1,
                {
                    "pos": (np.nan, np.nan),
                    "edge_id": -1,
                    "bin_ind": row["bin_ind"],
                    "bin_ind_flat": row["bin_ind_flat"],
                    "is_track_interior": False,
                },
            )
        )
    track_graph_bin_centers.add_nodes_from(nodes_to_add)

    # --- 2. Add Intra-Segment Edges ---
    edges_to_add: List[Tuple[Any, Any, Dict[str, Any]]] = []
    intra_segment_edge_count = 0
    for edge_id, group in valid_centers_df.groupby("edge_id"):
        sorted_group = group.sort_values("linear_position")
        node_ids = sorted_group["node_id"].values
        for node1, node2 in zip(node_ids[:-1], node_ids[1:]):
            try:
                pos1 = _get_node_pos(track_graph_bin_centers, node1)
                pos2 = _get_node_pos(track_graph_bin_centers, node2)
            except KeyError as e:
                continue
            distance = np.linalg.norm(pos1 - pos2)
            if distance > 1e-9:
                edges_to_add.append(
                    (node1, node2, {"distance": distance, "edge_id": edge_id})
                )
                intra_segment_edge_count += 1
    track_graph_bin_centers.add_edges_from(edges_to_add)

    # --- 3. Add Inter-Segment Edges (Link segment ends) ---
    linking_edges_to_add: List[Tuple[Any, Any, Dict[str, Any]]] = []
    inter_segment_edge_count = 0
    augmented_graph = track_graph_bin_centers_edges
    original_node_ids_in_augmented = original_nodes_df["node_id"].unique()

    for original_node_id in original_node_ids_in_augmented:
        if not augmented_graph.has_node(original_node_id):
            continue  # Skip if original node somehow isn't in augmented graph

        # Find bin centers associated with this junction
        endpoint_centers: Set[Any] = set()  # Use set to store unique centers

        # Find direct neighbors of the original node (likely edge nodes)
        direct_neighbors = list(augmented_graph.neighbors(original_node_id))

        for intermediate_node in direct_neighbors:
            # Find neighbors of the intermediate node
            second_level_neighbors = list(augmented_graph.neighbors(intermediate_node))
            # Filter these to find bin centers present in our target graph
            for potential_center in second_level_neighbors:
                # Ensure it's not the original node itself and it IS a bin center
                if (
                    potential_center != original_node_id
                    and track_graph_bin_centers.has_node(potential_center)
                ):
                    endpoint_centers.add(potential_center)

        # Add edges between all pairs of these endpoint bin centers
        for node1, node2 in combinations(endpoint_centers, 2):
            # Avoid adding edges already added or self-loops (combinations handles self-loops)
            if not track_graph_bin_centers.has_edge(node1, node2):
                try:
                    pos1 = _get_node_pos(track_graph_bin_centers, node1)
                    pos2 = _get_node_pos(track_graph_bin_centers, node2)
                except KeyError as e:
                    continue

                distance = np.linalg.norm(pos1 - pos2)
                linking_edges_to_add.append(
                    (node1, node2, {"distance": distance, "edge_id": -1})
                )
                inter_segment_edge_count += 1

    track_graph_bin_centers.add_edges_from(linking_edges_to_add)

    return track_graph_bin_centers


def get_direction(
    env: "Environment",
    position: np.ndarray,
    position_time: Optional[np.ndarray] = None,
    sigma: float = 0.1,
    sampling_frequency: Optional[float] = None,
    classify_stop: bool = False,
    stop_speed_threshold: float = 1e-3,
) -> np.ndarray:
    """Get the direction of movement relative to the center of the track (inward/outward).

    Requires a fitted N-D environment with a corresponding track graph (`track_graph_nd_`).

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_dims)
        Position data.
    position_time : np.ndarray, shape (n_time,), optional
        Timestamps for position data. If None, assumes uniform sampling.
    sigma : float, optional
        Standard deviation (in seconds) for Gaussian smoothing of velocity towards center. Defaults to 0.1.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If None, estimated from `position_time`.
    classify_stop : bool, optional
        If True, classify speeds below `stop_speed_threshold` as "stop". Defaults to False.
    stop_speed_threshold : float, optional
        Speed threshold for classifying stops. Defaults to 1e-3.

    Returns
    -------
    direction : np.ndarray, shape (n_time,)
        Array of strings: "inward", "outward", or "stop".

    Raises
    ------
    RuntimeError
        If the environment has not been fitted or lacks the N-D track graph.
    ValueError
        If sampling frequency cannot be determined.
    """

    if not env._is_fitted:
        raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")
    if env.track_graph_nd_ is None:
        raise RuntimeError(
            "Direction finding requires a fitted N-D environment with a track graph ('track_graph_nd_') and precomputed distances."
        )

    if position_time is None:
        position_time = np.arange(position.shape[0])
    if sampling_frequency is None:
        sampling_frequency = 1 / np.mean(np.diff(position_time))

    centrality = nx.closeness_centrality(env.track_graph_nd_, distance="distance")
    center_node_id = list(centrality.keys())[np.argmax(list(centrality.values()))]

    bin_ind = env.get_bin_ind(position)

    velocity_to_center = gaussian_smooth(
        np.gradient(env.distance_between_bins[bin_ind, center_node_id]),
        sigma,
        sampling_frequency,
        axis=0,
        truncate=8,
    )
    direction = np.where(
        velocity_to_center < 0,
        "inward",
        "outward",
    )

    if classify_stop:
        direction[np.abs(velocity_to_center) < stop_speed_threshold] = "stop"

    return direction


@dataclass
class Environment:
    """Represents a spatial environment with a discrete grid and graph topology.

    Handles both N-dimensional open fields and 1-dimensional tracks defined
    by a graph. Fits a grid to the space, identifies the traversable interior,
    and computes graph representations and distances.

    Parameters
    ----------
    environment_name : str, optional
        Identifier for the environment. Defaults to "".
    place_bin_size : Union[float, Sequence[float]], optional
        Approximate size of position bins (cm or arbitrary units). Used for N-D
        gridding and setting the scale for 1-D binning. Defaults to 2.0.
    track_graph : Optional[nx.Graph], optional
        For 1-D environments only. A graph defining the track topology. Nodes
        must have a 'pos' attribute (x, y coordinates). Edges should represent
        physical connections and ideally have 'distance' (Euclidean length) and
        'edge_id' (unique integer) attributes. If None, an N-D environment is assumed.
        Defaults to None.
    edge_order : Optional[List[Tuple[Any, Any]]], optional
        Required if `track_graph` is provided. An ordered list of node pairs
        (edges) defining the linearization sequence of the 1-D track.
        Defaults to None.
    edge_spacing : Optional[Union[float, Sequence[float]]], optional
        Required if `track_graph` is provided. Spacing added between consecutive
        edges in `edge_order` during linearization. If float, uniform spacing.
        If Sequence, length must be `len(edge_order) - 1`. Defaults to 0.0.
    position_range : Optional[Sequence[Tuple[float, float]]], optional
        For N-D environments. Explicit boundaries [(min_dim1, max_dim1), ...]
        for the grid. If None, range is determined from `position` data during `fit`.
        Defaults to None.
    infer_track_interior : bool, optional
        For N-D environments. If True, infer the occupied track area from
        `position` data during `fit`. Ignored if `track_graph` is provided.
        Defaults to True.
    close_gaps : bool, optional
        For N-D inferred interiors. If True, close small gaps in the occupied area
        using binary closing. Defaults to False.
    fill_holes : bool, optional
        For N-D inferred interiors. If True, fill holes within the occupied area.
        Defaults to False.
    dilate : bool, optional
        For N-D inferred interiors. If True, expand the boundary of the occupied area.
        Defaults to False.
    bin_count_threshold : int, optional
        For N-D inferred interiors. Minimum samples in a bin to be considered occupied.
        Defaults to 0.
    is_track_interior_manual : Optional[NDArray[np.bool_]], optional
        For N-D environments. A manually specified boolean grid defining the
        track interior. If provided, overrides inference. Shape must match the
        bin grid derived from `place_bin_size` and `position_range`/`position`.
        Defaults to None.

    Attributes (Fitted)
    --------------------
    # Common Attributes
    is_1d : bool
        True if the environment is 1-Dimensional (track_graph provided).
    place_bin_centers_ : NDArray[np.float64], shape (n_bins, n_dims)
        Coordinates of the center of each valid bin. For 1D, shape is (n_bins, 1).
    is_track_interior_ : NDArray[np.bool_]
        Boolean array indicating valid bins. Shape depends on type:
        N-D: (n_bins_dim1, n_bins_dim2, ...) grid shape.
        1-D: (n_bins,) linear shape.
    centers_shape_ : Tuple[int, ...]
        Shape of the bin grid (bins per dimension). For 1D, (n_bins,).
    edges_ : Tuple[NDArray[np.float64], ...]
        Bin edges for each dimension. For 1D, contains linearized edges.

    # N-D Specific Attributes
    position_range_ : Optional[Sequence[Tuple[float, float]]]
        The actual position range used for gridding.
    is_track_boundary_ : Optional[NDArray[np.bool_]]
        Boolean grid indicating bins adjacent to the N-D track interior.
    track_graph_nd_ : Optional[nx.Graph]
        Graph connecting centers of adjacent interior N-D bins.

    # 1-D Specific Attributes
    track_graph_bin_centers_ : Optional[nx.Graph]
         Graph connecting only bin centers sequentially and at junctions.

    _is_fitted : bool
         Internal flag indicating if `fit` has been called.
    """

    environment_name: str = ""
    place_bin_size: Union[float, Tuple[float, ...]] = 2.0
    track_graph: Optional[nx.Graph] = None
    edge_order: Optional[List[Tuple[Any, Any]]] = None
    edge_spacing: Union[float, Sequence[float]] = 0.0
    position_range: Optional[Sequence[Tuple[float, float]]] = None
    infer_track_interior: bool = True
    close_gaps: bool = False
    fill_holes: bool = False
    dilate: bool = False
    bin_count_threshold: int = 0
    is_track_interior_manual: Optional[NDArray[np.bool_]] = None

    # Fitted attributes
    is_1d: bool = field(init=False)
    place_bin_centers_: Optional[NDArray[np.float64]] = field(init=False, default=None)
    is_track_interior_: Optional[NDArray[np.bool_]] = field(init=False, default=None)
    centers_shape_: Optional[Tuple[int, ...]] = field(init=False, default=None)
    edges_: Optional[Tuple[NDArray[np.float64], ...]] = field(init=False, default=None)

    ## N-D
    position_range_: Optional[Sequence[Tuple[float, float]]] = field(
        init=False, default=None
    )
    is_track_boundary_: Optional[NDArray[np.bool_]] = field(init=False, default=None)
    track_graph_nd_: Optional[nx.Graph] = field(init=False, default=None)

    ## 1-D
    track_graph_bin_centers_: Optional[nx.Graph] = field(init=False, default=None)

    # Internal flag
    _is_fitted: bool = field(init=False, default=False)

    def __post_init__(self):
        """Determine environment type after initialization."""
        self.is_1d = self.track_graph is not None
        if self.is_1d and (self.edge_order is None):
            raise ValueError(
                "`edge_order` must be provided for 1D environments (`track_graph` is set)."
            )

    def __eq__(self, other: object) -> bool:
        """Check equality based on environment name."""
        if isinstance(other, Environment):
            return self.environment_name == other.environment_name
        elif isinstance(other, str):
            return self.environment_name == other
        return NotImplemented

    def _fit_nd(self, position: Optional[NDArray[np.float64]] = None) -> None:
        """Fit method for N-dimensional environments."""
        if (
            position is None
            and self.position_range is None
            and self.is_track_interior_manual is None
        ):
            raise ValueError(
                "For N-D environments, must provide `position`, `position_range`, or `is_track_interior_manual`."
            )

        # 1. Create Grid
        # Use manual interior shape if provided to determine grid
        if self.is_track_interior_manual is not None:
            if self.position_range is None:
                print(
                    "Warning: `is_track_interior_manual` provided without `position_range`. Assuming range based on bin size and shape."
                )
                # Infer range approximately - this might not be ideal
                manual_shape = self.is_track_interior_manual.shape
                n_dims = self.is_track_interior_manual.ndim
                if isinstance(self.place_bin_size, (float, int)):
                    bin_sizes = np.array([float(self.place_bin_size)] * n_dims)
                else:  # Sequence
                    bin_sizes = np.asarray(self.place_bin_size)

                self.position_range_ = tuple(
                    (0.0, sh * bs) for sh, bs in zip(manual_shape, bin_sizes)
                )
            else:
                self.position_range_ = self.position_range  # Use provided range

            # Create grid based on manual shape and range
            n_bins = self.is_track_interior_manual.shape
            _, self.edges_ = np.histogramdd(
                np.zeros((1, n_dims)), bins=n_bins, range=self.position_range_
            )
            # Adjust edges for boundary bins (assuming create_grid adds them)
            centers_list = [get_centers(edge_dim) for edge_dim in self.edges_]
            self.centers_shape_ = tuple(len(c) for c in centers_list)
            mesh_centers = np.meshgrid(*centers_list, indexing="ij")
            self.place_bin_centers_ = np.stack(
                [c.ravel() for c in mesh_centers], axis=1
            )

            if self.centers_shape_ != self.is_track_interior_manual.shape:
                raise ValueError(
                    f"Shape of `is_track_interior_manual` {self.is_track_interior_manual.shape} "
                    f"does not match derived grid shape {self.centers_shape_} "
                    f"from `position_range` and `place_bin_size`."
                )
            self.is_track_interior_ = self.is_track_interior_manual

        else:
            # Create grid from position/position_range
            (
                self.edges_,
                self.place_bin_edges_,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = _create_grid(
                position=position,
                bin_size=self.place_bin_size,
                position_range=self.position_range,
                add_boundary_bins=True,
            )
            # Store the actual range used (needed if derived from position)
            self.position_range_ = tuple((e[0], e[-1]) for e in self.edges_)

            # 2. Determine Track Interior
            if self.infer_track_interior:
                if position is None:
                    raise ValueError(
                        "`position` data must be provided when `infer_track_interior` is True."
                    )
                self.is_track_interior_ = _infer_track_interior(
                    position=position,
                    edges=self.edges_,
                    fill_holes=self.fill_holes,
                    dilate=self.dilate,
                    bin_count_threshold=self.bin_count_threshold,
                )
            else:
                self.is_track_interior_ = np.zeros(self.centers_shape_, dtype=bool)
                # Create slice object (e.g., (slice(1,-1), slice(1,-1), ...))
                core_slice = tuple(slice(1, s - 1) for s in self.centers_shape_)
                self.is_track_interior_[core_slice] = True

        # 3. Determine Track Boundary (only if > 1D)
        if self.is_track_interior_.ndim > 1:
            self.is_track_boundary_ = _get_track_boundary(
                self.is_track_interior_, connectivity=1
            )
        else:
            self.is_track_boundary_ = None  # No meaningful boundary for 1D grid

        # 4. Create N-D Track Graph
        self.track_graph_nd_ = _make_nd_track_graph(
            self.place_bin_centers_, self.is_track_interior_, self.centers_shape_
        )

    def _fit_1d(self) -> None:
        """Fit method for 1-dimensional track environments."""
        if self.track_graph is None or self.edge_order is None:
            raise ValueError(
                "`track_graph` and `edge_order` are required for 1D fitting."
            )

        (
            self.place_bin_centers_,
            self.place_bin_edges_,
            self.is_track_interior_,
            self.centers_shape_,
            self.edges_,
            self.track_graph_bin_centers_,
        ) = _create_1d_track_grid_data(
            self.track_graph,
            self.edge_order,
            self.edge_spacing,
            self.place_bin_size,
        )

        # N-D specific attributes are None for 1D
        self.position_range_ = None
        self.is_track_boundary_ = None
        self.track_graph_nd_ = None

    def fit(self, position: Optional[NDArray[np.float64]] = None) -> "Environment":
        """Fits the discrete grid and graph representation of the environment.

        Based on the presence of `track_graph`, calls either the N-dimensional
        or 1-dimensional fitting routine.

        Parameters
        ----------
        position : Optional[NDArray[np.float64]], shape (n_time, n_dims), optional
            Position data of the animal. Required for N-D fitting if
            `position_range` and `is_track_interior_manual` are not provided,
            or if `infer_track_interior` is True. Not directly used for 1-D
            fitting (which relies on the `track_graph` geometry), but can be
            used by subsequent methods like `get_bin_indices`. Defaults to None.

        Returns
        -------
        self : Environment
            The fitted Environment instance.

        Raises
        ------
        ValueError
            If required parameters for the chosen environment type are missing
            (e.g., `edge_order` for 1D, or sufficient info for N-D grid).
        """
        self.is_1d = self.track_graph is not None

        if self.is_1d:
            self._fit_1d()
        else:
            # N-D requires position data unless range and manual interior are given
            if (
                self.position_range is None
                and self.is_track_interior_manual is None
                and position is None
            ) or (
                self.infer_track_interior
                and position is None
                and self.is_track_interior_manual is None
            ):
                raise ValueError(
                    "`position` data is required for N-D fitting under current settings."
                )
            self._fit_nd(position)

        self._is_fitted = True

        return self

    def plot_grid(
        self, ax: Optional[matplotlib.axes.Axes] = None
    ) -> matplotlib.axes.Axes:
        """Plots the spatial grid and track interior/graph.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing axes to plot on. If None, creates new axes. Defaults to None.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes used for plotting.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        ValueError
            If the environment is 1D but `track_graph` and `edge_order` are not set.
        NotImplementedError
            If the environment is not 1D or 2D.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted. Call fit() first.")

        if self.is_1d:
            # Plot 1D linearized track
            if ax is None:
                fig, ax = plt.subplots(figsize=(15, 2.5))
            if self.track_graph and self.edge_order:
                # Plot the original graph structure linearized
                plot_graph_as_1D(
                    self.track_graph,
                    self.edge_order,
                    self.edge_spacing,
                    ax=ax,
                    node_size=50,
                )

                # Overlay bin edges
                if self.edges_ and self.edges_[0] is not None:
                    edges_lin = self.edges_[0]
                    for edge_pos in edges_lin:
                        ax.axvline(
                            edge_pos, linewidth=0.5, color="black", linestyle=":"
                        )
                ax.set_title(f"{self.environment_name} (Linearized)")
                ax.set_xlabel("Linearized Position")
                ax.set_yticks([])  # Remove y-ticks for 1D plot
                ax.set_ylim(-0.1, 0.1)  # Adjust y-limits for node visibility

            else:
                raise ValueError(
                    "1D environment requires `track_graph` and `edge_order` to be set."
                )

        else:
            # Plot 2D grid
            if len(self.centers_shape_) != 2:
                raise NotImplementedError(
                    "Plotting is only implemented for 2D environments."
                )

            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 7))

            # Plot interior bins
            ax.pcolormesh(
                self.edges_[0],
                self.edges_[1],
                self.is_track_interior_.T,
                cmap="bone_r",
                alpha=0.7,
                shading="auto",
            )

            # Grid lines
            ax.set_xticks(self.edges_[0])
            ax.set_yticks(self.edges_[1])
            ax.set_xticks(get_centers(self.edges_[0]), minor=True)
            ax.set_yticks(get_centers(self.edges_[1]), minor=True)
            ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")

            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{self.environment_name} (Grid)")
            ax.set_xlabel("Position Dim 1")
            ax.set_ylabel("Position Dim 2")

            if self.position_range_:
                ax.set_xlim(self.position_range_[0])
                ax.set_ylim(self.position_range_[1])

        return ax

    def save(self, filename: str = "environment.pkl") -> None:
        """Saves the environment object as a pickled file.

        Parameters
        ----------
        filename : str, optional
            File name to save the environment to. Defaults to "environment.pkl".
        """
        with open(filename, "wb") as file_handle:
            pickle.dump(self, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Environment saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> "Environment":
        """Loads an Environment object from a pickled file.

        Parameters
        ----------
        filename : str
            Path to the file containing the pickled Environment object.

        Returns
        -------
        Environment
            The loaded Environment object.
        """
        with open(filename, "rb") as file_handle:
            environment = pickle.load(file_handle)

        return environment

    def get_manifold_distances(
        self, positions1: NDArray[np.float64], positions2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Computes shortest path distance between position pairs along the track.

        Uses precomputed distances on the environment's graph representation
        (N-D bin graph or 1-D augmented graph).

        Parameters
        ----------
        positions1 : NDArray[np.float64], shape (n_time, n_dims) or (n_dims,)
            First set of positions.
        positions2 : NDArray[np.float64], shape (n_time, n_dims) or (n_dims,)
            Second set of positions. Must have the same shape as positions1.

        Returns
        -------
        distances : NDArray[np.float64], shape (n_time,)
            Shortest path distance along the track for each pair. Returns np.inf
            if positions map to bins/nodes with no path between them or if
            mapping fails (e.g., outside track).

        Raises
        ------
        RuntimeError
            If the environment is not fitted.
        ValueError
            If input shapes mismatch or required distance attributes are missing.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")

        positions1 = np.atleast_2d(positions1)
        positions2 = np.atleast_2d(positions2)

        # Validate input shapes
        if positions1.shape != positions2.shape:
            raise ValueError("Shapes of position1 and position2 must match.")

        if (positions1.shape[0] == 0) or (positions2.shape[0] == 0):
            return np.zeros((0,), dtype=np.float64)

        bin_ind1 = self.get_bin_ind(positions1)
        bin_ind2 = self.get_bin_ind(positions2)

        distances = self.distance_between_bins[bin_ind1, bin_ind2]

        return distances

    def get_linear_position(self, position: np.ndarray) -> np.ndarray:
        """Get the linearized position along the track.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_dims)
            Position data.

        Returns
        -------
        linear_position : np.ndarray, shape (n_time,)
            Linearized position along the track.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")
        if not self.is_1d:
            raise ValueError(
                "Linear position calculation is only implemented for 1D environments."
            )

        return get_linearized_position(
            position,
            self.track_graph,
            edge_order=self.edge_order,
            edge_spacing=self.edge_spacing,
        )

    def get_fitted_track_graph(self) -> nx.Graph:
        """Get the fitted track graph of the environment.

        Returns
        -------
        nx.Graph
            The fitted track graph, which includes bin centers.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")
        if self.is_1d:
            return self.track_graph_bin_centers_
        else:
            return self.track_graph_nd_

    @cached_property
    def distance_between_bins(self) -> NDArray[np.float64]:
        """Get the distance between two nodes in the fitted track graph.
        Returns
        -------
        distance : np.ndarray, shape (n_bins, n_bins)
            The distance between all pairs of bins in the fitted track graph.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")

        return _get_distance_between_nodes(self.get_fitted_track_graph())

    def get_bin_center_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with information about the bin centers.

        Returns
        -------
        pd.DataFrame
            DataFrame containing information about the bin centers.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fit yet. Call `fit` first.")

        df = pd.DataFrame.from_dict(
            dict(self.get_fitted_track_graph().nodes(data=True)), orient="index"
        )
        df[["pos_x", "pos_y"]] = pd.DataFrame(df["pos"].tolist(), index=df.index)
        df = df.sort_values(by="bin_ind_flat")

        # set index name to node_id
        df.index.name = "node_id"

        return df

    def get_bin_ind(self, positions: np.ndarray) -> np.ndarray:
        """Get the bin index for a given position.

        Parameters
        ----------
        positions : np.ndarray, shape (n_time, n_dims)
            Position data.

        Returns
        -------
        bin_ind_flat : np.ndarray, shape (n_time,)
            The flattened bin index for each position. To get the 2D index,
            use np.unravel_index(bin_ind_flat, self.centers_shape_).
        """
        df = self.get_bin_center_dataframe()
        df = df[df["is_track_interior"]]
        xy_pos = df.loc[:, ["pos_x", "pos_y"]].to_numpy()
        tree = KDTree(xy_pos)
        positions = np.atleast_2d(positions)
        bin_ind = tree.query(positions, k=1)[1]

        return df["bin_ind_flat"].iloc[bin_ind].to_numpy()

    def get_bin_coordinates(self, bin_ind: np.ndarray) -> np.ndarray:
        """Get the coordinates of the bin centers for given bin indices.

        Parameters
        ----------
        bin_ind : np.ndarray, shape (n_bins,)
            The bin indices.

        Returns
        -------
        bin_coordinates : np.ndarray, shape (n_bins, n_dims)
            The coordinates of the bin centers.
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fit yet. Call `fit` first.")

        return self.place_bin_centers_[bin_ind]

    def assign_region_ids_to_bins(
        self,
        regions_definition: Dict[
            str, Union[List[int], Callable[[NDArray[np.float64]], bool]]
        ],
        default_region_id: Any = -1,
    ) -> NDArray[Any]:
        """Assigns a region identifier to each spatial bin.

        Regions can be defined by lists of bin indices (flat indices) or by a
        function that takes bin center coordinates and returns True if the bin
        belongs to the region.

        Parameters
        ----------
        regions_definition : Dict[str, Union[List[int], Callable[[NDArray[np.float64]], bool]]]
            A dictionary where keys are region names (or numeric IDs) and values define the region.
            - If `List[int]`: A list of bin flat indices belonging to this region.
            - If `Callable`: A function `func(bin_center_coords) -> bool`.
            `bin_center_coords` is an NDArray of shape (n_dims,).
        default_region_id : Any, optional
            Identifier for bins not assigned to any defined region. Defaults to -1.

        Returns
        -------
        region_ids_map : NDArray[Any]
            For 1-D environments, a 1D array of shape (n_bins,) containing the region ID for each bin.
            For N-D environments, a multi-dimensional array of shape matching `centers_shape_`.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.

        Examples
        --------
        >>> env = Environment()
        >>> env.fit(position_data)
        >>> regions = {
        ...     "RegionA": lambda coords: coords[0] < 5,
        ...     "RegionB": [0, 1, 2],
        ... }
        >>> region_ids = env.assign_region_ids_to_bins(regions)
        >>> print(region_ids)
        [0, 0, 0, -1, -1, ...]
        """
        if not self._is_fitted or self.place_bin_centers_ is None:
            raise RuntimeError("Environment has not been fitted. Call fit() first.")

        if self.is_1d:
            raise RuntimeError("1D Environment not fitted")

        num_total_bins: int
        if self.is_1d:
            num_total_bins = self.centers_shape_[0]
            bin_center_df = self.get_bin_center_dataframe()
            all_bin_centers_coords = bin_center_df.loc[:, ["pos_x", "pos_y"]].to_numpy()
            # And their corresponding flat indices (which are just 0 to n-1 for 1D in this context)
            all_bin_flat_indices = bin_center_df["bin_ind_flat"].to_numpy()
            output_shape = (num_total_bins,)
            # Initialize region_ids array based on 1D structure
            region_ids_flat = np.full(num_total_bins, default_region_id, dtype=object)

        else:  # N-D
            num_total_bins = self.place_bin_centers_.shape[0]
            all_bin_centers_coords = self.place_bin_centers_
            # Flat indices are just np.arange(num_total_bins) for N-D as place_bin_centers_ is already flat
            all_bin_flat_indices = np.arange(num_total_bins)
            output_shape = self.centers_shape_
            # Initialize region_ids_flat for N-D; will be reshaped later
            region_ids_flat = np.full(num_total_bins, default_region_id, dtype=object)

        for region_name_or_id, definition in regions_definition.items():
            if callable(definition):
                for i, bin_flat_idx in enumerate(all_bin_flat_indices):
                    # For N-D, bin_flat_idx is the index into all_bin_centers_coords
                    # For 1-D, bin_flat_idx is also the index here.
                    coords = all_bin_centers_coords[
                        i
                    ]  # Assuming all_bin_flat_indices are 0..N-1 sequential
                    if definition(coords):
                        region_ids_flat[bin_flat_idx] = region_name_or_id
            elif isinstance(definition, list):
                # Ensure indices are within bounds
                valid_indices = [idx for idx in definition if 0 <= idx < num_total_bins]
                region_ids_flat[valid_indices] = region_name_or_id
            else:
                raise TypeError(
                    f"Region definition for '{region_name_or_id}' must be a list of "
                    "bin indices or a callable function."
                )

        if self.is_1d:
            return region_ids_flat  # Already 1D
        else:
            return region_ids_flat.reshape(output_shape)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Environment object to a dictionary for DataJoint.
        Starts with self.__dict__ and transforms only types that DataJoint
        might not handle as well directly (e.g., NetworkX graphs).
        Assumes DataJoint can handle np.ndarray and pd.DataFrame objects directly.

        Returns
        -------
        data_dict : Dict[str, Any]
            A dictionary representation of the Environment object.
        """
        # Start with a shallow copy of the instance's dictionary
        data = self.__dict__.copy()

        # Add a class identifier/version for robustness during deserialization
        data["__classname__"] = self.__class__.__name__
        data["__module__"] = self.__class__.__module__
        data["class_version"] = "1.0.0"  # Versioning for future changes
        data["__type__"] = "Environment"

        # Selectively transform types
        for key, value in data.items():
            if isinstance(value, nx.Graph):
                # Convert NetworkX graphs to a more standard dict format for robustness
                # as direct pickling of graphs can sometimes be sensitive to library versions.
                data[key] = {
                    "__type__": "networkx_graph",
                    "node_link_data": nx.node_link_data(value),
                }
            # Pandas DataFrames and NumPy arrays are assumed to be handled directly by DataJoint.

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Environment":
        """Deserializes an Environment object from a dictionary created by to_dict().
        Handles NetworkX graphs generically, including those passed as init arguments.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary representation of the Environment object.

        Returns
        -------
        env : Environment
            The reconstructed Environment object.
        """
        if (
            data.get("__classname__") != cls.__name__
            or data.get("__module__") != cls.__module__
        ):
            raise ValueError(
                f"Dictionary is not for class {cls.__module__}.{cls.__name__}, "
                f"found {data.get('__module__')}.{data.get('__classname__')}"
            )

        # Create a working copy of the data to extract init args from
        # and to allow modification of values (e.g., reconstructing graphs)
        construction_data = data.copy()

        # Collect and reconstruct init arguments for the class constructor
        init_args = {}
        for f in fields(cls):  # `fields` from `dataclasses`
            if f.init:
                if f.name in construction_data:
                    value = construction_data[f.name]
                    # Check if this init arg's value is a serialized graph
                    if (
                        isinstance(value, dict)
                        and value.get("__type__") == "networkx_graph"
                    ):
                        init_args[f.name] = nx.node_link_graph(value["node_link_data"])
                    else:
                        init_args[f.name] = value
                elif f.default is not MISSING:
                    init_args[f.name] = f.default
                elif f.default_factory is not MISSING:
                    init_args[f.name] = f.default_factory()
                # If a required init arg (no default) is missing, cls(**init_args) will fail.

        # Create the instance
        env = cls(**init_args)

        # Restore the rest of the attributes from the original data dictionary
        # These are attributes not handled by __init__ (e.g., fitted attributes, cached properties)
        for key, value in data.items():
            if key not in init_args and key not in [
                "__classname__",
                "__module__",
                "class_version",
            ]:
                restored_value = value

                # Check if the value is a serialized graph
                if (
                    isinstance(value, dict)
                    and value.get("__type__") == "networkx_graph"
                ):
                    restored_value = nx.node_link_graph(value["node_link_data"])

                # Setting the attribute, including populating __dict__ for cached_properties
                setattr(env, key, restored_value)

        return env
