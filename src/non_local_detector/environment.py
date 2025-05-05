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
    - Constructs a `networkx` graph (`track_graphDD`) where nodes represent
      the centers of *interior* bins, and edges connect adjacent interior bins.
      This graph captures the connectivity of the valid space.
    - Can compute shortest-path distances between all pairs of interior bins
      on this graph (`distance_between_nodes_`).
    - Provides methods to find the bin index for a given position (`get_bin_ind`),
      calculate manifold distances between positions (`get_manifold_distances`),
      and determine movement direction relative to the track center (`get_direction`).

2.  **1-Dimensional Environments (Linear Tracks, W-Tracks):**
    - Requires a `networkx.Graph` (`track_graph`) defining the track's topology
      (nodes, edges, and their positions) along with edge ordering and spacing.
    - Linearizes the track based on the provided graph structure.
    - Creates bins along this linearized track.
    - Generates an augmented graph (`track_graph_with_bin_centers_edges_`) where
      both the original track nodes and the newly created bin centers/edges
      are represented as nodes.
    - Computes shortest-path distances between all nodes in this augmented graph
      (`distance_between_nodes_`).
    - Stores detailed information about original nodes, bin edges, and bin
      centers in pandas DataFrames (`original_nodes_df_`, etc.).

The central component is the `Environment` dataclass, which holds the parameters
defining the environment and stores the results of the fitting process (grid
details, graphs, distances) as attributes. The primary method `fit_place_grid`
is used to perform the discretization and graph construction based on the input
parameters and optional position data.

Helper functions are provided for tasks like grid generation (`get_grid`),
track interior inference (`get_track_interior`), graph construction
(`make_nD_track_graph_from_environment`, `get_track_grid`), distance calculation,
and visualization (`plot_grid`). The class also supports saving and loading
environment definitions using `pickle`.
"""

import pickle
from dataclasses import dataclass, field
from itertools import accumulate, combinations
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from track_linearization import plot_graph_as_1D

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
    if np.all(bin_size <= 0):
        raise ValueError("bin_size must be positive.")

    # Calculate number of bins, ensuring at least 1 bin even if extent is 0
    n_bins = np.ceil(extent / bin_size).astype(np.int32)
    n_bins[n_bins == 0] = 1  # Handle zero extent case

    return n_bins


def create_grid(
    position: Optional[NDArray[np.float64]] = None,
    bin_size: Union[float, Sequence[float]] = 2.0,
    position_range: Optional[Sequence[Tuple[float, float]]] = None,
    add_boundary_bins: bool = True,
) -> Tuple[
    Tuple[NDArray[np.float64], ...],
    NDArray[np.float64],
    Tuple[int, ...],
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

        if close_gaps:
            # Use connectivity=1 for 4-neighbor (2D) or 6-neighbor (3D) etc.
            structure = ndimage.generate_binary_structure(n_dims, connectivity=1)

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


def get_track_boundary(
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
            node_id, pos=tuple(node_position), is_track_interior=is_interior
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
    node_positions = nx.get_node_attributes(track_graph_nd, "pos")
    node_positions = np.asarray(list(node_positions.values()))
    distance = np.full((len(node_positions), len(node_positions)), np.inf)
    for to_node_id, from_node_id in nx.shortest_path_length(
        track_graph_nd,
        weight="distance",
    ):
        distance[to_node_id, list(from_node_id.keys())] = list(from_node_id.values())

    return distance


@dataclass
class Environment:
    """Represent the spatial environment with a discrete grid.

    Parameters
    ----------
    environment_name : str, optional
        Identifier for the environment. Defaults to "".
    place_bin_size : float or tuple[float], optional
        Approximate size of the position bins in each dimension. Defaults to 2.0.
    track_graph : networkx.Graph, optional
        Graph representing the 1D spatial topology. If provided, 1D methods are used.
    edge_order : tuple of 2-tuples, optional
        Required if `track_graph` is provided. The order of the edges in 1D space.
    edge_spacing : float or list, optional
        Required if `track_graph` is provided. Spacing between edges. Defaults to 0.
    is_track_interior : np.ndarray or None, optional
        Boolean array defining valid track areas. If None and `infer_track_interior`
        is True, it will be inferred from position data.
    position_range : Sequence[Tuple[float, float]], optional
        Outer bin edges for each dimension [(min_dim1, max_dim1), ...].
        If None, range is determined from position data.
    infer_track_interior : bool, optional
        If True and `is_track_interior` is None, infer track geometry from position data.
        Defaults to True. Ignored if `track_graph` is provided.
    close_gaps : bool, optional
        Close gaps in the inferred track interior. Defaults to False. Ignored if `track_graph` is provided.
    fill_holes : bool, optional
        Fill holes in the inferred track interior. Defaults to False. Ignored if `track_graph` is provided.
    dilate : bool, optional
        Inflate the inferred track interior. Defaults to False. Ignored if `track_graph` is provided.
    bin_count_threshold : int, optional
        Minimum number of samples in a bin to be considered part of the track interior.
        Defaults to 0. Ignored if `track_graph` is provided.

    Attributes
    ----------
    edges_ : tuple[np.ndarray, ...]
        Bin edges for each dimension.
    place_bin_edges_ : np.ndarray, shape (n_bins + 1, n_dims) or (n_total_bins + n_edges, n_dims)
        Edges of the place bins (linearized or N-D).
    place_bin_centers_ : np.ndarray, shape (n_bins, n_dims)
        Center coordinates of each place bin.
    centers_shape_ : tuple[int, ...]
        Shape of the grid in terms of bins per dimension.
    is_track_interior_ : np.ndarray, shape (n_bins,) or (n_bins_dim1, n_bins_dim2, ...)
        Boolean array indicating which bins are part of the track interior.
    is_track_boundary_ : np.ndarray or None
        Boolean array indicating boundary bins (only for N-D environments).
    track_graphDD : networkx.Graph or None
        Graph representation where nodes are bin centers (only for N-D environments).
    distance_between_nodes_ : Dict[int, Dict[int, float]] or np.ndarray
        Shortest path distances between nodes on the track graph (1D or N-D).
    track_graph_with_bin_centers_edges_ : nx.Graph or None
        Track graph with bin centers and edges added as nodes (only for 1D).
    original_nodes_df_ : pd.DataFrame or None
        Info about original track graph nodes (only for 1D).
    place_bin_edges_nodes_df_ : pd.DataFrame or None
        Info about nodes representing bin edges (only for 1D).
    place_bin_centers_nodes_df_ : pd.DataFrame or None
        Info about nodes representing bin centers (only for 1D).
    nodes_df_ : pd.DataFrame or None
        Combined info about all nodes in the augmented graph (only for 1D).
    """

    environment_name: str = ""
    place_bin_size: Union[float, Tuple[float, ...]] = 2.0
    track_graph: Optional[nx.Graph] = None
    edge_order: Optional[List[Tuple]] = None
    edge_spacing: Optional[Union[float, List[float]]] = 0.0
    is_track_interior: Optional[np.ndarray] = None
    position_range: Optional[Sequence[Tuple[float, float]]] = None
    infer_track_interior: bool = True
    close_gaps: bool = False
    fill_holes: bool = False
    dilate: bool = False
    bin_count_threshold: int = 0

    # Attributes to be fitted
    edges_: Optional[Tuple[np.ndarray, ...]] = None
    place_bin_edges_: Optional[np.ndarray] = None
    place_bin_centers_: Optional[np.ndarray] = None
    centers_shape_: Optional[Tuple[int, ...]] = None
    is_track_interior_: Optional[np.ndarray] = None
    is_track_boundary_: Optional[np.ndarray] = None
    track_graphDD: Optional[nx.Graph] = None  # For N-D case
    distance_between_nodes_: Optional[
        Union[Dict[int, Dict[int, float]], np.ndarray]
    ] = None
    track_graph_with_bin_centers_edges_: Optional[nx.Graph] = None  # For 1D case
    original_nodes_df_: Optional[pd.DataFrame] = None
    place_bin_edges_nodes_df_: Optional[pd.DataFrame] = None
    place_bin_centers_nodes_df_: Optional[pd.DataFrame] = None
    nodes_df_: Optional[pd.DataFrame] = None
    # Internal flag
    _is_fitted: bool = False

    def __eq__(self, other: str) -> bool:
        return self.environment_name == other

    def fit(
        self, position: Optional[np.ndarray] = None, infer_track_interior: bool = True
    ) -> "Environment":
        """Fits the environment based on provided position data.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_dim), optional
            Position of the animal.
        infer_track_interior : bool, optional
            Whether to infer the spatial geometry of track from position

        Returns
        -------
        self

        """
        return self.fit_place_grid(position, infer_track_interior)

    def fit_place_grid(
        self, position: Optional[np.ndarray] = None, infer_track_interior: bool = True
    ) -> "Environment":
        """Fits a discrete grid of the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_dim), optional
            Position of the animal.
        infer_track_interior : bool, optional
            Whether to infer the spatial geometry of track from position

        Returns
        -------
        self

        """
        if self.track_graph is None:
            (
                self.edges_,
                self.place_bin_edges_,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = create_grid(
                position,
                self.place_bin_size,
                self.position_range,
            )

            self.infer_track_interior = infer_track_interior

            if self.is_track_interior is None and self.infer_track_interior:
                self.is_track_interior_ = _infer_track_interior(
                    position,
                    self.edges_,
                    self.close_gaps,
                    self.fill_holes,
                    self.dilate,
                    self.bin_count_threshold,
                )
            elif self.is_track_interior is None and not self.infer_track_interior:
                self.is_track_interior_ = np.ones(self.centers_shape_, dtype=bool)

            if len(self.edges_) > 1:
                self.is_track_boundary_ = get_track_boundary(
                    self.is_track_interior_,
                    connectivity=1,
                )
            else:
                self.is_track_boundary_ = None

            self.track_graph_nd = _make_nd_track_graph(
                self.place_bin_centers_,
                self.is_track_interior_,
                self.centers_shape_,
            )

            self.distance_between_nodes_ = get_distance_between_nodes(
                self.track_graph_nd,
            )

        else:
            (
                self.place_bin_centers_,
                self.place_bin_edges_,
                self.is_track_interior_,
                self.distance_between_nodes_,
                self.centers_shape_,
                self.edges_,
                self.track_graph_with_bin_centers_edges_,
                self.original_nodes_df_,
                self.place_bin_edges_nodes_df_,
                self.place_bin_centers_nodes_df_,
                self.nodes_df_,
            ) = get_track_grid(
                self.track_graph,
                self.edge_order,
                self.edge_spacing,
                self.place_bin_size,
            )
            self.is_track_boundary_ = None

        self._is_fitted = True

        return self

    def plot_grid(self, ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
        """Plot the fitted spatial grid of the environment.

        Parameters
        ----------
        ax : plt.axes, optional
            Plot on this axis if given, by default None

        Returns
        -------
        ax : plt.axes
            The axis on which the grid is plotted.

        """
        if self.track_graph is not None:
            if ax is None:
                _, ax = plt.subplots(figsize=(15, 2))

            plot_graph_as_1D(
                self.track_graph, self.edge_order, self.edge_spacing, ax=ax
            )
            try:
                for edge in self.edges_[0]:
                    ax.axvline(edge.squeeze(), linewidth=0.5, color="black")
            except AttributeError:
                # Edges have not been fit yet
                pass
            ax.set_ylim((0, 0.1))
        else:
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 7))
            ax.pcolormesh(
                self.edges_[0], self.edges_[1], self.is_track_interior_.T, cmap="bone_r"
            )
            ax.set_xticks(self.edges_[0], minor=True)
            ax.set_yticks(self.edges_[1], minor=True)
            ax.grid(visible=True, which="minor")
            ax.grid(visible=False, which="major")

        return ax

    def save_environment(self, filename: str = "environment.pkl") -> None:
        """Saves the environment object as a pickled file.

        Parameters
        ----------
        filename : str, optional
            File name to save the environment to. Defaults to "environment.pkl".
        """
        with open(filename, "wb") as file_handle:
            pickle.dump(self, file_handle)

    @classmethod
    def load_environment(cls, filename: str = "environment.pkl") -> "Environment":
        """Loads a pickled environment object from a file.

        Parameters
        ----------
        filename : str, optional
            File name to load the environment from. Defaults to "environment.pkl".

        Returns
        -------
        Environment
            The loaded environment object.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_bin_ind(self, sample: np.ndarray) -> np.ndarray:
        """Find the indices of the bins to which each value in input array belongs.

        Uses the fitted grid edges (`self.edges_`).

        Parameters
        ----------
        sample : np.ndarray, shape (n_time, n_dim)
            Input data points.

        Returns
        -------
        bin_inds : np.ndarray, shape (n_time,)
            Flat index of the bin for each data point in `sample`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Environment has not been fitted yet. Call `fit_place_grid` first."
            )
        if self.edges_ is None:
            raise ValueError("Environment edges `edges_` are not defined.")

        # remove outer boundary edge
        edges = [e[1:-1] for e in self.edges_]

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
            np.searchsorted(edges[i], sample[:, i], side="right") for i in range(D)
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

    def get_manifold_distances(
        self, position1: np.ndarray, position2: np.ndarray
    ) -> np.ndarray:
        """Computes the distance between pairs of positions along the track manifold.

        This uses the precomputed shortest path distances between bin centers on the
        graph representation of the environment (either 1D or N-D).

        Parameters
        ----------
        position1 : np.ndarray, shape (n_time, n_dims) or (n_dims,)
            The first set of positions.
        position2 : np.ndarray, shape (n_time, n_dims) or (n_dims,)
             The second set of positions. Must have the same shape as position1.

        Returns
        -------
        distances : np.ndarray, shape (n_time,)
            The shortest path distance along the track for each pair of positions.
            Returns np.inf if no path exists between the bins corresponding to the positions.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        ValueError
            If input shapes mismatch or required attributes are missing.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Environment has not been fitted yet. Call `fit_place_grid` first."
            )
        if self.distance_between_nodes_ is None:
            raise ValueError("Distance between nodes has not been computed or stored.")

        position1 = np.atleast_2d(position1)
        position2 = np.atleast_2d(position2)

        # Validate input shapes
        if position1.shape != position2.shape:
            raise ValueError("Shapes of position1 and position2 must match.")

        bin_ind1 = self.get_bin_ind(position1)
        bin_ind2 = self.get_bin_ind(position2)

        if self.track_graph is not None:  # 1D case uses dict
            raise NotImplementedError(
                "Distance calculation for 1D track graph is not implemented."
            )
        else:
            distances = self.distance_between_nodes_[bin_ind1, bin_ind2]

        return distances

    def get_direction(
        self,
        position: np.ndarray,
        position_time: Optional[np.ndarray] = None,
        sigma: float = 0.1,
        sampling_frequency: Optional[float] = None,
        classify_stop: bool = False,
        stop_speed_threshold: float = 1e-3,
    ) -> np.ndarray:
        """Get the direction of movement relative to the center of the track (inward/outward).

        Requires a fitted N-D environment with a corresponding track graph (`track_graphDD`).

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

        if not self._is_fitted:
            raise RuntimeError(
                "Environment has not been fitted yet. Call `fit_place_grid` first."
            )
        if self.track_graphDD is None or self.distance_between_nodes_ is None:
            raise RuntimeError(
                "Direction finding requires a fitted N-D environment with a track graph ('track_graphDD') and precomputed distances."
            )

        if position_time is None:
            position_time = np.arange(position.shape[0])
        if sampling_frequency is None:
            sampling_frequency = 1 / np.mean(np.diff(position_time))

        centrality = nx.closeness_centrality(self.track_graphDD, distance="distance")
        center_node_id = list(centrality.keys())[np.argmax(list(centrality.values()))]

        bin_ind = self.get_bin_ind(position)

        velocity_to_center = gaussian_smooth(
            np.gradient(self.distance_between_nodes_[bin_ind, center_node_id]),
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


def get_track_segments_from_graph(track_graph: nx.Graph) -> np.ndarray:
    """Returns a 2D array of node positions corresponding to each edge.

    Parameters
    ----------
    track_graph : networkx Graph

    Returns
    -------
    track_segments : np.ndarray, shape (n_segments, n_nodes, n_space)

    """
    node_positions = nx.get_node_attributes(track_graph, "pos")
    return np.asarray(
        [
            (node_positions[node1], node_positions[node2])
            for node1, node2 in track_graph.edges()
        ]
    )


def project_points_to_segment(
    track_segments: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Finds the closet point on a track segment in terms of Euclidean distance

    Parameters
    ----------
    track_segments : np.ndarray, shape (n_segments, n_nodes, 2)
    position : np.ndarray, shape (n_time, 2)

    Returns
    -------
    projected_positions : np.ndarray, shape (n_time, n_segments, n_space)

    """
    segment_diff = np.diff(track_segments, axis=1).squeeze(axis=1)
    sum_squares = np.sum(segment_diff**2, axis=1)
    node1 = track_segments[:, 0, :]
    nx = (
        np.sum(segment_diff * (position[:, np.newaxis, :] - node1), axis=2)
        / sum_squares
    )
    nx[np.where(nx < 0)] = 0.0
    nx[np.where(nx > 1)] = 1.0
    return node1[np.newaxis, ...] + (
        nx[:, :, np.newaxis] * segment_diff[np.newaxis, ...]
    )


def _calculate_linear_position(
    track_graph: nx.Graph,
    position: np.ndarray,
    track_segment_id: np.ndarray,
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines the linear position given a 2D position and a track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    position : np.ndarray, shape (n_time, n_position_dims)
    track_segment_id : np.ndarray, shape (n_time,)
    edge_order : list of 2-tuples
    edge_spacing : float or list, len n_edges - 1

    Returns
    -------
    linear_position : np.ndarray, shape (n_time,)
    projected_track_positions_x : np.ndarray, shape (n_time,)
    projected_track_positions_y : np.ndarray, shape (n_time,)

    """
    is_nan = np.isnan(track_segment_id)
    track_segment_id[is_nan] = 0  # need to check
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[
        (np.arange(n_time), track_segment_id)
    ]

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) or isinstance(edge_spacing, float):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    counter = 0.0
    start_node_linear_position = []

    for ind, edge in enumerate(edge_order):
        start_node_linear_position.append(counter)

        try:
            counter += track_graph.edges[edge]["distance"] + edge_spacing[ind]
        except IndexError:
            pass

    start_node_linear_position = np.asarray(start_node_linear_position)

    track_segment_id_to_start_node_linear_position = {
        track_graph.edges[e]["edge_id"]: snlp
        for e, snlp in zip(edge_order, start_node_linear_position)
    }

    start_node_linear_position = np.asarray(
        [
            track_segment_id_to_start_node_linear_position[edge_id]
            for edge_id in track_segment_id
        ]
    )

    track_segment_id_to_edge = {track_graph.edges[e]["edge_id"]: e for e in edge_order}
    start_node_id = np.asarray(
        [track_segment_id_to_edge[edge_id][0] for edge_id in track_segment_id]
    )
    start_node_2D_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in start_node_id]
    )

    linear_position = start_node_linear_position + (
        np.linalg.norm(start_node_2D_position - projected_track_positions, axis=1)
    )
    linear_position[is_nan] = np.nan

    return (
        linear_position,
        projected_track_positions[:, 0],
        projected_track_positions[:, 1],
    )


def make_track_graph_with_bin_centers_edges(
    track_graph: nx.Graph, place_bin_size: float
) -> nx.Graph:
    """Insert the bin center and bin edge positions as nodes in the track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    place_bin_size : float

    Returns
    -------
    track_graph_with_bin_centers_edges : nx.Graph

    """
    track_graph_with_bin_centers_edges = track_graph.copy()
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
            track_graph_with_bin_centers_edges,
            [*new_node_ids],
            distance=dist_between_nodes[0],
        )
        nx.add_path(
            track_graph_with_bin_centers_edges, [node1, new_node_ids[0]], distance=0
        )
        nx.add_path(
            track_graph_with_bin_centers_edges, [node2, new_node_ids[-1]], distance=0
        )
        track_graph_with_bin_centers_edges.remove_edge(node1, node2)
        for ind, (node_id, pos) in enumerate(zip(new_node_ids, xy)):
            track_graph_with_bin_centers_edges.nodes[node_id]["pos"] = pos
            track_graph_with_bin_centers_edges.nodes[node_id]["edge_id"] = edge_ind
            if ind % 2:
                track_graph_with_bin_centers_edges.nodes[node_id]["is_bin_edge"] = False
            else:
                track_graph_with_bin_centers_edges.nodes[node_id]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node1]["edge_id"] = edge_ind
        track_graph_with_bin_centers_edges.nodes[node2]["edge_id"] = edge_ind
        track_graph_with_bin_centers_edges.nodes[node1]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node2]["is_bin_edge"] = True
        n_nodes = len(track_graph_with_bin_centers_edges.nodes)

    return track_graph_with_bin_centers_edges


def extract_bin_info_from_track_graph(
    track_graph: nx.Graph,
    track_graph_with_bin_centers_edges: nx.Graph,
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
) -> pd.DataFrame:
    """For each node, find edge_id, is_bin_edge, x_position, y_position, and
    linear_position.

    Parameters
    ----------
    track_graph : nx.Graph
    track_graph_with_bin_centers_edges : nx.Graph
    edge_order : list of 2-tuples
    edge_spacing : list, len n_edges - 1

    Returns
    -------
    nodes_df : pd.DataFrame
        Collect information about each bin

    """
    nodes_df = (
        pd.DataFrame.from_dict(
            dict(track_graph_with_bin_centers_edges.nodes(data=True)), orient="index"
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


def get_track_grid(
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
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
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
    track_graph_with_bin_centers_edges : nx.Graph
    original_nodes_df : pd.DataFrame
        Table of information about the original nodes in the track graph
    place_bin_edges_nodes_df : pd.DataFrame
        Table of information with bin edges and centers
    place_bin_centers_nodes_df : pd.DataFrame
        Table of information about bin centers
    nodes_df : pd.DataFrame
        Table of information with information about the original nodes,
        bin edges, and bin centers

    """
    track_graph_with_bin_centers_edges = make_track_graph_with_bin_centers_edges(
        track_graph, place_bin_size
    )
    nodes_df = extract_bin_info_from_track_graph(
        track_graph, track_graph_with_bin_centers_edges, edge_order, edge_spacing
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

    # Compute distance between nodes
    distance_between_nodes = dict(
        nx.all_pairs_dijkstra_path_length(
            track_graph_with_bin_centers_edges, weight="distance"
        )
    )

    # Figure out which points are on the track and not just gaps
    change_edge_ind = np.nonzero(
        np.diff(no_duplicate_place_bin_edges_nodes_df.edge_id)
    )[0]

    if isinstance(edge_spacing, int) or isinstance(edge_spacing, float):
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

    # Other needed information
    edges = [place_bin_edges]
    centers_shape = (place_bin_centers.size,)

    return (
        place_bin_centers[:, np.newaxis],
        place_bin_edges[:, np.newaxis],
        is_track_interior,
        distance_between_nodes,
        centers_shape,
        edges,
        track_graph_with_bin_centers_edges,
        original_nodes_df,
        place_bin_edges_nodes_df,
        place_bin_centers_nodes_df,
        nodes_df.reset_index(),
    )


def order_boundary(boundary: np.ndarray) -> np.ndarray:
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
    n_position_dims = len(edges)
    boundary = get_track_boundary(
        is_track_interior, n_position_dims=n_position_dims, connectivity=connectivity
    )

    inds = np.nonzero(boundary)
    centers = [get_centers(x) for x in edges]
    boundary = np.stack([center[ind] for center, ind in zip(centers, inds)], axis=1)
    return order_boundary(boundary)


def make_nD_track_graph_from_environment(environment: Environment) -> nx.Graph:
    """Create a graph of the track with nodes at the center of each on track bin and
    edges between adjacent bins on the track.

    Parameters
    ----------
    environment : Environment

    Returns
    -------
    track_graph : nx.Graph

    """
    track_graph = nx.Graph()
    axis_offsets = [-1, 0, 1]

    # Enumerate over nodes
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
    # Enumerate over nodes in the track interior
    for ind in zip(*np.nonzero(environment.is_track_interior_)):
        ind = np.array(ind)
        # Indices of adjacent nodes
        adj_inds = np.meshgrid(*[axis_offsets + i for i in ind], indexing="ij")
        # Remove out of bounds indices
        adj_inds = [
            inds[np.logical_and(inds >= 0, inds < dim_size)]
            for inds, dim_size in zip(adj_inds, environment.centers_shape_)
        ]

        # Is the adjacent node on the track?
        adj_on_track_inds = environment.is_track_interior_[tuple(adj_inds)]

        # Remove the center node
        center_idx = [n // 2 for n in adj_on_track_inds.shape]
        adj_on_track_inds[tuple(center_idx)] = False

        # Get the node ids of the center node
        node_id = np.ravel_multi_index(ind, environment.centers_shape_)

        # Get the node ids of the adjacent nodes on the track
        adj_node_ids = np.ravel_multi_index(
            [inds[adj_on_track_inds] for inds in adj_inds],
            environment.centers_shape_,
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
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


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
    # --- Compute and add a new attribute 'distance_weight' = 1 / distance ---
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


def make_track_graph_with_bin_centers(env: Environment) -> nx.Graph:
    """Creates a graph connecting bin centers sequentially along the track.

    Builds a graph using only bin center nodes. Connects adjacent centers
    within segments based on linear position. Links segment endpoints meeting
    at original track graph nodes by looking via intermediate nodes in the
    augmented graph.

    Parameters
    ----------
    env : Environment
        A fitted Environment object for a 1D track.

    Returns
    -------
    track_graph_with_bin_centers : nx.Graph
        Graph with bin centers as nodes, linked sequentially and at junctions.

    Raises
    ------
    RuntimeError, ValueError, KeyError as before.
    """
    if not env._is_fitted:
        raise RuntimeError("Environment not fitted.")
    if env.track_graph is None:
        raise RuntimeError("Requires a 1D environment.")
    required_attrs = [
        "place_bin_centers_nodes_df_",
        "original_nodes_df_",
        "track_graph_with_bin_centers_edges_",
    ]
    for attr in required_attrs:
        if getattr(env, attr, None) is None:
            raise ValueError(f"Missing required attribute '{attr}'.")

    track_graph_with_bin_centers = nx.Graph()
    centers_df = env.place_bin_centers_nodes_df_.copy()

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
                },
            )
        )
    track_graph_with_bin_centers.add_nodes_from(nodes_to_add)

    # --- 2. Add Intra-Segment Edges ---
    edges_to_add: List[Tuple[Any, Any, Dict[str, Any]]] = []
    intra_segment_edge_count = 0
    for edge_id, group in valid_centers_df.groupby("edge_id"):
        sorted_group = group.sort_values("linear_position")
        node_ids = sorted_group["node_id"].values
        for node1, node2 in zip(node_ids[:-1], node_ids[1:]):
            try:
                pos1 = _get_node_pos(track_graph_with_bin_centers, node1)
                pos2 = _get_node_pos(track_graph_with_bin_centers, node2)
            except KeyError as e:
                continue
            distance = np.linalg.norm(pos1 - pos2)
            if distance > 1e-9:
                edges_to_add.append(
                    (node1, node2, {"distance": distance, "edge_id": edge_id})
                )
                intra_segment_edge_count += 1
    track_graph_with_bin_centers.add_edges_from(edges_to_add)

    # --- 3. Add Inter-Segment Edges (Link segment ends) ---
    linking_edges_to_add: List[Tuple[Any, Any, Dict[str, Any]]] = []
    inter_segment_edge_count = 0
    augmented_graph = env.track_graph_with_bin_centers_edges_
    original_node_ids_in_augmented = env.original_nodes_df_["node_id"].unique()

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
                    and track_graph_with_bin_centers.has_node(potential_center)
                ):
                    endpoint_centers.add(potential_center)

        # Add edges between all pairs of these endpoint bin centers
        for node1, node2 in combinations(endpoint_centers, 2):
            # Avoid adding edges already added or self-loops (combinations handles self-loops)
            if not track_graph_with_bin_centers.has_edge(node1, node2):
                try:
                    pos1 = _get_node_pos(track_graph_with_bin_centers, node1)
                    pos2 = _get_node_pos(track_graph_with_bin_centers, node2)
                except KeyError as e:
                    continue

                distance = np.linalg.norm(pos1 - pos2)
                linking_edges_to_add.append(
                    (node1, node2, {"distance": distance, "edge_id": -1})
                )
                inter_segment_edge_count += 1

    track_graph_with_bin_centers.add_edges_from(linking_edges_to_add)

    return track_graph_with_bin_centers
