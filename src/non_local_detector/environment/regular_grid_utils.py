"""
Utility functions for creating and managing regular N-dimensional grid layouts.

This module provides a collection of helper functions used primarily by
`RegularGridLayout` and other grid-based layout engines within the
`non_local_detector.environment` package. These functions handle tasks such as:

- Defining the structure (bin edges, bin centers, shape) of a regular N-D grid
  based on data samples or specified dimension ranges (`_create_regular_grid`).
- Inferring which bins within this grid are "active" based on the density of
  provided data samples, often involving morphological operations to refine
  the active area (`_infer_active_bins_from_regular_grid`).
- Constructing a `networkx.Graph` that represents the connectivity between
  these active grid bins, allowing for orthogonal and diagonal connections
  (`_create_regular_grid_connectivity_graph`).
- Mapping continuous N-D points to their corresponding discrete bin indices
  within the grid, taking into account active areas (`_points_to_regular_grid_bin_ind`).

The module also includes functions for inferring dimensional properties from
data samples, which might be shared with or used by other utility modules.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import KDTree

from non_local_detector.environment.utils import get_centers, get_n_bins


def _create_regular_grid_connectivity_graph(
    full_grid_bin_centers: NDArray[np.float64],
    active_mask_nd: NDArray[np.bool_],
    grid_shape: Tuple[int, ...],
    connect_diagonal: bool = False,
) -> nx.Graph:
    """
    Create a graph connecting centers of active bins in an N-D grid.

    Nodes in the returned graph are indexed from `0` to `n_active_bins - 1`.
    Each node stores attributes:
    - 'pos': N-D coordinates of the active bin center.
    - 'source_grid_flat_index': Original flat index in the full conceptual grid.
    - 'original_grid_nd_index': Original N-D tuple index in the full grid.

    Edges connect active bins that are adjacent (orthogonally or, optionally,
    diagonally) in the N-D grid. Edges store 'distance' and 'weight'.

    Parameters
    ----------
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Coordinates of centers of *all* bins in the original full grid,
        ordered by flattened grid index (row-major).
    active_mask_nd : NDArray[np.bool_], shape (dim0_size, dim1_size, ...)
        N-dimensional boolean mask indicating active bins in the full grid.
        Must match `grid_shape`.
    grid_shape : Tuple[int, ...]
        The N-D shape (number of bins in each dimension) of the original full grid.
    connect_diagonal : bool, default=False
        If True, connect diagonally adjacent active bins. Otherwise, only
        orthogonally adjacent active bins are connected.

    Returns
    -------
    connectivity_graph : nx.Graph
        Graph of active bins. Nodes are re-indexed `0` to `n_active_bins - 1`.

    Raises
    ------
    ValueError
        If input shapes (`full_grid_bin_centers`, `active_mask_nd`,
        `grid_shape`) are inconsistent.
    """
    if full_grid_bin_centers.shape[0] != np.prod(grid_shape):
        raise ValueError(
            f"Mismatch: full_grid_bin_centers length ({full_grid_bin_centers.shape[0]}) "
            f"and product of grid_shape ({np.prod(grid_shape)})."
        )
    if active_mask_nd.shape != grid_shape:
        raise ValueError(
            f"Shape of active_mask_nd {active_mask_nd.shape} "
            f"does not match grid_shape {grid_shape}."
        )

    n_dims = active_mask_nd.ndim
    connectivity_graph = nx.Graph()

    # 1. Identify active bins and create mapping from original flat index to new node ID
    active_original_flat_indices = np.flatnonzero(active_mask_nd)
    n_active_bins = len(active_original_flat_indices)

    if n_active_bins == 0:
        return connectivity_graph  # Return an empty graph

    # Map: original_full_grid_flat_index -> new_active_bin_node_id (0 to n_active_bins-1)
    original_flat_to_new_node_id_map: Dict[int, int] = {
        original_idx: new_idx
        for new_idx, original_idx in enumerate(active_original_flat_indices)
    }

    # 2. Add nodes to the graph with new IDs (0 to n_active_bins-1) and attributes
    for new_node_id in range(n_active_bins):
        original_flat_idx = active_original_flat_indices[new_node_id]
        original_nd_idx_tuple = tuple(np.unravel_index(original_flat_idx, grid_shape))
        pos_coordinates = tuple(full_grid_bin_centers[original_flat_idx])

        connectivity_graph.add_node(
            new_node_id,
            pos=pos_coordinates,
            source_grid_flat_index=int(original_flat_idx),
            original_grid_nd_index=original_nd_idx_tuple,
        )

    # 3. Add edges between these new active node IDs
    # Iterate through each active bin using its *original* N-D index
    active_original_nd_indices_list = np.array(np.nonzero(active_mask_nd)).T

    # Define neighbor offsets
    if connect_diagonal:
        # All combinations of -1, 0, 1 across n_dims, excluding (0,0,...)
        neighbor_offsets = [
            offset
            for offset in itertools.product([-1, 0, 1], repeat=n_dims)
            if not all(o == 0 for o in offset)
        ]
    else:  # Orthogonal neighbors only
        neighbor_offsets = []
        for dim_idx in range(n_dims):
            for offset_val in [-1, 1]:
                offset = [0] * n_dims
                offset[dim_idx] = offset_val
                neighbor_offsets.append(tuple(offset))

    edges_to_add_with_attrs = []

    for current_original_nd_idx_arr in active_original_nd_indices_list:
        # current_original_nd_idx_arr is like np.array([r, c, z])
        current_original_flat_idx = np.ravel_multi_index(
            current_original_nd_idx_arr, grid_shape
        )
        u_new = original_flat_to_new_node_id_map[current_original_flat_idx]

        for offset_tuple in neighbor_offsets:
            neighbor_original_nd_idx_arr = current_original_nd_idx_arr + np.array(
                offset_tuple
            )
            neighbor_original_nd_idx_tuple = tuple(neighbor_original_nd_idx_arr)

            # Check if neighbor is within grid bounds
            if not all(
                0 <= neighbor_original_nd_idx_arr[d] < grid_shape[d]
                for d in range(n_dims)
            ):
                continue

            # Check if this neighbor is also active
            if active_mask_nd[neighbor_original_nd_idx_tuple]:
                neighbor_original_flat_idx = np.ravel_multi_index(
                    neighbor_original_nd_idx_tuple, grid_shape
                )
                v_new = original_flat_to_new_node_id_map[neighbor_original_flat_idx]

                # Add edge if u_new < v_new to avoid duplicates and self-loops (though u_new should not equal v_new here)
                if u_new < v_new:
                    pos_u = np.asarray(connectivity_graph.nodes[u_new]["pos"])
                    pos_v = np.asarray(connectivity_graph.nodes[v_new]["pos"])
                    distance = float(np.linalg.norm(pos_u - pos_v))
                    displacement_vector = pos_v - pos_u
                    weight = 1.0 / distance if distance > 0.0 else np.inf
                    edge_attrs: Dict[str, Any] = {
                        "distance": distance,
                        "vector": tuple(displacement_vector.tolist()),
                        "weight": weight,
                    }
                    if n_dims == 2:
                        edge_attrs["angle_2d"] = math.atan2(
                            displacement_vector[1], displacement_vector[0]
                        )

                    edges_to_add_with_attrs.append((u_new, v_new, edge_attrs))

    # Add all edges with their attributes
    connectivity_graph.add_edges_from(edges_to_add_with_attrs)

    # Add edge IDs to the graph
    # This is a unique ID for each edge in the graph, starting from 0
    # and incrementing by 1 for each edge
    for edge_id_counter, (u, v) in enumerate(connectivity_graph.edges()):
        connectivity_graph.edges[u, v]["edge_id"] = edge_id_counter

    return connectivity_graph


def _infer_active_bins_from_regular_grid(
    data_samples: NDArray[np.float64],
    edges: Tuple[NDArray[np.float64], ...],
    close_gaps: bool = False,
    fill_holes: bool = False,
    dilate: bool = False,
    bin_count_threshold: int = 0,
    boundary_exists: bool = False,
) -> NDArray[np.bool_]:
    """
    Infer active bins in a regular grid based on data sample density.

    This function first counts data samples in each grid bin defined by `edges`.
    Bins with counts above `bin_count_threshold` are initially marked active.
    Optional morphological operations (closing, filling, dilation) can then be
    applied to refine the active area.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional data samples (e.g., positions). NaNs are ignored.
    edges : Tuple[NDArray[np.float64], ...]
        A tuple where each element is a 1D array of bin edge positions for
        one dimension of the grid.
    close_gaps : bool, default=False
        If True, apply binary closing (dilation then erosion) to close small
        gaps in the active area.
    fill_holes : bool, default=False
        If True, apply binary hole filling to fill holes enclosed by active bins.
    dilate : bool, default=False
        If True, apply binary dilation to expand the boundary of the active area.
    bin_count_threshold : int, default=0
        Minimum number of samples a bin must contain to be initially
        considered active (before morphological operations).
    boundary_exists : bool, default=False
        If True, explicitly mark the outermost layer of bins in each dimension
        as inactive *after* morphological operations. This can be used if
        `add_boundary_bins` was True during grid creation to ensure these
        added boundary bins are not part of the inferred active track.

    Returns
    -------
    active_mask : NDArray[np.bool_], shape (n_bins_dim0, n_bins_dim1, ...)
        N-dimensional boolean mask indicating which bins in the grid are
        considered active or part of the track interior.
    """
    pos_clean = data_samples[~np.any(np.isnan(data_samples), axis=1)]

    if pos_clean.shape[0] == 0:
        # Handle case with no valid data_sampless
        grid_shape = tuple(len(e) - 1 for e in edges)
        warnings.warn(
            "infer_active_bins is True, but no data_samples provided. "
            "Defaulting to all bins active.",
            UserWarning,
        )
        return np.zeros(grid_shape, dtype=bool)

    bin_counts, _ = np.histogramdd(pos_clean, bins=edges)
    active_mask = bin_counts > bin_count_threshold

    n_dims = data_samples.shape[1]
    if n_dims > 1:
        # Use connectivity=1 for 4-neighbor (2D) or 6-neighbor (3D) etc.
        structure = ndimage.generate_binary_structure(n_dims, connectivity=2)

        if close_gaps:
            # Closing operation first (dilation then erosion) to close small gaps
            active_mask = ndimage.binary_closing(active_mask, structure=structure)

        if fill_holes:
            # Fill larger holes enclosed by occupied bins
            active_mask = ndimage.binary_fill_holes(active_mask, structure=structure)

        if dilate:
            # Expand the occupied area
            active_mask = ndimage.binary_dilation(active_mask, structure=structure)

        if boundary_exists:
            if active_mask.ndim == 1:
                if len(active_mask) > 0:
                    active_mask[-1] = False
            elif active_mask.ndim > 1 and active_mask.size > 0:
                for axis_n in range(active_mask.ndim):
                    slicer_first = [slice(None)] * active_mask.ndim
                    slicer_first[axis_n] = 0
                    active_mask[tuple(slicer_first)] = False
                    slicer_last = [slice(None)] * active_mask.ndim
                    slicer_last[axis_n] = -1
                    active_mask[tuple(slicer_last)] = False

    return active_mask.astype(bool)


def _create_regular_grid(
    data_samples: Optional[NDArray[np.float64]] = None,
    bin_size: Union[float, Sequence[float]] = 2.0,
    dimension_range: Optional[Sequence[Tuple[float, float]]] = None,
    add_boundary_bins: bool = False,
) -> Tuple[
    Tuple[NDArray[np.float64], ...],  # edges_tuple
    NDArray[np.float64],  # bin_centers
    Tuple[int, ...],  # centers_shape
]:
    """
    Calculate bin edges and centers for a regular N-dimensional spatial grid.

    The grid can be defined based on the extent of `data_samples` or an
    explicit `dimension_range`. Boundary bins can optionally be added.

    Parameters
    ----------
    data_samples : Optional[NDArray[np.float64]], shape (n_samples, n_dims), optional
        N-dimensional data samples. Used to determine grid extent if
        `dimension_range` is None. NaNs are ignored. Required if
        `dimension_range` is None. Defaults to None.
    bin_size : Union[float, Sequence[float]], default=2.0
        Desired approximate size of bins in each dimension. If a float,
        applied to all dimensions. If a sequence, must match `n_dims`.
    dimension_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit grid boundaries `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`.
        If None (default), boundaries are derived from `data_samples`.
    add_boundary_bins : bool, default=False
        If True, add one bin on each side of the grid in each dimension,
        effectively extending the `dimension_range` covered by the grid edges.

    Returns
    -------
    edges_tuple : Tuple[NDArray[np.float64], ...]
        Tuple where each element is a 1D array of bin edge positions for one
        dimension (shape `(n_bins_in_dim + 1,)`). Includes boundary bins if
        `add_boundary_bins` is True.
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Center coordinates of *every* bin in the (potentially boundary-extended)
        flattened grid, ordered row-major.
    grid_shape : Tuple[int, ...]
        N-D shape (number of bins in each dimension) of the created grid,
        including boundary bins if added.

    Raises
    ------
    ValueError
        If both `data_samples` and `dimension_range` are None.
        If `data_samples` (if used) are all NaN or empty.
        If `bin_size` sequence length doesn't match `n_dims`.
        If `dimension_range` (if used) sequence length doesn't match `n_dims`.
        If any `bin_size` component is not positive.
    """
    if data_samples is None and dimension_range is None:
        raise ValueError("Either `data_samples` or `dimension_range` must be provided.")
    if data_samples is not None:
        pos_nd = np.atleast_2d(data_samples)
        n_dims = pos_nd.shape[1]
        pos_clean = pos_nd[~np.any(np.isnan(pos_nd), axis=1)]
        if pos_clean.shape[0] == 0 and dimension_range is None:
            raise ValueError(
                "data_samples data contains only NaNs and no dimension_range provided."
            )
    elif dimension_range is not None:
        n_dims = len(dimension_range)
        pos_clean = None  # No data_samples data needed if range is given
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
    hist_range = dimension_range
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

    # Validate dimension_range dimensions if provided
    if dimension_range is not None and len(dimension_range) != n_dims:
        raise ValueError(
            f"`dimension_range` length ({len(dimension_range)}) must match "
            f"number of dimensions ({n_dims})."
        )

    # Calculate number of bins for the core range
    n_bins_core = get_n_bins(pos_clean, bin_sizes, hist_range)  # Pass array bin_sizes

    # Calculate core edges using histogramdd (even if data_samples is None, to get uniform bins)
    # Need dummy data if no data_samples provided
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
    bin_centers = np.stack([center.ravel() for center in mesh_centers], axis=1)

    edges_tuple: Tuple[NDArray[np.float64], ...] = tuple(final_edges_list)

    return edges_tuple, bin_centers, centers_shape


def _points_to_regular_grid_bin_ind(
    points: NDArray[np.float64],
    grid_edges: Tuple[NDArray[np.float64], ...],
    grid_shape: Tuple[int, ...],
    active_mask: NDArray[np.bool_] = None,
) -> NDArray[np.int_]:
    """
    Map N-D points to their corresponding bin indices in a regular grid.

    If `active_mask` is provided, maps to indices relative to active bins
    (0 to `n_active_bins - 1`). Otherwise, maps to flat indices of the
    full conceptual grid.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, n_dims)
        N-dimensional points to map. NaNs are filtered out.
    grid_edges : Tuple[NDArray[np.float64], ...]
        Tuple where each element is a 1D array of bin edge positions for one
        dimension of the full grid.
    grid_shape : Tuple[int, ...]
        N-D shape (number of bins in each dimension) of the full grid.
    active_mask : Optional[NDArray[np.bool_]], shape `grid_shape`, optional
        If provided, an N-D boolean mask indicating active bins in the full
        grid. Output indices will be relative to these active bins.
        If None (default), output indices are flat indices of the full grid.

    Returns
    -------
    bin_indices : NDArray[np.int_], shape (n_valid_points,)
        Integer indices of the bins corresponding to each valid input point.
        - If `active_mask` is provided: Indices are `0` to `n_active_bins - 1`.
          Points outside active bins or grid boundaries get -1.
        - If `active_mask` is None: Flat indices of the full grid. Points
          outside grid boundaries get -1 or out-of-range indices (depending
          on `np.digitize` behavior for edge cases, typically -1 due to clipping).
    """
    points_atleast_2d = np.atleast_2d(points)
    valid_input_mask = ~np.any(np.isnan(points_atleast_2d), axis=1)

    # Initialize output assuming all points are invalid or unmapped
    # Output shape should match original number of points (before NaN filter)
    output_indices = np.full(points_atleast_2d.shape[0], -1, dtype=np.int_)

    if not np.any(valid_input_mask):  # No valid points after NaN filter
        return output_indices

    valid_points = points_atleast_2d[valid_input_mask]
    if valid_points.shape[0] == 0:  # Should be caught by above, but defensive
        return output_indices

    n_dims = valid_points.shape[1]
    if n_dims != len(grid_edges) or n_dims != len(grid_shape):
        # This case should ideally be caught earlier or raise a more specific error.
        # For now, assume dimensions match if we reach here.
        # If not, returning all -1s is a safe fallback.
        warnings.warn(
            "Dimensionality mismatch between points, grid_edges, or grid_shape.",
            RuntimeWarning,
        )
        return output_indices

    # Calculate N-D indices for valid_points
    multi_bin_idx_list = []
    point_is_within_grid_bounds = np.ones(valid_points.shape[0], dtype=bool)

    for i in range(n_dims):
        # np.digitize returns indices from 1 to len(bins)+1
        # We subtract 1 to get 0-based bin indices
        dim_indices = np.digitize(valid_points[:, i], grid_edges[i]) - 1

        # Check if indices are within the valid range [0, grid_shape[i]-1]
        point_is_within_grid_bounds &= (dim_indices >= 0) & (
            dim_indices < grid_shape[i]
        )
        multi_bin_idx_list.append(dim_indices)

    # Initialize flat indices for valid_points to -1
    original_bin_flat_idx_for_valid_points = np.full(
        valid_points.shape[0], -1, dtype=np.int_
    )

    if np.any(point_is_within_grid_bounds):
        # Filter to only points that are fully within grid bounds for ravel_multi_index
        coords_for_ravel = tuple(
            idx[point_is_within_grid_bounds] for idx in multi_bin_idx_list
        )

        # np.ravel_multi_index requires all coords to be within dimension bounds
        original_bin_flat_idx_for_valid_points[point_is_within_grid_bounds] = (
            np.ravel_multi_index(coords_for_ravel, grid_shape)
        )

    # Place these calculated flat indices (or -1 for out-of-bounds) back into the full output array
    # This mapping depends on whether an active_mask is used for final indexing.

    final_mapped_indices_for_valid_points = np.full(
        valid_points.shape[0], -1, dtype=np.int_
    )

    if active_mask is not None:
        # Create mapping from original_full_grid_flat_index to active_bin_id (0 to N-1)
        # This should only be created once if possible, e.g., stored on the layout object
        active_original_flat_indices = np.flatnonzero(active_mask)
        original_flat_to_active_id_map: Dict[int, int] = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(active_original_flat_indices)
        }

        for i, orig_flat_idx in enumerate(original_bin_flat_idx_for_valid_points):
            if orig_flat_idx != -1:  # If it was a valid original flat index
                # Check if this original flat index corresponds to an active bin
                if active_mask.ravel()[orig_flat_idx]:
                    final_mapped_indices_for_valid_points[i] = (
                        original_flat_to_active_id_map[orig_flat_idx]
                    )
                # else it remains -1 (was in grid, but not in active_mask)
            # else it remains -1 (was out of grid bounds)
    else:
        # No active_mask, so original_bin_flat_idx_for_valid_points are the final indices
        # (where -1 means out of bounds)
        final_mapped_indices_for_valid_points = original_bin_flat_idx_for_valid_points

    output_indices[valid_input_mask] = final_mapped_indices_for_valid_points
    return output_indices
