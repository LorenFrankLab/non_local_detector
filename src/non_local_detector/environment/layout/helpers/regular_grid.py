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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from non_local_detector.environment.layout.helpers.utils import get_centers, get_n_bins


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
    diagonally) in the N-D grid. Edges store 'distance', 'vector', and 'angle_2d'
    attributes.

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
                    edge_attrs: Dict[str, Any] = {
                        "distance": distance,
                        "vector": tuple(displacement_vector.tolist()),
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
        if active_mask.ndim == 1 or (
            active_mask.ndim == 2 and active_mask.shape[1] == 1
        ):
            if active_mask.size > 0:
                active_mask[0] = False
            if active_mask.size > 1:
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
    Define bin edges and centers for a regular N-D Cartesian grid.

    Parameters
    ----------
    data_samples : ndarray of shape (n_samples, n_dims), optional
        Used to infer dimension ranges if `dimension_range` is None.
        NaNs are ignored. If None, `dimension_range` must be provided.
    bin_size : float or sequence of floats, default=2.0
        If float, same bin size along every dimension. If sequence, must match `n_dims`.
    dimension_range : sequence of (min, max) tuples, length `n_dims`, optional
        Explicit bounding box for each dimension, used if `data_samples` is None.
    add_boundary_bins : bool, default=False
        If True, extends each axis by one extra bin on both ends.

    Returns
    -------
    edges_tuple : tuple of ndarrays
        Each element is a 1D array of bin-edge coordinates for that dimension,
        length = n_bins_dim + 1.
    bin_centers : ndarray, shape (∏(n_bins_dim), n_dims)
        Cartesian product of centers of each bin (flattened).
    centers_shape : tuple of ints
        Number of bins along each dimension, e.g. (n_x, n_y, n_z).

    Raises
    ------
    ValueError
        - If both `data_samples` and `dimension_range` are None.
        - If `data_samples` is provided but not a 2D array.
        - If `dimension_range` length ≠ inferred `n_dims`.
        - If `bin_size` sequence length ≠ `n_dims`, or any `bin_size` ≤ 0.
    """
    # 1) Determine dimensionality
    if data_samples is None and dimension_range is None:
        raise ValueError("Either `data_samples` or `dimension_range` must be provided.")
    if data_samples is not None:
        samples = np.asarray(data_samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError(f"`data_samples` must be 2D, got shape {samples.shape}.")
        n_dims = samples.shape[1]
        # Remove NaNs
        samples = samples[~np.isnan(samples).any(axis=1)]
        if samples.size == 0 and dimension_range is None:
            raise ValueError(
                "`data_samples` has no valid points and no `dimension_range` given."
            )
    else:
        samples = None
        n_dims = len(dimension_range)

    # 2) Normalize & validate bin_size
    if isinstance(bin_size, (float, int)):
        bin_sizes = np.full(n_dims, float(bin_size))
    else:
        bin_sizes = np.asarray(bin_size, dtype=float)
        if bin_sizes.ndim != 1 or bin_sizes.shape[0] != n_dims:
            raise ValueError(f"`bin_size` length must be {n_dims}, got {bin_sizes}.")
    if np.any(bin_sizes <= 0.0):
        raise ValueError("All elements of `bin_size` must be positive.")

    # 3) Determine dimension ranges
    if dimension_range is not None:
        if len(dimension_range) != n_dims:
            raise ValueError(
                f"`dimension_range` length ({len(dimension_range)}) must match n_dims ({n_dims})."
            )
        ranges = []
        for (lo, hi), size in zip(dimension_range, bin_sizes):
            lo_f, hi_f = float(min(lo, hi)), float(max(lo, hi))
            # If user gave a zero-span range (lo == hi), expand by 0.5 * bin_size
            if np.isclose(lo_f, hi_f):
                lo_f -= 0.5 * size
                hi_f += 0.5 * size
            ranges.append((lo_f, hi_f))
    else:
        # Infer from `samples`
        ranges = []
        for dim in range(n_dims):
            dim_vals = samples[:, dim]
            lo_f, hi_f = float(np.nanmin(dim_vals)), float(np.nanmax(dim_vals))
            if np.isclose(lo_f, hi_f):
                # If all data is constant, expand by half a bin
                lo_f -= 0.5 * bin_sizes[dim]
                hi_f += 0.5 * bin_sizes[dim]
            ranges.append((lo_f, hi_f))

    # 4) Compute number of bins in each dimension
    data_for_bins = samples if samples is not None else np.zeros((1, n_dims))
    n_bins = get_n_bins(data_for_bins, bin_sizes, ranges)  # ensures at least 1

    # 5) Generate core edges via np.histogramdd
    #    Use a dummy point at the center if `samples` is None
    if samples is None:
        dummy = np.array([[(lo + hi) / 2.0 for lo, hi in ranges]])
        _, core_edges = np.histogramdd(dummy, bins=n_bins, range=ranges)
    else:
        _, core_edges = np.histogramdd(samples, bins=n_bins, range=ranges)

    # 6) Optionally add boundary bins by extending each edge array
    final_edges = []
    for edges_dim, size in zip(core_edges, bin_sizes):
        diff = np.diff(edges_dim)
        if diff.size == 0:
            step = size
        else:
            step = diff[0]
        if add_boundary_bins:
            extended = np.concatenate(
                ([edges_dim[0] - step], edges_dim, [edges_dim[-1] + step])
            )
        else:
            extended = edges_dim
        final_edges.append(extended.astype(float))

    edges_tuple = tuple(final_edges)

    # 7) Compute centers for each dimension
    centers_per_dim = [get_centers(e) for e in final_edges]
    centers_shape = tuple(len(c) for c in centers_per_dim)

    # 8) Build the full Cartesian product of centers
    mesh = np.meshgrid(*centers_per_dim, indexing="ij")
    bin_centers = np.stack([m.ravel() for m in mesh], axis=-1)

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
