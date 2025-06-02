"""
Utility functions for creating and managing hexagonal grid layouts.

This module provides helper functions specifically for environments that use
hexagonal tiling. These functions cover:
- Generation of hexagonal grid coordinates (`_create_hex_grid`).
- Conversion between Cartesian and hexagonal (cube/axial) coordinate systems
  (`_cartesian_to_fractional_cube`, `_round_fractional_cube_to_integer_axial`,
  `_axial_to_offset_bin_indices`).
- Mapping continuous 2D points to discrete hexagonal bin indices
  (`_points_to_hex_bin_ind`).
- Inferring active hexagonal bins based on data sample occupancy
  (`_infer_active_bins_from_hex_grid`).
- Determining neighbor relationships in the hexagonal grid
  (`_get_hex_grid_neighbor_deltas`).
- Constructing a connectivity graph for active hexagonal bins
  (`_create_hex_connectivity_graph`).

These utilities are primarily used by the `HexagonalLayout` engine.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def _create_hex_grid(
    data_samples: Optional[NDArray[np.float64]],
    dimension_range: Optional[Sequence[Tuple[float, float]]] = None,
    hexagon_width: float = 1.0,
) -> Tuple[
    NDArray[np.float64],  # bin_centers
    Tuple[int, int],  # centers_shape
    float,  # hex_radius
    float,  # hex_orientation
    float,  # min_x
    float,  # min_y
    Sequence[Tuple[float, float]],  # effective_dimension_range
]:
    """
    Generate a 2D hexagonal grid (pointy-top) that covers either:
    - A user-specified bounding box (`dimension_range`), or
    - The min/max extent of `data_samples`.

    Parameters
    ----------
    data_samples : ndarray of shape (n_samples, 2), optional
        2D points to infer bounding box if `dimension_range` is None.
        NaNs are ignored. If None, `dimension_range` must be provided.
    dimension_range : sequence of ((min_x, max_x), (min_y, max_y)), optional
        If provided, must be length 2. Otherwise inferred from `data_samples`.
    hexagon_width : float, default=1.0
        Distance between parallel sides of each hexagon; must be positive.

    Returns
    -------
    bin_centers : ndarray, shape (n_hex_rows * n_hex_cols, 2)
        (x, y) coordinates of every hexagon center in the full grid.
    centers_shape : (n_hex_rows, n_hex_cols)
        The 2D shape of the conceptual hex-grid, before flattening.
    hex_radius : float
        Distance from center to any vertex (hexagon side length).
    hex_orientation : float
        Orientation in radians (0.0 = point-up).
    min_x : float
        Minimum x coordinate of the bounding box (grid origin).
    min_y : float
        Minimum y coordinate of the bounding box (grid origin).
    effective_dimension_range : [ (min_x, max_x), (min_y, max_y) ]
        The actual range used (either user‐provided or inferred).

    Raises
    ------
    ValueError
        - If `hexagon_width <= 0`.
        - If neither `dimension_range` nor `data_samples` (with valid shape) is provided.
        - If `dimension_range` is provided but not length 2 or has min > max.
        - If `data_samples` is not a 2D array of shape (n_samples, 2).
    """
    # 1) Validate hexagon_width
    if hexagon_width <= 0:
        raise ValueError("`hexagon_width` must be a positive float.")

    hex_orientation = 0.0  # point-up

    # 2) Clean & validate data_samples
    if data_samples is not None:
        ds_arr = np.asarray(data_samples, dtype=float)
        if ds_arr.ndim != 2 or ds_arr.shape[1] != 2:
            raise ValueError("`data_samples` must be shape (n_samples, 2).")
        # Remove any rows containing NaNs
        ds_arr = ds_arr[~np.isnan(ds_arr).any(axis=1)]
    else:
        ds_arr = np.empty((0, 2), dtype=float)

    # 3) Determine bounding box
    if dimension_range is not None:
        if len(dimension_range) != 2:
            raise ValueError("`dimension_range` must be ((min_x,max_x),(min_y,max_y)).")
        (ux0, ux1), (uy0, uy1) = dimension_range
        min_x, max_x = sorted((float(ux0), float(ux1)))
        min_y, max_y = sorted((float(uy0), float(uy1)))
        effective_range = [(min_x, max_x), (min_y, max_y)]
    else:
        if ds_arr.shape[0] == 0:
            raise ValueError("`data_samples` is empty; cannot infer bounding box.")
        min_vals = np.min(ds_arr, axis=0)
        max_vals = np.max(ds_arr, axis=0)
        min_x, max_x = float(min_vals[0]), float(max_vals[0])
        min_y, max_y = float(min_vals[1]), float(max_vals[1])
        effective_range = [(min_x, max_x), (min_y, max_y)]

    # 4) Compute hex geometry constants
    vertical_step = (np.sqrt(3.0) / 2.0) * hexagon_width
    hex_radius = hexagon_width / np.sqrt(3)

    # 5) Compute how many hexes fit in x/y direction (ensure at least one)
    span_x = max_x - min_x
    span_y = max_y - min_y
    n_hex_cols = max(1, int(np.ceil(span_x / hexagon_width)) + 1)
    n_hex_rows = max(1, int(np.ceil(span_y / vertical_step)) + 1)

    # 6) Create a regular grid of (row, col) indices
    col_idx, row_idx = np.meshgrid(
        np.arange(n_hex_cols), np.arange(n_hex_rows), indexing="xy"
    )
    col_idx = col_idx.astype(float)
    row_idx = row_idx.astype(float)

    # 7) Shift odd rows horizontally by half‐cell
    col_idx[1::2, :] += 0.5

    # 8) Scale to physical coordinates, then translate by (min_x, min_y)
    x_coords = col_idx * hexagon_width + min_x
    y_coords = row_idx * vertical_step + min_y

    bin_centers = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    centers_shape = (n_hex_rows, n_hex_cols)

    return (
        bin_centers,
        centers_shape,
        hex_radius,
        hex_orientation,
        min_x,
        min_y,
        effective_range,
    )


def _cartesian_to_fractional_cube(
    points_x: NDArray[np.float64],
    points_y: NDArray[np.float64],
    hex_radius: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert Cartesian (x, y) coordinates to fractional axial coordinates of a hex grid.

    Axial coordinates represent positions in a hexagonal grid using two axes (q, r),
    simplifying neighbor and distance calculations compared to Cartesian coordinates.

    Parameters
    ----------
    points_x : NDArray[np.float64], shape (n_points,)
        X-coordinate(s) in Cartesian space, typically relative to the hexagonal
        grid's origin.
    points_y : NDArray[np.float64], shape (n_points,)
        Y-coordinate(s) in Cartesian space, relative to the grid's origin.
    hex_radius : float
        Width of each hexagon in the grid, measured as the distance between opposite sides.

    Returns
    -------
    q_frac : NDArray[np.float64], shape (n_points,)
        Fractional q cube coordinate.
    r_frac : NDArray[np.float64], shape (n_points,)
        Fractional r cube coordinate.
    s_frac : NDArray[np.float64], shape (n_points,)
        Fractional s cube coordinate, calculated as -q_frac - r_frac.

    Notes
    -----
    The conversion follows the standard formulas (see
    https://www.redblobgames.com/grids/hexagons/#coordinates):

        q = (sqrt(3)/3 * x - 1/3 * y) / hex_width
        r = (2/3 * y) / hex_width

    Fractional axial coordinates allow for smooth interpolation before rounding
    to discrete hex indices.

    Points outside the grid bounds or containing NaNs in inputs will produce NaN outputs.

    """
    if hex_radius == 0:  # Avoid division by zero
        # If hex_radius is zero, implies a degenerate grid.
        # Return zeros or handle as an error appropriately.
        # For now, returning zeros, assuming points effectively map to a single point.
        zero_coords = np.zeros_like(points_x)
        return zero_coords, zero_coords, zero_coords

    q_frac = (np.sqrt(3.0) / 3.0 * points_x - 1.0 / 3.0 * points_y) / hex_radius
    r_frac = (2.0 / 3.0 * points_y) / hex_radius
    s_frac = -q_frac - r_frac
    return q_frac, r_frac, s_frac


def _round_fractional_cube_to_integer_axial(
    q_frac: NDArray[np.float64],
    r_frac: NDArray[np.float64],
    s_frac: NDArray[np.float64],
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Round fractional cube coordinates to integer axial coordinates (q, r).

    This rounding method ensures that the sum of the corresponding integer
    cube coordinates (q_int + r_int + s_int) would be zero by adjusting
    the coordinate with the largest fractional difference from its rounded integer.

    Parameters
    ----------
    q_frac : NDArray[np.float64], shape (n_points,)
        Fractional q cube coordinates.
    r_frac : NDArray[np.float64], shape (n_points,)
        Fractional r cube coordinates.
    s_frac : NDArray[np.float64], shape (n_points,)
        Fractional s cube coordinates (where s_frac = -q_frac - r_frac).

    Returns
    -------
    q_axial : NDArray[np.int_], shape (n_points,)
        Integer q axial coordinate.
    r_axial : NDArray[np.int_], shape (n_points,)
        Integer r axial coordinate.
    """
    # Initial rounding to nearest integer
    q_round_f: NDArray[np.float64] = np.round(q_frac)
    r_round_f: NDArray[np.float64] = np.round(r_frac)
    s_round_f: NDArray[np.float64] = np.round(
        s_frac
    )  # s_round = np.round(-q_frac - r_frac)

    # Differences from original fractional values
    q_diff: NDArray[np.float64] = np.abs(q_round_f - q_frac)
    r_diff: NDArray[np.float64] = np.abs(r_round_f - r_frac)
    s_diff: NDArray[np.float64] = np.abs(s_round_f - s_frac)

    # Initialize axial coordinates with the basic rounded values
    q_axial: NDArray[np.int_] = q_round_f.astype(int)
    r_axial: NDArray[np.int_] = r_round_f.astype(int)
    # s_axial for sum check, not directly returned for offset conversion
    # s_axial: NDArray[np.int_] = s_round_f.astype(np.int_)

    # Correct coordinates to ensure q_axial + r_axial + s_axial = 0
    # Condition where q_diff is strictly the largest
    cond1: NDArray[np.bool_] = (q_diff > r_diff) & (q_diff > s_diff)
    q_axial[cond1] = (-r_round_f[cond1] - s_round_f[cond1]).astype(int)
    # r_axial[cond1] remains r_round_f[cond1]

    # Condition where r_diff is strictly the largest (and q_diff was not)
    cond2: NDArray[np.bool_] = (~cond1) & (r_diff > s_diff)
    r_axial[cond2] = (-q_round_f[cond2] - s_round_f[cond2]).astype(int)
    # q_axial[cond2] remains q_round_f[cond2]

    # Condition where s_diff is largest (or ties put it here), and others were not
    # In this case, s_axial would be adjusted:
    # s_axial[cond3] = (-q_round_f[cond3] - r_round_f[cond3]).astype(np.int_)
    # q_axial and r_axial remain as their initially rounded values, which is correct.
    # No explicit action needed for q_axial, r_axial under cond3 as they are already set.

    return q_axial, r_axial


def _axial_to_offset_bin_indices(
    q_axial: NDArray[np.int_],
    r_axial: NDArray[np.int_],
    n_hex_x: int,
    n_hex_y: int,
) -> NDArray[np.int_]:
    """
    Convert integer axial coordinates (q, r) to 1D bin indices.

    This conversion is for an "odd-r" offset grid arrangement where odd rows
    are shifted relative to even rows. The resulting 1D index corresponds
    to a flattened row-major order of the conceptual rectangular grid of hexagons.

    Parameters
    ----------
    q_axial : NDArray[np.int_], shape (n_points,)
        Integer q axial coordinates.
    r_axial : NDArray[np.int_], shape (n_points,)
        Integer r axial coordinates.
    n_hex_x : int
        Number of hexagons in the x-direction (columns) of the full conceptual grid.
    n_hex_y : int
        Number of hexagons in the y-direction (rows) of the full conceptual grid.

    Returns
    -------
    NDArray[np.int_], shape (n_points,)
        1D bin indices corresponding to the input axial coordinates.
        Points that fall outside the defined grid dimensions (`n_hex_x`, `n_hex_y`)
        are assigned an index of -1.
    """
    # Convert axial to "odd-r" offset coordinates
    # col = q + (r - (r & 1)) / 2
    # row = r
    # (r_axial & 1) is 1 if r_axial is odd, 0 if even. Works for negative r_axial.
    grid_col: NDArray[np.int_] = q_axial + (r_axial - (r_axial & 1)) // 2
    grid_row: NDArray[np.int_] = r_axial

    # Calculate 1D bin index (row-major flattening)
    # bin_idx = row * num_cols + col
    bin_idx: NDArray[np.int_] = grid_row * n_hex_x + grid_col

    # Mark out-of-bounds indices as -1
    valid_mask: NDArray[np.bool_] = (
        (grid_col >= 0) & (grid_col < n_hex_x) & (grid_row >= 0) & (grid_row < n_hex_y)
    )

    final_bin_idx: NDArray[np.int_] = np.full_like(bin_idx, -1, dtype=np.int_)
    final_bin_idx[valid_mask] = bin_idx[valid_mask]

    return final_bin_idx


def _points_to_hex_bin_ind(
    points: NDArray[np.float64],
    grid_offset_x: float,
    grid_offset_y: float,
    hex_radius: float,
    centers_shape: Tuple[int, int],
) -> NDArray[np.int_]:
    """
    Assign 2D Cartesian data points to hexagon bin indices in a predefined grid.

    This function converts Cartesian points to hexagonal grid coordinates and
    then to 1D flat indices relative to the full conceptual grid of hexagons.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, 2)
        The (x, y) coordinates of the points to assign to bins.
        NaN input points result in an index of -1.
    grid_offset_x : float
        The x-coordinate of the grid's effective origin (e.g., min_x from
        `_create_hex_grid`).
    grid_offset_y : float
        The y-coordinate of the grid's effective origin (e.g., min_y from
        `_create_hex_grid`).
    hex_radius : float
        The radius of the hexagons (distance from center to vertex).
    centers_shape : Tuple[int, int]
        The shape (n_hex_rows, n_hex_cols) of the full conceptual grid of hexagons.

    Returns
    -------
    NDArray[np.int_], shape (n_points,)
        The 1D bin index (relative to the full conceptual grid) for each
        data point. Points falling outside the defined grid area or NaN input
        points are assigned an index of -1.
    """
    n_points = points.shape[0]
    if n_points == 0:
        return np.array([], dtype=np.int_)

    points = np.atleast_2d(points)

    output_indices: NDArray[np.int_] = np.full(n_points, -1, dtype=np.int_)

    # Identify valid (non-NaN) points
    valid_mask: NDArray[np.bool_] = ~np.isnan(points).any(axis=1)
    if not np.any(valid_mask):
        return output_indices  # All points are NaN or empty after all

    valid_points: NDArray[np.float64] = points[valid_mask]

    # Adjust points relative to the grid's effective origin
    # (grid_offset_x, grid_offset_y) is the center of hex (0,0) in grid indices
    adj_points_x: NDArray[np.float64] = valid_points[:, 0] - grid_offset_x
    adj_points_y: NDArray[np.float64] = valid_points[:, 1] - grid_offset_y

    # Convert Cartesian to fractional cube coordinates
    q_frac, r_frac, s_frac = _cartesian_to_fractional_cube(
        adj_points_x, adj_points_y, hex_radius
    )

    # Round fractional cube coordinates to integer axial coordinates
    q_axial, r_axial = _round_fractional_cube_to_integer_axial(q_frac, r_frac, s_frac)

    # Convert axial coordinates to 1D bin indices
    n_hex_y, n_hex_x = centers_shape
    bin_indices_for_valid_points = _axial_to_offset_bin_indices(
        q_axial, r_axial, n_hex_x, n_hex_y
    )

    output_indices[valid_mask] = bin_indices_for_valid_points

    return output_indices


def _infer_active_bins_from_hex_grid(
    data_samples: NDArray[np.float64],
    centers_shape: Tuple[int, int],
    hex_radius: float,
    min_x: float,
    min_y: float,
    bin_count_threshold: int = 0,
) -> NDArray[np.int_]:
    """
    Infer active bins in a hexagonal grid based on data sample occupancy.

    Maps `data_samples` to their respective hexagon bins within the full
    conceptual grid. Hexagons are marked active if their occupancy count
    exceeds `bin_count_threshold`.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_samples, 2)
        2D data samples (e.g., positions).
    centers_shape : Tuple[int, int]
        Shape (n_hex_rows, n_hex_cols) of the full conceptual grid of hexagons.
    hex_radius : float
        Radius of the hexagons.
    min_x : float
        Minimum x-coordinate of the grid's effective origin.
    min_y : float
        Minimum y-coordinate of the grid's effective origin.
    bin_count_threshold : int, optional, default=0
        Minimum number of samples a hexagon must contain to be considered active.

    Returns
    -------
    NDArray[np.int_], shape (n_active_hex_bins,)
        An array of original flat indices (relative to the full conceptual grid)
        of the hexagons that were deemed active. Returns an empty array if no
        bins are found to be active.
    """
    bin_ind = _points_to_hex_bin_ind(
        points=data_samples,
        grid_offset_x=min_x,
        grid_offset_y=min_y,
        hex_radius=hex_radius,
        centers_shape=centers_shape,
    )
    # Filter out invalid bin indices
    bin_ind = bin_ind[bin_ind >= 0]

    if len(bin_ind) == 0:
        # No valid bins found, return an empty array
        return np.array([], dtype=int)

    # Count occurrences of each unique bin ind
    bin_ind, bin_count = np.unique(bin_ind, return_counts=True)

    # Filter bins based on the count threshold
    bin_ind = bin_ind[bin_count >= bin_count_threshold]

    return bin_ind.astype(int)


def _get_hex_grid_neighbor_deltas(is_odd_row: bool) -> List[Tuple[int, int]]:
    """
    Return (delta_col, delta_row) coordinate deltas for hexagon neighbors.

    This is for an "odd-r" pointy-top hex grid, where odd rows (1, 3, ...)
    are typically shifted right relative to even rows (0, 2, ...).
    The y-axis (row index) increases "upwards" in grid coordinates.

    Parameters
    ----------
    is_odd_row : bool
        True if the current hexagon's row index is odd, False otherwise.

    Returns
    -------
    List[Tuple[int, int]]
        A list of 6 (delta_col, delta_row) tuples, each representing the
        offset to one of the 6 neighbors.
    """
    if is_odd_row:
        # Neighbors for an odd row (shifted right): E, W, NW, NE, SW, SE
        # (dc, dr)
        return [(1, 0), (-1, 0), (0, 1), (1, 1), (0, -1), (1, -1)]
    else:  # Even row
        # Neighbors for an even row: E, W, NW, NE, SW, SE
        # (dc, dr)
        return [(1, 0), (-1, 0), (-1, 1), (0, 1), (-1, -1), (0, -1)]


def _create_hex_connectivity_graph(
    active_original_flat_indices: NDArray[np.int_],
    full_grid_bin_centers: NDArray[np.float64],
    centers_shape: Tuple[int, int],
):
    """
    Create a connectivity graph for active bins in a hexagonal grid.

    Nodes in the returned graph are indexed from `0` to `n_active_bins - 1`.
    Each node corresponds to an active hexagon and stores attributes:
    - 'pos': (x, y) coordinates of the active hexagon's center.
    - 'source_grid_flat_index': Original flat index in the full conceptual hex grid.
    - 'original_grid_nd_index': (row, col) index in the full conceptual hex grid.

    Edges connect active hexagons that are immediate neighbors in the lattice.
    Edges store 'distance', 'vector', and 'angle_2d'.

    Parameters
    ----------
    active_original_flat_indices : NDArray[np.int_], shape (n_active_bins,)
        Array of original flat indices (relative to the full conceptual grid)
        for hexagons that are active.
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_hex_bins, 2)
        Center coordinates of *all* potential hexagons in the full grid.
    centers_shape : Tuple[int, int]
        Shape (n_hex_rows, n_hex_cols) of the full conceptual grid of hexagons.

    Returns
    -------
    nx.Graph
        Connectivity graph of active hexagonal bins. Nodes are re-indexed
        `0` to `n_active_bins - 1`.
    """
    connectivity_graph = nx.Graph()

    # 1. Identify active bins and create mapping from original flat index to new node ID
    n_active_bins = len(active_original_flat_indices)

    if n_active_bins == 0:
        return connectivity_graph  # Return an empty graph

    # Map: original_full_grid_flat_index -> new_active_bin_node_id (0 to n_active_bins-1)
    original_flat_to_new_node_id_map: Dict[int, int] = {
        original_idx: new_idx
        for new_idx, original_idx in enumerate(active_original_flat_indices)
    }

    # 2. Add nodes to the graph with new IDs (0 to n_active_bins-1) and attributes
    n_hex_y, n_hex_x = centers_shape
    for node_id, original_flat_idx in enumerate(active_original_flat_indices):
        row_idx = original_flat_idx // n_hex_x
        col_idx = original_flat_idx % n_hex_x
        original_nd_idx = (row_idx, col_idx)
        pos_coordinates = tuple(full_grid_bin_centers[original_flat_idx])

        connectivity_graph.add_node(
            node_id,
            pos=pos_coordinates,
            source_grid_flat_index=int(original_flat_idx),
            original_grid_nd_index=original_nd_idx,
        )

    # 3. Add edges between these new active node IDs
    # Iterate through each active bin using its *original* N-D index
    edges_to_add_with_attrs = []
    for node_id, original_flat_idx in enumerate(active_original_flat_indices):
        row_idx = original_flat_idx // n_hex_x
        col_idx = original_flat_idx % n_hex_x

        # Determine if the current row is odd or even
        is_odd_row = (row_idx % 2) == 1

        # Get the neighbor deltas based on the row parity
        neighbor_deltas = _get_hex_grid_neighbor_deltas(is_odd_row)

        # Add edges to neighbors
        for delta_col, delta_row in neighbor_deltas:
            neighbor_row = row_idx + delta_row
            neighbor_col = col_idx + delta_col

            # Check if the neighbor is within bounds
            if not (0 <= neighbor_row < n_hex_y and 0 <= neighbor_col < n_hex_x):
                continue

            # Calculate the original flat index of the neighbor
            neighbor_flat_index = neighbor_row * n_hex_x + neighbor_col

            # Check if the neighbor is an active bin
            if neighbor_flat_index in original_flat_to_new_node_id_map:
                # Get the new node ID for the neighbor
                neighbor_node_id = original_flat_to_new_node_id_map[neighbor_flat_index]

                # Add an edge between the current node and its neighbor
                if node_id < neighbor_node_id:
                    # Avoid duplicate edges in undirected graph
                    pos_u = np.asarray(connectivity_graph.nodes[node_id]["pos"])
                    pos_v = np.asarray(
                        connectivity_graph.nodes[neighbor_node_id]["pos"]
                    )
                    distance = float(np.linalg.norm(pos_u - pos_v))
                    displacement_vector = pos_v - pos_u
                    edge_attrs: Dict[str, Any] = {
                        "distance": distance,
                        "vector": tuple(displacement_vector.tolist()),
                        "angle_2d": math.atan2(
                            displacement_vector[1], displacement_vector[0]
                        ),
                    }
                    edges_to_add_with_attrs.append(
                        (node_id, neighbor_node_id, edge_attrs)
                    )

    # Add all edges with their attributes
    connectivity_graph.add_edges_from(edges_to_add_with_attrs)

    # Add edge IDs to the graph
    # This is a unique ID for each edge in the graph, starting from 0
    # and incrementing by 1 for each edge
    for edge_id_counter, (u, v) in enumerate(connectivity_graph.edges()):
        connectivity_graph.edges[u, v]["edge_id"] = edge_id_counter

    return connectivity_graph
