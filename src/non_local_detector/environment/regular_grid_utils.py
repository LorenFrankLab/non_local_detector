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


def _infer_active_elements_from_samples(
    candidate_element_centers: NDArray[np.float64],
    data_samples: NDArray[np.float64],
    bin_count_threshold: int = 0,
) -> Tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.int_]]:
    """
    Infers active elements from a set of candidate centers based on data sample occupancy.

    This function maps `data_samples` to the nearest `candidate_element_centers`
    and then determines which candidates are "active" by checking if their
    occupancy count exceeds `bin_count_threshold`.

    Parameters
    ----------
    candidate_element_centers : NDArray[np.float64], shape (n_candidates, n_dims)
        The N-dimensional coordinates of the centers of all potential elements (bins, cells, points).
    data_samples : NDArray[np.float64], shape (n_samples, n_dims)
        The N-dimensional data samples (e.g., recorded data_sampless) used to determine occupancy.
        NaNs within this array will be filtered out.
    bin_count_threshold : int, optional
        The minimum number of data samples that must map to a candidate element
        for it to be considered active. If 0, any occupancy makes it active.

    Returns
    -------
    inferred_1d_mask_on_candidates : NDArray[np.bool_]
        A 1D boolean mask with the same length as `candidate_element_centers`.
        `True` indicates that the corresponding candidate element is active.
    final_active_centers : NDArray[np.float64], shape (n_active_elements, n_dims)
        The subset of `candidate_element_centers` that were deemed active.
    source_indices_of_active_centers : NDArray[np.int_], (n_active_elements,)
        The original indices (from `candidate_element_centers`) of the elements
        that were deemed active.

    Raises
    ------
    ValueError
        If `candidate_element_centers` or `data_samples` have incompatible dimensions
        or if `bin_count_threshold` is negative.
    """
    if bin_count_threshold < 0:
        raise ValueError("bin_count_threshold must be non-negative.")

    n_candidates, n_dims_candidates = candidate_element_centers.shape
    _, n_dims_samples = data_samples.shape

    if n_candidates == 0:
        warnings.warn(
            "No candidate element centers provided for interior inference. "
            "Returning no active elements.",
            UserWarning,
        )
        return (
            np.array([], dtype=bool),
            np.empty((0, n_dims_candidates if n_dims_candidates > 0 else 0)),
            np.array([], dtype=np.int_),
        )

    if n_dims_candidates != n_dims_samples:
        raise ValueError(
            f"Dimensionality mismatch: candidate_element_centers have {n_dims_candidates} dims, "
            f"while data_samples have {n_dims_samples} dims."
        )

    # Filter out NaN data_samples
    valid_samples_mask = ~np.any(np.isnan(data_samples), axis=1)
    valid_data_samples = data_samples[valid_samples_mask]

    if valid_data_samples.shape[0] == 0:
        warnings.warn(
            "No valid (non-NaN) data samples provided for interior inference. "
            "No elements will be marked as active based on occupancy.",
            UserWarning,
        )
        # All candidates will be considered non-active by occupancy
        inferred_1d_mask_on_candidates = np.zeros(n_candidates, dtype=bool)
        final_active_centers = np.empty((0, n_dims_candidates))
        source_indices_of_active_centers = np.array([], dtype=np.int_)
        return (
            inferred_1d_mask_on_candidates,
            final_active_centers,
            source_indices_of_active_centers,
        )

    # Build KD-tree on candidate centers to map data samples
    try:
        candidate_kdtree = KDTree(candidate_element_centers)
    except (
        Exception
    ) as e:  # Catch potential QhullError for degenerate candidate_element_centers
        warnings.warn(
            f"KDTree construction failed on candidate centers: {e}. "
            "Cannot infer active elements from samples.",
            RuntimeWarning,
        )
        # Treat as if no elements become active by occupancy
        inferred_1d_mask_on_candidates = np.zeros(n_candidates, dtype=bool)
        final_active_centers = np.empty((0, n_dims_candidates))
        source_indices_of_active_centers = np.array([], dtype=np.int_)
        return (
            inferred_1d_mask_on_candidates,
            final_active_centers,
            source_indices_of_active_centers,
        )

    # Query the KD-tree: for each valid_data_sample, find the index of the nearest candidate_element_center
    try:
        _, assigned_candidate_indices = candidate_kdtree.query(valid_data_samples)
    except Exception as e:  # Catch errors if query points have wrong dimension etc.
        warnings.warn(
            f"KDTree query failed during active element inference: {e}. "
            "No elements will be marked as active based on occupancy.",
            RuntimeWarning,
        )
        inferred_1d_mask_on_candidates = np.zeros(n_candidates, dtype=bool)
        final_active_centers = np.empty((0, n_dims_candidates))
        source_indices_of_active_centers = np.array([], dtype=np.int_)
        return (
            inferred_1d_mask_on_candidates,
            final_active_centers,
            source_indices_of_active_centers,
        )

    # Calculate occupancy counts for each candidate element
    # assigned_candidate_indices contains indices from 0 to n_candidates-1
    occupancy_counts = np.bincount(assigned_candidate_indices, minlength=n_candidates)

    # Determine which candidates are active based on the threshold
    inferred_1d_mask_on_candidates = occupancy_counts > bin_count_threshold

    if not np.any(inferred_1d_mask_on_candidates):
        warnings.warn(
            "Inferring active elements resulted in no candidates meeting the "
            "bin_count_threshold. All elements will be considered non-active "
            "by this inference step.",
            UserWarning,
        )

    final_active_centers = candidate_element_centers[inferred_1d_mask_on_candidates]
    source_indices_of_active_centers = np.flatnonzero(
        inferred_1d_mask_on_candidates
    ).astype(np.int_)

    return (
        inferred_1d_mask_on_candidates,
        final_active_centers,
        source_indices_of_active_centers,
    )


def _infer_dimension_ranges_from_samples(
    data_samples: NDArray[np.float64],
    buffer_around_data: Union[float, Sequence[float]] = 0.0,
) -> Sequence[Tuple[float, float]]:
    """
    Infers the min/max range for each dimension from data samples.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_samples, n_dims)
        The data points from which to infer ranges. NaNs are ignored.
    buffer_around_data : Union[float, Sequence[float]], default 0.0
        A buffer to add to the min and max of the inferred range in each dimension.
        If a float, the same buffer is applied to all dimensions.
        If a sequence, it specifies the buffer for each dimension.
        This is useful if the data points are all identical or collinear,
        or if a margin around the data is desired.

    Returns
    -------
    Sequence[Tuple[float, float]]
        The inferred ranges as `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`.

    Raises
    ------
    ValueError
        If data_samples are all NaN, empty after NaN removal, or if
        dimensionality mismatch.
    """
    if data_samples.ndim != 2:
        raise ValueError(
            f"data_samples must be a 2D array, shape is {data_samples.shape}"
        )
    n_dimensions = data_samples.shape[1]

    clean_samples = data_samples[~np.any(np.isnan(data_samples), axis=1)]
    if clean_samples.shape[0] == 0:
        raise ValueError("All 'data_samples' are NaN or the array is empty.")

    min_vals = np.min(clean_samples, axis=0)
    max_vals = np.max(clean_samples, axis=0)

    if isinstance(buffer_around_data, (float, int)):
        buffer_values = np.array([float(buffer_around_data)] * n_dimensions)
    elif len(buffer_around_data) == n_dimensions:
        buffer_values = np.asarray(buffer_around_data, dtype=float)
    else:
        raise ValueError(
            f"buffer_around_data sequence length ({len(buffer_around_data)}) "
            f"must match number of dimensions ({n_dimensions})."
        )

    inferred_ranges = []
    for dim_idx in range(n_dimensions):
        d_min, d_max = min_vals[dim_idx], max_vals[dim_idx]
        buffer = buffer_values[dim_idx]

        # Ensure range has some extent if data is point-like or buffer is zero
        if np.isclose(d_min, d_max):
            if np.isclose(buffer, 0.0):  # If buffer is also zero, data is a point
                warnings.warn(
                    f"Dimension {dim_idx} has zero extent and no buffer specified. "
                    "Using a default small buffer of 1.0 around the point.",
                    UserWarning,
                )
                d_min -= 0.5  # Default small extent
                d_max += 0.5
            else:  # Buffer will create the extent
                d_min -= buffer
                d_max += buffer
        else:  # Data already has extent
            d_min -= buffer
            d_max += buffer

        inferred_ranges.append((d_min, d_max))

    return tuple(inferred_ranges)


def _create_regular_grid_connectivity_graph(
    full_grid_bin_centers: NDArray[np.float64],
    active_mask_nd: NDArray[np.bool_],
    grid_shape: Tuple[int, ...],
    connect_diagonal: bool = False,
) -> nx.Graph:
    """
    Creates a graph connecting centers of active/interior bins in an N-D grid.

    Nodes in the returned graph are indexed from 0 to n_active_bins - 1.
    Each node carries attributes:
    - 'pos': The N-D coordinates of the active bin center.
    - 'source_grid_flat_index': The original flat index of this bin in the full grid.
    - 'original_grid_nd_index': The original N-D tuple index in the full grid.
    - 'is_active': Always True.

    Edges connect active bins that are adjacent in the N-D grid.

    Parameters
    ----------
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Coordinates of the center of *all* bins in the original full grid,
        ordered by flattened grid index.
    active_mask_nd : NDArray[np.bool_], shape (dim0_size, dim1_size, ...)
        N-dimensional boolean mask indicating which bins in the full grid
        are active/interior. Must match `grid_shape`.
    grid_shape : Tuple[int, ...]
        The N-D shape of the original full grid.
    connect_diagonal : bool, default=False
        If True, connect diagonally adjacent active bins in addition to
        orthogonally adjacent ones.

    Returns
    -------
    connectivity_graph : nx.Graph
        Graph of active bins with re-indexed nodes (0 to n_active_bins - 1).

    Raises
    ------
    ValueError
        If input shapes or dimensionalities are inconsistent.
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
    """Infers the interior bins of the track based on data_samples density.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_time, n_dims)
        data_samples data. NaNs are ignored.
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
    is_track_interior = bin_counts > bin_count_threshold

    n_dims = data_samples.shape[1]
    if n_dims > 1:
        # Use connectivity=1 for 4-neighbor (2D) or 6-neighbor (3D) etc.
        structure = ndimage.generate_binary_structure(n_dims, connectivity=2)

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
            if is_track_interior.ndim == 1:
                if len(is_track_interior) > 0:
                    is_track_interior[-1] = False
            elif is_track_interior.ndim > 1 and is_track_interior.size > 0:
                for axis_n in range(is_track_interior.ndim):
                    slicer_first = [slice(None)] * is_track_interior.ndim
                    slicer_first[axis_n] = 0
                    is_track_interior[tuple(slicer_first)] = False
                    slicer_last = [slice(None)] * is_track_interior.ndim
                    slicer_last[axis_n] = -1
                    is_track_interior[tuple(slicer_last)] = False

    return is_track_interior.astype(bool)


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
    """Calculates bin edges and centers for a spatial grid.

    Creates a grid based on provided data_samples data or range. Handles multiple
    data_samples dimensions and optionally adds boundary bins around the core grid.

    Parameters
    ----------
    data_samples : Optional[NDArray[np.float64]], shape (n_time, n_dims), optional
        data_samples data. Used to determine grid extent if `dimension_range`
        is None. NaNs are ignored. Required if `dimension_range` is None.
        Defaults to None.
    bin_size : Union[float, Sequence[float]], optional
        Desired approximate size of bins in each dimension. If a sequence,
        must match the number of dimensions. Defaults to 2.0.
    dimension_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit grid boundaries [(min_dim1, max_dim1), ...]. If None,
        boundaries are derived from `data_samples`. Defaults to None.
    add_boundary_bins : bool, optional
        If True, add one bin on each side of the grid in each dimension,
        extending the range. Defaults to False.

    Returns
    -------
    edges : Tuple[NDArray[np.float64], ...]
        Tuple containing bin edges for each dimension (shape (n_bins_d + 1,)).
        Includes boundary bins if `add_boundary_bins` is True.
    place_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Center coordinates of each bin in the flattened grid.
    centers_shape : Tuple[int, ...]
        Shape of the grid (number of bins in each dimension).

    Raises
    ------
    ValueError
        If both `data_samples` and `dimension_range` are None.
        If `bin_size` sequence length doesn't match dimensions.
        If `dimension_range` sequence length doesn't match dimensions.
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
    """Maps points to their corresponding bin indices in a regular grid.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_time, n_dims)
        NaNs are ignored.
    edges : Tuple[NDArray[np.float64], ...]
        Bin edges for each dimension, as returned by `create_grid`.
    centers_shape : Tuple[int, ...]
    activate_mask : Optional[NDArray[bool]], shape (n_x, ...)

    Returns
    -------
    flat_bin_indices : NDArray[np.int_], shape (n_time, n_dims)
        Indices of the bins corresponding to each data_sample point.
        Each row corresponds to a data_sample, and each column corresponds
        to a dimension. The indices are 0-based and correspond to the bin
        edges provided.
    """
    points = np.atleast_2d(points)
    points = points[~np.any(np.isnan(points), axis=1)]

    n_dims = points.shape[1]

    multi_bin_idx = tuple(
        np.digitize(points[:, i], grid_edges[i]) - 1 for i in range(n_dims)
    )
    original_bin_flat_idx = np.ravel_multi_index(multi_bin_idx, grid_shape)

    if active_mask is not None:
        # could store this in a dict for faster lookup
        active_original_flat_indices = np.flatnonzero(active_mask)
        original_flat_to_new_node_id_map: Dict[int, int] = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(active_original_flat_indices)
        }

        # Map the original bin indices to the new active bin indices
        return np.array(
            [
                original_flat_to_new_node_id_map.get(idx, -1)
                for idx in original_bin_flat_idx
            ],
            dtype=int,
        )
    else:
        # No active mask provided, return original bin indices
        return original_bin_flat_idx
