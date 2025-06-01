"""
General utility functions for the non_local_detector.environment package.

This module provides helper functions used across various components of the
environment definition and processing, such as calculating bin properties,
inferring geometric features from data samples, plotting graphs, and
computing distances.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy.typing import NDArray
from scipy.spatial import KDTree


def get_centers(bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the center of each bin given its edges.

    Parameters
    ----------
    bin_edges : NDArray[np.float64], shape (n_edges,)
        A 1D array of sorted coordinates representing the edges that define
        a sequence of bins. For `N` bins, there will be `N+1` edges.

    Returns
    -------
    NDArray[np.float64], shape (n_edges - 1,)
        A 1D array containing the center coordinate of each bin.
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def get_n_bins(
    data_samples: NDArray[np.float64],
    bin_size: Union[float, Sequence[float]],
    dimension_range: Optional[Sequence[Tuple[float, float]]] = None,
) -> NDArray[np.int_]:
    """
    Calculate the number of bins needed for each dimension of a dataset.

    The number of bins is determined based on the extent of the data (or a
    specified `dimension_range`) and the desired `bin_size`.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional data samples. Used to determine the data extent if
        `dimension_range` is not provided. NaNs are ignored for range calculation.
    bin_size : Union[float, Sequence[float]]
        The desired size of the bins. If a float, this size is applied to
        all dimensions. If a sequence, it specifies the bin size for each
        dimension and its length must match `n_dims`. Must be positive.
    dimension_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit range `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]` for
        each dimension. If None (default), the range is calculated from
        the min/max of `data_samples`.

    Returns
    -------
    NDArray[np.int_], shape (n_dims,)
        An array containing the calculated number of bins required for each
        dimension. Each value is at least 1.

    Raises
    ------
    ValueError
        If `bin_size` is not positive or if its length (if a sequence)
        does not match the number of dimensions.
        If `dimension_range` (if provided) does not have two values (min, max)
        per dimension.
    """
    if dimension_range is not None:
        # Ensure dimension_range is numpy array for consistent processing
        pr = np.asarray(dimension_range)
        if pr.shape[1] != 2:
            raise ValueError("dimension_range must be sequence of (min, max) pairs.")
        extent = np.diff(pr, axis=1).squeeze(axis=1)
    else:
        # Ignore NaNs when calculating range from data
        extent = np.nanmax(data_samples, axis=0) - np.nanmin(data_samples, axis=0)

    # Ensure bin_size is positive
    bin_size = np.asarray(bin_size, dtype=float)
    if np.any(bin_size <= 0.0):
        raise ValueError("bin_size must be positive.")

    # Calculate number of bins, ensuring at least 1 bin even if extent is 0
    n_bins = np.ceil(extent / bin_size).astype(np.int32)
    n_bins[n_bins == 0] = 1  # Handle zero extent case

    return n_bins


def _infer_active_elements_from_samples(
    candidate_element_centers: NDArray[np.float64],
    data_samples: NDArray[np.float64],
    bin_count_threshold: int = 0,
) -> Tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.int_]]:
    """
    Infer active elements from candidates based on data sample occupancy.

    This function maps `data_samples` to the nearest `candidate_element_centers`
    using a KD-tree. Candidates are marked "active" if their occupancy count
    (number of mapped data samples) exceeds `bin_count_threshold`.

    Parameters
    ----------
    candidate_element_centers : NDArray[np.float64], shape (n_candidates, n_dims)
        N-dimensional coordinates of the centers of all potential elements
        (e.g., bins, cells).
    data_samples : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional data samples (e.g., recorded positions) used to
        determine occupancy. NaNs within this array are filtered out.
    bin_count_threshold : int, optional, default=0
        Minimum number of data samples that must map to a candidate element
        for it to be considered active. If 0, any occupancy makes it active.

    Returns
    -------
    inferred_1d_mask_on_candidates : NDArray[np.bool_], shape (n_candidates,)
        A 1D boolean mask. `True` indicates the corresponding candidate
        element is active.
    final_active_centers : NDArray[np.float64], shape (n_active_elements, n_dims)
        The subset of `candidate_element_centers` that were deemed active.
    source_indices_of_active_centers : NDArray[np.int_], shape (n_active_elements,)
        Original indices (from `candidate_element_centers`) of the elements
        that were deemed active.

    Raises
    ------
    ValueError
        If `candidate_element_centers` or `data_samples` have incompatible
        dimensions, or if `bin_count_threshold` is negative.
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
    Infer min/max range for each dimension from data samples, with a buffer.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_samples, n_dims)
        Data points from which to infer ranges. NaNs are ignored.
    buffer_around_data : Union[float, Sequence[float]], default=0.0
        Buffer to add to the min and max of the inferred range in each
        dimension. If a float, applied to all dimensions. If a sequence,
        specifies buffer per dimension. Useful for ensuring extent if data
        is collinear or a margin around the data is desired.

    Returns
    -------
    Sequence[Tuple[float, float]]
        Inferred ranges: `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`.

    Raises
    ------
    ValueError
        If `data_samples` are all NaN, empty after NaN removal, not 2D,
        or if `buffer_around_data` dimensionality mismatches `data_samples`.
    TypeError
        If `buffer_around_data` is not a float or sequence of floats.
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


def _generic_graph_plot(
    graph: nx.Graph,
    name: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Provide a generic plotting function for a NetworkX graph with 2D/3D positions.

    Nodes are expected to have a 'pos' attribute containing their coordinates.
    The plot can be 2D or 3D based on the dimensionality of these positions.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to plot. Nodes must have a 'pos' attribute.
    name : str
        A name for the graph, used in the plot title.
    ax : Optional[matplotlib.axes.Axes], optional
        Matplotlib Axes object to plot on. If None (default), a new figure
        and axes are created. For 3D plots, if `ax` is provided, it must be
        a 3D-enabled Axes.
    **kwargs : Any
        Additional keyword arguments for customization:
        - `figsize` (Tuple[float, float]): Size of the figure if created.
        - `node_kwargs` (Dict[str, Any]): Keyword arguments for `nx.draw_networkx_nodes`.
        - `edge_kwargs` (Dict[str, Any]): Keyword arguments for `nx.draw_networkx_edges`.

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib Axes object on which the graph was plotted.

    Raises
    ------
    ValueError
        If `graph` is empty or None, or if a 3D plot is attempted on a 2D `ax`.
    """

    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty or None. Cannot plot an empty graph.")

    node_positions = nx.get_node_attributes(graph, "pos")
    if not node_positions:  # No 'pos' attributes
        raise ValueError("Nodes in graph are missing 'pos' attribute for plotting.")

    is_3d = len(next(iter(node_positions.values()))) == 3

    if ax is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    elif is_3d and not isinstance(ax, Axes3D):  # More specific check for 3D axes
        # Check if it has 'plot_surface' or similar 3D methods
        if not hasattr(ax, "plot_surface") and not hasattr(ax, "plot3D"):
            raise ValueError("Provided 'ax' is not a 3D Axes, but data is 3D.")

    default_node_kwargs = {
        "node_size": 20,
        "ax": ax,
    }  # Pass ax to networkx draw functions
    node_kwargs_final = {**default_node_kwargs, **kwargs.get("node_kwargs", {})}
    nx.draw_networkx_nodes(graph, pos=node_positions, **node_kwargs_final)

    default_edge_kwargs = {"alpha": 0.5}  # edge_color handled below
    edge_kwargs_final = {**default_edge_kwargs, **kwargs.get("edge_kwargs", {})}

    if is_3d:
        edge_xyz = np.array(
            [(node_positions[u], node_positions[v]) for u, v in graph.edges()]
        )
        if edge_xyz.size > 0:
            segments_3d = []
            for start_node_coords, end_node_coords in edge_xyz:
                segments_3d.append([start_node_coords, end_node_coords])

            # Extract common MPL LineCollection kwargs, provide defaults
            edge_color = edge_kwargs_final.pop("edge_color", "gray")
            linewidths = edge_kwargs_final.pop(
                "linewidths", edge_kwargs_final.pop("linewidth", 1)
            )
            alpha = edge_kwargs_final.pop("alpha", 0.5)

            line_collection = Line3DCollection(
                segments_3d,
                colors=edge_color,
                linewidths=linewidths,
                alpha=alpha,
                **edge_kwargs_final,  # Pass remaining specific kwargs
            )
            if isinstance(ax, Axes3D):  # Check if ax is indeed a 3D axis
                ax.add_collection3d(line_collection)
            else:
                warnings.warn(
                    "Attempting to add 3D edges to a non-3D Axes object. Edges may not display correctly.",
                    UserWarning,
                )

    else:  # 2D case
        edge_kwargs_final_2d = {**edge_kwargs_final, "ax": ax}
        if "edge_color" not in edge_kwargs_final_2d:  # Set default if not provided
            edge_kwargs_final_2d["edge_color"] = "gray"
        nx.draw_networkx_edges(graph, pos=node_positions, **edge_kwargs_final_2d)
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"{name} Graph")
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    if is_3d and isinstance(ax, Axes3D):
        ax.set_zlabel("Dim 2")
        # Attempt to set aspect ratio for 3D plots if possible
        # This is often tricky and depends on Matplotlib version and backend
        try:
            ax.set_box_aspect([1, 1, 1])  # For newer matplotlib
        except AttributeError:
            try:
                ax.pbaspect = [1, 1, 1]  # Older attribute
            except AttributeError:
                pass  # Fallback, may not be perfectly equal aspect

    return ax


def flat_to_multi_index(
    flat_indices: Union[int, np.ndarray],
    grid_shape: Tuple[int, ...],
    graph: nx.Graph,
) -> Union[Tuple[int, ...], Tuple[np.ndarray, ...]]:
    """
    Convert active-bin flat index(es) (0..N-1) to N-D grid index(es).

    Parameters
    ----------
    flat_indices : Union[int, np.ndarray]
        A single active-bin flat index or an array of flat indices. These
        refer to row indices in the active-bin list (i.e., node IDs).
    grid_shape : Tuple[int, ...]
        The shape of the full N-D grid (e.g., (n_rows, n_cols) for 2D).
    graph : nx.Graph
        The connectivity graph where each node ID is an active-bin ID,
        and node attributes include 'original_grid_nd_index' or
        'source_grid_flat_index'.

    Returns
    -------
    Union[Tuple[int, ...], Tuple[np.ndarray, ...]]
        - If `flat_indices` is a single int, return a tuple of N ints: the N-D index.
          If conversion fails for that index, return a tuple of N `np.nan`s.
        - If `flat_indices` is an array of ints, return a tuple of N NumPy arrays,
          each containing that coordinate for each input index. If conversion fails
          for some index, its coordinate elements are `np.nan`.

    Raises
    ------
    ValueError
        If none of the active_flat_idx values are valid node IDs, or if `grid_shape` is invalid.
    """
    # Ensure we have a NumPy array for iteration
    is_scalar = np.isscalar(flat_indices)
    flat_arr = np.atleast_1d(np.asarray(flat_indices, dtype=int))
    node_data_lookup: Dict[int, Dict[str, Any]] = dict(graph.nodes(data=True))

    n_dims = len(grid_shape)
    # Prepare a list to collect tuples of length n_dims
    output_nd_list = []

    for active_idx in flat_arr:
        # Check if this node ID exists in node_data_lookup
        if active_idx not in node_data_lookup:
            warnings.warn(
                f"Active flat index {active_idx} not found in node_data_lookup. "
                "Returning NaNs for this index.",
                UserWarning,
            )
            output_nd_list.append(tuple([np.nan] * n_dims))
            continue

        data = node_data_lookup[active_idx]

        # If original N-D tuple is directly available, use it
        original_index_key: str = "original_grid_nd_index"
        fallback_key: str = "source_grid_flat_index"
        if original_index_key in data and data[original_index_key] is not None:
            orig_nd = data[original_index_key]
            if not (isinstance(orig_nd, (tuple, list)) and len(orig_nd) == n_dims):
                warnings.warn(
                    f"Node {active_idx} has invalid '{original_index_key}' attribute. "
                    "Returning NaNs.",
                    UserWarning,
                )
                output_nd_list.append(tuple([np.nan] * n_dims))
            else:
                output_nd_list.append(tuple(orig_nd))
        # Otherwise, attempt to use the fallback full-grid flat index
        elif fallback_key in data and data[fallback_key] is not None:
            full_flat = data[fallback_key]
            try:
                nd_idx = np.unravel_index(int(full_flat), grid_shape)
                output_nd_list.append(tuple(int(x) for x in nd_idx))
            except Exception:
                warnings.warn(
                    f"Cannot unravel fallback index {full_flat} for node {active_idx}. "
                    "Returning NaNs.",
                    UserWarning,
                )
                output_nd_list.append(tuple([np.nan] * n_dims))
        else:
            warnings.warn(
                f"Node {active_idx} missing both '{original_index_key}' and '{fallback_key}'. "
                "Returning NaNs.",
                UserWarning,
            )
            output_nd_list.append(tuple([np.nan] * n_dims))

    # Convert list of tuples into tuple of arrays
    # e.g., output_nd_list = [(r0,c0), (r1,c1), ...] for 2D
    # then final = (array([r0,r1,...]), array([c0,c1,...]))
    final = tuple(
        np.array([item[d] for item in output_nd_list], dtype=float)
        for d in range(n_dims)
    )

    if is_scalar:
        # Return a tuple of scalars (first element of each array)
        return tuple(int(val[0]) if not np.isnan(val[0]) else np.nan for val in final)
    return final


def multi_index_to_flat(
    *nd_idx_per_dim: Union[int, np.ndarray],
    grid_shape: Tuple[int, ...],
    active_mask: np.ndarray,
    source_flat_lookup: Dict[int, int],
) -> Union[int, np.ndarray]:
    """
    Convert N-D grid index(es) to active-bin flat index(es) (0..N-1).

    Parameters
    ----------
    *nd_idx_per_dim : Union[int, np.ndarray]
        N separate arguments, one per dimension. Each can be an int (scalar)
        or a NumPy array of ints (broadcastable). Examples:
          - (row, col) for a single 2D point
          - (rows_array, cols_array) for multiple points
        Alternatively, a single argument that is a list/tuple of length N
        can be passed, in which case it is interpreted as:
          - shape (N, n_points) or (n_points, N) or (N,) for a single N-D index.
    grid_shape : Tuple[int, ...]
        The shape of the full grid (e.g. (n_rows, n_cols) for 2D).
    active_mask : np.ndarray
        An N-D boolean mask of shape=grid_shape, where True indicates the grid
        cell is active. Inactive cells are not part of the state space.
    source_flat_lookup : Dict[int, int]
        A mapping from full-grid flat index → active-bin flat index (state ID).
        If a grid cell is inactive, it should not appear (or return -1).

    Returns
    -------
    Union[int, np.ndarray]
        - If input implies a single grid point, returns the active-bin ID (int), or -1 if out of bounds or inactive.
        - If input implies multiple grid points (arrays), returns an array of the same broadcast shape,
          with each element either the active-bin ID or -1.

    Raises
    ------
    ValueError
        If the number of input arrays does not match len(grid_shape), or if input arrays cannot broadcast.
    """
    # Handle the case of a single iterable argument, e.g. ([r1,r2],[c1,c2]) or ((r1,c1),(r2,c2))
    if len(nd_idx_per_dim) == 1 and isinstance(nd_idx_per_dim[0], (list, tuple)):
        temp = np.asarray(nd_idx_per_dim[0])
        if temp.ndim == 2 and temp.shape[0] == len(grid_shape):
            # shape = (n_dims, n_points)
            nd_idx_per_dim = tuple(temp[d] for d in range(len(grid_shape)))
        elif temp.ndim == 2 and temp.shape[1] == len(grid_shape):
            # shape = (n_points, n_dims)
            nd_idx_per_dim = tuple(temp[:, d] for d in range(len(grid_shape)))
        elif temp.ndim == 1 and temp.shape[0] == len(grid_shape):
            # single N-D index given as a 1-D array
            nd_idx_per_dim = tuple(np.array([int(val)]) for val in temp)
        else:
            raise ValueError("Invalid format for single argument N-D index.")

    # Now expect len(nd_idx_per_dim) == len(grid_shape)
    if len(nd_idx_per_dim) != len(grid_shape):
        raise ValueError(
            f"Expected {len(grid_shape)} index arrays, got {len(nd_idx_per_dim)}."
        )

    # Convert each to a NumPy array of dtype int
    nd_arrays = tuple(
        np.atleast_1d(np.asarray(idx, dtype=int)) for idx in nd_idx_per_dim
    )

    # Determine the common broadcast shape
    try:
        common_shape = np.broadcast(*nd_arrays).shape
    except ValueError:
        raise ValueError("N-D index arrays could not be broadcast together.")

    # Initialize output with -1
    flat_output = np.full(common_shape, -1, dtype=int)

    # Create a mask of indices that lie within the grid bounds
    in_bounds = np.ones(common_shape, dtype=bool)
    for dim, arr in enumerate(nd_arrays):
        in_bounds &= (arr >= 0) & (arr < grid_shape[dim])

    if not np.any(in_bounds):
        # No in-bounds points; return either scalar -1 or the array of -1s
        if common_shape == (1,) and all(np.isscalar(idx) for idx in nd_idx_per_dim):
            return int(-1)
        return flat_output

    # Extract the valid in-bounds indices for each dimension
    valid_nd = tuple(arr[in_bounds] for arr in nd_arrays)

    # Check which of those valid N-D indices correspond to active bins
    try:
        active_mask_vals = active_mask[valid_nd]
    except IndexError:
        # If indices out of range of active_mask, treat them as inactive
        active_mask_vals = np.zeros(len(valid_nd[0]), dtype=bool)

    if not np.any(active_mask_vals):
        # None of the in-bounds indices are active
        if common_shape == (1,) and all(np.isscalar(idx) for idx in nd_idx_per_dim):
            return int(-1)
        return flat_output

    # Keep only those indices that are both in-bounds & active
    truly_active_nd = tuple(dim_arr[active_mask_vals] for dim_arr in valid_nd)

    # Convert these N-D grid coords to full-grid flat indices
    full_flat_inds = np.ravel_multi_index(truly_active_nd, grid_shape)

    # Map full-grid flat indices to active-bin IDs using source_flat_lookup
    active_ids = np.array(
        [source_flat_lookup.get(int(ff), -1) for ff in full_flat_inds], dtype=int
    )

    # Place these active IDs back into the correct positions of the output array
    final_mask = np.zeros(common_shape, dtype=bool)
    final_mask[in_bounds] = active_mask_vals
    flat_output[final_mask] = active_ids

    # If this was originally scalar inputs, return a scalar
    if common_shape == (1,) and all(np.isscalar(idx) for idx in nd_idx_per_dim):
        return int(flat_output[0])
    return flat_output


def find_boundary_nodes(
    graph: nx.Graph,
    grid_shape: Tuple[int, ...] = None,
    active_mask: np.ndarray = None,
    layout_kind: str = None,
) -> np.ndarray:
    """
    Identify boundary nodes in a connectivity graph G.

    A node is considered “boundary” if:
      1) For an N>1 grid layout (grid_shape and active_mask provided, and
         connectivity_threshold_factor is None), it lies on the edge of the
         active region (i.e., at least one of its N-D neighbors is either
         out of bounds or inactive).
      2) Otherwise, use a degree-based heuristic:
         - Compute median_degree = median(node_degree) over all nodes.
         - If connectivity_threshold_factor is provided (>0), threshold = factor * median_degree.
           Nodes with degree < threshold are boundary.
         - If no factor:
           • For Hexagonal (layout_kind=="Hexagonal") and median_degree>5, threshold=5.5.
           • For 1D (len(grid_shape)==1 or layout_kind=="Graph"), threshold=1.5 if max_degree>=2.
           • For general fallback, threshold = median_degree.
           Nodes with degree < threshold are boundary.
      3) If no nodes found by the above and no factor given and not an N>1 grid, fallback:
         - Let max_degree = max(node_degree). If median_degree == max_degree, do nothing.
           Else, mark nodes with degree < max_degree.

    Parameters
    ----------
    graph : nx.Graph
        The connectivity graph where each node ID is an active-bin ID,
        and node attributes include possibly 'original_grid_nd_index'.
    grid_shape : Tuple[int, ...], optional
        The shape of the full grid. Only used if len(grid_shape) > 1 and
        active_mask is provided.
    active_mask : np.ndarray, optional
        An N-D boolean mask of shape=grid_shape. True=active bin.
    layout_kind : str, optional
        A string identifier (e.g. "RegularGrid", "Hexagonal", "Graph").
        Used for special heuristics (e.g. hexagon degrees).


    Returns
    -------
    np.ndarray[np.int_]
        Sorted array of node IDs (active-bin IDs) identified as boundary.

    Raises
    ------
    ValueError
        If G is empty or invalid parameters are provided.
    """
    if graph.number_of_nodes() == 0:
        return np.array([], dtype=int)

    boundary_bin_indices = []

    # Grid-specific logic for N-D grids (N > 1)
    is_nd_grid_layout_with_mask = (
        active_mask is not None and grid_shape is not None and len(grid_shape) > 1
    )

    if is_nd_grid_layout_with_mask:
        n_dims = len(grid_shape)

        for active_node_id in graph.nodes():
            node_data = graph.nodes[active_node_id]
            original_nd_idx = node_data.get("original_grid_nd_index")
            if original_nd_idx is None:
                warnings.warn(
                    f"Node {active_node_id} missing 'original_grid_nd_index'. "
                    "Cannot use N-D grid boundary logic. Consider degree-based method.",
                    UserWarning,
                )
                # Potentially fall back to degree-based for this node or skip
                continue  # Skip this node for grid logic

            is_boundary = False
            for dim_idx in range(n_dims):
                for offset_val in [-1, 1]:
                    neighbor_nd_idx_list = list(original_nd_idx)
                    neighbor_nd_idx_list[dim_idx] += offset_val
                    neighbor_nd_idx = tuple(neighbor_nd_idx_list)

                    if not (0 <= neighbor_nd_idx[dim_idx] < grid_shape[dim_idx]):
                        is_boundary = True
                        break
                    if not active_mask[neighbor_nd_idx]:
                        is_boundary = True
                        break
                if is_boundary:
                    break
            if is_boundary:
                boundary_bin_indices.append(active_node_id)
    else:
        # Degree-based heuristic for:
        # - Non-grid layouts
        # - 1D grid layouts (where len(grid_shape) == 1)
        # - N-D grid layouts if connectivity_threshold_factor is provided
        # - N-D grid layouts if 'original_grid_nd_index' is missing for nodes

        degrees = dict(graph.degree())
        if not degrees:
            return np.array([], dtype=int)

        all_degree_values = list(degrees.values())
        median_degree = np.median(all_degree_values)
        max_degree_val = np.max(all_degree_values) if all_degree_values else 0

        if layout_kind == "Hexagonal" and median_degree > 5:
            threshold_degree = 5.5  # Bins with < 6 neighbors
        elif layout_kind == "Graph":
            # Threshold just below typical internal degree.
            threshold_degree = (
                1.5 if max_degree_val >= 2 else max_degree_val
            )  # Catches degree 1
        elif is_nd_grid_layout_with_mask and median_degree > (
            2 * len(grid_shape) - 1
        ):  # For N>1 grids when falling back
            threshold_degree = 2 * len(grid_shape) - 0.5
        else:  # General fallback if no specific layout recognized or no factor
            threshold_degree = median_degree  # Bins with degree < median are boundary

        for node_id, degree in degrees.items():
            if degree < threshold_degree:
                boundary_bin_indices.append(node_id)

        # If the above found nothing, and it's not a specific grid case where that's expected,
        # a simple fallback for general graphs: degree < max_degree
        if (
            not boundary_bin_indices
            and not is_nd_grid_layout_with_mask  # Avoid if it was an ND-grid that correctly found no boundaries
            and max_degree_val > 0
        ):
            # This ensures that if all nodes have the same degree (e.g. a k-regular graph like a circle)
            # it doesn't mark all as boundary unless median was already low.
            # Only mark as boundary if degree is strictly less than the absolute max.
            # This is a very general fallback.
            if median_degree == max_degree_val:  # e.g. all nodes have same degree
                pass  # Don't mark any as boundary unless threshold_degree was already < max_degree_val
            else:
                for node_id, degree in degrees.items():
                    if (
                        degree < max_degree_val
                    ):  # Stricter than degree < median_degree for some cases
                        if node_id not in boundary_bin_indices:  # Avoid re-adding
                            boundary_bin_indices.append(node_id)

    return np.array(sorted(list(set(boundary_bin_indices))), dtype=int)


def map_active_data_to_grid(
    grid_shape: Tuple[int, ...],
    active_mask: np.ndarray,
    active_bin_data: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Map a 1D array of data corresponding to active bins onto a full N-D grid.

    This is useful for visualizing data (e.g., place fields, posterior
    probabilities) on grid-based layouts using functions like `pcolormesh`.

    Parameters
    ----------
    active_bin_data : NDArray[np.float64], shape (n_bins,)
        A 1D array of values, one for each active bin, ordered consistently
        with `self.bin_centers` (i.e., by active bin index 0 to `n_bins - 1`).
    fill_value : float, optional
        The value to use for bins in the full grid that are not active,
        by default np.nan.

    Returns
    -------
    NDArray[np.float64], shape (dim0_size, dim1_size, ...)
        An N-D array with shape `self.grid_shape`. Active bin locations are
        filled with `active_bin_data`, others with `fill_value`.

    Raises
    ------
    ValueError
        If the environment is not grid-based (missing `grid_shape` or
        `active_mask`), or if `active_bin_data` has an incorrect shape
        or incompatible data type.
    """
    if grid_shape is None or active_mask is None:
        raise ValueError(
            "This method is applicable only to grid-based environments "
            "that have 'grid_shape' and 'active_mask' attributes."
        )
    if not isinstance(active_bin_data, np.ndarray) or active_bin_data.ndim != 1:
        raise ValueError("active_bin_data must be a 1D NumPy array.")

    n_bins = active_mask.sum()
    if active_bin_data.shape[0] != n_bins:
        raise ValueError(
            f"Length of active_bin_data ({active_bin_data.shape[0]}) "
            f"must match the number of active bins ({n_bins})."
        )

    # Create an array for the full grid, filled with the fill_value
    # Ensure dtype compatibility, e.g., promote fill_value to active_bin_data.dtype
    # or choose a suitable default like float if active_bin_data can be int.
    dtype = np.result_type(active_bin_data.dtype, type(fill_value))
    full_grid_data = np.full(grid_shape, fill_value, dtype=dtype)

    # Place the active data into the grid using the N-D active_mask
    full_grid_data[active_mask] = active_bin_data
    return full_grid_data
