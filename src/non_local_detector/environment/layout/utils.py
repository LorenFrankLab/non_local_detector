"""
General utility functions for the non_local_detector.environment package.

This module provides helper functions used across various components of the
environment definition and processing, such as calculating bin properties,
inferring geometric features from data samples, plotting graphs, and
computing distances.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

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


def _get_distance_between_bins(
    connectivity_graph: nx.Graph,
) -> NDArray[np.float64]:
    """
    Calculate shortest path distances between all pairs of bins in a graph.

    Uses `networkx.shortest_path_length` with the 'distance' attribute of
    edges as the weight.

    Parameters
    ----------
    connectivity_graph : nx.Graph
        A NetworkX graph where nodes represent bins (indexed `0` to `N-1`)
        and edges connect adjacent bins. Edges are expected to have a
        'distance' attribute representing the cost/length of traversing that edge.


    Returns
    -------
    distance : np.ndarray, shape (n_active_nodes, n_active_nodes)
        Matrix of shortest path distances between all pairs of active bin nodes.
    """
    n_active_bins = connectivity_graph.number_of_nodes()
    distance_matrix = np.full((n_active_bins, n_active_bins), np.inf)

    all_pairs_shortest_paths = nx.shortest_path_length(
        connectivity_graph, weight="distance"
    )

    for source_node_id, target_lengths_dict in all_pairs_shortest_paths:
        # source_node_id is an integer from 0 to n_active_bins - 1
        for target_node_id, length in target_lengths_dict.items():
            distance_matrix[source_node_id, target_node_id] = length

    return distance_matrix
