from __future__ import annotations

import itertools
import warnings
from typing import Dict, Optional, Sequence, Set, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import KDTree


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
    data_samples: NDArray[np.float64],
    bin_size: Union[float, Sequence[float]],
    dimension_range: Optional[Sequence[Tuple[float, float]]] = None,
) -> NDArray[np.int_]:
    """Calculates the number of bins needed for each dimension.

    Parameters
    ----------
    data_samples : NDArray[np.float64], shape (n_time, n_dims)
        data_samples data to determine range if `dimension_range` is not given.
    bin_size : float or Sequence[float]
        The desired size(s) of the bins.
    dimension_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit range [(min_dim1, max_dim1), ...] for each dimension.
        If None, range is calculated from `data_samples`. Defaults to None.

    Returns
    -------
    n_bins : NDArray[np.int_], shape (n_dims,)
        Number of bins required for each dimension.
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


def _generic_graph_plot(
    graph: nx.Graph,
    name: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:

    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty or None. Cannot plot an empty graph.")

    node_positions = nx.get_node_attributes(graph, "pos")

    is_3d = len(next(iter(node_positions.values()))) == 3

    if ax is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    elif is_3d and not hasattr(ax, "plot3D"):
        raise ValueError("Provided 'ax' is not 3D, but data is 3D.")

    default_node_kwargs = {"node_size": 20}
    node_kwargs = {**default_node_kwargs, **kwargs.get("node_kwargs", {})}
    nx.draw_networkx_nodes(graph, pos=node_positions, ax=ax, **node_kwargs)

    default_edge_kwargs = {"alpha": 0.5, "edge_color": "gray"}
    edge_kwargs = {**default_edge_kwargs, **kwargs.get("edge_kwargs", {})}
    nx.draw_networkx_edges(graph, pos=node_positions, ax=ax, **edge_kwargs)

    ax.set_title(f"{name} Graph")
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    if is_3d and hasattr(ax, "set_zlabel"):
        ax.set_zlabel("Dim 2")

    if not is_3d:
        ax.set_aspect("equal", adjustable="box")
    elif hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1, 1, 1])
    return ax


def _get_distance_between_bins(track_graph_nd: nx.Graph) -> NDArray[np.float64]:
    """Calculates the shortest path distances between bins in the track graph.

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
