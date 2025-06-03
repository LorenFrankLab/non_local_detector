"""
TriangularMeshLayout: a LayoutEngine that discretizes a 2D corridor region via a Delaunay triangulation.
Instead of requiring the user to supply sample points, this layout generates a uniform grid of interior points
(spaced by `point_spacing`) clipped to the provided boundary polygon. Each triangle whose centroid lies
inside the polygon is kept as an active bin. Connectivity is inferred by shared faces.

Conforms to the LayoutEngine protocol:
  - Attributes:    bin_centers, connectivity, dimension_ranges,
                   grid_edges, grid_shape, active_mask,
                   _layout_type_tag, _build_params_used
  - Properties:    is_1d
  - Methods:       build(boundary_polygon, point_spacing),
                   point_to_bin_index(points),
                   plot(ax, ...), bin_sizes()
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Point, Polygon


# --------------------------------------------------------------------------
# Module-level Private Helper Functions
# --------------------------------------------------------------------------
def _generate_interior_points_for_mesh(
    boundary_polygon: Polygon, point_spacing: float
) -> NDArray[np.float64]:
    """
    Generate a uniform grid of points spaced by `point_spacing` that lie inside `boundary_polygon`.

    Parameters
    ----------
    boundary_polygon : shapely.geometry.Polygon
        The polygon defining the boundary.
    point_spacing : float
        Spacing between generated points.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (M, 2) representing the interior sample points.
    """
    minx, miny, maxx, maxy = boundary_polygon.bounds
    xs = np.arange(minx + point_spacing / 2, maxx, point_spacing)
    ys = np.arange(miny + point_spacing / 2, maxy, point_spacing)

    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    candidates = np.column_stack((xv.ravel(), yv.ravel()))

    if candidates.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    mask_pts = shapely.covers(boundary_polygon, shapely.points(candidates))
    return candidates[mask_pts]


def _triangulate_points(sample_points: NDArray[np.float64]) -> Delaunay:
    """
    Attempt a Delaunay triangulation on `sample_points`.

    Parameters
    ----------
    sample_points : NDArray[np.float64]
        Points to triangulate, shape (N, 2).

    Returns
    -------
    scipy.spatial.Delaunay
        The Delaunay triangulation object.

    Raises
    ------
    ValueError
        If triangulation fails (e.g., points are colinear or insufficient).
    """
    if sample_points.shape[0] < 3:
        raise ValueError(
            f"Triangulation requires at least 3 points, got {sample_points.shape[0]}."
        )
    try:
        return Delaunay(sample_points)
    except QhullError as e:
        raise ValueError(
            "Delaunay triangulation failed. Points may be colinear, "
            "insufficient, or have other geometric issues."
        ) from e
    except TypeError as e:  # Can happen if points array is not float/double
        raise ValueError(
            f"Delaunay triangulation failed due to input type: {e}. Ensure points are float."
        ) from e


def _filter_active_simplices_by_centroid(
    triangulation: Delaunay, boundary_polygon: Polygon
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    Filter Delaunay simplices by requiring:
      1) The triangle's centroid lies inside or on the boundary (using `covers`).
      2) All three vertices lie inside or on the boundary.
    """
    simplices = triangulation.simplices  # shape (M, 3)
    points = triangulation.points  # shape (N, 2)

    if simplices.shape[0] == 0:
        return np.array([], dtype=int), np.empty((0, 2), dtype=float)

    # (1) compute centroids of each triangle
    triangle_vertices = points[simplices]  # (M, 3, 2)
    all_centroids = np.mean(triangle_vertices, axis=1)  # (M, 2)
    active_mask = shapely.covers(boundary_polygon, shapely.points(all_centroids))

    active_original_indices = np.flatnonzero(active_mask)
    return active_original_indices, all_centroids


def _build_mesh_connectivity_graph(
    active_original_simplex_indices: NDArray[np.int_],
    all_centroids: NDArray[np.float64],
    original_simplex_to_active_idx_map: Dict[int, int],
    delaunay_obj: Delaunay,
) -> nx.Graph:
    """
    Build a NetworkX graph for active simplices (triangles).

    Nodes correspond to active triangles. Edges connect adjacent active triangles.

    Parameters
    ----------
    active_original_simplex_indices : NDArray[np.int_]
        Original Delaunay indices of simplices that are active.
    all_centroids : NDArray[np.float64]
        Centroids of ALL original Delaunay simplices, indexed by original simplex index.
    original_simplex_to_active_idx_map : Dict[int, int]
        Maps an original Delaunay simplex index to a contiguous active triangle index (0..N-1).
    delaunay_obj : scipy.spatial.Delaunay
        The full Delaunay triangulation object.

    Returns
    -------
    nx.Graph
        The connectivity graph of active triangles.
    """
    n_active_triangles = len(active_original_simplex_indices)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_active_triangles))

    # Add node attributes
    for active_idx, original_simplex_idx in enumerate(active_original_simplex_indices):
        centroid_coords = all_centroids[original_simplex_idx]
        graph.nodes[active_idx]["pos"] = tuple(centroid_coords)
        # `source_grid_flat_index` refers to the index in the full Delaunay triangulation
        graph.nodes[active_idx]["source_grid_flat_index"] = int(original_simplex_idx)
        # `original_grid_nd_index` for a list of triangles can be a 1-tuple of that index
        graph.nodes[active_idx]["original_grid_nd_index"] = (int(original_simplex_idx),)

    # Add edges between adjacent active triangles
    delaunay_neighbors = delaunay_obj.neighbors  # shape (n_total_simplices, 3)

    for active_idx_u, original_simplex_idx_u in enumerate(
        active_original_simplex_indices
    ):
        for neighbor_original_idx_v in delaunay_neighbors[original_simplex_idx_u]:
            # If neighbor is -1, it's a boundary of the convex hull
            # If neighbor is not in our map, it means it wasn't an active triangle
            if (
                neighbor_original_idx_v != -1
                and neighbor_original_idx_v in original_simplex_to_active_idx_map
            ):

                active_idx_v = original_simplex_to_active_idx_map[
                    neighbor_original_idx_v
                ]

                # Avoid duplicate edges (graph is undirected)
                if active_idx_u < active_idx_v:
                    pos_u = np.array(graph.nodes[active_idx_u]["pos"])
                    pos_v = np.array(graph.nodes[active_idx_v]["pos"])

                    distance = float(np.linalg.norm(pos_u - pos_v))
                    displacement_vector = pos_v - pos_u
                    angle = float(
                        np.arctan2(displacement_vector[1], displacement_vector[0])
                    )

                    graph.add_edge(
                        active_idx_u,
                        active_idx_v,
                        distance=distance,
                        vector=tuple(displacement_vector.tolist()),
                        angle_2d=angle,
                    )
    return graph


def _compute_mesh_dimension_ranges(
    bin_centers_array: NDArray[np.float64],
) -> Optional[List[Tuple[float, float]]]:
    """Compute [(min_x, max_x), (min_y, max_y)] from active bin_centers."""
    if bin_centers_array.shape[0] == 0:
        return None
    min_coords = np.min(bin_centers_array, axis=0)
    max_coords = np.max(bin_centers_array, axis=0)
    return [
        (float(min_coords[d]), float(max_coords[d]))
        for d in range(bin_centers_array.shape[1])
    ]


def _sample_polygon_boundary(poly: Polygon, point_spacing: float) -> np.ndarray:
    """
    Given a Shapely Polygon, return an (M,2)-array of points sampled along
    the exterior boundary (including the vertices) at approximately 'point_spacing' intervals.
    """
    coords = np.array(poly.exterior.coords)  # shape = (N_vertices+1, 2)
    boundary_pts = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        edge_len = np.hypot(x1 - x0, y1 - y0)
        if edge_len == 0:
            continue

        # How many samples along this edge? 1 point every ~point_spacing (plus both endpoints).
        n_segs = max(int(np.ceil(edge_len / point_spacing)), 1)
        # Generate n_segs+1 points equally spaced including both ends.
        for t in np.linspace(0.0, 1.0, n_segs + 1):
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            boundary_pts.append([px, py])

    # Deduplicate (because consecutive edges share endpoints):
    boundary_pts = np.unique(np.array(boundary_pts), axis=0)
    return boundary_pts  # shape = (M, 2)
