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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely.vectorized
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from non_local_detector.environment.layout.layout_engine import LayoutEngine


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

    mask_pts = shapely.vectorized.covers(
        boundary_polygon, candidates[:, 0], candidates[:, 1]
    )
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

    # (2) first pass: does the centroid lie in (or on) the boundary?
    mask_centroids = shapely.vectorized.covers(
        boundary_polygon, all_centroids[:, 0], all_centroids[:, 1]
    )

    active_mask = np.zeros(mask_centroids.shape[0], dtype=bool)
    prepped = prep(boundary_polygon)

    for idx in np.flatnonzero(mask_centroids):
        # If centroid is covered, then check each of the 3 vertices
        verts = triangle_vertices[idx]  # shape (3, 2)
        # Use `prepped.covers` to include boundary‐points as 'inside'
        if all(prepped.covers(Point(v)) for v in verts):
            active_mask[idx] = True

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
                        weight=1 / distance if distance != 0 else np.inf,
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


# --------------------------------------------------------------------------
# TriangularMeshLayout Class
# --------------------------------------------------------------------------
class TriangularMeshLayout(LayoutEngine):
    """
    A LayoutEngine that builds a triangular mesh over interior points
    (auto-generated) clipped to a boundary polygon. Each triangle whose centroid
    lies inside the polygon is kept as an active bin. Connectivity by shared faces.
    """

    _layout_type_tag: str = "TriangularMesh"
    _build_params_used: Dict[str, Any]

    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: Optional[Sequence[Tuple[float, float]]]
    grid_edges: Tuple[()]  # Explicitly empty tuple for non-grid
    grid_shape: Optional[Tuple[int, ...]]
    active_mask: Optional[NDArray[np.bool_]]

    # Internal state
    _full_delaunay_tri: Optional[Delaunay]
    _original_simplex_to_active_idx_map: Optional[Dict[int, int]]
    _active_original_simplex_indices: Optional[
        NDArray[np.int_]
    ]  # Store original indices of active ones
    _boundary_polygon_stored: Optional[Polygon]  # Store the actual polygon object

    def __init__(self):
        self.bin_centers = np.empty((0, 2), dtype=float)
        self.connectivity = nx.Graph()
        self.dimension_ranges = None
        self.grid_edges = ()  # For non-grid layouts, this is typically empty
        self.grid_shape = None
        self.active_mask = None  # Will be updated in build
        self.is_1d = False  # This is a 2D layout
        self._build_params_used = {}

        self._full_delaunay_tri = None
        self._original_simplex_to_active_idx_map = None
        self._active_original_simplex_indices = None
        self._boundary_polygon_stored = None

    def build(self, boundary_polygon: Polygon, point_spacing: float) -> None:
        """
        Build the triangular mesh layout.

        Parameters
        ----------
        boundary_polygon : shapely.geometry.Polygon
            The polygon defining the boundary. Triangles with centroids
            outside this polygon are discarded.
        point_spacing : float
            Desired spacing (in same units as polygon) between generated sample points
            used for triangulation.
        """
        if not isinstance(boundary_polygon, Polygon):
            raise TypeError("boundary_polygon must be a Shapely Polygon.")
        if boundary_polygon.is_empty:
            raise ValueError("boundary_polygon cannot be empty.")
        if boundary_polygon.geom_type == "MultiPolygon":
            raise ValueError(
                "MultiPolygon boundaries are not directly supported. "
                "Please provide a single Polygon component."
            )
        if point_spacing <= 0:
            raise ValueError(f"point_spacing must be positive, got {point_spacing}.")

        # Store build parameters.
        # For boundary_polygon, store exterior and interior coords for better serialization.
        # Note: This simple serialization of polygon might not capture all Shapely Polygon features perfectly.
        boundary_exterior_coords = list(boundary_polygon.exterior.coords)
        boundary_interior_coords_list = [
            list(interior.coords) for interior in boundary_polygon.interiors
        ]
        self._build_params_used = {
            "boundary_exterior_coords": boundary_exterior_coords,
            "boundary_interior_coords_list": boundary_interior_coords_list,
            "point_spacing": float(point_spacing),
        }
        self._boundary_polygon_stored = (
            boundary_polygon  # Store the actual object for use
        )

        # 1. Generate sample points for triangulation
        sample_points = _generate_interior_points_for_mesh(
            boundary_polygon, point_spacing
        )
        if sample_points.shape[0] < 3:  # Delaunay needs at least N+1 points in N-D
            raise ValueError(
                f"Not enough interior sample points ({sample_points.shape[0]}) generated "
                "to form any triangle. Try decreasing point_spacing or ensuring "
                "the polygon is large enough relative to the spacing."
            )

        # 2. Perform Delaunay triangulation
        self._full_delaunay_tri = _triangulate_points(sample_points)

        # 3. Filter active simplices (triangles)
        active_original_indices, all_centroids = _filter_active_simplices_by_centroid(
            self._full_delaunay_tri, boundary_polygon
        )
        n_total_delaunay_triangles = self._full_delaunay_tri.simplices.shape[0]

        if active_original_indices.size == 0:
            raise ValueError(
                "No triangles found with centroids inside the boundary polygon. "
                "Check boundary_polygon shape, point_spacing, or point generation strategy."
            )

        self._active_original_simplex_indices = active_original_indices

        # 4. Create mapping from original Delaunay simplex index to active triangle index
        self._original_simplex_to_active_idx_map = {
            orig_idx: active_idx
            for active_idx, orig_idx in enumerate(active_original_indices)
        }

        # 5. Populate core attributes for active triangles
        self.bin_centers = all_centroids[active_original_indices]
        n_active_triangles = self.bin_centers.shape[0]

        # 6. Build connectivity graph for active triangles
        self.connectivity = _build_mesh_connectivity_graph(
            active_original_indices,
            all_centroids,  # Pass all centroids
            self._original_simplex_to_active_idx_map,
            self._full_delaunay_tri,
        )

        # 7. Compute dimension_ranges based on active bin centers
        self.dimension_ranges = _compute_mesh_dimension_ranges(self.bin_centers)

        # 8. Set grid-related attributes for protocol conformance
        # The "conceptual grid" here is the list of all Delaunay triangles.
        self.grid_shape = (
            n_total_delaunay_triangles,
        )  # Shape of the full Delaunay triangle list
        self.active_mask = np.zeros(n_total_delaunay_triangles, dtype=bool)
        self.active_mask[active_original_indices] = True
        # self.grid_edges remains an empty tuple as it's not a rectilinear grid.

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map each 2D point to an active triangle index, or -1 if outside.

        Uses Delaunay.find_simplex() and then maps to active triangle indices.
        Note: This method assigns a point to an active triangle if the point falls
        within the simplex corresponding to that active triangle. If the boundary
        polygon is non-convex, a point might be outside the boundary polygon but
        still within a simplex whose centroid was inside the boundary polygon.
        For stricter checking, an additional `boundary_polygon.contains(point)`
        test could be applied.

        Parameters
        ----------
        points : NDArray[np.float64]
            Points to map, shape (M, 2).

        Returns
        -------
        NDArray[np.int_]
            Active triangle indices for each point, shape (M,).
            -1 if point is outside the convex hull of mesh points or
            if the simplex found is not among the active ones.
        """
        if (
            self._full_delaunay_tri is None
            or self._original_simplex_to_active_idx_map is None
        ):
            raise RuntimeError("TriangularMeshLayout is not built. Call build() first.")

        points_2d = np.atleast_2d(points)
        if points_2d.ndim != 2 or points_2d.shape[1] != 2:
            raise ValueError(f"Expected points of shape (M, 2), got {points_2d.shape}.")

        # Find which original Delaunay simplex each point belongs to
        # Returns -1 if outside the convex hull of self._full_delaunay_tri.points
        original_simplex_indices = self._full_delaunay_tri.find_simplex(points_2d)

        # Map these original simplex indices to active triangle indices
        # Default to -1 (unassigned)
        active_triangle_idxs = np.full(original_simplex_indices.shape, -1, dtype=int)

        # Create a mask for points that fell into any simplex (not -1)
        valid_simplex_mask = original_simplex_indices != -1

        if np.any(valid_simplex_mask):
            found_original_simplices = original_simplex_indices[valid_simplex_mask]
            # Vectorized lookup array
            n_total = self._full_delaunay_tri.simplices.shape[0]
            orig2active_arr = np.full(n_total, -1, dtype=int)
            for (
                orig_idx,
                active_idx,
            ) in self._original_simplex_to_active_idx_map.items():
                orig2active_arr[orig_idx] = active_idx

            # Now do one vectorized assignment
            active_triangle_idxs[valid_simplex_mask] = orig2active_arr[
                found_original_simplices
            ]

        valid_mask = original_simplex_indices != -1
        if np.any(valid_mask):
            found_orig = original_simplex_indices[valid_mask]
            mapped_to_active = np.array(
                [
                    self._original_simplex_to_active_idx_map.get(orig_idx, -1)
                    for orig_idx in found_orig
                ],
                dtype=int,
            )

            # Temporarily assign those bins
            active_triangle_idxs[valid_mask] = mapped_to_active

            # Now enforce: each point must itself be inside or on the boundary
            if self._boundary_polygon_stored is not None:
                import shapely.vectorized

                xcoords = points_2d[valid_mask, 0]
                ycoords = points_2d[valid_mask, 1]
                # Use covers to allow points exactly on boundary
                on_or_inside = shapely.vectorized.covers(
                    self._boundary_polygon_stored, xcoords, ycoords
                )
                # Wherever a “valid” point is not on or inside the polygon, override to -1
                idxs_valid = np.flatnonzero(valid_mask)
                for i_local, mask_val in enumerate(on_or_inside):
                    if not mask_val:
                        active_triangle_idxs[idxs_valid[i_local]] = -1

        return active_triangle_idxs

    @property
    def is_1d(self) -> bool:
        """Always False, as this is a 2D mesh layout."""
        return False

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show_triangles: bool = True,
        show_centroids: bool = True,
        show_connectivity: bool = True,
        show_boundary: bool = True,
        triangle_kwargs: Optional[Dict[str, Any]] = None,
        centroid_kwargs: Optional[Dict[str, Any]] = None,
        connectivity_kwargs: Optional[Dict[str, Any]] = None,
        boundary_kwargs: Optional[Dict[str, Any]] = None,
    ) -> plt.Axes:
        """
        Plot the triangular mesh layout.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, a new figure and axes are created.
        show_triangles : bool, optional
            Whether to draw the filled active triangles. Defaults to True.
        show_centroids : bool, optional
            Whether to draw the centroids of active triangles. Defaults to True.
        show_connectivity : bool, optional
            Whether to draw edges of the connectivity graph. Defaults to True.
        show_boundary : bool, optional
            Whether to draw the original boundary polygon. Defaults to True.
        triangle_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for `matplotlib.collections.PatchCollection` of triangles.
        centroid_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for `ax.scatter` plotting centroids.
        connectivity_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for plotting connectivity edges.
        boundary_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for plotting the boundary polygon.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout was plotted.
        """
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
            or self.bin_centers is None
            or self.connectivity is None
            or self.dimension_ranges is None
        ):
            raise RuntimeError("TriangularMeshLayout is not built. Call build() first.")

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))  # Default figsize

        # Default kwargs
        _triangle_kwargs = {
            "alpha": 0.4,
            "facecolor": "lightblue",
            "edgecolor": "gray",
            "linewidth": 0.5,
        }
        if triangle_kwargs:
            _triangle_kwargs.update(triangle_kwargs)

        _centroid_kwargs = {"color": "blue", "s": 10, "zorder": 3}
        if centroid_kwargs:
            _centroid_kwargs.update(centroid_kwargs)

        _connectivity_kwargs = {
            "color": "black",
            "alpha": 0.5,
            "linewidth": 0.75,
            "zorder": 2,
        }
        if connectivity_kwargs:
            _connectivity_kwargs.update(connectivity_kwargs)

        _boundary_kwargs = {
            "color": "black",
            "linewidth": 1.5,
            "linestyle": "--",
            "zorder": 4,
        }
        if boundary_kwargs:
            _boundary_kwargs.update(boundary_kwargs)

        # Plot boundary polygon
        if show_boundary and self._boundary_polygon_stored:
            xb, yb = self._boundary_polygon_stored.exterior.xy
            ax.plot(xb, yb, label="Boundary", **_boundary_kwargs)
            for interior in self._boundary_polygon_stored.interiors:
                xbi, ybi = interior.xy
                ax.plot(xbi, ybi, **_boundary_kwargs)

        # Plot active triangles
        if show_triangles:
            patches: List[MplPolygon] = []
            mesh_points = self._full_delaunay_tri.points
            active_simplices_vertices = mesh_points[
                self._full_delaunay_tri.simplices[self._active_original_simplex_indices]
            ]

            for vertices in active_simplices_vertices:  # Iterate over (N_active, 3, 2)
                patches.append(MplPolygon(vertices, closed=True))

            pc = PatchCollection(patches, **_triangle_kwargs)
            ax.add_collection(pc)

        # Plot centroids (which are self.bin_centers)
        if show_centroids and self.bin_centers.shape[0] > 0:
            ax.scatter(
                self.bin_centers[:, 0], self.bin_centers[:, 1], **_centroid_kwargs
            )

        # Plot connectivity edges
        if show_connectivity:
            for u, v in self.connectivity.edges():
                pos_u = self.connectivity.nodes[u]["pos"]
                pos_v = self.connectivity.nodes[v]["pos"]
                ax.plot(
                    [pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], **_connectivity_kwargs
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.dimension_ranges[0])
        ax.set_ylim(self.dimension_ranges[1])
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(self._layout_type_tag)
        if (
            show_boundary and self._boundary_polygon_stored
        ):  # Add legend if boundary shown
            ax.legend()
        return ax

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Return the area of each active triangle.

        Uses the formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|.
        Alternatively, via cross product: 0.5 * |(p1-p0) x (p2-p0)|.

        Returns
        -------
        NDArray[np.float64]
            Array of areas, shape (n_active_triangles,).
        """
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
        ):
            raise RuntimeError("TriangularMeshLayout is not built. Call build() first.")

        active_simplices = self._full_delaunay_tri.simplices[
            self._active_original_simplex_indices
        ]
        mesh_points = self._full_delaunay_tri.points

        # Get vertices for all active triangles: shape (n_active_triangles, 3, 2)
        triangle_vertices = mesh_points[active_simplices]

        # Vectorized area calculation using cross product
        # p0, p1, p2 are arrays of shape (n_active_triangles, 2)
        p0 = triangle_vertices[:, 0, :]
        p1 = triangle_vertices[:, 1, :]
        p2 = triangle_vertices[:, 2, :]

        # (p1-p0) and (p2-p0)
        vec1 = p1 - p0  # shape (n_active_triangles, 2)
        vec2 = p2 - p0  # shape (n_active_triangles, 2)

        # Cross product for 2D vectors (v1x*v2y - v1y*v2x)
        cross_product_values = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]

        areas = 0.5 * np.abs(cross_product_values)
        return areas
