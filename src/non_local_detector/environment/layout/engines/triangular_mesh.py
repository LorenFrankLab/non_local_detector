from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely.vectorized
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from non_local_detector.environment.layout.helpers.triangular_mesh import (
    _build_mesh_connectivity_graph,
    _compute_mesh_dimension_ranges,
    _filter_active_simplices_by_centroid,
    _generate_interior_points_for_mesh,
    _sample_polygon_boundary,
    _triangulate_points,
)


class TriangularMeshLayout:
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

    is_1d: bool = False  # This is a 2D layout

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
        exterior_coords = _sample_polygon_boundary(boundary_polygon, point_spacing)
        # 4) Stack the interior grid points with the boundary vertices.
        if sample_points.size == 0:
            sample_points = exterior_coords.copy()
        else:
            sample_points = np.vstack([sample_points, exterior_coords])

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

        # 6. Build connectivity graph for active triangles
        self.connectivity = _build_mesh_connectivity_graph(
            active_original_indices,
            all_centroids,
            self._original_simplex_to_active_idx_map,
            self._full_delaunay_tri,
        )

        # 7. Compute dimension_ranges based on active bin centers
        self.dimension_ranges = _compute_mesh_dimension_ranges(self.bin_centers)

        # 8. Set grid-related attributes for protocol conformance
        # The "conceptual grid" here is the list of all Delaunay triangles.
        self.grid_shape = (n_total_delaunay_triangles,)
        self.active_mask = np.zeros(n_total_delaunay_triangles, dtype=bool)
        self.active_mask[active_original_indices] = True
        # self.grid_edges remains an empty tuple as it's not a rectilinear grid.

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map each 2D point to an active triangle index, or -1 if outside.

        Uses Delaunay.find_simplex() → original-simplex index → active-triangle index via
        a fast lookup array. Then enforces that the point itself must lie inside (or on)
        the boundary polygon.

        Returns
        -------
        NDArray[np.int_]
            Each entry is in [0..n_active-1] or -1 if outside.
        """
        if (
            self._full_delaunay_tri is None
            or self._original_simplex_to_active_idx_map is None
        ):
            raise RuntimeError("TriangularMeshLayout is not built. Call build() first.")

        pts2d = np.atleast_2d(points).astype(np.float64, copy=False)
        if pts2d.ndim != 2 or pts2d.shape[1] != 2:
            raise ValueError(f"Expected points of shape (M, 2), got {pts2d.shape}.")

        # 1) Find which Delaunay simplex each point falls into (-1 if outside hull)
        orig_simplices = self._full_delaunay_tri.find_simplex(pts2d)

        # 2) Build a 1D lookup array once, mapping original simplex idx -> active idx
        n_total = self._full_delaunay_tri.simplices.shape[0]
        orig2active_arr = np.full(n_total, -1, dtype=int)
        for orig_idx, active_idx in self._original_simplex_to_active_idx_map.items():
            orig2active_arr[orig_idx] = active_idx

        # 3) Initialize result array to -1
        active_triangle_idxs = np.full(orig_simplices.shape, -1, dtype=int)

        # 4) Wherever orig_simplices != -1, do a vectorized assignment
        valid_mask = orig_simplices != -1
        if np.any(valid_mask):
            found_orig = orig_simplices[valid_mask]
            active_triangle_idxs[valid_mask] = orig2active_arr[found_orig]

            # 5) Now ensure each point is itself inside (or on) the boundary
            if self._boundary_polygon_stored is not None:
                xcoords = pts2d[valid_mask, 0]
                ycoords = pts2d[valid_mask, 1]
                on_or_inside = shapely.vectorized.contains(
                    self._boundary_polygon_stored, xcoords, ycoords
                )
                idxs = np.flatnonzero(valid_mask)
                for local_i, keep in enumerate(on_or_inside):
                    if not keep:
                        active_triangle_idxs[idxs[local_i]] = -1

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

            for vertices in active_simplices_vertices:  # Iterate over (n_active, 3, 2)
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
        Return the N-dimensional volume of each active N-simplex (bin).

        For 2D, this is area. For 3D, this is volume, and so on.
        The volume of an N-simplex with vertices v0, v1, ..., vn is calculated as:
        V = (1 / n!) * |det([v1-v0, v2-v0, ..., vn-v0])|
        where n is the number of dimensions.

        Returns
        -------
        NDArray[np.float64]
            Array of N-dimensional volumes, shape (n_active_simplices,).

        Raises
        ------
        RuntimeError
            If the layout is not built (e.g., `_full_delaunay_tri` or
            `_active_original_simplex_indices` is None).
        ValueError
            If the dimensionality is less than 1.
        """
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
        ):  # pragma: no cover
            raise RuntimeError("TriangularMeshLayout is not built. Call build() first.")

        # Get the vertices of all active N-simplices
        # .points has shape (total_points, n_dim)
        # .simplices has shape (total_simplices, n_dim + 1)
        all_mesh_points = self._full_delaunay_tri.points
        active_simplices_vertex_indices = self._full_delaunay_tri.simplices[
            self._active_original_simplex_indices
        ]

        if active_simplices_vertex_indices.shape[0] == 0:
            return np.array([], dtype=float)  # No active simplices

        # nsimplex_vertices will have shape:
        # (num_active_simplices, num_vertices_per_simplex, n_dim)
        nsimplex_vertices = all_mesh_points[active_simplices_vertex_indices]

        n_dim = all_mesh_points.shape[1]

        if n_dim == 1:
            # For 1D simplices (line segments), volume is length
            # Vertices are (n_active, 2, 1)
            # v0 is (n_active, 1), v1 is (n_active, 1)
            v0 = nsimplex_vertices[:, 0, :]  # First vertex of each simplex
            v1 = nsimplex_vertices[:, 1, :]  # Second vertex of each simplex
            lengths = np.abs(v1 - v0).squeeze(axis=-1)
            return lengths

        # Select one vertex from each simplex as the origin (v0)
        # v0 will have shape (num_active_simplices, n_dim)
        v0 = nsimplex_vertices[:, 0, :]

        # Create vectors from v0 to all other vertices (v1-v0, v2-v0, ..., vn-v0)
        # These vectors form the rows (or columns) of the matrix for the determinant.
        # nsimplex_vertices[:, 1:, :] has shape (n_active, n_dim, n_dim)
        # v0[:, np.newaxis, :] broadcasts v0 to match for subtraction
        # matrix_for_determinant will have shape (n_active, n_dim, n_dim)
        matrix_for_determinant = nsimplex_vertices[:, 1:, :] - v0[:, np.newaxis, :]

        # Calculate the determinant for each simplex's matrix
        # np.linalg.det operates on the last two axes by default.
        determinants = np.linalg.det(matrix_for_determinant)

        # Volume = abs(determinant) / n!
        n_factorial = float(math.factorial(n_dim))
        volumes = np.abs(determinants) / n_factorial

        return volumes
