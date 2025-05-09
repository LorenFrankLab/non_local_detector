from __future__ import annotations

import itertools
import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, KDTree, Voronoi

from non_local_detector.environment.environment import (
    _create_1d_track_grid_data,
    _create_grid,
    _infer_track_interior,
    _make_nd_track_graph,
    get_centers,
)

try:
    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ---------------------------------------------------------------------------
# GeometryEngine protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class GeometryEngine(Protocol):
    """Contract all engines must satisfy (with *Environment* field names)."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: nx.Graph | None  # For N-D grid-like structures
    track_graph_bin_centers_: nx.Graph | None  # For 1-D or other graph structures

    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(self, **kwargs) -> None: ...
    def point_to_bin(self, pts: NDArray[np.float64]) -> NDArray[np.int_]: ...
    def neighbors(self, flat_idx: int) -> List[int]: ...


# ---------------------------------------------------------------------------
# KD-tree mixin implementing generic mapping helpers
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """Mixin providing *point_to_bin* and *neighbors* via KD-tree + NetworkX."""

    _kdtree: Optional[KDTree] = None
    _flat_indices_of_kdtree_nodes: Optional[NDArray[np.int_]] = None

    def _build_kdtree(self, interior_mask: Optional[NDArray[np.bool_]] = None) -> None:
        if not hasattr(self, "place_bin_centers_") or self.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ not set; call build() before _build_kdtree()"
            )

        if interior_mask is not None:
            # Validate mask shape against place_bin_centers_ and centers_shape_
            expected_mask_len = self.place_bin_centers_.shape[0]
            flat_mask = interior_mask.ravel()

            if len(flat_mask) != expected_mask_len:
                raise ValueError(
                    f"Flattened interior_mask length {len(flat_mask)} "
                    f"does not match place_bin_centers_ length {expected_mask_len}."
                )
            if interior_mask.shape != self.centers_shape_:
                # Allow 1D mask for 1D centers_shape even if place_bin_centers is (N,1)
                if not (
                    interior_mask.ndim == 1
                    and self.centers_shape_ == interior_mask.shape
                    and self.place_bin_centers_.shape[0] == interior_mask.shape[0]
                ):
                    raise ValueError(
                        f"Provided interior_mask shape {interior_mask.shape} "
                        f"does not match centers_shape_ {self.centers_shape_}."
                    )

            points_for_kdtree = self.place_bin_centers_[flat_mask]
            self._flat_indices_of_kdtree_nodes = np.where(flat_mask)[0].astype(np.int32)
        else:
            points_for_kdtree = self.place_bin_centers_
            self._flat_indices_of_kdtree_nodes = np.arange(
                self.place_bin_centers_.shape[0], dtype=np.int32
            )

        if points_for_kdtree.shape[0] > 0:
            self._kdtree = KDTree(points_for_kdtree)
        else:
            self._kdtree = None
            # Ensure _flat_indices_of_kdtree_nodes is empty if no points
            if (
                self._flat_indices_of_kdtree_nodes is not None
                and self._flat_indices_of_kdtree_nodes.size > 0
            ):
                self._flat_indices_of_kdtree_nodes = np.array([], dtype=np.int32)

    def point_to_bin(self, pts: NDArray[np.float64]) -> NDArray[np.int_]:  # type: ignore[override]
        if self._kdtree is None:
            return np.full(np.atleast_2d(pts).shape[0], -1, dtype=np.int32)

        if self._flat_indices_of_kdtree_nodes is None:
            raise RuntimeError("_flat_indices_of_kdtree_nodes not set.")

        _distances, indices_in_kdtree_subset = self._kdtree.query(np.atleast_2d(pts))

        if indices_in_kdtree_subset.size == 0:  # No points in KDTree query result
            return np.full(np.atleast_2d(pts).shape[0], -1, dtype=np.int32)

        original_flat_indices = self._flat_indices_of_kdtree_nodes[
            indices_in_kdtree_subset
        ]
        return original_flat_indices.astype(np.int32)

    def neighbors(self, flat_idx: int) -> List[int]:  # type: ignore[override]
        graph_to_use = None
        if hasattr(self, "track_graph_nd_") and self.track_graph_nd_ is not None:
            graph_to_use = self.track_graph_nd_
        elif (
            hasattr(self, "track_graph_bin_centers_")
            and self.track_graph_bin_centers_ is not None
        ):
            graph_to_use = self.track_graph_bin_centers_

        if graph_to_use is None:
            # Try to check if the other graph attribute exists but is None explicitly
            if hasattr(self, "track_graph_nd_") and hasattr(
                self, "track_graph_bin_centers_"
            ):
                # Both are defined on the class but are None
                raise ValueError(
                    "Both track_graph_nd_ and track_graph_bin_centers_ are None."
                )
            raise ValueError(
                "No suitable graph attribute (track_graph_nd_ or track_graph_bin_centers_) found or set."
            )

        if flat_idx not in graph_to_use:
            return []
        return list(graph_to_use.neighbors(flat_idx))


class RegularGridEngine(_KDTreeMixin):
    """Axis-aligned rectangular (N-D) grid engine.
    Can infer interior based on position data or define all bins as interior.
    """

    place_bin_centers_: NDArray[np.float64]
    place_bin_edges_: Optional[NDArray[np.float64]] = None
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        place_bin_size: Union[float, Sequence[float]],
        position_range: Optional[Sequence[Tuple[float, float]]] = None,
        position: Optional[NDArray[np.float64]] = None,
        add_boundary_bins: bool = False,
        infer_track_interior: bool = False,
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        bin_count_threshold: int = 0,
    ) -> None:
        (
            self.edges_,
            self.place_bin_edges_,
            self.place_bin_centers_,
            self.centers_shape_,
        ) = _create_grid(
            position=position,
            bin_size=place_bin_size,
            position_range=position_range,
            add_boundary_bins=add_boundary_bins,
        )

        if infer_track_interior:
            if position is None:
                raise ValueError(
                    "`position` data must be provided when `infer_track_interior` is True."
                )
            self.interior_mask_ = _infer_track_interior(
                position=position,
                edges=self.edges_,
                close_gaps=close_gaps,
                fill_holes=fill_holes,
                dilate=dilate,
                bin_count_threshold=bin_count_threshold,
                boundary_exists=add_boundary_bins,
            )
        else:
            self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        self.track_graph_nd_ = _make_nd_track_graph(
            self.place_bin_centers_, self.interior_mask_, self.centers_shape_
        )
        self._build_kdtree(interior_mask=self.interior_mask_)


class TrackGraphEngine(_KDTreeMixin):
    """1-D topological track engine."""

    place_bin_centers_: NDArray[np.float64]
    place_bin_edges_: Optional[NDArray[np.float64]] = None
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        track_graph: nx.Graph,
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        place_bin_size: float,
    ) -> None:
        (
            self.place_bin_centers_,
            self.place_bin_edges_,
            is_interior,
            self.centers_shape_,
            self.edges_,
            self.track_graph_bin_centers_,
        ) = _create_1d_track_grid_data(
            track_graph,
            edge_order,
            edge_spacing,
            place_bin_size,
        )
        self.interior_mask_ = is_interior
        self._build_kdtree(interior_mask=self.interior_mask_)


if SHAPELY_AVAILABLE:

    class ShapelyGridEngine(_KDTreeMixin):
        """Mask a regular 2D grid by a Shapely Polygon."""

        place_bin_centers_: NDArray[np.float64]
        edges_: Tuple[NDArray[np.float64], ...]
        centers_shape_: Tuple[int, ...]
        track_graph_nd_: Optional[nx.Graph] = None
        track_graph_bin_centers_: Optional[nx.Graph] = None
        interior_mask_: Optional[NDArray[np.bool_]] = None

        def build(
            self,
            *,
            polygon: Polygon,
            place_bin_size: Union[float, Sequence[float]],
            add_boundary_bins: bool = False,
        ) -> None:
            minx, miny, maxx, maxy = polygon.bounds
            pos_range: Sequence[Tuple[float, float]] = [(minx, maxx), (miny, maxy)]

            (
                _edges_tuple,
                _place_bin_edges_flat,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = _create_grid(
                position=None,
                bin_size=place_bin_size,
                position_range=pos_range,
                add_boundary_bins=add_boundary_bins,
            )
            self.edges_ = _edges_tuple

            mask_flat = np.array(
                [polygon.contains(Point(*p)) for p in self.place_bin_centers_]
            )
            self.interior_mask_ = mask_flat.reshape(self.centers_shape_)

            self.track_graph_nd_ = _make_nd_track_graph(
                self.place_bin_centers_, self.interior_mask_, self.centers_shape_
            )
            self._build_kdtree(interior_mask=self.interior_mask_)

else:
    ShapelyGridEngine = None


class MaskedGridEngine(_KDTreeMixin):
    """Build from existing boolean mask + edges."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        mask: NDArray[np.bool_],
        edges: Tuple[NDArray[np.float64], ...],
    ) -> None:
        self.edges_ = edges
        self.centers_shape_ = mask.shape
        self.interior_mask_ = mask

        centers_list = [get_centers(edge_dim) for edge_dim in self.edges_]
        if not all(len(cl) == s for cl, s in zip(centers_list, self.centers_shape_)):
            raise ValueError(
                "Shape mismatch between mask and centers derived from edges."
            )

        mesh = np.meshgrid(*centers_list, indexing="ij")
        self.place_bin_centers_ = np.stack([m.ravel() for m in mesh], axis=1)

        self.track_graph_nd_ = _make_nd_track_graph(
            self.place_bin_centers_, self.interior_mask_, self.centers_shape_
        )
        self._build_kdtree(interior_mask=self.interior_mask_)


class DelaunayGraphEngine(_KDTreeMixin):
    """Build graph via Delaunay triangulation of point cloud. All points are considered interior."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        points: NDArray[np.float64],
        max_edge_length: Optional[float] = None,
    ) -> None:
        if points.ndim != 2:
            raise ValueError("Input points must be a 2D array (n_points, n_dims).")
        self.place_bin_centers_ = np.asarray(points)
        n_points = self.place_bin_centers_.shape[0]
        if n_points == 0:
            raise ValueError("Input points array cannot be empty.")

        self.centers_shape_ = (n_points,)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        if points.shape[1] < 2:
            raise ValueError("Delaunay triangulation requires at least 2D points.")

        tri = Delaunay(self.place_bin_centers_)
        G = nx.Graph()
        G.add_nodes_from(range(n_points))

        for i in range(n_points):
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i

        for simplex in tri.simplices:
            for i, j in itertools.combinations(simplex, 2):
                d = np.linalg.norm(
                    self.place_bin_centers_[i] - self.place_bin_centers_[j]
                )
                if max_edge_length is None or d <= max_edge_length:
                    G.add_edge(i, j, distance=d)

        for eid, (u, v) in enumerate(G.edges()):
            G.edges[u, v]["edge_id"] = eid

        self.track_graph_nd_ = G  # For Delaunay, nd_graph is the primary graph
        self.track_graph_bin_centers_ = G  # Alias for consistency
        self._build_kdtree(interior_mask=self.interior_mask_)


class HexagonalGridEngine(_KDTreeMixin):
    """Tiles a 2D rectangle into a hexagonal lattice."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        place_bin_size: float,
        position_range: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> None:
        (xmin, xmax), (ymin, ymax) = position_range
        dx = place_bin_size
        dy = math.sqrt(3) * dx / 2
        centers_list: List[Tuple[float, float]] = []

        for row_idx in range(
            int(math.ceil((ymax - ymin) / dy)) + 2
        ):  # +2 for robust coverage
            y = ymin + row_idx * dy
            if y > ymax + dy and row_idx > 0:  # Allow one row slightly beyond
                break
            current_xmin = (
                xmin - (dx / 4) + (dx / 2) if (row_idx % 2) else xmin - (dx / 4)
            )  # Shift for denser packing
            for col_idx in range(
                int(math.ceil((xmax - current_xmin) / dx)) + 2
            ):  # +2 for robust coverage
                x = current_xmin + col_idx * dx
                if x > xmax + dx and col_idx > 0:  # Allow one col slightly beyond
                    break
                # Add point if its center is within or very close to the bounding box
                if (xmin - dx) <= x <= (xmax + dx) and (ymin - dy) <= y <= (ymax + dy):
                    centers_list.append((x, y))

        # Filter to keep only points whose centers are strictly within the original range
        temp_centers = np.array(centers_list)
        if temp_centers.size > 0:
            valid_mask = (
                (temp_centers[:, 0] >= xmin)
                & (temp_centers[:, 0] <= xmax)
                & (temp_centers[:, 1] >= ymin)
                & (temp_centers[:, 1] <= ymax)
            )
            self.place_bin_centers_ = temp_centers[valid_mask]
        else:
            self.place_bin_centers_ = np.empty((0, 2))

        self.centers_shape_ = (self.place_bin_centers_.shape[0],)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        G = nx.Graph()
        G.add_nodes_from(range(len(self.place_bin_centers_)))
        for i in range(len(self.place_bin_centers_)):
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i

        if self.place_bin_centers_.shape[0] > 1:
            tree = KDTree(self.place_bin_centers_)
            pairs = tree.query_pairs(r=dx * 1.1)  # Connect to 6 nearest for hex
            for i, j in pairs:
                dist = np.linalg.norm(
                    self.place_bin_centers_[i] - self.place_bin_centers_[j]
                )
                G.add_edge(i, j, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)


class QuadtreeGridEngine(_KDTreeMixin):
    """Adaptive quadtree tiling of 2D space to a maximum depth.
    Bin centers are the centers of the quadtree cells.
    """

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    # Store cell boundaries for potential visualization or other uses
    quadtree_cells_: Optional[List[Tuple[float, float, float, float]]] = None

    def build(
        self,
        *,
        position_range: Tuple[Tuple[float, float], Tuple[float, float]],
        max_depth: int = 4,
        # Optional: points to guide subdivision (not implemented here, but a common extension)
        # points_to_subdivide_around: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")

        (xmin, xmax), (ymin, ymax) = position_range
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "position_range min must be less than max for both dimensions."
            )

        leaf_nodes_centers: List[Tuple[float, float]] = []
        leaf_nodes_bounds: List[Tuple[float, float, float, float]] = (
            []
        )  # Store x0,y0,x1,y1

        # Recursive subdivision function
        def subdivide(x0, y0, x1, y1, current_depth):
            if current_depth == max_depth:
                # At max depth, this cell is a leaf node
                leaf_nodes_centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
                leaf_nodes_bounds.append((x0, y0, x1, y1))
                return

            # Subdivide into four children
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            subdivide(x0, y0, mx, my, current_depth + 1)  # Bottom-left
            subdivide(mx, y0, x1, my, current_depth + 1)  # Bottom-right
            subdivide(x0, my, mx, y1, current_depth + 1)  # Top-left
            subdivide(mx, my, x1, y1, current_depth + 1)  # Top-right

        subdivide(xmin, ymin, xmax, ymax, 0)

        self.place_bin_centers_ = np.array(leaf_nodes_centers)
        self.quadtree_cells_ = leaf_nodes_bounds  # Store cell boundaries

        if (
            self.place_bin_centers_.shape[0] == 0
        ):  # Should not happen with max_depth >=0
            self.centers_shape_ = (0,)
            self.interior_mask_ = np.array([], dtype=bool)
            self.track_graph_bin_centers_ = nx.Graph()
        else:
            self.centers_shape_ = (self.place_bin_centers_.shape[0],)
            self.interior_mask_ = np.ones(
                self.centers_shape_, dtype=bool
            )  # All leaf cells are "interior"

            G = nx.Graph()
            G.add_nodes_from(range(len(self.place_bin_centers_)))
            for i in range(len(self.place_bin_centers_)):
                G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
                G.nodes[i]["is_track_interior"] = True
                G.nodes[i]["bin_ind"] = (i,)
                G.nodes[i]["bin_ind_flat"] = i
                # Store cell bounds with node
                G.nodes[i]["bounds"] = self.quadtree_cells_[i]

            # Build adjacency: leaf cells are adjacent if their boundaries touch.
            # A simpler proxy for quadtrees is connecting centers if they are "close enough"
            # relative to the smallest cell size.
            min_cell_width = (xmax - xmin) / (2**max_depth)
            min_cell_height = (ymax - ymin) / (2**max_depth)
            # Connect if centers are within ~1.5 times the smallest cell dimension
            # This is an approximation for adjacency. True quadtree adjacency is more complex.
            connect_radius = 1.5 * max(min_cell_width, min_cell_height)

            if self.place_bin_centers_.shape[0] > 1:
                tree = KDTree(self.place_bin_centers_)
                pairs = tree.query_pairs(r=connect_radius)
                for i, j in pairs:
                    dist = np.linalg.norm(
                        self.place_bin_centers_[i] - self.place_bin_centers_[j]
                    )
                    G.add_edge(i, j, distance=dist)
            self.track_graph_bin_centers_ = G

        self._build_kdtree(interior_mask=self.interior_mask_)


class VoronoiPartitionEngine(_KDTreeMixin):
    """Partitions N-D space via Voronoi tessellation of seed points.
    Bin centers are the centroids of the finite Voronoi regions.
    """

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    voronoi_diagram_: Optional[Voronoi] = None  # Store the Voronoi object

    def build(self, *, seeds: NDArray[np.float64]) -> None:
        if seeds.ndim != 2:
            raise ValueError("Input seeds must be a 2D array (n_seeds, n_dims).")
        if (
            seeds.shape[0] < seeds.shape[1] + 1
        ):  # Voronoi condition for non-degenerate cells
            raise ValueError(
                f"Need at least {seeds.shape[1]+1} seed points for {seeds.shape[1]}-D Voronoi."
            )

        self.voronoi_diagram_ = Voronoi(seeds)

        finite_region_centroids: List[NDArray[np.float64]] = []
        finite_region_indices: List[int] = (
            []
        )  # Original indices of points whose regions are finite

        for i, region_idx in enumerate(self.voronoi_diagram_.point_region):
            region_vertices_indices = self.voronoi_diagram_.regions[region_idx]
            if not region_vertices_indices or -1 in region_vertices_indices:
                continue  # Skip infinite regions or empty regions

            region_vertices = self.voronoi_diagram_.vertices[region_vertices_indices]
            finite_region_centroids.append(np.mean(region_vertices, axis=0))
            finite_region_indices.append(i)  # Store original seed index

        if not finite_region_centroids:
            self.place_bin_centers_ = np.empty((0, seeds.shape[1]))
        else:
            self.place_bin_centers_ = np.array(finite_region_centroids)

        self.centers_shape_ = (self.place_bin_centers_.shape[0],)
        self.interior_mask_ = np.ones(
            self.centers_shape_, dtype=bool
        )  # All finite centroids are "interior"

        G = nx.Graph()
        # Nodes in the graph will be 0 to N-1, where N is number of finite regions.
        # We need a mapping from original seed index to this new graph node ID.
        seed_idx_to_graph_node_id: Dict[int, int] = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(finite_region_indices)
        }

        G.add_nodes_from(range(self.place_bin_centers_.shape[0]))
        for i in range(self.place_bin_centers_.shape[0]):
            original_seed_idx = finite_region_indices[i]
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i
            G.nodes[i]["original_seed_index"] = original_seed_idx  # Store mapping

        # Build adjacency: regions sharing a ridge (edge in Voronoi diagram)
        for ridge_point_indices, ridge_vertex_indices in zip(
            self.voronoi_diagram_.ridge_points, self.voronoi_diagram_.ridge_vertices
        ):
            # ridge_point_indices are indices into the original 'seeds' array
            p1_original_idx, p2_original_idx = ridge_point_indices

            # Check if both points correspond to finite regions we've kept
            if (
                p1_original_idx in seed_idx_to_graph_node_id
                and p2_original_idx in seed_idx_to_graph_node_id
            ):
                node_u = seed_idx_to_graph_node_id[p1_original_idx]
                node_v = seed_idx_to_graph_node_id[p2_original_idx]

                # Calculate distance between centroids of these two regions
                dist = np.linalg.norm(
                    self.place_bin_centers_[node_u] - self.place_bin_centers_[node_v]
                )
                G.add_edge(node_u, node_v, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)


class MeshSurfaceEngine(_KDTreeMixin):
    """Uses an existing triangular mesh: vertices as bins, edges as adjacency."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(self, *, vertices: NDArray[np.float64], faces: NDArray[np.int_]) -> None:
        if vertices.ndim != 2:
            raise ValueError("Vertices must be a 2D array (n_vertices, n_dims).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces must be a 2D array (n_faces, 3) of vertex indices.")

        self.place_bin_centers_ = vertices.copy()
        self.centers_shape_ = (vertices.shape[0],)
        self.interior_mask_ = np.ones(
            self.centers_shape_, dtype=bool
        )  # All vertices are "interior"

        G = nx.Graph()
        G.add_nodes_from(range(len(vertices)))
        for i in range(len(vertices)):
            G.nodes[i]["pos"] = tuple(vertices[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i

        for tri_face in faces:
            # Ensure indices are within bounds
            if np.any(tri_face >= len(vertices)) or np.any(tri_face < 0):
                raise ValueError("Face indices out of bounds for vertices.")
            i, j, k = tri_face
            for u, v in ((i, j), (j, k), (k, i)):
                if not G.has_edge(u, v):  # Add edge only if it doesn't exist
                    dist = np.linalg.norm(vertices[u] - vertices[v])
                    G.add_edge(u, v, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)


class ImageMaskEngine(_KDTreeMixin):
    """Converts a 2D boolean image mask into pixel-center bins and a graph.
    Connects 4-neighbors or 8-neighbors.
    """

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    pixel_to_node_map_: Optional[Dict[Tuple[int, int], int]] = (
        None  # Map (row,col) to flat_idx
    )

    def build(self, *, mask: NDArray[np.bool_], connect_diagonal: bool = False) -> None:
        if mask.ndim != 2:
            raise ValueError("ImageMaskEngine currently only supports 2D masks.")

        # Pixel coordinates (row, col) of True values in the mask
        row_coords, col_coords = np.nonzero(mask)
        # Bin centers are pixel centers (col_coord + 0.5, row_coord + 0.5)
        # Using (x,y) convention, so (col, row)
        self.place_bin_centers_ = np.stack((col_coords + 0.5, row_coords + 0.5), axis=1)

        num_interior_pixels = self.place_bin_centers_.shape[0]
        self.centers_shape_ = (num_interior_pixels,)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        G = nx.Graph()
        G.add_nodes_from(range(num_interior_pixels))
        self.pixel_to_node_map_ = {}

        for i in range(num_interior_pixels):
            r, c = row_coords[i], col_coords[i]
            self.pixel_to_node_map_[(r, c)] = i
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)  # Flat index as tuple for consistency
            G.nodes[i]["bin_ind_flat"] = i
            G.nodes[i]["pixel_coord"] = (r, c)  # Store original pixel coord

        # Build adjacency graph by checking neighbors in the mask
        for i in range(num_interior_pixels):
            r1, c1 = row_coords[i], col_coords[i]

            # Define offsets for neighbors
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity
            if connect_diagonal:
                offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # 8-connectivity

            for dr, dc in offsets:
                r2, c2 = r1 + dr, c1 + dc
                # Check if neighbor is within mask bounds and is True
                if 0 <= r2 < mask.shape[0] and 0 <= c2 < mask.shape[1] and mask[r2, c2]:

                    j = self.pixel_to_node_map_[(r2, c2)]  # Get flat index of neighbor
                    if i < j:  # Add edge only once
                        dist = np.linalg.norm(
                            self.place_bin_centers_[i] - self.place_bin_centers_[j]
                        )
                        G.add_edge(i, j, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
_ENGINE_MAP: Dict[str, type[GeometryEngine]] = {
    "Grid": RegularGridEngine,
    "RegularGrid": RegularGridEngine,
    "1D": TrackGraphEngine,
    "TrackGraph": TrackGraphEngine,
    "Mask": MaskedGridEngine,
    "MaskedGrid": MaskedGridEngine,
    "Delaunay": DelaunayGraphEngine,
    "Hexagonal": HexagonalGridEngine,
    "Quadtree": QuadtreeGridEngine,
    "Voronoi": VoronoiPartitionEngine,
    "Mesh": MeshSurfaceEngine,
    "ImageMask": ImageMaskEngine,
}

if SHAPELY_AVAILABLE and ShapelyGridEngine is not None:
    _ENGINE_MAP["Shapely"] = ShapelyGridEngine
    _ENGINE_MAP["ShapelyGrid"] = ShapelyGridEngine


def make_engine(kind: str, **kwargs) -> GeometryEngine:
    """Instantiate and build an engine by *kind* string.

    Examples
    --------
    >>> eng = make_engine("Grid", place_bin_size=2.0, position_range=[(0,50),(0,50)])
    >>> eng.place_bin_centers_.shape # Example output
    (625, 2)
    >>> hex_eng = make_engine("Hexagonal", place_bin_size=5.0, position_range=[(0,20),(0,20)])
    >>> hex_eng.place_bin_centers_.shape # Example output
    (12, 2)
    """
    normalized_kind = "".join(filter(str.isalnum, kind)).lower()

    found_key = None
    for k_map in _ENGINE_MAP.keys():
        if "".join(filter(str.isalnum, k_map)).lower() == normalized_kind:
            found_key = k_map
            break

    if found_key is None:
        raise ValueError(
            f"Unknown engine kind '{kind}'. Available: {list(_ENGINE_MAP.keys())}"
        )

    eng_cls = _ENGINE_MAP[found_key]
    eng = eng_cls()
    eng.build(**kwargs)
    return eng
