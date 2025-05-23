"""
Defines the spatial layout engines for discretizing continuous space.

This module provides the `LayoutEngine` protocol, which outlines the interface
for all spatial discretization strategies. It includes concrete implementations
such as:
- `RegularGridLayout`: For N-dimensional rectilinear grids.
- `HexagonalLayout`: For 2D hexagonal tilings.
- `GraphLayout`: For layouts defined by user-provided graphs, often used for
  linearized tracks.
- `ShapelyPolygonLayout`: For 2D grid layouts masked by a Shapely polygon.
- `MaskedGridLayout`: For N-D grids defined by an explicit mask and edge definitions.
- `ImageMaskLayout`: For 2D layouts derived from a boolean image mask.

It also contains mixin classes (`_KDTreeMixin`, `_GridMixin`) to provide common
functionality to layout implementations, and factory helper functions
(`create_layout`, `list_available_layouts`, `get_layout_parameters`) for
instantiating and inspecting layout engines. These engines are fundamental
to the `Environment` class, defining its geometry and connectivity.
"""

from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
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

import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray
from scipy.spatial import KDTree
from track_linearization import get_linearized_position as _get_linearized_position
from track_linearization import plot_graph_as_1D, plot_track_graph

from non_local_detector.environment.graph_utils import (
    _create_graph_layout_connectivity_graph,
    _find_bin_for_linear_position,
    _get_graph_bins,
    _project_1d_to_2d,
)
from non_local_detector.environment.hex_grid_utils import (
    _create_hex_connectivity_graph,
    _create_hex_grid,
    _infer_active_bins_from_hex_grid,
    _points_to_hex_bin_ind,
)
from non_local_detector.environment.regular_grid_utils import (
    _create_regular_grid,
    _create_regular_grid_connectivity_graph,
    _infer_active_bins_from_regular_grid,
    _points_to_regular_grid_bin_ind,
    get_centers,
)
from non_local_detector.environment.utils import (
    _generic_graph_plot,
    _infer_dimension_ranges_from_samples,
)

try:
    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Polygon = None  # type: ignore
    Point = None  # type: ignore


PolygonType = type[Polygon]


# --------------------------
# LayoutEngine protocol
# --------------------------
@runtime_checkable
class LayoutEngine(Protocol):
    """
    Protocol defining the interface for all spatial layout engines.

    A LayoutEngine is responsible for discretizing a continuous N-dimensional
    space into a set of bins or elements and constructing a graph representation
    of their connectivity.

    Attributes
    ----------
    bin_centers_ : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node.
        Shape is (n_active_bins, n_dims).
    connectivity_graph_ : Optional[nx.Graph]
        Graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to rows in `bin_centers_`.
        **Mandatory Node Attributes**:
            - 'pos': Tuple[float, ...] - N-D coordinates of the active bin center.
            - 'source_grid_flat_index': int - Flat index in the original
              full conceptual grid from which this active bin originated.
            - 'original_grid_nd_index': Tuple[int, ...] - N-D tuple index
              in the original full conceptual grid.
        **Mandatory Edge Attributes**:
            - 'distance': float - Euclidean distance between connected bin centers.
            - 'weight': float - Cost for pathfinding, often equals 'distance'.
        **Recommended Edge Attributes**:
            - 'vector': Tuple[float, ...] - Displacement vector between centers.
            - 'angle_2d': Optional[float] - Angle of displacement for 2D layouts.
            - 'edge_id': int - Unique ID for the edge within this graph.
    is_1d : bool
        True if the layout represents a primarily 1-dimensional structure
        (e.g., a linearized track), False otherwise.
    dimension_ranges_ : Optional[Sequence[Tuple[float, float]]]
        The actual min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry.
    grid_edges_ : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts: A tuple of 1D arrays, where each array
        contains the bin edge positions for one dimension of the *original,
        full grid*. `None` or `()` for non-grid or point-based layouts.
    grid_shape_ : Optional[Tuple[int, ...]]
        For grid-based layouts: The N-D shape (number of bins in each
        dimension) of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept:
        Typically `(n_active_bins,)`.
    active_mask_ : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid* (shape matches `grid_shape_`).
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers_`.
    _layout_type_tag : str
        A string identifier for the type of layout (e.g., "RegularGrid").
        Used for introspection and serialization.
    _build_params_used : Dict[str, Any]
        A dictionary of the parameters used to construct this layout instance.
        Used for introspection and serialization.

    """

    # --- Required Data Attributes ---
    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    # Attributes primarily for GRID-BASED Layouts
    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    # Internal Attributes for Introspection/Serialization
    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # --- Required Methods ---
    def build(self, **kwargs) -> None:
        """
        Construct the layout's geometry, bins, and connectivity graph.

        This method is responsible for populating all the attributes defined
        in the `LayoutEngine` protocol (e.g., `bin_centers_`,
        `connectivity_graph_`, etc.) based on the provided keyword arguments.
        The specific arguments required will vary depending on the concrete
        implementation of the layout engine.
        """
        ...

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map continuous N-D points to discrete active bin indices.

        The returned indices range from `0` to `n_active_bins - 1`.
        A value of -1 indicates that the corresponding point did not map
        to any active bin (e.g., it's outside the defined environment).

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map to bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            An array of active bin indices corresponding to the input points.
            -1 for points outside the layout's active bins.
        """
        ...

    def get_bin_neighbors(self, bin_index: int) -> List[int]:
        """
        Find indices of neighboring active bins for a given active bin index.

        This typically uses the `connectivity_graph_`. The input `bin_index`
        and returned indices are relative to the active bins (0 to N-1).

        Parameters
        ----------
        bin_index : int
            The index (0 to `n_active_bins - 1`) of the active bin for which
            to find neighbors.

        Returns
        -------
        List[int]
            A list of active bin indices that are neighbors to `bin_index`.
        """
        ...

    @property
    @abstractmethod
    def is_1d(self) -> bool:
        """
        Indicate if the layout structure is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the layout represents a 1D structure (e.g., a linearized
            track), False otherwise.
        """
        ...

    @abstractmethod
    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the layout's geometry.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        **kwargs : Any
            Additional keyword arguments for plot customization, specific to
            the layout engine implementation.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        ...

    @abstractmethod
    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """
        Return the area (2D) or volume (3D+) of each active bin.

        For 1D layouts, this typically returns the length of each bin.
        The measure should correspond to the dimensionality of the space
        the bins occupy.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array where each element is the area/volume/length of the
            corresponding active bin.
        """
        ...


# ---------------------------------------------------------------------------
# KD-tree mixin (for point_to_bin_index)
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """
    Mixin providing point-to-bin mapping and neighbor finding using a KD-tree.

    This mixin uses a KD-tree built on bin centers for nearest-neighbor
    searches (`point_to_bin_index`) and a NetworkX graph for connectivity
    to find neighbors (`get_bin_neighbors`).

    Assumes the inheriting class defines:
    - `self.bin_centers_`: NDArray of active bin center coordinates.
    - `self.connectivity_graph_`: NetworkX graph of active bins.

    The `_build_kdtree` method must be called by the inheriting layout's
    `build` method after `bin_centers_` is finalized.

    Attributes:
    ----------
    _kdtree : Optional[KDTree]
        KDTree for fast nearest-neighbor search.
    _kdtree_nodes_to_bin_indices_map : Optional[NDArray[np.int_]], shape (n_active_bins,)
        Maps KDTree node index to original source_index.
    """

    _kdtree: Optional[KDTree] = None
    _kdtree_nodes_to_bin_indices_map: Optional[NDArray[np.int_]] = None

    def _build_kdtree(
        self,
        points_for_tree: NDArray[np.float64],
        mask_for_points_in_tree: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        """
        Builds the KD-tree from `points_for_tree`.

        If `mask_for_points_in_tree` is provided, it's used to select a subset
        of `points_for_tree` to include in the KD-tree. The
        `_kdtree_nodes_to_bin_indices_map` will then map the KD-tree's
        internal node indices (0 to k-1) back to the original indices within
        `points_for_tree` that were selected by the mask.

        If `points_for_tree` are the layout's final `bin_centers_` (which are
        all active), `mask_for_points_in_tree` should be `None` or all `True`.
        In this common case, `_kdtree_nodes_to_bin_indices_map` will effectively
        be `np.arange(len(points_for_tree))`.

        Parameters
        ----------
        points_for_tree : NDArray[np.float64], shape (n_total_points, n_dims)
            The set of points from which to build the KD-tree. Typically, these
            are `self.bin_centers_` of the layout.
        mask_for_points_in_tree : Optional[NDArray[np.bool_]], shape (n_total_points,), optional
            A boolean mask indicating which points from `points_for_tree`
            to include in the KD-tree. If None, all points are used.
            Defaults to None.

        Raises
        ------
        ValueError
            If `mask_for_points_in_tree` has an incompatible shape.
        """
        if points_for_tree.ndim != 2 or points_for_tree.shape[0] == 0:
            self._kdtree = None
            self._kdtree_nodes_to_bin_indices_map = np.array([], dtype=np.int32)
            if points_for_tree.shape[0] == 0 and points_for_tree.ndim == 2:
                # no points to build tree from
                return
            else:
                raise ValueError(
                    "points_for_tree must be a 2D array with shape (n_points, n_dims)."
                )

        final_points_for_kdtree_construction: NDArray[np.float64]

        # This map will store the indices from `points_for_tree` that correspond
        # to the nodes (0 to k-1) in the `self._kdtree` object.
        source_indices_of_kdtree_nodes: NDArray[np.int_]

        if mask_for_points_in_tree is not None:
            if (
                mask_for_points_in_tree.ndim != 1
                or mask_for_points_in_tree.shape[0] != points_for_tree.shape[0]
            ):
                raise ValueError(
                    "mask_for_points_in_tree must be 1D and match the "
                    "number of rows in points_for_tree."
                )
            final_points_for_kdtree_construction = points_for_tree[
                mask_for_points_in_tree
            ]
            source_indices_of_kdtree_nodes = np.flatnonzero(
                mask_for_points_in_tree
            ).astype(np.int32)
        else:
            # No mask provided, use all points from points_for_tree
            final_points_for_kdtree_construction = points_for_tree
            source_indices_of_kdtree_nodes = np.arange(
                points_for_tree.shape[0], dtype=np.int32
            )

        if final_points_for_kdtree_construction.shape[0] > 0:
            try:
                self._kdtree = KDTree(final_points_for_kdtree_construction)
                self._kdtree_nodes_to_bin_indices_map = source_indices_of_kdtree_nodes
            except (
                Exception
            ) as e:  # Catch potential errors from KDTree (e.g. QhullError for certain inputs)
                warnings.warn(
                    f"KDTree construction failed: {e}. point_to_bin_index may not work.",
                    RuntimeWarning,
                )
                self._kdtree = None
                self._kdtree_nodes_to_bin_indices_map = np.array([], dtype=np.int32)
        else:  # No points to build the tree (either initially or after masking)
            self._kdtree = None
            self._kdtree_nodes_to_bin_indices_map = np.array([], dtype=np.int32)

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map N-D points to active bin indices using nearest-neighbor search.

        Finds the nearest active bin center in `self.bin_centers_` (on which
        the KD-tree was built) to each query point.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_query_points, n_dims)
            N-D array of query points.

        Returns
        -------
        NDArray[np.int_], shape (n_query_points,)
            Array of active bin indices (0 to `n_active_bins - 1`).
            Returns -1 for all points if the KD-tree is not built or is empty.
        """
        query_points = np.atleast_2d(points)
        n_query_points = query_points.shape[0]

        if (
            self._kdtree is None
            or self._kdtree_nodes_to_bin_indices_map is None
            or self._kdtree_nodes_to_bin_indices_map.size == 0
        ):
            # This means no valid points were used to build the KD-tree.
            return np.full(n_query_points, -1, dtype=np.int32)

        try:
            _, kdtree_internal_indices = self._kdtree.query(query_points)
        except Exception as e:
            # Catch errors if query points have wrong dimension etc.
            warnings.warn(
                f"KDTree query failed: {e}. Returning -1 for all points.",
                RuntimeWarning,
            )
            return np.full(n_query_points, -1, dtype=np.int32)

        # kdtree_internal_indices are indices into the array used to build the KDTree
        # (i.e., final_points_for_kdtree_construction in _build_kdtree).
        # We need to ensure these indices are valid for accessing _kdtree_nodes_to_bin_indices_map.
        # The size of _kdtree.data is len(final_points_for_kdtree_construction).
        max_valid_kdtree_idx = self._kdtree.data.shape[0] - 1  # type: ignore

        # Clip indices to be within the valid range of the KD-tree's internal point list
        # This handles cases where KDTree might return an out-of-bounds index if empty or single point.
        kdtree_internal_indices = np.clip(
            kdtree_internal_indices, 0, max_valid_kdtree_idx
        )

        # Map these internal KD-tree indices back to the original bin indices
        # (which are 0 to N-1, corresponding to rows in self.bin_centers_)
        # using the map created in _build_kdtree.
        final_bin_indices = self._kdtree_nodes_to_bin_indices_map[
            kdtree_internal_indices
        ]

        return final_bin_indices.astype(np.int32)

    def get_bin_neighbors(self, bin_index: int) -> List[int]:
        """
        Find indices of neighboring active bins for a given active bin index.

        Uses the `connectivity_graph_` which should have nodes `0` to
        `n_active_bins - 1`.

        Parameters
        ----------
        bin_index : int
            Index (0 to `n_active_bins - 1`) of the active bin for which
            to find neighbors.

        Returns
        -------
        List[int]
            List of indices of neighboring active bins.

        Raises
        ------
        AttributeError
            If `connectivity_graph_` is not defined on the instance.
        ValueError
            If `connectivity_graph_` is None (e.g., not built).
        """
        if not hasattr(self, "connectivity_graph_"):
            raise AttributeError(
                f"{self.__class__.__name__} does not have 'connectivity_graph_' attribute."
            )

        graph_to_use: Optional[nx.Graph] = getattr(self, "connectivity_graph_", None)

        if graph_to_use is None:
            raise ValueError(
                f"{self.__class__.__name__}.connectivity_graph_ is None. "
                "Ensure the layout is built."
            )

        if bin_index not in graph_to_use:
            # This could happen if bin_index is out of range for the number of active bins
            # or if the graph was unexpectedly empty or misconfigured.
            warnings.warn(
                f"Bin index {bin_index} not found in connectivity_graph_. Returning no neighbors.",
                RuntimeWarning,
            )
            return []

        return list(graph_to_use.neighbors(bin_index))


class _GridMixin:
    """
    Mixin for grid-based layout engines (e.g., RegularGrid, ShapelyPolygon).

    Provides common functionality for layouts that are based on an underlying
    N-dimensional grid, such as `point_to_bin_index` using grid definitions,
    default plotting, and `get_bin_area_volume` for uniform grids.

    Assumes the inheriting class defines grid-specific attributes like
    `grid_edges_`, `grid_shape_`, `active_mask_`, `bin_centers_`, and
    `connectivity_graph_`.
    """

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None
    bin_centers_: Optional[NDArray[np.float64]] = None
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None
    _layout_type_tag: str = "_Grid_Layout"

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map N-D points to active bin indices based on grid structure.

        Uses the grid's `grid_edges_`, `grid_shape_`, and `active_mask_`
        to determine the corresponding active bin for each point.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, n_dims)
            N-D points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Active bin indices (0 to N-1). -1 for points outside active areas.

        Raises
        ------
        RuntimeError
            If grid attributes (`grid_edges_`, `grid_shape_`) are not set.
        """
        if self.grid_edges_ is None or self.grid_shape_ is None:
            raise RuntimeError("Grid layout not built; edges or shape missing.")

        return _points_to_regular_grid_bin_ind(
            points=points,
            grid_edges=self.grid_edges_,
            grid_shape=self.grid_shape_,
            active_mask=self.active_mask_,
        )

    def get_bin_neighbors(self, bin_index: int) -> List[int]:
        """
        Find indices of neighboring active bins using the connectivity graph.

        Parameters
        ----------
        bin_index : int
            Index (0 to `n_active_bins - 1`) of the active bin.

        Returns
        -------
        List[int]
            List of neighboring active bin indices.

        Raises
        ------
        ValueError
            If `connectivity_graph_` is None.
        """
        if not hasattr(self, "connectivity_graph_"):
            raise AttributeError(
                f"{self.__class__.__name__} does not have 'connectivity_graph_' attribute."
            )

        graph_to_use: Optional[nx.Graph] = getattr(self, "connectivity_graph_", None)

        if graph_to_use is None:
            raise ValueError(
                f"{self.__class__.__name__}.connectivity_graph_ is None. "
                "Ensure the layout is built."
            )

        if bin_index not in graph_to_use:
            # This could happen if bin_index is out of range for the number of active bins
            # or if the graph was unexpectedly empty or misconfigured.
            warnings.warn(
                f"Bin index {bin_index} not found in connectivity_graph_. Returning no neighbors.",
                RuntimeWarning,
            )
            return []

        return list(graph_to_use.neighbors(bin_index))

    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        figsize=(7, 7),
        cmap: str = "bone_r",
        alpha: float = 0.7,
        draw_connectivity_graph: bool = True,
        node_size: float = 20,
        node_color: str = "blue",
    ) -> matplotlib.axes.Axes:
        """
        Plot the grid-based layout.

        For 2D grids, displays the `active_mask_` using `pcolormesh` and
        optionally overlays the `connectivity_graph_`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        figsize : Tuple[float, float], default=(7, 7)
            Size of the figure if a new one is created.
        cmap : str, default="bone_r"
            Colormap for the `active_mask_` plot.
        alpha : float, default=0.7
            Transparency for the `active_mask_` plot.
        draw_connectivity_graph : bool, default=True
            If True, draw the connectivity graph nodes and edges.
        node_size : float, default=20
            Size of nodes if `draw_connectivity_graph` is True.
        node_color : str, default="blue"
            Color of nodes if `draw_connectivity_graph` is True.
        **kwargs : Any
            Additional keyword arguments passed to `ax.pcolormesh` or
            NetworkX drawing functions.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout was plotted.

        Raises
        ------
        RuntimeError
            If layout attributes are not built.
        NotImplementedError
            If attempting to plot a non-2D grid layout with this method.
        """
        if (
            self.bin_centers_ is None
            or self.grid_edges_ is None
            or self.active_mask_ is None
            or self.grid_shape_ is None
            or self.connectivity_graph_ is None
        ):
            raise RuntimeError("Layout not built. Call `build` first.")

        is_2d_grid = len(self.grid_shape_) == 2 and len(self.grid_edges_) == 2

        if is_2d_grid:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.pcolormesh(
                self.grid_edges_[0],
                self.grid_edges_[1],
                self.active_mask_.T,
                cmap=cmap,
                alpha=alpha,
                shading="auto",
            )
            ax.set_xticks(self.grid_edges_[0])
            ax.set_yticks(self.grid_edges_[1])
            ax.grid(True, ls="-", lw=0.5, c="gray")
            ax.set_aspect("equal")
            ax.set_title(f"{self._layout_type_tag} (2D Grid)")
            ax.set_xlabel("Dimension 0")
            ax.set_ylabel("Dimension 1")
            if self.dimension_ranges_:
                ax.set_xlim(self.dimension_ranges_[0])
                ax.set_ylim(self.dimension_ranges_[1])

            if draw_connectivity_graph:
                node_position = nx.get_node_attributes(self.connectivity_graph_, "pos")
                nx.draw_networkx_nodes(
                    self.connectivity_graph_,
                    node_position,
                    ax=ax,
                    node_size=node_size,
                    node_color=node_color,
                )
                for node_id1, node_id2 in self.connectivity_graph_.edges:
                    pos = np.stack((node_position[node_id1], node_position[node_id2]))
                    ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

            return ax
        else:
            raise NotImplementedError(
                "Plotting for non-2D grid layouts is not implemented yet."
            )

    @property
    def is_1d(self) -> bool:
        """
        Indicate if the grid layout is 1-dimensional.

        Standard grid layouts (RegularGrid, etc.) are generally N-D (N>=1).
        This mixin's default assumes not strictly 1D in the sense of a
        linearized track (which GraphLayout would handle).

        Returns
        -------
        bool
            False, as this mixin is for general N-D grids.
        """
        return False

    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """
        Calculate area/volume for each active bin, assuming a uniform grid.

        Computes the product of bin side lengths for each dimension from
        `grid_edges_`. Assumes all bins in the grid have the same dimensions.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area/volume for each active bin.

        Raises
        ------
        RuntimeError
            If `grid_edges_` or `bin_centers_` is not populated.
        """
        if self.grid_edges_ is None or self.bin_centers_ is None:  # pragma: no cover
            raise RuntimeError("Layout not built; grid_edges_ or bin_centers_ missing.")
        if not self.grid_edges_ or not all(
            len(e) > 1 for e in self.grid_edges_
        ):  # pragma: no cover
            raise ValueError(
                "grid_edges_ are not properly defined for area/volume calculation."
            )

        # Assume uniform bin sizes from the first diff of each dimension's edges
        bin_dimension_sizes = np.array(
            [np.diff(edge_dim)[0] for edge_dim in self.grid_edges_]
        )
        single_bin_measure = np.prod(bin_dimension_sizes)

        return np.full(self.bin_centers_.shape[0], single_bin_measure)


# ---------------------------------------------------------------------------
# Specific LayoutEngine Implementations
# ---------------------------------------------------------------------------


class RegularGridLayout(_GridMixin):
    """
    Axis-aligned rectangular N-D grid layout.

    Discretizes space into a uniform N-dimensional grid. Can infer the
    active portion of this grid based on provided data samples using occupancy
    and morphological operations. Inherits grid-based functionalities from
    `_GridMixin`.
    """

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None
    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    def __init__(self):
        """Initialize a RegularGridLayout engine."""
        self._layout_type_tag = "RegularGrid"
        self._build_params_used = {}
        # Initialize all protocol attributes to satisfy type checkers, even if None
        self.bin_centers_ = np.empty((0, 0))
        self.connectivity_graph_ = None
        self.dimension_ranges_ = None
        self.grid_edges_ = None
        self.grid_shape_ = None
        self.active_mask_ = None

    def build(
        self,
        *,
        bin_size: Union[float, Sequence[float]],
        dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        data_samples: Optional[NDArray[np.float64]] = None,
        add_boundary_bins: bool = False,
        infer_active_bins: bool = True,
        dilate: bool = True,
        fill_holes: bool = True,
        close_gaps: bool = True,
        bin_count_threshold: int = 0,
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the regular N-D grid layout.

        Parameters
        ----------
        bin_size : Union[float, Sequence[float]]
            Desired size of bins in each dimension.
        dimension_ranges : Optional[Sequence[Tuple[float, float]]], optional
            Explicit `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]` for the grid.
            If None, range is inferred from `data_samples`.
        data_samples : Optional[NDArray[np.float64]], shape (n_samples, n_dims), optional
            Data used to infer `dimension_ranges` (if not provided) and/or to
            infer active bins (if `infer_active_bins` is True).
        add_boundary_bins : bool, default=False
            If True, adds one bin on each side of the grid, extending the range.
        infer_active_bins : bool, default=True
            If True and `data_samples` are provided, infers active bins based
            on occupancy and morphological operations.
        dilate : bool, default=False
            If `infer_active_bins` is True, dilates the inferred active area.
        fill_holes : bool, default=False
            If `infer_active_bins` is True, fills holes in the inferred active area.
        close_gaps : bool, default=False
            If `infer_active_bins` is True, closes gaps in the inferred active area.
        bin_count_threshold : int, default=0
            If `infer_active_bins` is True, minimum samples in a bin to be
            considered initially occupied.
        connect_diagonal_neighbors : bool, default=True
            If True, connects diagonal neighbors in the connectivity graph.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        # --- Determine dimension_ranges if not provided ---
        if dimension_ranges is not None:
            self.dimension_ranges_ = dimension_ranges
        else:
            # Infer ranges from data_samples
            if data_samples is None:
                raise ValueError(
                    "dimension_ranges must be provided if data_samples is None."
                )

            buffer_for_inference = (
                bin_size / 2.0
                if isinstance(bin_size, (float, int, np.number))
                else bin_size
            )
            # Infer ranges from data_samples
            self.dimension_ranges_ = _infer_dimension_ranges_from_samples(
                data_samples=data_samples,
                buffer_around_data=buffer_for_inference,
            )

        (
            self.grid_edges_,
            full_grid_bin_centers,
            self.grid_shape_,
        ) = _create_regular_grid(
            data_samples=data_samples,
            bin_size=bin_size,
            dimension_range=self.dimension_ranges_,
            add_boundary_bins=add_boundary_bins,
        )

        if infer_active_bins and data_samples is not None:
            self.active_mask_ = _infer_active_bins_from_regular_grid(
                data_samples=data_samples,
                edges=self.grid_edges_,
                close_gaps=close_gaps,
                fill_holes=fill_holes,
                dilate=dilate,
                bin_count_threshold=bin_count_threshold,
                boundary_exists=add_boundary_bins,
            )
        else:
            # No data_samples or not inferring active bins, use all bins
            self.active_mask_ = np.ones(self.grid_shape_, dtype=bool)

        if not np.any(self.active_mask_):
            raise ValueError(
                "No active bins found. Check your data_samples and bin_size."
            )

        self.bin_centers_ = full_grid_bin_centers[self.active_mask_.ravel()]
        self.connectivity_graph_ = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask_,
            grid_shape=self.grid_shape_,
            connect_diagonal=connect_diagonal_neighbors,
        )


class HexagonalLayout(_KDTreeMixin):
    """
    2D layout that tiles a rectangular area with a hexagonal lattice.

    Bin centers are the centers of the hexagons. Hexagons are connected to their
    immediate neighbors. Active hexagons can be inferred from data sample
    occupancy. Uses `_KDTreeMixin` for neighbor finding after the connectivity
    graph is built, but `point_to_bin_index` is specialized for hexagonal grids.
    """

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = ()
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # Layout Specific
    hexagon_width: Optional[float] = None
    _source_flat_to_active_id_map: Optional[Dict[int, int]] = None

    def __init__(self):
        """Initialize a HexagonalLayout engine."""
        self._layout_type_tag = "Hexagonal"
        self._build_params_used = {}
        self.bin_centers_ = np.empty((0, 2))
        self.connectivity_graph_ = None
        self.dimension_ranges_ = None
        self.grid_edges_ = ()
        self.grid_shape_ = None
        self.active_mask_ = None
        self.hexagon_width = None
        self.hex_radius_ = None
        self.hex_orientation_ = None
        self.grid_offset_x_ = None
        self.grid_offset_y_ = None
        self._source_flat_to_active_id_map = None

    def build(
        self,
        *,
        hexagon_width: float,
        dimension_ranges: Optional[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = None,
        data_samples: Optional[NDArray[np.float64]] = None,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
    ) -> None:
        """
        Build the hexagonal grid layout.

        Parameters
        ----------
        hexagon_width : float
            The width of the hexagons (distance between parallel sides).
        dimension_ranges : Optional[Tuple[Tuple[float,float], Tuple[float,float]]], optional
            Explicit `[(min_x, max_x), (min_y, max_y)]` for the area to tile.
            If None (default), range is inferred from `data_samples`.
        data_samples : Optional[NDArray[np.float64]], shape (n_samples, 2), optional
            2D data used to infer `dimension_ranges` (if not provided) and/or
            to infer active hexagons (if `infer_active_bins` is True).
            Defaults to None.
        infer_active_bins : bool, default=True
            If True and `data_samples` are provided, infers active hexagons
            based on occupancy. If False, all hexagons within the defined
            area are considered active.
        bin_count_threshold : int, default=0
            If `infer_active_bins` is True, the minimum number of samples a
            hexagon must contain to be considered active.

        Raises
        ------
        ValueError
            If `dimension_ranges` and `data_samples` are both None, or if
            `hexagon_width` is not positive.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        self.hexagon_width = hexagon_width
        (
            full_grid_bin_centers,
            self.grid_shape_,
            self.hex_radius_,
            self.hex_orientation_,
            self.grid_offset_x_,
            self.grid_offset_y_,
            self.dimension_ranges_,
        ) = _create_hex_grid(
            data_samples=data_samples,
            dimension_range=dimension_ranges,
            hexagon_width=self.hexagon_width,
        )
        if infer_active_bins and data_samples is not None:
            active_bin_original_flat_indices = _infer_active_bins_from_hex_grid(
                data_samples=data_samples,
                centers_shape=self.grid_shape_,
                hex_radius=self.hex_radius_,
                min_x=self.grid_offset_x_,
                min_y=self.grid_offset_y_,
                bin_count_threshold=bin_count_threshold,
            )
        else:
            active_bin_original_flat_indices = np.arange(len(full_grid_bin_centers))

        nd_active_mask = np.zeros(self.grid_shape_, dtype=bool).ravel()
        nd_active_mask[active_bin_original_flat_indices] = True
        self.active_mask_ = nd_active_mask.reshape(self.grid_shape_)

        self.bin_centers_ = full_grid_bin_centers[active_bin_original_flat_indices]

        self.connectivity_graph_ = _create_hex_connectivity_graph(
            active_original_flat_indices=active_bin_original_flat_indices,
            full_grid_bin_centers=full_grid_bin_centers,
            centers_shape=self.grid_shape_,
        )

        self._source_flat_to_active_id_map = {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity_graph_.nodes(data=True)
        }

    @property
    def is_1d(self) -> bool:
        """Hexagonal layouts are 2-dimensional."""
        return False

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the hexagonal layout.

        Displays active hexagons and their connectivity graph.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments:
            - `show_hexagons` (bool, default=True): Whether to draw hexagon cells.
            - `hexagon_kwargs` (dict): Kwargs for `matplotlib.patches.RegularPolygon`.
            - Other kwargs are passed to `_generic_graph_plot` for the graph.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        ax = _generic_graph_plot(
            ax=ax,
            graph=self.connectivity_graph_,
            name=self._layout_type_tag,
            **kwargs,
        )

        if (
            kwargs.get("show_hexagons", True)
            and self.hex_radius_ is not None
            and self.bin_centers_ is not None
            and self.bin_centers_.shape[0] > 0
        ):

            hex_kws = kwargs.get(
                "hexagon_kwargs",
                {
                    "edgecolor": "gray",
                    "facecolor": "none",
                    "alpha": 0.5,
                    "linewidth": 0.5,
                },
            )

            ax.scatter(
                self.bin_centers_[:, 0],
                self.bin_centers_[:, 1],
                s=1,
                label="hexagonal grid",
            )
            patches = [
                RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=self.hex_radius_,
                    orientation=self.hex_orientation_,
                )
                for x, y in self.bin_centers_
            ]

            collection = PatchCollection(patches, **hex_kws)
            ax.add_collection(collection)
            ax.plot(
                self.bin_centers_[:, 0],
                self.bin_centers_[:, 1],
                marker="o",
                markersize=1,
                color="blue",
                linestyle="None",
                label="midpoint",
            )

            ax.set_title(f"{self._layout_type_tag} Layout")
            padding = 1.1 * self.hex_radius_
            ax.set_xlim(
                (
                    self.dimension_ranges_[0][0] - padding,
                    self.dimension_ranges_[0][1] + padding,
                )
            )
            ax.set_ylim(
                (
                    self.dimension_ranges_[1][0] - padding,
                    self.dimension_ranges_[1][1] + padding,
                )
            )
            ax.set_aspect("equal", adjustable="box")
        return ax

    def point_to_bin_index(self, points):
        """
        Map 2D points to active hexagonal bin indices.

        Uses specialized logic to determine which hexagon each point falls into,
        then maps this to an active bin index.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, 2)
            2D points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Active bin indices (0 to N-1). -1 for points not in an active hexagon.
        """
        if (
            self.grid_offset_x_ is None
            or self.grid_offset_y_ is None
            or self.hex_radius_ is None
            or self.grid_shape_ is None
            or self._source_flat_to_active_id_map is None
        ):
            # This can happen if build() failed or was incomplete (e.g. no active bins)
            warnings.warn(
                "HexagonalLayout is not fully initialized or has no active bins. "
                "Cannot map points to bin indices.",
                RuntimeWarning,
            )
            return np.full(points.shape[0], -1, dtype=np.int_)

        original_flat_indices = _points_to_hex_bin_ind(
            points=points,
            grid_offset_x=self.grid_offset_x_,
            grid_offset_y=self.grid_offset_y_,
            hex_radius=self.hex_radius_,
            centers_shape=self.grid_shape_,
        )
        return np.array(
            [
                self._source_flat_to_active_id_map.get(idx, -1)
                for idx in original_flat_indices
            ],
            dtype=int,
        )

    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """
        Calculate the area of each hexagonal bin.

        All active hexagons are assumed to have the same area.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area for each active hexagonal bin.

        Raises
        ------
        RuntimeError
            If `hex_radius_` or `bin_centers_` is not populated.
        """
        if self.hex_radius_ is None or self.bin_centers_ is None:  # pragma: no cover
            raise RuntimeError("Layout not built; hex_radius_ or bin_centers_ missing.")

        # Area of a regular hexagon: (3 * sqrt(3) / 2) * side_length^2
        # For pointy-top hexagons, side_length (s) is equal to hex_radius_ (R, center to vertex).
        single_hex_area = 3.0 * np.sqrt(3.0) / 2.0 * self.hex_radius_**2.0
        return np.full(self.bin_centers_.shape[0], single_hex_area)


class GraphLayout(_KDTreeMixin):
    """
    Layout defined by a user-provided graph, typically for 1D tracks.

    The graph's nodes (with 'pos' attributes) and a specified edge order
    are used to create a linearized representation of the space, which is
    then binned. Connectivity is derived from this binned structure.
    Uses `_KDTreeMixin` for point mapping and neighbor finding on the
    N-D embeddings of the linearized bin centers.
    """

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # Layout Specific
    linear_bin_centers_: Optional[NDArray[np.float64]] = None

    def __init__(self):
        """Initialize a GraphLayout engine."""
        self._layout_type_tag = "Graph"
        self._build_params_used = {}
        self.bin_centers_ = np.empty((0, 0), dtype=np.float64)
        self.connectivity_graph_ = None
        self.dimension_ranges_ = None
        self.grid_edges_ = None
        self.grid_shape_ = None
        self.active_mask_ = None
        self.linear_bin_centers_ = None

    def build(
        self,
        *,
        graph_definition: nx.Graph,  # Original user-provided graph
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        bin_size: float,  # Linearized bin size
    ) -> None:
        """
        Build the graph-based (linearized track) layout.

        Parameters
        ----------
        graph_definition : nx.Graph
            The original NetworkX graph. Nodes must have a 'pos' attribute
            (e.g., `(x, y)` coordinates) and edges should ideally have a
            'distance' attribute if not relying on Euclidean distance calculation.
        edge_order : List[Tuple[Any, Any]]
            An ordered sequence of edge tuples (node_id_1, node_id_2) from
            `graph_definition` that defines the ordering of edges in the
            linear space.
        edge_spacing : Union[float, Sequence[float]]
            Spacing (gap) to insert between consecutive edges in `edge_order`
            during linearization. If float, same gap for all. If sequence,
            specifies each gap; length must be `len(edge_order) - 1`.
        bin_size : float
            The desired length of each bin along the linearized space.

        Raises
        ------
        TypeError
            If `graph_definition` is not a NetworkX graph.
        ValueError
            If `edge_order` is empty or `bin_size` is not positive.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        if not isinstance(graph_definition, nx.Graph):
            raise TypeError("graph_definition must be a NetworkX graph.")
        if not edge_order:  # Empty edge_order means no path to linearize
            raise ValueError("edge_order must not be empty.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        (self.linear_bin_centers_, self.grid_edges_, self.active_mask_, edge_ids) = (
            _get_graph_bins(
                graph=graph_definition,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                bin_size=bin_size,
            )
        )
        self.bin_centers_ = _project_1d_to_2d(
            self.linear_bin_centers_,
            graph_definition,
            edge_order,
            edge_spacing,
        )
        self.grid_shape_ = (len(self.bin_centers_),)
        self.connectivity_graph_ = _create_graph_layout_connectivity_graph(
            graph=graph_definition,
            bin_centers_nd=self.bin_centers_,
            linear_bin_centers=self.linear_bin_centers_,
            original_edge_ids=edge_ids,
            active_mask=self.active_mask_,
            edge_order=edge_order,
        )
        self.dimension_ranges_ = (
            np.min(self.bin_centers_[:, 0]),
            np.max(self.bin_centers_[:, 0]),
        ), (np.min(self.bin_centers_[:, 1]), np.max(self.bin_centers_[:, 1]))

        # --- Build KDTree ---
        self._build_kdtree(points_for_tree=self.bin_centers_[self.active_mask_])

    @property
    def is_1d(self) -> bool:
        """Graph layouts are treated as 1-dimensional due to linearization."""
        return True

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the N-D embedding of the graph-based layout.

        Displays the original graph used for definition, the N-D positions of
        the binned track segments (active bin centers), and their connectivity.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments:
            - `figsize` (tuple): Figure size if `ax` is None.
            - `node_kwargs` (dict): Kwargs for plotting original graph nodes.
            - `edge_kwargs` (dict): Kwargs for plotting original graph edges.
            - `bin_node_kwargs` (dict): Kwargs for plotting active bin center nodes.
            - `bin_edge_kwargs` (dict): Kwargs for plotting connectivity graph edges.
            - `show_bin_edges` (bool): Whether to project and plot 1D bin edges in N-D.


        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        # Draw the original graph nodes
        original_node_pos = nx.get_node_attributes(
            self._build_params_used["graph_definition"], "pos"
        )
        nx.draw_networkx_nodes(
            self._build_params_used["graph_definition"],
            original_node_pos,
            ax=ax,
            node_size=300,
            node_color="#1f77b4",
        )
        # Draw the original graph edges
        for node_id1, node_id2 in self._build_params_used["graph_definition"].edges:
            pos = np.stack(
                (
                    original_node_pos[node_id1],
                    original_node_pos[node_id2],
                )
            )
            ax.plot(
                pos[:, 0], pos[:, 1], color="gray", zorder=-1, label="original edges"
            )

        for node_id, pos in original_node_pos.items():
            plt.text(
                pos[0],
                pos[1],
                str(node_id),
                ha="center",
                va="center",
                zorder=10,
            )

        # Draw the bin centers
        bin_centers = nx.get_node_attributes(self.connectivity_graph_, "pos")
        nx.draw_networkx_nodes(
            self.connectivity_graph_,
            bin_centers,
            ax=ax,
            node_size=30,
            node_color="black",
        )

        # Draw connectivity graph edges
        for node_id1, node_id2 in self.connectivity_graph_.edges:
            pos = np.stack((bin_centers[node_id1], bin_centers[node_id2]))
            ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

        grid_line_2d = _project_1d_to_2d(
            self.grid_edges_[0],
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        )
        for grid_line in grid_line_2d:
            ax.plot(
                grid_line[0],
                grid_line[1],
                color="gray",
                marker="+",
                alpha=0.8,
                label="bin edges",
            )
        return ax

    def plot_linear_layout(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the 1D linearized representation of the graph layout.

        Uses `track_linearization.plot_graph_as_1D` to display the track
        segments and nodes in their 1D linearized positions. Overlays the
        1D bin edges from `self.grid_edges_`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments passed to
            `track_linearization.plot_graph_as_1D` and for customizing
            the appearance of bin edge lines.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the 1D layout is plotted.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 3)))

        plot_graph_as_1D(
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
            ax=ax,
            **kwargs,
        )
        for grid_line in self.grid_edges_[0]:
            ax.axvline(grid_line, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{self._layout_type_tag} Layout")
        ax.set_xlabel("Linearized Position")
        ax.set_ylabel("Bin Index")

        return ax

    def get_linearized_coordinate(
        self, data_points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convert N-D points to 1D linearized coordinates along the track.

        Uses `track_linearization.get_linearized_position`.

        Parameters
        ----------
        data_points : NDArray[np.float64], shape (n_points, n_dims)
            N-D points to linearize.

        Returns
        -------
        NDArray[np.float64], shape (n_points,)
            1D linearized coordinates. NaNs may be returned for points
            far from the track.
        """
        return _get_linearized_position(
            data_points,
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        ).linear_position.to_numpy()

    def map_linear_to_nd_coordinate(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Map 1D linearized coordinates back to N-D coordinates on the track graph.

        Parameters
        ----------
        linear_coordinates : NDArray[np.float64], shape (n_points,)
            1D linearized coordinates to map.

        Returns
        -------
        NDArray[np.float64], shape (n_points, n_dims)
            N-D coordinates corresponding to the input linear positions.
        """
        return _project_1d_to_2d(
            linear_coordinates,
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        )

    def linear_point_to_bin_ind(self, data_points):
        """
        Map 1D linearized positions to active 1D bin indices.

        Parameters
        ----------
        linear_positions : NDArray[np.float64], shape (n_points,)
            1D linearized positions.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Indices of the active 1D bins corresponding to each linear position.
            Returns -1 for positions outside active bins or in gaps.
            Note: These are indices relative to the set of *active* 1D bins,
            not indices into the full `linear_bin_centers_all` array.
        """
        return _find_bin_for_linear_position(
            data_points, bin_edges=self.grid_edges_[0], active_mask=self.active_mask_
        )

    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """
        Return the length of each active 1D bin along the linearized track.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the length of each active linearized bin.

        Raises
        ------
        RuntimeError
            If `grid_edges_` or `active_mask_` is not populated.
        """
        if self.grid_edges_ is None or self.active_mask_ is None:  # pragma: no cover
            raise RuntimeError("Layout not built; grid_edges_ or active_mask_ missing.")
        if not self.grid_edges_ or self.grid_edges_[0].size <= 1:  # pragma: no cover
            raise ValueError(
                "grid_edges_ (1D) are not properly defined for length calculation."
            )

        all_1d_bin_lengths = np.diff(self.grid_edges_[0])
        return all_1d_bin_lengths[self.active_mask_]


if SHAPELY_AVAILABLE:

    class ShapelyPolygonLayout(_GridMixin):
        """
        2D grid layout masked by a Shapely Polygon.

        Creates a regular grid based on the polygon's bounds and specified
        `bin_size`. Only grid cells whose centers are contained within the
        polygon are considered active. Inherits grid functionalities from
        `_GridMixin`.
        """

        bin_centers_: NDArray[np.float64]
        connectivity_graph_: Optional[nx.Graph] = None
        dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

        grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
        grid_shape_: Optional[Tuple[int, ...]] = None
        active_mask_: Optional[NDArray[np.bool_]] = None

        _layout_type_tag: str
        _build_params_used: Dict[str, Any]

        # Layout Specific
        _polygon_definition: Optional[PolygonType] = None

        def __init__(self):
            """Initialize a ShapelyPolygonLayout engine."""
            self._layout_type_tag = "ShapelyPolygon"
            self._build_params_used = {}
            self.bin_centers_ = np.empty((0, 2), dtype=np.float64)  # 2D Layout
            self.connectivity_graph_ = None
            self.dimension_ranges_ = None
            self.grid_edges_ = None
            self.grid_shape_ = None
            self.active_mask_ = None
            self.polygon_definition_ = None

        def build(
            self,
            *,
            polygon: PolygonType,
            bin_size: Union[float, Sequence[float]],
            connect_diagonal_neighbors: bool = True,
        ) -> None:
            """
            Build the Shapely Polygon masked grid layout.

            Parameters
            ----------
            polygon : shapely.geometry.Polygon
                The Shapely Polygon object that defines the boundary of the
                active area.
            bin_size : Union[float, Sequence[float]]
                The side length(s) of the grid cells. If float, cells are
                square (or cubic). If sequence (length 2 for 2D), specifies
                (width, height).
            connect_diagonal_neighbors : bool, default=True
                If True, connect diagonally adjacent active grid cells in the
                `connectivity_graph_`.

            Raises
            ------
            RuntimeError
                If the 'shapely' package is not installed (should be caught by
                SHAPELY_AVAILABLE check at class definition).
            TypeError
                If `polygon` is not a Shapely Polygon.
            """
            if not SHAPELY_AVAILABLE:
                raise RuntimeError("ShapelyGridEngine requires the 'shapely' package.")

            if not isinstance(polygon, Polygon):
                raise TypeError("polygon must be a Shapely Polygon object.")

            self._build_params_used = locals().copy()  # Store all passed params
            del self._build_params_used["self"]  # Remove self from the dictionary

            self.polygon_definition_ = polygon
            minx, miny, maxx, maxy = polygon.bounds
            self.dimension_ranges_ = [(minx, maxx), (miny, maxy)]

            (
                self.grid_edges_,
                full_grid_bin_centers,
                self.grid_shape_,
            ) = _create_regular_grid(
                data_samples=None,
                bin_size=bin_size,
                dimension_range=self.dimension_ranges_,
                add_boundary_bins=False,
            )

            # 1. Intrinsic mask from Shapely
            pts_to_check = (
                full_grid_bin_centers[:, :2]
                if full_grid_bin_centers.shape[0] > 0
                else np.empty((0, 2))
            )
            shapely_mask_flat = (
                np.array([polygon.contains(Point(*p)) for p in pts_to_check])
                if pts_to_check.shape[0] > 0
                else np.array([], dtype=bool)
            )
            self.active_mask_ = shapely_mask_flat.reshape(self.grid_shape_)

            self.bin_centers_ = full_grid_bin_centers[self.active_mask_.ravel()]
            self.connectivity_graph_ = _create_regular_grid_connectivity_graph(
                full_grid_bin_centers=full_grid_bin_centers,
                active_mask_nd=self.active_mask_,
                grid_shape=self.grid_shape_,
                connect_diagonal=connect_diagonal_neighbors,
            )

    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        figsize=(7, 7),
        cmap: str = "bone_r",
        alpha: float = 0.7,
        draw_connectivity_graph: bool = True,
        node_size: float = 20,
        node_color: str = "blue",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot the ShapelyPolygon layout.

        Displays the active grid cells and overlays the defining polygon.
        Inherits base grid plotting from `_GridMixin.plot` and adds
        polygon visualization.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        figsize : Tuple[float, float], default=(7, 7)
            Size of the figure if `ax` is None.
        **kwargs : Any
            Additional keyword arguments passed to `_GridMixin.plot()`
            (e.g., `cmap`, `alpha` for the grid) and for polygon plotting
            (e.g., `polygon_kwargs` which is a dict for `ax.fill`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        if (
            self.bin_centers_ is None
            or self.grid_edges_ is None
            or self.active_mask_ is None
            or self.grid_shape_ is None
            or self.connectivity_graph_ is None
        ):
            raise RuntimeError("Layout not built. Call `build` first.")

        is_2d_grid = len(self.grid_shape_) == 2 and len(self.grid_edges_) == 2

        if is_2d_grid:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.pcolormesh(
                self.grid_edges_[0],
                self.grid_edges_[1],
                self.active_mask_.T,
                cmap=cmap,
                alpha=alpha,
                shading="auto",
            )
            ax.set_xticks(self.grid_edges_[0])
            ax.set_yticks(self.grid_edges_[1])
            ax.grid(True, ls="-", lw=0.5, c="gray")
            ax.set_aspect("equal")
            ax.set_title(f"{self._layout_type_tag} (2D Grid)")
            ax.set_xlabel("Dimension 0")
            ax.set_ylabel("Dimension 1")
            if self.dimension_ranges_:
                ax.set_xlim(self.dimension_ranges_[0])
                ax.set_ylim(self.dimension_ranges_[1])

            if draw_connectivity_graph:
                node_position = nx.get_node_attributes(self.connectivity_graph_, "pos")
                nx.draw_networkx_nodes(
                    self.connectivity_graph_,
                    node_position,
                    ax=ax,
                    node_size=node_size,
                    node_color=node_color,
                )
                for node_id1, node_id2 in self.connectivity_graph_.edges:
                    pos = np.stack((node_position[node_id1], node_position[node_id2]))
                    ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

            # Plot polygon
            poly_patch_kwargs = kwargs.get(
                "polygon_kwargs", {"alpha": 0.3, "fc": "gray", "ec": "black"}
            )
            if hasattr(self.polygon_definition_, "geoms"):  # MultiPolygon
                for geom in self.polygon_definition_.geoms:
                    if hasattr(geom, "exterior"):
                        x, y = geom.exterior.xy
                        ax.fill(x, y, **poly_patch_kwargs)
            elif hasattr(self.polygon_definition_, "exterior"):  # Polygon
                x, y = self.polygon_definition_.exterior.xy
                ax.fill(x, y, **poly_patch_kwargs)

            return ax
        else:
            raise NotImplementedError(
                "Plotting for non-2D grid layouts is not implemented yet."
            )

else:
    ShapelyPolygonLayout = None  # type: ignore


class MaskedGridLayout(_GridMixin):  # type: ignore
    """
    Layout from a pre-defined N-D boolean mask and explicit grid edges.

    Allows for precise specification of active bins in an N-dimensional grid
    by providing the complete grid structure (`grid_edges`) and a mask
    (`active_mask`) that designates which cells of that grid are active.
    Inherits grid functionalities from `_GridMixin`.
    """

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]
    bin_size_: Optional[NDArray[np.float64]] = None

    def __init__(self):
        """Initialize a MaskedGridLayout engine."""
        self._layout_type_tag = "MaskedGrid"
        self._build_params_used = {}
        self.bin_centers_ = np.empty((0, 0), dtype=np.float64)
        self.connectivity_graph_ = None
        self.dimension_ranges_ = None
        self.grid_edges_ = None
        self.grid_shape_ = None
        self.active_mask_ = None
        self.bin_size_ = None

    def build(
        self,
        *,
        active_mask: NDArray[np.bool_],  # User's N-D definition mask
        grid_edges: Tuple[NDArray[np.float64], ...],
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the layout from a mask and grid edges.

        Parameters
        ----------
        active_mask : NDArray[np.bool_]
            N-dimensional boolean array where `True` indicates an active bin.
            Its shape must correspond to the number of bins defined by `grid_edges`
            (i.e., `tuple(len(e)-1 for e in grid_edges)`).
        grid_edges : Tuple[NDArray[np.float64], ...]
            A tuple where each element is a 1D NumPy array of bin edge
            positions for that dimension, defining the full grid structure.
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active grid cells.

        Raises
        ------
        ValueError
            If `active_mask` shape does not match `grid_edges` definition,
            or if `grid_edges` are invalid.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        self.active_mask_ = active_mask
        self.grid_edges_ = grid_edges
        self.grid_shape_ = tuple(len(edge) - 1 for edge in grid_edges)

        if self.active_mask_.shape != self.grid_shape_:
            raise ValueError(
                f"active_mask shape {self.active_mask_.shape} must match "
                f"the shape implied by grid_edges {self.grid_shape_}."
            )

        # Create full_grid_bin_centers as (N_total_bins, N_dims) array
        centers_per_dim = [get_centers(edge_dim) for edge_dim in self.grid_edges_]
        mesh_centers_list = np.meshgrid(*centers_per_dim, indexing="ij", sparse=False)
        full_grid_bin_centers = np.stack(
            [c.ravel() for c in mesh_centers_list], axis=-1
        )

        self.bin_size_ = np.array(
            [np.diff(edge_dim)[0] for edge_dim in self.grid_edges_], dtype=np.float64
        )

        self.dimension_ranges_ = tuple(
            (edge_dim[0], edge_dim[-1]) for edge_dim in self.grid_edges_
        )
        self.bin_centers_ = full_grid_bin_centers[self.active_mask_.ravel()]

        self.connectivity_graph_ = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask_,
            grid_shape=self.grid_shape_,
            connect_diagonal=connect_diagonal_neighbors,
        )

        self._build_params_used = {
            "active_mask": active_mask,
            "grid_edges": grid_edges,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }


class ImageMaskLayout(_GridMixin):
    """
    2D layout derived from a boolean image mask.

    Each `True` pixel in the input `image_mask` corresponds to an active bin
    in the environment. The spatial scale of these pixel-bins is determined
    by `bin_size`. Inherits grid functionalities from `_GridMixin`.
    """

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    def __init__(self):
        """Initialize an ImageMaskLayout engine."""
        self._layout_type_tag = "ImageMask"
        self._build_params_used = {}
        self.bin_centers_ = np.empty((0, 2), dtype=np.float64)
        self.connectivity_graph_ = None
        self.dimension_ranges_ = None
        self.grid_edges_ = None
        self.grid_shape_ = None
        self.active_mask_ = None

    def build(
        self,
        *,
        image_mask: NDArray[np.bool_],  # Defines candidate pixels
        bin_size: Union[float, Tuple[float, float]] = 1.0,  # one pixel
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the layout from a 2D image mask.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where `True` pixels define active bins.
        bin_size : Union[float, Tuple[float, float]], default=1.0
            The spatial size of each pixel.
            If float: pixels are square (size x size).
            If tuple (width, height): specifies pixel_width and pixel_height.
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active pixel-bins.

        Raises
        ------
        TypeError
            If `image_mask` is not a NumPy array.
        ValueError
            If `image_mask` is not 2D, not boolean, or `bin_size` is invalid,
            or if `image_mask` contains no True values or non-finite values.
        """

        if not isinstance(image_mask, np.ndarray):
            raise TypeError("image_mask must be a numpy array.")
        if image_mask.ndim != 2:
            raise ValueError("image_mask must be a 2D array.")
        if not np.issubdtype(image_mask.dtype, np.bool_):
            raise ValueError("image_mask must be a boolean array.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")
        if not np.any(image_mask):
            raise ValueError("image_mask must contain at least one True value.")
        if not np.all(np.isfinite(image_mask)):
            raise ValueError("image_mask must not contain NaN or Inf values.")

        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        # Determine bin_sizes for x and y (units per pixel)
        bin_size_x: float
        bin_size_y: float
        if isinstance(bin_size, (float, int, np.number)):
            bin_size_x = float(bin_size)
            bin_size_y = float(bin_size)
        elif isinstance(bin_size, (list, tuple, np.ndarray)) and len(bin_size) == 2:
            bin_size_x = float(bin_size[0])  # width of pixel
            bin_size_y = float(bin_size[1])  # height of pixel
        else:
            raise ValueError(
                "bin_size for ImageMaskLayout must be a float or a 2-element sequence (width, height)."
            )

        if bin_size_x <= 0 or bin_size_y <= 0:
            raise ValueError("bin_size components must be positive.")

        n_rows, n_cols = image_mask.shape
        self.grid_shape_ = (n_rows, n_cols)  # Note: (rows, cols) often (y_dim, x_dim)
        y_edges = np.arange(n_rows + 1) * bin_size_y
        x_edges = np.arange(n_cols + 1) * bin_size_x
        self.grid_edges_ = (y_edges, x_edges)
        self.dimension_ranges_ = (
            (x_edges[0], x_edges[-1]),
            (y_edges[0], y_edges[-1]),
        )

        y_centers = (np.arange(n_rows) + 0.5) * bin_size_y
        x_centers = (np.arange(n_cols) + 0.5) * bin_size_x
        xv, yv = np.meshgrid(
            x_centers, y_centers, indexing="xy"
        )  # x is cols, y is rows
        full_grid_bin_centers = np.stack((xv.ravel(), yv.ravel()), axis=1)

        self.active_mask_ = image_mask
        self.bin_centers_ = full_grid_bin_centers[self.active_mask_.ravel()]
        self.connectivity_graph_ = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask_,
            grid_shape=self.grid_shape_,
            connect_diagonal=connect_diagonal_neighbors,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
_LAYOUT_MAP: Dict[str, type[LayoutEngine]] = {
    "RegularGrid": RegularGridLayout,
    "MaskedGrid": MaskedGridLayout,
    "ImageMask": ImageMaskLayout,
    "Hexagonal": HexagonalLayout,
    "Graph": GraphLayout,
}
if SHAPELY_AVAILABLE and ShapelyPolygonLayout is not None:
    _LAYOUT_MAP["ShapelyPolygon"] = ShapelyPolygonLayout


def list_available_layouts() -> List[str]:
    """
    List user-friendly type strings for all available layout engines.

    Returns
    -------
    List[str]
        A sorted list of unique string identifiers for available
        `LayoutEngine` types (e.g., "RegularGrid", "Hexagonal").
    """
    unique_options: List[str] = []
    processed_normalized_options: set[str] = set()
    for opt in _LAYOUT_MAP.keys():
        norm_opt = "".join(filter(str.isalnum, opt)).lower()
        if norm_opt not in processed_normalized_options:
            is_alias = any(
                "".join(filter(str.isalnum, added)).lower() == norm_opt
                for added in unique_options
            )
            if not is_alias:
                unique_options.append(opt)
            processed_normalized_options.add(norm_opt)
    return sorted(unique_options)


def get_layout_parameters(layout_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve expected build parameters for a specified layout engine type.

    Inspects the `build` method signature of the specified `LayoutEngine`
    class to determine its required and optional parameters.

    Parameters
    ----------
    layout_type : str
        The string identifier of the layout engine type (case-insensitive,
        ignores non-alphanumeric characters).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary where keys are parameter names for the `build` method.
        Each value is another dictionary containing:
        - 'annotation': The type annotation of the parameter.
        - 'default': The default value, or `inspect.Parameter.empty` if no default.
        - 'kind': The parameter kind (e.g., 'keyword-only').

    Raises
    ------
    ValueError
        If `layout_type` is unknown.
    """
    normalized_kind_query = "".join(filter(str.isalnum, layout_type)).lower()
    found_key = next(
        (
            k
            for k in _LAYOUT_MAP
            if "".join(filter(str.isalnum, k)).lower() == normalized_kind_query
        ),
        None,
    )
    if not found_key:
        raise ValueError(
            f"Unknown engine kind '{layout_type}'. Available: {list_available_layouts()}"
        )
    engine_class = _LAYOUT_MAP[found_key]
    sig = inspect.signature(engine_class.build)
    params_info: Dict[str, Dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params_info[name] = {
            "annotation": (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            ),
            "default": (
                param.default
                if param.default is not inspect.Parameter.empty
                else inspect.Parameter.empty
            ),
            "kind": param.kind.description,
        }
    return params_info


def create_layout(kind: str, **kwargs) -> LayoutEngine:
    """
    Instantiate and build a layout engine by its 'kind' string.

    This factory function finds the appropriate `LayoutEngine` class based
    on the `kind` string, instantiates it, and calls its `build` method
    with the provided `**kwargs`.

    Parameters
    ----------
    kind : str
        The string identifier of the layout engine type to create
        (case-insensitive, ignores non-alphanumeric characters).
        See `list_available_layouts()` for options.
    **kwargs : Any
        Keyword arguments to be passed to the `build` method of the
        selected layout engine.

    Returns
    -------
    LayoutEngine
        A fully built instance of the specified layout engine.

    Raises
    ------
    ValueError
        If `kind` is an unknown layout engine type.
    """
    normalized_kind_query = "".join(filter(str.isalnum, kind)).lower()
    found_key = next(
        (
            k
            for k in _LAYOUT_MAP
            if "".join(filter(str.isalnum, k)).lower() == normalized_kind_query
        ),
        None,
    )
    if not found_key:
        raise ValueError(
            f"Unknown engine kind '{kind}'. Available: {list_available_layouts()}"
        )
    engine_class = _LAYOUT_MAP[found_key]
    engine_instance = engine_class()
    build_sig = inspect.signature(engine_instance.build)
    valid_build_params = {p_name for p_name in build_sig.parameters if p_name != "self"}
    actual_build_kwargs = {k: v for k, v in kwargs.items() if k in valid_build_params}
    engine_instance.build(**actual_build_kwargs)

    return engine_instance
