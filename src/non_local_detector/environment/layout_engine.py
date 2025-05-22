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
    Protocol defining the interface for all layout definitions.

    A LayoutEngine is responsible for discretizing a continuous N-dimensional
    space into a set of bins/elements and constructing a graph representation
    of their connectivity.

    Attributes
    ----------
    bin_centers_ : NDArray[np.float64], shape (n_active_bins, n_dims).
        Coordinates of the center of each *active* bin
    connectivity_graph_ : Optional[nx.Graph]
        Graph where nodes are indexed 0 to n_active_bins-1, directly corresponding
        to `bin_centers_`. Nodes should have a 'pos' attribute (from `bin_centers_`)
        and a 'source_index' attribute mapping back to an original definition
        (e.g., flat index in a full grid, original input point index).
    is_1d : bool
        True if the layout represents a 1-dimensional structure.
    dimension_ranges_ : Optional[Sequence[Tuple[float, float]]]
        The actual min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry.
    grid_edges_ : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts: Bin edges for each dimension of the *original, full grid*.
        None or () for non-grid layouts.
    grid_shape_ : Optional[Tuple[int, ...]]
        For grid-based layouts: The N-D shape of the *original, full grid*.
        For point-based/cell-based layouts: (n_active_bins,).
    active_mask_ : Optional[NDArray[np.bool_]]
        - For grid-based layouts: N-D boolean mask on the *original, full grid*.
        - For point-based/cell-based layouts: 1D array of all True,
          shape (n_active_bins,), corresponding to `bin_centers_`.
    _layout_type_tag : str
        String identifier for the type of layout (e.g., "RegularGrid").
    _build_params_used : Dict[str, Any]
        Dictionary of parameters used to build this layout instance.

    Properties
    ----------
    is_1d : bool
        True if the layout represents a 1-dimensional structure.
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
        Constructs the layout's geometry, bins, and connectivity graph.
        Populates all attributes defined above.
        """
        ...

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Maps continuous N-D points to discrete active bin indices (0 to N-1).
        Returns -1 for points not mapped to an active bin.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, n_dims)
            N-D array of points to map to bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Array of bin indices corresponding to the input points.
            -1 for points outside the layout's active bins.
        """
        ...

    def get_bin_neighbors(self, bin_index: int) -> List[int]:
        """
        Finds indices of neighboring active bins for a given active bin index (0 to N-1).
        Uses `connectivity_graph_`.

        Parameters
        ----------
        bin_index : int
            Index of the active bin for which to find neighbors.
        Returns
        -------
        List[int]
            List of indices of neighboring active bins.
        """
        ...

    @property
    @abstractmethod
    def is_1d(self) -> bool:
        """True if the layout structure is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the layout is 1D, False otherwise.
        """
        ...

    @abstractmethod
    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the layout's geometry.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes]
            Axes to plot on. If None, a new figure and axes are created.
        **kwargs : Additional keyword arguments for customization.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        ...

    @abstractmethod
    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """Returns the area/volume of each bin in the layout.

        Returns
        -------
        NDArray[np.float64]
            Array of bin areas/volumes.
        """
        ...


# ---------------------------------------------------------------------------
# KD-tree mixin (for point_to_bin_index)
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """
    Mixin providing `point_to_bin_index` and `get_bin_neighbors`
    functionality using a KD-tree built on bin centers and a NetworkX graph
    for connectivity.

    It finds the nearest bin center to a given point and returns the index of
    that bin center in `bin_centers_`. The KD-tree is built on the active
    bin centers, and the graph is used to find neighboring bins.

    This mixin assumes that the class it's mixed into will define:
    - `self.bin_centers_`: An NDArray of shape (n_active_bins, n_dims)
      containing the coordinates of the active bin centers. This is the array
      on which the KD-tree will be built.
    - `self.connectivity_graph_`: A NetworkX graph where nodes are integers
      from `0` to `n_active_bins - 1`, corresponding to the rows of
      `self.bin_centers_`.

    The `_build_kdtree` method should be called by the implementing layout's
    `build` method after `bin_centers_` is finalized.

    Attributes:
    ----------
    _kdtree : Optional[KDTree]
        KDTree for fast nearest-neighbor search.
    _kdtree_source_indices_map : Optional[NDArray[np.int_]], shape (n_active_bins,)
        Maps KDTree node index to original source_index.
    """

    _kdtree: Optional[KDTree] = None
    _kdtree_source_indices_map: Optional[NDArray[np.int_]] = None

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
        Maps continuous N-D points to discrete active bin indices (0 to N-1)
        by finding the nearest bin center in `self.bin_centers_`.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_query_points, n_dims)
            N-D array of query points.

        Returns
        -------
        NDArray[np.int_], shape (n_query_points,)
            Array of active bin indices (corresponding to rows in `self.bin_centers_`
            and node IDs in `self.connectivity_graph_`).
            Returns -1 for points if the KD-tree is not built or is empty.
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
        Finds indices of neighboring active bins for a given active bin index.

        Uses the `connectivity_graph_` which should be defined by the layout
        and have nodes `0` to `n_active_bins - 1`.

        Parameters
        ----------
        bin_index : int
            Index of the active bin (0 to n_active_bins - 1) for which to find neighbors.

        Returns
        -------
        List[int]
            List of indices of neighboring active bins.

        Raises
        ------
        AttributeError
            If `connectivity_graph_` is not defined on the instance.
        ValueError
            If `connectivity_graph_` is None (not built).
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
    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None
    bin_centers_: Optional[NDArray[np.float64]] = None
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None
    _layout_type_tag: str = "_Grid_Layout"

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
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
        Finds indices of neighboring active bins for a given active bin index.

        Uses the `connectivity_graph_` which should be defined by the layout
        and have nodes `0` to `n_active_bins - 1`.

        Parameters
        ----------
        bin_index : int
            Index of the active bin (0 to n_active_bins - 1) for which to find neighbors.

        Returns
        -------
        List[int]
            List of indices of neighboring active bins.

        Raises
        ------
        AttributeError
            If `connectivity_graph_` is not defined on the instance.
        ValueError
            If `connectivity_graph_` is None (not built).
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
        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes]
            Axes to plot on. If None, a new figure and axes are created.
        figsize : tuple, default=(7, 7)
            Size of the figure if a new one is created.
        cmap : str, default="bone_r"
            Colormap for the active mask.
        alpha : float, default=0.7
            Transparency level for the active mask.
        draw_connectivity_graph : bool, default=True
            If True, draws the connectivity graph on top of the active mask.

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
        """RegularGridLayout is N-D"""
        return False

    def get_bin_area_volume(self) -> NDArray[np.float64]:
        return (
            np.prod(
                np.array([np.diff(edge)[0] for edge in self.grid_edges_]),
                axis=0,
            )
            .reshape(self.grid_shape_)
            .T
        )


# ---------------------------------------------------------------------------
# Specific LayoutEngine Implementations
# ---------------------------------------------------------------------------


class RegularGridLayout(_GridMixin):
    """
    Axis-aligned rectangular N-D grid layout.

    Discretizes space into a uniform N-dimensional grid. Can infer the
    active portion of this grid based on provided data samples.
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
        Builds the regular grid layout.

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
    immediate neighbors in the lattice. Can infer an active subset of these
    hexagons based on data sample occupancy, leveraging an intermediate
    regular grid for robust morphological operations.
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
        return False

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
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
            if self.dimension_ranges_:
                ax.set_xlim(self.dimension_ranges_[0])
                ax.set_ylim(self.dimension_ranges_[1])
            ax.set_aspect("equal", adjustable="box")
        return ax

    def point_to_bin_index(self, points):
        """
        Maps continuous N-D points to discrete active bin indices (0 to N-1).
        Returns -1 for points not mapped to an active bin.
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
        return (
            3.0 * np.sqrt(3.0) / 2.0 * self.hex_radius_**2.0 * np.ones(self.grid_shape_)
        )


class GraphLayout(_KDTreeMixin):
    """User-provided graph layout.

    The graph is expected to be a NetworkX graph with nodes having 'pos'
    attributes for their coordinates. The graph is used to define the
    connectivity of the layout. The layout is not necessarily regular or
    uniform, and the bin centers are derived from the graph's node positions.
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
    nd_bin_centers_: NDArray[np.float64] = None

    def __init__(self):
        self._layout_type_tag = "Graph"
        self._build_params_used = {}

    def build(
        self,
        *,
        graph_definition: nx.Graph,  # Original user-provided graph
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        bin_size: float,  # Linearized bin size
    ) -> None:
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        if not isinstance(graph_definition, nx.Graph):
            raise TypeError("graph_definition must be a NetworkX graph.")
        if not edge_order:  # Empty edge_order means no path to linearize
            raise ValueError("edge_order must not be empty.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        (self.bin_centers_, self.grid_edges_, self.active_mask_, edge_ids) = (
            _get_graph_bins(
                graph=graph_definition,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                bin_size=bin_size,
            )
        )
        self.grid_shape_ = (len(self.bin_centers_),)
        self.nd_bin_centers_ = _project_1d_to_2d(
            self.bin_centers_,
            graph_definition,
            edge_order,
            edge_spacing,
        )
        self.connectivity_graph_ = _create_graph_layout_connectivity_graph(
            graph=graph_definition,
            bin_centers_2D=self.nd_bin_centers_,
            bin_centers_1D=self.bin_centers_,
            edge_ids=edge_ids,
            active_mask=self.active_mask_,
            edge_order=edge_order,
        )
        self.dimension_ranges_ = (
            np.min(self.nd_bin_centers_[:, 0]),
            np.max(self.nd_bin_centers_[:, 0]),
        ), (np.min(self.nd_bin_centers_[:, 1]), np.max(self.nd_bin_centers_[:, 1]))

        # --- Build KDTree ---
        self._build_kdtree(points_for_tree=self.nd_bin_centers_[self.active_mask_])

    @property
    def is_1d(self) -> bool:
        return True  # GraphLayout is 1D by default

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))
        kwargs["node_size"] = kwargs.get("node_size", 10)
        node_position = nx.get_node_attributes(self.connectivity_graph_, "pos")
        nx.draw_networkx_nodes(self.connectivity_graph_, node_position, ax=ax, **kwargs)

        original_node_pos = nx.get_node_attributes(
            self._build_params_used["graph_definition"], "pos"
        )
        for node_id, pos in original_node_pos.items():
            plt.text(
                pos[0],
                pos[1],
                str(node_id),
                fontsize=8,
                ha="center",
                va="center",
                zorder=10,
            )
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
                marker="o",
                alpha=0.5,
            )
        return ax

    def plot_linear_layout(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 2))

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
        return _get_linearized_position(
            data_points,
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        ).linear_position.to_numpy_array()

    def map_linear_to_nd_coordinate(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _project_1d_to_2d(
            linear_coordinates,
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        )

    def linear_point_to_bin_ind(self, data_points):
        return _find_bin_for_linear_position(
            data_points, bin_edges=self.grid_edges_[0], active_mask=self.active_mask_
        )

    def get_bin_area_volume(self) -> NDArray[np.float64]:
        return np.ones(self.grid_shape_) * self._build_params_used["bin_size"]


if SHAPELY_AVAILABLE:

    class ShapelyPolygonLayout(_GridMixin):
        """2D grid layout masked by a Shapely Polygon."""

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
            self._layout_type_tag = "ShapelyPolygon"
            self._build_params_used = {}

        def build(
            self,
            *,
            polygon: PolygonType,
            bin_size: Union[float, Sequence[float]],
            connect_diagonal_neighbors: bool = True,
        ) -> None:
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
        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes]
            Axes to plot on. If None, a new figure and axes are created.
        figsize : tuple, default=(7, 7)
            Size of the figure if a new one is created.
        cmap : str, default="bone_r"
            Colormap for the active mask.
        alpha : float, default=0.7
            Transparency level for the active mask.
        draw_connectivity_graph : bool, default=True
            If True, draws the connectivity graph on top of the active mask.

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
    """Layout from a pre-defined N-D boolean mask and grid edges."""

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    def __init__(self):
        self._layout_type_tag = "MaskedGrid"
        self._build_params_used = {}  # type: ignore

    def build(
        self,
        *,
        active_mask: NDArray[np.bool_],  # User's N-D definition mask
        grid_edges: Tuple[NDArray[np.float64], ...],
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        self.active_mask_ = active_mask
        self.grid_edges_ = grid_edges

        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        full_grid_bin_centers = np.meshgrid(
            *[get_centers(edge) for edge in grid_edges],
            indexing="ij",
            sparse=False,
        )
        self.bin_size_ = np.array(
            [np.diff(edge)[0] for edge in grid_edges], dtype=np.float64
        )
        self.grid_shape_ = tuple(len(edge) - 1 for edge in grid_edges)
        self.dimension_ranges_ = [(edge[0], edge[-1]) for edge in grid_edges]
        self.bin_centers_ = full_grid_bin_centers[self.active_mask_.ravel()]

        self.connectivity_graph_ = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask_,
            grid_shape=self.grid_shape_,
            connect_diagonal=connect_diagonal_neighbors,
        )


class ImageMaskLayout(_GridMixin):
    """2D layout from a boolean image mask, pixel centers are bins."""

    bin_centers_: NDArray[np.float64]
    connectivity_graph_: Optional[nx.Graph] = None
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    def __init__(self):
        self._layout_type_tag = "ImageMask"
        self._build_params_used = {}  # type: ignore

    def build(
        self,
        *,
        image_mask: NDArray[np.bool_],  # Defines candidate pixels
        bin_size: Union[float, Tuple[float, float]] = 1.0,  # one pixel
        connect_diagonal_neighbors: bool = True,
    ) -> None:

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
    """Lists user-friendly 'type' strings for all available layout definitions."""
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
    """Retrieves expected build parameters for a specified layout type."""
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
    """Instantiates and builds a layout definition by its 'kind' string."""
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
