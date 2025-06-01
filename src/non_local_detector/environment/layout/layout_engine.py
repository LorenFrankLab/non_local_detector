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
import shapely.vectorized
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
from track_linearization import get_linearized_position as _get_linearized_position
from track_linearization import plot_graph_as_1D

from non_local_detector.environment.layout.graph import (
    _create_graph_layout_connectivity_graph,
    _find_bin_for_linear_position,
    _get_graph_bins,
    _project_1d_to_2d,
)
from non_local_detector.environment.layout.hex_grid import (
    _create_hex_connectivity_graph,
    _create_hex_grid,
    _infer_active_bins_from_hex_grid,
    _points_to_hex_bin_ind,
)
from non_local_detector.environment.layout.mixins import _GridMixin, _KDTreeMixin
from non_local_detector.environment.layout.regular_grid import (
    _create_regular_grid,
    _create_regular_grid_connectivity_graph,
    _infer_active_bins_from_regular_grid,
    get_centers,
)
from non_local_detector.environment.layout.triangular import (
    _build_mesh_connectivity_graph,
    _compute_mesh_dimension_ranges,
    _filter_active_simplices_by_centroid,
    _generate_interior_points_for_mesh,
    _triangulate_points,
)
from non_local_detector.environment.layout.utils import (
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
    bin_centers : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node.
        Shape is (n_active_bins, n_dims).
    connectivity : Optional[nx.Graph]
        Graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to rows in `bin_centers`.
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
    dimension_ranges : Optional[Sequence[Tuple[float, float]]]
        The actual min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry.
    grid_edges : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts: A tuple of 1D arrays, where each array
        contains the bin edge positions for one dimension of the *original,
        full grid*. `None` or `()` for non-grid or point-based layouts.
    grid_shape : Optional[Tuple[int, ...]]
        For grid-based layouts: The N-D shape (number of bins in each
        dimension) of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept:
        Typically `(n_active_bins,)`.
    active_mask : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid* (shape matches `grid_shape`).
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers`.
    _layout_type_tag : str
        A string identifier for the type of layout (e.g., "RegularGrid").
        Used for introspection and serialization.
    _build_params_used : Dict[str, Any]
        A dictionary of the parameters used to construct this layout instance.
        Used for introspection and serialization.

    """

    # --- Required Data Attributes ---
    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    # Attributes primarily for GRID-BASED Layouts
    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    # Internal Attributes for Introspection/Serialization
    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # --- Required Methods ---
    def build(self, **kwargs) -> None:
        """
        Construct the layout's geometry, bins, and connectivity graph.

        This method is responsible for populating all the attributes defined
        in the `LayoutEngine` protocol (e.g., `bin_centers`,
        `connectivity`, etc.) based on the provided keyword arguments.
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
    def bin_sizes(self) -> NDArray[np.float64]:
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

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None
    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    def __init__(self):
        """Initialize a RegularGridLayout engine."""
        self._layout_type_tag = "RegularGrid"
        self._build_params_used = {}
        # Initialize all protocol attributes to satisfy type checkers, even if None
        self.bin_centers = np.empty((0, 0))
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

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
            self.dimension_ranges = dimension_ranges
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
            self.dimension_ranges = _infer_dimension_ranges_from_samples(
                data_samples=data_samples,
                buffer_around_data=buffer_for_inference,
            )

        (
            self.grid_edges,
            full_grid_bin_centers,
            self.grid_shape,
        ) = _create_regular_grid(
            data_samples=data_samples,
            bin_size=bin_size,
            dimension_range=self.dimension_ranges,
            add_boundary_bins=add_boundary_bins,
        )

        if infer_active_bins and data_samples is not None:
            self.active_mask = _infer_active_bins_from_regular_grid(
                data_samples=data_samples,
                edges=self.grid_edges,
                close_gaps=close_gaps,
                fill_holes=fill_holes,
                dilate=dilate,
                bin_count_threshold=bin_count_threshold,
                boundary_exists=add_boundary_bins,
            )
        else:
            # No data_samples or not inferring active bins, use all bins
            self.active_mask = np.ones(self.grid_shape, dtype=bool)

        if not np.any(self.active_mask):
            raise ValueError(
                "No active bins found. Check your data_samples and bin_size."
            )

        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
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

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = ()
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # Layout Specific
    hexagon_width: Optional[float] = None
    _source_flat_to_active_id_map: Optional[Dict[int, int]] = None

    def __init__(self):
        """Initialize a HexagonalLayout engine."""
        self._layout_type_tag = "Hexagonal"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2))
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = ()
        self.grid_shape = None
        self.active_mask = None
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
            self.grid_shape,
            self.hex_radius_,
            self.hex_orientation_,
            self.grid_offset_x_,
            self.grid_offset_y_,
            self.dimension_ranges,
        ) = _create_hex_grid(
            data_samples=data_samples,
            dimension_range=dimension_ranges,
            hexagon_width=self.hexagon_width,
        )
        if infer_active_bins and data_samples is not None:
            active_bin_original_flat_indices = _infer_active_bins_from_hex_grid(
                data_samples=data_samples,
                centers_shape=self.grid_shape,
                hex_radius=self.hex_radius_,
                min_x=self.grid_offset_x_,
                min_y=self.grid_offset_y_,
                bin_count_threshold=bin_count_threshold,
            )
        else:
            active_bin_original_flat_indices = np.arange(len(full_grid_bin_centers))

        nd_active_mask = np.zeros(self.grid_shape, dtype=bool).ravel()
        nd_active_mask[active_bin_original_flat_indices] = True
        self.active_mask = nd_active_mask.reshape(self.grid_shape)

        self.bin_centers = full_grid_bin_centers[active_bin_original_flat_indices]

        self.connectivity = _create_hex_connectivity_graph(
            active_original_flat_indices=active_bin_original_flat_indices,
            full_grid_bin_centers=full_grid_bin_centers,
            centers_shape=self.grid_shape,
        )

        self._source_flat_to_active_id_map = {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity.nodes(data=True)
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
            graph=self.connectivity,
            name=self._layout_type_tag,
            **kwargs,
        )

        if (
            kwargs.get("show_hexagons", True)
            and self.hex_radius_ is not None
            and self.bin_centers is not None
            and self.bin_centers.shape[0] > 0
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
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
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
                for x, y in self.bin_centers
            ]

            collection = PatchCollection(patches, **hex_kws)
            ax.add_collection(collection)
            ax.plot(
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
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
                    self.dimension_ranges[0][0] - padding,
                    self.dimension_ranges[0][1] + padding,
                )
            )
            ax.set_ylim(
                (
                    self.dimension_ranges[1][0] - padding,
                    self.dimension_ranges[1][1] + padding,
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
            or self.grid_shape is None
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
            centers_shape=self.grid_shape,
        )
        return np.array(
            [
                self._source_flat_to_active_id_map.get(idx, -1)
                for idx in original_flat_indices
            ],
            dtype=int,
        )

    def bin_sizes(self) -> NDArray[np.float64]:
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
            If `hex_radius_` or `bin_centers` is not populated.
        """
        if self.hex_radius_ is None or self.bin_centers is None:  # pragma: no cover
            raise RuntimeError("Layout not built; hex_radius_ or bin_centers missing.")

        # Area of a regular hexagon: (3 * sqrt(3) / 2) * side_length^2
        # For pointy-top hexagons, side_length (s) is equal to hex_radius_ (R, center to vertex).
        single_hex_area = 3.0 * np.sqrt(3.0) / 2.0 * self.hex_radius_**2.0
        return np.full(self.bin_centers.shape[0], single_hex_area)


class GraphLayout(_KDTreeMixin):
    """
    Layout defined by a user-provided graph, typically for 1D tracks.

    The graph's nodes (with 'pos' attributes) and a specified edge order
    are used to create a linearized representation of the space, which is
    then binned. Connectivity is derived from this binned structure.
    Uses `_KDTreeMixin` for point mapping and neighbor finding on the
    N-D embeddings of the linearized bin centers.
    """

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # Layout Specific
    linear_bin_centers_: Optional[NDArray[np.float64]] = None

    def __init__(self):
        """Initialize a GraphLayout engine."""
        self._layout_type_tag = "Graph"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 0), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
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

        (self.linear_bin_centers_, self.grid_edges, self.active_mask, edge_ids) = (
            _get_graph_bins(
                graph=graph_definition,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                bin_size=bin_size,
            )
        )
        self.bin_centers = _project_1d_to_2d(
            self.linear_bin_centers_,
            graph_definition,
            edge_order,
            edge_spacing,
        )
        self.grid_shape = (len(self.bin_centers),)
        self.connectivity = _create_graph_layout_connectivity_graph(
            graph=graph_definition,
            bin_centers_nd=self.bin_centers,
            linear_bin_centers=self.linear_bin_centers_,
            original_edge_ids=edge_ids,
            active_mask=self.active_mask,
            edge_order=edge_order,
        )
        self.dimension_ranges = (
            np.min(self.bin_centers[:, 0]),
            np.max(self.bin_centers[:, 0]),
        ), (np.min(self.bin_centers[:, 1]), np.max(self.bin_centers[:, 1]))

        # --- Build KDTree ---
        self._build_kdtree(points_for_tree=self.bin_centers[self.active_mask])

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
        bin_centers = nx.get_node_attributes(self.connectivity, "pos")
        nx.draw_networkx_nodes(
            self.connectivity,
            bin_centers,
            ax=ax,
            node_size=30,
            node_color="black",
        )

        # Draw connectivity graph edges
        for node_id1, node_id2 in self.connectivity.edges:
            pos = np.stack((bin_centers[node_id1], bin_centers[node_id2]))
            ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

        grid_line_2d = _project_1d_to_2d(
            self.grid_edges[0],
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
        1D bin edges from `self.grid_edges`.

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
        for grid_line in self.grid_edges[0]:
            ax.axvline(grid_line, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{self._layout_type_tag} Layout")
        ax.set_xlabel("Linearized Position")
        ax.set_ylabel("Bin Index")

        return ax

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def linear_to_nd(
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
            data_points, bin_edges=self.grid_edges[0], active_mask=self.active_mask
        )

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Return the length of each active 1D bin along the linearized track.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the length of each active linearized bin.

        Raises
        ------
        RuntimeError
            If `grid_edges` or `active_mask` is not populated.
        """
        if self.grid_edges is None or self.active_mask is None:  # pragma: no cover
            raise RuntimeError("Layout not built; grid_edges or active_mask missing.")
        if not self.grid_edges or self.grid_edges[0].size <= 1:  # pragma: no cover
            raise ValueError(
                "grid_edges (1D) are not properly defined for length calculation."
            )

        all_1d_bin_lengths = np.diff(self.grid_edges[0])
        return all_1d_bin_lengths[self.active_mask]


if SHAPELY_AVAILABLE:

    class ShapelyPolygonLayout(_GridMixin):
        """
        2D grid layout masked by a Shapely Polygon.

        Creates a regular grid based on the polygon's bounds and specified
        `bin_size`. Only grid cells whose centers are contained within the
        polygon are considered active. Inherits grid functionalities from
        `_GridMixin`.
        """

        bin_centers: NDArray[np.float64]
        connectivity: Optional[nx.Graph] = None
        dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

        grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
        grid_shape: Optional[Tuple[int, ...]] = None
        active_mask: Optional[NDArray[np.bool_]] = None

        _layout_type_tag: str
        _build_params_used: Dict[str, Any]

        # Layout Specific
        _polygon_definition: Optional[PolygonType] = None

        def __init__(self):
            """Initialize a ShapelyPolygonLayout engine."""
            self._layout_type_tag = "ShapelyPolygon"
            self._build_params_used = {}
            self.bin_centers = np.empty((0, 2), dtype=np.float64)  # 2D Layout
            self.connectivity = None
            self.dimension_ranges = None
            self.grid_edges = None
            self.grid_shape = None
            self.active_mask = None
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
                `connectivity`.

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
            self.dimension_ranges = [(minx, maxx), (miny, maxy)]

            (
                self.grid_edges,
                full_grid_bin_centers,
                self.grid_shape,
            ) = _create_regular_grid(
                data_samples=None,
                bin_size=bin_size,
                dimension_range=self.dimension_ranges,
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
            self.active_mask = shapely_mask_flat.reshape(self.grid_shape)

            self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
            self.connectivity = _create_regular_grid_connectivity_graph(
                full_grid_bin_centers=full_grid_bin_centers,
                active_mask_nd=self.active_mask,
                grid_shape=self.grid_shape,
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
            self.bin_centers is None
            or self.grid_edges is None
            or self.active_mask is None
            or self.grid_shape is None
            or self.connectivity is None
        ):
            raise RuntimeError("Layout not built. Call `build` first.")

        is_2d_grid = len(self.grid_shape) == 2 and len(self.grid_edges) == 2

        if is_2d_grid:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.pcolormesh(
                self.grid_edges[0],
                self.grid_edges[1],
                self.active_mask.T,
                cmap=cmap,
                alpha=alpha,
                shading="auto",
            )
            ax.set_xticks(self.grid_edges[0])
            ax.set_yticks(self.grid_edges[1])
            ax.grid(True, ls="-", lw=0.5, c="gray")
            ax.set_aspect("equal")
            ax.set_title(f"{self._layout_type_tag} (2D Grid)")
            ax.set_xlabel("Dimension 0")
            ax.set_ylabel("Dimension 1")
            if self.dimension_ranges:
                ax.set_xlim(self.dimension_ranges[0])
                ax.set_ylim(self.dimension_ranges[1])

            if draw_connectivity_graph:
                node_position = nx.get_node_attributes(self.connectivity, "pos")
                nx.draw_networkx_nodes(
                    self.connectivity,
                    node_position,
                    ax=ax,
                    node_size=node_size,
                    node_color=node_color,
                )
                for node_id1, node_id2 in self.connectivity.edges:
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

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]
    bin_size_: Optional[NDArray[np.float64]] = None

    def __init__(self):
        """Initialize a MaskedGridLayout engine."""
        self._layout_type_tag = "MaskedGrid"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 0), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
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

        self.active_mask = active_mask
        self.grid_edges = grid_edges
        self.grid_shape = tuple(len(edge) - 1 for edge in grid_edges)

        if self.active_mask.shape != self.grid_shape:
            raise ValueError(
                f"active_mask shape {self.active_mask.shape} must match "
                f"the shape implied by grid_edges {self.grid_shape}."
            )

        # Create full_grid_bin_centers as (N_total_bins, N_dims) array
        centers_per_dim = [get_centers(edge_dim) for edge_dim in self.grid_edges]
        mesh_centers_list = np.meshgrid(*centers_per_dim, indexing="ij", sparse=False)
        full_grid_bin_centers = np.stack(
            [c.ravel() for c in mesh_centers_list], axis=-1
        )

        self.bin_size_ = np.array(
            [np.diff(edge_dim)[0] for edge_dim in self.grid_edges], dtype=np.float64
        )

        self.dimension_ranges = tuple(
            (edge_dim[0], edge_dim[-1]) for edge_dim in self.grid_edges
        )
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]

        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
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

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    def __init__(self):
        """Initialize an ImageMaskLayout engine."""
        self._layout_type_tag = "ImageMask"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

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
        self.grid_shape = (n_rows, n_cols)  # Note: (rows, cols) often (y_dim, x_dim)
        y_edges = np.arange(n_rows + 1) * bin_size_y
        x_edges = np.arange(n_cols + 1) * bin_size_x
        self.grid_edges = (y_edges, x_edges)
        self.dimension_ranges = (
            (x_edges[0], x_edges[-1]),
            (y_edges[0], y_edges[-1]),
        )

        y_centers = (np.arange(n_rows) + 0.5) * bin_size_y
        x_centers = (np.arange(n_cols) + 0.5) * bin_size_x
        xv, yv = np.meshgrid(
            x_centers, y_centers, indexing="xy"
        )  # x is cols, y is rows
        full_grid_bin_centers = np.stack((xv.ravel(), yv.ravel()), axis=1)

        self.active_mask = image_mask
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )


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

        Uses Delaunay.find_simplex() → original‐simplex index → active‐triangle index via
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
