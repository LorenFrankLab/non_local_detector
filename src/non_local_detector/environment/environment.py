from __future__ import annotations

import pickle
import warnings
from dataclasses import asdict, dataclass, field
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from non_local_detector.environment.layout.layout_engine import (
    GraphLayout,
    LayoutEngine,
    RegularGridLayout,
    create_layout,
)
from non_local_detector.environment.layout.utils import _get_distance_between_bins
from non_local_detector.environment.regions import Regions

if TYPE_CHECKING:
    # For type hinting shapely without a hard dependency
    import shapely.geometry as _shp_type  # Renamed to avoid conflict

try:
    import shapely.geometry as _shp

    _HAS_SHAPELY = True
except ModuleNotFoundError:
    _HAS_SHAPELY = False

    class _shp:  # type: ignore[misc]
        class Polygon:
            pass  # noqa N801


PolygonType = type[_shp.Polygon]  # type: ignore[misc]


# --- Decorator ---
def check_fitted(method):
    """
    Decorator to ensure that an Environment method is called only after fitting.

    Raises
    ------
    RuntimeError
        If the method is called on an Environment instance that has not been
        fully initialized (i.e., `_is_fitted` is False).
    """

    @wraps(method)
    def _inner(self: "Environment", *args, **kwargs):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                f"{self.__class__.__name__}.{method.__name__}() "
                "requires the environment to be fully initialized. "
                "Ensure it was created with a factory method."
            )
        return method(self, *args, **kwargs)

    return _inner


# --- Main Environment Class ---
@dataclass
class Environment:
    """
    Represents a discretized N-dimensional space with connectivity.

    This class serves as a comprehensive model of a spatial environment,
    discretized into bins or nodes. It stores the geometric properties of these
    bins (e.g., centers, areas), their connectivity, and provides methods for
    various spatial queries and operations.

    Instances are typically created using one of the provided classmethod
    factories (e.g., `Environment.from_data_samples(...)`,
    `Environment.from_graph(...)`). These factories handle the underlying
    `LayoutEngine` setup.

    Attributes
    ----------
    name : str
        A user-defined name for the environment.
    layout : LayoutEngine
        The layout engine instance that defines the geometry and connectivity
        of the discretized space.
    bin_centers_ : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node in the environment.
        Shape is (n_active_bins, n_dims). Populated by `_setup_from_layout`.
    connectivity_graph_ : nx.Graph
        A NetworkX graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to the rows of `bin_centers_`. Edges represent
        adjacency between bins. Populated by `_setup_from_layout`.
    dimension_ranges_ : Optional[Sequence[Tuple[float, float]]]
        The effective min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry. Populated by `_setup_from_layout`.
    grid_edges_ : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts, a tuple where each element is a 1D array of
        bin edge positions for that dimension of the *original, full grid*.
        `None` or `()` for non-grid or point-based layouts. Populated by
        `_setup_from_layout`.
    grid_shape_ : Optional[Tuple[int, ...]]
        For grid-based layouts, the N-D shape of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept, this
        may be `(n_active_bins,)`. Populated by `_setup_from_layout`.
    active_mask_ : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid*.
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers_`.
        Populated by `_setup_from_layout`.
    regions : RegionManager
        Manages symbolic spatial regions defined within this environment.
    _is_1d_env : bool
        Internal flag indicating if the environment's layout is primarily 1-dimensional.
        Set based on `layout.is_1d`.
    _is_fitted : bool
        Internal flag indicating if the environment has been fully initialized
        and its layout-dependent attributes are populated.
    _layout_type_used : Optional[str]
        The string identifier of the `LayoutEngine` type used to create this
        environment (e.g., "RegularGrid"). For introspection and serialization.
    _layout_params_used : Dict[str, Any]
        A dictionary of the parameters used to build the `LayoutEngine` instance.
        For introspection and serialization.

    """

    name: str
    layout: LayoutEngine

    # --- Attributes populated from the layout instance ---
    bin_centers_: NDArray[np.float64] = field(init=False)
    connectivity_graph_: nx.Graph = field(init=False)
    dimension_ranges_: Optional[Sequence[Tuple[float, float]]] = field(init=False)

    # Grid-specific context (populated if layout is grid-based)
    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = field(init=False)
    grid_shape_: Optional[Tuple[int, ...]] = field(init=False)
    active_mask_: Optional[NDArray[np.bool_]] = field(init=False)

    # Region Management
    regions: Regions = field(init=False, repr=False)

    # Internal state
    _is_1d_env: bool = field(init=False)
    _is_fitted: bool = field(init=False, default=False)

    # For introspection and serialization
    _layout_type_used: Optional[str] = field(init=False, default=None)
    _layout_params_used: Dict[str, Any] = field(init=False, default_factory=dict)

    # Cache the mapping from source flat indices to active node IDs
    _source_flat_to_active_node_id_map: Optional[Dict[int, int]] = field(
        init=False, default=None, repr=False
    )

    def __init__(
        self,
        name: str = "",
        layout: LayoutEngine = RegularGridLayout,
        layout_type_used: Optional[str] = None,
        layout_params_used: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Environment.

        Note: This constructor is primarily intended for internal use by factory
        methods. Users should typically create Environment instances using
        classmethods like `Environment.from_data_samples(...)`. The provided
        `layout` instance is assumed to be already built and configured.

        Parameters
        ----------
        name : str, optional
            Name for the environment, by default "".
        layout : LayoutEngine
            A fully built LayoutEngine instance that defines the environment's
            geometry and connectivity.
        layout_type_used : Optional[str], optional
            The string identifier for the type of layout used. If None, it's
            inferred from `layout._layout_type_tag`. Defaults to None.
        layout_params_used : Optional[Dict[str, Any]], optional
            Parameters used to build the layout. If None, inferred from
            `layout._build_params_used`. Defaults to None.

        """
        self.name = name
        self.layout = layout

        self._layout_type_used = (
            layout_type_used
            if layout_type_used
            else getattr(layout, "_layout_type_tag", None)
        )
        self._layout_params_used = (
            layout_params_used
            if layout_params_used is not None
            else getattr(layout, "_build_params_used", {})
        )

        self._is_1d_env = self.layout.is_1d

        # Initialize attributes that will be populated by _setup_from_layout
        self.bin_centers_ = np.empty((0, 0))  # Placeholder
        self.connectivity_graph_ = nx.Graph()
        self.dimension_ranges_ = None
        self.grid_edges_ = ()
        self.grid_shape_ = None
        self.active_mask_ = None
        self._is_fitted = False  # Will be set by _setup_from_layout

        self._setup_from_layout()  # Populate attributes from the built layout
        self.regions = Regions()

    def __repr__(self: "Environment") -> str:
        """
        Generate an unambiguous string representation of the Environment.

        Returns
        -------
        str
            A string representation of the Environment object, including its
            name, layout type, and key geometric properties if fitted.

        """
        class_name = self.__class__.__name__
        env_name_repr = f"name={self.name!r}"
        layout_type_repr = f"layout_type={self._layout_type_used!r}"

        if not self._is_fitted:
            # If not fitted, show minimal information
            return f"{class_name}({env_name_repr}, {layout_type_repr}, fitted=False)"

        # If fitted, provide more details
        try:
            dims_repr = f"n_dims={self.n_dims}"
        except RuntimeError:
            # Should not happen if _is_fitted is True and n_dims is correctly implemented
            dims_repr = "n_dims='Error'"

        active_bins_repr = "active_bins='N/A'"
        if self.bin_centers_ is not None:
            active_bins_repr = f"active_bins={self.bin_centers_.shape[0]}"

        return (
            f"{class_name}("
            f"{env_name_repr}, "
            f"{layout_type_repr}, "
            f"{dims_repr}, "
            f"{active_bins_repr}, "
            f"fitted=True"
            f")"
        )

    def _setup_from_layout(self) -> None:
        """
        Populate Environment attributes from its (built) LayoutEngine.

        This internal method is called after the `LayoutEngine` is associated
        with the Environment. It copies essential geometric and connectivity
        information from the layout to the Environment's attributes.
        It also applies fallbacks for certain grid-specific attributes if the
        layout is point-based to ensure consistency.
        """

        self.bin_centers_ = self.layout.bin_centers_
        self.connectivity_graph_ = getattr(
            self.layout, "connectivity_graph_", nx.Graph()
        )
        self.dimension_ranges_ = self.layout.dimension_ranges_

        # Grid-specific attributes
        self.grid_edges_ = getattr(self.layout, "grid_edges_", ())
        self.grid_shape_ = getattr(self.layout, "grid_shape_", None)
        self.active_mask_ = getattr(self.layout, "active_mask_", None)

        # If it's not a grid layout, grid_shape_ might be (n_active_bins,),
        # and active_mask_ might be 1D all True. This is fine.
        # Ensure they are at least None if not applicable from layout
        if self.grid_shape_ is None and self.bin_centers_ is not None:
            # Fallback for point-based
            self.grid_shape_ = (self.bin_centers_.shape[0],)
        if self.active_mask_ is None and self.bin_centers_ is not None:
            # Fallback for point-based
            self.active_mask_ = np.ones(self.bin_centers_.shape[0], dtype=bool)

        self._is_fitted = True

    @cached_property
    @check_fitted
    def _source_flat_to_active_node_id_map(self) -> Dict[int, int]:
        """
        Get or create the mapping from original full grid flat indices
        to active bin IDs (0 to n_active_bins - 1).

        The map is cached on the instance for subsequent calls. This method
        is intended for internal use by other Environment or related manager methods.

        Returns
        -------
        Dict[int, int]
            A dictionary mapping `source_grid_flat_index` from graph nodes
            to the `active_bin_id` (which is the graph node ID).

        Raises
        ------
        RuntimeError
            If the connectivity graph is not available, or if all nodes are
            missing the 'source_grid_flat_index' attribute required for the map.
        """
        return {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity_graph_.nodes(data=True)
            if "source_grid_flat_index" in data
        }

    # --- Factory Methods ---
    @classmethod
    def from_data_samples(
        cls,
        data_samples: NDArray[np.float64],
        name: str = "",
        layout_type: str = "RegularGrid",
        bin_size: Optional[Union[float, Sequence[float]]] = 2.0,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
        **layout_specific_kwargs: Any,
    ) -> Environment:
        """
        Create an Environment, primarily inferring geometry from data samples.

        This factory method initializes a `LayoutEngine` of the specified
        `layout_type` using the provided `data_samples` to determine spatial
        extents and, optionally, to infer which bins/regions are active.

        Parameters
        ----------
        data_samples : NDArray[np.float64], shape (n_samples, n_dims)
            N-dimensional coordinates of data samples (e.g., animal positions)
            used to define the environment's geometry and active areas.
        name : str, optional
            A name for the created environment. Defaults to "".
        layout_type : str, optional
            The type of layout to use (e.g., "RegularGrid", "Hexagonal").
            Defaults to "RegularGrid".
        bin_size : Optional[Union[float, Sequence[float]]], optional
            The characteristic size of the discretization bins.
            For "RegularGrid", this is the side length(s) of the grid cells.
            For "Hexagonal", this is interpreted as the hexagon width.
            Defaults to 2.0.
        infer_active_bins : bool, optional
            If True, the layout will attempt to infer active bins based on
            the density or presence of `data_samples`. Defaults to True.
        bin_count_threshold : int, optional
            If `infer_active_bins` is True, this is the minimum number of
            samples a bin must contain to be initially considered active.
            Defaults to 0.
        dilate : bool, optional
            If `infer_active_bins` is True (primarily for grid-based layouts),
            whether to apply a dilation operation to expand the active area.
            Defaults to False.
        fill_holes : bool, optional
            If `infer_active_bins` is True (primarily for grid-based layouts),
            whether to fill holes within the inferred active area.
            Defaults to False.
        close_gaps : bool, optional
            If `infer_active_bins` is True (primarily for grid-based layouts),
            whether to close small gaps in the inferred active area.
            Defaults to False.
        add_boundary_bins : bool, optional
            For "RegularGrid" layout, whether to add a layer of inactive
            boundary bins around the inferred active area. Defaults to False.
        connect_diagonal_neighbors : bool, optional
            For grid-based layouts, whether to connect diagonally adjacent bins
            in the `connectivity_graph_`. Defaults to True.
        **layout_specific_kwargs : Any
            Additional keyword arguments passed directly to the constructor
            of the chosen `LayoutEngine`.

        Returns
        -------
        Environment
            A new Environment instance.

        """
        build_params: Dict[str, Any] = {
            "data_samples": data_samples,
            "infer_active_bins": infer_active_bins,
            "bin_count_threshold": bin_count_threshold,
            **layout_specific_kwargs,
        }
        if layout_type.lower() == "regulargrid":
            build_params.update(
                {
                    "bin_size": bin_size,
                    "add_boundary_bins": add_boundary_bins,
                    "dilate": dilate,
                    "fill_holes": fill_holes,
                    "close_gaps": close_gaps,
                    "connect_diagonal_neighbors": connect_diagonal_neighbors,
                }
            )
        elif layout_type.lower() == "hexagonal":
            # For Hexagonal, bin_size is typically interpreted as hexagon_width
            build_params.update(
                {
                    "hex_width": bin_size,
                }
            )

        layout_instance = create_layout(kind=layout_type, **build_params)
        return cls(name, layout_instance, layout_type, build_params)

    @classmethod
    def with_dimension_ranges(
        cls,
        dimension_ranges: Sequence[Tuple[float, float]],
        name: str = "",
        layout_type: str = "RegularGrid",
        bin_size: Optional[Union[float, Sequence[float]]] = 2.0,
        **layout_specific_kwargs: Any,
    ) -> Environment:
        """
        Create an Environment with explicitly defined spatial boundaries.

        This factory method initializes a `LayoutEngine` covering the specified
        `dimension_ranges`. Unlike `from_data_samples`, this method does not
        typically infer active bins from data unless `data_samples` and relevant
        inference parameters are passed via `layout_specific_kwargs`.

        Parameters
        ----------
        dimension_ranges : Sequence[Tuple[float, float]]
            A sequence of (min, max) tuples defining the extent for each
            dimension, e.g., `[(x_min, x_max), (y_min, y_max)]`.
        name : str, optional
            A name for the created environment. Defaults to "".
        layout_type : str, optional
            The type of layout to use (e.g., "RegularGrid", "Hexagonal").
            Defaults to "RegularGrid".
        bin_size : Optional[Union[float, Sequence[float]]], optional
            The characteristic size of the discretization bins.
            Interpreted by the specific `layout_type`. Defaults to 2.0.
        **layout_specific_kwargs : Any
            Additional keyword arguments passed directly to the constructor
            of the chosen `LayoutEngine`. For example, to infer active bins
            within these ranges, pass `data_samples`, `infer_active_bins=True`, etc.

        Returns
        -------
        Environment
            A new Environment instance.

        """
        build_params: Dict[str, Any] = {
            "dimension_ranges": dimension_ranges,
            **layout_specific_kwargs,
        }
        if layout_type.lower() == "regulargrid":
            build_params.update(
                {
                    "bin_size": bin_size,
                }
            )
        elif layout_type.lower() == "hexagonal":
            build_params.update(
                {
                    "hexagon_width": bin_size,  # Assuming bin_size is hex_width for hexagonal
                }
            )

        layout_instance = create_layout(kind=layout_type, **build_params)
        return cls(name, layout_instance, layout_type, build_params)

    # --- Specialized Factories (signatures as discussed previously) ---
    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        bin_size: float,
        name: str = "",
        **kwargs,
    ) -> Environment:
        """
        Create an Environment from a user-defined graph structure.

        This method is used for 1D environments where the spatial layout is
        defined by a graph, an ordered list of its edges, and spacing between
        these edges. The track is then linearized and binned.

        Parameters
        ----------
        graph : nx.Graph
            The NetworkX graph defining the track segments. Nodes are expected
            to have a 'pos' attribute for their N-D coordinates.
        edge_order : List[Tuple[Any, Any]]
            An ordered list of edge tuples (node1, node2) from `graph` that
            defines the 1D bin ordering.
        edge_spacing : Union[float, Sequence[float]]
            The spacing to insert between consecutive edges in `edge_order`
            during linearization. If a float, applies to all gaps. If a
            sequence, specifies spacing for each gap.
        bin_size : float
            The length of each bin along the linearized track.
        name : str, optional
            A name for the created environment. Defaults to "".
        **kwargs : Any
            Additional parameters for the GraphLayout, though specific ones
            are explicitly listed.

        Returns
        -------
        Environment
            A new Environment instance with a `GraphLayout`.

        """
        layout_instance = create_layout(
            kind="Graph",
            graph_definition=graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            bin_size=bin_size,
        )
        return cls(name, layout_instance, "Graph", kwargs)

    @classmethod
    def from_shapely_polygon(
        cls,
        polygon: PolygonType,
        bin_size: Optional[Union[float, Sequence[float]]] = 2.0,
        name: str = "",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
        """
        Create a 2D grid Environment masked by a Shapely Polygon.

        A regular grid is formed based on the polygon's bounds and `bin_size`.
        Only grid cells whose centers are contained within the polygon are
        considered active.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The Shapely Polygon object that defines the boundary of the active area.
        bin_size : Optional[Union[float, Sequence[float]]], optional
            The side length(s) of the grid cells. Defaults to 2.0.
        name : str, optional
            A name for the created environment. Defaults to "".
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.

        Returns
        -------
        Environment
            A new Environment instance with a `ShapelyPolygonLayout`.

        Raises
        ------
        RuntimeError
            If the 'shapely' package is not installed.

        """
        layout_params = {
            "polygon": polygon,
            "bin_size": bin_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }
        layout_instance = create_layout(kind="ShapelyPolygon", **layout_params)
        return cls(name, layout_instance, "ShapelyPolygon", layout_params)

    @classmethod
    def from_nd_mask(
        cls,
        active_mask: NDArray[np.bool_],
        grid_edges: Tuple[NDArray[np.float64], ...],
        name: str = "",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
        """
        Create an Environment from a pre-defined N-D boolean mask and grid edges.

        This factory method allows for precise specification of active bins in
        an N-dimensional grid.

        Parameters
        ----------
        active_mask : NDArray[np.bool_]
            An N-dimensional boolean array where `True` indicates an active bin.
            The shape of this mask must correspond to the number of bins implied
            by `grid_edges` (i.e., `tuple(len(e)-1 for e in grid_edges)`).
        grid_edges : Tuple[NDArray[np.float64], ...]
            A tuple where each element is a 1D NumPy array of bin edge positions
            for that dimension, defining the underlying full grid.
        name : str, optional
            A name for the created environment. Defaults to "".
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.

        Returns
        -------
        Environment
            A new Environment instance with a `MaskedGridLayout`.

        """
        layout_params = {
            "active_mask": active_mask,
            "grid_edges": grid_edges,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }
        layout_instance = create_layout(kind="MaskedGrid", **layout_params)
        return cls(name, layout_instance, "MaskedGrid", layout_params)

    @classmethod
    def from_image_mask(
        cls,
        image_mask: NDArray[np.bool_],
        bin_size: Union[float, Tuple[float, float]] = 1.0,
        connect_diagonal_neighbors: bool = True,
        name: str = "",
    ) -> Environment:
        """
        Create a 2D Environment from a binary image mask.

        Each `True` pixel in the `image_mask` becomes an active bin in the
        environment. The `bin_size` determines the spatial scale of these pixels.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where `True` pixels define active bins.
        bin_size : Union[float, Tuple[float, float]], optional
            The spatial size of each pixel. If a float, pixels are square.
            If a tuple `(width, height)`, specifies pixel dimensions.
            Defaults to 1.0 (each pixel is 1x1 spatial unit).
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active pixel-bins.
            Defaults to True.
        name : str, optional
            A name for the created environment. Defaults to "".

        Returns
        -------
        Environment
            A new Environment instance with an `ImageMaskLayout`.

        """

        layout_params = {
            "image_mask": image_mask,
            "bin_size": bin_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }

        layout_instance = create_layout(kind="ImageMask", **layout_params)
        return cls(name, layout_instance, "ImageMask", layout_params)

    # Fallback factory for advanced use or deserialization
    @classmethod
    def from_custom_layout(
        cls,
        layout_type: str,
        layout_params: Dict[str, Any],
        name: str = "",
    ) -> Environment:
        """
        Create an Environment with a specified layout type and its build parameters.

        This is an advanced factory method primarily used for deserialization or
        when the layout construction logic is handled externally.

        Parameters
        ----------
        layout_type : str
            The string identifier of the `LayoutEngine` to use
            (e.g., "RegularGrid", "Hexagonal").
        layout_params : Dict[str, Any]
            A dictionary of parameters that will be passed to the `build`
            method of the chosen `LayoutEngine`.
        name : str, optional
            A name for the created environment. Defaults to "".

        Returns
        -------
        Environment
            A new Environment instance.

        """
        layout_instance = create_layout(kind=layout_type, **layout_params)
        return cls(name, layout_instance, layout_type, layout_params)

    @property
    def is_1d(self) -> bool:
        """
        Indicate if the environment's layout is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the underlying `LayoutEngine` (`self.layout`) reports
            itself as 1-dimensional (e.g., `GraphLayout`), False otherwise.
            This is determined by `self.layout.is_1d`.

        """
        return self._is_1d_env

    @property
    @check_fitted
    def n_dims(self) -> int:
        """
        Return the number of spatial dimensions of the active bin centers.

        Returns
        -------
        int
            The number of dimensions (e.g., 1 for a line, 2 for a plane).
            Derived from the shape of `self.bin_centers_`.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted or if `bin_centers_`
            is not available.
        """
        return self.bin_centers_.shape[1]

    @check_fitted
    def get_connectivity_graph(self) -> nx.Graph:
        """
        Return the primary connectivity graph of active bins.

        Nodes in this graph are integers from `0` to `N-1`, where `N` is the
        number of active bins. These node IDs directly correspond to the row
        indices in `self.bin_centers_`. Edges connect adjacent active bins.

        Returns
        -------
        nx.Graph
            The connectivity graph. Node attributes typically include 'pos'
            (N-D coordinates), 'source_grid_flat_index', and
            'original_grid_nd_index'. Edge attributes typically include
            'distance'.

        Raises
        ------
        ValueError
            If the connectivity graph is not available (e.g., not yet built).
        RuntimeError
            If called before the environment is fitted.

        """
        if self.connectivity_graph_ is None:
            raise ValueError("Connectivity graph is not available.")
        return self.connectivity_graph_

    @cached_property
    @check_fitted
    def distance_between_bins(self) -> NDArray[np.float64]:
        """
        Compute shortest path distances between all pairs of active bins.

        The distance is calculated using the `connectivity_graph_`, where
        edge weights typically represent the Euclidean distance between
        connected bin centers.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins, n_active_bins)
            A matrix where element `(i, j)` is the shortest path distance
            between active bin `i` and active bin `j`. `np.inf` indicates
            no path.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return _get_distance_between_bins(self.get_connectivity_graph())

    @check_fitted
    def get_bin_ind(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map N-dimensional continuous points to discrete active bin indices.

        This method delegates to the `point_to_bin_index` method of the
        underlying `LayoutEngine`.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            An array of active bin indices (0 to `n_active_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin (e.g., it's outside the environment).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return self.layout.point_to_bin_index(points_nd)

    @check_fitted
    def is_point_active(self, points_nd: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Check if N-dimensional continuous points fall within any active bin.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to check.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            A boolean array where `True` indicates the corresponding point
            maps to an active bin, and `False` indicates it does not.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return self.get_bin_ind(points_nd) != -1

    @check_fitted
    def get_bin_neighbors(self, bin_index: int) -> List[int]:
        """
        Find indices of neighboring active bins for a given active bin index.

        This method delegates to the `get_bin_neighbors` method of the
        underlying `LayoutEngine`, which typically uses the `connectivity_graph_`.

        Parameters
        ----------
        bin_index : int
            The index (0 to `n_active_bins - 1`) of the active bin for which
            to find neighbors.

        Returns
        -------
        List[int]
            A list of active bin indices that are neighbors to `bin_index`.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return self.layout.get_bin_neighbors(bin_index)

    @check_fitted
    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """
        Calculate the area (for 2D) or volume (for 3D+) of each active bin.

        For 1D environments, this typically returns the length of each bin.
        This method delegates to the `get_bin_area_volume` method of the
        underlying `LayoutEngine`.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array containing the area/volume/length of each active bin.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return self.layout.get_bin_area_volume()

    @check_fitted
    def get_bin_attributes_dataframe(self) -> pd.DataFrame:  # Renamed
        """
        Create a Pandas DataFrame with attributes of each active bin.

        The DataFrame is constructed from the node data of the
        `connectivity_graph_`. Each row corresponds to an active bin.
        Columns include the bin's N-D position (split into `pos_dim0`,
        `pos_dim1`, etc.) and any other attributes stored on the graph nodes.

        Returns
        -------
        pd.DataFrame
            A DataFrame where the index is `active_bin_id` (0 to `n_active_bins - 1`)
            and columns contain bin attributes.

        Raises
        ------
        ValueError
            If there are no active bins in the environment.
        RuntimeError
            If called before the environment is fitted.
        """
        graph = self.get_connectivity_graph()
        if graph.number_of_nodes() == 0:
            raise ValueError("No active bins in the environment.")

        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "active_bin_id"  # Index is 0..N-1

        if "pos" in df.columns and not df["pos"].dropna().empty:
            # Ensure all 'pos' are consistently tuples/lists before converting
            df["pos"] = df["pos"].apply(lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else (np.nan,) * self.n_dims)  # type: ignore

            pos_df = pd.DataFrame(df["pos"].tolist(), index=df.index)
            pos_df.columns = [f"pos_dim{i}" for i in range(pos_df.shape[1])]
            df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        return df

    @check_fitted
    def get_manifold_distances(
        self, points1: NDArray[np.float64], points2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculate manifold distances between pairs of points.

        Points are first mapped to their respective active bin indices.
        The manifold distance is then the shortest path distance between these
        bins, retrieved from `self.distance_between_bins`.

        Parameters
        ----------
        points1 : NDArray[np.float64], shape (n_pairs, n_dims) or (n_dims,)
            First set of N-dimensional points.
        points2 : NDArray[np.float64], shape (n_pairs, n_dims) or (n_dims,)
            Second set of N-dimensional points. Must have the same shape as `points1`.

        Returns
        -------
        NDArray[np.float64], shape (n_pairs,)
            Array of manifold distances. If a point in a pair does not map to
            an active bin, or if the bins are disconnected, the distance
            will be `np.inf`.

        Raises
        ------
        ValueError
            If `points1` and `points2` have mismatched shapes or if input arrays
            are empty but dimensions are incompatible.
        RuntimeError
            If called before the environment is fitted.
        """
        p1, p2 = np.atleast_2d(points1), np.atleast_2d(points2)
        if p1.shape != p2.shape:
            raise ValueError("Shape mismatch.")
        if p1.shape[0] == 0:
            return np.array([], dtype=np.float64)
        bin1, bin2 = self.get_bin_ind(p1), self.get_bin_ind(p2)
        dist_matrix = self.distance_between_bins
        n_active_bins_in_matrix = dist_matrix.shape[0]
        distances = np.full(len(p1), np.inf, dtype=np.float64)
        valid_mask = (
            (bin1 != -1)
            & (bin2 != -1)
            & (bin1 < n_active_bins_in_matrix)
            & (bin2 < n_active_bins_in_matrix)
        )
        if np.any(valid_mask):
            distances[valid_mask] = dist_matrix[bin1[valid_mask], bin2[valid_mask]]
        # Warning for out-of-bounds indices can be simplified as get_bin_ind now returns relative to active.
        if np.any(
            ((bin1 != -1) & ~((bin1 < n_active_bins_in_matrix) & (bin1 >= 0)))
            | ((bin2 != -1) & ~((bin2 < n_active_bins_in_matrix) & (bin2 >= 0)))
        ):
            warnings.warn(
                "Some bin indices from get_bin_ind were out of bounds for distance_between_bins matrix. This is unexpected.",
                RuntimeWarning,
            )
        return distances.squeeze()

    @check_fitted
    def get_shortest_path(
        self, source_active_bin_idx: int, target_active_bin_idx: int
    ) -> List[int]:
        """
        Find the shortest path between two active bins.

        The path is a sequence of active bin indices (0 to n_active_bins - 1)
        connecting the source to the target. Path calculation uses the
        'distance' attribute on the edges of the `connectivity_graph_`
        as weights.

        Parameters
        ----------
        source_active_bin_idx : int
            The active bin index (0 to n_active_bins - 1) for the start of the path.
        target_active_bin_idx : int
            The active bin index (0 to n_active_bins - 1) for the end of the path.

        Returns
        -------
        List[int]
            A list of active bin indices representing the shortest path from
            source to target. The list includes both the source and target indices.
            Returns an empty list if the source and target are the same, or if
            no path exists, or if nodes are not found.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        nx.NodeNotFound
            If `source_active_bin_idx` or `target_active_bin_idx` is not
            a node in the `connectivity_graph_`.
        """
        graph = self.get_connectivity_graph()

        if source_active_bin_idx == target_active_bin_idx:
            return [source_active_bin_idx]

        try:
            path = nx.shortest_path(
                graph,
                source=source_active_bin_idx,
                target=target_active_bin_idx,
                weight="distance",
            )
            return path
        except nx.NetworkXNoPath:
            warnings.warn(
                f"No path found between active bin {source_active_bin_idx} "
                f"and {target_active_bin_idx}.",
                UserWarning,
            )
            return []
        except nx.NodeNotFound as e:
            # Re-raise if the user provides an invalid node index for active bins
            raise nx.NodeNotFound(
                f"Node not found in connectivity graph: {e}. "
                "Ensure source/target indices are valid active bin indices."
            ) from e

    @check_fitted
    def get_linearized_coordinate(
        self, points_nd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convert N-dimensional points to 1D linearized coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `get_linearized_coordinate` method.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            N-dimensional points to linearize.

        Returns
        -------
        NDArray[np.float64], shape (n_points,)
            1D linearized coordinates corresponding to the input points.

        Raises
        ------
        TypeError
            If the environment is not 1D or not based on a `GraphLayout`.
        RuntimeError
            If called before the environment is fitted.
        """
        if not self.is_1d or not isinstance(self.layout, GraphLayout):
            raise TypeError("Linearized coordinate only for GraphLayout environments.")
        return self.layout.get_linearized_coordinate(points_nd)

    @check_fitted
    def map_linear_to_grid_coordinate(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convert 1D linearized coordinates back to N-dimensional coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `map_linear_to_grid_coordinate` method.

        Parameters
        ----------
        linear_coordinates : NDArray[np.float64], shape (n_points,)
            1D linearized coordinates to map to N-D space.

        Returns
        -------
        NDArray[np.float64], shape (n_points, n_dims)
            N-dimensional coordinates corresponding to the input linear coordinates.

        Raises
        ------
        TypeError
            If the environment is not 1D or not based on a `GraphLayout`.
        RuntimeError
            If called before the environment is fitted.
        """
        if not self.is_1d or not isinstance(self.layout, GraphLayout):
            raise TypeError("Mapping linear to N-D only for GraphLayout environments.")
        return self.layout.map_linear_to_grid_coordinate(linear_coordinates)

    @check_fitted
    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        show_regions: bool = False,
        layout_plot_kwargs: Optional[Dict[str, Any]] = None,
        regions_plot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Plot the environment's layout and optionally defined regions.

        This method delegates plotting of the base layout to the `plot` method
        of the underlying `LayoutEngine`. If `show_regions` is True, it then
        overlays any defined spatial regions managed by `self.regions`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        show_regions : bool, optional
            If True, plot defined spatial regions on top of the layout.
            Defaults to False.
        layout_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the `layout.plot()` method.
            Defaults to None.
        regions_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the `regions.plot_regions()` method.
            Defaults to None.
        **kwargs : Any
            Additional keyword arguments that are passed to `layout.plot()`.
            These can be overridden by `layout_plot_kwargs`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the environment was plotted.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

        ax = self.layout.plot(ax=ax, **l_kwargs)

        if show_regions and hasattr(self, "regions") and self.regions is not None:
            r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
            self.regions.plot_regions(ax=ax, **r_kwargs)

        plot_title = self.name
        if (
            self.layout
            and hasattr(self.layout, "_layout_type_tag")
            and self.layout._layout_type_tag
        ):
            plot_title += f" ({self.layout._layout_type_tag})"

        # Only set title if layout.plot didn't set one or user didn't pass one via kwargs to layout.plot
        # This is hard to check perfectly. A common convention is for plotting functions to not
        # override titles if the axes already has one.
        # For simplicity, if ax.get_title() is empty, set it.
        if ax.get_title() == "":
            ax.set_title(plot_title)

        return ax

    def plot_1D(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        layout_plot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Plot a 1D representation of the environment, if applicable.

        This method is primarily for environments where `is_1d` is True
        (e.g., using `GraphLayout`). It calls the `plot_linear_layout`
        method of the underlying layout if it exists and the layout is 1D.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        layout_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the layout's 1D plotting method.
        **kwargs : Any
            Additional keyword arguments passed to the layout's 1D plotting method.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the 1D layout was plotted, or the original `ax`
            if plotting was not applicable.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        AttributeError
            If `self.layout.is_1d` is True but the layout does not have a
            `plot_linear_layout` method.
        """
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
        if self.layout.is_1d:
            if hasattr(self.layout, "plot_linear_layout"):
                ax = self.layout.plot_linear_layout(ax=ax, **l_kwargs)  # type: ignore
            else:
                warnings.warn(
                    f"Layout '{self._layout_type_used}' is 1D but does not "
                    "have a 'plot_linear_layout' method. Skipping 1D plot.",
                    UserWarning,
                )
        else:
            warnings.warn(
                "Environment is not 1D. Skipping 1D plot. Use regular plot() method.",
                UserWarning,
            )

        return ax

    def save(self, filename: str = "environment.pkl") -> None:
        """
        Save the Environment object to a file using pickle.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the environment to.
            Defaults to "environment.pkl".
        """
        with open(filename, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Environment saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> Environment:
        """
        Load an Environment object from a pickled file.

        Parameters
        ----------
        filename : str
            The name of the file to load the environment from.

        Returns
        -------
        Environment
            The loaded Environment object.

        Raises
        ------
        TypeError
            If the loaded object is not an instance of the Environment class.
        """
        with open(filename, "rb") as fh:
            environment = pickle.load(fh)
        if not isinstance(environment, cls):
            raise TypeError(f"Loaded object is not type {cls.__name__}")
        return environment

    # --- Serialization ---
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the Environment object to a dictionary.

        This method captures the essential information needed to reconstruct
        the Environment, including its name, the type of layout used, the
        parameters for that layout, and any defined regions.
        Complex objects within `_layout_params_used` (like NetworkX graphs or
        Shapely polygons) are converted to serializable formats.

        Returns
        -------
        Dict[str, Any]
            A dictionary representing the Environment object.
            Keys include:
            - "__classname__": Name of the class.
            - "__module__": Module where the class is defined.
            - "name": Name of the environment.
            - "_layout_type_used": Type of layout engine.
            - "_layout_params_used": Parameters for the layout engine,
              with complex objects serialized.
            - "_regions_data": Serialized data for defined regions.
        """
        # Ensure complex objects in _layout_params_used are serializable
        # For example, nx.Graph for GraphLayout should be converted to node_link_data
        serializable_layout_params = self._layout_params_used.copy()
        if (
            self._layout_type_used == "Graph"
            and "graph_definition" in serializable_layout_params
        ):
            if isinstance(serializable_layout_params["graph_definition"], nx.Graph):
                serializable_layout_params["graph_definition"] = nx.node_link_data(
                    serializable_layout_params["graph_definition"]
                )
        # Add similar handling for Shapely Polygons if they are in params
        if (
            self._layout_type_used == "ShapelyPolygon"
            and "polygon" in serializable_layout_params
        ):
            if _HAS_SHAPELY and isinstance(serializable_layout_params["polygon"], _shp.Polygon):  # type: ignore
                # Convert Shapely polygon to a serializable format, e.g., WKT or GeoJSON interface
                try:
                    serializable_layout_params["polygon"] = {
                        "__shapely_polygon__": True,
                        "exterior_coords": list(
                            serializable_layout_params["polygon"].exterior.coords
                        ),
                        "interior_coords_list": [
                            list(interior.coords)
                            for interior in serializable_layout_params[
                                "polygon"
                            ].interiors
                        ],
                    }
                except Exception:
                    warnings.warn(
                        "Could not serialize Shapely polygon in layout_params.",
                        UserWarning,
                    )

        data = {
            "__classname__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "name": self.name,
            "_layout_type_used": self._layout_type_used,
            "_layout_params_used": serializable_layout_params,
            "_regions_data": [asdict(info) for info in self.regions.values()],
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Environment:
        """
        Deserialize an Environment object from a dictionary.

        This method reconstructs an Environment instance from a dictionary
        created by `to_dict`. It handles the deserialization of complex
        objects like NetworkX graphs and Shapely polygons stored within
        the layout parameters and region data.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary representing the Environment object, typically
            generated by `to_dict()`.

        Returns
        -------
        Environment
            A new Environment instance reconstructed from the dictionary.

        Raises
        ------
        ValueError
            If the dictionary is not for this Environment class or if
            essential information like `_layout_type_used` is missing.
        """
        if not (
            data.get("__classname__") == cls.__name__
            and data.get("__module__") == cls.__module__
        ):
            raise ValueError("Dictionary is not for this Environment class.")

        env_name = data["name"]
        layout_type = data.get("_layout_type_used")
        layout_params = data.get("_layout_params_used", {})

        if not layout_type:
            raise ValueError("Cannot deserialize: _layout_type_used missing.")

        # Deserialize complex objects in layout_params
        if layout_type == "Graph" and "graph_definition" in layout_params:
            if isinstance(
                layout_params["graph_definition"], dict
            ):  # Assuming node_link_data
                layout_params["graph_definition"] = nx.node_link_graph(
                    layout_params["graph_definition"]
                )

        if layout_type == "ShapelyPolygon" and "polygon" in layout_params:
            if isinstance(layout_params["polygon"], dict) and layout_params[
                "polygon"
            ].get("__shapely_polygon__"):
                if _HAS_SHAPELY:
                    try:
                        shell = layout_params["polygon"]["exterior_coords"]
                        holes = layout_params["polygon"]["interior_coords_list"]
                        layout_params["polygon"] = _shp.Polygon(shell, holes if holes else None)  # type: ignore
                    except Exception:
                        warnings.warn(
                            "Could not deserialize Shapely polygon from layout_params.",
                            UserWarning,
                        )
                        layout_params["polygon"] = None  # Or raise error
                else:
                    warnings.warn(
                        "Shapely not available, cannot deserialize polygon for ShapelyPolygonLayout.",
                        UserWarning,
                    )
                    layout_params["polygon"] = None  # Or skip this layout type

        # Use from_custom_layout which calls create_layout
        env = cls.from_custom_layout(
            name=env_name,
            layout_type=layout_type,
            layout_params=layout_params,
        )

        # Restore regions
        if "_regions_data" in data and data["_regions_data"] is not None:
            for region_info_data in data["_regions_data"]:
                # Handle polygon deserialization for regions too
                if (
                    region_info_data["kind"] == "polygon"
                    and isinstance(region_info_data["data"], dict)
                    and region_info_data["data"].get("__shapely_polygon__")
                ):
                    if _HAS_SHAPELY:
                        try:
                            shell = region_info_data["data"]["exterior_coords"]
                            holes = region_info_data["data"]["interior_coords_list"]
                            region_info_data["data"] = _shp.Polygon(shell, holes if holes else None)  # type: ignore
                        except Exception:
                            warnings.warn(
                                f"Could not deserialize polygon in region '{region_info_data['name']}'.",
                                UserWarning,
                            )
                            continue
                    else:
                        warnings.warn(
                            f"Shapely not available, cannot deserialize polygon region '{region_info_data['name']}'.",
                            UserWarning,
                        )
                        continue

                # Ensure 'metadata' key exists if RegionInfo expects it
                if "metadata" not in region_info_data:
                    region_info_data["metadata"] = {}

                try:
                    env.regions.add(
                        name=region_info_data["name"],
                        kind=region_info_data["kind"],
                        **{region_info_data["kind"]: region_info_data["data"]},
                        **region_info_data.get("metadata", {}),
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to add region '{region_info_data['name']}' during deserialization: {e}",
                        UserWarning,
                    )

        return env

    @check_fitted
    def flat_to_grid_bin_index(
        self, flat_indices: Union[int, NDArray[np.int_]]
    ) -> Union[Tuple[int, ...], Tuple[NDArray[np.int_], ...]]:
        """
        Convert active bin flat indices (0..N-1) to N-D grid indices.

        This method translates a flat index (which refers to an active bin,
        e.g., a row in `self.bin_centers_`) back to its original N-dimensional
        index within the environment's full conceptual grid. This is primarily
        meaningful for grid-based layouts.

        Parameters
        ----------
        flat_indices : Union[int, NDArray[np.int_]]
            A single flat index or an array of flat indices for active bins.
            These indices range from `0` to `n_active_bins - 1`.

        Returns
        -------
        Union[Tuple[int, ...], Tuple[NDArray[np.int_], ...]]
            If input is a single int: A tuple of N integers representing the
            N-D grid index (e.g., `(row, col, ...)`). Contains `np.nan` if
            conversion fails for an index.
            If input is an array: A tuple of N arrays, where each array
            contains the indices for one dimension. (e.g., `(rows_array, cols_array, ...)`).
            Contains `np.nan` for failed conversions.

        Raises
        ------
        TypeError
            If the environment is not N-D grid-based (e.g., if it's 1D or
            lacks a clear N-D `grid_shape_` or `active_mask_`).
        RuntimeError
            If called before the environment is fitted or if the connectivity
            graph is unavailable.
        """
        if (
            self.grid_shape_ is None
            or len(self.grid_shape_) <= 1
            or self.active_mask_ is None
            or self.active_mask_.ndim <= 1
        ):
            raise TypeError(
                "N-D index conversion is primarily for N-D grid-based layouts "
                "with a defined N-D active_mask_ and grid_shape_."
            )
        if self.connectivity_graph_ is None:
            raise RuntimeError(
                "Connectivity graph not available for mapping source indices."
            )

        is_scalar = np.isscalar(flat_indices)
        flat_indices_arr = np.atleast_1d(np.asarray(flat_indices, dtype=int))

        output_nd_indices_list = []

        for active_flat_idx in flat_indices_arr:
            if not (0 <= active_flat_idx < self.connectivity_graph_.number_of_nodes()):
                warnings.warn(
                    f"Active flat_index {active_flat_idx} is out of bounds for connectivity_graph_ nodes. Returning NaNs.",
                    UserWarning,
                )
                output_nd_indices_list.append(tuple([np.nan] * len(self.grid_shape_)))
                continue

            node_data = self.connectivity_graph_.nodes[active_flat_idx]

            # Prefer 'original_grid_nd_index' if directly available
            if (
                "original_grid_nd_index" in node_data
                and node_data["original_grid_nd_index"] is not None
            ):
                output_nd_indices_list.append(node_data["original_grid_nd_index"])
            elif (
                "source_grid_flat_index" in node_data
            ):  # Fallback to unraveling source_grid_flat_index
                original_full_grid_flat_idx = node_data["source_grid_flat_index"]
                output_nd_indices_list.append(
                    tuple(
                        np.unravel_index(original_full_grid_flat_idx, self.grid_shape_)
                    )
                )
            else:
                warnings.warn(
                    f"Node {active_flat_idx} in connectivity_graph_ missing necessary source index information for N-D conversion. Returning NaNs.",
                    UserWarning,
                )
                output_nd_indices_list.append(tuple([np.nan] * len(self.grid_shape_)))

        if (
            not output_nd_indices_list
        ):  # Should not happen if flat_indices_arr was not empty
            return tuple(np.array([], dtype=int) for _ in range(len(self.grid_shape_)))  # type: ignore

        # Convert list of tuples to tuple of arrays
        final_output_nd_indices = tuple(
            np.array([item[d] for item in output_nd_indices_list])
            for d in range(len(self.grid_shape_))
        )

        if is_scalar:
            # For scalar input, return a tuple of ints/NaNs
            return tuple(val[0] if not np.isnan(val[0]) else np.nan for val in final_output_nd_indices)  # type: ignore
        return final_output_nd_indices

    @check_fitted
    def grid_to_flat_bin_index(
        self, *nd_idx_per_dim: Union[int, NDArray[np.int_]]
    ) -> Union[int, NDArray[np.int_]]:
        """
        Convert N-D grid indices to active bin flat indices (0..N-1).

        This method takes N-D indices (referring to the original full
        conceptual grid of the environment) and maps them to the
        corresponding flat index of an *active* bin (0 to `n_active_bins - 1`).
        If the N-D index is outside the grid bounds or maps to a bin that is
        not active, -1 is returned. This is primarily meaningful for
        grid-based layouts.

        Parameters
        ----------
        *nd_idx_per_dim : Union[int, NDArray[np.int_]]
            N arguments, one for each dimension of the grid. Each argument can
            be an integer (for a single point query) or a NumPy array of
            integers (for multiple points, must be broadcastable).
            Example: `env.grid_to_flat_bin_index(rows, cols)` for a 2D grid.
            Alternatively, can be a single tuple/list of N-D indices or
            arrays of N-D indices, e.g., `env.grid_to_flat_bin_index([(r1,c1), (r2,c2)])`
            or `env.grid_to_flat_bin_index(([r1,r2],[c1,c2]))`.

        Returns
        -------
        Union[int, NDArray[np.int_]]
            If input implies a single N-D point: An integer representing the
            active bin flat index, or -1 if not active/out of bounds.
            If input implies multiple N-D points: A NumPy array of active bin
            flat indices, with -1 for non-active/out-of-bounds points.

        Raises
        ------
        TypeError
            If the environment is not N-D grid-based.
        RuntimeError
            If called before the environment is fitted or if the connectivity
            graph is unavailable.
        ValueError
            If the number of N-D index arguments doesn't match the environment's
            dimensions, or if input arrays cannot be broadcast.
        """
        if (
            self.grid_shape_ is None
            or len(self.grid_shape_) <= 1
            or self.active_mask_ is None
            or self.active_mask_.ndim <= 1
        ):
            raise TypeError("N-D index conversion is for N-D grid-based layouts.")
        if self.connectivity_graph_ is None:
            raise RuntimeError(
                "Connectivity graph not available for mapping to active indices."
            )

        # Allow caller to pass a single iterable (tuple/list of arrays/ints) or separate arrays/ints
        nd_indices_tuple: Tuple[NDArray[np.int_], ...]
        if (
            len(nd_idx_per_dim) == 1
            and isinstance(nd_idx_per_dim[0], (list, tuple))
            and not np.isscalar(nd_idx_per_dim[0][0])
        ):
            # Input like env.grid_to_flat_bin_index( ([r1,r2],[c1,c2]) ) or env.grid_to_flat_bin_index( ( (r1,c1), (r2,c2) ) )
            # The latter needs to be transposed if it's (n_points, n_dims)
            temp_input = np.asarray(nd_idx_per_dim[0])
            if (
                temp_input.ndim == 2
                and temp_input.shape[0] == len(self.grid_shape_)
                and temp_input.shape[1] > 0
            ):  # (n_dims, n_points)
                nd_indices_tuple = tuple(
                    temp_input[d] for d in range(len(self.grid_shape_))
                )
            elif temp_input.ndim == 2 and temp_input.shape[1] == len(
                self.grid_shape_
            ):  # (n_points, n_dims)
                nd_indices_tuple = tuple(
                    temp_input[:, d] for d in range(len(self.grid_shape_))
                )
            elif temp_input.ndim == 1 and len(temp_input) == len(
                self.grid_shape_
            ):  # Single N-D index as a list/tuple
                nd_indices_tuple = tuple(
                    np.array([val]) for val in temp_input
                )  # Make each a 1-element array
            else:
                raise ValueError("Invalid format for single argument N-D index.")
        else:  # Separate arguments per dimension
            nd_indices_tuple = tuple(
                np.atleast_1d(np.asarray(idx, dtype=int)) for idx in nd_idx_per_dim
            )

        if len(nd_indices_tuple) != len(self.grid_shape_):
            raise ValueError(
                f"Expected {len(self.grid_shape_)} N-D indices, got {len(nd_indices_tuple)}"
            )

        # Determine output shape based on input arrays (assuming they broadcast correctly)
        try:
            common_shape = np.broadcast(*nd_indices_tuple).shape
        except ValueError:
            raise ValueError("N-D index arrays could not be broadcast together.")

        output_flat_indices = np.full(common_shape, -1, dtype=int)

        # Create a mask for N-D indices that are within the full grid bounds
        in_bounds_mask = np.ones(common_shape, dtype=bool)
        for dim_idx, dim_coords in enumerate(nd_indices_tuple):
            in_bounds_mask &= (dim_coords >= 0) & (
                dim_coords < self.grid_shape_[dim_idx]
            )

        if not np.any(in_bounds_mask):
            return (
                output_flat_indices[0]
                if np.isscalar(nd_idx_per_dim[0])
                and len(nd_idx_per_dim) == len(self.grid_shape_)
                and common_shape == (1,)
                else output_flat_indices
            )

        # Get the N-D indices that are in bounds
        valid_nd_indices_tuple = tuple(idx[in_bounds_mask] for idx in nd_indices_tuple)

        # Check if these in-bounds N-D grid cells are active
        are_these_bins_active_on_full_grid = self.active_mask_[valid_nd_indices_tuple]

        # Further filter to only N-D indices that are both in-bounds AND active
        truly_active_nd_indices_tuple = tuple(
            dim_coords[are_these_bins_active_on_full_grid]
            for dim_coords in valid_nd_indices_tuple
        )

        if (
            truly_active_nd_indices_tuple[0].size == 0
        ):  # No active bins found for given N-D indices
            return (
                output_flat_indices[0]
                if np.isscalar(nd_idx_per_dim[0])
                and len(nd_idx_per_dim) == len(self.grid_shape_)
                and common_shape == (1,)
                else output_flat_indices
            )

        # Convert these N-D indices (of active bins on full grid) to original_full_grid_flat_indices
        original_flat_indices_of_targets = np.ravel_multi_index(
            truly_active_nd_indices_tuple, self.grid_shape_
        )

        final_active_bin_ids = np.array(
            [
                self._source_flat_to_active_node_id_map.get(orig_flat_idx, -1)
                for orig_flat_idx in original_flat_indices_of_targets
            ],
            dtype=np.int_,
        )

        # Place these final_active_bin_ids back into the original output shape
        # The `in_bounds_mask` needs to be further filtered by `are_these_bins_active_on_full_grid`
        # to correctly map back.
        final_placement_mask = np.zeros_like(in_bounds_mask)
        final_placement_mask[in_bounds_mask] = are_these_bins_active_on_full_grid

        output_flat_indices[final_placement_mask] = final_active_bin_ids

        # Handle scalar input case for return type
        if (
            np.isscalar(nd_idx_per_dim[0])
            and len(nd_idx_per_dim) == len(self.grid_shape_)
            and common_shape == (1,)
        ):
            return int(output_flat_indices[0])
        return output_flat_indices

    @check_fitted
    def get_boundary_bin_indices(
        self: "Environment", connectivity_threshold_factor: Optional[float] = None
    ) -> NDArray[np.int_]:
        # ... (graph and initial checks remain the same) ...
        graph = self.get_connectivity_graph()
        if graph.number_of_nodes() == 0:
            return np.array([], dtype=int)

        boundary_bin_indices = []

        # Grid-specific logic for N-D grids (N > 1)
        is_nd_grid_layout_with_mask = (
            hasattr(self, "active_mask_")
            and self.active_mask_ is not None
            and hasattr(self, "grid_shape_")
            and self.grid_shape_ is not None
            and len(self.grid_shape_) > 1  # Explicitly for N-D (N>1) grids
        )

        if is_nd_grid_layout_with_mask and connectivity_threshold_factor is None:
            # ... (existing N-D grid boundary detection logic remains the same) ...
            active_mask_nd = self.active_mask_
            grid_shape_nd = self.grid_shape_
            n_dims = len(grid_shape_nd)

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

                        if not (0 <= neighbor_nd_idx[dim_idx] < grid_shape_nd[dim_idx]):
                            is_boundary = True
                            break
                        if not active_mask_nd[neighbor_nd_idx]:
                            is_boundary = True
                            break
                    if is_boundary:
                        break
                if is_boundary:
                    boundary_bin_indices.append(active_node_id)
        else:
            # Degree-based heuristic for:
            # - Non-grid layouts
            # - 1D grid layouts (where len(grid_shape_) == 1)
            # - N-D grid layouts if connectivity_threshold_factor is provided
            # - N-D grid layouts if 'original_grid_nd_index' is missing for nodes

            degrees = dict(graph.degree())
            if not degrees:
                return np.array([], dtype=int)

            all_degree_values = list(degrees.values())
            median_degree = np.median(all_degree_values)
            max_degree_val = np.max(all_degree_values) if all_degree_values else 0

            threshold_degree: float
            if connectivity_threshold_factor is not None:
                if connectivity_threshold_factor <= 0:  # pragma: no cover
                    raise ValueError("connectivity_threshold_factor must be positive.")
                threshold_degree = connectivity_threshold_factor * median_degree
            else:
                # Default heuristics if no factor provided
                if self._layout_type_used == "Hexagonal" and median_degree > 5:
                    threshold_degree = 5.5  # Bins with < 6 neighbors
                elif (  # For 1D grids or path-like graphs
                    hasattr(self, "grid_shape_")
                    and self.grid_shape_ is not None
                    and len(self.grid_shape_) == 1
                ) or (self.is_1d and isinstance(self.layout, GraphLayout)):
                    # For linear structures, ends typically have degree 1, internal degree 2
                    # Threshold just below typical internal degree.
                    threshold_degree = (
                        1.5 if max_degree_val >= 2 else max_degree_val
                    )  # Catches degree 1
                elif is_nd_grid_layout_with_mask and median_degree > (
                    2 * len(self.grid_shape_) - 1
                ):  # For N>1 grids when falling back
                    threshold_degree = 2 * len(self.grid_shape_) - 0.5
                else:  # General fallback if no specific layout recognized or no factor
                    threshold_degree = (
                        median_degree  # Bins with degree < median are boundary
                    )

            for node_id, degree in degrees.items():
                if degree < threshold_degree:
                    boundary_bin_indices.append(node_id)

            # If the above found nothing, and it's not a specific grid case where that's expected,
            # a simple fallback for general graphs: degree < max_degree
            if (
                not boundary_bin_indices
                and connectivity_threshold_factor is None
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

    @check_fitted
    def get_graph_layout_properties_for_linearization(
        self: "Environment",
    ) -> Optional[Dict[str, Any]]:
        """
        If the environment uses a GraphLayout, returns properties needed
        for linearization using the `track_linearization` library.

        These properties are typically passed to `track_linearization.get_linearized_position`.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary with keys 'track_graph', 'edge_order', 'edge_spacing'
            if the layout is `GraphLayout` and parameters are available.
            Returns `None` otherwise. The 'track_graph' herein refers to the
            original graph definition used to build the `GraphLayout`.
        """
        if isinstance(self.layout, GraphLayout):
            # _layout_params_used stores the kwargs passed to the layout's build method
            graph_def = self._layout_params_used.get("graph_definition")
            edge_order = self._layout_params_used.get("edge_order")
            # edge_spacing can be 0.0, so check for None explicitly if it's optional
            edge_spacing = self._layout_params_used.get("edge_spacing")

            if (
                graph_def is not None
                and edge_order is not None
                and edge_spacing is not None
            ):
                return {
                    "track_graph": graph_def,
                    "edge_order": edge_order,
                    "edge_spacing": edge_spacing,
                }
            else:
                warnings.warn(
                    "GraphLayout instance is missing some expected build parameters "
                    "('graph_definition', 'edge_order', or 'edge_spacing') "
                    "in _layout_params_used.",
                    UserWarning,
                )
        return None

    @check_fitted
    def get_indices_for_values_in_region(
        self: "Environment", values: NDArray[np.float64], region_name: str
    ) -> NDArray[np.bool_]:
        """
        Determines which time points (rows in `values` array) correspond
        to locations falling within a specified named region of this environment.

        A value is considered in the region if it maps to an active bin
        that is part of the defined region.

        Parameters
        ----------
        values : NDArray[np.float64]
            N-dimensional position data, shape (n_time_points, n_dims).
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask of shape (n_time_points,). `True` indicates the
            position at that time point is within the specified region.

        Raises
        ------
        AttributeError
            If the environment does not have a `regions` (RegionManager) attribute.
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If `values` array has incorrect dimensionality.
        """
        if not hasattr(self, "regions") or self.regions is None:
            raise AttributeError(f"Environment '{self.name}' has no 'regions' manager.")
        if values.ndim != 2 or values.shape[1] != self.n_dims:
            raise ValueError(
                f"Values array must be 2D with shape (n_points, {self.n_dims}), "
                f"got {values.shape}."
            )

        # Map continuous positions to active bin indices (0 to N-1, or -1 if outside)
        # `get_bin_ind` is already @check_fitted
        active_bin_indices_for_positions = self.get_bin_ind(values)

        # Get the set of active bin indices that constitute the target region
        # `bins_in_region` also raises KeyError if region_name not found
        # and is @check_fitted via region_mask
        try:
            bins_in_target_region = self.regions.bins_in_region(region_name)
        except KeyError:
            # Re-raise with more context or let original error propagate
            raise KeyError(
                f"Region '{region_name}' not found in environment '{self.name}'."
            )

        if bins_in_target_region.size == 0:
            # Region is defined but contains no active bins from this environment
            return np.zeros(values.shape[0], dtype=bool)

        # Create a boolean mask: True if the mapped active_bin_index for a position
        # is present in the set of bins defining the target region.
        # np.isin is efficient for this.
        is_in_region_mask = np.isin(
            active_bin_indices_for_positions, bins_in_target_region
        )

        return is_in_region_mask

    @check_fitted
    def bins_in_region(self, region_name: str) -> NDArray[np.int_]:
        """
        Get active bin indices that fall within a specified named region.

        Parameters
        ----------
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.int_]
            Array of active bin indices (0 to n_active_bins - 1)
            that are part of the region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region kind is unsupported or mask dimensions mismatch.
        """
        if not hasattr(self, "regions") or self.regions is None:  # Should always exist
            # This case should ideally not be reached if __init__ guarantees self.regions
            raise AttributeError(
                f"Environment '{self.name}' has no 'regions' manager."
            )  # pragma: no cover

        region_info = self.regions[region_name]  # Can raise KeyError

        if region_info.kind == "point":
            point_nd = np.asarray(region_info.data).reshape(1, -1)
            if point_nd.shape[1] != self.n_dims:
                raise ValueError(
                    f"Region point dimension {point_nd.shape[1]} "
                    f"does not match environment dimension {self.n_dims}."
                )
            bin_idx = self.get_bin_ind(point_nd)
            return bin_idx[bin_idx != -1]

        elif region_info.kind == "polygon":
            if not _HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon region queries require 'shapely'.")
            if self.n_dims != 2:  # pragma: no cover
                raise ValueError(
                    "Polygon regions are only supported for 2D environments."
                )

            polygon = region_info.data
            contained_mask = np.array(
                [polygon.contains(_shp.Point(center)) for center in self.bin_centers_],
                dtype=bool,
            )
            return np.flatnonzero(contained_mask)

        elif region_info.kind == "mask":
            mask_data = np.asarray(region_info.data, dtype=bool)

            if (
                self.grid_shape_ is None or self.active_mask_ is None
            ):  # pragma: no cover
                raise ValueError(
                    "Mask regions are only supported for environments with "
                    "a defined grid_shape and active_mask (typically grid-based layouts)."
                )
            if mask_data.shape != self.grid_shape_:
                raise ValueError(
                    f"Region mask shape {mask_data.shape} does not match "
                    f"environment grid_shape {self.grid_shape_}."
                )

            # Mask from region applies to the *full original grid*
            # We need active environment bins whose original grid positions fall in this mask
            effective_mask_on_full_grid = mask_data & self.active_mask_

            original_flat_indices_in_region_and_active = np.flatnonzero(
                effective_mask_on_full_grid
            )

            if not original_flat_indices_in_region_and_active.size:
                return np.array([], dtype=int)

            # Map these original full grid flat indices to active bin IDs
            # Need to ensure _source_flat_to_active_node_id_map is populated
            if (
                self._source_flat_to_active_node_id_map is None
                or not self._source_flat_to_active_node_id_map
            ):  # Check if empty
                # This can happen if connectivity_graph_ has no 'source_grid_flat_index'
                # For example, if it's a non-grid layout that doesn't add it.
                # Or if all nodes are missing it.
                # If the layout is a grid, this map should exist.
                # If it's a GraphLayout, its source_grid_flat_index refers to linearized bins.
                # The "mask" kind for GraphLayout means the mask_data should be 1D.
                if (
                    self.is_1d
                    and mask_data.ndim == 1
                    and len(mask_data) == self.grid_shape_[0]
                ):
                    # For GraphLayout, source_grid_flat_index is 0..n_linear_bins-1
                    # and these are also the active_node_ids if all linearized bins are nodes.
                    # The active_mask_ for GraphLayout is all True, shape (n_linear_bins,).
                    # So effective_mask_on_full_grid is just mask_data.
                    # The indices from np.flatnonzero(mask_data) are directly the active_bin_ids.
                    return np.flatnonzero(mask_data)  # These are the active node IDs
                else:
                    warnings.warn(
                        "Source flat to active node ID map is not available or not applicable. "
                        "Cannot map mask region for this layout type accurately without it.",
                        UserWarning,
                    )
                    return np.array([], dtype=int)

            active_bin_ids_in_region = [
                self._source_flat_to_active_node_id_map.get(orig_flat_idx, -1)
                for orig_flat_idx in original_flat_indices_in_region_and_active
            ]
            return np.array(
                [bid for bid in active_bin_ids_in_region if bid != -1], dtype=int
            )

        else:  # pragma: no cover
            raise ValueError(f"Unsupported region kind: {region_info.kind}")

    @check_fitted
    def region_mask(self, region_name: str) -> NDArray[np.bool_]:
        """
        Get a boolean mask over active bins indicating membership in a region.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array of shape (n_active_bins,). True if an active bin
            is part of the region.
        """
        active_bins_for_mask = self.bins_in_region(region_name)
        mask = np.zeros(self.bin_centers_.shape[0], dtype=bool)
        if active_bins_for_mask.size > 0:
            mask[active_bins_for_mask] = True
        return mask

    @check_fitted
    def region_center(self, region_name: str) -> Optional[NDArray[np.float64]]:
        """
        Calculate the center of a specified named region.

        - For 'point' regions, returns the point itself.
        - For 'polygon' regions, returns the centroid of the polygon.
        - For 'mask' regions, returns the mean of the bin_centers_ of the
          active bins included in the region.

        Returns
        -------
        Optional[NDArray[np.float64]]
            N-D coordinates of the region's center, or None if the region
            is empty or center cannot be determined.
        """
        region_info = self.regions[region_name]

        if region_info.kind == "point":
            return np.asarray(region_info.data)
        elif region_info.kind == "polygon":
            if not _HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon region queries require 'shapely'.")
            return np.array(region_info.data.centroid.coords[0])  # type: ignore
        elif region_info.kind == "mask":
            active_bin_ids = self.bins_in_region(region_name)
            if active_bin_ids.size == 0:
                return None
            return np.mean(self.bin_centers_[active_bin_ids], axis=0)
        return None  # pragma: no cover

    @check_fitted
    def get_region_area(self, region_name: str) -> float:
        """
        Calculate the area/volume of a specified named region.

        - For 'point' regions, area is 0.0.
        - For 'polygon' regions, uses Shapely's area.
        - For 'mask' regions, sums the area/volume of active bins in the region.
        """
        region_info = self.regions[region_name]

        if region_info.kind == "point":
            return 0.0
        elif region_info.kind == "polygon":
            if not _HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon area calculation requires 'shapely'.")
            return region_info.data.area  # type: ignore
        elif region_info.kind == "mask":
            active_bin_ids = self.bins_in_region(region_name)
            if active_bin_ids.size == 0:
                return 0.0
            bin_areas_volumes = self.get_bin_area_volume()
            return np.sum(bin_areas_volumes[active_bin_ids])
        return 0.0  # pragma: no cover

    @check_fitted
    def create_buffered_region(
        self,
        source_region_name_or_point: Union[str, NDArray[np.float64]],
        buffer_distance: float,
        new_region_name: str,
        **meta: Any,
    ) -> CoreRegion:
        """
        Creates a new polygon region by buffering an existing region or a point.
        The new region is added to `self.regions`.

        Parameters
        ----------
        source_region_name_or_point : Union[str, NDArray[np.float64]]
            If str, the name of an existing 'point' or 'polygon' region.
            If NDArray, the N-D coordinates of a point to buffer.
        buffer_distance : float
            The distance for the buffer operation.
        new_region_name : str
            Name for the newly created buffered region.
        **meta : Any
            Additional metadata for the new region.

        Returns
        -------
        CoreRegion
            The newly created and added Region object.

        Raises
        ------
        RuntimeError
            If Shapely is not installed.
        ValueError
            If source region is not 2D, or other issues.
        """
        if not _HAS_SHAPELY:  # pragma: no cover
            raise RuntimeError("Buffering requires Shapely.")

        # This method now delegates to self.regions (CoreRegions instance)
        # CoreRegions.buffer() already handles adding the region.
        return self.regions.buffer(
            source=source_region_name_or_point,
            distance=buffer_distance,
            new_name=new_region_name,
            **meta,
        )

    @check_fitted
    def get_indices_for_values_in_region(
        self: "Environment", values: NDArray[np.float64], region_name: str
    ) -> NDArray[np.bool_]:
        """
        Determines which time points (rows in `values` array) correspond
        to locations falling within a specified named region of this environment.
        ...
        """
        if not hasattr(self, "regions") or self.regions is None:  # pragma: no cover
            raise AttributeError(f"Environment '{self.name}' has no 'regions' manager.")
        if values.ndim != 2 or values.shape[1] != self.n_dims:
            raise ValueError(
                f"Values array must be 2D with shape (n_points, {self.n_dims}), "
                f"got {values.shape}."
            )

        active_bin_indices_for_positions = self.get_bin_ind(values)

        # Use the new Environment method
        bins_in_target_region = self.bins_in_region(region_name)

        if bins_in_target_region.size == 0:
            return np.zeros(values.shape[0], dtype=bool)

        is_in_region_mask = np.isin(
            active_bin_indices_for_positions, bins_in_target_region
        )

        return is_in_region_mask
