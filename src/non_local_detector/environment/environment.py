from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
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
from non_local_detector.environment.regions import Region, Regions

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
    factories (e.g., `Environment.from_samples(...)`,
    `Environment.from_graph(...)`). These factories handle the underlying
    `LayoutEngine` setup.

    Attributes
    ----------
    name : str
        A user-defined name for the environment.
    layout : LayoutEngine
        The layout engine instance that defines the geometry and connectivity
        of the discretized space.
    bin_centers : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node in the environment.
        Shape is (n_active_bins, n_dims). Populated by `_setup_from_layout`.
    connectivity : nx.Graph
        A NetworkX graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to the rows of `bin_centers`. Edges represent
        adjacency between bins. Populated by `_setup_from_layout`.
    dimension_ranges : Optional[Sequence[Tuple[float, float]]]
        The effective min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry. Populated by `_setup_from_layout`.
    grid_edges : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts, a tuple where each element is a 1D array of
        bin edge positions for that dimension of the *original, full grid*.
        `None` or `()` for non-grid or point-based layouts. Populated by
        `_setup_from_layout`.
    grid_shape : Optional[Tuple[int, ...]]
        For grid-based layouts, the N-D shape of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept, this
        may be `(n_active_bins,)`. Populated by `_setup_from_layout`.
    active_mask : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid*.
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers`.
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
    bin_centers: NDArray[np.float64] = field(init=False)
    connectivity: nx.Graph = field(init=False)
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = field(init=False)

    # Grid-specific context (populated if layout is grid-based)
    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = field(init=False)
    grid_shape: Optional[Tuple[int, ...]] = field(init=False)
    active_mask: Optional[NDArray[np.bool_]] = field(init=False)

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
        classmethods like `Environment.from_samples(...)`. The provided
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
        self.bin_centers = np.empty((0, 0))  # Placeholder
        self.connectivity = nx.Graph()
        self.dimension_ranges = None
        self.grid_edges = ()
        self.grid_shape = None
        self.active_mask = None
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
        if self.bin_centers is not None:
            active_bins_repr = f"active_bins={self.bin_centers.shape[0]}"

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

        self.bin_centers = self.layout.bin_centers
        self.connectivity = getattr(self.layout, "connectivity", nx.Graph())
        self.dimension_ranges = self.layout.dimension_ranges

        # Grid-specific attributes
        self.grid_edges = getattr(self.layout, "grid_edges", ())
        self.grid_shape = getattr(self.layout, "grid_shape", None)
        self.active_mask = getattr(self.layout, "active_mask", None)

        # If it's not a grid layout, grid_shape might be (n_active_bins,),
        # and active_mask might be 1D all True. This is fine.
        # Ensure they are at least None if not applicable from layout
        if self.grid_shape is None and self.bin_centers is not None:
            # Fallback for point-based
            self.grid_shape = (self.bin_centers.shape[0],)
        if self.active_mask is None and self.bin_centers is not None:
            # Fallback for point-based
            self.active_mask = np.ones(self.bin_centers.shape[0], dtype=bool)

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
            for node_id, data in self.connectivity.nodes(data=True)
            if "source_grid_flat_index" in data
        }

    # --- Factory Methods ---
    @classmethod
    def from_samples(
        cls,
        data_samples: NDArray[np.float64],
        name: str = "",
        layout_kind: str = "RegularGrid",
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
        layout_kind : str, optional
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
            in the `connectivity`. Defaults to True.
        **layout_specific_kwargs : Any
            Additional keyword arguments passed directly to the constructor
            of the chosen `LayoutEngine`.

        Returns
        -------
        Environment
            A new Environment instance.

        Raises
        ------
        NotImplementedError
            If the specified `layout_kind` is not implemented.

        """
        layout_params: Dict[str, Any] = {
            "data_samples": data_samples,
            "infer_active_bins": infer_active_bins,
            "bin_count_threshold": bin_count_threshold,
            **layout_specific_kwargs,
        }
        if layout_kind.lower() == "regulargrid":
            layout_params.update(
                {
                    "bin_size": bin_size,
                    "add_boundary_bins": add_boundary_bins,
                    "dilate": dilate,
                    "fill_holes": fill_holes,
                    "close_gaps": close_gaps,
                    "connect_diagonal_neighbors": connect_diagonal_neighbors,
                }
            )
        elif layout_kind.lower() == "hexagonal":
            # For Hexagonal, bin_size is typically interpreted as hexagon_width
            layout_params.update(
                {
                    "hexagon_width": bin_size,
                }
            )
        else:
            raise NotImplementedError(
                f"Layout kind '{layout_kind}' is not implemented for from_samples."
            )

        return cls.from_layout(kind=layout_kind, layout_params=layout_params, name=name)

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        bin_size: float,
        name: str = "",
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

        Returns
        -------
        Environment
            A new Environment instance with a `GraphLayout`.

        """
        layout_params = {
            "graph_definition": graph,
            "edge_order": edge_order,
            "edge_spacing": edge_spacing,
            "bin_size": bin_size,
        }
        return cls.from_layout(kind="Graph", layout_params=layout_params, name=name)

    @classmethod
    def from_polygon(
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
        return cls.from_layout(
            kind="ShapelyPolygon",
            layout_params=layout_params,
            name=name,
        )

    @classmethod
    def from_mask(
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

        return cls.from_layout(
            kind="MaskedGrid",
            layout_params=layout_params,
            name=name,
        )

    @classmethod
    def from_image(
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

        return cls.from_layout(kind="ImageMask", layout_params=layout_params, name=name)

    @classmethod
    def from_layout(
        cls,
        kind: str,
        layout_params: Dict[str, Any],
        name: str = "",
    ) -> Environment:
        """
        Create an Environment with a specified layout type and its build parameters.

        Parameters
        ----------
        kind : str
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
        layout_instance = create_layout(kind=kind, **layout_params)
        return cls(name, layout_instance, kind, layout_params)

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
            Derived from the shape of `self.bin_centers`.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted or if `bin_centers`
            is not available.
        """
        return self.bin_centers.shape[1]

    @property
    @check_fitted
    def n_bins(self) -> int:
        """
        Return the number of active bins in the environment.

        This is determined by the number of rows in `self.bin_centers`.

        Returns
        -------
        int
            The number of active bins (0 if not fitted).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return self.bin_centers.shape[0]

    @check_fitted
    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
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
    def contains(self, points_nd: NDArray[np.float64]) -> NDArray[np.bool_]:
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
        return self.bin_at(points_nd) != -1

    def bin_center_of(self, bin_indices: NDArray[np.int_]) -> NDArray[np.float64]:
        """
        Return the N-D center point(s) of the specified active bin index/indices.

        This directly indexes the `bin_centers` attribute.

        Parameters
        ----------
        bin_indices : NDArray[np.int_] or int
            Integer index or array of indices for the active bin(s)
            (0 to `n_bins - 1`).

        Returns
        -------
        NDArray[np.float64], shape (n_requested_bins, n_dims) or (n_dims,)
            The N-D center point(s) corresponding to `bin_indices`.
        """
        return self.bin_centers[np.asarray(bin_indices, dtype=int)]

    @check_fitted
    def neighbors(self, bin_index: int) -> List[int]:
        """
        Find indices of neighboring active bins for a given active bin index.

        This method delegates to the `neighbors` method of the
        underlying `LayoutEngine`, which typically uses the `connectivity`.

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
        return list(self.connectivity.neighbors(bin_index))

    @cached_property
    @check_fitted
    def bin_size(self) -> NDArray[np.float64]:
        """
        Calculate the area (for 2D) or volume (for 3D+) of each active bin.

        For 1D environments, this typically returns the length of each bin.
        This method delegates to the `bin_size` method of the
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
        return self.layout.bin_size()

    @cached_property
    def get_all_euclidean_distances(self) -> NDArray[np.float64]:
        """
        Calculate the Euclidean distance between all pairs of active bin centers.

        Returns
        -------
        euclidean_distances : NDArray[np.float64], shape (n_bins, n_bins)
            A square distance matrix where element (i, j) is the Euclidean
            distance between the centers of active bin i and active bin j.
            The diagonal elements (distance from a bin to itself) are 0.
        """
        if self.n_bins == 0:
            return np.empty((0, 0), dtype=float)
        if self.n_bins == 1:
            return np.zeros((self.n_bins, self.n_bins), dtype=float)

        from scipy.spatial.distance import pdist, squareform

        return squareform(pdist(self.bin_centers, metric="euclidean"))

    @cached_property
    def get_all_geodesic_distances(self) -> NDArray[np.float64]:
        """
        Calculate the shortest path (geodesic) distance between all
        pairs of active bins in the environment, using the 'distance'
        attribute of edges in the connectivity graph as weights.

        Returns
        -------
        geodesic_distances : NDArray[np.float64], shape (n_bins, n_bins)
            A square matrix where element (i, j) is the shortest
            path distance between active bin i and active bin j.
            Returns np.inf if no path exists between two bins.
            Diagonal elements are 0.
        """
        if self.connectivity is None or self.n_bins == 0:
            return np.empty((0, 0), dtype=np.float64)

        dist_matrix = np.full((self.n_bins, self.n_bins), np.inf, dtype=np.float64)
        np.fill_diagonal(dist_matrix, 0.0)

        # path_lengths is an iterator of (source_node, {target_node: length})
        path_lengths = nx.shortest_path_length(
            G=self.connectivity, source=None, target=None, weight="distance"
        )
        for source_idx, targets in path_lengths:
            for target_idx, length in targets.items():
                dist_matrix[source_idx, target_idx] = length

        return dist_matrix

    def get_diffusion_kernel(
        self, bandwidth_sigma: float, edge_weight: str = "weight"
    ) -> NDArray[np.float64]:
        """
        Computes the diffusion kernel matrix for all active bins.

        This method utilizes the `connectivity` graph of the environment and
        delegates to an external `compute_diffusion_kernels` function.
        The resulting matrix can represent the influence or probability flow
        between bins after a diffusion process controlled by `bandwidth_sigma`.

        Parameters
        ----------
        bandwidth_sigma : float
            The bandwidth (standard deviation) of the Gaussian kernel used in
            the diffusion process. Controls the spread of the kernel.
        edge_weight : str, optional
            The edge attribute from the connectivity graph to use as weights
            for constructing the Graph Laplacian. Defaults to "weight".

        Returns
        -------
        kernel_matrix : NDArray[np.float64], shape (n_bins, n_bins)
            The diffusion kernel matrix. Element (i, j) can represent the
            influence of bin j on bin i after diffusion. Columns typically
            sum to 1 if normalized by the underlying computation.

        Raises
        ------
        ValueError
            If the connectivity graph is not available or if n_bins is 0.
        ImportError
            If JAX (a dependency of `compute_diffusion_kernels`) is not installed.
        """
        from non_local_detector.diffusion_kernels import compute_diffusion_kernels

        return compute_diffusion_kernels(
            self.connectivity, bandwidth_sigma=bandwidth_sigma, weight=edge_weight
        )

    def get_geodesic_distance(
        self,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float:
        """
        Calculate the geodesic distance between two points in the environment.

        Points are first mapped to their nearest active bins using `self.bin_at()`.
        The geodesic distance is then the shortest path length in the
        `connectivity` graph between these bins, using the specified `edge_weight`.

        Parameters
        ----------
        point1 : PtArr, shape (n_dims,) or (1, n_dims)
            The first N-dimensional point.
        point2 : PtArr, shape (n_dims,) or (1, n_dims)
            The second N-dimensional point.
        edge_weight : str, optional
            The edge attribute to use as weight for path calculation,
            by default "distance". If None, the graph is treated as unweighted.

        Returns
        -------
        float
            The geodesic distance. Returns `np.inf` if points do not map to
            valid active bins, if bins are disconnected, or if the connectivity
            graph is not available.
        """
        source_bin = self.bin_at(np.atleast_2d(point1))[0]
        target_bin = self.bin_at(np.atleast_2d(point2))[0]

        if source_bin == -1 or target_bin == -1:
            # One or both points didn't map to a valid active bin
            return np.inf

        try:
            return nx.shortest_path_length(
                self.connectivity,
                source=source_bin,
                target=target_bin,
                weight=edge_weight,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return np.inf

    @check_fitted
    def get_bin_attributes_dataframe(self) -> pd.DataFrame:  # Renamed
        """
        Create a Pandas DataFrame with attributes of each active bin.

        The DataFrame is constructed from the node data of the
        `connectivity`. Each row corresponds to an active bin.
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
        graph = self.connectivity
        if graph.number_of_nodes() == 0:
            raise ValueError("No active bins in the environment.")

        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "active_bin_id"  # Index is 0..N-1

        if "pos" in df.columns and not df["pos"].dropna().empty:
            pos_df = pd.DataFrame(df["pos"].tolist(), index=df.index)
            pos_df.columns = [f"pos_dim{i}" for i in range(pos_df.shape[1])]
            df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        return df

    @check_fitted
    def get_shortest_path(
        self, source_active_bin_idx: int, target_active_bin_idx: int
    ) -> List[int]:
        """
        Find the shortest path between two active bins.

        The path is a sequence of active bin indices (0 to n_active_bins - 1)
        connecting the source to the target. Path calculation uses the
        'distance' attribute on the edges of the `connectivity`
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
            a node in the `connectivity`.
        """
        graph = self.connectivity

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
            from non_local_detector.environment.regions.plot import plot_regions

            r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
            plot_regions(self.regions, ax=ax, **r_kwargs)

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

    @check_fitted
    def flat_to_grid_bin_index(
        self, flat_indices: Union[int, NDArray[np.int_]]
    ) -> Union[Tuple[int, ...], Tuple[NDArray[np.int_], ...]]:
        """
        Convert active bin flat indices (0..N-1) to N-D grid indices.

        This method translates a flat index (which refers to an active bin,
        e.g., a row in `self.bin_centers`) back to its original N-dimensional
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
            lacks a clear N-D `grid_shape` or `active_mask`).
        RuntimeError
            If called before the environment is fitted or if the connectivity
            graph is unavailable.
        """
        if (
            self.grid_shape is None
            or len(self.grid_shape) <= 1
            or self.active_mask is None
            or self.active_mask.ndim <= 1
        ):
            raise TypeError(
                "N-D index conversion is primarily for N-D grid-based layouts "
                "with a defined N-D active_mask and grid_shape."
            )
        if self.connectivity is None:
            raise RuntimeError(
                "Connectivity graph not available for mapping source indices."
            )

        is_scalar = np.isscalar(flat_indices)
        flat_indices_arr = np.atleast_1d(np.asarray(flat_indices, dtype=int))

        output_nd_indices_list = []

        for active_flat_idx in flat_indices_arr:
            if not (0 <= active_flat_idx < self.connectivity.number_of_nodes()):
                warnings.warn(
                    f"Active flat_index {active_flat_idx} is out of bounds for connectivity nodes. Returning NaNs.",
                    UserWarning,
                )
                output_nd_indices_list.append(tuple([np.nan] * len(self.grid_shape)))
                continue

            node_data = self.connectivity.nodes[active_flat_idx]

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
                        np.unravel_index(original_full_grid_flat_idx, self.grid_shape)
                    )
                )
            else:
                warnings.warn(
                    f"Node {active_flat_idx} in connectivity missing necessary source index information for N-D conversion. Returning NaNs.",
                    UserWarning,
                )
                output_nd_indices_list.append(tuple([np.nan] * len(self.grid_shape)))

        if (
            not output_nd_indices_list
        ):  # Should not happen if flat_indices_arr was not empty
            return tuple(np.array([], dtype=int) for _ in range(len(self.grid_shape)))  # type: ignore

        # Convert list of tuples to tuple of arrays
        final_output_nd_indices = tuple(
            np.array([item[d] for item in output_nd_indices_list])
            for d in range(len(self.grid_shape))
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
            self.grid_shape is None
            or len(self.grid_shape) <= 1
            or self.active_mask is None
            or self.active_mask.ndim <= 1
        ):
            raise TypeError("N-D index conversion is for N-D grid-based layouts.")
        if self.connectivity is None:
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
                and temp_input.shape[0] == len(self.grid_shape)
                and temp_input.shape[1] > 0
            ):  # (n_dims, n_points)
                nd_indices_tuple = tuple(
                    temp_input[d] for d in range(len(self.grid_shape))
                )
            elif temp_input.ndim == 2 and temp_input.shape[1] == len(
                self.grid_shape
            ):  # (n_points, n_dims)
                nd_indices_tuple = tuple(
                    temp_input[:, d] for d in range(len(self.grid_shape))
                )
            elif temp_input.ndim == 1 and len(temp_input) == len(
                self.grid_shape
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

        if len(nd_indices_tuple) != len(self.grid_shape):
            raise ValueError(
                f"Expected {len(self.grid_shape)} N-D indices, got {len(nd_indices_tuple)}"
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
                dim_coords < self.grid_shape[dim_idx]
            )

        if not np.any(in_bounds_mask):
            return (
                output_flat_indices[0]
                if np.isscalar(nd_idx_per_dim[0])
                and len(nd_idx_per_dim) == len(self.grid_shape)
                and common_shape == (1,)
                else output_flat_indices
            )

        # Get the N-D indices that are in bounds
        valid_nd_indices_tuple = tuple(idx[in_bounds_mask] for idx in nd_indices_tuple)

        # Check if these in-bounds N-D grid cells are active
        are_these_bins_active_on_full_grid = self.active_mask[valid_nd_indices_tuple]

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
                and len(nd_idx_per_dim) == len(self.grid_shape)
                and common_shape == (1,)
                else output_flat_indices
            )

        # Convert these N-D indices (of active bins on full grid) to original_full_grid_flat_indices
        original_flat_indices_of_targets = np.ravel_multi_index(
            truly_active_nd_indices_tuple, self.grid_shape
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
            and len(nd_idx_per_dim) == len(self.grid_shape)
            and common_shape == (1,)
        ):
            return int(output_flat_indices[0])
        return output_flat_indices

    @check_fitted
    def get_boundary_bin_indices(
        self: "Environment", connectivity_threshold_factor: Optional[float] = None
    ) -> NDArray[np.int_]:
        graph = self.connectivity
        if graph.number_of_nodes() == 0:
            return np.array([], dtype=int)

        boundary_bin_indices = []

        # Grid-specific logic for N-D grids (N > 1)
        is_nd_grid_layout_with_mask = (
            hasattr(self, "active_mask")
            and self.active_mask is not None
            and hasattr(self, "grid_shape")
            and self.grid_shape is not None
            and len(self.grid_shape) > 1  # Explicitly for N-D (N>1) grids
        )

        if is_nd_grid_layout_with_mask and connectivity_threshold_factor is None:
            active_mask_nd = self.active_mask
            grid_shape_nd = self.grid_shape
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
            # - 1D grid layouts (where len(grid_shape) == 1)
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
                    hasattr(self, "grid_shape")
                    and self.grid_shape is not None
                    and len(self.grid_shape) == 1
                ) or (self.is_1d and isinstance(self.layout, GraphLayout)):
                    # For linear structures, ends typically have degree 1, internal degree 2
                    # Threshold just below typical internal degree.
                    threshold_degree = (
                        1.5 if max_degree_val >= 2 else max_degree_val
                    )  # Catches degree 1
                elif is_nd_grid_layout_with_mask and median_degree > (
                    2 * len(self.grid_shape) - 1
                ):  # For N>1 grids when falling back
                    threshold_degree = 2 * len(self.grid_shape) - 0.5
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
        region_info = self.regions[region_name]

        if region_info.kind == "point":
            point_nd = np.asarray(region_info.data).reshape(1, -1)
            if point_nd.shape[1] != self.n_dims:
                raise ValueError(
                    f"Region point dimension {point_nd.shape[1]} "
                    f"does not match environment dimension {self.n_dims}."
                )
            bin_idx = self.bin_at(point_nd)
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
                [polygon.contains(_shp.Point(center)) for center in self.bin_centers],
                dtype=bool,
            )
            return np.flatnonzero(contained_mask)

        else:  # pragma: no cover
            raise ValueError(f"Unsupported region kind: {region_info.kind}")

    @check_fitted
    def mask_for_region(self, region_name: str) -> NDArray[np.bool_]:
        """
        Get a boolean mask over active bins indicating membership in a region.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array of shape (n_active_bins,). True if an active bin
            is part of the region.
        """
        active_bins_for_mask = self.bins_in_region(region_name)
        mask = np.zeros(self.bin_centers.shape[0], dtype=bool)
        if active_bins_for_mask.size > 0:
            mask[active_bins_for_mask] = True
        return mask

    @check_fitted
    def region_center(self, region_name: str) -> Optional[NDArray[np.float64]]:
        """
        Calculate the center of a specified named region.

        - For 'point' regions, returns the point itself.
        - For 'polygon' regions, returns the centroid of the polygon.
        - For 'mask' regions, returns the mean of the bin_centers of the
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
        return 0.0  # pragma: no cover

    @check_fitted
    def create_buffered_region(
        self,
        source_region_name_or_point: Union[str, NDArray[np.float64]],
        buffer_distance: float,
        new_region_name: str,
        **meta: Any,
    ) -> Region:
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
        Region
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

        return self.regions.buffer(
            source=source_region_name_or_point,
            distance=buffer_distance,
            new_name=new_region_name,
            **meta,
        )
