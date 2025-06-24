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

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.engines.graph import GraphLayout
from non_local_detector.environment.layout.engines.regular_grid import RegularGridLayout
from non_local_detector.environment.layout.factories import create_layout
from non_local_detector.environment.layout.helpers.utils import find_boundary_nodes
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
        regions: Optional[Regions] = None,
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
        if layout_type_used is not None:
            self._setup_from_layout()  # Populate attributes from the built layout
        if regions is not None:
            if not isinstance(regions, Regions):
                raise TypeError(
                    f"Expected 'regions' to be a Regions instance, got {type(regions)}."
                )
            self.regions = regions
        else:
            # Initialize with an empty Regions instance if not provided
            self.regions = Regions()

    def __eq__(self, other: str) -> bool:
        return self.name == other

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
        Create an Environment by binning (discretizing) `data_samples` into a layout grid.

        Parameters
        ----------
        data_samples : array, shape (n_samples, n_dims)
            Coordinates of sample points used to infer which bins are “active.”
        name : str, default ""
            Optional name for the resulting Environment.
        layout_kind : str, default "RegularGrid"
            Either "RegularGrid" or "Hexagonal" (case-insensitive). Determines
            bin shape. For "Hexagonal", `bin_size` is interpreted as `hexagon_width`.
        bin_size : float or sequence of floats, default 2.0
            For RegularGrid: length of each square bin side. For Hexagonal: hexagon width.
        infer_active_bins : bool, default True
            If True, only bins containing ≥ `bin_count_threshold` samples are “active.”
        bin_count_threshold : int, default 0
            Minimum number of data points required for a bin to be considered “active.”
        dilate : bool, default False
            If True, apply morphological dilation to the active-bin mask.
        fill_holes : bool, default False
            If True, fill holes in the active-bin mask.
        close_gaps : bool, default False
            If True, close small gaps between active bins.
        add_boundary_bins : bool, default False
            If True, add peripheral bins around the bounding region of samples.
        connect_diagonal_neighbors : bool, default True
            If True, connect grid bins diagonally when building connectivity.

        Returns
        -------
        env : Environment
            A newly created Environment, fitted to the discretized samples.

        Raises
        ------
        ValueError
            If `data_samples` is not 2D or contains invalid coordinates.
        NotImplementedError
            If `layout_kind` is neither "RegularGrid" nor "Hexagonal".
        """
        # Convert and validate data_samples array
        data_samples = np.asarray(data_samples, dtype=float)
        if data_samples.ndim != 2:
            raise ValueError(
                f"data_samples must be a 2D array of shape (n_points, n_dims), "
                f"got shape {data_samples.shape}."
            )

        # Standardize layout_kind to lowercase for comparison
        kind_lower = layout_kind.lower()
        if kind_lower not in ("regulargrid", "hexagonal"):
            raise NotImplementedError(
                f"Layout kind '{layout_kind}' is not supported. "
                "Use 'RegularGrid' or 'Hexagonal'."
            )

        # Build the dict of layout parameters
        layout_params: Dict[str, Any] = {
            "data_samples": data_samples,
            "infer_active_bins": infer_active_bins,
            "bin_count_threshold": bin_count_threshold,
            **layout_specific_kwargs,
        }

        if kind_lower == "regulargrid":
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
        elif kind_lower == "hexagonal":
            layout_params.update(
                {
                    "hexagon_width": bin_size,
                }
            )
        else:
            raise NotImplementedError(
                f"Layout kind '{layout_kind}' is not supported. "
                "Use 'RegularGrid' or 'Hexagonal'."
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
        regions: Optional[Regions] = None,
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
        regions : Optional[Regions], optional
            A Regions instance to manage symbolic spatial regions within the environment.

        Returns
        -------
        Environment
            A new Environment instance.

        """
        layout_instance = create_layout(kind=kind, **layout_params)
        return cls(name, layout_instance, kind, layout_params, regions=regions)

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
    def layout_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters used to build the layout engine.

        This includes all parameters that were passed to the `build` method
        of the underlying `LayoutEngine`.

        Returns
        -------
        Dict[str, Any]
            A dictionary of parameters used to create the layout.
            Useful for introspection and serialization.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        """
        return self._layout_params_used

    @property
    @check_fitted
    def layout_type(self) -> str:
        """
        Return the type of layout used in the environment.

        Returns
        -------
        str
            The layout type (e.g., "RegularGrid", "Hexagonal").
        """
        return self._layout_type_used

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

    @check_fitted
    def bin_center_of(
        self, bin_indices: Union[int, Sequence[int], NDArray[np.int_]]
    ) -> NDArray[np.float64]:
        """
        Given one or more active-bin indices, return their N-D center coordinates.

        Parameters
        ----------
        bin_indices : int or sequence of int
            Index (or list/array of indices) of active bins (0 <= idx < self.n_bins).

        Returns
        -------
        centers : array, shape (len(bin_indices), n_dims) if multiple indices,
                        (n_dims,) if single index
            The center coordinate(s) of the requested bin(s).

        Raises
        ------
        RuntimeError
            If the environment is not fitted.
        IndexError
            If any bin index is out of range.
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
    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Calculate the area (for 2D) or volume (for 3D+) of each active bin.

        This represent the actual size of each bin in the environment, as
        opposed to the requested `bin_size` which is the nominal size used
        during layout creation.

        For 1D environments, this typically returns the length of each bin.
        This method delegates to the `bin_sizes` method of the
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
        return self.layout.bin_sizes()

    @cached_property
    @check_fitted
    def boundary_bins(self) -> NDArray[np.int_]:
        """Get the boundary bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_boundary_bins,)
            An array of indices of the boundary bins in the environment.
            These are the bins that are at the edges of the active area.
        """
        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
            layout_kind=self._layout_type_used,
        )

    @cached_property
    @check_fitted
    def linearization_properties(
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
            Returns `None` otherwise.
        """
        if isinstance(self.layout, GraphLayout):
            return {
                "track_graph": self._layout_params_used.get("graph_definition"),
                "edge_order": self._layout_params_used.get("edge_order"),
                "edge_spacing": self._layout_params_used.get("edge_spacing"),
            }

    @cached_property
    @check_fitted
    def bin_attributes(self) -> pd.DataFrame:
        """
        Build a DataFrame of attributes for each active bin (node) in the environment's graph.

        Returns
        -------
        df : pandas.DataFrame
            Rows are indexed by `active_bin_id` (int), matching 0..(n_bins-1).
            Columns correspond to node attributes. If a 'pos' attribute exists
            for any node and is non-null, it will be expanded into columns
            'pos_dim0', 'pos_dim1', ..., with numeric coordinates.

        Raises
        ------
        ValueError
            If there are no active bins (graph has zero nodes).
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

    @cached_property
    @check_fitted
    def edge_attributes(self) -> pd.DataFrame:
        """
        Return a Pandas DataFrame where each row corresponds to one directed edge
        (u → v) in the connectivity graph, and columns include all stored edge
        attributes (e.g. 'distance', 'vector', 'weight', 'angle_2d', etc.).

        The DataFrame will have a MultiIndex of (source_bin, target_bin). If you
        prefer flat columns, you can reset the index.

        Returns
        -------
        pd.DataFrame
            A DataFrame whose index is a MultiIndex (source_bin, target_bin),
            and whose columns are the union of all attribute-keys stored on each edge.

        Raises
        ------
        ValueError
            If there are no edges in the connectivity graph.
        RuntimeError
            If called before the environment is fitted.
        """
        G = self.connectivity
        if G.number_of_edges() == 0:
            raise ValueError("No edges in the connectivity graph.")

        # Build a dict of edge_attr_dicts keyed by (u, v)
        # networkx's G.edges(data=True) yields (u, v, attr_dict)
        edge_dict: dict[tuple[int, int], dict] = {
            (u, v): data.copy() for u, v, data in G.edges(data=True)
        }

        # Convert that to a DataFrame, using the (u, v) tuples as a MultiIndex
        df = pd.DataFrame.from_dict(edge_dict, orient="index")
        # The index is now a MultiIndex of (u, v)
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["source_bin", "target_bin"]
        )

        return df

    def distance_between(
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
    def shortest_path(
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
    def to_linear(self, points_nd: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert N-dimensional points to 1D linearized coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `to_linear` method.

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
        return self.layout.to_linear(points_nd)

    @check_fitted
    def linear_to_nd(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convert 1D linearized coordinates back to N-dimensional coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `linear_to_nd` method.

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
        return self.layout.linear_to_nd(linear_coordinates)

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

    @check_fitted
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
        region = self.regions[region_name]

        if region.kind == "point":
            point_nd = np.asarray(region.data).reshape(1, -1)
            if point_nd.shape[1] != self.n_dims:
                raise ValueError(
                    f"Region point dimension {point_nd.shape[1]} "
                    f"does not match environment dimension {self.n_dims}."
                )
            bin_idx = self.bin_at(point_nd)
            return bin_idx[bin_idx != -1]

        elif region.kind == "polygon":
            if not _HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon region queries require 'shapely'.")
            if self.n_dims != 2:  # pragma: no cover
                raise ValueError(
                    "Polygon regions are only supported for 2D environments."
                )

            from shapely import vectorized

            polygon = region.data
            contained_mask = vectorized.contains(
                polygon, self.bin_centers[:, 0], self.bin_centers[:, 1]
            )

            return np.flatnonzero(contained_mask)

        else:  # pragma: no cover
            raise ValueError(f"Unsupported region kind: {region.kind}")

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
