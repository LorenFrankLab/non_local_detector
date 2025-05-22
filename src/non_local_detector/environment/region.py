"""
Manages symbolic spatial regions of interest (ROIs) within an Environment.

This module provides the `RegionInfo` dataclass to store details about a
specific region and the `RegionManager` class to handle collections of these
regions, their geometric definitions, and interactions with the associated
`Environment` instance. Regions can be defined by points, boolean masks, or
Shapely polygons (if Shapely is installed).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import matplotlib.axes
import numpy as np
from numpy.typing import NDArray

from non_local_detector.environment.layout_engine import _GridMixin

# For type hinting Environment without circular import
if TYPE_CHECKING:
    from non_local_detector.environment.environment import Environment

# Shapely for polygon operations (optional dependency)
try:
    import shapely.geometry as _shp
    from shapely.ops import unary_union

    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

    # Define dummy classes for type hinting if Shapely is not installed
    class _shp:  # type: ignore[no-redef]
        class Polygon:
            pass

        class Point:
            pass

        class MultiPolygon:
            pass

        class LineString:
            pass

        class MultiLineString:
            pass

        class GeometryCollection:
            pass

    unary_union = None  # type: ignore


def _point_in_polygon(
    points: NDArray[np.float_], polygon_geom: _shp.Polygon
) -> NDArray[np.bool_]:
    """
    Check if 2D points are inside a Shapely Polygon.

    Parameters
    ----------
    points : NDArray[np.float_], shape (n_points, 2)
        Array of (x, y) coordinates for the points to check.
    polygon_geom : shapely.geometry.Polygon
        The Shapely Polygon object against which to check the points.

    Returns
    -------
    NDArray[np.bool_], shape (n_points,)
        A boolean array where `True` indicates the corresponding point is
        inside or on the boundary of the polygon.

    Raises
    ------
    RuntimeError
        If the 'shapely' package is not installed.
    """
    if not _HAS_SHAPELY:  # pragma: no cover
        raise RuntimeError("Shapely is required for polygon operations.")
    if points.ndim == 1:  # Single point
        points = points.reshape(1, -1)
    if points.shape[0] == 0:
        return np.array([], dtype=bool)
    if points.shape[1] != 2:
        raise ValueError("Points must be 2-dimensional for polygon checking.")

    try:
        from shapely import points as shapely_points_vec

        return polygon_geom.contains(shapely_points_vec(points))
    except AttributeError:
        # Fallback for older Shapely versions
        # Check if points are inside the polygon using a loop
        # This is less efficient but works with older Shapely versions
        # and without the vectorized points function.
        return np.array([polygon_geom.contains(_shp.Point(p)) for p in points])


@dataclass
class RegionInfo:
    """
    Container for information about a defined spatial region.

    Attributes
    ----------
    name : str
        User-supplied unique identifier for the region.
    kind : str
        The type of geometric definition for the region.
        Must be one of ``{"point", "mask", "polygon"}``.
    data : Any
        The geometric data defining the region:
        - If `kind` is "point": `NDArray[np.float_]` of shape `(n_dims,)`
          representing the coordinates of the point.
        - If `kind` is "mask": `NDArray[np.bool_]` with N-D shape matching the
          environment's full `grid_shape_` (for grid-based layouts). `True`
          values indicate that a grid cell is part of the region.
        - If `kind` is "polygon": A `shapely.geometry.Polygon` object (for 2D
          layouts).
    metadata : dict
        Arbitrary key-value store for additional information about the region
        (e.g., plotting arguments, semantic labels).
    """

    name: str
    kind: str
    data: Any
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate region data after initialization."""
        if self.kind not in {"point", "mask", "polygon"}:
            raise ValueError(
                f"Unknown region kind: '{self.kind}'. Must be 'point', 'mask', or 'polygon'."
            )
        if self.kind == "polygon" and not _HAS_SHAPELY:
            raise RuntimeError(
                "Cannot create region of kind 'polygon': "
                "The 'shapely' package is not installed."
            )
        if self.kind == "point":
            if not isinstance(self.data, np.ndarray) or self.data.ndim != 1:
                raise TypeError(
                    "Region data for kind 'point' must be a 1D NumPy array."
                )
        elif self.kind == "mask":
            if not isinstance(self.data, np.ndarray) or self.data.dtype != bool:
                raise TypeError(
                    "Region data for kind 'mask' must be a boolean NumPy array."
                )
        elif self.kind == "polygon":
            if not isinstance(self.data, _shp.Polygon):  # type: ignore
                raise TypeError(
                    "Region data for kind 'polygon' must be a Shapely Polygon."
                )


class RegionManager:
    """
    Manages symbolic spatial regions within an associated Environment.

    This class provides methods to define regions using points, N-D boolean masks,
    or Shapely polygons. It allows querying these regions in relation to the
    discretized bins of the parent Environment.
    """

    def __init__(self, environment_facade: "Environment"):
        """
        Initialize the RegionManager.

        Parameters
        ----------
        environment_facade : Environment
            A reference to the `Environment` instance this manager serves.
            This facade provides access to the environment's geometric
            properties (e.g., `bin_centers_`, `grid_shape_`, `active_mask_`,
            `get_bin_ind`, `n_dims`).
        """
        self._env = environment_facade  # Facade to access Environment's state
        self._regions: Dict[str, RegionInfo] = {}

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the RegionManager.

        Returns
        -------
        str
            A string representation showing the class name, the name of the
            associated environment, and the number of defined regions.
        """
        env_name = "N/A"
        if self._env is not None and hasattr(self._env, "environment_name"):
            env_name = self._env.environment_name

        return (
            f"{self.__class__.__name__}("
            f"environment_name={env_name!r}, "
            f"n_regions={len(self._regions)}"
            f")"
        )

    def _check_env_fitted(self, method_name: str):
        """
        Ensure the associated Environment is fitted before proceeding.

        Parameters
        ----------
        method_name : str
            The name of the method calling this check (for error messages).

        Raises
        ------
        RuntimeError
            If the parent Environment (`self._env`) is not fitted.
        """
        if not self._env._is_fitted:  # Accessing internal _is_fitted via facade
            raise RuntimeError(
                f"RegionManager.{method_name}() requires the parent Environment to be fitted."
            )

    def add_region(
        self,
        name: str,
        *,
        point: Optional[Union[Tuple[float, ...], NDArray[np.float_]]] = None,
        mask: Optional[NDArray[np.bool_]] = None,
        polygon: Optional[Union["_shp.Polygon", List[Tuple[float, float]]]] = None,
        **metadata,
    ) -> None:
        """
        Register a new spatial region with a unique name.

        Exactly one of `point`, `mask`, or `polygon` must be provided to
        define the region's geometry.

        Parameters
        ----------
        name : str
            Unique identifier for the region.
        point : Optional[Union[Tuple[float, ...], NDArray[np.float_]]], optional
            Coordinates (same dimensionality as environment) defining a point
            region. The region will effectively correspond to the active bin
            closest to this point.
        mask : Optional[NDArray[np.bool_]], optional
            Boolean N-D array defining the region. For grid-based layouts,
            this mask must match the environment's full `grid_shape_` and
            applies to the conceptual full grid.
        polygon : Optional[Union[shapely.geometry.Polygon, List[Tuple[float,float]]]], optional
            A Shapely Polygon object or a list of (x, y) vertex tuples
            defining a polygonal region. Applicable primarily for 2D
            environments. Bins whose centers fall within this polygon are
            considered part of the region.
        **metadata : Any
            Additional key-value pairs to store with the region (e.g.,
            `plot_kwargs={'color': 'red'}`).

        Raises
        ------
        ValueError
            If `name` already exists, if not exactly one geometric specifier
            is provided, or if inputs are inconsistent with the environment
            (e.g., dimension mismatch for point, shape mismatch for mask).
        RuntimeError
            If `polygon` is specified but Shapely is not installed.
        TypeError
            If the data provided for `point`, `mask`, or `polygon` is of
            an incorrect type.
        """
        self._check_env_fitted("add_region")
        if sum(v is not None for v in (point, mask, polygon)) != 1:
            raise ValueError(
                "Must provide exactly one of 'point', 'mask', or 'polygon' to define a region."
            )
        if name in self._regions:
            raise ValueError(f"Region '{name}' already exists. Choose a unique name.")

        kind: str
        data_processed: Any

        if point is not None:
            kind = "point"
            data_processed = np.asarray(point, dtype=float)
            if data_processed.ndim == 0:  # Scalar provided
                if self._env.n_dims == 1:
                    data_processed = np.array([data_processed])  # Make it (1,)
                else:
                    raise ValueError(
                        f"Scalar point provided for {self._env.n_dims}D environment."
                    )
            if data_processed.shape != (self._env.n_dims,):
                raise ValueError(
                    f"Point dimensions ({data_processed.shape}) mismatch environment "
                    f"dimensions ({self._env.n_dims,})."
                )
        elif mask is not None:
            kind = "mask"
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                raise TypeError(
                    "Region data for kind 'mask' must be a boolean NumPy array."
                )
            if self._env.grid_shape_ is not None:
                if mask.shape != self._env.grid_shape_:
                    raise ValueError(
                        f"Mask shape {mask.shape} mismatches environment's "
                        f"full grid_shape_ {self._env.grid_shape_}."
                    )
            data_processed = mask.copy()

        elif polygon is not None:
            kind = "polygon"
            if not _HAS_SHAPELY:
                raise RuntimeError(
                    "Cannot add polygon region: 'shapely' package is not installed."
                )
            if self._env.n_dims != 2:
                raise ValueError(
                    "Polygon regions are only supported for 2D environments."
                )
            if isinstance(polygon, list):  # User provided list of coords
                try:
                    data_processed = _shp.Polygon(polygon)  # type: ignore
                except Exception as e:
                    raise ValueError(
                        f"Could not create Shapely Polygon from provided coordinates: {e}"
                    )
            elif isinstance(polygon, _shp.Polygon):  # type: ignore
                data_processed = polygon
            else:
                raise TypeError(
                    "polygon must be a Shapely Polygon instance or a list of vertex coordinates."
                )
            if not data_processed.is_valid:
                warnings.warn(
                    f"Polygon for region '{name}' is not valid (e.g., self-intersecting). "
                    "Behavior of geometric operations may be undefined.",
                    UserWarning,
                )
        else:  # Should be caught by the sum check
            raise ValueError("Internal error: No region specifier found.")

        self._regions[name] = RegionInfo(
            name=name, kind=kind, data=data_processed, metadata=metadata
        )

    def remove_region(self, name: str) -> None:
        """
        Remove a region from the manager.

        If the specified region name does not exist, this method does nothing
        (silently ignores the request).

        Parameters
        ----------
        name : str
            The unique identifier of the region to remove.
        """
        self._regions.pop(name, None)

    def list_regions(self) -> List[str]:
        """
        Return a list of names of all registered regions.

        The order of names corresponds to the order of insertion.

        Returns
        -------
        List[str]
            A list of all defined region names.
        """
        return list(self._regions.keys())

    def get_region_info(self, name: str) -> RegionInfo:
        """
        Retrieve the `RegionInfo` object for a named region.

        Parameters
        ----------
        name : str
            The unique identifier of the region.

        Returns
        -------
        RegionInfo
            The `RegionInfo` object containing details about the specified region.

        Raises
        ------
        KeyError
            If a region with the given `name` is not found.
        """
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found.")
        return self._regions[name]

    @property
    def environment(self) -> "Environment":
        """
        Access the parent `Environment` instance.

        Returns
        -------
        Environment
            The `Environment` instance this `RegionManager` is associated with.
        """
        return self._env

    @environment.setter
    def environment(self, value: "Environment"):
        """
        Set the parent `Environment` instance.

        This is typically done only once during initialization.
        A warning is issued if attempting to overwrite an existing association.

        Parameters
        ----------
        value : Environment
            The `Environment` instance to associate with this manager.
        """
        if hasattr(self, "_env") and self._env is not None:
            warnings.warn(
                "RegionManager is already associated with an Environment. Overwriting.",
                UserWarning,
            )
        self._env = value

    def region_mask(self, name: str) -> NDArray[np.bool_]:
        """
        Return a 1D boolean mask indicating active environment bins in the region.

        The returned mask has a length equal to the number of *active bins*
        in the associated `Environment` (`env.bin_centers_.shape[0]`).
        `True` at an index `i` means that active bin `i` is part of the
        specified region.

        Parameters
        ----------
        name : str
            The name of the region for which to generate the mask.

        Returns
        -------
        NDArray[np.bool_], shape (n_active_bins,)
            A 1D boolean mask over the active bins of the environment.

        Raises
        ------
        KeyError
            If the region `name` is not found.
        RuntimeError
            If the environment is not fitted, or if required components
            (like Shapely for polygon regions) are missing.
        ValueError
            If region data or environment properties are inconsistent
            (e.g., mask shape mismatch for 'mask' kind regions).
        """
        self._check_env_fitted("region_mask")
        info = self.get_region_info(name)

        n_active_bins = self._env.bin_centers_.shape[0]
        active_bin_is_in_region_1d = np.zeros(n_active_bins, dtype=bool)

        if n_active_bins == 0:  # No active bins in the environment
            return active_bin_is_in_region_1d

        if info.kind == "point":
            # Find which active bin the point maps to
            mapped_bin_idx_arr = self._env.get_bin_ind(
                np.atleast_2d(info.data)
            )  # Returns array
            if mapped_bin_idx_arr.size > 0 and mapped_bin_idx_arr[0] != -1:
                active_bin_is_in_region_1d[mapped_bin_idx_arr[0]] = True

        elif info.kind == "mask":  # N-D mask defined on the full conceptual grid
            if (
                self._env.grid_shape_ is None
                or self._env.active_mask_ is None
                or not (
                    isinstance(self._env.active_mask_, np.ndarray)
                    and self._env.active_mask_.ndim == len(self._env.grid_shape_)
                )
            ):
                raise ValueError(
                    "Region 'mask' kind requires a grid-based environment with a defined N-D active_mask_."
                )
            if info.data.shape != self._env.grid_shape_:
                raise ValueError(
                    f"Region mask shape {info.data.shape} mismatches environment's full grid_shape_ {self._env.grid_shape_}."
                )

            # Combine region's N-D mask with environment's N-D active_mask_
            # This gives an N-D mask of bins that are BOTH in the region AND active in the environment
            effective_nd_mask_for_region = info.data & self._env.active_mask_

            # Get original flat indices (in the full grid) of these doubly-selected bins
            original_flat_indices_in_region_and_active = np.flatnonzero(
                effective_nd_mask_for_region
            )

            if original_flat_indices_in_region_and_active.size > 0:
                # Map these original full-grid flat indices to 0..N-1 active bin indices

                for orig_flat_idx in original_flat_indices_in_region_and_active:
                    active_node_id = self._env._source_flat_to_active_node_id_map.get(
                        orig_flat_idx
                    )
                    if (
                        active_node_id is not None
                    ):  # Should always be found if logic is correct
                        active_bin_is_in_region_1d[active_node_id] = True

        elif info.kind == "polygon":
            if not _HAS_SHAPELY or self._env.n_dims != 2:
                raise RuntimeError(
                    "Polygon regions require Shapely and a 2D environment."
                )
            # self._env.bin_centers_ are the N-D (here 2D) coordinates of *active* bins
            if self._env.bin_centers_.shape[0] > 0:
                points_inside = _point_in_polygon(
                    self._env.bin_centers_[:, :2], info.data
                )
                active_bin_is_in_region_1d = points_inside

        return active_bin_is_in_region_1d

    def bins_in_region(self, name: str) -> NDArray[np.int_]:
        """
        Return active bin indices (0..N-1) that are part of a named region.

        Parameters
        ----------
        name : str
            The name of the region.

        Returns
        -------
        NDArray[np.int_], shape (n_bins_in_region,)
            An array of integer indices for active bins that fall within
            the specified region.

        Raises
        ------
        KeyError
            If the region `name` is not found.
        RuntimeError
            If the environment is not fitted.
        """
        self._check_env_fitted("bins_in_region")
        active_bin_mask_1d = self.region_mask(name)
        return np.flatnonzero(active_bin_mask_1d).astype(np.int_)

    def region_center(self, name: str) -> Optional[NDArray[np.float64]]:
        """
        Calculate the geometric center of active bins within a specified region.

        - For "point" regions, returns the defining point itself.
        - For "mask" or "polygon" regions, computes the mean N-D coordinates
          of the centers of all active environment bins that fall within the region.

        Parameters
        ----------
        name : str
            The name of the region.

        Returns
        -------
        Optional[NDArray[np.float64]], shape (n_dims,)
            The N-D coordinates of the region's center. Returns `None` if the
            region is empty (contains no active bins) or if the region itself
            is not defined.

        Raises
        ------
        KeyError
            If the region `name` is not found.
        RuntimeError
            If the environment is not fitted.
        """
        self._check_env_fitted("region_center")
        info = self.get_region_info(name)
        if info.kind == "point":
            return np.asarray(info.data)

        active_bins_indices_in_region = self.bins_in_region(name)
        if active_bins_indices_in_region.size == 0:
            warnings.warn(
                f"Region '{name}' contains no active bins. Cannot calculate center.",
                UserWarning,
            )
            return None
        return np.mean(self._env.bin_centers_[active_bins_indices_in_region], axis=0)

    def nearest_region(
        self,
        position: NDArray[np.float64],
        candidate_region_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Find the named region whose center is closest to a query position.

        Distance is calculated as Euclidean distance between the query position
        and the `region_center()` of each candidate region.

        Parameters
        ----------
        position : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The N-D query position.
        candidate_region_names : Optional[List[str]], optional
            A list of region names to consider. If None (default), all
            defined regions are considered.

        Returns
        -------
        Optional[str]
            The name of the nearest region. Returns `None` if no regions are
            defined, no suitable candidate regions have a calculable center,
            or if `candidate_region_names` is empty.

        Raises
        ------
        ValueError
            If the dimensionality of `position` does not match the
            environment's `n_dims`.
        RuntimeError
            If the environment is not fitted.
        """
        self._check_env_fitted("nearest_region")
        query_pos = np.atleast_2d(position)
        if query_pos.shape[1] != self._env.n_dims:
            raise ValueError(
                f"Position dimensions ({query_pos.shape[1]}) mismatch environment ({self._env.n_dims})."
            )

        best_name: Optional[str] = None
        min_dist_sq = np.inf  # Use squared distance to avoid sqrt

        regions_to_check = (
            candidate_region_names
            if candidate_region_names is not None
            else self.list_regions()
        )

        for region_name in regions_to_check:
            center = self.region_center(
                region_name
            )  # This already checks if region exists
            if center is not None:
                dist_sq = np.sum(
                    (query_pos.squeeze() - center) ** 2
                )  # Squared Euclidean
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_name = region_name
        return best_name

    def get_region_area(self, region_name: str) -> Optional[float]:
        """
        Calculate the area/volume of a defined region.

        - For "polygon" regions (2D only): Returns the polygon's area using Shapely.
        - For "mask" regions on a grid: Sums the area/volume of active environment
          bins that are part of the region. Assumes uniform bin sizes if calculated
          from grid edges.
        - For "point" regions: Returns 0.0.

        Parameters
        ----------
        region_name : str
            The name of the region.

        Returns
        -------
        Optional[float]
            The calculated area/volume of the region. Returns `None` if the area
            cannot be determined (e.g., Shapely not available for a polygon, or
            layout type does not support area calculation for a mask).

        Raises
        ------
        KeyError
            If the region `region_name` is not found.
        RuntimeError
            If the environment is not fitted.
        """
        self._check_env_fitted("get_region_area")
        info = self.get_region_info(region_name)  # Raises KeyError if name not found

        if info.kind == "polygon":
            if _HAS_SHAPELY and self._env.n_dims == 2:
                return info.data.area  # type: ignore
            else:  # pragma: no cover
                warnings.warn(
                    "Cannot calculate area for polygon region: Shapely not "
                    "available or environment is not 2D.",
                    UserWarning,
                )
                return None
        elif info.kind == "mask":
            if (
                self._env.grid_edges_
                and len(self._env.grid_edges_) == self._env.n_dims
                and all(
                    isinstance(e, np.ndarray) and e.size > 1
                    for e in self._env.grid_edges_
                )
            ):
                # Assumes uniform bins if calculating from grid_edges
                bin_dimension_sizes = [
                    np.diff(edge_dim)[0] for edge_dim in self._env.grid_edges_
                ]
                single_bin_measure = np.prod(bin_dimension_sizes)
                num_active_bins_in_mask_region = np.sum(self.region_mask(region_name))
                return num_active_bins_in_mask_region * single_bin_measure
            elif hasattr(
                self._env.layout, "get_bin_area_volume"
            ):  # Fallback if layout provides per-bin measures
                active_bins_in_reg_indices = self.bins_in_region(region_name)
                if active_bins_in_reg_indices.size == 0:
                    return 0.0
                all_bin_areas = self._env.layout.get_bin_area_volume()
                if (
                    len(all_bin_areas) == self._env.bin_centers_.shape[0]
                ):  # Matches active bins
                    return np.sum(all_bin_areas[active_bins_in_reg_indices])
                else:  # pragma: no cover
                    warnings.warn(
                        f"Area calculation for mask region '{region_name}' on layout type "
                        f"'{self._env._layout_type_used}' with non-standard bin area reporting "
                        "not fully supported yet.",
                        UserWarning,
                    )
                    return None
            else:  # pragma: no cover
                warnings.warn(
                    f"Layout type for mask region '{region_name}' does not support "
                    "area calculation based on grid edges or get_bin_area_volume.",
                    UserWarning,
                )
                return None
        elif info.kind == "point":
            return 0.0
        return None  # Should be unreachable if kind is validated

    def create_buffered_region(
        self,
        source_region_name_or_point: Union[str, NDArray[np.float_]],
        buffer_distance: float,
        new_region_name: str,
        **metadata,
    ) -> None:
        """
        Create a new polygonal region by buffering an existing region or a point.

        This operation is currently supported only for 2D environments and
        requires the 'shapely' package.
        - If `source_region_name_or_point` is a region name:
            - "polygon" kind: Buffers the existing polygon.
            - "point" kind: Buffers the point.
            - "mask" kind: Computes the convex hull of active bins within the
              mask region and then buffers this hull.
        - If `source_region_name_or_point` is an N-D array: Treats it as a point to buffer.

        Parameters
        ----------
        source_region_name_or_point : Union[str, NDArray[np.float_]]
            The name of an existing region, or a 2D NumPy array `(x, y)`
            representing a point.
        buffer_distance : float
            The distance by which to buffer the source geometry. Positive values
            expand, negative values shrink (if applicable).
        new_region_name : str
            The name for the newly created buffered region.
        **metadata : Any
            Additional metadata for the new region.

        Raises
        ------
        RuntimeError
            If Shapely is not installed.
        ValueError
            If the environment is not 2D, or if `source_region_name_or_point`
            is an array with incorrect dimensions.
        TypeError
            If `source_region_name_or_point` is of an unsupported type.
        KeyError
            If `source_region_name_or_point` is a string but no such region exists.
        """
        self._check_env_fitted("create_buffered_region")
        if not _HAS_SHAPELY:
            raise RuntimeError("Buffering requires Shapely.")
        if self._env.n_dims != 2:
            raise ValueError(
                "Buffering regions is currently only supported for 2D environments."
            )
        if buffer_distance < 0:
            warnings.warn(
                "Negative buffer distance will shrink the region.", UserWarning
            )

        source_geom: Optional["_shp.BaseGeometry"] = None  # type: ignore
        if isinstance(source_region_name_or_point, str):
            source_info = self.get_region_info(source_region_name_or_point)
            if source_info.kind == "polygon":
                source_geom = source_info.data
            elif source_info.kind == "point":
                source_geom = _shp.Point(source_info.data)  # type: ignore
            elif source_info.kind == "mask":
                # For mask, could take convex hull of active bins in region, then buffer
                active_bins_idx = self.bins_in_region(source_info.name)
                if active_bins_idx.size > 0:
                    points_for_hull = self._env.bin_centers_[active_bins_idx][:, :2]
                    if points_for_hull.shape[0] >= 3:
                        source_geom = _shp.MultiPoint(points_for_hull).convex_hull  # type: ignore
                    elif (
                        points_for_hull.shape[0] > 0
                    ):  # Buffer individual points if too few for hull
                        source_geom = _shp.MultiPoint(points_for_hull).buffer(1e-9).buffer(buffer_distance)  # type: ignore
                if source_geom is None:
                    warnings.warn(
                        f"Cannot create buffer from mask region '{source_info.name}', no active points.",
                        UserWarning,
                    )
                    return
        elif isinstance(source_region_name_or_point, np.ndarray):
            point_arr = np.asarray(source_region_name_or_point)
            if point_arr.shape == (2,):
                source_geom = _shp.Point(point_arr)  # type: ignore
            else:
                raise ValueError("If providing a point directly, it must be 2D.")
        else:
            raise TypeError(
                "source_region_name_or_point must be a region name or a 2D point array."
            )

        if source_geom is None:
            warnings.warn(
                f"Could not derive a valid geometry from '{source_region_name_or_point}' to buffer.",
                UserWarning,
            )
            return

        buffered_polygon = source_geom.buffer(buffer_distance)
        self.add_region(name=new_region_name, polygon=buffered_polygon, **metadata)

    def get_region_relationship(
        self,
        region_name1: str,
        region_name2: str,
        relationship_type: str = "intersection",
        add_as_new_region: Optional[str] = None,
        **new_region_metadata,
    ) -> Optional[Any]:  # Shapely geom or bool mask
        """
        Compute geometric relationship (e.g., intersection) between two regions.

        Currently best supported for polygon-defined regions in 2D environments
        using Shapely. Other combinations (e.g., mask-mask) may be added in
        the future.

        Parameters
        ----------
        region_name1 : str
            Name of the first region.
        region_name2 : str
            Name of the second region.
        relationship_type : str, optional
            The type of geometric relationship to compute. Supported for
            polygons: "intersection", "union", "difference",
            "symmetric_difference". Defaults to "intersection".
        add_as_new_region : Optional[str], optional
            If provided, and the result is a valid Shapely Polygon, the
            resulting geometry will be added as a new region with this name.
            Defaults to None (do not add).
        **new_region_metadata : Any
            Metadata for the new region if `add_as_new_region` is specified.

        Returns
        -------
        Optional[shapely.geometry.BaseGeometry]
            The resulting Shapely geometry from the operation. Returns `None`
            if the operation is not supported for the region kinds, if Shapely
            is unavailable, or if the result is an empty geometry.

        Raises
        ------
        RuntimeError
            If Shapely is required but not installed.
        KeyError
            If `region_name1` or `region_name2` are not found.
        ValueError
            If `relationship_type` is unsupported.
        """
        self._check_env_fitted("get_region_relationship")
        if not _HAS_SHAPELY:
            raise RuntimeError("Region relationships require Shapely.")

        info1 = self.get_region_info(region_name1)
        info2 = self.get_region_info(region_name2)

        result_geom: Optional["_shp.BaseGeometry"] = None  # type: ignore

        if info1.kind == "polygon" and info2.kind == "polygon":
            poly1: "_shp.Polygon" = info1.data  # type: ignore
            poly2: "_shp.Polygon" = info2.data  # type: ignore
            if relationship_type == "intersection":
                result_geom = poly1.intersection(poly2)
            elif relationship_type == "union":
                result_geom = poly1.union(poly2)
            elif relationship_type == "difference":
                result_geom = poly1.difference(poly2)
            elif relationship_type == "symmetric_difference":
                result_geom = poly1.symmetric_difference(poly2)
            else:
                raise ValueError(f"Unsupported relationship_type: {relationship_type}")

            if result_geom is not None and result_geom.is_empty:
                warnings.warn(
                    f"'{relationship_type}' of '{region_name1}' and '{region_name2}' is empty.",
                    UserWarning,
                )
                return None

            if add_as_new_region and result_geom is not None and isinstance(result_geom, _shp.Polygon):  # type: ignore
                self.add_region(name=add_as_new_region, polygon=result_geom, **new_region_metadata)  # type: ignore
            return result_geom
        else:
            # TODO: Implement for mask-mask (np.logical_and/or/xor)
            # TODO: Implement for polygon-mask (rasterize polygon or polygonize mask)
            warnings.warn(
                "Region relationships currently best supported for polygon-polygon. Returning None.",
                UserWarning,
            )
            return None

    def plot_regions(
        self,
        ax: matplotlib.axes.Axes,
        region_names: Optional[List[str]] = None,
        default_plot_kwargs: Optional[Dict[str, Any]] = None,
        **specific_region_plot_kwargs,
    ) -> None:
        """
        Plot specified (or all) defined regions onto the provided Matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Matplotlib axes object on which to plot the regions.
        region_names : Optional[List[str]], optional
            A list of names of regions to plot. If None (default), all
            defined regions are plotted.
        default_plot_kwargs : Optional[Dict[str, Any]], optional
            A dictionary of default keyword arguments to apply to all regions
            being plotted (e.g., `{'alpha': 0.5}`). These can be overridden
            by metadata stored in `RegionInfo` or by `specific_region_plot_kwargs`.
        **specific_region_plot_kwargs : Any
            Keyword arguments specific to individual regions. The key should be
            the region name, and the value a dictionary of plot kwargs for that
            region (e.g., `MyRegion={'color': 'blue', 'linestyle': '--'}`).
        """
        self._check_env_fitted("plot_regions")

        if ax is None:
            ax = plt.gca()  # Get current axes if not provided

        regions_to_plot = (
            region_names if region_names is not None else self.list_regions()
        )
        if not regions_to_plot:
            return

        # Check if environment is 2D for polygon plotting
        is_2d_env = self._env.n_dims == 2

        for name in regions_to_plot:
            if name not in self._regions:
                warnings.warn(f"Region '{name}' not found for plotting.", UserWarning)
                continue
            info = self._regions[name]

            # Combine default kwargs, region metadata kwargs, and specific call kwargs
            plot_opts = default_plot_kwargs.copy() if default_plot_kwargs else {}
            plot_opts.update(info.metadata.get("plot_kwargs", {}))
            plot_opts.update(specific_region_plot_kwargs.get(name, {}))

            label = plot_opts.pop("label", name)
            default_color_from_opts = plot_opts.pop("color", None)
            if default_color_from_opts is not None:
                color = default_color_from_opts
            elif hasattr(ax, "_get_lines") and hasattr(ax._get_lines, "get_next_color"):
                # Use the get_next_color() method of the _process_plot_var_args object
                color = ax._get_lines.get_next_color()
            else:
                # Fallback if the above method isn't available (e.g., different Axes type)
                color = None
            alpha = plot_opts.pop("alpha", 0.4)

            if info.kind == "point":
                if info.data.shape[0] >= 2:  # Plot first 2 dims if N-D point
                    ax.scatter(
                        *info.data[:2],
                        label=label,
                        color=color,
                        marker=plot_opts.pop("marker", "x"),
                        s=plot_opts.pop("s", 100),
                        **plot_opts,
                    )
                elif info.data.shape[0] == 1:  # 1D point, plot on y=0 line
                    ax.scatter(
                        info.data[0],
                        0,
                        label=label,
                        color=color,
                        marker=plot_opts.pop("marker", "x"),
                        s=plot_opts.pop("s", 100),
                        **plot_opts,
                    )

            elif info.kind == "polygon":
                if not _HAS_SHAPELY or not is_2d_env:
                    warnings.warn(
                        f"Cannot plot polygon region '{name}'. Requires Shapely and 2D env.",
                        UserWarning,
                    )
                    continue

                from matplotlib.patches import PathPatch  # Local import
                from matplotlib.path import Path as MplPath  # Local import for clarity

                poly_data: "_shp.Polygon" = info.data  # type: ignore

                all_verts_list = []
                all_codes_list = []

                # Process exterior
                # Convert CoordinateSequence to a list of coordinate tuples
                exterior_coords_list = list(poly_data.exterior.coords)

                if (
                    len(exterior_coords_list) >= 2
                ):  # Need at least two points to draw something
                    all_verts_list.extend(exterior_coords_list)
                    all_codes_list.append(MplPath.MOVETO)
                    # Add LINETO for all subsequent points except the last one,
                    # as CLOSEPOLY will handle closing to the first point.
                    all_codes_list.extend(
                        [MplPath.LINETO] * (len(exterior_coords_list) - 2)
                    )
                    all_codes_list.append(MplPath.CLOSEPOLY)  # Close the exterior path
                elif (
                    len(exterior_coords_list) == 1
                ):  # A single point, should not happen for valid polygon
                    all_verts_list.extend(exterior_coords_list)
                    all_codes_list.append(MplPath.MOVETO)

                # Process interiors (holes)
                for interior in poly_data.interiors:
                    interior_coords_list = list(
                        interior.coords
                    )  # Convert CoordinateSequence
                    if len(interior_coords_list) >= 2:
                        all_verts_list.extend(interior_coords_list)
                        all_codes_list.append(MplPath.MOVETO)
                        all_codes_list.extend(
                            [MplPath.LINETO] * (len(interior_coords_list) - 2)
                        )
                        all_codes_list.append(MplPath.CLOSEPOLY)
                    elif len(interior_coords_list) == 1:
                        all_verts_list.extend(interior_coords_list)
                        all_codes_list.append(MplPath.MOVETO)

                if not all_verts_list:
                    warnings.warn(
                        f"Polygon region '{name}' resulted in no vertices to plot.",
                        UserWarning,
                    )
                    continue

                # Create the Path object
                # Vertices should be a NumPy array for Matplotlib Path
                path = MplPath(np.array(all_verts_list), all_codes_list)
                patch = PathPatch(
                    path, label=label, facecolor=color, alpha=alpha, **plot_opts
                )
                ax.add_patch(patch)

            elif info.kind == "mask":
                # Plotting an N-D mask region is best done by coloring the active bins
                # that fall within this mask.
                active_bins_in_this_region = self.bins_in_region(name)
                if active_bins_in_this_region.size > 0:
                    centers_to_plot = self._env.bin_centers_[active_bins_in_this_region]
                    if centers_to_plot.shape[1] >= 2:  # Plot first 2 dims
                        ax.scatter(
                            centers_to_plot[:, 0],
                            centers_to_plot[:, 1],
                            label=label,
                            color=color,
                            alpha=alpha,
                            s=plot_opts.pop("s", 10),  # Smaller default for many bins
                            marker=plot_opts.pop("marker", "."),
                            **plot_opts,
                        )
                    elif centers_to_plot.shape[1] == 1:
                        ax.scatter(
                            centers_to_plot[:, 0],
                            np.zeros_like(centers_to_plot[:, 0]),
                            label=label,
                            color=color,
                            alpha=alpha,
                            s=plot_opts.pop("s", 10),
                            marker=plot_opts.pop("marker", "."),
                            **plot_opts,
                        )
                else:
                    warnings.warn(
                        f"Mask region '{name}' contains no active bins to plot.",
                        UserWarning,
                    )

        # Add legend if any labels were generated
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # Only add legend if there's something to label
            ax.legend(handles, labels)
