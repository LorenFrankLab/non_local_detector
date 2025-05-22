"""Manages the region of interest (ROI) annotation for the Environment class."""

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
    points: NDArray[np.float_], polygon: _shp.Polygon
) -> NDArray[np.bool_]:
    """
    Checks if points are inside a polygon using Shapely.

    Parameters
    ----------
    points : NDArray[np.float_]
        Array of shape (n_points, 2) containing the coordinates of the points.
    polygon : _shp.Polygon
        Shapely Polygon object.

    Returns
    -------
    NDArray[np.bool_]
        Boolean array indicating whether each point is inside the polygon.
    """
    if not _HAS_SHAPELY:
        raise RuntimeError("Shapely is required for polygon operations.")
    if points.shape[0] == 0:
        return np.array([], dtype=bool)
    return np.array([polygon.contains(_shp.Point(p)) for p in points])


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
        One of ``{"point", "mask", "polygon"}``.
    data : Any
        The geometric data defining the region:
        - If `kind` is "point": `NDArray[np.float_]` of shape `(n_dims,)`.
        - If `kind` is "mask": `NDArray[np.bool_]` with N-D shape matching the
          environment's full `grid_shape_` (for grid-based layouts).
        - If `kind` is "polygon": A `shapely.geometry.Polygon` object (for 2D layouts).
    metadata : dict
        Arbitrary key-value store for additional information about the region.
    """

    name: str
    kind: str
    data: Any
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
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
    Manages symbolic spatial regions within an Environment.

    This class provides methods to define regions using points, masks, or
    polygons, and to query relationships between these regions and the
    environment's discretized bins.
    """

    def __init__(self, environment_facade: "Environment"):
        """
        Initializes the RegionManager.

        Parameters
        ----------
        environment_facade : Environment
            A reference to the Environment instance this manager serves.
            This provides access to the environment's geometric properties
            (e.g., bin_centers_, grid_shape_, active_mask_, get_bin_ind, n_dims).
        """
        self._env = environment_facade  # Facade to access Environment's state
        self._regions: Dict[str, RegionInfo] = {}

    def _check_env_fitted(self, method_name: str):
        """Helper to check if the associated environment is fitted."""
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
        Registers a region with a unique name.

        Exactly one of `point`, `mask`, or `polygon` must be provided to define
        the region's geometry.

        Parameters
        ----------
        name : str
            Unique identifier for the region.
        point : Optional[Union[Tuple[float, ...], NDArray[np.float_]]], optional
            Coordinates (same dimensionality as environment) defining a point region.
            The region will effectively be the active bin closest to this point.
        mask : Optional[NDArray[np.bool_]], optional
            Boolean array defining the region. For grid-based layouts, this mask
            must match the environment's full `grid_shape_` and applies to the
            conceptual full grid.
        polygon : Optional[Union[shapely.geometry.Polygon, List[Tuple[float,float]]]], optional
            A Shapely Polygon object or a list of (x, y) vertex tuples defining
            a polygonal region. Applicable primarily for 2D environments. Bins whose
            centers fall within this polygon are considered part of the region.
        metadata : dict, optional
            Additional key-value pairs to store with the region.

        Raises
        ------
        ValueError
            If `name` already exists, if not exactly one geometric specifier is
            provided, or if inputs are inconsistent with the environment.
        RuntimeError
            If `polygon` is specified but Shapely is not installed.
        TypeError
            If data for a kind is of the wrong type.
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
        """Removes region `name` from the registry. Silently ignores if absent."""
        self._regions.pop(name, None)

    def list_regions(self) -> List[str]:
        """Returns a list of all registered region names, in insertion order."""
        return list(self._regions.keys())

    def get_region_info(self, name: str) -> RegionInfo:
        """Retrieves the `RegionInfo` object for a named region."""
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found.")
        return self._regions[name]

    @property
    def environment(self) -> "Environment":
        """Provides access to the parent Environment instance."""
        return self._env

    @environment.setter
    def environment(self, value: "Environment"):
        """Sets the parent Environment instance (should typically be done once)."""
        if hasattr(self, "_env") and self._env is not None:
            warnings.warn(
                "RegionManager is already associated with an Environment. Overwriting.",
                UserWarning,
            )
        self._env = value

    def region_mask(self, name: str) -> NDArray[np.bool_]:
        """
        Returns a 1D boolean mask indicating which *active bins* of the
        environment (indexed 0 to N-1) are part of the specified region.
        """
        self._check_env_fitted("region_mask")
        info = self.get_region_info(name)

        n_active_bins = self._env.bin_centers_.shape[0]
        active_bin_is_in_region_1d = np.zeros(n_active_bins, dtype=bool)

        if n_active_bins == 0:  # No active bins in the environment
            return active_bin_is_in_region_1d

        if info.kind == "point":
            # Find which active bin the point maps to
            mapped_bin_idx_arr = self._env.get_bin_ind(info.data)  # Returns array
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
                # This requires a map from source_grid_flat_index to active_bin_id
                if not hasattr(self._env, "_source_flat_to_active_node_id_map_cached"):
                    # Cache this map on the environment if not present
                    self._env._source_flat_to_active_node_id_map_cached = {  # type: ignore
                        data["source_grid_flat_index"]: node_id
                        for node_id, data in self._env.connectivity_graph_.nodes(
                            data=True
                        )
                        if "source_grid_flat_index" in data
                    }

                for orig_flat_idx in original_flat_indices_in_region_and_active:
                    active_node_id = (
                        self._env._source_flat_to_active_node_id_map_cached.get(
                            orig_flat_idx
                        )
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
        """Returns active bin indices (0..N-1) that are part of region `name`."""
        self._check_env_fitted("bins_in_region")
        active_bin_mask_1d = self.region_mask(name)
        return np.flatnonzero(active_bin_mask_1d).astype(np.int_)

    def region_center(self, name: str) -> Optional[NDArray[np.float64]]:
        """
        Calculates the geometric center of the active bins within the specified region.
        For "point" regions, returns the point itself.
        Returns None if the region is empty or not defined.
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
        Finds the named region whose center is closest (Euclidean distance)
        to the given query `position`.

        Parameters
        ----------
        position : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The query position.
        candidate_region_names : Optional[List[str]], optional
            If provided, search only among these named regions.
            Defaults to None (search all defined regions).

        Returns
        -------
        Optional[str]
            The name of the nearest region, or None if no regions are defined
            or no suitable candidates are found.
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
        Calculates the area of a defined region (primarily for 2D layouts).

        For polygon regions, uses Shapely's area.
        For mask regions on a grid, sums the area of active bins within the region.
        Returns None if area cannot be determined or region is not 2D/grid-like.
        """
        self._check_env_fitted("get_region_area")
        info = self.get_region_info(region_name)

        if info.kind == "polygon":
            if _HAS_SHAPELY and self._env.n_dims == 2:
                return info.data.area  # type: ignore
            else:
                warnings.warn(
                    "Cannot calculate area for polygon region: Shapely not available or not 2D env.",
                    UserWarning,
                )
                return None
        elif info.kind == "mask":
            if self._env.layout and hasattr(self._env.layout, "get_bin_area_volume"):
                active_bins_in_reg_indices = self.bins_in_region(region_name)
                if active_bins_in_reg_indices.size == 0:
                    return 0.0

                # This requires layout.get_bin_area_volume() to return areas for *active* bins (0..N-1)
                # Or, if it returns areas for original grid bins, we need mapping.
                # Simpler: if all bins in a grid layout have same area:
                if (
                    isinstance(self._env.layout, _GridMixin)
                    and self._env.grid_edges_
                    and len(self._env.grid_edges_) == self._env.n_dims
                ):
                    bin_areas_per_dim = [
                        np.diff(edges_dim)[0] for edges_dim in self._env.grid_edges_
                    ]  # Assumes uniform bins
                    single_bin_area = np.prod(bin_areas_per_dim)
                    num_active_bins_in_mask_region = np.sum(
                        self.region_mask(region_name)
                    )  # This mask is 1D on active bins
                    return num_active_bins_in_mask_region * single_bin_area
                else:  # Fallback if per-bin area isn't straightforward
                    warnings.warn(
                        f"Area calculation for mask region '{region_name}' on non-uniform or non-grid layout not fully supported yet.",
                        UserWarning,
                    )
                    return None  # Placeholder for more complex area sum
            else:
                warnings.warn(
                    f"Layout type for mask region '{region_name}' does not support area calculation.",
                    UserWarning,
                )
                return None
        elif info.kind == "point":
            return 0.0  # A point has no area
        return None

    def create_buffered_region(
        self,
        source_region_name_or_point: Union[str, NDArray[np.float_]],
        buffer_distance: float,
        new_region_name: str,
        **metadata,
    ) -> None:
        """Creates a new polygonal region by buffering an existing region or a point (2D only)."""
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
        Computes geometric relationship (intersection, union, difference) between two regions.
        Currently best supported for polygon regions in 2D.
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
        """Plots specified (or all) defined regions onto the provided axes."""
        self._check_env_fitted("plot_regions")

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
            color = plot_opts.pop(
                "color",
                (
                    next(ax._get_lines.prop_cycler)["color"]
                    if hasattr(ax, "_get_lines")
                    else None
                ),
            )  # Cycle color
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
                from matplotlib.path import Path as MplPath  # Local import

                poly_data: "_shp.Polygon" = info.data  # type: ignore
                path_data = [MplPath.MOVETO] + poly_data.exterior.coords.tolist()  # type: ignore
                for interior in poly_data.interiors:  # type: ignore
                    path_data.extend([MplPath.MOVETO] + interior.coords.tolist())

                codes, verts = zip(*path_data)
                path = MplPath(verts, codes)
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
