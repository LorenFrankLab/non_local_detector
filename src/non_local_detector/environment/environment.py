from __future__ import annotations

import pickle
import warnings
from dataclasses import asdict, dataclass, field
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from non_local_detector.environment.layout_engine import (
    GraphLayout,
    LayoutEngine,
    RegularGridLayout,
    create_layout,
)
from non_local_detector.environment.region import RegionInfo, RegionManager
from non_local_detector.environment.utils import _get_distance_between_bins

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
@dataclass(repr=False)
class Environment:
    """
    Represents a discretized N-dimensional space with connectivity.

    Use classmethod factories to create instances, for example:
    - `Environment.from_data_samples(...)`
    - `Environment.with_dimension_ranges(...)`
    - `Environment.from_track_definition(...)`
    - `Environment.from_shapely_polygon(...)`
    - `Environment.from_nd_mask(...)`
    - `Environment.from_image_mask(...)`
    """

    environment_name: str
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
    regions: RegionManager = field(init=False, repr=False)

    # Internal state
    _is_1d_env: bool = field(init=False)
    _is_fitted: bool = field(init=False, default=False)

    # For introspection and serialization
    _layout_type_used: Optional[str] = field(init=False, default=None)
    _layout_params_used: Dict[str, Any] = field(init=False, default_factory=dict)

    # Primary constructor - intended for use by factory methods
    def __init__(
        self,
        environment_name: str = "",
        layout: LayoutEngine = RegularGridLayout,
        layout_type_used: Optional[str] = None,
        layout_params_used: Optional[Dict[str, Any]] = None,
    ):
        """
        Internal constructor. Users should use factory methods.
        Assumes `layout` is already built.
        """
        self.environment_name = environment_name
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
        self.regions = RegionManager(self)

    def __html_repr__(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        self.plot(ax=ax)

    def _setup_from_layout(self):
        """Populates Environment attributes from its (built) LayoutEngine."""

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

    # --- Factory Methods ---
    @classmethod
    def from_data_samples(
        cls,
        data_samples: NDArray[np.float64],
        environment_name: str = "",
        layout_type: str = "RegularGrid",
        # Common params (defaults should match typical use)
        bin_size: Optional[Union[float, Sequence[float]]] = 2.0,
        # For RegularGrid/Shapely(Grid base)
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        # Common RegularGrid inference params
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
        **layout_specific_kwargs: Any,
    ) -> Environment:
        """Creates an Environment, primarily inferring geometry from data_samples."""
        build_params: Dict[str, Any] = {
            "data_samples": data_samples,  # dimension_ranges will be inferred by layout
            "infer_active_bins": infer_active_bins,
            "bin_count_threshold": bin_count_threshold,
            **layout_specific_kwargs,  # Must come before specific overrides below
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
            build_params.update(
                {
                    "hex_width": bin_size,  # Assuming bin_size is hex_width for hexagonal
                }
            )

        layout_instance = create_layout(kind=layout_type, **build_params)
        return cls(environment_name, layout_instance, layout_type, build_params)

    @classmethod
    def with_dimension_ranges(
        cls,
        dimension_ranges: Sequence[Tuple[float, float]],
        environment_name: str = "",
        layout_type: str = "RegularGrid",
        bin_size: Optional[Union[float, Sequence[float]]] = 2.0,
        **layout_specific_kwargs: Any,
    ) -> Environment:
        """Creates an Environment with explicitly defined spatial boundaries."""
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
        return cls(environment_name, layout_instance, layout_type, build_params)

    # --- Specialized Factories (signatures as discussed previously) ---
    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        bin_size: float,
        environment_name: str = "",
        **kwargs,
    ) -> Environment:
        layout_instance = create_layout(
            kind="Graph",
            graph_definition=graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            bin_size=bin_size,
        )
        return cls(environment_name, layout_instance, "Graph", kwargs)

    @classmethod
    def from_shapely_polygon(
        cls,
        polygon: PolygonType,
        bin_size: Optional[Union[float, Sequence[float]]] = 2.0,
        environment_name: str = "",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
        layout_params = {
            "polygon": polygon,
            "bin_size": bin_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }
        layout_instance = create_layout(kind="ShapelyPolygon", **layout_params)
        return cls(environment_name, layout_instance, "ShapelyPolygon", layout_params)

    @classmethod
    def from_nd_mask(
        cls,
        environment_name: str = "",
        active_mask: NDArray[np.bool_] = None,
        grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None,
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
        layout_params = {
            "active_mask": active_mask,
            "grid_edges": grid_edges,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }
        layout_instance = create_layout(kind="MaskedGrid", **layout_params)
        return cls(environment_name, layout_instance, "MaskedGrid", layout_params)

    @classmethod
    def from_image_mask(
        cls,
        image_mask: NDArray[np.bool_],  # Defines candidate pixels
        bin_size: Union[float, Tuple[float, float]] = 1.0,  # one pixel
        connect_diagonal_neighbors: bool = True,
        environment_name: str = "",
        **kwargs,
    ) -> Environment:
        """Creates an Environment from a binary image mask."""
        layout_params = {
            "image_mask": image_mask,
            "bin_size": bin_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }

        layout_instance = create_layout(kind="ImageMask", **layout_params)
        return cls(environment_name, layout_instance, "ImageMask", layout_params)

    # Fallback factory for advanced use or deserialization
    @classmethod
    def from_custom_layout(
        cls,
        layout_type: str,
        layout_params: Dict[str, Any],
        environment_name: str = "",
    ) -> Environment:
        """Creates Environment with any specified layout and its build parameters."""
        layout_instance = create_layout(kind=layout_type, **layout_params)
        return cls(environment_name, layout_instance, layout_type, layout_params)

    @property
    def is_1d(self) -> bool:
        """True if the environment's layout structure is primarily 1-dimensional."""
        return self._is_1d_env  # Set in __init__ from layout.is_1d

    @property
    @check_fitted
    def n_dims(self) -> int:
        """Number of spatial dimensions of the active bin centers."""
        if self.bin_centers_ is None:  # Should be caught by check_fitted
            raise RuntimeError("Layout not fitted or bin_centers_ not available.")
        return self.bin_centers_.shape[1]

    @check_fitted
    def get_connectivity_graph(self) -> nx.Graph:
        """
        Returns the primary connectivity graph of active/interior bins.
        Nodes in this graph (0 to N-1) directly correspond to rows in
        `self.bin_centers_`.
        """
        if self.connectivity_graph_ is None:
            raise ValueError("Connectivity graph is not available.")
        return self.connectivity_graph_

    @cached_property  # type: ignore[attr-defined]
    @check_fitted
    def distance_between_bins(self) -> NDArray[np.float64]:
        """Shortest path distances between all pairs of active bins."""
        return _get_distance_between_bins(self.get_connectivity_graph())

    @check_fitted
    def get_bin_ind(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        return self.layout.point_to_bin_index(points_nd)

    @check_fitted
    def is_point_active(self, points_nd: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Checks if the given points are within active bins."""
        return self.get_bin_ind(points_nd) != -1

    @check_fitted
    def get_bin_neighbors(self, bin_index: int) -> List[int]:
        """Finds indices of neighboring active bins for a given active bin index."""
        return self.layout.get_bin_neighbors(bin_index)

    @check_fitted
    def get_bin_area_volume(self) -> NDArray[np.float64]:
        """Calculates the area/volume of a given active bin."""
        return self.layout.get_bin_area_volume()

    @check_fitted
    def get_bin_attributes_dataframe(self) -> pd.DataFrame:  # Renamed
        """Creates a DataFrame with information about each active bin."""
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
        # ... (Implementation as before, uses self.get_bin_ind and self.distance_between_bins) ...
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
    def get_linearized_coordinate(
        self, points_nd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self.is_1d or not isinstance(self.layout, GraphLayout):
            raise TypeError("Linearized coordinate only for GraphLayout environments.")
        return self.layout.get_linearized_coordinate(points_nd)

    @check_fitted
    def map_linear_to_nd_coordinate(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self.is_1d or not isinstance(self.layout, GraphLayout):
            raise TypeError("Mapping linear to N-D only for GraphLayout environments.")
        return self.layout.map_linear_to_nd_coordinate(linear_coordinates)

    @check_fitted
    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        show_regions: bool = False,
        layout_plot_kwargs: Optional[Dict[str, Any]] = None,
        regions_plot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the environment's layout and optionally defined regions."""
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

        ax = self.layout.plot(ax=ax, **l_kwargs)

        if show_regions and hasattr(self, "regions") and self.regions is not None:
            r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
            self.regions.plot_regions(ax=ax, **r_kwargs)

        plot_title = self.environment_name
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
        **kwargs,
    ):
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
        if self.layout.is_1d:
            ax = self.layout.plot_linear_layout(ax=ax, **l_kwargs)

        return ax

    def save(self, filename: str = "environment.pkl") -> None:
        """Saves the Environment object to a file using pickle."""
        with open(filename, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Environment saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> Environment:
        """Loads an Environment object from a pickled file."""
        with open(filename, "rb") as fh:
            environment = pickle.load(fh)
        if not isinstance(environment, cls):
            raise TypeError(f"Loaded object is not type {cls.__name__}")
        return environment

    # --- Serialization ---
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Environment object to a dictionary."""
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
            "environment_name": self.environment_name,
            "_layout_type_used": self._layout_type_used,
            "_layout_params_used": serializable_layout_params,
            "_regions_data": [asdict(info) for info in self.regions._regions.values()],
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Environment:
        """Deserializes an Environment object from a dictionary."""
        if not (
            data.get("__classname__") == cls.__name__
            and data.get("__module__") == cls.__module__
        ):
            raise ValueError("Dictionary is not for this Environment class.")

        env_name = data["environment_name"]
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
            environment_name=env_name,
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
                    env.regions.add_region(
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
    def flat_to_nd_bin_index(
        self, flat_indices: Union[int, NDArray[np.int_]]
    ) -> Union[Tuple[int, ...], Tuple[NDArray[np.int_], ...]]:
        """
        Converts active bin flat indices (0..N-1) to their original N-D grid indices.
        Only meaningful for grid-based layouts that have an underlying full grid.
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
    def nd_to_flat_bin_index(
        self, *nd_idx_per_dim: Union[int, NDArray[np.int_]]
    ) -> Union[int, NDArray[np.int_]]:
        """
        Converts N-D grid indices (of the original full grid) to active bin flat indices (0..N-1).
        Returns -1 if the N-D index is outside bounds or not part of an active bin.
        Only meaningful for grid-based layouts.
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
            # Input like env.nd_to_flat_bin_index( ([r1,r2],[c1,c2]) ) or env.nd_to_flat_bin_index( ( (r1,c1), (r2,c2) ) )
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

        # Map these original_flat_indices to the new active bin indices (0 to N-1)
        if not hasattr(self, "_source_flat_to_active_node_id_map_cached"):
            self._source_flat_to_active_node_id_map_cached = {  # Cache this map
                data["source_grid_flat_index"]: node_id
                for node_id, data in self.connectivity_graph_.nodes(data=True)
                if "source_grid_flat_index" in data
            }

        final_active_bin_ids = np.array(
            [
                self._source_flat_to_active_node_id_map_cached.get(orig_flat_idx, -1)
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
