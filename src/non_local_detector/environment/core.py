"""
environment/core.py
===================

"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:  # avoid run-time circular dep
    from matplotlib.axes import Axes as MatplotlibAxes

    from ..regions.core import Regions  # noqa: F401


# ---------------------------------------------------------------------
# type aliases
PtArr = NDArray[np.float_]
IdxArr = NDArray[np.int_]

PointToBin = Callable[["Environment", PtArr], IdxArr]
PlotFunc = Callable[["Environment", "Any"], None]
AreaFn = Callable[["Environment"], NDArray[np.float_]]

# For 1D specific helpers
LinearProjectionFn = Callable[["Environment", PtArr], PtArr]
LinearToBinFn = Callable[["Environment", PtArr], IdxArr]


@dataclass(frozen=True)
class LayoutResult:
    """
    Dataclass specifying the output structure of a layout engine.
    """

    bin_centers_: NDArray[np.float64]  # Shape (N, D)
    connectivity_graph_: nx.Graph
    dimension_ranges_: Tuple[Tuple[float, float], ...]
    is_1d_: bool
    point_to_bin_func_: PointToBin
    active_bin_areas_: NDArray[np.float64]

    # Optional grid metadata
    grid_edges_: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    active_mask_nd_: Optional[NDArray[np.bool_]] = None

    # Plotting
    plot_layout_func_: Optional[PlotFunc] = None
    plot_1d_layout_func_: Optional[PlotFunc] = None

    # 1D specific attributes
    graph_definition_: Optional[nx.Graph] = None
    edge_order_: Optional[Tuple[Tuple[int, int], ...]] = None
    edge_spacing_: Optional[float] = None
    lin_func_: Optional[LinearProjectionFn] = None
    lin_to_nd_func_: Optional[LinearToBinFn] = None
    nd_points_to_linear_func_: Optional[LinearProjectionFn] = None

    def __post_init__(self: LayoutResult) -> None:
        if self.bin_centers_.ndim != 2:
            raise ValueError("LayoutResult: bin_centers_ must be 2-D (n_bins, n_dims).")
        n_bins, _ = self.bin_centers_.shape
        if self.connectivity_graph_.number_of_nodes() != n_bins:
            raise ValueError(
                "LayoutResult: connectivity_graph_ must have same node count as bin_centers_ rows."
            )

        if self.is_1d_:
            if self.graph_definition_ is None:
                warnings.warn(
                    "LayoutResult: graph_definition_ is missing for a 1D layout.",
                    UserWarning,
                )
            if self.edge_order_ is None:
                warnings.warn(
                    "LayoutResult: edge_order_ is missing for a 1D layout.", UserWarning
                )
            if self.edge_spacing_ is None:
                warnings.warn(
                    "LayoutResult: edge_spacing_ is missing for a 1D layout.",
                    UserWarning,
                )
            if self.lin_func_ is None:
                warnings.warn(
                    "LayoutResult: lin_func_ is missing for a 1D layout.", UserWarning
                )
            if self.lin_to_nd_func_ is None:
                warnings.warn(
                    "LayoutResult: lin_to_nd_func_ is missing for a 1D layout.",
                    UserWarning,
                )

        if self.active_mask_nd_ is not None and self.grid_shape_ is not None:
            if self.active_mask_nd_.shape != self.grid_shape_:
                raise ValueError(
                    "LayoutResult: active_mask_nd_ shape must equal grid_shape_."
                )

        for helper_name in [
            "point_to_bin_func_",
        ]:
            if not callable(getattr(self, helper_name)):
                raise TypeError(
                    f"LayoutResult: Required helper '{helper_name}' must be callable."
                )
        if self.plot_layout_func_ is not None and not callable(self.plot_layout_func_):
            raise TypeError(
                "LayoutResult: 'plot_layout_func_' must be callable if provided."
            )
        if self.plot_1d_layout_func_ is not None and not callable(
            self.plot_1d_layout_func_
        ):
            raise TypeError(
                "LayoutResult: 'plot_1d_layout_func_' must be callable if provided."
            )


class LinearAdapter:
    __slots__ = ("_env", "_graph_definition", "_edge_order", "_edge_spacing")

    def __init__(
        self,
        env: "Environment",
        graph_definition: nx.Graph,
        edge_order: Sequence[tuple[int, int]],
        edge_spacing: float,
    ):
        self._env = env
        self._graph_definition = graph_definition
        self._edge_order = tuple(edge_order)
        self._edge_spacing = float(edge_spacing)

    @property
    def graph_definition(self) -> nx.Graph:
        return self._graph_definition

    @property
    def edge_order(self) -> tuple[tuple[int, int], ...]:
        return self._edge_order

    @property
    def edge_spacing(self) -> float:
        return self._edge_spacing

    def to_linear(self, bin_ids: IdxArr) -> PtArr:
        if self._env._lin is None:
            raise AttributeError(
                "'_lin' function not available on the environment for this 1D layout."
            )
        return self._env._lin(self._env, np.asarray(bin_ids, dtype=int))

    def to_bin(self, linear_points: PtArr) -> IdxArr:
        if self._env._lin_to_nd is None:
            raise AttributeError(
                "'_lin_to_nd' function not available on the environment for this 1D layout."
            )
        return self._env._lin_to_nd(self._env, np.asarray(linear_points, dtype=float))

    def from_points(self, points: PtArr) -> PtArr:
        return self.to_linear(self._env.bin_at(points))

    def to_linear_cont(self, points: PtArr) -> PtArr:
        if self._env._nd_points_to_linear_func_ is None:
            raise AttributeError(
                "Continuous projection ('_nd_points_to_linear_func_') unavailable on the environment for this layout."
            )
        return self._env._nd_points_to_linear_func_(
            self._env, np.asarray(points, dtype=float)
        )

    def plot(self, ax: Optional[MatplotlibAxes] = None, **kw: Any) -> Any:
        if self._env._plot_1d is None:
            raise AttributeError(
                "1D plotting function ('_plot_1d') not available on the environment."
            )
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        self._env._plot_1d(self._env, ax, **kw)
        return ax


# ---------------------------------------------------------------------
# Environment dataclass
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Environment:
    name: str
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: Tuple[Tuple[float, float], ...]
    is_1d: bool = False
    regions: Optional[Regions] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _pt2bin: PointToBin
    _area: AreaFn
    _plot: Optional[PlotFunc] = None
    _plot_1d: Optional[PlotFunc] = field(default=None, repr=False)

    _graph_definition: Optional[nx.Graph] = field(default=None, repr=False)
    _edge_order: Optional[Tuple[Tuple[int, int], ...]] = field(default=None, repr=False)
    _edge_spacing: Optional[float] = field(default=None, repr=False)
    _lin: Optional[LinearProjectionFn] = field(default=None, repr=False)
    _lin_to_nd: Optional[LinearToBinFn] = field(default=None, repr=False)
    _nd_points_to_linear_func_: Optional[LinearProjectionFn] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        centers = self.bin_centers
        if centers.ndim != 2:
            raise ValueError("bin_centers must be 2-D (n_bins, n_dims).")
        n_bins, n_dims = centers.shape

        if self.connectivity.number_of_nodes() != n_bins:
            raise ValueError("connectivity must have same node count as bin rows.")

        if len(self.dimension_ranges) != n_dims:
            raise ValueError("dimension_ranges must have one (min,max) per column.")

        if self.active_mask is not None and self.grid_shape is not None:
            if self.active_mask.shape != self.grid_shape:
                raise ValueError("active_mask shape must equal grid_shape.")

        for helper_name in ["_pt2bin", "_area"]:
            if not callable(getattr(self, helper_name)):
                raise TypeError(f"Injected helper '{helper_name}' must be callable.")
        if self._plot is not None and not callable(self._plot):
            raise TypeError("Injected helper '_plot' must be callable if provided.")
        if self._plot_1d is not None and not callable(self._plot_1d):
            raise TypeError("Injected helper '_plot_1d' must be callable if provided.")

    def __repr__(self) -> str:
        return (
            f"Environment(name={self.name!r}, n_bins={self.n_bins}, "
            f"n_dims={self.n_dims}, is_1d={self.is_1d}, regions={self.regions is not None})"
        )

    @classmethod
    def from_layout(
        cls,
        layout_result: LayoutResult,
        *,
        name: str = "env",
        regions: Optional["Regions"] = None,
    ) -> "Environment":
        """
        Convenience: promote LayoutResult to immutable Environment.
        """
        return cls(
            name=name,
            bin_centers=layout_result.bin_centers_,
            connectivity=layout_result.connectivity_graph_,
            dimension_ranges=layout_result.dimension_ranges_,
            is_1d=layout_result.is_1d_,
            regions=regions,
            grid_edges=layout_result.grid_edges_,
            grid_shape=layout_result.grid_shape_,
            active_mask=layout_result.active_mask_nd_,
            _pt2bin=layout_result.point_to_bin_func_,
            _area=lambda s: layout_result.active_bin_areas_,
            _plot=layout_result.plot_layout_func_,
            _plot_1d=layout_result.plot_1d_layout_func_,
            _graph_definition=layout_result.graph_definition_,
            _edge_order=layout_result.edge_order_,
            _edge_spacing=layout_result.edge_spacing_,
            _lin=layout_result.lin_func_,
            _lin_to_nd=layout_result.lin_to_nd_func_,
            _nd_points_to_linear_func_=layout_result.nd_points_to_linear_func_,
        )

    @property
    def linear(self) -> LinearAdapter:
        if (
            not self.is_1d
            or self._graph_definition is None
            or self._edge_order is None
            or self._edge_spacing is None
        ):
            raise AttributeError(
                "This environment does not carry a complete linear graph definition or is not 1D."
            )
        return LinearAdapter(
            self,
            self._graph_definition,
            self._edge_order,
            self._edge_spacing,
        )

    def with_regions(self, regions: "Regions") -> "Environment":
        return dataclasses.replace(self, regions=regions)

    @property
    def n_bins(self) -> int:
        return self.bin_centers.shape[0]

    @property
    def n_dims(self) -> int:
        return self.bin_centers.shape[1]

    def bin_at(self, points: PtArr) -> IdxArr:
        return self._pt2bin(self, np.asarray(points, dtype=float))

    def contains(self, points: PtArr) -> NDArray[np.bool_]:
        return self.bin_at(points) != -1

    def bin_center_of(self, bin_indices: IdxArr) -> PtArr:
        return self.bin_centers[np.asarray(bin_indices, dtype=int)]

    def neighbors(self, bin_index: int) -> list[int]:
        return list(self.connectivity.neighbors(bin_index))

    def bin_area(self) -> NDArray[np.float64]:
        return self._area(self).astype(float, copy=False)

    def mask_for_region(self, name: str) -> NDArray[np.bool_]:
        if self.regions is None:
            raise AttributeError("Environment has no Regions attached.")
        from ..regions.core import _HAS_SHAPELY as _REGIONS_HAS_SHAPELY

        if _REGIONS_HAS_SHAPELY:
            from shapely.geometry import Point as ShapelyPoint  # type: ignore

        reg = self.regions[name]
        mask = np.zeros(self.n_bins, dtype=bool)

        if reg.kind == "polygon":
            if not _REGIONS_HAS_SHAPELY:
                raise RuntimeError(
                    "Polygon regions require Shapely, which is not installed."
                )
            if self.n_dims != 2 or reg.n_dims != 2:
                warnings.warn(
                    f"Polygon region '{reg.name}' and Environment must be 2D for mapping."
                )
                return mask
            from shapely import points as shapely_points_vec  # type: ignore

            mask[:] = reg.data.contains(shapely_points_vec(self.bin_centers))
        elif reg.kind == "point":
            if reg.n_dims != self.n_dims:
                warnings.warn(
                    f"Point region '{reg.name}' dimensionality ({reg.n_dims}) "
                    f"mismatches Environment ({self.n_dims}). Cannot map accurately."
                )
                return mask
            idx = self.bin_at(reg.data).item()
            if idx != -1 and 0 <= idx < self.n_bins:
                mask[idx] = True
        else:
            warnings.warn(
                f"Unsupported region kind '{reg.kind}' for mask generation.",
                UserWarning,
            )
        return mask

    def get_indices_for_points_in_region(
        self, points: PtArr, region_name: str
    ) -> NDArray[np.bool_]:
        """
        Determines which input points correspond to locations falling
        within a specified named region of this environment.

        A point is considered in the region if it maps to an active bin
        that is part of the defined region.

        Parameters
        ----------
        points : PtArr, shape (n_points, n_dims)
            N-dimensional point data.
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask of shape (n_points,). `True` indicates the
            point at that index is within the specified region.

        Raises
        ------
        AttributeError
            If `self.regions` is None.
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If `points` array has incorrect dimensionality for `self.bin_at`.
        """
        if self.regions is None:
            raise AttributeError(f"Environment '{self.name}' has no 'regions' manager.")

        active_bin_indices_for_points = self.bin_at(points)
        region_membership_for_active_bins = self.mask_for_region(region_name)

        output_mask = np.zeros(points.shape[0], dtype=bool)
        validly_mapped_points_mask = active_bin_indices_for_points != -1

        if np.any(validly_mapped_points_mask):
            # Get the actual active bin indices for these validly mapped points
            mapped_active_bin_ids = active_bin_indices_for_points[
                validly_mapped_points_mask
            ]

            # Check region membership for these specific active bin IDs
            # Ensure mapped_active_bin_ids are valid indices for region_membership_for_active_bins
            if np.any(
                (mapped_active_bin_ids < 0)
                | (mapped_active_bin_ids >= len(region_membership_for_active_bins))
            ):
                warnings.warn(
                    "Some bin indices from bin_at were out of bounds for region_membership_for_active_bins. "
                    "This indicates an internal inconsistency.",
                    RuntimeWarning,
                )
                # Proceed cautiously or raise an error, for now, let indexing handle it (may error)
                # A robust way is to filter mapped_active_bin_ids further:
                valid_indices_for_lookup = (mapped_active_bin_ids >= 0) & (
                    mapped_active_bin_ids < len(region_membership_for_active_bins)
                )

                true_mapped_active_bin_ids = mapped_active_bin_ids[
                    valid_indices_for_lookup
                ]

                # Create a temporary mask for where to place results from valid_indices_for_lookup
                temp_placement_mask = np.zeros(len(mapped_active_bin_ids), dtype=bool)
                temp_placement_mask[valid_indices_for_lookup] = True

                # Update the relevant part of output_mask
                # First, get the subset of validly_mapped_points_mask where true_mapped_active_bin_ids apply
                final_placement_indices = np.where(validly_mapped_points_mask)[0][
                    temp_placement_mask
                ]
                output_mask[final_placement_indices] = (
                    region_membership_for_active_bins[true_mapped_active_bin_ids]
                )

            else:  # All mapped_active_bin_ids are valid indices
                output_mask[validly_mapped_points_mask] = (
                    region_membership_for_active_bins[mapped_active_bin_ids]
                )

        return output_mask

    @cached_property
    def get_all_euclidean_distances(self) -> NDArray[np.float64]:
        """
        Calculate the Euclidean distance between all active points
        in the environment.

        Returns
        -------
        euclidean_distances: NDArray[np.float64], shape (n_active_bins, n_active_bins)
            Distance matrix containing the pairwise distances
            between active bin centers.
        """
        if self.n_bins == 0:
            return np.empty((0, 0), dtype=float)
        if self.n_bins == 1:
            return np.zeros((self.n_bins, self.n_bins), dtype=float)

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
            Returns np.inf if no path exists.
        """
        if self.connectivity is None or self.n_bins == 0:
            return np.empty((0, 0), dtype=np.float64)

        dist_matrix = np.full((self.n_bins, self.n_bins), np.inf, dtype=np.float64)
        np.fill_diagonal(dist_matrix, 0.0)

        # path_lengths is an iterator of (source_node, {target_node: length})
        path_lengths = nx.all_pairs_shortest_path_length(
            self.connectivity, weight="distance"
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

        This method utilizes the 'connectivity' graph of the environment and
        delegates to `compute_diffusion_kernels`. The resulting matrix
        can represent the influence or probability flow between bins after
        a diffusion process controlled by `bandwidth_sigma`.

        Parameters
        ----------
        bandwidth_sigma : float
            The bandwidth (standard deviation) of the Gaussian kernel used in
            the diffusion process. This controls the spread of the kernel.
        edge_weight : str, optional
            The edge attribute from the connectivity graph to use as weights
            for constructing the Graph Laplacian. Defaults to "weight".
            The `LayoutEngine` is expected to provide 'weight' and/or 'distance'
            attributes on graph edges.

        Returns
        -------
        kernel_matrix : NDArray[np.float64], shape (n_bins, n_bins)
            The diffusion kernel matrix. For example, element (i, j) can represent
            the influence of bin j on bin i after diffusion. The columns of the
            matrix produced by `compute_diffusion_kernels` are normalized to sum to 1.

        Raises
        ------
        ValueError
            If the connectivity graph is not available or if there are no bins.
        ImportError
            If JAX is required by `compute_diffusion_kernels` but not installed.
        """
        from non_local_detector.diffusion_kernels import compute_diffusion_kernels

        return compute_diffusion_kernels(
            self.connectivity, bandwidth_sigma=bandwidth_sigma, weight=edge_weight
        )

    def get_geodesic_distance(
        self, point1: PtArr, point2: PtArr, edge_weight: str = "distance"
    ) -> float:
        """
        Calculate the geodesic distance between two points in the environment.
        Points are first mapped to their nearest active bins.
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

    def get_shortest_path(
        self,
        source_bin_idx: int,
        target_bin_idx: int,
        edge_weight_key: str = "distance",
    ) -> list[int]:
        """
        Find the shortest path between two active bins.

        The path is a sequence of active bin indices (0 to n_bins - 1)
        connecting the source to the target. Path calculation uses the
        specified `edge_weight_key` attribute on the edges of the
        `connectivity` graph as weights.

        Parameters
        ----------
        source_bin_idx : int
            The active bin index for the start of the path.
        target_bin_idx : int
            The active bin index for the end of the path.
        edge_weight_key : str, optional
            The edge attribute key to use as weight for path calculation,
            by default "distance". If None, treats graph as unweighted.

        Returns
        -------
        list[int]
            A list of active bin indices representing the shortest path.
            Includes source and target. Returns `[source_bin_idx]` if source equals target.
            Returns an empty list if no path exists.

        Raises
        ------
        ValueError
            If the connectivity graph is not available.
        nx.NodeNotFound
            If `source_bin_idx` or `target_bin_idx` is not a node in the graph.
        """
        if not (
            0 <= source_bin_idx < self.n_bins and 0 <= target_bin_idx < self.n_bins
        ):
            # More informative than letting NetworkX raise NodeNotFound for out-of-range indices
            raise nx.NodeNotFound(
                f"Source ({source_bin_idx}) or target ({target_bin_idx}) "
                f"bin index out of range for {self.n_bins} active bins."
            )
        try:
            return nx.shortest_path(
                self.connectivity,
                source=source_bin_idx,
                target=target_bin_idx,
                weight=edge_weight_key,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path exists, return an empty list
            return []

    def map_active_data_to_grid(
        self, active_bin_data: np.ndarray, fill_value: float = np.nan
    ) -> np.ndarray:
        if self.grid_shape is None or self.active_mask is None:
            raise ValueError(
                "This method is applicable only to grid-based environments "
                "that have 'grid_shape' and 'active_mask' attributes."
            )
        if not isinstance(active_bin_data, np.ndarray) or active_bin_data.ndim != 1:
            raise ValueError("active_bin_data must be a 1D NumPy array.")
        if active_bin_data.shape[0] != self.n_bins:
            raise ValueError(
                f"Length of active_bin_data ({active_bin_data.shape[0]}) "
                f"must match the number of active bins ({self.n_bins})."
            )

        # Create an array for the full grid, filled with the fill_value
        # Ensure dtype compatibility, e.g., promote fill_value to active_bin_data.dtype
        # or choose a suitable default like float if active_bin_data can be int.
        dtype = np.result_type(active_bin_data.dtype, type(fill_value))
        full_grid_data = np.full(self.grid_shape, fill_value, dtype=dtype)

        # Place the active data into the grid using the N-D active_mask
        full_grid_data[self.active_mask] = active_bin_data
        return full_grid_data

    def get_bin_attributes_dataframe(self) -> pd.DataFrame:
        """
        Create a Pandas DataFrame with attributes of each active bin.

        The DataFrame is constructed from the node data of the
        `connectivity` graph. Each row corresponds to an active bin.
        Columns include the bin's N-D position (split into `pos_dim0`,
        `pos_dim1`, etc.) and any other attributes stored on the graph nodes.

        Returns
        -------
        pd.DataFrame
            A DataFrame where the index is `active_bin_id` (0 to n_bins - 1)
            and columns contain bin attributes.

        Raises
        ------
        ValueError
            If the connectivity graph is not available or there are no
            active bins (nodes) in the graph.
        RuntimeError
            If essential attributes like n_dims are not available (e.g.,
            if called on an improperly initialized Environment instance).
        """
        graph = self.connectivity

        if graph.number_of_nodes() == 0:
            raise ValueError(
                "The connectivity graph has no active bins (nodes). "
                "Ensure the environment is properly initialized."
            )
        # Convert node data (attributes) to a DataFrame
        # Node IDs in the graph (0 to n_bins-1) become the DataFrame index
        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "active_bin_id"

        # Ensure all 'pos' are consistently tuples/lists before converting
        df["pos"] = df["pos"].apply(lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else (np.nan,) * self.n_dims)  # type: ignore

        pos_df = pd.DataFrame(df["pos"].tolist(), index=df.index)
        pos_df.columns = [f"pos_dim{i}" for i in range(pos_df.shape[1])]
        df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        return df.sort_index()

    def plot(self, ax: Optional[MatplotlibAxes] = None, **kw: Any) -> Any:
        if self._plot is None:
            raise AttributeError("No plot helper was attached by the layout builder.")
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        self._plot(self, ax, **kw)
        return ax
