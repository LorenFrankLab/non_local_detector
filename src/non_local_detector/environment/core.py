"""
environment/core.py
===================

Defines the core `Environment` class, which represents a discretized
N-dimensional space, and the `LayoutResult` data structure that bridges
layout engine outputs to `Environment` instantiation. It also includes
the `LinearAdapter` for 1D environment-specific operations.
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

    from .layout.layout_engine import (
        AreaFn,
        IdxArr,
        LayoutResult,
        LinearProjectionFn,
        LinearToBinFn,
        PlotFunc,
        PointToBin,
        PtArr,
    )
    from .regions.core import Regions  # noqa: F401


class LinearAdapter:
    """
    Provides a 1D-specific view and methods for 1D environments.

    This adapter is typically accessed via `Environment.linear` for environments
    where `is_1d` is True. It offers specialized coordinate transformations
    and plotting relevant to linearized tracks.

    Parameters
    ----------
    env : Environment
        The parent 1D `Environment` instance.
    graph_definition : nx.Graph
        The original graph defining the 1D track structure.
    edge_order : Sequence[tuple[int, int]]
        The ordered sequence of edges that constitute the linearized track.
    edge_spacing : float
        The spacing applied between edges during linearization.

    Attributes
    ----------
    graph_definition : nx.Graph
        The original graph defining the 1D track structure.
    edge_order : tuple[tuple[int, int], ...]
        The ordered sequence of edges that constitute the linearized track.
    edge_spacing : float
        The spacing applied between edges during linearization.
    """

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
        """The original `nx.Graph` used to define the 1D track layout."""
        return self._graph_definition

    @property
    def edge_order(self) -> tuple[tuple[int, int], ...]:
        """The ordered sequence of edge tuples defining the linear bins."""
        return self._edge_order

    @property
    def edge_spacing(self) -> float:
        """The spacing (gap length) applied between edges during linearization."""
        return self._edge_spacing

    def to_linear(self, bin_indices: IdxArr) -> PtArr:
        """
        Convert active 1D bin indices to their corresponding linear positions.

        Parameters
        ----------
        bin_indices : IdxArr
            Array of active 1D bin indices (0 to n_bins - 1).

        Returns
        -------
        PtArr
            Array of 1D linear positions (e.g., in cm) for each bin index.
        """
        return self._env._lin(self._env, np.asarray(bin_indices, dtype=int))

    def to_bin(self, linear_points: PtArr) -> IdxArr:
        """
        Convert 1D linear positions to active 1D bin indices.

        Parameters
        ----------
        linear_points : PtArr
            Array of 1D linear positions (e.g., in cm).

        Returns
        -------
        IdxArr
            Array of active 1D bin indices. Points outside active areas or in
            gaps may map to -1 or be handled by the underlying function.
        """
        return self._env._lin_to_nd(self._env, np.asarray(linear_points, dtype=float))

    def from_points(self, points: PtArr) -> PtArr:
        """
        Map N-D points to 1D linear positions via nearest active bin.

        First, N-D points are mapped to their nearest active bin in the
        environment. Then, these bin indices are converted to their
        1D linear positions.

        Parameters
        ----------
        points : PtArr
            Array of N-dimensional points.

        Returns
        -------
        PtArr
            Array of 1D linear positions.
        """
        return self.to_linear(self._env.bin_at(points))

    def to_linear_cont(self, points: PtArr) -> PtArr:
        """
        Project N-D points to their continuous 1D linear positions.

        This provides a continuous projection onto the linearized track,
        not necessarily tied to bin centers.

        Parameters
        ----------
        points : PtArr
            Array of N-dimensional points.

        Returns
        -------
        PtArr
            Array of continuous 1D linear positions.
        """
        return self._env._nd_points_to_linear_func_(
            self._env, np.asarray(points, dtype=float)
        )

    def plot(self, ax: Optional[MatplotlibAxes] = None, **kw: Any) -> MatplotlibAxes:
        """
        Plot the 1D representation of the environment.

        Delegates to the environment's specific 1D plotting function (`_plot_1d`).

        Parameters
        ----------
        ax : Optional[MatplotlibAxes], default=None
            Matplotlib axes to plot on. If None, new axes are created.
        **kw : Any
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        MatplotlibAxes
            The axes on which the 1D layout was plotted.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        self._env._plot_1d(self._env, ax, **kw)
        return ax


# ---------------------------------------------------------------------
# Environment dataclass
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Environment:
    """
    Represents a discretized N-dimensional space with connectivity.

    This class models an environment (spatial or abstract) that has been
    discretized into a set of 'bins' or 'nodes'. It stores the representative
    'center' (an N-D point/vector) for each active bin, the connectivity
    between these bins, and other metadata like dimension ranges and bin 'areas' (sizes).

    Instances are created via the `Environment.from_layout` classmethod,
    which takes a `LayoutResult` object produced by a `LayoutEngine`.

    Attributes
    ----------
    name : str
        A user-defined name for the environment.
    bin_centers : NDArray[np.float64]
        Coordinates of the center of each *active* bin. Shape (n_bins, n_dims).
    connectivity : nx.Graph
        NetworkX graph where nodes are `0` to `n_bins - 1`, representing
        active bins. Edges indicate adjacency. Node attributes often include
        `'pos'`, `'source_grid_flat_index'`, `'original_grid_nd_index'`.
        Edge attributes often include `'distance'`, `'weight'`.
    dimension_ranges : Tuple[Tuple[float, float], ...]
        The effective min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry.
    is_1d : bool, default=False
        True if the environment represents a primarily 1-dimensional structure.
    regions : Optional[Regions], default=None
        A `Regions` manager object for handling named spatial regions, if any.
    grid_edges : Optional[Tuple[NDArray[np.float64], ...]], default=None
        For grid-based layouts: tuple of 1D arrays of bin edge positions for
        each dimension of the original, full grid.
    grid_shape : Optional[Tuple[int, ...]], default=None
        For grid-based layouts: N-D shape of the original, full grid.
    active_mask : Optional[NDArray[np.bool_]], default=None
        For grid-based layouts: N-D boolean mask of active bins on the
        original, full grid.
    """

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
        """
        Validate the Environment instance after initialization.

        Ensures consistency between `bin_centers`, `connectivity`,
        `dimension_ranges`, and grid attributes if present. Also checks
        that essential injected helper callables are indeed callable.

        Raises
        ------
        ValueError
            If `bin_centers` is not 2D.
            If `connectivity` node count mismatches `bin_centers` rows (`n_bins`).
            If `dimension_ranges` length mismatches `n_dims`.
            If `active_mask` (if present) shape mismatches `grid_shape`.
        TypeError
            If injected helpers `_pt2bin` or `_area` are not callable.
            If `_plot` or `_plot_1d` (if present) are not callable.
        """
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
        """Return a string representation of the Environment."""
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
        Create an `Environment` instance from a `LayoutResult`.

        This is the primary factory method for constructing `Environment` objects.
        It takes the output of a `LayoutEngine`'s build process (`LayoutResult`)
        and initializes a new, immutable `Environment`.

        Parameters
        ----------
        layout_result : LayoutResult
            The fully specified result from a layout engine.
        name : str, optional
            Name for the environment, by default "env".
        regions : Optional[Regions], optional
            A `Regions` manager object to associate with the environment,
            by default None.

        Returns
        -------
        Environment
            An initialized, immutable `Environment` instance.
        """
        return cls(
            name=name,
            bin_centers=layout_result.bin_centers,
            connectivity=layout_result.connectivity,
            dimension_ranges=layout_result.dimension_ranges,
            is_1d=layout_result.is_1d_,
            regions=regions,
            grid_edges=layout_result.grid_edges,
            grid_shape=layout_result.grid_shape,
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
        """
        Access 1D-specific operations via a `LinearAdapter`.

        This adapter is only available if `is_1d` is True and the necessary
        1D layout definition attributes (`_graph_definition`, `_edge_order`,
        `_edge_spacing`) are present.

        Returns
        -------
        LinearAdapter
            An adapter providing 1D-specific views and methods.

        Raises
        ------
        AttributeError
            If the environment is not 1D or lacks the required 1D layout definitions.
        """
        if not self.is_1d:
            raise AttributeError("LinearAdapter is only available for 1D environments.")
        if (
            self._graph_definition is None
            or self._edge_order is None
            or self._edge_spacing is None
            or self._lin is None  # Add these checks
            or self._lin_to_nd is None
            or self._nd_points_to_linear_func_ is None
            or self._plot_1d is None  # If plot is considered essential
        ):
            raise AttributeError(
                "This 1D environment lacks some required linear definition attributes or methods."
            )
        return LinearAdapter(
            self,
            self._graph_definition,
            self._edge_order,
            self._edge_spacing,
        )

    def with_regions(self, regions: "Regions") -> "Environment":
        """
        Return a new `Environment` instance with updated `Regions`.

        Since `Environment` is immutable, this method creates a new instance
        that is a copy of the current one but with the specified `regions`.

        Parameters
        ----------
        regions : Regions
            The new `Regions` object to associate with the environment.

        Returns
        -------
        Environment
            A new `Environment` instance with the updated regions.
        """
        return dataclasses.replace(self, regions=regions)

    @property
    def n_bins(self) -> int:
        """
        Number of active bins in the environment.

        This corresponds to the number of nodes in the `connectivity` graph
        and the number of rows in `bin_centers`.
        """
        return self.bin_centers.shape[0]

    @property
    def n_dims(self) -> int:
        """
        Number of dimensions of the points/vectors representing bin centers.

        This corresponds to the number of columns in `bin_centers`.
        """
        return self.bin_centers.shape[1]

    def bin_at(self, points: PtArr) -> IdxArr:
        """
        Map N-dimensional continuous points to discrete active bin indices.

        Uses the injected `_pt2bin` function from the layout engine.

        Parameters
        ----------
        points : PtArr, shape (n_points, n_dims) or (n_dims,)
            An array of N-dimensional points to map. If a single point (1D array),
            it will be treated as `(1, n_dims)`.

        Returns
        -------
        IdxArr, shape (n_points,)
            An array of active bin indices (0 to `n_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin.
        """
        return self._pt2bin(self, np.atleast_2d(points).astype(float))

    def contains(self, points: PtArr) -> NDArray[np.bool_]:
        """
        Check if N-dimensional continuous points fall within any active bin.

        A point is considered contained if `bin_at(point)` returns a non-negative
        bin index.

        Parameters
        ----------
        points : PtArr, shape (n_points, n_dims) or (n_dims,)
            An array of N-dimensional points to check.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            Boolean array. `True` if the point maps to an active bin.
        """
        return self.bin_at(np.atleast_2d(points).astype(float)) != -1

    def bin_center_of(self, bin_indices: IdxArr) -> PtArr:
        """
        Return the N-D center point(s) of the specified active bin index/indices.

        This directly indexes the `bin_centers` attribute.

        Parameters
        ----------
        bin_indices : IdxArr or int
            Integer index or array of indices for the active bin(s)
            (0 to `n_bins - 1`).

        Returns
        -------
        PtArr, shape (n_requested_bins, n_dims) or (n_dims,)
            The N-D center point(s) corresponding to `bin_indices`.
        """
        return self.bin_centers[np.asarray(bin_indices, dtype=int)]

    def neighbors(self, bin_index: int) -> list[int]:
        """
        Return a list of active bin indices neighboring a given active bin.

        Uses the `connectivity` graph.

        Parameters
        ----------
        bin_index : int
            The active bin index (0 to `n_bins - 1`).

        Returns
        -------
        list[int]
            A list of active bin indices that are neighbors to `bin_index`.
            Returns an empty list if the `bin_index` is not in the graph
            or has no neighbors.

        Raises
        ------
        ValueError
            If the connectivity graph is not available.
        nx.NetworkXError
            If `bin_index` is not a node in the connectivity graph.
        """
        return list(self.connectivity.neighbors(bin_index))

    def bin_area(self) -> NDArray[np.float64]:
        """
        Return the nominal "area" for each active bin.

        Depending on the number of dimensions (`n_dims`) of the environment
        and whether it's a spatial environment, this can represent:
        - Length for 1D environments.
        - Area for 2D environments.
        - Volume for 3D environments.
        - A more abstract "size" or "capacity" for non-spatial
          environments or those with other dimensionalities.

        The values are derived from the layout engine via the injected `_area` function.

        Returns
        -------
        NDArray[np.float64], shape (n_bins,)
            An array containing the area/size of each active bin.
        """
        return self._area(self).astype(float, copy=False)

    def mask_for_region(self, name: str) -> NDArray[np.bool_]:
        """
        Return a boolean mask indicating which active bins are in a named region.

        Parameters
        ----------
        name : str
            The name of the region (must be defined in `self.regions`).

        Returns
        -------
        NDArray[np.bool_], shape (n_bins,)
            A boolean mask. `True` for active bins within the specified region.

        Raises
        ------
        AttributeError
            If `self.regions` is None.
        KeyError
            If `name` is not a defined region.
        RuntimeError
            If Shapely is required for a polygon region but not installed.
        """
        if self.regions is None:
            raise AttributeError("Environment has no Regions attached.")
        from .regions.core import _HAS_SHAPELY as _REGIONS_HAS_SHAPELY

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
        Determine which input points fall into active bins within a specified region.

        Parameters
        ----------
        points : PtArr, shape (n_points, n_dims) or (n_dims,)
            N-dimensional point data.
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            Boolean mask. `True` if the point at that index falls within an
            active bin of the specified region.

        Raises
        ------
        AttributeError
            If `self.regions` is None.
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If `points` array has dimensionality inconsistent with `self.n_dims`.
        """
        if self.regions is None:
            raise AttributeError(f"Environment '{self.name}' has no 'regions' manager.")

        active_bin_indices_for_points = self.bin_at(points)
        region_membership_for_active_bins = self.mask_for_region(region_name)

        output_mask = np.zeros(points.shape[0], dtype=bool)
        # Points that successfully mapped to an active bin
        validly_mapped_points_mask = active_bin_indices_for_points != -1

        if np.any(validly_mapped_points_mask):
            # Get the bin indices for points that mapped successfully
            # These indices are guaranteed to be within [0, n_bins - 1]
            # if _pt2bin adheres to its contract.
            mapped_bin_indices = active_bin_indices_for_points[
                validly_mapped_points_mask
            ]

            # Check if these bins are in the specified region
            output_mask[validly_mapped_points_mask] = region_membership_for_active_bins[
                mapped_bin_indices
            ]

        return output_mask

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
        self, point1: PtArr, point2: PtArr, edge_weight: str = "distance"
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

    def get_shortest_path(
        self,
        source_bin_idx: int,
        target_bin_idx: int,
        edge_weight_key: str = "distance",
    ) -> list[int]:
        """
        Find the shortest path between two active bins using NetworkX.

        The path is a sequence of active bin indices (0 to `n_bins` - 1)
        connecting the source to the target. Path calculation uses the
        specified `edge_weight_key` attribute on the edges of the
        `connectivity` graph as weights.

        Parameters
        ----------
        source_bin_idx : int
            The active bin index (0 to `n_bins` - 1) for the start of the path.
        target_bin_idx : int
            The active bin index (0 to `n_bins` - 1) for the end of the path.
        edge_weight_key : str, optional
            The edge attribute key to use as weight for path calculation,
            by default "distance". If None, treats graph as unweighted.

        Returns
        -------
        list[int]
            A list of active bin indices representing the shortest path.
            Includes source and target. Returns `[source_bin_idx]` if source equals target.
            Returns an empty list if no path exists or if indices are invalid.

        Raises
        ------
        ValueError
            If the connectivity graph is not available.
        nx.NodeNotFound
            If `source_bin_idx` or `target_bin_idx` is valid range but not found
            in graph (should indicate graph inconsistency).
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
        """
        Map a 1D array of data corresponding to active bins onto a full N-D grid.

        This is useful for visualizing data (e.g., place fields, posterior
        probabilities) on grid-based layouts using functions like `pcolormesh`.

        Parameters
        ----------
        active_bin_data : NDArray[np.float64], shape (n_bins,)
            A 1D array of values, one for each active bin, ordered consistently
            with `self.bin_centers` (i.e., by active bin index 0 to `n_bins - 1`).
        fill_value : float, optional
            The value to use for bins in the full grid that are not active,
            by default np.nan.

        Returns
        -------
        NDArray[np.float64], shape (dim0_size, dim1_size, ...)
            An N-D array with shape `self.grid_shape`. Active bin locations are
            filled with `active_bin_data`, others with `fill_value`.

        Raises
        ------
        ValueError
            If the environment is not grid-based (missing `grid_shape` or
            `active_mask`), or if `active_bin_data` has an incorrect shape
            or incompatible data type.
        """
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
        `connectivity` graph. Each row corresponds to an active bin, indexed
        by `active_bin_id` (0 to `n_bins - 1`).
        Columns include the bin's N-D position (as `pos_dim0`, `pos_dim1`, etc.,
        derived from `self.bin_centers`) and any other attributes stored on
        the graph nodes (e.g., 'source_grid_flat_index', 'original_grid_nd_index').

        Returns
        -------
        pd.DataFrame
            DataFrame of bin attributes.

        Raises
        ------
        ValueError
            If the connectivity graph is not available or has no nodes
            (implying no active bins).
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

        pos_df = pd.DataFrame(df["pos"].tolist(), index=df.index)
        pos_df.columns = [f"pos_dim{i}" for i in range(pos_df.shape[1])]
        df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        return df.sort_index()

    def plot(self, ax: Optional[MatplotlibAxes] = None, **kw: Any) -> MatplotlibAxes:
        """
        Plot the environment's layout.

        Delegates to the layout-specific plotting function (`_plot`)
        provided during initialization.

        Parameters
        ----------
        ax : Optional[MatplotlibAxes], default=None
            Matplotlib axes to plot on. If None, new axes are created.
        **kw : Any
            Additional keyword arguments passed to the layout's plotting function.

        Returns
        -------
        MatplotlibAxes
            The axes on which the layout was plotted.

        Raises
        ------
        AttributeError
            If no plot helper (`_plot`) was attached.
        """
        if self._plot is None:
            raise AttributeError("No plot helper was attached by the layout builder.")
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        self._plot(self, ax, **kw)
        return ax
