from __future__ import annotations

import itertools
import math
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
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import Delaunay, KDTree, Voronoi

from non_local_detector.environment.geometry_utils import (
    _create_1d_track_grid_data,
    _create_grid,
    _infer_track_interior,
    _make_nd_track_graph,
    get_centers,
)

try:
    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ---------------------------------------------------------------------------
# GeometryEngine protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class GeometryEngine(Protocol):
    """Defines the interface for all geometry engines.

    A GeometryEngine is responsible for discretizing a spatial environment
    into bins and constructing a graph representation of this discretized space.
    It provides methods for building the geometry, mapping points to bins,
    finding neighbors, and plotting.

    Attributes
    ----------
    place_bin_centers_ : NDArray[np.float64]
        Coordinates of the center of each bin, shape (n_bins, n_dims).
        For grid-based engines, these are typically the centers of the grid cells
        that are considered part of the environment (e.g., interior bins).
        For graph-based engines (like Delaunay or 1D tracks), these are the
        positions of the graph nodes.
    edges_ : Tuple[NDArray[np.float64], ...]
        Bin edges for each dimension. For N-D grids, this is a tuple where
        each element is an array of edge positions for that dimension.
        For 1D environments, it typically contains a single array of
        linearized edge positions. For point-cloud based engines (Delaunay,
        Hexagonal, etc.), this might be an empty tuple if a regular grid
        structure is not applicable.
    centers_shape_ : Tuple[int, ...]
        Shape of the bin grid (number of bins in each dimension).
        For N-D grids, this reflects the grid dimensions (e.g., (n_x_bins, n_y_bins)).
        For 1D environments, it's typically (n_bins,).
        For non-grid engines, it might be (n_points,).
    track_graph_nd_ : Optional[nx.Graph]
        A graph representation suitable for N-dimensional, grid-like environments.
        Nodes usually represent bin centers (or their flat indices), and edges
        connect adjacent/neighboring bins. Node attributes often include 'pos'
        (coordinates), 'is_track_interior', 'bin_ind' (N-D index),
        and 'bin_ind_flat'. Edges often have a 'distance' attribute.
        May be None if not applicable (e.g., for strictly 1D engines where
        `track_graph_bin_centers_` is primary).
    track_graph_bin_centers_ : Optional[nx.Graph]
        A graph where nodes directly correspond to `place_bin_centers_`.
        This is often the primary graph for 1D environments (linear tracks)
        or for engines based on point clouds (e.g., Delaunay, Voronoi)
        where bins are directly associated with specific points.
        May be None if `track_graph_nd_` is the primary representation.
    interior_mask_ : Optional[NDArray[np.bool_]]
        A boolean array indicating which bins/areas are considered part of the
        valid environment.
        For grid-based engines, this usually has the shape `centers_shape_`.
        For other engines, it might be a 1D array corresponding to
        `place_bin_centers_`. If None, all defined bins/points are considered
        interior.
    """

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: nx.Graph | None  # For N-D grid-like structures
    track_graph_bin_centers_: nx.Graph | None  # For 1-D or other graph structures

    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(self, **kwargs) -> None:
        """Constructs the geometry, binning, and graph(s) for the environment.

        This method is called to initialize or re-initialize the engine's
        attributes (e.g., `place_bin_centers_`, `track_graph_nd_`) based on
        the provided parameters.

        Parameters
        ----------
        **kwargs
            Engine-specific parameters required for building the geometry.
            See the `build` method of individual engine implementations for
            details on their specific parameters.
        """
        ...

    def point_to_bin(self, pts: NDArray[np.float64]) -> NDArray[np.int_]:
        """Maps continuous spatial points to discrete bin indices.

        Parameters
        ----------
        pts : NDArray[np.float64], shape (n_samples, n_dims)
            The continuous spatial positions to be binned.

        Returns
        -------
        NDArray[np.int_]
            The flat indices of the bins corresponding to each input point.
            Shape is (n_samples,). Returns -1 for points that cannot be
            mapped to a bin (e.g., outside the defined area or if no
            bins exist).
        """
        ...

    def neighbors(self, flat_idx: int) -> List[int]:
        """Finds the flat indices of neighboring bins for a given bin.

        Neighbor definitions depend on the specific engine and its graph
        representation.

        Parameters
        ----------
        flat_idx : int
            The flat index of the bin for which to find neighbors.

        Returns
        -------
        List[int]
            A list of flat indices of the neighboring bins. Returns an
            empty list if the bin has no neighbors or is not part of the graph.
        """
        ...

    @property
    @abstractmethod
    def is_1d(self) -> bool:
        """Return True if the engine represents a 1-dimensional environment.

        Returns
        -------
        bool
            True if the environment is 1D, False otherwise.
        """
        ...

    @abstractmethod
    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the geometry of the environment.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are
            created. Defaults to None.
        **kwargs
            Additional keyword arguments to customize the plot. Common arguments
            might include `node_size`, `edge_color` for graph plots,
            or `cmap` for grid plots. See specific engine `plot` methods
            for their supported kwargs.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.
        """
        ...


# ---------------------------------------------------------------------------
# KD-tree mixin implementing generic mapping helpers
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """Mixin providing `point_to_bin` and `neighbors` via KD-tree and NetworkX.

    This mixin assumes that the class it's mixed into will define
    `place_bin_centers_` (or `spatial_place_bin_centers_` for some engines),
    and one or both of `track_graph_nd_` or `track_graph_bin_centers_`
    after its `build` method is called.

    Attributes
    ----------
    _kdtree : Optional[scipy.spatial.KDTree]
        The KD-tree built from the (potentially masked) `place_bin_centers_`.
    _flat_indices_of_kdtree_nodes : Optional[NDArray[np.int_]]
        Array mapping indices from the KD-tree back to the original flat indices
        of `place_bin_centers_`.
    """

    _kdtree: Optional[KDTree] = None
    _flat_indices_of_kdtree_nodes: Optional[NDArray[np.int_]] = None

    def _build_kdtree(self, interior_mask: Optional[NDArray[np.bool_]] = None) -> None:
        """Builds the KD-tree from `place_bin_centers_`.

        If `interior_mask` is provided, the KD-tree is built only from the
        points/bin centers marked as True in the mask.

        Parameters
        ----------
        interior_mask : Optional[NDArray[np.bool_]], optional
            A boolean mask indicating which points from `self.place_bin_centers_`
            to include in the KD-tree.
            If 1D, its length must match `self.place_bin_centers_.shape[0]`.
            If N-D (for grid-based engines), its shape must match
            `self.centers_shape_`, and `self.place_bin_centers_` is assumed
            to be the flattened full grid.
            If None, all points in `self.place_bin_centers_` are used.
            Defaults to None.

        Raises
        ------
        ValueError
            If `place_bin_centers_` is not set, or if `interior_mask` has
            an incompatible shape.
        """
        if not hasattr(self, "place_bin_centers_") or self.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ not set; call build() before _build_kdtree()"
            )

        current_place_bin_centers = self.place_bin_centers_  # Points to build KDTree on
        num_points = current_place_bin_centers.shape[0]

        if interior_mask is not None:
            # Scenario 1: interior_mask is N-D and applies to a full grid
            # from which current_place_bin_centers may or may not be derived.
            if (
                interior_mask.ndim > 1
                and hasattr(self, "centers_shape_")
                and interior_mask.shape == self.centers_shape_
            ):
                if num_points != np.prod(self.centers_shape_):
                    # This is complex: place_bin_centers_ might be a subset already,
                    # and an N-D mask is given. This scenario needs clear definition.
                    # Assuming for now place_bin_centers_ IS the full grid if N-D mask is given.
                    warnings.warn(
                        "N-D interior_mask provided, but place_bin_centers_ does not seem to be the full grid. "
                        "Ensure consistency.",
                        UserWarning,
                    )
                mask_to_apply_flat = interior_mask.ravel()
                if len(mask_to_apply_flat) != num_points:
                    raise ValueError(
                        f"Raveled N-D interior_mask length {len(mask_to_apply_flat)} "
                        f"does not match place_bin_centers_ length {num_points}."
                    )

            # Scenario 2: interior_mask is 1D
            elif interior_mask.ndim == 1:
                if interior_mask.shape[0] != num_points:
                    raise ValueError(
                        f"1D interior_mask length {interior_mask.shape[0]} "
                        f"does not match place_bin_centers_ length {num_points}."
                    )
                mask_to_apply_flat = interior_mask
            else:  # Mask shape is unusable
                raise ValueError(
                    f"interior_mask shape {interior_mask.shape} is not compatible. "
                    "Must be N-D matching centers_shape_ or 1D matching place_bin_centers_."
                )

            points_for_kdtree = current_place_bin_centers[mask_to_apply_flat]
            # Indices here are into the original current_place_bin_centers
            self._flat_indices_of_kdtree_nodes = np.where(mask_to_apply_flat)[0].astype(
                np.int32
            )
        else:  # No mask
            points_for_kdtree = current_place_bin_centers
            self._flat_indices_of_kdtree_nodes = np.arange(num_points, dtype=np.int32)

        if points_for_kdtree.shape[0] > 0:
            self._kdtree = KDTree(points_for_kdtree)
        else:
            self._kdtree = None
            self._flat_indices_of_kdtree_nodes = np.array([], dtype=np.int32)

    def point_to_bin(self, pts: NDArray[np.float64]) -> NDArray[np.int_]:
        """Maps continuous spatial points to the nearest bin's flat index.

        Uses the KD-tree built from (interior) bin centers.

        Parameters
        ----------
        pts : NDArray[np.float64], shape (n_samples, n_dims)
            The spatial positions to be binned.

        Returns
        -------
        NDArray[np.int_]
            The flat indices of the bins corresponding to each input point.
            Shape is (n_samples,). Returns -1 if the KD-tree is empty or
            if mapping fails.
        """
        if self._kdtree is None:
            return np.full(np.atleast_2d(pts).shape[0], -1, dtype=np.int32)

        if self._flat_indices_of_kdtree_nodes is None:
            # This should not happen if _kdtree is not None and build was successful
            raise RuntimeError(
                "_flat_indices_of_kdtree_nodes not set. Ensure _build_kdtree was called."
            )

        _, indices_in_kdtree_subset = self._kdtree.query(np.atleast_2d(pts))

        if (
            indices_in_kdtree_subset.size == 0
            or self._flat_indices_of_kdtree_nodes.size == 0
        ):  # No points in KDTree query result or kdtree was empty
            return np.full(np.atleast_2d(pts).shape[0], -1, dtype=np.int32)

        # Ensure indices are valid for _flat_indices_of_kdtree_nodes
        valid_indices = np.clip(
            indices_in_kdtree_subset, 0, self._flat_indices_of_kdtree_nodes.shape[0] - 1
        )

        original_flat_indices = self._flat_indices_of_kdtree_nodes[valid_indices]
        return original_flat_indices.astype(np.int32)

    def neighbors(self, flat_idx: int) -> List[int]:
        """Finds neighbors of a bin using the engine's track graph.

        Parameters
        ----------
        flat_idx : int
            The flat index of the bin.

        Returns
        -------
        List[int]
            List of flat indices of neighboring bins. Returns an empty list
            if the bin is not in the graph or has no neighbors.

        Raises
        ------
        ValueError
            If no suitable graph attribute (`track_graph_nd_` or
            `track_graph_bin_centers_`) is found or set in the engine.
        """
        graph_to_use: Optional[nx.Graph] = None
        if hasattr(self, "track_graph_nd_") and self.track_graph_nd_ is not None:
            graph_to_use = self.track_graph_nd_
        elif (
            hasattr(self, "track_graph_bin_centers_")
            and self.track_graph_bin_centers_ is not None
        ):
            graph_to_use = self.track_graph_bin_centers_

        if graph_to_use is None:
            if hasattr(self, "track_graph_nd_") and hasattr(
                self, "track_graph_bin_centers_"
            ):
                raise ValueError(
                    "Both track_graph_nd_ and track_graph_bin_centers_ are None."
                )
            raise ValueError(
                "No suitable graph attribute (track_graph_nd_ or "
                "track_graph_bin_centers_) found or set."
            )

        if flat_idx not in graph_to_use:
            # If flat_idx refers to an original index of place_bin_centers_
            # but this bin was filtered out by interior_mask, it won't be in the graph.
            return []
        return list(graph_to_use.neighbors(flat_idx))


class RegularGridEngine(_KDTreeMixin):
    """Engine for discretizing space into an N-dimensional regular grid.

    This engine creates an axis-aligned rectangular grid. It can infer the
    "track interior" (occupied bins) from position data or define all bins
    within the specified range as interior. A graph connects adjacent
    interior bins.
    """

    place_bin_centers_: NDArray[np.float64]
    place_bin_edges_: Optional[NDArray[np.float64]] = (
        None  # Typically not the primary edge representation
    )
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        place_bin_size: Union[float, Sequence[float]],
        position_range: Optional[Sequence[Tuple[float, float]]] = None,
        position: Optional[NDArray[np.float64]] = None,
        add_boundary_bins: bool = False,
        infer_track_interior: bool = False,
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        bin_count_threshold: int = 0,
    ) -> None:
        """Builds the regular grid geometry.

        Parameters
        ----------
        place_bin_size : Union[float, Sequence[float]]
            The desired size of the bins. If a float, used for all dimensions.
            If a sequence, specifies size for each dimension.
        position_range : Optional[Sequence[Tuple[float, float]]], optional
            Explicit grid boundaries `[(min_dim1, max_dim1), ...]`. If None,
            boundaries are derived from `position` data. Defaults to None.
        position : Optional[NDArray[np.float64]], shape (n_time, n_dims), optional
            Position data. Required if `position_range` is None or if
            `infer_track_interior` is True. NaNs are ignored. Defaults to None.
        add_boundary_bins : bool, default=False
            If True, adds one bin on each side of the grid in each dimension,
            extending the range slightly.
        infer_track_interior : bool, default=True
            If True, infers the occupied track area from `position` data.
            If False, all bins within the defined grid are considered interior.
        dilate : bool, default=False
            If `infer_track_interior` is True, setting this to True will
            expand the boundary of the inferred occupied area.
        fill_holes : bool, default=False
            If `infer_track_interior` is True, setting this to True will
            fill holes within the inferred occupied area.
        close_gaps : bool, default=False
            If `infer_track_interior` is True, setting this to True will
            close small gaps in the inferred occupied area.
        bin_count_threshold : int, default=0
            If `infer_track_interior` is True, this is the minimum number of
            position samples a bin must contain to be considered part of the
            track interior.

        Raises
        ------
        ValueError
            If `position` data is required but not provided for interior inference
            or range determination.
        """
        (
            self.edges_,
            self.place_bin_edges_,
            self.place_bin_centers_,
            self.centers_shape_,
        ) = _create_grid(
            position=position,
            bin_size=place_bin_size,
            position_range=position_range,
            add_boundary_bins=add_boundary_bins,
        )

        if infer_track_interior:
            if position is None:
                raise ValueError(
                    "`position` data must be provided when `infer_track_interior` is True."
                )
            self.interior_mask_ = _infer_track_interior(
                position=position,
                edges=self.edges_,
                close_gaps=close_gaps,
                fill_holes=fill_holes,
                dilate=dilate,
                bin_count_threshold=bin_count_threshold,
                boundary_exists=add_boundary_bins,
            )
        else:
            self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        self.track_graph_nd_ = _make_nd_track_graph(
            self.place_bin_centers_, self.interior_mask_, self.centers_shape_
        )
        self._build_kdtree(interior_mask=self.interior_mask_)

    @property
    def is_1d(self) -> bool:
        """Returns False, as RegularGridEngine is for N-D grids."""
        return False  # RegularGridEngine is for N-D grids

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the 2D grid and track interior, or a graph for N-D.

        For 2D grids, this method uses `pcolormesh` to show the `interior_mask_`.
        For grids of other dimensions, it attempts to plot `track_graph_nd_`
        or a scatter plot of `place_bin_centers_`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
            Defaults to None.
        **kwargs
            Additional keyword arguments:
            figsize : tuple, optional
                Figure size `(width, height)` if `ax` is None. Default (7,7) for 2D.
            cmap : str, optional
                Colormap for `pcolormesh` (2D plot). Default 'bone_r'.
            alpha : float, optional
                Alpha transparency for `pcolormesh` (2D plot). Default 0.7.
            node_size : int, optional
                Node size for N-D graph fallback plot. Default 10.
            scatter_kwargs : dict, optional
                Keyword arguments for scatter plot (N-D fallback if no graph).

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.

        Raises
        ------
        RuntimeError
            If the engine has not been built yet.
        """
        if self.place_bin_centers_ is None or self.edges_ is None:
            raise RuntimeError("Engine has not been built yet. Call `build` first.")

        if len(self.centers_shape_) != 2:
            # Fallback for non-2D grids: plot graph or scatter
            if ax is None:
                fig = plt.figure()
                is_3d_data = self.place_bin_centers_.shape[1] == 3
                ax = fig.add_subplot(111, projection="3d" if is_3d_data else None)  # type: ignore

            if self.track_graph_nd_ is not None:
                pos = {
                    i: self.place_bin_centers_[i] for i in self.track_graph_nd_.nodes()
                }
                nx.draw(
                    self.track_graph_nd_,
                    pos,
                    ax=ax,
                    node_size=kwargs.get("node_size", 10),
                )
            elif self.place_bin_centers_.shape[0] > 0:  # Scatter plot if no graph
                if self.place_bin_centers_.shape[1] == 1:
                    ax.scatter(
                        self.place_bin_centers_[:, 0],
                        np.zeros_like(self.place_bin_centers_[:, 0]),
                        **kwargs,
                    )
                elif self.place_bin_centers_.shape[1] == 2:
                    ax.scatter(
                        self.place_bin_centers_[:, 0],
                        self.place_bin_centers_[:, 1],
                        **kwargs,
                    )
                elif self.place_bin_centers_.shape[1] == 3 and hasattr(ax, "scatter3D"):
                    ax.scatter3D(self.place_bin_centers_[:, 0], self.place_bin_centers_[:, 1], self.place_bin_centers_[:, 2], **kwargs)  # type: ignore
            ax.set_title(f"{self.__class__.__name__} (N-D fallback)")
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))

        if self.interior_mask_ is not None:
            ax.pcolormesh(
                self.edges_[0],
                self.edges_[1],
                self.interior_mask_.T,  # Transpose for pcolormesh convention
                cmap=kwargs.get("cmap", "bone_r"),
                alpha=kwargs.get("alpha", 0.7),
                shading="auto",
            )

        ax.set_xticks(self.edges_[0])
        ax.set_yticks(self.edges_[1])
        ax.grid(True, ls="-", lw=0.5, c="gray")
        ax.set_aspect("equal")
        ax.set_title(f"{self.__class__.__name__} (2D Grid)")
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        if self.edges_ and len(self.edges_) == 2:
            ax.set_xlim(self.edges_[0][0], self.edges_[0][-1])
            ax.set_ylim(self.edges_[1][0], self.edges_[1][-1])
        return ax


class TrackGraphEngine(_KDTreeMixin):
    """Engine for 1-D environments defined by a topological track graph.

    This engine linearizes a provided track graph and creates bins along this
    linearized representation. The `place_bin_centers_` attribute stores
    linearized positions, while `spatial_place_bin_centers_` (internal) holds
    the original N-D coordinates used for KD-tree mapping.
    """

    place_bin_centers_: NDArray[np.float64]  # Linearized positions (n_bins, 1)
    place_bin_edges_: Optional[NDArray[np.float64]] = None  # Linearized edges
    edges_: Tuple[
        NDArray[np.float64], ...
    ]  # Tuple containing one array: (linearized_edges,)
    centers_shape_: Tuple[int, ...]  # (n_bins,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not used by this engine
    track_graph_bin_centers_: Optional[nx.Graph] = (
        None  # Graph with original N-D 'pos' attributes for bins
    )
    interior_mask_: Optional[NDArray[np.bool_]] = None  # 1D mask for linearized bins

    # Store parameters for potential re-linearization or inspection
    track_graph_definition_: Optional[nx.Graph] = None
    edge_order_definition_: Optional[List[Tuple[Any, Any]]] = None
    edge_spacing_definition_: Optional[Union[float, Sequence[float]]] = None
    spatial_place_bin_centers_: Optional[NDArray[np.float64]] = (
        None  # N-D coordinates of bins
    )

    def build(
        self,
        *,
        track_graph: nx.Graph,
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        place_bin_size: float,
        position: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Builds the 1-D track geometry.

        Parameters
        ----------
        track_graph : nx.Graph
            The topological graph of the track. Nodes must have a 'pos'
            attribute with N-D coordinates.
        edge_order : List[Tuple[Any, Any]]
            An ordered list of node pairs (edges from `track_graph`)
            defining the sequence for linearization.
        edge_spacing : Union[float, Sequence[float]]
            Spacing(s) to insert between segments during linearization.
            If a float, applied uniformly. If a sequence, applied between
            consecutive edges in `edge_order`.
        place_bin_size : float
            The desired size of bins along the linearized track.
        position : Optional[NDArray[np.float64]], optional
            Position data for the track. Ignored in this engine but
            included for consistency with other engines. Defaults to None.

        Raises
        ------
        ValueError
            If `track_graph` nodes are missing the 'pos' attribute.
        RuntimeError
            If `track_graph_bin_centers_` is not created by helper functions.
        """
        if position is not None:
            warnings.warn(
                "Position data is not used in TrackGraphEngine. It is included for consistency with other engines.",
                UserWarning,
            )

        self.track_graph_definition_ = track_graph
        self.edge_order_definition_ = edge_order
        self.edge_spacing_definition_ = edge_spacing

        (
            self.place_bin_centers_,  # (n_bins, 1) with linearized positions
            self.place_bin_edges_,  # (n_edges, 1) with linearized positions
            is_interior,  # 1D boolean array for interior bins
            self.centers_shape_,  # (n_bins,)
            self.edges_,  # Tuple: (linearized_1d_edges_array,)
            self.track_graph_bin_centers_,  # Graph with N-D 'pos' attributes for bins
        ) = _create_1d_track_grid_data(
            track_graph,
            edge_order,
            edge_spacing,
            place_bin_size,  # This is the linear bin size
        )

        if self.track_graph_bin_centers_ is None:
            raise RuntimeError(
                "track_graph_bin_centers_ was not created by _create_1d_track_grid_data."
            )

        # Extract N-D spatial positions from the generated bin graph for KDTree
        nodes_df = pd.DataFrame.from_dict(
            dict(self.track_graph_bin_centers_.nodes(data=True)), orient="index"
        )
        if "pos" not in nodes_df.columns:
            raise ValueError(
                "TrackGraphEngine's track_graph_bin_centers_ nodes must have 'pos' attribute."
            )
        if "bin_ind_flat" not in nodes_df.columns:
            raise ValueError(
                "TrackGraphEngine's track_graph_bin_centers_ nodes must have 'bin_ind_flat' attribute."
            )

        nodes_df = nodes_df.sort_values(by="bin_ind_flat")
        self.spatial_place_bin_centers_ = np.array(nodes_df["pos"].tolist())

        # The interior_mask_ should correspond to these spatial bins
        # is_interior from _create_1d_track_grid_data is already aligned with linearized bins
        # which should map 1:1 with the nodes in track_graph_bin_centers_
        self.interior_mask_ = (
            np.array(nodes_df["is_track_interior"].tolist(), dtype=bool)
            if "is_track_interior" in nodes_df  # is_track_interior should be on nodes
            else is_interior  # Fallback, ensure is_interior matches node count
        )
        if len(self.interior_mask_) != self.spatial_place_bin_centers_.shape[0]:
            # This indicates a mismatch that needs debugging in _create_1d_track_grid_data
            # or how attributes are set on track_graph_bin_centers_
            raise RuntimeError(
                f"Length of interior_mask ({len(self.interior_mask_)}) does not match "
                f"number of spatial_place_bin_centers ({self.spatial_place_bin_centers_.shape[0]})."
            )

        # For _KDTreeMixin, `place_bin_centers_` needs to be the spatial coordinates.
        # We will store the linearized ones in self.place_bin_centers_ (as per class attr type)
        # but use spatial for KDTree construction.
        original_place_bin_centers_for_kdtree = self.place_bin_centers_
        self.place_bin_centers_ = self.spatial_place_bin_centers_
        self._build_kdtree(interior_mask=self.interior_mask_)
        self.place_bin_centers_ = (
            original_place_bin_centers_for_kdtree  # Restore linearized
        )

    @property
    def is_1d(self) -> bool:
        """Returns True, as TrackGraphEngine is for 1-D environments."""
        return True

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the 1D track graph using its N-D spatial node positions.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
            Defaults to None.
        **kwargs
            Additional keyword arguments for `nx.draw_networkx_nodes` and
            `nx.draw_networkx_edges`:
            figsize : tuple, optional
                Figure size `(width, height)` if `ax` is None. Default (8,8).
            node_size : int, optional
                Size of nodes in the plot. Default 20.
            node_color : str or sequence, optional
                Color of nodes. If None, interior nodes are 'blue', others 'red'.
            edge_alpha : float, optional
                Alpha transparency for edges. Default 0.5.
            edge_color : str, optional
                Color of edges. Default 'gray'.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.

        Raises
        ------
        RuntimeError
            If the engine has not been built or lacks spatial bin centers.
        ValueError
            If a 3D plot is requested but `ax` is not a 3D axes.
        """
        if (
            self.track_graph_bin_centers_ is None
            or self.spatial_place_bin_centers_ is None
        ):
            raise RuntimeError(
                "Engine has not been built or spatial_place_bin_centers_ is missing. Call `build` first."
            )

        graph_to_plot = self.track_graph_bin_centers_
        node_positions = {
            node: data["pos"]
            for node, data in graph_to_plot.nodes(data=True)
            if "pos" in data
        }

        if not node_positions:  # Should not happen if build was successful
            if ax is None:
                _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))
            ax.text(
                0.5,
                0.5,
                "No positional data in graph nodes.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{self.__class__.__name__}")
            return ax

        # Determine if plot is 2D or 3D
        first_pos_val = next(iter(node_positions.values()))
        is_3d_plot = len(first_pos_val) == 3

        if ax is None:
            fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            ax = fig.add_subplot(111, projection="3d" if is_3d_plot else None)  # type: ignore
        elif is_3d_plot and not hasattr(ax, "plot3D"):  # ax provided but not 3D
            raise ValueError(
                "Provided 'ax' is not 3D, but data is 3D. Create a 3D subplot."
            )

        # Default node colors based on interior_mask_
        node_colors = kwargs.get("node_color", None)
        if node_colors is None and self.interior_mask_ is not None:
            # Assuming node IDs in graph are 0 to num_nodes-1 (bin_ind_flat)
            # and align with interior_mask_ indices.
            node_list = sorted(list(graph_to_plot.nodes()))
            try:
                # Map node IDs (bin_ind_flat) to interior_mask_
                # This assumes that nodes in track_graph_bin_centers_ are indexed
                # in a way that corresponds to self.interior_mask_
                # self.interior_mask_ is aligned with self.spatial_place_bin_centers_
                # and nodes in track_graph_bin_centers_ are derived from these.
                color_map_list = []
                for node_id in node_list:
                    # We need to find which index in self.interior_mask corresponds to node_id
                    # Typically, node_id from track_graph_bin_centers is 'bin_ind_flat'
                    # And self.interior_mask is ordered by 'bin_ind_flat'
                    bin_flat_index = graph_to_plot.nodes[node_id].get("bin_ind_flat")
                    if bin_flat_index is not None and 0 <= bin_flat_index < len(
                        self.interior_mask_
                    ):
                        color_map_list.append(
                            "blue" if self.interior_mask_[bin_flat_index] else "red"
                        )
                    else:
                        color_map_list.append("gray")  # Fallback for nodes not in mask
                node_colors = color_map_list

            except (
                IndexError,
                KeyError,
            ):  # If node IDs don't align or attributes missing
                warnings.warn(
                    "Could not map all nodes to interior_mask for coloring. Defaulting to blue.",
                    UserWarning,
                )
                node_colors = "blue"
        elif node_colors is None:
            node_colors = "blue"

        nx.draw_networkx_nodes(
            graph_to_plot,
            pos=node_positions,
            ax=ax,  # type: ignore
            node_size=kwargs.get("node_size", 20),
            node_color=node_colors,
        )
        nx.draw_networkx_edges(
            graph_to_plot,
            pos=node_positions,
            ax=ax,  # type: ignore
            alpha=kwargs.get("edge_alpha", 0.5),
            edge_color=kwargs.get("edge_color", "gray"),
        )

        ax.set_title(f"{self.__class__.__name__}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        if is_3d_plot and hasattr(ax, "set_zlabel"):
            ax.set_zlabel("Z coordinate")  # type: ignore

        if not is_3d_plot:
            ax.set_aspect("equal", adjustable="box")
        elif hasattr(ax, "set_box_aspect") and hasattr(ax, "get_xlim"):  # For 3D axes
            # Attempt to make aspect somewhat equal for 3D
            x_lim, y_lim, z_lim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()  # type: ignore
            ax.set_box_aspect([np.ptp(x_lim), np.ptp(y_lim), np.ptp(z_lim)])  # type: ignore

        return ax  # type: ignore


if SHAPELY_AVAILABLE:

    class ShapelyGridEngine(_KDTreeMixin):
        """Engine that creates a 2D grid and masks it using a Shapely Polygon.

        This engine first defines a regular rectangular grid that encompasses the
        bounds of the provided Shapely Polygon. Then, it determines which grid
        bin centers fall within the polygon, marking these as the interior of
        the environment. A graph connects adjacent interior bins.
        """

        place_bin_centers_: NDArray[np.float64]  # Bin centers within the polygon
        edges_: Tuple[NDArray[np.float64], ...]  # Edges of the encompassing grid
        centers_shape_: Tuple[int, ...]  # Shape of the encompassing grid
        track_graph_nd_: Optional[nx.Graph] = None  # Graph of interior bins
        track_graph_bin_centers_: Optional[nx.Graph] = (
            None  # Typically not the primary for this engine
        )
        interior_mask_: Optional[NDArray[np.bool_]] = (
            None  # Boolean mask matching centers_shape_
        )
        polygon_definition_: Optional[Polygon] = None  # The input Shapely polygon

        def build(
            self,
            *,
            polygon: Polygon,
            place_bin_size: Union[float, Sequence[float]],
            add_boundary_bins: bool = False,
        ) -> None:
            """Builds the grid masked by the Shapely polygon.

            Parameters
            ----------
            polygon : shapely.geometry.Polygon
                The Shapely Polygon object used to define the environment's shape.
            place_bin_size : Union[float, Sequence[float]]
                The size of the bins for the underlying rectangular grid.
                If a float, used for both x and y dimensions.
                If a sequence (length 2), specifies size for x and y dimensions.
            add_boundary_bins : bool, default=False
                If True, adds one bin on each side of the grid (defined by
                polygon bounds) before masking.

            Raises
            ------
            RuntimeError
                If Shapely is not installed.
            ValueError
                If `place_bin_size` is incompatible with 2D.
            """
            if not SHAPELY_AVAILABLE:
                raise RuntimeError("ShapelyGridEngine requires the 'shapely' package.")
            self.polygon_definition_ = polygon
            minx, miny, maxx, maxy = polygon.bounds
            pos_range: Sequence[Tuple[float, float]] = [(minx, maxx), (miny, maxy)]

            (
                self.edges_,
                self.place_bin_edges_,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = _create_grid(
                position=None,
                bin_size=place_bin_size,
                position_range=pos_range,
                add_boundary_bins=add_boundary_bins,
            )

            # Ensure place_bin_centers_ has 2 dimensions for polygon.contains
            if (
                self.place_bin_centers_.ndim == 1
            ):  # Should not happen if _create_grid is for 2D
                points_to_check = self.place_bin_centers_[
                    :, np.newaxis
                ]  # Make it (N,1)
            elif self.place_bin_centers_.shape[1] == 1:  # Also (N,1)
                points_to_check = np.hstack(
                    [self.place_bin_centers_, np.zeros_like(self.place_bin_centers_)]
                )  # Dummy Y
            elif self.place_bin_centers_.shape[1] >= 2:
                points_to_check = self.place_bin_centers_[
                    :, :2
                ]  # Use first two dims for polygon check
            else:  # empty
                points_to_check = np.empty((0, 2))

            mask_flat = np.array([polygon.contains(Point(*p)) for p in points_to_check])
            self.interior_mask_ = mask_flat.reshape(self.centers_shape_)

            self.track_graph_nd_ = _make_nd_track_graph(
                self.place_bin_centers_, self.interior_mask_, self.centers_shape_
            )
            self._build_kdtree(interior_mask=self.interior_mask_)

        @property
        def is_1d(self) -> bool:
            """Returns False, as ShapelyGridEngine is 2D."""
            return False

        def plot(
            self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
        ) -> matplotlib.axes.Axes:
            """Plots the 2D grid, Shapely polygon, and interior mask.

            Parameters
            ----------
            ax : Optional[matplotlib.axes.Axes], optional
                Existing Matplotlib axes to plot on. If None, new axes are created.
            **kwargs
                Additional keyword arguments:
                figsize : tuple, optional
                    Figure size `(width, height)` if `ax` is None. Default (7,7).
                polygon_kwargs : dict, optional
                    Keyword arguments for plotting the Shapely polygon
                    (passed to `ax.fill`). Default `{'alpha': 0.3, 'fc': 'gray', 'ec': 'black'}`.
                grid_cmap : str, optional
                    Colormap for `pcolormesh` of the interior mask. Default 'viridis'.
                grid_alpha : float, optional
                    Alpha transparency for `pcolormesh` of the interior mask. Default 0.5.
                show_grid_lines : bool, optional
                    If True, draw major grid lines. Default True.
                show_bin_centers : bool, optional
                    If True, scatter plot the centers of interior bins. Default False.
                center_size : float, optional
                    Size of markers for bin centers if `show_bin_centers` is True. Default 5.
                center_color : str, optional
                    Color of markers for bin centers. Default 'red'.

            Returns
            -------
            matplotlib.axes.Axes
                The Matplotlib axes object used for plotting.

            Raises
            ------
            RuntimeError
                If Shapely is not available or if the engine has not been built.
            """
            if not SHAPELY_AVAILABLE:
                raise RuntimeError("Shapely is required for plotting this engine.")
            if (
                self.place_bin_centers_ is None
                or self.edges_ is None
                or self.polygon_definition_ is None
            ):
                raise RuntimeError("Engine has not been built yet. Call `build` first.")

            if len(self.centers_shape_) != 2:
                # Fallback to default graph or scatter plot for N-D data
                # This is similar to RegularGridEngine's fallback
                if ax is None:
                    fig = plt.figure()
                    is_3d_data = self.place_bin_centers_.shape[1] == 3
                    ax = fig.add_subplot(111, projection="3d" if is_3d_data else None)  # type: ignore
                if self.track_graph_nd_ is not None:
                    pos = {
                        i: self.place_bin_centers_[i]
                        for i in self.track_graph_nd_.nodes()
                    }
                    nx.draw(
                        self.track_graph_nd_,
                        pos,
                        ax=ax,
                        node_size=kwargs.get("node_size", 10),
                    )
                elif self.place_bin_centers_.shape[0] > 0:  # Scatter plot
                    if self.place_bin_centers_.shape[1] == 2:
                        ax.scatter(
                            self.place_bin_centers_[:, 0],
                            self.place_bin_centers_[:, 1],
                            **kwargs,
                        )
                    # Add other dim scatters if needed
                ax.set_title(f"{self.__class__.__name__} (N-D fallback)")
                return ax

            if ax is None:
                _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))

            # Plot the polygon
            poly_patch_kwargs = kwargs.get(
                "polygon_kwargs", {"alpha": 0.3, "fc": "gray", "ec": "black"}
            )
            if hasattr(self.polygon_definition_, "exterior"):  # Is a Polygon
                x, y = self.polygon_definition_.exterior.xy
                ax.fill(x, y, **poly_patch_kwargs)
            elif hasattr(self.polygon_definition_, "geoms"):  # Is a MultiPolygon
                for geom in self.polygon_definition_.geoms:
                    x, y = geom.exterior.xy
                    ax.fill(x, y, **poly_patch_kwargs)

            # Plot grid and interior mask (pcolormesh)
            if self.interior_mask_ is not None and len(self.edges_) == 2:
                ax.pcolormesh(
                    self.edges_[0],
                    self.edges_[1],
                    self.interior_mask_.T,
                    cmap=kwargs.get(
                        "cmap", "viridis"
                    ),  # Different cmap to distinguish from polygon
                    alpha=kwargs.get("alpha", 0.5),
                    shading="auto",
                )
                ax.set_xticks(self.edges_[0])
                ax.set_yticks(self.edges_[1])
                ax.grid(True, ls=":", lw=0.5, c="black")

            # Plot bin centers if desired
            if (
                kwargs.get("show_bin_centers", False)
                and self.place_bin_centers_.shape[1] >= 2
            ):
                mask_for_centers = (
                    self.interior_mask_.ravel()
                    if self.interior_mask_ is not None
                    else np.ones(self.place_bin_centers_.shape[0], dtype=bool)
                )
                ax.scatter(
                    self.place_bin_centers_[mask_for_centers, 0],
                    self.place_bin_centers_[mask_for_centers, 1],
                    s=kwargs.get("center_size", 5),
                    c=kwargs.get("center_color", "red"),
                    alpha=0.7,
                )

            ax.set_aspect("equal")
            ax.set_title(f"{self.__class__.__name__}")
            ax.set_xlabel("Dimension 0")
            ax.set_ylabel("Dimension 1")
            if self.edges_ and len(self.edges_) == 2:
                ax.set_xlim(self.edges_[0][0], self.edges_[0][-1])
                ax.set_ylim(self.edges_[1][0], self.edges_[1][-1])
            else:  # Fallback from polygon bounds
                minx, miny, maxx, maxy = self.polygon_definition_.bounds
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)

            return ax

else:
    ShapelyGridEngine = None  # type: ignore


class MaskedGridEngine(_KDTreeMixin):
    """Engine that builds geometry from a pre-defined N-D boolean mask and grid edges.

    This engine is useful when the environment's interior shape is already known
    as a boolean grid. It uses this mask directly to define `interior_mask_`
    and `place_bin_centers_`.
    """

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None  # Not typically used
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        mask: NDArray[np.bool_],
        edges: Tuple[NDArray[np.float64], ...],
    ) -> None:
        """Builds the environment from a mask and edges.

        Parameters
        ----------
        mask : NDArray[np.bool_]
            An N-dimensional boolean array where True indicates bins that are
            part of the environment interior.
        edges : Tuple[NDArray[np.float64], ...]
            A tuple of 1D arrays, where each array defines the bin edge
            positions for one dimension. The length of `edges` must match
            `mask.ndim`. The number of bins derived from `edges[i]`
            (i.e., `len(edges[i]) - 1`) must match `mask.shape[i]`.

        Raises
        ------
        ValueError
            If the shape of the mask is inconsistent with the dimensions
            derived from `edges`.
        """
        self.edges_ = edges
        self.centers_shape_ = mask.shape
        self.interior_mask_ = mask

        centers_list = [get_centers(edge_dim) for edge_dim in self.edges_]
        if not all(len(cl) == s for cl, s in zip(centers_list, self.centers_shape_)):
            raise ValueError(
                "Shape mismatch between mask and centers derived from edges."
            )

        mesh = np.meshgrid(*centers_list, indexing="ij")
        self.place_bin_centers_ = np.stack([m.ravel() for m in mesh], axis=1)

        self.track_graph_nd_ = _make_nd_track_graph(
            self.place_bin_centers_, self.interior_mask_, self.centers_shape_
        )
        self._build_kdtree(interior_mask=self.interior_mask_)

    @property
    def is_1d(self) -> bool:
        """Returns False, as MaskedGridEngine is typically N-D."""
        return False

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the 2D masked grid, or a graph representation for N-D.

        Similar to `RegularGridEngine.plot`. For 2D, uses `pcolormesh`
        with `self.interior_mask_`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
        **kwargs
            Additional keyword arguments:
            figsize : tuple, optional
                Figure size `(width, height)` if `ax` is None. Default (7,7) for 2D.
            cmap : str, optional
                Colormap for `pcolormesh` (2D plot). Default 'bone_r'.
            alpha : float, optional
                Alpha transparency for `pcolormesh` (2D plot). Default 0.7.
            node_size : int, optional
                Node size for N-D graph fallback plot. Default 10.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.

        Raises
        ------
        RuntimeError
            If the engine has not been built yet.
        """
        # This plot method can be very similar to RegularGridEngine's plot
        if (
            self.place_bin_centers_ is None
            or self.edges_ is None
            or self.interior_mask_ is None
        ):
            raise RuntimeError("Engine has not been built yet. Call `build` first.")

        if len(self.centers_shape_) != 2:
            # Fallback for non-2D grids (same as RegularGridEngine)
            if ax is None:
                fig = plt.figure()
                is_3d_data = self.place_bin_centers_.shape[1] == 3
                ax = fig.add_subplot(111, projection="3d" if is_3d_data else None)  # type: ignore
            if self.track_graph_nd_ is not None:
                pos = {
                    i: self.place_bin_centers_[i] for i in self.track_graph_nd_.nodes()
                }
                nx.draw(
                    self.track_graph_nd_,
                    pos,
                    ax=ax,
                    node_size=kwargs.get("node_size", 10),
                )
            elif self.place_bin_centers_.shape[0] > 0:
                if self.place_bin_centers_.shape[1] == 2:
                    ax.scatter(
                        self.place_bin_centers_[:, 0],
                        self.place_bin_centers_[:, 1],
                        **kwargs,
                    )
                # Add other dim scatters if needed
            ax.set_title(f"{self.__class__.__name__} (N-D fallback)")
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))

        ax.pcolormesh(
            self.edges_[0],
            self.edges_[1],
            self.interior_mask_.T,  # Transpose for pcolormesh convention
            cmap=kwargs.get("cmap", "bone_r"),
            alpha=kwargs.get("alpha", 0.7),
            shading="auto",
        )

        ax.set_xticks(self.edges_[0])
        ax.set_yticks(self.edges_[1])
        ax.grid(True, ls="-", lw=0.5, c="gray")
        ax.set_aspect("equal")
        ax.set_title(f"{self.__class__.__name__} (2D Masked Grid)")
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        if self.edges_ and len(self.edges_) == 2:
            ax.set_xlim(self.edges_[0][0], self.edges_[0][-1])
            ax.set_ylim(self.edges_[1][0], self.edges_[1][-1])
        return ax


class DelaunayGraphEngine(_KDTreeMixin):
    """Build graph via Delaunay triangulation of point cloud. All points are considered interior."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()  # No explicit grid edges
    centers_shape_: Tuple[int, ...]  # (n_points,)
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = None  # Alias to track_graph_nd_
    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(
        self,
        *,
        points: NDArray[np.float64],
        max_edge_length: Optional[float] = None,
    ) -> None:
        if points.ndim != 2:
            raise ValueError("Input points must be a 2D array (n_points, n_dims).")
        self.place_bin_centers_ = np.asarray(points)
        n_points = self.place_bin_centers_.shape[0]
        if n_points == 0:
            raise ValueError("Input points array cannot be empty.")

        self.centers_shape_ = (n_points,)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        if points.shape[1] < 2:  # Delaunay requires at least 2 dimensions
            # For 1D points, create a simple sequential graph
            G = nx.Graph()
            G.add_nodes_from(range(n_points))
            for i in range(n_points):
                G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
                G.nodes[i]["is_track_interior"] = True
                G.nodes[i]["bin_ind"] = (i,)
                G.nodes[i]["bin_ind_flat"] = i
            if n_points > 1:
                sorted_indices = np.argsort(self.place_bin_centers_[:, 0])
                for i in range(n_points - 1):
                    u, v = sorted_indices[i], sorted_indices[i + 1]
                    d = np.linalg.norm(
                        self.place_bin_centers_[u] - self.place_bin_centers_[v]
                    )
                    if max_edge_length is None or d <= max_edge_length:
                        G.add_edge(u, v, distance=d)
        else:  # 2D or higher
            tri = Delaunay(self.place_bin_centers_)
            G = nx.Graph()
            G.add_nodes_from(range(n_points))

            for i in range(n_points):
                G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
                G.nodes[i]["is_track_interior"] = True
                G.nodes[i]["bin_ind"] = (i,)  # N-D index is just (flat_idx,)
                G.nodes[i]["bin_ind_flat"] = i

            for simplex in tri.simplices:
                for i, j in itertools.combinations(simplex, 2):
                    d = np.linalg.norm(
                        self.place_bin_centers_[i] - self.place_bin_centers_[j]
                    )
                    if max_edge_length is None or d <= max_edge_length:
                        G.add_edge(i, j, distance=d)

        for eid, (u, v) in enumerate(G.edges()):  # Add edge_id if not present
            if "edge_id" not in G.edges[u, v]:
                G.edges[u, v]["edge_id"] = eid

        self.track_graph_nd_ = G
        self.track_graph_bin_centers_ = (
            G  # Alias for consistency as it's a point-based graph
        )
        self._build_kdtree(
            interior_mask=self.interior_mask_
        )  # interior_mask is all True

    @property
    def is_1d(self) -> bool:
        # Delaunay is typically for 2D+ point clouds.
        # If it was built with 1D points, graph might be 1D-like.
        # For simplicity, consider it False unless specifically built for 1D behavior.
        if (
            self.place_bin_centers_ is not None
            and self.place_bin_centers_.shape[1] == 1
        ):
            return True  # If input points were 1D
        return False

    def _generic_graph_plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        graph_to_plot: Optional[nx.Graph] = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        if self.place_bin_centers_ is None:
            raise RuntimeError("Engine not built.")
        if graph_to_plot is None:
            graph_to_plot = self.track_graph_nd_  # Default to nd graph

        if graph_to_plot is None or graph_to_plot.number_of_nodes() == 0:
            if ax is None:
                _, ax = plt.subplots()
            # Scatter plot if no graph or empty graph
            if self.place_bin_centers_.shape[0] > 0:
                if self.place_bin_centers_.shape[1] == 1:
                    ax.scatter(
                        self.place_bin_centers_[:, 0],
                        np.zeros_like(self.place_bin_centers_[:, 0]),
                        **kwargs.get("scatter_kwargs", {}),
                    )
                elif self.place_bin_centers_.shape[1] == 2:
                    ax.scatter(
                        self.place_bin_centers_[:, 0],
                        self.place_bin_centers_[:, 1],
                        **kwargs.get("scatter_kwargs", {}),
                    )
                elif self.place_bin_centers_.shape[1] == 3:
                    if ax is None or not hasattr(ax, "scatter3D"):
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection="3d")  # type: ignore
                    ax.scatter(self.place_bin_centers_[:, 0], self.place_bin_centers_[:, 1], self.place_bin_centers_[:, 2], **kwargs.get("scatter_kwargs", {}))  # type: ignore
            ax.set_title(f"{self.__class__.__name__} Centers")
            return ax  # type: ignore

        node_positions = {
            node: data["pos"]
            for node, data in graph_to_plot.nodes(data=True)
            if "pos" in data
        }
        if not node_positions:  # Fallback if 'pos' attribute is missing
            if ax is None:
                _, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "Graph nodes lack 'pos' attribute for plotting.",
                ha="center",
                va="center",
            )
            return ax  # type: ignore

        is_3d = len(next(iter(node_positions.values()))) == 3

        if ax is None:
            fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            ax = fig.add_subplot(111, projection="3d" if is_3d else None)  # type: ignore
        elif is_3d and not hasattr(ax, "plot3D"):
            raise ValueError("Provided 'ax' is not 3D, but data is 3D.")

        default_node_kwargs = {"node_size": 20}
        node_kwargs = {**default_node_kwargs, **kwargs.get("node_kwargs", {})}
        nx.draw_networkx_nodes(graph_to_plot, pos=node_positions, ax=ax, **node_kwargs)  # type: ignore

        default_edge_kwargs = {"alpha": 0.5, "edge_color": "gray"}
        edge_kwargs = {**default_edge_kwargs, **kwargs.get("edge_kwargs", {})}
        nx.draw_networkx_edges(graph_to_plot, pos=node_positions, ax=ax, **edge_kwargs)  # type: ignore

        ax.set_title(f"{self.__class__.__name__} Graph")
        ax.set_xlabel("Dim 0")
        ax.set_ylabel("Dim 1")
        if is_3d and hasattr(ax, "set_zlabel"):
            ax.set_zlabel("Dim 2")  # type: ignore

        if not is_3d:
            ax.set_aspect("equal", adjustable="box")
        elif hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])  # type: ignore
        return ax  # type: ignore

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the Delaunay graph."""
        return self._generic_graph_plot(ax, self.track_graph_nd_, **kwargs)


class HexagonalGridEngine(_KDTreeMixin):
    """Tiles a 2D rectangle into a hexagonal lattice."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()  # No explicit grid edges
    centers_shape_: Tuple[int, ...]  # (n_points,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    place_bin_size: Optional[float] = None

    def build(
        self,
        *,
        place_bin_size: float,
        position_range: Optional[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = None,
        position: Optional[NDArray[np.float64]] = None,
        infer_track_interior: bool = False,
        bin_count_threshold: int = 0,
    ) -> None:
        """Builds the 2D hexagonal grid.

        Parameters
        ----------
        place_bin_size : float
            The side length of the hexagons, which is also approximately the
            distance between the centers of adjacent hexagons in the tiling.
        position_range : Tuple[Tuple[float, float], Tuple[float, float]]
            The 2D rectangular area `((xmin, xmax), (ymin, ymax))` to tile.
            Hexagon centers initially generated within this range are considered.
        position : Optional[NDArray[np.float64]], shape (n_time, 2), optional
            2D position data. If `infer_track_interior` is True, this data is
            used to determine which generated hexagons are part of the track.
            Defaults to None.
        infer_track_interior : bool, default=False
            If True and `position` data is provided, infers the track interior
            by keeping only hexagons whose centers are sufficiently occupied by
            the position data.
        bin_count_threshold : int, default=0
            If `infer_track_interior` is True, this is the minimum number of
            position samples that must map to a hexagon's center (or its vicinity)
            for it to be considered part of the track interior.

        Raises
        ------
        ValueError
            If `place_bin_size` is not positive, or if `position` data is
            provided for inference and is not 2D.
        """
        if place_bin_size <= 0:
            raise ValueError("place_bin_size must be positive.")
        self.place_bin_size = place_bin_size

        current_position_range: Tuple[Tuple[float, float], Tuple[float, float]]

        if position_range is not None:
            current_position_range = position_range
        elif position is not None:
            if (
                position.ndim != 2 or position.shape[1] != 2
            ):  # Assuming 2D for Hexagonal
                raise ValueError(
                    "If position_range is not provided, position data must be 2D."
                )

            pos_clean = position[~np.any(np.isnan(position), axis=1)]
            if pos_clean.shape[0] == 0:
                raise ValueError(
                    "If position_range is not provided, position data cannot be all NaNs or empty."
                )

            xmin, ymin = np.nanmin(pos_clean, axis=0)
            xmax, ymax = np.nanmax(pos_clean, axis=0)

            # Handle cases where data might be a single point or collinear
            if np.isclose(xmin, xmax):
                warnings.warn(
                    "X-extent of position data is zero. Creating a minimal range around the data for X.",
                    UserWarning,
                )
                xmin -= place_bin_size  # Or some other sensible default extent
                xmax += place_bin_size
            if np.isclose(ymin, ymax):
                warnings.warn(
                    "Y-extent of position data is zero. Creating a minimal range around the data for Y.",
                    UserWarning,
                )
                ymin -= place_bin_size
                ymax += place_bin_size

            current_position_range = ((xmin, xmax), (ymin, ymax))
            warnings.warn(
                f"position_range not provided, inferred as {current_position_range} from position data.",
                UserWarning,
            )
        else:
            raise ValueError(
                "Either 'position_range' or 'position' data must be provided "
                "for HexagonalGridEngine to define its tiling area."
            )

        (xmin, xmax), (ymin, ymax) = current_position_range
        s = place_bin_size  # side length

        # Geometry for point-up hexagons (centers form a triangular lattice)
        hex_width_total = 2 * s  # Distance between parallel vertical sides
        hex_height_total = (
            math.sqrt(3) * s
        )  # Distance between parallel horizontal sides (for flat-top)
        # For pointy-top, this is the full height.

        # Spacing between centers for a pointy-topped hexagonal grid
        # dx_centers: horizontal distance between centers of hexagons in the same "conceptual" row.
        # (e.g. (0,0) and (s*sqrt(3), 0) if first row starts at y=0)
        # For a common "axial coordinate" type system or dense packing:
        # horizontal distance between centers: s * 1.5
        # vertical distance between rows of centers: s * sqrt(3) / 2
        col_spacing_x = s * 1.5
        row_spacing_y = s * math.sqrt(3) / 2.0

        # --- Step 1: Generate all candidate hexagon centers ---
        initial_centers_list: List[Tuple[float, float]] = []
        row_idx = 0
        current_y = ymin
        # Extend generation slightly to catch hexagons whose centers are just outside
        # but might cover the boundary.
        while (
            current_y < ymax + row_spacing_y
        ):  # Add hex_height_total/2 or s for better coverage margin
            # Offset for staggering: even/odd rows are shifted
            current_x_offset = (col_spacing_x / 2.0) if (row_idx % 2 != 0) else 0.0
            current_x = xmin + current_x_offset
            while (
                current_x < xmax + col_spacing_x
            ):  # Add hex_width_total/2 or s for margin
                # Crude check if center is roughly near the desired area before strict filtering
                if (xmin - s) <= current_x <= (xmax + s) and (
                    ymin - s
                ) <= current_y <= (
                    ymax + s
                ):  # Margin s for pointy-top height consideration
                    initial_centers_list.append((current_x, current_y))
                current_x += col_spacing_x
            current_y += row_spacing_y
            row_idx += 1

        if not initial_centers_list:
            initial_candidate_centers = np.empty((0, 2))
        else:
            temp_centers = np.array(initial_centers_list)
            # Filter to keep only centers strictly within the original range
            strict_valid_mask = (
                (temp_centers[:, 0] >= xmin)
                & (temp_centers[:, 0] <= xmax)
                & (temp_centers[:, 1] >= ymin)
                & (temp_centers[:, 1] <= ymax)
            )
            initial_candidate_centers = temp_centers[strict_valid_mask]

        n_initial_candidates = initial_candidate_centers.shape[0]

        # --- Step 2: Initial Interior Mask (all generated candidates) ---
        # This 1D mask corresponds to initial_candidate_centers
        calculated_interior_mask_1d = np.ones(n_initial_candidates, dtype=bool)

        # --- Step 3: Infer Interior based on position data (if requested) ---
        if infer_track_interior and position is not None:
            if n_initial_candidates == 0:  # No candidates to infer upon
                warnings.warn(
                    "Cannot infer interior: no initial hexagon candidates generated for the given range.",
                    UserWarning,
                )
                self.place_bin_centers_ = np.empty((0, 2))
                self.centers_shape_ = (0,)
                self.interior_mask_ = np.array([], dtype=bool)
                self.track_graph_bin_centers_ = nx.Graph()
                self._build_kdtree(interior_mask=self.interior_mask_)
                return

            if position.ndim != 2 or position.shape[1] != 2:
                raise ValueError(
                    f"Position data shape {position.shape} must be (n_time, 2) for HexagonalGridEngine."
                )

            valid_positions = position[~np.any(np.isnan(position), axis=1)]
            if valid_positions.shape[0] > 0:
                # Map positions to the initial set of candidate hexagon centers
                temp_kdtree = KDTree(initial_candidate_centers)
                _, assigned_bin_indices = temp_kdtree.query(valid_positions)

                occupancy_counts = np.bincount(
                    assigned_bin_indices, minlength=n_initial_candidates
                )
                calculated_interior_mask_1d = occupancy_counts > bin_count_threshold
            else:  # No valid positions after NaN removal
                calculated_interior_mask_1d = np.zeros(n_initial_candidates, dtype=bool)
                warnings.warn(
                    "No valid positions provided for interior inference after NaN removal.",
                    UserWarning,
                )

            if not np.any(calculated_interior_mask_1d):
                warnings.warn(
                    "Inferring interior resulted in no hexagon centers meeting the threshold. "
                    "The environment will be empty of active bins if this was the only criterion.",
                    UserWarning,
                )

        # --- Step 4: Update attributes based on the final interior mask ---
        self.interior_mask_ = calculated_interior_mask_1d  # This is the crucial 1D mask

        # Filter place_bin_centers_ to only include interior ones
        self.place_bin_centers_ = initial_candidate_centers[self.interior_mask_]
        self.centers_shape_ = (
            self.place_bin_centers_.shape[0],
        )  # Shape of the *final* interior bins

        # --- Step 5: Build Graph on FINAL INTERIOR hexagon centers ---
        G = nx.Graph()
        n_final_interior_points = self.place_bin_centers_.shape[0]

        if n_final_interior_points > 0:
            for i_node in range(n_final_interior_points):
                G.add_node(
                    i_node,  # Node ID is index in the final self.place_bin_centers_
                    pos=tuple(self.place_bin_centers_[i_node]),
                    is_track_interior=True,  # By definition, these are now the interior bins
                    bin_ind=(i_node,),  # Simple index for point-based engines
                    bin_ind_flat=i_node,
                )

            if n_final_interior_points > 1:
                # Build KDTree on the *final* interior points for neighbor finding
                final_kdtree = KDTree(self.place_bin_centers_)
                # Connect pairs within s * 1.1 (s is side length / inter-center distance)
                # This captures the 6 immediate neighbors in a perfect lattice.
                # The actual distance between centers of touching hexagons is 's'.
                # Using a slightly larger radius for robustness.
                pairs = final_kdtree.query_pairs(r=s * 1.05)  # More precise radius
                for (
                    u_new,
                    v_new,
                ) in pairs:  # u_new, v_new are indices for self.place_bin_centers_
                    dist = np.linalg.norm(
                        self.place_bin_centers_[u_new] - self.place_bin_centers_[v_new]
                    )
                    # Additional check: ensure distance is very close to 's' for true hex neighbors
                    if np.isclose(dist, s):  # Only connect true hexagonal neighbors
                        G.add_edge(u_new, v_new, distance=dist)

        self.track_graph_bin_centers_ = G

        if n_final_interior_points > 0:
            self._build_kdtree(
                interior_mask=np.ones(n_final_interior_points, dtype=bool)
            )
        else:
            self._build_kdtree(interior_mask=np.array([], dtype=bool))

    @property
    def is_1d(self) -> bool:
        return False  # Hexagonal grid is 2D

    # Use generic graph plot from Delaunay engine as it's suitable
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the hexagonal grid graph."""
        return self._generic_graph_plot(ax, self.track_graph_bin_centers_, **kwargs)


class QuadtreeGridEngine(_KDTreeMixin):
    """Engine that adaptively tiles a 2D space using a Quadtree structure.

    The space is recursively subdivided into quadrants up to a maximum depth.
    The centers of the leaf cells of the quadtree become the `place_bin_centers_`.
    A graph connects adjacent leaf cells.
    """

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    quadtree_cells_: Optional[List[Tuple[float, float, float, float]]] = (
        None  # (x0,y0,x1,y1)
    )

    def build(
        self,
        *,
        position_range: Tuple[
            Tuple[float, float], Tuple[float, float]
        ],  # (xmin, xmax), (ymin, ymax)
        max_depth: int = 4,
    ) -> None:
        """Builds the Quadtree grid.

        Parameters
        ----------
        position_range : Tuple[Tuple[float, float], Tuple[float, float]]
            The initial rectangular area `((xmin, xmax), (ymin, ymax))` to subdivide.
        max_depth : int, default=4
            The maximum depth of the quadtree subdivision. A depth of 0 means
            a single cell covering the `position_range`.

        Raises
        ------
        ValueError
            If `max_depth` is negative or `position_range` is invalid.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")

        (xmin, xmax), (ymin, ymax) = position_range
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "position_range min must be less than max for both dimensions."
            )

        leaf_nodes_centers: List[Tuple[float, float]] = []
        leaf_nodes_bounds: List[Tuple[float, float, float, float]] = []

        def subdivide(x0, y0, x1, y1, current_depth):
            if current_depth == max_depth:
                leaf_nodes_centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
                leaf_nodes_bounds.append((x0, y0, x1, y1))
                return

            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            subdivide(x0, y0, mx, my, current_depth + 1)
            subdivide(mx, y0, x1, my, current_depth + 1)
            subdivide(x0, my, mx, y1, current_depth + 1)
            subdivide(mx, my, x1, y1, current_depth + 1)

        subdivide(xmin, ymin, xmax, ymax, 0)

        self.place_bin_centers_ = np.array(leaf_nodes_centers)
        self.quadtree_cells_ = leaf_nodes_bounds

        if self.place_bin_centers_.shape[0] == 0:
            self.centers_shape_ = (0,)
            self.interior_mask_ = np.array([], dtype=bool)
            self.track_graph_bin_centers_ = nx.Graph()
        else:
            self.centers_shape_ = (self.place_bin_centers_.shape[0],)
            self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

            G = nx.Graph()
            G.add_nodes_from(range(len(self.place_bin_centers_)))
            for i in range(len(self.place_bin_centers_)):
                G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
                G.nodes[i]["is_track_interior"] = True
                G.nodes[i]["bin_ind"] = (i,)
                G.nodes[i]["bin_ind_flat"] = i
                G.nodes[i]["bounds"] = self.quadtree_cells_[i]

            min_cell_width = (xmax - xmin) / (2**max_depth)
            min_cell_height = (ymax - ymin) / (2**max_depth)
            connect_radius = 1.5 * max(min_cell_width, min_cell_height)

            if self.place_bin_centers_.shape[0] > 1:
                tree = KDTree(self.place_bin_centers_)
                pairs = tree.query_pairs(r=connect_radius)
                for i, j in pairs:
                    dist = np.linalg.norm(
                        self.place_bin_centers_[i] - self.place_bin_centers_[j]
                    )
                    G.add_edge(i, j, distance=dist)
            self.track_graph_bin_centers_ = G

        self._build_kdtree(interior_mask=self.interior_mask_)

    @property
    def is_1d(self) -> bool:
        """Returns False, as QuadtreeGridEngine is 2D."""
        return False

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the Quadtree graph and optionally the cell boundaries.

        Uses a generic graph plotting utility for the cell centers and their
        connections. See `_generic_graph_plot` for common `**kwargs`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
        **kwargs
            Additional keyword arguments:
            show_cell_boundaries : bool, optional
                If True, draws the rectangular boundaries of the quadtree
                leaf cells. Default False.
            cell_boundary_kwargs : dict, optional
                Keyword arguments for plotting cell boundaries (passed to
                `matplotlib.patches.Rectangle`). Default
                `{'fill': False, 'edgecolor': 'blue', 'alpha': 0.3}`.
            Other kwargs are passed to `_generic_graph_plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.
        """
        ax = self._generic_graph_plot(ax, self.track_graph_bin_centers_, **kwargs)  # type: ignore

        if (
            kwargs.get("show_cell_boundaries", False)
            and self.quadtree_cells_ is not None
        ):
            from matplotlib.patches import Rectangle

            for x0, y0, x1, y1 in self.quadtree_cells_:
                rect = Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="blue", alpha=0.3
                )
                ax.add_patch(rect)
        ax.set_title(f"{self.__class__.__name__} Graph")  # Override title
        return ax


class VoronoiPartitionEngine(_KDTreeMixin):
    """Engine that partitions N-D space using Voronoi tessellation of seed points.

    Bin centers (`place_bin_centers_`) are defined as the centroids of the
    finite Voronoi regions generated from the input seed points.
    The graph connects centroids of Voronoi regions that share a ridge.
    """

    place_bin_centers_: NDArray[np.float64]  # Centroids of finite Voronoi regions
    edges_: Tuple[NDArray[np.float64], ...] = ()  # No regular grid edges
    centers_shape_: Tuple[int, ...]  # (n_finite_regions,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = (
        None  # Graph of Voronoi region centroids
    )
    interior_mask_: Optional[NDArray[np.bool_]] = None  # All True for finite regions
    voronoi_diagram_: Optional[Voronoi] = None  # The scipy.spatial.Voronoi object
    seed_points_: Optional[NDArray[np.float64]] = None  # Original input seed points

    def build(self, *, seeds: NDArray[np.float64]) -> None:
        """Builds the Voronoi partition.

        Parameters
        ----------
        seeds : NDArray[np.float64], shape (n_seeds, n_dims)
            The seed points used to generate the Voronoi diagram.
            Requires at least `n_dims + 1` seed points.

        Raises
        ------
        ValueError
            If `seeds` is not a 2D array or has insufficient points for
            the dimensionality.
        """
        if seeds.ndim != 2:
            raise ValueError("Input seeds must be a 2D array (n_seeds, n_dims).")
        if seeds.shape[0] < seeds.shape[1] + 1:
            raise ValueError(
                f"Need at least {seeds.shape[1]+1} seed points for {seeds.shape[1]}-D Voronoi."
            )
        self.seed_points_ = seeds
        self.voronoi_diagram_ = Voronoi(seeds)

        finite_region_centroids: List[NDArray[np.float64]] = []
        finite_region_indices: List[int] = []

        for i, region_idx in enumerate(self.voronoi_diagram_.point_region):
            region_vertices_indices = self.voronoi_diagram_.regions[region_idx]
            if not region_vertices_indices or -1 in region_vertices_indices:
                continue
            region_vertices = self.voronoi_diagram_.vertices[region_vertices_indices]
            finite_region_centroids.append(np.mean(region_vertices, axis=0))
            finite_region_indices.append(i)

        self.place_bin_centers_ = (
            np.array(finite_region_centroids)
            if finite_region_centroids
            else np.empty((0, seeds.shape[1]))
        )
        self.centers_shape_ = (self.place_bin_centers_.shape[0],)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        G = nx.Graph()
        if self.place_bin_centers_.shape[0] == 0:
            self.track_graph_bin_centers_ = G
            self._build_kdtree(interior_mask=self.interior_mask_)
            return

        seed_idx_to_graph_node_id: Dict[int, int] = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(finite_region_indices)
        }

        G.add_nodes_from(range(self.place_bin_centers_.shape[0]))
        for i in range(self.place_bin_centers_.shape[0]):
            original_seed_idx = finite_region_indices[i]
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i
            G.nodes[i]["original_seed_index"] = original_seed_idx

        for ridge_point_indices, _ in zip(
            self.voronoi_diagram_.ridge_points, self.voronoi_diagram_.ridge_vertices
        ):
            p1_original_idx, p2_original_idx = ridge_point_indices
            if (
                p1_original_idx in seed_idx_to_graph_node_id
                and p2_original_idx in seed_idx_to_graph_node_id
            ):
                node_u = seed_idx_to_graph_node_id[p1_original_idx]
                node_v = seed_idx_to_graph_node_id[p2_original_idx]
                dist = np.linalg.norm(
                    self.place_bin_centers_[node_u] - self.place_bin_centers_[node_v]
                )
                G.add_edge(node_u, node_v, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)

    @property
    def is_1d(self) -> bool:
        """Checks if the input seed points were 1-dimensional."""
        if self.seed_points_ is not None and self.seed_points_.shape[1] == 1:
            return True
        return False

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the graph of Voronoi region centroids and optionally the Voronoi diagram.

        Uses a generic graph plotting utility for the centroids and their
        connections. See `_generic_graph_plot` for common `**kwargs`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
        **kwargs
            Additional keyword arguments:
            show_voronoi_diagram : bool, optional
                If True and the environment is 2D, plots the Voronoi diagram
                (ridges and vertices) using `scipy.spatial.voronoi_plot_2d`.
                Default False.
            show_seed_points : bool, optional
                If True, scatter plots the original seed points. Default False.
            seed_point_kwargs : dict, optional
                Keyword arguments for plotting seed points.
                Default `{'c': 'red', 's': 25, 'marker': 'x'}`.
            voronoi_plot_kwargs : dict, optional
                Keyword arguments passed to `scipy.spatial.voronoi_plot_2d`.
                E.g., `{'show_vertices': False, 'line_colors': 'orange'}`.
            Other kwargs are passed to `_generic_graph_plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.
        """
        ax = self._generic_graph_plot(ax, self.track_graph_bin_centers_, **kwargs)  # type: ignore

        if (
            kwargs.get("show_voronoi_diagram", False)
            and self.voronoi_diagram_ is not None
        ):
            from scipy.spatial import voronoi_plot_2d  # Conditional import

            try:
                voronoi_plot_2d(
                    self.voronoi_diagram_,
                    ax=ax,
                    show_vertices=kwargs.get("show_voronoi_vertices", True),
                    line_colors=kwargs.get("voronoi_line_colors", "orange"),
                    line_width=kwargs.get("voronoi_line_width", 1),
                    point_size=kwargs.get("voronoi_point_size", 2),
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"Could not plot Voronoi diagram: {e}")
        ax.set_title(f"{self.__class__.__name__} Graph")  # Override title
        return ax


class MeshSurfaceEngine(_KDTreeMixin):
    """Engine that uses a pre-existing triangular mesh (vertices and faces).

    The vertices of the mesh become the `place_bin_centers_`. The graph connects
    vertices that share an edge in any triangular face.
    """

    place_bin_centers_: NDArray[np.float64]  # Mesh vertices
    edges_: Tuple[NDArray[np.float64], ...] = ()  # No regular grid edges
    centers_shape_: Tuple[int, ...]  # (n_vertices,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used for this engine
    track_graph_bin_centers_: Optional[nx.Graph] = (
        None  # The mesh graph (vertices as nodes)
    )
    interior_mask_: Optional[NDArray[np.bool_]] = None  # All True for provided vertices
    faces_definition_: Optional[NDArray[np.int_]] = None  # Input faces (n_faces, 3)

    def build(self, *, vertices: NDArray[np.float64], faces: NDArray[np.int_]) -> None:
        """Builds the graph from mesh vertices and faces.

        Parameters
        ----------
        vertices : NDArray[np.float64], shape (n_vertices, n_dims)
            The coordinates of the mesh vertices.
        faces : NDArray[np.int_], shape (n_faces, 3)
            An array where each row contains three integer indices referring to
            rows in the `vertices` array, defining a triangular face.

        Raises
        ------
        ValueError
            If `vertices` or `faces` have incorrect shapes or if face indices
            are out of bounds.
        """
        if vertices.ndim != 2:
            raise ValueError("Vertices must be a 2D array (n_vertices, n_dims).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces must be a 2D array (n_faces, 3) of vertex indices.")

        self.place_bin_centers_ = vertices.copy()
        self.faces_definition_ = faces.copy()
        self.centers_shape_ = (vertices.shape[0],)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        G = nx.Graph()
        if vertices.shape[0] == 0:
            self.track_graph_bin_centers_ = G
            self._build_kdtree(interior_mask=self.interior_mask_)
            return

        G.add_nodes_from(range(len(vertices)))
        for i in range(len(vertices)):
            G.nodes[i]["pos"] = tuple(vertices[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i

        for tri_face in faces:
            if np.any(tri_face >= len(vertices)) or np.any(tri_face < 0):
                raise ValueError("Face indices out of bounds for vertices.")
            i, j, k = tri_face
            for u, v in ((i, j), (j, k), (k, i)):
                if not G.has_edge(u, v):
                    dist = np.linalg.norm(vertices[u] - vertices[v])
                    G.add_edge(u, v, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)

    @property
    def is_1d(self) -> bool:
        """Checks if the mesh vertices are 1-dimensional."""
        # Typically a 2D surface in 3D space, or a 2D mesh.
        # Could be 1D if vertices are collinear and faces connect them sequentially, but not typical.
        if (
            self.place_bin_centers_ is not None
            and self.place_bin_centers_.shape[1] == 1
        ):
            return True
        return False

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the mesh graph and optionally the mesh faces.

        Uses a generic graph plotting utility for the vertices and their
        connections. See `_generic_graph_plot` for common `**kwargs`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
        **kwargs
            Additional keyword arguments:
            show_mesh_faces : bool, optional
                If True and the mesh is 3D (`self.place_bin_centers_.shape[1] == 3`),
                plots the triangular faces using `ax.plot_trisurf`. Default False.
            mesh_kwargs : dict, optional
                Keyword arguments passed to `ax.plot_trisurf` if
                `show_mesh_faces` is True. Default `{'color': 'lightblue', 'alpha': 0.5}`.
            Other kwargs are passed to `_generic_graph_plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.
        """
        ax = self._generic_graph_plot(ax, self.track_graph_bin_centers_, **kwargs)  # type: ignore

        if (
            kwargs.get("show_mesh_faces", False)
            and self.faces_definition_ is not None
            and self.place_bin_centers_ is not None
            and self.place_bin_centers_.shape[1] == 3
            and hasattr(ax, "plot_trisurf")
        ):  # Only for 3D

            ax.plot_trisurf(  # type: ignore
                self.place_bin_centers_[:, 0],
                self.place_bin_centers_[:, 1],
                self.place_bin_centers_[:, 2],
                triangles=self.faces_definition_,
                color=kwargs.get("mesh_color", "lightblue"),
                alpha=kwargs.get("mesh_alpha", 0.5),
            )
        ax.set_title(f"{self.__class__.__name__} Graph")  # Override title
        return ax


class ImageMaskEngine(_KDTreeMixin):
    """Engine that converts a 2D boolean image mask into pixel-center bins and a graph.

    Each True pixel in the input mask becomes a bin center (offset by 0.5 to be
    at the pixel center). The graph connects adjacent True pixels, either using
    4-connectivity (orthogonal neighbors) or 8-connectivity (orthogonal + diagonal).
    """

    place_bin_centers_: NDArray[np.float64]  # Centers of True pixels (col+0.5, row+0.5)
    edges_: Tuple[
        NDArray[np.float64], ...
    ] = ()  # No regular grid edges in the typical sense
    centers_shape_: Tuple[int, ...]  # (n_true_pixels,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = None  # Graph of True pixel centers
    interior_mask_: Optional[NDArray[np.bool_]] = (
        None  # All True for generated pixel centers
    )
    pixel_to_node_map_: Optional[Dict[Tuple[int, int], int]] = (
        None  # Maps (row,col) of True pixel to node_id
    )
    image_mask_definition_: Optional[NDArray[np.bool_]] = (
        None  # The input 2D boolean mask
    )

    def build(self, *, mask: NDArray[np.bool_], connect_diagonal: bool = True) -> None:
        """Builds the graph from the image mask.

        Parameters
        ----------
        mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where True indicates pixels that are part of
            the environment.
        connect_diagonal : bool, default=False
            If True, connects pixels that are diagonally adjacent (8-connectivity).
            If False, only connects orthogonally adjacent pixels (4-connectivity).

        Raises
        ------
        ValueError
            If `mask` is not a 2D array.
        """
        if mask.ndim != 2:
            raise ValueError("ImageMaskEngine currently only supports 2D masks.")
        self.image_mask_definition_ = mask

        row_coords, col_coords = np.nonzero(mask)
        self.place_bin_centers_ = np.stack((col_coords + 0.5, row_coords + 0.5), axis=1)

        num_interior_pixels = self.place_bin_centers_.shape[0]
        self.centers_shape_ = (num_interior_pixels,)
        self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)

        G = nx.Graph()
        if num_interior_pixels == 0:
            self.track_graph_bin_centers_ = G
            self._build_kdtree(interior_mask=self.interior_mask_)
            return

        self.pixel_to_node_map_ = {}
        for i in range(num_interior_pixels):
            r, c = row_coords[i], col_coords[i]
            self.pixel_to_node_map_[(r, c)] = i
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i
            G.nodes[i]["pixel_coord"] = (r, c)

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connect_diagonal:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for i in range(num_interior_pixels):
            r1, c1 = row_coords[i], col_coords[i]
            for dr, dc in offsets:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < mask.shape[0] and 0 <= c2 < mask.shape[1] and mask[r2, c2]:
                    if self.pixel_to_node_map_ is None:
                        continue  # Should not happen
                    j = self.pixel_to_node_map_[(r2, c2)]
                    if i < j:
                        dist = np.linalg.norm(
                            self.place_bin_centers_[i] - self.place_bin_centers_[j]
                        )
                        G.add_edge(i, j, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)

    @property
    def is_1d(self) -> bool:
        """Returns False, as ImageMaskEngine is 2D."""
        return False

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the graph derived from the image mask, or the mask itself.

        If `show_image_mask` is True in `**kwargs`, displays the input boolean
        mask using `imshow`. Otherwise, plots the graph of connected pixel
        centers using `_generic_graph_plot`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
        **kwargs
            Additional keyword arguments:
            show_image_mask : bool, optional
                If True, displays the original boolean mask image. Default False
                (plots the graph).
            imshow_kwargs : dict, optional
                Keyword arguments passed to `ax.imshow` if `show_image_mask`
                is True. Default `{'origin': 'lower', 'cmap': 'gray', 'interpolation': 'nearest'}`.
            Other kwargs are passed to `_generic_graph_plot` if plotting the graph.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.
        """
        if (
            kwargs.get("show_image_mask", False)
            and self.image_mask_definition_ is not None
        ):
            if ax is None:
                _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))
            ax.imshow(
                self.image_mask_definition_,
                origin="lower",
                cmap=kwargs.get("cmap", "gray"),
                interpolation=kwargs.get("interpolation", "nearest"),
            )
            ax.set_title(f"{self.__class__.__name__} - Image Mask")
            ax.set_xlabel("Columns (X)")
            ax.set_ylabel("Rows (Y)")
        else:
            ax = self._generic_graph_plot(ax, self.track_graph_bin_centers_, **kwargs)  # type: ignore
            ax.set_title(f"{self.__class__.__name__} Graph from Mask")  # Override title
        return ax


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
_ENGINE_MAP: Dict[str, type[GeometryEngine]] = {
    "Grid": RegularGridEngine,
    "RegularGrid": RegularGridEngine,
    "1D": TrackGraphEngine,
    "TrackGraph": TrackGraphEngine,
    "Mask": MaskedGridEngine,
    "MaskedGrid": MaskedGridEngine,
    "Delaunay": DelaunayGraphEngine,
    "Hexagonal": HexagonalGridEngine,
    "Quadtree": QuadtreeGridEngine,
    "Voronoi": VoronoiPartitionEngine,
    "Mesh": MeshSurfaceEngine,
    "ImageMask": ImageMaskEngine,
}

if SHAPELY_AVAILABLE and ShapelyGridEngine is not None:
    _ENGINE_MAP["Shapely"] = ShapelyGridEngine
    _ENGINE_MAP["ShapelyGrid"] = ShapelyGridEngine


def make_engine(kind: str, **kwargs) -> GeometryEngine:
    """Instantiate and build an engine by *kind* string.

    Examples
    --------
    >>> eng = make_engine("Grid", place_bin_size=2.0, position_range=[(0,50),(0,50)])
    >>> hasattr(eng, 'place_bin_centers_') and eng.place_bin_centers_ is not None
    True
    >>> hex_eng = make_engine("Hexagonal", place_bin_size=5.0, position_range=[(0,20),(0,20)])
    >>> hasattr(hex_eng, 'place_bin_centers_') and hex_eng.place_bin_centers_ is not None
    True
    """
    normalized_kind = "".join(filter(str.isalnum, kind)).lower()

    found_key = None
    for k_map in _ENGINE_MAP.keys():
        if "".join(filter(str.isalnum, k_map)).lower() == normalized_kind:
            found_key = k_map
            break

    if found_key is None:
        raise ValueError(
            f"Unknown engine kind '{kind}'. Available: {list(_ENGINE_MAP.keys())}"
        )

    eng_cls = _ENGINE_MAP[found_key]
    eng = eng_cls()
    eng.build(**kwargs)
    return eng


def list_available_engines() -> List[str]:
    """Lists the 'kind' strings for all available geometry engines.

    These kinds can be used for the `engine_kind` parameter when
    initializing an `Environment`.

    Returns
    -------
    List[str]
        A list of unique engine kind identifiers.
    """
    unique_options = []
    processed_normalized_options = set()
    for opt in _ENGINE_MAP.keys():
        # Provide the primary, more readable key if there are aliases
        norm_opt = "".join(filter(str.isalnum, opt)).lower()
        if norm_opt not in processed_normalized_options:
            # Prefer keys that don't have "Grid" if a shorter alias exists, or just take the first encountered.
            # This logic can be refined based on how you want to present aliases.
            is_alias_of_already_added = False
            for added_opt in unique_options:
                if "".join(filter(str.isalnum, added_opt)).lower() == norm_opt:
                    is_alias_of_already_added = True
                    break
            if not is_alias_of_already_added:
                unique_options.append(opt)
            processed_normalized_options.add(norm_opt)
    return sorted(unique_options)
