from __future__ import annotations

import itertools
import math
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
    """Contract all engines must satisfy (with *Environment* field names)."""

    place_bin_centers_: NDArray[np.float64]
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: nx.Graph | None  # For N-D grid-like structures
    track_graph_bin_centers_: nx.Graph | None  # For 1-D or other graph structures

    interior_mask_: Optional[NDArray[np.bool_]] = None

    def build(self, **kwargs) -> None: ...

    def point_to_bin(self, pts: NDArray[np.float64]) -> NDArray[np.int_]: ...

    def neighbors(self, flat_idx: int) -> List[int]: ...

    @property
    @abstractmethod
    def is_1d(self) -> bool:
        """Return True if the engine represents a 1-dimensional environment."""
        ...

    @abstractmethod
    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plot the geometry of the environment.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing axes to plot on. If None, creates new axes.
            Defaults to None.
        **kwargs
            Additional keyword arguments to pass to the plotting functions.
            Common kwargs include `node_size`, `edge_color` for graph plots,
            or `cmap` for grid plots.

        Returns
        -------
        matplotlib.axes.Axes
            The axes used for plotting.
        """
        ...


# ---------------------------------------------------------------------------
# KD-tree mixin implementing generic mapping helpers
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """Mixin providing *point_to_bin* and *neighbors* via KD-tree + NetworkX."""

    _kdtree: Optional[KDTree] = None
    _flat_indices_of_kdtree_nodes: Optional[NDArray[np.int_]] = None

    def _build_kdtree(self, interior_mask: Optional[NDArray[np.bool_]] = None) -> None:
        if not hasattr(self, "place_bin_centers_") or self.place_bin_centers_ is None:
            raise ValueError(
                "place_bin_centers_ not set; call build() before _build_kdtree()"
            )

        if interior_mask is not None:
            # Validate mask shape against place_bin_centers_ and centers_shape_
            expected_mask_len = self.place_bin_centers_.shape[0]
            flat_mask = interior_mask.ravel()

            if len(flat_mask) != expected_mask_len:
                # This case might occur if place_bin_centers_ is already filtered
                # and interior_mask refers to the original full grid.
                # The current logic expects interior_mask to align with self.place_bin_centers_
                # if self.place_bin_centers_ is itself a flattened representation of a grid
                # described by self.centers_shape_.
                # If place_bin_centers_ can be arbitrary points (not from a grid),
                # then interior_mask should be a 1D array of the same length.
                if (
                    self.place_bin_centers_.shape[0] == interior_mask.shape[0]
                    and interior_mask.ndim == 1
                ):
                    pass
                elif (
                    hasattr(self, "centers_shape_")
                    and interior_mask.shape == self.centers_shape_
                ):
                    pass
                else:
                    raise ValueError(
                        f"Flattened interior_mask length {len(flat_mask)} (from shape {interior_mask.shape}) "
                        f"does not match place_bin_centers_ length {expected_mask_len} "
                        f"nor does its shape match centers_shape_ {getattr(self, 'centers_shape_', None)}."
                    )

            points_for_kdtree = self.place_bin_centers_[flat_mask]
            self._flat_indices_of_kdtree_nodes = np.where(flat_mask)[0].astype(np.int32)
        else:
            points_for_kdtree = self.place_bin_centers_
            self._flat_indices_of_kdtree_nodes = np.arange(
                self.place_bin_centers_.shape[0], dtype=np.int32
            )

        if points_for_kdtree.shape[0] > 0:
            self._kdtree = KDTree(points_for_kdtree)
        else:
            self._kdtree = None
            # Ensure _flat_indices_of_kdtree_nodes is empty if no points
            if (
                self._flat_indices_of_kdtree_nodes is not None
                and self._flat_indices_of_kdtree_nodes.size > 0
            ):
                self._flat_indices_of_kdtree_nodes = np.array([], dtype=np.int32)

    def point_to_bin(self, pts: NDArray[np.float64]) -> NDArray[np.int_]:
        if self._kdtree is None:
            return np.full(np.atleast_2d(pts).shape[0], -1, dtype=np.int32)

        if self._flat_indices_of_kdtree_nodes is None:
            # This should not happen if _kdtree is not None and build was successful
            raise RuntimeError(
                "_flat_indices_of_kdtree_nodes not set. Ensure _build_kdtree was called."
            )

        _distances, indices_in_kdtree_subset = self._kdtree.query(np.atleast_2d(pts))

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
        graph_to_use = None
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
    """Axis-aligned rectangular (N-D) grid engine.
    Can infer interior based on position data or define all bins as interior.
    """

    place_bin_centers_: NDArray[np.float64]
    place_bin_edges_: Optional[NDArray[np.float64]] = None
    edges_: Tuple[NDArray[np.float64], ...]
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None
    track_graph_bin_centers_: Optional[nx.Graph] = (
        None  # Not typically used by this engine
    )
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
        """Return True if the engine represents a 1-dimensional environment."""
        return False  # RegularGridEngine is for N-D grids

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the 2D grid and track interior."""
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
    """1-D topological track engine."""

    place_bin_centers_: NDArray[np.float64]
    place_bin_edges_: Optional[NDArray[np.float64]] = None
    edges_: Tuple[NDArray[np.float64], ...]  # Linearized bin edges
    centers_shape_: Tuple[int, ...]
    track_graph_nd_: Optional[nx.Graph] = None  # Not used by this engine
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None

    # Store parameters needed for potential re-linearization or plotting
    track_graph_definition_: Optional[nx.Graph] = None
    edge_order_definition_: Optional[List[Tuple[Any, Any]]] = None
    edge_spacing_definition_: Optional[Union[float, Sequence[float]]] = None

    def build(
        self,
        *,
        track_graph: nx.Graph,
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        place_bin_size: float,
    ) -> None:
        self.track_graph_definition_ = track_graph
        self.edge_order_definition_ = edge_order
        self.edge_spacing_definition_ = edge_spacing

        (
            self.place_bin_centers_,  # (n_bins, 1) for linearized position
            self.place_bin_edges_,  # (n_edges, 1) for linearized position
            is_interior,
            self.centers_shape_,  # (n_bins,)
            self.edges_,  # Tuple containing one array: (linearized_edges,)
            self.track_graph_bin_centers_,  # Graph with original 2D/3D 'pos' attributes
        ) = _create_1d_track_grid_data(
            track_graph,
            edge_order,
            edge_spacing,
            place_bin_size,
        )
        # place_bin_centers_ from _create_1d_track_grid_data is (n_bins, 1) with linearized positions.
        # For KDTree, we need the actual spatial coordinates from the graph.
        if self.track_graph_bin_centers_ is not None:
            # Extract 2D/3D positions from the graph for KDTree
            # Node IDs in track_graph_bin_centers_ correspond to bin_ind_flat
            # Ensure place_bin_centers_ for KDTree matches these positions
            # and interior_mask aligns with these nodes.

            # Create a DataFrame from graph nodes to easily get positions
            nodes_df = pd.DataFrame.from_dict(
                dict(self.track_graph_bin_centers_.nodes(data=True)), orient="index"
            )
            if "pos" not in nodes_df.columns:
                raise ValueError(
                    "TrackGraphEngine's track_graph_bin_centers_ nodes must have 'pos' attribute."
                )

            # Sort nodes by 'bin_ind_flat' to ensure alignment
            # 'bin_ind_flat' should be 0 to N-1 for the bins
            nodes_df = nodes_df.sort_values(by="bin_ind_flat")

            # This will be the N-D coordinates of bin centers.
            self.spatial_place_bin_centers_ = np.array(
                nodes_df["pos"].tolist()
            )  # (n_bins, n_dims)

            # The interior_mask from _create_1d_track_grid_data is 1D (n_bins,)
            self.interior_mask_ = (
                np.array(nodes_df["is_track_interior"].tolist(), dtype=bool)
                if "is_track_interior" in nodes_df
                else is_interior
            )

            # Override place_bin_centers_ to be spatial for KDTree and point_to_bin
            # The original linearized place_bin_centers_ is still available implicitly via edges_
            # Or store it separately if Environment needs it directly.
            # For now, the KDTreeMixin will use spatial_place_bin_centers_
            # We need to ensure the _KDTreeMixin refers to this.
            # A bit of a hack: temporarily assign for _build_kdtree, then restore.
            original_pb_centers = self.place_bin_centers_  # Store linearized
            self.place_bin_centers_ = self.spatial_place_bin_centers_
            self._build_kdtree(interior_mask=self.interior_mask_)
            self.place_bin_centers_ = original_pb_centers  # Restore linearized centers

        else:
            raise RuntimeError("track_graph_bin_centers_ was not created.")

    @property
    def is_1d(self) -> bool:
        """Return True if the engine represents a 1-dimensional environment."""
        return True

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the 1D track graph with bin centers."""
        if self.track_graph_bin_centers_ is None or not hasattr(
            self, "spatial_place_bin_centers_"
        ):
            raise RuntimeError(
                "Engine has not been built or spatial_place_bin_centers_ is missing. Call `build` first."
            )

        graph = self.track_graph_bin_centers_

        # Node positions are the original 2D/3D coordinates
        node_positions = {
            node: data["pos"] for node, data in graph.nodes(data=True) if "pos" in data
        }

        if not node_positions:
            if ax is None:
                _, ax = plt.subplots()
            ax.text(
                0.5, 0.5, "No positional data in graph nodes.", ha="center", va="center"
            )
            ax.set_title(f"{self.__class__.__name__}")
            return ax

        # Determine if plot is 2D or 3D based on node positions
        first_pos_val = next(iter(node_positions.values()))
        is_3d_plot = len(first_pos_val) == 3

        if ax is None:
            fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            if is_3d_plot:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        elif is_3d_plot and not hasattr(ax, "plot3D"):  # ax provided but not 3D
            # Cannot easily change projection of existing ax, warn or raise
            raise ValueError("Provided 'ax' is not 3D, but data is 3D.")

        node_colors = None
        if self.interior_mask_ is not None and hasattr(
            self, "spatial_place_bin_centers_"
        ):
            # Colors nodes based on whether they are 'interior' if mask aligns with graph nodes
            # This assumes node IDs in graph are 0 to num_nodes-1, matching interior_mask indices
            # which track_graph_bin_centers should ensure via 'bin_ind_flat'
            node_list = sorted(list(graph.nodes()))  # Ensure order for coloring
            try:
                node_colors = [
                    "blue" if self.interior_mask_[node_id] else "red"
                    for node_id in node_list
                ]
            except IndexError:  # If node IDs don't align with mask
                node_colors = "blue"

        nx.draw_networkx_nodes(
            graph,
            pos=node_positions,
            ax=ax,  # type: ignore
            node_size=kwargs.get("node_size", 20),
            node_color=kwargs.get("node_color", node_colors),
        )
        nx.draw_networkx_edges(
            graph,
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
        elif hasattr(ax, "set_box_aspect"):  # For 3D axes in newer matplotlib
            ax.set_box_aspect([1, 1, 1])  # type: ignore

        # Optionally, plot linearized bin edges if they make sense in this context
        # self.edges_ contains linearized bin edges. Plotting them on the 2D/3D graph
        # is not straightforward unless we map them back or plot them separately.
        # For now, this plot focuses on the spatial graph structure.

        return ax  # type: ignore


if SHAPELY_AVAILABLE:

    class ShapelyGridEngine(_KDTreeMixin):
        """Mask a regular 2D grid by a Shapely Polygon."""

        place_bin_centers_: NDArray[np.float64]
        edges_: Tuple[NDArray[np.float64], ...]
        centers_shape_: Tuple[int, ...]
        track_graph_nd_: Optional[nx.Graph] = None
        track_graph_bin_centers_: Optional[nx.Graph] = None  # Not typically used
        interior_mask_: Optional[NDArray[np.bool_]] = None
        polygon_definition_: Optional[Polygon] = None

        def build(
            self,
            *,
            polygon: Polygon,
            place_bin_size: Union[float, Sequence[float]],
            add_boundary_bins: bool = False,
        ) -> None:
            if not SHAPELY_AVAILABLE:
                raise RuntimeError("ShapelyGridEngine requires the 'shapely' package.")
            self.polygon_definition_ = polygon
            minx, miny, maxx, maxy = polygon.bounds
            pos_range: Sequence[Tuple[float, float]] = [(minx, maxx), (miny, maxy)]

            (
                _edges_tuple,
                _place_bin_edges_flat,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = _create_grid(
                position=None,
                bin_size=place_bin_size,
                position_range=pos_range,
                add_boundary_bins=add_boundary_bins,
            )
            self.edges_ = _edges_tuple

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
            return False

        def plot(
            self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
        ) -> matplotlib.axes.Axes:
            """Plots the 2D grid, Shapely polygon, and interior mask."""
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
    """Build from existing boolean mask + edges."""

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
        return False

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the 2D grid and interior mask."""
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

    def build(
        self,
        *,
        place_bin_size: float,  # Radius of hexagon
        position_range: Tuple[
            Tuple[float, float], Tuple[float, float]
        ],  # (xmin, xmax), (ymin, ymax)
    ) -> None:
        (xmin, xmax), (ymin, ymax) = position_range
        # dx is distance between centers of adjacent hexes horizontally
        # dy is distance between centers of rows of hexes
        # For point-up hexagons:
        dx = (
            place_bin_size * 3 / 2
        )  # Width of bounding box of hex if side = place_bin_size? No, this isn't right.
        # If place_bin_size is side length `s`:
        # width of hex = 2s. Horizontal distance between centers = 1.5s?
        # Let place_bin_size be distance between parallel sides (apothem * 2)
        # Or let place_bin_size be distance between centers (equivalent to side length for regular tiling)
        # Assume place_bin_size (s) = distance between centers.
        # So, s = side length of hexagon.
        s = place_bin_size
        hex_width = 2 * s  # Total width of a hexagon
        hex_height = math.sqrt(3) * s  # Total height of a hexagon

        col_spacing = 1.5 * s  # Horizontal distance between centers in same row
        row_spacing = (
            hex_height / 2
        )  # Vertical distance for staggered rows (sqrt(3)/2 * s)

        centers_list: List[Tuple[float, float]] = []
        row_idx = 0
        current_y = ymin
        while current_y < ymax + row_spacing:  # Iterate through rows
            current_x_offset = (dx / 2) if (row_idx % 2 != 0) else 0  # Stagger odd rows
            current_x = xmin + current_x_offset
            while current_x < xmax + col_spacing:  # Iterate through columns
                # Check if center is within a slightly expanded bounding box to catch edge cases
                if (xmin - s) <= current_x <= (xmax + s) and (
                    ymin - hex_height / 2
                ) <= current_y <= (ymax + hex_height / 2):
                    centers_list.append((current_x, current_y))
                current_x += col_spacing
            current_y += row_spacing
            row_idx += 1

        temp_centers = np.array(centers_list)
        if temp_centers.size > 0:
            # Filter to keep only points whose centers are strictly within the original range
            # (or some reasonable margin)
            valid_mask = (
                (temp_centers[:, 0] >= xmin)
                & (temp_centers[:, 0] <= xmax)
                & (temp_centers[:, 1] >= ymin)
                & (temp_centers[:, 1] <= ymax)
            )
            self.place_bin_centers_ = temp_centers[valid_mask]
        else:
            self.place_bin_centers_ = np.empty((0, 2))

        self.centers_shape_ = (self.place_bin_centers_.shape[0],)
        self.interior_mask_ = np.ones(
            self.centers_shape_, dtype=bool
        )  # All generated centers are interior

        G = nx.Graph()
        if self.place_bin_centers_.shape[0] == 0:
            self.track_graph_bin_centers_ = G
            self._build_kdtree(interior_mask=self.interior_mask_)
            return

        G.add_nodes_from(range(len(self.place_bin_centers_)))
        for i in range(len(self.place_bin_centers_)):
            G.nodes[i]["pos"] = tuple(self.place_bin_centers_[i])
            G.nodes[i]["is_track_interior"] = True
            G.nodes[i]["bin_ind"] = (i,)
            G.nodes[i]["bin_ind_flat"] = i

        if self.place_bin_centers_.shape[0] > 1:
            tree = KDTree(self.place_bin_centers_)
            # Connect to neighbors within s * 1.1 (should capture 6 neighbors for perfect lattice)
            pairs = tree.query_pairs(r=s * 1.1)
            for i, j in pairs:
                dist = np.linalg.norm(
                    self.place_bin_centers_[i] - self.place_bin_centers_[j]
                )
                G.add_edge(i, j, distance=dist)

        self.track_graph_bin_centers_ = G
        self._build_kdtree(interior_mask=self.interior_mask_)

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
    """Adaptive quadtree tiling of 2D space to a maximum depth.
    Bin centers are the centers of the quadtree cells.
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
        return False  # Quadtree is 2D

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the Quadtree graph and optionally cell boundaries."""
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
    """Partitions N-D space via Voronoi tessellation of seed points.
    Bin centers are the centroids of the finite Voronoi regions.
    """

    place_bin_centers_: NDArray[np.float64]  # Centroids of finite Voronoi regions
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]  # (n_finite_regions,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    voronoi_diagram_: Optional[Voronoi] = None
    seed_points_: Optional[NDArray[np.float64]] = None  # Store original seeds

    def build(self, *, seeds: NDArray[np.float64]) -> None:
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
        # Voronoi is typically for 2D+
        if self.seed_points_ is not None and self.seed_points_.shape[1] == 1:
            return True  # If input seeds were 1D
        return False

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the Voronoi graph and optionally the Voronoi diagram."""
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
    """Uses an existing triangular mesh: vertices as bins, edges as adjacency."""

    place_bin_centers_: NDArray[np.float64]  # Mesh vertices
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]  # (n_vertices,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    faces_definition_: Optional[NDArray[np.int_]] = None

    def build(self, *, vertices: NDArray[np.float64], faces: NDArray[np.int_]) -> None:
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
        """Plots the mesh graph and optionally the mesh faces."""
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
    """Converts a 2D boolean image mask into pixel-center bins and a graph.
    Connects 4-neighbors or 8-neighbors.
    """

    place_bin_centers_: NDArray[np.float64]  # (col+0.5, row+0.5)
    edges_: Tuple[NDArray[np.float64], ...] = ()
    centers_shape_: Tuple[int, ...]  # (n_true_pixels,)
    track_graph_nd_: Optional[nx.Graph] = None  # Not typically used
    track_graph_bin_centers_: Optional[nx.Graph] = None
    interior_mask_: Optional[NDArray[np.bool_]] = None
    pixel_to_node_map_: Optional[Dict[Tuple[int, int], int]] = None
    image_mask_definition_: Optional[NDArray[np.bool_]] = None

    def build(self, *, mask: NDArray[np.bool_], connect_diagonal: bool = False) -> None:
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
        return False  # Image mask is 2D

    # Use generic graph plot from Delaunay engine
    _generic_graph_plot = DelaunayGraphEngine._generic_graph_plot

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the graph from the image mask, or the mask itself."""
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
