import warnings
from typing import Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from non_local_detector.environment.layout.helpers.regular_grid import (
    _points_to_regular_grid_bin_ind,
)


# ---------------------------------------------------------------------------
# KD-tree mixin (for point_to_bin_index)
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """
    Mixin providing point-to-bin mapping and neighbor finding using a KD-tree.

    This mixin uses a KD-tree built on bin centers for nearest-neighbor
    searches (`point_to_bin_index`) and a NetworkX graph for connectivity
    to find neighbors (`neighbors`).

    Assumes the inheriting class defines:
    - `self.bin_centers`: NDArray of active bin center coordinates.
    - `self.connectivity`: NetworkX graph of active bins.

    The `_build_kdtree` method must be called by the inheriting layout's
    `build` method after `bin_centers` is finalized.

    Attributes:
    ----------
    _kdtree : Optional[KDTree]
        KDTree for fast nearest-neighbor search.
    _kdtree_nodes_to_bin_indices_map : Optional[NDArray[np.int_]], shape (n_active_bins,)
        Maps KDTree node index to original source_index.
    """

    _kdtree: Optional[KDTree] = None
    _kdtree_nodes_to_bin_indices_map: Optional[NDArray[np.int_]] = None

    def _build_kdtree(
        self,
        points_for_tree: NDArray[np.float64],
        mask_for_points_in_tree: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        """
        Builds the KD-tree from `points_for_tree`.

        If `mask_for_points_in_tree` is provided, it's used to select a subset
        of `points_for_tree` to include in the KD-tree. The
        `_kdtree_nodes_to_bin_indices_map` will then map the KD-tree's
        internal node indices (0 to k-1) back to the original indices within
        `points_for_tree` that were selected by the mask.

        If `points_for_tree` are the layout's final `bin_centers` (which are
        all active), `mask_for_points_in_tree` should be `None` or all `True`.
        In this common case, `_kdtree_nodes_to_bin_indices_map` will effectively
        be `np.arange(len(points_for_tree))`.

        Parameters
        ----------
        points_for_tree : NDArray[np.float64], shape (n_total_points, n_dims)
            The set of points from which to build the KD-tree. Typically, these
            are `self.bin_centers` of the layout.
        mask_for_points_in_tree : Optional[NDArray[np.bool_]], shape (n_total_points,), optional
            A boolean mask indicating which points from `points_for_tree`
            to include in the KD-tree. If None, all points are used.
            Defaults to None.

        Raises
        ------
        ValueError
            If `mask_for_points_in_tree` has an incompatible shape.
        """
        if points_for_tree.ndim != 2:
            raise ValueError(
                "points_for_tree must be a 2D array with shape (n_points, n_dims)."
            )
        if points_for_tree.shape[0] == 0:
            self._kdtree = None
            self._kdtree_nodes_to_bin_indices_map = np.array([], dtype=np.int32)

        final_points_for_kdtree_construction: NDArray[np.float64]

        # This map will store the indices from `points_for_tree` that correspond
        # to the nodes (0 to k-1) in the `self._kdtree` object.
        source_indices_of_kdtree_nodes: NDArray[np.int_]

        if mask_for_points_in_tree is not None:
            if (
                mask_for_points_in_tree.ndim != 1
                or mask_for_points_in_tree.shape[0] != points_for_tree.shape[0]
            ):
                raise ValueError(
                    "mask_for_points_in_tree must be 1D and match the "
                    "number of rows in points_for_tree."
                )
            final_points_for_kdtree_construction = points_for_tree[
                mask_for_points_in_tree
            ]
            source_indices_of_kdtree_nodes = np.flatnonzero(
                mask_for_points_in_tree
            ).astype(np.int32)
        else:
            # No mask provided, use all points from points_for_tree
            final_points_for_kdtree_construction = points_for_tree
            source_indices_of_kdtree_nodes = np.arange(
                points_for_tree.shape[0], dtype=np.int32
            )

        if final_points_for_kdtree_construction.shape[0] > 0:
            try:
                self._kdtree = KDTree(final_points_for_kdtree_construction)
                self._kdtree_nodes_to_bin_indices_map = source_indices_of_kdtree_nodes
            except (
                Exception
            ) as e:  # Catch potential errors from KDTree (e.g. QhullError for certain inputs)
                warnings.warn(
                    f"KDTree construction failed: {e}. point_to_bin_index may not work.",
                    RuntimeWarning,
                )
                self._kdtree = None
                self._kdtree_nodes_to_bin_indices_map = np.array([], dtype=np.int32)
        else:  # No points to build the tree (either initially or after masking)
            self._kdtree = None
            self._kdtree_nodes_to_bin_indices_map = np.array([], dtype=np.int32)

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map N-D points to active bin indices using nearest-neighbor search.

        Finds the nearest active bin center in `self.bin_centers` (on which
        the KD-tree was built) to each query point.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_query_points, n_dims)
            N-D array of query points.

        Returns
        -------
        NDArray[np.int_], shape (n_query_points,)
            Array of active bin indices (0 to `n_active_bins - 1`).
            Returns -1 for all points if the KD-tree is not built or is empty.
        """
        query_points = np.atleast_2d(points)
        n_query_points = query_points.shape[0]

        if (
            self._kdtree is None
            or self._kdtree_nodes_to_bin_indices_map is None
            or self._kdtree_nodes_to_bin_indices_map.size == 0
        ):
            # This means no valid points were used to build the KD-tree.
            return np.full(n_query_points, -1, dtype=np.int32)

        try:
            _, kdtree_internal_indices = self._kdtree.query(query_points)
        except Exception as e:
            # Catch errors if query points have wrong dimension etc.
            warnings.warn(
                f"KDTree query failed: {e}. Returning -1 for all points.",
                RuntimeWarning,
            )
            return np.full(n_query_points, -1, dtype=np.int32)

        # kdtree_internal_indices are indices into the array used to build the KDTree
        # (i.e., final_points_for_kdtree_construction in _build_kdtree).
        # We need to ensure these indices are valid for accessing _kdtree_nodes_to_bin_indices_map.
        # The size of _kdtree.data is len(final_points_for_kdtree_construction).
        max_valid_kdtree_idx = self._kdtree.data.shape[0] - 1  # type: ignore

        # Clip indices to be within the valid range of the KD-tree's internal point list
        # This handles cases where KDTree might return an out-of-bounds index if empty or single point.
        kdtree_internal_indices = np.clip(
            kdtree_internal_indices, 0, max_valid_kdtree_idx
        )

        # Map these internal KD-tree indices back to the original bin indices
        # (which are 0 to N-1, corresponding to rows in self.bin_centers)
        # using the map created in _build_kdtree.
        final_bin_indices = self._kdtree_nodes_to_bin_indices_map[
            kdtree_internal_indices
        ]

        return final_bin_indices.astype(np.int32)


class _GridMixin:
    """
    Mixin for grid-based layout engines (e.g., RegularGrid, ShapelyPolygon).

    Provides common functionality for layouts that are based on an underlying
    N-dimensional grid, such as `point_to_bin_index` using grid definitions,
    default plotting, and `bin_size` for uniform grids.

    Assumes the inheriting class defines grid-specific attributes like
    `grid_edges`, `grid_shape`, `active_mask`, `bin_centers`, and
    `connectivity`.
    """

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None
    bin_centers: Optional[NDArray[np.float64]] = None
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None
    _layout_type_tag: str = "_Grid_Layout"

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map N-D points to active bin indices based on grid structure.

        Uses the grid's `grid_edges`, `grid_shape`, and `active_mask`
        to determine the corresponding active bin for each point.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, n_dims)
            N-D points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Active bin indices (0 to N-1). -1 for points outside active areas.

        Raises
        ------
        RuntimeError
            If grid attributes (`grid_edges`, `grid_shape`) are not set.
        """
        if self.grid_edges is None or self.grid_shape is None:
            raise RuntimeError("Grid layout not built; edges or shape missing.")

        return _points_to_regular_grid_bin_ind(
            points=points,
            grid_edges=self.grid_edges,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
        )

    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        figsize=(7, 7),
        cmap: str = "bone_r",
        alpha: float = 0.7,
        show_connectivity: bool = True,
        node_size: float = 20,
        node_color: str = "blue",
    ) -> matplotlib.axes.Axes:
        """
        Plot the grid-based layout.

        For 2D grids, displays the `active_mask` using `pcolormesh` and
        optionally overlays the `connectivity`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        figsize : Tuple[float, float], default=(7, 7)
            Size of the figure if a new one is created.
        cmap : str, default="bone_r"
            Colormap for the `active_mask` plot.
        alpha : float, default=0.7
            Transparency for the `active_mask` plot.
        draw_connectivity_graph : bool, default=True
            If True, draw the connectivity graph nodes and edges.
        node_size : float, default=20
            Size of nodes if `draw_connectivity_graph` is True.
        node_color : str, default="blue"
            Color of nodes if `draw_connectivity_graph` is True.
        **kwargs : Any
            Additional keyword arguments passed to `ax.pcolormesh` or
            NetworkX drawing functions.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout was plotted.

        Raises
        ------
        RuntimeError
            If layout attributes are not built.
        NotImplementedError
            If attempting to plot a non-2D grid layout with this method.
        """
        if (
            self.bin_centers is None
            or self.grid_edges is None
            or self.active_mask is None
            or self.grid_shape is None
            or self.connectivity is None
        ):
            raise RuntimeError("Layout not built. Call `build` first.")

        is_2d_grid = len(self.grid_shape) == 2 and len(self.grid_edges) == 2

        if is_2d_grid:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.pcolormesh(
                self.grid_edges[0],
                self.grid_edges[1],
                self.active_mask.T,
                cmap=cmap,
                alpha=alpha,
                shading="auto",
            )
            ax.set_xticks(self.grid_edges[0])
            ax.set_yticks(self.grid_edges[1])
            ax.grid(True, ls="-", lw=0.5, c="gray")
            ax.set_aspect("equal")
            ax.set_title(f"{self._layout_type_tag} (2D Grid)")
            ax.set_xlabel("Dimension 0")
            ax.set_ylabel("Dimension 1")
            if self.dimension_ranges:
                ax.set_xlim(self.dimension_ranges[0])
                ax.set_ylim(self.dimension_ranges[1])

            if show_connectivity:
                node_position = nx.get_node_attributes(self.connectivity, "pos")
                nx.draw_networkx_nodes(
                    self.connectivity,
                    node_position,
                    ax=ax,
                    node_size=node_size,
                    node_color=node_color,
                )
                for node_id1, node_id2 in self.connectivity.edges:
                    pos = np.stack((node_position[node_id1], node_position[node_id2]))
                    ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

            return ax
        else:
            raise NotImplementedError(
                "Plotting for non-2D grid layouts is not implemented yet."
            )

    @property
    def is_1d(self) -> bool:
        """
        Indicate if the grid layout is 1-dimensional.

        Standard grid layouts (RegularGrid, etc.) are generally N-D (N>=1).
        This mixin's default assumes not strictly 1D in the sense of a
        linearized track (which GraphLayout would handle).

        Returns
        -------
        bool
            False, as this mixin is for general N-D grids.
        """
        return False

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Calculate area/volume for each active bin, assuming a uniform grid.

        Computes the product of bin side lengths for each dimension from
        `grid_edges`. Assumes all bins in the grid have the same dimensions.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area/volume for each active bin.

        Raises
        ------
        RuntimeError
            If `grid_edges` or `bin_centers` is not populated.
        """
        if self.grid_edges is None or self.bin_centers is None:  # pragma: no cover
            raise RuntimeError("Layout not built; grid_edges or bin_centers missing.")
        if not self.grid_edges or not all(
            len(e) > 1 for e in self.grid_edges
        ):  # pragma: no cover
            raise ValueError(
                "grid_edges are not properly defined for area/volume calculation."
            )

        # Assume uniform bin sizes from the first diff of each dimension's edges
        bin_dimension_sizes = np.array(
            [np.diff(edge_dim)[0] for edge_dim in self.grid_edges]
        )
        single_bin_measure = np.prod(bin_dimension_sizes)

        return np.full(self.bin_centers.shape[0], single_bin_measure)
