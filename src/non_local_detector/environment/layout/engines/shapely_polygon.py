from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import Polygon

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.helpers.regular_grid import (
    _create_regular_grid,
    _create_regular_grid_connectivity_graph,
)
from non_local_detector.environment.layout.mixins import _GridMixin


class ShapelyPolygonLayout(_GridMixin):
    """
    2D grid layout masked by a Shapely Polygon.

    Creates a regular grid based on the polygon's bounds and specified
    `bin_size`. Only grid cells whose centers are contained within the
    polygon are considered active. Inherits grid functionalities from
    `_GridMixin`.
    """

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # Layout Specific
    _polygon_definition: Optional[Polygon] = None

    def __init__(self):
        """Initialize a ShapelyPolygonLayout engine."""
        self._layout_type_tag = "ShapelyPolygon"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)  # 2D Layout
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.polygon_definition_ = None

    def build(
        self,
        *,
        polygon: Polygon,
        bin_size: Union[float, Sequence[float]],
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the Shapely Polygon masked grid layout.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The Shapely Polygon object that defines the boundary of the
            active area.
        bin_size : Union[float, Sequence[float]]
            The side length(s) of the grid cells. If float, cells are
            square (or cubic). If sequence (length 2 for 2D), specifies
            (width, height).
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active grid cells in the
            `connectivity`.

        Raises
        ------
        RuntimeError
            If the 'shapely' package is not installed (should be caught by
            SHAPELY_AVAILABLE check at class definition).
        TypeError
            If `polygon` is not a Shapely Polygon.
        """

        if not isinstance(polygon, Polygon):
            raise TypeError("polygon must be a Shapely Polygon object.")

        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        self.polygon_definition_ = polygon
        minx, miny, maxx, maxy = polygon.bounds
        self.dimension_ranges = [(minx, maxx), (miny, maxy)]

        (
            self.grid_edges,
            full_grid_bin_centers,
            self.grid_shape,
        ) = _create_regular_grid(
            data_samples=None,
            bin_size=bin_size,
            dimension_range=self.dimension_ranges,
            add_boundary_bins=False,
        )

        # 1. Intrinsic mask from Shapely
        pts_to_check = (
            full_grid_bin_centers[:, :2]
            if full_grid_bin_centers.shape[0] > 0
            else np.empty((0, 2))
        )
        shapely_mask_flat = (
            shapely.contains(polygon, shapely.points(pts_to_check))
            if pts_to_check.shape[0] > 0
            else np.array([], dtype=bool)
        )
        self.active_mask = shapely_mask_flat.reshape(self.grid_shape)

        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )

    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        figsize=(7, 7),
        cmap: str = "bone_r",
        alpha: float = 0.7,
        draw_connectivity_graph: bool = True,
        node_size: float = 20,
        node_color: str = "blue",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot the ShapelyPolygon layout.

        Displays the active grid cells and overlays the defining polygon.
        Inherits base grid plotting from `_GridMixin.plot` and adds
        polygon visualization.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        figsize : Tuple[float, float], default=(7, 7)
            Size of the figure if `ax` is None.
        **kwargs : Any
            Additional keyword arguments passed to `_GridMixin.plot()`
            (e.g., `cmap`, `alpha` for the grid) and for polygon plotting
            (e.g., `polygon_kwargs` which is a dict for `ax.fill`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
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

            if draw_connectivity_graph:
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

            # Plot polygon
            poly_patch_kwargs = kwargs.get(
                "polygon_kwargs", {"alpha": 0.3, "fc": "gray", "ec": "black"}
            )
            if hasattr(self.polygon_definition_, "geoms"):  # MultiPolygon
                for geom in self.polygon_definition_.geoms:
                    if hasattr(geom, "exterior"):
                        x, y = geom.exterior.xy
                        ax.fill(x, y, **poly_patch_kwargs)
            elif hasattr(self.polygon_definition_, "exterior"):  # Polygon
                x, y = self.polygon_definition_.exterior.xy
                ax.fill(x, y, **poly_patch_kwargs)

            return ax
        else:
            raise NotImplementedError(
                "Plotting for non-2D grid layouts is not implemented yet."
            )
