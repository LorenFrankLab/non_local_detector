import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
import matplotlib.axes
import networkx as nx
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.helpers.hexagonal import (
    _create_hex_connectivity_graph,
    _create_hex_grid,
    _infer_active_bins_from_hex_grid,
    _points_to_hex_bin_ind,
)
from non_local_detector.environment.layout.helpers.utils import _generic_graph_plot


class HexagonalLayout:
    """
    2D layout that tiles a rectangular area with a hexagonal lattice.

    Bin centers are the centers of the hexagons. Hexagons are connected to their
    immediate neighbors. Active hexagons can be inferred from data sample
    occupancy.
    """

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = ()
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # Layout Specific
    hexagon_width: Optional[float] = None
    _source_flat_to_active_id_map: Optional[Dict[int, int]] = None

    def __init__(self):
        """Initialize a HexagonalLayout engine."""
        self._layout_type_tag = "Hexagonal"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2))
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = ()
        self.grid_shape = None
        self.active_mask = None
        self.hexagon_width = None
        self.hex_radius_ = None
        self.hex_orientation_ = None
        self.grid_offset_x_ = None
        self.grid_offset_y_ = None
        self._source_flat_to_active_id_map = None

    def build(
        self,
        *,
        hexagon_width: float,
        dimension_ranges: Optional[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = None,
        data_samples: Optional[NDArray[np.float64]] = None,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
    ) -> None:
        """
        Build the hexagonal grid layout.

        Parameters
        ----------
        hexagon_width : float
            The width of the hexagons (distance between parallel sides).
        dimension_ranges : Optional[Tuple[Tuple[float,float], Tuple[float,float]]], optional
            Explicit `[(min_x, max_x), (min_y, max_y)]` for the area to tile.
            If None (default), range is inferred from `data_samples`.
        data_samples : Optional[NDArray[np.float64]], shape (n_samples, 2), optional
            2D data used to infer `dimension_ranges` (if not provided) and/or
            to infer active hexagons (if `infer_active_bins` is True).
            Defaults to None.
        infer_active_bins : bool, default=True
            If True and `data_samples` are provided, infers active hexagons
            based on occupancy. If False, all hexagons within the defined
            area are considered active.
        bin_count_threshold : int, default=0
            If `infer_active_bins` is True, the minimum number of samples a
            hexagon must contain to be considered active.

        Raises
        ------
        ValueError
            If `dimension_ranges` and `data_samples` are both None, or if
            `hexagon_width` is not positive.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        self.hexagon_width = hexagon_width
        (
            full_grid_bin_centers,
            self.grid_shape,
            self.hex_radius_,
            self.hex_orientation_,
            self.grid_offset_x_,
            self.grid_offset_y_,
            self.dimension_ranges,
        ) = _create_hex_grid(
            data_samples=data_samples,
            dimension_range=dimension_ranges,
            hexagon_width=self.hexagon_width,
        )
        if infer_active_bins and data_samples is not None:
            active_bin_original_flat_indices = _infer_active_bins_from_hex_grid(
                data_samples=data_samples,
                centers_shape=self.grid_shape,
                hex_radius=self.hex_radius_,
                min_x=self.grid_offset_x_,
                min_y=self.grid_offset_y_,
                bin_count_threshold=bin_count_threshold,
            )
        else:
            active_bin_original_flat_indices = np.arange(len(full_grid_bin_centers))

        nd_active_mask = np.zeros(self.grid_shape, dtype=bool).ravel()
        nd_active_mask[active_bin_original_flat_indices] = True
        self.active_mask = nd_active_mask.reshape(self.grid_shape)

        self.bin_centers = full_grid_bin_centers[active_bin_original_flat_indices]

        self.connectivity = _create_hex_connectivity_graph(
            active_original_flat_indices=active_bin_original_flat_indices,
            full_grid_bin_centers=full_grid_bin_centers,
            centers_shape=self.grid_shape,
        )

        self._source_flat_to_active_id_map = {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity.nodes(data=True)
        }

    @property
    def is_1d(self) -> bool:
        """Hexagonal layouts are 2-dimensional."""
        return False

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the hexagonal layout.

        Displays active hexagons and their connectivity graph.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments:
            - `show_hexagons` (bool, default=True): Whether to draw hexagon cells.
            - `hexagon_kwargs` (dict): Kwargs for `matplotlib.patches.RegularPolygon`.
            - Other kwargs are passed to `_generic_graph_plot` for the graph.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        ax = _generic_graph_plot(
            ax=ax,
            graph=self.connectivity,
            name=self._layout_type_tag,
            **kwargs,
        )

        if (
            kwargs.get("show_hexagons", True)
            and self.hex_radius_ is not None
            and self.bin_centers is not None
            and self.bin_centers.shape[0] > 0
        ):

            hex_kws = kwargs.get(
                "hexagon_kwargs",
                {
                    "edgecolor": "gray",
                    "facecolor": "none",
                    "alpha": 0.5,
                    "linewidth": 0.5,
                },
            )

            ax.scatter(
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
                s=1,
                label="hexagonal grid",
            )
            patches = [
                RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=self.hex_radius_,
                    orientation=self.hex_orientation_,
                )
                for x, y in self.bin_centers
            ]

            collection = PatchCollection(patches, **hex_kws)
            ax.add_collection(collection)
            ax.plot(
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
                marker="o",
                markersize=1,
                color="blue",
                linestyle="None",
                label="midpoint",
            )

            ax.set_title(f"{self._layout_type_tag} Layout")
            padding = 1.1 * self.hex_radius_
            ax.set_xlim(
                (
                    self.dimension_ranges[0][0] - padding,
                    self.dimension_ranges[0][1] + padding,
                )
            )
            ax.set_ylim(
                (
                    self.dimension_ranges[1][0] - padding,
                    self.dimension_ranges[1][1] + padding,
                )
            )
            ax.set_aspect("equal", adjustable="box")
        return ax

    def point_to_bin_index(self, points):
        """
        Map 2D points to active hexagonal bin indices.

        Uses specialized logic to determine which hexagon each point falls into,
        then maps this to an active bin index.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, 2)
            2D points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Active bin indices (0 to N-1). -1 for points not in an active hexagon.
        """
        if (
            self.grid_offset_x_ is None
            or self.grid_offset_y_ is None
            or self.hex_radius_ is None
            or self.grid_shape is None
            or self._source_flat_to_active_id_map is None
        ):
            # This can happen if build() failed or was incomplete (e.g. no active bins)
            warnings.warn(
                "HexagonalLayout is not fully initialized or has no active bins. "
                "Cannot map points to bin indices.",
                RuntimeWarning,
            )
            return np.full(points.shape[0], -1, dtype=np.int_)

        original_flat_indices = _points_to_hex_bin_ind(
            points=points,
            grid_offset_x=self.grid_offset_x_,
            grid_offset_y=self.grid_offset_y_,
            hex_radius=self.hex_radius_,
            centers_shape=self.grid_shape,
        )
        return np.array(
            [
                self._source_flat_to_active_id_map.get(idx, -1)
                for idx in original_flat_indices
            ],
            dtype=int,
        )

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Calculate the area of each hexagonal bin.

        All active hexagons are assumed to have the same area.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area for each active hexagonal bin.

        Raises
        ------
        RuntimeError
            If `hex_radius_` or `bin_centers` is not populated.
        """
        if self.hex_radius_ is None or self.bin_centers is None:  # pragma: no cover
            raise RuntimeError("Layout not built; hex_radius_ or bin_centers missing.")

        # Area of a regular hexagon: (3 * sqrt(3) / 2) * side_length^2
        # For pointy-top hexagons, side_length (s) is equal to hex_radius_ (R, center to vertex).
        single_hex_area = 3.0 * np.sqrt(3.0) / 2.0 * self.hex_radius_**2.0
        return np.full(self.bin_centers.shape[0], single_hex_area)
