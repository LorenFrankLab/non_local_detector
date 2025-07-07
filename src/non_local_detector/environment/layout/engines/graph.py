from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from track_linearization import get_linearized_position as _get_linearized_position
from track_linearization import plot_graph_as_1D

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.helpers.graph import (
    _create_graph_layout_connectivity_graph,
    _find_bin_for_linear_position,
    _get_graph_bins,
    _project_1d_to_2d,
)
from non_local_detector.environment.layout.mixins import _KDTreeMixin


class GraphLayout(_KDTreeMixin):
    """
    Layout defined by a user-provided graph, typically for 1D tracks.

    The graph's nodes (with 'pos' attributes) and a specified edge order
    are used to create a linearized representation of the space, which is
    then binned. Connectivity is derived from this binned structure.
    Uses `_KDTreeMixin` for point mapping and neighbor finding on the
    N-D embeddings of the linearized bin centers.
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
    linear_bin_centers_: Optional[NDArray[np.float64]] = None

    def __init__(self):
        """Initialize a GraphLayout engine."""
        self._layout_type_tag = "Graph"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 0), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.linear_bin_centers_ = None

    def build(
        self,
        *,
        graph_definition: nx.Graph,  # Original user-provided graph
        edge_order: List[Tuple[Any, Any]],
        edge_spacing: Union[float, Sequence[float]],
        bin_size: float,  # Linearized bin size
    ) -> None:
        """
        Build the graph-based (linearized track) layout.

        Parameters
        ----------
        graph_definition : nx.Graph
            The original NetworkX graph. Nodes must have a 'pos' attribute
            (e.g., `(x, y)` coordinates) and edges should ideally have a
            'distance' attribute if not relying on Euclidean distance calculation.
        edge_order : List[Tuple[Any, Any]]
            An ordered sequence of edge tuples (node_id_1, node_id_2) from
            `graph_definition` that defines the ordering of edges in the
            linear space.
        edge_spacing : Union[float, Sequence[float]]
            Spacing (gap) to insert between consecutive edges in `edge_order`
            during linearization. If float, same gap for all. If sequence,
            specifies each gap; length must be `len(edge_order) - 1`.
        bin_size : float
            The desired length of each bin along the linearized space.

        Raises
        ------
        TypeError
            If `graph_definition` is not a NetworkX graph.
        ValueError
            If `edge_order` is empty or `bin_size` is not positive.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        if not isinstance(graph_definition, nx.Graph):
            raise TypeError("graph_definition must be a NetworkX graph.")
        if not edge_order:  # Empty edge_order means no path to linearize
            raise ValueError("edge_order must not be empty.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        (linear_bin_centers, self.grid_edges, self.active_mask, edge_ids) = (
            _get_graph_bins(
                graph=graph_definition,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                bin_size=bin_size,
            )
        )

        self.linear_bin_centers_ = linear_bin_centers[self.active_mask]
        self.bin_centers = _project_1d_to_2d(
            self.linear_bin_centers_,
            graph_definition,
            edge_order,
            edge_spacing,
        )
        self.grid_shape = (len(self.grid_edges[0]),)
        self.connectivity = _create_graph_layout_connectivity_graph(
            graph=graph_definition,
            bin_centers_nd=self.bin_centers,
            linear_bin_centers=self.linear_bin_centers_,
            original_edge_ids=edge_ids,
            edge_order=edge_order,
        )
        self.dimension_ranges = (
            np.min(self.bin_centers[:, 0]),
            np.max(self.bin_centers[:, 0]),
        ), (np.min(self.bin_centers[:, 1]), np.max(self.bin_centers[:, 1]))

        # --- Build KDTree ---
        self._build_kdtree(points_for_tree=self.bin_centers)

    @property
    def is_1d(self) -> bool:
        """Graph layouts are treated as 1-dimensional due to linearization."""
        return True

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the N-D embedding of the graph-based layout.

        Displays the original graph used for definition, the N-D positions of
        the binned track segments (active bin centers), and their connectivity.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments:
            - `figsize` (tuple): Figure size if `ax` is None.
            - `node_kwargs` (dict): Kwargs for plotting original graph nodes.
            - `edge_kwargs` (dict): Kwargs for plotting original graph edges.
            - `bin_node_kwargs` (dict): Kwargs for plotting active bin center nodes.
            - `bin_edge_kwargs` (dict): Kwargs for plotting connectivity graph edges.
            - `show_bin_edges` (bool): Whether to project and plot 1D bin edges in N-D.


        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        # Draw the original graph nodes
        original_node_pos = nx.get_node_attributes(
            self._build_params_used["graph_definition"], "pos"
        )
        nx.draw_networkx_nodes(
            self._build_params_used["graph_definition"],
            original_node_pos,
            ax=ax,
            node_size=300,
            node_color="#1f77b4",
        )
        # Draw the original graph edges
        for node_id1, node_id2 in self._build_params_used["graph_definition"].edges:
            pos = np.stack(
                (
                    original_node_pos[node_id1],
                    original_node_pos[node_id2],
                )
            )
            ax.plot(
                pos[:, 0], pos[:, 1], color="gray", zorder=-1, label="original edges"
            )

        for node_id, pos in original_node_pos.items():
            plt.text(
                pos[0],
                pos[1],
                str(node_id),
                ha="center",
                va="center",
                zorder=10,
            )

        # Draw the bin centers
        bin_centers = nx.get_node_attributes(self.connectivity, "pos")
        nx.draw_networkx_nodes(
            self.connectivity,
            bin_centers,
            ax=ax,
            node_size=30,
            node_color="black",
        )

        # Draw connectivity graph edges
        for node_id1, node_id2 in self.connectivity.edges:
            pos = np.stack((bin_centers[node_id1], bin_centers[node_id2]))
            ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

        grid_line_2d = _project_1d_to_2d(
            self.grid_edges[0],
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        )
        for grid_line in grid_line_2d:
            ax.plot(
                grid_line[0],
                grid_line[1],
                color="gray",
                marker="+",
                alpha=0.8,
                label="bin edges",
            )
        return ax

    def plot_linear_layout(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the 1D linearized representation of the graph layout.

        Uses `track_linearization.plot_graph_as_1D` to display the track
        segments and nodes in their 1D linearized positions. Overlays the
        1D bin edges from `self.grid_edges`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments passed to
            `track_linearization.plot_graph_as_1D` and for customizing
            the appearance of bin edge lines.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the 1D layout is plotted.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 3)))

        plot_graph_as_1D(
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
            ax=ax,
            **kwargs,
        )
        for grid_line in self.grid_edges[0]:
            ax.axvline(grid_line, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{self._layout_type_tag} Layout")
        ax.set_xlabel("Linearized Position")
        ax.set_ylabel("Bin Index")

        return ax

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert N-D points to 1D linearized coordinates along the track.

        Uses `track_linearization.get_linearized_position`.

        Parameters
        ----------
        data_points : NDArray[np.float64], shape (n_points, n_dims)
            N-D points to linearize.

        Returns
        -------
        NDArray[np.float64], shape (n_points,)
            1D linearized coordinates. NaNs may be returned for points
            far from the track.
        """
        return _get_linearized_position(
            data_points,
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        ).linear_position.to_numpy()

    def linear_to_nd(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Map 1D linearized coordinates back to N-D coordinates on the track graph.

        Parameters
        ----------
        linear_coordinates : NDArray[np.float64], shape (n_points,)
            1D linearized coordinates to map.

        Returns
        -------
        NDArray[np.float64], shape (n_points, n_dims)
            N-D coordinates corresponding to the input linear positions.
        """
        return _project_1d_to_2d(
            linear_coordinates,
            self._build_params_used["graph_definition"],
            self._build_params_used["edge_order"],
            self._build_params_used["edge_spacing"],
        )

    def linear_point_to_bin_ind(self, data_points):
        """
        Map 1D linearized positions to active 1D bin indices.

        Parameters
        ----------
        linear_positions : NDArray[np.float64], shape (n_points,)
            1D linearized positions.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Indices of the active 1D bins corresponding to each linear position.
            Returns -1 for positions outside active bins or in gaps.
            Note: These are indices relative to the set of *active* 1D bins,
            not indices into the full `linear_bin_centers_all` array.
        """
        return _find_bin_for_linear_position(
            data_points, bin_edges=self.grid_edges[0], active_mask=self.active_mask
        )

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Return the length of each active 1D bin along the linearized track.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the length of each active linearized bin.

        Raises
        ------
        RuntimeError
            If `grid_edges` or `active_mask` is not populated.
        """
        if self.grid_edges is None or self.active_mask is None:  # pragma: no cover
            raise RuntimeError("Layout not built; grid_edges or active_mask missing.")
        if not self.grid_edges or self.grid_edges[0].size <= 1:  # pragma: no cover
            raise ValueError(
                "grid_edges (1D) are not properly defined for length calculation."
            )

        all_1d_bin_lengths = np.diff(self.grid_edges[0])
        return all_1d_bin_lengths[self.active_mask]
