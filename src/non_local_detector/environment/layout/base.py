from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable

import matplotlib
import networkx as nx
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class LayoutEngine(Protocol):
    """
    Protocol defining the interface for all spatial layout engines.

    A LayoutEngine is responsible for discretizing a continuous N-dimensional
    space into a set of bins or elements and constructing a graph representation
    of their connectivity.

    Attributes
    ----------
    bin_centers : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node.
        Shape is (n_active_bins, n_dims).
    connectivity : Optional[nx.Graph]
        Graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to rows in `bin_centers`.
        **Mandatory Node Attributes**:
            - 'pos': Tuple[float, ...] - N-D coordinates of the active bin center.
            - 'source_grid_flat_index': int - Flat index in the original
              full conceptual grid from which this active bin originated.
            - 'original_grid_nd_index': Tuple[int, ...] - N-D tuple index
              in the original full conceptual grid.
        **Mandatory Edge Attributes**:
            - 'distance': float - Euclidean distance between connected bin centers.
            - 'vector': Tuple[float, ...] - Displacement vector between centers.
            - 'edge_id': int - Unique ID for the edge within this graph.
        **Recommended Edge Attributes**:
            - 'angle_2d': Optional[float] - Angle of displacement for 2D layouts.

    is_1d : bool
        True if the layout represents a primarily 1-dimensional structure
        (e.g., a linearized track), False otherwise.
    dimension_ranges : Optional[Sequence[Tuple[float, float]]]
        The actual min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry.
    grid_edges : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts: A tuple of 1D arrays, where each array
        contains the bin edge positions for one dimension of the *original,
        full grid*. `None` or `()` for non-grid or point-based layouts.
    grid_shape : Optional[Tuple[int, ...]]
        For grid-based layouts: The N-D shape (number of bins in each
        dimension) of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept:
        Typically `(n_active_bins,)`.
    active_mask : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid* (shape matches `grid_shape`).
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers`.
    _layout_type_tag : str
        A string identifier for the type of layout (e.g., "RegularGrid").
        Used for introspection and serialization.
    _build_params_used : Dict[str, Any]
        A dictionary of the parameters used to construct this layout instance.
        Used for introspection and serialization.

    """

    # --- Required Data Attributes ---
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: Sequence[Tuple[float, float]]

    # Attributes primarily for GRID-BASED Layouts
    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    # Internal Attributes for Introspection/Serialization
    _layout_type_tag: str
    _build_params_used: Dict[str, Any]

    # --- Required Methods ---
    def build(self, **kwargs) -> None:
        """
        Construct the layout's geometry, bins, and connectivity graph.

        This method is responsible for populating all the attributes defined
        in the `LayoutEngine` protocol (e.g., `bin_centers`,
        `connectivity`, etc.) based on the provided keyword arguments.
        The specific arguments required will vary depending on the concrete
        implementation of the layout engine.
        """
        ...

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Map continuous N-D points to discrete active bin indices.

        The returned indices range from `0` to `n_active_bins - 1`.
        A value of -1 indicates that the corresponding point did not map
        to any active bin (e.g., it's outside the defined environment).

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map to bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            An array of active bin indices corresponding to the input points.
            -1 for points outside the layout's active bins.
        """
        ...

    @property  # pragma: no cover
    def is_1d(self) -> bool:
        """
        Indicate if the layout structure is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the layout represents a 1D structure (e.g., a linearized
            track), False otherwise.
        """
        ...

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Return the area (2D) or volume (3D+) of each active bin.

        For 1D layouts, this typically returns the length of each bin.
        The measures should correspond to the dimensionality of the space
        the bins occupy.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array where each element is the area/volume/length of the
            corresponding active bin.
        """
        ...

    def plot(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """
        Plot the layout's geometry.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        **kwargs : Any
            Additional keyword arguments for plot customization, specific to
            the layout engine implementation.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.
        """
        ...
