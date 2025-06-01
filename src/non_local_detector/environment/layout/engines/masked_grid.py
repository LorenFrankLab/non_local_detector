from typing import Any, Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.helpers.regular_grid import (
    _create_regular_grid_connectivity_graph,
)
from non_local_detector.environment.layout.helpers.utils import get_centers
from non_local_detector.environment.layout.mixins import _GridMixin


class MaskedGridLayout(_GridMixin):
    """
    Layout from a pre-defined N-D boolean mask and explicit grid edges.

    Allows for precise specification of active bins in an N-dimensional grid
    by providing the complete grid structure (`grid_edges`) and a mask
    (`active_mask`) that designates which cells of that grid are active.
    Inherits grid functionalities from `_GridMixin`.
    """

    bin_centers: NDArray[np.float64]
    connectivity: Optional[nx.Graph] = None
    dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None

    grid_edges: Optional[Tuple[NDArray[np.float64], ...]] = None
    grid_shape: Optional[Tuple[int, ...]] = None
    active_mask: Optional[NDArray[np.bool_]] = None

    _layout_type_tag: str
    _build_params_used: Dict[str, Any]
    bin_size_: Optional[NDArray[np.float64]] = None

    def __init__(self):
        """Initialize a MaskedGridLayout engine."""
        self._layout_type_tag = "MaskedGrid"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.bin_size_ = None

    def build(
        self,
        *,
        active_mask: NDArray[np.bool_],  # User's N-D definition mask
        grid_edges: Tuple[NDArray[np.float64], ...],
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the layout from a mask and grid edges.

        Parameters
        ----------
        active_mask : NDArray[np.bool_]
            N-dimensional boolean array where `True` indicates an active bin.
            Its shape must correspond to the number of bins defined by `grid_edges`
            (i.e., `tuple(len(e)-1 for e in grid_edges)`).
        grid_edges : Tuple[NDArray[np.float64], ...]
            A tuple where each element is a 1D NumPy array of bin edge
            positions for that dimension, defining the full grid structure.
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active grid cells.

        Raises
        ------
        ValueError
            If `active_mask` shape does not match `grid_edges` definition,
            or if `grid_edges` are invalid.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        self.active_mask = active_mask
        self.grid_edges = grid_edges
        self.grid_shape = tuple(len(edge) - 1 for edge in grid_edges)

        if self.active_mask.shape != self.grid_shape:
            raise ValueError(
                f"active_mask shape {self.active_mask.shape} must match "
                f"the shape implied by grid_edges {self.grid_shape}."
            )

        # Create full_grid_bin_centers as (N_total_bins, N_dims) array
        centers_per_dim = [get_centers(edge_dim) for edge_dim in self.grid_edges]
        mesh_centers_list = np.meshgrid(*centers_per_dim, indexing="ij", sparse=False)
        full_grid_bin_centers = np.stack(
            [c.ravel() for c in mesh_centers_list], axis=-1
        )

        self.bin_size_ = np.array(
            [np.diff(edge_dim)[0] for edge_dim in self.grid_edges], dtype=np.float64
        )

        self.dimension_ranges = tuple(
            (edge_dim[0], edge_dim[-1]) for edge_dim in self.grid_edges
        )
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]

        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )
