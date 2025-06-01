from typing import Any, Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.helpers.regular_grid import (
    _create_regular_grid,
    _create_regular_grid_connectivity_graph,
    _infer_active_bins_from_regular_grid,
)
from non_local_detector.environment.layout.helpers.utils import (
    _infer_dimension_ranges_from_samples,
)
from non_local_detector.environment.layout.mixins import _GridMixin


class RegularGridLayout(_GridMixin):
    """
    Axis-aligned rectangular N-D grid layout.

    Discretizes space into a uniform N-dimensional grid. Can infer the
    active portion of this grid based on provided data samples using occupancy
    and morphological operations. Inherits grid-based functionalities from
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

    def __init__(self):
        """Initialize a RegularGridLayout engine."""
        self._layout_type_tag = "RegularGrid"
        self._build_params_used = {}
        # Initialize all protocol attributes to satisfy type checkers, even if None
        self.bin_centers = np.empty((0, 0))
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

    def build(
        self,
        *,
        bin_size: Union[float, Sequence[float]],
        dimension_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        data_samples: Optional[NDArray[np.float64]] = None,
        add_boundary_bins: bool = False,
        infer_active_bins: bool = True,
        dilate: bool = True,
        fill_holes: bool = True,
        close_gaps: bool = True,
        bin_count_threshold: int = 0,
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the regular N-D grid layout.

        Parameters
        ----------
        bin_size : Union[float, Sequence[float]]
            Desired size of bins in each dimension.
        dimension_ranges : Optional[Sequence[Tuple[float, float]]], optional
            Explicit `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]` for the grid.
            If None, range is inferred from `data_samples`.
        data_samples : Optional[NDArray[np.float64]], shape (n_samples, n_dims), optional
            Data used to infer `dimension_ranges` (if not provided) and/or to
            infer active bins (if `infer_active_bins` is True).
        add_boundary_bins : bool, default=False
            If True, adds one bin on each side of the grid, extending the range.
        infer_active_bins : bool, default=True
            If True and `data_samples` are provided, infers active bins based
            on occupancy and morphological operations.
        dilate : bool, default=False
            If `infer_active_bins` is True, dilates the inferred active area.
        fill_holes : bool, default=False
            If `infer_active_bins` is True, fills holes in the inferred active area.
        close_gaps : bool, default=False
            If `infer_active_bins` is True, closes gaps in the inferred active area.
        bin_count_threshold : int, default=0
            If `infer_active_bins` is True, minimum samples in a bin to be
            considered initially occupied.
        connect_diagonal_neighbors : bool, default=True
            If True, connects diagonal neighbors in the connectivity graph.
        """
        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        # --- Determine dimension_ranges if not provided ---
        if dimension_ranges is not None:
            self.dimension_ranges = dimension_ranges
        else:
            # Infer ranges from data_samples
            if data_samples is None:
                raise ValueError(
                    "dimension_ranges must be provided if data_samples is None."
                )

            buffer_for_inference = (
                bin_size / 2.0
                if isinstance(bin_size, (float, int, np.number))
                else bin_size
            )
            # Infer ranges from data_samples
            self.dimension_ranges = _infer_dimension_ranges_from_samples(
                data_samples=data_samples,
                buffer_around_data=buffer_for_inference,
            )

        (
            self.grid_edges,
            full_grid_bin_centers,
            self.grid_shape,
        ) = _create_regular_grid(
            data_samples=data_samples,
            bin_size=bin_size,
            dimension_range=self.dimension_ranges,
            add_boundary_bins=add_boundary_bins,
        )

        if infer_active_bins and data_samples is not None:
            self.active_mask = _infer_active_bins_from_regular_grid(
                data_samples=data_samples,
                edges=self.grid_edges,
                close_gaps=close_gaps,
                fill_holes=fill_holes,
                dilate=dilate,
                bin_count_threshold=bin_count_threshold,
                boundary_exists=add_boundary_bins,
            )
        else:
            # No data_samples or not inferring active bins, use all bins
            self.active_mask = np.ones(self.grid_shape, dtype=bool)

        if not np.any(self.active_mask):
            raise ValueError(
                "No active bins found. Check your data_samples and bin_size."
            )

        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )
