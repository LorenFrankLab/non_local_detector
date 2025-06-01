from typing import Any, Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.mixins import _GridMixin
from non_local_detector.environment.layout.regular_grid import (
    _create_regular_grid_connectivity_graph,
)
from non_local_detector.environment.layout.utils import get_centers


class MaskedGridLayout(_GridMixin):  # type: ignore
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
        self.bin_centers = np.empty((0, 0), dtype=np.float64)
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

        self._build_params_used = {
            "active_mask": active_mask,
            "grid_edges": grid_edges,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }


class ImageMaskLayout(_GridMixin):
    """
    2D layout derived from a boolean image mask.

    Each `True` pixel in the input `image_mask` corresponds to an active bin
    in the environment. The spatial scale of these pixel-bins is determined
    by `bin_size`. Inherits grid functionalities from `_GridMixin`.
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
        """Initialize an ImageMaskLayout engine."""
        self._layout_type_tag = "ImageMask"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

    def build(
        self,
        *,
        image_mask: NDArray[np.bool_],  # Defines candidate pixels
        bin_size: Union[float, Tuple[float, float]] = 1.0,  # one pixel
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """
        Build the layout from a 2D image mask.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where `True` pixels define active bins.
        bin_size : Union[float, Tuple[float, float]], default=1.0
            The spatial size of each pixel.
            If float: pixels are square (size x size).
            If tuple (width, height): specifies pixel_width and pixel_height.
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active pixel-bins.

        Raises
        ------
        TypeError
            If `image_mask` is not a NumPy array.
        ValueError
            If `image_mask` is not 2D, not boolean, or `bin_size` is invalid,
            or if `image_mask` contains no True values or non-finite values.
        """

        if not isinstance(image_mask, np.ndarray):
            raise TypeError("image_mask must be a numpy array.")
        if image_mask.ndim != 2:
            raise ValueError("image_mask must be a 2D array.")
        if not np.issubdtype(image_mask.dtype, np.bool_):
            raise ValueError("image_mask must be a boolean array.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")
        if not np.any(image_mask):
            raise ValueError("image_mask must contain at least one True value.")
        if not np.all(np.isfinite(image_mask)):
            raise ValueError("image_mask must not contain NaN or Inf values.")

        self._build_params_used = locals().copy()  # Store all passed params
        del self._build_params_used["self"]  # Remove self from the dictionary

        # Determine bin_sizes for x and y (units per pixel)
        bin_size_x: float
        bin_size_y: float
        if isinstance(bin_size, (float, int, np.number)):
            bin_size_x = float(bin_size)
            bin_size_y = float(bin_size)
        elif isinstance(bin_size, (list, tuple, np.ndarray)) and len(bin_size) == 2:
            bin_size_x = float(bin_size[0])  # width of pixel
            bin_size_y = float(bin_size[1])  # height of pixel
        else:
            raise ValueError(
                "bin_size for ImageMaskLayout must be a float or a 2-element sequence (width, height)."
            )

        if bin_size_x <= 0 or bin_size_y <= 0:
            raise ValueError("bin_size components must be positive.")

        n_rows, n_cols = image_mask.shape
        self.grid_shape = (n_rows, n_cols)  # Note: (rows, cols) often (y_dim, x_dim)
        y_edges = np.arange(n_rows + 1) * bin_size_y
        x_edges = np.arange(n_cols + 1) * bin_size_x
        self.grid_edges = (y_edges, x_edges)
        self.dimension_ranges = (
            (x_edges[0], x_edges[-1]),
            (y_edges[0], y_edges[-1]),
        )

        y_centers = (np.arange(n_rows) + 0.5) * bin_size_y
        x_centers = (np.arange(n_cols) + 0.5) * bin_size_x
        xv, yv = np.meshgrid(
            x_centers, y_centers, indexing="xy"
        )  # x is cols, y is rows
        full_grid_bin_centers = np.stack((xv.ravel(), yv.ravel()), axis=1)

        self.active_mask = image_mask
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )
