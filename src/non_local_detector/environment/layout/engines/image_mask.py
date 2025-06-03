from typing import Any, Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.helpers.regular_grid import (
    _create_regular_grid_connectivity_graph,
)
from non_local_detector.environment.layout.mixins import _GridMixin


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
        full_grid_bin_centers = np.stack(
            (
                yv.ravel(),
                xv.ravel(),
            ),
            axis=1,
        )

        self.active_mask = image_mask
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )
