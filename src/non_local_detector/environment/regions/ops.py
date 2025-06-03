"""
ops.py

Operations for querying point-in-region membership using Shapely and NumPy.

This module provides fast, vectorized functions to determine whether one or more
2D points lie inside polygonal (or point) Regions. It leverages Shapely's
vectorized predicates for bulk point-in-polygon tests and returns either a boolean
mask of “inside any region” or, for each point, the list of Regions that contain it.
Additionally, users can request a DataFrame representation where each column is a
region and each row corresponds to a point, indicating membership as booleans. Users
can also limit the query to a subset of regions by name.

The key functions are:

- points_in_any_region:  Given an (n, 2) array of points, returns an (n,) boolean
  array indicating whether each point is inside at least one Region.
- regions_containing_points:  Given an (n, 2) array of points, either returns, for each
  point, the list of Region objects that contain it, or a pandas DataFrame of shape
  (n, m) where m is the number of selected regions and values are booleans.

Both functions accept an optional `transform` that maps pixel-space coordinates to
the Regions' coordinate space via a SpatialTransform with a `.forward()` method.

Examples
--------
>>> from pathlib import Path
>>> import numpy as np
>>> from .io import load_labelme_json
>>> from ..transforms import SpatialTransform
>>> from ops import points_in_any_region, regions_containing_points
>>>
>>> transform = SpatialTransform(...)  # pixel→world transform
>>> rois = load_labelme_json(Path("annotations.json"), transform=transform)
>>> pts_pixel = np.random.rand(1000, 2) * 512
>>> mask = points_in_any_region(pts_pixel, rois, transform=transform)
>>> matches = regions_containing_points(
...     pts_pixel,
...     rois,
...     transform=transform,
...     include_boundary=True,
...     region_names=["regionA", "regionB"],
...     return_dataframe=True
... )
"""

import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from shapely import Polygon, points
from shapely.geometry import Polygon

from ..transforms import SpatialTransform
from .core import Region, Regions


def _prepare_points(
    pts: Union[Sequence[Sequence[float]], NDArray[np.float64]],
    transform: Optional[SpatialTransform] = None,
) -> NDArray[np.float64]:
    """
    Convert input points to a NumPy array and optionally transform them.

    Parameters
    ----------
    pts : Union[Sequence[Sequence[float]], NDArray[np.float64]], shape (n_points, 2)
    transform : Optional[SpatialTransform], default=None
        If provided, a callable that maps input coordinates to the
        Regions' coordinate space.

    Returns
    -------
    NDArray[np.float64]
        Processed points as an (N, 2) NumPy array of float64.

    Raises
    ------
    ValueError
        If points cannot be converted to shape (N, 2).
    """
    try:
        processed_pts = np.asarray(pts, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Could not convert points to NumPy array: {e}") from e

    if processed_pts.ndim == 1:  # Handle single point case [x, y]
        if processed_pts.shape[0] == 2:
            processed_pts = processed_pts.reshape(1, 2)
        else:
            raise ValueError(
                f"Single point must have 2 coordinates, got shape {processed_pts.shape}"
            )
    elif processed_pts.ndim != 2 or processed_pts.shape[1] != 2:
        raise ValueError(
            f"Points array must be of shape (N, 2), got {processed_pts.shape}"
        )

    if processed_pts.size == 0:  # Handle empty array of points (0, 2)
        return processed_pts  # Return as is, (0,2) array

    if transform is not None:
        processed_pts = transform(processed_pts)  # Assuming transform is callable
    return processed_pts


def _get_points_in_single_region_mask(
    transformed_pts: NDArray[np.float64],
    region: Region,
    point_tolerance: float,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """
    Determine which points lie within a single Region.

    Parameters
    ----------
    transformed_pts : NDArray[np.float64], shape (n_points, 2)
        Points already in the Region's coordinate space.
    region : Region
        The Region object to test against.
    point_tolerance : float
        Tolerance for comparing query points to point Regions.
    include_boundary : bool, default=True
        If True, points on the boundary of the polygon are considered inside.

    Returns
    -------
    in_region_mask : NDArray[np.bool_], shape (n_points,)
        A boolean array of shape (N,) where True indicates the point is inside.
    """
    if transformed_pts.shape[0] == 0:
        return np.array([], dtype=bool)

    xs = transformed_pts[:, 0]
    ys = transformed_pts[:, 1]

    if region.kind == "point":
        if not isinstance(region.data, np.ndarray):
            # This should ideally not happen if Region construction is correct
            raise TypeError(
                f"Region '{region.name}' of kind 'point' has data of type {type(region.data)}, expected np.ndarray."
            )
        if region.data.shape[0] != 2:  # Assuming 2D points for this module
            warnings.warn(
                f"Region '{region.name}' is a point of dimension {region.data.shape[0]}, but 2D comparison "
                "is being performed. This may lead to unexpected results if not 2D."
            )
            # Fallback or specific handling for non-2D points might be needed,
            # for now, proceed if it's (2,) or raise error
            if region.data.ndim != 1 or region.data.shape[0] != 2:
                raise ValueError(
                    f"Point region '{region.name}' data must be a 1D array of 2 coordinates for 2D comparison."
                )

        px, py = region.data[0], region.data[1]
        return (np.abs(xs - px) <= point_tolerance) & (
            np.abs(ys - py) <= point_tolerance
        )
    elif region.kind == "polygon":
        # Region.data for "polygon" is shapely.geometry.Polygon
        if not isinstance(region.data, Polygon):
            # This should ideally not happen
            raise TypeError(
                f"Region '{region.name}' of kind 'polygon' has data of type {type(region.data)}, expected shapely.geometry.Polygon."
            )
        point_geometries = points(
            transformed_pts
        )  # Vectorized creation of Point objects

        if include_boundary:
            return shapely.covers(region.data, point_geometries)
        else:
            return shapely.contains(region.data, point_geometries)
    else:  # Should not be reached if Region.kind is properly constrained by Literal type
        warnings.warn(
            f"Region '{region.name}' has unknown kind '{region.kind}'. Skipping."
        )
        return np.zeros(transformed_pts.shape[0], dtype=bool)


def points_in_any_region(
    pts: Union[Sequence[Sequence[float]], NDArray[np.float64]],
    regions: Regions,
    *,
    transform: Optional[SpatialTransform] = None,
    point_tolerance: float = 1e-8,
) -> NDArray[np.bool_]:
    """
    Determine whether each point lies inside any of the provided Regions.

    Parameters
    ----------
    pts : Union[Sequence[Sequence[float]], NDArray[np.float64]], shape (n_points, 2)
        Array-like of points. If `transform` is not None, coordinates
        are assumed to be in the input space of the transform; otherwise,
        they must match the coordinate space of each Region.data.
    regions : Regions
        A container of Region objects.
    transform : Optional[SpatialTransform], default=None
        If provided, a callable that maps input coordinates to
        the Regions' coordinate space.
    point_tolerance : float, default=1e-8
        Tolerance for comparing query points to point Regions.

    Returns
    -------
    mask : NDArray[np.bool_], shape (n_points,)
        A boolean array where `mask[i]` is True if the i-th point
        is inside at least one Region.
    """
    transformed_pts = _prepare_points(pts, transform)
    n_points = transformed_pts.shape[0]

    if n_points == 0:
        return np.array([], dtype=bool)
    if not regions:  # No regions to check against
        return np.zeros(n_points, dtype=bool)

    overall_mask = np.zeros(n_points, dtype=bool)

    for region in regions.values():
        region_mask = _get_points_in_single_region_mask(
            transformed_pts, region, point_tolerance
        )
        overall_mask |= region_mask
        if overall_mask.all():  # Early exit if all points are already covered
            break
    return overall_mask


def regions_containing_points(
    pts: Union[Sequence[Sequence[float]], NDArray[np.float64]],
    regions: Regions,
    *,
    transform: Optional[SpatialTransform] = None,
    region_names: Optional[Sequence[str]] = None,
    return_dataframe: bool = True,
    point_tolerance: float = 1e-8,
) -> Union[List[List[Region]], pd.DataFrame]:
    """
    For each point, identify all Regions that contain it.

    Parameters
    ----------
    pts : Union[Sequence[Sequence[float]], NDArray[np.float64]], shape (n_points, 2)
        Array-like of points. If `transform` is not None, these are
        assumed to be in the input space of the transform; otherwise,
        they must match the coordinate space of Region.data.
    regions : Regions
        A container of Region objects.
    transform : Optional[SpatialTransform], default=None
        If provided, a callable that maps input coordinates to
        the Regions' coordinate space.
    region_names : Optional[Sequence[str]], default=None
        If provided, only consider Regions whose `name` is in this sequence.
        If None, all regions in `regions` are used.
    return_dataframe : bool, default=True
        If True, return a pandas DataFrame of shape (n_points, n_selected_regions),
        where each column is the region name and each value is True/False.
        If False, return a list of lists: each sublist contains Region objects
        that contain the corresponding point.
    point_tolerance : float, default=1e-8
        Tolerance for comparing query points to point Regions.

    Returns
    -------
    Union[List[List[Region]], pd.DataFrame], length n_points
        - If `return_dataframe` is False: A list of length n_points. Each element is
          a list of :class:`Region` objects that contain the corresponding point.
        - If `return_dataframe` is True: A pandas DataFrame with index
          range(n_points) and columns equal to selected region names.
          Entry (i, col) is True if pts[i] is inside that region.

    Notes
    -----
    - If `region_names` contains names not found in `regions`, they are silently ignored.
    """
    transformed_pts = _prepare_points(pts, transform)
    n_points = transformed_pts.shape[0]

    if n_points == 0:  # Handle case with no input points
        if return_dataframe:
            return pd.DataFrame(
                columns=(
                    region_names
                    if region_names is not None
                    else [r.name for r in regions.values()]
                )
            )
        else:
            return []

    # Filter regions by name if requested
    selected_regions: List[Region]
    if region_names is not None:
        # Maintain order of region_names if possible, and ensure uniqueness
        name_set = set(region_names)
        selected_regions = [
            regions[name]
            for name in region_names
            if name in regions and name in name_set
        ]
    else:
        selected_regions = list(regions.values())

    if not selected_regions and return_dataframe:  # No regions to form columns
        return pd.DataFrame(index=np.arange(n_points))

    if return_dataframe:
        # Initialize a DataFrame with shape (n_points, len(selected_regions))
        # Using list comprehension for column names ensures they match selected_regions order
        df_columns = [reg.name for reg in selected_regions]
        df = pd.DataFrame(
            index=np.arange(n_points),
            columns=df_columns,
            dtype=bool,
        )
        for reg in selected_regions:
            df[reg.name] = _get_points_in_single_region_mask(
                transformed_pts, reg, point_tolerance
            )
        return df
    else:
        # Build list of lists of Region objects
        output: List[List[Region]] = [[] for _ in range(n_points)]
        for region in selected_regions:
            point_mask = _get_points_in_single_region_mask(
                transformed_pts, region, point_tolerance
            )
            # Iterate through points that are in the current region
            for i in np.flatnonzero(point_mask):
                output[i].append(region.name)
        return output
