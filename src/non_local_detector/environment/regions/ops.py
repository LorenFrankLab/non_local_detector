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

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import shapely.vectorized as sv
from numpy.typing import NDArray

from ..transforms import SpatialTransform
from .core import Region, Regions


def points_in_any_region(
    pts: Union[Sequence[Sequence[float]], NDArray[np.floating]],
    regions: Regions,
    *,
    transform: Optional[SpatialTransform] = None,
    point_tolerance: float = 1e-8,
) -> NDArray[np.bool_]:
    """
    Determine whether each point lies inside any of the provided Regions.

    Parameters
    ----------
    pts : Union[Sequence[Sequence[float]], NDArray[np.floating]]
        Array-like of shape (n_points, 2). If `transform` is not None, coordinates
        are assumed to be in pixel space; otherwise, they must match the coordinate
        space of each Region.data (Shapely geometry).
    regions : Regions
        A container of Region objects whose `data` attribute is a Shapely geometry
        (e.g., polygon or point). Polygons may be tested with `contains` or `covers`.
    transform : Optional[SpatialTransform], default=None
        If provided, a SpatialTransform with a `.forward()` method that maps
        pixel→world coordinates. The entire (n_points, 2) array is passed through
        `transform.forward` before testing.
    point_tolerance : float, default=1e-8
        Tolerance for comparing query points to point Regions. Two coordinates are
        considered equal if their absolute difference is less than `point_tolerance`.


    Returns
    -------
    mask : NDArray[np.bool_]
        A boolean array of shape (n_points,) where `mask[i]` is True if and only
        if the i-th point is inside at least one Region.

    Notes
    -----
    - Uses Shapely's vectorized predicates (`shapely.vectorized.contains` / `.covers`)
      to test all points against each polygon in one call, which is significantly
      faster than looping over points in Python.
    - Early exits once all points are marked inside at least one region.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n_points = pts.shape[0]

    if transform is not None:
        pts = transform(pts)

    xs = pts[:, 0]
    ys = pts[:, 1]
    mask_any = np.zeros(n_points, dtype=bool)

    for reg in regions.values():
        if reg.kind == "point":
            # Point Region: data is a NumPy array [px, py]
            px, py = reg.data  # type: ignore[attr-defined]
            # Compute mask where both |x - px| < tol and |y - py| < tol
            point_mask = (np.abs(xs - px) <= point_tolerance) & (
                np.abs(ys - py) <= point_tolerance
            )
            mask_any |= point_mask
        else:
            # Polygon or other geometry: data is a Shapely geometry
            geom = reg.data  # type: ignore[attr-defined]
            mask_any |= sv.contains(geom, xs, ys)

        if mask_any.all():
            break

    return mask_any


def regions_containing_points(
    pts: Union[Sequence[Sequence[float]], NDArray[np.floating]],
    regions: Regions,
    *,
    transform: Optional[SpatialTransform] = None,
    region_names: Optional[Sequence[str]] = None,
    return_dataframe: bool = True,
    point_tolerance: float = 1e-8,
) -> Union[List[List[Region]], NDArray[np.bool_], pd.DataFrame]:
    """
    For each point, return either a list of all Regions that contain it, or a DataFrame
    where each column corresponds to a region and values are booleans indicating
    membership.

    Parameters
    ----------
    pts : Union[Sequence[Sequence[float]], NDArray[np.floating]]
        Array-like of shape (n_points, 2). If `transform` is not None, these are
        assumed to be in pixel space; otherwise they must match the coordinate
        space of Region.data.
    regions : Regions
        A container of Region objects (each with a Shapely geometry in `.data`).
    transform : Optional[SpatialTransform], default=None
        If provided, a SpatialTransform with a `.forward()` method that maps pixel→world.
        Applied once to the entire (n_points, 2) array.
    region_names : Optional[Sequence[str]], default=None
        If provided, only consider Regions whose `name` is in this sequence. If None,
        all regions in `regions` are used.
    return_dataframe : bool, default=False
        If True, return a pandas DataFrame of shape (n_points, n_selected_regions),
        where each column is the region name and each value is True/False for membership.
        If False, return a list of lists: each sublist contains Region objects
        that contain the corresponding point.
    point_tolerance : float, default=1e-8
        Tolerance for comparing query points to point Regions. Two coordinates are
        considered equal if their absolute difference is less than `point_tolerance`.

    Returns
    -------
    result : Union[List[List[Region]], pandas.DataFrame]
        - If `return_dataframe` is False: a list of length n_points. Each element is
          a list of Region objects that contain the corresponding point. If no region
          contains the point, the list is empty.
        - If `return_dataframe` is True: a pandas DataFrame with index range(n_points)
          and columns equal to selected region names. Entry (i, col) is True if pts[i]
          is inside that region.

    Notes
    -----
    - If `region_names` contains names not found in `regions`, they are silently ignored.
    - Requires pandas if `return_dataframe` is True (imported within function).
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n_points = pts.shape[0]

    if transform is not None:
        pts = transform(pts)

    xs, ys = pts.T

    # Filter regions by name if requested
    if region_names is not None:
        selected = [reg for reg in regions.values() if reg.name in set(region_names)]
    else:
        selected = list(regions.values())

    if return_dataframe:
        # Initialize a DataFrame with shape (n_points, len(selected))
        df = pd.DataFrame(
            data=np.zeros((n_points, len(selected)), dtype=bool),
            index=np.arange(n_points),
            columns=[reg.name for reg in selected],
        )
        # Fill each column by vectorized test
        for idx, reg in enumerate(selected):
            if reg.kind == "point":
                # Point Region: data is a NumPy array [px, py]
                px, py = reg.data
                # Compute mask where both |x - px| < tol and |y - py| < tol
                point_mask = (np.abs(xs - px) <= point_tolerance) & (
                    np.abs(ys - py) <= point_tolerance
                )
                df.iloc[:, idx] = point_mask
            else:
                df.iloc[:, idx] = sv.contains(reg.data, xs, ys)
        return df

    # Otherwise, build list of lists of Region objects
    output: List[List[Region]] = [[] for _ in range(n_points)]
    for region in selected:
        if region.kind == "point":
            # Point Region: data is a NumPy array [px, py]
            px, py = region.data
            # Compute mask where both |x - px| < tol and |y - py| < tol
            point_mask = (np.abs(xs - px) <= point_tolerance) & (
                np.abs(ys - py) <= point_tolerance
            )
            matching_indices = np.nonzero(point_mask)[0]
        else:
            # Polygon or other geometry: data is a Shapely geometry
            matching_indices = np.nonzero(sv.contains(region.data, xs, ys))[0]
        for i in matching_indices:
            output[i].append(region.name)

    return output
