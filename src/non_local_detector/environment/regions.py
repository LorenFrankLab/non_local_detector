# -------------------------------------------------------------
# Environment Region Registry Patch (Shapely-optional)
# -------------------------------------------------------------
"""Add **symbolic region support** (reward wells, zones, polygons) to an
already-fitted :class:`non_local_detector.environment.Environment`.

The patch monkey-patches new methods onto *any* imported ``Environment``
class at import-time, so you do **not** need to modify the original
source file immediately.  Simply ``import environment_region_registry_patch``
*after* importing the core module, and the following API becomes
available:

High-level API
--------------
``env.add_region(name, *, point=<xyz> | mask=<bool-array> | polygon=<shapely.Polygon>)``
    Register a symbolic region.  One (and only one) of the location
    specs must be supplied.

``env.remove_region(name)``
    Delete a previously added region.

``env.list_regions() -> list[str]``
    Return the list of registered names.

``env.region_mask(name) -> np.ndarray[bool]``
    Boolean mask (same shape as ``centers_shape_``) for the region.

``env.bins_in_region(name) -> np.ndarray[int]``
    Flattened bin indices that belong to the region.

``env.region_center(name) -> np.ndarray``
    Geometric centre of the region (mean of bin centres or exact point).

``env.nearest_region(pos) -> str | None``
    Return the region name whose centre is closest (Euclidean) to *pos*.

Polygon support requires ``shapely``; if it is unavailable the module
still loads, but any polygon registration raises a clear
``RuntimeError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np

from non_local_detector.environment.environment import Environment, check_fitted

try:
    import shapely.geometry as _shp  # noqa: E402

    _HAS_SHAPELY = True
except ModuleNotFoundError:  # polygon support disabled
    _HAS_SHAPELY = False

    class _shp:  # type: ignore[misc]
        """Dummy shim so type references still work."""

        class Polygon:  # noqa: N801
            pass


@dataclass
class RegionInfo:
    """Container for a symbolic region.

    Parameters
    ----------
    name
        User-supplied identifier (must be unique per environment).
    kind
        One of ``{"point", "mask", "polygon"}``.
    data
        Payload whose interpretation depends on *kind*:

        * **``point``**   - *np.ndarray* (shape ``(n_dims,)``)
        * **``mask``**    - Boolean array matching ``centers_shape_``
        * **``polygon``** - :class:`shapely.geometry.Polygon`

    metadata
        Arbitrary key-value store forwarded from :pyfunc:`add_region`.
    """

    name: str
    kind: str  # "point" | "mask" | "polygon"
    data: Any
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"point", "mask", "polygon"}:
            raise ValueError(f"Unknown region kind: {self.kind}")
        if self.kind == "polygon" and not _HAS_SHAPELY:
            raise RuntimeError("Polygon regions require the 'shapely' package.")


# Attach region store if not present
if not hasattr(Environment, "_regions"):
    # str → RegionInfo
    #   (dict of registered regions)
    Environment._regions = {}


def _point_in_polygon(
    points: np.ndarray, polygon: "_shp.Polygon"
) -> np.ndarray:  # noqa: D401
    """Return a boolean array telling which *points* lie inside *polygon*.

    Notes
    -----
    * Requires ``shapely``.
    """

    if not _HAS_SHAPELY:
        raise RuntimeError("Polygon support requested but Shapely is not installed.")
    try:
        return np.array(polygon.contains_points(points), bool)  # type: ignore[attr-defined]
    except AttributeError:
        return np.fromiter((polygon.contains(_shp.Point(xy)) for xy in points), bool)


# ------------------------------------------------------------------
# Region management API - monkey-patched onto Environment
# ------------------------------------------------------------------
@check_fitted
def add_region(
    self: Environment,
    name: str,
    *,
    point: Tuple[float, ...] | None = None,
    mask: np.ndarray | None = None,
    polygon: "_shp.Polygon | list[Tuple[float,float]]" | None = None,
    **metadata,
):
    """Register a region *name* using one of three specifiers.

    Exactly **one** of ``point``, ``mask`` or ``polygon`` must be given.

    Parameters
    ----------
    name
        Unique identifier for the region.
    point
        Physical coordinates (same dimensionality as positions).  Faster
        when you only need a single bin (e.g. reward well).
    mask
        Boolean array of shape ``env.centers_shape_`` - useful when you
        already computed a mask elsewhere.
    polygon
        *Shapely* polygon or list of ``(x, y)`` vertices for 2-D engines
        - lets you delineate irregular zones (start box, wings, etc.).
    metadata
        Additional keyword pairs stored verbatim in :attr:`RegionInfo.metadata`.
    """
    if sum(v is not None for v in (point, mask, polygon)) != 1:
        raise ValueError("Must provide exactly one of point / mask / polygon.")
    if name in self._regions:
        raise ValueError(f"Region '{name}' already exists.")

    # Determine kind + data storage
    if point is not None:
        kind, data = "point", np.asarray(point, float)
    elif mask is not None:
        if mask.shape != self.centers_shape_:
            raise ValueError("Mask shape mismatches environment grid.")
        kind, data = "mask", mask.astype(bool)
    else:  # polygon given
        if not _HAS_SHAPELY:
            raise RuntimeError("Install 'shapely' to use polygon regions.")
        if isinstance(polygon, list):  # coordinates → Polygon
            polygon = _shp.Polygon(polygon)
        if not isinstance(polygon, _shp.Polygon):
            raise TypeError(
                "polygon must be a shapely.geometry.Polygon or list of coords."
            )
        kind, data = "polygon", polygon

    self._regions[name] = RegionInfo(name=name, kind=kind, data=data, metadata=metadata)


@check_fitted
def remove_region(self: Environment, name: str):
    """Remove *name* from the registry (silently ignored if absent)."""
    self._regions.pop(name, None)


def list_regions(self: Environment) -> List[str]:
    """Return a list of registered region names (in insertion order)."""
    return list(self._regions.keys())


@check_fitted
def region_mask(self: Environment, name: str) -> np.ndarray:
    """Boolean occupancy mask for *name*.

    The returned array has the same shape as ``env.centers_shape_``.
    """
    info = self._regions[name]
    if info.kind == "mask":
        return info.data.copy()

    # flat mask initialised false
    flat = np.zeros(np.prod(self.centers_shape_), bool)

    if info.kind == "point":
        idx = self.get_bin_ind(info.data)
        flat[idx] = True
    else:  # polygon
        pts = self.place_bin_centers_[:, :2]  # xy only for test
        inside = _point_in_polygon(pts, info.data)
        flat[inside] = True

    return flat.reshape(self.centers_shape_)


@check_fitted
def bins_in_region(self: Environment, name: str) -> np.ndarray:
    """Return flattened indices of bins inside *name*."""
    return np.flatnonzero(self.region_mask(name))


@check_fitted
def region_center(self: Environment, name: str) -> np.ndarray:
    """Geometric center of the region.

    * For ``point`` regions, this simply returns the point.
    * For ``mask`` / ``polygon``, returns the mean of all bin centers.

    Parameters
    ----------
    name
        Name of the region.
    Returns
    -------
    np.ndarray
        Geometric center of the region.

    """
    info = self._regions[name]
    if info.kind == "point":
        return info.data
    return self.place_bin_centers_[self.bins_in_region(name)].mean(axis=0)


@check_fitted
def nearest_region(self: Environment, position: np.ndarray) -> str | None:
    """Nearest region (Euclidean) to *position*.

    Parameters
    ----------
    position
        Array of shape ``(n_dims,)`` or ``(n_samples, n_dims)``.
    Returns
    -------
    str | None
        Region name with minimal mean distance, or ``None`` if no regions
        are registered.
    """
    pos = np.atleast_2d(position)
    best_name, best_d = None, np.inf
    for name in self.list_regions():
        c = region_center(self, name)
        d = np.linalg.norm(pos - c, axis=1).mean()
        if d < best_d:
            best_name, best_d = name, d
    return best_name


# ------------------------------------------------------------------
# Patch onto Environment class
# ------------------------------------------------------------------
Environment.add_region = add_region  # type: ignore[attr-defined]
Environment.remove_region = remove_region  # type: ignore[attr-defined]
Environment.list_regions = list_regions  # type: ignore[attr-defined]
Environment.region_mask = region_mask  # type: ignore[attr-defined]
Environment.bins_in_region = bins_in_region  # type: ignore[attr-defined]
Environment.region_center = region_center  # type: ignore[attr-defined]
Environment.nearest_region = nearest_region  # type: ignore[attr-defined]
