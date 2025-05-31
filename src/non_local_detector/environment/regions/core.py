"""
regions/core.py
===============

Pure data layer for *continuous* regions of interest (ROIs).

* Depends only on the standard library and NumPy.
* Shapely is **optional** and imported lazily—only when you create
  or load a polygon region.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------
# Optional Shapely import
# ---------------------------------------------------------------------
try:
    import shapely.geometry as _shp

    _HAS_SHAPELY = True
    from shapely.geometry import (
        Polygon as PolygonLike,  # type alias for static checkers
    )
except ModuleNotFoundError:  # pragma: no cover  – Shapely not installed
    _HAS_SHAPELY = False

    class _Dummy:  # pylint: disable=too-few-public-methods
        """Fallback stand-in when Shapely is absent."""

    _shp = _Dummy()  # type: ignore
    PolygonLike = Any


# ---------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------
PointCoords = NDArray[np.float64] | Iterable[float]
Kind = Literal["point", "polygon"]

# ---------------------------------------------------------------------
# Region — immutable value object
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """
    Immutable description of a spatial ROI.

    Parameters
    ----------
    name
        Unique region identifier.
    kind
        Either ``"point"`` or ``"polygon"``.
    data
        • point → ``np.ndarray`` with shape ``(n_dims,)``
        • polygon → :class:`shapely.geometry.Polygon` (always 2-D)
    metadata
        Optional, JSON-serialisable attributes (colour, label, …).
    """

    name: str
    kind: Kind
    data: NDArray[np.float64] | PolygonLike
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # filled in post-init
    n_dims: int = field(init=False, repr=False)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def __post_init__(self) -> None:
        # Freeze metadata to prevent accidental mutation through aliasing
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        if self.kind == "point":
            arr = np.asarray(self.data, dtype=float)
            if arr.ndim != 1:
                raise ValueError("Point data must be a 1-D array-like.")
            object.__setattr__(self, "data", arr)
            object.__setattr__(self, "n_dims", arr.shape[0])

        elif self.kind == "polygon":
            if not _HAS_SHAPELY:
                raise RuntimeError("Polygon regions require the 'shapely' package.")
            if not isinstance(self.data, _shp.Polygon):
                raise TypeError("data must be a Shapely Polygon for kind='polygon'.")
            object.__setattr__(self, "n_dims", 2)

        else:  # pragma: no cover
            raise ValueError(f"Unknown kind {self.kind!r}")

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    def __str__(self) -> str:  # noqa: D401 – readable str(region)
        return self.name

    # -----------------------------------------------------------------
    # Serialisation helpers (JSON-friendly)
    # -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        if self.kind == "point":
            geom = self.data.tolist()  # type: ignore[union-attr]
        else:  # polygon
            geom = _shp.mapping(self.data)  # type: ignore[attr-defined]

        return {
            "name": self.name,
            "kind": self.kind,
            "geom": geom,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Region":
        kind: Kind = payload["kind"]  # type: ignore[assignment]
        if kind == "point":
            data = np.asarray(payload["geom"], dtype=float)
        elif kind == "polygon":
            if not _HAS_SHAPELY:
                raise RuntimeError("Loading a polygon Region requires Shapely.")
            data = _shp.shape(payload["geom"])  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unknown kind {kind!r}")
        return cls(
            name=payload["name"],
            kind=kind,
            data=data,
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------
# Regions — mutable mapping
# ---------------------------------------------------------------------


class Regions(MutableMapping[str, Region]):
    """
    A small `dict`-like container mapping *name → Region*.

    Provides the usual mapping API plus a few helpers
    (`add`, `remove`, `list_names`, `buffer`, …).
    """

    # -------------- Mapping interface --------------------------------
    def __init__(self, items: Iterable[Region] | None = None) -> None:
        self._store: dict[str, Region] = {}
        if items is not None:
            for reg in items:
                self[reg.name] = reg  # reuse validation in __setitem__

    def __getitem__(self, key: str) -> Region:
        return self._store[key]

    def __setitem__(self, key: str, value: Region) -> None:
        if key in self._store:
            raise KeyError(f"Region {key!r} already exists — use update() instead.")
        if key != value.name:
            raise ValueError("Key must match Region.name")
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        self._store.pop(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:  # noqa: D401
        inside = ", ".join(f"{n}({r.kind})" for n, r in self._store.items())
        return f"{self.__class__.__name__}({inside})"

    # -------------- Convenience helpers ------------------------------
    def add(
        self,
        name: str,
        *,
        point: PointCoords | None = None,
        polygon: PolygonLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Region:
        """
        Create & insert a new Region. Exactly one of *point* or *polygon*
        must be supplied. Returns the newly created Region.
        """
        if (point is None) == (polygon is None):
            raise ValueError("Specify **one** of 'point' or 'polygon'.")
        if name in self:
            raise KeyError(f"Duplicate region name {name!r}.")

        if point is not None:
            region = Region(name, "point", np.asarray(point), metadata or {})
        else:
            if not _HAS_SHAPELY:
                raise RuntimeError("'polygon' kind requires Shapely.")
            region = Region(name, "polygon", polygon, metadata or {})

        self[name] = region
        return region

    def remove(self, name: str) -> None:
        """Delete region *name*; no error if it is absent."""
        self._store.pop(name, None)

    def list_names(self) -> list[str]:
        """Return region names in insertion order."""
        return list(self._store)

    # ----------- lightweight geometry helper -------------------------
    def area(self, name: str) -> float:
        """Return area for a polygon region; 0.0 for points."""
        region = self[name]
        if region.kind == "polygon":
            if not _HAS_SHAPELY:
                raise RuntimeError("Area computation requires Shapely.")
            return region.data.area  # type: ignore[attr-defined]
        return 0.0

    def region_center(self, region_name: str) -> Optional[NDArray[np.float64]]:
        """
        Calculate the center of a specified named region.

        - For 'point' regions, returns the point itself.
        - For 'polygon' regions, returns the centroid of the polygon.

        Returns
        -------
        Optional[NDArray[np.float64]]
            N-D coordinates of the region's center, or None if the region
            is empty or center cannot be determined.

        Raises
        ------
        KeyError
            If `region_name` is not present in this collection.
        RuntimeError
            If attempting to compute a polygon centroid but Shapely is not installed.
        """
        if region_name not in self._store:
            raise KeyError(f"Region '{region_name}' not found in this collection.")

        region = self._store[region_name]

        if region.kind == "point":
            return np.asarray(region.data, dtype=float)
        elif region.kind == "polygon":
            if not _HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon region queries require 'shapely'.")
            return np.array(region.data.centroid.coords[0], dtype=float)  # type: ignore
        return None  # pragma: no cover

    def buffer(
        self,
        source: str | NDArray[np.float64],
        distance: float,
        new_name: str,
        **meta: Any,
    ) -> Region:
        """
        Create *and return* a buffered polygon region.

        *source* may be an existing region name or a raw 2-D point array.
        """
        if not _HAS_SHAPELY:
            raise RuntimeError("Buffering requires Shapely.")

        # derive geometry in cm space
        if isinstance(source, str):
            src = self[source]
            if src.kind == "polygon":
                geom = src.data
            elif src.kind == "point" and src.n_dims == 2:
                geom = _shp.Point(src.data)  # type: ignore[attr-defined]
            else:
                raise ValueError("Can only buffer 2-D point or polygon regions.")
        else:  # raw coords
            arr = np.asarray(source, dtype=float)
            if arr.shape != (2,):
                raise ValueError("Raw source must be shape (2,) for buffering.")
            geom = _shp.Point(arr)  # type: ignore[attr-defined]

        poly = geom.buffer(distance)
        if not isinstance(poly, _shp.Polygon):  # type: ignore[attr-defined]
            raise ValueError("Buffer produced non-polygon geometry.")

        return self.add(new_name, polygon=poly, metadata=meta)

    # -------------- Serialization helpers ---------------------------
    _FMT = "Regions-v1"

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write collection to disk in a simple, version-tagged schema."""
        payload = {
            "format": self._FMT,
            "regions": [r.to_dict() for r in self._store.values()],
        }
        Path(path).write_text(json.dumps(payload, indent=indent))

    @classmethod
    def from_json(cls, path: str | Path) -> "Regions":
        """Load collection saved by :meth:`to_json`."""
        blob = json.loads(Path(path).read_text())
        if blob.get("format") != cls._FMT:
            warnings.warn(f"Unexpected format tag {blob.get('format')!r}")
        return cls(Region.from_dict(d) for d in blob["regions"])
