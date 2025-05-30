"""
regions
=======

Immutable *continuous* ROI objects plus optional helpers.

This sub-package offers:

* :class:`Region`   - an immutable, hashable ROI (point or polygon)
* :class:`Regions`  - a small `MutableMapping[str, Region]`
* Lightweight import / export helpers
* A single convenience plot function (Matplotlib & Shapely imported lazily)

"""

from __future__ import annotations

# --- core value objects ---------------------------------
from .core import Region, Regions

# --- thin adapters --------------------------------------
from .io import load_labelme_json, mask_to_region, regions_from_json, regions_to_json
from .plot import plot_regions

__all__ = [
    "Region",
    "Regions",
    "regions_from_json",
    "regions_to_json",
    "load_labelme_json",
    "mask_to_region",
    "plot_regions",
]
