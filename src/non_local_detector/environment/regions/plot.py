"""
regions/plot.py
===============

Lightweight helpers for visualising continuous ROIs that live in a
:class:`regions.core.Regions` collection.

You only pay the import cost of *matplotlib* and *shapely* the moment
you call :func:`plot_regions`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from numpy.typing import NDArray

from ..transforms import SpatialTransform
from .core import Region, Regions


# ---------------------------------------------------------------------
# public helper
# ---------------------------------------------------------------------
def plot_regions(
    regions: Regions,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    region_names: Sequence[str] | None = None,
    default_kwargs: Mapping[str, Any] | None = None,
    world_to_pixel: SpatialTransform | None = None,
    add_legend: bool = True,
    **per_region_kwargs: Mapping[str, Any],
) -> None:
    """
    Draw a subset (or all) regions onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination Axes.
    regions : Regions
        Collection to draw.
    region_names : optional list or tuple
        If given, plot only these names; otherwise plot *all*.
    default_kwargs : dict, optional
        Plot kwargs applied to every region (unless overridden).
    world_to_pixel : SpatialTransform, optional
        If supplied, coordinates are mapped **through** this transform
        *before* plotting - handy for overlaying cm-space polygons on
        pixel-space video frames.
    **per_region_kwargs
        Per-region overrides::

            plot_regions(ax, regs,
                         Stem={'alpha':.1},
                         Reward={'edgecolor':'red'})

    Notes
    -----
    * Points → `ax.scatter`
    * Polygons → `matplotlib.patches.PathPatch`
    * Legend labels default to the region name.
    """

    try:
        import shapely.geometry as _shp
    except ModuleNotFoundError:  # polygon plotting will warn later if needed
        _shp = None

    if region_names is None:
        region_names = tuple(regions.keys())

    if not region_names:
        return  # nothing to draw

    if ax is None:
        ax = plt.gca()

    for name in region_names:
        if name not in regions:
            plt.warning(f"plot_regions: '{name}' not in collection; skipping.")
            continue

        reg: Region = regions[name]

        # base → per-region kw → metadata → kwargs passed in call
        opts: dict[str, Any] = dict(default_kwargs or {})
        opts.update(reg.metadata.get("plot_kwargs", {}))
        opts.update(per_region_kwargs.get(name, {}))

        label = opts.pop("label", name)  # legend label
        alpha = opts.pop("alpha", 0.5)

        # optional coordinate transform
        def _map(pts: NDArray[np.float64]) -> NDArray[np.float64]:
            return world_to_pixel(pts) if world_to_pixel else pts

        # ---- draw according to kind ---------------------------------
        if reg.kind == "point":
            xy = _map(np.asarray(reg.data, dtype=float))
            ax.scatter(
                xy[0],
                xy[1],
                marker=opts.pop("marker", "x"),
                s=opts.pop("s", 100),
                alpha=alpha,
                label=label,
                **opts,
            )

        elif reg.kind == "polygon":
            if _shp is None:
                plt.warning(f"Can't draw polygon '{name}': shapely not installed.")
                continue

            poly = reg.data  # already shapely.Polygon

            # exterior + (optional) holes  → Path
            def _ring_to_path(r):
                pts = _map(np.asarray(r.coords)[:, :2])
                return MplPath(pts)

            path = MplPath.make_compound_path(
                _ring_to_path(poly.exterior),
                *[_ring_to_path(i) for i in poly.interiors],
            )

            patch = PathPatch(
                path,
                label=label,
                facecolor=opts.pop("facecolor", opts.pop("color", None)),
                alpha=alpha,
                **opts,
            )
            ax.add_patch(patch)
        else:
            plt.warning(f"Unknown region kind '{reg.kind}' for '{name}'; skipping.")

    # add a legend if *any* labels were produced
    handles, labels = ax.get_legend_handles_labels()
    if handles and add_legend:
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.autoscale(enable=True, axis="both", tight=True)
