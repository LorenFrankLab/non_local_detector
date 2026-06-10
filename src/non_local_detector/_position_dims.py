"""Canonical position-dimension naming for decoder posteriors.

The decoders lay out their posteriors with a ``state_bins`` MultiIndex whose
position levels are named by :func:`get_position_dim_names`. Consumers that need
to marginalize over space detect those levels with :func:`get_position_dims`.
Centralizing the convention here keeps the constructors (which build the names)
and the consumers (which detect them) from drifting apart.

This module deliberately depends only on ``xarray`` so it can be imported from
both ``models`` and ``model_checking`` without pulling in heavy dependencies or
creating an import cycle.
"""

from __future__ import annotations

import xarray as xr

# Per-axis labels for 2-6 dimensional environments. Beyond six dimensions we
# fall back to numbered names (``dim0_position``, ``dim1_position``, ...).
_DIM_LABELS = ("x", "y", "z", "w", "v", "u")


def get_position_dim_names(n_position_dims: int) -> list[str]:
    """Canonical position-dimension names for an ``n_position_dims``-D environment.

    Parameters
    ----------
    n_position_dims : int
        Number of spatial dimensions (>= 1).

    Returns
    -------
    list of str
        ``["position"]`` for 1D; ``["x_position", "y_position", ...]`` using the
        labels x/y/z/w/v/u for 2-6D; ``["dim0_position", "dim1_position", ...]``
        for more than six dimensions.
    """
    if n_position_dims == 1:
        return ["position"]
    if n_position_dims <= len(_DIM_LABELS):
        return [f"{_DIM_LABELS[i]}_position" for i in range(n_position_dims)]
    return [f"dim{i}_position" for i in range(n_position_dims)]


def get_position_dims(data: xr.DataArray) -> list[str]:
    """Position dimensions present on ``data``, in array order.

    Matches the names produced by :func:`get_position_dim_names`: the dim named
    ``position`` (1D) plus any dim whose name ends in ``_position`` (2D+).

    Parameters
    ----------
    data : xarray.DataArray
        A posterior (or any array) whose position dimensions follow the
        canonical naming.

    Returns
    -------
    list of str
        The matching dimension names, in the order they appear on ``data``.
    """
    return [d for d in data.dims if d == "position" or d.endswith("_position")]
