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

import re

import xarray as xr

# Per-axis labels for 2-6 dimensional environments. Beyond six dimensions we
# fall back to numbered names (``dim0_position``, ``dim1_position``, ...).
_DIM_LABELS = ("x", "y", "z", "w", "v", "u")

# Numbered-dimension fallback label (``dim0``, ``dim1``, ...) used beyond six
# dimensions; the trailing ``_position`` is matched separately.
_NUMBERED_LABEL = re.compile(r"dim\d+")


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

    Raises
    ------
    ValueError
        If ``n_position_dims < 1``.
    """
    if n_position_dims < 1:
        raise ValueError(f"n_position_dims must be >= 1; got {n_position_dims}.")
    if n_position_dims == 1:
        return ["position"]
    if n_position_dims <= len(_DIM_LABELS):
        return [f"{_DIM_LABELS[i]}_position" for i in range(n_position_dims)]
    return [f"dim{i}_position" for i in range(n_position_dims)]


def _is_position_dim(name: object) -> bool:
    """Whether ``name`` is one of the canonical position-dim names.

    Matches *exactly* the closed vocabulary produced by
    :func:`get_position_dim_names` — ``position``, the labelled
    ``x``/``y``/``z``/``w``/``v``/``u``\\ ``_position`` names, or the numbered
    ``dim{i}_position`` fallback — so a stray ``_position``-suffixed dim such as
    ``head_position`` or ``linear_position`` is not mistaken for a decoder
    position axis.
    """
    if not isinstance(name, str):
        return False
    if name == "position":
        return True
    if not name.endswith("_position"):
        return False
    label = name[: -len("_position")]
    return label in _DIM_LABELS or _NUMBERED_LABEL.fullmatch(label) is not None


def get_position_dims(data: xr.DataArray) -> list[str]:
    """Position dimensions present on ``data``, in array order.

    Matches the names produced by :func:`get_position_dim_names`: the dim named
    ``position`` (1D), the labelled ``x``/``y``/``z``/``w``/``v``/``u``
    ``_position`` names (2-6D), or the numbered ``dim{i}_position`` fallback
    (>6D). Other dims that merely end in ``_position`` (e.g. ``head_position``)
    are *not* matched.

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
    return [d for d in data.dims if _is_position_dim(d)]
