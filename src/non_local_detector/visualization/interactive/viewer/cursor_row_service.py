"""``CursorRowService`` — single-bin reads from a ``DecoderDataSource``.

The three built-in bin-synced panels (``QtSlicePanel``,
``Qt2DImagePanel``, ``QtProjected2DPanel``) each register a
synchronous row-provider so the cursor stays in sync even when the
async window-load lags. The providers all need the same primitives:

- a single ``(n_state_bins,)`` row out of ``acausal_posterior`` /
  ``log_likelihood`` / ``predictive_posterior``;
- the animal's 1D position at that bin (for the slice panel's
  ``true_position`` overlay);
- the animal's 2D position at that bin (for the 2D image panel's
  XY marker);
- the optional raw 2D position from ``RunBundle.position_2d``
  (for the projected-2D panel's "prefer raw, fall back to graph
  projection" precedence).

This module centralizes those primitives behind one typed surface
so each panel's row-provider closure is a small composition rather
than its own data-source plumbing. Adding a panel that needs a new
single-bin field starts here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.data_source import (
        DecoderDataSource,
    )


StateBinsField = Literal["posterior", "likelihood", "predictive"]


class CursorRowService:
    """Single-bin accessors over a ``DecoderDataSource``.

    Bounds-checks ``t_idx`` once at each entry point so callers can
    rely on ``None`` returns rather than catching ``IndexError``.
    """

    def __init__(self, data_source: DecoderDataSource) -> None:
        self._data_source = data_source

    @property
    def n_time(self) -> int:
        return self._data_source.n_time

    def state_bins_row(
        self, t_idx: int, which: StateBinsField
    ) -> np.ndarray | None:
        """Return a single ``(n_state_bins,)`` row from ``results[which]``.

        ``None`` for out-of-range ``t_idx`` or when the optional
        variable is absent (``likelihood`` and ``predictive`` may not
        be in the bundle). ``"posterior"`` is always present per the
        ``RunBundle`` invariant.
        """
        if t_idx < 0 or t_idx >= self.n_time:
            return None
        var_present = {
            "posterior": True,  # required
            "likelihood": "log_likelihood" in self._data_source.available_outputs,
            "predictive": "predictive_posterior"
            in self._data_source.available_outputs,
        }.get(which, False)
        if not var_present:
            return None
        row = self._data_source.slice_at_index(t_idx, which=which)
        return None if row is None else np.asarray(row)

    def position_1d(self, t_idx: int) -> float | None:
        """1D animal position at ``t_idx`` (scalar). ``None`` for 2D bundles."""
        if t_idx < 0 or t_idx >= self.n_time:
            return None
        position_slice = self._data_source.load_position(slice(t_idx, t_idx + 1))
        if (
            position_slice is None
            or position_slice.ndim != 1
            or position_slice.size == 0
        ):
            return None
        return float(position_slice[0])

    def position_2d(self, t_idx: int) -> np.ndarray | None:
        """2D animal position at ``t_idx``. ``None`` for 1D bundles."""
        if t_idx < 0 or t_idx >= self.n_time:
            return None
        position_slice = self._data_source.load_position(slice(t_idx, t_idx + 1))
        if (
            position_slice is None
            or position_slice.ndim != 2
            or position_slice.shape != (1, 2)
        ):
            return None
        xy = np.asarray(position_slice[0], dtype=float)
        return xy if np.all(np.isfinite(xy)) else None

    def raw_position_2d(self, t_idx: int) -> np.ndarray | None:
        """Optional raw 2D track XY at ``t_idx`` (``RunBundle.position_2d``).

        Used by the projected-2D panel's "prefer raw, fall back to
        graph projection" precedence; ``None`` for bundles that don't
        ship ``position_2d``.
        """
        if t_idx < 0 or t_idx >= self.n_time:
            return None
        position_slice = self._data_source.load_position_2d(slice(t_idx, t_idx + 1))
        if (
            position_slice is None
            or position_slice.ndim != 2
            or position_slice.shape != (1, 2)
        ):
            return None
        xy = np.asarray(position_slice[0], dtype=float)
        return xy if np.all(np.isfinite(xy)) else None
