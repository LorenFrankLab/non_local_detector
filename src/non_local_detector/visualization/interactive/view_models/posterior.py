"""``PosteriorHeatmapModel`` — schema-aware per-row posterior collapse.

Wraps ``analysis.posterior.collapse_posterior_to_position`` with a
strategy chosen via ``select_reduction(state_names, bin_sizes_)``.
Per-row routing means the heatmap output is bit-identical (within
float64 precision) to the SlicePanel's collapsed-posterior fallback
and the dataset-level static-plot algorithm — all three call sites
share one implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.analysis.posterior import (
    PosteriorReduction,
    _non_local_state_ids,
    _spatial_state_ids,
    _validate_rectangular_spatial,
    collapse_posterior_to_position,
    select_reduction,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


class PosteriorHeatmapModel:
    """Per-row posterior → ``(n_visible, n_pos)`` collapse for the heatmap panel.

    Construction picks ``reduction`` via ``select_reduction`` (or the
    user override) and caches the per-detector projection (selected
    state ids, the column mask into ``state_bins``, and ``n_pos``)
    once. ``update_window`` then collapses an entire visible window
    in one vectorized NumPy reduction. ``set_active_run`` rebinds the
    cache when the viewer swaps to a different detector schema.
    """

    def __init__(
        self,
        detector: _DetectorBase,
        reduction: PosteriorReduction | None = None,
        zero_mass_fill: float = np.nan,
    ) -> None:
        self._zero_mass_fill = zero_mass_fill
        self._bind(detector, reduction)

    def _bind(
        self,
        detector: _DetectorBase,
        reduction: PosteriorReduction | None,
    ) -> None:
        """Cache everything that's invariant per-detector for hot-path reuse."""
        self._detector = detector
        self._reduction = reduction or select_reduction(
            detector.state_names, detector.bin_sizes_
        )
        # Validate once + cache n_pos so per-tick collapse skips it.
        _, self._n_pos = _validate_rectangular_spatial(
            detector, "PosteriorHeatmapModel"
        )
        # Selected discrete-state ids for the chosen reduction.
        if self._reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL:
            self._selected_state_ids = _non_local_state_ids(detector)
        else:
            # MARGINAL and CONDITIONAL_ON_SPATIAL both pick spatial states.
            self._selected_state_ids = _spatial_state_ids(detector)
        # Mirror analysis.posterior._conditional_row's validation: the
        # vectorized hot path silently produces zero_mass_fill rows on
        # an empty selection, while the per-row primitive
        # collapse_at routes to raises ValueError. Catch the
        # mis-configuration up-front so both code paths agree.
        if (
            self._reduction is not PosteriorReduction.MARGINAL
            and self._selected_state_ids.size == 0
        ):
            raise ValueError(
                f"PosteriorHeatmapModel({self._reduction!r}) requires at "
                "least one selected state, but the detector has none. "
                "For CONDITIONAL_NON_LOCAL, no state name contains "
                "'Non-Local'. Pick a different reduction (e.g. MARGINAL) "
                "or use a detector whose schema matches the strategy."
            )
        # Boolean mask into state_bins — the per-row primitive uses
        # this; computing it once per swap means no per-tick re-build.
        self._selected_mask = np.isin(
            np.asarray(detector.state_ind_), self._selected_state_ids
        )

    @property
    def reduction(self) -> PosteriorReduction:
        return self._reduction

    @property
    def detector(self) -> _DetectorBase:
        return self._detector

    @property
    def n_pos(self) -> int:
        return self._n_pos

    def set_active_run(
        self,
        detector: _DetectorBase,
        reduction: PosteriorReduction | None = None,
    ) -> None:
        """Rebind to a new detector + auto-detect a new reduction strategy.

        Called by ``ViewerCore`` on M-key swap. Pass ``reduction``
        explicitly to override the auto-detect.
        """
        self._bind(detector, reduction)

    def update_window(self, posterior_window: np.ndarray) -> np.ndarray:
        """Collapse a ``(n_visible, n_state_bins)`` window to ``(n_visible, n_pos)``.

        Vectorized: select selected-state columns out of every row,
        sum across the state axis, divide by per-row mass for the
        ``CONDITIONAL_*`` strategies (``MARGINAL`` skips the divide).
        """
        if posterior_window.ndim != 2:
            raise ValueError(
                "PosteriorHeatmapModel expects a 2D window "
                f"(n_visible, n_state_bins). Got shape "
                f"{posterior_window.shape}."
            )
        return self.collapse_rows(posterior_window)

    def collapse_rows(self, posterior_window: np.ndarray) -> np.ndarray:
        """Vectorized window-level collapse.

        Always returns ``(n_visible, n_pos)``; the ``n_pos`` axis is
        preserved on empty input so downstream renderers can still
        infer the position grid width.

        **Contract**: input is a 2D ``(n_visible, n_state_bins)``
        numpy array; the column axis is integer-positional, not the
        xarray ``state_bins`` MultiIndex. ``state_ind`` (1D, present
        on both eager and zarr-direct data sources) is the column
        labeller. This keeps the collapse path identical regardless
        of whether the underlying data source goes through xarray
        or reads ``zarr.Array[sl, :]`` directly (Phase 5.3 audit).
        """
        n_visible = posterior_window.shape[0]
        if n_visible == 0:
            return np.empty((0, self._n_pos), dtype=np.float64)

        # Cast to float64 so accumulation matches the static-plot
        # inline algorithm — see analysis.posterior._conditional_row.
        selected = (
            posterior_window[:, self._selected_mask]
            .reshape(n_visible, self._selected_state_ids.size, self._n_pos)
            .astype(np.float64)
        )
        column_sum = selected.sum(axis=1)  # (n_visible, n_pos)

        if self._reduction is PosteriorReduction.MARGINAL:
            return column_sum

        # CONDITIONAL_*: divide by per-row mass; rows with zero
        # selected mass take ``zero_mass_fill``.
        mass = np.nansum(column_sum, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = column_sum / mass
        zero_mask = (mass.squeeze(axis=1) == 0) | np.isnan(mass.squeeze(axis=1))
        if zero_mask.any():
            out[zero_mask] = self._zero_mass_fill
        return out

    def collapse_at(
        self, posterior_window: np.ndarray, indices: Sequence[int]
    ) -> np.ndarray:
        """Collapse only the rows at ``indices``.

        Always returns ``(len(indices), n_pos)``; an empty
        ``indices`` preserves the ``n_pos`` axis.
        """
        indices = list(indices)
        if not indices:
            return np.empty((0, self._n_pos), dtype=np.float64)
        # Per-row routing keeps bit-for-bit equivalence with
        # ``collapse_posterior_to_position`` for unit-test parity.
        rows = [
            collapse_posterior_to_position(
                posterior_window[i],
                self._detector,
                self._reduction,
                zero_mass_fill=self._zero_mass_fill,
            )
            for i in indices
        ]
        return np.stack(rows)
