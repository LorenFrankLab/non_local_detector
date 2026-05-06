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
    _validate_rectangular_spatial,
    collapse_posterior_to_position,
    select_reduction,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


class PosteriorHeatmapModel:
    """Per-row posterior → ``(n_visible, n_pos)`` collapse for the heatmap panel.

    Construction picks ``reduction`` via ``select_reduction`` (or the
    user override). ``update_window`` returns a ``(n_visible, n_pos)``
    NumPy array; ``set_active_run`` rebinds the strategy when the
    viewer swaps to a different detector schema.
    """

    def __init__(
        self,
        detector: _DetectorBase,
        reduction: PosteriorReduction | None = None,
        zero_mass_fill: float = np.nan,
    ) -> None:
        self._detector = detector
        self._reduction = reduction or select_reduction(
            detector.state_names, detector.bin_sizes_
        )
        self._zero_mass_fill = zero_mass_fill

    @property
    def reduction(self) -> PosteriorReduction:
        return self._reduction

    @property
    def detector(self) -> _DetectorBase:
        return self._detector

    def set_active_run(
        self,
        detector: _DetectorBase,
        reduction: PosteriorReduction | None = None,
    ) -> None:
        """Rebind to a new detector + auto-detect a new reduction strategy.

        Called by ``ViewerCore`` on M-key swap. Pass ``reduction``
        explicitly to override the auto-detect.
        """
        self._detector = detector
        self._reduction = reduction or select_reduction(
            detector.state_names, detector.bin_sizes_
        )

    def update_window(self, posterior_window: np.ndarray) -> np.ndarray:
        """Collapse a ``(n_visible, n_state_bins)`` window to ``(n_visible, n_pos)``.

        Per-row routing through ``collapse_posterior_to_position``
        keeps every reduction path on the same primitive.
        """
        if posterior_window.ndim != 2:
            raise ValueError(
                "PosteriorHeatmapModel expects a 2D window "
                f"(n_visible, n_state_bins). Got shape "
                f"{posterior_window.shape}."
            )
        return self.collapse_rows(posterior_window)

    @property
    def n_pos(self) -> int:
        """Number of position bins this model collapses to."""
        _, n_pos = _validate_rectangular_spatial(
            self._detector, "PosteriorHeatmapModel"
        )
        return n_pos

    def collapse_rows(self, posterior_window: np.ndarray) -> np.ndarray:
        """Vectorized helper used by ``update_window``.

        Public for the SlicePanel posterior-fallback path and the
        Phase 1c property tests. Always returns ``(n_visible, n_pos)``;
        on an empty window the ``n_pos`` axis is preserved so
        downstream renderers can still infer the position grid width.
        """
        n_visible = posterior_window.shape[0]
        if n_visible == 0:
            return np.empty((0, self.n_pos), dtype=np.float64)
        rows = [
            collapse_posterior_to_position(
                posterior_window[i],
                self._detector,
                self._reduction,
                zero_mass_fill=self._zero_mass_fill,
            )
            for i in range(n_visible)
        ]
        return np.stack(rows)

    def collapse_at(
        self, posterior_window: np.ndarray, indices: Sequence[int]
    ) -> np.ndarray:
        """Collapse only the rows at ``indices`` (Phase 4 SlicePanel use).

        Always returns ``(len(indices), n_pos)``; an empty
        ``indices`` preserves the ``n_pos`` axis.
        """
        indices = list(indices)
        if not indices:
            return np.empty((0, self.n_pos), dtype=np.float64)
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
