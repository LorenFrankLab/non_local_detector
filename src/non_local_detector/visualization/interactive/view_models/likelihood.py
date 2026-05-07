"""``LikelihoodHeatmapModel`` — schema-aware per-row log-likelihood collapse.

Wraps ``analysis.posterior.collapse_log_likelihood_to_position`` over a
window of log-likelihood rows. Output ``(n_visible, n_pos)`` arrays
are peak-normalized per row by the underlying helper, so the heatmap
displays "joint likelihood across spatial states at position x" — see
the docstring for the divergence vs the posterior heatmap on
non-rectangular detectors (e.g. NL with ``local_position_std=1.0``
where the likelihood includes ``Local`` while the posterior heatmap
under ``CONDITIONAL_NON_LOCAL`` excludes it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.analysis.posterior import (
    _validate_rectangular_spatial,
    collapse_log_likelihood_to_position,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


class LikelihoodHeatmapModel:
    """Per-row ``log_likelihood`` → ``(n_visible, n_pos)`` collapse.

    Construction caches the spatial-state ids + ``n_pos`` once;
    ``set_active_run`` rebinds them on M-key swap.
    """

    def __init__(self, detector: _DetectorBase) -> None:
        self._bind(detector)

    def _bind(self, detector: _DetectorBase) -> None:
        self._detector = detector
        # Validate once + cache n_pos.
        _, self._n_pos = _validate_rectangular_spatial(
            detector, "LikelihoodHeatmapModel"
        )

    @property
    def detector(self) -> _DetectorBase:
        return self._detector

    @property
    def n_pos(self) -> int:
        return self._n_pos

    def set_active_run(self, detector: _DetectorBase) -> None:
        """Rebind to a new detector schema (M-key swap)."""
        self._bind(detector)

    def update_window(self, log_lik_window: np.ndarray) -> np.ndarray:
        """Collapse a ``(n_visible, n_state_bins)`` log-likelihood window."""
        if log_lik_window.ndim != 2:
            raise ValueError(
                "LikelihoodHeatmapModel expects a 2D window "
                f"(n_visible, n_state_bins). Got shape "
                f"{log_lik_window.shape}."
            )
        n_visible = log_lik_window.shape[0]
        if n_visible == 0:
            return np.empty((0, self._n_pos), dtype=np.float64)
        rows = [
            collapse_log_likelihood_to_position(log_lik_window[i], self._detector)
            for i in range(n_visible)
        ]
        return np.stack(rows)
