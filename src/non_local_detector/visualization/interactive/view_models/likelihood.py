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

from non_local_detector.analysis.posterior import _validate_rectangular_spatial

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
        # Validate + cache the rectangular-spatial layout. The
        # vectorised collapse needs a boolean mask selecting the
        # spatial-state columns out of ``state_bins`` and the count
        # of spatial states for the reshape. Building once here
        # avoids re-deriving it for every window load.
        spatial_state_ids, n_pos = _validate_rectangular_spatial(
            detector, "LikelihoodHeatmapModel"
        )
        self._n_pos = n_pos
        state_ind = np.asarray(detector.state_ind_)
        self._spatial_state_ids = np.asarray(spatial_state_ids)
        self._selected_mask = np.isin(state_ind, self._spatial_state_ids)
        self._n_spatial_states = int(self._spatial_state_ids.size)

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
        """Collapse a ``(n_visible, n_state_bins)`` log-likelihood window.

        Vectorised over the time axis: select the spatial-state
        columns, replace non-finite entries with ``-inf``, max-subtract
        per row, exponentiate, sum across the spatial-state axis, then
        peak-normalise per row. Equivalent to calling
        ``collapse_log_likelihood_to_position`` on every row but
        without the Python-level loop — at ~1.5k rows per window
        the loop dominated cursor-update latency.
        """
        if log_lik_window.ndim != 2:
            raise ValueError(
                "LikelihoodHeatmapModel expects a 2D window "
                f"(n_visible, n_state_bins). Got shape "
                f"{log_lik_window.shape}."
            )
        n_visible = log_lik_window.shape[0]
        if n_visible == 0:
            return np.empty((0, self._n_pos), dtype=np.float64)

        # ``(n_visible, n_spatial_states, n_pos)``. Preserve the input
        # dtype so this collapse path matches the per-row helper
        # bit-for-bit (the per-row helper does no dtype promotion and
        # the result stays in the input dtype — important for the
        # numerical-equivalence regression tests).
        log_per_state = log_lik_window[:, self._selected_mask].reshape(
            n_visible, self._n_spatial_states, self._n_pos
        )
        # Per-row scalar mode: NaN → -inf, then max-subtract on the
        # finite entries. Rows with no finite entries collapse to
        # all-zeros to match the per-row helper's contract.
        log_per_state = np.where(
            np.isfinite(log_per_state), log_per_state, -np.inf
        )
        # Per-row max across (state, pos) — broadcast for subtraction.
        # ``where=isfinite`` keeps the max from picking up -inf when
        # any finite value exists; ``initial=-inf`` is the fallback
        # when a row is fully non-finite.
        finite_mask = np.isfinite(log_per_state)
        any_finite = finite_mask.any(axis=(1, 2))
        # Compute per-row max only over finite entries.
        per_row_max = np.where(
            any_finite,
            np.where(finite_mask, log_per_state, -np.inf).max(axis=(1, 2)),
            0.0,
        )
        log_per_state = log_per_state - per_row_max[:, None, None]
        lik_per_state = np.exp(log_per_state)
        lik_curve = lik_per_state.sum(axis=1)  # (n_visible, n_pos)
        # Per-row peak-normalise; rows with peak == 0 stay zero.
        peaks = lik_curve.max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normed = np.where(peaks > 0, lik_curve / peaks, lik_curve)
        # Rows that started fully non-finite must end up all-zero.
        normed = np.where(any_finite[:, None], normed, 0.0)
        return normed
