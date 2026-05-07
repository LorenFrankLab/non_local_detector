"""``SliceModel`` — per-bin slice readouts for the right-column panel.

Wraps the analysis-layer collapse helpers + per-cell place-field
lookup so ``QtSlicePanel.update_for_index`` can render the cursor
bin's likelihood/posterior curve, predictive overlay, and per-cell
rows from a single dataclass output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.analysis.place_fields import extract_per_cell_place_fields
from non_local_detector.analysis.posterior import (
    PosteriorReduction,
    collapse_log_likelihood_to_position,
    collapse_posterior_to_position,
    select_reduction,
)
from non_local_detector.visualization.interactive.view_models.base import (
    BinPayload,
    CellSlice,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


# Title strings used by the panel; constants so tests can match
# them exactly without re-typing the prose.
TOP_CURVE_LIKELIHOOD_LABEL = "Likelihood (peak-normalised, all spatial states)"
TOP_CURVE_POSTERIOR_FALLBACK_LABEL = (
    "Posterior (likelihood unavailable; collapsed via active reduction)"
)


class SliceModel:
    """Per-bin slice over the active run's results + spike times.

    Construction caches the detector schema, the peak-normalised
    per-cell place fields, and the time grid; ``update_for_index``
    routes a single time-bin's posterior / log-likelihood / predictive
    rows through the analysis-layer collapse helpers and gathers the
    cells that fired in the cursor bin.

    The caller (``QtSlicePanel``) is responsible for fetching the
    per-bin rows from the data source and passing them in. Bin
    boundaries are inferred from the time grid as midpoints to
    neighbors.
    """

    def __init__(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        reduction: PosteriorReduction | None = None,
    ) -> None:
        self._bind(detector, spike_times, time, reduction)

    def _bind(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        reduction: PosteriorReduction | None,
    ) -> None:
        self._detector = detector
        self._spike_times = [np.asarray(st, dtype=np.float64) for st in spike_times]
        self._time = np.asarray(time, dtype=np.float64)
        self._reduction = reduction or select_reduction(
            detector.state_names, np.asarray(detector.bin_sizes_)
        )
        self._per_cell_pf_normalized = self._peak_normalize(
            extract_per_cell_place_fields(detector)
        )
        # Precompute the per-bin (cell_id, count) lookup so per-tick
        # cell readouts are O(n_active_cells_in_bin) instead of
        # O(n_cells × spikes_per_cell). The full session usually
        # has a few tens of cells × ~10^4 spikes, and the previous
        # per-call scan dominated cursor-update latency on fast drags.
        self._per_bin_cell_counts = self._build_bin_index()

    @staticmethod
    def _peak_normalize(per_cell_pf: np.ndarray) -> np.ndarray:
        # Peak-normalize per cell. Cells with all-zero / all-NaN place
        # fields stay all-zero — never divide by 0 / NaN.
        peaks = np.nanmax(per_cell_pf, axis=1, keepdims=True)
        safe = np.where(np.isfinite(peaks) & (peaks > 0), peaks, 1.0)
        out = per_cell_pf / safe
        out = np.where(np.isfinite(out), out, 0.0)
        return out

    @property
    def detector(self) -> _DetectorBase:
        return self._detector

    @property
    def reduction(self) -> PosteriorReduction:
        return self._reduction

    @property
    def n_cells(self) -> int:
        return len(self._spike_times)

    def set_active_run(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        reduction: PosteriorReduction | None = None,
    ) -> None:
        """Rebind to a new run (M-key swap)."""
        self._bind(detector, spike_times, time, reduction)

    def cell_slice(self, cell_id: int, spike_count: int = 0) -> CellSlice:
        """Return a ``CellSlice`` for ``cell_id``, regardless of activity.

        Used by the panel's pin path: pinned cells that didn't fire in
        the cursor bin still need their place-field row rendered.
        ``spike_count`` defaults to 0 (the typical pin case); callers
        with a real count (e.g. cell is both active *and* pinned) pass
        it explicitly.
        """
        if cell_id < 0 or cell_id >= self.n_cells:
            raise IndexError(
                f"cell_id={cell_id} out of range for {self.n_cells} cells"
            )
        return CellSlice(
            cell_id=cell_id,
            place_field_norm=self._per_cell_pf_normalized[cell_id],
            spike_count=spike_count,
        )

    def update_for_index(
        self,
        t_idx: int,
        posterior_row: np.ndarray,
        log_lik_row: np.ndarray | None = None,
        predictive_row: np.ndarray | None = None,
    ) -> BinPayload:
        """Return a ``BinPayload`` describing the cursor bin."""
        if t_idx < 0 or t_idx >= self._time.size:
            raise IndexError(
                f"t_idx={t_idx} out of range for time grid of size {self._time.size}"
            )
        t = float(self._time[t_idx])
        if log_lik_row is not None:
            top_curve = collapse_log_likelihood_to_position(
                log_lik_row, self._detector
            )
            top_curve_label = TOP_CURVE_LIKELIHOOD_LABEL
        else:
            top_curve = collapse_posterior_to_position(
                posterior_row, self._detector, self._reduction
            )
            top_curve_label = TOP_CURVE_POSTERIOR_FALLBACK_LABEL
        predictive_curve = (
            collapse_posterior_to_position(
                predictive_row, self._detector, self._reduction
            )
            if predictive_row is not None
            else None
        )
        cells = self._cells_at_index(t_idx)
        return BinPayload(
            t_idx=t_idx,
            t=t,
            top_curve=top_curve,
            top_curve_label=top_curve_label,
            predictive_curve=predictive_curve,
            cells=tuple(cells),
        )

    def _bin_edges(self, t_idx: int) -> tuple[float, float]:
        """Return ``(t_lo, t_hi)`` for bin ``t_idx`` using midpoints."""
        n = self._time.size
        if n == 1:
            return float(self._time[0]), float(self._time[0])
        t = float(self._time[t_idx])
        if t_idx == 0:
            half = (self._time[1] - self._time[0]) / 2.0
        elif t_idx == n - 1:
            half = (self._time[n - 1] - self._time[n - 2]) / 2.0
        else:
            half_lo = (t - self._time[t_idx - 1]) / 2.0
            half_hi = (self._time[t_idx + 1] - t) / 2.0
            return float(t - half_lo), float(t + half_hi)
        return float(t - half), float(t + half)

    def _build_bin_index(self) -> list[dict[int, int]]:
        """Return ``per_bin[t_idx] = {cell_id: spike_count_in_bin}``.

        Spikes are assigned to a bin via ``searchsorted`` over
        midpoint-derived bin edges (matching ``_bin_edges``). Spikes
        before the first edge or after the last edge are dropped.
        """
        n_bins = self._time.size
        per_bin: list[dict[int, int]] = [dict() for _ in range(n_bins)]
        if n_bins == 0:
            return per_bin
        edges = self._bin_edges_array()
        for cell_id, st in enumerate(self._spike_times):
            if st.size == 0:
                continue
            idx = np.searchsorted(edges, st, side="right") - 1
            valid = (idx >= 0) & (idx < n_bins)
            for bin_i in idx[valid]:
                bucket = per_bin[int(bin_i)]
                bucket[cell_id] = bucket.get(cell_id, 0) + 1
        return per_bin

    def _bin_edges_array(self) -> np.ndarray:
        """Return ``(n_bins + 1,)`` midpoint-derived bin edges.

        Matches the per-bin-edge convention used by ``_bin_edges``:
        each bin is centered on ``time[i]`` and bounded by the
        midpoint to its neighbors. Used by ``_build_bin_index`` for a
        single vectorised ``searchsorted`` call.
        """
        time = self._time
        n = time.size
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if n == 1:
            half = 0.5
            return np.array([time[0] - half, time[0] + half], dtype=np.float64)
        midpoints = (time[1:] + time[:-1]) / 2.0
        first = float(time[0] - (midpoints[0] - time[0]))
        last = float(time[-1] + (time[-1] - midpoints[-1]))
        return np.concatenate([[first], midpoints, [last]])

    def _cells_at_index(self, t_idx: int) -> list[CellSlice]:
        """Return active-cell slices for bin ``t_idx`` from the prebuilt index."""
        if t_idx < 0 or t_idx >= len(self._per_bin_cell_counts):
            return []
        bucket = self._per_bin_cell_counts[t_idx]
        # Sort by cell_id for stable downstream rendering.
        return [
            CellSlice(
                cell_id=cell_id,
                place_field_norm=self._per_cell_pf_normalized[cell_id],
                spike_count=count,
            )
            for cell_id, count in sorted(bucket.items())
        ]
