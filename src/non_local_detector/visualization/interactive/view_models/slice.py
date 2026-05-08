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
    _validate_rectangular_spatial,
    collapse_log_likelihood_to_position,
    collapse_posterior_to_position,
    select_reduction,
)
from non_local_detector.visualization.interactive.view_models.base import (
    BinPayload,
    CellSlice,
    SpikeEventIndex,
    bin_edges_array,
    bin_edges_at,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


# Title strings used by the panel; constants so tests can match
# them exactly without re-typing the prose.
TOP_CURVE_LIKELIHOOD_LABEL = "Likelihood (peak-normalised, all spatial states)"
TOP_CURVE_POSTERIOR_FALLBACK_LABEL = (
    "Posterior (likelihood unavailable; collapsed via active reduction)"
)


def collapse_log_likelihood_per_spatial_state(
    log_lik_row: np.ndarray, detector: _DetectorBase
) -> tuple[np.ndarray, ...]:
    """Return one peak-normalized likelihood curve per spatial state.

    This preserves the state-space-check top-slice convention where
    each spatial state's likelihood is visible instead of immediately
    summed into one aggregate curve.
    """
    spatial_state_ids, n_pos = _validate_rectangular_spatial(
        detector, "collapse_log_likelihood_per_spatial_state"
    )
    state_ind = np.asarray(detector.state_ind_)
    curves: list[np.ndarray] = []
    for state_id in spatial_state_ids:
        log_state = np.asarray(log_lik_row[state_ind == state_id])
        log_state = np.where(np.isfinite(log_state), log_state, -np.inf)
        if not np.isfinite(log_state).any():
            curves.append(np.zeros(n_pos, dtype=log_state.dtype))
            continue
        finite_max = log_state[np.isfinite(log_state)].max()
        curve = np.exp(log_state - finite_max)
        peak = curve.max()
        curves.append(curve / peak if peak > 0 else curve)
    return tuple(curves)


class SliceModel:
    """Per-bin slice over the active run's results + spike times.

    Construction caches the detector schema, the peak-normalised
    per-cell place fields, and the time grid; ``update_for_index``
    routes a single time-bin's posterior / log-likelihood / predictive
    rows through the analysis-layer collapse helpers and gathers the
    cells that fired in the cursor bin.

    The caller (``QtSlicePanel``) is responsible for fetching the
    per-bin rows from the data source and passing them in. Bin
    boundaries use the statespacecheck active-bin convention:
    bin ``i`` covers ``[time[i], time[i + 1])``.
    """

    def __init__(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        reduction: PosteriorReduction | None = None,
        event_index: SpikeEventIndex | None = None,
    ) -> None:
        self._bind(detector, spike_times, time, reduction, event_index)

    def _bind(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        reduction: PosteriorReduction | None,
        event_index: SpikeEventIndex | None,
    ) -> None:
        self._detector = detector
        self._spike_times = [np.asarray(st, dtype=np.float64) for st in spike_times]
        self._time = np.asarray(time, dtype=np.float64)
        self._reduction = reduction or select_reduction(
            detector.state_names, np.asarray(detector.bin_sizes_)
        )
        self._per_cell_place_fields = self._clean_place_fields(
            extract_per_cell_place_fields(detector)
        )
        self._per_cell_pf_normalized = self._peak_normalize(
            self._per_cell_place_fields
        )
        # Cache the per-spatial-state projection so per-tick top-curve
        # collapse is one ``[mask].reshape`` plus per-row max-subtract
        # / exp / peak-normalise — no ``state_ind == s`` rebuild, no
        # ``_validate_rectangular_spatial`` walk per tick.
        spatial_state_ids, n_pos_per_state = _validate_rectangular_spatial(
            detector, "SliceModel"
        )
        self._spatial_state_ids = np.asarray(spatial_state_ids)
        self._spatial_n_pos = int(n_pos_per_state)
        self._spatial_n_states = int(self._spatial_state_ids.size)
        self._spatial_state_mask = np.isin(
            np.asarray(detector.state_ind_), self._spatial_state_ids
        )
        self._event_index = event_index or SpikeEventIndex.from_spike_times(
            self._spike_times, self._time
        )

    @staticmethod
    def _clean_place_fields(per_cell_pf: np.ndarray) -> np.ndarray:
        """Return finite, non-negative expected spike counts per bin."""
        rates = np.asarray(per_cell_pf, dtype=float)
        rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(rates, 0.0, None)

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
        event_index: SpikeEventIndex | None = None,
    ) -> None:
        """Rebind to a new run (M-key swap)."""
        self._bind(detector, spike_times, time, reduction, event_index)

    def cell_slice(self, cell_id: int, spike_count: int = 0) -> CellSlice:
        """Return a ``CellSlice`` for ``cell_id``, regardless of activity.

        Used by the panel's pin path: pinned cells that didn't fire in
        the cursor bin still need their place-field row rendered.
        ``spike_count`` defaults to 0 (the typical pin case); callers
        with a real count (e.g. cell is both active *and* pinned) pass
        it explicitly.
        """
        if cell_id < 0 or cell_id >= self.n_cells:
            raise IndexError(f"cell_id={cell_id} out of range for {self.n_cells} cells")
        return CellSlice(
            cell_id=cell_id,
            place_field_norm=self._cell_curve_norm(cell_id, spike_count),
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
        top_curves: tuple[np.ndarray, ...] = ()
        if log_lik_row is not None:
            top_curve = collapse_log_likelihood_to_position(log_lik_row, self._detector)
            top_curves = self._per_state_curves(log_lik_row)
            top_curve_label = TOP_CURVE_LIKELIHOOD_LABEL
        else:
            top_curve = collapse_posterior_to_position(
                posterior_row, self._detector, self._reduction
            )
            top_curve_label = TOP_CURVE_POSTERIOR_FALLBACK_LABEL
        # Slice-panel overlays mirror statespacecheck's visual
        # convention: predictive / filtered / smoothed rows are
        # collapsed marginally across spatial states so local-dominated
        # bins still display a meaningful curve. The panel then
        # peak-normalizes for plotting.
        predictive_curve = (
            collapse_posterior_to_position(
                predictive_row, self._detector, PosteriorReduction.MARGINAL
            )
            if predictive_row is not None
            else None
        )
        cells = self._cells_at_index(t_idx)
        return BinPayload(
            t_idx=t_idx,
            t=t,
            top_curve=top_curve,
            top_curves=top_curves,
            top_curve_label=top_curve_label,
            predictive_curve=predictive_curve,
            cells=tuple(cells),
        )

    def _per_state_curves(self, log_lik_row: np.ndarray) -> tuple[np.ndarray, ...]:
        """Per-spatial-state peak-normalised likelihood curves (vectorised).

        Same output as ``collapse_log_likelihood_per_spatial_state`` but
        uses the cached ``_spatial_state_mask``/``_spatial_n_pos`` so
        the per-tick path doesn't re-derive them.
        """
        if self._spatial_n_states == 0:
            return ()
        # ``(n_states, n_pos)`` — rows in ``_spatial_state_ids`` order
        # because ``state_ind_`` groups by state and ``_spatial_state_mask``
        # preserves that ordering.
        log_per_state = log_lik_row[self._spatial_state_mask].reshape(
            self._spatial_n_states, self._spatial_n_pos
        )
        log_per_state = np.where(np.isfinite(log_per_state), log_per_state, -np.inf)
        any_finite = np.isfinite(log_per_state).any(axis=1)
        # Per-row max over finite entries; fully-non-finite rows fall
        # through to the all-zero output below.
        per_row_max = np.where(
            any_finite,
            np.where(np.isfinite(log_per_state), log_per_state, -np.inf).max(axis=1),
            0.0,
        )
        lik = np.exp(log_per_state - per_row_max[:, None])
        peaks = lik.max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normed = np.where(peaks > 0, lik / peaks, lik)
        # Restore the all-zero contract for fully-non-finite rows.
        normed = np.where(any_finite[:, None], normed, 0.0)
        return tuple(normed[i] for i in range(self._spatial_n_states))

    def _bin_edges(self, t_idx: int) -> tuple[float, float]:
        """Return left-edge ``(t_lo, t_hi)`` for bin ``t_idx``."""
        return bin_edges_at(self._time, t_idx)

    def _bin_edges_array(self) -> np.ndarray:
        """Return ``(n_bins + 1,)`` left-edge bin boundaries (left-edge convention)."""
        return bin_edges_array(self._time)

    def _cells_at_index(self, t_idx: int) -> list[CellSlice]:
        """Return active-cell slices for bin ``t_idx`` from event ids."""
        return self.cell_slices_for_events(self._event_index.event_ids_at_bin(t_idx))

    def cell_slices_for_events(self, event_ids: np.ndarray) -> list[CellSlice]:
        """Return one ``CellSlice`` per unique cell in ``event_ids``."""
        event_ids = np.asarray(event_ids, dtype=np.int64)
        if event_ids.size == 0:
            return []
        cell_ids = self._event_index.cell_ids[event_ids]
        unique_cell_ids, counts = np.unique(cell_ids, return_counts=True)
        return [
            self.cell_slice(int(cell_id), spike_count=int(count))
            for cell_id, count in zip(unique_cell_ids, counts, strict=True)
        ]

    def _cell_curve_norm(self, cell_id: int, spike_count: int) -> np.ndarray:
        """Return the per-cell row curve for an observed spike count.

        Active rows show the single-cell Poisson likelihood
        ``P(k=spike_count | position)`` up to the position-independent
        factorial constant, peak-normalized for plotting. Inactive pinned
        rows keep the place-field display so a pinned cell remains
        interpretable when it did not fire in the current bin.
        """
        if spike_count <= 0:
            return self._per_cell_pf_normalized[cell_id]
        rate = self._per_cell_place_fields[cell_id]
        log_lik = np.full(rate.shape, -np.inf, dtype=float)
        positive = rate > 0.0
        log_lik[positive] = spike_count * np.log(rate[positive]) - rate[positive]
        if not np.isfinite(log_lik).any():
            return np.zeros_like(rate, dtype=float)
        log_lik -= np.max(log_lik[np.isfinite(log_lik)])
        likelihood = np.exp(log_lik)
        likelihood = np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(likelihood))
        return likelihood / peak if peak > 0.0 else likelihood

    def event_ids_at_bin(self, t_idx: int) -> np.ndarray:
        """Expose event ids for tests/viewer parity checks."""
        return self._event_index.event_ids_at_bin(t_idx)
