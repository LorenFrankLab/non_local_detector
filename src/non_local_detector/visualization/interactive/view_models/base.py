"""Core view-model dataclasses for the interactive viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr

from non_local_detector._validation import (
    ensure_array_1d,
    ensure_matching_lengths,
    ensure_monotonic_increasing,
)
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
    find_duplicate_overlay_names,
)
from non_local_detector.visualization.interactive.view_models.series import (
    MetricSpec,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


ExtraMetricValue = Union[MetricSpec, "pd.Series"]


def bin_edges_array(time: np.ndarray) -> np.ndarray:
    """Return ``(n_bins + 1,)`` left-edge bin edges for a 1D time grid.

    Left-edge convention: bin ``i`` covers ``[time[i], time[i + 1])``;
    the trailing edge is inferred from the last delta.
    Empty input returns an empty array; size-1 input returns a unit
    interval centred on ``time[0]``.
    """
    time = np.asarray(time)
    n = time.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        t = float(time[0])
        return np.array([t, t + 1.0], dtype=np.float64)
    return np.concatenate(
        [time.astype(np.float64), [float(time[-1] + (time[-1] - time[-2]))]]
    )


def bin_edges_at(time: np.ndarray, t_idx: int) -> tuple[float, float]:
    """Return ``(t_lo, t_hi)`` for bin ``t_idx`` under the left-edge
    convention; matches ``bin_edges_array`` at indices ``t_idx``,
    ``t_idx + 1``."""
    n = time.size
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        t = float(time[0])
        return t, t
    t_lo = float(time[t_idx])
    if t_idx < n - 1:
        return t_lo, float(time[t_idx + 1])
    return t_lo, float(t_lo + (time[-1] - time[-2]))


@dataclass(frozen=True)
class ViewState:
    """Per-load request snapshot.

    The window-load worker tags its result with ``request_id``; the
    viewer drops a result if its ``request_id`` is older than the latest
    committed one. Mirrors the statespacecheck pattern.

    Carries only what's needed to fulfil one window-load request.
    Pinned-event state, overlay state, and active-run state live on
    ``ViewerCore`` (those don't influence which window the worker
    fetches).
    """

    request_id: int
    t_center: float
    t_width: float
    load_acausal: bool = False


@dataclass(frozen=True)
class PositionGrid:
    """Position-axis description, abstracted over 1D / 2D.

    v1 ships only the 1D path (``ndim == 1``). v3+ extends to 2D
    decoders (``ndim == 2``) by populating the second-axis fields.
    Panels that visualize position-distributions take a
    ``PositionGrid`` and dispatch on ``.ndim``.
    """

    ndim: int
    centers: np.ndarray
    is_interior: np.ndarray | None = None
    centers_y: np.ndarray | None = None
    is_interior_y: np.ndarray | None = None

    @classmethod
    def from_environment(cls, environment) -> PositionGrid:
        """Build a ``PositionGrid`` from a fitted ``Environment``."""
        centers = np.asarray(environment.place_bin_centers_)
        n_pos_dims = centers.shape[1]
        is_interior = (
            np.asarray(environment.is_track_interior_).ravel()
            if environment.is_track_interior_ is not None
            else None
        )
        if n_pos_dims == 1:
            return cls(ndim=1, centers=centers.squeeze(-1), is_interior=is_interior)
        if n_pos_dims == 2:
            return cls(
                ndim=2,
                centers=centers[:, 0],
                centers_y=centers[:, 1],
                is_interior=is_interior,
            )
        raise ValueError(
            f"PositionGrid supports 1D or 2D environments, got n_pos_dims={n_pos_dims}."
        )


@dataclass(frozen=True)
class SpikeEvent:
    """One sorted spike event in the interactive viewer event index."""

    event_id: int
    cell_id: int
    time: float
    time_index: int


@dataclass(frozen=True)
class SpikeEventIndex:
    """Stable event table built from per-cell spike times.

    Events are sorted by ``(time, cell_id, within_cell_ordinal)``. The
    row position is the stable ``event_id`` used by raster clicks and
    pin state. Only spikes that fall inside the decoder's left-edge
    bin range ``[time[0], inferred_final_edge)`` are retained.

    **Run-local.** ``event_id`` values are only meaningful within a
    single index. A different ``RunBundle`` (or even the same bundle
    rebuilt against a different time grid) produces a fresh index
    with potentially different ids — the viewer's M-key swap clears
    any pinned event id for this reason. Pass the index instance
    explicitly when sharing across view-models so they don't drift.
    """

    times: np.ndarray
    cell_ids: np.ndarray
    time_indices: np.ndarray
    cell_event_ids: tuple[np.ndarray, ...]

    @classmethod
    def from_spike_times(
        cls, spike_times: list[np.ndarray], time: np.ndarray
    ) -> SpikeEventIndex:
        time = np.asarray(time, dtype=np.float64)
        n_time = int(time.size)
        n_cells = len(spike_times)
        if n_time == 0 or n_cells == 0:
            empty_i64 = np.empty(0, dtype=np.int64)
            empty_f64 = np.empty(0, dtype=np.float64)
            return cls(
                times=empty_f64,
                cell_ids=empty_i64,
                time_indices=empty_i64,
                cell_event_ids=tuple(empty_i64.copy() for _ in range(n_cells)),
            )

        # Filter per-cell BEFORE concatenation so we don't allocate
        # full-session flat arrays just to mask out edge spikes.
        edges = bin_edges_array(time)
        edge_lo = float(edges[0])
        edge_hi = float(edges[-1])
        event_times: list[np.ndarray] = []
        event_cells: list[np.ndarray] = []
        event_ordinals: list[np.ndarray] = []
        event_time_indices: list[np.ndarray] = []
        for cell_id, st in enumerate(spike_times):
            spikes = np.asarray(st, dtype=np.float64)
            if spikes.size == 0:
                continue
            valid = (spikes >= edge_lo) & (spikes < edge_hi)
            if not valid.any():
                continue
            kept_spikes = spikes[valid]
            kept_indices = np.searchsorted(edges, kept_spikes, side="right") - 1
            kept_ordinals = np.flatnonzero(valid).astype(np.int64, copy=False)
            event_times.append(kept_spikes)
            event_cells.append(np.full(kept_spikes.shape, cell_id, dtype=np.int64))
            event_ordinals.append(kept_ordinals)
            event_time_indices.append(kept_indices.astype(np.int64, copy=False))
        if not event_times:
            empty_i64 = np.empty(0, dtype=np.int64)
            return cls(
                times=np.empty(0, dtype=np.float64),
                cell_ids=empty_i64,
                time_indices=empty_i64,
                cell_event_ids=tuple(empty_i64.copy() for _ in range(n_cells)),
            )

        flat_times = np.concatenate(event_times)
        flat_cells = np.concatenate(event_cells)
        flat_ordinals = np.concatenate(event_ordinals)
        flat_time_indices = np.concatenate(event_time_indices)
        order = np.lexsort((flat_ordinals, flat_cells, flat_times))
        times = flat_times[order].astype(np.float64, copy=False)
        cell_ids = flat_cells[order].astype(np.int64, copy=False)
        time_indices = flat_time_indices[order]

        event_ids_by_cell = []
        for cell_id in range(n_cells):
            event_ids_by_cell.append(
                np.flatnonzero(cell_ids == cell_id).astype(np.int64, copy=False)
            )
        return cls(
            times=times,
            cell_ids=cell_ids,
            time_indices=time_indices,
            cell_event_ids=tuple(event_ids_by_cell),
        )

    @property
    def n_events(self) -> int:
        return int(self.times.size)

    def event_at(self, event_id: int) -> SpikeEvent:
        if event_id < 0 or event_id >= self.n_events:
            raise IndexError(
                f"event_id={event_id} out of range for {self.n_events} events"
            )
        return SpikeEvent(
            event_id=int(event_id),
            cell_id=int(self.cell_ids[event_id]),
            time=float(self.times[event_id]),
            time_index=int(self.time_indices[event_id]),
        )

    def event_ids_at_bin(self, t_idx: int) -> np.ndarray:
        """Return event ids whose precomputed decoder bin is ``t_idx``."""
        if self.time_indices.size == 0:
            return np.empty(0, dtype=np.int64)
        i0 = int(np.searchsorted(self.time_indices, t_idx, side="left"))
        i1 = int(np.searchsorted(self.time_indices, t_idx, side="right"))
        return np.arange(i0, i1, dtype=np.int64)

    def event_ids_for_window(self, sl: slice) -> np.ndarray:
        """Return event ids whose bins fall in ``[sl.start, sl.stop)``."""
        if self.time_indices.size == 0:
            return np.empty(0, dtype=np.int64)
        i0 = int(np.searchsorted(self.time_indices, sl.start, side="left"))
        i1 = int(np.searchsorted(self.time_indices, sl.stop, side="left"))
        return np.arange(i0, i1, dtype=np.int64)

    def event_ids_for_cell_window(
        self, cell_id: int, t_start: float, t_stop: float
    ) -> np.ndarray:
        """Return event ids for one cell with times in ``[t_start, t_stop)``."""
        if cell_id < 0 or cell_id >= len(self.cell_event_ids):
            return np.empty(0, dtype=np.int64)
        event_ids = self.cell_event_ids[cell_id]
        times = self.times[event_ids]
        i0 = int(np.searchsorted(times, t_start, side="left"))
        i1 = int(np.searchsorted(times, t_stop, side="left"))
        return event_ids[i0:i1]

    def event_id_for_cell_time(self, cell_id: int, t: float) -> int | None:
        """Return the first exact event id for ``(cell_id, t)``, if present."""
        if cell_id < 0 or cell_id >= len(self.cell_event_ids):
            return None
        event_ids = self.cell_event_ids[cell_id]
        times = self.times[event_ids]
        i = int(np.searchsorted(times, t, side="left"))
        if i >= times.size or times[i] != t:
            return None
        return int(event_ids[i])

    def nearest_event_id_for_cell_time(
        self, cell_id: int, t: float, *, atol: float = 1e-6
    ) -> int | None:
        """Return the closest event id for ``(cell_id, t)`` within ``atol``.

        Float-rounded ``t`` (e.g. from a pyqtgraph float32 spot
        position) won't survive strict equality against the float64
        spike-time table; this helper picks the nearer of the two
        bracketing events and returns it iff its absolute distance
        from ``t`` is within ``atol``. Returns ``None`` when no event
        exists within the tolerance.
        """
        if cell_id < 0 or cell_id >= len(self.cell_event_ids):
            return None
        event_ids = self.cell_event_ids[cell_id]
        times = self.times[event_ids]
        if times.size == 0:
            return None
        i = int(np.searchsorted(times, t, side="left"))
        candidates: list[int] = []
        if i < times.size:
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        best = min(candidates, key=lambda j: abs(float(times[j]) - t))
        if abs(float(times[best]) - t) > atol:
            return None
        return int(event_ids[best])


@dataclass(frozen=True)
class WindowPayload:
    """Output of a window load — what a ``TimeAxisPanel.update_window`` consumes.

    Each field carries data already shaped to ``(n_visible, ...)`` for
    the window the worker resolved. ``request_id`` lets the viewer
    drop stale results.
    """

    request_id: int
    time: np.ndarray
    indices: slice
    time_start: float | None = None
    time_stop: float | None = None
    # ``t_center`` / ``t_width`` carry the requested view state so
    # panels can render at relative coordinates (``time - t_center``)
    # against a fixed ``[-t_width/2, +t_width/2]`` x-range without
    # re-deriving the view from ``time_start`` / ``time_stop`` —
    # which is unstable at session edges and on non-uniform grids.
    t_center: float = 0.0
    t_width: float = 0.0
    posterior: np.ndarray | None = None
    likelihood: np.ndarray | None = None
    predictive: np.ndarray | None = None
    state_probabilities: np.ndarray | None = None
    # 1D position (true behaviour) interpolated onto ``time``. Used
    # for the white trace overlaid on the posterior + likelihood
    # heatmaps. ``None`` when the bundle has no position or when the
    # position is 2D (heatmap overlay is 1D only in v1).
    position: np.ndarray | None = None


@dataclass(frozen=True)
class CellSlice:
    """Per-cell row payload for the SlicePanel.

    ``place_field_norm`` is the row curve to draw. For active cells this
    is the observed-count Poisson likelihood over position; for inactive
    pinned cells it falls back to the normalized place field.
    """

    cell_id: int
    place_field_norm: np.ndarray
    spike_count: int = 0


@dataclass(frozen=True)
class BinPayload:
    """Output of a single-bin load — what a ``BinSyncedPanel.update_for_index`` consumes.

    ``top_curve`` is the population-likelihood (or fallback collapsed
    posterior) over position; ``predictive_curve`` is the predictive
    overlay; ``cells`` are the per-cell rows for cells that fired in
    this bin.
    """

    t_idx: int
    t: float
    top_curve: np.ndarray | None = None
    top_curves: tuple[np.ndarray, ...] = ()
    top_curve_label: str = ""
    predictive_curve: np.ndarray | None = None
    cells: tuple[CellSlice, ...] = ()


@dataclass
class RunBundle:
    """User-facing viewer input bundle.

    Mutable so users can attach ``event_overlays`` / ``extra_metrics``
    after construction. ``__post_init__`` validates internal
    consistency once at construction; the data source revalidates the
    cross-bundle invariants (time-grid alignment, overlay schema)
    when bundles are loaded together.

    Parameters
    ----------
    results : xr.Dataset
        Decoder output. Must contain ``acausal_posterior`` and
        ``acausal_state_probabilities``. May optionally contain
        ``log_likelihood`` and ``predictive_posterior`` (panels that
        consume them disable themselves with a clear title-bar
        message when the array is missing).
    detector : _DetectorBase
        Fitted detector. Must expose ``state_ind_``, ``state_names``,
        ``encoding_model_``, and ``environments[0]``.
    spike_times : list[np.ndarray]
        Per-cell spike times in absolute seconds. Length must match
        the detector's encoding-model neuron count.
    position_time : np.ndarray, shape (n_position_time,)
        Monotonic absolute time in seconds for each position sample.
    position : np.ndarray
        Shape ``(n_position_time,)`` for 1D detectors,
        ``(n_position_time, 2)`` for 2D detectors.
    speed : np.ndarray, optional
        Shape ``(n_position_time,)``.
    events : pd.DataFrame, optional
        Per-spike event metrics (HPD overlap, KL divergence, spike
        prob).
    extra_metrics : dict[str, MetricSpec | pd.Series]
    event_overlays : list[EventOverlay]
        Names must be unique within the bundle (so the navigator's
        ``active_overlay_name`` resolves unambiguously).
    """

    results: xr.Dataset
    detector: _DetectorBase
    spike_times: list[np.ndarray]
    position_time: np.ndarray
    position: np.ndarray
    speed: np.ndarray | None = None
    events: pd.DataFrame | None = None
    extra_metrics: dict[str, ExtraMetricValue] = field(default_factory=dict)
    event_overlays: list[EventOverlay] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Required results variables.
        for var in ("acausal_posterior", "acausal_state_probabilities"):
            if var not in self.results:
                raise ValueError(
                    f"RunBundle.results is missing required variable {var!r}. "
                    "Got variables: "
                    f"{sorted(self.results.data_vars)!r}"
                )

        position_time = np.asarray(self.position_time)
        ensure_array_1d(position_time, "RunBundle.position_time")
        ensure_monotonic_increasing(
            position_time, "RunBundle.position_time", strict=True
        )

        env = self.detector.environments[0]
        n_pos_dims = env.place_bin_centers_.shape[1]
        position = np.asarray(self.position)
        # Accept 1D position arrays for 1D detectors (the model layer
        # auto-newaxes them); compare against the column count for 2D.
        position_n_cols = 1 if position.ndim == 1 else position.shape[1]
        if position_n_cols != n_pos_dims:
            raise ValueError(
                f"RunBundle.position has {position_n_cols} spatial "
                f"dim(s) but the detector's environment expects "
                f"{n_pos_dims}."
            )
        ensure_matching_lengths(
            position,
            position_time,
            "RunBundle.position",
            "RunBundle.position_time",
        )

        # spike_times length matches encoding-model neuron count.
        n_neurons_expected = self._infer_n_neurons()
        if n_neurons_expected is not None and (
            len(self.spike_times) != n_neurons_expected
        ):
            raise ValueError(
                f"RunBundle.spike_times has {len(self.spike_times)} "
                f"entries but the detector's encoding model carries "
                f"{n_neurons_expected} neurons."
            )

        duplicates = find_duplicate_overlay_names(self.event_overlays)
        if duplicates:
            raise ValueError(
                f"RunBundle.event_overlays has duplicate names: {duplicates}. "
                "Each overlay name must be unique within a bundle so that "
                "`active_overlay_name` resolves unambiguously."
            )

    def _infer_n_neurons(self) -> int | None:
        """Return per-encoding-model-entry n_neurons, or None if ambiguous."""
        encoding = getattr(self.detector, "encoding_model_", None)
        if not encoding:
            return None
        keys = list(encoding.keys())
        if len(keys) != 1:
            return None  # multi-environment / multi-group: skip strict check.
        place_fields = encoding[keys[0]].get("place_fields")
        if place_fields is None:
            return None
        return int(place_fields.shape[0])

    @property
    def available_outputs(self) -> set[str]:
        """Return the set of optional output arrays present in ``results``.

        ``acausal_posterior`` and ``acausal_state_probabilities`` are
        always required (validated in ``__post_init__``); this set
        reports which of the optional arrays
        (``log_likelihood``, ``predictive_posterior``,
        ``causal_posterior``, etc.) are also present.
        """
        return set(self.results.data_vars)

    @classmethod
    def from_predict(
        cls,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        position: np.ndarray,
        position_time: np.ndarray,
        speed: np.ndarray | None = None,
        return_outputs: str | list[str] | set[str] | None = "all",
        events: pd.DataFrame | None = None,
        extra_metrics: dict[str, ExtraMetricValue] | None = None,
        event_overlays: list[EventOverlay] | None = None,
    ) -> RunBundle:
        """Convenience constructor: call ``detector.predict(...)`` then wrap.

        Defaults ``return_outputs="all"`` so the bundle ships with the
        full panel set (likelihood, predictive overlay, smoothed
        posterior, state probabilities) by default; pass a tighter
        value for memory savings.
        """
        results = detector.predict(  # type: ignore[attr-defined]
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=position_time,
            return_outputs=return_outputs,
        )
        return cls(
            results=results,
            detector=detector,
            spike_times=spike_times,
            position_time=position_time,
            position=position,
            speed=speed,
            events=events,
            extra_metrics=extra_metrics or {},
            event_overlays=event_overlays or [],
        )
