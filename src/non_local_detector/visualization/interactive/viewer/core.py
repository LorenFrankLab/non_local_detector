"""``ViewerCore`` — Qt-free state + orchestration.

Owns:

- ``current_view_state`` (the frozen ``ViewState`` snapshot — what
  panels render against).
- The active-run state and ``set_active_run(...)`` rebind logic.
- The pinned-event state and the recenter-on-click handler.
- The event-overlay set + active-overlay selector + navigator
  (``next_event`` / ``prev_event`` given a ``t_center``).
- The stale-result rejection rule (drop results whose ``request_id``
  is older than the latest committed).

Does **not** own rendering, threading primitives, keyboard shortcuts,
or layout — those live in the frontend (``viewer/qt.py`` for v1,
``viewer/panel_.py`` for v2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.visualization.interactive.view_models.base import (
    ViewState,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.data_source import (
        InMemoryDecoderDataSource,
    )
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )
    from non_local_detector.visualization.interactive.viewer.backend import (
        BackendAdapter,
    )


class ViewerCore:
    """Frontend-agnostic state + orchestration for the decoder viewer."""

    def __init__(
        self,
        data_source: InMemoryDecoderDataSource,
        backend: BackendAdapter,
        t_center: float | None = None,
        t_width: float = 1.0,
    ) -> None:
        self._data_source = data_source
        self._backend = backend
        self._t_width = float(t_width)
        time = data_source.time
        self._t_center = (
            float(t_center) if t_center is not None else float(time[len(time) // 2])
        )
        self._next_request_id = 0
        self._latest_committed_request_id = -1
        self._current_view_state = self._build_view_state()
        self._pinned_event_row: int | None = None
        self._active_overlay_name: str | None = None
        self._on_window_loaded: Callable[[WindowPayload], None] | None = None
        self._on_active_run_changed_callbacks: list[Callable[[str], None]] = []

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def data_source(self) -> InMemoryDecoderDataSource:
        return self._data_source

    @property
    def current_view_state(self) -> ViewState:
        return self._current_view_state

    @property
    def t_center(self) -> float:
        return self._t_center

    @property
    def t_width(self) -> float:
        return self._t_width

    @property
    def active_run_name(self) -> str:
        return self._data_source.active_run_name

    @property
    def pinned_event_row(self) -> int | None:
        return self._pinned_event_row

    @property
    def active_overlay_name(self) -> str | None:
        return self._active_overlay_name

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def on_window_loaded(self, callback: Callable[[WindowPayload], None]) -> None:
        """Register the callback the backend invokes when a load completes."""
        self._on_window_loaded = callback

    def on_active_run_changed(self, callback: Callable[[str], None]) -> None:
        """Register a callback fired when ``set_active_run`` swaps runs.

        The callback receives the *new* active-run name. Multiple
        callbacks may be registered (one per panel that needs to
        rebind its view-model on schema swap — Posterior, Likelihood,
        StateProb, Raster, Slice).
        """
        self._on_active_run_changed_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Time navigation
    # ------------------------------------------------------------------

    def set_t_center(self, t_center: float) -> None:
        """Move the view center; emits a new load request."""
        self._t_center = float(t_center)
        self._current_view_state = self._build_view_state()
        self.request_load()

    def set_t_width(self, t_width: float) -> None:
        if t_width <= 0:
            raise ValueError(f"t_width must be positive. Got {t_width}.")
        self._t_width = float(t_width)
        self._current_view_state = self._build_view_state()
        self.request_load()

    def step_left(self, n_bins: int = 1) -> None:
        time = self._data_source.time
        t_idx = int(np.searchsorted(time, self._t_center))
        new_idx = max(0, t_idx - n_bins)
        self.set_t_center(float(time[new_idx]))

    def step_right(self, n_bins: int = 1) -> None:
        time = self._data_source.time
        t_idx = int(np.searchsorted(time, self._t_center))
        new_idx = min(len(time) - 1, t_idx + n_bins)
        self.set_t_center(float(time[new_idx]))

    # ------------------------------------------------------------------
    # Window loading + stale-result rejection
    # ------------------------------------------------------------------

    def request_load(self) -> None:
        """Dispatch a window-load request via the backend adapter."""
        state = self._current_view_state
        self._backend.schedule_window_load(state, self._handle_load_result)

    def _handle_load_result(self, payload: WindowPayload) -> None:
        """Invoked on the UI thread when a backend load completes."""
        if payload.request_id <= self._latest_committed_request_id:
            return  # Stale: a newer request has already committed.
        self._latest_committed_request_id = payload.request_id
        if self._on_window_loaded is not None:
            self._on_window_loaded(payload)

    def _build_view_state(self) -> ViewState:
        request_id = self._next_request_id
        self._next_request_id += 1
        return ViewState(
            request_id=request_id,
            t_center=self._t_center,
            t_width=self._t_width,
        )

    # ------------------------------------------------------------------
    # Active-run state (model swap)
    # ------------------------------------------------------------------

    def set_active_run(self, name: str) -> None:
        """Swap to a different run; preserves view state + pin + overlay name.

        Notifies every panel registered via ``on_active_run_changed``
        *before* dispatching the new load — panels rebind their
        view-models against the new detector schema, then receive the
        new payload collapsed under the correct schema.
        """
        self._data_source.set_active_run(name)
        for callback in self._on_active_run_changed_callbacks:
            callback(name)
        # Rebuild the request snapshot under the new run so the next
        # load tags freshly. Pin / overlay name / t_center / t_width
        # are intentionally preserved.
        self._current_view_state = self._build_view_state()
        self.request_load()

    # ------------------------------------------------------------------
    # Pinned-event state
    # ------------------------------------------------------------------

    def pin_event(self, row: int) -> None:
        self._pinned_event_row = int(row)

    def unpin_event(self) -> None:
        self._pinned_event_row = None

    # ------------------------------------------------------------------
    # Event overlays + navigator
    # ------------------------------------------------------------------

    def set_active_overlay(self, name: str | None) -> None:
        if name is None:
            self._active_overlay_name = None
            return
        # Validate against the active run's overlay set; the data
        # source has already enforced cross-bundle name alignment.
        names = {ovl.name for ovl in self._data_source.active_run.event_overlays}
        if name not in names:
            raise ValueError(f"No overlay named {name!r}. Available: {sorted(names)!r}")
        self._active_overlay_name = name

    def next_event(self) -> float | None:
        """Recenter on the next event of the active overlay, return its time.

        Returns ``None`` if no active overlay is selected, or if there
        is no event past ``t_center``.
        """
        return self._jump_to_event(direction=+1)

    def prev_event(self) -> float | None:
        """Recenter on the previous event of the active overlay."""
        return self._jump_to_event(direction=-1)

    def _jump_to_event(self, direction: int) -> float | None:
        overlay = self._active_overlay()
        if overlay is None:
            return None
        target = _event_target(overlay, self._t_center, direction=direction)
        if target is None:
            return None
        self.set_t_center(target)
        return target

    def _active_overlay(self) -> EventOverlay | None:
        if self._active_overlay_name is None:
            return None
        for ovl in self._data_source.active_run.event_overlays:
            if ovl.name == self._active_overlay_name:
                return ovl
        return None


def _event_target(
    overlay: EventOverlay, t_center: float, direction: int
) -> float | None:
    """Return the event time to jump to, given direction (+1 next / -1 prev).

    For ``points`` overlays: smallest/largest ``times[i]`` past
    ``t_center``.

    For ``intervals`` overlays: midpoint of the next/previous interval
    whose ``t_start[i]`` (next) or ``t_end[i]`` (prev) is past
    ``t_center``. When ``t_center`` is currently inside an interval,
    "next" skips to the next one — "go to next event," not "go to end
    of this one."
    """
    if direction not in (-1, +1):
        raise ValueError(f"direction must be -1 or +1, got {direction}")
    if overlay.kind == "points":
        times = np.asarray(overlay.times)
        if direction > 0:
            candidates = times[times > t_center]
            return float(candidates.min()) if candidates.size else None
        candidates = times[times < t_center]
        return float(candidates.max()) if candidates.size else None
    if overlay.kind == "intervals":
        t_start = np.asarray(overlay.t_start)
        t_end = np.asarray(overlay.t_end)
        if direction > 0:
            mask = t_start > t_center
            if not mask.any():
                return None
            i = int(np.flatnonzero(mask)[0])
        else:
            mask = t_end < t_center
            if not mask.any():
                return None
            i = int(np.flatnonzero(mask)[-1])
        return float((t_start[i] + t_end[i]) / 2.0)
    raise ValueError(
        f"Unknown overlay kind {overlay.kind!r}; expected 'points' or 'intervals'."
    )
