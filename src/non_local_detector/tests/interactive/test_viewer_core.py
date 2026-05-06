"""Tests for the Qt-free ``ViewerCore``.

These exercise the orchestration logic (state transitions,
stale-result rejection, overlay navigation, model swap) against an
in-memory backend stub — no Qt event loop required.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest

from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.view_models.base import (
    ViewState,
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)
from non_local_detector.visualization.interactive.viewer.backend import (
    BackendAdapter,
)
from non_local_detector.visualization.interactive.viewer.core import ViewerCore


@dataclass
class StubBackend(BackendAdapter):
    """Capture-everything BackendAdapter for unit tests.

    ``schedule_window_load`` records the request and (by default)
    invokes the callback synchronously with a payload that echoes the
    state's ``request_id``. Tests can disable auto-fire to inspect
    queued requests.
    """

    auto_fire: bool = True
    requests: list[ViewState] = field(default_factory=list)
    pending_callbacks: list[tuple[ViewState, Callable[[WindowPayload], None]]] = field(
        default_factory=list
    )
    posted_callbacks: list[Callable[[], None]] = field(default_factory=list)

    def schedule_window_load(
        self, state: ViewState, on_done: Callable[[WindowPayload], None]
    ) -> None:
        self.requests.append(state)
        if self.auto_fire:
            on_done(_payload_for(state))
        else:
            self.pending_callbacks.append((state, on_done))

    def post_to_ui_thread(self, fn: Callable[[], None]) -> None:
        self.posted_callbacks.append(fn)

    def fire_pending(self, indices: list[int] | None = None) -> None:
        """Manually fire queued callbacks (for stale-result tests)."""
        if indices is None:
            indices = list(range(len(self.pending_callbacks)))
        for i in indices:
            state, cb = self.pending_callbacks[i]
            cb(_payload_for(state))


def _payload_for(state: ViewState) -> WindowPayload:
    return WindowPayload(
        request_id=state.request_id,
        time=np.array([state.t_center]),
        indices=slice(0, 1),
    )


@pytest.fixture
def core_factory(multi_run_bundles):
    """Build a fresh ViewerCore + stub backend per test."""

    def _make(
        active_run: str = "nl",
        t_width: float = 0.5,
        auto_fire: bool = True,
    ) -> tuple[ViewerCore, StubBackend, InMemoryDecoderDataSource]:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        ds.set_active_run(active_run)
        backend = StubBackend(auto_fire=auto_fire)
        core = ViewerCore(ds, backend, t_width=t_width)
        return core, backend, ds

    return _make


@pytest.mark.unit
class TestViewerCoreBasics:
    def test_initial_state_centered_on_session_midpoint(self, core_factory) -> None:
        core, _, ds = core_factory()
        time = ds.time
        assert core.t_center == pytest.approx(float(time[len(time) // 2]))
        assert core.t_width == 0.5
        assert core.active_run_name == "nl"
        assert core.pinned_event_row is None
        assert core.active_overlay_name is None

    def test_set_t_center_emits_new_request(self, core_factory) -> None:
        core, backend, _ = core_factory()
        baseline = len(backend.requests)
        core.set_t_center(core.t_center + 0.1)
        assert len(backend.requests) == baseline + 1
        assert backend.requests[-1].t_center == pytest.approx(core.t_center)

    def test_set_t_width_validates_positive(self, core_factory) -> None:
        core, _, _ = core_factory()
        with pytest.raises(ValueError, match="t_width"):
            core.set_t_width(0.0)
        with pytest.raises(ValueError, match="t_width"):
            core.set_t_width(-1.0)

    def test_step_left_right_clamped(self, core_factory) -> None:
        core, _, ds = core_factory()
        time = ds.time
        # Step to the leftmost bin then try one more step.
        n = len(time)
        for _ in range(n + 5):
            core.step_left()
        assert core.t_center == pytest.approx(float(time[0]))
        for _ in range(n + 5):
            core.step_right()
        assert core.t_center == pytest.approx(float(time[-1]))


@pytest.mark.unit
class TestViewerCoreStaleRejection:
    """Stale-result rejection drops out-of-order responses."""

    def test_late_arrival_dropped(self, core_factory) -> None:
        core, backend, _ = core_factory(auto_fire=False)
        received: list[WindowPayload] = []
        core.on_window_loaded(received.append)

        # Issue request A.
        core.request_load()
        # Issue request B (newer).
        core.set_t_center(core.t_center + 0.1)

        assert len(backend.pending_callbacks) == 2
        # Fire B first, then A — the late A must be dropped.
        backend.fire_pending(indices=[1, 0])
        # Only the B response was committed.
        assert len(received) == 1
        assert received[0].request_id == backend.pending_callbacks[1][0].request_id

    def test_same_request_id_dropped_once_committed(self, core_factory) -> None:
        core, backend, _ = core_factory(auto_fire=False)
        received: list[WindowPayload] = []
        core.on_window_loaded(received.append)

        core.request_load()
        backend.fire_pending(indices=[0])
        # Re-fire the same callback — should be dropped (already committed).
        backend.fire_pending(indices=[0])
        assert len(received) == 1


@pytest.mark.unit
class TestViewerCoreModelSwap:
    """``set_active_run`` rebinds the data source + preserves view state."""

    def test_swap_preserves_t_center_and_pin(self, core_factory) -> None:
        core, backend, _ = core_factory()
        original_t = core.t_center
        core.pin_event(7)
        baseline_requests = len(backend.requests)
        core.set_active_run("cf")

        assert core.active_run_name == "cf"
        assert core.t_center == pytest.approx(original_t)
        assert core.pinned_event_row == 7
        # A new load was emitted after swap.
        assert len(backend.requests) > baseline_requests

    def test_swap_increments_request_id(self, core_factory) -> None:
        core, _, _ = core_factory()
        before = core.current_view_state.request_id
        core.set_active_run("cf")
        assert core.current_view_state.request_id > before


@pytest.mark.unit
class TestViewerCoreOverlayNavigation:
    """Active-overlay selection + ``next_event`` / ``prev_event``."""

    def _attach_overlay_to_all(self, multi_run_bundles, overlay: EventOverlay) -> None:
        for bundle in multi_run_bundles.values():
            bundle.event_overlays.append(overlay)

    def test_set_active_overlay_validates_name(self, multi_run_bundles) -> None:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        backend = StubBackend()
        core = ViewerCore(ds, backend)
        with pytest.raises(ValueError, match="overlay"):
            core.set_active_overlay("does_not_exist")

    def test_next_prev_event_points(self, multi_run_bundles) -> None:
        original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
        try:
            overlay = EventOverlay.points(
                name="evt", times=np.array([1.0, 2.0, 3.0, 4.0])
            )
            self._attach_overlay_to_all(multi_run_bundles, overlay)
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            core = ViewerCore(ds, StubBackend(), t_center=2.5, t_width=0.5)
            core.set_active_overlay("evt")
            assert core.next_event() == pytest.approx(3.0)
            assert core.t_center == pytest.approx(3.0)
            assert core.prev_event() == pytest.approx(2.0)
            assert core.t_center == pytest.approx(2.0)
            # Past last point → no-op.
            core.set_t_center(5.0)
            assert core.next_event() is None
        finally:
            for n, b in multi_run_bundles.items():
                b.event_overlays[:] = original[n]

    def test_next_prev_event_intervals_skip_current(self, multi_run_bundles) -> None:
        original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
        try:
            overlay = EventOverlay.intervals(
                name="windows",
                t_start=np.array([1.0, 5.0, 10.0]),
                t_end=np.array([2.0, 7.0, 12.0]),
            )
            self._attach_overlay_to_all(multi_run_bundles, overlay)
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            # t_center=6 is inside the second interval [5, 7].
            core = ViewerCore(ds, StubBackend(), t_center=6.0, t_width=0.5)
            core.set_active_overlay("windows")
            # Next: midpoint of [10, 12] = 11.0 (skips current).
            assert core.next_event() == pytest.approx(11.0)
            # Prev from a centre-of-current (e.g. t_center=6): previous
            # interval whose t_end < t_center is [1, 2] (t_end=2 < 6),
            # midpoint 1.5.
            core.set_t_center(6.0)
            assert core.prev_event() == pytest.approx(1.5)
        finally:
            for n, b in multi_run_bundles.items():
                b.event_overlays[:] = original[n]

    def test_no_active_overlay_returns_none(self, core_factory) -> None:
        core, _, _ = core_factory()
        assert core.next_event() is None
        assert core.prev_event() is None
