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
    required_outputs_calls: list[set[str]] = field(default_factory=list)

    def schedule_window_load(
        self, state: ViewState, on_done: Callable[[WindowPayload], None]
    ) -> None:
        self.requests.append(state)
        if self.auto_fire:
            on_done(_payload_for(state))
        else:
            self.pending_callbacks.append((state, on_done))

    def build_payload(self, state: ViewState) -> WindowPayload:
        return _payload_for(state)

    def set_required_outputs(self, outputs: set[str]) -> None:
        self.required_outputs_calls.append(set(outputs))

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

    def test_set_t_width_clamps_to_min_max(self, core_factory) -> None:
        """Sub-floor / above-ceiling values get clipped, not raw-stored.

        Pins the fix for the "window size 0" bug: ``_scale_t_width``'s
        keyboard ``[`` halving and the CLI ``--t-width`` flag could
        bypass the slider's ``[MIN, MAX]_WINDOW_SECONDS`` clamp and
        drive ``t_width`` arbitrarily small (the old ``1e-6`` floor),
        at which point ``window_indices`` returns an empty slice and
        the heatmap renders nothing.
        """
        from non_local_detector.visualization.interactive.viewer.core import (
            MAX_T_WIDTH_SECONDS,
            MIN_T_WIDTH_SECONDS,
        )

        core, _, _ = core_factory()
        core.set_t_width(1e-9)
        assert core.t_width == pytest.approx(MIN_T_WIDTH_SECONDS)
        core.set_t_width(1e6)
        assert core.t_width == pytest.approx(MAX_T_WIDTH_SECONDS)

    def test_constructor_clamps_t_width(self, multi_run_bundles) -> None:
        """A CLI ``--t-width 0`` or stale persisted value can't sneak past."""
        from non_local_detector.visualization.interactive.viewer.core import (
            MIN_T_WIDTH_SECONDS,
        )

        ds = InMemoryDecoderDataSource(multi_run_bundles)
        backend = StubBackend()
        core = ViewerCore(ds, backend, t_width=1e-9)
        assert core.t_width == pytest.approx(MIN_T_WIDTH_SECONDS)

    def test_refresh_dispatches_with_fresh_request_id(self, core_factory) -> None:
        """``refresh`` must mint a new ``request_id`` so the payload commits.

        After the first window has committed, calling ``request_load``
        with the same ``_current_view_state`` re-uses the committed
        ``request_id`` and the new payload is silently dropped by the
        ``request_id <= latest_committed`` rule. ``refresh`` exists
        precisely so external callers (slice-overlay-mode toggle) can
        re-fetch the same window when ``_required_outputs`` widens.
        """
        core, backend, _ = core_factory(auto_fire=True)
        # Seed an initial commit (``__init__`` builds the state but
        # doesn't fire a load).
        core.request_load()
        committed_before = core._latest_committed_request_id
        prior_request_id = core.current_view_state.request_id
        assert committed_before >= 0

        baseline_requests = len(backend.requests)
        core.refresh()
        assert len(backend.requests) == baseline_requests + 1
        # The dispatched state must carry a fresh request_id and that
        # request_id must commit (auto_fire fires the callback inline).
        assert backend.requests[-1].request_id > prior_request_id
        assert core._latest_committed_request_id > committed_before

    def test_request_load_alone_is_dropped_after_commit(self, core_factory) -> None:
        """Demonstrates why ``refresh`` is necessary, not redundant.

        ``request_load`` reuses the existing ``_current_view_state``,
        so a same-window re-dispatch *after* a prior commit hits the
        stale-result rule and is dropped. Pinning this explicitly so
        future readers don't accidentally swap ``refresh`` back to
        ``request_load``.
        """
        core, backend, _ = core_factory(auto_fire=True)
        # Seed an initial commit so the stale-result rule has
        # something to compare against.
        core.request_load()
        committed_before = core._latest_committed_request_id
        assert committed_before >= 0

        core.request_load()
        # The backend saw the call …
        assert backend.requests[-1].request_id == committed_before
        # … but the commit pointer didn't advance — the payload was dropped.
        assert core._latest_committed_request_id == committed_before

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

    def test_older_payload_arriving_first_is_dropped(self, core_factory) -> None:
        """Regression: request 0 must be dropped if request 1 has been
        issued, even when request 0 is the *first* payload to arrive
        (i.e. no commit has happened yet).

        The naive rule ``payload.request_id > _latest_committed_request_id``
        accepts request 0 here because nothing has committed yet — but
        the user has already moved on to request 1, so rendering
        request 0's data briefly shows an obsolete window.
        """
        core, backend, _ = core_factory(auto_fire=False)
        received: list[WindowPayload] = []
        core.on_window_loaded(received.append)

        core.request_load()  # request A
        core.set_t_center(core.t_center + 0.1)  # request B

        request_a_id = backend.pending_callbacks[0][0].request_id
        request_b_id = backend.pending_callbacks[1][0].request_id
        assert request_a_id < request_b_id

        # Fire A *first* (before B). A is stale.
        backend.fire_pending(indices=[0])
        assert received == []
        # Then fire B — should commit.
        backend.fire_pending(indices=[1])
        assert len(received) == 1
        assert received[0].request_id == request_b_id


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

    def test_on_active_run_changed_callbacks_fire_before_load(
        self, core_factory
    ) -> None:
        """Callbacks fire on swap *before* the new load is dispatched.

        Order matters: panels rebind their view-models on the
        callback, then the new payload arrives and gets collapsed
        under the correct schema. Reversed order would briefly
        collapse the new payload under the stale schema.
        """
        core, backend, _ = core_factory()
        events: list[str] = []

        def _record_run_change(name: str) -> None:
            events.append(f"run_changed:{name}:{len(backend.requests)}")

        core.on_active_run_changed(_record_run_change)
        before_requests = len(backend.requests)
        core.set_active_run("cf")

        assert events == [f"run_changed:cf:{before_requests}"]
        # After the callback, exactly one additional load was issued.
        assert len(backend.requests) == before_requests + 1

    def test_multiple_active_run_callbacks_all_fire(self, core_factory) -> None:
        """Several panels can register; every callback fires on each swap."""
        core, _, _ = core_factory()
        a_calls: list[str] = []
        b_calls: list[str] = []
        core.on_active_run_changed(a_calls.append)
        core.on_active_run_changed(b_calls.append)
        core.set_active_run("cf")
        core.set_active_run("nsf")
        assert a_calls == ["cf", "nsf"]
        assert b_calls == ["cf", "nsf"]


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


@pytest.mark.unit
class TestViewerCoreOverlayDispatch:
    """``on_overlays_changed`` + ``set_overlay_visibility`` + ``refresh_overlays``."""

    def _attach_overlay_to_all(self, multi_run_bundles, overlay):
        for bundle in multi_run_bundles.values():
            bundle.event_overlays.append(overlay)

    def test_refresh_overlays_pushes_visible_set(self, multi_run_bundles) -> None:
        original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
        try:
            swr = EventOverlay.points(name="swr", times=np.array([1.0]))
            theta = EventOverlay.points(name="theta", times=np.array([2.0]))
            self._attach_overlay_to_all(multi_run_bundles, swr)
            self._attach_overlay_to_all(multi_run_bundles, theta)
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            core = ViewerCore(ds, StubBackend())
            received: list[list[EventOverlay]] = []
            core.on_overlays_changed(received.append)
            core.refresh_overlays()
            assert len(received) == 1
            assert {ovl.name for ovl in received[0]} == {"swr", "theta"}
        finally:
            for n, b in multi_run_bundles.items():
                b.event_overlays[:] = original[n]

    def test_set_overlay_visibility_filters(self, multi_run_bundles) -> None:
        original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
        try:
            swr = EventOverlay.points(name="swr", times=np.array([1.0]))
            theta = EventOverlay.points(name="theta", times=np.array([2.0]))
            self._attach_overlay_to_all(multi_run_bundles, swr)
            self._attach_overlay_to_all(multi_run_bundles, theta)
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            core = ViewerCore(ds, StubBackend())
            received: list[list[EventOverlay]] = []
            core.on_overlays_changed(received.append)
            # Hide theta.
            core.set_overlay_visibility("theta", False)
            assert {ovl.name for ovl in received[-1]} == {"swr"}
            # Re-enable theta.
            core.set_overlay_visibility("theta", True)
            assert {ovl.name for ovl in received[-1]} == {"swr", "theta"}
        finally:
            for n, b in multi_run_bundles.items():
                b.event_overlays[:] = original[n]

    def test_off_overlays_changed_unregisters_callback(self, multi_run_bundles) -> None:
        """``off_overlays_changed`` removes a previously registered callback."""
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        core = ViewerCore(ds, StubBackend())
        received: list[list[EventOverlay]] = []
        core.on_overlays_changed(received.append)
        core.refresh_overlays()
        assert len(received) == 1
        core.off_overlays_changed(received.append)
        core.refresh_overlays()
        # No new dispatch; received still length 1.
        assert len(received) == 1

    def test_swap_dispatches_overlays_for_new_run(self, multi_run_bundles) -> None:
        """Per-run overlay data updates on swap.

        Both runs declare the same ``("non_local_events", "points")``
        schema but with different times; after swap, the panel must
        get the new run's data.
        """
        original = {n: list(b.event_overlays) for n, b in multi_run_bundles.items()}
        try:
            multi_run_bundles["nl"].event_overlays.append(
                EventOverlay.points(
                    name="non_local_events", times=np.array([1.0, 2.0, 3.0])
                )
            )
            multi_run_bundles["cf"].event_overlays.append(
                EventOverlay.points(
                    name="non_local_events", times=np.array([10.0, 20.0])
                )
            )
            multi_run_bundles["nsf"].event_overlays.append(
                EventOverlay.points(name="non_local_events", times=np.array([100.0]))
            )
            multi_run_bundles["dec"].event_overlays.append(
                EventOverlay.points(name="non_local_events", times=np.array([1000.0]))
            )
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            core = ViewerCore(ds, StubBackend())
            received: list[list[EventOverlay]] = []
            core.on_overlays_changed(received.append)
            core.refresh_overlays()  # initial NL
            assert received[-1][0].times.tolist() == [1.0, 2.0, 3.0]
            core.set_active_run("cf")
            # After swap, the panel got the CF overlay data.
            assert received[-1][0].times.tolist() == [10.0, 20.0]
        finally:
            for n, b in multi_run_bundles.items():
                b.event_overlays[:] = original[n]
