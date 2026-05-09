"""Qt frontend: ``QApplication`` ownership, ``QtBackendAdapter``, ``QtViewer``.

This module is the only place in the codebase that creates a
``QApplication`` and the only Qt-importing entry under
``viewer/``. ``app.py`` lazy-imports ``launch_qt`` here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.panels.qt.likelihood import (
    QtLikelihoodHeatmapPanel,
)
from non_local_detector.visualization.interactive.panels.qt.posterior import (
    QtPosteriorHeatmapPanel,
)
from non_local_detector.visualization.interactive.panels.qt.raster import (
    QtRasterPanel,
)
from non_local_detector.visualization.interactive.panels.qt.series import (
    IntervalSeriesPanel,
    LineSeriesPanel,
    ScatterSeriesPanel,
)
from non_local_detector.visualization.interactive.panels.qt.slice import (
    OverlayMode,
    QtSlicePanel,
)
from non_local_detector.visualization.interactive.panels.qt.state_prob import (
    QtStateProbabilityPanel,
)
from non_local_detector.visualization.interactive.view_models.base import (
    PositionGrid,
    RunBundle,
    ViewState,
    WindowPayload,
    bin_edges_at,
)
from non_local_detector.visualization.interactive.view_models.likelihood import (
    LikelihoodHeatmapModel,
)
from non_local_detector.visualization.interactive.view_models.posterior import (
    PosteriorHeatmapModel,
)
from non_local_detector.visualization.interactive.view_models.raster import (
    RasterModel,
)
from non_local_detector.visualization.interactive.view_models.series import (
    IntervalSeriesModel,
    LineSeriesModel,
    MetricSpec,
    ScatterSeriesModel,
)
from non_local_detector.visualization.interactive.view_models.slice import (
    SliceModel,
)
from non_local_detector.visualization.interactive.view_models.state_prob import (
    StateProbabilityModel,
)
from non_local_detector.visualization.interactive.viewer.backend import (
    BackendAdapter,
)
from non_local_detector.visualization.interactive.viewer.core import (
    MAX_T_WIDTH_SECONDS,
    MIN_T_WIDTH_SECONDS,
    ViewerCore,
)

_LOAD_LOGGER = logging.getLogger(__name__)


# Auto-scroll constants (mirror upstream statespacecheck). Multipliers
# of real-time playback rate; tick fires at AUTOSCROLL_TICK_HZ Hz and
# advances the slider by ``rate / TICK_HZ`` seconds per tick.
AUTOSCROLL_TICK_HZ = 30.0
AUTOSCROLL_SPEED_OPTIONS: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)
AUTOSCROLL_DEFAULT_SPEED = 0.05
WINDOW_SLIDER_RESOLUTION = 1000
# Re-export the core's t_width bounds so the slider, wheel, and
# keyboard halve/double paths all clamp to the same range.
MIN_WINDOW_SECONDS = MIN_T_WIDTH_SECONDS
MAX_WINDOW_SECONDS = MAX_T_WIDTH_SECONDS

# Qt's offscreen macOS default can report "Sans Serif" even though no
# such family is installed, which triggers a slow alias-population path
# the first time labels render. Pick an installed concrete family once
# per QApplication.
_QT_FONT_FAMILY_PREFERENCES = (
    ".AppleSystemUIFont",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "Noto Sans",
)

# Layout constants mirror statespacecheck-paper-viewer after dropping the
# three diagnostic metric rows: 1200x900 default window, ~70/30 body
# split, heatmap rows at stretch 2, compact rows at stretch 1, and a
# right-column slice panel aligned to the posterior heatmap's vertical
# extent (2 of 6 units).
_DEFAULT_WINDOW_WIDTH = 1200
_DEFAULT_WINDOW_HEIGHT = 900
_LEFT_COLUMN_HEATMAP_STRETCH = 2
_LEFT_COLUMN_COMPACT_STRETCH = 1
_LEFT_COLUMN_EXTRA_STRETCH = _LEFT_COLUMN_COMPACT_STRETCH
_RIGHT_COLUMN_SLICE_STRETCH = 2
_RIGHT_COLUMN_TRAILING_STRETCH = 4

# Splitter weights for the body's left vs right column. Sum doesn't
# matter; pyqtgraph divides by total. 7:3 mirrors the paper viewer.
_BODY_SPLITTER_LEFT_STRETCH = 7
_BODY_SPLITTER_RIGHT_STRETCH = 3

# Tight margins/spacing so the panels read as one figure rather than
# four separate boxes.
_OUTER_MARGIN = 4
_OUTER_SPACING = 4
_COLUMN_MARGIN = 0
_COLUMN_SPACING = 2
_SLICE_COLUMN_SPACING = 0

_SLICE_OVERLAY_CHOICES: tuple[tuple[OverlayMode, str], ...] = (
    ("predictive", "Predictive (causal)"),
    ("filtered", "Filtered"),
    ("smoothed", "Smoothed (acausal)"),
)


def _format_speed(speed: float) -> str:
    """Render a multiplier as ``"1×"`` / ``"2×"`` / ``"0.05×"`` etc."""
    if speed >= 1.0 and float(speed).is_integer():
        return f"{speed:.0f}×"
    return f"{speed:.2g}×"


def _ensure_qapplication() -> QtWidgets.QApplication:
    """Return the singleton ``QApplication`` with a concrete installed font."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ensure_concrete_application_font(app)
    return app


def _ensure_concrete_application_font(app: QtWidgets.QApplication) -> None:
    """Replace Qt's missing generic default font family when needed."""
    families = set(QtGui.QFontDatabase.families())
    current = app.font().family()
    if current in families:
        return
    for family in _QT_FONT_FAMILY_PREFERENCES:
        if family in families:
            font = QtGui.QFont(app.font())
            font.setFamily(family)
            app.setFont(font)
            return


class _LoadSignals(QtCore.QObject):
    """Signal bridge for thread-safe payload delivery (statespacecheck pattern).

    A worker running on the threadpool emits ``done(payload)``; the
    receiving slot runs on the UI thread.
    """

    done = QtCore.Signal(object)


class QtBackendAdapter(BackendAdapter):
    """Qt implementation of the backend protocol.

    Schedules window-load work on a background Python executor and
    marshals the result back via a ``QObject`` signal so the UI
    thread is the one that touches widgets.
    """

    # 16 ms ≈ one screen frame at 60 Hz — used for center-time
    # scrubbing where the user expects to see the window track the
    # slider in near-realtime. Matches statespacecheck's debounce.
    SCRUB_DEBOUNCE_MS = 16
    # Wheel- / slider-driven *resize* fires per-pixel and each load
    # touches arrays whose size scales with ``n_visible``; coalescing
    # the burst into one trailing load (~100 ms after the last event)
    # is the difference between a smooth drag and the UI pinning the
    # CPU on every wheel detent at 10 s windows.
    RESIZE_DEBOUNCE_MS = 100
    # Backwards-compatible alias for the per-frame debounce. Tests
    # that monkey-patched the old ``LOAD_DEBOUNCE_MS`` constant still
    # work via this name.
    LOAD_DEBOUNCE_MS = SCRUB_DEBOUNCE_MS

    def __init__(self, data_source: InMemoryDecoderDataSource) -> None:
        self._data_source = data_source
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nld-viewer"
        )
        self._closed = False
        self._signals = _LoadSignals()
        self._signals.done.connect(
            self._deliver_payload, type=QtCore.Qt.QueuedConnection
        )
        # Single-shot debounce timer that drains the latest pending
        # state. The interval is set per-call (scrub vs resize) so the
        # initial value here is just a placeholder.
        self._debounce_timer = QtCore.QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._flush_pending)
        self._pending_state: ViewState | None = None
        self._pending_callback: Callable[[WindowPayload], None] | None = None
        # ``True`` while a worker job is running; new schedule calls
        # park their state in ``_pending_state`` instead of submitting
        # another job. ``_deliver_payload`` consults this when re-
        # arming the debounce.
        self._inflight = False
        # ``t_width`` of the most recently *scheduled* state. Used to
        # detect resize bursts (``state.t_width`` changed since last
        # call) and pick the long debounce. ``_in_resize_burst`` keeps
        # us on the long debounce until a flush completes, so a
        # transient same-width tick mid-drag doesn't snap-fire a load.
        self._last_scheduled_t_width: float | None = None
        self._in_resize_burst = False
        # Optional per-time arrays that the worker can skip when no
        # visible panel consumes them. Defaults to *all* loaded; the
        # viewer narrows this set when e.g. the slice overlay is
        # ``"smoothed"`` (predictive isn't needed). Loading a
        # ``predictive_posterior`` array we'll just throw away is the
        # easiest large-window cycle to recover.
        self._required_outputs: set[str] = {
            "posterior",
            "likelihood",
            "predictive",
            "state_probabilities",
            "position",
        }

    def set_required_outputs(self, outputs: set[str]) -> None:
        """Narrow the optional outputs the worker loads per window.

        ``"posterior"`` is implicitly always required (the heatmap is
        the viewer's primary panel) — passing a set that omits it is
        treated as a bug rather than a feature, so the worker still
        loads it and just doesn't read the override.
        """
        self._required_outputs = set(outputs) | {"posterior"}

    def schedule_window_load(
        self, state: ViewState, on_done: Callable[[WindowPayload], None]
    ) -> None:
        if self._closed:
            return
        # Coalesce: latest state wins. The previous pending state is
        # dropped — its ``request_id`` will never commit because
        # ``ViewerCore._handle_load_result`` checks the active
        # request_id, but more importantly the executor never even
        # sees it, saving the per-load NumPy + xarray work.
        is_resize_step = (
            self._last_scheduled_t_width is not None
            and state.t_width != self._last_scheduled_t_width
        )
        if is_resize_step:
            self._in_resize_burst = True
        self._last_scheduled_t_width = state.t_width
        self._pending_state = state
        self._pending_callback = on_done
        if self._inflight:
            # The current job's ``_deliver_payload`` will re-arm the
            # debounce after it completes.
            return
        debounce_ms = (
            self.RESIZE_DEBOUNCE_MS if self._in_resize_burst else self.SCRUB_DEBOUNCE_MS
        )
        # ``QTimer.start(ms)`` is restart-safe: if the timer is
        # already running it's restarted with the new interval, which
        # is exactly the trailing-edge-only debounce we want for
        # resize bursts.
        self._debounce_timer.start(debounce_ms)

    def _flush_pending(self) -> None:
        """Submit the latest pending state to the executor."""
        if self._closed or self._pending_state is None:
            return
        if self._inflight:
            # A debounce fire collided with a still-running job; the
            # delivery handler will pick up where we left off.
            return
        state = self._pending_state
        on_done = self._pending_callback
        self._pending_state = None
        self._pending_callback = None
        self._inflight = True
        # The drag (if any) is over once we commit a load — subsequent
        # same-width events go back to the short scrub debounce.
        self._in_resize_burst = False

        def _work() -> None:
            # ``_inflight`` MUST be cleared on completion, success or
            # failure. We always emit a signal back to the UI thread —
            # ``_deliver_payload`` clears the flag in its ``finally``
            # block. Without this, an exception in ``_build_payload``
            # would leave ``_inflight=True`` forever and the backend
            # would silently drop every subsequent ``schedule_window_load``
            # because the new state would just park in ``_pending_state``.
            try:
                payload = self._build_payload(state)
            except Exception as exc:  # noqa: BLE001 — see comment above
                if not self._closed:
                    self._signals.done.emit((None, exc))
                return
            if self._closed:
                return
            self._signals.done.emit((on_done, payload))

        self._executor.submit(_work)

    def _deliver_payload(
        self,
        result: tuple[
            Callable[[WindowPayload], None] | None, WindowPayload | BaseException
        ],
    ) -> None:
        try:
            if self._closed:
                return
            on_done, payload_or_exc = result
            if on_done is None:
                # Worker raised; ``payload_or_exc`` is the exception.
                # Log and let the UI carry on so the user can keep
                # navigating. The previous view stays on screen.
                exc = payload_or_exc
                assert isinstance(exc, BaseException)
                _LOAD_LOGGER.warning(
                    "interactive viewer window-load worker raised; "
                    "view will refresh on the next request",
                    exc_info=exc,
                )
                return
            assert not isinstance(payload_or_exc, BaseException)
            on_done(payload_or_exc)
        finally:
            self._inflight = False
            # Burst handling: if a new state arrived during the flight,
            # kick the debounce so the next pending state runs without
            # waiting for another debounce window.
            if self._pending_state is not None and not self._debounce_timer.isActive():
                # A new state arrived while the worker was busy. Use
                # the same debounce policy ``schedule_window_load``
                # would have, so a mid-flight resize burst doesn't
                # snap-fire on the trailing event.
                debounce_ms = self.RESIZE_DEBOUNCE_MS if self._in_resize_burst else 0
                self._debounce_timer.start(debounce_ms)

    def shutdown(self, *, wait: bool = True) -> None:
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def post_to_ui_thread(self, fn: Callable[[], None]) -> None:
        QtCore.QTimer.singleShot(0, fn)

    def _build_payload(self, state: ViewState) -> WindowPayload:
        sl = self._data_source.window_indices(state.t_center, state.t_width)
        time = self._data_source.time[sl]
        edges = self._data_source.time_edges
        time_start = float(edges[sl.start]) if time.size else None
        time_stop = float(edges[sl.stop]) if time.size else None
        required = self._required_outputs
        posterior = self._data_source.load_posterior(sl)
        likelihood = (
            self._data_source.load_likelihood(sl)
            if "likelihood" in required
            and "log_likelihood" in self._data_source.available_outputs
            else None
        )
        # ``predictive`` is consumed only by the slice panel's
        # ``"predictive"`` / ``"filtered"`` overlays. When the slice
        # is in ``"smoothed"`` mode we skip the load entirely — at 10 s
        # windows that's a (~330, n_state_bins) array that no panel
        # would have rendered.
        predictive = (
            self._data_source.load_predictive(sl)
            if "predictive" in required
            and "predictive_posterior" in self._data_source.available_outputs
            else None
        )
        state_probabilities = (
            self._data_source.load_state_probabilities(sl)
            if "state_probabilities" in required
            else None
        )
        position = (
            self._data_source.load_position(sl) if "position" in required else None
        )
        return WindowPayload(
            request_id=state.request_id,
            time=np.asarray(time),
            indices=sl,
            time_start=time_start,
            time_stop=time_stop,
            posterior=posterior,
            likelihood=likelihood,
            predictive=predictive,
            state_probabilities=state_probabilities,
            position=position,
        )


class QtViewer(QtWidgets.QMainWindow):
    """v1 viewer: full left-column stack (raster / state-prob / likelihood /
    posterior) + slider + optional ``extra_panels``.

    Layout top-to-bottom matches ``plot_non_local_model``: raster on
    top, state-probability lines, likelihood heatmap, posterior
    heatmap, then any user/auto extras, then the time slider.
    """

    def __init__(
        self,
        data_source: InMemoryDecoderDataSource,
        t_width: float = 1.0,
        extra_panels: list | None = None,
        extra_bin_panels: list | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("non_local_detector — interactive viewer")
        self.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
        # Ensure user-closed windows are actually destroyed (otherwise
        # they linger in QApplication.topLevelWidgets() until GC).
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self._data_source = data_source
        self._backend = QtBackendAdapter(data_source)
        self._initial_t_center: float | None = None
        self._core = ViewerCore(data_source, self._backend, t_width=t_width)
        # Remember the *clamped* width so ``R``-reset can round-trip
        # through ``set_t_width`` without re-raising on a CLI-supplied
        # zero/negative value (the core clamps silently at construction
        # but rejects out-of-range values from ``set_t_width``).
        self._initial_t_width = self._core.t_width
        self._initial_t_center = self._core.t_center

        # Auto-scroll state. Lazy-allocated timer so we don't burn a
        # QTimer slot when playback is never used. ``_autoscroll_cursor``
        # is a float absolute-time accumulator initialised at play
        # start: each tick adds ``rate / TICK_HZ`` seconds to it,
        # independent of slider quantization. Without this, sub-bin
        # ticks (e.g. default 0.05× / 30Hz ≈ 1.67ms vs a 2ms bin)
        # would round back to the same index, ``setValue`` would skip,
        # ``_core.t_center`` would never advance, and playback would
        # appear frozen on coarse time grids.
        self._autoscroll_rate = AUTOSCROLL_DEFAULT_SPEED
        self._autoscroll_timer: QtCore.QTimer | None = None
        self._autoscroll_cursor: float | None = None
        self._autoscroll_resync_lock = False
        self._pinned_time: float | None = None
        self._pinned_event_id: int | None = None
        self._initial_load_requested = False

        active_run = data_source.active_run
        detector = active_run.detector
        grid = PositionGrid.from_environment(detector.environments[0])

        # Built-in left-column view-models + panels. Order top-to-bottom
        # mirrors the static ``plot_non_local_model`` figure.
        self._raster_model = RasterModel(
            detector, active_run.spike_times, event_index=data_source.event_index
        )
        self._raster_panel = QtRasterPanel(model=self._raster_model)
        self._state_prob_model = StateProbabilityModel(detector)
        self._state_prob_panel = QtStateProbabilityPanel(model=self._state_prob_model)
        self._likelihood_model = LikelihoodHeatmapModel(detector)
        self._likelihood_panel = QtLikelihoodHeatmapPanel(
            model=self._likelihood_model, position_centers=grid.centers
        )
        self._posterior_model = PosteriorHeatmapModel(detector)
        self._panel = QtPosteriorHeatmapPanel(
            model=self._posterior_model, position_centers=grid.centers
        )
        # Top-to-bottom visual order matches
        # ``statespacecheck-paper-viewer`` (posterior + likelihood
        # heatmaps dominate the column; raster and state-probability
        # are compact rows beneath). The list order is also the
        # left-column layout add order.
        self._builtin_panels: list = [
            self._panel,
            self._likelihood_panel,
            self._raster_panel,
            self._state_prob_panel,
        ]
        # Per-panel stretch in the left column. Heatmaps get a tall
        # weight; raster + state-prob compact. ``extra_panels`` use
        # ``_LEFT_COLUMN_EXTRA_STRETCH`` (compact). See
        # ``_LEFT_COLUMN_STRETCH`` mapping below.
        self._builtin_panel_stretch: dict[int, int] = {
            id(self._panel): _LEFT_COLUMN_HEATMAP_STRETCH,
            id(self._likelihood_panel): _LEFT_COLUMN_HEATMAP_STRETCH,
            id(self._raster_panel): _LEFT_COLUMN_COMPACT_STRETCH,
            id(self._state_prob_panel): _LEFT_COLUMN_COMPACT_STRETCH,
        }

        # Right-column slice panel — per-bin readout of population
        # likelihood/posterior + predictive overlay + per-cell rows
        # for cells active in the cursor bin (and pinned cells).
        self._slice_model = SliceModel(
            detector=detector,
            spike_times=active_run.spike_times,
            time=np.asarray(active_run.results["time"].values),
            event_index=data_source.event_index,
        )
        self._slice_panel = QtSlicePanel(
            model=self._slice_model, position_centers=grid.centers
        )
        self._slice_panel.set_row_provider(self._slice_row_at)
        # Tell the backend which optional outputs the visible panels
        # actually consume so it skips per-window loads we'd just throw
        # away (e.g. ``predictive_posterior`` when the slice is in the
        # ``"smoothed"`` overlay mode). Recomputed whenever the slice
        # overlay changes; the initial sync runs here against the
        # default overlay mode.
        self._sync_required_outputs_with_panels()
        # Raster spike click → pin that spike's cell row on the slice
        # panel. The viewer owns the connection so swap can re-bind cleanly
        # (the panel references survive a swap; pin state clears via
        # ``rebind_after_swap``).
        self._raster_panel.event_clicked.connect(self._on_event_clicked)

        # User-supplied extras are owned by the caller; auto-built
        # extras get rebuilt on M-key swap. No way to tell apart
        # post-hoc (both end up as a list) so capture intent here.
        self._extra_panels_user_supplied = extra_panels is not None
        self._extra_panels: list = (
            list(extra_panels)
            if extra_panels is not None
            else _auto_panels_from_extra_metrics(active_run.extra_metrics)
        )
        self._all_panels: list = [*self._builtin_panels, *self._extra_panels]

        # Bin-synced plugins are a separate lane: caller-owned (always
        # treated as user-supplied), driven via the same buffered
        # set_window_buffer + update_for_index path as ``QtSlicePanel``.
        # No auto-build path here — bin-synced rendering depends on
        # project-specific data (e.g. video frames, metric scalars).
        self._extra_bin_panels: list = (
            list(extra_bin_panels) if extra_bin_panels is not None else []
        )

        # Slider: integer indices into the time grid; map to t_center.
        n_time = data_source.n_time
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(n_time - 1)
        self._slider.setValue(n_time // 2)
        self._slider.valueChanged.connect(self._on_slider_value_changed)

        # Controls bar: overlay-selector dropdown + per-overlay
        # visibility checkboxes. Hidden when the bundle has no
        # overlays — keeps the window clean for the common case.
        self._controls_bar = self._build_controls_bar()

        # Two-column body inside the root vertical layout.
        # Left column: built-in time-axis panels + extras.
        # Right column: slice panel + bin-synced extras.
        # Slider stretches across the full width below both columns.
        # Body uses a ``QSplitter`` so the user can drag the divider
        # to rebalance the columns; default stretch factors mirror
        # statespacecheck-paper-viewer (~70/30).
        self._left_column_layout = QtWidgets.QVBoxLayout()
        self._left_column_layout.setContentsMargins(
            _COLUMN_MARGIN, _COLUMN_MARGIN, _COLUMN_MARGIN, _COLUMN_MARGIN
        )
        self._left_column_layout.setSpacing(_COLUMN_SPACING)
        for panel in self._builtin_panels:
            stretch = self._builtin_panel_stretch[id(panel)]
            self._left_column_layout.addWidget(panel, stretch=stretch)
        self._extras_insert_index = self._left_column_layout.count()
        for extra in self._extra_panels:
            self._left_column_layout.addWidget(
                extra, stretch=_LEFT_COLUMN_EXTRA_STRETCH
            )
        left_column = QtWidgets.QWidget()
        left_column.setLayout(self._left_column_layout)

        # Right column: slice panel on top, extra bin-synced plugins
        # below it. Top-aligned with a trailing stretch so the slice
        # panel and plugins keep their natural heights instead of
        # filling the full window — matches the paper-viewer aesthetic
        # where the right column sits at the top with empty space below.
        self._right_column_layout = QtWidgets.QVBoxLayout()
        self._right_column_layout.setContentsMargins(
            _COLUMN_MARGIN, _COLUMN_MARGIN, _COLUMN_MARGIN, _COLUMN_MARGIN
        )
        self._right_column_layout.setSpacing(_SLICE_COLUMN_SPACING)
        self._right_column_layout.addWidget(
            self._slice_panel, stretch=_RIGHT_COLUMN_SLICE_STRETCH
        )
        for bin_panel in self._extra_bin_panels:
            self._right_column_layout.addWidget(bin_panel, stretch=0)
        self._right_column_layout.addStretch(_RIGHT_COLUMN_TRAILING_STRETCH)
        right_column = QtWidgets.QWidget()
        right_column.setLayout(self._right_column_layout)

        self._body_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._body_splitter.addWidget(left_column)
        self._body_splitter.addWidget(right_column)
        self._body_splitter.setStretchFactor(0, _BODY_SPLITTER_LEFT_STRETCH)
        self._body_splitter.setStretchFactor(1, _BODY_SPLITTER_RIGHT_STRETCH)
        # ``setStretchFactor`` only resolves the ratio when both child
        # widgets have non-trivial size policies — in practice with
        # Expanding panels we still see a near-50/50 initial split.
        # Force the desired ratio with explicit pixel-equivalent
        # ``setSizes``; the user can drag from there.
        self._body_splitter.setSizes(
            [
                _BODY_SPLITTER_LEFT_STRETCH * 100,
                _BODY_SPLITTER_RIGHT_STRETCH * 100,
            ]
        )
        # Don't allow either pane to fully collapse on drag; small
        # minimum keeps the column dragable but always visible.
        self._body_splitter.setChildrenCollapsible(False)

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.setContentsMargins(
            _OUTER_MARGIN, _OUTER_MARGIN, _OUTER_MARGIN, _OUTER_MARGIN
        )
        self._layout.setSpacing(_OUTER_SPACING)
        self._layout.addWidget(self._body_splitter, stretch=1)
        self._layout.addWidget(self._controls_bar, stretch=0)
        container = QtWidgets.QWidget()
        container.setLayout(self._layout)
        self.setCentralWidget(container)

        self._wire_panels(self._all_panels)

        self._core.on_window_loaded(self._on_window_loaded)
        self._core.on_active_run_changed(self._rebind_panels)
        self._core.on_t_center_changed(self._sync_autoscroll_cursor_to_core)
        self._core.on_t_center_changed(self._sync_cursor_markers)
        self._core.refresh_overlays()
        # Initial cursor-marker placement — t_center_changed only
        # fires on subsequent moves, not on construction.
        self._sync_cursor_markers(self._core.t_center)

        # Keyboard shortcuts. ``[`` / ``]`` shrink/grow the window
        # width; ``Shift+Left`` / ``Shift+Right`` step a full window
        # at a time; ``R`` resets center + width.
        for key_seq, fn in (
            (QtGui.QKeySequence(QtCore.Qt.Key_Left), self._step_left),
            (QtGui.QKeySequence(QtCore.Qt.Key_Right), self._step_right),
            (QtGui.QKeySequence("Shift+Left"), lambda: self._step_window(-1)),
            (QtGui.QKeySequence("Shift+Right"), lambda: self._step_window(+1)),
            (QtGui.QKeySequence("["), lambda: self._scale_t_width(0.5)),
            (QtGui.QKeySequence("]"), lambda: self._scale_t_width(2.0)),
            (QtGui.QKeySequence("R"), self._reset_view),
            (QtGui.QKeySequence("N"), self._core.next_event),
            (QtGui.QKeySequence("Shift+N"), self._core.prev_event),
            (QtGui.QKeySequence("Escape"), self._clear_pins),
            (QtGui.QKeySequence("M"), self._cycle_model),
            (QtGui.QKeySequence("Space"), self._toggle_play),
            (QtGui.QKeySequence(","), lambda: self._step_speed(-1)),
            (QtGui.QKeySequence("."), lambda: self._step_speed(+1)),
        ):
            shortcut = QtGui.QShortcut(key_seq, self)
            shortcut.activated.connect(fn)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt naming convention
        """Kick off the first async window load once the viewer is shown."""
        super().showEvent(event)
        if self._initial_load_requested:
            return
        self._initial_load_requested = True
        QtCore.QTimer.singleShot(0, self._core.request_load)

    def _wire_panels(self, panels: list) -> None:
        """Wire panels into the core (overlays, click, x-link, wheel).

        ``self._panel`` (posterior) is the canonical x-link anchor.
        """
        link_target = self._panel.x_link_target()
        for panel in panels:
            self._core.on_overlays_changed(panel.set_event_overlays)
            handler = getattr(panel, "click_handler", None)
            if handler is not None:
                handler(self._set_t_center_from_panel_click)
            target = getattr(panel, "x_link_target", lambda: None)()
            if target is not None and target is not link_target:
                target.setXLink(link_target)
            viewport = getattr(panel, "viewport", lambda: None)()
            if viewport is not None:
                viewport.installEventFilter(self)

    def _build_controls_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        overlays = self._data_source.active_run.event_overlays
        run_names = self._data_source.run_names
        multi_run = len(run_names) > 1

        layout.addWidget(QtWidgets.QLabel("Center time:"))
        layout.addWidget(self._slider, stretch=1)
        self._time_label = QtWidgets.QLabel(self._format_time_label())
        self._time_label.setMinimumWidth(260)
        layout.addWidget(self._time_label)
        layout.addSpacing(12)

        layout.addWidget(QtWidgets.QLabel("Window:"))
        self._window_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._window_slider.setRange(0, WINDOW_SLIDER_RESOLUTION)
        self._window_slider.setValue(self._window_slider_value_for(self._core.t_width))
        self._window_slider.setMaximumWidth(140)
        self._window_slider.valueChanged.connect(self._on_window_slider_changed)
        layout.addWidget(self._window_slider)
        self._window_label = QtWidgets.QLabel(self._format_window_label())
        self._window_label.setMinimumWidth(70)
        layout.addWidget(self._window_label)
        layout.addSpacing(12)

        self._per_cell_checkbox = QtWidgets.QCheckBox("Per-cell rows")
        self._per_cell_checkbox.setChecked(True)
        self._per_cell_checkbox.toggled.connect(self._slice_panel.set_per_cell_visible)
        layout.addWidget(self._per_cell_checkbox)
        layout.addSpacing(12)

        layout.addWidget(QtWidgets.QLabel("Slice overlay:"))
        self._slice_overlay_combo = QtWidgets.QComboBox()
        for mode_key, mode_label in _SLICE_OVERLAY_CHOICES:
            self._slice_overlay_combo.addItem(mode_label, userData=mode_key)
        for i in range(self._slice_overlay_combo.count()):
            if self._slice_overlay_combo.itemData(i) == self._slice_panel.overlay_mode:
                self._slice_overlay_combo.setCurrentIndex(i)
                break
        self._slice_overlay_combo.currentIndexChanged.connect(
            self._on_slice_overlay_changed
        )
        layout.addWidget(self._slice_overlay_combo)
        layout.addSpacing(12)

        # Play / pause + speed are always present — auto-scroll is a
        # universal affordance, not gated on overlays / multi-run. The
        # combo's ``itemData`` carries the float multiplier directly so
        # the per-tick path doesn't have to reparse the display label.
        self._play_button = QtWidgets.QToolButton()
        self._play_button.setText("▶")
        self._play_button.setToolTip("Play / pause auto-scroll (Space)")
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self._on_play_toggled)
        layout.addWidget(self._play_button)

        layout.addWidget(QtWidgets.QLabel("Speed (,/.):"))
        self._speed_combo = QtWidgets.QComboBox()
        for speed in AUTOSCROLL_SPEED_OPTIONS:
            self._speed_combo.addItem(_format_speed(speed), userData=speed)
        default_idx = AUTOSCROLL_SPEED_OPTIONS.index(AUTOSCROLL_DEFAULT_SPEED)
        self._speed_combo.setCurrentIndex(default_idx)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_combo_changed)
        layout.addWidget(self._speed_combo)
        layout.addSpacing(12)

        # Model-selector dropdown (M-key cycles). Only present when
        # multiple runs are loaded; the M-key path goes *through* the
        # combo so user-clicks and keyboard cycling share one signal
        # path (combo.currentIndexChanged → core.set_active_run).
        self._model_combo: QtWidgets.QComboBox | None = None
        if multi_run:
            layout.addWidget(QtWidgets.QLabel("Model (M):"))
            self._model_combo = QtWidgets.QComboBox()
            for name in run_names:
                self._model_combo.addItem(name, userData=name)
            self._model_combo.setCurrentText(self._data_source.active_run_name)
            self._model_combo.currentIndexChanged.connect(self._on_active_run_changed)
            layout.addWidget(self._model_combo)
            layout.addSpacing(12)

        if overlays:
            # Overlay-selector dropdown picks the navigator target.
            layout.addWidget(QtWidgets.QLabel("Overlay (N/Shift+N):"))
            self._overlay_combo = QtWidgets.QComboBox()
            self._overlay_combo.addItem("(none)", userData=None)
            for ovl in overlays:
                self._overlay_combo.addItem(ovl.name, userData=ovl.name)
            self._overlay_combo.currentIndexChanged.connect(
                self._on_active_overlay_changed
            )
            layout.addWidget(self._overlay_combo)
            # Per-overlay visibility checkboxes.
            layout.addSpacing(12)
            layout.addWidget(QtWidgets.QLabel("Visible:"))
            self._overlay_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
            for ovl in overlays:
                cb = QtWidgets.QCheckBox(ovl.name)
                cb.setChecked(True)
                cb.toggled.connect(
                    lambda checked, name=ovl.name: self._core.set_overlay_visibility(
                        name, checked
                    )
                )
                layout.addWidget(cb)
                self._overlay_checkboxes[ovl.name] = cb
        layout.addStretch(1)
        return bar

    def _on_active_overlay_changed(self, index: int) -> None:
        name = self._overlay_combo.itemData(index)
        self._core.set_active_overlay(name)

    def _on_slice_overlay_changed(self, index: int) -> None:
        mode = self._slice_overlay_combo.itemData(index)
        if mode is None:
            return
        self._slice_panel.set_overlay_mode(mode)
        self._sync_required_outputs_with_panels()
        # Switching from ``"smoothed"`` to ``"predictive"`` /
        # ``"filtered"`` widens ``_required_outputs`` to include
        # ``predictive_posterior``, but the panel still holds the old
        # buffered payload (where ``predictive`` is ``None``) so the
        # overlay would render blank until the user nudged the slider
        # or t_width. Force a same-window reload so newly-required
        # outputs land in the buffer immediately. ``set_t_center`` /
        # ``set_t_width`` self-no-op on unchanged values, so a direct
        # ``request_load`` is the only way to re-fetch the same window.
        self._core.request_load()

    def _sync_required_outputs_with_panels(self) -> None:
        """Tell the backend which optional outputs panels actually use.

        ``predictive_posterior`` is the one big array we can drop on a
        per-window basis: only the slice panel reads it, and only in
        ``"predictive"`` / ``"filtered"`` overlay modes. ``"smoothed"``
        renders the posterior row instead, so loading
        ``predictive_posterior`` is wasted work — large windows
        materialise a ``(n_visible, n_state_bins)`` array we'd
        immediately throw away.
        """
        outputs = {"posterior", "likelihood", "state_probabilities", "position"}
        if self._slice_panel.overlay_mode in {"predictive", "filtered"}:
            outputs.add("predictive")
        # ``BackendAdapter`` is the abstract protocol; the QtViewer
        # always builds a ``QtBackendAdapter`` so the cast is safe and
        # keeps non-Qt backends from inheriting the override.
        if isinstance(self._backend, QtBackendAdapter):
            self._backend.set_required_outputs(outputs)

    def _on_active_run_changed(self, index: int) -> None:
        if self._model_combo is None:
            return
        name = self._model_combo.itemData(index)
        if name is None or name == self._data_source.active_run_name:
            return
        self._core.set_active_run(name)

    def _cycle_model(self) -> None:
        """M-key handler: advance to the next run in the dropdown order.

        Single-run viewers have no model dropdown, so this is a no-op
        in that case.
        """
        if self._model_combo is None:
            return
        next_index = (self._model_combo.currentIndex() + 1) % self._model_combo.count()
        self._model_combo.setCurrentIndex(next_index)

    # --------------------------------------------------------------
    # Auto-scroll
    # --------------------------------------------------------------

    def _on_play_toggled(self, on: bool) -> None:
        if on:
            self._start_autoscroll()
            self._play_button.setText("⏸")
        else:
            self._stop_autoscroll()
            self._play_button.setText("▶")

    def _on_speed_combo_changed(self, index: int) -> None:
        speed = self._speed_combo.itemData(index)
        if speed is not None:
            self._autoscroll_rate = float(speed)

    def _toggle_play(self) -> None:
        """Space-key handler — flip the play button's checked state."""
        self._play_button.toggle()

    def _step_speed(self, delta: int) -> None:
        """``,`` / ``.`` — step through ``AUTOSCROLL_SPEED_OPTIONS``."""
        new_idx = max(
            0,
            min(
                self._speed_combo.count() - 1, self._speed_combo.currentIndex() + delta
            ),
        )
        if new_idx != self._speed_combo.currentIndex():
            self._speed_combo.setCurrentIndex(new_idx)

    def _format_time_label(self) -> str:
        rel = self._core.t_center - float(self._data_source.time[0])
        return (
            f"t={self._core.t_center:.3f}  "
            f"({rel:.2f} s into session, w={self._core.t_width:.2f} s)"
        )

    def _format_window_label(self) -> str:
        return f"{self._core.t_width:.2f} s"

    def _window_slider_value_for(self, window_seconds: float) -> int:
        w = float(np.clip(window_seconds, MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS))
        log_min = np.log10(MIN_WINDOW_SECONDS)
        log_max = np.log10(MAX_WINDOW_SECONDS)
        frac = (np.log10(w) - log_min) / (log_max - log_min)
        return int(round(frac * WINDOW_SLIDER_RESOLUTION))

    def _window_seconds_for_slider(self, value: int) -> float:
        frac = value / WINDOW_SLIDER_RESOLUTION
        log_min = np.log10(MIN_WINDOW_SECONDS)
        log_max = np.log10(MAX_WINDOW_SECONDS)
        return float(10 ** (log_min + frac * (log_max - log_min)))

    def _sync_control_labels(self) -> None:
        if hasattr(self, "_time_label"):
            self._time_label.setText(self._format_time_label())
        if hasattr(self, "_window_label"):
            self._window_label.setText(self._format_window_label())
        if hasattr(self, "_window_slider"):
            with QtCore.QSignalBlocker(self._window_slider):
                self._window_slider.setValue(
                    self._window_slider_value_for(self._core.t_width)
                )

    def _on_window_slider_changed(self, value: int) -> None:
        self._core.set_t_width(self._window_seconds_for_slider(value))
        self._sync_control_labels()

    def _start_autoscroll(self) -> None:
        if self._autoscroll_timer is not None:
            return
        # Initialise the float playback cursor from the current center
        # so playback continues from wherever the user left the slider.
        self._autoscroll_cursor = float(self._core.t_center)
        timer = QtCore.QTimer(self)
        timer.setInterval(int(round(1000.0 / AUTOSCROLL_TICK_HZ)))
        timer.timeout.connect(self._autoscroll_tick)
        self._autoscroll_timer = timer
        timer.start()

    def _stop_autoscroll(self) -> None:
        if self._autoscroll_timer is None:
            return
        self._autoscroll_timer.stop()
        self._autoscroll_timer.deleteLater()
        self._autoscroll_timer = None
        self._autoscroll_cursor = None

    def _autoscroll_tick(self) -> None:
        """One tick of playback. Accumulates ``rate / TICK_HZ`` seconds
        into the float playback cursor and only updates the slider
        when the cursor crosses a bin boundary. The slider remains
        the single source of truth for ``_core.t_center`` + slice
        updates (its ``valueChanged`` wires both)."""
        if self._autoscroll_cursor is None:
            return
        dt = self._autoscroll_rate / AUTOSCROLL_TICK_HZ
        self._autoscroll_cursor += dt
        new_t = self._autoscroll_cursor
        time = self._data_source.time
        t_max = float(time[-1])
        if new_t >= t_max:
            # Reached end of session — auto-pause. Setting checked=False
            # toggles the button which fires ``_on_play_toggled(False)``
            # and stops/destroys the timer.
            if self._play_button.isChecked():
                self._play_button.setChecked(False)
            return
        new_idx = int(np.searchsorted(time, new_t, side="right") - 1)
        new_idx = max(0, min(len(time) - 1, new_idx))
        if self._slider.value() != new_idx:
            # Tick-driven slider update — block the resync path so
            # ``_on_slider_value_changed`` doesn't snap our float
            # cursor back to ``time[new_idx]`` and lose the sub-bin
            # accumulation we just built up.
            self._autoscroll_resync_lock = True
            try:
                self._slider.setValue(new_idx)
            finally:
                self._autoscroll_resync_lock = False

    def _on_window_loaded(self, payload) -> None:
        for panel in self._all_panels:
            panel.update_window(payload)
        # Pin the link-target's viewbox X range to the loaded window.
        # Two reasons we don't rely on pyqtgraph's autoRange:
        #   (1) ``ImageItem.setRect`` doesn't update the viewbox
        #       autoRange bounds — the heatmap stays at its default
        #       empty [-0.5, 0.5] X range until told otherwise.
        #   (2) The other built-in panels are ``setXLink``'d to the
        #       posterior so they inherit its (broken) range.
        # Setting the range here lets every X-linked panel render
        # against the actual loaded window.
        if payload.time.size:
            t_start = (
                float(payload.time_start)
                if payload.time_start is not None
                else float(payload.time[0])
            )
            t_stop = (
                float(payload.time_stop)
                if payload.time_stop is not None
                else float(payload.time[-1])
            )
            if t_stop > t_start:
                self._panel.getPlotItem().vb.setXRange(t_start, t_stop, padding=0)
        slider_value = self._slider.value()
        for bin_panel in (self._slice_panel, *self._extra_bin_panels):
            bin_panel.set_window_buffer(payload)
            # Re-render at the current cursor — the new buffer may
            # extend coverage past the cursor's previous reach.
            bin_panel.update_for_index(slider_value)

    def _slice_row_at(self, t_idx: int):
        """Single-row fallback for slice ticks outside the buffered window.

        Returns log-likelihood only; the slice panel exponentiates one
        row at render time when the filtered overlay is active, so we
        never pay the full-window exp() cost on either path.
        """
        if t_idx < 0 or t_idx >= self._data_source.n_time:
            return None
        posterior_row = self._data_source.slice_at_index(t_idx, which="posterior")
        if posterior_row is None:
            return None
        log_lik_row = (
            self._data_source.slice_at_index(t_idx, which="likelihood")
            if "log_likelihood" in self._data_source.available_outputs
            else None
        )
        predictive_row = (
            self._data_source.slice_at_index(t_idx, which="predictive")
            if "predictive_posterior" in self._data_source.available_outputs
            else None
        )
        position = self._data_source.load_position(slice(t_idx, t_idx + 1))
        true_position = (
            float(position[0]) if position is not None and position.size else None
        )
        return (
            np.asarray(posterior_row),
            np.asarray(log_lik_row) if log_lik_row is not None else None,
            np.asarray(predictive_row) if predictive_row is not None else None,
            true_position,
        )

    def _step_window(self, direction: int) -> None:
        self._clear_pins()
        new_t = self._core.t_center + direction * self._core.t_width
        self._core.set_t_center(new_t)
        self._sync_slider_and_bin_panels_for_time(new_t)

    def _step_left(self) -> None:
        self._clear_pins()
        self._core.step_left()
        self._sync_slider_and_bin_panels_for_time(self._core.t_center)

    def _step_right(self) -> None:
        self._clear_pins()
        self._core.step_right()
        self._sync_slider_and_bin_panels_for_time(self._core.t_center)

    def _scale_t_width(self, factor: float) -> None:
        # ``ViewerCore.set_t_width`` clamps to ``[MIN, MAX]_T_WIDTH``;
        # the explicit clamp here is just so the keyboard halve/double
        # bottoms out at the same floor (otherwise repeated ``[`` would
        # propose 2^-N seconds and only get clamped once it reached
        # the core, leaving the next ``[`` to halve the clamped value).
        new_width = float(
            np.clip(
                self._core.t_width * factor, MIN_T_WIDTH_SECONDS, MAX_T_WIDTH_SECONDS
            )
        )
        self._core.set_t_width(new_width)
        self._sync_control_labels()

    def _reset_view(self) -> None:
        self._clear_pins()
        if self._initial_t_center is not None:
            self._core.set_t_center(self._initial_t_center)
            self._sync_slider_and_bin_panels_for_time(self._initial_t_center)
        self._core.set_t_width(self._initial_t_width)
        self._sync_control_labels()

    def eventFilter(self, obj, event) -> bool:
        # Wheel over a time-axis panel scrubs the window width.
        if event.type() == QtCore.QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta != 0:
                factor = 0.9 if delta > 0 else 1.1
                self._scale_t_width(factor)
                return True
        return super().eventFilter(obj, event)

    @property
    def core(self) -> ViewerCore:
        return self._core

    def _on_slider_value_changed(self, value: int) -> None:
        time = self._data_source.time
        if not self._autoscroll_resync_lock:
            self._clear_pins()
        # Calling ``core.set_t_center`` fires
        # ``_sync_autoscroll_cursor_to_core``, which handles the
        # resync (lock-gated against tick-driven setValue).
        self._core.set_t_center(float(time[value]))
        self._sync_control_labels()
        # Drive the slice panel + extra bin-synced plugins synchronously
        # off the slider so per-tick cursor updates land sub-ms (the
        # heavier window load is async and refreshes the buffer when it
        # commits).
        for bin_panel in (self._slice_panel, *self._extra_bin_panels):
            bin_panel.update_for_index(value)

    def _set_t_center_from_panel_click(self, t: float) -> None:
        self._clear_pins()
        self._core.set_t_center(float(t))
        self._sync_slider_and_bin_panels_for_time(float(t))

    def _time_to_bin_index(self, t: float) -> int:
        """Return the left-edge-binned index containing absolute time ``t``."""
        time = self._data_source.time
        if time.size <= 1:
            return 0
        idx = int(np.searchsorted(time, float(t), side="right") - 1)
        return max(0, min(time.size - 1, idx))

    def _sync_slider_and_bin_panels_for_time(self, t: float) -> int:
        """Update slider + bin-synced panels without re-entering slider handlers."""
        t_idx = self._time_to_bin_index(t)
        with QtCore.QSignalBlocker(self._slider):
            self._slider.setValue(t_idx)
        self._sync_control_labels()
        for bin_panel in (self._slice_panel, *self._extra_bin_panels):
            bin_panel.update_for_index(t_idx)
        return t_idx

    def _sync_cursor_markers(self, new_t_center: float) -> None:
        """Push ``(t_center, t_lo, t_hi)`` to every TimeAxisPanel.

        ``t_center`` is the dashed-line position; ``[t_lo, t_hi]`` is
        the active-bin band. Bin edges use the same left-edge
        convention as ``SliceModel``. Built-in panels all mix in
        ``CursorMarkersMixin``; user-supplied ``extra_panels`` opt in
        by exposing a ``set_cursor_markers`` method (called via
        ``getattr`` so the kwarg stays backward-compatible).
        """
        t_idx = self._time_to_bin_index(new_t_center)
        t_lo, t_hi = self._bin_edges_at(t_idx)
        for panel in self._builtin_panels:
            panel.set_cursor_markers(new_t_center, t_lo, t_hi)
        for panel in self._extra_panels:
            setter = getattr(panel, "set_cursor_markers", None)
            if setter is not None:
                setter(new_t_center, t_lo, t_hi)

    def _bin_edges_at(self, t_idx: int) -> tuple[float, float]:
        """Return left-edge ``(t_lo, t_hi)`` for active bin ``t_idx``."""
        return bin_edges_at(self._data_source.time, t_idx)

    def _sync_autoscroll_cursor_to_core(self, new_t_center: float) -> None:
        """Re-anchor the float playback cursor to ``new_t_center``.

        Subscribed to ``ViewerCore.on_t_center_changed`` so every
        navigation path that recenters during play (slider drag,
        Shift+Left/Right step-window, R reset, Left/Right step,
        click-to-recenter, N/Shift+N event navigator) brings the
        playback cursor along. The lock suppresses this for
        tick-driven setValue → set_t_center → callback → here, so
        sub-bin accumulation in the cursor isn't snapped back to
        the slider's quantized time on every tick.
        """
        if self._autoscroll_cursor is not None and not self._autoscroll_resync_lock:
            self._autoscroll_cursor = float(new_t_center)

    def _rebind_panels(self, _new_run_name: str) -> None:
        """Rebind all panels to the new active run's detector.

        Fires *before* the new load is dispatched so payload collapse
        runs under the new schema. Also syncs the model dropdown so a
        programmatic ``core.set_active_run(...)`` keeps the UI in
        lockstep (signal blocker prevents the combo's
        ``currentIndexChanged`` from firing back into ``set_active_run``).
        """
        if self._model_combo is not None:
            with QtCore.QSignalBlocker(self._model_combo):
                self._model_combo.setCurrentText(_new_run_name)
        new_run = self._data_source.active_run
        new_detector = new_run.detector
        grid = PositionGrid.from_environment(new_detector.environments[0])

        self._posterior_model.set_active_run(new_detector)
        self._panel.set_position_centers(grid.centers)

        self._likelihood_model.set_active_run(new_detector)
        self._likelihood_panel.set_position_centers(grid.centers)

        self._state_prob_model.set_active_run(new_detector)
        self._state_prob_panel.rebind_after_swap()

        self._raster_model.set_active_run(
            new_detector, new_run.spike_times, event_index=self._data_source.event_index
        )
        self._raster_panel.rebind_after_swap()

        self._slice_model.set_active_run(
            new_detector,
            new_run.spike_times,
            np.asarray(new_run.results["time"].values),
            event_index=self._data_source.event_index,
        )
        self._slice_panel.set_position_centers(grid.centers)
        self._slice_panel.rebind_after_swap()

        # Bin-synced plugins drop run-local caches if they expose
        # ``rebind_after_swap``. The hook is optional in the protocol
        # so plugins without run-local state need not implement it.
        for bin_panel in self._extra_bin_panels:
            rebind = getattr(bin_panel, "rebind_after_swap", None)
            if callable(rebind):
                rebind()

        if not self._extra_panels_user_supplied:
            self._rebuild_auto_extras(new_run.extra_metrics)
        self._pinned_time = None
        self._pinned_event_id = None
        self._refresh_pin_markers()

    def _rebuild_auto_extras(self, extra_metrics: dict) -> None:
        """Tear down auto-built extras and rebuild from new ``extra_metrics``.

        Overlay callbacks must be unregistered *before* ``deleteLater``
        so a queued ``_dispatch_overlays`` can't call into a widget the
        runtime has scheduled for deletion.
        """
        for old_panel in self._extra_panels:
            self._core.off_overlays_changed(old_panel.set_event_overlays)
            self._left_column_layout.removeWidget(old_panel)
            old_panel.setParent(None)
            old_panel.deleteLater()

        new_extras = _auto_panels_from_extra_metrics(extra_metrics)
        for offset, panel in enumerate(new_extras):
            self._left_column_layout.insertWidget(
                self._extras_insert_index + offset,
                panel,
                _LEFT_COLUMN_EXTRA_STRETCH,
            )

        self._extra_panels = new_extras
        self._all_panels = [*self._builtin_panels, *self._extra_panels]
        self._wire_panels(new_extras)

    def _on_event_clicked(self, event_id: int) -> None:
        event_id = int(event_id)
        if event_id == self._pinned_event_id:
            self._clear_pins()
            return
        event = self._data_source.spike_event_at(event_id)
        self._slice_panel.clear_pins()
        self._slice_panel.pin_cell(event.cell_id)
        self._pinned_event_id = event_id
        self._pinned_time = event.time
        self._refresh_pin_markers()
        self._core.set_t_center(event.time)
        self._sync_slider_and_bin_panels_for_time(event.time)

    def _on_spike_clicked(self, cell_id: int, t: float) -> None:
        """Compatibility shim for tests/plugins still emitting cell/time.

        ``t`` from a pyqtgraph float32 spot position won't survive
        strict equality against the float64 spike-time table, so the
        shim uses ``nearest_event_id_for_cell_time`` (atol=1e-6).
        Logs the miss so a no-op pin from a stale plugin shows up in
        the log instead of silently failing.
        """
        cell_id = int(cell_id)
        t = float(t)
        event_id = self._data_source.event_index.nearest_event_id_for_cell_time(
            cell_id, t
        )
        if event_id is None:
            _LOAD_LOGGER.debug(
                "spike_clicked(cell_id=%d, t=%g) did not match any event "
                "(no event within 1e-6 of t for this cell)",
                cell_id,
                t,
            )
            return
        self._on_event_clicked(event_id)

    def _clear_pins(self) -> None:
        self._slice_panel.clear_pins()
        self._pinned_time = None
        self._pinned_event_id = None
        self._refresh_pin_markers()

    def _refresh_pin_markers(self) -> None:
        pinned_cell_id = (
            self._data_source.spike_event_at(self._pinned_event_id).cell_id
            if self._pinned_event_id is not None
            else None
        )
        for panel in (*self._builtin_panels, *self._extra_panels):
            if panel is self._raster_panel:
                self._raster_panel.set_spike_pin_marker(
                    self._pinned_time,
                    pinned_cell_id,
                )
                continue
            setter = getattr(panel, "set_pin_marker", None)
            if setter is not None:
                setter(self._pinned_time)

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming convention
        """Drop self from the live-viewer registry on close."""
        self._backend.shutdown()
        try:
            _LIVE_VIEWERS.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)


# Keeps non-blocking ``launch_qt`` viewers alive. Without this, a
# ``QMainWindow`` with no retained Python reference is GC'd by
# PySide6 and disappears immediately after ``launch_qt(block=False)``
# returns. The blocking path doesn't strictly need it (the event
# loop pins the window), but we register both branches uniformly so
# `closeEvent` cleanup is symmetrical.
_LIVE_VIEWERS: list[QtViewer] = []


def _auto_panels_from_extra_metrics(extra_metrics: dict) -> list:
    """Build series panels from ``RunBundle.extra_metrics`` entries.

    Each ``MetricSpec`` lifts to the matching panel class. A
    ``pd.Series`` lifts to a default ``LineSeriesPanel`` named after
    the dict key.
    """
    import pandas as pd  # type: ignore[import-untyped]

    panels: list = []
    for name, value in extra_metrics.items():
        if isinstance(value, MetricSpec):
            panels.append(_panel_for_metric_spec(value))
        elif isinstance(value, pd.Series):
            t = np.asarray(value.index, dtype=float)
            y = np.asarray(value.values, dtype=float)
            panels.append(LineSeriesPanel(LineSeriesModel(name=name, t=t, y=y)))
        else:
            raise TypeError(
                f"extra_metrics[{name!r}] must be a MetricSpec or pd.Series; "
                f"got {type(value).__name__}."
            )
    return panels


# MetricSpec.kind → (model class, panel class). The single source of
# truth for "which Qt panel renders which spec kind"; adding a new
# kind means appending one entry here. ``MetricSpec.__post_init__``
# already validates ``kind`` against the same set of strings.
_METRIC_SPEC_DISPATCH: dict[str, tuple[type, type]] = {
    "line": (LineSeriesModel, LineSeriesPanel),
    "scatter": (ScatterSeriesModel, ScatterSeriesPanel),
    "intervals": (IntervalSeriesModel, IntervalSeriesPanel),
}


def _panel_for_metric_spec(spec: MetricSpec):
    """Lift a ``MetricSpec`` to the matching Qt panel."""
    try:
        model_cls, panel_cls = _METRIC_SPEC_DISPATCH[spec.kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown MetricSpec.kind: {spec.kind!r}; expected one of "
            f"{sorted(_METRIC_SPEC_DISPATCH)!r}"
        ) from exc
    return panel_cls(model_cls.from_metric_spec(spec))


def launch_qt_with_source(
    data_source: InMemoryDecoderDataSource,
    t_width: float = 1.0,
    block: bool = True,
    extra_panels: list | None = None,
    extra_bin_panels: list | None = None,
) -> int:
    """Open a ``QtViewer`` against a pre-built data source.

    Bypasses the eager-bundle path so callers (the CLI in particular)
    can hand in an ``InMemoryDecoderDataSource`` whose ``RunBundle``
    results Datasets may be either eager (NetCDF) or zarr-backed (via
    the optional ``results.zarr/`` cache validated by
    ``load_zarr_cache_or_fall_back``). ``launch_qt`` continues to wrap
    this for the ``RunBundle`` / dict-of-bundles convenience case.
    """
    app = _ensure_qapplication()
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    viewer = QtViewer(
        data_source,
        t_width=t_width,
        extra_panels=extra_panels,
        extra_bin_panels=extra_bin_panels,
    )
    _LIVE_VIEWERS.append(viewer)
    viewer.show()
    if block:
        return int(app.exec())
    return 0


def launch_qt(
    bundles: RunBundle | dict[str, RunBundle],
    t_width: float = 1.0,
    block: bool = True,
    extra_panels: list | None = None,
    extra_bin_panels: list | None = None,
) -> int:
    """Open a ``QtViewer`` against the supplied bundle(s).

    Parameters
    ----------
    bundles : RunBundle | dict[str, RunBundle]
        Single bundle or named multi-run dict.
    t_width : float, optional
        Initial window width in seconds.
    block : bool, optional
        If True (default), call ``QApplication.exec()`` and block
        until the window closes; returns the exit code. If False,
        creates the window and returns immediately — useful for
        tests and notebook-driven inspection. The viewer is kept
        alive in a module-level registry until the user closes the
        window (otherwise PySide6 would GC it the moment this
        function returns).

    Returns
    -------
    int
        Exit code from ``QApplication.exec()`` when ``block=True``;
        ``0`` otherwise.
    """
    if isinstance(bundles, RunBundle):
        data_source = InMemoryDecoderDataSource.from_single(bundles)
    else:
        data_source = InMemoryDecoderDataSource(bundles)
    return launch_qt_with_source(
        data_source,
        t_width=t_width,
        block=block,
        extra_panels=extra_panels,
        extra_bin_panels=extra_bin_panels,
    )
