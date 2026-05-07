"""Qt frontend: ``QApplication`` ownership, ``QtBackendAdapter``, ``QtViewer``.

This module is the only place in the codebase that creates a
``QApplication`` and the only Qt-importing entry under
``viewer/``. ``app.py`` lazy-imports ``launch_qt`` here.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

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
from non_local_detector.visualization.interactive.viewer.core import ViewerCore

if TYPE_CHECKING:
    pass


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


def _format_speed(speed: float) -> str:
    """Render a multiplier as ``"1×"`` / ``"2×"`` / ``"0.05×"`` etc."""
    if speed >= 1.0 and float(speed).is_integer():
        return f"{speed:.0f}×"
    return f"{speed:.2g}×"


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

    def __init__(self, data_source: InMemoryDecoderDataSource) -> None:
        self._data_source = data_source
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nld-viewer")
        self._closed = False
        self._signals = _LoadSignals()
        self._signals.done.connect(
            self._deliver_payload, type=QtCore.Qt.QueuedConnection
        )

    def schedule_window_load(
        self, state: ViewState, on_done: Callable[[WindowPayload], None]
    ) -> None:
        if self._closed:
            return

        def _work() -> None:
            payload = self._build_payload(state)
            if self._closed:
                return
            self._signals.done.emit((on_done, payload))

        self._executor.submit(_work)

    def _deliver_payload(
        self, result: tuple[Callable[[WindowPayload], None], WindowPayload]
    ) -> None:
        if self._closed:
            return
        on_done, payload = result
        on_done(payload)

    def shutdown(self, *, wait: bool = True) -> None:
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def post_to_ui_thread(self, fn: Callable[[], None]) -> None:
        QtCore.QTimer.singleShot(0, fn)

    def _build_payload(self, state: ViewState) -> WindowPayload:
        sl = self._data_source.window_indices(state.t_center, state.t_width)
        time = self._data_source.time[sl]
        posterior = self._data_source.load_posterior(sl)
        likelihood = (
            self._data_source.load_likelihood(sl)
            if "log_likelihood" in self._data_source.available_outputs
            else None
        )
        predictive = (
            self._data_source.load_predictive(sl)
            if "predictive_posterior" in self._data_source.available_outputs
            else None
        )
        state_probabilities = self._data_source.load_state_probabilities(sl)
        return WindowPayload(
            request_id=state.request_id,
            time=np.asarray(time),
            indices=sl,
            posterior=posterior,
            likelihood=likelihood,
            predictive=predictive,
            state_probabilities=state_probabilities,
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
        # Ensure user-closed windows are actually destroyed (otherwise
        # they linger in QApplication.topLevelWidgets() until GC).
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self._data_source = data_source
        self._backend = QtBackendAdapter(data_source)
        self._initial_t_width = float(t_width)
        self._initial_t_center: float | None = None
        self._core = ViewerCore(data_source, self._backend, t_width=t_width)
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
        self._initial_load_requested = False

        active_run = data_source.active_run
        detector = active_run.detector
        grid = PositionGrid.from_environment(detector.environments[0])

        # Built-in left-column view-models + panels. Order top-to-bottom
        # mirrors the static ``plot_non_local_model`` figure.
        self._raster_model = RasterModel(detector, active_run.spike_times)
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
        self._builtin_panels: list = [
            self._raster_panel,
            self._state_prob_panel,
            self._likelihood_panel,
            self._panel,
        ]

        # Right-column slice panel — per-bin readout of population
        # likelihood/posterior + predictive overlay + per-cell rows
        # for cells active in the cursor bin (and pinned cells).
        self._slice_model = SliceModel(
            detector=detector,
            spike_times=active_run.spike_times,
            time=np.asarray(active_run.results["time"].values),
        )
        self._slice_panel = QtSlicePanel(
            model=self._slice_model, position_centers=grid.centers
        )
        # Raster click → toggle pin on the slice panel. The viewer
        # owns the connection so swap can re-bind cleanly (the panel
        # references survive a swap; pin state clears via
        # ``rebind_after_swap``).
        self._raster_panel.cell_clicked.connect(self._slice_panel.toggle_pin)

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
        # Right column: slice panel.
        # Slider stretches across the full width below both columns.
        self._left_column_layout = QtWidgets.QVBoxLayout()
        for panel in self._builtin_panels:
            self._left_column_layout.addWidget(panel, stretch=1)
        self._extras_insert_index = self._left_column_layout.count()
        for extra in self._extra_panels:
            self._left_column_layout.addWidget(extra, stretch=1)
        left_column = QtWidgets.QWidget()
        left_column.setLayout(self._left_column_layout)

        # Right column: slice panel on top, extra bin-synced plugins
        # below it. Built via a fresh QVBoxLayout so the slice panel
        # plus any plugins live under one widget that the body's
        # QHBoxLayout owns.
        right_column_layout = QtWidgets.QVBoxLayout()
        right_column_layout.addWidget(self._slice_panel, stretch=2)
        for bin_panel in self._extra_bin_panels:
            right_column_layout.addWidget(bin_panel, stretch=1)
        right_column = QtWidgets.QWidget()
        right_column.setLayout(right_column_layout)

        body_layout = QtWidgets.QHBoxLayout()
        body_layout.addWidget(left_column, stretch=2)
        body_layout.addWidget(right_column, stretch=1)
        body_widget = QtWidgets.QWidget()
        body_widget.setLayout(body_layout)

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._controls_bar, stretch=0)
        self._layout.addWidget(body_widget, stretch=1)
        self._layout.addWidget(self._slider, stretch=0)
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
            (QtGui.QKeySequence(QtCore.Qt.Key_Left), self._core.step_left),
            (QtGui.QKeySequence(QtCore.Qt.Key_Right), self._core.step_right),
            (QtGui.QKeySequence("Shift+Left"), lambda: self._step_window(-1)),
            (QtGui.QKeySequence("Shift+Right"), lambda: self._step_window(+1)),
            (QtGui.QKeySequence("["), lambda: self._scale_t_width(0.5)),
            (QtGui.QKeySequence("]"), lambda: self._scale_t_width(2.0)),
            (QtGui.QKeySequence("R"), self._reset_view),
            (QtGui.QKeySequence("N"), self._core.next_event),
            (QtGui.QKeySequence("Shift+N"), self._core.prev_event),
            (QtGui.QKeySequence("Escape"), self._slice_panel.clear_pins),
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
                handler(self._core.set_t_center)
            target = getattr(panel, "x_link_target", lambda: None)()
            if target is not None and target is not link_target:
                target.setXLink(link_target)
            viewport = getattr(panel, "viewport", lambda: None)()
            if viewport is not None:
                viewport.installEventFilter(self)

    def _build_controls_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)
        overlays = self._data_source.active_run.event_overlays
        run_names = self._data_source.run_names
        multi_run = len(run_names) > 1

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
            0, min(self._speed_combo.count() - 1, self._speed_combo.currentIndex() + delta)
        )
        if new_idx != self._speed_combo.currentIndex():
            self._speed_combo.setCurrentIndex(new_idx)

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
        slider_value = self._slider.value()
        for bin_panel in (self._slice_panel, *self._extra_bin_panels):
            bin_panel.set_window_buffer(payload)
            # Re-render at the current cursor — the new buffer may
            # extend coverage past the cursor's previous reach.
            bin_panel.update_for_index(slider_value)

    def _step_window(self, direction: int) -> None:
        self._core.set_t_center(self._core.t_center + direction * self._core.t_width)

    def _scale_t_width(self, factor: float) -> None:
        new_width = max(1e-6, self._core.t_width * factor)
        self._core.set_t_width(new_width)

    def _reset_view(self) -> None:
        if self._initial_t_center is not None:
            self._core.set_t_center(self._initial_t_center)
        self._core.set_t_width(self._initial_t_width)

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
        # Calling ``core.set_t_center`` fires
        # ``_sync_autoscroll_cursor_to_core``, which handles the
        # resync (lock-gated against tick-driven setValue).
        self._core.set_t_center(float(time[value]))
        # Drive the slice panel + extra bin-synced plugins synchronously
        # off the slider so per-tick cursor updates land sub-ms (the
        # heavier window load is async and refreshes the buffer when it
        # commits).
        for bin_panel in (self._slice_panel, *self._extra_bin_panels):
            bin_panel.update_for_index(value)

    def _sync_cursor_markers(self, new_t_center: float) -> None:
        """Push ``(t_center, t_lo, t_hi)`` to every TimeAxisPanel.

        ``t_center`` is the dashed-line position; ``[t_lo, t_hi]`` is
        the active-bin band. Bin edges come from the time grid's
        midpoints (matches ``SliceModel._bin_edges``). Built-in
        panels all mix in ``CursorMarkersMixin``; user-supplied
        ``extra_panels`` opt in by exposing a ``set_cursor_markers``
        method (called via ``getattr`` so the kwarg stays
        backward-compatible).
        """
        time = self._data_source.time
        t_idx = int(np.searchsorted(time, new_t_center, side="right") - 1)
        t_idx = max(0, min(time.size - 1, t_idx))
        t_lo, t_hi = self._bin_edges_at(t_idx)
        for panel in self._builtin_panels:
            panel.set_cursor_markers(new_t_center, t_lo, t_hi)
        for panel in self._extra_panels:
            setter = getattr(panel, "set_cursor_markers", None)
            if setter is not None:
                setter(new_t_center, t_lo, t_hi)

    def _bin_edges_at(self, t_idx: int) -> tuple[float, float]:
        """Return ``(t_lo, t_hi)`` for bin ``t_idx`` using midpoints to neighbors.

        Mirrors ``SliceModel._bin_edges`` — kept inline here rather
        than lifted to a shared helper because the only other caller
        is the slice model and the math is six lines.
        """
        time = self._data_source.time
        n = time.size
        if n <= 1:
            t = float(time[0]) if n == 1 else 0.0
            return t, t
        t = float(time[t_idx])
        if t_idx == 0:
            half = (time[1] - time[0]) / 2.0
            return float(t - half), float(t + half)
        if t_idx == n - 1:
            half = (time[n - 1] - time[n - 2]) / 2.0
            return float(t - half), float(t + half)
        half_lo = (t - time[t_idx - 1]) / 2.0
        half_hi = (time[t_idx + 1] - t) / 2.0
        return float(t - half_lo), float(t + half_hi)

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
        if (
            self._autoscroll_cursor is not None
            and not self._autoscroll_resync_lock
        ):
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

        self._raster_model.set_active_run(new_detector, new_run.spike_times)
        self._raster_panel.rebind_after_swap()

        self._slice_model.set_active_run(
            new_detector,
            new_run.spike_times,
            np.asarray(new_run.results["time"].values),
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
                self._extras_insert_index + offset, panel, 1
            )

        self._extra_panels = new_extras
        self._all_panels = [*self._builtin_panels, *self._extra_panels]
        self._wire_panels(new_extras)

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


def _panel_for_metric_spec(spec: MetricSpec):
    """Lift a ``MetricSpec`` to the matching Qt panel."""
    if spec.kind == "line":
        return LineSeriesPanel(LineSeriesModel.from_metric_spec(spec))
    if spec.kind == "scatter":
        return ScatterSeriesPanel(ScatterSeriesModel.from_metric_spec(spec))
    if spec.kind == "intervals":
        return IntervalSeriesPanel(IntervalSeriesModel.from_metric_spec(spec))
    raise ValueError(f"Unknown MetricSpec.kind: {spec.kind!r}")


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

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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
