"""Qt frontend: ``QApplication`` ownership, ``QtBackendAdapter``, ``QtViewer``.

This module is the only place in the codebase that creates a
``QApplication`` and the only Qt-importing entry under
``viewer/``. ``app.py`` lazy-imports ``launch_qt`` here.
"""

from __future__ import annotations

from collections.abc import Callable
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
from non_local_detector.visualization.interactive.view_models.state_prob import (
    StateProbabilityModel,
)
from non_local_detector.visualization.interactive.viewer.backend import (
    BackendAdapter,
)
from non_local_detector.visualization.interactive.viewer.core import ViewerCore

if TYPE_CHECKING:
    pass


class _LoadSignals(QtCore.QObject):
    """Signal bridge for thread-safe payload delivery (statespacecheck pattern).

    A worker running on the threadpool emits ``done(payload)``; the
    receiving slot runs on the UI thread.
    """

    done = QtCore.Signal(object)


class QtBackendAdapter(BackendAdapter):
    """Qt implementation of the backend protocol.

    Schedules window-load work on the global ``QThreadPool`` and
    marshals the result back via a ``QObject`` signal so the UI
    thread is the one that touches widgets.
    """

    def __init__(self, data_source: InMemoryDecoderDataSource) -> None:
        self._data_source = data_source
        self._thread_pool = QtCore.QThreadPool.globalInstance()

    def schedule_window_load(
        self, state: ViewState, on_done: Callable[[WindowPayload], None]
    ) -> None:
        signals = _LoadSignals()
        signals.done.connect(on_done, type=QtCore.Qt.QueuedConnection)

        def _work() -> None:
            payload = self._build_payload(state)
            signals.done.emit(payload)

        runnable = QtCore.QRunnable.create(_work)
        self._thread_pool.start(runnable)

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

        # Built-in panels are at fixed layout positions so M-key
        # swap can rebuild only the extras block.
        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._controls_bar, stretch=0)
        for panel in self._builtin_panels:
            self._layout.addWidget(panel, stretch=1)
        self._extras_insert_index = self._layout.count()
        for extra in self._extra_panels:
            self._layout.addWidget(extra, stretch=1)
        self._layout.addWidget(self._slider, stretch=0)
        container = QtWidgets.QWidget()
        container.setLayout(self._layout)
        self.setCentralWidget(container)

        self._wire_panels(self._all_panels)

        self._core.on_window_loaded(self._on_window_loaded)
        self._core.on_active_run_changed(self._rebind_panels)
        self._core.refresh_overlays()

        # Keyboard shortcuts. ``[`` / ``]`` shrink/grow the window
        # width; ``Shift+Left`` / ``Shift+Right`` step a full window
        # at a time; ``R`` resets center + width.
        QtCore.QTimer.singleShot(0, self._core.request_load)
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
        ):
            shortcut = QtGui.QShortcut(key_seq, self)
            shortcut.activated.connect(fn)

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
        if not overlays:
            bar.hide()
            return bar
        # Overlay-selector dropdown picks the navigator target.
        layout.addWidget(QtWidgets.QLabel("Overlay (N/Shift+N):"))
        self._overlay_combo = QtWidgets.QComboBox()
        self._overlay_combo.addItem("(none)", userData=None)
        for ovl in overlays:
            self._overlay_combo.addItem(ovl.name, userData=ovl.name)
        self._overlay_combo.currentIndexChanged.connect(self._on_active_overlay_changed)
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

    def _on_window_loaded(self, payload) -> None:
        for panel in self._all_panels:
            panel.update_window(payload)

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
        self._core.set_t_center(float(time[value]))

    def _rebind_panels(self, _new_run_name: str) -> None:
        """Rebind all panels to the new active run's detector.

        Fires *before* the new load is dispatched so payload collapse
        runs under the new schema.
        """
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
            self._layout.removeWidget(old_panel)
            old_panel.setParent(None)
            old_panel.deleteLater()

        new_extras = _auto_panels_from_extra_metrics(extra_metrics)
        for offset, panel in enumerate(new_extras):
            self._layout.insertWidget(self._extras_insert_index + offset, panel, 1)

        self._extra_panels = new_extras
        self._all_panels = [*self._builtin_panels, *self._extra_panels]
        self._wire_panels(new_extras)

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming convention
        """Drop self from the live-viewer registry on close."""
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

    viewer = QtViewer(data_source, t_width=t_width, extra_panels=extra_panels)
    _LIVE_VIEWERS.append(viewer)
    viewer.show()
    if block:
        return int(app.exec())
    return 0
