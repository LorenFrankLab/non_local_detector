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
from non_local_detector.visualization.interactive.panels.qt.posterior import (
    QtPosteriorHeatmapPanel,
)
from non_local_detector.visualization.interactive.view_models.base import (
    PositionGrid,
    RunBundle,
    ViewState,
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.posterior import (
    PosteriorHeatmapModel,
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
        return WindowPayload(
            request_id=state.request_id,
            time=np.asarray(time),
            indices=sl,
            posterior=posterior,
            likelihood=likelihood,
            predictive=predictive,
        )


class QtViewer(QtWidgets.QMainWindow):
    """Minimal v1 viewer: posterior heatmap + center-time slider."""

    def __init__(
        self,
        data_source: InMemoryDecoderDataSource,
        t_width: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("non_local_detector — interactive viewer")
        # Ensure user-closed windows are actually destroyed (otherwise
        # they linger in QApplication.topLevelWidgets() until GC).
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self._data_source = data_source
        self._backend = QtBackendAdapter(data_source)
        self._core = ViewerCore(data_source, self._backend, t_width=t_width)

        grid = PositionGrid.from_environment(
            data_source.active_run.detector.environments[0]
        )
        self._posterior_model = PosteriorHeatmapModel(data_source.active_run.detector)
        self._panel = QtPosteriorHeatmapPanel(
            model=self._posterior_model, position_centers=grid.centers
        )
        self._core.on_window_loaded(self._panel.update_window)
        self._core.on_active_run_changed(self._rebind_panels)

        # Slider: integer indices into the time grid; map to t_center.
        n_time = data_source.n_time
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(n_time - 1)
        self._slider.setValue(n_time // 2)
        self._slider.valueChanged.connect(self._on_slider_value_changed)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._panel, stretch=1)
        layout.addWidget(self._slider, stretch=0)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Keyboard shortcuts.
        QtCore.QTimer.singleShot(0, self._core.request_load)
        for key, fn in (
            (QtCore.Qt.Key_Left, self._core.step_left),
            (QtCore.Qt.Key_Right, self._core.step_right),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(fn)

    @property
    def core(self) -> ViewerCore:
        return self._core

    def _on_slider_value_changed(self, value: int) -> None:
        time = self._data_source.time
        self._core.set_t_center(float(time[value]))

    def _rebind_panels(self, _new_run_name: str) -> None:
        """Rebind all panels to the new active run's detector.

        Triggered by ``ViewerCore.set_active_run`` *before* the new
        load is dispatched so the panel collapses the new payload
        under the correct schema.
        """
        new_detector = self._data_source.active_run.detector
        self._posterior_model.set_active_run(new_detector)
        grid = PositionGrid.from_environment(new_detector.environments[0])
        self._panel._position_centers = grid.centers

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


def launch_qt(
    bundles: RunBundle | dict[str, RunBundle],
    t_width: float = 1.0,
    block: bool = True,
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

    viewer = QtViewer(data_source, t_width=t_width)
    _LIVE_VIEWERS.append(viewer)
    viewer.show()
    if block:
        return int(app.exec())
    return 0
