"""Interactive decoder viewer (PySide6 + pyqtgraph).

The top-level imports below are GUI-toolkit-free; the ``[viewer]``
extra (``PySide6``, ``pyqtgraph``) is required only for the Qt
frontend in ``viewer/qt.py`` and ``panels/qt/``.
"""

from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.panels import (
    BinSyncedPanel,
    TimeAxisPanel,
)
from non_local_detector.visualization.interactive.view_models import (
    BinPayload,
    CellSlice,
    EventOverlay,
    MetricSpec,
    PositionGrid,
    PosteriorHeatmapModel,
    RunBundle,
    ViewState,
    WindowPayload,
)


def launch(
    bundle,
    t_width: float = 1.0,
    block: bool = True,
    extra_panels: list | None = None,
    extra_bin_panels: list | None = None,
) -> int:
    """Open the Qt viewer against a single ``RunBundle`` or named dict.

    Lazy-imports the Qt frontend so the rest of the package stays
    GUI-toolkit-free.

    Parameters
    ----------
    bundle : RunBundle | dict[str, RunBundle]
    t_width : float, optional
    block : bool, optional
        If True (default), blocks on ``QApplication.exec()`` and
        returns the exit code. If False, the window is created but
        ``exec()`` is not entered — useful for tests.
    extra_panels : list of TimeAxisPanel, optional
        Custom time-axis panels appended below the built-in
        left-column stack. Each must implement ``TimeAxisPanel``.
    extra_bin_panels : list of BinSyncedPanel, optional
        Custom bin-synced plugins stacked below the built-in slice
        panel in the right column. Each must implement
        ``BinSyncedPanel`` (``set_window_buffer`` +
        ``update_for_index``); ``rebind_after_swap`` is optional.
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        launch_qt,
    )

    return launch_qt(
        bundle,
        t_width=t_width,
        block=block,
        extra_panels=extra_panels,
        extra_bin_panels=extra_bin_panels,
    )


__all__ = [
    "BinPayload",
    "BinSyncedPanel",
    "CellSlice",
    "EventOverlay",
    "InMemoryDecoderDataSource",
    "MetricSpec",
    "PositionGrid",
    "PosteriorHeatmapModel",
    "RunBundle",
    "TimeAxisPanel",
    "ViewState",
    "WindowPayload",
    "launch",
]
