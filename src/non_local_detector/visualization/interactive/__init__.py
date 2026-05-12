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
    SpikeEvent,
    SpikeEventIndex,
    ViewState,
    WindowPayload,
)


def launch_qt(
    bundles=None,
    t_width: float = 1.0,
    block: bool = True,
    extra_panels: list | None = None,
    extra_bin_panels: list | None = None,
    show_projected_2d: bool = False,
    show_projected_1d: bool = False,
    *,
    detector=None,
    results=None,
    spike_times=None,
    position=None,
    position_time=None,
    position_2d=None,
    position_2d_time=None,
    projection_track_graph=None,
    projection_edge_order=None,
    projection_edge_spacing=0.0,
    speed=None,
    name: str = "default",
) -> int:
    """Public entry point: open the interactive Qt viewer.

    Lazy-imports the Qt frontend so the rest of the package stays
    GUI-toolkit-free; signature mirrors
    ``viewer.qt.launch_qt`` exactly. See that function's docstring
    for the per-component vs bundle-form dispatch contract.

    Notebook quickstart::

        from non_local_detector.visualization.interactive import launch_qt
        results = detector.predict(spike_times=..., time=..., position=..., position_time=...)
        launch_qt(detector=detector, results=results, spike_times=..., position=..., position_time=...)
    """
    from non_local_detector.visualization.interactive.viewer.qt import (
        launch_qt as _launch_qt,
    )

    return _launch_qt(
        bundles,
        t_width=t_width,
        block=block,
        extra_panels=extra_panels,
        extra_bin_panels=extra_bin_panels,
        show_projected_2d=show_projected_2d,
        show_projected_1d=show_projected_1d,
        detector=detector,
        results=results,
        spike_times=spike_times,
        position=position,
        position_time=position_time,
        position_2d=position_2d,
        position_2d_time=position_2d_time,
        projection_track_graph=projection_track_graph,
        projection_edge_order=projection_edge_order,
        projection_edge_spacing=projection_edge_spacing,
        speed=speed,
        name=name,
    )


# Back-compat alias — older code uses ``launch(bundle)``.
launch = launch_qt


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
    "SpikeEvent",
    "SpikeEventIndex",
    "TimeAxisPanel",
    "ViewState",
    "WindowPayload",
    "launch",
    "launch_qt",
]
