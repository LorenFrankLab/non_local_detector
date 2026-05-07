# Interactive decoder viewer — plugin contract

Authoring reference for custom panels in the v1 Qt viewer. For
end-user docs (running the viewer, the `--run` / `--run-from-dir`
flags, the bundle directory format) see the CLI `--help`:

```bash
python -m non_local_detector.visualization.interactive --help
```

## When you don't need a plugin

For a simple time-series alongside your decoder output, attach it
to the bundle's `extra_metrics` and the viewer auto-builds a
matching panel:

```python
from non_local_detector.visualization.interactive.view_models.base import (
    RunBundle,
)
from non_local_detector.visualization.interactive.view_models.series import (
    MetricSpec,
)

bundle = RunBundle(
    results=results,
    detector=detector,
    spike_times=spike_times,
    position_time=position_time,
    position=position,
    speed=speed,
    extra_metrics={
        # pd.Series → LineSeriesPanel (time-indexed)
        "theta_power": theta_series,
        # MetricSpec.scatter → ScatterSeriesPanel (clickable, recenters)
        "spike_quality": MetricSpec.scatter(
            name="spike_quality", t=event_times, y=qualities
        ),
        # MetricSpec.intervals → IntervalSeriesPanel (translucent bands)
        "ripples": MetricSpec.intervals(
            name="ripples", t_start=ripple_starts, t_end=ripple_ends
        ),
    },
)
```

The viewer rebuilds the auto-extras from the new bundle's
`extra_metrics` on every M-key model swap, so each run can carry
its own metrics.

## When you do need a plugin

For project-specific rendering (e.g. ratemaps, pose overlays, custom
math), implement one of the panel ABCs and pass it via
`extra_panels=[...]`. User-supplied panels are owned by the caller
and **not** torn down on swap.

### `TimeAxisPanel` Protocol

For panels that render a window of decoder time bins. All built-in
left-column panels (raster, state-prob, likelihood, posterior)
implement this.

```python
class TimeAxisPanel(Protocol):
    def update_window(self, payload: WindowPayload) -> None: ...
    def x_link_target(self) -> Any: ...
    def click_handler(self, callback: Callable[[float], None]) -> None: ...
    def set_event_overlays(self, overlays: list[EventOverlay]) -> None: ...
```

| Method | Contract |
|---|---|
| `update_window(payload)` | Render the time-axis window described by `payload`. Called once per committed window load. The payload carries `time` (1-D, n_visible), `posterior`, `likelihood` (may be None), `predictive` (may be None), `state_probabilities`, and `indices` (the slice into the full session). |
| `x_link_target()` | Return the Qt object other panels link x-axes to (typically `self.getPlotItem()` for `pg.PlotWidget` subclasses). The viewer wires every panel's x-axis to the posterior panel's link target. |
| `click_handler(callback)` | Register a callback invoked when the user clicks empty space on the panel; the callback receives the clicked x-coordinate in absolute seconds. The viewer uses this to recenter. |
| `set_event_overlays(overlays)` | Replace this panel's overlay set. Called whenever overlay visibility changes. Idempotent — the new list fully replaces previously rendered markers. |

### `BinSyncedPanel` Protocol

For panels that render the cursor's single time bin. The right-column
slice panel implements this.

```python
class BinSyncedPanel(Protocol):
    def update_for_index(self, t_idx: int, payload: BinPayload) -> None: ...
```

`BinPayload` carries `t_idx`, `t`, `top_curve`, `top_curve_label`,
`predictive_curve` (may be None), and `cells` (tuple of
`CellSlice`). v1 ships one `BinSyncedPanel` (`QtSlicePanel`); the
ABC exists so future per-bin readouts (HPD, KL, custom metrics) can
plug in alongside it.

## Mixins available for Qt panel implementations

```python
from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,    # default click_handler/x_link_target via pg.PlotWidget
    EventOverlayMixin,     # default set_event_overlays via InfiniteLine/LinearRegionItem
)
```

Both mixins are designed to compose with `pg.PlotWidget`. Subclass
the protocol's intent, mix in these for the boilerplate, and only
override what's project-specific.

`ClickRecenterMixin._handle_click` skips when the underlying mouse
event has been accepted by a child item (e.g. a `pg.ScatterPlotItem`
spot). This is what lets the raster panel pin a clicked spike
without also recentering the view.

## Minimal custom panel

```python
import numpy as np
import pyqtgraph as pg

from non_local_detector.visualization.interactive.panels.qt._mixins import (
    ClickRecenterMixin,
    EventOverlayMixin,
)


class MyMetricPanel(pg.PlotWidget, ClickRecenterMixin, EventOverlayMixin):
    """Render some pre-computed metric vs window time."""

    def __init__(self, metric_t: np.ndarray, metric_y: np.ndarray) -> None:
        super().__init__(background="w")
        self._metric_t = metric_t
        self._metric_y = metric_y
        self._curve = self.plot([], [], pen=pg.mkPen("k"))
        self._install_click_recenter()
        self._overlay_items = []  # required by EventOverlayMixin

    def update_window(self, payload) -> None:
        # Use payload.time to slice your metric to the visible window.
        sl = (self._metric_t >= payload.time[0]) & (
            self._metric_t <= payload.time[-1]
        )
        self._curve.setData(self._metric_t[sl], self._metric_y[sl])
```

Inject it:

```python
from non_local_detector.visualization.interactive.viewer.qt import launch_qt

launch_qt(bundle, extra_panels=[MyMetricPanel(metric_t, metric_y)])
```

## Validation

The package's `panels.base` exports both protocols as `runtime_checkable`.
For sanity-checking a custom class:

```python
from non_local_detector.visualization.interactive.panels.base import TimeAxisPanel

assert isinstance(my_panel, TimeAxisPanel)
```

Note that `runtime_checkable` only verifies the named methods exist;
it does not type-check signatures. The CI lint test
`tests/lint/test_import_boundary.py` will catch Qt imports placed
outside the `panels/qt/` and `viewer/qt.py` allowlist if you
contribute panels back to this repo.

## See also

- [`view_models/base.py`](view_models/base.py) — `RunBundle`,
  `WindowPayload`, `BinPayload`, `CellSlice` definitions
- [`view_models/events.py`](view_models/events.py) — `EventOverlay`
  constructors (`points` / `intervals`)
- [`panels/qt/series.py`](panels/qt/series.py) — reference
  implementations for line/scatter/intervals panels
