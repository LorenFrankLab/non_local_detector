<!-- markdownlint-disable MD060 -->

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

For project-specific rendering (ratemaps, pose overlays, custom
math), implement one of the two panel ABCs and pass it via the
matching kwarg. User-supplied panels are owned by the caller and
**not** torn down on model swap.

| Plugin lane  | Kwarg                    | Where it renders                                | Protocol         |
| ------------ | ------------------------ | ----------------------------------------------- | ---------------- |
| Window-based | `extra_panels=[...]`     | Below the built-in left column                  | `TimeAxisPanel`  |
| Bin-based    | `extra_bin_panels=[...]` | Below the built-in slice panel (right column)   | `BinSyncedPanel` |

The two lanes are **separate** — passing a `BinSyncedPanel` via
`extra_panels` raises `AttributeError: ... 'set_event_overlays'`
at viewer construction time (the time-axis wiring step requires
overlay registration). Use the right kwarg for the protocol your
panel implements.

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

| Method                         | Contract |
| ------------------------------ | -------- |
| `update_window(payload)`       | Render the time-axis window described by `payload`. Called once per committed window load. The payload carries `time` (1-D, n_visible), `posterior`, `likelihood` (may be None), `predictive` (may be None), `state_probabilities`, and `indices` (the slice into the full session). |
| `x_link_target()`              | Return the Qt object other panels link x-axes to (typically `self.getPlotItem()` for `pg.PlotWidget` subclasses). The viewer wires every panel's x-axis to the posterior panel's link target. |
| `click_handler(callback)`      | Register a callback invoked when the user clicks empty space on the panel; the callback receives the clicked x-coordinate as a *relative* offset in seconds against the panel's fixed `[-t_width/2, +t_width/2]` x-range (Phase 3.1). The viewer adds the current `t_center` back before recentering. |
| `set_event_overlays(overlays)` | Replace this panel's overlay set. Called whenever overlay visibility changes. Idempotent — the new list fully replaces previously rendered markers. |

### `BinSyncedPanel` Protocol

For panels that render the cursor's single time bin. The viewer
drives bin-synced plugins on the same buffered low-latency path the
built-in `QtSlicePanel` uses: each new `WindowPayload` is handed to
every bin plugin via `set_window_buffer`, then per-tick cursor
moves dispatch `update_for_index(t_idx)` synchronously.

```python
class BinSyncedPanel(Protocol):
    def set_window_buffer(self, payload: WindowPayload) -> None: ...
    def update_for_index(self, t_idx: int) -> None: ...
```

| Method | Contract |
|---|---|
| `set_window_buffer(payload)` | Cache the latest `WindowPayload`. Called by the viewer once per committed window load. May be a no-op if your panel doesn't need the buffered window. |
| `update_for_index(t_idx)` | Render the cursor's bin. **Must stay cursor-synchronous** even when `t_idx` falls outside the buffered window (the async window-load can lag behind fast playback). Three reasonable strategies — pick one and document it: (a) read from the buffered payload when `t_idx` is covered AND fall back to a single-row synchronous fetch otherwise (built-in `QtSlicePanel`, `Qt2DImagePanel`); (b) derive the render from a per-session cache that doesn't depend on the buffer at all (`Qt2DCellGridPanel` reads the active-cell list from `SliceModel.cells_at_index(t_idx)`); (c) explicitly clear the render when out-of-buffer rather than freezing on the last frame. A bare no-op for out-of-buffer ticks would visibly lag the cursor during playback — don't do that. Called per slider tick (synchronously) and once after each window-load commit. |

`rebind_after_swap()` is **optional**: implement it if your plugin
caches run-local state (e.g. a per-cell place-field map keyed by
cell id). The viewer calls it via `getattr` after
`ViewerCore.set_active_run` so plugins without run-local state
need not implement it.

```python
class CachingBinPanel:
    def rebind_after_swap(self) -> None:  # optional
        self._buffered_payload = None
        self._cell_field_cache.clear()
```

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
        # Time-axis panels render at relative coordinates against the
        # fixed ``[-t_width/2, +t_width/2]`` x-range (Phase 3.1). Use
        # ``payload.time`` to slice your metric to the visible window,
        # then subtract ``payload.t_center`` before plotting.
        sl = (self._metric_t >= payload.time[0]) & (
            self._metric_t <= payload.time[-1]
        )
        rel_t = self._metric_t[sl] - payload.t_center
        self._curve.setData(rel_t, self._metric_y[sl])
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
