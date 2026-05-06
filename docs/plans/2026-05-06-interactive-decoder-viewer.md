# Interactive Decoder Viewer

**Status:** Draft v3 — addresses review findings on RunBundle scope,
results contract, and schema-aware posterior reduction.
**Branch:** `interactive-decoder-viewer`
**Author:** Eric Denovellis (with Claude)
**Date:** 2026-05-06

## TL;DR

Build an interactive viewer that combines the multi-panel time-axis stack
of `continuum_swr_replay.visualization.plots.plot_detector` (left column)
with the right-hand population-likelihood + per-cell-likelihood slice
column from `statespacecheck-paper-viewer`'s
`statespacecheck_paper.interactive`. The viewer scrubs through a session
in real time, with the left column showing what the animal is doing /
what the network is doing across a time window and the right column
showing the decoded distribution and per-cell evidence at the cursor's
single time bin.

The viewer is a **sub-package of `non_local_detector`**
(`non_local_detector.visualization.interactive`), gated behind a `[viewer]`
optional dependency group. Project-specific panels (SWR, MUA, theta) live
in `continuum-swr-replay` as plug-in `TimeAxisPanel` subclasses against the
contract `non_local_detector` defines.

**v1** ships a PySide6 + pyqtgraph viewer for **sorted-spikes 1D**
decoders, with a clean **model/view split** that lets v2 add a Panel /
holoviews / bokeh backend for remote-server use without rewriting panel
logic. **v1 also supports loading multiple model runs and swapping
between them in the same window** (the statespacecheck pattern: M-key
toggle, model dropdown). v3+ extends to 2D decoders (movie-style),
animal-video overlay, and side-by-side model comparison.

---

## Decided

These were initially open and have been settled.

### Where this code lives — same repo, sub-package

`src/non_local_detector/visualization/interactive/` inside the
`non_local_detector` repo. Optional dependency group `[viewer]` for the
GUI deps. No separate repo. (Reasoning in the chat — coupling to results
schema is tight, and extras + lazy imports already solve the lean-install
concern, see `figurl_1D.py`.)

### Spike data scope (D3) — sorted only for v1

All four sorted-spikes 1D single-environment detectors in
`non_local_detector`: `SortedSpikesDecoder`,
`ContFragSortedSpikesClassifier`,
`NoSpikeContFragSortedSpikesClassifier`, and
`NonLocalSortedSpikesDetector`. Clusterless analogues (mark
distributions, per-electrode-group "spikes", per-mark diagnostics)
and multi-environment classifiers are deferred — see "Deferred-model
pathway" for the v3+ design sketch. Each deferred variant has a
clear extension point in v1's architecture, not a dead end.

### Event-time highlighting — global overlay + navigator

A common scenario the panel system alone does not address well:
"highlight these N event times across **every** time-axis panel
simultaneously, and let me jump between them." Examples — SWR ripple
times, user-detected replay windows, behavioral epochs, perturbation
trials. Putting these on a single `IntervalSeriesPanel` works for
visualization but doesn't solve the navigator UX (every panel
needs the marker; users want next/prev jump).

v1 ships a first-class **event overlay** mechanism:

```python
from non_local_detector.visualization.interactive import EventOverlay

bundle.event_overlays = [
    EventOverlay.points(name="SWR peaks", times=swr_peak_times, color="#1f77b4"),
    EventOverlay.intervals(name="Ripple windows",
                           t_start=ripple.start_time, t_end=ripple.end_time,
                           color="#1f77b4", alpha=0.15),
    EventOverlay.points(name="My replay events",
                        times=replay_event_times, color="purple"),
]
```

Behavior:

- Each overlay draws a translucent vertical line (for `points`) or
  shaded vertical band (for `intervals`) on **every** time-axis
  panel — posterior, likelihood, raster, state-prob, and every
  user `extra_panels` entry — at consistent x-coordinates.
- The viewer's controls bar gains an **overlay selector** dropdown
  that picks the "active" overlay and its color / visibility, plus a
  navigator with **`N` / `Shift+N` keyboard shortcuts**. Navigation
  semantics, given current `t_center`:

  - For a `points` overlay (single `times` array, sorted ascending):
    - `N` → smallest `times[i]` > `t_center`. Recenter on it.
    - `Shift+N` → largest `times[i]` < `t_center`. Recenter on it.
  - For an `intervals` overlay (`t_start`, `t_end`):
    - `N` → midpoint of the next interval whose `t_start[i]` >
      `t_center`. (When `t_center` is currently inside an interval,
      this skips to the next one — "go to next event," not "go to
      end of this one.")
    - `Shift+N` → midpoint of the previous interval whose
      `t_end[i]` < `t_center`. Symmetric.
  - At endpoints (no next / previous), the shortcut is a no-op and
    a brief status message announces "no more events."

  All overlays stay visible regardless of which is active; "active"
  only chooses the navigator's target set.

- Multiple overlays render simultaneously with their own colors. A
  visibility checkbox per overlay toggles each on/off.
- Click semantics:
  - Click a `points` line (any panel) → recenter on that exact time
    and pin (Esc unpins; mirrors raster pin/unpin UX).
  - Click an `intervals` shaded band → recenter on the **clicked
    x-position** (not the interval's start or midpoint) and pin.
    Matches the rest of the viewer's "click empty space recenters
    where you clicked" UX, and lets the user inspect any specific
    moment within a long interval.

- Renders on top of all panel content (high z-order).

Implementation: orchestrated by `ViewerCore`, drawn by panels.
`TimeAxisPanel.set_event_overlays(overlays: list[EventOverlay])` is
part of the protocol; the core calls it on every panel whenever
`bundle.event_overlays` or per-overlay visibility changes.

For the Qt backend, an `EventOverlayMixin` in `panels/qt/_mixins.py`
provides a default `set_event_overlays` implementation that:

- For `EventOverlay.points(times=...)`: creates one
  `pg.InfiniteLine(angle=90, pen=...)` per `times[i]`, batched and
  pooled across calls so overlay churn doesn't allocate per-frame.
- For `EventOverlay.intervals(t_start=..., t_end=...)`: creates one
  `pg.LinearRegionItem` per `(t_start[i], t_end[i])` pair with a
  translucent brush.
- Removes prior overlay items before adding new ones (so the call
  is idempotent).

All four generic series panels (`Line`, `MultiLine`, `Scatter`,
`Interval`) plus the built-in panels (`Posterior`, `Likelihood`,
`StateProbability`, `Raster`) inherit `EventOverlayMixin`. User
panels that subclass `TimeAxisPanel` directly can either inherit the
mixin (default behavior) or override `set_event_overlays` for a
custom coordinate system.

CLI parity: `--overlay name:kind:path:column[,end_column]:color`
repeated, e.g.
`--overlay swrs:intervals:ripple_times.parquet:start_time,end_time:#1f77b4`.

#### Event overlays — multi-run alignment

Overlays are stored on the `RunBundle`, but their **per-name schema
must align across every loaded bundle**, and **names must be unique
within each bundle**. The schema check is on `(name, kind)` tuples,
not just names — same-named overlays must agree on whether they are
`points` or `intervals`, since N / Shift+N navigation semantics
differ between the two (the navigator's "next event" rule is
defined per-kind in the section above). The data source enforces
both rules at construction:

```python
# 1. Within-bundle: names must be unique.
for run_name, bundle in runs.items():
    names = [ovl.name for ovl in bundle.event_overlays]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Run {run_name!r} has duplicate overlay names: {duplicates}. "
            f"Each overlay name must be unique within a bundle so that "
            f"`active_overlay_name` resolves unambiguously. To layer two "
            f"overlays of different kinds for the same events, give them "
            f"distinct names (e.g. 'swr_intervals' and 'swr_peaks')."
        )

# 2. Across-bundles: same (name, kind) tuple set.
def overlay_schema(bundle):
    return frozenset((ovl.name, ovl.kind) for ovl in bundle.event_overlays)

overlay_schemas = {name: overlay_schema(b) for name, b in runs.items()}
expected = next(iter(overlay_schemas.values()))
mismatched = {n: s for n, s in overlay_schemas.items() if s != expected}
if mismatched:
    raise ValueError(
        f"All RunBundles loaded together must declare the same overlay "
        f"(name, kind) schema. Got: {overlay_schemas!r}. "
        f"Same-named overlays must agree on `kind` (points vs intervals) "
        f"because N / Shift+N navigation semantics differ between them. "
        f"To compare two fits whose overlay schemas differ, either drop "
        f"the differing overlays from one bundle, give them distinct "
        f"names so the navigator treats them as separate overlays, or "
        f"run them in separate viewer instances."
    )
```

`EventOverlay` carries a `kind: Literal["points", "intervals"]`
field set by its `.points(...)` / `.intervals(...)` constructors;
this is the field the schema check compares.

Per-run overlay **data** (`times`, `t_start`, `t_end`, `color`) may
differ — that's the supported use case for model-derived overlays
(e.g. each run computes its own "high-confidence non-local times" and
exposes them under the same overlay name `"non_local_events"` with
`kind="points"`). The navigator follows the active run's data for
the active overlay; on swap, the active overlay name is preserved
(it's guaranteed to exist in the new run, with the same kind, by the
schema-alignment check) and the panels re-render with the new run's
overlay data — so navigation behavior stays consistent across the
swap.

For session-level overlays that are identical across all runs (SWR
times from the recording, behavioral epochs, etc.), the simplest
pattern is to attach the same `EventOverlay` objects to every
bundle:

```python
swr_overlay = EventOverlay.intervals(name="SWR", t_start=..., t_end=...)
for bundle in (cont_bundle, contfrag_bundle, nl_bundle):
    bundle.event_overlays.append(swr_overlay)
```

Reset semantics: if the user removes an overlay from one bundle but
not others (e.g. via a setter on `RunBundle`), the data source
re-validates on the next swap and either accepts it (still aligned)
or raises with the same error message above.

### Coverage of `plot_detector` — panel-by-panel mapping

Every panel produced by
[continuum_swr_replay.visualization.plots.plot_detector](../../../continuum-swr-replay/src/continuum_swr_replay/visualization/plots.py#L931)
maps to either a built-in viewer panel or a generic series panel:

| `plot_detector` panel                   | Viewer counterpart                                                                  | Notes                              |
|-----------------------------------------|-------------------------------------------------------------------------------------|------------------------------------|
| `plot_theta_lfp` (line)                 | `LineSeriesPanel`                                                                   |                                    |
| `plot_theta_power_zscore` (fill + thr.) | `LineSeriesPanel(fill_below=True, thresholds=[2.0])`                                | thresholds + fill_below in v1      |
| `plot_theta_phase` (line, ±π)           | `LineSeriesPanel(y_range=(-π, π))`                                                  |                                    |
| `plot_ripple_consensus_trace_zscore`    | `LineSeriesPanel(thresholds=[2.0]) + EventOverlay.intervals(...)`                   | shading via global overlay         |
| `plot_multiunit_firing`                 | `LineSeriesPanel(thresholds=[2.0]) + EventOverlay.intervals(...)` (HSE)             | same pattern                       |
| `plot_spike_times` (raster)             | built-in `RasterPanel` (place-field-peak sort + non-local shading)                  |                                    |
| `plot_acausal_probabilities`            | built-in `StateProbabilityPanel`; for user metrics, `MultiLineSeriesPanel`          | multi-line in v1                   |
| `plot_conditional_non_local_posterior`  | built-in `PosteriorHeatmapPanel` (with overlaid trajectory)                         |                                    |
| `plot_speed` (fill_between)             | `LineSeriesPanel(fill_below=True)`                                                  | fill_below in v1                   |
| `plot_head_direction` (scatter)         | `ScatterSeriesPanel`                                                                |                                    |

**`plot_peri_event_probability` is out of scope** for the interactive
viewer: it is event-aligned (x = peri-event time), not session-time-
aligned, so it doesn't belong in the time-axis stack. It remains a
static `matplotlib` figure, and Phase 5 documents this explicitly.

### Frontend swap-readiness — viewer split into core + adapter

The Qt frontend is one of two planned: v1 ships Qt; v2 ships
Panel/holoviews/bokeh for remote-server use. To make v2 cheap, the
viewer is split into three explicit layers, with v2 implementing only
the third:

| Layer                      | Module                                | Imports Qt?  | What v2 has to write   |
|----------------------------|---------------------------------------|--------------|------------------------|
| Data + view-models         | `data_source.py`, `view_models/`      | no           | nothing                |
| **Viewer core**            | `viewer/core.py`                      | **no**       | nothing                |
| Backend adapter (protocol) | `viewer/backend.py`                   | no           | one new class          |
| CLI / launch dispatch      | `app.py`, `__init__.py`               | no (lazy)    | add `panel_` branch    |
| Qt frontend                | `viewer/qt.py`, `panels/qt/`          | yes          | (already done in v1)   |
| Panel frontend (v2)        | `viewer/panel_.py`, `panels/panel_/`  | no Qt; bokeh | parallel to `qt/`      |

`viewer/core.py` owns:

- `ViewState` (frozen dataclass: `t_center`, `t_width`, `request_id`).
- The active-run state and `set_active_run(...)` rebind logic.
- The pinned-event state and the recenter-on-click handler.
- The event-overlay set + active-overlay selector + navigator
  (next-event / previous-event computation given a `t_center`).
- The stale-result rejection rule (drop results whose `request_id` is
  older than the latest committed).

It does **not** own:

- Any rendering (panels handle that).
- Threading primitives (those go through `BackendAdapter`).
- Keyboard shortcut binding (frontend wires keys to `core.step_left()`,
  `core.next_event()` etc.).
- Layout / window decoration.

`viewer/backend.py` defines a small `BackendAdapter` protocol the core
uses for I/O the frontend has to provide:

```python
class BackendAdapter(Protocol):
    def schedule_window_load(
        self, state: ViewState, on_done: Callable[[WindowPayload], None]
    ) -> None: ...
    """Run a background load; call on_done on the UI thread when ready."""

    def post_to_ui_thread(self, fn: Callable[[], None]) -> None: ...
    """Marshal a callback onto the frontend's event loop."""
```

`viewer/qt.py` implements `BackendAdapter` via `QThreadPool` + a
`QObject` signal bridge (statespacecheck pattern). v2's `viewer/panel_.py`
implements it via `panel.io.unlocked()` + a thread executor — different
plumbing, same interface.

What this means concretely for the user's question:

- **v2's effort is bounded.** It writes one `BackendAdapter`, one
  `panels/panel_/` directory of renderers (one file per panel,
  consuming existing view-models), and a thin layout module. It
  re-uses every other line of code from v1.
- **Tests are mostly portable.** Tests against `viewer/core.py` and
  `view_models/*` run unchanged on both backends; only the
  pixel/render tests are backend-specific.
- **CI gate extended.** The Phase 1b AST-based import-boundary
  test (no `pyqtgraph` / `PySide6` imports outside the Qt
  allowlist; see Phase 1b for the `ast.parse(...)` pseudocode)
  covers `viewer/core.py` and `viewer/backend.py` automatically
  because it walks every `*.py` under
  `src/non_local_detector/visualization/interactive/` — those
  modules must stay GUI-toolkit-free.

This split is set up in v1 even though v2 isn't built — splitting after
the fact is much harder than splitting from day one.

### Backend directory layout — subdirectories per backend

`panels/qt/` (v1) and future `panels/panel_/` (v2) are sibling
subdirectories that each contain one file per panel. Each backend is
self-contained: maintainers reviewing v2's Panel implementation only
need to read `panels/panel_/`. The shared logic lives in `view_models/`
and is consumed identically by both. Rejected: a flat `panels/` with
file-level suffixes (`posterior_qt.py`, `posterior_panel.py`) — diffs
across backends become harder to read and the directory bloats.

### Sample notebook — yes, ship one with v1

`notebooks/interactive_viewer_demo.ipynb` walks through:

1. Building a `RunBundle` from a fitted detector + a session's spike
   times + position info.
2. Calling `launch(bundle)` to open the viewer.
3. The minimal `predict(return_outputs=...)` snippet for users who want
   the LikelihoodHeatmap and predictive overlay.
4. Building a multi-run `dict[str, RunBundle]` for model comparison.
5. **Adding user-computed metrics** — a `pd.Series` of replay scores
   attached via `bundle.extra_metrics`; a `MetricSpec.scatter(...)`
   with click-to-recenter; a custom
   `LineSeriesPanel(..., fill_below=True, thresholds=[2.0])` passed via
   `extra_panels=[...]`; a `MultiLineSeriesPanel` showing user-defined
   probabilities side-by-side. Shows each of the three sized
   affordances on the same dataset.
6. **Highlighting event times** — attach a `bundle.event_overlays =
   [EventOverlay.points(...), EventOverlay.intervals(...)]`,
   demonstrate the overlay selector dropdown, the visibility
   checkboxes, and the `N` / `Shift+N` navigator across SWR ripple
   times + custom replay events on the same axes.

Discoverability for this kind of tool is dominated by "did the user see
how to start it?" — a notebook is far better than docstrings for that.

### Entry points — both CLI and notebook callable

v1 ships both: `python -m non_local_detector.visualization.interactive
...` from the shell, and `launch(...)` callable from a notebook (which
pops a Qt window). Mirrors the statespacecheck pattern. Notebook path
requires Qt to be available — for remote-DISPLAY-less environments, v2's
Panel backend will provide a `launch_in_browser(...)` notebook helper.

### Viewer input contract — `RunBundle` dataclass, not a `(results, detector)` tuple

`predict()` returns an `xr.Dataset` with `acausal_posterior` and
`acausal_state_probabilities` only by default
([base.py:2388](../../src/non_local_detector/models/base.py#L2388)).
`log_likelihood`, `causal_posterior`, and `predictive_posterior` are
appended only when the user passed `return_outputs=...` to `predict()`
([base.py:2401](../../src/non_local_detector/models/base.py#L2401),
[base.py:2429](../../src/non_local_detector/models/base.py#L2429)). The
results object also does not contain decode-time spike times, animal
position, position time, or speed — those live alongside the results in
the caller's notebook
([static.py:98](../../src/non_local_detector/visualization/static.py#L98)
shows the static plot taking them as separate inputs).

So a `(results, detector)` pair is not sufficient to render even the
left-column raster panel. The viewer takes a structured `RunBundle`:

```python
@dataclass
class RunBundle:
    """User-facing viewer input bundle. Mutable so users can build it
    incrementally (e.g. attach event_overlays / extra_metrics after
    construction). Validated at construction *and* whenever the viewer
    consumes it (via `validate()`)."""
    results: xr.Dataset           # must contain acausal_posterior + acausal_state_probabilities;
                                  #   may contain log_likelihood, predictive_posterior, etc.
    detector: _DetectorBase       # fitted; provides state_ind_, state_names, encoding_model_,
                                  #   environments[0]
    spike_times: list[np.ndarray] # per-cell spike time arrays (decode-time)
    position_time: np.ndarray     # (n_position_time,), absolute seconds
    position: np.ndarray          # (n_position_time,) for 1D, (n_position_time, 2) for 2D
    speed: np.ndarray | None = None  # (n_position_time,), optional; some panels need it
    events: pd.DataFrame | None = None  # optional precomputed per-spike event table
                                  #   (HPD overlap, KL divergence, spike prob);
                                  #   if absent, MetricPanel-style panels are disabled
    extra_metrics: dict[str, MetricSpec | pd.Series] = field(default_factory=dict)
                                  # optional user-computed metrics;
                                  #   pd.Series → line panel; MetricSpec for
                                  #   scatter / interval renderings.
    event_overlays: list[EventOverlay] = field(default_factory=list)
                                  # optional point/interval event overlays
                                  #   drawn across every time-axis panel.
                                  #   Navigator (N / Shift+N) jumps between
                                  #   events of the active overlay.
                                  #   Construction validates two rules:
                                  #     1. Within this bundle: overlay names
                                  #        must be unique (so the navigator's
                                  #        active_overlay_name resolves
                                  #        unambiguously).
                                  #     2. Across all runs in multi-run mode:
                                  #        the (name, kind) schema set must
                                  #        match — same-named overlays must
                                  #        agree on whether they are points
                                  #        or intervals, since N / Shift+N
                                  #        navigation semantics differ
                                  #        between the two kinds.
                                  #   Per-run overlay *data* (times,
                                  #   t_start, t_end, color) may differ
                                  #   freely under a matching schema.
                                  #   Schema mismatches, kind drifts, and
                                  #   within-bundle duplicates all raise
                                  #   ValueError at construction. See
                                  #   "Event overlays — multi-run
                                  #   alignment" below for full pseudocode.

    def __post_init__(self) -> None:
        """Validate internal consistency."""
        # monotonic position_time, dimensionality vs. detector environment,
        # len(spike_times) == n_neurons (see "Validates internal consistency"
        # in the surrounding text).
        ...

    def required_outputs(self) -> set[str]:
        """Return the set of `return_outputs` values needed for full viewer functionality."""
```

`RunBundle` is **not** frozen — the documented usage pattern
(`bundle.event_overlays = [...]`, `bundle.extra_metrics["my_score"] = ...`)
mutates it after construction. `__post_init__` validates internal
consistency once at construction; the viewer revalidates relevant
fields (e.g. that `event_overlays` reference the same time grid)
when it consumes the bundle. (`ViewState`, in contrast, **is** frozen
— it's a per-load request snapshot, not user-built state.)

Construction validates that `position_time` is monotonic, that `position`
has the right second dimension for the detector's environment, and that
`spike_times` length matches the detector's encoding model's neuron
count. Mismatches raise with a clear message at construction.

Single-run callers use a `RunBundle.from_predict(...)` helper that wraps
the common case — passes through to a fitted detector's `predict()` and
attaches the supplied position / spike data.

### Viewer-ready `results` contract

The viewer needs more than the default `predict()` output. Specifically:

| Variable                      | Required for                                                | How to obtain                                      |
|-------------------------------|-------------------------------------------------------------|----------------------------------------------------|
| `acausal_posterior`           | PosteriorHeatmapPanel, SlicePanel population (smoothed)     | always present (default `predict()`)               |
| `acausal_state_probabilities` | StateProbabilityPanel                                       | always present                                     |
| `log_likelihood`              | LikelihoodHeatmapPanel, SlicePanel population (top curve)   | `predict(return_outputs=["log_likelihood"])`       |
| `predictive_posterior`        | SlicePanel predictive overlay (top-row blue line)           | `predict(return_outputs=["predictive_posterior"])` |

`log_likelihood` here is the **aggregate** likelihood across cells —
sorted-spikes KDE accumulates each neuron's contribution into a single
`(n_time, n_interior)` matrix and subtracts the no-spike term
([sorted_spikes_kde.py:342](../../src/non_local_detector/likelihoods/sorted_spikes_kde.py#L342)).
**The per-cell SlicePanel rows do not need `log_likelihood`.** Each row
shows the cell's place field (extracted via
`extract_per_cell_place_fields(detector)` from
`non_local_detector.analysis.place_fields` — see "Place-field
extraction" in the math section for why per-cell display uses this
helper rather than the state-aligned variant) overlaid against the
predictive distribution; per-cell row availability is gated on
`spike_times` (always in `RunBundle`) plus the detector exposing
extractable place fields (i.e. it is sorted-spikes, single-env,
single-group). Optional per-spike event metrics (HPD
overlap, KL divergence, spike prob) come from the `RunBundle.events`
field if present, and decorate each row when available — they're not
required to render the row itself.

The data source detects which arrays are present at construction and
records a `available_outputs: set[str]` attribute. Panels that require a
missing array **disable themselves with a clear, actionable error
message in their title bar** rather than crashing — e.g.
"LikelihoodHeatmapPanel disabled: log_likelihood not in results.
Re-run predict(return_outputs=['log_likelihood', 'predictive_posterior'])
and rebuild this RunBundle." This means a viewer launched on
default-`predict()` output still works for the posterior + state-prob +
raster panels; the user gets a clear path forward when they want the
full set.

Document this contract explicitly in the `[viewer]` extra's README.
Phase 1b verifies it with a smoke test that constructs a RunBundle from
each shape of results dataset (default-predict, predict-with-loglik,
predict-with-everything) and asserts the right panels enable / disable.

### Caching (D4) — in-memory v1, Zarr v2

v1 operates on a loaded `xr.Dataset` results object held in RAM and
slices in-place. The data-source class is the only thing that touches
the dataset, so v2 swaps in a Zarr-backed source without touching panel
code.

### Plugin contract (D5) — yes, build it

`non_local_detector` ships a `TimeAxisPanel` ABC plus concrete
implementations of the decoder-output panels. Project-specific panels
(SWR, MUA, theta) subclass the ABC in their own packages and are passed
to the viewer at construction time. ~50 LOC of base class.

### User-defined metric panels — three sizes of affordance

Subclassing `TimeAxisPanel` is the right tool for bespoke
interactivity, but it's overkill for the common case
("I have a `pd.Series` of replay scores indexed by time, just plot
it"). v1 ships three sized affordances so the simple case stays
simple:

**Size 1 — `extra_metrics` on `RunBundle` (zero-code).** Add a
`pd.Series` (or DataFrame with `time` index) to
`bundle.extra_metrics["my_score"]` and the viewer renders it as a
line panel automatically, stacked below the built-in panels in the
left column. Choose the rendering by passing a small `MetricSpec`
dataclass instead:

```python
from non_local_detector.visualization.interactive import MetricSpec

bundle.extra_metrics = {
    "replay_score": my_replay_score_series,                # → line panel
    "swr_events":   MetricSpec.intervals(start_times, end_times),  # → shaded intervals
    "spike_quality": MetricSpec.scatter(t, y, color="purple"),     # → scatter, click-to-recenter
}
```

**Size 2 — generic panel classes (one line per panel).** When the user
wants control over title, color, y-range, click handler, or panel
height without subclassing:

```python
from non_local_detector.visualization.interactive.panels.qt import (
    LineSeriesPanel, MultiLineSeriesPanel,
    ScatterSeriesPanel, IntervalSeriesPanel,
)

launch(bundle, extra_panels=[
    LineSeriesPanel(name="Replay score", t=t, y=score, color="purple"),
    LineSeriesPanel(                          # filled-area + threshold line
        name="Theta power", t=t, y=theta_power_z,
        fill_below=True, thresholds=[2.0],
    ),
    MultiLineSeriesPanel(                     # several lines on one panel
        name="State probabilities",
        t=t,
        ys={"Non-Local": non_local_p, "Local": local_p, "No-Spike": no_spike_p},
        colors={"Non-Local": "red", "Local": "blue", "No-Spike": "grey"},
        y_range=(0, 1.05),
    ),
    ScatterSeriesPanel(
        name="Spike quality", t=spike_t, y=quality, color="green",
        click_recenters=True,                 # click a point → recenter
    ),
    IntervalSeriesPanel(name="My events", t_start=starts, t_end=ends),
])
```

These four classes ship in `panels/qt/series.py` and consume
`view_models/series.py`. They subclass `TimeAxisPanel` internally.
Together with the built-in panels they cover every panel kind that
the static `plot_detector` produces (see "Coverage of `plot_detector`"
below for the panel-by-panel mapping).

**Size 3 — `TimeAxisPanel` subclass (full control).** For panels that
need shaded backgrounds tied to model state, multi-line displays with
synced highlights, custom hover tooltips, or anything else. This is
what `continuum-swr-replay`'s `SWRTracePanel` /
`MUARatePanel` / `ThetaLFPPanel` will use in Phase 5 — though most of
them turn out to be thin wrappers around `LineSeriesPanel` +
`IntervalSeriesPanel`.

All three sizes share the same view-state plumbing (x-axis link,
center-time marker, click-to-recenter on empty-space, wheel-to-resize
window). The user gets that for free regardless of which size they
choose.

### Model comparison — swap pattern in v1, side-by-side in v3

The user wants to compare different model runs (e.g. continuous vs.
contfrag, two re-fits with different transition priors, the same model
on two recording sessions). Two architectures available:

- **Swap (statespacecheck pattern, in v1).** The viewer is constructed
  with a dict of named `RunBundle`s (`{name: RunBundle}` — see
  "Viewer input contract" below for the dataclass). One run is active
  at a time. The user swaps by M-key or by a "Model" dropdown in the
  controls bar; all panels re-render with the swapped data.
  View state (center time, window width, pinned event row) is
  preserved across swaps so you can compare two runs at the *same*
  time bin instantly. Compatible runs must share the same time grid
  (decoded on the same session); mismatched grids raise at viewer
  construction.
- **Side-by-side (v3).** Two parallel time-axis stacks + slice columns
  in the same window, sharing the center-time slider. Bigger
  architectural change — needs a layout that handles 2× panels and
  decisions about which runs share which controls (pin? overlay?).

v1 ships swap. Side-by-side is a v3 plan once the viewer's primitives
are settled.

### Backend strategy (D2) — Qt primary, browser-as-fallback architected from v1

The user works on remote servers where Qt over X-forwarding / VNC is
painful, and has tried Panel/holoviews/bokeh and found per-tick
interaction not fast enough. So:

- **v1 ships a PySide6 + pyqtgraph viewer.** This is the primary
  experience: native desktop, sub-ms per-tick, click + keyboard nav.
- **v1 enforces a model/view split** (see below) so the Qt rendering
  layer is separable from the data-transformation layer.
- **v2 ships a Panel/holoviews backend** consuming the same view-models.
  It will be slower per-tick — degraded interaction is accepted as the
  trade for working over a browser. This is the remote-server path.
- The static `plot_detector` figure remains the third option for the
  "I just want to share a screenshot" case.

The single hard architectural rule that makes this work: **`pyqtgraph`
and `PySide6` imports live only inside `viewer/qt.py` and
`panels/qt/`** (and analogously, v2's Panel/holoviews/bokeh imports
will live only inside `viewer/panel_.py` and `panels/panel_/`). All
other modules — including `app.py`, `__init__.py`, `view_models/`,
`viewer/core.py`, `viewer/backend.py`, and `data_source.py` — are
GUI-toolkit-free and route to the chosen backend via lazy imports
(see "Hard architectural rule" below for the dispatch pattern). The
trailing underscore on `panel_/` is intentional: it avoids shadowing
the third-party `panel` library (which the v2 backend imports)
inside the subpackage.

---

## Open decisions

None. All architectural and layout decisions are now settled.

---

## Goal

Replace the static `plot_detector` figure as the day-to-day exploratory
tool for inspecting decoder output. Specifically:

- **Scrub temporal context.** Drag a slider or hit ← / → to move the
  center time. The left-column panels update to show a window
  centered on the cursor. The right-column slice panel updates to
  show the decoded distribution at the cursor's single time bin.
- **Inspect per-cell evidence.** When a bin shows non-local activity,
  the user wants to see *which cells fired* and *how each cell's place
  field overlaps the predictive distribution*. The right-column per-cell
  rows make this immediate.
- **Pin events.** Click on a spike in the raster panel to pin it; the
  pinned spike's per-cell row stays visible across scrolling so the
  user can compare it to the current bin.
- **Compose project-specific context.** Downstream projects
  (`continuum-swr-replay`) add their own time-axis panels (SWR,
  theta, MUA) without changing the viewer.
- **Re-use across futures.** Same pattern extends to 2D decoders
  (movie-style) and animal-video overlay (v3+).

## Reference designs

- `continuum-swr-replay`'s `plots.plot_detector`
  ([continuum-swr-replay/src/continuum_swr_replay/visualization/plots.py:931](../../../continuum-swr-replay/src/continuum_swr_replay/visualization/plots.py#L931))
  — establishes the left-column panel set and the per-panel
  `(data, time_slice, ax)` composability.
- `statespacecheck-paper-viewer`'s `interactive` package
  ([statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/))
  — establishes the PySide6 + pyqtgraph + threadpool architecture, the
  right-column SlicePanel design, the click-to-pin + keyboard nav UX,
  and the Zarr-cache windowed-read pattern.
- `continuum-swr-replay`'s `interactive.py` (Panel + holoviews)
  ([continuum-swr-replay/src/continuum_swr_replay/visualization/interactive.py](../../../continuum-swr-replay/src/continuum_swr_replay/visualization/interactive.py))
  — the prior Panel attempt the user found not fast enough.
  v2's second backend learns from this: degraded interaction
  is the trade for remote-display capability.

The new viewer is essentially: **statespacecheck's right column +
plot_detector's left column, behind a backend-agnostic panel-plugin
contract.**

## Re-usable components & future extensions

The architecture is shaped by three concrete future extensions called out
up front so v1's design doesn't paint them into corners.

### Future 1 — 2D decoder support

`SortedSpikesDecoder` and friends fit on 1D linearized position; they also
have 2D analogues (and the `make_single_environment_movie` /
`make_non_local_movie` paths already render 2D posteriors). When extended
to 2D:

- The left-column posterior heatmap becomes a sequence of 2D images —
  i.e. a "movie panel" instead of a 2D array (time × position) heatmap.
- The right-column SlicePanel's population-likelihood plot becomes a 2D
  heatmap (one frame).
- The right-column per-cell rows become 2D place-field tiles with the
  spike position marked.
- The "raster" panel is largely unchanged.

To keep this cheap, **v1 abstracts position dimensionality behind a
`PositionGrid` type**: 1D returns `(n_pos,)` bin centers, 2D returns
`(n_x, n_y)` bin grids. Panels that visualize position-distributions
take a `PositionGrid` and dispatch on `.ndim`.

### Future 2 — Animal-video overlay

The user wants, eventually, to overlay the 2D decoded posterior on top of
the recorded video frame for the cursor's time bin. This slots into the
viewer as one additional panel:

- A `BinSyncedPanel` (panel that updates at the cursor's bin index, not
  over a window — just like the right-column SlicePanel does) that loads
  a video file, indexes frames by timestamp, and renders the current
  frame with the 2D posterior alpha-blended on top.

The architectural upshot: **v1 splits panel protocols into two ABCs**
— `TimeAxisPanel` (window-based, left column) and `BinSyncedPanel`
(point-based, right column / overlay). Both can host either 1D or 2D
data. The `SlicePanel` is a `BinSyncedPanel`. A future `VideoOverlayPanel`
is also a `BinSyncedPanel`.

### Future 3 — Browser/remote backend

For remote server work, v2 adds Panel/holoviews implementations of the
panels. The model/view split makes this cheap: only the renderer needs
new code; the data transformations and the data source are unchanged.

---

## High-level architecture

```text
src/non_local_detector/visualization/interactive/
├── __init__.py            # public API: launch(), DecoderViewer
├── app.py                 # CLI entry point (argparse + lazy import shim;
│                          #   NO PySide6/pyqtgraph imports — those live
│                          #   in viewer/qt.py, which app.py loads only
│                          #   after argparse decides which backend to use)
├── data_source.py         # InMemoryDecoderDataSource (v1)
├── view_models/           # backend-agnostic data transforms
│   ├── __init__.py
│   ├── base.py            # ViewState, PositionGrid, RunBundle, payload dataclasses
│   ├── posterior.py       # PosteriorHeatmapModel (1D + 2D-ready)
│   ├── likelihood.py      # LikelihoodHeatmapModel
│   ├── state_prob.py      # StateProbabilityModel
│   ├── raster.py          # RasterModel
│   ├── series.py          # LineSeriesModel, MultiLineSeriesModel,
│   │                      #   ScatterSeriesModel, IntervalSeriesModel,
│   │                      #   MetricSpec
│   ├── events.py          # EventOverlay
│   └── slice.py           # SliceModel (per-bin payload, 1D + 2D-ready)
├── viewer/                # state + orchestration (Qt-free) + Qt window
│   ├── __init__.py
│   ├── core.py            # ViewerCore: state, navigator, pin, overlays
│   ├── backend.py         # BackendAdapter Protocol (no Qt imports)
│   └── qt.py              # QtViewer(QMainWindow) + QtBackendAdapter
├── panels/
│   ├── __init__.py
│   ├── base.py            # TimeAxisPanel + BinSyncedPanel ABCs
│   └── qt/                # PySide6 + pyqtgraph rendering (v1)
│       ├── __init__.py
│       ├── posterior.py   # consumes view_models.posterior
│       ├── likelihood.py
│       ├── state_prob.py
│       ├── raster.py
│       ├── series.py      # LineSeriesPanel, MultiLineSeriesPanel,
│       │                  #   ScatterSeriesPanel, IntervalSeriesPanel
│       └── slice.py
└── README.md              # how to subclass TimeAxisPanel; install
                           # the [viewer] extra
```

(In v2, `viewer/panel_.py` and `panels/panel_/` land alongside the
`qt/` siblings; both consume the same `view_models/` and `viewer/core.py`.)

(In v2, `panels/panel_/` lands alongside `panels/qt/` with parallel
implementations, all consuming the same `view_models/` modules.)

`continuum-swr-replay` adds its own panel module:

```text
continuum_swr_replay/visualization/interactive_panels/
├── view_models.py         # SWRTraceModel, MUARateModel, ThetaLFPModel,
│                          #  SpeedModel, HeadDirectionModel
└── qt.py                  # Qt panels consuming the view models above
```

…and a thin convenience wrapper that constructs the viewer with these
panels pre-registered (`continuum_swr_replay.visualization.launch_detector_viewer`).

### Hard architectural rule

**`pyqtgraph` and `PySide6` may only be imported under
`viewer/qt.py` and `panels/qt/`.** Every other module —
`__init__.py`, `app.py`, `data_source.py`, `view_models/`,
`viewer/core.py`, `viewer/backend.py` — must be plain Python +
NumPy + pandas + xarray. **`app.py` is explicitly inside the
boundary**: it owns the CLI (argparse) but does **not** create the
`QApplication` or import any Qt module at top level. Backend
selection works via a lazy dispatch:

```python
# app.py (illustrative)
import argparse

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--backend", choices=["qt", "panel"], default="qt")
    # ... other CLI flags ...
    args = parser.parse_args(argv)
    if args.backend == "qt":
        from .viewer.qt import launch_qt   # lazy: pulls PySide6 here
        return launch_qt(...)
    else:
        from .viewer.panel_ import launch_panel  # v2 lazy import
        return launch_panel(...)
```

`viewer/qt.py` owns `QApplication` creation, the
`QtBackendAdapter`, and the `QtViewer` window class. v2's
`viewer/panel_.py` will own the equivalent Panel-server bootstrap.
This keeps Qt imports in exactly one module per backend and lets
`app.py` stay backend-agnostic.

Enforced by an **AST-based import-statement scan** (Phase 1b
deliverable; full pseudocode there). The scan walks every `*.py`
under `src/non_local_detector/visualization/interactive/`, parses
each file with `ast.parse(...)`, and fails the build if any
`Import` / `ImportFrom` node names a top-level module of
`pyqtgraph` or `PySide6` and the containing file is not under the
allowlist (`viewer/qt.py`, `panels/qt/`). Grep was explicitly
rejected as the implementation because the same phase adds
`pyqtgraph` and `PySide6` to `pyproject.toml`'s `[viewer]` extra
and the deferred-model section names them in prose — both
legitimate string occurrences that grep would false-positive on.

This is the single rule that makes v2's second backend cheap —
viewer state, navigator, and event-overlay logic are all
GUI-toolkit-free, so v2 only writes panel renderers + a
`BackendAdapter` and re-uses everything else.

### Data flow

```text
                     ┌────────────────────┐
                     │ DecoderDataSource  │   (v1: in-memory xr.Dataset)
                     │  (no Qt imports)   │
                     └─────────┬──────────┘
                               │  window_indices, load_posterior, ...
                               ▼
                     ┌────────────────────┐
                     │   view_models.*    │   (no Qt imports — pure
                     │  data transforms,  │    data → arrays /
                     │  payload classes   │    payload dataclasses)
                     └─────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼                             ▼
   ┌─────────────────────┐         ┌──────────────────────┐
   │  panels/qt/* (v1)   │         │  panels/panel_/* (v2)│
   │  pyqtgraph render   │         │  holoviews render    │
   └─────────────────────┘         └──────────────────────┘
                ▲                             ▲
                │                             │
   ┌────────────┴─────────────────────────────┴──────────┐
   │                  DecoderViewer                       │
   │   ViewerCore state { ViewState (req snapshot),       │
   │                       pinned_event_row, overlays }   │
   │   WindowLoadWorker (QThreadPool)                     │
   │   per-tick path (in-RAM ring buffer)                 │
   └──────────────────────────────────────────────────────┘
```

Per-tick latency target (Qt): **< 16 ms** for the right-column slice
update (a single array index into the in-RAM buffer). Window-load
latency target: **< 200 ms** for the heaviest in-memory slice.

### View state object

Two distinct concepts share the word "state" in this design and
must not be conflated:

1. **`ViewState`** (the frozen dataclass below): a per-load
   **request snapshot**. The window-load worker receives one of
   these and reports back with its `request_id`; stale results are
   dropped. Mirrors statespacecheck's pattern. **Does not** carry
   pinned-event state, overlay state, or anything else that isn't
   needed to fulfill a single window-load request.
2. **ViewerCore state** (mutable fields on `ViewerCore`): the
   broader UI state the core owns —
   `current_view_state: ViewState`, `pinned_event_row: int | None`,
   `active_overlay: str | None`, `active_run_name: str`, etc.
   This is what is "preserved across model swaps" in Phase 6: the
   swap mutates `active_run_name` but leaves the other ViewerCore
   fields unchanged.

The frozen request snapshot:

```python
@dataclass(frozen=True)
class ViewState:
    request_id: int
    t_center: float
    t_width: float
    load_acausal: bool   # only when overlay == "smoothed"
```

Stale-result rejection: each window-load worker tags its result with
`request_id`; when the result arrives on the main thread, the viewer
drops it if `request_id < latest_committed_request_id`. Guarantees a
fast scrub never displays the output of an in-flight request that has
been superseded.

The non-`ViewState` ViewerCore fields (pinned event, overlays,
active run) live separately because they don't influence which
window the worker should fetch — they're rendering-side
concerns that the panels read directly from `ViewerCore`.

### Panel ABCs

```python
class TimeAxisPanel(Protocol):
    """Window-based panel rendered in the left column."""

    def update_window(self, payload: WindowPayload) -> None: ...
    def x_link_target(self) -> Any: ...   # what to setXLink against
    def click_handler(self, callback: Callable[[float], None]) -> None: ...
    def set_event_overlays(self, overlays: list[EventOverlay]) -> None: ...
    """Replace this panel's event-overlay set. Called by ViewerCore
    whenever bundle.event_overlays or per-overlay visibility changes.
    Idempotent: a new list fully replaces previously rendered markers.
    Implementations draw both `points` overlays (vertical lines at
    each `time`) and `intervals` overlays (shaded vertical bands
    between `t_start[i]` and `t_end[i]`). The default Qt
    implementation in `panels.qt._mixins.EventOverlayMixin` uses
    `pg.InfiniteLine` and `pg.LinearRegionItem` so subclasses get
    overlay support for free."""

class BinSyncedPanel(Protocol):
    """Point-based panel rendered to the right of the time-axis stack."""

    def update_for_index(self, t_idx: int, payload: BinPayload) -> None: ...
```

Concrete panels in `panels/qt/` subclass the appropriate ABC plus the
shared `EventOverlayMixin` (which provides the default
`set_event_overlays` implementation against pyqtgraph primitives) and
implement Qt-specific rendering. Project-specific panels do the same.
A panel only overrides `set_event_overlays` directly when its
coordinate system isn't a flat x-axis (e.g. a 2D heatmap mapping
event time to a different visual cue).

## v1 scope

### In

- **Sub-package**: `src/non_local_detector/visualization/interactive/`
  with optional dep group `[viewer]` (`PySide6`, `pyqtgraph`).
- **Sorted-spikes 1D** decoders only:
  - `SortedSpikesDecoder` (1 state, `MARGINAL` reduction).
  - `ContFragSortedSpikesClassifier` (2 states, `MARGINAL`
    reduction).
  - `NoSpikeContFragSortedSpikesClassifier` (3 states incl. one
    `No-Spike` singleton, `CONDITIONAL_ON_SPATIAL` reduction —
    the new strategy added in this plan to give NoSpikeContFrag a
    curve that integrates to 1.0 over position by conditioning on
    the spatial states).
  - `NonLocalSortedSpikesDetector` (4 states with at least one
    singleton, `CONDITIONAL_NON_LOCAL` reduction).

  All four sorted-spikes 1D single-environment detectors in
  `non_local_detector` are covered. Multi-environment and
  clusterless variants are deferred to v3+ — see "Deferred-model
  pathway" near the end of the plan for the v3+ design sketch.
- **In-memory** `DecoderDataSource` constructed from a dict of named
  `RunBundle`s (`{name: RunBundle}`). One run is active at a time;
  `set_active_run(name)` rebinds the source's hot-path arrays.
  Single-run callers use the `from_single(bundle)` classmethod. All
  runs in the dict must share the same time grid (they must be decoded
  from the same recording session); mismatched grids raise at
  construction with a message naming the offending pair.
- **View-models for all built-in panels**:
  - `PosteriorHeatmapModel` — schema-aware reduction (see "Posterior
    reductions per detector schema" below). Re-normalizes per-time,
    masks non-track bins.
  - `LikelihoodHeatmapModel` — schema-aware position-axis collapse
    per row. `log_likelihood` rows are over `state_bins`
    (shape `sum(detector.bin_sizes_)`, non-rectangular for NL —
    `[n_pos, 1, n_pos, n_pos]` for the Track 0 fixture's
    `local_position_std=1.0`). The model applies
    `analysis.posterior.collapse_log_likelihood_to_position(...)`
    (the same helper the SlicePanel top curve uses) to **every
    visible row** and stacks the results into a `(n_visible, n_pos)`
    array. This:
    - Maps non-finite bins to `-inf`, subtracts the per-row max,
      exponentiates (statespacecheck pattern — pure rescaling, no
      probability claim).
    - Sums across spatial-state bin slices (selects only states
      where `bin_sizes_[s] > 1`); singleton states (`No-Spike`,
      and `Local` when `local_position_std is None`) are dropped
      from the position axis.
    - Peak-normalizes per row for plotting.
    Output shape is consistently `(n_visible, n_pos)` regardless of
    whether the detector is rectangular (Decoder / ContFrag) or
    non-rectangular (NL). Title-bar text mirrors the SlicePanel:
    "Likelihood across all spatial states; heatmap below shows
    non-local states only" (so users see why the likelihood and
    posterior heatmaps may diverge for NL fits with
    `local_position_std=1.0`).
  - `StateProbabilityModel`.
  - `RasterModel`.
  - `SliceModel` — population likelihood + per-cell rows for the
    cursor's bin.
- **Qt panels** consuming each view-model (the things actually
  rendered): `panels/qt/posterior.py`, etc.
- **`TimeAxisPanel` and `BinSyncedPanel` ABCs** — public API for
  downstream packages.
- **Generic series panels** — `LineSeriesPanel` (with `fill_below=True`
  and `thresholds=[...]` options), `MultiLineSeriesPanel`,
  `ScatterSeriesPanel`, `IntervalSeriesPanel` in `panels/qt/series.py`,
  consuming `view_models/series.py`. Plus a `MetricSpec` dataclass for
  the zero-code `bundle.extra_metrics` path. `ScatterSeriesPanel` wires
  click-to-recenter by default. Together with the built-in panels these
  cover every panel kind in the static `plot_detector` (see "Coverage
  of `plot_detector`" in the Decided section).
- **Event overlays** — `EventOverlay.points(...)` and
  `EventOverlay.intervals(...)` constructors plus a viewer-level
  rendering loop that draws marker lines / shaded bands on every
  time-axis panel. Multi-overlay support; per-overlay visibility
  toggle; navigator dropdown + `N` / `Shift+N` shortcuts to jump
  between events of the active overlay.
- **DecoderViewer** Qt window: posterior + likelihood + raster + state
  prob in left column; SlicePanel in right column. Center-time slider,
  threadpool window-load worker, ring buffer for per-tick path,
  stale-result rejection.
- **Keyboard nav**: ←/→ (one bin), Shift+←/→ (one window), Space
  (play/pause auto-scroll), `[` / `]` (window width), `R` (reset),
  `Esc` (unpin), `M` (cycle to next model run when ≥2 runs are
  loaded), `N` / `Shift+N` (next / previous event in the active
  overlay).
- **Click handlers**: click a raster spike to pin; click empty space
  in any time-axis panel to recenter on the click x.
- **Model swap UI**: "Model" dropdown in the controls bar (statespacecheck
  pattern). M-key cycles to the next run. Visible only when the data
  source has more than one run.
- **CLI entry**: `python -m non_local_detector.visualization.interactive`.
  Accepts `--run name:results.nc:model.pkl:spikes.npz:position.parquet`
  repeated for multi-run loading. CLI input file formats:

  - `results.nc`: NetCDF written from `predict()` output. Required
    variables: `acausal_posterior`, `acausal_state_probabilities`.
    Optional (gates additional panels): `log_likelihood`,
    `predictive_posterior` — see "Viewer-ready results contract"
    above for the panel-by-panel matrix.
  - `model.pkl`: joblib pickle of the fitted detector
    (`SortedSpikesDecoder`, `ContFragSortedSpikesClassifier`,
    `NoSpikeContFragSortedSpikesClassifier`, or
    `NonLocalSortedSpikesDetector`).
  - `spikes.npz`: NumPy file (`np.savez`) with one object-dtype array
    `spike_times` of shape `(n_neurons,)`, each entry an `np.ndarray`
    of float64 absolute spike timestamps in seconds. Cell ordering
    must match the detector's encoding-model neuron order.
  - `position.parquet`: pandas DataFrame written via
    `df.to_parquet(...)`. **Index** is absolute time in seconds
    (DatetimeIndex or float64 — float64 preferred for consistency
    with `position_time` semantics).
    [base.py:4003](../../src/non_local_detector/models/base.py#L4003)
    establishes that `position_time` is required whenever `position`
    is supplied; we follow the same convention. **Required column(s)**:
    `position` (1D detectors) or `x_position` + `y_position`
    (2D detectors, v3+). **Optional column**: `speed`.
    Document this schema in the `--help` output and the README.

  CLI's `--help` also documents the `predict(return_outputs=[...])`
  requirement for the optional results variables.

  - `--metric name:kind:path:column` (repeatable, optional) attaches a
    user-computed metric. `kind` is one of `line` / `scatter` /
    `interval`. `path` is a parquet file with a `time` index;
    `column` selects the value column (or `column,color` for an
    optional color literal). Adds the metric as an `extra_metrics`
    entry on the constructed `RunBundle`.
- **Notebook entry**:
  `non_local_detector.visualization.interactive.launch(...)` accepts
  either a single `RunBundle` or a dict of named `RunBundle`s. v1 ships
  this notebook callable alongside the CLI; v2 will add a parallel
  `launch_in_browser(...)` for the Panel backend.
- **CI gate** that fails if `pyqtgraph` or `PySide6` is imported
  anywhere outside `viewer/qt.py` and `panels/qt/` — i.e. flags
  any such import in `__init__.py`, `app.py`, `data_source.py`,
  `view_models/`, `viewer/core.py`, or `viewer/backend.py`. Plus a
  parallel rule for `encoding_model_[...]` access outside
  `analysis/place_fields.py` and `models/base.py` (the writer).
  See "Hard architectural rule" in the Decided section for the
  rationale and lazy-dispatch pattern.

### Out (v2 / v3)

- **v2**: Panel/holoviews/bokeh backend (`panels/panel_/`) consuming
  the same view-models. Notebook-served browser viewer for remote work.
- **v2**: `ZarrDecoderDataSource` for sessions too large to fit in
  memory.
- **v2**: Diagnostic-metric panels (HPD overlap, KL divergence, spike
  prob) — `MetricPanel` from statespacecheck. Requires porting
  `event_*` computations to `non_local_detector.analysis`.
- **v3**: 2D position decoder support (movie-style posterior, 2D
  per-cell place fields). v1's `PositionGrid` abstraction makes this
  swap-in.
- **v3**: Animal-video overlay panel (a `BinSyncedPanel` that loads a
  video and alpha-blends the 2D posterior on the current frame).
- **v3**: Side-by-side model comparison (two parallel column stacks
  in the same window, sharing the center-time slider).
- **v3+**: Multi-environment classifiers
  (`MultiEnvironmentClusterlessClassifier`).
- **v3+**: Clusterless decoders.
- 2D figurl-style viewer.

### Explicitly not deferred — keep in v1

- The data source must be a clean abstraction with one method per
  hot-path read.
- The view-model / Qt-renderer split must be enforced (CI gate).
- The `TimeAxisPanel` / `BinSyncedPanel` ABCs must be public API at v1
  with a deprecation policy — downstream packages depend on them
  immediately.

## Phase plan

Each phase ends with a checkpoint where you review before I move on.
Each phase can be its own PR.

Phase 1 is split into three reviewable sub-PRs (1a, 1b, 1c). Each is
independently mergeable; the viewer doesn't import any new code until 1c.
Splitting keeps each PR's surface small enough to review carefully and
easy to roll back if needed.

### Phase 1a — Analysis helpers (repo cleanup, no viewer scaffolding)

Pure repo-level cleanup: introduces general detector utilities and
refactors the existing static plot to use them. No viewer code, no new
optional dependencies, no `pyqtgraph` / `PySide6` imports anywhere.
Reviewable as a standalone PR by anyone familiar with the existing
plotting code.

- `src/non_local_detector/analysis/place_fields.py` (new module)
  with `extract_per_cell_place_fields(detector)` and
  `extract_state_aligned_place_fields(detector)` (see "Place-field
  extraction" in the math section). Re-exported from
  `non_local_detector.analysis.__init__`.
- `src/non_local_detector/analysis/posterior.py`: add three helpers,
  re-exported from `non_local_detector.analysis.__init__`:
  - `conditional_non_local_posterior(results, detector,
    zero_mass_fill=np.nan)` — extracted from the inlined algorithm in
    [static.py:166](../../src/non_local_detector/visualization/static.py#L166).
    **Dataset-level helper** that allocates a full-session
    `(n_time, n_pos)` array. Used **only** by the static plot
    `plot_non_local_model` (and any other caller that genuinely
    needs the entire session reduced at once). View-models —
    `PosteriorHeatmapModel`, the SlicePanel posterior fallback,
    the predictive overlay — must **not** call this on every
    window update; they call the row-level
    `collapse_posterior_to_position(post_row, detector,
    CONDITIONAL_NON_LOCAL)` instead, which delegates to the same
    private `_conditional_row(...)` this dataset-level helper
    iterates over (so output is bit-identical when both paths run
    on the same row). See "PosteriorHeatmapModel" in Phase 1c and
    the "Two distinct collapse helpers" block in Phase 4 for the
    canonical view-model call pattern.
  - `collapse_log_likelihood_to_position(log_lik_row, detector)` —
    schema-aware log-domain collapse with the all-non-finite
    short-circuit (full algorithm specified in
    "SliceModel — Top likelihood curve" below; promoted here to
    Phase 1a because Phase 3's `LikelihoodHeatmapModel` and Phase 4's
    `SliceModel` both consume it). Pure log-likelihood handling: no
    probability normalization, only peak-normalization for display.
  - `collapse_posterior_to_position(post_row, detector, reduction,
    zero_mass_fill=np.nan)` —
    probability-domain analogue. Selects state-bin slices and
    column-sums them; **does no additional renormalization beyond
    what each strategy explicitly requires**. The `zero_mass_fill`
    parameter (default `np.nan`) controls what **both
    `CONDITIONAL_*` paths** (`CONDITIONAL_NON_LOCAL` and
    `CONDITIONAL_ON_SPATIAL`) write on zero-mass rows — they share
    the same `_conditional_row` private helper, so the fill
    semantics are identical. The `MARGINAL` path ignores it (it
    never divides). Mirrors the same-named parameter on the
    dataset-level `conditional_non_local_posterior` so callers can
    pick consistent fill semantics across all three call sites.
    - `MARGINAL`: sum the **spatial-state** slices (`bin_sizes_[s] >
      1`) column-wise; return the result as-is. No conditional
      renormalization. The helper output is a full-position row with
      non-interior bins NaN-padded (the spatial-state slices'
      non-interior columns inherit the NaN that
      `_create_masked_posterior` wrote at
      [base.py:2241](../../src/non_local_detector/models/base.py#L2241);
      column-summing them keeps the NaN in place). Output sums to
      1.0 over the **interior** portion of the row for fully-spatial
      schemas (`SortedSpikesDecoder`, `ContFragSortedSpikesClassifier`)
      because `post_row` sums to 1.0 across its interior bins.
      For NL with `MARGINAL` (a non-default user override, since
      `select_reduction` picks `CONDITIONAL_NON_LOCAL` for NL),
      the result is genuinely <1.0 — it equals the **singleton-state
      complement**, evaluated with `np.nansum` to skip the
      NaN-padded non-interior bins on both sides:

      ```text
      np.nansum(out, axis=-1) == 1 - np.nansum(post[:, singleton_bins], axis=-1)
      ```

      where `singleton_bins = ~(bin_sizes_[state_ind_] > 1)` — i.e.
      whichever bins belong to states that the construction-time rule
      [base.py:1034](../../src/non_local_detector/models/base.py#L1034)
      `obs.is_no_spike or (obs.is_local and local_position_std is None)`
      flagged as singletons. Concretely:
      - Track 0 fixture (`local_position_std=1.0`,
        `bin_sizes_=[n_pos, 1, n_pos, n_pos]`): only `No-Spike` is
        singleton, so `MARGINAL` sums to `1 - P(No-Spike)`.
      - Deferred-marker fixture (`local_position_std=None`,
        `bin_sizes_=[1, 1, n_pos, n_pos]`): both `Local` and
        `No-Spike` are singletons, so `MARGINAL` sums to
        `1 - P(Local) - P(No-Spike)`.

      That is the correct mathematical marginal; users who want a
      curve that integrates to 1.0 should use one of the
      `CONDITIONAL_*` strategies (which are also the auto-detect
      defaults for any schema with singleton states).
    - `CONDITIONAL_ON_SPATIAL`: sum the **spatial-state** slices
      (`bin_sizes_[s] > 1`), then divide by their total mass
      (renormalization is intrinsic to "conditional"). Default for
      `NoSpikeContFragSortedSpikesClassifier` (state_names
      `["No-Spike", "Continuous", "Fragmented"]` —
      `select_reduction` picks `CONDITIONAL_ON_SPATIAL` because
      no state name contains `"Non-Local"` but `bin_sizes_` has a
      singleton). Output sums to 1.0 on rows with positive
      spatial mass; on zero-mass rows the helper writes
      `zero_mass_fill`. For NL with this strategy as a non-default
      override, `Local` is included when it is spatial (i.e.
      `local_position_std=1.0`) — that is the difference from
      `CONDITIONAL_NON_LOCAL` which excludes Local by name.
    - `CONDITIONAL_NON_LOCAL`: sum the slices for states whose
      name contains `"Non-Local"`, then divide by their total mass
      (renormalization is intrinsic to "conditional"). Default for
      `NonLocalSortedSpikesDetector` and statespacecheck's paper
      view. Output sums to 1.0 on rows with positive non-local
      mass; on zero-mass rows the helper writes `zero_mass_fill`.
    - **Shared `_conditional_row(post_row, detector,
      selected_state_ids, zero_mass_fill)` private helper.**
      Both `CONDITIONAL_*` strategies route through this single
      private function — they differ only in the
      `selected_state_ids` they pass. `selected_state_ids` is a
      sequence of **discrete state ids** (indices into
      `detector.state_names`, length ≤ `n_discrete_states_`), not
      state-bin column indices into `state_ind_`. The helper
      internally expands those state ids to a column mask via
      `np.isin(detector.state_ind_, selected_state_ids)` before
      summing and dividing — caller-side code stays in the
      smaller, more intuitive state-id space.

      Strategy → `selected_state_ids` map:
      - `CONDITIONAL_ON_SPATIAL` → state ids where
        `detector.bin_sizes_[s] > 1`.
      - `CONDITIONAL_NON_LOCAL` → state ids where
        `"Non-Local" in detector.state_names[s]`.

      The dataset-level `conditional_non_local_posterior(results,
      detector)` (extracted from the algorithm at
      [static.py:155-169](../../src/non_local_detector/visualization/static.py#L155))
      iterates time and calls this private helper per row with
      the non-local indices. `collapse_posterior_to_position(...,
      reduction)` calls it directly per row with the
      strategy-appropriate indices. This is what "single-row
      variant" refers to throughout the rest of the plan.

    Used by the `LikelihoodHeatmapModel`'s "log_likelihood
    missing" fallback, the SlicePanel predictive / smoothed
    overlay, and the per-cell row predictive overlay (all Phase 3
    / Phase 4 consumers).

  Placing all three in Phase 1a means every Phase 3 / Phase 4 panel
  view-model imports a stable, tested helper rather than introducing
  one alongside its first consumer.
- `src/non_local_detector/visualization/static.py`: refactor
  [`static.py:128`](../../src/non_local_detector/visualization/static.py#L128)
  to call `extract_per_cell_place_fields(detector)`, and refactor
  [`static.py:166`](../../src/non_local_detector/visualization/static.py#L166)
  to call `conditional_non_local_posterior(...)`. Assert the
  computed conditional-non-local-posterior array is bit-identical
  before/after via `np.testing.assert_allclose(after, before,
  atol=1e-14, equal_nan=True)` — bare `==` would fail on the
  matching NaNs at non-interior positions written by the
  algorithm at
  [static.py:169](../../src/non_local_detector/visualization/static.py#L169).
- Property tests for both place-field helpers (against Track 0
  simulated detectors — **all four v1 detectors**):
  - `extract_per_cell_place_fields` succeeds for
    `SortedSpikesDecoder`, `ContFragSortedSpikesClassifier`,
    `NoSpikeContFragSortedSpikesClassifier`, and
    `NonLocalSortedSpikesDetector` (all single encoding-model entry
    in v1 defaults); shape always `(n_cells, n_pos)`.
  - `extract_state_aligned_place_fields` succeeds for
    `SortedSpikesDecoder` (shape `(n_cells, n_pos)`) and
    `ContFragSortedSpikesClassifier` (shape `(n_cells, 2 * n_pos)`,
    with the result being two horizontal copies of the per-cell
    output since ContFrag's two `ObservationModel()` entries share
    one encoding group); **raises `ValueError` for both
    detectors with singleton states**:
    - `NoSpikeContFragSortedSpikesClassifier` —
      `bin_sizes_ = [1, n_pos, n_pos]`, `No-Spike` singleton.
    - `NonLocalSortedSpikesDetector` — `bin_sizes_` always contains
      at least one singleton (`No-Spike`): `[n_pos, 1, n_pos, n_pos]`
      for the Track 0 fixture's `local_position_std=1.0` config,
      `[1, 1, n_pos, n_pos]` for the deferred-marker
      `local_position_std=None` config.

    In every case, the error message names the offending
    `bin_sizes_` and points the user to
    `extract_per_cell_place_fields` for per-cell display.
- Multi-entry detector → clear error: hand-construct a detector with
  two distinct encoding-model keys; assert
  `extract_per_cell_place_fields` raises with both keys named.
- Property tests for the two new collapse helpers
  (`tests/analysis/test_posterior_collapse.py`):
  - `collapse_log_likelihood_to_position`:
    - Track 0 NL `nl_loglik` row → output shape `(n_pos,)`,
      finite, peak ≈ 1.0 after normalization.
    - Track 0 ContFrag `cf_loglik` row → same shape, peak ≈ 1.0.
    - **All-non-finite row guard**: pass an array of all-NaN values;
      assert output is exactly `np.zeros(n_pos)` (matches the
      step-3 short-circuit in the helper algorithm). Also pass an
      array of all-`-inf` values; assert same all-zero output.
    - **Mixed finite + `-inf` row**: pass a row where one spatial
      state's slice is all `-inf` and the other(s) have finite
      values; assert the `-inf` slice's columns contribute exactly
      `0.0` (via `exp(-inf)`) without producing NaN anywhere.
    - **No spatial states** edge case: hand-construct a detector
      where all `bin_sizes_` are 1 (purely singleton); assert the
      helper raises `ValueError` with a clear message
      ("Detector has no spatial states.").
  - `collapse_posterior_to_position`:
    - **Shared row-level math**: both
      `collapse_posterior_to_position(..., reduction=CONDITIONAL_NON_LOCAL)`
      and the dataset-level `conditional_non_local_posterior(results,
      detector, ...)` are required to route through the same generic
      private `_conditional_row(post_row, detector,
      selected_state_ids, zero_mass_fill)`. Both call sites pass
      `selected_state_ids` = the **discrete state ids** (indices
      into `detector.state_names`) for states whose name contains
      `"Non-Local"`; the helper internally expands those to a bin
      mask via `np.isin(detector.state_ind_, selected_state_ids)`
      (the dataset-level helper computes the state-id list once and
      iterates over time; `collapse_posterior_to_position` with
      `reduction=CONDITIONAL_NON_LOCAL` does it per row). The
      `CONDITIONAL_ON_SPATIAL` strategy uses the same private helper
      with `selected_state_ids` = state ids where
      `detector.bin_sizes_[s] > 1` instead. Test against the
      private function directly — the dataset-level helper takes an
      `xr.Dataset` and reads `results.acausal_posterior`
      ([static.py:98](../../src/non_local_detector/visualization/static.py#L98),
      [static.py:139](../../src/non_local_detector/visualization/static.py#L139)),
      so a bare NumPy row would not satisfy its contract.
      Concretely (all bit-identity checks use
      `np.array_equal(actual, expected, equal_nan=True)` because the
      conditional posterior intentionally writes NaN at non-interior
      positions per
      [static.py:169](../../src/non_local_detector/visualization/static.py#L169)
      — plain `==` fails on matching NaNs since `nan == nan` is
      `False` per IEEE-754):
      - Pick a representative `t_idx` from the Track 0 NL bundle.
        Compute the non-local-state ids
        (`nl_state_ids = np.flatnonzero(["Non-Local" in s for s in
        detector.state_names])`) and then
        `expected = _conditional_row(
        results.acausal_posterior.isel(time=t_idx).values, detector,
        selected_state_ids=nl_state_ids, zero_mass_fill=np.nan)`.
      - Assert `np.array_equal(collapse_posterior_to_position(
        results.acausal_posterior.isel(time=t_idx).values, detector,
        CONDITIONAL_NON_LOCAL, zero_mass_fill=np.nan), expected,
        equal_nan=True)`.
      - Assert `np.array_equal(
        conditional_non_local_posterior(results, detector,
        zero_mass_fill=np.nan).isel(time=t_idx).values, expected,
        equal_nan=True)` (the dataset-level helper iterates over
        time and calls the private row helper per row; row `t_idx`
        of its output must equal what the private helper produces
        directly).
      - Together these prove both public helpers route through one
        private implementation rather than re-implementing the
        algorithm.
    - **`zero_mass_fill` plumbing**: construct a synthetic
      `post_row` where the non-local-state-bin mass is exactly 0
      (a "no non-local activity" bin). Assert that
      `collapse_posterior_to_position(post_row, det,
      CONDITIONAL_NON_LOCAL, zero_mass_fill=np.nan)` returns an
      all-NaN row (use
      `np.all(np.isnan(out))`, **not** `out == np.full(..., np.nan)`)
      and with `zero_mass_fill=0.0` returns an all-zero row
      (`np.array_equal(out, np.zeros(n_pos))`). Build a
      corresponding one-row `xr.Dataset` (with `acausal_posterior`
      carrying that row plus the matching `state_bins` coordinate)
      and assert `conditional_non_local_posterior(...)` produces
      the same single-row outputs under each fill setting (using
      `np.all(np.isnan(...))` for the NaN case and
      `np.array_equal(..., equal_nan=True)` for cross-helper
      equivalence) — confirming both helpers honor the parameter
      consistently end-to-end.
    - ContFrag and Decoder with `reduction=MARGINAL` → output is
      a column-sum of spatial-state slices with **no additional
      renormalization**. Because both detectors are fully spatial
      and `acausal_posterior` sums to 1.0 over the interior portion
      of the full state_bins axis, the helper's output (a
      full-position row with non-interior bins NaN-padded by
      `_create_masked_posterior`) has interior bins summing to 1.0.
      Assert
      `np.allclose(np.nansum(out, axis=-1), 1.0, atol=1e-10)` for
      a typical row (or equivalently `out[..., interior].sum(axis=-1)`
      with the position-axis interior mask), **without** the
      helper performing a divide step.
    - NL with `reduction=MARGINAL` (non-default override): output
      is the genuine spatial-state marginal — row sum equals
      `1 - np.nansum(post[:, singleton_bins], axis=-1)` per row,
      not 1.0, where `singleton_bins` is the set of bins whose
      `bin_sizes_[state_ind_[bin]] == 1`. For the Track 0 NL
      fixture (`local_position_std=1.0`) only `No-Spike` is
      singleton so this reduces to `1 - P(No-Spike)`; for the
      deferred `local_position_std=None` fixture it reduces to
      `1 - P(Local) - P(No-Spike)`. Assert
      `np.allclose(np.nansum(out, axis=-1),
       1.0 - np.nansum(post[:, singleton_bins], axis=-1),
       atol=1e-10)`. Test against both fixtures (the slow-marker
      one as a `@pytest.mark.slow` parametrization) so a future
      implementer who "fixes" the marginal to renormalize gets
      caught against either schema.
    - **NoSpikeContFrag with `reduction=CONDITIONAL_ON_SPATIAL`
      (the default for this schema)**: against the Track 0 `nsf_*`
      bundles, the helper is row-level so the test loops over
      representative `t_idx` values from the dataset (or picks one
      and asserts a single row). For each chosen `t_idx`, with
      `post_row = results.acausal_posterior.isel(time=t_idx).values`
      and `out = collapse_posterior_to_position(post_row, detector,
      CONDITIONAL_ON_SPATIAL, zero_mass_fill=np.nan)`, assert:
      - `out.shape == (n_pos,)`.
      - When `spatial_mass_row = 1 - results.acausal_state_probabilities
        .sel(states="No-Spike").isel(time=t_idx).values` is positive:
        `np.allclose(np.nansum(out), 1.0, atol=1e-10)` (scalar — the
        strategy renormalizes by spatial mass so the row integrates
        to 1.0).
      - The implicit divisor matches `1 - P(No-Spike)`: assert
        `np.allclose(out * spatial_mass_row,
         spatial_column_sum_row, atol=1e-10)` where
        `spatial_column_sum_row = np.nansum(
         post_row[..., spatial_state_bins].reshape(n_spatial, n_pos),
         axis=0)` is the `(n_pos,)` column-sum of the spatial-state
        slices for that one row — proves the helper conditions on
        the right state set. (Use the corresponding batch form —
        `out * spatial_mass[:, None]` against a `(n_time, n_pos)`
        column-sum array — only when iterating over multiple rows
        and stacking; the per-row form here is the contract.)
      - Cross-helper consistency (single row): pick `post_row` as
        above and call `_conditional_row(post_row, detector,
        selected_state_ids=[1, 2],   # Continuous + Fragmented
        zero_mass_fill=np.nan)`. Assert
        `np.array_equal(collapse_posterior_to_position(post_row,
        detector, CONDITIONAL_ON_SPATIAL, zero_mass_fill=np.nan),
        _conditional_row(...), equal_nan=True)`.
      - `zero_mass_fill` plumbing: same NaN/0.0 dual assertion as
        the CONDITIONAL_NON_LOCAL test above, but via the
        `CONDITIONAL_ON_SPATIAL` path against a synthetic
        single-row `post_row` with `P(No-Spike) = 1.0` (zero
        spatial mass on that row).
    - Output shape `(n_pos,)` regardless of detector schema.

**Verification** (Track 0):

`plot_non_local_model` is currently exercised only from notebooks
([sorted_spikes_detector_test.py:68](../../notebooks/01_models_and_validation/sorted_spikes_detector_test.py#L68))
and the existing
[golden regression tests](../../src/non_local_detector/tests/test_golden_regression.py)
compare decoder posterior arrays directly, **not** the static plot's
output. Existing tests passing therefore does not prove the static-plot
refactor is bit-identical. Phase 1a adds dedicated refactor tests:

1. `tests/visualization/test_static_plot_refactor.py::test_conditional_non_local_posterior_matches_inline`:
   - Build the Track 0 simulated NL bundle.
   - Compute the conditional non-local posterior using the inline
     algorithm pasted verbatim from current
     [static.py:155–169](../../src/non_local_detector/visualization/static.py#L155)
     into the test fixture.
   - Compute via the new
     `analysis.posterior.conditional_non_local_posterior(results, detector)`.
   - Assert `np.allclose(inline_result, helper_result, equal_nan=True,
     atol=1e-14)`.
2. `tests/visualization/test_static_plot_refactor.py::test_place_field_peak_sort_matches_inline`:
   - Same fixture.
   - Compute the neuron sort order using the inline pattern from
     [static.py:128–131](../../src/non_local_detector/visualization/static.py#L128).
   - Compute via the new helper:
     `np.argsort(env.place_bin_centers_[np.nanargmax(extract_per_cell_place_fields(detector), axis=1)].squeeze())`.
   - Assert sort orders are identical.
3. `tests/analysis/test_place_fields.py` covers the helpers'
   contractual shape + multi-entry-error tests (already enumerated
   in the deliverables list above).

Plus: `uv run pytest -k "place_fields or posterior_helper or
static_plot_refactor"` passes; the existing golden / snapshot tests
remain green (sanity check that no decoder-side behavior shifted, even
though those tests don't directly cover the static-plot path).

### Phase 1b — RunBundle + in-memory data source

Introduces the viewer sub-package shell + the data-flow contract,
still without any rendering code or GUI deps. Reviewable by someone
checking the data-flow surface.

- Create `src/non_local_detector/visualization/interactive/` package
  with `__init__.py`, `view_models/__init__.py`, `view_models/base.py`,
  `view_models/events.py`, `view_models/series.py` (carrying only
  the `MetricSpec` dataclass at this phase; the series-model classes
  ship in Phase 3 in the same file), `data_source.py`. No panels,
  no viewer module yet.
- Add `[viewer]` optional-dependency group in `pyproject.toml`.
- `view_models/events.py`: `EventOverlay` dataclass with
  `.points(name, times, color, ...)` /
  `.intervals(name, t_start, t_end, color, ...)` constructors.
  Created in Phase 1b — not Phase 3 — because `RunBundle.event_overlays`
  and the `TimeAxisPanel.set_event_overlays(...)` Protocol method
  both reference this type, and the panel ABC lands in Phase 1c. The
  Phase 3 deliverables for event overlays add only the Qt rendering
  mixin and the viewer-side dispatch loop, not the dataclass itself.
- `view_models/series.py`: `MetricSpec` dataclass union with
  `MetricSpec.line(...)`, `MetricSpec.scatter(...)`, and
  `MetricSpec.intervals(...)` constructors. Created in Phase 1b
  (not Phase 3) for the same reason as `view_models/events.py`:
  `RunBundle.extra_metrics` declares `dict[str, MetricSpec |
  pd.Series]` as its type, so the dataclass must exist before
  Phase 1c's `view_models/base.py` imports it. The model
  classes that consume `MetricSpec` (`LineSeriesModel`,
  `MultiLineSeriesModel`, `ScatterSeriesModel`,
  `IntervalSeriesModel`) land in Phase 3 in the same file —
  Phase 3 adds them alongside, but does not redefine
  `MetricSpec`.
- `view_models/base.py`: `RunBundle` dataclass (`results`, `detector`,
  `spike_times`, `position_time`, `position`, optional `speed`,
  optional `events`, `extra_metrics`, `event_overlays`).
  `__post_init__` validates internal consistency
  (monotonic `position_time`, position dimensionality matches detector
  environment, `len(spike_times) == n_neurons`). `from_predict(...)`
  classmethod for the common case. Imports `MetricSpec` from
  `view_models/series.py` for the `extra_metrics` field type.
  `view_models/base.py` imports `EventOverlay` from
  `view_models/events.py` for the `event_overlays` field's type
  annotation; `view_models/__init__.py` re-exports both.
- `data_source.py`: `InMemoryDecoderDataSource(runs: dict[str,
  RunBundle])` plus a `from_single(...)` classmethod. Hot-path
  methods: `window_indices`, `load_posterior`, `load_likelihood`
  (raises clearly when `log_likelihood` not in results),
  `load_acausal`, `load_predictive`, `slice_at_index`,
  `events_in_window`. Source holds `active_run`,
  `set_active_run(name)`, and `available_outputs: set[str]` derived
  from the active run's results variables. Construction validates:
  (a) time-grid alignment across all runs;
  (b) overlay names are unique within each bundle (so
  `active_overlay_name` resolves unambiguously);
  (c) overlay `(name, kind)` schema matches across all runs (so the
  navigator's `active_overlay_name` survives swaps with consistent
  N / Shift+N semantics — same name must mean same kind, points
  vs intervals).
  See "Event overlays — multi-run alignment" in the Decided section
  for the full pseudocode and rationale. Raises clearly on any of
  the three mismatches with the offending bundle name(s) and the
  conflicting schemas in the message.
- CI gate: a `pytest` step enforcing the **global allowlist** from
  the "Hard architectural rule" section in Decided. The gate
  inspects **Python `import` statements only** (not arbitrary
  string occurrences) within
  `src/non_local_detector/visualization/interactive/**/*.py`,
  excluding the allowlisted Qt sources (`viewer/qt.py` and
  `panels/qt/**/*.py`). Implementation: walk each `*.py` file in
  the subtree with `ast.parse(...)` and inspect every `Import` /
  `ImportFrom` node — fail if any imports `pyqtgraph` or `PySide6`
  (or a submodule of either) and the file is not under an
  allowlisted Qt path. **AST-based, not grep**, because:
  - The same phase adds `PySide6` and `pyqtgraph` as `[viewer]`
    optional dependencies in `pyproject.toml` (those mentions are
    legitimate), and the deferred-model section names them in
    prose. A `grep pyqtgraph` would match those.
  - String-mention checks would also flag in-comment references
    or docstring examples that don't actually import the modules.

  Pseudocode:

  ```python
  import ast, pathlib
  ROOT = pathlib.Path("src/non_local_detector/visualization/interactive")
  ALLOWLIST = {"viewer/qt.py", "panels/qt/"}  # path prefixes
  BANNED = {"pyqtgraph", "PySide6"}

  for path in ROOT.rglob("*.py"):
      rel = path.relative_to(ROOT)
      if any(str(rel).startswith(p) for p in ALLOWLIST):
          continue
      tree = ast.parse(path.read_text())
      for node in ast.walk(tree):
          modules = []
          if isinstance(node, ast.Import):
              modules = [alias.name for alias in node.names]
          elif isinstance(node, ast.ImportFrom):
              modules = [node.module] if node.module else []
          for m in modules:
              top = (m or "").split(".")[0]
              if top in BANNED:
                  raise AssertionError(
                      f"{path}: imports {m!r} but is not under "
                      f"the Qt allowlist ({sorted(ALLOWLIST)})."
                  )
  ```

  A parallel rule for `detector.encoding_model_[...]` access uses
  the same AST approach but a **broader scan scope**: it walks
  every `*.py` under **`src/non_local_detector/`** (not just the
  interactive subtree). The two rules legitimately have different
  scopes:

  - **Qt-import rule** is interactive-only. Qt code can appear in
    tests, notebooks, or other tools without violating the
    architecture; we only enforce GUI-toolkit-free internals
    inside the viewer subpackage.
  - **`encoding_model_[...]` rule** is repo-wide. After Phase 1a's
    refactor of [`static.py:128`](../../src/non_local_detector/visualization/static.py#L128)
    to call `extract_per_cell_place_fields(detector)`, the only
    legitimate `encoding_model_[...]` access in the entire repo is
    inside `analysis/place_fields.py` (the helper) and
    `models/base.py` (the writer that builds the dict). Any new
    consumer should go through the helpers, so the rule scans
    repo-wide to catch regressions anywhere — not just inside the
    viewer.

  Walks for `Subscript` nodes whose `value` is an `Attribute`
  matching `*.encoding_model_`, and fails the build if the
  containing file is outside the allowlist
  (`src/non_local_detector/analysis/place_fields.py` and
  `src/non_local_detector/models/base.py`). Both rules ship as a
  single `tests/lint/test_import_boundary.py` (or equivalent)
  that runs in the default `pytest` invocation; the file
  contains two separate test functions whose path-walk roots
  differ. Both rules fail the build on any match.

  Note that the Qt-allowlist source paths
  (`viewer/qt.py`, `panels/qt/`) don't exist yet at Phase 1b —
  those land in Phase 2 — but the gate's checker code references
  them as future-proof allowlist entries so the rule doesn't need
  a follow-up patch when those files appear.
- Smoke tests for the **viewer-ready results contract**: build three
  RunBundles from one fitted detector — default `predict()`,
  `return_outputs=["log_likelihood"]`, `return_outputs="all"` —
  assert `available_outputs` reflects each, that `load_likelihood`
  succeeds on the loglik-bearing variants and raises clearly on the
  default one. Multi-run construction with mismatched time grids →
  clear construction error.

**Verification** (Track 0): `uv run pytest -k "data_source or
run_bundle"` passes against the simulated fixture's twelve `RunBundle`
variants; default `uv sync` does not pull `PySide6`;
`uv sync --extra viewer` does.

### Phase 1c — First view-model + panel ABCs (still no Qt)

Closes out the Phase 1 contract surface: the first concrete view-model
and the panel protocols. Still no rendering, no Qt, no viewer harness.
Reviewable as the final no-GUI piece before Phase 2 wires it all up.

- `view_models/base.py`: add `ViewState` (frozen),
  `PositionGrid`, `WindowPayload`, `BinPayload`, `CellSlice`
  dataclasses.
- `view_models/posterior.py`: `PosteriorHeatmapModel.update_window(...)`
  returning a `(n_visible, n_pos)` array. Uses
  `select_reduction(state_names, bin_sizes_)` to pick the strategy
  at construction time, then routes per row through
  `analysis.posterior.collapse_posterior_to_position(post_row,
  detector, reduction)` for **every** strategy — the same helper
  the SlicePanel uses. This means:
  - NL detectors →
    `collapse_posterior_to_position(..., CONDITIONAL_NON_LOCAL)`
    per visible row (delegates to the same private
    `_conditional_row(..., selected_state_ids=non_local_state_ids,
    ...)` the dataset-level `conditional_non_local_posterior`
    uses, so heatmap rows match the static-plot reference
    bit-identically when both run on the same data).
  - NoSpikeContFrag →
    `collapse_posterior_to_position(..., CONDITIONAL_ON_SPATIAL)`
    per visible row (delegates to the same private
    `_conditional_row(..., selected_state_ids=spatial_state_ids,
    ...)`).
  - ContFrag and Decoder →
    `collapse_posterior_to_position(..., MARGINAL)` per visible
    row. The existing dataset-level
    `conditional_non_local_posterior` does **not** support these
    schemas (the static algorithm at
    [static.py:156](../../src/non_local_detector/visualization/static.py#L156)
    selects states by `"Non-Local" in state`); the
    `collapse_posterior_to_position` row helper handles them
    uniformly through the `MARGINAL` branch.

  Routing through `collapse_posterior_to_position` uniformly means
  PosteriorHeatmapModel needs one code path for all four reductions
  and inherits the helper's NaN-handling and zero-mass-fill
  semantics for free. (Performance note: the per-row Python loop
  over ~hundreds of visible rows per `update_window` call is
  acceptable for v1 — windows are small. If a future profile shows
  it as a hot spot, the helper can grow a vectorized over-time
  variant in Phase 1a without changing the model's call site.)
- `panels/__init__.py`, `panels/base.py`: `TimeAxisPanel` and
  `BinSyncedPanel` Protocols (including `set_event_overlays`).
- Property tests for `PosteriorHeatmapModel`, exercising **all four**
  v1 detectors from the Track 0 fixture:
  - **NL bundle**: assert `reduction == CONDITIONAL_NON_LOCAL` and
    that the rendered array matches the Phase 1a static-plot output
    within `atol=1e-6` (using
    `np.testing.assert_allclose(..., equal_nan=True)`).
  - **ContFrag bundle**: assert `reduction == MARGINAL` and that
    **every** row sums to 1.0 via `np.nansum(out, axis=-1)`
    (equivalently `out[..., interior].sum(axis=-1)`). The
    `MARGINAL` strategy never divides, so there is no active /
    empty row split — every row carries 1.0 of spatial mass for
    a fully-spatial detector. Drives the rectangular / multi-state
    path.
  - **NoSpikeContFrag bundle**: assert
    `reduction == CONDITIONAL_ON_SPATIAL` and that **every** row
    with positive spatial mass sums to 1.0 via `np.nansum`. Verify
    the spatial mass equals `1 - P(No-Spike)` per row by computing
    `1 - acausal_state_probabilities.sel(states="No-Spike").values`
    (label-based selection on the `states` coordinate at
    [base.py:2364](../../src/non_local_detector/models/base.py#L2364);
    `isel` would treat `"No-Spike"` as a positional index and
    raise) and asserting that's the divisor the helper used —
    i.e. `out * spatial_mass[:, None]` matches the column-summed
    spatial slices before division. On rows with zero spatial mass
    (which requires `P(No-Spike) ≈ 1.0` — rare in the Track 0
    fixture), `out` is the configured `zero_mass_fill` (default
    NaN). Drives the third reduction strategy added in this plan.
  - **SortedSpikesDecoder bundle**: assert `reduction == MARGINAL`
    on the single-state schema; every row sums to 1.0 via
    `np.nansum`. Same "no active/empty split" rule as ContFrag
    (single-state schemas trivially have full spatial mass).
  - **Schema-swap reduction test (data-source + view-model layer
    only — no viewer instantiation, since no viewer module exists
    until Phase 2):** construct an `InMemoryDecoderDataSource` from
    all four Track 0 bundles, then for each name in
    `("nl", "cf", "nsf", "dec")` call
    `data_source.set_active_run(name)`, construct a fresh
    `PosteriorHeatmapModel(state_names=
    data_source.active_run.detector.state_names,
    bin_sizes_=data_source.active_run.detector.bin_sizes_)` (using
    the documented `active_run` field on the data source — see
    Phase 1b API list), and assert the model's `reduction`
    attribute matches the expected strategy:
    - `NL → CONDITIONAL_NON_LOCAL`
    - `CF → MARGINAL`
    - `NSF → CONDITIONAL_ON_SPATIAL`
    - `Decoder → MARGINAL`

    Also verify that re-rendering the same `t_idx` after each swap
    produces a different output array (the smoothed posterior at
    one time bin will differ across detectors). The corresponding
    **viewer-level** swap test, where `M`-key triggers
    `set_active_run` plus view-state preservation and panel
    re-render, lives in Phase 6 against the actual `DecoderViewer`.

**Verification** (Track 0): `uv run pytest -k "view_models or
posterior_model"` passes; the Phase 1a + 1b + 1c stack is import-clean
without `[viewer]` deps installed (no pyqtgraph imports anywhere).

### Phase 2 — One Qt panel + viewer harness

- `panels/qt/posterior.py`: `QtPosteriorHeatmapPanel(pg.PlotWidget)`
  consuming `PosteriorHeatmapModel`. Tested numerically against the
  matplotlib version (rendered array `np.allclose`, `atol=1e-6`).
- `viewer/core.py`: `ViewerCore` (Qt-free) holding `ViewState`,
  active-run state, pinned-event state. `step_left()`,
  `step_right()`, `set_t_center()`, `request_load()` methods. Stale-
  result rejection via `request_id` comparison.
- `viewer/backend.py`: `BackendAdapter` `Protocol` with
  `schedule_window_load(state, on_done)` and
  `post_to_ui_thread(fn)`. No GUI imports.
- `viewer/qt.py`: `QApplication` creation +
  `QtViewer(QMainWindow)` + `QtBackendAdapter` implementing
  `BackendAdapter` via `QThreadPool` + `_LoadSignals` bridge
  object. Holds the one panel + center-time slider; binds ← / →
  keys to `core.step_*` methods. Exports `launch_qt(...)` —
  the function `app.py` lazy-imports.
- `app.py`: pure CLI / lazy-dispatch shim (no Qt imports at
  top level). `python -m non_local_detector.visualization.interactive
  --run default:results.nc:model.pkl:spikes.npz:position.parquet`
  parses args via argparse, constructs the `RunBundle`(s), then
  `from .viewer.qt import launch_qt; launch_qt(bundles, ...)` only
  after the backend is selected. v2 will add a parallel
  `from .viewer.panel_ import launch_panel` branch.
- Notebook callable: `launch(bundle)` (in `__init__.py`) where
  `bundle` is a `RunBundle` (or, for multi-run mode in Phase 6, a
  `dict[str, RunBundle]`). Itself a thin wrapper that lazy-imports
  the chosen backend.

**Verification** (Track 0): launches against the simulated `nl_bundle`;
slider scrubs; the rendered posterior matches `plot_non_local_model`'s
matplotlib output for the same time slice within `atol=1e-6` on the
underlying array (pixel-perfect rendering not required).

### Phase 3 — Remaining left-column panels + generic series panels

- `view_models/likelihood.py`, `view_models/state_prob.py`,
  `view_models/raster.py`.
- `panels/qt/likelihood.py`, `panels/qt/state_prob.py`,
  `panels/qt/raster.py`. X-axes linked.
- **Generic series panels** (this phase, not deferred):
  - `view_models/series.py`: add `LineSeriesModel`,
    `MultiLineSeriesModel`, `ScatterSeriesModel`, and
    `IntervalSeriesModel` to the file that already carries
    `MetricSpec` from Phase 1b (`MetricSpec.line(...)`,
    `.multi_line(...)`, `.scatter(...)`, `.intervals(...)`
    constructors). **Phase 3 does not redefine `MetricSpec`** —
    that dataclass shipped in Phase 1b alongside `RunBundle.extra_metrics`'s
    type annotation. `LineSeriesModel` carries `fill_below: bool`
    and `thresholds: list[float]` options for parity with
    `plot_detector`'s line panels. Pure data, no Qt.
  - `panels/qt/series.py`: `LineSeriesPanel`,
    `MultiLineSeriesPanel`, `ScatterSeriesPanel`,
    `IntervalSeriesPanel`. Each subclasses `TimeAxisPanel` and
    consumes the matching view-model.
  - `DecoderViewer` accepts an `extra_panels: list[TimeAxisPanel]`
    kwarg, plus auto-construction of panels from
    `bundle.extra_metrics` when `extra_panels` is unset.
  - `ScatterSeriesPanel.click_recenters=True` by default — clicks
    pin and recenter on the clicked point's time, matching the
    raster panel's UX.
- **Event overlays** (this phase — the `EventOverlay` dataclass
  itself ships in Phase 1b in `view_models/events.py`; this phase
  adds the panel-side rendering and the viewer-level dispatch):
  - `panels/base.py`: extend `TimeAxisPanel` Protocol with
    `set_event_overlays(overlays: list[EventOverlay]) -> None`
    (idempotent — replaces previously drawn markers). Imports
    `EventOverlay` from `view_models.events`.
  - `panels/qt/_mixins.py`: `EventOverlayMixin` providing the default
    Qt implementation against `pg.InfiniteLine` (points) and
    `pg.LinearRegionItem` (intervals). All four generic series
    panels and all four built-in left-column panels inherit it.
  - `viewer/core.py` gains overlay state + a render dispatch loop
    that calls `panel.set_event_overlays(visible_overlays)` on every
    panel when `bundle.event_overlays` or per-overlay visibility
    changes. Navigator (`next_event()` / `prev_event()`) on the core
    is also Qt-free.
  - `viewer/qt.py` controls bar: overlay selector dropdown + per-
    overlay visibility checkboxes; `N` / `Shift+N` keyboard
    shortcuts wired to `core.next_event()` / `core.prev_event()`.
  - Click-on-overlay-line recenters and pins (handled by the mixin's
    `pg.InfiniteLine.sigClicked`-style hookup), mirroring the
    raster pin/unpin UX.
- Wheel-over-time-axis = window-width scrub.
- `[`, `]`, `R`, Shift+←/→ shortcuts.

**Verification** (Track 0):

- Full left-column stack rendered against the simulated `nl_bundle`;
  visual diff against this repo's
  [`plot_non_local_model`](../../src/non_local_detector/visualization/static.py#L98)
  for an event window centered on `event_times[0]`. **Note:**
  `plot_non_local_model` renders raster, state probabilities,
  conditional non-local posterior, and speed —
  [no likelihood panel](../../src/non_local_detector/visualization/static.py#L145).
  The visual diff therefore covers four of the left-column panels;
  the LikelihoodHeatmapPanel needs its own targeted test (next
  bullet). Continuum-side parity (`plot_detector`) is **not**
  asserted in Phase 3 because Track 0's `make_simulated_data`
  returns plain arrays, not the project's `data` dict that
  `plot_detector` consumes. That visual comparison lands in
  **Phase 5** against Track B (continuum-formatted real session)
  where the dict shape is available.
- **LikelihoodHeatmapModel non-rectangular schema test** (closes
  the gap above): against the Track 0 `nl_loglik` bundle (NL
  detector with `bin_sizes_ = [n_pos, 1, n_pos, n_pos]`), call
  `LikelihoodHeatmapModel.update_window(...)` for a representative
  window, and assert:
  - Output array shape is exactly `(n_visible, n_pos)` —
    not `(n_visible, sum(bin_sizes_))` and not
    `(n_visible, n_pos * n_states)`.
  - Each row matches a freshly computed
    `analysis.posterior.collapse_log_likelihood_to_position(
        results.log_likelihood.isel(time=t).values, detector)`
    bit-identically — assert via
    `np.testing.assert_allclose(out_row, expected, atol=1e-14,
    equal_nan=True)`. The log-collapse helper produces no NaN by
    construction (non-finite inputs are mapped to `-inf` then
    exponentiated to `0.0`), so `equal_nan=True` is defensive but
    cheap and matches the convention used for posterior-domain
    comparisons elsewhere. This proves the heatmap and the
    SlicePanel top curve are collapsing through the same helper.
  - On rows that contain only non-finite log_likelihood
    (synthetic edge case — e.g. all-NaN window slice), the
    output row is all-zero (matches the helper's `-inf` →
    `exp(-inf) = 0` semantics).
  Repeat against the `cf_loglik` bundle (rectangular,
  `bin_sizes_ = [n_pos, n_pos]`) to confirm the same code path
  also works for the rectangular case (output shape
  `(n_visible, n_pos)` — Local + NL collapses to position-only,
  rectangular collapses similarly across the two states).
- Generic series panels exercised: attach a synthetic
  `pd.Series` (e.g. `is_event.astype(float)` resampled to a coarser
  rate) as `bundle.extra_metrics["is_event"]` and verify the viewer
  auto-renders it as a line panel with the right t-range. Repeat with
  `MetricSpec.scatter(...)` over `event_times[:, 0]` and assert
  click-to-recenter works.
- `LineSeriesPanel(fill_below=True, thresholds=[2.0])` renders both
  the filled area and the dashed threshold line; toggling
  `fill_below=False` removes the fill while keeping the line.
- `MultiLineSeriesPanel` renders a 3-line panel from a synthetic
  `{"a": ..., "b": ..., "c": ...}` ys dict with distinct colors and
  a shared y-range.
- Event overlays exercised: attach
  `bundle.event_overlays=[EventOverlay.intervals(name="sim events",
  t_start=event_times[:, 0], t_end=event_times[:, 1])]`, verify
  shaded bands render on every time-axis panel at consistent
  x-coordinates, and that `N` / `Shift+N` jump to next / previous
  event. Add a second `EventOverlay.points(...)` and verify both
  render simultaneously with their own colors and that the navigator
  follows the active overlay.

### Phase 4 — Right-column SlicePanel

- `view_models/slice.py`: `SliceModel.update_for_index(t_idx)` returning
  a `BinPayload`. **Important shape note**: `acausal_posterior`,
  `log_likelihood`, and `predictive_posterior` rows are all over
  `state_bins`
  ([base.py:2388](../../src/non_local_detector/models/base.py#L2388))
  whose length is `len(detector.state_ind_) =
  sum(detector.bin_sizes_)`. **The state grid is not always
  rectangular**: `NonLocalSortedSpikesDetector` is non-rectangular
  whenever any of its observation models is a singleton, which is
  controlled by the `local_position_std` constructor argument and
  the `is_no_spike` / `is_local` observation-model flags
  ([base.py:1029-1053](../../src/non_local_detector/models/base.py#L1029)).
  Two relevant configurations for v1:

  - `local_position_std=1.0` (Track 0 fixture default — fast to fit):
    `bin_sizes_ = [n_pos, 1, n_pos, n_pos]`. `Local` is a spatial
    state (with width `local_position_std` around the animal's
    position); `No-Spike` is the only singleton.
  - `local_position_std=None` (the original singleton-Local case):
    `bin_sizes_ = [1, 1, n_pos, n_pos]`. Both `Local` and
    `No-Spike` are singletons. Slow to fit; covered as a
    deferred-marker test (see Track 0 fixture notes below).

  Both are non-rectangular because at least `No-Spike` is a singleton,
  so naive `row.reshape(n_states, n_pos)` raises `ValueError` either
  way. A naive
  `row.reshape(n_states, n_pos)` raises `ValueError` for
  non-rectangular detectors. SliceModel uses **schema-aware
  state-bin slicing** instead:

  ```python
  # For each state s, the row's columns belonging to that state.
  state_slices = {
      s: np.where(detector.state_ind_ == s)[0]
      for s in range(detector.n_discrete_states_)
  }
  spatial_states = [s for s in state_slices if detector.bin_sizes_[s] > 1]
  ```

  Spatial states (slice length > 1) plot as curves over position;
  singleton states (slice length == 1) carry a scalar that doesn't
  belong on a position axis and are skipped from the top plot.

  - **Top likelihood curve — state-collapsed.** `log_likelihood` is
    stored in **log space over `state_bins`**
    ([base.py:3897](../../src/non_local_detector/models/base.py#L3897),
    [base.py:2401](../../src/non_local_detector/models/base.py#L2401)),
    so it does **not** receive the posterior reduction strategy
    (`MARGINAL` / `CONDITIONAL_NON_LOCAL`). Posterior reductions
    use probability semantics — sum, divide by mass — and would be
    incorrect on raw log values. Instead, the SliceModel uses a
    dedicated helper
    `collapse_log_likelihood_to_position(log_lik_row, detector)`
    that follows pure log-likelihood handling:

    ```python
    # 1. NaN-clean (non-interior bins are NaN-padded by _convert_results_to_xarray).
    log_lik_row = np.where(np.isfinite(log_lik_row), log_lik_row, -np.inf)
    # 2. Identify spatial states and stack their per-state slices into
    #    an (n_spatial_states, n_pos) array. Singletons are skipped.
    spatial_states = [s for s in range(detector.n_discrete_states_)
                      if detector.bin_sizes_[s] > 1]
    n_pos = detector.bin_sizes_[spatial_states[0]]
    log_per_state = np.stack(
        [log_lik_row[detector.state_ind_ == s] for s in spatial_states]
    )  # (n_spatial_states, n_pos)
    # 3. All-non-finite guard. If every spatial entry is -inf (e.g. the
    #    whole row was masked / NaN), `np.max(log_per_state)` is -inf
    #    and `log_per_state - np.max(...)` is `(-inf) - (-inf) = nan`
    #    by IEEE-754, not zero. Short-circuit to an all-zero curve so
    #    the heatmap row reads as "no likelihood at any position"
    #    (matches the post-exp semantics of finite -inf inputs).
    if not np.isfinite(log_per_state).any():
        return np.zeros(n_pos, dtype=log_per_state.dtype)
    # 4. Subtract the per-row max of the FINITE entries before exp
    #    (float32 overflow guard, statespacecheck pattern; pure
    #    rescaling — no probability claim). Using max-of-finite (not
    #    plain np.max) keeps any persistent -inf columns at -inf
    #    after subtraction, so they exponentiate to exactly 0.0.
    finite_max = log_per_state[np.isfinite(log_per_state)].max()
    log_per_state = log_per_state - finite_max
    lik_per_state = np.exp(log_per_state)
    # 5. Collapse across spatial states by sum (NOT a probability
    #    operation — this is just "joint likelihood across spatial
    #    states at position x", peak-normalized for display).
    lik_curve = lik_per_state.sum(axis=0)
    # 6. Peak-normalize for plotting.
    peak = lik_curve.max()
    return lik_curve / peak if peak > 0 else lik_curve
    ```

    No call to `conditional_non_local_posterior`. No "divide by total
    mass." The only normalization is peak-normalization for display,
    which is shape-preserving and doesn't make a probability claim.

    Per-state likelihood curves are **deferred to v3+**: with the v1
    default detectors all spatial states share one observation model
    ([_defaults.py:80](../../src/non_local_detector/models/_defaults.py#L80),
    [_defaults.py:182-186](../../src/non_local_detector/models/_defaults.py#L182))
    and the likelihood computation reuses identical likelihoods for
    repeated observation settings
    ([base.py:3893](../../src/non_local_detector/models/base.py#L3893)),
    so per-state curves would be visually identical and add visual
    noise. State-distinguishing dynamics live in the
    StateProbabilityPanel (transitions/dynamics) and the
    PosteriorHeatmapPanel (state-conditioned posterior), not in
    per-state emission curves. When fits use distinct
    observation_models per state — meaningful for v3+ — the
    likelihood curves diverge and per-state plotting becomes
    informative; that's when this bullet revisits.
  - **Top curve fallback when `log_likelihood` is missing.** Fall
    back to a single state-collapsed posterior curve from
    `acausal_posterior[t_idx]`. Because `acausal_posterior` is in
    the probability domain (not log space), this fallback path
    **does** use the posterior reduction strategy:
    `select_reduction(detector.state_names, detector.bin_sizes_)`
    → `collapse_posterior_to_position(post_row, detector,
    reduction)`, which applies `MARGINAL` (column-sum spatial
    states, no renormalization), `CONDITIONAL_ON_SPATIAL`
    (sum-and-divide-by-spatial-mass — the strategy for
    NoSpikeContFrag), or `CONDITIONAL_NON_LOCAL`
    (sum-and-divide-by-non-local-mass) as appropriate — i.e. the
    same probability-domain helper the predictive overlay uses
    below. Title-bar note: "Likelihood not available — showing
    collapsed posterior. Re-run
    `predict(return_outputs=['log_likelihood'])` for the population
    likelihood curve."
  - **Predictive / posterior overlay — state-collapsed.** When
    `predictive_posterior` is available, collapse its row via the
    same reduction and peak-normalize. Per-cell row overlays use
    the same collapsed curve so the predictive prior on every row
    has the consistent "what the heatmap shows" semantics. When
    `predictive_posterior` is missing, the overlay hides; per-cell
    rows fall back to the collapsed `acausal_posterior[t_idx]`
    curve.

  Two distinct collapse helpers live in
  `non_local_detector.analysis.posterior` and **ship in Phase 1a**
  (so they're available before Phase 3's LikelihoodHeatmapModel
  and this Phase 4 SliceModel both call them):

  - `collapse_log_likelihood_to_position(log_lik_row, detector)` —
    log-space input, max-subtracted exp, sum across spatial states,
    peak-normalize. Includes the all-non-finite short-circuit (step
    3 in the algorithm above). **No probability normalization.**
    Used by the LikelihoodHeatmapModel (Phase 3) and the SlicePanel
    top likelihood curve (this phase).
  - `collapse_posterior_to_position(post_row, detector, reduction,
    zero_mass_fill=np.nan)` —
    probability-domain input. Applies the chosen reduction.
    `zero_mass_fill` controls the zero-mass-row write for **both
    `CONDITIONAL_*` strategies** (they share `_conditional_row`);
    the `MARGINAL` path ignores it (it never divides):
    - `MARGINAL`: column-sum the spatial-state slices, no additional
      renormalization. The output row is full-position with
      non-interior bins NaN-padded (inherited from `post_row`'s
      `_create_masked_posterior` padding). Sums use `np.nansum` to
      skip those non-interior NaN columns:
      `np.nansum(out, axis=-1) == 1.0` for fully-spatial schemas
      (Decoder, ContFrag); for NL with this reduction the result is
      the genuine spatial-state marginal
      `np.nansum(out, axis=-1) == 1 - np.nansum(post[:, singleton_bins], axis=-1)`
      (= `1 - P(No-Spike)` for the Track 0 fixture's
      `local_position_std=1.0`, = `1 - P(Local) - P(No-Spike)` for
      the deferred-marker `local_position_std=None`). Typically not
      what NL users want — `select_reduction` defaults NL to
      `CONDITIONAL_NON_LOCAL`.
    - `CONDITIONAL_ON_SPATIAL`: sum the **spatial-state** slices
      (`bin_sizes_[s] > 1`) and divide by their total mass
      (renormalization is intrinsic to "conditional"). Default for
      `NoSpikeContFragSortedSpikesClassifier` (state_names
      `["No-Spike", "Continuous", "Fragmented"]` — `select_reduction`
      picks `CONDITIONAL_ON_SPATIAL` because no state name contains
      `"Non-Local"` but `bin_sizes_` has a singleton). Output sums
      to 1.0 on rows with positive spatial mass; `zero_mass_fill`
      on zero-mass rows. Routes through the same `_conditional_row`
      private helper as `CONDITIONAL_NON_LOCAL` with
      `selected_state_ids` = state ids where
      `detector.bin_sizes_[s] > 1`.
    - `CONDITIONAL_NON_LOCAL`: sum non-local-state slices and
      divide by their total mass (renormalization is intrinsic to
      "conditional"). Output sums to 1.0 on rows with positive
      non-local mass; `zero_mass_fill` on zero-mass rows. Routes
      through `_conditional_row` with `selected_state_ids` = state
      ids where `"Non-Local" in detector.state_names[s]`.

    Used for the top-curve fallback (when `log_likelihood` is
    missing), the LikelihoodHeatmap's posterior fallback, the
    predictive / smoothed overlay, and per-cell row overlays. The
    SlicePanel and PosteriorHeatmapModel pick the strategy via
    `select_reduction(detector.state_names, detector.bin_sizes_)`
    so all three reductions are exercised across v1's four
    sorted-spikes 1D detectors:
    - `MARGINAL` → `SortedSpikesDecoder`, `ContFragSortedSpikesClassifier`.
    - `CONDITIONAL_ON_SPATIAL` → `NoSpikeContFragSortedSpikesClassifier`.
    - `CONDITIONAL_NON_LOCAL` → `NonLocalSortedSpikesDetector`.

  Panels never reach into `state_ind_` or `bin_sizes_` directly;
  the helpers encapsulate the schema-aware spatial-state slicing.
  - **Per-cell rows** — list of `CellSlice` dataclasses for cells that
    spiked in this bin. `CellSlice` carries `cell_id`,
    `place_field_norm` (peak-normalized row of
    `analysis.place_fields.extract_per_cell_place_fields(detector)`),
    and optional
    `event_hpd_overlap` / `event_kl_divergence` / `event_spike_prob`
    populated from `RunBundle.events` when that sidecar is present.
    Per-cell rows render whether or not `log_likelihood` /
    `predictive_posterior` are in the results — they're driven by
    `spike_times` + place fields from the detector, both of which
    the `RunBundle` always carries.
- `panels/qt/slice.py`: `QtSlicePanel(QWidget)` with population plot +
  per-cell row pool (`MAX_PER_CELL_PLOTS = 6`) + pinned-row logic.
- Pinning: click a raster spike, the spike's cell stays in the slice
  panel until `Esc` or click-pinned-again.
- Per-tick path uses an in-RAM ring buffer.

**Verification**:

- (Track 0, loglik-bearing bundle) Scrub to `event_times[0]` in the
  simulated `nl_all` (or `nl_loglik`) bundle. Per-cell rows show
  the cells that the simulation injected at that event (ground
  truth from the simulation). The population **top curve** is
  asserted bit-identical to a freshly computed
  `collapse_log_likelihood_to_position(log_lik_row, detector)`
  applied to `results.log_likelihood.isel(time=event_idx).values`
  via `np.testing.assert_allclose(top, expected, atol=1e-14,
  equal_nan=True)` — i.e. **a likelihood curve summed across all
  spatial states** (`Local + Non-Local Continuous + Non-Local
  Fragmented` for the Track 0 NL fixture's `local_position_std=1.0`
  config), peak-normalized. The curve is **not** the conditional
  non-local posterior; the title-bar text reads "Likelihood across
  all spatial states; heatmap shows non-local states only" so the
  divergence from the heatmap above is visible.
- (Track 0, predictive overlay) On the same bundle, the slice
  panel's **overlay** curve (separate from the top curve) is
  asserted bit-identical to
  `collapse_posterior_to_position(predictive_row, detector,
  reduction=CONDITIONAL_NON_LOCAL)`. This output **does** carry
  NaN at non-interior positions (the conditional non-local
  algorithm writes NaN there per
  [static.py:169](../../src/non_local_detector/visualization/static.py#L169)),
  so the assertion uses `np.testing.assert_allclose(overlay,
  expected, atol=1e-14, equal_nan=True)` — bare `==` would fail on
  the matching NaNs. The overlay matches the heatmap's reduction
  because both are probability-domain curves and the conditional
  non-local reduction is the right semantics for them.
- (Track 0 fallback) Re-run with the default-`predict()`
  `nl_default` variant: `log_likelihood` and `predictive_posterior`
  are both absent. The top curve falls back to
  `collapse_posterior_to_position(post_row, detector,
  reduction=CONDITIONAL_NON_LOCAL)` from `acausal_posterior`
  (probability domain → reduction is valid here). Title-bar text
  switches to "Likelihood not available — showing collapsed
  posterior. Re-run `predict(return_outputs=['log_likelihood'])`
  for the population likelihood curve." Predictive overlay hides;
  per-cell rows render normally (driven by `spike_times` + place
  fields, not by the missing arrays).
- (Track A) If `NLD_REAL_DATA_BUNDLE_DIR` is set and contains a
  `contfrag/` subdirectory, repeat against
  `$NLD_REAL_DATA_BUNDLE_DIR/contfrag/` (loaded via
  `--run-from-dir contfrag:$NLD_REAL_DATA_BUNDLE_DIR/contfrag/`).
  Assert that scrubbing to a known confidently-non-local time bin
  populates the per-cell rows with reasonable cell IDs and place
  fields.

### Phase 5 — Plugin contract + downstream consumer

- Document `TimeAxisPanel` / `BinSyncedPanel` ABCs in the
  package README.
- In `continuum-swr-replay`, add `interactive_panels/view_models.py` +
  `interactive_panels/qt.py` for the panels that need project-specific
  data shapes. **With v1's `fill_below`, `thresholds`,
  `MultiLineSeriesPanel`, and `EventOverlay`, most continuum panels
  collapse into one-liners** built from generic builtins:
  - SWR consensus trace = `LineSeriesPanel(thresholds=[2.0])` plus
    `EventOverlay.intervals(...)` from `data["ripple_times"]`.
  - MUA rate = same pattern, using `data["hse_times"]`.
  - Theta LFP = plain `LineSeriesPanel`.
  - Theta power z-score = `LineSeriesPanel(fill_below=True,
    thresholds=[2.0])`.
  - Speed = `LineSeriesPanel(fill_below=True)`.
  - Head direction = `ScatterSeriesPanel(y_range=(-π, π))`.

  The continuum module's job in Phase 5 is then mostly to pre-build
  these from the `data` dict via a `panels_for_continuum_data(data)`
  helper, plus the `RunBundle.from_continuum_data(...)` adapter
  documented in Track B.
- **Document `plot_peri_event_probability` as out of scope.** It is
  event-aligned (x-axis = peri-event time, not session time), so it
  is not interactive in the same way and remains a static figure.
  Keep the matplotlib version in `continuum_swr_replay.visualization.plots`
  unchanged.
- Convenience wrapper:
  `continuum_swr_replay.visualization.launch_detector_viewer(data,
  bundle, ...)` — accepts the project's `data` dict and a `RunBundle`
  (or `dict[str, RunBundle]` for multi-run), constructs the viewer
  with the project panels pre-registered.
- Document the migration path away from `plot_detector`.

**Verification** (Track B): `continuum-swr-replay`'s integration test
loads a small NWB slice via `load_data(...)`, fits a
`SortedSpikesDecoder`, builds a `RunBundle` via
`RunBundle.from_continuum_data(...)`, and launches the viewer
headlessly via `pytest-qt qtbot.waitExposed`. Visual diff against the
static `plot_detector` figure for the same time slice.

### Phase 6 — Polish, model-swap UI + ship v1

- Auto-scroll (Space, speed combo).
- Smoothed-overlay support if results contain `acausal_posterior`.
- **Model swap UI**: "Model" dropdown in the controls bar (visible
  only when the data source has more than one run); M-key cycles.
  Re-binds all panels' view-models against the new active run while
  preserving the **ViewerCore state fields that aren't tied to the
  run identity**: `current_view_state` (the frozen `ViewState`
  snapshot — its `t_center`, `t_width`, `request_id`,
  `load_acausal`), `pinned_event_row`, and the active-overlay
  selection (which is a **name** — guaranteed to exist in the new
  run's overlay set by the data source's overlay-name-alignment
  validation). Only `active_run_name` changes on swap; the panels
  re-render with the new run's overlay *data* under the same active
  overlay name. Place-field
  ordering for the raster panel re-sorts on swap; the slice panel's
  per-cell rows refetch from the new run. (See "View state object"
  in the Decided section for the `ViewState` (frozen request
  snapshot) vs. ViewerCore-state distinction.)
- README with a screencast covering both single-run and model-swap
  flows.
- Update `non_local_detector/visualization/__init__.py` to lazy-expose
  `launch` from the `interactive` sub-package (gated on `[viewer]`).

**Verification**:

- (Track 0) Multi-run viewer launches with **all four** Track 0
  bundles:
  `{"nl": nl_bundle, "cf": cf_bundle, "nsf": nsf_bundle,
  "dec": dec_bundle}`. M-key cycles through them while the
  ViewerCore state fields unrelated to run identity are preserved:
  `current_view_state`'s `t_center` / `t_width` plus
  `pinned_event_row` and the active overlay selection.
  (`request_id` is not preserved — it increments per load — but
  the rest of the `ViewState` snapshot is rebuilt with the same
  `t_center`/`t_width` after swap.) Only `active_run_name`
  mutates. Each swap must:
  - Re-render the StateProbabilityPanel line set with the new
    schema (NL has 4 states, ContFrag has 2, NoSpikeContFrag has
    3, Decoder has 1).
  - Re-select the PosteriorHeatmapPanel reduction strategy via
    `select_reduction(state_names, bin_sizes_)`: NL →
    `CONDITIONAL_NON_LOCAL`, NSF → `CONDITIONAL_ON_SPATIAL`,
    ContFrag and Decoder → `MARGINAL`. Heatmap re-renders with the
    new strategy.
  - Re-bind the slice panel's `collapse_log_likelihood_to_position`
    helper to the new detector's `state_ind_` / `bin_sizes_` so
    spatial-state slicing follows the active schema.

  Cycling NL → ContFrag → NSF → Decoder → NL therefore exercises
  all three `CONDITIONAL_*` / `MARGINAL` reduction paths and both
  rectangular and non-rectangular state-bin layouts in one test.
- (Track 0 overlay alignment) Attach a named SWR-events overlay
  (same `EventOverlay.intervals(name="SWR", t_start=..., t_end=...)`
  object) to **all four bundles** (`nl`, `cf`, `nsf`, `dec`),
  plus a per-run model-derived overlay (each bundle constructs its
  own `EventOverlay.points(name="non_local_events", times=...)`
  from the run's results — for `nsf` and `dec`, this is whatever
  per-run "interesting moments" the user defines; for `nl` it's
  the conditional non-local high-confidence times; for `cf` it's
  the high-confidence Fragmented times). Launch the multi-run
  viewer; assert construction succeeds (overlay schemas match:
  every run exposes
  `{("SWR", "intervals"), ("non_local_events", "points")}`).
  Select the active overlay to `"non_local_events"`, hit M to
  swap NL →
  ContFrag, and assert: (a) the navigator's active overlay name is
  still `"non_local_events"`, (b) the panel-rendered overlay
  markers for `"non_local_events"` now show the ContFrag run's
  data (not NL's), (c) `"SWR"` overlay markers are unchanged
  across the swap because all four bundles attached the same
  object, (d) N / Shift+N semantics for the active overlay stay
  consistent across the swap (every run declares it as `points`,
  so the "next event ≥ t_center" rule applies regardless of which
  run is active). Cycle through all four bundles
  (`NL → ContFrag → NSF → Decoder`) and assert these properties
  hold across every swap pair.

  Negative cases (each must raise `ValueError` at viewer
  construction with the offending detail in the message):
  - **Schema mismatch by `kind`**: one bundle has
    `EventOverlay.points(name="events", times=...)`, another has
    `EventOverlay.intervals(name="events", t_start=..., t_end=...)`.
    Assert the message names both schemas (different `(name,
    kind)` tuples for the same name).
  - **Schema mismatch by missing overlay**: one bundle has
    `[swr]`, another has `[swr, theta]`. Assert the message names
    the divergent schemas.
  - **Within-bundle duplicate name**: one bundle has two overlays
    both named `"events"` (one points, one intervals). Assert the
    message names the duplicate name and the offending bundle.
- (Track A, optional) If `NLD_REAL_DATA_BUNDLE_DIR` is set and
  contains both `continuous/` and `contfrag/` subdirectories, the
  multi-run viewer launches via
  `--run-from-dir continuous:$NLD_REAL_DATA_BUNDLE_DIR/continuous/
  --run-from-dir contfrag:$NLD_REAL_DATA_BUNDLE_DIR/contfrag/` and
  M-key swaps work the same way (view state preserved across
  swap; `PosteriorHeatmapModel` re-selects reduction strategy on
  each swap per Risk 7).
- (Demo notebook) `notebooks/interactive_viewer_demo.ipynb` runs
  end-to-end on a kernel with `[viewer]` installed.

### v2 (separate plan when v1 ships)

- `panels/panel_/` Panel/holoviews backend consuming the same
  view-models.
- `launch_in_browser(...)` notebook helper that serves the Panel viewer.
- `ZarrDecoderDataSource`.
- `MetricPanel` (HPD overlap, KL divergence, spike prob).

### v3 (separate plan)

- 2D-decoder support: `PositionGrid` 2D path, "movie" panels.
- `VideoOverlayPanel` (`BinSyncedPanel` that loads a video file and
  alpha-blends the 2D posterior on the current frame).
- Clusterless support — see "Deferred-model pathway" below for the
  per-class extension sketch.
- Multi-environment support — see "Deferred-model pathway".

## Deferred-model pathway (v3+ design sketch)

`non_local_detector` ships ten detector classes; v1 covers the four
sorted-spikes 1D single-environment variants. The other six are
deferred to v3+, but each has a clear extension point in v1's
architecture rather than a dead end. This section sketches what
v3+ work each deferred class needs, so the deferral can be
re-opened with a concrete starting point rather than re-litigating
v1 design decisions.

### Clusterless detectors (5 classes)

`ClusterlessDecoder`, `ContFragClusterlessClassifier`,
`NoSpikeContFragClusterlessClassifier`, `NonLocalClusterlessDetector`,
`MultiEnvironmentClusterlessClassifier` (the last also requires the
multi-environment extension below).

The state-space + reduction story is **identical** to the
sorted-spikes side: every clusterless class is a parallel of one of
v1's four sorted-spikes detectors, with the same `state_names`,
`bin_sizes_`, and `state_ind_` schema. So `PosteriorHeatmapModel`,
`StateProbabilityModel`, `LikelihoodHeatmapModel`, all collapse
helpers, and `select_reduction(...)` work unchanged.

What needs to be redesigned is the **per-cell / per-mark
visualization paradigm**:

- `extract_per_cell_place_fields(detector)` — sorted-spikes only.
  Clusterless `encoding_model_` entries store *mark distributions*
  (KDE over `(position, mark)`), not per-cell place fields.
  v3+ adds a parallel
  `extract_per_electrode_group_mark_field(detector, electrode_group)`
  helper. The sorted-spikes helper continues to raise on
  clusterless detectors with a clear pointer.
- `RasterPanel` — assumes sorted-spikes per-cell `spike_times`.
  Clusterless data has per-electrode-group spike events with mark
  features attached. v3+ adds a `MarkRasterPanel` (or extends
  `RasterPanel` with a `kind: Literal["sorted", "clusterless"]`
  parameter) that plots events colored by their mark vector
  projection (e.g. PC1 of the mark space, or an electrode-group
  identifier).
- `SlicePanel` per-cell rows — assumes one row per cell. For
  clusterless, "per-cell" becomes "per-electrode-group" or "per-mark
  cluster": v3+ ships per-electrode-group rows that plot the
  electrode-group's marginalized place field (over marks) plus
  spike events from that group in the cursor bin. The predictive /
  posterior overlay is unchanged.
- `RunBundle.spike_times` — currently `list[np.ndarray]`. v3+ adds
  an optional `spike_marks: list[np.ndarray]` field
  (`(n_events, n_features)` per electrode group) so the bundle
  carries the mark vectors alongside the event times. Single
  `RunBundle` dataclass; the new field is `None` for sorted-spikes
  callers.

Estimated v3 effort: one design sketch + one PR per class kind
(sorted/clusterless), with the heavy lifting in MarkRasterPanel and
the per-electrode-group SlicePanel rows. The reduction-strategy and
heatmap machinery is unchanged.

### Multi-environment detectors (1 class)

`MultiEnvironmentSortedSpikesClassifier` (the clusterless variant
also lands here, after the clusterless work above).

`detector.encoding_model_` has multiple keys, one per
`(environment_name, encoding_group)`. v1's
`extract_per_cell_place_fields` raises with a clear error pointing
at this section.

What needs to be redesigned:

- **Active-environment selector UI**: a new dropdown in the
  controls bar (next to the Model dropdown) that picks the active
  `environment_name`. View-models that consume place fields
  (SliceModel, RasterModel) re-bind to the active environment's
  `encoding_model_` entry on selector change. The
  PosteriorHeatmapModel either:
  - Filters to the active environment's state-bin slices (default —
    matches the single-env semantics of v1), or
  - Renders all environments in a faceted layout (horizontal small
    multiples, one per environment), under a "compare environments"
    toggle.
- **`extract_per_cell_place_fields` extension**: take an
  `environment_name` argument; the v1 single-env helper becomes a
  thin wrapper that defaults to the unique environment when the
  detector has only one. Multi-env callers pass the active
  environment name explicitly.
- **Position grid per environment**: each environment has its own
  `place_bin_centers_`. The viewer's position-axis becomes
  per-environment, swapping when the active environment changes.
  `PositionGrid` (already abstract in v1 for the 1D-vs-2D path)
  gains an `environment_name` association.
- **Multi-run + multi-env interaction**: v1's multi-run mode
  validates time-grid and overlay-name alignment across runs.
  v3+ multi-env adds per-environment validation (every loaded
  bundle must share the active environment's identity, or the swap
  re-binds environments).

Estimated v3 effort: significant UI work (active-environment
dropdown, faceted layout option), modest helper work (the
`extract_per_cell_place_fields` extension is small). The reduction-
strategy and time-axis machinery is unchanged.

### Coverage table after v3

| Class                                    | v1 | v3                             |
| ---------------------------------------- | -- | ------------------------------ |
| `SortedSpikesDecoder`                    | ✓  | unchanged                      |
| `ContFragSortedSpikesClassifier`         | ✓  | unchanged                      |
| `NoSpikeContFragSortedSpikesClassifier`  | ✓  | unchanged                      |
| `NonLocalSortedSpikesDetector`           | ✓  | unchanged                      |
| `ClusterlessDecoder`                     | —  | covered (clusterless paradigm) |
| `ContFragClusterlessClassifier`          | —  | covered                        |
| `NoSpikeContFragClusterlessClassifier`   | —  | covered                        |
| `NonLocalClusterlessDetector`            | —  | covered                        |
| `MultiEnvironmentSortedSpikesClassifier` | —  | covered (multi-env UI)         |
| `MultiEnvironmentClusterlessClassifier`  | —  | covered (both extensions)      |

Every detector class in `non_local_detector` has a documented
visualization pathway. v1 ships 4/10; v3+ extends the same
architecture (view-models, panel ABCs, reduction helpers) to
the remaining 6.

## Test data strategy

Per CLAUDE.md's "test on simulated, then real" workflow. Two data
tracks; simulated drives every phase, real data validates phases 4–6.

### Track 0 — Simulated data (every phase)

The canonical entry point for sorted-spikes simulation is
`make_simulated_data(...)` in
[sorted_spikes_simulation.py:507](../../src/non_local_detector/simulate/sorted_spikes_simulation.py#L507).
It returns `(speed, position, spike_times, time, event_times,
sampling_frequency, is_event)` from a deterministic seed. The same
helper is used in
[notebooks/01_models_and_validation/sorted_spikes_detector_test.py](../../notebooks/01_models_and_validation/sorted_spikes_detector_test.py)
to drive the existing `plot_non_local_model` static plot — that script
is the canonical pattern; mirror it.

Phase 1b adds a shared pytest fixture in
`src/non_local_detector/tests/interactive/conftest.py` that follows the
existing notebook script:

```python
from non_local_detector import (
    ContFragSortedSpikesClassifier,
    NonLocalSortedSpikesDetector,
    NoSpikeContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

speed, position, spike_times, time, event_times, sampling_frequency, is_event = (
    make_simulated_data(n_neurons=25, seed=0)
)

# NonLocal detector: matches the notebook's params exactly.
# With local_position_std=1.0, Local is a spatial state and only No-Spike
# is a singleton: bin_sizes_ = [n_pos, 1, n_pos, n_pos]. Non-rectangular.
# (The fully-singleton variant local_position_std=None gives
# [1, 1, n_pos, n_pos] but is much slower to fit; see deferred-marker
# fixture below.)
nl_detector = NonLocalSortedSpikesDetector(
    sorted_spikes_algorithm="sorted_spikes_kde",
    non_local_position_penalty=1.0,
    non_local_penalty_std=5.0,
    local_position_std=1.0,
)
# IMPORTANT: each detector is FIT EXACTLY ONCE via estimate_parameters(),
# but its returned dataset is DISCARDED. All three result variants per
# detector are then built from det.predict(...), which consumes the
# fully-fit parameters without further mutation.
#
# Why discard estimate_parameters' return value:
#   - estimate_parameters runs EM iterations (base.py:1861).
#   - Each iteration is E-step then M-step (base.py:1881).
#   - The final iteration's M-step mutates the detector AFTER the
#     E-step that produced the returned posteriors; the dataset returned
#     at base.py:2026 is computed from parameters BEFORE the final
#     M-step's mutations land.
#   - So the dataset and the post-fit detector object disagree by one
#     M-step's worth of update. Tiny in absolute terms, but it would
#     mean the bundle's `results` were generated under parameters the
#     bundle's `detector` no longer holds — exactly the
#     consistency property the test fixture is supposed to guarantee.
#
# Calling predict(...) after estimate_parameters() avoids this: predict
# does not refit and uses the post-final-M-step parameters that the
# detector currently holds.

# Fit once (discard the EM-internal return value), then predict three
# times for the three return_outputs shapes.
nl_detector.estimate_parameters(
    position_time=time, position=position, spike_times=spike_times,
    is_training=~is_event, time=time,
    # return_outputs choice here does not matter — we discard the result.
)
nl_results_default = nl_detector.predict(
    position_time=time, position=position, spike_times=spike_times,
    time=time,
    # no return_outputs → only acausal_posterior + acausal_state_probabilities
)
nl_results_loglik = nl_detector.predict(
    position_time=time, position=position, spike_times=spike_times,
    time=time, return_outputs=["log_likelihood"],
)
nl_results_all = nl_detector.predict(
    position_time=time, position=position, spike_times=spike_times,
    time=time, return_outputs="all",  # → filter, predictive,
                                      #   predictive_posterior, log_likelihood
                                      #   (per OUTPUT_INCLUDES["all"], base.py:80)
)

# ContFrag classifier: rectangular, two spatial states sharing one obs model.
# bin_sizes_ = [n_pos, n_pos], state_names = ["Continuous", "Fragmented"].
# Matches v1's "select_reduction → MARGINAL" path.
cf_detector = ContFragSortedSpikesClassifier(
    sorted_spikes_algorithm="sorted_spikes_kde",
)
cf_detector.estimate_parameters(  # fit once, discard return
    position_time=time, position=position, spike_times=spike_times,
    is_training=~is_event, time=time,
)
# (cf_results_default / cf_results_loglik / cf_results_all built via
#  cf_detector.predict(...) — same predict-only pattern as NL above;
#  omitted here for brevity.)

# Plain decoder: same data, different state schema (Continuous only).
# bin_sizes_ = [n_pos], single state. select_reduction → MARGINAL.
dec_detector = SortedSpikesDecoder(sorted_spikes_algorithm="sorted_spikes_kde")
dec_detector.estimate_parameters(  # fit once, discard return
    position_time=time, position=position, spike_times=spike_times,
    is_training=~is_event, time=time,  # match NL/CF: train on non-event bins
)
# (dec_results_* via dec_detector.predict(...))

# NoSpikeContFrag classifier: 3 states (No-Spike + Continuous + Fragmented).
# bin_sizes_ = [1, n_pos, n_pos] (No-Spike singleton, two spatial states).
# state_names = ["No-Spike", "Continuous", "Fragmented"]. None contain
# "Non-Local", so select_reduction picks CONDITIONAL_ON_SPATIAL — sums
# the two spatial states and divides by their total mass to give a
# curve that integrates to 1.0 over position.
nsf_detector = NoSpikeContFragSortedSpikesClassifier(
    sorted_spikes_algorithm="sorted_spikes_kde",
)
nsf_detector.estimate_parameters(  # fit once, discard return
    position_time=time, position=position, spike_times=spike_times,
    is_training=~is_event, time=time,
)
# (nsf_results_* via nsf_detector.predict(...))

# The full 4 × 3 matrix (4 detectors × 3 return_outputs variants
# = 12 RunBundles total) is built in the fixture via:
#
#     # Each detector is fit exactly once via estimate_parameters() and the
#     # returned dataset is discarded. All three variants per detector come
#     # from predict() against the post-final-M-step detector state, so each
#     # bundle's `results` was produced under exactly the parameters its
#     # bundle's `detector` currently holds.
#
#     PREDICT_VARIANTS = {
#         "default": None,                # → only acausal_*
#         "loglik":  ["log_likelihood"],  # → + log_likelihood
#         "all":     "all",               # → + filter, predictive,
#                                         #     predictive_posterior, log_likelihood
#                                         #   (per OUTPUT_INCLUDES["all"], base.py:80)
#     }
#     DETECTOR_FIT_KWARGS = {
#         "nl":  dict(detector=nl_detector,  is_training=~is_event),
#         "cf":  dict(detector=cf_detector,  is_training=~is_event),
#         "nsf": dict(detector=nsf_detector, is_training=~is_event),
#         "dec": dict(detector=dec_detector, is_training=~is_event),
#     }
#     bundles = {}
#     for det_key, fit_kwargs in DETECTOR_FIT_KWARGS.items():
#         kwargs = dict(fit_kwargs)
#         det = kwargs.pop("detector")
#         # Fit ONCE — discard the EM-internal return; we only want the
#         # mutation it leaves on `det`.
#         _ = det.estimate_parameters(
#             position_time=time, position=position, spike_times=spike_times,
#             time=time, **kwargs,
#         )
#         # All three variants come from predict() against the now-fit detector.
#         for var_key, return_outputs in PREDICT_VARIANTS.items():
#             results = det.predict(
#                 position_time=time, position=position, spike_times=spike_times,
#                 time=time, return_outputs=return_outputs,
#             )
#             bundles[f"{det_key}_{var_key}"] = RunBundle(
#                 results=results, detector=det,
#                 spike_times=spike_times, position_time=time,
#                 position=position, speed=speed,
#             )
#     # bundles has 12 entries: nl_default/nl_loglik/nl_all,
#     # cf_default/cf_loglik/cf_all, nsf_default/nsf_loglik/nsf_all,
#     # dec_default/dec_loglik/dec_all.
#     # All three "nl_*" bundles share one fitted nl_detector; same for
#     # cf, nsf, dec.
#     # All three results per detector were computed under the SAME
#     # post-final-M-step parameters the detector currently holds.
#
# Note: the viewer only consumes acausal_posterior, acausal_state_probabilities,
# log_likelihood, and predictive_posterior (see "Viewer-ready results
# contract" above). The "all" variant additionally produces causal_posterior,
# causal_state_probabilities, and predictive_state_probabilities, which the
# viewer ignores; using "all" keeps the fixture aligned with the smoke-test
# prose without affecting viewer behavior.
```

`return_outputs` is the canonical knob for what lands in the returned
xarray. `store_log_likelihood=True` only stores the array on the
detector instance (`self.log_likelihood_`,
[base.py:2020](../../src/non_local_detector/models/base.py#L2020)) and
is deprecated in favor of `return_outputs="log_likelihood"`
([base.py:1820](../../src/non_local_detector/models/base.py#L1820));
do **not** rely on it for the fixture. The dataset variables are
gated on `requested_outputs`
([base.py:2031](../../src/non_local_detector/models/base.py#L2031)),
which is normalized from `return_outputs`.

The fixture exposes:

- A `RunBundle` per detector for default (no `return_outputs`),
  `return_outputs=["log_likelihood"]`, and `return_outputs="all"`
  variants — three RunBundle shapes per detector, **twelve total**
  across `NonLocalSortedSpikesDetector` (`nl_*`),
  `ContFragSortedSpikesClassifier` (`cf_*`),
  `NoSpikeContFragSortedSpikesClassifier` (`nsf_*`), and
  `SortedSpikesDecoder` (`dec_*`). Same results-shape gating
  applies to `predict(...)` for users who fit and predict
  separately; both code paths take the same `return_outputs` kwarg.
- All four detectors share the same time grid, spike_times, and
  position, so any subset can be combined into a multi-run dict
  for model-swap testing.
- `event_times` from the simulation as **ground truth** for
  non-local events: tests can scrub the viewer to a known event
  window and assert which cells appear in the SlicePanel per-cell
  rows.
- A multi-run dict
  (`{"nl": nl_bundle, "cf": cf_bundle, "nsf": nsf_bundle, "dec":
  dec_bundle}`) for the model-swap path — same time grid, four
  distinct schemas spanning v1's full detector matrix:
  - `nl`: 4 states with singletons → `CONDITIONAL_NON_LOCAL`.
  - `nsf`: 3 states with one singleton → `CONDITIONAL_ON_SPATIAL`
    (the third reduction strategy added in this plan).
  - `cf`: 2 fully-spatial states → `MARGINAL`.
  - `dec`: 1 fully-spatial state → `MARGINAL`.

  Cycling through all four exercises every reduction path and both
  rectangular and non-rectangular state-bin layouts.

Why simulated first: deterministic seeds, fast (the existing notebook
script runs end-to-end in seconds), ground-truth `event_times` for the
SlicePanel verification, CI-friendly. Use this fixture for every
phase's verification.

**Implication for the SlicePanel top likelihood curve.** With
`local_position_std=1.0` the NL fixture has three spatial states
(`Local`, `Non-Local Continuous`, `Non-Local Fragmented`). The
`collapse_log_likelihood_to_position` helper sums likelihoods across
**all** spatial states (the helper is reduction-strategy-free per the
log-space contract above), so the top curve includes Local's
contribution. This is intentional — the curve shows "the joint
likelihood of any spatial state at position x", which differs from
the heatmap's `CONDITIONAL_NON_LOCAL` reduction (which excludes Local
because it's a "decode = animal position" state, not a non-local
state). Document this divergence in the panel's title bar
("Likelihood across all spatial states; heatmap shows non-local
states only") so users don't misread.

**Deferred singleton-Local fixture (covered later, not in default CI).**
The fully-singleton variant (`local_position_std=None`,
`bin_sizes_ = [1, 1, n_pos, n_pos]`) takes substantially longer to
fit because the EM updates for the discrete `Local` state are slower
than the continuous-Gaussian variant. Add a parametrized version of
the fixture marked `@pytest.mark.slow` that constructs an NL detector
with `local_position_std=None` and asserts the SlicePanel handles
the all-singleton-Local case (top curve sums only the two NL states;
Local's likelihood is a scalar that gets dropped from the position
plot). This test is gated on a `--run-slow` pytest flag so default
CI stays fast; nightly / pre-release runs include it. Track this as
a Phase 1c follow-up (no later than v1 release) since the singleton
path is the original NL design and v3+ multi-environment work likely
revisits it.

### Track A — statespacecheck real data (Phases 4 + 6)

Statespacecheck's intermediates
(`data/intermediates/cont_results.nc`, `cont_frag_results.nc`,
`cont_model.pkl`, `cont_frag_model.pkl`) are the canonical real-data
benchmark for the upstream paper figures
([cache.py:66/71](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L66)).
The upstream cache reconstructs `spike_times` and `position` from the
same upstream pipeline.

For our viewer we need a `RunBundle` per fitted model. Spikes /
position must come from the **same source statespacecheck uses**, not
from the model pickle's internals. Two paths, in order of preference:

**Path 1 (preferred): consume statespacecheck's existing cache
sidecars for spikes/position/place_fields, but always source `results`
from the original NetCDF.** statespacecheck's
`figure04_<model>.zarr` schema makes `acausal_posterior` **optional**
([data_source.py:124](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/data_source.py#L124),
[cache.py:226](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L226)
— the writer only includes it when the source dataset has it). But
RunBundle requires `acausal_posterior` and `acausal_state_probabilities`
("always present" in the results contract), so the Zarr is unsafe as
the default `results` source. Use it only as a fallback after
explicit validation that both variables are present.

Devtool:

```text
python -m non_local_detector.visualization.interactive.devtools \
    bundle-from-statespacecheck-cache \
    --cache-dir <statespacecheck cache_dir>/ \
    --intermediates-dir <path/to/intermediates>/ \
    --model continuous \
    --out <bundles_root>/continuous/
```

The `--model` argument takes upstream's literal model keys
**`continuous`** or **`contfrag`** (not `cont` or `cont_frag`); the
devtool internally calls statespacecheck's
[`model_paths(intermediates_dir, model)`](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L62)
to resolve the on-disk filenames, which differ from the model key:
`continuous → cont_results.nc + cont_model.pkl`,
`contfrag → cont_frag_results.nc + cont_frag_model.pkl`. Mirroring
upstream exactly avoids the user having to remember the underscore
quirk.

The statespacecheck cache directory contains per-model files
(`figure04_<model>.zarr`, `figure04_<model>_events.parquet`,
`figure04_<model>_place_fields.npz` —
[cache.py:86](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L86))
**plus** model-independent shared sidecars (`figure04_meta.npz` and
`figure04_spike_times.npy` —
[cache.py:93-105](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L93))
that hold the session's time grid, linear position, and per-cell
spike times (model-independent because they describe the recording,
not the decoder). Fitted-model pickles and source NetCDF results
live in `intermediates_dir`
([cache.py:308](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L308)).
Both directories are required because RunBundle needs the detector
and results from `intermediates_dir`, plus the spikes / position /
place_fields from `cache_dir`. The devtool:

- Resolves intermediates paths via
  `cache_mod.model_paths(intermediates_dir, model)`.
- Loads `results` from `model_paths.results_nc`; validates that
  `acausal_posterior` and `acausal_state_probabilities` are present.
- Loads the fitted detector via
  `joblib.load(model_paths.model_pkl)`.
- Reads the shared `<cache-dir>/figure04_meta.npz` for time + linear
  position.
- Reads the shared `<cache-dir>/figure04_spike_times.npy` for
  per-cell spike times.
- Cross-checks the per-model
  `<cache-dir>/figure04_<model>_place_fields.npz` against
  `analysis.place_fields.extract_state_aligned_place_fields(detector)[:, detector.is_track_interior_state_bins_]`
  (interior-masked, since the cache stores interior-only place
  fields per [data_source.py:144](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/data_source.py#L144))
  to catch cache/intermediates version mismatches.
- Writes a **CLI-compatible bundle directory** at the `--out` path
  containing exactly the four files the viewer's `--run` flag
  consumes:

  ```text
  <out_dir>/
  ├── results.nc       # xarray Dataset (validated above)
  ├── model.pkl        # joblib-pickled fitted detector
  ├── spikes.npz       # np.savez with object-dtype 'spike_times' array
  └── position.parquet # time-indexed DataFrame with 'position' column
  ```

  No new bundle file format is introduced — the devtool just emits
  the four files the CLI already knows how to read. Optional sidecars
  (`events.parquet`, `extra_metrics.json`, `event_overlays.json`) may
  be added later by the same convention; absent ones simply mean those
  `RunBundle` fields stay at their `default_factory` values when the
  viewer constructs the bundle.

  The viewer accepts these in two ways:

  - Explicit:
    `--run continuous:<out>/results.nc:<out>/model.pkl:<out>/spikes.npz:<out>/position.parquet`
  - Convenience (recommended):
    `--run-from-dir continuous:<out_dir>/` — expands to the four
    expected filenames inside the directory and is the form
    Phase 6's tests use.

Optional flags:

- `--results-nc <path>` overrides the auto-resolved
  `model_paths.results_nc` for unusual layouts.
- `--model-pkl <path>` overrides the auto-resolved
  `model_paths.model_pkl`.
- `--results-from-zarr` substitutes the per-model Zarr in the cache
  directory for the source NetCDF when the latter is unavailable.
  The devtool then explicitly validates that `acausal_posterior` and
  `acausal_state_probabilities` are present in the Zarr (the Zarr's
  `acausal_posterior` is optional —
  [data_source.py:124](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/data_source.py#L124))
  and refuses to run otherwise.

No reach into model-pickle private attributes; results, detector, and
cache sidecars all come from explicitly-named, validated sources.

**Path 2 (fallback): build from raw data, mirror statespacecheck's
build pipeline.** When the statespacecheck cache isn't built yet, the
devtool replicates the upstream loader pattern at
[cache.py:367–405](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L367):

1. Load raw spike-times and position from the source recording files
   via the same upstream loader statespacecheck uses
   (`load_neural_recording_from_files(raw_data_dir, animal_date_epoch)`).
2. Load the fitted detector via `joblib.load(model_paths.model_pkl)`
   (resolved by the same upstream `model_paths(intermediates_dir,
   model)` used in Path 1).
3. Re-linearize raw position with `detector.environments[0]` (so
   `RunBundle.position` aligns with the decoder's bin grid).
4. Extract state-aligned place fields via
   `analysis.place_fields.extract_state_aligned_place_fields(detector)`
   and apply the interior mask
   `detector.is_track_interior_state_bins_` to get
   `(n_cells, n_states * n_interior)` — **interior-only**, since
   that's the axis statespacecheck validates against and stores in
   the cache ([cache.py:373–435](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/cache.py#L373),
   [data_source.py:144](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/data_source.py#L144)).
   Validate:

   ```python
   place_fields_interior = state_aligned[:, detector.is_track_interior_state_bins_]
   predictive_interior = results["predictive_posterior"].dropna(dim="state_bins")
   assert place_fields_interior.shape[0] == len(spike_times)
   assert place_fields_interior.shape[1] == predictive_interior.shape[-1]
   ```

   This mirrors statespacecheck's check (cell count, then interior
   bin count after dropna across the `state_bins` dimension), not a
   full-axis (with-NaN) comparison.
5. Construct the `RunBundle` and save.

Devtool flag: `--from-raw <raw_data_dir>` selects this path; default
is `--from-cache`.

Build both bundles into a parent directory and pin that directory
behind a single env var **`NLD_REAL_DATA_BUNDLE_DIR`**:

```text
$NLD_REAL_DATA_BUNDLE_DIR/
├── continuous/
│   ├── results.nc
│   ├── model.pkl
│   ├── spikes.npz
│   └── position.parquet
└── contfrag/
    ├── results.nc
    ├── model.pkl
    ├── spikes.npz
    └── position.parquet
```

Tests that need real data resolve subdirectories by name
(`continuous`, `contfrag`) and skip cleanly when
`NLD_REAL_DATA_BUNDLE_DIR` is unset or the requested subdirectory is
missing. Phase 4 uses `$NLD_REAL_DATA_BUNDLE_DIR/contfrag/` for the
SlicePanel against a real ContFrag session; Phase 6 uses both
subdirectories for the cont ↔ contfrag model-swap test.

A single env var pointing at the parent dir scales naturally to
additional bundles (re-fits with different priors, other sessions,
etc.) — each goes in its own named subdirectory.

### Track B — continuum-formatted sessions (Phase 5)

Day-to-day, data flows through
`continuum_swr_replay.data_loaders.load_data(...)`, which returns a
dict whose `position_info` is a time-indexed DataFrame (columns
`linear_position`, `head_speed`, `head_orientation`, …) and whose
`spike_times` is `dict[str, list[np.ndarray]]` keyed by brain area.

Phase 5 adds:

- `RunBundle.from_continuum_data(data, results, detector,
  brain_area="HPC")` helper, **shipped in `continuum-swr-replay`'s
  `interactive_panels` module** (not in `non_local_detector` — it
  depends on continuum's `data` shape). Translates `data["position_info"]`
  into `position_time` / `position` / `speed`, takes
  `data["spike_times"][brain_area]` for `spike_times`, and wraps
  `(results, detector)` into a `RunBundle`.
- A continuum-side integration test that loads a small NWB slice,
  fits a quick `SortedSpikesDecoder`, builds a RunBundle, and
  launches the viewer headlessly (`pytest-qt`'s `qtbot.waitExposed`).
  Validates the wiring and the continuum→`RunBundle` adapter, not
  pixel content.

### What NOT to commit

- No real-data files in this repo. Reference statespacecheck's
  intermediates by path / env var.
- No large simulated fixtures cached on disk. The fixture regenerates
  each test run from seeds (fast — total under 5 s for the 30 s
  trajectory + 20-cell fit).

### Edge cases (CLAUDE.md item 4)

Probed in Phases 1, 4, and 6:

- Default-`predict()` results (no `log_likelihood`) → SlicePanel top
  curve falls back to posterior with a title-bar note.
- Empty time bins (zero non-local mass) → masked / lightgrey.
- Multi-run construction with mismatched time grids → clear error
  at construction.
- `NonLocalSortedSpikesDetector` ↔ `SortedSpikesDecoder` swap
  (different state_names) → posterior reduction strategy adapts;
  StateProbabilityPanel re-renders its line set.
- Bin with > `MAX_PER_CELL_PLOTS` (= 6) firing cells → "(+K more)"
  truncation indicator.

## Mathematical / numerical considerations

### Posterior reductions per detector schema

The PosteriorHeatmapModel cannot just "sum over Non-Local states" —
v1's supported detectors have different state vocabularies and only
one of them has a `Non-Local` concept:

| Detector class                   | Default `state_names`                | Reduction                                          |
|----------------------------------|--------------------------------------|----------------------------------------------------|
| `SortedSpikesDecoder`            | `["Continuous"]`                     | Marginal: sum over the single state                |
| `ContFragSortedSpikesClassifier` | `["Continuous", "Fragmented"]`       | Marginal: sum over both states                     |
| `NonLocalSortedSpikesDetector`   | 4 states incl. two `Non-Local …`     | Conditional non-local: sum non-local states / mass |

State-name defaults sourced from [_defaults.py:63](../../src/non_local_detector/models/_defaults.py#L63),
[_defaults.py:82](../../src/non_local_detector/models/_defaults.py#L82),
and [_defaults.py:189](../../src/non_local_detector/models/_defaults.py#L189).
Both marginal reductions produce the smoothed posterior over position
`p(x_t | y_{1:T})`.

The view-model exposes a `reduction` strategy parameter that auto-detects
based on `detector.state_names`:

```python
class PosteriorReduction(Enum):
    # Sum the spatial-state slices column-wise; no additional
    # renormalization. Output already sums to 1.0 over interior bins
    # for fully-spatial schemas (Decoder, ContFrag) because
    # acausal_posterior sums to 1.0 over the INTERIOR PORTION of the
    # full state_bins axis (non-interior columns are NaN-padded by
    # _create_masked_posterior at base.py:2241; row sums must use
    # np.nansum to skip them). For schemas with singleton states
    # (NL, NoSpikeContFrag), the row sum is 1 - mass-of-singletons,
    # not 1.0 — typically not what users want; CONDITIONAL_*
    # strategies are usually preferred there.
    MARGINAL = "marginal"
    # Sum the spatial-state slices (bin_sizes_[s] > 1) and divide
    # by their total mass. Output sums to 1 on rows with positive
    # spatial mass; zero_mass_fill on zero-mass rows. The default
    # strategy for non-NL schemas that have singleton states (e.g.
    # NoSpikeContFragSortedSpikesClassifier with state_names
    # ["No-Spike", "Continuous", "Fragmented"]) — produces a
    # position curve that integrates to 1 by conditioning on the
    # event "decoder is in any spatial state" (i.e. not in the
    # No-Spike singleton).
    CONDITIONAL_ON_SPATIAL = "conditional_on_spatial"
    # Sum the slices for states whose name contains "Non-Local",
    # then divide by their total mass. Statespacecheck-style
    # paper view that excludes Local even when Local is spatial
    # (i.e. local_position_std is set). Use this on NL detectors
    # for "conditional non-local" replay analysis. Differs from
    # CONDITIONAL_ON_SPATIAL on NL with local_position_std=1.0:
    # this excludes Local; CONDITIONAL_ON_SPATIAL would include it.
    CONDITIONAL_NON_LOCAL = "conditional_non_local"  # statespacecheck pattern

def select_reduction(
    state_names: list[str], bin_sizes_: np.ndarray
) -> PosteriorReduction:
    """Pick a sensible default reduction for the given detector schema.

    Auto-detect rules, in priority order:
      1. Any state name contains "Non-Local" → CONDITIONAL_NON_LOCAL
         (the canonical NL replay-analysis view).
      2. Schema has singleton states but no "Non-Local" name →
         CONDITIONAL_ON_SPATIAL (e.g. NoSpikeContFrag — produce a
         curve that integrates to 1 by conditioning on spatial
         states).
      3. Fully-spatial schema → MARGINAL (Decoder, ContFrag — the
         spatial mass already sums to 1 so MARGINAL and
         CONDITIONAL_ON_SPATIAL coincide; MARGINAL is cheaper).
    """
    if any("Non-Local" in s for s in state_names):
        return PosteriorReduction.CONDITIONAL_NON_LOCAL
    if (np.asarray(bin_sizes_) == 1).any():
        return PosteriorReduction.CONDITIONAL_ON_SPATIAL
    return PosteriorReduction.MARGINAL
```

Users can override at construction
(`PosteriorHeatmapModel(..., reduction=PosteriorReduction.MARGINAL)`)
when the auto-detect picks the wrong thing for a custom detector.

The `CONDITIONAL_NON_LOCAL` strategy reuses the algorithm in
`plot_conditional_non_local_posterior`. **Phase 1a deliverable**:
refactor that algorithm into a stand-alone function in
`non_local_detector.analysis.posterior` named
`conditional_non_local_posterior(results, detector)` (dataset-level,
operates on the full session), and factor a generic private
`_conditional_row(post_row, detector, selected_state_ids,
zero_mass_fill)` out of it for row-level callers. `selected_state_ids`
is a list of **discrete state ids** (indices into
`detector.state_names`); the helper expands those to a column mask
internally via `np.isin(detector.state_ind_, selected_state_ids)`.
The dataset-level helper computes the non-local state ids once and
iterates over time calling the private helper per row; both
`CONDITIONAL_*` public strategies route through this same private
helper with strategy-appropriate `selected_state_ids` (see "Phase 1a
helper deliverables" above for the strategy-to-state-ids map).
Call sites:

- **Static plot** (`plot_non_local_model`): calls the dataset-level
  `conditional_non_local_posterior(results, detector)` once and
  receives a full-session `(n_time, n_pos)` array. Same shape and
  cost as the existing inline computation at
  [static.py:159](../../src/non_local_detector/visualization/static.py#L159).
- **View-models** (PosteriorHeatmapModel, SlicePanel, LikelihoodHeatmap
  posterior fallback, predictive overlay, per-cell row overlays):
  call `collapse_posterior_to_position(post_row, detector,
  reduction)` per visible row (or per cursor bin). For `reduction
  == CONDITIONAL_NON_LOCAL`, that helper delegates to the same
  generic private `_conditional_row(...)` the dataset-level helper
  iterates over (with the non-local state indices); for `reduction
  == CONDITIONAL_ON_SPATIAL` it calls the same `_conditional_row`
  with the spatial-state indices instead. **View-models must NOT
  call the dataset-level `conditional_non_local_posterior(...)` on
  every window update** — it allocates a `(n_time, n_pos)` buffer
  scaled to the full session length, which is wasteful for a
  small visible window.

Assert bit-identity between the dataset-level helper's output
(under default `zero_mass_fill=np.nan`) and the old inline
implementation via
`np.testing.assert_allclose(..., atol=1e-14, equal_nan=True)`.

The existing `non_local_model.py` helper that rejects datasets with
fewer than 4 states ([non_local_model.py:52](../../src/non_local_detector/models/non_local_model.py#L52))
is the right shape for the conditional-non-local path; the marginal
path needs no such check.

### Conditional non-local posterior — algorithm

```text
p(x_t | non_local_t) = sum_{s in non_local_states} p(x_t, s_t = s) /
                      sum_{x', s in non_local_states} p(x', s_t = s)
```

The denominator can be zero on bins where the model assigns no mass to
non-local states. The current inline implementation
([static.py:166](../../src/non_local_detector/visualization/static.py#L166))
divides without a guard, producing `nan`. The refactored standalone
function should preserve that behavior bit-for-bit (so we can assert
`1e-14` equivalence in Phase 1a's refactor test) but **also expose a
`zero_mass_fill` parameter** (default `np.nan`) that the view-model
sets to `np.nan` so the heatmap masks empty bins explicitly. Test:
construct a synthetic results dataset with a known empty bin and
assert it renders as `nan` (and as masked / lightgrey in the panel).

### Likelihood normalization

`log_likelihood` rows in results are NaN-padded at non-interior
state-bin columns (`_create_masked_posterior` writes NaN at the
non-interior positions, [base.py:2242](../../src/non_local_detector/models/base.py#L2242)).
The correct preprocessing before `np.exp` is:

1. **Map non-finite values (NaN and any -inf already present) to
   `-inf`**, not to `0`. After exponentiation, `-inf` produces
   `0.0`, which correctly excludes those bins from sums and
   peak-normalization. Cleaning to `0` would yield `exp(0) = 1.0`,
   which would falsely treat masked bins as having unit
   likelihood and would dominate any later reduction.
2. **Subtract the global max of the finite entries** from the
   row (statespacecheck pattern) for float32 overflow safety.
   Apply this **after** the `-inf` mapping so `-inf` columns stay
   `-inf` (subtracting a finite max from `-inf` is `-inf`).
3. `np.exp(...)` produces `0.0` at masked bins and finite
   non-negative values elsewhere.

### Mathematical invariants (per CLAUDE.md, tested)

The raw posterior loaded from results sums to 1 on every row, but the
view-model's reduced posterior may have all-zero rows when the selected
states have no mass for that time bin (the conditional-non-local
denominator can be zero). So the invariants are split:

- **Raw `acausal_posterior` from `load_posterior`** (interior bins
  summed). `acausal_posterior` is NaN-padded at non-interior
  state-bin columns by `_create_masked_posterior`
  ([base.py:2241](../../src/non_local_detector/models/base.py#L2241)),
  so the invariant uses `np.nansum` to skip those columns —
  matching the convention applied to all reduced-posterior
  invariants below: `np.allclose(np.nansum(post, axis=-1), 1.0,
  atol=1e-10)`. (`load_posterior` returns the full-axis array
  with NaN-padded non-interior columns; it does not strip them
  to interior-only, so consumers either `np.nansum` or
  `post[..., interior].sum(axis=-1)` with the explicit interior
  mask. A bare `post.sum(axis=-1)` would propagate NaN through
  the non-interior bins and the assertion would always fail.)
- **Reduced posterior from view-model** (after schema-aware
  reduction). The expected row-sum **depends on the active
  reduction strategy** — the two strategies have different
  normalization contracts. Note: `reduced` rows are full-position
  arrays where non-interior bins are NaN-padded (by
  `_create_masked_posterior`,
  [base.py:2241](../../src/non_local_detector/models/base.py#L2241)),
  so all row-sum assertions below use **`np.nansum(reduced, axis=-1)`**
  (or equivalently `reduced[..., interior].sum(axis=-1)` with the
  position-axis interior mask). A bare `reduced.sum(axis=-1)` would
  propagate NaN through the non-interior bins and the assertion
  would always fail.
  - `CONDITIONAL_NON_LOCAL` (renormalizes — sum-and-divide-by-mass
    is intrinsic to "conditional"):
    - On rows with positive non-local mass:
      `np.allclose(np.nansum(reduced[active_rows], axis=-1), 1.0,
       atol=1e-10)`.
    - On rows with zero non-local mass (the denominator is zero):
      `np.all(np.isnan(reduced[empty_rows]))` if
      `zero_mass_fill=np.nan` (default), or all-zero if
      `zero_mass_fill=0.0` — assert per configured fill. (Here
      every position bin in those rows is fill, so a plain
      `np.isnan` / `==0.0` check is correct without the nansum
      idiom.)
    - Active-row mask:

      ```python
      mass = np.nansum(post[:, non_local_state_bins], axis=-1)
      active_rows = mass > 0
      ```

      `post[:, non_local_state_bins]` includes NaN at non-interior
      positions: non-local states are **spatial** states whose
      bin-interior mask is inherited from the environment's
      `is_track_interior_` at
      [base.py:1038-1049](../../src/non_local_detector/models/base.py#L1038),
      and `_create_masked_posterior`
      ([base.py:2388](../../src/non_local_detector/models/base.py#L2388))
      writes NaN at those non-interior columns when assembling the
      results dataset. A bare `.sum(axis=-1)` would propagate NaN
      into `mass`, and `mass > 0` on NaN evaluates to `False`,
      misclassifying every row as zero-mass. Use `np.nansum` to
      ignore the NaN-padded non-interior bins and sum only the real
      mass — matching the canonical pattern at
      [static.py:166](../../src/non_local_detector/visualization/static.py#L166)
      where the existing inline `conditional_non_local_posterior`
      computation divides by `np.nansum(...)` for exactly this
      reason. (Singleton states are an unrelated case at
      [base.py:1037](../../src/non_local_detector/models/base.py#L1037)
      where their single bin is always interior — but non-local
      states are not singleton, so that observation does not apply
      here.)
  - `CONDITIONAL_ON_SPATIAL` (renormalizes the same way, with a
    different state selector — used by NoSpikeContFrag and
    available as a non-default override on any detector with
    singleton states):
    - Same row-sum / zero-mass-fill / active-row-mask invariants as
      `CONDITIONAL_NON_LOCAL`, but `non_local_state_bins` is
      replaced with `spatial_state_bins` (bins whose
      `bin_sizes_[state_ind_[bin]] > 1`):

      ```python
      mass = np.nansum(post[:, spatial_state_bins], axis=-1)
      active_rows = mass > 0
      ```

      For NoSpikeContFrag (`bin_sizes_ = [1, n_pos, n_pos]`),
      `spatial_state_bins` is the union of the Continuous and
      Fragmented state-bin slices; the mass equals
      `1 - P(No-Spike)` per row. For NL (`bin_sizes_ =
      [n_pos, 1, n_pos, n_pos]`) under this non-default override,
      `spatial_state_bins` includes `Local`'s slice in addition to
      the two Non-Local states' slices — that's the difference from
      `CONDITIONAL_NON_LOCAL` which excludes Local by name.
    - On rows with positive spatial mass:
      `np.allclose(np.nansum(reduced[active_rows], axis=-1), 1.0,
       atol=1e-10)`.
    - On rows with zero spatial mass (which requires
      `P(non-spatial-state) ≈ 1.0`, e.g. `P(No-Spike) ≈ 1.0` on a
      no-spike bin): zero-mass-fill semantics same as
      `CONDITIONAL_NON_LOCAL`.
  - `MARGINAL` (no renormalization — column-sum only):
    - For **fully-spatial detectors** (`SortedSpikesDecoder`,
      `ContFragSortedSpikesClassifier` — every state has
      `bin_sizes_[s] > 1`):
      `np.allclose(np.nansum(reduced, axis=-1), 1.0, atol=1e-10)`
      on every row, because `acausal_posterior` already sums to
      1.0 over the interior portion of the full `state_bins` axis
      and all that mass lives on spatial states.
    - For **detectors with singleton states** (`NonLocalSortedSpikesDetector`
      with `MARGINAL` as a non-default override): the row sum
      equals the spatial-state mass per row, which is genuinely
      `1 - np.nansum(post[:, singleton_bins], axis=-1)` — NOT 1.0.
      The singleton bin set depends on the construction-time rule
      [base.py:1034](../../src/non_local_detector/models/base.py#L1034)
      (`is_no_spike or (is_local and local_position_std is None)`),
      so:
      - Track 0 NL fixture (`local_position_std=1.0`,
        `bin_sizes_=[n_pos, 1, n_pos, n_pos]`): only `No-Spike` is
        singleton, so the row sum = `1 - P(No-Spike)`.
      - Deferred-marker NL fixture (`local_position_std=None`,
        `bin_sizes_=[1, 1, n_pos, n_pos]`): both `Local` and
        `No-Spike` are singletons, so the row sum =
        `1 - P(Local) - P(No-Spike)`.

      Assert (using `np.nansum` on both sides for non-interior
      NaN safety; singleton state-bins are always interior per
      [base.py:1037](../../src/non_local_detector/models/base.py#L1037)
      so `post[:, singleton_bins]` carries no NaN, but using
      `np.nansum` keeps the idiom uniform and robust to future
      schema changes):

      ```python
      np.allclose(
          np.nansum(reduced, axis=-1),
          1.0 - np.nansum(post[:, singleton_bins], axis=-1),
          atol=1e-10,
      )
      ```

      (Also tested explicitly in the Phase 1c property tests for
      `collapse_posterior_to_position` against both fixture
      variants so a future "fix" that adds renormalization gets
      caught against either schema.)
    - There is no "active rows" / "empty rows" split for
      `MARGINAL` because the strategy never divides — it can never
      produce NaN from a zero denominator.
- **Log-likelihood finiteness — interior bins only.**
  `log_likelihood` is NaN-padded at non-interior state-bin columns
  by `_create_masked_posterior`
  ([base.py:2242](../../src/non_local_detector/models/base.py#L2242)),
  and the helper above maps non-finite values to `-inf` before
  `np.exp`. So a blanket `np.all(np.isfinite(loglik))` is
  intentionally false on masked bins. Correct invariants:
  - Pre-mask check (raw output of the likelihood computation,
    before `_create_masked_posterior` runs):
    `np.all(np.isfinite(raw_log_likelihood))`.
  - Post-mask check (what the data source returns to the viewer):
    `np.all(np.isfinite(loglik[detector.is_track_interior_state_bins_]))`.
- `np.all(likelihood >= 0)` after exponentiation (across the full
  axis — masked bins exponentiate from `-inf` to exactly `0.0`,
  satisfying `≥ 0`).
- `np.allclose(state_probs.sum(axis=-1), 1.0, atol=1e-10)`.

### Place-field extraction

The relevant non_local_detector primitives:

- `detector.observation_models` — list of `ObservationModel`
  instances, one per state. Each carries `environment_name` and
  `encoding_group` attributes.
- `detector.encoding_model_` — dict keyed by `(environment_name,
  encoding_group)` (per the fit loop at
  [base.py:3665](../../src/non_local_detector/models/base.py#L3665)).
  Each value is a dict whose `"place_fields"` entry has shape
  `(n_cells, n_position_bins)`. Multiple observation models that
  share an encoding group produce **one** entry, not one per state.

This means ContFrag (two observation models referencing the same
encoding group) has **one** `encoding_model_` entry, so its
`place_fields` is `(n_cells, n_position_bins)` — **not**
`(n_cells, n_state_bins)`. The `predictive_posterior`'s `state_bins`
axis is full-width `n_states * n_position_bins`, so consumers that
want to align place fields against `predictive_posterior` must
concatenate per state by iterating `observation_models`.

The existing in-repo consumer
[`static.py:128`](../../src/non_local_detector/visualization/static.py#L128)
hardcodes `detector.encoding_model_[("", 0)]["place_fields"]`, which
is fragile (it leaks an internal default key into a public plotting
function) and silently breaks for any model with a non-default
environment name or encoding group. The viewer's per-cell display
needs `(n_cells, n_position_bins)`; Track A's cache-builder
validation needs `(n_cells, n_state_bins)`. Two distinct shapes →
two helpers, each with one job, both shipped in a new general-purpose
module **`src/non_local_detector/analysis/place_fields.py`** (not in
the viewer — these are detector utilities useful to anyone working
with a fitted model):

```python
def analysis.place_fields.extract_per_cell_place_fields(detector: _DetectorBase) -> np.ndarray:
    """Per-cell display shape: (n_cells, n_position_bins).

    For per-cell display use cases — slice-panel per-cell row plots,
    raster place-field-peak sort, the static plot's raster sort. State
    doesn't matter for display; for v1's single-encoding-group
    detectors all states share the same place fields.

    v1 restricts to a single ``encoding_model_`` entry (single
    environment, single encoding group) so "which state's fields do we
    show" never has to be answered. Multi-environment / multi-group
    detectors are on the v3+ roadmap.

    Returns
    -------
    np.ndarray, shape (n_cells, n_position_bins)
    """
    keys = list(detector.encoding_model_.keys())
    if len(keys) != 1:
        raise ValueError(
            f"v1 viewer per-cell display requires exactly one "
            f"encoding-model entry. Found {len(keys)} entries with keys "
            f"{keys}. Multi-environment / multi-group support is on the "
            f"v3+ roadmap; consider re-fitting as a single-group model "
            f"or wait for v3."
        )
    return detector.encoding_model_[keys[0]]["place_fields"]


def extract_state_aligned_place_fields(detector: _DetectorBase) -> np.ndarray:
    """State-bin-aligned shape for **rectangular** detectors only.

    Returns place fields concatenated along axis=1 so the result
    aligns with the rectangular portion of the ``predictive_posterior``'s
    ``state_bins`` axis: shape ``(n_cells, n_states * n_position_bins)``.

    **Rectangular detectors only.** Defined as: every entry in
    ``detector.bin_sizes_`` equals ``n_position_bins`` (no singleton
    ``Local`` / ``No-Spike`` states). This covers ``SortedSpikesDecoder``
    (single state, n_pos bins) and ``ContFragSortedSpikesClassifier``
    (two states, both spatial). It does **not** cover
    ``NonLocalSortedSpikesDetector`` regardless of
    ``local_position_std``: ``No-Spike`` is always a singleton
    (``bin_sizes_[1] == 1``), so ``bin_sizes_`` is always
    non-rectangular — ``[n_pos, 1, n_pos, n_pos]`` when
    ``local_position_std`` is set, ``[1, 1, n_pos, n_pos]`` when it
    is ``None``
    ([base.py:1029-1037](../../src/non_local_detector/models/base.py#L1029),
    [_defaults.py:182-186](../../src/non_local_detector/models/_defaults.py#L182)).
    The helper raises a clear error for those because
    "concatenated place fields" don't have a meaning for the
    singleton ``No-Spike`` column. Use
    ``extract_per_cell_place_fields`` (per-cell display, the slice
    panel's row builder, the raster sort) instead.

    Iterates ``detector.observation_models`` and looks up each state's
    ``encoding_model_`` entry by its ``(environment_name,
    encoding_group)`` key. For shared-encoding-group cases (ContFrag
    default) the same per-cell fields are repeated
    ``len(detector.observation_models)`` times; for hypothetical
    rectangular multi-group fits each state's fields are picked up
    distinctly.

    For Track A cache-builder validation, callers typically apply
    ``detector.is_track_interior_state_bins_`` to the result and
    compare against ``predictive_posterior.dropna(dim="state_bins")``
    — i.e. interior-only — since that is what statespacecheck stores
    and validates ([data_source.py:144](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/interactive/data_source.py#L144)).
    Track A is unaffected by the rectangular-only restriction
    because statespacecheck only caches ``continuous`` and
    ``contfrag`` (both rectangular).

    Returns
    -------
    np.ndarray, shape (n_cells, n_states * n_position_bins)

    Raises
    ------
    ValueError
        If the detector is non-rectangular (any ``bin_sizes_[i] != n_position_bins``).
    """
    bin_sizes = np.asarray(detector.bin_sizes_)
    if bin_sizes.size == 0 or not np.all(bin_sizes == bin_sizes[0]) or bin_sizes[0] == 1:
        raise ValueError(
            f"extract_state_aligned_place_fields requires a rectangular "
            f"detector (every state has the same n_position_bins worth of "
            f"bins, with n_position_bins > 1). Got bin_sizes_={bin_sizes.tolist()}. "
            f"This typically means the detector has singleton states like "
            f"`Local` or `No-Spike` (e.g. NonLocalSortedSpikesDetector). "
            f"For per-cell display use extract_per_cell_place_fields. "
            f"For state-aligned place fields on non-rectangular detectors, "
            f"build a custom layout from detector.state_ind_ + bin_sizes_."
        )
    return np.concatenate(
        [
            detector.encoding_model_[(obs.environment_name, obs.encoding_group)]["place_fields"]
            for obs in detector.observation_models
        ],
        axis=1,
    )
```

Statespacecheck's
[`extract_place_fields_concat`](../../../statespacecheck-paper-viewer/src/statespacecheck_paper/real_data_analysis.py#L409)
implements the same idea against the same primitives — listed here
as a corroborating reference, not a code dependency. The new helpers
are written natively against `non_local_detector` and re-export
through `non_local_detector.analysis.__init__`.

Caller routing (each module uses exactly one of these; all import
from `non_local_detector.analysis.place_fields`):

| Caller                                            | Helper                                  |
|---------------------------------------------------|-----------------------------------------|
| `SliceModel` per-cell row builder                 | `extract_per_cell_place_fields`         |
| `RasterModel` place-field-peak sort               | `extract_per_cell_place_fields`         |
| Static `plot_non_local_model` refactor (Phase 1)  | `extract_per_cell_place_fields`         |
| Track A Path 2 cache-builder validation           | `extract_state_aligned_place_fields`    |

**Nothing in the repo reaches into
`detector.encoding_model_[...]` directly except the helpers and
the writer.** Enforced by an AST-based scan paired with — but
**broader-scoped than** — the Qt-import rule (see Phase 1b for
the `ast.parse(...)` pseudocode and the rationale for the scope
split): walks every `*.py` under **`src/non_local_detector/`**
(repo-wide, not just the interactive subtree), looks for
`Subscript` nodes whose `value` is an `Attribute` matching
`*.encoding_model_`, and fails if any are found in files outside
the allowlist (`src/non_local_detector/analysis/place_fields.py`
— the helper — and `src/non_local_detector/models/base.py` — the
fit loop that writes the dict). The repo-wide scope catches any
new consumer added anywhere, not just inside the viewer. AST-based
rather than grep so prose mentions in docs/comments don't
false-positive.

**Phase 1a deliverables (the place-field-extraction half of Phase 1a;
the conditional-posterior helper has its own Phase 1a deliverable
above):**

1. Create `src/non_local_detector/analysis/place_fields.py` with the
   two helpers above. Re-export both from
   `non_local_detector.analysis` package init.
2. Refactor [`static.py:128`](../../src/non_local_detector/visualization/static.py#L128)
   to call `analysis.place_fields.extract_per_cell_place_fields(detector)` (the existing
   `np.nanargmax(place_fields, axis=1)` followed by indexing into
   `env.place_bin_centers_` is consistent with `(n_cells, n_pos)`
   shape, so this is a drop-in cleanup). Assert the static-plot
   output (the conditional-non-local-posterior array, which
   carries NaN at non-interior positions per
   [static.py:169](../../src/non_local_detector/visualization/static.py#L169))
   is bit-identical before/after via
   `np.testing.assert_allclose(after, before, atol=1e-14,
   equal_nan=True)`.
3. Property tests for both helpers (Track 0 fixture, **all four
   v1 detectors**):
   - `extract_per_cell_place_fields(d).shape == (n_cells, n_pos)`
     for SortedSpikesDecoder, ContFrag, NoSpikeContFrag, and NL
     detectors (all have a single encoding-model entry in v1
     defaults).
   - `extract_state_aligned_place_fields(d).shape ==
     (n_cells, sum(d.bin_sizes_))` for **rectangular** detectors:
     `(n_cells, n_pos)` for SortedSpikesDecoder and
     `(n_cells, 2 * n_pos)` for ContFrag (where the result is two
     horizontal copies of the per-cell output because the two
     `ObservationModel()` entries share the encoding group —
     [_defaults.py:80](../../src/non_local_detector/models/_defaults.py#L80)).
   - `extract_state_aligned_place_fields(detector)` **raises
     `ValueError`** for **both** detectors with singleton states:
     `NoSpikeContFragSortedSpikesClassifier` (`bin_sizes_=[1, n_pos,
     n_pos]`, `No-Spike` singleton) and
     `NonLocalSortedSpikesDetector` (`bin_sizes_=[n_pos, 1, n_pos,
     n_pos]` for `local_position_std=1.0`, or `[1, 1, n_pos, n_pos]`
     for `local_position_std=None`). In every case the error
     message names the non-rectangular `bin_sizes_` and points to
     `extract_per_cell_place_fields` for per-cell display.
4. Multi-entry detector → clear error: hand-construct a detector
   with two distinct encoding-model keys, assert
   `extract_per_cell_place_fields` raises with both keys named.

### Place fields for ContFrag

`ContFragSortedSpikesClassifier` has `n_states = 2`. The slice panel's
per-cell rows show one place field per cell. Decision: collapse across
states (use the Continuous state's place field — these are the canonical
place fields). Mirrors statespacecheck's
`_extract_place_fields_concat`. Document the choice.

## Risks

1. **PySide6 install pain on macOS / Linux.** Mitigated by optional
   extra. Test install instructions on both platforms before shipping.
2. **`fig.scenes` / event-loop reentrancy.** Threadpool worker emits a
   signal that touches widgets — must use `QtCore.Signal` and connect
   on the main thread (statespacecheck's `_LoadSignals(QObject)`
   pattern; replicate verbatim).
3. **Qt UI tests are flaky in headless CI.** Mitigation: use
   `pytest-qt` `qtbot` for non-rendering logic (data flow, signal
   emission). Pixel-perfect rendering tests run locally only, marked
   `@pytest.mark.gui` and skipped in CI.
4. **Plugin contract drift.** If `TimeAxisPanel` / `BinSyncedPanel`
   change after v1, downstream panels break. Mitigation: ship the ABCs
   at v1 with a documented deprecation policy; resist API changes for
   at least one minor version.
5. **View-model / renderer split discipline drift.** A maintainer
   importing `pyqtgraph` from a view-model module would silently break
   v2. Mitigation: CI AST-based import-boundary test (Phase 1b)
   that walks every `*.py` under
   `src/non_local_detector/visualization/interactive/` and fails
   if any Qt module is imported outside the
   `viewer/qt.py` + `panels/qt/` allowlist. Grep was rejected as
   the implementation because the same phase adds `pyqtgraph` /
   `PySide6` to `pyproject.toml`'s `[viewer]` extra and the
   deferred-model section names them in prose — both legitimate
   string occurrences that grep would false-positive on.
6. **PositionGrid abstraction may be wrong for 2D.** v1 has no 2D
   tests. Mitigation: design `PositionGrid` carefully reading the
   existing 2D paths in `figurl_2D.py` and `make_single_environment_movie`,
   but accept that v3 may need to revise it.
7. **Per-run schema drift in multi-run mode.** Two loaded runs may
   have different `state_names` and `bin_sizes_` (e.g. one is
   `NonLocalSortedSpikesDetector` with 4 states, another is
   `NoSpikeContFragSortedSpikesClassifier` with 3 states and a
   `No-Spike` singleton, another is `SortedSpikesDecoder` with 1
   state). On swap, the StateProbabilityPanel re-renders its line
   set (acceptable), and the PosteriorHeatmapPanel **re-selects
   the reduction strategy** via `select_reduction(state_names,
   bin_sizes_)`:
   - Any state name contains `"Non-Local"` → `CONDITIONAL_NON_LOCAL`
     (NL detectors).
   - Else, any singleton state present (`bin_sizes_[s] == 1` for
     some `s`) → `CONDITIONAL_ON_SPATIAL` (NoSpikeContFrag).
   - Else (fully-spatial) → `MARGINAL` (Decoder, ContFrag).

   The PosteriorHeatmapModel exposes a
   `set_active_run(state_names, bin_sizes_)` method (or equivalent
   re-bind hook) that the swap handler calls; the model recomputes
   the strategy and triggers a re-render. Phase 6 verification
   (Track 0 + Track A) cycles NL → ContFrag → NoSpikeContFrag →
   Decoder and asserts the heatmap re-renders correctly under
   each strategy: NL → conditional non-local (sums to 1 on
   non-local-active rows), ContFrag and Decoder → marginal (sums
   to 1 over interior bins, all states spatial), NoSpikeContFrag
   → conditional-on-spatial (sums to 1 on rows with positive
   spatial mass). Title bar updates to indicate the new reduction
   on each swap.

## Out-of-scope (explicitly)

- **No new decoding math.** This is a viewer for existing outputs.
- **No new likelihood models.** Per-cell `event_*` metrics
  (HPD overlap, KL divergence, spike prob) are v2 work.
- **No GUI for fitting / training.** Read-only viewer.
- **No "shareable cloud URL" model.** That's what `figurl_*.py` is
  for. v2's browser backend is for remote-server interactive use, not
  for sharing snapshots.
- **No reimplementation of Bokeh inside Qt for "shareability".** Pick
  one backend per use case and let the others serve their own.

## Open questions for review

None. Both prior layout questions resolved: subdirectories per backend,
sample notebook ships with v1.
