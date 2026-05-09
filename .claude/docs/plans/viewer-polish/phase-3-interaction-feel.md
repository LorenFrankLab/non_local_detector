<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Phase 3 — Interaction feel + parity

**Goal:** kill absolute-tick churn during playback/scrub and fix
the wheel-resize feel on touchpads. Match
`statespacecheck-paper-viewer`'s hot-path interaction defaults.
Commit cluster on `interactive-decoder-viewer`.

**Estimated effort:** ~4 hours.

**Skill:** `scientific-tdd` for new behaviour (relative
rendering, wheel-delta scaling).

**Dependencies:** Phase 2 done and reviewed by user. **Task 3.1
is the most invasive of this phase — visual-regression test it
first.**

> Read [README.md](README.md) first if you haven't already. The
> Working-with-this-plan-in-Claude-Code rules apply to every task
> here.

---

## 3.1 — Relative time-axis rendering

**Issue:** [viewer/qt.py:1003](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L1003)
pins panels to absolute time ranges; every load triggers full
x-axis relabeling. Reference
(`statespacecheck-paper-viewer/viewer.py:848`) keeps visible range
fixed at `[-w/2, +w/2]` and renders data at `t - t_center`.
Eliminates absolute-tick churn during autoscroll.

- [x] Update each panel's `update_window` (PosteriorHeatmapPanel,
  LikelihoodHeatmapPanel, RasterPanel, StateProbabilityPanel,
  generic series panels) to render at relative coordinates
  (`payload.time - t_center`).
- [x] Update `EventOverlayMixin`, the pin line, and the true-position
  line to use relative coordinates.
- [x] Lock the visible x-range once with
  `setXRange(-t_width/2, +t_width/2, padding=0)` and update only
  when `t_width` changes (not when `t_center` changes).
- [x] **Tick labels stay relative** (e.g. `-0.5 s`, `0`, `+0.5 s`).
  Do *not* add a tick formatter that remaps to absolute time —
  absolute labels would change on every scrub and re-introduce the
  exact churn this task is meant to eliminate. Reference does the
  same: relative axis labels + a separate absolute readout.
- [x] The existing absolute-time readout in the controls bar (the
  `t=...` label rendered by `_format_time_label`) is the one place
  absolute time is shown; verify it still updates correctly under
  relative rendering.
- [x] Cross-panel x-link: keep the existing wiring; relative
  coordinates work the same way.
- [x] Tests:
  - Pin behaviour on a uniform grid: scroll one window forward,
    assert the visible x-range did NOT change AND the rendered
    tick labels are unchanged (deterministic relative ticks).
  - Pin behaviour on a non-uniform grid (the case the half-bin-pad
    work was hard to get right): heatmap pixels still align to
    centers.
  - Absolute-time readout in the controls bar still tracks
    `t_center` accurately.
  - Multi-panel x-link still ties scrolling.

---

## 3.2 — Wheel-delta-based scaling

**Issue:** [viewer/qt.py:1106](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L1106)
uses fixed `0.9 / 1.1` per wheel event; touchpads emit many small
delta events that should produce one fluid resize, not 30 discrete
steps.

- [x] Replace fixed factor with `factor = exp(-delta * RESIZE_GAIN)`
  where `delta = event.angleDelta().y()` and
  `RESIZE_GAIN ≈ 0.001` (tune to match
  `statespacecheck-paper-viewer/viewer.py:1086` feel).
- [x] Define `RESIZE_GAIN` as a module-level constant near
  `MIN_T_WIDTH_SECONDS` for discoverability.
- [x] Test: 30 small events of `delta=120/30 = 4` produce
  approximately the same final `t_width` as one event of `delta=120`
  (within 1%).

---

## 3.3 — Cross-platform run-spec CLI

**Issue:** [app.py:74–90](../../../../src/non_local_detector/visualization/interactive/app.py#L74-L90)
splits `--run` on every colon and expects exactly 5 parts. Windows
paths embed `:` (drive letter), so the colon-delimited format is
**fundamentally ambiguous** on Windows — sequential
`partition(":")` calls don't help: the first path's drive colon is
indistinguishable from a field separator.

**Fix:** stop trying to make `:` work as a delimiter for paths.
Provide an unambiguous alternative.

- [x] Add a new flag `--run-files NAME RESULTS MODEL SPIKES POSITION`
  using `nargs=5`. Each value is a separate shell argument, so
  Windows drive letters are no longer ambiguous. May be repeated
  for multi-run mode (mirrors the `action="append"` behaviour of
  `--run` / `--run-from-dir`).
- [x] Update `_parse_run_arg` (or add a sibling for `--run-files`)
  to emit the same 5-field `spec` dict that `_load_run` consumes.
- [x] Keep `--run` as-is (back-compat) but document in its `help=`
  string that the colon-delimited form does not support paths
  containing `:` and that `--run-files` (or `--run-from-dir`) is
  the cross-platform alternative.
- [x] Update the help epilog to lead with `--run-from-dir` and
  `--run-files` examples; demote the colon-delimited `--run` to
  "POSIX shorthand".
- [x] Tests:
  - `--run-files default results.nc model.pkl spikes.npz position.parquet`
    parses and round-trips the existing CLI smoke test
  - `--run-files default 'C:\foo\results.nc' 'C:\foo\model.pkl'
    'C:\foo\spikes.npz' 'C:\foo\position.parquet'` parses on POSIX
    test runners (string handling only — file presence is checked
    later)
  - Multi-run: two `--run-files` invocations produce two distinct
    bundles in the data source

**Why not just deprecate `--run`:** existing scripts and notebooks
likely use it on POSIX. Keep it working, document the limitation,
add the unambiguous alternative.

---

## Phase 3 done when

- [x] All three tasks pass tests
- [x] Full sweep green (verify command from
  [README.md](README.md) §Per-phase verification) — 276 passed
  (Phase 2 baseline 269 → +7 new tests)
- [ ] **[manual]** touchpad smoke produces a fluid resize (surface to
  user — Claude cannot perform live touchpad input)

Then **stop and wait for user review** before proceeding to
[Phase 4](phase-4-architecture-hygiene.md).
