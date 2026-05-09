<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Phase 1 — Pre-merge blockers

**Goal:** unblock CI and eliminate one segfault risk + one boundary
bug. Land as a small commit cluster on
`interactive-decoder-viewer` (commits prefixed `viewer:` per the
existing branch convention).

**Estimated effort:** ~1 hour.

**Skill:** none (tactical fixes).

**Dependencies:** none — Phase 1 is the first commit cluster on
the branch.

> Read [README.md](README.md) first if you haven't already. The
> Working-with-this-plan-in-Claude-Code rules in §"Working with
> this plan" apply to every task here.

---

## 1.1 — `ruff format src/` on unformatted files

- [x] Run `uv run ruff format src/`. Five files are unformatted
  (flagged by `ruff format --check src/`):
  `devtools/bundle_from_statespacecheck.py`,
  `panels/qt/posterior.py`, `view_models/likelihood.py`,
  `view_models/raster.py`, `view_models/slice.py`.
- [x] Verify `uv run ruff format --check src/` exits 0.

**Done when:** CI quality-gate passes locally.

---

## 1.2 — `closeEvent` stops the autoscroll timer

**Bug:** [viewer/qt.py:1336](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L1336)
`closeEvent` shuts down the backend executor but leaves
`self._autoscroll_timer` firing `_autoscroll_tick` against a Qt
widget already in its destruction path. PySide6
use-after-partial-delete → segfault when a user closes the window
while playback is running.

- [x] Add `self._stop_autoscroll()` as the first line of `closeEvent`
  in [viewer/qt.py](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py).
- [x] Add a `gui`-marked test (in
  [test_qt_viewer.py](../../../../src/non_local_detector/tests/interactive/test_qt_viewer.py))
  that constructs a viewer, calls `_start_autoscroll`, calls
  `closeEvent` (or `close()`), and asserts:
  - `viewer._autoscroll_timer is None`
  - `viewer._play_button.isChecked() is False`
  - The test does *not* segfault on teardown (use
    `monkeypatch.setattr` to hold the timer-target alive long enough
    to verify the assertion).

**Done when:** test passes; no Qt teardown errors visible in pytest
output.

---

## 1.3 — `window_indices()` clamps at session boundaries

**Bug:** [data_source.py:181](../../../../src/non_local_detector/visualization/interactive/data_source.py#L181)
returns raw `searchsorted` bounds; `qt.py` indexes
`edges[sl.stop]` afterwards. Manual window stepping
([viewer/qt.py:1068](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L1068))
can push `t_center` past recording end → out-of-range edge lookup
or empty/bad window. Reference implementation clamps at
`statespacecheck-paper-viewer/data_source.py:335`.

- [x] In `InMemoryDecoderDataSource.window_indices`, clamp `sl.start`
  to `≥ 0` and `sl.stop` to `≤ len(self.time)`. **The returned slice
  must always be non-empty when the session is non-empty** — match
  the reference invariant at
  `statespacecheck-paper-viewer/data_source.py:335`.
  When `t_center` is past the end, return the rightmost
  `t_width`-wide window (`slice(len-N, len)`); when before the
  start, the leftmost. The blank-window UX at the boundary case is
  exactly what we want to avoid — there is always *some* recorded
  data to display.
- [x] Update the docstring: "returned slice is always valid for
  `time` and `time_edges` indexing AND non-empty when the session
  is non-empty; out-of-session `t_center` clamps to the nearest
  valid window."
- [x] Add a test in
  [test_data_source.py](../../../../src/non_local_detector/tests/interactive/test_data_source.py)
  that pins:
  - `window_indices(t_center=time[0], t_width=large)` →
    `slice(0, len(time))`
  - `window_indices(t_center=time[-1] + 100, t_width=1.0)` →
    `sl.stop == len(time)` AND `sl.start < sl.stop` (non-empty,
    rightmost window)
  - `window_indices(t_center=time[0] - 100, t_width=1.0)` →
    `sl.start == 0` AND non-empty (leftmost window)
  - `window_indices(t_center=time[-1], t_width=2.0)` →
    `sl.stop == len(time)` (no overshoot)
- [ ] **[manual]** `_step_window(direction=+1)` repeatedly past
  session end must not raise and must keep the rightmost window
  visible. Surface to user — Claude can confirm "doesn't raise"
  programmatically; the "keeps rightmost window visible" check is
  a live-GUI verification.

**Done when:** boundary test passes; `_step_window` past end keeps
the visible window non-empty and pinned at the session edge.

---

## Phase 1 done when

- [x] All three tasks pass tests
- [x] Full sweep green (run the verify command from
  [README.md](README.md) §Per-phase verification) — 257 passed
  (baseline 252 → +5 new tests)
- [ ] `[manual]` item from 1.3 surfaced to user

Then **stop and wait for user review** before proceeding to
[Phase 2](phase-2-ux-correctness.md).
