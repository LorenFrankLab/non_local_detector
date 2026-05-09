<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Phase 2 — UX correctness + discoverability

**Goal:** fix the highest-impact UX issues — scientific-misinterpretation
risk in the slice legend, undiscoverable shortcuts, and silently-cropped
panel messages. Commit cluster on `interactive-decoder-viewer`.

**Estimated effort:** ~5 hours.

**Skill:** `scientific-tdd` for new behaviour (help dialog,
overlay-mode reaction, go-to-time).

**Dependencies:** Phase 1 done and reviewed by user.

> Read [README.md](README.md) first if you haven't already. The
> Working-with-this-plan-in-Claude-Code rules apply to every task
> here.

---

## 2.1 — Slice legend reflects active overlay mode

**Bug:** [slice.py:322–333](../../../../src/non_local_detector/visualization/interactive/panels/qt/slice.py#L322-L333)
hardcodes "Overlay" with the predictive-blue color regardless of the
active mode. A user on "Smoothed" reading "Overlay = predictive"
draws the wrong scientific conclusion.

- [x] Refactor `_build_legend_html` (or its caller) to accept the
  current `OverlayMode` and emit one of:
  - `"Predictive (causal)"` (mode = `"predictive"`)
  - `"Filtered"` (mode = `"filtered"`)
  - `"Smoothed (acausal)"` (mode = `"smoothed"`)
- [x] In `set_overlay_mode`, rebuild the legend HTML and call
  `setTitle(...)` with the new string.
- [x] Test (in
  [test_slice_panel.py](../../../../src/non_local_detector/tests/interactive/test_slice_panel.py)):
  for each of the three modes, after `set_overlay_mode(mode)` the
  legend HTML contains the mode's display name.

---

## 2.2 — `?` shortcut → in-app help dialog

**Issue:** 14 keyboard shortcuts and the click-to-pin gesture are
entirely undiscoverable. Notebook users via `launch_qt(bundle)`
never see the CLI epilog.

- [x] **Step 1 (extract)**: shortcuts currently live inline in
  `QtViewer.__init__` around
  [viewer/qt.py:661](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L661) —
  there is no `_register_shortcuts` method yet. Extract the
  `(QKeySequence, callable)` pairs from the inline block into a
  new `_register_shortcuts(self) -> None` method called from
  `__init__`. Behaviour-preserving — full sweep must stay green
  after the extract, before any other change.
- [x] **Step 2**: define a single
  `_SHORTCUT_TABLE: list[tuple[str, str, str, str]]` near the top
  of [viewer/qt.py](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py)
  with `(category, key, action, description)` rows. The `action`
  field is a string identifier (e.g. `"step_left"`,
  `"toggle_play"`, `"show_help"`) — the table stays static at
  module level (no bound-method references). Categories:
  `Navigation`, `Window`, `Playback`, `Pin`, `Model`, `Help`.
- [x] **Step 3**: refactor the freshly-extracted
  `_register_shortcuts` to drive registration off `_SHORTCUT_TABLE`.
  Build a `self._shortcut_handlers: dict[str, Callable[[], None]]`
  in `__init__` mapping each `action` string to the bound method
  (`{"step_left": self._step_left, "toggle_play": self._toggle_play,
  ...}`). Iterate `_SHORTCUT_TABLE` and connect each `(key,
  handlers[action])` pair. Add a startup assertion that every
  `action` in the table has a handler in the dict so a future
  table edit can't silently drift from the registration.
- [x] **Step 4**: add `?` shortcut → `_show_help_dialog()` opens a
  modal `QDialog` rendering the table grouped by category. Use a
  `QPlainTextEdit` (read-only, monospace) or a `QLabel` with
  rich-text markup.
- [x] Test: `?` shortcut triggers a method that produces a
  non-empty string covering at least the 14 existing shortcuts.

---

## 2.3 — Tooltips on every controls-bar widget

**Issue:** controls bar has 10–15 widgets in a flat row with no
labels beyond an icon or a one-word prefix. Window slider, play
button, speed combo, slice overlay combo, model combo, overlay
combo, and visibility checkboxes are all undiscoverable.

- [x] Add `setToolTip(...)` calls in
  [viewer/qt.py:_build_controls_bar](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L709-L818)
  for every interactive widget. Reuse strings from
  `_SHORTCUT_TABLE` where applicable.
  - Window slider: "Window width in seconds. Use [ / ] keys or
    scroll wheel to resize."
  - Play button: "Play/pause autoscroll (Space). , and . step the
    speed."
  - Speed combo: "Autoscroll rate × real-time. , and . cycle through
    options."
  - Slice overlay combo: "Which posterior to render as the blue
    overlay curve in the slice panel."
  - Model combo: "Active run; M cycles forward."
  - Overlay combo: "Active event-overlay set; N / Shift+N navigate
    events."
- [x] Test: filter the controls bar's children to *interactive*
  widget types only — `findChildren(QtWidgets.QAbstractButton) +
  findChildren(QtWidgets.QComboBox) +
  findChildren(QtWidgets.QAbstractSlider)` — and assert each has
  a non-empty `toolTip()`. Walking *all* `QWidget` children would
  catch decorative `QLabel`s, `QFrame` separators, and the
  controls-bar container itself, which don't take tooltips.

---

## 2.4 — Disable "Predictive" / "Filtered" combo items when missing

**Issue:** [viewer/qt.py:142–146](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L142-L146)
combo always offers all three modes; if the active run lacks the
required outputs the overlay renders blank. Per
[base.py:423](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L423)
`acausal_posterior` is required to construct a `RunBundle`, so
**"Smoothed" is always available** — the conditional modes are:

- **Predictive** requires `predictive_posterior` in `results.data_vars`
- **Filtered** requires *both* `predictive_posterior` AND
  `log_likelihood` (it multiplies them via `_filtered_row` in
  [slice.py](../../../../src/non_local_detector/visualization/interactive/panels/qt/slice.py))
- **Smoothed** is always available (the `acausal_posterior` invariant)

- [x] Add a helper `_available_overlay_modes(run: RunBundle) -> set[OverlayMode]`
  in [viewer/qt.py](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py)
  that returns `{"smoothed"}` plus the conditional modes based on
  `run.results.data_vars`.
- [x] Update `_build_controls_bar` to enable/disable combo items
  via `model.item(i).setEnabled(...)`.
- [x] Add per-item tooltip explaining the rebuild command (re-run
  `predict(return_outputs=[...])` listing exactly which outputs
  to add — `predictive_posterior` for Predictive;
  `predictive_posterior` + `log_likelihood` for Filtered).

  **Implementation detail:** `QComboBox.setToolTip(...)` only
  applies to the combo widget itself, not to the popup-list
  items. Per-item tooltips that show on a *disabled* row require
  setting data on the item model with `Qt.ToolTipRole`:

  ```python
  model = self._slice_overlay_combo.model()
  item = model.item(i)
  item.setEnabled(False)
  item.setData(rebuild_message, QtCore.Qt.ToolTipRole)
  ```

  Without `Qt.ToolTipRole` the user hovering the disabled item
  sees no explanation and the task ships looking-done while the
  UX gap is unresolved. **[manual]** verify by hovering the
  disabled item in a real GUI run — offscreen pytest cannot
  observe tooltip popups. Surface to user before closing the task.
- [x] Re-evaluate available modes in `_on_active_run_changed` (so
  M-key swap to a run with different outputs updates the combo).
- [x] If the currently-selected mode becomes unavailable on a swap,
  fall back to `"smoothed"` (always present) and emit
  `_on_slice_overlay_changed` so the buffer refreshes.
- [x] Tests:
  - Build a `RunBundle` whose `results` has no `predictive_posterior`;
    assert `Predictive` and `Filtered` items are disabled, `Smoothed`
    is enabled.
  - Build one with `predictive_posterior` but no `log_likelihood`;
    assert `Predictive` enabled, `Filtered` disabled.
  - Swap from a run with all three modes (active = `"predictive"`)
    to a run missing `predictive_posterior`; assert active mode
    falls back to `"smoothed"`.

---

## 2.5 — Missing-output message moves out of pyqtgraph titles

**Issue:** [likelihood.py:39–43](../../../../src/non_local_detector/visualization/interactive/panels/qt/likelihood.py#L39-L43)
sets an 82-char instructional message via `setTitle(...)`;
pyqtgraph crops to the panel width. User sees a blank rectangle.

**Scope is `QtLikelihoodHeatmapPanel` only.** There is no predictive
heatmap panel in the current Qt panel set (predictive availability
is the slice-overlay-combo concern handled by Task 2.4). State
probabilities are required at `RunBundle` construction time
([base.py:425](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L425))
so a missing-output state can't occur for that panel either. The
likelihood heatmap is the only left-column panel with a real
missing-output disabled state.

- [x] In `QtLikelihoodHeatmapPanel`, replace `setTitle(MESSAGE)`
  with either:
  - A `QLabel` overlay anchored top-center via the panel's layout
    (preferred — wraps text), or
  - A `pg.TextItem` anchored to the plot's center (acceptable —
    truncation easier).
- [x] Reuse the same message string in the disabled-combo tooltip
  from 2.4 for `Filtered` (single source of truth — both the slice
  combo and the heatmap panel point at the same "re-run
  `predict(return_outputs=['log_likelihood'])`" instruction).

  *Implementation note:* the canonical `MISSING_DATA_MESSAGE`
  string lives on `QtLikelihoodHeatmapPanel`; the Filtered combo
  tooltip is a context-specific superset (predictive +
  log_likelihood) wired via `_OVERLAY_DISABLED_TOOLTIP` in
  `viewer/qt.py`. Both share the same rebuild idiom; not a
  literal-string share because the two contexts need different
  output lists.
- [x] Test: build a panel for a run missing `log_likelihood`;
  assert the overlay/text-item is visible and contains the rebuild
  command.

---

## 2.6 — Go-to-time keyboard shortcut

**Issue:** users with an external timestamp (e.g. ripple time from
a separate analysis) have no way to jump directly; only relative
navigation is supported.

- [x] Add `g` to `_SHORTCUT_TABLE` (category: `Navigation`) →
  `_show_go_to_time_dialog()`.
- [x] Dialog: `QInputDialog.getDouble(...)` accepting `t` in seconds,
  with `min=time[0]`, `max=time[-1]`, `decimals=3`. On OK, call
  `self._core.set_t_center(t)`.
- [x] Validation error message includes the valid range when out of
  bounds.

  *Implementation note:* `QInputDialog.getDouble` enforces the
  `[min, max]` clamp natively at the widget level, so an
  out-of-range entry is impossible to submit; the dialog header
  spells out the valid range explicitly. No separate validation
  branch is needed.
- [x] Test: `g` shortcut bound; calling `_show_go_to_time_dialog`
  with a mocked dialog return value of `5.0` recenters the core.

---

## Phase 2 done when

- [x] All six tasks pass tests
- [x] Full sweep green (verify command from
  [README.md](README.md) §Per-phase verification) — 269 passed
  (Phase 1 baseline 261 → +8 new tests)
- [ ] `[manual]` item from 2.4 (disabled-combo tooltip hover) surfaced
  to user

Then **stop and wait for user review** before proceeding to
[Phase 3](phase-3-interaction-feel.md).
