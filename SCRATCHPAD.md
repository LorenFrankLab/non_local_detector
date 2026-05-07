<!-- markdownlint-disable MD024 MD004 MD050 MD031 MD032 -->

# Interactive Decoder Viewer — Scratchpad

Branch-local working notes for the v1 implementation. Use freely;
**delete before merging**.

Pair file: [TASKS.md](TASKS.md) (the structured task tracker).
Plan: [docs/plans/2026-05-06-interactive-decoder-viewer.md](docs/plans/2026-05-06-interactive-decoder-viewer.md).

---

## Current focus

> What I'm working on right now. Update when context-switching.

**M1–M5 done. M6 effectively done modulo two boxes (README
screencast + Track-A optional verification + kernel run of demo
notebook with `[viewer]` installed). Pre-merge checklist next.**

Branch is in pre-merge state. Outstanding work is itemised in
[TASKS.md](TASKS.md) Milestone 6; the only blocking concern is
the cumulative-Qt-state segfault below (mitigated, not fully
fixed) and the user's call on README screencast scope.

**Cumulative-Qt-state segfault** (mitigated, not fully fixed):
running the full `test_viewer_extras.py` (33 GUI tests) crashes
inside pyqtgraph after ~14–16 viewers in a single process. Each
test constructs a fresh `QtViewer` (4 panels × ~16 graphics
items now that cursor markers are wired). pyqtgraph's
`ViewBoxMenu` actions and `LinearRegionItem` internals
accumulate Qt state that the per-test `_clear_qt_viewer_registry`
fixture can't fully drain. Mitigations applied in `6f03718`:

- Aggressive cleanup in [conftest.py](src/non_local_detector/tests/interactive/conftest.py):
  close all `QApplication.topLevelWidgets()` + `gc.collect()` +
  multiple `processEvents()` passes between tests.
- Disabled per-panel right-click menu via
  `getPlotItem().setMenuEnabled(False)` in
  [_mixins.py](src/non_local_detector/visualization/interactive/panels/qt/_mixins.py)
  `_install_click_recenter` — eliminates the original
  `axisCtrlTemplate_generic.py:36 setupUi` crash signature.

Net effect: pushed the crash from test ~13 → test ~16 (~9–11
tests further). Tests pass individually and in subset
selections. Path forward if it must be fully fixed: `pytest-forked`
for per-test process isolation (adds dev dep) or finer
pyqtgraph internal cleanup (removeItem on every child of every
PlotItem before deleteLater). Surface to user before merge.

**Important user-set rule (do not violate):**

- The user explicitly told me **"you're not allowed to defer things
  until consulting me"**. Don't punt items to a later milestone
  without surfacing them first. The plan's natural deferrals
  (Milestone 5 visual diff, Milestone 6 polish) are fine; my own
  ad-hoc "I'll do this in M4 instead" is not.
- The user reviews each milestone for issues before approving the
  next. Wait for their go before starting M4.

---

## Branch commits (since branching from main)

Latest at top:

- `9734952` Fix two findings on the lazy-launch chunk (P1 + P2)
- `ecc962a` Demo notebook (paired ipynb,py:percent)
- `5a4753b` Lazy-expose `launch` from `non_local_detector.visualization`
- `94f9f50` Smoothed-overlay choice in slice panel
- `6f03718` Cursor markers on every TimeAxisPanel + Qt segfault mitigation
- `9751ca2` Resync autoscroll cursor on every navigation path that recenters
- `52ad257` Fix sub-bin playback freeze in auto-scroll
- `8c0b1d2` Add auto-scroll: play/pause + speed combo + Space/,/. (M6)
- `1c8a943` Fix two follow-ups on d8a978e
- `d8a978e` Add real extra_bin_panels plugin lane (M5 doc-vs-code fix)
- `1eea901` Add model swap UI: dropdown + M-key + view-state preservation (M6)
- `ebcdefe` Document plugin contract for the interactive viewer (Milestone 5)
- `d38fe27` Close M4 'in-RAM window buffer' checkbox
- `de5b46d` Wire QtSlicePanel into QtViewer (Milestone 4)
- `89239ee` Add pin state to QtSlicePanel + SliceModel.cell_slice helper
- `1e10cfb` M3 close-out 3 — series-panel construction tests (9 GUI tests)
- `ce8b2f5` M3 close-out 2 — overlay selector + visibility checkboxes
- `da434db` M3 close-out 1 — raster non-local shading
- `f888429` Mark M3 done in TASKS / SCRATCHPAD
- `2ed3622` M3 step 4 — extra_panels + auto-render + wheel/keyboard
- `d722c86` M3 step 3 — overlay dispatch + N/Shift+N
- `c2e8889` M3 step 2 — generic series view-models + panels
- `aa24ed9` M3 step 1 — Likelihood/StateProb/Raster view-models + panels
- `e24d45e` Validate empty selected-state set in PosteriorHeatmapModel
- `7cb9867` Simplify Phase 1c + Phase 2
- `9b3b23e` Order-independent Qt viewer cleanup
- `94a0054` Retain QtViewer in `_LIVE_VIEWERS`
- `6268d3a` Tighten stale-result rejection
- `a92d944` Phase 2 review fixes (stale model on swap + overlay arrays)
- `211b6d7` Phase 1c followups + CLI smoke test
- `7d38177` Milestone 2 — Qt panel + viewer harness
- `6864059` Singleton-Local NL fixture + empty-input fix
- `65cafa1` Phase 1c — PosteriorHeatmapModel + payload dataclasses
- `6b8fb71` Simplify Phase 1a + 1b
- `1595874` Phase 1b — RunBundle + InMemoryDecoderDataSource
- `e17fbc4` Phase 1a — analysis helpers + static.py refactor

---

## Open questions / decisions discovered during implementation

> Anything the plan didn't anticipate that needs a call before
> proceeding. Surface to the user; once resolved, fold into
> `docs/plans/2026-05-06-interactive-decoder-viewer.md` so the plan
> stays the source of truth.

(empty)

---

## Issues encountered (and resolutions)

> Bugs caught by user review of each milestone. The user reviews
> diffs and flags issues; I fix + add regression tests. Pattern
> reliably surfaces real bugs — keep the loop.

- **Phase 2** — `QtViewer` swap held a stale `PosteriorHeatmapModel`
  bound to the original detector. Fixed by adding
  `ViewerCore.on_active_run_changed(callback)` + `QtViewer._rebind_panels`
  (commit `a92d944`).
- **Phase 2** — `EventOverlayMixin._render_overlay` used
  `overlay.times or []`, raising "ambiguous truth value" on
  multi-element NumPy arrays. Fixed with explicit `is None` checks
  (commit `a92d944`).
- **Phase 2** — Stale-result rejection rule was too loose: an older
  request committing first slipped through. Tightened to `payload.request_id
  == current_view_state.request_id` (commit `6268d3a`).
- **Phase 2** — `launch_qt(block=False)` returned 0 but PySide6 GC'd the
  window. Added module-level `_LIVE_VIEWERS` registry +
  `WA_DeleteOnClose` + autouse cleanup fixture (commits `94a0054`,
  `9b3b23e`).
- **Phase 1c followup** — Vectorized `PosteriorHeatmapModel.collapse_rows`
  diverged from the per-row `collapse_at`/`collapse_posterior_to_position`
  helper on invalid `CONDITIONAL_*` configs (vectorized produced
  `zero_mass_fill` rows; per-row raised). Added the same
  selected-state validation in `_bind` (commit `e24d45e`).
- **Pattern**: I keep deferring construction-time tests as
  "visual diff is M5 work" — they're not the same thing. The user
  has called this out twice. Construction tests = does the panel
  actually wire up the option? Visual diff = pixel/render comparison
  against `plot_detector`. Don't conflate.
- **M3 review (4 issues, all fixed; uncommitted)** — User flagged
  that TASKS.md M3 boxes were checked but the live viewer didn't
  actually render the left-column stack. Pattern: construction tests
  pass on isolated panel classes, but the integration in `QtViewer`
  was missing.
  1. *High*: `QtViewer.__init__` only constructed
     `QtPosteriorHeatmapPanel`. Added Likelihood/StateProb/Raster
     models + panels, laid out top-to-bottom matching
     `plot_non_local_model`, factored x-link/overlay/click/wheel
     wiring into `_wire_panels`. Test:
     `test_qt_viewer_constructs_full_left_column_stack`.
  2. *High*: `_build_payload` never set
     `WindowPayload.state_probabilities`, so the StateProb panel and
     Raster non-local shading silently no-op'd. Added
     `InMemoryDecoderDataSource.load_state_probabilities` (always
     present per `RunBundle.__post_init__`); `_build_payload` now
     populates the field. Tests:
     `test_load_state_probabilities_returns_window_slice` +
     `test_qt_viewer_payload_routes_to_all_left_column_panels`.
  3. *Medium*: `QtRasterPanel.update_for_window` and
     `rebind_after_swap` were indented inside `_contiguous_spans`
     after its `return` — unreachable nested defs, not class
     methods. Outdented into the class. Test:
     `test_qt_raster_panel_class_methods_are_accessible` (asserts
     `hasattr(QtRasterPanel, "update_for_window")`).
  4. *Medium*: `_rebind_panels` only updated the posterior model;
     auto-built extras and the Likelihood/StateProb/Raster models
     stayed bound to the previous run. Added per-built-in
     `set_active_run` + `rebind_after_swap` calls; auto-built extras
     are torn down + rebuilt from the new run's `extra_metrics`
     while user-supplied `extra_panels` are preserved (the user
     owns their lifecycle). Tests:
     `test_qt_viewer_swap_rebinds_built_in_panel_models`,
     `test_qt_viewer_swap_rebuilds_auto_extras`,
     `test_qt_viewer_swap_preserves_user_supplied_extras`.

  Tests: 111 → 118 (7 new). Lint + suite green with `-m "not slow"`.
- **M3 review follow-up #5** — Even after #4, `_rebuild_auto_extras`
  removed widgets from the layout but left their
  `set_event_overlays` bound methods in
  `ViewerCore._on_overlays_changed_callbacks`. Future
  `_dispatch_overlays` would call into `deleteLater`'d Qt widgets.
  Added symmetric `ViewerCore.off_overlays_changed(callback)`;
  `_rebuild_auto_extras` now unregisters before delete. Tests:
  `test_off_overlays_changed_unregisters_callback` (direct unit) +
  the existing `test_qt_viewer_swap_rebuilds_auto_extras` extended
  to assert (a) the core callback list length matches `_all_panels`
  after swap, (b) no callback owner is a torn-down panel, and (c)
  `core.refresh_overlays()` after swap doesn't raise.

  Tests: 118 → 119 (1 new). Lint + suite green with `-m "not slow"`.
- **M3 review follow-up #6** — `QtLikelihoodHeatmapPanel.update_window`
  silently cleared the image when `payload.likelihood is None`,
  leaving `nl_default` users with an unexplained blank panel. The
  `RunBundle` docstring promises optional-array panels "disable
  themselves with a clear title-bar message". Added
  `MISSING_DATA_MESSAGE` class constant + `_set_title_message` that
  routes through `pg.PlotWidget.setTitle`; cleared on the next valid
  payload. Internal mirror `_title_message` lets tests assert the
  surfaced text without reaching into pyqtgraph LabelItem internals.
  Test: `test_qt_likelihood_panel_explains_missing_log_likelihood`
  asserts the message names `predict` + `log_likelihood` and clears
  on next valid payload.

  Tests: 119 → 120 (1 new). Lint + suite green with `-m "not slow"`.
- **M3 review follow-up #7** — Follow-up #6's `_set_title_message`
  used `pg.PlotWidget.setTitle(None)` to clear, but PG's
  `setTitle(None)` only hides the label; the cached
  `titleLabel.text` keeps the previous string. Verified directly:
  after `setTitle('hello')` then `setTitle(None)`,
  `titleLabel.text` still equals `'hello'` (and any future code
  path that re-shows the label without resetting text would surface
  the stale warning). My #6 test only asserted the mirrored
  `_title_message`, so it passed while user-visible text stayed
  stale. Fix: when clearing, call `setTitle('')` first (forces text
  → '') then `setTitle(None)` (hides the label). Test
  strengthened to assert both `panel.plotItem.titleLabel.text` and
  `titleLabel.isVisible()` — catches the regression even if the
  mirror diverges from the widget. Pattern note: when wrapping a
  third-party API behind a mirror attribute, never assert only on
  the mirror — always assert the externally observable state too,
  or the mirror becomes a tautology.

  Tests: 120 → 120 (test strengthened in place). Lint + suite green.
- **M4 Track A devtool (uncommitted)** —
  `bundle-from-statespacecheck-cache` CLI bundles the upstream
  `statespacecheck-paper-viewer` cache + intermediates layout into a
  CLI-compatible viewer bundle directory.
  - **Module**: `visualization/interactive/devtools/{__init__.py,
    __main__.py, bundle_from_statespacecheck.py}`. `__main__.py`
    uses argparse subparsers so future devtool subcommands slot in.
  - **Path table re-encoded locally** (12-line `_model_paths`)
    rather than importing `statespacecheck_paper.interactive.cache`,
    so the devtool runs in the project's venv without the upstream
    package installed. If upstream changes the convention, both
    repos break together regardless of import vs. local copy.
  - **Place-field cross-check**: compares the cache's interior-only
    `figure04_<model>_place_fields.npz` against
    `extract_state_aligned_place_fields(detector)[:, is_track_interior_state_bins_]`
    with `rtol=1e-5, atol=1e-6`. Error message names the likely
    fix ("rebuild whichever is older") plus shape + max-abs-diff.
  - **Detector pickle**: read source via `joblib.load` — verified
    against the real upstream `cont_model.pkl` that plain
    `pickle.load` fails with `UnpicklingError: invalid load key`
    because joblib's numpy codec produces files outside the pure
    pickle format. (My initial commit `eab0478` got this wrong; user
    review caught it before any real-data check ran. Test fixture
    now writes via `joblib.dump` so it reproduces the failure mode
    when the loader is wrong.) Round-tripped on output via
    `detector.save_model` so `app._load_run` keeps using the
    canonical pickle-based loader.
  - **Source bug in `_DetectorBase.load_results`** (caught by user
    review against real `cont_results.nc`): the loader passed every
    coord on `state_bins` (including 0-D scalars like `environments`,
    `encoding_groups`) to `set_index`, which raised
    `ValueError: PandasMultiIndex only accepts 1-dimensional
    variables`. Filter added — only 1-D coords sharing the
    `state_bins` dim become MultiIndex levels; scalar coords stay as
    scalar coords. Two regression tests in
    `tests/models/test_results_persistence.py` use 0-D scalar coords
    that mirror the real schema; verified they fail without the fix
    with the same error class. Devtool fixture's scalar-coord helper
    initially used multi-element coords (which are valid 1-D index
    levels) and missed the bug entirely; corrected to true 0-D
    scalars so the devtool happy-path test now exercises the real
    schema. End-to-end verified against real
    `cont_results.nc + cont_model.pkl`.
  - **Zarr fallback**: `--results-from-zarr` lazy-imports `zarr`
    with a clear "install zarr or drop the flag" error if missing.
    `acausal_posterior` is optional in upstream's Zarr; devtool
    revalidates both required vars and refuses to run if either is
    absent.
  - **Tests**: 10 against synthesized cache layout from `cf_fitted`
    + `sim_session` fixtures (no upstream data needed); +1
    skip-if-no-zarr for the Zarr fallback.
    - Happy path round-trips through `app._load_run`.
    - Out-dir auto-created on missing nested parents.
    - Errors: unknown model, missing required results var,
      place-fields shape mismatch, place-fields value mismatch,
      mutually-exclusive `--results-nc` + `--results-from-zarr`.
    - `--results-nc` override + `__main__.main` argv dispatch +
      position parquet preserves time index.
  - **Plan deferrals (surfaced)**:
    - ~~`--run-from-dir` convenience flag~~ — added in the
      follow-up commit (see "M4 `--run-from-dir`" entry below).
    - Full `NLD_REAL_DATA_BUNDLE_DIR`-rooted bundle-build docs
      deferred to Phase 6 docs pass; CLI `--help` covers
      per-invocation usage now.

  Tests: 120 → 130 (10 new + 1 skip). Lint + suite green with
  `-m "not slow"`.
- **M4 `--run-from-dir` (uncommitted)** — `app.py` gains
  `--run-from-dir name:dir/`. Splits on the first colon (paths may
  contain colons), validates the directory exists and contains all
  four bundle files (`results.nc`, `model.pkl`, `spikes.npz`,
  `position.parquet`), then expands to the same internal spec
  `--run` produces. Multiple `--run-from-dir` flags compose with
  each other and with `--run`. Help epilog now leads with the
  `--run-from-dir` form for the common case (devtool output).
  Tests in `test_cli.py`: happy path loads through the full Qt
  pipeline; missing dir, missing bundle file, malformed `name:dir`
  arg all surface clear argparse errors. Tests: 132 → 136 (4 new).
  Lint + interactive suite green.
- **M4 SliceModel (uncommitted)** —
  `view_models/slice.py:SliceModel`. `update_for_index(t_idx,
  posterior_row, log_lik_row, predictive_row)` returns a
  ``BinPayload``:
  - **Top curve**: `collapse_log_likelihood_to_position` when
    `log_lik_row` is present, else falls back to
    `collapse_posterior_to_position(posterior_row, detector,
    reduction)`. `top_curve_label` constants
    (`TOP_CURVE_LIKELIHOOD_LABEL`,
    `TOP_CURVE_POSTERIOR_FALLBACK_LABEL`) drive the panel-side
    title text — single source of truth for tests + panel.
  - **Predictive overlay**: same collapse path with
    `predictive_row`; `BinPayload.predictive_curve` is `None` when
    the row is missing (panel hides).
  - **Per-cell rows**: bin window inferred from time-grid midpoints
    (handles uniform + non-uniform grids; first/last bins use
    one-sided half). Cells that fired in `[t_lo, t_hi]` produce a
    `CellSlice` carrying the peak-normalised place-field row +
    spike count. Place-field normalisation is safe against
    all-zero / all-NaN cells (never divides by 0).
  - **Schema awareness**: routes everything through the
    `analysis.posterior` collapse helpers, which already enforce
    `_validate_rectangular_spatial`. `select_reduction` picks the
    default (NL → CONDITIONAL_NON_LOCAL, CF → MARGINAL).
    `set_active_run` rebinds detector + spike_times + time +
    reduction default in one call for M-key swap.
  - **Tests** (10 in `test_slice_model.py`):
    - reduction defaults: NL → CONDITIONAL_NON_LOCAL, CF → MARGINAL.
    - top curve bit-identical via `np.testing.assert_allclose(
      atol=1e-14, equal_nan=True)` against the per-row helper for
      both the loglik path and the posterior fallback.
    - predictive overlay bit-identical (uses `nl_all` bundle for
      `predictive_posterior` — `nl_fitted` is the EM result and
      doesn't carry it).
    - per-cell rows: real-spike happy path; empty-window returns
      `()`; place-field rows are bounded `[0, 1]` and finite.
    - `set_active_run` swaps detector + reduction default.
    - `update_for_index` validates bounds (IndexError on out-of-range).
  - **Deferred (surfaced)**: `CellSlice` event-metric fields
    (`event_hpd_overlap`, `event_kl_divergence`, `event_spike_prob`)
    stay at dataclass defaults until the panel pulls them off
    `bundle.events`. That's a polish follow-up after the basic
    panel renders.
  - **Correction**: I initially marked "in-RAM ring buffer
    (statespacecheck pattern)" as deferred too. User caught the
    deferral and asked me not to. Re-read upstream
    `panels.py:909`: the "ring buffer" is actually a panel-side
    *window cache* (`set_window_buffer(sl, post, lik, acausal)`),
    not a model concern. SliceModel's row-in/row-out API already
    supports it — implement the buffer in the QtSlicePanel chunk
    (next commit). Lesson: when the plan says "(statespacecheck
    pattern)", look at upstream code instead of guessing whether
    it's needed.

  Tests: 136 → 144 (10 new). Lint + interactive suite green.
- **M4 QtSlicePanel base (uncommitted)** —
  `panels/qt/slice.py:QtSlicePanel`. Population top plot (top curve
  + dashed predictive overlay), pre-allocated per-cell row pool
  (`MAX_PER_CELL_PLOTS = 6`), `(+K more)` truncation indicator,
  panel-side window buffer (`set_window_buffer(payload)` →
  `update_for_index(t_idx)` indexes locally). Out-of-buffer
  `t_idx` is a no-op. `rebind_after_swap` drops the buffer +
  clears the rendered items.
  - **Lessons logged from this chunk**:
    - User caught me deferring the "in-RAM ring buffer" — I'd
      misread it as a model-side concern. Reading upstream
      `panels.py:909` clarified it's a panel-side window cache.
      Pattern: when the plan says "(statespacecheck pattern)",
      open the upstream file before guessing.
    - `event_times` is shape `(n_events, 2)` (start/end pairs),
      not 1-D. Reading the simulation contract before testing
      against it would've avoided this.
    - pyqtgraph `PlotDataItem.getData()` returns `(None, None)`
      for empty arrays (verified — even pre-init with
      `np.empty(0)` round-trips through None). Tests must
      `assert x is None or x.size == 0`.
    - Qt's `widget.isVisible()` requires the parent to be
      `.show()`n; in offscreen tests the panel isn't shown so
      visibility is always False. Use `not widget.isHidden()`
      instead — that's the "would be visible if shown" check.
  - **Tests** (7 in `test_slice_panel.py`):
    - top curve bit-identical against `collapse_log_likelihood_to_position`.
    - predictive overlay bit-identical against `collapse_posterior_to_position(..., CONDITIONAL_NON_LOCAL)`.
    - `nl_default` fallback: top curve renders posterior collapse, predictive hides.
    - out-of-buffer `t_idx` is a no-op (no crash, no stale-data overwrite).
    - per-cell rows show active cells (scans bin window for the first hit because the simulated bin is sparse).
    - truncation indicator activates when n_cells > MAX_PER_CELL_PLOTS.
    - `rebind_after_swap` drops buffer + clears rendered items.
  - **Pinning surfaced**, not deferred: panel-side state
    (`pin_cell` / `unpin_cell` / `clear_pins` + render keeps pinned
    rows across bins, requires SliceModel `cell_slice(cell_id)`)
    is a small addition; raster→slice click wiring belongs in the
    viewer chunk. Pending user decision on whether to do the
    panel-side state in this chunk's commit (would need an
    amendment) or as part of the viewer chunk.

  Tests: 144 → 151 (7 new). Lint + interactive suite green.
- **M4 SlicePanel pinning (uncommitted)** — User picked option A:
  panel-side state now, raster click wiring in the viewer chunk.
  - `SliceModel.cell_slice(cell_id, spike_count=0)` returns a
    `CellSlice` for any cell with bounds validation. Used by the
    panel to render pinned cells that didn't fire in the bin.
  - `QtSlicePanel`: `pin_cell`, `unpin_cell`, `toggle_pin`,
    `clear_pins`, `pinned_cell_ids` (frozenset snapshot).
    `_maybe_rerender` re-runs `update_for_index(self._last_t_idx)`
    on pin changes so the user sees the change immediately.
  - Render order: pinned cells first (sorted by `cell_id` for
    stability), then active not-already-pinned. Dedup: a cell
    that is both pinned and active is rendered once with its
    real spike count (active CellSlice wins over the
    `cell_slice(spike_count=0)` placeholder).
  - `rebind_after_swap` clears the pin set — cell IDs are
    run-local, silently re-applying old pins across a model swap
    would surface the wrong place fields.
  - Label format: `"#<id>"` for active-only, `"#<id> ★"` for
    pinned (bool tested without color reliance).
  - `_last_t_idx` is reset by `rebind_after_swap` so `_maybe_rerender`
    no-ops correctly across swaps.
  - **Tests** (10 new — 3 in test_slice_model.py for
    `cell_slice`, 7 in test_slice_panel.py for pinning):
    - `cell_slice` happy path, explicit `spike_count`, bounds.
    - pin/unpin/toggle/clear set semantics.
    - bounds validation on `pin_cell`.
    - pinned-inactive cell renders with count=0 + ★ marker.
    - pinned-active cell keeps its real count from the bin.
    - render order: pinned first (cell_id ascending), then active.
    - dedup: cell that's both pinned and active appears once.
    - `rebind_after_swap` empties the pin set.

  Tests: 151 → 161 (10 new). Lint + interactive suite green.
- **M4 SlicePanel viewer integration (uncommitted)** — wires
  `QtSlicePanel` into `QtViewer` end-to-end.
  - **Layout**: split into a right-column `QHBoxLayout` body. Left
    column holds the existing built-in panels + extras
    (`_left_column_layout`); right column holds the slice panel.
    Slider stretches across the full width below the body. Existing
    `_extras_insert_index` now indexes into `_left_column_layout`
    so `_rebuild_auto_extras` keeps working unchanged.
  - **Construction from `data_source.active_run`**: per user spec,
    `SliceModel(detector=run.detector, spike_times=run.spike_times,
    time=np.asarray(run.results["time"].values))`. Same shape on
    swap via `set_active_run(...)`.
  - **Per-tick path**: slider → `_on_slider_value_changed(value)` →
    `core.set_t_center(time[value])` (existing) **plus**
    `slice_panel.update_for_index(value)` (sub-ms direct render
    against the buffered window).
  - **Window-load path**: `_on_window_loaded(payload)` adds
    `slice_panel.set_window_buffer(payload)` then
    `slice_panel.update_for_index(slider.value())` so the slice
    re-renders against the freshly-arrived buffer.
  - **Raster click**: `QtRasterPanel.cell_clicked: Signal(int)`
    resolves the clicked spot's y-row through
    `RasterModel.sort_indices` to the cell id (matches upstream's
    sigClicked-on-scatter pattern). Viewer connects
    `raster.cell_clicked → slice_panel.toggle_pin` (toggle
    semantics — click pinned cell to unpin). Click conflict with
    `ClickRecenterMixin` resolved in the mixin: skips when
    `mouse_event.isAccepted()` is True (scatter accepts on point
    hit, leaving empty-area clicks for the recenter handler).
  - **Esc shortcut**: registered in the existing keyboard-shortcut
    block; routes to `slice_panel.clear_pins`.
  - **Swap rebind** (`_rebind_panels`): added `slice_model.set_active_run(
    new_detector, new_run.spike_times,
    np.asarray(new_run.results["time"].values))` +
    `slice_panel.set_position_centers(grid.centers)` +
    `slice_panel.rebind_after_swap()` (the latter clears pins +
    drops the stale buffer).
  - **Tests** (6 new in test_qt_viewer.py):
    - viewer constructs slice model + panel from active_run; panel
      is in the body layout.
    - slider tick re-renders the slice (using a buffer-resident
      target so update_for_index actually fires).
    - window-load installs the buffer.
    - raster `cell_clicked.emit(3)` toggles the pin.
    - Esc shortcut clears all pins.
    - swap rebinds the slice model + clears pins.
  - **Lessons logged**:
    - When a test depends on an in-RAM window buffer, the target
      `t_idx` must fall inside the buffered slice. My first attempt
      used `t_idx=0` against a 0.5s buffer → silent no-op rendered
      identical data → test failed with "arrays equal" instead of
      "out of buffer". Pick targets relative to `payload.indices`,
      not absolute slider extremes.
    - Existing test
      `test_qt_viewer_constructs_full_left_column_stack` checked
      the outer `_layout` for built-in widgets, but the layout
      restructure moved them into a nested `_left_column_layout`.
      Updated the test to match. Pattern: layout restructures
      always require auditing tests that reach into widget
      hierarchies.

  Tests: 161 → 167 (6 new + 1 fixture-corrected existing test).
  Lint + interactive suite green.
- **M5 in-repo docs (uncommitted)** — plugin-author reference at
  `src/non_local_detector/visualization/interactive/README.md`.
  Covers:
  - When you don't need a plugin (the `extra_metrics` auto-build
    path with `MetricSpec` + `pd.Series`).
  - `TimeAxisPanel` Protocol (the four-method contract used by
    every left-column panel) with a per-method table.
  - `BinSyncedPanel` Protocol (used by `QtSlicePanel`; reserved for
    future per-bin readouts).
  - Composing with `ClickRecenterMixin` + `EventOverlayMixin`,
    including the `mouse_event.isAccepted()` quirk that lets
    scatter-point clicks pin without recentering.
  - Minimal custom `TimeAxisPanel` example.
  - `runtime_checkable` validation pointer +
    `tests/lint/test_import_boundary.py` Qt-import allowlist note
    for in-tree contributions.

  No new tests (docs-only change). Lint clean.
- **M6 model-swap UI** (commit `1eea901`) — controls bar now shows
  a `Model (M):` dropdown when multi-run; M-key cycles through
  runs in dropdown order. The combo's `currentIndexChanged`
  routes to `core.set_active_run`; `_rebind_panels` syncs the
  combo back (signal-blocked) after a programmatic
  `core.set_active_run`. View state (`t_center`, `t_width`,
  `active_overlay_name`) preserved across swap by virtue of
  existing rebind plumbing (verified by new test). Controls bar
  visibility now requires both single-run AND no overlays before
  hiding (model dropdown alone keeps it visible). 5 new tests + 1
  reframed visibility test.
- **M6 `extra_bin_panels` plugin lane (uncommitted)** — fixes a
  user-caught hole in the M5 README docs (commit `ebcdefe`):
  README documented `BinSyncedPanel` as part of the plugin contract
  via `extra_panels`, but the viewer's `extra_panels` path requires
  `update_window` + `set_event_overlays` (TimeAxisPanel surface).
  A bin-only widget passed via `extra_panels` would AttributeError
  at viewer construction (`_wire_panels` step).
  - **Protocol fix in `panels/base.py`**: `BinSyncedPanel` now
    matches `QtSlicePanel`'s actual buffered design:
    `set_window_buffer(payload)` + `update_for_index(t_idx)`.
    `rebind_after_swap` is documented as optional and called by the
    viewer via `getattr` so plugins without run-local caches can
    omit it.
  - **New kwarg `extra_bin_panels`** on `QtViewer`, `launch_qt`,
    and the public `interactive.launch`. Bin plugins are stacked
    below the built-in `QtSlicePanel` in the right column; the
    viewer drives them on the same buffered low-latency path
    (`set_window_buffer` on window load + `update_for_index` on
    each slider tick AND once after each window-load commit).
  - **Layout change**: right column now has its own `QVBoxLayout`
    (slice panel on top, extras below). Body's `QHBoxLayout` holds
    the left column + this right column.
  - **README rewrite**: split the plugin section into a
    "two-lane" table (TimeAxisPanel via `extra_panels`,
    BinSyncedPanel via `extra_bin_panels`); explicit warning that
    mixing them fails at construction. Per-method contract tables
    for both protocols.
  - **Tests** (6 new in `test_viewer_extras.py`):
    - layout: bin panels appear under `QtSlicePanel`.
    - `set_window_buffer` dispatched on window load (also drives
      one immediate `update_for_index` at the slider value).
    - slider tick reaches bin plugins synchronously.
    - swap calls `rebind_after_swap` when present.
    - swap with a plugin that omits `rebind_after_swap` doesn't
      crash (optional-hook contract).
    - `extra_panels=[bin_only_widget]` raises `AttributeError`
      with `set_event_overlays` in the message at viewer
      construction (documented contract: lanes are separate; fail
      early rather than render nothing).
  - **Test plumbing**: added a `_make_recording_bin_panel(*,
    with_rebind=True)` factory because PySide6's `QWidget` is
    only imported lazily in this test module — a class
    declaration at module scope referencing `QtWidgets.QWidget`
    would fail when the `[viewer]` extra isn't installed.
  - **Lessons logged**:
    - When two protocols share a kwarg slot in code but a doc
      promises both lanes, the runtime check fails at the first
      `getattr` for the missing method — silent in tests that
      don't construct a viewer with the wrong-lane plugin. User
      caught it by direct construction. Lesson: when a README
      table promises plugin variants, write a runtime test that
      passes each variant through the documented kwarg.
- **M6 auto-scroll (uncommitted)** — playback controls + Space /
  `,` / `.` shortcuts. Mirrors upstream constants
  (`AUTOSCROLL_TICK_HZ = 30.0`, `AUTOSCROLL_SPEED_OPTIONS =
  (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)`,
  `AUTOSCROLL_DEFAULT_SPEED = 0.05`).
  - Controls bar gets a play `QToolButton` (▶/⏸) + speed
    `QComboBox`. The combo's `itemData` carries the float
    multiplier directly so the per-tick path doesn't reparse the
    label.
  - `_autoscroll_timer` is lazy-allocated on play, destroyed on
    pause. Each tick: `t_center += rate / TICK_HZ` →
    `np.searchsorted` to bin index → `slider.setValue(idx)`. The
    slider is the single source of truth — its `valueChanged`
    already drives `core.set_t_center` + the per-bin slice
    update.
  - Auto-pause at end of session (toggling `play_button.setChecked(False)`
    fires `_on_play_toggled(False)` which destroys the timer).
  - Controls bar is now always visible (play+speed are universal);
    earlier "hide if no overlays AND single-run" branch removed.
  - **Tests** (7 new + 1 reframed):
    - default state paused; default speed 0.05×.
    - speed combo populated with the upstream preset list.
    - `_toggle_play()` flips checked state + button text + timer
      lifecycle.
    - speed-combo selection updates `_autoscroll_rate`.
    - `_step_speed(±1)` advances the combo and clamps at the
      ends.
    - `_autoscroll_tick()` at 8× speed advances the slider.
    - end-of-session tick auto-pauses (timer destroyed,
      play_button unchecked).
    - reframed: `test_qt_viewer_controls_bar_always_visible`
      replaces the old "hidden when single-run + no overlays" test.

  19 → 26 tests in `test_viewer_extras.py`. Lint green.
- **M6 sub-bin playback freeze fix (uncommitted)** — User caught
  that the auto-scroll commit (`8c0b1d2`) computed
  `new_t = self._core.t_center + dt` per tick and quantized to a
  slider index. At default 0.05× / 30Hz (≈1.67ms per tick) versus
  a 2ms simulated bin width, the quantized index equalled the
  current slider, `setValue` was skipped, `_core.t_center` never
  advanced, and playback froze. The new test in `8c0b1d2`
  explicitly avoided default speed (used 8×) and so missed this.
  - **Fix**: float playback cursor `_autoscroll_cursor` initialised
    on play start from the current `_core.t_center`. Each tick
    accumulates `rate / TICK_HZ` into the cursor; slider
    `setValue` only fires when the cursor crosses a bin boundary.
    Stop releases the cursor (`= None`) — that's the in-play
    sentinel.
  - **Manual scrub during play** must re-anchor the cursor to the
    slider's quantized time so playback continues from the
    user's drag target. Implemented via
    `_autoscroll_resync_lock` flag the tick path raises around
    its own `setValue`, so the slot can distinguish tick-driven
    setValue (skip resync, preserve sub-bin accumulation) from
    user drags (re-anchor cursor).
  - **Tests** (2 new, 1 reframed):
    - `test_autoscroll_accumulates_subbin_progress_at_default_speed`
      — explicit regression: at default speed, cursor advances
      per tick even when slider doesn't, and after enough ticks
      the slider does cross a bin.
    - `test_manual_scrub_during_play_resyncs_cursor` — slider
      drag during play snaps the cursor to the target time.
    - existing `test_autoscroll_tick_advances_slider` reframed:
      now calls `_toggle_play()` first to initialize the cursor
      (mirroring real usage; calling `_autoscroll_tick` without
      play active is now a documented no-op).

  26 → 28 tests in `test_viewer_extras.py`. Lint green.
  - **Lesson logged**: my "sufficient speed for visible advance"
    test (8×) hid the very freeze the user was worried about. When
    a tick frequency / bin width interaction is at issue, the
    regression test must use the *minimum* speed in the user-
    visible range (here: the default 0.05×, which is also the
    speed users will hit first). Picking a fast value to "make the
    test work" is the symptom of the bug.
- **M6 cursor resync — non-slider paths (uncommitted)** — User
  caught that the sub-bin fix only resynced the cursor on slider
  drags. Several controls move ``t_center`` without touching the
  slider: Shift+Left/Right (`_step_window`), R (`_reset_view`),
  panel click handlers wired to `core.set_t_center`, and N/Shift+N
  (`core.next_event` / `prev_event`). During play, those would
  leave the float cursor stale and the next tick would pull the
  view back.
  - **Fix**: lift the resync to ViewerCore via a new
    `on_t_center_changed(callback)` API. ``set_t_center`` fires
    the callback synchronously after committing the new value. The
    viewer subscribes a single `_sync_autoscroll_cursor_to_core`
    handler — every recenter path now resyncs uniformly.
  - The lock semantics are unchanged: tick-driven slider setValue
    still sets `_autoscroll_resync_lock` around its call so the
    resync handler skips and sub-bin accumulation is preserved.
  - Removed the redundant inline resync in
    `_on_slider_value_changed` — the slider path now goes through
    `core.set_t_center` → callback like every other path.
  - **Tests** (2 new): `test_core_set_t_center_during_play_resyncs_cursor`
    (general mechanism via direct `core.set_t_center`) +
    `test_step_window_during_play_resyncs_cursor` (specific
    Shift+Left/Right path). The pre-existing slider resync test
    still passes because the slider path now routes through the
    same callback.
  - **Lesson logged**: when several controls converge on one state
    write (here: `core.set_t_center`), centralise the side-effect
    *at the write*, not at each call site. Subscribing to
    `on_t_center_changed` covers the keyboard/click/event-jump
    paths I'd otherwise miss one-by-one.

  28 → 30 tests in `test_viewer_extras.py`. Lint green.

---

## Test results

> Notable runs, especially anything that needed multiple iterations.

### Track 0 simulated fixture

**Current totals (after M3 close-out, commit `1e10cfb`):**

- 111 interactive + lint tests pass with `-m "not slow"`.
- 4 slow tests (singleton-Local fixture) pass when run with the
  default invocation (no `--run-slow` flag — the project convention
  is `-m "not slow"` to skip).
- Per-detector EM fits: NL ≈30s, NSF ≈22s, CF ≈13s, Decoder ≈2s.
  Singleton-Local NL ≈3 minutes.
- Session-scoped fixtures keep each fit to once per pytest invocation.

**Run commands:**

```bash
# Full interactive + lint (fast):
QT_QPA_PLATFORM=offscreen uv run pytest \
  src/non_local_detector/tests/interactive/ \
  src/non_local_detector/tests/lint/ -m "not slow"

# Include singleton-Local slow tests (~5 min total):
QT_QPA_PLATFORM=offscreen uv run pytest \
  src/non_local_detector/tests/interactive/ \
  src/non_local_detector/tests/lint/

# Full repo (ran successfully after Phase 1a, 835 tests):
uv run pytest src/non_local_detector/tests/ -x -m "not slow" -q
```

**Key requirement:** GUI tests need `QT_QPA_PLATFORM=offscreen` for
headless platforms (CI / no-display dev machines).

### Track A statespacecheck real data

> Whether `NLD_REAL_DATA_BUNDLE_DIR` is set, what subdirectories
> exist, what skipped vs ran.

(empty — Track A devtool comes in Milestone 4.)

### Track B continuum integration

(empty — Milestone 5.)

---

## Performance notes

> Anything measured: per-tick latency, window-load timing, memory
> footprint of fixtures, etc.

- `PosteriorHeatmapModel.collapse_rows` was vectorized in commit
  `7cb9867`: per-row Python loop replaced with one cached
  `selected_mask` + one `np.sum(axis=1)`. Estimated ~100× speedup
  on slider tick at typical n_visible=1000.
- `conditional_non_local_posterior` (dataset-level) similarly
  vectorized in commit `6b8fb71` to match the static-plot inline
  algorithm exactly.
- ViewerCore `set_t_center` / `set_t_width` early-return on
  identical values (commit `7cb9867`) so slider re-fires at the
  same position don't burn requests.
- Outstanding (not yet a measurable problem):
  - `_LoadSignals` + `QRunnable` allocated per-tick in
    `QtBackendAdapter.schedule_window_load`. Could be reused;
    deferred until profiling flags it.

---

## API drift from plan

> If implementation reveals the planned signature is wrong (typo,
> missing arg, type mismatch with existing code), record it here and
> propose a plan amendment before changing call sites.

- `make_simulated_data` actually returns 8 elements (the docstring +
  return-type annotation said 7). Updated the annotation + docstring
  to reflect the trailing `place_fields` array. Fixture helpers
  unpack 8 elements.
- The plan's `1e-10` tolerance for posterior reduction sums and
  bit-identity comparisons is too tight for float32-stored
  posteriors. Used `1e-6` for posterior-row sums, and forced float64
  inside `_conditional_row` so refactor bit-identity assertions hold.
- The plan's `--run-slow` pytest flag for the singleton-Local fixture
  doesn't match the project's existing convention (`@pytest.mark.slow`
  with `-m "not slow"` to opt out). Used the existing convention.
- The plan documents a `joblib.dump`/`joblib.load` for the CLI
  `model.pkl`, but `_DetectorBase.save_model`/`load_model` already
  uses `pickle` directly. The CLI now routes through the canonical
  pair to keep the `state_bins` MultiIndex round-trip working
  (`xr.open_dataset` alone drops it).

---

## Phase boundary follow-ups

> Things noticed in one phase that belong in a later phase. Don't
> implement them out of order — log here and pick up in the right
> phase.

### Phase 1a → 1b — done

### Phase 1b → 1c — done

### Phase 1c → 2 — done

### Phase 2 → 3 — done

### Phase 3 → 4

- The plan's "Full left-column stack visual diff against
  `plot_non_local_model`" verification (TASKS.md line 492) is the
  only unchecked M3 item; defer to Milestone 5 per the plan's own
  Track B grouping.
- SlicePanel will need access to `RasterPayload` per-cell sort
  indices (already exposed on `RasterModel.sort_indices`). The
  per-cell rows in the slice panel should match the raster sort.
- The per-bin "non-local active" mask logic in
  `QtRasterPanel._render_non_local_regions` is reusable for the
  SlicePanel's "non-local at cursor" highlight. If it grows beyond
  threshold-comparison, consider lifting `_non_local_mass(probs,
  detector)` into `analysis.posterior`.

### Phase 4 → 5

(empty — fill as Milestone 4 progresses.)

### Phase 5 → 6

(empty)

---

## v2 / v3 deferred

> Things explicitly out of v1 scope but worth capturing while context
> is fresh. Move to a new plan file when v1 ships.

### v2 candidates

- Panel/holoviews/bokeh backend (`viewer/panel_.py`, `panels/panel_/`).
- `ZarrDecoderDataSource` for sessions too large for memory.
- `MetricPanel` (HPD overlap, KL divergence, spike prob) — requires
  porting `event_*` computations to `non_local_detector.analysis`.
- `_LoadSignals` reuse in `QtBackendAdapter` (currently allocated
  per-tick — small win, defer until profiling flags it).

### v3+ candidates

- 2D-decoder support (`PositionGrid` 2D path, "movie" panels).
- `VideoOverlayPanel` (`BinSyncedPanel` that loads a video file and
  alpha-blends 2D posterior on the current frame).
- Clusterless support (5 deferred classes — see plan's
  "Deferred-model pathway" section).
- Multi-environment support (2 deferred classes — see plan's
  "Deferred-model pathway" section).
- Side-by-side model comparison (two parallel column stacks).

---

## Misc

> Anything that doesn't fit above. Snippets, links, reminders.

**Recurring conftest pattern.** The autouse
`_clear_qt_viewer_registry` in `tests/interactive/conftest.py`
closes + `deleteLater()`s every QtViewer registered during a test,
then drains events twice. Without it the GUI suite is order-dependent
because PySide6 keeps closed top-level widgets in
`QApplication.topLevelWidgets()` until they're properly destroyed.
`QtViewer` sets `WA_DeleteOnClose` to make the cleanup actually
reap the widget.

**Detector type-checker noise.** Many test files raise ty warnings
about "object has no attribute X" because the `_simulated_detectors`
helper returns `FittedDetector` with `detector: object`. These are
ty false positives — the runtime type is fine. Ignored throughout.
