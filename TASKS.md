<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Interactive Decoder Viewer — Implementation Tasks

Branch-local task tracker for the v1 implementation of
[docs/plans/2026-05-06-interactive-decoder-viewer.md](docs/plans/2026-05-06-interactive-decoder-viewer.md).

**This file is temporary** — delete before merging the v1 PR. Same for
`SCRATCHPAD.md`.

Milestones map to plan phases. Each task is concrete and testable;
mark with `[x]` when done. If a task is blocked, note why on the line
or in `SCRATCHPAD.md` and leave the box unchecked.

---

## Milestone 1a — Analysis helpers (repo cleanup, no viewer scaffolding)

Pure repo cleanup; reviewable as a standalone PR. No viewer code,
no new optional deps.

### `src/non_local_detector/analysis/place_fields.py`

- [x] Create new module `src/non_local_detector/analysis/place_fields.py`.
- [x] Implement `extract_per_cell_place_fields(detector) -> np.ndarray`
  returning shape `(n_cells, n_position_bins)`. Raises `ValueError`
  on detectors with > 1 encoding-model entry, naming the offending keys.
- [x] Implement `extract_state_aligned_place_fields(detector) -> np.ndarray`
  returning shape `(n_cells, n_states * n_position_bins)`. Raises
  `ValueError` on non-rectangular detectors (any `bin_sizes_[s] == 1`),
  naming the offending `bin_sizes_` and pointing to
  `extract_per_cell_place_fields` for per-cell display.
- [x] Re-export both helpers from
  `non_local_detector.analysis.__init__`.

### `src/non_local_detector/analysis/posterior.py`

- [x] Add `conditional_non_local_posterior(results, detector,
  zero_mass_fill=np.nan) -> xr.DataArray` extracted from
  [static.py:155-169](src/non_local_detector/visualization/static.py#L155).
  Dataset-level helper; allocates full-session `(n_time, n_pos)` array.
- [x] Add `collapse_log_likelihood_to_position(log_lik_row, detector)
  -> np.ndarray` returning shape `(n_pos,)`. Includes all-non-finite
  short-circuit (return `np.zeros(n_pos)` when no spatial entries are
  finite). Per the algorithm in the plan: NaN → -inf, subtract finite
  max, exp, sum across spatial states, peak-normalize.
- [x] Add `collapse_posterior_to_position(post_row, detector,
  reduction, zero_mass_fill=np.nan) -> np.ndarray` returning shape
  `(n_pos,)`. Dispatches on `PosteriorReduction`:
  - [x] `MARGINAL`: column-sum spatial-state slices, no
    renormalization.
  - [x] `CONDITIONAL_ON_SPATIAL`: route through `_conditional_row`
    with `selected_state_ids = [s for s where bin_sizes_[s] > 1]`.
  - [x] `CONDITIONAL_NON_LOCAL`: route through `_conditional_row`
    with `selected_state_ids = [s for s where "Non-Local" in
    state_names[s]]`.
- [x] Add private `_conditional_row(post_row, detector,
  selected_state_ids, zero_mass_fill) -> np.ndarray`. Both
  `CONDITIONAL_*` paths call this. The dataset-level
  `conditional_non_local_posterior` iterates time and calls it per row.
- [x] Add `PosteriorReduction` enum with `MARGINAL`,
  `CONDITIONAL_ON_SPATIAL`, `CONDITIONAL_NON_LOCAL` variants.
- [x] Add `select_reduction(state_names, bin_sizes_) ->
  PosteriorReduction` with priority order: `"Non-Local"` in any name
  → `CONDITIONAL_NON_LOCAL`; else any singleton → `CONDITIONAL_ON_SPATIAL`;
  else → `MARGINAL`.
- [x] Re-export all four public helpers and `PosteriorReduction` from
  `non_local_detector.analysis.__init__`.

### Refactor `src/non_local_detector/visualization/static.py`

- [x] Refactor [static.py:128](src/non_local_detector/visualization/static.py#L128)
  to call `extract_per_cell_place_fields(detector)`.
- [x] Refactor [static.py:155-169](src/non_local_detector/visualization/static.py#L155)
  to call `conditional_non_local_posterior(results, detector)`.

### Tests

- [x] `src/non_local_detector/tests/analysis/test_place_fields.py`:
  - [x] `extract_per_cell_place_fields` succeeds for all four v1
    detectors (`SortedSpikesDecoder`, `ContFragSortedSpikesClassifier`,
    `NoSpikeContFragSortedSpikesClassifier`,
    `NonLocalSortedSpikesDetector`); shape always `(n_cells, n_pos)`.
  - [x] `extract_state_aligned_place_fields` succeeds for
    `SortedSpikesDecoder` (shape `(n_cells, n_pos)`) and
    `ContFragSortedSpikesClassifier` (shape `(n_cells, 2 * n_pos)`,
    horizontal copies).
  - [x] `extract_state_aligned_place_fields` raises `ValueError`
    for `NoSpikeContFragSortedSpikesClassifier` (`bin_sizes_=[1, n_pos,
    n_pos]`) and `NonLocalSortedSpikesDetector` (`local_position_std=1.0`
    config; `local_position_std=None` parametrized in Phase 1c).
  - [x] Multi-entry detector → `extract_per_cell_place_fields`
    raises with both keys named.
- [x] `src/non_local_detector/tests/analysis/test_posterior_collapse.py`:
  - [x] `collapse_log_likelihood_to_position`:
    - [x] Track 0 NL `nl_loglik` row → shape `(n_pos,)`, finite,
      peak ≈ 1.0.
    - [x] Track 0 ContFrag `cf_loglik` row → same shape, peak ≈ 1.0.
    - [x] All-non-finite row guard: all-NaN → `np.zeros(n_pos)`.
      All-`-inf` → same.
    - [x] Mixed finite + `-inf` row: `-inf` slice's columns
      contribute exactly 0.0 without producing NaN.
    - [x] No-spatial-states detector → raises `ValueError`.
  - [x] `collapse_posterior_to_position`:
    - [x] Shared row-level math — `np.array_equal(...,
      equal_nan=True)` check that `collapse_posterior_to_position`
      and the dataset-level helper both equal a direct
      `_conditional_row(...)` call.
    - [x] `zero_mass_fill` plumbing — NaN/0.0 dual assertion for both
      `CONDITIONAL_NON_LOCAL` and `CONDITIONAL_ON_SPATIAL` paths.
    - [x] ContFrag/Decoder with `MARGINAL` →
      `np.allclose(np.nansum(out, axis=-1), 1.0, atol=1e-10)` per row,
      no divide step.
    - [x] NL with `MARGINAL` (override) → row sum equals
      `1 - np.nansum(post[:, singleton_bins], axis=-1)`. (Track 0
      `local_position_std=1.0` only; the `local_position_std=None`
      parametrization is a Phase 1c follow-up.)
    - [x] NoSpikeContFrag with `CONDITIONAL_ON_SPATIAL` (default) →
      shape `(n_pos,)`; on rows with positive spatial mass,
      `np.allclose(np.nansum(out), 1.0, atol=1e-6)`; divisor
      matches `1 - P(No-Spike)`; cross-helper consistency via
      `_conditional_row`.
- [x] `src/non_local_detector/tests/visualization/test_static_plot_refactor.py`:
  - [x] `test_conditional_non_local_posterior_matches_inline` —
    paste inline algorithm from current static.py, assert
    `np.testing.assert_allclose(after, before, atol=1e-14,
    equal_nan=True)`.
  - [x] `test_place_field_peak_sort_matches_inline` — same pattern
    for the raster sort order.

### Verification (Track 0)

- [x] `uv run pytest -k "place_fields or posterior_collapse or
  static_plot_refactor"` passes.
- [x] Existing golden / snapshot tests remain green.

---

## Milestone 1b — RunBundle + in-memory data source

Viewer sub-package shell + data-flow contract. No rendering code,
no GUI deps.

### Package scaffolding

- [x] Create `src/non_local_detector/visualization/interactive/`
  package with:
  - [x] `__init__.py`
  - [x] `view_models/__init__.py`
  - [x] `view_models/base.py`
  - [x] `view_models/events.py`
  - [x] `view_models/series.py` (only `MetricSpec` at this phase)
  - [x] `data_source.py`
- [x] Add `[viewer]` optional-dependency group in `pyproject.toml`
  (`PySide6`, `pyqtgraph`).

### `view_models/events.py`

- [x] `EventOverlay` dataclass with `kind: Literal["points",
  "intervals"]`.
- [x] `EventOverlay.points(name, times, color, ...)` constructor.
- [x] `EventOverlay.intervals(name, t_start, t_end, color, ...)`
  constructor.

### `view_models/series.py` (Phase 1b: MetricSpec only)

- [x] `MetricSpec` dataclass union with `MetricSpec.line(...)`,
  `MetricSpec.scatter(...)`, `MetricSpec.intervals(...)` constructors.
- [x] (Series-model classes deferred to Phase 3 — same file.)

### `view_models/base.py`

- [x] `RunBundle` dataclass (mutable, not frozen):
  - [x] Fields: `results`, `detector`, `spike_times`,
    `position_time`, `position`, optional `speed`, optional `events`,
    `extra_metrics: dict[str, MetricSpec | pd.Series]`,
    `event_overlays: list[EventOverlay]`.
  - [x] `__post_init__` validates monotonic `position_time`,
    position dimensionality matches detector environment,
    `len(spike_times) == n_neurons`.
  - [x] `from_predict(...)` classmethod.

### `data_source.py`

- [x] `InMemoryDecoderDataSource(runs: dict[str, RunBundle])`.
  - [x] `from_single(bundle)` classmethod.
  - [x] Construction validates: (a) time-grid alignment across all
    runs, (b) overlay names unique within each bundle, (c)
    `(name, kind)` schema matches across all runs. Each mismatch
    raises with a clear message.
- [x] Hot-path methods:
  - [x] `window_indices(t_center, t_width)`
  - [x] `load_posterior(sl)` → `(n_visible, n_state_bins)` float32
  - [x] `load_likelihood(sl)` (raises clearly when `log_likelihood`
    not in results)
  - [x] `load_acausal(sl)` (returns None if absent)
  - [x] `load_predictive(sl)` (returns None if absent)
  - [x] `slice_at_index(t_idx, which="posterior"|"likelihood"|...)`
  - [x] `events_in_window(sl)`
- [x] State: `active_run` (RunBundle), `set_active_run(name)`,
  `available_outputs: set[str]`.

### CI gate (`src/non_local_detector/tests/lint/test_import_boundary.py`)

- [x] Function 1 — Qt-import scan: walk `*.py` under
  `src/non_local_detector/visualization/interactive/`, AST-parse,
  fail on `Import` / `ImportFrom` of `pyqtgraph` or `PySide6`
  outside allowlist `{viewer/qt.py, panels/qt/}`.
- [x] Function 2 — `encoding_model_[...]` scan: walk `*.py` under
  `src/non_local_detector/` (repo-wide), AST-parse, fail on
  `Subscript` of an `Attribute` matching `*.encoding_model_` outside
  allowlist `{analysis/place_fields.py, models/base.py}`.
- [x] Both functions ship in one file; both run by default `pytest`.

### Track 0 fixture (`src/non_local_detector/tests/interactive/conftest.py`)

- [x] Use `make_simulated_data(n_neurons=25, seed=0)` from
  `non_local_detector.simulate.sorted_spikes_simulation`.
- [x] Construct four detectors:
  - [x] `nl_detector = NonLocalSortedSpikesDetector(...,
    local_position_std=1.0)` (matches notebook params).
  - [x] `cf_detector = ContFragSortedSpikesClassifier(...)`.
  - [x] `nsf_detector = NoSpikeContFragSortedSpikesClassifier(...)`.
  - [x] `dec_detector = SortedSpikesDecoder(...)`.
- [x] For each detector: `estimate_parameters(...)` once with
  `is_training=~is_event` (discard return), then `predict(...)` three
  times for `default` / `loglik` / `all` variants.
- [x] Assemble 12 RunBundles (4 detectors × 3 variants) keyed
  `<det>_<variant>`.
- [x] Multi-run dict `{"nl": ..., "cf": ..., "nsf": ..., "dec": ...}`
  for swap tests.

### Smoke tests

- [x] Build three RunBundles from one fitted detector
  (default-`predict()`, `return_outputs=["log_likelihood"]`,
  `return_outputs="all"`); assert `available_outputs` reflects each;
  `load_likelihood` succeeds on loglik-bearing variants and raises
  cleanly on default.
- [x] Multi-run construction with mismatched time grids → clear
  construction error.
- [x] Multi-run construction with overlay name-set mismatch /
  schema mismatch / within-bundle duplicate names — each raises
  with the offending detail in the message.

### Verification (Track 0)

- [x] `uv run pytest -k "data_source or run_bundle"` passes against
  the simulated fixture's 12 RunBundle variants.
- [x] Default `uv sync` does not pull `PySide6`; `uv sync --extra
  viewer` does.

---

## Milestone 1c — First view-model + panel ABCs (still no Qt)

### `view_models/base.py` (additions)

- [x] Add `ViewState` (frozen): `request_id`, `t_center`, `t_width`,
  `load_acausal`.
- [x] Add `PositionGrid`, `WindowPayload`, `BinPayload`, `CellSlice`
  dataclasses.

### `view_models/posterior.py`

- [x] `PosteriorHeatmapModel`:
  - [x] Construction: pick `reduction` via
    `select_reduction(state_names, bin_sizes_)` (override allowed).
  - [x] `update_window(...)` returns `(n_visible, n_pos)` array.
  - [x] Per-row routing through
    `analysis.posterior.collapse_posterior_to_position(post_row,
    detector, reduction)`.
  - [x] `set_active_run(state_names, bin_sizes_)` rebinds strategy
    + triggers re-render.

### `panels/__init__.py`, `panels/base.py`

- [x] `TimeAxisPanel` Protocol with: `update_window(payload)`,
  `x_link_target()`, `click_handler(callback)`,
  `set_event_overlays(overlays)`.
- [x] `BinSyncedPanel` Protocol with: `update_for_index(t_idx,
  payload)`.

### Tests

- [x] `src/non_local_detector/tests/interactive/test_posterior_model.py`:
  - [x] NL bundle → `reduction == CONDITIONAL_NON_LOCAL`; rendered
    array matches dataset-level helper bit-identically on interior
    bins (`atol=1e-12`, `equal_nan=True`).
  - [x] ContFrag bundle → `reduction == MARGINAL`; every row sums
    to 1.0 via `np.nansum`; no active/empty split.
  - [x] NoSpikeContFrag bundle → `reduction == CONDITIONAL_ON_SPATIAL`;
    positive-spatial-mass rows sum to 1.0.
  - [x] Decoder bundle → `reduction == MARGINAL`; rows sum to 1.0
    trivially.
  - [x] Schema-swap reduction test (data-source + view-model layer
    only — no viewer instantiation): for each of `("nl", "cf", "nsf",
    "dec")`, `set_active_run(name)` then construct fresh
    `PosteriorHeatmapModel` and assert reduction matches expected
    strategy. Assert re-render at same `t_idx` produces a different
    output array per detector.

### Singleton-Local NL fixture (Phase 1c follow-up — required before v1 ships)

The plan's Track 0 fixture defaults to `local_position_std=1.0`
(fast, `bin_sizes_=[n_pos, 1, n_pos, n_pos]`). The fully-singleton
variant `local_position_std=None` (`bin_sizes_=[1, 1, n_pos, n_pos]`)
is the original NL design and is referenced by Phase 1a place-field
tests + Phase 1c posterior-collapse `MARGINAL` tests as a slow-marker
parametrization. **Required before v1 ships**, not deferred to v3.

- [x] Add ``nl_singleton_fitted`` fixture in
  ``src/non_local_detector/tests/interactive/conftest.py`` and
  ``src/non_local_detector/tests/analysis/conftest.py``
  (constructs an NL detector with ``local_position_std=None``).
- [x] Mark the slow variant ``@pytest.mark.slow``; the project's
  existing ``slow`` marker is registered in ``pyproject.toml``.
  Project convention is to skip slow tests with ``-m "not slow"``;
  CI runs all tests by default (no ``--run-slow`` flag introduced).
- [x] Phase 1a place-field tests: ``test_nl_detector_singleton_local_raises``
  exercises the ``[1, 1, n_pos, n_pos]`` schema raise path
  (separate test rather than parametrized — equivalent coverage).
- [x] Phase 1a ``collapse_posterior_to_position`` NL ``MARGINAL``
  override test: ``test_nl_singleton_marginal_override_equals_one_minus_singletons``
  asserts row sum equals ``1 - P(Local) - P(No-Spike)`` against the
  fully-singleton schema.
- [x] Phase 1c SlicePanel design check:
  ``test_singleton_local_drops_local_from_position_curve`` in
  ``tests/analysis/test_posterior_collapse.py`` confirms
  ``collapse_log_likelihood_to_position`` keeps shape ``(n_pos,)``
  and that singleton-state bins (``Local`` + ``No-Spike``) don't
  contribute to the output.

### Verification (Track 0)

- [x] `uv run pytest -k "posterior_model"` passes (13 tests).
- [x] Singleton-Local fixture covered by 4 ``@pytest.mark.slow``
  tests (3 in test_posterior_collapse + 1 in test_place_fields).
- [x] Phase 1a + 1b + 1c stack is import-clean without `[viewer]`
  deps installed (CI gate enforces — see lint test).

---

## Milestone 2 — One Qt panel + viewer harness

First Qt rendering. `[viewer]` extra required from this point.

### Qt panel

- [x] `panels/qt/__init__.py`.
- [x] `panels/qt/_mixins.py`: `EventOverlayMixin` providing default
  `set_event_overlays` via `pg.InfiniteLine` (points) and
  `pg.LinearRegionItem` (intervals). Idempotent (clears prior items).
- [x] `panels/qt/posterior.py`: `QtPosteriorHeatmapPanel(pg.PlotWidget)`
  consuming `PosteriorHeatmapModel` + `EventOverlayMixin`.
- [x] Test: headless smoke test confirms ImageItem receives the
  collapsed array. (Pixel-perfect matplotlib comparison deferred —
  the model output is already covered bit-identically against the
  dataset helper in Phase 1c.)

### Viewer harness

- [x] `viewer/__init__.py`.
- [x] `viewer/core.py`:
  - [x] `ViewerCore` (Qt-free) holding `current_view_state: ViewState`,
    `pinned_event_row`, `active_overlay`, `active_run_name`.
  - [x] Methods: `step_left()`, `step_right()`, `set_t_center()`,
    `request_load()`, `set_active_run()`, `next_event()`,
    `prev_event()`.
  - [x] Stale-result rejection via `request_id` comparison.
- [x] `viewer/backend.py`: `BackendAdapter` Protocol with
  `schedule_window_load(state, on_done)` and `post_to_ui_thread(fn)`.
- [x] `viewer/qt.py`:
  - [x] `QApplication` creation (only place in the codebase).
  - [x] `QtBackendAdapter` implementing `BackendAdapter` via
    `QThreadPool` + `_LoadSignals` bridge object (statespacecheck
    pattern).
  - [x] `QtViewer(QMainWindow)` with one panel + center-time slider.
    Binds ← / → keys to `core.step_*`.
  - [x] Exports `launch_qt(...)` for `app.py` to import.

### CLI / notebook entry

- [x] `app.py`: argparse + lazy dispatch. `--run
  default:results.nc:model.pkl:spikes.npz:position.parquet`. Document
  CLI input file formats (results.nc / model.pkl / spikes.npz /
  position.parquet — schema in `--help`). Exposes `main(argv=None)
  -> int` for the `__main__.py` shim and tests.
- [x] `__main__.py`: required for `python -m
  non_local_detector.visualization.interactive` to dispatch to
  `app.main()`.
- [x] Notebook callable: `launch(bundle)` in `__init__.py` (lazy
  import of `viewer.qt.launch_qt`).

### Verification (Track 0)

- [x] `python -m non_local_detector.visualization.interactive
  --run default:<paths>` end-to-end CLI smoke test:
  ``test_python_m_interactive_launches_against_simulated_bundle``
  serializes the Track 0 NL bundle to ``tmp_path/{results.nc,
  model.pkl, spikes.npz, position.parquet}``, runs ``app.main(argv)``
  with ``block=False``, asserts exit code 0. Plus two CLI parser
  tests (malformed --run, missing --run).
- [x] Headless `QtViewer` constructs against the simulated
  `multi_run_bundles` fixture; `launch_qt(block=False)` and
  `set_active_run` round-trip without errors.
- [x] Default install (no `[viewer]`) still imports the
  GUI-toolkit-free package shell cleanly — CI gate enforces.

---

## Milestone 3 — Remaining left-column panels + generic series panels

### Built-in left-column view-models + Qt panels

- [x] `view_models/likelihood.py` + `panels/qt/likelihood.py`:
  `LikelihoodHeatmapModel` per-row collapse via
  `collapse_log_likelihood_to_position`; output `(n_visible, n_pos)`.
- [x] `view_models/state_prob.py` + `panels/qt/state_prob.py`:
  `StateProbabilityModel` (multi-line over states).
- [x] `view_models/raster.py` + `panels/qt/raster.py`: `RasterModel`
  + `QtRasterPanel` with place-field-peak sort. Non-local shaded
  bar deferred to Milestone 4 (depends on per-bin state
  probabilities the slice panel surfaces).
- [x] X-axes linked across all left-column panels.

### Generic series panels

- [x] `view_models/series.py`: add `LineSeriesModel` (with
  `fill_below: bool` and `thresholds: list[float]` options),
  `MultiLineSeriesModel`, `ScatterSeriesModel`, `IntervalSeriesModel`
  alongside the existing `MetricSpec`.
- [x] `panels/qt/series.py`: `LineSeriesPanel`,
  `MultiLineSeriesPanel`, `ScatterSeriesPanel` (default
  `click_recenters=True`), `IntervalSeriesPanel`.

### Event overlays (panel-side rendering + viewer dispatch)

- [x] `TimeAxisPanel` Protocol already declares
  `set_event_overlays(overlays)` (added in Phase 1c).
- [x] All four generic panels and four built-in left-column panels
  inherit `EventOverlayMixin`.
- [x] `viewer/core.py` overlay rendering loop: `on_overlays_changed`
  subscribe API + `_dispatch_overlays` push the visible set to
  every registered panel.
- [x] Navigator: `next_event()`/`prev_event()` per overlay kind
  (already in Phase 2 ViewerCore — points + intervals semantics).
- [x] `viewer/qt.py`: `N` / `Shift+N` keyboard shortcuts wired.
  Overlay selector dropdown + per-overlay visibility checkboxes
  deferred to Milestone 6 (UI polish — the underlying
  `set_overlay_visibility` and `set_active_overlay` core APIs
  are already in place).

### Viewer extras

- [x] `extra_panels: list[TimeAxisPanel]` kwarg on `QtViewer` /
  `launch_qt`.
- [x] Auto-construction of panels from `bundle.extra_metrics` when
  `extra_panels` is unset (`MetricSpec.line/scatter/intervals` →
  matching panel; `pd.Series` → `LineSeriesPanel`).
- [x] Wheel-over-time-axis = window-width scrub.
- [x] Keyboard: `[`, `]`, `R`, Shift+←/→.

### Tests

- [x] LikelihoodHeatmapModel non-rectangular schema test (NL
  `nl_loglik`): output shape `(n_visible, n_pos)`; each row
  bit-identical to `collapse_log_likelihood_to_position(...)` via
  `np.testing.assert_allclose(..., atol=1e-14, equal_nan=True)`;
  all-NaN window slice → all-zero row.
- [x] LikelihoodHeatmapModel rectangular (CF `cf_loglik`): same
  shape, same code path.
- [x] Generic series panel auto-render: `bundle.extra_metrics` →
  panel set (pd.Series → line panel; MetricSpec.scatter/intervals
  → matching panels). Click-to-recenter wired via
  `ScatterSeriesPanel`.
- [ ] `LineSeriesPanel(fill_below=True, thresholds=[2.0])` renders
  filled area + dashed threshold visually verified (deferred to a
  Milestone 5 visual diff; the construction path is exercised by
  the auto-render test).
- [ ] `MultiLineSeriesPanel` rendering test (deferred — Milestone 5
  visual diff against `plot_detector` covers it).
- [x] Event overlays: ViewerCore dispatch + visibility filtering +
  swap behavior covered in `test_viewer_core.py`. Per-panel
  marker rendering covered in `test_qt_viewer.py`. End-to-end
  multi-overlay visual check deferred to Milestone 5.

### Verification (Track 0)

- [ ] Full left-column stack rendered against simulated `nl_bundle`;
  visual diff against [`plot_non_local_model`](src/non_local_detector/visualization/static.py#L98)
  for event window centered on `event_times[0]` (deferred to
  Milestone 5 — needs end-to-end render comparison against the
  static plot).

---

## Milestone 4 — Right-column SlicePanel

### Track A devtool (prerequisite for real-data verification)

The plan's "Track A — statespacecheck real data" section (used by
Phase 4 SlicePanel real-data check and Phase 6 model-swap test)
requires a CLI-compatible bundle directory built from
statespacecheck's intermediates. Build the devtool here so Phase 4
and Phase 6 verifications can run.

- [ ] Add
  `src/non_local_detector/visualization/interactive/devtools/__init__.py`
  and `devtools/bundle_from_statespacecheck.py` implementing the
  CLI shown below:

  ```text
  python -m non_local_detector.visualization.interactive.devtools \
      bundle-from-statespacecheck-cache \
      --cache-dir <statespacecheck cache_dir>/ \
      --intermediates-dir <path/to/intermediates>/ \
      --model continuous|contfrag \
      --out <bundles_root>/<model_name>/
  ```

- [ ] Use upstream statespacecheck's `model_paths(intermediates_dir,
  model)` to resolve filenames (`continuous → cont_results.nc +
  cont_model.pkl`, `contfrag → cont_frag_results.nc +
  cont_frag_model.pkl`).
- [ ] Loads `results` from `model_paths.results_nc`; validates
  `acausal_posterior` + `acausal_state_probabilities` present.
- [ ] Loads detector via `joblib.load(model_paths.model_pkl)`.
- [ ] Reads shared `<cache-dir>/figure04_meta.npz` for time + linear
  position.
- [ ] Reads shared `<cache-dir>/figure04_spike_times.npy` for
  per-cell spike times.
- [ ] Cross-checks `<cache-dir>/figure04_<model>_place_fields.npz`
  against
  `extract_state_aligned_place_fields(detector)[:, detector.is_track_interior_state_bins_]`
  (interior-only).
- [ ] Writes a CLI-compatible bundle directory at `--out` with
  `results.nc`, `model.pkl`, `spikes.npz`, `position.parquet` (the
  four files the viewer's `--run-from-dir` flag reads).
- [ ] Optional flags: `--results-nc <path>`, `--model-pkl <path>`,
  `--results-from-zarr` (with explicit `acausal_posterior` +
  `acausal_state_probabilities` validation when substituting the
  Zarr store).
- [ ] Document the build command for both bundles
  (`continuous/`, `contfrag/`) under one parent dir referenced by
  `NLD_REAL_DATA_BUNDLE_DIR`.

### `view_models/slice.py`

- [ ] `SliceModel.update_for_index(t_idx)` returning `BinPayload`.
  - [ ] Schema-aware state-bin slicing using `detector.state_ind_`
    + `bin_sizes_` (NOT blind reshape — non-rectangular detectors
    raise on `(n_states, n_pos)` reshape).
  - [ ] Top likelihood curve via
    `collapse_log_likelihood_to_position(...)` when `log_likelihood`
    available; fallback to collapsed posterior via
    `collapse_posterior_to_position(..., reduction)` when missing,
    with title-bar note.
  - [ ] Predictive overlay via
    `collapse_posterior_to_position(predictive_row, detector,
    reduction)` when `predictive_posterior` available; hide when
    missing.
  - [ ] Per-cell rows: list of `CellSlice` dataclasses. Each
    carries `cell_id`, `place_field_norm` (peak-normalized row of
    `extract_per_cell_place_fields(detector)`), optional event
    metrics from `bundle.events`. Rendered for cells that spiked in
    the cursor's bin.
- [ ] Per-tick path uses in-RAM ring buffer (statespacecheck pattern).

### `panels/qt/slice.py`

- [ ] `QtSlicePanel(QWidget)`:
  - [ ] Population top plot.
  - [ ] Pre-allocated per-cell row pool (`MAX_PER_CELL_PLOTS = 6`).
  - [ ] Pinning logic: click raster spike → cell stays in slice
    panel until `Esc` or click-pinned-again.
  - [ ] `(+K more)` truncation indicator when > 6 cells fired.

### Tests

- [ ] Track 0 NL `nl_all` bundle: scrub to `event_times[0]`; per-cell
  rows show simulation-injected cells (ground truth from
  `make_simulated_data`); top curve bit-identical to
  `collapse_log_likelihood_to_position(...)` via
  `np.testing.assert_allclose(..., atol=1e-14, equal_nan=True)`.
- [ ] Predictive overlay bit-identical to
  `collapse_posterior_to_position(predictive_row, ...,
  CONDITIONAL_NON_LOCAL)` (with `equal_nan=True`).
- [ ] Track 0 NL `nl_default` (no `log_likelihood`,
  `predictive_posterior`): top curve falls back to collapsed
  posterior; predictive overlay hides; per-cell rows render
  normally.
- [ ] Track A optional: against `$NLD_REAL_DATA_BUNDLE_DIR/contfrag/`
  (skip if env var unset), per-cell rows populate at known
  confidently-non-local time bin.

---

## Milestone 5 — Plugin contract + downstream consumer

### Documentation

- [ ] Document `TimeAxisPanel` / `BinSyncedPanel` ABCs in the
  package README.

### `continuum-swr-replay` (separate repo, separate PR)

- [ ] Add `interactive_panels/view_models.py` +
  `interactive_panels/qt.py` for project-specific panels:
  - [ ] `SWRTracePanel` = `LineSeriesPanel(thresholds=[2.0])` +
    `EventOverlay.intervals(data["ripple_times"])`.
  - [ ] `MUARatePanel` = same pattern with `data["hse_times"]`.
  - [ ] `ThetaLFPPanel` = `LineSeriesPanel`.
  - [ ] `SpeedPanel` = `LineSeriesPanel(fill_below=True)`.
  - [ ] `HeadDirectionPanel` = `ScatterSeriesPanel(y_range=(-π, π))`.
- [ ] `RunBundle.from_continuum_data(data, results, detector,
  brain_area="HPC")` adapter.
- [ ] Convenience wrapper:
  `continuum_swr_replay.visualization.launch_detector_viewer(data,
  bundle, ...)`.
- [ ] Document migration path away from `plot_detector` (and
  explicit note: `plot_peri_event_probability` stays as static
  figure — event-aligned, not session-time-aligned).

### Verification (Track B)

- [ ] Continuum integration test: `load_data(small NWB slice)` →
  fit `SortedSpikesDecoder` → `RunBundle.from_continuum_data(...)` →
  launch viewer headlessly via `pytest-qt qtbot.waitExposed`.
- [ ] Visual diff against static `plot_detector` for same time slice.

---

## Milestone 6 — Polish, model-swap UI + ship v1

### Auto-scroll

- [ ] `Space` toggles play/pause.
- [ ] Speed combo: `0.05x, 0.1x, 0.25x, 0.5x, 1x, 2x, 4x, 8x`.
- [ ] `,` / `.` step speed up/down through preset list.
- [ ] Default speed: `0.05x` (statespacecheck default).

### Smoothed-overlay support

- [ ] When results contain `acausal_posterior`, expose smoothed
  overlay choice in slice panel's overlay selector.

### Model swap UI

- [ ] "Model" dropdown in controls bar (visible only when
  `len(runs) > 1`).
- [ ] `M`-key cycles to next run.
- [ ] Swap preserves: `current_view_state` (`t_center`, `t_width`,
  `load_acausal`), `pinned_event_row`, active overlay name.
- [ ] Swap mutates: `active_run_name` only.
- [ ] On swap, re-bind:
  - [ ] StateProbabilityPanel line set to new schema.
  - [ ] PosteriorHeatmapPanel reduction strategy via
    `select_reduction(state_names, bin_sizes_)`.
  - [ ] SlicePanel's `collapse_log_likelihood_to_position` to new
    `state_ind_` / `bin_sizes_`.
  - [ ] RasterPanel place-field-peak sort to new detector.

### Documentation

- [ ] README with screencast covering single-run and model-swap
  flows.
- [ ] `non_local_detector/visualization/__init__.py` lazy-exposes
  `launch` from `interactive` sub-package (gated on `[viewer]`).

### Demo notebook

- [ ] `notebooks/interactive_viewer_demo.ipynb` walks through:
  - [ ] Building a `RunBundle` from fitted detector + session data.
  - [ ] `launch(bundle)`.
  - [ ] `predict(return_outputs=...)` snippet for full panel set.
  - [ ] Multi-run `dict[str, RunBundle]` for model comparison.
  - [ ] Three sized affordances for user metrics
    (`bundle.extra_metrics`, generic panel classes, `TimeAxisPanel`
    subclass).
  - [ ] Event overlays via `bundle.event_overlays`.

### Verification

- [ ] (Track 0) Multi-run viewer launches with
  `{"nl": ..., "cf": ..., "nsf": ..., "dec": ...}`. Cycle through
  all four; each swap re-binds StateProbabilityPanel /
  PosteriorHeatmapPanel reduction / SlicePanel collapse helper.
- [ ] (Track 0 overlay alignment) Attach shared SWR overlay +
  per-run model-derived overlay to all four bundles; assert
  schema match; positive case (overlay name persists across
  swaps); three negative cases (kind drift, missing overlay,
  within-bundle duplicate).
- [ ] (Track A optional) If `NLD_REAL_DATA_BUNDLE_DIR` is set,
  multi-run viewer launches against `continuous/` + `contfrag/`
  subdirectories; M-key swaps preserve view state.
- [ ] Demo notebook runs end-to-end on kernel with `[viewer]`
  installed.

### v2 / v3+ items (out of v1 scope — capture as separate issues if discovered)

The plan's "Out (v2 / v3)" and "Deferred-model pathway" sections
enumerate work explicitly deferred from v1 (Panel/holoviews backend,
Zarr data source, MetricPanel, 2D decoder support, video overlay,
clusterless detectors, multi-environment detectors). Track them in
`SCRATCHPAD.md` as they come up; convert to GitHub issues + a new
plan file when v1 ships.

---

## Pre-merge checklist

- [ ] All milestone tasks above checked off.
- [ ] `uv run pytest` green.
- [ ] `uv run ruff check src/` green.
- [ ] `uv run ruff format --check src/` green.
- [ ] CI passes on the PR.
- [ ] **Delete `TASKS.md` and `SCRATCHPAD.md` from the branch
  before merging** (these files are branch-local working notes).
