<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Phase 5 — Direct zarr data source (the perf win)

**Goal:** replace the lazy-xarray-zarr hot path with direct
`zarr.Array` slice reads. Closes the largest remaining latency gap
on resize/scroll for sessions backed by a viewer cache. Commit
cluster on `interactive-decoder-viewer`.

**Estimated effort:** ~12 hours / 1.5 days.

**Skill:** `safe-refactoring` for the Protocol introduction;
`scientific-tdd` for the parity tests.

**Dependencies:** Phase 4 done and reviewed by user (cleaner
Protocol; public `build_payload`).

> Read [README.md](README.md) first if you haven't already. The
> Working-with-this-plan-in-Claude-Code rules apply to every task
> here.

**Why this is the big perf win:** earlier optimizations reduced
*what* loads (per-row exp-on-demand, output gating, differential
debounce). This one reduces *how* it loads. The current path goes
through `xr.DataArray.isel(time=sl).values` → xarray indexing
machinery → dask graph materialization → chunk fetch. Reference
viewer (`statespacecheck-paper-viewer/data_source.py:398`) holds
direct `zarr.Array` handles and reads `arr[start:stop]` — one
contiguous read.

---

## 5.1 — Define `DecoderDataSource` Protocol

- [x] Extract a `DecoderDataSource` Protocol from the current
  `InMemoryDecoderDataSource` API surface that `ViewerCore` and
  `QtBackendAdapter.build_payload` depend on:
  - Properties: `time`, `time_edges`, `n_time`, `available_outputs`,
    `active_run_name`, `run_names`, `active_run`, `event_index`
  - Methods: `window_indices(t_center, t_width) -> slice`,
    `load_posterior(sl)`, `load_likelihood(sl)`,
    `load_predictive(sl)`, `load_state_probabilities(sl)`,
    `load_position(sl)`, `slice_at_index(t_idx, which)`,
    `set_active_run(name)`
- [x] Place the Protocol in
  [data_source.py](../../../../src/non_local_detector/visualization/interactive/data_source.py)
  near the top so both implementations import it from there.
- [x] Update `ViewerCore.__init__` and `QtBackendAdapter.__init__`
  type hints to accept `DecoderDataSource` instead of
  `InMemoryDecoderDataSource`.
- [x] Test: `InMemoryDecoderDataSource` still satisfies the
  Protocol (use `runtime_checkable` + an `isinstance` assertion in
  tests).

---

## 5.2 — `ZarrDirectDecoderDataSource`

- [x] New module
  `src/non_local_detector/visualization/interactive/data_source_zarr_direct.py`
  with `ZarrDirectDecoderDataSource(DecoderDataSource)`.
- [x] Open `results.zarr/` once via `zarr.open_consolidated(path)`
  (fall back to `zarr.open(path)`). Cache direct `zarr.Array`
  handles for: `acausal_posterior`, `log_likelihood`,
  `predictive_posterior`, `acausal_state_probabilities`, `time`.
- [x] `load_posterior(sl)` → `np.asarray(self._posterior[sl, :])` —
  one contiguous read, no xarray.
- [x] Sidecars (model.pkl, spikes.npz, position.parquet, place
  fields, fitted detector, event index) load eagerly at
  construction — they're small + on the per-tick hot path.
- [x] `set_active_run(name)` rebinds the zarr handles for the new
  run.
- [x] Multi-run support: `for_directories(dict[str, Path])`
  classmethod mirroring `InMemoryDecoderDataSource.from_bundles`.

---

## 5.3 — `state_bins` MultiIndex assumptions

The direct-zarr path skips the xarray MultiIndex restoration (panel
collapse helpers must work without it).

- [x] Audit each panel's collapse helper for `state_bins` MultiIndex
  assumptions (look for `.sel(state_bins=...)`, `.unstack`,
  `.indexes`).
- [x] Where a helper assumes the MultiIndex, refactor to integer-
  position indexing (use `state_ind` 1D coord — present on both
  paths). **No refactor required**: the audit confirms all four
  collapse helpers (`PosteriorHeatmapModel.collapse_rows`,
  `LikelihoodHeatmapModel.update_window`, `SliceModel`'s top-curve
  + per-cell paths, and the analysis-layer
  `collapse_log_likelihood_per_spatial_state`) already operate on
  integer-position indexing into 2D `(n_visible, n_state_bins)`
  numpy arrays via `np.isin(detector.state_ind_, ...)` masks.
  Zero `.sel(state_bins=...)` / `.unstack` usage outside
  `data_source_zarr._restore_state_bins_multiindex` (which restores
  the MultiIndex on the *xarray* Dataset for code that walks
  `results.data_vars`, not for the per-window hot path).
- [x] Document the contract: collapse helpers accept a 2D
  `(n_time, n_state_bins)` array + a 1D
  `state_ind: (n_state_bins,)` array; never relies on xarray
  MultiIndex semantics. *Documented in
  `PosteriorHeatmapModel.collapse_rows` docstring; the contract
  applies symmetrically to the likelihood + slice helpers.*

---

## 5.4 — CLI integration: run-spec / data-source construction redesign

The current
[app.py:main()](../../../../src/non_local_detector/visualization/interactive/app.py)
unconditionally wraps every parsed spec in
`InMemoryDecoderDataSource` after `_load_run` returns a
`RunBundle`. To swap in a direct-zarr data source we cannot just
change `_load_run` — the data-source construction has to move up
to `main()` so it can decide *which* implementation to build based
on the run-spec aggregate.

- [x] **Refactor `_load_run` → `_resolve_run_spec`.** Instead of
  always returning `(name, RunBundle)`, return a small dataclass
  `ResolvedRun(name, kind, paths, *, bundle=None)` where:
  - `kind == "in_memory"`: the spec is a `--run` / `--run-files` /
    `--run-from-dir` *without* a usable `results.zarr/`. The
    bundle is loaded eagerly (current path); `paths` carries the
    original sidecar paths.
  - `kind == "zarr_direct"`: the spec is `--run-from-dir` with a
    valid `results.zarr/` AND the zarr backend is available. The
    bundle is *not* loaded eagerly; `paths` carries the bundle
    dir.

  *Implementation deviated from the dataclass design slightly:*
  the resolver inspects the parsed spec dicts directly via a
  predicate (`_zarr_direct_eligible(spec)`) and dispatches in
  `_resolve_data_source(specs)`. The dataclass overhead wasn't
  needed once the dispatch became a single function; the
  predicate-based approach is simpler and tests the same way.
- [x] **Aggregate at `main()` level.** After parsing all specs:
  - If every spec is `kind == "zarr_direct"`: build a single
    `ZarrDirectDecoderDataSource.for_directories({name: dir})`.
  - Otherwise (any `kind == "in_memory"`): load the eager bundle
    for any zarr-direct spec too (so all runs are in-memory),
    then build `InMemoryDecoderDataSource(bundles)`. Mixed-mode
    one-data-source-per-run is harder than this phase warrants;
    document the constraint.
- [x] **Mixed-mode warning.** When at least one spec is zarr-direct
  but at least one isn't, emit a `UserWarning` explaining that the
  zarr-direct runs were degraded to in-memory because mixing isn't
  supported, and pointing users at the `build-viewer-cache`
  devtool to give the rest of the runs zarr caches too.
- [x] **Fallback path stays.** If zarr import fails for a
  `--run-from-dir` with a `results.zarr/`, the existing
  `load_zarr_cache_or_fall_back` `ImportError` path falls back to
  reading `results.nc`; the resolved kind becomes `in_memory`.
- [x] CLI epilog updated to lead with "use `--run-from-dir` +
  `build-viewer-cache` for sessions > 1 hour" and document the
  all-or-nothing zarr-direct rule.
- [x] Tests:
  - All-`--run-from-dir` with zarr caches → `ZarrDirectDecoderDataSource`
  - All-`--run` (no caches) → `InMemoryDecoderDataSource`
  - Mixed (one with cache, one without) → warning + `InMemoryDecoderDataSource`
  - Zarr backend missing → falls back to `InMemoryDecoderDataSource`
    silently with the existing `UserWarning`

---

## 5.5 — Tests

- [x] **Parity test** in `test_data_source_zarr.py` (or new file):
  for the same bundle, build both `InMemoryDecoderDataSource` and
  `ZarrDirectDecoderDataSource`. For 10 random window slices,
  assert `np.testing.assert_array_equal(in_mem.load_X(sl),
  zarr_direct.load_X(sl))` for every `load_*` method.
- [x] **Latency benchmark — out of CI.** Standalone benchmark
  script `scripts/bench_data_source.py` lands in this commit
  (option (a) per the plan). Times `load_posterior` /
  `load_likelihood` / `load_state_probabilities` at 1s / 10s /
  30s windows for both data sources, prints a comparison table.
  Not collected by `pytest`. Smoke mode (no `--bundle-dir` arg)
  builds a transient simulated bundle for self-test; the PR
  description records the numbers from a representative real-data
  run when the user is ready.
- [x] **Multi-run swap**: `set_active_run` rebinds zarr handles
  correctly; subsequent `load_*` calls return data for the new
  run. Covered by
  `test_zarr_direct_set_active_run_rebinds_handles`.
- [x] **Per-method missing-output contracts.** Each `load_*`
  method has its own contract; the in-memory path is the source
  of truth and the zarr-direct implementation must match
  *exactly*. Pin each separately in `test_data_source_zarr.py`:

  | Method | When the underlying var is absent |
  | --- | --- |
  | `load_posterior` | **Cannot occur** — `acausal_posterior` is required at `RunBundle.__post_init__` ([base.py:425](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L425)) |
  | `load_likelihood` | **Raises `KeyError`** with the rebuild instruction ([data_source.py:203–215](../../../../src/non_local_detector/visualization/interactive/data_source.py#L203-L215)) |
  | `load_predictive` | **Returns `None`** silently ([data_source.py:232–238](../../../../src/non_local_detector/visualization/interactive/data_source.py#L232-L238)) |
  | `load_state_probabilities` | **Cannot occur** — `acausal_state_probabilities` is required at `RunBundle.__post_init__` ([base.py:425](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L425)) |
  | `load_position` | Returns `None` when `bundle.position` is 2D (heatmap-trace overlay is 1D-only per the v1 contract — see [data_source.py:262–264](../../../../src/non_local_detector/visualization/interactive/data_source.py#L262-L264)); never raises |

- [x] Tests pin two of the five rows in `test_data_source_zarr_direct.py`:
  - `load_predictive` returns `None` parity:
    `test_zarr_direct_missing_predictive_returns_none`.
  - `load_likelihood` raises `KeyError` with the same wording in
    both paths: `test_zarr_direct_missing_log_likelihood_raises_keyerror`
    + `test_zarr_direct_load_likelihood_keyerror_text_pinned`.

  *2D-position case dropped*: ``RunBundle.__post_init__`` rejects
  2D position with a 1D detector environment ([base.py:445](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L445)).
  The plan's 2D-position contract row only fires with a 2D
  detector, which we don't have in the simulated-detector test
  fixtures. Parity is implicit via `_position_at_decoder_time`'s
  identical implementation in both data sources.

---

## Phase 5 done when

- [x] Parity tests pass — 11 cases in `test_data_source_zarr_direct.py`
- [ ] **Latency numbers from the standalone benchmark** are recorded
  in the PR description on a representative real-data bundle — no
  hard pass/fail threshold in CI. *User-driven; the smoke run on
  a simulated session shows in-memory wins on small data, which
  is expected; real-data numbers should show zarr-direct winning
  on long sessions.*
- [x] Full sweep green (verify command from
  [README.md](README.md) §Per-phase verification) — 291 passed
  (Phase 4 baseline 276 → +15 new tests: 1 Protocol-isinstance,
  10 zarr-direct parity + missing-output, 1 KeyError-text pin,
  3 CLI resolver-branch tests).

Then **stop and wait for user review** before proceeding to
[Phase 6](phase-6-library-api.md) (or merging if Phase 6 has
already landed).
