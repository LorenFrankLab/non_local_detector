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

- [ ] New module
  `src/non_local_detector/visualization/interactive/data_source_zarr_direct.py`
  with `ZarrDirectDecoderDataSource(DecoderDataSource)`.
- [ ] Open `results.zarr/` once via `zarr.open_consolidated(path)`
  (fall back to `zarr.open(path)`). Cache direct `zarr.Array`
  handles for: `acausal_posterior`, `log_likelihood`,
  `predictive_posterior`, `acausal_state_probabilities`, `time`.
- [ ] `load_posterior(sl)` → `np.asarray(self._posterior[sl, :])` —
  one contiguous read, no xarray.
- [ ] Sidecars (model.pkl, spikes.npz, position.parquet, place
  fields, fitted detector, event index) load eagerly at
  construction — they're small + on the per-tick hot path.
- [ ] `set_active_run(name)` rebinds the zarr handles for the new
  run.
- [ ] Multi-run support: `for_directories(dict[str, Path])`
  classmethod mirroring `InMemoryDecoderDataSource.from_bundles`.

---

## 5.3 — `state_bins` MultiIndex assumptions

The direct-zarr path skips the xarray MultiIndex restoration (panel
collapse helpers must work without it).

- [ ] Audit each panel's collapse helper for `state_bins` MultiIndex
  assumptions (look for `.sel(state_bins=...)`, `.unstack`,
  `.indexes`).
- [ ] Where a helper assumes the MultiIndex, refactor to integer-
  position indexing (use `state_ind` 1D coord — present on both
  paths).
- [ ] Document the contract: collapse helpers accept a 2D
  `(n_time, n_state_bins)` array + a 1D
  `state_ind: (n_state_bins,)` array; never relies on xarray
  MultiIndex semantics.

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

- [ ] **Refactor `_load_run` → `_resolve_run_spec`.** Instead of
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
- [ ] **Aggregate at `main()` level.** After parsing all specs:
  - If every spec is `kind == "zarr_direct"`: build a single
    `ZarrDirectDecoderDataSource.for_directories({name: dir})`.
  - Otherwise (any `kind == "in_memory"`): load the eager bundle
    for any zarr-direct spec too (so all runs are in-memory),
    then build `InMemoryDecoderDataSource(bundles)`. Mixed-mode
    one-data-source-per-run is harder than this phase warrants;
    document the constraint.
- [ ] **Mixed-mode warning.** When at least one spec is zarr-direct
  but at least one isn't, emit a `UserWarning` explaining that the
  zarr-direct runs were degraded to in-memory because mixing isn't
  supported, and pointing users at the `build-viewer-cache`
  devtool to give the rest of the runs zarr caches too.
- [ ] **Fallback path stays.** If zarr import fails for a
  `--run-from-dir` with a `results.zarr/`, the existing
  `load_zarr_cache_or_fall_back` `ImportError` path falls back to
  reading `results.nc`; the resolved kind becomes `in_memory`.
- [ ] CLI epilog updated to lead with "use `--run-from-dir` +
  `build-viewer-cache` for sessions > 1 hour" and document the
  all-or-nothing zarr-direct rule.
- [ ] Tests:
  - All-`--run-from-dir` with zarr caches → `ZarrDirectDecoderDataSource`
  - All-`--run` (no caches) → `InMemoryDecoderDataSource`
  - Mixed (one with cache, one without) → warning + `InMemoryDecoderDataSource`
  - Zarr backend missing → falls back to `InMemoryDecoderDataSource`
    silently with the existing `UserWarning`

---

## 5.5 — Tests

- [ ] **Parity test** in `test_data_source_zarr.py` (or new file):
  for the same bundle, build both `InMemoryDecoderDataSource` and
  `ZarrDirectDecoderDataSource`. For 10 random window slices,
  assert `np.testing.assert_array_equal(in_mem.load_X(sl),
  zarr_direct.load_X(sl))` for every `load_*` method.
- [ ] **Latency benchmark — out of CI.** Do *not* put a hard
  speedup threshold in pytest. Synthetic bundles + cold-cache
  variance + machine-dependent disk speed make any `assert >= 3×`
  too flaky to gate on. Two acceptable shapes:

  - **(a) Standalone benchmark script** in
    `scripts/bench_data_source.py` that times `load_posterior`
    /`load_likelihood`/`load_state_probabilities` for windows of
    1s / 10s / 30s on both data sources. Prints a comparison
    table; not invoked by `pytest`. PR description records the
    numbers from a representative real-data run.
  - **(b) Opt-in pytest** marked `bench` and skipped by default
    via `pytest.mark.skipif("not os.getenv('NLD_BENCH')")`.
    Reports timings without asserting a ratio.

  Either is fine; (a) keeps benchmark code out of the test
  collection path entirely and is preferred. Hard parity on
  values stays in the parity test above — that's the
  correctness gate; the speedup is reported, not asserted.
- [ ] **Multi-run swap**: `set_active_run` rebinds zarr handles
  correctly; subsequent `load_*` calls return data for the new
  run.
- [ ] **Per-method missing-output contracts.** Each `load_*`
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

- [ ] Tests pin three of the five rows (the two "cannot occur"
  rows are guarded at construction by `RunBundle.__post_init__`,
  so they're already covered indirectly — `RunBundle` validates
  and coerces `position` at
  [base.py:433](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L433),
  so `position=None` would fail construction):
  - Build a run with `predictive_posterior` removed from
    `results`; `available_outputs` excludes it; both data sources
    return `None` from `load_predictive(sl)`.
  - Build a run with `log_likelihood` removed; both data sources
    raise `KeyError` from `load_likelihood(sl)` with the same
    message text (regression-pin the rebuild-instruction string).
  - Build a bundle with **2D position** (shape `(n_pos_time, 2)`);
    both data sources return `None` from `load_position(sl)`. If
    a future API decision makes `RunBundle.position` explicitly
    optional, this test grows a third arm for `position=None`;
    until then, 2D is the only legitimate "no 1D trace" case.

---

## Phase 5 done when

- Parity tests pass
- Latency numbers from the standalone benchmark (or opt-in
  `bench`-marked pytest) are **recorded in the PR description** on
  a representative bundle — no hard pass/fail threshold in CI
- Full sweep green (verify command from
  [README.md](README.md) §Per-phase verification)

Then **stop and wait for user review** before proceeding to
[Phase 6](phase-6-library-api.md) (or merging if Phase 6 has
already landed).
