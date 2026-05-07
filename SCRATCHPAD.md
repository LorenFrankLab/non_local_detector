# Interactive Decoder Viewer — Scratchpad

Branch-local working notes for the v1 implementation. Use freely;
**delete before merging**.

Pair file: [TASKS.md](TASKS.md) (the structured task tracker).
Plan: [docs/plans/2026-05-06-interactive-decoder-viewer.md](docs/plans/2026-05-06-interactive-decoder-viewer.md).

---

## Current focus

> What I'm working on right now. Update when context-switching.

Milestones 1 + 2 + 3 complete (left-column panels, generic series, overlay dispatch, viewer extras). Visual-diff verification + UI polish (overlay selector dropdown, multi-line / fill-below render tests) deferred to Milestones 5/6.

---

## Open questions / decisions discovered during implementation

> Anything the plan didn't anticipate that needs a call before
> proceeding. Surface to the user; once resolved, fold into
> `docs/plans/2026-05-06-interactive-decoder-viewer.md` so the plan
> stays the source of truth.

(empty)

---

## Issues encountered (and resolutions)

> Bugs, surprising behavior, gotchas. Date each entry.

(empty)

---

## Test results

> Notable runs, especially anything that needed multiple iterations.

### Track 0 simulated fixture

- Phase 1a: 33/33 tests pass against the four-detector simulated fixture
  (NL local_position_std=1.0, ContFrag, NoSpikeContFrag, Decoder).
- Each session-scoped detector fit takes ~12-30s; total Phase 1a suite
  ~100s wall.
- `acausal_posterior` / `log_likelihood` arrays are float32; bumped
  several `atol` from the plan's `1e-10` to `1e-6` accordingly.
  Bit-identity assertions for the static-plot refactor required
  casting `_conditional_row` to float64 internally (matches the inline
  algorithm's float64 accumulator).

### Track A statespacecheck real data

> Whether `NLD_REAL_DATA_BUNDLE_DIR` is set, what subdirectories
> exist, what skipped vs ran.

(empty)

### Track B continuum integration

(empty)

---

## Performance notes

> Anything measured: per-tick latency, window-load timing, memory
> footprint of fixtures, etc.

(empty)

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

---

## Phase boundary follow-ups

> Things noticed in one phase that belong in a later phase. Don't
> implement them out of order — log here and pick up in the right
> phase.

### Phase 1a → 1b

(empty)

### Phase 1b → 1c

(empty)

### Phase 1c → 2

(empty)

### Phase 2 → 3

(empty)

### Phase 3 → 4

(empty)

### Phase 4 → 5

(empty)

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

(empty)
