<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Phase 4 — Architecture hygiene

**Goal:** clean up dead code, encapsulation leaks, and Protocol
strain *before* Phase 5 lands a new data-source class. Behaviour-
preserving refactor; commit cluster on
`interactive-decoder-viewer`.

**Estimated effort:** ~4 hours.

**Skill:** `safe-refactoring` (every task is behaviour-preserving).

**Dependencies:** Phase 3 done and reviewed by user. **Phase 5
depends on this phase.**

> Read [README.md](README.md) first if you haven't already. The
> Working-with-this-plan-in-Claude-Code rules apply to every task
> here.

---

## 4.1 — Remove dead code

- [ ] Remove `ViewState.load_acausal: bool = False` field
  ([view_models/base.py:85](../../../../src/non_local_detector/visualization/interactive/view_models/base.py#L85))
  and `InMemoryDecoderDataSource.load_acausal` method (no production
  callers).
- [ ] Remove `data_source.results` passthrough property
  ([data_source.py:375–376](../../../../src/non_local_detector/visualization/interactive/data_source.py#L375-L376))
  — no production callers, leaks active-run indirection.
- [ ] Remove `BackendAdapter.post_to_ui_thread`
  ([viewer/backend.py:33](../../../../src/non_local_detector/visualization/interactive/viewer/backend.py#L33))
  from the Protocol *or* document as reserved (no production
  callers; tests stub it).

---

## 4.2 — Fix misleading docstrings

- [ ] [data_source.py:267](../../../../src/non_local_detector/visualization/interactive/data_source.py#L267)
  `load_position` docstring claims `set_active_run` clears the
  cache. It doesn't (per-run keying is intentional). Rewrite to
  describe actual behaviour: "keyed by run name — entry for the
  previous run is retained and reused on next visit."

---

## 4.3 — Tighten Protocol (eliminate isinstance leaks)

- [ ] Add `set_required_outputs(set[str]) -> None` to
  [viewer/backend.py:BackendAdapter](../../../../src/non_local_detector/visualization/interactive/viewer/backend.py)
  as a **required** Protocol method (not a default-no-op). A
  default no-op would silently swallow output gating for any
  future backend implementation that forgot to override it —
  exactly the backend-drift the Protocol cleanup is meant to
  prevent. `StubBackend` in tests must gain an explicit
  implementation (record the calls or store the set; either is
  fine).
- [ ] Remove the `isinstance(self._backend, QtBackendAdapter)` guard
  in `_sync_required_outputs_with_panels`
  ([viewer/qt.py:858–859](../../../../src/non_local_detector/visualization/interactive/viewer/qt.py#L858-L859))
  — the Protocol now guarantees the method exists.
- [ ] Promote `_build_payload` → `build_payload` (public) since 9
  tests depend on it as the synchronous-entry contract. Add to
  Protocol so a future backend implementation has a clear contract.

---

## 4.4 — Defensive teardown

- [ ] [viewer/core.py:188](../../../../src/non_local_detector/visualization/interactive/viewer/core.py#L188)
  `off_overlays_changed` uses `list.remove()` which raises
  `ValueError` on a double-unregister. Wrap in
  `try/except ValueError: pass` (or `if cb in list:` guard) so
  teardown tolerates double-calls.

---

## 4.5 — *Deferred*: cross-bundle alignment validation perf

The original draft of this plan suggested memoizing
`_validate_time_grid_alignment`
([data_source.py:168–169](../../../../src/non_local_detector/visualization/interactive/data_source.py#L168-L169))
keyed on `id(bundle)`. **That is unsafe** —
[data_source.py:162](../../../../src/non_local_detector/visualization/interactive/data_source.py#L162)
explicitly notes RunBundles are mutable; an `id()`-keyed cache
would mask post-construction mutation of (e.g.) `bundle.position`
because `id(bundle)` is unchanged but the data is.

Two safe alternatives:

a) Validate once at construction in
   `InMemoryDecoderDataSource.__init__`, then **freeze** an
   immutable snapshot of `time` / overlays / etc. that the data
   source owns going forward. Bundles can mutate but the data
   source's internal state is fixed.

b) Add an explicit `invalidate_alignment()` API the caller invokes
   after mutating a bundle. Defaults to invalidate-on-every-call
   (current behaviour) for safety.

(a) is the right answer if validation cost is actually a hotspot;
(b) is a smaller change but pushes correctness onto callers. Both
are bigger than this phase's "behaviour-preserving hygiene"
charter.

- [ ] **Action: leave `_validate_time_grid_alignment` unchanged in
  Phase 4.** If the per-call cost becomes a measurable hotspot in
  practice (profile a 10-run M-key loop), open a separate refactor
  ticket for option (a). Do not ship `id()` memoization.

---

## Phase 4 done when

- Tasks 4.1–4.4 land; 4.5 stays deferred per its rationale block
- Full sweep green (verify command from
  [README.md](README.md) §Per-phase verification)
- **No behavioural production changes beyond the defensive
  teardown in 4.4.** Test/stub updates *are* expected and allowed:
  4.3 specifically requires `StubBackend` to gain a
  `set_required_outputs` implementation, and `build_payload`
  becoming public will let test files drop the underscored access
  pattern. The diff invariant is "production code paths render
  identical results"; not "no test files touched".

Then **stop and wait for user review** before proceeding to
[Phase 5](phase-5-direct-zarr.md).
