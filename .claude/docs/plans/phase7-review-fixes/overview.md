# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

All references are at `f2c3e2b9` on `feat/phase7-performance`.

- `src/non_local_detector/likelihoods/common.py:341-537`: sorted non-local
  emission (`_ordered_poisson_count_block`, `_poisson_full_count_log_likelihood`,
  `_poisson_packed_log_likelihood`, `_poisson_nonlocal_log_likelihood`).
  Restored to the 73813467 matrix version in phase 1a.
- `src/non_local_detector/likelihoods/common.py:761-800`:
  `_ordered_spike_row_add` / `_ordered_spike_row_sum`. Removed in phase 2.
- `src/non_local_detector/likelihoods/common.py:803-853`:
  `sum_spikes_into_rows`. Its backend dispatch moves to `deterministic_row_sum`
  in phase 2. The signature and host-side validation are unchanged.
- `src/non_local_detector/likelihoods/common.py:856-887`: `log_gaussian_pdf`
  barrier. Narrowed in phase 1b.
- `src/non_local_detector/likelihoods/clusterless_diffusion.py:699-750,826-875`,
  `clusterless_gmm.py:110-130`, `streamed_kde.py:380-405`: serial-loop callers.
  Phase 2.
- `src/non_local_detector/likelihoods/sorted_spikes_kde.py:489-504`,
  `sorted_spikes_glm.py:676-688`: non-local count matrix. Call sites restored
  in phase 1a, row-blocked in phase 4.
- `src/non_local_detector/checkpointed_inference.py:188-223`
  (`_sum_evidence`) and `:371-374` (operator handoff): phase 3. The digest check
  at `:510-513` is unchanged.
- `src/non_local_detector/result_store.py:85-109,173-215`: phase 3.
- `src/non_local_detector/graph_distances.py:37-195`,
  `environment.py:1031-1038`: phase 4.
- `src/non_local_detector/models/base.py:1777-1790,2188-2280,200,3628-3638`:
  phase 5.
- Unchanged (verified genuine fixes, see [appendix E5](appendix.md#e5-classification-of-the-six-ci-failures)):
  structured Gaussian scale caps `transition_operators.py:175-210`; traced
  query-tile padding `streamed_kde.py:56-130,152-163`; stable checkpointed
  evidence accumulation; the replay digest check.

## Scope and dependency policy

### Goals

- Remove the 79156376 changes that only reproduce a less accurate reference's
  rounding, and keep the ones that fix real defects.
- Keep checkpointed replay deterministic on GPU without serializing reductions
  over spikes.
- Fix the confirmed review findings: correctness (overflow masking, validation
  order, xarray floor), memory (rows × neurons counts, lazy-distance budget and
  recomputation, mixed-precision kernel buffer) and efficiency (operator
  transfers, result-store scans).
- Bring tests and scripts in line with project conventions (markers, no
  milestone names).

### Non-Goals

- Changing the replay contract. The digest check stays exact; tolerant replay
  is out of scope.
- Restructuring `streamed_kde._joint_core` loop order (finding 9). The
  recomputation is confirmed ([appendix E4](appendix.md#e4-streamed-joint-core-recomputes-the-mark-kernel)),
  but the fix trades it against a `(n_encoding, decoding_tile)` mark-kernel
  cache, and no timing exists yet. **Revisit when** a profile of streamed
  row-returning prediction at production sizes shows the mark-kernel `exp`
  taking more than 20% of `_joint_core` time. Phase 5 only corrects the
  comment.
- Unifying the local sorted/no-spike `>256`-row per-neuron paths with the
  compiled paths. That would change the numerics of the default dense
  prediction and its snapshots. Phase 5 only deduplicates the constant and the
  private import.
- Editing frozen validation records under `docs/performance_artifacts/` or
  re-running hour-scale qualification.
- Rewording "Phase 7" prose inside `docs/performance_validation.md`. Phase 5
  updates only links to renamed files.

### Dependency policy

- Phase 5 raises the xarray floor to `>=2023.8`. Both result-labelling paths
  already require `xr.Coordinates`
  ([appendix E6](appendix.md#e6-review-findings-confirmed-by-code-reading)).
  No other dependency changes.

## Metrics

- **Numerics.** Golden and snapshot tests pass unchanged. Accuracy assertions
  compare against float64 oracles built from the same float32 inputs; where a
  test asserted agreement with a less accurate reference, the replacement
  asserts "no worse than the reference relative to float64"
  ([designs D4](designs.md#d4-accuracy-relative-sorted-emission-test)).
- **Determinism.** 50 repeated evaluations give one SHA per case for the eight
  clusterless local/non-local cases on A100, and on CPU.
- **Performance** (each measured before and after on the same machine):
  - sorted sparse/burst emission returns to the matrix timings in
    `docs/performance_validation.md:303-311`;
  - the GPU spike-row reduction is no slower than today at checkpoint-sized
    chunks and faster at ≥2k spikes;
  - mixed-precision `_log_kernel_matrix` temp bytes drop from kernel size to
    about 0;
  - the result-store full read over 3000 chunks scales linearly.
- **Memory.** Full-grid non-local sorted prediction allocates host counts of at
  most `block_rows × n_neurons`.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A CI-runtime codegen difference reappears after reverting the ordering (x86 cannot be compiled locally) | Every phase that touches numerics runs its affected modules on the CI stack: `python -m pytest` with JAX 0.11.2 locally, plus a CI run on the branch ([appendix](appendix.md#running-on-the-ci-jax-version)). |
| The segmented scan is slower than the serial loop for checkpoint-sized chunks on GPU | Phase 2 measures on A100 before choosing. The dispatcher has a static small-`n` branch ([designs D1](designs.md#d1-deterministic-spike-row-reduction)). |
| Tests that assert sequential rounding as a contract fail after phase 2 | These are listed by name in [phase 2](phase-2-deterministic-spike-reductions.md). They are restated as float64-accuracy plus run-to-run determinism assertions, not deleted. |
| Removing the mixed-precision barrier breaks x86 x64 CI | Phase 1b is gated on x86 evidence. A fallback is pre-specified. |
| Executor runs the wrong JAX and draws false conclusions | `uv run --with ... pytest` silently uses the project `.venv`; always use `python -m pytest` and print `jax.__version__` ([appendix](appendix.md#running-on-the-ci-jax-version)). This already happened once during the investigation. |

## Rollout Strategy

Each phase is one PR into `feat/phase7-performance`. The branch merges to
`main` after all phases land, or after any subset the user chooses. Merging is
not part of this plan. Phases 1a and 2 change rounding of likelihood outputs at
the float32-ulp level relative to the current branch head, but not relative to
the S6 qualification for phase 1a. No public API changes except two:
`LazyGraphDistances(max_cache_bytes=...)` (new keyword) and
`max_dense_transition_bytes=True` now raising.

Order: 1a and 1b first (small, gated), then 2 and 3 in either order, then 4
(depends on 1a), then 5 (depends on 1a for the shared constant).

## Open Questions

1. **Tolerance change for the sorted test (blocks phase 1a).** CLAUDE.md
   requires approval to change numerical tolerances. Proposed replacement
   assertions: [designs D4](designs.md#d4-accuracy-relative-sorted-emission-test).
   Current answer: pending user approval.
2. **How to obtain x86 evidence for phase 1b.** Options: push the branch and
   let CI run (outward-facing, so confirm first), or run on a Linux x86 host
   the user names. Current answer: pending.
3. **Default `max_cache_bytes` for lazy graph distances.** Current answer:
   64 MiB ([designs D2](designs.md#d2-graph-distance-row-cache-and-caller-owned-outputs)).
   Revisit if the phase 4 replay benchmark shows cache misses at 1 cm grids.
4. **xarray floor bump vs guard.** Current answer: bump to `>=2023.8`, because
   a guard cannot work when `xr.Coordinates` does not exist.

## Estimated Effort

| Phase | Production LOC (approx.) | Test LOC (approx.) |
| --- | --- | --- |
| 1a | −200 / +90 (restore matrix, delete packed path) | ~60 |
| 1b | ~10 | ~40 |
| 2 | +80 / −45, plus call-site edits in 4 files | ~120 |
| 3 | ~60 | ~80 |
| 4 | ~120 | ~100 |
| 5 | ~60, plus file renames | ~30 |
