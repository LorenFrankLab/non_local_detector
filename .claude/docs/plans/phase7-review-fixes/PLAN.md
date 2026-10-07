# Checkpointed-Performance Review Fixes Implementation Plan

**Status:** Phases 1a, 3 and 4 done; phase 2 implemented and CPU-validated (`_SMALL_SERIAL_SPIKES = None`, `_serial_row_sum` kept for the benchmark). Waiting for the A100/x86 host: phase 1b, phase 2 GPU baseline (run `scripts/benchmark_spike_row_reduction.py` at 5bf46a29 and at the phase 2 commit) and GPU determinism.

This plan fixes the problems found when reviewing `feat/phase7-performance`
against `main`, before that branch merges.

- It removes two changes that only reproduced a less accurate test
  reference's rounding: the ordered sorted-spike emission, and the
  mixed-precision Gaussian barrier, subject to x86 confirmation.
- It replaces the serial GPU spike-row loop with a deterministic parallel
  reduction, so checkpointed replay stays bit-exact without serializing over
  spikes.
- It fixes the confirmed correctness, memory and efficiency findings in
  checkpointed inference, the result store, host count matrices and lazy
  graph distances.
- It brings tests and scripts in line with project conventions.

Each phase is one PR into `feat/phase7-performance`.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need a per-component design?** [designs.md](designs.md).
3. **Need broader scope / risks / open questions / deferred work?** [overview.md](overview.md).
4. **Need the measurements behind a decision, or how to rerun them?** [appendix.md](appendix.md) and [evidence/](evidence/).

## Files

- [overview.md](overview.md) — integration points, goals and non-goals (including deferred finding 9), metrics, risks, open questions (two gates)
- [designs.md](designs.md) — D1 deterministic row reduction, D2 graph-distance cache, D3 result-store lookup, D4 accuracy-relative sorted test
- Phases (each ships as a separable PR):
  - [phase-1a-sorted-matrix-emission.md](phase-1a-sorted-matrix-emission.md) — restore the 73813467 matrix emission; gated on tolerance approval
  - [phase-1b-gaussian-barrier-scope.md](phase-1b-gaussian-barrier-scope.md) — keep the singleton barrier, drop the kernel-sized mixed-precision one; gated on x86 evidence
  - [phase-2-deterministic-spike-reductions.md](phase-2-deterministic-spike-reductions.md) — segmented-scan reduction replaces the serial loop; A100 baseline first
  - [phase-3-checkpointed-inference-fixes.md](phase-3-checkpointed-inference-fixes.md) — evidence-sum error masking, operator device transfer, result-store scans
  - [phase-4-memory-bounds.md](phase-4-memory-bounds.md) — row-blocked non-local counts; lazy-distance row cache and caller-owned outputs (after 1a)
  - [phase-5-housekeeping.md](phase-5-housekeeping.md) — validation order, xarray floor, script defaults, shared threshold, renames, markers (after 1a and 4)
- [appendix.md](appendix.md) — evidence E1–E6 with numbers and reproduction commands
- [evidence/](evidence/) — the investigation scripts and pytest plugins

## Review-finding index

| Code-review finding | Phase |
| --- | --- |
| 1. GPU serial `fori_loop` in `sum_spikes_into_rows` | 2 |
| 2. Transition operator re-transferred per jitted call | 3 |
| 3. Ordered loop replaces scatter in clusterless diffusion/GMM | 2 |
| 4. Non-local sorted rows × neurons host counts | 4 |
| 5. Lazy distances: no cache; budget error on long recordings | 4 |
| 6. `_sum_evidence` swallows unrelated `OverflowError` | 3 |
| 7. `from_pandas_multiindex` below the xarray floor | 5 |
| 8. macOS-only benchmark output path | 5 |
| 9. Streamed mark kernel recomputed per spatial tile | deferred (comment fix in 5) |
| 10. Result-store O(n_chunks × n_rows) reads, O(n²) writes | 3 |
| 11. Mixed-precision barrier materializes kernel-sized buffer | 1b |
| 12. Validation after refit; `True` accepted as budget | 5 |
| 13. Duplicated 256 threshold and private import | 5 (constant and import only) |
| 14. Milestone names in tests and scripts | 5 |
| 15. Missing test markers | 5 |
| Accumulation-order follow-up: ordered sorted emission | 1a |
