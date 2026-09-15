# Likelihood Defect Remediation

Status: Phase 1 implemented and validated. Phase 0 is implemented, optimized,
and validated in the working branch. Remaining phases have the readiness states
recorded below.

## Summary

An audit of `src/non_local_detector/likelihoods/` and its integration with
`models/base.py` and `core.py` found four defects that silently change decoded
posteriors, plus numerical, validation, and calibration gaps. A follow-up audit
reproduced two additional core HMM defects: normalization loses almost all
probability mass for small positive normalizers, and a maximum likelihood in an
unreachable state causes avoidable underflow and false `-inf` evidence. Every
finding was reproduced with a runnable script before being written down; the
reproduction is quoted in the phase that fixes it.

Headline defects: sorted-spike encoding discards its training / environment /
encoding-group mask (place fields 40–45% low); the clusterless GMM floors a
log-density before forming a ratio (~10^11 likelihood distortion in a realistic
tail); chunked prediction drops every spike falling between two chunks; and
`time` is read inconsistently as bin edges vs. timestamps, leaving a final row
that can never contain a spike and rates calibrated to an unstated assumption.

**Read [shared-contracts.md](shared-contracts.md) first.** Of the three contracts
the phases assume, only **C2 is settled** (with measured evidence). **C1 is
withdrawn** — its policy was shown to be non-monotonic — and **C3 is split**: its
decode vocabulary is settled, its encoding-exposure half is not. No phase
depending on C1 or on encoding exposure can be executed.

**Completed foundation: [phase 0](phase-0-core-hmm.md), core HMM correctness.**
Its correctness fixes and performance revision are implemented on
`fix/core-hmm-conditioning`, with reference tests and measured numerical and
runtime comparisons recorded in the phase document. Full-suite validation:
**1329 passed / 4 skipped**, with golden files and existing tolerances unchanged.
It is independent of C1's likelihood-flooring decision and C3's time/exposure
decisions. Keep the existing phase numbers so links and work on phases 1–8
remain valid.

## Production workload and scope

The remaining performance work targets the user's **180 cm × 180 cm arena,
1 cm or 2 cm spatial bins, and one hour at 2 ms decoding intervals**. This is
1.8 million observations and approximately 16,202 or 64,802 combined hidden-state
bins for the default four-state non-local model. See
[overview.md](overview.md#representative-workload) for assumptions, storage
estimates, and acceptance criteria.

Phase 3 owns correct global event binning and likelihood allocation limited to
the requested rows. Phase 7 now includes checkpointed smoothing, incremental
output storage, structured transition operations, and the existing measured
kernel optimizations. It must demonstrate both numerical parity and feasible
memory use at the representative dimensions; small conditioning benchmarks
alone cannot establish full-session feasibility.

This expansion leaves **Phase 0 and Phase 1 completion criteria unchanged**.
The outstanding likelihood corrections and C1/C3 decisions retain their
dependencies and priority. Production integration of Phase 7 follows the
corrected likelihood and time contracts; independent prototypes can establish
feasibility earlier. Proposed designs are not implementation-ready until their
acceptance criteria have been exercised against the real code.

## Status of this plan — read before executing

Drafts 1 and 2 were each reviewed and each found to contain implementation
blockers, all of the same class: **prescriptive code written into this document
and never executed.** Reproduced examples include a phase-2 acceptance test its
own fix would have failed, a phase-4 guard that did not fix the NaN it targeted,
a `sample_cell_durations` helper that undercounts exposure by `(N-1)/N`, and a
uniformity tolerance that rejects every recording timestamped in Unix seconds.

The `Falsification` sections added in draft 2 helped reviewers catch this; they
did not prevent it, because the author still did not run the snippets.

**Process for the remainder of this work: prototype, then write.** No phase file
should contain an implementation snippet that has not been executed. A phase is
ready when its fix has been prototyped against the real code and its acceptance
criteria have been run and observed to fail on `main`.

Contract sections are exempt only where the claim is itself verified — C1 and C2
below now carry measured evidence rather than reasoning.

### Readiness

| Phase | State |
|---|---|
| 0 | **Implemented, optimized, and validated** on `fix/core-hmm-conditioning` (`c1f7e33`; benchmark script `e79501a`). Full suite: **1329 passed / 4 skipped**; reference tests: **63 passed / 1 skipped** in default float32 and **64 passed** with x64 enabled. Golden files and existing tolerances unchanged; ruff and format pass. Numerical comparisons are bit-identical to the pre-optimization fix; stationary kernels take 6.5–14.3% less time in the measured CPU benchmark. See [phase 0](phase-0-core-hmm.md). Independent of C1/C3. |
| 1 | **Implemented and reviewed** on `fix/sorted-spikes-exposure-mask` (`8c8765f` plus regression coverage). Mask and zero-exposure regressions fail against the relevant pre-fix behavior. Post-review full suite: **1266 passed / 3 skipped**, goldens unchanged; ruff and format pass. |
| 2 | Needs prototyping: aggregate flooring, zero-mass detection, and the newly in-scope log-KDE clamps |
| 3 | **Revised requirements; needs prototyping.** Bin events globally and restrict decoding spikes before density evaluation; cover no-spike states, masks, and position-dependent terms. Validate likelihood-specific memory bounds. Full posterior retention is assigned to Phase 7a; the former >4× total-memory target is withdrawn. |
| 4 | Nearly ready: weight normalizer overflows on large float32 weights; the GLM objective fix promised in the index below is missing |
| 5 | Rewrite against the corrected C2 — the hard-window machinery is deleted, not repaired |
| 6a–6d | Needs prototyping: exposure endpoints, a central edge validator, ULP-aware uniformity, rate-vs-expected-count test semantics, and 6b/6d must be atomic |
| 7 | **Expanded scope; needs prototyping.** 7a: bounded memory smoothing and incremental/compact outputs. 7b: structured forward/backward transitions. 7c: measured likelihood and compilation optimizations. Validate the representative workload, existing numerical tolerances, host/device peak memory, and end-to-end runtime before claiming production support. |
| 8 | Mostly ready; 9 call sites in `likelihoods/` source, not 13; reconsider whether to drop `indices_are_sorted` rather than reject previously-accepted input |

## Reading order

1. [shared-contracts.md](shared-contracts.md) — C1 degeneracy policy, C2 exposure
   ownership, C3 time vocabulary.
2. [overview.md](overview.md) — goals, non-goals, architecture map, deferred items.
3. [phase-0-core-hmm.md](phase-0-core-hmm.md), then the remaining phase files in
   order. A later phase's readiness does not resolve an earlier phase's blockers.

## Phase index

| Phase | File | Ships | Goldens |
|---|---|---|---|
| 0 | [phase-0-core-hmm.md](phase-0-core-hmm.md) | Stable Bayesian conditioning and posterior normalization in both core paths | Measure; defect cases must change, well-conditioned cases retain existing tolerances |
| 1 | [phase-1-exposure-mask.md](phase-1-exposure-mask.md) | Sorted fits honour the training/group/environment mask; zero-exposure is defined | None expected |
| 2 | [phase-2-gmm-log-intensity.md](phase-2-gmm-log-intensity.md) | One GMM log-intensity policy (C1) across local, non-local, ground-process | GMM goldens move |
| 3 | [phase-3-chunk-boundary.md](phase-3-chunk-boundary.md) | Backends bin globally and allocate only requested rows | None |
| 4 | [phase-4-numerical-hardening.md](phase-4-numerical-hardening.md) | KDE reducer NaN, scale-invariant weighted GMM, GLM objective | Small drift possible |
| 5 | [phase-5-windows-and-validation.md](phase-5-windows-and-validation.md) | Window ownership (C2), damping rejection, validators | None |
| 6a | [phase-6a-time-vocabulary.md](phase-6a-time-vocabulary.md) | `time_edges`/`time_centers`/`bin_durations` through every caller | All move |
| 6b | [phase-6b-rate-units.md](phase-6b-rate-units.md) | Backend-by-backend conversion to Hz | All move |
| 6c | [phase-6c-uniform-bins.md](phase-6c-uniform-bins.md) | Detector requires uniform bins; nonuniform stays on the direct API | None |
| 6d | [phase-6d-model-compat.md](phase-6d-model-compat.md) | Saved models with per-sample rates are detected and rejected | None |
| 7 | [phase-7-performance.md](phase-7-performance.md) | Checkpointed smoothing, incremental/compact outputs, structured transitions, and measured kernel optimizations | None (parity against the corrected baseline) |
| 8 | [phase-8-remaining-findings.md](phase-8-remaining-findings.md) | `to_density` volume, MRF disconnected components, sorted-index contract | None expected |

## Approval gates

Per `CLAUDE.md`, stop and request explicit approval before updating any snapshot
(`--snapshot-update`) or modifying golden regression data. Reached in phases 2, 4,
6a, and 6b, and in phase 0 if measured regression outputs require updates.
Phase 6b additionally changes a documented numerical bound (C1 is
changed in phase 2) — include both in the numerical-validation analysis.

The user has confirmed **no backwards-compatibility window is required**
(`CLAUDE.md` default). Phase 6d therefore rejects incompatible saved models with a
clear error rather than migrating them.
