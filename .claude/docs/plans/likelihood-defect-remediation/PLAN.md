# Likelihood Defect Remediation

Status: Not started.

## Summary

An audit of `src/non_local_detector/likelihoods/` and its integration with
`models/base.py` and `core.py` found four defects that silently change decoded
posteriors, plus numerical, validation, and calibration gaps. Every finding was
reproduced with a runnable script before being written down; the reproduction is
quoted in the phase that fixes it.

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
| 1 | **Prototyped, passing, uncommitted** on `fix/sorted-spikes-exposure-mask`. Spy test fails on `main`, passes after; 1262 passed / 3 skipped, goldens unchanged |
| 2 | Needs prototyping: aggregate flooring, zero-mass detection, and the newly in-scope log-KDE clamps |
| 3 | Needs redesign: the clusterless path must filter spikes before density evaluation, not slice a full-time reduction; no-spike states bypass both registries (`base.py:3163`); the >4× memory target is unachievable while full posteriors are retained (`core.py:671`, `:700`) |
| 4 | Nearly ready: weight normalizer overflows on large float32 weights; the GLM objective fix promised in the index below is missing |
| 5 | Rewrite against the corrected C2 — the hard-window machinery is deleted, not repaired |
| 6a–6d | Needs prototyping: exposure endpoints, a central edge validator, ULP-aware uniformity, rate-vs-expected-count test semantics, and 6b/6d must be atomic |
| 7 | Reasonable after minor test hardening |
| 8 | Mostly ready; 9 call sites in `likelihoods/` source, not 13; reconsider whether to drop `indices_are_sorted` rather than reject previously-accepted input |

## Reading order

1. [shared-contracts.md](shared-contracts.md) — C1 degeneracy policy, C2 exposure
   ownership, C3 time vocabulary.
2. [overview.md](overview.md) — goals, non-goals, architecture map, deferred items.
3. Phase files in order.

## Phase index

| Phase | File | Ships | Goldens |
|---|---|---|---|
| 1 | [phase-1-exposure-mask.md](phase-1-exposure-mask.md) | Sorted fits honour the training/group/environment mask; zero-exposure is defined | None expected |
| 2 | [phase-2-gmm-log-intensity.md](phase-2-gmm-log-intensity.md) | One GMM log-intensity policy (C1) across local, non-local, ground-process | GMM goldens move |
| 3 | [phase-3-chunk-boundary.md](phase-3-chunk-boundary.md) | Backends bin globally and allocate only requested rows | None |
| 4 | [phase-4-numerical-hardening.md](phase-4-numerical-hardening.md) | KDE reducer NaN, scale-invariant weighted GMM, GLM objective | Small drift possible |
| 5 | [phase-5-windows-and-validation.md](phase-5-windows-and-validation.md) | Window ownership (C2), damping rejection, validators | None |
| 6a | [phase-6a-time-vocabulary.md](phase-6a-time-vocabulary.md) | `time_edges`/`time_centers`/`bin_durations` through every caller | All move |
| 6b | [phase-6b-rate-units.md](phase-6b-rate-units.md) | Backend-by-backend conversion to Hz | All move |
| 6c | [phase-6c-uniform-bins.md](phase-6c-uniform-bins.md) | Detector requires uniform bins; nonuniform stays on the direct API | None |
| 6d | [phase-6d-model-compat.md](phase-6d-model-compat.md) | Saved models with per-sample rates are detected and rejected | None |
| 7 | [phase-7-performance.md](phase-7-performance.md) | Matmul, JIT bucketing, block copies — with measured evidence | None (parity) |
| 8 | [phase-8-remaining-findings.md](phase-8-remaining-findings.md) | `to_density` volume, MRF disconnected components, sorted-index contract | None expected |

## Approval gates

Per `CLAUDE.md`, stop and request explicit approval before updating any snapshot
(`--snapshot-update`) or modifying golden regression data. Reached in phases 2, 4,
6a, and 6b. Phase 6b additionally changes a documented numerical bound (C1 is
changed in phase 2) — include both in the numerical-validation analysis.

The user has confirmed **no backwards-compatibility window is required**
(`CLAUDE.md` default). Phase 6d therefore rejects incompatible saved models with a
clear error rather than migrating them.
