# Likelihood Defect Remediation

Status: Phases 0–4 are merged on `main` (Phase 0 `c808dde`; Phase 1 `8c8765f` +
`5bd63d4`; Phase 2 `86e22f0`; Phase 3 `ffc85a1`; Phase 4 `ee2cc21`). The
remaining phase plans were re-verified against `ee2cc21` on 2026-09-22 and have
the readiness states recorded below.

## Summary

An audit of `src/non_local_detector/likelihoods/` and its integration with
`models/base.py` and `core.py` found four defects that silently change decoded
posteriors, plus numerical, validation, and calibration gaps. A follow-up audit
reproduced two additional core HMM defects: normalization loses almost all
probability mass for small positive normalizers, and a maximum likelihood in an
unreachable state causes avoidable underflow and false `-inf` evidence. The
historical findings have recorded reproductions or contract-violation evidence
in their phase files. Later design candidates and production requirements still
need prototyping; they are not validated fixes.

Historical headline defects: sorted-spike encoding discarded its training / environment /
encoding-group mask (place fields 40–45% low; fixed in Phase 1); the clusterless
GMM floored a log-density before forming a ratio (~10^11 intensity distortion in
a realistic tail; fixed in Phase 2); chunked prediction dropped every spike
falling between two chunks (fixed in Phase 3); and `time` is read inconsistently
as bin edges vs. timestamps, leaving a final row that can never contain a spike
and rates calibrated to an unstated assumption (Phase 6). Still open: the
clusterless encoding fit bridges group-mask gaps and counts a gap spike twice at
full weight (Phase 5).

**Read [shared-contracts.md](shared-contracts.md) first. C2 and C3a are settled;
C1 is partially resolved and C3b is unresolved.** C1's former policy was
withdrawn as non-monotonic; phase 2 shipped the GMM arithmetic corrections
under the existing floors, and the package-wide policy (a background firing
model) is deferred to a separate proposal. Implementation selecting an
unresolved floor or encoding-exposure policy is blocked. This does not block
independent prototypes or changes preserving the current policy, as specified
below.

**Completed foundation: [phase 0](phase-0-core-hmm.md), core HMM correctness.**
Its correctness fixes and performance revision are merged on `main`
(`c808dde`), with reference tests and measured numerical and
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

**Process for the remainder of this work: prototype, then prescribe code.**
Requirements and explicit design options may be recorded before implementation;
unexecuted implementation snippets must not be presented as a solution. Readiness
requires resolved dependencies, a working prototype, and passing acceptance
checks. Defect regressions must demonstrate the relevant pre-fix failure (or a
documented contract violation); preservation/parity checks may already pass on
the baseline. Performance work needs measured baseline costs, not an invented
failing correctness test. Pin revisions rather than relying on a moving `main`.

Shared-contract measurements establish the stated problem or decision only;
they do not establish that an unresolved policy or implementation is ready.

## Execution order and baselines

Phase numbers identify scope and reading order. These dependencies determine
release order; implementation details remain subject to prototyping.

| Work | Required contract or predecessor | Release constraint |
|---|---|---|
| 0 and 1 | Existing likelihood/time semantics | Complete; preserve their regression coverage. Future policy changes belong to the phase introducing them. |
| 2 | Existing floors and zero-rate fallback (narrowed C1 decision) | Complete; the package-wide policy is deferred and any later floor change belongs to the phase introducing it. |
| 3 | Current unchunked event ownership | Complete (merged at `ffc85a1`); it chose neither C1 nor C3b and preserved the endpoint convention, so the corrected behavior must be retained through 6a. |
| 4 | Existing baseline floor policy, preserved explicitly | Complete (merged at `ee2cc21`); existing floors preserved; no deferred C1 policy selected. |
| 5 | Settled C2 weighted-event ownership | Remove hard windows and move the clusterless fit to full-timeline mask weights (as Phase 1 did for sorted); do not independently settle acquisition endpoints/gaps from C3b. |
| 6a + 6c | Settled C3a and resolved C3b for encoding-cell migration | Ship the edge/coordinate migration and detector uniformity guard together. |
| 6b + 6d | 6a/6c, Phase 5 event weights, resolved C3b, and applicable C1 decisions | Ship Hz conversion, metadata plumbing, and legacy-model rejection atomically. |
| 8 | Applicable C1 decisions for the zero-exposure floor (not Phase 6b units) | Can ship before Phase 7 and does not require Phase 6b units; sorted-index work was done in Phase 3 (GPU validation outstanding). |
| 7 | Corrected likelihood/time baseline and relevant Phase 8 fixes | Pure-core prototypes may run earlier; production claims require the integrated corrected baseline. |

Phases 0–5 remain the correctness priority before the Phase 6 migration. Review
Phase 6a/6c separately from 6b/6d so time/coordinate effects and rate-unit effects
remain attributable. No ordering here selects C1 or C3b. Record actual golden
changes at each step; approval requirements apply to changes observed, not to
an assumption that every golden must change.

### Readiness

| Phase | State |
|---|---|
| 0 | **Implemented, optimized, and validated**; merged at `c808dde` (`c1f7e33`; benchmark script `e79501a`). Full suite: **1329 passed / 4 skipped**; reference tests: **63 passed / 1 skipped** in default float32 and **64 passed** with x64 enabled. Golden files and existing tolerances unchanged; ruff and format pass. Numerical comparisons are bit-identical to the pre-optimization fix; stationary kernels take 6.5–14.3% less time than that intermediate draft. The corrected 200-bin filter remains approximately 13% slower than the pre-Phase-0 `main` baseline in the recorded CPU runs. See [phase 0](phase-0-core-hmm.md). Independent of C1/C3. |
| 1 | **Implemented and reviewed**; on `main` (`8c8765f`; regression coverage `5bd63d4`). Mask and zero-exposure regressions fail against the relevant pre-fix behavior. Post-review full suite: **1266 passed / 3 skipped**, goldens unchanged; ruff and format pass. |
| 2 | **Implemented (narrowed scope)**; merged at `86e22f0` (`fc20abb`): raw log ratios at all three GMM spike-intensity sites and one shared log-space ground-process helper (rate inside the exponent, no underflow guard, NaN propagates) on both fit and local paths; the zero-rate fallback, mean-rate floor, and summed-intensity clip are preserved. Reference tests fail on `main` and pass on the branch in float32 and float64; no GMM golden/snapshot fixture exists. The package-wide degeneracy policy (background firing model) is deferred. See [phase 2](phase-2-gmm-log-intensity.md). |
| 3 | **Implemented and validated**; merged at `ffc85a1` (based on `86e22f0`). Backends bin spikes against the full timeline and evaluate only requested rows, including boundary spikes. Both detector families and transition drivers return requested likelihoods in global order. Follow-ups invalidate stale stored likelihoods, preserve positional dtype compatibility, select features before conversion, bound digitization to chunk boundaries, prepare No-Spike duration once, validate JAX sorted-index hints, and share spike ordering across states/chunks within each prediction. Later predictions recheck mutated inputs. The six dedicated modules contain 262 tests; current numerical, full-suite and runtime evidence is recorded in [Phase 3](phase-3-chunk-boundary.md). Goldens, snapshots and tolerances are unchanged. Memory/runtime measurements are CPU-only on small grids; full posterior retention remains Phase 7a, and the former total-memory reduction target remains withdrawn. |
| 4 | **Implemented and validated**; merged at `ee2cc21` (based on `ffc85a1`). Empty-tile operands are safe, diagonal/spherical Gaussian distances and covariance updates are centered, and weighted GMM initialization/EM/objectives use stable weight normalization. Clusterless fits retain zero-weight exclusion before data validation. Full suite: **1762 passed / 6 skipped** after the second review round; focused GMM/KDE-tile/clusterless-weight tests: **120 passed** in both default float32 and x64 modes. |
| 5 | **Needs prototyping against settled C2** (claims re-verified at `ee2cc21`). Remove hard windows in both helpers; move the clusterless fit from the subset timeline to full-timeline mask weights so interpolation no longer bridges gaps (this changes clusterless fits whose group mask has gaps); use per-neuron weighted GLM counts; validate populations before pairing/indexing and in sorted diffusion/MRF; reject any nonzero `encoding_update_damping` with `ValidationError` before mutation. Optional non-local `position` needs regression tests only (the GMM defect was fixed by `4f4e862`). |
| 6a–6d | **C3b unresolved; needs prototyping.** Central edge validation and encoding-cell alignment; 6a/6c ship together and 6b/6d ship atomically. Uniformity checks must account for timestamp representability. |
| 7 | **Expanded scope; needs prototyping.** 7a: bounded memory smoothing and incremental/compact outputs. 7b: structured forward/backward transitions. 7c: measured likelihood and compilation optimizations. Validate the representative workload, existing numerical tolerances, host/device peak memory, and end-to-end runtime before claiming production support. |
| 8 | **Density/component fixes need prototyping** (both reproduced at `ee2cc21`). Distinguish mass from density on variable-volume bins (sorted diffusion is the outlier; clusterless diffusion already uses the mass convention) and define unoccupied-component behavior consistently with the selected policy. Phase 3 now sets sorted-index hints from a verified ordering at every reduction site and shares ordering checks across states/chunks within each prediction. GPU validation remains outstanding. |

## Reading order

1. [shared-contracts.md](shared-contracts.md) — C1 degeneracy policy, C2 exposure
   ownership, C3 time vocabulary.
2. [overview.md](overview.md) — goals, non-goals, architecture map, deferred items.
3. [phase-0-core-hmm.md](phase-0-core-hmm.md), then the remaining phase files in
   order. A later phase's readiness does not resolve an earlier phase's blockers.

## Phase index

| Phase | File | Ships | Goldens |
|---|---|---|---|
| 0 | [phase-0-core-hmm.md](phase-0-core-hmm.md) | Stable Bayesian conditioning and posterior normalization in both core paths | Validated unchanged; defect regressions now match the mathematical reference |
| 1 | [phase-1-exposure-mask.md](phase-1-exposure-mask.md) | Sorted fits honour the training/group/environment mask; zero-exposure is defined | Validated unchanged |
| 2 | [phase-2-gmm-log-intensity.md](phase-2-gmm-log-intensity.md) | Raw log-intensity ratios and a log-space ground process in the clusterless GMM, local and non-local | No GMM fixtures exist; KDE goldens unaffected |
| 3 | [phase-3-chunk-boundary.md](phase-3-chunk-boundary.md) | Backends bin globally and allocate only requested rows | Validated unchanged; corrected chunks match the unchunked reference |
| 4 | [phase-4-numerical-hardening.md](phase-4-numerical-hardening.md) | KDE reducer NaN, stable Gaussian distances/covariances, scale-invariant weighted GMM and objective | Validated unchanged; affected primitive and estimator differences match independent references |
| 5 | [phase-5-windows-and-validation.md](phase-5-windows-and-validation.md) | Canonical event weights and hard-window removal (C2), damping rejection, validators | Clusterless fits with gapped group masks change by design; measure and attribute. Full-coverage fixtures are expected to retain parity but cannot show absence of effect |
| 6a | [phase-6a-time-vocabulary.md](phase-6a-time-vocabulary.md) | Edges, centers, encoding cells, and row alignment; ships with 6c | Shape, coordinate, and numerical changes possible; measure |
| 6b | [phase-6b-rate-units.md](phase-6b-rate-units.md) | Backend-by-backend conversion to Hz; ships with 6d | Stored units change; likelihood/posterior effects require derivation |
| 6c | [phase-6c-uniform-bins.md](phase-6c-uniform-bins.md) | Detector requires uniform bins; nonuniform stays on the direct API | None |
| 6d | [phase-6d-model-compat.md](phase-6d-model-compat.md) | Saved models with per-sample rates are detected and rejected | None |
| 7 | [phase-7-performance.md](phase-7-performance.md) | Checkpointed smoothing, incremental/compact outputs, structured transitions, and measured kernel optimizations | None (parity against the corrected baseline) |
| 8 | [phase-8-remaining-findings.md](phase-8-remaining-findings.md) | `to_density` volume, MRF disconnected components, sorted-index contract | Changes possible on affected fixtures; measure |

## Approval gates

Per `CLAUDE.md`, provide the required numerical-change analysis before requesting
approval for an observed snapshot/golden update or a numerical-tolerance or
convergence-criterion change. Phases 2, 4, 5, 6a/6b, and 8 may produce legitimate
numerical changes; none is permission to update reference files automatically.
Phase 7 preserves its corrected baseline. Phase 0/1 completed without golden
updates. Commit/push permissions remain those in `CLAUDE.md` and the session.

The user has confirmed **no backwards-compatibility window is required**
(`CLAUDE.md` default). Phase 6d therefore rejects incompatible saved models with a
clear error rather than migrating them.
