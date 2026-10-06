# Phase 6 worktree scope — 2026-10-05

The branch is `feat/time-edges-uniform-bins`; the review baseline was `6d0193f`.
The user requested completing/reviewing 6a before implementing/reviewing 6b.
That order has now been followed with an isolated, frozen 6a baseline. Both
phase pairs are now committed in logical groups; neither is merged or released.

## Source commits — 2026-10-05

| Commit | Scope |
|---|---|
| `ef74945` | 6a completion: explicit-bin API, result contracts, and script consumers. |
| `b63ddc7` | Atomic 6b/6d implementation: physical Hz rates, support/exposure, legacy guards, and approved snapshots. |
| `351a630` | 6c completion: uniform-grid, overflow, and learned-clock validation. |
| `5eced1e` | 6d completion: active-entry and missing-encoding-state validation. |

Migration documentation, examples, and the validation/plan records follow in a
separate commit. Each source commit matches its corresponding accepted source
checkpoint, excluding the ignored build-generated `_version.py`. The original
review hashes retain that generated file for reproducing the test environment.

## Finished and reviewed: 6a/6c package work

The accepted source checkpoint is `/private/tmp/nld-phase6a-accepted`, its diff
is `/private/tmp/nld-phase6a-accepted.patch`, and source hashes are saved in
`/private/tmp/nld-phase6a-accepted-source-hashes.json`. This tree contains the
API/coordinate/uniform-grid changes and no 6b seconds/Hz/support/unit markers.

The full completion suite passed **2,125 / 6 skipped / 0 failed**, including all
four goldens and eight snapshots unchanged. Focused chunk/row/API checks and
float64 checks pass. Two independent reviews accepted the package corrections,
including singleton spatial/time axes, the small NetCDF index-loading repair,
and early fitted-covariate Viterbi alignment checks. Controlled decoder and
nonlocal outputs were bit-identical to the original 6a branch reference.
See the [6a completion record](phase-6a-time-vocabulary.md#completion-review--2026-10-05).

This review-completion evidence is separate from the historical 6a endpoint
migration and its earlier approved golden update. Encoding support and Hz
changes were not attributed to 6a.

## Implemented and reviewed: 6b/6d, together

The current workspace layers these changes on the accepted 6a source:

The reviewed 6b candidate is saved at `/private/tmp/nld-phase6b-reviewed`, with
source hashes in `/private/tmp/nld-phase6b-reviewed-source-hashes.json` and a
separate implementation diff at `/private/tmp/nld-phase6b-vs-accepted6a.patch`.
This checkpoint preserves the pre-approval reviewed source. Snapshot corrections
were subsequently approved and applied; all eight repository snapshots pass.

- Original acquisition/tracking support, NaN breaks, integrated exposure seconds,
  seconds occupancy weights, and dimensionless event weights applied once.
- Hz fields/coefficients on all eight backends; physical per-row event and ground
  duration factors, including the already-Hz No-Spike parameter.
- Physical floor interpretation, seconds-normalized GLM objective/penalty, and
  MRF numerical-unit adjustment, retaining the deferred C1 policy and criteria.
- Fitted unit/exposure metadata, unknown/legacy-contract rejection, cached-path
  guards, and recovery through estimation refitting.
- Reviewed callback correction: unmarked custom likelihoods retain full-grid
  support; partial requests require full edges and `row_slice_aware`. Edge
  nudging cannot preserve both duration and boundary ownership.

The eight-backend inventory is complete in the
[6b implementation record](phase-6b-rate-units.md#implementation-and-review--2026-10-05).
All **16 public calibration cases** pass with 30/500 Hz tracking and 2/4 ms
bins; they fail against the accepted 6a baseline. All seven currently
serializable model families keep exact results after save/load. GLM units are
verified in memory; its independent Patsy pickle defect remains deferred.
Marker/version/cached-path and legacy-refit coverage passes for both detector
families. Final independent correctness and UX reviews found no remaining
concrete package defect.

Final numerical/full-suite evidence is in
[time_grid_validation.md](../../../../docs/time_grid_validation.md). Three
duration-sensitive likelihood snapshot corrections were explicitly approved and
applied on 2026-10-05, after independent reference derivations. All eight
repository snapshots pass, completing the 6b/6d package acceptance gate. The
accepted source plus approved references is preserved at
`/private/tmp/nld-phase6b-accepted`, with source hashes in
`/private/tmp/nld-phase6b-accepted-source-hashes.json`. Golden files and numerical
tolerances are unchanged.

## Phase 6c completion review

The subsequent explicit 6c review found private/cached HMM, grid-constructor,
overflow, and saved-clock validation gaps. These are corrected under the same
4-ULP/0.1% uniformity and 1% resolution policy; no constant is changed.
`time_edges_from_centers` and `calculate_time_edges` now validate generated
boundaries. Cached prediction and base Viterbi validate the complete grid and
known transition clock before work. Invalid widths/tolerances require refitting.

The separate runtime/test diff against accepted 6b is
`/private/tmp/nld-phase6c-vs-accepted6b.patch`. Red→green defect regressions,
both-family covariate ordering/mutation checks, manual/generated 1.8-million-bin
timelines at zero/Unix origins, and exact valid-grid numerical comparisons are
recorded in the [6c completion review](phase-6c-uniform-bins.md#completion-review--2026-10-05).
The final focused run passed 300 checks and the final float64 run passed 120.
The final complete suite passed **2,269 / 6 skipped / 0 failed**, including
all goldens and snapshots. The accepted source checkpoint is
`/private/tmp/nld-phase6c-accepted`; hashes are recorded in
`/private/tmp/nld-phase6c-accepted-source-hashes.json`. Full evidence is in the
numerical validation report. Phase 6c package acceptance is complete.

Both final independent reviewers found no remaining concrete 6c defect.
Standalone registry likelihoods retain variable-duration support; public core
row APIs and duration-calibrated transitions remain outside the detector guard.

## Phase 6d completion review

The subsequent explicit 6d audit reproduced missing-active-entry and
missing-encoding-container bypasses against accepted 6c. The guard now checks
the complete set of spike-state encoding keys before likelihood/HMM work,
including cached prediction and base Viterbi. Unsupported entries/markers
require refitting. Unused dictionaries and No-Spike-only keys do not block valid
decoding; the No-Spike constructor rate remains Hz. Generic fixed-likelihood
base estimators retain their existing behavior.

The separate runtime/test diff is
`/private/tmp/nld-phase6d-vs-accepted6c.patch`. The new module adds 128 portable
regressions, including faithful legacy pickle inspection without `__init__`,
constructor/get-params isolation, weighted refits, and direct predictor units
on all eight backends. The final focused run passes **314 cases** and float64
compatibility/persistence passes **177 cases**. Both final independent reviews
find no remaining concrete 6d defect. Controlled decoder/nonlocal arrays are
bit-identical to accepted 6c; golden files, formulas, references, and tolerances
are unchanged during this audit. Targeted mypy retains 158 existing diagnostics.

The final full suite passes **2,397 / 6 skipped / 0 failed** (721.62 seconds),
including all four goldens and all eight approved snapshots. The 175-file
source manifest still matches. Ruff/format pass on all 61 changed Python files;
whitespace checks pass. The accepted source checkpoint is
`/private/tmp/nld-phase6d-accepted`, with hashes in
`/private/tmp/nld-phase6d-accepted-source-hashes.json`. Previous accepted source
checkpoints remain intact. Full evidence is recorded in the
[6d completion record](phase-6d-model-compat.md#completion-review--2026-10-05).
Phase 6a–6d package work is complete and committed. The rate/unit change ships atomically with 6b;
external release coordination is separate.

## Release and deferred work

The combined migration guide and executable example now cover gap masks,
per-segment decode tracking, event ownership, metadata, direct-call units,
custom chunk callbacks, and refitting. This is package migration evidence;
Spyglass and replay/shuffle repositories remain unmodified. Adapter API calls,
original tracking support, masks/covariates, result labels, saved-model/result
re-population, dependency coordination, and validation on the supported
DataJoint/NWB stack remain release gates.

Phases 7/8, the C1 background firing model, general encoding schemas, singleton
geometry, general GLM knot defaults, GLM serialization, and EmpiricalMovement
repairs remain outside this implementation. Passing package tests does not
claim completion of those items or downstream rollout.


## PR review follow-up — 2026-10-06

The user requested fixing all findings after the published `338b4f8` review.
This follow-up preserves the accepted checkpoints above and changes only
Phase 6 support/units, decode rows, compatibility, bounded chunk preparation,
and consumer migration. The new scientific behaviors have independent analytic
references; no new reference approval, tolerance change, convergence change,
C1 policy, Phase 7/8 work, or continuous-time transition model is introduced.

| Commit | Follow-up scope |
|---|---|
| `3ca874b` | Explicit interval support and physical-time rate diagnostics. |
| `98bb20d` | Model inputs, stable provenance, prepared chunk grids, and the float64 independent CI reference. |
| `7c9dd29` | Tracking alignment, actual consumer calls, Hz displays, portable consumer tests, and stale output cleanup. |

The separate documentation/companion commit follows these source groups.

The follow-up closes explicit-interval clipping by outside NaNs, physical-time
model checks, stored nonstationary covariate row mismatches, all-missing
population/mark validation, caller-input aliasing in results, and repeated
recording-sized chunk workspace. Consumer fixes execute actual notebook and
profiler calls, preserve measured tracking gaps and categorical/circular
semantics, correct fitted-Hz display units, and remove stale changed-cell
outputs. Source-dependent consumer tests skip only when wheel installations
omit those source files; public alignment tests remain active.

The [validation record](../../../../docs/time_grid_validation.md#pr-review-follow-up--2026-10-06)
contains red/green evidence, bit-identical valid-input controls, unchanged golden
files, 237 float64 cases, unchanged mypy diagnostics, and benchmark scope.
The frozen complete suite passes **2,656 / 6 skipped / 0 failed** (825.84
seconds); all 231 source/test/notebook hashes match and goldens/snapshots pass.
GitHub CI is tracked in [PR #59 checks](https://github.com/LorenFrankLab/non_local_detector/pull/59/checks). The current companion passes **51 tests**, independently
repeated; archived adapter passes **27**. Both patches and baseline hashes are
saved in the repository. The current observed-time test needs a test-hunk rebase
before any live application; production files remain untouched.

A [Spyglass adapter companion](../../../../docs/spyglass_migration/README.md)
is prepared and tested separately. The live checkout advanced independently
with unrelated dirty edits and already pins NLD 0.6.9; it is untouched. This
supersedes the earlier unbounded-dependency finding for that checkout, while
preserving the archived adapter evidence. Normal DataJoint database fixtures,
selected backend persistence, supported matrix/conda checks, representative
recomputed scientific results, and coordinated releases remain rollout gates.
