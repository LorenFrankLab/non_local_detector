# Phase 6d — Detect incompatible fitted rate units

**Worktree status (2026-10-05):** package work is complete and independently
reviewed; ships atomically with 6b. The separate completion audit against
accepted 6c closed missing-active-entry and missing-encoding-state bypasses.
Final full suite: **2,397 passed / 6 skipped / 0 failed**. Seven supported
persistence paths, legacy inspection/refitting, and all guarded dispatch paths
pass. The approved 6b snapshot corrections remain applied; all eight snapshots
pass. Completion changes are committed at `5eced1e`; external rollout remains pending. See the
[completion record](#completion-review--2026-10-05) and
[scope record](phase-6-worktree-scope.md).

> **IMPLEMENTED, REVIEWED, ACCEPTED, AND COMMITTED; SHIPS ATOMICALLY WITH 6b.** There must be no release
> interval in which an old per-position-sample model is silently interpreted as
> Hz. The former implementation snippets and universal extra-key acceptance
> requirement are withdrawn; metadata ownership must be tested explicitly.
>
> Re-verified against `main` at `ee2cc21` (2026-09-22); line references are to
> that revision.

## Problem

Encoding dictionaries are stored inside pickled detectors. A pre-6b model holds
rates or fields per position sample; post-6b decoding uses Hz and multiplies by
decode duration. On a 500 Hz encoding grid the interpretations differ by a factor
of 500, allowing plausible but miscalibrated results if no unit check exists.
`save_model`/`load_model` are bare pickle (`base.py:2425-2458`) with no nld
version or unit metadata. Spyglass stores fitted classifiers this way and
reloads them for inspection (its `fetch_model()` reads
`encoding_model_[…]["place_fields"]`), so loading must keep working.

Adding metadata also affects dispatch:
`likelihood_func(time, position_time, position, spike_times, **self.encoding_model_[(env, group)], ...)`
(`models/base.py:3298-3306`, `:4318-4326`). A new key raises `TypeError` in the
sorted KDE, sorted GLM, clusterless KDE and clusterless log-KDE predictors; it is
silently absorbed by sorted diffusion/MRF (`**_encoding_extras`), GMM
(`**kwargs`) and clusterless diffusion (`**encoding_model`), and GMM and MRF fits
already rely on that absorption for diagnostic keys. Validation therefore cannot
depend on predictor signatures; it belongs at dispatch. Unit/exposure metadata must
be handled in the same change that introduces it.

## Decision and scope

The user requires no backwards-compatibility window. Reject incompatible models
with a clear refit instruction; do not infer units from numeric values or guess a
conversion using incomplete original exposure information. A small unit marker
is sufficient for this scope; a general versioned encoding schema is deferred.

The independent Patsy `DesignInfo` serialization defect for GLM detectors is
tracked in [overview.md](overview.md#found-during-review-not-yet-scheduled). This
phase does not silently absorb that repair or claim a GLM save/load round trip
works before the separate defect is fixed.

## Falsification and tasks

- Construct a legacy encoding dictionary without a unit marker and reproduce
  the absence of a compatibility check. Define old/new unit expectations with
  the 6b backend inventory, including backends that do not use `mean_rates`.
- Prototype one shared marker convention and record encoding exposure where
  needed. Every applicable fit and EM refit must emit truthful metadata. No-spike
  uses an already-Hz parameter and bypasses normal encoding dispatch; specify
  its treatment rather than assuming it has a registry-fit dictionary.
- Validate units before model-based likelihood evaluation, using actual
  encoding keys, for each `(environment, encoding_group)` entry that dispatch
  will use. Cover `predict`, `most_likely_sequence` and
  `compute_log_likelihood`. Public `estimate_parameters` refits before
  evaluating any likelihood (`base.py:3731`, `:4678`), so it must accept legacy
  models and validate after the refit. At `ee2cc21` no public API consumes a
  precomputed likelihood (`log_likelihood_` is output-only); only the private
  `_predict(log_likelihoods=…)` bypasses dispatch.
  A cached array's provenance/shape must not make an incompatible fitted model
  appear safe accidentally.
- Audit low-level direct predictor calls separately. Document where the caller
  is responsible for rate units and where supplied metadata is validated; do
  not claim a model-level guard protects calls that bypass it.
- Prototype explicit metadata handling at dispatch or declared keyword support
  in predictors. Every registered fit's returned dictionary must be usable by
  its matching predictor through supported paths. Accepting arbitrary unknown
  keys is not required and must not silently hide misspelled model parameters.
- Put the marker inside `encoding_model_` entries (or a trailing-underscore
  fitted attribute), never in constructor configuration or `get_params()`.
  Spyglass rebuilds detectors from `vars(params)` on an unfitted parameter
  object (the nld #43 issue); learned metadata belongs in `vars(fitted_detector)`
  and its pickle, not in that parameter object's configuration. Legacy pickles restore
  `__dict__` without running `__init__`, so check units before accessing any
  attribute 6a/6b add, or users get `AttributeError` instead of a refit message.
- Release coordination: at the recorded downstream audit, nld was at `v0.6.9`
  and Spyglass planned a `>=0.7.0,<0.8` pin while unpinned. Ship 6b/6d outside that range or
  coordinate the pin; existing spyglass `.pkl` classifiers will fail to decode
  (remedy: re-populate).
- Document missing, legacy, and unknown marker behavior, and the refit remedy.
  Loading may remain possible for inspection, but incompatible decoding must
  fail before likelihood evaluation.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Legacy and unknown units | Missing, old, or unsupported units are rejected before decoding through all guarded model paths. |
| Current units | Applicable registry fits and refits provide the agreed marker and exposure metadata; no-spike exceptions are explicit. |
| Metadata plumbing | Fit → supported predictor dispatch works for both registries, including strict signatures. Recognized metadata cannot cause an unexpected-key failure. |
| Persistence | Round-trip numerical parity for currently serializable model families (baseline at `ee2cc21`: 7 of 8 backends round-trip with bit-identical posteriors; GLM fails at save); GLM marker/dispatch validation is tested in memory until its independent serialization defect is repaired. |
| Bypass paths | Cached prediction, model likelihood assembly, and estimation obey the declared validation contract; direct low-level callers have an explicit unit contract. |
| Atomic rollout | The 6b Hz implementation and 6d checks are tested and released together. |

Run affected models/backends and the full suite alongside 6b. Marker plumbing
alone should not change newly fitted numerical outputs. Inspect whether golden
fixtures refit or load saved models; a needed reference update follows the
existing numerical-change analysis/approval process. (Goldens pickle inputs and
posteriors and refit; the only save/load test is
`test_clusterless_diffusion_integration.py:310-381`.) Include the joint breaking
change and refit instructions in the release note.

## Implementation and review — 2026-10-05

Every registered fit/refit emits `rate_units="Hz"` and
`encoding_exposure_seconds`. The detector stamps fitted
`time_contract_={"version": 1, "rate_units": "Hz", "encoding_time_units": "seconds"}`.
This is fitted state rather than constructor configuration, preserving downstream
parameter reconstruction. No-Spike remains an already-Hz constructor parameter
and does not invent a registry encoding dictionary.

`predict`, `most_likely_sequence`, and `compute_log_likelihood` reject missing or
unknown contracts, missing/old per-entry rate markers, and unknown transition
clock provenance before evaluating likelihoods. Private cached `_predict` also
checks any fitted encoding model. All newly added metadata is accepted through
each of the eight matching predictor dispatch paths. Direct low-level callers
default to Hz and validate supplied markers; callers omitting metadata remain
responsible for the units of their own arrays.

Public estimation refits from original recording inputs before decoding, so
legacy models recover through that route instead of being permanently rejected.
Initial regression tests cover both sorted and clusterless detector families, all four
guarded decode paths, three per-entry/version corruptions, and estimation recovery
with deliberately removed legacy metadata (**26 cases**). Separate top-level
missing-contract and transition-clock/cached regressions remain in the public
contract suite.

All seven currently serializable backends preserve exact predictions, result
metadata, Hz markers, and physical exposure after save/load. The eighth, GLM,
passes fit/predict calibration and marker dispatch in memory. Its Patsy
`DesignInfo` pickle defect is recorded publicly and remains deferred, as required
by this phase's scope. Existing golden fixtures refit from pinned inputs and
continue to pass without file/tolerance changes.

Two independent reviewers found no remaining concrete package defect. Joint
validation and the approved snapshot corrections are recorded in the
[6b implementation record](phase-6b-rate-units.md#implementation-and-review--2026-10-05)
and [numerical analysis](../../../../docs/time_grid_validation.md). Source commits
are recorded in the scope record; neither phase is merged or released. Downstream adapter changes, dependency coordination,
and classifier/result re-population still gate release.

## Completion review — 2026-10-05

The explicit 6d completion audit uses the frozen accepted 6c tree as its
baseline. It reproduced two remaining compatibility gaps: validating only
existing dictionary values missed a deleted active entry (and incorrectly
rejected unused or No-Spike-only legacy entries), and private cached prediction
or base Viterbi could bypass validation when the complete `encoding_model_`
attribute was absent.

The shared guard now validates every `(environment, encoding_group)` required
by a spike observation before any state likelihood or HMM work. Missing entries,
non-mapping entries, and missing, old, or unsupported unit markers produce a
refit error. Unused entries do not block decoding. No-Spike uses its constructor
rate in Hz and requires no encoding entry, while retaining the detector's global
time/transition-clock contract. Both concrete detector families always validate
that contract through cached prediction and base Viterbi, even if fitted
encoding state is absent. Generic base estimators with fixed likelihood arrays
and no encoding model retain their existing behavior.

The new compatibility module adds **128 cases** covering both detector families,
separate environments and encoding groups, all guarded paths, unsupported marker
and entry types, unused/No-Spike entries, constructor reconstruction, legacy
pickle inspection without calling `__init__`, weighted encoding refits, and
explicit/default direct-predictor units on all eight backends. Portable legacy
pickle cases restore faithful per-position-sample rates without rewriting or
converting a saved model. Existing seven-backend save/load parity and in-memory
GLM validation remain green; GLM serialization stays deferred.

Red evidence is saved in `/private/tmp/nld-phase6d-red.log` (**22 failures / 58
passes**), `/private/tmp/nld-phase6d-marker-types-red.log` (**32 failures**), and
`/private/tmp/nld-phase6d-bypass-red.log` (**8 failures / 8 passes**). The final
focused run passes **314 cases**, including the generic core estimator tests;
float64 compatibility/persistence passes **177 cases**. Both independent final
reviews report no remaining concrete compatibility or API/UX defect.

Controlled decoder/nonlocal outputs are bit-identical to accepted 6c. All eight
golden input/output files remain byte-identical to accepted 6a, and no likelihood
formula, reference, tolerance, or convergence criterion changed in this review.
Targeted mypy retains the same **158 existing diagnostics** as accepted 6c.
Ruff and formatting pass for all 61 changed Python files; whitespace checks
pass. The final complete suite passes **2,397 / 6 skipped / 0 failed**
(721.62 seconds), including all four goldens and all eight approved snapshots.
Log: `/private/tmp/nld-phase6d-final-full.log`. The 175 frozen Python-source
hashes in `/private/tmp/nld-phase6d-final-run-source-hashes.json` still match.

The accepted source checkpoint is `/private/tmp/nld-phase6d-accepted`, with
hashes in `/private/tmp/nld-phase6d-accepted-source-hashes.json`. Accepted 6a,
6b, and 6c checkpoints are preserved separately. Package acceptance is complete.

The separate runtime/test diff is
`/private/tmp/nld-phase6d-vs-accepted6c.patch`. This phase still ships atomically
with 6b. Downstream adapters, dependency pins, and model/result re-population
remain release work. The audit itself made no commit, push, or downstream write;
the user subsequently authorized the source commits recorded above.
