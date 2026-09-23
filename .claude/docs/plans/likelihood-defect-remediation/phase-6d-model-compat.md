# Phase 6d — Detect incompatible fitted rate units

> **NEEDS PROTOTYPING; SHIPS ATOMICALLY WITH 6b.** There must be no release
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
  fitted attribute), never in constructor or `vars()` state: spyglass rebuilds
  detectors from `vars(params)` (the nld #43 issue). Legacy pickles restore
  `__dict__` without running `__init__`, so check units before accessing any
  attribute 6a/6b add, or users get `AttributeError` instead of a refit message.
- Release coordination: nld is at `v0.6.9`; spyglass plans a
  `>=0.7.0,<0.8` pin and is currently unpinned. Ship 6b/6d outside that range or
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
