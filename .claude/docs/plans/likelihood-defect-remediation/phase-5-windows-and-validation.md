# Phase 5 — Canonical event ownership and validation gaps

> **NEEDS PROTOTYPING AGAINST SETTLED C2.** Hard group windows are removed, not
> repaired. The former half-interval window implementation and disjoint-window
> tests contradicted C2 and are withdrawn. Other validation changes retain the
> scopes below; no backend damping-rebuild architecture is added.

## Contracts and scope

[C2](shared-contracts.md#c2--exposure-ownership-and-spike-weights) assigns encoding
spikes by interpolated sample weights. Phase 1 passed the exposure mask correctly
and remains complete. This phase makes event ownership consistent with those
weights. It does not choose C1 floors or C3b acquisition endpoints/gap exposure.
Any case requiring an unresolved C3b choice must be identified rather than
settled through an implicit window rule.

## 1. Remove hard group windows

The old helpers widened each contiguous run by a global first time difference.
The recorded mask `[T,T,F,T,T]` at times 0…4 put the spike at t=2 into both
windows. Its canonical weight is zero, so it should contribute nothing.

Overlap alone is not the mathematical defect: at t=1.75 the same mask gives
weight 0.25 and its complement gives 0.75. Both groups legitimately receive
fractional contributions whose total is one. No disjoint-window construction
can preserve every nonzero interpolated weight.

### Requirements and falsification

- Delete run-window filtering and select events with positive canonical weight
  within the supported recording domain. Preserve spike/feature alignment and
  retain the fractional weight itself for fitting; selecting a spike does not
  make its weight one.
- Apply each event weight once. Where a backend already interpolates event
  weights, avoid applying them again at the detector layer. Preserve weighted
  exposure on the original position timeline.
- Prototype GLM weighted sufficient statistics explicitly. Ordinary integer
  spike counts multiplied by sample-row weights cannot in general reproduce
  interpolated per-event weights. Prevent double-weighting weighted counts and
  distinguish the event term from the exposure term. Carry this contract into
  Phase 6a's encoding cells and Phase 6b's duration offsets.
- Inventory equivalent ownership in KDE, diffusion, MRF, and clusterless fits.
  Do not equate their representations without testing the same weighted-event
  reference. Subsetting the time grid must not redefine interpolation or exposure.

Before editing, reproduce a nonzero-weight event discarded by a hard window and
an excluded event with zero canonical weight. Test complementary/multiple groups
against explicit interpolated weights, including fractional/EM weights and
jittered times. Restrict endpoint/gap reference cases to the declared recording
contract; retain unresolved C3b cases as blockers for that part of the work.

### Validate before pairing populations

Group helpers can truncate spike/feature collections before backend validation.
Validate population lengths and each electrode's spike/feature row alignment
before any pairing or indexing, using package `ValidationError` diagnostics.
A bare strict zip is insufficient. Prototype a shared validator where appropriate
and exercise the public detector path, not only the backend validator.

## 2. Reject unsupported encoding damping before mutation

The current blend updates place fields and a derived no-spike term without
rebuilding other fitted state. Recorded consumers include KDE marginal models
and mean rates, diffusion/MRF interior log fields, GLM coefficients, and
clusterless models without place fields.

The recorded audit found **no registered backend with a coherent implementation
of this place-field-only blend**. Recheck the inventory at the implementation
revision and reject nonzero damping for that unsupported set before either
concrete wrapper's first fit/refit or other detector mutation. Zero damping
retains current behavior. Do not invent support or rebuild all encoding models
inside this phase; any newly supported case needs explicit local/non-local
consistency evidence.

Test rejection from both public estimation wrappers and verify fitted state is
unchanged after the error. The release note must identify the unsupported
parameter value and its reason, rather than describing damping as a silent no-op.

## 3. Optional position in non-local clusterless prediction

The recorded GMM non-local path accepts `position=None` at the detector boundary
but dereferences `.ndim` in its predictor, raising `AttributeError`. Prototype a
position-independent non-local path and explicit validation when position is
required by a local observation or configured position-dependent term.

Audit KDE/log-KDE/diffusion paths against the same requirements. Test public and
direct calls, with local-position kernels and non-local penalties accounted for;
`is_local=False` alone does not prove every configured model is position-free.

## 4. Population validation in sorted diffusion and MRF

Validate the number of spike trains against fitted place fields and any interior
log fields before count construction or matrix multiplication. Cover local and
non-local paths and both public backend names, including shared implementations.
Re-inventory current validators before adding duplicates.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Canonical ownership | Explicit per-event weights match interpolated sample weights; zero-weight events contribute nothing and no positive-weight event is discarded by a group window. |
| Partition of unity | Complementary/multiple group contributions sum to the original event weight, including fractional boundary events. Disjoint windows are not required. |
| Backend sufficient statistics | KDE/GLM/diffusion/MRF/clusterless paths apply event weights once and use the corresponding exposure; GLM weighted counts have an independent reference. |
| Population alignment | Collection and per-electrode row mismatches raise package diagnostics before truncation or indexing. |
| Damping | Unsupported nonzero damping fails before mutation; zero damping and unaffected fits retain their behavior. |
| Optional position | Supported position-free prediction works; configurations needing position reject its absence explicitly. |
| Diffusion/MRF validation | Mismatches fail through both entry points and both local/non-local branches. |

Use existing applicable numerical tolerances and preserve Phase 1 regressions.
Run affected model/backend tests and the full suite for implementation changes.
Boundary-ownership changes may alter fitted models and posteriors; unchanged
full-coverage fixtures alone cannot prove the absence of such effects. Attribute
any golden difference under the existing numerical-validation process rather
than guaranteeing that every fixture remains unchanged.

Review removal of all window-dependent paths, event weighting exactly once,
GLM sufficient-statistic/exposure alignment, validation before mutation, and the
recorded damping-support inventory. This phase fixes ownership; Phase 3 separately
fixes decoding-bin ownership, and Phase 7 later changes storage/evaluation strategy.
