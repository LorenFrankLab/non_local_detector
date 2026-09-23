# Phase 5 — Canonical event ownership and validation gaps

> **NEEDS PROTOTYPING AGAINST SETTLED C2.** Hard group windows are removed, not
> repaired. The former half-interval window implementation and disjoint-window
> tests contradicted C2 and are withdrawn. Other validation changes retain the
> scopes below; no backend damping-rebuild architecture is added.
>
> Claims below were re-verified against `main` at `ee2cc21` (2026-09-22), with
> reproductions for the window, weighting, damping, population, and position
> findings. Line references are to that revision.

## Contracts and scope

[C2](shared-contracts.md#c2--exposure-ownership-and-spike-weights) assigns encoding
spikes by interpolated sample weights. Phase 1 made the **sorted** fit pass the
exposure mask as weights on the full timeline and remains complete. The
**clusterless** fit does not yet do this (see §1). This phase makes event
ownership consistent with C2 for both families. It does not choose C1 floors or
C3b acquisition endpoints/gap exposure. Any case requiring an unresolved C3b
choice must be identified rather than settled through an implicit window rule.

## 1. Remove hard group windows

### Current behavior (verified)

Both helpers — `_get_group_spike_data` (clusterless, `models/base.py:2908-2981`)
and `_get_group_spikes` (sorted, `:3931-3985`) — label contiguous runs of the
group mask and keep spikes in `[run_start - dt, run_end + dt]`, where `dt` is the
**global first difference** `position_time[1] - position_time[0]` taken after
NaN-position rows are dropped. The code comment says "half a time bin"; the code
widens by a full `dt`. The per-run selections are concatenated, so a spike inside
two runs' windows is **duplicated** within the group.

The two families then diverge:

| | Sorted (`fit_encoding_model`, `:4081-4114`) | Clusterless (`fit_encoding_model`, `:3072-3087`) |
|---|---|---|
| Timeline passed to backend | full `position_time` | subset `position_time[is_group]` |
| Weights passed | `is_group` (or `weights * is_group`) on the full timeline | `weights[is_group]`, or `None` → ones on the subset |
| Per-spike weight inside the backend | `interp(t, full grid, mask·w)` — fractional at transitions, 0 in gaps | `interp(t, subset grid, w)` — interpolation **bridges gaps**; the mask never reaches the event weight |

Reproduction, mask `[T,T,F,T,T]` at times `0…4`, spikes at `[1.5, 2.0, 2.5]`:

| | window spikes | per-spike weights |
|---|---|---|
| Sorted | `[1.5, 2.0, 2.0, 2.5]` | `[0.5, 0, 0, 0.5]` — duplicate is inert |
| Clusterless (`weights=None`) | `[1.5, 2.0, 2.0, 2.5]` | `[1, 1, 1, 1]` — gap spike counted **twice at full weight**, boundary spikes at 1 instead of 0.5 |
| C2 canonical | `[1.5, 2.0, 2.5]` | `[0.5, 0, 0.5]` |

The window also **discards positive-weight events** when sampling is
non-uniform: with `position_time = [0, 0.5, 2, 3, 4]` and the same mask, the
spike at t=1.2 has canonical weight 0.53 but lies outside the `dt = 0.5` window.
It likewise misplaces boundaries after dropped NaN rows or with jittered times.

Overlap alone is not the mathematical defect: at t=1.75 the mask gives weight
0.25 and its complement 0.75. Both groups legitimately receive fractional
contributions whose total is one. No disjoint-window construction can preserve
every nonzero interpolated weight.

### Requirements and falsification

- Delete run-window filtering in both helpers and select events with positive
  canonical weight within the supported recording domain. Preserve spike/feature
  alignment and retain the fractional weight itself for fitting; selecting a
  spike does not make its weight one.
- **Clusterless fit: stop subsetting the timeline.** Pass the full
  `position_time`/`position` and `weights = is_group` (or `weights * is_group`)
  to the backend, as Phase 1 did for sorted. Occupancy/exposure and the
  mean-rate denominator then come from the mask-weighted full timeline instead
  of the subset; confirm each clusterless backend (KDE, log-KDE, GMM,
  diffusion) supports zero-weight samples in its occupancy fit (the GMM drops
  them since Phase 4) and that interpolation no longer bridges gaps.
- Apply each event weight once. Where a backend already interpolates event
  weights, avoid applying them again at the detector layer. Preserve weighted
  exposure on the original position timeline.
- Prototype GLM weighted sufficient statistics explicitly. Today the GLM bins
  each spike to its left sample row (`np.digitize`) and multiplies both the
  event and exposure terms by that row's shared weight
  (`sorted_spikes_glm.py:179-187`, `:356`). That cannot reproduce interpolated
  per-event weights in general: with mask `[1,1,0,1,1]`, spikes at `[1.1, 1.9]`
  give weighted count 2.0 versus interpolated 1.0, and the required row
  multiplier differs per neuron. Use per-neuron weighted counts
  `c_i = Σ_{s ∈ row i} w_s` for the event term and the per-sample weight only
  for the exposure term, so no event is weighted twice. Carry this contract into
  Phase 6a's encoding cells and Phase 6b's duration offsets.
- Note that the GLM evaluates its design at the row's position sample, while
  KDE, diffusion, and MRF use the interpolated position at the spike time. A
  shared weighted-event reference must state which spike-position convention it
  checks; do not equate the backends' representations without testing the same
  weighted-event reference.
- Inventory equivalent ownership in KDE (`interpn` per-spike weights),
  diffusion/MRF (`pixellate_interior_fields` weighted spatial bincount), and the
  clusterless fits. Subsetting the time grid must not redefine interpolation or
  exposure.

Before editing, reproduce the clusterless double-counted gap spike, a
nonzero-weight event discarded by a hard window, and an excluded event with zero
canonical weight. Test complementary/multiple groups against explicit
interpolated weights, including fractional/EM weights and jittered times.
Restrict endpoint/gap reference cases to the declared recording contract; retain
unresolved C3b cases as blockers for that part of the work.

### Validate before pairing populations

Only the clusterless helper pairs collections: `_get_group_spike_data` zips
`spike_times` with `spike_waveform_features` using `strict=False`
(`base.py:2941-2942`) before any backend validation. Verified through the public
`ClusterlessDecoder.fit` on all four clusterless backends:

- 4 spike-time arrays with 3 feature arrays (or 3 with 4) **fits silently** with
  3 electrodes — the GMM backend's `validate_population_lengths` runs after the
  truncation and never sees the mismatch.
- A per-electrode row mismatch raises a raw numpy `IndexError` from the helper's
  boolean indexing (`base.py:2961`), not a package diagnostic. When a group has
  no training coverage (`n_groups == 0`) the helper slices `features[:0]` without
  comparing rows, so the mismatch passes unnoticed there (inferred from code).
- At predict time the GMM raises `ValidationError`; KDE, log-KDE, and diffusion
  raise a bare `ValueError` from `zip(strict=True)` for population mismatches.

The sorted helper iterates one list and cannot truncate; the sorted gap is the
predict-side count check in §4.

Validate population lengths and each electrode's spike/feature row alignment in
the detector's `fit`/`fit_encoding_model` **before** `_get_group_spike_data`,
and on the predict path for the non-GMM clusterless backends, using package
`ValidationError` diagnostics. A bare strict zip is insufficient. Reuse the
existing `validate_population_lengths` / `_validate_spike_feature_pair` helpers
where they fit, and exercise the public detector path, not only the backend
validator.

## 2. Reject unsupported encoding damping before mutation

`_apply_encoding_damping` (`base.py:1852-1874`) blends only `place_fields` and
recomputes `no_spike_part_log_likelihood` from the blend; it skips any entry
without `place_fields`. It runs only inside `_DetectorBase.estimate_parameters`,
when `estimate_encoding_model=True`, a `"Local"` state exists, and the
effective-sample-size guard allows the update.

Verified per backend under the default local path (`local_position_std=None`):

| Backend | Local path reads | Non-local path reads | Effect of the blend |
|---|---|---|---|
| `sorted_spikes_glm` | `coefficients` | `place_fields`, `no_spike_part` | Local unchanged, non-local damped; coefficients stale |
| `sorted_spikes_kde` | `occupancy_model`, `marginal_models`, `mean_rates` | `place_fields`, `no_spike_part` | Local unchanged, non-local damped |
| `sorted_spikes_diffusion` / `_mrf` | `place_fields` | `interior_log_place_fields`, `no_spike_part` | Inconsistent **within** the non-local path: stale log fields for the count term, blended fields for the no-spike term |
| all four clusterless backends | — | — | No `place_fields` key: **silent no-op** |

With `local_position_std` set, the Local state also reads `place_fields`, so
GLM/KDE predictions would be self-consistent (their other fitted state still
stale). This phase does not pursue that configuration-dependent support.

**Decision: reject any nonzero `encoding_update_damping`**, regardless of
backend, local-position configuration, or whether the update would actually be
consumed (no Local state, `estimate_encoding_model=False`, or the ESS guard
skipping it). One unconditional rule is simpler than a support matrix that
depends on runtime state, and no configuration is coherent across all fitted
state. Zero damping retains current behavior.

Validate **before** the wrapper's first `fit`: today
`ClusterlessDetector.estimate_parameters` (`base.py:3731`) and
`SortedSpikesDetector.estimate_parameters` (`:4678`) call `self.fit(...)` first,
and the `[0, 1)` range check (`:2015-2018`) runs afterwards — a rejected value
already replaced `encoding_model_` (reproduced) and raises `ValueError` rather
than `ValidationError`. Do not invent support or rebuild encoding models inside
this phase; any newly supported case needs explicit local/non-local consistency
evidence.

Test rejection from both public estimation wrappers and verify fitted state is
unchanged after the error (this test fails on current `main`). The release note
must identify the rejected parameter value and its reason, rather than
describing damping as a silent no-op. No existing test exercises damping.

## 3. Optional position in non-local clusterless prediction — tests only

The recorded defect (GMM predictor dereferencing `position.ndim` before the
`is_local` dispatch) was fixed by `4f4e862` (2026-09-15), after this plan was
drafted. Verified on `main`:

- All four clusterless predictors read position only under `if is_local:`.
- The detector boundary already requires position when needed:
  `needs_position = any(obs.is_local) or non_local_position_penalty > 0 or
  local_position_std is not None` raises `ValidationError("Missing required
  parameter: position …")` (`base.py:3230-3252` clusterless,
  `:4256-4278` sorted).
- `ClusterlessDecoder` predicts with `position=None` on every clusterless
  backend; `NonLocalClusterlessDetector` rejects it with that `ValidationError`.

Remaining work is regression coverage: public and direct `position=None`
non-local prediction for GMM, KDE, and log-KDE (only a direct diffusion call is
tested today), and a test for the missing-position `ValidationError`.
`needs_position` is also true whenever `local_position_std` is set, even with no
local state; decide whether that is intended rather than changing it silently.

## 4. Population validation in sorted diffusion and MRF

`sorted_spikes_mrf` re-exports the diffusion predictor
(`sorted_spikes_mrf.py:62-73`), so one fix covers both registered names.
`predict_sorted_spikes_diffusion_log_likelihood`
(`sorted_spikes_diffusion.py:582-728`) has no population check; sorted KDE and
GLM call `validate_population_lengths`. Verified:

- Through both `SortedSpikesDecoder` and `NonLocalSortedSpikesDetector`, a
  neuron-count mismatch raises cryptic JAX shape errors (`dot_general` on the
  non-local path, broadcasting `mul` on the local path).
- **Silent case:** called directly with `is_local=True`, a 1-neuron fit
  predicted with 0 or 2 spike trains returns finite output of the wrong
  population, because the `(n, 1)` rate array broadcasts. The public detector
  masks this only because its non-local matmul raises later.

Validate the number of spike trains explicitly against `place_fields` and
`interior_log_place_fields` before count construction or matrix multiplication,
on local and non-local paths, through both public backend names. Do not rely on
shape errors. Re-inventory current validators before adding duplicates.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Canonical ownership | Explicit per-event weights match interpolated sample weights on the full timeline for sorted **and** clusterless fits; zero-weight events contribute nothing, no event is duplicated, and no positive-weight event is discarded. |
| Partition of unity | Complementary/multiple group contributions sum to the original event weight, including fractional boundary events. Disjoint windows are not required. |
| Clusterless exposure | Occupancy and mean rate come from mask-weighted samples on the full timeline; interpolation does not bridge gaps. |
| Backend sufficient statistics | KDE/GLM/diffusion/MRF/clusterless paths apply event weights once and use the corresponding exposure; GLM weighted counts have an independent reference with a stated spike-position convention. |
| Population alignment | Collection and per-electrode row mismatches raise package diagnostics through the public fit before truncation or indexing, and through predict for every clusterless backend. |
| Damping | Any nonzero damping fails before mutation through both public wrappers with `ValidationError`; fitted state is unchanged; zero damping retains behavior. |
| Optional position | Regression tests cover existing position-free non-local prediction (GMM/KDE/log-KDE, public and direct) and the missing-position `ValidationError`. |
| Diffusion/MRF validation | Mismatches fail through both entry points and both local/non-local branches, including the direct local call that currently broadcasts silently. |

Use existing applicable numerical tolerances and preserve Phase 1 regressions.
Run affected model/backend tests and the full suite for implementation changes.
Boundary-ownership changes may alter fitted models and posteriors — the
clusterless switch to full-timeline weights changes every clusterless fit whose
group mask has gaps; unchanged full-coverage fixtures alone cannot prove the
absence of such effects. Attribute any golden difference under the existing
numerical-validation process rather than guaranteeing that every fixture
remains unchanged.

Review removal of all window-dependent paths, event weighting exactly once,
the clusterless full-timeline exposure, GLM sufficient-statistic/exposure
alignment, validation before mutation, and the damping rejection. This phase
fixes ownership; Phase 3 separately fixes decoding-bin ownership, and Phase 7
later changes storage/evaluation strategy.
