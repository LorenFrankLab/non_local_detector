# Plan: Predict-time Local State Initial Conditions

## Motivation

With the multi-bin Local state introduced in PR #20 (`local_position_std`), the
initial conditions for the Local state are **uniform across all interior bins**:
`p_local / n_interior_bins` per bin. This is a placeholder, not a prior.

Problem: at t=0 we **know where the animal is** — we have tracked position data.
Claiming uniform ICs is a fiction. A user inspecting `detector.initial_conditions_`
expecting a sensible prior sees a meaningless spread. The actual prior lives in the
per-timestep kernel applied to the likelihood, which obscures intent.

This plan concentrates the Local state's IC at the animal's first-timestep bin,
making `initial_conditions_` an honest representation of what we know before spike
observations.

## Design

### Core Concept

At `predict()` (and `estimate_parameters()`) time, replace the Local state's rows
of `initial_conditions_` with a delta concentrated at the interior bin containing
the animal's first interpolated position. Use the existing snap-aware
`Environment.get_bin_ind` so the delta always lands on an interior bin, even when
the animal sits at an arm junction (the fixed edge case from PR #20).

The forward pass at t=0 becomes:

```
α_0(Local, b) = δ(b, animal_bin_0) · P(spikes_0 | b) · kernel(b | animal_0)
```

For `b != animal_bin_0`, α is zero. For `b == animal_bin_0`, it picks up the peak
of the spatial likelihood at that bin. Subsequent timesteps evolve via the
continuous transition (`Uniform()` for Local, per the PR #20 choice) and the
per-timestep kernel continues to anchor Local to the animal.

### Architectural Rationale: Predict-time, not Fit-time

The IC must be computed at predict time because it depends on the **test-time**
animal position, which isn't known at fit time. This is a departure from the
current contract where `initial_conditions_` is a fit-time attribute independent
of test data.

Two implementation strategies considered:

1. **Override approach (chosen):** `predict()` computes a local-state IC delta from
   the first animal position and passes it into `_predict` as an optional override.
   `detector.initial_conditions_` (the stored array) stays uniform; the override
   only applies within the single predict call.

2. **Mutation approach (rejected):** Overwrite `detector.initial_conditions_` at
   predict time. Cleaner to inspect but contaminates the fitted model with
   test-data state; causes subtle bugs if the same model predicts on multiple
   datasets.

The override approach preserves fitted-model purity at the cost of making
`detector.initial_conditions_` slightly misleading (it's uniform even though the
model actually uses a delta). Mitigation: add a helper
`detector.compute_local_initial_conditions(position_time, position, time)` that
returns the predict-time IC explicitly, so users inspecting the prior can see it
on demand.

### Key Design Decisions

1. **Delta IC, not Gaussian-shaped IC.** We commit to the single bin containing
   the animal at t=0. Rationale: the per-timestep kernel already provides spatial
   smoothing at t=0 (it fires at t=0 like any other timestep), so a Gaussian IC
   would double-apply the smoothing. A Gaussian IC + kernel-skip-at-t=0 special
   case is more code for marginal benefit — the delta-IC + kernel-at-every-t
   combination gives essentially the same first-timestep posterior as a
   Gaussian-IC + no-kernel-at-t=0 combination for all reasonable σ.

2. **Snap-aware bin selection.** Use `env.get_bin_ind` on the first interpolated
   animal position. This inherits the PR #20 snap behavior: if the animal lands
   on an arm boundary, the delta lands on the nearest interior bin (not a gap
   bin). No special-casing needed here.

3. **NaN first-position fallback.** If the animal's first position is NaN
   (tracking dropout on the initial frame), fall back to the uniform IC. This
   matches the kernel's NaN handling and avoids placing the delta at an arbitrary
   default bin.

4. **Only applies when `local_position_std is not None`.** Legacy (`None`) Local
   is single-bin; IC is already correct (p_local mass on the one Local bin). No
   change.

5. **Mass preservation.** The delta IC carries the same total mass as the
   uniform IC it replaces. `delta · p_local` sums to `p_local`, matching the
   Local state's discrete initial probability. No normalization gymnastics.

6. **Kernel stays in the likelihood.** No removal or modification of the
   per-timestep kernel, including its `log(n_bins)` mass-balance. This keeps the
   change scoped: no modifications to `_compute_local_position_kernel` or the
   non-local penalty path. The numerical impact is confined to the first
   timestep's posterior.

### Out of Scope

Items deliberately deferred:

- **Removing the `log(n_bins)` mass-balance from the kernel.** That requires
  rethinking the relationship between IC and likelihood together; would be a
  separate observation-channel reframing PR.
- **Gaussian-shaped IC.** Considered but rejected (see Design Decision #1).
- **Moving the kernel from likelihood to transition matrix.** Larger refactor,
  breaks `jax.lax.scan`'s static-transition assumption. Separate plan.
- **Adaptive (velocity-aware) IC.** Future extension.

## Implementation

### Files

- `src/non_local_detector/models/base.py`
  - Add `_compute_predict_time_local_ic(position_time, position, time)` on
    `_DetectorBase`. Returns a numpy array of shape `(n_state_bins,)` matching
    `initial_conditions_`, with Local rows replaced by the delta and other
    rows copied through unchanged. Returns `None` when
    `local_position_std is None` or position is None.
  - Modify `_predict(...)` to accept an optional `initial_conditions_override`
    parameter; use it when non-None, else fall back to
    `self.initial_conditions_[is_track_interior]` as today.
  - Thread the override through `predict()` and `estimate_parameters()` on both
    `SortedSpikesDetector` and `ClusterlessDetector`.

- `src/non_local_detector/tests/models/test_predict_time_local_ic.py` (new)
  - Tests as listed below.

### Test Plan (TDD — failing first)

1. **`test_ic_concentrates_at_animal_bin_0`**
   - Fit detector with `local_position_std=0.5`, predict on test data.
   - Assert `results.acausal_posterior.sel(state="Local").isel(time=0)` peaks at
     the bin closest to `animal_position[0]`.
   - Assert that bins more than ~3σ from the animal have posterior probability
     below a small threshold.

2. **`test_ic_respects_snap_at_arm_junction`**
   - Use the two-arm track from PR #20's test fixtures.
   - Animal's first position is exactly on the arm-boundary edge (the float
     equality bug case).
   - Assert the delta lands on the arm's last interior bin, not a gap bin.

3. **`test_legacy_local_position_std_none_unchanged`**
   - Fit with `local_position_std=None`, predict.
   - Assert the Local state's IC is a single bin with mass p_local (legacy
     1-bin semantics).
   - No override should be applied in this path.

4. **`test_nan_first_position_falls_back_to_uniform`**
   - Test data where `position[0]` is NaN.
   - Assert the override is `None`, and the uniform stored IC is used.
   - The kernel's NaN-mask at t=0 also fires, so the first timestep is a uniform
     flat kernel. No crash.

5. **`test_ic_override_does_not_mutate_stored_ic`**
   - Fit, record `detector.initial_conditions_`, predict, assert stored
     `initial_conditions_` is unchanged (bit-identical).

6. **`test_helper_method_returns_predict_time_ic`**
   - `detector.compute_local_initial_conditions(position_time, position, time)`
     returns the IC that would be used. Assert shape and values match the
     override passed internally to `_predict`.

### Verification

After implementation:
- Run full test suite (`uv run pytest`).
- Run property-based tests (`uv run pytest -m property -v`).
- Run golden regression (`uv run pytest src/non_local_detector/tests/test_golden_regression.py -v`) — **expected to shift** since first-timestep posteriors change. Requires explicit snapshot-update approval per CLAUDE.md.
- Run snapshot tests (`uv run pytest -m snapshot -v`) — same expected shift.
- Real-data notebook check: `results["acausal_posterior"].sel(state="Local").isel(time=0)` should have near-zero probability except at the bin near the animal's first position.

## Pros

1. **Matches user intuition.** `initial_conditions_` (via the helper method)
   reflects what we actually know at t=0.
2. **Small, localized change.** ~50–80 lines, one new helper, one new optional
   argument on `_predict`. No HMM core or kernel math changes.
3. **Leverages PR #20 infrastructure.** Reuses the snap-aware `get_bin_ind` so
   arm-junction edge cases are handled for free.
4. **No breaking change for `local_position_std=None`.** Legacy path untouched.
5. **Forward-compatible.** Doesn't block the later observation-channel refactor;
   in fact, it clarifies the IC/likelihood split that refactor depends on.

## Cons

1. **IC becomes predict-time data-dependent.** Breaks a cleanliness principle:
   fitted state shouldn't depend on test data. Calling `predict()` with
   different test data produces different effective ICs (though not different
   stored ICs with the override approach).
2. **`detector.initial_conditions_` is slightly misleading.** It stays uniform
   even though the model uses a delta. Users must call
   `compute_local_initial_conditions(...)` to see the real prior. Documentation
   can mitigate but not eliminate the surprise.
3. **Golden regression + snapshot updates required.** First-timestep posterior
   values change, which ripples into acausal posteriors throughout the
   timeseries (weakly, since IC decays with transitions, but detectably).
   Explicit user approval needed.
4. **`log(n_bins)` mass-balance hack remains.** The kernel in the likelihood
   still compensates for an assumed-uniform IC. With a delta IC this creates a
   small scaling asymmetry at t=0 (the delta is `p_local` at one bin rather
   than `p_local/n_bins`, and the kernel adds `log(n_bins)` on top). This
   doesn't break math — normalization per forward step absorbs it — but it's a
   known cosmetic issue that stays.
5. **Smoothing at t=0 slightly reduced.** The delta commits to one bin; for
   noisy tracking, a Gaussian-IC alternative would handle that tracking noise
   better at t=0 (subsequent timesteps recover via the kernel).
6. **Doesn't address the deeper concerns.** The observation-channel reframing
   and the kernel-in-transition alternative are still future work.

## Effort Estimate

- Implementation: ~2–4 hours
- Tests: ~2–3 hours (6 unit tests + golden/snapshot refresh)
- Golden/snapshot analysis + approval flow: 1 hour
- Documentation: 30 minutes
- **Total: ~half a day of focused work**

## Dependencies

- PR #20 (`local_position_std` / gap-bin snap fix) must be merged first. This
  plan uses the snap-aware `get_bin_ind` and the multi-bin Local state
  introduced in that PR.

## Open Questions

1. Should `compute_local_initial_conditions` be a public method on the
   detector, or a private helper? Public is nicer for users but expands the
   API surface.
2. Should we add a warning when the override changes the first-timestep
   posterior by more than a threshold vs the uniform-IC baseline? Helps users
   understand the change but adds runtime cost.
3. Snapshot test strategy: do we update snapshots en masse (treat this as a
   behavior change with scientific justification), or gate the override behind
   a `use_predict_time_ic=True` flag so existing behavior is preserved by
   default? The flag adds API noise but lets the feature land without a large
   snapshot churn. Recommend: no flag, update snapshots, treat as intended
   behavior change for `local_position_std is not None` path (legacy `None`
   path untouched so most snapshots shouldn't move).
