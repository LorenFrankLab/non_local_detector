# Plan: Local State Observation-Channel Cleanup

## Motivation

PR #20 introduced the multi-bin Local state (`local_position_std`) with two
coupled compromises:

1. **Uniform initial conditions for Local**: `p_local / n_interior_bins` per bin.
   A placeholder, not a prior. At t=0 we know where the animal is, yet
   `detector.initial_conditions_` claims uniform ignorance.

2. **`log(n_bins)` compensation in the per-timestep kernel**: the kernel is
   arithmetically rescaled to sum to `n_bins` (not 1) across bins, so that
   multiplication with the uniform IC gives total Local mass comparable to
   the legacy 1-bin case. A bookkeeping trick to paper over #1.

These are not independent. The uniform IC needs the `log(n_bins)` compensation,
and the `log(n_bins)` compensation only makes sense given the uniform IC. They
must be fixed together.

Goal: an honest two-piece model of the Local state:

- **IC** reflects "we know where the animal is at t=0."
- **Kernel** is a proper log-density of the tracked-position observation given
  the underlying bin — no arithmetic compensation.

Scientifically this aligns with the intended interpretation: the Local state
represents "spikes come from a population coding for the animal's current
tracked position, within measurement noise σ." Everything else (theta
sequences, replay, anticipatory firing driven by task state) stays Non-Local,
where it belongs.

## Design

### Core Concept

Two simultaneous, coupled changes:

**Change A — Concentrate IC at the animal's first-timestep bin.** At
`predict()` (and `estimate_parameters()`) time, replace the Local state's
rows of `initial_conditions_` with a delta at the interior bin containing
the first interpolated animal position (via snap-aware `env.get_bin_ind`).

**Change B — Remove the `log(n_bins)` compensation from the kernel.** The
per-timestep kernel becomes the log of a proper Gaussian density over
position (not a renormalized probability-like thing).

Together, the forward pass at t=0 becomes:

```
α_0(Local, b) = δ(b, animal_0_bin) · P(spikes_0 | b) · N(animal_0; bin_center_b, σ²)
```

Only b = animal_0_bin contributes. The Gaussian density evaluates at the
animal's observed position given bin center as the mean; for σ small it
peaks high (commits strongly to Local), for σ large it spreads out (hedges
toward Non-Local). Either way, σ is a physical parameter — the standard
deviation of the tracked-position measurement error — not a modeling knob.

### Architectural Rationale

#### Why IC must be predict-time, not fit-time

The IC depends on test-time animal position, which isn't known at fit time.
This breaks the current contract that `initial_conditions_` is test-data-
independent. Two implementation strategies:

1. **Override approach (chosen).** `predict()` computes a local-state IC
   from the first animal position and passes it to `_predict` as an
   optional override. `detector.initial_conditions_` (the stored array)
   stays as-is; the override applies only within the single predict call.

2. **Mutation approach (rejected).** Overwrite `detector.initial_conditions_`
   at predict time. Cleaner to inspect but contaminates the fitted model
   with test-data state; causes subtle bugs if the same model predicts on
   multiple datasets.

Mitigation for the override approach's inspection weakness: add a helper
`detector.compute_local_initial_conditions(position_time, position, time)`
that returns the predict-time IC explicitly.

#### Why Change A and Change B must move together

They're two sides of the same balance. The current code has:

```
α_0(b) = IC(b) · L_0(b)
       = (p_local / n_bins) · P(spikes|b) · kernel(b) · n_bins
       ────────────────       ─────────────────────────────
         Change A removes       Change B removes log(n_bins)
         the 1/n_bins factor
```

Remove just Change B → Local mass shrinks by n_bins, Non-Local dominates
(this is exactly the bug that commit `30aa07c` fixed by introducing the
compensation in the first place).

Remove just Change A → IC is no longer a valid probability distribution
over Local bins (`p_local` at one bin, plus `p_local / n_bins` at all
others, doesn't sum to p_local). Math ill-defined.

Remove both → Clean math, balanced forward step, honest semantics.

### Key Design Decisions

1. **Delta IC, not Gaussian-shaped IC.** We commit to the single bin
   containing the animal at t=0. A Gaussian-shaped IC would sort of
   double-count: the kernel at t=0 already provides spatial smoothing
   via the Gaussian density, so an additional Gaussian in the IC would
   just squeeze the posterior tighter without adding information.

2. **Snap-aware bin selection.** Use `env.get_bin_ind` on the first
   interpolated animal position. Inherits the PR #20 snap behavior: if
   the animal lands on an arm boundary, the delta lands on the nearest
   interior bin (not a gap bin).

3. **NaN first-position fallback.** If `position[0]` is NaN (tracking
   dropout on the initial frame), fall back to the uniform IC. The
   kernel's NaN-mask at t=0 also fires, giving a flat first timestep.
   Subsequent timesteps recover once tracking returns.

4. **Kernel as proper log-density.** Change from
   `-0.5 · d² / σ²  - logsumexp + log(n_bins)` to
   `-0.5 · log(2π σ²) - 0.5 · d² / σ²`. The normalization constant
   `-0.5 · log(2π σ²)` is a bin-independent scalar that shifts log-
   likelihoods uniformly; it does NOT affect posteriors but DOES affect
   marginal log-likelihood magnitudes (correct for model comparison).

5. **Only applies when `local_position_std is not None`.** Legacy
   (`None`) Local is single-bin; both IC and likelihood are already
   correct. No change.

6. **σ=0 becomes invalid.** A Dirac density can't be represented in log
   space. Reject `local_position_std=0` in the validator. Users who want
   effectively-delta behavior can pass `local_position_std=0.01` (or the
   smallest value smaller than any bin width — still numerically well-
   behaved). This is a semantic restriction, not a loss of capability.

7. **Total Local mass depends on σ.** Under the new semantics, Local
   total mass at t=0 is `p_local · P(spikes|animal_bin) · N(animal; animal_bin_center, σ²)`.
   The density factor varies with σ — small σ → large density at the
   animal's bin → Local commits strongly; large σ → small density → Local
   hedges. This is scientifically correct (narrow σ = confident tracking
   = strong local commitment) but it means the Local-vs-Non-Local balance
   shifts with σ. Users should calibrate σ to their actual tracking noise.

8. **Non-local penalty unaffected.** The non-local penalty is a soft
   repulsive prior, not an observation. Its code path stays exactly as-is.

### Out of Scope

Items deliberately deferred:

- **Renaming `local_position_std` → `position_tracking_std` or similar.**
  The current name is misleading (it suggests "spread of the local state"
  rather than "tracking measurement noise"), but the rename is a breaking
  API change. Doing it together with the semantic change would compound
  migration pain. Separate PR.
- **Transition-matrix formulation of Local.** See Future Extensions
  below.
- **Velocity drift, heading-conditional place fields, or any extension
  that would absorb theta sequences into Local.** Theta sequences are
  non-local by scientific design; they belong in the Non-Local states.
- **Non-local penalty refactor.** Different semantic (repulsive prior,
  not observation); separate concern.

### Future Extensions

#### Transition-matrix version of Local

The current design (kept by this plan) expresses the spatial anchor of
the Local state in the **likelihood**: at each timestep, the per-bin
likelihood gets multiplied by a Gaussian density centered at the
animal's observed position. The continuous transition `Local → Local`
is `Uniform()`, intentionally memoryless: Local carries no bin-level
memory across timesteps, only total mass.

A mathematically equivalent alternative expresses the anchor in the
**transition matrix** instead:

- `IC(Local, b) ∝ N(animal_0; bin_center_b, σ²)` (same as here, but
  normalized to integrate to p_local via a sum-to-1 proper prior).
- `T_t(Local, b | anything) ∝ N(animal_{t+1}; bin_center_b, σ²)` — the
  transition into the Local state at bin b depends on how close that
  bin is to the animal's next-timestep position. Time-varying transition.
- `log L(spikes | Local, b) = log P(spikes | bin=b)` — the likelihood
  becomes pure and state-independent across Local and Non-Local.

Posteriors under this formulation are identical to the likelihood-side
formulation (multiplication in the forward step commutes). But the
structural decomposition is cleaner: the transition governs how state
evolves (including spatial preferences), and the likelihood governs
how observations arise given state.

**Why it's future work, not this PR:**

- **Breaks `jax.lax.scan`'s static-transition assumption.** The HMM
  forward/backward in `core.py` is optimized around a single
  transition matrix that is JIT-compiled once. A time-varying
  transition matrix either forces the
  `chunked_filter_smoother_covariate_dependent` path (slower; currently
  intended for covariate-driven discrete transitions, not continuous
  per-bin spatial priors) or requires a new scan kernel that accepts
  time-indexed transitions cheaply.
- **API growth.** Needs a new continuous transition type, e.g.
  `PositionAnchored(std=σ)`, alongside `RandomWalk`, `Uniform`,
  `Identity`, `EmpiricalMovement`. That type would need serializable
  parameters, fit-time and predict-time hooks, and documentation.
- **User mental model shift.** "Local is a non-stationary HMM state
  whose transition depends on the animal's current position" is a
  heavier concept than "Local has a spatial prior in its likelihood."
  Worth paying only if the cleaner decomposition enables extensions
  that are hard under the current approach.

**What it would enable that the likelihood version makes harder:**

- **Pure, comparable observation likelihoods.** `log_likelihood`
  would be exactly `log P(spikes | state, bin)` — no observation
  channel mixed in. Downstream tools and diagnostics get cleaner data.
- **State-specific spatial dynamics.** Other non-local states could
  have their own transition priors (e.g. a Non-Local-Forward state
  biased toward bins ahead of the animal's path) without creeping
  into the likelihood layer.
- **Cleaner handling of tracking dropouts.** NaN position would
  reduce to a flat transition row without needing a separate
  likelihood-level mask.

**When to actually do it:** if/when a feature request comes in that
needs a state-specific spatial dynamics term that isn't cleanly
expressible as a likelihood multiplier. Until then, the likelihood
version is sufficient and substantially cheaper to maintain.

## Implementation

### Files

- `src/non_local_detector/models/base.py`
  - **Change A**: Add `_compute_predict_time_local_ic(position_time, position, time)`
    on `_DetectorBase`. Returns a numpy array of shape `(n_state_bins,)`
    matching `initial_conditions_`, with Local rows replaced by the delta
    and other rows copied through unchanged. Returns `None` when
    `local_position_std is None` or position is None or position[0] is NaN.
  - **Change A**: Modify `_predict(...)` to accept an optional
    `initial_conditions_override` parameter; use it when non-None, else
    fall back to `self.initial_conditions_[is_track_interior]`.
  - **Change A**: Thread the override through `predict()` and
    `estimate_parameters()` on both `SortedSpikesDetector` and
    `ClusterlessDetector`.
  - **Change B**: Rewrite `_compute_local_position_kernel` to return
    `-0.5 · log(2π σ²) - 0.5 · sq_dist / σ²` (proper Gaussian log-density),
    removing the `logsumexp(...) + log(n_bins)` renormalization.
  - **Change B**: Update the docstring to describe the new semantics
    (observation log-density, not a normalized kernel; total Local mass
    is σ-dependent).
  - **σ=0 rejection**: Update the `local_position_std` validator to
    require `> 0` (currently accepts `>= 0`).

- `src/non_local_detector/tests/models/test_local_observation_channel.py` (new)
  - Tests as listed below.

- `src/non_local_detector/tests/models/test_local_position_std.py`
  - Remove tests that asserted the old `exp sum == n_bins` invariant
    (they asserted the compensation logic directly).
  - Remove the σ=0 acceptance tests (σ=0 is now rejected).
  - Update kernel-property tests to the new log-density semantics.

### Test Plan (TDD — failing first)

1. **`test_ic_concentrates_at_animal_bin_0`**
   - Fit detector with `local_position_std=0.5`, predict on test data.
   - Assert `results.acausal_posterior.sel(state="Local").isel(time=0)`
     peaks at the bin closest to `animal_position[0]`.
   - Assert that bins more than ~3σ from the animal have posterior
     probability below a small threshold.

2. **`test_ic_respects_snap_at_arm_junction`**
   - Two-arm track from PR #20's test fixtures.
   - Animal's first position exactly on the arm-boundary edge.
   - Assert the delta lands on the arm's last interior bin.

3. **`test_kernel_is_proper_log_density`**
   - For a range of σ and animal positions, assert
     `log_kernel(b) == -0.5 * log(2π σ²) - 0.5 * d(b, animal)² / σ²`
     (bit-exact up to float32 precision).
   - Assert it matches `scipy.stats.norm.logpdf` on the distance.

4. **`test_kernel_no_longer_sums_to_n_bins`**
   - Assert `exp(log_kernel).sum(axis=1) != n_bins` in general.
     Confirms the compensation is actually removed.

5. **`test_total_local_mass_depends_on_sigma`**
   - Fit two detectors with different σ values, same data, same spikes.
   - Assert `results.acausal_state_probabilities.sel(state="Local")`
     differs meaningfully. Concretely: smaller σ → stronger Local peak
     at the animal's position at frames where the animal is well-tracked.

6. **`test_legacy_local_position_std_none_unchanged`**
   - Fit with `local_position_std=None`, predict.
   - Assert the Local state's IC is a single bin with mass p_local.
   - No override applied; no kernel changes apply (legacy 1-bin path).

7. **`test_nan_first_position_falls_back_to_uniform`**
   - Test data where `position[0]` is NaN.
   - Assert the override is `None`, uniform stored IC used.
   - Kernel's NaN-mask at t=0 fires; first timestep is flat.

8. **`test_ic_override_does_not_mutate_stored_ic`**
   - Fit, record `detector.initial_conditions_`, predict; assert the
     stored array is unchanged.

9. **`test_helper_method_returns_predict_time_ic`**
   - `detector.compute_local_initial_conditions(position_time, position, time)`
     returns the IC that would be used. Shape and values match what's
     passed internally to `_predict`.

10. **`test_zero_sigma_is_rejected`**
    - `NonLocalSortedSpikesDetector(local_position_std=0.0)` raises
      `ValidationError`. Same for `NonLocalClusterlessDetector`.

### Verification

- Run full test suite: `uv run pytest`.
- Run property tests: `uv run pytest -m property -v`.
- Run golden regression: `uv run pytest src/non_local_detector/tests/test_golden_regression.py -v`
  → **expected to shift**. Marginal log-likelihoods change (by a
  σ-dependent additive constant per timestep) and first-timestep
  posteriors shift. Requires explicit snapshot-update approval per
  CLAUDE.md.
- Run snapshot tests: `uv run pytest -m snapshot -v` → same expected
  shifts.
- Real-data notebook check at t=52_992, 100_000, 210_000: Local posterior
  concentrated near the animal's bin, not diffuse or flat.

## Pros

1. **Honest semantics.** IC reflects what we know. Kernel is a proper
   observation log-density. No compensation math.
2. **Cleaner code.** `_compute_local_position_kernel` drops the logsumexp
   renormalization and the `+log(n_bins)` addition. Fewer moving parts.
3. **Marginal log-likelihood becomes meaningful.** Users doing model
   comparison across σ values or across local_position_std=None vs a
   numeric value get values that are directly interpretable as log-
   densities (not an arbitrary constant shift from the compensation).
4. **σ gains a physical interpretation.** It's the standard deviation of
   tracked-position measurement noise. Users can estimate it from
   tracking pipeline diagnostics rather than tuning by eye.
5. **Forward-compatible.** Clears the path for later optional extensions
   (parameter rename, transition-matrix refactor) without pre-committing.
6. **Leverages PR #20 infrastructure.** Snap-aware `get_bin_ind` handles
   arm-junction edge cases for free.
7. **Theta sequences still classified as Non-Local.** No change to that
   behavior — this cleanup doesn't move scientific categories.

## Cons

1. **IC becomes predict-time data-dependent.** Breaks a cleanliness
   principle: fitted state shouldn't depend on test data. Calling
   `predict()` with different test data produces different effective ICs.
2. **`detector.initial_conditions_` is slightly misleading.** Stays
   uniform even though the model uses a delta. The helper method
   `compute_local_initial_conditions(...)` mitigates but doesn't eliminate
   the surprise for users who inspect the stored attribute.
3. **Total Local mass now σ-dependent.** Under the old semantics, the
   `log(n_bins)` compensation hid this: total Local mass after kernel
   normalization was σ-independent. Now it's not. Scientifically correct
   but requires users to understand that σ is a real hyperparameter with
   a semantic role, not just a smoothness knob.
4. **Not bit-compatible with legacy `None` or with PR #20 behavior.**
   `local_position_std=None` is untouched, but any detector fit with
   PR #20's `local_position_std > 0` and rerun under this PR gets
   different posteriors. Regression analyses using the old model need to
   be re-run or documented as "fit with the pre-cleanup kernel."
5. **Golden regression + snapshot updates required.** Needs explicit
   user approval per CLAUDE.md.
6. **σ=0 goes away.** Anyone using `local_position_std=0.0` (delta-kernel
   mode from PR #20) has to change to a small positive value. Release
   notes must flag this.
7. **Does not rename the parameter.** The name `local_position_std`
   remains conceptually slightly misleading; a follow-up PR can rename it
   to `position_tracking_std` with a migration guide.

## Effort Estimate

- Implementation: ~4–6 hours
  - Change A (IC override): ~2 hours
  - Change B (kernel density): ~1 hour
  - σ=0 validator: 30 minutes
  - Threading overrides through `_predict`, `predict`, `estimate_parameters`: ~1 hour
  - Docstring updates: 30 minutes
- Tests: ~3–4 hours
  - 10 unit tests, mostly small.
  - Updating existing tests that asserted the old normalization: ~1 hour.
- Golden/snapshot analysis + approval flow: ~2 hours
  - Running before/after, producing the diff analysis required by CLAUDE.md.
- Documentation: ~1 hour
  - Release notes, docstring updates, a paragraph in the user guide
    explaining σ semantics.
- **Total: ~1–1.5 days of focused work.**

## Dependencies

- PR #20 (`local_position_std` / gap-bin snap fix) must be merged first.
  This plan uses the snap-aware `get_bin_ind` and the multi-bin Local
  state introduced in that PR.

## Open Questions

1. Should `compute_local_initial_conditions` be a public method on the
   detector, or a private helper? Public is nicer for users but expands
   the API surface.
2. Should we add a warning or deprecation note when users pass
   `local_position_std=0.0` in the first release after this PR, before
   hard-rejecting it?
3. Snapshot test strategy: update snapshots en masse (treat as intended
   behavior change for the `local_position_std is not None` path), or
   gate the behavior behind a flag for one release to preserve existing
   behavior by default? Recommend: no flag, update snapshots, release
   notes flag the breaking change. Legacy `None` path is unchanged, so
   most users won't notice.
4. Should we also add a `position_tracking_std` alias (pointing to
   `local_position_std`) in this PR as a forward-compatibility gesture,
   or hold that for the dedicated rename PR? Recommend: hold.
