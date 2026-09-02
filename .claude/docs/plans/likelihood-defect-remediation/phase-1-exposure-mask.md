# Phase 1 — Sorted fits honour the exposure mask

> **PROTOTYPED AND PASSING** on branch `fix/sorted-spikes-exposure-mask`.
> The spy test fails on `main` (`assert None is not None` — "the detector dropped
> the exposure mask") and passes after the two-file change below. Full suite:
> **1262 passed, 3 skipped**; golden regressions **4 passed** unchanged; ruff and
> format clean. Not committed — awaiting review.
>
> Corrections made during prototyping, versus what this file originally said:
> - The fractional-boundary-weight test is **KDE-only**. The GLM bins whole spikes
>   and weights sample *rows* (`sorted_spikes_glm.py:339`), so it cannot express a
>   fractional per-spike weight; asserting one there would be meaningless.
> - "Occupancy is exactly 0 outside the group" is **not assertable** — Gaussian
>   KDE occupancy and EPS-floored fields are never exactly zero. The shipped test
>   asserts a peak-in-range / peak-out-of-range ratio instead.
> - Zero exposure splits into three cases that must **not** be conflated. Measured:
>   a structurally absent group (all-zero weights) returned `NaN`; a zero-spike
>   unit with real exposure already returned `log(EPS)`; a low-mass EM update
>   fitted normally. Only the first was broken, and the guard now makes it agree
>   with the second.

Zero-exposure handling is included here (not deferred to phase 4) because this
phase is what first makes all-zero-exposure groups reachable. Without it, phase 1
is not independently shippable.

## Problem

`SortedSpikesDetector.fit_encoding_model` computes the selection mask and drops
it when the caller supplied no explicit weights:

```python
# src/non_local_detector/models/base.py:3920-3926
is_group = is_training & is_encoding & is_environment
if weights is not None:
    group_weights = np.where(is_group, weights, 0.0)
else:
    group_weights = None          # <-- mask discarded
```

then passes **full** position arrays with **group-filtered** spikes
(`models/base.py:3946-3957`). Each fit substitutes uniform weights
(`sorted_spikes_kde.py:138`, `sorted_spikes_glm.py:322`,
`sorted_spikes_diffusion.py:463`, `sorted_spikes_mrf.py:766`), so occupancy
accumulates over held-out samples, other environments, and other encoding groups.

The clusterless path already subsets correctly (`models/base.py:2980-2989`).

Reproduced — 40 s at 100 Hz, group = first half, one neuron:

```
                            mean_rate    (disjoint positions)
as shipped (weights=None)     0.05000     <- 2x low
weights=is_group              0.10000
subset arrays (truth)         0.10000

with OVERLAPPING coverage, place fields do not cancel:
  KDE peak: 0.12750 vs truth 0.21251   -> 40% low
  GLM peak: 0.11650 vs truth 0.21281   -> 45% low   (excluded rows enter as
                                                     observed zero-spike samples)
```

## Contracts referenced

- [C2 — Exposure ownership](shared-contracts.md#c2--exposure-ownership-and-spike-weights),
  **only** the settled part: `weights[i]` is the exposure of position sample `i`,
  and the detector must never pass `weights=None` when it has computed a mask.

This phase does **not** depend on C1 (withdrawn) or C3 (unsettled), and does not
touch the hard-window machinery C2 now says to delete — that is phase 5. It
changes one expression in `models/base.py` plus a zero-exposure guard, and is
executable while the rest of the plan is blocked.

Scoped out of this phase deliberately: fractional event ownership for the GLM.
The GLM bins whole spikes and weights *sample rows*, so it cannot express a
fractional per-spike weight; passing the mask as sample weights is well-defined
for it and is all this phase requires. Canonical fractional ownership is a
phase-5 concern and may need weighted sufficient statistics there.

## Falsification

Before implementing, write and run the detector-level test from the validation
table. It must **fail** against current `main`. If it passes, the test is not
reaching the defect — most likely it constructs weights by hand and passes them
to a backend that already handles them correctly, rather than exercising the
detector's `weights=None` path. Fix the test before touching the fix.

## Tasks

### 1. Pass the mask as exposure weights

Replace `models/base.py:3921-3926` with:

```python
            # The mask is this model's exposure: a sample outside the selection
            # must contribute no occupancy, because only in-group spikes are
            # counted. weights=None means "uniform over every supplied sample",
            # which is only true when the mask selects everything.
            group_weights = (
                is_group.astype(float) if weights is None else weights * is_group
            )
```

Leave the call at `:3946-3957` otherwise unchanged — full arrays are correct
*given* correct weights, and keeping them full preserves alignment with the
interpolation grid used for per-spike weights ([C2](shared-contracts.md#c2--exposure-ownership-and-spike-weights)).

### 2. Define zero exposure for the GLM

`fit_poisson_regression` initializes from `jnp.average(spikes, weights=weights)`
(`sorted_spikes_glm.py:191`), which is `0/0` when a group has no exposure —
newly reachable after task 1. Reproduced: coefficients `[nan, 0., 0.]`.

Insert before the `avg_rate` computation:

```python
    weight_sum = float(jnp.sum(weights))
    if weight_sum <= 0.0:
        # No effective exposure (an encoding group with no training coverage, or
        # zero posterior mass during EM). The rate is unidentified; return an
        # intercept-only model at the EPS floor so the caller gets a defined
        # near-zero place field rather than NaN coefficients.
        return jnp.concatenate(
            [jnp.asarray([jnp.log(EPS)]), jnp.zeros(design_matrix.shape[1] - 1)]
        )
```

The KDE path already handles this: `weighted_mean_rate` returns `0.0` when
`weight_sum <= 0` (`common.py:202`). The MRF already guards at
`sorted_spikes_mrf.py:825`. Confirm the diffusion path — if it lacks a guard, add
the equivalent here rather than deferring.

### 3. Confirm every sorted backend consumes the weights

Read and confirm (no edit expected) that each fit routes `weights` into
occupancy, spike density, and mean rate: `sorted_spikes_kde.py:138-141,205-212`;
`sorted_spikes_glm.py:322-325,340-345`; `sorted_spikes_diffusion.py:463`;
`sorted_spikes_mrf.py:766`. If any ignores `weights`, stop and report — that is a
separate defect.

### 4. CHANGELOG

`Fixed`: sorted-spike encoding models accumulated occupancy over held-out
samples and other environments / encoding groups whenever `weights` was not
supplied explicitly, biasing place fields low (measured 40–45% on overlapping
coverage). Fits with one environment, one encoding group, and `is_training`
all-`True` are unaffected. Also: a sorted GLM fit for a group with no training
coverage returned NaN coefficients and now returns an explicit zero-rate model.

## Validation

**Detector-level** (this is the test that actually reaches the defect) in
`src/non_local_detector/tests/models/`:

| Test | Asserts |
|---|---|
| `test_detector_passes_mask_as_weights` | Monkeypatch the registered fit function in `_SORTED_SPIKES_ALGORITHMS` with a spy; assert the detector calls it with `weights` equal to `is_group.astype(float)`, not `None`. Fails on `main`. |
| `test_two_encoding_groups_do_not_contaminate` (slow) | Two encoding groups over disjoint position ranges: each group's place field peak is within its own range, and the ratio of peak-in-range to peak-out-of-range exceeds 10. **Not** "occupancy is exactly 0 outside" — Gaussian KDE occupancy and EPS-floored fields are never exactly zero. |
| `test_partial_training_mask_rate` (slow) | With `is_training` selecting half the samples, `mean_rates[0]` equals in-group spikes / in-group samples within `rtol=1e-6`. |

**Likelihood-level** in `tests/likelihoods/test_sorted_spikes_kde.py` and
`test_sorted_spikes_glm.py`:

| Test | Asserts |
|---|---|
| `test_mask_weights_match_subset_fit` | Full arrays with `weights=mask` equals subset arrays with `weights=None`, `rtol=1e-5`. **Spikes must be placed at least one sample interval away from every mask transition** — the invariant is exact only there ([C2](shared-contracts.md#c2--exposure-ownership-and-spike-weights)). |
| `test_boundary_spike_weight_is_fractional` | A spike within one sample of a transition receives a weight strictly between 0 and 1, documenting why the invariant is conditional. |
| `test_glm_zero_exposure_returns_eps_model` | All-zero weights give finite coefficients and an EPS-floor place field; no NaN. |

```bash
uv run pytest src/non_local_detector/tests/likelihoods src/non_local_detector/tests/models -q
uv run pytest src/non_local_detector/tests/test_golden_regression.py -q
uvx ruff check src/non_local_detector/likelihoods/ src/non_local_detector/models/
```

Goldens are expected to **pass unchanged** — they fit a single group with full
training coverage, where `is_group` is all-`True` and `is_group.astype(float)` is
exactly the previous uniform weighting. If one moves, that fixture has a
non-trivial mask; attribute the diff before requesting approval.

## Review

Dispatch `code-reviewer`. Ask specifically whether the spy test genuinely fails
on `main` (run it there), and whether any sorted backend treats `weights` as
decoration rather than exposure.
