# Phase 6b — Rates in Hz and duration-scaled intensities

**Worktree status (2026-10-05):** implemented and independently reviewed against
the frozen accepted 6a/6c checkpoint, together with 6d. Physical calibration,
support, metadata, and persistence acceptance checks pass. Three numerical
snapshot corrections were explicitly approved and applied; all eight
snapshots pass and package acceptance is complete. Source is committed at
`b63ddc7`; external rollout remains pending. See the
[implementation record](#implementation-and-review--2026-10-05) and
[scope record](phase-6-worktree-scope.md).

> **C3b RESOLVED (2026-09-25) — IMPLEMENTED, REVIEWED, ACCEPTED, AND COMMITTED.** Depends on 6a/6c and ships
> atomically with 6d's model-unit validation and metadata plumbing. Phase 5
> (canonical event weights, clusterless full-timeline weights) is merged. By the
> user's rollout decision this phase also carries C3b's encoding support
> (valid intervals, supported half-cells, NaN breaks in support, the seven
> encoding clip sites from 6a) and exposure in seconds, plus C3b item 4's
> unit-bearing floors and GLM normalization/penalty migration. Acceptance:
> the same physical recording encoded at different position sampling rates
> recovers the same known firing rate on all eight backends, then decodes at a
> different bin width. The former unexecuted formulas and
> constant-posterior-rescale criterion are replaced by the unit requirements
> below.
>
> Re-verified against `main` at `ee2cc21` (2026-09-22) by fitting every
> registered backend; line references are to that revision.

## Recorded problem

`weighted_mean_rate` (`common.py:194-211`) returns spikes per weighted position
sample; its docstring misnames the denominator "weighted occupancy time". All
eight registered encoding fits accept `sampling_frequency` and none reads it;
the detector passes it unconditionally (`base.py:3085`, `:4110`). The original
audit recorded:

```text
position fs=30 Hz:  mean_rate=0.166667/sample -> 5 Hz
position fs=500 Hz: mean_rate=0.010000/sample -> 5 Hz
encode at 30 Hz, decode at 500 Hz:
  expected count per decode bin: 0.01
  model's summed intensity:     0.1735 (~17.4x)
```

Reproduced at `ee2cc21` on a different fixture (homogeneous 5 Hz unit, 30 Hz
position, 2 ms decode rows): every registered backend gives an expected count of
0.168–0.170 per row (~16.9×); no-spike gives the correct 0.0100. No backend uses
the decode bin width.

Updating that helper alone is insufficient. GLM coefficients (a per-sample
count regression with no offset, `sorted_spikes_glm.py:310`, `:359`) and MRF
exposure offsets (weighted sample counts, `sorted_spikes_mrf.py:811`) carry
sample units through other paths. Both diffusion backends normalize occupancy
to a density, so their sample unit enters only through `weighted_mean_rate`, as
in KDE/GMM. MRF also stores helper `mean_rates` that prediction never uses.
No-spike rates are already documented in Hz (`no_spike.py:58-60`) and must not
be converted twice; no-spike applies one scalar `median(diff(time))` to every
row (`no_spike.py:26-32`, `:116`).

## Contracts and inventory

Use [C3](shared-contracts.md#c3--time-vocabulary) for exposure/decoding intervals,
C2 for weighted events, and C1 for the applicable floor policy. Complete the
following inventory against the actual code before choosing implementation:

Pre-fix state at `ee2cc21` (fill in the post-fix columns while prototyping):

| Backend | Fit exposure today | Stored unit-bearing keys | Event term | Expected-count term | Local-path consumer | Extra-key dispatch |
|---|---|---|---|---|---|---|
| sorted KDE | Σ sample weights (`:207`) | `mean_rates`, `place_fields` = mean_rate·marg/occ, EPS floor (`:223-239`), `no_spike_part_log_likelihood` | `xlogy(n, place_field)` (`:413`) | `−Σ place_fields` (`:418`) | recomputes from `mean_rates`·marginal/occupancy models (`:382-391`) | strict (`TypeError`) |
| sorted GLM | one row per position sample, no offset (`:310`, `:359`) | `coefficients`, `place_fields` = exp(Xβ) (`:365-376`) | `xlogy` (`:523`) | `−Σ place_fields` (`:528`) | `coefficients` (`:496`) | strict |
| sorted diffusion | Σ weights (helper) | `mean_rates`, `place_fields`, `interior_log_place_fields` (`:344-379`); `occupancy` is a density | `counts @ interior_log_place_fields` | `−no_spike_part` (`:726`) | `place_fields` interpolation (`:197-247`) | absorbs (`**_encoding_extras`) |
| sorted MRF | offset = weighted sample counts | `place_fields` = exp(eta); `occupancy` raw sample counts; `mean_rates` unused | shared with diffusion | shared | shared | absorbs |
| clusterless KDE / log-KDE | Σ weights | `mean_rates`, `summed_ground_process_intensity` (`clusterless_kde.py:315-325`; `_log.py:1524-1535`) | log(mean_rate·marg/occ) (`clusterless_kde.py:98`; `_log.py:94`) | `−summed_gpi`; local `mean_rates`·gpi/occ (`clusterless_kde.py:680`; `_log.py:1975`) | — | strict |
| clusterless GMM | Σ weights | `mean_rates` (EPS-clipped, `:489-491`), `summed_gpi` via `_ground_process_intensity` (`:119-148`) | `safe_log(mean_rate)` + log ratio (`:759`, `:953`) | `−summed_gpi` (`:719`); local `:963` | — | absorbs (`**kwargs`) |
| clusterless diffusion | Σ weights | `mean_rates`, `summed_gpi` (`:434-437`) | `safe_log(mean_rate·p/occ)` (`:688`, `:808`) | `−summed_gpi` (`:702`); local nearest-bin lookup (`:579`) | — | absorbs (`**encoding_model`) |
| no-spike | n/a | constructor parameter in Hz | `xlogy(n, λ·median dt)` | `λ·median dt` (`no_spike.py:116-134`) | duration prepared once per prediction (`base.py:117`) | bypasses dispatch |

Clusterless fits receive the subset `position_time[is_group]` while sorted fits
receive full-timeline mask weights (until Phase 5 unifies them); exposure cells
must be computed before subsetting or under the C3b gap policy. Occupancy
estimators weight each sample equally, so with jittered or gapped timestamps the
occupancy *shape* is sample-weighted, not time-weighted; converting only the
scalar exposure does not fix that — decide it explicitly.

Target exposure is seconds and stored rate fields are Hz. For a Poisson count
likelihood with rate `r` and duration `dt`, the expected count is `r × dt`.
Record the corresponding event `log(dt)` contribution and every retained/omitted
normalization constant for each backend, including marked point-process paths.
Establish cross-state comparability: today spiking states use `log(dt_enc)`
implicitly and no-spike uses `log(dt_dec)`, so they agree only when the two
intervals are equal.

## Falsification and prototype requirements

- Use controlled fixtures with known physical exposure and event counts to
  reproduce calibration errors. A backend already passing a reference test is
  not evidence that the test is invalid; no-spike already uses Hz. Classify each
  inventory row by its actual pre-fix behavior.
- Calculate encoding exposure on the original acquisition timeline as weighted
  sample-cell durations according to resolved C3b. Do not infer it from weighted
  sample count times median spacing or bridge dropped-tracking gaps implicitly.
- Prototype separately: GLM duration offsets (update both `coefficients` and
  `place_fields`; the objective divides by the weight sum,
  `sorted_spikes_glm.py:186-188`, so the effective `l2_penalty` changes if that
  becomes seconds), MRF exposure offsets (and its per-sample numerical bounds:
  `1e-6` initial rate and `1e-9` total occupancy at `sorted_spikes_mrf.py:267-268`,
  `_ETA_CLIP = 30` at `:80`), and helper-based mean rates (KDE, log-KDE, GMM,
  both diffusion backends, and the unused MRF `mean_rates`). Preserve Phase 5 weighted-event
  statistics and apply event weights once. Store fields and coefficients with
  explicit units rather than multiplying an old per-sample fit by an extra `dt`.
- Apply duration factors consistently in local/non-local event and ground-process
  terms, no-spike (replace the scalar `_no_spike_time_bin_size` hook,
  `base.py:106-118`, with row-sliced durations), chunked paths, and missing rows.
  No likelihood backend consumes covariates.
- Inventory rate floors, zero-exposure fallbacks, and regularization under the
  new units. C1 may need an explicit unit interpretation; do not silently turn
  a per-sample numerical bound into the same numeric Hz bound. The EPS rate
  floors are listed in the C1 inventory (shared-contracts.md).
- Derive metadata from fit, validate its meaning through 6d, and pass only the
  intended predictor arguments. No new key may break strict predictor signatures.
- Audit each encoding-fit `sampling_frequency` argument. Remove ignored arguments
  where exposure now comes from timestamps; retain and document any genuinely
  used argument. Clusterless kwargs are not filtered by signature, so a leftover
  key raises `TypeError` (except GMM); about 109 test lines in 32 files pass
  `sampling_frequency=`. The `clusterless_kde_log.py:1410-1421` docstring
  wrongly says decode bins are built at `sampling_frequency` and that per-sample
  rates are intentional; rewrite it. Detector `sampling_frequency` is currently
  read only by `calculate_time_bins` (no internal callers) and the ignored fit
  arguments; no transition code reads it. Decide its role with 6a/6c.
- Other consumers of stored fields: `visualization/static.py:128` reads
  `place_fields`; many tests call predictors directly with explicit
  `mean_rates=`.

## Calibration and preservation requirements

On a controlled uniform fixture with encoding interval `dt_enc`, a stored
per-sample rate `r_old` corresponds to `r_hz = r_old / dt_enc`. The expected count
for a decode interval is `r_old × dt_decode / dt_enc`.

When `dt_decode == dt_enc`, expected counts and the corresponding Poisson
likelihood are preserved if ownership, exposure, evaluation points, floor
behavior, and likelihood constants are otherwise identical. The **stored rate**
changes units; a posterior does not acquire a common multiplicative `dt` factor.
Normalized posteriors cannot be validated by such a rescale.

| Coverage | Required evidence |
|---|---|
| Sampling-rate calibration | Matched physical-exposure/event fixtures on different position grids recover the same rate units; do not require identical unconstrained spatial fits after changing sample locations. |
| Decode duration | At fixed rate/position, expected counts scale with duration; event terms use the same declared time convention. |
| Unequal encode/decode intervals | Independent reference reproduces the expected count in the recorded 30 Hz → 500 Hz case; add an end-to-end test (none exists) whose summed intensity matches the true expected count within 5% (target from the withdrawn draft). |
| GLM/MRF/diffusion | Stored fields/coefficients have Hz meaning; keep `test_fit_poisson_regression_preserves_small_positive_exposure` (counts per row, rtol 1e-5) passing and add a new Hz known-rate GLM test, testing offsets independently. |
| Endpoints/gaps | Total exposure and excluded intervals follow resolved C3b, including short inputs and missing tracking. |
| Equal intervals | Expected counts and likelihoods agree under the conditions above; compare stored units separately from HMM results. |
| Direct nonuniform likelihoods | Both registries and no-spike use per-row durations; detector-level HMM calls reject nonuniform grids through 6c. |
| Model compatibility | Every newly stamped fit is usable and old rate units are rejected through the atomic 6d change. |

Run all affected backends and the full suite, including snapshot/golden checks.
Attribute actual changes to unit conversion, exposure, floor interpretation, and
other previously reviewed time migration effects. Not every golden necessarily
changes: the golden fixtures decode on the position grid (dt_dec == dt_enc,
`test_golden_regression.py:180-182`, `:301-303`, `:374-376`), so expect
preservation up to float32 rounding from the r/dt·dt round trip and any change
in floor interpretation; state the tolerance used to call them unchanged. Apply the
existing numerical-change approval process before modifying references or bounds.

The release note must describe Hz storage, duration scaling, any removed fit
arguments, and required refitting of incompatible models. Review the completed
backend matrix and 6d compatibility tests together.

## Implementation and review — 2026-10-05

The user requested 6a completion/review before 6b. The separate accepted baseline
is `/private/tmp/nld-phase6a-accepted`, not the earlier mixed worktree. All rate
and encoding-support changes below are attributed to 6b/6d against that source.
The implementation is committed at `b63ddc7`. The user approved all three snapshot
corrections on 2026-10-05; they are applied and all eight snapshots pass.

### Completed backend inventory

`E = Σ w_i ∫ basis_i(t) dt` is weighted exposure seconds on the original
recording support. Event weights remain dimensionless and are applied once.
`d_j` is the actual duration of decode row j. Occupancy KDE/GMM/diffusion fits use
the seconds weights, so occupancy shape also accounts for irregular sampling.

| Backend | Fit/exposure and stored units | Event term | Ground/count term | Local consumer / metadata |
|---|---|---|---|---|
| Sorted KDE | Weighted events / E; `mean_rates`, `place_fields` Hz | `xlogy(n, field_Hz * d_j)` | `-d_j * sum(fields_Hz)` | KDE rate ratio uses the same Hz means and seconds occupancy; explicit marker keywords |
| Sorted GLM | Exposure-offset Poisson fit; coefficients define log-Hz; fields Hz | `xlogy(n, exp(Xβ) * d_j)` | `-d_j * sum(exp(Xβ))` | Same fitted design/coefficient path; explicit marker keywords |
| Sorted diffusion | Seconds occupancy; weighted events / E; fields Hz, cached interior fields log-Hz | `counts @ log(fields_Hz)` plus total counts `log(d_j)` | `-d_j * sum(fields_Hz)` | Hz field interpolation; marker checked by shared predictor |
| Sorted MRF | Graph-bin seconds exposure offsets; η defines log-Hz; occupancy seconds | Shared diffusion Poisson predictor | Shared diffusion ground term | Shared local interpolation; fit emits marker/exposure |
| Clusterless KDE | Weighted events / E; means and summed ground fields Hz | Mark intensity times `d_j`; native density floors evaluated at the 2 ms reference then duration conversion | `-d_j * summed_ground_Hz` | Local ground/mark KDE uses Hz means; explicit marker keywords |
| Clusterless log-KDE | Same exposure/rate contract as probability KDE | Same physical duration conversion in log space | Same Hz-duration ground term | Local log-KDE uses Hz means; explicit marker keywords |
| Clusterless GMM | Weighted events / E; means and summed ground fields Hz | `log(mean_Hz) + log(joint/occupancy) + log(d_j)` per event | `-d_j * summed_ground_Hz` | Same physical units in local mixture ratios and ground calculation; recognized marker checked |
| Clusterless diffusion | Seconds occupancy; weighted events / E; means and summed ground fields Hz | `log(mark_intensity_Hz) + log(d_j)` per event | `-d_j * summed_ground_Hz` | Local ground lookup and mark interpolation retain the same units; marker checked |
| No-Spike | Constructor rate was already Hz; no registry fit dictionary | `xlogy(n, rate_Hz * d_j)` | `-population_rate_Hz * d_j` | Full-grid durations prepared once and sliced by global rows; no second unit conversion |

The historical `no_spike_part_log_likelihood` key stores the sum of Hz rates,
not a log likelihood. All predictors retain their prior omission of count
factorial constants; no count-factorial term is added. Gaussian mark-density
normalization is retained. The event-duration factor is common to all
states for a given observation, while the integrated-rate penalty depends on
state/position and can change the posterior.

### Support, numerical units, and review corrections

- `EncodingSupport` validates the original timestamp/position timeline before
  masking. Uniform defaults cover N*dt; acquisition bounds clip the original
  interpolation basis. Irregular or disconnected tracking requires explicit
  ordered intervals. NaNs split support; each declared segment needs a finite
  sample. Endpoints are held within encoding segments. Adjacent segments remain
  separate and own their shared boundary once on the right.
- Seconds occupancy weights and dimensionless interpolated event weights are
  separate. Analytic full/clipped/NaN/disconnected/adjacent tests exercise all
  eight backend fits. The ignored fit `sampling_frequency` argument is removed.
- Unit-bearing EPS floors are divided by the historical 0.002 seconds; density
  floors retain their original units. Probability/log-KDE private primitives
  keep their native safeguards and public predictors convert their evidence
  consistently. This preserves the deferred C1 policy.
- GLM objective normalization and L2 penalty use seconds: default 0.5 equals
  the former 0.001 at 500 Hz. Stationary/singleton tracking defines the spline
  from its environment before evaluating actual encoding samples; a one-center
  environment uses an intercept. No artificial exposure/events are introduced.
- MRF warm starts, exposure safeguards, and η bounds carry the 2 ms unit
  conversion. Its objective removes only the corresponding count-dependent
  unit constant to preserve the relative convergence test. Solver criteria and
  tolerances are unchanged.
- Review reproduced a Unix-timestamp failure in the legacy chunk adapter:
  moving its closing edge changes physical exposure as well as event ownership.
  Unmarked callbacks now support only full-grid requests; chunked callbacks
  require `row_slice_aware` with full edges/global rows. A red→green regression
  verifies rejection rather than silent altered likelihoods; marked paths keep
  exact original durations and boundary ownership.
- Encoding endpoint holding is not automatically decoding interpolation.
  The migration guide gives conservative whole-bin masks on continuous finite
  sample spans and per-segment tracking inputs. It retains the accepted 6a
  removed-helper and Viterbi guidance and documents direct-call units and GLM
  serialization limitations.

### Acceptance evidence

- All **16 public calibration cases** (eight backends × 30/500 Hz tracking)
  recover the independently known 5 Hz rate when decoded in 2 ms and 4 ms bins,
  using the existing `rtol=1e-5`. The same final fixtures fail against accepted
  6a, demonstrating the rate error (and its stationary-GLM limitation).
- All **seven supported model save/load families** preserve exact result
  datasets and their fitted Hz/exposure/clock metadata. GLM is verified in memory;
  its independent Patsy pickle defect remains deferred.
- **26 additional compatibility cases** cover missing/unknown per-entry markers,
  unsupported contract versions, likelihood/Viterbi/predict/cached rejection
  before likelihood work, and legacy recovery through estimation refitting in
  both detector families. **111 chunk/API tests** pass after callback review.
- **101 selected float64 tests** pass. The runnable example yields 400 bins,
  two independent sequences, and 68 missing observations with normalized
  posteriors. Two independent reviewers found no remaining package correctness
  or API/UX defect after the final corrections.
- Controlled 500 Hz before/after data preserve centers exactly. Max posterior
  differences from accepted 6a are `3.58e-7` (sorted decoder) and `2.38e-7`
  (nonlocal); max log-likelihood difference is `1.91e-6`, from float32 unit
  arithmetic. Outputs remain finite/normalized. Four golden files and numerical
  tolerances are unchanged.
- Pre-approval complete suite: **2,198 passed / 6 skipped / 3 failed**, with
  only the subsequently approved snapshot expectations failing. The final focused run passed
  107; lint/format/whitespace pass and targeted mypy introduces no new messages.
  The required four-part numerical analysis is recorded in [time_grid_validation.md](../../../../docs/time_grid_validation.md).
  The three duration-sensitive snapshot corrections were executed separately
  against independent references, then explicitly approved and applied under
  `CLAUDE.md`. All eight repository snapshots now pass. The post-approval
  full run is recorded with the phase 6c completion evidence below.

Release still requires coordinated Spyglass/replay adapter migration, original
tracking support, aligned masks/covariates, dependency pins, and re-population of
saved scientific results. External repositories were not modified or validated
on a real DataJoint/NWB stack. Phases 7/8, the C1 background model, singleton
geometry, general GLM knot defaults, and EmpiricalMovement defects remain separate.


## PR follow-up — 2026-10-06

Explicit interval support and physical-time model-checking acceptance is recorded in the
[follow-up validation](../../../../docs/time_grid_validation.md#pr-review-follow-up--2026-10-06)
and [scope record](phase-6-worktree-scope.md#pr-review-follow-up--2026-10-06).
The accepted checkpoint above remains unchanged; these fixes do not alter
existing reference data, tolerance policies, convergence criteria, or deferred
scientific policies. Final frozen suite: **2,656 / 6 skipped / 0 failed**; all source hashes match.
GitHub CI is tracked in [PR #59 checks](https://github.com/LorenFrankLab/non_local_detector/pull/59/checks); downstream release qualification remains required.
