# Phase 6b — Rates in Hz and duration-scaled intensities

> **BLOCKED ON C3b AND APPLICABLE C1 DECISIONS — NEEDS PROTOTYPING.** Depends on
> 6a/6c and ships atomically with 6d's model-unit validation and metadata plumbing.
> The former unexecuted formulas and constant-posterior-rescale criterion are
> replaced by the unit requirements below. Also depends on Phase 5 (canonical
> event weights, hard-window removal, clusterless full-timeline weights), which
> is not yet implemented.
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
