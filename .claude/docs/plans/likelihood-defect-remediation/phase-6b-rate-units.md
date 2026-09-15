# Phase 6b — Rates in Hz and duration-scaled intensities

> **BLOCKED ON C3b AND APPLICABLE C1 DECISIONS — NEEDS PROTOTYPING.** Depends on
> 6a/6c and ships atomically with 6d's model-unit validation and metadata plumbing.
> The former unexecuted formulas and constant-posterior-rescale criterion are
> replaced by the unit requirements below.

## Recorded problem

`weighted_mean_rate` returns spikes per weighted position sample. Most encoding
fits accept a sampling-frequency argument without using it to establish exposure.
The original audit recorded:

```text
position fs=30 Hz:  mean_rate=0.166667/sample -> 5 Hz
position fs=500 Hz: mean_rate=0.010000/sample -> 5 Hz
encode at 30 Hz, decode at 500 Hz:
  expected count per decode bin: 0.01
  model's summed intensity:     0.1735 (~17.4x)
```

Updating that helper alone is insufficient. GLM coefficients, MRF offsets, and
diffusion exposure fields encode sample-based units through other paths.
No-spike rates are already documented in Hz and must not be converted twice.

## Contracts and inventory

Use [C3](shared-contracts.md#c3--time-vocabulary) for exposure/decoding intervals,
C2 for weighted events, and C1 for the applicable floor policy. Complete the
following inventory against the actual code before choosing implementation:

| Backend | Fit exposure | Stored rate/field units | Event term | Expected-count term | Local/non-local consumers | Metadata dispatch |
|---|---|---|---|---|---|---|
| sorted KDE | | | | | | |
| sorted GLM | | | | | | |
| sorted diffusion | | | | | | |
| sorted MRF | | | | | | |
| clusterless KDE | | | | | | |
| clusterless log-KDE | | | | | | |
| clusterless GMM | | | | | | |
| clusterless diffusion | | | | | | |
| no-spike | | | | | | |

Target exposure is seconds and stored rate fields are Hz. For a Poisson count
likelihood with rate `r` and duration `dt`, the expected count is `r × dt`.
Record the corresponding event `log(dt)` contribution and every retained/omitted
normalization constant for each backend, including marked point-process paths.
Preserve cross-state comparability and evidence conventions explicitly.

## Falsification and prototype requirements

- Use controlled fixtures with known physical exposure and event counts to
  reproduce calibration errors. A backend already passing a reference test is
  not evidence that the test is invalid; no-spike already uses Hz. Classify each
  inventory row by its actual pre-fix behavior.
- Calculate encoding exposure on the original acquisition timeline as weighted
  sample-cell durations according to resolved C3b. Do not infer it from weighted
  sample count times median spacing or bridge dropped-tracking gaps implicitly.
- Prototype GLM duration offsets, MRF exposure offsets, diffusion exposure fields,
  and helper-based mean rates separately. Preserve Phase 5 weighted-event
  statistics and apply event weights once. Store fields and coefficients with
  explicit units rather than multiplying an old per-sample fit by an extra `dt`.
- Apply duration factors consistently in local/non-local event and ground-process
  terms, no-spike, chunked paths, missing rows, and covariate-driven prediction.
- Inventory rate floors, zero-exposure fallbacks, and regularization under the
  new units. C1 may need an explicit unit interpretation; do not silently turn
  a per-sample numerical bound into the same numeric Hz bound.
- Derive metadata from fit, validate its meaning through 6d, and pass only the
  intended predictor arguments. No new key may break strict predictor signatures.
- Audit each encoding-fit `sampling_frequency` argument. Remove ignored arguments
  where exposure now comes from timestamps; retain and document any genuinely
  used argument. Detector sampling frequency remains relevant to grid generation
  and per-step transition interpretation.

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
| Unequal encode/decode intervals | Independent reference reproduces the expected count in the recorded 30 Hz → 500 Hz case; retain the existing 5% end-to-end recovery target on its controlled fixture. |
| GLM/MRF/diffusion | Stored fields/coefficients have Hz meaning; preserve the existing known-rate GLM recovery target while testing offsets independently. |
| Endpoints/gaps | Total exposure and excluded intervals follow resolved C3b, including short inputs and missing tracking. |
| Equal intervals | Expected counts and likelihoods agree under the conditions above; compare stored units separately from HMM results. |
| Direct nonuniform likelihoods | Both registries and no-spike use per-row durations; detector-level HMM calls reject nonuniform grids through 6c. |
| Model compatibility | Every newly stamped fit is usable and old rate units are rejected through the atomic 6d change. |

Run all affected backends and the full suite, including snapshot/golden checks.
Attribute actual changes to unit conversion, exposure, floor interpretation, and
other previously reviewed time migration effects. Not every golden necessarily
changes: well-controlled equal-interval cases can preserve likelihoods. Apply the
existing numerical-change approval process before modifying references or bounds.

The release note must describe Hz storage, duration scaling, any removed fit
arguments, and required refitting of incompatible models. Review the completed
backend matrix and 6d compatibility tests together.
