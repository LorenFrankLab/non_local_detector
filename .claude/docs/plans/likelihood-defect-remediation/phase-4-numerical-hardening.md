# Phase 4 — Numerical hardening

> **NEEDS PROTOTYPING.** Retain the recorded defect evidence; the earlier
> unexecuted implementation snippets are removed. Fix numerical operations under
> an explicitly recorded existing/selected floor policy without selecting C1.
> Coordinate overlapping likelihood sites with Phase 2.

Phase 0 owns core HMM conditioning. Phase 1 already implements the sorted GLM
zero-exposure guard. The GLM objective already normalizes its weighted data term
by total weight before adding regularization; the former index entry claiming
that implementation was missing is withdrawn. Preserve those behaviors rather
than treating them as new Phase 4 deliverables.

## Defect 1 — Empty first tile produces NaN in compensated log-KDE

`_compensated_linear_marginal_chunked` seeds its running maximum with `-inf`.
A first encoding tile containing only zero weights also has maximum `-inf`.
The original audit recorded:

```text
tile size 4; first 4 of 8 weights zero:
  tiled has NaN: True; untiled has NaN: False
  zeros only in the second tile: no NaN
```

The all-zero-electrode guard does not cover a partially weighted electrode.
Guarding only the running-sum rescale also fails: per-chunk subtraction still
forms `-inf - (-inf)` before the square-root scale, contaminating later work.

Prototype safe operands for both running rescaling and per-chunk scale
construction. Empty/padded tiles contribute no mass and must not introduce NaN
into later valid tiles. Keep the distinction between an empty contribution and
a raw invalid input. Verify forward and supported autodiff paths; selecting a
finite result after evaluating an invalid unused branch can still corrupt a
derivative. The all-zero final output follows the baseline/selected C1 policy;
this phase does not mandate `LOG_EPS` independently.

## Defect 2 — Expanded Gaussian distances cancel in float32

The diagonal/spherical Gaussian paths expand squared differences into quadratic
and cross terms. Recorded unit-variance examples with true squared distance 1:

```text
|mu|=1e2: 1.0; |mu|=1e3: 1.0; |mu|=1e4: 0.0; |mu|=1e5: -2048.0
```

A negative squared distance is invalid. This is a float32 cancellation example;
no claim about typical waveform amplitudes is needed to establish the defect.

Prototype centered differences and component-wise evaluation that avoids a
large components × samples × dimensions workspace. The full-covariance branch's
memory constraints are a reference for evaluation strategy. Compare Gaussian
probability evaluation at the same fixed parameters; separately fitting diagonal
and full-covariance models does not guarantee identical fitted parameters.

## Defect 3 — Weighted GMM fitting changes under global weight rescaling

A fixed absolute epsilon added to effective component counts can dominate small
weights and distort means, covariance estimates, and mixture normalization.
Recorded two-cluster example:

```text
weight scale 1:    means [5.008, 24.956], mixture sum 1.000
weight scale 1e-6: means [4.978, 24.808], mixture sum 1.006
weight scale 1e-9: means [0.000, 3.764], mixture sum 6.960
```

Normalizing only the final mixture weights cannot restore damaged means.
Inventory initialization, KMeans, EM updates, convergence, restart selection,
and reported objective for every consumer of `sample_weight`.

Prototype a stable rescaling convention before those consumers. Directly dividing
by the mean is not an accepted implementation: the sum used to calculate that
mean can overflow for large finite float32 weights. Include small and large
representable rescalings, heterogeneous weights, zero-weight observations, and
all-zero weights. Any max-first or other normalization candidate must handle its
own zero/invalid operands and be tested, rather than prescribed here.

Use a weighted objective consistently for convergence, restart selection, and
reporting when the fit is weighted. Preserve the estimator's documented
pre-/post-M-step evaluation timing; compare the objective at the parameter state
it actually describes. Verify the installed KMeans weighted-fit interface before
using it, and ensure zero-weight data cannot influence initialization.

## Falsification and acceptance

Freeze pre-fix examples before editing. Existing preservation tests may already
pass on the baseline; new defect tests must isolate the failure being corrected.

| Test family | Required evidence |
|---|---|
| Empty-tile reducer | Tiled/untiled agreement with the first, middle, and last tile empty; all-zero and padded inputs follow the selected contract. Valid cases are finite and invalid inputs stay visible. |
| Gaussian distances | Nonnegative squared distances matching the independent reference at large coordinates; retain the existing `rtol=1e-4` reference target for the recorded cases. |
| Covariance forms | Diagonal/spherical evaluation agrees with equivalent fixed full-covariance parameters; do not compare independently fitted models as if they were identical. |
| Weighted estimator | Means, covariances, mixture weights, and scores agree across the recorded rescalings at `rtol=1e-4`; also test large finite weights without overflow. |
| Mixture normalization | Unit sum within the existing `1e-6` target, alongside whole-estimator checks. |
| Initialization/objective | Zero-weight data have no influence; weighted objectives/restart selection match an independent calculation at the same parameter state. |
| GLM preservation | Retain Phase 1 zero-exposure/small-positive-exposure regressions and existing weighted-objective behavior. |

Record well-conditioned Gaussian scores before editing and retain `rtol=1e-5`
parity there. These existing plan targets are not authority to relax test
thresholds, alter regularization, or change convergence criteria. Diagnose any
new collapse warning; changing a fixture's `reg_covar` is not an automatic way
to make the regression disappear.

Run affected likelihood, GMM optimization, integration, and golden tests, plus
the full suite for implementation changes. Measure memory for component-wise
Gaussian evaluation. Golden differences may arise in affected numerical cases;
analyze primitive changes and downstream posterior effects before requesting an
actual reference update. Review operand safety and consistent weight/objective
handling across all consumers.
