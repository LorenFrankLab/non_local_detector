# Phase 4 — Numerical hardening

> **IMPLEMENTED AND VALIDATED** on
> `fix/likelihood-numerical-hardening`, based on
> `ffc85a140a3dc91a7562bb452bb51ce8e15dceda` (merged Phase 3); merged to `main`
> at `ee2cc21`.
> Existing likelihood floors, regularization, and convergence tolerances are
> preserved. This phase does not select C1's deferred background model.

Phase 0 owns core HMM conditioning. Phase 1 already implements the sorted GLM
zero-exposure guard. The GLM objective already normalizes its weighted data term
by total weight before adding regularization; the former index entry claiming
that implementation was missing is withdrawn. Preserve those behaviors rather
than treating them as new Phase 4 deliverables.

## Implementation record

The defect evidence and acceptance requirements below are retained as the
original specification. The following records the executed implementation.

- **Empty KDE tiles:** both compensated reducers substitute a safe operand for
  negative-infinite maxima before subtraction, covering kernel normalization,
  running rescaling, and tile scales. Empty and padded contributions have zero
  mass and reach the existing `LOG_EPS` combiner policy. Raw NaN remains visible,
  including on zero-weight observations; the former all-zero final override
  no longer masks it. Tests compare values and derivatives with respect to
  decoding marks against direct Gaussian products for precomputed and streaming
  position kernels. This does not promise derivatives at the boundary of an
  exact-zero encoding weight; that pre-existing `log(weight)` boundary remains.
- **Gaussian scoring:** diagonal and spherical paths subtract means before
  squaring, then accumulate features into the required samples × components
  output. A fixed unroll of eight permits fusion without constructing a
  components × samples × features workspace or an unbounded compiled body.
  Static feature and precision shape checks prevent indexed accumulation from
  silently accepting incompatible models. Full/tied scoring is unchanged.
- **Weighted GMM:** validate weights on the host, remove zero-weight rows, divide
  positive weights by their maximum, and then rescale to mean one in host
  float64 before conversion to the data dtype. This avoids overflowing a raw
  sum, flushing float32 subnormals during device arithmetic, or narrowing large
  finite float64 weights prematurely. The input weights are copied, not mutated.
  The installed scikit-learn 1.8.0 `KMeans.fit(X, y=None, sample_weight=None)`
  interface was verified; initialization now consumes the weights. Zero-weight
  rows cannot affect either KMeans or random initialization. EM, stopping,
  restart selection, and `lower_bound_` all use the same weighted objective.
  The objective still describes the E-step parameters before the final M-step.
  Mixture weights divide by the sum of guarded counts, so the existing small
  count guard cannot inflate their total above one.
- **Clusterless exclusion order preserved** (superseded by the review rounds
  below: exclusion now lives in `GaussianMixtureModel.fit` via
  `_effective_weight_mask`, which also drops rows below `10·eps` after
  rescaling to mean one, and direct `fit` checks finiteness on kept rows only):
  `_fit_gmm_density` validates
  weights and removes zero-weight rows before the GMM checks data finiteness.
  This retains the baseline's physical-subset behavior when an excluded spike
  has a NaN or infinite waveform feature. Invalid positive-weight observations
  still raise, and direct `GaussianMixtureModel.fit` retains its existing
  full-input finiteness check. Host weights remain unnarrowed until normalization;
  an all-positive mask does not gather the training array.
- **Covariance correction required by the scaling tests:** normalization alone
  left diagonal, spherical, and tied fits sensitive to rounding because their
  second-moment formulas also suffered cancellation. Independent centered
  covariance tests reproduced this defect. These estimators now center before
  accumulating, visiting one component at a time to bound workspace. The
  full-covariance estimator was already centered. Count guards and `reg_covar`
  are unchanged. This additional arithmetic has a measured fitting cost below.

The host normalization belongs to the existing Python fit/validation boundary;
there is no host conversion inside the JIT-compiled EM loop. Shapes and
covariance type control compilation; data values, empty-tile locations, and
weight values do not introduce new static arguments.

### Numerical evidence

Against the frozen pre-fix checkout, the shipped test modules
(`test_kde_empty_tiles.py`, `test_gmm_numerical_hardening.py`) fail **49 of 80**
collected cases (31 pass); an earlier scratch reference set (48 cases, 32
failures) is not in the tree. Failures isolated empty-first-tile values, all-empty
derivatives, concealed invalid input, large-coordinate Gaussian cancellation,
weight scaling, zero-weight initialization, and weighted objective selection.
The separate centered-covariance reference set failed **3 of 6** baseline cases
(the large-offset cases). Preservation cases were expected to pass already.

Review added regressions for feature/precision shape mismatch, feature counts
on either side of the unroll boundary, float32 weights down to `1e-40`, and
native float64 weights at `1e-200` and `1e200`. A shape/subnormal probe (scratch, not in the tree) failed
**5 of 6** cases before its corrections; two x64-gated narrowing cases (since removed as redundant with the native-float64 weight test) likewise failed
before normalization was moved ahead of dtype conversion.

A subsequent public-API exclusion regression covered NaN, positive infinity,
and negative infinity at zero and positive spike weights. Before restoring
the wrapper's exclusion order, the three zero-weight cases failed and the three
positive-weight validation cases passed. After the fix, all six pass; excluded
cases match an explicit spike subset in fitted parameters, occupancy, rates,
and summed ground-process intensity.

Whole-estimator scale comparisons cover all four covariance forms at scales
`1e-9`, `1e-6`, `1e35`, and `1e37`, including heterogeneous and zero weights.
Means, covariances, mixture weights, scores, and the reported objective meet the
existing `rtol=1e-4` target; mixture sums meet `atol=1e-6`. A controlled
two-component/two-restart test computes responsibilities and objectives
independently and checks both the winning objective and fitted means. Large
Gaussian-coordinate cases match centered float64 references and equivalent
fixed full-covariance parameters; recovered squared distances are nonnegative.

For well-conditioned fixed Gaussian parameters, the first 256 saved output rows
from the 100,000-sample, 32-component baseline/current benchmarks meet the
existing `rtol=1e-5` target:

| Features | Covariance | Maximum absolute change | Maximum relative change |
|---|---|---|---|
| 8 | Diagonal | 7.63e-6 | 2.99e-7 |
| 8 | Spherical | 7.63e-6 | 3.97e-7 |
| 32 | Diagonal | 1.53e-5 | 2.44e-7 |
| 32 | Spherical | 3.05e-5 | 3.70e-7 |

These are float32 evaluation-order differences, distinct from the intentional
corrections in large-coordinate and weighted-fit cases. No tolerance, fixture,
golden, or snapshot has been changed. No new covariance-collapse warning was
observed in either full-suite run.

### Performance and memory

`scripts/benchmark_gaussian_numerics.py` records synchronized execution after
three warmups, 15 repeat timings, compiled buffer sizes, and fixed-parameter
score samples. Comparison uses the pinned baseline above via `PYTHONPATH`.
Measurements: CPU, JAX 0.9.0, float32, 100,000 samples, 32 components. Compilation
and host initialization are excluded. Memory is XLA's reported temporary buffer
allocation, not process RSS or measured accelerator peak memory. Required score
output is another 12.8 MB in every score case below; MB means 1,000,000 bytes.

| Features | Operation | Baseline ms | Current ms | Baseline temporary MB | Current temporary MB |
|---|---|---|---|---|---|
| 8 | Diagonal score | 0.533 | 0.468 | 16.001 | 0.000128 |
| 8 | Spherical score | 0.319 | 0.369 | 3.601 | 0.000128 |
| 32 | Diagonal score | 1.050 | 2.842 | 25.608 | 0.000424 |
| 32 | Spherical score | 0.633 | 2.426 | 13.204 | 0.004432 |
| 8 | Full EM, 5 iterations | 59.109 | 61.154 | 26.022 | 26.022 |
| 8 | Tied EM, 5 iterations | 37.640 | 67.183 | 26.013 | 38.402 |
| 8 | Diagonal EM, 5 iterations | 8.014 | 25.462 | 32.800 | 28.814 |
| 8 | Spherical EM, 5 iterations | 6.504 | 22.936 | 20.400 | 28.814 |
| 32 | Full EM, 5 iterations | 234.872 | 215.648 | 38.672 | 38.672 |
| 32 | Tied EM, 5 iterations | 122.955 | 234.440 | 38.414 | 38.418 |
| 32 | Diagonal EM, 5 iterations | 12.767 | 63.673 | 52.000 | 26.005 |
| 32 | Spherical EM, 5 iterations | 9.863 | 65.106 | 39.600 | 26.005 |

The default full-covariance arithmetic is unchanged; its timing spread is not
evidence of a speedup. Centered tied covariance changes work from approximately
`O(N D² + K D²)` to `O(K N D²)`. Centered diagonal/spherical updates retain
`O(N K D)` work but lose the old matrix-product throughput. Their fitting cost
is material: roughly 3.2–3.5× at eight features and 5.0–6.6× at 32 in this
workload. Diagonal/spherical scoring at 32 features also costs more despite the
memory reduction. Bounded batching prototypes did not remove these trade-offs;
this is a correctness change, with further throughput work left to Phase 7c.
These measurements do not establish full-session feasibility or GPU performance.

### Validation results

(Recorded before review round 2; after it the full suite is **1762 passed /
6 skipped**, see PLAN.md.)

- Final full suite: **1751 passed / 6 skipped**, 249 warnings, in 717.93 seconds.
  This includes likelihood, integration, probability-property, GLM-preservation,
  golden, and snapshot tests. Skips cover three float64-only cases, two manual
  visualizations, and an optional `sortingview` dependency. Golden and snapshot
  files are unchanged.
- Targeted numerical references, existing GMM tests, clusterless weight tests,
  and GMM log-intensity regressions: **129 passed** in default float32 and
  **129 passed** with `JAX_ENABLE_X64=1`.
- Ruff check/format and whitespace validation pass.
- Mypy retains the same **52 pre-existing errors** in the two touched production
  modules; comparison of diagnostics found no additions.
- Independent final source review found no remaining correctness blocker.

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

## Review round (2026-09-19)

A branch review fixed the following before commit:

- `GaussianMixtureModel.fit` now validates weights first, keeps only rows whose
  weight survives narrowing to the feature dtype (`_effective_weight_mask`:
  ratio to the largest weight at or above the dtype's smallest normal), and
  checks `X` finiteness on those rows only. The clusterless wrapper no longer
  duplicates the exclusion, and its component reservations use the same mask.
- The unweighted `n_samples < n_components` rejection has its own message; the
  weighted convergence objective is recorded as a criterion change in the
  CHANGELOG.
- Tied and diagonal covariance M-steps use a feature-axis / component loop
  with `unroll=2` (not 8): the XLA scratch buffer scales with the unroll factor
  and with `n_samples`, so 2 keeps whole-fit temporaries within ~1.5x of the
  previous release. The scoring loop keeps `unroll=8`, which allocates no
  measured temporary.
- The empty-component normalization test now creates empty components and
  uses an explicit float32-ulp tolerance; the six `-inf` guards in the KDE-log
  reducers share one helper; the benchmark script checks the EM iteration
  count on an untimed run before timing.

## Review round (2026-09-22)

A second branch review changed the following before commit:

- `_effective_weight_mask` now drops rows whose weight, rescaled to mean one
  over the positive rows, is below the M-step's `10 * eps` count guard (was:
  ratio to the largest weight below the dtype's smallest normal). Weights of
  1e-10 to 1e-30 were still reserving guard-level phantom components at the
  origin. The dropped rows together carry under `10 * eps` of the total weight.
- The clusterless wrapper counts effective rows with `_effective_sample_count`
  on the exact array each fit uses (`pos_for_occ`, `enc_pos`, joint samples),
  not with `position.dtype`; under x64, float32 positions still produce float64
  GPI/joint arrays.
- A full warm start (all three init arrays) is exempt from the
  fewer-rows-than-components rejection, since KMeans is skipped.
- `_validate_fit_inputs` checks finiteness with a per-row flag instead of
  gathering `X[keep]`, which `fit` gathers again.
- Diag/spherical scoring replaced the `unroll=8` feature loop with a fused
  broadcast-reduce centered on each component's own mean: exact, 0 bytes of
  scratch, and 1.4x faster than the loop at 32 features (still ~2.2x slower
  than the uncentered release form there). The reviewer's alternative, centering
  once on a shared point and keeping the matmul expansion, was measured and
  rejected: for components at +-1e3 with sd 0.1 it gives a 48x variance error
  and 43-nat score error in float32. The M-step loops stay for the same reason;
  every exact alternative measured (component `lax.map`, batched map, fused
  broadcast) was 1.3-9x slower than the current loops.
