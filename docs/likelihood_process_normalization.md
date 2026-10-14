# Marked-process normalization

For each electrode, the signal event intensity must integrate over waveform
marks to the ground-process intensity used in the no-spike compensator:

```text
integral lambda(x, m) dm = lambda_ground(x)
log likelihood = sum_events log lambda(x, m) - duration * lambda_ground(x)
```

The normalization changes below apply to the signal density. The existing
low-level likelihood floors and zero-rate contracts are retained; raw GMM event
intensities and unfloored diffusion ground intensities are also available for a
proper background-process mixture. Tracking exposure, bandwidths, GMM fitting,
convergence criteria, and numerical test tolerances are unchanged.

## GMM conditional marks

The GPI and joint GMM can have different component counts, covariance formats,
and fitted position densities. The event intensity now retains the separately
fitted ground-process estimator and normalizes marks with the joint model's own
analytical position marginal:

```text
log lambda_ground = log rate + log p_gpi(x) - log p_occupancy(x)
log lambda(x,m) = log lambda_ground + log p_joint(x,m) - log p_joint_position(x)
```

The position marginal retains the joint mixture's weights and position means.
Its covariances are the position covariance subblocks for full/tied models,
position variances for diagonal models, and unchanged variances for spherical
models. Slicing precision factors would compute a conditional distribution,
so marginal Cholesky factors are computed from the covariance blocks. These
small factors are reused across prediction chunks. Conditional marks are scored
directly with Gaussian conditional means and Schur-complement covariances,
rather than subtracting large joint and marginal log densities. Component
position-log offsets are removed before adding mixture weights, preserving
conditional responsibilities in deep spatial tails. The compiled, spatially
tiled, local, and raw-event paths share this
normalization. Zero-weight electrodes remain zero-rate and bypass density fitting.

The regression fixture deliberately fits different position densities to the
GPI and joint model. At three position bins, its ground intensities are
`[3.205028, 2.202781, 1.179064]`. Before this correction, numerical integration
of the public event intensities returned `[2.337809, 2.852084, 2.595651]`, up to
120.1% relative error. Afterward the mark integrals recover the separately fitted
ground intensities within the existing float32 reference tolerance. This changes
event likelihoods when the GPI differs from the joint position marginal; the
fitted GPI and compensator retain their previous estimator.

The direct conditional calculation also preserves the mark normalizer far from
the encoding data. For an independent standard position/mark Gaussian with mark
zero, the correct event term is `-0.918938533` at every position. Joint-minus-
marginal subtraction in float32 returned `-0.919922` at position 200 and `-1`
at position 2000. Direct conditional scoring retains the Gaussian reference,
including mixture weights and correlated conditional means in these tails.

## Fixed positive diffusion operator

A truncated heat kernel can have negative lobes. Clipping and normalizing a
whole field separately for each decoding mark is nonlinear: smoothing the sum
of mark fields differs from summing their smoothed values. The operator now
corrects each spatial point-source column before any fields are mixed:

```text
h_s = max(Q * exp(-t * eigenvalues) * Q[s,:].T, 0)
h_s[x outside source s's connected component] = 0
h_s = h_s / sum_x h_s[x]
H_positive = matrix of fixed columns h_s
```

Occupancy, GPI, and joint marks all use `H_positive`. Columns conserve mass and
cannot connect disconnected components. Since the operator is fixed and linear,
integrating the Gaussian mark kernels recovers the same GPI field exactly up to
floating-point summation. Position-bin measures remain in the densities.

On a three-node path with spatial bandwidth `0.1`, rank `2`, and two endpoint
spikes with well-separated marks, the old mark-integrated spatial field was
`[0.713674, 0.572653, 0.713674]`, while smoothing the combined ground field gave
`[0.666667, 0.666667, 0.666667]`: a 14.1% discrepancy. The corrected operator
produces the same `[0.713674, 0.572653, 0.713674]` for both calculations. Occupancy
is also recomputed with that operator, so the final intensity ratios are coherent.
Full-rank kernels change only where column correction affects numerical roundoff;
deliberately aggressive truncation can change spatial likelihoods substantially.
Saved diffusion encoding models using the old nonlinear correction require refit.

Prediction compacts each mark field onto the sorted union of encoding spike
bins, `S`, and uses `H_positive[:, S] @ D[S]`. This removes only exactly zero
rows: every retained encoding spike contributes its source bin even when its
current EM weight is zero, so subsequent zero-to-positive weight changes remain
valid. Columns are normalized across all destination bins before local selection.

Source columns are generated in tiles even when a cache fits. Only the occupied
float32 columns and their support key are retained, within the aggregate limit
`min(64 MiB, memory_budget / 4)`; otherwise there is no dense kernel cache and
each field is accumulated from tiles of its occupied source columns.
The environment shares one cached operator across encoding groups and EM refits,
replacing it when the basis, bandwidth, or source support changes. Occupancy may
have support beyond encoding bins; its occupied columns are reconstructed without
retaining an additional operator cache. Grid refits invalidate this
cache, and serialization excludes the transient environment copy. A prediction
with a reduced budget can bypass a larger fitted cache. As with existing decode
blocking, one source column is the minimum workspace for tiny budgets.
The encoding model also records the environment's grid revision, so a refit
changing physical coordinates or bin widths is rejected even when interior-bin
indices and counts remain identical. That check also applies to raw operator
reuse by the background likelihood.

The retained operator costs `4 * n_bins * len(S)` bytes plus its integer support
key. For 4,608 bins and 1,000 occupied source bins, its values need 17.6 MiB,
compared with 81 MiB for all columns. Uncached reconstruction likewise scales
with `len(S)` rather than every spatial bin. Connected-component isolation,
column normalization, bin measures, and `Precision.HIGHEST` are preserved.

The unfloored GMM path also exposes a pure array-parameter scoring function,
allowing the background likelihood to compile conditional marks, spatial density
ratios, mixture evaluation, and deterministic row reduction together. Host checks
and cached Schur-complement preparation remain outside that executable.

Encoding models retain compact source indices and an empty operator placeholder;
the environment owns the persistent matrix. Different encoding groups reuse a
union of occupied columns when it fits the same aggregate cache budget. If the
union exceeds that budget, lookup replaces the cached support. A model kept alive
after replacement therefore cannot retain an evicted operator. Prediction selects
its needed columns into a temporary buffer, which is never memoized across
observation states or chunks. Existing columns are copied exactly when the union
grows, preserving their original float32 rounding.

## Numerical and performance checks

Tests integrate public local and non-local event intensities over Gaussian marks,
exercise all four GMM covariance formats, compare compiled and tiled GMM paths,
and check cached/tiled diffusion, ranks, component mass, positivity, and linearity.
Existing zero-weight, chunk, padding, geometry, save/load, and compiled-executable
reuse tests remain active. References asserting the old GMM formula were updated
to analytical marginal expectations, including an independent SciPy mixture
reference; no tolerances or stored golden/snapshot data were changed.

A CPU microbenchmark used Python 3.13, JAX 0.9 float32, 512 interior bins, two
electrodes, 500 encoding and 100 decoding spikes per electrode, four mark
dimensions, 64-spike blocks, and the median of seven warm predictions. Both runs
used automatic heat-rank selection and default memory budgets; the new cache
was 1 MiB. These measurements describe this workload, not a speed guarantee.

| Spatial bandwidth | Resolved rank | Warm prediction before | Warm prediction after | Fit before | Fit after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 512 | 2.622 ms | 1.490 ms | 145.7 ms | 138.6 ms |
| 12 | 72 | 1.996 ms | 1.620 ms | 113.6 ms | 155.9 ms |

Before changes, all four stored golden tests and 57 property tests passed.
After changes, targeted GMM/weights/optimization/golden checks passed (78 tests,
one existing skip), and targeted diffusion/normalization/integration checks passed
(57 tests). The final validation record is reported with the PR. Stored golden
references exercise KDE likelihoods and remain unchanged.

The first complete normalization suite passed 3697 tests with seven skips; its
only failure was the missing generated `_version.py` in the isolated worktree.
Restoring that ignored build artifact made the version check pass. After the
direct-conditional precision and grid-revision follow-ups, the GMM, diffusion,
normalization, optimization, padding, and save/load suite passed 127 tests with
one existing skip. The follow-ups added 23 analytical regression cases; no
golden files or numerical tolerances were changed.

The occupied-source performance follow-up passed 199 diffusion, normalization,
integration, cache, and property checks, plus 72 GMM/padding/optimization/golden
checks with one existing skip. Four stored golden comparisons remain unchanged.
New compact/full comparisons cover cached and budget-forced tiled operators,
disconnected components, zero-to-positive EM weights, source-support replacement,
bandwidth/rank replacement, and the aggregate cache bound. A seeded two-electrode
before/after float32 capture differed by at most `4.3e-7` relative in occupancy
and GPI and `4.2e-7` relative in local/non-local likelihoods (`7.63e-6` maximum
absolute log difference). These changes reflect matrix shape and accumulation
rounding after zero rows are removed; normalization and component isolation
remain within the existing tests. No stored references or tolerances were edited.
