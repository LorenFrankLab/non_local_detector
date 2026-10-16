# KDE density ratios in finite tails

Gaussian KDEs have positive density at every finite evaluation point. A density
that underflows in float32 therefore does not indicate an unsupported spatial
bin. The intensity `rate * spike_density / occupancy_density` now uses true log
densities for sorted KDE, clusterless KDE, and clusterless log KDE.

The fitted model retains its existing probability-space occupancy array and
adds `log_occupancy`. Zero fitted weight is represented as `-inf` in the true
log-density helpers, rather than the legacy KDE log-density floor. Unsupported
numerator or occupancy mass produces zero signal; NaN density computations remain
visible. Tiny positive mean rates enter through their true logarithms. Existing
final rate and marked-intensity floors remain in the low-level likelihood contract.

The ground-process ratio is evaluated as:

```text
log intensity = log rate + (log spike density - log occupancy density)
intensity = exp(log intensity)
```

Marked kernels divide the position kernel by occupancy before adding the mark
log kernel. This also avoids losing a small mark normalizer when joint and
occupancy log densities are very large negative values. Local kernels keep the
position and mark contributions separate. The linear clusterless implementation
retains its ordinary probability-space event arithmetic and uses a log-space
fallback for small or invalid occupancy densities. Its non-local kernel compiles
that fallback only when some bin's log occupancy is below `LOG_EPS` or NaN, and
its local kernel, which receives the occupancy at each spike evaluated
beforehand, only when some spike's log occupancy is below `LOG_EPS`. Tiled
prediction keeps the existing encoding and evaluation workspace limits and row
accumulation.

Tiled log-density evaluation carries a kernel offset and stabilized weighted
sum across encoding tiles. It normalizes that sum before adding its logarithm
to the offset, rather than subtracting two large log sums afterward. Identical
kernels therefore retain their exact log kernel even in distant tails. Empty
encoding support and zero-weight prefixes bypass the relevant reductions; the
tests also check finite, analytically correct tail gradients.

True log densities take one pass per block of evaluation points, as the
probability-space KDE does. Each point's log kernels are shifted by an upper
bound on its largest log kernel before they are exponentiated, and weights enter
relative to the largest weight, so neither terms nor sums overflow. For 1D
positions the bound is exact: `kde_log_density` sorts the positive-weight sample
positions on the host and finds each point's nearest one by binary search, so
distant tails stay in the single pass. Otherwise the bound is the kernel's peak,
and the pass computes the probability-space KDE. A term is lost only if it
underflows below float `tiny`; a point whose scaled sum is below `n_samples *
tiny / (4 * eps)` could have lost more than `4 * eps` of it that way and is
evaluated in log space instead: `kde_log_density` re-evaluates those points
after its compiled pass. `traced_kde_log_density`, used inside compiled kernels,
does so for each block under `lax.cond`; callers that map it in batches, where
that branch would become a select, evaluate in log space throughout. Log space
offsets each point by its largest weighted term, keeping each kernel relative to
that offset before its log weight is added, so it recovers what either scale
loses, including mass carried by weights whose ratio to the largest is below
float `tiny`. Weights enter all three forms relative to the largest, so their
sum cannot overflow.

The exact 1D bound matters in practice. In a simulated 350,000-bin recording
with five electrodes (position bandwidth 6 cm), scaling by the peak would have
sent 25-47% of decode positions to log space for the ground-process densities of
four electrodes, whose tails there reached 131-315 nats below the peak. The sort
runs on the host because XLA's GPU sort took 0.7-1.6 s to compile for each array
size between 4k and 20k elements (A100, JAX 0.9.0), and fitting compiles one
density kernel per electrode. Occupancy at spike positions is scaled by the
peak: in that recording the occupancy at all 350,000 decode positions stayed in
the single pass.

For five identical position and spike samples at `x=0`, with position bandwidth
1 and mean rate 1 Hz, the true ground intensity is 1 Hz across a `0..200` track.
Previously a one-second no-spike bin at `x=20` scored `-5e-13` instead of `-1`.
Both clusterless KDE variants and sorted KDE now score `-1`, locally and across
the spatial grid. A zero mark with unit mark bandwidth contributes
`-0.918938533` at every position. Previously the log KDE score at `x=12.5` was
`-28.32417`; the corrected score matches the Gaussian reference throughout the
grid, including `x=200` local queries.

The tests cover untiled, sample/spatial tiled, encoding chunked, and streaming
paths, with both one and nine mark dimensions. They preserve existing numerical
tolerances and stored golden data. A reference that treated a zero-mass KDE as
a positive `EPS` density was corrected to apply `LOG_EPS` only to the final event
intensity. Numerical effects in finite supported tails are intentional changes
to scientific likelihoods; ordinary kernel, padding, and gradient checks remain
active.

Validation at implementation: 91 focused tests passed, all four stored golden
regressions passed unchanged, and ten likelihood property tests passed. The full
property and test-suite results are recorded with the PR.

The compiled sorted local helper retains its existing probability-occupancy
argument and ordinary arithmetic. Public prediction additionally supplies true
log occupancy for NaN diagnostics and, from the host, the rows whose log
occupancy is below `log(EPS)`. Only those rows are evaluated in log space, so a
request without them compiles and runs no log-space KDE; inside the helper's
batched map over neurons an in-graph branch would have evaluated both forms for
every row. This preserves legacy low-level vector-position and
unsupported-position calls while keeping the public finite-tail correction. The
local accumulation and tail suites passed 113 tests after that compatibility
check; all 57 property tests also passed.

Fitting evaluates every neuron's (and, for clusterless KDE, every electrode's)
log density in batched passes over groups of similar sample counts, instead of
compiling one kernel per sample count; each batch's kernels share one model's
workspace budget. Fits and sorted local prediction take the probability
occupancy as the exponential of the true log density rather than evaluating the
KDE a second time.

Independent weighted-tail review found that the common log-KDE helper also
rounded log weights before their normalization could cancel. It now offsets each
point by its largest weighted term and takes each kernel relative to that offset
before adding its log weight, then normalizes the stabilized mass before adding
the offset. Weight rescaling keeps its sum representable, and empty support
bypasses undefined reductions. Fifteen additional regressions verify
identical-source densities at positions 200 through 1,457, global weight-scale
invariance, analytical Gaussian gradients and zero gradients for
empty/zero-weight support. These and the existing common, tail and four golden
cases pass at their unchanged tolerances (101 cases).

Custom `KDEModel` subclasses retain their own `predict_log` behavior. Local
prediction shares row interpolation, occupancy, and ground-process log densities
through the prediction's `_SpikeTimeOrder`. Non-local scoring computes each
electrode's spatial log kernel inside its compiled scorer rather than caching
it. Linear KDE derives probability and log position kernels from one
calculation.

Preparation stays within one immutable prediction, with a 32 MiB and 256-entry
LRU limit. Its byte accounting conservatively counts strong source-array
references as well as results, including old fitted leaves after replacement.
Keys include model samples, weights, bandwidth and block policy, original
tracking inputs, time grid, and row range. A fresh prediction gets a fresh
cache. Kernels can accept a prepared GPI density while retaining their original
uncached contract for direct callers. The ordinary probability arithmetic,
intensity floors, log-tail fallback, and numerical tolerances are unchanged.

On the local CPU, the existing seeded `linear_kde` workload with reusable
prediction preparation, three warmups and seven synchronized calls measured
local prediction at 2.154 ms before and 1.069 ms after, nonlocal prediction at
2.018 and 1.648 ms, and public likelihood construction at 5.038 and 3.787 ms.
Five fresh warm fits measured 13.92 and 14.02 ms; no fit speedup is claimed.
Nonlocal outputs were bitwise equal to the captured baseline, and local/public
outputs differed by at most 2.98e-8 from reduction placement. All outputs were
finite and repeatable, with zero new warm compilations. Hardware-specific
CPU/GPU confirmation remains part of the full performance review.

The seven-fix likelihood/model-checking run passed 1,754 cases and skipped
three; its sole failure was an old test explicitly requiring tiny positive
mean rates to be replaced by `EPS`. That expectation conflicts with this
repair's declared true-positive-rate formula. The test now checks the analytical
`log(rate) + log(density) - log(occupancy)` value and the exact relative-rate
shift, with its existing comparison tolerance. No implementation changed.
