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
fallback for small or invalid occupancy densities. Tiled prediction keeps the
existing encoding and evaluation workspace limits and row accumulation.

Tiled log-density evaluation carries a kernel offset and stabilized weighted
sum across encoding tiles. It normalizes that sum before adding its logarithm
to the offset, rather than subtracting two large log sums afterward. Identical
kernels therefore retain their exact log kernel even in distant tails. Empty
encoding support and zero-weight prefixes bypass the relevant reductions; the
tests also check finite, analytically correct tail gradients.

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
log occupancy for the tail fallback and NaN diagnostics. This preserves legacy
low-level vector-position and unsupported-position calls while keeping the
public finite-tail correction. The local accumulation and tail suites passed
113 tests after that compatibility check; all 57 property tests also passed.
