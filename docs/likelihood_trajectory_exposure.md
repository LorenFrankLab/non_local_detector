# Continuous trajectory exposure in encoding likelihoods

This change corrects a spatial integration error. Previously, exposure was
integrated in time and then assigned to the measured tracking positions, while
spikes were placed along the continuous interpolated trajectory. Coarse tracking
could therefore put spikes in locations that received almost no estimated
exposure and create extreme false place fields.

## Change

`EncodingSupport.quadrature` integrates each declared continuous tracking segment
on its original interpolation basis. It preserves acquisition clipping, missing
position splits, held segment endpoints, and existing event ownership. Its
positive four-point Gauss–Legendre rules subdivide movement to at most half the
model's spatial scale. Rectangular histogram edge crossings are additional
subinterval boundaries. Positions and weights remain piecewise linear.

Weighted exposure is `duration * interpolated_weight` at each quadrature node.
Linear weights integrate exactly, including fractional EM responsibilities.
Complementary groups reuse the same geometry and partition unweighted exposure.
Quadrature fractions are computed before converting nodes to absolute timestamps;
rounding near Unix epochs therefore cannot corrupt path positions or integrated
weights. Existing timestamp precision validation is unchanged.

Sorted and clusterless KDE occupancy models use quadrature positions and exposure.
Diffusion and MRF occupancy histograms integrate residence times along the same
path used for spike positioning. GMM occupancy fitting uses weighted quadrature
samples. Mean firing rates and `encoding_exposure_seconds` retain their existing
physical-time definition.

The GLM now evaluates its basis at actual spike positions and accumulates
`S_n = sum_events weight * basis`. Its data likelihood is

```text
L_n(beta) = S_n @ beta - sum_q exposure_q * exp(basis_q @ beta)
```

The spline definition is anchored to the environment rather than the tracking
sample distribution. This avoids changing its bounds and centering merely by
resampling a trajectory or updating EM weights. The documented penalty remains
unchanged. Newton line searches evaluate objective differences with `expm1`,
avoiding cancellation between two large float32 objectives. Existing tolerances,
iteration limits, and convergence criteria are unchanged. Prediction still uses
the established positive rate floor. Already-linearized one-dimensional track
positions are preserved instead of projected onto the graph a second time.

Geometry is immutable and weight-independent. The latest geometry is cached only
when it occupies at most 32 MiB; larger geometries are returned without being
retained on the support. Integration storage scales with quadrature nodes and
position dimensions, not neuron count. Construction first counts nodes in
2,048-row source blocks, then fills preallocated arrays in blocks of at most
8,192 subintervals. It retains no small arrays per tracking interval. Subdivision
counts are checked before integer conversion/allocation; requests exceeding the
512 MiB geometry budget raise a configuration error with guidance to coarsen the
spatial scale or shorten the encoding period. Ordinary continuous tracking uses
four nodes per measured interval, plus endpoint holds and any spatial/grid
subdivisions. A one-hour 500 Hz recording therefore retains approximately
7.2 million nodes: about 330 MiB in 1D or 385 MiB in 2D before subdivisions.

Weighted exposure and diffusion histogram accumulation use 32,768-node blocks,
avoiding full-recording gather temporaries. KDE evaluation and GLM exposure sums use
the existing bounded evaluation blocks. GLM event accumulation uses blocks of
65,536 spikes and stores only neuron-by-basis sufficient statistics, avoiding a
quadrature-node-by-neuron count matrix. Fine spatial scales or large movements
between tracking samples can increase geometry size and fitting cost; histogram
crossings are exact for a straight path in a rectangular grid, while graph
projection and KDE/GLM integrals use the stated spatial quadrature approximation.

## Numerical differences and affected metrics

The baseline golden regression suite passed all four cases. With the corrected
exposure, the sorted-spikes golden case passes its unchanged tolerance; the three
clusterless golden cases fail their unchanged tolerance. The random-walk case is
also a clusterless decoder, so its posterior changes through occupancy, not a
change to the transition operator. No golden files, snapshot references, or
comparison tolerances were updated.

The following measurements compare the pinned original arrays with newly computed
arrays. Relative differences are measured at the largest absolute difference.
Indices use zero-based array indexing.

| Case / array | Maximum absolute change | Relative change at that entry | Index | Original | Corrected |
| --- | ---: | ---: | --- | ---: | ---: |
| Clusterless decoder posterior | 2.458691597e-6 | 3.512123478e-5 | (38, 13) | 0.07000584155321121 | 0.07000830024480820 |
| Random-walk clusterless posterior | 2.123415470e-6 | 3.655566613e-5 | (39, 13) | 0.05808717757463455 | 0.05808930099010468 |
| Sorted decoder posterior | 1.117587090e-6 | 5.454437953e-6 | (7, 2) | 0.2048950046300888 | 0.20489612221717834 |
| Nonlocal detector posterior | 9.417533875e-6 | 1.458470065e-5 | (33, 0) | 0.6457132101058960 | 0.6457226276397705 |
| Nonlocal detector state probabilities | 9.417533875e-6 | 1.458470065e-5 | (33, 0) | 0.6457132101058960 | 0.6457226276397705 |

Every compared array is finite. Maximum row-normalization errors, computed after
converting the float32 outputs to float64, are 2.32e-7, 2.40e-7, 1.90e-7, and
3.23e-7 for these four cases. No posterior or state-probability row changes its
maximum-probability category. These high-frequency tracking examples have small
posterior differences; the coarse-tracking counterexample below has a large,
scientifically material correction.

## Independent validation

New regressions verify:

- An analytic clipped triangular weight integrates to 0.9375 seconds, and
  complementary masks partition exposure at every node.
- Missing tracking is excluded, endpoint values are held, and histogram residence
  times agree with an independent analytic two-dimensional bin-crossing example.
- Weighted Gaussian occupancy integrals agree with independent SciPy adaptive
  integration, including irregular timestamps and acquisition clipping.
- Node timestamp rounding at large epochs preserves exact linear-weight exposure.
- A log-linear point-process GLM agrees with its independently solved analytic MLE.
- KDE, diffusion, MRF, clusterless ground rates, and GMM occupancy moments behave
  consistently with their declared continuous trajectories.
- Cache reuse is weight-independent and respects its memory cap.

Existing synthetic tests that intended disconnected occupancy snapshots or
teleports between rooms/corner edges now declare missing tracking or explicit
tracking intervals. Hard-subset comparisons now declare the same support on both
sides: a binary posterior weight alone creates linear boundary ramps, whereas a
truncated tracking record holds endpoint positions. Internal GLM spies now inspect
actual event statistics. Assertion tolerances have not been relaxed.

## Before/after counterexample

Use `time = arange(11)` and one-dimensional positions
`[0,10,0,10,0,10,0,10,0,10,0]`, with 200 uniformly spaced spikes between 0.025 and
9.975 seconds. The trajectory moves uniformly between its endpoints, so the true
spatial firing rate is constant at 20 Hz. Fit on acquisition bounds `[0,10]`,
spatial bandwidth 1, and 0.5-unit bins.

| Model | Original rate range (Hz) | Corrected rate range (Hz) |
| --- | ---: | ---: |
| Sorted KDE | 6.20–734,975.56 | 20.00000–20.06195 |
| Sorted diffusion | 5.18–36,777.09 | 20.00000–20.00000 |

Both versions use exactly 10 seconds of exposure. Resample the identical linear
trajectory to 1,001 position samples while retaining the same spikes. The
corrected KDE fields change by at most 9.54e-7 relative; corrected diffusion fields
are identical. KDE's remaining 0.31% variation reflects its finite spike sample
and boundary smoothing, rather than missing occupancy between tracking samples.

For the GMM oracle, a uniform path from 0 to 10 has mean 5 and variance 100/12.
The corrected one-component occupancy fit recovers these moments (plus the
configured covariance regularizer); an endpoint-only fit instead has variance 25.

## Construction scaling

An independent function benchmark uses a smooth two-dimensional trajectory at
500 Hz, a 3.5-unit spatial scale, and explicit acquisition bounds. Traced peaks
include NumPy allocations during the quadrature call; geometry bytes are the sum
of the six retained node arrays. These measurements describe construction, not
the full KDE/GMM/GLM fitting cost.

| Tracking samples | Nodes | Construction time | Resident geometry | Traced peak | Workspace above geometry |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 39,996 | 0.0084 s | 2.24 MB | 3.51 MB | 1.27 MB |
| 100,000 | 399,996 | 0.0808 s | 22.40 MB | 23.67 MB | 1.27 MB |
| 200,000 | 799,996 | 0.1576 s | 44.80 MB | 46.07 MB | 1.27 MB |

Time and resident geometry scale linearly; working storage remains fixed. The
200,000-sample geometry exceeds the cache budget and is correctly uncached.
The former builder accumulated per-interval arrays in Python lists and is no
longer used. Regression tests cover source/fill block boundaries and reject
extreme bandwidths before constructing unrepresentable or oversized node arrays.

## Verification status

The 20 new regressions pass, including construction resource limits and block
boundary conservation. The latest focused exposure/time/mask run passes all 49
cases; the two final source/fill block regressions also pass. The complete property
suite passes 57 cases, with 3,638 deselected and no failures. Focused sorted-model checks initially found eight
failures caused by endpoint-statistic spies and synthetic tracking-contract
assumptions; their corrected tests pass without changing tolerances. The focused
117-case time/weights/clusterless run initially passed 115 cases; its two
hard-subset tracking-contract cases now pass, together with all 17 new tests
(19/19 in that targeted verification, before the two construction regressions
were added).

Golden verification currently reports one passing and three failing original
reference comparisons, as quantified above. Full-suite verification is pending
resource coordination with the other likelihood-fix branches. Original golden
references remain available for an explicit review of this intentional algorithm
change.
