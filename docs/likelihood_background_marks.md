# A shared reference measure for clusterless marks

## Problem and resulting behavior

Active clusterless states score a waveform density, whereas the former No-Spike
state scored only spike counts. Changing waveform units therefore changed their
posterior odds. A constant lower bound on a marked intensity has the same
dimensional problem: it cannot be a normalized density over unbounded marks.

Each electrode now has one normalized, position-independent reference density
`q_e(m)`, shared across environments and encoding groups. No-Spike's intensity is
`r_0 q_e(m)`, so its count likelihood gains the observed marks' log reference
densities. Active states use the coherent low-rate component

```
lambda'_s,e(x,m) = lambda_s,e(x,m) + beta q_e(m)
Lambda'_s,e(x)   = Lambda_s,e(x)   + beta
```

`beta` is the existing numerical rate safeguard, `RATE_EPS_HZ = 5e-13 Hz`.
The event sum uses `logaddexp` before time-bin reduction, and the compensator
includes `beta` for every electrode, including zero-rate electrodes. The
background path evaluates the original marked density without its flat floor.
All four registered clusterless algorithms support this path.

If marks are changed to `m' = A m + b` with a diagonal nonsingular unit-change
matrix and model bandwidths/covariances transformed accordingly, both original
and reference densities gain the same factor `1 / |det A|`. Every state gains
the same observed-event log Jacobian; posterior odds are unchanged.

## Fitting choices and lifecycle

The reference is a diagonal Gaussian. Its location and variance are weighted
empirical moments of all user-permitted training spikes within the requested
environments and valid encoding support. Active encoding-group labels do not
restrict this shared reference: even a requested group with zero training
coverage can use legitimate training marks from other groups. Training masks
and user weights are formed on the full tracking timeline, then interpolated
once to spike times using the same valid support as the active model. Spikes
shared by multiple observation states or overlapping group masks enter once.
Decoding marks and spikes excluded by training/support/zero-weight masks do not
enter it.

A coordinate with zero empirical variance uses the configured mark-kernel
bandwidth. For GMM it uses the square root of the configured covariance
regularization. These values have waveform units and must change with the
units. Nondegenerate coordinates need no additional covariance prior.

For an electrode with no permitted marks, the detector borrows the reference
fitted from permitted training marks on other electrodes with the same mark
dimensionality. This is an **empirical borrowing assumption**: feature axes and
units must be comparable across those electrodes. Fitted
`background_mark_sources_` records each reference as `electrode`,
`pooled_same_dimension` (with donor electrode indices), or `unidentified`.
If no compatible training marks exist anywhere, empty decoding trains are
supported; a subsequently observed mark raises an actionable validation error.
No held-out location or arbitrary origin is supplied to fill that gap.

The reference is frozen during posterior-weighted encoding updates. Explicit
`fit`/`fit_encoding_model` calls rebuild it; the EM refit requests preservation.
The parameters, provenance, and dimensions survive pickle serialization.
Saved models without the background-mark contract must be refitted from
training data. Population/dimension validation precedes missing-data shortcuts.

The Gaussian reference is deliberately simple. It does not claim to fit every
mark correlation or waveform tail; changing it to a heavier-tailed or richer
family would be a separate statistical model change validated on held-out data.

## Numerics, workspace, and backend integration

KDE reference-aware scoring uses an unfloored log occupancy ratio. The kernel
maximum and normalized weighted mass stay separate until spatial numerator
and occupancy terms cancel. Mark normalizers are added after this cancellation;
otherwise large negative spatial logs can round away both weight offsets and
order-one mark evidence. Ground-rate ratios use the same separated parts.

The joint kernel is scaled separately for each output mark and position, then
evaluated by GEMM. Weights are normalized before multiplication, so common
weight rescaling does not introduce product underflow. A conservative lost-mass
bound detects suspect positive sums as well as zero sums; these use a logsumexp
fallback. Zero-weight tiles bypass the joint reduction, and the weighted
density reduction uses safe zero logs, preserving finite analytic gradients
through inactive prefixes. Encoding tiles accumulate in log space, and compiled
loops keep executable size independent of the tile count. Existing
encoding/position tile options bound both event and compensator temporary
workspace. The default dense path hoists only the current electrode's position
matrix; it does not retain a separate matrix for every electrode or chunk.

Diffusion applies its spatial operator to training-standardized mark kernels
and restores the mark Jacobian after transport. The standardized encoding
marks are cached once per prediction, while selected decoding marks are
standardized once per electrode/chunk. Its existing memory budget caps event
block size. GMM/diffusion raw-backend hooks consume the independent
joint/ground normalization repair when that change is installed.

Standalone lower-level predictors without a supplied reference retain their
historical scores. Detectors always supply the shared training-fitted reference.
The underlying GMM/diffusion joint-ground identity is a separate repair; adding
a normalized background component does not itself repair an inconsistent
original intensity.

## Numerical validation

1. **Difference.** Before the repair, scaling two waveform coordinates and
   bandwidths by 10 changed No-Spike versus active log odds by `4.6051702` per
   spike. Afterward the regression passes at its original `1e-5` absolute
   tolerance for KDE, log-KDE, and diffusion. GMM tests transform the same
   fitted physical model and cover full/tied/diagonal/spherical covariances.
   (Independent mark scaling promotes a spherical covariance to diagonal.)

2. **Explanation.** No-Spike now scores marks with respect to the same continuous
   reference measure as active states. The additive rate's integral is included
   in the silent-bin score, instead of flooring events alone. This intentionally
   changes No-Spike odds in existing recordings.

3. **Invariant checks.** Numerical quadrature of the Gaussian reference yields
   total mass one. Local and non-local zero-rate electrode scores equal
   `log(beta*q(m)*dt) - beta*dt`. Weighted fitting excludes zero-weight NaNs;
   training masks exclude held-out spikes; posterior refits preserve the same
   reference. Pickle, row slices, dense/checkpointed inference, and all four
   backend covariance/zero-rate cases are covered. Unequally weighted repeated
   spatial samples preserve Gaussian mark and silent-bin evidence at positions
   200 and 1457 bandwidths away. The adversarial positive-GEMM case recovers 256
   small contributions that a plain float32 sum can omit, including uniformly
   scaled weights of `1e-10`. Analytic point/sample/weight gradients stay finite
   through zero-weight prefixes. An instrumented 37-position evaluation checks
   that every occupancy/GPI query respects a configured position tile of four.

4. **Pinned before/after case.** Three golden tests remain unchanged. The
   NonLocalClusterlessDetector golden case intentionally changes: maximum
   posterior/state-probability absolute difference `0.0011673723347485065`, mean
   posterior absolute difference `4.023980636702618e-6`, and mean state-probability
   absolute difference `0.00017032473988365382`. The exact largest change is:

   | Pinned array entry | Original | Corrected |
   | --- | ---: | ---: |
   | `posterior[44, 1]` (No-Spike) | `0.004626445006579161` | `0.003459072671830654` |
   | `state_probs[44, 1]` (No-Spike) | `0.004626445006579161` | `0.003459072671830654` |

   Both state-category and posterior-bin argmax comparisons have **zero changes
   across all 50 rows**. All values are finite; posterior and state-probability
   maximum float32 normalization errors are `3.5762786865234375e-7`. The fixture
   includes an electrode
   with no training spikes, which uses the documented pooled reference.
   **No golden/snapshot files or numerical tolerances were updated.**

Final focused validation has **271 passing tests and the one explained NonLocal
golden mismatch** in 40.33 seconds. This includes all background regressions,
canonical event ownership, model unit compatibility, probability/HMM properties,
and four pinned golden tests. The two independent numerical/gradient regression
files first reproduced 27 failures, then passed all 41 cases after repair.

The earlier full suite had **3683 passing, seven skipped, and eight failing
tests** in 1396.91 seconds. Four failures concerned training coverage now fixed
by the permitted-reference scope; two used the former count-only No-Spike
expectation and now use the fitted Gaussian's mathematical expectation. One
Hypothesis property exceeded its unchanged 200 ms deadline during first-call
JAX compilation but passed on rerun; no probability assertion failed. The final
focused run passes all seven affected cases/properties. The remaining golden
difference is documented above. No tolerances, deadlines, convergence criteria,
or reference outputs were changed.

On the shared CPU, a warmed 20-run median with 1000 encoding spikes, 1000 decode
spikes, four mark coordinates, 72 spatial bins, and event blocks of 100 is
**2.33 ms for the legacy standalone path and 6.50 ms for the coherent background
path** (2.79 times the runtime, an additional 4.17 ms). The background path was
11.53 ms before hoisting the current electrode's spatial matrix. These are
shared-machine measurements, not a controlled GPU comparison; the extra
normalized reference and stable per-event reductions have a material cost.
