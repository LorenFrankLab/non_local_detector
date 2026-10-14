# Fitted conditional CDFs for clusterless mark calibration

## Difference and compatibility

`interval_rescaling_transform` now evaluates a fixed fitted conditional CDF at
the **observed waveform marks**. Previously it ranked the observed density
values using their own empirical CDF, which produced uniform-looking ranks by
construction and could not assess the fitted mark model.

Supply `conditional_mark_cdf` as a callable with signature
`(spike_times, observed_marks, feature_order)`. The marks retain their original
column order; `feature_order` tells the callable which sequential conditionals
to compute. Returned CDF columns follow that order. Precomputed fitted CDF
values are also accepted when feature permutation is disabled. An array cannot
be reconditioned by merely permuting its columns, so that combination is rejected.

The former positional `joint_mark_intensity` argument remains optional for
calling compatibility, but density evaluations alone cannot determine a CDF.
Calls without `conditional_mark_cdf` fail with an actionable error instead of
returning a misleading goodness-of-fit result. Invalid ground rates are checked
first, preserving existing physical-time validation. CDF outputs must have one
value per spike/feature and must be finite probabilities in `[0,1]`.

`empirical_cdf` and `rosenblatt_transform` remain available for descriptive
ranks. The latter is explicitly documented as a historical name and also has
the clearer alias `empirical_marginal_rank_transform`. These ranks do not remove
dependence and do not establish goodness of fit. Integer input now retains
fractional ranks instead of truncating them into integers.

## Why the fitted transformation works

For each held-out observed mark `m_j`, the sequential transformation evaluates

```
u_j = F_j(m_j | time/position, m_(feature_order[:j]))
```

`GaussianMixtureConditionalCDF` implements these conditional CDFs analytically.
Each Gaussian component is conditioned on observed position and preceding
marks. Its mixture responsibility is updated using their component densities;
the resulting CDF is the responsibility-weighted sum of normal CDFs. Full
covariance conditionals use covariance subblocks and a Cholesky-derived
conditional variance, avoiding subtractive cancellation in a Schur complement.

The class supports full, tied, diagonal, and spherical GMMs through `from_gmm`.
`from_kde` represents a fixed mark/joint KDE as diagonal Gaussian components,
with optional encoding positions and position bandwidth. Its diagonal path
avoids allocating full covariance matrices or zero conditional-gain products.
The standalone `gaussian_mixture_rosenblatt_transform` exposes the same
calculation for supplied fitted component parameters and optional given positions.

Component parameters are copied, read-only, and never fitted from query marks.
Query batches bound temporary arrays; Gaussian position factors are prepared
once. Only the latest feature-order recipe is cached, so repeated random
permutations cannot accumulate a factorial number of parameter caches.

## Usage

```python
from non_local_detector.model_checking.clusterless import (
    GaussianMixtureConditionalCDF,
    interval_rescaling_transform,
)

# joint_gmm was fitted on separate training data. Position coordinates precede
# the waveform coordinates in its component means/covariances.
fitted_cdf = GaussianMixtureConditionalCDF.from_gmm(
    joint_gmm,
    n_position_dims=n_position_dims,
    position_at_time=observed_position_at_time,
    batch_size=128,
)
u_intervals, u_marks = interval_rescaling_transform(
    intensity_sample_times,
    held_out_spike_times,
    held_out_waveform_features,
    ground_process_rate_hz,
    conditional_mark_cdf=fitted_cdf,
)
```

The caller must supply the conditional distribution of the model being checked.
For non-Gaussian or graph-diffusion position conditioning, provide a matching
fitted CDF callable. A conditional mark diagnostic tests both uniformity and
dependence between transformed coordinates; marginal uniformity alone cannot
detect every misspecified multivariate distribution.

## Numerical validation

1. **Difference.** Density-only rescaling previously succeeded and returned
   empirical uniform-looking ranks. The regression now rejects that call until
   a fitted conditional CDF is provided. Temporal rescaling and likelihood/HMM
   computations are unchanged; no golden/snapshot files or numerical tolerances
   were updated.

2. **Explanation.** The new output measures the probability assigned to each
   observed mark by a training-fixed model. An empirical CDF computed from the
   observations tests their ranks rather than that model. Conditioning feature
   order changes actual conditional distributions, not just output layout.

3. **Invariant checks.** Single-Gaussian and position-conditioned Gaussian
   transforms match independent analytical normal CDF references at `1e-12`.
   Mixture conditionals match independently calculated posterior component
   probabilities. All GMM covariance parameterizations, weighted KDE components,
   zero-weight excluded parameters, immutable fitted snapshots, feature-order
   permutations, finite/range validation, and bounded caches are covered.
   Query batches of 17 versus 128 gave maximum difference `0.0` in the correlated
   Gaussian demonstration.

4. **Before/after demonstration.** For 4,000 fixed-seed standard-normal held-out
   marks, former empirical density ranks give KS statistic `0.00025` and
   p-value `1.0` by construction. A fixed correct normal CDF gives statistic
   `0.0253517`, p-value `0.0114944`; a deliberately shifted model gives statistic
   `0.8703843`, p-value `0.0`. For 5,000 bivariate marks with correlation `0.85`,
   separate marginal CDFs leave transformed correlation `0.8339591`, whereas
   sequential fitted conditionals give `-0.0268026`. The tests check exact
   conditional formulas as well as these held-out distribution properties.

The complete `tests/model_checking` subset passes. Repository-wide validation is
coordinated separately; these changes do not modify fitted likelihoods or
posterior golden outputs.
