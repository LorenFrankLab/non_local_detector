# Goodness of fit for clusterless encoding models

`non_local_detector.model_checking.clusterless` tests whether a fitted
clusterless encoding model describes one electrode's spikes, using the
marked point process transforms of Yousefi, Amidi, Nazari and Eden (2020,
*Neural Computation* 32, 2145-2186). A transform maps each spike (its time and
its `d` waveform features) to a point in `[0, 1]^(d+1)`. For spikes generated
by the model the points are uniform, so a uniformity test of the points tests
the model.

```python
from non_local_detector.model_checking.clusterless import (
    ircm_transform,
    ks_uniformity_test,
    mdci_transform,
    multivariate_ks_test,
    pearson_chi_squared_test,
)

# A clusterless detector (such as ClusterlessDecoder) fitted on training data;
# its encoding_model_ is keyed by (environment name, encoding group).
encoding_model = detector.encoding_model_[("", 0)]
u, v = ircm_transform(
    encoding_model,
    detector.clusterless_algorithm,
    electrode,
    held_out_position_time,
    held_out_position,
    held_out_spike_times[electrode],
    held_out_spike_waveform_features[electrode],
)
points = np.column_stack((u, v))
ks_uniformity_test(points).p_value      # one p-value per coordinate
pearson_chi_squared_test(points).p_value
multivariate_ks_test(points, rng=0).p_value
```

Use data the model was not fitted to: a KDE scores its own encoding spikes
too well, and the points then look more uniform than the model deserves.

## The model being tested

Every supported likelihood defines a joint mark intensity
`lambda(x, m) = Lambda(x) f(m | x)`. `Lambda(x)` is the ground intensity in Hz
at position `x`, and `f(m | x)` is a Gaussian mixture over marks:

- `clusterless_kde` and `clusterless_kde_log`: one component per encoding
  spike, centred on its marks with bandwidth `waveform_std`, weighted by its
  encoding weight times the Gaussian position kernel at `x`.
  `Lambda(x) = mean_rate * p_gpi(x) / p_occupancy(x)`.
- `clusterless_diffusion`: the same components, weighted by the graph heat
  kernel from the spike's bin to the bin containing `x`. The model is defined
  on interior bins. At full rank this mixture is the model; at a truncated
  rank the decoder clips each diffused mark column, and the mixture
  approximates it.
- `clusterless_gmm`: the joint mixture conditioned on `x`. The decoder
  integrates `mean_rate * p_gpi(x) / p_occupancy(x)` over time but scores
  spikes with `mean_rate * p_joint(x, m) / p_occupancy(x)`. Those two are fitted
  separately, and the second integrates over marks to
  `mean_rate * p_joint(x) / p_occupancy(x)` instead of the first. So no single
  marked point process has both. The transforms keep the decoder's ground
  intensity and its conditional mark density,
  `lambda(x, m) = Lambda(x) p_joint(x, m) / p_joint(x)`. The position-dependent
  factor `p_joint(x) / p_gpi(x)` that the decoder also applies to every spike
  is not tested.

All intensities and CDFs come from the stored encoding model, in float64
NumPy and log space. Positions are interpolated to the spike times and
linearized for track graphs as during fitting.

## Transforms

**IRCM** (interval rescaling, conditional mark distribution; the paper's
Algorithm 1). Each interspike interval is rescaled by the ground intensity
along the trajectory, and each spike's marks by the sequential (Rosenblatt)
conditional CDFs of `f(m | x)` at the spike's position:

```text
u_i     = 1 - exp(-integral from s_(i-1) to s_i of Lambda(x(t)) dt)
v_i^(l) = F(m_i^(l) | m_i^(1), ..., m_i^(l-1), x(s_i))
```

The first interval starts at `position_time[0]`. The ground intensity is
evaluated at every position sample and integrated as a piecewise-linear
function of time. For a correct model the points are independent and uniform.
A nonuniform `u` points to the rate model, and a nonuniform `v^(l)` with
uniform earlier columns points to the model of feature `l`.

**MDCI** (mark density, conditional intensity; Algorithm 2). Marks are
transformed by the sequential CDFs of the electrode's mark density over the
whole period, `f(m) = Gamma(m) / integral Gamma`, with
`Gamma(m) = integral from 0 to T of lambda(x(t), m) dt`. Each spike's time is
then transformed by the distribution of times given its marks:

```text
u_i = integral from 0 to s_i of lambda(x(t), m_i) dt / Gamma(m_i)
```

The unordered points are uniform. They are not independent in time order.
These models depend on time only through position, so given their number the
points are an independent sample, and all three tests below apply. The paper
argues that MDCI, which keeps the spikes of one unit close together, is the
more sensitive transform for misfit of individual units, and that IRCM, which
mixes units, is more sensitive to misfit of the timing and interactions
between spikes.

The time integrals of MDCI use the interior place bins, as the non-local
likelihood does. Each position sample is assigned to its bin, and the
integrand is interpolated linearly in time between samples. An interval gives
half its duration to each endpoint's bin, and a spike inside an interval
receives the exact integral of the two interpolation weights. The error of
this approximation shrinks with the bin size relative to the position
bandwidth. `f(m)` is again a Gaussian mixture, over (bin, component) pairs for
the GMM, so the mark CDFs are exact.

Both transforms take `feature_order`, the order in which features are
conditioned. Every order gives uniform points for a correct model.
`interval_rescaling_transform` is IRCM for any model given as a sampled
ground intensity and a conditional mark CDF callable;
`GaussianMixtureConditionalCDF` implements such a callable for Gaussian
mixtures and KDEs.

## Uniformity tests

- `ks_uniformity_test`: a one-sample Kolmogorov-Smirnov test of each
  coordinate. Each coordinate has its own p-value; compare their minimum with
  `alpha / (d + 1)` to test them together. Uniform coordinates do not imply
  jointly uniform points.
- `pearson_chi_squared_test`: counts in `M = k^(d+1)` equal sub-cubes against
  `n / M` each, with `M - 1` degrees of freedom. By default `k` is the largest
  number, at least 2, that leaves five expected points per sub-cube.
- `multivariate_ks_test`: the largest gap between the points' empirical CDF
  and the uniform CDF `prod_l z_l`, at each point and just below it. The null
  distribution depends on the number of points and dimensions, so the p-value
  comes from `n_simulations` uniform samples of the same size (default 200,
  so the smallest p-value is 1/201). The statistic costs
  `O(n^2 (d + 1))`, and inputs above `max_samples` (default 2,000) are
  subsampled without replacement. It detects dependence between coordinates
  that the other two tests can miss.

The paper found the chi-square and multivariate KS tests the most powerful
of the six it compared. The other four (distance to boundary, discrepancy,
Ripley statistics, minimal spanning tree) are not implemented.

## Example: a correct model and two misspecified ones

This example simulates one electrode in continuous time: three units with
Gaussian place fields on a 100 cm track and 2-D Gaussian mark clusters. It then
applies IRCM under the true model and under two wrong ones: one whose mark
clusters are 1.5 times too wide, and one that ignores the place fields and
fires at the mean rate everywhere. Because each place field is proportional to
a Gaussian in position, the true `f(m | x)` is exactly a
`GaussianMixtureConditionalCDF` over `(x, m)`.

```python
import numpy as np

from non_local_detector.model_checking.clusterless import (
    GaussianMixtureConditionalCDF,
    interval_rescaling_transform,
    ks_uniformity_test,
    multivariate_ks_test,
    pearson_chi_squared_test,
)

rng = np.random.default_rng(0)
duration, peak_rate, field_std = 300.0, 20.0, 8.0
place_means = np.array([20.0, 50.0, 80.0])
mark_means = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 4.0]])


def trajectory(t):
    return 50.0 + 45.0 * np.sin(2 * np.pi * t / 20.0)


def unit_rates(x):
    return peak_rate * np.exp(-0.5 * ((x[:, None] - place_means) / field_std) ** 2)


# Simulate in continuous time by thinning a Poisson process of rate 3 * peak_rate.
n_candidates = rng.poisson(3 * peak_rate * duration)
candidates = np.sort(rng.uniform(0.0, duration, n_candidates))
rates = unit_rates(trajectory(candidates))
keep = rng.uniform(size=n_candidates) * 3 * peak_rate < rates.sum(axis=1)
spike_times, rates = candidates[keep], rates[keep]
unit_cdf = np.cumsum(rates / rates.sum(axis=1, keepdims=True), axis=1)
unit = (rng.uniform(size=len(rates))[:, None] > unit_cdf).sum(axis=1)
marks = mark_means[unit] + rng.standard_normal((len(unit), 2))

# Ground intensity at 500 Hz position samples.
time = np.arange(0.0, duration + 1e-9, 0.002)
ground = unit_rates(trajectory(time)).sum(axis=1)


def mark_model(mark_std):
    # Each place field is proportional to a Gaussian in x, so f(m | x) is a
    # Gaussian mixture over (x, m) conditioned on x.
    return GaussianMixtureConditionalCDF(
        np.ones(3),
        np.column_stack((place_means, mark_means)),
        np.column_stack((np.full(3, field_std**2), np.full((3, 2), mark_std**2))),
        covariance_type="diag",
        n_position_dims=1,
        position_at_time=lambda t: trajectory(t)[:, None],
    )


constant_rate = np.full_like(time, len(spike_times) / duration)
models = {
    "true model": (ground, mark_model(1.0)),
    "marks too wide": (ground, mark_model(1.5)),
    "constant rate": (constant_rate, mark_model(1.0)),
}
print(f"{len(spike_times)} spikes")
for name, (ground_intensity, mark_cdf) in models.items():
    u, v = interval_rescaling_transform(
        time, spike_times, marks, ground_intensity, mark_cdf
    )
    points = np.column_stack((u, v))
    chi2 = pearson_chi_squared_test(points)
    print(
        f"{name}: KS p (u, v1, v2) {ks_uniformity_test(points).p_value}, "
        f"chi-square p {chi2.p_value:.3g} ({chi2.n_bins_per_dim ** 3} cells), "
        f"multivariate KS p {multivariate_ks_test(points, rng=0).p_value:.3f}"
    )
```

| Model | KS p, `u` | KS p, `v1` | KS p, `v2` | Chi-square p | Multivariate KS p |
| --- | --- | --- | --- | --- | --- |
| true model | 0.45 | 0.78 | 0.012 | 0.74 | 0.52 |
| marks too wide | 0.45 | 1.1e-30 | 4.1e-32 | 4.4e-148 | 0.005 |
| constant rate | 6.4e-11 | 0.78 | 0.012 | 0.25 | 0.005 |

With 3,464 spikes:

- **True model.** Every p-value is a draw from U(0, 1). Across 40 simulations
  of this model with different seeds, each coordinate's KS p-value fell below
  0.05 in 5% to 7.5% of them, as it should. In this simulation `v2`'s 0.012
  is such a draw: it would fail a Bonferroni test of the three coordinates at
  0.05 (0.017), which a correct model does 5% of the time.
- **Marks too wide.** The marks fall near the middle of the model's too-wide
  mark distributions, so `v` crowds toward 0.5. `u` is unchanged because the
  rate model is right.
- **Constant rate.** `u` departs from uniform, while `v` is exactly as under
  the true model, because the mark model is right. The chi-square test misses
  this: its 512 sub-cubes spread a departure in one coordinate thin. The
  per-coordinate KS test finds it, and so does the multivariate KS test, whose
  p-value is at its floor of 1/201 for 200 simulations.

The misfit coordinate identifies the misfit component: `u` for the rate model,
`v` for the mark model.

## Interpreting results

- **Spike times on a coarse grid bias IRCM's `u`.** The rescaling assumes
  that spike times are continuous. When spikes are recorded only to a sample
  grid on which `Lambda * dt` is not small, the intervals are discrete and
  `u` is not uniform even for the true model. The package simulator
  (`make_simulated_run_data`) draws spikes on a 1 ms grid with up to 0.098
  probability per sample. Under its true intensity, held-out IRCM `u` on
  tetrode 0 (4,515 spikes, seed 0) gave KS p = 5.5e-19, while the
  discrete-time rescaling of Haslinger, Pipa and Brown (2010) gave p = 0.59.
  MDCI's `u`, which normalizes over the whole period, gave p = 0.46 under the
  same intensity. Use spike times at acquisition resolution.
- **Thousands of spikes resolve kernel smoothing.** A KDE's bandwidths smooth
  the place fields and mark clusters it estimates. On the same simulation,
  fitted on the first half, the smallest per-coordinate KS p-value of IRCM's
  marks was 5.7e-145 at `position_std=6, waveform_std=2` and 1.5e-5 at
  `1.5, 0.5`. A rejection says where the model departs from the data (here,
  in its marks), not that the decoder is unusable.

## Differences from the reference implementation

The MATLAB code at
<https://github.com/YousefiLab/Marked-PointProcess-Goodness-of-Fit>
accompanies the paper. This module differs as follows:

- Intensities are rates in Hz integrated over time. The reference uses
  probabilities per time step (`GenSpike.m` draws a spike when the intensity
  exceeds a uniform draw).
- Mark CDFs are exact Gaussian mixture CDFs at the observed marks. The
  reference sums the intensity over a grid of marks and rounds each mark to
  the grid (`IRCM.m` lines 32-34, `MDCI.m` lines 13-20).
- `Gof_Pearson.m` line 12 tests column `d` for every coordinate, so its
  statistic counts only the last coordinate in five bins. Here every
  coordinate defines the sub-cubes.
- `Gof_MKS.m` line 26 sorts the uniform CDF values separately from the
  empirical CDF values they are compared with, which breaks their pairing.
  Here both are evaluated at the same point.

## Validation

`tests/model_checking/test_marked_point_process_gof.py` fits each backend on
a simulated recording (a 100 cm track, two units on one electrode with
separate place fields and mark clusters). The tests simulate held-out spikes
from each fitted model with an independent implementation of its intensity.

- **Calibration.** For every backend, IRCM and MDCI points of spikes
  simulated from the fitted model have every per-coordinate KS p-value above
  1e-3. Marks shifted by 1.5 bandwidths give p < 1e-6 for the first mark
  coordinate, and IRCM with a doubled rate gives p < 1e-6 for `u`.
- **Exactness.** IRCM mark CDFs equal the model's conditional mixture computed
  independently (relative tolerance 1e-7). MDCI `u` and `v` equal a direct sum
  over bins. The tolerance is 5e-5 relative for the two KDE backends, whose
  stored occupancy is float32 with up to 2.9e-5 relative error, and 1e-6
  otherwise.
- **Conditioning.** `tests/model_checking/test_mark_calibration.py` checks the
  conditional mixture CDFs against closed forms and numerical quadrature,
  including a three-component full-covariance mixture over two position and
  two mark coordinates (absolute tolerance 1e-7).
- **Tests.** The three uniformity tests accept uniform samples, reject
  nonuniform ones, and (multivariate KS) reject dependent coordinates with
  uniform marginals.

On an Apple M1 Max CPU, with 174,999 held-out position samples, 175 bins and
4,515 spikes with 4 features against 4,458 encoding spikes
(`clusterless_kde`), IRCM took 15.0 s and MDCI 3.4 s.
