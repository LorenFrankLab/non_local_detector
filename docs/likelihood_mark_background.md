# Unit-consistent clusterless mark evidence

## Problem

A clusterless state scores each spike with a marked intensity `λ(x, m)`. That is a
density over waveform features `m`, so its numeric value carries the inverse volume of
the feature units (for example per µV⁴ for four amplitudes). Two parts of the model
did not carry that factor:

- **No-Spike** scored only the spike count, at `no_spike_rate` Hz.
- **Active states** floored each spike's log intensity at a fixed constant (`LOG_EPS`
  for KDE, `LOG_RATE_EPS_HZ` for GMM/diffusion). A floor in absolute units also does
  not scale with the feature units or with the number of features.

As a result, the odds between states depended on how the marks were expressed:

- **Units:** on a 22-tetrode recording, expressing the amplitudes in 10× units moved
  the mean No-Spike probability from 0.097 to 0.9998 over a 60 s decode.
- **Number of features:** with 8 or more µV features at `waveform_std=24`, every
  single-spike bin of a simulated run was floored to `LOG_EPS` at every position. The
  spikes then carried no position information at all.

## Model

Each electrode `e` has one normalized reference density `q_e(m)`, `mark_reference_`:

- It is a product of per-feature Student-t densities with 3 degrees of freedom,
  centred on the weighted mean of the training marks from every encoding group of
  the modelled environments.
- Each feature's scale is `sqrt(training variance + h²)`, the spread of the marks
  after the model's mark kernel, with `h` the mark bandwidth (`waveform_std`, or
  `sqrt(gmm_reg_covar)`). A constant feature therefore has scale `h`.
- Its tails fall polynomially, so far from the training marks `q` exceeds every
  Gaussian mark kernel and every GMM component, whatever their widths.
- An electrode without training spikes has no reference.
- Detectors fit it once and pass it to every state; encoding models do not carry it.
  Direct callers of the predict functions fit it with `fit_mark_references`.
- Posterior-weighted encoding updates in `estimate_parameters` keep it fixed.

The marked intensities, with `β = RATE_EPS_HZ = 5e-13 Hz`, are:

| State | Marked intensity | Expected count |
| --- | --- | --- |
| No-Spike | `r₀ q_e(m)` | `r₀` per electrode |
| Active state | `λ_e(x, m) + β q_e(m)` | `Λ_e(x) + β` per electrode |

Under `m' = A m + c` with `A` diagonal (a per-feature change of units, offset,
sign or order), and with bandwidths or covariances transformed alike, every
density gains the same factor `1/|det A|`. Every spike's term therefore shifts
by `-log|det A|` in every state at every position, and the posterior is
unchanged. A transform that mixes features does not qualify: KDE bandwidths and
`q` are diagonal, so they cannot follow it.
Without a reference, both the active background and No-Spike omit `q`, so the electrode
remains consistent.

`β` is a unit-consistent numerical floor, not a fitted background rate. A spike
far enough from every training mark has `λ ≪ β q` at every position, because
`q`'s polynomial tails outlast the Gaussian kernels. Then, because `β < r₀`
(default `r₀ = 1e-10 Hz`), it favours No-Spike over every active state by
`log(r₀/β)` (+5.3 nats). An artifact spike therefore does not break a stretch of
population silence, which is what the No-Spike state models. A Gaussian `q` with
the training variance alone did not guarantee this: a feature whose training
spread is narrower than `h` gives `q` thinner tails than the kernels. On a real
recording, artifact spikes in injected silences then held No-Spike at 0.66–0.78
for KDE and 0.60–0.68 for GMM, instead of 0.999.

## Numerics

Each spike's term is `logaddexp(log λ, b)`, with `b = log β + log q(m)` in the
backend's units:

- KDE: `LOG_EPS + log q`, because its rates are pre-multiplied by 2 ms.
- GMM and diffusion: `LOG_RATE_EPS_HZ + log q`.

A zero intensity (zero rate, no occupancy, zero mass) gives `b`. NaN propagates.

**Exactness.** In float32, `logaddexp(x, b)` differs from `b` by less than `eps`
whenever `x < b + log(eps)`, which is about `b − 15.9`. For `|b| ≥ 2` that is
below one rounding step of `b`, so the term rounds to exactly `b`. So `λ` must be
exact only above that per-spike threshold. On a real
recording, `b` lies about 19 nats below the old `LOG_EPS` floor (median −53.2 against
−34.5), so values that the floor used to hide must now be computed.

| Backend | How it meets the threshold |
| --- | --- |
| **KDE exactness test** | `_log_marginal_floor` is separable into a per-spike and a per-bin part and compares against each spike's threshold instead of `LOG_EPS`. Spikes that may have lost mass above it are recomputed with a direct logsumexp. |
| **KDE** | One log-space estimator (`clusterless_kde`). A probability-space sum loses up to `n·tiny/W` per entry, and on a real recording every electrode's threshold lay 5–146 nats below what float32 could resolve, so the probability-space implementation was removed. The scaled-GEMM gate decides per spike, and only flagged spikes get the exact logsumexp: 6 of 16,094 real spikes. |
| **GMM** | Log-space throughout; no change. |
| **Diffusion** | See below. |

**Diffusion.** The heat-kernel field `P` resolves only about `1e-6` of each spike's peak:

- Against a float64 full-rank heat kernel on a 723-bin real 2-D grid, the float32
  field's median relative error was 4.5e-4 at 1e-6 to 1e-4 of the peak.
- It was 0.15 or more below 1e-8, and values there are often exactly 0.
- Each spike would need a median of 66 nats of range, beyond even float64's ~36.

Diffusion therefore uses one fixed spatial kernel `H` (`_spatial_kernel`): column
`s` is the heat kernel applied to a unit mass at bin `s`, clipped to be
non-negative, zero outside `s`'s connected component, cut below
`_FIELD_RESOLUTION · max(column)` (where `_FIELD_RESOLUTION = max(float32 eps,
heat-kernel rank tolerance) = 1e-6`) and normalized to unit mass. Occupancy, the
ground process and every spike's marked field are this one linear map of their
sources:

`λ(x, m) = c(x) Σᵢ wᵢ H[x, bᵢ] K(m; mᵢ)` and `Λ(x) = c(x) Σᵢ wᵢ H[x, bᵢ]`,

with `c(x) = r / (Σᵢ wᵢ · dV(x) · occupancy(x))`. Because each mark kernel
integrates to one, `∫ λ dm = Λ` exactly, and a disconnected component stays exactly
zero. Clipping or flooring each spike's diffused mark column instead does not
commute with integrating over marks: such a floor had added intensity that `Λ`
never counted (0.5 Hz in a graph-disconnected component where `Λ` is only `β`).
Model checking uses the same kernel columns and the same unresolved-occupancy
mask, so its intensity is the decoder's.

Each spike's mark kernels are combined per source bin in log space (a log-sum-exp
over that bin's encoding samples), so no source is lost however far its marks lie
from another's. The spatial product is then applied in each spike's own scale:
relative to its strongest source, with the offset added back in log space, so the
kernel normalizer, which carries the mark units, cannot underflow a spike's field.
A source more than 80 nats below the strongest may still underflow in that
product. Each source's largest possible contribution to any bin is known (its log
mark density plus `max_x log(c(x) H[x, s])`), so a spike is recomputed by an exact
log-space reduction over its sources whenever such a source could reach its
background floor. This keeps, for example, the only source reaching a
graph-disconnected component even when another component has a much stronger
source. On a real recording no held-out spike needed it (0 of 75,458).

When the whole `(n_bins, n_bins)` kernel fits half of `memory_budget` (5,053 bins
at 1 cm on a real recording still does), it is built once per prediction and
shared by every electrode, state and chunk. Otherwise columns are built from the
low-rank heat-kernel basis in tiles sized from `memory_budget`, for just the
source bins that hold encoding samples, so memory grows with `n_bins × tile`
rather than `n_bins²`: a 20,000-bin grid compiles within a 256 MiB workspace,
where the whole kernel alone would take 1.6 GB. Integrating the predicted intensity over a fine mark grid recovers `Λ + β`
to float32 rounding (8e-7).

Occupancy is a diffused field too. Bins whose occupancy is below the same
`1e-6` of its maximum are unvisited, so the heat kernel does not resolve them.
They get zero marked and ground intensity, and their spikes are explained by the
background alone, as for KDE's zero-occupancy bins. Without this rule, a mark
field divided by near-zero occupancy gave unvisited bins up to ~100 nats of
spurious evidence.

## Validation

- `tests/likelihoods/test_clusterless_mark_background.py` covers:
  - the Jacobian identity `LL' + n_spikes · log|det A| = LL` for every algorithm and its
    tiled or streamed path;
  - outlier marks scoring exactly `b` with the fitted reference;
  - diffusion marks 7 bandwidths from training in microvolts and millivolts;
  - zero-rate electrodes;
  - the `n_e β` compensator.
- `tests/models/test_no_spike_mark_reference.py` covers:
  - state odds that do not depend on the units;
  - the outlier odds `log(β/r₀)` for KDE, GMM and diffusion with the fitted
    reference and mark kernels wider than the training spread;
  - the No-Spike formula, the EM-frozen reference, chunk parity and saved-model refit.
- `tests/likelihoods/test_clusterless_diffusion.py` covers the predicted
  intensity integrated over marks (the expected count), a spike's term in a
  graph-disconnected room (the background, in every bin), and diffused fields
  with 4 and 16 features that differ between microvolts and millivolts by exactly
  the Jacobian.
- `TestScaledGemmJointIntensity` compares the KDE reduction with a logsumexp
  oracle within 1e-4. At 8 and 9 mark dimensions, 100% of the compared terms
  lie below the old `LOG_EPS` floor yet above the background.
