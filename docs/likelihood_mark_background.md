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

- It is a diagonal Gaussian with the weighted mean and variance of the training marks,
  taken from every encoding group of the modelled environments.
- A constant training feature takes the mark bandwidth (`waveform_std`, or
  `sqrt(gmm_reg_covar)`) as its scale.
- An electrode without training spikes has no reference.
- Detectors share one reference across all states. Posterior-weighted encoding updates
  in `estimate_parameters` keep it fixed.

The marked intensities, with `β = RATE_EPS_HZ = 5e-13 Hz`, are:

| State | Marked intensity | Expected count |
| --- | --- | --- |
| No-Spike | `r₀ q_e(m)` | `r₀` per electrode |
| Active state | `λ_e(x, m) + β q_e(m)` | `Λ_e(x) + β` per electrode |

Under `m' = A m + c`, with bandwidths or covariances transformed alike, every density
gains the same factor `1/|det A|`. Every spike's term therefore shifts by
`-log|det A|` in every state at every position, and the posterior is unchanged.
Without a reference, both the active background and No-Spike omit `q`, so the electrode
remains consistent.

`β` is a unit-consistent numerical floor, not a fitted background rate. Because
`β < r₀` (default `r₀ = 1e-10 Hz`), a spike no state can explain favours No-Spike over
every active state by `log(r₀/β)` (+5.3 nats). An artifact spike therefore does not
break a stretch of population silence, which is what the No-Spike state models.

## Numerics

Each spike's term is `logaddexp(log λ, b)`, with `b = log β + log q(m)` in the
backend's units:

- KDE: `LOG_EPS + log q`, because its rates are pre-multiplied by 2 ms.
- GMM and diffusion: `LOG_RATE_EPS_HZ + log q`.

A zero intensity (zero rate, no occupancy, zero mass) gives `b`. NaN propagates.

**Exactness.** In float32, `logaddexp(x, b) == b` whenever `x < b + log(eps)`, which is
about `b − 15.9`. So `λ` must be exact only above that per-spike threshold. On a real
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

Field values below `_FIELD_RESOLUTION · max(column)`, where
`_FIELD_RESOLUTION = max(float32 eps, heat-kernel rank tolerance) = 1e-6`, are
therefore held at that level. This is unit-consistent, because the floor scales with
each spike's own peak. It keeps about 14 nats of peak-versus-tail contrast per spike
and only overestimates evidence in deep tails.

Occupancy is a diffused field too. Bins whose occupancy is below the same
`1e-6` of its maximum are unvisited, so the heat kernel does not resolve them.
They get zero marked and ground intensity, and their spikes are explained by the
background alone, as for KDE's zero-occupancy bins. Without this rule, a floored
mark field divided by near-zero occupancy gave unvisited bins up to ~100 nats of
spurious evidence.

The floor also bounds how strongly a single spike can exclude a graph-disconnected
component, to about 14 nats; the exact intensity there is zero.

## Validation

- `tests/likelihoods/test_clusterless_mark_background.py` covers:
  - the Jacobian identity `LL' + n_spikes · log|det A| = LL` for every algorithm and its
    tiled or streamed path;
  - outlier marks scoring exactly `b`;
  - zero-rate electrodes;
  - the `n_e β` compensator.
- `tests/models/test_no_spike_mark_reference.py` covers:
  - state odds that do not depend on the units;
  - the outlier odds `log(β/r₀)`;
  - the No-Spike formula, the EM-frozen reference, chunk parity and saved-model refit.
- `TestScaledGemmJointIntensity` compares the KDE reduction with a logsumexp
  oracle within 1e-4. At 8 and 9 mark dimensions, 100% of the compared terms
  lie below the old `LOG_EPS` floor yet above the background.
