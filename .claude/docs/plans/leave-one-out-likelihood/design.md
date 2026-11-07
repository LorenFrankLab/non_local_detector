# A statistically proper Local state: scoring encoding-period spikes out of sample

Written 2026-11-06 against main `e55cff9f`. Not started. This is a design, not an
implementation plan: it fixes what the change means and how it is judged, and leaves
the code structure to the builder. Re-verify every line number before relying on it.

## Goal

The Local state exists to capture local coding of the animal's position at the
population level. For that, its likelihood has to be what a model that had not seen a
spike would assign to it. Today, when the decoded data were also used to fit the
encoding model, every spike helps to explain itself, and the Local state's advantage
over the other states is partly that.

Make the likelihood of encoding-period data an out-of-sample likelihood, for every
likelihood model, by default. This is a property of the model, not an option for
careful users. Decoding the encoding period should then behave like decoding held-out
data, without the user having to split the session.

## The requirement, for every likelihood model

A fitted model must have no advantage on the data it was fitted to. Two tests define
it, and every stage below is judged by both:

1. **No in-sample excess.** Simulate several sessions from the same cells and
   trajectory with independent spikes, and fit a detector on each. A session's Local
   log likelihood under its own model, minus its mean under the other sessions'
   models, is zero within sampling error. `evidence/local_in_sample_optimism.py`
   measures this through the package's own code.
2. **Agreement with a refit.** For a single spike, the value equals (exact methods) or
   closely approximates (one-step methods) the value from a model refitted without
   that spike.

## The problem, shown for the clusterless KDE

The KDE likelihoods store every encoding spike and score a decoded spike by summing
kernels over them. For the clusterless KDE the marked intensity of electrode `e` is

    lambda_e(x, m) = sum_i w_i K_x(x - x_i) K_m(m - m_i) / (T occupancy(x))

where `i` runs over the electrode's encoding spikes (position `x_i` at the spike time,
mark `m_i`, weight `w_i`), `T` is the weighted encoding time in seconds and
`occupancy` is the occupancy density. The package writes this as
`mean_rate * joint density / occupancy`; the mean rate and the density's normalizer
cancel to the form above.

If decoded spike `j` is itself encoding spike `j`, the sum contains its own term,
`w_j K_x(x - x_j) K_m(0)`. That term is the largest mark kernel in the row and, at the
animal's position (the Local state), also the largest position kernel. A model fitted
without spike `j` would not have it. Nothing in the package removes it:

- No likelihood path has leave-one-out or own-spike handling (searched
  `src/non_local_detector/likelihoods` and `models` for leave-one-out, own-kernel,
  exclude-self and cross-fit wording; no matches).
- The clusterless KDE fit keeps the encoding marks, positions and weights but not the
  encoding spike times (`clusterless_kde.py:1833-1858`), so predict cannot tell which
  decoded spikes are encoding spikes.

Leaving the term out is exact, not an approximation: dropping it from the sum gives
precisely the intensity of a model refitted without spike `j`, because the refit's
smaller mean rate and smaller normalizer cancel in the same way.

For a Poisson spike train this is also the complete correction. Given that a spike
occurred at `j`, the rest of a Poisson process is distributed as if it had not
(Slivnyak's theorem), so the other spikes of the same pass are legitimate data. Real
spike trains are not Poisson: bursts and lap-to-lap rate changes make spikes close in
time to `j` more common than the fitted rate predicts. Removing that effect needs a
time window (or blocked folds) with the matching occupancy removed. The real
recordings below show both parts: the own kernel, and a smaller remainder it does not
remove.

## Evidence

Scripts and outputs are in `evidence/`. Every number is the Local term only: the
intensity at the animal's position and the spike's own mark.

### Simulation (`own_kernel_share.py`, output in `own_kernel_share_output.txt`)

Package simulator, 5 electrodes, default bandwidths (`position_std` sqrt(12.5) cm,
`waveform_std` 24). "Noise" is Gaussian noise added to every mark feature, because the
simulator's marks are much cleaner than recorded ones. "Share" is the fraction of a
spike's kernel sum that is its own kernel; leaving it out changes the log intensity by
`log(1 - share)`.

| Mark dims | Noise SD | Laps | Mean own share | Spikes with share > 0.5 | Mean change, nats/spike |
| --- | --- | --- | --- | --- | --- |
| 4 | 0 | 10 | 0.001 | 0 | 0.001 |
| 4 | 0 | 40 | 0.000 | 0 | 0.000 |
| 4 | 20 | 10 | 0.011 | 0 | 0.011 |
| 8 | 20 | 10 | 0.073 | 0.8% | 0.082 |
| 32 | 10 | 10 | 0.207 | 3.4% | 0.249 |
| 32 | 20 | 10 | 0.999 | 100% | 8.669 |
| 32 | 20 | 40 | 0.997 | 100% | 7.119 |

- With clean, low-dimensional marks the own kernel is negligible.
- It grows with the spread of a unit's marks times the number of mark dimensions,
  relative to `waveform_std`. At 32 dimensions and noise 20 the in-sample intensity is
  almost entirely the spike matching itself.
- Spikes within 2 s of the spike (the same pass) are 5-6% of the kernel sum at 10 laps
  and 1.4% at 40 laps. In this Poisson simulation that is data, not bias.

### Real recordings (`own_kernel_share_real.py`)

Run 2026-11-06 on breeze; outputs in `own_kernel_share_real_rs58.txt` and
`own_kernel_share_real_j16.txt` (originals in `/cumulus/edeno/loo_gate/`). The script
scores a sample of 1,500 spikes per electrode four ways from the same encoding spikes:
nothing left out (what the package does today), the spike's own kernel left out, a 2 s
window left out, and the spike's whole time block left out (5 blocks; the last two
with the matching occupancy removed). `position_std` is sqrt(12.5) cm.

Spikes while the animal runs (speed above 4 cm/s). "Own term" is today's in-sample
value minus the own-kernel-left-out value. "Left after" is the own-kernel-left-out
value minus the blocked held-out value.

| Recording | `waveform_std` | Mean own share | Spikes with share > 0.5 | Own term, nats/spike | Own term, nats/s | Left after, nats/s | Of that, 2 s window, nats/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RS58, 26-32 mark dims | 24 (default) | 0.34 | 31% | 0.77 | 717 | 57 | 39 |
| RS58 | 16 | 0.76 | 79% | 3.26 | 3,355 | 134 | 84 |
| RS58 | 12 | 0.95 | 97% | 7.17 | 7,831 | 263 | 144 |
| j16, 4 mark dims | 24 (default) | 0.06 | 2.3% | 0.11 | 18 | 28 | 10 |
| j16 | 16 | 0.10 | 5.4% | 0.21 | 40 | 50 | 16 |
| j16 | 12 | 0.15 | 9.8% | 0.36 | 72 | 71 | 20 |

- **RS58, default bandwidth:** the own kernel inflates the Local intensity by 717
  nats per second of running, or 387 nats/s over the whole epoch. For scale, the
  KDE's held-out position information on RS58 was about 169 nats/s over the whole
  epoch (flat-referenced, 5 folds, 2026-10-18). The own kernel is 93% of the gap
  between in-sample and blocked held-out.
- **j16, default bandwidth:** 18 nats/s, 40% of the gap. The typical spike is barely
  affected (median share 0.011); the mean comes from a tail of spikes with few
  neighbours in mark space (99th percentile share 0.73).
- **What leaving out the spike does not remove:** 57 nats/s on RS58 and 28 nats/s on
  j16. Spikes within 2 s account for 39 and 10 of those. The rest includes the
  blocked folds' 20% smaller training set, so it overstates what a wider window could
  recover.

Limits of this measurement:

- It is a float64 NumPy re-implementation of the Local term, not the package's code
  path, on a sample of spikes; nats/s scales each electrode's sample mean by its
  spike rate.
- Encoding used the whole epoch, as the held-out harnesses in `/cumulus/edeno/lowocc`
  do.
- The effect on posteriors and state probabilities was not measured. The own kernel
  is centred on the animal's position, so it should pull the decoded position toward
  the animal and favour Local; by how much is the first real-data validation target
  below.

**Verdict:** worth building. On many-channel marks the in-sample Local term is
dominated by spikes matching themselves, and leaving the spike out removes nearly all
of that. On tetrodes it removes under half of the in-sample optimism, so the
time-window follow-up below is also justified.

### All eight models, simulation (`local_in_sample_optimism.py`)

Output in `local_in_sample_optimism_output.txt`. This is the no-in-sample-excess test
of the requirement, run through the package's own code on the models as they are
today. Three simulated sessions of 233 s (10 laps, 20 cells, about 21,500 spikes),
default settings. Excess is a session's Local log likelihood under its own model minus
its mean under the other two sessions' models, mean +/- standard error over sessions.
The sorted models do not use marks, so they have one column.

| Model | Excess, 4 mark dims (noise SD 10), nats/s | Excess, 16 mark dims (noise SD 15), nats/s |
| --- | --- | --- |
| `clusterless_gmm` | 38.16 +/- 0.15 | 416.9 +/- 2.0 |
| `clusterless_mark_glm` | 2.62 +/- 0.33 | 120.0 +/- 4.5 |
| `clusterless_diffusion` | 0.20 +/- 0.10 | 19.67 +/- 0.24 |
| `clusterless_kde` | 0.19 +/- 0.11 | 18.96 +/- 0.24 |
| `sorted_spikes_mrf` | 0.63 +/- 0.03 | |
| `sorted_spikes_glm` | 0.35 +/- 0.02 | |
| `sorted_spikes_kde` | 0.13 +/- 0.04 | |
| `sorted_spikes_diffusion` | 0.13 +/- 0.04 | |

- **No model passes today.** Every excess is positive.
- **The mixture models have the largest advantage on their own data**, and it grows
  fastest with mark dimension: the GMM's is 4.5 nats per spike at 16 dimensions. In
  total nats it is the same order as their parameter counts (GMM, my arithmetic:
  about 8,900 nats against about 6,700 parameters at 4 dimensions, 97,000 against
  55,000 at 16).
- **The KDE's excess matches its own-kernel term.** At 4 dimensions and noise SD 10
  it is 0.0021 nats per spike here; the kernel-sum calculation above gives 0.001 with
  no added noise and 0.011 at noise SD 20. That is what the Poisson argument predicts,
  and it means stage 1 should bring this test to zero.
- **The sorted models' excess is small here**, under 1 nat/s. Real sorted units have
  isolated out-of-field spikes that this simulation lacks.

Limits: Poisson spiking, three sessions, marks with added Gaussian noise. The excess
counts both what a model gains on its own data and what the other models lose from
estimation error; that sum is what "an advantage on its own data" means.

## Decided by Eric (2026-11-06)

- The exclusion is on by default: if it is conceptually the better design it should
  be the default.
- Every likelihood model gets it, not only the KDE. The aim is a Local state that
  behaves as it should statistically, which makes this a model design issue.

## Stages

Each stage is its own change, judged by the two tests above.

| Stage | Models | How the spike's own contribution is removed | Exact? |
| --- | --- | --- | --- |
| 1 | Clusterless KDE | Drop its kernel term from the sum | Yes |
| 2 | Clusterless diffusion | Drop its term; also linear in the spikes | Yes |
| 3 | Clusterless GMM, clusterless mark GLM | To be chosen by measurement (see below) | Depends |
| 4 | Sorted KDE, sorted diffusion | Drop its term | Yes |
| 5 | Sorted GLM, sorted MRF | One Newton step from the full fit with the spike removed | To second order in the spike's leverage |
| 6 | Model checking | The same exclusion in its own mark model | Yes |
| 7 | All | Spikes close in time (same pass): a time window or blocked folds | Separate design |

The order follows the measured excess: clusterless models first, the mixtures before
the sorted models. Stage 1 is designed in full below. Notes for the others:

- **Stage 2.** The heat kernel replaces the Gaussian position kernel. The fit already
  stores its encoding spikes (`encoding_bin_indices`, `encoding_marks`,
  `encoding_weights`), and its excess equals the KDE's.
- **Stage 3.** The largest excess of any model, and the mark GLM beat the tuned KDE
  on 3 of 4 real recordings, so this stage matters as much as stage 1. These are
  mixtures fitted to a local optimum, with components pruned, so a derivative-based
  correction is not trustworthy. Two candidates:
  - remove the spike's own responsibility-weighted contribution from each
    component's sufficient statistics, holding the other spikes' responsibilities
    fixed;
  - score each time block with a model fitted without it (blocked cross-fitting,
    several fits).

  Choose the cheaper one that passes both tests of the requirement. Measure before
  designing: this document does not know which will pass.
- **Stage 4.** The sorted KDE's place fields are precomputed sums (`place_fields`,
  `marginal_models`; `sorted_spikes_kde.py:505-520`), so the left-out intensity is a
  per-spike correction to a stored field. Subtract only while the own share is below a
  stated bound, and recompute the sum without the spike above it (constraint 1 of
  stage 1). The sorted KDE has no background term, so isolated spikes will meet
  near-zero rates, as they already do on held-out data. Its golden case trains on the
  non-event samples and decodes the first 50 samples
  (`tests/test_golden_regression.py:293-326`), so that file may change and would then
  need approval.
- **Stage 5.** Both fit a penalized Poisson regression by Newton steps and have the
  Hessian at the optimum (`sorted_spikes_glm.py:1000-1011`, `1170-1194`;
  `_penalized_hessian`, `sorted_spikes_mrf.py:272`). Removing one spike with basis
  row `b_j` moves the coefficients by about `-H^-1 b_j`, so
  `log rate_without_j(x) = log rate(x) - b(x)' H^-1 b_j` to first order. The GLM fit
  already sums spikes over position cells, so the correction is one small matrix per
  cell (position bins by position cells), looked up per spike. Neither fit keeps the
  Hessian today. The penalty chosen by REML is held fixed.
- **Stage 6.** The clusterless mark checks rebuild the KDE in NumPy from the stored
  encoding spikes (`KDEElectrode`, `model_checking/_clusterless_models.py:153`), so on
  encoding data they have the same own-spike term.
- **Stage 7.** Leaving out the spike alone removes under half of the in-sample
  optimism on tetrodes (see Real recordings), so the goal is not met there without
  this. For the kernel models it is a window of rows with the matching occupancy
  removed: removing a pass's spikes without its occupancy biases the rate down by
  that pass's share of the occupancy. Encoding spikes are stored in time order, so a
  window is a contiguous run of rows; a builder choosing how to represent left-out
  rows in stage 1 should know this is coming.

## Stage 1 design: clusterless KDE

### What is left out

For each decoded spike, every encoding spike on the same electrode with exactly the
same spike time and exactly the same mark. Normally that is one spike. A spike stored
twice (a data error seen once already, in the j16 sorted units) is otherwise a perfect
self-match, so all copies go.

Matching on the mark as well as the time keeps the feature inert when someone decodes
a different session whose timestamps happen to coincide.

- Compare marks after casting the decoded marks to the dtype the fit stored, so a
  float64 copy of float32 features still matches.
- If most spikes that match on time fail the mark comparison, the marks passed to
  predict are not the ones passed to fit. Warn once with the counts, and leave those
  spikes in. A handful of time-only matches is coincidence and stays silent.

Nothing else changes: not the fit, not the occupancy, not the mark reference, not the
decoding of data outside the encoding period.

### What the fit must keep

The encoding spike times, per electrode, row-aligned with `encoding_positions`,
`encoding_spike_waveform_features` and `encoding_weights` (the spikes inside the
encoding support, `clusterless_kde.py:1787-1794`).

- The fitted model is passed to predict as keyword arguments (`base.py:2186-2192`)
  and the KDE predict function has no `**kwargs`, so the new entry needs a predict
  parameter and a place in the algorithm's permitted encoding-model keys
  (`likelihoods/__init__.py:570-592`).
- The saved-model format must round-trip it.

### Where the term is removed (clusterless KDE)

Both decoding paths score each decoded spike against the electrode's encoding spikes:

- Non-local states: `predict_clusterless_kde_log_likelihood`
  (`clusterless_kde.py:1862`), then `block_estimate_log_joint_mark_intensity` (1468),
  `estimate_log_joint_mark_intensity` (962) and `_estimate_with_enc_chunking` (588),
  with dense, tiled and streaming variants and an exact-repair pass
  (`_repair_flagged_in_groups`, 112).
- Local state: `compute_local_log_likelihood` (2195) and `_add_electrode_local_terms`
  (1381).

The background mixing stays as it is: `logaddexp(log lambda, log background)`. A spike
with no other spike near its mark then falls to the background, exactly as an unseen
mark does in held-out data.

### Fixed constraints, with reasons

1. **Remove the term before the sum, never by subtracting it from the total.** When
   the own share is near 1 (shown above to happen), `total - own` cancels
   catastrophically in float32.
2. **Apply the same removal in the exact-repair pass.** The GEMM path scales its
   factors by peaks that do not depend on the kernels (the mark normalizer and each
   bin's largest weighted position kernel) and recomputes some spikes with a direct
   logsumexp (Notes at `clusterless_kde.py:1045-1056`). Its scaling therefore needs no
   change, but a repaired spike must leave out the same rows.
3. **Identical on every numerical path.** Dense, tiled, streaming, exact repair, Local
   and non-local, chunked and checkpointed prediction (`row_slice`). The existing
   zero-weight rule ("a zero weight drops that spike out of the reduction entirely",
   `clusterless_kde.py:1022-1026`) is the reference behaviour to match.
4. **No new compilations per shape.** Encoding arrays are padded with zero-weight rows
   so electrodes share compiled kernels (`padded_samples`,
   `likelihoods/common/kernel_density.py:232`; used at `clusterless_kde.py:2355-2364`).
   Whatever identifies the left-out rows must live in that padded layout and keep
   shapes static.
5. **On by default; with it off, results are bit-identical to main.** Eric decided
   this on 2026-11-06: if it is conceptually the better design it should be the
   default. It is: the in-sample value is not the likelihood of data the model has
   not seen, the correction is exact, and it does nothing when the decoded data and
   the encoding data do not overlap.
6. **The clusterless golden regression data must not change.** Both clusterless golden
   cases decode data after the encoding period (`tests/test_golden_regression.py:218-233`
   and `372-392`), so no decoded spike is an encoding spike. If they change, the
   implementation is wrong; do not regenerate them.
7. **Models saved before this change decode exactly as before.** They have no stored
   spike times, so nothing can be left out. Decode them unchanged and emit one
   `CompatibilityWarning` that says a refit is needed to get the exclusion. No error,
   no second matching rule.
8. **Do not correct the expected-count term.** Leaving one spike out changes an
   electrode's ground intensity by about one part in the number of its spikes near
   that position, on a term of roughly 0.2 nats per 2 ms bin. Measure the size once
   and record it; do not build machinery for it.
9. **Do not name this document, or "plan" or "phase", in code, tests or docstrings.**

### Switch

Suggested: a fit-time clusterless KDE parameter, stored in the encoding model and read
by predict, the route `use_streaming` already takes. Suggested name
`exclude_own_spikes`, default on. It then also applies inside `estimate_parameters`,
which re-scores the same data on every iteration.

The default must hold whether or not the user passes their own parameter dict. A
passed dict replaces the registry defaults wholesale (`likelihoods/__init__.py:412`),
so the fit function's own default has to be on.

### What changes for users

Any decode that overlaps the encoding data gives different results from the previous
version. That includes the common pattern of fitting on running periods and decoding
the whole session, the tutorials that do so, and Spyglass decoding entries repopulated
with the new version. Decodes of data outside the encoding period, and of models saved
before the change, are unchanged.

- The changelog entry goes under "Changed" and says which decodes differ and how to
  turn it off.
- The clusterless tutorial and the model-choice tutorial explain it where they discuss
  held-out data.
- Regenerated tutorial outputs are part of the change and are read before merging.

## Left to the builder

- How left-out rows are represented and passed to the kernels.
- Whether the Local and non-local paths share one helper.
- Module and function layout, test layout, the parameter name.
- How the saved format stores the new per-electrode array.

## Validation

This section is for stage 1. Every later stage is judged by the two tests of the
requirement, plus the bit-identical, saved-model and compile-count checks below
applied to its own models.

**Hard** (a miss blocks the change):

- Switch off: bit-identical to main on every path.
- Default settings: the clusterless golden regression cases pass untouched
  (constraint 6).
- The stored-artifact compatibility tests decode the old pickles to the same results
  as before, with the one warning from constraint 7.
- Switch on, decoded data disjoint from the encoding data: bit-identical to switch off.
- Switch on, overlapping data: each matched spike's log intensity equals a brute-force
  reference, on every path in constraint 3. Use the tolerance the existing path-parity
  tests use (`tests/likelihoods/test_clusterless_kde_streaming.py`,
  `tests/integration/test_clusterless_kde_parity.py`); do not add a looser one.
  - The reference is the same fitted model with that encoding spike's weight set to
    zero **and** the electrode's mean rate multiplied by
    `(sum_w - w_j) / sum_w`. Zeroing the weight alone renormalizes the density by the
    remaining weight (`_encoding_log_weights`, `clusterless_kde.py:460`) and is off by
    `log(sum_w / (sum_w - w_j))`.
  - The occupancy and the mark reference stay those of the full fit.
- The reference case includes a spike whose own share exceeds 0.99, and a spike stored
  twice.
- The compile-count checks in `tests/likelihoods/test_clusterless_kde_padding.py`
  still pass with the switch on.
- Outputs are finite wherever they are finite with the switch off, and NaN inputs
  propagate as before.
- A fitted model saves, loads and decodes to the same result with the switch on.

**Target** (statistical; report, never loosen after the fact):

- The no-in-sample-excess test of the requirement, on Poisson simulations with
  4-dimensional and with many-dimensional noisy marks.
  - Pass: the excess with default settings is within two standard errors of zero.
  - Also report the excess with the switch off, which shows the test has power.
  - Compare models on the same session, as the evidence script does. Comparing one
    model across sessions has a standard error near 9 nats/s on a 233 s session,
    because a session's spike count noise dominates.
- Real recordings (j16, RS58): report Local log likelihood per second, state
  probabilities while running and decoding error for the encoding period with the
  switch off, with it on, and held out.
- Cost: warm predict time and peak memory with the switch on within 5% of off on the
  existing likelihood benchmark. Measure; if it misses, report the number.

## Non-goals

- Designing stage 7 (same-pass spikes) or choosing the stage 3 method here.
- Changing any bandwidth or tolerance.
- Fixing near-zero KDE tails in the sorted model.

## Anti-goals

- Any change in results with the switch off, or for a model saved before the change.
- A result that depends on which numerical path ran.
- A switch that is on but does nothing without saying so.

## Open decisions for Eric

None. The golden-data approvals named in the stages are requested when they arise.

## Calls made in this document (not Eric's)

- A spike matches on electrode, exact time and exact mark (after casting to the stored
  dtype); a mostly mismatched set warns once.
- The switch is a fit-time algorithm parameter.
- A model without stored spike times decodes as before with one warning.
- The expected-count term is left alone.
- The simulation test compares models fitted to independent sessions on the same
  data, not blocked folds.
- "Every model" is read as including model checking (stage 6).
- The order of the stages (by measured excess), and one Newton step as the method
  for the GLM and MRF.

## Related

- Issue #31: P(Local) at a spike depends on that spike's own fit. With weighted
  encoding refits the own term carries the spike's Local weight, a self-reinforcing
  loop this change removes. It does not address the selection effect among
  neighbouring spikes.
- `evidence/own_kernel_share.py`, `evidence/own_kernel_share_real.py`.
