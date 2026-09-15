# Phase 2 — Clusterless GMM log intensity and ground process in log space

> **IMPLEMENTED (narrowed scope)** on `fix/gmm-log-intensity-ordering`
> (2026-09-14). Two numerical defects in `likelihoods/clusterless_gmm.py` are
> fixed as pure arithmetic corrections; no floor is added or removed elsewhere,
> and no package-wide degeneracy policy is chosen. That policy — a background
> firing model making impossible and very-unlikely spikes commensurate, uniform
> handling across backends, and No-Spike behaviour — is **deferred to a
> separate modelling proposal** (see [C1](shared-contracts.md#c1--degeneracy-policy-for-log-intensities)).

## Problem and recorded evidence

The clusterless GMM log intensity is `log(rate) + log p(pos, mark) − log p(pos)`.
The non-local, bin-tiled non-local, and local paths clamped the joint numerator
at `LOG_EPS ≈ −34.5` before the subtraction:

```text
log_rate = 0, log_joint = -60, log_occupancy = -8
raw log ratio:              -52.00
numerator-clamped result:   -26.54
```

The ~25.5-log-unit error inflates the intensity by ~1e11. Worse, at bins whose
occupancy log density is very negative (an overfit 2-D fixture reached −504),
the clamped numerator made every observed spike score **+150 to +2341** — the
spike was reported as overwhelmingly *more* likely at the bins the fitted
occupancy model considered least plausible. The existing fixture
(`test_kde_gmm_comparison`) asserted only finiteness and never noticed.

The fit-time ground process `rate · p_gpi(x) / p_occ(x)` was formed from the two
exponentiated densities, so two densities that each underflow float32 gave
`0 / 0`, which an `occupancy > 0` guard replaced with `rate · EPS`; the correct
ratio for `log p_gpi = −800`, `log p_occ = −790`, `rate = 2` is `9.08e-5`. The
local path already used the log difference but multiplied by the rate outside
the exponential, so `1e-15 · exp(95)` overflowed where `exp(log 1e-15 + 95)` is
representable.

## What shipped

1. **Spike intensity.** The three `jnp.clip(joint_logp, min=LOG_EPS)` calls
   (untiled, tiled, local) are removed; the contribution is the raw
   `log(rate) + (log_joint − log_occupancy)`, difference first — review found
   that `(log_rate + joint) − occ` rounds the rate term away once the log
   densities reach ~`1e10` in float32 (event term `0` instead of `−34.54` for
   `rate = 1e-15`). Both operands come from
   `score_samples` (log-space mixture evaluation), so the arithmetic preserves
   finite log densities; no fallback is introduced for unexpected non-finite
   scores, which stay visible.
2. **Ground process.** One helper, `_ground_process_intensity`, computes
   `exp(log(rate) + (gpi_logp − log_occupancy))` — difference first so nearly
   equal large log densities subtract exactly — and is used by both the
   fit-time bin intensities and the local expected counts. The `occupancy > 0`
   guard is gone: float32 underflow is precision loss, not absent support, and
   an unvisited bin is not a true `0/0` under a nonsingular GMM (both
   densities stay positive; their ratio is merely poorly constrained). NaN
   propagates. When the complete intensity — rate included, aggregated over
   electrodes — exceeds the working dtype's range it overflows to `inf` and
   that bin's log likelihood is `−inf`, replacing the previous `EPS`
   substitution. This affects poorly sampled regions and sharply fitted
   models; it is not an exposure-based support criterion, and zero posterior
   mass there is conditional on another candidate keeping a finite likelihood
   (an all-`−inf` row follows phase 0's fallback).
3. **Preserved.** The zero-rate electrode fallback (`LOG_EPS` per observed
   decode spike), the `EPS` floor on the fitted mean rate, and the single `EPS`
   clip on the summed ground-process intensity.

**Known open policy question (not resolved here):** an empty electrode's spike
scores `LOG_EPS` while a fitted electrode's tail spike can score far below it.
Do not remove the zero-rate fallback until a background model replaces it.

## Validation

`tests/likelihoods/test_clusterless_gmm_log_intensity.py` — 16 tests, all
against closed-form float64 references on hand-built single-Gaussian models
(or the fitted models' own log densities), never another call into the code
under test; event terms are isolated by subtracting a matched no-spike
baseline from the same public path:

| Case | `main` | branch |
|---|---|---|
| Tail spike event term, untiled / tiled / local (`−59…−69` vs clamped `−31`) | fail | pass |
| Deep-tail rate term (`log p ≈ −1e10`, `rate = 1e-15`), accumulator / untiled / tiled / local | large positive (clamp)† | `−34.54` |
| Local expected counts, `1e-15 · exp(95)` | `−inf` | `−2.7e26` |
| Helper, `(2, −800, −790)` and `(2, −120, −60)` | n/a | `9.08e-5`, `2e^-60` |
| Fit far bins, identical GPI/occupancy models (both densities underflow) | `EPS` | `rate` |
| Local no-spike term at a 150-nat-far position, identical models | `−rate`* | `−rate` |
| NaN occupancy / NaN decode position | NaN* | NaN |
| Fit vs local expected counts at bin centres | disagree at underflow bins | agree |
| Accumulator scatter is the raw sum | pass | pass |

\* `main`'s local path had no guard, so these two already held there; an
intermediate draft of this branch broke both and was corrected in review.
† On `main` the numerator clamp dominates through the public paths; the
ulp-quantized `0`, `−64`, `−32` values were produced by the intermediate
unclamped draft with the old `(log_rate + joint) − occ` ordering, which is
what the ordering swap corrected.
Passes identically with `JAX_ENABLE_X64=1` (dtype-aware preconditions).

Regression surface (all of `tests/likelihoods/`, GMM optimization and
agreement files, `test_golden_regression`, `tests/integration/`): recorded in
the CHANGELOG entry and the commit message. Existing tests that asserted the
defect were changed: `test_likelihood_edge_cases::test_clusterless_gmm_extreme_waveform`
required a 100-σ outlier to score above `−50` (now `≈ −6.3e5`, its raw log
density); the three GMM non-local finiteness checks in
`test_kde_gmm_comparison` relied on the underflow guard masking an overfit
model's overflow (now `−inf` at those bins; the assertions became "no NaN");
`test_kde_gmm_numerical_comparison` fitted 16/16/32 components to ~100 spikes
per electrode (on `main` that gave positive log likelihoods up to `+5260` and
a KDE↔GMM Spearman of 0.08) — the ordinary comparisons now use 4/4/8 (Spearman
0.37 on `main`, 0.36 here) and the 16/16/32 fit is kept as a stress
regression whose overflow locations are checked against a float64 aggregate
of the fitted models' own log densities, with finite bins matching that
reference and every time bin keeping a finite candidate; and
`test_clusterless_gmm_optimization::test_gmm_jax_array_inputs` (a thin
trajectory on a 2-D grid, so off-trajectory bins overflow) now feeds the
JAX and NumPy paths identical float32 values and requires bit-identical
predictions including the `−inf` mask. The stress test derives its overflow
cutoff from the working dtype (one bin exceeds even float64). Changes are
attributed to numerator-clamp removal and guard removal respectively; no
existing tolerance was relaxed and no golden changed.

## Archived — superseded by the narrowed decision

The original phase text required, before implementation, a package-wide floor
inventory (backend × local/non-local × spike/ground-process), a choice among the
C1 options, resolution of zero-numerator-over-zero-occupancy and
all-degenerate-aggregate semantics, and a numerical-change analysis for
possible golden updates. None of that gates this phase any more: the shipped
change adds no floor and no aggregate policy, no GMM golden or snapshot fixture
exists, and the KDE goldens are unaffected. Those questions move to the
background-model proposal recorded under C1. The withdrawn "floor only `-inf`"
rule must not be restored (it is non-monotonic: an impossible observation
would score `−34.54` while a `1e-44` one scores `−101.31`).
