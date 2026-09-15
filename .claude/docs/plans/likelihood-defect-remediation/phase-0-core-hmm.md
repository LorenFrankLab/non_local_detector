# Phase 0 — Core HMM conditioning and normalization

> **IMPLEMENTED** on `fix/core-hmm-conditioning` (2026-09-14; CPU,
> JAX 0.9.0, NumPy 2.4.1, Python 3.13).
> `_normalize` divides by the exact sum and preserves the zero-input contract.
> `_condition_on_primal` uses reachable-likelihood and log-joint shifts, then
> multiplies by the prior in probability space. Invalid NaN/+inf inputs remain
> visible, including on zero-prior states when all reachable states are impossible.
> An explicit `jax.custom_jvp` computes normalized derivatives only when
> differentiation is requested. It preserves zero-prior and tied-likelihood
> gradients, batches, large common offsets, and mixed second derivatives.
> Ordinary filtering uses one exponential and three reductions per step,
> versus two exponentials and four reductions in the preceding implementation.
> Reference tests: **63 passed / 1 skipped** in default float32 and **64 passed**
> with `JAX_ENABLE_X64=1` and actual dtype assertions. The performance and
> numerical comparisons are recorded below. Full suite: **1329 passed / 4 skipped**;
> golden files and existing tolerances unchanged. Ruff and format checks pass.

**Priority: immediate correctness work, before likelihood-flooring changes and
performance optimization.** This phase adds the missing HMM findings without renumbering
phases 1–8. It is independent of the unresolved C1 likelihood-flooring and C3
time/exposure decisions. The runnable code below reproduces the defects; no
unexecuted implementation patch is prescribed.

## Problem 1 — Positive normalizers lose probability mass

`core._normalize` divides by `sum(u) + 1e-15`. When the sum is positive but much
smaller than that fixed epsilon, the returned probabilities do not sum to one.
This is reachable through the public `core.filter`, not only by calling a helper.

Measured on 2026-09-14 with JAX 0.9.0, NumPy 2.4.1, Python 3.13, CPU/float32:

| Quantity | Value |
|---|---|
| Prior | `[1e-20, 1.0]` (float32) |
| Log-likelihood | `[0.0, -100.0]` |
| Reference posterior | Approximately `[1.0, 3.720076e-24]` |
| Returned posterior | Approximately `[9.999900e-6, 0.0]` |
| Returned probability sum | Approximately `0.00001`, not `1.0` |

The reference uses the actual float32 inputs converted to float64 for independent
calculation. The tiny second component may be negligible at the chosen precision;
the lost total mass is the defect. The log evidence in this example is finite
and approximately correct, so checking evidence alone misses it.

## Problem 2 — An unreachable maximum causes avoidable underflow

`core._condition_on` shifts log-likelihoods by their maximum before multiplying
by the prior. That maximum can belong to a zero-prior state. Reachable states
then underflow even though their log joint probabilities can be normalized
stably. The zero-normalizer fallback misclassifies this as impossible data.

| Quantity | Value |
|---|---|
| Prior | `[0.5, 0.5, 0.0]` |
| Log-likelihood | `[-1000.0, -1001.0, 0.0]` |
| Reference posterior | Approximately `[0.731059, 0.268941, 0.0]` |
| Returned posterior | `[0.5, 0.5, 0.0]` |
| Reference log evidence | Approximately `-1000.379885` |
| Returned log evidence | `-inf`, with a support-mismatch/underflow warning |

Every input log-likelihood is finite and two states have positive prior mass.
This observation is possible. Preserving the prior and emitting a warning does
not make its posterior or evidence correct.

## Runnable reproduction

Run from the repository root. This uses the installed checkout and an independent
SciPy log-sum-exp reference. It prints both failures without modifying source,
snapshots, or fitted models. The snippet was executed against the unmodified
package; it is a reproducer, not a proposed fix.

```bash
uv run python - <<'PY'
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from non_local_detector.core import filter

cases = [
    ("small_normalizer", [1e-20, 1.0], [0.0, -100.0]),
    ("unreachable_maximum", [0.5, 0.5, 0.0], [-1000.0, -1001.0, 0.0]),
]
for name, prior_values, likelihood_values in cases:
    prior = np.asarray(prior_values, dtype=np.float32)
    ll = np.asarray(likelihood_values, dtype=np.float32)
    with np.errstate(divide="ignore"):
        log_joint = np.log(prior.astype(np.float64)) + ll.astype(np.float64)
    expected_evidence = logsumexp(log_joint)
    expected = np.exp(log_joint - expected_evidence)
    (evidence, _), (posterior, _) = filter(
        jnp.asarray(prior), jnp.eye(len(prior)), jnp.asarray(ll[None, :])
    )
    actual = np.asarray(posterior)[0]
    print(name)
    print("  expected posterior:", expected)
    print("  actual posterior:  ", actual, "sum:", float(actual.sum()))
    print("  log evidence expected/actual:", expected_evidence, float(evidence))
    print("  posterior matches:", np.allclose(actual, expected, rtol=1e-5, atol=1e-7))
    print("  evidence finite:", bool(np.isfinite(float(evidence))))
PY
```

Before the fix, both cases print `posterior matches: False`; the second also
prints `evidence finite: False`. Turn these into failing regression tests before
prototyping the fix. The comparison tolerances here distinguish order-one errors
from rounding; they do not authorize changing existing test tolerances.

## Scope and mathematical requirements

1. Normalize positive finite mass without adding a fixed quantity that changes
   its total. Audit `_normalize`'s callers in both filters and both smoothers;
   preserve its returned normalization constant and existing zero-input contract.
2. Evaluate Bayesian conditioning stably from the combined prior and observation
   information. Prototype a log-joint/log-sum-exp calculation against the
   reference above. Do not mask the failure with a small-normalizer threshold,
   likelihood floor, or posterior normalization after the full decoding run.
3. Preserve exact zero prior mass for finite observation likelihoods. Increasing
   the finite likelihood of an unreachable state must not change the posterior
   or evidence of the reachable states.
4. Distinguish genuinely zero support from numerical underflow. For all-`-inf`
   likelihoods, or no overlap between positive prior and finite likelihood,
   preserve the existing operational fallback: return the prior and `-inf`
   evidence. This is a diagnostic convention for an undefined conditional
   distribution, not a claim that Bayes' rule defines that posterior.
5. Preserve NaN visibility, including a NaN at a zero-prior state. NaN input is
   not an impossible observation and must not become a clean fallback.
6. Retain the current API, transition indexing, missing-observation handling,
   and time-bin semantics. Phase 0 neither selects a C1 floor policy nor
   implements the phase-3 or phase-6 fixes.

## Validation required before shipping

The table is an acceptance specification; only the two one-step reproductions
above have been run for this phase. Prototype and execute the remaining checks
before claiming readiness.

| Check | Required evidence |
|---|---|
| Small positive normalizer | Both direct conditioning and `core.filter` reproduce case 1's reference posterior and finite evidence; probability sums are one within existing dtype-appropriate tolerances. |
| Unreachable maximum | Both direct conditioning and `core.filter` reproduce case 2; no false impossible-observation warning. |
| Genuine impossibility | All-`-inf` likelihood and disjoint-support cases retain prior fallback, `-inf` evidence, and the existing diagnostics. |
| NaN propagation | Mixed finite/NaN and `-inf`/NaN inputs remain visibly invalid, including NaN on zero-prior support. |
| Stable invariants | Permuting states permutes the posterior; a representable common likelihood offset leaves the posterior unchanged and shifts log evidence accordingly; changing only an unreachable state's finite likelihood has no effect. |
| Both transition paths | Exercise stationary and covariate-dependent filters with matched expanded transitions and compare to an independent reference. |
| Multi-step filtering and smoothing | For tiny models, independently enumerate state paths in log space and compare filtered/smoothed distributions and total evidence. Check carried predictions as well as returned posteriors. |
| Chunked drivers | Check both transition paths with one and multiple chunks, cached and callback likelihoods, including singleton chunks and missing rows. Callbacks return synthetic rows by global index, isolating this phase from the known spike-binning defect in phase 3. |
| Float32 and float64 | Run default float32 and a separate process with `JAX_ENABLE_X64=1`; assert actual dtypes so silent truncation cannot satisfy the float64 check. |
| Existing behavior | Run core, property, integration, EM/predict-consistency, and golden tests. Evaluate exact transition-only EM where applicable; do not assume the known approximate encoding update guarantees monotonic likelihood. |
| Runtime impact | Measure compilation and steady execution on representative filter/smoother shapes, synchronize JAX results, and record device/dtype. No speedup or performance parity is claimed before measurement. |

Tests must check values against the independent reference, not merely finiteness,
agreement between two callers of the same implementation, or normalized output.
Case 2 is normalized today and is still wrong.

## Existing test baseline and its limits

The follow-up audit ran:

```bash
uv run pytest -q -o addopts='' \
  src/non_local_detector/tests/core/test_core_utilities.py \
  src/non_local_detector/tests/core/test_hmm_algorithms.py \
  src/non_local_detector/tests/core/test_chunked_parity.py \
  src/non_local_detector/tests/integration/test_core_kde_integration.py
```

Result: **116 passed, 26 warnings** on CPU while both reproductions above failed
their mathematical reference. Warnings included float64 requests being truncated
to float32; this baseline does not validate true float64 execution.

`test_normalize_with_very_small_values` currently uses values around `1e-10`, well
above the `1e-15` epsilon, so it does not cover case 1. Existing zero-support
fallback tests remain useful but do not distinguish case 2's possible observation
from a genuinely impossible one. Preserve those tests and add the missing cases.

## Numerical-change review and release note

Capture baselines before changing the core. Defect cases must change; measure
well-conditioned regression differences against existing tolerances. Do not
promise unchanged goldens before running them, and do not widen tolerances to
hide a discrepancy. Follow the existing numerical-validation process and request
approval only if a snapshot/golden update or other existing approval gate is
actually reached.

The release note should describe both triggers: small positive normalization
constants could yield posteriors whose mass was far below one; a highest
likelihood in a zero-prior state could yield an incorrect prior fallback and
`-inf` evidence for possible observations. Keep the phase-3 chunk-boundary and
phase-6 exposure corrections separate so each numerical change is attributable.

## Performance revision — 2026-09-14

Gradient-only work is now in `_condition_on_jvp`. The ordinary forward pass
keeps both stabilizing shifts and exact normalization, while avoiding the
second exponential, the sum of unreachable mass, and `log1p`. The custom rule
uses centered log evidence so large likelihood offsets do not lose derivative
precision. It also provides the identity/zero tangents of the selected prior
fallback, handles nondifferentiable integer likelihood inputs, and saturates
only normalized derivatives that exceed the dtype's range.

The added tests cover simultaneous prior/likelihood JVPs, batched prior
gradients, prior and mixed Hessians at zero prior mass, offsets through `-1e6`,
and both differentiation directions at the impossible-data fallback. The three
new fallback/integer-input regressions failed on the initial custom-JVP
prototype and passed after its operand guards were added. Independent code
review found no remaining actionable issues.

### Measured execution time

CPU/float32, 3000 time steps, fixed dense transitions and precomputed synthetic
likelihoods. Each kernel was compiled separately and warmed up four times;
25 repetitions alternated the implementations in random order and synchronized
every result. The before column uses a source snapshot saved immediately before
this optimization; `main` predates the Phase 0 correctness fixes. Compilation
and host-side diagnostics are excluded. These are kernel timings, so the
end-to-end effect also depends on likelihood computation and smoothing.

| Kernel | `main` (ms) | Before (ms) | After (ms) | Less time vs. before (the intermediate draft, *not* `main`) |
|---|---:|---:|---:|---:|
| Stationary, 32 bins | 2.561 | 3.125 | 2.677 | 14.3% |
| Stationary, 200 bins | 14.530 | 17.570 | 16.424 | 6.5% |
| Stationary, 1000 bins | 185.852 | 217.948 | 195.423 | 10.3% |
| Covariate-dependent, 200 bins | 193.681 | 215.334 | 207.083 | 3.8% |
| Smoother, 200 bins | 24.428 | 25.046 | 24.161 | 3.5% |

Background CPU load was high; interpret the exact percentages with that
limitation. The percentages compare against the intermediate draft, not
`main`; the `main` column shows the true cost of stabilized conditioning
(~+13 % at 200 bins, confirmed by a separate paired run at low load: 42–43 ms
vs 47–49 ms for a 20000-step, 200-state stationary filter). The structural reduction in forward work is also confirmed by the
JAXpr (two exponentials to one; four reductions to three). The optimization
removes part of Phase 0's overhead; stable conditioning still has a cost relative
to `main`.

Use the reproducible [benchmark script](../../../../scripts/benchmark_hmm_conditioning.py):

```bash
uv run python scripts/benchmark_hmm_conditioning.py \
  --time-steps 3000 --repetitions 25 --output /tmp/hmm-benchmark.json
```

An optional `--baseline-core /path/to/core_before.py` adds a saved implementation
to the comparison. The JSON includes compilation times, execution samples,
quartiles, device/dtype, source hashes, and background load.

### Numerical equivalence

The optimized and saved pre-optimization versions returned **bit-identical
posteriors and evidence** for 20,000 deterministic float32 one-step cases,
including zero priors, tiny positive priors, and likelihoods between `-1e4` and
`1e4`. Against the independent float64 reference, maximum posterior error was
`7.68e-8`; evidence error was `4.92e-4` at likelihood magnitudes up to `1e4`
(float32 rounding scale). Maximum probability-mass error was `5.96e-8`, with no
nonfinite outputs. A 1000-step, 200-state model also returned bit-identical
filtered, predicted, and smoothed probabilities and total evidence.

Full-suite validation: **1329 passed / 4 skipped**, in 21m 41s on the busy CPU.
The separate x64 reference run passed all **64** cases. Golden files and existing
tolerances are unchanged; ruff and format checks pass. Changes remain uncommitted
on `fix/core-hmm-conditioning`.
