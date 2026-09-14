# Phase 0 — Core HMM conditioning and normalization

**Priority: immediate correctness work, before likelihood-flooring changes and
performance optimization. Status: both defects reproduced; fix not prototyped
or implemented.** This phase adds the missing HMM findings without renumbering
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
