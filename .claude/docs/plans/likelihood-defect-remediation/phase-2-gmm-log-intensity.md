# Phase 2 — Consistent log-intensity and ground-process policy

> **BLOCKED ON C1 — NEEDS PROTOTYPING.** The ratio defect is reproduced below.
> The former floor-only-`-inf` policy and its implementation snippets are
> withdrawn. This file does not choose among the options in
> [C1](shared-contracts.md#c1--degeneracy-policy-for-log-intensities).

## Problem and recorded evidence

The clusterless GMM log intensity is `log(rate) + log p(pos, mark) − log p(pos)`.
The non-local, bin-tiled non-local, and local paths clamp the joint numerator
before subtraction. The original audit located these in `clusterless_gmm.py`
near lines 722, 754, and 876; re-inventory the named computations before editing.

```text
log_rate = 0, log_joint = -60, log_occupancy = -8
raw log ratio:              -52.00
numerator-clamped result:   -26.54
```

The approximately 25.5-log-unit error inflates the intensity by about 1e11.
Ground-process paths also exponentiate separately small densities before dividing,
which can lose a finite ratio through underflow.

A finished-intensity floor would intentionally change the raw result from -52
according to the selected bound. That is a policy choice, distinct from the
numerator-ordering defect. The earlier requirement to preserve every finite
value while flooring only exact `-inf` was non-monotonic and must not be restored.

## Decisions required before implementation

Complete C1's backend × local/non-local × spike/ground-process floor inventory.
Record whether the chosen scope includes sorted likelihoods, probability-space
KDE clamps, and log-KDE caller clamps. Resolve zero numerator with zero occupancy,
all-degenerate ground-process aggregates, and genuinely invalid NaN inputs.

Distinguish four stages explicitly: raw log ratio, per-electrode ground-process
term, aggregate ground-process intensity, and finished likelihood. If an
aggregate floor is selected, applying it per electrode before summation gives
`n_electrodes × EPS` instead of one floor and is not equivalent. An all-empty
population also needs a declared result.

Phase 1's existing EPS fallback remains its completed baseline. Any change to it
caused by expanding C1 to sorted backends belongs to this phase and must be
identified in the numerical-change analysis. Phase 0 conditioning is preserved.

## Falsification and prototype requirements

1. Freeze the relevant pre-fix revision and reproduce numerator clipping and
   separately-underflowing ground-process ratios through the actual paths.
2. Define expected outputs from the selected C1 policy. Test the raw arithmetic
   separately from any finished floor; a -52 raw ratio alone does not determine
   the final policy-dependent intensity.
3. Test monotonicity through tiny positive and exact-zero mass, zero occupancy,
   zero-over-zero, empty populations, and raw NaN inputs. Do not infer support
   from a NaN created by subtracting two `-inf` values; inspect operands.
4. Prototype shared arithmetic/policy boundaries across the audited sites.
   Existing helpers, including `_log_joint_from_log_marginal`, are subjects of
   the audit rather than automatically correct references.
5. Compare local/non-local and tiled/untiled evaluation on matched positions,
   marks, rates, and occupancy. Preserve shapes and apply ground-process
   aggregation in log space where needed for finite ratios.

No helper signature or executable implementation is prescribed until these
choices have been exercised against the real backends.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Ratio arithmetic | Independent log-space reference for the recorded tail and underflow cases, before applying the selected finished policy. |
| Policy consistency | Finite tails, exact zeros, unsupported ratios, and aggregate behavior match the documented C1 decision at every in-scope site. |
| Diagnostics | Genuine invalid inputs remain visible; genuine impossibility follows the chosen likelihood policy and Phase 0 core contract. |
| Aggregation | Multiple degenerate electrodes do not accidentally multiply an aggregate floor; empty-population behavior is explicit. |
| Integration | Local/non-local, tiled/untiled, stationary/covariate, and affected detector paths agree with their independent references at existing applicable tolerances. |

Run affected backend, cross-model, integration, and golden regressions. Preserve
existing numerical tolerances. Pin before/after inputs and record primitive
changes from ratio ordering, zero-mass handling, floor scope, and aggregation.

## Numerical-change review

Golden changes are possible, not guaranteed for every fixture. Attribute changes
first at the likelihood/ground-process layer. HMM normalization and temporal
propagation can change posterior bins whose own likelihood did not change; a
requirement that posterior differences stay only in numerator-clamped bins is
incorrect. Unaffected primitive calculations must retain parity.

Describe the selected policy and its scientific effects in the release note.
Provide the repository's numerical-change analysis before requesting any actual
snapshot/golden or numerical-bound change. Review the completed floor inventory,
operand handling, aggregation order, and shared-helper consumers independently.
