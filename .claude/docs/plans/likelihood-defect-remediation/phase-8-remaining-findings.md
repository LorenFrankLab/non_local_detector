# Phase 8 — Remaining audit findings

> **NEEDS PROTOTYPING.** These are separate correctness/contract tasks with the
> recorded evidence below. Unexecuted implementation snippets are removed. The
> density/exposure work uses the corrected Phase 6 units and applicable C1
> decisions; sorted-index contract work can be prototyped independently.

Phase numbering does not require delaying these fixes behind Phase 7. Relevant
Phase 8 corrections must be included in the baseline used to claim production
support for affected Phase 7 backends/configurations. Separate changes when their
numerical effects need independent attribution.

## 1. Distinguish per-bin mass from density

`diffusion.to_density` receives smoothed count fields and currently divides each
column by `bin_sizes @ smoothed`. That produces an integral-one field but does
not implement conversion of per-bin mass `m_i` to density
`(m_i / volume_i) / sum(m)` on unequal-volume bins.

Establish the caller's quantity and units before changing it: the diffusion
operator, occupancy, and spike-marginal fields must agree on whether values are
mass or density. Phase 6b changes exposure units but does not by itself settle
this spatial-volume conversion.

The original audit compared downstream spike-marginal/occupancy ratios:

```text
current:  [0.4035 0.9282 1.8107 3.4473 0.4218 0.0468]
proposed: [0.7490 1.7229 3.3608 6.3987 0.7828 0.0868]
ratio shapes proportional: True; global scale factor: 0.5387
equal-volume results identical: True
```

The spatial volume factors cancel between numerator and denominator, leaving a
per-neuron scale difference in this fixture. Individual density arrays on
unequal volumes do not differ merely by a column constant. Downstream posteriors
also need not change by a constant: rates enter state-dependent expected-count
terms and the HMM normalizes and propagates the resulting evidence.

### Prototype and acceptance

- Compare against an independent count/volume reference with unequal volumes;
  integral-one normalization alone is insufficient because both formulas can
  satisfy it. Check nonnegative/zero-mass columns and valid positive volumes.
- Verify occupancy and spike-marginal caller units and the induced rate ratio.
  Preserve equal-volume behavior at the existing `rtol=1e-6` target.
- Attribute variable-volume field/rate changes before examining likelihoods and
  posteriors. Do not require a per-neuron-constant posterior or golden diff.

## 2. Unoccupied disconnected MRF components

The original audit found only a total-exposure check. A disconnected component
with no exposure can retain a global warm-start rate through its unpenalized
null mode even when another component is occupied.

Prototype exposure accounting per connected component using the original
recording's corrected exposure units. An unoccupied component is unidentified;
apply the backend's declared zero-exposure behavior consistently with applicable
C1 decisions. If that policy remains the EPS fallback, test EPS explicitly;
this file does not independently choose a new floor. Emit a diagnostic identifying
the unoccupied component count.

Test two-component and connected controls, all-zero exposure, supported fitted
outputs/caches, and prediction. A connected control and unaffected occupied
components should retain the reference behavior; investigate any change rather
than masking it with regularization or a relaxed tolerance.

## 3. Sorted-index assertions on unvalidated spike times

The historical audit found nine likelihood `segment_sum` sites specifying
`indices_are_sorted=True` without checking the corresponding spike ordering.
CPU/XLA examples produced matching results with/without the assertion; GPU
behavior was not tested. This is a contract/portability issue, not evidence of an
observed GPU failure.

**Addressed in the Phase 3 JAX follow-up:** all nine consumers now enable the
hint only when `select_spikes_in_rows` returns a slice, which establishes sorted
IDs (or an empty selection). The mask fallback passes `False` and keeps original
spike/feature order. Tests check the actual compiler promise in both prediction
paths, including zero-rate electrodes, and CPU likelihoods remain bit-identical.
See [Phase 3's audit](phase-3-chunk-boundary.md#jax-audit-follow-up).

The remaining performance task is to establish ordering once rather than scan
every recording-length spike train on every chunk. Re-inventory consumers when
implementing that prepared-input contract and profile it. Preserve accepted
unsorted inputs and paired spike/feature/weight alignment. Rejection would be a
separate public behavior choice; GPU validation remains outstanding.

### Acceptance

- Sorted, unsorted, and duplicate-time inputs are handled according to the chosen
  public contract and compared with an independent reduction reference.
- If inputs are reordered, aligned features/weights are reordered identically;
  if rejected, errors identify the affected unit before evaluation.
- Exercise actual reduction consumers in both core transition configurations,
  not only the bin-index helper. Test GPU behavior when available and report
  untested devices explicitly. A CPU test need not fail before the fix to
  establish that an unproved compiler assertion was a contract violation.

## Validation and release

Run affected likelihood/model, integration, and golden tests. Re-inventory the
current code before applying historical line references or site counts. Preserve
existing test tolerances and use independent references for changed primitives.

Golden changes are possible for variable-volume or disconnected fixtures;
unchanged regular-grid/sorted-input controls remain useful. Analyze actual
likelihood and posterior consequences rather than guaranteeing that all goldens
stay fixed or that their differences are constant. Follow the existing approval
process for observed reference changes.

Review mass/density semantics and caller units, component-exposure handling, and
the chosen sorted-index strategy. Phase 7 must preserve these corrected numerical
contracts while changing storage or transition representation.
