# Phase 8 — Remaining audit findings

> **NEEDS PROTOTYPING.** These are separate correctness/contract tasks with the
> recorded evidence below. Unexecuted implementation snippets are removed.
> Sections 1–2 do not depend on Phase 6 time units: the volume bias in §1 is
> independent of time units (a common time factor cancels in the rate ratio),
> and zero exposure in §2 is unit-free; they use applicable C1 decisions only for
> the zero-exposure floor. The sorted-index work (§3) is complete in Phase 3
> except for GPU validation.
>
> §1 and §2 were reproduced on `main` at `ee2cc21` (2026-09-22); line
> references are to that revision.

Phase numbering does not require delaying these fixes behind Phase 7. Relevant
Phase 8 corrections must be included in the baseline used to claim production
support for affected Phase 7 backends/configurations. Separate changes when their
numerical effects need independent attribution.

## 1. Distinguish per-bin mass from density

`diffusion.to_density` receives smoothed count fields and currently divides each
column by `bin_sizes @ smoothed`. That produces an integral-one field but does
not implement conversion of per-bin mass `m_i` to density
`(m_i / volume_i) / sum(m)` on unequal-volume bins (`diffusion.py:549-571`;
its only caller is `sorted_spikes_diffusion.py:511-523`).

`pixellate_interior_fields` returns weighted counts and `diffuse` is
mass-conserving (`diffusion.py:433`), so the smoothed columns are per-bin
masses. `clusterless_diffusion` already uses the mass convention
`H·O/(Σw·dV)` (`clusterless_diffusion.py:33-38, 360, 433`); the sorted backend
is the outlier. Unequal volumes occur on track graphs whose edge lengths are not
multiples of `place_bin_size` — e.g. the existing fixture
`make_linear_track_env` (widths 1.0 and 0.917) gives fitted rates 1.042× the
mass-convention rate. N-D grids use products of per-axis widths, usually
uniform. The Laplacian is combinatorial, so on unequal volumes its steady state
equalizes mass per bin rather than density; that modeling question is out of
scope here.

Establish the caller's quantity and units before changing it: the diffusion
operator, occupancy, and spike-marginal fields must agree on whether values are
mass or density. Phase 6b changes exposure units but does not by itself settle
this spatial-volume conversion.

The original audit compared downstream spike-marginal/occupancy ratios:

```text
current:  [0.4035 0.9282 1.8107 3.4473 0.4218 0.0468]
proposed: [0.7490 1.7229 3.3608 6.3987 0.7828 0.0868]
ratio shapes proportional: True; per-neuron scale factor for this neuron: 0.5387
equal-volume results identical: True
```

The spatial volume factors cancel between numerator and denominator, leaving a
per-neuron scale difference, `Σ(v·o)·Σs / (Σ(v·s)·Σo)` in general. Individual density arrays on
unequal volumes do not differ merely by a column constant. Downstream posteriors
also need not change by a constant: rates enter state-dependent expected-count
terms and the HMM normalizes and propagates the resulting evidence.

### Prototype and acceptance

- Compare against an independent count/volume reference with unequal volumes;
  integral-one normalization alone is insufficient because both formulas can
  satisfy it. Check nonnegative/zero-mass columns and valid positive volumes.
- Verify occupancy and spike-marginal caller units and the induced rate ratio.
  With equal volumes the two formulas are algebraically identical; require
  agreement to a few ulp. Add a units-free check: the occupancy-weighted mean of
  the fitted rate equals `mean_rate` (the mass convention satisfies this
  exactly; the current code does not).
- Update `test_to_density_integrates_to_one_uniform_and_nonuniform` and
  `test_to_density_nonuniform_matches_oracle` (`test_diffusion.py:615, 638`);
  the latter's "oracle" reuses the current normalization and is not independent.
- Attribute variable-volume field/rate changes before examining likelihoods and
  posteriors. Do not require a per-neuron-constant posterior or golden diff.

## 2. Unoccupied disconnected MRF components

The original audit found only a total-exposure check. A disconnected component
with no exposure can retain a global warm-start rate through its unpenalized
null mode even when another component is occupied.

Reproduced (`sorted_spikes_mrf.py:267-270` warm start, `:824` total-exposure
check): `make_two_room_env` with zero right-room weights gives a constant
right-room rate of 0.0948 = `counts.sum()/occupancy.sum()`, with no warning;
`sorted_spikes_diffusion` gives EPS on the same input. One exposure sample with
zero spikes gives 1.4e-4 instead, so define "unoccupied" explicitly (exactly
zero vs. a threshold). Within-component zero-occupancy patches are intentionally
interpolated and must not change (`test_sorted_spikes_mrf.py:251`); keep the
controls `test_mrf_penalty_does_not_smooth_across_a_wall`,
`test_zero_effective_weight_returns_eps_place_fields` and
`test_default_rank_covers_disconnected_components`. The null coefficient is
stored in `mrf_coefficients`, which the fitted-output tests must cover.

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

**Addressed in the Phase 3 JAX follow-up:** the hint is set in one place,
`sum_spikes_into_rows` (`common.py:445-466`, 9 call sites including 3
spike-count sites), plus two direct `segment_sum` calls in
`clusterless_diffusion.py:691, 811`; all read `SpikeSelection.indices_are_sorted`, established by the ordering check
(or an early empty selection), independently of the indexer's representation.
The unsorted fallback passes `False` and keeps original spike/feature order.
Tests check the actual compiler promise in both prediction
paths, including zero-rate electrodes, and CPU likelihoods remain bit-identical.
See [Phase 3's audit](phase-3-chunk-boundary.md#jax-audit-follow-up).

**Also implemented in Phase 3's ordering follow-up:** detector predictions share
one host conversion and ordering check per distinct spike-time object across
states/chunks. Preparation is local to each prediction, so repeated calls recheck
even arrays modified in place. Direct backend calls remain independently checked.
Unsorted inputs remain accepted with their original alignment and per-chunk masks.
See [ordering preparation](phase-3-chunk-boundary.md#ordering-preparation-follow-up)
for scope and validation. GPU validation remains outstanding; rejection or automatic
sorting would be a separate public behavior choice.

### Status and remaining acceptance

Done except GPU. Phase 3 chose the contract: unsorted input is accepted,
spike/feature alignment is preserved, and the hint is `False` for it. Covered by
`test_shuffled_spike_order_row_slice_parity` and
`test_selected_features_stay_paired_with_their_spike`
(`test_row_slice_edge_cases.py`), with duplicate times via `SPIKE_KINDS`.
Remaining: run these on a GPU and report untested devices explicitly. Rejecting
or auto-sorting input would be a new public-behavior proposal, not part of this
phase.

## Validation and release

Run affected likelihood/model, integration, and golden tests. Re-inventory the
current code before applying historical line references or site counts. Preserve
existing test tolerances and use independent references for changed primitives.

No golden or snapshot fixture exercises `sorted_spikes_diffusion` or
`sorted_spikes_mrf` (`test_golden_regression.py` uses KDE only), so goldens
should stay unchanged; any golden diff indicates an unintended change. Expected
changes appear in the diffusion and MRF unit tests; analyze actual likelihood
and posterior consequences there rather than assuming constant differences. Follow the existing approval
process for observed reference changes.

Review mass/density semantics and caller units, component-exposure handling, and
the chosen sorted-index strategy. Phase 7 must preserve these corrected numerical
contracts while changing storage or transition representation.
