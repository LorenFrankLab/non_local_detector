# Phase 6c — Detector uniformity guard

> **SETTLED REQUIREMENT — VALIDATOR NEEDS PROTOTYPING.** Ship with 6a, when the
> explicit edge API is introduced. The earlier fixed-relative-tolerance snippet
> is withdrawn because large absolute timestamps can make an intended uniform
> grid fail it. Complete nonuniform direct-likelihood support arrives with 6b/6d.
>
> Re-verified against `main` at `ee2cc21` (2026-09-22); line references are to
> that revision.

## Problem and contract

Core HMM transitions advance once per observation and are not scaled by each
row's duration (`core.py:464`, `:587`, `:1350`; the drivers use only
`len(time)`). A 1 s row and a 1 ms row otherwise apply the same transition. At
`ee2cc21` no detector route checks uniformity (a 10 ms-then-50 ms grid decodes
without error through `predict`, `estimate_parameters` and
`most_likely_sequence`), and no-spike already assumes it by applying
`median(diff(time))` to every row (`no_spike.py:26-32`, `:114-116`).
Correct duration scaling of the observation likelihood alone cannot calibrate
the resulting nonuniform-time HMM.

Under [C3](shared-contracts.md#c3--time-vocabulary), detector entry points therefore
require uniform edges. Direct likelihood functions support nonuniform edges
once 6b duration scaling is implemented; they do not apply HMM transitions.
Duration-calibrated transitions remain deferred in [overview.md](overview.md#deferred-with-triggers).
Uniformity does not rescale a fitted per-step transition to a different uniform
interval; document the interval for which the transition configuration is used.
Name both per-step transition sources: `RandomWalk.movement_var`
(`continuous_state_transitions.py:136-160`) and `EmpiricalMovement`, estimated
per position sample with `speedup` applied via `matrix_power` (`:438-540`).

## Falsification and implementation requirements

1. Use 6a's central edge validation before inspecting duration arrays. Cover all
   detector routes reaching the HMM, including predict, parameter estimation,
   Viterbi, and covariate-dependent transitions, before fitting or mutation.
   Baseline: `predict` and `most_likely_sequence` do not validate `time`
   (decreasing or NaN `time` decodes without error);
   `_DetectorBase.estimate_parameters` checks only finiteness and non-strict
   monotonicity (`base.py:1976-1978`), and the public wrappers call `self.fit`
   first (`:3731`, `:4678`), so a rejected grid already leaves the detector
   refit with `_encoding_model_data` attached. Put the guard in both public
   `estimate_parameters` wrappers before `self.fit` (the placement Phase 5 uses
   for damping), and in `predict`/`most_likely_sequence` before
   covariate-transition prediction. `predict` also sets
   `_degenerate_timesteps_` (`base.py:1825`). State whether the public
   `core.py` functions (`filter`, `chunked_filter_smoother*`,
   `most_likely_sequence*`) are covered or explicitly out of scope.
2. Reproduce both genuine irregularity and representational jitter. For example,
   2 ms intervals on a large absolute timestamp have larger subtraction error
   than intervals starting at zero. A bare `allclose(diff, diff[0], rtol=1e-6,
   atol=0)` is not the chosen implementation.
3. Prototype an error bound accounting for timestamp dtype, magnitude/ULPs, and
   nominal interval. Preserve adequate precision while validating. If supplied
   timestamps cannot represent distinct bins, reject them with a clear precision
   error; a permissive tolerance must not accept duplicated or unresolved edges.
   Measured (1.8M bins, 2 ms): float64 maximum relative spacing error is 2e-10
   at origin 0 and 7.2e-5 at a 1.7e9 Unix-epoch origin, where
   `allclose(rtol=1e-6)` fails for any dt below about 0.24 s. float32 keeps only
   29 unique edges at 1.7e9 and has ~10% spacing error even at origin 0.
   Decide whether the nominal interval is inferred from the edges or
   cross-checked against `sampling_frequency` (currently unused for decoding).
4. `calculate_time_bins` (`base.py:2406-2423`, no internal callers) returns
   `t0 + arange(ceil(dur*fs))/fs`, excludes the end point, and gives an
   origin-dependent count (150 vs 151 bins for the same 0.3 s at t0=0 vs 1.7e9);
   6a must redefine it as an edge generator this guard accepts. Verify its grids
   and manual grids at different origins, lengths, and frequencies. Reject detectable real irregularity with
   a message explaining the per-step transition restriction.
5. Document the direct-likelihood/detector distinction with the correct rollout
   stage. Do not advertise nonuniform likelihood calibration before 6b/6d.

## Downstream impact

Spyglass decodes with `time=position_info.index` — raw camera timestamps
(irregular) when not upsampled, or `linspace(start, end, ceil(dur*rate)+1)` grids
(uniform but not exactly `1/rate`) — and `estimate_parameters` receives
concatenated epochs with gaps, all at Unix-epoch origins (from a local spyglass
checkout, not verified against upstream). The guard will reject some of these.
Record the rejections, the remedy (per-interval uniform grids), and coordinate
with the spyglass pin.

Select and record numerical validation bounds through prototyping and the
repository's applicable numerical-review process. This plan does not prescribe
an untested replacement tolerance or authorize changes to existing test bounds.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Basic validity | Central validation handles insufficient, repeated, nonfinite, decreasing, and malformed edges before duration comparison. |
| Uniform grids | Generated/manual grids at zero and large absolute (Unix-epoch) origins are accepted when their representable spacing is consistent with the intended interval; existing goldens all start at t0=0, so add large-origin fixtures. |
| Long recordings | Include the 1.8-million-bin, 2 ms timeline without allocating spatial posterior arrays. |
| Genuine nonuniformity | Detectable irregular grids fail at every detector entry before mutation; account for the uncertainty of the timestamp representation. |
| Insufficient precision | Inputs unable to resolve the nominal bins fail clearly rather than silently accepting a different grid. |
| Direct likelihoods | After 6b/6d, per-bin expected counts match known rate × duration for both registries and no-spike; full likelihood values need not be proportional to duration. |

Run affected model, core-integration, and likelihood tests. Valid uniform-grid
outputs should retain parity; invalid inputs gain an explicit error. Measure
any golden impact with the associated 6a migration rather than assuming a new
reference is required by the validator itself. Review bypass paths and the
representability argument for the selected bound.
