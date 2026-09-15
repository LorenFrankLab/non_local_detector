# Phase 6c — Detector uniformity guard

> **SETTLED REQUIREMENT — VALIDATOR NEEDS PROTOTYPING.** Ship with 6a, when the
> explicit edge API is introduced. The earlier fixed-relative-tolerance snippet
> is withdrawn because large absolute timestamps can make an intended uniform
> grid fail it. Complete nonuniform direct-likelihood support arrives with 6b/6d.

## Problem and contract

Core HMM transitions advance once per observation and are not scaled by each
row's duration. A 1 s row and a 1 ms row otherwise apply the same transition.
Correct duration scaling of the observation likelihood alone cannot calibrate
the resulting nonuniform-time HMM.

Under [C3](shared-contracts.md#c3--time-vocabulary), detector entry points therefore
require uniform edges. Direct likelihood functions support nonuniform edges
once 6b duration scaling is implemented; they do not apply HMM transitions.
Duration-calibrated transitions remain deferred in [overview.md](overview.md#deferred-with-triggers).
Uniformity does not rescale a fitted per-step transition to a different uniform
interval; document the interval for which the transition configuration is used.

## Falsification and implementation requirements

1. Use 6a's central edge validation before inspecting duration arrays. Cover all
   detector routes reaching the HMM, including predict, parameter estimation,
   Viterbi, and covariate-dependent transitions, before fitting or mutation.
2. Reproduce both genuine irregularity and representational jitter. For example,
   2 ms intervals on a large absolute timestamp have larger subtraction error
   than intervals starting at zero. A bare `allclose(diff, diff[0], rtol=1e-6,
   atol=0)` is not the chosen implementation.
3. Prototype an error bound accounting for timestamp dtype, magnitude/ULPs, and
   nominal interval. Preserve adequate precision while validating. If supplied
   timestamps cannot represent distinct bins, reject them with a clear precision
   error; a permissive tolerance must not accept duplicated or unresolved edges.
4. Verify generated `calculate_time_bins` grids and manual grids at different
   origins, lengths, and frequencies. Reject detectable real irregularity with
   a message explaining the per-step transition restriction.
5. Document the direct-likelihood/detector distinction with the correct rollout
   stage. Do not advertise nonuniform likelihood calibration before 6b/6d.

Select and record numerical validation bounds through prototyping and the
repository's applicable numerical-review process. This plan does not prescribe
an untested replacement tolerance or authorize changes to existing test bounds.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Basic validity | Central validation handles insufficient, repeated, nonfinite, decreasing, and malformed edges before duration comparison. |
| Uniform grids | Generated/manual grids at zero and large absolute origins are accepted when their representable spacing is consistent with the intended interval. |
| Long recordings | Include the 1.8-million-bin, 2 ms timeline without allocating spatial posterior arrays. |
| Genuine nonuniformity | Detectable irregular grids fail at every detector entry before mutation; account for the uncertainty of the timestamp representation. |
| Insufficient precision | Inputs unable to resolve the nominal bins fail clearly rather than silently accepting a different grid. |
| Direct likelihoods | After 6b/6d, per-bin expected counts match known rate × duration for both registries and no-spike; full likelihood values need not be proportional to duration. |

Run affected model, core-integration, and likelihood tests. Valid uniform-grid
outputs should retain parity; invalid inputs gain an explicit error. Measure
any golden impact with the associated 6a migration rather than assuming a new
reference is required by the validator itself. Review bypass paths and the
representability argument for the selected bound.
