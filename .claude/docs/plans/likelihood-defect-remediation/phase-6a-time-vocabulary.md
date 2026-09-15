# Phase 6a — Time vocabulary, coordinates, and encoding-cell migration

> **BLOCKED ON C3b FOR ENCODING CELLS — NEEDS PROTOTYPING.** C3a decode vocabulary
> is settled. The former unexecuted helpers and shape-only parity claims are
> withdrawn. This phase changes row alignment and evaluation coordinates but
> does not perform the Hz conversion assigned to 6b.

## Dependencies and rollout

Use [C3](shared-contracts.md#c3--time-vocabulary) as the contract authority and
[PLAN.md](PLAN.md#execution-order-and-baselines) for release order. Ship 6a with
6c's detector uniformity guard. Then ship Hz conversion and model-unit rejection
as the atomic 6b/6d change. Coordinate tests with Phase 3 global decoding-bin
ownership and Phase 5 fractional encoding-event ownership.

## Recorded problem

Decode helpers currently digitize into `len(time)-1` possible bins while callers
allocate `len(time)` rows. The final row cannot receive a spike:

```text
time = [0..5], spikes = [0.5,1.5,2.5,3.5,4.5]
counts = [1,1,1,1,1,0]
```

The no-spike documentation describes N+1 edges but returns `len(time)` rows.
The sorted GLM also sends position sample centers to the decode count helper;
changing that helper in isolation would produce N-1 counts against an N-row
encoding design matrix.

## Falsification

Pin the pre-migration revision. Reproduce unreachable final output rows and the
GLM shape mismatch caused by treating sample centers as decode edges. Preservation
checks must use matched physical intervals and observation coordinates; merely
comparing the first rows of two differently defined time arrays is insufficient.

## Tasks to prototype

1. **Central edge validation and derived arrays.** Validate one-dimensional,
   finite, strictly increasing edges with at least two entries at all applicable
   public boundaries, before indexing or expensive work. Derive `N = len(edges)-1`,
   centers, and durations once and preserve timestamp precision. Exclude events
   outside the decoding interval before indexing; do not clip them silently into
   endpoint bins. Internal bins are left-closed/right-open
   and the recording's final bin includes the final edge.
2. **Encoding cells have their own contract.** Resolve C3b's acquisition endpoints,
   single-sample input, gaps, and missing-tracking behavior before implementing
   sample cells. A helper clamped to the first/last sample centers undercounts
   exposure under the extrapolated-cell reference and is not accepted. Keep
   encoding counts aligned with the design matrix and exposure rows, preserving
   Phase 5 fractional event weights exactly once. Do not independently invent
   an encoding endpoint policy through the decode helper.
3. **Propagate rows and coordinates explicitly.** Predictors, no-spike outputs,
   missing masks, transition covariates, local-position kernels, non-local
   penalties, result coordinates, and Viterbi coordinates use N rows. Interpolate
   position at centers. Core HMM arrays index observations, not edges; chunk
   drivers must carry the global observation range without an extra time row.
   Preserve Phase 3 event ownership at chunk boundaries.
4. **Generated grids and callers.** Prototype `calculate_time_bins` and migration
   examples with an explicit policy for requested intervals not divisible by the
   nominal bin width; generated detector grids must satisfy 6c. Audit every
   `len(time)`/shape use and classify it as an edge count or observation count.
   Include cached, chunked, stationary/covariate, EM, and Viterbi callers.
5. **No-spike and public documentation.** Fix no-spike input examples, annotations,
   row counts, and coordinate descriptions. Duration-calibrated intensity changes
   belong to 6b; no-spike is already documented in Hz. Describe the 6a interim
   unit convention explicitly, and advertise complete nonuniform likelihood
   support only after 6b/6d.

Helper names and signatures are prototype decisions. Record the old-to-new time
mapping in examples; reusing an old center array as new edges changes the decoded
interval and is not a compatibility transformation.

## Acceptance

| Coverage | Required evidence |
|---|---|
| Edge validation | Empty/single-edge, nonfinite, repeated, decreasing, and malformed arrays fail before indexing; valid single-bin input works. |
| Event assignment | Every in-range event is counted once, including exact boundaries and the final edge; out-of-range events never leak into endpoint bins. |
| Encoding alignment | Counts/weighted sufficient statistics, design rows, and exposure have matching lengths; endpoint/gap expectations follow resolved C3b. |
| All likelihood paths | Both registries and no-spike return N rows; matched positions are evaluated at centers, including local kernels and penalties. |
| All callers | Missing masks, covariates, cached/chunked arrays, EM, Viterbi, and output coordinates agree on N observation rows. |
| Detector guard | 6c ships with the new edge API and validates generated grids and user-supplied edges. |

Use independent reference values and existing applicable tolerances. A final
observation need not contain a spike; it must be capable of receiving one and
have the correct likelihood for its data.

## Numerical-change attribution

This is not purely a shape change: center-based position interpolation and
encoding-cell assignment can alter likelihood values. Removing a dummy terminal
observation can also change earlier smoothed posteriors through the backward
pass. Therefore an unchanged prefix of the old posterior is not an acceptance
requirement.

Separate row/coordinate changes, altered observation likelihoods, and propagated
HMM effects in the numerical analysis. Preserve rate units in this phase so 6b's
conversion remains attributable. Run the full suite, affected snapshot/golden
checks, and a complete caller audit. Measure which references actually change;
request approval only for an observed reference or numerical-contract update.
