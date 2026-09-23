# Phase 6a — Time vocabulary, coordinates, and encoding-cell migration

> **BLOCKED ON C3b FOR ENCODING CELLS — NEEDS PROTOTYPING.** C3a decode vocabulary
> is settled. The former unexecuted helpers and shape-only parity claims are
> withdrawn. This phase changes row alignment and evaluation coordinates but
> does not perform the Hz conversion assigned to 6b.
>
> Claims were re-verified against `main` at `ee2cc21` (2026-09-22); line
> references are to that revision. If Phase 5 gives the GLM its own encoding
> row-assignment helper that preserves today's assignment exactly, the C3a
> decode migration no longer shares a helper with encoding and could ship
> independently of C3b; only encoding-cell changes would remain blocked.

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

The ownership rule is `np.digitize(t, time[1:-1])` (`common.py:264`,
`:359-362`, `:888`), which Phase 3 kept when it moved decoding to global spike
binning. Phase 3 (`1f2b6b4`) also aligned the no-spike docs with this
convention (one row per timestamp, terminal row never owns a spike;
`no_spike.py:52-54,95-97`). `get_spike_time_bin_ind` (`common.py:251-264`) is
unused in production but is the expected-row oracle in several tests, and its
docstring still calls `time` "edges".

The sorted GLM fit passes `position_time` to the decode count helper
(`sorted_spikes_glm.py:316,356`): its last sample row never receives a spike,
and spikes outside `[t[0], t[N-1]]` are dropped. Changing the shared helper in
place would yield N-1 counts against the N-row design matrix. Phase 5 rewrites
this counting (per-neuron weighted counts), so re-verify after Phase 5.

No public boundary validates decode edges: only `estimate_parameters` checks
`time` (`base.py:1976-1978`, non-strict, so repeated timestamps pass);
`predict`, `most_likely_sequence`, the direct `predict_*` functions and
`get_spikecount_per_time_bin` do not. Repeated edges silently produce an empty
zero-width row, and decreasing edges silently misbin.

## Falsification

Pin the pre-migration revision. Reproduce unreachable final output rows,
silent acceptance of repeated or decreasing timestamps, and the GLM's empty
terminal sample row. The N-1 vs N GLM mismatch is not a current failure;
demonstrate it by patching the helper. Preservation
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
   an encoding endpoint policy through the decode helper. Encoding spike-range
   filters currently clamp to `[position_time[0], position_time[-1]]` at six
   sites (`sorted_spikes_kde.py:185`, `sorted_spikes_diffusion.py:317`,
   `clusterless_kde.py:281`, `clusterless_kde_log.py:1493`,
   `clusterless_gmm.py:450`, `clusterless_diffusion.py:390`) while exposure uses
   all N samples; they must follow the resolved C3b endpoint policy.
3. **Propagate rows and coordinates explicitly.** Predictors, no-spike outputs,
   missing masks, transition covariates, local-position kernels, non-local
   penalties, result coordinates, and Viterbi coordinates use N rows. Interpolate
   position at centers. Core HMM arrays index observations, not edges; chunk
   drivers must carry the global observation range without an extra time row.
   This includes the `n_chunks > n_time` checks (`core.py:855,1485`), the
   legacy callback branch's `time[time_inds]` (`core.py:750`), and the public
   `row_slice_aware` callback contract (`core.py:619-644`), which documents
   `time` as `(n_time,)` — changing it breaks third-party callbacks. Preserve
   Phase 3's property that any row range equals the full result sliced; the
   Phase 3 ownership convention itself (`digitize(t, time[1:-1])`, empty final
   row) is what this phase replaces, so update `select_spikes_in_rows`
   (`common.py:349-442`) and the tests that pin it
   (`test_row_slice_edge_cases.py`, `test_kde_common.py`,
   `likelihoods/conftest.py`, `test_chunk_boundary_edge_cases.py`,
   `test_chunk_likelihood_preparation.py`, `test_row_slice_parity.py`).
   The four NaN-position → `is_missing` sites (`base.py:3511,3725,4459,4672`)
   compare a per-position-sample mask to `len(time)`; with edges that alignment
   breaks, and mapping NaN samples to decode bins is a C3b gap-policy question.
4. **Generated grids and callers.** `calculate_time_bins` (`base.py:2406-2423`)
   is public but has no callers; it returns `ceil(duration·fs)` left
   timestamps and excludes the interval end. Decide whether to migrate it to
   return edges (with an explicit policy for intervals not divisible by the bin
   width, satisfying 6c) or remove it. The detector itself never generates a
   grid; users pass `time`. Migrate the simulators (which return `time` equal
   to the position-sample grid) and the README/docstring examples. Audit every
   `len(time)`/shape use and classify it as an edge count or observation count,
   including cached, chunked, stationary/covariate, EM, and Viterbi callers.
   EM encoding-weight alignment (`base.py:2094-2101`) must interpolate from
   centers unconditionally; it currently skips interpolation whenever the
   lengths happen to match. Viterbi is never chunked (`core.py:1121-1122`).
5. **No-spike and public documentation.** Re-edit the no-spike docs (already
   corrected for the old convention in `1f2b6b4`) to N rows from N+1 edges,
   and migrate or remove `get_spike_time_bin_ind` and its test-oracle uses. Duration-calibrated intensity changes
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
