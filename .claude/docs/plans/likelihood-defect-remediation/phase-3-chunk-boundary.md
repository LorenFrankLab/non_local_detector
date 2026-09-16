# Phase 3 — Backends bin globally, allocate only requested rows

> **IMPLEMENTED AND VALIDATED** on branch `fix/likelihood-chunk-boundaries`
> (baseline `86e22f08252bfe248ad8e53b5b8dbbfd1b73f110`); the changes are
> uncommitted in the working tree at the time of writing. See the
> [Implementation record](#implementation-record) at the end of this file for the
> chosen interface, the ownership-of-slicing table, the measured likelihood-memory
> evidence, and what was deliberately deferred. The former unexecuted
> implementation snippets and >4× total decoding-memory target were withdrawn
> before implementation and were not reinstated. This phase owns correct event
> ownership and likelihood-specific allocation bounds. Full-session posterior
> storage and checkpointed smoothing belong to
> [Phase 7a](phase-7-performance.md); nothing here bounds total HMM memory.
>
> The requirement sections below are kept as written (they are what the
> implementation was held to); the record states what was actually built.

## Problem

With `n_chunks > 1`, likelihood caching is disabled in `models/base.py`.
The core calls the likelihood function with the full spike arrays and a sliced
time array. Each predictor clips spikes to its own `[time[0], time[-1]]`, so a
spike between adjacent chunks is outside both.

Reproduced under the current timestamp convention:

```text
time = [0..5], spikes = [0.5,1.5,2.5,3.5,4.5], chunks = [0:3], [3:6]
full    : [1 1 1 1 1 0]   total 5
chunked : [1 1 0 1 1 0]   total 4     -> 1 spike lost per boundary
```

The existing `tests/integration/test_core_kde_integration.py` test precomputes
likelihoods and therefore avoids this defect. Regression coverage must exercise
the public path with likelihood caching disabled.

## Scope and dependencies

`n_chunks` is intended to bound likelihood memory, including density evaluation
and reductions. Computing a full-time likelihood and then slicing it does not
meet that requirement. Restricting the final reduction alone also leaves the
clusterless density evaluation over all decoding spikes in memory.

The [representative workload](overview.md#representative-workload) has 1.8 million
time rows and approximately 16,202–64,802 combined hidden bins. Phase 3 must
produce correct likelihood chunks for that future pipeline, but its completion
does not establish bounded memory for the complete HMM or returned posteriors.

Preserve current unchunked event ownership in this phase. Use the settled decode
vocabulary in [C3](shared-contracts.md#c3--time-vocabulary) when naming concepts;
do not independently change the endpoint convention, intensity units, encoding
exposure, C1 floors, or Phase 0 conditioning. Phase 6a migrates the time API, and
6b/6d calibrate intensities and model units. Revalidate chunk ownership against
those contracts before Phase 7 integration.

## Falsification

Before implementation, reproduce the defect through public `predict` with
spikes strictly between adjacent chunk timestamps and caching disabled. Verify
that the unchunked path includes their contributions and the current chunked
path loses them. A helper-only test or cached-likelihood test cannot establish
that the integration defect is fixed.

For memory, profile a single requested likelihood chunk while increasing the
recording length and holding chunk length and fitted model fixed. Identify
full-time spatial arrays or density evaluations over decoding spikes outside
the requested range. Record the failing allocation behavior before rewriting
each affected backend.

## Tasks

### 1. Prototype global event ownership with bounded evaluation

- Prototype an interface that identifies a row range in the full decoding
  timeline. `row_slice` is a candidate API, not a settled signature.
- Assign each decoding spike to its global observation bin consistently, then
  select the spikes belonging to the requested rows **before** expensive density
  evaluation. Slice waveform features using exactly the same selection.
- For clusterless reductions, translate selected global bin IDs into local
  chunk indices and allocate only the chunk's output rows. A full-time
  `segment_sum` followed by slicing is insufficient.
- For sorted backends, form the counts and likelihood arrays needed for the
  requested rows. Avoid repeatedly binning the full recording for every chunk;
  prototype reusable event indices or indexed range lookup.
- Coordinate ordering assumptions with Phase 8's sorted-index contract. A range
  lookup must establish its required order and preserve spike/feature alignment;
  it cannot silently assume inputs are sorted.
- Inventory the fixed encoding-model allocations separately from decoding
  workspace. Large encoding-spike × position kernels are a Phase 7c profiling
  target; limiting time rows does not by itself remove them.

No implementation snippet becomes prescriptive until this behavior has been
run against the real backends and compared with the unchunked reference.

### 2. Cover the entire model assembly path

Thread the row range through both sorted and clusterless detector assembly,
including all registered backends and the no-spike model, which bypasses both
registries. Allocate the assembled likelihood as chunk rows × hidden bins.
Align missing-data masks, local-position likelihoods, non-local penalties,
position interpolation, and time-varying transition inputs with those same
global rows. Define ownership of slicing at each layer so arrays are neither
left unsliced nor sliced twice.

### 3. Integrate both core transition paths

Exercise stationary and covariate-dependent transitions independently; both
detector families can use either core path. Preserve the generic likelihood
callback contract: bind the new backend-specific information at the model layer
or provide an explicit adapter, rather than injecting an unsupported keyword
into arbitrary user callables. A legacy callback must not be presented as a
bounded/global-binning implementation without evidence that it satisfies those
contracts. Choose and test the compatibility behavior in the prototype.

### 4. Preserve requested likelihood outputs

When the caller explicitly requests likelihoods, return the correct rows in
global order for cached and uncached prediction. During Phase 3 an explicitly
requested full likelihood result may still consume `T × N` memory; record that
exception. Unrequested likelihood chunks should be released after use. The
incremental output mechanism and full-session memory guarantee are Phase 7a
work, not prerequisites for shipping the boundary-spike correction.

### 5. Release note

Describe which chunk-boundary spikes were lost, the corrected event ownership,
and the behavior of requested likelihood outputs. Report measured likelihood
memory changes separately from total HMM/output memory.

## Validation

Use existing applicable tolerances; numerical contracts and golden data are not
changed to accommodate chunking. Test proposed APIs only after their signatures
have been selected by prototyping.

| Coverage | Required observation |
|---|---|
| Public prediction, cached vs. uncached | Corrected chunked outputs and evidence agree with the unchunked reference for sorted/clusterless detectors and stationary/covariate transitions. |
| Event ownership | Every in-range spike contributes once across chunk boundaries; cover exact boundaries, recording endpoints under the active contract, chunks with no spikes, singleton row ranges, and a ragged final chunk. Validate empty row requests according to the chosen interface. |
| Backend row ranges | Each supported backend's requested rows equal the corresponding full-reference rows; clusterless waveform selection stays aligned with selected spikes. |
| Assembly branches | No-spike states, missing rows, local-position kernels, and non-local penalties use the same global row indices. |
| Generic callbacks | Existing callback arguments remain valid; adapters and any unsupported capability handling are explicit and tested. |
| Requested outputs | Likelihood rows are ordered and complete when requested; unrequested chunk arrays are not accumulated. |

### Memory evidence

Measure likelihood production separately from the HMM and result assembly.
With fitted inputs and chunk length fixed, increasing total duration must not
introduce a full-time spatial likelihood array or density evaluation for all
decoding spikes on each call. Vary chunk length to show which workspaces scale
with requested rows and selected spikes. Declare recording-sized input/index
metadata and resident encoding allocations separately.

Use allocation/shape checks plus measured host/device peaks on representative
backends. Compiler temporary-memory estimates and live-array snapshots are
supporting evidence; neither alone measures the complete transient process
peak. Report the method and its limitations.

The old **>4× reduction in total predict memory** is not an acceptance criterion:
full posterior retention makes it inappropriate for this phase. End-to-end
bounded-memory acceptance is specified in Phase 7a.

Run core/integration tests and the affected backend tests, including golden
regressions. Cached-path goldens are expected to remain unchanged; investigate
unexpected numerical changes under the existing numerical-validation process.

## Review

Have the implementation reviewed for event ownership before density evaluation,
every model assembly return path, generic callback compatibility, and both core
transition paths. Include the parity and likelihood-specific memory evidence.

---

## Implementation record

Baseline `86e22f08252bfe248ad8e53b5b8dbbfd1b73f110`, branch
`fix/likelihood-chunk-boundaries`. Implemented in three tasks (implementation and
parity; edge-case validation and memory evidence; documentation and full
validation), each independently reviewed. No golden or snapshot file was read,
regenerated, or modified; no existing tolerance, fixture, or convergence
criterion was weakened.

### Chosen interface

`row_slice: slice | None = None` — one optional keyword, added to every public
backend predictor, the no-spike model, the shared count helper, and both
detectors' `compute_log_likelihood`:

```python
predict_<backend>_log_likelihood(time, ..., is_local=False, row_slice=None)
predict_no_spike_log_likelihood(time, spike_times, no_spike_rate=1e-10, row_slice=None)
get_spikecount_per_time_bin(spike_times, time, row_slice=None)
ClusterlessDetector.compute_log_likelihood(..., is_missing=None, row_slice=None)
SortedSpikesDetector.compute_log_likelihood(..., is_missing=None, row_slice=None)
```

Contract, documented on each function:

- `time` is **always the full decoding timeline**, whatever rows are requested,
  so spike-to-row ownership does not depend on how the rows were chunked.
- The result has `row_stop - row_start` rows and equals the full-time result
  sliced by `row_slice`.
- `row_slice=None` is the previous behaviour exactly (default for every existing
  caller).
- Only contiguous (unit-step) slices are accepted; any other step raises
  `ValidationError`. Negative/open bounds are normalized through
  `slice.indices(n_time)`, and a backwards range collapses to an empty one.

Two shared helpers in `likelihoods/common.py` decide ownership once:

```python
resolve_row_slice(row_slice, n_time) -> (row_start, row_stop)
select_spikes_in_rows(spike_times, time, row_start, row_stop) -> SpikeSelection
select_spike_rows(array, selection) -> selected_rows
```

`select_spikes_in_rows` derives its bounds *from* the existing
`np.digitize(t, time[1:-1])` rule rather than approximating it: owning row
`>= row_start` ⇔ `t >= time[row_start]`; owning row `< row_stop` ⇔
`t < time[row_stop]`, extended to `t <= time[-1]` inclusive when the request
reaches the last owning row. The `SpikeSelection` result holds the `indexer`,
local `bin_ind` (`global_row - row_start`), an explicit `indices_are_sorted`
guarantee, and the original `n_spikes`. `select_spike_rows` checks the original
row count before selecting a paired array, including mismatches outside the
requested chunk. This shape-only validation does not read array values. Every
`segment_sum`/scatter uses `num_segments = n_rows`.

Core gained a marker protocol (`core.py`) instead of keyword injection:

```python
ROW_SLICE_ATTRIBUTE = "accepts_row_slice"
row_slice_aware(func)      # decorator; marks the callback
accepts_row_slice(func)    # unwraps bound methods, functools.partial, __wrapped__
_call_log_likelihood_chunk(...)   # used by both chunked drivers
accumulate_log_likelihoods: bool  # new flag on both chunked drivers
```

### Ownership of slicing (input × layer)

| Input | Sliced by | How |
| --- | --- | --- |
| `time` (bin edges / ownership) | **nobody**, for a row-aware callback | the full array is passed; a legacy callback still gets `time[chunk]` |
| output rows | backend | `resolve_row_slice` → `jnp.zeros((n_rows, …))`, `num_segments=n_rows` |
| `is_missing` | **core** (unchanged) | `is_missing[time_inds]`; the model sizes a default to `n_rows` |
| decoding spike times | backend, via `select_spikes_in_rows` | select spikes owned by global rows `[start, stop)`, then bin against that range's internal boundaries |
| decoding waveform features | backend, via the **identical** indexer | selected before any density evaluation |
| position interpolation, local-position kernel, non-local penalty | model / backend | evaluated at `time[row_start:row_stop]` |
| time-varying discrete transitions | **core** (unchanged) | `discrete_transition_matrix_jax[time_inds]` |
| encoding model (fit-time arrays) | **nobody** | untouched; encoding exposure unchanged |
| `no_spike` bin duration | **nobody** | `np.median(np.diff(time))` over the full timeline, prepared by the model once per `_predict` call and reused across chunks; direct likelihood calls compute it on demand |

Nothing is sliced twice: a row-aware backend slices only what the core does not.

### Ordering decision

`select_spikes_in_rows` **establishes** its precondition instead of assuming it:
it verifies ascending order in one `O(n_spikes)` vector pass
(`np.all(spike_times[1:] >= spike_times[:-1])`). Detector predictions now share
that result across states and chunks through a prediction-local `_SpikeTimeOrder`
object; standalone backend calls still check their own inputs. If ascending, two
`np.searchsorted` calls give the contiguous range, the returned `slice` is a
view for the waveform features, and only the *selected* subset is digitized
against the boundaries inside the requested row range. Global selection happens
first, so removing the earlier boundaries simply makes the indices local; the
final timestamp remains excluded from the boundaries to preserve the final
empty row. This also bounds `np.digitize`'s linear monotonicity check to the
chunk instead of scanning the full timeline per unit. If not ascending (NaNs
fail the test too), it falls back to a boolean mask, i.e. exactly the baseline
selection, **in input order**.

All nine `indices_are_sorted=` arguments use the selection's explicit
`indices_are_sorted` field. The helper establishes this guarantee from the
ordering check, independently of the representation of its indexer. Arbitrary
input order is preserved and passes `False`; verified ascending inputs and
early empty selections pass `True`. This removes the pre-existing unsupported
compiler promise without sorting or changing spike/feature alignment. Sorted
inputs keep the fast path. The ordering-preparation follow-up implements the
once-per-prediction work originally deferred to [Phase 8](phase-8-remaining-findings.md).

No `astype`/float32 downcast happens on the selection path (`np.asarray` only),
so timestamp precision is preserved.

### Compatibility behaviour of the chunked drivers

`models.base._prepare_likelihood_callback` owns prediction-local preparation.
The row-slice decorator opts a callback into full `time` plus a global row range
and an already sliced `is_missing` mask. Declaring the private preparation
keywords separately opts it into shared ordering and No-Spike duration work.
The helper binds only declared keywords using `partial`, which preserves core's
marker lookup. Precomputed likelihoods bypass preparation; custom callbacks
without these keywords retain their existing interface.

Per chunk, `_call_log_likelihood_chunk` dispatches:

- **marked callback** → `func(time, *args, is_missing=is_missing_chunk,
  row_slice=slice(start, stop))` — full time plus global rows;
- **unmarked (legacy) callback** → `func(time[time_inds], *args,
  is_missing=is_missing_chunk)` — byte-for-byte the previous call. **No new
  keyword is ever injected into an arbitrary callable.**

Both detector `compute_log_likelihood` methods carry `@row_slice_aware`.
`accepts_row_slice` sees through bound methods, `functools.partial` (nested
included) and `__wrapped__`-only decorators, because silently demoting a marked
callback to the legacy branch would drop boundary spikes while still returning
plausible numbers. Both driver docstrings state explicitly that a legacy callable
keeps the chunk-local call, drops spikes between chunks, and **is not a
global-binning implementation**. The Viterbi drivers still raise
`NotImplementedError` for `n_chunks > 1` and call the callback once with the full
time, so they are unaffected.

### Requested likelihood outputs

Previously `predict(return_outputs='log_likelihood', n_chunks>1)` returned a
dataset with **no** `log_likelihood` variable (`predict` forced
`cache_likelihood=True`; `_predict` immediately re-disabled it for `n_chunks>1`,
so the driver returned `None`). Now both chunked drivers take
`accumulate_log_likelihoods`; when set, the first chunk determines the dtype and
column count of a single preallocated host output. Each chunk is copied into its
global row slice **before** donation to the jitted filter. Assignment owns the
copied values, so CPU views cannot alias a donated buffer. No list of chunk
copies or final concatenation is retained alongside the output. The returned
likelihood covers every row in global order. `_predict` takes `accumulate_log_likelihoods`; both `predict` methods pass
`accumulate_log_likelihoods=("log_likelihood" in requested_outputs)` and force
caching only when `n_chunks == 1`. `_DetectorBase._estimate_parameters` passes
`accumulate_log_likelihoods=(store_log_likelihood or "log_likelihood" in
requested_outputs)` on the **final E-step only**; in-loop E-steps retain no
likelihood chunk array.

This `T × N` allocation is the documented exception recorded by task 4 of this
phase. It is called out in the `cache_likelihood` docstring of both public
`predict` methods, of `_DetectorBase.estimate_parameters`, in `_predict`, and in
both driver docstrings. When the likelihood is **not** requested, no chunk array
is retained.

### Behaviour changes beyond the boundary fix

- **A stored `log_likelihood_` is never an input.** `_estimate_parameters` (both
  E-steps) and `most_likely_sequence` used to read
  `getattr(self, "log_likelihood_", None)`, so a run that stored it contaminated
  the next run on different data — 97.6 % of `acausal_posterior` entries wrong
  (max abs 0.115) with identical timeline lengths, independent of
  `cache_likelihood`. The attribute is now dropped at the start of every
  estimation run and on any refit (`_fit`, both `fit_encoding_model`
  implementations, and the EM M-step's encoding update), and the reuse sites
  consume a run-local variable. Within one run, a likelihood computed in that run
  is **now** reused across EM iterations — a new runtime optimisation, not
  previously-existing behaviour: at baseline the attribute was written only
  *after* the loop, so `cache_likelihood=True` recomputed every iteration. It is
  bit-identical to recomputing (the likelihood depends only on the data and the
  encoding model) and is dropped as soon as the M-step updates the encoding
  model. `most_likely_sequence` always recomputes. `store_log_likelihood=True`
  still stores the final likelihood as an output
  (`models/base.py:1794` helper, `:1649`, `:2012`, `:2119`, `:2243`, `:2352`,
  `:2370`, `:2997`, `:3982`).
- **`accumulate_log_likelihoods` is keyword-only and last** in both chunked
  drivers (`core.py:722-723`, `:1347-1348`). It had been inserted before the
  pre-existing `dtype`, so a positional `dtype=jnp.float64` set the accumulation
  flag and left the computation in float32.
- **Per-spike selection happens before any conversion**, through
  `likelihoods.common.select_spike_rows`, in all four clusterless backends for
  both the decoding spike times and the waveform features
  (`clusterless_kde:477,609-612`, `clusterless_kde_log:1695,1836,1880`,
  `clusterless_gmm:746,924-925`, `clusterless_diffusion:602,610,725`).
  Converting first made a chunk call scale with the whole recording's spikes
  (`memory-evidence.md` D2), and for a `jax.Array` input it also **retained** the
  gathered host copy on the input array via `jax.Array._npy_value`
  (`memory-evidence.md` D4). The helper selects on device where JAX allows it and
  falls back to gather-then-slice under explicitly-sharded meshes; that fallback
  also fixed five of eight backend x `is_local` combinations that previously
  raised `ShardingTypeError` on explicitly-sharded inputs. The one remaining
  recording-length read is `select_spikes_in_rows`' own pass over the 1-D spike
  times, which its ascending-order check requires. Detector predictions now
  share that host conversion and ordering result across chunks; direct backend
  calls without preparation still perform the read themselves.
  The fallback now catches only JAX's `ShardingTypeError`: unrelated indexing,
  device and memory failures propagate without materializing a full host copy.
  The private exception import is guarded for older supported JAX releases
  lacking that type; those releases catch no exceptions here. Eight regression
  cases verify original exception identity and no host copy for slice and mask
  selection. Both Auto and Explicit two-device checks still pass.
- **`clusterless_gmm`'s local branch no longer converts the recording-length
  `position` at all** (`clusterless_gmm.py:865-875`). It was device-copied on
  every chunk call purely to read `.dtype` (every consumer took
  `np.asarray(position)` straight back to the host), costing one float32 copy of
  the whole record per call: host peak grew +1,928 KiB over a 2,000 -> 400,000
  sample position record at a fixed 5-row request, versus +375 KiB for
  `clusterless_kde` and +370 KiB for `clusterless_diffusion`. The working dtype
  now comes from a one-sample probe (`_as_jnp(position[:1]).dtype`, the dtype a
  full conversion would have produced) and the interpolation receives the host
  array, as it already did in every other backend. After: +373 KiB, matching the
  other two; the residual is `scipy.interpolate.interpn`'s own ~1 byte/sample
  temporary inside `get_position_at_time`, which is shared, pre-existing, and not
  a device buffer. Numerical note: the local GMM path now interpolates position
  in the caller's dtype (float64 in practice) instead of a float32 truncation,
  which is what `clusterless_kde` and `clusterless_diffusion` always did — the
  interpolated positions are at worst as accurate as before, no snapshot or
  golden fixture covers `clusterless_gmm`, and the full GMM test surface passes
  unchanged.

- `predict_no_spike_log_likelihood` uses the full timeline's bin duration,
  prepared once by `_predict` or computed on demand for direct calls.
  The expression (`np.median(np.diff(time))`) is unchanged; because a
  row-aware callback now receives the full `time` instead of `time[chunk]`, a
  chunk can no longer rescale `no_spike_rate`. This restores unchunked parity.
  Two legacy cases actually differed (verified): non-uniform timestamps, where
  each chunk's median interval is its own; and a single-row chunk
  (`n_chunks == n_time`), where `np.diff` is empty, the median is `NaN`, and the
  chunk's whole no-spike log likelihood became `NaN` (measured: 6/6 rows `NaN`
  on a 6-row timeline). A ragged final chunk on a *uniform* timeline is **not**
  affected — the median interval is 1.0 in every chunk of a 13-row/5-chunk
  split.
- `clusterless_gmm` no longer converts the full-recording `position` /
  `position_time` before the `is_local` dispatch (the non-local branch never
  read them). The local path now interpolates host position values at the
  caller's precision, with the numerical effect described above. Side effect:
  `predict_clusterless_gmm_log_likelihood(position=None, is_local=False)` now
  returns a result instead of raising `AttributeError`, matching the other
  clusterless backends, whose non-local paths never touch `position` either.
- The incidental re-export of `likelihoods.common.get_spike_time_bin_ind` was
  dropped from `clusterless_kde`, `clusterless_kde_log`, `clusterless_gmm` and
  `clusterless_diffusion`, which no longer use it. Import it from
  `likelihoods.common`.

### Numerical parity

Sorted-spikes backends and `no_spike` accumulate integer spike counts per row
before any float work and are **bit-identical** (`rtol=0, atol=0`) between a row
request and the corresponding slice of the full-time result. The clusterless
backends scatter-add one float32 row per selected spike
(`jax.ops.segment_sum`), so a different row range can group the reduction
differently; measured deviations are at float32 epsilon (≤ 1.2e-7 relative:
`clusterless_kde` 1.5e-5 abs, `clusterless_gmm` 2.0 abs on values of order 2e7,
`clusterless_diffusion` 3.8e-6 abs), and the tests use `rtol=1e-6, atol=1e-5`
for that family.

`clusterless_kde_log` additionally differs by ≤ 7.6e-6 absolute on non-local log
intensities. Its `_compensated_linear_marginal` stabilizer is the maximum over
the **decoding** axis of the current block, so which spikes share a block changes
the rounding. Confirmed as reordering, not a defect: with one spike per block the
row result is bit-identical to the full-time result, and the *full-time* result
alone moves by the same 3.8e-6 when `block_size` changes; the differing rows are
scattered through the interior of the range, not at its ends. A lost or
double-counted spike moves a log likelihood by O(1) (measured 23–46 across
backends), five orders of magnitude above these tolerances.

### Likelihood-memory evidence (summary)

Full tables, method and limitations:
`.superpowers/sdd/phase-3-chunk-boundary/memory-evidence.md`; benchmark
`scripts/benchmark_chunk_likelihood_memory.py`.

**Claim under test.** Producing one requested chunk costs
`O(chunk rows × position bins)` + `O(selected spikes)` and does not grow with the
total length of the recording. **Not claimed:** anything about total HMM memory —
the driver still holds the full `time`, per-unit spike-time and waveform-feature
arrays, `is_missing`, and the posterior it is assembling; those are
recording-sized by construction and are reported separately.

*Duration sweep* (recording 60 s → 960 s = 6,001 → 96,001 rows, chunk fixed at
500 rows, one fitted encoding model held constant). Every backend returns exactly
`(500, 50)` = 97.7 KiB with `num_segments = 500`, 400 selected spikes and 6.2 KiB
of selected waveform features at **all five durations**, while the full-time
`(n_time, 50)` array a naive implementation would build grows 1.1 → 18.3 MiB and
is never allocated. `tracemalloc` peak per call:

| backend | 60 s | 120 s | 240 s | 480 s | 960 s | slope |
| --- | --- | --- | --- | --- | --- | --- |
| `sorted_spikes_kde` | 66.8 KiB | 66.9 | 66.6 | 66.8 | 65.9 | −0.009 B/sample |
| `sorted_spikes_glm` | 65.9 KiB | 65.6 | 65.8 | 65.9 | 65.3 | −0.006 B/sample |
| `clusterless_kde` | 66.4 KiB | 65.9 | 65.0 | 65.2 | 65.3 | −0.013 B/sample |
| `clusterless_kde_log` | 53.5 KiB | 53.0 | 53.5 | 52.7 | 52.7 | −0.009 B/sample |
| `clusterless_gmm` | 54.2 KiB | 53.9 | 53.7 | 54.1 | 58.2 | +0.045 B/sample |
| `clusterless_diffusion` | 94.8 KiB | 94.1 | 94.3 | 94.3 | 110.7 | +0.180 B/sample |

`jax.live_arrays()` delta was exactly the returned chunk (97.7 KiB) for every
backend at every duration. The small reproducible step at 960 s
(`clusterless_diffusion` +16 KiB, `clusterless_gmm` +4 KiB) is a step, not noise;
a `tracemalloc` snapshot diff attributes it to no production allocation (small
JAX dispatch/sharding bookkeeping blocks), and it remains **unexplained beyond
"JAX per-call bookkeeping that scales weakly with argument size"**. At 0.18
B/sample it is ~1000× smaller than a full float32 array copy.

*Chunk sweep* (recording fixed at 960 s, chunk 125 → 2,000 rows).
`num_segments == chunk rows` and selected spikes grow exactly linearly with the
request; the `live_arrays()` delta tracks the output 24.4 → 390.6 KiB.

*Device/XLA.* `clusterless_kde_log.estimate_log_joint_mark_intensity` is the one
jitted likelihood kernel lowerable with concrete arguments; its
`memory_analysis()` temp workspace is exactly linear in **selected** decoding
spikes (1.2 → 18.3 MiB over 125 → 2,000 spikes, a constant 9.4 KiB/spike) with no
dependence on recording length.

*Budgets kept separate* (at 960 s): resident encoding allocations 23.4–414.9 KiB
(a property of the fit, independent of duration); recording-sized input/index
metadata ≈ 4.1 MiB (`time`, `position_time`, `position` 750 KiB each,
`spike_times` 600 KiB, `spike_waveform_features` 1.2 MiB, `is_missing` 93.8 KiB) —
linear in duration and **unchanged by this phase**; per-chunk workspace ≈ 100 KiB.

*Cost of the ordering check:* the earlier 21.2–56.3 µs timings measured the
entire selector, including `np.digitize`'s full-timeline monotonicity scan, and
could not isolate spike-order validation. A follow-up measurement on a one-hour
recording at 2 ms resolution found 0.429 ms for the old selector, of which
0.412 ms was full-timeline digitization and 0.015 ms was the ascending-spike
check (72,000 spikes/unit). The digitization scan is now bounded to the chunk.
The ascending check costs `O(n_spikes)` time and roughly one byte per spike
transiently. The ordering-preparation follow-up now pays it once per distinct
spike-time object per detector prediction, instead of once per state/chunk.
Standalone backend calls without preparation retain the original check. The
previous ~0.4 ms extrapolation for that check is withdrawn.

**Method and limitations, as recorded in `memory-evidence.md`:** measurements are
`tracemalloc` peak, 1 ms-sampled RSS delta, `jax.live_arrays()` byte deltas and
one compiled module's `memory_analysis()`, each preceded by an identical warm-up
call and `jax.block_until_ready`. These are **scaled** experiments (100 cm track,
50 interior position bins, minutes of simulated data); the 180 × 180 cm × 1 h
allocation was deliberately not attempted. Per-chunk output scales as
`chunk_rows × n_state_bins × 4 B`, so a ~32k-state-bin arena implies ~64 MB for a
500-row chunk — a **linear extrapolation** of the measured shape rule, not a
measurement. The device-side numbers are CPU-backend numbers; GPU allocation
behaviour was not measured (no GPU available, and `memory_stats()` is `None` on
the CPU backend). RSS sampling at 1 ms can miss a shorter peak, so it is
corroboration rather than the primary measurement. Only `is_local=False` was
swept. A single run per point, no repetitions or confidence intervals: the
duration sweep's flatness is robust (five points within ~1 KiB) but individual
peaks are ±20 KiB noisy.

### Runtime follow-up

Two recording-length-dependent regressions were found after the initial memory
validation: `np.digitize(selected, time[1:-1])` checked all time boundaries for
each unit/chunk, and No-Spike recomputed the full median time step for every
chunk. The former now uses only the requested row boundaries after global spike
selection. The latter is prepared once in `_DetectorBase._predict` and bound
to that call's likelihood callback with `functools.partial`. It is not stored
on the detector or in a global cache, so later predictions, including a mutated
time array, compute a fresh duration. Supplied likelihoods skip preparation;
custom likelihood overrides retaining the older signature remain callable.
The same callback serves stationary and covariate-dependent filtering.

Reproduce with:

```sh
uv run python scripts/benchmark_chunk_likelihood_runtime.py /tmp/phase3-runtime
```

The script also supports baseline source via `PYTHONPATH`. Measured against
pre-PR `86e22f0` and pre-performance-fix PR `a04c4da`, in separate sequential
CPU processes: one hour, 2 ms decoding bins (1.8 million rows), eight units at
20 Hz, a fixed 500-row chunk, 50 spatial bins, 300 encoding spikes/unit and
four waveform features. Each call selects 20 spikes/unit. Three warmups then
15 timed calls, synchronized with `jax.block_until_ready`; fitting and JIT
compilation excluded. The selected chunk has no spike in the interval the old
baseline dropped, making the work comparable.

| Backend | Path | Pre-PR main ms | Pre-fix PR ms | Fixed ms |
| --- | --- | ---: | ---: | ---: |
| Sorted KDE | non-local | 2.507 | 5.954 | 2.573 |
| Sorted KDE | local | 2.492 | 5.844 | 2.594 |
| Clusterless log-KDE | non-local | 3.832 | 5.886 | 3.297 |
| Clusterless log-KDE | local | 9.260 | 10.505 | 8.245 |
| GMM | non-local | 2.791 | 5.696 | 1.582 |
| GMM | local | 6.352 | 8.119 | 4.076 |
| Diffusion | non-local | 8.678 | 11.000 | 6.818 |
| Diffusion | local | 13.838 | 15.036 | 11.909 |
| No-Spike | — | 1.808 | 8.444 | 1.480 |

All 18 output arrays across 60-second and one-hour recordings were bit-identical
to the pre-fix PR. Per-chunk likelihood time fell 21–83% at one hour. No-Spike's
one-time preparation took 4.42 ms for the hour timeline in this run and is
excluded from its per-chunk figure; ordinary direct backend calls without a
prepared duration still compute the full median. Sorted KDE is within ~4% of
main; the other measured paths are faster. These are single-session CPU
measurements on a small spatial grid, not an end-to-end HMM benchmark, a GPU
benchmark or a production-arena scaling claim. No golden or tolerance changed.

### Reviewer-clarity follow-up

The spike selection now carries named metadata rather than encoding the JAX
sorted-index promise in its indexer type. Feature pairing is validated from the
original row counts before selection. Before the fix, sorted inputs accepted
extra feature rows and prefix chunks could miss a short feature array; unsorted
inputs rejected the same mismatch. Thirty-two backend cases cover both paths,
both ordering cases, and both missing/extra rows. Twelve of these failed before
the fix; the GMM entry point already had independent pairing validation.

Both core drivers now copy requested likelihood chunks into one preallocated
host output. The two driver storage regressions failed before the fix with
about 2.13 MB retained for a 1.05 MB likelihood; both now retain about 1.07 MB,
below a 1.5-output budget that permits bookkeeping but excludes a second full copy. Numerical
kernels are stubbed only in this allocation test; the existing real-driver
accumulation tests continue to cover numerical results and donation safety.
Callback preparation is centralized and its two opt-in contracts documented.
Compatibility wording now distinguishes exact spike counts, reduction rounding,
and local GMM position-interpolation precision.

Validation: **48 targeted tests passed**, **207 float64 tests passed** (20
allocation/sharding cases deselected), and all **18 saved likelihood arrays were
bit-identical** to the pre-fix branch outputs. Ruff checks passed. An independent
review found no additional code issues. Goldens, snapshots and tolerances were
not changed.

The full run completed with **1654 passed, 6 skipped and one Hypothesis timing
failure** in 971.36 s. The unchanged `test_normalize_preserves_proportions`
exceeded its 200 ms deadline (267.44 ms initially, 0.71 ms on Hypothesis retry);
its numerical assertions did not fail. The isolated rerun passed, followed by
**all 17 tests in the property module** on a fresh run. This is recorded separately from a clean full-suite pass; numerical assertions and
wall-clock settings were not changed.

### Test coverage and validation results

296 tests in six modules, including the runtime, exception-handling, JAX,
ordering-preparation and reviewer-clarity follow-ups:

| module | tests | covers |
| --- | --- | --- |
| `tests/likelihoods/test_row_slice_parity.py` | 36 | every registered backend × both `is_local`, plus `no_spike`: a row range equals the full-time slice, and every row partition tiles the full result; zero-rate sentinels asserted per backend in the fixture; `resolve_row_slice` normalization and non-unit-step rejection |
| `tests/likelihoods/test_row_slice_edge_cases.py` | 179 | helper- and backend-level edge cases: endpoint convention, spikes on timestamps / between timestamps / duplicates, irregular and ragged partitions, empty and singleton row requests, spike-free chunks and all-units-empty, spike/feature alignment under shuffled input in both local and non-local paths, direct checks of JAX sorted-index promises, original spike/feature row-count validation independent of ordering, NumPy/JAX feature-selection allocation checks, propagation of unexpected selection errors without host copies, plus guard-the-guard assertions that the legacy chunk-local call really does differ |
| `tests/integration/test_chunk_boundary_spikes.py` | 7 | public `predict(n_chunks=5, cache_likelihood=False)` vs `n_chunks=1` for both detector families, the covariate-dependent core path, `is_missing` straddling every boundary, and requested `log_likelihood` from `predict` and from `estimate_parameters` |
| `tests/integration/test_chunk_boundary_edge_cases.py` | 26 | the same public path over ragged chunk counts (5/6/7 with `n_time % n_chunks != 0`), singleton chunks (`n_chunks == n_time`), spike-free chunks, unsorted spike input, `is_missing`, and preservation of a legitimate `-inf` mask (delta local-position kernel), comparing acausal + causal posteriors, both state-probability sets, evidence and the full `log_likelihood` |
| `tests/core/test_row_slice_callback.py` | 13 | both chunked drivers: a marked callback receives the full time and tiling global rows, a legacy callback receives the sliced time and no `row_slice` (and yields a different answer), the marker survives bound methods / `partial` / `__wrapped__`, accumulated rows cover every row without retaining a second full host array, and positional dtype compatibility is preserved (two tests require x64) |
| `tests/models/test_chunk_likelihood_preparation.py` | 35 | digitization scans only chunk boundaries while matching independent global indices; No-Spike computes the full median once per prediction for both detector families and both drivers, refreshes mutated timelines, skips preparation for supplied likelihoods, and supports custom callbacks with the older signature; ordering is checked once per prediction/direct call, refreshed after input mutation, and skipped for non-owning rows; host conversion is shared for NumPy/JAX/list spike times |

| Command | Result |
| --- | --- |
| `uv run --no-sync pytest -q` after the JAX sorted-index fix | **1588 passed, 6 skipped** in 693.21 s (11:33), including property, integration, EM, snapshot and golden tests. The skips are unchanged from the exception-handling run below. |
| Exception-handling regressions, before tightening the fallback | **8 failed**: injected `RuntimeError`, `ValueError`, `TypeError` and `MemoryError` were swallowed for both mask and slice selection |
| Selection and sharding tests after tightening the fallback (`-k 'select_spike_rows or sharded'`) | **18 passed**, including both two-device sharding modes; an isolated compatibility check also confirmed module import and ordinary selection when the private JAX exception type is unavailable |
| `uv run --no-sync python -m doctest src/non_local_detector/likelihoods/no_spike.py` | All **8 examples passed** after correcting the output shape; the two shape assertions failed before the documentation fix |
| `uv run --no-sync pytest -q` after tightening the selection fallback | **1580 passed, 6 skipped** in 731.62 s, including property, integration, EM, snapshot and golden tests; skip reasons unchanged from the performance-fix run below |
| Runtime regression tests, before the performance fix (16 cases at that point) | **10 failed, 6 passed**: digitization scanned 999 boundaries for a 20-row request; No-Spike computed three full medians for three chunks, in both detector families and both drivers |
| `JAX_ENABLE_X64=1 uv run pytest` on `test_chunk_likelihood_preparation.py`, `test_row_slice_parity.py`, `test_row_slice_edge_cases.py`, `test_row_slice_callback.py` after the performance fix | **180 passed** in 66.16 s, including the two positional dtype compatibility tests skipped without x64 |
| `uv run python scripts/benchmark_chunk_likelihood_runtime.py /tmp/phase3-runtime` after the performance fix | All **18** saved likelihood arrays bit-identical to pre-fix PR `a04c4da`; runtime table above |
| `uv run pytest src/non_local_detector/tests/integration/test_chunk_boundary_spikes.py -q -p no:randomly` (**on the untouched baseline**, 4 tests at that point) | **4 failed** — `TypeError: get_spikecount_per_time_bin() got an unexpected keyword argument 'row_slice'`; sorted chunked-vs-unchunked `acausal_posterior` mismatched in 51,627/52,200 entries (98.9 %, max abs 0.34); clusterless 23,917/53,400 (44.8 %, max abs 0.011); `assert 'log_likelihood' in chunked` |
| `uv run --no-sync pytest -q` after the performance fix | **1572 passed, 6 skipped** in 659.80 s (10:59), including property, integration, EM, snapshot and golden tests. Skips: three tests requiring `JAX_ENABLE_X64`, two manual visualization tests and one missing optional `sortingview` dependency. The two skipped row-callback dtype tests passed in the separate x64 run above. |
| Initial implementation: `JAX_ENABLE_X64=1 uv run pytest <the original five modules> -q --no-header -p no:randomly` | **167 passed** in 70.52 s; no float64-specific skip or behaviour change in these modules |
| `uv run --no-sync ruff check src/ scripts/benchmark_chunk_likelihood_memory.py scripts/benchmark_chunk_likelihood_runtime.py` | All checks passed |
| `uv run --no-sync ruff format --check src/ scripts/benchmark_chunk_likelihood_memory.py scripts/benchmark_chunk_likelihood_runtime.py` | 157 files already formatted |
| `git status --porcelain src/non_local_detector/tests/golden_data` | empty — all 8 golden `.pkl` files untouched; no snapshot fixture modified; `--snapshot-update` was never passed |
| `uv run python scripts/benchmark_chunk_likelihood_memory.py --backends clusterless_gmm --sweep duration` | independent re-run: peaks 56.7 / 56.8 / 55.9 / 55.8 / 59.6 KiB over 60 → 960 s (flat; pre-fix was 80.1 → 435.0 KiB), with identical shape columns at every duration |

Pre-existing ruff findings in unrelated scripts (2 × `B007` in
`scripts/profile_optimized_kde.py` and 14 legacy files that `ruff format` would
rewrite) were left alone. Runtime-follow-up checks cover the modified Python
files, including both chunk-likelihood benchmark scripts.

### JAX audit follow-up

The [JAX `segment_sum` contract](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html)
allows unsorted IDs, but `indices_are_sorted=True` promises that they are sorted.
The nine clusterless likelihood reductions now make that promise only when the
selection helper established it. The regression checks inspect actual IDs at the
JAX call boundary: CPU scatter implementations need not exploit the hint, so
numerical parity alone would not catch a false promise. Before the fix, **7 of
16 cases failed** the contract check; after it, **all 16 passed**, including local
and non-local paths and fitted and zero-rate electrodes. The same **16 passed**
under `JAX_ENABLE_X64=1`. The full suite passed **1588 tests, with 6 unchanged
skips**. Ruff check/format and independent code review were clean.

Numerical comparison saved 32 likelihood arrays before and after: four
clusterless backends × local/non-local × sorted/shuffled × full/requested rows.
All were **bit-identical** on JAX 0.9.0 CPU. Goldens and tolerances are unchanged.

The surrounding audit verified module-level JIT functions, scan-based HMM
recurrences, and explicit host copies of requested likelihoods before buffer
donation. With `jax.log_compiles(True)`, each clusterless backend/path was warmed
on a seven-row request, then called three more times with changed waveform values
and alternating row offsets but the same 14 selected spikes per electrode. All
eight paths logged **zero additional compilations**. JAX feature selection also
reused its dynamic-slice executable across equal-sized selections at different
offsets. These checks cover fixed shapes: different selected-spike counts, chunk
lengths, or block remainders can still compile additional executables.

Host event selection preserves timestamp precision, and SciPy interpolation
remains outside compiled kernels. Selected JAX features return to the host to
avoid mixing their input sharding with single-device encoding arrays. The
explicit-sharding full-input fallback remains a documented limitation. The
ordering follow-up below shares the spike-time ordering check across chunks.
Padding/bucketing or a consistent device
placement design needs a separate measured change; this audit does not claim
GPU performance or eliminate those transfers. Only CPU devices were available.

### Ordering preparation follow-up

`_predict` now binds a fresh `_SpikeTimeOrder` to its likelihood callback. Each
spike train is converted to host values and checked for ascending order on first
use, then reused across observation states and chunks. All eight registered
backends (including MRF's shared diffusion predictor), both local paths and
No-Spike forward that object to the common selector. A direct
`compute_log_likelihood` call creates its own preparation. Direct backend calls
without the private preparation keyword retain independent input checking.

The cache is local to the call and holds strong references to its source arrays,
so object identity cannot be recycled while the cached result is live. Nothing
is stored on the detector or globally, and no cache enters JAX kernels. Every
later prediction rechecks its inputs, including arrays modified in place.
Precomputed likelihoods bypass preparation, and custom callbacks with older
signatures remain supported. Inputs must remain unchanged during one prediction.
Empty row requests and the terminal empty row skip spike reads and preparation.

Sorted inputs retain `indices_are_sorted=True`. The repeated work changes from
`O(states × chunks × spikes)` order validation to one `O(spikes)` check per
distinct input plus binary searches per requested range. Unsorted inputs retain
their original order and still need a per-chunk `O(spikes)` selection mask.
This does not remove full recording inputs, posterior storage, density work or
variable-shape JAX compilation costs.

Regression checks count the actual recording-length ordering reduction: all
eight detector-family × transition-driver × chunk-count cases failed before the
change and passed after it. They then shuffle the same input arrays in place and
verify that the next prediction rechecks and matches independent direct
likelihoods. Additional coverage checks direct-call freshness, one host conversion
for NumPy/JAX/list inputs, no work for non-owning rows, and prepared/unprepared
sorted/shuffled parity through every backend and both local modes. The runtime
benchmark's 18 saved likelihood arrays were bit-identical before and after.

Validation: the full regression run passed **1617 tests, with 6 unchanged skips**
in 824.29 s. That run began before the final empty-row early return; **182 focused
tests passed on the final implementation**, including all four new empty-row
regressions, all backend row-slice edge cases and detector preparation checks.
The preparation and sorted/shuffled parity subset also passed **67 tests under
`JAX_ENABLE_X64=1`**. Independent review and ruff check/format were clean. No golden,
snapshot or tolerance changed.

The initial separate-process 20 Hz timing comparison was mixed (roughly −7% to
+11% savings on one-hour inputs), so it does not establish a broad speedup. The
benchmark now offers a paired mode to isolate the preparation effect:

```bash
uv run python scripts/benchmark_chunk_likelihood_runtime.py /tmp/ordering-paired \
  --compare-ordering --spikes-per-second 100
```

It alternates prepared and unprepared calls on the same fitted model and inputs,
warms both paths three times, and measures thirty synchronized calls per path.
Every paired result is checked for exact equality. This compares this branch
with/without ordering reuse, not against `main`. CPU, eight electrodes, one hour
at 100 spikes/s/electrode (360,000 spike times each), 2 ms time bins, a fixed
500-row request, 50 spatial bins, 300 encoding spikes/electrode and four waveform
features; fitting and compilation are excluded.

| backend / path | recheck each chunk (ms) | prepared (ms) | less time |
| --- | ---: | ---: | ---: |
| sorted KDE non-local | 2.764 | 2.140 | 22.6% |
| sorted KDE local | 2.569 | 2.041 | 20.6% |
| log-KDE non-local | 2.425 | 1.831 | 24.5% |
| log-KDE local | 7.208 | 6.633 | 8.0% |
| GMM non-local | 3.506 | 3.489 | 0.5% |
| GMM local | 4.367 | 3.773 | 13.6% |
| diffusion non-local | 7.290 | 6.605 | 9.4% |
| diffusion local | 11.033 | 10.384 | 5.9% |
| No-Spike | 1.939 | 1.325 | 31.6% |

Preparing ordering for all eight electrodes cost **0.57–0.64 ms once**; No-Spike's
separate timeline-median preparation cost 4.43 ms once. Those costs are reported
separately from per-chunk timings. GMM non-local was essentially unchanged; the
other paths saved about 6–32%. On the 60-second recordings the paired differences
were only −1.9% to +2.2%. These are small-grid likelihood-only CPU measurements,
not GPU or end-to-end HMM speedups, and do not establish production-arena scaling.

### Deferred — explicitly NOT implemented here

- **Phase 6a endpoint migration.** The endpoint convention is preserved exactly:
  a spike at `time[-1]` still lands in row `n-2` and the final row still never
  owns a spike, so `predict(...)` 's last row has no spike contribution.
  `test_endpoint_convention_is_unchanged` pins this and will need updating when
  6a migrates it.
- **Phase 7a full-session posterior and incremental outputs.** Unchanged. The
  `T × N` accumulation for an explicitly requested likelihood, the full posterior
  arrays and the recording-sized inputs all remain.
- **GPU validation and device placement.** Ordering preparation is implemented;
  CPU coverage includes both sharding modes, but GPU runtime and a device-only
  feature-selection path remain unmeasured. Unsorted-input masks remain linear
  in the full spike count per chunk.
- **C1 floors and C3b encoding exposure.** Untouched. No floor, background model
  or occupancy guard was added; `LOG_EPS` sentinels, `safe_log`, `EPS` clips,
  legitimate `-inf`s and NaN diagnostics are unchanged.
- **Empty row requests from the public path.** `slice(a, a)` is defined and tested
  at the helper and backend layers, but it is unreachable from `predict`: both
  chunked drivers raise `ValueError` for `n_chunks > n_time` before splitting
  (`core.py:794`, `core.py:1416`), already pinned by
  `tests/core/test_chunked_parity.py`.
- **Length validation on a passed-in `log_likelihoods` (was M-2 of the final
  review; the hazard it described is now closed by invalidation).** The user's
  review showed the length check was the wrong fix: two runs with the *same*
  timeline length still contaminated each other (97.6 % of posterior entries
  wrong). A stored `log_likelihood_` is now treated as an OUTPUT and is never
  consumed by a later call, so no stale array can reach
  `log_likelihoods_jax[time_inds]` through the detector. What remains is the
  underlying robustness gap for a caller who passes `log_likelihoods=` to a
  chunked driver *directly*: neither driver checks
  `log_likelihoods.shape[0] == n_time`, and a JAX gather with out-of-range
  indices clamps silently instead of raising. Pre-existing for `n_chunks == 1`
  and no longer reachable from the detector API. **Trigger / follow-up:** a
  one-line guard in both chunked drivers — `if log_likelihoods is not None and
  log_likelihoods.shape[0] != n_time: raise ValidationError(...)` — before the
  `jnp.asarray` conversion.
- **Usability observation, not fixed.** `clusterless_gmm`'s default mixture sizes
  (64/32/32) raise `RuntimeError: GMM fitting failed: a component's covariance
  became singular` on small or low-rank encoding fits. The error is specific and
  actionable, so both the benchmark and the edge-case fixture work around it with
  16/8/8 components and `reg_covar=1e-4`.
