> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 3 — Backends bin globally, allocate only requested rows

## Problem

With `n_chunks > 1`, likelihood caching is disabled
(`models/base.py:1719-1721`), so each chunk recomputes from the **full** spike
arrays against a **sliced** time array (`core.py:618-622` stationary,
`core.py:1221-1225` covariate-dependent). Each predictor clips spikes to its own
`[time[0], time[-1]]`, so a spike between adjacent chunks is outside both.

Reproduced:

```
time = [0..5], spikes = [0.5,1.5,2.5,3.5,4.5], chunks = [0:3], [3:6]
  full    : [1 1 1 1 1 0]   total 5
  chunked : [1 1 0 1 1 0]   total 4     -> 1 spike lost per boundary
```

The existing integration test avoids this path by precomputing likelihoods
(`tests/integration/test_core_kde_integration.py:81`).

## Design decision

`n_chunks` exists to bound **likelihood** peak memory, not only the HMM pass
(user-confirmed). So "compute the full likelihood inside each chunk and slice the
result" is not acceptable: it has full-likelihood peak memory *and* ~`n_chunks`×
the work, which is strictly worse than simply caching once.

An earlier draft proposed exactly that. It was also broken in detail: it passed a
chunk-sized `is_missing` alongside a full-length `time`, and both detectors apply
that mask before any slice would occur (`models/base.py:3237`, `:4198`), so normal
chunks raise on broadcast and singleton masks silently broadcast.

**This phase instead pushes global binning into the backends.** A backend receives
all edges and a row range; it bins events against the full edge array and
allocates only `(stop − start, n_bins)`.

## Contracts referenced

- [C3 — Time vocabulary](shared-contracts.md#c3--time-vocabulary) — this phase
  introduces `row_slice` but does **not** change edge semantics; it must work
  under the current convention and survive phase 6a unchanged.

## Falsification

Before implementing, write `test_no_spikes_lost_across_chunks` so it drives the
**public predict path** with `n_chunks>1`, not `get_spikecount_per_time_bin` over
slices. A helper-level test cannot pass while the helper is unchanged, and this
phase does not change it — an earlier draft made that mistake. The test must fail
on `main` for the right reason: a spike present in the unchunked posterior and
absent in the chunked one.

## Tasks

### 1. Add `row_slice` to the backend signature

Every predictor in both registries gains a keyword-only
`row_slice: tuple[int, int] | None = None`. Semantics: bin against all of `time`;
return only rows `[start, stop)`.

For the sorted backends this is mechanical — `get_spikecount_per_time_bin` already
produces a full-length count vector, so the change is to slice the count matrix
before the matmul rather than after:

```python
    counts = _spike_counts_matrix(spike_times, time, ...)   # global binning
    if row_slice is not None:
        counts = counts[row_slice[0] : row_slice[1]]
    log_likelihood = counts @ log_interior_fields
```

For the clusterless backends the segment reduction already targets global bins;
slice `num_segments` output rather than the inputs:

```python
    contribution = jax.ops.segment_sum(..., num_segments=n_time)  # global
    if row_slice is not None:
        contribution = contribution[row_slice[0] : row_slice[1]]
```

Slicing after a global `segment_sum` does not reduce that reduction's peak. If
profiling in task 4 shows the reduction dominates, the follow-up is to restrict
`segment_ids` to the range and offset them — record that as a note, do not
speculatively implement it.

### 2. Thread it through the model layer

Both `compute_log_likelihood` implementations (`models/base.py:3047`, `:4011`)
gain `row_slice`. They must:

- pass it to the backend;
- allocate the assembly array as `(stop − start, n_state_bins)`, not `(n_time, …)`;
- slice `is_missing` **themselves** — the caller passes the full mask, and the
  model layer applies `is_missing[start:stop]` at `:3237` / `:4198`. This is what
  the earlier draft got wrong.
- slice anything else indexed by time: the non-local position penalty
  (`_compute_non_local_position_penalty`) and the local-position kernel
  (`_compute_local_position_kernel`), both of which return per-time arrays.

### 3. Use it from both core paths

At `core.py:618-622` and `core.py:1221-1225`, pass the full `time` and the row
range. Do **not** invent a new kwarg on the generic `log_likelihood_func`
parameter — `core` accepts arbitrary callables, and adding a required kwarg breaks
them. Instead have `models/base.py` bind it before handing the callable to core:

```python
        # base.py, where log_likelihood_func is constructed for core
        log_likelihood_func = functools.partial(
            self.compute_log_likelihood, ...
        )
        # core calls: log_likelihood_func(time, *args, is_missing=..., row_slice=...)
```

and give core a capability check so a user-supplied callable without `row_slice`
still works:

```python
    supports_row_slice = "row_slice" in inspect.signature(log_likelihood_func).parameters
```

falling back to the current sliced-time behaviour with a `logger.warning` naming
the boundary-spike limitation.

Assert chunks are contiguous once at the top of the loop
(`np.array_equal(time_inds_np, np.arange(start, stop))`) rather than assuming it.

### 4. Restore `return_outputs="log_likelihood"` for chunked runs

With caching disabled, multi-chunk prediction returns no likelihood. Accumulate
the per-chunk arrays and assign `self.log_likelihood_` after the forward pass,
guarded by whether the caller asked for it — do not always retain it, that would
defeat the memory purpose.

Update the comment at `models/base.py:1719-1721` to say the disable is a memory
trade-off, not a behavioural one. (The earlier draft titled this task "remove the
caching special case" and then said to keep it; it stays, with an accurate
comment.)

### 5. CHANGELOG

`Fixed`: prediction with `n_chunks > 1` dropped spikes falling between adjacent
chunks — one per boundary per unit — so chunked and unchunked results differed.
`return_outputs="log_likelihood"` now works with `n_chunks > 1`.

## Validation

New `tests/integration/test_chunk_parity.py`. **Both core paths must be covered**
— they are stationary (`core.py:490`) vs. covariate-dependent (`core.py:1081`),
*not* sorted vs. clusterless, so parametrize over transition model as well as
detector type:

| Test | Asserts |
|---|---|
| `test_chunked_matches_unchunked[stationary-sorted]` (slow) | `n_chunks=1` vs `4` acausal posteriors equal, `rtol=1e-5`. Fixture places ≥1 spike strictly between `time[k-1]` and `time[k]` for every boundary `k`, else the test is vacuous. |
| `test_chunked_matches_unchunked[stationary-clusterless]` (slow) | Same. |
| `test_chunked_matches_unchunked[covariate-sorted]` (slow) | Same, routing through `chunked_filter_smoother_covariate_dependent`. |
| `test_chunked_matches_unchunked[covariate-clusterless]` (slow) | Same. |
| `test_chunked_log_likelihood_returned` | `return_outputs="log_likelihood"` with `n_chunks=4` returns an `(n_time, n_state_bins)` array equal to the unchunked one. |
| `test_row_slice_is_a_slice_of_full` | For each registry backend, `predict(..., row_slice=(a,b))` equals `predict(...)[a:b]` exactly. Parametrized over both registries. |
| `test_callable_without_row_slice_still_runs` | A user-supplied likelihood callable lacking `row_slice` runs and warns. |

### Memory evidence (required)

This phase exists to preserve a memory property, so assert it:

| Test | Asserts |
|---|---|
| `test_chunking_bounds_peak_memory` (slow) | Peak device allocation during a `n_chunks=8` predict is below that of `n_chunks=1` by a factor > 4 on a fixture sized so the full likelihood is ~8× a chunk. Measure with `jax.live_arrays()` sampling or `memory_analysis()`; state the method in the test docstring. |

If this test cannot be made to pass, the design has not achieved its purpose —
stop and report rather than shipping.

```bash
uv run pytest src/non_local_detector/tests/integration src/non_local_detector/tests/core -q
```

Goldens unaffected (they use the cached path).

## Review

Dispatch `code-reviewer`. Ask specifically whether `row_slice` is honoured on
every return path of both `compute_log_likelihood` implementations — including
`is_missing`, the non-local penalty, and the local-position kernel — and whether
any backend slices before binning rather than after.
