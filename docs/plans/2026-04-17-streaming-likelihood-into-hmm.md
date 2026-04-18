# Streaming Likelihood Into HMM Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the full `(n_time, n_state_bins)` log-likelihood array from the critical path by streaming likelihood computation directly into the HMM's chunked filter. Instead of computing all likelihoods first and then filtering, compute likelihoods per time chunk and feed them to the filter immediately, discarding each chunk after use.

**Architecture:** The infrastructure already exists: `core.chunked_filter_smoother` supports on-demand likelihood generation via `cache_log_likelihoods=False` (core.py:349-382). When uncached, it calls `log_likelihood_func(time[chunk_indices], ...)` per chunk. The problem: the current `compute_log_likelihood` methods in `base.py` compute the ENTIRE `(n_time, n_state_bins)` array regardless of the time slice passed — they internally loop over all electrodes/neurons and all spike times. The fix is making the likelihood functions time-slice-aware so they only compute the requested chunk.

**Branch:** Execute on branch `streaming-likelihood-hmm` off `main`. Test on CPU locally, then validate on GPU hardware before merging.

**Tech Stack:** JAX, NumPy, pytest

**Why this matters:**

| Recording | n_time | n_state_bins | Full array size |
|---|---|---|---|
| 4ms / 15min | 225k | 200 | 172 MB |
| 2ms / 60min | 1.8M | 200 | 1.4 GB |
| 2ms / 60min | 500 | 3.4 GB |

This array is the single largest allocation in the pipeline. With streaming, peak memory drops to `chunk_size × n_state_bins × 4 bytes` — e.g., 10k × 200 × 4 = 8 MB per chunk.

---

## Verification Strategy

1. **Accuracy (primary):** End-to-end reference-equality test: run the full decoder (`predict`) with streaming vs cached, compare posteriors. Tolerance: 1e-10 (structural refactor, same computation in different order).
2. **Memory:** Measure peak allocation with and without streaming on a large synthetic dataset.
3. **Correctness of time slicing:** Each likelihood function, when called with a time subset, produces the same result as slicing the full output.

---

## Current Data Flow

```
predict() → _predict() → chunked_filter_smoother()
                              ↓
                    if cache_log_likelihoods:
                        log_likelihoods = compute_log_likelihood(ALL time)  ← FULL ARRAY
                        for chunk: ll_chunk = log_likelihoods[chunk_indices]
                    else:
                        for chunk: ll_chunk = compute_log_likelihood(chunk_time)  ← PER CHUNK
```

The `else` branch already works at the `core.py` level. The issue is upstream: `compute_log_likelihood` in `base.py` doesn't efficiently handle partial time ranges. It computes likelihoods for ALL observation models and ALL electrodes/neurons regardless of the time slice.

---

## Task 1: Make Sorted Spikes Likelihood Time-Slice-Efficient

The sorted spikes likelihood functions (`predict_sorted_spikes_kde_log_likelihood`, `predict_sorted_spikes_glm_log_likelihood`) accept a `time` parameter that already determines the output range. When called with a time slice, the functions should only count spikes and compute likelihoods for that slice.

**Current behavior:** The functions already work correctly with a time slice — `get_spikecount_per_time_bin(neuron_spike_times, time)` only counts spikes within the given `time` edges. The output shape is `(len(time), n_bins)`. No code changes needed for correctness.

**Performance issue:** The neuron loop iterates over ALL neurons for each time chunk call. With the vectorized approach from the sorted spikes plan (sparse segment_sum or dense matmul), each call is a single operation regardless. But with the current neuron loop, calling `compute_log_likelihood` per chunk means `n_chunks × n_neurons` Python loop iterations instead of `n_neurons`. This is why the sorted spikes plan (Tasks 1-2) should be implemented BEFORE this streaming plan — the vectorized helpers make per-chunk calls efficient.

**Files:**
- No changes to sorted spikes likelihood files (already slice-compatible)
- Test: verify that `predict_sorted_spikes_*_log_likelihood(time_slice, ...)` produces the same result as `predict_sorted_spikes_*_log_likelihood(full_time, ...)[slice_indices]`

### Step 1: Write the slice-equivalence test

```python
class TestSortedSpikesTimeSlicing:
    """Verify sorted spikes likelihood is slice-equivalent."""

    def test_kde_nonlocal_slice_matches_full(self, kde_fixture):
        """Computing on a time slice matches slicing the full result."""
        full_result = predict_sorted_spikes_kde_log_likelihood(
            time=full_time, ..., is_local=False,
        )
        # Take a chunk from the middle
        chunk_start, chunk_end = 100, 300
        chunk_time = full_time[chunk_start:chunk_end]
        chunk_result = predict_sorted_spikes_kde_log_likelihood(
            time=chunk_time, ..., is_local=False,
        )
        # Should match the corresponding slice of the full result
        np.testing.assert_allclose(
            chunk_result, full_result[chunk_start:chunk_end],
            atol=1e-10, rtol=1e-10,
        )
```

**Important edge case:** The time array for sorted spikes is bin edges, so slicing needs care — `time[chunk_start:chunk_end]` gives `chunk_end - chunk_start` edges = `chunk_end - chunk_start` time bins (since `get_spikecount_per_time_bin` uses `time[1:-1]` as internal edges). Verify the boundary behavior.

### Step 2: Run test, confirm it passes without code changes

### Step 3: Commit test

---

## Task 2: Make Clusterless Likelihood Time-Slice-Efficient

The clusterless likelihood functions (`predict_clusterless_kde_log_likelihood`, `predict_clusterless_kde_log_log_likelihood`, `predict_clusterless_gmm_log_likelihood`) are more complex because they involve per-electrode spike filtering, position interpolation, and segment_sum — all of which depend on the time range.

**Current behavior:** These functions accept `time` and filter spikes to `[time[0], time[-1]]`. When called with a time slice, they correctly produce output only for that slice. The segment_sum uses `num_segments=len(time)`.

**Performance issue:** Each call recomputes the position distance matrix (`log_kde_distance`), the mark kernel, and the marginal for ALL encoding spikes even when decoding only a small time chunk. The encoding-side computation is independent of the time slice — it depends on encoding data (fixed) and position bins (fixed). Only the decoding-side (which spikes fall in the time chunk) changes.

**Optimization:** Cache encoding-side quantities (position distance, GEMM precompute) across time chunks. This is already planned in the clusterless plan (Task 2: precompute GEMM quantities, and the position distance is per-electrode constant). With those in place, per-chunk calls only recompute the decoding-side work.

**Files:**
- Minimal changes to clusterless likelihood functions — they already accept time slices
- The clusterless plan's Task 2 (GEMM precompute) should be implemented first
- Test: slice-equivalence test same as Task 1

### Step 1: Write slice-equivalence test for clusterless

```python
class TestClusterlessTimeSlicing:
    def test_kde_log_nonlocal_slice_matches_full(self, clusterless_fixture):
        full_result = predict_clusterless_kde_log_likelihood(
            time=full_time, ..., is_local=False,
        )
        chunk_time = full_time[100:300]
        chunk_result = predict_clusterless_kde_log_likelihood(
            time=chunk_time, ..., is_local=False,
        )
        np.testing.assert_allclose(
            chunk_result, full_result[100:300], atol=1e-4, rtol=1e-4,
        )
```

**Note:** The clusterless tolerance is 1e-4 (not 1e-10) because the compensated-linear path may produce slightly different results when processing different spike subsets due to different global_max values per chunk.

### Step 2: Run test, verify it passes

### Step 3: Commit

---

## Task 3: Make `compute_log_likelihood` Cache Encoding-Side Work

The `compute_log_likelihood` method in `base.py` is called once per HMM chunk when `cache_log_likelihoods=False`. Currently it recomputes everything from scratch each time. The encoding-side work (position distances, place fields, GEMM quantities) is constant across chunks — only the decoding-side (which spikes fall in each time chunk) varies.

**Approach:** On first call, cache encoding-side quantities. On subsequent calls (same predict session), reuse them. This avoids redundant O(n_enc × n_pos) work per chunk.

**For sorted spikes:** Place fields are already precomputed at fit time and stored in `self.encoding_model_`. `log(place_fields)` can be cached after fit (sorted spikes plan Task 5). No per-predict caching needed.

**For clusterless:** The position distance matrix and GEMM precompute depend on encoding data (fixed per electrode). Cache them after the first `compute_log_likelihood` call in a predict session:

```python
def compute_log_likelihood(self, time, ...):
    # Lazily compute and cache encoding-side quantities
    if not hasattr(self, '_cached_encoding_quantities'):
        self._cached_encoding_quantities = {}
        for electrode_idx, encoding_model in self.encoding_model_.items():
            # Cache position distance, GEMM precompute, etc.
            ...

    # Use cached quantities for this time chunk's computation
    ...
```

**Cleanup:** Clear the cache after `predict` returns (or at the start of each `predict` call) to avoid stale state.

**Files:**
- Modify: `src/non_local_detector/models/base.py` (`ClusterlessDetector.compute_log_likelihood`)
- Test: verify caching doesn't change results

### Step 1: Write test

```python
class TestComputeLogLikelihoodCaching:
    def test_cached_matches_uncached(self, fitted_detector, test_data):
        """Cached encoding quantities produce same result as uncached."""
        result_uncached = detector.compute_log_likelihood(
            time, position_time, position, spike_times, spike_waveform_features,
        )
        result_cached = detector.compute_log_likelihood(
            time, position_time, position, spike_times, spike_waveform_features,
        )
        np.testing.assert_allclose(result_uncached, result_cached, atol=1e-10)
```

### Step 2: Implement, test, commit

---

## Task 4: Wire Streaming Into `_predict`

Enable streaming by default when `n_chunks > 1`. The infrastructure is already there — `_predict` already sets `cache_likelihood = False` when `n_chunks > 1` (base.py:1351-1353). The chunked filter then calls `compute_log_likelihood` per chunk.

**Current behavior (base.py:1351-1353):**
```python
if n_chunks > 1 and cache_likelihood:
    logger.info("Disabling likelihood caching for chunked processing")
    cache_likelihood = False
```

This already works. The only issue is that `compute_log_likelihood` may be inefficient when called per chunk (Tasks 1-3 address this).

**What this task does:** After Tasks 1-3, verify that the full pipeline works end-to-end with streaming:

```python
# This should work with bounded memory:
results = detector.predict(
    spikes=spikes_test,
    time=time_test,
    position=position_test,
    n_chunks=10,  # triggers streaming
)
```

**Files:**
- Minimal changes — verify existing plumbing works
- Test: end-to-end predict with `n_chunks > 1` matches `n_chunks=1`

### Step 1: Write end-to-end streaming test

```python
class TestStreamingPredict:
    def test_chunked_matches_unchunked(self, fitted_detector, test_data):
        """predict(n_chunks=10) matches predict(n_chunks=1)."""
        result_1 = detector.predict(
            ..., n_chunks=1, return_outputs="all",
        )
        result_10 = detector.predict(
            ..., n_chunks=10, return_outputs="all",
        )
        np.testing.assert_allclose(
            result_1["acausal_posterior"],
            result_10["acausal_posterior"],
            atol=1e-10,
        )

    def test_chunked_sorted_spikes(self, fitted_sorted_detector, test_data):
        """Sorted spikes decoder works with streaming."""
        result_1 = detector.predict(..., n_chunks=1)
        result_5 = detector.predict(..., n_chunks=5)
        np.testing.assert_allclose(
            result_1["acausal_posterior"],
            result_5["acausal_posterior"],
            atol=1e-10,
        )
```

### Step 2: Run tests, profile memory

Key measurement: peak memory with `n_chunks=1` vs `n_chunks=10` on a large dataset. The chunked version should use ~10x less memory for the likelihood array.

### Step 3: Commit

---

## Task 5: Optionally Skip Returning Log-Likelihoods

When `cache_log_likelihoods=False`, the full log-likelihood array is never materialized. But the return signature of `chunked_filter_smoother` always includes `log_likelihoods` in position 5 of the return tuple. Currently it returns `None` when uncached.

The caller (`_predict`) stores this in the results xarray. If the user didn't request log-likelihoods via `return_outputs`, we can skip storing them entirely.

**Current behavior (core.py:452):**
```python
log_likelihoods,  # Keep as original (may be None or NumPy)
```

When streaming, this is `None`. The downstream code in `_predict` already handles `None` by not including it in results when `return_outputs` doesn't include `"log_likelihood"`.

**What this task does:** Verify the `return_outputs` parameter correctly prevents log-likelihood storage, and document the memory-saving interaction between `n_chunks > 1` and `return_outputs`.

**Files:**
- Possibly add documentation/logging
- Test: verify `predict(n_chunks=10, return_outputs=None)` doesn't store log-likelihoods

---

## Execution Order and Dependencies

```
Sorted spikes plan Task 1 (vectorize non-local)     ← prerequisite for efficient per-chunk calls
Clusterless plan Task 2 (precompute GEMM)            ← prerequisite for efficient per-chunk calls
    ↓
Task 1 (sorted spikes slice-equivalence test)        ← verify slicing works
Task 2 (clusterless slice-equivalence test)          ← verify slicing works
    ↓
Task 3 (cache encoding-side work)                    ← avoid redundant per-chunk computation
    ↓
Task 4 (end-to-end streaming test)                   ← verify full pipeline
    ↓
Task 5 (skip returning log-likelihoods)              ← polish
```

**Key dependency:** This plan assumes the sorted spikes vectorization (plan 2, Task 1) and clusterless GEMM precompute (plan 1, Task 2) are done first. Without those, per-chunk likelihood calls are inefficient (Python neuron/electrode loops run once per chunk instead of once total).

---

## Expected Impact

| Metric | Cached (current) | Streaming |
|---|---|---|
| Peak log-likelihood memory | `n_time × n_state_bins × 4` (1.4 GB at 2ms/60min/200 bins) | `chunk_size × n_state_bins × 4` (8 MB at 10k chunk) |
| Total compute | Same | Same (each time bin computed exactly once) |
| Overhead | None | Per-chunk function call overhead (~1ms per chunk) |
| When to use | Short recordings, user wants log-likelihoods | Long recordings, memory-constrained |

The streaming path is activated automatically when `n_chunks > 1`, which `_predict` already does. Users get the memory savings without changing their code — they just set `n_chunks` based on their memory budget.
