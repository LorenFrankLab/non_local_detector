# Streaming Likelihood Into HMM Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Make the existing `n_chunks > 1` streaming path efficient by eliminating redundant per-chunk encoder-side computation. The plumbing already works: `_predict` disables caching for `n_chunks > 1` (base.py:1351), and `chunked_filter_smoother` calls `log_likelihood_func(time[chunk_indices], ...)` per chunk (core.py:373). The likelihood functions already respect time slices. What's missing is caching encoder-side work (position distances, GEMM quantities, place fields) across chunks so each chunk only pays for its decoding-side work.

**Branch:** Execute on branch `streaming-likelihood-hmm` off `main`. Test on CPU locally, then validate on GPU hardware before merging.

**Tech Stack:** JAX, NumPy, pytest

**Why this matters:**

| Recording | n_time | n_state_bins | Full array size |
|---|---|---|---|
| 4ms / 15min | 225k | 200 | 172 MB |
| 2ms / 60min | 1.8M | 200 | 1.4 GB |
| 2ms / 60min | 1.8M | 500 | 3.4 GB |

With `n_chunks > 1`, this array is never materialized — each chunk produces `(chunk_size, n_state_bins)` which the filter consumes and discards. But without caching, each chunk redundantly recomputes encoder-side quantities (position distances, place fields, GEMM scaling) that are constant across all chunks.

---

## What Already Works

The streaming pipeline is already wired:

```
_predict(n_chunks=10):
    cache_likelihood = False  # base.py:1351-1353
    chunked_filter_smoother(..., cache_log_likelihoods=False):
        for chunk in time_chunks:
            ll_chunk = compute_log_likelihood(time[chunk_inds], ...)  # core.py:373
            filter(ll_chunk)  # consumes and discards
```

The likelihood functions (`predict_sorted_spikes_*`, `predict_clusterless_kde_*`) already accept partial time arrays and produce correctly-sized output. No new plumbing needed.

## What's Redundant Per Chunk

**Sorted spikes (non-local):** Each chunk call re-loops over all neurons, recomputing `get_spikecount_per_time_bin` and `xlogy`. With the vectorized path from the sorted spikes plan (sparse segment_sum or dense matmul), this becomes efficient — but `log(place_fields)` is still recomputed per call. Cache it.

**Clusterless KDE (non-local):** Each chunk call re-loops over all electrodes, each time:
- Recomputing `log_kde_distance(interior_bins, enc_positions, pos_std)` — O(n_enc × n_pos), constant across chunks
- Recomputing the GEMM scaling (inv_sigma, Y, y2) — O(n_enc × n_wf), constant across chunks
- Only the spike filtering and segment_sum are chunk-specific

**Clusterless GMM (non-local):** Each chunk call re-evaluates GMM models on all encoding data — same redundancy.

---

## Verification Strategy

1. **Accuracy (primary):** `predict(n_chunks=K)` matches `predict(n_chunks=1)` on realistic simulated data. Tolerance: 1e-4 (matching existing chunked-parity test tolerances in the codebase, not 1e-10 — floating-point reassociation from different chunking boundaries causes benign differences).
2. **Memory:** Peak allocation with `n_chunks=10` is ~10x smaller than `n_chunks=1` for the log-likelihood array.
3. **Performance:** Per-chunk time decreases after caching (the first chunk is slow, subsequent chunks skip encoder-side work).

---

## Task 1: Cache Encoding-Side Quantities Across Chunks

When `compute_log_likelihood` is called repeatedly (once per chunk in the streaming path), encoder-side work should be computed once and reused.

**For sorted spikes:** Cache `log(place_fields[:, is_track_interior])` and `no_spike_part[is_track_interior]` after fit. This is trivial — add to the encoding model dict in the fit functions. Already planned in sorted spikes plan Task 5.

**For clusterless KDE:** Cache per-electrode position distances and GEMM precomputed quantities. Already planned in clusterless plan Task 2. The position distance `log_kde_distance(interior_bins, enc_positions, pos_std)` depends only on encoding data and environment — compute once at the start of `predict`, not per chunk.

**Implementation approach:** Add a `_prepare_for_streaming` method to the detector classes that precomputes and caches these quantities at the start of `predict` when `n_chunks > 1`. Clear the cache after `predict` returns.

```python
# In _predict(), before calling chunked_filter_smoother:
if n_chunks > 1:
    self._prepare_streaming_cache(
        position_time, position, spike_times, spike_waveform_features, ...
    )
try:
    result = chunked_filter_smoother(...)
finally:
    self._clear_streaming_cache()
```

The `compute_log_likelihood` method checks for the cache and uses it when available.

**Files:**
- Modify: `src/non_local_detector/models/base.py` (add `_prepare_streaming_cache`, `_clear_streaming_cache`)
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_kde.py` (cache log_place_fields after fit)
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_glm.py` (same)
- Test: `src/non_local_detector/tests/test_streaming_predict.py` (new)

### Step 1: Write the test

```python
class TestStreamingCache:
    def test_clusterless_chunked_matches_unchunked(self, fitted_clusterless_detector):
        """predict(n_chunks=5) matches predict(n_chunks=1)."""
        result_1 = detector.predict(..., n_chunks=1)
        result_5 = detector.predict(..., n_chunks=5)
        np.testing.assert_allclose(
            result_1["acausal_posterior"],
            result_5["acausal_posterior"],
            atol=1e-4, rtol=1e-4,
        )

    def test_sorted_spikes_chunked_matches_unchunked(self, fitted_sorted_detector):
        """predict(n_chunks=5) matches predict(n_chunks=1)."""
        result_1 = detector.predict(..., n_chunks=1)
        result_5 = detector.predict(..., n_chunks=5)
        np.testing.assert_allclose(
            result_1["acausal_posterior"],
            result_5["acausal_posterior"],
            atol=1e-4, rtol=1e-4,
        )
```

### Step 2: Implement caching, test, profile, commit

---

## Task 2: Profile and Document Memory Savings

Measure peak memory with and without streaming on realistic data:

```python
# Without streaming (n_chunks=1): full (n_time, n_bins) materialized
result_1 = detector.predict(..., n_chunks=1)

# With streaming (n_chunks=10): only (chunk_size, n_bins) per chunk
result_10 = detector.predict(..., n_chunks=10)
```

Document the memory difference and add guidance to the docstring/README about when to use `n_chunks > 1`.

---

## Task 3: Verify `return_outputs` Interaction

When streaming (`n_chunks > 1`), the full log-likelihood array is never built. Verify that:
- `return_outputs=None` (default) — no log-likelihoods stored, no error
- `return_outputs="log_likelihood"` — log-likelihoods are `None` in results (or raise a clear error explaining they're unavailable with `n_chunks > 1`)
- `return_outputs="all"` — same behavior for log-likelihoods, other outputs present

**Files:**
- Possibly modify `_predict` to warn when `return_outputs` includes `"log_likelihood"` but `n_chunks > 1`
- Test: verify the interaction

---

## Execution Order and Dependencies

```
Sorted spikes plan Task 5 (cache log_place_fields)     ← prerequisite
Clusterless plan Task 2 (precompute GEMM)               ← prerequisite
    ↓
Task 1 (cache encoding-side across chunks)              ← main work
    ↓
Task 2 (profile memory savings)                         ← validation
    ↓
Task 3 (return_outputs interaction)                     ← polish
```

This plan is lightweight because the streaming infrastructure already exists. The work is caching encoder-side quantities so per-chunk calls don't redo constant work.

---

## Expected Impact

| Metric | n_chunks=1 (current default) | n_chunks=10 (streaming) |
|---|---|---|
| Log-likelihood peak memory | `n_time × n_bins × 4` bytes | `chunk_size × n_bins × 4` bytes |
| Encoder-side compute | 1× (computed once) | 1× (cached after first chunk) |
| Decoder-side compute | Same | Same (each time bin computed once) |
| Per-chunk overhead | N/A | Function call + cache lookup (~1ms) |
