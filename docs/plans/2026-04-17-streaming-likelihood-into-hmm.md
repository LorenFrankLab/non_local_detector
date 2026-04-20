# Streaming Likelihood Into HMM Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

> **Plan audit (2026-04-20):** Updated for current `main` after the PR #19 revert
> (#22).  Line numbers verified; prerequisite framing removed (the original
> listed PR #19's and the sorted-spikes-gpu plan's internals as prereqs —
> PR #19 is reverted and the sorted plan hasn't started, so this plan
> stands on its own now).  Validation section expanded to match the
> workflow codified in `.claude/` memory post-PR-19.

**Goal:** Make the existing `n_chunks > 1` streaming path efficient by eliminating redundant per-chunk encoder-side computation. The plumbing already works: `_predict` disables caching for `n_chunks > 1` (`base.py:1351`), and `chunked_filter_smoother` (`core.py:253`) calls `log_likelihood_func(time[time_inds_np], ...)` per chunk (`core.py:376`). The likelihood functions already respect time slices. What's missing is caching encoder-side work (position distances, log-place-fields, etc.) across chunks so each chunk only pays for its decoding-side work.

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
_predict(n_chunks=10):              # base.py:1295
    cache_likelihood = False         # base.py:1351-1353 — forced off when n_chunks > 1
    chunked_filter_smoother(...,     # core.py:253
        cache_log_likelihoods=False):
        for time_inds in time_chunks:
            ll_chunk = log_likelihood_func(time[time_inds_np], ...)  # core.py:376
            filter(ll_chunk)          # consumes and discards
```

There is also `chunked_filter_smoother_covariate_dependent` (`core.py:753`) for covariate-dependent transitions — carries the same per-chunk redundancy story.  Task 1's cache must serve both dispatchers.

The likelihood functions — `predict_sorted_spikes_kde_log_likelihood` (`sorted_spikes_kde.py:237`), `predict_sorted_spikes_glm_log_likelihood` (`sorted_spikes_glm.py:334`), `predict_clusterless_kde_log_likelihood` (`clusterless_kde.py:296` and its log-module twin at `clusterless_kde_log.py:1319`) — already accept partial time arrays and produce correctly-sized output. No new plumbing needed.

## What's Redundant Per Chunk

**Sorted spikes (non-local):** Each chunk call re-loops over all neurons, recomputing `get_spikecount_per_time_bin` and `xlogy` terms.  Independent of any vectorization plan, `log(place_fields)` is recomputed per chunk.  Cache `log_place_fields[:, is_track_interior]` (and the no-spike Poisson baseline) at fit time or at first chunk — both are constant for a given encoding model and environment.

**Clusterless KDE (non-local):** Each chunk call re-loops over all electrodes, each time recomputing `log_kde_distance(interior_bins, enc_positions, pos_std)` — O(n_enc × n_pos), constant across chunks.  This is the biggest cache target.  (Note: PR #19 also precomputed the GEMM scaling (`inv_sigma`, `Y`, `y2`) inside the per-electrode path, but PR #19 was reverted via #22; those helpers no longer exist on `main`.  The streaming caching here does not depend on them — it targets the higher-level `log_kde_distance` output directly.)

**Clusterless GMM (non-local):** Each chunk call re-evaluates GMM models on all encoding data — same redundancy.  Defer: GMM is rarely used in production and isn't the priority target.

---

## Verification Strategy

1. **Accuracy — unit-test gate:** `predict(n_chunks=K)` matches `predict(n_chunks=1)` on realistic simulated data. Tolerance: 1e-4 absolute / 1e-4 relative (matching existing chunked-parity test tolerances in the codebase, not 1e-10 — floating-point reassociation from different chunking boundaries causes benign differences).  Run across both `clusterless_kde` and `clusterless_kde_log`, and across both `chunked_filter_smoother` and `chunked_filter_smoother_covariate_dependent`.
2. **Accuracy — real-data gate:** Run the PR #14 / PR #19 validation workflow (`state-space-playground/scripts/decode_hpc_for_branch_comparison.py`) on the 2D HPC session at n_chunks=1 AND n_chunks=10, and diff the posteriors.  Expected: max|diff| ≤ 1e-4 absolute, nan_mismatch = 0.  Also run cross-path parity (`kde` vs `kde_log` at same SHA) to confirm no regressions in either module.  See `.claude/` memory `real_data_pr_validation_workflow.md` for the full recipe.
3. **Memory:** Peak GPU allocation with `n_chunks=10` is ~10× smaller than `n_chunks=1` for the log-likelihood array on a 2D / 22-tetrode shape (expected: drop from ~380 MB to ~40 MB for the likelihood slab, verified via `jax.profiler.save_device_memory_profile`).
4. **Performance:** Per-chunk wall-clock time decreases after caching (the first chunk is slow — builds cache; subsequent chunks skip encoder-side work).  Report first-chunk vs steady-state-chunk timings in the validation artifact.
5. **Compile-time:** Report compile+first-call time on real-data shapes.  If it exceeds 20 min on any production shape, flag as a pathology per the PR #19 lesson.

---

## Task 1: Cache Encoding-Side Quantities Across Chunks

When the per-chunk `log_likelihood_func(time[time_inds_np], ...)` call inside `chunked_filter_smoother` fires repeatedly (once per chunk), encoder-side work should be computed once and reused.

**For sorted spikes:** Cache `log(place_fields[:, is_track_interior])` and the no-spike Poisson baseline after fit.  This is a two-line addition to the fit function's return dict (`fit_sorted_spikes_kde_encoding_model`, `fit_sorted_spikes_glm_encoding_model`) — the likelihood function then reads from the dict instead of recomputing.

**For clusterless KDE:** Cache per-electrode `log_kde_distance(interior_place_bin_centers, encoding_positions_i, position_std)` output — an `(n_electrodes, n_enc, n_pos)` stack.  This is the biggest cache target and dominates per-chunk redundant work.  Compute once at the start of `predict` (or at the first chunk call), store in an attribute the likelihood function can read.

**Implementation approach:**

Cache lives on the detector as `self._streaming_cache`, a dict keyed by algorithm name.  Populated lazily at the start of streaming `predict`, cleared in a `finally` block:

```python
# In _predict(), before calling chunked_filter_smoother when n_chunks > 1:
self._streaming_cache = None  # fresh slate
try:
    result = chunked_filter_smoother(..., log_likelihood_func=self._likelihood_with_cache, ...)
finally:
    self._streaming_cache = None  # release memory
```

The `_likelihood_with_cache` wrapper builds the cache on first call (when the arrays are needed and we have the full context) and passes the cached tensors as additional kwargs to the underlying likelihood function.  The underlying likelihood functions gain optional `precomputed_<name>` kwargs that, when provided, are used instead of recomputing.

**Backward-compatibility:** unchanged user-facing API.  The cache is an internal optimization; `precomputed_*` kwargs default to `None` and are only set by the streaming wrapper.  Direct callers of the likelihood function pass no cache and see no behavior change.

**Files:**
- Modify: `src/non_local_detector/models/base.py` — add `_streaming_cache` attribute and `_likelihood_with_cache` wrapper; route `chunked_filter_smoother` call through the wrapper when `n_chunks > 1`.
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_kde.py` — add `precomputed_log_place_fields` kwarg to predict fn.
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_glm.py` — same (GLM has no place_fields but does cache `log_place_fields` equivalent: the design matrix × coefficients eval).
- Modify: `src/non_local_detector/likelihoods/clusterless_kde.py` — add `precomputed_log_position_distances` kwarg to predict fn.
- Modify: `src/non_local_detector/likelihoods/clusterless_kde_log.py` — same.
- Test: `src/non_local_detector/tests/test_streaming_predict.py` — new file.

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
Task 1 (cache encoding-side across chunks)              ← main work
    ↓
Task 2 (profile memory savings)                         ← validation
    ↓
Task 3 (return_outputs interaction)                     ← polish
```

No prerequisites — the streaming infrastructure and likelihood-function time-slicing are already in place on current `main`.  The original plan listed PR #19's GEMM-precompute helpers and the sorted-spikes-GPU plan's `log_place_fields` cache as prerequisites; those are not in place (PR #19 is reverted; sorted plan not started), but streaming caching does not actually depend on them — it caches higher-level outputs (`log_kde_distance`, `log_place_fields`) that are recomputable today.

This plan is lightweight because the streaming infrastructure already exists. The work is caching encoder-side quantities so per-chunk calls don't redo constant work.

---

## Expected Impact

| Metric | n_chunks=1 (current default) | n_chunks=10 (streaming) |
|---|---|---|
| Log-likelihood peak memory | `n_time × n_bins × 4` bytes | `chunk_size × n_bins × 4` bytes |
| Encoder-side compute | 1× (computed once) | 1× (cached after first chunk) |
| Decoder-side compute | Same | Same (each time bin computed once) |
| Per-chunk overhead | N/A | Function call + cache lookup (~1ms) |
