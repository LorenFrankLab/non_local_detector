# Streaming Likelihood Into HMM Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

> **Plan history:** v1 (2026-04-17) framed streaming as a per-chunk
> redundant-work problem, with Task 1 caching encoder-side quantities
> across chunks. Post-mortem analysis (2026-04-20) showed that
> per-chunk encoder work is <0.003% of per-chunk total work at
> realistic scales — caching saves a rounding error. The real user
> problem is **memory**, not compute redundancy. This v2 rewrite
> pivots around memory-aware auto-chunking, default-on streaming for
> memory-constrained workloads, and enabling chronic recordings that
> currently don't run at all.

**Goal:** Make memory-constrained workloads work. Long recordings (1
hour to days), large state spaces (many discrete states × many
position bins), and smaller GPUs (16–40 GB) currently fail with OOM or
don't run at all. Streaming infrastructure already exists; the gap is
ergonomic and default behavior — users don't know about `n_chunks` and
the default `n_chunks=1` materializes the full `(n_time, n_state_bins)`
likelihood array, blowing memory on chronic recordings.

**Branch:** `streaming-likelihood-hmm` off `main`. CPU-first
implementation; simulate smaller-GPU memory budgets via
`XLA_PYTHON_CLIENT_MEM_FRACTION` for the "can we actually analyze
this now?" benchmark.

**Tech Stack:** JAX, NumPy, pytest

## User problems this solves

| Problem | Current symptom | After |
|---|---|---|
| Users don't know `n_chunks` exists or how to pick it | OOM or silently slow on chronic recordings | `n_chunks="auto"` picks based on device memory |
| Labs without A100 80 GB | Can't run typical workloads on 16–40 GB GPUs | Default auto-chunking keeps per-chunk likelihood under device memory |
| Multiple discrete states × many position bins | State-bin count explodes, full array OOMs | Per-chunk allocation stays bounded |
| Chronic recordings (1 h to days) | Literally not runnable today (likelihood array exceeds any single GPU) | First-class support; streaming enabled by default |

**Reference sizes** for a typical classifier:

| Recording | n_time | n_state_bins | Full likelihood (fp32) |
|---|---|---|---|
| 15 min @ 4 ms / 200 state-bins | 225 k | 200 | 180 MB |
| 60 min @ 2 ms / 500 state-bins | 1.8 M | 500 | 3.4 GB |
| 60 min @ 2 ms / 13 680 state-bins (4 states × 3420 pos bins, 2D) | 1.8 M | 13 680 | **98 GB** (OOM on 80 GB A100) |
| 2 h @ 2 ms / 13 680 state-bins | 3.6 M | 13 680 | **196 GB** (OOM anywhere) |

The first two fit unchunked on any GPU. Rows 3–4 are the target
workloads — currently unrunnable without manual `n_chunks` tuning.

---

## What Already Works

The streaming pipeline is wired end-to-end on current `main`:

```
_predict(n_chunks=10):                  # base.py:1295
    cache_likelihood = False             # base.py:1351 — forced off when n_chunks > 1
    chunked_filter_smoother(...,         # core.py:253
        cache_log_likelihoods=False):
        for time_inds in time_chunks:
            ll_chunk = log_likelihood_func(time[time_inds_np], ...)  # core.py:376
            filter(ll_chunk)              # consumes and discards
```

Verified in `tests/test_streaming_predict.py` (added 2026-04-20):
`detector.predict(n_chunks=5)` matches `detector.predict(n_chunks=1)`
to 1e-4 on both sorted-spikes and clusterless detectors, end-to-end.
No plumbing changes needed.

There is also `chunked_filter_smoother_covariate_dependent`
(`core.py:753`) for covariate-dependent transitions — same
infrastructure, serves the same role for classifiers with
covariate-dependent discrete transitions.

---

## Why encoder caching was dropped

The v1 plan's Task 1 proposed caching `log_kde_distance` and related
encoder-side quantities across chunks. Back-of-envelope on a realistic
clusterless workload (22 tetrodes, 20 k encoding spikes each, 3420
position bins, 71 k decoding time bins per chunk at `n_chunks=10`):

- Encoder work per chunk: O(n_enc × n_pos) ≈ 6.8 × 10⁷ ops
- Decoder work per chunk: O(n_dec_chunk × n_enc × n_pos) ≈ 3.0 × 10¹² ops
- Ratio: **~45 000× more decoder work than encoder work per chunk**

Even aggressive chunking (`n_chunks=100`) keeps the ratio in the
thousands. Caching the encoder saves a rounding error. The memory
savings (from not materializing the full likelihood) are the real win
and they happen automatically from chunking itself, with or without
the cache.

---

## Design

### 1. Memory-aware `n_chunks="auto"` selector

Accept `n_chunks: int | Literal["auto"] = "auto"` on
`detector.predict()`. When `"auto"`, compute:

```
budget_bytes = memory_budget_bytes or (0.40 * jax_device_bytes_limit)
n_state_bins = self.is_track_interior_state_bins_.sum()
per_time_bytes = n_state_bins * 4  # fp32 likelihood slab
chunk_size = max(1, budget_bytes // per_time_bytes)
n_chunks = ceil(n_time / chunk_size)
```

40% of device memory is a conservative default — leaves headroom for
the filter state, transition matrix, encoding model, and runtime
scratch. The explicit `memory_budget_bytes` kwarg on `predict()` lets
advanced users override.

**When `n_chunks="auto"` resolves to 1**: the full likelihood fits
comfortably; no chunking; behavior identical to current
`n_chunks=1`. Only workloads that actually need it pay the streaming
overhead (which is small: per-chunk compile once, then reused).

### 2. Default value change

Change the default on all four `predict()` signatures
(`NonLocalClusterlessDetector`, `NonLocalSortedSpikesDetector`,
`ContFragClusterlessClassifier`, `ContFragSortedSpikesClassifier`, etc.)
from `n_chunks: int = 1` to `n_chunks: int | Literal["auto"] = "auto"`.

**Backward compatibility**: users who pass `n_chunks=1` explicitly
keep current behavior. Users who pass no `n_chunks` (the common case)
now auto-select and never OOM on large workloads. Users who pass
`n_chunks=10` keep current behavior. This is a strictly
additive change for the default path.

### 3. `return_outputs` interaction

When streaming is on (resolved `n_chunks > 1`), the full
`(n_time, n_state_bins)` likelihood array is never materialized. If the
user asks for `return_outputs="log_likelihood"` or
`return_outputs="all"` (includes log-likelihood), we have a choice:

- **Option A**: silently materialize it anyway (defeats the point).
- **Option B**: raise a clear error asking the user to either drop
  the log-likelihood output or pass explicit `n_chunks=1`.
- **Option C**: accumulate per-chunk likelihoods into a single array
  even when streaming — costs memory but gives the user what they asked
  for; emit a warning.

Pick **B** for v2. Users who want the log-likelihood array are already
explicit about it; failing loud is better than silently using memory
the user was trying to avoid. Error message includes the auto-resolved
n_chunks and the explicit-override recipe.

### 4. Validation via simulated smaller-GPU memory

We don't have a chronic-recording dataset to test on, but we can
simulate memory pressure on the existing HPC session by capping JAX's
device memory via `XLA_PYTHON_CLIENT_MEM_FRACTION`:

```bash
# Simulate an 8 GB device on an 80 GB A100:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
    python scripts/decode_hpc_for_branch_comparison.py \
        --label memcap-8gb --mode 2d \
        --clusterless-algorithm clusterless_kde_log \
        --detector-class non_local --skip-sorted
```

The 2D HPC 22-tetrode NonLocal predict at `n_chunks=1` allocates
~33 GB peak (measured during PR #19 validation). Under `MEM_FRACTION=0.10`
(8 GB) this should OOM with `n_chunks=1`; with `n_chunks="auto"` the
heuristic should pick ~5 chunks and succeed with the same posterior.

That's the "does streaming actually fix the user problem" gate.

---

## Verification Strategy

### Unit-level (CPU)

1. **Chunked-parity tests** (already added at
   `tests/test_streaming_predict.py`) remain green: `predict(n_chunks=1)`,
   `predict(n_chunks=5)`, `predict(n_chunks="auto")` all agree to
   1e-4 on sorted-spikes and clusterless.
2. **`_resolve_n_chunks` heuristic tests**: given a fixed
   `memory_budget_bytes`, given `n_time` and `n_state_bins`, verify
   picked `n_chunks` keeps per-chunk bytes under budget.
3. **`return_outputs` interaction**: when streaming on +
   `log_likelihood` requested, raise the expected error with the
   expected message.

### Real-data GPU (A100, simulated memory caps)

4. **Budget threshold**: `n_chunks="auto"` with
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` (8 GB simulated) runs the 2D
   HPC session to completion where `n_chunks=1` OOMs. Posteriors match
   our previously-saved baseline (`main-post_2d_contfrag_clusterless_kde_*`
   or equivalent) to 1e-4.
5. **Budget unconstrained**: `n_chunks="auto"` at full 80 GB picks
   `n_chunks=1` (full array fits). Zero behavior change from current
   default.
6. **Chronic-scale simulation**: scale up the existing 20-min epoch
   artificially (stack 3× copies, synthetic extension) to produce a
   1-hour equivalent and show it now runs at `n_chunks="auto"` on a
   simulated 8 GB device.

### Compile-time (from PR #19 post-mortem)

7. Compile+first-predict on every benchmark above must complete in
   under 20 minutes. No scan-path shapes in this plan, so ptxas
   pathology risk is low, but track it anyway as a regression gate.

---

## Tasks

### Task 1: `n_chunks="auto"` heuristic + helper

**Goal:** implement `_resolve_n_chunks(n_chunks, *, n_time, n_state_bins, memory_budget_bytes=None) -> int`. Query device memory on "auto"; passthrough on int.

**Files:**
- New: `src/non_local_detector/core.py` — add `_resolve_n_chunks` helper near existing `chunked_filter_smoother` definition, or a small new module `src/non_local_detector/streaming.py` if cleaner.
- New: `src/non_local_detector/tests/test_streaming_predict.py` (expand existing file) — heuristic unit tests with synthetic device-memory sizes.

**Signature:**

```python
def _resolve_n_chunks(
    n_chunks: int | Literal["auto"],
    *,
    n_time: int,
    n_state_bins: int,
    memory_budget_bytes: int | None = None,
    dtype_bytes: int = 4,  # fp32
    safety_fraction: float = 0.40,
) -> int:
    """Pick n_chunks so each chunk's likelihood slab fits in budget.

    'auto' queries ``jax.devices()[0].memory_stats()["bytes_limit"]``
    and applies ``safety_fraction``.  Passthrough for int.  Validates
    n_chunks >= 1.
    """
```

Unit test matrix (no GPU required):

- `_resolve_n_chunks(1, ..., budget=...)` → 1 (passthrough)
- `_resolve_n_chunks("auto", n_time=1_000, n_state_bins=200, budget=10_000_000)` → 1 (fits)
- `_resolve_n_chunks("auto", n_time=1_800_000, n_state_bins=13_680, budget=8*2**30)` → ≥12 (chronic-scale)
- Reject `n_chunks=0`, negative, non-int/string
- Reject `budget <= 0`

### Task 2: Wire `"auto"` into detector.predict + flip defaults

**Goal:** all 4 detector classes accept `n_chunks: int | Literal["auto"] = "auto"`, resolve via Task 1's helper before dispatching to `chunked_filter_smoother`.

**Files:**
- Modify: `src/non_local_detector/models/base.py` — update `predict()` signatures in each detector class (~4 call sites, see `grep -n "def predict" src/non_local_detector/models/base.py`), add resolution step in `_predict()` (line ~1295) before `chunked_filter_smoother` call.
- Modify: `src/non_local_detector/models/non_local_model.py`,
  `cont_frag_model.py`, `multienvironment_model.py`, `decoder.py`,
  `nospike_cont_frag_model.py` — update signatures to match.
- Expand: `tests/test_streaming_predict.py` — add parity test for
  `n_chunks="auto"`.

### Task 3: `return_outputs` interaction

**Goal:** when the resolved `n_chunks > 1` and the user's
`return_outputs` includes `log_likelihood`, raise `ValidationError`
with a helpful message.

**Files:**
- Modify: `src/non_local_detector/models/base.py` — at the top of
  `_predict`, after resolving `n_chunks`, check `return_outputs` and
  raise. Message must include:
  - What was requested
  - What the resolved n_chunks is
  - How to override: `pass n_chunks=1 explicitly, or drop
    log_likelihood from return_outputs`
- Test: in `tests/test_streaming_predict.py`, assert the raise.

### Task 4: Real-data benchmark with simulated memory cap

**Goal:** produce the demonstration artifact that
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` workload succeeds with auto and
OOMs without it.

**Files:**
- Extend: `/cumulus/edeno/state-space-playground/scripts/decode_hpc_for_branch_comparison.py`
  to accept `--n-chunks {auto,1,N}` and `--memory-fraction` passthrough
  (already indirectly respects XLA env vars).
- New benchmark script or notebook documenting:
  - Baseline at 80 GB, `n_chunks=1`: works
  - Baseline at 80 GB, `n_chunks="auto"`: works, picks 1, identical output
  - Simulated 8 GB (`MEM_FRACTION=0.10`), `n_chunks=1`: OOM
  - Simulated 8 GB, `n_chunks="auto"`: succeeds, output matches 80 GB baseline
- Commit the benchmark log + summary JSONs to `docs/benchmarks/` or
  attach to the PR.

### Task 5 (optional): Chronic-recording simulated extension

If the user wants to demonstrate "we can now analyze 1-hour recordings
that couldn't run before", build a synthetic long-recording benchmark
by concatenating the 20-min HPC epoch with itself, time-shifted, to
make a 1-hour session. Show that the 1-hour decode:

- Doesn't run at `n_chunks=1` on simulated 8 GB
- Runs at `n_chunks="auto"` on simulated 8 GB
- Produces posteriors consistent with the 20-min baseline on each
  1/3rd segment (up to edge effects at stitching points)

This is the user-visible deliverable: "feature that didn't work, now
works." Skip if the 2D HPC 22-tet case at MEM_FRACTION=0.10 is
compelling enough on its own.

---

## Execution Order

```
Task 1 (heuristic helper + unit tests)         ← pure CPU, fast iteration
    ↓
Task 2 (wire "auto" into predict + flip defaults)  ← touches 5 detector files
    ↓
Task 3 (return_outputs guard)                  ← small, independent
    ↓
Task 4 (real-data benchmark w/ simulated memory cap)  ← GPU, the user gate
    ↓
Task 5 (optional chronic extension)            ← synthetic long-session demo
```

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| Default behavior on 22-tet 2D 1-hr workload | OOM on 16 GB, works on 80 GB | Works on both; auto-chunking transparent |
| User who passes no `n_chunks` on chronic 2-hr 4-state 2D | Silent OOM; user has to figure out chunking | Auto-chunks to fit; succeeds; emits info log |
| User who passes `n_chunks=1` explicitly | Current behavior | Current behavior (unchanged) |
| `return_outputs="all"` + auto picks >1 chunk | Would silently miss `log_likelihood` | Clear error: pass `n_chunks=1` or drop log_likelihood |
| Compile time on real-data 2D | 138 s for ContFrag at n_chunks=1 | Similar per-chunk compile, amortized over chunks — should be comparable |

## Related

- Post-PR-19 validation workflow memory file: `.claude/.../real_data_pr_validation_workflow.md` — validation gates this plan must clear.
- PR #19 redesign plan (clusterless GPU v2): `docs/plans/2026-04-20-clusterless-gpu-optimization-redesign.md` — orthogonal, but shares the "don't ship kernel optimizations without real-data validation on production shapes" principle.
- Sorted-spikes GPU plan: `docs/plans/2026-04-17-sorted-spikes-gpu-optimization.md` — orthogonal; streaming caching there can be added if measurement warrants.
