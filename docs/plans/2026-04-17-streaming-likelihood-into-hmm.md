# Streaming Likelihood + Memory-Aware Auto Plan (v3)

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

> **Plan history:**
> - **v1 (2026-04-17)**: framed streaming as a per-chunk cache
>   optimization. Back-of-envelope analysis + user feedback showed
>   per-chunk encoder work is <0.003% of per-chunk total at real
>   scale.
> - **v2 (2026-04-20, Tasks 1–3 implemented)**: pivoted to memory-aware
>   `n_chunks="auto"` heuristic that models the likelihood-slab
>   memory. Task 4 validation discovered the ~**8× peak / slab**
>   empirical ratio at 22-tet/2D HPC — meaning our single-knob
>   likelihood-slab-only heuristic silently under-chunks and users still
>   OOM.
> - **v3 (2026-04-20)**: user wants the whole memory process figured
>   out for the user. Extend to multi-knob auto: co-select
>   `n_chunks`, `block_size`, `enc_tile_size`, `pos_tile_size` such
>   that estimated peak ≤ device memory × safety. Tasks 1–3 stay as
>   foundations; Tasks 4–8 are new work.

**Goal:** `detector.predict(...)` never OOMs on user-visible workloads.
Users pass no memory knobs. The detector inspects device memory + the
workload shape, picks every memory knob, and runs. For chronic
recordings on small GPUs: it chunks aggressively. For typical
workloads on a big GPU: it picks the fast-path configuration (1 chunk,
large block size, no tiling). For everything in between: it picks the
least-tuned configuration that still fits.

**Branch:** `streaming-likelihood-hmm` off `main`. Tasks 1–3 already
landed; Tasks 4–8 continue here.

**Tech Stack:** JAX, NumPy, pytest.

---

## User problems this solves

| Problem | Current state (post–Tasks 1–3) | After Level 3 |
|---|---|---|
| Users don't know what memory knobs exist | `n_chunks="auto"` picks likelihood chunks, but not block_size / tile sizes | `memory_budget="auto"` picks everything |
| Likelihood-slab not the bottleneck at typical 22-tet/2D | auto under-chunks (slab fits budget but peak 8× slab OOMs) | peak model accounts for KDE intermediates, auto chunks correctly |
| Small GPUs (16–40 GB) on production workloads | OOMs mid-predict; users confused | auto tightens knobs, keeps peak ≤ budget |
| Chronic recordings (1 h to days) on any GPU | full-time likelihood slab exceeds memory | auto enables streaming + tighter tiles as recording length grows |
| Multi-state classifiers × dense grids | likelihood array grows linearly, OOMs | auto scales chunking with n_state_bins |

---

## The v2 finding that motivated v3

At 22-tetrode / 2D 3420 pos / ContFrag (2 states) on A100 80 GB (PR #19
validation artifact):

- Likelihood slab (ContFrag): `n_time × n_state_bins × 4B` = 709 321 ×
  1448 × 4 = **4.1 GB**
- Observed peak predict memory: **33 GB**
- Ratio: peak / slab ≈ **8×**

Sources of the other 29 GB: per-electrode KDE position distance
`(n_enc, n_pos) × 4B` ≈ 116 MB × live-at-a-time, per-electrode mark
kernel `(n_dec_chunk, n_enc) × 4B` ≈ 0.5–2 GB per electrode,
transition matrix, XLA autotuning workspace, fit-time intermediates
lingering across fit→predict boundary.

**Implication**: a heuristic that budgets only the likelihood slab
lets `safety_fraction = 0.40` through, but actual peak is ~8× bigger
and OOMs on anything smaller than the 80 GB A100. The v2 heuristic
gives a false sense of "fits". Users still OOM and still have to
figure out `block_size` / `enc_tile_size` by hand.

v3 fixes this by modeling all the knobs.

---

## What already works (foundation from Tasks 1–3)

Already committed on this branch:

- **Task 1** (`ad08c2a`): `streaming.py::_resolve_n_chunks(n_chunks, *, n_time, n_state_bins, memory_budget_bytes=..., safety_fraction=...)` — passthrough for int, auto-select for `"auto"`. 22 unit tests.
- **Task 2** (`410bad3`): `n_chunks: int | Literal["auto"] = "auto"` as the default on all 9 predict/estimate signatures in `base.py`. Backward-compat: explicit int still works.
- **Task 3** (`caf78d4`): `_guard_return_outputs_streaming` raises when resolved `n_chunks > 1` and user asked for `return_outputs="log_likelihood"`. 4 guard tests.

These remain valid. v3 extends (doesn't rewrite) them.

---

## Design: multi-knob memory-aware auto

### Knobs and what they control

| Knob | Axis | What it bounds |
|---|---|---|
| `n_chunks` | time (decoding) | `(n_time/n_chunks, n_state_bins) × 4B` likelihood slab |
| `block_size` | decoding-spikes (per-electrode, per-chunk) | `(block_size, n_pos) × 4B` intermediate in the decoding-spike fori_loop |
| `enc_tile_size` | encoding-spikes | `(enc_tile, n_pos) × 4B` for chunked logsumexp over encoding |
| `pos_tile_size` | position-bins | `(block_size, pos_tile) × 4B` tiles per position |

Ordering of preference (fast → slow):
1. `n_chunks=1`, `block_size` large, no tiling (fastest; single GPU compile, single chunk).
2. `n_chunks>1`: time-axis chunking (modest overhead from chunk dispatch + per-chunk compile cache).
3. `enc_tile_size` set: encoding-chunked path (slower, does online logsumexp).
4. `pos_tile_size` set: position-tiled path (slowest, extra Python iterations).

Auto prefers knobs in that order: increase `n_chunks` before reducing `block_size`; only enable tiling when chunking alone doesn't fit.

### Memory model (per-algorithm)

Each likelihood module exposes:

```python
def _estimate_predict_peak_bytes(
    *,
    n_time: int,
    n_state_bins: int,
    n_pos: int,
    encoding_model: dict,
    dec_spike_counts: list[int],  # per-electrode, total over n_time
    n_waveform_features: int,
    n_chunks: int,
    block_size: int,
    enc_tile_size: int | None,
    pos_tile_size: int | None,
    dtype_bytes: int = 4,
) -> int:
    """Return estimated peak bytes for a predict call with these knobs."""
```

For `clusterless_kde` (and `_log` variant) the model is:

```
likelihood_slab = ceil(n_time / n_chunks) * n_state_bins * dtype_bytes

# Per-electrode, per-chunk live allocations (max one electrode at a time):
enc_live = n_enc  # per electrode
pos_dim = pos_tile_size or n_pos

# log_position_distance (or chunk if enc_tile_size):
position_distance = (enc_tile_size or enc_live) * pos_dim * dtype_bytes

# Mark kernel for the decoding-spike block:
n_dec_per_chunk_per_electrode = ceil(max(dec_spike_counts) / n_chunks)
block_peak = block_size * (enc_tile_size or enc_live) * dtype_bytes
mark_kernel_per_block = block_peak
block_output = block_size * pos_dim * dtype_bytes

per_electrode_peak = position_distance + mark_kernel_per_block + block_output

# Per-chunk accumulator (persists across electrodes):
per_chunk_output = ceil(n_time / n_chunks) * n_pos * dtype_bytes

# Fixed overheads:
transition = n_state_bins ** 2 * dtype_bytes
fixed_scratch = 1 * 2**30  # 1 GB XLA autotuning + transients (empirical)

peak = max(
    likelihood_slab + per_chunk_output + per_electrode_peak,
    per_chunk_output * 2,  # build-next-chunk while filter consumes current
) + transition + fixed_scratch
```

Constants (`fixed_scratch = 1 GB`) are empirical, refined via Task 6 real-data measurement.

For `sorted_spikes_kde` / `sorted_spikes_glm` the model is simpler (no mark kernel). For `clusterless_gmm` it's different (GMM eval instead of KDE). Each module implements its own.

### Heuristic: multi-knob selector

```python
def auto_select_memory_knobs(
    *,
    workload: WorkloadInfo,
    memory_budget_bytes: int,
    safety_fraction: float = 0.90,  # use up to 90% of the budget; 10% headroom
    peak_estimator: Callable[..., int],
) -> dict:
    """Pick (n_chunks, block_size, enc_tile_size, pos_tile_size).

    Preference order: try least-tuned first, escalate only when peak
    exceeds budget.  Safety_fraction stays at 0.90 because the peak
    estimator already accounts for algorithm-specific overhead — the
    safety margin is just for model imprecision, not for hidden
    multipliers.
    """
    effective_budget = int(memory_budget_bytes * safety_fraction)

    # Stage 1: unchunked, big block, no tiling
    for n_chunks in _chunk_ladder(workload.n_time):  # [1, 2, 4, 8, 16, ...]
        for block_size in _block_ladder():  # [10000, 1000, 100]
            peak = peak_estimator(
                **workload.asdict(),
                n_chunks=n_chunks,
                block_size=block_size,
                enc_tile_size=None,
                pos_tile_size=None,
            )
            if peak <= effective_budget:
                return {
                    "n_chunks": n_chunks,
                    "block_size": block_size,
                    "enc_tile_size": None,
                    "pos_tile_size": None,
                }

    # Stage 2: enable encoding tiling
    for n_chunks in _chunk_ladder(workload.n_time):
        for enc_tile in _enc_tile_ladder(workload):
            peak = peak_estimator(
                **workload.asdict(),
                n_chunks=n_chunks,
                block_size=1000,
                enc_tile_size=enc_tile,
                pos_tile_size=None,
            )
            if peak <= effective_budget:
                return {"n_chunks": n_chunks, "block_size": 1000, "enc_tile_size": enc_tile, "pos_tile_size": None}

    # Stage 3: position tiling (slowest, last resort)
    ...

    raise RuntimeError(
        f"Workload does not fit in {memory_budget_bytes / 2**30:.1f} GB even "
        f"with max chunking + tiling.  Reduce n_state_bins / n_pos / "
        f"encoding cloud size."
    )
```

This is greedy but predictable. The ladders (`_chunk_ladder`, `_block_ladder`, `_enc_tile_ladder`) are bounded to avoid pathological small sizes (e.g. `block_size < 100` is launch-overhead-bound on GPU, rejected).

### API surface

```python
detector.predict(
    ...,
    memory_budget: int | Literal["auto"] | None = "auto",
    # Existing knobs stay as explicit overrides:
    n_chunks: int | Literal["auto"] = "auto",
    # block_size / enc_tile_size / pos_tile_size live on
    # clusterless_algorithm_params dict at the detector level
)
```

- `memory_budget="auto"`: query JAX device memory, run the heuristic.
- `memory_budget=<int>`: use that many bytes (for explicit control on shared GPUs, or to simulate smaller memory).
- `memory_budget=None`: disable auto-selection; use existing explicit knobs (backward-compat).
- Explicit `n_chunks` / `block_size` / etc. override the auto-picked values when provided.

When auto-picked knobs differ from current explicit defaults, log the resolved configuration at INFO level:

```
Auto-selected memory knobs: n_chunks=3, block_size=1000, enc_tile_size=None, pos_tile_size=None (budget=8.0 GB, estimated peak=7.2 GB).
```

---

## Verification Strategy

### Unit-level (CPU, fast)

1. **`_estimate_predict_peak_bytes` tests**: given synthetic workloads, peak estimator returns plausible byte counts that scale with inputs as expected (doubling `n_time` at fixed `n_chunks` doubles the slab; doubling `n_chunks` halves it; etc.).
2. **`auto_select_memory_knobs` tests**: given a peak-estimator + budget, heuristic picks config whose estimated peak ≤ budget × safety. Tiny budgets force tiling. Huge budgets return fast-path. No budget → raise.
3. **Chunked-parity tests** (already have from v2): `predict(n_chunks=1)` == `predict(n_chunks=K)` on sorted-spikes + clusterless to 1e-4.
4. **Memory-budget equivalence**: `predict(memory_budget=<big>)` == `predict(n_chunks=1)` (fast-path resolution) to 1e-4.
5. **Auto knob override precedence**: explicit `n_chunks=3` with `memory_budget="auto"` keeps `n_chunks=3` and auto-picks only the remaining knobs.

### Real-data GPU (A100, simulated memory caps via `XLA_PYTHON_CLIENT_MEM_FRACTION`)

6. **80 GB baseline, auto**: matches `n_chunks=1` explicit (no chunking, fast-path picked) to 1e-4.
7. **40 GB cap** (MEM_FRACTION=0.50): auto picks n_chunks > 1 (since 80 GB / 40 GB = 2×), predict succeeds, output matches baseline.
8. **24 GB cap** (MEM_FRACTION=0.30): auto picks more aggressive knobs (n_chunks + possibly block_size), predict succeeds, output matches baseline.
9. **12 GB cap** (MEM_FRACTION=0.15): auto enables encoding tiling, predict succeeds, output matches baseline.
10. **8 GB cap** (MEM_FRACTION=0.10): stress case — auto tightens everything, may still fail if fit's non-chunked KDEModel eval OOMs. If it does, documented as a known limit (fit-time memory is a separate v4 target).

### Chronic-scale synthetic demo (Task 8)

11. Synthetic 1-hour recording by concatenating/repeating the HPC 20-min epoch. Auto at simulated 16 GB: picks `n_chunks=5+` and succeeds. At `memory_budget=None` (auto disabled): OOM on 16 GB. User-facing deliverable.

### Compile-time (from PR #19 post-mortem)

12. Every benchmark above: first-predict compile time ≤ 20 min. No scan path in this plan (we revert-preserved that ban), so ptxas pathology risk is low, but tracked as a regression gate.

---

## Tasks

### Task 1: `_resolve_n_chunks` heuristic + unit tests ✅ (already done, `ad08c2a`)

### Task 2: Flip `n_chunks` default to `"auto"` on detector.predict signatures ✅ (already done, `410bad3`)

### Task 3: `return_outputs` + streaming guard ✅ (already done, `caf78d4`)

### Task 4: Per-algorithm `_estimate_predict_peak_bytes` functions

For each non-local likelihood module (`sorted_spikes_kde`, `sorted_spikes_glm`, `clusterless_kde`, `clusterless_kde_log`, `clusterless_gmm`), add:

```python
def _estimate_predict_peak_bytes(
    *, n_time, n_state_bins, n_pos, encoding_model, dec_spike_counts,
    n_waveform_features, n_chunks, block_size,
    enc_tile_size=None, pos_tile_size=None, dtype_bytes=4,
) -> int:
    """Return estimated peak bytes for a predict call."""
```

Model derivation goes in the docstring. Constants (e.g. `fixed_scratch`) start conservative and are tightened via Task 6 measurement.

**Files:**
- `src/non_local_detector/likelihoods/sorted_spikes_kde.py`
- `src/non_local_detector/likelihoods/sorted_spikes_glm.py`
- `src/non_local_detector/likelihoods/clusterless_kde.py`
- `src/non_local_detector/likelihoods/clusterless_kde_log.py`
- `src/non_local_detector/likelihoods/clusterless_gmm.py`
- `src/non_local_detector/tests/test_memory_model.py` (new)

### Task 5: `auto_select_memory_knobs` heuristic

```python
# src/non_local_detector/streaming.py
def auto_select_memory_knobs(workload_info, memory_budget_bytes, *, peak_estimator, safety_fraction=0.90) -> dict:
    ...
```

- Ladder helpers: `_chunk_ladder(n_time)`, `_block_ladder()`, `_enc_tile_ladder(...)`.
- Greedy search over (stage_1 → stage_2 → stage_3) preferring fast configs.
- Returns `dict` of picked knobs.
- `RuntimeError` when no config fits.

**Tests:**
- Given synthetic `peak_estimator` (callable returning bytes), verify the returned config has estimated_peak ≤ budget × safety.
- Verify stage escalation: huge-budget → stage 1; medium → stage 2; tiny → stage 3.
- Verify the "no config fits" case raises.

### Task 6: Real-data calibration of memory model constants

Run `detector.predict(n_chunks=1, block_size=1000)` at full 80 GB on the reference workload (22-tet / 2D / ContFrag), capture `jax.devices()[0].memory_stats()["peak_bytes_in_use"]`. Repeat for `n_chunks=2, 4, 8`. Fit/verify the model's constants.

Write the calibration results to `docs/benchmarks/streaming-memory-model.md` with the measured datapoints so future tuning is data-driven.

**Files:**
- `/cumulus/edeno/state-space-playground/scripts/benchmark_memory_model.py` (new)
- `docs/benchmarks/streaming-memory-model.md` (new)
- Model constants updated in Task 4's functions.

### Task 7: Wire `memory_budget` into detector.predict

Replace the current `n_chunks="auto"` resolution with a broader `memory_budget="auto"` resolution that picks all knobs at once.

```python
# In predict():
if memory_budget == "auto":
    memory_budget = _query_device_memory_bytes() or None
if memory_budget is not None:
    auto_knobs = auto_select_memory_knobs(
        workload=..., memory_budget_bytes=memory_budget,
        peak_estimator=<per-algorithm estimator>,
    )
    # Apply auto-selected knobs, respecting any explicit user overrides.
    n_chunks = n_chunks if isinstance(n_chunks, int) else auto_knobs["n_chunks"]
    # Pass block_size / enc_tile_size / pos_tile_size into clusterless_algorithm_params.
```

**Files:**
- `src/non_local_detector/models/base.py`
- `src/non_local_detector/tests/test_streaming_predict.py` (extend)

### Task 8: Real-data validation at multiple memory caps

Matrix: (80 GB baseline, 40 GB, 24 GB, 12 GB, 8 GB) × (ContFrag, NonLocal) × (clusterless_kde, clusterless_kde_log) × (`memory_budget="auto"`, `memory_budget=None`).

Expected outcomes (updated by Task 6 data):
- 80 GB + auto: picks fast-path, matches `n_chunks=1` baseline to 1e-4.
- 40 GB + auto: picks chunked, succeeds, matches baseline.
- 24 GB + auto: may enable tiling, succeeds, matches baseline.
- 12 GB + auto: aggressive knobs, succeeds on moderate workloads.
- 8 GB + auto: may still fail on 22-tet/2D due to fit-time OOM (documented limit).

Commit the benchmark run log + summary JSONs to `docs/benchmarks/` or attach to the PR.

### Task 9 (optional): Chronic-recording synthetic demo

Concatenate the 20-min HPC epoch 3× to produce a 1-hour equivalent. Show:

- At 16 GB simulated + `memory_budget=None`: OOM.
- At 16 GB simulated + `memory_budget="auto"`: succeeds, picks n_chunks ≥ 5 (likelihood slab scales with time).
- Posteriors consistent with the 20-min baseline on each 1/3rd segment.

This is the user-visible deliverable.

---

## Execution Order

```
Task 1–3 ✅ done
    ↓
Task 4 (per-algorithm peak estimators)           ← 1-2 days
    ↓
Task 5 (auto_select_memory_knobs heuristic)      ← 1 day
    ↓
Task 6 (real-data calibration of constants)      ← 0.5 day
    ↓
Task 7 (wire into detector.predict)              ← 0.5 day
    ↓
Task 8 (multi-cap real-data validation)          ← 0.5 day (GPU time dominates)
    ↓
Task 9 (optional chronic-scale synthetic demo)   ← 0.5 day
```

---

## Out of scope for this plan

- **Fit-time memory auto-selection.** Fit's KDEModel eval has its own
  block_size. Included as a separate follow-up when fit OOMs are observed.
- **Bringing back PR #19's scan-over-electrodes path.** That path has a
  known ptxas compile pathology (issue #21). Clusterless v2 redesign
  (`docs/plans/2026-04-20-clusterless-gpu-optimization-redesign.md`)
  addresses it.
- **GMM full support.** `clusterless_gmm` gets a trivial estimator
  (one that returns "n_time × n_state_bins × 4" + 20% slack) initially.
  Precise model deferred.

---

## Expected Impact (user-facing)

| Scenario | Current (v2, Tasks 1–3 only) | After Level 3 |
|---|---|---|
| 22-tet 2D ContFrag, 80 GB GPU | auto picks 1 chunk; works | same; no regression |
| 22-tet 2D ContFrag, 40 GB GPU | auto picks 1 chunk; OOM on KDE intermediates | auto picks 2 chunks + block_size; succeeds |
| 22-tet 2D ContFrag, 16 GB GPU | OOM | auto picks 4 chunks + block_size + enc_tile; succeeds |
| Chronic 1 h recording, 80 GB GPU | auto picks 1 chunk; works if likelihood slab fits | same |
| Chronic 1 h recording, 16 GB GPU | OOM immediately | auto picks 8+ chunks; succeeds |
| Multi-state classifier, 13 680 state bins | auto picks 1 chunk for moderate n_time; OOM on longer recordings | auto scales chunking with n_time × n_state_bins |

Users who never set memory knobs get a decoder that just works on their GPU. Users who do set knobs keep full control (auto yields to explicit values).

---

## Related

- Post-PR-19 validation workflow memory file: `.claude/.../real_data_pr_validation_workflow.md` — validation gates this plan must clear.
- PR #19 redesign plan (clusterless GPU v2): `docs/plans/2026-04-20-clusterless-gpu-optimization-redesign.md` — orthogonal; shares the "validate on real-data at multiple shapes" principle.
- Sorted-spikes GPU plan: `docs/plans/2026-04-17-sorted-spikes-gpu-optimization.md` — orthogonal.
- PR #14 (landed `2026-04-20`): `kde_distance` log-space rewrite — unifies the numerical internals of `clusterless_kde` and `clusterless_kde_log`, making their memory profiles identical, which simplifies per-algorithm memory models in Task 4.
