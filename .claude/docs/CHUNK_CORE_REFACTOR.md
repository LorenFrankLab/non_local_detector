
* JAX-accelerated forward/backward (filter + smoother) runs in **chunks** on device.
* Outputs (e.g., posteriors over `state_bins`) are ultimately wrapped in **xarray**.
* Your scale can be **time=1,976,441** and **state_bins=10,650** → one posterior is ~78.4 GiB (float32), so naive concatenation or eager expansion will OOM host RAM.

# Context (why we’re changing things)

* **Problem 1 — Device OOM risk:** large arrays on GPU/TPU; concatenation on device is not feasible.
  ✅ We already chunk compute; we’ll ensure we never build giant device arrays (write per-chunk to host).
* **Problem 2 — Host OOM risk:** even if device is fine, **final arrays** are tens to hundreds of GiB.
  ✅ We’ll write large outputs directly to **`np.memmap`** (disk-backed) and integrate **dask-backed, lazy expansion** into xarray so RAM stays ≈ one chunk.
* **Target workflow:** preserve current public API/variable names; add opt-in flags; keep numerics identical on small data; enable streaming NetCDF writes.

---

# Phase 0 — Groundwork (no behavior change)

## PR-0.1: Output allocator (feature-flagged)

**Goal:** Centralize array allocation; togglable memmap without changing defaults.

* Changes

  * Add `_alloc_out(shape, dtype, name, memmap_dir=None, mode="w+")` in `core.py`.
  * Thread kwargs through chunked functions:
    `use_memmap: bool = False`, `memmap_dir: str | None = None`.
* Tests

  * Shapes/dtypes correct for both branches (memmap vs empty).
  * Memmap persists slices across reopen.
* Acceptance

  * All current tests pass with `use_memmap` off (default).

## PR-0.2: Posterior storage dtype knob

**Goal:** Allow downcasting at **write-time** only.

* Changes

  * New arg `posterior_dtype: np.dtype = np.float32`.
    (Compute stays float32/bfloat16 on device; cast when writing to host.)
* Tests

  * Tiny fixture: `float16/uint16` write, compare to float32 baseline within tolerance.
* Acceptance

  * No behavior change with default; docs clarify trade-offs.

---

# Phase 1 — Streamed writes (remove big concats)

## PR-1.1: Forward pass → preallocated host slices

**Goal:** Eliminate `concatenate` (host/device).

* Changes

  * In `core.py` chunked forward, write each chunk directly into preallocated host buffers via `_alloc_out`.
  * `jax.block_until_ready()` + `del` large device arrays per iteration.
* Tests

  * Small synthetic dataset: exact match to prior results.
  * Peak device memory lower (smoke measurement).
* Acceptance

  * Numerics match baseline; API unchanged.

## PR-1.2: Backward pass smoothing by chunk

**Goal:** Keep device memory ≈ one chunk.

* Changes

  * For each reversed time chunk: send only that **filtered slice** host→device, smooth, write back to host.
  * Carry boundary state between chunks.
* Tests

  * Numerics match baseline on small data.
  * No concatenations present; device peak steady.
* Acceptance

  * Identical outputs within tolerance.

---

# Phase 2 — Host footprint control (opt-in)

## PR-2.1: Memmap large outputs

**Goal:** Avoid host OOM for massive posteriors.

* Changes

  * If `use_memmap=True`, allocate `causal_posterior_out`/`acausal_posterior_out` via memmap (e.g., `/scratch/hmm_out/{name}.dat`).
* Tests

  * Reopen memmaps read-only; verify content/shape/dtype.
  * Mid-run interruption doesn’t corrupt headers (files can be reopened).
* Acceptance

  * RAM usage ~ one chunk on benchmark; doc notes on SSD requirement.

## PR-2.2: Optional quantization (advanced)

**Goal:** Halve disk size vs float16.

* Changes

  * Flag `quantize_posteriors: bool = False` (with `posterior_dtype=np.uint16`) and helpers `to_uint16`/`from_uint16` (clipped 0–1).
* Tests

  * Round-trip error bounds documented (e.g., MAE < 1e-4 on random probs).
* Acceptance

  * Off by default; docs on accuracy tradeoffs.

---

# Phase 3 — xarray without RAM spikes

## PR-3.1: Lazy expansion with dask in `base.py`

**Goal:** Don’t materialize `(time × state_bins)` arrays.

* Changes

  * Add `_lazy_masked_posterior(interior, mask, n_total_bins, time_chunks, dtype)` using **dask.map_blocks** to scatter interior columns into full `state_bins`, filling `NaN` elsewhere.
  * `_convert_results_to_xarray(...)` gets kwargs:
    `lazy_expand: bool = True`, `time_chunks: int = 25_000`, `posterior_out_dtype=np.float32`.
  * When `lazy_expand=True`, wrap memmap (or NumPy) interior arrays as dask and lazily expand to full `state_bins`.
* Tests

  * `is_dask_collection` for the big variables.
  * `compute()` on small fixture equals eager `_create_masked_posterior`.
  * Masked NaNs correct; chunk shapes `(time_chunks, n_total_bins)`.
* Acceptance

  * Conversion uses O(1 chunk) RAM; numerics identical on compute.

## PR-3.2: Streamed NetCDF writes

**Goal:** Save without loading everything.

* Changes

  * In `save_results()`: provide `encoding` (per-var chunk sizes, `zlib=True`, `complevel=4–6`), pick `engine="h5netcdf"` or `netcdf4`.
* Tests

  * Write→reopen→equality (small fixture).
  * File size reduced ≥30% with compression.
* Acceptance

  * Large runs complete without OOM; files reopen cleanly.

---

# Phase 4 — Optional footprint minimization

## PR-4.1: Skip persisting causal posterior

**Goal:** Halve disk if downstream only needs smoothed.

* Changes

  * `persist_causal: bool = True` → if False, don’t preallocate/write `causal_posterior_out`; keep chunk-local for smoothing.
* Tests

  * Smoother outputs unchanged; dataset omits causal posterior variable.
* Acceptance

  * Disk/time reduced; docs clarify downstream implications.

## PR-4.2: Derived summaries only

**Goal:** Orders-of-magnitude smaller artifacts.

* Changes

  * Flags: `emit_map`, `emit_entropy`, `emit_state_aggregates` → compute per chunk; write small arrays (memmap or in-RAM).
* Tests

  * MAP equals `argmax` of posterior; entropy matches `-∑p log p`; aggregates equal baseline.
* Acceptance

  * Greatly reduced footprint with correct summaries.

---

## Files & touch points

* **`core.py`**

  * Add `_alloc_out`; integrate in chunked forward/backward.
  * Add kwargs: `use_memmap`, `memmap_dir`, `posterior_dtype`, `quantize_posteriors`, `persist_causal`, summary flags.
  * Ensure all writes are **slice assignments**; remove concat paths.
* **`base.py`**

  * Add `_lazy_masked_posterior` (dask).
  * Update `_convert_results_to_xarray` to support `lazy_expand`, `time_chunks`, `posterior_out_dtype`.
  * Update `save_results()` encoding for streamed writes.

---

## Rollout & risk management

* **Incremental PRs** with tight scopes; feature flags default to current behavior until battle-tested.
* **Golden tests**: compare outputs vs baseline on small/medium fixtures; set explicit tolerances for float16/quantized paths.
* **Bench harness**: tiny script that reports wall-time, peak device/host memory, and file size before/after (markdown table in PR).
* **Backout plan**: all new behavior is behind flags; toggling off returns to current behavior.

---

## Success criteria (exec-friendly)

* **Device peak** ~ size of one chunk (verified via profiler).
* **Host peak** ~ one chunk when `use_memmap=True` and `lazy_expand=True`.
* **Numerics** identical on small/medium fixtures (within set tolerances if downcasting/quantizing enabled).
* **IO**: NetCDF saves complete without OOM; reopen and validate; compression effective.
* **DX**: one cohesive config surface; docstrings + `MIGRATING.md` updated; CI covers both eager and lazy/memmap modes.

---

## Copy-paste snippets (for PRs)

**Allocator (core.py)**

```python
from pathlib import Path
import numpy as np

def _alloc_out(shape, dtype, name, memmap_dir=None, mode="w+"):
    if memmap_dir is None:
        return np.empty(shape, dtype=dtype)
    path = Path(memmap_dir); path.mkdir(parents=True, exist_ok=True)
    return np.memmap(path / f"{name}.dat", dtype=dtype, mode=mode, shape=shape)
```

**Lazy expansion (base.py)**

```python
import dask.array as da
import numpy as np

def _lazy_masked_posterior(interior_data, is_track_interior, n_total_bins, time_chunks, dtype):
    interior_da = da.from_array(interior_data, chunks=(time_chunks, interior_data.shape[1]))
    mask = np.asarray(is_track_interior)

    def _expand(block, mask, n_total_bins):
        out = np.full((block.shape[0], n_total_bins), np.nan, dtype=block.dtype)
        out[:, mask] = block
        return out

    return da.map_blocks(
        _expand, interior_da, dtype=dtype,
        chunks=(interior_da.chunksize[0], n_total_bins),
        mask=mask, n_total_bins=n_total_bins,
    )
```
