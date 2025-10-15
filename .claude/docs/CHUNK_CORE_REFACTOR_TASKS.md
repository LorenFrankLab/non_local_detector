# TASKS.md

A milestone-oriented task list for the chunked filter/smoother refactor, focused on **incremental, testable changes** and xarray integration at large scale (`time≈1,976,441`, `state_bins≈10,650`).

---

## Milestone 0 — Groundwork (no behavior changes)

- **PR-0.1: Output allocator (feature-flagged)**
  - [ ] Add `_alloc_out(shape, dtype, name, memmap_dir=None, mode="w+")` in `core.py`.
  - [ ] Thread kwargs through chunked functions: `use_memmap=False`, `memmap_dir=None` (defaults preserve behavior).
  - [ ] Unit tests: shape/dtype correctness for memmap vs in-RAM; memmap slice persists after reopen.
  - [ ] Docs: brief docstring + README note.
  - **Acceptance:** all existing tests pass when `use_memmap=False`.

- **PR-0.2: Posterior storage dtype knob**
  - [ ] Add `posterior_dtype: np.dtype = np.float32` (cast only on **write**).
  - [ ] Tests: float16/uint16 writes compare to float32 baseline within tolerance on small fixtures.
  - [ ] Docs: explain accuracy trade-offs and recommended defaults.
  - **Acceptance:** default path numerically identical to baseline.

---

## Milestone 1 — Streamed writes (remove big concatenations)

- **PR-1.1: Forward pass → preallocated host slices**
  - [ ] Replace list-accumulation/concats with direct slice writes to `_alloc_out` buffers.
  - [ ] Add `jax.block_until_ready(...)` and `del` of large device arrays after each chunk.
  - [ ] Tests: small synthetic equivalence; GPU peak mem smoke test.
  - **Acceptance:** numerics match baseline; API unchanged.

- **PR-1.2: Backward pass smoothing by chunk**
  - [ ] For each reversed time chunk: host slice → device → smooth → write back.
  - [ ] Maintain proper boundary/initial condition between chunks.
  - [ ] Tests: equivalence vs baseline; ensure no device-side concats remain.
  - **Acceptance:** device peak memory ≈ one chunk.

---

## Milestone 2 — Host footprint control (opt‑in)

- **PR-2.1: Memmap large outputs**
  - [ ] Allocate `causal_posterior_out`/`acausal_posterior_out` as memmaps when `use_memmap=True`.
  - [ ] Ensure directory creation and deterministic filenames (`{memmap_dir}/{name}.dat`).
  - [ ] Tests: reopen read-only; validate shape/dtype/values; mid-run interruption safety.
  - **Acceptance:** host RAM usage ~ size of one chunk on benchmark.

- **PR-2.2: Optional quantization path**
  - [ ] Add `quantize_posteriors: bool = False` and helpers `to_uint16`/`from_uint16` (clip 0–1).
  - [ ] Tests: round-trip MAE < 1e-4 on random probabilities.
  - [ ] Docs: trade-offs + guidance on when to enable.
  - **Acceptance:** disabled by default; metrics documented.

---

## Milestone 3 — xarray without RAM spikes

- **PR-3.1: Lazy expansion with dask (`base.py`)**
  - [ ] Implement `_lazy_masked_posterior(interior, mask, n_total_bins, time_chunks, dtype)` using `dask.map_blocks`.
  - [ ] Add kwargs to conversion: `lazy_expand=True`, `time_chunks=25_000`, `posterior_out_dtype=np.float32`.
  - [ ] Wire lazy path in `_convert_results_to_xarray` when `lazy_expand=True`; keep eager fallback.
  - [ ] Tests: dask-backed variables; `compute()` equality vs eager; NaN placement correctness.
  - **Acceptance:** conversion uses O(1 chunk) RAM; identical numerics after compute.

- **PR-3.2: Streamed NetCDF writes**
  - [ ] In `save_results()`, add per-var `encoding` (e.g., `chunksizes`, `zlib`, `complevel=4–6`) and choose `engine` (`h5netcdf` or `netcdf4`).
  - [ ] Tests: write → reopen → equality on small fixtures; file size reduction ≥30% with compression.
  - **Acceptance:** large dataset saves finish without OOM; files reopen cleanly.

---

## Milestone 4 — Footprint minimization (optional toggles)

- **PR-4.1: Skip persisting causal posterior**
  - [ ] Add `persist_causal: bool = True`; if `False`, don’t preallocate/write causal posterior; keep chunk-local only.
  - [ ] Tests: smoother numerics unchanged; dataset omits causal when disabled.
  - **Acceptance:** disk footprint/time reduced ≈ 50% when disabled.

- **PR-4.2: Derived summaries only**
  - [ ] Flags: `emit_map`, `emit_entropy`, `emit_state_aggregates`.
  - [ ] Compute per chunk; write small arrays (memmap or RAM).
  - [ ] Tests: MAP==argmax; entropy matches `-∑p log p`; aggregates equal baseline.
  - **Acceptance:** orders-of-magnitude smaller artifacts with correct summaries.

---

## Cross-cutting Tasks

- [ ] Add a small **benchmark harness** (synthetic data) reporting wall-time, peak device/host memory, file sizes.
- [ ] Add a **golden numerics** test suite comparing new vs baseline outputs on small/medium fixtures (tolerances defined).
- [ ] Centralize configuration in a `Config` dataclass; ensure all new knobs are documented.
- [ ] Update docs: `MIGRATING.md`, function docstrings, README examples.
- [ ] CI matrix: run tests with `use_memmap` on/off and `lazy_expand` on/off.

---

## Success Criteria

- Device peak memory ~ one chunk; host peak ~ one chunk with `memmap`+`lazy_expand`.
- Numerics identical to baseline on small/medium fixtures (with tolerances for downcast/quantized modes).
- NetCDF saves stream to disk without OOM; reopened datasets validate; compression effective.
- Clear config surface and docs; easy rollback via feature flags.
