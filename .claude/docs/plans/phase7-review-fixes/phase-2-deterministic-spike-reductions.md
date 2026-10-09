# Phase 2 — Deterministic parallel spike-row reductions

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs D1](designs.md#d1-deterministic-spike-row-reduction)

**Inputs to read first:**

- [appendix E3](appendix.md#e3-serial-spike-row-loop-structure): the serial loop's jaxpr (`scan` length = number of spikes), why determinism is required (replay digest), and what the A100 hour runs did and did not exercise.
- [designs D1](designs.md#d1-deterministic-spike-row-reduction): the full code for `deterministic_segment_sum`, `deterministic_row_sum`, `deterministic_row_add`, and the encoding-spike pre-sort.
- `src/non_local_detector/likelihoods/common.py:761-853`: `_ordered_spike_row_add`, `_ordered_spike_row_sum`, `sum_spikes_into_rows` (CPU dispatch at `:844-852`).
- Call sites: `clusterless_diffusion.py:699-750` (local) and `:826-875` (non-local); `clusterless_gmm.py:109-130` (`_accumulate_log_likelihood_block`); `streamed_kde.py:376-405` (row-returning `score_positions`).
- `src/non_local_detector/tests/likelihoods/test_deterministic_spike_reduction.py`: existing reduction tests. Several assert equality with a sequential reference (listed in Tasks).
- `docs/performance_artifacts/phase7/gpu-replay.json`: the protocol of the existing 8-case × 50-repeat determinism record. Its header lists the sizes; no script for it is checked in.

**Contracts referenced:** none.

**Designs referenced:** [designs D1](designs.md#d1-deterministic-spike-row-reduction).

## Gate

Tasks 1 and 6 need an idle CUDA GPU (the earlier records used an A100 80GB).
Ask the user which host to use. Check `nvidia-smi` before claiming a device, and
set `CUDA_VISIBLE_DEVICES` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Tasks

1. **Baseline (before any code change).**
   - Add `benchmarks/benchmark_spike_row_reduction.py`. It does synchronized,
     warmed median timings (≥5 repeats) of the current
     `common._ordered_spike_row_sum`, the D1 `deterministic_segment_sum`
     (import it from the prototype
     [evidence/det_segsum.py](evidence/det_segsum.py) until task 2 lands),
     and `jax.ops.segment_sum`.
   - Grid: n_spikes ∈ {16, 64, 256, 2_000, 20_000, 200_000} × columns
     ∈ {64, 500, 16_930}, sorted and unsorted ids. Skip cells whose values
     array exceeds 2 GB.
   - Output: a JSON with device, JAX version and source hash.
   - Run on CPU and on the GPU.
   - Also time one end-to-end `predict_clusterless_diffusion_log_likelihood`
     with 20,000 encoding spikes on one electrode, 2,000 decode rows, and
     non-local and local paths, on both platforms.
2. **Add the primitives.** Put `_segmented_add`, `deterministic_segment_sum`,
   `deterministic_row_sum` and `deterministic_row_add` from D1 into
   `common.py`, replacing `common.py:761-800`.
   - Set `_SMALL_SERIAL_SPIKES` from task 1: the largest n at which the serial
     loop beats the scan on the GPU, at every column count. Use `None` if
     there is no such n.
   - Keep `_serial_row_sum` (today's loop body, renamed) only if
     `_SMALL_SERIAL_SPIKES` is not `None`. Otherwise delete it.
3. **Rewire `sum_spikes_into_rows`.** At `common.py:844-853`, replace the
   CPU/else dispatch with
   `deterministic_row_sum(values, jnp.asarray(safe_indices), selection.n_rows,
   indices_are_sorted=selection.indices_are_sorted)`. Keep the host
   validation and invalid-index normalization at `:828-843` unchanged, and
   rewrite the docstring Notes (`:819-826`) to describe the new dispatch.
4. **Rewire the other callers.**
   - `clusterless_diffusion.py`: pre-sort each electrode's encoding spikes by
     bin, as in D1, just before `enc_bins = jnp.asarray(electrode_bins)` at
     `:699` and `:826`.
   - Replace `_ordered_spike_row_sum(weighted_kernel, enc_bins, n_bins)`
     (`:736`, `:860`) with
     `deterministic_row_sum(..., indices_are_sorted=True)`.
   - Replace `_ordered_spike_row_sum(lc, seg_block, selection.n_rows)`
     (`:748`, `:873`) with `deterministic_row_sum(lc, seg_block,
     selection.n_rows, indices_are_sorted=selection.indices_are_sorted)`.
   - `clusterless_gmm.py:129`: use `deterministic_row_add(log_likelihood,
     log_contribution, segment_ids, bin_ids[0])`. Update the comment at
     `:126-128`; it should now say the initial term is preserved and
     contributions are reduced deterministically.
   - `streamed_kde.py:391,400`: use `deterministic_row_add(sums, finished,
     rows, column_start)`. These rows come from a time-sorted selection, so
     pass `indices_are_sorted=True` only if the selection's flag is threaded
     through. Otherwise leave the default (`False`).
5. **Restate the order-asserting tests** in
   `test_deterministic_spike_reduction.py`.
   - `test_fixed_input_order_and_invalid_row_drop` and
     `test_repeated_unequal_float_rows_are_bitwise_identical` still pass on
     CPU, because the CPU dispatch is sequential. Keep them as CPU tests.
   - Add a parametrized backend-independent pair that calls
     `deterministic_segment_sum` directly:
     `test_deterministic_segment_sum_matches_float64_reference[*]` and
     `test_deterministic_segment_sum_is_bitwise_repeatable`.
   - `test_gmm_tile_updates_keep_initial_term_and_each_spike_rounding_order`:
     keep the numeric assertion. Its constructed values are exact under both
     orders. Rename it to
     `test_gmm_tile_updates_keep_initial_term_without_cancellation_loss`.
   - Do not weaken any tolerance. A failure is a finding to report, not a
     reason to loosen a tolerance.
6. **Determinism check script and test.**
   - Add `benchmarks/check_likelihood_determinism.py`, reproducing the
     `gpu-replay.json` protocol: 8 cases (`clusterless_kde`,
     `clusterless_kde_log`, `clusterless_gmm`, `clusterless_diffusion` ×
     local/non-local), 2 electrodes, 72 encoding / 51 decoding marks,
     254 bins, row slice `[2, 12]`, shuffled aligned features, three same-bin
     unequal marks, 50 repeats.
   - It writes per-case `distinct_sha256` to JSON. Run it on CPU and GPU.
   - Add an integration test with 10 repeats on CPU,
     `test_clusterless_likelihoods_are_bitwise_repeatable`, in
     `tests/likelihoods/test_deterministic_spike_reduction.py`. Mark it
     `@pytest.mark.integration`.
7. **Comparison.** Rerun the task 1 benchmark and end-to-end timings on the
   new code. Report deltas against the baseline. Then run the affected
   modules on CPU (JAX 0.9.0 and the 0.11.2 overlay, see
   [appendix](appendix.md#running-on-the-ci-jax-version)) and on the GPU with
   `JAX_ENABLE_X64=1`:
   - `test_deterministic_spike_reduction.py`
   - `test_clusterless_diffusion*.py`
   - `test_clusterless_gmm*.py`
   - `test_gmm*.py`
   - `test_clusterless_kde*.py`
   - `test_streamed_kde.py`
   - `tests/core/test_checkpointed_inference.py`
8. **Docs.**
   - In `docs/performance_validation.md:39-46`, rewrite "Ordered event
     reductions" as a deterministic segmented reduction with a fixed tree.
     Cite the new determinism JSON next to the frozen `gpu-replay.json`, and
     state the GPU timing delta from task 7.
   - In `CHANGELOG.md` `[Unreleased]` → "Checkpointed prediction and
     structured transitions", add one bullet: GPU likelihood reductions are
     deterministic across replay passes without serializing over spikes.

## Deliberately not in this phase

- Relaxing the replay digest to a tolerance check. That is out of scope, see
  [overview non-goals](overview.md#non-goals).
- Restructuring `_joint_core` loop order (finding 9).
- `XLA_FLAGS` deterministic-ops settings ([designs D1, alternatives](designs.md#d1-deterministic-spike-row-reduction)).
- Changing the CPU `segment_sum` path's `indices_are_sorted` hint logic,
  which was verified in earlier work.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_deterministic_segment_sum_matches_float64_reference[sorted/unsorted × tail ∈ {(), (5,), (2,3)}]` | ≤ sequential `segment_sum`'s own error against float64 (measured scan 6e-7–1.1e-6 vs 9e-7–1.4e-6); invalid ids dropped |
| `test_deterministic_segment_sum_is_bitwise_repeatable` | 20 repeats give identical `uint32` views |
| empty / all-invalid / NaN-row / gradient cases (extend existing tests) | zeros shape `(n_rows, ...)`; NaN confined to its row; `grad` gathers the cotangent at owned rows, 0 for invalid |
| `test_clusterless_likelihoods_are_bitwise_repeatable` (`integration`) | one SHA per case over 10 CPU repeats |
| `benchmarks/check_likelihood_determinism.py` on GPU | one SHA per case over 50 repeats, all 8 cases |
| Existing modules listed in task 7 | pass at unchanged tolerances on CPU (both JAX versions) and GPU (x64) |
| `benchmarks/benchmark_spike_row_reduction.py` before/after | GPU: no regression at n ≤ 64; faster at n ≥ 2,000. CPU: unchanged within noise (same `segment_sum` path) |

## Fixtures

Synthetic only. The determinism protocol sizes come from the `gpu-replay.json`
header. No real data is needed.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- `grep -rn "_ordered_spike_row" src scripts` returns nothing.
