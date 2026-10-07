# Phase 3 — Checkpointed inference and result-store fixes

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs D3](designs.md#d3-result-store-chunk-lookup)

**Inputs to read first:**

- [appendix E6](appendix.md#e6-review-findings-confirmed-by-code-reading): the confirmed findings this phase fixes.
- `src/non_local_detector/checkpointed_inference.py:186-223`: `_sum_evidence`, where the `try` around `math.fsum(finite)` at `:208-216` also catches errors raised by the chunk generator.
- `src/non_local_detector/checkpointed_inference.py:362-374`: operator handoff. Pytree operators keep NumPy leaves and are re-converted by every jitted call at `:488`, `:526`, `:538`.
- `src/non_local_detector/checkpointed_inference.py:87-100`: `_CallableTransition`. It has no array leaves and must not be passed to `device_put`.
- `src/non_local_detector/result_store.py:85-109` (writer overlap check) and `:162-215` (`_ChunkedArray` reader).
- `src/non_local_detector/tests/core/test_checkpointed_inference.py:650-715`: existing `_sum_evidence` semantics tests.

**Contracts referenced:** none.

**Designs referenced:** [designs D3](designs.md#d3-result-store-chunk-lookup).

## Tasks

- **Make `_sum_evidence` let chunk errors propagate.** Replace
  `checkpointed_inference.py:188-223` with:

  ```python
  def _sum_evidence(chunks):
      """Float64 sum of per-chunk evidence with IEEE nonfinite propagation.

      Holds one float per chunk. Exceptions raised while producing chunks
      propagate unchanged; only ``math.fsum``'s own overflow is handled.
      """
      values = [float(value) for value in chunks]
      has_nan = any(math.isnan(value) for value in values)
      has_pos, has_neg = math.inf in values, -math.inf in values
      if has_nan or (has_pos and has_neg):
          return math.nan
      if has_pos:
          return math.inf
      if has_neg:
          return -math.inf
      try:
          return math.fsum(values)
      except OverflowError:
          # Extreme finite inputs can overflow fsum's partials; keep the
          # scalar IEEE overflow behavior of sequential addition.
          total = 0.0
          for value in values:
              total += value
          return total
  ```

  An hour at 256-row chunks is about 7,000 floats. The call site at `:500`
  is unchanged.
- **Move the transition operator to the device once.** In the `else` branch
  at `checkpointed_inference.py:371-374`, after the `_CallableTransition`
  wrap, add `transition_operator = jax.device_put(transition_operator)`, but
  only when it was *not* wrapped. The `_DenseTransition` branch
  (`:368-370`) already builds device arrays.
- **Result-store reader.** Implement the reader part of
  [designs D3](designs.md#d3-result-store-chunk-lookup) in `_ChunkedArray`
  (`result_store.py:162-215`). Keep the `max_read_bytes` check, the
  shape/dtype validation, and the `_mmap.close()` in `finally` unchanged.
- **Result-store writer.** Implement the writer part of D3 in
  `IncrementalResultWriter`. Initialize `self._intervals = {name: [] for
  name in variables}` in `__init__` (`result_store.py:34-60`), and replace the
  `any(...)` scan at `:101-104`.
- **Measurement.** Before and after, report:
  1. a 3000-chunk × 256-row full read of one compact variable (review
     baseline: 1.5 s);
  2. per-call time of `_forward_chunk` with a `BlockTransitionOperator`
     holding one 128 MiB `DenseBlock` fallback, NumPy leaves vs after the fix
     (review baseline on CPU: 11.9 ms vs 9.3 ms).

  Use a throwaway script and quote the numbers in the PR.
- **Docs.** None user-facing. The CHANGELOG bullet on result stores already
  promises lazy reopening; the fix makes that hold at hour scale.

## Deliberately not in this phase

- The replay digest check and stable evidence semantics. Unchanged.
- Any change to the result-store on-disk format or manifest version.
- Lazy graph distances (finding 5) is [phase 4](phase-4-memory-bounds.md).
- The xarray floor (finding 7) is [phase 5](phase-5-housekeeping.md).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sum_evidence_propagates_chunk_source_errors` (new) | a source that yields `1.0` then raises `OverflowError("callback")` makes `_sum_evidence` raise that error (today it returns `1.0`) |
| existing `_sum_evidence` tests (`test_checkpointed_inference.py:650-715`) | NaN, ±inf, mixed-inf and fsum-overflow semantics unchanged |
| `test_checkpointed_operator_leaves_reach_jit_as_device_arrays` (new) | monkeypatch `_forward_chunk`/`_backward_chunk` with wrappers that record leaf types; for a structured operator with NumPy leaves, every recorded leaf is a `jax.Array`; a `_CallableTransition` operator still runs |
| `test_chunked_reads_match_dense_for_random_outer_selections` (new) | 40 chunks of random sizes written in shuffled order; random sorted, unsorted and duplicate row selections plus bin slices equal the in-memory array |
| `test_writer_rejects_overlap_in_any_order` (new) | overlapping writes raise regardless of order; adjacent chunks `[0,5)`, `[5,9)` are accepted |
| `tests/core/test_checkpointed_inference.py`, result-store tests | pass unchanged |

## Fixtures

Synthetic arrays in `tmp_path`. The structured operator test can reuse the
`BlockTransitionOperator` fixtures in
`src/non_local_detector/tests/transitions/test_transition_operators.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
