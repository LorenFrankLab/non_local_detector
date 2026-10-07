# Phase 4 — Bound host counts and lazy graph distances

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs D2](designs.md#d2-graph-distance-row-cache-and-caller-owned-outputs)

**Depends on:** [phase 1a](phase-1a-sorted-matrix-emission.md). The row
blocking below relies on the restored matrix emission's fixed 64-row kernels.

**Inputs to read first:**

- [appendix E6](appendix.md#e6-review-findings-confirmed-by-code-reading): the count-matrix and lazy-distance findings.
- `src/non_local_detector/likelihoods/sorted_spikes_kde.py:489-504` and `sorted_spikes_glm.py:676-688`: non-local branches that build `_spike_counts_matrix` for every requested row.
- `src/non_local_detector/likelihoods/sorted_spikes_diffusion.py:594-623`: `_spike_counts_matrix`. It accepts `row_slice` and `_spike_time_order`.
- `src/non_local_detector/graph_distances.py:37-195`: `LazyGraphDistances`. Note `_entries` at `:107-137`.
- `src/non_local_detector/environment.py:955-1045`: `get_distances_to_interior_bins`, with the N-D lazy branch at `:1031-1038`. `LazyGraphDistances` is constructed at `environment.py:601`.
- `src/non_local_detector/models/base.py:1150-1175`: the non-local penalty caller. The local-kernel caller uses the same environment method.
- `src/non_local_detector/tests/environment/test_lazy_graph_distances.py`: existing lazy-distance tests.

**Contracts referenced:** none.

**Designs referenced:** [designs D2](designs.md#d2-graph-distance-row-cache-and-caller-owned-outputs).

## Tasks

- **Row-block the non-local sorted emission.** In both non-local branches,
  replace the single `_spike_counts_matrix` call with a loop over row blocks.
  Add one shared constant next to the emission in
  `likelihoods/common.py`:

  ```python
  # A multiple of the emission's fixed 64-row kernel, so blocked and unblocked
  # requests are bitwise identical. Host counts are at most this many rows.
  NONLOCAL_COUNT_BLOCK_ROWS = 4096
  ```

  ```python
  rates = jnp.asarray(place_fields)[:, is_track_interior]
  summed = no_spike_part_log_likelihood[is_track_interior]
  blocks = []
  for start in range(row_start, row_stop, NONLOCAL_COUNT_BLOCK_ROWS):
      stop = min(start + NONLOCAL_COUNT_BLOCK_ROWS, row_stop)
      counts = _spike_counts_matrix(
          spike_times, time_edges, "Non-Local Likelihood", True,
          slice(start, stop), _spike_time_order=_spike_time_order,
      )
      blocks.append(_poisson_nonlocal_log_likelihood(
          jnp.asarray(counts), rates,
          durations[start - row_start:stop - row_start], summed,
      ))
  log_likelihood = (
      jnp.concatenate(blocks) if blocks
      else _poisson_nonlocal_log_likelihood(
          jnp.zeros((0, len(spike_times))), rates, durations, summed
      )
  )
  ```

  - Move the progress bar from the per-block count call to the block loop:
    `tqdm(range(...), disable=disable_progress_bar)`.
  - Peak memory becomes the output, the concatenation copy, and one count
    block. The output plus its copy matches the per-neuron loop on `main`,
    which held one `(n_rows, n_bins)` temporary.
- **Lazy graph distances.** Implement
  [designs D2](designs.md#d2-graph-distance-row-cache-and-caller-owned-outputs)
  in `graph_distances.py`:
  - `max_cache_bytes`, `_row_cache`, `__getstate__`/`__setstate__`,
    `_rows_for` and `cross_distances`;
  - route `_entries`' Dijkstra batches through `_rows_for`;
  - update the class docstring at `:38-45`, which currently says "Queries do
    not retain a dense cache", to describe the bounded row cache and the
    caller-owned output.
- **Environment caller.** In `environment.py:1031-1038`, branch on
  `isinstance(self.distance_between_nodes_, LazyGraphDistances)` and call
  `cross_distances(position_bin_inds, interior_bin_indices)`. Keep the `np.ix_`
  path for dense arrays.
- **Measurement.** Run one checkpointed `predict` on a 2-D Cartesian
  environment, before and after: `transition_representation="structured"`,
  `local_position_std` set, about 1,000 interior bins, 20,000 rows,
  chunk_size 256. Report:
  - total time spent in `get_distances_to_interior_bins` (wrap it with a timer);
  - the number of `dijkstra` source rows computed (wrap
    `graph_distances.dijkstra`).

  Also run a dense-mode `predict` whose `n_time × n_interior × 8` exceeds
  256 MiB, to show it no longer raises `GraphDistanceBudgetError`.
- **Docs.** In `CHANGELOG.md`, change the `[Unreleased]` bullet sentence
  "Supported N-D setup defers all-pairs graph distances and evaluates
  requested distances in bounded batches" to add that recently used source
  rows are cached within a byte budget, and that position-to-bin distance
  requests are no longer limited by the dense-matrix budget.

## Deliberately not in this phase

- The local sorted, local GLM and no-spike `>256`-row paths. They are
  unchanged ([overview non-goals](overview.md#non-goals)).
- Any change to `transition_operators.py`'s use of lazy distances
  (`:832-840`), apart from the cache benefit it gets automatically.
- Restructuring `streamed_kde._joint_core` (finding 9).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_nonlocal_sorted_blocks_match_unblocked[kde/glm]` (new; monkeypatch `NONLOCAL_COUNT_BLOCK_ROWS` to 128 with 1,000 rows) | bitwise equal to a single-block call; `_spike_counts_matrix` never receives more than 128 rows (spy) |
| `test_cross_distances_match_dense_shortest_paths` (new) | equals `nx.floyd_warshall`/dense `np.ix_` on a small 2-D grid with an exterior hole, including `inf` for unreachable bins |
| `test_cross_distances_output_is_not_budgeted` (new) | with `max_dense_bytes=1024`, a 1,000 × 50 request succeeds; `to_dense()` still raises `GraphDistanceBudgetError` |
| `test_row_cache_avoids_repeat_dijkstra_and_respects_budget` (new) | a second identical call runs 0 Dijkstra sources (spy); cache bytes never exceed `max_cache_bytes`; `max_cache_bytes=0` disables the cache |
| `test_lazy_distances_pickle_drops_cache_and_loads_old_state` (new) | a pickle round trip has an empty cache; `__setstate__` with a state lacking `max_cache_bytes`/`_row_cache` loads with defaults |
| existing `test_lazy_graph_distances.py`, `tests/transitions/`, `tests/models/` structured-representation tests | pass unchanged |

## Fixtures

Small synthetic 2-D environments, built the same way as in
`test_lazy_graph_distances.py`. Sorted spikes are synthetic, with a fixed seed.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
