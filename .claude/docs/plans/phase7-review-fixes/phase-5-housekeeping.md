# Phase 5 — Validation order, dependency floor, naming and test hygiene

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Depends on:** [phase 1a](phase-1a-sorted-matrix-emission.md), because the
sorted test module renamed here is edited there, and
[phase 4](phase-4-memory-bounds.md), because the shared constant sits next to
`NONLOCAL_COUNT_BLOCK_BYTES`.

**Inputs to read first:**

- [appendix E6](appendix.md#e6-review-findings-confirmed-by-code-reading): validation order, bool budget, xarray floor, benchmark path.
- `src/non_local_detector/models/base.py:1740-1790`: `initialize_continuous_state_transition` validation (`:1777-1788`).
- `src/non_local_detector/models/base.py:2188-2280`: `_DetectorBase._fit`. The environments are refit at `:2259-2264` before validation runs.
- `src/non_local_detector/models/base.py:200` (checkpointed labelling) and `:3628-3638` (dense fallback that cannot work below xarray 2023.8).
- `pyproject.toml:32`, `environment.yml:13`, `environment_gpu.yml:16`: `xarray >=2023.1`.
- `scripts/benchmark_phase7b_operators.py:40-42` and `benchmarks/benchmark_checkpointed_inference.py:40-42`: hard-coded `/private/tmp` and `/tmp` defaults.
- `src/non_local_detector/likelihoods/no_spike.py:149`, `sorted_spikes_kde.py:423`, `sorted_spikes_glm.py:630`: the three copies of the 256-row threshold. `sorted_spikes_diffusion.py:594-623`: `_spike_counts_matrix`, imported privately by `no_spike.py:25`, `sorted_spikes_kde.py:78`, `sorted_spikes_glm.py:80`.
- `src/non_local_detector/likelihoods/streamed_kde.py:245`: the inaccurate reuse comment ([appendix E4](appendix.md#e4-streamed-joint-core-recomputes-the-mark-kernel)).

**Contracts referenced:** none.

**Designs referenced:** none.

## Tasks

- **Validate transition arguments before any mutation.** Add a module-level
  helper in `models/base.py` and call it as the first statement of
  `_DetectorBase._fit` (`base.py:2188`). Also replace the inline checks at
  `base.py:1777-1788` with a call to it.

  ```python
  def _validate_transition_arguments(transition_representation, max_dense_transition_bytes):
      if transition_representation not in {"dense", "structured", "auto"}:
          raise ValidationError(
              "transition_representation must be dense, structured, or auto"
          )
      if (
          isinstance(max_dense_transition_bytes, (bool, np.bool_))
          or not isinstance(max_dense_transition_bytes, (int, np.integer))
          or max_dense_transition_bytes <= 0
      ):
          raise ValidationError("max_dense_transition_bytes must be a positive integer")
  ```

- **xarray floor.**
  - Change `xarray >=2023.1` to `xarray >=2023.8` in `pyproject.toml:32`,
    `environment.yml:13` and `environment_gpu.yml:16`. Run `uv lock` and
    confirm only the constraint metadata changes.
  - In `base.py:3628-3638`, delete the `hasattr` fallback and keep only the
    `xr.Coordinates.from_pandas_multiindex` branch. Simplify the code that
    follows, which checks `mindex_coords is None`.
  - Add a CHANGELOG `[Unreleased]` line stating the new minimum xarray.
- **Benchmark output defaults.** In both scripts, make the default
  `Path(tempfile.gettempdir()) / "<script-stem>.json"` and call
  `args.output.parent.mkdir(parents=True, exist_ok=True)` before writing.
- **Deduplicate the threshold and the private import (no numerics change).**
  - Move `_spike_counts_matrix` from `sorted_spikes_diffusion.py:594-623` to
    `likelihoods/common.py`, keeping the same name and body. Import it from
    `common` in all four modules, including `sorted_spikes_diffusion.py`
    itself.
  - Add `COMPILED_ROW_LIMIT = 256` to `common.py` with a one-line comment
    saying that larger requests use the per-neuron path to avoid a
    rows-by-population count buffer. Use it at the three threshold sites.
  - Verify the change is bitwise: run `tests/likelihoods/` and the snapshot
    tests before and after with identical results.
- **Rename milestone-named files** with `git mv`, and fix every import and
  doc link:

  | Old | New |
| --- | --- |
  | `tests/likelihoods/test_phase7_local_accumulation.py` | `tests/likelihoods/test_local_poisson_accumulation.py` |
  | `tests/likelihoods/test_phase7c_marked_blocks.py` | `tests/likelihoods/test_marked_kde_blocks.py` |
  | `tests/likelihoods/test_phase7c_sorted_accumulation.py` | `tests/likelihoods/test_sorted_nonlocal_accumulation.py` |
  | `tests/models/test_phase7_diagnostics.py` | `tests/models/test_checkpointed_diagnostics.py` |
  | `tests/models/test_phase7_prediction.py` | `tests/models/test_checkpointed_prediction.py` |
  | `scripts/benchmark_phase7_pipeline.py` | `benchmarks/benchmark_native_pipeline.py` |
  | `scripts/benchmark_phase7a_replay_cache.py` | `benchmarks/benchmark_replay_cache.py` |
  | `scripts/benchmark_phase7b_operators.py` | `benchmarks/benchmark_transition_operators.py` |
  | `scripts/benchmark_phase7c_buckets.py` | `benchmarks/benchmark_kde_buckets.py` |
  | `scripts/benchmark_phase7c_likelihoods.py` | `benchmarks/benchmark_likelihood_kernels.py` |
  | `scripts/qualify_phase7_long_encoding.py` | `benchmarks/qualify_long_encoding.py` |

  - Remove "Phase 7a"/"Phase 7c" from the first docstring lines of
    `benchmarks/benchmark_checkpointed_inference.py:1` and
    `benchmark_phase7c_likelihoods.py:1`, and rename the
    `_phase7c_baseline_` module prefix at `benchmark_phase7c_likelihoods.py:60`
    to `_baseline_`.
  - Update links in `docs/performance_prediction.md:203` and
    `docs/performance_validation.md:130,209`. Do not edit frozen
    `docs/performance_artifacts/` JSON; it records the historical paths.
- **Shared test helper and markers.**
  - Move `recording(family)` (`test_phase7_prediction.py:15-41`) into
    `src/non_local_detector/tests/models/conftest.py` as a fixture named
    `checkpoint_recording`. It returns the builder function. Update both
    model test modules to request it.
  - Add `pytestmark = pytest.mark.unit` to
    `tests/environment/test_lazy_graph_distances.py` and
    `tests/core/test_state_marginal_precision.py`, and
    `pytestmark = pytest.mark.integration` to the renamed
    `test_checkpointed_prediction.py`.
  - Check whether the other renamed modules carry markers, and add
    `unit`/`integration` where missing.
- **Fix the misleading comment** at `streamed_kde.py:245`. State that the
  mark tile is computed once per encoding tile for the spatial tile passed in,
  and that `_joint_core` passes one spatial tile per call.

## Deliberately not in this phase

- Unifying the `>256`-row per-neuron paths with the compiled paths. That
  changes the default dense numerics ([overview non-goals](overview.md#non-goals)).
- Restructuring `_joint_core` to stop recomputing the mark kernel. Deferred
  ([overview non-goals](overview.md#non-goals)).
- Rewording "Phase 7" prose in `docs/performance_validation.md` beyond the
  link targets.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_fit_rejects_invalid_transition_arguments_before_refit[Dense/structured-typo/True/0/-1]` (new) | `ValidationError` is raised, and the detector's previously fitted `environments[0].place_bin_centers_` is the same object as before the call |
| `test_fit_accepts_numpy_integer_budget` (new) | `np.int64(2**20)` is accepted |
| full suite `uv run python -m pytest -m "not slow"` | passes after the renames; `pytest --collect-only -q` lists no `phase7` names |
| `uv run python -m pytest -m unit --collect-only -q` | includes the lazy-distance and state-marginal tests |
| likelihood and snapshot tests before/after the threshold/import move | identical results (bitwise) |
| `python benchmarks/benchmark_transition_operators.py --help` and a smoke run with defaults on Linux | writes to the system temp dir without error |

## Fixtures

`checkpoint_recording` in `tests/models/conftest.py` (moved, not new).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- `git grep -n "phase7\|Phase 7[abc]" -- src scripts` returns nothing.
