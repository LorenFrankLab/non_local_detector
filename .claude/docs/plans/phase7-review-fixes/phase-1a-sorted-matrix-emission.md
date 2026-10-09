# Phase 1a — Restore the matrix sorted-spike emission

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs D4](designs.md#d4-accuracy-relative-sorted-emission-test)

**Inputs to read first:**

- [appendix E1](appendix.md#e1-sorted-emission-order-bought-agreement-not-accuracy): why the ordered path is being removed (the matrix path was the more accurate side of the CI failure).
- [appendix: running on the CI JAX version](appendix.md#running-on-the-ci-jax-version): `python -m pytest`, never bare `pytest`, with the overlay.
- `src/non_local_detector/likelihoods/common.py:341-537`: current ordered emission (`_ordered_poisson_count_block` 341, `_poisson_full_count_log_likelihood` 370-419, `_poisson_packed_log_likelihood` 421-501, wrapper `_poisson_nonlocal_log_likelihood` 504-537).
- `git show 73813467:src/non_local_detector/likelihoods/common.py`, lines 341-433: the pre-fix matrix `_poisson_nonlocal_log_likelihood` (64-row blocks, `Precision.HIGHEST` matmul, `xlogy` scan fallback for zeros, non-finite values and nonuniform durations).
- `src/non_local_detector/likelihoods/sorted_spikes_kde.py:491-504` and `sorted_spikes_glm.py:677-688`: call sites (now pass host `counts`; pre-fix passed `jnp.asarray(counts)`).
- `src/non_local_detector/tests/likelihoods/test_phase7c_sorted_accumulation.py`: tests added or changed by 79156376 (`test_host_counts_preserve_ordered_neuron_arithmetic` 201-242, gradient test 245-286, many-neurons test 318-375).

**Contracts referenced:** none.

**Designs referenced:** [designs D4](designs.md#d4-accuracy-relative-sorted-emission-test).

## Gate

CLAUDE.md requires user approval before changing numerical tolerances. This
phase replaces two exact or `1e-6` comparisons against a float32 sequential
reference with float64-oracle assertions
([designs D4](designs.md#d4-accuracy-relative-sorted-emission-test)). Do not
start the test tasks until the user has approved the D4 assertions, or given
replacements.

## Tasks

- **Baseline.** On the branch head, record:
  1. `uv run python -m pytest src/non_local_detector/tests/likelihoods/test_phase7c_sorted_accumulation.py src/non_local_detector/tests/likelihoods/test_sorted_spikes*.py -q` (JAX 0.9.0) and the same with the JAX 0.11.2 overlay;
  2. the [evidence/order_accuracy.py](evidence/order_accuracy.py) output for `nonlocal` 0 and 1 on JAX 0.11.2;
  3. `scripts/benchmark_phase7_sorted_accumulation.py` CPU timings, using `--baseline-common` set to the 73813467 `common.py`.

  Save these under `/private/tmp/` or the session scratchpad, not in the repo.
- **Restore the matrix emission.** Replace `common.py:341-537` with the
  73813467 `_poisson_nonlocal_log_likelihood` (`git show 73813467:src/non_local_detector/likelihoods/common.py`,
  lines 341-433), copied verbatim. Delete `_ordered_poisson_count_block`,
  `_poisson_full_count_log_likelihood`, `_poisson_packed_log_likelihood` and
  the host-packing wrapper. Nothing else imports them; check with
  `grep -rn "_poisson_packed\|_poisson_full_count\|_ordered_poisson" src scripts`.
- **Call sites.** At `sorted_spikes_kde.py:500` and `sorted_spikes_glm.py:684`,
  pass `jnp.asarray(counts)` again, as at 73813467.
- **Restore the tests to their pre-fix state, then apply D4.**
  1. `test_host_counts_preserve_ordered_neuron_arithmetic`: keep the `zero`
     and `nonfinite` kinds as `assert_array_equal`. Those kinds take the
     `xlogy` scan fallback, which is still sequential. For `sparse` and
     `coincident`, assert D4 rule 1 (≤16 float32 ulp against the float64
     oracle; in float64 mode, ≤16 float64 ulp). Rename the test to
     `test_host_counts_match_float64_oracle_and_keep_xlogy_fallback`.
  2. Gradient test (245-286): restore its 73813467 form, or keep the current
     one if it passes unchanged against the matrix path at `rtol=atol=1e-10`.
     That tolerance is not changed.
  3. `test_many_neurons_match_native_posterior_and_evidence` (318-375): apply
     D4 rules 1 and 2. Keep the `is_missing` assertion and the check that the
     monkeypatch was restored.
- **Remove the obsolete benchmark.** Delete
  `scripts/benchmark_phase7_sorted_accumulation.py`; it only compares ordered
  and matrix versions. Keep its frozen artifacts
  (`docs/performance_artifacts/phase7/ci-fix-sorted-*.json`).
- **Docs.**
  - `docs/performance_validation.md:258-263`: replace the "Sorted KDE/GLM
    emissions accumulate active neurons in population order" bullet with one
    sentence stating that the matrix accumulation is retained, and that its
    CI-runtime difference from the per-neuron float32 reference was the
    reference's own rounding (cite E1's numbers: decoder 4/5130, 3.457e-05).
  - `docs/performance_validation.md:303-318`: replace the ordered-emission
    benchmark paragraph with a sentence pointing to the frozen artifacts as
    historical measurements of a reverted variant.
  - `.claude/docs/plans/likelihood-defect-remediation/phase-7-performance.md:339-347`:
    update the "CI follow-up" paragraph the same way.
  - No CHANGELOG entry. The `[Unreleased]` section never described the
    ordered emission.

## Deliberately not in this phase

- Row-blocking the non-local count matrix (review finding 4) is
  [phase 4](phase-4-memory-bounds.md). It depends on this phase's matrix path.
- The 256-row threshold and the private `_spike_counts_matrix` import are
  [phase 5](phase-5-housekeeping.md).
- Any change to the local (`is_local=True`) sorted paths or `no_spike.py`.
- Snapshot or golden updates. None are expected, since S6 passed with the
  matrix path. If any snapshot changes, stop and follow the snapshot approval
  process in CLAUDE.md.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_many_neurons_match_native_posterior_and_evidence[False/True]` (JAX 0.9.0 and 0.11.2 overlay) | D4 rule 1 on emissions; D4 rule 2 on all output variables and `marginal_log_likelihoods`; `is_missing` preserved |
| `test_host_counts_match_float64_oracle_and_keep_xlogy_fallback[*]` | sparse/coincident ≤16 ulp against float64; zero/nonfinite bitwise equal to sequential `xlogy`; row partitions `[0:1]`, `[1:73]`, `[73:139]` bitwise equal to the full call (64-row fixed blocks) |
| `test_sorted_accumulation_matches_per_neuron_xlogy`, `test_finite_nonlocal_fields_do_not_dispatch_xlogy_for_each_neuron`, `test_many_neurons_preserve_singleton_and_ragged_chunk_values`, `test_float32_parameters_preserve_enabled_x64_accumulator_dtype` | unchanged tests pass unchanged |
| Full suite, `uv run python -m pytest -m "not slow"` | passes; snapshot/golden tests unchanged |
| CI on the PR (Linux x86, Python 3.13 job) | all jobs green; this is the only way to confirm the x86 behavior |
| [evidence/order_accuracy.py](evidence/order_accuracy.py) after the change | "OUT ordered" lines now describe the matrix path; distances to the oracle are no larger than baseline |

## Fixtures

No new fixtures. The many-neurons scenario is synthesized in the test (seed 7303).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- The restored `_poisson_nonlocal_log_likelihood` is byte-identical to 73813467 (`diff` the function bodies).
