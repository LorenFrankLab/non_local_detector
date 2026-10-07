# Phase 1b — Limit the Gaussian divisor barrier to singleton tiles

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [appendix E2](appendix.md#e2-gaussian-divisor-barrier): the singleton trigger is needed (the CI failure reproduces exactly without it on JAX 0.11.2), while the mixed-precision trigger costs a kernel-sized buffer and was not needed on arm64.
- [appendix: running on the CI JAX version](appendix.md#running-on-the-ci-jax-version).
- `src/non_local_detector/likelihoods/common.py:856-887`: `log_gaussian_pdf`; the triggers are at `common.py:881-886`.
- `src/non_local_detector/likelihoods/common.py:911-937`: `_log_kernel_matrix`, the hot caller (shapes `(1, n_eval)` and `(n_samples, 1)` per dimension).
- `src/non_local_detector/tests/likelihoods/test_kde_common.py:25-92`: the three Gaussian tests added with the barrier (mixed-dtype true division, large bandwidth, integer inputs).

**Contracts referenced:** none.

**Designs referenced:** none.

## Gate

Before deleting the mixed-precision trigger, get Linux x86 evidence. The arm64
evidence is in [appendix E2](appendix.md#e2-gaussian-divisor-barrier). Ask the
user which route to use
([overview open question 2](overview.md#open-questions)):

- (a) a Linux x86 host with JAX/jaxlib 0.11.2, which is how the earlier
  `mixed-precision-ci.log` was produced; or
- (b) pushing the branch so GitHub CI runs. This is outward-facing, so confirm
  first. Note that GitHub CI does not enable x64 globally; the x64 coverage
  comes from tests that switch precision internally.

On that host, run the 5-module, 171-test matrix from
[appendix E2 item 2](appendix.md#e2-gaussian-divisor-barrier) with the
`singletonbarrier` plugin, x64 off and on, plus
[evidence/hlo_div2.py](evidence/hlo_div2.py).

- **Pass** (171/171 in both modes): do the "Narrow" task below.
- **Fail** under x64: do the "Fallback" task instead, and tell the user it
  changes the accumulator dtype for x64-with-float32-inputs users.

## Tasks

- **Narrow (gate passed).** In `log_gaussian_pdf`, delete the
  `mixed_precision` computation and condition (`common.py:881-884`) so the
  barrier applies only when `x.size == 1 or mean.size == 1`. Rewrite the
  comment at `common.py:876-879` to state the measured reason: XLA:CPU can
  rewrite division by a broadcast scalar into multiplication by a rounded
  reciprocal for singleton-shaped tiles, which makes tiled and untiled KDE
  disagree by about 2e-6 relative in Gaussian tails. The barrier costs only a
  vector-sized buffer for singleton tiles.
- **Fallback (gate failed under x64).** Keep `log_gaussian_pdf` singleton-only
  as above, and make `_log_kernel_matrix` accumulate in the inputs' dtype. At
  `common.py:931` today, `jnp.zeros((samples.shape[0], eval_points.shape[0]))`
  defaults to float64 under x64. Use
  `dtype=jnp.result_type(eval_points, samples, std, 1.0)` instead. Float32
  inputs then no longer form a mixed-precision kernel, so the trigger has
  nothing to protect.
- **Memory regression test.** Add
  `test_kernel_matrix_barrier_is_vector_sized_for_float32_inputs_under_x64` to
  `test_kde_common.py`. Under x64 with float32 inputs (n_eval=64, n_samples=2000,
  2 dims), walk `jax.make_jaxpr(_log_kernel_matrix)(...)` (recurse into
  sub-jaxprs as in [evidence/inspect_jaxprs.py](evidence/inspect_jaxprs.py))
  and assert that no `optimization_barrier` operand has more than
  `max(n_eval, n_samples)` elements. Also assert that the singleton case
  (n_samples=1) still contains a barrier, so the guard can't be dropped
  silently.
- **Docs.** In `docs/performance_validation.md:264-267`, change the second
  sentence of the "Singleton Gaussian tails" bullet. Under Narrow, it states
  that the mixed-precision guard was removed after x86 verification. Under
  Fallback, it states that float32 kernels now accumulate in float32.

## Deliberately not in this phase

- An explicit reciprocal (`(x - mean) * (1 / sigma)`) for shape-independent
  arithmetic. It flushes to zero for `sigma > 1/finfo.tiny` and breaks
  `test_gaussian_large_bandwidth_does_not_flush_standardized_coordinates`.
- Any change to the query-tile padding in `streamed_kde.py` (a real gradient
  fix, [appendix E5](appendix.md#e5-classification-of-the-six-ci-failures)).
- Any tolerance change. None is needed in either branch of the gate.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_streamed_fit_and_local_nonlocal_rows_preserve_original_support[*]` (JAX 0.9.0, 0.11.2 overlay, x86 host) | streamed and untiled fits agree at `FLOAT32_ROUNDING`; the `[False-uniform-False]` case is the singleton guard's regression test |
| `test_kernel_matrix_barrier_is_vector_sized_for_float32_inputs_under_x64` | no kernel-sized barrier operand under x64 with float32 inputs; singleton barrier present |
| `test_gaussian_mixed_dtype_preserves_true_division[*]`, `test_gaussian_large_bandwidth_does_not_flush_standardized_coordinates`, `test_integer_gaussian_and_kde_inputs_use_floating_arithmetic[*]` | pass unchanged |
| The 171-test, 5-module matrix from appendix E2, x64 off and on | all pass on arm64 and x86 |
| `evidence/inspect_jaxprs.py` section 1, x64 on | "current" temp bytes for 20000×1000 drop from 160 MB to ≈0 |

## Fixtures

None new. Tests synthesize inputs inline.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- The x86 gate evidence (host, JAX version, pass counts) is quoted in the PR description.
