# Appendix — Investigation evidence and how to reproduce it

[← back to PLAN.md](PLAN.md)

All measurements were taken on 2026-10-07 on `feat/phase7-performance` at
`f2c3e2b9`, macOS arm64, unless stated otherwise. "CI runtime" means the failed
Python 3.13 run <https://github.com/LorenFrankLab/non_local_detector/actions/runs/37550790919>
(Linux x86-64, JAX/jaxlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1). Scripts live in
[evidence/](evidence/).

- [Running on the CI JAX version](#running-on-the-ci-jax-version)
- [E1. Sorted emission: order bought agreement, not accuracy](#e1-sorted-emission-order-bought-agreement-not-accuracy)
- [E2. Gaussian divisor barrier](#e2-gaussian-divisor-barrier)
- [E3. Serial spike-row loop structure](#e3-serial-spike-row-loop-structure)
- [E4. Streamed joint core recomputes the mark kernel](#e4-streamed-joint-core-recomputes-the-mark-kernel)
- [E5. Classification of the six CI failures](#e5-classification-of-the-six-ci-failures)
- [E6. Review findings confirmed by code reading](#e6-review-findings-confirmed-by-code-reading)

## Running on the CI JAX version

```bash
# Scripts: the overlay environment really runs JAX 0.11.2.
uv run --with "jax==0.11.2" --with "jaxlib==0.11.2" python SCRIPT.py

# Tests: use `python -m pytest`. The bare `pytest` entry point has a shebang to
# .venv/bin/python and silently runs the project's JAX (0.9.0), not the overlay.
uv run --with "jax==0.11.2" --with "jaxlib==0.11.2" python -m pytest ...
```

Check with `python -c "import jax; print(jax.__version__)"` inside the same
command before trusting a result. arm64 does not reproduce every x86 codegen
decision; E1's CI difference is not reproducible on arm64, E2's is.

The pre-fix sorted emission is extracted with
`git show 73813467:src/non_local_detector/likelihoods/common.py > common_prefix.py`.

## E1. Sorted emission: order bought agreement, not accuracy

Script: [evidence/order_accuracy.py](evidence/order_accuracy.py)
`<common_prefix.py> <nonlocal 0|1>`. It reruns the scenario from
`test_many_neurons_match_native_posterior_and_evidence`
(`src/non_local_detector/tests/likelihoods/test_phase7c_sorted_accumulation.py:318-375`;
96 neurons, all firing in rows 100 and 512) with three emissions: current ordered,
pre-fix matrix (73813467), and an oracle that sums the same float32 inputs in
float64 and rounds once to float32.

Comparing the oracle run against the ordered run with the test's own assertion
(`rtol=atol=1e-6`), JAX 0.11.2:

| Model | CI failure (matrix vs per-neuron reference) | Oracle vs ordered (local) |
| --- | --- | --- |
| Decoder | 4/5130, max abs 3.4570694e-05, rel 9.170327e-05 | 4/5130, 3.45706940e-05, 9.17032667e-05 |
| Non-local | 89/11286, 1.7017126e-05, rel 9.081025e-05 | 64/11286, 1.7077e-05, 9.108e-05 |

Conclusion: on the CI runtime the matrix path produced the correctly rounded
posterior (exactly, in the decoder case); the per-neuron float32 reference is the
less accurate side. Both ordered and matrix emissions carry up to 5.5–6.3 float32
ulp (≈1.7e-4 absolute) emission error on the 96-neuron rows; the correctly
rounded oracle carries ≤0.5 ulp. On arm64 (JAX 0.9.0 and 0.11.2) matrix and
ordered agree within 3e-7.

Cost of the ordered path, from `docs/performance_validation.md:303-311`:
1.49× (CPU) / 2.26× (A100) slower than the matrix path for sparse chunks,
9.0× / 13.5× slower for a simultaneous burst.

## E2. Gaussian divisor barrier

Current code: `src/non_local_detector/likelihoods/common.py:856-887`
(`log_gaussian_pdf`). The barrier fires when `x.size == 1 or mean.size == 1`
(singleton tile) or when x64 is enabled with float32 inputs ("mixed precision").

1. **Singleton trigger is needed.** On JAX 0.11.2 arm64, the pre-fix formula
   reproduces the CI failure of
   `test_streamed_fit_and_local_nonlocal_rows_preserve_original_support[False-uniform-False]`
   exactly: streamed `10.737826`, reference `10.737849` (CI: `10.737826347351074`
   vs `10.737849235534668`). Script: [evidence/probe_test_values.py](evidence/probe_test_values.py)
   `production|division|reciprocal`.
2. **Test matrix, JAX 0.11.2 arm64, `python -m pytest`, 171 tests** in
   `test_clusterless_kde_streaming.py`, `test_streamed_kde.py`, `test_kde_common.py`,
   `test_clusterless_kde.py`, `test_clusterless_kde_log_parity.py`
   (plugins in [evidence/plugins/](evidence/plugins/), loaded with
   `PYTHONPATH=evidence/plugins ... -p <name>`):

   | Variant | x64 off | x64 on |
| --- | --- | --- |
   | current (singleton + mixed-precision barrier) | 171 pass | 171 pass |
   | `nobarrier` (pre-fix) | 1 fail (the CI case) | 1 fail (same) |
   | `singletonbarrier` (drop mixed-precision trigger) | 171 pass | 171 pass |

3. **Mixed-precision trigger memory.** Compiled `_log_kernel_matrix` with x64
   on, float32 inputs, 20000 samples × 1000 eval points, 2 dims: jaxpr barrier
   operands `float32[20000, 1000]` ×2, `memory_analysis().temp_size_in_bytes`
   160 MB vs 0 MB without the barrier. x64 off, non-singleton: no barrier, 0 MB.
   Script: [evidence/inspect_jaxprs.py](evidence/inspect_jaxprs.py) section 1.
4. **Which operation each case compiles to (arm64).** At the `_log_kernel_matrix`
   call site, optimized HLO keeps an elementwise `divide` of kernel shape in all
   four cases (x64 off/on × singleton/non-singleton), with or without the
   barrier. Script: [evidence/hlo_div2.py](evidence/hlo_div2.py). The
   reciprocal rewrite the code comment describes was not observed here at the
   call site; a bare `(x - m) / s` jit *is* rewritten to `multiply` by a scalar
   reciprocal ([evidence/inspect_jaxprs.py](evidence/inspect_jaxprs.py)),
   so the decision is fusion-context dependent. **x86 was not inspected** (no
   container runtime locally).
5. **Accuracy.** Float64 oracle for the CI case
   ([evidence/barrier_accuracy.py](evidence/barrier_accuracy.py)): at bin 9 of
   `summed_ground_process_intensity`, true division has 8.8e-7 relative error
   and the reciprocal form 1.25e-6. The barrier aligns tile shapes on the
   slightly more accurate operation; both are within float32 Gaussian-tail error.

## E3. Serial spike-row loop structure

Script: [evidence/inspect_jaxprs.py](evidence/inspect_jaxprs.py) section 2.
`_ordered_spike_row_sum` (`common.py:796-800`, body `common.py:761-793`)
traces to one `scan` whose `length` equals the number of spikes (50 → `length=50`,
20000 → `length=20000`). Each step does three `dynamic_slice`s, an add, a select
and a `dynamic_update_slice` on the full `(n_rows, n_cols)` carry; optimized HLO
has one `while`, no `scatter`. `jax.ops.segment_sum` compiles to a single
`scatter-add`. Callers: `common.py:853`, `clusterless_diffusion.py:736,748,860,873`,
`clusterless_gmm.py:129`, `streamed_kde.py:391,400`.

The A100 hour-scale runs did use the loop: recorded hash of
`likelihoods/common.py` `82a0f2807eeb` in
`docs/performance_artifacts/phase7/gpu-hour-*-spatial.json` matches commit
aaea7bf0. Those runs had 200 encoding spikes per electrode and about 10 decoding
spikes per 256-row chunk, so the loop was short; no run compared it with a
parallel reduction.

Why determinism is required at all: checkpointed replay hashes each recomputed
likelihood chunk and raises on mismatch
(`src/non_local_detector/checkpointed_inference.py:510-513`); CUDA float atomics
changed identical chunks between passes (`docs/performance_validation.md:39`).

## E4. Streamed joint core recomputes the mark kernel

Script: [evidence/jaxpr_streamed.py](evidence/jaxpr_streamed.py). For
`_joint_core` (`src/non_local_detector/likelihoods/streamed_kde.py:305-424`) with
64 decoded, 400 encoding, 2000 positions and tiles encoding=100, position=64,
decoding=32, the loop nesting of `exp`/`dot_general` is:

```
scan[31] > scan[2] > scan[4]            exp          [100, 32]   <- mark kernel
scan[31] > scan[2] > scan[4] > scan[1]  exp          [100, 64]   <- position kernel
scan[31] > scan[2] > scan[4] > scan[1]  dot_general  [32,100] x [100,64]
```

The mark kernel sits inside the 31-iteration spatial-tile loop, while the
position loop inside `_joint_numerator` runs once, so the comment at
`streamed_kde.py:245` ("reuse it across all spatial tiles") does not hold for
this call pattern. No timing was taken.

## E5. Classification of the six CI failures

| Test | Nature | Current fix | Plan |
| --- | --- | --- | --- |
| `test_original_degenerate_nan_and_missing_semantics[missing]` | 1-ulp float32 evidence difference | test selects reference accumulation | keep |
| `test_streamed_fit_and_local_nonlocal_rows_preserve_original_support[False-uniform-False]` | tile-shape-dependent divisor rounding (E2) | singleton + mixed barrier | [phase 1b](phase-1b-gaussian-barrier-scope.md) |
| `test_many_neurons_match_native_posterior_and_evidence[False/True]` | test reference less accurate than matrix (E1) | ordered packed emission | [phase 1a](phase-1a-sorted-matrix-emission.md) |
| `test_gaussian_forward_large_finite_input_with_disjoint_axis_mask` | real flush-to-zero: output 0 vs 7.3e37 | scale cap in `transition_operators.py:175-210` | keep |
| `test_large_finite_backward_values_do_not_overflow_before_normalization[RandomWalk]` | same, 0 vs 1e38 | same | keep |

The streamed query-tile padding for traced calls
(`streamed_kde.py:56-130,152-163`) works around a lost last-row gradient, not a
rounding preference; keep.

## E6. Review findings confirmed by code reading

| Review finding | Evidence |
| --- | --- |
| `_sum_evidence` swallows unrelated `OverflowError` | `checkpointed_inference.py:207-216`: `math.fsum(finite)` consumes the forward generator inside `try`, so callback errors are caught as fsum overflow. |
| xarray floor | `xr.Coordinates` absent in xarray 2023.7.0, present in 2023.8.0 (checked in throwaway `uv --python 3.11` envs). Floor is `xarray >=2023.1` (`pyproject.toml:32`). The dense fallback at `models/base.py:3628-3638` also fails on <2023.8 because `hasattr(xr.Coordinates, ...)` raises when `xr.Coordinates` is missing. |
| Bool budget accepted | `models/base.py:1782-1788` checks `isinstance(..., (int, np.integer))`; `True` passes. |
| Validation after refit | `models/base.py:2259-2264` refits environments using `transition_representation == "dense"` before `initialize_continuous_state_transition` validates (`base.py:1777-1788`). |
| Benchmark output path | `scripts/benchmark_phase7b_operators.py:40-42` defaults to `/private/tmp/...`. |
| Lazy distances: no cache, output budgeted | `graph_distances.py:107-137` budgets `rows.size * 8` against `max_dense_bytes` and reruns Dijkstra per call; `environment.py:1031-1038` passes the full `(n_positions, n_interior)` request. |
| Result-store scans | `result_store.py:185-188` tests every chunk against all requested rows; `result_store.py:101-104` checks each write against all prior chunks. |
| Rows × neurons host counts | `sorted_spikes_kde.py:491-504`, `sorted_spikes_glm.py:677-688` build `_spike_counts_matrix` for all requested rows; `no_spike.py:149-163` and the local branches deliberately avoid this above 256 rows. |
