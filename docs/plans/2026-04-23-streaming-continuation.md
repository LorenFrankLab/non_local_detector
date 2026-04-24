# Streaming Plan — Continuation from Fresh Session (2026-04-23)

> **For Claude:** Start by reading this doc, then
> `docs/plans/2026-04-17-streaming-likelihood-into-hmm.md` (the v3 plan),
> then the commits on branch `streaming-likelihood-hmm`.  Do NOT
> re-derive the scope — just pick up where the last session stopped.

## TL;DR

PR [#23](https://github.com/LorenFrankLab/non_local_detector/pull/23) is
**draft** on branch `streaming-likelihood-hmm`, 13 commits ahead of
main.  The API surface (`memory_budget: int | Literal["auto"] | None`
on fit / predict / estimate_parameters) is fully implemented for both
`ClusterlessDetector` and `SortedSpikesDetector`.  128 unit +
integration tests green on CPU.  `memory_budget="auto"` at full 80 GB
produces **bit-exact** output vs the pre-PR `n_chunks=1` baseline —
non-regression invariant holds.

**What blocks closing out the plan:** the memory model under-estimates
peak memory significantly on real-data workloads, so the auto
selector's chosen knobs OOM at memory-constrained budgets (e.g. 24
GB simulated).  **This needs a dedicated Task 6 calibration pass**
before Task 8 validation can demonstrate the feature working at
reduced memory.

---

## PR #23 current state

### Commits on `streaming-likelihood-hmm`

```
cf3abe1  Task 7b (fit) + Task 8 partial: wire memory_budget into fit + calibration bump
0c258db  Task 7b: extend multi-knob selector wiring to SortedSpikesDetector
41954ab  Task 7b: wire multi-knob selector into predict (clusterless)
9d865d0  Task 7a: expose memory_budget parameter on fit/predict/estimate_parameters
09e9fff  Task 5: multi-knob memory-aware selectors for predict + fit
ef6b74d  Task 4b: _estimate_fit_peak_bytes for every likelihood module
0620602  Task 4a: _estimate_predict_peak_bytes for every likelihood module
e333c7a  Extend streaming plan v3 to cover fit + estimate_parameters
6d1932a  Rewrite streaming plan to v3: multi-knob memory-aware auto
caf78d4  Task 3: guard return_outputs='log_likelihood' when streaming
410bad3  Task 2: Flip n_chunks default to 'auto' on all predict signatures
ad08c2a  Task 1: _resolve_n_chunks memory-aware heuristic
0fadc08  Rewrite streaming plan around memory, not compute redundancy
eb324fc  Audit streaming-likelihood plan for current main
```

### Key files

| File | What it contains |
|---|---|
| `src/non_local_detector/streaming.py` | `_resolve_n_chunks`, `auto_select_predict_memory_knobs`, `auto_select_fit_memory_knobs`, ladder helpers |
| `src/non_local_detector/likelihoods/clusterless_kde.py` | `_estimate_predict_peak_bytes` + `_estimate_fit_peak_bytes` (with 5× safety multiplier on predict) |
| `src/non_local_detector/likelihoods/clusterless_kde_log.py` | Re-exports clusterless_kde estimators |
| `src/non_local_detector/likelihoods/sorted_spikes_kde.py` | Same two estimators for sorted-spikes KDE |
| `src/non_local_detector/likelihoods/sorted_spikes_glm.py` | Same for sorted-spikes GLM |
| `src/non_local_detector/likelihoods/clusterless_gmm.py` | Stub estimators delegating to clusterless_kde |
| `src/non_local_detector/models/base.py` | Auto resolution wired into `_predict`, `fit_encoding_model`, `.fit()`, `.predict()` |
| `src/non_local_detector/tests/test_memory_model.py` | 56 unit tests for estimators |
| `src/non_local_detector/tests/test_streaming_predict.py` | 40 tests: chunked-parity + selector + auto-matches + guard + memory_budget |

### Test counts

- **56 unit tests** in `test_memory_model.py` — per-algorithm peak estimator shape invariants (chunking reduces peak, tiling reduces peak, fp64 vs fp32, GMM delegates to KDE, etc.).
- **40 integration tests** in `test_streaming_predict.py` — chunked-parity, auto-matches-unchunked, memory_budget matches baseline, return_outputs guard, selector ladder transitions.
- All green on CPU.

### API surface (stable; backward-compatible)

```python
detector.fit(
    position_time, position, spike_times, spike_waveform_features,
    memory_budget: int | Literal["auto"] | None = "auto",
)

detector.predict(
    spike_times, time,
    memory_budget: int | Literal["auto"] | None = "auto",
    n_chunks: int | Literal["auto"] = "auto",
    # ...plus all existing kwargs
)

detector.estimate_parameters(
    ...,
    memory_budget: int | Literal["auto"] | None = "auto",
)
```

- `"auto"` queries `jax.devices()[0].memory_stats()["bytes_limit"]`.
- `int` is an explicit byte budget (for simulation, shared GPUs).
- `None` disables auto entirely — falls back to existing explicit knobs.

---

## Validated on real data

**Run A** (`/tmp/streaming-task8/task8-baseline_*`): `n_chunks=1` at 80 GB — fit 25 s, predict compile+first 137 s, steady med 85 s.

**Run B** (`/tmp/streaming-task8/task8-auto80gb_*`): `memory_budget="auto"` at 80 GB — fit 24 s, predict compile+first 137 s, steady med 85 s.

**Correctness:** `max|diff| = 0.000e+00` on both `acausal_posterior` and `acausal_state_probabilities` between Run A and Run B.  Bit-exact; `memory_budget="auto"` at full memory is a pure passthrough.

---

## What's blocking Task 6 + Task 8 completion

At `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` (simulated 24 GB on A100):

- The memory model says auto's chosen knobs fit the budget.
- XLA allocates tensors the model doesn't account for, OOMs.
- Two observed failing allocations from different code paths:
  - During **fit**: `f64[n_enc_max × n_enc_max]` ≈ 8 GB transpose per tetrode.  I don't know exactly where this comes from in the `KDEModel.fit` / `kde_distance` call graph — needs instrumentation.
  - During **predict** (earlier Task 4 runs): `f64[723, n_time, 1]` ≈ 4 GB intermediate.  Shape suggests per-electrode operation over full-time axis.  Not chunked by our `n_chunks` because it's computed inside the per-electrode loop.

## Root-cause hypotheses for the model under-estimate

1. **JAX async dispatch keeps intermediates alive.**  The Python for-loop over electrodes in `predict_clusterless_kde_log_likelihood` dispatches 22 JIT calls in rapid succession without blocking.  Each call's `position_distance` + `log_joint_mark_intensity` lingers until the filter consumes the chunk's likelihood.  At peak, ~22× `per_electrode_live` coexists, not 1×.  My 5× multiplier is close but probably off.
2. **Unmodeled XLA transposes.**  The `(n_enc_max, n_enc_max)` shape looks like it comes from XLA choosing a transpose-based fusion for a matmul.  Dense transpose is `n × n × dtype` — not `n_enc × n_pos × dtype` as my model assumes.
3. **Fit-time occupancy eval also async.**  Similar story: per-electrode KDE evals don't block, keep lingering intermediates alive.

## Why ad-hoc multiplier bumps aren't enough

Bumped clusterless predict multiplier 2.0 → 5.0 based on observed peak at 80 GB (33 GB peak / 7.5 GB modeled).  At 24 GB, workload STILL OOMs — the model is still under by some factor.  Needs empirical calibration across multiple budgets, not a single observation.

---

## Task 6: dedicated calibration plan (next session)

### Goal

Fit the model's constants (`_SAFETY_MULTIPLIER`, `fixed_scratch`) against empirical `peak_bytes_in_use` observations across a sweep of workload shapes + knob configs, per algorithm + operation type.

### Step-by-step

1. **Build instrumented benchmark** `state-space-playground/scripts/benchmark_memory_model.py`:
   ```python
   # For each config in a grid:
   jax.clear_caches()
   dev = jax.devices()[0]
   dev.memory_stats()  # reset peak
   # Run fit or predict
   jax.block_until_ready(result)
   peak_bytes = dev.memory_stats()["peak_bytes_in_use"]
   modeled = estimator(**workload, **knobs)
   log {config, modeled, observed, ratio}
   ```
   Sweep: (n_chunks ∈ {1, 2, 4, 8}) × (block_size ∈ {1000, 10000}) × (enc_tile_size ∈ {None, 4096}) × (n_tetrodes ∈ {22, 64}) × (n_time ∈ {100k, 709k, 1.8M}).  Maybe 48 configs × ~5 min each = ~4 GPU hours.
2. **Fit/bound the model** against observations.  If ratio observed/modeled is stable within an algorithm (e.g. always 4×-5× for clusterless predict), use that as the multiplier.  If it scales with n_electrodes (likely), add `n_electrodes` as a workload param and scale per-electrode term by `n_electrodes`.
3. **Update constants** in `_estimate_predict_peak_bytes` / `_estimate_fit_peak_bytes` per-algorithm.  Add `fixed_scratch` that's empirically calibrated.  Document the calibration commit message with the observed data points.
4. **Commit artifact** to `docs/benchmarks/streaming-memory-model.md`:
   - Table of (config, modeled, observed, ratio).
   - Plot of modeled vs observed.
   - Resulting constants + explanation of how they were fit.

### Alternative: query-first approach

Instead of building a predictive model from tensor sizes, **run the JIT-compile step alone at selector time** and read back the actual buffer sizes XLA chose.  More accurate but requires carefully not executing the kernel; tricky.  Defer unless step 1-3 doesn't converge.

---

## Task 8: real-data validation (after Task 6)

Once the model is calibrated:

1. Re-run the Task 8 matrix:
   - 80 GB baseline (n_chunks=1 + memory_budget=None), memory_budget="auto" at each of 80/40/24/12/8 GB via `XLA_PYTHON_CLIENT_MEM_FRACTION`.
   - For each, capture: fit wall-clock, predict compile+first, predict steady med, `peak_bytes_in_use`.
   - Compare posteriors vs baseline to 1e-4.
2. Expect the calibrated model to pick knobs that actually fit.  If a workload can't be auto-fit at a given budget, it should raise `RuntimeError` from the selector (already handled), not OOM mid-execution.

## Task 9 (optional): chronic 1-hour synthetic demo

Stack the 20-min HPC epoch 3× to create 1-hour equivalent; show auto enables it on 16 GB simulated where explicit `n_chunks=1` OOMs.  The user-visible "feature that didn't work, now works" deliverable.  Skip if Task 8 is compelling on its own.

---

## Reference artifacts from this session

### Successful (keep)

- `/tmp/streaming-task8/task8-baseline_*` — 80 GB n_chunks=1 baseline artifacts
- `/tmp/streaming-task8/task8-auto80gb_*` — 80 GB memory_budget=auto (matches baseline)
- Comparison showed `max|diff|=0.0` — the bit-exact non-regression invariant

### Failed (useful as negative evidence)

- `/tmp/streaming-task8/run-auto24gb.log` — 24 GB OOM on `(n_enc_max, n_enc_max)` fit transpose
- `/tmp/pr19-postmerge-validation/...` — PR #19 era artifacts with observed 33 GB peak for NonLocal 2D

---

## Session-independent setup notes

### Playground venv drift

The editable install in `/cumulus/edeno/state-space-playground/.venv` keeps reverting to a stale git-installed version.  After checkout / branch switch, re-link with:

```bash
VIRTUAL_ENV=/cumulus/edeno/state-space-playground/.venv uv pip install \
    -e /cumulus/edeno/non_local_detector --no-deps \
    --python /cumulus/edeno/state-space-playground/.venv/bin/python
```

Then verify with:

```bash
/cumulus/edeno/state-space-playground/.venv/bin/python -c "
import non_local_detector as m
print(m.__file__, m.__version__)
from non_local_detector.streaming import auto_select_predict_memory_knobs
"
```

The `__file__` should start with `/cumulus/edeno/non_local_detector/src/`.

### Running on GPU

The machine has 10× A100 80 GB.  `state_space_playground.gpu.pick_free_gpu()` picks one automatically.  Runs typically use GPU 1 or higher (0 is often in use).

### Simulating smaller GPU memory

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.10  # 8 GB on 80 GB A100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30  # 24 GB
XLA_PYTHON_CLIENT_MEM_FRACTION=0.50  # 40 GB
```

`bytes_limit` from `jax.devices()[0].memory_stats()` will correctly report the reduced value.

### Decode script

`/cumulus/edeno/state-space-playground/scripts/decode_hpc_for_branch_comparison.py`
 accepts `--n-chunks` but not yet `--memory-budget` — add if Task 8 benchmark needs it.

---

## Decisions pending review

1. **Merge Task 7b/7b-fit as-is?** The correctness story is clean (bit-exact at full memory) but the memory savings story is aspirational.  Options:
   - A: keep in draft, do Task 6 calibration in same PR
   - B: merge as-is with clear docs that `memory_budget="auto"` is correct but may over-promise at constrained budgets — users should pass `None` to disable if they hit unexpected behavior
   - C: revert Task 7b, keep only Tasks 1-5 (pure machinery with no wiring) — feels like halfway
2. **Task 6 scope**: is the "run calibration benchmark + fit constants" approach enough, or should we also redesign the model to query XLA for actual buffer sizes after trace?

My lean: **A** (do Task 6 in same PR).  The draft PR is the right signal.  Memory savings without calibration is an incomplete feature; shipping it with known OOMs on constrained budgets is worse than leaving it in draft until calibration lands.

---

## Plan files

- `docs/plans/2026-04-17-streaming-likelihood-into-hmm.md` — v3 plan with Tasks 1-9
- `docs/plans/2026-04-20-clusterless-gpu-optimization-redesign.md` — separate, orthogonal (PR #19 v2 redesign)
- **This file** — continuation-specific

## Related issues / PRs

- PR #14 (merged) — `kde_distance` log-space rewrite
- PR #19 (reverted via #22) — scan path ptxas pathology (issue #21)
- PR #22 (merged) — the revert + post-mortem
- PR #23 (draft, this work) — streaming + memory-aware auto
- Issue #21 — ptxas compile pathology (still open)

## Contact / context

Session ended 2026-04-23.  User priority: the memory-aware auto should "just work" end-to-end so users on any GPU can analyze any workload without thinking about knobs.  Current PR delivers the infrastructure + correct behavior at full memory but not yet the constrained-memory win.  Task 6 is the critical-path item before Task 8 validates and PR can leave draft.
