# Clusterless KDE GPU Optimization — Redesign Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Context:** PR #19 (merged `fcf5100`, reverted via #22) optimized the
`clusterless_kde_log` non-local likelihood for GPU and claimed
~9× speedup on a synthetic 64-tetrode benchmark.  Post-merge real-data
validation found that the scan-over-electrodes path
(`_predict_nonlocal_electrode_scan_impl`, original commit `f248064`)
produces a fused XLA kernel that `ptxas` cannot compile in bounded time
on 2D HPC shapes common to production (e.g. 22 tetrodes, 4-D waveform,
3420 position bins, 709k time bins — ptxas stuck >30 min).  Pre-PR
`clusterless_kde_log` on the same shape compiled in 168 s and ran
cleanly at 83 s per predict.

**Goal:** Redesign the clusterless GPU optimization with shape-aware
routing and compile-safety baked in from day one, not as a post-hoc
flag.  Ship correctness + compile-safety + performance as a single
coherent PR.

**Branch:** Execute on branch `clusterless-gpu-optimization-v2` off
`main`.  Reference commits on `gpu-opt-investigation` (the preserved
original PR #19 history) for cherry-picking candidate building blocks.

**Tech Stack:** JAX (jit, lax.fori_loop, lax.scan, ops.segment_sum,
make_jaxpr), NumPy, pytest.

---

## Why PR #19 failed (the three root causes)

### 1. Pre-merge validation exercised the wrong code path

PR #19's real-data validation used `clusterless_algorithm="clusterless_kde"`
(the default, which maps to `clusterless_kde.py` — a 594-line product-space
module that PR #19 did not modify at all).  The scan path lives in
`clusterless_kde_log.py`, only used when the user explicitly passes
`clusterless_algorithm="clusterless_kde_log"`.  "Bit-exact vs main" was
trivially true: both branches ran identical unchanged code.

**Mitigation for v2:** all PR validation MUST run with both `clusterless_kde`
AND `clusterless_kde_log` explicitly (see updated workflow in
`.claude/` memory: `real_data_pr_validation_workflow.md`), AND
compare against pre-PR main, AND include cross-path parity between the
two modules at the same SHA.

### 2. Unit tests + synthetic benchmark did not predict real-data compile behavior

Unit tests use small shapes (few electrodes, few position bins) where
XLA's fusion decisions produce tractable PTX.  The 64-tetrode synthetic
benchmark reportedly compiled OK (no reproducer artifact available).
But at 22-tetrode / 4-D / 3420-pos shape, ptxas hit a register-spill
pathology on the fused scan body.

**Mitigation for v2:** require real-data compile-time benchmarks on
≥3 shape classes before merge.  See §Verification below.

### 3. Compile time was not treated as a pass/fail criterion

PR #19 timed steady-state throughput but never tracked compile+first.
A 30-min compile is a hard production blocker even if steady-state is
fast — every session pays it, EM iterations multiply it.

**Mitigation for v2:** compile+first is reported as a first-class metric
in the benchmark artifacts, with a 20-minute pass/fail threshold on
production shapes.

---

## What validated correctly (reference material, not re-implementation)

During post-merge investigation, these components of PR #19 produced
correct output on real 2D HPC data via the fallback path
(`enc_tile_size=4096` forces the Python-loop-per-electrode dispatch):

| component | pre-PR commit on `gpu-opt-investigation` | verified on real data |
|---|---|---|
| Block `fori_loop` (Task 1) | `aec22af` | ✅ via fallback |
| Enc GEMM precompute (Task 2) | `d0c9265` | ✅ via fallback |
| Fused `segment_sum` (Task 4) | `a7edf6d` | ✅ via fallback |
| Dec GEMM hoist (Phase 0) | `cd9ae1b` | ✅ via fallback |

Correctness floor (post-PR fallback vs pre-PR unchunked, both on
real-data 2D HPC): `max|diff| ≈ 8e-6` on `acausal_posterior`, within
fp64 log-space ULP noise.  These can be re-introduced into v2 as
reference implementations if the v2 approach needs equivalents.

**What did not validate:**

- Scan over electrodes (Task 3, `f248064`): compile pathology at 2D HPC shape
- `auto_select_tile_sizes` + `block_size="auto"` (Tasks 5, `8d8afcc` +
  `2885ae0`): wired to scan path; not exercised on any real-data shape

---

## v2 Design Requirements

### R1. Shape-aware router with compile-timeout fallback

The entry point decides per-call whether to use the scan body or the
per-electrode loop based on:

- Estimated fused-kernel size (function of `n_electrodes × n_pos × block_size × n_wf`)
- A hard compile-time budget (e.g. 15 min on first call)
- A cached decision keyed by input shape signature so repeat calls skip the estimation

If the router chose scan and compile exceeds the budget, fall back to
the per-electrode loop transparently — user sees a warning, not a hang.

### R2. No undocumented opt-in flag for broken code

If a code path isn't validated on the full production shape matrix,
it doesn't ship.  No "use_scan_path_experimental=False" escape hatches.

### R3. Compile-time as a test gate

Unit tests should include a compile-time assertion on
production-shape inputs.  If the compile exceeds the budget, the test
fails — caught in CI, not in post-merge validation.

### R4. Real-data validation on ≥3 shape classes before merge

Minimum validation matrix (see
`state-space-playground/scripts/decode_hpc_for_branch_comparison.py`):

| shape class | representative | purpose |
|---|---|---|
| 1D HPC | `j1620210710_ / 02_r1 / --mode 1d` | linearized, ~175 pos bins; minimal fusion stress |
| 2D HPC (production) | `j1620210710_ / 02_r1 / --mode 2d` | ~3420 pos bins; the shape that broke PR #19 |
| High tetrode count | synthetic simulated or `senor20201030_` (64-tet) | scan's target win case |

For each: bit-exact diff vs pre-PR kde_log; compile+first; steady-state
median; peak memory.  Both `clusterless_kde` and `clusterless_kde_log`
tested, cross-path parity checked.

### R5. Memory wins separately motivated

If `segment_sum` fusion (Task 4) is re-introduced, justify with a
workload where it matters (e.g. session that would OOM without it).
At 22-tetrode 2D / 80 GB A100, peak was 33 GB — memory wasn't
a constraint there.

---

## Verification Strategy

1. **Accuracy (gate):** end-to-end `max|diff|` vs pre-PR on the three
   shape classes.  Tolerance: ≤ 1e-10 for refactors, ≤ 1e-4 for log-space
   reorder noise on fp32, ≤ 1e-5 on fp64.
2. **Compile (gate):** first-predict compile time ≤ 20 min on every
   shape class in R4.
3. **Steady-state (informational):** report min/med/max over ≥3 runs
   with `block_until_ready`; flag regressions vs pre-PR at any shape.
4. **Memory (informational):** peak GPU bytes reported per config.

All of these land as `<label>_summary.json` + `comparison.txt`
artifacts attached to the PR description.

---

## Task breakdown (TBD — this plan is a brief, not a task list)

The detailed task decomposition depends on the router's exact design
(e.g. static estimator vs measured-first-call adaptive, one fused scan
body vs multiple smaller ones).  Proposed sequence when this plan is
picked up:

1. Router + compile-budget fallback infrastructure (no perf work yet)
2. Re-introduce Tasks 1/2/4 building blocks inside the per-electrode
   path with validation on all three shape classes
3. Investigate scan ptxas pathology (XLA flags, scan body size
   reduction, fusion barriers); ship scan only when compile-safe on
   all three shapes
4. `auto_select_tile_sizes` only if scan ships

---

## Lessons codified

Baked into `.claude/` memory (`real_data_pr_validation_workflow.md`):

- Always specify `--clusterless-algorithm` explicitly; don't rely on
  defaults.
- Always compare against previous main (detach to the commit before
  the PR landed), not just sibling branches.
- Always include cross-path parity (`kde` vs `kde_log` at the same SHA).
- Report compile-time as a first-class metric; flag hangs.
- Don't benchmark `predict_clusterless_kde_log_likelihood` directly —
  stripping surrounding `.predict()` fusion barriers produces a kernel
  ptxas can't handle (sunk 80 min of GPU time on this during PR #19).

## Related

- Original PR: #19 (merged `fcf5100`, reverted via #22)
- Regression diagnostic: #21
- Investigation branch (preserved): `gpu-opt-investigation`
- Streaming plan (orthogonal, not blocked): `2026-04-17-streaming-likelihood-into-hmm.md`
- Sorted-spikes plan (different file, same validation workflow applies):
  `2026-04-17-sorted-spikes-gpu-optimization.md`
