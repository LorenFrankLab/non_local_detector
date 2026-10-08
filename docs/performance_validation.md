# Phase 7 validation record

The frozen S6 qualification below records configurations on `feat/phase7-performance`,
based on merged Phase 6 (`fe1b2e9`). Subsequent CI-runtime corrections have a
separate validation record at the end of this document; earlier hour-scale
measurements retain their original source provenance. The [prediction guide](performance_prediction.md) describes
the API; the [phase plan](../.claude/docs/plans/likelihood-defect-remediation/phase-7-performance.md)
remains the acceptance checklist.

## Numerical evidence

The frozen Phase 6 numerical baseline passed 2,655 tests with six skips. A Git
archive omitted the generated version file; its separate import check passed
after restoring that artifact. Golden/snapshot files, numerical tolerances and
convergence criteria remain unchanged.

The frozen S6 candidate passes the complete suite: **3,137 passed, 26 skipped**,
including the shipped notebooks, consumer scripts and new encoding-tile
controls. A new one-dimensional local-KDE regression passes on current and
older runtimes. The [suite record](performance_artifacts/phase7/full-suite-s6.json)
identifies the exact source; the [built wheel](performance_artifacts/phase7/wheel-s6.json)
matches all 61 runtime modules and passes installed compact/spatial checks.

The user approved stable checkpointed evidence accumulation. For 1.8M float32
increments of `-0.02`, the original float32 carry produces `-35355.44921875`.
The actual input values summed in float64 give `-35999.999195337296`.
Checkpointed stable totals match that oracle across chunk groupings; posterior
arithmetic is unchanged. Explicit core reference mode retains the original
carry, and the dense driver keeps its previous accumulation.

CUDA default multiplication precision changed a reported state probability
from the posterior sum `0.9900000095` to `0.990234375`. Explicit `HIGHEST`
precision removes the excess TF32 error in the affected contractions.
Independent posterior-sum and intensity/gradient oracles and all 64 existing
shuffled/ragged likelihood cases pass at unchanged tolerances. Explicit dense
checkpoint products also request `HIGHEST`; their 12 CPU and 12 CUDA controls
pass with x64 enabled. No separate GEMV defect was reproduced.

CUDA float atomics also changed identical likelihood chunks between passes.
Spike-row reductions off CPU now use a segmented scan with a fixed reduction
tree and no scatters, so repeated evaluations of the same shapes are bitwise
identical without adding contributions one spike at a time; CPU keeps its
sequential scatter and verified sorted-index hint. Pairwise diffusion
component reductions are unchanged. Exact replay hashes remain required; no
implicit full-session likelihood cache was added. On an A100 (JAX 0.9.0), the
scan takes 0.3-3.2 ms per reduction from 16 to 200,000 spikes. The serial loop
it replaces takes 0.5-0.8 ms at 16 spikes, 18-31 ms at 2,000 and 1.5-1.6 s
at 200,000; the scan is similar at 16 spikes and 30-2,000x faster from 2,000.
From 2,000 spikes the scan is within about 2x of the nondeterministic scatter
(up to about 6x slower on sub-millisecond 16-spike calls); the scatter fails to
launch at 16,930 columns with sorted ids. An xprof trace of three calls at
20,000 x 500 shows 180,003 kernel launches for the serial loop, 177 for the
scan and 6 for the scatter. A 20,000-encoding-spike clusterless diffusion
likelihood chunk drops from 0.37 s to 0.06 s. Each new scan shape costs about
0.8 s to compile on the A100, so concrete off-CPU calls pad spike counts to
powers of two: 64 different spike counts create 7 executables (10.8 s) instead
of 64. `benchmarks/check_likelihood_determinism.py` gives one
SHA per case over 50 evaluations for four clusterless algorithms, default and
collision-heavy, with x64 off and on; the affected likelihood and checkpoint
modules pass on the A100 with x64 (444 tests). See the
[reduction benchmarks](performance_artifacts/deterministic_reductions/a100-reduction-benchmark-5a4047d8.json)
(baseline: [5bf46a29](performance_artifacts/deterministic_reductions/a100-reduction-benchmark-5bf46a29.json)),
[determinism](performance_artifacts/deterministic_reductions/a100-determinism-x64-on.json),
[xprof summary](performance_artifacts/deterministic_reductions/a100-xprof-20000x500.txt),
[bucketed recompilation](performance_artifacts/deterministic_reductions/a100-recompilation-bucketed.json) and
[end-to-end and test record](performance_artifacts/deterministic_reductions/a100-checks.txt).
The earlier [replay report](performance_artifacts/phase7/gpu-replay.json) and
[gradient controls](performance_artifacts/phase7/gpu-gradient-controls.json)
measured the serial loop.

Older-runtime testing exposed an existing GLM test-oracle error. Its float64
reference used `EPS`, although both `fe1b2e9` and this branch clip local rates at
`RATE_EPS_HZ`. Both implementations were bit-identical and agreed with the
correct independent oracle within `3.55e-15`; only the test reference was fixed.
The older tested compatibility stack passes **415 tests** with genuine global x64 on
Python 3.11.15, JAX/jaxlib 0.6.2, NumPy 1.26.4 and SciPy 1.12.0. The
[causality report](performance_artifacts/phase7/compatibility-oracle.json) and
[runtime manifest](performance_artifacts/phase7/compatibility-source.json)
record the exact reference correction and unchanged runtime bytes.

## Workloads and measurement

Native hour checks use 500 Hz, a 180 × 180 cm arena, four default states,
structured Cartesian transitions, chunks of 256 rows, float32 native arrays
and host float64 evidence. Actual padded combined dimensions are **16,930 at
2 cm** and **66,250 at 1 cm**. Benchmark interiors include the padding bins.
Both families use position bandwidth 6 cm and evaluation block size 10,000;
clusterless waveform bandwidth is 24 per feature. Tiled benchmarks retain
these values explicitly when supplying an algorithm parameter dictionary.

CPU checks use two units/electrodes at 5 Hz, two clusterless waveform
dimensions and ten seconds of encoding. The 2 cm runs return compact outputs;
1 cm runs write selections `[10, 10.02]` and `[3599.98, 3600]`, conditioned on
the complete hour.

A100 checks use 64 sorted units at 5 Hz or eight clusterless electrodes at
20 Hz with four waveform dimensions. Encoding lasts ten seconds. Both grids
write the full spatial output plus state probabilities, exercise bounded
first/last spatial reads, and delete only their own validated stores. One
spatial array is approximately 122 GB at 2 cm or 477 GB at 1 cm before metadata.

Budgets are 8 GiB process RSS and 8 GiB total device use per process. CUDA
preallocation is disabled; fraction 0.08 caps the A100 80 GB allocator at
6.4 GiB. NVML records context plus device allocation, separately from JAX
allocator counters. RSS sampling and OS high-water RSS include process inputs
and compiler; filesystem page cache is excluded. Disk sampling records both
allocated and apparent store bytes before cleanup.
Shared-host cache/dirty/writeback measurements are reported separately, without
attributing other users' activity to these jobs. These runs do not qualify an
8 GiB physical host, cold-cache reads, or storage durability/fsync throughput.
The [disk ledger](performance_artifacts/phase7/gpu-spatial-disk-allocation.json)
samples allocated store bytes before cleanup; its peaks are sampled lower
bounds, separate from exact logical sizes in each pipeline report.
[Shared-host memory](performance_artifacts/phase7/gpu-shared-system-memory.json)
and [ZFS ARC](performance_artifacts/phase7/gpu-zfs-arc.json) summaries retain
the filesystem-cache scope and late sampler start explicitly.

Bounded cases use at least five synchronized repetitions. Each full-hour case
uses one measured complete run after a 0.64-second warm-up covering 256/64-row
shapes. Qualification jobs run concurrently; elapsed times reflect that load.
Cross-platform long-encoding controls share workload configurations and seeds;
generated floating-point input hashes can differ across platforms. Each paired
untiled/tiled comparison uses identical inputs on its own platform.
Dense production references are rejected by preflight rather than allocated.
Completed full-hour measurements are below. Elapsed prediction time includes
likelihood evaluation, exact replay, output conversion and requested storage;
fit and cold warm-up are recorded separately. All eight hour checks passed.
Checkpoint cleanup is inside prediction. The benchmark's deletion of validated
output stores and its complete state-normalization check occur after the
prediction timer; wrapper elapsed times include that additional work.

| Runtime / family | Grid / output | Prediction minutes | Peak process GiB | Peak total device GiB |
| --- | --- | ---: | ---: | ---: |
| CPU / 2 sorted units | 2 cm / compact | 46.9 | 0.82 | — |
| CPU / 2 clusterless electrodes | 2 cm / compact | 49.2 | 0.98 | — |
| CPU / 2 sorted units | 1 cm / selected spatial | 111.4 | 1.18 | — |
| CPU / 2 clusterless electrodes | 1 cm / selected spatial | 112.6 | 1.30 | — |
| A100 / 64 sorted units | 2 cm / full spatial | 63.3 | 1.50 | 1.46 |
| A100 / 64 sorted units | 1 cm / full spatial | 72.8 | 1.90 | 1.71 |
| A100 / 8 clusterless electrodes | 2 cm / full spatial | 81.7 | 2.35 | 1.46 |
| A100 / 8 clusterless electrodes | 1 cm / full spatial | 120.1 | 2.73 | 2.21 |

The [2 cm sorted report](performance_artifacts/phase7/gpu-hour-sorted-2cm-spatial.json)
and [1 cm sorted report](performance_artifacts/phase7/gpu-hour-sorted-1cm-spatial.json)
record their source/input hashes and actual output sizes. The original hour
script times fit dispatch; subsequent benchmark versions explicitly synchronize
the fitted encoding arrays before reporting fit time. Hour inference and export
already synchronize their outputs.

## Measured choices

The [replay/cache prototype](../benchmarks/benchmark_replay_cache.py)
compares `T=1,024`, `N=256` and checkpoint lengths 64/256, with five interleaved
pairs on CPU and A100. Both paths produce bitwise-identical state probabilities
and stable evidence; the cache stores chunks on disk with at most 262,144
likelihood bytes in callback memory. Total logical likelihood cache is
1,050,624 bytes; checkpoint bytes are 32,768/8,192 for lengths 64/256.

| Checkpoint rows | CPU replay/cache ms | A100 replay/cache ms |
| --- | ---: | ---: |
| 64 | 28.460 / 29.821 | 276.264 / 256.850 |
| 256 | 18.730 / 19.180 | 129.023 / 126.067 |

The [CPU](performance_artifacts/phase7/cpu-replay-cache.json),
[A100](performance_artifacts/phase7/gpu-replay-cache.json) and
[summary](performance_artifacts/phase7/replay-cache-summary.json) record all
paired values, cold invocations, resource peaks and source/script hashes.
Process/device peaks remain below 2 GiB. The analytic callback is cheap and
cache reads use just-written files; these results do not qualify neural-backend
caching, cold-disk reads or durability. Production uses deterministic replay.

The sparse marked-block fix stops capacity at the actual mark count. Previously
two marks were padded to 10,000 rows, creating a 1.325 GB temporary at 1 cm.
An isolated five-pair comparison reduced median kernel time from 59.636 ms to
0.396 ms, preserving outputs. This is not a native pipeline speedup.

Compiled Local/No-Spike work reduced matched A100 64-unit, 320-row compact
medians from 2.805 s to 0.608 s at 2 cm, and 2.984 s to 0.766 s at 1 cm.
Input hashes match and state probability differences stay below `4.2e-7`.
These five-repeat sequential comparisons are against the initial Phase 7
candidate, not `main`; they are not interleaved full-hour speedup evidence.
Large legacy requests retain per-neuron workspace instead of tracing the
recording or allocating rows × population matrices.
The [paired values](performance_artifacts/phase7/gpu-batching-parity.json)
record exact input matching and posterior/evidence differences.

Weighted padding prototypes compare exact shapes, powers of two, multiples of
256 and powers of √2 in CPU/CUDA float32 and actual float64. Parity passes.
The 256 policy reduces cold compilation but increases warm time and resident
inputs. A100 float32 exact/256 medians are 27.84/32.67 ms; float64 medians are
30.74/35.64 ms. Production retains exact shapes; fitted samples and their
normalization are unchanged.
The [float32 CUDA report](performance_artifacts/phase7/gpu-buckets-f32.json),
[float64 CUDA report](performance_artifacts/phase7/gpu-buckets-f64.json), and
[CPU reports](performance_artifacts/phase7/cpu-buckets-f32.json) preserve all
five warm timings, shape counts and input/source hashes.

Optional linear-KDE encoding tiles pass 121 helper/integration controls on
current JAX 0.9 and tested JAX 0.6.2, including actual float64, independent
weighted-density/intensity references, exact tails, local combined kernels,
global row ownership, gradients and allocation guards. A compiled float32
scalar-tail gradient issue is fixed by rematerializing the exact tail; eager
and compiled gradients agree with the independent oracle at unchanged
tolerances. The [causality record](performance_artifacts/phase7/streamed-kde-gradient-causality.json)
retains the before/after values. Default fitted fields, likelihoods, posterior
and evidence remain bitwise equal to S5 in
[paired controls](performance_artifacts/phase7/default-kde-source-equivalence.json).

The [CPU long-encoding check](performance_artifacts/phase7/cpu-long-encoding-workspace.json)
uses one electrode at 20 Hz, four mark dimensions, 108,001 tracking samples and
72,000 encoding spikes over one hour, then decodes 0.64 seconds at 1 cm.
With encoding/position tiles of 1,024/8,192, synchronized fit takes 3.793 s,
five warm predictions have a 3.907 s median, and peak process RSS is 1.223 GiB.
This qualifies fit and short-prediction workspace; full-hour decoding with
hour-long training requires separate throughput qualification. The
[matching A100 check](performance_artifacts/phase7/gpu-long-encoding-workspace.json)
also passes: fit 7.361 s, five warm predictions median 0.914 s, process peak
1.674 GiB and total device peak 1.741 GiB.

[Large-encoding CPU controls](performance_artifacts/phase7/cpu-long-encoding-numerics.json)
compare 106/71 encoding tiles with independent float64 references in bounded
query subsets. Float32 local/joint log errors are at most `7.55e-6`/`1.93e-6`;
float64 log error is at most `2.49e-14`. Density ratio diagnostics retain up to
`1.85e-5` relative float32 error, also present in the original backend's
weighted reduction. The [isolation report](performance_artifacts/phase7/long-encoding-rounding-isolation.json)
distinguishes this existing precision limit from changes caused by tiling.
Applicable tolerances remain unchanged. The
[matching A100 numerical controls](performance_artifacts/phase7/gpu-long-encoding-numerics.json)
also pass: float32 local/joint log errors at most `1.70e-6`/`1.77e-6`, and
float64 log errors at most `3.55e-15`.
The [published qualifier](../benchmarks/qualify_long_encoding.py) reproduces
the seeded inputs and independent references without allocating a full
encoding-by-arena matrix.

Five interleaved S6 untiled/tiled comparisons preserve state probabilities
and evidence at unchanged tolerances. On the small matched case, median CPU
prediction is 26.920/31.980 ms and A100 prediction is 273.012/273.654 ms.
Tiling bounds the long-encoding workspace; these
[CPU](performance_artifacts/phase7/cpu-streamed-kde-interleaved.json) and
[A100](performance_artifacts/phase7/gpu-streamed-kde-interleaved.json)
measurements record its small-case overhead.

## Limits and provenance

Hour claims apply to the recorded sorted KDE and linear clusterless KDE
configurations. Small log/GMM/diffusion/MRF correctness controls do not qualify
their large spatial workspaces. Encoding-event × spatial-bin kernels, compiled
executables, inputs, compact outputs and recording metadata need separate
budgets. Ten-second encoding does not qualify large encoding populations;
encoding-kernel streaming is an explicit separate option with its own checks.
Smaller GPUs, full-hour covariate workloads, EM and Viterbi need separate
qualification. Phase 8 volume/MRF work remains separate.
Declared dependency floors (Python 3.10, JAX 0.4.27, NumPy 1.25 and SciPy 1.10)
were not exercised by the older-stack checks.

GPU snapshot S4 contains 60 Python files with aggregate SHA-256
`11baf69a8a8c84c9974d9f44b768b9583e098a576ad0ac7b4cdbacefb073112b`.
CPU hour runs use S2. S2→S4 changes only inactive large-row fallback guards
and dense checkpoint precision; the 256/64-row structured hour paths have
identical executed kernels and equivalent dispatch. Later branch docstring
corrections are nonfunctional. Manifests distinguish these snapshots rather
than claiming all reports came from one identical checkout.
S6 adds only opt-in linear-KDE tiling to the S5 runtime and includes 61 Python
files, aggregate SHA-256
`8b96d773cf5cdb5ae678facfe04fdd30edb096fbb5dbb97b848134695bfab13c`.
Its [manifest](performance_artifacts/phase7/source-s6.json) identifies the
complete frozen source and synchronized benchmark script.
See the [provenance audit](performance_artifacts/phase7/qualification-provenance.json)
and [source-equivalence checks](performance_artifacts/phase7/source-equivalence.json).

## CI-runtime corrections after S6

The Python 3.13 CI run reproduced six failures with JAX/jaxlib 0.11.2,
NumPy 2.5.3 and SciPy 1.18.1. The other Python jobs were cancelled by fail-fast.
The corrections preserve existing tolerances, golden files and convergence
criteria:

- The legacy evidence-semantics test explicitly selects reference accumulation.
  Stable checkpointed evidence retains its independent float64-sum tests.
- Sorted KDE/GLM emissions keep the fixed 64-row matrix accumulation. Its CI
  difference from the per-neuron float32 reference (decoder posterior 4/5130
  elements, max 3.457e-05) is attributed to the reference's own rounding: in
  a local arm64 run, the same model with the float64 sum of the same float32
  inputs, rounded once, differs from the per-neuron order by exactly that
  pattern for the decoder (4/5130, 3.45706940e-05) and within 0.4% for the
  non-local model (64 vs 89 elements, 1.708e-05 vs 1.702e-05). The CI x86
  matrix output itself was not rerun. The test now requires emissions within
  16 float32 ulp of that float64 sum (6.3 ulp measured) and posteriors no
  further from the float64-emission run than the per-neuron reference. Integer counts
  enter `xlogy` as floats, and uniform-duration blocks keep each row's own
  duration derivative; both fixes leave forward values bitwise unchanged.
- Singleton Gaussian tails materialize their broadcast divisor before division.
  A broader guard for float32 kernels under enabled float64 materialized a
  kernel-sized buffer (160 MB at 20000 x 1000) and was removed after the
  affected KDE modules passed without it on the CI x86 stack (JAX 0.11.2,
  x64 on and off; [record](performance_artifacts/deterministic_reductions/x86-ci-stack-gate.txt)).
  Integer, mixed-dtype and weak scalar promotion remain unchanged.
- Structured Gaussian products cap only extreme finite normalization scales,
  preventing a compiler-hoisted reciprocal from flushing to zero. Ordinary
  scales keep their original arithmetic.
- Transformed tiled-density calls use fixed evaluation-query tiles to preserve
  compiled tail gradients. Concrete native calls retain their original exact
  query-tail shapes and rounding. Encoding samples, weights and normalization
  are never padded. Transformed and concrete primals can differ slightly in
  floating-point rounding; compiled values and derivatives are checked against
  independent analytic references at the existing tolerances.

The [correction manifest](performance_artifacts/phase7/ci-fix-source.json)
identifies the updated runtime. Local full-suite candidates passed **3,155 tests,
26 skipped**. Follow-up controls cover the final Gaussian and query changes.
On the exact CI CPU stack, all **351 enabled affected tests passed, 20 skipped**;
the final query change additionally passed all 121 density/native controls.
An A100 passed all **371 affected tests with actual float64 enabled**, followed
by all 121 controls for the final query change, including the compiled-primal
regression. The [test record](performance_artifacts/phase7/ci-fix-validation.json)
distinguishes each tested snapshot. The final source also passes 265 numerical
controls on the older tested Python 3.11/JAX 0.6.2 stack with actual float64.
A zephyr full-suite attempt reported LLVM
allocation failures; it is not recorded as a successful full CI-stack run.

Updated-source native A100 checks use 2,000 decoding rows, 66,250 combined
hidden bins, the same 64-unit/eight-electrode populations and ten seconds of
encoding as the earlier GPU qualification. Posterior comparisons to S6 pass
at unchanged `rtol=atol=1e-6`, with maximum absolute differences of `2.39e-7`
(sorted) and `4.77e-7` (clusterless). The sorted output also matches an
independent eager per-neuron reference within `3.28e-7`. Process RSS peaks are
1.72/2.27 GiB; JAX allocator peaks are recorded separately and remain below
8 GiB. These are short compact checks, not repeated hour-scale spatial exports
or NVML measurements. The [native comparison record](performance_artifacts/phase7/ci-fix-native-comparison.json)
contains exact source/input hashes, timings, resource counters and reference
override provenance.

An ordered sparse emission briefly replaced the matrix path; its
[CPU](performance_artifacts/phase7/ci-fix-sorted-cpu.json) and
[A100](performance_artifacts/phase7/ci-fix-sorted-gpu.json) kernel timings
(1.49×/2.26× slower than the matrix path for sparse chunks, 9.02×/13.5× for a
simultaneous burst) are retained as historical measurements of that reverted
variant. The earlier full-hour timings and built-wheel checks remain
measurements of S2/S4/S6 and are not relabelled as qualification of the
updated runtime.

## Comparison with main

Same simulated workloads on `main` (8fdec995) and this branch, run with
`benchmarks/compare_prediction_modes.py`: a 180 cm 2-D arena, 64 sorted units
at 5 Hz or 8 clusterless electrodes at 20 Hz with 4 marks, 10 s of encoding.
Times are steady-state warm predictions where available, otherwise the first
prediction (including compilation). `main` dense is the default `predict`;
branch compact is `inference_mode="checkpointed", output_mode="compact"` with
structured transitions. Branch dense and chunked modes match `main` in time
and memory, and `main`'s `n_chunks` does not reduce memory.

| A100, 2 cm (16,930 state bins) | main dense | branch compact | ratio |
| --- | ---: | ---: | ---: |
| Sorted 30 s / 120 s | 40.3 s / 147.9 s | 25.8 s / 103.4 s | 1.56x / 1.43x |
| Clusterless 30 s / 120 s | 39.0 s / 147.0 s | 46.1 s / 187.1 s | 0.84x / 0.79x |
| Device peak at 120 s | 21.5 GB | 0.15 GB | |
| Host RSS at 120 s | 26.3 GB | 1.5-2.4 GB | |
| Fit | 235-273 s | 2.6-3.9 s | |

| CPU (64 GB), main dense vs branch compact | sorted | clusterless |
| --- | ---: | ---: |
| 2 cm, 30 s | 645 s vs 18.0 s (36x) | 594 s vs 23.0 s (26x) |
| 4 cm, 30 s to 16 min | 10.4-11.2x faster | 6.5-7.0x faster |
| 4 cm, 16 min host RSS | 26.1 GB vs 1.6 GB | 29.9 GB vs 4.6 GB |
| Fit, 2 cm | 168.5 s vs 1.1 s | 165.7 s vs 0.9 s |

Capacity: on the A100, dense prediction (either version) at 2 cm completes
4 minutes (41.8 GB device, 39 GB host) and fails at 8 minutes (out of device
memory). Branch compact completes 60 minutes with 0.15 GB device memory:
3,024 s for sorted and 5,603 s for clusterless (1.6/2.6 GB host). On the
64 GB CPU at 4 cm, `main` dense completes 16 minutes (26-30 GB) and 32 minutes
would exceed RAM, while compact completes 60 minutes in 644 s (sorted) and
943 s (clusterless) with at most 4.5 GB.

Where compact time goes on the A100 (30 s, 2 cm; `benchmarks/profile_checkpointed_prediction.py`):
sorted spends about 50% in the per-step forward/backward kernels, 35% in
host-side work and 15% in likelihoods; clusterless spends 57% in likelihoods
(both passes), 26% in forward/backward kernels and 17% host-side. Host
profiles show SHA-256 replay digests at 15% (sorted) and 7% (clusterless) of
the prediction, and about 34,000 eager JAX dispatches per clusterless
prediction from likelihood code outside `jit`. An xprof trace of a 2 s compact
prediction attributes 55% of kernel time to the per-step structured
transition product, with the GPU busy about half the time. The
[bottleneck fixes](#bottleneck-fixes-after-the-comparison) below address these.

Agreement and precision: the branch's state probabilities match `main` to
2e-6 (CPU, dense) and 6e-4 (CPU, compact); on the A100 both versions differ
by up to about 2e-3, the same size as `main`'s own dense-versus-chunked
difference there. Against a float64 reference, float32 state probabilities
in both versions err by up to about 5% on CPU and 1.8% on the A100 for this
workload; the source of that float32 error has not been isolated.

Records: [A100 runs](performance_artifacts/main_vs_branch/a100-runs.json),
[CPU runs](performance_artifacts/main_vs_branch/cpu-runs.json),
[stage profiles](performance_artifacts/main_vs_branch/a100-stage-profiles.json),
[xprof summary](performance_artifacts/main_vs_branch/a100-xprof-compact-2s.txt) and
[agreement and precision](performance_artifacts/main_vs_branch/agreement-and-precision.txt).
Runs marked failed with `UnboundLocalError` completed their prediction; an
earlier version of the comparison script could not save states with
`--repeat 0`.

## Bottleneck fixes after the comparison

The comparison above found per-step transition kernels, replay digests, host
round trips and per-chunk likelihood overhead dominating compact prediction.
These changes address them, measured on the same workloads (30 s, 2 cm,
16,930 state bins; A100 with JAX 0.9.0, and the development Mac CPU):

| Warm compact prediction, 30 s | A100 sorted | A100 clusterless | CPU sorted | CPU clusterless |
| --- | ---: | ---: | ---: | ---: |
| Before (09d1d67a, 256-row chunks) | 25.8 s | 47.4 s | 18.0 s | 23.2 s |
| Likelihood chunks kept on the device, checksum replay, pipelined host work | 14.0 s | 40.2 s | 16.5 s | 21.7 s |
| plus compiled local likelihood row blocks, 4,096-row chunks | 11.6 s | 12.3 s | 16.3 s | 18.2 s |
| plus fused rank-one transition blocks, 4,096-row chunks | 2.9 s | 4.3 s | 16.4 s | 17.6 s |
| plus the byte-based default chunk size (3,963 rows) | 2.9 s | 4.2 s | 15.8 s | 16.8 s |

- **Replay.** Likelihood chunks stay on the device. Missing-row neutralization,
  degenerate/NaN masks and an order-independent uint32 checksum are computed in
  one jitted call, replacing a host SHA-256 and each chunk's round trip through
  host memory, and each chunk's host work overlaps device work on the next. CPU state probabilities
  are bitwise unchanged, as are A100 ones with autotuning disabled.
- **Local likelihood row blocks.** Sorted local and no-spike likelihoods above
  256 rows fell back to per-neuron uncompiled loops, so larger chunks were
  slower for sorted spikes (31.2 s at 1,024 rows). They now run compiled over
  row blocks within the spike-count budget, bitwise unchanged.
- **Fused transition blocks.** 15 of the 16 blocks of the default non-local
  operator are rank one. Folding them into two matrix products per step cut an
  A100 forward step from 215 to 55 µs and a backward step from 311 to 78 µs
  (`benchmarks/benchmark_checkpoint_scan_steps.py`). The summation order changes:
  state probabilities moved by at most 5.4e-7 on the A100 with autotuning
  disabled and 7.3e-6 on CPU, with the same most likely state at every bin and
  rows still summing to 1 within 4.8e-7. Operator products and filter/smoother
  outputs still match the dense reference within the existing tolerances.
- **Chunk size.** The default now targets 256 MiB of float32 likelihoods per
  chunk (at least 256 rows). For this grid the A100 device peak rose from
  0.15 GB at 256 rows to 1.48 GB at 3,963 rows, with host RSS unchanged
  (1.3-2.5 GB). On CPU, where chunks live in host memory, peak RSS rose from
  0.9 to 3.4 GB (sorted) and from 2.8 to 4.8 GB (clusterless). First A100
  predictions, including compilation, fell from 59.7 to 23.5 s (sorted) and
  from 104.3 to 56.9 s (clusterless).

Hour scale on the A100, first prediction including compilation: sorted 60 min
took 363.9 s (3,023.8 s before) and clusterless 60 min took 680.8 s
(5,603.5 s before), with device peaks of 1.48 and 1.36 GB. Host RSS was 1.55 GB
for sorted (1.60 GB before) and 4.59 GB for clusterless (2.58 GB before); the
clusterless increase was not attributed. Warm cost is
linear: 0.096 s per recording second for sorted and 0.14 s for clusterless.

Not adopted, with measurements in the records below: unrolling the
forward/backward scans (14.6 s at unroll 2 versus 14.0 s, and slower
compilation at 4 and 8); XLA command buffers for while loops (no change);
removing the restriction wrappers or adding optimization barriers to per-block
NaN checks (slower on the A100); `--xla_gpu_deterministic_ops=true` (a
prediction had not finished after 13.5 minutes, versus about 1.5 minutes).

Separate A100 processes are not bitwise reproducible by default: XLA
autotuning chooses kernels per process, and state probabilities from repeated
runs of unchanged code differed by up to 3.1e-4. With
`XLA_FLAGS=--xla_gpu_autotune_level=0`, three runs were bitwise identical at
the same speed.

What remains: on the A100, per-step forward/backward kernels take 87% of a
sorted prediction and 70% of a clusterless one (about 20 fusions and 2-4 cuBLAS
calls per step). Clusterless likelihoods take about 1 s of 4.2 s, and their
first prediction compiles one program per new spike-count shape: 1,275
compilations and 137.9 s for a 120 s recording, versus 17.1 s once warm.

Records: [A100 runs](performance_artifacts/bottlenecks/a100-runs.json),
[CPU runs](performance_artifacts/bottlenecks/cpu-runs.json),
[A100 reproducibility](performance_artifacts/bottlenecks/a100-reproducibility.json),
[A100 scan steps](performance_artifacts/bottlenecks/a100-scan-steps.json) and
[A100 stage profiles](performance_artifacts/bottlenecks/a100-stage-profiles.json).

## Scaling with data size

One factor at a time from the 30 s baseline above (64 sorted units at 5 Hz or 8
clusterless electrodes at 20 Hz, 10 s of encoding, 2 cm grid), A100, default
chunk size, code at 1f777c84:

| Factor | Range | Warm, sorted / clusterless | First call, sorted / clusterless |
| --- | --- | --- | --- |
| Encoding length | 10 s to 30 min | 2.95 to 3.22 s / 4.42 to 5.00 s | 25 to 34 s / 67 to 95 s |
| Decoding spike rate | 4× | 2.99 s / 4.39 s | 27 s / 80 s |
| Sorted units | 64, 256, 1,024 | 2.95, 3.23, 5.16 s | 25, 55, 232 s |
| Clusterless electrodes | 8, 32, 128 | 4.42, 8.13, 16.85 s | 67, 121, 169 s |
| Grid | 4, 2, 1 cm (4,420 to 66,250 bins) | 2.48, 2.95, 4.30 s / 2.62, 4.42, 9.09 s | 18, 25, 29 s / 15, 67, 79 s |

Device peaks stayed at 1.35-1.50 GB, because default chunks shrink from 15,183
to 1,012 rows as grids grow, except with 30 min of encoding: 3.7 GB (sorted)
and 5.1 GB (clusterless), measured over the whole process including fitting.
Recording length scales linearly, as in the hour-scale runs above.

Population size drove first-call time through compilation. With 1,024 sorted
units, 195 s of a 235 s first prediction compiled the local KDE kernel, whose
graph grows with the number of units. JAX's persistent compilation cache cut a
second process's first prediction to 44 s when shapes repeated exactly; a 33 s
recording, whose last chunk differs, still took 151 s. With 128 electrodes,
126 s of a 179 s first prediction went to 1,698 compilations, and warm
likelihoods made 35,000 eager dispatches.

Compiling each electrode's clusterless terms with padded spike counts
(a5dc6e3a) addresses the clusterless case. Rates spread 8× across electrodes
(`--rate-spread 3`) mimic electrodes with different encoding counts. Each row
pairs the previous and compiled code in runs made at the same time; the A100
pairs ran ten at once on a second A100 host, so the previous code's 128-electrode
times differ from the sweep above (16.85 s warm, 169 s first call):

| Clusterless, 30 s unless noted | First call | Warm | Host RSS |
| --- | ---: | ---: | ---: |
| A100, 8 electrodes | 66 to 25 s | 4.3 to 3.2 s | 2.4 to 1.5 GB |
| A100, 128 electrodes | 182 to 29 s | 22.0 to 6.9 s | 4.3 to 1.5 GB |
| A100, 128 electrodes, rates spread 8× | 953 to 131 s | 24.4 to 5.6 s | 7.0 to 2.4 GB |
| A100, 600 s, rates spread 8× | 768 to 139 s | 77 to 62 s | 6.3 to 1.9 GB |
| CPU, 8 electrodes | 30 to 19 s | 17.0 to 16.4 s | 4.8 to 3.3 GB |

For a 120 s recording with 128 spread electrodes, the previous code had not
finished its first prediction after 3,000 s; the compiled version took 152 s,
114 s of it compiling 43 variants of each kernel. Coarser encoding padding
compiles less but computes more: two sizes per octave gave 73 s of compilation
and 6.6 s warm, powers of two 52 s and 7.4 s, against 114 s and 5.6 s at four
per octave. State probabilities moved by at most 3.0e-6 on CPU, with the same
most likely state at every bin, and error against a float64 computation was
unchanged.

Grouping sorted units by padded encoding-sample count and evaluating each group
in batches of 16 (d36234c8) bounds the local KDE kernel's graph. On the A100
(paired runs, medians of three at 1,024 units):

| Sorted, 30 s unless noted | First call | Warm |
| --- | ---: | ---: |
| 64 units | 24.5 to 19.1 s | 2.93 to 3.00 s |
| 256 units, rates spread 8× | 86.7 to 40.9 s | 3.14 to 3.27 s |
| 1,024 units | 220 to 22 s | 4.29 to 4.41 s |
| 1,024 units, rates spread 8× | 281 to 43 s | 4.24 to 4.26 s |
| CPU, 64 / 256 spread units | 19.4 to 17.3 s / 31.5 to 18.3 s | +0.8% / +0.6% |

For a 120 s recording with 1,024 spread units, compilation fell from 254 to 34 s
and the first prediction from 296 to 58 s. Without batching, compilation was
faster (19 s) but warm predictions at 1,024 units took 4.81 s, 12% longer
than the previous code's 4.29 s.
Local log-likelihoods changed by at most 5.7e-7 relative and state
probabilities by at most 2.3e-6 on CPU, with the same most likely state at
every bin; error against a float64 computation was unchanged.

Records: [scaling runs](performance_artifacts/scaling/a100-scaling-runs.json),
[diagnostics and cache test](performance_artifacts/scaling/a100-scaling-diagnostics.txt),
[clusterless runs](performance_artifacts/scaling/a100-clusterless-runs.json),
[CPU clusterless runs](performance_artifacts/scaling/cpu-clusterless-runs.json),
[compilation counts](performance_artifacts/scaling/a100-clusterless-compiles.txt),
[128-electrode profile](performance_artifacts/scaling/a100-profile-clusterless-128.json),
[numerical comparison](performance_artifacts/scaling/numerics-clusterless-compiled.txt),
[sorted runs](performance_artifacts/scaling/a100-sorted-runs.json),
[CPU sorted runs](performance_artifacts/scaling/cpu-sorted-runs.json),
[sorted compilation](performance_artifacts/scaling/a100-sorted-compiles.txt) and
[sorted numerical comparison](performance_artifacts/scaling/numerics-sorted-compiled.txt).

### Compiled GMM and log-space KDE likelihoods

`clusterless_gmm` and `clusterless_kde_log` now evaluate each electrode's
local and non-local terms in one compiled call per chunk, as the default
clusterless KDE does, with decoding spikes padded to a few sizes. Log-space
KDE also pads encoding samples with zero weights. Before, the GMM score
recompiled for each spike-block shape, and the log-space KDE dispatched
separately compiled operations whose shapes changed with every electrode's
spike count. A100, 30 s at 2 cm, 8 electrodes at 20 Hz (GMM with 120 s of
encoding, because its fit is singular with 10 s; log-space KDE with 10 s):

| Algorithm | First call | Compilations | Warm |
| --- | ---: | ---: | ---: |
| `clusterless_gmm` | 114 to 33 s | 521 to 90 | 4.42-4.45 to 4.08-4.22 s |
| `clusterless_gmm`, autotuning off | 78 to 23 s | 521 to 90 | 4.49-4.55 to 4.01-4.20 s |
| `clusterless_kde_log` | 77 to 28 s | 857 to 83 | 4.34-4.47 to 3.23-3.27 s |
| `clusterless_kde_log`, autotuning off | 62 to 21 s | 857 to 83 | 4.44-4.56 to 3.21-3.25 s |

GMM non-local spikes are scored at every bin, so their padding sets the extra
work. Padding them to powers of two, as the KDE paths do, left the A100 warm
time within autotuning variation (4.52-4.60 s with default autotuning,
4.36-4.39 s without) but made CPU warm predictions 6-9% slower. Four sizes per
octave (at most 25% padding up to one block) removed that cost: on CPU, 10 s
at 4 cm took 2.64-2.73 s against 2.66-2.70 s before, interleaved, with
bitwise-identical state probabilities. Log-space KDE on CPU (30 s at 2 cm)
took 17.8 s instead of 38.4 s for the first prediction and 15.8 s instead of
21.0-21.3 s once compiled.

With XLA GPU autotuning off (`XLA_FLAGS=--xla_gpu_autotune_level=0`), both
versions use the same kernels: GMM state probabilities moved by at most
5.1e-7, and log-space KDE state probabilities were bitwise identical. With the
default autotuning, separate runs of the new GMM code differed by up to
3.7e-4, as much as old-versus-new, with the same most likely state at every
bin. Against an eager per-spike reference, likelihoods agree to 2.7e-7 (GMM)
and 3.7e-7 (log-space KDE) relative in float32, and to 6.3e-16 in float64, on
JAX 0.9.0 and 0.11.2. The GMM `bin_tile_size` path and the log-space KDE
streaming and encoding-tiled paths are unchanged.

Records: [A100 runs and state comparisons](performance_artifacts/compile/a100-gmm-kde-log-runs.json)
and [CPU runs](performance_artifacts/compile/cpu-gmm-kde-log-runs.json),
measured with `benchmarks/profile_compilations.py` (the A100 runs before the
final GMM runs used an earlier copy that took the benchmark directory as an
argument).
