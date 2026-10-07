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
implicit full-session likelihood cache was added. The
[replay report](performance_artifacts/phase7/gpu-replay.json) (one SHA across
50 evaluations in each of eight clusterless cases) and
[gradient controls](performance_artifacts/phase7/gpu-gradient-controls.json)
measured the earlier serial loop; `scripts/check_likelihood_determinism.py`
reproduces the check for the current reduction (CPU: one SHA per algorithm
over 50 evaluations; GPU rerun pending).

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

The [replay/cache prototype](../scripts/benchmark_replay_cache.py)
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
The [published qualifier](../scripts/qualify_long_encoding.py) reproduces
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
  x64 on and off). Integer, mixed-dtype and weak scalar promotion remain
  unchanged.
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
