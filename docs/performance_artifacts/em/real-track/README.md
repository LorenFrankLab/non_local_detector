# EM on a real recording: linearized track, A100

`benchmarks/profile_em.py` reports for `estimate_parameters` on run epoch
`02_r1` of `j1620210710_.nwb`: 23.6 min at 500 Hz (709,319 decode rows), a
10-node linearized track at 2 cm (514 state bins, 498 interior), 3 EM
iterations. Sorted runs use 305 CA1 units (1.63 M spikes); clusterless runs use
22 tetrodes with 4-D marks (418 k spikes). Every run used one NVIDIA A100 80GB
PCIe on breeze, JAX 0.9.0 with CUDA 12, float32, default XLA autotuning unless
noted, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The recordings were exported from the state-space-playground session cache with
`export_session.py` (this directory). Each report records the exporter hash,
window, package-source hash and device.

## Code states

| `package_source_sha256` prefix | Code |
| --- | --- |
| `b0ac0a76` | Before this work (`ddcbe6e`) |
| `8b3d936b` | Plus the jittered-duration matrix emission (`2ad9c07`) |
| `63157abd` | Plus KDE byte-budgeted blocks, scan unroll and the contraction M-step (`927f1c8`, `21563df`, `925b6ac`) |
| `54a54858` | `cc25c63`: plus the linearization cache, single-bincount counts and compiled many-block KDEs |
| `0173cb67` | `8b3d936b` plus the KDE byte budget (clusterless baseline only) |
| `095d5cbb` | `c23aea7`, before the likelihood-defect remediation plan (run via `PYTHONPATH`) |
| `75660e30` | `cc25c63` plus the row-blocked M-step (`7a2c64b`) |
| `309b74fc` | `a789beb`: plus the track-graph GLM fix |
| `6559795f` | `104e024`: plus compiled per-electrode diffusion prediction |
| `273a29a3` | `6559795f` plus padded weighted GMM refits |
| `309b74fc` (GLM) | `a789beb`, per-cell BFGS GLM fits |
| `2ba28161` | `ddeaa01`: batched Newton GLM fits |
| `27957feb` | `3925f84` before its review fixes (validation and docs only): structured-transition EM |
| `1edc8ee5` | `27957feb` plus unrolled checkpointed scans (`1ad6a90`) |
| `9680b2d4` | `a69d107`: plus structured-EM result stores and the replay cache |
| `f21609be` | `d7d68fb` (dense EM, used for the KDE scaling sweep) |
| `52068f24` | `f4dd609`: rebased, with the count-scaling and KDE-dtype fixes and per-run memory release in the profiler |
| `ac120b53` | `52068f24` plus the prediction replay cache (`005fc10`) |

## Full recording (709,319 rows)

| Report | Code | First call | Warm (3 iterations) | Device peak | Host peak |
| --- | --- | ---: | ---: | ---: | ---: |
| `sorted-track-full` | `8b3d936b` | 542.2 s | 261.9 s | 53.0 GB | 15.8 GB |
| `sorted-track-full-fixes` | `63157abd` | 434.1 s | 192.0 s | 22.4 GB | 15.6 GB |
| `sorted-track-full-cc25c63` | `54a54858` | 403.9 s | **168.4 s** | 22.5 GB | 15.8 GB |
| `clusterless-track-full-oom-before-kde-budget` | `8b3d936b` | out of memory (105.7 GiB requested) | — | — | — |
| `clusterless-track-full-fixes` | `63157abd` | 341.2 s | 141.5 s | 10.0 GB | 13.8 GB |
| `clusterless-track-full-cc25c63` | `54a54858` | 328.3 s | **131.4 s** | 10.0 GB | 14.0 GB |

The `clusterless-track-full` report (source `0173cb67`: `8b3d936b` plus the
KDE byte budget, copied in before it was committed) is the clusterless
baseline: first call 423.6 s, warm 225.6 s, device peak 10.0 GB. Without the
byte budget the clusterless recording does not fit.

Warm stage times per E-step or iteration (instrumented warm run, which
synchronizes at stage boundaries):

| Stage | Sorted before → final | Clusterless before → final |
| --- | --- | --- |
| Filter/smoother, per E-step | 32.9 → 21.0 s | 32.7 → 19.4 s |
| Likelihood, per E-step | 15.4 → 13.3 s | 11.2 → 9.8 s |
| Discrete M-step, per iteration | 11.0 → 0.5 s | 11.0 → 0.5 s |
| Encoding M-step, per iteration | 8.5 → 4.9 s | 2.6 → 1.0 s |

The final runs' `.cprofile.txt` files list the host functions with the most own
time in an uninstrumented warm run.

### Against the code before the remediation plan

| Report | First call | Warm (3 iterations) | Device peak | Host peak |
| --- | ---: | ---: | ---: | ---: |
| `preplan-sorted-track-full` | 824.1 s | 247.9 s | 53.0 GB | 16.4 GB |
| `preplan-clusterless-track-full` | 491.9 s | 209.8 s | 53.0 GB | 17.1 GB |

The final code is 32% (sorted) and 37% (clusterless) faster warm, 51% and
33% faster on the first call, and uses 2.4x and 5x less device memory. The
plan's corrections change the fitted values, so these compare cost only.

## 2-D grid of the same recording

`--environment 2d` at 2 cm: 20,178 grid bins, 3,298 interior state bins.

| Report | Outcome |
| --- | --- |
| `*-2d-full-dense-oom` | Default dense EM runs out of device memory in the first E-step (`cc25c63`) |
| `*-2d-full-chunks8-mstep-oom` | With `n_chunks=8` the E-step fits; the full-length M-step contraction of `925b6ac` did not |
| `sorted-2d-full-chunks8` | Row-blocked M-step: first call 870.1 s, warm 503.4 s; device 56.4 GB, host 175.2 GB |
| `clusterless-2d-full-chunks8` | First call 898.5 s, warm 499.9 s; device 56.3 GB, host 174.7 GB |

The filter/smoother takes 68-70% of the warm time (about 87 s per E-step).
The 175 GB host peak comes from dense EM keeping full-length posteriors; the
pre-plan code was not run in 2-D.

### Clusterless diffusion on the 2-D grid

With `n_chunks=8`, `clusterless_diffusion` prediction dispatched its per-block
operations separately, and the shapes varied with every block, chunk and
electrode. They overflowed JAX's per-operation cache, so every warm E-step
recompiled. Compiling each electrode's prediction as one call (`104e024`)
removes the recompilation:

| Report | Code | Host | First call | Instrumented warm (3 iterations) | Warm likelihood compiles | Device peak | Host peak |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `diffusion-2d-full-chunks8-before` | `75660e30` | breeze | 4043.3 s | 4018.5 s | 29,152 (2387 s) | 56.3 GB | 178.1 GB |
| `diffusion-2d-full-chunks8` | `6559795f` | zephyr | 907.8 s | **695.0 s** | 0 | 56.3 GB | 174.4 GB |

The likelihood stage went from 3233.9 s to 16.6 s per warm run. The track
with `n_chunks=8` showed the same recompilation (unchunked, it never
recompiled):

| Report | Code | Host | First call | Instrumented warm | Warm likelihood (compiles) | Device peak | Host peak |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `diffusion-track-full-chunks8-before` | `309b74fc` | breeze | 3590.5 s | 3063.0 s | 2911.7 s (26,965) | 8.8 GB | 20.1 GB |
| `diffusion-track-full-chunks8` | `6559795f` | zephyr | 510.4 s | **113.4 s** | 16.3 s (0) | 8.7 GB | 16.4 GB |
 The two runs
used different hosts with the same GPU model: the new run could not fit on
breeze while other jobs held about 33 GB of each GPU. The like-for-like check
is a 120 s window on one breeze GPU with `XLA_FLAGS=--xla_gpu_autotune_level=0`
and 2 iterations:

| Report | Code | First call | Instrumented warm | Host peak |
| --- | --- | ---: | ---: | ---: |
| `diffusion-2d-120s-chunks8-autotune0-before` | `309b74fc` | 687.6 s | 69.3 s | 26.2 GB |
| `diffusion-2d-120s-chunks8-autotune0` | `6559795f` | 319.3 s | 43.8 s | 22.8 GB |

Their state probabilities differ by at most 7.8e-5, with the same most likely
state in all 59,999 bins.

### Clusterless GMM refits

`clusterless_gmm` with `clusterless_algorithm_params={"gmm_reg_covar": 1.0}`
(the default `1e-6` stops with a singular covariance on this recording),
track, 3 iterations, both runs on zephyr:

| Report | Code | First call | Instrumented warm | Encoding M-step (compiles) | Host peak |
| --- | --- | ---: | ---: | ---: | ---: |
| `gmm-regcovar1-track-full-before` | `6559795f` | 619.7 s | 439.5 s | 303.5 s (1,494) | 19.4 GB |
| `gmm-regcovar1-track-full` | `273a29a3` | 285.5 s | **159.1 s** | 43.0 s (216) | 16.2 GB |

Device peak was 10.0 GB in both. The fitted posteriors are not comparable
run to run: from the second iteration the marginal log likelihood moves by up
to 1e6 between refits, and `gmm-regcovar1-track-full-breeze-a789beb` (the
previous GMM code on breeze) differs from the zephyr run of the same code in
the most likely state at 156,736 of 709,319 bins, about as much as the
previous and new code differ on zephyr (143,210).

### Sorted-spike GLM

`--algorithm sorted_spikes_glm`, track, 3 iterations. Each EM iteration refits
all 305 cells' GLMs.

| Report | Code | Host | First call | Instrumented warm |
| --- | --- | --- | ---: | ---: |
| `glm-track-full-before` | `309b74fc` | breeze | 12,791.5 s | stopped |
| `glm-track-full` | `2ba28161` | zephyr | 240.8 s | **217.0 s** |

The previous code fitted each cell with SciPy BFGS (about 10 s per cell, 55
minutes per refit); its instrumented run was not completed.

## Structured-transition (checkpointed) EM

`--transition-representation structured` (2-D) or `auto` (track graph, which
needs a dense fallback block), clusterless 2-D and sorted track, 3 iterations,
default chunk size. All on zephyr; the dense comparisons ran there too.

| Report | Code | First call | Instrumented warm | Device peak | Host peak |
| --- | --- | ---: | ---: | ---: | ---: |
| `dense-2d-full-chunks8-zephyr` | `27957feb` | 863.1 s | 500.6 s | 56.3 GB | 184.1 GB |
| `structured-2d-full` | `27957feb` | 1,117.8 s | 705.9 s | **2.5 GB** | 128.4 GB |
| `auto-track-sorted-full` | `27957feb` | 697.7 s | 423.4 s | 24.8 GB | 11.1 GB |
| `auto-track-sorted-full-unrolled` | `1edc8ee5` | 662.4 s | 361.0 s | 24.8 GB | 11.1 GB |
| `structured-2d-120s-mem` | `27957feb` | 266.6 s (1 iteration) | — | 2.02 GB | 8.6 GB |
| `structured-2d-240s-mem` | `27957feb` | 398.3 s (1 iteration) | — | 2.16 GB | 14.7 GB |

Device memory stays flat as the window doubles; host memory still grows
because the final E-step returns the full posterior, as dense EM does. The
track's device peak comes from the sorted-spike likelihood, not the
transitions. Structured EM is slower where dense EM fits: each E-step
evaluates the likelihood twice (the forward pass and the replay), and the
filter and smoother steps of the structured operator took 510 s per warm run
on the 2-D grid against 351 s for dense matrix products, which the A100 reads
at its memory bandwidth (about 82 s per pass for 6,600 interior bins). On the
track (dense compare: `sorted-track-full-cc25c63`, 168 s warm on breeze) the
dense path is faster still.

### Result stores and the replay cache

`structured-2d-full-rowcache-{4g,24g}`: the same 2-D run with
`--result-path` on the host's local disk, with the default 4 GiB replay cache
and with 24 GiB, enough for every chunk.

| Report | Instrumented warm | Device peak | Host peak |
| --- | ---: | ---: | ---: |
| `dense-2d-full-chunks8-zephyr` (dense EM) | 500.6 s | 56.3 GB | 184.1 GB |
| `structured-2d-full` (before) | 705.9 s | 2.5 GB | 128.4 GB |
| `structured-2d-full-rowcache-4g` | 454.0 s | 2.5 GB | 9.6 GB |
| `structured-2d-full-rowcache-24g` | **356.4 s** | 2.5 GB | 14.3 GB |

Writing the final posterior to a store (8.8 GB per run on disk) removes it
from host memory. Caching each E-step's filtered rows lets the replay skip
both the second likelihood evaluation and the replayed forward pass for the
cached chunks; with every chunk cached, structured EM is 29% faster than dense
EM on this grid. These runs also include the unrolled scans (`1ad6a90`).

### Host memory per run

Reports before `f4dd609` measured each run while the previous run's model and
results were still alive, so their host peaks include an extra posterior (up
to 57 GB on the 2-D grid). `profile_em.py` now releases them and records each
run's own peak. Remeasured on zephyr, 2-D grid, 3 iterations, 16 GiB replay
cache:

| Report | Instrumented warm | Device peak | Host peak (per run) |
| --- | ---: | ---: | ---: |
| `mem-dense-2d-full-chunks8` | 501.0 s | 56.3 GB | 109.5 GB |
| `mem-structured-2d-full-inmemory` | 406.8 s | 2.4 GB | 71.4 GB |
| `mem-structured-2d-full-store` | **356.7 s** | 2.4 GB | **14.2 GB** |

On the track (`rowcache-auto-track-{sorted,clusterless}-full`, `auto`
transitions, 16 GiB replay cache), structured EM took 229.6 s (sorted) and
208.1 s (clusterless) warm, with host peaks of 9.6 and 8.3 GB and device peaks
of 24.8 GB (the sorted likelihood) and 4.0 GB. Dense EM is faster there (168.4
and 131.4 s on breeze, `54a54858`), which is why `estimate_parameters(
transition_representation="auto")` keeps dense EM for small state spaces.

### Clusterless KDE likelihood against recording length

`kde-scale-track-{180,360,720,1500}s`: dense EM, track, clusterless KDE, one
iteration (two E-steps), windows from the start of the recording.

| Window | Spikes | Likelihood | Filter/smoother | Device peak | Host peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 min | 69,274 | 1.00 s | 5.12 s | 1.5 GB | 5.7 GB |
| 6 min | 123,071 | 1.86 s | 10.04 s | 2.5 GB | 7.2 GB |
| 12 min | 227,470 | 5.90 s | 19.79 s | 5.1 GB | 10.9 GB |
| 23.6 min | 417,799 | 20.61 s | 39.17 s | 10.0 GB | 16.0 GB |

The filter and smoother grow linearly. The KDE likelihood grows about 3.5x
per doubling at these lengths: during EM both the encoding and the decoding
spikes grow with the recording, and each decoding spike is compared with
every encoding spike. A simulated sweep on the M1 Max CPU
(`kde-scale-simulated-cpu-{60..960}s`, 8 cm bins) shows the same: 7.64, 29.29
and 102.29 s at 240, 480 and 960 s, against 35.8, 79.9 and 157.8 s for the
filter and smoother. Extrapolated, not measured: a 2-hour session would spend
about 25 times as long in the likelihood as this 23.6 min one, which would
then take longer than the filter and smoother.

### Accuracy against float64

`accuracy-{dense64,dense32,structured32}-2d-120s`: 120 s of the 2-D grid, 2
iterations, with `JAX_ENABLE_X64=1` for the dense float64 reference.

| Run | State probabilities vs float64 (max / mean) | Most likely state changes | First-iteration evidence |
| --- | --- | ---: | ---: |
| dense float64 | — | — | -1,031,847.06 |
| dense float32 | 1.73e-3 / 1.67e-5 | 2 | -1,031,847.12 |
| structured float32 | 1.70e-3 / 1.67e-5 | 2 | -1,031,837.61 |

The posteriors are equally accurate. The evidence differs because of how the
dense filter accumulates it: its float64 run reports the same total as its
float32 run, so its running total is not computed in float64 there. Filtering
the same model and likelihoods with per-row increments summed in float64
gives -1,031,835.82 for the dense matrix in float64, -1,031,837.72 in float32,
and the same values for the structured operator; checkpointed EM sums its
increments that way.
Dense EM sums them that way too since `a5c7ef1`.

## 120 s and 240 s windows

| Report | What it shows |
| --- | --- |
| `sorted-track-120s`, `clusterless-track-120s` | Stage breakdown before any change |
| `sorted-track-120s-durationfix` | Sorted warm 44.3 → 29.9 s from the matrix emission |
| `*-track-240s-mem` | First-call device peak doubles with the window (4.99 → 9.89 GB), both families |
| `sorted-120s-autotune4`, `sorted-120s-autotune0` | `XLA_FLAGS=--xla_gpu_autotune_level=0`: first call 149.0 → 59.4 s, device peak 4.99 → 2.61 GB, warm unchanged (16.4 s, one iteration) |
| `sorted-120s-cache1`, `sorted-120s-cache2` | Two processes sharing `JAX_COMPILATION_CACHE_DIR`: first call 150.0 → 30.5 s in the second |

## Reproducing

```bash
export CUDA_VISIBLE_DEVICES=<idle GPU> XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python benchmarks/profile_em.py --recording <session>.npz \
    --bin-size 2 --max-iter 3 --cprofile --require-backend gpu --output <dir>
```

Add `--duration 120` for a window, or `--runs first --max-iter 1` for a
first-call measurement.
