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
