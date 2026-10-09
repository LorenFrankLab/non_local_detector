S6 is frozen at 61 runtime Python files, aggregate `8b96d773cf5cdb5ae678facfe04fdd30edb096fbb5dbb97b848134695bfab13c`; archive SHA256 `202b38133a86eaf80b6de7c93ccdb5cc6db33a63088939a2a52daa1ea342e29a`. The full suite completed with 3137 passes and 26 skips. Wheel SHA256 `edb3ad48f785cf4fd71c5a4af85c2e7b2fe9aa90e467d3b016f6356ebc8c286d` contains exact S6 runtime bytes; installed sorted compact and tiled clusterless spatial smokes passed with nonzero events.

All eight one-hour decoding runs completed. CPU hours retain S2 hashes and GPU hours retain S4 hashes, using benchmark `4f84217f...`. Source proofs establish equivalence of the measured default paths through S6: inactive guards/DenseHighest for the structured 256-row/final-64-row S2 paths, an inactive 1D guard for S4 two-column positions, and bitwise default/explicit-None fitted-field/raw-likelihood/native-output equality from S5 to S6. This does not claim whole-package byte identity.

| Hour configuration | Prediction/export/read minutes | Peak host GB | Device NVML GB |
|---|---:|---:|---:|
| CPU sorted,2 cm,compact | 46.88 | 0.876 | — |
| CPU sorted,1 cm,selected spatial | 111.39 | 1.267 | — |
| CPU marked,2 cm,compact | 49.22 | 1.053 | — |
| CPU marked,1 cm,selected spatial | 112.57 | 1.393 | — |
| GPU sorted,2 cm,full spatial | 63.33 | 1.610 | 1.569 |
| GPU sorted,1 cm,full spatial | 72.82 | 2.040 | 1.838 |
| GPU marked,2 cm,full spatial | 81.66 | 2.520 | 1.565 |
| GPU marked,1 cm,full spatial | 120.12 | 2.926 | 2.370 |

Every hour conditions 1.8 million bins. CPU uses two units/electrodes at 5 Hz and ten seconds of encoding; the 1 cm runs save only 20 spatial rows. GPU uses 64 sorted units at 5 Hz or eight marked electrodes at 20 Hz/four marks and writes all spatial rows. Each hour is one warmed timed traversal. Different populations/output modes prevent a direct CPU/GPU speedup comparison. Prediction timers include export, bounded first/last reads and inside-predict checkpoint cleanup; full normalization postchecks and benchmark-store deletion are outside the timer. Wrapper elapsed includes setup and those later operations.

GPU full exports have apparent sizes 122.003 GB/477.109 GB at 2 cm/1 cm. Sampled allocated spatial peaks are 121.874–121.926 GB/477.102 GB, excluding checkpoint files of 0.483 GB/1.871 GB. Peaks are sampled lower bounds before cleanup. Process RSS excludes filesystem caches; NVML includes device context/allocator. Shared-host cache and ZFS ARC were sampled after jobs started and cannot be attributed to these jobs or treated as a pre-job baseline. A process-plus-device budget does not establish a system-wide cache bound.

S6 one-hour encoding probes use one electrode at 20 Hz: 72,000 events, 108,001 tracking samples, four marks, 1 cm/66,250 hidden bins, tiles 1024/8192 and a short 0.64-second decode. CPU fit/predict median are 3.79/3.91 seconds with 1.313 GB RSS; GPU 7.36/0.914 seconds with 1.798 GB RSS and 1.869 GB NVML. Both use stable S6 source and benchmark `21b90233...`. These prove the recorded long-encoding workspace case, not eight-electrode training or another full decoding hour.

Published numerical script `abf6ac2a...` preserves the original `eec336a...` AST apart from import order/formatting. Its independent oracle uses all training events but only three centers/seven marks. Float32 typed CPU/GPU inputs match exactly; seven original/float64 arrays differ across hosts, so each float64 oracle uses its own actual inputs. CPU tiny float32 densities retain baseline relative errors up to 1.85e-5 and fail ratio diagnostics; finished local/joint log intensities pass the existing controls. This does not establish uniformly 1e-6 relative density accuracy. Enabled float64 results are near rounding error.

Five interleaved small native pairs preserve numerical results. CPU original/tiled prediction medians are 26.92/31.98 ms; GPU 273.01/273.65 ms. This is a correctness check, not a tiling speedup claim. Original/tiled inputs match within each platform; raw position hashes differ across platforms. Kernel tiles bound forward intermediates, while resident inputs, requested outputs and reverse-mode storage remain outside that bound.

All 121 helper/integration controls pass on current JAX and the older tested Python 3.11/JAX 0.6.2/NumPy 1.26.4/SciPy 1.12 stack. Additional actual-x64 CPU/CUDA gates pass 89 helper/existing-block and 46 integration cases per platform; these overlap earlier counts and the full suite. Declared Python 3.10/JAX 0.4.27/NumPy 1.25/SciPy 1.10 floors remain untested.

No qualification gate remains pending. No source or benchmark hash mismatch was found. The final JSON records every raw artifact, copied public artifact hash, measured scope and source-equivalence limit.
