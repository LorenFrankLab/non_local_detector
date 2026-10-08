# Benchmarks

Reproducible performance and determinism measurements for `non_local_detector`.
Each script is standalone: run it with the project environment and pass `--help`
for its options.

```bash
uv run python benchmarks/<script>.py --help
```

`src/non_local_detector/tests/test_benchmark_scripts.py` runs every script's
`--help` in the test suite. A script that stops importing against the current
package fails CI instead of silently going stale.

## Measurement conventions

New and updated scripts follow the
[JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html):

- Time compilation plus the first call separately from steady-state execution.
- Warm up, call `jax.block_until_ready` at every timing boundary, and report
  min/median/max over at least five repetitions.
- Keep shapes and dtypes fixed within a measurement. Report how many executables
  a range of shapes creates when shapes vary in production.
- Check outputs against a reference while timing them.
- Record the device, JAX version, x64 setting and a hash of the measured source
  in the report, so results stay attributable after the code changes.
- For memory, record XLA's compiled buffer sizes (`memory_analysis()`) or device
  peaks. Use an xprof trace for transient workspace and kernel-level time.

## GPU runs

- Pick an idle device first. `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
  shows memory and utilization, and `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`
  shows which processes are on each device.
- Then set the device and disable preallocation:
  `export CUDA_VISIBLE_DEVICES=<idle index> XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Run long jobs inside `tmux`.
- Scripts that accept `--require-backend gpu` refuse to fall back to CPU.

To compare against an earlier commit, run the script from a `git archive` of
that commit. Alternatively, put the old `src/` first on `PYTHONPATH` and run
the current script; scripts that inspect the package handle older function
names.

## Scripts

| Script | Measures | Typical use |
| --- | --- | --- |
| `compare_prediction_modes.py` | One fit and prediction of a simulated 2-D recording: fit time, compile and warm prediction time, peak host RSS and device memory, and saved state probabilities. Modes are `dense`, `chunked` (`n_chunks`) and checkpointed `compact`. `--algorithm` selects a likelihood and `--rate-spread` gives units different spike rates. Works on older checkouts via `PYTHONPATH` and refuses modes they lack. | One configuration per process: `--family --mode --duration --arena --bin-size` |
| `benchmark_em.py` | EM parameter estimation (`estimate_parameters`) on the same simulated session: first and warm wall time, iterations, marginal log-likelihoods, peak host RSS and device memory. EM uses the dense filter/smoother. | `--family --duration --bin-size --max-iter [--algorithm]` |
| `profile_checkpointed_prediction.py` | Splits one warm checkpointed compact prediction into stages: local, non-local and no-spike likelihoods (forward pass and replay), forward/backward kernels, checkpoint I/O and the remainder. Uses the `compare_prediction_modes.py` workload. | Finding bottlenecks: `--family --duration --bin-size` |
| `profile_compilations.py` | XLA compilations during a first checkpointed compact prediction of the same workload, by function (count and seconds), then warm prediction time; saves state probabilities for comparison between source trees. `--algorithm` selects a likelihood. | Compile-time changes: `--family --algorithm --encoding-duration` |
| `benchmark_checkpoint_scan_steps.py` | Per-step cost of the jitted forward/backward chunk kernels with the restricted structured transition operator of the same detector workload, plus fusions and cuBLAS calls launched per step. `--xprof-dir` traces one warm chunk of each. | Transition-operator changes: `--chunk-size --bin-size` |
| `benchmark_spike_row_reduction.py` | Deterministic segmented-scan spike-row reduction versus scatter (and the serial loop in older checkouts). Reports compile and steady-state time, XLA temp bytes, agreement with scatter, and recompilation across spike counts. | GPU: `--require-backend gpu`; about 5 min on an A100 |
| `check_likelihood_determinism.py` | Bitwise repeatability of clusterless likelihoods: 4 algorithms × default/collision-heavy cases. Exits nonzero on drift and records digests for cross-run comparison. | After reduction or likelihood changes, on CPU and GPU (`JAX_ENABLE_X64=0/1`) |
| `benchmark_chunk_likelihood_runtime.py` | One 500-row likelihood chunk on 60 s and 1 h recordings, per backend. Optionally measures spike-ordering reuse (`--compare-ordering`). | `OUTPUT_DIR [--spikes-per-second N]` |
| `benchmark_chunk_likelihood_memory.py` | How a chunk's likelihood scales with chunk rows and selected spikes. | `--quick`, or `--sweep all` |
| `benchmark_checkpointed_inference.py` | Checkpointed filter/smoother runtime, RSS and checkpoint bytes on synthetic likelihoods. | `--rows --bins --chunk-size` |
| `benchmark_replay_cache.py` | Exact likelihood replay compared with a bounded disk-cache prototype. | `--platform {cpu,gpu}` |
| `benchmark_native_pipeline.py` | End-to-end detector prediction with explicit host/device budgets. | `--family --platform --arena ...`; hour-scale runs take hours |
| `benchmark_transition_operators.py` | Structured transition products against dense fallbacks. | `--extent --bin-size --dtype` |
| `benchmark_likelihood_kernels.py` | Likelihood kernels compared with a baseline source tree. | `--baseline-dir --output-dir` |
| `benchmark_kde_buckets.py` | Experimental weighted KDE sample buckets (does not change fitted models). | `--output [--x64]` |
| `benchmark_gaussian_numerics.py` | Gaussian score/EM runtime, compiled buffers and preserved outputs. | `OUTPUT_DIR --samples --components` |
| `benchmark_hmm_conditioning.py` | Compiled HMM kernels compared with a Git baseline or a saved core. | `--baseline-ref` |
| `benchmark_sorted_spikes_diffusion.py` | Sorted-spikes diffusion eigendecomposition, diffusion and fit cost across grid sizes. | `[--neurons N] [--bin-sizes ...]` |
| `qualify_long_encoding.py` | Long-encoding linear KDE compared with independent float64 references. | `--output-dir --platform` |

Measured results backing documented claims live in
`docs/performance_artifacts/`. Those records keep the script paths that were
current when they were produced.
