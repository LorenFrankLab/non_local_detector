# Settings for your hardware

This page suggests settings for three kinds of machine: an 80 GB A100, a GPU
with 24 GB of memory, and a CPU-only computer. It covers decoding with a fitted
model and parameter estimation (EM). How the decoding modes work is described in
[performance_prediction.md](performance_prediction.md).

All numbers come from the benchmark workloads in
[performance_validation.md](performance_validation.md): a simulated 180 × 180 cm
arena, 2 ms bins, 64 sorted units at 5 Hz or 8 clusterless electrodes at 20 Hz
with four waveform features, using the default likelihoods (sorted and
clusterless KDE); memory for the other likelihood algorithms was not measured.
The runs used an A100 80 GB and an Apple M1 Max laptop (10 cores, 64 GB). No
24 GB card was used: 24 GB memory figures come from an A100 with its allocator
capped, so speed on a 24 GB card is unmeasured. Your populations, encoding length and grid change these numbers;
time a short segment of your own data before a long run.

## Summary

| | A100 80 GB | 24 GB GPU | CPU |
| --- | --- | --- | --- |
| Decoding mode | Structured fit, checkpointed prediction | Same | Same |
| `chunk_size` | Default | Default | Default; `256` if memory is tight |
| Decoding memory, 1–2 cm | About 1.5 GB of device memory | Fits; measured within a 4 GB cap | 3.3–4.0 GB of RAM at 2 cm |
| Decoding one hour at 2 cm | 6 min (sorted), 11 min (clusterless) | Not measured | About 30 min (projected from 30 s runs) |
| EM | 60 s at 2 cm used 15.5 GB of device memory | 60 s at 2 cm fits; longer sessions need a coarser grid or a shorter segment | 60 s at 4 cm took 8 min and 10 GB of RAM |
| Precision | float32 (default) | float32; float64 is slow on most 24 GB cards | float32 |

## GPU setup

Set these environment variables before starting Python:

- `CUDA_VISIBLE_DEVICES=<index>` picks the card. On a shared machine, choose an
  idle one first (`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`).
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX from reserving most of the
  card at startup, so other jobs can share it. Leave preallocation on if the
  card is yours alone.
- To hold a run to a memory budget, keep preallocation on and set
  `XLA_PYTHON_CLIENT_MEM_FRACTION` to the budget's share of the card. For
  example, `0.30` on an 80 GB card gives 24 GB, which is how you can check
  whether a run will fit on a 24 GB card.
- `XLA_FLAGS=--xla_gpu_autotune_level=0` makes repeated runs in separate
  processes bitwise identical. It didn't change prediction time on the A100.
  Without it, state probabilities from separate runs of unchanged code differed
  by up to 3.1e-4.

## Decoding

Fit with `transition_representation="structured"` and predict with
`inference_mode="checkpointed"` (see
[performance_prediction.md](performance_prediction.md)). Device memory then
depends on `chunk_size` and the grid, not on recording length. Run time is
proportional to recording length.

**Memory.** With the default `chunk_size`, 120 s recordings at 1 cm and 2 cm
peaked at 1.36–1.38 GB of device memory. They ran within a 4 GB allocator cap,
so any of these GPUs has room to spare. One hour on the A100 peaked at 1.36–1.48
GB of device memory and 1.6–4.6 GB of host RAM. Device memory rose only when the
encoding data were large: with 30 minutes of encoding, peaks were 3.7 GB
(sorted) and 5.1 GB (clusterless), including fitting.

On a CPU, chunks live in RAM: 30 s at 2 cm peaked at 3.3–3.5 GB with the default
`chunk_size` (64 sorted units or 8 electrodes; 4.0 GB with 256 units). A
smaller `chunk_size` lowers this at some cost in speed: with 256-row chunks, an
earlier version peaked at 0.9 GB (sorted) and 2.8 GB (clusterless).

**Speed.** On the A100, one hour at 2 cm took 364 s (sorted) and 681 s
(clusterless), including compilation; after compilation, each recording second
costs 0.096 s and 0.14 s. These hour runs predate the latest clusterless
changes, which made 30 s clusterless runs faster. Once compiled, a 30 s sorted
recording took 2.5 s at 4 cm, 2.95 s at 2 cm and 4.3 s at 1 cm, and a 30 s
clusterless recording took 3.2 s at 2 cm. On the M1 Max, 30 s at 2 cm took
15.7–16.6 s for either family, 0.52–0.55 s per recording second; an hour at
that rate is about half an hour, a projection rather than a measured run.

**Outputs.** Prefer `output_mode="compact"` (state probabilities) unless you
need the spatial posterior. One spatial posterior for an hour takes about
122 GB on disk at 2 cm and 477 GB at 1 cm; use `selected_intervals` to keep only
the times you need.

**Large populations.** The first prediction includes compilation, which grows
with the number of distinct electrode and unit sizes. With 128 clusterless
electrodes at equal rates, a 30 s first prediction took 29 s on the A100
(6.9 s once compiled); with rates spread 8× across electrodes it took 131 s
(5.6 s once compiled). JAX's persistent compilation cache
(`jax.config.update("jax_compilation_cache_dir", "<path>")`) reuses compiled
code across processes, but only when array shapes repeat exactly, for example
when decoding the same recording again.

**Long encoding with the linear clusterless KDE.** Its kernels grow with the
number of encoding spikes. If they don't fit, set `encoding_block_size` and
`position_block_size` (see
[performance_prediction.md](performance_prediction.md)).

## Parameter estimation (EM)

`estimate_parameters` always uses the dense filter and smoother. It keeps
several arrays of size (time bins × state bins) in memory, so its memory grows
with session length. At 2 cm each such float32 array takes about 2 GB per minute
of recording; at 4 cm, about 0.53 GB.

Measured with 60 s sessions and 3 EM iterations:

| Machine | Grid | Run time after compilation | Device memory | Host RAM |
| --- | --- | ---: | ---: | ---: |
| A100 (shared with other jobs) | 4 cm | 147–168 s | 3.9 GB | 8.3–8.4 GB |
| A100 (shared with other jobs) | 2 cm | 604 s (sorted), 1,281–1,364 s (clusterless) | 15.5 GB | 24 GB |
| M1 Max CPU | 4 cm | 488–516 s | — | 9.8–10.2 GB |

Only one session length was measured per grid, so how fast memory grows is an
estimate: the smoother holds at least three of these arrays, so each extra
minute adds at least 6 GB at 2 cm and 1.6 GB at 4 cm. A 24 GB card fits 60 s
at 2 cm but probably not much more. For longer sessions, estimate parameters on
a segment that fits, or on a coarser grid, and check the peak with
`XLA_PYTHON_CLIENT_MEM_FRACTION` first. Host RAM also grows, because the
posteriors are copied to the host.

Every EM call refits the environment and computes graph distances between all
bins, which takes about 10 s at 4 cm and 170 s at 2 cm on the M1 Max. At 1 cm
the dense transition matrix alone is 17.6 GB, so EM doesn't fit on a 24 GB
card, and the graph distances would take roughly 50 minutes (extrapolated).

## Precision

The package computes in float32 by default, and every number on this page used
float32. Enabling float64 (`jax_enable_x64`) doubles array memory, and most GPUs
other than data-center cards like the A100 run float64 arithmetic many times
slower than float32.

## Where the numbers come from

- Decoding and scaling: [performance_validation.md](performance_validation.md)
  and `docs/performance_artifacts/` (the `bottlenecks/` and `scaling/` records).
- EM, memory caps and graph-distance timing: `docs/performance_artifacts/em/`.
