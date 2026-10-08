# Checkpointed prediction

These opt-in APIs build on the 0.7 time/Hz contracts. Existing `fit()` and
`predict()` calls keep their dense behavior. The modes apply to prediction with
fitted parameters; EM and Viterbi retain their existing algorithms. For
suggested settings on 24 GB GPUs, A100s and CPUs, see
[hardware_settings.md](hardware_settings.md).

Fit a supported Cartesian model without constructing a combined dense
transition matrix:

```python
detector.fit(
    position_time=position_time,
    position=position,
    spike_times=encoding_spikes,
    encoding_time_range=[encoding_start, encoding_stop],
    transition_representation="structured",
)

probabilities = detector.predict(
    spike_times=decoding_spikes,
    position_time=position_time,
    position=position,
    time_edges=time_edges,
    inference_mode="checkpointed",
    output_mode="compact",
    return_outputs=["filter", "predictive"],
)
```

Clusterless calls also provide `spike_waveform_features` at fit and prediction.
All times are seconds. Decode edges retain the uniform-grid contract described
in [time_grid_migration.md](time_grid_migration.md).

## Choose the result you need

| Mode | Result | Storage |
| --- | --- | --- |
| Default dense prediction | Full smoothed posterior and state probabilities | In memory, proportional to recording length × hidden bins |
| Checkpointed compact | Smoothed state probabilities; optional filtered and predictive state probabilities | In memory, proportional to recording length × discrete states |
| Checkpointed spatial | Full smoothed posterior and state probabilities; optional spatial outputs | Incremental directory store; spatial arrays are lazy |

Compact prediction still runs the complete spatial HMM. It reduces each requested
output while the spatial chunk is available. Spatial working memory depends on
`chunk_size`, not the entire recording length. Recording inputs, coordinates,
diagnostics and compact outputs still grow with recording length. Boundary
messages go to disk and are replayed during backward smoothing; likelihoods are
evaluated twice. Replay compares a checksum of each likelihood chunk and fails
if its values change between passes. Keep fitted parameters and callback inputs fixed
throughout prediction.

Chunking bounds forward/backward arrays and likelihood output rows. Likelihood
workspaces need a separate budget: the default linear clusterless KDE still
constructs an encoding-event × spatial-bin position kernel and a weighted
kernel. Reducing `chunk_size` does not remove them. Include encoding data,
these kernels and compiled executables in host/device preflight; large encoding
populations do not inherit the HMM memory guarantee.

For longer training sets, the linear `clusterless_kde` backend can also tile
encoding samples and position queries during fit and prediction:

`clusterless_algorithm_params` replaces the detector's default parameter
dictionary. When adding tile limits, retain your existing bandwidths and block
size. The example below explicitly preserves the detector defaults; supplying
only tile limits would instead use the standalone backend's different defaults.

```python
detector = NonLocalClusterlessDetector(
    clusterless_algorithm_params={
        "position_std": 6.0,
        "waveform_std": 24.0,
        "block_size": 10_000,
        "encoding_block_size": 1024,
        "position_block_size": 8192,
    },
)
```

The fitted model carries these optional limits into both local and nonlocal
prediction. Samples and weights remain unchanged; normalization and intensity
floors apply after all encoding tiles contribute. Defaults retain the original
untiled kernels. These limits bound kernel tiles; resident inputs, returned
arrays and differentiation storage still need budgets. Smaller tiles can
increase execution time. Use checkpointed row chunks alongside these limits
for long recordings. Other likelihood backends retain their own workspaces.

`return_outputs="filter"` adds filtered state probabilities in compact mode and
also adds the filtered spatial posterior in spatial mode. `"predictive"` adds
predictive state probabilities. `"predictive_posterior"`, `"log_likelihood"` and
`"all"` require spatial mode. The full log evidence remains available as
`results.attrs["marginal_log_likelihoods"]`.

The string `"predictive"` also requests the spatial posterior in spatial mode.
Lists/sets select named outputs individually: `["predictive"]` requests state
probabilities and `["predictive_posterior"]` requests the spatial posterior.
Checkpointed totals sum per-bin log-evidence increments in float64 on the host;
posterior calculations keep their original precision. This avoids the existing
hour-scale drift from repeatedly adding increments to a float32 total. The dense
driver retains its previous total accumulation.

Checkpointed prediction uses `chunk_size`. Leave the older `n_chunks=1` and
`cache_likelihood=False` controls at their defaults. By default each chunk holds
enough rows for its float32 likelihoods to take about 256 MiB, and at least 256
rows: 3,963 rows for a 2 cm grid of 16,930 state bins. Larger chunks spend less
time on per-chunk overhead, and working memory grows with chunk rows. For that
grid and a 30 s recording, the A100 device peak was 0.15 GB at 256 rows and
1.48 GB at the default; on CPU, where chunks live in host memory, peak RSS rose
from 0.9 to 3.4 GB (sorted) and from 2.8 to 4.8 GB (clusterless). Pass a smaller
`chunk_size` to fit a smaller budget; 256 restores the previous footprint. `checkpoint_dir` chooses
where temporary boundary files are written; files are removed after success or
failure. Supplying `result_path` also persists compact outputs when desired.

## Spatial outputs and selections

```python
results = detector.predict(
    spike_times=decoding_spikes,
    position_time=position_time,
    position=position,
    time_edges=time_edges,
    inference_mode="checkpointed",
    output_mode="spatial",
    result_path="session-posterior",
    selected_intervals=[[10.0, 12.0], [90.0, 91.0]],
)
window = results.acausal_posterior.sel(time=slice(10.0, 10.5)).values
reopened = detector.load_results("session-posterior")
```

Selections are ordered, non-overlapping start/stop intervals. They select bins
whose **complete edges** lie within an interval. Filtering and smoothing still
condition on the entire supplied recording, including observations outside the
requested intervals. Selections save output space; they do not reduce the amount
of inference. To select another interval later, repeat prediction with the same
recording and fitted model. Temporary checkpoints are not a resume format.

The directory contains a versioned manifest, coordinates, and bounded NumPy
chunk files. Completion publishes the directory atomically. An existing
destination is never overwritten. Failed writes clean up unpublished data;
readers reject incomplete or inconsistent manifests. State/bin coordinates,
excluded-bin NaNs, missing masks and the original final-bin closure are restored
when reopening, without needing the fitted model.

Reading maps and closes one chunk at a time. A single explicit spatial read is
limited to 512 MiB by default. Select fewer rows/bins before calling `.values`;
whole-recording NumPy conversion and ordinary xarray reductions can exceed this
limit. Set `max_read_bytes` in `predict()` or `load_results()` only for a feasible
explicit read. `save_results(..., "export.nc")` exports a feasible compact or
spatial result to NetCDF; spatial export reads the requested arrays and retains
the read limit. Use the directory directly for large recordings.

## Transitions and unsupported configurations

`transition_representation="structured"` requires a proven operator. Supported
blocks include uniform, identity, scalar/rectangular state transitions and
untruncated separable Gaussian random walks on Cartesian Euclidean grids, with
interior holes and the existing row normalization. Multi-bin Local states retain
their existing transition upgrades. Discrete covariate weights stay indexed by
global decode row. Structured setup also defers N-D graph all-pairs distances;
optional position kernels request exact distances in bounded batches.

Nonseparable covariance, graph/diffusion movement, directional movement,
empirical movement and custom transitions require a dense fallback unless a
specific proven operator applies. `transition_representation="auto"` permits
that fallback under `max_dense_transition_bytes` (256 MiB by default). The budget
covers the sum of retained fallback blocks. A fallback does not carry a
structured-memory guarantee. Unsupported extreme Gaussian underflow cases are
reported explicitly rather than changing their probabilities.

Opaque custom constructors require legacy eager environments; auto mode rejects
them when graph distances are deferred because their distance-access contract
is unverified. A feasible explicit dense fit can still use them with checkpointed
prediction, but does not avoid dense setup or bound arbitrary custom workspace.

For structured models, `continuous_state_transitions_` is a read-only lazy dense
view. Feasible selections and explicit conversion are budgeted; oversized dense
access from plotting, dense prediction or Viterbi fails before allocation.
Pickling retains the compact operator. EM refits its model using the existing
dense path; phase 7 does not make EM or Viterbi suitable for full-hour spatial
workloads. Use a separate feasible configuration to estimate parameters.

## Missing data and sequences

Missing rows contribute a neutral observation likelihood and still propagate
the same transition once per bin. They do not reset the HMM or split the
recording. Selected output intervals also do not create independent sequences.
Use separate prediction calls with separate time grids when recordings should
have independent initial conditions. Global impossible-observation and NaN
diagnostics cover the entire recording even if only a few rows are returned.

## CPU and CUDA qualification

Both paths use JAX and the same statistical model. Structured transition products,
sorted count accumulation, linear marked-intensity products and dense
state-probability aggregation request the highest multiplication precision so
CUDA does not silently use TF32 for these products. Native likelihood and
posterior arrays remain float32; checkpointed evidence totals use host float64.
Actual float64 posterior qualification belongs to the pure numerical core with
x64 enabled.

Configure CUDA allocator reservation before starting Python when enforcing a
device budget. The benchmark defaults to
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and reports active, reserved and pooled
allocation separately. The recorded A100 runs used an allocator fraction of
0.08 plus device-wide NVML measurements; that fraction is specific to an 80 GB
card. Application working memory, CUDA context and filesystem cache need
separate accounting.

Repeated CUDA predictions in separate processes are not bitwise identical by
default, because XLA autotuning chooses kernels in each process. For the
benchmark workload in the [validation record](performance_validation.md), state
probabilities from separate runs of unchanged code differed by up to 3.1e-4.
Setting `XLA_FLAGS=--xla_gpu_autotune_level=0` before starting Python made three
A100 runs bitwise identical without changing their prediction time. Within one
process, checkpoint replay is deterministic and checked.

Run `benchmarks/benchmark_native_pipeline.py` for native pipeline measurements and
the dedicated checkpoint, operator and likelihood scripts for component
measurements. Reports include synchronized runs, source/input hashes, dimensions,
precision, process peaks and disk usage. `--platform gpu` fails if CUDA is
unavailable. Shortened recordings and synthetic populations establish only the
configurations recorded in their reports; CPU measurements do not qualify CUDA
or smaller GPU hardware. Full-hour production claims require measured runs at
both spatial sizes and declared host/device/disk budgets. The Phase 8 diffusion
volume and unoccupied-MRF correctness work remains separate. See the
[validation record](performance_validation.md) for measured configurations,
source provenance and limits.
