# Phase 7 — Measured performance at production scale

> **EXPANDED SCOPE — NEEDS PROTOTYPING.** The production workload and memory
> arithmetic are established in [overview.md](overview.md#representative-workload).
> Checkpointed smoothing, output interfaces, and structured transitions below are
> requirements and design candidates, not a validated implementation. The earlier
> unexecuted code snippets are removed. Phase 0/1 completion criteria are unchanged.

## Scope, order, and dependencies

> Re-verified against `main` at `ee2cc21` (2026-09-22): none of 7a–7c is
> implemented; workload arithmetic is exact; line references are to that
> revision. The "former ~108×" timing and the overview's exploratory
> 10,000-bin comparison have no recorded artifact.

Target decoding with fitted encoding and transition parameters for a 180 cm ×
180 cm arena, 1 cm or 2 cm bins, and one hour at 2 ms resolution. This gives
1.8 million observations and approximately 16,202 or 64,802 combined hidden bins
in the default four-state non-local model. Use actual fitted/padded dimensions
in every benchmark report.

Production integration follows the corrected likelihood/time contracts:
Phases 0–4 are merged on `main` (`ee2cc21`), including Phase 3's global event
ownership and row-bounded likelihood chunks, followed by Phase 6's intensity
units. Preserve the C1 floors in force at the baseline revision (zero-rate
fallback, mean-rate floor, summed-intensity clip); the package-wide C1 policy
is deferred ([shared-contracts C1](shared-contracts.md)), so nothing here waits
on it. Separate pure-core
prototypes can run earlier against an identified corrected reference. They must
not delay the outstanding likelihood fixes or introduce competing C1/C3 policies.
Use the release groups in [PLAN.md](PLAN.md#execution-order-and-baselines):
6a/6c together, then atomic 6b/6d. Applicable Phase 8 correctness fixes can ship
before Phase 7 and must be included in the baseline for affected configurations.

Implement and review these as independently measurable changes:

| Work | Purpose | Dependencies |
|---|---|---|
| 7a | Bound filtering/smoothing working memory and write requested outputs incrementally. | Phase 3 row-slice likelihood chunks (merged); final integration uses Phase 6 time/units. |
| 7b | Apply structured forward and backward transitions without a combined dense matrix where supported. | Phase 0 numerical contracts and an explicit transition-capability inventory; integrate with 7a for the large workload. |
| 7c | Improve likelihood kernels, compilation, and remaining allocation hotspots. | Profile the corrected backends; preserve the baseline floor and duration policies. |

7a and 7b can be prototyped independently on small references. End-to-end
production acceptance requires their integrated behavior plus any 7c work needed
to meet the declared resource budget. Keep the existing generic dense path for
reference and unsupported models, with explicit resource limits before fallback.

This phase preserves the statistical model. Reducing resolution, truncating
Gaussian tails, approximate smoothing, and backend substitution across the
known log-vs-probability parity gap are outside its scope. It does not establish
bounded memory for EM estimation or Viterbi. Audit shared callers for regressions,
but report those APIs separately from decoding support.

## Measurement protocol

Every task must supply numerical, timing, and memory evidence:

1. **Reference and parity first.** Freeze the corrected baseline revision,
   configuration, inputs, and tractable reference outputs before edits. Check
   posteriors, evidence, and marginals at existing applicable tolerances. Phase
   7 must not change golden data or relax tolerances. A new output mode must
   agree with the corresponding selection/reduction of the full reference.
2. **Synchronization and repetition.** Synchronize every JAX result. Warm each
   variant before timing; use randomized/interleaved comparisons with at least
   five measured repetitions for bounded workloads. Report median and spread,
   including paired runtime ratios when comparing implementations.
3. **Compilation and end-to-end work.** Report compilation separately from warm
   execution. Also measure the user-visible run, including likelihoods,
   checkpoint replay, host/device transfers, output conversion, and disk I/O.
   Short kernel timings alone cannot establish full-session throughput.
4. **Host/device peaks.** Record peak host RAM, device allocation, and output/
   checkpoint disk usage. Include resident inputs and model arrays, transient
   workspace, allocator reservation where relevant, and conversion buffers.
   Compiler `memory_analysis()` and live-array snapshots explain components;
   they are not sufficient evidence of a complete transient process peak.
   State the measurement method, sampling limitations, and background load.
5. **Dimensions, precision, and hardware.** Record `T`, spatial/interior/padded
   bins, combined `N`, discrete-state configuration, neuron/tetrode counts,
   spike counts/rates, encoding duration, chunk size, dtype, and hardware. Assert
   the dtype actually used. CPU results do not establish GPU results.
6. **Resource budget before scale-up.** Record host/device memory limits, disk
   capacity, and requested outputs. Establish safe allocation bounds before
   launching the full hour. Choose chunk/cache limits against that budget;
   do not attempt a full-size dense reference that exceeds it.

Use a staged benchmark ladder: small dense parity fixtures; realistic spatial
sizes with shortened recordings; duration scaling at fixed chunk/cache limits;
then full-hour runs on identified hardware. Exercise stationary/covariate paths
and supported sorted/clusterless backends. Routine CI uses bounded parity and
allocation regressions; full-hour runs are explicit release/performance checks.
For each claimed production configuration, complete a measured full-hour run;
label projections and untested combinations. A speedup factor cannot compensate
for failed parity or exceeding the declared memory budget.

## 7a: Bounded memory smoothing and outputs

### Problem and falsification

The current chunked drivers accumulate filtered and predicted arrays, concatenate
them, then accumulate and concatenate smoothed arrays. Model output selection
occurs afterward, and xarray conversion creates padded spatial arrays. Thus
`n_chunks` and `return_outputs` do not currently bound spatial working memory.
Verified at `ee2cc21`:

- Both drivers keep Python lists of filtered, predictive and smoothed chunks and
  concatenate them (`core.py:843-1002`, `:1473-1647`), so each list and its
  concatenation are briefly alive together.
- The predictive posterior is retained although the smoother recomputes
  `filtered @ T` and never reads it (`core.py:587`) — dropping it when not
  requested is a cheap first step.
- `n_chunks=1` (the default) or `cache_likelihood=True` evaluates the full
  T×N likelihood in one call (`core.py:880-890`); a requested `log_likelihood`
  allocates a full T×N host buffer (`core.py:709-724`).
- Each detector assembles likelihood rows into a float32 `(n_rows, N)` array
  with the shared non-local columns duplicated by eager `.at[].set`
  (`base.py:3315`, `:4334`), and `_create_masked_posterior`
  (`base.py:2541-2564`) allocates a float32 NaN-padded copy of each spatial
  output. A float64 run therefore still assembles float32 likelihoods and
  outputs; compact/incremental modes must cover the log-likelihood output and
  state the dtype actually used.
- `save_results` writes eagerly with `to_netcdf` (`base.py:2475`); an
  incremental writer is new work.

Before implementation, profile those allocations on a tractable run with fixed
chunk length and increasing duration. Exercise the default result as well as
explicit additional outputs. Record which arrays are retained and where peak
memory grows; do not allocate the full-hour dense target to demonstrate this.

### Output requirements

Prototype output selection and an incremental result writer together:

- A compact mode retains requested discrete-state probabilities and diagnostics
  without retaining a full-session spatial posterior. Inference still uses the
  full spatial distribution; it does not collapse the HMM to four hidden bins.
- A full spatial mode writes chunks to a disk-backed result or another bounded
  consumer. The writer must not collect all chunks in a list, concatenate them,
  or trigger eager full-array loading during xarray/result conversion.
- Support requested time selections with the same whole-recording conditioning
  as the full result. Decoding each requested interval independently is not
  equivalent. If intervals are selected after a compact run, preserve/recompute
  the required forward and backward boundary information and compatible model
  inputs so selected posteriors can be reconstructed with full context.
- Preserve timestamps, state/bin coordinates, missing rows, and padding masks.
  Apply output masking and coordinate assembly in bounded pieces. Return
  diagnostics in global time coordinates.
- Keep an explicit in-memory result option for feasible runs. Honor output
  selection during inference and allocation; do not quietly fall back from an
  incremental mode to full `T × N` storage. Define writer failures and cleanup
  so an incomplete result cannot be mistaken for a completed recording.

Select API names, storage format, chunk layout, and metadata conventions by
prototyping. No particular library or resume protocol is mandated here.

### Checkpointing candidate and its trade-offs

Prototype an exact forward/backward schedule: run forward while storing boundary
messages; visit chunks in reverse; restore each forward boundary and recompute
its internal forward distributions; smooth using the message from the following
chunk; emit the requested outputs and release the chunk workspace. The backward
boundary is the smoothed row of the following chunk (`core.py:990`) and the
terminal condition uses global `ind`/`n_time` (`core.py:591`), so a checkpoint
replay must pass global indices. `np.array_split` yields at most two chunk
lengths (two compiles per kernel); a fixed-length checkpoint schedule should
not add shapes on replay. Preserve the
current transition indexing, initial/final conditions, and Phase 0 fallback and
NaN semantics. Replayed forward calculations must not double-count evidence or
diagnostics. Time-varying transitions and likelihoods must use the same global
rows and fitted parameters as the original pass.

This trades additional forward work for reduced storage. Measure recomputing
likelihoods versus a bounded disk cache; retaining all likelihoods in RAM defeats
the target. Compare checkpoint intervals using both runtime and peak memory.
Checkpoints themselves require `O((T/L)N)` storage for chunk length `L`: keep them
on disk with a bounded cache for duration-independent spatial working RAM.
An optional all-checkpoints-in-RAM strategy must be budgeted and identified as
having duration-dependent memory. Account separately for recording inputs,
compact outputs, and index metadata.

Illustration for `N = 64,802`, `T = 1,800,000`, float32, and `L = 2,000`:

| Component | Calculated size |
|---|---:|
| Chunk duration | 4 seconds |
| One chunk of spatial probabilities | 518.4 MB |
| Approximately 900 forward checkpoints | 233.3 MB |
| Four discrete-state probabilities for the hour | 28.8 MB |
| One full spatial output written to disk | 466.6 GB |

These are component sizes, not a peak-memory guarantee or a chosen default.
Several working arrays, model/likelihood data, padded bins, and I/O buffers add
cost. Disk capacity and write/read throughput remain relevant even with bounded
RAM. The budget must include transitions; 7a alone does not remove their `N²`
storage.

### Acceptance

- Match full-reference filtered/smoothed posteriors, requested marginals, and
  evidence across chunk sizes, including singleton/ragged chunks, exact
  boundary events, missing rows, impossible observations, and NaN inputs.
- Cover both core transition paths and both detector families through public
  APIs. Verify output selections against the full conditioned reference.
- Demonstrate no retained full `T × N` spatial arrays in compact/incremental
  modes, including during final result conversion. Verify duration scaling at
  fixed chunk/cache limits and account for checkpoint disk growth separately.
- Test numerical output and ordering after writing/reading an incremental result,
  writer error handling, and release of consumed buffers. Validate production
  dimensions under the declared budget before claiming bounded-memory support.

## 7b: Structured forward and backward transitions

### Problem and falsification

The model currently builds a dense combined transition matrix. Stationary
filtering multiplies by it at every step; smoothing applies both forward and
backward products (`core.py:464`, `:587-590`). The covariate path forms
`continuous * discrete[ix_(state_ind, state_ind)]` over N×N at every scan step
(`core.py:1172-1195`): measured at N = 2048 on CPU, the compiled temporary
workspace is 2×N² for the filter and 3×N² for the smoother (stationary: under
25 KB). Setup allocates the padded float64 `continuous_state_transitions_`
(`base.py:1241`), three more float64 N² temporaries in `_predict`
(`base.py:1795-1798`) and a float32 device copy (`core.py:866`); at 1 cm each
float64 N² is 33.6 GB. `_euclidean_random_walk` is a Python loop over bins
calling scipy `multivariate_normal` (~1.1e9 pdf evaluations at 1 cm), a
setup-time cost separate from memory. Inspect construction, combination, and
application: optimizing multiplication after allocating the same dense matrix
cannot satisfy the memory target.

Record dense reference products and resource use before prototyping. Inventory
which transition types and configurations permit an exact structured operation
and which require a generic fallback.

### Candidates to prototype

- For uniform blocks, sum source mass and distribute it over eligible destination
  bins. For identity and single-bin blocks, use the corresponding direct
  operations. Preserve rectangular cross-state blocks and all masks/weights.
- Preserve the model's discrete-state structure instead of expanding every
  discrete weight over a combined `N × N` array. In the default four-state model,
  only Non-Local Continuous → Non-Local Continuous uses a random walk; the other
  blocks use uniform or single-bin transitions.
- For Euclidean Gaussian movement with separable covariance on a Cartesian grid,
  prototype applying movement along each axis with the original row normalization
  and source/destination masks. Derive both forward and backward applications;
  applying a generic image blur is not sufficient evidence of HMM equivalence.
- For graph/manifold distances, nonseparable covariance, directional constraints,
  custom transitions, or incompatible grid structure, dispatch only to a proven
  operator. Retain a tested dense fallback with a feasibility check. Never
  silently replace such a model with a Euclidean or truncated approximation.
- Include in the inventory: `EmpiricalMovement`, `RandomWalkDirection1/2`,
  `Identity`, multi-bin local (`local_position_std`, which upgrades Local blocks
  to `Uniform`, `base.py:1289-1311`), multi-environment `Uniform`, and zero-sum
  rows mapped to 0 by `_normalize_row_probability`. `estimate_movement_var`
  returns a full `np.cov` matrix (usually nonseparable) despite documenting
  `(n_position_dim,)`, and the scalar `movement_mean` shifts every axis.
- `continuous_state_transitions_` is public: plotting, EM, Viterbi and tests
  read it, and `save_model` pickles it. It spans padded total bins (182×182
  per state at 1 cm in 2-D, not 180×180). Decide whether structured models
  expose a lazy dense view.

An exploratory float64 check on a 5 × 7 grid with excluded bins, scalar movement
variance 6.0, and movement mean 0.25 compared separable forward/backward products
with the current dense Euclidean implementation. Maximum absolute errors were
`1.39e-17` and `1.67e-16`. This supports the candidate algebra; it does not
validate a production operator, float32 behavior, derivatives, or every mask.
Reproduce and extend it in the implementation tests before relying on it.
Its inputs (seed, excluded bins, test vectors) were not recorded; an
independent reproduction (4 excluded bins, seed 0) gave `1.39e-17` / `3.33e-16`.
Commit the script with the implementation tests.

For a complete 180 × 180 grid, the random-walk block has approximately 1.05 billion
pairwise terms; two full axis operations use approximately 11.7 million terms.
This is operation-count arithmetic, not a measured 90× decoding speedup. Actual
performance depends on compilation, memory traffic, hardware, masking, and the
likelihood workload.

### Acceptance

- Match dense forward and backward products, row-stochastic behavior, and full
  filter/smoother outputs at existing applicable tolerances. Cover nonuniform
  priors, boundary bins, interior holes, identity/uniform blocks, zero discrete
  transition probabilities, and initial/terminal conditions.
- Test stationary and time-varying discrete weights, supported grid/covariance
  configurations, generic fallbacks, and both detector families. Verify supported
  gradients through priors, likelihoods, and transition parameters, including
  zero-support cases covered by Phase 0.
- Demonstrate that supported structured paths avoid a combined `N × N` allocation
  in model setup, device transfer, filtering, and smoothing. A fallback must not
  be labelled as satisfying that structured-memory guarantee.
- Measure both nominal grid sizes, compilation, execution, and host/device peak
  memory. Integrate with 7a and compare the same requested outputs and precision.

## 7c: Measured likelihood and compilation optimizations

Retain the earlier candidates and prioritize them from profiles after the major
allocation/transition costs are understood. A backend allocation that prevents
the production run is a prerequisite, not an optional late speed improvement.

### 1. Matrix accumulation for sorted-spike likelihoods

Prototype shared per-chunk spike-count construction and matrix accumulation in
`sorted_spikes_kde` and `sorted_spikes_glm`, using the diffusion implementation as
an existing reference pattern. Preserve Phase 3 global event ownership and Phase
6 duration factors. Compare against an independent per-neuron `xlogy` reference.
Today both backends loop over neurons eagerly with one `(n_rows, n_bins)`
temporary per neuron (`sorted_spikes_kde.py:397-416`,
`sorted_spikes_glm.py:507-527`). Reuse `_spike_counts_matrix`
(`sorted_spikes_diffusion.py:550-580`), which already handles `row_slice`.
Spike ordering is already shared per prediction (`_SpikeTimeOrder`, `ac0a007`);
per-neuron counts are not.

The former ~108× timing was unsynchronized and is not accepted performance
evidence. Re-measure under this phase's protocol. The algebraic replacement
requires finite log fields. Under the current floors fitted place fields are
clipped to at least `EPS` (`sorted_spikes_kde.py:223-235`,
`sorted_spikes_glm.py:364-371`); if a later C1 policy permits exact zero intensities,
`xlogy(0, 0)` cannot be replaced by `0 * log(0)`. Test that case explicitly and
preserve the policy rather than adding a new floor for performance.

### 2. Bound KDE and kernel recompilation

Measured at `ee2cc21`: `common.kde` compiles once per `(eval_block, n_samples)`
shape — two per new sample count because the last eval block is ragged — and
the linear `clusterless_kde` path compiles the jitted `log_gaussian_pdf`
(`common.py:469`) per `(n_enc, n_dec_block)`, where the ragged decoding block
changes from chunk to chunk. Bucket both the sample and the eval/decoding-block
dimensions.

Prototype sample-count buckets with masked/zero-weight padding, verifying that
padded samples do not enter either normalization or downstream consumers.
Compare powers of two, multiples of 256 above a floor, and powers of `sqrt(2)`
on recorded spike-count distributions. Measure compilation count/time, total
runtime, and memory together. Ragged/empty inputs and padded `samples_`/`weights_`
consumers need parity tests. Select the bucket policy from evidence; there is no
fixed cache-count promise before measurement.

### 3. Avoid unnecessary clusterless block copies

`block_estimate_log_joint_mark_intensity` in `clusterless_kde` is eager (not
jitted): each block copies the full `(n_dec, n_pos)` array via
`dynamic_update_slice` and recomputes the `(n_enc, n_pos)` temporary
`encoding_weights[:, None] * position_distance` (`clusterless_kde.py:98`,
`:112-162`). Compare jitting or donation, hoisting the weighted kernel, and
applying each block directly to rows via `sum_spikes_into_rows`. `common.block_kde`
and `clusterless_kde_log` already use list-then-concatenate; do not adopt that
pattern here without measuring its peak: it can keep both all blocks and the
final output alive. Preserve the settled
finished-intensity flooring semantics and ragged/empty-block results.

### 4. Encoding-kernel memory if required by the budget

Profile the linear `clusterless_kde` encoding-spike × position kernel at the
recorded encoding duration and spike counts. `position_distance`
(n_enc × n_interior_pos) is recomputed per electrode on every likelihood call,
i.e. per chunk (`clusterless_kde.py:493`), and `clusterless_kde` is the default
algorithm; caching it across chunks trades RAM for time and must be budgeted. If it prevents the declared memory
budget from being met, prototype tiling/streaming that same backend with numerical
parity. This work need not wait for retiring the duplicate linear/log paths;
substitution across their known parity gap remains out of scope. Also account
for waveform dimensions, density workspaces, and cache reuse in real backends.

### Acceptance

Keep per-neuron/matrix, padded/unpadded, and blocked/unblocked reference tests,
including empty and ragged cases, at existing applicable tolerances. Verify
shared-helper callers and both local/non-local consumers. Report component and
end-to-end improvements under the same output, precision, and resource settings;
revisit profiling after each accepted change rather than assuming the next
candidate remains the bottleneck.

## Integration and release evidence

Before claiming production support, provide the following for each declared
configuration:

- Corrected baseline and optimized revisions, small/reference parity results,
  existing golden results, float32 and actual enabled-float64 checks, and tests
  for the supported differentiation paths touched by the change.
- The representative-workload configuration, actual fitted dimensions, hardware,
  requested outputs, chunk/cache choices, and recorded resource budgets.
- Host/device peak memory, checkpoint/output sizes, compilation, warm execution,
  and full-hour end-to-end runtime with the likelihood and I/O costs included.
- Supported structured configurations and fallback limits; untested devices,
  backends, EM, and Viterbi must not inherit the decoding claim implicitly.

Review each subtask and the integrated pipeline with its numerical and resource
artifacts. Run the full suite for implementation changes, plus relevant core,
backend, integration, and golden regressions. A Phase 7 golden discrepancy is
investigated as a regression; the response is not to update goldens or widen
existing tolerances. Record measured trade-offs and limitations in the release
note rather than a projected universal speedup.
