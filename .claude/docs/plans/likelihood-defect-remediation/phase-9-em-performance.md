# Phase 9 — EM parameter estimation on users' hardware

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [Phase 7 measurement protocol](phase-7-performance.md#measurement-protocol)

> **NEEDS DESIGN AND PROTOTYPING.** This file records requirements, measured
> baselines, design options and the experiments that must decide between them.
> It contains no implementation code, following the plan's
> [process rule](PLAN.md#status-of-this-plan--read-before-executing). Line
> references are to `e9e41cbe` on `feat/phase7-performance` (2026-10-08).

## Scope, order, and dependencies

Phase 7 bounded the memory of decoding with fitted parameters. EM
(`estimate_parameters`, both detector families) did not change: it still runs
the dense filter/smoother and keeps three recording-sized posteriors. Users run
on hardware ranging from 24 GB GPUs to 80 GB A100s and CPU-only hosts. Phase 9
makes EM's cost and limits known on each of those tiers, and then removes the
`T × N` memory term.

| Work | Purpose | Ships | Depends on |
|---|---|---|---|
| 9a | Remove fixed costs that dense EM pays but does not need, attribute the rest without a profiler, and document dense-EM limits per hardware tier. | One PR; useful immediately at the sizes dense EM can handle. | Phase 7 benchmarks (`benchmarks/benchmark_em.py`). Independent of 9b. |
| 9b | Bounded-memory EM: run the E-step on the checkpointed, structured-transition path and accumulate the M-step's sufficient statistics there. | One PR after its prototype passes. | Phase 7 checkpointed driver, structured operators and compact outputs; 9a's measurement harness (not its code changes). |

Both sub-phases preserve the statistical model and the current M-steps.
Approximate EM (subsampled time, truncated transitions, stochastic updates)
would change the estimator and is out of scope (see
[overview non-goals](overview.md#non-goals)). The Local-only encoding M-step
(issue #31, `models/base.py:3019-3025`) is unchanged. Viterbi is not part of
this phase.

## How EM runs today

| Step | Code | Needs | Size |
|---|---|---|---|
| Internal refit | `estimate_parameters` calls `self.fit` without `transition_representation` (`models/base.py:4854`, `:5952`), so the fit is dense and requests all-pairs graph distances (`:2276` → `environment.py:588-599`). | Grid, environment graph | All-pairs distances: `8·B²` bytes for `B` interior spatial bins |
| Dense transition | Interior rows/columns of the dense continuous transition (`models/base.py:2924-2926`). | — | `N²` entries for `N` interior state bins |
| E-step | Dense `_predict` (`models/base.py:2958`) → `chunked_filter_smoother` (`core.py:756`), which returns host NumPy posteriors (`core.py:1016-1025`). | Likelihood (cached across iterations only when the encoding is not refit; cleared at `models/base.py:3035`) | acausal, causal and predictive posteriors, each `T × N` |
| Encoding M-step | Local-state probability per row, interpolated to position samples (`models/base.py:2984-3000`), then refit (`:3026`). | `acausal_state_probabilities[:, Local]` | `T` |
| Discrete M-step | `_estimate_discrete_transition` (`models/base.py:3041`; `discrete_state_transitions.py:1092`). Per row, `_aggregate_factorized_xi_by_state_jax` (`discrete_state_transitions.py:226-246`) applies the dense continuous transition to the smoother ratio masked by each target state, inside the scan at `:276`. The posteriors are copied back to the device (`:686-688`). | All three posteriors and the dense transition | Stationary: `S × S` counts (`:625`). Covariate: `(T−1) × S × S` responses (`:548`), fitted at `:767`. |
| Initial conditions | `acausal_posterior[0]` and `acausal_state_probabilities[0]` (`models/base.py:3074-3089`). | First smoothed row | `N` |
| Convergence | Relative change of the marginal log-likelihood below `tolerance` (`models/base.py:3094-3101`; `core.py:1857`). | Evidence total | scalar |
| Final E-step | One more dense `_predict` (`models/base.py:3149-3175`); its posteriors are the returned result. | — | `T × N` |

## Measured baseline

Records: [A100 EM runs](../../../../docs/performance_artifacts/em/a100-em-runs.json),
[CPU EM runs](../../../../docs/performance_artifacts/em/cpu-em-runs.json),
[profile summary](../../../../docs/performance_artifacts/em/em-profile-summary.txt),
[all-pairs distance timing](../../../../docs/performance_artifacts/em/cpu-all-pairs-distances.json).
The workload is `benchmark_em.py`: a 60 s simulated session at 500 Hz (30,000
rows), a 180 cm arena, 64 sorted units at 5 Hz or 8 clusterless electrodes at
20 Hz, and 3 EM iterations. Each run covers the internal fit, 4 E-steps and 3
M-steps.

| Hardware | Grid (`N`) | Family | Warm seconds | Peak device | Peak host RSS |
|---|---|---|---:|---:|---:|
| A100, shared host (breeze) | 4 cm (4,420) | sorted / clusterless | 147–149 / 150–168 | 3.9 GB | 8.3–8.4 GB |
| A100, shared host (breeze) | 2 cm (16,930) | sorted / clusterless | 604–605 / 1,281–1,364 | 15.5 GB | 23.9–24.1 GB |
| A100, isolated, **under cProfile** (zephyr) | 4 cm | clusterless | 77.5–81.1 | 3.9 GB | 8.4–8.5 GB |
| M1 Max CPU, 64 GB | 4 cm | sorted / clusterless | 488–490 / 488–516 | — | 9.8–10.2 GB |

Ranges span the commit before the bottleneck work (`09d1d67a`) and a later
branch commit (`cd106d1e` on the A100); the Phase 7 likelihood speedups barely
moved EM. Paired runs' marginal log-likelihoods differ by at most 0.094
(4.7e-7 relative). No unprofiled, uncontended A100 EM time exists yet; 9a
records one.

Known components, with their limits:

- **All-pairs graph distances.** Without a profiler on the M1 Max, the grid fit
  takes 9.9 s with all-pairs distances versus 0.13 s without at 4 cm
  (2,209 bins). At 2 cm (8,464 bins) it takes 168.4 s versus 0.57 s, with a
  943 MB process peak. Every `estimate_parameters` call pays this once. Time
  grew about `B^2.1` between these grids. Extrapolated (not run), 1 cm (33,124
  bins) would take roughly 50 min and an 8.8 GB float64 matrix. The cProfile
  summary attributes 42.8 s over two 4 cm fits to `_dijkstra_multisource`, but
  that profile also sampled a monitoring thread and inflates per-call Python
  overhead, so use the unprofiled timing.
- **Dense filter/smoother:** 24.9 s over 8 calls at 4 cm (profiled).
- **Likelihood:** 9.3 s over 8 calls at 4 cm (profiled).
- **Discrete M-step:** 4.5 s over 6 calls at 4 cm (profiled). The per-row
  product costs `N²·S`, about 15× more per row at 2 cm than at 4 cm.
- **Unattributed:** roughly half of each profiled 4 cm run is not in the
  entries above.
- **Memory arithmetic, not measured:** one float32 posterior is `4·T·N`
  bytes. At 2 cm that is 2.03 GB for 60 s and 122 GB for an hour. The dense
  transition is at least `4·N²` bytes: 1.15 GB at 2 cm and 17.6 GB at 1 cm.
  Dense EM therefore cannot run an hour at 2 cm on any listed tier. At 1 cm,
  60 s needs about 41 GB of device memory for the transition plus three
  posteriors, more than a 24 GB GPU has.

Lower bounds for bounded-memory EM come from Phase 7 prediction measurements,
not EM measurements. On the A100, a compact prediction of one hour at 2 cm took
363.9 s (sorted) and 680.8 s (clusterless) including compilation; warm cost is
0.096 and 0.14 s per recording second
([validation record](../../../../docs/performance_validation.md#bottleneck-fixes-after-the-comparison)).
On CPU, the Phase 7 hour checks at 2 cm (two units or electrodes) took
46.9–49.2 minutes, measured before the bottleneck fixes. An E-step costs at
least one such pass, so 20 iterations (the default `max_iter`) would take hours
on an A100 and most of a day on CPU.
How many iterations representative data needs is unmeasured.

## Hardware tiers

Every acceptance run names its tier and budget, and follows the
[Phase 7 measurement protocol](phase-7-performance.md#measurement-protocol).

| Tier | Budget | How to measure |
|---|---|---|
| A100 80 GB | The Phase 7 hour-check budgets (8 GiB process RSS, 8 GiB device), unless a run declares a likelihood cache with its own budget | zephyr or breeze; pick an idle device; run in tmux |
| 24 GB GPU | 24 GB device | Emulate on an A100 with `XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`, as the 4 GB and 8 GB pool runs did with fractions 0.05 and 0.1 (`mem4-*` and `mem8-*` in `a100-em-runs.json`; compact predictions, not EM). Emulation bounds memory, not throughput: label speed on a real 24 GB card as unmeasured unless one is used. Consumer 24 GB cards have slow float64, so run float64 parity checks on CPU or an A100. |
| CPU | Host RAM budget: open question 3 | M1 Max (10 cores, 64 GB) is the available host. Record core count and the BLAS/XLA thread settings. |

## 9a: Dense EM fixed costs and resource limits

### Problem

Dense EM pays for all-pairs graph distances in every call, including the
default open-field configuration, whose Euclidean `RandomWalk`
(`use_manifold_distance=False`, `continuous_state_transitions.py:159`) never
reads them (`:224-229`). At 2 cm that is about 168 s per call on the M1 Max. The cost
of copying three `T × N` posteriors between host and device for the M-step is
unmeasured, and about half of the profiled time is unattributed. Users have no documented
limit telling them what session length and grid dense EM can handle on their
hardware.

### Hazard any distance change must handle

Passing `compute_all_pairs_distances=False` to the dense fit is **not** safe on
its own. When distances are not a NumPy array, the manifold-distance
`RandomWalk` silently falls back to Euclidean distances
(`continuous_state_transitions.py:231-243`), which changes the movement model
without warning. Direction-aware `RandomWalk` raises instead (`:248-263`). The
structured path avoids both by building budgeted dense views only for
consumers that need them, and by rejecting opaque custom blocks
(`transition_operators.py:980-1020`). `Environment.get_distances_to_interior_bins`
already accepts deferred distances (`environment.py:1035-1039`; used by the
position penalties at `models/base.py:1184` and `:1297`), but its cost per
call with deferred distances is unmeasured. The 1-D track-graph path computes
its own distances (`environment.py:619-630`) and is unaffected.

### Design options

- **A. Distances on demand (candidate).** The dense fit requests deferred
  distances, as the structured fit does. The dense transition builder then
  obtains a dense matrix only for consumers that declare they need one
  (manifold or direction-aware `RandomWalk`), using the structured path's
  budgeted-view mechanism. Unknown custom transitions keep today's dense
  matrix, the conservative choice. The silent Euclidean fallback must become
  unreachable or raise.
- **B. Faster exact all-pairs** for configurations that really need the dense
  matrix, for example a compiled shortest-path routine instead of NetworkX's
  Python Dijkstra. Distances must be identical, or the differences analyzed
  (equal-length paths can sum in a different order), before adoption.
  B complements A.
- **C. Reuse distances across fits** of an unchanged grid, so EM's internal
  refit reuses a preceding user fit. This helps only when a fit precedes EM,
  and it adds a stale-cache risk. Pursue it only if measurements show repeated
  fits matter.

### Experiments

1. **Unprofiled stage attribution.** Extend `benchmarks/benchmark_em.py` to
   time each stage with synchronization: internal fit (distances and transition
   build separately), likelihood, filter/smoother, host conversion, encoding
   M-step, discrete M-step and result conversion. Run two durations per grid
   to separate fixed from per-row cost and memory. Measure 4 cm and 2 cm on all
   three tiers. Also time the encoding M-step for each sorted and clusterless
   algorithm at 4 cm; fit costs differ widely between algorithms and are
   unmeasured inside EM.
2. **Distance option A**: parity of the resulting transition matrices and EM
   outputs against today's dense fit for Euclidean, manifold and
   direction-aware `RandomWalk`, 1-D track graphs, multiple environments and
   position penalties. Also measure lookup cost with deferred distances.
3. **Option B** only if experiment 2 leaves manifold users paying minutes.
4. **M-step round trip**: measure the device↔host posterior copies at 2 cm.
   Keeping posteriors on the device raises the device peak, so judge it
   against the tier budgets rather than adopting it by default.

### Acceptance

- Outputs unchanged for every configuration in experiment 2: transitions and
  posteriors identical, or within existing tolerances with analysis. Any
  tolerance or reference change needs the approval in
  [PLAN.md approval gates](PLAN.md#approval-gates).
- Paired before/after EM timings on each tier at 4 cm and 2 cm, with stage
  breakdowns saved under `docs/performance_artifacts/em/`.
- Documentation, as tasks of this PR: the `estimate_parameters` docstrings
  (`models/base.py:2796`, `:4729`, `:5832`) state that EM keeps three
  `T × N` posteriors; a "Parameter estimation (EM)" section in
  `docs/performance_prediction.md` gives measured dense-EM peaks and the
  largest measured session per grid and tier; and a `CHANGELOG.md` entry.

### Deliberately not in 9a

- Bounded-memory EM (9b).
- GLM/MRF/diffusion fitting speed beyond measuring it inside EM.
- Likelihood compilation work. That belongs to the Phase 7 branch, whose
  improvements EM inherits through the shared backends.

## 9b: Bounded-memory EM

### Requirement

EM whose device and host working memory do not grow with `T × N`. Allowed
growth with `T` is limited to `O(T·S²)` statistics plus the outputs the user
requests, under the declared tier budgets. The M-steps must be mathematically
unchanged, and the dense path stays available as the reference and as the
fallback for configurations the structured operator does not support.
Unsupported configurations over budget are rejected by preflight, as Phase 7
does for prediction.

### What the M-steps need, and where a checkpointed pass can supply it

| Consumer | Statistic | Size | Source in a checkpointed E-step |
|---|---|---|---|
| Encoding M-step | Local-state probability per row | `T` | Existing compact state-probability output |
| Stationary discrete M-step | Expected counts | `S × S` | **New:** per-block contraction during the backward replay |
| Covariate discrete M-step | Expected responses per row | `(T−1) × S × S` (230 MB float64 for an hour, `S = 4`) | **New:** the same contraction, emitted per row |
| Initial conditions | First smoothed row and its state probabilities | `N` | Final carry of the backward pass |
| Convergence and monotonicity warnings | Marginal log-likelihood | scalar | Forward pass, using the stable evidence accumulation approved in Phase 7 |
| `degenerate_timesteps_` | Degenerate rows | ≤ `T` | Forward-pass diagnostics |
| Returned result | The user's requested outputs | Phase 7a output modes | Existing |

Per row, the counts are the discrete weight times the sum, over source bins
in state `i`, of the filtered probability times the continuous block from
state `i` to state `j` applied to the smoother ratio
`smoothed_{t+1} / predicted_{t+1}` restricted to state `j`. The checkpointed
backward step (`checkpointed_inference.py:157-169`) already forms that ratio
with the same zero-denominator rule (`_divide_safe`; compare
`discrete_state_transitions.py:249-256`). It applies every block to it, but
sums over target states with the discrete weights before returning.
`FusedBlockTransition.backward` (`transition_operators.py:573`) also folds
rank-one blocks together. The statistic therefore needs the per-block
products before that sum. That is the central prototype question.

### Design options

- **A. Checkpointed sufficient statistics (candidate).** Accumulate the table
  above during the forward pass and the replayed backward pass.
  - Memory: chunk-bounded, plus the statistics.
  - Arithmetic: the same as today; only summation order and chunking change.
  - Cost: at least one compact prediction per iteration, plus the
    contraction, plus a second likelihood evaluation during replay.
- **B. Dense posteriors on disk**, feeding today's M-step in row chunks.
  Rejected for the target workload: three posteriors of an hour at 2 cm are
  366 GB to write and read every iteration.
- **C. Approximate statistics.** Out of scope; this changes the estimator.

Sub-choices for A, to settle by experiment:

- **Contraction.** Either expose per-pair block products from the operator
  (cheap if done inside the existing backward step), or apply the backward
  operator once per target-state mask (about `S×` the backward cost).
- **Likelihood on replay.** Recompute it, keep forward chunks in host RAM
  within a declared budget (60 s at 2 cm is 2 GB), or use a disk cache
  (`benchmarks/benchmark_replay_cache.py` has a prototype). The encoding refit
  changes the likelihood every iteration, so caching across iterations would
  cost as much memory as dense EM.
- **Precision.** New contractions on CUDA must use the highest matmul
  precision, as the Phase 7 production contractions do. Default TF32
  measurably changed state aggregation there.

### Experiments

1. **Contraction prototype.** Compare per-row counts with
   `_aggregate_factorized_xi_by_state_jax` on random valid posteriors:
   - every supported block kind (uniform, identity, Gaussian grid, dense
     fallback), restricted interiors, several environments, and per-row
     covariate weights;
   - float64 and float32.

   Report the added per-step cost over a plain backward step on an A100 and on
   CPU (extend `benchmarks/benchmark_checkpoint_scan_steps.py`).
2. **One-iteration parity.** Checkpointed against dense E-step statistics on
   small grids: counts or responses, Local weights, the first smoothed row and
   evidence.
3. **Full EM parity**, comparing the trajectories of marginal log-likelihood,
   discrete transitions, refit encodings and initial conditions. Cover:
   - both families; stationary and covariate transitions;
   - frozen transition rows; `is_missing`; several environments; 1-D track
     graphs;
   - models without a Local state; each `estimate_*` flag turned off.

   In float32 the dense driver sums evidence naively; the Phase 7 example
   lost 1.8% over 1.8M rows, against EM's default relative `tolerance` of
   1e-4. Compare marginal log-likelihoods against a float64 run or the
   stable-sum oracle, not the float32 dense total.
4. **Memory against duration.** At least two durations per grid on each tier.
   Device and host peaks must stay flat apart from the `O(T·S²)` statistics
   and requested outputs.
5. **Likelihood strategy.** Time recomputation, the host cache and the disk
   cache on each tier.
6. **Hour scale, staged.**
   - Smoke test: 60 s, one iteration.
   - Extrapolate runtime and memory.
   - Full hour at 2 cm, three iterations, on the A100 and the 24 GB-emulated
     tier, both families; 1 cm where the budget allows.
   - CPU: 60 s and 10 min measured. A full hour only if the user accepts the
     extrapolated time.

   Record iterations to convergence wherever runs converge.

### Acceptance

- Experiments 1–3 pass at tolerances proposed from the measured float64 and
  float32 differences and approved before use. No golden or snapshot
  references change.
- Experiment 4 shows flat memory in `T` on every tier. Experiment 6's measured
  runs fit their declared budgets. Projections and untested combinations are
  labelled.
- Public API (open questions 1–2): how checkpointed EM is selected and what it
  returns, documented in the `estimate_parameters` docstrings and in the EM
  section of `docs/performance_prediction.md` (with measured tier limits),
  plus a `CHANGELOG.md` entry. Code, tests and docstrings do not name this
  plan or its phases.
- The dense EM path remains as reference and fallback. Nothing in it is
  removed by this phase.

### Deliberately not in 9b

- Bounded-memory Viterbi.
- Changing the encoding M-step's Local-only weighting (issue #31).
- Approximate or stochastic EM.
- Distance and attribution work (9a).

## Open questions

1. **Selecting checkpointed EM.** Should `estimate_parameters` take
   `transition_representation` (forwarded to its internal fit) and a chunk
   size, and should the default stay `"dense"` until 9b is qualified?
2. **Return value.** Should checkpointed EM return compact outputs by default
   (the dense default returns the full `T × N` acausal posterior), following
   `predict`'s output modes?
3. **CPU budget.** What host-RAM budget should CPU acceptance use?
4. **24 GB card.** Is a real 24 GB GPU available for throughput measurements,
   or should 24 GB speed stay labelled unmeasured?

## Review

Before opening each sub-phase's PR, dispatch `code-reviewer` against the diff
and confirm:

- The sub-phase's acceptance items are met with saved artifacts.
- "Deliberately not in" lists are honored.
- Parity tests exercise the stated configurations and compare against an
  independent reference (dense EM, float64 or the stable-sum oracle), not
  against the implementation under test.
- No tolerance or reference file changed without the recorded approval.
- Documentation tasks are done in the same PR.
