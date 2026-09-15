# Overview

## Goals

1. Fix the two core HMM defects in [phase 0](phase-0-core-hmm.md), then the four
   likelihood/integration defects that change decoded posteriors without being
   caught by the existing tests.
2. Remove numerical failure modes producing NaN or impossible values (negative
   squared distances, mixture weights summing to 24, means collapsing to 0 under
   a global weight rescale).
3. Establish one time vocabulary ([C3](shared-contracts.md#c3--time-vocabulary))
   and calibrate Poisson intensities to real bin durations.
4. Make full-session decoding feasible for the representative workload below
   through bounded working memory, incremental outputs, and structured
   transitions, preserving the corrected statistical model and numerical
   contracts.
5. Take additional kernel and compilation improvements when profiling shows a
   benefit within the same memory budget.

## Non-goals

- Retiring the duplicate `clusterless_kde.py` / `clusterless_kde_log.py`
  implementations — blocked on the known ~3e-2 log-vs-prob golden parity gap.
- A typed/versioned encoding-model schema replacing the `dict` +
  `**encoding_model` splat. Phase 6d adds a single unit marker, not a schema.
- Clearing the historical mypy backlog (190 errors at the original audit).
  Phases must not add errors relative to their pinned baseline; a sweep is separate.
- A backend-owned EM damping rebuild protocol. Phase 5 rejects damping where it
  cannot be supported.
- Changing spatial/temporal resolution, truncating Gaussian tails, or using
  approximate smoothing to obtain performance parity. These would change the
  scientific model and require separate scope and validation.
- A bounded-memory redesign of whole-session EM or Viterbi. Phase 7 targets
  decoding with fitted parameters; its filtering/smoothing results must not be
  presented as evidence that every inference or fitting API scales similarly.

## Representative workload

User-supplied target: a **180 cm × 180 cm environment**, **1 cm or 2 cm bins**,
and **one hour of recording at 2 ms per decoding bin** (1,800,000 observations).
Use the default non-local model as the first production profile: Local and
No-Spike each have one bin (`local_position_std=None`); Non-Local Continuous and
Non-Local Fragmented each have a complete spatial distribution.

Let `B` be the usable spatial-bin count, `N = 2B + 2` the combined hidden-bin
count, and `T` the decoding length. The following are calculated storage
estimates, **not measured peak allocations or runtime promises**. They assume
the whole arena is usable and omit padding/interior-mask differences; benchmark
reports must record actual fitted dimensions, including padded output bins.
GB and TB are decimal units; array estimates use float32.

| Quantity | 2 cm bins | 1 cm bins |
|---|---:|---:|
| Nominal spatial grid | 90 × 90 | 180 × 180 |
| Spatial bins `B` | 8,100 | 32,400 |
| Combined hidden bins `N` | 16,202 | 64,802 |
| One full-session posterior, `4TN` bytes | 116.7 GB | 466.6 GB |
| Three posterior arrays | 350.0 GB | 1.40 TB |
| One dense transition, `4N²` bytes | 1.05 GB | 16.80 GB |

Initial transition construction currently uses NumPy float64, doubling the last
row before accounting for intermediate copies. The core accumulates filtered,
predicted, and smoothed arrays even when the public result omits some of them;
result conversion also expands arrays to include non-interior bins. Increasing
`n_chunks` alone does not bound this full-session storage. Likelihoods, fitted
encoding kernels, checkpoints, and temporary workspace add further costs.

The four discrete-state probabilities require only **28.8 MB** for the hour in
float32. Saving that compact result still requires spatial inference internally.
Saving a full spatial posterior requires the corresponding disk/output capacity
even when working RAM is bounded.

### Acceptance at representative dimensions

- **Numerical agreement:** filtering, smoothing, evidence, and discrete-state
  marginals agree with the corrected dense reference on tractable fixtures at
  existing applicable tolerances. Preserve impossible-data fallback, NaN
  visibility, and supported differentiation behavior. Phase 7 changes neither
  golden data nor numerical tolerances.
- **Allocation follows requested outputs:** compact and incremental-output modes
  retain no full `T × N` spatial arrays in host or device RAM. Full spatial
  results can be written in chunks; selecting a smaller result must affect the
  computation and allocation, including final result conversion.
- **Measured resource bounds:** record a host-RAM budget, device-memory budget,
  checkpoint/output disk capacity, and chunk size before each large benchmark.
  Demonstrate compliance over increasing duration at fixed chunk size. Count
  resident model/transition data, checkpoint caches, transient workspace, and
  conversion/I/O buffers. Identify any structures that still grow with `T`.
- **Representative measurement:** exercise both nominal grid sizes, the two core
  transition paths, and supported sorted/clusterless backends. Use small dense
  reference cases, then realistic spatial dimensions with shorter durations,
  then the full hour after the resource bounds have been demonstrated. Repeated
  synchronized kernel timings and end-to-end times serve different purposes;
  report compilation, likelihood calculation, HMM work, transfers, and output
  I/O separately where measurable.
- **Evidence limits:** retain Phase 0's small-kernel results as regression
  evidence. Neither them nor the exploratory 10,000-bin CPU comparison
  establishes full-hour or GPU feasibility. Actual target hardware, neuron/
  tetrode counts, spike rates, encoding duration, and output requirements remain
  to be recorded; untested combinations must be labelled explicitly. Do not
  allocate an infeasible dense full-hour reference or require full-hour jobs in
  routine CI.

These targets add work to [Phase 7](phase-7-performance.md). They do not reopen
Phase 0/1 or require architectural work before shipping the remaining correctness
fixes. [Phase 3](phase-3-chunk-boundary.md) establishes correct likelihood chunks;
the integration baseline for Phase 7 incorporates the settled likelihood and
time/exposure contracts and relevant Phase 8 fixes. Follow the release dependencies
in [PLAN.md](PLAN.md#execution-order-and-baselines), rather than numeric order alone.

## Architecture map

```
detector.fit()
  └─ fit_encoding_model()          models/base.py:2900 (clusterless), :3890 (sorted)
     ├─ is_group = is_training & is_encoding & is_environment
     ├─ group spikes               _get_group_spike_data / _get_group_spikes
     └─ registry fit fn            likelihoods/__init__.py:41 (sorted), :59 (clusterless)

detector.predict()
  └─ _predict()                   models/base.py
     ├─ bind compute_log_likelihood callback (clusterless or sorted)
     └─ core chunked HMM          consumes cached likelihoods or calls the callback
        ├─ chunked_filter_smoother
        └─ chunked_filter_smoother_covariate_dependent
```

**The two core paths are stationary vs. covariate-dependent transitions — not
sorted vs. clusterless.** The first draft of this plan mislabelled them, which
would have let a test suite exercise one path twice and miss the other. Any phase
touching `core.py` must cover both, and both detector types route through
whichever path their transition model selects.

Both filters call `_condition_on`, which uses `_normalize`; both smoothers also
call `_normalize` directly. Phase 0 validated these callers independently of C1
and C3b. Its completed coverage includes the genuine-zero-support fallback and
NaN visibility; later phases preserve those contracts.

## Cross-cutting risks

| Risk | Mitigation |
|---|---|
| Broad Phase 6 reference updates mask a regression from an earlier phase. | Phases 0–5 ship first; review 6a/6c separately from 6b/6d and attribute only observed reference changes. |
| Treating every small normalizer as impossible hides a valid Bayesian update; requiring every evidence value to be finite erases true impossibility. | Phase 0 tests positive-support observations, true zero support, and NaN inputs separately against an independent reference. |
| Phase 1 exposes all-zero-exposure groups whose guards used to live in phase 4. | Zero-exposure handling moved **into** phase 1, so it is independently shippable. |
| Phases 2 and 4 both move GMM outputs; a combined diff is unattributable. | Separate changes and trace changed primitives. Posterior normalization and temporal propagation can change bins whose own likelihood was unchanged. |
| Fixing float32 cancellation changes well-conditioned results too, via summation order. | Assert invariants (Mahalanobis ≥ 0) plus `rtol=1e-5` parity on well-conditioned fixtures, not bit-exactness. |
| A phase's acceptance criteria are never run against its own proposed fix. | Reproduce the defect or contract violation, then exercise the phase's prototype and acceptance checks before calling it ready. Performance work also needs a measured baseline. |
| Small CPU kernels appear fast while full-session decoding exhausts memory. | Use the representative dimensions, account for model/output storage, and measure host/device peaks and end-to-end throughput. |
| Chunked smoothing resets context or drops spikes at boundaries. | Phase 3 preserves global event ownership; Phase 7a carries both forward and backward boundary messages and tests parity across chunk sizes. |
| An apparently exact transition optimization changes boundaries or silently approximates a different movement model. | Phase 7b requires row normalization, forward and backward operator parity, explicit capability checks, and a tested fallback. |

## Deferred with triggers

Recorded so they are not silently dropped.

| Item | Trigger to revisit |
|---|---|
| Duration-calibrated continuous/discrete transitions, so nonuniform detector bins become valid. Currently `core.py:643` applies one transition per row regardless of duration; phase 6c restricts detectors to uniform bins instead. | When a user needs nonuniform decoding, or when variable-`dt` event-based decoding is scoped. |
| Retiring the linear `clusterless_kde` implementation in favor of the log path. | When the log-vs-prob golden parity gap closes. Profiling and, if required by the production memory budget, tiling the linear path's `(n_encoding_spikes, n_position_bins)` kernel are now assigned to Phase 7c; they need not wait for backend retirement. |
| `GaussianMixture.score_samples` computes discarded responsibilities (`gmm.py:1068`). | Next GMM cleanup. |
| Typed/versioned encoding-model schema. | When a third consumer of the encoding dict appears, or after phase 6d's unit marker proves insufficient. |

## Environment

All commands run through `uv run`. Baseline at planning time:

```
uv run pytest src/non_local_detector/tests/likelihoods -q
# 376 passed, 2 skipped, 23 warnings in 173s
uvx ruff check src/non_local_detector/likelihoods/     # All checks passed
uv run mypy src/non_local_detector/likelihoods/        # 190 errors in 11 files
```

These are historical baseline results, not current test counts. Preserve the
covered tests and lint cleanliness; remeasure the mypy baseline on the pinned
implementation revision rather than assuming the old count still applies.

Follow-up core audit (2026-09-14, CPU, JAX 0.9.0, NumPy 2.4.1, Python 3.13):
116 existing tests passed across `core/test_core_utilities.py`,
`core/test_hmm_algorithms.py`, `core/test_chunked_parity.py`, and
`integration/test_core_kde_integration.py`. Both phase-0 reproductions still
failed the independent mathematical reference. The run emitted 26 warnings,
including requested float64 arrays being truncated to float32; phase 0 requires
a separate run that actually enables and verifies float64. These results are a
pre-fix audit baseline. Phase 0 subsequently passed 1329 full-suite tests with
4 skipped and all 64 reference cases with x64 enabled; see its implementation
record rather than treating the audit failure as current readiness.

## Open questions

**Blocking.**

- **C1's package-wide degeneracy policy is deferred**, not chosen. "Floor only
  `-inf`" is non-monotonic (an impossible observation scores −34.54 while a
  1e-44 one scores −101.31). Phase 2 shipped the GMM arithmetic corrections
  under the existing floors; the recommended replacement is a background
  firing model, which needs a separate modelling proposal. See
  [shared-contracts.md](shared-contracts.md#c1--degeneracy-policy-for-log-intensities).
- **C3b's draft `sample_cell_durations` disagrees with full-cell exposure**:
  it returns `(N-1)/N` of that reference and gives a single sample zero exposure.
  Choose acquisition endpoints, single-sample requirements, and a policy for
  long timestamp gaps (exposure vs. missing data). The encoding-cell migration
  in 6a and rate calibration in 6b require this decision; independent decode
  vocabulary and validator prototypes can proceed.

**Delegated judgement calls.**

- Phase 4: diagnose new collapse warnings against the frozen reference. Neither
  increasing fixture regularization nor lowering a floor is an automatic fix;
  numerical-criterion changes follow the repository review process.
- Phase 5: the recorded damping audit found no coherent supported backend for
  the existing blend. Recheck that inventory at implementation time and reject
  unsupported damping before mutation; a general rebuild protocol is out of scope.

## Found during review, not yet scheduled

- **`save_model` is broken for every GLM detector.** Reproduced end-to-end:
  `sorted_spikes_kde` saves fine, `sorted_spikes_glm` raises
  `NotImplementedError: Sorry, pickling not yet supported` because the encoding
  dict holds a Patsy `DesignInfo` (`sorted_spikes_glm.py:316`) and `save_model`
  uses stdlib `pickle` (`models/base.py:2337`). This is a live user-facing bug
  independent of this plan and should be fixed on its own, not folded into
  phase 6d. Remedies: reconstruct `DesignInfo` from the formula and knots at load
  time, or store the spline basis numerically instead of the Patsy object.
