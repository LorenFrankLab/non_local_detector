# Time-grid and rate-unit validation

**Plan status:** Phase 6a–6d has accepted checkpoints and source commits listed
in the scope record. The 2026-10-06 PR review exposed additional package and
consumer defects; their fixes and acceptance evidence are recorded below.
The earlier complete-suite checkpoint passed **2,397 / 6 skipped / 0 failed**;
the follow-up frozen suite passes **2,656 / 6 skipped / 0 failed**. The user approved the three 6b snapshot
corrections on 2026-10-05; they remain applied. Separate 6c and 6d completion
checks are recorded below; downstream release coordination remains pending. See the
[phase scope record](../.claude/docs/plans/likelihood-defect-remediation/phase-6-worktree-scope.md).

The implementation changes physical rate units and decode event ownership.
Saved models and downstream scientific results require refitting/recomputation;
agreement on the historical 500 Hz fixtures does not imply equivalence for other
tracking or decode rates. Numerical tolerances and golden data are unchanged
from accepted 6a. Relative to GitHub main, the PR includes the previously
approved endpoint-migration golden update in `bd5a8fa`; this review does not
regenerate any golden file or change an existing assertion tolerance.

## 1. Differences

Three existing likelihood snapshots assume unit-duration expected counts, even
though their decode bins last 2 or 10 seconds:

| Snapshot | Existing reference | Corrected result |
| --- | --- | --- |
| Sorted KDE argmax | Interior bin 3 | Interior bin 1 |
| Sorted KDE top three | `{1, 2, 3}` | `{1, 3, 4}` |
| Clusterless KDE argmax | Interior bin in `{5, 6, 7}` | Interior bin 9 |

The approved reference changes are recorded in
[`time_grid_snapshot_reference.patch`](time_grid_snapshot_reference.patch).
The user approved these changes on 2026-10-05 and the patch has been applied.
All eight repository snapshot tests pass; all four likelihood snapshot bodies
also passed their independent reference checks before approval.

All four pinned golden regression tests pass with the existing tolerances and
files. Measured maximum absolute differences from their saved float32 outputs:

| Golden fixture | Maximum posterior difference |
| --- | --- |
| Clusterless decoder | `4.25e-7` |
| Random-walk transition decoder | `4.69e-7` |
| Sorted-spikes decoder | `7.15e-7` |
| Nonlocal detector | `2.38e-7` |

The nonlocal state probabilities differ by at most `2.38e-7`. These values are
within the pre-existing golden tolerances (`rtol=1e-6`, `atol=1e-6`).

## 2. Explanation

Encoding rates are weighted events divided by weighted **seconds**, rather than
position sample counts. Every sorted-spike likelihood scores expected counts
`rate_Hz * bin_duration`, and every marked-process likelihood scales both its
ground-process and event terms by the actual bin duration.

For one observed event, the sorted reference is
`log(rate_Hz * duration) - rate_Hz * duration`. The previous reference omitted
duration. With a two-second bin, a single event favors a rate near 0.5 Hz rather
than 1 Hz. Likewise, one marked event in a ten-second bin can favor a lower-rate
location away from the encoding mode: predicting many events is penalized by
the integrated ground process. The proposed clusterless reference calculates
Gaussian spatial/mark densities in float64 with SciPy and includes this term.

Rate safeguards are converted at the historical 2 ms reference; density
safeguards retain their original units. GLM regularization is expressed per
second, with default 0.5 corresponding to the former 0.001 at 500 Hz. MRF
safeguards and objective unit constants are rescaled without changing solver
tolerances.

## 3. Invariant and regression validation

- Analytic basis integrals cover full, clipped, disconnected, and NaN-split
  tracking support; adjacent intervals conserve boundary-event mass.
- All eight registered backends return Hz rates and weighted exposure seconds.
  Five events in one second give 5 Hz with either 30 Hz or 500 Hz tracking.
- Empty-bin likelihoods equal minus the integrated ground rate. Doubling bin
  duration adds `log(2)` to each event's evidence in both local and nonlocal
  paths. Requested rows agree with slices of full likelihoods.
- Singleton recording support is explicit. GLM tests cover singleton samples,
  stationary tracking, and NaN gaps leaving one finite sample, including an
  intercept-only one-bin environment.
- Public API tests cover legacy/cached model rejection, actual bin boundaries,
  effective missingness, pickle and NetCDF serialization, concatenated
  final-edge flags, Viterbi,
  and singleton results retaining both time and spatial dimensions.
- The runnable migration example decodes 400 bins in two independent sequences
  with 68 missing observations and normalized finite posteriors.
- Pinned golden outputs are finite and normalized within float32 precision;
  the largest measured normalization error is `3.58e-7`. Existing probability,
  transition, covariance, chunking, and likelihood property tests are included
  in the full regression run.
- Public fit/predict calibration: 16 cases across all eight backends, with
  30/500 Hz tracking and 2/4 ms decoding, pass the independent known-rate
  reference (`rtol=1e-5` unchanged). All 16 fail against accepted 6a. All seven
  supported model save/load paths preserve exact result datasets and units.
- Another 26 cases cover missing/unsupported per-entry markers, unsupported
  contract versions, rejection before likelihood work through predict, Viterbi,
  likelihood and private cached paths, plus legacy-estimation recovery in both
  families. GLM unit/dispatch tests run in memory; its Patsy pickle defect is
  deferred and documented publicly.
- Final focused support/rate/persistence/chunk run: 107 passed. The reviewed
  chunk/API suite passed 111. Selected physical-rate/API float64 run: 101 passed.
  The three core dtype cases skipped in the default environment passed during
  6a completion; core is unchanged in 6b.
- Independent final correctness and API/UX reviews report no concrete package
  defect. Ruff/format and whitespace checks pass. Targeted mypy has 158 existing
  errors versus 167 on accepted 6a under the same options on shared files (plus
  the new exposure module), with no new
  diagnostic messages; type checking is not globally clean.
- Pre-approval complete suite: **2,198 passed / 6 skipped / 3 failed** (694.52 seconds).
  The three failures were the duration-sensitive snapshot references above,
  subsequently approved and corrected. All four golden tests and the other five snapshots pass. Log:
  `/private/tmp/nld-phase6b-final-all-tests.log`.
  The final annotation/NumPy type-narrowing edits also pass the 107-test focused
  run; they do not change the already prepared NumPy encoding arrays.
- After explicit approval and applying the patch, **all eight snapshots pass**
  (`/private/tmp/nld-approved-snapshots.log`). No golden files or numerical
  tolerances were changed.
- Lint, formatting, whitespace checks, and notebook code-cell compilation pass.
  The notebook's external data were unavailable, so it was not run on real NWB
  recordings. Spyglass/DataJoint integration still requires downstream migration.
- All migration-guide code blocks compile; the actual conservative local-gap
  example and documented row-aware callback recipe were executed successfully.
  The accepted 6a source hashes remain intact, and all eight pinned golden
  input/output files are byte-identical to that checkpoint. Reviewed 6b source
  and its separate diff are saved at `/private/tmp/nld-phase6b-reviewed` and
  `/private/tmp/nld-phase6b-vs-accepted6a.patch`.

The review also reproduced altered physical duration when a legacy custom chunk
callback received a closing edge moved by one ulp. At Unix origin, a 5 Hz empty
bin changed by about `1.19e-6` solely from chunking. Partial requests now require
`row_slice_aware` with full original edges and global rows. A regression fails
before that guard and passes afterward; marked No-Spike chunks reproduce every
original representable duration exactly.

A separate controlled 500 Hz fixture compares accepted 6a to 6b with ownership
and decode centers held fixed. Centers match exactly; maximum posterior
differences are `3.58e-7` for sorted decoding and `2.38e-7` for nonlocal detection.
Maximum log-likelihood difference is `1.91e-6`, consistent with float32 unit
conversion arithmetic; finite/normalization checks pass. This comparison
attributes rate effects independently of the earlier 6a time-coordinate change.

## 4. Concrete before/after case

Consider five spikes during one second of 30 Hz tracking, decoded in 2 ms bins.
The old rate was `5 / 30 = 0.166667` spikes per position sample. Treating that
value as a decode-bin count predicted `0.166667` events in each 2 ms bin.

The corrected fit gives `5 / 1 second = 5 Hz`, and each decode bin predicts
`5 * 0.002 = 0.01` events. The previous expected count was 16.67 times too large.
For an empty bin, the correct log likelihood is `-0.01`, rather than `-0.166667`.
At matching 500 Hz tracking and decoding, the new physical units recover the
historical expected count, which explains the preserved golden regressions.

This correction can change decoded posteriors and scientific conclusions when
the original tracking and decode rates differ. Refit models and recompute
likelihoods, posteriors, shuffles, and derived statistics after migration.

## Phase 6c completion validation

The approved snapshot corrections are applied. Subsequent 6c fixes enforce the
same grid policy through cached prediction/base Viterbi and both constructors,
and reject invalid clock provenance and overflowed durations. No tolerance
constant, floor, solver criterion, golden file, or valid-grid likelihood formula
is changed.

- Red→green reproductions cover the cached uniformity/learned-width bypass,
  unresolved generator widths before allocation, outer-edge precision after
  center conversion, finite endpoints with overflowed differences, and invalid
  or impossibly permissive saved clocks. Covariate validation order is checked
  in both detector families with transition/fit/likelihood sentinels.
- Final focused run: **300 passed**, including the grid/model/chunk/support
  modules, all four goldens, and all four likelihood snapshot references.
  `/private/tmp/nld-phase6c-final-focused.log`.
- Final float64 run: **120 passed**, including the three core dtype cases
  skipped in the default environment. `/private/tmp/nld-phase6c-final-x64.log`.
  Targeted mypy retains 158 existing diagnostics, matching reviewed 6b with no
  new messages. Lint/format and whitespace checks pass.
- Final complete suite on the final source: **2,269 passed / 6 skipped / 0
  failed** (1,218.45 seconds). This includes all four unchanged goldens and all
  eight approved snapshots. `/private/tmp/nld-phase6c-final-full.log`.
  Source hashes are pinned in
  `/private/tmp/nld-phase6c-final-run-source-hashes.json` and still match.
  Three dtype skips pass in the separate float64 run; the other three skips
  are two manual visualization tests and the unavailable optional sortingview
  dependency. An earlier integrated run passed 2,260 before the last nine
  regressions were added; the final run includes them.
- Controlled decoder/nonlocal arrays are bit-identical to reviewed 6b for
  centers, likelihoods, posteriors, and state probabilities. Both generated and
  manual 1.8-million-bin 2 ms grids pass at origins 0 and `1.7e9`; the generated
  edges are byte-identical to the original formula. No full-session spatial
  posterior allocation is used for this check.
- Both final independent reviews find no remaining concrete 6c correctness or
  API/UX defect. The runnable migration example still returns 400 bins across
  two independent sequences with 68 missing observations. Notebook code cells
  compile; real NWB/DataJoint and downstream adapter execution remain pending.

Acceptance records are in the
[6c plan](../.claude/docs/plans/likelihood-defect-remediation/phase-6c-uniform-bins.md#completion-review--2026-10-05).

## Phase 6d completion validation

The final compatibility audit validates the complete set of active spike
encoding keys before dispatch. Deleting a required dictionary or the complete
encoding attribute can no longer bypass the unit guard, including through a
cached likelihood or base Viterbi callback. Unused legacy entries and
No-Spike-only entries do not block valid decoding; No-Spike retains its
constructor Hz rate and the global fitted clock/contract checks.

- New portable regressions: **128 cases** across both detector families,
  multiple groups/environments, missing/unsupported entries and markers,
  constructor/get-params isolation, legacy file inspection without `__init__`,
  truthful weighted refits, and all eight direct predictors. Three red runs
  establish the entry/marker/container defects before their respective fixes;
  logs are linked in the 6d completion record.
- Final focused suite: **314 passed**, including the generic fixed-likelihood
  core estimator regressions. `/private/tmp/nld-phase6d-final-focused.log`.
- Float64 compatibility/persistence: **177 passed**.
  `/private/tmp/nld-phase6d-x64.log`. All seven serializable backends preserve
  exact result datasets after save/load. GLM unit/dispatch coverage remains in
  memory; its independent Patsy serialization defect is deferred.
- Controlled decoder/nonlocal arrays are bit-identical to accepted 6c. All
  eight pinned golden input/output files are byte-identical to accepted 6a.
  No likelihood formula, snapshot/golden reference, numerical tolerance, or
  convergence criterion changed during this completion audit.
- Targeted mypy reports the same **158 existing diagnostics** as accepted 6c,
  after normalizing line numbers. Both independent final compatibility and UX
  reviews report no remaining concrete defect.
- Final complete suite: **2,397 passed / 6 skipped / 0 failed** (721.62 seconds),
  including all four goldens and all eight approved snapshots.
  `/private/tmp/nld-phase6d-final-full.log`. The six skips are the unchanged
  dtype/manual-visualization/optional-dependency skips described above.
  The frozen 175-file source manifest at
  `/private/tmp/nld-phase6d-final-run-source-hashes.json` still matches.
  Ruff/format pass on all 61 changed Python files; whitespace checks pass.
- The accepted checkpoint is `/private/tmp/nld-phase6d-accepted`, with hashes
  in `/private/tmp/nld-phase6d-accepted-source-hashes.json` and a separate
  runtime/test diff at `/private/tmp/nld-phase6d-vs-accepted6c.patch`.
  Earlier accepted checkpoints remain intact. Package completion is separate
  from downstream Spyglass migration and release coordination.

Acceptance records are in the
[6d plan](../.claude/docs/plans/likelihood-defect-remediation/phase-6d-model-compat.md#completion-review--2026-10-05).


## PR review follow-up — 2026-10-06

The review baseline is published commit `338b4f8`, preserved in the accepted 6d
checkpoint. The fixes stay within Phase 6 support, units, decode rows, and
compatibility. No new snapshot/golden reference, existing tolerance, convergence
criterion, floor policy, or transition model is introduced.

### Reproductions and scientific changes

- An explicit tracking interval `[-0.5, 2.75]` on samples `[0, 1, 2, 3]`
  incorrectly lost endpoint support when only the sample at 3 was NaN.
  Declared support now wins over missing samples outside it. Exposure changes
  from 3 to the correct 3.25 seconds, supported events from 3 to 4, and the
  constant-position mean rate from 1 to `4 / 3.25 = 1.2307692307692308` Hz.
  Internal NaNs still split support. Six new regressions establish the defect;
  100 seeded implicit-support and legacy-checker cases remain bit-identical.
- Clusterless rescaling previously integrated Hz rates at unit sample spacing.
  It now integrates physical time, including exact integrals of the piecewise
  linear rate at off-knot spikes. Independent analytic integrals differ by at
  most `4.44e-16`. Sorted checks add explicit Hz/sample-time/bin-edge paths,
  retain their expected-count default, and keep separate trials separate.
  Hz arithmetic converts to float64 before integer or float32 accumulation;
  explicit legacy paths retain their existing arithmetic. Scientific focused
  checks: **177 passed**; logs are `/private/tmp/nld-review-fix-science-green.log`
  and `/private/tmp/nld-review-fix-science-numerics.log`.
- Stored covariate-transition tensors previously reused a mismatched time axis
  during prediction. Prediction now rejects that mismatch before likelihood/HMM
  work, including cached paths; new aligned prediction covariates still work.
- Entirely missing tracking previously hid dropped units and changed waveform
  dimensions. All eight backends validate active fitted populations before
  neutral likelihood shortcuts. Empty populations fitted as empty, NoSpike-only
  states, and unused encoding entries remain supported.
- Returned bin bounds and missingness previously shared caller memory. Changing
  an input after prediction could silently change saved provenance. Results now
  own those arrays. The model-input/provenance module has **114 passing cases**,
  including red runs for both all-missing dimensions and input aliasing.
  Independent model/compatibility review: **242 passed**.
- Modified notebook/profile calls retained removed parameters, displayed fitted
  Hz fields multiplied by frequency, or paired 500-bin results with 31 raw
  camera rows. Actual consumer calls now execute on small fitted models;
  perfect trajectories yield 500 zero ahead/behind distances. The public
  alignment helper preserves missingness, declared segments, categorical
  ownership, and shortest-arc circular interpolation. Stale outputs in changed
  notebook cells are cleared. The runnable independent-interval example also
  uses the requested shared stop when constructed edges round past it.

### Chunk performance and preservation

The former chunk callback scanned the full grid five times and allocated
28.8 MB for 20 requested rows on a 1.8-million-bin recording with 50 spatial
bins. A call-local validated grid and tracking preparation now avoid repeated
full-grid scans. Prepared callback workspace is about **39.6 KB** and median
latency **0.76 ms**, versus **7.16 ms** before the fix. With a learned clock and
NaN gaps it remains about 39.6 KB / 0.80 ms. Workspace stays flat from 18,000 to
1,800,000 bins. These measurements exclude the one-time full-grid validation,
tracking preparation, and recording-sized posterior output; standalone calls
still validate raw inputs. Sixty tests enforce full/chunk equality and forbid
per-chunk whole-grid validation or mask rebuilding. Logs:
`/private/tmp/nld_review_chunk_fixed.log` and
`/private/tmp/nld_review_chunk_clock_gaps.log`.

Controlled sorted-decoder and nonlocal-detector time, likelihood, posterior,
and state arrays are bit-identical to accepted 6d. All eight golden input/output
files remain byte-identical. The new physical/checker/model/grid regressions
pass with float64 enabled: **237 passed**
(`/private/tmp/nld-review-fix-x64.log`). Targeted mypy retains the same **158**
normalized diagnostic messages as accepted 6d; no global type-clean claim is
made.

The failed Linux Python 3.11 CI assertion normalized a manually constructed
float32 posterior to `1.0000001192092896`, one ulp above 1. Its independent test
reference now computes the softmax in float64. The existing assertion and its
`atol=1e-10` are unchanged; production dtype and tolerances are unchanged.

### Downstream evidence and remaining qualification

The [Spyglass companion](spyglass_migration/README.md) supplies both archived
and current adapter patches, exact baseline hashes, an alternative archived
legacy cap, actual-source tests, and rollout guides. The current `cbc30e208`
companion passes **51 tests**, independently repeated; archived `8d3cb4408`
passes **27**. Qualification uses supported Python 3.11, DataJoint 0.14.9, and
PyNWB 3.1.3, with four synthetic NWB round trips and all four sorted/clusterless
prediction/EM paths. Actual `make` executes with read-only fixtures before
persistence, followed by real fitting on its prepared inputs.

The current companion also closes short between-camera interval skipping,
camera/decode mask mixing, one-ulp shifted-stop label loss, and recording-sized
interval broadcasting. Connected neural availability `[0.11, 0.3]` previously
became 0.1667 seconds of tapered camera weights; the original basis now clips
to the exact 0.19-second acquisition window, giving `1 / 0.19 = 5.2631579` Hz
for one event instead of 6 Hz. Caller masks and group weights stay separate.
Disconnected physical fitting windows that the native acquisition-range API
cannot represent raise an actionable error; select one supported range or fit
separate models. Decode masks still support multiple windows.

Interval-mask workspace is O(number of bins), about 0.884 MB for 18,000 bins
with either 10, 50, or 100 windows, versus 23.6 MB for the former 50-window
broadcast. These are companion mask measurements, separate from the native
20-row likelihood benchmark above. Logs are
`/private/tmp/spyglass-current-time-grid-l2wgxd03/py311-test-results.txt` and
`/private/tmp/spyglass-current-time-grid-l2wgxd03/independent-py311-test-results.txt`.

The live checkout independently advanced to `cbc30e208` with unrelated edits
and pins `non-local-detector==0.6.9`; it remains protected and untouched. The six
production/dependency hashes match the captured current baseline, while
`test_observed_time.py` advanced again and needs a test-hunk rebase before
applying the full companion. All patches apply-check on their captured isolated
baselines, and unrelated copied files remain byte-identical.
Database-backed pipeline tests, selected backend persistence, the full Spyglass
Python/conda matrix, and representative recomputed scientific outputs remain
release qualification steps. No production database write or release occurs
as part of this review fix.

### Final package acceptance

- Frozen complete suite: **2,656 passed / 6 skipped / 0 failed**, in 825.84
  seconds. `/private/tmp/nld-review-fix-final-full.log`. All four goldens and
  all eight approved snapshots pass. The six skips are unchanged from accepted
  6d. The earlier interrupted run was stopped for the documentation-only shared
  tracking-endpoint clarification, not a test failure.
- All **231 source/test/notebook files** match the frozen manifest at
  `/private/tmp/nld-review-fix-final-source-hashes.json` after completion.
- Actual source-consumer checks: **22 passed**. An empty artifact-root run
  passes all **five public alignment tests** and skips only the **17 source-file
  consumers**, so installed wheel/sdist/archive tests do not depend on checkout
  notebooks. `/private/tmp/nld-migrated-consumers-test-results.txt` and
  `/private/tmp/nld-source-only-consumer-test-results.txt`.
- Ruff and formatting pass on all **180 package Python files** and the changed
  profiling/example scripts. The paired exploratory notebook retains its same
  32 pre-existing lint diagnostics; no new diagnostic is introduced. All
  **158 code cells in the eight follow-up notebooks** compile, and consumer
  tests execute the relevant calls. Plain compilation is not a substitute for
  these runtime checks.
- Both independent final package reviews find no remaining concrete functional
  defect. The shared tracking sample contract is now explicit in the helper,
  encoding support, and guides without changing interpolation or event rules.

Refreshed GitHub CI, including Python 3.10–3.13 and wheel/sdist/archive checks,
is tracked in [PR #59 checks](https://github.com/LorenFrankLab/non_local_detector/pull/59/checks).
The independently reviewed downstream companion is complete as a reviewable
artifact, with the release qualification limits above.
