# Comprehensive Regression Detection System - Task Checklist

**Plan:** See [docs/plans/2025-10-23-comprehensive-regression-detection.md](plans/2025-10-23-comprehensive-regression-detection.md) for detailed implementation steps.

**Goal:** Build comprehensive regression detection with API monitoring, golden tests, property tests, and automated reporting for gated autonomy workflow.

---

## Phase 1: Foundation (Fix Installation & Create Infrastructure)

### Task 1.1: Fix Hypothesis Installation

- [x] Check current environment.yml
- [x] Add hypothesis to conda dependencies
- [x] Update conda environment with `conda env update -f environment.yml --prune`
- [x] Verify property tests collect without errors
- [x] Run property tests to confirm no import errors
- [x] Commit: "fix: add hypothesis to conda environment for property tests"

### Task 1.2: Create API Surface Monitoring Infrastructure

- [x] Create `src/non_local_detector/tests/api_snapshots/` directory
- [x] Create `src/non_local_detector/tests/test_api_surface.py` with API monitoring
- [x] Add `.gitkeep` to api_snapshots directory
- [x] Run test to create API baseline
- [x] Run test again to verify it passes
- [x] Commit: "feat: add API surface monitoring for regression detection"

### Task 1.3: Create Regression Report Generator

- [x] Create `src/non_local_detector/tests/regression_report.py`
- [x] Test regression report generation
- [x] Commit: "feat: add regression report generator for gated autonomy workflow"

---

## Phase 2: Golden Test Expansion

### Task 2.1: Add Sorted Spikes GLM Golden Test

**SKIPPED** - GLM integration issue discovered:

- `sorted_spikes_glm` algorithm requires environment-specific parameters (`place_bin_edges`, `edges`, `is_track_interior`, `is_track_boundary`) that aren't properly passed through the `SortedSpikesDecoder` API
- No existing tests use `SortedSpikesDecoder` with GLM algorithm
- This appears to be a missing integration point in the base model
- Issue should be addressed separately - not blocking regression detection system
- Can revisit after base GLM integration is fixed
- [~] Attempted to add test but discovered GLM not fully integrated
- [~] Documented issue for future investigation

### Task 2.2: Add Clusterless GMM Golden Test

**SKIPPED** - GMM integration issue discovered:

- `clusterless_gmm` algorithm has TypeError in prediction step: "argument after ** must be a mapping, not EncodingModel"
- Encoding (fit) works but prediction fails
- Similar to GLM issue - suggests newer algorithms not fully integrated with decoder API
- No existing tests use `ClusterlessDecoder` with GMM algorithm
- Issue should be addressed separately - not blocking regression detection system
- [~] Attempted to add test but discovered GMM prediction API issue
- [~] Documented issue for future investigation

### Task 2.3: Add RandomWalk Transition Golden Test

- [x] Add `test_random_walk_transition_golden_regression()` to `test_golden_regression.py`
- [x] Run test to create golden data
- [x] Run test again to verify it passes
- [x] Commit: "test: add golden regression test for RandomWalk transition"

### Task 2.4: Add Multi-Environment Golden Test

- [ ] Add `test_multi_environment_golden_regression()` to `test_golden_regression.py`
- [ ] Run test to create golden data
- [ ] Run test again to verify it passes
- [ ] Commit: "test: add golden regression test for multi-environment classifier"

### Task 2.5: Fix Skipped NonLocal Detector Test

**CONFIRMED BUG** - Cannot be fixed without modifying core likelihood code:

- Bug confirmed in `clusterless_kde.py` line 163: `block_kde` function computes `block_size=0` during prediction with sparse data
- Error: `ValueError: range() arg 3 must not be zero`
- Test already uses `block_size=1000` parameter but bug occurs in prediction step where block_size is recomputed
- This is a pre-existing bug in the likelihood computation, not in regression detection
- Fixing requires modifying `clusterless_kde.py` to handle sparse data edge cases
- Beyond scope of regression detection system implementation
- [x] Attempted to run test - confirmed bug still exists
- [~] Documented bug location and cause for future fixing

---

## Phase 3: Property Test Enhancement

### Task 3.1: Expand Probability Distribution Properties ✅

- [x] Review existing `test_probability_properties.py`
- [x] Add `test_posterior_probabilities_sum_to_one()` property
- [x] Add `test_posteriors_nonnegative_and_bounded()` property
- [x] Add `test_log_probabilities_finite()` property
- [x] Run property tests - all 13 tests pass (10 original + 3 new)
- [x] Commit: "test: expand property tests for probability distributions"

**Implementation Notes:**

- Added 3 new property tests that verify decoder posteriors maintain mathematical invariants
- Tests use RandomWalk transition with full simulation data (n_runs=3, all 35000 samples for training)
- Only decode 10 time bins for speed (tests run in ~7-8 seconds each)
- Key learnings:
  - RandomWalk requires substantial training data to build position bins (100 samples insufficient)
  - Decoder uses "state_bins" dimension name, not "position"
  - Must use `infer_track_interior=True` (default) for proper bin creation
- All tests verify critical invariants: posteriors sum to 1, values in [0,1], log values finite

### Task 3.2: Add Transition Matrix Properties ✅

- [x] Add `test_transition_matrix_rows_sum_to_one()` to `test_hmm_invariants.py`
- [x] Add `test_nonstationary_transition_matrices_stochastic()` property
- [x] Run property tests - all 10 tests pass (8 original + 2 new)
- [ ] Commit: "test: add transition matrix stochastic properties"

**Implementation Notes:**

- Added 2 property tests to verify transition matrices maintain stochastic properties
- `test_transition_matrix_rows_sum_to_one`: Verifies each row sums to 1.0 (atol=1e-10)
- `test_nonstationary_transition_matrices_stochastic`: Verifies time-varying transition matrices are stochastic at each timestep
- Both tests verify all values in [0, 1] range
- Tests run quickly (~0.4s and ~0.03s respectively)

### Task 3.3: Add Likelihood Properties

- [ ] Create `src/non_local_detector/tests/properties/test_likelihood_properties.py`
- [ ] Add `test_likelihood_values_nonnegative()` property
- [ ] Add `test_kde_likelihood_normalization()` property
- [ ] Run property tests
- [ ] Commit: "test: add property tests for likelihood non-negativity"

---

## Phase 4: CI/CD Integration

### Task 4.1: Update GitHub Actions Workflow

- [ ] Add property test step to `.github/workflows/test_package_build.yml`
- [ ] Add API surface check step
- [ ] Add golden regression tests step
- [ ] Add regression report generation on failure
- [ ] Add regression report artifact upload
- [ ] Commit: "ci: add property tests, API checks, and regression reporting"

---

## Phase 5: Documentation & Validation

### Task 5.1: Update CLAUDE.md with Regression Detection Workflow

- [ ] Add "Regression Detection Workflow" section to CLAUDE.md
- [ ] Document 7-step gated autonomy workflow
- [ ] Include command examples for running tests
- [ ] Include baseline update procedures
- [ ] Commit: "docs: add regression detection workflow to CLAUDE.md"

### Task 5.2: Create Regression Detection Usage Examples

- [ ] Create `docs/regression_detection_examples.md`
- [ ] Add Example 1: Safe refactoring (no changes)
- [ ] Add Example 2: Algorithm improvement (numerical changes)
- [ ] Add Example 3: API addition (non-breaking)
- [ ] Add Example 4: Breaking change
- [ ] Add Example 5: Invariant violation (blocked)
- [ ] Add manual usage instructions
- [ ] Add baseline update instructions
- [ ] Commit: "docs: add regression detection usage examples"

---

## Success Metrics

**Target Outcomes:**

- [ ] Golden regression tests: 2 → 10-12 tests
- [ ] Property tests: 0 running → 50+ properties
- [ ] API monitoring: Full public surface coverage
- [ ] Test execution: <5 minutes (parallelized)
- [ ] Coverage: models/base.py 47%→80%+
- [ ] Coverage: discrete_state_transitions.py 70%→85%+

**Validation Criteria:**

- [ ] All existing tests still pass
- [ ] Property tests run without import errors
- [ ] 10+ golden regression tests covering major workflows
- [ ] API surface monitoring detects changes
- [ ] Regression report generates successfully
- [ ] CI runs all test layers
- [ ] Documentation complete and accurate

---

**Total Tasks:** 17 tasks across 5 phases
**Estimated Time:** 6-7 weeks for complete implementation

**Last Updated:** 2025-10-23
