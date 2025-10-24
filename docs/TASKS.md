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

- [ ] Add `test_sorted_spikes_glm_golden_regression()` to `test_golden_regression.py`
- [ ] Run test to create golden data
- [ ] Verify golden data file created
- [ ] Run test again to verify it passes
- [ ] Commit: "test: add golden regression test for sorted_spikes_glm"

### Task 2.2: Add Clusterless GMM Golden Test

- [ ] Add `test_clusterless_gmm_golden_regression()` to `test_golden_regression.py`
- [ ] Run test to create golden data
- [ ] Run test again to verify it passes
- [ ] Commit: "test: add golden regression test for clusterless_gmm"

### Task 2.3: Add RandomWalk Transition Golden Test

- [ ] Add `test_random_walk_transition_golden_regression()` to `test_golden_regression.py`
- [ ] Run test to create golden data
- [ ] Run test again to verify it passes
- [ ] Commit: "test: add golden regression test for RandomWalk transition"

### Task 2.4: Add Multi-Environment Golden Test

- [ ] Add `test_multi_environment_golden_regression()` to `test_golden_regression.py`
- [ ] Run test to create golden data
- [ ] Run test again to verify it passes
- [ ] Commit: "test: add golden regression test for multi-environment classifier"

### Task 2.5: Fix Skipped NonLocal Detector Test

- [ ] Try running the skipped test to identify bug
- [ ] Investigate clusterless_kde.py bug (block_size zero division)
- [ ] Fix bug or adjust test parameters
- [ ] Remove skip marker from test
- [ ] Verify test passes
- [ ] Commit: "fix: enable nonlocal_detector golden regression test"

---

## Phase 3: Property Test Enhancement

### Task 3.1: Expand Probability Distribution Properties

- [ ] Review existing `test_probability_properties.py`
- [ ] Add `test_posterior_probabilities_sum_to_one()` property
- [ ] Add `test_posteriors_nonnegative_and_bounded()` property
- [ ] Add `test_log_probabilities_finite()` property
- [ ] Run property tests with hypothesis statistics
- [ ] Commit: "test: expand property tests for probability distributions"

### Task 3.2: Add Transition Matrix Properties

- [ ] Add `test_transition_matrix_rows_sum_to_one()` to `test_hmm_invariants.py`
- [ ] Add `test_nonstationary_transition_matrices_stochastic()` property
- [ ] Run property tests
- [ ] Commit: "test: add transition matrix stochastic properties"

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
