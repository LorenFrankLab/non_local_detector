# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚨 CRITICAL: Claude Code Operational Rules

### Mandatory Skills Usage

Before starting any task, check if a skill applies and use it:

- **New features/algorithms** → Use `scientific-tdd` skill
- **Mathematical/algorithmic changes** → Use `numerical-validation` skill
- **Code restructuring** → Use `safe-refactoring` skill
- **JAX code (transformations, performance, debugging)** → Use `jax` skill

**Announce skill usage:** "I'm using the [skill-name] skill to [purpose]."

### Environment Rules (ENFORCED BY HOOKS)

- **Conda environment `non_local_detector` must be activated before Python commands**
- Hooks will warn if wrong environment detected
- Use full conda paths or activate environment first:
  ```bash
  conda activate non_local_detector
  ```

### Guided Autonomy Boundaries

**YOU CAN do automatically:**
- Read files, search code, explore codebase
- Run tests to check current behavior
- Make code changes
- Run tests to verify changes
- Run quality checks (ruff, black, mypy)
- Run numerical validation

**YOU MUST ASK PERMISSION before:**
- **Updating snapshots** (`--snapshot-update`) - REQUIRES FULL ANALYSIS FIRST
- **Committing changes** (`git commit`)
- **Pushing to remote** (`git push`)
- **Modifying golden regression data files**
- **Changing numerical tolerances or convergence criteria**

### Snapshot Update Approval Process

When snapshot tests show changes, YOU MUST provide this analysis before requesting approval:

1. **Diff**: Exact changes in snapshot/output (show the actual differences)
2. **Explanation**: Why the change occurred (code change → output change causality)
3. **Validation**: Proof mathematical properties still hold (invariants verified)
4. **Test case**: Before/after comparison demonstrating correctness

**Format:**
```
Snapshot Analysis:

1. DIFF:
   - test_model_output: posterior[10] changed from [0.342156, 0.657844] to [0.342157, 0.657843]
   - Max difference: 1e-6

2. EXPLANATION:
   Changed optimizer tolerance from 1e-6 to 1e-8, resulting in more precise convergence.

3. VALIDATION:
   ✓ Probabilities sum to 1.0 (deviation < 1e-14)
   ✓ Transition matrices stochastic
   ✓ No NaN/Inf values
   ✓ Property tests: 42/42 passed
   ✓ Golden regression: All invariants hold

4. TEST CASE:
   Old: State 2 probability = 0.6578 (strong preference)
   New: State 2 probability = 0.6578 (strong preference)
   Scientific conclusion: Unchanged

Approve snapshot update?
```

Only after user approval can snapshots be updated.

---

## Project Overview

`non_local_detector` is a Python package for decoding non-local neural activity from electrophysiological data. It uses Bayesian inference with Hidden Markov Models (HMMs) and various likelihood models to detect spatial replay events and decode position from neural spike data.

## Development Commands

### Environment Setup

The project uses a conda environment called `non_local_detector`. When running development commands, use the full path to the environment's executables:

- **Python**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/python`
- **Pytest**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest`
- **Ruff**: Use ruff from another conda environment (e.g., `/Users/edeno/miniconda3/envs/spectral_connectivity/bin/ruff`)
- **Black**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/black`

Alternatively, activate the environment first:
```bash
conda activate non_local_detector
```

### Installation

- **Development setup**: `mamba create env -f environment.yml` (CPU) or `mamba create env -f environment_gpu.yml` (GPU)
- **From pip**: `pip install non_local_detector` (CPU) or `pip install non_local_detector[gpu]` (GPU with CUDA)

### Testing

- **Run tests**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest` (uses pytest framework defined in pyproject.toml)
- **Run single test**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest src/non_local_detector/tests/test_version_import.py`
- **Test with coverage**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest --cov`
- **Run snapshot tests**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest -m snapshot` (regression tests using syrupy)
- **Run golden regression tests**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest src/non_local_detector/tests/test_golden_regression.py -v`
- **Update snapshots**: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest --snapshot-update` (after intentional behavior changes)

### Code Quality

- **Lint and fix**: `ruff check --fix src/`
- **Format code**: `ruff format src/` or `black src/`
- **Type checking**: `mypy src/non_local_detector/`
- **Check all quality tools**: `ruff check src/ && ruff format --check src/ && black --check src/ && mypy src/non_local_detector/`

#### Quality Gates

The CI pipeline enforces code quality standards on all pull requests:

1. **Ruff Linting**: All code must pass `ruff check src/` with zero issues
2. **Code Formatting**: Must pass both `ruff format --check src/` and `black --check src/`
3. **Type Checking**: `mypy` runs but currently allows errors (gradual adoption)
4. **Tests**: All existing tests must pass

**For Contributors**: Run the full quality check locally before pushing:
```bash
ruff check src/ && ruff format --check src/ && black --check src/ && pytest
```

### Building

- **Build package**: Uses hatchling build system (defined in pyproject.toml)
- **Version management**: Automatic versioning via hatch-vcs from git tags

## Numerical Accuracy Standards

### When Numerical Validation is Required

Run numerical validation (use `numerical-validation` skill) when modifying:
- `src/non_local_detector/core.py` (HMM algorithms)
- `src/non_local_detector/likelihoods/` (any likelihood model)
- `src/non_local_detector/continuous_state_transitions.py`
- `src/non_local_detector/discrete_state_transitions.py`
- `src/non_local_detector/initial_conditions.py`
- Any code with JAX transformations or numerical computations

### Tolerance Specifications

| Change Type | Max Acceptable Difference | Approval Required |
|-------------|---------------------------|-------------------|
| Pure refactoring | 1e-14 (floating-point noise) | No (informational) |
| Code optimization | 1e-10 | Yes |
| Algorithm modification | 1e-10 | Yes (with justification) |
| Differences > 1e-10 | Any magnitude | Yes (strong justification) |

### Mathematical Invariants (MUST ALWAYS HOLD)

These properties must be verified after any change to mathematical code:

1. **Probability distributions sum to 1.0**
   - Tolerance: 1e-10
   - Check: `np.allclose(probs.sum(axis=-1), 1.0, atol=1e-10)`

2. **Transition matrices are stochastic**
   - Rows sum to 1.0 (tolerance: 1e-10)
   - All values in [0, 1]
   - Check: `np.allclose(T.sum(axis=-1), 1.0, atol=1e-10) and np.all((T >= 0) & (T <= 1))`

3. **Log-probabilities are finite**
   - No NaN or Inf values
   - Check: `np.all(np.isfinite(log_probs))`

4. **Covariance matrices are positive semi-definite**
   - All eigenvalues >= 0 (tolerance: -1e-10 for numerical noise)
   - Check: `np.all(np.linalg.eigvalsh(cov) >= -1e-10)`

5. **Likelihoods are non-negative**
   - Check: `np.all(likelihood >= 0)`

### Validation Commands

After mathematical changes, run:

```bash
# Property-based tests (verify invariants with many random inputs)
/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest -m property -v

# Golden regression (validate against real scientific data)
/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest src/non_local_detector/tests/test_golden_regression.py -v

# Snapshot tests (detect any output changes)
/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest -m snapshot -v
```

## Workflow Selection Guide

Use this decision tree to select the appropriate workflow for your task:

### Task: "Add new feature" or "Implement new algorithm"
→ **Use `scientific-tdd` skill**
- Write test first (RED)
- Implement to pass test (GREEN)
- Refactor if needed
- Run numerical validation if mathematical code

### Task: "Fix bug"
→ **Check: Do existing tests cover this bug?**
- **Yes**: Fix directly, run tests to verify
- **No**: Use `scientific-tdd` skill to add test first

### Task: "Refactor code" or "Improve code structure"
→ **Use `safe-refactoring` skill**
- Verify zero behavioral changes
- All tests must match baseline exactly
- No snapshot differences allowed

### Task: "Modify algorithm" or "Change mathematical code"
→ **Use `scientific-tdd` + `numerical-validation` skills**
1. Use `scientific-tdd` to implement change with tests
2. Use `numerical-validation` to verify correctness and invariants

### Task: "Work with JAX code"
→ **Use `jax` skill for:**
- JAX transformations (jit, grad, vmap, scan)
- Performance optimization
- Debugging NaN/Inf issues
- Memory profiling
- Refactoring NumPy to JAX
- Analyzing JAXprs
- Understanding recompilation

### Task: "Optimize JAX performance" or "Debug JAX issues"
→ **Use `jax` + `numerical-validation` skills**
1. Use `jax` skill for optimization approach
2. Use `numerical-validation` to verify numerical equivalence

### Task: "Update dependencies" or "Upgrade packages"
→ **Comprehensive testing required:**
1. Run full test suite
2. Run numerical validation
3. Check for deprecation warnings
4. Verify no behavioral changes

### Task: "Add tests" or "Improve test coverage"
→ **Follow existing test patterns:**
- See `src/non_local_detector/tests/` for examples
- Use fixtures from `conftest.py`
- Add test markers if appropriate (unit, integration, property, snapshot)

## JAX Code Requirements

This codebase uses JAX as the primary computational backend for GPU acceleration and automatic differentiation.

### When to Use the JAX Skill

Use the `jax` skill when working with:
- **Core algorithms**: `src/non_local_detector/core.py` (HMM with jit/vmap/scan)
- **Likelihood models**: JAX transformations for likelihood calculations
- **Performance issues**: Optimization, memory profiling, compilation
- **Debugging**: NaN/Inf issues, shape mismatches, gradient problems
- **Refactoring**: Converting NumPy to JAX or vice versa

### JAX-Specific Validation

After changing JAX code, verify:
- ✓ No unexpected recompilation (check compilation warnings)
- ✓ No NaN/Inf in outputs (`np.all(np.isfinite(result))`)
- ✓ Shapes match expectations
- ✓ Both CPU and GPU code paths work (if applicable)
- ✓ Performance is acceptable (profile if critical)

### JAX Best Practices

- Prefer pure functions (no side effects)
- Use `jax.lax.scan` instead of Python loops
- Avoid in-place operations (JAX arrays are immutable)
- Use `jax.vmap` for vectorization
- Be mindful of memory with large arrays

## Architecture Overview

### Core Components

1. **Likelihood Models** (`src/non_local_detector/likelihoods/`):
   - `sorted_spikes_glm`: GLM-based likelihood for sorted spike data
   - `sorted_spikes_kde`: KDE-based likelihood for sorted spike data
   - `clusterless_kde`: KDE-based likelihood for clusterless (continuous) spike data
   - `clusterless_gmm`: GMM-based likelihood for clusterless spike data
   - Algorithms are registered in `_SORTED_SPIKES_ALGORITHMS` and `_CLUSTERLESS_ALGORITHMS` dictionaries

2. **State Transition Models**:
   - **Continuous**: `RandomWalk`, `EmpiricalMovement`, `Identity`, `Uniform` transitions
   - **Discrete**: Stationary/non-stationary diagonal and custom transition matrices

3. **Decoder Models** (`src/non_local_detector/models/`):
   - `NonLocalClusterlessDetector` / `NonLocalSortedSpikesDetector`: Main replay detection models
   - `ContFragClusterlessClassifier` / `ContFragSortedSpikesClassifier`: Fragmented decoding
   - `ClusterlessDecoder` / `SortedSpikesDecoder`: Basic position decoding
   - `MultiEnvironmentClusterlessClassifier`: Multi-environment decoding

4. **Core Algorithms** (`src/non_local_detector/core.py`):
   - Forward-backward algorithm implementation adapted from dynamax
   - Chunked filtering for large datasets
   - Viterbi algorithm for most likely sequences
   - JAX-based implementations for GPU acceleration

### Key Dependencies

- **JAX**: Primary computational backend for GPU acceleration
- **NumPy/SciPy**: Numerical computing
- **pandas/xarray**: Data handling
- **scikit-learn**: Machine learning utilities
- **track_linearization**: Spatial trajectory processing
- **matplotlib/seaborn**: Visualization

### Data Flow

1. **Encoding Models**: Fit likelihood models to training data (position + spikes)
2. **State Transitions**: Define movement models and discrete state transitions
3. **Decoding**: Apply HMM filtering/smoothing to decode position from test data
4. **Analysis**: Extract replay events, compute metrics, visualize results

### Testing Structure

- Test suite in `src/non_local_detector/tests/`
  - Unit tests for individual components
  - Integration tests for full workflows
  - Property-based tests using hypothesis
  - **Snapshot tests** (`tests/snapshots/`) for regression detection
- Extensive notebook-based testing and validation in `notebooks/`
- Test notebooks cover different likelihood models, simulation, and real data analysis
- Snapshot tests use `syrupy` to capture expected model outputs and detect regressions

### Key Files to Understand

- `src/non_local_detector/core.py`: Core HMM algorithms
- `src/non_local_detector/models/base.py`: Base classes for all models
- `src/non_local_detector/likelihoods/__init__.py`: Likelihood algorithm registry
- `src/non_local_detector/__init__.py`: Public API exports

## Development Notes

- The codebase uses JAX for numerical computations, enabling GPU acceleration
- Models follow scikit-learn estimator patterns (fit/predict interface)
- Extensive use of xarray for labeled multidimensional data
- The package handles both sorted spike data (discrete units) and clusterless data (continuous features)
- Environment configuration supports both CPU and GPU installations via different dependency sets
