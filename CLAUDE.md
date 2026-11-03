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

Task-specific combinations:

- **Fix a bug**: if existing tests already cover it, fix directly and run them; if not, use `scientific-tdd` to add a failing test first.
- **Modify an algorithm**: `scientific-tdd` to implement with tests, then `numerical-validation` to verify invariants.
- **Optimize or debug JAX**: `jax` skill for the approach, then `numerical-validation` to confirm numerical equivalence.
- **Upgrade dependencies**: run the full test suite and numerical validation, and check for deprecation warnings and behavioral changes.
- **Add tests**: follow the patterns in `tests/`, use `conftest.py` fixtures, and tag with the `unit` / `integration` / `property` / `slow` markers.

### Environment Rules

- **Use `uv run` to execute all Python commands** (preferred)
- Alternatively, the conda environment `non_local_detector` can be used
- `uv run` automatically uses the project's `.venv` and `uv.lock` for reproducibility
  ```bash
  uv run pytest
  uv run python script.py
  ```

### Guided Autonomy Boundaries

**YOU CAN do automatically:**
- Read files, search code, explore codebase
- Run tests to check current behavior
- Make code changes
- Run tests to verify changes
- Run quality checks (ruff, mypy)
- Run numerical validation

**YOU MUST ASK PERMISSION before:**
- **Committing changes** (`git commit`)
- **Pushing to remote** (`git push`)
- **Modifying golden regression data files**
- **Changing numerical tolerances or convergence criteria**

### Golden Regression Data

There are no snapshot tests. Regression protection is `tests/test_golden_regression.py`
with pickled golden data in `tests/golden_data/`. If it fails, do not regenerate the
data; report what changed, by how much, why, and which invariants still hold
(see `.claude/skills/numerical-validation/SKILL.md`), and ask before modifying it.

---

## Project Overview

`non_local_detector` is a Python package for decoding non-local neural activity from electrophysiological data. It uses Bayesian inference with Hidden Markov Models (HMMs) and various likelihood models to detect spatial replay events and decode position from neural spike data.

Likelihood algorithms are registered as `LikelihoodAlgorithm` records (fit, predict,
default params, encoding-model TypedDict; clusterless records, `ClusterlessLikelihoodAlgorithm`,
also carry the mark reference scale and model-checking electrode adapter) in the
`_SORTED_SPIKES_ALGORITHMS` and `_CLUSTERLESS_ALGORITHMS` dictionaries — a new algorithm
must be added there to be reachable (see CONTRIBUTING.md, "Adding a likelihood algorithm").

## Numerical Accuracy Standards

### When Numerical Validation is Required

Run numerical validation (use `numerical-validation` skill) when modifying:
- `src/non_local_detector/core.py` (HMM algorithms)
- `src/non_local_detector/likelihoods/` (any likelihood model)
- `src/non_local_detector/continuous_state_transitions.py`
- `src/non_local_detector/discrete_state_transitions.py`
- `src/non_local_detector/initial_conditions.py`
- Any code with JAX transformations or numerical computations

### Tolerances, Invariants, and Validation Commands

Tolerance thresholds by change type, the five mathematical invariants that must always
hold (probability normalization, stochastic transition matrices, finite log-probabilities,
positive semi-definite covariances, non-negative likelihoods), and the property /
golden-regression commands to run are in
`.claude/skills/numerical-validation/SKILL.md` (Steps 4-7).

## JAX Code Requirements

### JAX-Specific Validation

After changing JAX code, verify:
- ✓ No unexpected recompilation (check compilation warnings)
- ✓ No NaN/Inf in outputs (`np.all(np.isfinite(result))`)
- ✓ Shapes match expectations
- ✓ Both CPU and GPU code paths work (if applicable)
- ✓ Performance is acceptable (profile if critical)

### Array annotations

Follow the convention in `CONTRIBUTING.md` ("Array annotations"): ArrayLike inputs, `jax.Array` / `NDArray` outputs, never `jnp.ndarray`.
