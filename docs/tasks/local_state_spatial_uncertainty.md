# TASKS: Local State with Spatial Uncertainty

Implementation checklist for `docs/plans/2026-04-18-local_state_with_spatial_uncertainty.md`.
Each milestone ends with a verification gate that must pass before proceeding.

---

## Milestone 1: Parameter Plumbing & State Index

### 1.1 Add `local_position_std` parameter
- [x] Add `local_position_std: float | None = None` to `_SortedSpikesBaseDetector` and `_ClusterlessBaseDetector` constructors in `models/base.py`
- [x] Add `local_position_std` to `NonLocalClusterlessDetector` and `NonLocalSortedSpikesDetector` in `models/non_local_model.py`, passing through to super
- [x] Add input validation (must be positive or None), matching `_validate_penalty_params` style
- [x] Write unit test: validation rejects `local_position_std <= 0`
- [x] Write unit test: `local_position_std=None` accepted (default)

### 1.2 Modify `initialize_state_index` for multi-bin local
- [x] Change guard in `initialize_state_index` (~line 793): `obs.is_no_spike or (obs.is_local and self.local_position_std is None)` for 1-bin path
- [x] Verify local state falls through to full-bin `else` branch when `local_position_std` is set
- [x] Write unit test: multi-bin local state gets total spatial bins (same as non-local for same environment), with interior mask tracking which are interior
- [x] Write unit test: legacy `local_position_std=None` still gets 1 bin

### 1.3 Modify `initialize_initial_conditions` for multi-bin local
- [x] Use `dataclasses.replace(obs, is_local=False)` when `obs.is_local and self.local_position_std is not None` (~line 816)
- [x] Write unit test: multi-bin local gets uniform initial conditions over all bins
- [x] Write unit test: legacy path unchanged

### Verification gate 1
- [x] `uv run pytest` — all existing tests pass (no regressions)
- [x] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 2: Continuous Transitions

### 2.1 Handle multi-bin to 1-bin transitions
- [x] Add `elif n_row_bins > 1 and n_col_bins == 1` branch in `initialize_continuous_state_transitions` (~line 948) — each source bin transitions to single target with probability 1
- [x] Write unit test: Local (multi-bin) -> No-Spike (1-bin) transition matrix has correct shape and values

### 2.2 Auto-upgrade `Discrete()` for multi-bin local
- [x] Add early check in the `else` branch BEFORE `make_state_transition` call (~line 919): replace `Discrete` with `Uniform(environment_name=...)` when `local_position_std` is set and either from/to state is local
- [x] Use the local state's `obs.environment_name` for the `Uniform` constructor
- [x] Scope: only auto-upgrade same-environment transitions. For cross-environment blocks, `Uniform` requires `environment2_name` (see `continuous_state_transitions.py` line 285). Skip auto-upgrade when `from_obs.environment_name != to_obs.environment_name` and raise or warn if `Discrete` is used there with multi-bin local
- [x] Write unit test: `Discrete` auto-upgraded to `Uniform` for same-environment multi-bin local
- [ ] Write unit test: cross-environment `Discrete` with multi-bin local raises clear error
- [x] Write unit test: non-local `Discrete` transitions unaffected

### Verification gate 2
- [x] `uv run pytest` — all existing tests pass (752 passed)
- [x] Manually inspect transition matrix shapes for a multi-bin local model (quick script or test)
- [x] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 3: Position Uncertainty Kernel

### 3.1 Implement `_compute_local_position_kernel`
- [x] Add method to base class in `models/base.py`, mirroring `_compute_non_local_position_penalty`
- [x] Add explicit NaN/missing-position guard before bin lookup: `get_position_at_time` can return NaN rows, and `environment.get_bin_ind()` has no NaN guard. Detect NaN positions, assign a uniform (flat) kernel for those time steps, and skip bin lookup for them
- [x] Euclidean distance path: compute squared distances between interpolated animal position and interior bin centers
- [ ] Track graph distance path: use precomputed `distance_between_nodes_` (handle dict-of-dicts for 1D and array for 2D)
- [x] Normalize kernel per time step via `logsumexp`
- [x] Return shape `(n_time, n_interior_bins)`

### 3.2 Unit tests for position kernel
- [x] Kernel is valid log-probability (exp sums to 1 per time step, tolerance 1e-6 for JAX float32)
- [x] Kernel peak is at nearest bin to animal position
- [x] Kernel narrows as `local_position_std` decreases
- [x] Output shape is `(n_time, n_interior_bins)`
- [x] NaN positions produce uniform (flat) kernel for those time steps (not NaN or crash)

### Verification gate 3
- [x] `uv run pytest` — all tests pass
- [x] `uv run ruff check src/ && uv run ruff format --check src/`
- [x] Numerical validation: kernel probabilities sum to 1.0 (atol=1e-6, JAX float32 precision)

---

## Milestone 4: Likelihood Assembly

### 4.1 Modify `compute_log_likelihood` (clusterless path, ~line 2536)
- [x] Update `needs_position` guard to include `self.local_position_std is not None`
- [x] Compute `effective_is_local = obs.is_local and self.local_position_std is None` for caching and likelihood call
- [x] Add position kernel post-assembly: loop over local states, compute kernel per environment (cached), add to `log_likelihood` via `.at[:, is_state_bin].add(...)`

### 4.2 Modify `compute_log_likelihood` (sorted spikes path, ~line 3409)
- [x] Same three changes as 4.1 (mirror exactly)

### 4.3 Integration tests for likelihood assembly
- [x] Multi-bin local model produces finite log-likelihoods (no NaN/Inf)
- [x] Multi-bin local likelihood shape matches state bin count
- [x] Legacy `local_position_std=None` produces identical likelihoods to baseline

### Verification gate 4
- [x] `uv run pytest` — all tests pass (752 passed)
- [x] `uv run pytest -m property -v` — property-based tests pass (57 passed)
- [x] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 5: Results Assembly

### 5.1 Fix `_convert_results_to_xarray`
- [x] Change guard (~line 2037): `obs.is_no_spike or (obs.is_local and self.local_position_std is None)` for NaN-position path
- [x] Multi-bin local falls through to get real position coordinates from `environment.place_bin_centers_`

### 5.2 Fix `_convert_seq_to_df`
- [x] Same guard change (~line 2175) for Viterbi/MAP sequence conversion

### 5.3 Integration tests for results
- [x] Multi-bin local posterior has position coordinates (not NaN)
- [x] Posterior probabilities sum to 1.0 across all states
- [x] Legacy path produces identical xarray output

### Verification gate 5
- [x] `uv run pytest` — all tests pass (752 passed)
- [x] `uv run pytest -m snapshot -v` — snapshot tests unchanged for legacy path (4 passed)
- [x] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 6: End-to-End Validation

### 6.1 Backward compatibility
- [x] All existing snapshot tests produce identical results with `local_position_std=None` (4 passed)
- [x] Golden regression tests pass: `uv run pytest src/non_local_detector/tests/test_golden_regression.py -v` (4 passed)

### 6.2 New feature integration tests
- [ ] Fit and predict with `NonLocalClusterlessDetector(local_position_std=5.0)` on synthetic data — valid posteriors
- [x] Fit and predict with `NonLocalSortedSpikesDetector(local_position_std=5.0)` on synthetic data — valid posteriors
- [ ] Local state posterior is concentrated near animal's position (peak within 2 * std)
- [x] Combined local + non-local posteriors sum to 1.0 across all states and bins

### 6.3 EM re-estimation validation

- [x] Verify EM M-step mechanism: the current EM update uses only discrete local-state probability per time point (`acausal_state_probabilities[:, local_state_index]`) as weights to `fit_encoding_model`, NOT the spatial posterior over bins. Confirm this is still the case with multi-bin local
- [x] If the M-step now consumes spatial posterior (code changed), test that place fields are not distorted vs ground truth and add masking if needed
- [x] If the M-step still uses only discrete state probability (expected), document that multi-bin local does not affect EM encoding updates and no distortion test is needed

### 6.4 Interaction tests
- [x] `non_local_position_penalty > 0` and `local_position_std` enabled simultaneously — valid results
- [x] No double-counting artifacts between penalty and kernel

### 6.5 Add new snapshots
- [ ] Add snapshot test for `local_position_std=5.0` model output for ongoing regression protection
- [ ] Request snapshot approval with full analysis per CLAUDE.md process

### Verification gate 6 (final)
- [x] `uv run pytest` — full suite green (752 passed)
- [x] `uv run pytest -m property -v` — all property tests pass (57 passed)
- [x] `uv run pytest -m snapshot -v` — snapshots match (4 passed)
- [x] `uv run pytest src/non_local_detector/tests/test_golden_regression.py -v` — golden regression passes (4 passed)
- [x] `uv run ruff check src/ && uv run ruff format --check src/`
- [x] Mathematical invariants verified: probabilities sum to 1, transitions stochastic, no NaN/Inf

### 6.6 Code review
- [x] Run code-reviewer agent on the full PR (all changed files) — address any issues before merging
