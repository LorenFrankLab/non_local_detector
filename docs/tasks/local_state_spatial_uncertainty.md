# TASKS: Local State with Spatial Uncertainty

Implementation checklist for `docs/plans/2026-04-18-local_state_with_spatial_uncertainty.md`.
Each milestone ends with a verification gate that must pass before proceeding.

---

## Milestone 1: Parameter Plumbing & State Index

### 1.1 Add `local_position_std` parameter
- [ ] Add `local_position_std: float | None = None` to `_SortedSpikesBaseDetector` and `_ClusterlessBaseDetector` constructors in `models/base.py`
- [ ] Add `local_position_std` to `NonLocalClusterlessDetector` and `NonLocalSortedSpikesDetector` in `models/non_local_model.py`, passing through to super
- [ ] Add input validation (must be positive or None), matching `_validate_penalty_params` style
- [ ] Write unit test: validation rejects `local_position_std <= 0`
- [ ] Write unit test: `local_position_std=None` accepted (default)

### 1.2 Modify `initialize_state_index` for multi-bin local
- [ ] Change guard in `initialize_state_index` (~line 793): `obs.is_no_spike or (obs.is_local and self.local_position_std is None)` for 1-bin path
- [ ] Verify local state falls through to full-bin `else` branch when `local_position_std` is set
- [ ] Write unit test: multi-bin local state gets `n_interior_bins` bins
- [ ] Write unit test: legacy `local_position_std=None` still gets 1 bin

### 1.3 Modify `initialize_initial_conditions` for multi-bin local
- [ ] Use `dataclasses.replace(obs, is_local=False)` when `obs.is_local and self.local_position_std is not None` (~line 816)
- [ ] Write unit test: multi-bin local gets uniform initial conditions over all bins
- [ ] Write unit test: legacy path unchanged

### Verification gate 1
- [ ] `uv run pytest` — all existing tests pass (no regressions)
- [ ] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 2: Continuous Transitions

### 2.1 Handle multi-bin to 1-bin transitions
- [ ] Add `elif n_row_bins > 1 and n_col_bins == 1` branch in `initialize_continuous_state_transitions` (~line 948) — each source bin transitions to single target with probability 1
- [ ] Write unit test: Local (multi-bin) -> No-Spike (1-bin) transition matrix has correct shape and values

### 2.2 Auto-upgrade `Discrete()` for multi-bin local
- [ ] Add early check in the `else` branch BEFORE `make_state_transition` call (~line 919): replace `Discrete` with `Uniform(environment_name=...)` when `local_position_std` is set and either from/to state is local
- [ ] Use the local state's `obs.environment_name` for the `Uniform` constructor
- [ ] Write unit test: `Discrete` auto-upgraded to `Uniform` for multi-bin local
- [ ] Write unit test: non-local `Discrete` transitions unaffected

### Verification gate 2
- [ ] `uv run pytest` — all existing tests pass
- [ ] Manually inspect transition matrix shapes for a multi-bin local model (quick script or test)
- [ ] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 3: Position Uncertainty Kernel

### 3.1 Implement `_compute_local_position_kernel`
- [ ] Add method to base class in `models/base.py`, mirroring `_compute_non_local_position_penalty`
- [ ] Euclidean distance path: compute squared distances between interpolated animal position and interior bin centers
- [ ] Track graph distance path: use precomputed `distance_between_nodes_` (handle dict-of-dicts for 1D and array for 2D)
- [ ] Normalize kernel per time step via `logsumexp`
- [ ] Return shape `(n_time, n_interior_bins)`

### 3.2 Unit tests for position kernel
- [ ] Kernel is valid log-probability (exp sums to 1 per time step, tolerance 1e-10)
- [ ] Kernel peak is at nearest bin to animal position
- [ ] Kernel narrows as `local_position_std` decreases
- [ ] Output shape is `(n_time, n_interior_bins)`
- [ ] Handles NaN positions gracefully

### Verification gate 3
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff check src/ && uv run ruff format --check src/`
- [ ] Numerical validation: kernel probabilities sum to 1.0 (atol=1e-10)

---

## Milestone 4: Likelihood Assembly

### 4.1 Modify `compute_log_likelihood` (clusterless path, ~line 2536)
- [ ] Update `needs_position` guard to include `self.local_position_std is not None`
- [ ] Compute `effective_is_local = obs.is_local and self.local_position_std is None` for caching and likelihood call
- [ ] Add position kernel post-assembly: loop over local states, compute kernel per environment (cached), add to `log_likelihood` via `.at[:, is_state_bin].add(...)`

### 4.2 Modify `compute_log_likelihood` (sorted spikes path, ~line 3409)
- [ ] Same three changes as 4.1 (mirror exactly)

### 4.3 Integration tests for likelihood assembly
- [ ] Multi-bin local model produces finite log-likelihoods (no NaN/Inf)
- [ ] Multi-bin local likelihood shape matches state bin count
- [ ] Legacy `local_position_std=None` produces identical likelihoods to baseline

### Verification gate 4
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run pytest -m property -v` — property-based tests pass (invariants hold)
- [ ] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 5: Results Assembly

### 5.1 Fix `_convert_results_to_xarray`
- [ ] Change guard (~line 2037): `obs.is_no_spike or (obs.is_local and self.local_position_std is None)` for NaN-position path
- [ ] Multi-bin local falls through to get real position coordinates from `environment.place_bin_centers_`

### 5.2 Fix `_convert_seq_to_df`
- [ ] Same guard change (~line 2175) for Viterbi/MAP sequence conversion

### 5.3 Integration tests for results
- [ ] Multi-bin local posterior has position coordinates (not NaN)
- [ ] Posterior probabilities sum to 1.0 across all states
- [ ] Legacy path produces identical xarray output

### Verification gate 5
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run pytest -m snapshot -v` — snapshot tests unchanged for legacy path
- [ ] `uv run ruff check src/ && uv run ruff format --check src/`

---

## Milestone 6: End-to-End Validation

### 6.1 Backward compatibility
- [ ] All existing snapshot tests produce identical results with `local_position_std=None`
- [ ] Golden regression tests pass: `uv run pytest src/non_local_detector/tests/test_golden_regression.py -v`

### 6.2 New feature integration tests
- [ ] Fit and predict with `NonLocalClusterlessDetector(local_position_std=5.0)` on synthetic data — valid posteriors
- [ ] Fit and predict with `NonLocalSortedSpikesDetector(local_position_std=5.0)` on synthetic data — valid posteriors
- [ ] Local state posterior is concentrated near animal's position (peak within 2 * std)
- [ ] Combined local + non-local posteriors sum to 1.0 across all states and bins

### 6.3 EM re-estimation validation
- [ ] EM-estimated place fields with multi-bin local are not distorted vs ground truth
- [ ] If distortion observed, document and add masking of local state in EM updates

### 6.4 Interaction tests
- [ ] `non_local_position_penalty > 0` and `local_position_std` enabled simultaneously — valid results
- [ ] No double-counting artifacts between penalty and kernel

### 6.5 Add new snapshots
- [ ] Add snapshot test for `local_position_std=5.0` model output for ongoing regression protection
- [ ] Request snapshot approval with full analysis per CLAUDE.md process

### Verification gate 6 (final)
- [ ] `uv run pytest` — full suite green
- [ ] `uv run pytest -m property -v` — all property tests pass
- [ ] `uv run pytest -m snapshot -v` — snapshots match
- [ ] `uv run pytest src/non_local_detector/tests/test_golden_regression.py -v` — golden regression passes
- [ ] `uv run ruff check src/ && uv run ruff format --check src/`
- [ ] Mathematical invariants verified: probabilities sum to 1, transitions stochastic, no NaN/Inf
