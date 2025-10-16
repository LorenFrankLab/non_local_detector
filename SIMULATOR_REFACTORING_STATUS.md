# Simulator Refactoring Status

## Overview

This document tracks the progress of refactoring the `non_local_detector` simulator to emit decoder-ready outputs instead of NaN-padded tensors. The refactoring improves code clarity, eliminates error-prone data transformations, and establishes clear contracts between simulator and decoder components.

**Goal:** Convert simulator from returning NaN-padded `(T, F, E)` tensors to returning per-electrode lists of spike data via a well-defined `ClusterlessSimOutput` dataclass.

## Progress Summary

**Completed:** 6/10 PRs (PRs 0-5)
**Remaining:** 4/10 PRs (PRs 6-9)
**Status:** Currently ready to continue with PR 6

---

## Completed Pull Requests

### ✅ PR 0: Define Simulator Contract

**Commit:** `c9f2211` (Revert back to original)
**Files Created:**
- `src/non_local_detector/tests/_sim_contract.py`

**Summary:**
Created `ClusterlessSimOutput` dataclass that defines the standard output format for the clusterless simulator. This contract establishes clear expectations for all downstream decoders.

**Key Decisions:**
- Spike times as `list[np.ndarray]` (one array per electrode) instead of NaN-padded tensors
- All times in seconds (not samples)
- Position always 2D `(n_time, n_pos_dims)` even for 1D tracks
- Environment object included and pre-fitted with position data
- Optional `bin_widths` field for convenience

**Contract Definition:**
```python
@dataclass
class ClusterlessSimOutput:
    position_time: np.ndarray              # (n_time_position,), seconds
    position: np.ndarray                   # (n_time_position, n_pos_dims)
    edges: np.ndarray                      # (n_time_bins+1,), seconds
    spike_times: list[np.ndarray]          # per electrode, seconds, strictly increasing
    spike_waveform_features: list[np.ndarray]  # per electrode, (n_spikes_e, n_features)
    environment: Environment
    bin_widths: np.ndarray | None = None
```

---

### ✅ PR 1: Rewrite Clusterless Simulator

**Commit:** `c9f2211` (Revert back to original)
**Files Modified:**
- `src/non_local_detector/simulate/clusterless_simulation.py`

**Summary:**
Rewrote `make_simulated_run_data()` to return `ClusterlessSimOutput` instead of 5-tuple. Added seeding support and converted internal NaN-padded format to per-electrode lists.

**Key Changes:**
```python
# Old API (returned 5-tuple):
(time, position, sampling_frequency, multiunit_waveforms, multiunit_spikes) = make_simulated_run_data(...)

# New API (returns dataclass):
sim = make_simulated_run_data(seed=42)
# Access via: sim.position_time, sim.spike_times, sim.spike_waveform_features, etc.
```

**Implementation Details:**
- Conversion from NaN-padded to lists happens inside simulator:
  ```python
  spike_times = []
  spike_waveform_features = []
  for electrode_id in range(n_electrodes):
      spike_indicator = multiunits_spikes[:, electrode_id]
      electrode_spike_times = position_time[spike_indicator]
      electrode_features = multiunits[spike_indicator, :, electrode_id]
      spike_times.append(electrode_spike_times)
      spike_waveform_features.append(electrode_features)
  ```
- Environment is fitted with position data before returning
- Added `seed` parameter for reproducibility (default: 0)

---

### ✅ PR 2: Add Seeding Throughout Simulator Stack

**Commit:** `f8dad91` (Add snapshot testing instructions)
**Files Modified:**
- `src/non_local_detector/simulate/simulate.py`
- `src/non_local_detector/simulate/sorted_spikes_simulation.py`

**Summary:**
Added `seed` parameters to all random sampling functions throughout the simulator stack to enable deterministic testing.

**Functions Modified:**
- `simulate.py`:
  - `simulate_poisson_spikes(rate, sampling_frequency, seed=None)`
  - `simulate_multiunit_with_place_fields(..., seed=None)`

- `sorted_spikes_simulation.py`:
  - `simulate_poisson_spikes(rate, sampling_frequency, seed=None)`
  - `simulate_two_state_inhomogenous_poisson(..., seed=None)`
  - `make_fragmented_replay(..., seed=None)`
  - `make_simulated_data(..., seed=None)`

**Pattern Used:**
```python
def simulate_poisson_spikes(
    rate: np.ndarray, sampling_frequency: int, seed: int | None = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)
```

---

### ✅ PR 3: Migrate Existing Tests to New API

**Commit:** `a936131` (Add snapshot tests)
**Files Modified:**
- `src/non_local_detector/tests/snapshots/test_clusterless_model_snapshots.py`

**Summary:**
Updated test fixture to use new `ClusterlessSimOutput` API, eliminating 15 lines of manual conversion logic.

**Before (old API):**
```python
(time, position, sampling_frequency, multiunit_waveforms, multiunit_spikes) = make_simulated_run_data(...)

# 15 lines of manual conversion from NaN-padded to lists
spike_times = []
spike_waveform_features = []
for electrode_id in range(n_electrodes):
    spike_indicator = multiunit_spikes[:, electrode_id]
    electrode_spike_times = time[spike_indicator]
    electrode_features = multiunit_waveforms[spike_indicator, :, electrode_id]
    spike_times.append(electrode_spike_times)
    spike_waveform_features.append(electrode_features)

position = position[:, np.newaxis]  # Manual reshape
```

**After (new API):**
```python
sim = make_simulated_run_data(seed=42)

return {
    "time": sim.position_time,
    "position": sim.position,  # Already 2D
    "spike_times": sim.spike_times,  # Already per-electrode lists
    "spike_waveform_features": sim.spike_waveform_features,
}
```

**Benefits:**
- Eliminated error-prone manual conversion
- Reduced code duplication across test suite
- Tests are now more readable and maintainable

---

### ✅ PR 4: Add Comprehensive Contract Tests

**Commit:** `931d914` (Add tests for discrete transition estimation)
**Files Created:**
- `src/non_local_detector/tests/test_simulator_contract.py`

**Summary:**
Created 14 comprehensive tests that enforce all simulator contract invariants. These tests catch violations early and serve as living documentation of expected behavior.

**Test Coverage:**

1. **`test_shapes_and_lengths_per_electrode`** - Verifies:
   - Edges are 1D with ≥2 elements
   - Position time is 1D and matches position length
   - Position is 2D `(n_time, n_pos_dims)`
   - Spike times and features have matching electrode counts
   - Features are `(n_spikes, 4)` per electrode
   - All arrays have correct dtypes (floating)

2. **`test_times_sorted_and_in_bounds`** - Verifies:
   - Position time is strictly increasing
   - Edges are strictly increasing
   - Spike times are within `[edges[0], edges[-1]]`
   - Spike times are strictly increasing per electrode
   - All times are finite

3. **`test_no_nans_finite_features`** - Verifies:
   - No NaN values in spike waveform features
   - No NaN values in position
   - No NaN values in position_time
   - All values are finite

4. **`test_optional_empty_electrode`** - Verifies:
   - Empty electrodes have shape `(0,)` for times
   - Empty electrodes have shape `(0, 4)` for features
   - Empty electrodes are valid outputs

5. **`test_bin_widths_consistency`** - Verifies:
   - `bin_widths` matches `np.diff(edges)` if provided

6. **`test_environment_fitted`** - Verifies:
   - Environment object is provided
   - Environment has `place_bin_centers_` attribute (fitted)

7. **`test_deterministic_seeding`** - Verifies:
   - Same seed produces identical spike counts
   - Same seed produces identical spike times
   - Same seed produces identical spike features

8. **`test_different_seeds_produce_different_outputs`** - Verifies:
   - Different seeds produce different outputs

9. **`test_spike_times_units_are_seconds`** - Verifies:
   - Spike times are in seconds (not samples)
   - Position time is in seconds
   - Times are in reasonable ranges

10. **`test_variable_electrode_counts`** (parametrized: 1, 2, 4, 8) - Verifies:
    - Simulator works with different numbers of electrodes

11. **`test_position_is_2d`** - Verifies:
    - Position is always 2D even for 1D tracks
    - Shape is `(n_time, 1)` for 1D tracks

**Testing Pattern:**
```python
def test_shapes_and_lengths_per_electrode() -> None:
    """Verify output shapes and per-electrode array lengths match."""
    n_tetrodes = 3
    place_field_means = np.arange(0, 120, 10)  # 12 neurons, divisible by 3
    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=1,
        seed=0,
    )

    assert sim.edges.ndim == 1
    assert len(sim.spike_times) == n_tetrodes
    # ... more assertions
```

**Performance:** All 14 tests pass in <1 second combined.

---

### ✅ PR 5: Add Oracle Correctness Tests

**Commit:** `8a7abab` (Fix type hints and code duplication)
**Files Created:**
- `src/non_local_detector/tests/test_oracle_correctness.py`

**Summary:**
Created 3 oracle tests that verify decoder correctness by simulating data with known ground truth and testing that decoders can accurately recover true positions.

**Test Coverage:**

1. **`test_nonlocal_kde_top1_accuracy_high`** - Verifies:
   - KDE decoder achieves ≥80% top-1 accuracy on simulated data
   - Tests fundamental encoding/decoding pipeline correctness

2. **`test_nonlocal_kde_top3_accuracy_very_high`** - Verifies:
   - True position is in top-3 decoded positions ≥90% of the time
   - Tests that decoder is consistently close to truth

3. **`test_delta_t_scaling_normalized`** - Verifies:
   - Bin width changes don't drastically affect decoding accuracy
   - Standard bins (50) achieve ≥60% accuracy
   - Wide bins (25) achieve ≥50% accuracy
   - Accuracy difference is ≤30 percentage points
   - Tests that Δt scaling in likelihood is correct

**Key Implementation Details:**

**Constants:**
```python
POSITION_STD = 6.0
WAVEFORM_STD = 24.0
POSITION_TOLERANCE = 10.0
TOP1_ACCURACY_THRESHOLD = 0.80
TOP3_ACCURACY_THRESHOLD = 0.90
DELTA_T_STANDARD_ACCURACY_THRESHOLD = 0.60
DELTA_T_WIDE_ACCURACY_THRESHOLD = 0.50
DELTA_T_MAX_DIFFERENCE = 0.30
```

**Helper Functions:**
```python
def get_decoded_position_bins(
    log_likelihood: npt.NDArray[np.floating],
    place_bin_centers: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Get the most likely position bin for each time point."""
    max_likelihood_bins = np.argmax(log_likelihood, axis=1)
    return place_bin_centers[max_likelihood_bins]

def compute_top_k_accuracy(
    true_positions: npt.NDArray[np.floating],
    log_likelihood: npt.NDArray[np.floating],
    place_bin_centers: npt.NDArray[np.floating],
    k: int = 1,
    position_tolerance: float = POSITION_TOLERANCE,
) -> float:
    """Compute top-k accuracy: fraction of times true position is in top k bins."""
    # ... implementation
```

**Shared Fixture (eliminates code duplication):**
```python
@pytest.fixture
def kde_decoding_setup() -> dict[str, Any]:
    """Generate simulated data and encoding model for KDE decoding tests.

    Returns dict with keys:
        - sim: ClusterlessSimOutput with full simulation data
        - encoding_model: Fitted KDE encoding model
        - test_edges: Time bin edges for decoding
        - log_likelihood: Log-likelihood array (n_time_bins, n_position_bins)
        - true_positions: True positions at test bin centers
        - place_bin_centers: Position bin centers from environment
    """
    # Generate simulated data (4 tetrodes, 16 neurons, 3 runs)
    sim = make_simulated_run_data(
        n_tetrodes=4,
        place_field_means=np.arange(0, 160, 10),
        sampling_frequency=500,
        n_runs=3,
        seed=42,
    )

    # 70/30 train/test split
    n_encode = int(0.7 * len(sim.position_time))

    # Fit encoding model on training data
    encoding_model = fit_clusterless_kde_encoding_model(...)

    # Decode on test data
    log_likelihood = predict_clusterless_kde_log_likelihood(...)

    return {...}
```

**Code Quality:**
- Complete type hints using `numpy.typing`
- Magic numbers extracted to module-level constants
- Shared fixture eliminates ~160 lines of duplication
- All tests pass in ~8.5 seconds
- All quality gates passing (ruff, black)

**Why These Tests Matter:**
- Oracle tests prove the fundamental correctness of encoding/decoding pipeline
- If these tests fail, it indicates a serious regression in decoder accuracy
- Complements snapshot tests (which detect any change) by testing semantic correctness

---

## Remaining Pull Requests

### ⬜ PR 6: Test KDE/GMM Agreement on Same Data

**Status:** Not started
**Estimated Effort:** Low
**Files to Create:**
- `src/non_local_detector/tests/test_clusterless_likelihood_agreement.py`

**Goal:**
Verify that `clusterless_kde` and `clusterless_gmm` likelihood models produce qualitatively similar results on the same simulated data.

**Test Strategy:**
```python
@pytest.fixture
def clusterless_sim_data():
    """Generate simulated data for likelihood agreement tests."""
    return make_simulated_run_data(
        n_tetrodes=3,
        place_field_means=np.arange(0, 120, 10),
        sampling_frequency=500,
        n_runs=2,
        seed=100,
    )

def test_kde_gmm_top_position_agreement(clusterless_sim_data):
    """Test that KDE and GMM agree on most likely position most of the time."""
    # Fit both models on same training data
    kde_model = fit_clusterless_kde_encoding_model(...)
    gmm_model = fit_clusterless_gmm_encoding_model(...)

    # Predict on same test data
    kde_ll = predict_clusterless_kde_log_likelihood(...)
    gmm_ll = predict_clusterless_gmm_log_likelihood(...)

    # Compare most likely positions
    kde_positions = get_decoded_position_bins(kde_ll, place_bin_centers)
    gmm_positions = get_decoded_position_bins(gmm_ll, place_bin_centers)

    # Should agree on most likely position >70% of the time
    agreement = np.mean(np.all(kde_positions == gmm_positions, axis=1))
    assert agreement >= 0.70, f"KDE/GMM agreement {agreement:.2%} too low"

def test_kde_gmm_correlation(clusterless_sim_data):
    """Test that KDE and GMM likelihoods are correlated."""
    # ... compute correlation of likelihood values
    # Should be >0.5 correlation
```

**Expected Outcome:**
- 2-3 tests comparing KDE and GMM outputs
- Agreement on top positions (>70%)
- Positive correlation of likelihood values (>0.5)
- Tests pass in <10 seconds

---

### ⬜ PR 7: Test Posterior Probability Properties

**Status:** Not started
**Estimated Effort:** Medium
**Files to Create:**
- `src/non_local_detector/tests/test_posterior_properties.py`

**Goal:**
Test mathematical properties of posterior distributions that must hold regardless of implementation details (normalization, causality, smoothness).

**Test Strategy:**
```python
@pytest.fixture
def decoder_with_results():
    """Fit decoder and get posterior results."""
    sim = make_simulated_run_data(seed=200)
    decoder = ClusterlessDecoder(...).fit(...)
    results = decoder.predict(...)
    return decoder, results, sim

def test_posterior_normalized(decoder_with_results):
    """Test that posterior probabilities sum to 1 at each time point."""
    _, results, _ = decoder_with_results
    posterior = results.acausal_posterior.values

    # Sum over position dimensions
    position_sum = np.sum(posterior, axis=1)

    # Should be very close to 1.0 at each time point
    np.testing.assert_allclose(position_sum, 1.0, rtol=1e-5,
                               err_msg="Posterior not normalized")

def test_acausal_smoother_than_causal(decoder_with_results):
    """Test that acausal posterior is smoother than causal."""
    _, results, _ = decoder_with_results

    # Compute variance of position estimate over time
    causal_var = np.var(results.causal_posterior.values, axis=0).mean()
    acausal_var = np.var(results.acausal_posterior.values, axis=0).mean()

    # Acausal should be smoother (lower variance)
    assert acausal_var <= causal_var, "Acausal not smoother than causal"

def test_posterior_entropy_reasonable(decoder_with_results):
    """Test that posterior entropy is in reasonable range."""
    _, results, _ = decoder_with_results
    posterior = results.acausal_posterior.values

    # Compute entropy at each time point
    entropy = -np.sum(posterior * np.log(posterior + 1e-10), axis=1)

    # Should be between 0 (certain) and log(n_bins) (uniform)
    n_bins = posterior.shape[1]
    max_entropy = np.log(n_bins)
    assert np.all((entropy >= 0) & (entropy <= max_entropy))
```

**Expected Outcome:**
- 4-5 tests verifying mathematical properties
- Normalization, causality, entropy bounds
- Tests catch numerical issues and implementation bugs

---

### ⬜ PR 8: Add Property-Based Tests

**Status:** Not started
**Estimated Effort:** Medium-High
**Files to Create:**
- `src/non_local_detector/tests/test_simulator_properties.py`

**Goal:**
Use Hypothesis to generate random test cases and verify simulator properties hold across wide parameter ranges.

**Test Strategy:**
```python
from hypothesis import given, strategies as st

@given(
    n_tetrodes=st.integers(min_value=1, max_value=10),
    n_runs=st.integers(min_value=1, max_value=5),
    sampling_frequency=st.integers(min_value=100, max_value=2000),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_simulator_always_produces_valid_output(
    n_tetrodes, n_runs, sampling_frequency, seed
):
    """Test that simulator produces valid output for random parameters."""
    # Generate compatible place field means
    n_neurons = n_tetrodes * 4
    place_field_means = np.linspace(0, 200, n_neurons, endpoint=False)

    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=sampling_frequency,
        n_runs=n_runs,
        seed=seed,
    )

    # Should always satisfy basic contracts
    assert len(sim.spike_times) == n_tetrodes
    assert all(np.all(np.diff(st) > 0) for st in sim.spike_times if len(st) > 0)
    assert np.all(np.isfinite(sim.position))

@given(
    position_std=st.floats(min_value=1.0, max_value=20.0),
    waveform_std=st.floats(min_value=5.0, max_value=50.0),
)
def test_encoding_model_fits_with_random_parameters(position_std, waveform_std):
    """Test that encoding model fits successfully with random parameters."""
    sim = make_simulated_run_data(seed=42)

    # Should not raise errors
    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=sim.position_time,
        position=sim.position,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
        environment=sim.environment,
        position_std=position_std,
        waveform_std=waveform_std,
        block_size=100,
        disable_progress_bar=True,
    )

    # Should produce valid outputs
    assert "occupancy" in encoding_model
    assert len(encoding_model["mean_rates"]) == len(sim.spike_times)
```

**Expected Outcome:**
- 3-5 property-based tests using Hypothesis
- Tests run 100+ random examples each
- Catches edge cases and unexpected parameter combinations

---

### ⬜ PR 9: Add Golden Regression Test

**Status:** Not started
**Estimated Effort:** Low
**Files to Create:**
- `src/non_local_detector/tests/test_golden_regression.py`
- `src/non_local_detector/tests/golden_data/` (directory)

**Goal:**
Create a golden regression test that saves complete decoder outputs to disk and verifies exact numerical match on future runs. Complements snapshot tests by saving actual arrays.

**Test Strategy:**
```python
import pickle
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden_data"

@pytest.fixture
def golden_path():
    """Path to golden data directory."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    return GOLDEN_DIR

def test_clusterless_decoder_golden_regression(golden_path):
    """Test that decoder produces identical outputs to saved golden data."""
    # Generate deterministic simulation
    sim = make_simulated_run_data(
        n_tetrodes=4,
        place_field_means=np.arange(0, 160, 10),
        sampling_frequency=500,
        n_runs=3,
        seed=12345,  # Fixed seed for golden test
    )

    # Fit decoder
    decoder = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    ).fit(...)

    # Predict
    results = decoder.predict(...)

    # Save or compare
    golden_file = golden_path / "clusterless_decoder_golden.pkl"

    if not golden_file.exists():
        # Save golden data (run once to create baseline)
        with open(golden_file, "wb") as f:
            pickle.dump({
                "posterior": results.acausal_posterior.values,
                "state_probs": results.acausal_state_probabilities.values,
            }, f)
        pytest.skip("Golden data created, skipping comparison")

    # Load golden data
    with open(golden_file, "rb") as f:
        golden = pickle.load(f)

    # Compare with tight tolerance
    np.testing.assert_allclose(
        results.acausal_posterior.values,
        golden["posterior"],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Posterior does not match golden data"
    )
```

**Expected Outcome:**
- 2-3 golden regression tests for different decoder types
- Saved baseline data in `golden_data/` directory
- Tests catch any numerical changes (even floating point differences)
- Clear instructions for updating golden data when needed

**When to Update Golden Data:**
- After intentional algorithm changes
- After dependency updates that affect numerics
- Run with `--update-golden` flag (implement in conftest.py)

---

## How to Continue

### Prerequisites

1. **Environment Setup:**
   ```bash
   conda activate non_local_detector
   cd /Users/edeno/Documents/GitHub/non_local_detector
   ```

2. **Verify Current State:**
   ```bash
   # All tests should pass
   pytest src/non_local_detector/tests/test_simulator_contract.py -v
   pytest src/non_local_detector/tests/test_oracle_correctness.py -v

   # Quality checks should pass
   ruff check src/
   ruff format --check src/
   black --check src/
   ```

### Starting PR 6 (KDE/GMM Agreement)

1. **Create new test file:**
   ```bash
   touch src/non_local_detector/tests/test_clusterless_likelihood_agreement.py
   ```

2. **Use this template:**
   ```python
   """Tests for clusterless likelihood model agreement.

   These tests verify that different likelihood models (KDE, GMM) produce
   qualitatively similar results on the same simulated data. This helps catch
   regressions where one model drastically diverges from expected behavior.
   """

   import numpy as np
   import pytest

   from non_local_detector.likelihoods.clusterless_kde import (
       fit_clusterless_kde_encoding_model,
       predict_clusterless_kde_log_likelihood,
   )
   from non_local_detector.likelihoods.clusterless_gmm import (
       fit_clusterless_gmm_encoding_model,
       predict_clusterless_gmm_log_likelihood,
   )
   from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

   @pytest.fixture
   def clusterless_sim_data():
       """Generate simulated data for likelihood agreement tests."""
       # ... implementation

   def test_kde_gmm_top_position_agreement(clusterless_sim_data):
       """Test that KDE and GMM agree on most likely position most of the time."""
       # ... implementation
   ```

3. **Follow the workflow:**
   ```bash
   # Write tests
   # Run tests
   pytest src/non_local_detector/tests/test_clusterless_likelihood_agreement.py -v

   # Quality checks
   ruff check src/non_local_detector/tests/test_clusterless_likelihood_agreement.py
   ruff format src/non_local_detector/tests/test_clusterless_likelihood_agreement.py

   # Commit
   git add src/non_local_detector/tests/test_clusterless_likelihood_agreement.py
   git commit -m "Add tests for KDE/GMM likelihood agreement

   Test that different clusterless likelihood models produce qualitatively
   similar results on same simulated data. Helps catch model regressions.

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

4. **Request code review:**
   Use Claude Code's code-reviewer agent after completing the implementation.

### Code Patterns Established

**Testing Pattern:**
```python
# 1. Create fixture for shared test data
@pytest.fixture
def test_data():
    sim = make_simulated_run_data(seed=FIXED_SEED)
    # ... prepare data
    return {...}

# 2. Add type hints to helper functions
def helper_function(
    param1: npt.NDArray[np.floating],
    param2: int,
) -> npt.NDArray[np.floating]:
    """Clear docstring."""
    # ... implementation

# 3. Extract constants to module level
THRESHOLD = 0.80
PARAMETER = 6.0

# 4. Write focused test functions
@pytest.mark.slow  # Mark slow tests
def test_specific_property(test_data: dict) -> None:
    """Test docstring explaining what property is being tested."""
    result = compute_something(test_data["input"])
    assert result >= THRESHOLD, f"Result {result} below threshold {THRESHOLD}"
```

**Simulator Usage Pattern:**
```python
# Always use seed for deterministic tests
sim = make_simulated_run_data(
    n_tetrodes=4,
    place_field_means=np.arange(0, 160, 10),  # Ensure divisible by n_tetrodes
    sampling_frequency=500,
    n_runs=3,
    seed=42,  # CRITICAL for reproducibility
)

# Access via dataclass fields
position_time = sim.position_time
spike_times = sim.spike_times  # List of arrays
environment = sim.environment  # Already fitted
```

**Quality Gate Workflow:**
```bash
# 1. Write code
# 2. Run tests
pytest path/to/test_file.py -v

# 3. Lint and format
ruff check path/to/file.py
ruff format path/to/file.py

# 4. Verify black compliance
black --check path/to/file.py

# 5. Commit with conventional format
git add path/to/file.py
git commit -m "Type: Brief description

Detailed explanation of changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Key Files Reference

**Simulator Code:**
- `src/non_local_detector/simulate/clusterless_simulation.py` - Main clusterless simulator
- `src/non_local_detector/simulate/sorted_spikes_simulation.py` - Sorted spikes simulator
- `src/non_local_detector/simulate/simulate.py` - Core simulation functions

**Contract and Tests:**
- `src/non_local_detector/tests/_sim_contract.py` - ClusterlessSimOutput definition
- `src/non_local_detector/tests/test_simulator_contract.py` - Contract enforcement
- `src/non_local_detector/tests/test_oracle_correctness.py` - Correctness validation

**Likelihood Models:**
- `src/non_local_detector/likelihoods/clusterless_kde.py` - KDE likelihood
- `src/non_local_detector/likelihoods/clusterless_gmm.py` - GMM likelihood
- `src/non_local_detector/likelihoods/__init__.py` - Algorithm registry

**Decoder Models:**
- `src/non_local_detector/models/decoder.py` - Base decoders
- `src/non_local_detector/models/base.py` - Base classes

### Testing Best Practices

1. **Always use fixtures for expensive setup** (simulation, model fitting)
2. **Add type hints to all helper functions** using `numpy.typing`
3. **Extract magic numbers to module-level constants** with descriptive names
4. **Use descriptive test names** that explain what property is being tested
5. **Mark slow tests** with `@pytest.mark.slow`
6. **Use seed=42 or other fixed seeds** for reproducibility
7. **Write focused tests** that test one property each
8. **Include clear assertion messages** explaining what went wrong
9. **Test edge cases** (empty electrodes, single time point, etc.)
10. **Follow AAA pattern** (Arrange, Act, Assert)

### Common Pitfalls to Avoid

1. **Don't forget seeding** - All random operations must be seeded for reproducible tests
2. **Ensure place_field_means divisible by n_tetrodes** - Otherwise simulator will fail
3. **Remember position is always 2D** - Even for 1D tracks, shape is `(n_time, 1)`
4. **Spike times are per-electrode lists** - Not NaN-padded tensors
5. **Run both ruff format AND black** - Both must pass for CI
6. **Use relative imports** - `from non_local_detector.simulate...`
7. **Check array shapes carefully** - Off-by-one errors common with bin edges vs centers

---

## Success Metrics

### Completed Metrics (PRs 0-5)

- ✅ ClusterlessSimOutput contract defined and documented
- ✅ Simulator emits decoder-ready per-electrode lists
- ✅ All simulator functions support deterministic seeding
- ✅ Existing tests migrated to new API
- ✅ 14 contract tests passing (<1s total)
- ✅ 3 oracle tests passing (~8.5s total)
- ✅ All quality gates passing (ruff, black)
- ✅ Code duplication eliminated from test fixtures
- ✅ Complete type hints on test helper functions

### Remaining Metrics (PRs 6-9)

- ⬜ KDE/GMM agreement tests (2-3 tests, <10s)
- ⬜ Posterior property tests (4-5 tests, <5s)
- ⬜ Property-based tests (3-5 tests, 100+ examples each)
- ⬜ Golden regression tests (2-3 tests with saved baselines)
- ⬜ All tests passing in CI
- ⬜ Test coverage >90% for simulator module
- ⬜ Documentation updated with new API examples

---

## Notes and Decisions

### Design Decisions

1. **Why per-electrode lists instead of tensors?**
   - Eliminates NaN padding and associated complexity
   - Matches natural structure of data (electrodes can have different spike counts)
   - Cleaner API for downstream decoders
   - Reduces memory usage (no padding)

2. **Why return dataclass instead of tuple?**
   - Self-documenting (named fields)
   - Type-safe access
   - Easy to extend without breaking existing code
   - Better IDE support

3. **Why include fitted Environment in output?**
   - Decoders need place bins for likelihood computation
   - Eliminates need for separate environment fitting step
   - Ensures consistency between training and test data

4. **Why default seed=0 instead of seed=None?**
   - Encourages deterministic testing by default
   - Users can explicitly pass seed=None for randomness
   - Reduces confusion about reproducibility

### Test Philosophy

1. **Contract tests** enforce structural invariants (shapes, types, sorting)
2. **Oracle tests** verify semantic correctness (decoder accuracy on known truth)
3. **Property tests** verify behavior across parameter ranges
4. **Snapshot tests** catch unintended regressions
5. **Golden tests** provide exact numerical baselines

### Performance Notes

- Contract tests: <1s total (fast, run in CI)
- Oracle tests: ~8.5s total (marked `@pytest.mark.slow`)
- Snapshot tests: ~15s total (marked `@pytest.mark.snapshot`)
- Full test suite: ~30s (acceptable for CI)

### Future Improvements

1. Consider adding `SortedSpikesSimOutput` dataclass for sorted spikes
2. Add visualization utilities for simulation outputs
3. Consider caching expensive fixture results (encoding models)
4. Add performance benchmarks for simulator
5. Document simulator parameters and their effects

---

## Related Documentation

- Main project docs: [CLAUDE.md](CLAUDE.md)
- Contract definition: [_sim_contract.py](src/non_local_detector/tests/_sim_contract.py)
- Simulator implementation: [clusterless_simulation.py](src/non_local_detector/simulate/clusterless_simulation.py)
- Example notebooks: [notebooks/](notebooks/)

---

**Last Updated:** 2025-10-16
**Current Commit:** `8a7abab` (Fix type hints and code duplication in oracle tests)
**Next PR:** PR 6 - Test KDE/GMM Agreement on Same Data
