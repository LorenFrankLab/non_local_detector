# Snapshot Tests

This directory contains snapshot tests for the `non_local_detector` package. Snapshot tests capture the expected output of models and algorithms, providing regression detection during refactoring and development.

## Overview

Snapshot tests use the [syrupy](https://github.com/tophat/syrupy) pytest plugin to automatically manage expected outputs. When a test runs, it compares the current output against a saved snapshot. If they differ, the test fails, alerting you to unexpected behavioral changes.

## Test Files

- **`test_model_snapshots.py`**: Snapshot tests for sorted spike models
  - `NonLocalSortedSpikesDetector` - Replay detection with local/non-local states
  - `ContFragSortedSpikesClassifier` - Continuous vs. fragmented classification
  - `SortedSpikesDecoder` - Basic position decoding
  - Encoding model properties and algorithm comparisons

- **`test_clusterless_model_snapshots.py`**: Snapshot tests for clusterless models
  - `ClusterlessDecoder` - Position decoding from waveform features
  - `ContFragClusterlessClassifier` - Classification with clusterless data
  - Encoding model properties

## Running Snapshot Tests

### Run all snapshot tests
```bash
pytest src/non_local_detector/tests/snapshots/ -v
```

### Run only snapshot-marked tests
```bash
pytest -m snapshot
```

### Run a specific test file
```bash
pytest src/non_local_detector/tests/snapshots/test_model_snapshots.py -v
```

### Exclude slow tests
```bash
pytest src/non_local_detector/tests/snapshots/ -m "not slow"
```

## Updating Snapshots

When you intentionally change model behavior, you need to update the snapshots:

### Update all snapshots
```bash
pytest src/non_local_detector/tests/snapshots/ --snapshot-update
```

### Update specific test snapshots
```bash
pytest src/non_local_detector/tests/snapshots/test_model_snapshots.py::test_sorted_spikes_decoder_snapshot --snapshot-update
```

### Review snapshot changes
After updating, use git to review what changed:
```bash
git diff src/non_local_detector/tests/snapshots/__snapshots__/
```

**Important**: Always review snapshot diffs carefully before committing. Unexpected changes may indicate bugs.

## Snapshot Storage

Snapshots are stored in the `__snapshots__/` directory as `.ambr` (AMber) files:
- `test_model_snapshots.ambr` - Snapshots for sorted spike models
- `test_clusterless_model_snapshots.ambr` - Snapshots for clusterless models

These files are human-readable and should be committed to version control.

## What Gets Snapshotted

Due to the large size of model outputs, we snapshot **summary statistics** rather than full arrays:

### State Probabilities
```python
{
    "shape": (n_time,),
    "states": ["Local", "Non-Local Continuous", ...],
    "mean_per_state": {...},
    "max_per_state": {...},
    "min_per_state": {...},
}
```

### Posterior Distributions
```python
{
    "shape": (n_time, n_position_bins),
    "dtype": "float32",
    "mean": 0.0123,
    "std": 0.0456,
    "min": 0.0,
    "max": 0.789,
    "sum": 1234.56,
    "first_5": [0.1, 0.2, ...],
    "last_5": [..., 0.8, 0.9],
}
```

### Model Attributes
- Transition matrix properties (shape, sum checks)
- Initial conditions
- Encoding model parameters

## Writing New Snapshot Tests

### Basic Pattern

```python
import pytest
from syrupy.assertion import SnapshotAssertion

@pytest.mark.snapshot
def test_my_model_snapshot(simulated_data: dict, snapshot: SnapshotAssertion):
    """Snapshot test for MyModel."""
    # Fit model
    model = MyModel().fit(
        simulated_data["time"],
        simulated_data["position"],
        simulated_data["spike_times"],
    )

    # Generate predictions
    results = model.predict(
        spike_times=simulated_data["spike_times"],
        time=simulated_data["time"],
    )

    # Snapshot summary statistics
    summary = {
        "state_probs_mean": float(np.mean(results.acausal_state_probabilities)),
        "state_probs_shape": results.acausal_state_probabilities.shape,
    }
    assert summary == snapshot
```

### Key Principles

1. **Use fixed random seeds** in fixtures for reproducibility
2. **Snapshot summary statistics**, not full arrays (to keep snapshots manageable)
3. **Use descriptive snapshot names** via `snapshot(name="descriptive_name")`
4. **Document what's being tested** in docstrings
5. **Mark tests appropriately**: `@pytest.mark.snapshot`, `@pytest.mark.slow`

### Custom Serializers

Helper functions for consistent serialization:

```python
def serialize_xarray_summary(data_array):
    """Serialize xarray DataArray to summary statistics."""
    arr = np.asarray(data_array)
    return {
        "shape": arr.shape,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        # ... more statistics
    }

def serialize_state_probabilities(state_probs):
    """Serialize state probabilities for snapshot comparison."""
    return {
        "shape": state_probs.shape,
        "states": list(state_probs.states.values),
        "mean_per_state": {...},
        # ... per-state statistics
    }
```

## Fixtures

### Sorted Spike Data
```python
@pytest.fixture
def simulated_data():
    """Generate simulated sorted spike data with fixed random seed."""
    np.random.seed(42)
    return make_simulated_data(n_neurons=30)
```

### Clusterless Data
```python
@pytest.fixture
def clusterless_simulated_data():
    """Generate simulated clusterless data with fixed random seed."""
    np.random.seed(42)
    return make_simulated_run_data(...)
```

## Troubleshooting

### Test fails after code change
1. **Expected**: Review the changes in model behavior
2. **If intentional**: Update the snapshot with `--snapshot-update`
3. **If unexpected**: Investigate the code change that caused the difference

### Snapshot is too large
- Switch to summary statistics instead of full arrays
- Use sampling (e.g., `first_5`, `last_5`) for large arrays
- Consider testing on a smaller dataset

### Test is too slow
- Mark with `@pytest.mark.slow`
- Reduce dataset size in fixture
- Use smaller parameter values (e.g., fewer neurons, shorter time periods)

### Random failures
- Ensure fixtures use fixed random seeds
- Check for non-deterministic operations (e.g., dictionary iteration order)
- Use JAX with fixed seed: `jax.random.PRNGKey(42)`

## Best Practices

1. **Review all snapshot changes** in PRs - they represent behavior changes
2. **Keep snapshots small** - use summary statistics, not full arrays
3. **Test representative cases** - cover main algorithms and edge cases
4. **Document skipped tests** - explain why and when to re-enable
5. **Use parametrize** for testing multiple algorithms/configurations
6. **Commit snapshots** - they're part of the test suite

## CI/CD Integration

Snapshot tests run automatically in CI:
- Tests fail if outputs don't match snapshots
- Use `--snapshot-update` locally, never in CI
- Always review snapshot diffs before merging PRs

## Further Reading

- [Syrupy Documentation](https://github.com/tophat/syrupy)
- [Snapshot Testing Benefits](https://jestjs.io/docs/snapshot-testing)
- Main README: [../../README.md](../../README.md)
