"""Shared test fixtures and utilities for non_local_detector tests.

This module provides:
1. Environment fixtures for common test scenarios
2. Test data generators for spike and position data
3. Assertion helpers for validating probability distributions and HMM outputs

Usage:
    pytest automatically discovers fixtures in conftest.py. Import assertion
    helpers explicitly if needed:

    from conftest import assert_probability_distribution
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment


# ==============================================================================
# ENVIRONMENT FIXTURES
# ==============================================================================


@pytest.fixture
def simple_1d_environment():
    """Simple 1D linear environment with 1.0 bin size, range [0, 10].

    Returns:
        Environment: Fitted environment with 11 position bins.
    """
    env = Environment(
        environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),)
    )
    # Provide position array for fit_place_grid to determine dimensions
    dummy_pos = np.linspace(0.0, 10.0, 11)[:, None]
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    return env


@pytest.fixture
def simple_100_environment():
    """1D linear environment with 100 position bins, range [0, 100].

    Returns:
        Environment: Fitted environment with 50 sample positions.
    """
    env = Environment(
        environment_name="line", place_bin_size=5.0, position_range=((0.0, 100.0),)
    )
    position = np.linspace(0, 100, 50)[:, None]
    env = env.fit_place_grid(position, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    return env


@pytest.fixture
def simple_2d_environment():
    """Simple 2D open field environment with 5.0 bin size.

    Returns:
        Environment: Fitted 2D environment with 10x10 grid.
    """
    env = Environment(
        environment_name="open_field",
        place_bin_size=5.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    )
    # Create some sample positions in 2D
    x = np.linspace(0, 50, 20)
    y = np.linspace(0, 50, 20)
    xx, yy = np.meshgrid(x, y)
    position = np.column_stack([xx.ravel(), yy.ravel()])
    env = env.fit_place_grid(position, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    return env


# ==============================================================================
# HMM TEST DATA FIXTURES
# ==============================================================================


@pytest.fixture
def simple_hmm_2state():
    """Simple 2-state HMM parameters for testing.

    Returns:
        dict: Contains 'init', 'trans', and 'n_states' keys.
    """
    return {
        "init": jnp.array([0.5, 0.5]),
        "trans": jnp.array([[0.9, 0.1], [0.1, 0.9]]),
        "n_states": 2,
    }


@pytest.fixture
def simple_hmm_3state():
    """Simple 3-state HMM with cyclic transitions.

    Returns:
        dict: Contains 'init', 'trans', and 'n_states' keys.
    """
    return {
        "init": jnp.array([1.0, 0.0, 0.0]),
        # Deterministic cycle: 0 -> 1 -> 2 -> 0
        "trans": jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "n_states": 3,
    }


@pytest.fixture
def random_hmm(request):
    """Parametrized HMM with random initialization.

    Params:
        n_states (int): Number of states (passed via pytest.mark.parametrize)

    Returns:
        dict: Contains 'init', 'trans', and 'n_states' keys.
    """
    n_states = getattr(request, "param", 5)  # Default to 5 states
    np.random.seed(42)

    # Random initial distribution
    init = np.random.rand(n_states)
    init = init / init.sum()

    # Random stochastic matrix
    trans = np.random.rand(n_states, n_states)
    trans = trans / trans.sum(axis=1, keepdims=True)

    return {
        "init": jnp.asarray(init),
        "trans": jnp.asarray(trans),
        "n_states": n_states,
    }


# ==============================================================================
# SPIKE DATA FIXTURES
# ==============================================================================


@pytest.fixture
def minimal_spike_data():
    """Minimal spike data for basic testing.

    Returns:
        dict: Contains 'spike_times' (list of arrays) and 'time' array.
    """
    time = np.linspace(0.0, 10.0, 101)
    # Two neurons with a few spikes each
    spike_times = [np.array([2.0, 5.0, 5.1]), np.array([1.5, 7.2])]

    return {"spike_times": spike_times, "time": time}


@pytest.fixture
def synthetic_spike_data():
    """Synthetic spike data with position-dependent firing.

    Returns:
        dict: Contains 'spike_times', 'spike_features', 'position', 'time'.
    """
    n_time = 100
    n_neurons = 5
    n_features = 4

    time = np.linspace(0, 10, n_time)
    position = np.random.randn(n_time, 2) * 50  # 2D position

    # Generate spike times (random uniform)
    spike_times = [np.sort(np.random.uniform(0, 10, 20)) for _ in range(n_neurons)]

    # Generate spike features (e.g., waveform features)
    spike_features = [
        np.random.randn(len(st), n_features) for st in spike_times
    ]

    return {
        "spike_times": spike_times,
        "spike_features": spike_features,
        "position": position,
        "time": time,
    }


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================


def assert_probability_distribution(arr, axis=-1, rtol=1e-5, atol=1e-6):
    """Assert array is a valid probability distribution.

    Checks:
    1. All values are non-negative
    2. All values are <= 1
    3. Sum along axis equals 1.0

    Args:
        arr: Array to validate (numpy or jax array)
        axis: Axis along which to sum (default: -1)
        rtol: Relative tolerance for sum check
        atol: Absolute tolerance for sum check

    Raises:
        AssertionError: If any validation fails.
    """
    assert np.all(arr >= 0), "Probabilities must be non-negative"
    assert np.all(arr <= 1), "Probabilities must be <= 1"
    sums = arr.sum(axis=axis)
    assert np.allclose(sums, 1.0, rtol=rtol, atol=atol), (
        f"Probabilities must sum to 1 along axis {axis}, got sums: {sums}"
    )


def assert_stochastic_matrix(matrix, rtol=1e-5, atol=1e-6):
    """Assert matrix is row-stochastic (each row sums to 1).

    Args:
        matrix: 2D array to validate
        rtol: Relative tolerance for row sum check
        atol: Absolute tolerance for row sum check

    Raises:
        AssertionError: If validation fails.
    """
    assert matrix.ndim == 2, "Stochastic matrix must be 2D"
    assert matrix.shape[0] == matrix.shape[1], "Must be square matrix"
    assert np.all(matrix >= 0), "Elements must be non-negative"

    row_sums = matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, rtol=rtol, atol=atol), (
        f"Rows must sum to 1, got row sums: {row_sums}"
    )


def assert_all_finite(arr, name="array"):
    """Assert all values are finite (not NaN or inf).

    Args:
        arr: Array to validate (numpy or jax array)
        name: Name for error message

    Raises:
        AssertionError: If array contains NaN or inf.
    """
    # Handle both numpy and jax arrays
    if hasattr(arr, "shape"):
        finite_check = np.isfinite if isinstance(arr, np.ndarray) else jnp.isfinite
        assert np.all(finite_check(arr)), (
            f"{name} contains NaN or inf values. "
            f"Min: {np.min(arr)}, Max: {np.max(arr)}, "
            f"NaNs: {np.sum(np.isnan(arr))}, Infs: {np.sum(np.isinf(arr))}"
        )


def assert_valid_shape(arr, expected_shape, name="array"):
    """Assert array has expected shape.

    Args:
        arr: Array to validate
        expected_shape: Expected shape tuple
        name: Name for error message

    Raises:
        AssertionError: If shapes don't match.
    """
    assert arr.shape == expected_shape, (
        f"{name} has shape {arr.shape}, expected {expected_shape}"
    )


def assert_non_negative(arr, name="array"):
    """Assert all values are non-negative.

    Args:
        arr: Array to validate
        name: Name for error message

    Raises:
        AssertionError: If array contains negative values.
    """
    assert np.all(arr >= 0), (
        f"{name} contains negative values. Min: {np.min(arr)}"
    )


def assert_in_range(arr, min_val, max_val, name="array"):
    """Assert all values are in specified range [min_val, max_val].

    Args:
        arr: Array to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name for error message

    Raises:
        AssertionError: If any values are outside range.
    """
    assert np.all(arr >= min_val), (
        f"{name} contains values below {min_val}. Min: {np.min(arr)}"
    )
    assert np.all(arr <= max_val), (
        f"{name} contains values above {max_val}. Max: {np.max(arr)}"
    )


def assert_entropy_decreased(filtered, smoothed, name="entropy"):
    """Assert smoothed distribution has lower or equal entropy than filtered.

    Entropy should decrease when incorporating future information.

    Args:
        filtered: Filtered probability distribution
        smoothed: Smoothed probability distribution
        name: Name for error message

    Raises:
        AssertionError: If smoothed entropy exceeds filtered entropy.
    """

    def entropy(p):
        """Compute Shannon entropy, handling zeros."""
        return -np.sum(p * np.log(p + 1e-10), axis=-1)

    filtered_entropy = entropy(filtered).mean()
    smoothed_entropy = entropy(smoothed).mean()

    assert smoothed_entropy <= filtered_entropy + 1e-5, (
        f"{name}: smoothed entropy ({smoothed_entropy:.6f}) should not exceed "
        f"filtered entropy ({filtered_entropy:.6f})"
    )


def assert_valid_state_sequence(states, n_states, name="states"):
    """Assert Viterbi output is a valid state sequence.

    Args:
        states: Array of state indices
        n_states: Total number of states
        name: Name for error message

    Raises:
        AssertionError: If any state index is invalid.
    """
    # Handle both numpy and jax arrays
    if hasattr(states, "__array__"):
        states_np = np.asarray(states)
        assert np.all((states_np >= 0) & (states_np < n_states)), (
            f"{name} must be in [0, {n_states}), got values in "
            f"[{np.min(states_np)}, {np.max(states_np)}]"
        )
