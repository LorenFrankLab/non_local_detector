"""Tests for simulator contract compliance.

These tests verify that simulator outputs conform to the ClusterlessSimOutput
contract defined in tests._sim_contract. They enforce shapes, units, and
invariants to catch regressions early.
"""

import numpy as np
import pytest

from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


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

    # Basic shape checks
    assert sim.edges.ndim == 1, "Edges must be 1D"
    assert sim.edges.size >= 2, "Must have at least 2 bin edges"
    assert sim.position_time.ndim == 1, "Position time must be 1D"
    assert sim.position.ndim == 2, "Position must be 2D (n_time, n_pos_dims)"
    assert sim.position.shape[0] == sim.position_time.shape[0], (
        "Position and position_time must have same length"
    )

    # Per-electrode list lengths
    n_electrodes = len(sim.spike_times)
    assert len(sim.spike_waveform_features) == n_electrodes, (
        "spike_times and spike_waveform_features must have same length"
    )
    assert n_electrodes == 3, f"Expected 3 electrodes, got {n_electrodes}"

    # Per-electrode array shapes
    for i, (times, features) in enumerate(
        zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ):
        assert times.ndim == 1, f"Electrode {i}: spike times must be 1D"
        assert features.ndim == 2, f"Electrode {i}: features must be 2D"
        assert features.shape[0] == times.shape[0], (
            f"Electrode {i}: feature count must match spike count"
        )
        assert features.shape[1] == 4, f"Electrode {i}: expected 4 features per spike"

    # Dtypes
    assert np.issubdtype(sim.position_time.dtype, np.floating), (
        "position_time must be float"
    )
    assert np.issubdtype(sim.position.dtype, np.floating), "position must be float"
    assert np.issubdtype(sim.edges.dtype, np.floating), "edges must be float"
    for i, times in enumerate(sim.spike_times):
        assert np.issubdtype(times.dtype, np.floating), (
            f"Electrode {i}: spike times must be float"
        )
    for i, features in enumerate(sim.spike_waveform_features):
        assert np.issubdtype(features.dtype, np.floating), (
            f"Electrode {i}: features must be float"
        )


def test_times_sorted_and_in_bounds() -> None:
    """Verify times are sorted and within valid bounds."""
    sim = make_simulated_run_data(
        n_tetrodes=2, sampling_frequency=500, n_runs=1, seed=1
    )

    # Position time must be sorted
    assert np.all(np.diff(sim.position_time) > 0), (
        "position_time must be strictly increasing"
    )

    # Edges must be sorted
    assert np.all(np.diff(sim.edges) > 0), "edges must be strictly increasing"

    # Spike times bounds and sorting
    t_min, t_max = sim.edges[0], sim.edges[-1]
    for i, times in enumerate(sim.spike_times):
        if times.size == 0:
            continue  # Empty electrode is valid
        assert np.all(np.isfinite(times)), f"Electrode {i}: times must be finite"
        assert np.all((times >= t_min) & (times <= t_max)), (
            f"Electrode {i}: spike times must be within [t_min, t_max]"
        )
        if times.size > 1:
            assert np.all(np.diff(times) > 0), (
                f"Electrode {i}: spike times must be strictly increasing"
            )


def test_no_nans_finite_features() -> None:
    """Verify no NaN values in features and all values are finite."""
    sim = make_simulated_run_data(
        n_tetrodes=2, sampling_frequency=500, n_runs=1, seed=2
    )

    # Check spike waveform features for NaNs
    for i, features in enumerate(sim.spike_waveform_features):
        assert np.all(np.isfinite(features)), f"Electrode {i}: features must be finite"
        assert not np.any(np.isnan(features)), f"Electrode {i}: no NaN values allowed"

    # Check position for NaNs
    assert np.all(np.isfinite(sim.position)), "position must be finite"
    assert not np.any(np.isnan(sim.position)), "position must not contain NaN"

    # Check position_time for NaNs
    assert np.all(np.isfinite(sim.position_time)), "position_time must be finite"
    assert not np.any(np.isnan(sim.position_time)), "position_time must not contain NaN"


def test_optional_empty_electrode() -> None:
    """Verify that empty electrodes have correct shapes."""
    # Use very short duration to potentially get empty electrodes
    sim = make_simulated_run_data(
        n_tetrodes=4, sampling_frequency=1000, n_runs=1, seed=3
    )

    for i, (times, features) in enumerate(
        zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ):
        if times.size == 0:
            # Empty electrode
            assert times.shape == (0,), f"Electrode {i}: empty times must be shape (0,)"
            assert features.shape[0] == 0, (
                f"Electrode {i}: empty features must have 0 rows"
            )
            assert features.shape[1] == 4, (
                f"Electrode {i}: empty features must still have 4 columns"
            )
            assert features.shape == (
                0,
                4,
            ), f"Electrode {i}: empty features must be shape (0, 4)"


def test_bin_widths_consistency() -> None:
    """Verify bin_widths matches np.diff(edges)."""
    sim = make_simulated_run_data(
        n_tetrodes=2, sampling_frequency=500, n_runs=1, seed=4
    )

    if sim.bin_widths is not None:
        expected_bin_widths = np.diff(sim.edges)
        np.testing.assert_allclose(
            sim.bin_widths,
            expected_bin_widths,
            rtol=1e-10,
            err_msg="bin_widths must equal np.diff(edges)",
        )


def test_environment_fitted() -> None:
    """Verify that environment is fitted with position data."""
    sim = make_simulated_run_data(
        n_tetrodes=2, sampling_frequency=500, n_runs=1, seed=5
    )

    # Check that environment exists and has expected attributes after fitting
    assert sim.environment is not None, "environment must be provided"
    assert hasattr(sim.environment, "place_bin_centers_"), (
        "environment must be fitted (has place_bin_centers_)"
    )
    assert sim.environment.place_bin_centers_ is not None, (
        "place_bin_centers_ must be set after fitting"
    )


def test_deterministic_seeding() -> None:
    """Verify that same seed produces identical outputs."""
    seed = 42
    n_tetrodes = 3
    place_field_means = np.arange(0, 120, 10)  # 12 neurons, divisible by 3
    sim1 = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=2,
        seed=seed,
    )
    sim2 = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=2,
        seed=seed,
    )

    # Check spike counts are identical
    counts1 = [len(t) for t in sim1.spike_times]
    counts2 = [len(t) for t in sim2.spike_times]
    assert counts1 == counts2, "Same seed must produce identical spike counts"

    # Check spike times are identical
    for i, (t1, t2) in enumerate(zip(sim1.spike_times, sim2.spike_times, strict=True)):
        np.testing.assert_array_equal(
            t1,
            t2,
            err_msg=f"Electrode {i}: spike times must be identical with same seed",
        )

    # Check features are identical
    for i, (f1, f2) in enumerate(
        zip(sim1.spike_waveform_features, sim2.spike_waveform_features, strict=True)
    ):
        np.testing.assert_array_equal(
            f1, f2, err_msg=f"Electrode {i}: features must be identical with same seed"
        )


def test_different_seeds_produce_different_outputs() -> None:
    """Verify that different seeds produce different outputs."""
    n_tetrodes = 3
    place_field_means = np.arange(0, 120, 10)  # 12 neurons, divisible by 3
    sim1 = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=2,
        seed=10,
    )
    sim2 = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=2,
        seed=20,
    )

    # Check that at least one electrode has different spike counts or times
    counts1 = [len(t) for t in sim1.spike_times]
    counts2 = [len(t) for t in sim2.spike_times]

    # It's extremely unlikely (probability ~0) that different seeds produce
    # identical spike patterns
    assert counts1 != counts2 or not all(
        np.array_equal(t1, t2)
        for t1, t2 in zip(sim1.spike_times, sim2.spike_times, strict=True)
    ), "Different seeds should produce different outputs"


def test_spike_times_units_are_seconds() -> None:
    """Verify that spike times are in seconds (not samples)."""
    sampling_frequency = 500  # Hz
    duration_runs = 1
    track_height = 175
    running_speed = 15

    # Expected duration in seconds
    expected_duration = duration_runs * 2 * track_height / running_speed

    sim = make_simulated_run_data(
        n_tetrodes=2,
        sampling_frequency=sampling_frequency,
        n_runs=duration_runs,
        track_height=track_height,
        running_speed=running_speed,
        seed=6,
    )

    # Check position time is in reasonable range (seconds, not samples)
    assert sim.position_time[-1] < expected_duration * 2, (
        "position_time should be in seconds"
    )
    assert sim.position_time[-1] > expected_duration * 0.5, (
        "position_time seems too small"
    )

    # Check spike times are in same range as position time
    for i, times in enumerate(sim.spike_times):
        if times.size > 0:
            assert times[-1] <= sim.position_time[-1], (
                f"Electrode {i}: spike times must be within position time range"
            )
            # Spike times should be in seconds (< 100), not samples (> 1000)
            assert times[-1] < 1000, (
                f"Electrode {i}: spike times appear to be in samples, not seconds"
            )


@pytest.mark.parametrize("n_tetrodes", [1, 2, 4, 8])
def test_variable_electrode_counts(n_tetrodes: int) -> None:
    """Test that simulator works with different numbers of electrodes."""
    # Adjust place fields to be divisible by n_tetrodes
    n_neurons = n_tetrodes * 4  # 4 neurons per tetrode
    place_field_means = np.linspace(0, 200, n_neurons, endpoint=False)

    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=1,
        seed=7,
    )

    assert len(sim.spike_times) == n_tetrodes, (
        f"Expected {n_tetrodes} electrodes in spike_times"
    )
    assert len(sim.spike_waveform_features) == n_tetrodes, (
        f"Expected {n_tetrodes} electrodes in spike_waveform_features"
    )


def test_position_is_2d():
    """Verify position is always 2D even for 1D tracks."""
    sim = make_simulated_run_data(
        n_tetrodes=2, sampling_frequency=500, n_runs=1, seed=8
    )

    assert sim.position.ndim == 2, "position must be 2D"
    assert sim.position.shape[1] == 1, (
        "position must have 1 spatial dimension for 1D track"
    )
