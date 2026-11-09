"""Tests for simulator contract compliance.

These tests verify that simulator outputs conform to the ClusterlessSimOutput
contract defined in simulate.clusterless_simulation. They enforce shapes,
units, and invariants to catch regressions early.
"""

import numpy as np
import pytest

from non_local_detector.simulate import simulate_poisson_spikes
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

pytestmark = pytest.mark.unit


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
    """A silent electrode yields times of shape (0,) and features of shape (0, 4)."""
    # A 20 ms recording on a 1 cm track with one neuron per tetrode (peak rate
    # 100 Hz): only the field at 0 cm is on the track, so the other electrodes
    # are nearly silent; seed 3 gives spike counts [2, 0, 0, 0].
    sim = make_simulated_run_data(
        n_tetrodes=4,
        place_field_means=np.array([0.0, 10.0, 20.0, 30.0]),
        sampling_frequency=1000,
        n_runs=1,
        track_height=1.0,
        running_speed=100.0,
        seed=3,
    )

    spike_counts = [times.size for times in sim.spike_times]
    assert 0 in spike_counts, f"Expected at least one empty electrode: {spike_counts}"
    assert any(count > 0 for count in spike_counts), spike_counts

    for i, (times, features) in enumerate(
        zip(sim.spike_times, sim.spike_waveform_features, strict=True)
    ):
        if times.size > 0:
            continue
        assert times.shape == (0,), f"Electrode {i}: empty times must be shape (0,)"
        assert features.shape == (0, 4), (
            f"Electrode {i}: empty features must be shape (0, 4)"
        )
        assert np.issubdtype(times.dtype, np.floating)
        assert np.issubdtype(features.dtype, np.floating)


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


def test_simulate_poisson_spikes_returns_counts() -> None:
    """``simulate_poisson_spikes`` returns integer Poisson counts, not a 0/1 mask.

    The package unified two historically conflicting definitions (binary
    indicator vs. raw counts) onto counts. This pins that contract so a future
    re-binarization is caught directly rather than only transitively (every
    in-repo consumer currently masks the result with ``> 0``). Callers that need
    a boolean indicator recover it via ``counts > 0``, per the docstring.
    """
    # Rate high enough that some bins receive more than one spike.
    rate = np.full(2000, 200.0)
    counts = simulate_poisson_spikes(rate, sampling_frequency=100, seed=0)

    assert counts.shape == rate.shape, "counts must match the input rate shape"
    assert np.issubdtype(counts.dtype, np.integer), "counts must be integer-valued"
    assert np.all(counts >= 0), "Poisson counts are non-negative"
    assert counts.max() > 1, (
        "counts semantics must allow more than one spike per bin; a value of 1 "
        "everywhere would indicate the old binary-indicator behavior"
    )

    # The boolean spike indicator is recoverable via the documented `> 0` step.
    indicator = counts > 0
    assert indicator.dtype == np.bool_
    assert indicator.sum() <= counts.sum()


def test_get_firing_rate_without_weights_matches_unit_weights() -> None:
    """weights=None (the default) weights every time point equally."""
    from non_local_detector.simulate.sorted_spikes_simulation import get_firing_rate

    rng = np.random.default_rng(0)
    n_time = 200
    position = rng.uniform(0, 10, size=(n_time, 1))
    is_spike = (rng.uniform(size=n_time) < 0.2).astype(float)
    place_bin_centers = np.linspace(0, 10, 11)[:, np.newaxis]
    args = (
        is_spike,
        position,
        place_bin_centers,
        np.ones(11, dtype=bool),
        np.ones(n_time, dtype=bool),
        np.full(11, 0.1),
        np.array([1.0]),
    )

    np.testing.assert_allclose(
        get_firing_rate(*args), get_firing_rate(*args, weights=np.ones(n_time))
    )


def _gaussian_field(x, mean, max_rate, variance):
    return max_rate * np.exp(-0.5 * (x - mean) ** 2 / variance)


def test_place_field_rate_does_not_depend_on_the_trajectory() -> None:
    """The field peaks at ``max_rate`` at its center wherever the animal went.

    A field at 70 cm read at 60 cm is the same whether the trajectory reaches
    the center (0-100 cm) or stops short of it (0-60 cm).
    """
    from non_local_detector.simulate import simulate_place_field_firing_rate

    full = simulate_place_field_firing_rate([70.0], np.linspace(0, 100, 1001)[:, None])
    short = simulate_place_field_firing_rate([70.0], np.linspace(0, 60, 601)[:, None])

    expected = _gaussian_field(60.0, 70.0, max_rate=15.0, variance=12.5)
    np.testing.assert_allclose(full[600], expected, rtol=1e-12)
    np.testing.assert_allclose(short[-1], expected, rtol=1e-12)


def test_place_field_beyond_the_track_follows_its_gaussian() -> None:
    """A field centered off the track is only its Gaussian tail on the track."""
    from non_local_detector.simulate import simulate_place_field_firing_rate
    from non_local_detector.simulate.sorted_spikes_simulation import (
        create_place_field,
    )

    position = np.linspace(0.0, 100.0, 10_001)
    rate = simulate_place_field_firing_rate([120.0], position[:, None])
    sorted_rate = create_place_field(120.0, position)

    np.testing.assert_allclose(
        rate[-1], _gaussian_field(100.0, 120.0, 15.0, 12.5), rtol=1e-12
    )
    np.testing.assert_allclose(
        sorted_rate[-1],
        0.001 + _gaussian_field(100.0, 120.0, 20.0, 12.5),
        rtol=1e-12,
    )


def test_nan_position_leaves_other_rates_finite() -> None:
    """A NaN position gives a NaN rate at that sample only."""
    from non_local_detector.simulate import simulate_place_field_firing_rate
    from non_local_detector.simulate.sorted_spikes_simulation import (
        create_place_field,
    )

    position = np.linspace(0.0, 100.0, 101)
    position[10] = np.nan
    finite = np.ones(position.shape, dtype=bool)
    finite[10] = False

    for rate in (
        simulate_place_field_firing_rate([50.0], position[:, None]),
        create_place_field(50.0, position),
    ):
        assert np.all(np.isfinite(rate[finite]))
        assert np.isnan(rate[10])


def test_multiunit_spike_probability_and_neuron_choice_are_unbiased() -> None:
    """Each sample spikes with probability ``sum(rate) / fs``, split by rate.

    Two neurons with fields at the animal's position fire at 100 Hz each,
    sampled at 1 kHz: a sample spikes with probability 0.2 and each neuron
    owns half of the spikes. Neuron identity is read from the mark cluster
    (centers 0 and 5, unit variance, four dimensions).
    """
    from non_local_detector.simulate import simulate_multiunit_with_place_fields

    n_time, sampling_frequency, max_rate = 200_000, 1000, 100.0
    marks = simulate_multiunit_with_place_fields(
        np.zeros((2, 1)),
        np.zeros((n_time, 1)),
        max_rate=max_rate,
        sampling_frequency=sampling_frequency,
        rng=np.random.default_rng(0),
    )

    has_spike = ~np.isnan(marks[:, 0])
    p_spike = 2 * max_rate / sampling_frequency
    expected = n_time * p_spike
    standard_error = np.sqrt(n_time * p_spike * (1 - p_spike))
    assert abs(has_spike.sum() - expected) < 3 * standard_error, has_spike.sum()

    from_second = marks[has_spike].mean(axis=1) > 2.5
    fraction_standard_error = np.sqrt(0.25 / has_spike.sum())
    assert abs(from_second.mean() - 0.5) < 3 * fraction_standard_error, (
        from_second.mean()
    )


def test_multiunit_has_no_spike_at_a_nan_position() -> None:
    from non_local_detector.simulate import simulate_multiunit_with_place_fields

    position = np.zeros((1000, 1))
    position[::2] = np.nan
    marks = simulate_multiunit_with_place_fields(
        np.zeros((2, 1)),
        position,
        max_rate=400.0,
        sampling_frequency=1000,
        rng=np.random.default_rng(0),
    )

    assert np.all(np.isnan(marks[::2]))
    assert np.any(~np.isnan(marks[1::2, 0]))


def test_multiunit_refuses_more_than_one_expected_spike_per_sample() -> None:
    """One mark row per sample cannot hold a total rate above the sampling rate."""
    from non_local_detector.simulate import simulate_multiunit_with_place_fields

    with pytest.raises(ValueError, match="sampling_frequency"):
        simulate_multiunit_with_place_fields(
            np.zeros((2, 1)),
            np.zeros((10, 1)),
            max_rate=600.0,
            sampling_frequency=1000,
            rng=np.random.default_rng(0),
        )
