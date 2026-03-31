"""Property-based tests for simulator using Hypothesis.

These tests use Hypothesis to generate random test cases and verify that simulator
properties hold across wide parameter ranges. This helps catch edge cases and
unexpected parameter combinations.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.models.decoder import ClusterlessDecoder
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

# Hypothesis strategies for valid parameter ranges
n_tetrodes_strategy = st.integers(min_value=1, max_value=8)
n_runs_strategy = st.integers(min_value=1, max_value=5)
sampling_frequency_strategy = st.integers(min_value=100, max_value=2000)
seed_strategy = st.integers(min_value=0, max_value=10000)
position_std_strategy = st.floats(min_value=1.0, max_value=20.0)
waveform_std_strategy = st.floats(min_value=5.0, max_value=50.0)


@settings(max_examples=20, deadline=None)
@given(
    n_tetrodes=n_tetrodes_strategy,
    n_runs=n_runs_strategy,
    sampling_frequency=sampling_frequency_strategy,
    seed=seed_strategy,
)
def test_simulator_always_produces_valid_output(
    n_tetrodes: int,
    n_runs: int,
    sampling_frequency: int,
    seed: int,
) -> None:
    """Test that simulator produces valid output for random parameters.

    This property test verifies that the simulator satisfies basic contracts
    regardless of input parameters.
    """
    # Generate compatible place field means (divisible by n_tetrodes)
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
    assert len(sim.spike_times) == n_tetrodes, "Wrong number of electrodes"
    assert len(sim.spike_waveform_features) == n_tetrodes, (
        "Wrong number of feature arrays"
    )

    # All spike times should be strictly increasing per electrode
    for electrode_id, st_array in enumerate(sim.spike_times):
        if len(st_array) > 1:
            assert np.all(np.diff(st_array) > 0), (
                f"Electrode {electrode_id} times not strictly increasing"
            )

    # All data should be finite
    assert np.all(np.isfinite(sim.position)), "Position contains non-finite values"
    assert np.all(np.isfinite(sim.position_time)), (
        "Position time contains non-finite values"
    )

    # Spike features should be finite
    for electrode_id, features in enumerate(sim.spike_waveform_features):
        assert np.all(np.isfinite(features)), (
            f"Electrode {electrode_id} features contain non-finite values"
        )


@settings(max_examples=10, deadline=None)
@given(
    position_std=position_std_strategy,
    waveform_std=waveform_std_strategy,
)
def test_encoding_model_fits_with_random_parameters(
    position_std: float,
    waveform_std: float,
) -> None:
    """Test that encoding model fits successfully with random parameters.

    This verifies that the encoding model is robust to different smoothing
    parameter choices.
    """
    # Use fixed simulation for consistent test
    sim = make_simulated_run_data(
        n_tetrodes=3,
        place_field_means=np.arange(0, 120, 10),
        sampling_frequency=500,
        n_runs=2,
        seed=42,
    )

    # Should not raise errors
    try:
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
    except Exception as e:
        pytest.fail(
            f"Encoding model fit failed with position_std={position_std}, waveform_std={waveform_std}: {e}"
        )

    # Should produce valid outputs
    assert "occupancy" in encoding_model, "Missing occupancy in encoding model"
    assert "mean_rates" in encoding_model, "Missing mean_rates in encoding model"
    assert len(encoding_model["mean_rates"]) == len(sim.spike_times), (
        "Wrong number of mean rates"
    )

    # All outputs should be finite
    assert np.all(np.isfinite(encoding_model["occupancy"])), (
        "Occupancy contains non-finite values"
    )
    for rate in encoding_model["mean_rates"]:
        assert np.isfinite(rate), "Mean rate contains non-finite values"


@settings(max_examples=10, deadline=None)
@given(
    n_tetrodes=st.integers(min_value=2, max_value=6),
    seed=seed_strategy,
)
def test_decoder_output_shapes_consistent(
    n_tetrodes: int,
    seed: int,
) -> None:
    """Test that decoder outputs have consistent shapes.

    This verifies that decoder output dimensions are correct regardless
    of the number of electrodes.
    """
    # Generate simulation
    n_neurons = n_tetrodes * 4
    place_field_means = np.linspace(0, 160, n_neurons, endpoint=False)

    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        sampling_frequency=500,
        n_runs=2,
        seed=seed,
    )

    # Fit decoder
    decoder = ClusterlessDecoder(
        environments=sim.environment,
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    )

    # Use 70% for training
    n_encode = int(0.7 * len(sim.position_time))

    decoder.fit(
        position_time=sim.position_time[:n_encode],
        position=sim.position[:n_encode],
        spike_times=[
            st[st <= sim.position_time[n_encode - 1]] for st in sim.spike_times
        ],
        spike_waveform_features=[
            swf[st <= sim.position_time[n_encode - 1]]
            for st, swf in zip(
                sim.spike_times, sim.spike_waveform_features, strict=True
            )
        ],
    )

    # Predict on test data
    # Note: After GMM fix, decoder returns n outputs from n time inputs
    n_time_bins = 10
    test_time = np.linspace(
        sim.position_time[n_encode],
        sim.position_time[-1],
        n_time_bins,
    )

    results = decoder.predict(
        time=test_time,
        position_time=sim.position_time[n_encode:],
        position=sim.position[n_encode:],
        spike_times=[
            st[st > sim.position_time[n_encode - 1]] for st in sim.spike_times
        ],
        spike_waveform_features=[
            swf[st > sim.position_time[n_encode - 1]]
            for st, swf in zip(
                sim.spike_times, sim.spike_waveform_features, strict=True
            )
        ],
    )

    # Check output shapes
    posterior = results.acausal_posterior.values
    assert posterior.ndim == 2, f"Posterior should be 2D, got {posterior.ndim}D"
    assert posterior.shape[0] == n_time_bins, (
        f"Wrong time dimension: expected {n_time_bins}, got {posterior.shape[0]}"
    )

    # Note: Normalization is tested in test_posterior_properties.py
    # This test focuses on shape consistency across random parameters


@settings(max_examples=15, deadline=None)
@given(
    seed1=seed_strategy,
    seed2=seed_strategy,
)
def test_different_seeds_produce_different_outputs(
    seed1: int,
    seed2: int,
) -> None:
    """Test that different seeds produce different spike patterns.

    This verifies that seeding actually affects the random generation.
    """
    # Ensure seeds are different using Hypothesis assume()
    assume(seed1 != seed2)

    sim1 = make_simulated_run_data(
        n_tetrodes=3,
        place_field_means=np.arange(0, 120, 10),
        sampling_frequency=500,
        n_runs=2,
        seed=seed1,
    )

    sim2 = make_simulated_run_data(
        n_tetrodes=3,
        place_field_means=np.arange(0, 120, 10),
        sampling_frequency=500,
        n_runs=2,
        seed=seed2,
    )

    # At least one electrode should have different spike counts
    spike_counts1 = [len(st) for st in sim1.spike_times]
    spike_counts2 = [len(st) for st in sim2.spike_times]

    assert spike_counts1 != spike_counts2, (
        f"Different seeds produced same spike counts: {spike_counts1}"
    )


@settings(max_examples=10, deadline=None)
@given(
    seed=seed_strategy,
)
def test_same_seed_produces_identical_outputs(
    seed: int,
) -> None:
    """Test that same seed produces identical outputs.

    This verifies deterministic behavior for reproducibility.
    """
    params = {
        "n_tetrodes": 3,
        "place_field_means": np.arange(0, 120, 10),
        "sampling_frequency": 500,
        "n_runs": 2,
        "seed": seed,
    }

    sim1 = make_simulated_run_data(**params)
    sim2 = make_simulated_run_data(**params)

    # Spike counts should be identical
    for i, (st1, st2) in enumerate(
        zip(sim1.spike_times, sim2.spike_times, strict=True)
    ):
        assert len(st1) == len(st2), (
            f"Electrode {i}: Different spike counts with same seed"
        )
        np.testing.assert_array_equal(
            st1, st2, err_msg=f"Electrode {i}: Different spike times with same seed"
        )

    # Spike features should be identical
    for i, (swf1, swf2) in enumerate(
        zip(sim1.spike_waveform_features, sim2.spike_waveform_features, strict=True)
    ):
        np.testing.assert_array_equal(
            swf1,
            swf2,
            err_msg=f"Electrode {i}: Different spike features with same seed",
        )
