"""Parity tests for clusterless KDE implementations.

Tests that the original (clusterless_kde.py) and log-space (clusterless_kde_log.py)
implementations produce numerically equivalent results.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_original,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model as fit_original,
)
from non_local_detector.likelihoods.clusterless_kde import (
    kde_distance,
)
from non_local_detector.likelihoods.clusterless_kde import (
    predict_clusterless_kde_log_likelihood as predict_original,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity as estimate_log,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    fit_clusterless_kde_encoding_model as fit_log,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    log_kde_distance,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    predict_clusterless_kde_log_likelihood as predict_log,
)


@pytest.fixture
def synthetic_clusterless_data():
    """Generate synthetic clusterless data for testing.

    Returns realistic waveform features and positions.
    """
    np.random.seed(42)

    # Time and position
    n_time_pos = 1000
    position_time = np.linspace(0, 10, n_time_pos)
    position = np.linspace(0, 100, n_time_pos)[:, None]  # 1D position

    # Spike data for 3 electrodes
    n_electrodes = 3
    spike_times = []
    spike_waveform_features = []

    for _ in range(n_electrodes):
        # Random spike times
        n_spikes = np.random.randint(50, 100)
        electrode_spike_times = np.sort(
            np.random.uniform(position_time[0], position_time[-1], n_spikes)
        )
        spike_times.append(electrode_spike_times)

        # Random waveform features (4 dimensions, realistic scale)
        electrode_features = np.random.randn(n_spikes, 4) * 20.0 + 50.0
        spike_waveform_features.append(electrode_features)

    return {
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
    }


@pytest.fixture
def simple_environment():
    """Create a simple 1D environment."""
    env = Environment(
        environment_name="line", place_bin_size=5.0, position_range=((0.0, 100.0),)
    )
    # Fit with dummy position
    dummy_pos = np.linspace(0.0, 100.0, 21)[:, None]
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    return env


def test_kde_distance_vs_log_kde_distance():
    """Test that kde_distance and log_kde_distance are consistent.

    kde_distance should equal exp(log_kde_distance).
    """
    eval_points = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    samples = jnp.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]])
    std = jnp.array([1.0, 1.0])

    # Compute both versions
    distance = kde_distance(eval_points, samples, std)
    log_distance = log_kde_distance(eval_points, samples, std)

    # Check consistency: exp(log_distance) ≈ distance
    assert np.allclose(
        np.exp(log_distance), distance, rtol=1e-5, atol=1e-8
    ), "kde_distance and log_kde_distance are inconsistent"


def test_estimate_intensity_moderate_features():
    """Test joint mark intensity estimation with moderate feature distances.

    With moderate features, both implementations should give similar results.
    """
    n_enc_spikes = 50
    n_dec_spikes = 20
    n_pos_bins = 30
    n_features = 4

    # Moderate feature values (not extreme)
    np.random.seed(42)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 10 + 50)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 10 + 50)
    enc_weights = jnp.ones(n_enc_spikes)
    waveform_stds = jnp.array([5.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 5.0

    # Compute position distance (same for both)
    enc_positions = jnp.array(np.random.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])

    position_distance = kde_distance(interior_bins, enc_positions, position_std)
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # Original implementation
    ll_original = estimate_original(
        dec_features,
        enc_features,
        enc_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
    )

    # Log-space implementation
    ll_log = estimate_log(
        dec_features,
        enc_features,
        enc_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
    )

    # Check shapes
    assert ll_original.shape == (n_dec_spikes, n_pos_bins), "Original shape wrong"
    assert ll_log.shape == (n_dec_spikes, n_pos_bins), "Log-space shape wrong"

    # Check values are close (moderate features should agree well)
    assert np.allclose(
        ll_original, ll_log, rtol=1e-3, atol=1e-4
    ), f"Implementations differ: max diff = {np.abs(ll_original - ll_log).max()}"

    # Check all values are finite
    assert np.all(np.isfinite(ll_original)), "Original has non-finite values"
    assert np.all(np.isfinite(ll_log)), "Log-space has non-finite values"


def test_estimate_intensity_extreme_features():
    """Test joint mark intensity with extreme feature distances.

    With extreme features, log-space should be more numerically stable.
    We just verify both return finite values.
    """
    n_enc_spikes = 30
    n_dec_spikes = 10
    n_pos_bins = 20
    n_features = 4

    # Extreme feature values (large distances)
    np.random.seed(42)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 50 + 100)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 50 + 200)
    enc_weights = jnp.ones(n_enc_spikes)
    waveform_stds = jnp.array([10.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 2.0

    # Position distance
    enc_positions = jnp.array(np.random.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])

    position_distance = kde_distance(interior_bins, enc_positions, position_std)
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # Original implementation may underflow
    ll_original = estimate_original(
        dec_features,
        enc_features,
        enc_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        position_distance,
    )

    # Log-space should handle better
    ll_log = estimate_log(
        dec_features,
        enc_features,
        enc_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
    )

    # Log-space should always be finite
    assert np.all(
        np.isfinite(ll_log)
    ), "Log-space has non-finite values even with extreme features"

    # Original may have issues, but shouldn't crash
    # (May have -inf values due to underflow, which is OK)
    assert ll_original.shape == (n_dec_spikes, n_pos_bins), "Original shape wrong"


def test_fit_encoding_model_parity(synthetic_clusterless_data, simple_environment):
    """Test that both fit functions produce equivalent encoding models."""
    data = synthetic_clusterless_data
    env = simple_environment

    # Fit with both implementations
    enc_original = fit_original(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    enc_log = fit_log(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Check that key outputs are equivalent
    assert np.allclose(
        enc_original["occupancy"], enc_log["occupancy"], rtol=1e-5, atol=1e-6
    ), "Occupancy differs"

    assert np.allclose(
        enc_original["mean_rates"], enc_log["mean_rates"], rtol=1e-5, atol=1e-6
    ), "Mean rates differ"

    assert np.allclose(
        enc_original["summed_ground_process_intensity"],
        enc_log["summed_ground_process_intensity"],
        rtol=1e-5,
        atol=1e-6,
    ), "Summed GPI differs"

    # Check that encoding positions are the same
    for i in range(len(data["spike_times"])):
        assert np.allclose(
            enc_original["encoding_positions"][i],
            enc_log["encoding_positions"][i],
            rtol=1e-5,
            atol=1e-6,
        ), f"Encoding positions differ for electrode {i}"


def test_predict_local_likelihood_parity(
    synthetic_clusterless_data, simple_environment
):
    """Test that local likelihood predictions are consistent."""
    data = synthetic_clusterless_data
    env = simple_environment

    # Fit encoding model (use original, should be same as log)
    enc = fit_original(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Decode on a small time window
    time = np.linspace(5.0, 6.0, 20)

    # Predict with original
    ll_original = predict_original(
        time=time,
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        encoding_spike_weights=enc["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.array(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=enc["position_std"],
        waveform_std=enc["waveform_std"],
        is_local=True,
        block_size=100,
        disable_progress_bar=True,
    )

    # Predict with log-space
    ll_log = predict_log(
        time=time,
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        encoding_spike_weights=enc["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.array(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=enc["position_std"],
        waveform_std=enc["waveform_std"],
        is_local=True,
        block_size=100,
        disable_progress_bar=True,
    )

    # Check shapes
    assert ll_original.shape == (len(time), 1), "Original local likelihood shape wrong"
    assert ll_log.shape == (len(time), 1), "Log-space local likelihood shape wrong"

    # Check values are close
    # Note: Local likelihood uses different code paths, may have larger differences
    assert np.allclose(
        ll_original, ll_log, rtol=1e-2, atol=1e-3
    ), f"Local likelihoods differ: max diff = {np.abs(ll_original - ll_log).max()}"

    # Check finite
    assert np.all(np.isfinite(ll_original)), "Original has non-finite local likelihood"
    assert np.all(np.isfinite(ll_log)), "Log-space has non-finite local likelihood"


def test_predict_nonlocal_likelihood_parity(
    synthetic_clusterless_data, simple_environment
):
    """Test that non-local likelihood predictions are consistent."""
    data = synthetic_clusterless_data
    env = simple_environment

    # Fit encoding model
    enc = fit_original(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Decode on a small time window
    time = np.linspace(5.0, 6.0, 10)

    # Predict with original
    ll_original = predict_original(
        time=time,
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        encoding_spike_weights=enc["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.array(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=enc["position_std"],
        waveform_std=enc["waveform_std"],
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    # Predict with log-space
    ll_log = predict_log(
        time=time,
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        encoding_spike_weights=enc["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.array(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=enc["position_std"],
        waveform_std=enc["waveform_std"],
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    # Check shapes (non-local has n_position_bins columns)
    n_bins = enc["occupancy"].shape[0]
    assert ll_original.shape == (len(time), n_bins), "Original non-local shape wrong"
    assert ll_log.shape == (len(time), n_bins), "Log-space non-local shape wrong"

    # Check values are close
    # Non-local likelihood is the critical test - this is where scan fix matters
    assert np.allclose(
        ll_original, ll_log, rtol=1e-2, atol=1e-3
    ), f"Non-local likelihoods differ: max diff = {np.abs(ll_original - ll_log).max()}"

    # Check finite
    assert np.all(
        np.isfinite(ll_original)
    ), "Original has non-finite non-local likelihood"
    assert np.all(np.isfinite(ll_log)), "Log-space has non-finite non-local likelihood"


def test_memory_efficiency_scan_vs_vmap():
    """Verify that scan-based implementation doesn't materialize huge arrays.

    This test doesn't check memory directly, but verifies the implementation
    runs without OOM on moderately large problem sizes that would fail with vmap.
    """
    # Problem size that would OOM with vmap but not with scan
    n_enc_spikes = 500
    n_dec_spikes = 100
    n_pos_bins = 200
    n_features = 4

    np.random.seed(42)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 10 + 50)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 10 + 50)
    enc_weights = jnp.ones(n_enc_spikes)
    waveform_stds = jnp.array([5.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 5.0

    # Position distance
    enc_positions = jnp.array(np.random.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # This should not OOM with scan-based implementation
    ll_log = estimate_log(
        dec_features,
        enc_features,
        enc_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
    )

    # Verify result
    assert ll_log.shape == (n_dec_spikes, n_pos_bins), "Shape is wrong"
    assert np.all(np.isfinite(ll_log)), "Result has non-finite values"
