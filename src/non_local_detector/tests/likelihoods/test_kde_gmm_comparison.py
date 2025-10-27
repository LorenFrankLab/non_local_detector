"""Comprehensive comparison tests for KDE vs GMM clusterless likelihood implementations.

This test module verifies that both KDE and GMM implementations:
1. Can run end-to-end (fit + predict)
2. Have consistent API signatures
3. Produce valid outputs with the same input data
4. Support both local and non-local likelihood computation

The tests are NOT verifying numerical equivalence (KDE and GMM are different algorithms),
but rather ensuring both implementations work correctly and have compatible interfaces.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)


@pytest.fixture
def shared_simulation_data():
    """Create shared test data for both KDE and GMM implementations.

    This fixture generates synthetic data that can be used to test both
    likelihood implementations with identical inputs.
    """
    np.random.seed(42)

    # Time parameters
    dt = 0.02  # 20 ms bins
    n_time = 50
    time = np.arange(n_time + 1) * dt  # time bin edges

    # Position parameters (encoding period)
    position_time = np.linspace(0, time[-1], 200)
    position = np.column_stack(
        [
            np.linspace(0, 10, len(position_time)),  # x coordinate
            np.sin(np.linspace(0, 2 * np.pi, len(position_time))) * 2,  # y coordinate
        ]
    )

    # Spike parameters (encoding period)
    n_electrodes = 3
    n_features = 4
    encoding_spike_times = []
    encoding_spike_features = []

    for elec_idx in range(n_electrodes):
        # Generate random spike times within encoding period
        n_spikes = np.random.randint(30, 50)
        times = np.sort(np.random.uniform(time[0], time[-1], n_spikes))
        encoding_spike_times.append(times)

        # Generate random waveform features
        features = np.random.randn(n_spikes, n_features).astype(np.float32)
        encoding_spike_features.append(features)

    # Decoding period spikes (subset of encoding spikes for simplicity)
    decoding_spike_times = [times[:20] for times in encoding_spike_times]
    decoding_spike_features = [
        feats[:20] for feats in encoding_spike_features
    ]

    # Create and fit environment
    environment = Environment(position_range=[(0, 10), (-3, 3)])
    environment = environment.fit_place_grid(
        position=position, infer_track_interior=True
    )

    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "encoding_spike_times": encoding_spike_times,
        "encoding_spike_features": encoding_spike_features,
        "decoding_spike_times": decoding_spike_times,
        "decoding_spike_features": decoding_spike_features,
        "environment": environment,
    }


def test_kde_end_to_end_pipeline(shared_simulation_data):
    """Test that KDE implementation can run complete fit + predict pipeline.

    This verifies:
    1. fit_clusterless_kde_encoding_model produces valid encoding model
    2. predict_clusterless_kde_log_likelihood works with fitted model
    3. Both local and non-local likelihood computation work
    4. Outputs have expected shapes and finite values
    """
    data = shared_simulation_data

    # Convert to JAX arrays
    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    encoding_spike_times = [jnp.asarray(st) for st in data["encoding_spike_times"]]
    encoding_spike_features = [
        jnp.asarray(sf) for sf in data["encoding_spike_features"]
    ]
    decoding_spike_times = [jnp.asarray(st) for st in data["decoding_spike_times"]]
    decoding_spike_features = [
        jnp.asarray(sf) for sf in data["decoding_spike_features"]
    ]
    time = jnp.asarray(data["time"])

    # Step 1: Fit encoding model
    kde_encoding = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_spike_times,
        spike_waveform_features=encoding_spike_features,
        environment=data["environment"],
        sampling_frequency=50,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )

    # Verify encoding model structure
    assert isinstance(kde_encoding, dict)
    required_keys = {
        "occupancy",
        "occupancy_model",
        "gpi_models",
        "encoding_spike_waveform_features",
        "encoding_positions",
        "environment",
        "mean_rates",
        "summed_ground_process_intensity",
        "position_std",
        "waveform_std",
        "block_size",
    }
    assert required_keys.issubset(kde_encoding.keys())

    # Step 2: Predict non-local likelihood
    ll_nonlocal = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=decoding_spike_times,
        spike_waveform_features=decoding_spike_features,
        occupancy=kde_encoding["occupancy"],
        occupancy_model=kde_encoding["occupancy_model"],
        gpi_models=kde_encoding["gpi_models"],
        encoding_spike_waveform_features=kde_encoding[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=kde_encoding["encoding_positions"],
        environment=data["environment"],
        mean_rates=jnp.asarray(kde_encoding["mean_rates"]),
        summed_ground_process_intensity=kde_encoding[
            "summed_ground_process_intensity"
        ],
        position_std=jnp.asarray(kde_encoding["position_std"]),
        waveform_std=jnp.asarray(kde_encoding["waveform_std"]),
        is_local=False,
        block_size=100,
        disable_progress_bar=True,
    )

    # Verify non-local output
    assert ll_nonlocal.ndim == 2
    assert ll_nonlocal.shape[0] == len(time)
    assert ll_nonlocal.shape[1] > 0  # interior bins
    assert jnp.all(jnp.isfinite(ll_nonlocal))

    # Step 3: Predict local likelihood
    ll_local = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=decoding_spike_times,
        spike_waveform_features=decoding_spike_features,
        occupancy=kde_encoding["occupancy"],
        occupancy_model=kde_encoding["occupancy_model"],
        gpi_models=kde_encoding["gpi_models"],
        encoding_spike_waveform_features=kde_encoding[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=kde_encoding["encoding_positions"],
        environment=data["environment"],
        mean_rates=jnp.asarray(kde_encoding["mean_rates"]),
        summed_ground_process_intensity=kde_encoding[
            "summed_ground_process_intensity"
        ],
        position_std=jnp.asarray(kde_encoding["position_std"]),
        waveform_std=jnp.asarray(kde_encoding["waveform_std"]),
        is_local=True,
        block_size=100,
        disable_progress_bar=True,
    )

    # Verify local output
    assert ll_local.shape == (len(time), 1)
    assert jnp.all(jnp.isfinite(ll_local))


def test_gmm_end_to_end_pipeline(shared_simulation_data):
    """Test that GMM implementation can run complete fit + predict pipeline.

    This verifies:
    1. fit_clusterless_gmm_encoding_model produces valid encoding model
    2. predict_clusterless_gmm_log_likelihood works with fitted model
    3. Both local and non-local likelihood computation work
    4. Outputs have expected shapes and finite values
    """
    data = shared_simulation_data

    # Convert to JAX arrays
    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    encoding_spike_times = [jnp.asarray(st) for st in data["encoding_spike_times"]]
    encoding_spike_features = [
        jnp.asarray(sf) for sf in data["encoding_spike_features"]
    ]
    decoding_spike_times = [jnp.asarray(st) for st in data["decoding_spike_times"]]
    decoding_spike_features = [
        jnp.asarray(sf) for sf in data["decoding_spike_features"]
    ]
    time = jnp.asarray(data["time"])

    # Step 1: Fit encoding model
    # Note: Using fewer components to match small dataset size
    gmm_encoding = fit_clusterless_gmm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_spike_times,
        spike_waveform_features=encoding_spike_features,
        environment=data["environment"],
        sampling_frequency=50,
        weights=None,
        gmm_components_occupancy=8,
        gmm_components_gpi=8,
        gmm_components_joint=16,
        gmm_covariance_type_occupancy="full",
        gmm_covariance_type_gpi="full",
        gmm_covariance_type_joint="full",
        gmm_random_state=0,
        disable_progress_bar=True,
    )

    # Verify encoding model structure
    assert isinstance(gmm_encoding, dict)
    required_keys = {
        "environment",
        "occupancy_model",
        "interior_place_bin_centers",
        "occupancy_bins",
        "log_occupancy_bins",
        "gpi_models",
        "joint_models",
        "mean_rates",
        "summed_ground_process_intensity",
        "position_time",
    }
    assert required_keys.issubset(gmm_encoding.keys())

    # Step 2: Predict non-local likelihood
    ll_nonlocal = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=decoding_spike_times,
        spike_waveform_features=decoding_spike_features,
        encoding_model=gmm_encoding,
        is_local=False,
        spike_block_size=1000,
        bin_tile_size=None,
        disable_progress_bar=True,
    )

    # Verify non-local output
    assert ll_nonlocal.ndim == 2
    assert ll_nonlocal.shape[0] == len(time)
    assert ll_nonlocal.shape[1] > 0  # interior bins
    assert jnp.all(jnp.isfinite(ll_nonlocal))

    # Step 3: Predict local likelihood
    ll_local = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=decoding_spike_times,
        spike_waveform_features=decoding_spike_features,
        encoding_model=gmm_encoding,
        is_local=True,
        disable_progress_bar=True,
    )

    # Verify local output
    assert ll_local.shape == (len(time) - 1, 1)  # GMM uses time bin centers
    assert jnp.all(jnp.isfinite(ll_local))


def test_api_consistency_fit_functions(shared_simulation_data):
    """Test that KDE and GMM fit functions accept compatible input signatures.

    This verifies that both implementations can be called with the same
    basic input data, even though their specific parameters differ.
    """
    data = shared_simulation_data

    # Common required parameters
    common_params = {
        "position_time": jnp.asarray(data["position_time"]),
        "position": jnp.asarray(data["position"]),
        "spike_times": [jnp.asarray(st) for st in data["encoding_spike_times"]],
        "spike_waveform_features": [
            jnp.asarray(sf) for sf in data["encoding_spike_features"]
        ],
        "environment": data["environment"],
        "sampling_frequency": 50,
        "disable_progress_bar": True,
    }

    # Both should work with common parameters (plus their specific ones)
    kde_encoding = fit_clusterless_kde_encoding_model(
        **common_params,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
        block_size=100,
    )

    gmm_encoding = fit_clusterless_gmm_encoding_model(
        **common_params,
        weights=None,
        gmm_components_occupancy=8,
        gmm_components_gpi=8,
        gmm_components_joint=16,
        gmm_random_state=0,
    )

    # Both should return dictionaries with common keys
    common_keys = {"environment", "mean_rates", "summed_ground_process_intensity"}
    assert common_keys.issubset(kde_encoding.keys())
    assert common_keys.issubset(gmm_encoding.keys())


def test_api_consistency_predict_functions(shared_simulation_data):
    """Test that KDE and GMM predict functions accept compatible input signatures.

    This verifies that both implementations can process predictions with
    similar calling patterns, even though their encoding models differ.
    """
    data = shared_simulation_data

    # Fit both models first
    common_fit_params = {
        "position_time": jnp.asarray(data["position_time"]),
        "position": jnp.asarray(data["position"]),
        "spike_times": [jnp.asarray(st) for st in data["encoding_spike_times"]],
        "spike_waveform_features": [
            jnp.asarray(sf) for sf in data["encoding_spike_features"]
        ],
        "environment": data["environment"],
        "sampling_frequency": 50,
        "disable_progress_bar": True,
    }

    kde_encoding = fit_clusterless_kde_encoding_model(
        **common_fit_params,
        position_std=np.sqrt(12.5),
        waveform_std=24.0,
    )

    gmm_encoding = fit_clusterless_gmm_encoding_model(
        **common_fit_params,
        gmm_components_occupancy=8,
        gmm_components_gpi=8,
        gmm_components_joint=16,
    )

    # Common predict parameters
    time = jnp.asarray(data["time"])
    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    spike_times = [jnp.asarray(st) for st in data["decoding_spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["decoding_spike_features"]]

    # KDE prediction requires unpacking encoding model
    ll_kde = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_encoding["occupancy"],
        occupancy_model=kde_encoding["occupancy_model"],
        gpi_models=kde_encoding["gpi_models"],
        encoding_spike_waveform_features=kde_encoding[
            "encoding_spike_waveform_features"
        ],
        encoding_positions=kde_encoding["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_encoding["mean_rates"],
        summed_ground_process_intensity=kde_encoding[
            "summed_ground_process_intensity"
        ],
        position_std=kde_encoding["position_std"],
        waveform_std=kde_encoding["waveform_std"],
        is_local=False,
        disable_progress_bar=True,
    )

    # GMM prediction uses encoding model dictionary directly
    ll_gmm = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        encoding_model=gmm_encoding,
        is_local=False,
        disable_progress_bar=True,
    )

    # Both should produce valid likelihood arrays
    assert ll_kde.ndim == 2
    assert ll_gmm.ndim == 2
    assert jnp.all(jnp.isfinite(ll_kde))
    assert jnp.all(jnp.isfinite(ll_gmm))


def test_kde_gmm_output_shape_consistency(shared_simulation_data):
    """Test that KDE and GMM produce outputs with consistent shapes.

    While the numerical values will differ (different algorithms), the
    output shapes should be compatible for the same input data.
    """
    data = shared_simulation_data

    # Fit both models
    common_params = {
        "position_time": jnp.asarray(data["position_time"]),
        "position": jnp.asarray(data["position"]),
        "spike_times": [jnp.asarray(st) for st in data["encoding_spike_times"]],
        "spike_waveform_features": [
            jnp.asarray(sf) for sf in data["encoding_spike_features"]
        ],
        "environment": data["environment"],
        "disable_progress_bar": True,
    }

    kde_enc = fit_clusterless_kde_encoding_model(**common_params, position_std=1.0)
    gmm_enc = fit_clusterless_gmm_encoding_model(
        **common_params, gmm_components_occupancy=4, gmm_components_gpi=4, gmm_components_joint=8
    )

    # Predict with both
    time = jnp.asarray(data["time"])
    spike_times = [jnp.asarray(st) for st in data["decoding_spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["decoding_spike_features"]]

    ll_kde = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=common_params["position_time"],
        position=common_params["position"],
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_enc["occupancy"],
        occupancy_model=kde_enc["occupancy_model"],
        gpi_models=kde_enc["gpi_models"],
        encoding_spike_waveform_features=kde_enc["encoding_spike_waveform_features"],
        encoding_positions=kde_enc["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_enc["mean_rates"],
        summed_ground_process_intensity=kde_enc["summed_ground_process_intensity"],
        position_std=kde_enc["position_std"],
        waveform_std=kde_enc["waveform_std"],
        is_local=False,
        disable_progress_bar=True,
    )

    ll_gmm = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=common_params["position_time"],
        position=common_params["position"],
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        encoding_model=gmm_enc,
        is_local=False,
        disable_progress_bar=True,
    )

    # Shape consistency checks
    assert ll_kde.ndim == ll_gmm.ndim == 2
    assert ll_kde.shape[0] == ll_gmm.shape[0] == len(time)
    # Note: number of interior bins might differ slightly due to environment fitting
    assert ll_kde.shape[1] > 0 and ll_gmm.shape[1] > 0


def test_both_support_local_and_nonlocal_modes(shared_simulation_data):
    """Test that both implementations support local and non-local likelihood modes.

    This is a critical requirement for the decoder models that use these
    likelihood functions.
    """
    data = shared_simulation_data

    common_params = {
        "position_time": jnp.asarray(data["position_time"]),
        "position": jnp.asarray(data["position"]),
        "spike_times": [jnp.asarray(st) for st in data["encoding_spike_times"]],
        "spike_waveform_features": [
            jnp.asarray(sf) for sf in data["encoding_spike_features"]
        ],
        "environment": data["environment"],
        "disable_progress_bar": True,
    }

    # Fit models
    kde_enc = fit_clusterless_kde_encoding_model(**common_params, position_std=1.0)
    gmm_enc = fit_clusterless_gmm_encoding_model(
        **common_params, gmm_components_occupancy=4, gmm_components_gpi=4, gmm_components_joint=8
    )

    time = jnp.asarray(data["time"])
    spike_times = [jnp.asarray(st) for st in data["decoding_spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in data["decoding_spike_features"]]

    # Test KDE: local and non-local
    kde_nonlocal = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=common_params["position_time"],
        position=common_params["position"],
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_enc["occupancy"],
        occupancy_model=kde_enc["occupancy_model"],
        gpi_models=kde_enc["gpi_models"],
        encoding_spike_waveform_features=kde_enc["encoding_spike_waveform_features"],
        encoding_positions=kde_enc["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_enc["mean_rates"],
        summed_ground_process_intensity=kde_enc["summed_ground_process_intensity"],
        position_std=kde_enc["position_std"],
        waveform_std=kde_enc["waveform_std"],
        is_local=False,
        disable_progress_bar=True,
    )

    kde_local = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=common_params["position_time"],
        position=common_params["position"],
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=kde_enc["occupancy"],
        occupancy_model=kde_enc["occupancy_model"],
        gpi_models=kde_enc["gpi_models"],
        encoding_spike_waveform_features=kde_enc["encoding_spike_waveform_features"],
        encoding_positions=kde_enc["encoding_positions"],
        environment=data["environment"],
        mean_rates=kde_enc["mean_rates"],
        summed_ground_process_intensity=kde_enc["summed_ground_process_intensity"],
        position_std=kde_enc["position_std"],
        waveform_std=kde_enc["waveform_std"],
        is_local=True,
        disable_progress_bar=True,
    )

    # Test GMM: local and non-local
    gmm_nonlocal = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=common_params["position_time"],
        position=common_params["position"],
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        encoding_model=gmm_enc,
        is_local=False,
        disable_progress_bar=True,
    )

    gmm_local = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=common_params["position_time"],
        position=common_params["position"],
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        encoding_model=gmm_enc,
        is_local=True,
        disable_progress_bar=True,
    )

    # Verify shapes
    assert kde_nonlocal.ndim == 2 and kde_nonlocal.shape[1] > 1  # many positions
    assert kde_local.ndim == 2 and kde_local.shape[1] == 1  # single position
    assert gmm_nonlocal.ndim == 2 and gmm_nonlocal.shape[1] > 1  # many positions
    assert gmm_local.ndim == 2 and gmm_local.shape[1] == 1  # single position

    # Verify finite values
    assert jnp.all(jnp.isfinite(kde_nonlocal))
    assert jnp.all(jnp.isfinite(kde_local))
    assert jnp.all(jnp.isfinite(gmm_nonlocal))
    assert jnp.all(jnp.isfinite(gmm_local))
