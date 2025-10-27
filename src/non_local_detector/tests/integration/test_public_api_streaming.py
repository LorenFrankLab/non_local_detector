"""Integration tests for streaming via public API.

Tests that the public API functions (fit/predict) properly expose streaming
and tiling parameters to users.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde_log import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing public API."""
    np.random.seed(42)

    # Time and position data (use numpy for Environment)
    n_time = 100
    position_time = np.linspace(0, 10, n_time)
    position = np.column_stack(
        [
            np.sin(position_time * 2 * np.pi / 10) * 50 + 50,
            np.cos(position_time * 2 * np.pi / 10) * 50 + 50,
        ]
    )

    # Spike data for 3 electrodes
    n_electrodes = 3
    spike_times = []
    spike_waveform_features = []

    for _ in range(n_electrodes):
        # Random spike times during the recording
        n_spikes = np.random.randint(80, 120)
        times = np.sort(np.random.uniform(0, 10, n_spikes))
        spike_times.append(times)

        # Random waveform features (4D)
        features = np.random.randn(n_spikes, 4) * 10
        spike_waveform_features.append(features)

    # Environment (requires numpy arrays)
    environment = Environment(
        place_bin_size=10.0,
        track_graph=None,
        edge_order=None,
        edge_spacing=None,
    )
    position_2d = position if position.ndim > 1 else position[:, np.newaxis]
    environment.fit_place_grid(position_2d)

    return {
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
        "environment": environment,
    }


class TestPublicAPIStreaming:
    """Tests for streaming parameters in public API."""

    def test_predict_with_enc_tiling(self, sample_data):
        """predict_clusterless_kde_log_likelihood works with enc_tile_size."""
        # Fit encoding model
        encoding_model = fit_clusterless_kde_encoding_model(
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            environment=sample_data["environment"],
            sampling_frequency=500,
            position_std=np.sqrt(12.5),
            waveform_std=24.0,
            block_size=50,
            disable_progress_bar=True,
        )

        # Decode time
        decode_time = jnp.linspace(1, 9, 20)

        # Predict WITHOUT tiling (baseline)
        log_likelihood_baseline = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            block_size=50,
            disable_progress_bar=True,
            # No tiling
        )

        # Predict WITH tiling
        log_likelihood_tiled = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            block_size=50,
            disable_progress_bar=True,
            enc_tile_size=30,  # Use encoding tiling
        )

        # Should produce same results
        assert log_likelihood_baseline.shape == log_likelihood_tiled.shape
        assert jnp.allclose(
            log_likelihood_baseline, log_likelihood_tiled, rtol=1e-5, atol=1e-8
        )

    def test_predict_with_streaming(self, sample_data):
        """predict_clusterless_kde_log_likelihood works with streaming mode."""
        # Fit encoding model
        encoding_model = fit_clusterless_kde_encoding_model(
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            environment=sample_data["environment"],
            sampling_frequency=500,
            position_std=np.sqrt(12.5),
            waveform_std=24.0,
            block_size=50,
            disable_progress_bar=True,
        )

        # Decode time
        decode_time = jnp.linspace(1, 9, 20)

        # Predict WITHOUT streaming (baseline)
        log_likelihood_precomputed = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            block_size=50,
            disable_progress_bar=True,
            enc_tile_size=30,  # Use tiling
            use_streaming=False,  # Precomputed mode
        )

        # Predict WITH streaming
        log_likelihood_streaming = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            block_size=50,
            disable_progress_bar=True,
            enc_tile_size=30,  # Required for streaming
            use_streaming=True,  # Streaming mode
        )

        # Should produce same results
        assert log_likelihood_precomputed.shape == log_likelihood_streaming.shape
        assert jnp.allclose(
            log_likelihood_precomputed,
            log_likelihood_streaming,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_predict_with_both_tilings(self, sample_data):
        """predict works with both enc_tile_size and pos_tile_size."""
        # Fit encoding model
        encoding_model = fit_clusterless_kde_encoding_model(
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            environment=sample_data["environment"],
            sampling_frequency=500,
            position_std=np.sqrt(12.5),
            waveform_std=24.0,
            disable_progress_bar=True,
        )

        decode_time = jnp.linspace(1, 9, 15)

        # Baseline
        log_likelihood_baseline = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            disable_progress_bar=True,
        )

        # With both tilings
        log_likelihood_tiled = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            disable_progress_bar=True,
            enc_tile_size=30,  # Tile encoding
            pos_tile_size=20,  # Tile positions
        )

        assert log_likelihood_baseline.shape == log_likelihood_tiled.shape
        assert jnp.allclose(
            log_likelihood_baseline, log_likelihood_tiled, rtol=1e-5, atol=1e-8
        )

    def test_predict_streaming_with_pos_tiling(self, sample_data):
        """Streaming works with position tiling."""
        encoding_model = fit_clusterless_kde_encoding_model(
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            environment=sample_data["environment"],
            disable_progress_bar=True,
        )

        decode_time = jnp.linspace(1, 9, 10)

        # Precomputed with both tilings
        log_likelihood_precomputed = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            disable_progress_bar=True,
            enc_tile_size=30,
            pos_tile_size=15,
            use_streaming=False,
        )

        # Streaming with both tilings
        log_likelihood_streaming = predict_clusterless_kde_log_likelihood(
            time=decode_time,
            position_time=sample_data["position_time"],
            position=sample_data["position"],
            spike_times=sample_data["spike_times"],
            spike_waveform_features=sample_data["spike_waveform_features"],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            environment=sample_data["environment"],
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            disable_progress_bar=True,
            enc_tile_size=30,
            pos_tile_size=15,
            use_streaming=True,
        )

        assert jnp.allclose(
            log_likelihood_precomputed,
            log_likelihood_streaming,
            rtol=1e-5,
            atol=1e-8,
        )
