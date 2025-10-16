"""Tests for sorted spikes GLM likelihood model.

Tests the Poisson GLM implementation for sorted spike data, including spline
basis generation, model fitting, and likelihood prediction.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_poisson_regression,
    fit_sorted_spikes_glm_encoding_model,
    make_spline_design_matrix,
    make_spline_predict_matrix,
    predict_sorted_spikes_glm_log_likelihood,
)


@pytest.fixture
def simple_1d_environment():
    """Create a simple 1D linear environment for testing."""
    env = Environment(
        environment_name="test_track",
        place_bin_size=5.0,
        position_range=((0.0, 100.0),),
    )
    position = np.linspace(0, 100, 101)[:, None]
    env = env.fit_place_grid(position=position, infer_track_interior=False)
    return env


@pytest.fixture
def simple_spike_data():
    """Generate simple synthetic spike data for testing."""
    n_time = 100
    n_neurons = 3
    sampling_frequency = 500.0

    # Create position trajectory
    position_time = np.linspace(0, 1, n_time)
    position = np.linspace(0, 100, n_time)[:, None]

    # Create spike times for each neuron
    # Neuron 0: spikes around position 25
    # Neuron 1: spikes around position 50
    # Neuron 2: spikes around position 75
    spike_times = []
    for center_pos in [25, 50, 75]:
        # Find times when animal is near center position
        near_center = np.abs(position.squeeze() - center_pos) < 10
        spike_time_indices = np.where(near_center)[0]
        # Sample some of those times
        n_spikes = min(10, len(spike_time_indices))
        selected = np.random.choice(spike_time_indices, n_spikes, replace=False)
        spike_times.append(position_time[sorted(selected)])

    return {
        "position_time": position_time,
        "position": position,
        "spike_times": [jnp.asarray(st) for st in spike_times],
        "sampling_frequency": sampling_frequency,
    }


class TestSplineDesignMatrix:
    """Test spline design matrix generation."""

    def test_make_spline_design_matrix_returns_correct_shape(self):
        """Design matrix should have correct dimensions."""
        # Arrange
        n_time = 50
        position = np.linspace(0, 100, n_time)[:, None]
        place_bin_edges = np.array([[0, 100]])
        knot_spacing = 10.0

        # Act
        design_matrix = make_spline_design_matrix(
            position, place_bin_edges, knot_spacing
        )

        # Assert
        assert design_matrix.shape[0] == n_time
        assert design_matrix.shape[1] > 1  # At least intercept + some basis functions

    def test_make_spline_design_matrix_includes_intercept(self):
        """First column should be all ones (intercept)."""
        # Arrange
        position = np.linspace(0, 100, 50)[:, None]
        place_bin_edges = np.array([[0, 100]])

        # Act
        design_matrix = make_spline_design_matrix(position, place_bin_edges)

        # Assert
        assert np.allclose(design_matrix[:, 0], 1.0)

    def test_make_spline_design_matrix_with_2d_position(self):
        """Should handle 2D position data."""
        # Arrange
        n_time = 30
        position = np.column_stack([
            np.linspace(0, 100, n_time),
            np.linspace(0, 50, n_time)
        ])
        place_bin_edges = np.array([[0, 100], [0, 50]])
        knot_spacing = 20.0

        # Act
        design_matrix = make_spline_design_matrix(
            position, place_bin_edges, knot_spacing
        )

        # Assert
        assert design_matrix.shape[0] == n_time
        assert design_matrix.shape[1] > 1

    def test_make_spline_predict_matrix_matches_design_shape(self):
        """Predict matrix should have consistent basis functions."""
        # Arrange
        n_fit = 50
        n_predict = 30
        position_fit = np.linspace(0, 100, n_fit)[:, None]
        position_predict = np.linspace(10, 90, n_predict)[:, None]
        place_bin_edges = np.array([[0, 100]])

        design_matrix = make_spline_design_matrix(position_fit, place_bin_edges)
        design_info = design_matrix.design_info

        # Act
        predict_matrix = make_spline_predict_matrix(design_info, position_predict)

        # Assert
        assert predict_matrix.shape[0] == n_predict
        assert predict_matrix.shape[1] == design_matrix.shape[1]  # Same number of basis functions

    def test_make_spline_predict_matrix_handles_nan_positions(self):
        """Should handle NaN positions gracefully."""
        # Arrange
        position_fit = np.linspace(0, 100, 50)[:, None]
        place_bin_edges = np.array([[0, 100]])

        design_matrix = make_spline_design_matrix(position_fit, place_bin_edges)
        design_info = design_matrix.design_info

        # Create prediction positions with NaN
        position_predict = np.array([[10.0], [np.nan], [50.0], [np.nan]])

        # Act
        predict_matrix = make_spline_predict_matrix(design_info, jnp.asarray(position_predict))

        # Assert
        assert predict_matrix.shape[0] == 4
        # NaN positions should produce NaN rows
        assert jnp.all(jnp.isnan(predict_matrix[1, :]))
        assert jnp.all(jnp.isnan(predict_matrix[3, :]))
        # Non-NaN positions should be finite
        assert jnp.all(jnp.isfinite(predict_matrix[0, :]))
        assert jnp.all(jnp.isfinite(predict_matrix[2, :]))


class TestPoissonRegression:
    """Test Poisson regression fitting."""

    def test_fit_poisson_regression_returns_coefficients(self):
        """Should return coefficient array of correct size."""
        # Arrange
        n_time = 100
        n_basis = 10
        design_matrix = np.random.randn(n_time, n_basis)
        design_matrix[:, 0] = 1.0  # Intercept
        spikes = np.random.poisson(5, size=n_time)
        weights = np.ones(n_time)

        # Act
        coefficients = fit_poisson_regression(
            design_matrix, spikes, weights, l2_penalty=1e-3
        )

        # Assert
        assert coefficients.shape == (n_basis,)
        assert jnp.all(jnp.isfinite(coefficients))

    def test_fit_poisson_regression_with_zero_spikes(self):
        """Should handle neuron with no spikes."""
        # Arrange
        n_time = 50
        n_basis = 5
        design_matrix = np.random.randn(n_time, n_basis)
        design_matrix[:, 0] = 1.0
        spikes = np.zeros(n_time)  # No spikes
        weights = np.ones(n_time)

        # Act
        coefficients = fit_poisson_regression(design_matrix, spikes, weights)

        # Assert
        assert jnp.all(jnp.isfinite(coefficients))
        # With no spikes, predicted rate should be very low
        predicted_rate = jnp.exp(design_matrix @ coefficients)
        assert jnp.mean(predicted_rate) < 1.0

    def test_fit_poisson_regression_with_uniform_spikes(self):
        """Should handle uniform spike distribution."""
        # Arrange
        n_time = 100
        n_basis = 8
        design_matrix = np.random.randn(n_time, n_basis)
        design_matrix[:, 0] = 1.0
        spikes = np.ones(n_time) * 5  # Constant spike count
        weights = np.ones(n_time)

        # Act
        coefficients = fit_poisson_regression(design_matrix, spikes, weights)

        # Assert
        # With uniform data, spatial coefficients should be near zero
        # (only intercept should be significant)
        assert jnp.all(jnp.isfinite(coefficients))
        assert np.abs(coefficients[1:]).max() < np.abs(coefficients[0])

    def test_fit_poisson_regression_respects_weights(self):
        """Weighting should affect fit."""
        # Arrange
        n_time = 50
        n_basis = 5
        design_matrix = np.random.randn(n_time, n_basis)
        design_matrix[:, 0] = 1.0
        spikes = np.random.poisson(3, size=n_time)

        # Fit with uniform weights
        weights_uniform = np.ones(n_time)
        coef_uniform = fit_poisson_regression(design_matrix, spikes, weights_uniform)

        # Fit with non-uniform weights (down-weight second half)
        weights_skewed = np.ones(n_time)
        weights_skewed[n_time // 2:] = 0.1
        coef_skewed = fit_poisson_regression(design_matrix, spikes, weights_skewed)

        # Assert - coefficients should differ
        assert not jnp.allclose(coef_uniform, coef_skewed, rtol=0.1)


class TestFitGLMEncodingModel:
    """Test full GLM encoding model fitting."""

    def test_fit_glm_encoding_model_returns_expected_keys(
        self, simple_1d_environment, simple_spike_data
    ):
        """Should return dictionary with all expected keys."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data

        # Act
        encoding = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            disable_progress_bar=True,
        )

        # Assert
        expected_keys = {
            "coefficients",
            "place_fields",
            "design_info",
            "place_bin_centers",
            "is_track_interior",
        }
        assert expected_keys.issubset(encoding.keys())

    def test_fit_glm_encoding_model_place_fields_shape(
        self, simple_1d_environment, simple_spike_data
    ):
        """Place fields should have correct shape."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data
        n_neurons = len(data["spike_times"])

        # Act
        encoding = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            disable_progress_bar=True,
        )

        # Assert
        place_fields = encoding["place_fields"]
        n_place_bins = env.place_bin_centers_.shape[0]
        assert len(place_fields) == n_neurons
        for pf in place_fields:
            assert pf.shape[0] == n_place_bins
            assert jnp.all(pf >= 0)  # Rates should be non-negative

    def test_fit_glm_encoding_model_with_custom_knot_spacing(
        self, simple_1d_environment, simple_spike_data
    ):
        """Should respect custom knot spacing parameter."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data

        # Act with different knot spacings
        encoding_coarse = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            emission_knot_spacing=30.0,  # Coarse
            disable_progress_bar=True,
        )

        encoding_fine = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            emission_knot_spacing=10.0,  # Fine
            disable_progress_bar=True,
        )

        # Assert - finer spacing should produce more complex model (more coefficients)
        n_coef_coarse = encoding_coarse["coefficients"][0].shape[0]
        n_coef_fine = encoding_fine["coefficients"][0].shape[0]
        assert n_coef_fine >= n_coef_coarse


class TestPredictGLMLogLikelihood:
    """Test GLM log-likelihood prediction."""

    def test_predict_glm_log_likelihood_nonlocal_returns_correct_shape(
        self, simple_1d_environment, simple_spike_data
    ):
        """Non-local prediction should return likelihood for all bins."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data

        encoding = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            disable_progress_bar=True,
        )

        # Create decoding time window
        time = np.linspace(0, 0.5, 10)

        # Act
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            time=jnp.asarray(time),
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            place_bin_centers=encoding["place_bin_centers"],
            coefficients=encoding["coefficients"],
            design_info=encoding["design_info"],
            is_track_interior=encoding["is_track_interior"],
            sampling_frequency=data["sampling_frequency"],
            is_local=False,
            disable_progress_bar=True,
        )

        # Assert
        n_time_bins = len(time) - 1
        n_place_bins = np.sum(env.is_track_interior_)
        assert log_likelihood.shape == (n_time_bins, n_place_bins)
        assert jnp.all(jnp.isfinite(log_likelihood))

    def test_predict_glm_log_likelihood_local_returns_correct_shape(
        self, simple_1d_environment, simple_spike_data
    ):
        """Local prediction should return likelihood only at current position."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data

        encoding = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            disable_progress_bar=True,
        )

        time = np.linspace(0, 0.5, 10)

        # Act
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            time=jnp.asarray(time),
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            place_bin_centers=encoding["place_bin_centers"],
            coefficients=encoding["coefficients"],
            design_info=encoding["design_info"],
            is_track_interior=encoding["is_track_interior"],
            sampling_frequency=data["sampling_frequency"],
            is_local=True,
            disable_progress_bar=True,
        )

        # Assert
        n_time_bins = len(time) - 1
        assert log_likelihood.shape == (n_time_bins,)  # One value per time bin
        assert jnp.all(jnp.isfinite(log_likelihood))

    def test_predict_glm_log_likelihood_with_no_spikes(
        self, simple_1d_environment, simple_spike_data
    ):
        """Should handle time periods with no spikes."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data

        encoding = fit_sorted_spikes_glm_encoding_model(
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            sampling_frequency=data["sampling_frequency"],
            disable_progress_bar=True,
        )

        # Use time period with no spikes (well beyond data)
        time = np.linspace(10.0, 10.5, 10)

        # Act
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            time=jnp.asarray(time),
            position_time=jnp.asarray(data["position_time"]),
            position=jnp.asarray(data["position"]),
            spike_times=data["spike_times"],
            place_bin_centers=encoding["place_bin_centers"],
            coefficients=encoding["coefficients"],
            design_info=encoding["design_info"],
            is_track_interior=encoding["is_track_interior"],
            sampling_frequency=data["sampling_frequency"],
            is_local=False,
            disable_progress_bar=True,
        )

        # Assert - should still produce valid likelihoods (negative due to Poisson)
        assert jnp.all(jnp.isfinite(log_likelihood))
        assert jnp.all(log_likelihood < 0)  # Log likelihood should be negative
