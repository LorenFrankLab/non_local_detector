"""Tests for sorted spikes GLM likelihood model.

Tests the Poisson GLM implementation for sorted spike data, including spline
basis generation, model fitting, and likelihood prediction.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import EPS
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
    sampling_frequency = float(n_time)  # Match position sampling rate

    # Create position trajectory
    position_time = np.linspace(0, 1, n_time)
    position = np.linspace(0, 100, n_time)[:, None]

    # Create spike times for each neuron
    # Neuron 0: spikes around position 25
    # Neuron 1: spikes around position 50
    # Neuron 2: spikes around position 75
    rng = np.random.default_rng(0)
    spike_times = []
    for center_pos in [25, 50, 75]:
        # Find times when animal is near center position
        near_center = np.abs(position.squeeze() - center_pos) < 10
        spike_time_indices = np.where(near_center)[0]
        # Sample some of those times
        n_spikes = min(10, len(spike_time_indices))
        selected = rng.choice(spike_time_indices, n_spikes, replace=False)
        spike_times.append(position_time[sorted(selected)])

    return {
        "position_time": position_time,
        "position": position,
        "spike_times": [jnp.asarray(st) for st in spike_times],
        "sampling_frequency": sampling_frequency,
    }


@pytest.mark.unit
class TestSplineDesignMatrix:
    """Test spline design matrix generation."""

    def test_make_spline_design_matrix_returns_correct_shape(self):
        """Design matrix should have correct dimensions."""
        # Arrange
        n_time = 50
        position = np.linspace(0, 100, n_time)[:, None]
        place_bin_edges = np.linspace(0, 100, 21)[:, None]  # 20 bins, shape (21, 1)
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
        place_bin_edges = np.linspace(0, 100, 21)[:, None]  # 20 bins, shape (21, 1)

        # Act
        design_matrix = make_spline_design_matrix(position, place_bin_edges)

        # Assert
        assert np.allclose(design_matrix[:, 0], 1.0)

    def test_make_spline_design_matrix_with_2d_position(self):
        """Should handle 2D position data."""
        # Arrange
        n_time = 30
        position = np.column_stack(
            [np.linspace(0, 100, n_time), np.linspace(0, 50, n_time)]
        )
        place_bin_edges = np.column_stack(
            [
                np.linspace(0, 100, 11),  # 10 bins in x
                np.linspace(0, 50, 11),  # 10 bins in y
            ]
        )  # shape (11, 2)
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
        place_bin_edges = np.linspace(0, 100, 21)[:, None]  # 20 bins, shape (21, 1)

        design_matrix = make_spline_design_matrix(position_fit, place_bin_edges)
        design_info = design_matrix.design_info

        # Act
        predict_matrix = make_spline_predict_matrix(design_info, position_predict)

        # Assert
        assert predict_matrix.shape[0] == n_predict
        assert (
            predict_matrix.shape[1] == design_matrix.shape[1]
        )  # Same number of basis functions

    def test_make_spline_predict_matrix_handles_nan_positions(self):
        """Should handle NaN positions gracefully."""
        # Arrange
        position_fit = np.linspace(0, 100, 50)[:, None]
        place_bin_edges = np.linspace(0, 100, 21)[:, None]  # 20 bins, shape (21, 1)

        design_matrix = make_spline_design_matrix(position_fit, place_bin_edges)
        design_info = design_matrix.design_info

        # Create prediction positions with NaN
        position_predict = np.array([[10.0], [np.nan], [50.0], [np.nan]])

        # Act
        predict_matrix = make_spline_predict_matrix(
            design_info, jnp.asarray(position_predict)
        )

        # Assert
        assert predict_matrix.shape[0] == 4
        # NaN positions should produce NaN rows
        assert jnp.all(jnp.isnan(predict_matrix[1, :]))
        assert jnp.all(jnp.isnan(predict_matrix[3, :]))
        # Non-NaN positions should be finite
        assert jnp.all(jnp.isfinite(predict_matrix[0, :]))
        assert jnp.all(jnp.isfinite(predict_matrix[2, :]))


@pytest.mark.unit
class TestPoissonRegression:
    """Test Poisson regression fitting."""

    def test_fit_poisson_regression_returns_coefficients(self):
        """Should return coefficient array of correct size."""
        # Arrange
        rng = np.random.default_rng(0)
        n_time = 100
        n_basis = 10
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0  # Intercept
        spikes = rng.poisson(5, size=n_time)
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
        rng = np.random.default_rng(0)
        n_time = 50
        n_basis = 5
        design_matrix = rng.standard_normal((n_time, n_basis))
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

    def test_fit_poisson_regression_with_zero_exposure(self):
        """No exposure yields finite EPS-floor rates even if counts are nonzero."""
        position = np.linspace(-1.0, 1.0, 20)
        design_matrix = np.column_stack([np.ones(position.size), position, position**2])

        coefficients = fit_poisson_regression(
            design_matrix,
            spikes=np.full(position.size, 2.0),
            weights=np.zeros(position.size),
        )

        assert coefficients.shape == (design_matrix.shape[1],)
        assert np.all(np.isfinite(coefficients))
        predicted_rate = np.exp(design_matrix @ np.asarray(coefficients))
        np.testing.assert_allclose(predicted_rate, EPS, rtol=1e-5, atol=0.0)

    @pytest.mark.parametrize("weight_scale", [1.0, 1e-8])
    def test_fit_poisson_regression_preserves_small_positive_exposure(
        self, weight_scale
    ):
        """Small positive weights still recover the known constant Poisson rate.

        The spikes are weighted by the same scale as the exposure, as the
        weighted event counts from a uniformly down-weighted fit are.
        """
        position = np.linspace(-1.0, 1.0, 20)
        design_matrix = np.column_stack([np.ones(position.size), position])

        coefficients = fit_poisson_regression(
            design_matrix,
            spikes=np.full(position.size, 2.0 * weight_scale),
            weights=np.full(position.size, weight_scale),
        )

        assert np.all(np.isfinite(coefficients))
        predicted_rate = np.exp(design_matrix @ np.asarray(coefficients))
        np.testing.assert_allclose(predicted_rate, 2.0, rtol=1e-5, atol=0.0)

    def test_fit_poisson_regression_with_uniform_spikes(self):
        """Should handle uniform spike distribution."""
        # Arrange
        rng = np.random.default_rng(0)
        n_time = 100
        n_basis = 8
        design_matrix = rng.standard_normal((n_time, n_basis))
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
        rng = np.random.default_rng(0)
        n_time = 50
        n_basis = 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        spikes = rng.poisson(3, size=n_time)

        # Fit with uniform weights
        weights_uniform = np.ones(n_time)
        coef_uniform = fit_poisson_regression(design_matrix, spikes, weights_uniform)

        # Fit with non-uniform weights (down-weight second half)
        weights_skewed = np.ones(n_time)
        weights_skewed[n_time // 2 :] = 0.1
        coef_skewed = fit_poisson_regression(design_matrix, spikes, weights_skewed)

        # Assert - coefficients should differ
        assert not jnp.allclose(coef_uniform, coef_skewed, rtol=0.1)

    def test_glm_large_gradient_warned(self, monkeypatch):
        """A fit that leaves a large final gradient emits a ``UserWarning``.

        Convergence is judged by the actual gradient norm rather than SciPy's
        ``success`` flag, because BFGS frequently reports ``success=False``
        ("precision loss") even at a perfectly good minimum. Here we force a
        result whose gradient is far from zero, which is genuine
        non-convergence and must warn.
        """
        # Arrange - a small, well-conditioned regression input.
        rng = np.random.default_rng(0)
        n_time = 50
        n_basis = 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        spikes = rng.poisson(3, size=n_time)
        weights = np.ones(n_time)

        from scipy.optimize import OptimizeResult  # type: ignore[import-untyped]

        import non_local_detector.likelihoods.sorted_spikes_glm as glm_module

        def fake_minimize(fun, x0, **kwargs):
            return OptimizeResult(
                x=np.asarray(x0),
                jac=np.full(n_basis, 0.5),  # large gradient -> not converged
                success=False,
                message="forced non-convergence for test",
                nit=7,
                fun=1.234567,
            )

        monkeypatch.setattr(glm_module, "minimize", fake_minimize)

        # Act / Assert
        with pytest.warns(UserWarning, match="may not have converged"):
            fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=1e-3)

    def test_glm_diverged_nonfinite_warned(self, monkeypatch):
        """A fit that diverges to a non-finite loss/gradient must warn.

        Regression: ``grad_norm = max(|jac|)`` is ``NaN`` for a diverged fit and
        ``NaN > tol`` is ``False``, so the gradient-only check silently passed a
        broken fit. The check now also fires on a non-finite loss/gradient.
        """
        rng = np.random.default_rng(0)
        n_time, n_basis = 50, 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        spikes = rng.poisson(3, size=n_time)
        weights = np.ones(n_time)

        from scipy.optimize import OptimizeResult  # type: ignore[import-untyped]

        import non_local_detector.likelihoods.sorted_spikes_glm as glm_module

        def fake_minimize(fun, x0, **kwargs):
            return OptimizeResult(
                x=np.asarray(x0),
                jac=np.full(n_basis, np.nan),  # diverged -> non-finite gradient
                success=False,
                message="forced divergence for test",
                nit=7,
                fun=np.nan,
            )

        monkeypatch.setattr(glm_module, "minimize", fake_minimize)

        with pytest.warns(UserWarning, match="may not have converged"):
            fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=1e-3)

    def test_glm_benign_precision_loss_does_not_warn(self, monkeypatch):
        """``success=False`` with a near-zero gradient is benign precision loss
        and must NOT warn (this is the dominant noise source in practice)."""
        rng = np.random.default_rng(0)
        n_time = 50
        n_basis = 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        spikes = rng.poisson(3, size=n_time)
        weights = np.ones(n_time)

        from scipy.optimize import OptimizeResult  # type: ignore[import-untyped]

        import non_local_detector.likelihoods.sorted_spikes_glm as glm_module

        def fake_minimize(fun, x0, **kwargs):
            return OptimizeResult(
                x=np.asarray(x0),
                jac=np.full(n_basis, 1e-8),  # effectively zero gradient
                success=False,
                message="Desired error not necessarily achieved due to precision loss.",
                nit=42,
                fun=1.234567,
            )

        monkeypatch.setattr(glm_module, "minimize", fake_minimize)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=1e-3)

        convergence_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "converge" in str(w.message).lower()
        ]
        assert convergence_warnings == [], (
            "Benign precision-loss exit with a near-zero gradient should not "
            f"warn; got: {[str(w.message) for w in convergence_warnings]}"
        )


@pytest.mark.unit
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
            "emission_design_info",
            "no_spike_part_log_likelihood",
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


@pytest.mark.unit
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
            environment=env,
            coefficients=encoding["coefficients"],
            emission_design_info=encoding["emission_design_info"],
            place_fields=encoding["place_fields"],
            no_spike_part_log_likelihood=encoding["no_spike_part_log_likelihood"],
            is_track_interior=encoding["is_track_interior"],
            is_local=False,
            disable_progress_bar=True,
        )

        # Assert
        n_place_bins = np.sum(env.is_track_interior_)
        assert log_likelihood.shape == (len(time), n_place_bins)
        assert jnp.all(jnp.isfinite(log_likelihood))

        with pytest.raises(ValidationError, match="population lengths do not match"):
            predict_sorted_spikes_glm_log_likelihood(
                time=jnp.asarray(time),
                position_time=jnp.asarray(data["position_time"]),
                position=jnp.asarray(data["position"]),
                spike_times=data["spike_times"][:-1],
                environment=env,
                coefficients=encoding["coefficients"],
                emission_design_info=encoding["emission_design_info"],
                place_fields=encoding["place_fields"],
                no_spike_part_log_likelihood=encoding["no_spike_part_log_likelihood"],
                is_track_interior=encoding["is_track_interior"],
                is_local=False,
                disable_progress_bar=True,
            )

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
            environment=env,
            coefficients=encoding["coefficients"],
            emission_design_info=encoding["emission_design_info"],
            place_fields=encoding["place_fields"],
            no_spike_part_log_likelihood=encoding["no_spike_part_log_likelihood"],
            is_track_interior=encoding["is_track_interior"],
            is_local=True,
            disable_progress_bar=True,
        )

        # Assert
        assert log_likelihood.shape == (len(time), 1)  # One position per time point
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
            environment=env,
            coefficients=encoding["coefficients"],
            emission_design_info=encoding["emission_design_info"],
            place_fields=encoding["place_fields"],
            no_spike_part_log_likelihood=encoding["no_spike_part_log_likelihood"],
            is_track_interior=encoding["is_track_interior"],
            is_local=False,
            disable_progress_bar=True,
        )

        # Assert - should still produce valid likelihoods (negative due to Poisson)
        assert jnp.all(jnp.isfinite(log_likelihood))
        assert jnp.all(log_likelihood < 0)  # Log likelihood should be negative


def _fit_glm(env, position_time, position, spike_times, weights=None):
    return fit_sorted_spikes_glm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        weights=weights,
        disable_progress_bar=True,
    )


def _spy_event_counts(monkeypatch):
    """Record the (event counts, exposure) pairs the GLM hands the optimizer."""
    from non_local_detector.likelihoods import sorted_spikes_glm

    captured = []

    def spy(design_matrix, spikes, weights, l2_penalty):
        captured.append((np.asarray(spikes), np.asarray(weights)))
        return fit_poisson_regression(design_matrix, spikes, weights, l2_penalty)

    monkeypatch.setattr(sorted_spikes_glm, "fit_poisson_regression", spy)
    return captured


@pytest.mark.unit
class TestWeightedEventOwnership:
    """The GLM event term splits each spike's interpolated weight between the
    two position samples that bracket it.

    A spike a fraction ``a`` of the way from sample ``i`` to ``i + 1`` adds
    ``(1 - a) * w_i`` to row ``i`` and ``a * w_{i+1}`` to row ``i + 1``. The two
    parts sum to the spike's interpolated weight, so group ownership is
    unchanged, and a row receives event mass only where its own exposure
    weight is positive. Putting the whole weight on the left row gives a row
    with zero exposure positive events at a 0 -> 1 mask transition, and the
    spline rate there diverges.
    """

    def test_constant_rate_mle_is_weighted_events_per_exposure(self):
        """With an intercept-only design the MLE is ``sum(c) / sum(w)``."""
        weighted_counts = np.array([1.0, 0.5, 0.0, 0.25, 2.0])
        exposure = np.array([1.0, 0.0, 0.5, 1.0, 0.75])
        coefficients = fit_poisson_regression(
            np.ones((5, 1)), weighted_counts, exposure, l2_penalty=0.0
        )
        np.testing.assert_allclose(
            np.exp(coefficients[0]),
            weighted_counts.sum() / exposure.sum(),
            rtol=1e-5,
        )

    def test_mask_transition_events_stay_on_exposed_rows(
        self, monkeypatch, simple_1d_environment
    ):
        """Mask ``[1, 1, 0, 1, 1]``. Spikes at 1.1 and 1.9 carry 0.9 and 0.1 and
        stay on row 1. The spike at 2.5 carries 0.5 and goes to row 3 (row 2 has
        no exposure), and the spike at 3.0 sits on row 3. The fit must stay
        finite: the left-row convention put 0.5 events on the unexposed row 2
        and the place field diverged."""
        captured = _spy_event_counts(monkeypatch)
        mask = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
        position_time = np.arange(5.0)
        position = np.linspace(10.0, 90.0, 5)[:, None]
        spike_times = [np.array([1.1, 1.9, 2.5, 3.0])]

        encoding = _fit_glm(
            simple_1d_environment, position_time, position, spike_times, mask
        )

        ((event_counts, exposure),) = captured
        np.testing.assert_allclose(event_counts, [0.0, 1.0, 0.0, 1.5, 0.0])
        np.testing.assert_array_equal(exposure, mask)
        assert np.all(np.isfinite(encoding["place_fields"]))
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            position_time, position_time, position, spike_times, **encoding
        )
        assert np.all(np.isfinite(log_likelihood))

    def test_event_mass_matches_interpolated_weights_and_exposure_support(
        self, monkeypatch, simple_1d_environment
    ):
        """For jittered samples, fractional weights, and a mask with gaps, each
        spike's interpolated weight is conserved and no unexposed row gets
        event mass."""
        captured = _spy_event_counts(monkeypatch)
        rng = np.random.default_rng(0)
        n = 200
        position_time = np.cumsum(rng.uniform(0.5, 1.5, n))
        position = rng.uniform(5.0, 95.0, (n, 1))
        weights = rng.uniform(0.0, 1.0, n) * ((np.arange(n) // 9) % 3 != 2)
        spike_times = np.sort(rng.uniform(position_time[0], position_time[-1], 400))

        encoding = _fit_glm(
            simple_1d_environment, position_time, position, [spike_times], weights
        )

        ((event_counts, _),) = captured
        np.testing.assert_allclose(
            event_counts.sum(),
            np.interp(spike_times, position_time, weights).sum(),
            rtol=1e-12,
        )
        np.testing.assert_array_equal(event_counts[weights == 0.0], 0.0)
        assert np.all(np.isfinite(encoding["place_fields"]))

    def test_unit_weights_split_each_spike_between_bracketing_samples(
        self, monkeypatch, simple_1d_environment
    ):
        """Without weights every in-range spike carries 1, split linearly
        between the samples on either side of it."""
        captured = _spy_event_counts(monkeypatch)
        position_time = np.arange(5.0)
        spike_times = [np.array([-0.5, 0.25, 1.0, 3.75, 4.0, 4.5])]

        _fit_glm(
            simple_1d_environment,
            position_time,
            np.linspace(10.0, 90.0, 5)[:, None],
            spike_times,
        )

        ((event_counts, _),) = captured
        np.testing.assert_allclose(event_counts, [0.75, 1.25, 0.0, 0.25, 1.75])
