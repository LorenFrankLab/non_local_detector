"""Tests for sorted spikes GLM likelihood model.

Tests the Poisson GLM implementation for sorted spike data, including spline
basis generation, model fitting, and likelihood prediction.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import sorted_spikes_glm
from non_local_detector.likelihoods.common import EPS
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_poisson_regression,
    fit_poisson_regressions,
    fit_sorted_spikes_glm_encoding_model,
    make_spline_design_matrix,
    make_spline_predict_matrix,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.tests.likelihoods._sorted_spikes_helpers import (
    dot_general_precisions,
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
        """A spatially uniform count gives an intercept-only fit at that rate."""
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
        # Constant counts of 5 per unit exposure, with the penalty only on the
        # non-intercept terms: the optimum is exactly beta = (log 5, 0, ..., 0).
        coefficients = np.asarray(coefficients)
        np.testing.assert_allclose(coefficients[1:], 0.0, atol=1e-3)
        np.testing.assert_allclose(np.exp(coefficients[0]), 5.0, rtol=1e-4)

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

    def test_glm_unconverged_fit_warned(self, monkeypatch):
        """A fit stopped with a large gradient emits a ``UserWarning``."""
        rng = np.random.default_rng(0)
        n_time, n_basis = 50, 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        spikes = rng.poisson(3, size=n_time)
        weights = np.ones(n_time)
        monkeypatch.setattr(sorted_spikes_glm, "GLM_MAX_NEWTON_STEPS", 0)

        with pytest.warns(UserWarning, match="may not have converged"):
            fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=1e-3)

    def test_glm_nonfinite_fit_warned(self):
        """A non-finite loss or gradient must warn, not pass silently.

        ``max(|gradient|)`` is ``NaN`` for such a fit and ``NaN > tol`` is
        ``False``, so a gradient-only check would accept it.
        """
        rng = np.random.default_rng(0)
        n_time, n_basis = 50, 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        design_matrix[3, 2] = np.nan
        spikes = rng.poisson(3, size=n_time)
        weights = np.ones(n_time)

        with pytest.warns(UserWarning, match="may not have converged"):
            fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=1e-3)

    def test_glm_converged_fit_does_not_warn(self):
        rng = np.random.default_rng(0)
        n_time, n_basis = 50, 5
        design_matrix = rng.standard_normal((n_time, n_basis))
        design_matrix[:, 0] = 1.0
        spikes = rng.poisson(3, size=n_time)
        weights = np.ones(n_time)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit_poisson_regression(design_matrix, spikes, weights, l2_penalty=1e-3)


def _poisson_objective64(design_matrix, spikes, weights, coefficients, l2_penalty):
    """The fitted objective in float64, per cell: the penalized negative total
    log likelihood, divided by the total exposure."""
    rate = np.maximum(np.exp(coefficients @ design_matrix.T), EPS)
    data = np.where(spikes > 0, spikes * np.log(rate), 0.0) - weights * rate
    penalty = l2_penalty * np.sum(coefficients[:, 1:] ** 2, axis=1)
    return (-data.sum(axis=1) + penalty) / weights.sum()


@pytest.mark.unit
class TestBatchedPoissonRegression:
    """All cells share the design matrix and exposure; one solve fits them all."""

    @pytest.fixture
    def problem(self):
        rng = np.random.default_rng(3)
        n_time, n_basis = 400, 8
        position = np.linspace(-1.0, 1.0, n_time)
        design_matrix = np.column_stack(
            [np.ones(n_time)]
            + [
                np.exp(-0.5 * ((position - c) / 0.2) ** 2)
                for c in np.linspace(-1, 1, 7)
            ]
        )
        weights = rng.uniform(0.2, 1.0, n_time)
        weights[rng.uniform(size=n_time) < 0.2] = 0.0
        true = rng.normal(0.0, 1.0, (6, n_basis))
        true[:, 0] = np.log(rng.uniform(0.5, 5.0, 6))
        spikes = rng.poisson(weights * np.exp(true @ design_matrix.T)).astype(float)
        spikes[0] = 0.0  # a silent cell
        return design_matrix, spikes, weights

    def test_batched_fit_reaches_the_float64_optimum(self, problem):
        from scipy.optimize import minimize

        design_matrix, spikes, weights = problem
        # 1e-3 per second of exposure: the conditioning these tests were
        # written for, now that the penalty is on the total log likelihood.
        l2_penalty = 1e-3 * weights.sum()
        actual = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, l2_penalty)
        )
        assert actual.shape == (spikes.shape[0], design_matrix.shape[1])

        def objective(beta, y):
            return _poisson_objective64(
                design_matrix, y[None], weights, beta[None], l2_penalty
            )[0]

        for cell, y in enumerate(spikes[1:], start=1):
            x0 = np.zeros(design_matrix.shape[1])
            x0[0] = np.log(y.sum() / weights.sum())
            reference = minimize(
                objective, x0, args=(y,), method="BFGS", options={"gtol": 1e-10}
            ).x
            np.testing.assert_allclose(actual[cell], reference, rtol=1e-3, atol=1e-3)
            assert objective(actual[cell], y) <= objective(reference, y) + 1e-6

    def test_batched_rows_match_single_cell_fits(self, problem):
        """Batching all cells reaches the single-cell optimum.

        The batched and single-cell solves sum in different float32 orders on
        some CPUs, so their Newton steps can stop at slightly different points
        of a flat optimum (coefficients differed by up to 1.8e-4 on a Linux CI
        runner and are identical on macOS); the float64 objective is what must
        agree. A 2e-4 coefficient perturbation changes it by about 1.7e-7.
        """
        design_matrix, spikes, weights = problem
        # 1e-3 per second of exposure: the conditioning these tests were
        # written for, now that the penalty is on the total log likelihood.
        l2_penalty = 1e-3 * weights.sum()
        batched = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, l2_penalty)
        )
        single = np.stack(
            [
                np.asarray(
                    fit_poisson_regression(design_matrix, y, weights, l2_penalty)
                )
                for y in spikes
            ]
        )
        np.testing.assert_allclose(
            _poisson_objective64(design_matrix, spikes, weights, batched, l2_penalty),
            _poisson_objective64(design_matrix, spikes, weights, single, l2_penalty),
            rtol=0,
            atol=1e-8,
        )

    def test_row_blocks_and_padding_do_not_change_the_fit(self, problem, monkeypatch):
        """Long recordings sum over many row blocks, the last one padded.

        Blocks change the float32 summation order, so the Newton steps stop at
        slightly different points of a flat optimum (coefficients agree to
        about 1e-3); the float64 objective is what must agree.
        """
        design_matrix, spikes, weights = problem
        # 1e-3 per second of exposure: the conditioning these tests were
        # written for, now that the penalty is on the total log likelihood.
        l2_penalty = 1e-3 * weights.sum()
        whole = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, l2_penalty)
        )
        n_cells, n_basis = spikes.shape[0], design_matrix.shape[1]
        # 37 rows per block: 400 rows need 11 blocks and 7 padding rows.
        monkeypatch.setattr(
            sorted_spikes_glm, "_GLM_BLOCK_BYTES", 37 * 4 * n_cells * n_basis
        )
        sorted_spikes_glm._poisson_regression_sums.clear_cache()
        blocked = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, l2_penalty)
        )
        assert np.all(np.isfinite(blocked))
        np.testing.assert_allclose(
            _poisson_objective64(design_matrix, spikes, weights, blocked, l2_penalty),
            _poisson_objective64(design_matrix, spikes, weights, whole, l2_penalty),
            rtol=0,
            atol=1e-8,
        )

    def test_penalty_does_not_grow_with_recording_length(self, problem):
        """The penalty is on the total log likelihood, so a recording repeated
        ``k`` times fits like the original with the penalty divided by ``k``.
        (A penalty on the per-second average would instead give every
        repetition the same fit, letting the prior grow with the data.)"""
        design_matrix, spikes, weights = problem
        k, l2_penalty = 4, 2.0
        repeated = np.asarray(
            fit_poisson_regressions(
                np.tile(design_matrix, (k, 1)),
                np.tile(spikes, (1, k)),
                np.tile(weights, k),
                l2_penalty,
            )
        )
        once = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, l2_penalty / k)
        )
        np.testing.assert_allclose(repeated, once, rtol=0, atol=2e-3)
        unscaled = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, l2_penalty)
        )
        assert np.abs(unscaled - once).max() > 0.05

    def test_unsupported_basis_function_stays_at_zero(self, problem):
        """A basis function with no weighted support is set only by the penalty."""
        design_matrix, spikes, weights = problem
        design_matrix = np.column_stack(
            [design_matrix, np.where(weights == 0.0, 1.0, 0.0)]
        )
        coefficients = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, 1e-3)
        )
        np.testing.assert_allclose(coefficients[:, -1], 0.0, atol=1e-6)

    def test_singular_newton_system_warns_and_reaches_the_optimum(self, problem):
        """Without a penalty, an unsupported basis function makes every
        Hessian exactly singular. Least squares still reaches the optimum of
        the identified coefficients, and the fit warns once naming the cells.
        """
        design_matrix, spikes, weights = problem
        unsupported = np.column_stack(
            [design_matrix, np.where(weights == 0.0, 1.0, 0.0)]
        )
        reference = np.asarray(
            fit_poisson_regressions(design_matrix, spikes, weights, 0.0)
        )
        with pytest.warns(UserWarning, match="singular") as record:
            coefficients = np.asarray(
                fit_poisson_regressions(unsupported, spikes, weights, 0.0)
            )
        singular = [w for w in record if "singular" in str(w.message)]
        assert len(singular) == 1
        assert "[1, 2, 3, 4, 5]" in str(singular[0].message)
        np.testing.assert_array_equal(coefficients[:, -1], 0.0)
        # Least-squares directions converge less tightly than Newton steps:
        # objectives on Linux CI were 4e-8 from the Newton fit's.
        np.testing.assert_allclose(
            _poisson_objective64(
                design_matrix, spikes, weights, coefficients[:, :-1], 0.0
            ),
            _poisson_objective64(design_matrix, spikes, weights, reference, 0.0),
            rtol=0,
            atol=1e-7,
        )


def _fit_encoding(env, data, **kwargs):
    return fit_sorted_spikes_glm_encoding_model(
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        disable_progress_bar=True,
        **kwargs,
    )


def _spy_l2_penalty(monkeypatch):
    """Record the ``l2_penalty`` the encoding fit hands the optimizer."""
    penalties = []

    def spy(design_matrix, spikes, weights, l2_penalty, **kwargs):
        penalties.append(l2_penalty)
        return fit_poisson_regressions(
            design_matrix, spikes, weights, l2_penalty, **kwargs
        )

    monkeypatch.setattr(sorted_spikes_glm, "fit_poisson_regressions", spy)
    return penalties


@pytest.mark.unit
class TestLogRatePrior:
    """By default the coefficient penalty is a Gaussian prior on log-rate."""

    @pytest.mark.parametrize(
        ("kwargs", "prior_std"),
        [
            ({}, sorted_spikes_glm.GLM_LOG_RATE_PRIOR_STD),
            ({"log_rate_prior_std": 1.0}, 1.0),
        ],
    )
    @pytest.mark.parametrize("knot_spacing", [5.0, 20.0])
    def test_prior_sets_the_log_rate_sd_at_the_bins(
        self,
        monkeypatch,
        simple_1d_environment,
        simple_spike_data,
        kwargs,
        prior_std,
        knot_spacing,
    ):
        """Under ``Normal(0, 1 / (2 l2_penalty))`` coefficients, the spline's
        log-rate deviation from baseline has the prior variance on average
        over the interior bins, whatever the knot spacing."""
        penalties = _spy_l2_penalty(monkeypatch)
        env = simple_1d_environment
        encoding = _fit_encoding(
            env, simple_spike_data, emission_knot_spacing=knot_spacing, **kwargs
        )
        centers = env.place_bin_centers_[env.is_track_interior_.ravel()]
        spline = np.asarray(
            make_spline_predict_matrix(encoding["emission_design_info"], centers)
        )[:, 1:]
        (l2_penalty,) = penalties
        coefficient_variance = 1.0 / (2.0 * l2_penalty)
        implied_variance = coefficient_variance * np.sum(spline**2, axis=1)
        np.testing.assert_allclose(implied_variance.mean(), prior_std**2, rtol=1e-6)

    def test_intercept_only_basis_has_no_penalty(self):
        assert sorted_spikes_glm.log_rate_prior_l2_penalty(np.ones((5, 1))) == 0.0

    def test_explicit_l2_penalty_overrides_the_prior(
        self, monkeypatch, simple_1d_environment, simple_spike_data
    ):
        penalties = _spy_l2_penalty(monkeypatch)
        _fit_encoding(simple_1d_environment, simple_spike_data, l2_penalty=0.3)
        assert penalties == [0.3]

    @pytest.mark.parametrize("prior_std", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_prior_std_raises(
        self, simple_1d_environment, simple_spike_data, prior_std
    ):
        with pytest.raises(ValidationError, match="log_rate_prior_std"):
            _fit_encoding(
                simple_1d_environment,
                simple_spike_data,
                log_rate_prior_std=prior_std,
            )


def test_encoding_fit_solves_all_cells_in_one_call(
    monkeypatch, simple_1d_environment, simple_spike_data
):
    calls = []

    def spy(design_matrix, spikes, weights, l2_penalty, **kwargs):
        calls.append(np.shape(spikes))
        return fit_poisson_regressions(
            design_matrix, spikes, weights, l2_penalty, **kwargs
        )

    monkeypatch.setattr(sorted_spikes_glm, "fit_poisson_regressions", spy)
    env, data = simple_1d_environment, simple_spike_data
    encoding = _fit_encoding(env, data)
    n_cells = len(data["spike_times"])
    assert len(calls) == 1 and calls[0][0] == n_cells
    assert encoding["coefficients"].shape[0] == n_cells


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
        encoding = _fit_encoding(env, data)

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
        encoding = _fit_encoding(env, data)

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
        encoding_coarse = _fit_encoding(env, data, emission_knot_spacing=30.0)
        encoding_fine = _fit_encoding(env, data, emission_knot_spacing=10.0)

        # Assert - finer spacing puts more knots on the track: more coefficients
        n_coef_coarse = encoding_coarse["coefficients"][0].shape[0]
        n_coef_fine = encoding_fine["coefficients"][0].shape[0]
        assert n_coef_fine > n_coef_coarse


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

        encoding = _fit_encoding(env, data)

        # Create decoding time window
        time = np.linspace(0, 0.5, 10)

        # Act
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            time_edges=jnp.asarray(time_edges_from_centers(time)),
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
                time_edges=jnp.asarray(time_edges_from_centers(time)),
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

        encoding = _fit_encoding(env, data)

        time = np.linspace(0, 0.5, 10)

        # Act
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            time_edges=jnp.asarray(time_edges_from_centers(time)),
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
        """With no spikes, the non-local LL is -duration * the summed place fields."""
        # Arrange
        env = simple_1d_environment
        data = simple_spike_data

        encoding = _fit_encoding(env, data)

        # Use time period with no spikes (well beyond data)
        time = np.linspace(10.0, 10.5, 10)
        time_edges = jnp.asarray(time_edges_from_centers(time))

        # Act
        log_likelihood = predict_sorted_spikes_glm_log_likelihood(
            time_edges=time_edges,
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

        # Assert - with no spikes each bin's LL is -duration * summed rate
        durations = np.diff(np.asarray(time_edges, dtype=np.float64))
        interior = np.asarray(encoding["is_track_interior"])
        expected = (
            -durations[:, None]
            * np.asarray(encoding["no_spike_part_log_likelihood"])[interior]
        )
        assert np.all(expected < 0)
        np.testing.assert_allclose(log_likelihood, expected, rtol=1e-5, atol=1e-6)


def _fit_glm(
    env,
    position_time,
    position,
    spike_times,
    weights=None,
    *,
    valid_position_intervals=None,
):
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
        valid_position_intervals=valid_position_intervals,
        disable_progress_bar=True,
    )


def _spy_event_counts(monkeypatch):
    """Record the per-sample (event counts, exposure) pairs of the GLM fit,
    before they are summed over position cells for the optimizer."""
    captured = []
    aggregate = sorted_spikes_glm._aggregate_by_position_cell

    def spy(position, counts, exposure, *args, **kwargs):
        captured.extend((np.asarray(row), np.asarray(exposure)) for row in counts)
        return aggregate(position, counts, exposure, *args, **kwargs)

    monkeypatch.setattr(sorted_spikes_glm, "_aggregate_by_position_cell", spy)
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
            position_time, position, spike_times, time_edges=position_time, **encoding
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
            simple_1d_environment,
            position_time,
            position,
            [spike_times],
            weights,
            valid_position_intervals=[[position_time[0], position_time[-1]]],
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
        """Interior events split linearly; endpoint half-cell events belong
        entirely to their nearest endpoint sample."""
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
        np.testing.assert_allclose(event_counts, [1.75, 1.25, 0.0, 0.25, 2.75])


def _corner_track_data(environment_name="corner", seed=0):
    """An L-shaped track (both coordinates vary) and two place cells along it."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_node(2, pos=(10.0, 8.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    graph.add_edge(1, 2, distance=8.0, edge_id=1)
    env = Environment(
        environment_name=environment_name,
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
    ).fit_place_grid()

    rng = np.random.default_rng(seed)
    n_time, sampling_frequency = 6000, 100
    time = np.arange(n_time) / sampling_frequency
    along = 0.2 + np.abs((np.cumsum(rng.normal(0, 0.3, n_time)) % 35.2) - 17.6)
    position = np.where(
        along[:, None] < 10.0,
        np.column_stack([along, np.zeros(n_time)]),
        np.column_stack([np.full(n_time, 10.0), along - 10.0]),
    )
    place_centers = (3.0, 14.0)
    spike_times = [
        time[
            rng.random(n_time)
            < 40.0 * np.exp(-((along - c) ** 2) / 2.0) / sampling_frequency
        ]
        for c in place_centers
    ]
    return env, time, position, spike_times, place_centers


@pytest.mark.unit
def test_track_graph_fit_linearizes_two_dimensional_position():
    """On a track graph, 2-D positions are linearized before the spline basis.

    The environment's bin edges are 1-D along the track. The fit built the
    spline from raw (x, y) and indexed a second column of the 1-D edges, so EM
    with the GLM could not run on a real linearized track.
    """
    env, time, position, spike_times, _ = _corner_track_data()
    linear = env._linear_position(position)[:, None]

    def fit(positions):
        return fit_sorted_spikes_glm_encoding_model(
            position_time=time,
            position=positions,
            spike_times=spike_times,
            environment=env,
            place_bin_edges=env.place_bin_edges_,
            edges=env.edges_,
            is_track_interior=env.is_track_interior_,
            is_track_boundary=env.is_track_boundary_,
            emission_knot_spacing=2.0,
            disable_progress_bar=True,
        )

    from_xy = fit(position)
    from_linear = fit(linear)

    np.testing.assert_allclose(
        from_xy["place_fields"], from_linear["place_fields"], rtol=1e-6
    )
    np.testing.assert_allclose(
        from_xy["coefficients"], from_linear["coefficients"], rtol=1e-6, atol=1e-8
    )


@pytest.mark.integration
def test_detector_fits_glm_on_a_track_graph():
    """The detector's GLM fit runs with a track graph and 2-D positions."""
    from non_local_detector import NonLocalSortedSpikesDetector

    env, time, position, spike_times, place_centers = _corner_track_data(
        environment_name=""
    )
    detector = NonLocalSortedSpikesDetector(
        environments=env,
        sorted_spikes_algorithm="sorted_spikes_glm",
        sorted_spikes_algorithm_params={
            "emission_knot_spacing": 2.0,
            "disable_progress_bar": True,
        },
    ).fit(position_time=time, position=position, spike_times=spike_times)

    encoding = next(iter(detector.encoding_model_.values()))
    place_fields = np.asarray(encoding["place_fields"])
    linear_centers = detector.environments[0].place_bin_centers_.ravel()
    assert np.all(np.isfinite(place_fields))
    for center, field in zip(place_centers, place_fields, strict=True):
        assert abs(linear_centers[np.argmax(field)] - center) < 2.0


@pytest.mark.unit
def test_place_field_rates_request_highest_precision():
    """Non-local place fields use the local likelihood's full-precision product.

    Without HIGHEST, CUDA may evaluate the float32 product in TF32, so stored
    non-local rates could disagree with the local rates at the same position.
    CPU always computes full float32, so this checks the request in the
    traced program, under the default matmul precision so an ambient setting
    cannot satisfy it.
    """
    import jax

    design = jnp.ones((7, 4), jnp.float32)
    coefficients = jnp.ones((3, 4), jnp.float32)
    with jax.default_matmul_precision("default"):
        traced = jax.make_jaxpr(sorted_spikes_glm._glm_rates)(design, coefficients)
    precisions = dot_general_precisions(traced.jaxpr)
    assert precisions
    assert all(
        precision == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
        for precision in precisions
    )
    np.testing.assert_allclose(
        sorted_spikes_glm._glm_rates(design, coefficients),
        np.full((7, 3), np.exp(4.0)),
        rtol=1e-6,
    )
