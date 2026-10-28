"""Tests for smoothing and uncertainty of covariate-dependent transitions.

Covers the ``quantile_bs`` boundary quantiles, the periodic ``cyclic_bs``
basis, the curvature (smoothness) penalty on each spline term, REML selection
of each term's strength, and the Laplace covariance and intervals of the
fitted transition curves.
"""

import warnings

import numpy as np
import pytest
from patsy import PatsyError, dmatrix
from scipy.special import softmax

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    _fit_covariate_transition,
    _max_transition_change,
    _smoothness_matrix,
    _transition_log_prior,
    cyclic_bs,  # noqa: F401  (used inside patsy formulas)
    dirichlet_gradient,
    dirichlet_hessian,
    dirichlet_neg_log_likelihood,
    estimate_non_stationary_state_transition_from_responses,
    predict_discrete_state_transition_intervals,
    predict_discrete_state_transitions,
    quantile_bs,  # noqa: F401  (used inside patsy formulas)
)

N_STATES = 3


def _speeds(rng: np.random.Generator, n_time: int) -> np.ndarray:
    """Speed-like covariate: a block of exact zeros and a right-skewed tail."""
    speed = rng.gamma(1.5, 12.0, n_time)
    speed[rng.uniform(size=n_time) < 0.2] = 0.0
    return speed


def _true_transition(speed: np.ndarray) -> np.ndarray:
    """Smooth speed-dependent transitions, shape (n_time, n_states, n_states)."""
    x = speed / 40.0
    logits = np.zeros((speed.size, N_STATES, N_STATES))
    logits[:, 0, 0] = 4.0 + 1.5 * np.tanh(2.0 * (x - 0.5))
    logits[:, 1, 1] = 3.0 - 1.0 * x
    logits[:, 2, 2] = 2.0 + 0.5 * np.sin(np.pi * x)
    logits[:, 0, 1] = 0.5 * x
    return softmax(logits, axis=-1)


def _sampled_responses(rng: np.random.Generator, transition: np.ndarray) -> np.ndarray:
    """One categorical draw per row and time bin, shape (n_time - 1, n_states, n_states)."""
    cumulative = np.cumsum(transition[:-1], axis=-1)
    draws = rng.uniform(size=cumulative.shape[:2] + (1,))
    next_state = np.minimum((draws > cumulative).sum(axis=-1), N_STATES - 1)
    return np.eye(N_STATES)[next_state]


def _design(speed: np.ndarray, formula: str = "1 + quantile_bs(speed, n_knots=8)"):
    return dmatrix(formula, {"speed": speed}, NA_action="raise")


@pytest.mark.unit
class TestQuantileBSplineBoundary:
    def test_upper_quantile_clips_the_tail_to_the_boundary(self):
        rng = np.random.default_rng(0)
        speed = _speeds(rng, 5_000)
        design = _design(speed, "1 + quantile_bs(speed, upper_quantile=0.99)")
        boundary = np.quantile(speed, 0.99)

        new = {"speed": np.array([boundary, boundary + 5.0, speed.max() + 50.0])}
        rows = np.asarray(
            dmatrix(design.design_info, new, NA_action="raise"), dtype=float
        )

        np.testing.assert_allclose(rows[1], rows[0])
        np.testing.assert_allclose(rows[2], rows[0])

    def test_interior_knots_lie_inside_the_quantile_boundaries(self):
        rng = np.random.default_rng(1)
        speed = _speeds(rng, 5_000)
        design = _design(
            speed,
            "1 + quantile_bs(speed, n_knots=6, lower_quantile=0.05, "
            "upper_quantile=0.95)",
        )
        spline = _quantile_bs_transform(design)
        lower, upper = np.quantile(speed, [0.05, 0.95])

        assert spline._lower == pytest.approx(lower)
        assert spline._upper == pytest.approx(upper)
        interior = spline._interior_knots
        assert interior.size == 6
        assert np.all((interior > lower) & (interior < upper))

    def test_default_boundaries_are_the_data_range(self):
        rng = np.random.default_rng(2)
        speed = _speeds(rng, 2_000)
        spline = _quantile_bs_transform(_design(speed, "1 + quantile_bs(speed)"))

        assert spline._lower == speed.min()
        assert spline._upper == speed.max()

    @pytest.mark.parametrize(
        "kwargs", ["lower_quantile=0.5, upper_quantile=0.5", "upper_quantile=1.5"]
    )
    def test_invalid_quantiles_raise(self, kwargs):
        speed = _speeds(np.random.default_rng(3), 500)
        # patsy re-raises the transform's ValueError as a PatsyError.
        with pytest.raises(PatsyError, match="quantile"):
            _design(speed, f"1 + quantile_bs(speed, {kwargs})")


def _quantile_bs_transform(design):
    """The fitted ``_QuantileBSpline`` of a one-``quantile_bs`` design."""
    (factor_info,) = [
        info
        for factor, info in design.design_info.factor_infos.items()
        if "quantile_bs" in factor.code
    ]
    (transform,) = factor_info.state["transforms"].values()
    return transform


@pytest.mark.unit
class TestCurvaturePenalty:
    """``_smoothness_matrix`` is the integrated squared second derivative of
    each ``quantile_bs`` term, with the covariate rescaled to [0, 1] over the
    spline's boundaries."""

    def test_penalty_is_the_integrated_squared_second_derivative(self):
        """A cubic spline reproduces u**2 exactly, whose curvature integral
        over [0, 1] is 4; linear functions cost nothing."""
        rng = np.random.default_rng(4)
        speed = _speeds(rng, 3_000)
        design = _design(speed)
        spline = _quantile_bs_transform(design)
        grid = np.linspace(spline._lower, spline._upper, 2_001)
        grid_design = np.asarray(
            dmatrix(design.design_info, {"speed": grid}), dtype=float
        )
        u = (grid - spline._lower) / (spline._upper - spline._lower)
        (penalty,) = _smoothness_matrix(design.design_info)

        for target, expected in ((u**2, 4.0), (3.0 * u - 1.0, 0.0), (u**3, 12.0)):
            coefficients, *_ = np.linalg.lstsq(grid_design, target, rcond=None)
            np.testing.assert_allclose(grid_design @ coefficients, target, atol=1e-8)
            assert coefficients @ penalty @ coefficients == pytest.approx(
                expected, abs=1e-6
            )

    def test_penalty_sits_on_the_spline_columns_only(self):
        rng = np.random.default_rng(5)
        speed = _speeds(rng, 2_000)
        design = _design(speed, "1 + np.log1p(speed) + quantile_bs(speed, n_knots=4)")
        (penalty,) = _smoothness_matrix(design.design_info)
        spline_columns = design.design_info.term_name_slices[
            "quantile_bs(speed, n_knots=4)"
        ]

        assert penalty.shape == (design.shape[1],) * 2
        np.testing.assert_allclose(penalty, penalty.T)
        assert np.all(np.linalg.eigvalsh(penalty) > -1e-10)
        outside = np.ones(design.shape[1], dtype=bool)
        outside[spline_columns] = False
        assert np.all(penalty[outside] == 0.0)
        assert np.all(penalty[:, outside] == 0.0)
        assert np.any(penalty[spline_columns, spline_columns] != 0.0)

    def test_formula_without_quantile_bs_has_no_penalty(self):
        speed = _speeds(np.random.default_rng(6), 200)
        design = _design(speed, "1 + speed")
        assert _smoothness_matrix(design.design_info).shape == (0, 2, 2)


PERIOD = 2.0 * np.pi
CYCLIC = f"cyclic_bs(phase, lower=0.0, upper={PERIOD!r}, n_knots=8)"


def _cyclic_design(phase: np.ndarray, term: str = CYCLIC):
    return dmatrix(f"1 + {term}", {"phase": phase}, NA_action="raise")


@pytest.mark.unit
class TestCyclicBSpline:
    """``cyclic_bs`` is a periodic cubic B-spline over ``[lower, upper)``
    with knots at quantiles of the fitted values."""

    def test_curve_is_periodic_with_continuous_curvature(self):
        """Values outside the period wrap, and a curve and its first two
        derivatives join smoothly across ``lower`` / ``upper``."""
        rng = np.random.default_rng(30)
        design = _cyclic_design(rng.uniform(0, PERIOD, 2_000))
        coefficients = rng.normal(0.0, 1.0, design.shape[1])

        def curve(x):
            return np.asarray(dmatrix(design.design_info, {"phase": x})) @ coefficients

        x = np.linspace(0.0, PERIOD, 50, endpoint=False)
        np.testing.assert_allclose(curve(x + PERIOD), curve(x), atol=1e-12)
        np.testing.assert_allclose(curve(x - PERIOD), curve(x), atol=1e-12)

        step = 1e-3
        across = np.arange(-300, 301) * step  # spans the wrap at 0
        second = np.gradient(np.gradient(curve(across), step), step)[2:-2]
        assert np.max(np.abs(np.diff(second))) < 0.05 * np.max(np.abs(second))

    def test_penalty_is_the_integrated_squared_second_derivative(self):
        """sin(2 pi u) on u in [0, 1) has curvature integral (2 pi)^4 / 2;
        a constant (the intercept) costs nothing."""
        rng = np.random.default_rng(31)
        design = _cyclic_design(
            rng.uniform(0, PERIOD, 5_000),
            f"cyclic_bs(phase, lower=0.0, upper={PERIOD!r}, n_knots=30)",
        )
        grid = np.linspace(0.0, PERIOD, 4_001)
        grid_design = np.asarray(dmatrix(design.design_info, {"phase": grid}))
        (penalty,) = _smoothness_matrix(design.design_info)

        coefficients, *_ = np.linalg.lstsq(grid_design, np.sin(grid) + 2.0, rcond=None)
        np.testing.assert_allclose(
            grid_design @ coefficients, np.sin(grid) + 2.0, atol=1e-4
        )
        assert coefficients @ penalty @ coefficients == pytest.approx(
            (2 * np.pi) ** 4 / 2, rel=1e-3
        )
        assert penalty[0] == pytest.approx(np.zeros(design.shape[1]))

    def test_columns_and_intercept_are_full_rank(self):
        design = _cyclic_design(np.random.default_rng(32).uniform(0, PERIOD, 1_000))
        assert np.linalg.matrix_rank(np.asarray(design)) == design.shape[1]

    @pytest.mark.parametrize(
        "kwargs",
        ["lower=1.0, upper=1.0", "lower=0.0, upper=6.0, n_knots=3"],
    )
    def test_invalid_arguments_raise(self, kwargs):
        phase = np.random.default_rng(33).uniform(0, 6, 500)
        with pytest.raises(PatsyError, match="cyclic_bs"):
            _cyclic_design(phase, f"cyclic_bs(phase, {kwargs})")

    def test_too_few_distinct_values_raise(self):
        with pytest.raises(ValueError, match="distinct"):
            _cyclic_design(np.repeat([0.0, 1.0, 2.0], 50))


def _two_term_problem(seed: int, n_time: int = 8_000):
    """Responses whose logits are linear in speed and wiggly in phase."""
    rng = np.random.default_rng(seed)
    speed = _speeds(rng, n_time)
    phase = rng.uniform(0, PERIOD, n_time)
    design = dmatrix(
        f"1 + quantile_bs(speed, n_knots=6) + {CYCLIC}",
        {"speed": speed, "phase": phase},
        NA_action="raise",
    )
    logits = np.zeros((n_time, N_STATES, N_STATES))
    logits[:, np.arange(N_STATES), np.arange(N_STATES)] = 2.5
    logits[:, :, 0] += (speed / 40.0)[:, None] + 1.2 * np.sin(2.0 * phase)[:, None]
    return design, _sampled_responses(rng, softmax(logits, axis=-1))


@pytest.mark.unit
class TestPerTermSmoothness:
    def test_each_spline_term_gets_its_own_penalty(self):
        design, _ = _two_term_problem(seed=34, n_time=2_000)
        penalties = _smoothness_matrix(design.design_info)
        slices = list(design.design_info.term_slices.values())[1:]

        assert penalties.shape == (2, design.shape[1], design.shape[1])
        for penalty, columns in zip(penalties, slices, strict=True):
            outside = np.ones(design.shape[1], dtype=bool)
            outside[columns] = False
            assert np.all(penalty[outside] == 0.0)
            assert np.all(penalty[:, outside] == 0.0)
            assert np.any(penalty[columns, columns] != 0.0)

    def test_reml_smooths_the_linear_term_more_than_the_wiggly_one(self):
        """Each row's strengths follow its own terms: a linear effect of
        speed is smoothed far more than a wiggly effect of phase, and the
        per-term strengths score at least as well as one shared strength."""
        design, response = _two_term_problem(seed=35)
        design_matrix = np.asarray(design, dtype=float)
        penalties = _smoothness_matrix(design.design_info)
        start = np.zeros((design.shape[1], N_STATES, N_STATES - 1))

        fit = _fit_covariate_transition(
            start,
            design_matrix,
            response,
            smoothness="reml",
            smoothness_matrix=penalties,
        )

        assert fit.smoothness.shape == (N_STATES, 2)
        assert np.all(fit.smoothness[:, 0] > 10.0 * fit.smoothness[:, 1])
        for row in range(N_STATES):
            per_term = _laml(
                design_matrix[:-1],
                response[:, row],
                fit.coefficients[:, row],
                fit.smoothness[row],
                penalties,
                0.1,
            )
            for shared in fit.smoothness[row]:
                shared_fit = _fit_covariate_transition(
                    start,
                    design_matrix,
                    response,
                    smoothness=shared,
                    smoothness_matrix=penalties,
                )
                assert (
                    per_term
                    >= _laml(
                        design_matrix[:-1],
                        response[:, row],
                        shared_fit.coefficients[:, row],
                        shared,
                        penalties,
                        0.1,
                    )
                    - 1e-6
                )

    def test_fixed_smoothness_applies_to_every_term(self):
        design, response = _two_term_problem(seed=36, n_time=1_000)
        fit = _fit_covariate_transition(
            np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
            np.asarray(design, dtype=float),
            response,
            smoothness=0.3,
            smoothness_matrix=_smoothness_matrix(design.design_info),
        )
        np.testing.assert_array_equal(fit.smoothness, np.full((N_STATES, 2), 0.3))


def _row_problem(seed: int = 7, n_time: int = 6_000):
    rng = np.random.default_rng(seed)
    speed = _speeds(rng, n_time)
    design = _design(speed)
    truth = _true_transition(speed)
    response = _sampled_responses(rng, truth)
    return rng, speed, design, truth, response


@pytest.mark.unit
class TestSmoothedRowObjective:
    def test_gradient_and_hessian_match_finite_differences(self):
        rng, _, design, _, response = _row_problem(n_time=400)
        design_matrix = np.asarray(design, dtype=float)
        penalty = _smoothness_matrix(design.design_info)
        coefficients = rng.normal(0.0, 0.3, design.shape[1] * (N_STATES - 1))
        kwargs = {
            "alpha": np.array([5.0, 1.0, 1.0]),
            "l2_penalty": 0.05,
            "smoothness": 0.7,
            "smoothness_matrix": penalty,
        }
        args = (design_matrix[:-1], response[:, 0])

        gradient = dirichlet_gradient(coefficients, *args, **kwargs)
        hessian = dirichlet_hessian(coefficients, *args, **kwargs)
        step = 1e-5
        for index in range(coefficients.size):
            shift = np.zeros_like(coefficients)
            shift[index] = step
            loss_up = dirichlet_neg_log_likelihood(
                coefficients + shift, *args, **kwargs
            )
            loss_down = dirichlet_neg_log_likelihood(
                coefficients - shift, *args, **kwargs
            )
            assert gradient[index] == pytest.approx(
                (loss_up - loss_down) / (2 * step), rel=1e-5, abs=1e-6
            )
            np.testing.assert_allclose(
                hessian[index],
                (
                    dirichlet_gradient(coefficients + shift, *args, **kwargs)
                    - dirichlet_gradient(coefficients - shift, *args, **kwargs)
                )
                / (2 * step),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_smoothness_adds_its_quadratic_form_per_logit(self):
        rng, _, design, _, response = _row_problem(n_time=300)
        design_matrix = np.asarray(design, dtype=float)
        (penalty,) = _smoothness_matrix(design.design_info)
        coefficients = rng.normal(0.0, 0.3, (design.shape[1], N_STATES - 1))
        args = (coefficients.ravel(), design_matrix[:-1], response[:, 0])

        unsmoothed = dirichlet_neg_log_likelihood(*args, l2_penalty=0.0)
        smoothed = dirichlet_neg_log_likelihood(
            *args, l2_penalty=0.0, smoothness=2.5, smoothness_matrix=penalty
        )

        assert smoothed - unsmoothed == pytest.approx(
            2.5 * np.trace(coefficients.T @ penalty @ coefficients)
        )


def _laml(design_matrix, response, coefficients, smoothness, penalty, l2_penalty):
    """Laplace approximate marginal likelihood of one row, up to a constant.

    ``-L(beta) + 0.5 log|Q|_+ - 0.5 log|A|`` with ``L`` the penalized loss,
    ``Q`` the prior precision (twice the penalty) and ``A`` the loss Hessian.
    ``penalty`` is one term's matrix or a stack of them, and ``smoothness``
    one strength or one per term.
    """
    kwargs = {
        "l2_penalty": l2_penalty,
        "smoothness": smoothness,
        "smoothness_matrix": penalty,
    }
    flat = coefficients.ravel()
    loss = dirichlet_neg_log_likelihood(flat, design_matrix, response, **kwargs)
    hessian = dirichlet_hessian(flat, design_matrix, response, **kwargs)
    n_coefficients = design_matrix.shape[1]
    penalties = np.reshape(penalty, (-1, n_coefficients, n_coefficients))
    strengths = np.broadcast_to(smoothness, (penalties.shape[0],))
    curvature = np.tensordot(strengths, penalties, axes=1)
    penalty = penalties.sum(axis=0)
    ridge = np.diag(np.r_[0.0, np.full(n_coefficients - 1, l2_penalty)])
    precision = 2.0 * np.kron(ridge + curvature, np.eye(N_STATES - 1))
    # The pseudo-determinant over the penalized directions, whose number the
    # strength does not change: every coefficient but the intercepts with an
    # L2 penalty, else the curvature penalty's null space is unpenalized too.
    rank = (N_STATES - 1) * (
        n_coefficients - 1
        if l2_penalty > 0
        else np.linalg.matrix_rank(penalty, hermitian=True)
    )
    log_pdet = np.sum(np.log(np.linalg.eigvalsh(precision)[-rank:]))
    return -loss + 0.5 * log_pdet - 0.5 * np.linalg.slogdet(hessian)[1]


@pytest.mark.unit
class TestRemlSmoothness:
    @pytest.mark.parametrize(
        ("n_time", "ridge"),
        # Short with the default ridge has a LAML with two local maxima.
        [(6_000, 1e-3), (1_500, 0.1)],
    )
    def test_reml_maximizes_the_laplace_marginal_likelihood(self, n_time, ridge):
        """The REML strength scores (nearly) as well as the best strength on
        a dense grid of the Laplace approximate marginal likelihood."""
        _, _, design, _, response = _row_problem(seed=8, n_time=n_time)
        design_matrix = np.asarray(design, dtype=float)
        penalty = _smoothness_matrix(design.design_info)
        start = np.zeros((design.shape[1], N_STATES, N_STATES - 1))
        common = {
            "transition_regularization": ridge,
            "smoothness_matrix": penalty,
        }

        fit = _fit_covariate_transition(
            start, design_matrix, response, smoothness="reml", **common
        )

        def score(row, smoothness, coefficients):
            return _laml(
                design_matrix[:-1],
                response[:, row],
                coefficients[:, row],
                smoothness,
                penalty,
                ridge,
            )

        # REML's search range; above it the LAML is numerical noise.
        grid = np.logspace(-12, 6, 37)
        grid_fits = [
            _fit_covariate_transition(
                start, design_matrix, response, smoothness=value, **common
            )
            for value in grid
        ]
        for row in range(N_STATES):
            grid_scores = [
                score(row, value, grid_fit.coefficients)
                for value, grid_fit in zip(grid, grid_fits, strict=True)
            ]
            reml_score = score(row, fit.smoothness[row], fit.coefficients)
            assert reml_score >= max(grid_scores) - 0.05, (
                f"row {row}: REML strength {fit.smoothness[row, 0]:.3g} scores "
                f"{reml_score:.3f}, grid best {max(grid_scores):.3f} at "
                f"{grid[int(np.argmax(grid_scores))]:.3g}"
            )

    @pytest.mark.parametrize(
        ("glitch", "ridge"),
        # Without the L2 penalty the LAML counts the curvature penalty's own
        # rank, whose smallest genuine eigenvalues are 1e-12 of the largest
        # here (the null direction's are round-off, 1e-23).
        [(1_500.0, 0.1), (20_000.0, 0.0)],
    )
    def test_reml_searches_below_its_grid_when_glitches_stretch_the_spline(
        self, glitch, ridge
    ):
        """Tracking glitches far above the data stretch the spline's range,
        so the data fill a sliver of it and even the grid's weakest strength
        over-smooths them. REML follows the LAML below its grid."""
        rng = np.random.default_rng(8)
        speed = _speeds(rng, 6_000)
        speed[rng.choice(speed.size, 5, replace=False)] = glitch
        design = _design(speed)
        response = _sampled_responses(rng, _true_transition(np.minimum(speed, 200.0)))
        design_matrix = np.asarray(design, dtype=float)
        penalty = _smoothness_matrix(design.design_info)
        start = np.zeros((design.shape[1], N_STATES, N_STATES - 1))

        fit = _fit_covariate_transition(
            start,
            design_matrix,
            response,
            transition_regularization=ridge,
            smoothness="reml",
            smoothness_matrix=penalty,
        )

        grid = np.logspace(-12, 6, 37)
        grid_fits = [
            _fit_covariate_transition(
                start,
                design_matrix,
                response,
                transition_regularization=ridge,
                smoothness=value,
                smoothness_matrix=penalty,
            )
            for value in grid
        ]
        for row in range(N_STATES):
            grid_scores = [
                _laml(
                    design_matrix[:-1],
                    response[:, row],
                    grid_fit.coefficients[:, row],
                    value,
                    penalty,
                    ridge,
                )
                for value, grid_fit in zip(grid, grid_fits, strict=True)
            ]
            reml_score = _laml(
                design_matrix[:-1],
                response[:, row],
                fit.coefficients[:, row],
                fit.smoothness[row],
                penalty,
                ridge,
            )
            assert reml_score >= max(grid_scores) - 0.05, (
                f"row {row}: REML strength {fit.smoothness[row, 0]:.3g} scores "
                f"{reml_score:.3f}, grid best {max(grid_scores):.3f} at "
                f"{grid[int(np.argmax(grid_scores))]:.3g}"
            )

    def test_reml_smooths_a_linear_truth_more_than_a_wiggly_one(self):
        rng = np.random.default_rng(9)
        speed = _speeds(rng, 8_000)
        design = _design(speed)
        x = speed / 40.0
        linear = np.zeros((speed.size, N_STATES, N_STATES))
        linear[:, :, 0] = (1.0 - x)[:, None]
        wiggly = np.zeros_like(linear)
        wiggly[:, :, 0] = (1.5 * np.sin(3.0 * np.pi * np.clip(x, 0, 2)))[:, None]
        fits = [
            _fit_covariate_transition(
                np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
                np.asarray(design, dtype=float),
                _sampled_responses(rng, softmax(logits, axis=-1)),
                transition_regularization=1e-3,
                smoothness="reml",
                smoothness_matrix=_smoothness_matrix(design.design_info),
            )
            for logits in (linear, wiggly)
        ]

        assert np.all(fits[0].smoothness > 10.0 * fits[1].smoothness)

    def test_reml_reduces_curve_error_against_no_smoothing(self):
        """Over seeds, REML smoothing recovers the true curves better than
        the unsmoothed fit of the same basis."""
        errors = {"none": [], "reml": []}
        for seed in range(4):
            _, speed, design, truth, response = _row_problem(seed=20 + seed)
            for name, smoothness in (("none", 0.0), ("reml", "reml")):
                fit = _fit_covariate_transition(
                    np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
                    np.asarray(design, dtype=float),
                    response,
                    transition_regularization=1e-3,
                    smoothness=smoothness,
                    smoothness_matrix=_smoothness_matrix(design.design_info),
                )
                errors[name].append(
                    np.sqrt(np.mean((fit.transition[:-1] - truth[:-1]) ** 2))
                )

        assert np.mean(errors["reml"]) < np.mean(errors["none"])

    def test_frozen_rows_keep_their_smoothness_and_get_zero_covariance(self):
        _, _, design, _, response = _row_problem(n_time=1_000)
        start = np.zeros((design.shape[1], N_STATES, N_STATES - 1))
        frozen = np.array([False, True, False])

        fit = _fit_covariate_transition(
            start,
            np.asarray(design, dtype=float),
            response,
            smoothness="reml",
            smoothness_matrix=_smoothness_matrix(design.design_info),
            initial_smoothness=np.array([1.0, 3.0, 1.0]),
            frozen_rows=frozen,
        )

        assert fit.smoothness[1] == 3.0
        np.testing.assert_array_equal(fit.coefficients[:, 1], start[:, 1])
        np.testing.assert_array_equal(fit.covariance[1], 0.0)

    def test_reml_without_a_spline_term_raises(self):
        speed = _speeds(np.random.default_rng(10), 200)
        design = _design(speed, "1 + speed")
        with pytest.raises(ValueError, match="quantile_bs"):
            _fit_covariate_transition(
                np.zeros((2, N_STATES, N_STATES - 1)),
                np.asarray(design, dtype=float),
                np.zeros((199, N_STATES, N_STATES)),
                smoothness="reml",
                smoothness_matrix=_smoothness_matrix(design.design_info),
            )

    def test_public_estimator_accepts_smoothness(self):
        _, _, design, _, response = _row_problem(n_time=800)
        start = np.zeros((design.shape[1], N_STATES, N_STATES - 1))
        penalty = _smoothness_matrix(design.design_info)

        coefficients, transition = (
            estimate_non_stationary_state_transition_from_responses(
                start,
                np.asarray(design, dtype=float),
                response,
                smoothness=0.5,
                smoothness_matrix=penalty,
            )
        )
        fit = _fit_covariate_transition(
            start,
            np.asarray(design, dtype=float),
            response,
            smoothness=0.5,
            smoothness_matrix=penalty,
        )

        np.testing.assert_allclose(coefficients, fit.coefficients)
        np.testing.assert_allclose(transition, fit.transition)


@pytest.mark.unit
class TestLaplaceIntervals:
    def test_covariance_is_the_inverse_penalized_hessian(self):
        _, _, design, _, response = _row_problem(n_time=2_000)
        design_matrix = np.asarray(design, dtype=float)
        penalty = _smoothness_matrix(design.design_info)
        fit = _fit_covariate_transition(
            np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
            design_matrix,
            response,
            transition_regularization=0.1,
            smoothness=2.0,
            smoothness_matrix=penalty,
        )

        for row in range(N_STATES):
            hessian = dirichlet_hessian(
                fit.coefficients[:, row].ravel(),
                design_matrix[:-1],
                response[:, row],
                l2_penalty=0.1,
                smoothness=2.0,
                smoothness_matrix=penalty,
            )
            np.testing.assert_allclose(
                fit.covariance[row] @ hessian, np.eye(hessian.shape[0]), atol=1e-8
            )

    def test_delta_method_matches_sampled_coefficients(self):
        """The logit-scale delta-method interval of each probability matches
        quantiles of probabilities pushed through from sampled coefficients."""
        rng, _, design, _, response = _row_problem(n_time=3_000)
        fit = _fit_covariate_transition(
            np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
            np.asarray(design, dtype=float),
            response,
            transition_regularization=0.1,
            smoothness="reml",
            smoothness_matrix=_smoothness_matrix(design.design_info),
        )
        new = {"speed": np.linspace(0.0, 60.0, 7)}
        lower, upper = predict_discrete_state_transition_intervals(
            design, fit.coefficients, fit.covariance, new, coverage=0.9
        )

        new_design = np.asarray(dmatrix(design.design_info, new), dtype=float)
        n_samples = 20_000
        samples = np.stack(
            [
                rng.multivariate_normal(
                    fit.coefficients[:, row].ravel(), fit.covariance[row], n_samples
                ).reshape(n_samples, design.shape[1], N_STATES - 1)
                for row in range(N_STATES)
            ],
            axis=2,
        )
        logits = np.einsum("tc,scrk->strk", new_design, samples)
        sampled = softmax(
            np.concatenate((logits, np.zeros(logits.shape[:-1] + (1,))), axis=-1),
            axis=-1,
        )
        sampled_lower, sampled_upper = np.quantile(sampled, [0.05, 0.95], axis=0)
        width = sampled_upper - sampled_lower

        # The sampled quantiles carry Monte Carlo error of a few percent
        # of the widest interval.
        np.testing.assert_allclose(lower, sampled_lower, atol=0.1 * width.max())
        np.testing.assert_allclose(upper, sampled_upper, atol=0.1 * width.max())

    def test_intervals_bracket_the_estimate_and_narrow_with_data(self):
        widths = []
        for n_time in (2_000, 20_000):
            _, _, design, _, response = _row_problem(seed=11, n_time=n_time)
            fit = _fit_covariate_transition(
                np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
                np.asarray(design, dtype=float),
                response,
                transition_regularization=0.1,
                smoothness=1.0,
                smoothness_matrix=_smoothness_matrix(design.design_info),
            )
            new = {"speed": np.linspace(0.0, 50.0, 11)}
            estimate = predict_discrete_state_transitions(design, fit.coefficients, new)
            lower, upper = predict_discrete_state_transition_intervals(
                design, fit.coefficients, fit.covariance, new
            )
            assert np.all((lower > 0.0) & (upper < 1.0))
            assert np.all((lower <= estimate) & (estimate <= upper))
            widths.append(np.mean(upper - lower))

        assert widths[1] < 0.5 * widths[0]

    @pytest.mark.slow
    def test_intervals_cover_the_true_curve(self):
        """Averaged over the speed range and repeated recordings, 90%
        intervals contain the true transition probability about 90% of the
        time (Bayesian intervals have this across-the-curve coverage)."""
        covered = []
        new = {"speed": np.linspace(1.0, 50.0, 25)}
        for seed in range(40):
            _, _, design, _, response = _row_problem(seed=100 + seed, n_time=4_000)
            fit = _fit_covariate_transition(
                np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
                np.asarray(design, dtype=float),
                response,
                transition_regularization=1e-3,
                smoothness="reml",
                smoothness_matrix=_smoothness_matrix(design.design_info),
            )
            lower, upper = predict_discrete_state_transition_intervals(
                design, fit.coefficients, fit.covariance, new, coverage=0.9
            )
            truth = _true_transition(new["speed"])
            covered.append((lower <= truth) & (truth <= upper))

        # Measured 0.878 with REML and 0.897 without smoothing.
        assert 0.84 <= np.mean(covered) <= 0.97


@pytest.mark.unit
class TestTransitionLogPriorSmoothness:
    def test_log_prior_includes_the_smoothness_penalty(self):
        """The EM objective's covariate log prior is minus the M-step loss
        with no expected transitions, smoothness included."""
        rng, _, design, _, _ = _row_problem(n_time=500)
        design_matrix = np.asarray(design, dtype=float)
        penalty = _smoothness_matrix(design.design_info)
        coefficients = rng.normal(0.0, 0.3, (design.shape[1], N_STATES, N_STATES - 1))
        smoothness = np.array([0.5, 2.0, 7.0])
        stickiness = np.array([10.0, 0.0, 3.0])
        no_counts = np.zeros((design.shape[0] - 1, N_STATES, N_STATES))

        log_prior = _transition_log_prior(
            no_counts,
            None,
            coefficients,
            design_matrix,
            1.0,
            stickiness,
            0.05,
            smoothness=smoothness,
            smoothness_matrix=penalty,
        )
        alpha = 1.0 + np.diag(stickiness)
        expected = -sum(
            dirichlet_neg_log_likelihood(
                coefficients[:, row].ravel(),
                design_matrix[:-1],
                no_counts[:, row],
                alpha=alpha[row],
                l2_penalty=0.05,
                smoothness=smoothness[row],
                smoothness_matrix=penalty,
            )
            for row in range(N_STATES)
        )

        assert log_prior == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
class TestTransitionTypeSmoothness:
    @pytest.mark.parametrize("smoothness", [-1.0, "gcv"])
    def test_invalid_smoothness_raises(self, smoothness):
        with pytest.raises(ValueError, match="smoothness"):
            DiscreteNonStationaryCustom(
                values=np.full((N_STATES, N_STATES), 1.0 / N_STATES),
                smoothness=smoothness,
            )

    def test_smoothness_without_quantile_bs_raises_at_construction_of_design(self):
        model = DiscreteNonStationaryCustom(
            values=np.full((N_STATES, N_STATES), 1.0 / N_STATES),
            formula="1 + speed",
            smoothness="reml",
        )
        with pytest.raises(ValueError, match="quantile_bs"):
            model.make_state_transition({"speed": np.linspace(0, 10, 50)})

    @pytest.mark.parametrize("smoothness", ["reml", 0.5])
    def test_smoothness_without_an_intercept_raises(self, smoothness):
        model = DiscreteNonStationaryCustom(
            values=np.full((N_STATES, N_STATES), 1.0 / N_STATES),
            formula="0 + quantile_bs(speed)",
            smoothness=smoothness,
        )
        with pytest.raises(ValueError, match="intercept"):
            model.make_state_transition({"speed": np.linspace(0, 10, 50)})

    def test_no_intercept_is_allowed_without_smoothness(self):
        model = DiscreteNonStationaryCustom(
            values=np.full((N_STATES, N_STATES), 1.0 / N_STATES),
            formula="0 + quantile_bs(speed)",
        )
        model.make_state_transition({"speed": np.linspace(0, 10, 50)})


@pytest.mark.unit
class TestSplineEdgeCases:
    @pytest.mark.parametrize(
        ("formula", "column"),
        [
            ("1 + quantile_bs(x)", "x"),
            (f"1 + cyclic_bs(x, lower=0.0, upper={PERIOD!r})", "x"),
        ],
    )
    def test_missing_covariates_follow_the_na_action(self, formula, column):
        """NaN covariates do not reach knot placement: patsy's NA_action
        decides, so "raise" names the missing values and "drop" fits the
        rest."""
        x = np.random.default_rng(37).uniform(0.0, 6.0, 500)
        x[[10, 200]] = np.nan
        with pytest.raises(PatsyError, match="missing"):
            dmatrix(formula, {column: x}, NA_action="raise")
        dropped = dmatrix(formula, {column: x}, NA_action="drop")
        assert dropped.shape[0] == 498
        assert np.all(np.isfinite(np.asarray(dropped)))

    def test_linear_cyclic_spline_has_no_curvature_penalty(self):
        design = _cyclic_design(
            np.random.default_rng(38).uniform(0, PERIOD, 500),
            f"cyclic_bs(phase, lower=0.0, upper={PERIOD!r}, degree=1)",
        )
        (penalty,) = _smoothness_matrix(design.design_info)
        assert not np.any(penalty)

    def test_reml_row_without_transitions_keeps_the_strongest_smoothing(self):
        """A row with no expected transitions has a singular Hessian and no
        Laplace approximation: REML keeps the strongest smoothing, quietly,
        and the other rows are fit as usual."""
        _, _, design, _, response = _row_problem(n_time=1_500)
        response = response.copy()
        response[:, 1] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fit = _fit_covariate_transition(
                np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
                np.asarray(design, dtype=float),
                response,
                smoothness="reml",
                smoothness_matrix=_smoothness_matrix(design.design_info),
            )
        assert fit.smoothness[1] == pytest.approx([1e6])
        assert np.all(fit.smoothness[[0, 2]] < 1e6)
        assert np.all(np.isfinite(fit.transition))


@pytest.mark.unit
def test_a_row_without_transitions_has_no_intervals():
    """A row never visited, with a flat prior, has an unidentified intercept:
    its intervals are unavailable (NaN), not zero-width."""
    _, _, design, _, response = _row_problem(n_time=1_500)
    response = response.copy()
    response[:, 1] = 0.0
    fit = _fit_covariate_transition(
        np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
        np.asarray(design, dtype=float),
        response,
        smoothness="reml",
        smoothness_matrix=_smoothness_matrix(design.design_info),
    )
    lower, upper = predict_discrete_state_transition_intervals(
        design,
        fit.coefficients,
        fit.covariance,
        {"speed": np.array([0.0, 20.0, 60.0])},
    )

    assert np.all(np.isnan(fit.covariance[1]))
    assert np.all(np.isnan(lower[:, 1]) & np.isnan(upper[:, 1]))
    assert np.all(upper[:, [0, 2]] - lower[:, [0, 2]] > 0.0)


@pytest.mark.unit
def test_max_transition_change_matches_the_full_transitions():
    rng, _, design, _, _ = _row_problem(n_time=3_000)
    old = rng.normal(0.0, 0.5, (design.shape[1], N_STATES, N_STATES - 1))
    new = old + rng.normal(0.0, 0.05, old.shape)
    old_transitions = _transitions_on_design(np.asarray(design), old)
    new_transitions = _transitions_on_design(np.asarray(design), new)

    assert _max_transition_change(np.asarray(design), old, new) == pytest.approx(
        np.max(np.abs(new_transitions - old_transitions)), rel=1e-12
    )


def _transitions_on_design(design_matrix, coefficients):
    """Covariate transitions from a design matrix, shape (n_time, n_states, n_states)."""
    logits = np.einsum("tc,crk->trk", design_matrix, coefficients)
    logits = np.concatenate((logits, np.zeros(logits.shape[:-1] + (1,))), axis=-1)
    return softmax(logits, axis=-1)


@pytest.mark.unit
def test_a_failing_newton_optimizer_falls_back_to_gradient_steps(monkeypatch):
    """If trust-exact raises (as scipy 1.12's can on a badly conditioned
    Hessian), the row is refit with L-BFGS-B instead of failing the M-step."""
    from non_local_detector import discrete_state_transitions as module

    real_minimize = module.minimize
    methods = []

    def flaky_minimize(*args, **kwargs):
        methods.append(kwargs["method"])
        if kwargs["method"] == "trust-exact":
            raise ValueError("array must not contain infs or NaNs")
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(module, "minimize", flaky_minimize)
    _, _, design, truth, response = _row_problem(n_time=2_000)
    fit = _fit_covariate_transition(
        np.zeros((design.shape[1], N_STATES, N_STATES - 1)),
        np.asarray(design, dtype=float),
        response,
        smoothness=1.0,
        smoothness_matrix=_smoothness_matrix(design.design_info),
    )

    assert methods == ["trust-exact", "L-BFGS-B"] * N_STATES
    assert np.sqrt(np.mean((fit.transition - truth) ** 2)) < 0.05
