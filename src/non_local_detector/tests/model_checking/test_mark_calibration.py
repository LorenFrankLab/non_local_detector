"""Held-out marks must be tested against a fitted conditional distribution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import kstest, norm

from non_local_detector.model_checking.clusterless import interval_rescaling_transform


@pytest.mark.unit
def test_density_only_inputs_cannot_claim_mark_calibration():
    time = np.linspace(0, 1, 20)
    spikes = np.array([0.2, 0.4, 0.8])
    marks = np.array([[-2.0], [0.0], [2.0]])
    with pytest.raises(ValueError, match="conditional_mark_cdf"):
        interval_rescaling_transform(
            time, spikes, marks, np.ones(len(time)), np.array([[1.0], [20.0], [100.0]])
        )


@pytest.mark.unit
def test_observed_marks_use_fixed_cdf_and_wrong_model_is_rejected():
    rng = np.random.default_rng(10)
    marks = rng.normal(size=(4000, 1))
    spikes = np.linspace(0.01, 0.99, len(marks))
    time = np.array([0.0, 1.0])

    def good(times, observed, order):
        return norm.cdf(observed[:, order])

    def bad(times, observed, order):
        return norm.cdf(observed[:, order] - 3)

    _, correct = interval_rescaling_transform(
        time, spikes, marks, np.ones(2), conditional_mark_cdf=good
    )
    _, wrong = interval_rescaling_transform(
        time, spikes, marks, np.ones(2), conditional_mark_cdf=bad
    )
    assert kstest(correct[:, 0], "uniform").pvalue > 0.001
    assert kstest(wrong[:, 0], "uniform").pvalue < 1e-100


@pytest.mark.unit
def test_correlated_gaussian_uses_sequential_conditionals():
    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    rho = 0.85
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(32)
    marks = rng.multivariate_normal(np.zeros(2), covariance, size=5000)
    fitted = GaussianMixtureConditionalCDF([1.0], [[0.0, 0.0]], covariance[None])
    result = fitted(np.arange(len(marks)), marks, np.arange(2))
    expected = np.column_stack(
        (
            norm.cdf(marks[:, 0]),
            norm.cdf((marks[:, 1] - rho * marks[:, 0]) / np.sqrt(1 - rho * rho)),
        )
    )
    assert_allclose(result, expected, atol=1e-12)
    assert abs(np.corrcoef(result.T)[0, 1]) < 0.04
    assert all(kstest(result[:, i], "uniform").pvalue > 0.001 for i in range(2))
    independent = GaussianMixtureConditionalCDF(
        [1.0], [[0.0, 0.0]], np.ones((1, 2)), covariance_type="diag"
    )
    misspecified = independent(np.arange(len(marks)), marks, np.arange(2))
    # Marginal uniformity alone cannot detect the omitted correlation.
    assert np.corrcoef(misspecified.T)[0, 1] > 0.7


@pytest.mark.unit
def test_position_conditioning_matches_known_gaussian():
    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    position = np.array([-2.0, 0.0, 3.0])[:, None]
    marks = np.array([[-2.5], [0.2], [4.0]])
    covariance = np.array([[1.0, 1.5], [1.5, 2.61]])
    fitted = GaussianMixtureConditionalCDF(
        [1.0],
        [[0.0, 0.0]],
        covariance[None],
        n_position_dims=1,
        position_at_time=lambda times: position,
        batch_size=2,
    )
    actual = fitted(np.arange(3), marks, np.array([0]))
    assert_allclose(
        actual[:, 0], norm.cdf((marks[:, 0] - 1.5 * position[:, 0]) / 0.6), atol=1e-12
    )


@pytest.mark.unit
def test_mixture_responsibilities_condition_on_prior_features_and_order(monkeypatch):
    from scipy.special import logsumexp

    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    means = np.array([[-2.0, -1.0], [2.0, 1.0]])
    fitted = GaussianMixtureConditionalCDF(
        [0.4, 0.6], means, np.ones((2, 2)), "diag", batch_size=1
    )
    marks = np.array([[-1.8, -0.5], [1.5, 1.4], [0.0, 0.7]])
    result = fitted.transform(marks)
    log_weights = np.log([0.4, 0.6])[None] + norm.logpdf(
        marks[:, 0, None] - means[:, 0]
    )
    responsibilities = np.exp(log_weights - logsumexp(log_weights, axis=1)[:, None])
    expected_second = np.sum(
        responsibilities * norm.cdf(marks[:, 1, None] - means[:, 1]), axis=1
    )
    assert_allclose(result[:, 1], expected_second, atol=1e-12)
    reversed_result = fitted.transform(marks, feature_order=[1, 0])
    assert not np.allclose(reversed_result, result[:, ::-1])
    monkeypatch.setattr(np.random, "permutation", lambda count: np.array([1, 0]))
    _, callback_result = interval_rescaling_transform(
        np.array([0.0, 1.0]),
        np.array([0.1, 0.5, 0.9]),
        marks,
        np.ones(2),
        conditional_mark_cdf=fitted,
        permute_waveform_features=True,
    )
    assert_allclose(callback_result, reversed_result, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gmm_wrapper_snapshots_fitted_parameters_and_batches(covariance_type):
    from non_local_detector.likelihoods.gmm import (
        GaussianMixtureModel,
        _compute_precision_cholesky,
    )
    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    model = GaussianMixtureModel(n_components=2, covariance_type=covariance_type)
    model.weights_ = np.array([0.3, 0.7])
    model.means_ = np.array([[-2.0, -1.0], [2.0, 1.0]])
    covariance = {
        "full": np.array([np.eye(2), np.eye(2) * 2]),
        "tied": np.eye(2),
        "diag": np.array([[1.0, 1.0], [2.0, 2.0]]),
        "spherical": np.array([1.0, 2.0]),
    }[covariance_type]
    model.covariances_ = covariance
    model.precisions_chol_ = _compute_precision_cholesky(covariance, covariance_type)
    fitted = GaussianMixtureConditionalCDF.from_gmm(model, batch_size=2)
    marks = np.random.default_rng(11).normal(size=(17, 2))
    expected = GaussianMixtureConditionalCDF.from_gmm(model, batch_size=17).transform(
        marks
    )
    assert_allclose(fitted.transform(marks), expected, atol=1e-12)
    model.means_[:] = 1e3
    assert_allclose(fitted.transform(marks), expected, atol=1e-12)
    assert not fitted.means.flags.writeable


@pytest.mark.unit
def test_kde_position_weights_and_excluded_components():
    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    # A third zero-weight component cannot affect the fitted distribution.
    marks = np.array([[-2.0], [2.0], [100.0]])
    position = np.array([[-1.0], [1.0], [100.0]])
    fitted = GaussianMixtureConditionalCDF.from_kde(
        marks,
        0.5,
        weights=[0.2, 0.8, 0.0],
        encoding_positions=position,
        position_std=0.4,
        batch_size=1,
    )
    query_marks = np.array([[-1.0], [1.0]])
    query_position = np.array([[-0.7], [0.7]])
    component_probability = np.array([0.2, 0.8])[None] * norm.pdf(
        (query_position - position[:2].T) / 0.4
    )
    component_probability = (
        component_probability / component_probability.sum(axis=1)[:, None]
    )
    expected = np.sum(
        component_probability * norm.cdf((query_marks - marks[:2].T) / 0.5), axis=1
    )
    assert_allclose(
        fitted.transform(query_marks, given_positions=query_position)[:, 0],
        expected,
        atol=1e-12,
    )
    assert len(fitted.weights) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "cdf",
    [
        np.zeros((3, 2)),
        np.full((3, 1), np.nan),
        np.full((3, 1), -0.1),
        np.full((3, 1), 1.1),
    ],
)
def test_invalid_fitted_cdf_outputs_raise(cdf):
    with pytest.raises(ValueError, match="conditional_mark_cdf"):
        interval_rescaling_transform(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.4, 0.8]),
            np.zeros((3, 1)),
            np.ones(2),
            conditional_mark_cdf=cdf,
        )


@pytest.mark.unit
def test_precomputed_cdf_cannot_be_reconditioned_by_permutation():
    with pytest.raises(ValueError, match="permutation requires a callable"):
        interval_rescaling_transform(
            np.array([0.0, 1.0]),
            np.array([0.2]),
            np.zeros((1, 2)),
            np.ones(2),
            conditional_mark_cdf=np.full((1, 2), 0.5),
            permute_waveform_features=True,
        )


@pytest.mark.unit
def test_integer_empirical_ranks_remain_descriptive_fractions():
    from non_local_detector.model_checking.clusterless import (
        empirical_marginal_rank_transform,
    )

    assert_allclose(
        empirical_marginal_rank_transform(np.array([[1], [2], [3], [4]])),
        np.array([[0.25], [0.5], [0.75], [1.0]]),
    )


@pytest.mark.unit
def test_zero_weight_component_parameters_do_not_enter_fitted_cdf():
    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    fitted = GaussianMixtureConditionalCDF(
        [1.0, 0.0], [[0.0], [np.nan]], [[1.0], [np.nan]], "diag"
    )
    marks = np.array([[-2.0], [0.0], [2.0]])
    assert_allclose(fitted.transform(marks), norm.cdf(marks), atol=1e-12)


@pytest.mark.unit
def test_mixture_utility_matches_reusable_conditional_cdf():
    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
        gaussian_mixture_rosenblatt_transform,
    )

    marks = np.array([[-2.0], [0.0], [2.0]])
    positions = np.array([[-1.0], [0.0], [1.0]])
    covariance = np.array([[[1.0, 0.5], [0.5, 1.0]]])
    expected = GaussianMixtureConditionalCDF(
        [1.0], [[0.0, 0.0]], covariance, n_position_dims=1
    ).transform(marks, given_positions=positions)
    actual = gaussian_mixture_rosenblatt_transform(
        marks, [1.0], [[0.0, 0.0]], covariance, given_positions=positions, batch_size=2
    )
    assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.unit
def test_permutation_conditioning_cache_remains_bounded():
    from itertools import permutations

    from non_local_detector.model_checking.clusterless import (
        GaussianMixtureConditionalCDF,
    )

    fitted = GaussianMixtureConditionalCDF(
        [1.0], [[0.0, 0.0, 0.0, 0.0]], np.eye(4)[None]
    )
    marks = np.arange(8).reshape(2, 4) / 3
    for order in permutations(range(4)):
        transformed = fitted.transform(marks, feature_order=order)
        assert_allclose(transformed, norm.cdf(marks[:, order]), atol=1e-12)
        assert len(fitted._recipes) == 1
