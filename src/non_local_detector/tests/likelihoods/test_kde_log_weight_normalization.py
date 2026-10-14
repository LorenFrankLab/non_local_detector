"""Normalized mixtures of identical kernels retain the single-kernel density."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.common import LOG_EPS, log_kde

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [3, 7, 257])
@pytest.mark.parametrize("point", [200.0, 1040.7, 1457.0])
def test_weighted_identical_sources_preserve_single_kernel_in_deep_tails(count, point):
    query = jnp.asarray([[point]])
    weights = jnp.geomspace(0.1, 0.001, count)
    reference = log_kde(query, jnp.zeros((1, 1)), jnp.ones(1), jnp.ones(1))
    actual = log_kde(query, jnp.zeros((count, 1)), jnp.ones(1), weights)
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("count", [0, 4])
def test_log_kde_zero_weights_have_zero_safe_query_and_sample_gradients(count):
    query = jnp.array([[1.5]])
    samples = jnp.arange(float(count))[:, None]
    weights = jnp.zeros(count)

    def score(points, sources):
        return log_kde(points, sources, jnp.ones(1), weights).sum()

    value, (query_grad, sample_grad) = jax.value_and_grad(score, argnums=(0, 1))(
        query, samples
    )
    np.testing.assert_allclose(value, LOG_EPS)
    np.testing.assert_array_equal(query_grad, 0.0)
    np.testing.assert_array_equal(sample_grad, 0.0)


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e38])
def test_log_kde_is_invariant_to_global_weight_scale(scale):
    query = jnp.array([[0.5]])
    samples = jnp.arange(8.0)[:, None]
    weights = jnp.linspace(0.1, 1.0, 8)
    reference = log_kde(query, samples, jnp.ones(1), weights)
    actual = log_kde(query, samples, jnp.ones(1), weights * scale)
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_log_kde_weighted_gaussian_density_and_query_gradient_match_oracle():
    query = jnp.array([[0.25], [1.75]])
    samples = jnp.array([[0.0], [1.0], [2.0]])
    weights = jnp.array([0.1, 0.3, 0.6])

    def score(points):
        return log_kde(points, samples, jnp.ones(1), weights).sum()

    actual = log_kde(query, samples, jnp.ones(1), weights)
    difference = np.asarray(query) - np.asarray(samples).T
    kernels = np.exp(-0.5 * difference.astype(float) ** 2) / np.sqrt(2 * np.pi)
    mass = kernels * np.asarray(weights)[None, :]
    expected = np.log(mass.sum(axis=1) / np.asarray(weights).sum())
    expected_grad = (-difference * mass).sum(axis=1) / mass.sum(axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jax.grad(score)(query)[:, 0], expected_grad, rtol=1e-6)
