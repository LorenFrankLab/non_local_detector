"""Analytic gradients for tiled background KDEs with inactive encoding rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.background_marks import _kde_log_density

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tile", [1, 2, None])
@pytest.mark.parametrize("point", [2.0, 2.5])
@pytest.mark.parametrize("argument", ["points", "samples", "weights"])
def test_background_kde_zero_prefix_has_finite_analytic_gradients(
    tile, point, argument
):
    points = jnp.array([[point]])
    samples = jnp.arange(4.0)[:, None]
    weights = jnp.array([0.0, 0.0, 1.0, 1.0])
    active_kernel = np.exp(-0.5 * (point - np.array([2.0, 3.0])) ** 2)
    responsibilities = active_kernel / active_kernel.sum()
    expected_density = np.log(active_kernel.mean()) - 0.5 * np.log(2.0 * np.pi)

    def score(points, samples, weights):
        return _kde_log_density(
            points, samples, jnp.array([1.0]), weights, sample_tile=tile
        ).sum()

    np.testing.assert_allclose(
        score(points, samples, weights), expected_density, atol=1e-6
    )
    if argument == "points":
        gradient = jax.grad(score, argnums=0)(points, samples, weights)
        expected = [[np.sum(responsibilities * (np.array([2.0, 3.0]) - point))]]
    elif argument == "samples":
        gradient = jax.grad(score, argnums=1)(points, samples, weights)
        expected = np.array(
            [0.0, 0.0, *(responsibilities * (point - np.array([2.0, 3.0])))]
        )[:, None]
    else:
        gradient = jax.grad(score, argnums=2)(points, samples, weights)
        # Zero responsibilities designate inactive rows. Their established
        # boundary derivative is zero; active normalized weights have the usual
        # kernel responsibility minus normalization derivative.
        expected = np.array([0.0, 0.0, *(responsibilities - 0.5)])
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(gradient, expected, atol=1e-6)


@pytest.mark.parametrize("tile", [1, 2, None])
def test_background_kde_nan_encoding_sample_propagates_even_when_inactive(tile):
    samples = jnp.array([[np.nan], [1.0], [2.0], [3.0]])
    result = _kde_log_density(
        jnp.array([[2.5]]),
        samples,
        jnp.array([1.0]),
        jnp.array([0.0, 0.0, 1.0, 1.0]),
        sample_tile=tile,
    )
    assert np.isnan(np.asarray(result)).all()
