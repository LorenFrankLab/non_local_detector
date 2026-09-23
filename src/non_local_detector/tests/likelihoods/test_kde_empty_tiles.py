"""Empty encoding tiles must be inert in values and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity,
)
from non_local_detector.likelihoods.common import LOG_EPS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tile_size", [None, 4])
@pytest.mark.parametrize("empty_tile", [0, 1, 2, "all"])
def test_empty_encoding_tiles_match_independent_density_and_gradient(
    streaming, tile_size, empty_tile
):
    # Ten samples in tiles of four exercise both a partial last tile and padding.
    enc = jnp.linspace(-1.0, 1.0, 10)[:, None]
    pos = jnp.linspace(0.0, 2.0, 10)[:, None]
    dec = jnp.array([[-0.7], [0.2], [0.6]])
    bins = jnp.array([[0.1], [0.8], [1.4]])
    occ = jnp.array([0.2, 0.4, 0.7])
    weights = np.linspace(0.5, 2.0, 10)
    if empty_tile == "all":
        weights[:] = 0.0
    else:
        weights[4 * empty_tile : 4 * (empty_tile + 1)] = 0.0
    weights = jnp.asarray(weights)
    log_pos = -0.5 * (pos - bins.T) ** 2 - 0.5 * jnp.log(2 * jnp.pi)

    def evaluate(marks):
        return estimate_log_joint_mark_intensity(
            marks,
            enc,
            jnp.ones(1),
            occ,
            0.7,
            log_pos,
            enc_tile_size=tile_size,
            use_streaming=streaming and tile_size is not None,
            encoding_positions=pos,
            position_eval_points=bins,
            position_std=jnp.ones(1),
            encoding_weights=weights,
        )

    def reference(marks):
        # Direct Gaussian products, with no max compensation or tiled reduction.
        mark_kernel = jnp.exp(-0.5 * (enc - marks.T) ** 2) / jnp.sqrt(2 * jnp.pi)
        position_kernel = jnp.exp(-0.5 * (pos - bins.T) ** 2) / jnp.sqrt(2 * jnp.pi)
        total = jnp.sum(weights)
        density = jnp.sum(
            weights[:, None, None]
            * mark_kernel[:, :, None]
            * position_kernel[:, None, :],
            axis=0,
        ) / jnp.where(total > 0, total, 1.0)
        positive = density > 0
        return jnp.where(
            positive,
            jnp.log(jnp.where(positive, density, 1.0) * 0.7 / occ),
            LOG_EPS,
        )

    expected = reference(dec)
    expected_grad = jax.grad(lambda marks: reference(marks).sum())(dec)
    actual = evaluate(dec)
    gradient = jax.grad(lambda marks: evaluate(marks).sum())(dec)
    assert np.isfinite(actual).all()
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gradient, expected_grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("tile_size", [None, 4])
@pytest.mark.parametrize("all_zero", [False, True])
def test_invalid_mark_is_not_hidden_by_empty_weights(tile_size, all_zero):
    enc = jnp.zeros((9, 1)).at[0, 0].set(jnp.nan)
    weights = jnp.zeros(9) if all_zero else jnp.ones(9).at[:4].set(0)
    actual = estimate_log_joint_mark_intensity(
        jnp.zeros((2, 1)),
        enc,
        jnp.ones(1),
        jnp.ones(3),
        1.0,
        jnp.zeros((9, 3)),
        enc_tile_size=tile_size,
        encoding_weights=weights,
    )
    assert np.isnan(actual).all()


@pytest.mark.parametrize("tile_size", [None, 4])
def test_zero_mass_position_rows_do_not_poison_remaining_rows(tile_size):
    log_pos = jnp.zeros((9, 3)).at[:4].set(-jnp.inf)
    actual = estimate_log_joint_mark_intensity(
        jnp.zeros((2, 1)),
        jnp.zeros((9, 1)),
        jnp.ones(1),
        jnp.ones(3),
        1.0,
        log_pos,
        enc_tile_size=tile_size,
    )
    expected = np.log(5 / 9 / np.sqrt(2 * np.pi))
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
