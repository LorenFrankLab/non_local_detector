"""A spike's intensity must not depend on other queries in its decode batch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    _compute_log_mark_kernel_gemm,
    estimate_log_joint_mark_intensity,
)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 1])
def test_tail_intensity_is_independent_of_other_decoding_spikes(enc_tile_size):
    """A representable density ratio survives beside a much larger mark kernel."""
    std = 0.001
    remote = np.zeros((1, 8), dtype=np.float32)
    remote[0, 0] = np.sqrt(180.0) * std
    encoded = jnp.zeros((2, 8))
    kwargs = {
        "encoding_spike_waveform_features": encoded,
        "waveform_stds": jnp.full(8, std),
        "occupancy": jnp.asarray([1e-14]),
        "mean_rate": 1.0,
        "log_position_distance": jnp.zeros((2, 1)),
        "enc_tile_size": enc_tile_size,
    }
    alone = estimate_log_joint_mark_intensity(jnp.asarray(remote), **kwargs)
    together = estimate_log_joint_mark_intensity(
        jnp.asarray(np.concatenate([np.zeros_like(remote), remote])), **kwargs
    )
    expected = -90.0 - 8 * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    np.testing.assert_allclose(alone[0, 0], expected, atol=3e-5, rtol=1e-5)
    np.testing.assert_allclose(together[1], alone[0], atol=3e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 1, 2])
def test_opposing_mark_and_position_support_preserves_small_joint_density(
    enc_tile_size,
):
    """Separate factor maxima are insufficient when different sources dominate."""
    std = 0.001
    encoded = jnp.asarray([[0.0], [np.sqrt(180.0) * std]])
    out = estimate_log_joint_mark_intensity(
        jnp.zeros((1, 1)),
        encoded,
        jnp.asarray([std]),
        jnp.asarray([1e-14]),
        1.0,
        jnp.asarray([[-90.0], [0.0]]),
        enc_tile_size=enc_tile_size,
    )
    expected = -90.0 - np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    # Do not test the caller's final floor: this raw log ratio is finite.
    np.testing.assert_allclose(out[0, 0], expected, atol=3e-5, rtol=1e-5)


@pytest.mark.unit
def test_mark_kernel_preserves_distances_after_large_translation():
    decoded = jnp.asarray([[1.0]])
    encoded = jnp.asarray([[0.0]])
    std = jnp.asarray([1.0])
    ordinary = _compute_log_mark_kernel_gemm(decoded, encoded, std)
    translated = _compute_log_mark_kernel_gemm(decoded + 10000, encoded + 10000, std)
    np.testing.assert_allclose(translated, ordinary, atol=1e-6, rtol=1e-6)


@pytest.mark.unit
def test_zero_weight_support_has_a_finite_zero_weight_gradient():
    def score(weights):
        return estimate_log_joint_mark_intensity(
            jnp.zeros((1, 2)),
            jnp.zeros((2, 2)),
            jnp.ones(2),
            jnp.ones(1),
            1.0,
            jnp.zeros((2, 1)),
            encoding_weights=weights,
        ).sum()

    gradient = jax.grad(score)(jnp.zeros(2))
    np.testing.assert_array_equal(gradient, np.zeros(2))


@pytest.mark.unit
def test_zero_weight_prefix_tiles_do_not_poison_feature_gradients():
    def score(features):
        return estimate_log_joint_mark_intensity(
            jnp.asarray([[2.5]]),
            features,
            jnp.ones(1),
            jnp.ones(1),
            1.0,
            jnp.zeros((4, 1)),
            encoding_weights=jnp.asarray([0.0, 0.0, 1.0, 1.0]),
            enc_tile_size=2,
        ).sum()

    gradient = jax.grad(score)(jnp.arange(4.0)[:, None])
    np.testing.assert_allclose(
        gradient[:, 0], [0.0, 0.0, 0.25, -0.25], rtol=1e-6, atol=1e-6
    )


@pytest.mark.unit
def test_broadcast_bandwidth_normalizes_every_mark_dimension():
    kernel = _compute_log_mark_kernel_gemm(
        jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.asarray([2.0])
    )
    expected = -2 * np.log(2 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(kernel, expected, rtol=1e-6, atol=1e-6)
