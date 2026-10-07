"""Linear marked KDE block optimizations preserve finished intensity floors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import non_local_detector.likelihoods.clusterless_kde as marked_kde
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
)
from non_local_detector.likelihoods.common import EPS, LOG_EPS
from non_local_detector.tests.likelihoods.conftest import FLOAT32_ROUNDING

pytestmark = pytest.mark.unit


def arguments(n_dec, n_enc, weight_kind):
    rng = np.random.default_rng(7304)
    values = {
        "decoding_spike_waveform_features": jnp.asarray(rng.normal(size=(n_dec, 3))),
        "encoding_spike_waveform_features": jnp.asarray(rng.normal(size=(n_enc, 3))),
        "waveform_stds": jnp.ones(3),
        "occupancy": jnp.array([0.2, 0.0, 0.7, 0.5]),
        "mean_rate": 0.2,
        "position_distance": jnp.asarray(rng.uniform(0, 0.4, (n_enc, 4))),
        "encoding_weights": None,
    }
    if weight_kind != "none":
        weights = np.zeros(n_enc) if weight_kind == "zero" else rng.uniform(0, 2, n_enc)
        values["encoding_weights"] = jnp.asarray(weights)
    return values


def independent_finished_intensity_and_gradient(values):
    """Host float64 Gaussian mixture oracle, independent of JAX dot policy."""
    decoding = np.asarray(values["decoding_spike_waveform_features"], dtype=np.float64)
    encoding = np.asarray(values["encoding_spike_waveform_features"], dtype=np.float64)
    std = np.asarray(values["waveform_stds"], dtype=np.float64)
    occupancy = np.asarray(values["occupancy"], dtype=np.float64)
    weights = values["encoding_weights"]
    weights = (
        np.ones(len(encoding))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    position = np.asarray(values["position_distance"], dtype=np.float64)
    kernel = norm.pdf(decoding[None, :, :], loc=encoding[:, None, :], scale=std).prod(
        axis=-1
    )
    weighted_position = weights[:, None] * position
    numerator = kernel.T @ weighted_position
    density = (
        numerator / weights.sum() if weights.sum() > 0 else np.zeros_like(numerator)
    )
    intensity = np.zeros_like(density)
    np.divide(density, occupancy, out=intensity, where=occupancy > 0)
    intensity *= values["mean_rate"]
    finished = np.log(np.maximum(intensity, EPS))

    derivative = (encoding[:, None, :] - decoding[None, :, :]) / std**2
    derivative = np.einsum("er,ep,erd->rpd", kernel, weighted_position, derivative)
    ratio = np.zeros_like(derivative)
    np.divide(
        derivative, numerator[:, :, None], out=ratio, where=numerator[:, :, None] > 0
    )
    active = (occupancy[None, :] > 0) & (intensity > EPS)
    gradient = np.where(active[:, :, None], ratio, 0).sum(axis=1)
    return finished, gradient


@pytest.mark.parametrize("n_dec,n_enc", [(0, 7), (1, 0), (1, 7), (13, 9)])
@pytest.mark.parametrize("weight_kind", ["none", "zero", "weighted"])
def test_marked_blocks_match_finished_unblocked_intensity(n_dec, n_enc, weight_kind):
    values = arguments(n_dec, n_enc, weight_kind)
    expected = jnp.clip(estimate_log_joint_mark_intensity(**values), min=LOG_EPS)
    actual = block_estimate_log_joint_mark_intensity(**values, block_size=4)
    oracle, _ = independent_finished_intensity_and_gradient(values)
    assert actual.shape == (n_dec, 4)
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)
    np.testing.assert_allclose(actual, oracle, **FLOAT32_ROUNDING)
    assert np.isfinite(actual).all()
    if n_dec:
        np.testing.assert_allclose(actual[:, 1], LOG_EPS, **FLOAT32_ROUNDING)


def test_marked_blocks_do_not_eagerly_copy_the_complete_output_per_block(monkeypatch):
    calls = []
    original = jax.lax.dynamic_update_slice

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(jax.lax, "dynamic_update_slice", record)
    actual = block_estimate_log_joint_mark_intensity(
        **arguments(13, 9, "weighted"), block_size=4
    )
    actual.block_until_ready()
    assert len(calls) <= 1, "every decoding block still dispatches a full-output update"


@pytest.mark.parametrize("n_decoding", [1, 2, 17])
def test_oversized_mark_blocks_do_not_allocate_requested_padding(
    monkeypatch, n_decoding
):
    """A sparse native chunk must not create a 10k-row spatial output.

    Intercept before compilation/allocation: the old padded shape would require
    1.325 GB for this output alone, while actual marks need at most 2.3 MB.
    """
    n_position_bins = 33_124

    def bounded_core(features, *args):
        block_size = args[-1]
        assert block_size == n_decoding
        assert features.shape[0] == n_decoding
        return jnp.zeros((n_decoding, n_position_bins), dtype=features.dtype)

    monkeypatch.setattr(marked_kde, "_blocked_joint_mark_intensity", bounded_core)
    actual = block_estimate_log_joint_mark_intensity(
        jnp.ones((n_decoding, 3)),
        jnp.ones((2, 3)),
        jnp.ones(3),
        jnp.ones(n_position_bins),
        0.2,
        jnp.ones((2, n_position_bins)),
        block_size=10_000,
    )
    assert actual.shape == (n_decoding, n_position_bins)


def test_marked_block_gradients_match_unblocked_kernel():
    values = arguments(13, 9, "weighted")
    _, oracle = independent_finished_intensity_and_gradient(values)
    marks = values.pop("decoding_spike_waveform_features")
    expected = jax.grad(
        lambda data: jnp.clip(
            estimate_log_joint_mark_intensity(data, **values), min=LOG_EPS
        ).sum()
    )(marks)
    actual = jax.grad(
        lambda data: block_estimate_log_joint_mark_intensity(
            data, **values, block_size=4
        ).sum()
    )(marks)
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)
    np.testing.assert_allclose(actual, oracle, **FLOAT32_ROUNDING)
    assert np.isfinite(actual).all()
