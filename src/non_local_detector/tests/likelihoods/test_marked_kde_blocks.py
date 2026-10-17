"""Blocked marked KDE estimation preserves the finished spike terms."""

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
from non_local_detector.likelihoods.common import _padded_spike_count
from non_local_detector.tests.likelihoods.conftest import (
    FLOAT32_ROUNDING,
    realistic_log_background,
)

pytestmark = pytest.mark.unit


def arguments(n_dec, n_enc, weight_kind):
    rng = np.random.default_rng(7304)
    values = {
        "decoding_spike_waveform_features": jnp.asarray(rng.normal(size=(n_dec, 3))),
        "encoding_spike_waveform_features": jnp.asarray(rng.normal(size=(n_enc, 3))),
        "waveform_stds": jnp.ones(3),
        "occupancy": jnp.array([0.2, 0.0, 0.7, 0.5]),
        "mean_rate": 0.2,
        "log_position_distance": jnp.log(rng.uniform(0, 0.4, (n_enc, 4))),
        "encoding_weights": None,
    }
    if weight_kind != "none":
        weights = np.zeros(n_enc) if weight_kind == "zero" else rng.uniform(0, 2, n_enc)
        values["encoding_weights"] = jnp.asarray(weights)
    return values


def independent_finished_intensity_and_gradient(values, log_background):
    """Host float64 Gaussian mixture oracle, independent of JAX dot policy.

    Each finished term is ``logaddexp(log intensity, log_background)``.
    """
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
    position = np.exp(np.asarray(values["log_position_distance"], dtype=np.float64))
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
    with np.errstate(divide="ignore"):
        log_intensity = np.log(intensity)  # -inf where the intensity is zero
    log_background = np.asarray(log_background, dtype=np.float64)[:, None]
    finished = np.logaddexp(log_intensity, log_background)

    derivative = (encoding[:, None, :] - decoding[None, :, :]) / std**2
    derivative = np.einsum("er,ep,erd->rpd", kernel, weighted_position, derivative)
    ratio = np.zeros_like(derivative)
    np.divide(
        derivative, numerator[:, :, None], out=ratio, where=numerator[:, :, None] > 0
    )
    # d logaddexp(log I, b) / dx = I / (I + B) * d log I / dx; zero where I == 0.
    share = np.exp(log_intensity - finished)
    gradient = (share[:, :, None] * ratio).sum(axis=1)
    return finished, gradient, intensity


def unblocked_finished(values, log_background):
    """The unblocked estimator's terms with the same background."""
    return estimate_log_joint_mark_intensity(**values, log_background=log_background)


@pytest.mark.parametrize("n_dec,n_enc", [(0, 7), (1, 0), (1, 7), (13, 9)])
@pytest.mark.parametrize("weight_kind", ["none", "zero", "weighted"])
def test_marked_blocks_match_finished_unblocked_intensity(n_dec, n_enc, weight_kind):
    values = arguments(n_dec, n_enc, weight_kind)
    log_background = realistic_log_background(
        values["decoding_spike_waveform_features"]
    )
    actual = block_estimate_log_joint_mark_intensity(
        **values, block_size=4, log_background=log_background
    )
    oracle, _, intensity = independent_finished_intensity_and_gradient(
        values, log_background
    )
    expected = unblocked_finished(values, log_background)
    assert actual.shape == (n_dec, 4)
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)
    np.testing.assert_allclose(actual, oracle, **FLOAT32_ROUNDING)
    assert np.isfinite(actual).all()
    if n_dec:
        # Zero occupancy: no intensity, so the term is the background.
        np.testing.assert_allclose(actual[:, 1], log_background, **FLOAT32_ROUNDING)


def test_marked_blocks_do_not_eagerly_copy_the_complete_output_per_block(monkeypatch):
    calls = []
    original = jax.lax.dynamic_update_slice

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(jax.lax, "dynamic_update_slice", record)
    values = arguments(13, 9, "weighted")
    actual = block_estimate_log_joint_mark_intensity(
        **values,
        block_size=4,
        log_background=realistic_log_background(
            values["decoding_spike_waveform_features"]
        ),
    )
    actual.block_until_ready()
    assert len(calls) <= 1, "every decoding block still dispatches a full-output update"


@pytest.mark.parametrize("n_decoding", [1, 2, 17])
def test_oversized_mark_blocks_do_not_allocate_requested_padding(
    monkeypatch, n_decoding
):
    """A sparse native chunk must not create a 10k-row spatial output.

    Intercept before compilation/allocation: padding to the block would require
    1.325 GB for this output alone, while the few padded sizes shared across
    chunks need at most 4.2 MB.
    """
    n_position_bins = 33_124
    n_padded = _padded_spike_count(n_decoding, 10_000)
    assert n_padded <= 2 * max(n_decoding, 16)

    def bounded_core(features, *args, log_background, **kwargs):
        assert features.shape[0] == n_padded
        assert log_background.shape == (n_padded,)
        return jnp.zeros((n_padded, n_position_bins), dtype=features.dtype)

    monkeypatch.setattr(marked_kde, "estimate_log_joint_mark_intensity", bounded_core)
    actual = block_estimate_log_joint_mark_intensity(
        jnp.ones((n_decoding, 3)),
        jnp.ones((2, 3)),
        jnp.ones(3),
        jnp.ones(n_position_bins),
        0.2,
        jnp.zeros((2, n_position_bins)),
        block_size=10_000,
        log_background=realistic_log_background(jnp.ones((n_decoding, 3))),
    )
    assert actual.shape == (n_decoding, n_position_bins)


def test_marked_block_gradients_match_unblocked_kernel():
    values = arguments(13, 9, "weighted")
    log_background = realistic_log_background(
        values["decoding_spike_waveform_features"]
    )
    _, oracle, intensity = independent_finished_intensity_and_gradient(
        values, log_background
    )
    marks = values.pop("decoding_spike_waveform_features")
    expected = jax.grad(
        lambda data: unblocked_finished(
            {"decoding_spike_waveform_features": data, **values},
            log_background,
        ).sum()
    )(marks)
    actual = jax.grad(
        lambda data: block_estimate_log_joint_mark_intensity(
            data, **values, block_size=4, log_background=log_background
        ).sum()
    )(marks)
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)
    np.testing.assert_allclose(actual, oracle, **FLOAT32_ROUNDING)
    assert np.isfinite(actual).all()
