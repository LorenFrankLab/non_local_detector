"""Encoding/position tiles preserve the log-space KDE and its background term."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import non_local_detector.likelihoods.clusterless_kde as log_kde
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    log_kde_distance,
)
from non_local_detector.tests.conftest import precision_mode
from non_local_detector.tests.likelihoods.conftest import (
    FLOAT32_ROUNDING,
    realistic_log_background,
    tolerance,
)

pytestmark = pytest.mark.unit


def weights_for(n_samples, kind):
    if kind == "none":
        return None
    return jnp.asarray(
        np.zeros(n_samples)
        if kind == "zero"
        else np.random.default_rng(7421).uniform(0, 2, n_samples)
    )


def joint_inputs(n_decoding, n_encoding, n_positions, weight_kind):
    rng = np.random.default_rng(7423)
    decoding_features = jnp.asarray(rng.normal(size=(n_decoding, 3)))
    return {
        "decoding_features": decoding_features,
        "encoding_features": jnp.asarray(rng.normal(size=(n_encoding, 3))),
        "encoding_positions": jnp.asarray(rng.normal(size=(n_encoding, 2))),
        "place_bin_centers": jnp.asarray(rng.normal(size=(n_positions, 2))),
        "waveform_stds": jnp.asarray([0.7, 1.1, 1.3]),
        "position_std": jnp.asarray([0.8, 1.2]),
        "occupancy": jnp.asarray(
            np.where(np.arange(n_positions) == 1, 0.0, rng.uniform(0.2, 1, n_positions))
        ),
        "mean_rate": 0.2,
        "encoding_weights": weights_for(n_encoding, weight_kind),
        "log_background": realistic_log_background(decoding_features),
    }


def joint_oracle(values):
    dec, enc, positions, centers = [
        np.asarray(values[k], np.float64)
        for k in (
            "decoding_features",
            "encoding_features",
            "encoding_positions",
            "place_bin_centers",
        )
    ]
    mark = norm.pdf(
        dec[None], loc=enc[:, None], scale=np.asarray(values["waveform_stds"])
    ).prod(axis=-1)
    position = norm.pdf(
        centers[None], loc=positions[:, None], scale=np.asarray(values["position_std"])
    ).prod(axis=-1)
    weights = values["encoding_weights"]
    weights = np.ones(len(enc)) if weights is None else np.asarray(weights, np.float64)
    numerator = mark.T @ (weights[:, None] * position)
    density = (
        numerator / weights.sum() if weights.sum() > 0 else np.zeros_like(numerator)
    )
    intensity = np.zeros_like(density)
    np.divide(
        density, values["occupancy"], out=intensity, where=values["occupancy"] > 0
    )
    intensity *= values["mean_rate"]
    with np.errstate(divide="ignore"):
        log_intensity = np.log(intensity)  # -inf where the intensity is zero
    terms = np.logaddexp(
        log_intensity, np.asarray(values["log_background"], np.float64)[:, None]
    )
    return terms, (mark, position, numerator, weights, log_intensity)


def _estimate(values, **options):
    return block_estimate_log_joint_mark_intensity(
        values["decoding_features"],
        values["encoding_features"],
        values["waveform_stds"],
        values["occupancy"],
        values["mean_rate"],
        encoding_weights=values["encoding_weights"],
        log_background=values["log_background"],
        **options,
    )


def baseline_joint(values):
    """Untiled estimate from the precomputed log position kernel."""
    return _estimate(
        values,
        log_position_distance=log_kde_distance(
            values["place_bin_centers"],
            values["encoding_positions"],
            values["position_std"],
        ),
    )


def streamed_joint(values, encoding_tile_size=4, position_tile_size=3, block_size=5):
    """Estimate building the position kernel per encoding x position tile."""
    return _estimate(
        values,
        block_size=block_size,
        enc_tile_size=encoding_tile_size,
        pos_tile_size=position_tile_size,
        use_streaming=True,
        encoding_positions=values["encoding_positions"],
        position_eval_points=values["place_bin_centers"],
        position_std=values["position_std"],
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "shape", [(0, 7, 5), (1, 0, 5), (1, 7, 0), (1, 7, 5), (13, 9, 7)]
)
@pytest.mark.parametrize("weight_kind", ["none", "zero", "weighted"])
def test_joint_tiles_match_finished_oracle(x64, shape, weight_kind):
    with precision_mode(x64):
        values = joint_inputs(*shape, weight_kind)
        actual = streamed_joint(values)
        oracle, _ = joint_oracle(values)
        assert actual.shape == (shape[0], shape[2])
        np.testing.assert_allclose(actual, oracle, **tolerance(x64))
        np.testing.assert_allclose(actual, baseline_joint(values), **tolerance(x64))


@pytest.mark.parametrize("x64", [False, True])
def test_finished_joint_gradients_match_independent_oracle(x64):
    with precision_mode(x64):
        values = joint_inputs(13, 9, 7, "weighted")
        oracle, (mark, position, numerator, weights, log_intensity) = joint_oracle(
            values
        )
        derivative = (
            np.asarray(values["encoding_features"])[:, None]
            - np.asarray(values["decoding_features"])[None]
        ) / np.asarray(values["waveform_stds"]) ** 2
        raw_derivative = np.einsum(
            "ed,ep,edf->dpf", mark, weights[:, None] * position, derivative
        )
        ratio = np.zeros_like(raw_derivative)
        np.divide(
            raw_derivative,
            numerator[:, :, None],
            out=ratio,
            where=numerator[:, :, None] > 0,
        )
        # d logaddexp(log I, b) / dx = I / (I + B) * d log I / dx.
        share = np.exp(log_intensity - oracle)
        expected = (share[:, :, None] * ratio).sum(axis=1)
        marks = values.pop("decoding_features")
        actual = jax.grad(
            lambda x: streamed_joint({"decoding_features": x, **values}).sum()
        )(marks)
        np.testing.assert_allclose(actual, expected, **tolerance(x64))
        untiled = jax.grad(
            lambda x: baseline_joint({"decoding_features": x, **values}).sum()
        )(marks)
        np.testing.assert_allclose(actual, untiled, **tolerance(x64))


@pytest.mark.parametrize("weight_kind", ["none", "zero"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_inputs_keep_baseline_masks_and_background(weight_kind, value):
    values = joint_inputs(7, 6, 5, weight_kind)
    values["encoding_features"] = values["encoding_features"].at[2, 0].set(value)
    actual = streamed_joint(values)
    expected = baseline_joint(values)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING, equal_nan=True)


def test_float32_inputs_match_the_untiled_dtype_under_x64():
    with precision_mode(True):
        values = {
            key: (value.astype(jnp.float32) if hasattr(value, "dtype") else value)
            for key, value in joint_inputs(7, 6, 5, "weighted").items()
        }
        actual = streamed_joint(values)
        expected = baseline_joint(values)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)


def test_position_kernels_are_bounded_by_tiles_not_full_encoding(monkeypatch):
    calls = []
    original = log_kde.log_kde_distance_streaming

    def record(points, samples, std):
        calls.append((len(samples), len(points)))
        assert len(samples) <= 8
        assert len(points) <= 7
        return original(points, samples, std)

    def forbidden(*args, **kwargs):
        pytest.fail("streaming built the full encoding x position kernel")

    monkeypatch.setattr(log_kde, "log_kde_distance_streaming", record)
    monkeypatch.setattr(log_kde, "log_kde_distance", forbidden)
    # Trace afresh so the patched kernels are the ones compiled.
    log_kde.estimate_log_joint_mark_intensity.clear_cache()
    values = joint_inputs(23, 49, 37, "weighted")
    actual = streamed_joint(
        values, encoding_tile_size=8, position_tile_size=7, block_size=5
    )
    actual.block_until_ready()
    log_kde.estimate_log_joint_mark_intensity.clear_cache()
    assert calls
    assert actual.shape == (23, 37)
