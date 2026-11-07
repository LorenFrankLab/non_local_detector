"""Encoding/position tiles preserve the log-space KDE and its background term."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    log_kde_distance,
)
from tests.conftest import precision_mode
from tests.likelihoods._kde_reference import (
    reference_log_joint_mark_intensity,
    reference_log_joint_mark_intensity_gradient,
    reference_log_position_kernel,
)
from tests.likelihoods.conftest import (
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


def _reference_arguments(values):
    """``values`` in the float64 references' positional order."""
    return (
        values["decoding_features"],
        values["encoding_features"],
        values["waveform_stds"],
        values["occupancy"],
        values["mean_rate"],
        reference_log_position_kernel(
            values["place_bin_centers"],
            values["encoding_positions"],
            values["position_std"],
        ),
    )


def joint_oracle(values):
    """Float64 finished terms ``logaddexp(log intensity, log background)``."""
    return reference_log_joint_mark_intensity(
        *_reference_arguments(values),
        values["encoding_weights"],
        values["log_background"],
    )


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
        oracle = joint_oracle(values)
        assert actual.shape == (shape[0], shape[2])
        np.testing.assert_allclose(actual, oracle, **tolerance(x64))
        np.testing.assert_allclose(actual, baseline_joint(values), **tolerance(x64))


@pytest.mark.parametrize("x64", [False, True])
def test_finished_joint_gradients_match_independent_oracle(x64):
    with precision_mode(x64):
        values = joint_inputs(13, 9, 7, "weighted")
        expected = reference_log_joint_mark_intensity_gradient(
            *_reference_arguments(values),
            values["log_background"],
            values["encoding_weights"],
        )
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
    original = clusterless_kde.log_kde_distance_streaming

    def record(points, samples, std):
        calls.append((len(samples), len(points)))
        assert len(samples) <= 8
        assert len(points) <= 7
        return original(points, samples, std)

    def forbidden(*args, **kwargs):
        pytest.fail("streaming built the full encoding x position kernel")

    monkeypatch.setattr(clusterless_kde, "log_kde_distance_streaming", record)
    monkeypatch.setattr(clusterless_kde, "log_kde_distance", forbidden)
    # Trace afresh so the patched kernels are the ones compiled.
    clusterless_kde.estimate_log_joint_mark_intensity.clear_cache()
    values = joint_inputs(23, 49, 37, "weighted")
    actual = streamed_joint(
        values, encoding_tile_size=8, position_tile_size=7, block_size=5
    )
    actual.block_until_ready()
    clusterless_kde.estimate_log_joint_mark_intensity.clear_cache()
    assert calls
    assert actual.shape == (23, 37)
