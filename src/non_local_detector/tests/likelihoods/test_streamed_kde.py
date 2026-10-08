"""Sample/spatial tiles preserve the linear KDE model and its final floor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import non_local_detector.likelihoods.streamed_kde as streamed
from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
)
from non_local_detector.likelihoods.common import EPS, LOG_EPS, kde
from non_local_detector.tests.conftest import precision_mode
from non_local_detector.tests.likelihoods.conftest import FLOAT32_ROUNDING, tolerance

pytestmark = pytest.mark.unit


def weights_for(n_samples, kind):
    if kind == "none":
        return None
    return jnp.asarray(
        np.zeros(n_samples)
        if kind == "zero"
        else np.random.default_rng(7421).uniform(0, 2, n_samples)
    )


def density_oracle(points, samples, std, weights):
    points, samples, std = [np.asarray(x, np.float64) for x in (points, samples, std)]
    weights = (
        np.ones(len(samples)) if weights is None else np.asarray(weights, np.float64)
    )
    kernel = norm.pdf(points[None], loc=samples[:, None], scale=std).prod(axis=-1)
    return (
        weights @ kernel / weights.sum() if weights.sum() > 0 else np.zeros(len(points))
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("n_points,n_samples", [(0, 5), (3, 0), (1, 1), (13, 9)])
@pytest.mark.parametrize("weight_kind", ["none", "zero", "weighted"])
def test_density_matches_independent_combined_kernel(
    x64, n_points, n_samples, weight_kind
):
    with precision_mode(x64):
        rng = np.random.default_rng(7422)
        points = jnp.asarray(rng.normal(size=(n_points, 5)))
        samples = jnp.asarray(rng.normal(size=(n_samples, 5)))
        std = jnp.asarray([0.7, 1.1, 0.9, 1.7, 0.6])
        weights = weights_for(n_samples, weight_kind)
        actual = streamed._sample_tiled_density(
            points, samples, std, weights, sample_tile_size=4, eval_tile_size=3
        )
        assert actual.shape == (n_points,)
        np.testing.assert_allclose(
            actual, density_oracle(points, samples, std, weights), **tolerance(x64)
        )
        baseline = kde(
            points,
            samples,
            std,
            jnp.ones(n_samples) if weights is None else weights,
        )
        assert actual.dtype == baseline.dtype
        np.testing.assert_allclose(actual, baseline, **tolerance(x64))


def joint_inputs(n_decoding, n_encoding, n_positions, weight_kind):
    rng = np.random.default_rng(7423)
    return {
        "decoding_features": jnp.asarray(rng.normal(size=(n_decoding, 3))),
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
    return np.log(np.maximum(intensity, EPS)), (mark, position, numerator, weights)


def baseline_joint(values):
    position_distance = jnp.exp(
        streamed._log_kernel_matrix(
            values["place_bin_centers"],
            values["encoding_positions"],
            values["position_std"],
        )
    )
    return jnp.clip(
        estimate_log_joint_mark_intensity(
            values["decoding_features"],
            values["encoding_features"],
            values["waveform_stds"],
            values["occupancy"],
            values["mean_rate"],
            position_distance,
            values["encoding_weights"],
        ),
        min=LOG_EPS,
    )


TILES = {"encoding_tile_size": 4, "position_tile_size": 3, "decoding_tile_size": 5}


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "shape", [(0, 7, 5), (1, 0, 5), (1, 7, 0), (1, 7, 5), (13, 9, 7)]
)
@pytest.mark.parametrize("weight_kind", ["none", "zero", "weighted"])
def test_joint_tiles_match_finished_linear_oracle(x64, shape, weight_kind):
    with precision_mode(x64):
        values = joint_inputs(*shape, weight_kind)
        actual = streamed._streamed_joint_mark_log_intensity(**values, **TILES)
        oracle, _ = joint_oracle(values)
        assert actual.shape == (shape[0], shape[2])
        np.testing.assert_allclose(actual, oracle, **tolerance(x64))
        np.testing.assert_allclose(actual, baseline_joint(values), **tolerance(x64))


def test_local_kernel_exponentiates_combined_dimensions_once():
    with precision_mode(False):
        points = jnp.array([[15.0, 0.0]])
        samples = jnp.array([[0.0, 0.0]])
        std = jnp.array([1.0, 1e-20])
        actual = streamed._sample_tiled_density(
            points, samples, std, sample_tile_size=2, eval_tile_size=2
        )
        oracle = density_oracle(points, samples, std, None)
        assert actual[0] > 0  # the separately exponentiated position kernel is zero
        np.testing.assert_allclose(actual / oracle, 1.0, **FLOAT32_ROUNDING)


@pytest.mark.parametrize("x64", [False, True])
def test_density_and_finished_joint_gradients_match_independent_oracle(x64):
    with precision_mode(x64):
        values = joint_inputs(13, 9, 7, "weighted")
        oracle, (mark, position, numerator, weights) = joint_oracle(values)
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
        expected = np.where((oracle > LOG_EPS)[:, :, None], ratio, 0).sum(axis=1)
        marks = values.pop("decoding_features")
        actual = jax.grad(
            lambda x: streamed._streamed_joint_mark_log_intensity(
                decoding_features=x, **values, **TILES
            ).sum()
        )(marks)
        np.testing.assert_allclose(actual, expected, **tolerance(x64))

        points = jnp.concatenate((marks, jnp.ones((13, 1))), axis=1)
        samples = jnp.concatenate(
            (values["encoding_features"], jnp.zeros((9, 1))), axis=1
        )
        std = jnp.concatenate((values["waveform_stds"], jnp.ones(1)))
        compiled_density = jax.jit(
            lambda x: streamed._sample_tiled_density(
                x,
                samples,
                std,
                values["encoding_weights"],
                sample_tile_size=4,
                eval_tile_size=3,
            )
        )(points)
        np.testing.assert_allclose(
            compiled_density,
            density_oracle(points, samples, std, values["encoding_weights"]),
            **tolerance(x64),
        )
        kernel = norm.pdf(
            np.asarray(points)[None],
            loc=np.asarray(samples)[:, None],
            scale=np.asarray(std),
        ).prod(axis=-1)
        expected_density_grad = (
            np.einsum(
                "e,ed,edf->df",
                weights,
                kernel,
                (np.asarray(samples)[:, None] - np.asarray(points)[None])
                / np.asarray(std) ** 2,
            )
            / weights.sum()
        )
        actual_density_grad = jax.grad(
            lambda x: streamed._sample_tiled_density(
                x,
                samples,
                std,
                values["encoding_weights"],
                sample_tile_size=4,
                eval_tile_size=3,
            ).sum()
        )(points)
        np.testing.assert_allclose(
            actual_density_grad, expected_density_grad, **tolerance(x64)
        )
        compiled_density_grad = jax.jit(
            jax.grad(
                lambda x: streamed._sample_tiled_density(
                    x,
                    samples,
                    std,
                    values["encoding_weights"],
                    sample_tile_size=4,
                    eval_tile_size=3,
                ).sum()
            )
        )(points)
        np.testing.assert_allclose(
            compiled_density_grad, expected_density_grad, **tolerance(x64)
        )


@pytest.mark.parametrize("weight_kind", ["none", "zero"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_inputs_keep_baseline_masks_and_final_floor(weight_kind, value):
    values = joint_inputs(7, 6, 5, weight_kind)
    values["encoding_features"] = values["encoding_features"].at[2, 0].set(value)
    actual = streamed._streamed_joint_mark_log_intensity(**values, **TILES)
    expected = baseline_joint(values)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING, equal_nan=True)


def test_float32_inputs_keep_enabled_x64_accumulator():
    with precision_mode(True):
        values = {
            key: (value.astype(jnp.float32) if hasattr(value, "dtype") else value)
            for key, value in joint_inputs(7, 6, 5, "weighted").items()
        }
        actual = streamed._streamed_joint_mark_log_intensity(**values, **TILES)
        expected = baseline_joint(values)
        assert actual.dtype == expected.dtype == jnp.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        density = streamed._sample_tiled_density(
            values["place_bin_centers"],
            values["encoding_positions"],
            values["position_std"],
            values["encoding_weights"],
            sample_tile_size=4,
            eval_tile_size=3,
        )
        reference_density = kde(
            values["place_bin_centers"],
            values["encoding_positions"],
            values["position_std"],
            values["encoding_weights"],
        )
        assert density.dtype == reference_density.dtype == jnp.float64
        np.testing.assert_allclose(density, reference_density, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("x64", [False, True])
def test_scalar_decoded_tail_matrix_and_ordered_row_gradients(x64):
    with precision_mode(x64):
        values = joint_inputs(13, 9, 7, "weighted")
        marks = values.pop("decoding_features")
        ids = np.arange(13) % 4
        tiles = TILES | {"decoding_tile_size": 4}
        expected = jax.grad(
            lambda x: baseline_joint(values | {"decoding_features": x}).sum()
        )(marks)
        actual = jax.grad(
            lambda x: streamed._streamed_joint_mark_log_intensity(
                decoding_features=x, **values, **tiles
            ).sum()
        )(marks)
        rows = jax.grad(
            lambda x: streamed._streamed_joint_mark_row_sums(
                decoding_features=x, **values, row_indices=ids, n_rows=4, **tiles
            ).sum()
        )(marks)
        np.testing.assert_allclose(actual, expected, **tolerance(x64))
        np.testing.assert_allclose(rows, expected, **tolerance(x64))


def test_row_wrapper_preserves_input_order_invalid_drop_and_replay():
    values = joint_inputs(13, 9, 7, "weighted")
    ids = np.array([2, -1, 2, 0, 99, 1, 2, 1, 3, 0, 1, 3, 2])
    individual = np.asarray(
        streamed._streamed_joint_mark_log_intensity(**values, **TILES)
    )
    expected = np.zeros((5, 7), individual.dtype)
    for row, intensity in zip(ids, individual, strict=True):
        if 0 <= row < 5:
            expected[row] += intensity
    actual = streamed._streamed_joint_mark_row_sums(
        **values, row_indices=ids, n_rows=5, **TILES
    )
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)
    reference = np.asarray(actual).copy()
    for _ in range(20):
        np.testing.assert_array_equal(
            streamed._streamed_joint_mark_row_sums(
                **values, row_indices=ids, n_rows=5, **TILES
            ),
            reference,
        )


def test_kernel_inputs_are_bounded_by_tiles_not_full_encoding(monkeypatch):
    calls = []
    original = streamed._log_kernel_matrix
    original_numerator = streamed._joint_numerator

    def bounded_numerator(*args):
        assert args[3].shape[0] <= args[-1], (
            "decoded tile still spans full spatial grid"
        )
        return original_numerator(*args)

    def record(points, samples, std):
        calls.append((len(samples), len(points)))
        assert len(samples) <= 8
        assert len(points) <= 7
        return original(points, samples, std)

    monkeypatch.setattr(streamed, "_log_kernel_matrix", record)
    monkeypatch.setattr(streamed, "_joint_numerator", bounded_numerator)
    values = joint_inputs(23, 49, 37, "weighted")
    actual = streamed._streamed_joint_mark_row_sums(
        **values,
        row_indices=np.arange(23) % 5,
        n_rows=5,
        encoding_tile_size=8,
        position_tile_size=7,
        decoding_tile_size=5,
    )
    actual.block_until_ready()
    assert calls
    assert actual.shape == (5, 37)
    density = streamed._sample_tiled_density(
        values["decoding_features"],
        values["encoding_features"],
        values["waveform_stds"],
        values["encoding_weights"],
        sample_tile_size=8,
        eval_tile_size=7,
    )
    density.block_until_ready()
    assert density.shape == (23,)


def test_row_wrapper_drops_unsigned_ids_before_device_integer_conversion():
    values = joint_inputs(3, 9, 7, "weighted")
    actual = streamed._streamed_joint_mark_row_sums(
        **values,
        row_indices=np.array([0, 2**32, 2**64 - 1], np.uint64),
        n_rows=2,
        **TILES,
    )
    individual = streamed._streamed_joint_mark_log_intensity(**values, **TILES)
    np.testing.assert_array_equal(actual[0], individual[0])
    np.testing.assert_array_equal(actual[1], np.zeros(7))


@pytest.mark.parametrize("name", ["sample_tile_size", "eval_tile_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_density_rejects_invalid_tile_configuration(name, value):
    tiles = {"sample_tile_size": 2, "eval_tile_size": 2, name: value}
    with pytest.raises(ValueError, match="positive integer"):
        streamed._sample_tiled_density(
            jnp.zeros((2, 1)), jnp.ones((3, 1)), 1.0, **tiles
        )
