"""Waveform offsets must preserve Gaussian distances and decoded evidence."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp  # type: ignore[import-untyped]

from non_local_detector import ClusterlessDecoder
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    _compute_log_mark_kernel_gemm,
    estimate_log_joint_mark_intensity,
)
from non_local_detector.tests.conftest import precision_mode
from non_local_detector.tests.likelihoods.conftest import (
    realistic_log_background,
    tolerance,
)

pytestmark = pytest.mark.unit


def _assert_allclose(actual, expected, x64=False):
    bounds = tolerance(x64)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=bounds["rtol"],
        atol=bounds["atol"],
    )


def _gaussian_log_kernel(decoding, encoding, std):
    """Float64 reference formed from each pair's original feature difference."""
    difference = (
        np.asarray(encoding, dtype=np.float64)[:, None, :]
        - np.asarray(decoding, dtype=np.float64)[None, :, :]
    ) / np.asarray(std, dtype=np.float64)
    return -0.5 * np.sum(difference**2, axis=-1) - np.sum(
        np.log(np.asarray(std, dtype=np.float64) * np.sqrt(2 * np.pi))
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("n_features", [1, 4, 10])
def test_mark_kernel_preserves_nearby_distances_in_separated_clusters(x64, n_features):
    """A common origin cannot resolve nearby marks in every distant cluster."""
    with precision_mode(x64):
        dtype = "float64" if x64 else "float32"
        direction = np.where(np.arange(n_features) % 2, -1, 1).astype(dtype)
        encoding = np.array([0, 100000, 100024], dtype=dtype)[:, None] * direction
        decoding = np.array([100000, 100012, 0], dtype=dtype)[:, None] * direction
        std = np.full(n_features, 24.0, dtype=dtype)
        expected = _gaussian_log_kernel(decoding, encoding, std)
        actual = jax.jit(_compute_log_mark_kernel_gemm)(
            jnp.asarray(decoding), jnp.asarray(encoding), jnp.asarray(std)
        )
        assert np.isfinite(actual).all()
        _assert_allclose(actual, expected, x64)


def test_mark_kernel_identical_large_features_do_not_produce_nan():
    """Squaring the feature magnitude must not overflow before subtraction."""
    with precision_mode(False):
        base = np.float32(1e20)
        encoding = np.array([base, np.nextafter(base, np.float32(np.inf))])[:, None]
        decoding = np.array([[base]], dtype=np.float32)
        std = np.ones(1, dtype=np.float32)
        actual = jax.jit(_compute_log_mark_kernel_gemm)(
            jnp.asarray(decoding), jnp.asarray(encoding), jnp.asarray(std)
        )
        assert np.isfinite(actual).all()
        _assert_allclose(actual, _gaussian_log_kernel(decoding, encoding, std))


@pytest.mark.parametrize("x64", [False, True])
def test_mark_kernel_opposite_finite_features_with_large_bandwidth(x64):
    """A broad kernel can have finite distances even if feature subtraction overflows.

    The bandwidth keeps ``1 / sigma`` a normal float. From JAX 0.12, XLA
    divides by a scalar bandwidth by multiplying with its reciprocal, so a
    bandwidth within a factor of about 4 of the dtype's maximum flushes every
    distance to zero, overflowing or not.
    """
    with precision_mode(x64):
        dtype = "float64" if x64 else "float32"
        magnitude = np.float64(1e308) if x64 else np.float32(3e38)
        encoding = np.array([[magnitude], [-magnitude]], dtype=dtype)
        decoding = encoding[::-1].copy()
        sigma = magnitude / 16
        std = np.array([sigma], dtype=dtype)
        # The unlike pairs are exactly 32 standard deviations apart. Form the
        # normalizer separately: sigma * sqrt(2*pi) would overflow in float64.
        normalizer = -np.log(float(sigma)) - 0.5 * np.log(2 * np.pi)
        expected = normalizer - 0.5 * np.array([[32.0**2, 0.0], [0.0, 32.0**2]])
        actual = jax.jit(_compute_log_mark_kernel_gemm)(
            jnp.asarray(decoding), jnp.asarray(encoding), jnp.asarray(std)
        )
        assert np.isfinite(actual).all()
        _assert_allclose(actual, expected, x64)


@pytest.mark.parametrize(
    "dtype,encoding_value,decoding_value",
    [(np.uint16, 100, 124), (np.int16, 20000, -20000)],
)
def test_mark_kernel_integer_features_do_not_wrap(
    dtype, encoding_value, decoding_value
):
    """Raw ADC marks must be promoted before unsigned/signed subtraction."""
    encoding = np.array([[encoding_value]], dtype=dtype)
    decoding = np.array([[decoding_value]], dtype=dtype)
    std = np.array([24.0], dtype=np.float32)
    actual = jax.jit(_compute_log_mark_kernel_gemm)(
        jnp.asarray(decoding), jnp.asarray(encoding), jnp.asarray(std)
    )
    _assert_allclose(actual, _gaussian_log_kernel(decoding, encoding, std))


@pytest.mark.parametrize("n_features", [4, 10])
@pytest.mark.parametrize("path", ["full", "chunked", "streaming"])
def test_mark_offsets_match_weighted_gaussian_reference_on_every_path(n_features, path):
    """Both marginal algorithms retain weights and distances through tiling."""
    direction = np.where(np.arange(n_features) % 2, -1, 1).astype(np.float32)
    encoding = (
        np.array(
            [0, 24, 48, 100000, 100024, 100048, 100000, 100024, 100048],
            dtype=np.float32,
        )[:, None]
        * direction
    )
    decoding = np.array([100000, 100012], dtype=np.float32)[:, None] * direction
    std = np.full(n_features, 24.0, dtype=np.float32)
    encoding_position = np.linspace(0, 10, len(encoding), dtype=np.float32)[:, None]
    evaluation_position = np.linspace(0, 10, 5, dtype=np.float32)[:, None]
    position_std = np.array([2.0], dtype=np.float32)
    log_position = _gaussian_log_kernel(
        evaluation_position, encoding_position, position_std
    )
    weights = np.array([1, 0.5, 0, 2, 1, 0.25, 0.5, 0, 1], dtype=np.float32)
    log_weights = np.full(len(weights), -np.inf)
    log_weights[weights > 0] = np.log(
        weights[weights > 0].astype(float) / weights.sum()
    )
    # Keep every intensity above log_background + log(eps), the range where it
    # changes the spike's term and the estimate must be exact.
    occupancy = np.full(len(evaluation_position), 1e-6, dtype=np.float32)
    mean_rate = 5.0
    log_background = np.asarray(realistic_log_background(decoding), dtype=np.float64)
    log_intensity = (
        np.log(mean_rate)
        + logsumexp(
            _gaussian_log_kernel(decoding, encoding, std)[:, :, None]
            + log_position[:, None, :]
            + log_weights[:, None, None],
            axis=0,
        )
        - np.log(occupancy.astype(float))
    )
    assert (
        log_intensity > log_background[:, None] + np.log(np.finfo(np.float32).eps)
    ).all()
    expected = np.logaddexp(log_intensity, log_background[:, None])
    policy: dict[str, Any] = {}
    if path != "full":
        policy["enc_tile_size"] = 4
    if path == "streaming":
        policy.update(
            use_streaming=True,
            encoding_positions=jnp.asarray(encoding_position),
            position_eval_points=jnp.asarray(evaluation_position),
            position_std=jnp.asarray(position_std),
        )
    actual = estimate_log_joint_mark_intensity(
        jnp.asarray(decoding),
        jnp.asarray(encoding),
        jnp.asarray(std),
        jnp.asarray(occupancy),
        mean_rate,
        None if path == "streaming" else jnp.asarray(log_position),
        encoding_weights=jnp.asarray(weights),
        log_background=jnp.asarray(log_background, dtype=jnp.float32),
        **policy,
    )
    assert np.isfinite(actual).all()
    _assert_allclose(actual, expected)


@pytest.mark.parametrize("path", ["full", "chunked", "streaming"])
def test_public_decoder_posterior_is_invariant_to_waveform_translation(path):
    """Adding the same offset to training/decode marks preserves the posterior."""
    position_time = np.linspace(0, 10, 101)
    position = position_time[:, None]
    policy: dict[str, Any] = {}
    if path != "full":
        policy["enc_tile_size"] = 1
    if path == "streaming":
        policy["use_streaming"] = True

    def decode(offset):
        environment = Environment(place_bin_size=1.0, position_range=((0.0, 10.0),))
        decoder = ClusterlessDecoder(
            environments=[environment],
            infer_track_interior=False,
            clusterless_algorithm="clusterless_kde",
            clusterless_algorithm_params={
                "position_std": 1.0,
                "waveform_std": 24.0,
                "block_size": 16,
                "disable_progress_bar": True,
                **policy,
            },
        ).fit(
            position_time,
            position,
            [np.array([2.0, 8.0])],
            [np.array([[offset], [offset + 24.0]])],
        )
        result = decoder.predict(
            [np.array([0.001])],
            [np.array([[offset]])],
            time_edges=np.array([0.0, 0.002]),
        )
        return np.asarray(result.acausal_posterior)

    expected = decode(0.0)
    actual = decode(100000.0)
    assert np.isfinite(actual).all()
    _assert_allclose(actual.sum(), 1.0)
    _assert_allclose(actual, expected)
