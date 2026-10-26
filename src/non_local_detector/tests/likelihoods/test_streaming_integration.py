"""Streaming position kernels in ``estimate_log_joint_mark_intensity``.

Streaming computes each encoding tile's position kernel on the fly instead of
materializing the full ``(n_encoding_spikes, n_position_bins)`` matrix. Every
tiling must still give the joint mark intensity, which is checked against an
independent float64 log-space reference built with SciPy.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
)
from non_local_detector.tests.likelihoods.conftest import realistic_log_background

pytestmark = pytest.mark.unit

MEAN_RATE = 2.5
N_ENCODING_SPIKES = 50
# Float32 tiles against the float64 reference: measured at most 7e-6 absolute
# on log intensities of order 10 and 9e-5 on order 1000 (extreme marks). A
# dropped or double-counted encoding tile moves an entry by O(0.1) or more.
REFERENCE_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}


def _inputs(
    seed,
    n_position_dims,
    *,
    n_decoding_spikes=8,
    n_position_bins=30,
    n_features=3,
    feature_scale=1.0,
    waveform_std=1.0,
):
    """Random encoding/decoding marks and positions, as keyword arguments."""
    rng = np.random.default_rng(seed)
    return {
        "decoding_features": jnp.asarray(
            rng.standard_normal((n_decoding_spikes, n_features)) * feature_scale
        ),
        "encoding_features": jnp.asarray(
            rng.standard_normal((N_ENCODING_SPIKES, n_features)) * feature_scale
        ),
        "waveform_stds": jnp.asarray(rng.uniform(0.8, 1.5, n_features) * waveform_std),
        "occupancy": jnp.asarray(rng.random(n_position_bins) * 0.7 + 0.2),
        "encoding_positions": jnp.asarray(
            rng.standard_normal((N_ENCODING_SPIKES, n_position_dims))
        ),
        "position_eval_points": jnp.asarray(
            rng.standard_normal((n_position_bins, n_position_dims))
        ),
        "position_std": jnp.asarray(rng.uniform(0.8, 1.2, n_position_dims)),
    }


def _float64_reference(inputs, log_background):
    """``logaddexp(log intensity, log background)`` per spike and bin, float64.

    The intensity is ``mean_rate * mean_e K_mark(m, m_e) K_pos(x, x_e) /
    occupancy(x)`` with Gaussian kernels.

    Returns
    -------
    reference : np.ndarray, shape (n_decoding_spikes, n_position_bins)
    """
    as64 = {key: np.asarray(value, dtype=np.float64) for key, value in inputs.items()}
    log_mark = norm.logpdf(
        as64["decoding_features"][:, None],
        as64["encoding_features"][None],
        as64["waveform_stds"],
    ).sum(axis=-1)
    log_position = norm.logpdf(
        as64["encoding_positions"][:, None],
        as64["position_eval_points"][None],
        as64["position_std"],
    ).sum(axis=-1)
    log_intensity = (
        np.log(MEAN_RATE)
        + logsumexp(log_mark[:, :, None] + log_position[None], axis=1)
        - np.log(N_ENCODING_SPIKES)
        - np.log(as64["occupancy"])
    )
    return np.logaddexp(
        log_intensity, np.asarray(log_background, dtype=np.float64)[:, None]
    )


def _streaming(inputs, log_background, *, block_size=None, **tiling):
    """The streaming estimate, blocked over decoding spikes if ``block_size``."""
    args = (
        inputs["decoding_features"],
        inputs["encoding_features"],
        inputs["waveform_stds"],
        inputs["occupancy"],
        MEAN_RATE,
    )
    kwargs = {
        "log_position_distance": None,
        "use_gemm": True,
        "use_streaming": True,
        "encoding_positions": inputs["encoding_positions"],
        "position_eval_points": inputs["position_eval_points"],
        "position_std": inputs["position_std"],
        "log_background": log_background,
        **tiling,
    }
    if block_size is None:
        return np.asarray(estimate_log_joint_mark_intensity(*args, **kwargs))
    return np.asarray(
        block_estimate_log_joint_mark_intensity(*args, block_size=block_size, **kwargs)
    )


@pytest.mark.parametrize(
    ("enc_tile_size", "pos_tile_size", "n_position_dims", "block_size"),
    [
        (25, None, 2, None),
        (20, 15, 2, None),
        (10, None, 1, None),
        (49, None, 1, None),
        # One encoding spike per tile.
        (1, None, 1, None),
        # A tile as large as the encoding set cannot be chunked; streaming
        # falls back to one materialized kernel instead of raising.
        (N_ENCODING_SPIKES, None, 2, None),
        # Blocked over decoding spikes (8 spikes in blocks of 3).
        (20, None, 2, 3),
        (25, 15, 2, 3),
    ],
)
def test_streaming_matches_float64_reference(
    enc_tile_size, pos_tile_size, n_position_dims, block_size
):
    """Every encoding, position and decoding tiling gives the joint intensity."""
    inputs = _inputs(enc_tile_size, n_position_dims)
    log_background = realistic_log_background(inputs["decoding_features"])

    actual = _streaming(
        inputs,
        log_background,
        block_size=block_size,
        enc_tile_size=enc_tile_size,
        pos_tile_size=pos_tile_size,
    )

    expected = _float64_reference(inputs, log_background)
    # The background must not mask the intensity, or the comparison is vacuous.
    assert np.all(expected > np.asarray(log_background)[:, None] + 1.0)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, **REFERENCE_TOLERANCE)


def test_streaming_keeps_extreme_mark_distances_finite():
    """Marks hundreds of kernel widths apart give a finite log intensity.

    The background is ``-inf`` so the output is the intensity itself: a
    finite background would hide an intensity that underflowed to ``-inf``.
    """
    inputs = _inputs(789, 2, feature_scale=10.0, waveform_std=0.5)
    log_background = jnp.full(inputs["decoding_features"].shape[0], -jnp.inf)

    actual = _streaming(inputs, log_background, enc_tile_size=10)

    expected = _float64_reference(inputs, log_background)
    # Some intensities are below float32's smallest normal number, where a
    # linear-space product would underflow to zero.
    assert np.any(expected < np.log(np.finfo(np.float32).tiny))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, **REFERENCE_TOLERANCE)


def test_streaming_zero_occupancy_bins_score_the_background():
    """Zero-occupancy bins have no intensity, so each spike scores its background
    there (finite, not ``-inf``); the other bins keep their intensity."""
    inputs = _inputs(101, 1, n_position_bins=8)
    inputs["occupancy"] = jnp.array([0.0, 0.5, 0.3, 0.0, 0.6, 0.2, 0.0, 0.4])
    log_background = realistic_log_background(inputs["decoding_features"])

    actual = _streaming(inputs, log_background, enc_tile_size=8)

    zero_occupancy = np.asarray(inputs["occupancy"]) == 0.0
    np.testing.assert_allclose(
        actual[:, zero_occupancy],
        np.broadcast_to(np.asarray(log_background)[:, None], (8, 3)),
        rtol=1e-6,
    )
    with np.errstate(divide="ignore"):  # log(0) occupancy, not compared
        expected = _float64_reference(inputs, log_background)
    np.testing.assert_allclose(
        actual[:, ~zero_occupancy], expected[:, ~zero_occupancy], **REFERENCE_TOLERANCE
    )


def _estimate(inputs, **kwargs):
    return estimate_log_joint_mark_intensity(
        inputs["decoding_features"],
        inputs["encoding_features"],
        inputs["waveform_stds"],
        inputs["occupancy"],
        MEAN_RATE,
        log_position_distance=None,
        use_streaming=True,
        log_background=realistic_log_background(inputs["decoding_features"]),
        **kwargs,
    )


def test_streaming_requires_enc_tile_size():
    inputs = _inputs(42, 1)
    with pytest.raises(ValueError, match="use_streaming=True requires enc_tile_size"):
        _estimate(
            inputs,
            use_gemm=True,
            enc_tile_size=None,
            encoding_positions=inputs["encoding_positions"],
            position_eval_points=inputs["position_eval_points"],
            position_std=inputs["position_std"],
        )


def test_streaming_requires_position_params():
    with pytest.raises(
        ValueError,
        match="use_streaming=True requires encoding_positions, position_eval_points",
    ):
        _estimate(_inputs(123, 1), use_gemm=True, enc_tile_size=5)


def test_streaming_with_use_gemm_false_raises_clear_error():
    """The linear-space (``use_gemm=False``) reference path does not stream; the
    combination is rejected up front rather than failing later with a
    confusing ``exp(None)`` TypeError."""
    inputs = _inputs(2024, 1)
    with pytest.raises(ValueError, match="use_gemm"):
        _estimate(
            inputs,
            use_gemm=False,
            enc_tile_size=5,
            encoding_positions=inputs["encoding_positions"],
            position_eval_points=inputs["position_eval_points"],
            position_std=inputs["position_std"],
        )
