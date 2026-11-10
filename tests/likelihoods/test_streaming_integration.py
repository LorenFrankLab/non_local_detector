"""Streaming position kernels in ``estimate_log_joint_mark_intensity``.

Streaming computes each encoding tile's position kernel on the fly instead of
materializing the full ``(n_encoding_spikes, n_position_bins)`` matrix. Extreme
mark distances and zero-occupancy bins must still give the joint mark
intensity, checked against an independent float64 log-space reference, and
missing streaming inputs are rejected. The tiling sweep is in
``test_kde_streaming_tiles``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
)
from tests.likelihoods._kde_reference import (
    reference_log_joint_mark_intensity,
    reference_log_position_kernel,
)
from tests.likelihoods.conftest import realistic_log_background

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

    Returns
    -------
    reference : np.ndarray, shape (n_decoding_spikes, n_position_bins)
    """
    return reference_log_joint_mark_intensity(
        inputs["decoding_features"],
        inputs["encoding_features"],
        inputs["waveform_stds"],
        inputs["occupancy"],
        MEAN_RATE,
        reference_log_position_kernel(
            inputs["position_eval_points"],
            inputs["encoding_positions"],
            inputs["position_std"],
        ),
        log_background=log_background,
    )


def _streaming(inputs, log_background, **tiling):
    """The streaming estimate over one block of decoding spikes."""
    return np.asarray(
        estimate_log_joint_mark_intensity(
            inputs["decoding_features"],
            inputs["encoding_features"],
            inputs["waveform_stds"],
            inputs["occupancy"],
            MEAN_RATE,
            log_position_distance=None,
            use_streaming=True,
            encoding_positions=inputs["encoding_positions"],
            position_eval_points=inputs["position_eval_points"],
            position_std=inputs["position_std"],
            log_background=log_background,
            **tiling,
        )
    )


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
        _estimate(_inputs(123, 1), enc_tile_size=5)
