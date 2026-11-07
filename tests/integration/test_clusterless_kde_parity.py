"""Clusterless KDE joint mark intensity in the far mark tails.

When the decoding marks sit far from the encoding marks, most kernel
products underflow float32, so summing them in linear space would return
only the background there. The GEMM path must still match a float64
log-sum-exp of the same kernels.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_log,
)
from non_local_detector.likelihoods.clusterless_kde import (
    log_kde_distance,
)
from tests.likelihoods._kde_reference import reference_log_joint_mark_intensity

pytestmark = pytest.mark.unit


def test_gemm_matches_float64_reference_where_linear_kernels_underflow():
    n_enc_spikes = 30
    n_dec_spikes = 10
    n_pos_bins = 20
    n_features = 4

    # Decoding and encoding marks are far apart: the largest log mark kernel
    # is about -46 and most are below -100, beyond float32 range.
    rng = np.random.default_rng(42)
    dec_features = (rng.standard_normal((n_dec_spikes, n_features)) * 50 + 100).astype(
        np.float32
    )
    enc_features = (rng.standard_normal((n_enc_spikes, n_features)) * 50 + 200).astype(
        np.float32
    )
    waveform_stds = np.full(n_features, 10.0, dtype=np.float32)
    occupancy = np.full(n_pos_bins, 0.1, dtype=np.float32)
    mean_rate = 2.0

    enc_positions = jnp.array(rng.uniform(0, 100, (n_enc_spikes, 1)), jnp.float32)
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins), jnp.float32)[:, None]
    log_position_distance = log_kde_distance(
        interior_bins, enc_positions, jnp.array([5.0])
    )

    log_intensity = reference_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
    )  # (n_dec_spikes, n_pos_bins)
    # A background 30 nats below each spike's smallest intensity: low enough
    # that the result is the mark intensity, not the floor.
    log_background = log_intensity.min(axis=1) - 30.0
    expected = np.logaddexp(log_intensity, log_background[:, None])

    # Premise: most intensities are below the smallest float32 subnormal
    # (log 1.4e-45 = -103.3), where every kernel product underflows.
    assert (log_intensity < -110.0).mean() > 0.5

    result = np.asarray(
        estimate_log(
            jnp.asarray(dec_features),
            jnp.asarray(enc_features),
            jnp.asarray(waveform_stds),
            jnp.asarray(occupancy),
            mean_rate,
            log_position_distance,
            log_background=jnp.asarray(log_background, jnp.float32),
        )
    )
    assert result.shape == (n_dec_spikes, n_pos_bins)
    assert np.all(result - log_background[:, None] > 25.0)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-4)
