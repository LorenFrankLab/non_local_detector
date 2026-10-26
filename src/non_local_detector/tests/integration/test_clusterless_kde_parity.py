"""Clusterless KDE joint mark intensity in the far mark tails.

When the decoding marks sit far from the encoding marks, most kernel
products underflow float32 and the linear-space reference
(``use_gemm=False``) path returns only the background there. The default GEMM
path must still match a float64 log-sum-exp of the same kernels.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_log,
)
from non_local_detector.likelihoods.clusterless_kde import (
    log_kde_distance,
)

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

    # float64 reference: log(mean_rate * mean_e K_mark K_pos / occupancy).
    log_mark = norm.logpdf(
        dec_features[None].astype(np.float64),
        enc_features[:, None].astype(np.float64),
        waveform_stds.astype(np.float64),
    ).sum(axis=-1)  # (n_enc_spikes, n_dec_spikes)
    log_intensity = (
        np.log(mean_rate)
        + logsumexp(
            log_mark[:, :, None]
            + np.asarray(log_position_distance, np.float64)[:, None, :],
            axis=0,
        )
        - np.log(n_enc_spikes)
        - np.log(occupancy.astype(np.float64))
    )  # (n_dec_spikes, n_pos_bins)
    # A background 30 nats below each spike's smallest intensity: low enough
    # that the result is the mark intensity, not the floor.
    log_background = log_intensity.min(axis=1) - 30.0
    expected = np.logaddexp(log_intensity, log_background[:, None])

    def estimate(use_gemm):
        return np.asarray(
            estimate_log(
                jnp.asarray(dec_features),
                jnp.asarray(enc_features),
                jnp.asarray(waveform_stds),
                jnp.asarray(occupancy),
                mean_rate,
                log_position_distance,
                use_gemm=use_gemm,
                log_background=jnp.asarray(log_background, jnp.float32),
            )
        )

    # Premise: wherever the intensity is below the smallest float32 subnormal
    # (log 1.4e-45 = -103.3) every kernel product underflows, and the
    # linear-space reference returns only the background. That is most bins.
    underflow = log_intensity < -110.0
    assert underflow.mean() > 0.5
    linear = estimate(use_gemm=False)
    np.testing.assert_allclose(
        linear[underflow],
        np.broadcast_to(log_background[:, None], expected.shape)[underflow],
        rtol=1e-6,
    )

    result = estimate(use_gemm=True)
    assert result.shape == (n_dec_spikes, n_pos_bins)
    assert np.all(result - log_background[:, None] > 25.0)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-4)
