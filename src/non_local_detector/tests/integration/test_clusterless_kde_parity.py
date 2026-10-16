"""Log-space clusterless KDE estimator on the reference (non-GEMM) path.

``clusterless_kde`` is the log-space estimator; these check that
the log-space joint mark intensity stays finite where a probability-space
kernel would underflow, and that a moderately large problem runs.
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
from non_local_detector.tests.likelihoods.conftest import realistic_log_background


@pytest.mark.integration
def test_estimate_intensity_extreme_features():
    """Joint mark intensity stays finite with extreme feature distances.

    The mark kernels underflow in probability space; the log-space estimator
    must still return finite values of the right shape.
    """
    n_enc_spikes = 30
    n_dec_spikes = 10
    n_pos_bins = 20
    n_features = 4

    # Extreme feature values (large distances)
    rng = np.random.default_rng(42)
    dec_features = jnp.array(rng.standard_normal((n_dec_spikes, n_features)) * 50 + 100)
    enc_features = jnp.array(rng.standard_normal((n_enc_spikes, n_features)) * 50 + 200)
    jnp.ones(n_enc_spikes)
    waveform_stds = jnp.array([10.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 2.0

    # Position distance
    enc_positions = jnp.array(rng.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])

    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    ll_log = estimate_log(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=False,
        log_background=realistic_log_background(dec_features),
    )

    assert ll_log.shape == (n_dec_spikes, n_pos_bins), "Shape is wrong"
    assert np.all(np.isfinite(ll_log)), (
        "Log-space has non-finite values even with extreme features"
    )


@pytest.mark.integration
def test_memory_efficiency_scan_vs_vmap():
    """Verify that scan-based implementation doesn't materialize huge arrays.

    This test doesn't check memory directly, but verifies the implementation
    runs without OOM on moderately large problem sizes that would fail with vmap.
    """
    # Problem size that would OOM with vmap but not with scan
    n_enc_spikes = 500
    n_dec_spikes = 100
    n_pos_bins = 200
    n_features = 4

    rng = np.random.default_rng(42)
    dec_features = jnp.array(rng.standard_normal((n_dec_spikes, n_features)) * 10 + 50)
    enc_features = jnp.array(rng.standard_normal((n_enc_spikes, n_features)) * 10 + 50)
    waveform_stds = jnp.array([5.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 5.0

    # Position distance
    enc_positions = jnp.array(rng.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # This should not OOM with vmap-based implementation
    ll_log = estimate_log(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=False,
        log_background=realistic_log_background(dec_features),
    )

    # Verify result
    assert ll_log.shape == (n_dec_spikes, n_pos_bins), "Shape is wrong"
    assert np.all(np.isfinite(ll_log)), "Result has non-finite values"
