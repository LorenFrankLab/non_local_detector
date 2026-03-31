"""Test encoding spike tiling with enc_tile_size parameter.

This module tests the memory optimization feature that chunks computation
over encoding spikes using online logsumexp accumulation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity,
    log_kde_distance,
)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [10, 25, 50, 99])
def test_enc_tile_size_equivalence(enc_tile_size):
    """Verify enc_tile_size produces same results as no tiling.

    Parameters
    ----------
    enc_tile_size : int
        Encoding chunk size to test
    """
    n_enc_spikes = 100  # Large enough to test chunking
    n_dec_spikes = 20
    n_pos_bins = 30
    n_features = 4

    np.random.seed(42)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 10 + 50)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 10 + 50)
    waveform_stds = jnp.array([5.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 5.0

    # Compute position distance
    enc_positions = jnp.array(np.random.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # No tiling (baseline)
    result_no_tiling = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=None,
        enc_tile_size=None,
    )

    # With encoding tiling
    result_with_enc_tiling = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=None,
        enc_tile_size=enc_tile_size,
    )

    # Should be numerically equivalent
    max_diff = np.max(np.abs(result_no_tiling - result_with_enc_tiling))
    assert np.allclose(
        result_no_tiling, result_with_enc_tiling, rtol=1e-5, atol=1e-7
    ), f"enc_tile_size={enc_tile_size}: Max diff = {max_diff}"

    # Check shapes
    assert result_no_tiling.shape == (n_dec_spikes, n_pos_bins)
    assert result_with_enc_tiling.shape == (n_dec_spikes, n_pos_bins)

    # Check all values are finite
    assert np.all(np.isfinite(result_no_tiling))
    assert np.all(np.isfinite(result_with_enc_tiling))


@pytest.mark.unit
@pytest.mark.parametrize(
    "enc_tile_size,pos_tile_size",
    [
        (30, 15),  # Both tiling
        (25, 40),  # pos_tile_size > n_pos (no position tiling)
        (150, 20),  # enc_tile_size > n_enc (no encoding tiling)
    ],
)
def test_enc_tile_size_with_pos_tile_size(enc_tile_size, pos_tile_size):
    """Test combined encoding and position tiling.

    Parameters
    ----------
    enc_tile_size : int
        Encoding chunk size
    pos_tile_size : int
        Position chunk size
    """
    n_enc_spikes = 100
    n_dec_spikes = 15
    n_pos_bins = 40
    n_features = 4

    np.random.seed(123)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 10 + 50)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 10 + 50)
    waveform_stds = jnp.array([5.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.1
    mean_rate = 5.0

    # Position distance
    enc_positions = jnp.array(np.random.uniform(0, 100, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 100, n_pos_bins))[:, None]
    position_std = jnp.array([5.0])
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # No tiling
    result_baseline = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=None,
        enc_tile_size=None,
    )

    # Both enc and pos tiling
    result_both_tiling = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=pos_tile_size,
        enc_tile_size=enc_tile_size,
    )

    # Should match
    max_diff = np.max(np.abs(result_baseline - result_both_tiling))
    assert np.allclose(result_baseline, result_both_tiling, rtol=1e-5, atol=1e-7), (
        f"enc_tile_size={enc_tile_size}, pos_tile_size={pos_tile_size}: Max diff = {max_diff}"
    )


@pytest.mark.unit
def test_enc_tile_size_edge_cases():
    """Test edge cases for encoding tiling."""
    n_enc_spikes = 10
    n_dec_spikes = 5
    n_pos_bins = 8
    n_features = 2

    np.random.seed(456)
    dec_features = jnp.array(np.random.randn(n_dec_spikes, n_features) * 5)
    enc_features = jnp.array(np.random.randn(n_enc_spikes, n_features) * 5)
    waveform_stds = jnp.array([2.0] * n_features)
    occupancy = jnp.ones(n_pos_bins) * 0.05
    mean_rate = 2.0

    enc_positions = jnp.array(np.random.uniform(0, 50, (n_enc_spikes, 1)))
    interior_bins = jnp.array(np.linspace(0, 50, n_pos_bins))[:, None]
    position_std = jnp.array([3.0])
    log_position_distance = log_kde_distance(interior_bins, enc_positions, position_std)

    # Baseline
    result_baseline = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        enc_tile_size=None,
    )

    # Test: enc_tile_size = 1 (smallest possible)
    result_tile1 = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        enc_tile_size=1,
    )
    assert np.allclose(result_baseline, result_tile1, rtol=1e-5, atol=1e-7)

    # Test: enc_tile_size = n_enc (no chunking)
    result_tile_full = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        enc_tile_size=n_enc_spikes,
    )
    assert np.allclose(result_baseline, result_tile_full, rtol=1e-5, atol=1e-7)
