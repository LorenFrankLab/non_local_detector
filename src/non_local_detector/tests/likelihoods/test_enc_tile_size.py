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


def test_enc_tile_size_equivalence():
    """Verify enc_tile_size produces same results as no tiling."""
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

    # With encoding tiling (chunk size 25)
    result_with_enc_tiling = estimate_log_joint_mark_intensity(
        dec_features,
        enc_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=None,
        enc_tile_size=25,
    )

    # Should be numerically equivalent
    assert np.allclose(
        result_no_tiling, result_with_enc_tiling, rtol=1e-5, atol=1e-8
    ), f"Max diff: {np.max(np.abs(result_no_tiling - result_with_enc_tiling))}"

    # Check shapes
    assert result_no_tiling.shape == (n_dec_spikes, n_pos_bins)
    assert result_with_enc_tiling.shape == (n_dec_spikes, n_pos_bins)

    # Check all values are finite
    assert np.all(np.isfinite(result_no_tiling))
    assert np.all(np.isfinite(result_with_enc_tiling))

    print(f"✓ enc_tile_size=25 matches no tiling")
    print(f"  Shape: {result_no_tiling.shape}")
    print(f"  Max diff: {np.max(np.abs(result_no_tiling - result_with_enc_tiling)):.2e}")


def test_enc_tile_size_with_pos_tile_size():
    """Test combined encoding and position tiling."""
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
        pos_tile_size=15,  # Tile positions
        enc_tile_size=30,  # Tile encoding spikes
    )

    # Should match
    assert np.allclose(
        result_baseline, result_both_tiling, rtol=1e-5, atol=1e-8
    ), f"Max diff: {np.max(np.abs(result_baseline - result_both_tiling))}"

    print(f"✓ Combined enc_tile_size=30 + pos_tile_size=15 matches baseline")
    print(f"  Max diff: {np.max(np.abs(result_baseline - result_both_tiling)):.2e}")


if __name__ == "__main__":
    test_enc_tile_size_equivalence()
    test_enc_tile_size_with_pos_tile_size()
    print("\n✅ All enc_tile_size tests passed!")
