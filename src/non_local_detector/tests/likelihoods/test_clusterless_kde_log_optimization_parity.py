"""Test that optimized log-space version uses all optimizations correctly."""

import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    block_estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
)


@pytest.mark.parametrize("pos_tile_size", [None, 10, 50])
def test_pos_tiling_matches_no_tiling(simple_1d_environment, pos_tile_size):
    """Test that position tiling produces same results as no tiling."""
    env = simple_1d_environment
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]

    enc_times = [np.array([2.0, 5.0, 7.5])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=enc_times,
        spike_waveform_features=enc_feats,
        environment=env,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    dec_feats = np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)

    is_track_interior = env.is_track_interior_.ravel()
    interior_place_bin_centers = env.place_bin_centers_[is_track_interior]

    from non_local_detector.likelihoods.clusterless_kde_log import kde_distance

    electrode_encoding_positions = encoding["encoding_positions"][0]
    electrode_encoding_features = encoding["encoding_spike_waveform_features"][0]

    position_distance = kde_distance(
        interior_place_bin_centers,
        electrode_encoding_positions,
        std=encoding["position_std"],
    )

    # Baseline: no tiling
    result_no_tile = block_estimate_log_joint_mark_intensity(
        dec_feats,
        electrode_encoding_features,
        encoding["waveform_std"],
        encoding["occupancy"],
        encoding["mean_rates"][0],
        position_distance,
        block_size=8,
        use_gemm=True,
        pos_tile_size=None,
    )

    # With tiling
    result_tiled = block_estimate_log_joint_mark_intensity(
        dec_feats,
        electrode_encoding_features,
        encoding["waveform_std"],
        encoding["occupancy"],
        encoding["mean_rates"][0],
        position_distance,
        block_size=8,
        use_gemm=True,
        pos_tile_size=pos_tile_size,
    )

    # Should match exactly
    assert result_no_tile.shape == result_tiled.shape
    assert np.allclose(
        np.asarray(result_no_tile), np.asarray(result_tiled), rtol=1e-12, atol=1e-14
    )
