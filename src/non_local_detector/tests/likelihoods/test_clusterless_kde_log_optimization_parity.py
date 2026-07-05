"""Test that optimized log-space version uses all optimizations correctly."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    block_estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    log_kde_distance,
)


def _synthetic_block_inputs(n_dec, seed=7):
    rng = np.random.default_rng(seed)
    n_enc, n_pos, n_features = 30, 25, 4
    dec = jnp.asarray(rng.standard_normal((n_dec, n_features)) * 10 + 50)
    enc = jnp.asarray(rng.standard_normal((n_enc, n_features)) * 10 + 50)
    wf_std = jnp.array([5.0] * n_features)
    occ = jnp.asarray(rng.random(n_pos) * 0.6 + 0.2)
    enc_pos = jnp.asarray(rng.standard_normal((n_enc, 2)))
    pos_eval = jnp.asarray(rng.standard_normal((n_pos, 2)))
    log_pos = log_kde_distance(pos_eval, enc_pos, jnp.array([1.0, 1.0]))
    return dec, enc, wf_std, occ, log_pos


def test_final_block_padding_is_inert():
    """A single block with an edge-padded tail (block_size > n_dec) must give
    bit-identical results to a single unpadded block (block_size == n_dec).
    Guards the pad/slice logic: padded rows must not leak into the kept rows."""
    dec, enc, wf_std, occ, log_pos = _synthetic_block_inputs(n_dec=10)
    unpadded = block_estimate_log_joint_mark_intensity(
        dec, enc, wf_std, occ, 2.5, log_pos, block_size=10
    )
    padded = block_estimate_log_joint_mark_intensity(
        dec, enc, wf_std, occ, 2.5, log_pos, block_size=16
    )
    assert padded.shape == (10, occ.shape[0])
    assert jnp.allclose(unpadded, padded, rtol=1e-6, atol=1e-7)


def test_multi_block_matches_single_block():
    """Splitting decoding spikes across multiple blocks (with a padded final
    block) must match a single-block computation. n_dec=10 with block_size=4
    forces blocks of 4+4+2 (padded tail); the tolerance still catches a
    scrambled concat order or a wrong [:actual_len] slice (order-1 errors),
    while allowing the inherent per-block float32 stabilization noise."""
    dec, enc, wf_std, occ, log_pos = _synthetic_block_inputs(n_dec=10)
    single = block_estimate_log_joint_mark_intensity(
        dec, enc, wf_std, occ, 2.5, log_pos, block_size=100
    )
    multi = block_estimate_log_joint_mark_intensity(
        dec, enc, wf_std, occ, 2.5, log_pos, block_size=4
    )
    assert single.shape == (10, occ.shape[0]) and multi.shape == (10, occ.shape[0])
    assert jnp.allclose(single, multi, rtol=1e-4, atol=1e-4)


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
        np.atleast_1d(np.asarray(encoding["waveform_std"])),
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
        np.atleast_1d(np.asarray(encoding["waveform_std"])),
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
