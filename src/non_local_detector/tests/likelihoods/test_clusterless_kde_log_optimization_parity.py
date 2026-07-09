"""Test that optimized log-space version uses all optimizations correctly."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
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


def _fit_kwargs(env, **overrides):
    kwargs = {
        "position_time": np.linspace(0.0, 10.0, 101),
        "position": np.linspace(0.0, 10.0, 101)[:, None],
        "spike_times": [np.array([2.0, 5.0, 7.5])],
        "spike_waveform_features": [
            np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)
        ],
        "environment": env,
        "position_std": np.sqrt(1.0),
        "waveform_std": 1.0,
        "block_size": 8,
        "disable_progress_bar": True,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("bad_std", [0.0, -1.0])
def test_fit_rejects_nonpositive_position_std(simple_1d_environment, bad_std):
    """A zero/negative bandwidth is a config error and must raise at fit time
    (loudly), not be silently clamped to a near-delta kernel inside the kernel."""
    with pytest.raises(ValueError, match="position_std"):
        fit_clusterless_kde_encoding_model(
            **_fit_kwargs(simple_1d_environment, position_std=bad_std)
        )


@pytest.mark.parametrize("bad_std", [0.0, -1.0])
def test_fit_rejects_nonpositive_waveform_std(simple_1d_environment, bad_std):
    with pytest.raises(ValueError, match="waveform_std"):
        fit_clusterless_kde_encoding_model(
            **_fit_kwargs(simple_1d_environment, waveform_std=bad_std)
        )


@pytest.mark.parametrize("n_features", [4, 10])
def test_encoding_weights_consistent_across_paths(n_features):
    """Non-uniform ``encoding_weights`` must give the same joint mark intensity
    on every numerical path.

    ``n_features=4`` exercises the compensated-linear matmul paths, ``n_features=10``
    (> the compensated-linear feature cap) the logsumexp paths. For each, the
    reference linear path (``use_gemm=False``), the non-chunked GEMM path, the
    encoding-chunked path, and the streaming-chunked path must all match. The
    weight vector includes a zero, which drops that spike out of the reduction --
    a case that would produce NaN if the weight were folded into the mark kernel's
    row-max stabilization rather than added as a separate log term.
    """
    rng = np.random.default_rng(11)
    n_enc, n_pos, n_pos_dims = 30, 25, 2
    # Keep the mark kernel well above the LOG_EPS floor even at 10 features: draw
    # encoding marks at unit scale and make the decoding marks near-coincident with
    # the first few, with a bandwidth wide enough that 10-D products do not underflow.
    enc = jnp.asarray(rng.standard_normal((n_enc, n_features)))
    dec = jnp.asarray(np.asarray(enc[:6]) + rng.standard_normal((6, n_features)) * 0.05)
    wf_std = jnp.array([2.0] * n_features)
    occ = jnp.asarray(rng.random(n_pos) * 0.6 + 0.2)
    enc_pos = jnp.asarray(rng.standard_normal((n_enc, n_pos_dims)))
    pos_eval = jnp.asarray(rng.standard_normal((n_pos, n_pos_dims)))
    position_std = jnp.array([1.0, 1.0])
    log_pos = log_kde_distance(pos_eval, enc_pos, position_std)
    mean_rate = 2.5

    # Non-uniform weights with a zero (spike index 3 drops out entirely).
    w = jnp.asarray(rng.random(n_enc) * 2.0 + 0.5).at[3].set(0.0)

    reference = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        use_gemm=False,
        encoding_weights=w,
    )
    unweighted = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        use_gemm=False,
    )
    # The weights must actually change the result (else they are ignored).
    assert not np.allclose(np.asarray(reference), np.asarray(unweighted))

    non_chunked = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        use_gemm=True,
        encoding_weights=w,
    )
    chunked = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        use_gemm=True,
        enc_tile_size=8,
        encoding_weights=w,
    )
    streaming = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        None,
        use_gemm=True,
        enc_tile_size=8,
        use_streaming=True,
        encoding_positions=enc_pos,
        position_eval_points=pos_eval,
        position_std=position_std,
        encoding_weights=w,
    )

    ref = np.asarray(reference)
    for name, out in [
        ("non_chunked", non_chunked),
        ("chunked", chunked),
        ("streaming", streaming),
    ]:
        out = np.asarray(out)
        assert np.all(np.isfinite(out)), f"{name} produced non-finite values"
        assert np.allclose(out, ref, rtol=1e-4, atol=1e-4), (
            f"{name} != reference; max|diff|={np.abs(out - ref).max():.3e}"
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
