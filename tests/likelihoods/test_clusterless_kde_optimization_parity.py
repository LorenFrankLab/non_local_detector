"""Clusterless KDE joint mark intensity agrees across its computation paths.

Decode-spike blocking (including a padded final block), the GEMM path,
encoding chunking and streaming give the same intensity, also with non-uniform
encoding weights.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
    log_kde_distance,
)
from tests.likelihoods._kde_reference import reference_log_joint_mark_intensity
from tests.likelihoods.conftest import realistic_log_background


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
    return dec, enc, wf_std, occ, log_pos, realistic_log_background(dec)


def test_final_block_padding_is_inert():
    """A single block with an edge-padded tail (block_size > n_dec) must give
    bit-identical results to a single unpadded block (block_size == n_dec).
    Guards the pad/slice logic: padded rows must not leak into the kept rows."""
    dec, enc, wf_std, occ, log_pos, log_background = _synthetic_block_inputs(n_dec=10)
    unpadded = block_estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        2.5,
        log_pos,
        block_size=10,
        log_background=log_background,
    )
    padded = block_estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        2.5,
        log_pos,
        block_size=16,
        log_background=log_background,
    )
    assert padded.shape == (10, occ.shape[0])
    assert jnp.allclose(unpadded, padded, rtol=1e-6, atol=1e-7)


def test_multi_block_matches_single_block():
    """Splitting decoding spikes across multiple blocks (with a padded final
    block) must match a single-block computation. n_dec=10 with block_size=4
    forces blocks of 4+4+2 (padded tail); the tolerance still catches a
    scrambled concat order or a wrong [:actual_len] slice (order-1 errors),
    while allowing the inherent per-block float32 stabilization noise."""
    dec, enc, wf_std, occ, log_pos, log_background = _synthetic_block_inputs(n_dec=10)
    single = block_estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        2.5,
        log_pos,
        block_size=100,
        log_background=log_background,
    )
    multi = block_estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        2.5,
        log_pos,
        block_size=4,
        log_background=log_background,
    )
    assert single.shape == (10, occ.shape[0])
    assert multi.shape == (10, occ.shape[0])
    assert jnp.allclose(single, multi, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_features", [4, 10])
def test_encoding_weights_consistent_across_paths(n_features):
    """Non-uniform ``encoding_weights`` must give the same joint mark intensity
    on every numerical path.

    At 4 and 10 mark features, the non-chunked GEMM path, the encoding-chunked
    path, and the streaming-chunked path must all match a float64 logsumexp
    reference. The weight vector includes a zero, which drops that spike out of
    the reduction --
    a case that would produce NaN if the weight were folded into the mark kernel's
    row-max stabilization rather than added as a separate log term.
    """
    rng = np.random.default_rng(11)
    n_enc, n_pos, n_pos_dims = 30, 25, 2
    # Keep the intensity well above each spike's background even at 10 features: draw
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
    log_background = realistic_log_background(dec)

    # Non-uniform weights with a zero (spike index 3 drops out entirely).
    w = jnp.asarray(rng.random(n_enc) * 2.0 + 0.5).at[3].set(0.0)

    ref = reference_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        encoding_weights=w,
        log_background=log_background,
    )
    unweighted = reference_log_joint_mark_intensity(
        dec, enc, wf_std, occ, mean_rate, log_pos, log_background=log_background
    )
    # The weights must actually change the result (else they are ignored).
    assert not np.allclose(ref, unweighted, rtol=1e-4, atol=1e-4)

    non_chunked = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        encoding_weights=w,
        log_background=log_background,
    )
    chunked = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        log_pos,
        enc_tile_size=8,
        encoding_weights=w,
        log_background=log_background,
    )
    streaming = estimate_log_joint_mark_intensity(
        dec,
        enc,
        wf_std,
        occ,
        mean_rate,
        None,
        enc_tile_size=8,
        use_streaming=True,
        encoding_positions=enc_pos,
        position_eval_points=pos_eval,
        position_std=position_std,
        encoding_weights=w,
        log_background=log_background,
    )

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
