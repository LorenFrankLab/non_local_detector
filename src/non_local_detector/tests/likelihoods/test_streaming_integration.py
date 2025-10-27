"""Tests for streaming integration in estimate_log_joint_mark_intensity.

This module tests that the streaming implementation (computing position kernel
on-the-fly) produces identical results to the precomputed approach while using
less memory.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
    log_kde_distance,
)


class TestStreamingIntegrationBasic:
    """Tests for basic streaming functionality."""

    def test_streaming_equivalence_basic(self):
        """Streaming mode produces identical results to precomputed mode."""
        np.random.seed(42)

        # Setup
        n_dec = 10
        n_enc = 50
        n_pos = 20
        n_features = 4
        n_pos_dims = 2

        dec_features = jnp.array(np.random.randn(n_dec, n_features))
        enc_features = jnp.array(np.random.randn(n_enc, n_features))
        waveform_stds = jnp.array([1.0] * n_features)
        occupancy = jnp.array(np.random.rand(n_pos) * 0.8 + 0.1)
        mean_rate = 2.5

        # Position data
        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([1.0] * n_pos_dims)

        # Precompute position kernel
        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        # Precomputed mode
        result_precomputed = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=True,
            enc_tile_size=25,
            pos_tile_size=None,
            use_streaming=False,
        )

        # Streaming mode
        result_streaming = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,  # Not used in streaming mode
            use_gemm=True,
            enc_tile_size=25,
            pos_tile_size=None,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        # Verify equivalence
        assert result_streaming.shape == result_precomputed.shape
        assert jnp.allclose(result_streaming, result_precomputed, rtol=1e-5, atol=1e-8)

    def test_streaming_equivalence_with_pos_tiling(self):
        """Streaming works with both enc_tile_size and pos_tile_size."""
        np.random.seed(123)

        n_dec = 8
        n_enc = 40
        n_pos = 30
        n_features = 3
        n_pos_dims = 2

        dec_features = jnp.array(np.random.randn(n_dec, n_features))
        enc_features = jnp.array(np.random.randn(n_enc, n_features))
        waveform_stds = jnp.array([1.5, 1.0, 2.0])
        occupancy = jnp.array(np.random.rand(n_pos) * 0.7 + 0.2)
        mean_rate = 3.0

        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([0.8, 1.2])

        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        # Precomputed with position tiling
        result_precomputed = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=True,
            enc_tile_size=20,
            pos_tile_size=15,
            use_streaming=False,
        )

        # Streaming with position tiling
        result_streaming = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            use_gemm=True,
            enc_tile_size=20,
            pos_tile_size=15,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        assert jnp.allclose(result_streaming, result_precomputed, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("enc_tile_size", [10, 25, 49])
    def test_streaming_various_enc_tile_sizes(self, enc_tile_size):
        """Streaming works with various encoding tile sizes."""
        np.random.seed(456)

        n_dec = 5
        n_enc = 50
        n_pos = 15
        n_features = 2
        n_pos_dims = 1

        dec_features = jnp.array(np.random.randn(n_dec, n_features))
        enc_features = jnp.array(np.random.randn(n_enc, n_features))
        waveform_stds = jnp.array([1.0, 1.0])
        occupancy = jnp.array(np.random.rand(n_pos) * 0.5 + 0.3)
        mean_rate = 2.0

        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([1.0])

        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        result_precomputed = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=True,
            enc_tile_size=enc_tile_size,
            use_streaming=False,
        )

        result_streaming = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            use_gemm=True,
            enc_tile_size=enc_tile_size,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        assert jnp.allclose(result_streaming, result_precomputed, rtol=1e-5, atol=1e-8)


class TestStreamingIntegrationEdgeCases:
    """Tests for streaming edge cases and validation."""

    def test_streaming_requires_enc_tile_size(self):
        """Streaming without enc_tile_size raises ValueError."""
        np.random.seed(42)

        dec_features = jnp.array(np.random.randn(5, 2))
        enc_features = jnp.array(np.random.randn(10, 2))
        waveform_stds = jnp.array([1.0, 1.0])
        occupancy = jnp.array(np.random.rand(8))
        mean_rate = 1.5
        enc_positions = jnp.array(np.random.randn(10, 1))
        pos_eval_points = jnp.array(np.random.randn(8, 1))
        position_std = jnp.array([1.0])

        with pytest.raises(
            ValueError, match="use_streaming=True requires enc_tile_size"
        ):
            estimate_log_joint_mark_intensity(
                dec_features,
                enc_features,
                waveform_stds,
                occupancy,
                mean_rate,
                log_position_distance=None,
                use_gemm=True,
                enc_tile_size=None,  # Missing!
                use_streaming=True,
                encoding_positions=enc_positions,
                position_eval_points=pos_eval_points,
                position_std=position_std,
            )

    def test_streaming_requires_position_params(self):
        """Streaming without position parameters raises ValueError."""
        np.random.seed(123)

        dec_features = jnp.array(np.random.randn(5, 2))
        enc_features = jnp.array(np.random.randn(10, 2))
        waveform_stds = jnp.array([1.0, 1.0])
        occupancy = jnp.array(np.random.rand(8))
        mean_rate = 1.5

        with pytest.raises(
            ValueError,
            match="use_streaming=True requires encoding_positions, position_eval_points",
        ):
            estimate_log_joint_mark_intensity(
                dec_features,
                enc_features,
                waveform_stds,
                occupancy,
                mean_rate,
                log_position_distance=None,
                use_gemm=True,
                enc_tile_size=5,
                use_streaming=True,
                # Missing position parameters!
            )

    def test_streaming_enc_tile_size_equals_n_enc(self):
        """Streaming raises error when enc_tile_size >= n_enc (no actual chunking)."""
        np.random.seed(789)

        n_enc = 30
        dec_features = jnp.array(np.random.randn(8, 3))
        enc_features = jnp.array(np.random.randn(n_enc, 3))
        waveform_stds = jnp.array([1.0, 1.5, 0.8])
        occupancy = jnp.array(np.random.rand(12) * 0.6 + 0.2)
        mean_rate = 2.5

        enc_positions = jnp.array(np.random.randn(n_enc, 2))
        pos_eval_points = jnp.array(np.random.randn(12, 2))
        position_std = jnp.array([1.0, 1.0])

        # Streaming with enc_tile_size = n_enc should raise error
        with pytest.raises(
            ValueError,
            match="use_streaming=True requires enc_tile_size < n_encoding_spikes",
        ):
            estimate_log_joint_mark_intensity(
                dec_features,
                enc_features,
                waveform_stds,
                occupancy,
                mean_rate,
                log_position_distance=None,
                use_gemm=True,
                enc_tile_size=n_enc,  # Equal to n_enc - should fail
                use_streaming=True,
                encoding_positions=enc_positions,
                position_eval_points=pos_eval_points,
                position_std=position_std,
            )

    def test_streaming_enc_tile_size_one(self):
        """Streaming works with enc_tile_size=1 (extreme chunking)."""
        np.random.seed(101)

        n_enc = 10
        dec_features = jnp.array(np.random.randn(3, 2))
        enc_features = jnp.array(np.random.randn(n_enc, 2))
        waveform_stds = jnp.array([1.0, 1.0])
        occupancy = jnp.array(np.random.rand(5) * 0.5 + 0.3)
        mean_rate = 1.8

        enc_positions = jnp.array(np.random.randn(n_enc, 1))
        pos_eval_points = jnp.array(np.random.randn(5, 1))
        position_std = jnp.array([1.2])

        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        result_precomputed = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=True,
            enc_tile_size=1,
            use_streaming=False,
        )

        result_streaming = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            use_gemm=True,
            enc_tile_size=1,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        assert jnp.allclose(result_streaming, result_precomputed, rtol=1e-5, atol=1e-8)


class TestStreamingBlockEstimate:
    """Tests for streaming with block_estimate_log_joint_mark_intensity."""

    def test_block_estimate_streaming_equivalence(self):
        """block_estimate with streaming matches precomputed mode."""
        np.random.seed(42)

        n_dec = 50  # Large enough to trigger blocking
        n_enc = 40
        n_pos = 20
        n_features = 4
        n_pos_dims = 2
        block_size = 15

        dec_features = jnp.array(np.random.randn(n_dec, n_features))
        enc_features = jnp.array(np.random.randn(n_enc, n_features))
        waveform_stds = jnp.array([1.0] * n_features)
        occupancy = jnp.array(np.random.rand(n_pos) * 0.8 + 0.1)
        mean_rate = 2.5

        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([1.0] * n_pos_dims)

        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        # Precomputed mode
        result_precomputed = block_estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            block_size=block_size,
            use_gemm=True,
            enc_tile_size=20,
            pos_tile_size=None,
            use_streaming=False,
        )

        # Streaming mode
        result_streaming = block_estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            block_size=block_size,
            use_gemm=True,
            enc_tile_size=20,
            pos_tile_size=None,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        assert result_streaming.shape == result_precomputed.shape
        assert jnp.allclose(result_streaming, result_precomputed, rtol=1e-5, atol=1e-8)

    def test_block_estimate_streaming_with_both_tilings(self):
        """block_estimate streaming works with enc_tile_size and pos_tile_size."""
        np.random.seed(456)

        n_dec = 60
        n_enc = 50
        n_pos = 30
        n_features = 3
        n_pos_dims = 2
        block_size = 20

        dec_features = jnp.array(np.random.randn(n_dec, n_features))
        enc_features = jnp.array(np.random.randn(n_enc, n_features))
        waveform_stds = jnp.array([1.2, 0.9, 1.5])
        occupancy = jnp.array(np.random.rand(n_pos) * 0.7 + 0.2)
        mean_rate = 3.5

        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([0.9, 1.1])

        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        result_precomputed = block_estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            block_size=block_size,
            use_gemm=True,
            enc_tile_size=25,
            pos_tile_size=15,
            use_streaming=False,
        )

        result_streaming = block_estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            block_size=block_size,
            use_gemm=True,
            enc_tile_size=25,
            pos_tile_size=15,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        assert jnp.allclose(result_streaming, result_precomputed, rtol=1e-5, atol=1e-8)


class TestStreamingNumericalStability:
    """Tests for streaming numerical stability with extreme values."""

    def test_streaming_stability_extreme_features(self):
        """Streaming remains stable with extreme feature distances."""
        np.random.seed(789)

        n_dec = 5
        n_enc = 20
        n_pos = 10
        n_features = 4
        n_pos_dims = 2

        # Create extreme feature distances
        dec_features = jnp.array(np.random.randn(n_dec, n_features) * 10.0)
        enc_features = jnp.array(np.random.randn(n_enc, n_features) * 10.0)
        waveform_stds = jnp.array([0.5] * n_features)  # Small std → large distances
        occupancy = jnp.array(np.random.rand(n_pos) * 0.5 + 0.3)
        mean_rate = 2.0

        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([1.0, 1.0])

        result_streaming = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            use_gemm=True,
            enc_tile_size=10,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        # Verify all values are finite
        assert jnp.all(jnp.isfinite(result_streaming))
        assert result_streaming.shape == (n_dec, n_pos)

    def test_streaming_stability_zero_occupancy(self):
        """Streaming handles zero occupancy correctly (masking)."""
        np.random.seed(101)

        n_dec = 3
        n_enc = 15
        n_pos = 8
        n_features = 2
        n_pos_dims = 1

        dec_features = jnp.array(np.random.randn(n_dec, n_features))
        enc_features = jnp.array(np.random.randn(n_enc, n_features))
        waveform_stds = jnp.array([1.0, 1.0])

        # Some zero occupancy positions
        occupancy = jnp.array([0.0, 0.5, 0.3, 0.0, 0.6, 0.2, 0.0, 0.4])
        mean_rate = 1.5

        enc_positions = jnp.array(np.random.randn(n_enc, n_pos_dims))
        pos_eval_points = jnp.array(np.random.randn(n_pos, n_pos_dims))
        position_std = jnp.array([1.0])

        log_position_distance = log_kde_distance(
            pos_eval_points, enc_positions, position_std
        )

        result_precomputed = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance,
            use_gemm=True,
            enc_tile_size=8,
            use_streaming=False,
        )

        result_streaming = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            log_position_distance=None,
            use_gemm=True,
            enc_tile_size=8,
            use_streaming=True,
            encoding_positions=enc_positions,
            position_eval_points=pos_eval_points,
            position_std=position_std,
        )

        # Both should have -inf at zero occupancy positions
        zero_occ_mask = occupancy == 0.0
        assert jnp.all(jnp.isneginf(result_precomputed[:, zero_occ_mask]))
        assert jnp.all(jnp.isneginf(result_streaming[:, zero_occ_mask]))

        # Non-zero positions should match
        non_zero_mask = ~zero_occ_mask
        assert jnp.allclose(
            result_streaming[:, non_zero_mask],
            result_precomputed[:, non_zero_mask],
            rtol=1e-5,
            atol=1e-8,
        )
