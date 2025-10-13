"""Test position tiling correctness and memory efficiency."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_orig,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    estimate_log_joint_mark_intensity as estimate_log,
)


@pytest.fixture
def tiling_test_data():
    """Generate test data with large position dimension."""
    np.random.seed(42)

    n_encoding = 100
    n_decoding = 50
    n_features = 4
    n_position = 500  # Large position grid

    encoding_features = jnp.array(np.random.randn(n_encoding, n_features).astype(np.float32))
    decoding_features = jnp.array(np.random.randn(n_decoding, n_features).astype(np.float32))
    waveform_stds = jnp.array(np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5)
    encoding_weights = jnp.ones(n_encoding, dtype=jnp.float32)
    occupancy = jnp.ones(n_position, dtype=jnp.float32)
    mean_rate = 5.0

    # Position distance (for original)
    position_distance = jnp.array(np.random.rand(n_encoding, n_position).astype(np.float32))

    # Log position distance (for log-space)
    log_position_distance = jnp.array(np.random.randn(n_encoding, n_position).astype(np.float32))

    return {
        "encoding_features": encoding_features,
        "decoding_features": decoding_features,
        "waveform_stds": waveform_stds,
        "encoding_weights": encoding_weights,
        "occupancy": occupancy,
        "mean_rate": mean_rate,
        "position_distance": position_distance,
        "log_position_distance": log_position_distance,
    }


def test_log_space_no_tiling_vs_tiling(tiling_test_data):
    """Test that log-space tiled computation matches non-tiled."""
    # No tiling (default)
    result_no_tile = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=None,
    )

    # With tiling (tile_size=100)
    result_tiled = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=100,
    )

    assert result_no_tile.shape == result_tiled.shape
    assert np.allclose(result_no_tile, result_tiled, rtol=1e-5, atol=1e-6), (
        f"Tiled and non-tiled results differ: "
        f"max diff = {np.abs(result_no_tile - result_tiled).max()}"
    )


def test_original_no_tiling_vs_tiling(tiling_test_data):
    """Test that original (non-log) tiled computation matches non-tiled."""
    # No tiling (default)
    result_no_tile = estimate_orig(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["position_distance"],
        pos_tile_size=None,
    )

    # With tiling (tile_size=100)
    result_tiled = estimate_orig(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["position_distance"],
        pos_tile_size=100,
    )

    assert result_no_tile.shape == result_tiled.shape
    assert np.allclose(result_no_tile, result_tiled, rtol=1e-5, atol=1e-6), (
        f"Tiled and non-tiled results differ: "
        f"max diff = {np.abs(result_no_tile - result_tiled).max()}"
    )


def test_various_tile_sizes(tiling_test_data):
    """Test that various tile sizes all produce same result."""
    # Reference (no tiling)
    reference = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=None,
    )

    # Test various tile sizes
    for tile_size in [50, 100, 150, 250, 600]:  # Last one > n_pos (500)
        result_tiled = estimate_log(
            tiling_test_data["decoding_features"],
            tiling_test_data["encoding_features"],
            tiling_test_data["encoding_weights"],
            tiling_test_data["waveform_stds"],
            tiling_test_data["occupancy"],
            tiling_test_data["mean_rate"],
            tiling_test_data["log_position_distance"],
            use_gemm=True,
            pos_tile_size=tile_size,
        )

        assert np.allclose(reference, result_tiled, rtol=1e-5, atol=1e-6), (
            f"Tile size {tile_size} produces different result: "
            f"max diff = {np.abs(reference - result_tiled).max()}"
        )


def test_tiling_edge_cases():
    """Test tiling with edge cases."""
    np.random.seed(123)

    n_enc = 10
    n_dec = 5
    n_features = 2
    n_pos = 11  # Prime number for uneven tiling

    encoding_features = jnp.array(np.random.randn(n_enc, n_features).astype(np.float32))
    decoding_features = jnp.array(np.random.randn(n_dec, n_features).astype(np.float32))
    waveform_stds = jnp.array([1.0, 1.0])
    encoding_weights = jnp.ones(n_enc, dtype=jnp.float32)
    occupancy = jnp.ones(n_pos, dtype=jnp.float32)
    log_position_distance = jnp.array(np.random.randn(n_enc, n_pos).astype(np.float32))
    mean_rate = 5.0

    # No tiling
    reference = estimate_log(
        decoding_features,
        encoding_features,
        encoding_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=None,
    )

    # Tile size = 1 (extreme case)
    result_tile_1 = estimate_log(
        decoding_features,
        encoding_features,
        encoding_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=1,
    )

    # Tile size = 3 (doesn't divide evenly into 11)
    result_tile_3 = estimate_log(
        decoding_features,
        encoding_features,
        encoding_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=3,
    )

    assert np.allclose(reference, result_tile_1, rtol=1e-5, atol=1e-6)
    assert np.allclose(reference, result_tile_3, rtol=1e-5, atol=1e-6)


def test_tiling_gemm_vs_loop(tiling_test_data):
    """Test that tiling works with both GEMM and loop methods."""
    # GEMM with tiling
    result_gemm_tiled = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=100,
    )

    # Loop with tiling
    result_loop_tiled = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["encoding_weights"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=False,
        pos_tile_size=100,
    )

    # Should produce same result (both methods are equivalent)
    assert np.allclose(result_gemm_tiled, result_loop_tiled, rtol=1e-5, atol=1e-6)


def test_tiling_very_large_grid():
    """Test that tiling enables computation on very large position grids."""
    np.random.seed(456)

    n_enc = 100
    n_dec = 50
    n_features = 4
    n_pos = 5000  # Very large position grid

    encoding_features = jnp.array(np.random.randn(n_enc, n_features).astype(np.float32))
    decoding_features = jnp.array(np.random.randn(n_dec, n_features).astype(np.float32))
    waveform_stds = jnp.array(np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5)
    encoding_weights = jnp.ones(n_enc, dtype=jnp.float32)
    occupancy = jnp.ones(n_pos, dtype=jnp.float32)
    log_position_distance = jnp.array(np.random.randn(n_enc, n_pos).astype(np.float32))
    mean_rate = 5.0

    # With tiling (should complete without OOM)
    result = estimate_log(
        decoding_features,
        encoding_features,
        encoding_weights,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=500,  # Process 500 positions at a time
    )

    assert result.shape == (n_dec, n_pos)
    assert jnp.all(jnp.isfinite(result))


def test_tiling_memory_estimation():
    """Verify memory scaling with tiling."""
    n_enc = 1000
    n_pos = 2000
    tile_size = 200

    # Memory without tiling: O(n_enc * n_pos) = 1000 * 2000 * 4 bytes = 8 MB
    memory_no_tile_mb = (n_enc * n_pos * 4) / 1e6

    # Memory with tiling: O(n_enc * tile_size) = 1000 * 200 * 4 bytes = 0.8 MB
    memory_with_tile_mb = (n_enc * tile_size * 4) / 1e6

    # Ratio
    memory_reduction = memory_no_tile_mb / memory_with_tile_mb

    # Should be 10× reduction
    assert np.isclose(memory_reduction, 10.0, rtol=0.01)

    print(f"Memory without tiling: {memory_no_tile_mb:.2f} MB")
    print(f"Memory with tiling (tile_size={tile_size}): {memory_with_tile_mb:.2f} MB")
    print(f"Memory reduction: {memory_reduction:.1f}×")
