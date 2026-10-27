"""Position tiling (``pos_tile_size``) of the GEMM KDE path matches the untiled result."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity as estimate_log,
)
from non_local_detector.tests.likelihoods.conftest import realistic_log_background

pytestmark = pytest.mark.unit


@pytest.fixture
def tiling_test_data():
    """Generate test data with large position dimension."""
    rng = np.random.default_rng(42)

    n_encoding = 100
    n_decoding = 50
    n_features = 4
    n_position = 500  # Large position grid

    encoding_features = jnp.array(
        rng.standard_normal((n_encoding, n_features)).astype(np.float32)
    )
    decoding_features = jnp.array(
        rng.standard_normal((n_decoding, n_features)).astype(np.float32)
    )
    waveform_stds = jnp.array(
        np.abs(rng.standard_normal(n_features).astype(np.float32)) + 0.5
    )
    occupancy = jnp.ones(n_position, dtype=jnp.float32)
    mean_rate = 5.0

    log_position_distance = jnp.array(
        rng.standard_normal((n_encoding, n_position)).astype(np.float32)
    )

    return {
        "encoding_features": encoding_features,
        "decoding_features": decoding_features,
        "waveform_stds": waveform_stds,
        "occupancy": occupancy,
        "mean_rate": mean_rate,
        "log_position_distance": log_position_distance,
    }


def test_log_space_no_tiling_vs_tiling(tiling_test_data):
    """Test that log-space tiled computation matches non-tiled."""
    # No tiling (default)
    result_no_tile = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=None,
        log_background=realistic_log_background(tiling_test_data["decoding_features"]),
    )

    # With tiling (tile_size=100)
    result_tiled = estimate_log(
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=100,
        log_background=realistic_log_background(tiling_test_data["decoding_features"]),
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
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
        use_gemm=True,
        pos_tile_size=None,
        log_background=realistic_log_background(tiling_test_data["decoding_features"]),
    )

    # Test various tile sizes
    for tile_size in [50, 100, 150, 250, 600]:  # Last one > n_pos (500)
        result_tiled = estimate_log(
            tiling_test_data["decoding_features"],
            tiling_test_data["encoding_features"],
            tiling_test_data["waveform_stds"],
            tiling_test_data["occupancy"],
            tiling_test_data["mean_rate"],
            tiling_test_data["log_position_distance"],
            use_gemm=True,
            pos_tile_size=tile_size,
            log_background=realistic_log_background(
                tiling_test_data["decoding_features"]
            ),
        )

        assert np.allclose(reference, result_tiled, rtol=1e-5, atol=1e-6), (
            f"Tile size {tile_size} produces different result: "
            f"max diff = {np.abs(reference - result_tiled).max()}"
        )


def test_tiling_edge_cases():
    """Test tiling with edge cases."""
    rng = np.random.default_rng(123)

    n_enc = 10
    n_dec = 5
    n_features = 2
    n_pos = 11  # Prime number for uneven tiling

    encoding_features = jnp.array(
        rng.standard_normal((n_enc, n_features)).astype(np.float32)
    )
    decoding_features = jnp.array(
        rng.standard_normal((n_dec, n_features)).astype(np.float32)
    )
    waveform_stds = jnp.array([1.0, 1.0])
    occupancy = jnp.ones(n_pos, dtype=jnp.float32)
    log_position_distance = jnp.array(
        rng.standard_normal((n_enc, n_pos)).astype(np.float32)
    )
    mean_rate = 5.0

    # No tiling
    reference = estimate_log(
        decoding_features,
        encoding_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=None,
        log_background=realistic_log_background(decoding_features),
    )

    # Tile size = 1 (extreme case)
    result_tile_1 = estimate_log(
        decoding_features,
        encoding_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=1,
        log_background=realistic_log_background(decoding_features),
    )

    # Tile size = 3 (doesn't divide evenly into 11)
    result_tile_3 = estimate_log(
        decoding_features,
        encoding_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
        use_gemm=True,
        pos_tile_size=3,
        log_background=realistic_log_background(decoding_features),
    )

    assert np.allclose(reference, result_tile_1, rtol=1e-5, atol=1e-6)
    assert np.allclose(reference, result_tile_3, rtol=1e-5, atol=1e-6)


def test_tiled_gemm_matches_linear_reference(tiling_test_data):
    """The position-tiled GEMM path matches the linear-space reference.

    ``pos_tile_size`` only applies to the GEMM path, so the reference
    (``use_gemm=False``) is untiled. These marks are close enough that the
    reference does not underflow.
    """
    log_background = realistic_log_background(tiling_test_data["decoding_features"])
    args = (
        tiling_test_data["decoding_features"],
        tiling_test_data["encoding_features"],
        tiling_test_data["waveform_stds"],
        tiling_test_data["occupancy"],
        tiling_test_data["mean_rate"],
        tiling_test_data["log_position_distance"],
    )
    result_gemm_tiled = estimate_log(
        *args, use_gemm=True, pos_tile_size=100, log_background=log_background
    )
    result_linear = estimate_log(*args, use_gemm=False, log_background=log_background)

    assert np.allclose(result_gemm_tiled, result_linear, rtol=1e-5, atol=1e-6)


def test_tiling_very_large_grid():
    """Ten position tiles over a 5000-bin grid match the untiled result."""
    rng = np.random.default_rng(456)

    n_enc = 100
    n_dec = 50
    n_features = 4
    n_pos = 5000  # Very large position grid

    encoding_features = jnp.array(
        rng.standard_normal((n_enc, n_features)).astype(np.float32)
    )
    decoding_features = jnp.array(
        rng.standard_normal((n_dec, n_features)).astype(np.float32)
    )
    waveform_stds = jnp.array(
        np.abs(rng.standard_normal(n_features).astype(np.float32)) + 0.5
    )
    occupancy = jnp.ones(n_pos, dtype=jnp.float32)
    log_position_distance = jnp.array(
        rng.standard_normal((n_enc, n_pos)).astype(np.float32)
    )
    mean_rate = 5.0

    args = (
        decoding_features,
        encoding_features,
        waveform_stds,
        occupancy,
        mean_rate,
        log_position_distance,
    )
    log_background = realistic_log_background(decoding_features)
    reference = estimate_log(
        *args, use_gemm=True, pos_tile_size=None, log_background=log_background
    )
    result = estimate_log(
        *args,
        use_gemm=True,
        pos_tile_size=500,  # Process 500 positions at a time
        log_background=log_background,
    )

    assert result.shape == (n_dec, n_pos)
    assert jnp.all(jnp.isfinite(result))
    assert np.allclose(result, reference, rtol=1e-5, atol=1e-6), (
        f"max diff = {np.abs(result - reference).max()}"
    )
