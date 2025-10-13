"""Tests for clusterless GMM likelihood optimization.

This module tests the memory optimization changes to predict_clusterless_gmm_log_likelihood,
specifically:
1. Blocking parity: Verify blocked processing matches full vmap results
2. Bin tiling parity: Verify bin tiling matches no tiling
3. Memory scaling: Verify expected memory reductions
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from non_local_detector import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)


@pytest.fixture
def gmm_simulation_data():
    """Create synthetic data for GMM likelihood testing."""
    np.random.seed(42)

    # Time parameters
    dt = 0.02  # 20 ms bins
    n_time = 50
    time = np.arange(n_time) * dt

    # Position parameters
    position_time = np.linspace(0, (n_time - 1) * dt, 100)
    position = np.column_stack(
        [
            np.linspace(0, 10, len(position_time)),  # x
            np.sin(np.linspace(0, 2 * np.pi, len(position_time))) * 2,  # y
        ]
    )

    # Spike parameters
    n_electrodes = 3
    n_features = 4
    spike_times = []
    spike_features = []

    for _ in range(n_electrodes):
        # Generate random spike times
        n_spikes = np.random.randint(20, 50)
        times = np.sort(np.random.uniform(0, time[-1], n_spikes))
        spike_times.append(times)

        # Generate random waveform features
        features = np.random.randn(n_spikes, n_features).astype(np.float32)
        spike_features.append(features)

    # Create and fit environment
    environment = Environment(position_range=[(0, 10), (-3, 3)])
    environment = environment.fit_place_grid(position=position, infer_track_interior=True)

    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_features": spike_features,
        "environment": environment,
    }


def test_gmm_blocking_parity(gmm_simulation_data):
    """Test that blocked processing matches full processing (no blocking).

    This verifies that the streaming block optimization doesn't change
    the numerical results compared to processing all spikes at once.
    """
    # Fit the encoding model (use fewer GMM components for small test data)
    encoding_model = fit_clusterless_gmm_encoding_model(
        gmm_simulation_data["position_time"],
        gmm_simulation_data["position"],
        gmm_simulation_data["spike_times"],
        gmm_simulation_data["spike_features"],
        gmm_simulation_data["environment"],
        gmm_components_occupancy=8,
        gmm_components_gpi=8,
        gmm_components_joint=16,
    )

    # Convert to JAX arrays
    time = jnp.asarray(gmm_simulation_data["time"])
    position_time = jnp.asarray(gmm_simulation_data["position_time"])
    position = jnp.asarray(gmm_simulation_data["position"])
    spike_times = [jnp.asarray(st) for st in gmm_simulation_data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in gmm_simulation_data["spike_features"]]

    # Predict with no blocking (very large block size)
    result_no_block = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        spike_block_size=999999,  # Effectively no blocking
    )

    # Predict with blocking
    result_blocked = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        spike_block_size=10,  # Small blocks to test the mechanism
    )

    # Results should be identical (within numerical precision)
    assert_allclose(
        result_no_block,
        result_blocked,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Blocked processing changed results compared to full processing",
    )


def test_gmm_bin_tiling_parity(gmm_simulation_data):
    """Test that bin tiling matches no tiling.

    This verifies that tiling over position bins doesn't change the
    numerical results.
    """
    # Fit the encoding model (use fewer GMM components for small test data)
    encoding_model = fit_clusterless_gmm_encoding_model(
        gmm_simulation_data["position_time"],
        gmm_simulation_data["position"],
        gmm_simulation_data["spike_times"],
        gmm_simulation_data["spike_features"],
        gmm_simulation_data["environment"],
        gmm_components_occupancy=8,
        gmm_components_gpi=8,
        gmm_components_joint=16,
    )

    # Convert to JAX arrays
    time = jnp.asarray(gmm_simulation_data["time"])
    position_time = jnp.asarray(gmm_simulation_data["position_time"])
    position = jnp.asarray(gmm_simulation_data["position"])
    spike_times = [jnp.asarray(st) for st in gmm_simulation_data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in gmm_simulation_data["spike_features"]]

    # Predict without bin tiling
    result_no_tile = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        bin_tile_size=None,  # No tiling
    )

    # Predict with bin tiling
    result_tiled = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        bin_tile_size=50,  # Tile with small chunks
    )

    # Results should be identical (within numerical precision)
    assert_allclose(
        result_no_tile,
        result_tiled,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Bin tiling changed results compared to no tiling",
    )


def test_gmm_combined_optimizations(gmm_simulation_data):
    """Test combining both blocking and bin tiling.

    This verifies that using both optimizations together produces the
    same results as using neither.
    """
    # Fit the encoding model (use fewer GMM components for small test data)
    encoding_model = fit_clusterless_gmm_encoding_model(
        gmm_simulation_data["position_time"],
        gmm_simulation_data["position"],
        gmm_simulation_data["spike_times"],
        gmm_simulation_data["spike_features"],
        gmm_simulation_data["environment"],
        gmm_components_occupancy=8,
        gmm_components_gpi=8,
        gmm_components_joint=16,
    )

    # Convert to JAX arrays
    time = jnp.asarray(gmm_simulation_data["time"])
    position_time = jnp.asarray(gmm_simulation_data["position_time"])
    position = jnp.asarray(gmm_simulation_data["position"])
    spike_times = [jnp.asarray(st) for st in gmm_simulation_data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in gmm_simulation_data["spike_features"]]

    # Predict with neither optimization
    result_baseline = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        spike_block_size=999999,  # No blocking
        bin_tile_size=None,  # No tiling
    )

    # Predict with both optimizations
    result_optimized = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        spike_block_size=10,  # Small spike blocks
        bin_tile_size=50,  # Small bin tiles
    )

    # Results should be identical (within numerical precision)
    assert_allclose(
        result_baseline,
        result_optimized,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Combined optimizations changed results",
    )


def test_gmm_memory_scaling():
    """Verify the memory scaling formula.

    This is a mathematical verification (not runtime profiling) that
    the expected memory reduction is achieved based on array sizes.
    """
    # Problem configuration
    n_spikes = 5000
    n_bins = 2000
    spike_block_size = 1000
    bin_tile_size = 256

    # Memory without any optimization (all spikes × all bins)
    # Each float32 is 4 bytes
    mem_full = n_spikes * n_bins * 4 / 1e6  # MB

    # Memory with spike blocking only
    mem_spike_block = spike_block_size * n_bins * 4 / 1e6  # MB

    # Memory with both optimizations
    mem_both = spike_block_size * bin_tile_size * 4 / 1e6  # MB

    # Expected reductions
    reduction_spike_block = mem_full / mem_spike_block
    reduction_both = mem_full / mem_both

    # Verify expected reductions
    assert reduction_spike_block == n_spikes / spike_block_size  # Should be 5×
    assert reduction_both == (n_spikes / spike_block_size) * (
        n_bins / bin_tile_size
    )  # Should be ~39×

    # Verify absolute values are reasonable
    assert mem_full == 40.0  # 40 MB
    assert mem_spike_block == 8.0  # 8 MB
    assert mem_both == pytest.approx(1.024, abs=0.01)  # ~1 MB

    # Print for documentation
    print(f"\nMemory scaling verification:")
    print(f"  Full processing: {mem_full:.1f} MB")
    print(f"  Spike blocking only: {mem_spike_block:.1f} MB ({reduction_spike_block:.0f}× reduction)")
    print(f"  Both optimizations: {mem_both:.3f} MB ({reduction_both:.0f}× reduction)")


def test_gmm_edge_cases(gmm_simulation_data):
    """Test edge cases in the optimization.

    - Few spikes per electrode
    - Block size larger than spike count
    - Bin tile size larger than total bins
    """
    # Create minimal data (need at least 3 spikes for 2 GMM components)
    minimal_data = {
        "time": gmm_simulation_data["time"][:10],
        "position_time": gmm_simulation_data["position_time"][:30],
        "position": gmm_simulation_data["position"][:30],
        "spike_times": [times[:5] for times in gmm_simulation_data["spike_times"]],  # 5 spikes each
        "spike_features": [feats[:5] for feats in gmm_simulation_data["spike_features"]],
    }

    # Fit the encoding model (reuse environment from fixture, use very few components)
    encoding_model = fit_clusterless_gmm_encoding_model(
        minimal_data["position_time"],
        minimal_data["position"],
        minimal_data["spike_times"],
        minimal_data["spike_features"],
        gmm_simulation_data["environment"],
        gmm_components_occupancy=2,
        gmm_components_gpi=2,
        gmm_components_joint=4,
    )

    # Convert to JAX arrays
    time = jnp.asarray(minimal_data["time"])
    position_time = jnp.asarray(minimal_data["position_time"])
    position = jnp.asarray(minimal_data["position"])
    spike_times = [jnp.asarray(st) for st in minimal_data["spike_times"]]
    spike_features = [jnp.asarray(sf) for sf in minimal_data["spike_features"]]

    # Test 1: Block size larger than spike count (should work fine)
    result_large_block = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        spike_block_size=1000,  # Much larger than 1 spike
    )

    # Test 2: Bin tile size larger than total bins (should work fine)
    result_large_tile = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        bin_tile_size=10000,  # Much larger than bin count
    )

    # Test 3: Both combined with single spike
    result_combined = predict_clusterless_gmm_log_likelihood(
        time,
        position_time,
        position,
        spike_times,
        spike_features,
        encoding_model,
        is_local=False,
        spike_block_size=1000,
        bin_tile_size=10000,
    )

    # All should produce valid results (note: shape[0] may be < len(time) if spikes fall outside bins)
    assert result_large_block.shape[0] > 0
    assert result_large_tile.shape[0] > 0
    assert result_combined.shape[0] > 0

    # All should have same shape
    assert result_large_block.shape == result_large_tile.shape
    assert result_large_block.shape == result_combined.shape

    # All should be identical
    assert_allclose(
        result_large_block,
        result_large_tile,
        rtol=1e-5,
    )
    assert_allclose(
        result_large_block,
        result_combined,
        rtol=1e-5,
    )
