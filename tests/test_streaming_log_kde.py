#!/usr/bin/env python3
"""Test streaming log_kde_distance implementation.

This demonstrates the memory-efficient streaming approach for computing
position kernels without materializing the full (D × n_enc × n_pos) intermediate.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    log_kde_distance,
    log_kde_distance_streaming,
)


@pytest.mark.unit
def test_streaming_equivalence_2d():
    """Test that streaming gives same results as vmap for 2D positions."""
    n_eval = 20
    n_samp = 30
    n_dims = 2

    np.random.seed(42)
    eval_points = jnp.array(np.random.uniform(0, 100, (n_eval, n_dims)))
    samples = jnp.array(np.random.uniform(0, 100, (n_samp, n_dims)))
    std = jnp.array([5.0, 5.0])

    # Standard vmap approach
    result_vmap = log_kde_distance(eval_points, samples, std)

    # Streaming fori_loop approach
    result_streaming = log_kde_distance_streaming(eval_points, samples, std)

    # Should be exactly equal (both use same underlying computation)
    assert result_vmap.shape == (n_samp, n_eval)
    assert result_streaming.shape == (n_samp, n_eval)
    np.testing.assert_allclose(
        result_vmap, result_streaming, rtol=1e-7, atol=1e-10
    )


@pytest.mark.unit
@pytest.mark.parametrize("n_dims", [1, 2, 5, 10])
def test_streaming_equivalence_various_dims(n_dims):
    """Test streaming for various dimensionalities."""
    n_eval = 15
    n_samp = 20

    np.random.seed(123)
    eval_points = jnp.array(np.random.uniform(0, 50, (n_eval, n_dims)))
    samples = jnp.array(np.random.uniform(0, 50, (n_samp, n_dims)))
    std = jnp.array([3.0] * n_dims)

    result_vmap = log_kde_distance(eval_points, samples, std)
    result_streaming = log_kde_distance_streaming(eval_points, samples, std)

    np.testing.assert_allclose(
        result_vmap, result_streaming, rtol=1e-7, atol=1e-10
    )


@pytest.mark.unit
def test_streaming_memory_profile():
    """Verify streaming uses less memory for high-dimensional cases.

    This test demonstrates the memory advantage of streaming for D >> 1.
    For D=10, n_samp=100, n_eval=50:
    - vmap: materializes (10, 100, 50) = 50,000 float32 = 200 KB
    - streaming: materializes (100, 50) = 5,000 float32 = 20 KB
    10× memory reduction
    """
    n_eval = 50
    n_samp = 100
    n_dims = 10  # High-dimensional case

    np.random.seed(456)
    eval_points = jnp.array(np.random.uniform(0, 100, (n_eval, n_dims)))
    samples = jnp.array(np.random.uniform(0, 100, (n_samp, n_dims)))
    std = jnp.array([5.0] * n_dims)

    # Both should give same results
    result_vmap = log_kde_distance(eval_points, samples, std)
    result_streaming = log_kde_distance_streaming(eval_points, samples, std)

    np.testing.assert_allclose(
        result_vmap, result_streaming, rtol=1e-6, atol=1e-9
    )

    # Memory usage note (not directly testable without profiler):
    # vmap peak: 10 × 100 × 50 = 50,000 elements
    # streaming peak: 100 × 50 = 5,000 elements
    # Reduction factor: 10× (equals n_dims)


@pytest.mark.unit
def test_streaming_with_different_stds():
    """Test streaming with non-uniform standard deviations."""
    n_eval = 10
    n_samp = 15
    n_dims = 3

    np.random.seed(789)
    eval_points = jnp.array(np.random.uniform(0, 100, (n_eval, n_dims)))
    samples = jnp.array(np.random.uniform(0, 100, (n_samp, n_dims)))
    std = jnp.array([2.0, 5.0, 10.0])  # Different stds per dimension

    result_vmap = log_kde_distance(eval_points, samples, std)
    result_streaming = log_kde_distance_streaming(eval_points, samples, std)

    np.testing.assert_allclose(
        result_vmap, result_streaming, rtol=1e-7, atol=1e-10
    )


@pytest.mark.unit
def test_streaming_single_dimension():
    """Test streaming for 1D case (edge case)."""
    n_eval = 25
    n_samp = 30
    n_dims = 1

    np.random.seed(101)
    eval_points = jnp.array(np.random.uniform(0, 100, (n_eval, n_dims)))
    samples = jnp.array(np.random.uniform(0, 100, (n_samp, n_dims)))
    std = jnp.array([5.0])

    result_vmap = log_kde_distance(eval_points, samples, std)
    result_streaming = log_kde_distance_streaming(eval_points, samples, std)

    np.testing.assert_allclose(
        result_vmap, result_streaming, rtol=1e-7, atol=1e-10
    )


@pytest.mark.unit
def test_streaming_output_shape():
    """Verify output shape is correct."""
    n_eval = 12
    n_samp = 18
    n_dims = 4

    eval_points = jnp.ones((n_eval, n_dims))
    samples = jnp.ones((n_samp, n_dims)) * 2.0
    std = jnp.ones(n_dims)

    result = log_kde_distance_streaming(eval_points, samples, std)

    assert result.shape == (n_samp, n_eval)
    assert jnp.all(jnp.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
