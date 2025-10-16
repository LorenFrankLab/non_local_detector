"""Test parity and performance of GEMM vs loop-based mark kernel computation."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    _compute_log_mark_kernel_gemm,
    estimate_log_joint_mark_intensity,
    log_gaussian_pdf,
)


@pytest.fixture
def synthetic_waveform_data():
    """Generate synthetic waveform features for testing."""
    np.random.seed(42)

    n_encoding = 100
    n_decoding = 50
    n_features = 4
    n_position = 30

    # Generate random waveform features
    encoding_features = np.random.randn(n_encoding, n_features).astype(np.float32)
    decoding_features = np.random.randn(n_decoding, n_features).astype(np.float32)
    waveform_stds = np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5

    # Other parameters
    encoding_weights = np.ones(n_encoding, dtype=np.float32)
    occupancy = np.ones(n_position, dtype=np.float32)
    log_position_distance = np.random.randn(n_encoding, n_position).astype(np.float32)
    mean_rate = 5.0

    return {
        "encoding_features": jnp.array(encoding_features),
        "decoding_features": jnp.array(decoding_features),
        "waveform_stds": jnp.array(waveform_stds),
        "encoding_weights": jnp.array(encoding_weights),
        "occupancy": jnp.array(occupancy),
        "log_position_distance": jnp.array(log_position_distance),
        "mean_rate": mean_rate,
    }


def test_gemm_vs_loop_mark_kernel(synthetic_waveform_data):
    """Test that GEMM-based mark kernel matches loop-based implementation."""
    encoding_features = synthetic_waveform_data["encoding_features"]
    decoding_features = synthetic_waveform_data["decoding_features"]
    waveform_stds = synthetic_waveform_data["waveform_stds"]

    # Compute using GEMM
    logK_gemm = _compute_log_mark_kernel_gemm(
        decoding_features, encoding_features, waveform_stds
    )

    # Compute using loop
    n_enc = encoding_features.shape[0]
    n_dec = decoding_features.shape[0]
    logK_loop = jnp.zeros((n_enc, n_dec))
    for dec_dim, enc_dim, std_d in zip(
        decoding_features.T, encoding_features.T, waveform_stds, strict=False
    ):
        logK_loop += log_gaussian_pdf(
            x=jnp.expand_dims(dec_dim, axis=0),  # (1, n_dec)
            mean=jnp.expand_dims(enc_dim, axis=1),  # (n_enc, 1)
            sigma=std_d,
        )

    # Should be numerically equivalent
    assert logK_gemm.shape == logK_loop.shape
    assert np.allclose(logK_gemm, logK_loop, rtol=1e-5, atol=1e-6), (
        f"GEMM and loop mark kernels differ: "
        f"max diff = {np.abs(logK_gemm - logK_loop).max()}, "
        f"mean diff = {np.abs(logK_gemm - logK_loop).mean()}"
    )


def test_estimate_intensity_gemm_vs_loop(synthetic_waveform_data):
    """Test that estimate_log_joint_mark_intensity produces same results with GEMM vs loop."""
    # Compute with GEMM (default)
    log_joint_gemm = estimate_log_joint_mark_intensity(
        synthetic_waveform_data["decoding_features"],
        synthetic_waveform_data["encoding_features"],
        synthetic_waveform_data["encoding_weights"],
        synthetic_waveform_data["waveform_stds"],
        synthetic_waveform_data["occupancy"],
        synthetic_waveform_data["mean_rate"],
        synthetic_waveform_data["log_position_distance"],
        use_gemm=True,
    )

    # Compute with loop
    log_joint_loop = estimate_log_joint_mark_intensity(
        synthetic_waveform_data["decoding_features"],
        synthetic_waveform_data["encoding_features"],
        synthetic_waveform_data["encoding_weights"],
        synthetic_waveform_data["waveform_stds"],
        synthetic_waveform_data["occupancy"],
        synthetic_waveform_data["mean_rate"],
        synthetic_waveform_data["log_position_distance"],
        use_gemm=False,
    )

    # Should produce identical results
    assert log_joint_gemm.shape == log_joint_loop.shape
    assert np.allclose(log_joint_gemm, log_joint_loop, rtol=1e-5, atol=1e-6), (
        f"GEMM and loop intensity estimates differ: "
        f"max diff = {np.abs(log_joint_gemm - log_joint_loop).max()}, "
        f"mean diff = {np.abs(log_joint_gemm - log_joint_loop).mean()}"
    )


def test_gemm_various_feature_dimensions():
    """Test GEMM parity with different numbers of features."""
    np.random.seed(123)

    n_enc = 50
    n_dec = 20

    for n_features in [1, 2, 4, 8, 16]:
        encoding_features = jnp.array(
            np.random.randn(n_enc, n_features).astype(np.float32)
        )
        decoding_features = jnp.array(
            np.random.randn(n_dec, n_features).astype(np.float32)
        )
        waveform_stds = jnp.array(
            np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5
        )

        # GEMM
        logK_gemm = _compute_log_mark_kernel_gemm(
            decoding_features, encoding_features, waveform_stds
        )

        # Loop
        logK_loop = jnp.zeros((n_enc, n_dec))
        for dec_dim, enc_dim, std_d in zip(
            decoding_features.T, encoding_features.T, waveform_stds, strict=False
        ):
            logK_loop += log_gaussian_pdf(
                x=jnp.expand_dims(dec_dim, axis=0),
                mean=jnp.expand_dims(enc_dim, axis=1),
                sigma=std_d,
            )

        assert np.allclose(logK_gemm, logK_loop, rtol=1e-5, atol=1e-6), (
            f"Failed for n_features={n_features}: "
            f"max diff = {np.abs(logK_gemm - logK_loop).max()}"
        )


def test_gemm_edge_cases():
    """Test GEMM with edge cases like single spike, identical features, etc."""
    waveform_stds = jnp.array([1.0, 1.0, 1.0])

    # Single encoding spike
    encoding_single = jnp.array([[1.0, 2.0, 3.0]])
    decoding_multi = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logK_gemm = _compute_log_mark_kernel_gemm(
        decoding_multi, encoding_single, waveform_stds
    )
    assert logK_gemm.shape == (1, 2)
    assert jnp.all(jnp.isfinite(logK_gemm))

    # Single decoding spike
    encoding_multi = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    decoding_single = jnp.array([[1.0, 2.0, 3.0]])
    logK_gemm = _compute_log_mark_kernel_gemm(
        decoding_single, encoding_multi, waveform_stds
    )
    assert logK_gemm.shape == (2, 1)
    assert jnp.all(jnp.isfinite(logK_gemm))

    # Identical features (should give maximum kernel value)
    encoding_identical = jnp.array([[1.0, 2.0, 3.0]])
    decoding_identical = jnp.array([[1.0, 2.0, 3.0]])
    logK_gemm_identical = _compute_log_mark_kernel_gemm(
        decoding_identical, encoding_identical, waveform_stds
    )

    # Very different features (should give low kernel value)
    decoding_different = jnp.array([[10.0, 20.0, 30.0]])
    logK_gemm_different = _compute_log_mark_kernel_gemm(
        decoding_different, encoding_identical, waveform_stds
    )

    # Identical features should have higher kernel value (less negative in log space)
    assert logK_gemm_identical[0, 0] > logK_gemm_different[0, 0]


def test_gemm_normalization_constant():
    """Verify that GEMM includes proper Gaussian normalization constant."""
    n_features = 4
    waveform_stds = jnp.array([0.5, 1.0, 1.5, 2.0])

    # Expected normalization constant
    expected_log_norm = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )

    # For identical points, kernel should be exp(log_norm_const)
    # Test with same point
    same_point = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    logK_same = _compute_log_mark_kernel_gemm(same_point, same_point, waveform_stds)

    # At identical points, ||x - y||^2 = 0, so logK = log_norm_const
    assert np.allclose(logK_same[0, 0], expected_log_norm, rtol=1e-5, atol=1e-6), (
        f"Normalization constant incorrect: "
        f"expected {expected_log_norm}, got {logK_same[0, 0]}"
    )


def test_gemm_gradient_compatibility():
    """Verify that GEMM implementation is compatible with JAX autodiff."""
    import jax

    np.random.seed(999)

    encoding_features = jnp.array(np.random.randn(10, 4).astype(np.float32))
    decoding_features = jnp.array(np.random.randn(5, 4).astype(np.float32))
    waveform_stds = jnp.array([1.0, 1.0, 1.0, 1.0])

    # Define function to differentiate
    def loss_fn(dec_features):
        logK = _compute_log_mark_kernel_gemm(
            dec_features, encoding_features, waveform_stds
        )
        return jnp.sum(logK)

    # Should be able to compute gradient
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(decoding_features)

    assert grad.shape == decoding_features.shape
    assert jnp.all(jnp.isfinite(grad))
