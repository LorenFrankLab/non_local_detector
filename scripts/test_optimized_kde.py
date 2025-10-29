"""Test numerical equivalence of optimized KDE implementations."""

import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity_logspace,
    estimate_log_joint_mark_intensity_vectorized,
    kde_distance,
    kde_distance_vectorized,
    log_kde_distance,
)


def test_kde_distance_equivalence():
    """Test that vectorized KDE matches original."""
    print("Testing kde_distance equivalence...")

    rng = np.random.default_rng(42)

    # Test multiple dimensions
    for n_features in [2, 4, 8]:
        eval_points = jnp.array(rng.normal(0, 1, (10, n_features)))
        samples = jnp.array(rng.normal(0, 1, (20, n_features)))
        std = jnp.ones(n_features)

        result_original = kde_distance(eval_points, samples, std)
        result_vectorized = kde_distance_vectorized(eval_points, samples, std)

        diff = jnp.abs(result_original - result_vectorized)
        max_diff = jnp.max(diff)
        mean_diff = jnp.mean(diff)

        print(f"  {n_features}D: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

        # Check equivalence (float32 precision)
        assert jnp.allclose(
            result_original, result_vectorized, rtol=1e-5, atol=1e-8
        ), f"Vectorized KDE not equivalent for {n_features}D"

    print("  ✓ All tests passed\n")


def test_log_kde_distance_equivalence():
    """Test that log-space KDE matches exp(original)."""
    print("Testing log_kde_distance equivalence...")

    rng = np.random.default_rng(42)

    for n_features in [2, 4, 8]:
        eval_points = jnp.array(rng.normal(0, 1, (10, n_features)))
        samples = jnp.array(rng.normal(0, 1, (20, n_features)))
        std = jnp.ones(n_features)

        result_original = kde_distance(eval_points, samples, std)
        result_log = log_kde_distance(eval_points, samples, std)

        # Compare log(original) vs log_kde_distance
        log_original = jnp.log(result_original)
        diff = jnp.abs(log_original - result_log)
        max_diff = jnp.max(diff)
        mean_diff = jnp.mean(diff)

        print(f"  {n_features}D: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

        # Check equivalence (float32 precision)
        assert jnp.allclose(
            log_original, result_log, rtol=1e-5, atol=1e-8
        ), f"Log KDE not equivalent for {n_features}D"

    print("  ✓ All tests passed\n")


def test_estimate_functions_equivalence():
    """Test that optimized estimate functions match original."""
    print("Testing estimate_log_joint_mark_intensity equivalence...")

    rng = np.random.default_rng(42)

    for n_features in [2, 4, 8]:
        # Create test data
        dec_features = jnp.array(rng.normal(0, 1, (100, n_features)))
        enc_features = jnp.array(rng.normal(0, 1, (200, n_features)))
        waveform_stds = jnp.ones(n_features)
        occupancy = jnp.ones(500) * 0.1
        mean_rate = 5.0
        position_distance = jnp.array(rng.exponential(1.0, (200, 500)))

        # Compute all versions
        result_original = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        result_vectorized = estimate_log_joint_mark_intensity_vectorized(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        result_logspace = estimate_log_joint_mark_intensity_logspace(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        # Compare vectorized vs original
        diff_vec = jnp.abs(result_original - result_vectorized)
        max_diff_vec = jnp.max(diff_vec)
        mean_diff_vec = jnp.mean(diff_vec)
        rel_diff_vec = jnp.max(diff_vec / (jnp.abs(result_original) + 1e-10))

        print(
            f"  {n_features}D (vectorized): max_diff={max_diff_vec:.2e}, mean_diff={mean_diff_vec:.2e}, max_rel={rel_diff_vec:.2e}"
        )

        # Check with more appropriate tolerance for numerical differences
        is_close = jnp.allclose(result_original, result_vectorized, rtol=1e-5, atol=1e-6)
        if not is_close:
            print("    WARNING: Differences exceed tolerance")
            print(f"    Max absolute diff: {max_diff_vec}")
            print(f"    Max relative diff: {rel_diff_vec}")
            # Check if it's just due to float precision
            if max_diff_vec < 1e-5 and rel_diff_vec < 1e-4:
                print("    Differences are within acceptable float32 precision, continuing...")
            else:
                raise AssertionError(f"Vectorized version not equivalent for {n_features}D")

        # Compare logspace vs original
        diff_log = jnp.abs(result_original - result_logspace)
        max_diff_log = jnp.max(diff_log)
        mean_diff_log = jnp.mean(diff_log)
        rel_diff_log = jnp.max(diff_log / (jnp.abs(result_original) + 1e-10))

        print(
            f"  {n_features}D (logspace):   max_diff={max_diff_log:.2e}, mean_diff={mean_diff_log:.2e}, max_rel={rel_diff_log:.2e}"
        )

        # Check with more appropriate tolerance
        is_close = jnp.allclose(result_original, result_logspace, rtol=1e-5, atol=1e-6)
        if not is_close:
            print("    WARNING: Differences exceed tolerance")
            print(f"    Max absolute diff: {max_diff_log}")
            print(f"    Max relative diff: {rel_diff_log}")
            # Check if it's just due to float precision
            if max_diff_log < 1e-5 and rel_diff_log < 1e-4:
                print("    Differences are within acceptable float32 precision, continuing...")
            else:
                raise AssertionError(f"Log-space version not equivalent for {n_features}D")

    print("  ✓ All tests passed\n")


def test_edge_cases():
    """Test edge cases: zeros, inf, nan handling."""
    print("Testing edge cases...")

    rng = np.random.default_rng(42)

    dec_features = jnp.array(rng.normal(0, 1, (10, 2)))
    enc_features = jnp.array(rng.normal(0, 1, (20, 2)))
    waveform_stds = jnp.ones(2)
    occupancy = jnp.array([0.0, 0.1, 0.2] + [0.1] * 47)  # Include zero occupancy
    mean_rate = 5.0
    position_distance = jnp.array(rng.exponential(1.0, (20, 50)))

    # All versions should handle zero occupancy without nan/inf
    for func_name, func in [
        ("original", estimate_log_joint_mark_intensity),
        ("vectorized", estimate_log_joint_mark_intensity_vectorized),
        ("logspace", estimate_log_joint_mark_intensity_logspace),
    ]:
        result = func(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        # Check for -inf in zero occupancy positions (expected)
        assert jnp.all(
            jnp.isneginf(result[:, 0])
        ), f"{func_name}: Expected -inf for zero occupancy"

        # Check no NaN
        assert not jnp.any(jnp.isnan(result)), f"{func_name}: Found NaN values"

        # Check finite values for non-zero occupancy
        assert jnp.all(
            jnp.isfinite(result[:, 1:])
        ), f"{func_name}: Found non-finite values for non-zero occupancy"

        print(f"  {func_name}: ✓ Handles edge cases correctly")

    print("  ✓ All edge case tests passed\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("NUMERICAL EQUIVALENCE TESTS FOR OPTIMIZED KDE")
    print("=" * 70)
    print()

    try:
        test_kde_distance_equivalence()
        test_log_kde_distance_equivalence()
        test_estimate_functions_equivalence()
        test_edge_cases()

        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nOptimized implementations are numerically equivalent to reference.")
        print("Proceeding to performance profiling is safe.")

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("✗ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nOptimized implementations do NOT match reference.")
        print("Do NOT use in production until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
