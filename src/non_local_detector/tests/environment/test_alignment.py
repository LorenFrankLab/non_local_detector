import numpy as np
import pytest

from non_local_detector.environment.alignment import (
    apply_similarity_transform,
    get_2d_rotation_matrix,
    map_probabilities_to_nearest_target_bin,
)


# Minimal mock Environment class for testing
class MockEnvironment:
    def __init__(self, bin_centers, n_dims, is_fitted=True):
        self.bin_centers = bin_centers
        self.n_dims = n_dims
        self._is_fitted = is_fitted


def test_get_2d_rotation_matrix_identity():
    R = get_2d_rotation_matrix(0)
    np.testing.assert_allclose(R, np.eye(2), atol=1e-8)


def test_get_2d_rotation_matrix_90_deg():
    R = get_2d_rotation_matrix(90)
    expected = np.array([[0, -1], [1, 0]])
    np.testing.assert_allclose(R, expected, atol=1e-8)


def test_get_2d_rotation_matrix_180_deg():
    R = get_2d_rotation_matrix(180)
    expected = np.array([[-1, 0], [0, -1]])
    np.testing.assert_allclose(R, expected, atol=1e-8)


def test_apply_similarity_transform_identity():
    points = np.array([[1, 2], [3, 4]])
    R = np.eye(2)
    s = 1.0
    t = np.zeros(2)
    transformed = apply_similarity_transform(points, R, s, t)
    np.testing.assert_allclose(transformed, points)


def test_apply_similarity_transform_rotation_scale_translate():
    points = np.array([[1, 0], [0, 1]])
    R = get_2d_rotation_matrix(90)
    s = 2.0
    t = np.array([1, 1])
    transformed = apply_similarity_transform(points, R, s, t)
    expected = np.array([[1, 3], [-1, 1]])
    np.testing.assert_allclose(transformed, expected, atol=1e-8)


def test_apply_similarity_transform_empty_points():
    points = np.empty((0, 2))
    R = np.eye(2)
    s = 1.0
    t = np.zeros(2)
    transformed = apply_similarity_transform(points, R, s, t)
    assert transformed.size == 0


def test_apply_similarity_transform_invalid_shapes():
    points = np.array([[1, 2, 3]])
    R = np.eye(2)
    s = 1.0
    t = np.zeros(2)
    with pytest.raises(ValueError):
        apply_similarity_transform(points, R, s, t)


def test_map_probabilities_to_nearest_target_bin_basic():
    # 2D, 3 bins each, no transform
    src_bins = np.array([[0, 0], [1, 0], [0, 1]])
    tgt_bins = np.array([[0, 0], [1, 0], [0, 1]])
    src_probs = np.array([0.2, 0.5, 0.3])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 2)
    tgt_probs = map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)
    np.testing.assert_allclose(tgt_probs, src_probs)


def test_map_probabilities_to_nearest_target_bin_with_transform():
    # Rotate source by 90 deg, so [1,0] -> [0,1], [0,1] -> [-1,0]
    src_bins = np.array([[1, 0], [0, 1]])
    tgt_bins = np.array([[0, 1], [-1, 0]])
    src_probs = np.array([0.7, 0.3])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 2)
    R = get_2d_rotation_matrix(90)
    tgt_probs = map_probabilities_to_nearest_target_bin(
        src_env, tgt_env, src_probs, source_rotation_matrix=R
    )
    # After rotation: [1,0]->[0,1], [0,1]->[-1,0]
    np.testing.assert_allclose(tgt_probs, [0.7, 0.3])


def test_map_probabilities_to_nearest_target_bin_duplicate_mapping():
    # Two source bins map to same target bin
    src_bins = np.array([[0, 0], [0.1, 0]])
    tgt_bins = np.array([[0, 0]])
    src_probs = np.array([0.4, 0.6])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 2)
    tgt_probs = map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)
    np.testing.assert_allclose(tgt_probs, [1.0])


def test_map_probabilities_to_nearest_target_bin_empty_source():
    src_bins = np.empty((0, 2))
    tgt_bins = np.array([[0, 0], [1, 1]])
    src_probs = np.array([])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 2)
    tgt_probs = map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)
    np.testing.assert_allclose(tgt_probs, [0, 0])


def test_map_probabilities_to_nearest_target_bin_empty_target():
    src_bins = np.array([[0, 0]])
    tgt_bins = np.empty((0, 2))
    src_probs = np.array([1.0])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 2)
    tgt_probs = map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)
    assert tgt_probs.size == 0


def test_map_probabilities_to_nearest_target_bin_not_fitted():
    src_bins = np.array([[0, 0]])
    tgt_bins = np.array([[0, 0]])
    src_probs = np.array([1.0])
    src_env = MockEnvironment(src_bins, 2, is_fitted=False)
    tgt_env = MockEnvironment(tgt_bins, 2)
    with pytest.raises(RuntimeError):
        map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)


def test_map_probabilities_to_nearest_target_bin_missing_bin_centers():
    src_env = MockEnvironment(None, 2)
    tgt_env = MockEnvironment(np.array([[0, 0]]), 2)
    src_probs = np.array([])
    with pytest.raises(ValueError):
        map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)


def test_map_probabilities_to_nearest_target_bin_shape_mismatch():
    src_bins = np.array([[0, 0], [1, 1]])
    tgt_bins = np.array([[0, 0], [1, 1]])
    src_probs = np.array([1.0])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 2)
    with pytest.raises(ValueError):
        map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)


def test_map_probabilities_to_nearest_target_bin_dim_mismatch():
    src_bins = np.array([[0, 0]])
    tgt_bins = np.array([[0, 0, 0]])
    src_probs = np.array([1.0])
    src_env = MockEnvironment(src_bins, 2)
    tgt_env = MockEnvironment(tgt_bins, 3)
    with pytest.raises(ValueError):
        map_probabilities_to_nearest_target_bin(src_env, tgt_env, src_probs)
