import numpy as np

from non_local_detector.transitions.discrete.wrappers import diag_stickiness


def test_diag_stickiness_basic():
    diag_probs = [0.8, 0.6]
    result = diag_stickiness(diag_probs)
    expected_P = np.array([[0.8, 0.2], [0.4, 0.6]])
    np.testing.assert_allclose(result.matrix(), expected_P)
    assert result.concentration == 1.0
    assert result.stickiness == 0.0


def test_diag_stickiness_single_state():
    diag_probs = [1.0]
    result = diag_stickiness(diag_probs)
    expected_P = np.array([[1.0]])
    np.testing.assert_allclose(result.matrix(), expected_P)


def test_diag_stickiness_diag_probs_as_numpy_array():
    diag_probs = np.array([0.3, 0.7])
    result = diag_stickiness(diag_probs)
    expected_P = np.array([[0.3, 0.7], [0.3, 0.7]])
    np.fill_diagonal(expected_P, diag_probs)
    np.testing.assert_allclose(result.matrix(), expected_P)
