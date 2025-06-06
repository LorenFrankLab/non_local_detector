import numpy as np
import pytest

from non_local_detector.transitions.discrete.kernels.stationary import Stationary
from non_local_detector.transitions.discrete.wrappers import diag_stickiness


def test_diag_stickiness_basic():
    diag_probs = [0.8, 0.6]
    result = diag_stickiness(diag_probs)
    # The stationary transition matrix should be:
    # P[0,0]=0.8, P[0,1]=0.2
    # P[1,1]=0.6, P[1,0]=0.4
    expected_P = np.array([[0.8, 0.2], [0.4, 0.6]])
    np.testing.assert_allclose(result.matrix(), expected_P)
    assert result.concentration == 1.0
    # By default, stickiness=0.0 → becomes np.array([0.0, 0.0])
    assert np.all(result.stickiness == 0.0)
    assert isinstance(result, Stationary)


def test_diag_stickiness_single_state():
    diag_probs = [1.0]
    result = diag_stickiness(diag_probs)
    expected_P = np.array([[1.0]])
    np.testing.assert_allclose(result.matrix(), expected_P)
    assert result.n_states == 1


def test_diag_stickiness_diag_probs_as_numpy_array():
    diag_probs = np.array([0.3, 0.7])
    result = diag_stickiness(diag_probs, concentration=2.0, stickiness=1.5)
    # Build expected P matrix by the same logic:
    # P[0,0] = 0.3, P[0,1] = 0.7
    # P[1,1] = 0.7, P[1,0] = 0.3
    # Wait: That yields a uniform row for state 1? No:
    # Actually, for state i: P[i,i] = diag_probs[i], P[i, j ≠ i] = 1 - diag_probs[i].
    expected_P = np.array([[0.3, 0.7], [0.3, 0.7]])
    np.fill_diagonal(expected_P, diag_probs)
    np.testing.assert_allclose(result.matrix(), expected_P)
    # Check hyperparams forwarded
    assert result.concentration == 2.0
    # Internally, stickiness=1.5 → full vector length 2
    stick_vec = np.array([1.5, 1.5])
    assert np.allclose(result.stickiness, stick_vec)


@pytest.mark.parametrize(
    "bad_input",
    [
        0.5,  # scalar—not valid
        [[0.2, 0.8]],  # 2D list
        np.array([[0.3, 0.7]]),  # 2D array
    ],
)
def test_diag_stickiness_invalid_shape(bad_input):
    with pytest.raises(ValueError):
        diag_stickiness(bad_input)


@pytest.mark.parametrize(
    "diag_probs",
    [
        [-0.1, 0.5],  # negative entry
        [1.2, 0.0],  # >1 entry
    ],
)
def test_diag_stickiness_out_of_range(diag_probs):
    with pytest.raises(ValueError):
        diag_stickiness(diag_probs)
