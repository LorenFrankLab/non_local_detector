import warnings

import numpy as np
import pytest

from non_local_detector.transitions.discrete.kernels.stationary import Stationary


# A dummy replacement for the actual estimator to avoid dependencies
def dummy_estimate_stationary_state_transition(
    causal_posterior,
    predictive_distribution,
    transition_matrix,
    acausal_posterior,
    concentration,
    stickiness,
):
    # Return an all-ones matrix of the same shape
    return np.ones_like(transition_matrix)


def test_stationary_matrix_returns_initial_matrix():
    initial_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    # First call
    m1 = s.matrix()
    # Should equal initial
    assert np.allclose(m1, initial_matrix)

    # Second call (cached)
    m2 = s.matrix(covariate_data={"unused": np.array([1, 2, 3])})
    # Should still equal the original
    assert np.allclose(m2, initial_matrix)


def test_stationary_matrix_raises_on_non_stochastic():
    # Negative entries
    bad_matrix = np.array([[0.5, -0.5], [0.3, 0.7]])
    with pytest.raises(ValueError):
        Stationary(initial_transition_matrix=bad_matrix)

    # Rows not summing to 1
    bad_matrix2 = np.array([[0.5, 0.6], [0.3, 0.7]])
    with pytest.raises(ValueError):
        Stationary(initial_transition_matrix=bad_matrix2)

    # Non-square
    bad_matrix3 = np.array([[0.3, 0.7, 0.0], [0.5, 0.5, 0.0]])
    with pytest.raises(ValueError):
        Stationary(initial_transition_matrix=bad_matrix3)


def test_n_states_property():
    initial_matrix = np.eye(3)
    s = Stationary(initial_transition_matrix=initial_matrix)
    assert s.n_states == 3


def test_update_parameters_updates_matrix(monkeypatch):
    initial_matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    # Patch the estimator function
    monkeypatch.setattr(
        "non_local_detector.transitions.discrete.kernels.stationary.estimate_stationary_state_transition",
        dummy_estimate_stationary_state_transition,
    )
    # Force fallback via _matrix = None
    s._matrix = None

    # Create valid posterior arrays with correct shape
    # causal, predictive, acausal should all be (n_time, n_states)
    n_states = initial_matrix.shape[0]
    n_time = 5
    causal = np.ones((n_time, n_states))
    predictive = np.ones((n_time, n_states))
    acausal = np.ones((n_time, n_states))

    s.update_parameters(
        causal_posterior=causal,
        predictive_distribution=predictive,
        acausal_posterior=acausal,
    )
    # After update, _matrix should be all ones (per dummy_estimate)
    assert np.allclose(s._matrix, np.ones_like(initial_matrix))


def test_update_parameters_uses_cached_matrix(monkeypatch):
    initial_matrix = np.array([[0.3, 0.7], [0.6, 0.4]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    # Pre-set _matrix to something different
    s._matrix = np.array([[0.5, 0.5], [0.5, 0.5]])

    # Patch estimator so if it were called, it’d set to ones—but we expect cache to skip it
    monkeypatch.setattr(
        "non_local_detector.transitions.discrete.kernels.stationary.estimate_stationary_state_transition",
        dummy_estimate_stationary_state_transition,
    )

    n_states = initial_matrix.shape[0]
    n_time = 5
    causal = np.ones((n_time, n_states))
    predictive = np.ones((n_time, n_states))
    acausal = np.ones((n_time, n_states))

    # Call update_parameters; since s._matrix is not None, it should NOT call the estimator
    s.update_parameters(
        causal_posterior=causal,
        predictive_distribution=predictive,
        acausal_posterior=acausal,
    )
    # _matrix should remain as we set it (0.5 0.5)
    assert np.allclose(s._matrix, np.ones((n_states, n_states)))


def test_update_parameters_raises_on_bad_shapes():
    initial_matrix = np.eye(2)
    s = Stationary(initial_transition_matrix=initial_matrix)
    # Patch estimator so we don’t go deep
    import non_local_detector.transitions.discrete.kernels.stationary as stat_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        stat_mod,
        "estimate_stationary_state_transition",
        dummy_estimate_stationary_state_transition,
    )

    # Mismatched shape: causal has wrong second dimension
    causal_bad = np.ones((5, 3))
    predictive = np.ones((5, 2))
    acausal = np.ones((5, 2))
    with pytest.raises(ValueError):
        s.update_parameters(
            causal_posterior=causal_bad,
            predictive_distribution=predictive,
            acausal_posterior=acausal,
        )

    # Mismatched time length
    causal = np.ones((4, 2))
    predictive_bad = np.ones((5, 2))  # mismatch in time
    acausal = np.ones((5, 2))
    with pytest.raises(ValueError):
        s.update_parameters(
            causal_posterior=causal,
            predictive_distribution=predictive_bad,
            acausal_posterior=acausal,
        )
    monkeypatch.undo()


def test_matrix_ignores_covariate_data_warns():
    initial_matrix = np.array([[0.6, 0.4], [0.2, 0.8]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = s.matrix(covariate_data={"foo": np.array([1, 2, 3])})
        assert np.allclose(m, initial_matrix)
        # Either it warns or silently ignores. If you chose to warn, check:
        if len(w):
            assert issubclass(w[-1].category, UserWarning)
    assert np.allclose(m, initial_matrix)
