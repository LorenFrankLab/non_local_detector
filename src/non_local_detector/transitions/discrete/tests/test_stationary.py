import numpy as np

from non_local_detector.transitions.discrete.stationary import Stationary


class DummyDiscreteTransitionModel:
    pass


def dummy_estimate_stationary_state_transition(
    causal_posterior,
    predictive_distribution,
    transition_matrix,
    acausal_posterior,
    concentration,
    stickiness,
):
    # Just return a matrix of ones for testing
    return np.ones_like(transition_matrix)


def test_stationary_matrix_returns_initial_matrix(monkeypatch):
    initial_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    assert np.allclose(s.matrix(), initial_matrix)


def test_stationary_matrix_caches(monkeypatch):
    initial_matrix = np.array([[0.2, 0.8], [0.5, 0.5]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    # First call sets _matrix
    m1 = s.matrix()
    # Change initial_transition_matrix to see if cache is used
    s.initial_transition_matrix[:] = 0
    m2 = s.matrix()
    assert np.allclose(m1, m2)
    assert np.allclose(m2, initial_matrix)


def test_n_states_property():
    initial_matrix = np.eye(3)
    s = Stationary(initial_transition_matrix=initial_matrix)
    assert s.n_states == 3


def test_update_parameters_updates_matrix(monkeypatch):
    initial_matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    # Patch the estimator
    monkeypatch.setattr(
        "non_local_detector.transitions.discrete.stationary.estimate_stationary_state_transition",
        dummy_estimate_stationary_state_transition,
    )
    # Set _matrix to None to test fallback to initial_transition_matrix
    s._matrix = None
    causal = np.array([0, 1])
    predictive = np.array([1, 0])
    acausal = np.array([0, 1])
    s.update_parameters(
        causal_posterior=causal,
        predictive_distribution=predictive,
        acausal_posterior=acausal,
    )
    assert np.all(s._matrix == 1)


def test_update_parameters_uses_cached_matrix(monkeypatch):
    initial_matrix = np.array([[0.3, 0.7], [0.6, 0.4]])
    s = Stationary(initial_transition_matrix=initial_matrix)
    s._matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    monkeypatch.setattr(
        "non_local_detector.transitions.discrete.stationary.estimate_stationary_state_transition",
        dummy_estimate_stationary_state_transition,
    )
    causal = np.array([0, 1])
    predictive = np.array([1, 0])
    acausal = np.array([0, 1])
    s.update_parameters(
        causal_posterior=causal,
        predictive_distribution=predictive,
        acausal_posterior=acausal,
    )
    assert np.all(s._matrix == 1)
