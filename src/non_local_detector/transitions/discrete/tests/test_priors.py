import numpy as np
import pytest

from non_local_detector.transitions.discrete.priors import get_dirichlet_prior


def test_dirichlet_prior_with_scalar_stickiness():
    concentration = 0.5
    stickiness = 2.0
    n_states = 3
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    expected = concentration * np.ones((n_states,)) + stickiness * np.eye(n_states)
    expected = np.maximum(expected, 1.0)
    np.testing.assert_array_almost_equal(prior, expected)
    assert prior.shape == (n_states, n_states)


def test_dirichlet_prior_with_array_stickiness():
    concentration = 1.0
    stickiness = np.array([0.2, 0.5, 0.8])
    n_states = 3
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    expected = concentration * np.ones((n_states,)) + np.diag(stickiness)
    expected = np.maximum(expected, 1.0)
    np.testing.assert_array_almost_equal(prior, expected)
    assert prior.shape == (n_states, n_states)


def test_dirichlet_prior_minimum_value():
    concentration = 0.0
    stickiness = 0.0
    n_states = 4
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    assert np.all(prior >= 1.0)


def test_dirichlet_prior_with_negative_concentration():
    concentration = -2.0
    stickiness = 0.0
    n_states = 2
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    assert np.all(prior == 1.0)


def test_dirichlet_prior_with_large_stickiness():
    concentration = 0.1
    stickiness = 10.0
    n_states = 2
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    expected = concentration * np.ones((n_states,)) + stickiness * np.eye(n_states)
    expected = np.maximum(expected, 1.0)
    np.testing.assert_array_almost_equal(prior, expected)
