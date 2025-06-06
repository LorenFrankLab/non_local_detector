import numpy as np
import pytest

from non_local_detector.transitions.discrete.priors import get_dirichlet_prior


def test_dirichlet_prior_with_scalar_stickiness():
    concentration = 0.5
    stickiness = 2.0
    n_states = 3
    prior = get_dirichlet_prior(concentration, stickiness, n_states)

    # Expected: for i=j: concentration + stickiness, for i!=j: concentration
    base = concentration * np.ones((n_states, n_states))
    base += stickiness * np.eye(n_states)
    expected = np.maximum(base, 1.0)

    np.testing.assert_array_almost_equal(prior, expected)
    assert prior.shape == (n_states, n_states)
    # Off-diagonal entries should equal max(concentration, 1.0) = 1.0
    off_diag = prior[0, 1]
    assert off_diag == pytest.approx(1.0)


def test_dirichlet_prior_with_array_stickiness():
    concentration = 1.0
    stickiness = np.array([0.2, 0.5, 0.8])
    n_states = 3
    prior = get_dirichlet_prior(concentration, stickiness, n_states)

    # Build expected: row i, col i = concentration + stickiness[i], other cols = concentration
    base = concentration * np.ones((n_states, n_states))
    base += np.diag(stickiness)
    expected = np.maximum(base, 1.0)

    np.testing.assert_array_almost_equal(prior, expected)
    assert prior.shape == (n_states, n_states)


def test_dirichlet_prior_minimum_value():
    concentration = 0.0
    stickiness = 0.0
    n_states = 4
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    assert np.all(prior >= 1.0)
    # All entries should be exactly 1.0
    assert np.all(prior == 1.0)


def test_dirichlet_prior_with_negative_concentration():
    concentration = -2.0
    stickiness = 0.0
    n_states = 2
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    # concentration < 0 ⇒ all entries floored to 1.0
    assert np.all(prior == 1.0)


def test_dirichlet_prior_with_large_stickiness():
    concentration = 0.1
    stickiness = 10.0
    n_states = 2
    prior = get_dirichlet_prior(concentration, stickiness, n_states)
    base = concentration * np.ones((n_states, n_states))
    base += stickiness * np.eye(n_states)
    expected = np.maximum(base, 1.0)
    np.testing.assert_array_almost_equal(prior, expected)


def test_dirichlet_prior_invalid_stickiness_shape():
    """Passing a 1D array whose length != n_states should raise ValueError."""
    concentration = 1.0
    n_states = 3
    bad_stickiness = np.array([0.2, 0.5])  # length 2, but n_states=3
    with pytest.raises(ValueError):
        get_dirichlet_prior(concentration, bad_stickiness, n_states)


def test_dirichlet_prior_invalid_stickiness_type():
    """Passing a non-scalar, non-ndarray stickiness should raise TypeError."""
    concentration = 1.0
    stickiness = "not_an_array"  # wrong type
    with pytest.raises(TypeError):
        get_dirichlet_prior(concentration, stickiness, n_states=2)


def test_dirichlet_prior_zero_states():
    """n_states=0 is invalid—should raise a ValueError."""
    with pytest.raises(ValueError):
        get_dirichlet_prior(concentration=1.0, stickiness=0.0, n_states=0)
