import numpy as np
import pytest

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ConfigurationError
from non_local_detector.tests.conftest import assert_stochastic_matrix

pytestmark = pytest.mark.unit


def test_random_walk_row_stochastic_and_spread_var(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    rw_tight = RandomWalk(
        environment_name=env.environment_name, movement_var=0.1
    ).make_state_transition(envs)
    rw_wide = RandomWalk(
        environment_name=env.environment_name, movement_var=10.0
    ).make_state_transition(envs)

    assert rw_tight.shape == (env.place_bin_centers_.shape[0],) * 2
    assert_stochastic_matrix(rw_tight, atol=1e-7)
    assert_stochastic_matrix(rw_wide, atol=1e-7)

    # Off-diagonal mass increases with larger variance
    offdiag_tight = (1.0 - np.diag(rw_tight)).mean()
    offdiag_wide = (1.0 - np.diag(rw_wide)).mean()
    assert offdiag_wide > offdiag_tight


def test_random_walk_with_directional_constraints(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    base = RandomWalk(
        environment_name=env.environment_name, movement_var=3.0
    ).make_state_transition(envs)
    up = RandomWalkDirection1(
        environment_name=env.environment_name, movement_var=3.0
    ).make_state_transition(envs)
    down = RandomWalkDirection2(
        environment_name=env.environment_name, movement_var=3.0
    ).make_state_transition(envs)

    # Directional versions are masked upper/lower triangular
    assert np.allclose(
        up,
        np.triu(base) / np.clip(np.triu(base).sum(axis=1, keepdims=True), 1e-12, None),
    )
    assert np.allclose(
        down,
        np.tril(base) / np.clip(np.tril(base).sum(axis=1, keepdims=True), 1e-12, None),
    )


def test_random_walk_invalid_direction_raises(simple_1d_environment):
    """Invalid direction string should raise ValueError."""
    env = simple_1d_environment
    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=3.0,
        direction="sideways",
    )
    with pytest.raises(ValueError, match="direction must be 'inward' or 'outward'"):
        rw.make_state_transition((env,))


def test_identity_transition_is_identity_masked(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    tm = Identity(environment_name=env.environment_name).make_state_transition(envs)
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert_stochastic_matrix(tm, atol=1e-7)
    assert np.allclose(np.diag(tm), 1.0)
    assert np.allclose(tm - np.diag(np.diag(tm)), 0.0)


def test_uniform_transition_within_env(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    tm = Uniform(environment_name=env.environment_name).make_state_transition(envs)
    n = env.place_bin_centers_.shape[0]
    assert tm.shape == (n, n)
    assert_stochastic_matrix(tm, atol=1e-7)
    # Each row is uniform
    assert np.allclose(tm, np.ones((n, n)) / n)


def test_uniform_transition_between_envs(simple_1d_environment):
    env1 = simple_1d_environment
    env2 = Environment(
        environment_name="env2", place_bin_size=1.0, position_range=((0.0, 4.0),)
    ).fit_place_grid(
        position=np.linspace(0.0, 4.0, 5)[:, None], infer_track_interior=False
    )
    envs = (env1, env2)
    tm = Uniform(
        environment_name=env1.environment_name, environment2_name=env2.environment_name
    ).make_state_transition(envs)
    assert tm.shape == (
        env1.place_bin_centers_.shape[0],
        env2.place_bin_centers_.shape[0],
    )
    # Rectangular (env1 bins x env2 bins), so check row sums directly.
    np.testing.assert_allclose(tm.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7)
    # Rows uniform over destination bins
    assert np.allclose(np.unique(tm, axis=1), np.ones((tm.shape[0], 1)) / tm.shape[1])


def test_empirical_movement_row_stochastic_and_locality(simple_1d_environment):
    # Create a simple path that marches right
    env = simple_1d_environment
    envs = (env,)
    pos = np.linspace(0.0, 10.0, 51)[:, None]
    # Use empirical transition with provided positions
    tm = EmpiricalMovement(
        environment_name=env.environment_name, speedup=1, is_time_reversed=False
    ).make_state_transition(envs, position=pos)
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert_stochastic_matrix(tm, atol=1e-6)
    # Empirical movement should favor near-diagonal transitions on average
    nearer = np.mean(np.diag(tm))
    farther = np.mean(tm[:, 0] + tm[:, -1])
    assert nearer >= farther


@pytest.mark.parametrize("n_dims", [1, 2])
@pytest.mark.parametrize("explicit_range", [False, True])
def test_empirical_movement_uses_fitted_edges(n_dims, explicit_range):
    """Ordinary fitted environments define both source and destination bins."""
    if n_dims == 1:
        position = np.array([[0.5], [1.5], [2.5], [0.5]])
    else:
        position = np.array(
            [[0.5, 0.5], [0.5, 2.5], [2.5, 2.5], [2.5, 0.5], [0.5, 0.5]]
        )
    env = Environment(
        place_bin_size=1.0,
        position_range=[(0.0, 3.0)] * n_dims if explicit_range else None,
        infer_track_interior=False,
    ).fit_place_grid(position, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    n_bins = len(env.place_bin_centers_)
    # Visit every fitted bin once, staying away from edge ownership choices.
    position = np.concatenate((env.place_bin_centers_, env.place_bin_centers_[:1]))
    bins: np.ndarray = np.concatenate((np.arange(n_bins), np.array([0], dtype=int)))
    expected = np.zeros((n_bins,) * 2)
    np.add.at(expected, (bins[:-1], bins[1:]), 1)
    row_mass = expected.sum(axis=1, keepdims=True)
    expected /= np.where(row_mass > 0, row_mass, 1)

    transition = EmpiricalMovement().make_state_transition((env,), position)

    np.testing.assert_array_equal(transition, expected)
    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0)
    np.testing.assert_array_equal(transition.sum(axis=1)[row_mass[:, 0] > 0], 1)


def test_discrete_transition_is_ones():
    tm = Discrete().make_state_transition()
    assert tm.shape == (1, 1) and np.isclose(tm[0, 0], 1.0)


def test_random_walk_inward_no_trackgraphDD_raises(simple_1d_environment):
    """Direction-aware RandomWalk needs an N-D track graph and array distances.

    ``_handle_no_track_graph`` previously dereferenced
    ``self.environment.track_graphDD`` and ``distance_between_nodes_`` without
    checking they were available, raising a confusing ``AttributeError`` /
    ``TypeError``. The precondition is now surfaced as ``ConfigurationError``.
    """
    env = simple_1d_environment
    # Simulate an environment whose direction-aware path has nothing to
    # operate on: no N-D track graph and no array-valued distance lookup.
    env.track_graph = None
    env.track_graphDD = None
    env.distance_between_nodes_ = None

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=3.0,
        direction="inward",
        use_manifold_distance=True,
    )
    # Match a phrase unique to the deep ``_handle_no_track_graph`` guard (not the
    # up-front ``make_state_transition`` guard, which also contains
    # "direction-aware RandomWalk"), so the test pins which guard fired.
    with pytest.raises(ConfigurationError, match="requires an N-D track graph"):
        rw.make_state_transition((env,))


def test_random_walk_direction_without_manifold_distance_raises(simple_1d_environment):
    """``direction`` is honored only with ``use_manifold_distance=True`` on an
    N-D grid. With the default ``use_manifold_distance=False`` the directional
    constraint would be silently dropped (a plain random walk), so it must raise
    instead of quietly ignoring the request.
    """
    env = simple_1d_environment
    env.track_graph = None

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=3.0,
        direction="inward",
        use_manifold_distance=False,  # default -> direction would be ignored
    )
    with pytest.raises(ConfigurationError, match="use_manifold_distance"):
        rw.make_state_transition((env,))
