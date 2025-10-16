import numpy as np

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


def make_env_1d(n_bins=11, name="line"):
    env = Environment(
        environment_name=name,
        place_bin_size=1.0,
        position_range=((0.0, float(n_bins - 1)),),
    )
    # Provide dummy position to fit grid reliably
    pos = np.linspace(0.0, float(n_bins - 1), n_bins)[:, None]
    env = env.fit_place_grid(position=pos, infer_track_interior=False)
    return env


def rows_sum_to_one(mat: np.ndarray, atol=1e-7) -> bool:
    return np.allclose(mat.sum(axis=1), 1.0, atol=atol)


def test_random_walk_row_stochastic_and_spread_var():
    env = make_env_1d(n_bins=21)
    envs = (env,)
    rw_tight = RandomWalk(
        environment_name=env.environment_name, movement_var=0.1
    ).make_state_transition(envs)
    rw_wide = RandomWalk(
        environment_name=env.environment_name, movement_var=10.0
    ).make_state_transition(envs)

    assert rw_tight.shape == (env.place_bin_centers_.shape[0],) * 2
    assert rows_sum_to_one(rw_tight) and rows_sum_to_one(rw_wide)
    assert np.all(rw_tight >= 0.0) and np.all(rw_wide >= 0.0)

    # Off-diagonal mass increases with larger variance
    offdiag_tight = (1.0 - np.diag(rw_tight)).mean()
    offdiag_wide = (1.0 - np.diag(rw_wide)).mean()
    assert offdiag_wide > offdiag_tight


def test_random_walk_with_directional_constraints():
    env = make_env_1d(n_bins=15)
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


def test_identity_transition_is_identity_masked():
    env = make_env_1d(n_bins=10)
    envs = (env,)
    tm = Identity(environment_name=env.environment_name).make_state_transition(envs)
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert rows_sum_to_one(tm)
    assert np.allclose(np.diag(tm), 1.0)
    assert np.allclose(tm - np.diag(np.diag(tm)), 0.0)


def test_uniform_transition_within_env():
    env = make_env_1d(n_bins=12)
    envs = (env,)
    tm = Uniform(environment_name=env.environment_name).make_state_transition(envs)
    n = env.place_bin_centers_.shape[0]
    assert tm.shape == (n, n)
    assert rows_sum_to_one(tm)
    # Each row is uniform
    assert np.allclose(tm, np.ones((n, n)) / n)


def test_uniform_transition_between_envs():
    env1 = make_env_1d(n_bins=8, name="env1")
    env2 = make_env_1d(n_bins=5, name="env2")
    envs = (env1, env2)
    tm = Uniform(
        environment_name=env1.environment_name, environment2_name=env2.environment_name
    ).make_state_transition(envs)
    assert tm.shape == (
        env1.place_bin_centers_.shape[0],
        env2.place_bin_centers_.shape[0],
    )
    assert rows_sum_to_one(tm)
    # Rows uniform over destination bins
    assert np.allclose(np.unique(tm, axis=1), np.ones((tm.shape[0], 1)) / tm.shape[1])


def test_empirical_movement_row_stochastic_and_locality():
    # Create a simple path that marches right
    env = make_env_1d(n_bins=11)
    # Ensure histogram range set for 2D (pre- and post- positions)
    env.position_range = ((0.0, 10.0), (0.0, 10.0))
    envs = (env,)
    pos = np.linspace(0.0, 10.0, 51)[:, None]
    # Use empirical transition with provided positions
    tm = EmpiricalMovement(
        environment_name=env.environment_name, speedup=1, is_time_reversed=False
    ).make_state_transition(envs, position=pos)
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert rows_sum_to_one(tm, atol=1e-6)
    # Empirical movement should favor near-diagonal transitions on average
    nearer = np.mean(np.diag(tm))
    farther = np.mean(tm[:, 0] + tm[:, -1])
    assert nearer >= farther


def test_discrete_transition_is_ones():
    tm = Discrete().make_state_transition()
    assert tm.shape == (1, 1) and np.isclose(tm[0, 0], 1.0)
