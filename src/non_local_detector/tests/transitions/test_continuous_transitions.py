import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

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


@pytest.fixture
def make_env_1d():
    """Factory fixture for creating 1D environments with custom parameters."""

    def _make_env(n_bins=11, name="line"):
        env = Environment(
            environment_name=name,
            place_bin_size=1.0,
            position_range=((0.0, float(n_bins - 1)),),
        )
        pos = np.linspace(0.0, float(n_bins - 1), n_bins)[:, None]
        env = env.fit_place_grid(position=pos, infer_track_interior=False)
        return env

    return _make_env


def rows_sum_to_one(mat: np.ndarray, atol=1e-7) -> bool:
    return np.allclose(mat.sum(axis=1), 1.0, atol=atol)


def test_random_walk_row_stochastic_and_spread_var(make_env_1d):
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


def test_random_walk_with_directional_constraints(make_env_1d):
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


def test_identity_transition_is_identity_masked(make_env_1d):
    env = make_env_1d(n_bins=10)
    envs = (env,)
    tm = Identity(environment_name=env.environment_name).make_state_transition(envs)
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert rows_sum_to_one(tm)
    assert np.allclose(np.diag(tm), 1.0)
    assert np.allclose(tm - np.diag(np.diag(tm)), 0.0)


def test_uniform_transition_within_env(make_env_1d):
    env = make_env_1d(n_bins=12)
    envs = (env,)
    tm = Uniform(environment_name=env.environment_name).make_state_transition(envs)
    n = env.place_bin_centers_.shape[0]
    assert tm.shape == (n, n)
    assert rows_sum_to_one(tm)
    # Each row is uniform
    assert np.allclose(tm, np.ones((n, n)) / n)


def test_uniform_transition_between_envs(make_env_1d):
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


def test_empirical_movement_row_stochastic_and_locality(make_env_1d):
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


# ============================================================================
# SNAPSHOT TESTS
# ============================================================================


def serialize_transition_matrix_summary(trans_mat: np.ndarray) -> dict:
    """Serialize transition matrix to summary for snapshot comparison.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transition matrix

    Returns
    -------
    summary : dict
        Summary statistics suitable for snapshot comparison
    """
    return {
        "shape": trans_mat.shape,
        "dtype": str(trans_mat.dtype),
        "diagonal_mean": float(np.mean(np.diag(trans_mat)))
        if trans_mat.ndim == 2 and trans_mat.shape[0] == trans_mat.shape[1]
        else None,
        "diagonal_std": float(np.std(np.diag(trans_mat)))
        if trans_mat.ndim == 2 and trans_mat.shape[0] == trans_mat.shape[1]
        else None,
        "row_sums": np.sum(trans_mat, axis=-1).tolist()
        if trans_mat.size <= 500
        else {
            "mean": float(np.mean(np.sum(trans_mat, axis=-1))),
            "min": float(np.min(np.sum(trans_mat, axis=-1))),
            "max": float(np.max(np.sum(trans_mat, axis=-1))),
        },
        "mean": float(np.mean(trans_mat)),
        "std": float(np.std(trans_mat)),
        "min": float(np.min(trans_mat)),
        "max": float(np.max(trans_mat)),
        "values": trans_mat.tolist()
        if trans_mat.size <= 100
        else {
            "first_row": trans_mat[0].tolist() if trans_mat.shape[0] > 0 else [],
            "last_row": trans_mat[-1].tolist() if trans_mat.shape[0] > 0 else [],
            "diagonal_sample": np.diag(trans_mat)[:10].tolist()
            if trans_mat.ndim == 2 and trans_mat.shape[0] == trans_mat.shape[1]
            else None,
        },
    }


@pytest.mark.snapshot
def test_random_walk_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for RandomWalk transition matrix."""
    env = make_env_1d(n_bins=15)
    envs = (env,)

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=2.0,
        movement_mean=0.0,
    ).make_state_transition(envs)

    assert serialize_transition_matrix_summary(rw) == snapshot


@pytest.mark.snapshot
def test_random_walk_with_mean_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for RandomWalk with non-zero mean displacement."""
    env = make_env_1d(n_bins=12)
    envs = (env,)

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=1.5,
        movement_mean=0.5,
    ).make_state_transition(envs)

    assert serialize_transition_matrix_summary(rw) == snapshot


@pytest.mark.snapshot
def test_random_walk_direction1_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for RandomWalkDirection1 (upper triangular)."""
    env = make_env_1d(n_bins=10)
    envs = (env,)

    rw_dir1 = RandomWalkDirection1(
        environment_name=env.environment_name, movement_var=2.5
    ).make_state_transition(envs)

    assert serialize_transition_matrix_summary(rw_dir1) == snapshot


@pytest.mark.snapshot
def test_random_walk_direction2_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for RandomWalkDirection2 (lower triangular)."""
    env = make_env_1d(n_bins=10)
    envs = (env,)

    rw_dir2 = RandomWalkDirection2(
        environment_name=env.environment_name, movement_var=2.5
    ).make_state_transition(envs)

    assert serialize_transition_matrix_summary(rw_dir2) == snapshot


@pytest.mark.snapshot
def test_uniform_within_environment_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for Uniform transition within same environment."""
    env = make_env_1d(n_bins=8)
    envs = (env,)

    uniform = Uniform(environment_name=env.environment_name).make_state_transition(envs)

    assert serialize_transition_matrix_summary(uniform) == snapshot


@pytest.mark.snapshot
def test_uniform_between_environments_snapshot(
    make_env_1d, snapshot: SnapshotAssertion
):
    """Snapshot test for Uniform transition between two environments."""
    env1 = make_env_1d(n_bins=6, name="env1")
    env2 = make_env_1d(n_bins=9, name="env2")
    envs = (env1, env2)

    uniform = Uniform(
        environment_name=env1.environment_name, environment2_name=env2.environment_name
    ).make_state_transition(envs)

    assert serialize_transition_matrix_summary(uniform) == snapshot


@pytest.mark.snapshot
def test_identity_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for Identity transition matrix."""
    env = make_env_1d(n_bins=7)
    envs = (env,)

    identity = Identity(environment_name=env.environment_name).make_state_transition(
        envs
    )

    assert serialize_transition_matrix_summary(identity) == snapshot


@pytest.mark.snapshot
def test_empirical_movement_snapshot(make_env_1d, snapshot: SnapshotAssertion):
    """Snapshot test for EmpiricalMovement transition matrix."""
    env = make_env_1d(n_bins=11)
    env.position_range = ((0.0, 10.0), (0.0, 10.0))
    envs = (env,)

    # Create consistent position trajectory
    np.random.seed(42)
    pos = np.linspace(0.0, 10.0, 50)[:, None]

    empirical = EmpiricalMovement(
        environment_name=env.environment_name, speedup=1, is_time_reversed=False
    ).make_state_transition(envs, position=pos)

    assert serialize_transition_matrix_summary(empirical) == snapshot


@pytest.mark.snapshot
def test_empirical_movement_time_reversed_snapshot(
    make_env_1d, snapshot: SnapshotAssertion
):
    """Snapshot test for EmpiricalMovement with time reversal."""
    env = make_env_1d(n_bins=11)
    env.position_range = ((0.0, 10.0), (0.0, 10.0))
    envs = (env,)

    # Create consistent position trajectory
    np.random.seed(123)
    pos = np.linspace(0.0, 10.0, 50)[:, None]

    empirical = EmpiricalMovement(
        environment_name=env.environment_name, speedup=1, is_time_reversed=True
    ).make_state_transition(envs, position=pos)

    assert serialize_transition_matrix_summary(empirical) == snapshot


@pytest.mark.snapshot
def test_discrete_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for Discrete transition matrix."""
    discrete = Discrete().make_state_transition()

    assert serialize_transition_matrix_summary(discrete) == snapshot
