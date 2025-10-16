import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import (
    UniformInitialConditions,
    estimate_initial_conditions,
)
from non_local_detector.observation_models import ObservationModel

from non_local_detector.tests.conftest import assert_probability_distribution


@pytest.fixture
def make_env_1d():
    """Factory fixture for creating 1D environments with custom parameters."""
    def _make_env(n_bins=11):
        env = Environment(
            environment_name="line",
            place_bin_size=1.0,
            position_range=((0.0, float(n_bins - 1)),),
        )
        pos = np.linspace(0.0, float(n_bins - 1), n_bins)[:, None]
        env = env.fit_place_grid(position=pos, infer_track_interior=False)
        return env
    return _make_env


def test_uniform_initial_conditions_local_and_no_spike(make_env_1d):
    ic = UniformInitialConditions()
    envs = (make_env_1d(),)

    # Local decoding => single-bin initial conditions
    local_obs = ObservationModel(
        environment_name=envs[0].environment_name, is_local=True
    )
    arr = ic.make_initial_conditions(local_obs, envs)
    assert arr.shape == (1,)
    assert np.isclose(arr.sum(), 1.0) and np.isclose(arr[0], 1.0)

    # No-spike => single-bin initial conditions
    ns_obs = ObservationModel(
        environment_name=envs[0].environment_name, is_no_spike=True
    )
    arr2 = ic.make_initial_conditions(ns_obs, envs)
    assert arr2.shape == (1,) and np.isclose(arr2[0], 1.0)


def test_uniform_initial_conditions_nonlocal_with_mask(make_env_1d):
    env = make_env_1d(n_bins=10)
    # Mask out last two bins
    mask = np.ones(env.centers_shape_, dtype=bool)
    mask[-2:] = False
    env.is_track_interior_ = mask
    envs = (env,)
    ic = UniformInitialConditions()
    obs = ObservationModel(environment_name=env.environment_name, is_local=False)
    arr = ic.make_initial_conditions(obs, envs)
    assert arr.shape == (env.place_bin_centers_.shape[0],)
    assert np.isclose(arr.sum(), 1.0)
    # Zero probability on masked bins
    assert np.all(arr[-2:] == 0.0)
    # Uniform across unmasked bins
    assert np.allclose(arr[:-2], 1.0 / (env.place_bin_centers_.shape[0] - 2))


def test_estimate_initial_conditions_returns_first_row():
    post = np.random.default_rng(0).random((7, 5))
    post = post / post.sum(axis=1, keepdims=True)
    init = estimate_initial_conditions(post)
    assert init.shape == (post.shape[1],)
    assert np.allclose(init, post[0])


def test_uniform_initial_conditions_no_track_interior(make_env_1d):
    """Test when environment has no track interior mask."""
    env = make_env_1d(n_bins=5)
    env.is_track_interior_ = None  # No mask
    envs = (env,)
    ic = UniformInitialConditions()
    obs = ObservationModel(environment_name=env.environment_name, is_local=False)

    # Act
    arr = ic.make_initial_conditions(obs, envs)

    # Assert - should fallback to single bin
    assert arr.shape == (1,)
    assert np.isclose(arr[0], 1.0)


def test_uniform_initial_conditions_normalization(make_env_1d):
    """Test that initial conditions always sum to 1."""
    env = make_env_1d(n_bins=20)
    envs = (env,)
    ic = UniformInitialConditions()
    obs = ObservationModel(environment_name=env.environment_name, is_local=False)

    # Act
    arr = ic.make_initial_conditions(obs, envs)

    # Assert
    assert_probability_distribution(arr)


def test_estimate_initial_conditions_preserves_distribution():
    """Estimate should preserve the probability distribution at t=0."""
    # Arrange - create a non-uniform initial distribution
    post = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])

    # Act
    init = estimate_initial_conditions(post)

    # Assert
    assert_probability_distribution(init)
    assert np.array_equal(init, post[0])
