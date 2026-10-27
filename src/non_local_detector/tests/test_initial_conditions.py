import warnings

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.initial_conditions import (
    UniformInitialConditions,
    estimate_initial_conditions,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.tests.conftest import assert_probability_distribution


@pytest.mark.unit
def test_uniform_initial_conditions_local_and_no_spike(simple_1d_environment):
    ic = UniformInitialConditions()
    envs = (simple_1d_environment,)

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


@pytest.mark.unit
def test_uniform_initial_conditions_nonlocal_with_mask(simple_1d_environment):
    env = simple_1d_environment
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


@pytest.mark.unit
def test_estimate_initial_conditions_returns_first_row():
    post = np.random.default_rng(0).random((7, 5))
    post = post / post.sum(axis=1, keepdims=True)
    init = estimate_initial_conditions(post)
    assert init.shape == (post.shape[1],)
    assert np.allclose(init, post[0])


@pytest.mark.unit
def test_uniform_initial_conditions_no_track_interior(simple_1d_environment):
    """Test when environment has no track interior mask."""
    env = simple_1d_environment
    env.is_track_interior_ = None  # No mask
    envs = (env,)
    ic = UniformInitialConditions()
    obs = ObservationModel(environment_name=env.environment_name, is_local=False)

    # Act
    arr = ic.make_initial_conditions(obs, envs)

    # Assert - should fallback to single bin
    assert arr.shape == (1,)
    assert np.isclose(arr[0], 1.0)


@pytest.mark.unit
def test_uniform_initial_conditions_normalization(simple_1d_environment):
    """Test that initial conditions always sum to 1."""
    env = simple_1d_environment
    envs = (env,)
    ic = UniformInitialConditions()
    obs = ObservationModel(environment_name=env.environment_name, is_local=False)

    # Act
    arr = ic.make_initial_conditions(obs, envs)

    # Assert
    assert_probability_distribution(arr)


@pytest.fixture(scope="module")
def em_fit_with_unreachable_states():
    """A detector after one EM step whose discrete initial conditions put zero
    mass on the No-Spike (1) and Fragmented (3) states, so the smoothed
    probability of those states at the first time bin is exactly zero.

    Any ``RuntimeWarning`` (e.g. 0/0 in the continuous IC update) is raised.
    """
    _, position, spike_times, time, _, _, _, _ = make_simulated_data(
        seed=42, n_neurons=5
    )
    n_time = 4000
    time, position = time[:n_time], position[:n_time]
    detector = NonLocalSortedSpikesDetector(
        discrete_initial_conditions=np.array([0.5, 0.0, 0.5, 0.0])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # max_iter=1 never converges
        warnings.simplefilter("error", RuntimeWarning)
        detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=[times[times <= time[-1]] for times in spike_times],
            time_edges=time_edges_from_centers(time),
            max_iter=1,
            estimate_encoding_model=False,
            estimate_initial_conditions=True,
        )
    return detector


@pytest.mark.integration
class TestContinuousICUpdate:
    """The EM update of ``continuous_initial_conditions_`` divides the joint
    initial conditions by the smoothed discrete-state probability at the
    first time bin, guarding states whose probability is zero."""

    def test_zero_probability_states_get_zero_not_nan(
        self, em_fit_with_unreachable_states
    ):
        detector = em_fit_with_unreachable_states
        np.testing.assert_array_equal(
            detector.discrete_initial_conditions_[[1, 3]], 0.0
        )
        continuous_ic = detector.continuous_initial_conditions_
        assert continuous_ic.shape == detector.state_ind_.shape
        assert np.all(np.isfinite(continuous_ic))
        is_unreachable = np.isin(detector.state_ind_, [1, 3])
        np.testing.assert_array_equal(continuous_ic[is_unreachable], 0.0)

    def test_nonzero_states_are_joint_ic_over_discrete_ic(
        self, em_fit_with_unreachable_states
    ):
        detector = em_fit_with_unreachable_states
        for state in (0, 2):
            is_state = detector.state_ind_ == state
            discrete_ic = detector.discrete_initial_conditions_[state]
            assert discrete_ic > 0
            np.testing.assert_allclose(
                detector.continuous_initial_conditions_[is_state],
                detector.initial_conditions_[is_state] / discrete_ic,
                rtol=1e-12,
            )
            # Each reachable state's continuous IC is a distribution over its
            # own bins.
            np.testing.assert_allclose(
                detector.continuous_initial_conditions_[is_state].sum(),
                1.0,
                rtol=1e-5,
            )
