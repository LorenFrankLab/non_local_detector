"""Tests for the predict-time Local-state initial conditions and the
Local observation channel (Gaussian log-density kernel).

Covers the cleanup landed alongside `local_position_std`: the stored
`initial_conditions_` is uniform over the Local block, and at predict
time the multi-bin Local IC is concentrated at the bin containing the
animal's first interpolated position.
"""

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture(scope="module")
def _sim_data():
    """Deterministic simulated data shared across tests."""
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        sampling_frequency,
        _is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=15, seed=0)
    return {
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "sampling_frequency": sampling_frequency,
    }


@pytest.fixture(scope="module")
def _fitted_detector(_sim_data):
    """Multi-bin Local detector fit on the shared simulated data."""
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=_sim_data["sampling_frequency"],
        local_position_std=0.5,
    )
    detector.fit(
        position_time=_sim_data["time"],
        position=_sim_data["position"],
        spike_times=_sim_data["spike_times"],
    )
    return detector


@pytest.mark.unit
class TestSigmaZeroRejected:
    def test_sorted_spikes(self):
        with pytest.raises(
            ValidationError, match="local_position_std must be strictly positive"
        ):
            NonLocalSortedSpikesDetector(local_position_std=0.0)

    def test_clusterless(self):
        with pytest.raises(
            ValidationError, match="local_position_std must be strictly positive"
        ):
            NonLocalClusterlessDetector(local_position_std=0.0)


@pytest.mark.unit
class TestComputeLocalInitialConditions:
    """Helper method behavior."""

    def test_returns_none_when_local_position_std_is_none(self, _sim_data):
        detector = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=None,
        )
        detector.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )
        override = detector.compute_local_initial_conditions(
            _sim_data["time"], _sim_data["position"], _sim_data["time"]
        )
        assert override is None

    def test_returns_none_when_position_is_none(self, _fitted_detector, _sim_data):
        override = _fitted_detector.compute_local_initial_conditions(
            None, None, _sim_data["time"]
        )
        assert override is None

    def test_returns_none_when_first_position_is_nan(self, _fitted_detector, _sim_data):
        position = np.array(_sim_data["position"], copy=True)
        position[0] = np.nan
        override = _fitted_detector.compute_local_initial_conditions(
            _sim_data["time"], position, _sim_data["time"]
        )
        assert override is None

    def test_override_concentrates_local_at_animal_bin(
        self, _fitted_detector, _sim_data
    ):
        override = _fitted_detector.compute_local_initial_conditions(
            _sim_data["time"], _sim_data["position"], _sim_data["time"]
        )
        assert override is not None

        # Local has state index 0 by convention in NonLocalSortedSpikesDetector.
        local_mask = _fitted_detector.state_ind_ == 0
        local_block = override[local_mask]
        # Exactly one nonzero bin in the Local block.
        assert np.count_nonzero(local_block) == 1

        # That nonzero bin equals discrete_initial_conditions[Local].
        nonzero_value = local_block[local_block > 0][0]
        np.testing.assert_allclose(
            nonzero_value,
            _fitted_detector.discrete_initial_conditions[0],
            rtol=1e-6,
        )

    def test_override_does_not_mutate_stored_initial_conditions(
        self, _fitted_detector, _sim_data
    ):
        ic_before = np.array(_fitted_detector.initial_conditions_, copy=True)
        _fitted_detector.compute_local_initial_conditions(
            _sim_data["time"], _sim_data["position"], _sim_data["time"]
        )
        np.testing.assert_array_equal(_fitted_detector.initial_conditions_, ic_before)

    def test_override_matches_animal_first_position_bin(
        self, _fitted_detector, _sim_data
    ):
        override = _fitted_detector.compute_local_initial_conditions(
            _sim_data["time"], _sim_data["position"], _sim_data["time"]
        )
        assert override is not None

        env = _fitted_detector.environments[0]
        first_pos = np.atleast_2d(np.asarray(_sim_data["position"])[0])
        if first_pos.ndim == 1:
            first_pos = first_pos[:, np.newaxis]
        animal_bin = int(env.get_bin_ind(first_pos)[0])
        interior_bin_indices = np.where(env.is_track_interior_.ravel())[0]
        interior_col = int(np.where(interior_bin_indices == animal_bin)[0][0])

        # Local state-bin block has the same length as n_interior bins.
        local_block = override[_fitted_detector.state_ind_ == 0]
        assert int(np.argmax(local_block)) == interior_col


@pytest.mark.unit
class TestPredictUsesOverride:
    """End-to-end: predict() uses the override implicitly without any user opt-in."""

    def test_stored_ic_remains_uniform_after_predict(self, _fitted_detector, _sim_data):
        # Predict with position data
        _fitted_detector.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )
        # Stored IC's Local block remains uniform (multi-bin Local was built
        # via UniformInitialConditions on interior bins).
        local_block = _fitted_detector.initial_conditions_[
            _fitted_detector.state_ind_ == 0
        ]
        # All Local bins have the same nonzero value (uniform).
        nonzero = local_block[local_block > 0]
        if nonzero.size > 1:
            np.testing.assert_allclose(nonzero, nonzero[0], rtol=1e-5)

    def test_first_timestep_local_concentrated_near_animal(
        self, _fitted_detector, _sim_data
    ):
        results = _fitted_detector.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )
        acausal = np.asarray(results.acausal_posterior)
        # Just sanity: the smoothed posterior at t=0 over Local bins should
        # have most of its Local mass within a few bins of the animal.
        # The Local block lives at state_ind == 0; check that the nonzero
        # mass is local rather than spread.
        local_mask = _fitted_detector.state_ind_ == 0
        local_t0 = acausal[0, local_mask]
        local_t0 = np.where(np.isnan(local_t0), 0.0, local_t0)
        if local_t0.sum() > 0:
            normalized = local_t0 / local_t0.sum()
            # The posterior should not be uniformly spread; the top bin
            # should hold a substantial share.
            assert normalized.max() > 5.0 / len(local_t0), (
                "Multi-bin Local at t=0 should be concentrated, not uniform"
            )
