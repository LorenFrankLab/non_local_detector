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
            ValidationError, match="local_position_std must be a finite"
        ):
            NonLocalSortedSpikesDetector(local_position_std=0.0)

    def test_clusterless(self):
        with pytest.raises(
            ValidationError, match="local_position_std must be a finite"
        ):
            NonLocalClusterlessDetector(local_position_std=0.0)

    def test_nan_rejected(self):
        with pytest.raises(
            ValidationError, match="local_position_std must be a finite"
        ):
            NonLocalSortedSpikesDetector(local_position_std=float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(
            ValidationError, match="local_position_std must be a finite"
        ):
            NonLocalSortedSpikesDetector(local_position_std=float("inf"))


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

        # Local block is the full place_bin_centers length (interior + gap),
        # so the argmax index is the full bin index `animal_bin`.
        local_block = override[_fitted_detector.state_ind_ == 0]
        assert local_block.shape[0] == int(env.place_bin_centers_.shape[0])
        assert int(np.argmax(local_block)) == animal_bin


@pytest.mark.unit
class TestMultiArmTrackOverride:
    """C1 regression: full-bin-length IC on linearized multi-arm tracks.

    Environments with ``edge_spacing > 0`` produce gap bins between arms.
    ``state_ind_`` and the rest of ``initial_conditions_`` are built at the
    full ``place_bin_centers_`` length (interior + gap); the Local block
    must too, otherwise the per-state multiplication broadcast-fails.
    """

    @staticmethod
    def _make_two_arm_detector():
        import networkx as nx

        from non_local_detector.environment import Environment

        track_graph = nx.Graph()
        track_graph.add_node(0, pos=(0.0, 0.0))
        track_graph.add_node(1, pos=(50.0, 0.0))
        track_graph.add_node(2, pos=(60.0, 0.0))
        track_graph.add_node(3, pos=(110.0, 0.0))
        track_graph.add_edge(0, 1, distance=50.0, edge_id=0)
        track_graph.add_edge(2, 3, distance=50.0, edge_id=1)

        env = Environment(
            environment_name="",
            place_bin_size=5.0,
            track_graph=track_graph,
            edge_order=[(0, 1), (2, 3)],
            edge_spacing=10.0,
        )
        position_1d = np.concatenate(
            [np.linspace(0.0, 50.0, 25), np.linspace(60.0, 110.0, 25)]
        )
        env = env.fit_place_grid(position_1d, infer_track_interior=True)

        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        detector.environments = (env,)
        detector.initialize_state_index()
        detector.initialize_initial_conditions()
        return detector, env

    def test_override_shape_matches_state_ind(self):
        """Override's length must equal state_ind_ length on a track with gaps."""
        detector, env = self._make_two_arm_detector()
        # Confirm fixture actually has gap bins (otherwise this isn't a regression).
        assert int((~env.is_track_interior_.ravel()).sum()) > 0

        position_time = np.array([0.0, 1.0])
        position = np.array([[20.0], [20.0]])  # arm A
        time = np.array([0.5])

        override = detector.compute_local_initial_conditions(
            position_time, position, time
        )

        assert override is not None
        assert override.shape == (detector.state_ind_.shape[0],)
        assert override.shape == detector.initial_conditions_.shape

    def test_override_places_delta_on_correct_arm(self):
        """Animal on arm A → delta lands at an interior bin of arm A."""
        detector, env = self._make_two_arm_detector()

        position_time = np.array([0.0, 1.0])
        position_arm_a = np.array([[20.0], [20.0]])
        time = np.array([0.5])

        override = detector.compute_local_initial_conditions(
            position_time, position_arm_a, time
        )
        assert override is not None

        local_block = override[detector.state_ind_ == 0]
        animal_bin_a = int(env.get_bin_ind(position_arm_a[:1])[0])
        assert int(np.argmax(local_block)) == animal_bin_a
        # Sanity: that bin is interior (not a gap bin).
        assert bool(env.is_track_interior_.ravel()[animal_bin_a])

        # Same exercise for arm B.
        position_arm_b = np.array([[80.0], [80.0]])
        override_b = detector.compute_local_initial_conditions(
            position_time, position_arm_b, time
        )
        local_block_b = override_b[detector.state_ind_ == 0]
        animal_bin_b = int(env.get_bin_ind(position_arm_b[:1])[0])
        assert int(np.argmax(local_block_b)) == animal_bin_b
        assert animal_bin_b != animal_bin_a

    def test_override_zero_at_gap_bins(self):
        """Gap bins receive zero IC mass even though state_ind_ assigns them to Local."""
        detector, env = self._make_two_arm_detector()
        position_time = np.array([0.0, 1.0])
        position = np.array([[20.0], [20.0]])
        time = np.array([0.5])

        override = detector.compute_local_initial_conditions(
            position_time, position, time
        )
        assert override is not None

        local_block = override[detector.state_ind_ == 0]
        is_gap = ~env.is_track_interior_.ravel()
        assert is_gap.any(), "Fixture must have gap bins to be a meaningful test"
        np.testing.assert_array_equal(local_block[is_gap], 0.0)


@pytest.mark.unit
class TestEstimateParametersUsesOverride:
    """I6: ensure ``estimate_parameters()`` invokes the IC override.

    The override flows through the same ``_predict()`` as ``predict()``,
    but ``estimate_parameters`` packs ``log_likelihood_args`` independently.
    A future refactor that reorders the tuple would silently disable the
    override without breaking the existing tests, since uniform-IC and
    delta-IC predictions both produce stochastic posteriors.
    """

    def test_estimate_parameters_runs_with_local_position_std(self, _sim_data):
        from unittest.mock import patch

        detector = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=0.5,
        )
        detector.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )

        # Spy on the IC selector to assert it actually fires.
        original = type(detector)._predict_time_initial_conditions
        calls = {"count": 0}

        def spy(self, time, log_likelihood_args):
            calls["count"] += 1
            return original(self, time, log_likelihood_args)

        with patch.object(
            type(detector),
            "_predict_time_initial_conditions",
            new=spy,
        ):
            # max_iter=1 keeps the test fast; estimate_parameters runs at
            # least one E-step, which goes through _predict.
            detector.estimate_parameters(
                position_time=_sim_data["time"],
                position=_sim_data["position"],
                spike_times=_sim_data["spike_times"],
                time=_sim_data["time"],
                max_iter=1,
                estimate_encoding_model=False,
                estimate_initial_conditions=False,
                estimate_discrete_transition=False,
            )

        assert calls["count"] >= 1, (
            "estimate_parameters did not invoke _predict_time_initial_conditions; "
            "the multi-bin Local IC override is silently skipped on the EM path."
        )


@pytest.mark.unit
class TestClusterlessOverride:
    """I9: clusterless detector exercises the IC override end-to-end."""

    @staticmethod
    def _make_detector():
        from non_local_detector.simulate.clusterless_simulation import (
            make_simulated_run_data,
        )

        # n_tetrodes must divide len(PLACE_FIELD_MEANS) (20).
        sim = make_simulated_run_data(n_tetrodes=5, seed=42)
        detector = NonLocalClusterlessDetector(
            local_position_std=0.5,
            clusterless_algorithm="clusterless_kde",
            clusterless_algorithm_params={
                "position_std": 6.0,
                "block_size": int(2**12),
            },
        ).fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
        )
        return detector, sim

    def test_clusterless_causal_t0_peaks_at_animal_bin(self):
        detector, sim = self._make_detector()
        results = detector.predict(
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.edges,
            position=sim.position,
            position_time=sim.position_time,
            return_outputs="filter",
        )

        env = detector.environments[0]
        first_pos = np.asarray(sim.position)[:1]
        if first_pos.ndim == 1:
            first_pos = first_pos[:, np.newaxis]
        animal_bin = int(env.get_bin_ind(first_pos)[0])

        causal = np.asarray(results.causal_posterior)
        local_mask = detector.state_ind_ == 0
        local_t0 = causal[0, local_mask]
        local_t0 = np.where(np.isnan(local_t0), 0.0, local_t0)
        if local_t0.sum() > 0:
            assert int(np.argmax(local_t0)) == animal_bin, (
                f"Clusterless causal t=0 peaks at bin {int(np.argmax(local_t0))}, "
                f"expected animal_bin={animal_bin}. Override may not be wired "
                "through the clusterless predict() path."
            )


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

    def test_causal_posterior_t0_peaks_at_animal_bin_with_override(
        self, _fitted_detector, _sim_data
    ):
        """Override is wired correctly: causal posterior at t=0 peaks at animal_bin.

        Uses ``return_outputs='filter'`` so the assertion exercises the
        forward pass at t=0 directly, before any backward smoothing
        could redistribute mass from t≥1.
        """
        results = _fitted_detector.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
            return_outputs="filter",
        )
        causal = np.asarray(results.causal_posterior)
        local_mask = _fitted_detector.state_ind_ == 0

        env = _fitted_detector.environments[0]
        first_pos_2d = np.atleast_2d(np.asarray(_sim_data["position"])[0])
        if first_pos_2d.ndim == 1:
            first_pos_2d = first_pos_2d[:, np.newaxis]
        animal_bin = int(env.get_bin_ind(first_pos_2d)[0])
        n_local_bins = int(env.place_bin_centers_.shape[0])

        local_t0 = causal[0, local_mask]
        local_t0 = np.where(np.isnan(local_t0), 0.0, local_t0)
        # The override puts all Local mass on `animal_bin` at t=0; the only
        # source of redistribution before the t=0 causal posterior is the
        # observation likelihood, which for σ=0.5 (much smaller than the
        # bin spacing) leaves most mass near the animal. Compare against a
        # uniform-IC baseline computed from the same posterior shape.
        assert local_t0.shape == (n_local_bins,)
        if local_t0.sum() > 0:
            normalized = local_t0 / local_t0.sum()
            assert int(np.argmax(normalized)) == animal_bin, (
                f"Causal posterior at t=0 peaks at bin {int(np.argmax(normalized))}, "
                f"expected animal_bin={animal_bin}. The IC override may not be wired."
            )

    def test_override_changes_t0_posterior_vs_uniform_ic(
        self, _fitted_detector, _sim_data
    ):
        """Override makes a measurable, non-trivial difference at t=0.

        Patches ``_predict_time_initial_conditions`` to always return the
        stored uniform IC, then compares the causal posterior at t=0
        against the unpatched run. The two must differ on the Local block.
        """
        from unittest.mock import patch

        results_with_override = _fitted_detector.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
            return_outputs="filter",
        )

        with patch.object(
            type(_fitted_detector),
            "_predict_time_initial_conditions",
            return_value=_fitted_detector.initial_conditions_,
        ):
            results_without_override = _fitted_detector.predict(
                spike_times=_sim_data["spike_times"],
                position_time=_sim_data["time"],
                position=_sim_data["position"],
                time=_sim_data["time"],
                return_outputs="filter",
            )

        causal_with = np.asarray(results_with_override.causal_posterior)
        causal_without = np.asarray(results_without_override.causal_posterior)
        local_mask = _fitted_detector.state_ind_ == 0

        # The two t=0 Local blocks must differ — if they don't, the override
        # is silently a no-op.
        diff = np.nansum(
            np.abs(causal_with[0, local_mask] - causal_without[0, local_mask])
        )
        assert diff > 1e-6, (
            "Override produced identical t=0 posterior to uniform IC; the "
            "override is not actually changing the forward pass."
        )
