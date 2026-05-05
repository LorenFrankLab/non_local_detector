"""Tests for the predict-time Local-state initial conditions and the
Local observation channel (unnormalized spatial-anchor kernel).

Covers the cleanup landed alongside `local_position_std`: the stored
`initial_conditions_` is uniform over the Local block, and at predict
time the multi-bin Local IC is concentrated at the bin containing the
animal's first interpolated position.
"""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@contextmanager
def _spy_compute_local_initial_conditions(detector_cls):
    """Yield a Mock that wraps ``compute_local_initial_conditions``.

    Use ``mock.call_count`` to assert the override fired during a decode call.
    """
    original = detector_cls.compute_local_initial_conditions
    with patch.object(
        detector_cls,
        "compute_local_initial_conditions",
        autospec=True,
        side_effect=original,
    ) as mock_method:
        yield mock_method


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
class TestSigmaValidator:
    """``local_position_std`` accepts None / 0 / positive finite; rejects NaN / Inf / negative."""

    def test_zero_accepted_sorted_spikes(self):
        detector = NonLocalSortedSpikesDetector(local_position_std=0.0)
        assert detector.local_position_std == 0.0

    def test_zero_accepted_clusterless(self):
        detector = NonLocalClusterlessDetector(local_position_std=0.0)
        assert detector.local_position_std == 0.0

    def test_nan_rejected(self):
        with pytest.raises(ValidationError, match="local_position_std"):
            NonLocalSortedSpikesDetector(local_position_std=float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValidationError, match="local_position_std"):
            NonLocalSortedSpikesDetector(local_position_std=float("inf"))


@pytest.mark.unit
class TestDeltaKernel:
    """``local_position_std=0`` produces a one-hot kernel at the animal's bin per timestep."""

    @staticmethod
    def _make_detector_and_env():
        detector = NonLocalSortedSpikesDetector(local_position_std=0.0)
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()
        return detector, detector.environments[0]

    def test_kernel_is_one_hot_at_animal_bin(self):
        import jax.numpy as jnp

        detector, env = self._make_detector_and_env()
        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[50.0], [50.0]])

        log_kernel = np.asarray(
            detector._compute_local_position_kernel(
                jnp.array(time),
                jnp.array(position_time),
                jnp.array(animal_position),
                env,
            )
        )

        finite_mask = np.isfinite(log_kernel[0])
        assert finite_mask.sum() == 1, "Delta kernel must be finite at exactly one bin"
        # Mass balance: the lone finite cell carries log(n_bins) so
        # exp(log_kernel) sums to n_bins (compensates for the multi-bin
        # Local state's uniform 1/n_bins continuous IC).
        n_bins_interior = int(env.is_track_interior_.sum())
        np.testing.assert_allclose(
            log_kernel[0, finite_mask], np.log(n_bins_interior), atol=1e-5
        )
        assert np.all(log_kernel[0, ~finite_mask] == -np.inf)
        np.testing.assert_allclose(
            float(np.exp(log_kernel[0, finite_mask]).sum()),
            float(n_bins_interior),
            rtol=1e-5,
        )

        # The finite bin is the one containing the animal.
        interior_bin_indices = np.where(env.is_track_interior_.ravel())[0]
        expected_bin = int(env.get_bin_ind(np.array([[50.0]]))[0])
        expected_col = int(np.where(interior_bin_indices == expected_bin)[0][0])
        assert int(np.argmax(log_kernel[0])) == expected_col

    def test_nan_position_falls_back_to_flat_kernel(self):
        import jax.numpy as jnp

        detector, env = self._make_detector_and_env()
        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[np.nan], [np.nan]])

        log_kernel = np.asarray(
            detector._compute_local_position_kernel(
                jnp.array(time),
                jnp.array(position_time),
                jnp.array(animal_position),
                env,
            )
        )
        assert np.all(np.isfinite(log_kernel))
        np.testing.assert_allclose(log_kernel[0], 0.0, atol=1e-7)


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

    def test_first_position_nan_leaves_local_block_unchanged(
        self, _fitted_detector, _sim_data
    ):
        """NaN at t=0 → Local block falls back to stored IC, not a delta.

        The override walks each Local observation model; if the first
        decoding frame's position is NaN it skips that state (logs a
        warning) and leaves its block as-is. Functionally equivalent to
        the legacy ``return None`` path for single-Local models, but
        per-state so multi-environment configurations with one
        droppedout Local don't disable overrides for the rest.
        """
        position = np.array(_sim_data["position"], copy=True)
        position[0] = np.nan
        override = _fitted_detector.compute_local_initial_conditions(
            _sim_data["time"], position, _sim_data["time"]
        )
        # Helper still returns an array (single-Local case → just a copy
        # of the stored IC); Local block must equal the stored block.
        assert override is not None
        local_mask = _fitted_detector.state_ind_ == 0
        np.testing.assert_array_equal(
            override[local_mask], _fitted_detector.initial_conditions_[local_mask]
        )

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

    def test_override_preserves_em_updated_non_local_blocks(self, _sim_data):
        """EM-updated non-Local IC rows survive the override.

        The override starts from a copy of ``self.initial_conditions_`` and
        only writes the Local block, so any EM updates baked into
        ``initial_conditions_`` (non-Local rows or
        ``discrete_initial_conditions``) are preserved verbatim.
        """
        detector = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=0.5,
        )
        detector.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )

        # Manually perturb a non-Local block to a recognizable pattern,
        # mimicking what an EM ``estimate_initial_conditions=True`` step
        # would do once it has updated the non-Local IC.
        local_state_id = next(
            i for i, obs in enumerate(detector.observation_models) if obs.is_local
        )
        non_local_state_id = next(
            i
            for i, obs in enumerate(detector.observation_models)
            if not obs.is_local and not obs.is_no_spike
        )
        non_local_mask = detector.state_ind_ == non_local_state_id
        n_non_local_bins = int(non_local_mask.sum())
        # Pick something non-uniform but normalized so EM-style updates stay
        # consistent with discrete_initial_conditions.
        new_non_local_block = np.linspace(1.0, 2.0, n_non_local_bins).astype(
            detector.initial_conditions_.dtype
        )
        new_non_local_block /= new_non_local_block.sum()
        new_non_local_block *= float(
            detector.discrete_initial_conditions[non_local_state_id]
        )
        detector.initial_conditions_[non_local_mask] = new_non_local_block
        ic_before = np.array(detector.initial_conditions_, copy=True)

        override = detector.compute_local_initial_conditions(
            _sim_data["time"], _sim_data["position"], _sim_data["time"]
        )
        assert override is not None
        assert override.shape == detector.initial_conditions_.shape

        # Local block: one-hot at the animal's first-frame bin, scaled.
        env = detector.environments[0]
        first_pos = np.atleast_2d(np.asarray(_sim_data["position"])[0])
        if first_pos.ndim == 1:
            first_pos = first_pos[:, np.newaxis]
        animal_bin = int(env.get_bin_ind(first_pos)[0])
        local_block = override[detector.state_ind_ == local_state_id]
        assert int(np.argmax(local_block)) == animal_bin
        assert int((local_block > 0).sum()) == 1
        np.testing.assert_allclose(
            local_block.max(),
            float(detector.discrete_initial_conditions[local_state_id]),
            rtol=1e-6,
        )

        # Non-Local block: bit-equal to the (perturbed) stored IC.
        np.testing.assert_array_equal(override[non_local_mask], new_non_local_block)

        # Stored IC itself is not mutated.
        np.testing.assert_array_equal(detector.initial_conditions_, ic_before)

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
    """Full-bin-length IC on linearized multi-arm tracks.

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
    """``estimate_parameters()`` invokes the IC override.

    The override flows through the same ``_predict()`` as ``predict()``,
    but ``estimate_parameters`` packs ``log_likelihood_args`` independently.
    A future refactor that reorders the tuple would silently disable the
    override without breaking other tests, since uniform-IC and delta-IC
    predictions both produce stochastic posteriors.
    """

    def test_estimate_parameters_runs_with_local_position_std(self, _sim_data):
        detector = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=0.5,
        )
        detector.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )

        with _spy_compute_local_initial_conditions(type(detector)) as spy:
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
        assert spy.call_count >= 1, (
            "estimate_parameters did not invoke compute_local_initial_conditions; "
            "the multi-bin Local IC override is silently skipped on the EM path."
        )


@pytest.mark.unit
class TestClusterlessOverride:
    """Clusterless detector exercises the IC override end-to-end."""

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
        # Guard against the test silently passing if the Local block were
        # entirely zero/NaN (itself a regression mode).
        assert local_t0.sum() > 0, (
            "Clusterless Local mass at t=0 is zero or all-NaN; the override "
            "or the forward pass produced no Local probability."
        )
        assert int(np.argmax(local_t0)) == animal_bin, (
            f"Clusterless causal t=0 peaks at bin {int(np.argmax(local_t0))}, "
            f"expected animal_bin={animal_bin}. Override may not be wired "
            "through the clusterless predict() path."
        )


@pytest.mark.unit
class TestMostLikelySequenceUsesOverride:
    """``most_likely_sequence`` (Viterbi) invokes the IC override.

    Without this, ``predict()`` and ``most_likely_sequence()`` use
    different initial Local distributions when ``local_position_std``
    is set — a silent inconsistency between the smoother and the
    most-likely-state path.
    """

    def test_sorted_spikes_invokes_override(self, _fitted_detector, _sim_data):
        with _spy_compute_local_initial_conditions(type(_fitted_detector)) as spy:
            _fitted_detector.most_likely_sequence(
                position_time=_sim_data["time"],
                position=_sim_data["position"],
                spike_times=_sim_data["spike_times"],
                time=_sim_data["time"],
            )
        assert spy.call_count >= 1, (
            "most_likely_sequence did not invoke compute_local_initial_conditions; "
            "Viterbi is using a different initial Local distribution from predict()."
        )

    # Note: a "Viterbi sequence differs with vs without override" test
    # is too strong under correct n_bins mass balance — Local dominates
    # so cleanly on awake-behavior simulated data that Viterbi picks the
    # same path regardless of whether the t=0 Local IC is delta or
    # uniform. The spy test above (`test_sorted_spikes_invokes_override`)
    # already pins that `most_likely_sequence` calls the override; the
    # smoother-side `test_override_changes_t0_posterior_vs_uniform_ic`
    # exercises the behavioral effect on the posterior.


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
        # bin spacing) leaves most mass near the animal.
        assert local_t0.shape == (n_local_bins,)
        # Guard: an all-zero/NaN Local block is itself a regression mode.
        assert local_t0.sum() > 0, (
            "Local mass at t=0 is zero or all-NaN; cannot test argmax."
        )
        normalized = local_t0 / local_t0.sum()
        assert int(np.argmax(normalized)) == animal_bin, (
            f"Causal posterior at t=0 peaks at bin {int(np.argmax(normalized))}, "
            f"expected animal_bin={animal_bin}. The IC override may not be wired."
        )

    def test_override_changes_t0_posterior_vs_uniform_ic(
        self, _fitted_detector, _sim_data
    ):
        """Override makes a measurable, non-trivial difference at t=0.

        Patches ``compute_local_initial_conditions`` to always return None
        (so ``_predict()`` falls back to the stored uniform IC), then
        compares the causal posterior at t=0 against the unpatched run.
        The two must differ on the Local block.
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
            "compute_local_initial_conditions",
            return_value=None,
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


@pytest.mark.unit
class TestLocalStateOccupancyRegression:
    """Regression: Local state must dominate on awake-behavior simulated data.

    The IC mass-balance compensation in ``_compute_local_position_kernel``
    (and its delta-kernel sibling) cancels the multi-bin Local state's
    uniform ``1/n_bins`` continuous IC factor; without it Non-Local
    dominates by a factor of n_bins, regardless of σ. The previous test
    suite checked posterior validity (sums to 1, finite) but missed the
    occupancy collapse — this test pins it.

    Threshold of 0.5 leaves ample headroom around the legacy ``None``
    baseline of ~0.71 while flagging any reappearance of the
    n_bins-collapse failure mode.
    """

    @pytest.mark.parametrize("sigma", [0.0, 0.5, 5.0])
    def test_p_local_dominates_under_awake_behavior(self, _sim_data, sigma):
        detector = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=sigma,
        )
        detector.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )
        results = detector.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )
        state_probs = np.asarray(results.acausal_state_probabilities)
        mean_p_local = float(state_probs[:, 0].mean())
        assert mean_p_local > 0.5, (
            f"σ={sigma}: mean P(Local) = {mean_p_local:.3f}, expected > 0.5 "
            "on awake-behavior simulated data. If this fires, the multi-bin "
            "Local kernel may have lost its 1/n_bins mass-balance "
            "compensation — Non-Local dominates by a factor of n_bins."
        )
