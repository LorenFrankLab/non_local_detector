"""Tests for local_position_std parameter on detector classes.

Tests cover:
- Parameter acceptance (None default, positive values)
- Validation rejection (zero, negative values)
- Parameter storage on instance
- Backward compatibility (None preserves legacy behavior)
- State index allocation (multi-bin vs single-bin local)
- Initial conditions (uniform over all bins for multi-bin local)
- Continuous transitions (auto-upgrade Discrete, n-bin to 1-bin)
"""

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)


@pytest.mark.unit
class TestLocalPositionStdValidation:
    """Test local_position_std parameter validation."""

    def test_default_is_none(self):
        """Default local_position_std is None (legacy behavior)."""
        detector = NonLocalSortedSpikesDetector()
        assert detector.local_position_std is None

    def test_positive_value_accepted(self):
        """Positive local_position_std values are accepted."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        assert detector.local_position_std == 5.0

    def test_small_positive_value_accepted(self):
        """Small positive local_position_std values are accepted."""
        detector = NonLocalSortedSpikesDetector(local_position_std=0.01)
        assert detector.local_position_std == 0.01

    def test_zero_rejected(self):
        """local_position_std=0 is rejected with ValidationError."""
        with pytest.raises(
            ValidationError, match="local_position_std must be positive"
        ):
            NonLocalSortedSpikesDetector(local_position_std=0.0)

    def test_negative_rejected(self):
        """Negative local_position_std is rejected with ValidationError."""
        with pytest.raises(
            ValidationError, match="local_position_std must be positive"
        ):
            NonLocalSortedSpikesDetector(local_position_std=-1.0)

    def test_clusterless_default_is_none(self):
        """Default local_position_std is None on clusterless detector."""
        detector = NonLocalClusterlessDetector()
        assert detector.local_position_std is None

    def test_clusterless_positive_value_accepted(self):
        """Positive local_position_std accepted on clusterless detector."""
        detector = NonLocalClusterlessDetector(local_position_std=5.0)
        assert detector.local_position_std == 5.0

    def test_clusterless_zero_rejected(self):
        """local_position_std=0 rejected on clusterless detector."""
        with pytest.raises(
            ValidationError, match="local_position_std must be positive"
        ):
            NonLocalClusterlessDetector(local_position_std=0.0)

    def test_clusterless_negative_rejected(self):
        """Negative local_position_std rejected on clusterless detector."""
        with pytest.raises(
            ValidationError, match="local_position_std must be positive"
        ):
            NonLocalClusterlessDetector(local_position_std=-1.0)


@pytest.mark.unit
class TestMultiBinLocalStateIndex:
    """Test initialize_state_index with multi-bin local."""

    def test_legacy_local_gets_one_bin(self, simple_1d_environment):
        """Legacy local_position_std=None gives local state 1 bin."""
        detector = NonLocalSortedSpikesDetector()
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()

        # Default NonLocal model: state 0 = Local (1 bin),
        # state 1 = No-Spike (1 bin), states 2,3 = Non-Local (spatial bins)
        assert detector.bin_sizes_[0] == 1

    def test_multibin_local_gets_spatial_bins(self, simple_1d_environment):
        """Multi-bin local (local_position_std set) gets same bins as non-local."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()

        # Local state should now have the same number of bins as the non-local
        # states (states 2 and 3)
        local_bins = detector.bin_sizes_[0]
        nonlocal_bins = detector.bin_sizes_[2]
        assert local_bins == nonlocal_bins
        assert local_bins > 1

    def test_multibin_local_state_ind_correct(self, simple_1d_environment):
        """Multi-bin local state_ind_ maps all local bins to state 0."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()

        n_local_bins = detector.bin_sizes_[0]
        # First n_local_bins entries in state_ind_ should all be 0 (local state)
        assert np.all(detector.state_ind_[:n_local_bins] == 0)

    def test_multibin_local_has_interior_mask(self, simple_1d_environment):
        """Multi-bin local state gets correct track interior mask."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()

        n_local_bins = detector.bin_sizes_[0]
        local_interior = detector.is_track_interior_state_bins_[:n_local_bins]
        # Should match the environment's track interior
        env = detector.environments[0]
        expected_interior = env.is_track_interior_.ravel()
        np.testing.assert_array_equal(local_interior, expected_interior)


@pytest.mark.unit
class TestMultiBinLocalInitialConditions:
    """Test initialize_initial_conditions with multi-bin local."""

    def test_legacy_local_initial_conditions_unchanged(self):
        """Legacy local_position_std=None gives 1-element initial conditions for local."""
        detector = NonLocalSortedSpikesDetector()
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()
        detector.initialize_initial_conditions()

        # Local state (state 0) should have 1 bin
        assert detector.bin_sizes_[0] == 1

    def test_multibin_local_gets_uniform_initial_conditions(self):
        """Multi-bin local gets uniform initial conditions over all bins."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()
        detector.initialize_initial_conditions()

        n_local_bins = detector.bin_sizes_[0]
        assert n_local_bins > 1

        # Extract the initial conditions for the local state bins
        local_ic = detector.continuous_initial_conditions_[:n_local_bins]
        # Should be uniform (all equal) and sum to 1
        assert np.allclose(local_ic, local_ic[0])
        assert np.isclose(local_ic.sum(), 1.0, atol=1e-10)


def _init_detector_transitions(detector, position):
    """Helper: initialize detector through continuous transitions."""
    detector.initialize_environments(position)
    detector.initialize_state_index()
    detector.initialize_initial_conditions()
    detector.initialize_continuous_state_transition(
        detector.continuous_transition_types, position
    )


@pytest.mark.unit
class TestMultiBinLocalContinuousTransitions:
    """Test continuous state transitions with multi-bin local."""

    def test_multibin_local_to_nospike_transition_shape(self):
        """Local (multi-bin) -> No-Spike (1-bin) transition has correct shape."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        _init_detector_transitions(detector, position)

        # State 0 = Local (multi-bin), State 1 = No-Spike (1-bin)
        n_local_bins = detector.bin_sizes_[0]
        assert n_local_bins > 1

        # Extract the transition block from local -> no-spike
        local_mask = detector.state_ind_ == 0
        nospike_mask = detector.state_ind_ == 1
        block = detector.continuous_state_transitions_[np.ix_(local_mask, nospike_mask)]

        # Shape should be (n_local_bins, 1), all values 1.0
        assert block.shape == (n_local_bins, 1)
        np.testing.assert_allclose(block, 1.0)

    def test_discrete_auto_upgraded_to_uniform_for_multibin_local(self):
        """Discrete() is auto-upgraded to Uniform() for multi-bin local transitions."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        _init_detector_transitions(detector, position)

        # The Local->Local transition should now be a proper spatial transition
        # (not a 1x1 identity). Check it's stochastic.
        n_local_bins = detector.bin_sizes_[0]
        local_mask = detector.state_ind_ == 0
        block = detector.continuous_state_transitions_[np.ix_(local_mask, local_mask)]
        assert block.shape == (n_local_bins, n_local_bins)
        # Each row should sum to approximately 1 (stochastic)
        row_sums = block.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_nonlocal_discrete_transitions_unaffected(self):
        """Non-local Discrete() transitions are not affected by local_position_std."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        _init_detector_transitions(detector, position)

        # No-Spike state (state 1) should still have 1 bin
        assert detector.bin_sizes_[1] == 1

    def test_legacy_discrete_transition_unchanged(self):
        """Legacy local_position_std=None keeps Discrete() as 1x1 identity."""
        detector = NonLocalSortedSpikesDetector()
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        _init_detector_transitions(detector, position)

        # Local state should have 1 bin, transition is 1x1
        assert detector.bin_sizes_[0] == 1
        local_mask = detector.state_ind_ == 0
        block = detector.continuous_state_transitions_[np.ix_(local_mask, local_mask)]
        assert block.shape == (1, 1)
        assert block[0, 0] == 1.0

    def test_full_transition_matrix_stochastic(self):
        """Full continuous transition matrix is row-stochastic for multi-bin local."""
        detector = NonLocalSortedSpikesDetector(local_position_std=5.0)
        position = np.linspace(0, 10, 11)[:, np.newaxis]
        _init_detector_transitions(detector, position)

        T = detector.continuous_state_transitions_
        # Each row should sum to number of discrete states that row can
        # transition to (but within each block, rows sum to 1)
        # Check individual blocks are stochastic
        for from_state in range(detector.n_discrete_states_):
            for to_state in range(detector.n_discrete_states_):
                from_mask = detector.state_ind_ == from_state
                to_mask = detector.state_ind_ == to_state
                block = T[np.ix_(from_mask, to_mask)]
                if block.sum() > 0:  # Only check non-zero blocks
                    row_sums = block.sum(axis=1)
                    np.testing.assert_allclose(
                        row_sums,
                        1.0,
                        atol=1e-10,
                        err_msg=f"Block ({from_state},{to_state}) not stochastic",
                    )


@pytest.mark.unit
class TestComputeLocalPositionKernel:
    """Test _compute_local_position_kernel method."""

    def _make_fitted_detector(self, local_position_std=5.0):
        """Create a detector with environments initialized for kernel testing."""
        detector = NonLocalSortedSpikesDetector(local_position_std=local_position_std)
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        detector.initialize_environments(position)
        detector.initialize_state_index()
        return detector, position

    def test_kernel_is_valid_log_probability(self):
        """Kernel exp sums to 1 per time step (valid log-probability)."""
        import jax.numpy as jnp

        detector, position = self._make_fitted_detector(local_position_std=5.0)
        env = detector.environments[0]

        n_time = 10
        time = np.linspace(0, 1, n_time)
        position_time = np.linspace(0, 1, 50)
        animal_position = np.linspace(10, 90, 50)[:, np.newaxis]

        log_kernel = detector._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        # exp(log_kernel) should sum to n_interior_bins per time step
        # (scaled to compensate for 1/n_bins uniform continuous IC).
        # JAX computes in float32, so use rtol=1e-5.
        probs = np.exp(np.asarray(log_kernel))
        row_sums = probs.sum(axis=1)
        n_interior = int(env.is_track_interior_.sum())
        np.testing.assert_allclose(row_sums, n_interior, rtol=1e-5)

    def test_kernel_peak_at_nearest_bin(self):
        """Kernel peak is at the bin nearest to animal position."""
        import jax.numpy as jnp

        detector, position = self._make_fitted_detector(local_position_std=5.0)
        env = detector.environments[0]

        # Animal at position 50
        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[50.0], [50.0]])

        log_kernel = detector._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        # Find the bin center closest to 50
        is_interior = env.is_track_interior_.ravel()
        bin_centers = env.place_bin_centers_[is_interior].ravel()
        nearest_bin_idx = np.argmin(np.abs(bin_centers - 50.0))

        # The kernel peak should be at or near the nearest bin
        peak_idx = int(jnp.argmax(log_kernel[0]))
        assert abs(peak_idx - nearest_bin_idx) <= 1

    def test_kernel_narrows_with_smaller_std(self):
        """Kernel concentrates more with smaller local_position_std."""
        import jax.numpy as jnp

        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[50.0], [50.0]])

        # Wide kernel
        det_wide, _ = self._make_fitted_detector(local_position_std=20.0)
        env = det_wide.environments[0]
        log_kernel_wide = det_wide._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        # Narrow kernel
        det_narrow, _ = self._make_fitted_detector(local_position_std=2.0)
        env_n = det_narrow.environments[0]
        log_kernel_narrow = det_narrow._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env_n,
        )

        # Narrow kernel should have higher peak (more concentrated)
        assert float(jnp.max(log_kernel_narrow)) > float(jnp.max(log_kernel_wide))

    def test_kernel_output_shape(self):
        """Output shape is (n_time, n_interior_bins)."""
        import jax.numpy as jnp

        detector, position = self._make_fitted_detector(local_position_std=5.0)
        env = detector.environments[0]

        n_time = 7
        time = np.linspace(0, 1, n_time)
        position_time = np.linspace(0, 1, 50)
        animal_position = np.linspace(10, 90, 50)[:, np.newaxis]

        log_kernel = detector._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        n_interior = int(env.is_track_interior_.sum())
        assert log_kernel.shape == (n_time, n_interior)

    def test_nan_positions_produce_uniform_kernel(self):
        """NaN positions produce uniform (flat) kernel, not NaN or crash."""
        import jax.numpy as jnp

        detector, position = self._make_fitted_detector(local_position_std=5.0)
        env = detector.environments[0]

        # All NaN positions
        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[np.nan], [np.nan]])

        log_kernel = detector._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        # Should not contain NaN
        log_kernel_np = np.asarray(log_kernel)
        assert np.all(np.isfinite(log_kernel_np))
        # Should be uniform (all equal values, all zero since exp=1 per bin)
        np.testing.assert_allclose(log_kernel_np[0], log_kernel_np[0, 0], atol=1e-6)
        # exp should sum to n_interior_bins (scaled normalization)
        n_interior = int(env.is_track_interior_.sum())
        np.testing.assert_allclose(
            np.exp(log_kernel_np[0]).sum(), n_interior, rtol=1e-5
        )

    def test_kernel_with_track_graph(self):
        """Kernel uses track graph distances when track graph is available."""
        import jax.numpy as jnp
        import networkx as nx

        from non_local_detector.environment import Environment

        # Create a simple 1D linear track with track graph
        track_graph = nx.Graph()
        track_graph.add_node(0, pos=(0.0, 0.0))
        track_graph.add_node(1, pos=(100.0, 0.0))
        track_graph.add_edge(0, 1, distance=100.0, edge_id=0)

        env = Environment(
            environment_name="",
            place_bin_size=5.0,
            track_graph=track_graph,
            edge_order=[(0, 1)],
            edge_spacing=0.0,
        )
        position_1d = np.linspace(0, 100, 50)
        env = env.fit_place_grid(position_1d, infer_track_interior=True)

        detector = NonLocalSortedSpikesDetector(local_position_std=10.0)
        detector.environments = (env,)
        detector.initialize_state_index()

        # Animal at position 50
        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[50.0], [50.0]])

        log_kernel = detector._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        log_kernel_np = np.asarray(log_kernel)
        # Should be finite (no NaN from track graph path)
        assert np.all(np.isfinite(log_kernel_np))
        # Should sum to n_interior_bins in probability space
        is_interior = env.is_track_interior_.ravel()
        n_interior = int(is_interior.sum())
        np.testing.assert_allclose(
            np.exp(log_kernel_np[0]).sum(), n_interior, rtol=1e-5
        )
        # Peak should be near position 50
        bin_centers = env.place_bin_centers_[is_interior].ravel()
        peak_idx = int(np.argmax(log_kernel_np[0]))
        assert abs(bin_centers[peak_idx] - 50.0) < 10.0

    def test_kernel_off_track_produces_uniform(self):
        """Off-track animal position (gap bin) produces uniform kernel."""
        import jax.numpy as jnp
        import networkx as nx

        from non_local_detector.environment import Environment

        # Create a two-segment track with a gap between them.
        # edge_spacing > 0 creates gap bins with node_id == -1.
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
        position_1d = np.concatenate([np.linspace(0, 50, 25), np.linspace(60, 110, 25)])
        env = env.fit_place_grid(position_1d, infer_track_interior=True)

        detector = NonLocalSortedSpikesDetector(local_position_std=10.0)
        detector.environments = (env,)
        detector.initialize_state_index()

        # Place animal in the gap between the two segments
        time = np.array([0.5])
        position_time = np.array([0.0, 1.0])
        animal_position = np.array([[55.0], [55.0]])

        log_kernel = detector._compute_local_position_kernel(
            jnp.array(time),
            jnp.array(position_time),
            jnp.array(animal_position),
            env,
        )

        log_kernel_np = np.asarray(log_kernel)
        # Should be finite (no NaN — uniform fallback applied)
        assert np.all(np.isfinite(log_kernel_np)), (
            "Off-track position produced non-finite kernel"
        )
        # Should be uniform (all equal, all 0.0 since exp=1 per bin)
        np.testing.assert_allclose(log_kernel_np[0], log_kernel_np[0, 0], atol=1e-6)
        # Should sum to n_interior_bins (scaled normalization)
        is_interior = env.is_track_interior_.ravel()
        n_interior = int(is_interior.sum())
        np.testing.assert_allclose(
            np.exp(log_kernel_np[0]).sum(), n_interior, rtol=1e-5
        )
