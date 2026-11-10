"""Tests for initialization methods in models/base.py.

These tests cover the initialization methods that set up model components:
- _initialize_environments
- _initialize_observation_models
- _initialize_state_names
- initialize_environments (environment fitting)
- state_layout (the fitted state bins)
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.models import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    MultiEnvironmentClusterlessClassifier,
    NonLocalClusterlessDetector,
    SortedSpikesDecoder,
)
from non_local_detector.models._state_space import state_layout
from non_local_detector.models.base import (
    ObservationModel,
)


@pytest.mark.unit
class TestInitializeEnvironments:
    """Test _initialize_environments method."""

    def test_initialize_environments_with_none(self):
        """Test that None creates default environment."""
        # Arrange & Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([1.0]),
            continuous_initial_conditions_types=["uniform_on_track"],
            continuous_transition_types=["random_walk"],
            discrete_transition_stickiness=0.0,
            environments=None,
        )

        # Assert
        assert len(decoder.environments) == 1
        assert isinstance(decoder.environments[0], Environment)

    def test_initialize_environments_with_single_environment(
        self, simple_1d_environment
    ):
        """Test that single environment is converted to tuple."""
        # Arrange & Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([1.0]),
            continuous_initial_conditions_types=["uniform_on_track"],
            continuous_transition_types=["random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
        )

        # Assert
        assert len(decoder.environments) == 1
        assert isinstance(decoder.environments[0], Environment)
        assert decoder.environments[0] is simple_1d_environment

    def test_initialize_environments_with_tuple(self, simple_1d_environment):
        """Test that tuple of environments is preserved."""
        # Arrange
        env1 = simple_1d_environment
        env2 = Environment(environment_name="env2")

        # Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            discrete_transition_type=DiscreteStationaryDiagonal(np.full(2, 0.9)),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            observation_models=ObservationModel(),
            environments=(env1, env2),
            state_names=["state 0", "state 1"],
        )

        # Assert
        assert len(decoder.environments) == 2
        assert decoder.environments[0] is env1
        assert decoder.environments[1] is env2

    def test_initialize_environments_with_list(self, simple_1d_environment):
        """Test that list of environments is stored."""
        # Arrange
        env1 = simple_1d_environment
        env2 = Environment(environment_name="env2")

        # Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            discrete_transition_type=DiscreteStationaryDiagonal(np.full(2, 0.9)),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            observation_models=ObservationModel(),
            environments=[env1, env2],
            state_names=["state 0", "state 1"],
        )

        # Assert
        assert len(decoder.environments) == 2
        assert decoder.environments[0] is env1
        assert decoder.environments[1] is env2


@pytest.mark.unit
class TestInitializeObservationModels:
    """Test _initialize_observation_models method."""

    def test_initialize_observation_models_with_none(self, simple_1d_environment):
        """Test that None creates one default observation model per state.

        When observation_models=None, one observation model on the first
        environment is replicated for each continuous transition type. The
        public model classes substitute their own defaults for None before
        reaching this method, so it is called directly.
        """
        # Arrange
        decoder = SortedSpikesDecoder(environments=simple_1d_environment)

        # Act
        observation_models = decoder._initialize_observation_models(
            None, ["random_walk", "random_walk"], (simple_1d_environment,)
        )

        # Assert - One default observation model per continuous transition type
        assert len(observation_models) == 2
        assert all(
            isinstance(om, ObservationModel)
            and om.environment_name == simple_1d_environment.environment_name
            for om in observation_models
        )

    def test_initialize_observation_models_with_single_model(
        self, simple_1d_environment
    ):
        """Test that single observation model is replicated for all states."""
        # Arrange
        obs_model = ObservationModel(
            environment_name=simple_1d_environment.environment_name, is_local=False
        )

        # Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.33, 0.33, 0.34]),
            discrete_transition_type=DiscreteStationaryDiagonal(np.full(3, 0.9)),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=obs_model,
            state_names=["state 0", "state 1", "state 2"],
        )

        # Assert
        assert len(decoder.observation_models) == 3
        # All should be the same instance
        assert all(om is obs_model for om in decoder.observation_models)

    def test_initialize_observation_models_with_tuple(self, simple_1d_environment):
        """Test that tuple of observation models is preserved."""
        # Arrange
        obs1 = ObservationModel(
            environment_name=simple_1d_environment.environment_name, is_local=False
        )
        obs2 = ObservationModel(
            environment_name=simple_1d_environment.environment_name, is_local=True
        )

        # Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            discrete_transition_type=DiscreteStationaryDiagonal(np.full(2, 0.9)),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "identity"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=(obs1, obs2),
            state_names=["state 0", "state 1"],
        )

        # Assert
        assert len(decoder.observation_models) == 2
        assert decoder.observation_models[0] is obs1
        assert decoder.observation_models[1] is obs2

    @pytest.mark.parametrize("frozen_rows", ["no_spike", None])
    @pytest.mark.parametrize("observation_models", [["a"] * 4, [None] * 4])
    def test_non_observation_model_entries_raise(self, observation_models, frozen_rows):
        """Entries that are not ObservationModel instances are rejected at
        construction, whatever the frozen-row setting."""
        with pytest.raises(ValidationError, match="entries must be ObservationModel"):
            NonLocalClusterlessDetector(
                observation_models=observation_models,
                frozen_discrete_transition_rows=frozen_rows,
            )

    @pytest.mark.parametrize("observation_models", ["xxxx", 5])
    def test_non_sequence_observation_models_raise(self, observation_models):
        """A string or a number is not a sequence of observation models."""
        with pytest.raises(ValidationError, match="list, tuple or array"):
            NonLocalClusterlessDetector(observation_models=observation_models)

    def test_array_of_observation_models_accepted(self):
        """An object array of ObservationModel instances is a sequence too."""
        observation_models = np.array(
            [
                ObservationModel(is_local=True),
                ObservationModel(is_no_spike=True),
                ObservationModel(),
                ObservationModel(),
            ],
            dtype=object,
        )
        detector = NonLocalClusterlessDetector(observation_models=observation_models)
        np.testing.assert_array_equal(
            detector._frozen_discrete_transition_rows_mask_,
            [False, True, False, False],
        )

    @pytest.mark.parametrize("n_models", [3, 5])
    def test_one_observation_model_per_state_required(self, n_models):
        """The non-local detector has four states, so three or five observation
        models are rejected rather than misaligned with the states."""
        with pytest.raises(ValidationError, match="one observation model per"):
            NonLocalClusterlessDetector(
                observation_models=[ObservationModel()] * n_models
            )


@pytest.mark.unit
class TestInitializeStateNames:
    """Test _initialize_state_names method."""

    def test_initialize_state_names_with_custom_names(self):
        """Test that custom state names are preserved."""
        # Arrange
        custom_names = ["Local", "Non-Local", "No Spike"]

        # Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.3, 0.2]),
            discrete_transition_type=DiscreteStationaryDiagonal(np.full(3, 0.9)),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            observation_models=ObservationModel(),
            state_names=custom_names,
        )

        # Assert
        assert decoder.state_names == custom_names

    def test_initialize_state_names_length_mismatch_raises_error(self):
        """Test that mismatched state name length raises ValidationError."""
        # Arrange: 3 states but only 2 names
        with pytest.raises(
            ValidationError,
            match="Number of state names must match number of states",
        ):
            # Act
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.3, 0.2]),  # 3 states
                discrete_transition_type=DiscreteStationaryDiagonal(np.full(3, 0.9)),
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=[
                    "random_walk",
                    "random_walk",
                    "random_walk",
                ],
                discrete_transition_stickiness=0.0,
                observation_models=ObservationModel(),
                state_names=["Local", "Non-Local"],  # Only 2 names - WRONG!
            )

    def test_initialize_state_names_empty_list_for_single_state_raises_error(self):
        """Test that empty state names list for non-empty states raises error."""
        # Arrange & Act
        with pytest.raises(ValidationError, match="Number of state names must match"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([1.0]),  # 1 state
                continuous_initial_conditions_types=["uniform_on_track"],
                continuous_transition_types=["random_walk"],
                discrete_transition_stickiness=0.0,
                state_names=[],  # Empty - WRONG!
            )


@pytest.mark.unit
class TestInitializeEnvironmentsFitting:
    """Test initialize_environments method (environment fitting)."""

    def test_a_fitted_detector_is_refused(self):
        """Replacing ``environments_`` would leave the fitted state space, the
        encoding model and every decode on a grid they were not built for."""
        position = np.linspace(0, 100, 200)[:, np.newaxis]
        time = np.arange(200) / 500.0
        detector = SortedSpikesDecoder(sampling_frequency=500.0).fit(
            time, position, [np.array([0.1, 0.2])]
        )
        fitted_environments = detector.environments_

        with pytest.raises(ValidationError, match="fitted"):
            detector.initialize_environments(position[:100])

        assert detector.environments_ is fitted_environments

    def test_an_unfitted_detector_can_be_initialized_again(self):
        environment = Environment(environment_name="", place_bin_size=5.0)
        decoder = SortedSpikesDecoder(environments=environment)
        decoder.initialize_environments(np.linspace(0, 100, 50)[:, np.newaxis])
        decoder.initialize_environments(np.linspace(0, 50, 50)[:, np.newaxis])

        assert decoder.environments_[0].place_bin_edges_.max() == pytest.approx(50.0)

    def test_initialize_environments_fits_position_data(self, simple_1d_environment):
        """Test that initialize_environments fits the environment on position data."""
        # Arrange
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([1.0]),
            continuous_initial_conditions_types=["uniform_on_track"],
            continuous_transition_types=["random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]  # 1D position

        # Act
        decoder.initialize_environments(position)

        # Assert
        assert decoder.environments_[0].place_bin_centers_ is not None
        assert decoder.environments_[0].place_bin_centers_.shape[0] > 0

    def test_initialize_environments_linearizes_2d_position_with_track_graph(
        self, monkeypatch
    ):
        """initialize_environments linearizes 2D positions on a track graph.

        The track-graph place grid is built from the graph alone, so the
        test records the position handed to ``fit_place_grid`` and checks
        that it is the 1D linear position, not the raw 2D input.
        """
        # Arrange: Create a simple line graph
        g = nx.Graph()
        g.add_node(0, pos=(0.0, 0.0))
        g.add_node(1, pos=(10.0, 0.0))
        g.add_edge(0, 1, distance=10.0)
        # Assign stable edge ids
        for eid, e in enumerate(g.edges):
            g.edges[e]["edge_id"] = eid
        edge_order = [(0, 1)]
        edge_spacing = 0.0

        env = Environment(
            environment_name="line_graph",
            place_bin_size=1.0,
            track_graph=g,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
        )
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([1.0]),
            continuous_initial_conditions_types=["uniform_on_track"],
            continuous_transition_types=["random_walk"],
            discrete_transition_stickiness=0.0,
            environments=env,
        )

        # 2D positions along the x-axis from 0 to 10
        position_2d = np.stack(
            [np.linspace(0, 10, 50), np.zeros(50)], axis=1
        )  # (50, 2)

        received = []
        fit_place_grid = Environment.fit_place_grid

        def spy(self, position, **kwargs):
            received.append(np.asarray(position))
            return fit_place_grid(self, position, **kwargs)

        # The decoder fits a copy of ``env``, so spy on the class.
        monkeypatch.setattr(Environment, "fit_place_grid", spy)

        # Act
        decoder.initialize_environments(position_2d)

        # Assert: the grid was fit on the linear position, which for a
        # straight edge from (0, 0) to (10, 0) equals the x coordinate
        assert len(received) == 1
        assert received[0].squeeze().ndim == 1
        np.testing.assert_allclose(received[0].squeeze(), position_2d[:, 0])
        assert decoder.environments_[0].place_bin_centers_.shape[1] == 1


def _n_state_decoder(environment, discrete_initial_conditions):
    """Build a non-local SortedSpikesDecoder with fitted environments.

    The number of states is ``len(discrete_initial_conditions)``; every
    state uses a uniform-on-track initial condition, a random-walk
    transition, and a non-local observation model in ``environment``.
    """
    n_states = len(discrete_initial_conditions)
    decoder = SortedSpikesDecoder(
        discrete_initial_conditions=discrete_initial_conditions,
        discrete_transition_type=DiscreteStationaryDiagonal(
            np.full(len(discrete_initial_conditions), 0.9)
        ),
        continuous_initial_conditions_types=["uniform_on_track"] * n_states,
        continuous_transition_types=["random_walk"] * n_states,
        discrete_transition_stickiness=0.0,
        environments=environment,
        observation_models=tuple(
            ObservationModel(
                environment_name=environment.environment_name, is_local=False
            )
            for _ in range(n_states)
        ),
        state_names=[f"state {i}" for i in range(n_states)],
    )
    position = np.linspace(0, 100, 50)[:, np.newaxis]
    decoder.initialize_environments(position)
    return decoder


@pytest.mark.unit
class TestStateLayout:
    """Test state_layout on a decoder's fitted environments."""

    @staticmethod
    def _layout(decoder):
        return state_layout(
            decoder.observation_models,
            decoder.environments_,
            decoder.local_position_std,
        )

    def test_one_bin_count_per_discrete_state(self, simple_1d_environment):
        """bin_sizes has one positive count per discrete state."""
        decoder = _n_state_decoder(simple_1d_environment, np.array([0.5, 0.3, 0.2]))

        _, bin_sizes, _ = self._layout(decoder)

        assert len(bin_sizes) == 3
        assert all(bs > 0 for bs in bin_sizes)

    def test_state_ind_counts_match_bin_sizes(self, simple_1d_environment):
        """state_ind labels each bin with its state, bin_sizes[i] bins of state i."""
        decoder = _n_state_decoder(simple_1d_environment, np.array([0.5, 0.3, 0.2]))

        state_ind, bin_sizes, _ = self._layout(decoder)

        assert state_ind.ndim == 1
        assert len(state_ind) == sum(bin_sizes)
        np.testing.assert_array_equal(np.bincount(state_ind), bin_sizes)

    def test_is_track_interior_has_one_entry_per_bin(self, simple_1d_environment):
        """is_track_interior is a boolean flag per state bin."""
        decoder = _n_state_decoder(simple_1d_environment, np.array([1.0]))

        state_ind, _, is_track_interior = self._layout(decoder)

        assert is_track_interior.dtype == bool
        assert len(is_track_interior) == len(state_ind)


@pytest.mark.unit
class TestAlgorithmParamsResolution:
    """Algorithm-params dicts must be sklearn-clone-safe and never share state.

    The constructor stores the user's argument as-is (including ``None``)
    to preserve sklearn's clone-roundtrip contract. The
    ``_resolve_*_algorithm_params`` helpers return a fresh dict copy at
    each call so default-derived params and use-site dicts are never
    aliased across instances.
    """

    def test_default_constructor_stores_none(self):
        """Defaults stay as ``None`` after construction (sklearn convention).

        sklearn's ``clone()`` round-trip requires ``self.param == param``
        after ``__init__``; the dict-copy/default-resolution happens
        lazily.
        """
        d_cls = ClusterlessDecoder()
        d_sorted = SortedSpikesDecoder()
        assert d_cls.clusterless_algorithm_params is None
        assert d_sorted.sorted_spikes_algorithm_params is None

    def test_resolver_returns_default_copy_for_none(self):
        """The resolver returns a fresh dict equal to the module default.

        Mutating the returned dict must not change the module-level
        default or later resolver calls.
        """
        original = dict(_CLUSTERLESS_ALGORITHMS["clusterless_kde"].default_params)
        d = ClusterlessDecoder()
        resolved = d._resolve_clusterless_algorithm_params()
        assert resolved == _CLUSTERLESS_ALGORITHMS["clusterless_kde"].default_params
        # Fresh copy, not a reference to the module-level constant
        assert resolved is not _CLUSTERLESS_ALGORITHMS["clusterless_kde"].default_params
        resolved["block_size"] = 5
        assert _CLUSTERLESS_ALGORITHMS["clusterless_kde"].default_params == original
        assert d._resolve_clusterless_algorithm_params() == original

    def test_resolver_returns_distinct_dicts_across_instances(self):
        """Two ClusterlessDecoder instances' resolvers return distinct dicts."""
        d1 = ClusterlessDecoder()
        d2 = ClusterlessDecoder()
        r1 = d1._resolve_clusterless_algorithm_params()
        r2 = d2._resolve_clusterless_algorithm_params()
        assert r1 is not r2
        assert r1 == r2 == _CLUSTERLESS_ALGORITHMS["clusterless_kde"].default_params

    def test_resolver_copies_user_dict(self):
        """A user-supplied dict is copied by the resolver, not aliased."""
        params = {"position_std": 5.0, "waveform_std": 24.0, "block_size": 10_000}
        d = ClusterlessDecoder(clusterless_algorithm_params=params)
        # Constructor stores the user's dict as-is (sklearn convention)
        assert d.clusterless_algorithm_params is params
        # But the resolver returns a fresh copy
        resolved = d._resolve_clusterless_algorithm_params()
        assert resolved is not params
        assert resolved == params

    def test_sorted_spikes_resolver_returns_default_copy_for_none(self):
        """Same resolver contract for sorted-spikes params."""
        d = SortedSpikesDecoder()
        resolved = d._resolve_sorted_spikes_algorithm_params()
        assert resolved == _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"].default_params
        assert (
            resolved
            is not _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"].default_params
        )

    @pytest.mark.parametrize(
        "model_cls",
        [
            ContFragClusterlessClassifier,
            MultiEnvironmentClusterlessClassifier,
            NonLocalClusterlessDetector,
        ],
        ids=lambda cls: cls.__name__,
    )
    def test_clusterless_subclass_default_stores_none(self, model_cls):
        """Clusterless subclasses inherit the same contract."""
        d = model_cls()
        assert d.clusterless_algorithm_params is None
        resolved = d._resolve_clusterless_algorithm_params()
        assert resolved == _CLUSTERLESS_ALGORITHMS["clusterless_kde"].default_params

    def test_sklearn_clone_roundtrips(self):
        """sklearn.clone() must round-trip without raising.

        This is the sklearn-canonical contract: ``__init__`` must store
        params as-is (no transformation), so clone's constructor-output
        round-trip check succeeds.
        """
        from sklearn.base import clone

        d = ClusterlessDecoder()
        cloned = clone(d)
        # The cloned instance is a fresh detector with the same
        # constructor-level state.
        assert cloned.clusterless_algorithm_params is None
        # Same for an explicit-dict caller
        params = {"position_std": 7.0, "waveform_std": 24.0, "block_size": 10_000}
        d2 = ClusterlessDecoder(clusterless_algorithm_params=params)
        cloned2 = clone(d2)
        assert cloned2.clusterless_algorithm_params == params
