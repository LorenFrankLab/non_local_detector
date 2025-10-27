"""Tests for initialization methods in models/base.py.

These tests cover the initialization methods that set up model components:
- _initialize_environments
- _initialize_observation_models
- _initialize_state_names
- initialize_environments (environment fitting)
- initialize_state_index
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.models import SortedSpikesDecoder
from non_local_detector.models.base import ObservationModel


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
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
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
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=[env1, env2],
            state_names=["state 0", "state 1"],
        )

        # Assert
        assert len(decoder.environments) == 2
        # Environments are converted to tuple or kept as list depending on implementation
        assert isinstance(decoder.environments, (tuple, list))


@pytest.mark.unit
class TestInitializeObservationModels:
    """Test _initialize_observation_models method."""

    def test_initialize_observation_models_with_none(self, simple_1d_environment):
        """Test that None creates default observation models.

        When observation_models=None, SortedSpikesDecoder creates one observation
        model that gets replicated based on the number of continuous transition types.
        """
        # Arrange & Act
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=None,
            state_names=["state 0", "state 1"],
        )

        # Assert - Default observation models are created
        assert len(decoder.observation_models) >= 1
        assert all(
            isinstance(om, ObservationModel) for om in decoder.observation_models
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
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
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
        assert decoder.environments[0].place_bin_centers_ is not None
        assert decoder.environments[0].place_bin_centers_.shape[0] > 0

    def test_initialize_environments_with_multiple_environments(
        self, simple_1d_environment
    ):
        """Test initialize_environments with multiple environments."""
        # Arrange
        env1 = Environment(environment_name="env1")
        env2 = Environment(environment_name="env2")
        obs1 = ObservationModel(environment_name="env1", is_local=False)
        obs2 = ObservationModel(environment_name="env2", is_local=False)
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=(env1, env2),
            observation_models=(obs1, obs2),
            state_names=["state 0", "state 1"],
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        # Label first half as env1, second half as env2
        environment_labels = np.array(["env1"] * 25 + ["env2"] * 25)

        # Act
        decoder.initialize_environments(position, environment_labels=environment_labels)

        # Assert
        assert decoder.environments[0].place_bin_centers_ is not None
        assert decoder.environments[1].place_bin_centers_ is not None

    def test_initialize_environments_linearizes_2d_position_with_track_graph(self):
        """Test that initialize_environments linearizes 2D positions when track_graph is provided.

        This tests line 490 in base.py where get_linearized_position is called
        to convert 2D positions to 1D when a track_graph is present.
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

        # Act - This should trigger the linearization path (line 490)
        decoder.initialize_environments(position_2d)

        # Assert
        assert decoder.environments[0].place_bin_centers_ is not None
        # Should be linearized to 1D
        assert decoder.environments[0].place_bin_centers_.shape[1] == 1
        # Centers should be monotonically increasing for a simple line
        centers = decoder.environments[0].place_bin_centers_.squeeze()
        assert np.all(np.diff(centers) > 0)


@pytest.mark.unit
class TestInitializeStateIndex:
    """Test initialize_state_index method."""

    def test_initialize_state_index_sets_n_discrete_states(self, simple_1d_environment):
        """Test that initialize_state_index sets n_discrete_states_."""
        # Arrange
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.3, 0.2]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=tuple(
                ObservationModel(
                    environment_name=simple_1d_environment.environment_name,
                    is_local=False,
                )
                for _ in range(3)
            ),
            state_names=["state 0", "state 1", "state 2"],
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        decoder.initialize_environments(position)

        # Act
        decoder.initialize_state_index()

        # Assert
        assert decoder.n_discrete_states_ == 3

    def test_initialize_state_index_sets_state_ind(self, simple_1d_environment):
        """Test that initialize_state_index creates state_ind_ array."""
        # Arrange
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=tuple(
                ObservationModel(
                    environment_name=simple_1d_environment.environment_name,
                    is_local=False,
                )
                for _ in range(2)
            ),
            state_names=["state 0", "state 1"],
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        decoder.initialize_environments(position)

        # Act
        decoder.initialize_state_index()

        # Assert
        assert hasattr(decoder, "state_ind_")
        assert decoder.state_ind_.ndim == 1
        assert len(decoder.state_ind_) == decoder.n_state_bins_

    def test_initialize_state_index_sets_bin_sizes(self, simple_1d_environment):
        """Test that initialize_state_index creates bin_sizes_ array."""
        # Arrange
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=tuple(
                ObservationModel(
                    environment_name=simple_1d_environment.environment_name,
                    is_local=False,
                )
                for _ in range(2)
            ),
            state_names=["state 0", "state 1"],
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        decoder.initialize_environments(position)

        # Act
        decoder.initialize_state_index()

        # Assert
        assert hasattr(decoder, "bin_sizes_")
        assert len(decoder.bin_sizes_) == decoder.n_discrete_states_
        assert all(bs > 0 for bs in decoder.bin_sizes_)

    def test_initialize_state_index_n_state_bins_equals_sum_of_bin_sizes(
        self, simple_1d_environment
    ):
        """Test that n_state_bins_ equals sum of bin_sizes_."""
        # Arrange
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.3, 0.2]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=tuple(
                ObservationModel(
                    environment_name=simple_1d_environment.environment_name,
                    is_local=False,
                )
                for _ in range(3)
            ),
            state_names=["state 0", "state 1", "state 2"],
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        decoder.initialize_environments(position)

        # Act
        decoder.initialize_state_index()

        # Assert
        assert decoder.n_state_bins_ == sum(decoder.bin_sizes_)

    def test_initialize_state_index_sets_is_track_interior(self, simple_1d_environment):
        """Test that initialize_state_index sets is_track_interior_state_bins_."""
        # Arrange
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([1.0]),
            continuous_initial_conditions_types=["uniform_on_track"],
            continuous_transition_types=["random_walk"],
            discrete_transition_stickiness=0.0,
            environments=simple_1d_environment,
            observation_models=ObservationModel(
                environment_name=simple_1d_environment.environment_name, is_local=False
            ),
            state_names=["state 0"],
        )
        position = np.linspace(0, 100, 50)[:, np.newaxis]
        decoder.initialize_environments(position)

        # Act
        decoder.initialize_state_index()

        # Assert
        assert hasattr(decoder, "is_track_interior_state_bins_")
        assert decoder.is_track_interior_state_bins_.dtype == bool
        assert len(decoder.is_track_interior_state_bins_) == decoder.n_state_bins_
