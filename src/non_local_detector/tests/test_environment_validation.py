"""Tests for Environment validation."""

import networkx as nx
import numpy as np
import pytest

from non_local_detector import Environment
from non_local_detector.exceptions import ConfigurationError, ValidationError


class TestEnvironmentBasicValidation:
    """Test basic parameter validation for Environment."""

    def test_valid_environment_minimal(self):
        """Test that minimal valid environment works."""
        env = Environment()
        assert env.place_bin_size == 2.0
        assert env.bin_count_threshold == 0

    def test_valid_environment_with_bin_size(self):
        """Test environment with custom bin size."""
        env = Environment(place_bin_size=5.0)
        assert env.place_bin_size == 5.0

    def test_valid_environment_with_tuple_bin_size(self):
        """Test environment with multi-dimensional bin size."""
        env = Environment(place_bin_size=(5.0, 10.0))
        assert env.place_bin_size == (5.0, 10.0)

    def test_negative_place_bin_size_raises(self):
        """Test that negative place_bin_size raises."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(place_bin_size=-1.0)
        assert "place_bin_size" in str(exc_info.value)
        assert "positive" in str(exc_info.value).lower()

    def test_zero_place_bin_size_raises(self):
        """Test that zero place_bin_size raises."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(place_bin_size=0.0)
        assert "place_bin_size" in str(exc_info.value)

    def test_negative_tuple_place_bin_size_raises(self):
        """Test that negative values in tuple place_bin_size raise."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(place_bin_size=(5.0, -1.0))
        assert "place_bin_size[1]" in str(exc_info.value)

    def test_negative_bin_count_threshold_raises(self):
        """Test that negative bin_count_threshold raises."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(bin_count_threshold=-1)
        assert "bin_count_threshold" in str(exc_info.value)


class TestTrackGraphValidation:
    """Test validation of track_graph parameter."""

    def test_track_graph_requires_networkx_graph(self):
        """Test that track_graph must be a NetworkX Graph."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph={"node": "value"})
        assert "networkx" in str(exc_info.value).lower()

    def test_track_graph_requires_nodes(self):
        """Test that track_graph must have nodes."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=nx.Graph())
        assert "nodes" in str(exc_info.value).lower()

    def test_track_graph_requires_edges(self):
        """Test that track_graph must have edges."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g)
        assert "edges" in str(exc_info.value).lower()

    def test_track_graph_nodes_require_pos_attribute(self):
        """Test that all nodes must have 'pos' attribute."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1)  # Missing pos
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=0.0)
        assert "pos" in str(exc_info.value)
        assert "attribute" in str(exc_info.value).lower()

    def test_track_graph_edges_require_distance_attribute(self):
        """Test that all edges must have 'distance' attribute."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, edge_id=0)  # Missing distance
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=0.0)
        assert "distance" in str(exc_info.value)
        assert "attribute" in str(exc_info.value).lower()

    def test_track_graph_edges_require_edge_id_attribute(self):
        """Test that all edges must have 'edge_id' attribute."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0)  # Missing edge_id
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=0.0)
        assert "edge_id" in str(exc_info.value)
        assert "attribute" in str(exc_info.value).lower()

    def test_track_graph_requires_edge_order(self):
        """Test that track_graph requires edge_order parameter."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ConfigurationError) as exc_info:
            Environment(track_graph=g)
        assert "edge_order" in str(exc_info.value)

    def test_track_graph_requires_edge_spacing(self):
        """Test that track_graph requires edge_spacing parameter."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ConfigurationError) as exc_info:
            # Provide edge_order so we get to the edge_spacing check
            Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=None)
        assert "edge_spacing" in str(exc_info.value)

    def test_valid_track_graph_minimal(self):
        """Test that valid track_graph configuration works."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        env = Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=0.0)
        assert env.track_graph is g


class TestEdgeOrderValidation:
    """Test validation of edge_order parameter."""

    def test_edge_order_must_be_list(self):
        """Test that edge_order must be a list."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=(0, 1), edge_spacing=0.0)
        assert "edge_order" in str(exc_info.value)
        assert "list" in str(exc_info.value).lower()

    def test_edge_order_elements_must_be_tuples(self):
        """Test that edge_order elements must be 2-tuples."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[[0, 1]], edge_spacing=0.0)
        assert "edge_order[0]" in str(exc_info.value)
        assert "2-tuple" in str(exc_info.value)

    def test_edge_order_must_reference_existing_edges(self):
        """Test that edge_order must reference edges that exist."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[(0, 2)], edge_spacing=0.0)
        assert "edge_order[0]" in str(exc_info.value)
        assert "non-existent" in str(exc_info.value).lower()

    def test_edge_order_accepts_reversed_edges(self):
        """Test that edge_order accepts edges in reverse order (undirected)."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        # Should work with reversed edge (1, 0) since graph is undirected
        env = Environment(track_graph=g, edge_order=[(1, 0)], edge_spacing=0.0)
        assert env.edge_order == [(1, 0)]


class TestEdgeSpacingValidation:
    """Test validation of edge_spacing parameter."""

    def test_edge_spacing_float_positive(self):
        """Test that edge_spacing as float works."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        env = Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=15.0)
        assert env.edge_spacing == 15.0

    def test_edge_spacing_negative_raises(self):
        """Test that negative edge_spacing raises."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[(0, 1)], edge_spacing=-1.0)
        assert "edge_spacing" in str(exc_info.value)

    def test_edge_spacing_list_wrong_length_raises(self):
        """Test that edge_spacing list must have correct length."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_node(2, pos=(2, 2))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        g.add_edge(1, 2, distance=1.0, edge_id=1)
        with pytest.raises(ValidationError) as exc_info:
            Environment(
                track_graph=g,
                edge_order=[(0, 1), (1, 2)],
                edge_spacing=[15.0, 15.0],  # Should be length 1 (n_edges - 1)
            )
        assert "edge_spacing" in str(exc_info.value)
        assert "length" in str(exc_info.value).lower()

    def test_edge_spacing_list_correct_length(self):
        """Test that edge_spacing list with correct length works."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_node(2, pos=(2, 2))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        g.add_edge(1, 2, distance=1.0, edge_id=1)
        env = Environment(
            track_graph=g, edge_order=[(0, 1), (1, 2)], edge_spacing=[15.0]
        )
        assert env.edge_spacing == [15.0]

    def test_edge_spacing_list_element_negative_raises(self):
        """Test that negative values in edge_spacing list raise."""
        g = nx.Graph()
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_node(2, pos=(2, 2))
        g.add_edge(0, 1, distance=1.0, edge_id=0)
        g.add_edge(1, 2, distance=1.0, edge_id=1)
        with pytest.raises(ValidationError) as exc_info:
            Environment(track_graph=g, edge_order=[(0, 1), (1, 2)], edge_spacing=[-1.0])
        assert "edge_spacing[0]" in str(exc_info.value)


class TestPositionRangeValidation:
    """Test validation of position_range parameter."""

    def test_position_range_must_be_sequence(self):
        """Test that position_range must be a sequence."""
        # String is a sequence, so it will fail on the element level
        # Let's test with a scalar instead
        with pytest.raises(ValidationError) as exc_info:
            Environment(position_range=123)
        assert "position_range" in str(exc_info.value)

    def test_position_range_elements_must_be_tuples(self):
        """Test that position_range elements must be 2-tuples."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(position_range=[[0, 100]])
        assert "position_range[0]" in str(exc_info.value)
        assert "2-tuple" in str(exc_info.value)

    def test_position_range_must_contain_numeric_values(self):
        """Test that position_range must contain numeric values."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(position_range=[("0", "100")])
        assert "position_range[0]" in str(exc_info.value)
        assert "numeric" in str(exc_info.value).lower()

    def test_position_range_min_must_be_less_than_max(self):
        """Test that min must be less than max in position_range."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(position_range=[(100, 0)])
        assert "position_range[0]" in str(exc_info.value)
        assert "min" in str(exc_info.value).lower()
        assert "max" in str(exc_info.value).lower()

    def test_position_range_equal_min_max_raises(self):
        """Test that equal min and max raises."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(position_range=[(0, 0)])
        assert "position_range[0]" in str(exc_info.value)

    def test_valid_position_range(self):
        """Test that valid position_range works."""
        env = Environment(position_range=[(0, 100), (0, 100)])
        assert env.position_range == [(0, 100), (0, 100)]


class TestFitPlaceGridValidation:
    """Test validation in fit_place_grid method."""

    def test_position_must_be_ndarray(self):
        """Test that position must be numpy array."""
        env = Environment()
        with pytest.raises(ValidationError) as exc_info:
            env.fit_place_grid(position=[[0, 0], [1, 1]])
        assert "position" in str(exc_info.value)
        assert "numpy" in str(exc_info.value).lower()

    def test_position_must_be_finite(self):
        """Test that position must contain finite values."""
        from non_local_detector.exceptions import DataError

        env = Environment()
        position = np.array([[0, 0], [np.nan, 1], [2, 2]])
        with pytest.raises(DataError) as exc_info:
            env.fit_place_grid(position=position)
        assert "position" in str(exc_info.value)
        assert "nan" in str(exc_info.value).lower()

    def test_position_must_be_1d_or_2d(self):
        """Test that position must be 1D or 2D."""
        env = Environment()
        position = np.array([[[0, 0], [1, 1]]])  # 3D
        with pytest.raises(ValidationError) as exc_info:
            env.fit_place_grid(position=position)
        assert "position" in str(exc_info.value)
        assert "1D or 2D" in str(exc_info.value)

    def test_position_cannot_be_empty(self):
        """Test that position cannot be empty."""
        env = Environment()
        position = np.array([]).reshape(0, 2)
        with pytest.raises(ValidationError) as exc_info:
            env.fit_place_grid(position=position)
        assert "position" in str(exc_info.value)
        assert "no time points" in str(exc_info.value).lower()

    def test_valid_position_1d(self):
        """Test that valid 1D position works."""
        env = Environment()
        # Need to reshape to 2D for fit_place_grid or provide 2D from the start
        position = np.array([[0], [1], [2], [3], [4]])
        result = env.fit_place_grid(position=position)
        assert result is env
        assert env.place_bin_centers_ is not None

    def test_valid_position_2d(self):
        """Test that valid 2D position works."""
        env = Environment()
        position = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        result = env.fit_place_grid(position=position)
        assert result is env
        assert env.place_bin_centers_ is not None


class TestIsTrackInteriorValidation:
    """Test validation of is_track_interior parameter."""

    def test_is_track_interior_must_be_ndarray(self):
        """Test that is_track_interior must be numpy array."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(is_track_interior=[[True, False], [False, True]])
        assert "is_track_interior" in str(exc_info.value)
        assert "numpy" in str(exc_info.value).lower()

    def test_valid_is_track_interior(self):
        """Test that valid is_track_interior works."""
        is_interior = np.array([True, False, True, False])
        env = Environment(is_track_interior=is_interior)
        assert np.array_equal(env.is_track_interior, is_interior)
