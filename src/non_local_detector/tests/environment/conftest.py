from typing import List, Tuple

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from non_local_detector.environment import Environment
from non_local_detector.environment.layout.engines.graph import GraphLayout


# --- Fixtures ---
@pytest.fixture
def plus_maze_graph() -> nx.Graph:
    """
    Defines a simple plus-shaped maze graph.
    Center node (0) at (0, 0)
    Arm 1 (North): Node 1 at (0, 2)
    Arm 2 (East): Node 2 at (2, 0)
    Arm 3 (South): Node 3 at (0, -2)
    Arm 4 (West): Node 4 at (-2, 0)
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(0.0, 2.0))  # North
    graph.add_node(2, pos=(2.0, 0.0))  # East
    graph.add_node(3, pos=(0.0, -2.0))  # South
    graph.add_node(4, pos=(-2.0, 0.0))  # West

    # Add edge_id, as expected by track_linearization
    graph.add_edge(0, 1, distance=2.0, edge_id=0)
    graph.add_edge(0, 2, distance=2.0, edge_id=1)
    graph.add_edge(0, 3, distance=2.0, edge_id=2)
    graph.add_edge(0, 4, distance=2.0, edge_id=3)
    return graph


@pytest.fixture
def plus_maze_edge_order() -> List[Tuple[int, int]]:
    """Edge order for linearizing the plus maze."""
    # Path: West arm -> Center -> North arm -> Center -> East arm -> Center -> South arm
    return [(4, 0), (0, 1), (0, 2), (0, 3)]


@pytest.fixture
def plus_maze_data_samples() -> NDArray[np.float64]:
    """Regularly spaced data samples along the plus maze arms."""
    samples = [
        # Center
        [0.0, 0.0],
        # West arm: (-2,0) to (0,0)
        [-2.0, 0.0],
        [-1.5, 0.0],
        [-1.0, 0.0],
        [-0.5, 0.0],
        # North arm: (0,0) to (0,2)
        [0.0, 0.5],
        [0.0, 1.0],
        [0.0, 1.5],
        [0.0, 2.0],
        # East arm: (0,0) to (2,0)
        [0.5, 0.0],
        [1.0, 0.0],
        [1.5, 0.0],
        [2.0, 0.0],
        # South arm: (0,0) to (0,-2)
        [0.0, -0.5],
        [0.0, -1.0],
        [0.0, -1.5],
        [0.0, -2.0],
        # Some off-track points
        [3.0, 3.0],
        [-3.0, -3.0],
    ]
    return np.array(samples, dtype=np.float64)


@pytest.fixture
def graph_env(
    plus_maze_graph: nx.Graph, plus_maze_edge_order: List[Tuple[int, int]]
) -> Environment:
    """Environment created from the plus maze graph."""
    # Capture parameters explicitly to pass to Environment constructor
    # for correct serialization testing.
    layout_build_params = {
        "graph_definition": plus_maze_graph,
        "edge_order": plus_maze_edge_order,
        "edge_spacing": 0.0,
        "bin_size": 0.5,
    }
    layout_instance = GraphLayout()
    layout_instance.build(**layout_build_params)

    return Environment(
        name="PlusMazeGraph",
        layout=layout_instance,
        layout_type_used="Graph",
        layout_params_used=layout_build_params,
    )


@pytest.fixture
def grid_env_from_samples(
    plus_maze_data_samples: NDArray[np.float64],
) -> Environment:
    """Environment created as a RegularGrid from plus maze data samples."""
    return Environment.from_samples(
        data_samples=plus_maze_data_samples,
        bin_size=0.5,
        infer_active_bins=True,
        bin_count_threshold=0,  # A single sample makes a bin active
        dilate=False,  # Keep it simple, no dilation
        fill_holes=False,
        close_gaps=False,
        name="PlusMazeGrid",
        connect_diagonal_neighbors=False,  # Only orthogonal for easier neighbor check
    )


@pytest.fixture
def simple_graph_for_layout() -> nx.Graph:
    """Minimal graph with pos and distance attributes for GraphLayout."""
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1, pos=(1.0, 0.0))
    G.add_edge(0, 1, distance=1.0, edge_id=0)  # Add edge_id
    return G


@pytest.fixture
def simple_hex_env(plus_maze_data_samples) -> Environment:
    """Basic hexagonal environment for mask testing."""
    return Environment.from_samples(
        data_samples=plus_maze_data_samples,  # Use existing samples
        layout_kind="Hexagonal",
        bin_size=2.0,  # Reasonably large hexes
        name="SimpleHexEnvForMask",
        infer_active_bins=True,  # Important for source_flat_to_active_node_id_map
        bin_count_threshold=0,
    )


@pytest.fixture
def simple_graph_env(simple_graph_for_layout) -> Environment:
    """Basic graph environment for mask testing."""
    edge_order = [(0, 1)]
    # For serialization to pass correctly, ensure layout_params_used are captured
    layout_build_params = {
        "graph_definition": simple_graph_for_layout,
        "edge_order": edge_order,
        "edge_spacing": 0.0,
        "bin_size": 0.5,
    }
    layout_instance = GraphLayout()
    layout_instance.build(**layout_build_params)
    return Environment(
        name="SimpleGraphEnvForMask",
        layout=layout_instance,
        layout_type_used="Graph",
        layout_params_used=layout_build_params,
    )


@pytest.fixture
def grid_env_for_indexing(plus_maze_data_samples) -> Environment:
    """A 2D RegularGrid environment suitable for index testing."""
    return Environment.from_samples(
        data_samples=plus_maze_data_samples,  # Creates a reasonable grid
        bin_size=1.0,
        infer_active_bins=True,
        bin_count_threshold=0,
        name="GridForIndexing",
    )


@pytest.fixture
def env_all_active_2x2() -> Environment:
    """A 2x2 grid where all 4 cells are active."""
    active_mask = np.array([[True, True], [True, True]], dtype=bool)
    grid_edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    return Environment.from_mask(
        active_mask=active_mask,
        grid_edges=grid_edges,
        name="AllActive2x2",
        connect_diagonal_neighbors=False,  # Orthogonal connections for simpler graph
    )
