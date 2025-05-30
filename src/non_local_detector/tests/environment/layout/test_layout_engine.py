from typing import Any, Dict

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.environment import Environment
from non_local_detector.environment.layout.layout_engine import (
    GraphLayout,
    HexagonalLayout,
    ImageMaskLayout,
    LayoutEngine,
    MaskedGridLayout,
    RegularGridLayout,
    ShapelyPolygonLayout,
    create_layout,
    get_layout_parameters,
    list_available_layouts,
)
from non_local_detector.tests.environment.test_environment import plus_maze_data_samples

try:
    from shapely.geometry import Polygon as ShapelyPoly

    _HAS_SHAPELY_FOR_TEST = True
except ImportError:
    _HAS_SHAPELY_FOR_TEST = False
    ShapelyPoly = None


def add_edge_distances(graph: nx.Graph) -> nx.Graph:
    for u, v in graph.edges():
        pos_u = np.array(graph.nodes[u]["pos"])
        pos_v = np.array(graph.nodes[v]["pos"])
        graph.edges[u, v]["distance"] = np.linalg.norm(pos_v - pos_u)
    return graph


def test_list_available_layouts():
    layouts = list_available_layouts()
    assert isinstance(layouts, list)
    assert "RegularGrid" in layouts
    assert "Hexagonal" in layouts
    assert "Graph" in layouts
    assert "MaskedGrid" in layouts
    assert "ImageMask" in layouts


def test_get_layout_parameters_regular_grid():
    params = get_layout_parameters("RegularGrid")
    assert "bin_size" in params
    assert "dimension_ranges" in params
    assert params["bin_size"]["annotation"] is not None


def test_create_layout_regular_grid():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "RegularGrid",
        bin_size=2.0,
        data_samples=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.connectivity is not None
    assert layout.active_mask is not None
    assert layout.grid_edges is not None
    assert not layout.is_1d


def test_regular_grid_point_to_bin_index_and_neighbors():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "RegularGrid",
        bin_size=2.0,
        data_samples=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    points = np.array([[1.0, 1.0], [5.0, 5.0], [100.0, 100.0]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)
    neighbors = layout.neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_regular_grid_bin_area_volume():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "RegularGrid",
        bin_size=2.0,
        data_samples=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    areas = layout.bin_size()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers.shape[0]


def test_create_layout_hexagonal():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.connectivity is not None
    assert not layout.is_1d


def test_hexagonal_point_to_bin_index_and_neighbors():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    points = np.array([[1.0, 1.0], [5.0, 5.0], [100.0, 100.0]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)
    neighbors = layout.neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_hexagonal_bin_area_volume():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    areas = layout.bin_size()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers.shape[0]


def test_create_layout_masked_grid():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    edges = (
        np.arange(6).astype(float),
        np.arange(6).astype(float),
    )  # Ensure float for grid_edges
    layout = create_layout(
        "MaskedGrid",
        active_mask=mask,
        grid_edges=edges,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.bin_centers.shape[0] == np.sum(mask)  # Should be 9
    assert layout.connectivity is not None
    assert not layout.is_1d


def test_create_layout_image_mask():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = create_layout(
        "ImageMask",
        image_mask=mask,
        bin_size=1.0,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.connectivity is not None
    assert not layout.is_1d


def test_image_mask_point_to_bin_index_and_neighbors():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = create_layout(
        "ImageMask",
        image_mask=mask,
        bin_size=1.0,
    )
    points = np.array([[2.5, 2.5], [0.5, 0.5]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (2,)
    assert inds[0] >= 0
    neighbors = layout.neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_create_layout_graph():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    # Graph: 0 --1m-- 1 --1m-- 2. Total length 2m. Bin size 0.5m. Expected 4 bins.
    assert layout.bin_centers.shape[0] == 4
    assert layout.connectivity is not None
    assert layout.is_1d


def test_graph_point_to_bin_index_and_neighbors():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)
    neighbors = layout.neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_graph_bin_area_volume():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    lengths = layout.bin_size()
    assert np.allclose(lengths, lengths[0])
    assert lengths.shape[0] == layout.bin_centers.shape[0]


def test_create_layout_invalid_kind():
    with pytest.raises(ValueError):
        create_layout("NotARealLayout", foo=1)


def test_get_layout_parameters_invalid():
    with pytest.raises(ValueError):
        get_layout_parameters("NotARealLayout")


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
    return Environment.from_data_samples(
        data_samples=plus_maze_data_samples,  # Use existing samples
        layout_type="Hexagonal",
        hexagon_width=2.0,  # Reasonably large hexes
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
    return Environment.from_data_samples(
        data_samples=plus_maze_data_samples,  # Creates a reasonable grid
        bin_size=1.0,
        infer_active_bins=True,
        bin_count_threshold=0,
        name="GridForIndexing",
    )


MINIMAL_BUILD_PARAMS: Dict[str, Dict[str, Any]] = {
    "RegularGrid": {
        "bin_size": 1.0,
        "dimension_ranges": [(0, 2), (0, 2)],
        "infer_active_bins": False,  # Simplest case: all bins in range are active
    },
    "Hexagonal": {
        "hexagon_width": 1.0,
        "dimension_ranges": ((0, 2), (0, 2)),
        "infer_active_bins": False,
    },
    "Graph": {  # Requires a pre-built graph with 'pos', 'distance', 'edge_id'
        "graph_definition": nx.Graph(),  # Will be populated in test
        "edge_order": [],  # Will be populated in test
        "edge_spacing": 0.0,
        "bin_size": 0.5,
    },
    "MaskedGrid": {
        "active_mask": np.array([[True, False], [False, True]], dtype=bool),
        "grid_edges": (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])),
    },
    "ImageMask": {
        "image_mask": np.array([[True, False], [False, True]], dtype=bool),
        "bin_size": 1.0,
    },
}
if _HAS_SHAPELY_FOR_TEST and ShapelyPoly is not None:
    MINIMAL_BUILD_PARAMS["ShapelyPolygon"] = {
        "polygon": ShapelyPoly([(0, 0), (0, 1), (1, 1), (1, 0)]),
        "bin_size": 0.5,
    }


@pytest.mark.parametrize("layout_kind", list_available_layouts())
def test_layout_engine_protocol_adherence(
    layout_kind: str, simple_graph_for_layout: nx.Graph
):
    """
    Meta-test: checks if all layouts adhere to the LayoutEngine protocol.
    """
    if layout_kind not in MINIMAL_BUILD_PARAMS:
        pytest.skip(
            f"Minimal build parameters not defined for layout kind: {layout_kind}"
        )

    params = MINIMAL_BUILD_PARAMS[layout_kind].copy()

    if layout_kind == "Graph":
        # Setup specific params for GraphLayout
        params["graph_definition"] = simple_graph_for_layout
        params["edge_order"] = list(simple_graph_for_layout.edges())

    try:
        layout: LayoutEngine = create_layout(layout_kind, **params)
    except Exception as e:
        pytest.fail(f"Failed to create layout {layout_kind} with params {params}: {e}")

    # 1. Check presence of all protocol attributes
    assert hasattr(layout, "bin_centers"), f"{layout_kind} missing bin_centers"
    assert hasattr(layout, "connectivity"), f"{layout_kind} missing connectivity"
    assert hasattr(
        layout, "dimension_ranges"
    ), f"{layout_kind} missing dimension_ranges"
    assert hasattr(layout, "grid_edges"), f"{layout_kind} missing grid_edges"
    assert hasattr(layout, "grid_shape"), f"{layout_kind} missing grid_shape"
    assert hasattr(layout, "active_mask"), f"{layout_kind} missing active_mask"
    assert hasattr(
        layout, "_layout_type_tag"
    ), f"{layout_kind} missing _layout_type_tag"
    assert hasattr(
        layout, "_build_params_used"
    ), f"{layout_kind} missing _build_params_used"
    assert hasattr(layout, "is_1d"), f"{layout_kind} missing is_1d property"

    # 2. Check basic types and consistency
    assert isinstance(
        layout.bin_centers, np.ndarray
    ), f"{layout_kind}.bin_centers not ndarray"
    if layout.bin_centers.size > 0:  # Only check shape if not empty
        assert layout.bin_centers.ndim == 2, f"{layout_kind}.bin_centers not 2D"

    if (
        layout.connectivity is not None
    ):  # Optional for protocol, but usually present if active bins
        assert isinstance(
            layout.connectivity, nx.Graph
        ), f"{layout_kind}.connectivity not nx.Graph"
        if (
            layout.bin_centers.shape[0] > 0
        ):  # If there are active bins, graph should not be None
            assert layout.connectivity.number_of_nodes() == layout.bin_centers.shape[0]

    assert isinstance(layout.is_1d, bool), f"{layout_kind}.is_1d not bool"
    assert (
        layout._layout_type_tag == layout_kind
    ), f"{layout_kind}._layout_type_tag mismatch"
    # Ensure build_params_used stores the input params or a superset
    for k in params:
        assert (
            k in layout._build_params_used
        ), f"{layout_kind}._build_params_used missing '{k}'"

    if layout.active_mask is not None:
        assert isinstance(
            layout.active_mask, np.ndarray
        ), f"{layout_kind}.active_mask not ndarray"
        assert layout.active_mask.dtype == bool, f"{layout_kind}.active_mask not bool"
        if (
            layout.grid_shape is not None
            and len(layout.grid_shape) == layout.active_mask.ndim
        ):
            # For point-based layouts, grid_shape might be (N,) and active_mask (N,)
            # For grid-based, active_mask N-D shape matches grid_shape N-D shape
            if len(layout.grid_shape) > 1 or (
                len(layout.grid_shape) == 1
                and layout.grid_shape[0] != layout.active_mask.shape[0]
            ):
                # This check needs to be careful. If grid_shape is (N_active,), active_mask is (N_active,)
                # If grid_shape is (R,C), active_mask is (R,C)
                if not (
                    len(layout.grid_shape) == 1
                    and layout.grid_shape[0] == layout.active_mask.shape[0]
                    and layout.active_mask.ndim == 1
                ):
                    assert (
                        layout.active_mask.shape == layout.grid_shape
                    ), f"{layout_kind} active_mask shape vs grid_shape"

    # 3. Connectivity Graph Node/Edge Attributes (if graph exists and has elements)
    if layout.connectivity and layout.connectivity.number_of_nodes() > 0:
        sample_node = list(layout.connectivity.nodes())[0]
        node_data = layout.connectivity.nodes[sample_node]
        assert "pos" in node_data, f"{layout_kind} node missing 'pos'"
        assert isinstance(
            node_data["pos"], tuple
        ), f"{layout_kind} node 'pos' not tuple"
        assert (
            "source_grid_flat_index" in node_data
        ), f"{layout_kind} node missing 'source_grid_flat_index'"
        assert isinstance(
            node_data["source_grid_flat_index"], (int, np.integer)
        ), f"{layout_kind} node 'source_grid_flat_index' not int"
        assert (
            "original_grid_nd_index" in node_data
        ), f"{layout_kind} node missing 'original_grid_nd_index'"
        assert isinstance(
            node_data["original_grid_nd_index"], tuple
        ), f"{layout_kind} node 'original_grid_nd_index' not tuple"

        if layout.connectivity.number_of_edges() > 0:
            u, v, edge_data = list(layout.connectivity.edges(data=True))[0]
            assert "distance" in edge_data, f"{layout_kind} edge missing 'distance'"
            assert isinstance(
                edge_data["distance"], (float, np.floating)
            ), f"{layout_kind} edge 'distance' not float"
            assert (
                "weight" in edge_data
            ), f"{layout_kind} edge missing 'weight'"  # Usually present
            assert "edge_id" in edge_data  # Usually present
