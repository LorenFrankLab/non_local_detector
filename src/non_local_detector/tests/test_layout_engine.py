from typing import Any, Dict

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.environment import Environment
from non_local_detector.environment.layout_engine import (
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
from non_local_detector.tests.test_environment import plus_maze_data_samples

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
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.connectivity_graph_ is not None
    assert layout.active_mask_ is not None
    assert layout.grid_edges_ is not None
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
    neighbors = layout.get_bin_neighbors(inds[0])
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
    areas = layout.get_bin_area_volume()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers_.shape[0]


def test_create_layout_hexagonal():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.connectivity_graph_ is not None
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
    neighbors = layout.get_bin_neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_hexagonal_bin_area_volume():
    data = np.random.rand(100, 2) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    areas = layout.get_bin_area_volume()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers_.shape[0]


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
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.bin_centers_.shape[0] == np.sum(mask)  # Should be 9
    assert layout.connectivity_graph_ is not None
    assert not layout.is_1d


def test_create_layout_image_mask():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = create_layout(
        "ImageMask",
        image_mask=mask,
        bin_size=1.0,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.connectivity_graph_ is not None
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
    neighbors = layout.get_bin_neighbors(inds[0])
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
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    # Graph: 0 --1m-- 1 --1m-- 2. Total length 2m. Bin size 0.5m. Expected 4 bins.
    assert layout.bin_centers_.shape[0] == 4
    assert layout.connectivity_graph_ is not None
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
    neighbors = layout.get_bin_neighbors(inds[0])
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
    lengths = layout.get_bin_area_volume()
    assert np.allclose(lengths, lengths[0])
    assert lengths.shape[0] == layout.bin_centers_.shape[0]


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
        environment_name="SimpleHexEnvForMask",
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
        environment_name="SimpleGraphEnvForMask",
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
        environment_name="GridForIndexing",
    )


class TestMaskRegionsWithLayouts:

    def test_mask_region_with_graph_layout(self, simple_graph_env: Environment):
        """Test adding a 1D mask region to a GraphLayout environment."""
        env = simple_graph_env
        assert env.is_1d
        assert env.grid_shape_ is not None, "GraphLayout should have a 1D grid_shape"
        assert len(env.grid_shape_) == 1, "GraphLayout grid_shape should be 1D"

        n_linear_bins = env.grid_shape_[0]
        if n_linear_bins == 0:
            pytest.skip("Graph environment has no bins, cannot test mask region.")

        # Create a 1D mask for the GraphLayout's linearized bins
        # Example: make the first half of the track part of the region
        mask_1d = np.zeros(n_linear_bins, dtype=bool)
        mid_point = n_linear_bins // 2
        if mid_point == 0 and n_linear_bins > 0:
            mid_point = 1  # ensure at least one if possible
        mask_1d[:mid_point] = True

        env.regions.add_region(name="graph_mask_region", mask=mask_1d)

        # bins_in_region returns indices relative to active bins.
        # For GraphLayout, all linearized bins are typically active and correspond 1-to-1
        # with the source_grid_flat_index.
        region_bins = env.regions.bins_in_region("graph_mask_region")

        expected_bins_in_region = np.flatnonzero(mask_1d)
        assert np.array_equal(np.sort(region_bins), np.sort(expected_bins_in_region))

        # region_mask gives a mask over active bins
        active_bin_mask = env.regions.region_mask("graph_mask_region")
        assert active_bin_mask.shape == (env.bin_centers_.shape[0],)
        assert np.sum(active_bin_mask) == np.sum(mask_1d)
        assert np.all(active_bin_mask[:mid_point])
        if mid_point < n_linear_bins:
            assert np.all(~active_bin_mask[mid_point:])

    def test_mask_region_with_hexagonal_layout(self, simple_hex_env: Environment):
        """Test adding a 2D mask region to a HexagonalLayout environment."""
        env = simple_hex_env
        assert not env.is_1d
        assert env.grid_shape_ is not None, "HexagonalLayout should have a grid_shape"
        assert len(env.grid_shape_) == 2, "HexagonalLayout grid_shape should be 2D"

        # Create a 2D mask matching the HexagonalLayout's full conceptual grid
        # Example: make a strip or a corner of the hex grid part of the region
        hex_mask_2d = np.zeros(env.grid_shape_, dtype=bool)
        rows, cols = env.grid_shape_
        hex_mask_2d[0 : rows // 2, 0 : cols // 2] = (
            True  # Top-left quadrant of hex grid
        )

        env.regions.add_region(name="hex_mask_region", mask=hex_mask_2d)

        # bins_in_region returns indices relative to active environment bins
        region_bins = env.regions.bins_in_region("hex_mask_region")

        # Determine expected active bins:
        # 1. Get original flat indices of hex_mask_2d
        # 2. Filter these by which ones are *also* in env.active_mask_ (the active hexes)
        # 3. Map these original flat indices (of hexes in region AND active in env) to active bin IDs
        original_flat_indices_in_region_mask = np.flatnonzero(hex_mask_2d)

        expected_active_bins_in_region = []
        if env._source_flat_to_active_node_id_map is not None:
            for orig_flat_idx in original_flat_indices_in_region_mask:
                # Check if this original hex is actually an active hex in the environment
                # The env.active_mask_ is N-D, env.layout.active_mask_ is also N-D
                # This check can be done by seeing if orig_flat_idx is a key in the map
                active_node_id = env._source_flat_to_active_node_id_map.get(
                    orig_flat_idx
                )
                if active_node_id is not None:
                    expected_active_bins_in_region.append(active_node_id)

        assert np.array_equal(
            np.sort(region_bins), np.sort(expected_active_bins_in_region)
        )

        active_bin_mask_for_region = env.regions.region_mask("hex_mask_region")
        assert active_bin_mask_for_region.shape == (env.bin_centers_.shape[0],)
        assert np.sum(active_bin_mask_for_region) == len(expected_active_bins_in_region)


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
    assert hasattr(layout, "bin_centers_"), f"{layout_kind} missing bin_centers_"
    assert hasattr(
        layout, "connectivity_graph_"
    ), f"{layout_kind} missing connectivity_graph_"
    assert hasattr(
        layout, "dimension_ranges_"
    ), f"{layout_kind} missing dimension_ranges_"
    assert hasattr(layout, "grid_edges_"), f"{layout_kind} missing grid_edges_"
    assert hasattr(layout, "grid_shape_"), f"{layout_kind} missing grid_shape_"
    assert hasattr(layout, "active_mask_"), f"{layout_kind} missing active_mask_"
    assert hasattr(
        layout, "_layout_type_tag"
    ), f"{layout_kind} missing _layout_type_tag"
    assert hasattr(
        layout, "_build_params_used"
    ), f"{layout_kind} missing _build_params_used"
    assert hasattr(layout, "is_1d"), f"{layout_kind} missing is_1d property"

    # 2. Check basic types and consistency
    assert isinstance(
        layout.bin_centers_, np.ndarray
    ), f"{layout_kind}.bin_centers_ not ndarray"
    if layout.bin_centers_.size > 0:  # Only check shape if not empty
        assert layout.bin_centers_.ndim == 2, f"{layout_kind}.bin_centers_ not 2D"

    if (
        layout.connectivity_graph_ is not None
    ):  # Optional for protocol, but usually present if active bins
        assert isinstance(
            layout.connectivity_graph_, nx.Graph
        ), f"{layout_kind}.connectivity_graph_ not nx.Graph"
        if (
            layout.bin_centers_.shape[0] > 0
        ):  # If there are active bins, graph should not be None
            assert (
                layout.connectivity_graph_.number_of_nodes()
                == layout.bin_centers_.shape[0]
            )

    assert isinstance(layout.is_1d, bool), f"{layout_kind}.is_1d not bool"
    assert (
        layout._layout_type_tag == layout_kind
    ), f"{layout_kind}._layout_type_tag mismatch"
    # Ensure build_params_used stores the input params or a superset
    for k in params:
        assert (
            k in layout._build_params_used
        ), f"{layout_kind}._build_params_used missing '{k}'"

    if layout.active_mask_ is not None:
        assert isinstance(
            layout.active_mask_, np.ndarray
        ), f"{layout_kind}.active_mask_ not ndarray"
        assert layout.active_mask_.dtype == bool, f"{layout_kind}.active_mask_ not bool"
        if (
            layout.grid_shape_ is not None
            and len(layout.grid_shape_) == layout.active_mask_.ndim
        ):
            # For point-based layouts, grid_shape might be (N,) and active_mask (N,)
            # For grid-based, active_mask N-D shape matches grid_shape N-D shape
            if len(layout.grid_shape_) > 1 or (
                len(layout.grid_shape_) == 1
                and layout.grid_shape_[0] != layout.active_mask_.shape[0]
            ):
                # This check needs to be careful. If grid_shape is (N_active,), active_mask is (N_active,)
                # If grid_shape is (R,C), active_mask is (R,C)
                if not (
                    len(layout.grid_shape_) == 1
                    and layout.grid_shape_[0] == layout.active_mask_.shape[0]
                    and layout.active_mask_.ndim == 1
                ):
                    assert (
                        layout.active_mask_.shape == layout.grid_shape_
                    ), f"{layout_kind} active_mask shape vs grid_shape"

    # 3. Connectivity Graph Node/Edge Attributes (if graph exists and has elements)
    if layout.connectivity_graph_ and layout.connectivity_graph_.number_of_nodes() > 0:
        sample_node = list(layout.connectivity_graph_.nodes())[0]
        node_data = layout.connectivity_graph_.nodes[sample_node]
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

        if layout.connectivity_graph_.number_of_edges() > 0:
            u, v, edge_data = list(layout.connectivity_graph_.edges(data=True))[0]
            assert "distance" in edge_data, f"{layout_kind} edge missing 'distance'"
            assert isinstance(
                edge_data["distance"], (float, np.floating)
            ), f"{layout_kind} edge 'distance' not float"
            assert (
                "weight" in edge_data
            ), f"{layout_kind} edge missing 'weight'"  # Usually present
            assert "edge_id" in edge_data  # Usually present
