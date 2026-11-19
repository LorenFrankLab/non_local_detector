"""``Environment`` takes its documented arguments in any container or scalar type.

Parameter rows that pass through DataJoint can hold a tuple, a list or an
array, and NumPy scalars where Python ones were given. Every accepted form of
``edge_order``, ``edge_spacing`` and ``position_range`` is validated alike and
fits the same grid as the plain Python form.
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from track_linearization import make_track_graph

from non_local_detector.environment import _LEGACY_FITTED_ATTRIBUTES, Environment
from non_local_detector.exceptions import ValidationError

pytestmark = pytest.mark.unit

# Two 10 cm edges at a right angle, 2 cm bins.
NODE_POSITIONS = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
EDGES = [(0, 1), (1, 2)]
# With 15 cm between the edges, the first edge spans 0..10 and the second
# 25..35 of the linear axis.
SPACED_EDGES = np.array(
    [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0]
)
# On the first edge, 2 cm along it; on the second, 3 cm along it.
POINTS = np.array([[2.0, 0.0], [10.0, 3.0]])
LINEAR_POINTS = np.array([[2.0], [28.0]])

EDGE_ORDER_FORMS = {
    "list of tuples": [(0, 1), (1, 2)],
    "tuple of tuples": ((0, 1), (1, 2)),
    "list of lists": [[0, 1], [1, 2]],
    "array": np.array([[0, 1], [1, 2]]),
}
EDGE_SPACING_FORMS = {
    "int": 15,
    "float": 15.0,
    "np.int64": np.int64(15),
    "np.float32": np.float32(15),
    "np.float64": np.float64(15),
    "0-d array": np.array(15.0),
    "list": [15.0],
    "tuple": (15.0,),
    "array": np.array([15.0]),
    "list of np.int64": [np.int64(15)],
}
POSITION_RANGE_FORMS = {
    "list of tuples": [(0.0, 100.0), (0.0, 50.0)],
    "tuple of tuples": ((0.0, 100.0), (0.0, 50.0)),
    "list of lists": [[0.0, 100.0], [0.0, 50.0]],
    "array": np.array([[0.0, 100.0], [0.0, 50.0]]),
    "list of arrays": [np.array([0.0, 100.0]), np.array([0.0, 50.0])],
    "int endpoints": [(0, 100), (0, 50)],
    "np.int64 endpoints": [(np.int64(0), np.int64(100)), (np.int64(0), np.int64(50))],
    "np.float32 endpoints": [
        (np.float32(0), np.float32(100)),
        (np.float32(0), np.float32(50)),
    ],
}
OPEN_FIELD_POSITION = np.random.default_rng(0).uniform(
    [0.0, 0.0], [100.0, 50.0], size=(2_000, 2)
)


def _graph() -> nx.Graph:
    return make_track_graph(NODE_POSITIONS, EDGES)


def _track_environment(edge_order, edge_spacing) -> Environment:
    return Environment(
        place_bin_size=2.0,
        track_graph=_graph(),
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )


def _assert_same_grid(environment: Environment, reference: Environment) -> None:
    """Every fitted array and table of ``environment`` equals ``reference``'s."""
    assert len(environment.edges_) == len(reference.edges_)
    for edge, reference_edge in zip(environment.edges_, reference.edges_, strict=True):
        assert edge.dtype == reference_edge.dtype
        np.testing.assert_array_equal(edge, reference_edge)
    for name in ("place_bin_edges_", "place_bin_centers_", "is_track_interior_"):
        value, expected = getattr(environment, name), getattr(reference, name)
        assert value.dtype == expected.dtype, name
        np.testing.assert_array_equal(value, expected, err_msg=name)
    assert environment.centers_shape_ == reference.centers_shape_
    if reference.track_graph is not None:
        pd.testing.assert_frame_equal(environment.nodes_df_, reference.nodes_df_)
        assert environment.distance_between_nodes_ == reference.distance_between_nodes_


@pytest.fixture(scope="module")
def reference_track() -> Environment:
    return _track_environment([(0, 1), (1, 2)], 15.0).fit_place_grid()


@pytest.mark.parametrize(
    "edge_order", EDGE_ORDER_FORMS.values(), ids=list(EDGE_ORDER_FORMS)
)
@pytest.mark.parametrize(
    "edge_spacing", EDGE_SPACING_FORMS.values(), ids=list(EDGE_SPACING_FORMS)
)
def test_every_track_argument_form_fits_the_same_grid(
    edge_order, edge_spacing, reference_track
):
    environment = _track_environment(edge_order, edge_spacing).fit_place_grid()

    np.testing.assert_array_equal(environment.edges_[0], SPACED_EDGES)
    _assert_same_grid(environment, reference_track)
    np.testing.assert_allclose(
        environment.decoding_position(POINTS), LINEAR_POINTS, atol=1e-12
    )


@pytest.mark.parametrize(
    "edge_order", EDGE_ORDER_FORMS.values(), ids=list(EDGE_ORDER_FORMS)
)
def test_track_argument_forms_survive_the_parameter_round_trip(
    edge_order, reference_track
):
    """``Environment(**get_params())`` and a pickle keep the form and the grid."""
    environment = _track_environment(edge_order, (15.0,))

    rebuilt = Environment(**environment.get_params()).fit_place_grid()
    unpickled = pickle.loads(pickle.dumps(environment.fit_place_grid()))

    for copy in (rebuilt, unpickled):
        _assert_same_grid(copy, reference_track)
        np.testing.assert_array_equal(
            copy.decoding_position(POINTS), reference_track.decoding_position(POINTS)
        )


def test_released_parameter_row_with_a_tuple_edge_order_builds(reference_track):
    """A row shaped like ``vars(Environment(...))`` of release 0.6.9."""
    row = {
        "environment_name": "",
        "place_bin_size": 2.0,
        "track_graph": _graph(),
        "edge_order": ((0, 1), (1, 2)),
        "edge_spacing": (15.0,),
        "is_track_interior": None,
        "position_range": None,
        "infer_track_interior": True,
        "fill_holes": False,
        "dilate": False,
        "bin_count_threshold": 0,
    }

    environment = Environment(**row).fit_place_grid()

    _assert_same_grid(environment, reference_track)


def test_unpickled_environment_with_stored_tuples_decodes_without_a_refit(
    reference_track,
):
    """An environment pickled holding tuples keeps them, and still works.

    The state is laid out as releases that stored the fitted attributes as
    fields pickled it.
    """
    state = {
        **reference_track.get_params(),
        "edge_order": ((0, 1), (1, 2)),
        "edge_spacing": (15.0,),
    }
    for name, grid_name in _LEGACY_FITTED_ATTRIBUTES.items():
        state[name] = getattr(reference_track.place_grid_, grid_name)
    environment = Environment.__new__(Environment)
    environment.__setstate__(pickle.loads(pickle.dumps(state)))

    assert environment.edge_spacing == (15.0,)
    _assert_same_grid(environment, reference_track)
    np.testing.assert_allclose(
        environment.decoding_position(POINTS), LINEAR_POINTS, atol=1e-12
    )


@pytest.mark.parametrize(
    "edge_spacing",
    [
        (),
        (15.0, 15.0),
        (-5.0,),
        np.array([-5.0]),
        np.int64(-1),
        np.array([[15.0]]),
        "15",
        (True,),
    ],
    ids=[
        "empty tuple",
        "too long",
        "negative tuple",
        "negative array",
        "negative np.int64",
        "2-d array",
        "string",
        "bool",
    ],
)
def test_invalid_edge_spacing_is_refused_in_every_form(edge_spacing):
    with pytest.raises(ValidationError, match="edge_spacing"):
        _track_environment([(0, 1), (1, 2)], edge_spacing)


@pytest.mark.parametrize(
    "edge_order",
    [np.array([0, 1]), [(0, 1, 2), (1, 2)], ["01", (1, 2)], 7],
    ids=["1-d array", "triple", "string", "int"],
)
def test_invalid_edge_order_is_refused_in_every_form(edge_order):
    with pytest.raises(ValidationError, match="edge_order"):
        _track_environment(edge_order, 0.0)


def test_edge_order_that_omits_a_graph_edge_is_refused_naming_it():
    graph = make_track_graph(
        [(0, 0), (10, 0), (20, 0), (10, 10)], [(0, 1), (1, 2), (1, 3)]
    )

    with pytest.raises(ValidationError, match=r"\(1, 3\)"):
        Environment(track_graph=graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0)


@pytest.fixture(scope="module")
def reference_open_field() -> Environment:
    return Environment(
        place_bin_size=10.0, position_range=[(0.0, 100.0), (0.0, 50.0)]
    ).fit_place_grid(OPEN_FIELD_POSITION)


@pytest.mark.parametrize(
    "position_range", POSITION_RANGE_FORMS.values(), ids=list(POSITION_RANGE_FORMS)
)
def test_every_position_range_form_fits_the_same_grid(
    position_range, reference_open_field
):
    environment = Environment(
        place_bin_size=10.0, position_range=position_range
    ).fit_place_grid(OPEN_FIELD_POSITION)

    # One padding bin outside the range at each end of each dimension.
    np.testing.assert_array_equal(environment.edges_[0], np.arange(-10.0, 111.0, 10.0))
    np.testing.assert_array_equal(environment.edges_[1], np.arange(-10.0, 61.0, 10.0))
    _assert_same_grid(environment, reference_open_field)


@pytest.mark.parametrize(
    "position_range",
    [
        np.array([0.0, 100.0]),
        [(0.0, 50.0, 100.0)],
        np.array([[100.0, 0.0]]),
        [("0", "100")],
        [(True, 2.0)],
    ],
    ids=["1-d array", "triple", "array min above max", "strings", "bool"],
)
def test_invalid_position_range_is_refused_in_every_form(position_range):
    with pytest.raises(ValidationError, match="position_range"):
        Environment(position_range=position_range)
