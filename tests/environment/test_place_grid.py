"""The fitted geometry of an ``Environment`` is one immutable ``PlaceGrid``.

``fit_place_grid`` builds the whole grid before assigning it, so a fit that
raises leaves the environment as it was, previous fit included. The fitted
attributes (``place_bin_centers_``, ``edges_``, ...) read the grid; they raise
``NotFittedError`` before a fit, and after a configuration field the grid
depends on changes, until ``fit_place_grid`` runs again.
"""

import copy
import copyreg
import dataclasses
import pickle

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment, PlaceGrid, get_centers
from non_local_detector.exceptions import (
    ConfigurationError,
    NotFittedError,
    ValidationError,
)

pytestmark = pytest.mark.unit

POSITION = np.random.default_rng(0).uniform(0.0, 20.0, size=(500, 2))
FITTED_ATTRIBUTES = (
    "edges_",
    "place_bin_edges_",
    "place_bin_centers_",
    "centers_shape_",
    "is_track_interior_",
    "infer_track_interior_",
    "is_track_boundary_",
    "track_graphDD",
    "distance_between_nodes_",
    "track_graph_with_bin_centers_edges_",
    "original_nodes_df_",
    "place_bin_edges_nodes_df_",
    "place_bin_centers_nodes_df_",
    "nodes_df_",
)


def _track_environment() -> Environment:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    return Environment(
        place_bin_size=1.0, track_graph=graph, edge_order=[(0, 1)], edge_spacing=0.0
    )


@pytest.fixture
def fitted():
    environment = Environment(place_bin_size=2.0).fit_place_grid(POSITION)
    environment.set_cached("derived", np.arange(3))
    return environment


@pytest.mark.parametrize("name", FITTED_ATTRIBUTES)
def test_fitted_attributes_raise_before_a_fit(name):
    environment = Environment()
    with pytest.raises(NotFittedError, match="fit_place_grid"):
        getattr(environment, name)
    # NotFittedError is an AttributeError, so hasattr and getattr defaults
    # treat the attribute as absent.
    assert not hasattr(environment, name)
    assert getattr(environment, name, None) is None
    assert not environment._is_fitted


def test_not_fitted_error_is_a_configuration_error():
    with pytest.raises(ConfigurationError):
        _ = Environment().place_bin_centers_


@pytest.mark.parametrize("make", [lambda: Environment(), _track_environment])
def test_attributes_read_the_one_place_grid(make):
    environment = make()
    environment.fit_place_grid(None if environment.track_graph else POSITION)
    grid = environment.place_grid_
    assert isinstance(grid, PlaceGrid)
    assert environment.place_bin_centers_ is grid.place_bin_centers
    assert environment.is_track_interior_ is grid.is_track_interior
    assert environment.edges_ is grid.edges
    assert environment.distance_between_nodes_ is grid.distance_between_nodes
    assert environment._is_fitted
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.place_bin_centers = np.zeros((1, 2))  # type: ignore[misc]


def test_fitted_attributes_are_read_only(fitted):
    with pytest.raises(AttributeError):
        fitted.place_bin_centers_ = np.zeros((1, 2))


def test_failed_refit_leaves_the_previous_fit(fitted):
    grid = fitted.place_grid_
    with pytest.raises(ValidationError, match="position is required"):
        fitted.fit_place_grid(None)
    assert fitted.place_grid_ is grid
    np.testing.assert_array_equal(fitted.get_cached("derived"), np.arange(3))


def test_refit_failing_after_the_grid_is_built_leaves_the_previous_fit():
    """The interior mask is checked against the new grid, after it is built."""
    shape = Environment(place_bin_size=2.0).fit_place_grid(POSITION).centers_shape_
    environment = Environment(
        place_bin_size=2.0, is_track_interior=np.ones(shape, dtype=bool)
    )
    environment.fit_place_grid(POSITION)
    grid = environment.place_grid_
    environment.set_cached("derived", 1)

    with pytest.raises(ValidationError, match="is_track_interior shape"):
        # Twice the extent: twice the bins per dimension.
        environment.fit_place_grid(POSITION * 2.0)
    assert environment.place_grid_ is grid
    assert environment.get_cached("derived") == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("place_bin_size", lambda _: 5.0),
        ("position_range", lambda _: [(0.0, 40.0), (0.0, 40.0)]),
        ("is_track_interior", lambda env: np.ones(env.centers_shape_, dtype=bool)),
        ("fill_holes", lambda _: True),
        ("dilate", lambda _: True),
        ("bin_count_threshold", lambda _: 3),
    ],
)
def test_changed_grid_configuration_is_detected_at_use(fitted, name, value):
    setattr(fitted, name, value(fitted))
    with pytest.raises(NotFittedError, match=name):
        _ = fitted.place_bin_centers_
    with pytest.raises(ConfigurationError, match=name):
        fitted.get_bin_ind(POSITION[:3])
    assert not fitted._is_fitted
    fitted.fit_place_grid(POSITION)
    assert fitted.place_bin_centers_.shape[0] > 0


def test_changed_track_graph_setting_is_detected_at_use():
    environment = _track_environment().fit_place_grid()
    environment.edge_spacing = 3.0
    with pytest.raises(NotFittedError, match="edge_spacing"):
        _ = environment.place_bin_centers_


def test_settings_the_grid_ignores_keep_the_fit(fitted):
    grid = fitted.place_grid_
    fitted.environment_name = "renamed"
    # fit_place_grid's argument, not this field, decides the interior.
    fitted.infer_track_interior = False
    # An equal value is the same configuration.
    fitted.place_bin_size = float(fitted.place_bin_size)
    assert fitted.place_grid_ is grid


def test_copies_keep_their_fit(fitted):
    for copied in (copy.deepcopy(fitted), pickle.loads(pickle.dumps(fitted))):
        np.testing.assert_array_equal(
            copied.place_bin_centers_, fitted.place_bin_centers_
        )
        assert copied.place_grid_ is not fitted.place_grid_


def _legacy_state(environment, *, drop=()):
    """``vars()`` of ``environment`` as releases that stored the fitted
    attributes as fields pickled it, without the attributes in ``drop``."""
    state = {**environment.get_params(), "_cache": {}}
    for name in FITTED_ATTRIBUTES:
        if name not in drop:
            state[name] = getattr(environment, name)
    return state


def _unpickle_legacy(state):
    class _Legacy:
        def __reduce__(self):
            return (copyreg._reconstructor, (Environment, object, None), state)

    return pickle.loads(pickle.dumps(_Legacy()))


def test_legacy_open_field_without_distances_gets_those_of_a_fresh_fit(fitted):
    """Releases before 0.5.13 stored no graph or distances for an open field;
    loading rebuilds them as ``fit_place_grid`` does."""
    loaded = _unpickle_legacy(
        _legacy_state(fitted, drop=("distance_between_nodes_", "track_graphDD"))
    )

    np.testing.assert_array_equal(
        loaded.distance_between_nodes_, fitted.distance_between_nodes_
    )
    assert nx.utils.graphs_equal(loaded.track_graphDD, fitted.track_graphDD)
    centers = loaded.place_bin_centers_[loaded.is_track_interior_flat_]
    np.testing.assert_array_equal(
        loaded.get_manifold_distances(centers[:5], centers[5:10]),
        fitted.get_manifold_distances(centers[:5], centers[5:10]),
    )


def test_legacy_infer_track_interior_comes_from_the_configuration(fitted):
    """Released fits stored the ``infer_track_interior`` they used in the
    configuration field, and no ``infer_track_interior_``."""
    state = _legacy_state(fitted, drop=("infer_track_interior_",))
    state["infer_track_interior"] = False
    assert _unpickle_legacy(state).infer_track_interior_ is False


def test_legacy_track_graph_grid_without_distances_gets_them():
    """No stored grid loads without distances, so no reader needs a fallback."""
    environment = _track_environment().fit_place_grid()
    loaded = _unpickle_legacy(
        _legacy_state(environment, drop=("distance_between_nodes_",))
    )
    assert loaded.distance_between_nodes_ == environment.distance_between_nodes_


def test_legacy_unset_fitted_arguments_are_dropped_with_a_warning():
    """``vars(environment)`` rows of main (2025-10 to 2026-10) hold the fitted
    attributes as None; the constructor drops them and warns."""
    with pytest.warns(DeprecationWarning, match="place_bin_centers_") as record:
        environment = Environment(
            place_bin_size=3.0, place_bin_centers_=None, _is_fitted=False
        )
    assert len(record) == 1
    assert "_is_fitted" in str(record[0].message)
    assert environment.get_params() == Environment(place_bin_size=3.0).get_params()


def test_equal_configuration_in_another_container_keeps_the_fit():
    environment = Environment(position_range=[(0.0, 20.0), (0.0, 20.0)])
    environment.fit_place_grid(POSITION)
    grid = environment.place_grid_
    environment.position_range = ((0.0, 20.0), (0.0, 20.0))
    assert environment.place_grid_ is grid


def test_in_place_edits_of_configuration_are_detected():
    shape = Environment().fit_place_grid(POSITION).centers_shape_
    mask = np.ones(shape, dtype=bool)
    position_range = [(0.0, 20.0), (0.0, 20.0)]
    environment = Environment(is_track_interior=mask, position_range=position_range)
    environment.fit_place_grid(POSITION)

    mask[0, 0] = False
    with pytest.raises(NotFittedError, match="is_track_interior"):
        _ = environment.place_bin_centers_
    mask[0, 0] = True
    position_range[0] = (0.0, 30.0)
    with pytest.raises(NotFittedError, match="position_range"):
        _ = environment.place_bin_centers_


def test_cache_is_dropped_when_the_configuration_changes(fitted):
    fitted.place_bin_size = 5.0
    assert fitted.get_cached("derived") is None


def test_grid_configuration_is_read_only(fitted):
    configuration = fitted.place_grid_.configuration
    with pytest.raises(TypeError):
        configuration["place_bin_size"] = 3.0  # type: ignore[index]
    for copied in (copy.deepcopy(fitted), pickle.loads(pickle.dumps(fitted))):
        assert copied.place_grid_.configuration["place_bin_size"] == 2.0


def test_legacy_open_field_with_x_fastest_centers_gets_its_own_bin_graph():
    """Until 0.5.11 ``get_grid`` stored the centers of a 2-D grid x fastest
    (``np.meshgrid``'s default "xy" order) while ``is_track_interior_`` is
    shaped ``(n_x, n_y)``. The rebuilt graph keeps the stored bin order: node
    ``k`` is ``place_bin_centers_[k]``, with that bin's interior flag, and
    edges join neighbouring bins only."""
    rng = np.random.default_rng(0)
    # L-shaped field, 5 x 3 bins of 10 cm.
    position = np.concatenate(
        [
            rng.uniform([0, 0], [50, 10], (500, 2)),
            rng.uniform([0, 0], [10, 30], (500, 2)),
        ]
    )
    fitted = Environment(place_bin_size=10.0).fit_place_grid(position)
    n_x, n_y = fitted.centers_shape_
    assert n_x != n_y
    interior = fitted.is_track_interior_
    assert not np.array_equal(interior, interior[::-1, ::-1])
    centers = [get_centers(edge) for edge in fitted.edges_]
    x_fastest = np.stack([axis.ravel() for axis in np.meshgrid(*centers)], axis=1)
    state = {
        **_legacy_state(fitted, drop=("distance_between_nodes_", "track_graphDD")),
        "place_bin_centers_": x_fastest,
    }

    graph = _unpickle_legacy(state).track_graphDD

    for node, data in graph.nodes(data=True):
        np.testing.assert_array_equal(data["pos"], x_fastest[node])
        x_ind, y_ind = (
            np.flatnonzero(np.isclose(axis, value))[0]
            for axis, value in zip(centers, x_fastest[node], strict=True)
        )
        assert data["is_track_interior"] == interior[x_ind, y_ind]
    bin_size = np.array([np.diff(axis).max() for axis in centers])
    for u, v in graph.edges:
        step = np.abs(x_fastest[u] - x_fastest[v]) / bin_size
        assert np.all(step <= 1 + 1e-9), (u, v)
    assert graph.number_of_edges() == fitted.track_graphDD.number_of_edges()
