"""``Environment`` configuration: ``get_params`` round trips and saved parameter rows."""

from dataclasses import fields

import networkx as nx
import numpy as np
import pytest
from sklearn.base import clone

from non_local_detector import SortedSpikesDecoder, Uniform
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError

pytestmark = pytest.mark.unit

CONFIG_NAMES = {
    "environment_name",
    "place_bin_size",
    "track_graph",
    "edge_order",
    "edge_spacing",
    "is_track_interior",
    "position_range",
    "infer_track_interior",
    "fill_holes",
    "dilate",
    "bin_count_threshold",
}
POSITION = np.random.default_rng(0).uniform(0.0, 20.0, size=(500, 2))


def _track_environment() -> Environment:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_node(2, pos=(10.0, 8.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    graph.add_edge(1, 2, distance=8.0, edge_id=1)
    return Environment(
        environment_name="corner",
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=2.0,
    )


def _grid_environment() -> Environment:
    return Environment(environment_name="box", place_bin_size=2.0, fill_holes=True)


def _fit(environment: Environment) -> Environment:
    position = None if environment.track_graph is not None else POSITION
    return environment.fit_place_grid(position)


@pytest.mark.parametrize("make", [_grid_environment, _track_environment])
@pytest.mark.parametrize("fitted", [False, True])
def test_get_params_round_trips_the_configuration(make, fitted):
    environment = make()
    if fitted:
        _fit(environment)
    params = environment.get_params()
    assert set(params) == CONFIG_NAMES

    rebuilt = Environment(**params)
    assert not rebuilt._is_fitted
    assert rebuilt.place_bin_centers_ is None
    for name, value in params.items():
        assert getattr(rebuilt, name) is value, name

    np.testing.assert_array_equal(
        _fit(rebuilt).place_bin_centers_, _fit(make()).place_bin_centers_
    )


def test_get_params_keeps_the_track_graph_a_graph():
    environment = _fit(_track_environment())
    assert isinstance(environment.get_params()["track_graph"], nx.Graph)


@pytest.mark.parametrize("is_fitted", [False, np.False_])
def test_legacy_rows_with_unset_fitted_keys_construct(is_fitted):
    """Rows saved as ``vars(environment)`` by earlier releases still load."""
    row = _grid_environment().get_params()
    row.update(
        {
            item.name: None
            for item in fields(Environment)
            if item.metadata.get("fitted") and item.name != "_is_fitted"
        }
    )
    row["_is_fitted"] = is_fitted
    environment = Environment(**row)
    assert environment.get_params() == _grid_environment().get_params()
    np.testing.assert_array_equal(
        _fit(environment).place_bin_centers_,
        _fit(_grid_environment()).place_bin_centers_,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("place_bin_centers_", np.zeros((3, 1))),
        ("is_track_interior_", np.ones(3, dtype=bool)),
        ("track_graphDD", nx.Graph()),
        ("_is_fitted", True),
    ],
)
def test_fitted_values_in_the_constructor_raise(name, value):
    with pytest.raises(ValidationError, match=name) as error:
        Environment(**{name: value})
    assert "get_params" in str(error.value)


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        Environment("box")  # type: ignore[misc]


def test_sklearn_clone_gives_an_unfitted_copy_of_the_configuration():
    environment = _fit(_track_environment())
    cloned = clone(environment)
    assert cloned is not environment
    assert not cloned._is_fitted
    assert cloned.track_graph is not environment.track_graph
    assert nx.utils.graphs_equal(cloned.track_graph, environment.track_graph)
    assert cloned.edge_order == environment.edge_order


@pytest.mark.parametrize("make", [_grid_environment, _track_environment])
@pytest.mark.parametrize("infer_track_interior", [False, True])
def test_fitting_leaves_the_configuration_unchanged(make, infer_track_interior):
    environment = make()
    before = environment.get_params()
    position = None if environment.track_graph is not None else POSITION
    environment.fit_place_grid(position, infer_track_interior=infer_track_interior)
    after = environment.get_params()
    assert after.keys() == before.keys()
    for name, value in before.items():
        assert after[name] is value, name


def test_fit_stores_the_infer_track_interior_it_used():
    environment = _grid_environment()  # infer_track_interior=True
    environment.fit_place_grid(POSITION, infer_track_interior=False)
    assert environment.infer_track_interior is True
    assert environment.infer_track_interior_ is False
    assert environment.is_track_interior_.all()


def test_detector_environments_keep_the_caller_configuration():
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    environment = Environment(place_bin_size=5.0)
    decoder = SortedSpikesDecoder(
        environments=environment,
        continuous_transition_types=[[Uniform()]],
        infer_track_interior=False,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, [np.array([0.021, 0.067, 0.121])])
    assert decoder.environments_[0].get_params() == environment.get_params()
    assert decoder.environments_[0].infer_track_interior_ is False
