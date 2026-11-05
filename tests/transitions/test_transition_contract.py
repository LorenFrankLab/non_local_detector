"""Every transition and initial-condition class keeps the interface contract.

The classes are found by scanning their modules, so a new implementer is
checked without editing this file: a class without an entry in
``CONTINUOUS_CONFIGURATIONS``, ``DISCRETE_CONFIGURATIONS`` or
``INITIAL_CONDITION_CONFIGURATIONS`` is checked as built by its default
constructor, and skipped only if that constructor needs arguments. The
entries add configurations that need arguments or are worth checking
beyond the defaults. The rules are listed in CONTRIBUTING.md ("Adding a
transition").
"""

import inspect

import networkx as nx
import numpy as np
import pytest

from non_local_detector import continuous_state_transitions as continuous
from non_local_detector import discrete_state_transitions as discrete
from non_local_detector import initial_conditions
from non_local_detector._validation import PROBABILITY_SUM_TOLERANCE
from non_local_detector.environment import Environment
from non_local_detector.model_format._builtins import builtin_types
from non_local_detector.observation_models import ObservationModel
from non_local_detector.transition_operators import (
    UnsupportedTransitionError,
    _planned,
)

pytestmark = pytest.mark.unit


def _implementers(module, method):
    """Concrete classes defined in ``module`` with the interface's ``method``."""
    return sorted(
        (
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__ == module.__name__
            and not getattr(cls, "_is_protocol", False)
            and hasattr(cls, method)
        ),
        key=lambda cls: cls.__name__,
    )


CONTINUOUS = _implementers(continuous, "make_state_transition")
DISCRETE = _implementers(discrete, "make_state_transition")
INITIAL_CONDITIONS = _implementers(initial_conditions, "make_initial_conditions")
SAVED_TYPES = {entry[1] for entry in builtin_types()}

# Instances checked per class, in place of the default-constructed one; each
# entry also names the environment layouts it applies to (None: all of them).
CONTINUOUS_CONFIGURATIONS = {
    continuous.EmpiricalMovement: [
        (continuous.EmpiricalMovement(), None),
        (continuous.EmpiricalMovement(speedup=3, is_time_reversed=True), None),
    ],
    continuous.RandomWalk: [
        (continuous.RandomWalk(), None),
        (continuous.RandomWalk(movement_var=4.0, use_manifold_distance=True), None),
        # Direction applies to grids, not to track graphs.
        (
            continuous.RandomWalk(
                movement_var=4.0, use_manifold_distance=True, direction="inward"
            ),
            ("1-D", "2-D with holes"),
        ),
    ],
}

DISCRETE_CONFIGURATIONS = {
    discrete.DiscreteNonStationaryCustom: [
        discrete.DiscreteNonStationaryCustom(
            np.array([[0.9, 0.1], [0.2, 0.8]]), formula="1 + speed"
        ),
    ],
    discrete.DiscreteNonStationaryDiagonal: [
        discrete.DiscreteNonStationaryDiagonal(
            np.array([0.9, 0.8, 0.95]), formula="1 + speed"
        ),
    ],
    discrete.DiscreteStationaryCustom: [
        discrete.DiscreteStationaryCustom(np.array([[0.9, 0.1], [0.2, 0.8]])),
    ],
    discrete.DiscreteStationaryDiagonal: [
        discrete.DiscreteStationaryDiagonal(np.array([0.9, 0.8, 0.95])),
    ],
}

INITIAL_CONDITION_CONFIGURATIONS: dict[type, list] = {}


def _needs_arguments(cls):
    """Whether ``cls()`` lacks a value for a required constructor argument."""
    return any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in inspect.signature(cls).parameters.values()
    )


def _configurations(table, cls, default):
    """The instances of ``cls`` to check: its entry, else its default instance.

    ``default`` wraps ``cls()`` as an entry of ``table``. A class without an
    entry whose constructor needs arguments is skipped.
    """
    if cls in table:
        return table[cls]
    if _needs_arguments(cls):
        pytest.skip(
            f"{cls.__name__}() needs arguments; an entry for it in "
            "this module's configuration tables would let the contract test "
            "check it"
        )
    return [default(cls())]


def _track_graph():
    graph = nx.Graph()
    for node, xy in {0: (0.0, 0.0), 1: (30.0, 0.0), 2: (30.0, 30.0)}.items():
        graph.add_node(node, pos=xy)
    graph.add_edge(0, 1, distance=30.0, edge_id=0)
    graph.add_edge(1, 2, distance=30.0, edge_id=1)
    return graph


def _layouts():
    """``{layout: (environment, position)}``; positions also fit EmpiricalMovement."""
    time = np.linspace(0, 20, 401)
    linear = (10 + 9 * np.sin(time))[:, None]
    # A ring of samples leaves the grid's corners and center off the interior.
    angle = np.linspace(0, 8 * np.pi, 2_000)
    ring = np.column_stack((15 + 12 * np.cos(angle), 15 + 12 * np.sin(angle)))
    leg = np.linspace(0, 30, 300)
    track = np.column_stack(
        (np.r_[leg, np.full_like(leg, 30.0)], np.r_[np.zeros_like(leg), leg])
    )
    return {
        "1-D": (Environment(place_bin_size=2.0).fit_place_grid(linear), linear),
        "2-D with holes": (Environment(place_bin_size=3.0).fit_place_grid(ring), ring),
        "track graph": (
            Environment(
                place_bin_size=3.0,
                track_graph=_track_graph(),
                edge_order=[(0, 1), (1, 2)],
                edge_spacing=6.0,
            ).fit_place_grid(),
            track,
        ),
    }


@pytest.fixture(scope="module")
def layouts():
    layouts = _layouts()
    assert not layouts["2-D with holes"][0].is_track_interior_.all()
    return layouts


def _dense(block):
    """The block's entries as a dense array."""
    n_rows, n_columns = block.shape
    return block.entries(np.arange(n_rows)[:, None], np.arange(n_columns)[None, :])


def _assert_rebuilds(obj):
    """``type(obj)(**vars(obj))`` rebuilds it, as Spyglass stores parameters."""
    rebuilt = type(obj)(**vars(obj))
    assert vars(rebuilt).keys() == vars(obj).keys()
    for name, value in vars(obj).items():
        np.testing.assert_equal(vars(rebuilt)[name], value)


def test_every_implementer_is_found():
    """The scan finds the package's classes, so an empty scan cannot pass."""
    assert continuous.RandomWalk in CONTINUOUS
    assert discrete.DiscreteStationaryDiagonal in DISCRETE
    assert initial_conditions.UniformInitialConditions in INITIAL_CONDITIONS


@pytest.mark.parametrize("cls", CONTINUOUS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("layout", ["1-D", "2-D with holes", "track graph"])
def test_continuous_transition_contract(cls, layout, layouts):
    """A valid block whose structured form, if any, and shape agree with it."""
    environment, position = layouts[layout]
    environments = [environment]
    context = continuous.TransitionContext(position=position)
    for transition, applies_to in _configurations(
        CONTINUOUS_CONFIGURATIONS, cls, lambda instance: (instance, None)
    ):
        if applies_to is not None and layout not in applies_to:
            continue
        assert isinstance(transition, continuous.ContinuousTransition)
        block = transition.make_state_transition(environments, context)

        assert block.shape == transition.dense_block_shape(environments)
        assert np.all(np.isfinite(block))
        assert np.all(block >= 0)
        assert np.all(block.sum(axis=1) <= 1 + PROBABILITY_SUM_TOLERANCE)
        assert transition.graph_distance_environment(environments) in (
            None,
            environment,
        )

        # Planned only when its class defines structured_block itself; a
        # class without one is planned densely, so there is nothing to compare.
        planned = _planned(transition)
        if planned is None:
            continue
        try:
            structured = planned.structured_block(environments)
        except UnsupportedTransitionError:
            continue
        np.testing.assert_allclose(_dense(structured), block, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("cls", CONTINUOUS, ids=lambda cls: cls.__name__)
def test_continuous_transition_is_its_configuration(cls):
    for transition, _ in _configurations(
        CONTINUOUS_CONFIGURATIONS, cls, lambda instance: (instance, None)
    ):
        _assert_rebuilds(transition)
    assert cls in SAVED_TYPES, f"list {cls.__name__} in model_format/_builtins.py"


@pytest.mark.parametrize("cls", DISCRETE, ids=lambda cls: cls.__name__)
def test_discrete_transition_contract(cls):
    """``make_state_transition`` returns valid rows of the declared shape."""
    n_time = 50
    covariates = {"speed": np.linspace(0, 20, n_time)}
    for transition in _configurations(
        DISCRETE_CONFIGURATIONS, cls, lambda instance: instance
    ):
        assert isinstance(transition, discrete.DiscreteTransition)
        initial = transition.make_state_transition(covariates)
        discrete._check_initial_discrete_transition(initial, transition)

        matrix = np.asarray(initial.transition_matrix)
        n_states = transition.n_states
        expected_shape = (
            (n_states, n_states)
            if transition.is_stationary
            else (n_time, n_states, n_states)
        )
        assert matrix.shape == expected_shape
        assert np.all(matrix >= 0)
        np.testing.assert_allclose(matrix.sum(axis=-1), 1.0, atol=1e-10)
        assert (initial.coefficients is None) == transition.is_stationary
        assert (initial.design_matrix is None) == transition.is_stationary
        if transition.is_stationary:
            assert transition.smoothness == 0.0

        _assert_rebuilds(transition)
    assert cls in SAVED_TYPES, f"list {cls.__name__} in model_format/_builtins.py"


@pytest.mark.parametrize("cls", INITIAL_CONDITIONS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("layout", ["1-D", "2-D with holes", "track graph"])
def test_initial_conditions_contract(cls, layout, layouts):
    """A distribution over each kind of state's interior bins."""
    environment, _ = layouts[layout]
    spatial = ObservationModel()
    local = ObservationModel(is_local=True)
    no_spike = ObservationModel(is_no_spike=True)
    for conditions in _configurations(
        INITIAL_CONDITION_CONFIGURATIONS, cls, lambda instance: instance
    ):
        assert isinstance(conditions, initial_conditions.ContinuousInitialCondition)
        for observation, is_interior in (
            (spatial, environment.is_track_interior_flat_),
            (local, np.ones(1, dtype=bool)),
            (no_spike, np.ones(1, dtype=bool)),
        ):
            values = conditions.make_initial_conditions(observation, (environment,))
            initial_conditions._check_state_initial_conditions(
                values,
                state_name="state",
                initial_conditions=conditions,
                is_interior=is_interior,
                discrete_weight=1.0,
            )

        _assert_rebuilds(conditions)
    assert cls in SAVED_TYPES, f"list {cls.__name__} in model_format/_builtins.py"
