"""Compare the fitted state of a detector with that of its reloaded copy.

``assert_same_fitted_state`` walks every attribute of both detectors and
requires equal values of the same types: arrays bit for bit with the same
dtype and array library, environments by configuration and grid, graphs by
nodes and edges with their attributes.
"""

import dataclasses
import math

import jax
import networkx as nx
import numpy as np
import pandas as pd

from non_local_detector._formula_grammar import LoadedFormula
from non_local_detector.environment import Environment, PlaceGrid
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.transition_operators import (
    BlockTransitionOperator,
    LazyDenseTransition,
)

# A loaded detector's pending check of its rebuilt structures.
_LOADED_ONLY = {"_rebuild_check_"}


def _assert_arrays(a, b, where, tolerance=None):
    """Equal type, dtype and shape; values bit for bit, or within ``tolerance``
    (``assert_allclose`` keywords) when one is given."""
    assert isinstance(a, jax.Array) == isinstance(b, jax.Array), where
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (where, a.dtype, b.dtype)
    assert a.shape == b.shape, (where, a.shape, b.shape)
    if tolerance is None:
        np.testing.assert_array_equal(a, b, err_msg=where)
    else:
        np.testing.assert_allclose(a, b, err_msg=where, **tolerance)


def _assert_graphs(a, b, where):
    assert type(a) is type(b), where
    assert list(a.nodes) == list(b.nodes), where
    assert list(a.edges) == list(b.edges), where
    for node in a.nodes:
        assert_same(
            dict(a.nodes[node]), dict(b.nodes[node]), f"{where}.nodes[{node!r}]"
        )
    for edge in a.edges:
        assert_same(
            dict(a.edges[edge]), dict(b.edges[edge]), f"{where}.edges[{edge!r}]"
        )
    assert_same(dict(a.graph), dict(b.graph), f"{where}.graph")


def _state(obj):
    """What pickle saves of ``obj``: its class's own ``__getstate__`` if it
    defines one, else its ``__dict__``. (``object.__getstate__``, which gives
    the same on Python 3.11+, does not exist on 3.10.)"""
    own = getattr(type(obj), "__getstate__", None)
    if own is not None and own is not getattr(object, "__getstate__", None):
        return dict(obj.__getstate__() or {})
    return dict(getattr(obj, "__dict__", {}))


def _assert_distance_dicts(a, b, where):
    """Track-graph distances by content: the connected pairs and their values.

    The key order of these dicts is the order a shortest-path solver
    finalized the nodes in, and a distance's scalar type (``int`` 0 or a
    NumPy or Python float) follows its arithmetic. Neither is state: every
    reader looks a distance up by its pair of nodes.
    """
    assert isinstance(b, dict), where
    assert a.keys() == b.keys(), where
    for node, distances in a.items():
        assert distances == b[node], (where, node)


def assert_same(a, b, where="detector", tolerances=None):
    """Assert ``a`` and ``b`` hold the same values, recursively.

    ``tolerances`` maps the ``where`` path of an array to the
    ``assert_allclose`` keywords it is compared with; every other array is
    compared bit for bit. By default every array is.
    """
    tolerances = {} if tolerances is None else tolerances
    if isinstance(a, (np.ndarray, jax.Array)) and not hasattr(a, "design_info"):
        assert isinstance(b, (np.ndarray, jax.Array)), where
        _assert_arrays(a, b, where, tolerances.get(where))
    elif hasattr(a, "design_info") and isinstance(a, np.ndarray):  # DesignMatrix
        assert type(a) is type(b), where
        _assert_arrays(np.asarray(a), np.asarray(b), where)
        assert a.design_info.column_names == b.design_info.column_names, where
    elif isinstance(a, Environment):
        assert type(a) is type(b), where
        assert_same(a.get_params(), b.get_params(), f"{where}.config", tolerances)
        assert (a._place_grid is None) == (b._place_grid is None), where
        if a._place_grid is not None:
            assert_same(
                a._place_grid, b._place_grid, f"{where}.place_grid_", tolerances
            )
    elif isinstance(a, PlaceGrid):
        for field in dataclasses.fields(PlaceGrid):
            if field.name == "distance_between_nodes" and isinstance(
                a.distance_between_nodes, dict
            ):
                _assert_distance_dicts(
                    a.distance_between_nodes,
                    b.distance_between_nodes,
                    f"{where}.{field.name}",
                )
            elif field.name != "configuration":
                assert_same(
                    getattr(a, field.name),
                    getattr(b, field.name),
                    f"{where}.{field.name}",
                    tolerances,
                )
    elif isinstance(a, nx.Graph):
        _assert_graphs(a, b, where)
    elif isinstance(a, LazyGraphDistances):
        assert type(b) is LazyGraphDistances, where
        _assert_arrays(a.graph.toarray(), b.graph.toarray(), where)
        assert (a.directed, a.max_dense_bytes) == (b.directed, b.max_dense_bytes), where
    elif isinstance(a, LazyDenseTransition):
        assert type(b) is LazyDenseTransition, where
        _assert_arrays(np.asarray(a), np.asarray(b), where)
    elif isinstance(a, BlockTransitionOperator):
        assert type(b) is BlockTransitionOperator, where
        _assert_arrays(
            np.asarray(LazyDenseTransition(a)),
            np.asarray(LazyDenseTransition(b)),
            where,
        )
    elif isinstance(a, pd.DataFrame):
        # Legacy pickles hold object column labels where pandas 3 infers str.
        pd.testing.assert_frame_equal(a, b, check_column_type=False)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict), where
        assert list(a) == list(b), (where, list(a), list(b))
        for key in a:
            assert_same(a[key], b[key], f"{where}[{key!r}]", tolerances)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b), (where, type(a), type(b))
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_same(x, y, f"{where}[{i}]", tolerances)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float), where
        assert math.isnan(b), where
    elif isinstance(a, str) and isinstance(b, LoadedFormula):
        # A transition formula loaded without trust_formulas is marked as
        # read from a file.
        assert type(a) is str, (where, type(a))
        assert a == b, (where, a, b)
    elif isinstance(a, (str, int, float, bool, np.generic)) or a is None:
        assert type(a) is type(b), (where, type(a), type(b))
        assert a == b, (where, a, b)
    else:
        assert type(a) is type(b), (where, type(a), type(b))
        state_a, state_b = _state(a), _state(b)
        # A TensorSplineBasis caches its rebuilt patsy design.
        state_a.pop("design_info", None)
        state_b.pop("design_info", None)
        # Attribute order is not part of an object's state, and an attribute
        # left at its class default equals one set to that default.
        names = sorted(set(state_a) | set(state_b))
        assert_same(
            {name: state_a.get(name, getattr(a, name)) for name in names},
            {name: state_b.get(name, getattr(b, name)) for name in names},
            f"{where}<{type(a).__name__}>",
            tolerances,
        )


def assert_same_fitted_state(original, loaded, tolerances=None):
    """Every attribute of ``loaded`` equals that of ``original``.

    Arrays are compared bit for bit, except those whose path is a key of
    ``tolerances`` (see ``assert_same``).
    """
    assert type(loaded) is type(original)
    state_a = {k: v for k, v in vars(original).items() if k not in _LOADED_ONLY}
    state_b = {k: v for k, v in vars(loaded).items() if k not in _LOADED_ONLY}
    assert sorted(state_a) == sorted(state_b)
    for name in state_a:
        assert_same(state_a[name], state_b[name], name, tolerances)
