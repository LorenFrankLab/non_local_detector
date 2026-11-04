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

from non_local_detector.environment import Environment, PlaceGrid
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.transition_operators import (
    BlockTransitionOperator,
    LazyDenseTransition,
)

# A loaded detector's pending check of its rebuilt structures.
_LOADED_ONLY = {"_rebuild_check_"}


def _assert_arrays(a, b, where):
    assert isinstance(a, jax.Array) == isinstance(b, jax.Array), where
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (where, a.dtype, b.dtype)
    assert a.shape == b.shape, (where, a.shape, b.shape)
    np.testing.assert_array_equal(a, b, err_msg=where)


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


def assert_same(a, b, where="detector"):
    """Assert ``a`` and ``b`` hold the same values, recursively."""
    if isinstance(a, (np.ndarray, jax.Array)) and not hasattr(a, "design_info"):
        assert isinstance(b, (np.ndarray, jax.Array)), where
        _assert_arrays(a, b, where)
    elif hasattr(a, "design_info") and isinstance(a, np.ndarray):  # DesignMatrix
        assert type(a) is type(b), where
        _assert_arrays(np.asarray(a), np.asarray(b), where)
        assert a.design_info.column_names == b.design_info.column_names, where
    elif isinstance(a, Environment):
        assert type(a) is type(b), where
        assert_same(a.get_params(), b.get_params(), f"{where}.config")
        assert (a._place_grid is None) == (b._place_grid is None), where
        if a._place_grid is not None:
            assert_same(a._place_grid, b._place_grid, f"{where}.place_grid_")
    elif isinstance(a, PlaceGrid):
        for field in dataclasses.fields(PlaceGrid):
            if field.name != "configuration":
                assert_same(
                    getattr(a, field.name),
                    getattr(b, field.name),
                    f"{where}.{field.name}",
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
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict), where
        assert list(a) == list(b), (where, list(a), list(b))
        for key in a:
            assert_same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b), (where, type(a), type(b))
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float), where
        assert math.isnan(b), where
    elif isinstance(a, (str, int, float, bool, np.generic)) or a is None:
        assert type(a) is type(b), (where, type(a), type(b))
        assert a == b, (where, a, b)
    else:
        assert type(a) is type(b), (where, type(a), type(b))
        state_a = dict(a.__getstate__() or {})
        state_b = dict(b.__getstate__() or {})
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
        )


def assert_same_fitted_state(original, loaded):
    """Every attribute of ``loaded`` equals that of ``original``."""
    assert type(loaded) is type(original)
    state_a = {k: v for k, v in vars(original).items() if k not in _LOADED_ONLY}
    state_b = {k: v for k, v in vars(loaded).items() if k not in _LOADED_ONLY}
    assert sorted(state_a) == sorted(state_b)
    for name in state_a:
        assert_same(state_a[name], state_b[name], name)
