"""Defines spatial environments using discrete grids and graph representations.

This module provides the `Environment` class and associated helper functions to
represent spatial environments commonly used in neuroscience experiments, such
as open fields or linear tracks. The core idea is to discretize the continuous
space into a grid of bins and potentially represent the connectivity or topology
of the valid space using a graph.

The module supports two main types of environments:

1.  **N-Dimensional Environments (e.g., Open Field, W-Track):**
    - Discretizes the space into a regular N-D grid based on specified bin sizes
      or position data ranges.
    - Can automatically infer the "track interior" (the portion of the grid
      actually occupied by the animal) from position data using histogramming
      and optional morphological operations (filling holes, dilation).
    - Constructs a `networkx` graph (`track_graphDD`) where nodes represent
      the centers of *interior* bins, and edges connect adjacent interior bins.
      This graph captures the connectivity of the valid space.
    - Can compute shortest-path distances between all pairs of interior bins
      on this graph (`distance_between_nodes_`).
    - Provides methods to find the bin index for a given position (`get_bin_ind`),
      calculate manifold distances between positions (`get_manifold_distances`),
      and determine movement direction relative to the track center (`get_direction`).

2.  **1-Dimensional Environments (Linear Tracks, W-Tracks):**
    - Requires a `networkx.Graph` (`track_graph`) defining the track's topology
      (nodes, edges, and their positions) along with edge ordering and spacing.
    - Linearizes the track based on the provided graph structure.
    - Creates bins along this linearized track.
    - Generates an augmented graph (`track_graph_with_bin_centers_edges_`) where
      both the original track nodes and the newly created bin centers/edges
      are represented as nodes.
    - Computes shortest-path distances between all nodes in this augmented graph
      (`distance_between_nodes_`).
    - Stores detailed information about original nodes, bin edges, and bin
      centers in pandas DataFrames (`original_nodes_df_`, etc.).

The central component is the `Environment` dataclass, which holds the parameters
defining the environment. The primary method `fit_place_grid` performs the
discretization and graph construction from those parameters and optional
position data, and stores the result (grid details, graphs, distances) as one
immutable `PlaceGrid`, read through attributes such as `place_bin_centers_`.

Helper functions are provided for tasks like grid generation (`get_grid`),
track interior inference (`get_track_interior`), graph construction
(`make_nD_track_graph_from_environment`, `get_track_grid`), distance calculation,
and visualization (`plot_grid`). `Environment.get_params` returns an
environment's configuration, from which `Environment(**params)` rebuilds it
unfitted; `save_environment` and `load_environment` pickle a fitted one.
"""

import contextlib
import copy
import dataclasses
import functools
import hashlib
import pickle
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, overload

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import jax
from scipy import ndimage
from scipy.interpolate import interp1d
from track_linearization import (  # type: ignore[import-untyped]
    get_linearized_position,
    plot_graph_as_1D,
)

from non_local_detector import _validation as val
from non_local_detector._warnings import user_stacklevel
from non_local_detector.exceptions import (
    CompatibilityWarning,
    ConfigurationError,
    DataQualityWarning,
    NotFittedError,
    ValidationError,
)
from non_local_detector.graph_distances import (
    DEFAULT_MAX_DENSE_DISTANCE_BYTES,
    LazyGraphDistances,
)

__all__ = [
    "Environment",
    "PlaceGrid",
]


def get_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Return the center of each bin given the bin edges.

    Parameters
    ----------
    bin_edges : np.ndarray, shape (n_edges,)
        Monotonically increasing bin edges.

    Returns
    -------
    bin_centers : np.ndarray, shape (n_edges - 1,)
        Midpoint of each pair of adjacent edges.
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def find_environment_by_name(
    environments: Sequence["Environment"],
    name: str,
) -> "Environment":
    """Look up an environment in a sequence by its ``environment_name``.

    Parameters
    ----------
    environments : sequence of Environment
        Environments to search.
    name : str
        The ``environment_name`` to match.

    Returns
    -------
    Environment
        The first environment in ``environments`` whose ``environment_name``
        equals ``name``.

    Raises
    ------
    ValidationError
        If no environment with the given name is present. The message lists
        the available names.
    """
    for env in environments:
        if env.environment_name == name:
            return env
    available = [env.environment_name for env in environments]
    msg = f"No environment named {name!r}"
    raise ValidationError(
        msg,
        got=f"environment names {available}",
        hint=(
            "Give each transition and observation model the environment_name "
            "of one of the detector's environments."
        ),
    )


# Bytes of cached track linearizations kept per environment. EM linearizes the
# same position samples and spike positions in every E-step and encoding
# M-step; a 23.6 min recording needs about 30 MB.
_LINEAR_POSITION_CACHE_BYTES = 256 * 1024**2


# Configuration fields the fitted grid depends on. ``environment_name`` and
# ``infer_track_interior`` (overridden by the ``fit_place_grid`` argument) are
# not among them.
_GRID_CONFIGURATION = (
    "place_bin_size",
    "track_graph",
    "edge_order",
    "edge_spacing",
    "is_track_interior",
    "position_range",
    "fill_holes",
    "dilate",
    "bin_count_threshold",
)

# The fitted attributes earlier releases stored on the environment itself (as
# dataclass fields), mapped to the ``PlaceGrid`` field that now holds each.
_LEGACY_FITTED_ATTRIBUTES = {
    "edges_": "edges",
    "place_bin_edges_": "place_bin_edges",
    "place_bin_centers_": "place_bin_centers",
    "centers_shape_": "centers_shape",
    "is_track_interior_": "is_track_interior",
    "infer_track_interior_": "infer_track_interior",
    "is_track_boundary_": "is_track_boundary",
    "track_graphDD": "track_graphDD",
    "distance_between_nodes_": "distance_between_nodes",
    "track_graph_with_bin_centers_edges_": "track_graph_with_bin_centers_edges",
    "original_nodes_df_": "original_nodes_df",
    "place_bin_edges_nodes_df_": "place_bin_edges_nodes_df",
    "place_bin_centers_nodes_df_": "place_bin_centers_nodes_df",
    "nodes_df_": "nodes_df",
}


def _configuration_snapshot(value: Any) -> Any:
    """Return a copy of a configuration value to compare later values with.

    The track graph is kept by reference (it is compared by identity); other
    values are deep-copied, so an in-place edit of a list or array the
    environment holds shows as a change.
    """
    return value if isinstance(value, nx.Graph) else copy.deepcopy(value)


def _same_configuration(current: Any, fitted: Any) -> bool:
    """Whether a configuration value equals the one a grid was fitted with.

    Values are compared by value: sequences element by element whatever their
    container (a list equals the tuple with the same items), arrays with
    ``np.array_equal``. Graphs are compared by identity, so an in-place edit
    of the track graph is not detected.
    """
    if current is fitted:
        return True
    if isinstance(current, nx.Graph) or isinstance(fitted, nx.Graph):
        return False
    if isinstance(current, np.ndarray) or isinstance(fitted, np.ndarray):
        return bool(
            np.shape(current) == np.shape(fitted) and np.array_equal(current, fitted)
        )
    if isinstance(current, (list, tuple)) and isinstance(fitted, (list, tuple)):
        return len(current) == len(fitted) and all(
            _same_configuration(a, b) for a, b in zip(current, fitted, strict=True)
        )
    try:
        return bool(current == fitted)
    except (TypeError, ValueError):
        return False


def _graph_distances(
    track_graph: nx.Graph,
    *,
    compute_all_pairs_distances: bool = True,
    max_dense_distance_bytes: int = DEFAULT_MAX_DENSE_DISTANCE_BYTES,
) -> np.ndarray | LazyGraphDistances:
    """Shortest-path distances between the nodes of an N-D grid's graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        Graph of the grid's bin centers (``_make_nD_track_graph``).
    compute_all_pairs_distances : bool, optional
        If True, build the all-pairs distance matrix; otherwise keep an exact
        sparse query view. See :meth:`Environment.fit_place_grid`.
    max_dense_distance_bytes : int, optional
        See :meth:`Environment.fit_place_grid`.

    Returns
    -------
    distance_between_nodes : np.ndarray, shape (n_bins, n_bins), or LazyGraphDistances
    """
    if compute_all_pairs_distances:
        # Eager dense matrix, unbounded: max_dense_distance_bytes applies only
        # to the lazy path.
        n_nodes = len(track_graph)
        return LazyGraphDistances.from_graph(track_graph).to_dense(
            max_bytes=8 * n_nodes**2
        )
    return LazyGraphDistances.from_graph(
        track_graph, max_dense_bytes=max_dense_distance_bytes
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class PlaceGrid:
    """The fitted geometry of an :class:`Environment`.

    :meth:`Environment.fit_place_grid` builds one completely and assigns it in
    a single step; the environment's fitted attributes (``place_bin_centers_``
    and so on) read it. The fields cannot be reassigned. The arrays are shared
    with the environment's attributes, so treat them as read-only.

    Attributes
    ----------
    edges : tuple of np.ndarray
        Bin edges of each dimension.
    place_bin_edges : np.ndarray, shape (n_bins + 1, n_dims) or (n_total_bins + n_edges, n_dims)
        Edges of the place bins (linearized or N-D).
    place_bin_centers : np.ndarray, shape (n_bins, n_dims)
        Center coordinates of each place bin.
    centers_shape : tuple[int, ...]
        Number of bins in each dimension.
    is_track_interior : np.ndarray of bool, shape ``centers_shape``
        Which bins are part of the track interior.
    distance_between_nodes : dict, np.ndarray or LazyGraphDistances
        Shortest-path distances between the nodes of the track graph: a
        dict of dicts on a track graph, otherwise an
        ``(n_bins, n_bins)`` array or a deferred view of one.
    configuration : Mapping[str, Any]
        Read-only copy of the configuration fields the grid depends on, as
        they were when it was fitted; the environment compares its fields
        with them.
    infer_track_interior : bool or None
        The ``infer_track_interior`` argument of the fit. None on a track
        graph.
    is_track_boundary : np.ndarray of bool or None
        Boundary bins of an N-D grid with more than one dimension.
    track_graphDD : networkx.Graph or None
        Graph of the bin centers of an N-D grid.
    track_graph_with_bin_centers_edges : networkx.Graph or None
        The track graph with bin centers and edges as nodes (track graph only).
    original_nodes_df, place_bin_edges_nodes_df, place_bin_centers_nodes_df, nodes_df : pd.DataFrame or None
        The track graph's original, bin-edge, bin-center and all nodes
        (track graph only).
    """

    edges: tuple[np.ndarray, ...]
    place_bin_edges: np.ndarray
    place_bin_centers: np.ndarray
    centers_shape: tuple[int, ...]
    is_track_interior: np.ndarray
    distance_between_nodes: (
        dict[int, dict[int, float]] | np.ndarray | LazyGraphDistances
    )
    configuration: Mapping[str, Any]
    infer_track_interior: bool | None = None
    is_track_boundary: np.ndarray | None = None
    track_graphDD: nx.Graph | None = None
    track_graph_with_bin_centers_edges: nx.Graph | None = None
    original_nodes_df: pd.DataFrame | None = None
    place_bin_edges_nodes_df: pd.DataFrame | None = None
    place_bin_centers_nodes_df: pd.DataFrame | None = None
    nodes_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Make ``configuration`` a read-only view of its own copy."""
        object.__setattr__(
            self, "configuration", MappingProxyType(dict(self.configuration))
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the read-only configuration as a dict."""
        return {**self.__dict__, "configuration": dict(self.configuration)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled grid, with a read-only configuration."""
        self.__dict__.update(state)
        self.__post_init__()


def _legacy_nd_track_graph(
    place_bin_centers: np.ndarray,
    is_track_interior: np.ndarray,
    edges: Sequence[np.ndarray],
) -> nx.Graph:
    """Return the bin-center graph of an N-D grid stored by an earlier release.

    Until 0.5.11 ``get_grid`` built the centers with ``np.meshgrid``'s default
    "xy" indexing, so they vary along the first axis fastest while
    ``is_track_interior`` keeps the ``(n_x, n_y, ...)`` shape. The graph is
    built in the stored bin order, so node ``k`` is ``place_bin_centers[k]``
    with its own interior flag, as the stored encoding models and results
    index the bins.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_bins, n_dims)
        Bin centers in the stored bin order.
    is_track_interior : np.ndarray of bool, shape (n_x, n_y, ...)
        Interior flag of each bin, in the grid's shape.
    edges : sequence of np.ndarray
        The stored bin edges of each dimension.

    Returns
    -------
    track_graph : networkx.Graph
    """
    if len(edges) > 1:
        axes = [get_centers(np.asarray(edge)) for edge in edges]
        x_fastest = np.stack([axis.ravel() for axis in np.meshgrid(*axes)], axis=1)
        if place_bin_centers.shape == x_fastest.shape and np.allclose(
            place_bin_centers, x_fastest
        ):
            is_track_interior = np.swapaxes(is_track_interior, 0, 1)
    return _make_nD_track_graph(
        place_bin_centers, is_track_interior, is_track_interior.shape
    )


def _legacy_place_grid(
    environment: "Environment", fitted: dict[str, Any]
) -> PlaceGrid | None:
    """Return the grid of an environment pickled when fitted attributes were fields.

    Parameters
    ----------
    environment : Environment
        The unpickled environment, with its configuration restored.
    fitted : dict[str, Any]
        Its stored fitted attributes, keyed by ``PlaceGrid`` field name.

    Returns
    -------
    PlaceGrid or None
        None if the environment was not fitted.
    """
    if fitted.get("place_bin_centers") is None:
        return None
    fitted = {name: value for name, value in fitted.items() if value is not None}
    centers_shape = tuple(fitted["centers_shape"])
    # Earlier releases treated a missing mask as all bins interior.
    is_track_interior = fitted.get(
        "is_track_interior", np.ones(centers_shape, dtype=bool)
    )
    if environment.track_graph is None:
        # Releases before 0.5.13 stored no graph or distances for an N-D
        # grid; rebuild them as fit_place_grid does. The distances are
        # deferred, so unpickling a large grid does not compute all pairs.
        if "track_graphDD" not in fitted:
            fitted["track_graphDD"] = _legacy_nd_track_graph(
                fitted["place_bin_centers"], is_track_interior, fitted["edges"]
            )
        if "distance_between_nodes" not in fitted:
            fitted["distance_between_nodes"] = _graph_distances(
                fitted["track_graphDD"], compute_all_pairs_distances=False
            )
        # Released fits stored the infer_track_interior they used in the
        # configuration field.
        fitted.setdefault("infer_track_interior", environment.infer_track_interior)
    elif "distance_between_nodes" not in fitted:
        assert environment.edge_order is not None  # required with a track graph
        assert environment.edge_spacing is not None
        fitted["distance_between_nodes"] = get_track_grid(
            environment.track_graph,
            environment.edge_order,
            environment.edge_spacing,
            cast("float", environment.place_bin_size),
        )[3]
    return PlaceGrid(
        **{
            **fitted,
            "edges": tuple(fitted["edges"]),
            "centers_shape": centers_shape,
            "is_track_interior": is_track_interior,
            "configuration": environment._grid_configuration(),
        }
    )


@dataclass(eq=False, kw_only=True)
class Environment:
    """Represent the spatial environment with a discrete grid.

    Parameters
    ----------
    environment_name : str, optional
        Identifier for the environment. Defaults to "".
    place_bin_size : float or tuple[float], optional
        Approximate size of the position bins in each dimension. Defaults to 2.0.
    track_graph : networkx.Graph, optional
        Graph representing the 1D spatial topology. If provided, 1D methods are used.
    edge_order : tuple of 2-tuples, optional
        Required if `track_graph` is provided. The order of the edges in 1D space.
    edge_spacing : float or list, optional
        Required if `track_graph` is provided. Spacing between edges. Defaults to 0.
    is_track_interior : np.ndarray or None, optional
        Boolean array defining valid track areas. If None and `infer_track_interior`
        is True, it will be inferred from position data.
    position_range : Sequence[tuple[float, float]], optional
        Outer bin edges for each dimension [(min_dim1, max_dim1), ...].
        If None, range is determined from position data.
    infer_track_interior : bool, optional
        If True and `is_track_interior` is None, infer track geometry from position data.
        Defaults to True. Ignored if `track_graph` is provided. The
        ``infer_track_interior`` argument of :meth:`fit_place_grid` (True by
        default) takes precedence; the value a fit used is stored as
        ``infer_track_interior_``.
    fill_holes : bool, optional
        Fill holes in the inferred track interior. Defaults to False. Ignored if `track_graph` is provided.
    dilate : bool, optional
        Inflate the inferred track interior. Defaults to False. Ignored if `track_graph` is provided.
    bin_count_threshold : int, optional
        Minimum number of samples in a bin to be considered part of the track interior.
        Defaults to 0. Ignored if `track_graph` is provided.

    Attributes
    ----------
    place_grid_ : PlaceGrid
        The fitted geometry; the attributes below read it. Every fitted
        attribute raises ``NotFittedError`` before :meth:`fit_place_grid`, and
        after a configuration field the grid depends on (all but
        ``environment_name`` and ``infer_track_interior``) changes value,
        until :meth:`fit_place_grid` runs again. Values are compared by value,
        including in-place edits of lists and arrays; the track graph is
        compared by identity.
    edges_ : tuple[np.ndarray, ...]
        Bin edges for each dimension.
    place_bin_edges_ : np.ndarray, shape (n_bins + 1, n_dims) or (n_total_bins + n_edges, n_dims)
        Edges of the place bins (linearized or N-D).
    place_bin_centers_ : np.ndarray, shape (n_bins, n_dims)
        Center coordinates of each place bin.
    centers_shape_ : tuple[int, ...]
        Shape of the grid in terms of bins per dimension.
    is_track_interior_ : np.ndarray, shape (n_bins,) or (n_bins_dim1, n_bins_dim2, ...)
        Boolean array indicating which bins are part of the track interior.
    infer_track_interior_ : bool or None
        The ``infer_track_interior`` argument the last :meth:`fit_place_grid`
        used. None for track-graph environments.
    is_track_boundary_ : np.ndarray or None
        Boolean array indicating boundary bins (only for N-D environments).
    track_graphDD : networkx.Graph or None
        Graph representation where nodes are bin centers (only for N-D environments).
    distance_between_nodes_ : dict[int, dict[int, float]] or np.ndarray
        Shortest path distances between nodes on the track graph (1D or N-D).
    track_graph_with_bin_centers_edges_ : nx.Graph or None
        Track graph with bin centers and edges added as nodes (only for 1D).
    original_nodes_df_ : pd.DataFrame or None
        Info about original track graph nodes (only for 1D).
    place_bin_edges_nodes_df_ : pd.DataFrame or None
        Info about nodes representing bin edges (only for 1D).
    place_bin_centers_nodes_df_ : pd.DataFrame or None
        Info about nodes representing bin centers (only for 1D).
    nodes_df_ : pd.DataFrame or None
        Combined info about all nodes in the augmented graph (only for 1D).
    _cache : dict
        Derived data computed from the fitted grid on first use, such as the
        dense bin-distance matrix, track linearizations and likelihood-specific
        graphs and eigenbases. Read and written through
        :meth:`get_cached` and :meth:`set_cached`; cleared by
        :meth:`fit_place_grid`. Entries stored with ``transient=True`` are not
        pickled.

    Notes
    -----
    All arguments are keyword-only, and are the configuration only: the
    fitted attributes are set by :meth:`fit_place_grid`. (Unset fitted
    attributes, as in ``vars(environment)`` rows saved by earlier releases,
    are dropped with a ``DeprecationWarning``.) :meth:`get_params`
    returns the configuration, so ``Environment(**environment.get_params())``
    rebuilds an unfitted environment with the same configuration. Environments
    pickled by earlier releases, which stored the fitted attributes
    themselves, unpickle into the same form.
    """

    environment_name: str = ""
    place_bin_size: float | tuple[float, ...] = 2.0
    track_graph: nx.Graph | None = None
    edge_order: list[tuple[Any, ...]] | None = None
    edge_spacing: float | list[float] | None = 0.0
    is_track_interior: np.ndarray | None = None
    position_range: Sequence[tuple[float, float]] | None = None
    infer_track_interior: bool = True
    fill_holes: bool = False
    dilate: bool = False
    bin_count_threshold: int = 0

    # The fitted geometry, set whole by fit_place_grid; see ``place_grid_``.
    _place_grid: PlaceGrid | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Derived-data cache; see ``get_cached``.
    _cache: dict[str, tuple[Any, bool]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ``Environment`` uses identity-based equality and hashing: the dataclass is
    # declared ``eq=False`` (above), so both ``__eq__`` and ``__hash__`` stay
    # inherited from ``object`` and no explicit ``__hash__`` is set here.
    # Value-based equality is not viable — the fitted numpy / networkx
    # attributes are not value-comparable (a dataclass ``__eq__`` would raise
    # ``ValueError`` on two fitted environments' array fields) and have no
    # meaningful value-hash. Identity semantics keep ``__eq__`` / ``__hash__``
    # consistent (Python's ``a == b implies hash(a) == hash(b)`` holds), so
    # instances are safe as dict keys / set members; environments are addressed
    # by name via ``find_environment_by_name``, never by value-equality. Do not
    # switch to ``eq=True`` without also defining ``__hash__`` explicitly —
    # leaving it unset under ``eq=True`` makes instances unhashable (a loud
    # failure), which is preferable to silently re-introducing a value-equality
    # / identity-hash mismatch.

    def __post_init__(self) -> None:
        """Validate Environment parameters after initialization."""
        # Validate place_bin_size
        if isinstance(self.place_bin_size, tuple):
            for i, size in enumerate(self.place_bin_size):
                val.ensure_positive_scalar(
                    size,
                    f"place_bin_size[{i}]",
                    minimum=0.0,
                    strict=True,
                )
        else:
            val.ensure_positive_scalar(
                self.place_bin_size,
                "place_bin_size",
                minimum=0.0,
                strict=True,
            )

        # Validate bin_count_threshold
        val.ensure_positive_scalar(
            self.bin_count_threshold,
            "bin_count_threshold",
            minimum=0.0,
            strict=False,
        )

        # If track_graph is provided, validate it and its dependencies
        if self.track_graph is not None:
            self._validate_track_graph()

        # Validate position_range if provided
        if self.position_range is not None:
            self._validate_position_range()

        # Validate is_track_interior if provided
        if self.is_track_interior is not None:
            val.ensure_ndarray(self.is_track_interior, "is_track_interior")
            # Note: We can't validate shape until we know the grid dimensions

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002, FBT001, FBT002 -- scikit-learn signature
        """Return the constructor configuration of this environment.

        Only the configuration fields are returned, never the fitted
        attributes or the cache, so ``Environment(**environment.get_params())``
        builds an equivalent unfitted environment from a fitted or unfitted one.
        The values are the environment's own objects, not copies (as in
        scikit-learn); ``track_graph`` stays a ``networkx.Graph``.

        Parameters
        ----------
        deep : bool, optional
            Ignored: an environment holds no nested estimators. Accepted so
            ``sklearn.base.clone`` can rebuild an environment from its
            configuration. By default True.

        Returns
        -------
        params : dict[str, Any]
            Constructor keyword arguments, keyed by field name.
        """
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the environment without its transient cache entries.

        Entries cached with ``transient=True`` (for example device-resident JAX
        arrays keyed by a ``Device``) cannot be restored in another process;
        they are rebuilt on first use after unpickling.
        """
        state = self.__dict__.copy()
        state["_cache"] = {
            key: entry for key, entry in self._cache.items() if not entry[1]
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled environment.

        Pickles made without a cache get an empty one. Pickles made when the
        fitted attributes (``place_bin_centers_``, ...) were stored on the
        environment get a ``PlaceGrid`` holding them, fitted with the stored
        configuration.
        """
        state = dict(state)
        legacy = {
            grid_name: state.pop(name)
            for name, grid_name in _LEGACY_FITTED_ATTRIBUTES.items()
            if name in state
        }
        state.pop("_is_fitted", None)
        self.__dict__.update(state)
        self.__dict__.setdefault("_cache", {})
        if "_place_grid" not in state:
            self._place_grid = _legacy_place_grid(self, legacy)

    def _grid_configuration(self) -> dict[str, Any]:
        """Snapshots of the configuration fields the fitted grid depends on."""
        return {
            name: _configuration_snapshot(getattr(self, name))
            for name in _GRID_CONFIGURATION
        }

    def _changed_configuration(self, grid: PlaceGrid) -> list[str]:
        """Return the configuration fields that differ from those ``grid`` was fit with."""
        return [
            name
            for name, value in grid.configuration.items()
            if not _same_configuration(getattr(self, name), value)
        ]

    @property
    def place_grid_(self) -> PlaceGrid:
        """The fitted geometry (see the class docstring).

        Returns
        -------
        PlaceGrid

        Raises
        ------
        NotFittedError
            Before :meth:`fit_place_grid`, or after a configuration field the
            grid depends on changed.
        """
        grid = self._place_grid
        if grid is None:
            msg = "Environment has not been fitted"
            raise NotFittedError(
                msg,
                hint="Call environment.fit_place_grid(position) first.",
            )
        changed = self._changed_configuration(grid)
        if changed:
            msg = "Environment configuration changed since fit_place_grid"
            raise NotFittedError(
                msg,
                got=f"new values for {changed}",
                hint=(
                    "Call fit_place_grid again to fit the grid of the new "
                    "configuration."
                ),
            )
        return grid

    @property
    def _is_fitted(self) -> bool:
        """Whether the environment has a grid fitted with its configuration."""
        try:
            _ = self.place_grid_
        except NotFittedError:
            return False
        return True

    @property
    def edges_(self) -> tuple[np.ndarray, ...]:
        """Bin edges of each dimension (``PlaceGrid.edges``)."""
        return self.place_grid_.edges

    @property
    def place_bin_edges_(self) -> np.ndarray:
        """Edges of the place bins (``PlaceGrid.place_bin_edges``)."""
        return self.place_grid_.place_bin_edges

    @property
    def place_bin_centers_(self) -> np.ndarray:
        """Center of each place bin, shape (n_bins, n_dims)."""
        return self.place_grid_.place_bin_centers

    @property
    def centers_shape_(self) -> tuple[int, ...]:
        """Number of bins in each dimension."""
        return self.place_grid_.centers_shape

    @property
    def is_track_interior_(self) -> np.ndarray:
        """Which bins are part of the track interior, shape ``centers_shape_``."""
        return self.place_grid_.is_track_interior

    @property
    def infer_track_interior_(self) -> bool | None:
        """The ``infer_track_interior`` argument of the fit; None on a track graph."""
        return self.place_grid_.infer_track_interior

    @property
    def is_track_boundary_(self) -> np.ndarray | None:
        """Boundary bins of an N-D grid with more than one dimension, else None."""
        return self.place_grid_.is_track_boundary

    @property
    def track_graphDD(self) -> nx.Graph | None:
        """Graph of the bin centers of an N-D grid; None on a track graph."""
        return self.place_grid_.track_graphDD

    @property
    def distance_between_nodes_(
        self,
    ) -> dict[int, dict[int, float]] | np.ndarray | LazyGraphDistances:
        """Shortest-path distances between track-graph nodes (see ``PlaceGrid``)."""
        return self.place_grid_.distance_between_nodes

    @property
    def track_graph_with_bin_centers_edges_(self) -> nx.Graph | None:
        """The track graph with bin centers and edges as nodes, or None."""
        return self.place_grid_.track_graph_with_bin_centers_edges

    @property
    def original_nodes_df_(self) -> pd.DataFrame | None:
        """The track graph's original nodes, or None without a track graph."""
        return self.place_grid_.original_nodes_df

    @property
    def place_bin_edges_nodes_df_(self) -> pd.DataFrame | None:
        """The track graph's bin-edge nodes, or None without a track graph."""
        return self.place_grid_.place_bin_edges_nodes_df

    @property
    def place_bin_centers_nodes_df_(self) -> pd.DataFrame | None:
        """The track graph's bin-center nodes, or None without a track graph."""
        return self.place_grid_.place_bin_centers_nodes_df

    @property
    def nodes_df_(self) -> pd.DataFrame | None:
        """All nodes of the augmented track graph, or None without a track graph."""
        return self.place_grid_.nodes_df

    def _replace_place_grid(self, **changes: Any) -> "Environment":
        """Return a shallow copy whose grid has some fields replaced.

        Parameters
        ----------
        **changes : Any
            ``PlaceGrid`` fields and their new values.

        Returns
        -------
        Environment
            Shares the configuration and the other grid fields with this
            environment, and a copy of its persistent cache entries.
        """
        view = copy.copy(self)
        view._place_grid = dataclasses.replace(self.place_grid_, **changes)
        return view

    def get_cached(self, key: str) -> Any:
        """Return the derived data cached under ``key``, or None.

        Parameters
        ----------
        key : str
            Cache key chosen by the computation that stored the value.

        Returns
        -------
        value : Any
            The cached value, or None if nothing is cached under ``key``.
            Entries describe the fitted grid, so after a configuration field
            the grid depends on changes the cache is cleared and None
            returned.
        """
        grid = self._place_grid
        if grid is not None and self._changed_configuration(grid):
            self.clear_cache()
            return None
        entry = self._cache.get(key)
        return None if entry is None else entry[0]

    def set_cached(self, key: str, value: Any, *, transient: bool = False) -> Any:
        """Cache derived data under ``key`` until the grid is refit.

        The value must be a function of this environment's fitted grid only:
        :meth:`fit_place_grid` clears every entry.

        Parameters
        ----------
        key : str
            Cache key, unique to the computation that stores the value.
        value : Any
            The derived data.
        transient : bool, optional
            If True, the entry is not pickled (for values that cannot be
            restored in another process). By default False.

        Returns
        -------
        value : Any
            ``value``, so a cache miss can store and use it in one expression.
        """
        self._cache[key] = (value, transient)
        return value

    def clear_cache(self) -> None:
        """Drop all cached derived data."""
        self._cache = {}

    def _linear_position(self, points: "np.ndarray | jax.Array") -> np.ndarray:
        """Linear position of points on the track graph, cached by value.

        Equal point values give the stored result without projecting them onto
        the graph again. Results are copied, so callers own them. The cache is
        bounded by ``_LINEAR_POSITION_CACHE_BYTES`` (the latest result is always
        kept), cleared by ``fit_place_grid`` and not pickled.

        Parameters
        ----------
        points : np.ndarray, shape (n_points, n_position_dims)
            Positions to project onto the track graph.

        Returns
        -------
        linear_position : np.ndarray, shape (n_points,)
        """
        points = np.ascontiguousarray(points, dtype=float)
        key = hashlib.blake2b(
            np.asarray(points.shape).tobytes() + points.tobytes(), digest_size=16
        ).digest()
        cache = self.get_cached("linear_position")
        if cache is None:
            cache = self.set_cached("linear_position", OrderedDict(), transient=True)
        if key in cache:
            cache.move_to_end(key)
            return cast("np.ndarray", cache[key].copy())
        # A copy: a view would keep the linearization's other columns alive,
        # outside the bytes counted below.
        result = get_linearized_position(
            points,
            _with_edge_ids_in_edges_order(self.track_graph),
            edge_order=self.edge_order,
            edge_spacing=self.edge_spacing,
        ).linear_position.to_numpy(copy=True)
        cache[key] = result
        while (
            sum(value.nbytes for value in cache.values()) > _LINEAR_POSITION_CACHE_BYTES
            and len(cache) > 1
        ):
            cache.popitem(last=False)
        return cast("np.ndarray", result.copy())

    @overload
    def decoding_position(self, points: np.ndarray) -> np.ndarray: ...

    @overload
    def decoding_position(self, points: "jax.Array") -> "np.ndarray | jax.Array": ...

    def decoding_position(
        self, points: "np.ndarray | jax.Array"
    ) -> "np.ndarray | jax.Array":
        """Map points into the coordinates the place grid is binned in.

        On a track graph, multi-dimensional points (raw ``(x, y)`` tracking)
        are linearized onto the track; points with one column, or a 1-D array,
        are taken as already linear and returned unchanged. Without a track
        graph the points are returned unchanged.

        Parameters
        ----------
        points : np.ndarray or jax.Array, shape (n_points, n_position_dims) or (n_points,)
            Positions, such as the animal's tracked position.

        Returns
        -------
        decoding_points : np.ndarray, shape (n_points, n_decoding_dims)
            ``n_decoding_dims`` is 1 for linearized points and
            ``n_position_dims`` otherwise. Unchanged input is returned as is,
            the same object, including a 1-D array; an empty array of
            multi-dimensional points on a track graph gives shape ``(0, 1)``.
        """
        if self.track_graph is None or np.ndim(points) < 2 or points.shape[1] == 1:
            return points
        if points.shape[0] == 0:
            return np.empty((0, 1))
        return self._linear_position(points)[:, None]

    @property
    def is_track_interior_flat_(self) -> np.ndarray:
        """Flattened interior-bin mask.

        Returns
        -------
        is_track_interior : np.ndarray, shape (n_bins,)
            Boolean mask over ``place_bin_centers_`` (flattened
            ``is_track_interior_``).

        Raises
        ------
        NotFittedError
            If the environment is not fitted.
        """
        return np.asarray(self.is_track_interior_, dtype=bool).ravel()

    def _validate_track_graph(self) -> None:
        """Validate track_graph and its required parameters.

        Raises
        ------
        ValidationError
            If track_graph is invalid or required parameters are missing
        ConfigurationError
            If edge_order or edge_spacing are missing when track_graph is provided
        """
        # Validate track_graph is a NetworkX Graph
        val.ensure_networkx_graph(self.track_graph, "track_graph")
        assert self.track_graph is not None  # ensure_networkx_graph rejects None

        # Validate graph has sufficient nodes and edges
        val.ensure_graph_has_nodes(self.track_graph, "track_graph", min_nodes=2)
        val.ensure_graph_has_edges(self.track_graph, "track_graph", min_edges=1)

        # Validate required node attributes
        val.ensure_graph_nodes_have_attribute(self.track_graph, "track_graph", "pos")

        # Validate required edge attributes
        val.ensure_graph_edges_have_attribute(
            self.track_graph, "track_graph", "distance"
        )
        val.ensure_graph_edges_have_attribute(
            self.track_graph, "track_graph", "edge_id"
        )

        # Validate that edge_order is provided
        if self.edge_order is None:
            msg = "Missing edge_order for track_graph environment"
            raise ConfigurationError(
                msg,
                hint="When using track_graph, you must provide edge_order to specify how edges connect. Example: env = Environment(track_graph=graph, edge_order=edge_order, edge_spacing=15.0)",
            )

        # Validate that edge_spacing is provided
        if self.edge_spacing is None:
            msg = "Missing edge_spacing for track_graph environment"
            raise ConfigurationError(
                msg,
                hint="When using track_graph, you must provide edge_spacing to define bin size along edges. Example: env = Environment(track_graph=graph, edge_order=edge_order, edge_spacing=15.0)",
            )

        # Validate edge_order structure
        if not isinstance(self.edge_order, list):
            msg = "edge_order must be a list of tuples"  # type: ignore[unreachable]  # runtime check of user input
            raise ValidationError(
                msg,
                expected="list of 2-tuples",
                got=f"{type(self.edge_order).__name__}",
                hint="edge_order should be a list like [(0, 1), (1, 2), ...]",
                example="    edge_order = [(0, 1), (1, 2), (2, 3)]",
            )

        # Validate each edge in edge_order
        for i, edge in enumerate(self.edge_order):
            if not isinstance(edge, tuple) or len(edge) != 2:  # type: ignore[redundant-expr]  # runtime check of user input
                msg = f"edge_order[{i}] must be a 2-tuple"
                raise ValidationError(
                    msg,
                    expected="tuple of length 2",
                    got=f"{type(edge).__name__} of length {len(edge) if isinstance(edge, tuple | list) else 'N/A'}",  # type: ignore[redundant-expr]  # runtime check of user input
                    hint="Each edge should be a 2-tuple like (node1, node2)",
                    example=f"    edge_order[{i}] = (0, 1)  # Not {edge}",
                )

            # Check that edge exists in graph
            if edge not in self.track_graph.edges():
                # Try reversed edge (undirected graph)
                reversed_edge = (edge[1], edge[0])
                if reversed_edge not in self.track_graph.edges():
                    available_edges = list(self.track_graph.edges())[:5]
                    msg = f"edge_order[{i}] references non-existent edge {edge}"
                    raise ValidationError(
                        msg,
                        expected="edge that exists in track_graph",
                        got=f"edge {edge} not found",
                        hint=f"Available edges: {available_edges}...",
                        example=f"    # Check track_graph.edges(): {list(self.track_graph.edges())[:3]}",
                    )

        # Validate edge_spacing
        if isinstance(self.edge_spacing, int | float):
            val.ensure_positive_scalar(
                self.edge_spacing,
                "edge_spacing",
                minimum=0.0,
                strict=False,
            )
        elif isinstance(self.edge_spacing, list):
            expected_len = len(self.edge_order) - 1
            if len(self.edge_spacing) != expected_len:
                msg = "edge_spacing list has wrong length"
                raise ValidationError(
                    msg,
                    expected=f"length {expected_len} (one less than edge_order)",
                    got=f"length {len(self.edge_spacing)}",
                    hint="edge_spacing should have one entry for each gap between consecutive edges",
                    example=f"    # For {len(self.edge_order)} edges:\n    edge_spacing = [{', '.join(['15.0'] * expected_len)}]",
                )

            for i, spacing in enumerate(self.edge_spacing):
                val.ensure_positive_scalar(
                    spacing,
                    f"edge_spacing[{i}]",
                    minimum=0.0,
                    strict=False,
                )

    def _validate_position_range(self) -> None:
        """Validate position_range parameter.

        Raises
        ------
        ValidationError
            If position_range is invalid
        """
        if not isinstance(self.position_range, Sequence):
            msg = "position_range must be a sequence"
            raise ValidationError(
                msg,
                expected="list or tuple of (min, max) tuples",
                got=f"{type(self.position_range).__name__}",
                hint="position_range should be like [(xmin, xmax), (ymin, ymax)]",
                example="    position_range = [(0, 100), (0, 100)]",
            )

        for i, dim_range in enumerate(self.position_range):
            if not isinstance(dim_range, tuple) or len(dim_range) != 2:  # type: ignore[redundant-expr]  # runtime check of user input
                msg = f"position_range[{i}] must be a 2-tuple"  # type: ignore[unreachable]  # runtime check of user input
                raise ValidationError(
                    msg,
                    expected="(min, max) tuple",
                    got=f"{type(dim_range).__name__}",
                    hint="Each dimension range should be (min_value, max_value)",
                    example=f"    position_range[{i}] = (0.0, 100.0)",
                )

            min_val, max_val = dim_range
            if not isinstance(min_val, int | float) or not isinstance(  # type: ignore[redundant-expr]  # runtime check of user input
                max_val, int | float
            ):
                msg = f"position_range[{i}] must contain numeric values"  # type: ignore[unreachable]  # runtime check of user input
                raise ValidationError(
                    msg,
                    expected="(float, float)",
                    got=f"({type(min_val).__name__}, {type(max_val).__name__})",
                    hint="Range values must be numbers",
                    example=f"    position_range[{i}] = (0.0, 100.0)",
                )

            if min_val >= max_val:
                msg = f"position_range[{i}] has min >= max"
                raise ValidationError(
                    msg,
                    expected="min < max",
                    got=f"min={min_val}, max={max_val}",
                    hint="The minimum value must be less than the maximum value",
                    example=f"    position_range[{i}] = ({min_val}, {max_val + 1.0})",
                )

    def fit_place_grid(
        self,
        position: np.ndarray | None = None,
        *,
        infer_track_interior: bool = True,
        compute_all_pairs_distances: bool = True,
        max_dense_distance_bytes: int = DEFAULT_MAX_DENSE_DISTANCE_BYTES,
    ) -> "Environment":
        """Fit a discrete grid of the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_dim) or (n_time,), optional
            Position of the animal. Required without a ``track_graph``, where it
            sets the grid's extent. A one-dimensional ``(n_time,)`` array is
            treated as ``(n_time, 1)``.
        infer_track_interior : bool, optional
            Whether to infer the interior of the track from ``position`` when
            ``is_track_interior`` was not given, by default True. Stored as
            ``infer_track_interior_``; the ``infer_track_interior`` field is
            left unchanged. Ignored if ``track_graph`` is provided.
        compute_all_pairs_distances : bool, optional
            By default True, which builds the all-pairs graph-distance matrix.
            False keeps an exact sparse query view for Cartesian environments,
            avoiding quadratic setup. Explicit 1D track graphs keep their
            distance dictionary either way, without this memory bound.
        max_dense_distance_bytes : int, optional
            Byte limit for dense conversions and selections of the deferred
            Cartesian view, by default 256 MiB.

        Returns
        -------
        self : Environment
            The fitted environment, with a new ``place_grid_``. The grid is
            built completely before it is assigned: if the fit raises, the
            environment keeps its previous grid (or stays unfitted) and its
            cache. A new grid clears the cache.
        """
        # Validate position if provided
        if position is not None:
            val.ensure_ndarray(position, "position")
            # NaNs are tolerated here because get_grid and get_track_interior
            # both drop NaN rows internally; callers routinely pass position
            # arrays with missing samples. Infs are still rejected.
            if np.any(np.isinf(position)):
                msg = "position contains Inf values"
                raise ValidationError(
                    msg,
                    expected="finite or NaN values only",
                    got=f"{int(np.sum(np.isinf(position)))} Inf entries",
                    hint="Clean or drop Inf entries before calling fit_place_grid",
                )

            if position.ndim not in [1, 2]:
                msg = "position must be 1D or 2D"
                raise ValidationError(
                    msg,
                    expected="shape (n_time,) or (n_time, n_dims)",
                    got=f"shape {position.shape} (ndim={position.ndim})",
                    hint="Ensure position is a 1D or 2D array with time as first dimension",
                    example="    position = np.array([[0, 0], [1, 1], [2, 2]])  # Shape (n_time, n_dims)",
                )

            if position.shape[0] == 0:
                msg = "position has no time points"
                raise ValidationError(
                    msg,
                    expected="at least 1 time point",
                    got="0 time points (empty array)",
                    hint="Provide position data with at least one time point",
                    example="    position = np.array([[0, 0]])  # At least one position",
                )

            if position.ndim == 1:
                position = position[:, np.newaxis]

        if self.track_graph is None:
            if position is None:
                msg = "position is required to fit a grid without a track_graph"
                raise ValidationError(
                    msg,
                    expected="position array of shape (n_time, n_dim)",
                    got="position=None",
                    hint="Pass position, or give the Environment a track_graph",
                )
            grid = self._fit_cartesian_grid(
                position,
                infer_track_interior=infer_track_interior,
                compute_all_pairs_distances=compute_all_pairs_distances,
                max_dense_distance_bytes=max_dense_distance_bytes,
            )
        else:
            grid = self._fit_track_graph_grid()

        # Assigned whole, after every check: a fit that raises leaves the
        # previous grid and the cache derived from it.
        self._place_grid = grid
        self.clear_cache()
        return self

    def fit_place_grid_from_edges(
        self,
        edges: Sequence[ArrayLike],
        is_track_interior: ArrayLike | None = None,
        *,
        infer_track_interior: bool | None = False,
        compute_all_pairs_distances: bool = True,
        max_dense_distance_bytes: int = DEFAULT_MAX_DENSE_DISTANCE_BYTES,
    ) -> "Environment":
        """Fit the grid of an environment without a track graph from its bins.

        Rebuilds the grid :meth:`fit_place_grid` fitted, from the bin edges
        and interior mask it chose (``edges_`` and ``is_track_interior_``),
        without the position data. A saved model stores these instead of the
        grid; the bin centers, boundary, bin graph and distances follow from
        them as they do in :meth:`fit_place_grid`.

        Parameters
        ----------
        edges : sequence of array_like
            Bin edges of each dimension, each strictly increasing, shape
            (n_bins_d + 1,). For more than one dimension they include the
            boundary bins, as ``edges_`` does.
        is_track_interior : array_like of bool, shape (n_bins_1, ..., n_bins_d), optional
            Which bins are part of the track interior; by default every bin.
        infer_track_interior : bool or None, optional
            Recorded as ``infer_track_interior_``, the argument the original
            fit used; the mask is never inferred here. By default False.
        compute_all_pairs_distances : bool, optional
            See :meth:`fit_place_grid`. By default True.
        max_dense_distance_bytes : int, optional
            See :meth:`fit_place_grid`.

        Returns
        -------
        self : Environment
            The fitted environment, with a new ``place_grid_``.

        Raises
        ------
        ValidationError
            If the environment has a ``track_graph`` (whose grid
            :meth:`fit_place_grid` builds from the graph alone), an edge array
            is not one-dimensional, finite and strictly increasing, or the
            mask is not boolean with one entry per bin.
        """
        if self.track_graph is not None:
            msg = "fit_place_grid_from_edges needs an environment without a track_graph"
            raise ValidationError(
                msg,
                hint="Call fit_place_grid() instead: a track graph defines its grid.",
            )
        bin_edges = [np.array(edge, dtype=float) for edge in edges]
        for dim, edge in enumerate(bin_edges):
            if (
                edge.ndim != 1
                or edge.size < 2
                or not np.all(np.isfinite(edge))
                or np.any(np.diff(edge) <= 0)
            ):
                msg = f"edges[{dim}] must be finite and strictly increasing"
                raise ValidationError(
                    msg,
                    expected="a one-dimensional array of at least two edges",
                    got=f"shape {edge.shape}",
                )
        place_bin_edges, place_bin_centers, centers_shape = _grid_from_edges(bin_edges)
        if is_track_interior is None:
            interior = np.ones(centers_shape, dtype=bool)
        else:
            interior = np.array(is_track_interior)
            if interior.dtype != np.bool_ or interior.shape != centers_shape:
                msg = "is_track_interior must be a boolean mask of the grid's shape"
                raise ValidationError(
                    msg,
                    expected=f"bool, shape {centers_shape}",
                    got=f"{interior.dtype}, shape {interior.shape}",
                )
        self._place_grid = self._cartesian_place_grid(
            bin_edges,
            place_bin_edges,
            place_bin_centers,
            centers_shape,
            interior,
            infer_track_interior=infer_track_interior,
            compute_all_pairs_distances=compute_all_pairs_distances,
            max_dense_distance_bytes=max_dense_distance_bytes,
        )
        self.clear_cache()
        return self

    def _fit_cartesian_grid(
        self,
        position: np.ndarray,
        *,
        infer_track_interior: bool,
        compute_all_pairs_distances: bool,
        max_dense_distance_bytes: int,
    ) -> PlaceGrid:
        """Return the grid of an environment without a track graph.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_dims)
            Position samples that set the grid and, if inferred, its interior.
        infer_track_interior : bool
            Whether to infer the track interior from ``position``.
        compute_all_pairs_distances : bool
            Whether to build the all-pairs graph-distance matrix.
        max_dense_distance_bytes : int
            See :meth:`fit_place_grid`.

        Returns
        -------
        PlaceGrid
        """
        edges, place_bin_edges, place_bin_centers, centers_shape = get_grid(
            position,
            self.place_bin_size,
            self.position_range,
        )

        if self.is_track_interior is not None:
            val.ensure_ndarray(self.is_track_interior, "is_track_interior")
            if self.is_track_interior.dtype != np.bool_:
                msg = "is_track_interior must have boolean dtype"
                raise ValidationError(
                    msg,
                    expected="boolean array",
                    got=f"dtype {self.is_track_interior.dtype}",
                    hint="Provide a boolean mask of the Cartesian grid bins",
                )
            if self.is_track_interior.shape != centers_shape:
                msg = "is_track_interior shape must match the fitted grid"
                raise ValidationError(
                    msg,
                    expected=f"shape {centers_shape}",
                    got=f"shape {self.is_track_interior.shape}",
                    hint="Include the grid's boundary bins in the mask",
                )
            is_track_interior = self.is_track_interior.copy()
        elif infer_track_interior:
            is_track_interior = get_track_interior(
                position,
                edges,
                fill_holes=self.fill_holes,
                dilate=self.dilate,
                bin_count_threshold=self.bin_count_threshold,
            )
        else:
            is_track_interior = np.ones(centers_shape, dtype=bool)

        return self._cartesian_place_grid(
            edges,
            place_bin_edges,
            place_bin_centers,
            centers_shape,
            is_track_interior,
            infer_track_interior=infer_track_interior,
            compute_all_pairs_distances=compute_all_pairs_distances,
            max_dense_distance_bytes=max_dense_distance_bytes,
        )

    def _cartesian_place_grid(
        self,
        edges: Sequence[np.ndarray],
        place_bin_edges: np.ndarray,
        place_bin_centers: np.ndarray,
        centers_shape: tuple[int, ...],
        is_track_interior: np.ndarray,
        *,
        infer_track_interior: bool | None,
        compute_all_pairs_distances: bool,
        max_dense_distance_bytes: int,
    ) -> PlaceGrid:
        """Build the grid of an environment without a track graph from its bins.

        Parameters
        ----------
        edges : sequence of np.ndarray
            Bin edges of each dimension, each of shape (n_bins_d + 1,).
        place_bin_edges : np.ndarray, shape (n_total_edges, n_dims)
            Flattened bin edges of the grid.
        place_bin_centers : np.ndarray, shape (n_bins, n_dims)
            Flattened bin centers of the grid.
        centers_shape : tuple[int, ...]
            See :func:`get_grid`.
        is_track_interior : np.ndarray of bool, shape ``centers_shape``
            Which bins are part of the track interior.
        infer_track_interior : bool or None
            Recorded as ``infer_track_interior_``.
        compute_all_pairs_distances : bool
            See :meth:`fit_place_grid`.
        max_dense_distance_bytes : int
            See :meth:`fit_place_grid`.

        Returns
        -------
        PlaceGrid
        """
        is_track_boundary = (
            get_track_boundary(
                is_track_interior,
                n_position_dims=len(edges),
                connectivity=1,
            )
            if len(edges) > 1
            else None
        )

        track_graph_dd = _make_nD_track_graph(
            place_bin_centers, is_track_interior, centers_shape
        )
        distance_between_nodes = _graph_distances(
            track_graph_dd,
            compute_all_pairs_distances=compute_all_pairs_distances,
            max_dense_distance_bytes=max_dense_distance_bytes,
        )
        return PlaceGrid(
            edges=tuple(edges),
            place_bin_edges=place_bin_edges,
            place_bin_centers=place_bin_centers,
            centers_shape=centers_shape,
            is_track_interior=is_track_interior,
            distance_between_nodes=distance_between_nodes,
            configuration=self._grid_configuration(),
            infer_track_interior=infer_track_interior,
            is_track_boundary=is_track_boundary,
            track_graphDD=track_graph_dd,
        )

    def _fit_track_graph_grid(self) -> PlaceGrid:
        """Return the grid of an environment with a track graph.

        Returns
        -------
        PlaceGrid
        """
        # Note: track_graph validation is done in __post_init__
        # Track-graph path requires scalar place_bin_size.
        if isinstance(self.place_bin_size, tuple):
            msg = "place_bin_size must be a scalar when track_graph is provided"
            raise ValidationError(
                msg,
                expected="float",
                got=f"tuple {self.place_bin_size}",
                hint=(
                    "Linearized track-graph environments use a single bin size "
                    "along the manifold."
                ),
                example="place_bin_size=2.0",
            )
        assert self.edge_order is not None  # required by _validate_track_graph
        assert self.edge_spacing is not None  # required by _validate_track_graph
        (
            place_bin_centers,
            place_bin_edges,
            is_track_interior,
            distance_between_nodes,
            centers_shape,
            edges,
            track_graph_with_bin_centers_edges,
            original_nodes_df,
            place_bin_edges_nodes_df,
            place_bin_centers_nodes_df,
            nodes_df,
        ) = get_track_grid(
            self.track_graph,
            self.edge_order,
            self.edge_spacing,
            self.place_bin_size,
        )
        return PlaceGrid(
            edges=tuple(edges),
            place_bin_edges=place_bin_edges,
            place_bin_centers=place_bin_centers,
            centers_shape=centers_shape,
            is_track_interior=is_track_interior,
            distance_between_nodes=distance_between_nodes,
            configuration=self._grid_configuration(),
            track_graph_with_bin_centers_edges=track_graph_with_bin_centers_edges,
            original_nodes_df=original_nodes_df,
            place_bin_edges_nodes_df=place_bin_edges_nodes_df,
            place_bin_centers_nodes_df=place_bin_centers_nodes_df,
            nodes_df=nodes_df,
        )

    def plot_grid(self, ax: mpl.axes.Axes | None = None) -> mpl.axes.Axes:
        """Plot the fitted spatial grid of the environment.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Plot on this axis if given, by default None, which creates a new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis on which the grid is plotted.
        """
        if self.track_graph is not None:
            if ax is None:
                _, ax = plt.subplots(figsize=(15, 2))

            plot_graph_as_1D(
                self.track_graph, self.edge_order, self.edge_spacing, ax=ax
            )
            # Before fit_place_grid there are no bin edges to draw.
            if self._is_fitted:
                for edge in self.edges_[0]:
                    ax.axvline(edge.squeeze(), linewidth=0.5, color="black")
            ax.set_ylim((0, 0.1))
        elif len(self.edges_) == 1:
            if ax is None:
                _, ax = plt.subplots(figsize=(15, 2))
            edges = self.edges_[0]
            ax.pcolormesh(
                edges,
                [0.0, 1.0],
                self.is_track_interior_[np.newaxis, :],
                cmap="bone_r",
            )
            ax.vlines(edges, 0.0, 1.0, color="black", linewidth=0.5)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim((0, 1))
            ax.set_yticks([])
            ax.set_xlabel("Position")
        else:
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 7))
            ax.pcolormesh(
                self.edges_[0],
                self.edges_[1],
                self.is_track_interior_.T,
                cmap="bone_r",
            )
            ax.set_xticks(self.edges_[0], minor=True)
            ax.set_yticks(self.edges_[1], minor=True)
            ax.grid(visible=True, which="minor")
            ax.grid(visible=False, which="major")

        return ax

    def save_environment(self, filename: str = "environment.pkl") -> None:
        """Save the environment object as a pickled file.

        Parameters
        ----------
        filename : str, optional
            File name to save the environment to, by default "environment.pkl".
        """
        with Path(filename).open("wb") as file_handle:
            pickle.dump(self, file_handle)

    @classmethod
    def load_environment(cls, filename: str = "environment.pkl") -> "Environment":
        """Load a pickled environment object from a file.

        Parameters
        ----------
        filename : str, optional
            File name to load the environment from, by default "environment.pkl".

        Returns
        -------
        Environment
            The loaded environment object.
        """
        with Path(filename).open("rb") as f:
            return cast("Environment", pickle.load(f))

    def get_bin_ind(self, sample: np.ndarray) -> np.ndarray:
        """Find the indices of the bins to which each value in input array belongs.

        Uses the fitted grid edges (`self.edges_`). Gap-bin assignments
        (positions that would otherwise land in a structural non-interior
        bin, e.g. on an arm-boundary edge of a linearized track) are snapped
        to the nearest interior bin by raw position-coordinate distance.

        Parameters
        ----------
        sample : np.ndarray, shape (n_time, n_dim)
            Input data points.

        Returns
        -------
        bin_inds : np.ndarray, shape (n_time,)
            Flat index of the bin for each data point in `sample`, always an
            interior bin unless the grid has none.

        Raises
        ------
        NotFittedError
            Without a fit of the current configuration.

        Notes
        -----
        The large-snap ``DataQualityWarning`` only covers positions that land in a
        non-interior (gap) bin and are then snapped. In an all-interior
        environment (e.g. ``infer_track_interior=False``) an out-of-bounds
        position is clamped to the nearest edge bin by the histogram outlier
        logic without entering the snap path, so it is not warned about.
        Detecting out-of-bounds positions in that case would require a
        distance check on every call and is left as a follow-up.
        """
        # Raises NotFittedError (a ConfigurationError) without a current fit.
        grid = self.place_grid_

        # remove outer boundary edge
        edges = [e[1:-1] for e in grid.edges]

        try:
            # Sample is an ND-array.
            _N, D = sample.shape
        except (AttributeError, ValueError):
            # Sample is a sequence of 1D arrays.
            sample = np.atleast_2d(sample).T
            _N, D = sample.shape

        nbin = np.empty(D, np.intp)
        for i in range(D):
            nbin[i] = len(edges[i]) + 1  # includes an outlier on each end

        # Compute the bin number each sample falls into.
        Ncount = tuple(
            np.searchsorted(edges[i], sample[:, i], side="right") for i in range(D)
        )

        # Using digitize, values that fall on an edge are put in the right bin.
        # For the rightmost bin, we want values equal to the right edge to be
        # counted in the last bin, and not as an outlier.
        for i in range(D):
            # Find which points are on the rightmost edge.
            on_edge = sample[:, i] == edges[i][-1]
            # Shift these points one bin to the left.
            Ncount[i][on_edge] -= 1

        bin_inds = np.ravel_multi_index(Ncount, nbin)

        # Snap gap-bin assignments to nearest interior bin. Gap bins exist
        # as structural indexing artifacts (edge_spacing > 0 on a linearized
        # track, or holes in an occupancy-inferred interior). No valid
        # position belongs in them: positions at an arm-boundary edge land
        # there via searchsorted(side="right") tie-breaking, and positions
        # truly off-track should snap to the closest real bin.
        is_interior = grid.is_track_interior.ravel()
        needs_snap = ~is_interior[bin_inds]
        if needs_snap.any():
            interior_bin_indices = np.where(is_interior)[0]
            # Degenerate environment with no interior bins: skip snap
            # and return raw binning rather than crashing on argmin over
            # an empty axis.
            if len(interior_bin_indices) > 0:
                interior_centers = grid.place_bin_centers[interior_bin_indices]
                pos_to_snap = sample[needs_snap]
                # Broadcasted Euclidean distance:
                # (n_snap, n_interior, n_dim) -> (n_snap, n_interior)
                dists = np.linalg.norm(
                    pos_to_snap[:, np.newaxis, :] - interior_centers[np.newaxis, :, :],
                    axis=-1,
                )
                snap_distances = np.min(dists, axis=1)
                bin_inds[needs_snap] = interior_bin_indices[np.argmin(dists, axis=1)]

                # A small snap is expected (boundary tie-breaking, gap
                # bins). The threshold already allows snaps up to two
                # bin-widths plus the largest inter-arm gap, so exceeding
                # it usually means a tracking glitch or a position outside
                # the environment bounds — surface it. Non-finite snap
                # distances (a NaN/inf position) are handled explicitly:
                # np.max over them returns NaN, which compares False against
                # the threshold and would silently suppress the warning even
                # when a finite position in the same batch snapped far.
                finite = np.isfinite(snap_distances)
                max_snap = (
                    float(np.max(snap_distances[finite])) if finite.any() else 0.0
                )
                n_nonfinite = int((~finite).sum())
                threshold = self._snap_warn_threshold()
                if max_snap > threshold or n_nonfinite > 0:
                    warnings.warn(
                        f"{int(needs_snap.sum())} position(s) snapped to the "
                        f"nearest interior bin; max finite snap distance "
                        f"{max_snap:.3f} vs threshold {threshold:.3f} (2x "
                        f"place_bin_size plus the largest inter-arm gap on a "
                        f"linearized track), {n_nonfinite} non-finite. A snap "
                        f"beyond the threshold or a non-finite position "
                        f"usually indicates a tracking glitch or an "
                        f"out-of-bounds position.",
                        DataQualityWarning,
                        stacklevel=user_stacklevel(),
                    )

        return bin_inds

    def _snap_warn_threshold(self) -> float:
        """Snap distance beyond which ``get_bin_ind`` warns.

        Baseline is two bin-widths (boundary tie-breaking, sub-bin snaps).
        On a linearized track graph a position projected into an inter-arm
        gap is legitimately snapped to the nearest on-track bin by up to
        ~``edge_spacing``; that expected distance is added so routine
        into-gap snaps stay quiet and only snaps well beyond the gap (true
        tracking glitches / out-of-bounds positions) warn. Open-field grids
        have no gaps and keep the plain two-bin-width threshold.
        """
        if isinstance(self.place_bin_size, (tuple, list)):
            bin_term = 2.0 * max(self.place_bin_size)
        else:
            bin_term = 2.0 * float(self.place_bin_size)
        return bin_term + self._max_inter_arm_gap()

    def _max_inter_arm_gap(self) -> float:
        """Largest inter-arm gap width, or ``0.0`` when there are no gaps.

        Only linearized track-graph environments have gaps (``edge_spacing``
        between consecutive edges). Open-field grids return ``0.0``.
        """
        if self.track_graph is None or self.edge_spacing is None:
            return 0.0
        if isinstance(self.edge_spacing, (list, tuple)):
            return float(max(self.edge_spacing)) if len(self.edge_spacing) else 0.0
        return float(self.edge_spacing)

    def get_manifold_distances(
        self, position1: np.ndarray, position2: np.ndarray
    ) -> np.ndarray:
        """Compute the distance between pairs of positions along the track manifold.

        This uses the precomputed shortest path distances between bin centers on the
        graph representation of the environment (either 1D or N-D).

        Parameters
        ----------
        position1 : np.ndarray, shape (n_time, n_dims) or (n_dims,)
            The first set of positions.
        position2 : np.ndarray, shape (n_time, n_dims) or (n_dims,)
            The second set of positions. Must have the same shape as position1.

        Returns
        -------
        distances : np.ndarray, shape (n_time,)
            The shortest path distance along the track for each pair of positions.
            Returns np.inf if no path exists between the bins corresponding to the
            positions.
        """
        if not self._is_fitted:
            msg = "Environment has not been fitted yet. Call `fit_place_grid` first."
            raise RuntimeError(msg)
        distance_between_nodes = self.distance_between_nodes_

        position1 = np.atleast_2d(position1)
        position2 = np.atleast_2d(position2)

        # Validate input shapes
        if position1.shape != position2.shape:
            msg = "Position array shapes must match"
            raise ValidationError(
                msg,
                expected=f"position2 with shape {position1.shape}",
                got=f"position2 with shape {position2.shape}",
                hint="Both position arrays must have the same shape (n_points, n_dims)",
            )

        bin_ind1 = self.get_bin_ind(position1)
        bin_ind2 = self.get_bin_ind(position2)

        if self.track_graph is not None:  # 1D case uses dict-of-dicts
            if self.place_bin_centers_nodes_df_ is None:
                msg = (
                    "place_bin_centers_nodes_df_ is required for 1D manifold "
                    "distance lookup."
                )
                raise ValueError(msg)
            bin_to_node = np.asarray(self.place_bin_centers_nodes_df_.node_id)
            node_ids1 = bin_to_node[bin_ind1]
            node_ids2 = bin_to_node[bin_ind2]
            distances = np.full(node_ids1.shape, np.inf, dtype=float)
            for i, (n1, n2) in enumerate(zip(node_ids1, node_ids2, strict=True)):
                # node_id == -1 marks gap bins. After the get_bin_ind snap,
                # this case is unreachable for envs with is_track_interior_
                # set, but retained for backward compatibility with
                # partially-constructed or deserialized envs.
                if n1 == -1 or n2 == -1:
                    continue
                try:
                    distances[i] = distance_between_nodes[n1][n2]
                except KeyError:
                    # No path between the nodes on the track graph.
                    continue
        else:
            distances = cast("np.ndarray", distance_between_nodes)[bin_ind1, bin_ind2]

        return distances

    def get_distances_to_interior_bins(self, positions: np.ndarray) -> np.ndarray:
        """Compute distances from each position to every interior bin center.

        Returns an ``(n_positions, n_interior_bins)`` matrix of distances
        along the track manifold: on a track graph from a lazily-built dense
        bin-distance matrix, cached on the environment, for efficient
        ``np.ix_`` lookup; otherwise from the grid's graph distances
        (``distance_between_nodes_``). Positions that would otherwise land in
        gap bins are snapped to the nearest interior bin by
        :meth:`get_bin_ind`, so no NaN rows are produced.

        This is the all-pairs analogue of :meth:`get_manifold_distances`:
        rather than pairwise distances between matched position arrays,
        it returns the full cross-product of one position array against
        all interior bin centers. Used by the local-state position
        uncertainty kernel to compute its per-time-step Gaussian.

        Parameters
        ----------
        positions : np.ndarray, shape (n_positions, n_dims) or (n_positions,)
            Positions to compute distances from (typically the animal's
            interpolated position at each decoding time step).

        Returns
        -------
        distances : np.ndarray, shape (n_positions, n_interior_bins)
            Distance from each position to each interior bin. Rows for
            off-track positions are NaN; unreachable bin pairs are inf. The
            array is float64 and caller-owned: deferred graph distances budget
            only their Dijkstra workspace and row cache, not this output, so
            whole-recording requests allocate ``8 * n_positions *
            n_interior_bins`` bytes.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        """
        if not self._is_fitted:
            msg = "Environment has not been fitted yet. Call `fit_place_grid` first."
            raise RuntimeError(msg)

        is_interior = self.is_track_interior_.ravel()
        interior_bin_indices = np.where(is_interior)[0]

        if self.track_graph is not None:
            # 1D track graph: use dense bin-indexed distance matrix.
            if self.place_bin_centers_nodes_df_ is None:
                msg = (
                    "place_bin_centers_nodes_df_ is required for 1D track "
                    "graph distance lookup."
                )
                raise ValueError(msg)

            # Lazily build and cache dense (n_bins, n_bins) distance matrix
            # indexed by bin. Off-track bins (node_id == -1) get NaN.
            bin_dist = self.get_cached("bin_distance_matrix")
            if bin_dist is None:
                bin_to_node = np.asarray(self.place_bin_centers_nodes_df_.node_id)
                n_total_bins = len(bin_to_node)
                bin_dist = np.full((n_total_bins, n_total_bins), np.nan)
                for src_bin, src_node in enumerate(bin_to_node):
                    if src_node == -1:
                        continue
                    src_distances = cast(
                        "dict[int, dict[int, float]]", self.distance_between_nodes_
                    ).get(src_node, {})
                    for dst_bin, dst_node in enumerate(bin_to_node):
                        if dst_node == -1:
                            continue
                        bin_dist[src_bin, dst_bin] = src_distances.get(dst_node, np.inf)
                self.set_cached("bin_distance_matrix", bin_dist)

            position_bin_inds = self.get_bin_ind(np.asarray(positions))
            return cast(
                "np.ndarray",
                bin_dist[np.ix_(position_bin_inds, interior_bin_indices)],
            )

        if isinstance(self.distance_between_nodes_, LazyGraphDistances):
            # N-D deferred distances: exact rows for the visited source bins.
            position_bin_inds = self.get_bin_ind(np.asarray(positions))
            return cast(
                "np.ndarray",
                self.distance_between_nodes_.cross_distances(
                    position_bin_inds, interior_bin_indices
                ),
            )
        # N-D: distance_between_nodes_ is an (n_nodes, n_nodes) array.
        distances = cast("np.ndarray", self.distance_between_nodes_)
        position_bin_inds = self.get_bin_ind(np.asarray(positions))
        return cast(
            "np.ndarray", distances[np.ix_(position_bin_inds, interior_bin_indices)]
        )

    def get_direction(
        self,
        position: np.ndarray,
        position_time: np.ndarray | None = None,
        sigma: float = 0.1,
        sampling_frequency: float | None = None,
        *,
        classify_stop: bool = False,
        stop_speed_threshold: float = 1e-3,
    ) -> np.ndarray:
        """Get the direction of movement relative to the center of the track (inward/outward).

        Requires a fitted N-D environment with a corresponding track graph (`track_graphDD`).

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_dims)
            Position data.
        position_time : np.ndarray, shape (n_time,), optional
            Timestamps for position data. If None, assumes uniform sampling.
        sigma : float, optional
            Standard deviation (in seconds) for Gaussian smoothing of velocity towards center. Defaults to 0.1.
        sampling_frequency : float, optional
            Sampling frequency in Hz. If None, estimated from `position_time`.
        classify_stop : bool, optional
            If True, classify speeds below `stop_speed_threshold` as "stop". Defaults to False.
        stop_speed_threshold : float, optional
            Speed threshold for classifying stops. Defaults to 1e-3.

        Returns
        -------
        direction : np.ndarray, shape (n_time,)
            Array of strings: "inward", "outward", or "stop".

        Raises
        ------
        RuntimeError
            If the environment has not been fitted or lacks the N-D track graph.
        ValueError
            If sampling frequency cannot be determined.
        """
        if not self._is_fitted:
            msg = "Environment has not been fitted yet. Call `fit_place_grid` first."
            raise RuntimeError(msg)
        if self.track_graphDD is None:
            msg = "Direction finding requires a fitted N-D environment with a track graph ('track_graphDD') and precomputed distances."
            raise RuntimeError(msg)

        if position_time is None:
            position_time = np.arange(position.shape[0])
        if sampling_frequency is None:
            sampling_frequency = 1 / np.mean(np.diff(position_time))

        centrality = nx.closeness_centrality(self.track_graphDD, distance="distance")
        center_node_id = list(centrality.keys())[np.argmax(list(centrality.values()))]

        bin_ind = self.get_bin_ind(position)

        velocity_to_center = gaussian_smooth(
            np.gradient(
                cast("np.ndarray", self.distance_between_nodes_)[
                    bin_ind, center_node_id
                ]
            ),
            sigma,
            sampling_frequency,
            axis=0,
            truncate=8,
        )
        direction = np.where(
            velocity_to_center < 0,
            "inward",
            "outward",
        )

        if classify_stop:
            direction[np.abs(velocity_to_center) < stop_speed_threshold] = "stop"

        return direction


def _accept_unset_legacy_fitted_arguments(
    init: Callable[..., None],
) -> Callable[..., None]:
    """Wrap ``Environment.__init__`` to drop unset legacy fitted attributes.

    These are the attributes that ``vars(environment)`` rows of earlier
    releases hold.

    Those rows (main from 2025-10 to 2026-10, as Spyglass ``master`` passes
    them) hold every fitted attribute, None when unfitted, and ``_is_fitted``
    False. They are dropped with one ``DeprecationWarning`` naming them; a
    fitted value raises ``TypeError``.
    """

    @functools.wraps(init)
    def __init__(self: "Environment", **kwargs: Any) -> None:
        legacy = [name for name in _LEGACY_FITTED_ARGUMENTS if name in kwargs]
        if legacy:
            given = [name for name in legacy if not _is_unset(name, kwargs[name])]
            if given:
                msg = (
                    f"Environment() got fitted attributes {given}, which "
                    "fit_place_grid sets; pass the configuration only, e.g. "
                    "Environment(**environment.get_params())"
                )
                raise TypeError(msg)
            kwargs = {
                name: value for name, value in kwargs.items() if name not in legacy
            }
        # Construct first, so an unknown or invalid argument raises its own
        # error rather than the deprecation (an error under -W error).
        init(self, **kwargs)
        if legacy:
            warnings.warn(
                f"Environment() ignores the unset fitted attributes {legacy}, "
                "which parameter rows saved as vars(environment) hold; this "
                "will raise TypeError in a future release. Pass the "
                "configuration only (Environment().get_params() keys).",
                CompatibilityWarning,
                stacklevel=user_stacklevel(),
            )

    return __init__


# The fitted attributes ``vars(environment)`` rows of earlier releases hold.
_LEGACY_FITTED_ARGUMENTS = (*_LEGACY_FITTED_ATTRIBUTES, "_is_fitted")


def _is_unset(name: str, value: Any) -> bool:
    """Whether a legacy fitted argument holds no fitted value.

    That is None, or a false ``_is_fitted`` (saved rows may hold it as a NumPy
    boolean).
    """
    if value is None:
        return True
    return name == "_is_fitted" and isinstance(value, (bool, np.bool_)) and not value


Environment.__init__ = _accept_unset_legacy_fitted_arguments(  # type: ignore[method-assign]  # wraps the dataclass-generated __init__
    Environment.__init__
)


def get_n_bins(
    position: np.ndarray,
    bin_size: float | tuple[float, ...] = 2.5,
    position_range: Sequence[tuple[float, float]] | None = None,
) -> NDArray[np.int32]:
    """Get the number of bins needed to span a range given a bin size.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,) or (n_time, n_position_dims)
        Position samples. Their extent (maximum minus minimum) is used unless
        ``position_range`` is given.
    bin_size : float or tuple of float, optional
        Approximate bin size in each dimension, in the units of ``position``,
        by default 2.5.
    position_range : sequence of tuple of float, optional
        ``(min, max)`` for each dimension. If given, defines the extent
        instead of ``position``, by default None.

    Returns
    -------
    n_bins : np.ndarray of int32
        ``ceil(extent / bin_size)``, one entry per position dimension.
    """
    if position_range is not None:
        extent = np.diff(position_range, axis=1).squeeze()
    else:
        extent = np.ptp(position, axis=0)

    return np.ceil(extent / bin_size).astype(np.int32)


def get_grid(
    position: np.ndarray,
    bin_size: float | tuple[float, ...] = 2.5,
    position_range: Sequence[tuple[float, float]] | None = None,
) -> tuple[Sequence[np.ndarray], np.ndarray, np.ndarray, tuple[int, ...]]:
    """Calculate bin edges and centers for a spatial grid.

    Creates a grid based on the provided position data or range, using
    a specified bin size. Handles multiple position dimensions. For grids with
    more than one dimension, one boundary bin is added at each end of every
    dimension's edges; the edges of a 1-D grid are not padded.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
        Position data used to determine the grid extent if `position_range`
        is not provided. NaNs are ignored.
    bin_size : float or tuple of float, optional
        The desired approximate size of the bins in each dimension (one value
        per dimension if a tuple). The actual size may vary slightly to evenly
        cover the range. By default 2.5.
    position_range : list[tuple[float, float]] | None, optional
        A list of tuples, one for each position dimension, specifying the
        (min, max) boundary for the grid. If None, the min/max of the
        `position` data is used. By default None.

    Returns
    -------
    edges : tuple[np.ndarray, ...]
        A tuple containing the bin edges for each position dimension. Each
        element is a 1D array of shape (n_bins_d + 1,), where n_bins_d is
        the number of bins in that dimension `d`. Includes the boundary bins
        for grids with more than one dimension.
    place_bin_edges_flat : np.ndarray, shape (n_total_edges, n_position_dims)
        The meshgrid of the bin edges, flattened so each row is one edge-grid
        point.
    place_bin_centers_flat : np.ndarray, shape (n_total_bins, n_position_dims)
        The center coordinate of each bin in the flattened grid.
        `n_total_bins` is the product of the number of bins in each dimension.
    centers_shape : tuple[int, ...]
        The shape of the grid in terms of the number of bins in each dimension
        (e.g., (n_bins_x, n_bins_y)).
    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    n_bins = get_n_bins(position, bin_size, position_range)
    edges: Sequence[np.ndarray]
    _, edges = np.histogramdd(position, bins=n_bins, range=position_range)
    if len(edges) > 1:
        edges = [
            np.insert(
                edge,
                (0, len(edge)),
                (edge[0] - np.diff(edge)[0], edge[-1] + np.diff(edge)[0]),
            )
            for edge in edges
        ]
    return (edges, *_grid_from_edges(edges))


def _grid_from_edges(
    edges: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return the flattened bin edges and centers of the grid with these edges.

    Parameters
    ----------
    edges : sequence of np.ndarray
        Bin edges of each dimension, each of shape (n_bins_d + 1,).

    Returns
    -------
    place_bin_edges : np.ndarray, shape (n_total_edges, n_position_dims)
    place_bin_centers : np.ndarray, shape (n_total_bins, n_position_dims)
    centers_shape : tuple[int, ...]
        See :func:`get_grid`.
    """
    mesh_edges = np.meshgrid(*edges, indexing="ij")
    place_bin_edges = np.stack([edge.ravel() for edge in mesh_edges], axis=1)

    mesh_centers = np.meshgrid(*[get_centers(edge) for edge in edges], indexing="ij")
    place_bin_centers = np.stack([center.ravel() for center in mesh_centers], axis=1)
    centers_shape = tuple([len(edge) - 1 for edge in edges])

    return place_bin_edges, place_bin_centers, centers_shape


def get_track_interior(
    position: np.ndarray,
    bins: int | Sequence[int] | Sequence[np.ndarray],
    *,
    fill_holes: bool = False,
    dilate: bool = False,
    bin_count_threshold: int = 0,
) -> np.ndarray:
    """Infer the interior bins of the track from the positions.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
        Position samples used to count occupancy in each bin.
    bins : int or sequence of int or sequence of np.ndarray
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    fill_holes : bool, optional
        Fill any holes in the extracted track interior bins, by default False.
    dilate : bool, optional
        Inflate the extracted track interior bins, by default False.
    bin_count_threshold : int, optional
        A bin is considered on the track if it contains more than this number
        of samples, by default 0.

    Returns
    -------
    is_track_interior : np.ndarray, shape (n_bins_dim1, n_bins_dim2, ...)
        Boolean array, True for bins in the track interior as inferred from
        position.
    """
    bin_counts, _ = np.histogramdd(position, bins=bins)
    is_track_interior = (bin_counts > bin_count_threshold).astype(int)
    n_position_dims = position.shape[1]
    if n_position_dims > 1:
        structure = ndimage.generate_binary_structure(n_position_dims, 1)
        is_track_interior = ndimage.binary_closing(
            is_track_interior, structure=structure
        )
        if fill_holes:
            is_track_interior = ndimage.binary_fill_holes(is_track_interior)

        if dilate:
            is_track_interior = ndimage.binary_dilation(is_track_interior)
        # Zero the trailing "padding" bin along every spatial axis. get_grid
        # inserts an extra bin on each end so that binary_closing/dilation has
        # room to operate without bleeding past the actual data range; this
        # loop cleans those padding bins back out and generalizes to N-D
        # (previous code only handled the first two axes).
        for axis in range(n_position_dims):
            idx: list[slice | int] = [slice(None)] * n_position_dims
            idx[axis] = -1
            is_track_interior[tuple(idx)] = False
    return is_track_interior.astype(bool)


def get_track_segments_from_graph(track_graph: nx.Graph) -> np.ndarray:
    """Return the node positions of each edge in a track graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose nodes have a ``pos`` attribute.

    Returns
    -------
    track_segments : np.ndarray, shape (n_segments, 2, n_space)
        For each edge, the positions of its two end nodes.
    """
    node_positions = nx.get_node_attributes(track_graph, "pos")
    return np.asarray(
        [
            (node_positions[node1], node_positions[node2])
            for node1, node2 in track_graph.edges()
        ]
    )


def _segment_index_of_edge_id(track_graph: nx.Graph, edge_id: np.ndarray) -> np.ndarray:
    """Position in ``track_graph.edges`` of the edge carrying each ``edge_id``.

    ``edge_id`` is an edge label; the track segments
    (`get_track_segments_from_graph`) are in ``track_graph.edges`` order, which
    need not be the order of the labels.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose edges have unique ``edge_id`` attributes.
    edge_id : np.ndarray, shape (n,), int
        Edge labels, each one of the graph's ``edge_id`` values.

    Returns
    -------
    segment_index : np.ndarray, shape (n,), int
    """
    graph_edge_ids = np.asarray(
        [graph_edge_id for _, _, graph_edge_id in track_graph.edges(data="edge_id")]
    )
    sorter = np.argsort(graph_edge_ids)
    return cast(
        "np.ndarray",
        sorter[np.searchsorted(graph_edge_ids, edge_id, sorter=sorter)],
    )


def _with_edge_ids_in_edges_order(track_graph: nx.Graph) -> nx.Graph:
    """``track_graph``, or a copy whose ``edge_id`` is the position in ``graph.edges``.

    ``track_linearization.get_linearized_position`` numbers the segment of
    each position by its place in ``graph.edges`` and then looks that number
    up as an ``edge_id``, so it is right only when the two agree (as in graphs
    from ``track_linearization.make_track_graph``). Its linear position does
    not otherwise depend on ``edge_id``.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose edges have unique ``edge_id`` attributes.

    Returns
    -------
    track_graph : networkx.Graph
        ``track_graph`` itself if its ``edge_id`` values already are
        ``0, 1, ...`` in ``graph.edges`` order, else a relabeled copy.
    """
    edge_ids = [edge_id for _, _, edge_id in track_graph.edges(data="edge_id")]
    if edge_ids == list(range(len(edge_ids))):
        return track_graph
    # Graph.copy keeps the order of graph.edges.
    relabeled = track_graph.copy()
    for index, edge in enumerate(relabeled.edges):
        relabeled.edges[edge]["edge_id"] = index
    return relabeled


def project_points_to_segment(
    track_segments: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Find the closest point on each track segment in Euclidean distance.

    Parameters
    ----------
    track_segments : np.ndarray, shape (n_segments, 2, n_space)
        End-point positions of each segment.
    position : np.ndarray, shape (n_time, n_space)
        Positions to project.

    Returns
    -------
    projected_positions : np.ndarray, shape (n_time, n_segments, n_space)
        For each position, the closest point on each segment, clipped to the
        segment's end points.
    """
    segment_diff = np.diff(track_segments, axis=1).squeeze(axis=1)
    sum_squares = np.sum(segment_diff**2, axis=1)
    node1 = track_segments[:, 0, :]
    nx = (
        np.sum(segment_diff * (position[:, np.newaxis, :] - node1), axis=2)
        / sum_squares
    )
    nx[np.where(nx < 0)] = 0.0
    nx[np.where(nx > 1)] = 1.0
    return cast(
        "np.ndarray",
        node1[np.newaxis, ...] + (nx[:, :, np.newaxis] * segment_diff[np.newaxis, ...]),
    )


def _calculate_linear_position(
    track_graph: nx.Graph,
    position: np.ndarray,
    track_segment_id: np.ndarray,
    edge_order: list[tuple[Any, ...]],
    edge_spacing: float | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determine the linear position given a 2D position and a track graph.

    Parameters
    ----------
    track_graph : nx.Graph
        Track graph whose edges carry a ``distance`` attribute.
    position : np.ndarray, shape (n_time, n_position_dims)
        Position of the animal.
    track_segment_id : np.ndarray, shape (n_time,)
        ``edge_id`` of the edge each position is assigned to; NaN is treated
        as ``edge_id`` 0.
    edge_order : list of 2-tuples
        Order in which the edges are laid out in 1D.
    edge_spacing : float or list, len n_edges - 1
        Spacing between consecutive edges in 1D, either one value or one per
        gap.

    Returns
    -------
    linear_position : np.ndarray, shape (n_time,)
    projected_track_positions_x : np.ndarray, shape (n_time,)
    projected_track_positions_y : np.ndarray, shape (n_time,)

    """
    is_nan = np.isnan(track_segment_id)
    track_segment_id = track_segment_id.copy()
    track_segment_id[is_nan] = 0  # need to check
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[
        (np.arange(n_time), _segment_index_of_edge_id(track_graph, track_segment_id))
    ]

    n_edges = len(edge_order)
    if isinstance(edge_spacing, (int, float)):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    counter = 0.0
    start_node_linear_position: Any = []

    for ind, edge in enumerate(edge_order):
        start_node_linear_position.append(counter)

        with contextlib.suppress(IndexError):
            counter += track_graph.edges[edge]["distance"] + edge_spacing[ind]

    start_node_linear_position = np.asarray(start_node_linear_position)

    track_segment_id_to_start_node_linear_position = {
        track_graph.edges[e]["edge_id"]: snlp
        for e, snlp in zip(edge_order, start_node_linear_position, strict=False)
    }

    start_node_linear_position = np.asarray(
        [
            track_segment_id_to_start_node_linear_position[edge_id]
            for edge_id in track_segment_id
        ]
    )

    track_segment_id_to_edge = {track_graph.edges[e]["edge_id"]: e for e in edge_order}
    start_node_id = np.asarray(
        [track_segment_id_to_edge[edge_id][0] for edge_id in track_segment_id]
    )
    start_node_2D_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in start_node_id]
    )

    linear_position = start_node_linear_position + (
        np.linalg.norm(start_node_2D_position - projected_track_positions, axis=1)
    )
    linear_position[is_nan] = np.nan

    return (
        linear_position,
        projected_track_positions[:, 0],
        projected_track_positions[:, 1],
    )


def make_track_graph_with_bin_centers_edges(
    track_graph: nx.Graph, place_bin_size: float
) -> nx.Graph:
    """Insert the bin center and bin edge positions as nodes in the track graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose nodes have a ``pos`` attribute.
    place_bin_size : float
        Approximate size of the position bins, in the units of the node
        positions. Each edge is split into ``2 * ceil(length / place_bin_size) + 1``
        alternating bin-edge and bin-center nodes.

    Returns
    -------
    track_graph_with_bin_centers_edges : networkx.Graph
        Copy of the track graph in which each original edge is replaced by a
        path through the added nodes. Added nodes have ``pos``, ``edge_id`` and
        ``is_bin_edge`` attributes.
    """
    track_graph_with_bin_centers_edges = track_graph.copy()
    n_nodes = len(track_graph.nodes)

    for edge_ind, (node1, node2) in enumerate(track_graph.edges):
        # Preserve the user-assigned edge_id if present, otherwise fall back to
        # the enumeration index. Using the attribute keeps bin-center node
        # edge_ids consistent with _calculate_linear_position, which looks up
        # edge_id via track_graph.edges[e]["edge_id"].
        edge_id = track_graph.edges[(node1, node2)].get("edge_id", edge_ind)
        node1_x_pos, node1_y_pos = track_graph.nodes[node1]["pos"]
        node2_x_pos, node2_y_pos = track_graph.nodes[node2]["pos"]

        edge_size = np.linalg.norm(
            [(node2_x_pos - node1_x_pos), (node2_y_pos - node1_y_pos)]
        )
        n_bins = 2 * np.ceil(edge_size / place_bin_size).astype(np.int32) + 1
        if ~np.isclose(node1_x_pos, node2_x_pos):
            f = interp1d((node1_x_pos, node2_x_pos), (node1_y_pos, node2_y_pos))
            xnew = np.linspace(node1_x_pos, node2_x_pos, num=n_bins, endpoint=True)
            xy = np.stack((xnew, f(xnew)), axis=1)
        else:
            ynew = np.linspace(node1_y_pos, node2_y_pos, num=n_bins, endpoint=True)
            xnew = np.ones_like(ynew) * node1_x_pos
            xy = np.stack((xnew, ynew), axis=1)
        dist_between_nodes = np.linalg.norm(np.diff(xy, axis=0), axis=1)

        new_node_ids = n_nodes + np.arange(len(dist_between_nodes) + 1)
        nx.add_path(
            track_graph_with_bin_centers_edges,
            [*new_node_ids],
            distance=dist_between_nodes[0],
        )
        nx.add_path(
            track_graph_with_bin_centers_edges, [node1, new_node_ids[0]], distance=0
        )
        nx.add_path(
            track_graph_with_bin_centers_edges, [node2, new_node_ids[-1]], distance=0
        )
        track_graph_with_bin_centers_edges.remove_edge(node1, node2)
        for ind, (node_id, pos) in enumerate(zip(new_node_ids, xy, strict=False)):
            track_graph_with_bin_centers_edges.nodes[node_id]["pos"] = pos
            track_graph_with_bin_centers_edges.nodes[node_id]["edge_id"] = edge_id
            if ind % 2:
                track_graph_with_bin_centers_edges.nodes[node_id]["is_bin_edge"] = False
            else:
                track_graph_with_bin_centers_edges.nodes[node_id]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node1]["edge_id"] = edge_id
        track_graph_with_bin_centers_edges.nodes[node2]["edge_id"] = edge_id
        track_graph_with_bin_centers_edges.nodes[node1]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node2]["is_bin_edge"] = True
        n_nodes = len(track_graph_with_bin_centers_edges.nodes)

    return track_graph_with_bin_centers_edges


def extract_bin_info_from_track_graph(
    track_graph: nx.Graph,
    track_graph_with_bin_centers_edges: nx.Graph,
    edge_order: list[tuple[Any, ...]],
    edge_spacing: float | list[float],
) -> pd.DataFrame:
    """Find the edge, position and linear position of each node of the track graph.

    For each node, finds ``edge_id``, ``is_bin_edge``, ``x_position``,
    ``y_position`` and ``linear_position``.

    Parameters
    ----------
    track_graph : networkx.Graph
        The original track graph.
    track_graph_with_bin_centers_edges : networkx.Graph
        The track graph with bin edge and bin center nodes added, from
        `make_track_graph_with_bin_centers_edges`.
    edge_order : list of 2-tuples
        Order in which the edges are laid out in 1D.
    edge_spacing : float or list of float
        Spacing between consecutive edges in 1D, either one value or one per
        gap (``len(edge_order) - 1``).

    Returns
    -------
    nodes_df : pd.DataFrame
        One row per node (index ``node_id``) with the columns listed above,
        sorted by edge position and linear position.
    """
    nodes_df = (
        pd.DataFrame.from_dict(
            dict(track_graph_with_bin_centers_edges.nodes(data=True)), orient="index"
        )
        .assign(x_position=lambda df: np.asarray(list(df.pos))[:, 0])
        .assign(y_position=lambda df: np.asarray(list(df.pos))[:, 1])
        .drop(columns="pos")
    )
    node_linear_position, _, _ = _calculate_linear_position(
        track_graph,
        np.asarray(nodes_df.loc[:, ["x_position", "y_position"]]),
        np.asarray(nodes_df.edge_id),
        edge_order,
        edge_spacing,
    )
    nodes_df["linear_position"] = node_linear_position
    nodes_df = nodes_df.rename_axis(index="node_id")
    edge_avg_linear_position = (
        nodes_df.groupby("edge_id")
        .linear_position.mean()
        .rename("edge_avg_linear_position")
    )

    return (
        nodes_df.reset_index()
        .merge(edge_avg_linear_position, on="edge_id")
        .sort_values(by=["edge_avg_linear_position", "linear_position"], axis="rows")
        .set_index("node_id")
        .drop(columns="edge_avg_linear_position")
    )


def get_track_grid(
    track_graph: nx.Graph,
    edge_order: list[tuple[Any, ...]],
    edge_spacing: float | list[float],
    place_bin_size: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, dict[int, float]],
    tuple[int, ...],
    Sequence[np.ndarray],
    nx.Graph,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Figure out the 1D spatial bins given a track graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose nodes have a ``pos`` attribute.
    edge_order : list of 2-tuples
        Order in which the edges are laid out in 1D.
    edge_spacing : float or list of float
        Spacing between consecutive edges in 1D, either one value or one per
        gap (``len(edge_order) - 1``). Bins that fall in a gap with positive
        spacing are marked as not on the track.
    place_bin_size : float
        Approximate size of the position bins, in the units of the node
        positions.

    Returns
    -------
    place_bin_centers : np.ndarray, shape (n_bins, 1)
        Linear position of the center of each bin.
    place_bin_edges : np.ndarray, shape (n_bin_edges, 1)
        Linear position of the bin edges.
    is_track_interior : np.ndarray, shape (n_bins,)
        Boolean array, False for bins that fall in a gap between edges.
    distance_between_nodes : dict
        Shortest-path distance between every pair of nodes of
        `track_graph_with_bin_centers_edges`, as a dict of dicts.
    centers_shape : tuple of int
        Grid shape, ``(n_bins,)``.
    edges : list of np.ndarray
        One-element list holding the 1D bin edges.
    track_graph_with_bin_centers_edges : networkx.Graph
        Track graph with bin edge and bin center nodes added.
    original_nodes_df : pd.DataFrame
        Table of information about the original nodes in the track graph.
    place_bin_edges_nodes_df : pd.DataFrame
        Table of information about the added bin edge nodes.
    place_bin_centers_nodes_df : pd.DataFrame
        Table of information about bin centers, including bin centers that
        are not on the track.
    nodes_df : pd.DataFrame
        Table of information about the original nodes, bin edges, and bin
        centers.
    """
    track_graph_with_bin_centers_edges = make_track_graph_with_bin_centers_edges(
        track_graph, place_bin_size
    )
    nodes_df = extract_bin_info_from_track_graph(
        track_graph, track_graph_with_bin_centers_edges, edge_order, edge_spacing
    )

    # Dataframe with nodes from track graph only
    original_nodes = list(track_graph.nodes)
    original_nodes_df = nodes_df.loc[original_nodes].reset_index()

    # Dataframe with only added edge nodes
    place_bin_edges_nodes_df = nodes_df.loc[
        ~nodes_df.index.isin(original_nodes) & nodes_df.is_bin_edge
    ].reset_index()

    # Dataframe with only added center nodes
    place_bin_centers_nodes_df = nodes_df.loc[~nodes_df.is_bin_edge].reset_index()

    # Determine place bin edges and centers.
    # Make sure to remove duplicate nodes from bins with no gaps
    is_duplicate_edge = np.isclose(
        np.diff(np.asarray(place_bin_edges_nodes_df.linear_position)), 0.0
    )
    is_duplicate_edge = np.append(is_duplicate_edge, False)
    no_duplicate_place_bin_edges_nodes_df = place_bin_edges_nodes_df.iloc[
        ~is_duplicate_edge
    ]
    place_bin_edges = np.asarray(no_duplicate_place_bin_edges_nodes_df.linear_position)
    place_bin_centers = get_centers(place_bin_edges)

    # Compute distance between nodes
    distance_between_nodes = dict(
        nx.all_pairs_dijkstra_path_length(
            track_graph_with_bin_centers_edges, weight="distance"
        )
    )

    # Figure out which points are on the track and not just gaps
    change_edge_ind = np.nonzero(
        np.diff(no_duplicate_place_bin_edges_nodes_df.edge_id)
    )[0]

    if isinstance(edge_spacing, (int, float)):
        n_edges = len(edge_order)
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    is_track_interior = np.ones_like(place_bin_centers, dtype=bool)
    not_track = change_edge_ind[np.asarray(edge_spacing) > 0]
    is_track_interior[not_track] = False

    # Add information about bin centers not on track
    place_bin_centers_nodes_df = (
        pd.concat(
            (
                place_bin_centers_nodes_df,
                pd.DataFrame(
                    {
                        "linear_position": place_bin_centers[~is_track_interior],
                        "node_id": -1,
                        "edge_id": -1,
                        "is_bin_edge": False,
                    }
                ),
            )
        ).sort_values(by=["linear_position"], axis="rows")
    ).reset_index(drop=True)

    # Other needed information
    edges = [place_bin_edges]
    centers_shape = (place_bin_centers.size,)

    return (
        place_bin_centers[:, np.newaxis],
        place_bin_edges[:, np.newaxis],
        is_track_interior,
        distance_between_nodes,
        centers_shape,
        edges,
        track_graph_with_bin_centers_edges,
        original_nodes_df,
        place_bin_edges_nodes_df,
        place_bin_centers_nodes_df,
        nodes_df.reset_index(),
    )


def get_track_boundary(
    is_track_interior: np.ndarray, n_position_dims: int = 2, connectivity: int = 1
) -> np.ndarray:
    """Determine the boundary of the valid interior track bins.

    The boundary bins are not on the track but surround it.

    Parameters
    ----------
    is_track_interior : np.ndarray, shape (n_bins_x, n_bins_y)
        Boolean array, True for bins in the track interior.
    n_position_dims : int, optional
        Number of dimensions of the grid, by default 2.
    connectivity : int, optional
        Determines which elements of the output array belong to the structure,
        i.e., are considered as neighbors of the central element. Elements up
        to a squared distance of `connectivity` from the center are considered
        neighbors. `connectivity` may range from 1 (no diagonal elements are
        neighbors) to `n_position_dims` (all elements are neighbors). By
        default 1.

    Returns
    -------
    is_track_boundary : np.ndarray, same shape as ``is_track_interior``
        Boolean array, True for bins adjacent to the track interior.
    """
    structure = ndimage.generate_binary_structure(
        rank=n_position_dims, connectivity=connectivity
    )
    return cast(
        "np.ndarray",
        ndimage.binary_dilation(is_track_interior, structure=structure)
        ^ is_track_interior,
    )


def make_nD_track_graph_from_environment(environment: Environment) -> nx.Graph:
    """Create a graph of the track with nodes at the center of each bin.

    Nodes are the bin centers and edges connect adjacent on-track bins
    (including diagonals), weighted by their Euclidean distance.

    Parameters
    ----------
    environment : Environment
        A fitted N-D environment (``place_bin_centers_``, ``is_track_interior_``
        and ``centers_shape_`` set).

    Returns
    -------
    track_graph : networkx.Graph
        Graph with node attributes ``pos`` and ``is_track_interior`` and edge
        attributes ``distance`` and ``edge_id``.
    """
    return _make_nD_track_graph(
        environment.place_bin_centers_,
        environment.is_track_interior_,
        environment.centers_shape_,
    )


def _make_nD_track_graph(
    place_bin_centers: np.ndarray,
    is_track_interior: np.ndarray,
    centers_shape: tuple[int, ...],
) -> nx.Graph:
    """Graph of an N-D grid's bin centers; see ``make_nD_track_graph_from_environment``.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_bins, n_dims)
        Bin centers of the grid.
    is_track_interior : np.ndarray of bool, shape ``centers_shape``
        Interior flag of each bin, in the grid's shape.
    centers_shape : tuple[int, ...]
        Number of bin centers along each dimension.

    Returns
    -------
    track_graph : networkx.Graph
    """
    track_graph = nx.Graph()
    axis_offsets = [-1, 0, 1]

    # Enumerate over nodes
    interior_data = np.asarray(is_track_interior, dtype=bool).ravel()

    for node_id, (node_position, is_interior) in enumerate(
        zip(
            place_bin_centers,
            interior_data,
            strict=False,
        )
    ):
        track_graph.add_node(
            node_id, pos=tuple(node_position), is_track_interior=is_interior
        )

    edges: Any = []  # becomes an (n_edges, 2) array after concatenation
    ind: Any
    # Enumerate over nodes in the track interior
    for ind in zip(*np.nonzero(is_track_interior), strict=False):
        ind = np.array(ind)
        # Full 3**n_dims neighborhood block of grid indices (includes diagonals)
        adj_inds = np.meshgrid(*[axis_offsets + i for i in ind], indexing="ij")
        # Joint in-bounds mask across ALL dimensions. Masking each dimension
        # independently desynchronizes the per-axis index arrays whenever a node
        # sits on the grid boundary in a subset of dimensions (they end up with
        # different lengths and fail to broadcast).
        in_bounds = np.ones(adj_inds[0].shape, dtype=bool)
        for inds, dim_size in zip(adj_inds, centers_shape, strict=False):
            in_bounds &= (inds >= 0) & (inds < dim_size)
        adj_coords = [inds[in_bounds] for inds in adj_inds]

        # Is the adjacent node on the track?
        adj_on_track_inds = is_track_interior[tuple(adj_coords)]

        # Remove the center node (offset (0, ..., 0) from ``ind``)
        is_center = np.all(
            [coord == i for coord, i in zip(adj_coords, ind, strict=False)],
            axis=0,
        )
        adj_on_track_inds = adj_on_track_inds & ~is_center

        # Get the node ids of the center node
        node_id = np.ravel_multi_index(ind, centers_shape)

        # Get the node ids of the adjacent nodes on the track
        adj_node_ids = np.ravel_multi_index(
            [coord[adj_on_track_inds] for coord in adj_coords],
            centers_shape,
        )

        # Collect the edges for the graph
        edges.append(
            np.concatenate(
                np.meshgrid([node_id], adj_node_ids, indexing="ij"), axis=0
            ).T
        )

    edges = np.concatenate(edges)

    # Add edges to the graph with distance
    for node1, node2 in edges:
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


def gaussian_smooth(
    data: np.ndarray,
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: int = 8,
) -> np.ndarray:
    """Convolve the data with a 1D Gaussian along one axis.

    The standard deviation of the Gaussian is in time units (seconds when
    ``sampling_frequency`` is in Hz). The function is a wrapper around scipy's
    `gaussian_filter1d` with a constant (zero) boundary mode. The support is
    truncated at 8 standard deviations by default, instead of 4 in
    `gaussian_filter1d`.

    Parameters
    ----------
    data : np.ndarray, shape (..., n_time, ...)
        Input data array to be smoothed.
    sigma : float
        Standard deviation of the Gaussian kernel in time units.
    sampling_frequency : float
        Sampling frequency of the data in Hz.
    axis : int, optional
        Axis along which to apply the smoothing, by default 0.
    truncate : int, optional
        Truncate the filter at this many standard deviations, by default 8.

    Returns
    -------
    smoothed_data : np.ndarray, same shape as data
        Gaussian smoothed data array.
    """
    return cast(
        "np.ndarray",
        ndimage.gaussian_filter1d(
            data,
            sigma * sampling_frequency,
            truncate=truncate,
            axis=axis,
            mode="constant",
        ),
    )
