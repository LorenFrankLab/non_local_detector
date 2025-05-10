from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from non_local_detector.environment.geometry_engines import GeometryEngine, make_engine
from non_local_detector.environment.geometry_utils import _get_distance_between_bins

try:
    import shapely.geometry as _shp  # noqa: E402

    _HAS_SHAPELY = True
except ModuleNotFoundError:  # polygon support disabled
    _HAS_SHAPELY = False

    class _shp:  # type: ignore[misc]
        """Dummy shim so type references still work."""

        class Polygon:  # noqa: N801
            pass


def check_fitted(method):
    """
    Decorator for Environment instance methods that must only be
    called *after* `fit()`.

    Raises
    ------
    RuntimeError
        If the Environment has not yet been fitted.
    """

    @wraps(method)
    def _inner(self, *args, **kwargs):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                f"{self.__class__.__name__}.{method.__name__}() "
                "requires the environment to be fitted. "
                "Call `.fit()` first."
            )
        return method(self, *args, **kwargs)

    return _inner


@dataclass
class RegionInfo:
    """Container for a symbolic region.

    Parameters
    ----------
    name
        User-supplied identifier (must be unique per environment).
    kind
        One of ``{"point", "mask", "polygon"}``.
    data
        Payload whose interpretation depends on *kind*:

        * **``point``**   - *np.ndarray* (shape ``(n_dims,)``)
        * **``mask``**    - Boolean array matching ``centers_shape_``
        * **``polygon``** - :class:`shapely.geometry.Polygon`

    metadata
        Arbitrary key-value store forwarded from :pyfunc:`add_region`.
    """

    name: str
    kind: str  # "point" | "mask" | "polygon"
    data: Any
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"point", "mask", "polygon"}:
            raise ValueError(f"Unknown region kind: {self.kind}")
        if self.kind == "polygon" and not _HAS_SHAPELY:
            raise RuntimeError("Polygon regions require the 'shapely' package.")


def _point_in_polygon(
    points: np.ndarray, polygon: "_shp.Polygon"
) -> np.ndarray:  # noqa: D401
    """Return a boolean array telling which *points* lie inside *polygon*.

    Notes
    -----
    * Requires ``shapely``.
    """

    if not _HAS_SHAPELY:
        raise RuntimeError("Polygon support requested but Shapely is not installed.")
    try:
        return np.array(polygon.contains_points(points), bool)  # type: ignore[attr-defined]
    except AttributeError:
        return np.fromiter((polygon.contains(_shp.Point(xy)) for xy in points), bool)


@dataclass
class Environment:
    """
    Represents a spatial environment, using a pluggable GeometryEngine
    for discretization and graph topology.
    """

    environment_name: str = ""
    geometry_engine: Optional[GeometryEngine] = None  # Engine can be pre-configured

    engine_kind: Optional[str] = "Grid"  # Default engine kind
    engine_build_params: Dict[str, Any] = field(default_factory=dict)

    place_bin_centers_: Optional[NDArray[np.float64]] = field(init=False, default=None)
    edges_: Optional[Tuple[NDArray[np.float64], ...]] = field(init=False, default=None)
    centers_shape_: Optional[Tuple[int, ...]] = field(init=False, default=None)
    track_graph_nd_: Optional[nx.Graph] = field(init=False, default=None)
    track_graph_bin_centers_: Optional[nx.Graph] = field(init=False, default=None)
    interior_mask_: Optional[NDArray[np.bool_]] = field(init=False, default=None)
    position_range_: Optional[Sequence[Tuple[float, float]]] = field(
        init=False, default=None
    )

    _is_fitted: bool = field(init=False, default=False)
    _is_1d_env: Optional[bool] = field(init=False, default=None)
    _regions: Dict[str, "RegionInfo"] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if self.geometry_engine is None and self.engine_kind is None:
            raise ValueError(
                "Either a 'geometry_engine' instance or 'engine_kind' must be provided."
            )
        if self.geometry_engine is not None and self.engine_kind is not None:
            warnings.warn(
                "Both 'geometry_engine' and 'engine_kind' provided. "
                "The pre-configured 'geometry_engine' will be used.",
                UserWarning,
            )

    @property
    def is_1d(self) -> bool:
        """Checks if the environment is 1-dimensional."""
        if not self._is_fitted:
            # Before fitting, we can't be certain unless engine_kind implies it.
            # However, the definitive source is the engine itself.
            # If _is_1d_env was stored from a previous fit (e.g. after loading), use it.
            if self._is_1d_env is not None:
                return self._is_1d_env
            return (
                False  # Default assumption or could infer from engine_kind if desired
            )
        if self.geometry_engine is not None:
            return self.geometry_engine.is_1d
        if (
            self._is_1d_env is not None
        ):  # Fallback to stored value if engine is None post-fit (e.g. deserialization issue)
            return self._is_1d_env
        return False  # Should ideally not be reached if properly fitted/loaded

    def fit(
        self, position: Optional[NDArray[np.float64]] = None, **kwargs
    ) -> "Environment":
        """Fits the environment using its GeometryEngine.

        The GeometryEngine is responsible for creating the spatial discretization
        (bin centers, edges), graph topology, and interior mask.

        Parameters
        ----------
        position : Optional[NDArray[np.float64]], shape (n_time, n_dims), optional
            Position data, which may be used by the GeometryEngine for fitting.
            Defaults to None.
        **kwargs
            Additional keyword arguments to pass to the GeometryEngine's
            `build` method, overriding or supplementing `engine_build_params`.

        Returns
        -------
        Environment
            The fitted Environment instance.

        Raises
        ------
        RuntimeError
            If no geometry engine or engine kind is specified.
        """
        current_build_params = {**self.engine_build_params, **kwargs}
        if position is not None:
            current_build_params["position"] = position

        if self.geometry_engine is None:
            if not self.engine_kind:
                raise RuntimeError(
                    "No geometry engine or engine kind specified for fitting."
                )
            # make_engine is expected to return a *built* engine.
            self.geometry_engine = make_engine(self.engine_kind, **current_build_params)
        else:
            # Engine was pre-supplied. Call its build method with any new/overriding params.
            # This assumes the engine's build method is idempotent or can handle re-building.
            if (
                current_build_params or not self._is_fitted
            ):  # Build if new params or not yet fitted
                self.geometry_engine.build(**current_build_params)

        if self.geometry_engine is None:
            raise RuntimeError("Geometry engine could not be initialized or built.")

        # Pull canonical attributes from the built engine
        self.place_bin_centers_ = self.geometry_engine.place_bin_centers_
        self.edges_ = self.geometry_engine.edges_
        self.centers_shape_ = self.geometry_engine.centers_shape_
        self.track_graph_nd_ = self.geometry_engine.track_graph_nd_
        self.track_graph_bin_centers_ = self.geometry_engine.track_graph_bin_centers_
        self.interior_mask_ = self.geometry_engine.interior_mask_

        # Store the definitive 1D status from the engine
        self._is_1d_env = self.geometry_engine.is_1d

        # Position range might be an attribute of the engine, or derivable
        if hasattr(self.geometry_engine, "position_range_"):
            self.position_range_ = getattr(self.geometry_engine, "position_range_")
        elif self.edges_ and all(
            isinstance(e, np.ndarray) and e.size > 0 for e in self.edges_
        ):
            try:
                # Attempt to derive from edges if not directly provided by engine
                self.position_range_ = tuple((e[0], e[-1]) for e in self.edges_ if e.ndim == 1 and e.size >= 2)  # type: ignore
                if (
                    not self.position_range_
                    or len(self.position_range_) != self.place_bin_centers_.shape[1]
                ):
                    self.position_range_ = None  # Fallback if derivation is ambiguous
            except (IndexError, TypeError, AttributeError):
                self.position_range_ = None
        else:
            self.position_range_ = None

        self._is_fitted = True

        return self

    @check_fitted
    def get_fitted_track_graph(self) -> nx.Graph:
        """Returns the primary track graph after fitting.

        For 1D environments, this is typically `track_graph_bin_centers_`.
        For N-D environments, this is typically `track_graph_nd_`.
        The choice depends on the `GeometryEngine` implementation.

        Returns
        -------
        nx.Graph
            The fitted track graph.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        ValueError
            If no suitable graph is available from the geometry engine.
        """

        if self.is_1d:
            graph = self.track_graph_bin_centers_
            if graph is not None:
                return graph
            if self.track_graph_nd_ is not None:
                warnings.warn(
                    "1D environment using track_graph_nd_ as primary graph.",
                    UserWarning,
                )
                return self.track_graph_nd_
        else:  # N-D
            graph = self.track_graph_nd_
            if graph is not None:
                return graph
            if self.track_graph_bin_centers_ is not None:
                warnings.warn(
                    "N-D environment using track_graph_bin_centers_ as primary graph.",
                    UserWarning,
                )
                return self.track_graph_bin_centers_

        raise ValueError("No suitable graph is available from the geometry engine.")

    @cached_property
    @check_fitted
    def distance_between_bins(self) -> NDArray[np.float64]:
        """Shortest path distances between all pairs of bins in the fitted graph.

        Calculated using Dijkstra's algorithm on the graph returned by
        `get_fitted_track_graph()`.

        Returns
        -------
        NDArray[np.float64]
            A square matrix where element (i, j) is the shortest path distance
            between bin `i` and bin `j`. Shape is (n_total_bins, n_total_bins).

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        """
        return _get_distance_between_bins(self.get_fitted_track_graph())

    @check_fitted
    def get_bin_ind(self, positions: NDArray[np.float64]) -> NDArray[np.int_]:
        """Converts continuous spatial positions to discrete bin indices.

        Delegates to the `point_to_bin` method of the `GeometryEngine`.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            The spatial positions to be binned.

        Returns
        -------
        NDArray[np.int_]
            The flat indices of the bins corresponding to each input position.
            Shape is (n_samples,). Returns -1 for positions outside mapped areas.

        Raises
        ------
        RuntimeError
            If the environment or geometry engine is not fitted/initialized.
        """
        return self.geometry_engine.point_to_bin(positions)

    @check_fitted
    def get_bin_center_dataframe(self) -> pd.DataFrame:
        """Creates a DataFrame with information about each bin center.

        The DataFrame includes node ID, spatial dimensions ('pos_dim0', etc.),
        and other attributes from the graph nodes like 'is_track_interior',
        'bin_ind', 'bin_ind_flat', and 'edge_id'.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by 'node_id', containing bin center information.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        """
        graph = self.get_fitted_track_graph()
        if graph.number_of_nodes() == 0:
            # Define columns for an empty DataFrame for consistency
            cols = [
                "pos_dim0",  # Default, may add more based on typical dimensionality
                "is_track_interior",
                "bin_ind",
                "bin_ind_flat",
                "edge_id",
            ]
            # Attempt to determine number of position dimensions if possible
            if (
                self.place_bin_centers_ is not None
                and self.place_bin_centers_.ndim == 2
            ):
                n_dims = self.place_bin_centers_.shape[1]
                cols = [f"pos_dim{i}" for i in range(n_dims)] + cols[1:]

            return pd.DataFrame(columns=cols).set_index(pd.Index([], name="node_id"))

        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "node_id"

        # Extract multi-dimensional 'pos' into separate columns
        if "pos" in df.columns and not df["pos"].dropna().empty:
            first_pos_val = df["pos"].dropna().iloc[0]
            if isinstance(first_pos_val, (list, tuple, np.ndarray)):
                n_dims = len(first_pos_val)
                pos_cols = [f"pos_dim{i}" for i in range(n_dims)]
                pos_df = pd.DataFrame(
                    df["pos"].tolist(), index=df.index, columns=pos_cols
                )
                df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        # Ensure essential columns exist, providing defaults if necessary
        default_bin_ind_flat = (
            df.index.to_series().astype(int)  # Assuming node_id is the flat index
            if df.index.is_integer()
            else pd.Series(range(len(df)), index=df.index, dtype=int)
        )

        # Define columns and their defaults or sources
        # Order matters for how they might appear if created
        expected_cols_defaults = {
            "is_track_interior": True,  # Default to True if not specified by node
            "bin_ind": -1,  # Default, typically overridden by graph node data
            "bin_ind_flat": default_bin_ind_flat,
            "edge_id": -1,  # Default, may be overridden by graph node data
        }
        # Ensure position columns exist if not created from 'pos'
        if self.place_bin_centers_ is not None and self.place_bin_centers_.ndim == 2:
            for i in range(self.place_bin_centers_.shape[1]):
                if f"pos_dim{i}" not in df.columns:
                    # This case should be rare if 'pos' was handled, but as a fallback:
                    if self.place_bin_centers_.shape[0] == len(df):
                        df[f"pos_dim{i}"] = self.place_bin_centers_[:, i]
                    else:  # Cannot safely map, fill with NaN
                        df[f"pos_dim{i}"] = np.nan

        for col, default_val_source in expected_cols_defaults.items():
            if col not in df.columns:
                if callable(
                    default_val_source
                ):  # For default_bin_ind_flat if it was a function
                    df[col] = default_val_source()
                elif isinstance(default_val_source, pd.Series):
                    df[col] = default_val_source
                else:
                    df[col] = default_val_source

        if "bin_ind_flat" in df.columns:
            # Attempt to convert to int, coercing errors to NaN then -1
            # This handles cases where bin_ind_flat might be missing or not numeric
            df["bin_ind_flat"] = (
                pd.to_numeric(df["bin_ind_flat"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
            df = df.sort_values(by="bin_ind_flat")
        else:  # Should not happen if defaults are set correctly
            warnings.warn(
                "'bin_ind_flat' column missing after DataFrame construction.",
                RuntimeWarning,
            )

        return df

    @check_fitted
    def get_manifold_distances(
        self, positions1: NDArray[np.float64], positions2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Computes shortest path distances between pairs of positions on the track.

        Parameters
        ----------
        positions1 : NDArray[np.float64], shape (n_samples, n_dims) or (n_dims,)
            The first set of spatial positions.
        positions2 : NDArray[np.float64], shape (n_samples, n_dims) or (n_dims,)
            The second set of spatial positions. Must have the same shape as `positions1`.

        Returns
        -------
        NDArray[np.float64]
            An array of shortest path distances. Shape is (n_samples,).
            Returns `np.inf` for pairs where one or both positions are
            unmapped (-1 bin index) or if bins are disconnected.

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        ValueError
            If input shapes of `positions1` and `positions2` do not match.
        """
        p1 = np.atleast_2d(positions1)
        p2 = np.atleast_2d(positions2)
        if p1.shape != p2.shape:
            raise ValueError("Shape mismatch between positions1 and positions2.")
        if p1.shape[0] == 0:
            return np.array([], dtype=np.float64)

        bin1 = self.get_bin_ind(p1)
        bin2 = self.get_bin_ind(p2)

        dist_matrix = self.distance_between_bins
        n_bins_in_matrix = dist_matrix.shape[0]

        # Initialize distances to infinity
        distances = np.full(len(p1), np.inf, dtype=np.float64)

        # Identify valid bin indices (not -1 and within matrix bounds)
        valid_mask = (
            (bin1 != -1)
            & (bin2 != -1)
            & (bin1 < n_bins_in_matrix)
            & (bin2 < n_bins_in_matrix)
            & (bin1 >= 0)
            & (bin2 >= 0)  # Ensure non-negative for safety
        )

        if np.any(valid_mask):
            valid_bin1 = bin1[valid_mask]
            valid_bin2 = bin2[valid_mask]
            distances[valid_mask] = dist_matrix[valid_bin1, valid_bin2]

        # Warn if any original bins were out of bounds for the distance matrix
        # (get_bin_ind might return indices valid for place_bin_centers_, but dist_matrix
        # is based on graph nodes, which might be a subset if interior_mask was used).
        # The `distance_between_bins` from `_get_distance_between_bins` should be indexed by graph node IDs if these are 0..N-1.
        # The current implementation of `_get_distance_between_bins` already maps to `bin_ind_flat`.
        if np.any((bin1 != -1) & ((bin1 >= n_bins_in_matrix) | (bin1 < 0))) or np.any(
            (bin2 != -1) & ((bin2 >= n_bins_in_matrix) | (bin2 < 0))
        ):
            warnings.warn(
                "Some valid bin indices from get_bin_ind were outside the "
                "bounds of the distance_between_bins matrix. This may indicate "
                "a mismatch between all place bin centers and the nodes in the graph "
                "used for distances (e.g. due to interior_mask). Distances for these "
                "pairs will be inf.",
                RuntimeWarning,
            )

        return distances.squeeze()

    @check_fitted
    def get_linear_position(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculates linearized position along the track for 1D environments.

        Delegates to the `get_linearized_position` method of the
        `GeometryEngine` if available.

        Parameters
        ----------
        position : NDArray[np.float64], shape (n_samples, n_dims)
            The spatial positions.

        Returns
        -------
        NDArray[np.float64]
            The linearized positions. Shape is (n_samples,).

        Raises
        ------
        RuntimeError
            If the environment has not been fitted.
        TypeError
            If the environment is not 1-dimensional.
        AttributeError
            If the fitted `GeometryEngine` does not support linearization
            (i.e., lacks a `get_linearized_position` method).
        """
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted. Call `fit` first.")
        if not self.is_1d:
            raise TypeError("Linear position is only available for 1D environments.")
        if self.geometry_engine is None:
            raise RuntimeError("Geometry engine not available.")

        if hasattr(self.geometry_engine, "get_linearized_position"):
            # Assume the engine's method has the correct signature
            return self.geometry_engine.get_linearized_position(position)  # type: ignore
        else:
            raise AttributeError(
                f"The current geometry engine ({self.geometry_engine.__class__.__name__}) "
                "does not support 'get_linearized_position'."
            )

    @check_fitted
    def plot_grid(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the environment geometry by delegating to the GeometryEngine.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Existing Matplotlib axes to plot on. If None, new axes are created.
            Defaults to None.
        **kwargs
            Additional keyword arguments to be passed to the
            GeometryEngine's `plot` method.

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib axes object used for plotting.

        Raises
        ------
        RuntimeError
            If the environment or geometry engine has not been fitted/initialized.
        """
        ax = self.geometry_engine.plot(ax=ax, **kwargs)

        if self.environment_name and ax.get_title() == "":
            ax.set_title(
                f"Environment: {self.environment_name} ({self.geometry_engine.__class__.__name__})"
            )
        elif (
            self.environment_name
            and self.geometry_engine.__class__.__name__ not in ax.get_title()
        ):
            current_title = ax.get_title()
            ax.set_title(f"{self.environment_name} - {current_title}")

        return ax

    def save(self, filename: str = "environment.pkl") -> None:
        """Saves the Environment object to a file using pickle.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the environment to.
            Defaults to "environment.pkl".
        """
        with open(filename, "wb") as file_handle:
            pickle.dump(self, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Environment saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> "Environment":
        """Loads an Environment object from a pickled file.

        Parameters
        ----------
        filename : str
            The path to the file containing the pickled Environment object.

        Returns
        -------
        Environment
            The loaded Environment object.
        """
        with open(filename, "rb") as file_handle:
            environment = pickle.load(file_handle)
        if not isinstance(environment, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        return environment

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Environment object to a dictionary.

        This can be useful for saving the environment's configuration and
        fitted state in formats other than pickle, e.g., for databases.
        The `geometry_engine` itself is not directly serialized but is
        reconstructed via `engine_kind` and `engine_build_params` upon
        deserialization if needed.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the Environment.
        """
        data: Dict[str, Any] = {
            "__classname__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "environment_name": self.environment_name,
            "engine_kind": self.engine_kind,
            # Ensure engine_build_params are serializable (e.g. convert graphs to node_link_data if passed in params)
            "engine_build_params": self.engine_build_params,  # Caller's responsibility if these params are complex
            "_is_fitted": self._is_fitted,
            "_is_1d_env": self._is_1d_env,
        }

        if self._is_fitted:
            # Attributes derived from the engine
            data["place_bin_centers_"] = (
                self.place_bin_centers_.tolist()
                if self.place_bin_centers_ is not None
                else None
            )
            data["edges_"] = (
                [e.tolist() for e in self.edges_ if isinstance(e, np.ndarray)]
                if self.edges_ is not None
                else None
            )
            data["centers_shape_"] = self.centers_shape_
            data["interior_mask_"] = (
                self.interior_mask_.tolist()
                if self.interior_mask_ is not None
                else None
            )
            data["position_range_"] = self.position_range_

            # Graphs are stored as node-link data
            if self.track_graph_nd_ is not None:
                data["track_graph_nd_"] = nx.node_link_data(self.track_graph_nd_)
            else:
                data["track_graph_nd_"] = None

            if self.track_graph_bin_centers_ is not None:
                data["track_graph_bin_centers_"] = nx.node_link_data(
                    self.track_graph_bin_centers_
                )
            else:
                data["track_graph_bin_centers_"] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Environment":
        """Deserializes an Environment object from a dictionary.

        Reconstructs the `geometry_engine` if `engine_kind` and
        `engine_build_params` are provided in the dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary representation of the Environment, typically
            created by the `to_dict` method.

        Returns
        -------
        Environment
            The deserialized Environment object.

        Raises
        ------
        ValueError
            If the dictionary does not represent this Environment class.
        """
        if (
            data.get("__classname__") != cls.__name__
            or data.get("__module__") != cls.__module__
        ):
            raise ValueError(
                f"Dictionary is not for this Environment class ({cls.__module__}.{cls.__name__})."
            )

        env = cls(
            environment_name=data["environment_name"],
            engine_kind=data.get("engine_kind"),
            engine_build_params=data.get("engine_build_params", {}),
        )

        env._is_fitted = data.get("_is_fitted", False)
        env._is_1d_env = data.get("_is_1d_env")

        if env._is_fitted:
            # Attempt to reconstruct the geometry_engine if kind and params are available
            # This assumes make_engine uses these params to return a *built* engine.
            if env.engine_kind and env.engine_build_params is not None:
                try:
                    # If engine_build_params contained complex objects that were serialized
                    # (e.g. a graph as node_link_data), they would need deserialization here
                    # before passing to make_engine. Current `make_engine` expects live objects.
                    # This example assumes engine_build_params are directly usable.
                    env.geometry_engine = make_engine(
                        env.engine_kind, **env.engine_build_params
                    )
                except Exception as e:  # pylint: disable=broad-except
                    warnings.warn(
                        f"Could not reconstruct geometry engine of kind '{env.engine_kind}' "
                        f"during from_dict: {e}. Engine-dependent methods might not work. "
                        "Restoring attributes directly from dict.",
                        RuntimeWarning,
                    )
                    env.geometry_engine = (
                        None  # Ensure it's None if reconstruction fails
                    )
            elif env.geometry_engine is None:  # If no engine_kind to reconstruct from
                warnings.warn(
                    "Fitted environment loaded from dict, but no engine_kind specified. "
                    "Engine-dependent methods (like get_bin_ind, plot_grid) will fail "
                    "unless geometry_engine is manually set and built, or attributes are "
                    "sufficient for other methods.",
                    RuntimeWarning,
                )

            # Restore attributes directly from the dictionary.
            # These might overwrite attributes set by a reconstructed engine if the dict
            # represents a state saved *after* an engine run.
            env.place_bin_centers_ = (
                np.array(data["place_bin_centers_"])
                if data.get("place_bin_centers_") is not None
                else None
            )
            env.edges_ = (
                tuple(np.array(e) for e in data["edges_"])
                if data.get("edges_") is not None
                else None
            )
            env.centers_shape_ = data.get("centers_shape_")
            env.interior_mask_ = (
                np.array(data["interior_mask_"])
                if data.get("interior_mask_") is not None
                else None
            )
            env.position_range_ = data.get("position_range_")

            if data.get("track_graph_nd_") is not None:
                env.track_graph_nd_ = nx.node_link_graph(data["track_graph_nd_"])
            else:
                env.track_graph_nd_ = None

            if data.get("track_graph_bin_centers_") is not None:
                env.track_graph_bin_centers_ = nx.node_link_graph(
                    data["track_graph_bin_centers_"]
                )
            else:
                env.track_graph_bin_centers_ = None

            # If engine was reconstructed, its attributes should match these.
            # If engine could not be reconstructed, these restored attributes allow some functionality.
            # We also need to ensure _is_1d_env is correctly set based on the loaded engine if possible.
            if env.geometry_engine:
                env._is_1d_env = env.geometry_engine.is_1d
            # Else, it relies on the _is_1d_env from the dictionary.

        return env

    @check_fitted
    def add_region(
        self: Environment,
        name: str,
        *,
        point: Tuple[float, ...] | None = None,
        mask: np.ndarray | None = None,
        polygon: "_shp.Polygon | list[Tuple[float,float]]" | None = None,
        **metadata,
    ):
        """Register a region *name* using one of three specifiers.

        Exactly **one** of ``point``, ``mask`` or ``polygon`` must be given.

        Parameters
        ----------
        name
            Unique identifier for the region.
        point
            Physical coordinates (same dimensionality as positions).  Faster
            when you only need a single bin (e.g. reward well).
        mask
            Boolean array of shape ``env.centers_shape_`` - useful when you
            already computed a mask elsewhere.
        polygon
            *Shapely* polygon or list of ``(x, y)`` vertices for 2-D engines
            - lets you delineate irregular zones (start box, wings, etc.).
        metadata
            Additional keyword pairs stored verbatim in :attr:`RegionInfo.metadata`.
        """
        if sum(v is not None for v in (point, mask, polygon)) != 1:
            raise ValueError("Must provide exactly one of point / mask / polygon.")
        if name in self._regions:
            raise ValueError(f"Region '{name}' already exists.")

        # Determine kind + data storage
        if point is not None:
            kind, data = "point", np.asarray(point, float)
        elif mask is not None:
            if mask.shape != self.centers_shape_:
                raise ValueError("Mask shape mismatches environment grid.")
            kind, data = "mask", mask.astype(bool)
        else:  # polygon given
            if not _HAS_SHAPELY:
                raise RuntimeError("Install 'shapely' to use polygon regions.")
            if isinstance(polygon, list):  # coordinates → Polygon
                polygon = _shp.Polygon(polygon)
            if not isinstance(polygon, _shp.Polygon):
                raise TypeError(
                    "polygon must be a shapely.geometry.Polygon or list of coords."
                )
            kind, data = "polygon", polygon

        self._regions[name] = RegionInfo(
            name=name, kind=kind, data=data, metadata=metadata
        )

    @check_fitted
    def remove_region(self: Environment, name: str):
        """Remove *name* from the registry (silently ignored if absent)."""
        self._regions.pop(name, None)

    def list_regions(self: Environment) -> List[str]:
        """Return a list of registered region names (in insertion order)."""
        return list(self._regions.keys())

    @check_fitted
    def region_mask(self: Environment, name: str) -> np.ndarray:
        """Boolean occupancy mask for *name*.

        The returned array has the same shape as ``env.centers_shape_``.
        """
        info = self._regions[name]
        if info.kind == "mask":
            return info.data.copy()

        # flat mask initialised false
        flat = np.zeros(np.prod(self.centers_shape_), bool)

        if info.kind == "point":
            idx = self.get_bin_ind(info.data)
            flat[idx] = True
        else:  # polygon
            pts = self.place_bin_centers_[:, :2]  # xy only for test
            inside = _point_in_polygon(pts, info.data)
            flat[inside] = True

        return flat.reshape(self.centers_shape_)

    @check_fitted
    def bins_in_region(self: Environment, name: str) -> np.ndarray:
        """Return flattened indices of bins inside *name*."""
        return np.flatnonzero(self.region_mask(name))

    @check_fitted
    def region_center(self: Environment, name: str) -> np.ndarray:
        """Geometric center of the region.

        * For ``point`` regions, this simply returns the point.
        * For ``mask`` / ``polygon``, returns the mean of all bin centers.

        Parameters
        ----------
        name
            Name of the region.
        Returns
        -------
        np.ndarray
            Geometric center of the region.

        """
        info = self._regions[name]
        if info.kind == "point":
            return info.data
        return self.place_bin_centers_[self.bins_in_region(name)].mean(axis=0)

    @check_fitted
    def nearest_region(self: Environment, position: np.ndarray) -> str | None:
        """Nearest region (Euclidean) to *position*.

        Parameters
        ----------
        position
            Array of shape ``(n_dims,)`` or ``(n_samples, n_dims)``.
        Returns
        -------
        str | None
            Region name with minimal mean distance, or ``None`` if no regions
            are registered.
        """
        pos = np.atleast_2d(position)
        best_name, best_d = None, np.inf
        for name in self.list_regions():
            c = self.region_center(name)
            d = np.linalg.norm(pos - c, axis=1).mean()
            if d < best_d:
                best_name, best_d = name, d
        return best_name
