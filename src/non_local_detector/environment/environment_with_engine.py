from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from non_local_detector.environment.environment import (
    _get_distance_between_bins,
    get_linearized_position,
)
from non_local_detector.environment.geometry_engines import GeometryEngine, make_engine


@dataclass
class Environment:
    """
    Represents a spatial environment, using a pluggable GeometryEngine
    for discretization and graph topology.
    """

    environment_name: str = ""
    geometry_engine: Optional[GeometryEngine] = None  # Engine can be pre-configured

    engine_kind: Optional[str] = None
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
        if not self._is_fitted or self._is_1d_env is None:
            if self.geometry_engine is not None:
                # Heuristic: if track_graph_bin_centers_ is primary and track_graph_nd_ is None
                # A more robust way is if engines have an 'is_1d' property or by checking engine type.
                # from non_local_detector.environment.geometry_engines import TrackGraphEngine # Avoid circular import
                # if isinstance(self.geometry_engine, TrackGraphEngine): self._is_1d_env = True
                if (
                    self.track_graph_bin_centers_ is not None
                    and self.track_graph_nd_ is None
                ):
                    self._is_1d_env = True
                else:
                    self._is_1d_env = False
            else:  # Not fitted yet, or engine not determined
                return False  # Default assumption
        return self._is_1d_env

    def fit(
        self, position: Optional[NDArray[np.float64]] = None, **kwargs
    ) -> "Environment":
        current_build_params = {**self.engine_build_params, **kwargs}
        if position is not None:  # Ensure 'position' is in build_params if provided
            current_build_params["position"] = position

        if self.geometry_engine is None:
            if not self.engine_kind:
                raise RuntimeError(
                    "No geometry engine or engine kind specified for fitting."
                )
            self.geometry_engine = make_engine(self.engine_kind, **current_build_params)
            # Assuming make_engine returns a *built* engine.
        else:
            # Engine was pre-supplied. Call its build method with any new/overriding params.
            # This assumes the engine's build method is idempotent or can handle re-building.
            if current_build_params:  # Only call build if there are params
                self.geometry_engine.build(**current_build_params)

        # Pull canonical attributes from the built engine
        self.place_bin_centers_ = self.geometry_engine.place_bin_centers_
        self.edges_ = self.geometry_engine.edges_
        self.centers_shape_ = self.geometry_engine.centers_shape_
        self.track_graph_nd_ = self.geometry_engine.track_graph_nd_
        self.track_graph_bin_centers_ = self.geometry_engine.track_graph_bin_centers_

        if (
            hasattr(self.geometry_engine, "interior_mask_")
            and self.geometry_engine.interior_mask_ is not None
        ):
            self.interior_mask_ = self.geometry_engine.interior_mask_
        elif self.place_bin_centers_ is not None and self.centers_shape_ is not None:
            if len(self.centers_shape_) > 1:  # Grid-like
                self.interior_mask_ = np.ones(self.centers_shape_, dtype=bool)
            else:  # List-like
                self.interior_mask_ = np.ones(
                    self.place_bin_centers_.shape[0], dtype=bool
                )

        # Determine _is_1d_env after attributes are set
        if self.track_graph_bin_centers_ is not None and self.track_graph_nd_ is None:
            self._is_1d_env = True
        else:
            self._is_1d_env = False

        if (
            hasattr(self.geometry_engine, "position_range_")
            and self.geometry_engine.position_range_ is not None
        ):
            self.position_range_ = self.geometry_engine.position_range_
        elif self.edges_ and all(
            isinstance(e, np.ndarray) and e.size > 0 for e in self.edges_
        ):
            try:
                self.position_range_ = tuple((e[0], e[-1]) for e in self.edges_)  # type: ignore
                if len(self.position_range_) != len(self.edges_):
                    self.position_range_ = None
            except (IndexError, TypeError):
                self.position_range_ = None

        self._is_fitted = True
        return self

    def get_fitted_track_graph(self) -> nx.Graph:
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")
        graph = (
            self.track_graph_nd_
            if self.track_graph_nd_ is not None
            else self.track_graph_bin_centers_
        )
        if graph is None:
            raise ValueError("No suitable graph is available from the geometry engine.")
        return graph

    @cached_property
    def distance_between_bins(self) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fitted yet. Call `fit` first.")
        return _get_distance_between_bins(self.get_fitted_track_graph())

    def get_bin_ind(self, positions: NDArray[np.float64]) -> NDArray[np.int_]:
        if not self._is_fitted or self.geometry_engine is None:
            raise RuntimeError("Environment or geometry engine not fitted/initialized.")
        return self.geometry_engine.point_to_bin(positions)

    def get_bin_center_dataframe(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Environment has not been fit yet. Call `fit` first.")
        graph = self.get_fitted_track_graph()
        if graph.number_of_nodes() == 0:
            return pd.DataFrame(
                columns=[
                    "pos_dim0",
                    "pos_dim1",
                    "is_track_interior",
                    "bin_ind",
                    "bin_ind_flat",
                    "edge_id",
                ]
            ).set_index(pd.Index([], name="node_id"))

        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "node_id"

        if "pos" in df.columns and not df["pos"].dropna().empty:
            first_pos_val = df["pos"].dropna().iloc[0]
            if isinstance(first_pos_val, (list, tuple)):
                n_dims = len(first_pos_val)
                pos_cols = [f"pos_dim{i}" for i in range(n_dims)]
                pos_df = pd.DataFrame(
                    df["pos"].tolist(), index=df.index, columns=pos_cols
                )
                df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        default_bin_ind_flat = (
            df.index.to_series()
            if df.index.name == "node_id" and df.index.is_integer()
            else pd.Series(range(len(df)), index=df.index)
        )
        for col, default_val_source in [
            ("is_track_interior", True),
            ("bin_ind", -1),
            ("bin_ind_flat", default_bin_ind_flat),
            ("edge_id", -1),
        ]:
            if col not in df.columns:
                df[col] = (
                    default_val_source
                    if not callable(default_val_source)
                    else default_val_source()
                )

        if "bin_ind_flat" in df.columns:
            df = df.sort_values(by="bin_ind_flat")
        return df

    def get_manifold_distances(
        self, positions1: NDArray[np.float64], positions2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise RuntimeError("Fit first.")
        p1 = np.atleast_2d(positions1)
        p2 = np.atleast_2d(positions2)
        if p1.shape != p2.shape:
            raise ValueError("Shape mismatch.")
        if p1.shape[0] == 0:
            return np.array([], dtype=np.float64)

        bin1 = self.get_bin_ind(p1)
        bin2 = self.get_bin_ind(p2)
        valid = (bin1 != -1) & (bin2 != -1)
        dist = np.full(len(p1), np.inf)
        if np.any(valid):
            max_idx = self.distance_between_bins.shape[0] - 1
            vb1 = np.clip(bin1[valid], 0, max_idx)
            vb2 = np.clip(bin2[valid], 0, max_idx)
            if not np.array_equal(vb1, bin1[valid]) or not np.array_equal(
                vb2, bin2[valid]
            ):
                warnings.warn(
                    "Bin indices clipped for distance_between_bins.", RuntimeWarning
                )
            dist[valid] = self.distance_between_bins[vb1, vb2]
        return dist.squeeze()

    def get_linear_position(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise RuntimeError("Fit first.")
        if not self.is_1d:
            raise ValueError("Only for 1D environments.")

        # Assumes these params were stored in engine_build_params for TrackGraphEngine
        # Or that the engine itself can provide them.
        # This is a simplification.
        build_params = self.engine_build_params
        if self.geometry_engine and hasattr(
            self.geometry_engine, "engine_build_params_for_1d"
        ):  # Ideal
            build_params = self.geometry_engine.engine_build_params_for_1d  # type: ignore

        track_graph = build_params.get("track_graph")
        edge_order = build_params.get("edge_order")
        edge_spacing = build_params.get("edge_spacing", 0.0)  # Default if not found

        if track_graph is None or edge_order is None:
            raise ValueError("1D track parameters (track_graph, edge_order) not found.")

        return get_linearized_position(
            position, track_graph, edge_order=edge_order, edge_spacing=edge_spacing
        )

    def plot_grid(
        self, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
    ) -> matplotlib.axes.Axes:
        if not self._is_fitted:
            raise RuntimeError("Fit first.")
        if self.is_1d:
            if self.track_graph_bin_centers_ is None:
                raise ValueError("1D graph not set.")
            if ax is None:
                _, ax = plt.subplots(figsize=(15, 2.5))
            # Simplified 1D plot
            if (
                self.place_bin_centers_ is not None
                and self.place_bin_centers_.shape[1] == 1
            ):
                y_coords = np.zeros_like(self.place_bin_centers_[:, 0])
                mask = (
                    self.interior_mask_.ravel()
                    if self.interior_mask_ is not None
                    else np.ones(len(y_coords), dtype=bool)
                )
                ax.scatter(
                    self.place_bin_centers_[mask, 0], y_coords[mask], c="blue", s=20
                )
                if self.interior_mask_ is not None and np.any(~mask):
                    ax.scatter(
                        self.place_bin_centers_[~mask, 0],
                        y_coords[~mask],
                        c="red",
                        s=20,
                        marker="x",
                    )
            if self.edges_ and self.edges_[0] is not None:
                for edge_pos in self.edges_[0]:
                    ax.axvline(edge_pos, lw=0.5, c="k", ls=":")
            ax.set_title(f"{self.environment_name} (1D)")
            ax.set_xlabel("Linearized Position")
            ax.set_yticks([])
            ax.set_ylim(-0.5, 0.5)
        elif (
            self.edges_
            and len(self.edges_) == 2
            and self.centers_shape_
            and len(self.centers_shape_) == 2
        ):  # 2D Grid
            if ax is None:
                _, ax = plt.subplots(figsize=(7, 7))
            if self.interior_mask_ is not None:
                ax.pcolormesh(
                    self.edges_[0],
                    self.edges_[1],
                    self.interior_mask_.T,
                    cmap="bone_r",
                    alpha=0.7,
                    shading="auto",
                    **kwargs,
                )
            ax.set_xticks(self.edges_[0])
            ax.set_yticks(self.edges_[1])
            ax.grid(True, ls="-", lw=0.5, c="gray")
            ax.set_aspect("equal")
            ax.set_title(f"{self.environment_name} (2D Grid)")
            ax.set_xlabel("Dim 0")
            ax.set_ylabel("Dim 1")
            if self.position_range_:
                ax.set_xlim(self.position_range_[0])
                ax.set_ylim(self.position_range_[1])
        else:  # General N-D scatter
            if ax is None:
                is_3d = (
                    self.place_bin_centers_ is not None
                    and self.place_bin_centers_.shape[1] == 3
                )
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d" if is_3d else None)  # type: ignore
            if self.place_bin_centers_ is not None:
                pts = self.place_bin_centers_
                mask = (
                    self.interior_mask_.ravel()
                    if self.interior_mask_ is not None
                    else np.ones(len(pts), dtype=bool)
                )
                active_pts = pts[mask]
                if active_pts.shape[1] == 2:
                    ax.scatter(active_pts[:, 0], active_pts[:, 1], s=10, **kwargs)
                elif active_pts.shape[1] == 3 and hasattr(ax, "scatter3D"):
                    ax.scatter3D(active_pts[:, 0], active_pts[:, 1], active_pts[:, 2], s=10, **kwargs)  # type: ignore
            graph = self.get_fitted_track_graph()
            if graph and self.place_bin_centers_ is not None:
                node_coords = {
                    n: self.place_bin_centers_[n]
                    for n in graph.nodes()
                    if n < len(self.place_bin_centers_)
                }
                if node_coords:
                    nx.draw_networkx_edges(graph, pos=node_coords, ax=ax, alpha=0.3)
            ax.set_title(f"{self.environment_name} (N-D)")
        return ax

    def save(self, filename: str = "environment.pkl") -> None:
        with open(filename, "wb") as file_handle:
            pickle.dump(self, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Environment saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> "Environment":
        with open(filename, "rb") as file_handle:
            environment = pickle.load(file_handle)
        return environment

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Environment object to a dictionary."""
        data: Dict[str, Any] = {
            "__classname__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "environment_name": self.environment_name,
            "engine_kind": self.engine_kind,
            "engine_build_params": self.engine_build_params,  # May need deepcopy or custom serialization
            "_is_fitted": self._is_fitted,
            "_is_1d_env": self._is_1d_env,
        }
        if self._is_fitted:
            data["place_bin_centers_"] = (
                self.place_bin_centers_.tolist()
                if self.place_bin_centers_ is not None
                else None
            )
            data["edges_"] = (
                [e.tolist() for e in self.edges_] if self.edges_ is not None else None
            )
            data["centers_shape_"] = self.centers_shape_
            data["interior_mask_"] = (
                self.interior_mask_.tolist()
                if self.interior_mask_ is not None
                else None
            )
            data["position_range_"] = self.position_range_

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
        """Deserializes an Environment object from a dictionary."""
        if (
            data.get("__classname__") != cls.__name__
            or data.get("__module__") != cls.__module__
        ):
            raise ValueError("Dictionary is not for this Environment class.")

        # Create instance with engine kind and params (engine itself is not built yet)
        env = cls(
            environment_name=data["environment_name"],
            engine_kind=data.get(
                "engine_kind"
            ),  # Use .get for safety if key might be missing
            engine_build_params=data.get(
                "engine_build_params", {}
            ),  # Default to empty dict
        )

        env._is_fitted = data.get("_is_fitted", False)
        env._is_1d_env = data.get("_is_1d_env")  # Can be None if not fitted

        if env._is_fitted:
            # If fitted, the engine should have been built.
            # We need to reconstruct the engine instance and then assign attributes.
            # This assumes make_engine uses engine_build_params to return a *built* engine.
            if env.engine_kind and env.engine_build_params is not None:
                try:
                    env.geometry_engine = make_engine(
                        env.engine_kind, **env.engine_build_params
                    )
                except Exception as e:
                    warnings.warn(
                        f"Could not reconstruct geometry engine of kind '{env.engine_kind}' during from_dict: {e}. "
                        "Environment attributes will be set from dict, but engine methods might not work.",
                        RuntimeWarning,
                    )
                    env.geometry_engine = (
                        None  # Ensure it's None if reconstruction fails
                    )
            else:
                warnings.warn(
                    "Engine kind or build params missing for a fitted environment in from_dict. "
                    "Engine-dependent methods might not work.",
                    RuntimeWarning,
                )
                env.geometry_engine = None

            # Restore attributes directly from the dictionary
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

        return env
