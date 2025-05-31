"""
CompositeEnvironment: merges multiple Environment instances into a single unified Environment-like API.
Bridge edges between sub-environments are inferred automatically via mutual-nearest-neighbor (MNN).

This class exposes the same public interface as the base `Environment` class:
  - Properties: n_dims, n_bins, bin_centers, connectivity, is_1d, dimension_ranges,
                grid_edges, grid_shape, active_mask, regions
  - Methods:    bin_at, contains, neighbors, distance_between, bin_center_of,
                bin_attributes, edge_attributes

(Note: serialization methods such as save/load and factory methods like from_layout are not included,
since CompositeEnvironment wraps pre-fitted sub-environments.)
"""

from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from non_local_detector.environment.environment import Environment


class CompositeEnvironment:
    """
    A composite environment that merges multiple child Environment instances into one.

    It automatically infers “bridge” edges between every pair of sub-environments by finding
    mutually nearest neighbor bin-centers (MNN). It then presents the same interface as
    the base `Environment` class.
    """

    def __init__(
        self,
        subenvs: List[Environment],
        auto_bridge: bool = True,
        max_mnn_distance: Optional[float] = None,
    ):
        """
        Build a CompositeEnvironment from a list of pre-fitted Environment instances.

        Parameters
        ----------
        subenvs : List[Environment]
            A list of fitted Environment objects. All must share the same n_dims.
        auto_bridge : bool, default=True
            If True, automatically infer “bridge edges” between each pair of sub-environments
            using a mutual nearest-neighbor heuristic on their bin_centers.
        max_mnn_distance : Optional[float]
            If provided, any automatically inferred bridge whose Euclidean distance exceeds
            this threshold is discarded. If None, no distance filtering is applied.
        """
        if len(subenvs) == 0:
            raise ValueError("At least one sub-environment is required.")

        # Validate that all sub-environments share the same n_dims
        self._n_dims = subenvs[0].n_dims
        for e in subenvs:
            if e.n_dims != self._n_dims:
                raise ValueError("All sub-environments must share the same n_dims.")

        # Build index offsets for each sub-environment
        self._subenvs_info = []
        offset = 0
        for e in subenvs:
            n_bins = e.bin_centers.shape[0]
            self._subenvs_info.append(
                {"env": e, "start_idx": offset, "end_idx": offset + n_bins - 1}
            )
            offset += n_bins
        self._total_bins = offset

        # Stack all bin_centers into one array of shape (N_total, n_dims)
        self.bin_centers = np.vstack([e.bin_centers for e in subenvs])

        # Build the composite connectivity graph (nodes only for now)
        self.connectivity = nx.Graph()
        self.connectivity.add_nodes_from(range(self._total_bins))

        # Add each sub-environment’s edges, reindexed by offset
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            for u, v, data in env_i.connectivity.edges(data=True):
                self.connectivity.add_edge(u + base, v + base, **data)

        # Infer MNN-based bridges if requested
        self._bridge_list: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
        if auto_bridge:
            self._infer_mnn_bridges(max_mnn_distance)

        # Properties to match Environment interface
        self.is_1d = False
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.regions = {}

    def _add_bridge_edge(
        self, i_env: int, i_bin: int, j_env: int, j_bin: int, w: float
    ):
        """
        Add a bridge edge between bin i_bin of sub-environment i_env and bin j_bin of sub-environment j_env,
        with weight w. Raises ValueError if indices are out-of-range.
        """
        n_sub = len(self._subenvs_info)
        if not (0 <= i_env < n_sub) or not (0 <= j_env < n_sub):
            raise ValueError(f"Invalid sub-environment indices: {i_env}, {j_env}")

        block_i = self._subenvs_info[i_env]
        block_j = self._subenvs_info[j_env]
        max_i = block_i["end_idx"] - block_i["start_idx"]
        max_j = block_j["end_idx"] - block_j["start_idx"]
        if not (0 <= i_bin <= max_i) or not (0 <= j_bin <= max_j):
            raise ValueError(f"Bin index out-of-range for bridge: {i_bin}/{j_bin}")

        u = block_i["start_idx"] + i_bin
        v = block_j["start_idx"] + j_bin
        self.connectivity.add_edge(u, v, distance=w, weight=w)
        self._bridge_list.append(((i_env, i_bin), (j_env, j_bin), w))

    def _infer_mnn_bridges(self, max_distance: Optional[float] = None):
        """
        Infer “bridge edges” between every pair of sub-environments using a Mutual Nearest Neighbor (MNN) approach:

        1. For each pair (i, j) with i < j:
           a) Build KDTree_i on env_i.bin_centers
           b) Build KDTree_j on env_j.bin_centers
           c) For each bin center in env_i, find its nearest neighbor in env_j (nn_j_of_i)
           d) For each bin center in env_j, find its nearest neighbor in env_i (nn_i_of_j)
           e) If nn_j_of_i[i_idx] == j_idx and nn_i_of_j[j_idx] == i_idx, they are mutual nearest.
              Record (i_idx, j_idx, distance).
        2. If max_distance is provided, only keep pairs with distance ≤ max_distance.
        3. Add each pair as a bridge edge via `_add_bridge_edge`.
        """
        n_sub = len(self._subenvs_info)
        kdtrees = []
        for block in self._subenvs_info:
            centers = block["env"].bin_centers
            kdtrees.append(KDTree(centers, leaf_size=40))

        for i in range(n_sub):
            block_i = self._subenvs_info[i]
            centers_i = block_i["env"].bin_centers
            tree_i = kdtrees[i]

            for j in range(i + 1, n_sub):
                block_j = self._subenvs_info[j]
                centers_j = block_j["env"].bin_centers
                tree_j = kdtrees[j]

                # For each center in i → nearest in j
                dist_ij, idx_ij = tree_j.query(centers_i, k=1)
                idx_ij = idx_ij[:, 0]
                dist_ij = dist_ij[:, 0]

                # For each center in j → nearest in i
                dist_ji, idx_ji = tree_i.query(centers_j, k=1)
                idx_ji = idx_ji[:, 0]
                dist_ji = dist_ji[:, 0]

                for i_idx, j_idx in enumerate(idx_ij):
                    if idx_ji[j_idx] == i_idx:
                        d = dist_ij[i_idx]
                        if (max_distance is not None) and (d > max_distance):
                            continue
                        self._add_bridge_edge(i, i_idx, j, j_idx, float(d))

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (same as each sub-environment)."""
        return self._n_dims

    @property
    def n_bins(self) -> int:
        """Total number of active bins in the composite environment."""
        return self._total_bins

    def bin_at(self, points_nd: np.ndarray) -> np.ndarray:
        """
        Map each point in `points_nd` (shape (M, n_dims)) to a composite bin index.
        - Calls each subenv.bin_at(points_nd) → array of shape (M,) (sub-bin indices or -1).
        - Wherever a subenv value ≥ 0, set composite index = sub_idx + start_idx (first match wins).
        - Returns an integer array of shape (M,), with -1 for points outside all sub-environments.
        """
        if points_nd.ndim != 2 or points_nd.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected points_nd of shape (M, {self.n_dims}), got {points_nd.shape}"
            )

        M = points_nd.shape[0]
        out = np.full((M,), -1, dtype=int)

        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            sub_idxs = env_i.bin_at(points_nd)  # expects shape (M,)
            if sub_idxs.dtype not in (np.int32, np.int64):
                sub_idxs = sub_idxs.astype(int)
            mask = (sub_idxs >= 0) & (out == -1)
            out[mask] = sub_idxs[mask] + base

        return out

    def contains(self, points_nd: np.ndarray) -> np.ndarray:
        """
        Return a boolean array of shape (M,), True if each point in `points_nd` lies in any bin.
        Equivalent to self.bin_at(points_nd) != -1.
        """
        return self.bin_at(points_nd) != -1

    def neighbors(self, bin_index: int) -> List[int]:
        """
        Return a list of composite bin indices that are neighbors of `bin_index`
        in the merged connectivity graph.
        """
        if not (0 <= bin_index < self._total_bins):
            raise KeyError(
                f"Bin index {bin_index} is out of range [0..{self._total_bins - 1}]"
            )
        return list(self.connectivity.neighbors(bin_index))

    def distance_between(
        self,
        point1: Union[np.ndarray, List[float], Tuple[float, ...]],
        point2: Union[np.ndarray, List[float], Tuple[float, ...]],
        edge_weight: str = "distance",
    ) -> float:
        """
        Compute the shortest-path distance (weighted by `edge_weight`) between two points:
        1) Map each point to a bin index via bin_at (if a list/tuple is given, convert to a single-row array).
        2) If either bin = -1, return np.inf.
        3) Otherwise, return nx.shortest_path_length(self.connectivity, source=bin1, target=bin2, weight=edge_weight).
        """

        def _to_array(pt):
            arr = np.asarray(pt, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, self.n_dims)
            if arr.ndim != 2 or arr.shape[1] != self.n_dims:
                raise ValueError(
                    f"Expected a point of length {self.n_dims} or shape (1, {self.n_dims}), got {arr.shape}"
                )
            return arr

        arr1 = _to_array(point1)
        arr2 = _to_array(point2)

        bin1 = self.bin_at(arr1)[0]
        bin2 = self.bin_at(arr2)[0]
        if bin1 < 0 or bin2 < 0:
            return float(np.inf)
        return nx.shortest_path_length(
            self.connectivity, source=bin1, target=bin2, weight=edge_weight
        )

    def bin_center_of(self, bin_indices: Union[int, np.ndarray]) -> np.ndarray:
        """
        Return the N-D coordinate(s) of the specified composite bin index or indices.
        Accepts either a single int or a 1-D numpy array of ints.
        """
        return np.asarray(self.bin_centers)[bin_indices]

    def bin_attributes(self) -> pd.DataFrame:
        """
        Return a concatenated DataFrame of per-bin attributes:
        - Each sub-environment's bin_attributes() is copied.
        - A new column 'composite_bin_id' = (child_bin_id + start_idx) is added.
        """
        dfs = []
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            df = env_i.bin_attributes.copy()
            df["child_active_bin_id"] = df.index
            df["composite_bin_id"] = df.index + base
            dfs.append(df)
        composite_df = pd.concat(dfs, ignore_index=True)
        return composite_df

    def edge_attributes(self) -> pd.DataFrame:
        """
        Return a concatenated DataFrame of per-edge attributes:
        - Each sub-environment's edge_attributes() is copied with 'u_idx' and 'v_idx' shifted by start_idx.
        - MNN-inferred “bridge edges” are appended as additional rows with columns 'u_idx','v_idx','distance','weight'.
        """
        dfs = []
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            df = env_i.edge_attributes.copy()
            df["composite_source_bin"] = df["source_bin"] + base
            df["composite_target_bin"] = df["target_bin"] + base
            dfs.append(df)

        if self._bridge_list:
            bridge_rows = []
            for (i_env, i_bin), (j_env, j_bin), w in self._bridge_list:
                block_i = self._subenvs_info[i_env]
                block_j = self._subenvs_info[j_env]
                u = block_i["start_idx"] + i_bin
                v = block_j["start_idx"] + j_bin
                bridge_rows.append(
                    {
                        "composite_source_bin": u,
                        "composite_target_bin": v,
                        "distance": w,
                        "weight": 1 / w,
                    }
                )
            bridge_df = pd.DataFrame(bridge_rows)
            dfs.append(bridge_df)

        composite_edges_df = pd.concat(dfs, ignore_index=True)
        return composite_edges_df
