"""Exact bounded graph distances for fitted Cartesian environments.

The sparse graph is retained; each query computes only its distinct source
rows in bounded batches, and recently used source rows are kept in a bounded
cache. Row queries never build the all-pairs matrix; only an explicit dense
conversion does, within ``max_dense_bytes``. Unreachable nodes have
infinite distance, including isolated exterior nodes (whose self-distance is
zero), exactly as in the legacy NetworkX shortest-path matrix.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, cast

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from non_local_detector.exceptions import NonLocalDetectorError

# Internal module: no public API.
__all__: list[str] = []


DEFAULT_MAX_DENSE_DISTANCE_BYTES = 256 * 1024**2
DEFAULT_DISTANCE_WORKSPACE_BYTES = 8 * 1024**2
DEFAULT_DISTANCE_CACHE_BYTES = 64 * 1024**2


class GraphDistanceBudgetError(NonLocalDetectorError, MemoryError):
    """A requested exact graph-distance allocation exceeds its byte budget."""


def _budget(nbytes: int, maximum: Any, description: str) -> None:
    if (
        not isinstance(maximum, (int, np.integer))
        or isinstance(maximum, bool)
        or maximum < 0
    ):
        msg = "Graph distance byte budgets must be nonnegative integers"
        raise ValueError(msg)
    if nbytes > maximum:
        msg = (
            f"{description} requires {nbytes:,} bytes, exceeding the {maximum:,}-byte "
            "graph distance budget. A detector sets this budget from "
            "max_dense_transition_bytes: raise it, or use "
            'transition_representation="dense", which computes all-pairs '
            "distances when the environment is fitted. Otherwise request fewer "
            "source/destination rows."
        )
        raise GraphDistanceBudgetError(msg)


class LazyGraphDistances:
    """Read-only float64 graph-distance matrix with bounded row queries.

    Basic, paired and ``np.ix_`` selections compute exact shortest paths.
    ``np.asarray`` requires the full N-by-N result to fit ``max_dense_bytes``.
    ``max_workspace_bytes`` bounds each temporary Dijkstra row batch; it must
    accommodate at least one float64 N-element source row.
    ``max_cache_bytes`` bounds a least-recently-used cache of full source rows
    (a value below one row, ``8 * N`` bytes, disables it); indexing and
    ``cross_distances`` fill it, ``to_dense`` does not. ``cross_distances``
    returns a caller-owned block that is not limited by ``max_dense_bytes``.
    Pickling retains the sparse graph, ``directed`` and budgets, never the
    cache. The cache makes queries stateful: share an
    instance across threads only with external locking.

    Parameters
    ----------
    graph : scipy.sparse matrix or array_like, shape (n_nodes, n_nodes)
        Weighted adjacency matrix with finite, nonnegative edge distances.
    directed : bool, optional
        Whether to follow edge directions when finding shortest paths, by
        default False.
    max_dense_bytes : int, optional
        Byte limit for converting to a dense matrix, by default 256 MiB.
    max_workspace_bytes : int, optional
        Byte limit for each temporary batch of Dijkstra rows, by default 8 MiB.
    max_cache_bytes : int, optional
        Byte limit for the cache of full source rows, by default 64 MiB.

    Attributes
    ----------
    graph : scipy.sparse.csr_matrix
        Float64 CSR adjacency matrix of edge distances.
    directed : bool
        Whether shortest paths follow edge directions.
    max_dense_bytes, max_workspace_bytes, max_cache_bytes : int
        The byte budgets above.

    Raises
    ------
    ValueError
        If the adjacency is not square, has negative or non-finite edge
        distances, or a budget is not a nonnegative integer.
    """

    def __init__(
        self,
        graph: Any,
        *,
        directed: bool = False,
        max_dense_bytes: int = DEFAULT_MAX_DENSE_DISTANCE_BYTES,
        max_workspace_bytes: int = DEFAULT_DISTANCE_WORKSPACE_BYTES,
        max_cache_bytes: int = DEFAULT_DISTANCE_CACHE_BYTES,
    ) -> None:
        _budget(0, max_dense_bytes, "Dense graph distances")
        _budget(0, max_workspace_bytes, "Graph distance workspace")
        _budget(0, max_cache_bytes, "Graph distance row cache")
        graph = csr_matrix(graph, dtype=np.float64, copy=True)
        if graph.shape[0] != graph.shape[1]:
            msg = "Graph adjacency must be square"
            raise ValueError(msg)
        if not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0):
            msg = "Graph edge distances must be finite and nonnegative"
            raise ValueError(msg)
        self.graph = graph
        self.directed = bool(directed)
        self.max_dense_bytes = int(max_dense_bytes)
        self.max_workspace_bytes = int(max_workspace_bytes)
        self.max_cache_bytes = int(max_cache_bytes)
        self._row_cache: OrderedDict[int, NDArray[np.float64]] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_row_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("max_cache_bytes", DEFAULT_DISTANCE_CACHE_BYTES)
        state.setdefault("_row_cache", OrderedDict())
        self.__dict__.update(state)

    @classmethod
    def from_graph(cls, graph: nx.Graph, **kwargs: Any) -> LazyGraphDistances:
        """Build lazy distances from a NetworkX graph.

        Parameters
        ----------
        graph : networkx.Graph
            Graph whose nodes are the integers ``0`` through ``N - 1`` and
            whose edges have a ``distance`` attribute.
        **kwargs
            Byte budgets passed to the constructor (``max_dense_bytes``,
            ``max_workspace_bytes``, ``max_cache_bytes``).

        Returns
        -------
        distances : LazyGraphDistances
            Lazy shortest-path distances between the graph's nodes.

        Raises
        ------
        ValueError
            If the nodes are not indexed from zero through ``N - 1``.
        """
        if set(graph.nodes) != set(range(len(graph))):
            msg = "Graph distance nodes must be indexed from zero through N-1"
            raise ValueError(msg)
        sparse = nx.to_scipy_sparse_array(
            graph,
            nodelist=range(len(graph)),
            weight="distance",
            format="csr",
            dtype=float,
        )
        # SciPy csgraph accepts the ordinary CSR index width on all supported
        # SciPy versions; NetworkX sparse arrays may instead use int64 indices.
        if len(graph) < np.iinfo(np.int32).max:
            sparse.indices = sparse.indices.astype(np.int32)
            sparse.indptr = sparse.indptr.astype(np.int32)
        return cls(sparse, directed=graph.is_directed(), **kwargs)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_nodes, n_nodes)`` of the distance matrix."""
        return cast("tuple[int, int]", self.graph.shape)

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Data type of the distances, always float64."""
        return np.dtype(np.float64)

    @property
    def ndim(self) -> int:
        """Number of dimensions, always 2."""
        return 2

    @property
    def storage_nbytes(self) -> int:
        """Memory used by the stored sparse graph in bytes, excluding the row cache."""
        return cast(
            "int",
            self.graph.data.nbytes
            + self.graph.indices.nbytes
            + self.graph.indptr.nbytes,
        )

    def _entries(
        self,
        rows: ArrayLike,
        columns: ArrayLike,
        *,
        max_bytes: int | None = None,
    ) -> NDArray[np.float64]:
        rows, columns = np.broadcast_arrays(rows, columns)
        _budget(
            rows.size * 8,
            self.max_dense_bytes if max_bytes is None else max_bytes,
            "Selected graph distances",
        )
        result = np.empty(rows.shape, dtype=float)
        if not result.size:
            return result
        row_bytes = self.shape[1] * 8
        _budget(
            row_bytes,
            self.max_workspace_bytes,
            "One source-row graph distance workspace",
        )
        batch_rows = max(1, self.max_workspace_bytes // row_bytes)
        flat_rows, flat_columns = rows.ravel(), columns.ravel()
        sources, inverse = np.unique(flat_rows, return_inverse=True)
        flat_result = result.ravel()
        for start in range(0, len(sources), batch_rows):
            stop = min(start + batch_rows, len(sources))
            distances = self._rows_for(sources[start:stop])
            select = (inverse >= start) & (inverse < stop)
            flat_result[select] = distances[
                inverse[select] - start, flat_columns[select]
            ]
            del distances
        return result

    def _rows_for(self, sources: NDArray[np.integer]) -> NDArray[np.float64]:
        """Full float64 rows for unique ``sources``, computing only uncached rows.

        Callers bound ``len(sources)`` by the workspace budget.
        """
        row_bytes = self.shape[1] * 8
        result = np.empty((len(sources), self.shape[1]))
        missing = []
        for position, source in enumerate(sources):
            row = self._row_cache.get(int(source))
            if row is None:
                missing.append(position)
            else:
                self._row_cache.move_to_end(int(source))
                result[position] = row
        if missing:
            result[missing] = dijkstra(
                self.graph, directed=self.directed, indices=sources[missing]
            )
            if row_bytes <= self.max_cache_bytes:
                for position in missing:
                    self._row_cache[int(sources[position])] = result[position].copy()
                while len(self._row_cache) * row_bytes > self.max_cache_bytes:
                    self._row_cache.popitem(last=False)
        return result

    def cross_distances(
        self, rows: ArrayLike, columns: ArrayLike
    ) -> NDArray[np.float64]:
        """Exact ``(len(rows), len(columns))`` distances; the output is caller-owned.

        Parameters
        ----------
        rows : array-like of int, shape (n_rows,)
            Source node per output row; repeated sources are computed once.
        columns : array-like of int, shape (n_columns,)
            Destination nodes.

        Returns
        -------
        distances : np.ndarray, shape (n_rows, n_columns)
            Only the Dijkstra workspace and the row cache are budgeted; the
            caller already owns an array of this size.
        """
        rows, columns = np.asarray(rows), np.asarray(columns)
        for name, index in (("rows", rows), ("columns", columns)):
            if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
                msg = f"{name} must be a one-dimensional integer array"
                raise IndexError(msg)
        # Same normalization as ``__getitem__``: negative indices wrap and
        # out-of-range indices raise.
        rows = np.arange(self.shape[0])[rows.astype(np.intp)]
        columns = np.arange(self.shape[1])[columns.astype(np.intp)]
        out = np.empty((len(rows), len(columns)))
        if not out.size:
            return out
        row_bytes = self.shape[1] * 8
        _budget(
            row_bytes,
            self.max_workspace_bytes,
            "One source-row graph distance workspace",
        )
        batch_rows = max(1, self.max_workspace_bytes // row_bytes)
        sources, inverse = np.unique(rows, return_inverse=True)
        for start in range(0, len(sources), batch_rows):
            stop = min(start + batch_rows, len(sources))
            block = self._rows_for(sources[start:stop])[:, columns]
            select = (inverse >= start) & (inverse < stop)
            out[select] = block[inverse[select] - start]
        return out

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            if np.ndim(key) > 1 and np.asarray(key).dtype == bool:
                # A full-shape mask selects elements, not rows.
                return np.asarray(self)[key]
            key = (key, slice(None))
        if len(key) != 2 or any(item is None or item is Ellipsis for item in key):
            return np.asarray(self)[key]
        rows, columns = (
            np.arange(size)[item] for size, item in zip(self.shape, key, strict=True)
        )
        # Basic slices produce the Cartesian dimensions NumPy adds. Advanced
        # arrays instead broadcast pairwise, including the arrays from np.ix_.
        if isinstance(key[0], slice) and np.ndim(columns) > 0:
            rows = rows.reshape((-1,) + (1,) * np.ndim(columns))
        if isinstance(key[1], slice) and np.ndim(rows) > 0:
            if not isinstance(key[0], slice):
                rows = rows[..., None]
            columns = columns.reshape((1,) * (np.ndim(rows) - 1) + (-1,))
        result = self._entries(rows, columns)
        return result[()] if result.ndim == 0 else result

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[Any]:
        return self.to_dense(dtype=dtype)

    def to_dense(
        self, *, max_bytes: int | None = None, dtype: DTypeLike | None = None
    ) -> NDArray[Any]:
        """Convert to a dense matrix within a byte budget.

        Parameters
        ----------
        max_bytes : int, optional
            Byte limit for the dense result, by default ``max_dense_bytes``.
        dtype : dtype, optional
            Data type of the result, by default float64.

        Returns
        -------
        distances : np.ndarray, shape (n_nodes, n_nodes)
            Shortest-path distances between every pair of nodes.

        Raises
        ------
        GraphDistanceBudgetError
            If the dense matrix exceeds ``max_bytes`` or one source row
            exceeds ``max_workspace_bytes``.
        """
        target = np.dtype(float if dtype is None else dtype)
        maximum = self.max_dense_bytes if max_bytes is None else max_bytes
        _budget(
            self.shape[0] * self.shape[1] * max(8, target.itemsize),
            maximum,
            "Dense graph distances",
        )
        row_bytes = self.shape[1] * 8
        if self.shape[0]:
            _budget(
                row_bytes,
                self.max_workspace_bytes,
                "One source-row graph distance workspace",
            )
        result = np.empty(self.shape, dtype=target)
        if not result.size:
            return result
        batch_rows = max(1, self.max_workspace_bytes // row_bytes)
        # Paired indexing would add several N² index/sort/inverse buffers.
        for start in range(0, self.shape[0], batch_rows):
            stop = min(start + batch_rows, self.shape[0])
            distances = dijkstra(
                self.graph, directed=self.directed, indices=np.arange(start, stop)
            )
            result[start:stop] = distances
            del distances
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        msg = "Lazy graph distances are read-only"
        raise TypeError(msg)

    def __len__(self) -> int:
        return self.shape[0]
