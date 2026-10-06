"""Exact bounded graph distances for fitted Cartesian environments.

The sparse graph is retained; each query computes only its distinct source
rows in bounded batches. No all-pairs cache is created. Unreachable nodes have
infinite distance, including isolated exterior nodes (whose self-distance is
zero), exactly as in the legacy NetworkX shortest-path matrix.
"""

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_MAX_DENSE_DISTANCE_BYTES = 256 * 1024**2
DEFAULT_DISTANCE_WORKSPACE_BYTES = 8 * 1024**2


class GraphDistanceBudgetError(MemoryError):
    """A requested exact graph-distance allocation exceeds its byte budget."""


def _budget(nbytes, maximum, description):
    if (
        not isinstance(maximum, (int, np.integer))
        or isinstance(maximum, bool)
        or maximum < 0
    ):
        raise ValueError("Graph distance byte budgets must be nonnegative integers")
    if nbytes > maximum:
        raise GraphDistanceBudgetError(
            f"{description} requires {nbytes:,} bytes, exceeding the {maximum:,}-byte "
            "graph distance budget. Request fewer source/destination rows or explicitly "
            "choose a feasible budget."
        )


class LazyGraphDistances:
    """Read-only float64 graph-distance matrix with bounded row queries.

    Basic, paired and ``np.ix_`` selections compute exact shortest paths.
    ``np.asarray`` requires the full N-by-N result to fit ``max_dense_bytes``.
    ``max_workspace_bytes`` bounds each temporary Dijkstra row batch; it must
    accommodate at least one float64 N-element source row. Pickling retains
    only the sparse graph and budgets. Queries do not retain a dense cache.
    """

    def __init__(
        self,
        graph,
        *,
        directed=False,
        max_dense_bytes=DEFAULT_MAX_DENSE_DISTANCE_BYTES,
        max_workspace_bytes=DEFAULT_DISTANCE_WORKSPACE_BYTES,
    ):
        _budget(0, max_dense_bytes, "Dense graph distances")
        _budget(0, max_workspace_bytes, "Graph distance workspace")
        graph = csr_matrix(graph, dtype=np.float64, copy=True)
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("Graph adjacency must be square")
        if not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0):
            raise ValueError("Graph edge distances must be finite and nonnegative")
        self.graph = graph
        self.directed = bool(directed)
        self.max_dense_bytes = int(max_dense_bytes)
        self.max_workspace_bytes = int(max_workspace_bytes)

    @classmethod
    def from_graph(cls, graph, **kwargs):
        if set(graph.nodes) != set(range(len(graph))):
            raise ValueError(
                "Graph distance nodes must be indexed from zero through N-1"
            )
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
    def shape(self):
        return self.graph.shape

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def ndim(self):
        return 2

    @property
    def storage_nbytes(self):
        return (
            self.graph.data.nbytes
            + self.graph.indices.nbytes
            + self.graph.indptr.nbytes
        )

    def _entries(self, rows, columns, *, max_bytes=None):
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
            distances = dijkstra(
                self.graph, directed=self.directed, indices=sources[start:stop]
            )
            select = (inverse >= start) & (inverse < stop)
            flat_result[select] = distances[
                inverse[select] - start, flat_columns[select]
            ]
            del distances
        return result

    def __getitem__(self, key):
        if not isinstance(key, tuple):
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

    def __array__(self, dtype=None, copy=None):
        return self.to_dense(dtype=dtype)

    def to_dense(self, *, max_bytes=None, dtype=None):
        """Explicit budgeted dense conversion, with optional caller budget."""
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

    def __setitem__(self, key, value):
        raise TypeError("Lazy graph distances are read-only")

    def __len__(self):
        return self.shape[0]
