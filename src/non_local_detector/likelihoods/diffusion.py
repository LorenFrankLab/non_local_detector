"""Graph-diffusion spectral engine for manifold-aware place-field smoothing.

This module builds a finite-difference graph Laplacian ``L`` on an environment's
interior bins and applies the heat kernel ``exp(-t L)``, ``t = sigma**2 / 2``, as a
smoother that respects track geometry (walls, holes, junctions) in 1D and N-D.

Unlike Gaussian KDE, which smooths in the coordinate metric and smears across
walls, graph diffusion smooths along the diffusion distance on the domain:
reflecting (Neumann) boundaries are intrinsic (no edges to absent bins) and the
operator conserves mass (``1ᵀL = 0`` makes ``exp(-t L)`` column-stochastic).

The Laplacian uses the **finite-difference** edge weight ``w = 1 / distance**2`` on
a **face-adjacent** graph, which gives ``L ≈ -∂²`` on a regular grid; the resulting
heat kernel is a Gaussian of standard deviation ``sigma`` **independent of bin
size**. (The alternative ``exp(-d²/2σ²)`` weighting yields an effective bandwidth
that scales with the grid spacing.) Diagonal / Moore edges are excluded because
with ``1/d²`` weighting they oversmooth by ≈√2 in 2D.

Because the eigendecomposition depends only on the graph (not ``sigma`` and not the
fields being smoothed), it is computed once and cached on the ``Environment`` via
``cached_eigenbasis``, then reused across all neurons and all EM refits.
"""

import heapq
import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg

from non_local_detector.exceptions import ValidationError

if TYPE_CHECKING:
    from non_local_detector.environment import Environment


def build_laplacian(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    """Finite-difference symmetric graph Laplacian ``L = D - W``.

    The edge weight is the finite-difference coefficient ``w = 1 / distance**2``,
    so that ``L ≈ -∂²`` on a regular face-adjacent grid and the heat kernel
    ``exp(-t L)`` is a Gaussian of standard deviation ``sqrt(2 t)`` independent of
    bin size. ``L`` is symmetric with zero row sums (``1ᵀL = 0``) and is positive
    semi-definite with one null mode per connected component.

    Parameters
    ----------
    graph : nx.Graph
        Face-adjacent interior-bin graph with contiguous integer nodes
        ``0..n_nodes-1``. Each edge must carry a positive ``'distance'`` attribute
        (Euclidean distance between the two bin centers). ``sigma``-independent.

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n_nodes, n_nodes)
        The sparse symmetric graph Laplacian.
    """
    n_nodes = graph.number_of_nodes()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        distance = data["distance"]
        if not distance > 0:  # also catches NaN
            raise ValidationError(
                "graph edge has a non-positive 'distance'",
                expected="every edge 'distance' > 0",
                got=f"distance = {distance} on edge ({u}, {v})",
                hint="Interior-bin edges must carry a positive Euclidean distance.",
            )
        w = 1.0 / distance**2  # finite-difference weight
        rows += [u, v]
        cols += [v, u]
        vals += [w, w]

    weights = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    degree = np.asarray(weights.sum(axis=1)).ravel()
    return (scipy.sparse.diags(degree) - weights).tocsr()


def _require_rank_covers_components(rank: int, n_components: int) -> None:
    """Raise if a truncated ``rank`` would drop a connected component's null mode."""
    if rank < n_components:
        raise ValidationError(
            "rank is too small to retain every connected component's null mode",
            expected=f"rank >= n_components = {n_components}",
            got=f"rank = {rank}",
            hint=(
                "Each connected component contributes one zero mode that must be "
                "kept for mass conservation; raise rank to at least the number of "
                "connected components (or use rank=None for the full basis)."
            ),
        )


def diffusion_eigenbasis(
    L: scipy.sparse.spmatrix, rank: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of the graph Laplacian ``L = Q Λ Qᵀ``.

    Parameters
    ----------
    L : scipy.sparse.spmatrix, shape (n_bins, n_bins)
        Symmetric graph Laplacian from :func:`build_laplacian`.
    rank : int or None, optional
        Number of smallest-eigenvalue modes to return. ``None`` (default) returns
        all ``n_bins`` modes via dense ``scipy.linalg.eigh``. An integer ``rank <
        n_bins`` returns the ``rank`` smallest modes via
        ``scipy.sparse.linalg.eigsh`` with shift-invert at a small *negative* shift
        (``sigma=-1e-8``); ``sigma=0`` factorizes the singular Laplacian and is
        unreliable. The truncation must retain **all** zero modes (one per
        connected component), so ``rank`` must be at least the number of connected
        components.

    Returns
    -------
    eigvals : np.ndarray, shape (m,)
        Eigenvalues in ascending order, clipped to be non-negative. ``m == rank``
        for a truncated call, else ``n_bins``.
    eigvecs : np.ndarray, shape (n_bins, m)
        Corresponding orthonormal eigenvectors as columns.

    Raises
    ------
    ValidationError
        If ``rank`` is below the number of connected components (which would drop a
        null mode and break component-wise mass conservation).
    """
    n_bins = L.shape[0]

    if rank is None or rank >= n_bins:
        eigvals, eigvecs = scipy.linalg.eigh(L.toarray())
        return np.clip(eigvals, 0.0, None), eigvecs

    n_components = scipy.sparse.csgraph.connected_components(
        L, directed=False, return_labels=False
    )
    _require_rank_covers_components(rank, n_components)

    try:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k=rank, sigma=-1e-8, which="LM")
    except RuntimeError:
        # Shift-invert can fail on the (near-)singular Laplacian in some builds,
        # raising either "Factor is exactly singular" (a plain RuntimeError from the
        # sparse LU factorization) or an ArpackError (itself a RuntimeError
        # subclass). Both are caught here; fall back to the no-shift-invert
        # smallest-magnitude solver.
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k=rank, which="SM")

    order = np.argsort(eigvals)
    return np.clip(eigvals[order], 0.0, None), eigvecs[:, order]


def diffuse(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    sigma: float,
    fields: np.ndarray,
) -> np.ndarray:
    """Apply the heat kernel ``exp(-t L)``, ``t = sigma**2 / 2``, to ``fields``.

    Computed in the eigenbasis as ``Q (exp(-tΛ) ⊙ (Qᵀ F))``, never materializing
    the dense ``(n_bins, n_bins)`` kernel. All fields (occupancy + every neuron's
    count field) diffuse in a single batched matmul.

    Parameters
    ----------
    eigvals : np.ndarray, shape (m,)
        Laplacian eigenvalues from :func:`diffusion_eigenbasis`.
    eigvecs : np.ndarray, shape (n_bins, m)
        Corresponding eigenvectors as columns.
    sigma : float
        Smoothing standard deviation in coordinate units (bandwidth).
    fields : np.ndarray, shape (n_bins, n_fields)
        Count fields on the interior bins, one column per field.

    Returns
    -------
    smoothed : np.ndarray, shape (n_bins, n_fields)
        Diffused fields, clipped to be non-negative and renormalized so each
        column's total mass equals the input field's total (mass-conserving).

    Notes
    -----
    The heat kernel conserves each field's total mass because the constant (null)
    mode is always retained. At full rank the kernel is already non-negative, so
    clipping only removes round-off. A truncated basis, however, can produce
    non-tiny negative lobes (e.g. for a point source); clipping those alone would
    inflate the total, so each column is renormalized back to its input sum.
    """
    t = sigma**2 / 2.0
    coeff = np.exp(-t * eigvals)  # (m,)
    proj = eigvecs.T @ fields  # (m, n_fields)
    smoothed = eigvecs @ (coeff[:, None] * proj)  # (n_bins, n_fields)

    clipped = np.clip(smoothed, 0.0, None)
    input_mass = fields.sum(axis=0)  # conserved quantity
    clipped_mass = clipped.sum(axis=0)
    scale = np.divide(
        input_mass,
        clipped_mass,
        out=np.zeros_like(input_mass, dtype=float),
        where=clipped_mass > 0,
    )
    return clipped * scale


def to_density(smoothed: np.ndarray, bin_sizes: np.ndarray) -> np.ndarray:
    """Normalize each column of ``smoothed`` to an integral-one density.

    ``exp(-t L)`` conserves field *sums*, not the area integral; on non-uniform
    bins these differ. Each column is divided by its mass ``Σ bin_sizes_i *
    smoothed_i`` so that ``bin_sizes @ density == 1``. Columns with zero mass map
    to all zeros.

    Parameters
    ----------
    smoothed : np.ndarray, shape (n_bins, n_fields)
        Smoothed (non-negative) count fields.
    bin_sizes : np.ndarray, shape (n_bins,)
        Per-bin volumes (1D width, 2D area, ...).

    Returns
    -------
    density : np.ndarray, shape (n_bins, n_fields)
        Column-normalized densities; zero-mass columns are zero.
    """
    mass = bin_sizes @ smoothed  # (n_fields,)
    safe = np.where(mass > 0, mass, 1.0)
    return np.where(mass > 0, smoothed / safe, 0.0)


def environment_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Build the interior-bin graph, node order, and per-bin volumes for diffusion.

    Two branches, selected by ``environment.track_graph``:

    - **N-D / 1D grid** (``track_graph is None``): the subgraph of
      ``track_graphDD`` induced by interior bins, keeping only **face-adjacent**
      edges (bin centers differing in exactly one dimension); diagonal / Moore
      edges are dropped because with ``1/d²`` weighting they oversmooth.
    - **Linearized track graph**: interior bin centers of
      ``track_graph_with_bin_centers_edges_`` contracted to a bin-center adjacency
      graph, connecting each interior bin center to the neighbors reachable through
      bin-edge / junction nodes; junctions link arms, gap bins are excluded.

    Nodes are relabeled ``0..n_interior-1`` in interior flat-bin order
    (``np.where(is_track_interior_.ravel())[0]``), matching every consumer's
    place-field indexing.

    Parameters
    ----------
    environment : Environment
        A fitted environment with ``place_bin_centers_`` and ``is_track_interior_``.

    Returns
    -------
    graph : nx.Graph
        Interior-bin graph with nodes ``0..n_interior-1`` and ``'distance'``-weighted
        edges.
    node_order : np.ndarray, shape (n_interior,)
        Interior flat-bin indices in graph-node order (maps local nodes back to the
        full grid).
    bin_sizes : np.ndarray, shape (n_interior,)
        Per-interior-bin volume (1D width, 2D area, ...).

    Notes
    -----
    The result is cached on ``environment._diffusion_graph_`` (invalidated by
    ``fit_place_grid``) so the graph is built once and shared by
    :func:`cached_eigenbasis` and the likelihood fit. Callers must treat the
    returned graph and arrays as read-only.
    """
    cached = getattr(environment, "_diffusion_graph_", None)
    if cached is not None:
        return cached

    if environment.is_track_interior_ is None or environment.place_bin_centers_ is None:
        raise ValidationError(
            "environment must be fitted before building the diffusion graph",
            expected="a fitted Environment (place_bin_centers_, is_track_interior_ set)",
            got="an unfitted environment",
            hint="Call environment.fit_place_grid(position) first.",
        )

    if environment.track_graph is None:
        if environment.track_graphDD is None:
            raise ValidationError(
                "fitted N-D environment is missing its track graph",
                expected="track_graphDD set by fit_place_grid",
                got="track_graphDD is None",
            )
        result = _nd_grid_graph(environment)
    else:
        if environment.track_graph_with_bin_centers_edges_ is None:
            raise ValidationError(
                "fitted linearized environment is missing its bin-center graph",
                expected="track_graph_with_bin_centers_edges_ set by fit_place_grid",
                got="track_graph_with_bin_centers_edges_ is None",
            )
        result = _linearized_graph(environment)

    # The cached graph/arrays are shared across all neurons and every EM refit.
    # Freeze the graph topology and mark the arrays read-only so a stray mutation
    # cannot corrupt later callers. `nx.freeze` does NOT block edge-attribute
    # writes, so snapshot the Laplacian here — before the graph is handed out — and
    # derive the eigenbasis from that snapshot (see `cached_eigenbasis`); a later
    # `graph.edges[e]["distance"] = ...` then cannot change the cached basis.
    graph, node_order, bin_sizes = result
    nx.freeze(graph)
    node_order.setflags(write=False)
    bin_sizes.setflags(write=False)
    environment._diffusion_laplacian_ = build_laplacian(graph)
    environment._diffusion_graph_ = result
    return result


def check_smoothing_bandwidth(sigma: float, graph: nx.Graph) -> None:
    """Validate ``sigma`` and warn if it under-smooths the grid.

    Called by the likelihood fit (the engine's :func:`diffuse` is
    ``sigma``-parameterized per call, so the guard lives here). Raises for a
    non-positive bandwidth and warns when ``sigma`` is below the characteristic bin
    spacing — where the heat kernel barely spreads past a single bin and the
    smoothed field stays near the raw spike counts.

    Parameters
    ----------
    sigma : float
        Smoothing standard deviation (``position_std``) in coordinate units.
    graph : nx.Graph
        Interior-bin graph from :func:`environment_graph`; its ``'distance'`` edge
        attributes give the characteristic bin spacing.

    Raises
    ------
    ValidationError
        If ``sigma`` is not strictly positive.

    Warns
    -----
    UserWarning
        If ``sigma`` is below the characteristic (median) bin spacing, indicating
        the smoother will barely spread beyond a single bin.
    """
    if not sigma > 0:
        raise ValidationError(
            "smoothing bandwidth (position_std) must be positive",
            expected="position_std > 0",
            got=f"position_std = {sigma}",
        )

    distances = [data["distance"] for _, _, data in graph.edges(data=True)]
    if distances:
        characteristic = float(np.median(distances))
        if sigma < characteristic:
            warnings.warn(
                f"position_std ({sigma:g}) is below the characteristic bin spacing "
                f"({characteristic:g}); the diffusion smoother barely spreads beyond "
                "one bin, so place fields will stay close to the raw spike counts. "
                "Increase position_std or use a coarser grid.",
                UserWarning,
                stacklevel=2,
            )


def cached_eigenbasis(
    environment: "Environment", rank: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Laplacian eigenbasis for ``environment``, building it on a miss.

    The eigenbasis depends only on the environment's graph (not ``sigma`` and not
    the fields), so it is computed once and cached on
    ``environment._diffusion_eigenbasis_`` — a dict keyed by ``rank`` — and reused
    across all neurons and every EM refit. A cached full-rank (``rank=None``) entry
    serves any smaller-rank request by slicing its leading columns (valid because
    the modes are returned ascending). The cache is invalidated by
    ``fit_place_grid``.

    Parameters
    ----------
    environment : Environment
        A fitted environment.
    rank : int or None, optional
        Number of smallest modes to return; ``None`` (default) returns the full
        basis. See :func:`diffusion_eigenbasis`.

    Returns
    -------
    eigvals : np.ndarray, shape (m,)
    eigvecs : np.ndarray, shape (n_interior, m)
        The cached eigenbasis. ``node_order`` / ``bin_sizes`` come from
        :func:`environment_graph` (also cached), not from here.
    """
    cache = getattr(environment, "_diffusion_eigenbasis_", None)
    if cache is None:
        cache = {}
        environment._diffusion_eigenbasis_ = cache

    if rank in cache:
        return cache[rank]

    # Build (and cache) the Laplacian snapshot; environment_graph runs first so a
    # later mutation of the returned graph cannot reach this L.
    environment_graph(environment)
    laplacian = environment._diffusion_laplacian_

    if rank is not None and None in cache:
        eigvals, eigvecs = cache[None]
        if rank <= eigvals.shape[0]:
            # Validate before slicing: the full basis has no idea about `rank`, so
            # this path would otherwise bypass diffusion_eigenbasis's guard and
            # silently drop a component's null mode.
            n_components = scipy.sparse.csgraph.connected_components(
                laplacian, directed=False, return_labels=False
            )
            _require_rank_covers_components(rank, n_components)
            # Slices of the read-only full-rank basis are themselves read-only.
            sliced = (eigvals[:rank], eigvecs[:, :rank])
            cache[rank] = sliced
            return sliced

    eigvals, eigvecs = diffusion_eigenbasis(laplacian, rank)
    # Freeze the cached basis: it is reused across neurons and EM refits.
    eigvals.setflags(write=False)
    eigvecs.setflags(write=False)
    basis = (eigvals, eigvecs)
    cache[rank] = basis
    return basis


def _nd_grid_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Adapter branch for N-D / 1D grid environments (``track_graph is None``)."""
    node_order = np.where(environment.is_track_interior_.ravel())[0]
    track_graph = environment.track_graphDD
    subgraph = track_graph.subgraph(node_order)
    positions = nx.get_node_attributes(track_graph, "pos")
    relabel = {old: new for new, old in enumerate(node_order)}

    graph = nx.Graph()
    graph.add_nodes_from(range(node_order.size))
    for u, v, data in subgraph.edges(data=True):
        offset = np.asarray(positions[u]) - np.asarray(positions[v])
        # Keep only face-adjacent pairs (differ in exactly one dimension).
        if np.count_nonzero(np.abs(offset) > 1e-9) == 1:
            graph.add_edge(relabel[u], relabel[v], distance=data["distance"])

    # bin_sizes = product of per-dimension bin widths, over the interior bins.
    widths = [np.diff(edge) for edge in environment.edges_]
    bin_volume = np.ones(environment.centers_shape_)
    for axis_width in np.meshgrid(*widths, indexing="ij"):
        bin_volume = bin_volume * axis_width
    bin_sizes = bin_volume.ravel()[node_order]

    return graph, node_order, bin_sizes


def _linearized_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Adapter branch for linearized ``track_graph`` environments."""
    is_interior = environment.is_track_interior_.ravel()
    node_order = np.where(is_interior)[0]

    # Substrate node id for each interior place bin (gap bins carry -1 and are
    # excluded); order matches place_bin_centers_ / is_track_interior_.
    node_ids = environment.place_bin_centers_nodes_df_.node_id.to_numpy()
    interior_node_ids = node_ids[is_interior].astype(int)
    centers = {int(nid) for nid in interior_node_ids}
    local_of = {int(nid): local for local, nid in enumerate(interior_node_ids)}

    substrate = environment.track_graph_with_bin_centers_edges_
    graph = nx.Graph()
    graph.add_nodes_from(range(interior_node_ids.size))
    for start in interior_node_ids:
        for neighbor, distance in _neighbor_centers(substrate, int(start), centers):
            graph.add_edge(local_of[int(start)], local_of[neighbor], distance=distance)

    bin_sizes = np.diff(environment.place_bin_edges_.ravel())[is_interior]
    return graph, node_order, bin_sizes


def _neighbor_centers(
    substrate: nx.Graph, start: int, centers: set[int]
) -> list[tuple[int, float]]:
    """Bin-center nodes reachable from ``start`` without passing another center.

    Runs Dijkstra from ``start`` over ``substrate`` (bin-center, bin-edge, and
    junction nodes), treating every other center node as a sink (recorded but not
    expanded through). Returns ``(center_node, path_distance)`` pairs — the local
    bin-center adjacency the contraction needs; junction nodes link arms.
    """
    distances: dict[int, float] = {start: 0.0}
    heap: list[tuple[float, int]] = [(0.0, start)]
    neighbors: list[tuple[int, float]] = []
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances.get(node, np.inf):
            continue
        if node != start and node in centers:
            neighbors.append((node, dist))
            continue  # do not expand through another bin center
        for adjacent, edge_data in substrate[node].items():
            candidate = dist + edge_data["distance"]
            if candidate < distances.get(adjacent, np.inf):
                distances[adjacent] = candidate
                heapq.heappush(heap, (candidate, adjacent))
    return neighbors
