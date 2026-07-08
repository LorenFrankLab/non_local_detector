"""Graph-diffusion spectral engine for manifold-aware place-field smoothing.

This module builds a finite-difference graph Laplacian ``L`` on an environment's
interior bins and applies the heat kernel ``exp(-t L)``, ``t = sigma**2 / 2``, as a
smoother that respects track geometry (walls, holes, junctions) in 1D and N-D.

Unlike Gaussian KDE, which smooths in the coordinate metric and smears across
walls, graph diffusion smooths along the diffusion distance on the domain:
reflecting (Neumann) boundaries are intrinsic (no edges to absent bins) and the
operator conserves each field's total mass (``1ᵀL = 0`` gives ``exp(-t L)`` unit
column sums; entrywise non-negativity is a separate property of the Laplacian
heat semigroup).

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

# Defaults for heat_kernel_rank (bandwidth-aware auto-truncation): drop modes whose
# heat-kernel weight exp(-t*lambda) is below _HEAT_KERNEL_RANK_TOL, and fall back to the
# dense eigendecomposition once the resolved rank exceeds _HEAT_KERNEL_DENSE_FRACTION *
# n_bins (where a truncated eigsh no longer beats a dense eigh). _HEAT_KERNEL_RANK_START
# is the first probe size for the adaptive rank search.
_HEAT_KERNEL_RANK_TOL = 1e-6
_HEAT_KERNEL_DENSE_FRACTION = 0.5
_HEAT_KERNEL_RANK_START = 32


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


def _block_eigenbasis(
    block: scipy.sparse.spmatrix, rank: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a single **connected** Laplacian block.

    ``rank is None`` (or ``rank >= n``) uses dense ``scipy.linalg.eigh``; a smaller
    ``rank`` uses truncated ``scipy.sparse.linalg.eigsh`` with a small negative
    shift-invert, falling back to the no-shift-invert solver if the factorization
    fails. Eigenvalues are returned ascending and clipped to be non-negative.
    """
    n = block.shape[0]
    if rank is None or rank >= n:
        eigvals, eigvecs = scipy.linalg.eigh(block.toarray())
        return np.clip(eigvals, 0.0, None), eigvecs

    # Deterministic ARPACK start vector so the truncated basis is reproducible: eigsh
    # otherwise draws a random v0, which rotates the returned eigenvectors within
    # (near-)degenerate eigenspaces run-to-run. The eigenvalues and the diffusion /
    # penalty operators are invariant to that rotation, but a fixed v0 makes the basis
    # -- and hence fitted place fields -- reproducible. A generic (non-eigenvector)
    # direction avoids ARPACK stagnating on v0 == the constant null mode.
    v0 = np.random.default_rng(0).standard_normal(n)
    try:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
            block, k=rank, sigma=-1e-8, which="LM", v0=v0
        )
    except RuntimeError:
        # Shift-invert can fail on the (near-)singular Laplacian in some builds,
        # raising either "Factor is exactly singular" (a plain RuntimeError from the
        # sparse LU factorization) or an ArpackError (itself a RuntimeError
        # subclass). Both are caught here; fall back to the no-shift-invert
        # smallest-magnitude solver, which is less reliable, so warn.
        warnings.warn(
            "Shift-invert eigsh failed; falling back to the no-shift-invert "
            "which='SM' solver, which may return a lower-quality eigenbasis. "
            "Consider a smaller rank or the full (rank=None) decomposition.",
            UserWarning,
            stacklevel=2,
        )
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(block, k=rank, which="SM", v0=v0)

    order = np.argsort(eigvals)
    return np.clip(eigvals[order], 0.0, None), eigvecs[:, order]


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

    Notes
    -----
    On a **disconnected** graph each connected component is decomposed separately so
    every returned eigenvector is localized to a single component (zero elsewhere),
    then the globally smallest ``rank`` modes are kept. This matters for truncation:
    a single global ``eigsh`` can return eigenvectors rotated *across* components
    within a degenerate eigenspace, and a truncation that cut through such an
    eigenspace would smear a point source across a wall (leaking mass after the
    clip/renormalize in :func:`diffuse`). Component-local modes make truncation
    (and slicing a cached full basis) leak-free by construction.
    """
    n_bins = L.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(L, directed=False)
    if rank is not None:
        _require_rank_covers_components(rank, n_components)

    if n_components == 1:
        return _block_eigenbasis(L, rank)

    # Disconnected: decompose each component so modes are component-local, then keep
    # the globally smallest `rank` (each component contributes at most `rank`; the
    # `n_components` zero modes are the smallest, so all are retained).
    L = L.tocsr()
    per_component_rank = n_bins if rank is None else rank
    eigval_parts: list[np.ndarray] = []
    eigvec_parts: list[np.ndarray] = []
    for component in range(n_components):
        idx = np.flatnonzero(labels == component)
        block = L[idx][:, idx]
        block_rank = None if per_component_rank >= idx.size else per_component_rank
        block_vals, block_vecs = _block_eigenbasis(block, block_rank)
        padded = np.zeros((n_bins, block_vecs.shape[1]))
        padded[idx] = block_vecs
        eigval_parts.append(block_vals)
        eigvec_parts.append(padded)

    all_eigvals = np.concatenate(eigval_parts)
    all_eigvecs = np.concatenate(eigvec_parts, axis=1)
    keep = all_eigvals.size if rank is None else rank
    order = np.argsort(all_eigvals, kind="stable")[:keep]
    return all_eigvals[order], all_eigvecs[:, order]


def _adaptive_heat_kernel_basis(
    laplacian: scipy.sparse.spmatrix,
    sigma: float,
    tol: float,
    dense_fraction: float,
) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
    """Resolve the heat-kernel truncation rank and return the basis at that rank.

    The adaptive probe's *final* eigendecomposition is the returned basis (sliced to the
    resolved rank), so a caller building the basis does a single eigensolve rather than
    resolving the rank and then recomputing. Returns ``(None, None, None)`` when the
    bandwidth needs most modes (rank would exceed ``dense_fraction * n_bins``), signalling
    the caller to use the full dense basis. See :func:`heat_kernel_rank` for the math.
    """
    if not sigma > 0:
        raise ValidationError(
            "smoothing bandwidth (sigma) must be positive",
            expected="sigma > 0",
            got=f"sigma = {sigma}",
        )
    if not 0.0 < tol < 1.0:
        raise ValidationError(
            "heat-kernel truncation tolerance must lie in (0, 1)",
            expected="0 < tol < 1",
            got=f"tol = {tol}",
            hint="tol is a heat-kernel weight exp(-t*lambda); its cutoff -ln(tol) is "
            "only positive for tol in (0, 1).",
        )

    n_bins = laplacian.shape[0]
    n_components = scipy.sparse.csgraph.connected_components(
        laplacian, directed=False, return_labels=False
    )
    t = sigma**2 / 2.0
    lambda_cut = -np.log(tol) / t
    # Beyond this rank a truncated eigsh no longer beats a dense eigh, so use dense.
    max_trunc = int(dense_fraction * n_bins)
    if max_trunc <= n_components:
        return None, None, None

    k = min(max(n_components + 1, _HEAT_KERNEL_RANK_START), max_trunc)
    while True:
        eigvals, eigvecs = diffusion_eigenbasis(laplacian, rank=k)
        if eigvals[-1] >= lambda_cut:
            # Bracketed the cutoff: keep every mode at or below it (all null modes are).
            keep = max(
                int(np.searchsorted(eigvals, lambda_cut, side="right")), n_components
            )
            return keep, eigvals[:keep], eigvecs[:, :keep]
        if k >= max_trunc:
            return None, None, None  # cutoff not reached below the dense threshold
        # 2D Weyl: the eigenvalue-counting function grows ~linearly, so estimate the
        # index at lambda_cut (10% margin) and jump there, at least doubling k.
        estimated = int(np.ceil(1.1 * k * lambda_cut / max(float(eigvals[-1]), 1e-30)))
        if estimated > max_trunc:
            return None, None, None
        k = min(max(estimated, 2 * k), max_trunc)


def heat_kernel_rank(
    laplacian: scipy.sparse.spmatrix,
    sigma: float,
    tol: float = _HEAT_KERNEL_RANK_TOL,
    dense_fraction: float = _HEAT_KERNEL_DENSE_FRACTION,
) -> int | None:
    """Number of eigenmodes the heat kernel ``exp(-t L)`` needs at bandwidth ``sigma``.

    :func:`diffuse` weights mode ``k`` by ``exp(-t λ_k)``, ``t = sigma**2 / 2``, so modes
    with ``exp(-t λ_k) < tol`` are negligible: dropping them changes the smoothed field
    by at most ``tol * ‖field‖`` (their combined energy is below ``tol² ‖field‖²``). This
    returns the smallest rank that keeps every mode with weight ``>= tol`` — a
    near-lossless truncation whose size tracks the physical smoothing scale (``~ area /
    sigma²``), **not** the number of bins, so it stays small as the grid is refined.

    The rank is found adaptively without a full eigendecomposition: compute the smallest
    ``k`` eigenvalues; if the largest is still below the cutoff ``λ_cut = -ln(tol) / t``,
    jump ``k`` toward the estimated cutoff index (the 2D eigenvalue-counting function is
    ~linear, Weyl) and retry — typically one or two probes. To build the basis (not just
    query the rank), use :func:`cached_heat_kernel_eigenbasis`, which reuses the probe's
    final eigensolve instead of recomputing.

    Parameters
    ----------
    laplacian : scipy.sparse.spmatrix, shape (n_bins, n_bins)
        Symmetric graph Laplacian from :func:`build_laplacian`.
    sigma : float
        Smoothing standard deviation (``position_std``) in coordinate units; positive.
    tol : float, optional
        Heat-kernel weight below which a mode is dropped, in ``(0, 1)``. Smaller ``tol``
        keeps more modes (more accurate, larger rank). Default ``1e-6`` (error ``~1e-7``).
    dense_fraction : float, optional
        Return ``None`` (use the dense solver) when the resolved rank would exceed
        ``dense_fraction * n_bins``, where a truncated ``eigsh`` no longer beats a dense
        ``eigh``. Default ``0.5``.

    Returns
    -------
    rank : int or None
        The truncation rank (``>= n_components``, so every component's null mode is kept
        and :func:`diffuse` cannot leak mass across components), or ``None`` to signal the
        caller should use the full dense basis — a light bandwidth needing most modes.

    Raises
    ------
    ValidationError
        If ``sigma <= 0`` or ``tol`` is not in ``(0, 1)``.
    """
    return _adaptive_heat_kernel_basis(laplacian, sigma, tol, dense_fraction)[0]


def connected_component_labels(graph: nx.Graph) -> np.ndarray:
    """Per-node connected-component id (``0..n_components-1``), indexed by node.

    ``graph`` has contiguous integer nodes ``0..n_nodes-1`` (as built by
    :func:`build_laplacian` / :func:`environment_graph`), so the returned labels are
    aligned with the rows of the eigenbasis and the diffused fields.
    """
    labels = np.empty(graph.number_of_nodes(), dtype=int)
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        labels[list(nodes)] = component_id
    return labels


def n_connected_components(environment: "Environment") -> int:
    """Number of connected components of the environment's interior-bin graph.

    Each component contributes one Laplacian null (zero-eigenvalue) mode, so a truncated
    eigenbasis must keep **at least** this many modes (see :func:`diffusion_eigenbasis` /
    :func:`_require_rank_covers_components`). Callers that cap a default rank use this to
    ensure the cap never drops a component's null mode. Builds (and reuses) the cached
    Laplacian via :func:`environment_graph`.
    """
    environment_graph(environment)
    return int(
        scipy.sparse.csgraph.connected_components(
            environment._diffusion_laplacian_, directed=False, return_labels=False
        )
    )


def diffuse(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    sigma: float,
    fields: np.ndarray,
    component_labels: np.ndarray | None = None,
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
    component_labels : np.ndarray, shape (n_bins,), optional
        Connected-component id per bin (see :func:`connected_component_labels`).
        When given, mass is renormalized within each component; when None, over the
        whole column. Pass labels on disconnected graphs so clipping in one component
        cannot shift mass into another.

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

    On a disconnected graph the heat kernel conserves each *component's* mass
    independently, but a single global renormalization would redistribute mass
    between components whenever truncation lobes are clipped unevenly. Passing
    ``component_labels`` renormalizes per component to preserve that invariant.
    """
    t = sigma**2 / 2.0
    coeff = np.exp(-t * eigvals)  # (m,)
    proj = eigvecs.T @ fields  # (m, n_fields)
    smoothed = eigvecs @ (coeff[:, None] * proj)  # (n_bins, n_fields)
    clipped = np.clip(smoothed, 0.0, None)

    def _rescale_rows_to_input_mass(rows: np.ndarray) -> np.ndarray:
        """Scale the clipped rows so their per-field mass matches the input rows'."""
        input_mass = fields[rows].sum(axis=0)
        clipped_mass = clipped[rows].sum(axis=0)
        scale = np.divide(
            input_mass,
            clipped_mass,
            out=np.zeros_like(input_mass, dtype=float),
            where=clipped_mass > 0,
        )
        return clipped[rows] * scale

    if component_labels is None:
        return _rescale_rows_to_input_mass(np.arange(clipped.shape[0]))

    renormalized = np.empty_like(clipped, dtype=float)
    for label in np.unique(component_labels):
        rows = np.flatnonzero(component_labels == label)
        renormalized[rows] = _rescale_rows_to_input_mass(rows)
    return renormalized


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


def _freeze_and_cache(
    cache: dict, rank: int | None, eigvals: np.ndarray, eigvecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze a basis read-only (it is shared across neurons and EM refits), store it
    under ``rank``, and return it. Centralizes the read-only cache contract."""
    eigvals.setflags(write=False)
    eigvecs.setflags(write=False)
    basis = (eigvals, eigvecs)
    cache[rank] = basis
    return basis


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
    return _freeze_and_cache(cache, rank, eigvals, eigvecs)


def cached_heat_kernel_eigenbasis(
    environment: "Environment",
    sigma: float,
    tol: float = _HEAT_KERNEL_RANK_TOL,
    dense_fraction: float = _HEAT_KERNEL_DENSE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (and cache) the bandwidth-aware auto-truncated eigenbasis for ``sigma``.

    Resolves the heat-kernel rank (see :func:`heat_kernel_rank`) and returns the
    eigenbasis at that rank in a **single** eigendecomposition — the adaptive probe's
    final eigensolve is reused as the basis, rather than resolving the rank and then
    recomputing it. The result is stored in the same rank-keyed cache as
    :func:`cached_eigenbasis` (on ``environment._diffusion_eigenbasis_``), and the
    resolved rank is memoized per ``(sigma, tol, dense_fraction)`` on
    ``environment._diffusion_heat_kernel_rank_``, so EM refits at the same bandwidth
    reuse the basis **without re-running the adaptive probe** (zero eigensolves). When
    the bandwidth needs most modes the resolver falls back to the full dense basis. All
    diffusion caches are invalidated together by ``Environment.fit_place_grid``.

    Parameters
    ----------
    environment : Environment
        A fitted environment.
    sigma : float
        Smoothing standard deviation (``position_std``) in coordinate units.
    tol, dense_fraction : float, optional
        Forwarded to :func:`heat_kernel_rank`.

    Returns
    -------
    eigvals : np.ndarray, shape (m,)
    eigvecs : np.ndarray, shape (n_interior, m)
        The (read-only) cached eigenbasis at the resolved rank.
    """
    rank_cache = getattr(environment, "_diffusion_heat_kernel_rank_", None)
    if rank_cache is None:
        rank_cache = {}
        environment._diffusion_heat_kernel_rank_ = rank_cache

    key = (sigma, tol, dense_fraction)
    if key in rank_cache:
        # Rank already resolved for this bandwidth; the matching basis was cached below
        # on the first fit, so this reuses it with no adaptive probe and no eigensolve.
        return cached_eigenbasis(environment, rank_cache[key])

    environment_graph(environment)  # ensures the cached Laplacian snapshot
    rank, eigvals, eigvecs = _adaptive_heat_kernel_basis(
        environment._diffusion_laplacian_, sigma, tol, dense_fraction
    )
    rank_cache[key] = rank
    if rank is None:
        return cached_eigenbasis(
            environment, None
        )  # light bandwidth -> dense full basis

    cache = getattr(environment, "_diffusion_eigenbasis_", None)
    if cache is None:
        cache = {}
        environment._diffusion_eigenbasis_ = cache
    if rank in cache:
        return cache[rank]
    # Independent copies of the probe's final basis (sliced to the resolved rank) so the
    # frozen cache entry does not alias the probe's arrays.
    return _freeze_and_cache(
        cache, rank, np.ascontiguousarray(eigvals), np.ascontiguousarray(eigvecs)
    )


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
