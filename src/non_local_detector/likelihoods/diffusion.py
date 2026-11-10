"""Graph-diffusion spectral engine for manifold-aware place-field smoothing.

This module discretizes the heat equation on an environment's interior bins and
applies its kernel for ``t = sigma**2 / 2`` as a smoother that respects track
geometry (walls, holes, junctions) in 1D and N-D.

Unlike Gaussian KDE, which smooths in the coordinate metric and smears across
walls, graph diffusion smooths along the diffusion distance on the domain:
reflecting (Neumann) boundaries are intrinsic (no edges to absent bins) and the
operator conserves each field's total mass.

The operator is the **finite-volume** heat equation. Adjacent bins ``i`` and
``j`` exchange mass through a conductance ``c_ij``, and bin ``i`` has the volume
``m_i`` (its width on a track, its area or volume on a grid), so the masses ``p``
evolve as ``dp/dt = -(D - C) M⁻¹ p``, with ``C = [c_ij]``, ``D`` its row sums and
``M = diag(m)``:

- On a grid, face-adjacent bins at spacing ``h`` sharing a face of area ``A``
  have ``c = A / h``. Diagonal / Moore neighbors are not coupled.
- On a track graph, the bin centers, bin edges and track nodes form a resistor
  network with conductance ``1 / length`` per segment, and eliminating every
  node that is not a bin center (a Kron reduction) gives ``c_ij``: ``1 / h``
  between neighbors at separation ``h``, and ``g_i g_j / sum_k g_k`` between the
  end bins of the arms meeting at a vertex, ``g_k`` the conductance from end bin
  ``k`` to it. This is the Kirchhoff condition of diffusion on a metric graph:
  mass crossing a vertex of degree ``d`` splits evenly among its ``d`` arms.

On equal bins this is the finite-difference Laplacian with weight ``1 / h**2``.
The heat kernel is a Gaussian of standard deviation ``sigma`` whatever the bin
size, at junctions and across unequal bins too, and its long-time limit is a
uniform density.

The eigendecomposition is of the symmetric form ``L = M^{-1/2} (D - C) M^{-1/2}``
(see :func:`build_laplacian`); the mass kernel is ``M^{1/2} exp(-t L) M^{-1/2}``.
Because the eigendecomposition depends only on the graph (not ``sigma`` and not
the fields being smoothed), it is computed once and cached on the ``Environment``
via ``cached_eigenbasis``, then reused across all neurons and all EM refits.
"""

import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg

from non_local_detector._warnings import user_stacklevel
from non_local_detector.exceptions import (
    NumericalWarning,
    ParameterWarning,
    ValidationError,
)

if TYPE_CHECKING:
    from non_local_detector.environment import Environment


# Defaults for heat_kernel_rank (bandwidth-aware auto-truncation): drop modes whose
# heat-kernel weight exp(-t*lambda) is below _HEAT_KERNEL_RANK_TOL, and fall back to the
# dense eigendecomposition once the resolved rank exceeds _HEAT_KERNEL_DENSE_FRACTION *
# n_bins (where a truncated eigsh no longer beats a dense eigh). A shift-invert eigsh of
# k = f * n_bins modes costs as much as a dense eigh near f = 0.23-0.28 on 1-D paths of
# 1500-8000 bins and a 60x60 grid, and near f = 0.35 on a 100x100 grid, so the cutoff
# sits just below that range.
# _HEAT_KERNEL_RANK_START is the first probe size for the adaptive rank search.
_HEAT_KERNEL_RANK_TOL = 1e-6
_HEAT_KERNEL_DENSE_FRACTION = 0.2
_HEAT_KERNEL_RANK_START = 32

# Diffused-field values below this fraction of their column's maximum are not
# resolved: the float32 spectral product and the heat-kernel rank truncation
# (``_HEAT_KERNEL_RANK_TOL``) both perturb them by up to about this much.
FIELD_RESOLUTION = max(float(np.finfo(np.float32).eps), _HEAT_KERNEL_RANK_TOL)

# The diffusion operator this version builds. Encoding models whose fitted arrays
# depend on it record it, and decoding one that records another asks for a refit.
DIFFUSION_OPERATOR = "finite_volume"

# Keys of this module's entries in the Environment cache (``Environment.get_cached``),
# all cleared together by ``Environment.fit_place_grid``. The persistent ones name
# the operator: an environment unpickled from a version that built another one
# carries that operator's entries under other keys, which are never read.
_GRAPH_KEY = f"diffusion_graph:{DIFFUSION_OPERATOR}"
_LAPLACIAN_KEY = f"diffusion_laplacian:{DIFFUSION_OPERATOR}"
_EIGENBASIS_KEY = f"diffusion_eigenbasis:{DIFFUSION_OPERATOR}"
_HEAT_KERNEL_RANK_KEY = f"diffusion_heat_kernel_rank:{DIFFUSION_OPERATOR}"
_DEVICE_BASIS_KEY = "diffusion_device_basis"


def build_laplacian(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    """Symmetric finite-volume diffusion operator ``L = M^{-1/2} (D - C) M^{-1/2}``.

    ``C`` holds the edge conductances, ``D`` is its row sums and ``M`` the node
    volumes. The masses ``p`` of the bins evolve as ``dp/dt = -(D - C) M⁻¹ p``,
    which is ``M^{1/2} L M^{-1/2}``, so the mass kernel is
    ``M^{1/2} exp(-t L) M^{-1/2}``: it conserves total mass, and its long-time
    limit has mass proportional to volume (uniform density). On equal volumes
    ``h**d`` with face conductances ``h**(d-2)`` this is the finite-difference
    Laplacian with weight ``1 / h**2``. ``L`` is symmetric positive
    semi-definite with one null mode per connected component, ``sqrt(m)``
    restricted to the component.

    Parameters
    ----------
    graph : nx.Graph
        Interior-bin graph with contiguous integer nodes ``0..n_nodes-1``. Each
        node carries a positive ``'volume'`` and each edge a positive
        ``'conductance'`` (see :func:`environment_graph`).

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n_nodes, n_nodes)
        The sparse symmetric operator.

    Raises
    ------
    ValidationError
        If a node's ``'volume'`` or an edge's ``'conductance'`` is not a positive
        finite number.
    """
    n_nodes = graph.number_of_nodes()
    volumes = np.array([graph.nodes[node]["volume"] for node in range(n_nodes)])
    if not np.all(np.isfinite(volumes) & (volumes > 0)):
        msg = "graph node has a volume that is not a positive finite number"
        raise ValidationError(
            msg,
            expected="every node 'volume' > 0 and finite",
            got=f"volumes {volumes[~(np.isfinite(volumes) & (volumes > 0))]}",
        )
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        conductance = data["conductance"]
        if not 0 < conductance < np.inf:  # also catches NaN
            msg = "graph edge has a conductance that is not a positive finite number"
            raise ValidationError(
                msg,
                expected="every edge 'conductance' > 0 and finite",
                got=f"conductance = {conductance} on edge ({u}, {v})",
            )
        rows += [u, v]
        cols += [v, u]
        vals += [conductance, conductance]

    conductances = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_nodes, n_nodes)
    )
    degree = np.asarray(conductances.sum(axis=1)).ravel()
    scale = scipy.sparse.diags(1.0 / np.sqrt(volumes))
    return (scale @ (scipy.sparse.diags(degree) - conductances) @ scale).tocsr()


def _require_rank_covers_components(rank: int, n_components: int) -> None:
    """Raise if a truncated ``rank`` would drop a connected component's null mode."""
    if rank < n_components:
        msg = "rank is too small to retain every connected component's null mode"
        raise ValidationError(
            msg,
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
            NumericalWarning,
            stacklevel=user_stacklevel(),
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


def _weyl_exponent(eigvals: np.ndarray) -> float:
    """Growth exponent ``d / 2`` of the eigenvalue-counting function ``N(λ) ~ λ^(d/2)``.

    Weyl's law gives ``d = 1`` on a linearized track and ``d = 2`` on an open field.
    ``d`` is read off the probe's own spectrum, from how much ``λ`` grows while the
    mode count doubles over its upper half, and rounded to 1 or 2 (a higher-dimensional
    spectrum is capped at ``d = 2``, which underestimates its growth).

    Parameters
    ----------
    eigvals : np.ndarray, shape (k,)
        The ``k`` smallest Laplacian eigenvalues, ascending.

    Returns
    -------
    exponent : float
        ``0.5`` or ``1.0``; ``1.0`` when the upper half has no nonzero spread to
        measure (e.g. it is all null modes).
    """
    n_modes = eigvals.size
    half = n_modes // 2
    low, high = float(eigvals[half - 1]), float(eigvals[-1])
    if not 0.0 < low < high:
        return 1.0
    local = np.log(n_modes / half) / np.log(high / low)
    return cast(float, min(max(round(2.0 * local), 1), 2) / 2.0)


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
    bandwidth needs many modes (rank would exceed ``dense_fraction * n_bins``), signalling
    the caller to use the full dense basis. See :func:`heat_kernel_rank` for the math.
    """
    if not sigma > 0:
        msg = "smoothing bandwidth (sigma) must be positive"
        raise ValidationError(
            msg,
            expected="sigma > 0",
            got=f"sigma = {sigma}",
        )
    if not 0.0 < tol < 1.0:
        msg = "heat-kernel truncation tolerance must lie in (0, 1)"
        raise ValidationError(
            msg,
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
        # Weyl: the eigenvalue-counting function grows ~lambda**(d/2), so extrapolate
        # the index at lambda_cut (10% margin) and jump there, at least doubling k.
        ratio = lambda_cut / max(float(eigvals[-1]), 1e-30)
        estimated = int(np.ceil(1.1 * k * ratio ** _weyl_exponent(eigvals)))
        if estimated > max_trunc:
            return None, None, None
        k = min(max(estimated, 2 * k), max_trunc)


def heat_kernel_rank(
    laplacian: scipy.sparse.spmatrix,
    sigma: float,
    tol: float = _HEAT_KERNEL_RANK_TOL,
    dense_fraction: float = _HEAT_KERNEL_DENSE_FRACTION,
) -> int | None:
    """Return the number of eigenmodes the heat kernel ``exp(-t L)`` needs at bandwidth ``sigma``.

    :func:`diffuse` weights mode ``k`` by ``exp(-t λ_k)``, ``t = sigma**2 / 2``, so modes
    with ``exp(-t λ_k) < tol`` are negligible: dropping them changes the smoothed field
    by at most ``tol * ‖field‖`` (their combined energy is below ``tol² ‖field‖²``). This
    returns the smallest rank that keeps every mode with weight ``>= tol`` — a
    near-lossless truncation whose size tracks the physical smoothing scale (``~ area /
    sigma²``), **not** the number of bins, so it stays small as the grid is refined.

    The rank is found adaptively without a full eigendecomposition: compute the smallest
    ``k`` eigenvalues; if the largest is still below the cutoff ``λ_cut = -ln(tol) / t``,
    jump ``k`` toward the estimated cutoff index and retry — typically one or two probes.
    The estimate extrapolates the eigenvalue-counting function ``N(λ) ~ λ^(d/2)``
    (Weyl), with ``d = 1`` on a linearized track and ``d = 2`` on an open field, read
    off the probe's own spectrum. To build the basis (not just query the rank), use
    :func:`cached_heat_kernel_eigenbasis`, which reuses the probe's final eigensolve
    instead of recomputing.

    Parameters
    ----------
    laplacian : scipy.sparse.spmatrix, shape (n_bins, n_bins)
        Symmetric graph Laplacian from :func:`build_laplacian`.
    sigma : float
        Smoothing standard deviation (``position_std``) in coordinate units; positive.
    tol : float, optional
        Heat-kernel weight below which a mode is dropped, in ``(0, 1)``. Smaller ``tol``
        keeps more modes (more accurate, larger rank). Default ``1e-6`` (truncation
        error below ``1e-6 * ‖field‖``).
    dense_fraction : float, optional
        Return ``None`` (use the dense solver) when the resolved rank would exceed
        ``dense_fraction * n_bins``, where a truncated ``eigsh`` no longer beats a dense
        ``eigh``. Default ``0.2``.

    Returns
    -------
    rank : int or None
        The truncation rank (``>= n_components``, so every component's null mode is kept
        and :func:`diffuse` cannot leak mass across components), or ``None`` to signal the
        caller should use the full dense basis — a light bandwidth needing many modes.

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

    Parameters
    ----------
    graph : nx.Graph
        Interior-bin graph with contiguous integer nodes ``0..n_nodes-1``.

    Returns
    -------
    labels : np.ndarray, shape (n_nodes,)
        Connected-component id (``0..n_components-1``) for each node, indexed by node.
    """
    labels = np.empty(graph.number_of_nodes(), dtype=int)
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        labels[list(nodes)] = component_id
    return labels


def n_connected_components(environment: "Environment") -> int:
    """Return the number of connected components of the environment's interior-bin graph.

    Each component contributes one Laplacian null (zero-eigenvalue) mode, so a truncated
    eigenbasis must keep **at least** this many modes (see :func:`diffusion_eigenbasis` /
    :func:`_require_rank_covers_components`). Callers that cap a default rank use this to
    ensure the cap never drops a component's null mode. Builds (and reuses) the cached
    Laplacian via :func:`environment_graph`.

    Parameters
    ----------
    environment : Environment
        A fitted environment.

    Returns
    -------
    n_components : int
        Number of connected components of the interior-bin graph.
    """
    environment_graph(environment)
    return int(
        scipy.sparse.csgraph.connected_components(
            environment.get_cached(_LAPLACIAN_KEY),
            directed=False,
            return_labels=False,
        )
    )


def diffuse(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    sigma: float,
    fields: np.ndarray,
    component_labels: np.ndarray | None = None,
    bin_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Diffuse per-bin masses ``fields`` for ``t = sigma**2 / 2``.

    Applies the mass kernel ``M^{1/2} exp(-t L) M^{-1/2}`` (see
    :func:`build_laplacian`), computed in the eigenbasis as
    ``M^{1/2} Q (exp(-tΛ) ⊙ (Qᵀ M^{-1/2} F))``, never materializing the dense
    ``(n_bins, n_bins)`` kernel. All fields (occupancy + every neuron's count
    field) diffuse in a single batched matmul.

    Parameters
    ----------
    eigvals : np.ndarray, shape (m,)
        Eigenvalues from :func:`diffusion_eigenbasis`.
    eigvecs : np.ndarray, shape (n_bins, m)
        Corresponding eigenvectors as columns.
    sigma : float
        Smoothing standard deviation in coordinate units (bandwidth).
    fields : np.ndarray, shape (n_bins, n_fields)
        Count fields (masses) on the interior bins, one column per field.
    component_labels : np.ndarray, shape (n_bins,), optional
        Connected-component id per bin (see :func:`connected_component_labels`), by
        default None. When given, mass is renormalized within each component; when
        None, over the whole column. Pass labels on disconnected graphs so clipping in
        one component cannot shift mass into another.
    bin_sizes : np.ndarray, shape (n_bins,), optional
        The bin volumes ``M`` the operator was built with (see
        :func:`environment_graph`), by default None: equal volumes. Only their
        ratios enter.

    Returns
    -------
    smoothed : np.ndarray, shape (n_bins, n_fields)
        Diffused fields, clipped to be non-negative and renormalized so each
        column's total mass equals the input field's total (mass-conserving).

    Notes
    -----
    The kernel conserves each field's total mass because every component's null
    mode (``sqrt(m)``) is always retained. At full rank the kernel is already
    non-negative, so clipping only removes round-off. A truncated basis, however,
    can produce non-tiny negative lobes (e.g. for a point source); clipping those
    alone would inflate the total, so each column is renormalized back to its
    input sum.

    On a disconnected graph the heat kernel conserves each *component's* mass
    independently, but a single global renormalization would redistribute mass
    between components whenever truncation lobes are clipped unevenly. Passing
    ``component_labels`` renormalizes per component to preserve that invariant.
    """
    t = sigma**2 / 2.0
    coeff = np.exp(-t * eigvals)  # (m,)
    # sqrt of the volumes relative to the largest, exactly 1 on equal volumes.
    root = (
        np.ones((fields.shape[0], 1))
        if bin_sizes is None
        else np.sqrt(bin_sizes / np.max(bin_sizes))[:, None]
    )
    proj = eigvecs.T @ (fields / root)  # (m, n_fields)
    smoothed = root * (eigvecs @ (coeff[:, None] * proj))  # (n_bins, n_fields)
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
        return cast(np.ndarray, clipped[rows] * scale)

    if component_labels is None:
        return _rescale_rows_to_input_mass(np.arange(clipped.shape[0]))

    renormalized = np.empty_like(clipped, dtype=float)
    for label in np.unique(component_labels):
        rows = np.flatnonzero(component_labels == label)
        renormalized[rows] = _rescale_rows_to_input_mass(rows)
    return renormalized


@partial(jax.jit, static_argnames=("n_components",))
def _component_mass_sum(
    fields: jax.Array, labels: jax.Array, n_components: int
) -> jax.Array:
    """Sum each component's masses, one masked reduction per component.

    Masking and summing are fused; no component-by-node-by-field array is
    constructed. Each sum is an XLA reduction rather than a sequential
    float32 accumulation. Cost is O(n_components * n_bins * n_fields).
    """
    output = jnp.zeros((n_components, *fields.shape[1:]), dtype=fields.dtype)

    def sum_component(component: jax.Array, masses: jax.Array) -> jax.Array:
        mask = (labels == component).reshape((-1, *((1,) * (fields.ndim - 1))))
        mass = jnp.sum(jnp.where(mask, fields, 0), axis=0)
        return masses.at[component].set(mass, unique_indices=True)

    return cast(jax.Array, jax.lax.fori_loop(0, n_components, sum_component, output))


@partial(jax.jit, static_argnames=("sigma", "n_components"))
def _heat_kernel_apply_jit(
    eigvals: jax.Array,
    eigvecs: jax.Array,
    sigma: float,
    fields: jax.Array,
    labels: jax.Array,
    n_components: int,
) -> jax.Array:
    """Jitted core of :func:`heat_kernel_apply` (see it for the full contract)."""
    t = sigma**2 / 2.0
    coeff = jnp.exp(-t * eigvals)  # (m,)
    projected = jnp.matmul(eigvecs.T, fields, precision=jax.lax.Precision.HIGHEST)
    smoothed = jnp.matmul(
        eigvecs, coeff[:, None] * projected, precision=jax.lax.Precision.HIGHEST
    )  # (n_bins, n_fields)
    clipped = jnp.clip(smoothed, 0.0, None)
    # Per-component mass rescale; labels are 0..n_components-1.
    in_mass = _component_mass_sum(fields, labels, n_components)
    cl_mass = _component_mass_sum(clipped, labels, n_components)
    scale = jnp.where(cl_mass > 0, in_mass / jnp.where(cl_mass > 0, cl_mass, 1.0), 0.0)
    return cast(jax.Array, clipped * scale[labels])  # gather back to (n_bins, n_fields)


def heat_kernel_apply(
    eigvals: jax.Array,
    eigvecs: jax.Array,
    sigma: float,
    fields: jax.Array,
    component_labels: jax.Array | None = None,
    n_components: int | None = None,
) -> jax.Array:
    """JAX port of ``diffuse`` on equal volumes, clipped and mass-rescaled.

    Computes ``Q exp(-tΛ) Qᵀ F`` and mirrors :func:`diffuse` with
    ``bin_sizes=None`` exactly (clip to >=0, then rescale each column to its input
    mass, per connected component) so likelihood magnitude is truncation-rank
    stable. Jittable; the mass rescale reduces one
    component at a time (see ``_component_mass_sum``). ``component_labels`` are
    0..K-1 segment ids (a device/host int array; a single connected graph passes
    all zeros or None). With the basis of :func:`get_device_basis`, a unit point
    source's column normalized to unit mass is the mass kernel's column on any
    volumes.

    Parameters
    ----------
    eigvals : jax.Array, shape (m,)
        Eigenvalues from :func:`diffusion_eigenbasis`.
    eigvecs : jax.Array, shape (n_bins, m)
        Corresponding eigenvectors as columns.
    sigma : float
        Smoothing standard deviation in coordinate units (bandwidth).
    fields : jax.Array, shape (n_bins, n_fields)
        Count fields on the interior bins, one column per field.
    component_labels : jax.Array, shape (n_bins,), optional
        Connected-component id per bin, 0..K-1, by default None (treated as a
        single component).
    n_components : int, optional
        The **static** component count — pass it explicitly (the production
        caller gets it from :func:`get_device_basis`) to avoid a per-call
        device->host sync of the labels. If ``None``, it is derived host-side
        once (convenience for tests / one-off calls only — NOT the hot predict
        path).

    Returns
    -------
    smoothed : jax.Array, shape (n_bins, n_fields)
        Diffused fields, clipped to be non-negative and renormalized so each
        column's total mass equals the input field's total, per component
        (mass-conserving).
    """
    n_bins = eigvecs.shape[0]
    if component_labels is None:
        labels = jnp.zeros(n_bins, dtype=jnp.int32)
        n_components = 1
    else:
        labels = jnp.asarray(component_labels, dtype=jnp.int32)
        if n_components is None:
            n_components = (
                int(np.asarray(component_labels).max()) + 1
            )  # host sync — non-hot path only
    return cast(
        jax.Array,
        _heat_kernel_apply_jit(
            eigvals, eigvecs, float(sigma), fields, labels, int(n_components)
        ),
    )


def to_density(smoothed: np.ndarray, bin_sizes: np.ndarray) -> np.ndarray:
    """Convert each column of per-bin masses to an integral-one density.

    ``exp(-t L)`` conserves field *sums*, so smoothed count fields hold the mass
    in each bin. Bin ``i`` gets ``(smoothed_i / Σ smoothed) / bin_sizes_i``, so
    ``bin_sizes @ density == 1`` and ``bin_sizes * density`` is the fraction of
    the column's mass in each bin. Columns with zero mass map to all zeros.

    Parameters
    ----------
    smoothed : np.ndarray, shape (n_bins, n_fields)
        Smoothed (non-negative) per-bin masses, e.g. weighted counts or seconds.
    bin_sizes : np.ndarray, shape (n_bins,)
        Positive per-bin volumes (1D width, 2D area, ...).

    Returns
    -------
    density : np.ndarray, shape (n_bins, n_fields)
        Column-normalized densities; zero-mass columns are zero.
    """
    mass = smoothed.sum(axis=0)  # (n_fields,)
    safe = np.where(mass > 0, mass, 1.0)
    return np.where(mass > 0, smoothed / safe / bin_sizes[:, np.newaxis], 0.0)


def resolved_occupancy(occupancy: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
    """Bins whose diffused occupancy the heat kernel resolves.

    Occupancy is itself a diffused field, so in unvisited bins it is numerical
    noise below ``FIELD_RESOLUTION`` of its maximum. Those bins carry no
    encoding information: a ratio of a diffused spike field to this noise is
    not a rate, so the likelihoods treat them as zero-occupancy bins. A bin
    with zero occupancy is never resolved, even when no bin has any.

    Parameters
    ----------
    occupancy : array, shape (n_bins,)
        Diffused occupancy per bin.

    Returns
    -------
    resolved : array of bool, shape (n_bins,)
    """
    return (occupancy > 0.0) & (occupancy >= FIELD_RESOLUTION * occupancy.max())


def check_diffusion_operator(diffusion_operator: str | None) -> None:
    """Raise unless an encoding model was fitted with this version's operator.

    An encoding model that decodes by rebuilding the heat kernel from its
    environment combines that kernel with arrays it fitted with its own
    operator, so the two must be the same operator.

    Parameters
    ----------
    diffusion_operator : str or None
        The operator the encoding model records (``DIFFUSION_OPERATOR`` at its
        fit), or None for a model fitted before models recorded it.

    Raises
    ------
    ValidationError
        If ``diffusion_operator`` is not ``DIFFUSION_OPERATOR``.
    """
    if diffusion_operator == DIFFUSION_OPERATOR:
        return
    msg = (
        "The clusterless_diffusion encoding model was fitted with another "
        "diffusion operator than this version builds: its occupancy and "
        "intensities come from that operator, and decoding would combine them "
        "with this version's heat kernel. The diffusion operator changed to "
        "couple track junctions and unequal bin widths correctly."
    )
    raise ValidationError(
        msg,
        expected=f"diffusion_operator = {DIFFUSION_OPERATOR!r}",
        got=(
            "no recorded diffusion operator"
            if diffusion_operator is None
            else f"diffusion_operator = {diffusion_operator!r}"
        ),
        hint=(
            "Refit the model with the installed version, or decode with the "
            "version that fitted it."
        ),
    )


def environment_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Build the interior-bin graph, node order, and per-bin volumes for diffusion.

    Two branches, selected by ``environment.track_graph``:

    - **N-D / 1D grid** (``track_graph is None``): the subgraph of
      ``track_graphDD`` induced by interior bins, keeping only **face-adjacent**
      edges (bin centers differing in exactly one dimension), each with the
      conductance ``face area / center distance``; diagonal / Moore neighbors
      share no face.
    - **Linearized track graph**: the interior bin centers of
      ``track_graph_with_bin_centers_edges_``, coupled by the conductances of its
      resistor network with every other node eliminated (see
      :func:`_center_conductances`); junctions couple the arms' end bins with the
      Kirchhoff weights, and gap bins are excluded.

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
        Interior-bin graph with nodes ``0..n_interior-1``, each with its
        ``'volume'``, and edges with a ``'conductance'`` and the ``'distance'``
        between the two bin centers (the shortest path through the track, on a
        track graph). :func:`build_laplacian` builds the operator from it.
    node_order : np.ndarray, shape (n_interior,)
        Interior flat-bin indices in graph-node order (maps local nodes back to the
        full grid).
    bin_sizes : np.ndarray, shape (n_interior,)
        Per-interior-bin volume (1D width, 2D area, ...).

    Raises
    ------
    ValidationError
        If the environment is unfitted, or is missing the track graph required by its
        branch (``track_graphDD`` for an N-D grid,
        ``track_graph_with_bin_centers_edges_`` for a linearized track graph).

    Notes
    -----
    The result is cached on the environment (see ``Environment.get_cached``;
    cleared by ``fit_place_grid``) so the graph is built once and shared by
    :func:`cached_eigenbasis` and the likelihood fit. Callers must treat the
    returned graph and arrays as read-only.
    """
    cached = environment.get_cached(_GRAPH_KEY)
    if cached is not None:
        return cast(tuple[nx.Graph, np.ndarray, np.ndarray], cached)

    if not environment._is_fitted:
        msg = "environment must be fitted before building the diffusion graph"
        raise ValidationError(
            msg,
            expected="a fitted Environment (place_bin_centers_, is_track_interior_ set)",
            got="an unfitted environment",
            hint="Call environment.fit_place_grid(position) first.",
        )

    if environment.track_graph is None:
        if environment.track_graphDD is None:
            msg = "fitted N-D environment is missing its track graph"
            raise ValidationError(
                msg,
                expected="track_graphDD set by fit_place_grid",
                got="track_graphDD is None",
            )
        result = _nd_grid_graph(environment)
    else:
        if environment.track_graph_with_bin_centers_edges_ is None:
            msg = "fitted linearized environment is missing its bin-center graph"
            raise ValidationError(
                msg,
                expected="track_graph_with_bin_centers_edges_ set by fit_place_grid",
                got="track_graph_with_bin_centers_edges_ is None",
            )
        result = _linearized_graph(environment)

    # The cached graph/arrays are shared across all neurons and every EM refit.
    # Freeze the graph topology and mark the arrays read-only so a stray mutation
    # cannot corrupt later callers. `nx.freeze` does NOT block edge-attribute
    # writes, so snapshot the Laplacian here — before the graph is handed out — and
    # derive the eigenbasis from that snapshot (see `cached_eigenbasis`); a later
    # `graph.edges[e]["conductance"] = ...` then cannot change the cached basis.
    graph, node_order, bin_sizes = result
    nx.freeze(graph)
    node_order.setflags(write=False)
    bin_sizes.setflags(write=False)
    environment.set_cached(_LAPLACIAN_KEY, build_laplacian(graph))
    return cast(
        tuple[nx.Graph, np.ndarray, np.ndarray],
        environment.set_cached(_GRAPH_KEY, result),
    )


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
    ParameterWarning
        If ``sigma`` is below the characteristic (median) bin spacing, indicating
        the smoother will barely spread beyond a single bin.
    """
    if not sigma > 0:
        msg = "smoothing bandwidth (position_std) must be positive"
        raise ValidationError(
            msg,
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
                ParameterWarning,
                stacklevel=user_stacklevel(),
            )


def _freeze_and_cache(
    cache: dict[int | None, tuple[np.ndarray, np.ndarray]],
    rank: int | None,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze a basis read-only, store it under ``rank``, and return it.

    The basis is shared across neurons and EM refits; centralizing the freeze here
    keeps the read-only cache contract in one place.
    """
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
    the fields), so it is computed once, cached on the environment as a dict keyed
    by ``rank``, and reused across all neurons and every EM refit. A cached
    full-rank (``rank=None``) entry serves any smaller-rank request by slicing its
    leading columns (valid because the modes are returned ascending). The cache is
    invalidated by ``fit_place_grid``.

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
        Laplacian eigenvalues at the requested rank (read-only, cached).
    eigvecs : np.ndarray, shape (n_interior, m)
        Corresponding eigenvectors as columns (read-only, cached). ``node_order`` /
        ``bin_sizes`` come from :func:`environment_graph` (also cached), not from here.
    """
    cache = environment.get_cached(_EIGENBASIS_KEY)
    if cache is None:
        cache = environment.set_cached(_EIGENBASIS_KEY, {})

    if rank in cache:
        return cast(tuple[np.ndarray, np.ndarray], cache[rank])

    # Build (and cache) the Laplacian snapshot; environment_graph runs first so a
    # later mutation of the returned graph cannot reach this L.
    environment_graph(environment)
    laplacian = environment.get_cached(_LAPLACIAN_KEY)

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
    :func:`cached_eigenbasis` (on the environment), and the resolved rank is
    memoized per ``(sigma, tol, dense_fraction)`` on the environment too, so EM
    refits at the same bandwidth reuse the basis **without re-running the adaptive
    probe** (zero eigensolves). When the bandwidth needs many modes the resolver
    falls back to the full dense basis. All diffusion caches are invalidated
    together by ``Environment.fit_place_grid``.

    Parameters
    ----------
    environment : Environment
        A fitted environment.
    sigma : float
        Smoothing standard deviation (``position_std``) in coordinate units.
    tol, dense_fraction : float, optional
        Forwarded to :func:`heat_kernel_rank`, by default 1e-6 and 0.2 respectively.

    Returns
    -------
    eigvals : np.ndarray, shape (m,)
        Heat-kernel eigenvalues at the resolved rank (read-only, cached).
    eigvecs : np.ndarray, shape (n_interior, m)
        Corresponding eigenvectors as columns (read-only, cached).
    """
    rank_cache = environment.get_cached(_HEAT_KERNEL_RANK_KEY)
    if rank_cache is None:
        rank_cache = environment.set_cached(_HEAT_KERNEL_RANK_KEY, {})

    key = (sigma, tol, dense_fraction)
    if key in rank_cache:
        # Rank already resolved for this bandwidth; the matching basis was cached below
        # on the first fit, so this reuses it with no adaptive probe and no eigensolve.
        return cached_eigenbasis(environment, rank_cache[key])

    environment_graph(environment)  # ensures the cached Laplacian snapshot
    rank, eigvals, eigvecs = _adaptive_heat_kernel_basis(
        environment.get_cached(_LAPLACIAN_KEY),
        sigma,
        tol,
        dense_fraction,
    )
    rank_cache[key] = rank
    if rank is None:
        return cached_eigenbasis(
            environment, None
        )  # light bandwidth -> dense full basis

    cache = environment.get_cached(_EIGENBASIS_KEY)
    if cache is None:
        cache = environment.set_cached(_EIGENBASIS_KEY, {})
    if rank in cache:
        return cast(tuple[np.ndarray, np.ndarray], cache[rank])
    # Independent copies of the probe's final basis (sliced to the resolved rank) so the
    # frozen cache entry does not alias the probe's arrays.
    return _freeze_and_cache(
        cache, rank, np.ascontiguousarray(eigvals), np.ascontiguousarray(eigvecs)
    )


def get_device_basis(
    environment: "Environment", resolved_rank: int
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Device (jnp) eigenbasis for the clusterless_diffusion kernel columns.

    The eigenvectors are scaled by ``sqrt(m / max(m))``, ``m`` the bin volumes,
    so ``P exp(-tΛ) Pᵀ e_s``, ``P`` the scaled basis, is the mass kernel's column
    for a unit mass at bin ``s`` times ``m_s / max(m)``: once normalized to unit
    mass, as :func:`heat_kernel_apply` point-source columns are by their callers,
    it is the mass kernel's column on any volumes. On equal volumes the scale is
    exactly 1. The scaled basis is not orthonormal on unequal volumes, so it is
    for point-source columns only; :func:`diffuse` takes the volumes for other
    fields.

    Transient environment cache entry keyed by ``(resolved_rank, device, dtype)``.
    On a miss, the host basis is retrieved via :func:`cached_eigenbasis` and
    converted to device arrays once; a cache hit
    returns the exact same device array objects, so a repeated predict-time call at
    the same rank reuses the device-resident basis rather than re-transferring it.
    Stored with ``transient=True`` so it is not pickled, and cleared on grid refit
    by ``Environment.fit_place_grid``.

    Parameters
    ----------
    environment : Environment
        A fitted environment.
    resolved_rank : int
        Number of eigenmodes to retrieve (see :func:`cached_eigenbasis`).

    Returns
    -------
    eigvals : jax.Array, shape (resolved_rank,)
        Laplacian eigenvalues, on ``device``.
    eigvecs : jax.Array, shape (n_interior, resolved_rank)
        Corresponding eigenvectors as columns, scaled by ``sqrt(m / max(m))``,
        on ``device``.
    component_labels : jax.Array, shape (n_interior,)
        Connected-component id per interior bin, ``0..n_components-1``, on
        ``device``.
    n_components : int
        Number of connected components (a static Python int, computed once per
        cache entry) — pass it straight into :func:`heat_kernel_apply` to avoid a
        per-call device->host sync of the labels.
    """
    # The device the arrays will actually live on. jnp.asarray follows the active
    # default-device context, so select it explicitly and device_put onto it — the
    # cache key must name the SAME device the cached arrays are placed on.
    device = (
        jax.local_devices()[0]
        if jax.config.jax_default_device is None  # type: ignore[attr-defined]  # config option registered dynamically by JAX
        else jax.config.jax_default_device  # type: ignore[attr-defined]  # config option registered dynamically by JAX
    )
    dtype = jnp.float32
    key = (int(resolved_rank), device, dtype)
    cache = environment.get_cached(_DEVICE_BASIS_KEY)
    if cache is None:
        # Keyed by a JAX ``Device``, which cannot be restored after unpickling.
        cache = environment.set_cached(_DEVICE_BASIS_KEY, {}, transient=True)
    if key not in cache:
        eigvals, eigvecs = cached_eigenbasis(environment, resolved_rank)
        # Component labels from the existing helper (indexed by interior-bin/node
        # order) — do NOT rebuild the Laplacian from the exposed frozen graph.
        graph, _, bin_sizes = environment_graph(environment)
        labels_host = connected_component_labels(graph)  # np.ndarray, 0..K-1
        n_components = int(labels_host.max()) + 1  # static, computed once
        root = np.sqrt(bin_sizes / np.max(bin_sizes))[:, np.newaxis]
        cache[key] = (
            jax.device_put(np.asarray(eigvals, dtype=np.float32), device),
            jax.device_put(np.asarray(root * eigvecs, dtype=np.float32), device),
            jax.device_put(np.asarray(labels_host, dtype=np.int32), device),
            n_components,
        )
    return cast(tuple[jax.Array, jax.Array, jax.Array, int], cache[key])


def _nd_grid_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Adapter branch for N-D / 1D grid environments (``track_graph is None``).

    Face-adjacent bins at center distance ``h`` along an axis share a face whose
    area ``A`` is the product of their widths along the other axes (1 in 1D),
    so their conductance is ``A / h``; a bin's volume is the product of its
    widths.
    """
    node_order = np.where(environment.is_track_interior_.ravel())[0]
    track_graph = cast(nx.Graph, environment.track_graphDD)
    subgraph = track_graph.subgraph(node_order)
    positions = nx.get_node_attributes(track_graph, "pos")
    relabel = {old: new for new, old in enumerate(node_order)}

    # bin_sizes = product of per-dimension bin widths, over the interior bins.
    widths = [np.diff(edge) for edge in environment.edges_]
    bin_volume = np.ones(environment.centers_shape_)
    for axis_width in np.meshgrid(*widths, indexing="ij"):
        bin_volume = bin_volume * axis_width
    bin_sizes = bin_volume.ravel()[node_order]

    graph = nx.Graph()
    graph.add_nodes_from(
        (local, {"volume": volume}) for local, volume in enumerate(bin_sizes)
    )
    for u, v, data in subgraph.edges(data=True):
        offset = np.asarray(positions[u]) - np.asarray(positions[v])
        differs = np.flatnonzero(np.abs(offset) > 1e-9)
        # Keep only face-adjacent pairs (differ in exactly one dimension).
        if differs.size == 1:
            (axis,) = differs
            index = np.unravel_index(u, environment.centers_shape_)
            face_area = bin_volume[index] / widths[axis][index[axis]]
            graph.add_edge(
                relabel[u],
                relabel[v],
                distance=data["distance"],
                conductance=face_area / data["distance"],
            )

    return graph, node_order, bin_sizes


def _linearized_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Adapter branch for linearized ``track_graph`` environments.

    A bin's volume is its width; the conductances between bin centers come
    from :func:`_center_conductances` on the track's bin-center graph.
    """
    is_interior = environment.is_track_interior_.ravel()
    node_order = np.where(is_interior)[0]
    bin_sizes = np.diff(environment.place_bin_edges_.ravel())[is_interior]

    # Substrate node id for each interior place bin (gap bins carry -1 and are
    # excluded); order matches place_bin_centers_ / is_track_interior_.
    node_ids = cast(Any, environment.place_bin_centers_nodes_df_).node_id.to_numpy()
    interior_node_ids = node_ids[is_interior].astype(int)
    local_of = {int(nid): local for local, nid in enumerate(interior_node_ids)}

    graph = nx.Graph()
    graph.add_nodes_from(
        (local, {"volume": volume}) for local, volume in enumerate(bin_sizes)
    )
    substrate = cast(nx.Graph, environment.track_graph_with_bin_centers_edges_)
    for (u, v), (conductance, distance) in _center_conductances(
        substrate, set(local_of)
    ).items():
        graph.add_edge(
            local_of[u], local_of[v], distance=distance, conductance=conductance
        )
    return graph, node_order, bin_sizes


def _center_conductances(
    substrate: nx.Graph, centers: set[int]
) -> dict[tuple[int, int], tuple[float, float]]:
    """Conductances between bin centers of a track's bin-center graph.

    ``substrate`` joins bin centers, bin edges and track nodes by segments with
    a ``'distance'`` (length); each segment conducts ``1 / length``. The nodes
    a zero-length segment joins are merged. Every node that is not a bin center
    is then eliminated (Kron reduction): each connected group of such nodes,
    with the centers next to it, is a resistor network whose Schur complement
    onto those centers gives their pairwise conductances. Two centers in series
    through one bin edge, at lengths ``a`` and ``b``, get ``1 / (a + b)``; the
    end bins of the arms at a vertex get ``g_i g_j / sum_k g_k``.

    Parameters
    ----------
    substrate : nx.Graph
        The bin-center graph (``track_graph_with_bin_centers_edges_``).
    centers : set of int
        Its nodes that are interior bin centers.

    Returns
    -------
    conductances : dict[tuple[int, int], tuple[float, float]]
        For each coupled pair of centers, the conductance between them and
        the length of the shortest path joining them through non-center nodes.

    Raises
    ------
    ValidationError
        If two bin centers are joined by a zero-length segment.
    """
    nodes = list(substrate.nodes)
    position = {node: k for k, node in enumerate(nodes)}
    lengths = {
        (position[u], position[v]): float(data["distance"])
        for u, v, data in substrate.edges(data=True)
    }

    # Merge the nodes zero-length segments join; a merged group is a center
    # when it holds one.
    zero = np.array([pair for pair, length in lengths.items() if length == 0.0])
    zero = zero.reshape(-1, 2)
    _, group = scipy.sparse.csgraph.connected_components(
        scipy.sparse.coo_matrix(
            (np.ones(len(zero)), (zero[:, 0], zero[:, 1])),
            shape=(len(nodes), len(nodes)),
        ),
        directed=False,
    )
    center_of: dict[int, int] = {}
    for node in centers:
        if group[position[node]] in center_of:
            msg = "two bin centers of the track graph are at zero distance"
            raise ValidationError(
                msg,
                got=f"centers {center_of[group[position[node]]]} and {node}",
                hint="Every place bin needs a positive width.",
            )
        center_of[group[position[node]]] = node

    # Conductance (summed over parallel segments) and shortest length between
    # merged groups, as adjacency maps.
    neighbors: dict[int, dict[int, list[float]]] = {}
    for (u, v), length in lengths.items():
        a, b = int(group[u]), int(group[v])
        if a == b:
            continue
        for x, y in ((a, b), (b, a)):
            link = neighbors.setdefault(x, {}).setdefault(y, [0.0, np.inf])
            link[0] += 1.0 / length
            link[1] = min(link[1], length)

    coupled: dict[tuple[int, int], tuple[float, float]] = {}

    def couple(i: int, j: int, conductance: float, distance: float) -> None:
        key = (min(i, j), max(i, j))
        previous_conductance, previous_distance = coupled.get(key, (0.0, np.inf))
        coupled[key] = (
            previous_conductance + conductance,
            min(previous_distance, distance),
        )

    # Center-center segments couple directly.
    for a, links in neighbors.items():
        if a in center_of:
            for b, (conductance, length) in links.items():
                if b in center_of and a < b:
                    couple(center_of[a], center_of[b], conductance, length)

    # Eliminate each connected group of non-center nodes onto the centers
    # around it.
    visited: set[int] = set()
    for seed in neighbors:
        if seed in center_of or seed in visited:
            continue
        cluster = [seed]
        visited.add(seed)
        for node in cluster:  # breadth-first; grows while iterating
            for adjacent in neighbors[node]:
                if adjacent not in center_of and adjacent not in visited:
                    visited.add(adjacent)
                    cluster.append(adjacent)
        boundary = sorted(
            {b for node in cluster for b in neighbors[node] if b in center_of}
        )
        if len(boundary) < 2:
            continue  # a dead end (track end): no coupling
        if len(cluster) == 1:
            # One node joining the centers: c_ij = g_i g_j / sum_k g_k.
            links = neighbors[cluster[0]]
            total = sum(links[b][0] for b in boundary)
            for k, i in enumerate(boundary):
                for j in boundary[k + 1 :]:
                    couple(
                        center_of[i],
                        center_of[j],
                        links[i][0] * links[j][0] / total,
                        links[i][1] + links[j][1],
                    )
            continue
        inner = {node: k for k, node in enumerate(cluster)}
        outer = {b: k for k, b in enumerate(boundary)}
        g_inner = np.zeros((len(cluster), len(cluster)))
        g_outer = np.zeros((len(cluster), len(boundary)))
        length_inner = np.zeros((len(cluster), len(cluster)))
        length_outer = np.full((len(cluster), len(boundary)), np.inf)
        for node, k in inner.items():
            for adjacent, (conductance, length) in neighbors[node].items():
                if adjacent in inner:
                    g_inner[k, inner[adjacent]] = conductance
                    length_inner[k, inner[adjacent]] = length
                else:
                    g_outer[k, outer[adjacent]] = conductance
                    length_outer[k, outer[adjacent]] = length
        # Schur complement onto the boundary: C = G_ob^T L_oo^{-1} G_ob.
        l_inner = np.diag(g_inner.sum(axis=1) + g_outer.sum(axis=1)) - g_inner
        schur = g_outer.T @ np.linalg.solve(l_inner, g_outer)
        through = scipy.sparse.csgraph.shortest_path(length_inner, directed=False)
        # Shortest boundary-to-node path, then node to boundary.
        to_inner = np.min(length_outer.T[:, :, None] + through[None], axis=1)
        distance = np.min(to_inner[:, None, :] + length_outer.T[None], axis=2)
        for k, i in enumerate(boundary):
            for m in range(k + 1, len(boundary)):
                couple(
                    center_of[i],
                    center_of[boundary[m]],
                    float(schur[k, m]),
                    float(distance[k, m]),
                )
    return coupled
