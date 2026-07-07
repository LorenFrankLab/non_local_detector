"""Tests for the graph-diffusion spectral engine (``likelihoods/diffusion.py``).

The engine builds a finite-difference graph Laplacian on an environment's interior
bins, eigendecomposes it (cached on the ``Environment``), and applies the heat
kernel ``exp(-t L)`` as a manifold-aware smoother. These tests pin the numerical
contracts: Laplacian structure, mode reconstruction, grid-independent bandwidth,
mass conservation, density normalization, and the two environment adapters.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    cached_eigenbasis,
    check_smoothing_bandwidth,
    connected_component_labels,
    diffuse,
    diffusion_eigenbasis,
    environment_graph,
    to_density,
)


def path_graph(n_nodes: int, distance: float = 1.0) -> nx.Graph:
    """1D chain 0-1-...-(n-1) with uniform ``distance`` edge attribute."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for node in range(n_nodes - 1):
        graph.add_edge(node, node + 1, distance=distance)
    return graph


def grid_graph_2d(n_x: int, n_y: int, spacing: float = 1.0) -> nx.Graph:
    """Face-adjacent 2D grid graph; nodes 0..n_x*n_y-1 in row-major (ij) order."""
    graph = nx.Graph()
    for i in range(n_x):
        for j in range(n_y):
            graph.add_node(i * n_y + j, pos=(i * spacing, j * spacing))
    for i in range(n_x):
        for j in range(n_y):
            node = i * n_y + j
            if i + 1 < n_x:
                graph.add_edge(node, (i + 1) * n_y + j, distance=spacing)
            if j + 1 < n_y:
                graph.add_edge(node, i * n_y + (j + 1), distance=spacing)
    return graph


# ==============================================================================
# build_laplacian
# ==============================================================================


def test_laplacian_symmetric_zero_rowsum():
    """L is symmetric, has zero row sums (1ᵀL = 0), and is PSD."""
    graph = path_graph(6, distance=1.0)
    L = build_laplacian(graph)

    assert scipy.sparse.issparse(L)
    dense = L.toarray()
    # Symmetric
    np.testing.assert_allclose(dense, dense.T, atol=1e-12)
    # Zero row sums -> mass-conserving heat kernel
    np.testing.assert_allclose(dense @ np.ones(6), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.ones(6) @ dense, 0.0, atol=1e-12)
    # Positive semi-definite (single connected component -> one zero eigenvalue)
    eigvals = np.linalg.eigvalsh(dense)
    assert eigvals.min() > -1e-9
    assert np.count_nonzero(np.abs(eigvals) < 1e-9) == 1


def test_laplacian_finite_difference_weight():
    """Edge weight is the finite-difference 1/d²; L = D - W exactly."""
    # Distance 2 -> weight 1/4 on every edge of a 4-node path.
    graph = path_graph(4, distance=2.0)
    dense = build_laplacian(graph).toarray()

    w = 1.0 / 2.0**2  # 0.25
    expected = np.array(
        [
            [w, -w, 0.0, 0.0],
            [-w, 2 * w, -w, 0.0],
            [0.0, -w, 2 * w, -w],
            [0.0, 0.0, -w, w],
        ]
    )
    np.testing.assert_allclose(dense, expected, atol=1e-12)


def test_laplacian_disconnected_components_two_zero_modes():
    """Two disconnected chains -> two null modes (one per component)."""
    graph = nx.disjoint_union(path_graph(3), path_graph(4))
    dense = build_laplacian(graph).toarray()
    eigvals = np.linalg.eigvalsh(dense)
    # Multiplicity of the zero eigenvalue == number of connected components.
    assert np.count_nonzero(np.abs(eigvals) < 1e-9) == 2


@pytest.mark.parametrize("bad_distance", [0.0, -1.0, float("nan")])
def test_laplacian_rejects_nonpositive_distance(bad_distance):
    """A zero/negative/NaN edge distance fails loudly (not ZeroDivisionError / silent)."""
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, distance=1.0)
    graph.add_edge(1, 2, distance=bad_distance)
    with pytest.raises(ValidationError):
        build_laplacian(graph)


# ==============================================================================
# diffusion_eigenbasis
# ==============================================================================


def test_eigenbasis_dense_shapes_and_ascending():
    """Dense eig returns all n modes, ascending, with orthonormal eigenvectors."""
    L = build_laplacian(grid_graph_2d(4, 5))
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)

    n = L.shape[0]
    assert eigvals.shape == (n,)
    assert eigvecs.shape == (n, n)
    # Ascending, non-negative (clipped), smallest is the zero mode.
    assert np.all(np.diff(eigvals) >= -1e-12)
    assert eigvals.min() >= 0.0
    assert eigvals[0] < 1e-9
    # Orthonormal columns.
    np.testing.assert_allclose(eigvecs.T @ eigvecs, np.eye(n), atol=1e-10)


def test_mode_reconstruction_matches_expm():
    """Full-rank Σⱼ exp(-tλⱼ) vⱼvⱼᵀ equals scipy.linalg.expm(-t L)."""
    L = build_laplacian(grid_graph_2d(5, 5))
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)

    sigma = 2.0
    t = sigma**2 / 2.0
    reconstructed = (eigvecs * np.exp(-t * eigvals)) @ eigvecs.T
    exact = scipy.linalg.expm(-t * L.toarray())
    assert np.abs(reconstructed - exact).max() < 1e-8


def test_truncated_eigsh_matches_dense_slice():
    """Truncated eigsh returns the smallest-rank eigenpairs incl. the zero mode.

    Eigenvectors are only defined up to sign/rotation, so we compare the
    eigenvalues and the rank-``r`` reconstructed operator (basis-invariant on the
    path graph's simple spectrum), not the eigenvectors directly.
    """
    L = build_laplacian(path_graph(30))  # simple (non-degenerate) spectrum
    rank = 6

    dense_vals, dense_vecs = diffusion_eigenbasis(L, rank=None)
    trunc_vals, trunc_vecs = diffusion_eigenbasis(L, rank=rank)

    assert trunc_vals.shape == (rank,)
    assert trunc_vecs.shape == (L.shape[0], rank)
    # Zero mode retained.
    assert trunc_vals[0] < 1e-9
    # Eigenvalues match the dense smallest-rank slice (no singular-factor failure).
    np.testing.assert_allclose(trunc_vals, dense_vals[:rank], atol=1e-8)

    t = 3.0
    trunc_op = (trunc_vecs * np.exp(-t * trunc_vals)) @ trunc_vecs.T
    dense_op = (dense_vecs[:, :rank] * np.exp(-t * dense_vals[:rank])) @ dense_vecs[
        :, :rank
    ].T
    assert np.abs(trunc_op - dense_op).max() < 1e-8


def test_truncated_keeps_all_zero_modes():
    """On a disconnected interior both null modes appear in a truncated basis."""
    graph = nx.disjoint_union(path_graph(8), path_graph(8))
    L = build_laplacian(graph)
    eigvals, _ = diffusion_eigenbasis(L, rank=4)
    assert np.count_nonzero(eigvals < 1e-9) == 2


@pytest.mark.parametrize("rank", [3, 5, 7])
def test_truncated_modes_are_component_local_no_leak(rank):
    """Truncated modes stay component-local: a point source in one component of a
    disconnected graph cannot leak mass into another.

    Two identical components make every eigenvalue doubly degenerate, so a single
    global eigsh returns cross-component rotated eigenvectors; a truncation cutting
    through such an eigenspace previously leaked mass after diffuse's clip/renorm.
    """
    graph = nx.disjoint_union(path_graph(20), path_graph(20))
    L = build_laplacian(graph)
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=rank)

    source = np.zeros((40, 1))
    source[5] = 1.0  # first component only (nodes 0..19)
    smoothed = diffuse(eigvals, eigvecs, sigma=3.0, fields=source).ravel()

    assert smoothed[20:].sum() < 1e-9  # no leak into the untouched component
    np.testing.assert_allclose(smoothed.sum(), 1.0, atol=1e-9)  # mass conserved


@pytest.mark.parametrize("rank", [6, 12, 30])
def test_diffuse_conserves_each_component_mass_under_truncation(rank):
    """When BOTH components carry mass, a truncated basis clips truncation lobes in the
    lobed component; per-component renormalization must restore each component's own
    mass rather than rescaling the total (which would bleed mass from the clean
    component into the lobed one).
    """
    graph = nx.disjoint_union(path_graph(60), path_graph(60))
    for u, v in graph.edges:
        graph.edges[u, v]["distance"] = 1.0
    laplacian = build_laplacian(graph)
    labels = connected_component_labels(graph)

    field = np.zeros((120, 1))
    field[30, 0] = 1.0  # a point source (produces truncation lobes) -> mass 1
    field[60:, 0] = 3.0 / 60  # a clean uniform field in the other component -> mass 3

    eigvals, eigvecs = diffusion_eigenbasis(laplacian, rank=rank)
    smoothed = diffuse(
        eigvals, eigvecs, sigma=3.0, fields=field, component_labels=labels
    )[:, 0]

    np.testing.assert_allclose(smoothed[:60].sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(smoothed[60:].sum(), 3.0, atol=1e-9)


def test_connected_component_labels_matches_graph_components():
    """The label helper assigns one contiguous id per connected component."""
    graph = nx.disjoint_union(path_graph(4), path_graph(6))
    labels = connected_component_labels(graph)
    assert labels.shape == (10,)
    assert set(labels[:4]) == {labels[0]} and set(labels[4:]) == {labels[4]}
    assert labels[0] != labels[4]  # different components -> different labels


def test_truncated_rank_below_n_components_raises():
    """rank < number of connected components is rejected (would drop a null mode)."""
    graph = nx.disjoint_union(path_graph(5), path_graph(5))  # 2 components
    L = build_laplacian(graph)
    with pytest.raises(ValidationError):
        diffusion_eigenbasis(L, rank=1)


def test_truncated_eigsh_falls_back_when_shift_invert_fails(monkeypatch):
    """When shift-invert eigsh raises, the which='SM' fallback returns the basis.

    This is the branch that motivates the sigma=-1e-8 (not sigma=0) choice: some
    builds still raise "Factor is exactly singular" from the LU factorization.
    """
    import scipy.sparse.linalg as sparse_linalg

    real_eigsh = sparse_linalg.eigsh

    def flaky_eigsh(*args, **kwargs):
        if kwargs.get("sigma") is not None:  # the shift-invert attempt
            raise RuntimeError("Factor is exactly singular")
        return real_eigsh(*args, **kwargs)

    monkeypatch.setattr(sparse_linalg, "eigsh", flaky_eigsh)

    L = build_laplacian(path_graph(30))
    trunc_vals, trunc_vecs = diffusion_eigenbasis(L, rank=6)
    # rank=None uses dense eigh (not eigsh), so the reference is unaffected.
    dense_vals, _ = diffusion_eigenbasis(L, rank=None)

    assert trunc_vals.shape == (6,)
    assert trunc_vals[0] < 1e-9  # zero mode retained via the fallback
    np.testing.assert_allclose(trunc_vals, dense_vals[:6], atol=1e-8)


# ==============================================================================
# diffuse + to_density
# ==============================================================================


def _weighted_std_per_axis(field: np.ndarray, shape: tuple, spacing: float):
    """Recovered per-axis standard deviation of a diffused point source.

    ``field`` (flat, sums to 1) is reshaped to ``shape``; for each axis the
    marginal distribution is formed and its weighted std returned.
    """
    grid = field.reshape(shape)
    stds = []
    for axis in range(len(shape)):
        marginal = grid.sum(axis=tuple(a for a in range(len(shape)) if a != axis))
        marginal = marginal / marginal.sum()
        coords = np.arange(shape[axis]) * spacing
        mean = np.sum(marginal * coords)
        stds.append(np.sqrt(np.sum(marginal * (coords - mean) ** 2)))
    return np.array(stds)


def test_diffuse_conserves_mass():
    """exp(-tL) conserves field sums (Σ diffuse == Σ field) to ~1e-10."""
    rng = np.random.default_rng(0)
    L = build_laplacian(grid_graph_2d(6, 7))
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)
    fields = rng.uniform(0, 1, size=(L.shape[0], 3))  # occupancy + 2 neurons

    smoothed = diffuse(eigvals, eigvecs, sigma=1.5, fields=fields)

    assert smoothed.shape == fields.shape
    np.testing.assert_allclose(smoothed.sum(0), fields.sum(0), atol=1e-10)
    assert np.all(smoothed >= 0.0)  # clipped


def test_diffuse_truncated_point_source_conserves_mass():
    """A low-rank point-source kernel has negative lobes; clipping must not add mass.

    The constant/null mode is always retained, so total mass is a conserved
    quantity; ``diffuse`` renormalizes after clipping so a truncated basis cannot
    inflate the field sum.
    """
    L = build_laplacian(path_graph(60))
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=8)  # aggressive truncation
    source = np.zeros((60, 1))
    source[30] = 1.0

    smoothed = diffuse(eigvals, eigvecs, sigma=3.0, fields=source)

    assert np.all(smoothed >= 0.0)
    np.testing.assert_allclose(smoothed.sum(), 1.0, atol=1e-10)  # mass preserved


def test_diffuse_bandwidth_independent_of_bin_size_2d():
    """Recovered smoothing std == sigma within 5% across 2D bin sizes {0.5,1,2,4}.

    This is the finite-difference (1/d²) calibration guarantee: the physical
    bandwidth equals ``sigma`` regardless of grid spacing. Uses the operator
    directly (``expm_multiply``) so the fine grids stay tractable; the eig-based
    ``diffuse`` reproduces the same operator (see mode-reconstruction and
    analytic-Gaussian tests).
    """
    import scipy.sparse.linalg

    sigma = 4.0
    t = sigma**2 / 2.0
    half_extent = 3.5 * sigma  # >3σ containment so truncation barely affects std
    for spacing in (0.5, 1.0, 2.0, 4.0):
        n = int(round(2 * half_extent / spacing))
        L = build_laplacian(grid_graph_2d(n, n, spacing))
        center = (n // 2) * n + (n // 2)
        source = np.zeros(L.shape[0])
        source[center] = 1.0
        smoothed = scipy.sparse.linalg.expm_multiply(-t * L, source)

        stds = _weighted_std_per_axis(smoothed, (n, n), spacing)
        np.testing.assert_allclose(stds, sigma, rtol=0.05)


def test_moore_diagonal_edges_oversmooth():
    """8-connected (Moore) edges with 1/d² oversmooth by ≈√2 — the face-only guard."""
    import scipy.sparse.linalg

    sigma, spacing = 4.0, 1.0
    t = sigma**2 / 2.0
    n = int(round(2 * 3.5 * sigma / spacing))

    face = grid_graph_2d(n, n, spacing)
    moore = face.copy()
    diag = spacing * np.sqrt(2.0)
    for i in range(n - 1):
        for j in range(n - 1):
            moore.add_edge(i * n + j, (i + 1) * n + (j + 1), distance=diag)
            moore.add_edge((i + 1) * n + j, i * n + (j + 1), distance=diag)

    center = (n // 2) * n + (n // 2)
    source = np.zeros(n * n)
    source[center] = 1.0

    face_std = _weighted_std_per_axis(
        scipy.sparse.linalg.expm_multiply(-t * build_laplacian(face), source),
        (n, n),
        spacing,
    )
    moore_std = _weighted_std_per_axis(
        scipy.sparse.linalg.expm_multiply(-t * build_laplacian(moore), source),
        (n, n),
        spacing,
    )
    np.testing.assert_allclose(face_std, sigma, rtol=0.05)
    assert np.all(moore_std > 1.3 * sigma)  # ≈√2·sigma; justifies dropping diagonals


def test_diffuse_matches_analytic_gaussian_1d():
    """A single-point source diffused by `diffuse` matches the exact Gaussian, <2%."""
    sigma, h, n = 5.0, 1.0, 41
    center = n // 2
    L = build_laplacian(path_graph(n, distance=h))
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)

    source = np.zeros(n)
    source[center] = 1.0
    smoothed = diffuse(eigvals, eigvecs, sigma, source[:, None]).ravel()

    x = np.arange(n) * h
    analytic = (
        h
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((x - center * h) ** 2) / (2 * sigma**2))
    )
    within_2sigma = np.abs(x - center * h) <= 2 * sigma
    rel_err = (
        np.abs(smoothed[within_2sigma] - analytic[within_2sigma])
        / analytic[within_2sigma]
    )
    assert rel_err.max() < 0.02


def test_diffuse_matches_analytic_gaussian_2d():
    """2D single-point source ≈ exact isotropic Gaussian away from the boundary, <2%.

    ``sigma / h = 5`` keeps the 5-point stencil's lattice anisotropy (largest at
    the 45° corner of the 2σ disk) below the 2% tolerance.
    """
    sigma, h = 5.0, 1.0
    n = 41  # half-extent 4σ from the center
    center_ij = (n // 2, n // 2)
    L = build_laplacian(grid_graph_2d(n, n, h))
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)

    source = np.zeros(n * n)
    source[center_ij[0] * n + center_ij[1]] = 1.0
    smoothed = diffuse(eigvals, eigvecs, sigma, source[:, None]).ravel().reshape(n, n)

    coords = np.arange(n) * h
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    r2 = (xx - center_ij[0] * h) ** 2 + (yy - center_ij[1] * h) ** 2
    analytic = h**2 / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))

    within_2sigma = np.sqrt(r2) <= 2 * sigma
    rel_err = (
        np.abs(smoothed[within_2sigma] - analytic[within_2sigma])
        / analytic[within_2sigma]
    )
    assert rel_err.max() < 0.02


def test_to_density_integrates_to_one_uniform_and_nonuniform():
    """to_density normalizes each column to ∫=1; zero-mass columns -> zeros."""
    smoothed = np.array(
        [
            [2.0, 0.0, 1.0],
            [4.0, 0.0, 3.0],
            [6.0, 0.0, 1.0],
        ]
    )
    # Non-uniform bin volumes.
    bin_sizes = np.array([0.5, 2.0, 1.0])

    density = to_density(smoothed, bin_sizes)

    integral = bin_sizes @ density
    # Column 1 is all-zero -> stays zero (integral 0); others integrate to 1.
    np.testing.assert_allclose(integral, [1.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_array_equal(density[:, 1], 0.0)
    # Uniform bins reduce to divide-by-sum.
    uniform = to_density(smoothed[:, [0]], np.ones(3))
    np.testing.assert_allclose(uniform.ravel(), smoothed[:, 0] / smoothed[:, 0].sum())


def test_to_density_nonuniform_matches_oracle():
    """On a non-uniform 1D chain, diffuse+to_density matches an independent oracle."""
    import scipy.sparse.linalg

    rng = np.random.default_rng(1)
    n = 25
    distances = rng.uniform(0.5, 2.0, size=n - 1)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for k in range(n - 1):
        graph.add_edge(k, k + 1, distance=distances[k])
    L = build_laplacian(graph)
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)

    # bin_sizes for an interior node ~ mean of its adjacent edge lengths.
    bin_sizes = np.empty(n)
    bin_sizes[1:-1] = 0.5 * (distances[:-1] + distances[1:])
    bin_sizes[0] = distances[0]
    bin_sizes[-1] = distances[-1]

    field = rng.uniform(0.1, 1.0, size=(n, 1))
    density = to_density(diffuse(eigvals, eigvecs, 1.5, field), bin_sizes)

    t = 1.5**2 / 2.0
    oracle = scipy.sparse.linalg.expm_multiply(-t * L, field.ravel())
    oracle = np.clip(oracle, 0.0, None)
    oracle_density = oracle / (bin_sizes @ oracle)

    np.testing.assert_allclose(density.ravel(), oracle_density, atol=1e-8)
    np.testing.assert_allclose(bin_sizes @ density.ravel(), 1.0, atol=1e-12)


# ==============================================================================
# environment_graph adapter
# ==============================================================================


def make_2d_env(seed: int = 0) -> Environment:
    """2D open-field env with an inferred single-block interior (10x10 bins)."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(1.0, 49.0, size=(4000, 2))
    return Environment(
        environment_name="of",
        place_bin_size=5.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(position, infer_track_interior=True)


def make_2d_split_env() -> Environment:
    """2D env with two spatially separated clusters -> two interior components."""
    rng = np.random.default_rng(3)
    left = rng.uniform([2.0, 2.0], [18.0, 48.0], size=(3000, 2))
    right = rng.uniform([32.0, 2.0], [48.0, 48.0], size=(3000, 2))
    position = np.vstack([left, right])
    return Environment(
        environment_name="split",
        place_bin_size=2.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(position, infer_track_interior=True)


def make_linear_track_env(edge_spacing: float = 0.25) -> Environment:
    """Two-edge linearized track sharing a junction node (a continuous line)."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(5.0, 0.0))
    graph.add_node(2, pos=(10.5, 0.0))
    graph.add_edge(0, 1, distance=5.0)
    graph.add_edge(1, 2, distance=5.5)
    for eid, edge in enumerate(graph.edges):
        graph.edges[edge]["edge_id"] = eid
    return Environment(
        environment_name="line2",
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=edge_spacing,
    ).fit_place_grid()


def make_disconnected_track_env() -> Environment:
    """Two physically separate segments (no shared node) -> disconnected arms."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(5.0, 0.0))
    graph.add_node(2, pos=(0.0, 20.0))
    graph.add_node(3, pos=(5.0, 20.0))
    graph.add_edge(0, 1, distance=5.0)
    graph.add_edge(2, 3, distance=5.0)
    for eid, edge in enumerate(graph.edges):
        graph.edges[edge]["edge_id"] = eid
    return Environment(
        environment_name="two_arms",
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=2.0,
    ).fit_place_grid()


def diffuse_point_source(env: Environment, source_local: int, sigma: float):
    """Build L/eig from the adapter graph and diffuse a unit source at a local bin."""
    graph, node_order, bin_sizes = environment_graph(env)
    L = build_laplacian(graph)
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)
    source = np.zeros((graph.number_of_nodes(), 1))
    source[source_local] = 1.0
    smoothed = diffuse(eigvals, eigvecs, sigma, source).ravel()
    return graph, node_order, bin_sizes, smoothed


def test_environment_graph_nd_node_order_face_only_and_bin_sizes():
    """N-D adapter: correct node_order, face-adjacent-only edges, uniform bin_sizes."""
    env = make_2d_env()
    graph, node_order, bin_sizes = environment_graph(env)

    interior = np.where(env.is_track_interior_.ravel())[0]
    np.testing.assert_array_equal(node_order, interior)
    assert graph.number_of_nodes() == interior.size == 100
    # A 10x10 interior block has exactly 2*10*9 = 180 face edges, no diagonals.
    assert graph.number_of_edges() == 180

    pos = nx.get_node_attributes(env.track_graphDD, "pos")
    for u, v in graph.edges():
        diff = np.asarray(pos[node_order[u]]) - np.asarray(pos[node_order[v]])
        assert np.count_nonzero(np.abs(diff) > 1e-9) == 1  # face-adjacent only

    # Uniform 5x5 bins.
    assert bin_sizes.shape == (100,)
    np.testing.assert_allclose(bin_sizes, 25.0)


def test_environment_graph_nd_roundtrip():
    """Unit field at interior bin k diffuses to a bump whose argmax is k."""
    env = make_2d_env()
    # Central interior bin (flat index 6*12 + 6 in the 12x12 grid).
    interior = np.where(env.is_track_interior_.ravel())[0]
    k_local = int(np.where(interior == (6 * 12 + 6))[0][0])
    _, _, _, smoothed = diffuse_point_source(env, k_local, sigma=6.0)
    assert int(np.argmax(smoothed)) == k_local
    np.testing.assert_allclose(smoothed.sum(), 1.0, atol=1e-10)


def test_environment_graph_linearized_roundtrip_and_bin_sizes():
    """Linearized adapter: node_order, gap-excluded bin_sizes, and a bump round-trip."""
    env = make_linear_track_env()
    graph, node_order, bin_sizes = environment_graph(env)

    is_interior = env.is_track_interior_.ravel()
    np.testing.assert_array_equal(node_order, np.where(is_interior)[0])
    assert graph.number_of_nodes() == int(is_interior.sum()) == 11
    # Independent hand computation of the interior bin widths: edge 0 (length 5.0)
    # -> 5 bins of width 1.0; edge 1 (length 5.5) -> 6 bins of width 5.5/6; the
    # wide 0.25 gap bin between them is excluded.
    hand_computed = np.concatenate([np.full(5, 1.0), np.full(6, 5.5 / 6.0)])
    np.testing.assert_allclose(bin_sizes, hand_computed, atol=1e-9)
    assert 0.25 not in np.round(bin_sizes, 2)  # the gap bin is excluded

    for k in (2, 8):
        _, _, _, smoothed = diffuse_point_source(env, k, sigma=1.5)
        assert int(np.argmax(smoothed)) == k


def test_environment_graph_linearized_junction_connects_arms():
    """Arms sharing a junction node stay connected despite a linear-space gap bin."""
    env = make_linear_track_env(edge_spacing=0.25)
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 1


def make_y_track_env(edge_spacing: float = 2.0) -> Environment:
    """Three edges radiating from a shared junction node (a Y / T maze)."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))  # junction
    graph.add_node(1, pos=(6.0, 0.0))
    graph.add_node(2, pos=(-6.0, 0.0))
    graph.add_node(3, pos=(0.0, 6.0))
    graph.add_edge(0, 1, distance=6.0)
    graph.add_edge(0, 2, distance=6.0)
    graph.add_edge(0, 3, distance=6.0)
    for eid, edge in enumerate(graph.edges):
        graph.edges[edge]["edge_id"] = eid
    return Environment(
        environment_name="ytrack",
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (0, 2), (0, 3)],
        edge_spacing=edge_spacing,
    ).fit_place_grid()


def test_environment_graph_three_arm_junction_links_all_arms():
    """A 3-arm (Y) junction links every arm into one component and conserves mass."""
    env = make_y_track_env()
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 1
    # The contraction links the innermost bin of each arm through the shared
    # junction, producing a hub node of degree >= 3 (a plain chain maxes out at 2).
    assert max(dict(graph.degree()).values()) >= 3

    L = build_laplacian(graph)
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)
    source = np.zeros((graph.number_of_nodes(), 1))
    source[0] = 1.0
    smoothed = diffuse(eigvals, eigvecs, sigma=2.0, fields=source).ravel()
    np.testing.assert_allclose(smoothed.sum(), 1.0, atol=1e-10)


def test_environment_graph_disconnected_components_mass_conserved():
    """Two disconnected interior components each conserve their own mass."""
    env = make_2d_split_env()
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 2

    L = build_laplacian(graph)
    # A truncated basis must still carry both null modes.
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=6)
    assert np.count_nonzero(eigvals < 1e-9) == 2

    # Label nodes by component; mass placed in one component stays there.
    components = list(nx.connected_components(graph))
    comp0 = np.array(sorted(components[0]))
    eigvals_full, eigvecs_full = diffusion_eigenbasis(L, rank=None)
    source = np.zeros((graph.number_of_nodes(), 1))
    source[comp0[len(comp0) // 2]] = 1.0
    smoothed = diffuse(eigvals_full, eigvecs_full, sigma=4.0, fields=source).ravel()

    mask0 = np.zeros(graph.number_of_nodes(), dtype=bool)
    mask0[comp0] = True
    np.testing.assert_allclose(smoothed[mask0].sum(), 1.0, atol=1e-9)
    assert smoothed[~mask0].sum() < 1e-9  # no leak into the other component


def test_environment_graph_no_leak_across_disconnected_arms():
    """Linearized: mass on one physically separate arm does not reach the other."""
    env = make_disconnected_track_env()
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 2

    components = list(nx.connected_components(graph))
    arm0 = np.array(sorted(components[0]))
    L = build_laplacian(graph)
    eigvals, eigvecs = diffusion_eigenbasis(L, rank=None)
    source = np.zeros((graph.number_of_nodes(), 1))
    source[arm0[len(arm0) // 2]] = 1.0
    smoothed = diffuse(eigvals, eigvecs, sigma=3.0, fields=source).ravel()

    peak = smoothed.max()
    mask0 = np.zeros(graph.number_of_nodes(), dtype=bool)
    mask0[arm0] = True
    assert smoothed[~mask0].max() < 0.02 * peak


# ==============================================================================
# cached_eigenbasis + environment cache invalidation
# ==============================================================================


def test_environment_graph_is_cached():
    """Repeated environment_graph calls return the same cached object."""
    env = make_2d_env()
    first = environment_graph(env)
    second = environment_graph(env)
    assert first[0] is second[0]  # same graph object -> built once


def test_cached_eigenbasis_reuse_and_rank_keying():
    """cached_eigenbasis caches per rank; a full-rank entry serves smaller ranks."""
    env = make_2d_env()

    full1 = cached_eigenbasis(env, rank=None)
    full2 = cached_eigenbasis(env, rank=None)
    assert full1[0] is full2[0]  # identical object -> no recompute
    assert env._diffusion_eigenbasis_[None] is full1

    # A cached full-rank basis serves a smaller-rank request by slicing, and the
    # slice is stored under its own rank key.
    trunc_vals, trunc_vecs = cached_eigenbasis(env, rank=5)
    np.testing.assert_allclose(trunc_vals, full1[0][:5])
    np.testing.assert_allclose(trunc_vecs, full1[1][:, :5])
    assert 5 in env._diffusion_eigenbasis_


def test_cached_eigenbasis_rank_below_components_raises_even_when_full_cached():
    """Slicing a cached full basis must not bypass the rank >= n_components guard."""
    env = make_2d_split_env()  # two interior components -> two null modes required
    cached_eigenbasis(env, rank=None)  # cache the full basis first
    with pytest.raises(ValidationError):
        cached_eigenbasis(env, rank=1)  # would slice to a single null mode


def test_cached_eigenbasis_slice_is_leak_free_on_disconnected_env():
    """Slicing a cached full basis on a disconnected env must not leak across
    components (the full basis is component-local, so its leading columns are too)."""
    env = make_2d_split_env()
    cached_eigenbasis(env, rank=None)  # cache the full (component-local) basis
    eigvals, eigvecs = cached_eigenbasis(env, rank=8)  # served by slicing

    graph, _, _ = environment_graph(env)
    components = list(nx.connected_components(graph))
    comp0 = np.array(sorted(components[0]))
    source = np.zeros((graph.number_of_nodes(), 1))
    source[comp0[len(comp0) // 2]] = 1.0
    smoothed = diffuse(eigvals, eigvecs, sigma=3.0, fields=source).ravel()

    mask0 = np.zeros(graph.number_of_nodes(), dtype=bool)
    mask0[comp0] = True
    assert smoothed[~mask0].sum() < 1e-9  # no leak into the other component


def test_cached_objects_are_read_only():
    """Cached graph/arrays are frozen so a stray mutation can't corrupt the basis."""
    env = make_2d_env()
    graph, node_order, bin_sizes = environment_graph(env)
    assert nx.is_frozen(graph)
    with pytest.raises((ValueError, nx.NetworkXError)):
        graph.add_edge(0, 99)
    for arr in (node_order, bin_sizes):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 0

    eigvals, eigvecs = cached_eigenbasis(env, rank=None)
    assert not eigvals.flags.writeable
    assert not eigvecs.flags.writeable
    # The sliced smaller-rank basis inherits read-only from the full-rank arrays.
    trunc_vals, trunc_vecs = cached_eigenbasis(env, rank=4)
    assert not trunc_vals.flags.writeable
    assert not trunc_vecs.flags.writeable


def test_cached_eigenbasis_immune_to_graph_edge_attribute_mutation():
    """nx.freeze allows edge-attribute writes; the Laplacian snapshot must still
    protect the cached eigenbasis from a mutated edge distance."""
    reference, _ = cached_eigenbasis(make_2d_env(), rank=8)

    env = make_2d_env()  # identical grid (same seed)
    graph, _, _ = environment_graph(env)  # snapshots L before the mutation below
    an_edge = next(iter(graph.edges))
    graph.edges[an_edge]["distance"] = 999.0  # freeze does NOT block this

    got, _ = cached_eigenbasis(env, rank=8)  # built from the pristine snapshot
    np.testing.assert_allclose(got, reference, atol=1e-8)


def test_eig_cache_invalidated_on_refit():
    """fit_place_grid clears the eig + graph caches so they rebuild."""
    env = make_2d_env()
    basis = cached_eigenbasis(env, rank=None)
    graph = environment_graph(env)
    assert hasattr(env, "_diffusion_eigenbasis_")
    assert hasattr(env, "_diffusion_graph_")

    rng = np.random.default_rng(0)
    env.fit_place_grid(
        rng.uniform(1.0, 49.0, size=(4000, 2)), infer_track_interior=True
    )
    assert not hasattr(env, "_diffusion_eigenbasis_")
    assert not hasattr(env, "_diffusion_graph_")

    rebuilt = cached_eigenbasis(env, rank=None)
    assert rebuilt[0] is not basis[0]  # recomputed after invalidation
    assert environment_graph(env)[0] is not graph[0]


# ==============================================================================
# Validation guard rails
# ==============================================================================


def test_environment_graph_unfitted_env_raises():
    """An unfitted environment (no interior mask) raises ValidationError."""
    env = Environment(environment_name="unfit", place_bin_size=5.0)
    with pytest.raises(ValidationError):
        environment_graph(env)


def test_check_bandwidth_rejects_nonpositive_sigma():
    """position_std <= 0 is rejected."""
    graph = environment_graph(make_2d_env())[0]
    for bad in (0.0, -1.0):
        with pytest.raises(ValidationError):
            check_smoothing_bandwidth(bad, graph)


def test_check_bandwidth_warns_when_undersmoothing():
    """A sigma below ~1 bin spacing warns; a comfortably larger sigma does not."""
    graph = environment_graph(make_2d_env())[0]  # 5.0-unit bin spacing
    with pytest.warns(UserWarning):
        check_smoothing_bandwidth(1.0, graph)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        check_smoothing_bandwidth(15.0, graph)
