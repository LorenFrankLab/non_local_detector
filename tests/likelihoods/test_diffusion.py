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
    _EIGENBASIS_KEY,
    _GRAPH_KEY,
    _HEAT_KERNEL_RANK_KEY,
    _LAPLACIAN_KEY,
    build_laplacian,
    cached_eigenbasis,
    cached_heat_kernel_eigenbasis,
    check_smoothing_bandwidth,
    connected_component_labels,
    diffuse,
    diffusion_eigenbasis,
    environment_graph,
    heat_kernel_rank,
    to_density,
)
from tests.likelihoods._sorted_spikes_helpers import (
    make_2d_env,
    make_linear_track_env,
    make_two_width_env,
    path_graph,
)

pytestmark = pytest.mark.unit


def grid_graph_2d(n_x: int, n_y: int, spacing: float = 1.0) -> nx.Graph:
    """Face-adjacent 2D grid graph; nodes 0..n_x*n_y-1 in row-major (ij) order.

    Square bins of side ``spacing``: volume ``spacing**2``, and a face of
    length ``spacing`` at center distance ``spacing``, so conductance 1.
    """
    graph = nx.Graph()
    for i in range(n_x):
        for j in range(n_y):
            graph.add_node(
                i * n_y + j, pos=(i * spacing, j * spacing), volume=spacing**2
            )
    for i in range(n_x):
        for j in range(n_y):
            node = i * n_y + j
            if i + 1 < n_x:
                graph.add_edge(
                    node, (i + 1) * n_y + j, distance=spacing, conductance=1.0
                )
            if j + 1 < n_y:
                graph.add_edge(
                    node, i * n_y + (j + 1), distance=spacing, conductance=1.0
                )
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


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_laplacian_rejects_a_conductance_that_is_not_positive_and_finite(bad):
    """A zero/negative/NaN/infinite edge conductance fails loudly."""
    graph = path_graph(3)
    graph.edges[1, 2]["conductance"] = bad
    with pytest.raises(ValidationError, match="graph edge has a conductance"):
        build_laplacian(graph)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_laplacian_rejects_a_volume_that_is_not_positive_and_finite(bad):
    """A zero/negative/NaN node volume fails loudly."""
    graph = path_graph(3)
    graph.nodes[1]["volume"] = bad
    with pytest.raises(ValidationError, match="graph node has a volume"):
        build_laplacian(graph)


@pytest.mark.parametrize("missing", ["volume", "conductance"])
def test_laplacian_names_a_missing_attribute(missing):
    """A graph without node volumes or edge conductances, such as one built for
    the earlier ``distance``-only contract, is refused naming the attribute."""
    graph = path_graph(3)
    items = graph.nodes.values() if missing == "volume" else graph.edges.values()
    for data in items:
        del data[missing]
    with pytest.raises(ValidationError, match=f"'{missing}'"):
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


def test_truncated_eigsh_is_reproducible():
    """Truncated eigsh returns a bit-identical basis run-to-run.

    A square grid has a doubly-degenerate spectrum, so ARPACK with a random start
    vector would return eigenvectors rotated within each degenerate eigenspace,
    making the truncated basis -- and any downstream MRF fit that reads it -- differ
    between runs. ``_block_eigenbasis`` pins a deterministic ``v0``; assert the
    eigenpairs are byte-for-byte identical across repeated calls.
    """
    L = build_laplacian(grid_graph_2d(12, 12))  # degenerate spectrum
    rank = 20

    vals1, vecs1 = diffusion_eigenbasis(L, rank=rank)
    vals2, vecs2 = diffusion_eigenbasis(L, rank=rank)

    assert np.array_equal(vals1, vals2)
    assert np.array_equal(vecs1, vecs2)


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
    assert set(labels[:4]) == {labels[0]}
    assert set(labels[4:]) == {labels[4]}
    assert labels[0] != labels[4]  # different components -> different labels


def test_truncated_rank_below_n_components_raises():
    """rank < number of connected components is rejected (would drop a null mode)."""
    graph = nx.disjoint_union(path_graph(5), path_graph(5))  # 2 components
    L = build_laplacian(graph)
    with pytest.raises(ValidationError, match="rank is too small"):
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
            msg = "Factor is exactly singular"
            raise RuntimeError(msg)
        return real_eigsh(*args, **kwargs)

    monkeypatch.setattr(sparse_linalg, "eigsh", flaky_eigsh)

    L = build_laplacian(path_graph(30))
    with pytest.warns(UserWarning, match="Shift-invert eigsh failed"):
        trunc_vals, _trunc_vecs = diffusion_eigenbasis(L, rank=6)
    # rank=None uses dense eigh (not eigsh), so the reference is unaffected.
    dense_vals, _ = diffusion_eigenbasis(L, rank=None)

    assert trunc_vals.shape == (6,)
    assert trunc_vals[0] < 1e-9  # zero mode retained via the fallback
    np.testing.assert_allclose(trunc_vals, dense_vals[:6], atol=1e-8)


def test_fallback_eigsh_is_reproducible(monkeypatch):
    """The which='SM' fallback pins the same v0, so its basis is reproducible too.

    The fallback is a separate eigsh call from the shift-invert path; on a
    degenerate spectrum it would rotate eigenvectors run-to-run without a fixed
    start vector (differs by ~0.4 otherwise). Force the fallback on a degenerate
    square grid and assert the eigenpairs are bit-identical across repeated calls.
    """
    import scipy.sparse.linalg as sparse_linalg

    real_eigsh = sparse_linalg.eigsh

    def flaky_eigsh(*args, **kwargs):
        if kwargs.get("sigma") is not None:  # force the shift-invert attempt to fail
            msg = "Factor is exactly singular"
            raise RuntimeError(msg)
        return real_eigsh(*args, **kwargs)

    monkeypatch.setattr(sparse_linalg, "eigsh", flaky_eigsh)

    L = build_laplacian(grid_graph_2d(12, 12))  # degenerate spectrum
    with pytest.warns(UserWarning, match="Shift-invert eigsh failed"):
        vals1, vecs1 = diffusion_eigenbasis(L, rank=20)
    with pytest.warns(UserWarning, match="Shift-invert eigsh failed"):
        vals2, vecs2 = diffusion_eigenbasis(L, rank=20)

    assert np.array_equal(vals1, vals2)
    assert np.array_equal(vecs1, vecs2)


# ==============================================================================
# heat_kernel_rank (bandwidth-aware auto-truncation)
# ==============================================================================


def test_heat_kernel_rank_tracks_bandwidth():
    """More smoothing (larger sigma) needs fewer modes: rank is non-increasing."""
    L = build_laplacian(grid_graph_2d(30, 30))  # 900 bins
    n = L.shape[0]

    def effective(sigma):
        rank = heat_kernel_rank(L, sigma)
        return n if rank is None else rank

    ranks = [effective(sigma) for sigma in (2.0, 4.0, 8.0, 16.0)]
    assert ranks == sorted(ranks, reverse=True)
    assert ranks[-1] < n  # heavy smoothing truncates well below full rank


def test_heat_kernel_rank_is_near_lossless():
    """Diffusing with the auto-truncated basis matches the full basis to ~tol.

    Dropped modes k have exp(-t*lambda_k) < tol, so the truncation error is bounded by
    tol * ||field|| -- diffusion is near-lossless at the auto-selected rank.
    """
    rng = np.random.default_rng(0)
    L = build_laplacian(grid_graph_2d(30, 30))
    sigma, tol = 6.0, 1e-6
    full_vals, full_vecs = diffusion_eigenbasis(L, rank=None)
    rank = heat_kernel_rank(L, sigma, tol=tol)
    assert rank is not None
    assert rank < L.shape[0]
    trunc_vals, trunc_vecs = diffusion_eigenbasis(L, rank=rank)

    fields = rng.gamma(2.0, 1.0, size=(L.shape[0], 8))
    full = diffuse(full_vals, full_vecs, sigma, fields)
    trunc = diffuse(trunc_vals, trunc_vecs, sigma, fields)
    rel = np.max(np.linalg.norm(full - trunc, axis=0) / np.linalg.norm(full, axis=0))
    assert rel < 10 * tol


@pytest.mark.parametrize("sigma", [10.0, 20.0])
def test_heat_kernel_rank_truncates_one_dimensional_track(sigma, monkeypatch):
    """A 1-D track whose needed rank is below the dense cap gets a truncated rank.

    The eigenvalue-counting function of a 1-D track grows like sqrt(lambda), not
    linearly as on an open field, so a linear extrapolation from the first probe
    overshoots the cap and would fall back to the dense basis. The resolver must
    return exactly the number of modes with weight >= tol, in two eigensolves (the
    first probe plus one jump), and the truncated basis must diffuse like the full one.
    """
    import non_local_detector.likelihoods.diffusion as diffusion_mod

    L = build_laplacian(path_graph(1000))
    tol = 1e-6
    full_vals, full_vecs = diffusion_eigenbasis(L, rank=None)
    expected = int(
        np.searchsorted(full_vals, -np.log(tol) / (sigma**2 / 2), side="right")
    )
    # Below the dense cap -> must truncate.
    assert expected < diffusion_mod._HEAT_KERNEL_DENSE_FRACTION * L.shape[0]

    calls = {"n": 0}
    real = diffusion_mod.diffusion_eigenbasis

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(diffusion_mod, "diffusion_eigenbasis", counting)
    rank = heat_kernel_rank(L, sigma, tol=tol)
    assert rank == expected
    assert calls["n"] == 2

    trunc_vals, trunc_vecs = real(L, rank=rank)
    fields = np.zeros((L.shape[0], 3))
    fields[[10, 500, 990], [0, 1, 2]] = 1.0  # point sources, incl. near the ends
    full = diffuse(full_vals, full_vecs, sigma, fields)
    trunc = diffuse(trunc_vals, trunc_vecs, sigma, fields)
    rel = np.max(np.linalg.norm(full - trunc, axis=0) / np.linalg.norm(full, axis=0))
    assert rel < 10 * tol


def test_heat_kernel_rank_dense_fallback_for_small_bandwidth():
    """When smoothing barely spreads (sigma ~ bin spacing), nearly every mode matters,
    so the resolver returns None (use the exact dense eigendecomposition)."""
    L = build_laplacian(grid_graph_2d(30, 30, spacing=1.0))
    assert heat_kernel_rank(L, sigma=0.5) is None


def test_heat_kernel_rank_retains_all_zero_modes_on_disconnected_graph():
    """On a disconnected graph a truncated rank must cover every component's null mode
    (one per component), else diffuse would leak mass across components."""
    graph = nx.disjoint_union(path_graph(40), path_graph(40))
    L = build_laplacian(graph)
    n_components = 2
    # sigma = 20 on two 40-bin chains needs ~8 modes, below the dense cap (16 of 80
    # bins), so the resolver truncates rather than returning None.
    rank = heat_kernel_rank(L, sigma=20.0)
    assert rank is not None
    assert n_components <= rank < L.shape[0]
    eigvals, _ = diffusion_eigenbasis(L, rank=rank)
    assert np.count_nonzero(eigvals < 1e-9) == n_components


@pytest.mark.parametrize("bad_tol", [0.0, -1e-3, 1.0, 2.0])
def test_heat_kernel_rank_rejects_tol_outside_unit_interval(bad_tol):
    """tol must lie in (0, 1): the cutoff lambda = -ln(tol)/t is only defined there."""
    L = build_laplacian(grid_graph_2d(10, 10))
    with pytest.raises(ValidationError, match=r"tolerance must lie in \(0, 1\)"):
        heat_kernel_rank(L, sigma=4.0, tol=bad_tol)


def test_cached_heat_kernel_eigenbasis_reuses_probe_eigensolve(monkeypatch):
    """The fused builder resolves the rank AND returns the basis in the same eigensolves
    the rank probe uses -- it does not resolve, then recompute the basis. The
    resolved-rank basis is cached, so cached_eigenbasis reuses it with no further solve.
    """
    import non_local_detector.likelihoods.diffusion as diffusion_mod

    env = Environment(
        environment_name="fine",
        place_bin_size=2.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(
        np.random.default_rng(0).uniform(1.0, 49.0, size=(6000, 2)),
        infer_track_interior=True,
    )
    sigma = 8.0
    environment_graph(env)  # populate the cached Laplacian (no eigensolve)
    laplacian = env.get_cached(_LAPLACIAN_KEY)

    calls = {"n": 0}
    real = diffusion_mod.diffusion_eigenbasis

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(diffusion_mod, "diffusion_eigenbasis", counting)

    # Eigensolves the pure rank probe performs (operates on the Laplacian, no caching).
    rank = heat_kernel_rank(laplacian, sigma)
    probe_eigensolves = calls["n"]
    assert rank is not None
    assert rank < laplacian.shape[0]

    # The fused builder does the same number of eigensolves -- no extra recompute.
    calls["n"] = 0
    eigvals, eigvecs = cached_heat_kernel_eigenbasis(env, sigma)
    assert calls["n"] == probe_eigensolves
    assert eigvecs.shape[1] == rank

    # The resolved-rank basis is cached; cached_eigenbasis returns it with no eigensolve.
    calls["n"] = 0
    assert cached_eigenbasis(env, rank)[0] is eigvals
    assert calls["n"] == 0

    # A refit at the same bandwidth reuses the memoized rank and cached basis: the
    # adaptive probe does not run again (zero eigensolves), and the same object is
    # returned.
    calls["n"] = 0
    eigvals2, _ = cached_heat_kernel_eigenbasis(env, sigma)
    assert calls["n"] == 0
    assert eigvals2 is eigvals


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
        n = round(2 * half_extent / spacing)
        L = build_laplacian(grid_graph_2d(n, n, spacing))
        center = (n // 2) * n + (n // 2)
        source = np.zeros(L.shape[0])
        source[center] = 1.0
        smoothed = scipy.sparse.linalg.expm_multiply(-t * L, source)

        stds = _weighted_std_per_axis(smoothed, (n, n), spacing)
        np.testing.assert_allclose(stds, sigma, rtol=0.05)


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


def test_to_density_divides_mass_fractions_by_bin_volume():
    """to_density treats columns as per-bin masses: density_i = (m_i / Σm) / v_i.

    Hand-computed for non-uniform volumes. A column normalized by ``Σ v_i m_i``
    also integrates to one, so the integral alone does not pin the conversion.
    """
    smoothed = np.array(
        [
            [2.0, 0.0, 1.0],
            [4.0, 0.0, 3.0],
            [6.0, 0.0, 1.0],
        ]
    )
    bin_sizes = np.array([0.5, 2.0, 1.0])

    density = to_density(smoothed, bin_sizes)

    # Column 0: masses [2, 4, 6] / 12 = [1/6, 1/3, 1/2], divided by the volumes.
    # Column 2: masses [1, 3, 1] / 5 = [0.2, 0.6, 0.2], divided by the volumes.
    np.testing.assert_allclose(density[:, 0], [1 / 3, 1 / 6, 1 / 2], rtol=1e-14)
    np.testing.assert_allclose(density[:, 2], [0.4, 0.3, 0.2], rtol=1e-14)
    # Zero-mass columns map to zeros; the others integrate to one.
    np.testing.assert_array_equal(density[:, 1], 0.0)
    np.testing.assert_allclose(bin_sizes @ density, [1.0, 0.0, 1.0], atol=1e-12)
    # Uniform unit bins reduce to divide-by-sum.
    uniform = to_density(smoothed[:, [0]], np.ones(3))
    np.testing.assert_allclose(uniform.ravel(), smoothed[:, 0] / smoothed[:, 0].sum())


def test_to_density_equal_volumes_match_volume_weighted_normalization():
    """With equal volumes, mass and volume-weighted normalization agree to rounding.

    Equal-volume grids (the usual N-D case) therefore keep their previous fields.
    The formulas are algebraically equal and differ only in rounding. Summing n
    non-negative terms in any order has relative error at most (n - 1)u, so the
    volume-weighted form (n products, one sum, one quotient) and the mass form (one
    sum, two quotients) differ by at most (3n + 1)u. The summation order, and with
    it the exact ulp count, depends on the BLAS build and CPU.
    """
    rng = np.random.default_rng(0)
    n_bins = 40
    smoothed = rng.uniform(0.0, 5.0, size=(n_bins, 3))
    bin_sizes = np.full(n_bins, 2.5**2)

    volume_weighted = smoothed / (bin_sizes @ smoothed)

    unit_roundoff = np.finfo(np.float64).eps / 2
    np.testing.assert_allclose(
        to_density(smoothed, bin_sizes),
        volume_weighted,
        rtol=(3 * n_bins + 1) * unit_roundoff,
        atol=0.0,
    )


def test_to_density_nonuniform_matches_oracle():
    """On a non-uniform 1D chain, diffuse+to_density matches an independent oracle.

    The oracle integrates the mass equation ``dp/dt = -(D - C) M⁻¹ p`` with
    ``expm_multiply`` and converts the resulting per-bin masses to density as
    ``(mass / total mass) / volume``.
    """
    import scipy.sparse.linalg

    rng = np.random.default_rng(1)
    n = 25
    widths = rng.uniform(0.5, 2.0, size=n)
    spacing = 0.5 * (widths[:-1] + widths[1:])  # center-to-center distances
    graph = nx.Graph()
    graph.add_nodes_from((k, {"volume": widths[k]}) for k in range(n))
    for k in range(n - 1):
        graph.add_edge(k, k + 1, distance=spacing[k], conductance=1.0 / spacing[k])
    eigvals, eigvecs = diffusion_eigenbasis(build_laplacian(graph), rank=None)

    field = rng.uniform(0.1, 1.0, size=(n, 1))
    density = to_density(
        diffuse(eigvals, eigvecs, 1.5, field, bin_sizes=widths), widths
    )

    conductance = np.zeros((n, n))
    conductance[np.arange(n - 1), np.arange(1, n)] = 1.0 / spacing
    conductance += conductance.T
    mass_operator = (np.diag(conductance.sum(axis=1)) - conductance) / widths
    oracle = scipy.sparse.linalg.expm_multiply(
        -(1.5**2) / 2.0 * mass_operator, field.ravel()
    )
    oracle_density = oracle / oracle.sum() / widths

    np.testing.assert_allclose(density.ravel(), oracle_density, atol=1e-8)
    np.testing.assert_allclose(widths @ density.ravel(), 1.0, atol=1e-12)


# ==============================================================================
# environment_graph adapter
# ==============================================================================


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
    env = make_linear_track_env(edge_spacing=0.25)
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
    eigvals, _eigvecs = diffusion_eigenbasis(L, rank=6)
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
    assert env.get_cached(_EIGENBASIS_KEY)[None] is full1

    # A cached full-rank basis serves a smaller-rank request by slicing, and the
    # slice is stored under its own rank key.
    trunc_vals, trunc_vecs = cached_eigenbasis(env, rank=5)
    np.testing.assert_allclose(trunc_vals, full1[0][:5])
    np.testing.assert_allclose(trunc_vecs, full1[1][:, :5])
    assert 5 in env.get_cached(_EIGENBASIS_KEY)


def test_cached_eigenbasis_rank_below_components_raises_even_when_full_cached():
    """Slicing a cached full basis must not bypass the rank >= n_components guard."""
    env = make_2d_split_env()  # two interior components -> two null modes required
    cached_eigenbasis(env, rank=None)  # cache the full basis first
    with pytest.raises(ValidationError, match="rank is too small"):
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
        with pytest.raises(ValueError, match="read-only"):
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
    protect the cached eigenbasis from a mutated edge conductance."""
    reference, _ = cached_eigenbasis(make_2d_env(), rank=8)

    env = make_2d_env()  # identical grid (same seed)
    graph, _, _ = environment_graph(env)  # snapshots L before the mutation below
    an_edge = next(iter(graph.edges))
    graph.edges[an_edge]["conductance"] = 999.0  # freeze does NOT block this

    got, _ = cached_eigenbasis(env, rank=8)  # built from the pristine snapshot
    np.testing.assert_allclose(got, reference, atol=1e-8)


def test_eig_cache_invalidated_on_refit():
    """fit_place_grid clears the eig + graph + resolved-rank caches so they rebuild."""
    env = make_2d_env()
    basis = cached_eigenbasis(env, rank=None)
    graph = environment_graph(env)
    cached_heat_kernel_eigenbasis(env, sigma=6.0)  # populates the resolved-rank memo
    assert env.get_cached(_EIGENBASIS_KEY) is not None
    assert env.get_cached(_GRAPH_KEY) is not None
    assert env.get_cached(_HEAT_KERNEL_RANK_KEY) is not None

    rng = np.random.default_rng(0)
    env.fit_place_grid(
        rng.uniform(1.0, 49.0, size=(4000, 2)), infer_track_interior=True
    )
    assert env.get_cached(_EIGENBASIS_KEY) is None
    assert env.get_cached(_GRAPH_KEY) is None
    assert env.get_cached(_HEAT_KERNEL_RANK_KEY) is None

    rebuilt = cached_eigenbasis(env, rank=None)
    assert rebuilt[0] is not basis[0]  # recomputed after invalidation
    assert environment_graph(env)[0] is not graph[0]


# ==============================================================================
# Validation guard rails
# ==============================================================================


def test_environment_graph_unfitted_env_raises():
    """An unfitted environment (no interior mask) raises ValidationError."""
    env = Environment(environment_name="unfit", place_bin_size=5.0)
    with pytest.raises(ValidationError, match="environment must be fitted"):
        environment_graph(env)


def test_check_bandwidth_rejects_nonpositive_sigma():
    """position_std <= 0 is rejected."""
    graph = environment_graph(make_2d_env())[0]
    for bad in (0.0, -1.0):
        with pytest.raises(
            ValidationError,
            match=r"smoothing bandwidth \(position_std\) must be positive",
        ):
            check_smoothing_bandwidth(bad, graph)


def test_check_bandwidth_warns_when_undersmoothing():
    """A sigma below ~1 bin spacing warns; a comfortably larger sigma does not."""
    graph = environment_graph(make_2d_env())[0]  # 5.0-unit bin spacing
    with pytest.warns(UserWarning, match="below the characteristic bin spacing"):
        check_smoothing_bandwidth(1.0, graph)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        check_smoothing_bandwidth(15.0, graph)


# ==============================================================================
# Finite-volume operator: junctions, unequal bin widths, cache layout
# ==============================================================================


def make_star_env(n_arms: int, bin_size: float, arm_length: float = 29.9999):
    """``n_arms`` straight arms of ``arm_length`` meeting at one vertex.

    The default is just under 30 cm, so rounding of the arms' end coordinates
    cannot add a bin to one arm: every arm splits into equal bins of any
    ``bin_size`` that divides 30 cm. Sources within 7 cm of the vertex are more
    than 6 ``sqrt(12.5)`` from the arm ends.
    """
    from track_linearization import make_track_graph

    angles = np.linspace(0.0, 2.0 * np.pi, n_arms, endpoint=False)
    nodes = [(0.0, 0.0)] + [
        (arm_length * np.cos(a), arm_length * np.sin(a)) for a in angles
    ]
    edges = [(0, k) for k in range(1, n_arms + 1)]
    return Environment(
        environment_name=f"star{n_arms}",
        track_graph=make_track_graph(nodes, edges),
        edge_order=edges,
        edge_spacing=0.0,
        place_bin_size=bin_size,
    ).fit_place_grid()


def arm_and_distance_from_vertex(env: Environment):
    """Each interior bin's arm (edge id) and its center's distance from the vertex.

    The arms of :func:`make_star_env` start at the vertex, so a bin's distance
    from it is its linear position minus its arm's start.
    """
    _, node_order, bin_sizes = environment_graph(env)
    arm = env.place_bin_centers_nodes_df_.edge_id.to_numpy()[node_order]
    center = env.place_bin_centers_[node_order, 0]
    start = {e: center[arm == e].min() - bin_sizes[arm == e][0] / 2 for e in arm}
    return arm, center - np.array([start[e] for e in arm])


def mass_kernel_column(env: Environment, sigma: float, source: int) -> np.ndarray:
    """Masses after diffusing a unit mass from ``source`` for ``t = sigma**2 / 2``.

    The operator is symmetric in the volume-scaled variable ``M^{-1/2} m``, so
    the mass kernel is ``M^{1/2} exp(-t L) M^{-1/2}``.
    """
    import scipy.sparse.linalg

    graph, _, bin_sizes = environment_graph(env)
    laplacian = build_laplacian(graph)
    root = np.sqrt(bin_sizes)
    unit = np.zeros(bin_sizes.size)
    unit[source] = 1.0
    return root * scipy.sparse.linalg.expm_multiply(
        -(sigma**2) / 2.0 * laplacian, unit / root
    )


def dense_mass_kernel(env: Environment, sigma: float) -> np.ndarray:
    """``M^{1/2} exp(-t L) M^{-1/2}`` from a dense eigendecomposition."""
    graph, _, bin_sizes = environment_graph(env)
    eigvals, eigvecs = scipy.linalg.eigh(build_laplacian(graph).toarray())
    root = np.sqrt(bin_sizes)
    heat = (eigvecs * np.exp(-(sigma**2) / 2.0 * eigvals)) @ eigvecs.T
    return root[:, None] * heat / root[None, :]


def other_arm_mass(env: Environment, sigma: float, distance: float) -> tuple:
    """Mass reaching the other arms from the first arm's bin ``distance`` away.

    Returns the operator's mass and the metric-graph (Kirchhoff) value
    ``2 (d - 1) / d * P(Z > a / sigma)``, ``a`` the source's distance from the
    vertex: of the Gaussian mass that crosses the vertex, the fraction ``2 / d``
    enters each arm (``d`` arms).
    """
    from scipy.stats import norm

    arm, from_vertex = arm_and_distance_from_vertex(env)
    n_arms = np.unique(arm).size
    first = arm == arm.min()
    source = np.flatnonzero(first)[np.argmin(np.abs(from_vertex[first] - distance))]
    a = from_vertex[source]
    column = mass_kernel_column(env, sigma, source)
    exact = 2.0 * (n_arms - 1) / n_arms * norm.sf(a / sigma)
    return column[~first].sum(), exact


@pytest.mark.parametrize("n_arms", [3, 4])
@pytest.mark.parametrize("distance", [1.0, 7.0])
def test_junction_passes_the_metric_graph_share_to_the_other_arms(n_arms, distance):
    """At a vertex of degree 3 or 4, 2 cm bins and sigma = sqrt(12.5), the mass
    reaching the other arms is within 3% of the Kirchhoff heat kernel's.

    Coupling each pair of end bins at full strength sent up to 1.39 times the
    exact mass across a degree-4 vertex.
    """
    leak, exact = other_arm_mass(
        make_star_env(n_arms, bin_size=2.0), np.sqrt(12.5), distance
    )
    np.testing.assert_allclose(leak / exact, 1.0, atol=0.03)


@pytest.mark.parametrize("bin_size", [2.0, 1.0, 0.5, 0.25])
def test_junction_share_holds_at_every_bin_size(bin_size):
    """Degree 3, source about 7 cm from the vertex: within 2% of the exact
    share at every bin size, so the junction coupling does not depend on the
    grid (it was 1.24 times exact at 2 cm and 1.03 at 0.25 cm)."""
    leak, exact = other_arm_mass(
        make_star_env(3, bin_size=bin_size), np.sqrt(12.5), 7.0
    )
    np.testing.assert_allclose(leak / exact, 1.0, atol=0.02)


@pytest.mark.parametrize("distance", [1.0, 7.0])
def test_degree_two_vertex_is_a_straight_path(distance):
    """Two arms meeting at a vertex diffuse like one straight track."""
    leak, exact = other_arm_mass(
        make_star_env(2, bin_size=2.0), np.sqrt(12.5), distance
    )
    np.testing.assert_allclose(leak / exact, 1.0, atol=0.02)


def _finite_difference_laplacian(graph: nx.Graph) -> np.ndarray:
    """The ``1 / distance**2`` face-adjacent Laplacian of a uniform grid."""
    n_nodes = graph.number_of_nodes()
    laplacian = np.zeros((n_nodes, n_nodes))
    for u, v, data in graph.edges(data=True):
        weight = 1.0 / data["distance"] ** 2
        laplacian[u, v] = laplacian[v, u] = -weight
    laplacian[np.diag_indices(n_nodes)] = -laplacian.sum(axis=1)
    return laplacian


def _laplacian_from_bin_centers(env: Environment) -> np.ndarray:
    """The ``1 / h**2`` Laplacian of an equal-bin grid from its bin centers alone.

    Bins whose centers differ by the spacing ``h`` along exactly one axis are
    neighbors; nothing is read from the diffusion graph.
    """
    _, node_order, bin_sizes = environment_graph(env)
    centers = np.asarray(env.place_bin_centers_)[node_order]
    spacing = bin_sizes[0] ** (1.0 / centers.shape[1])
    offset = np.abs(centers[:, None, :] - centers[None, :, :])
    adjacent = (np.count_nonzero(offset > 1e-9, axis=2) == 1) & np.isclose(
        offset.sum(axis=2), spacing
    )
    laplacian = -adjacent.astype(float) / spacing**2
    laplacian[np.diag_indices(len(centers))] = adjacent.sum(axis=1) / spacing**2
    return laplacian


def make_uniform_track_path_env() -> Environment:
    """A single straight track-graph edge of 50 bins of 2 cm."""
    from track_linearization import make_track_graph

    return Environment(
        environment_name="path",
        track_graph=make_track_graph([(0.0, 0.0), (100.0, 0.0)], [(0, 1)]),
        edge_order=[(0, 1)],
        edge_spacing=0.0,
        place_bin_size=2.0,
    ).fit_place_grid()


@pytest.mark.parametrize(
    "make_env",
    [make_uniform_track_path_env, make_2d_env],
    ids=["track_path", "grid_2d"],
)
def test_uniform_grids_keep_the_finite_difference_operator(make_env):
    """On equal bins the operator is the ``1 / h**2`` finite-difference
    Laplacian, and its heat kernel matches that Laplacian's to 1e-12."""
    env = make_env()
    graph, _, bin_sizes = environment_graph(env)
    assert np.ptp(bin_sizes) <= 1e-12 * bin_sizes.max()
    laplacian = build_laplacian(graph).toarray()
    expected = _laplacian_from_bin_centers(env)

    np.testing.assert_allclose(laplacian, expected, rtol=1e-12, atol=0.0)
    sigma = np.sqrt(12.5)
    t = sigma**2 / 2.0
    root = np.sqrt(bin_sizes)
    kernel = root[:, None] * scipy.linalg.expm(-t * laplacian) / root[None, :]
    expected_kernel = scipy.linalg.expm(-t * expected)
    # Both kernels come from expm, so only the operators differ. Entries far in
    # the tail (~1e-23) carry the rounding of the large ones, which depends on
    # the platform's BLAS, so the absolute tolerance is relative to the peak.
    np.testing.assert_allclose(
        kernel, expected_kernel, rtol=1e-12, atol=1e-12 * expected_kernel.max()
    )


def make_w_track_env() -> Environment:
    """A W-track whose arms and connecting segments have unequal bin widths."""
    from track_linearization import make_track_graph

    nodes = [
        (0.0, 81.0),
        (0.0, 0.0),
        (31.0, 0.0),
        (31.0, 75.0),
        (61.0, 0.0),
        (61.0, 80.0),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)]
    return Environment(
        environment_name="w_track",
        track_graph=make_track_graph(nodes, edges),
        edge_order=edges,
        edge_spacing=0.0,
        place_bin_size=2.0,
    ).fit_place_grid()


def test_kernel_conserves_mass_on_a_track_with_unequal_bin_widths():
    """Every column of the mass kernel sums to one on a W-track whose segments
    have different bin widths, without the clip-and-rescale ``diffuse`` adds."""
    env = make_w_track_env()
    _, _, bin_sizes = environment_graph(env)
    assert np.ptp(bin_sizes) > 0.05

    kernel = dense_mass_kernel(env, np.sqrt(12.5))

    np.testing.assert_allclose(kernel.sum(axis=0), 1.0, rtol=0.0, atol=1e-10)


def test_long_diffusion_spreads_mass_to_uniform_density():
    """After a long diffusion the density (mass over bin width) is the same in
    the 2.0 cm and 1.05 cm regions; equal mass per bin made the narrow region
    1.9 times as dense."""
    env = make_two_width_env()
    _, _, bin_sizes = environment_graph(env)
    density = dense_mass_kernel(env, sigma=2000.0)[:, 0] / bin_sizes
    narrow = bin_sizes < 1.5

    np.testing.assert_allclose(
        density[narrow].mean() / density[~narrow].mean(), 1.0, atol=0.01
    )


def test_diffuse_with_bin_sizes_applies_the_mass_kernel():
    """``diffuse`` given the bin volumes applies ``M^{1/2} exp(-t L) M^{-1/2}``
    to masses, on a track with a junction and unequal bin widths."""
    env = make_w_track_env()
    _, _, bin_sizes = environment_graph(env)
    eigvals, eigvecs = cached_eigenbasis(env, rank=None)
    fields = np.random.default_rng(0).uniform(0.0, 1.0, size=(bin_sizes.size, 3))

    smoothed = diffuse(eigvals, eigvecs, 3.0, fields, bin_sizes=bin_sizes)

    np.testing.assert_allclose(
        smoothed, dense_mass_kernel(env, 3.0) @ fields, rtol=1e-9, atol=1e-12
    )


def test_zero_length_track_edges_build_a_finite_connected_operator():
    """Track nodes sit at zero distance from the bin edges they coincide with;
    the operator stays finite, connected and positive semi-definite, and its
    null vector is the square root of the bin volumes (uniform density)."""
    env = make_w_track_env()
    substrate = env.track_graph_with_bin_centers_edges_
    assert any(d["distance"] == 0.0 for _, _, d in substrate.edges(data=True))
    graph, _, bin_sizes = environment_graph(env)

    laplacian = build_laplacian(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(laplacian)

    assert np.all(np.isfinite(laplacian))
    assert nx.is_connected(graph)
    assert eigvals[0] > -1e-12
    assert eigvals[1] > 1e-6  # one null mode: connected
    null = eigvecs[:, 0] * np.sign(eigvecs[0, 0])
    np.testing.assert_allclose(
        null, np.sqrt(bin_sizes / bin_sizes.sum()), rtol=0.0, atol=1e-10
    )


def test_operator_cache_entries_of_earlier_versions_are_not_used():
    """An environment unpickled from an earlier version carries that version's
    graph, Laplacian, eigenbasis and resolved ranks under the earlier cache
    keys. They describe the earlier operator, so none of them is reused."""
    env = make_star_env(3, bin_size=2.0)
    graph, node_order, bin_sizes = environment_graph(make_star_env(3, bin_size=2.0))
    stale_laplacian = scipy.sparse.csr_matrix(_finite_difference_laplacian(graph))
    stale_basis = scipy.linalg.eigh(stale_laplacian.toarray())
    sigma = float(np.sqrt(12.5))
    for key, stale in {
        "diffusion_graph": (graph, node_order, bin_sizes),
        "diffusion_laplacian": stale_laplacian,
        "diffusion_eigenbasis": {None: stale_basis},
        "diffusion_heat_kernel_rank": {(sigma, 1e-6, 0.2): 3},
    }.items():
        env.set_cached(key, stale)

    fresh = make_star_env(3, bin_size=2.0)
    np.testing.assert_allclose(
        cached_eigenbasis(env, rank=None)[0], cached_eigenbasis(fresh, rank=None)[0]
    )
    assert cached_heat_kernel_eigenbasis(env, sigma)[0].size > 3
    assert not np.allclose(cached_eigenbasis(env, rank=None)[0], stale_basis[0])


def test_center_conductances_equal_the_dense_kron_reduction():
    """Eliminating each group of non-center nodes gives the conductances of
    the whole network's Schur complement onto the centers, here with a group
    of two nodes (a junction split in two) and a group of one (a bin edge)."""
    from non_local_detector.likelihoods.diffusion import _center_conductances

    lengths = {
        ("c0", "a"): 1.0,
        ("c1", "a"): 2.0,
        ("a", "b"): 3.0,
        ("b", "c2"): 1.5,
        ("b", "c3"): 0.5,
        ("c3", "e"): 0.7,
        ("e", "c4"): 0.8,
    }
    substrate = nx.Graph()
    for (u, v), length in lengths.items():
        substrate.add_edge(u, v, distance=length)
    centers = {"c0", "c1", "c2", "c3", "c4"}

    coupled = _center_conductances(substrate, centers)

    nodes = sorted(substrate.nodes)
    index = {node: k for k, node in enumerate(nodes)}
    full = np.zeros((len(nodes), len(nodes)))
    for (u, v), length in lengths.items():
        full[index[u], index[v]] = full[index[v], index[u]] = -1.0 / length
    full[np.diag_indices(len(nodes))] = -full.sum(axis=1)
    kept = [index[c] for c in sorted(centers)]
    gone = [index[n] for n in nodes if n not in centers]
    schur = full[np.ix_(kept, kept)] - full[np.ix_(kept, gone)] @ np.linalg.solve(
        full[np.ix_(gone, gone)], full[np.ix_(gone, kept)]
    )
    expected = {
        (i, j): -schur[a, b]
        for a, i in enumerate(sorted(centers))
        for b, j in enumerate(sorted(centers))
        if a < b and -schur[a, b] > 1e-12
    }

    assert set(coupled) == set(expected)
    for pair, conductance in expected.items():
        np.testing.assert_allclose(coupled[pair][0], conductance, rtol=1e-12)
    # Path lengths through the eliminated nodes.
    assert coupled[("c0", "c2")][1] == 1.0 + 3.0 + 1.5
    assert coupled[("c3", "c4")][1] == 0.7 + 0.8
