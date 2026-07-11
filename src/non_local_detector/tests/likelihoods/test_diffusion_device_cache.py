import pickle

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    diffuse,
    diffusion_eigenbasis,
    get_device_basis,
    heat_kernel_apply,
)


def _grid_basis(n=6, rank=None):
    g = nx.grid_2d_graph(n, n)
    g = nx.convert_node_labels_to_integers(g)
    for u, v in g.edges():
        g[u][v]["distance"] = 1.0
    L = build_laplacian(g)
    vals, vecs = diffusion_eigenbasis(L, rank=rank)
    return vals, vecs


def test_heat_kernel_apply_matches_diffuse_full_and_truncated():
    for rank in (None, 8):  # full-rank (no lobes) and truncated (negative lobes)
        vals, vecs = _grid_basis(n=6, rank=rank)
        rng = np.random.default_rng(0)
        fields = np.abs(rng.standard_normal((vecs.shape[0], 4)))  # point-ish sources
        ref = diffuse(vals, vecs, sigma=2.0, fields=fields, component_labels=None)
        got = np.asarray(
            heat_kernel_apply(
                jnp.asarray(vals), jnp.asarray(vecs), 2.0, jnp.asarray(fields), None
            )
        )
        assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
        # mass conservation regardless of rank
        assert_allclose(got.sum(0), np.asarray(fields).sum(0), rtol=1e-5, atol=1e-6)


def test_heat_kernel_apply_disconnected_components_preserve_mass_independently():
    """Two disjoint grids with UNEQUAL masses: each component's mass is preserved
    separately (a single global rescale would leak mass between them)."""
    import scipy.sparse.csgraph

    g = nx.disjoint_union(nx.grid_2d_graph(4, 4), nx.grid_2d_graph(3, 3))
    g = nx.convert_node_labels_to_integers(g)
    for u, v in g.edges():
        g[u][v]["distance"] = 1.0
    L = build_laplacian(g)
    n_comp, labels = scipy.sparse.csgraph.connected_components(L, directed=False)
    assert n_comp == 2
    vals, vecs = diffusion_eigenbasis(
        L, rank=6
    )  # truncated -> point sources produce negative lobes
    comp0 = np.flatnonzero(labels == 0)
    comp1 = np.flatnonzero(labels == 1)
    fields = np.zeros((vecs.shape[0], 2), dtype=float)
    # BOTH components carry mass in EACH column, with DIFFERENT ratios — so a single
    # global (whole-array) rescale would shift mass between components; only a
    # per-component rescale preserves each component's mass. (If each column's mass
    # lived in one component, global and per-component rescale would be identical and
    # the test could not tell them apart.)
    fields[comp0[0], 0] = 10.0
    fields[comp1[0], 0] = 1.0  # column 0: 10:1 across components
    fields[comp0[1], 1] = 2.0
    fields[comp1[1], 1] = 7.0  # column 1: 2:7 across components
    ref = diffuse(vals, vecs, sigma=1.5, fields=fields, component_labels=labels)
    got = np.asarray(
        heat_kernel_apply(
            jnp.asarray(vals), jnp.asarray(vecs), 1.5, jnp.asarray(fields), labels
        )
    )
    assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    # per-component mass preserved independently
    for comp in (0, 1):
        m = labels == comp
        assert_allclose(got[m].sum(0), fields[m].sum(0), rtol=1e-5, atol=1e-6)


def _fitted_env():
    env = Environment(position_range=[(0, 10)], place_bin_size=1.0)
    return env.fit_place_grid(
        position=np.linspace(0, 10, 50)[:, None], infer_track_interior=True
    )


def test_get_device_basis_hit_and_device_arrays():
    env = _fitted_env()
    _, vecs = __import__(
        "non_local_detector.likelihoods.diffusion", fromlist=["cached_eigenbasis"]
    ).cached_eigenbasis(env, None)
    rank = vecs.shape[1]
    b1 = get_device_basis(env, rank)
    b2 = get_device_basis(env, rank)
    assert isinstance(b1[1], jax.Array)  # device eigvecs
    assert b1[1] is b2[1]  # cache hit returns same object


def test_device_cache_invalidated_on_refit():
    env = _fitted_env()
    get_device_basis(
        env,
        __import__(
            "non_local_detector.likelihoods.diffusion", fromlist=["cached_eigenbasis"]
        )
        .cached_eigenbasis(env, None)[1]
        .shape[1],
    )
    assert hasattr(env, "_diffusion_device_basis_")
    env.fit_place_grid(
        position=np.linspace(0, 20, 80)[:, None], infer_track_interior=True
    )
    assert not hasattr(env, "_diffusion_device_basis_")


def test_device_cache_excluded_from_pickle():
    env = _fitted_env()
    get_device_basis(
        env,
        __import__(
            "non_local_detector.likelihoods.diffusion", fromlist=["cached_eigenbasis"]
        )
        .cached_eigenbasis(env, None)[1]
        .shape[1],
    )
    blob = pickle.dumps(env)  # must not raise on the JAX Device key
    env2 = pickle.loads(blob)
    assert not hasattr(env2, "_diffusion_device_basis_")
