# test_composite.py

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.composite import CompositeEnvironment
from non_local_detector.environment.environment import Environment


@pytest.mark.parametrize("auto_bridge", [True, False])
def test_two_envs_composite_basic_properties(
    graph_env, grid_env_from_samples, auto_bridge
):
    """
    Build a CompositeEnvironment from two different sub-environments (a
    Graph-based env and a RegularGrid-based env) and verify that:
      - n_dims stays consistent (should be 2).
      - n_bins equals sum of child bins (when auto_bridge=False) or
        possibly more if auto_bridge=True (because bridges add edges but
        do not change the total bin count).
      - Each child's bin_centers appear as a contiguous block in the composite.
    """
    # graph_env and grid_env_from_samples are both 2D Environments
    # We expect comp_env.n_dims == 2 in any case
    comp_env = CompositeEnvironment(
        [graph_env, grid_env_from_samples], auto_bridge=auto_bridge
    )

    # 1) dimensionality
    assert comp_env.n_dims == 2

    # 2) total number of bins = sum of each child's bins
    total_sub_bins = graph_env.n_bins + grid_env_from_samples.n_bins
    assert comp_env.n_bins == total_sub_bins

    # 3) the first block of comp_env.bin_centers should match graph_env.bin_centers
    #    and the next block should match grid_env_from_samples.bin_centers
    #    (up to ordering—the default CompositeEnvironment stacks children in the order given).
    bc = comp_env.bin_centers
    # split index
    split = graph_env.n_bins
    # Because CompositeEnvironment just concatenates bin_centers in that order,
    # we expect the first split rows ≈ graph_env.bin_centers, and the next rows ≈ grid_env_from_samples.bin_centers
    np.testing.assert_allclose(bc[:split], graph_env.bin_centers)
    np.testing.assert_allclose(bc[split:], grid_env_from_samples.bin_centers)

    # 4) connectivity: check that each child's original neighbors still exist internally
    #    For example, pick one bin index from graph_env, ensure its neighbors in comp_env
    #    contain exactly (or at least) those from graph_env.neighbors(...)
    idx_in_graph = 0
    comp_neighbors_graph_bin = set(comp_env.neighbors(idx_in_graph))
    original_graph_neighbors = set(graph_env.neighbors(idx_in_graph))
    # In the composite, those original neighbors will appear, but shifted by 0 because graph_env is first
    assert original_graph_neighbors.issubset(comp_neighbors_graph_bin)

    # 5) if auto_bridge=False, there should be NO cross-edges between the two sub-graphs.
    #    In other words, no neighbor of a graph_env bin should be >= split (the grid-env block).
    if not auto_bridge:
        # all neighbors of any graph-index must be < split
        for i in range(graph_env.n_bins):
            for nbr in comp_env.neighbors(i):
                assert nbr < split

    # 6) if auto_bridge=True, there should be at least one bridge: find a pair (i, j) where
    #    i < split (graph block) and j >= split (grid block), and that relationship is mutual-nearest.
    if auto_bridge:
        found_bridge = False
        for i in range(graph_env.n_bins):
            for nbr in comp_env.neighbors(i):
                if nbr >= split:
                    # verify that in the reverse direction, i appears in neighbors(nbr)
                    assert i in comp_env.neighbors(nbr)
                    found_bridge = True
                    break
            if found_bridge:
                break
        assert found_bridge, "Expected at least one bridging edge when auto_bridge=True"


@pytest.mark.parametrize(
    "sub_envs, expected_n_bins",
    [
        pytest.param(
            ("simple_graph_env", "simple_graph_env"),
            None,
            id="two_identical_graph_envs",
        ),
        pytest.param(
            ("simple_graph_env", "simple_hex_env"),
            None,
            id="graph_and_hex",
        ),
        pytest.param(
            ("env_all_active_2x2", "env_all_active_2x2"),
            8,
            id="two_2x2_mask_envs",
        ),
    ],
)
def test_composite_various_combinations(
    request, sub_envs, expected_n_bins, auto_bridge=False
):
    """
    Test CompositeEnvironment on various pairs of sub-environment fixtures.
    We use `request.getfixturevalue(...)` to look up fixtures by name.
    """
    # Pull in the two sub-environment fixtures by name
    left_env = request.getfixturevalue(sub_envs[0])
    right_env = request.getfixturevalue(sub_envs[1])

    # A quick sanity check: both sub-envs must be fitted and 2D
    assert left_env._is_fitted and right_env._is_fitted
    assert left_env.n_dims == right_env.n_dims == 2

    comp_env = CompositeEnvironment([left_env, right_env], auto_bridge=auto_bridge)

    # If expected_n_bins was provided, check it; otherwise, derive from sub-envs
    if expected_n_bins is None:
        expected_n_bins = left_env.n_bins + right_env.n_bins

    assert comp_env.n_bins == expected_n_bins

    # Check that each sub-block of bin_centers matches exactly
    split = left_env.n_bins
    np.testing.assert_allclose(comp_env.bin_centers[:split], left_env.bin_centers)
    np.testing.assert_allclose(comp_env.bin_centers[split:], right_env.bin_centers)

    # Verify connectivity counts: each child's internal edge-count is preserved
    # We do this by sub-graph:
    #   Extract all edges among indices [0:split) → must match left_env.connectivity.number_of_edges()
    left_edges_in_comp = [
        (u, v) for u, v in comp_env.connectivity.edges() if u < split and v < split
    ]
    assert len(left_edges_in_comp) == left_env.connectivity.number_of_edges()

    #   Same for the right block (shifted indices)
    right_edges_in_comp = [
        (u, v) for u, v in comp_env.connectivity.edges() if u >= split and v >= split
    ]
    # because right_env's nodes were appended starting at index `split`,
    # we subtract `split` from each endpoint and compare to right_env's edges
    remapped = {(u - split, v - split) for u, v in right_edges_in_comp}
    assert len(remapped) == right_env.connectivity.number_of_edges()
    # we won't check exact node-IDs, just the count


def test_bin_at_and_distance_consistency_within_composite(
    simple_graph_env, simple_hex_env
):
    """
    Confirm that CompositeEnvironment.bin_at(points) always prefers the graph subenv
    whenever graph_env.bin_at(pt) >= 0 (which for our Graph fixture is always true),
    and only falls back to hex if graph returned -1.

    We also verify that distance_between returns zero when both points map to the same
    bin, and a positive finite value otherwise.
    """
    comp_env = CompositeEnvironment(
        [simple_graph_env, simple_hex_env], auto_bridge=True
    )
    P = comp_env.n_bins

    # 1) A point exactly at the first graph bin-center
    pt_graph = simple_graph_env.bin_centers[0].reshape(1, -1)
    idx_graph_only = simple_graph_env.bin_at(pt_graph)[0]
    assert idx_graph_only >= 0

    idx_comp_graph = comp_env.bin_at(pt_graph)[0]
    assert idx_comp_graph == idx_graph_only

    # 2) A point exactly at the first hex bin-center
    pt_hex = simple_hex_env.bin_centers[0].reshape(1, -1)
    # Graph.bin_at(pt_hex) will still be >= 0, so composite returns that graph index
    idx_graph_for_hex = simple_graph_env.bin_at(pt_hex)[0]
    assert idx_graph_for_hex >= 0

    idx_comp_hex = comp_env.bin_at(pt_hex)[0]
    assert idx_comp_hex == idx_graph_for_hex

    # 3) distance_between(pt_graph, pt_graph) == 0
    assert comp_env.distance_between(pt_graph, pt_graph) == 0.0

    # 4) distance_between(pt_graph, pt_hex) must be >= 0, and because both map
    #    to the same graph bin, it will be zero.
    dist_cross = comp_env.distance_between(pt_graph, pt_hex)
    assert isinstance(dist_cross, float)
    # If both map to the same composite index, distance should be 0
    if idx_comp_graph == idx_comp_hex:
        assert dist_cross == 0.0
    else:
        assert dist_cross > 0

    # 5) A far-away point still returns a valid index
    outside_pt = np.array([[999.0, 999.0]])
    idx_out_graph = simple_graph_env.bin_at(outside_pt)[0]
    idx_out_comp = comp_env.bin_at(outside_pt)[0]
    assert idx_out_comp == idx_out_graph
    assert 0 <= idx_out_comp < P


def test_composite_multiway_bridge_consistency(
    simple_graph_env, simple_hex_env, env_all_active_2x2
):
    """
    Construct a composite of three sub-environments and verify:
      1) The total bin count is the sum of the three.
      2) Each sub-block is intact.
      3) If auto_bridge=True, there is at least one bridge between every pair of adjacent sub-env blocks.
      4) If auto_bridge=False, there are no cross-edges at all.
    """
    # Three sub-envs: graph, hex, 2×2 mask
    children = [simple_graph_env, simple_hex_env, env_all_active_2x2]
    total_bins = sum(env.n_bins for env in children)

    # (a) test without auto-bridging
    comp_no_bridge = CompositeEnvironment(children, auto_bridge=False)
    assert comp_no_bridge.n_bins == total_bins
    # verify no edge connects across any sub-block
    # we do that by ensuring that for every composite edge (u,v), either both < split1,
    # or both in [split1:split2), or both ≥ split2
    splits = np.cumsum([children[0].n_bins, children[1].n_bins])
    for u, v in comp_no_bridge.connectivity.edges():
        # find which block u belongs to
        def block_index(idx):
            if idx < splits[0]:
                return 0
            elif idx < splits[1]:
                return 1
            else:
                return 2

        assert block_index(u) == block_index(v)

    # (b) test with auto-bridging
    comp_with_bridge = CompositeEnvironment(children, auto_bridge=True)
    assert comp_with_bridge.n_bins == total_bins
    # verify at least one bridge between each adjacent pair of blocks:
    block_edges_found = {(0, 1): False, (1, 2): False, (0, 2): False}
    for u, v in comp_with_bridge.connectivity.edges():
        bu = 0 if u < splits[0] else (1 if u < splits[1] else 2)
        bv = 0 if v < splits[0] else (1 if v < splits[1] else 2)
        if bu != bv:
            key = tuple(sorted((bu, bv)))
            block_edges_found[key] = True

    assert block_edges_found[(0, 1)]  # graph ↔ hex
    assert block_edges_found[(1, 2)]  # hex ↔ 2×2 mask
    # For non-adjacent blocks (0,2), you might or might not see a direct bridge depending on
    # mutual-nearest-neighbor logic; we only require that adjacent pairs are bridged.


@pytest.mark.parametrize("n_children", [1, 2, 3, 5])
def test_composite_raises_on_empty_list(n_children):
    """
    CompositeEnvironment([]) should raise ValueError. CompositeEnvironment([env], ...) is okay.
    """
    from non_local_detector.environment.composite import CompositeEnvironment

    if n_children == 1:
        # single-element composite should be fine
        dummy = CompositeEnvironment(
            [Environment.from_samples(np.zeros((1, 2)), bin_size=1.0)], auto_bridge=True
        )
        assert isinstance(dummy, CompositeEnvironment)
    else:
        with pytest.raises(ValueError):
            CompositeEnvironment([], auto_bridge=False)
