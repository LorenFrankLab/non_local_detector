"""A zero-mean ``RandomWalk`` keeps a uniform density uniform.

With a flat likelihood the movement prior alone must not move probability
off walls, track ends or holes, or onto junctions: rows sum to one and
``sum_i w_i T_ij = w_j`` for bin volumes ``w`` (bin widths on a track graph,
equal on a grid), so a uniform density is stationary.
"""

import pickle

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from scipy.stats import norm

from non_local_detector.continuous_state_transitions import RandomWalk
from non_local_detector.environment import Environment
from non_local_detector.transition_operators import (
    GaussianGridBlock,
    LazyDenseTransition,
    build_transition_operator,
)

pytestmark = pytest.mark.unit

MOVEMENT_VAR = 6.0  # RandomWalk's default


def line(length=100.0):
    """0 to ``length`` cm in 2 cm bins, every bin on the track."""
    return Environment(place_bin_size=2.0).fit_place_grid(
        np.linspace(0.0, length, 1001)[:, None], infer_track_interior=False
    )


def field_with_hole():
    """100 x 100 cm open field in 4 cm bins with a 20 x 20 cm interior obstacle."""
    xy = np.stack(
        np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101)), -1
    ).reshape(-1, 2)
    kwargs = {"place_bin_size": 4.0}
    shape = Environment(**kwargs).fit_place_grid(xy, infer_track_interior=False)
    mask = np.ones(shape.centers_shape_, dtype=bool)
    mask[10:15, 8:13] = False
    return Environment(**kwargs, is_track_interior=mask).fit_place_grid(
        xy, infer_track_interior=False
    )


def track(positions, edges, edge_spacing=0.0):
    graph = nx.Graph()
    for node, position in positions.items():
        graph.add_node(node, pos=position)
    for edge_id, (node1, node2) in enumerate(edges):
        length = np.linalg.norm(np.subtract(positions[node1], positions[node2]))
        graph.add_edge(node1, node2, distance=float(length), edge_id=edge_id)
    return Environment(
        place_bin_size=2.0,
        track_graph=graph,
        edge_order=edges,
        edge_spacing=edge_spacing,
    ).fit_place_grid()


def y_track():
    """Y track whose arms have bins 2.0, 1.92 and 1.89 cm wide."""
    return track(
        {0: (0.0, 0.0), 1: (0.0, 50.0), 2: (-15.0, 70.0), 3: (20.0, 70.0)},
        [(0, 1), (1, 2), (1, 3)],
    )


def bin_widths(environment):
    widths = np.diff(environment.edges_[0])
    return np.where(environment.is_track_interior_flat_, widths, 0.0)


def assert_uniform_density_is_stationary(transition, volumes):
    """Rows sum to one and ``volumes @ transition == volumes`` to 1e-10."""
    interior = volumes > 0
    np.testing.assert_allclose(
        transition[interior].sum(axis=1), 1.0, rtol=0, atol=1e-10
    )
    np.testing.assert_array_equal(transition[~interior], 0.0)
    np.testing.assert_array_equal(transition[:, ~interior], 0.0)
    np.testing.assert_allclose(
        volumes @ transition / volumes.max(),
        volumes / volumes.max(),
        rtol=0,
        atol=1e-10,
    )


def test_uniform_stays_uniform_on_a_line():
    env = line()
    transition = RandomWalk().make_state_transition([env])
    n_bins = transition.shape[0]

    probability = np.full(n_bins, 1 / n_bins)
    for _ in range(50):
        probability = probability @ transition

    np.testing.assert_allclose(probability * n_bins, 1.0, rtol=0, atol=1e-8)
    assert_uniform_density_is_stationary(transition, np.ones(n_bins))


def test_uniform_stays_uniform_in_a_field_with_a_hole():
    """Including the corners, wall midpoints and the bins around the hole."""
    env = field_with_hole()
    interior = env.is_track_interior_flat_
    transition = RandomWalk().make_state_transition([env])

    probability = interior / interior.sum()
    for _ in range(50):
        probability = probability @ transition

    np.testing.assert_allclose(
        probability[interior] * interior.sum(), 1.0, rtol=0, atol=1e-8
    )
    assert_uniform_density_is_stationary(transition, interior.astype(float))


def test_uniform_density_stays_uniform_on_a_track_graph_with_unequal_bins():
    """Stationary mass is proportional to bin width at the arm ends and at
    the junction, as everywhere else."""
    env = y_track()
    widths = bin_widths(env)
    assert len(np.unique(np.round(widths[widths > 0], 4))) == 3
    transition = RandomWalk().make_state_transition([env])

    probability = widths / widths.sum()
    for _ in range(50):
        probability = probability @ transition

    np.testing.assert_allclose(
        probability / (widths / widths.sum()), 1.0, rtol=0, atol=1e-8
    )
    assert_uniform_density_is_stationary(transition, widths)


def test_disconnected_tracks_keep_their_mass():
    env = track(
        {0: (0.0, 0.0), 1: (30.0, 0.0), 2: (0.0, 20.0), 3: (17.0, 20.0)},
        [(0, 1), (2, 3)],
        edge_spacing=4.0,
    )
    edge = np.asarray(env.place_bin_centers_nodes_df_.edge_id)
    transition = RandomWalk().make_state_transition([env])

    for region, other in ((edge == 0, edge == 1), (edge == 1, edge == 0)):
        np.testing.assert_allclose(
            transition[region].sum(axis=1), 1.0, rtol=0, atol=1e-10
        )
        np.testing.assert_array_equal(transition[np.ix_(region, other)], 0.0)
    assert_uniform_density_is_stationary(transition, bin_widths(env))


def test_rows_far_from_the_walls_are_the_normalized_gaussian():
    """Only the boundary treatment changes: far from the walls each row is the
    Gaussian density at the bin centers divided by its sum. The boundary
    correction decays like exp(-sqrt(pi) x / sd), so it is below 1e-10 from
    13 movement standard deviations on."""
    env = line(200.0)
    centers = env.place_bin_centers_[:, 0]
    sd = np.sqrt(MOVEMENT_VAR)
    kernel = norm.pdf(centers[None, :], loc=centers[:, None], scale=sd)
    row_normalized = kernel / kernel.sum(axis=1, keepdims=True)
    distance_to_wall = np.minimum(centers - 0.0, 200.0 - centers)
    far = distance_to_wall > 13 * sd
    assert far.sum() >= 30

    transition = RandomWalk().make_state_transition([env])

    np.testing.assert_allclose(transition[far], row_normalized[far], rtol=0, atol=1e-10)


def reflected_walk(centers, length):
    """The Gaussian reflected at both walls of [0, length], at the bin centers."""
    reflected = np.zeros((len(centers), len(centers)))
    for image in range(-3, 4):
        for sign in (1, -1):
            reflected += norm.pdf(
                sign * centers[None, :] + 2 * image * length,
                loc=centers[:, None],
                scale=np.sqrt(MOVEMENT_VAR),
            )
    return reflected / reflected.sum(axis=1, keepdims=True)


def test_close_to_image_reflection_on_a_line():
    """At a wall the walk is close to reflecting the Gaussian at the wall."""
    env = line()
    reflected = reflected_walk(env.place_bin_centers_[:, 0], 100.0)

    transition = RandomWalk().make_state_transition([env])

    assert np.max(np.abs(transition - reflected)) <= 0.01


def test_steady_state_filter_at_a_wall_is_close_to_reflection():
    """A likelihood centred on the wall (sd 4 cm) keeps about as much
    posterior in the edge bin as the reflected walk does (0.553; truncating
    and normalizing rows kept 0.472)."""
    env = line()
    centers = env.place_bin_centers_[:, 0]
    likelihood = norm.pdf(centers, loc=0.0, scale=4.0)

    def steady_edge_posterior(transition):
        posterior = np.full(len(centers), 1 / len(centers))
        for _ in range(2000):
            posterior = likelihood * (posterior @ transition)
            posterior /= posterior.sum()
        return posterior[0]

    walk = steady_edge_posterior(RandomWalk().make_state_transition([env]))
    reflected = steady_edge_posterior(reflected_walk(centers, 100.0))

    assert abs(walk - reflected) <= 0.02


@pytest.mark.parametrize(
    "transition",
    [
        RandomWalk(),
        RandomWalk(use_manifold_distance=True),
    ],
)
def test_uniform_density_on_a_grid_with_graph_distance(transition):
    """The graph-distance walk on a grid also keeps a uniform density."""
    env = field_with_hole()
    matrix = transition.make_state_transition([env])

    assert_uniform_density_is_stationary(
        matrix, env.is_track_interior_flat_.astype(float)
    )


@pytest.mark.parametrize("make_environment", [line, field_with_hole])
def test_structured_block_matches_the_dense_walk(make_environment):
    """Forward, backward and pair products of the separable block equal the
    dense walk's in float32, to 1e-6 relative."""
    env = make_environment()
    n_bins = len(env.place_bin_centers_)
    dense = RandomWalk().make_state_transition([env])
    operator = build_transition_operator([[RandomWalk()]], [env], (n_bins,))
    assert isinstance(operator.blocks[0][0], GaussianGridBlock)
    rng = np.random.default_rng(0)
    probability = rng.random(n_bins)
    probability /= probability.sum()
    values = rng.random(n_bins)
    matrix = jnp.asarray(dense, dtype=jnp.float32)
    p32 = jnp.asarray(probability, dtype=jnp.float32)
    v32 = jnp.asarray(values, dtype=jnp.float32)

    np.testing.assert_allclose(operator.forward(p32), p32 @ matrix, rtol=1e-6, atol=0)
    np.testing.assert_allclose(operator.backward(v32), matrix @ v32, rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        operator.pair_products(p32, v32), [[p32 @ (matrix @ v32)]], rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(LazyDenseTransition(operator)), dense, rtol=0, atol=1e-12
    )


def test_structured_block_pickled_in_the_earlier_layout_applies_its_matrix():
    """A block pickled before it stored ``column_scale`` holds ``factors``,
    ``mask`` and the row sums ``row_normalization``, and still applies the
    truncated, row-normalized matrix ``diag(mask / row_normalization) K
    diag(mask)`` it was built with."""
    env = field_with_hole()
    n_bins = len(env.place_bin_centers_)
    block = build_transition_operator([[RandomWalk()]], [env], (n_bins,)).blocks[0][0]
    mask = env.is_track_interior_flat_
    factors = tuple(np.asarray(factor) for factor in block.factors)
    kernel = np.kron(*factors) * mask[None, :]
    kernel[~mask] = 0.0
    row_sums = kernel.sum(axis=1)
    earlier_state = {"factors": factors, "mask": mask, "row_normalization": row_sums}

    class EarlierBlock:
        """Pickles as a ``GaussianGridBlock`` with the earlier fields."""

        def __reduce_ex__(self, protocol):
            return (object.__new__, (GaussianGridBlock,), earlier_state)

    restored = pickle.loads(pickle.dumps(EarlierBlock()))
    assert type(restored) is GaussianGridBlock
    earlier = np.divide(
        kernel,
        row_sums[:, None],
        out=np.zeros_like(kernel),
        where=mask[:, None],
    )
    rng = np.random.default_rng(1)
    p32 = jnp.asarray(rng.random(n_bins) / n_bins, dtype=jnp.float32)
    v32 = jnp.asarray(rng.random(n_bins), dtype=jnp.float32)
    matrix = jnp.asarray(earlier, dtype=jnp.float32)

    np.testing.assert_allclose(restored.forward(p32), p32 @ matrix, rtol=1e-6, atol=0)
    np.testing.assert_allclose(restored.backward(v32), matrix @ v32, rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        restored.entries(np.arange(n_bins)[:, None], np.arange(n_bins)[None, :]),
        earlier,
        rtol=1e-12,
        atol=1e-15,
    )
