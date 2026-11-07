import networkx as nx
import numpy as np
import pytest

from non_local_detector.continuous_state_transitions import (
    ContinuousTransition,
    Discrete,
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    TransitionContext,
    Uniform,
    _normalize_row_probability,
    estimate_movement_var,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import (
    ConfigurationError,
    NotFittedError,
    ValidationError,
)
from tests.conftest import assert_stochastic_matrix, set_place_grid

pytestmark = pytest.mark.unit


def test_random_walk_row_stochastic_and_spread_var(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    rw_tight = RandomWalk(
        environment_name=env.environment_name, movement_var=0.1
    ).make_state_transition(envs)
    rw_wide = RandomWalk(
        environment_name=env.environment_name, movement_var=10.0
    ).make_state_transition(envs)

    assert rw_tight.shape == (env.place_bin_centers_.shape[0],) * 2
    assert_stochastic_matrix(rw_tight, atol=1e-7)
    assert_stochastic_matrix(rw_wide, atol=1e-7)

    # Off-diagonal mass increases with larger variance
    offdiag_tight = (1.0 - np.diag(rw_tight)).mean()
    offdiag_wide = (1.0 - np.diag(rw_wide)).mean()
    assert offdiag_wide > offdiag_tight


def test_random_walk_with_directional_constraints(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    base = RandomWalk(
        environment_name=env.environment_name, movement_var=3.0
    ).make_state_transition(envs)
    up = RandomWalkDirection1(
        environment_name=env.environment_name, movement_var=3.0
    ).make_state_transition(envs)
    down = RandomWalkDirection2(
        environment_name=env.environment_name, movement_var=3.0
    ).make_state_transition(envs)

    # Directional versions are masked upper/lower triangular
    assert np.allclose(
        up,
        np.triu(base) / np.clip(np.triu(base).sum(axis=1, keepdims=True), 1e-12, None),
    )
    assert np.allclose(
        down,
        np.tril(base) / np.clip(np.tril(base).sum(axis=1, keepdims=True), 1e-12, None),
    )


@pytest.mark.parametrize(
    ("n_dims", "movement_mean", "movement_var", "exact"),
    [
        (1, 0.0, 6.0, True),
        (1, 1.5, 0.01, True),
        (2, 0.0, 6.0, True),
        (2, np.array([1.0, -2.0]), np.array([2.0, 9.0]), True),
        (2, 0.0, np.array([[3.0, 1.2], [1.2, 2.0]]), False),
        (3, 0.0, np.diag([1.0, 4.0, 9.0]), True),
    ],
)
def test_euclidean_random_walk_matches_scipy_row_by_row(
    monkeypatch, n_dims, movement_mean, movement_var, exact
):
    """The blocked build equals scipy's density of each row: bit-identical for
    a scalar or diagonal covariance, to float64 rounding for a full one. Saved
    models compare rebuilt transitions with fingerprints of this build."""
    from scipy.stats import multivariate_normal

    from non_local_detector import continuous_state_transitions as cst

    rng = np.random.default_rng(0)
    centers = rng.uniform(0.0, 12.0, size=(23, n_dims))
    env = set_place_grid(Environment(), place_bin_centers=centers)
    expected = np.stack(
        [
            multivariate_normal(mean=center + movement_mean, cov=movement_var).pdf(
                centers
            )
            for center in centers
        ]
    )
    # Blocks of 2 rows: several blocks and a ragged last one.
    monkeypatch.setattr(cst, "_RANDOM_WALK_BLOCK_ENTRIES", 2 * len(centers))

    actual = cst._euclidean_random_walk(env, movement_mean, movement_var)

    if exact:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_euclidean_random_walk_accepts_flat_1d_centers():
    """Bin centers of shape (n_bins,) are one-dimensional positions, as for
    the per-row scipy build."""
    from scipy.stats import multivariate_normal

    from non_local_detector import continuous_state_transitions as cst

    centers = np.linspace(0.0, 9.0, 10)
    env = set_place_grid(Environment(), place_bin_centers=centers)
    expected = np.stack(
        [multivariate_normal(mean=c + 0.5, cov=2.0).pdf(centers) for c in centers]
    )

    actual = cst._euclidean_random_walk(env, 0.5, 2.0)

    np.testing.assert_array_equal(actual, expected)


def test_random_walk_on_one_bin_grid_is_identity():
    """A one-bin grid's Euclidean random walk is the (1, 1) matrix [[1]]:
    ``make_state_transition`` masks and normalizes it as a square matrix."""
    env = Environment(place_bin_size=100.0, position_range=((0.0, 30.0),))
    env = env.fit_place_grid(np.linspace(0.0, 30.0, 11)[:, None])
    assert env.place_bin_centers_.shape == (1, 1)

    transition = RandomWalk().make_state_transition([env])

    np.testing.assert_array_equal(transition, [[1.0]])


@pytest.mark.parametrize(("movement_mean", "movement_var"), [(0.0, 6.0), (1.5, 0.5)])
def test_track_graph_random_walk_matches_scipy_per_pair(
    monkeypatch, movement_mean, movement_var
):
    """The blocked track-graph build is bit-identical to scipy's density at
    each pair's graph distance, with 0 for bins off the track (the gaps that
    edge_spacing leaves)."""
    import networkx as nx
    from scipy.stats import multivariate_normal

    from non_local_detector import continuous_state_transitions as cst

    positions = {0: (0.0, 0.0), 1: (0.0, 20.0), 2: (-15.0, 35.0), 3: (15.0, 35.0)}
    edges = [(0, 1), (1, 2), (1, 3)]
    graph = nx.Graph()
    for node, position in positions.items():
        graph.add_node(node, pos=position)
    for edge_id, (node1, node2) in enumerate(edges):
        length = np.linalg.norm(np.subtract(positions[node1], positions[node2]))
        graph.add_edge(node1, node2, distance=float(length), edge_id=edge_id)
    env = Environment(
        place_bin_size=2.0, track_graph=graph, edge_order=edges, edge_spacing=5.0
    ).fit_place_grid()
    nodes = np.asarray(env.place_bin_centers_nodes_df_.node_id)
    distances = env.distance_between_nodes_
    assert np.any(nodes == -1)  # bins off the track
    gaussian = multivariate_normal(mean=movement_mean, cov=movement_var)
    expected = np.zeros((len(nodes), len(nodes)))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 in distances and node2 in distances[node1]:
                expected[i, j] = gaussian.pdf(distances[node1][node2])
    # Blocks of 3 rows: several blocks and a ragged last one.
    monkeypatch.setattr(cst, "_RANDOM_WALK_BLOCK_ENTRIES", 3 * len(nodes))

    actual = cst._random_walk_on_track_graph(
        env.place_bin_centers_, movement_mean, movement_var, nodes, distances
    )

    np.testing.assert_array_equal(actual, expected)


def test_random_walk_invalid_direction_raises_at_construction():
    """An invalid direction is rejected when the RandomWalk is made, not
    only when its matrix is built."""
    with pytest.raises(
        ValidationError, match="direction must be 'inward' or 'outward'"
    ):
        RandomWalk(movement_var=3.0, direction="sideways")


def test_random_walk_direction_assigned_after_construction_raises():
    """An invalid direction set after construction raises ValidationError
    when the matrix is built, not a bare KeyError."""
    position = np.array([[0.0, 0.0], [3.0, 4.0]])
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False
    )
    rw = RandomWalk(use_manifold_distance=True, direction="inward")
    rw.direction = "sideways"
    with pytest.raises(
        ValidationError, match="direction must be 'inward' or 'outward'"
    ):
        rw.make_state_transition((env,))


def test_environment_name_is_part_of_the_protocol():
    """Every package transition declares environment_name; Discrete has no
    place bins, so its environment_name is None."""
    assert Discrete().environment_name is None
    assert RandomWalk("track").environment_name == "track"
    assert vars(Discrete()) == {}


def test_identity_transition_is_identity_masked(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    tm = Identity(environment_name=env.environment_name).make_state_transition(envs)
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert_stochastic_matrix(tm, atol=1e-7)
    assert np.allclose(np.diag(tm), 1.0)
    assert np.allclose(tm - np.diag(np.diag(tm)), 0.0)


def test_uniform_transition_within_env(simple_1d_environment):
    env = simple_1d_environment
    envs = (env,)
    tm = Uniform(environment_name=env.environment_name).make_state_transition(envs)
    n = env.place_bin_centers_.shape[0]
    assert tm.shape == (n, n)
    assert_stochastic_matrix(tm, atol=1e-7)
    # Each row is uniform
    assert np.allclose(tm, np.ones((n, n)) / n)


def test_uniform_transition_between_envs(simple_1d_environment):
    env1 = simple_1d_environment
    env2 = Environment(
        environment_name="env2", place_bin_size=1.0, position_range=((0.0, 4.0),)
    ).fit_place_grid(
        position=np.linspace(0.0, 4.0, 5)[:, None], infer_track_interior=False
    )
    envs = (env1, env2)
    tm = Uniform(
        environment_name=env1.environment_name, environment2_name=env2.environment_name
    ).make_state_transition(envs)
    assert tm.shape == (
        env1.place_bin_centers_.shape[0],
        env2.place_bin_centers_.shape[0],
    )
    # Rectangular (env1 bins x env2 bins), so check row sums directly.
    np.testing.assert_allclose(tm.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7)
    # Rows uniform over destination bins
    assert np.allclose(np.unique(tm, axis=1), np.ones((tm.shape[0], 1)) / tm.shape[1])


def test_empirical_movement_row_stochastic_and_locality(simple_1d_environment):
    # Create a simple path that marches right
    env = simple_1d_environment
    envs = (env,)
    pos = np.linspace(0.0, 10.0, 51)[:, None]
    # Use empirical transition with provided positions
    tm = EmpiricalMovement(
        environment_name=env.environment_name, speedup=1, is_time_reversed=False
    ).make_state_transition(envs, TransitionContext(position=pos))
    assert tm.shape == (env.place_bin_centers_.shape[0],) * 2
    assert_stochastic_matrix(tm, atol=1e-6)
    # Empirical movement should favor near-diagonal transitions on average
    nearer = np.mean(np.diag(tm))
    farther = np.mean(tm[:, 0] + tm[:, -1])
    assert nearer >= farther


@pytest.mark.parametrize("n_dims", [1, 2])
@pytest.mark.parametrize("explicit_range", [False, True])
def test_empirical_movement_uses_fitted_edges(n_dims, explicit_range):
    """Ordinary fitted environments define both source and destination bins."""
    if n_dims == 1:
        position = np.array([[0.5], [1.5], [2.5], [0.5]])
    else:
        position = np.array(
            [[0.5, 0.5], [0.5, 2.5], [2.5, 2.5], [2.5, 0.5], [0.5, 0.5]]
        )
    env = Environment(
        place_bin_size=1.0,
        position_range=[(0.0, 3.0)] * n_dims if explicit_range else None,
        infer_track_interior=False,
    ).fit_place_grid(position, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    n_bins = len(env.place_bin_centers_)
    # Visit every fitted bin once, staying away from edge ownership choices.
    position = np.concatenate((env.place_bin_centers_, env.place_bin_centers_[:1]))
    bins: np.ndarray = np.concatenate((np.arange(n_bins), np.array([0], dtype=int)))
    expected = np.zeros((n_bins,) * 2)
    np.add.at(expected, (bins[:-1], bins[1:]), 1)
    row_mass = expected.sum(axis=1, keepdims=True)
    expected /= np.where(row_mass > 0, row_mass, 1)

    transition = EmpiricalMovement().make_state_transition(
        (env,), TransitionContext(position=position)
    )

    np.testing.assert_array_equal(transition, expected)
    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0)
    np.testing.assert_array_equal(transition.sum(axis=1)[row_mass[:, 0] > 0], 1)


@pytest.mark.parametrize(
    "transition",
    [
        Discrete(),
        EmpiricalMovement(),
        Identity(),
        RandomWalk(),
        RandomWalkDirection1(),
        RandomWalkDirection2(),
        Uniform(),
    ],
    ids=lambda transition: type(transition).__name__,
)
def test_transitions_share_one_signature(transition):
    """Every package transition is a ContinuousTransition called one way."""
    assert isinstance(transition, ContinuousTransition)
    env = Environment(place_bin_size=1.0).fit_place_grid(
        np.linspace(0.0, 10.0, 11)[:, None], infer_track_interior=False
    )
    context = TransitionContext(position=np.linspace(0.0, 10.0, 51)[:, None])
    matrix = transition.make_state_transition(environments=(env,), context=context)
    n_bins = 1 if isinstance(transition, Discrete) else len(env.place_bin_centers_)
    assert matrix.shape == (n_bins, n_bins)


@pytest.mark.parametrize("context", [None, TransitionContext()])
def test_empirical_movement_without_position_is_rejected(
    context, simple_1d_environment
):
    with pytest.raises(ValueError, match="position"):
        EmpiricalMovement().make_state_transition((simple_1d_environment,), context)


def test_context_defaults_select_all_samples_of_group_zero():
    """Unset training and encoding labels default as the detector fit does."""
    position = np.zeros((4, 1))
    context = TransitionContext(position, is_training=np.ones((4, 1)))
    np.testing.assert_array_equal(context.is_training, np.ones(4))
    np.testing.assert_array_equal(context.encoding_group_labels, np.zeros(4))
    assert context.environment_labels is None
    default = TransitionContext(position=position)
    np.testing.assert_array_equal(default.is_training, np.ones(4, dtype=bool))
    assert TransitionContext().encoding_group_labels is None


@pytest.mark.parametrize("excluded_by", ["training", "encoding_group", "environment"])
@pytest.mark.parametrize("is_time_reversed", [False, True])
def test_empirical_movement_does_not_pair_samples_across_excluded_samples(
    excluded_by, is_time_reversed
):
    """Samples on either side of an excluded stretch are not one step apart.

    The animal sits in the first bin, runs to the last one (excluded), and
    sits there. The included samples never move, so no one-step transition
    leaves either bin; pairing them after dropping the run gave a jump from
    the first bin to the last.
    """
    position = np.concatenate(
        [np.full(50, 5.0), np.linspace(5.0, 95.0, 50), np.full(50, 95.0)]
    )[:, None]
    env = Environment(
        place_bin_size=10.0, position_range=[(0.0, 100.0)]
    ).fit_place_grid(position, infer_track_interior=False)
    is_run = np.zeros(150, dtype=bool)
    is_run[50:100] = True
    context = {
        "training": TransitionContext(position, is_training=~is_run),
        "encoding_group": TransitionContext(
            position, encoding_group_labels=is_run.astype(int)
        ),
        "environment": TransitionContext(
            position, environment_labels=np.where(is_run, "other", "")
        ),
    }[excluded_by]

    transition = EmpiricalMovement(
        is_time_reversed=is_time_reversed
    ).make_state_transition((env,), context)

    expected = np.zeros((10, 10))
    expected[0, 0] = expected[-1, -1] = 1.0
    np.testing.assert_array_equal(transition, expected)


@pytest.mark.parametrize("outside", [400.0, -300.0])
def test_empirical_movement_leaves_out_samples_beyond_the_grid(outside):
    """Steps into or out of a sample beyond the grid's outer edges are not counted.

    A sweep up through the ten bins, 50 samples far outside the grid, and a
    sweep back down. Only the sweeps' steps count: pinning the outside
    samples to the edge bin made it a long stay there.
    """
    up = np.arange(5.0, 100.0, 10.0)
    position = np.concatenate([up, np.full(50, outside), up[::-1]])[:, None]
    env = Environment(
        place_bin_size=10.0, position_range=[(0.0, 100.0)]
    ).fit_place_grid(up[:, None], infer_track_interior=False)

    transition = EmpiricalMovement().make_state_transition(
        (env,), TransitionContext(position=position)
    )

    # Each inner bin is left once upward and once downward; the end bins once.
    expected = np.zeros((10, 10))
    for bin_ in range(1, 9):
        expected[bin_, bin_ + 1] = expected[bin_, bin_ - 1] = 0.5
    expected[0, 1] = expected[9, 8] = 1.0
    np.testing.assert_array_equal(transition, expected)


def _l_track_environment():
    """L-shaped track graph: bins 0-9 on one edge, gap bin 10, bins 11-20."""
    import networkx as nx

    graph = nx.Graph()
    for node, xy in {0: (0.0, 0.0), 1: (50.0, 0.0), 2: (50.0, 50.0)}.items():
        graph.add_node(node, pos=xy)
    graph.add_edge(0, 1, distance=50.0, edge_id=0)
    graph.add_edge(1, 2, distance=50.0, edge_id=1)
    return Environment(
        place_bin_size=5.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=15.0,
    ).fit_place_grid()


def _open_field_with_a_hole():
    """Grid of 10-unit bins whose center bin (12) is off the interior.

    The fit adds a bin beyond each end of the range, so the 0-30 range gives
    5 x 5 bins with edges -10, 0, ..., 40 in each dimension.
    """
    corners = np.array([[0.0, 0.0], [30.0, 30.0]])
    env = Environment(
        place_bin_size=10.0, position_range=[(0.0, 30.0), (0.0, 30.0)]
    ).fit_place_grid(corners, infer_track_interior=False)
    is_track_interior = np.ones((5, 5), dtype=bool)
    is_track_interior[2, 2] = False
    return set_place_grid(env, is_track_interior=is_track_interior)


@pytest.mark.parametrize("layout", ["track_graph", "open_field"])
def test_empirical_movement_bins_positions_as_the_environment_does(layout):
    """A position in a bin off the interior counts in the nearest interior bin.

    That is how ``Environment.get_bin_ind`` assigns positions everywhere in
    the package: the end of a track edge next to an ``edge_spacing`` gap, or
    a position in an open-field hole, snaps to the nearest interior bin.
    Binned by the raw bin edges instead, those steps went to bins the
    decoder drops, so the interior rows leaked mass. The bins below are
    worked out by hand.
    """
    if layout == "track_graph":
        env = _l_track_environment()
        # Linear positions: the centers of bins 0-9, the end of the first
        # edge (50, the gap bin's left edge; nearest interior center 47.5,
        # bin 9), the centers of bins 11-20 and the end of the track (115,
        # the last bin). Up and back down, twice.
        up = np.concatenate(
            [np.arange(2.5, 50, 5), [50.0], np.arange(67.5, 115, 5), [115.0]]
        )
        up_bins = np.concatenate([np.arange(10), [9], np.arange(11, 21), [20]])
        position = np.concatenate([up, up[::-1][1:], up[1:], up[::-1][1:]])
        bins = np.concatenate(
            [up_bins, up_bins[::-1][1:], up_bins[1:], up_bins[::-1][1:]]
        )
        position = position[:, None]
    else:
        env = _open_field_with_a_hole()
        # Flat bin index 5 * x_bin + y_bin, with x_bin = x // 10 + 1. (14, 18)
        # and (16, 15) are in the hole; their nearest interior centers are
        # (15, 25), bin 13, and (25, 15), bin 17.
        position = np.array(
            [
                [5, 5],
                [15, 5],
                [14, 18],
                [15, 25],
                [25, 25],
                [25, 15],
                [16, 15],
                [5, 15],
                [5, 5],
            ],
            dtype=float,
        )
        bins = np.array([6, 11, 13, 13, 18, 17, 17, 7, 6])
    interior = env.is_track_interior_flat_
    assert not interior.all()

    transition = EmpiricalMovement().make_state_transition(
        (env,), TransitionContext(position=position)
    )

    n_bins = interior.size
    expected = np.zeros((n_bins, n_bins))
    np.add.at(expected, (bins[:-1], bins[1:]), 1)
    expected /= np.maximum(expected.sum(axis=1, keepdims=True), 1)
    np.testing.assert_allclose(transition, expected, rtol=1e-12)
    assert not transition[~interior].any()
    assert not transition[:, ~interior].any()


@pytest.mark.parametrize("is_time_reversed", [False, True])
def test_empirical_movement_time_reversal_swaps_each_step(is_time_reversed):
    """Forward, a sweep up gives steps to the next bin; reversed, to the previous.

    An excluded stretch splits the sweep in two, so the reversed steps also
    have to be formed after leaving out the excluded samples.
    """
    up = np.arange(5.0, 100.0, 10.0)
    position = np.concatenate([up, np.full(20, 50.0), up])[:, None]
    is_training = np.ones(len(position), dtype=bool)
    is_training[10:30] = False
    env = Environment(
        place_bin_size=10.0, position_range=[(0.0, 100.0)]
    ).fit_place_grid(up[:, None], infer_track_interior=False)

    transition = EmpiricalMovement(
        is_time_reversed=is_time_reversed
    ).make_state_transition(
        (env,), TransitionContext(position, is_training=is_training)
    )

    expected = np.eye(10, k=-1 if is_time_reversed else 1)
    np.testing.assert_array_equal(transition, expected)


def test_context_construction_routes_give_identical_empirical_movement():
    """Unset labels mean the same as explicit all-training, group-0 labels."""
    env = Environment(place_bin_size=1.0).fit_place_grid(
        np.linspace(0.0, 10.0, 11)[:, None], infer_track_interior=False
    )
    position = np.linspace(0.0, 10.0, 51)[:, None]
    transition = EmpiricalMovement()
    implicit = transition.make_state_transition(
        (env,), TransitionContext(position=position)
    )
    explicit = transition.make_state_transition(
        (env,),
        TransitionContext(
            position=position,
            is_training=np.ones((51, 1), dtype=bool),
            encoding_group_labels=np.zeros(51, dtype=np.int32),
        ),
    )
    np.testing.assert_array_equal(implicit, explicit)
    assert implicit.sum() > 0


def test_discrete_transition_is_ones():
    tm = Discrete().make_state_transition()
    assert tm.shape == (1, 1)
    assert np.isclose(tm[0, 0], 1.0)


def test_random_walk_inward_no_trackgraphDD_raises(simple_1d_environment):
    """Direction-aware RandomWalk needs an N-D track graph and array distances.

    ``_handle_no_track_graph`` previously dereferenced
    ``self.environment.track_graphDD`` and ``distance_between_nodes_`` without
    checking they were available, raising a confusing ``AttributeError`` /
    ``TypeError``. The precondition is now surfaced as ``ConfigurationError``.
    """
    env = simple_1d_environment
    # Simulate an environment whose direction-aware path has nothing to
    # operate on: no N-D track graph and no array-valued distance lookup.
    env.track_graph = None
    set_place_grid(env, track_graphDD=None, distance_between_nodes=None)

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=3.0,
        direction="inward",
        use_manifold_distance=True,
    )
    # Match a phrase unique to the deep ``_handle_no_track_graph`` guard (not the
    # up-front ``make_state_transition`` guard, which also contains
    # "direction-aware RandomWalk"), so the test pins which guard fired.
    with pytest.raises(ConfigurationError, match="requires an N-D track graph"):
        rw.make_state_transition((env,))


def test_random_walk_direction_without_manifold_distance_raises(simple_1d_environment):
    """``direction`` is honored only with ``use_manifold_distance=True`` on an
    N-D grid. With the default ``use_manifold_distance=False`` the directional
    constraint would be silently dropped (a plain random walk), so it must raise
    instead of quietly ignoring the request.
    """
    env = simple_1d_environment
    env.track_graph = None

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=3.0,
        direction="inward",
        use_manifold_distance=False,  # default -> direction would be ignored
    )
    with pytest.raises(ConfigurationError, match="use_manifold_distance"):
        rw.make_state_transition((env,))


_INVALID_MOVEMENT_VARS = [
    -1.0,
    0.0,
    float("nan"),
    float("inf"),
    np.array([1.0, 0.0]),
    np.array([[1.0, 2.0], [2.0, 1.0]]),  # symmetric, not positive definite
    np.ones((2, 2, 2)),
    "large",
]
_INVALID_MOVEMENT_VAR_IDS = [
    "negative",
    "zero",
    "nan",
    "inf",
    "zero-variance-dim",
    "indefinite",
    "3-d",
    "string",
]


@pytest.mark.parametrize(
    "transition_class", [RandomWalk, RandomWalkDirection1, RandomWalkDirection2]
)
@pytest.mark.parametrize(
    "movement_var", _INVALID_MOVEMENT_VARS, ids=_INVALID_MOVEMENT_VAR_IDS
)
def test_invalid_movement_var_raises_at_construction(transition_class, movement_var):
    """A variance the Gaussian step cannot use is rejected when the transition
    is made; it constructed and failed at fit with a SciPy error."""
    with pytest.raises(ValidationError, match="movement_var"):
        transition_class(movement_var=movement_var)


@pytest.mark.parametrize(
    "movement_var",
    [
        6.0,
        1e-310,
        3,
        np.array(6.0),
        np.array([0.9, 1.7]),
        np.diag([0.9, 1.7]),
        np.array([[1.0, 0.2], [0.2, 1.0]]),
    ],
    ids=["float", "tiny", "int", "0-d", "diagonal", "diagonal-matrix", "full"],
)
def test_valid_movement_var_constructs(movement_var):
    """Scalars, per-dimension variances and positive-definite covariances,
    as stored by saved models and Spyglass rows, still construct."""
    assert RandomWalk(movement_var=movement_var).movement_var is movement_var


@pytest.mark.parametrize(
    "movement_mean",
    [float("nan"), float("inf"), np.array([0.0, np.nan]), "east"],
    ids=["nan", "inf", "nan-dim", "string"],
)
def test_invalid_movement_mean_raises_at_construction(movement_mean):
    with pytest.raises(ValidationError, match="movement_mean"):
        RandomWalk(movement_mean=movement_mean)


@pytest.mark.parametrize(
    "speedup",
    [0, -1, 1.0, 2.5, True, "2", None],
    ids=["zero", "negative", "float-one", "fraction", "bool", "string", "none"],
)
def test_invalid_speedup_raises_at_construction(speedup):
    """``speedup`` is the matrix power of the one-step transition: 0 gave the
    identity (no movement), a negative value the inverse."""
    with pytest.raises(ValidationError, match="speedup"):
        EmpiricalMovement(speedup=speedup)


@pytest.mark.parametrize("speedup", [1, 3, np.int64(2)], ids=["one", "three", "int64"])
def test_valid_speedup_constructs(speedup):
    assert EmpiricalMovement(speedup=speedup).speedup == speedup


def test_empirical_movement_without_a_step_is_refused():
    """Steps need two consecutive used samples: an all-zero block would give
    all-zero posteriors and a -inf marginal likelihood."""
    environment = Environment(place_bin_size=1.0).fit_place_grid(
        np.linspace(0.0, 10.0, 11)[:, None], infer_track_interior=False
    )
    position = np.linspace(0.0, 10.0, 21)[:, np.newaxis]
    context = TransitionContext(
        position=position, is_training=np.arange(len(position)) % 2 == 0
    )

    with pytest.raises(ValidationError, match="no two consecutive"):
        EmpiricalMovement().make_state_transition((environment,), context)


def test_normalize_row_probability_leaves_zero_rows_zero():
    """A zero row stays zero (no NaN); the others sum to 1."""
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])

    result = _normalize_row_probability(x)

    np.testing.assert_array_equal(result[1], 0.0)
    np.testing.assert_allclose(result[[0, 2]].sum(axis=1), 1.0)


def test_estimate_movement_var_1d_position():
    """Should estimate variance for 1D position."""
    # Arrange - linear motion with some noise
    rng = np.random.default_rng(42)
    position = np.linspace(0, 10, 100) + rng.standard_normal(100) * 0.1

    # Act
    var = estimate_movement_var(position)

    # Assert: the sample variance of the one-step displacements
    np.testing.assert_allclose(var, np.var(np.diff(position), ddof=1))


def test_estimate_movement_var_2d_position():
    """Should estimate covariance for 2D position."""
    # Arrange
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 100)
    position = np.column_stack(
        [t + rng.standard_normal(100) * 0.1, t + rng.standard_normal(100) * 0.1]
    )

    # Act
    var = estimate_movement_var(position)

    # Assert: the sample covariance of the one-step displacements
    assert var.shape == (2, 2)
    np.testing.assert_allclose(var, np.cov(np.diff(position, axis=0).T))


def test_estimate_movement_var_with_nans():
    """NaN samples are excluded: constant-speed motion with NaN at both
    ends has zero displacement variance, not NaN."""
    # Arrange
    position = np.linspace(0, 10, 100)
    position[:5] = np.nan
    position[-5:] = np.nan

    # Act
    var = estimate_movement_var(position)

    # Assert
    np.testing.assert_allclose(var, 0.0, atol=1e-12)


def test_estimate_movement_var_skips_steps_across_missing_samples():
    """No displacement is taken across a NaN stretch.

    Constant-speed motion with a tracking dropout, during which the
    animal moved 50 units: the steps on either side all have the same
    size, so the variance is zero; differencing across the dropout made
    the 50-unit jump one step.
    """
    # Arrange
    position = np.linspace(0, 10, 100)
    position[50:] += 50.0
    position[40:50] = np.nan

    # Act
    var = estimate_movement_var(position)

    # Assert
    np.testing.assert_allclose(var, 0.0, atol=1e-12)


def test_estimate_movement_var_constant_position():
    """Constant position should give near-zero variance."""
    # Arrange
    position = np.ones(100) * 5.0

    # Act
    var = estimate_movement_var(position)

    # Assert
    assert np.allclose(var, 0.0, atol=1e-10)


def test_estimate_movement_var_recovers_known_variance():
    """Should recover the true variance of a random walk."""
    # Arrange: generate random walk with known step variance
    rng = np.random.default_rng(123)
    true_var = 2.5
    steps = rng.standard_normal(10000) * np.sqrt(true_var)
    position = np.cumsum(steps)

    # Act
    estimated_var = estimate_movement_var(position)

    # Assert: should recover true variance within 10%
    assert abs(estimated_var - true_var) / true_var < 0.1, (
        f"Estimated variance {estimated_var:.4f} differs from true "
        f"variance {true_var:.4f} by more than 10%"
    )


@pytest.mark.parametrize(
    "transition",
    [
        RandomWalk(environment_name="test", movement_var=1.0),
        RandomWalkDirection1(environment_name="test", movement_var=1.0),
        RandomWalkDirection2(environment_name="test", movement_var=1.0),
        Uniform(environment_name="test"),
        Identity(environment_name="test"),
    ],
    ids=lambda transition: type(transition).__name__,
)
def test_transition_on_an_unfitted_environment_raises_not_fitted(transition):
    environments = (Environment(environment_name="test"),)

    with pytest.raises(NotFittedError, match="has not been fitted"):
        transition.make_state_transition(environments)


def test_random_walk_crosses_junction_by_track_distance():
    """Two perpendicular 10 cm edges are linearized with a 10 cm gap
    between them. Across the junction, adjacent bins are 2 cm apart along
    the track (12 cm apart linearized), so their transition probability
    equals that of adjacent bins within an edge; the gap bin gets none."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_node(2, pos=(10.0, 10.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    graph.add_edge(1, 2, distance=10.0, edge_id=1)
    env = Environment(
        environment_name="track",
        place_bin_size=2.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=10.0,
    ).fit_place_grid()
    # Bins 0-4 on the first edge, gap bin 5, bins 6-10 on the second edge.
    np.testing.assert_allclose(
        env.place_bin_centers_.ravel(),
        [1.0, 3.0, 5.0, 7.0, 9.0, 15.0, 21.0, 23.0, 25.0, 27.0, 29.0],
    )
    is_interior = env.is_track_interior_.ravel()
    np.testing.assert_array_equal(np.flatnonzero(~is_interior), [5])

    trans = RandomWalk(
        environment_name="track", movement_var=4.0
    ).make_state_transition((env,))

    assert trans.shape == (11, 11)
    np.testing.assert_allclose(trans[is_interior].sum(axis=1), 1.0)
    np.testing.assert_array_equal(trans[5], 0.0)
    np.testing.assert_array_equal(trans[:, 5], 0.0)
    # Last bin of edge 0 -> first bin of edge 1 vs. a within-edge step.
    np.testing.assert_allclose(trans[4, 6], trans[4, 3], rtol=1e-6)
    assert trans[4, 6] > 0.1
