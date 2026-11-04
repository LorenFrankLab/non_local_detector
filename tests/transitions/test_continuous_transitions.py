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
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ConfigurationError, ValidationError
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


@pytest.mark.parametrize("encoding_group", [0, 1])
def test_context_construction_routes_give_identical_empirical_movement(
    encoding_group,
):
    """Unset labels mean the same as explicit all-training, group-0 labels."""
    env = Environment(place_bin_size=1.0).fit_place_grid(
        np.linspace(0.0, 10.0, 11)[:, None], infer_track_interior=False
    )
    position = np.linspace(0.0, 10.0, 51)[:, None]
    transition = EmpiricalMovement(encoding_group=encoding_group)
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
    np.testing.assert_array_equal(implicit.sum(axis=1) > 0, encoding_group == 0)


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
